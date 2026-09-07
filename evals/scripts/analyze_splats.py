"""Sanity-check splat renders (normals) and compare TSDF meshes fused from feedforward vs splat depth.

Reports:
  Table 1: per primitive, over pixels with alpha > 0.5 in the renders of
           <results>/<prim>/ckpt.pt — unit-norm fraction of the rendered normal, and the angle
           between it and a finite-difference normal of the rendered depth (camera frame,
           checkpoint K).
  Table 2: TSDF meshes (same mesher settings) from pointcloud.zarr and from each primitive's
           rendered depth — vertices, triangles, connected components, largest-component
           triangle fraction, wall seconds. PLYs land at <results>/mesh_<source>.ply.

Usage (tmux, never a notebook):
  /opt/venv/reconstruction/bin/python evals/scripts/analyze_splats.py \
      --results evals/results/splats/tutorial --zarr data/outputs/pointcloud.zarr \
      --primitives 3dgs 2dgs --conf-percentile 20
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh import clean_repair_mesh, fuse_tsdf
from collab_splats.mesh.io import render_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import confidence_mask
from collab_splats.splats.rendering import load_checkpoint, render_views

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mirrors Reconstructor._run_tsdf_mesh; pinned so both sources fuse identically.
# sdf_trunc is derived (4 x voxel_size) and clean_repair always runs — neither is a knob.
MESH_KWARGS = {"voxel_size": 0.0025, "depth_trunc": 1.5}
ALPHA_MIN = 0.5
NORM_RANGE = (0.9, 1.1)


########################################
# Normals
########################################


def depth_to_normal(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Finite-difference surface normals of a z-depth map in the camera frame, facing the camera.

    - Back-projects the pixel grid with K, takes central differences along rows (dx) and
      columns (dy), normal = unit(cross(dx, dy)) — same stencil as gsplat.utils.depth_to_normal.
    - Flips any normal with a positive dot against its viewing ray so all face the camera.
    - Border pixels and pixels with a zero-length cross product are zero.
    """
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Back-project every pixel to a camera-frame point (z-depth convention)
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    z = depth.astype(np.float64)
    points = np.stack([(cols - cx) / fx * z, (rows - cy) / fy * z, z], axis=-1)

    # Central differences on the interior, cross product, unit-normalise
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    cross = np.cross(dx, dy)
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    unit = np.where(norm > 0, cross / np.maximum(norm, 1e-12), 0.0)

    # Face the camera: a normal with positive dot against the view ray points away
    faces_away = (unit * points[1:-1, 1:-1]).sum(-1, keepdims=True) > 0
    unit = np.where(faces_away, -unit, unit)

    normals = np.zeros((height, width, 3), dtype=np.float64)
    normals[1:-1, 1:-1] = unit
    return normals


def _angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between rows of two unit-vector arrays."""
    dots = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def analyze_normals(ckpt_path: Path, primitive: str) -> dict:
    """
    Table 1 statistics for one primitive's checkpoint, rendered and scored one view at a time.

    - Unit-norm fraction of the rendered normal (and after alpha division, meaningful for 2DGS).
    - Mean/median angle vs depth normals, raw and sign-folded (min(angle, 180 - angle)).
    """
    model, camera_opt, cam_to_world, intrinsics_all, image_ids, (height, width) = load_checkpoint(ckpt_path, "cuda")
    n_views = len(image_ids)

    n_pixels = 0
    n_unit = 0
    n_unit_alpha = 0
    angle_sum = 0.0
    folded_sum = 0.0
    angle_chunks = []
    folded_chunks = []

    # Renders stream one view at a time: the whole stack at 1080p is tens of GB
    renders = render_views(model, camera_opt, cam_to_world, intrinsics_all, height, width)
    for view, render in enumerate(renders):
        depth = render["depth"][0, ..., 0].cpu().numpy().astype(np.float64)
        alpha = render["alpha"][0, ..., 0].cpu().numpy().astype(np.float64)
        normal = render["normal"][0].cpu().numpy().astype(np.float64)
        intrinsics = intrinsics_all[view].cpu().numpy().astype(np.float64)

        # Opaque pixels only; depth normals are undefined on the one-pixel border
        mask = alpha > ALPHA_MIN
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False

        # Norm sanity, raw and alpha-divided
        norms = np.linalg.norm(normal, axis=-1)
        alpha_safe = np.maximum(alpha, 1e-6)
        norms_alpha = norms / alpha_safe
        in_range = (norms >= NORM_RANGE[0]) & (norms <= NORM_RANGE[1])
        in_range_alpha = (norms_alpha >= NORM_RANGE[0]) & (norms_alpha <= NORM_RANGE[1])
        n_pixels += int(mask.sum())
        n_unit += int((in_range & mask).sum())
        n_unit_alpha += int((in_range_alpha & mask).sum())

        # Angle vs depth normal where both are defined
        depth_normals = depth_to_normal(depth, intrinsics)
        depth_defined = np.linalg.norm(depth_normals, axis=-1) > 0
        angle_mask = mask & depth_defined & (norms > 0)
        unit_normal = normal[angle_mask] / norms[angle_mask][:, None]
        angles = _angles_deg(unit_normal, depth_normals[angle_mask])
        folded = np.minimum(angles, 180.0 - angles)
        angle_sum += float(angles.sum())
        folded_sum += float(folded.sum())
        angle_chunks.append(angles.astype(np.float32))
        folded_chunks.append(folded.astype(np.float32))

    all_angles = np.concatenate(angle_chunks)
    all_folded = np.concatenate(folded_chunks)
    n_angles = all_angles.size
    logger.info("%s: %d opaque pixels over %d views, %d with both normals", primitive, n_pixels, n_views, n_angles)

    return {
        "primitive": primitive,
        "n_views": int(n_views),
        "n_pixels": n_pixels,
        "unit_norm_fraction": n_unit / max(n_pixels, 1),
        "unit_norm_fraction_alpha_divided": n_unit_alpha / max(n_pixels, 1),
        "angle_mean_deg": angle_sum / max(n_angles, 1),
        "angle_median_deg": float(np.median(all_angles)) if n_angles else float("nan"),
        "angle_folded_mean_deg": folded_sum / max(n_angles, 1),
        "angle_folded_median_deg": float(np.median(all_folded)) if n_angles else float("nan"),
    }


########################################
# Meshes
########################################


def mesh_stats(mesh_path: Path) -> dict:
    """Vertex/triangle counts, connected components, and largest-component triangle fraction."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    n_triangles = len(mesh.triangles)
    _cluster_ids, cluster_sizes, _areas = mesh.cluster_connected_triangles()
    sizes = np.asarray(cluster_sizes)
    largest = int(sizes.max()) if sizes.size else 0
    return {
        "vertices": len(mesh.vertices),
        "triangles": n_triangles,
        "components": int(sizes.size),
        "largest_component_fraction": largest / max(n_triangles, 1),
    }


def build_mesh(name: str, inputs: tuple, results_dir: Path) -> dict:
    """
    Fuse and clean one (depths, rgbs, c2w, K) tuple, then flatten the PLY to mesh_<name>.ply.

    Args:
        name: source label — both the row's "source" field and the PLY's filename suffix.
        inputs: (depths, rgbs, c2w, K) exactly as fuse_tsdf takes them.
        results_dir: directory that receives mesh_<name>.ply.
    Returns:
        Stats dict: vertices, triangles, components, largest_component_fraction, source,
        seconds, path.
    """
    depths, rgbs, c2w, intrinsics = inputs
    work_dir = results_dir / f"_mesh_{name}"

    # Time fusion AND cleaning — clean_repair is not optional in the pipeline, so a
    # fuse-only number would understate what the mesh actually costs
    start = time.perf_counter()
    mesh_path = fuse_tsdf(depths, rgbs, c2w, intrinsics, work_dir, **MESH_KWARGS)
    clean_repair_mesh(mesh_path)
    seconds = time.perf_counter() - start

    # Move the PLY to its flat name; the work dir is only ever the mesher's scratch
    out_path = results_dir / f"mesh_{name}.ply"
    shutil.move(str(mesh_path), str(out_path))
    shutil.rmtree(work_dir, ignore_errors=True)

    stats = mesh_stats(out_path)
    stats["source"] = name
    stats["seconds"] = seconds
    stats["path"] = str(out_path)
    logger.info("%s mesh: %s", name, stats)
    return stats


########################################
# Tables
########################################


def normals_table(rows: list[dict]) -> str:
    """Markdown table 1."""
    lines = [
        "Table 1 — normals sanity (stored `normal` is camera-frame; alpha > 0.5 pixels; "
        "angle vs finite-difference depth normal, camera frame, stored K)",
        "",
        "| primitive | views | pixels | unit-norm frac | unit-norm frac (/alpha) | angle mean | angle median | folded mean | folded median |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = (
            f"| {row['primitive']} | {row['n_views']} | {row['n_pixels']} "
            f"| {row['unit_norm_fraction']:.4f} | {row['unit_norm_fraction_alpha_divided']:.4f} "
            f"| {row['angle_mean_deg']:.2f} | {row['angle_median_deg']:.2f} "
            f"| {row['angle_folded_mean_deg']:.2f} | {row['angle_folded_median_deg']:.2f} |"
        )
        lines.append(cells)
    return "\n".join(lines)


def mesh_table(rows: list[dict]) -> str:
    """Markdown table 2."""
    settings = ", ".join(f"{key}={value}" for key, value in MESH_KWARGS.items())
    lines = [
        f"Table 2 — TSDF mesh comparison ({settings})",
        "",
        "| source | vertices | triangles | components | largest comp. frac | seconds |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = (
            f"| {row['source']} | {row['vertices']} | {row['triangles']} | {row['components']} "
            f"| {row['largest_component_fraction']:.4f} | {row['seconds']:.1f} |"
        )
        lines.append(cells)
    return "\n".join(lines)


########################################
# Entry point
########################################


def main() -> None:
    """Parse arguments, run both analyses, write analysis.json, print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("evals/results/splats/tutorial"))
    parser.add_argument("--zarr", type=Path, default=Path("data/outputs/pointcloud.zarr"))
    parser.add_argument("--primitives", nargs="+", default=["3dgs", "2dgs"])
    parser.add_argument("--conf-percentile", type=float, default=20.0)
    args = parser.parse_args()

    # Primitives whose checkpoints exist; missing ones are skipped, not fatal
    available = []
    for primitive in args.primitives:
        ckpt_path = args.results / primitive / "ckpt.pt"
        if ckpt_path.exists():
            available.append((primitive, ckpt_path))
        else:
            logger.warning("Skipping %s: %s missing", primitive, ckpt_path)

    # Table 1
    normal_rows = [analyze_normals(ckpt_path, primitive) for primitive, ckpt_path in available]

    # Table 2: feedforward first, then each primitive's renders.
    # Model-resolution fusion — depth, images and K all come off the same forward pass, so
    # nothing is lifted here and no COLMAP camera is involved.
    ff = FeedforwardResult.load_zarr(args.zarr, load_images=True)
    depths = np.ascontiguousarray(ff.depth, dtype=np.float32)

    # Confidence gate before fusion; a reconstruction without confidence (sfm) fuses unmasked
    if args.conf_percentile is not None and ff.confidence is not None:
        keep = confidence_mask(np.asarray(ff.confidence), args.conf_percentile)
        depths = np.where(keep, depths, 0.0)

    # images is (N, 3, H, W) float in [0, 1]; fuse_tsdf takes (N, H, W, 3) uint8
    rgbs = np.asarray(ff.images).transpose(0, 2, 3, 1)
    rgbs = np.ascontiguousarray((np.clip(rgbs, 0.0, 1.0) * 255).round().astype(np.uint8))

    mesh_rows = [build_mesh("feedforward", (depths, rgbs, invert_poses(ff.extrinsics), ff.intrinsics), args.results)]

    # Splat renders leave the checkpoint at frame resolution carrying their own poses
    for primitive, ckpt_path in available:
        mesh_rows.append(build_mesh(primitive, render_tsdf_inputs(ckpt_path), args.results))

    # Persist and print
    analysis = {
        "zarr": str(args.zarr),
        "conf_percentile": args.conf_percentile,
        "mesh_kwargs": MESH_KWARGS,
        "normals": normal_rows,
        "meshes": mesh_rows,
    }
    out_path = args.results / "analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    logger.info("Wrote %s", out_path)
    print()
    print(normals_table(normal_rows))
    print()
    print(mesh_table(mesh_rows))


if __name__ == "__main__":
    main()
