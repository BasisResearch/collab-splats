"""
A/B one scene's mesh with and without mesh.mask_sky.

- runs the mesh stage twice against the SAME pointcloud.zarr, so the only difference is the flag
- reports component stats from analyze_splats.mesh_stats
- renders both meshes from ONE fixed camera, and writes a mask-overlay contact sheet
- compute only: run from a shell or tmux, never a notebook
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from collab_splats.preproc import frames
from collab_splats.semantics.segmentation import sky_masks
from collab_splats.wrapper.reconstructor import Reconstructor
from evals.scripts.analyze_splats import mesh_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


########################################
# Arms
########################################


def load_recon(config_path: Path, mask_sky: bool) -> Reconstructor:
    """
    A Reconstructor for one scene with mask_sky forced after the base.yaml merge.

    Args:
        config_path: scene config YAML; its pointcloud.zarr must already exist.
        mask_sky: the setting under test.

    Returns:
        A configured Reconstructor.
    """
    config = yaml.safe_load(config_path.read_text())
    recon = Reconstructor(config)
    recon.config["mesh"]["mask_sky"] = mask_sky
    return recon


def run_arm(config_path: Path, mask_sky: bool, results_dir: Path) -> dict:
    """
    Run the mesh stage once at a given mask_sky setting and collect its stats.

    Args:
        config_path: scene config; its pointcloud.zarr must already exist.
        mask_sky: the setting under test.
        results_dir: receives mesh_{on,off}.ply.

    Returns:
        Stats dict from mesh_stats plus 'mask_sky' and 'path'.
    """
    recon = load_recon(config_path, mask_sky)

    mesh_path = recon.mesh(overwrite=True)

    # Move the PLY aside before the other arm overwrites it in place
    arm = "on" if mask_sky else "off"
    out_path = results_dir / f"mesh_{arm}.ply"
    shutil.move(str(mesh_path), str(out_path))

    stats = mesh_stats(out_path)
    stats["mask_sky"] = mask_sky
    stats["path"] = str(out_path)
    logger.info("mask_sky=%s: %s", mask_sky, stats)
    return stats


########################################
# Renders
########################################


def render_meshes(arms: list[tuple[str, Path]], out_path: Path, size: int = 900) -> Path:
    """
    Render every arm's mesh from one shared camera and tile them side by side.

    - ONE OffscreenRenderer for the whole process: Open3D segfaults on a second one
    - the camera is framed on the FIRST arm and reused; a per-mesh camera would silently
      reframe when masking changes the bounds

    Args:
        arms: (label, PLY path) pairs; the first sets the camera, order sets the tiling.
        out_path: PNG to write.
        size: square render edge, pixels.

    Returns:
        Path to out_path.
    """
    renderer = o3d.visualization.rendering.OffscreenRenderer(size, size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    eye = center = up = None
    tiles = []
    for label, path in arms:
        mesh = o3d.io.read_triangle_mesh(str(path))
        mesh.compute_vertex_normals()

        # Frame on the first mesh only; a 3/4 view off the bounding box diagonal
        # - the offset is a fraction of the diagonal, so one constant frames any scene scale
        # - 0.35 not 0.6: at 60 degrees vertical fov the larger value leaves the mesh a quarter
        #   of the frame wide, which is too small to read floaters off
        if eye is None:
            box = mesh.get_axis_aligned_bounding_box()
            center = box.get_center()
            extent = np.linalg.norm(box.get_extent())
            eye = center + np.array([0.35, -0.5, 0.35]) * max(extent, 1e-6)
            up = np.array([0.0, 0.0, 1.0])

        renderer.scene.add_geometry(label, mesh, material)
        renderer.setup_camera(60.0, center, eye, up)
        tile = np.asarray(renderer.render_to_image()).copy()  # putText writes in place
        renderer.scene.remove_geometry(label)

        # Label in the corner, so a saved PNG is readable without its filename
        cv2.putText(tile, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(tile)

    cv2.imwrite(str(out_path), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
    logger.info("mesh renders: %d arms -> %s", len(tiles), out_path)
    return out_path


########################################
# Mask QA
########################################


def write_mask_contact_sheet(images_dir: Path, out_path: Path, stride: int = 10) -> Path:
    """
    Tile every stride-th frame with its sky mask overlaid, so false positives are visible.

    - masks come from sky_masks, so the sheet reads the same cache the mesh stage wrote
    - the composite is 75% red, not a tint: a subtle overlay is unreadable at tile scale

    Args:
        images_dir: the scene's keyframe directory.
        out_path: PNG to write.
        stride: take every stride-th frame.

    Returns:
        Path to out_path.
    """
    paths = frames.frame_paths(images_dir)[::stride]
    idxs = [frames.frame_idx_from_path(p) for p in paths]
    masks = sky_masks(images_dir, idxs=idxs)

    tiles = []
    for path, idx, mask in zip(paths, idxs, masks):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        composite = rgb.copy()
        composite[mask] = (0.25 * rgb[mask] + 0.75 * np.array([255, 0, 0], np.float32)).astype(np.uint8)
        tile = cv2.resize(composite, (320, 180))

        # frame_idx and sky fraction, so the sheet reads without its filenames
        cv2.putText(
            tile, f"{idx} {mask.mean():.3f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA
        )
        tiles.append(tile)

    # Four per row; pad the last row so vstack has uniform widths
    rows = [tiles[i : i + 4] for i in range(0, len(tiles), 4)]
    blank = np.zeros_like(tiles[0])
    rows = [row + [blank] * (4 - len(row)) for row in rows]
    sheet = np.vstack([np.hstack(row) for row in rows])

    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    logger.info("mask contact sheet: %d frames -> %s", len(tiles), out_path)
    return out_path


########################################
# Report
########################################


def comparison_table(off: dict, on: dict) -> str:
    """
    Markdown table of the two arms with a delta column.

    Args:
        off: stats from the mask_sky=false arm.
        on: stats from the mask_sky=true arm.

    Returns:
        The table as a string.
    """
    keys = ("vertices", "triangles", "components", "largest_component_fraction")
    lines = [
        "| metric | mask_sky off | mask_sky on | delta |",
        "|---|---|---|---|",
    ]
    for key in keys:
        a, b = off[key], on[key]
        delta = f"{b - a:+.4f}" if isinstance(a, float) else f"{b - a:+d}"
        fmt = "{:.4f}" if isinstance(a, float) else "{}"
        lines.append(f"| {key} | {fmt.format(a)} | {fmt.format(b)} | {delta} |")
    return "\n".join(lines)


def main() -> None:
    """
    Run both arms for one scene and write stats.json, a table and a contact sheet.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="scene config YAML")
    parser.add_argument("--results", type=Path, required=True, help="output directory")
    parser.add_argument("--stride", type=int, default=10, help="contact-sheet frame stride")
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)

    # Off first: it populates nothing, so an on-arm cache miss is genuinely the first run
    off = run_arm(args.config, mask_sky=False, results_dir=args.results)
    on = run_arm(args.config, mask_sky=True, results_dir=args.results)

    # Off first in the dict too: it sets the shared camera both tiles are framed with
    render_meshes(
        [("mask_sky off", Path(off["path"])), ("mask_sky on", Path(on["path"]))],
        args.results / "mesh_renders.png",
    )

    recon = load_recon(args.config, mask_sky=True)
    write_mask_contact_sheet(recon.images_dir, args.results / "sky_masks.png", stride=args.stride)

    (args.results / "stats.json").write_text(json.dumps({"off": off, "on": on}, indent=2))
    table = comparison_table(off, on)
    (args.results / "table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
