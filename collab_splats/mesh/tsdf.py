"""
Depth + RGB views fused into a mesh.

  - fuse_tsdf: the entry point, Open3D ScalableTSDFVolume then extract
  - fuse_tsdf_bands: one fuse per depth band, trimmed to its band and merged
  - writes out_dir/mesh.ply, the one name every reader probes
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from collab_splats.geometry.transforms import extract_intrinsics, invert_poses

logger = logging.getLogger(__name__)

# How close a coarse band's vertex must be to a finer band's surface to count as the same
# surface, in units of the coarse band's own voxel
CULL_VOXELS = 1.5


def fuse_tsdf(depths, rgbs, c2w, K, out_dir, voxel_size, depth_trunc, sdf_trunc=None, depth_min=0.0):
    """
    Integrate depth + RGB views into a ScalableTSDFVolume and write the extracted mesh.

    Args:
        depths: (N, H, W) float depth in world units, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 RGB at the same resolution as depths.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at the depth resolution.
        out_dir: Path or str, directory to create; the mesh is written to out_dir/mesh.ply.
        voxel_size: float, TSDF voxel edge in world units.
        depth_trunc: float, depth beyond this (world units) is ignored.
        sdf_trunc: float or None, truncation band in world units; None = 4 × voxel_size.
        depth_min: float, depth nearer than this is dropped; the near cut that makes depth banding possible.
    Returns:
        Path to out_dir/mesh.ply.
    """
    depths, rgbs, c2w, K = _check_views(depths, rgbs, c2w, K)
    if sdf_trunc is None:
        sdf_trunc = 4 * voxel_size

    mesh = _integrate_views(depths, rgbs, c2w, K, voxel_size, depth_trunc, sdf_trunc, depth_min)

    mesh_path = _write_mesh(mesh, out_dir)
    logger.info(
        "fuse_tsdf: %d views, voxel=%.4f sdf_trunc=%.4f -> %s (%d vertices)",
        len(depths),
        voxel_size,
        sdf_trunc,
        mesh_path,
        len(mesh.vertices),
    )
    return mesh_path


def fuse_tsdf_bands(depths, rgbs, c2w, K, out_dir, bands, sdf_trunc_mult=4.0):
    """
    Fuse each depth band at its own voxel size and merge, keeping the finest copy of a surface.

    One TSDF volume forces one voxel size on the whole scene, so a voxel fine enough for
    near thin structure costs the far field its memory. Fusing the same views once per
    band sidesteps that: near bands get a fine voxel over a small slice of surface area,
    far bands keep a coarse one.

    Args:
        depths: (N, H, W) float depth in world units, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 RGB at the same resolution as depths.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at the depth resolution.
        out_dir: Path or str, directory to create; the merged mesh is written to out_dir/mesh.ply.
        bands: list of {depth_min, depth_trunc, voxel_size}, ascending and contiguous.
        sdf_trunc_mult: truncation band as a multiple of each band's own voxel_size.
    Note:
        Bands are cut on depth along the optical axis, the same quantity depth_trunc uses.
    Returns:
        Path to out_dir/mesh.ply.
    """
    depths, rgbs, c2w, K = _check_views(depths, rgbs, c2w, K)
    bands = check_bands(bands)

    # A surface seen at two different ranges lands in two bands, so the merge has to pick one
    # copy. Finest band first, then cull from each coarser one whatever a finer band already
    # covers: a vertex is only ever dropped because a better copy of it exists, so unlike a
    # geometric partition (nearest camera, say) this cannot open a hole in the merged mesh
    merged = o3d.geometry.TriangleMesh()
    for band in sorted(bands, key=lambda b: b["voxel_size"]):
        depth_min, depth_trunc, voxel_size = band["depth_min"], band["depth_trunc"], band["voxel_size"]
        mesh = _integrate_views(depths, rgbs, c2w, K, voxel_size, depth_trunc, sdf_trunc_mult * voxel_size, depth_min)

        # Cull against the finer bands already merged. Within one of this band's own voxels the
        # two surfaces are the same surface — this band cannot resolve them apart in any case
        n_before = len(mesh.vertices)
        if n_before and len(merged.vertices):
            tree = cKDTree(np.asarray(merged.vertices))
            distance, _ = tree.query(np.asarray(mesh.vertices), workers=-1)
            mesh.remove_vertices_by_mask(distance < CULL_VOXELS * voxel_size)
        logger.info(
            "fuse_tsdf_bands: band [%.3f, %.3f) voxel=%.4f -> %d vertices (%d before cull)",
            depth_min,
            depth_trunc,
            voxel_size,
            len(mesh.vertices),
            n_before,
        )
        merged += mesh

    mesh_path = _write_mesh(merged, out_dir)
    logger.info(
        "fuse_tsdf_bands: %d views, %d bands -> %s (%d vertices)",
        len(depths),
        len(bands),
        mesh_path,
        len(merged.vertices),
    )
    return mesh_path


########  Helpers  ########


def check_bands(bands):
    """
    Reject band lists that are not ascending, contiguous and positively sized.

    Gaps between bands lose the geometry that falls in them and overlaps double-surface it,
    so contiguity is a correctness requirement, not a style preference.
    """
    if not isinstance(bands, (list, tuple)) or not bands:
        raise ValueError(f"bands must be a non-empty list of depth bands, got {bands!r}")

    checked = []
    for i, band in enumerate(bands):
        if not isinstance(band, dict) or set(band) != {"depth_min", "depth_trunc", "voxel_size"}:
            raise ValueError(f"band {i} must have keys depth_min, depth_trunc, voxel_size, got {band!r}")
        values = {}
        for key in ("depth_min", "depth_trunc", "voxel_size"):
            value = band[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"band {i} {key} must be a number, got {value!r}")
            values[key] = float(value)

        # Each band spans a positive depth range at a positive voxel size, and picks up
        # exactly where the previous one stopped
        if values["depth_min"] < 0 or values["depth_trunc"] <= values["depth_min"]:
            raise ValueError(
                f"band {i} must satisfy 0 <= depth_min < depth_trunc, "
                f"got [{values['depth_min']}, {values['depth_trunc']})"
            )
        if values["voxel_size"] <= 0:
            raise ValueError(f"band {i} voxel_size must be > 0, got {values['voxel_size']}")
        if i and values["depth_min"] != checked[-1]["depth_trunc"]:
            raise ValueError(
                f"band {i} starts at {values['depth_min']} but band {i - 1} ends at "
                f"{checked[-1]['depth_trunc']} — bands must be contiguous"
            )
        checked.append(values)

    return checked


def _check_views(depths, rgbs, c2w, K):
    """
    Coerce the view arrays and reject dtype, count and resolution mismatches.
    """
    depths = np.asarray(depths)
    rgbs = np.asarray(rgbs)
    c2w = np.asarray(c2w)
    K = np.asarray(K)

    # Input contract: uint8 color, one view count, K at the depth resolution
    if rgbs.dtype != np.uint8:
        raise ValueError(f"rgbs must be uint8 in [0, 255], got {rgbs.dtype}")
    n, h, w = depths.shape
    if rgbs.shape != (n, h, w, 3) or c2w.shape != (n, 4, 4) or K.shape != (n, 3, 3):
        raise ValueError(f"views disagree: depths {depths.shape}, rgbs {rgbs.shape}, c2w {c2w.shape}, K {K.shape}")
    cx, cy = K[:, 0, 2], K[:, 1, 2]
    if cx.min() < 0 or cx.max() > w or cy.min() < 0 or cy.max() > h:
        raise ValueError(
            f"Principal point outside the {w}x{h} depth grid (cx range [{cx.min():.1f}, {cx.max():.1f}], "
            f"cy range [{cy.min():.1f}, {cy.max():.1f}]) — intrinsics and depth are at different resolutions."
        )

    return depths, rgbs, c2w, K


def _integrate_views(depths, rgbs, c2w, K, voxel_size, depth_trunc, sdf_trunc, depth_min):
    """
    Integrate every view into a ScalableTSDFVolume and return the extracted mesh.
    """
    # Near cut: Open3D's RGBDImage takes only a far bound, so anything nearer is zeroed —
    # 0 already means "no observation" to the integrator, the same as beyond depth_trunc
    if depth_min > 0:
        depths = np.where(depths >= depth_min, depths, 0.0)

    # Integrate every view; Open3D wants world-to-camera extrinsics
    n, h, w = depths.shape
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    w2c = invert_poses(c2w)
    for i in tqdm(range(n), desc=f"TSDF integration (voxel {voxel_size:g})"):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgbs[i])),
            o3d.geometry.Image(np.ascontiguousarray(depths[i], dtype=np.float32)),
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        fx, fy, cx_i, cy_i = extract_intrinsics(K[i])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx_i, cy_i)
        volume.integrate(rgbd, intrinsic, np.asarray(w2c[i], dtype=np.float64))

    return volume.extract_triangle_mesh()


def _write_mesh(mesh, out_dir):
    """
    Write the mesh to out_dir/mesh.ply, the one name every reader probes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    return mesh_path
