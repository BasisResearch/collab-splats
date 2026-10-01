"""
Depth + RGB views fused into a mesh.

- create_tsdf_mesh: the entry point, Open3D ScalableTSDFVolume then extract
- writes out_dir/mesh.ply, the one name every reader probes
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm.auto import tqdm

from collab_splats.geometry.transforms import extract_intrinsics, invert_poses

logger = logging.getLogger(__name__)


########################
# Entry point
########################


def create_tsdf_mesh(
    depths: np.ndarray,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    out_dir: Path | str,
    *,
    voxel_size: float,
    depth_trunc: float,
    sdf_trunc: float | None = None,
) -> Path:
    """
    Integrate depth + RGB views into a ScalableTSDFVolume and write the extracted mesh.

    Args:
        depths: (N, H, W) depth in world units, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 RGB at the same resolution as depths.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at the depth resolution.
        out_dir: directory to create; the mesh is written to out_dir/mesh.ply.
        voxel_size: TSDF voxel edge in world units.
        depth_trunc: depth beyond this (world units) is ignored.
        sdf_trunc: truncation band in world units; None = 4 × voxel_size.

    Returns:
        Path to out_dir/mesh.ply.

    Raises:
        ValueError: sdf_trunc is narrower than voxel_size.
    """
    rgbs, c2w, K, depths = _validate_views(rgbs, c2w, K, depths)

    if sdf_trunc is None:
        sdf_trunc = 4 * voxel_size

    # A band narrower than a voxel punctures the surface between voxels
    if sdf_trunc < voxel_size:
        raise ValueError(f"create_tsdf_mesh: sdf_trunc {sdf_trunc} is narrower than voxel_size {voxel_size}")

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
        fx, fy, cx, cy = extract_intrinsics(K[i])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        volume.integrate(rgbd, intrinsic, np.asarray(w2c[i], dtype=np.float64))

    mesh = volume.extract_triangle_mesh()

    # Write out_dir/mesh.ply
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    logger.info(
        "create_tsdf_mesh: %d views, voxel=%.4f sdf_trunc=%.4f -> %s (%d vertices)",
        n,
        voxel_size,
        sdf_trunc,
        mesh_path,
        len(mesh.vertices),
    )
    return mesh_path


########################
# Helpers
########################


def _validate_views(
    rgbs: np.ndarray, c2w: np.ndarray, K: np.ndarray, depths: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Coerce the view arrays and reject dtype, count and resolution mismatches.

    - shared by create_tsdf_mesh and create_texture_mesh
    - depths, when given, must match the rgbs resolution
    """
    rgbs = np.asarray(rgbs)
    c2w = np.asarray(c2w)
    K = np.asarray(K)

    # Input contract: uint8 color, one view count, K at the image resolution
    if rgbs.dtype != np.uint8:
        raise ValueError(f"rgbs must be uint8 in [0, 255], got {rgbs.dtype}")

    n, h, w = rgbs.shape[:3]

    if rgbs.shape != (n, h, w, 3) or c2w.shape != (n, 4, 4) or K.shape != (n, 3, 3):
        raise ValueError(f"views disagree: rgbs {rgbs.shape}, c2w {c2w.shape}, K {K.shape}")

    if depths is not None:
        depths = np.asarray(depths)

        if depths.shape != (n, h, w):
            raise ValueError(f"views disagree: depths {depths.shape}, rgbs {rgbs.shape}")

    cx, cy = K[:, 0, 2], K[:, 1, 2]

    if cx.min() < 0 or cx.max() > w or cy.min() < 0 or cy.max() > h:
        raise ValueError(
            f"Principal point outside the {w}x{h} image grid (cx range [{cx.min():.1f}, {cx.max():.1f}], "
            f"cy range [{cy.min():.1f}, {cy.max():.1f}]) — intrinsics and images are at different resolutions."
        )

    return rgbs, c2w, K, depths
