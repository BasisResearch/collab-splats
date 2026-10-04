"""
Depth + RGB views fused into a mesh.

- create_tsdf_mesh: the entry point, Open3D VoxelBlockGrid (CUDA when available) then extract
- writes out_dir/mesh.ply, the one name every reader probes
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
from tqdm.auto import tqdm

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh.utils import validate_views

logger = logging.getLogger(__name__)


########################################################################
# Entry point
########################################################################


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
    Integrate depth + RGB views into a VoxelBlockGrid and write the extracted mesh.

    - runs on CUDA:0 when Open3D has CUDA, else on the CPU; same grid, same mesh
    - the block hashmap starts at 10k blocks and grows on its own

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
    rgbs, c2w, K, depths = validate_views(rgbs, c2w, K, depths)

    if sdf_trunc is None:
        sdf_trunc = 4 * voxel_size

    # A band narrower than a voxel punctures the surface between voxels
    if sdf_trunc < voxel_size:
        raise ValueError(f"create_tsdf_mesh: sdf_trunc {sdf_trunc} is narrower than voxel_size {voxel_size}")

    # CUDA:0 when Open3D was built with it, else the CPU
    on_cuda = o3c.cuda.is_available()
    device = o3c.Device("CUDA:0" if on_cuda else "CPU:0")

    # Sparse voxel grid of 16³ blocks: TSDF, weight and float color per voxel
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=(1, 1, 3),
        voxel_size=float(voxel_size),
        block_resolution=16,
        block_count=10_000,
        device=device,
    )

    # Convert once: Open3D wants contiguous float32 depth, float64 world-to-camera poses, float scalars
    n = len(depths)
    trunc_mult = float(sdf_trunc / voxel_size)
    depth_trunc = float(depth_trunc)
    depths = np.ascontiguousarray(depths, dtype=np.float32)
    rgbs = np.ascontiguousarray(rgbs)
    K = K.astype(np.float64)
    w2c = invert_poses(c2w)
    w2c = w2c.astype(np.float64)

    # Integrate every view; depth is already in world units (scale 1.0), color goes in as float in [0, 1]
    for i in tqdm(range(n), desc=f"TSDF integration (voxel {voxel_size:g}, {device})"):
        depth = o3c.Tensor(depths[i], device=device)
        depth = o3d.t.geometry.Image(depth)
        color = o3c.Tensor(rgbs[i], device=device)
        color = color.to(o3c.float32) / 255.0
        color = o3d.t.geometry.Image(color)
        intrinsic = o3c.Tensor(K[i])
        extrinsic = o3c.Tensor(w2c[i])
        blocks = grid.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1.0, depth_trunc, trunc_mult)
        grid.integrate(blocks, depth, color, intrinsic, intrinsic, extrinsic, 1.0, depth_trunc, trunc_mult)

    # Extract; averaged colors can overshoot 1.0 by float error, so clip before the PLY writer clamps noisily
    tmesh = grid.extract_triangle_mesh(weight_threshold=0.0)
    tmesh.vertex.colors = tmesh.vertex.colors.clip(0.0, 1.0)
    mesh = tmesh.to_legacy()

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
