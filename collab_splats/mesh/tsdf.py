"""
Depth + RGB views fused into a mesh.

- create_tsdf_mesh: the entry point, Open3D VoxelBlockGrid (CUDA when available) then extract
- compute_tsdf_voxel_size: voxel edge from the depth's own pixel footprint, coarsened to a memory budget
- no file IO: the caller writes the mesh once it is final
"""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d
import open3d.core as o3c
from tqdm.auto import tqdm

from collab_splats.geometry.projection import unproject_frames
from collab_splats.geometry.transforms import invert_poses, shift_intrinsics
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
    *,
    voxel_size: float,
    depth_trunc: float | None = None,
    sdf_trunc: float | None = None,
    max_blocks: int = 30_000,
) -> o3d.geometry.TriangleMesh:
    """
    Integrate depth + RGB views into a VoxelBlockGrid and extract its mesh.

    - runs on CUDA:0 when Open3D has CUDA, else on the CPU; same grid, same mesh
    - the block hashmap starts at 10k blocks and grows on its own
    - Open3D floors projected u, v (pixel-corner), so K shifts +0.5 once here
    - scenes past max_blocks fuse in spatial tiles; the welded mesh equals the single grid's

    Args:
        depths: (N, H, W) depth in world units, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 RGB at the same resolution as depths.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) pixel-center intrinsics at the depth resolution.
        voxel_size: TSDF voxel edge in world units.
        depth_trunc: depth beyond this (world units) is ignored; None keeps every depth.
        sdf_trunc: truncation band in world units; None = 4 × voxel_size.
        max_blocks: blocks per grid, halo included; Open3D extraction faults past 32,768.

    Returns:
        The extracted mesh with per-vertex color.

    Raises:
        ValueError: sdf_trunc is narrower than voxel_size.
    """
    rgbs, c2w, K, depths = validate_views(rgbs, c2w, K, depths)
    assert depths is not None

    if sdf_trunc is None:
        sdf_trunc = 4 * voxel_size

    # A band narrower than a voxel punctures the surface between voxels
    if sdf_trunc < voxel_size:
        raise ValueError(
            f"create_tsdf_mesh: sdf_trunc {sdf_trunc} is narrower than voxel_size {voxel_size}"
        )

    # CUDA:0 when Open3D was built with it, else the CPU
    on_cuda = o3c.cuda.is_available()
    device = o3c.Device("CUDA:0" if on_cuda else "CPU:0")

    # Sparse voxel grid of 16³ blocks: TSDF, weight and float color per voxel
    grid_kwargs = dict(
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
    depth_trunc = (
        float(depths.max()) + voxel_size if depth_trunc is None else float(depth_trunc)
    )
    depths = np.ascontiguousarray(depths, dtype=np.float32)
    rgbs = np.ascontiguousarray(rgbs)
    K = shift_intrinsics(K, (0.5, 0.5))
    w2c = invert_poses(c2w)
    w2c = w2c.astype(np.float64)

    # Probe the blocks each view's truncation band touches; their union is the scene
    probe = o3d.t.geometry.VoxelBlockGrid(**grid_kwargs)
    view_blocks = []
    for i in range(n):
        depth = o3c.Tensor(depths[i], device=device)
        depth = o3d.t.geometry.Image(depth)
        blocks = probe.compute_unique_block_coordinates(
            depth, o3c.Tensor(K[i]), o3c.Tensor(w2c[i]), 1.0, depth_trunc, trunc_mult
        )
        view_blocks.append(blocks.cpu().numpy().astype(np.int64).reshape(-1, 3))
    del probe
    union = np.unique(np.concatenate(view_blocks), axis=0)

    # One int64 key per block coord, for row-wise set tests; coords stay within ±2^20
    key_weights = np.array([1 << 42, 1 << 21, 1], dtype=np.int64)
    union_keys = (union + (1 << 20)) @ key_weights
    neighbors = np.stack(np.meshgrid(*[(-1, 0, 1)] * 3, indexing="ij"), -1)
    neighbors = neighbors.reshape(-1, 3)

    # Split k-d style at the longest axis' median until core + 1-block halo fits max_blocks
    # - Open3D's extract_triangle_mesh indexes with int32 and faults past 32,768 blocks
    # - the halo holds every block a core triangle's marching cube reads
    # - a scene that fits is one tile: the whole union
    tiles, stack = [], [union]
    while stack:
        core = stack.pop()
        halo = np.unique((core[:, None] + neighbors).reshape(-1, 3), axis=0)
        halo = halo[np.isin((halo + (1 << 20)) @ key_weights, union_keys)]

        if len(halo) <= max_blocks or len(core) == 1:
            tiles.append((core, halo))
            continue

        axis = int(np.argmax(np.ptp(core, axis=0)))
        cut = np.median(core[:, axis])
        below = core[:, axis] < cut
        if not below.any():
            below = core[:, axis] <= cut
        stack += [core[below], core[~below]]

    # Fuse each tile in a fresh grid; views integrate only their blocks inside the tile's halo
    parts = []
    for t, (core, halo) in enumerate(tiles):
        grid = o3d.t.geometry.VoxelBlockGrid(**grid_kwargs)
        halo_keys = (halo + (1 << 20)) @ key_weights
        desc = f"TSDF integration (voxel {voxel_size:g}, {device}, tile {t + 1}/{len(tiles)})"

        for i in tqdm(range(n), desc=desc):
            blocks = view_blocks[i]
            if len(tiles) > 1:
                blocks = blocks[np.isin((blocks + (1 << 20)) @ key_weights, halo_keys)]
            if not len(blocks):
                continue

            # Depth is already in world units (scale 1.0); color goes in as float in [0, 1]
            depth = o3c.Tensor(depths[i], device=device)
            depth = o3d.t.geometry.Image(depth)
            color = o3c.Tensor(rgbs[i], device=device)
            color = color.to(o3c.float32) / 255.0
            color = o3d.t.geometry.Image(color)
            intrinsic = o3c.Tensor(K[i])
            extrinsic = o3c.Tensor(w2c[i])
            grid.integrate(
                o3c.Tensor(blocks.astype(np.int32), device=device),
                depth,
                color,
                intrinsic,
                intrinsic,
                extrinsic,
                1.0,
                depth_trunc,
                trunc_mult,
            )

        # Extract; averaged colors can overshoot 1.0 by float error, so clip before the PLY writer clamps noisily
        tmesh = grid.extract_triangle_mesh(weight_threshold=0.0)
        tmesh.vertex.colors = tmesh.vertex.colors.clip(0.0, 1.0)
        del grid

        # One tile is the whole scene: the mesh as extracted
        if len(tiles) == 1:
            mesh = tmesh.to_legacy()
            break

        # Keep the triangles whose centroid voxel lies in a core block; halo copies belong to a neighbor
        v = tmesh.vertex.positions.cpu().numpy()
        f = tmesh.triangle.indices.cpu().numpy().astype(np.int64)
        cell = np.floor(v[f].mean(axis=1) / voxel_size).astype(np.int64)
        owner = (np.floor_divide(cell, 16) + (1 << 20)) @ key_weights
        keep = np.isin(owner, (core + (1 << 20)) @ key_weights)
        parts.append(
            (
                v,
                f[keep],
                tmesh.vertex.normals.cpu().numpy(),
                tmesh.vertex.colors.cpu().numpy(),
            )
        )

    # Weld tiles at bit-equal positions; attributes come from a copy a kept triangle references
    # - halo-edge copies carry one-sided normals, so the first copy is not safe
    if len(tiles) > 1:
        offsets = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
        v = np.concatenate([p[0] for p in parts])
        f = np.concatenate([p[1] + o for p, o in zip(parts, offsets)])
        normals = np.concatenate([p[2] for p in parts])
        colors = np.concatenate([p[3] for p in parts])

        _, weld = np.unique(v, axis=0, return_inverse=True)
        weld = weld.reshape(-1)
        source = np.zeros(weld.max() + 1, dtype=np.int64)
        source[weld[f.ravel()]] = f.ravel()
        used = np.unique(weld[f])
        remap = np.full(weld.max() + 1, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v[source[used]].astype(np.float64)),
            o3d.utility.Vector3iVector(remap[weld[f]].astype(np.int32)),
        )
        mesh.vertex_normals = o3d.utility.Vector3dVector(
            normals[source[used]].astype(np.float64)
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            colors[source[used]].astype(np.float64)
        )

    logger.info(
        "create_tsdf_mesh: %d views, %d blocks in %d tiles, voxel=%.4f sdf_trunc=%.4f -> %d vertices",
        n,
        len(union),
        len(tiles),
        voxel_size,
        sdf_trunc,
        len(mesh.vertices),
    )
    return mesh


def compute_tsdf_voxel_size(
    depths: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    *,
    depth_fx: float,
    depth_px: float = 4.0,
    ref_percentile: float = 50.0,
    max_gb: float = 8.0,
    stride: int = 8,
) -> float:
    """
    TSDF voxel edge from the depth's own resolution, coarsened until the grid fits a memory budget.

    - one depth pixel spans depth / depth_fx world units; the voxel is depth_px of them at the ref_percentile depth
    - so the voxel follows the scene's scale and the depth resolution, with no world-unit constant
    - if the surface's 16³ voxel blocks would exceed max_gb, the voxel is coarsened until they fit
    - block count scales with 1 / voxel², so each step coarsens by the square root of the overshoot

    Args:
        depths: (N, H, W) depth in world units, 0 = no observation.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at the depths' resolution.
        depth_fx: focal length in pixels of the grid the depth was predicted on.
        depth_px: voxel edge in depth-pixel footprints.
        ref_percentile: depth percentile the footprint is taken at; lower favors near surfaces.
        max_gb: memory budget for the voxel blocks the surface occupies.
        stride: pixel subsample per axis for the depth statistics and the block count.

    Returns:
        Voxel edge in world units.

    Raises:
        ValueError: depths holds no positive depth.
    """
    sub = depths[:, ::stride, ::stride]
    valid = sub > 0

    if not valid.any():
        raise ValueError("compute_tsdf_voxel_size: no positive depth")

    # Voxel from the depth pixel footprint at the reference depth
    ref_depth = float(np.percentile(sub[valid], ref_percentile))
    voxel = depth_px * ref_depth / depth_fx

    # Lift the subsample once; K follows the subsampled pixel grid
    K_sub = K.astype(np.float64)
    K_sub[:, :2] /= stride
    w2c = invert_poses(c2w)
    points = unproject_frames(sub, w2c, K_sub)
    points = points[valid]

    # Coarsen until the occupied blocks fit; tsdf + weight + rgb, float32 each, per voxel
    block_bytes = 16**3 * 5 * 4

    while True:
        blocks = np.floor(points / (16 * voxel)).astype(np.int64)
        n_blocks = len(np.unique(blocks, axis=0))
        need_gb = n_blocks * block_bytes / 1e9

        if need_gb <= max_gb:
            break

        voxel *= float(np.sqrt(need_gb / max_gb)) * 1.01

    logger.info(
        "compute_tsdf_voxel_size: %.5f (%.2f px at p%g depth %.4f, fx %.1f) | %d surface blocks, %.2f GB",
        voxel,
        voxel * depth_fx / ref_depth,
        ref_percentile,
        ref_depth,
        depth_fx,
        n_blocks,
        need_gb,
    )
    return voxel
