"""
Depth + RGB views fused into a mesh.

- create_tsdf_mesh: the entry point, per-pixel weighted fusion into an Open3D VoxelBlockGrid (CUDA when available) then extract
- compute_tsdf_voxel_size: voxel edge from the depth's own pixel footprint, coarsened to a memory budget
- calibrate_confidence: per-scene gate c* and slope gamma from cross-view depth residuals
- pixel_weights: (conf / c*)^(2 gamma) above the gate, 0 below it and on depth edges
- lift_weights: model-grid weights onto the frame grid, nearest, through each crop box
- no file IO: the caller writes the mesh once it is final; an optional volume ground (mesh.clean) fills voxels the views left below w_min
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
import torch.nn.functional as F
import torch.utils.dlpack as tdl
from scipy import ndimage
from tqdm.auto import tqdm

from collab_splats.geometry.projection import (
    depth_residual,
    unproject,
    unproject_frames,
)
from collab_splats.geometry.transforms import invert_poses, shift_intrinsics
from collab_splats.mesh.clean import VolumeGround
from collab_splats.mesh.utils import validate_views
from collab_splats.utils.torch_utils import full_fp32_matmul, get_device

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
    weights: np.ndarray,
    voxel_size: float,
    depth_trunc: float | None = None,
    sdf_trunc: float | None = None,
    w_min: float = 1.0,
    ground: VolumeGround | None = None,
    max_blocks: int = 30_000,
) -> o3d.geometry.TriangleMesh:
    """
    Integrate depth + RGB views into a VoxelBlockGrid and extract its mesh.

    - each pixel's sample carries its weight: tsdf and color are weighted running averages, the voxel weight their sum
    - runs on CUDA:0 when Open3D has CUDA, else on the CPU; same grid, same mesh
    - the block hashmap starts at 10k blocks and grows on its own
    - u, v floor (pixel-corner, as Open3D), so K shifts +0.5 once here
    - scenes past max_blocks fuse in spatial tiles; the welded mesh equals the single grid's
    - ground: its band blocks join the block union; each tile writes the ones in its halo; after the views, voxels inside its hull with weight <= w_min take its height distance at weight 1.01 w_min

    Args:
        depths: (N, H, W) depth in world units, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 RGB at the same resolution as depths.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) pixel-center intrinsics at the depth resolution.
        weights: (N, H, W) per-pixel fusion weight (pixel_weights); 0 skips the pixel.
        voxel_size: TSDF voxel edge in world units.
        depth_trunc: depth beyond this (world units) is ignored; None keeps every depth.
        sdf_trunc: truncation band in world units; None = 4 × voxel_size.
        w_min: a voxel needs summed weight above this to yield surface.
        ground: volume ground from build_volume_ground; None fuses the views alone.
        max_blocks: blocks per grid, halo included; Open3D extraction faults past 32,768.

    Returns:
        The extracted mesh with per-vertex color.

    Raises:
        ValueError: sdf_trunc is narrower than voxel_size.
        ValueError: weights is not shaped like depths.
    """
    rgbs, c2w, K, depths = validate_views(rgbs, c2w, K, depths)
    assert depths is not None

    # One weight per depth pixel
    if weights.shape != depths.shape:
        raise ValueError(
            f"create_tsdf_mesh: weights are {weights.shape} but depths are {depths.shape}"
        )

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
    torch_device = torch.device("cuda:0" if on_cuda else "cpu")

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
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    rgbs = np.ascontiguousarray(rgbs)
    K = shift_intrinsics(K, (0.5, 0.5))
    w2c = invert_poses(c2w)
    w2c = w2c.astype(np.float64)

    # Probe the blocks each view's truncation band touches; their union is the scene
    probe = o3d.t.geometry.VoxelBlockGrid(**grid_kwargs)
    view_blocks = []
    for i in range(n):
        # A view with no valid depth (e.g. all sky) touches no block; Open3D raises on it
        if not (depths[i] > 0).any():
            view_blocks.append(np.empty((0, 3), np.int64))
            continue

        depth = o3c.Tensor(depths[i], device=device)
        depth = o3d.t.geometry.Image(depth)
        blocks = probe.compute_unique_block_coordinates(
            depth, o3c.Tensor(K[i]), o3c.Tensor(w2c[i]), 1.0, depth_trunc, trunc_mult
        )
        view_blocks.append(blocks.cpu().numpy().astype(np.int64).reshape(-1, 3))
    del probe
    union = np.unique(np.concatenate(view_blocks), axis=0)

    # The volume ground's band blocks join the scene, so every tile holds its voxels
    if ground is not None:
        union = np.unique(np.concatenate([union, ground.blocks]), axis=0)

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
        desc = (
            f"weighted TSDF (voxel {voxel_size:g}, {device}, tile {t + 1}/{len(tiles)})"
        )

        for i in tqdm(range(n), desc=desc):
            blocks = view_blocks[i]
            if len(tiles) > 1:
                blocks = blocks[np.isin((blocks + (1 << 20)) @ key_weights, halo_keys)]
            if not len(blocks):
                continue

            # Weighted running average of this view's samples into the tile's voxels
            _integrate_weighted(
                grid,
                blocks,
                torch.from_numpy(depths[i]).to(torch_device),
                torch.from_numpy(rgbs[i]).to(torch_device).float(),
                torch.from_numpy(weights[i]).to(torch_device),
                K[i],
                w2c[i],
                depth_trunc=depth_trunc,
                sdf_trunc=sdf_trunc,
                device=device,
            )

        # Volume ground into the tile's voxels the views left below w_min
        if ground is not None:
            ground_blocks = ground.blocks
            if len(tiles) > 1:
                ground_blocks = ground_blocks[
                    np.isin((ground_blocks + (1 << 20)) @ key_weights, halo_keys)
                ]
            _inject_ground(grid, ground, ground_blocks, w_min=w_min, device=device)

        # Extract voxels whose summed weight passes w_min (Open3D keeps weight > threshold)
        tmesh = grid.extract_triangle_mesh(weight_threshold=float(w_min))
        del grid
        if on_cuda:
            torch.cuda.empty_cache()

        # A tile whose every sample weighs 0 (e.g. canopy) has no surface
        if not len(tmesh.vertex.positions):
            if len(tiles) == 1:
                mesh = o3d.geometry.TriangleMesh()
                break
            continue

        # Averaged colors can overshoot 1.0 by float error, so clip before the PLY writer clamps noisily
        tmesh.vertex.colors = tmesh.vertex.colors.clip(0.0, 1.0)

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
    if len(tiles) > 1 and parts:
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
    elif len(tiles) > 1:
        mesh = o3d.geometry.TriangleMesh()

    logger.info(
        "create_tsdf_mesh: %d views, %d blocks in %d tiles, ground=%d blocks, voxel=%.4f sdf_trunc=%.4f w_min=%.2f -> %d vertices",
        n,
        len(union),
        len(tiles),
        0 if ground is None else len(ground.blocks),
        voxel_size,
        sdf_trunc,
        w_min,
        len(mesh.vertices),
    )
    return _canonical_order(mesh)


def _canonical_order(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Same mesh with vertices sorted by position and faces sorted by their vertex indices.

    - Open3D's CUDA hashmap places blocks in a run-dependent order, so extraction order varies
    - downstream cleanup is order-sensitive; a fixed order makes the mesh stage repeatable
    - each face is rotated to start at its smallest index, which keeps its winding
    """
    if not len(mesh.vertices):
        return mesh

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles).astype(np.int64)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors)

    # Vertex order: position first, then normal and color to break ties
    keys = [v[:, 2], v[:, 1], v[:, 0]]
    if len(colors):
        keys = [colors[:, 2], colors[:, 1], colors[:, 0], *keys]
    if len(normals):
        keys = [normals[:, 2], normals[:, 1], normals[:, 0], *keys]
    order = np.lexsort(keys)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))

    # Faces: remap, rotate smallest index first, then sort rows
    f = rank[f]
    start = np.argmin(f, axis=1)[:, None]
    f = np.take_along_axis(f, (start + np.arange(3)) % 3, axis=1)
    f = f[np.lexsort(f.T[::-1])]

    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v[order]),
        o3d.utility.Vector3iVector(f.astype(np.int32)),
    )
    if len(normals):
        out.vertex_normals = o3d.utility.Vector3dVector(normals[order])
    if len(colors):
        out.vertex_colors = o3d.utility.Vector3dVector(colors[order])
    return out


########################################################################
# Weighted integration
########################################################################


def _integrate_weighted(
    grid: o3d.t.geometry.VoxelBlockGrid,
    blocks: np.ndarray,
    depth: torch.Tensor,
    color: torch.Tensor,
    weight: torch.Tensor,
    K: np.ndarray,
    w2c: np.ndarray,
    *,
    depth_trunc: float,
    sdf_trunc: float,
    device: o3c.Device,
) -> None:
    """
    One view's samples into the grid: per-pixel weight, tsdf and 0-255 color averaged in place.

    - K is already shifted +0.5: u, v floor like Open3D's own integrate
    - inliers: depth in (0, depth_trunc), sdf >= -sdf_trunc; sdf clamps to the band and scales to [-1, 1]
    - attribute buffers are re-read after activate, which can reallocate them
    """
    # Activate the view's blocks; voxel world coords and flat buffer indices
    hashmap = grid.hashmap()
    keys = o3c.Tensor(blocks.astype(np.int32), device=device)
    hashmap.activate(keys)
    buf_indices, _ = hashmap.find(keys)
    coords, flat = grid.voxel_coordinates_and_flattened_indices(buf_indices)
    coords = tdl.from_dlpack(coords.to_dlpack())
    flat = tdl.from_dlpack(flat.to_dlpack())

    # Voxels into the view; keep those landing on the depth grid
    R = torch.from_numpy(w2c[:3, :3]).float().to(coords.device)
    t = torch.from_numpy(w2c[:3, 3]).float().to(coords.device)
    xyz = coords @ R.T + t
    z = xyz[:, 2]
    u = torch.floor(float(K[0, 0]) * xyz[:, 0] / z + float(K[0, 2])).long()
    v = torch.floor(float(K[1, 1]) * xyz[:, 1] / z + float(K[1, 2])).long()
    height, width = depth.shape
    on_grid = (z > 0) & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    flat, z, u, v = flat[on_grid], z[on_grid], u[on_grid], v[on_grid]

    # Truncated signed distance along the ray's z; samples of weight 0 drop out
    d = depth[v, u]
    sdf = d - z
    inlier = (d > 0) & (d < depth_trunc) & (sdf >= -sdf_trunc)
    flat, u, v = flat[inlier], u[inlier], v[inlier]
    sdf = sdf[inlier].clamp(max=sdf_trunc) / sdf_trunc
    w = weight[v, u]
    keep = w > 0
    flat, sdf, w, u, v = flat[keep], sdf[keep], w[keep], u[keep], v[keep]

    # Weighted running average of tsdf and color; the voxel weight sums
    tsdf = tdl.from_dlpack(grid.attribute("tsdf").to_dlpack()).view(-1)
    total = tdl.from_dlpack(grid.attribute("weight").to_dlpack()).view(-1)
    rgb = tdl.from_dlpack(grid.attribute("color").to_dlpack()).view(-1, 3)
    w0 = total[flat]
    w1 = w0 + w
    tsdf[flat] = (tsdf[flat] * w0 + sdf * w) / w1
    rgb[flat] = (rgb[flat] * w0[:, None] + color[v, u] * w[:, None]) / w1[:, None]
    total[flat] = w1


def _inject_ground(
    grid: o3d.t.geometry.VoxelBlockGrid,
    ground: VolumeGround,
    blocks: np.ndarray,
    *,
    w_min: float,
    device: o3c.Device,
) -> None:
    """
    The ground's signed height distance into the grid's unobserved voxels inside its hull.

    - unobserved: summed weight <= w_min, so the voxel would not extract on its own
    - written at weight 1.01 w_min: Open3D extracts only weight > threshold
    - membrane height and color bilinear at the voxel's ground xy; color stays 0-255
    """
    if not len(blocks):
        return

    # Activate the ground's blocks; voxel world coords and flat buffer indices
    hashmap = grid.hashmap()
    keys = o3c.Tensor(blocks.astype(np.int32), device=device)
    hashmap.activate(keys)
    buf_indices, _ = hashmap.find(keys)
    coords, flat = grid.voxel_coordinates_and_flattened_indices(buf_indices)
    coords = tdl.from_dlpack(coords.to_dlpack()).double()
    flat = tdl.from_dlpack(flat.to_dlpack())
    on = coords.device

    # Each voxel in the ground frame
    frame = torch.from_numpy(ground.frame).double().to(on)
    center = torch.from_numpy(ground.center).double().to(on)
    local = (coords - center) @ frame.T

    # Inside-hull columns on the voxel-res mask
    inside = torch.from_numpy(ground.inside).to(on)
    rows_f, cols_f = inside.shape
    ix = torch.floor((local[:, 0] - ground.lo[0]) / ground.voxel_size).long()
    iy = torch.floor((local[:, 1] - ground.lo[1]) / ground.voxel_size).long()
    sel = (ix >= 0) & (iy >= 0) & (ix < cols_f) & (iy < rows_f)
    sel &= inside[iy.clamp(0, rows_f - 1), ix.clamp(0, cols_f - 1)]

    # Membrane height and color, bilinear at the voxel's ground xy
    # - fx/fy -0.5: cell centers at lo + (c + 0.5) res, prototype parity
    # - build_volume_ground bins coarse cells by round, i.e. centers at lo + c res
    rows_c, cols_c = ground.height.shape
    fx = (local[:, 0] - ground.lo[0]) / ground.res - 0.5
    fy = (local[:, 1] - ground.lo[1]) / ground.res - 0.5
    sample = torch.stack([fx / (cols_c - 1) * 2 - 1, fy / (rows_c - 1) * 2 - 1], -1)
    sample = sample[None, None].float()
    height_img = torch.from_numpy(ground.height).to(on).float()[None, None]
    height = F.grid_sample(height_img, sample, align_corners=True)[0, 0, 0]
    color_img = torch.from_numpy(ground.color).to(on).permute(2, 0, 1)[None].float()
    color = F.grid_sample(color_img, sample, align_corners=True)[0, :, 0].T

    # Unobserved voxels take the clamped height distance, just above w_min
    tsdf = tdl.from_dlpack(grid.attribute("tsdf").to_dlpack()).view(-1)
    total = tdl.from_dlpack(grid.attribute("weight").to_dlpack()).view(-1)
    rgb = tdl.from_dlpack(grid.attribute("color").to_dlpack()).view(-1, 3)
    sel &= total[flat] <= w_min
    sdf = (local[:, 2] - height).clamp(-ground.sdf_trunc, ground.sdf_trunc)
    sdf = (sdf / ground.sdf_trunc).float()
    tsdf[flat[sel]] = sdf[sel]
    total[flat[sel]] = 1.01 * w_min
    rgb[flat[sel]] = color[sel]


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


########################################################################
# Fusion weights
########################################################################


def calibrate_confidence(
    depth: np.ndarray,
    confidence: np.ndarray,
    extrinsics: np.ndarray,
    model_intrinsics: np.ndarray,
    *,
    sdf_trunc: float,
    min_pixels: int = 200,
) -> tuple[float, float]:
    """
    Per-scene confidence gate c* and error slope gamma from cross-view depth residuals.

    - every 2nd frame, every 2nd pixel: median |depth residual| over views ±2, ±5, ±10
    - occluded samples (sampled < 0.9 x expected) dropped; needs 2+ valid views
    - conf-bin medians of error / sdf_trunc, log-log line: c* where it crosses 1, gamma = -slope
    - runs on CUDA when available, else the CPU
    - projects at full fp32 matmul: a global TF32 (mapanything import) would round the residuals

    Args:
        depth: (N, h, w) model-grid depth, 0 = none.
        confidence: (N, h, w) depth confidence.
        extrinsics: (N, 4, 4) world-to-camera.
        model_intrinsics: (N, 3, 3) pixel-center intrinsics on the model grid.
        sdf_trunc: the fusion's TSDF band in world units.
        min_pixels: a conf bin enters the fit with more pixels than this.

    Returns:
        (c_star, gamma).

    Raises:
        ValueError: fewer than 4 frames, fewer than 3 conf bins, or error does not fall with confidence.
    """
    device = get_device()
    depth_t = torch.from_numpy(np.ascontiguousarray(depth, np.float32)).to(device)
    w2c = torch.from_numpy(np.asarray(extrinsics, np.float32)).to(device)
    K = torch.from_numpy(np.asarray(model_intrinsics, np.float32)).to(device)
    n = len(depth_t)

    # Per-pixel cross-view error on a stride-2 grid of every 2nd frame
    errs, confs, deps = [], [], []

    # Full fp32 matmul, so a global TF32 cannot round the residuals
    with full_fp32_matmul():
        for i in range(0, n, 2):
            others = [
                j for j in (i - 10, i - 5, i - 2, i + 2, i + 5, i + 10) if 0 <= j < n
            ]
            if len(others) < 2:
                continue

            points = unproject(depth_t[i], w2c[i], K[i])[::2, ::2].reshape(-1, 3)
            residual, expected, sampled, valid, _ = depth_residual(
                points, w2c[others], K[others], depth_t[others]
            )

            # Occluded or depthless samples out; a pixel needs two views left
            ok = valid & (sampled > 0) & (sampled >= 0.9 * expected)
            err = torch.where(ok, residual.abs(), torch.nan)
            err = torch.nanmedian(err, 0).values
            err[ok.sum(0) < 2] = torch.nan

            errs.append(err.cpu().numpy())
            confs.append(np.asarray(confidence[i, ::2, ::2]).ravel())
            deps.append(np.asarray(depth[i, ::2, ::2]).ravel())

    # Short sequences leave no frame with enough neighbors
    if not errs:
        raise ValueError(
            "calibrate_confidence: no frame has 2+ neighbor views; need >= 4 frames"
        )

    err = np.concatenate(errs)
    conf = np.concatenate(confs)
    keep = np.isfinite(err) & (np.concatenate(deps) > 0) & (err > 0)
    err, conf = err[keep], conf[keep]

    # Median error per confidence bin, in units of the band
    edges = (1, 1.5, 2, 3, 5, 10, 20, 40, 80, 1e9)
    medians = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf >= lo) & (conf < hi)
        if in_bin.sum() > min_pixels:
            medians.append(
                (np.median(conf[in_bin]), np.median(err[in_bin]) / sdf_trunc)
            )

    if len(medians) < 3:
        raise ValueError(
            f"calibrate_confidence: {len(medians)} conf bins with > {min_pixels} pixels; need 3"
        )

    # Log-log line; c* is where the error equals the band
    log_conf, log_err = np.log(np.array(medians)).T
    slope, intercept = np.polyfit(log_conf, log_err, 1)

    if slope >= 0:
        raise ValueError(
            f"calibrate_confidence: error does not fall with confidence (slope {slope:.3f})"
        )

    c_star = float(np.exp(-intercept / slope))
    gamma = float(-slope)
    logger.info(
        "calibrate_confidence: c* %.3f, gamma %.3f over %d bins (%d pixels)",
        c_star,
        gamma,
        len(medians),
        len(err),
    )
    return c_star, gamma


def pixel_weights(
    depth: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    c_star: float = 1.0,
    gamma: float = 0.0,
    edge_rtol: float = 0.03,
) -> np.ndarray:
    """
    Fusion weight per depth pixel: inverse error variance above the confidence gate.

    - error ~ conf^-gamma, so the inverse variance is conf^(2 gamma); normalized to 1 at c*
    - 0 below c*, where depth is 0, and on depth edges (VGGT-Omega depth_edge rule)
    - no confidence (sfm, splats renders): 1 on every pixel off an edge

    Args:
        depth: (N, H, W) depth, 0 = none.
        confidence: (N, H, W) depth confidence on the same grid, or None.
        c_star: confidence at which the cross-view error equals the TSDF band.
        gamma: slope of log error against log confidence, negated.
        edge_rtol: 3x3 (max - min) / depth above this marks an edge.

    Returns:
        (N, H, W) float32 weights.
    """
    valid = depth > 0

    # Gate and power on confidence, or unit weights without it
    if confidence is None:
        weights = valid.astype(np.float32)
    else:
        conf = np.asarray(confidence, np.float32)
        gated = valid & (conf >= c_star)
        weights = np.where(gated, (conf / c_star) ** (2 * gamma), 0.0)
        weights = weights.astype(np.float32)

    # Depth edges: a 3x3 relative depth jump straddles two surfaces
    hi = ndimage.maximum_filter(depth, size=(1, 3, 3))
    lo = ndimage.minimum_filter(depth, size=(1, 3, 3))
    edge = (hi - lo) / np.maximum(np.abs(depth), 1e-6) > edge_rtol
    weights[edge] = 0.0

    return weights


def lift_weights(
    weights: np.ndarray, original_coords: np.ndarray, frame_hw: tuple[int, int]
) -> np.ndarray:
    """
    Model-grid weights resized nearest into each frame's crop box; 0 outside the box.

    - nearest, so a zero-weight edge pixel stays zero instead of blurring into its neighbors

    Args:
        weights: (N, h, w) model-grid weights.
        original_coords: (N, 6) rows of x0, y0, x1, y1, W, H; the first four are the crop box.
        frame_hw: frame grid height and width.

    Returns:
        (N, H, W) float32 frame-grid weights.
    """
    height, width = frame_hw
    boxes = np.asarray(original_coords)[:, :4].round().astype(int)
    lifted = np.zeros((len(weights), height, width), np.float32)

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        lifted[i, y0:y1, x0:x1] = cv2.resize(
            weights[i], (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
        )

    return lifted
