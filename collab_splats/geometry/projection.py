"""
Pinhole projection: unprojection, projection, world-point lookup, cross-view depth agreement.

- tensor helpers work in the input dtype
- unproject_frames and multiview_depth_confidence take numpy, run float32 on get_device()
- sample_world_points: numpy bilinear lookup of a per-pixel world-point map
- poses are w2c OpenCV, column vectors, `x_cam = R @ x_world + t`
- K is on the depth map's own pixel grid
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from collab_splats.geometry.transforms import transform_points
from collab_splats.utils.torch_utils import (
    batch_iterator,
    full_fp32_matmul,
    get_device,
    to_numpy,
)

########################################################################
# Unprojection
########################################################################


def unproject(depth: Tensor, world_to_cam: Tensor, intrinsics: Tensor) -> Tensor:
    """
    World point of every depth pixel, sampled at integer pixel coordinates.

    - camera ray `((u - cx)/fx, (v - cy)/fy, 1)`, scaled by depth
    - `x_world = R^T (d * ray - t)`, written as `(p - t) @ R`
    - inverse written out: transform_points applies T, not its inverse, and batches only (B, P, 3) points

    Args:
        depth: (..., H, W) z-depth.
        world_to_cam: (..., 4, 4) or (..., 3, 4) w2c.
        intrinsics: (..., 3, 3) camera matrices.

    Returns:
        (..., H, W, 3) world points.
    """
    # Integer pixel grid in the intrinsics' dtype and device
    height, width = depth.shape[-2], depth.shape[-1]
    u = torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype)
    v = torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

    # Per-camera K broadcast over the grid: unit-z rays, scaled by depth
    fx = intrinsics[..., 0, 0, None, None]
    fy = intrinsics[..., 1, 1, None, None]
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    ray_x = (grid_u - cx) / fx
    ray_y = (grid_v - cy) / fy
    rays = torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)
    points_cam = rays * depth[..., None]

    # Camera -> world
    rotation = world_to_cam[..., None, :3, :3]
    translation = world_to_cam[..., None, None, :3, 3]
    return (points_cam - translation) @ rotation


def unproject_frames(
    depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, *, batch_size: int = 100
) -> np.ndarray:
    """
    World point of every depth pixel, unprojected batch by batch on get_device().

    - batch_size bounds device memory
    - full fp32 matmul: TF32, which a mapanything import enables, would round the rotation

    Args:
        depth: (N, H, W) z-depth.
        extrinsics: (N, 4, 4) or (N, 3, 4) w2c.
        intrinsics: (N, 3, 3) K on the depth grid.
        batch_size: frames unprojected per device batch.

    Returns:
        (N, H, W, 3) float32 world points.
    """
    device = get_device()
    world_points = np.empty((*depth.shape, 3), dtype=np.float32)

    # Full fp32 matmul: each batch on device, copied back into the host array
    with full_fp32_matmul():
        for batch_depth, world_to_cam, K, out in batch_iterator(
            batch_size, depth, extrinsics, intrinsics, world_points
        ):
            batch_depth = torch.as_tensor(batch_depth, dtype=torch.float32, device=device)
            world_to_cam = torch.as_tensor(world_to_cam, dtype=torch.float32, device=device)
            K = torch.as_tensor(K, dtype=torch.float32, device=device)
            points = unproject(batch_depth, world_to_cam, K)
            out[:] = to_numpy(points)

    return world_points


########################################################################
# Projection
########################################################################


def project(
    points_world: Tensor, world_to_cam: Tensor, intrinsics: Tensor, *, min_depth: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """
    Pixel coordinates of world points in one camera, or in each camera of a batch.

    - one (3, 3) K: 0-dim scalar terms, so a (3,) point returns (2,) and (P, 3) points keep their dtype
    - a (B, 3, 3) batch must share the points' dtype and device: focal terms broadcast as (B, 1)

    Args:
        points_world: (..., 3) world points; (P, 3) or (B, P, 3) with a pose batch.
        world_to_cam: (4, 4) or (3, 4) w2c, or a (B, 4, 4) batch.
        intrinsics: (3, 3) camera matrix, or a (B, 3, 3) batch.
        min_depth: perspective-divide floor; a point at or behind the camera divides by it.

    Returns:
        (pixels (..., 2), camera-frame points (..., 3)); (B, P, 2) and (B, P, 3) for a batch.
    """
    # World -> camera
    points_cam = transform_points(points_world, world_to_cam)

    # One K: 0-dim scalars, as before the pose batch; a batch broadcasts (B, 1) over the points
    if intrinsics.ndim == 2:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    else:
        fx, fy = intrinsics[..., 0, 0, None], intrinsics[..., 1, 1, None]
        cx, cy = intrinsics[..., 0, 2, None], intrinsics[..., 1, 2, None]

    # Clamped divide: unclamped, a mask's 0 * inf poisons the backward
    depth = points_cam[..., 2].clamp(min=min_depth)
    u = points_cam[..., 0] * fx / depth + cx
    v = points_cam[..., 1] * fy / depth + cy
    return torch.stack([u, v], dim=-1), points_cam


def sample_world_points(world_points: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinear world points at pixel coordinates (hloc interpolate_scan analog).

    - align_corners=True: pixel index i sits at coordinate i
    - invalid where the sample touches NaN or px falls outside the map; never a zero check

    Args:
        world_points: (H, W, 3) per-pixel world points.
        px: (K, 2) float32 xy on that grid.

    Returns:
        pts3d (K, 3) float32 and valid (K,) bool.
    """
    H, W, _ = world_points.shape

    # Normalize to [-1, 1] for grid_sample
    norm = px / np.array([[W - 1, H - 1]], dtype=np.float32) * 2 - 1
    norm = norm.astype(np.float32)
    grid = torch.from_numpy(norm)
    wp = torch.from_numpy(world_points).float()
    wp = wp.permute(2, 0, 1)
    sampled = F.grid_sample(wp[None], grid[None, None], align_corners=True, mode="bilinear")
    interp = sampled[0, :, 0]

    # NaN marks unmapped pixels; grid_sample pads out-of-bounds samples, so bounds are checked too
    nan = torch.isnan(interp)
    valid = ~nan.any(dim=0)
    in_bounds = (px[:, 0] >= 0) & (px[:, 0] <= W - 1) & (px[:, 1] >= 0) & (px[:, 1] <= H - 1)
    valid = valid.numpy() & in_bounds

    pts3d = interp.T.numpy()
    return pts3d.astype(np.float32), valid


########################################################################
# Cross-view depth residual
########################################################################


def depth_residual(
    points_world: Tensor, world_to_cam: Tensor, intrinsics: Tensor, depth: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Camera depth of world points against another view's depth map at their projected pixels.

    - the depth map is read nearest-neighbor: bilinear across a depth edge yields a depth on no surface
    - in bounds means inside the pixel-center span [0, W-1] x [0, H-1]
    - pixels returned, so callers need no second project
    - no tolerance here: each caller applies its own to the returned depths
    - a batch of B views returns each output with a leading B

    Args:
        points_world: (P, 3) world points.
        world_to_cam: (4, 4) w2c of the view whose depth is read, or a (B, 4, 4) batch.
        intrinsics: (3, 3) camera matrix on the depth map's pixel grid, or a (B, 3, 3) batch.
        depth: (H, W) z-depth of that view, or a (B, H, W) batch; 0 marks no depth.

    Returns:
        (residual, expected, sampled, valid, pixels), each (P,) but pixels (P, 2); (B, P) and (B, P, 2) for a batch
        - residual: expected minus sampled
        - expected: the point's z in the camera
        - sampled: the depth map at the projected pixel, 0 outside the grid
        - valid: in front of the camera and projected in bounds
        - pixels: project's (u, v) of each point
    """
    # Depth grid size; a batch shares one H, W
    height, width = depth.shape[-2:]

    # World -> pixel; project's clamped divide keeps points behind the camera finite
    pixels, points_cam = project(points_world, world_to_cam, intrinsics)
    expected = points_cam[..., 2]
    in_front = expected > 0

    # Pixels -> grid_sample's [-1, 1] frame, corners on the outer pixel centers
    grid_u = pixels[..., 0] / (width - 1) * 2 - 1
    grid_v = pixels[..., 1] / (height - 1) * 2 - 1
    grid = torch.stack([grid_u, grid_v], dim=-1)
    in_bounds = (grid[..., 0] >= -1) & (grid[..., 0] <= 1) & (grid[..., 1] >= -1) & (grid[..., 1] <= 1)

    # Nearest read of each view's depth map at its projected pixels
    depth_map = depth.reshape(-1, 1, height, width)
    grid = grid.reshape(depth_map.shape[0], 1, -1, 2)
    sampled = F.grid_sample(depth_map, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    sampled = sampled.reshape(expected.shape)

    # Signed residual: positive when the point lies behind the observed surface
    residual = expected - sampled
    return residual, expected, sampled, in_front & in_bounds, pixels


def depth_agreement(
    points_world: Tensor, world_to_cam: Tensor, intrinsics: Tensor, depth: Tensor, rel_thresh: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Whether world points agree with another view's depth map, within a relative tolerance.

    - occluded (the view sees a nearer surface) is no evidence: not seen
    - a hole in the view's depth map is seen but cannot agree
    - scale-free: tolerance is rel_thresh times the point's own depth
    - a batch of B views returns each output with a leading B

    Args:
        points_world: (P, 3) world points.
        world_to_cam: (4, 4) w2c of the view whose depth is read, or a (B, 4, 4) batch.
        intrinsics: (3, 3) camera matrix on the depth map's pixel grid, or a (B, 3, 3) batch.
        depth: (H, W) z-depth of that view, or a (B, H, W) batch; 0 marks no depth.
        rel_thresh: tolerance as a fraction of the point's depth.

    Returns:
        (agree, seen, rel_residual, expected), each (P,); (B, P) for a batch
        - rel_residual: (sampled - expected) / expected; exactly -1 where sampled is 0
        - expected: the point's z in the camera
    """
    residual, expected, sampled, valid, _ = depth_residual(points_world, world_to_cam, intrinsics, depth)

    # Nearer sampled surface = occluded; farther = free-space violation, still seen
    tol = rel_thresh * expected.abs()
    has_depth = sampled > 0
    occluded = (sampled < expected - tol) & has_depth
    seen = valid & ~occluded
    agree = seen & has_depth & (residual.abs() < tol)

    return agree, seen, (sampled - expected) / expected, expected


########################################################################
# Multiview depth confidence
########################################################################


def multiview_depth_confidence(
    depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, rel_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel count of other views that see the pixel and that agree with its depth.

    - port of mapanything/utils/multiview_confidence.py:125 (facebookresearch/map-anything)
    - every ordered pair, O(N^2); pixels with no source depth count nothing
    - depth, intrinsics and extrinsics on one pixel grid, OpenCV, +Z z-depth

    Args:
        depth: (N, H, W) z-depth per frame; 0 marks no depth.
        intrinsics: (N, 3, 3) K on the depth grid.
        extrinsics: (N, 4, 4) w2c.
        rel_thresh: agreement tolerance as a fraction of depth.

    Returns:
        (agree, seen), both (N, H, W) int32.

    Raises:
        ValueError: N disagrees across the arrays, or the principal point lies outside the depth
            grid (model-res depth paired with original-res intrinsics).
    """
    n, h, w = depth.shape

    # Alignment contract: same N, K on the depth grid
    if len(intrinsics) != n or len(extrinsics) != n:
        raise ValueError(
            f"length mismatch: depth has {n} frames, intrinsics {len(intrinsics)}, extrinsics {len(extrinsics)}"
        )
    cx, cy = float(intrinsics[0][0, 2]), float(intrinsics[0][1, 2])
    if not (0 < cx < w and 0 < cy < h):
        raise ValueError(
            f"intrinsics/depth resolution mismatch: principal point ({cx:.1f}, {cy:.1f}) lies outside a "
            f"{w}x{h} depth grid; model-resolution depth was probably paired with original-resolution intrinsics"
        )

    # Float32 tensors on one device; int32 accumulators over the flat pixel grid
    device = torch.device(get_device())
    depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    extrinsics_t = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    agree = torch.zeros(n, h * w, dtype=torch.int32, device=device)
    seen = torch.zeros(n, h * w, dtype=torch.int32, device=device)

    # Every source frame against every other frame
    for i in range(n):
        points = unproject(depth_t[i], extrinsics_t[i], intrinsics_t[i]).reshape(-1, 3)
        has_source = depth_t[i].reshape(-1) > 0

        for j in range(n):
            if i == j:
                continue

            agree_ij, seen_ij, _, _ = depth_agreement(points, extrinsics_t[j], intrinsics_t[j], depth_t[j], rel_thresh)
            seen_ij &= has_source
            seen[i] += seen_ij
            agree[i] += agree_ij & seen_ij

    return agree.reshape(n, h, w).cpu().numpy(), seen.reshape(n, h, w).cpu().numpy()
