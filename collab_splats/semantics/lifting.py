"""
Lift dense per-frame feature maps onto a reconstruction's points.

- multi-view: every point projects into every frame, weighted by confidence and depth consistency
- source fallback: points visible nowhere sample their own source pixel
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from collab_splats.geometry.projection import depth_residual
from collab_splats.utils.torch_utils import get_device

if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult


########################################################
########## Feature sampling ############################
########################################################


def _grid_sample_at_pixels(
    fmap: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """
    Bilinear-sample fmap (D, H_p, W_p) at (rows, cols) given in the image_size frame.

    - returns (P_i, D) float32 on fmap's device
    - pixel centers normalized to [-1, 1] with align_corners=False, so any (H_p, W_p) works
    """
    H, W = image_size

    # grid_sample wants (N, H_out, W_out, 2); treat P_i points as (1, 1, P_i, 2)
    x = (2 * cols + 1) / W - 1
    y = (2 * rows + 1) / H - 1
    grid = torch.stack([x, y], dim=-1)
    grid = grid.view(1, 1, -1, 2)
    fmap_b = fmap.unsqueeze(0)
    fmap_b = fmap_b.float()
    sampled = F.grid_sample(
        fmap_b,
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )

    # (1, D, 1, P_i) → (P_i, D)
    sampled = sampled.squeeze(0)
    sampled = sampled.squeeze(1)
    return sampled.T


def _sample_at_source_pixels(
    feature_maps: list[torch.Tensor],
    pixel_indices: np.ndarray,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """
    Source-frame-only lift: each point sampled at its own (frame_id, row, col).

    - fallback for points with zero multi-view weight
    - zero weight: never visible, or always depth-inconsistent
    """
    P = len(pixel_indices)
    D = feature_maps[0].shape[0]
    H, W = image_size
    out = torch.zeros((P, D), dtype=torch.float32)
    for i, fmap in enumerate(feature_maps):
        mask_i = pixel_indices[:, 0] == i
        if not mask_i.any():
            continue
        # Clip to model frame to handle any rounding drift from upstream
        rows = np.clip(pixel_indices[mask_i, 1], 0, H - 1)
        cols = np.clip(pixel_indices[mask_i, 2], 0, W - 1)
        rows_f = rows.astype(np.float32)
        cols_f = cols.astype(np.float32)
        rows_t = torch.from_numpy(rows_f)
        cols_t = torch.from_numpy(cols_f)
        rows_t = rows_t.to(fmap.device)
        cols_t = cols_t.to(fmap.device)
        sampled = _grid_sample_at_pixels(fmap, rows_t, cols_t, image_size)
        out[mask_i] = sampled.cpu()
    return out


def lift_features(
    feature_maps: list[torch.Tensor],
    result: PointcloudResult,
    *,
    depth_tol: float = 0.05,
) -> torch.Tensor:
    """
    Multi-view confidence-weighted lift of dense feature maps to per-point features.

    - each point in result.points projects into every frame
    - masked by depth_residual's in-bounds + depth consistency (|residual| / |z_proj| < depth_tol)
    - weighted by confidence at the same rounded pixel the depth test reads, then mean-reduced
    - points visible nowhere fall back to a source-frame sample at their pixel_indices entry

    Args:
        feature_maps: List of (D, H_p, W_p) per-frame dense features. Caller runs the
            extractor (and optional AE encode) first.
        result:       Carries points, pixel_indices, depth, extrinsics, model_intrinsics,
            model_height, model_width. `confidence` is optional — SfM results carry none
            and fall back to uniform per-pixel weights. Reload zarr with load_images=True
            before re-extracting so the extractor sees the depth map's FOV.
        depth_tol:    Relative depth tolerance for the visibility test.

    Returns:
        (P, D) float32 tensor of per-point features, aligned with result.points.
    """
    # Required fields — fail loud at function entry, not deep in the kernel
    for name in ("points", "pixel_indices", "depth", "extrinsics", "model_intrinsics"):
        assert getattr(result, name) is not None, (
            f"lift_features requires result.{name}; " f"load zarr with load_images=True or run pipeline fresh"
        )
    N = result.extrinsics.shape[0]
    assert len(feature_maps) == N, f"feature_maps count ({len(feature_maps)}) != frame count ({N})"

    H, W = result.model_height, result.model_width
    P = result.points.shape[0]
    D = feature_maps[0].shape[0]
    image_size = (H, W)

    # Whole kernel on one device in float32 (GPU when available); one .cpu() at the end
    device = get_device()

    points_c = np.ascontiguousarray(result.points)
    pts = torch.as_tensor(points_c, dtype=torch.float32, device=device)

    # Extrinsics and depth are (N, 4, 4) / (N, H, W) by contract (see PointcloudResult) — fail loud otherwise
    if result.extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics must be (N, 4, 4), got {result.extrinsics.shape}")
    if result.depth.ndim != 3:
        raise ValueError(f"depth must be (N, H, W), got {result.depth.shape}")
    extrinsics_c = np.ascontiguousarray(result.extrinsics)
    intrinsics_c = np.ascontiguousarray(result.model_intrinsics)
    ext = torch.as_tensor(extrinsics_c, dtype=torch.float32, device=device)  # (N, 4, 4)
    intr = torch.as_tensor(intrinsics_c, dtype=torch.float32, device=device)  # (N, 3, 3)

    # Conf → (N, H, W) float32 on device; absent confidence (SfM results) = uniform weights
    conf = (
        torch.ones((N, H, W), dtype=torch.float32, device=device)
        if result.confidence is None
        else result.confidence.to(device=device, dtype=torch.float32)
    )

    depth_c = np.ascontiguousarray(result.depth)
    depth = torch.as_tensor(depth_c, dtype=torch.float32, device=device)

    features_sum = torch.zeros((P, D), dtype=torch.float32, device=device)
    weights_sum = torch.zeros((P,), dtype=torch.float32, device=device)

    for i in range(N):
        fmap = feature_maps[i].to(device=device, dtype=torch.float32)  # (D, H_p, W_p)

        # Visibility: in front, in bounds, depth-consistent against frame i's depth map
        # - depth_residual projects all P points and reads the depth map nearest-neighbor
        # - the relative tolerance stays lifting's own, applied to the returned depths
        residual, expected, _, valid, pixels = depth_residual(pts, ext[i], intr[i], depth[i])
        depth_ok = residual.abs() / (expected.abs() + 1e-8) < depth_tol
        u = pixels[:, 0]
        v = pixels[:, 1]

        # Confidence weight at the pixel the depth test read
        # - rounded, as depth_residual's nearest read, so one pixel per point for test + weight
        # - nan/inf coords (degenerate points) → 0 for indexing/sampling; valid is False there
        u_safe = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v_safe = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        u_px = torch.round(u_safe)
        v_px = torch.round(v_safe)
        u_idx = u_px.clamp(0, W - 1)
        v_idx = v_px.clamp(0, H - 1)
        u_idx = u_idx.long()
        v_idx = v_idx.long()
        visible = valid & depth_ok
        w = conf[i, v_idx, u_idx] * visible.float()  # (P,)

        # Bilinear sample feature map at projected coords via the shared grid_sample helper
        sampled = _grid_sample_at_pixels(fmap, v_safe, u_safe, image_size)  # (P, D)

        features_sum += sampled * w.unsqueeze(-1)
        weights_sum += w

    # Weighted mean with eps for numerical safety
    features = features_sum / (weights_sum.unsqueeze(-1) + 1e-8)

    # Fallback: points with zero accumulated weight → source-frame sample
    zero_w = weights_sum < 1e-6
    any_zero = zero_w.any()
    if bool(any_zero):
        zero_idx = zero_w.detach()
        zero_idx = zero_idx.cpu()
        zero_idx = zero_idx.numpy()
        fallback = _sample_at_source_pixels(feature_maps, result.pixel_indices[zero_idx], image_size)
        features[zero_w] = fallback.to(device)

    features = features.detach()
    return features.cpu()
