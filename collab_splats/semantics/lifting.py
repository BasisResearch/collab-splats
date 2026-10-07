"""
Lift dense per-frame feature maps onto a reconstruction's points.

- multi-view: every point projects into every frame, weighted by confidence and depth consistency
- source fallback: points visible nowhere sample their own source pixel, when they have one
- transfer_features: point features onto other positions (mesh vertices, or smoothing in place)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from collab_splats.geometry.projection import depth_residual
from collab_splats.utils.torch_utils import get_device

if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult


########################################################
########## Feature sampling ############################
########################################################


def lift_features(
    frame_features: Callable[[int], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    result: PointcloudResult,
    *,
    depth_tol: float = 0.05,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Multi-view confidence-weighted lift of dense feature maps to per-point features.

    - each point in result.points projects into every frame; only visible points are sampled,
      so each frame costs its visible count, not P
    - masked by depth_residual's in-bounds + depth consistency (|residual| / |z_proj| < depth_tol)
    - weighted by confidence at the same rounded pixel the depth test reads, then mean-reduced
    - points visible nowhere fall back to a source-frame sample at their pixel_indices entry
    - without pixel_indices (e.g. mesh vertices) they stay zero: unobserved
    - indexed maps add each point's bilinear blend of its 4 nearest patches' lists at the id
      columns: the dense lift's result without the zero columns

    Args:
        frame_features: frame i's features, called once per frame, then once per fallback frame
            (dense maps only); a list passes `maps.__getitem__`. Either a dense (D, H_p, W_p) map,
            any float dtype, or an indexed map: an (ids, values) pair, each (K, H_p, W_p), listing
            K entries per patch of a num_classes-long vector whose other entries are 0.
        result: points, depth, extrinsics, model_intrinsics and model grid size; `confidence`
            (uniform when absent) and `pixel_indices` (no fallback when absent) are optional.
        depth_tol: relative depth tolerance for the visibility test.
        num_classes: indexed maps only; number of distinct ids, i.e. the output width. Values at
            the K listed ids per patch are summed per point; K is never a dimension. None for
            dense maps.

    Returns:
        (P, D) or (P, num_classes) float32 tensor of per-point features, aligned with result.points.

    Raises:
        ValueError: when a required field is missing, or extrinsics / depth have the wrong shape;
            indexed maps without num_classes, with pixel_indices, or with an id outside
            [0, num_classes); num_classes with dense maps.
    """
    # Required fields — fail loud at function entry, not deep in the kernel
    for name in ("points", "depth", "extrinsics", "model_intrinsics"):
        if getattr(result, name) is None:
            raise ValueError(f"lift_features requires result.{name}")

    N = result.extrinsics.shape[0]

    H, W = result.model_height, result.model_width
    P = result.points.shape[0]
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

    features_sum = None
    weights_sum = torch.zeros((P,), dtype=torch.float32, device=device)

    for i in range(N):
        fmap = frame_features(i)

        # Accumulator width: num_classes for indexed maps, else the first frame's channels
        if features_sum is None:
            indexed = not isinstance(fmap, torch.Tensor)

            if indexed and num_classes is None:
                raise ValueError("indexed (ids, values) maps need num_classes, the number of distinct ids")

            if indexed and result.pixel_indices is not None:
                raise ValueError("indexed maps take no pixel_indices fallback; pass a result without pixel_indices")

            if not indexed and num_classes is not None:
                raise ValueError("num_classes applies to indexed maps only; dense maps keep their channels")

            width = num_classes if indexed else fmap.shape[0]
            features_sum = torch.zeros((P, width), dtype=torch.float32, device=device)

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

        # Sample and accumulate visible points only; the rest carry weight 0
        idx = torch.nonzero(w > 0).squeeze(1)

        if indexed:
            ids, values = fmap
            _add_indexed_bilinear(features_sum, ids, values, idx, v_safe[idx], u_safe[idx], w[idx], image_size)
        else:
            fmap = fmap.to(device=device, dtype=torch.float32)  # (D, H_p, W_p)
            sampled = _grid_sample_at_pixels(fmap, v_safe[idx], u_safe[idx], image_size)  # (P_vis, D)
            sampled *= w[idx].unsqueeze(-1)
            features_sum.index_add_(0, idx, sampled)

        weights_sum += w

    # Weighted mean in place, eps for numerical safety
    features = features_sum
    features /= weights_sum.unsqueeze(-1) + 1e-8

    # Fallback: zero-weight points sample their own source pixel, clipped; loads only frames holding one
    zero_w = weights_sum < 1e-6
    if result.pixel_indices is not None and bool(zero_w.any()):
        pixel_indices = torch.as_tensor(result.pixel_indices, device=device)
        frame_ids = pixel_indices[zero_w, 0].unique()

        for i in frame_ids.tolist():
            mask_i = zero_w & (pixel_indices[:, 0] == i)
            fmap = frame_features(i)
            fmap = fmap.to(device=device, dtype=torch.float32)
            rows = pixel_indices[mask_i, 1].clamp(0, H - 1)
            cols = pixel_indices[mask_i, 2].clamp(0, W - 1)
            features[mask_i] = _grid_sample_at_pixels(fmap, rows.float(), cols.float(), image_size)

    features = features.detach()
    return features.cpu()


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


def _add_indexed_bilinear(
    acc: torch.Tensor,
    ids: torch.Tensor,
    values: torch.Tensor,
    idx: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    weights: torch.Tensor,
    image_size: tuple[int, int],
) -> None:
    """
    Add each point's bilinear blend of its 4 nearest patches' (id, value) lists into acc, in place.

    - ids, values (K, H_p, W_p); point j's blend goes to row idx[j], each value at column id
    - same corners and weights as grid_sample (border, align_corners=False), so it equals the
      dense lift of the same lists scattered to acc's width
    - raises ValueError on an id < 0 or >= acc width, which would land in another point's row
    """
    # An id outside [0, num_classes) would write into another point's row
    lo, hi = int(ids.min()), int(ids.max())

    if lo < 0 or hi >= acc.shape[1]:
        raise ValueError(f"ids span [{lo}, {hi}], outside [0, num_classes={acc.shape[1]})")

    # Lists as patch rows: (H_p * W_p, K) ids and values
    k, height, width = ids.shape
    ids = ids.to(device=acc.device, dtype=torch.long).reshape(k, -1).T
    values = values.to(device=acc.device, dtype=torch.float32).reshape(k, -1).T
    H, W = image_size

    # Patch-grid coords, clamped as border padding clamps them
    x = (cols + 0.5) * width / W - 0.5
    y = (rows + 0.5) * height / H - 0.5
    x = x.clamp(0, width - 1)
    y = y.clamp(0, height - 1)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x - x0
    wy = y - y0

    # Four nearest patches per point, each weighted by its bilinear share times the point weight
    corners = torch.cat([y0 * width + x0, y0 * width + x1, y1 * width + x0, y1 * width + x1])
    corner_w = torch.cat([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
    corner_w *= weights.repeat(4)

    # K entries per patch into the flat accumulator: row * num_classes + id
    flat = idx.repeat(4).unsqueeze(1) * acc.shape[1] + ids[corners]
    weighted = values[corners] * corner_w.unsqueeze(1)
    acc.view(-1).index_add_(0, flat.reshape(-1), weighted.reshape(-1))


########################################################
########## Feature transfer ############################
########################################################


def transfer_features(
    targets: np.ndarray,
    points: np.ndarray,
    features: np.ndarray,
    k: int = 5,
    max_dist: float = 0.03,
) -> np.ndarray:
    """
    Gaussian-weighted k-NN scatter of point features onto target positions.

    - targets: mesh vertices, or the points themselves to smooth features over k neighbors

    Args:
        targets: (M, 3) positions receiving features.
        points: (P, 3) point positions in the same frame.
        features: (P, D) per-point features.
        k: nearest targets each point contributes to.
        max_dist: points whose nearest target is farther than this (world units) are dropped.

    Returns:
        (M, D) array in features.dtype; targets no point reached are zero.
    """
    targets = np.asarray(targets)
    M = len(targets)
    D = features.shape[1]

    # Nearest-target query for every point; workers=-1 uses all cores
    tree = cKDTree(targets)
    distances, indices = tree.query(points, k=k, workers=-1)

    # k=1 collapses the neighbor axis; restore it so the kernel below is uniform
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Drop points whose closest target is beyond the truncation band
    valid_mask = distances[:, 0] <= max_dist

    if not np.any(valid_mask):
        return np.zeros((M, D), dtype=features.dtype)

    distances = distances[valid_mask]
    indices = indices[valid_mask]
    feats = features[valid_mask]

    # Move the aggregation to the GPU (float32); one .cpu() at the end
    device = get_device()
    distances = np.ascontiguousarray(distances)
    indices = np.ascontiguousarray(indices)
    feats = np.ascontiguousarray(feats)
    d = torch.as_tensor(distances, dtype=torch.float32, device=device)
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    f = torch.as_tensor(feats, dtype=torch.float32, device=device)

    # Gaussian kernel over neighbor distances; floored sigma keeps all-zero distances finite (uniform weights)
    sigma = d.mean().clamp_min(1e-12)
    w = torch.exp(-(d**2) / (2 * sigma**2))

    # Normalize weights per point (over k)
    w = w / w.sum(dim=1, keepdim=True)

    # Scatter weighted features to targets; accumulate weights for normalization
    acc = torch.zeros((M, D), dtype=torch.float32, device=device)
    wsum = torch.zeros((M, 1), dtype=torch.float32, device=device)

    for j in range(k):
        acc.index_add_(0, idx[:, j], f * w[:, j : j + 1])
        wsum.index_add_(0, idx[:, j], w[:, j : j + 1])

    # Normalize aggregated features by summed weights (untouched targets stay zero)
    nz = wsum.squeeze(1) > 0
    acc[nz] /= wsum[nz]

    return acc.cpu().numpy().astype(features.dtype)
