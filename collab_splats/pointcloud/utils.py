# collab_splats/pointcloud/utils.py
"""
Geometric pointcloud utilities: outlier masking, subsampling, plane fitting, feature lifting.

Coordinate convention, everywhere in this module:

- input from pycolmap is COLMAP world (Y-down) + OpenCV camera axes (X right, Y down, Z forward)
- world output (reproject_pixels) stays in the caller's frame — COLMAP world, Y-down
- nothing here converts to nerfstudio/OpenGL
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    invert_poses,
    rotation_align_vectors,
)

if TYPE_CHECKING:
    from .feedforward.base import FeedforwardResult

logger = logging.getLogger(__name__)


########################################################
########## Cleaning and subsampling ####################
########################################################


def clean_pointcloud(
    points: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    Statistical-outlier keep-mask over a world-point array.

    - all-True whenever open3d rejects everything — too few points, or a degenerate cloud
      with no spread for the neighborhood statistic to separate

    Args:
        points:       (P, 3) world points.
        nb_neighbors: Neighbors open3d averages over per point.
        std_ratio:    Distance cutoff in standard deviations of that average.

    Returns:
        (P,) bool array, True for the points open3d keeps.
    """
    pts = np.asarray(points, dtype=np.float64)

    # open3d returns an EMPTY keep set on clouds this small (measured: n<=2 -> 0 kept, o3d 0.19.0)
    # rather than raising, so clamp to all-True instead of silently deleting the cloud.
    if len(pts) <= nb_neighbors:
        return np.ones(len(pts), dtype=bool)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _, keep_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    keep = np.zeros(len(pts), dtype=bool)
    keep[np.asarray(keep_idx, dtype=int)] = True

    # Nothing survived: a degenerate or duplicate cloud, not a cloud of outliers. Keep everything
    # rather than emptying the reconstruction (same policy as confidence_mask).
    if not keep.any():
        logger.warning("clean_pointcloud: outlier removal rejected all %d points; keeping all", len(pts))
        return np.ones(len(pts), dtype=bool)

    return keep


def confidence_mask(conf: np.ndarray, percentile: float) -> np.ndarray:
    """
    Keep-mask for confidence strictly above a global percentile cutoff.

    - strict `>`, so a cutoff equal to the minimum still filters
    - uniform conf (nothing above the cutoff) keeps everything rather than deleting everything
    - shape-agnostic: (P,) point confidences on the pointcloud path, (N, H, W) maps on the mesh path

    Args:
        conf:       Confidence array of any shape.
        percentile: Cutoff percentile over all of `conf`.

    Returns:
        Bool array of `conf`'s shape; all-True if nothing clears the cutoff.
    """
    cutoff = np.percentile(conf, percentile)
    above = conf > cutoff
    return above if above.any() else np.ones(conf.shape, dtype=bool)


def subsample_points(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    conf: Optional[np.ndarray] = None,
    max_points: int = 50_000,
    conf_percentile: float = 20.0,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Confidence-filter then randomly cap a point set to a fixed budget.

    - exact point budget, unlike voxel downsampling whose count varies with scene extent
    - needed for scenes balanced across submaps
    - rng is seeded, so the cap is reproducible

    Args:
        points:          (P, 3) world points.
        colors:          (P, 3) colors, index-aligned with `points`, or None.
        conf:            (P,) per-point confidence, or None to skip filtering.
        max_points:      Point budget after filtering.
        conf_percentile: Cutoff passed to confidence_mask.

    Returns:
        (points, colors) index-aligned; colors is None if none was given.
    """
    # Drop points at/below the conf cutoff (see confidence_mask for the edge-case semantics)
    if conf is not None and len(conf) > 0:
        above = confidence_mask(conf, conf_percentile)
        points = points[above]
        colors = colors[above] if colors is not None else None

    # Random cap to the budget; seeded rng keeps results reproducible
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None

    return points, colors


########################################################
########## Geometry: plane fitting #####################
########################################################


def fit_dominant_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC floor plane, as the rigid transform that puts it at Z-up, z=0.

    - open3d segment_plane over the full cloud; largest inlier set is taken as the floor
    - no heuristic percentile

    Args:
        points: (N, 3) point cloud.

    Returns:
        R: (3, 3) rotation taking the floor normal onto [0, 0, 1].
        t: (3,) translation placing the floor at z=0 after that rotation.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    n_mag = np.linalg.norm([a, b, c])
    normal = np.array([a, b, c]) / n_mag
    d_norm = d / n_mag  # plane: normal · x + d_norm = 0; floor at z = -d_norm after rotation

    # Ensure normal points upward (positive Z component after alignment)
    if normal[2] < 0:
        normal = -normal
        d_norm = -d_norm

    R = rotation_align_vectors(normal, np.array([0.0, 0.0, 1.0]))
    # After R, floor is at z = -d_norm. Translate by d_norm to bring to z = 0.
    t = np.array([0.0, 0.0, d_norm])
    return R.astype(np.float64), t.astype(np.float64)


########################################################
########## Feature projection ##########################
########################################################


def _grid_sample_at_pixels(
    fmap: torch.Tensor,
    rows: np.ndarray,
    cols: np.ndarray,
    image_size: "tuple[int, int]",
) -> torch.Tensor:
    """Bilinear-sample fmap (D, H_p, W_p) at the given (rows, cols) in image_size frame.

    Returns (P_i, D) float32 tensor on CPU. Normalizes pixel centers to [-1, 1]
    using align_corners=False convention so the same coords work for any (H_p, W_p).
    """
    H, W = image_size
    gx = torch.from_numpy(((2 * cols.astype(np.float32) + 1) / W - 1)).float()
    gy = torch.from_numpy(((2 * rows.astype(np.float32) + 1) / H - 1)).float()
    # grid_sample wants (N, H_out, W_out, 2); treat P_i points as (1, 1, P_i, 2)
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        fmap.unsqueeze(0).float(),
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    # (1, D, 1, P_i) → (P_i, D)
    return sampled.squeeze(0).squeeze(1).T.cpu()


def _sample_at_source_pixels(
    feature_maps: "list[torch.Tensor]",
    pixel_indices: np.ndarray,
    image_size: "tuple[int, int]",
) -> torch.Tensor:
    """Source-frame-only lift: each point sampled at its (frame_id, row, col) only.

    Used as a fallback for points that have zero accumulated weight in multi-view
    aggregation (never visible / always depth-inconsistent).
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
        out[mask_i] = _grid_sample_at_pixels(fmap, rows, cols, image_size)
    return out


def lift_features(
    feature_maps: "list[torch.Tensor]",
    result: "FeedforwardResult",
    *,
    depth_tol: float = 0.05,
) -> torch.Tensor:
    """
    Multi-view confidence-weighted lift of dense feature maps to per-point features.

    - each point in result.points projects into every frame
    - masked by in-bounds + depth consistency (|z_proj - depth| / |z_proj| < depth_tol)
    - weighted by confidence at the projected pixel, then mean-reduced
    - points visible nowhere fall back to a source-frame sample at their pixel_indices entry

    Args:
        feature_maps: List of (D, H_p, W_p) per-frame dense features. Caller runs the
            extractor (and optional AE encode) first.
        result:       Carries points, pixel_indices, depth, extrinsics, intrinsics,
            model_height, model_width. `confidence` is optional — SfM results carry none
            and fall back to uniform per-pixel weights. Reload zarr with load_images=True
            before re-extracting so the extractor sees the depth map's FOV.
        depth_tol:    Relative depth tolerance for the visibility test.

    Returns:
        (P, D) float32 tensor of per-point features, aligned with result.points.
    """
    # Required fields — fail loud at function entry, not deep in the kernel
    for name in ("points", "pixel_indices", "depth", "extrinsics", "intrinsics"):
        assert getattr(result, name) is not None, (
            f"lift_features requires result.{name}; " f"load zarr with load_images=True or run pipeline fresh"
        )
    N = result.extrinsics.shape[0]
    assert len(feature_maps) == N, f"feature_maps count ({len(feature_maps)}) != frame count ({N})"

    H, W = result.model_height, result.model_width
    P = result.points.shape[0]
    D = feature_maps[0].shape[0]
    image_size = (H, W)

    # Run the whole kernel on the GPU in float32 (was CPU float64 numpy + CPU grid_sample —
    # the dominant cost on large clouds). Everything moves to the device once; one .cpu() at the end.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pts = torch.as_tensor(np.ascontiguousarray(result.points), dtype=torch.float32, device=device)
    pts_h = torch.cat([pts, torch.ones((P, 1), dtype=torch.float32, device=device)], dim=1)  # (P, 4)

    # Ensure extrinsics are (N, 4, 4); accept (N, 3, 4) by padding
    ext_np = result.extrinsics
    if ext_np.shape[-2:] == (3, 4):
        ext_np = extrinsics_to_homogeneous(ext_np)
    ext = torch.as_tensor(np.ascontiguousarray(ext_np), dtype=torch.float32, device=device)  # (N, 4, 4)
    intr = torch.as_tensor(np.ascontiguousarray(result.intrinsics), dtype=torch.float32, device=device)  # (N, 3, 3)

    # Conf → (N, H, W) float32 on device; absent confidence (SfM results) = uniform weights
    if result.confidence is None:
        conf = torch.ones((N, H, W), dtype=torch.float32, device=device)
    else:
        conf_np = (
            result.confidence.detach().cpu().numpy()
            if isinstance(result.confidence, torch.Tensor)
            else result.confidence
        )
        conf = torch.as_tensor(np.ascontiguousarray(conf_np), dtype=torch.float32, device=device)

    depth_np = result.depth
    if depth_np.ndim == 4:
        depth_np = depth_np[..., 0]
    depth = torch.as_tensor(np.ascontiguousarray(depth_np), dtype=torch.float32, device=device)

    features_sum = torch.zeros((P, D), dtype=torch.float32, device=device)
    weights_sum = torch.zeros((P,), dtype=torch.float32, device=device)

    pts_hT = pts_h.T  # (4, P) — reused every frame
    for i in range(N):
        fmap = feature_maps[i].to(device=device, dtype=torch.float32)  # (D, H_p, W_p)
        # Project all P points into frame i: world -> cam -> pixel
        proj = intr[i] @ (ext[i] @ pts_hT)[:3]  # (3, P)
        z = proj[2]
        safe_z = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
        u = proj[0] / safe_z
        v = proj[1] / safe_z

        # Visibility: in-bounds, in-front-of-camera, depth-consistent. nan/inf coords (degenerate
        # points) → 0 for indexing/sampling; in_bounds is False there so they contribute weight 0.
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        u_safe = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v_safe = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        u_idx = u_safe.clamp(0, W - 1).long()
        v_idx = v_safe.clamp(0, H - 1).long()
        z_depth = depth[i, v_idx, u_idx]
        depth_ok = (z - z_depth).abs() / (z.abs() + 1e-8) < depth_tol
        w = conf[i, v_idx, u_idx] * (in_bounds & depth_ok).float()  # (P,)

        # Bilinear sample feature map at projected coords (align_corners=False, border pad)
        gx = (2 * u_safe + 1) / W - 1
        gy = (2 * v_safe + 1) / H - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, P, 2)
        sampled = F.grid_sample(fmap.unsqueeze(0), grid, mode="bilinear", align_corners=False, padding_mode="border")
        sampled = sampled.squeeze(0).squeeze(1).T  # (P, D)

        features_sum += sampled * w.unsqueeze(-1)
        weights_sum += w

    # Weighted mean with eps for numerical safety
    features = features_sum / (weights_sum.unsqueeze(-1) + 1e-8)

    # Fallback: points with zero accumulated weight → source-frame sample
    zero_w = weights_sum < 1e-6
    if bool(zero_w.any()):
        zero_idx = zero_w.detach().cpu().numpy()
        fallback = _sample_at_source_pixels(feature_maps, result.pixel_indices[zero_idx], image_size)
        features[zero_w] = fallback.to(device)

    return features.detach().cpu()


def reproject_pixels(
    depth: np.ndarray,
    pixel_indices: np.ndarray,
    extrinsics_3x4: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    Reproject points to world space from stored source pixels and new poses.

    - use after BA instead of re-running unproject_and_filter_points
    - stored pixel_indices bypass the stochastic conf_mask subsampling, so the point set
      stays aligned with pre-BA features and colors

    Args:
        depth:          (N, H, W, 1) or (N, H, W) float32 depth maps.
        pixel_indices:  (P, 3) int32 — [frame_id, row, col] source pixel per point.
        extrinsics_3x4: (N, 3, 4) world-to-camera extrinsics, e.g. refined by BA.
        intrinsics:     (N, 3, 3) camera intrinsics.

    Returns:
        (P, 3) float32 world-space point positions.
    """
    fi = pixel_indices[:, 0]  # frame index per point  (P,)
    ri = pixel_indices[:, 1]  # pixel row per point    (P,)
    ci = pixel_indices[:, 2]  # pixel col per point    (P,)

    # depth at source pixel for each point
    if depth.ndim == 4:
        z = depth[fi, ri, ci, 0].astype(np.float64)
    else:
        z = depth[fi, ri, ci].astype(np.float64)

    # unproject source pixel to camera space with pinhole model
    fx = intrinsics[fi, 0, 0].astype(np.float64)
    fy = intrinsics[fi, 1, 1].astype(np.float64)
    cx = intrinsics[fi, 0, 2].astype(np.float64)
    cy = intrinsics[fi, 1, 2].astype(np.float64)
    x_cam = (ci - cx) * z / fx  # (P,)
    y_cam = (ri - cy) * z / fy  # (P,)

    # homogeneous camera-space coords: (P, 4)
    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)

    # build (N, 4, 4) world-to-cam and invert to cam-to-world
    N = extrinsics_3x4.shape[0]
    w2c = np.zeros((N, 4, 4), dtype=np.float64)
    w2c[:, :3, :] = extrinsics_3x4
    w2c[:, 3, 3] = 1.0
    cam2world = invert_poses(w2c)  # (N, 4, 4)

    # per-point transform: cam2world[fi] @ pts_cam[p]
    pts_world = np.einsum("pij,pj->pi", cam2world[fi], pts_cam)  # (P, 4)

    return pts_world[:, :3].astype(np.float32)


########################################################
########## Cross-frame attention utilities #############
########################################################


def cross_frame_attention_ratio(
    k: torch.Tensor,
    q: torch.Tensor,
    token_offset: int = 5,
) -> float:
    """
    How hard frame B attends to frame A, against A's own self-attention peak.

    - port of VGGT-SPARK get_similarity()
    - gates loop-closure candidates: high ratio = coherent overlapping geometry

    Args:
        k:            (B, heads, N_tokens, head_dim) key projections. N_tokens spans both
                      frames concatenated, so tokens_per_img = N_tokens // 2.
        q:            (B, heads, N_tokens, head_dim) query projections, same layout.
        token_offset: Patch tokens start here — skips the camera + register tokens that
                      precede them in VGGT-style models.

    Returns:
        Scalar in [0, inf): mean of the top-25% normalized cross-frame attention values
        (mean_top_quarter, matching VGGT-SPARK). >= 0.85 is the VGGT-SPARK acceptance
        threshold calibrated on VGGT-1B. 0.0 when token_offset >= tokens_per_img.
    """
    tokens_per_img = q.shape[2] // 2
    # Slice only the patch tokens from frame A (skip camera+register tokens)
    k_first = k[:, :, token_offset:tokens_per_img, :]
    if k_first.shape[2] == 0:
        return 0.0

    # Compute attention of all queries over first-frame patch keys
    attn = q @ k_first.transpose(-2, -1)  # (B, H, N_q, N_k_first)
    attn = attn.transpose(-2, -1)  # (B, H, N_k_first, N_q)
    attn = attn.softmax(dim=-1)
    attn = attn.mean(dim=1)  # (B, N_k_first, N_q) — avg over heads

    # Split queries by destination frame to separate self- vs cross-frame attention
    attn_to_first = attn[..., :tokens_per_img]  # first-frame self-attention
    attn_to_second = attn[..., tokens_per_img:]  # cross-frame attention to second

    # Ratio: how much cross-frame attention relative to self-attention peak
    max_self = attn_to_first.max(dim=-1)[0]  # (B, N_k_first)
    normalized = attn_to_second / (max_self.unsqueeze(-1) + 1e-8)
    ratio = normalized.max(dim=1)[0]  # (B, N_second)

    # Aggregate: mean of top-25% values, matching VGGT-SPARK mean_top_quarter()
    # - np.percentile(90) was used before and gives a lower scalar
    # - that dropped VGGT-X scores (~0.74) below the 0.85 threshold calibrated for VGGT-1B
    ratio_np = ratio.cpu().float().numpy().ravel()
    if ratio_np.size == 0:
        return 0.0
    thresh = float(np.percentile(ratio_np, 75))
    top_vals = ratio_np[ratio_np >= thresh]
    return float(top_vals.mean())
