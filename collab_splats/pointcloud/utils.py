"""
Point-set cleaning (outlier mask, confidence mask, random cap) and a loop-closure attention score.

- cross_frame_attention_ratio ports VGGT-SPARK get_similarity(); see docs/parity.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d
import torch

# Import only for type hints, to avoid a circular import with base.py
if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult

logger = logging.getLogger(__name__)


########################################################################
# Cleaning and subsampling
########################################################################


def outlier_mask(
    points: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    Statistical outlier keep-mask from open3d.

    - all-True when there are too few points, or when every point would be rejected

    Args:
        points: (P, 3) world points.
        nb_neighbors: neighbors open3d averages over per point.
        std_ratio: distance cutoff in standard deviations of that average.

    Returns:
        (P,) bool array, True for the points open3d keeps.
    """
    pts = np.asarray(points, dtype=np.float64)

    # Keep every point when there are too few for outlier removal
    if len(pts) <= nb_neighbors:
        return np.ones(len(pts), dtype=bool)

    # Remove statistical outliers and mark the points that stay
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _, keep_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    keep = np.zeros(len(pts), dtype=bool)
    keep[np.asarray(keep_idx, dtype=int)] = True

    # If nothing survived, keep every point instead
    if not keep.any():
        logger.warning("outlier_mask: outlier removal rejected all %d points; keeping all", len(pts))

        return np.ones(len(pts), dtype=bool)

    return keep


def clean_pointcloud(result: PointcloudResult, *, remove_outliers: bool, max_points: int) -> PointcloudResult:
    """
    Optional outlier removal, then a random cap on the point count.

    - one keep mask selects every per-point array, so they stay row-aligned

    Args:
        result: the cloud to clean.
        remove_outliers: run outlier_mask first.
        max_points: point cap, drawn after outlier removal.

    Returns:
        A new PointcloudResult with the kept points.
    """
    keep = outlier_mask(result.points) if remove_outliers else np.ones(len(result.points), dtype=bool)
    keep = subsample_points(keep, max_points)

    return result.select_points(keep)


def confidence_mask(conf: np.ndarray, percentile: float) -> np.ndarray:
    """
    Keep-mask for confidence strictly above a percentile of all values.

    - strict `>`, so a cutoff equal to the minimum still filters
    - when nothing is above the cutoff, e.g. uniform confidence, keeps values `>=` it
    - any shape: (P,) point confidences or (N, H, W) maps

    Args:
        conf: confidence array of any shape.
        percentile: cutoff percentile over all of `conf`.

    Returns:
        Bool array of `conf`'s shape; never all-False.
    """
    cutoff = np.percentile(conf, percentile)
    above = conf > cutoff

    if above.any():
        return above

    logger.warning("confidence_mask: nothing above p%.1f — keeping the pixels at the cutoff", percentile)

    return conf >= cutoff


def subsample_points(mask: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """
    Keep at most max_points of a mask's True entries, drawn at random.

    - exact point count, unlike voxel downsampling
    - seeded private rng: reproducible, global numpy rng untouched

    Args:
        mask: bool keep-mask, any shape.
        max_points: cap on True entries.
        seed: rng seed.

    Returns:
        Bool mask of `mask`'s shape; `mask` itself when already within the cap.
    """
    idx = np.flatnonzero(mask)

    if idx.size <= max_points:
        return mask

    # Randomly pick max_points of the kept points
    rng = np.random.default_rng(seed)
    drawn = rng.choice(idx, size=max_points, replace=False)
    keep = np.zeros(mask.size, bool)
    keep[drawn] = True

    return keep.reshape(mask.shape)


########################################################################
# Cross-frame attention utilities
########################################################################


def cross_frame_attention_ratio(
    k: torch.Tensor,
    q: torch.Tensor,
    *,
    token_offset: int,
) -> np.ndarray:
    """
    Cross-frame attention between two frames, relative to the self-attention peak.

    - per frame-B token: best frame-A key's attention to it, over that key's self peak
    - high ratios mean overlapping geometry; used to gate loop-closure candidates
    - upstream's scalar score is the mean of ratios at or above the 75th percentile

    Args:
        k: (B, heads, N_tokens, head_dim) keys; frame A's tokens, then frame B's, half each.
        q: (B, heads, N_tokens, head_dim) queries, same layout.
        token_offset: index of the first patch token, after the camera and register tokens.

    Returns:
        Flattened (B, N_tokens / 2) ratios, one per frame-B token.

    Raises:
        ValueError: token_offset leaves no patch tokens in frame A.
    """
    tokens_per_img = q.shape[2] // 2

    # Take the patch keys of the first frame
    k_first = k[:, :, token_offset:tokens_per_img, :]

    if k_first.shape[2] == 0:
        raise ValueError(f"token_offset {token_offset} leaves no patch tokens (tokens_per_img {tokens_per_img})")

    # Compute attention from each first-frame key to all queries, averaged over heads
    attn = q @ k_first.transpose(-2, -1)  # (B, H, N_q, N_k_first)
    attn = attn.transpose(-2, -1)  # (B, H, N_k_first, N_q)
    attn = attn.softmax(dim=-1)
    attn = attn.mean(dim=1)  # (B, N_k_first, N_q) — avg over heads

    # Split the attention into the first frame and the second frame
    attn_to_first = attn[..., :tokens_per_img]  # first-frame self-attention
    attn_to_second = attn[..., tokens_per_img:]  # cross-frame attention to second

    # Score each second-frame token by its cross attention relative to the self-attention peak
    max_self = attn_to_first.max(dim=-1)[0]  # (B, N_k_first)
    normalized = attn_to_second / (max_self.unsqueeze(-1) + 1e-8)
    ratio = normalized.max(dim=1)[0]  # (B, N_second)

    return ratio.cpu().float().numpy().ravel()
