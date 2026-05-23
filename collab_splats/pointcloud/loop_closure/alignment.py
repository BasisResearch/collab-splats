"""Overlap-region alignment utilities for loop closure.

Pure numpy module — no gtsam, no torch.
Populated across spec 2 (dedup_overlap) and spec 3 (umeyama_se3, overlap_region_align).
"""
from __future__ import annotations

import numpy as np


def dedup_overlap(
    submap_ids: list[int],
    submap_starts: list[int],
    corrected: dict[int, np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """Reconstruct (total_frames, 4, 4) from per-submap corrected poses, deduplicating overlap.

    Overlap frames (shared by consecutive submaps) are assigned to the earlier submap
    (canonical-owner rule). Later submaps' overlapping frames are silently dropped.

    Args:
        submap_ids: list of submap IDs in iteration order.
        submap_starts: global frame index of each submap's first frame.
        corrected: dict mapping submap_id → (K_i, 4, 4) corrected pose array.
        total_frames: N, the total number of input frames.

    Returns:
        (total_frames, 4, 4) float32 array.
    """
    out = np.zeros((total_frames, 4, 4), dtype=np.float32)
    assigned = np.zeros(total_frames, dtype=bool)
    for sid, start in zip(submap_ids, submap_starts):
        poses = corrected[sid]  # (K_i, 4, 4)
        k = poses.shape[0]
        for local_i in range(k):
            global_i = start + local_i
            if 0 <= global_i < total_frames and not assigned[global_i]:
                out[global_i] = poses[local_i]
                assigned[global_i] = True
    return out


def umeyama_se3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form SE(3) alignment via SVD (no scale).

    Args:
        source: (M, 3) points in source frame.
        target: (M, 3) corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (4, 4) float32 homogeneous T such that target ≈ T @ source.
    """
    M = source.shape[0]
    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        return np.eye(4, dtype=np.float32)
    w = w / w_sum

    src = source.astype(np.float64)
    tgt = target.astype(np.float64)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)

    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    H = (src_c * w[:, None]).T @ tgt_c  # (3, 3) weighted cross-covariance

    U, _, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_tgt - R @ mu_src

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def umeyama_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3) alignment via Umeyama (with scale).

    Args:
        source: (M, 3) float32/64 points in source frame.
        target: (M, 3) float32/64 corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (s, R, t): float scale, (3,3) float32 rotation, (3,) float32 translation
                   such that target ≈ s * R @ source + t.
    """
    M = source.shape[0]
    if M < 3:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    w = w / w_sum

    src = source.astype(np.float64)
    tgt = target.astype(np.float64)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)

    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    # RMS norms for scale estimation
    scale_src = float(np.sqrt((w * (src_c ** 2).sum(axis=1)).sum()))
    scale_tgt = float(np.sqrt((w * (tgt_c ** 2).sum(axis=1)).sum()))
    s = scale_tgt / scale_src if scale_src > 1e-9 else 1.0

    # Rotation via SVD of cross-covariance
    H = (src_c * s * w[:, None]).T @ tgt_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)

    t = (mu_tgt - s * R @ mu_src).astype(np.float32)
    return s, R, t


def overlap_region_align(submap_a, submap_b, overlap_frames: int) -> np.ndarray:
    """SE(3) transform T s.t. pts_a_world ≈ T @ pts_b_local.

    Uses last `overlap_frames` of submap_a and first `overlap_frames` of submap_b.
    Returns identity if either submap has no world_points.
    """
    if submap_a.world_points is None or submap_b.world_points is None:
        return np.eye(4, dtype=np.float32)

    O = min(overlap_frames, submap_a.world_points.shape[0], submap_b.world_points.shape[0])
    if O == 0:
        return np.eye(4, dtype=np.float32)

    pts_a = submap_a.world_points[-O:].reshape(-1, 3).astype(np.float64)
    pts_b = submap_b.world_points[:O].reshape(-1, 3).astype(np.float64)

    if submap_a.world_points_conf is not None and submap_b.world_points_conf is not None:
        w = ((submap_a.world_points_conf[-O:] + submap_b.world_points_conf[:O]) / 2).reshape(-1)
    elif submap_a.world_points_conf is not None:
        w = submap_a.world_points_conf[-O:].reshape(-1)
    elif submap_b.world_points_conf is not None:
        w = submap_b.world_points_conf[:O].reshape(-1)
    else:
        w = None

    return umeyama_se3(pts_b, pts_a, weights=w)


def overlap_region_align_sim3(submap_a, submap_b, overlap_frames: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) transform (s, R, t) s.t. pts_a_world ≈ s * R @ pts_b_local + t.

    Uses last `overlap_frames` of submap_a and first `overlap_frames` of submap_b.
    Returns (1.0, eye(3), zeros(3)) if either submap has no world_points.
    """
    if submap_a.world_points is None or submap_b.world_points is None:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    O = min(overlap_frames, submap_a.world_points.shape[0], submap_b.world_points.shape[0])
    if O == 0:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    pts_a = submap_a.world_points[-O:].reshape(-1, 3).astype(np.float64)
    pts_b = submap_b.world_points[:O].reshape(-1, 3).astype(np.float64)

    if submap_a.world_points_conf is not None and submap_b.world_points_conf is not None:
        w = ((submap_a.world_points_conf[-O:] + submap_b.world_points_conf[:O]) / 2).reshape(-1)
    elif submap_a.world_points_conf is not None:
        w = submap_a.world_points_conf[-O:].reshape(-1)
    elif submap_b.world_points_conf is not None:
        w = submap_b.world_points_conf[:O].reshape(-1)
    else:
        w = None

    return umeyama_sim3(pts_b, pts_a, weights=w)
