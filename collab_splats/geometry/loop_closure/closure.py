"""Loop closure orchestration utilities.

Houses pre-add gates, pose-graph build/dedup, and submap-output merging.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyR

from .graph import PoseGraph as _SL4PoseGraph
from .graph import decompose_camera, estimate_scale_pairwise
from .submap import Submap

log = logging.getLogger(__name__)

# Minimum points required for a reliable median scale estimate (matches VGGT-SLAM fallback threshold)
_MIN_CONF_POINTS = 100

_RNG = np.random.default_rng(42)


def _estimate_scale_pairwise_dist(X: np.ndarray, Y: np.ndarray) -> float:
    """Pairwise distance scale estimator — invariant to coordinate origin.

    median(||Y_i - Y_j|| / ||X_i - X_j||). Translation cancels in subtraction,
    so result is unbiased regardless of which camera frame X/Y are expressed in.
    """
    if X.shape[0] < 2:
        return 1.0
    n = min(X.shape[0], 500)
    idx = _RNG.choice(X.shape[0], (n, 2), replace=True)
    i, j = idx[:, 0], idx[:, 1]
    x_dists = np.linalg.norm(X[i] - X[j], axis=1)
    y_dists = np.linalg.norm(Y[i] - Y[j], axis=1)
    valid = x_dists > 1e-6
    return float(np.median(y_dists[valid] / x_dists[valid])) if valid.any() else 1.0


########################################
####### Absorbed from alignment.py #####
########################################


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
    H = (src_c * w[:, None]).T @ tgt_c
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

    scale_src = float(np.sqrt((w * (src_c**2).sum(axis=1)).sum()))
    scale_tgt = float(np.sqrt((w * (tgt_c**2).sum(axis=1)).sum()))
    s = scale_tgt / scale_src if scale_src > 1e-9 else 1.0

    H = (src_c * s * w[:, None]).T @ tgt_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)

    t = (mu_tgt - s * R @ mu_src).astype(np.float32)
    return s, R, t


########################################
####### Absorbed from retrieval.py #####
########################################


@dataclass
class LoopMatch:
    """Loop closure candidate produced by DINO-SALAD retrieval."""

    similarity_score: float
    query_submap_id: int
    detected_submap_id: int
    query_frame_idx: int
    detected_frame_idx: int
    accepted: bool = False
    # None on accepted matches; "verify_ratio" | "no_joint_poses" | "jump_ratio" on rejects.
    reject_reason: str | None = None


@dataclass
class LoopClosureConfig:
    submap_size: int = 20
    submap_overlap: int = 1  # 1 = VGGT-SLAM parity; 4 = old default
    # DINO-SALAD retrieval gate: accept candidate if L2(q, ref) < lc_retrieval_threshold.
    # L2 distance on unit-norm DINO-SALAD embeddings (range [0, 2]; typical good matches < 0.5).
    # 0.95 matches VGGT-SLAM main.py default (lc_thres=0.95). 0.0 disables retrieval.
    lc_retrieval_threshold: float = 0.95
    max_loops_per_submap: int = 5
    # None → resolve to the creator's default_verify_match_ratio at LoopClosure
    # wrapper init (fallback 0.85); an explicit float always wins.
    verify_match_ratio: float | None = None
    nms_frame_distance: int = 25
    min_submap_gap: int = 1
    # Inter-submap scale estimation method.
    # "rotation_only" — VGGT-SLAM default: T[:3,:3] applied to curr_pts (rotation only)
    # "se3"           — full SE3 T applied before norm ratio
    # "pairwise_dist" — pairwise distance ratio, translation-invariant
    # "none"          — skip scale estimation entirely; always use scale=1.0
    scale_method: Literal["se3", "rotation_only", "pairwise_dist", "none"] = "rotation_only"
    max_jump_ratio: float = math.inf  # reject loops where ‖ΔT.t‖/path_length > this; math.inf disables
    conf_threshold: float = 25.0  # confidence gate for scale estimation; matches VGGT-SLAM --conf_threshold 25

    @property
    def lc_threshold_l2(self) -> float:
        """L2 threshold passed to find_loop_closures. Alias for lc_retrieval_threshold."""
        return self.lc_retrieval_threshold


class LoopMatchQueue:
    """Max-heap keeping top-k lowest-distance LoopMatch candidates, with NMS."""

    def __init__(self, max_size: int, nms_frame_distance: int = 0) -> None:
        self._max_size = max_size
        self._nms = nms_frame_distance
        self._counter: int = 0
        self._heap: list = []

    def push(self, match: LoopMatch) -> None:
        heapq.heappush(self._heap, (-match.similarity_score, self._counter, match))
        self._counter += 1
        if len(self._heap) > self._max_size:
            heapq.heappop(self._heap)

    def get_matches(self) -> list[LoopMatch]:
        candidates = sorted([m for _, _, m in self._heap], key=lambda m: m.similarity_score)
        if self._nms <= 0:
            return candidates
        accepted: list[LoopMatch] = []
        for cand in candidates:
            suppressed = any(
                acc.detected_submap_id == cand.detected_submap_id
                and abs(acc.detected_frame_idx - cand.detected_frame_idx) < self._nms
                for acc in accepted
            )
            if not suppressed:
                accepted.append(cand)
        return accepted


def find_loop_closures(
    query_submap: "Submap",
    past_submaps: list["Submap"],
    lc_threshold: float,
    max_loops: int,
    nms_frame_distance: int = 0,
) -> list[LoopMatch]:
    """Return top-k loop closure candidates using pre-computed retrieval_vectors."""
    if not past_submaps:
        return []
    queue = LoopMatchQueue(max_size=max_loops, nms_frame_distance=nms_frame_distance)
    # For each query frame, find the nearest-neighbor frame in each past submap by
    # L2 distance over DINO-SALAD retrieval vectors; keep it if under lc_threshold.
    for q_idx in range(query_submap.retrieval_vectors.shape[0]):
        q_vec = query_submap.retrieval_vectors[q_idx]
        for past in past_submaps:
            dists = torch.cdist(q_vec.unsqueeze(0), past.retrieval_vectors).squeeze(0)
            best_idx = int(dists.argmin())
            best_dist = float(dists[best_idx])
            if best_dist < lc_threshold:
                queue.push(
                    LoopMatch(
                        similarity_score=best_dist,
                        query_submap_id=query_submap.submap_id,
                        detected_submap_id=past.submap_id,
                        query_frame_idx=q_idx,
                        detected_frame_idx=best_idx,
                    )
                )
    return queue.get_matches()


########################################
####### Pose merging & loop-jump utilities #####
########################################


def dedup_overlap(  # noqa: F811 — shadows import; canonical copy lives here
    submap_ids: list[int],
    submap_starts: list[int],
    corrected: dict[int, np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """Reconstruct (total_frames, 4, 4) from per-submap corrected poses, deduplicating overlap.

    First-writer-wins: the overlap frame belongs to two adjacent submaps; we keep
    submap-0's estimate (processed first). VGGT-SLAM does NOT dedup — its
    write_poses_to_file (map.py:142-162) emits every submap's frames, so the
    shared overlap frame appears twice in its TUM (a duplicate timestamp). evo
    associates by timestamp and keeps the first occurrence — also submap-0's — so
    the two pipelines agree on the boundary pose despite SLAM's duplicate row.
    """
    out = np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))
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


def translation_jump_check(
    submaps: list[Submap],
    query_idx: int,
    query_frame: int,
    detected_idx: int,
    detected_frame: int,
    lc_relative_pose: np.ndarray,
    max_jump_ratio: float = 0.2,
) -> tuple[bool, float]:
    """Reject loop if implied correction translation exceeds fraction of odom-path length.

    SE(3) port of MR.ScaleMaster's anchor-scale alarm: rejects loop closures whose
    proposed correction is incompatible with the accumulated odometry path between
    the two endpoints.

    Args:
        submaps: ordered list of normal submaps (not LC submaps).
        query_idx: index in `submaps` of the query submap.
        query_frame: local frame index inside the query submap.
        detected_idx: index in `submaps` of the detected (older) submap.
        detected_frame: local frame index inside the detected submap.
        lc_relative_pose: (4, 4) SE(3) relative pose detected→query (from LC).
        max_jump_ratio: reject if ‖ΔT.t‖ / path_length > this.

    Returns:
        (accept, ratio).
    """
    if query_idx == detected_idx and query_frame == detected_frame:
        return True, 0.0

    # Walk from detected → query along odom path, summing ‖t_step‖.
    path_length = 0.0
    pose_curr = submaps[detected_idx].poses[detected_frame]  # (4, 4) world-to-cam

    if detected_idx == query_idx:
        a, b = sorted([detected_frame, query_frame])
        for f in range(a, b):
            t_step = submaps[detected_idx].poses[f + 1][:3, 3] - submaps[detected_idx].poses[f][:3, 3]
            path_length += float(np.linalg.norm(t_step))
        pose_at_query = submaps[query_idx].poses[query_frame]
    else:
        # detected_idx → end of detected submap
        for f in range(detected_frame, submaps[detected_idx].poses.shape[0] - 1):
            t_step = submaps[detected_idx].poses[f + 1][:3, 3] - submaps[detected_idx].poses[f][:3, 3]
            path_length += float(np.linalg.norm(t_step))
        # Sum intra-submap path for intermediate submaps.
        # Cross-submap boundary jumps are skipped: each submap's poses are in that
        # submap's local frame (frame 0 = identity), so submap[si].poses[-1] and
        # submap[si+1].poses[0] are in different coordinate systems. Their difference
        # is geometrically meaningless. path_length is a lower bound; the check
        # is slightly lenient at boundaries, which is acceptable.
        for si in range(detected_idx + 1, query_idx):
            for f in range(submaps[si].poses.shape[0] - 1):
                t_step = submaps[si].poses[f + 1][:3, 3] - submaps[si].poses[f][:3, 3]
                path_length += float(np.linalg.norm(t_step))
        # start of query submap → query_frame
        for f in range(0, query_frame):
            t_step = submaps[query_idx].poses[f + 1][:3, 3] - submaps[query_idx].poses[f][:3, 3]
            path_length += float(np.linalg.norm(t_step))
        pose_at_query = submaps[query_idx].poses[query_frame]

    # Odom-implied relative pose detected→query
    pose_det = submaps[detected_idx].poses[detected_frame]
    T_odom = np.linalg.inv(pose_det.astype(np.float64)) @ pose_at_query.astype(np.float64)

    # ΔT = lc · inv(odom)
    delta_T = lc_relative_pose.astype(np.float64) @ np.linalg.inv(T_odom)
    delta_t_norm = float(np.linalg.norm(delta_T[:3, 3]))

    eps = 1e-6
    ratio = delta_t_norm / max(path_length, eps)
    return ratio < max_jump_ratio, ratio


########################################
####### SL(4) pose graph optimization ##
########################################


def _cam_local_points(pts: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points by a 4x4 world-to-cam pose into camera-local coords."""
    h = np.hstack([pts.astype(np.float64), np.ones((pts.shape[0], 1))])
    return (w2c.astype(np.float64) @ h.T).T[:, :3]


def _lc_anchor_scale(
    curr_submap: Submap,
    curr_idx: int,
    prior_submap: Submap,
    prior_idx: int,
    conf_threshold: float,
    scale_method: str,
) -> float | None:
    """Pixel-aligned anchor scale between an LC frame and its identical regular frame.

    Mirrors VGGT-SLAM solver.py:129-151: both frames are the SAME image, so their
    point grids pair pixel-for-pixel. Returns median(||prior_cam|| / ||curr_cam||)
    per `scale_method` (the factor scaling curr-submap units into prior-submap
    units), or None when points are unavailable or the grids cannot be aligned.
    """
    if scale_method == "none":
        return 1.0
    if curr_submap.world_points is None or prior_submap.world_points is None:
        return None

    # Resolution alignment: LC submaps carry full-res (H·W) grids while regular
    # submaps carry subsample-strided grids (arange(0, W, 8) × arange(0, H, 8),
    # row-major over (v, u) — see _raw_to_world_points). Resample the LC side
    # onto the regular grid so points pair pixel-for-pixel.
    lc_side, reg_side = (curr_submap, prior_submap) if curr_submap.is_lc_submap else (prior_submap, curr_submap)
    lc_flat_idx = None
    if lc_side.world_points.shape[1] != reg_side.world_points.shape[1]:
        if lc_side.frames is None:
            return None
        h_img, w_img = int(lc_side.frames.shape[-2]), int(lc_side.frames.shape[-1])
        if lc_side.world_points.shape[1] != h_img * w_img:
            return None
        us = np.arange(0, w_img, 8)  # regular submaps use subsample stride 8 (_raw_to_world_points)
        vs = np.arange(0, h_img, 8)
        if reg_side.world_points.shape[1] != len(us) * len(vs):
            return None
        uu, vv = np.meshgrid(us, vs)
        lc_flat_idx = (vv * w_img + uu).ravel()

    # Gather paired points + confs, resampling the LC side where needed
    def _frame_data(submap: Submap, idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        pts = submap.world_points[idx].astype(np.float64)
        conf = None
        if submap.world_points_conf is not None and submap.world_points_conf.shape[:2] == submap.world_points.shape[:2]:
            conf = submap.world_points_conf[idx].astype(np.float64)
        if submap is lc_side and lc_flat_idx is not None:
            pts = pts[lc_flat_idx]
            conf = conf[lc_flat_idx] if conf is not None else None
        return pts, conf

    curr_pts, curr_conf = _frame_data(curr_submap, curr_idx)
    prior_pts, prior_conf = _frame_data(prior_submap, prior_idx)

    # Back-transform both sides to their frame's camera-local coords so norms share
    # an origin (same convention as the sequential-edge prev_pts back-transform).
    curr_cam = _cam_local_points(curr_pts, curr_submap.poses[curr_idx])
    prior_cam = _cam_local_points(prior_pts, prior_submap.poses[prior_idx])

    # Confidence fallback chain (VGGT-SLAM solver.py:132-143): joint > thr,
    # else prior > thr, else prior > 0; missing conf side treated as pass-all.
    n = curr_cam.shape[0]
    mask = np.ones(n, dtype=bool)
    if prior_conf is not None:
        joint = prior_conf > conf_threshold
        if curr_conf is not None:
            joint = joint & (curr_conf > conf_threshold)
        if joint.sum() >= _MIN_CONF_POINTS:
            mask = joint
        elif (prior_conf > conf_threshold).sum() >= _MIN_CONF_POINTS:
            mask = prior_conf > conf_threshold
        else:
            mask = prior_conf > 0

    # T = inv(K_prior) @ K_curr (identity for a shared camera) — mirror the
    # sequential-edge scale_method conventions.
    T = np.eye(4)
    T[:3, :3] = np.linalg.inv(prior_submap.intrinsics[prior_idx].astype(np.float64)) @ curr_submap.intrinsics[
        curr_idx
    ].astype(np.float64)
    if scale_method == "rotation_only":
        curr_in_prior = (T[:3, :3] @ curr_cam.T).T
    else:
        curr_in_prior = _cam_local_points(curr_cam, T)

    if scale_method == "pairwise_dist":
        s = _estimate_scale_pairwise_dist(curr_in_prior[mask], prior_cam[mask])
    else:
        s = estimate_scale_pairwise(curr_in_prior[mask], prior_cam[mask])

    # Guard against degenerate estimates: depth-unprojected LC points can contain
    # NaN/inf (invalid pixels), letting a non-finite or ≤0 median through — which
    # would produce a singular diag(s,s,s,1) between-factor. Fall back to the
    # None path (scale 1.0 + once-per-loop warning at the call site).
    if not np.isfinite(s) or s <= 0:
        return None
    return s


def _loop_chain_relatives(
    P_lc0: np.ndarray,
    P_lc1: np.ndarray,
    s_a: float,
    s_b: float,
    K_q: np.ndarray | None = None,
    K_lc0: np.ndarray | None = None,
    K_lc1: np.ndarray | None = None,
    K_d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three loop-chain relatives in the graph's H_inner convention (H_j = H_i @ M).

    Anchor A (query → LC0, identical image): pure scale fold s_a (LC units →
    query-submap units) plus the K change; anchor B (LC1 → detected) symmetric
    with s_b (detected units → LC units); inner edge carries the LC relative
    P_lc0 @ inv(P_lc1). Identity anchors compose to P_lc0 @ inv(P_lc1), and for
    an LC run at k× scale s_a=1/k, s_b=k cancel to the metric relative
    (VGGT-SLAM solver.py:118-170 chain).
    """
    eye = np.eye(4, dtype=np.float64)
    K_q = eye if K_q is None else K_q
    K_lc0 = eye if K_lc0 is None else K_lc0
    K_lc1 = eye if K_lc1 is None else K_lc1
    K_d = eye if K_d is None else K_d
    # Anchor A: query → LC-frame-0 (identical image) — K change + scale fold s_a.
    H_rel_a = np.linalg.inv(K_q) @ K_lc0 @ np.diag([s_a, s_a, s_a, 1.0])
    # Inner edge: the LC pair's own relative pose, no K change.
    H_inner = P_lc0.astype(np.float64) @ np.linalg.inv(P_lc1.astype(np.float64))
    # Anchor B: LC-frame-1 → detected (identical image) — K change + scale fold s_b.
    H_rel_b = np.linalg.inv(K_lc1) @ K_d @ np.diag([s_b, s_b, s_b, 1.0])
    return H_rel_a, H_inner, H_rel_b


def run_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
    conf_threshold: float = 25.0,
    scale_method: Literal["se3", "rotation_only", "pairwise_dist", "none"] = "se3",
    debug_out: list | None = None,
) -> np.ndarray:
    """Build + optimize per-frame SL(4) pose graph; return (total_frames, 4, 4).

    Per-frame node building mirrors vggt_slam/solver.py:add_edge:
    - Inner frames: H_inner = poses[i-1] @ inv(poses[i]); node chained from prev
    - Inter-submap first frame: scale estimated via estimate_scale_pairwise on
      overlapping world_points, H_w = graph.get_homography(overlap_prev) @ T @ H_scale
      where T = inv(P_prev_ov) @ P_curr_ov (full w2c poses, not K-only)
    - Loop edges from lc_submaps (2-frame submaps with verified LC poses):
      scale-reconciled 3-edge chain through two graph-only LC nodes
      (anchor A, inner LC relative, anchor B — VGGT-SLAM solver.py:262-295)
    """
    if not submaps:
        return np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))

    pg = _SL4PoseGraph()
    global_node_id = 0
    frame_to_node: dict[tuple[int, int], int] = {}
    submap_node_ids: dict[int, list[int]] = {}

    for s_idx, submap in enumerate(submaps):
        k = submap.poses.shape[0]
        node_ids_this: list[int] = []

        for local_i in range(k):
            nid = global_node_id + local_i
            frame_to_node[(submap.submap_id, local_i)] = nid
            node_ids_this.append(nid)

        # Build K_4x4 from 3×3 intrinsics — matches VGGT-SLAM's proj_mats = K_4x4.
        # SLAM uses K_4x4 for T computation (inv(K_prev)@K_curr = I for same camera)
        # and for pose extraction (K @ inv(H_opt) → decompose_camera cancels K).
        K_4x4 = np.tile(np.eye(4, dtype=np.float64), (k, 1, 1))
        K_4x4[:, :3, :3] = submap.intrinsics.astype(np.float64)

        if s_idx == 0:
            # First node = I, prior = I — matches VGGT-SLAM add_homography(0, I) + add_prior_factor(0, I)
            pg.add_node(node_ids_this[0], np.eye(4))
            pg.add_prior(node_ids_this[0], np.eye(4))
            for local_i in range(1, k):
                H_inner = submap.poses[local_i - 1].astype(np.float64) @ np.linalg.inv(
                    submap.poses[local_i].astype(np.float64)
                )
                prev_H = pg.get_homography(node_ids_this[local_i - 1])
                pg.add_node(node_ids_this[local_i], prev_H @ H_inner)
                pg.add_sequential_edge(node_ids_this[local_i - 1], node_ids_this[local_i], H_inner)
        else:
            prev_submap = submaps[s_idx - 1]
            prev_K_4x4 = np.tile(np.eye(4, dtype=np.float64), (len(prev_submap.poses), 1, 1))
            prev_K_4x4[:, :3, :3] = prev_submap.intrinsics.astype(np.float64)
            O = min(overlap_frames, k, len(prev_submap.poses))

            # T = inv(K_prev[-1]) @ K_curr[0] — same as VGGT-SLAM's proj_mats-based T.
            # For fixed camera (K_prev = K_curr) this equals I exactly.
            T = np.linalg.inv(prev_K_4x4[-1]) @ K_4x4[0]

            scale = 1.0
            if submap.world_points is not None and prev_submap.world_points is not None and O > 0:
                # curr_pts: world_points[0] = cam_pts[0] (W2C[0]=I for VGGT) — same as SLAM t1.
                curr_pts = submap.world_points[:O].reshape(-1, 3).astype(np.float64)

                # prev_pts: SLAM uses camera-local depth (pointclouds[K-1], camera frame of last frame).
                # Our world_points[-1] = C2W[-1] @ cam_pts[-1] (cam0-frame, not camera-local).
                # Back-transform: W2C[-1] @ world_pts[-1] = cam_pts[-1] (camera-local), matching SLAM.
                prev_wps = prev_submap.world_points[-O:]  # (O, P, 3)
                P_per = prev_wps.shape[1]
                prev_cam_list = []
                for oi in range(O):
                    W2C = prev_submap.poses[-O + oi].astype(np.float64)
                    wh = np.hstack([prev_wps[oi], np.ones((P_per, 1), dtype=np.float64)])
                    prev_cam_list.append((W2C @ wh.T).T[:, :3])
                prev_pts = np.concatenate(prev_cam_list, axis=0)

                n = curr_pts.shape[0]

                # Confidence filtering: match VGGT-SLAM solver.py:132-143
                mask = np.ones(n, dtype=bool)
                if (
                    submap.world_points_conf is not None
                    and prev_submap.world_points_conf is not None
                    and submap.world_points_conf.shape[:2] == submap.world_points.shape[:2]
                    and prev_submap.world_points_conf.shape[:2] == prev_submap.world_points.shape[:2]
                ):
                    curr_conf = submap.world_points_conf[:O].reshape(-1)
                    prev_conf = prev_submap.world_points_conf[-O:].reshape(-1)
                    joint_mask = (curr_conf > conf_threshold) & (prev_conf > conf_threshold)
                    if joint_mask.sum() >= _MIN_CONF_POINTS:
                        mask = joint_mask
                    else:
                        # VGGT-SLAM fallback: prior_conf > thresh only (not OR mask).
                        prior_only_mask = prev_conf > conf_threshold
                        if prior_only_mask.sum() >= _MIN_CONF_POINTS:
                            mask = prior_only_mask

                curr_h = np.hstack([curr_pts, np.ones((n, 1))])
                if scale_method == "none":
                    # No scale estimation — use identity scale (scale=1.0).
                    scale = 1.0
                else:
                    if scale_method == "rotation_only":
                        # VGGT-SLAM style: apply only rotation part of T, drop translation.
                        # norm(R@X) == norm(X), so effectively uses world-frame norms.
                        curr_in_prev = (T[:3, :3] @ curr_pts.T).T
                    else:
                        # "se3" (default): full SE3 — translation shifts anchor, introduces bias
                        curr_in_prev = (T @ curr_h.T).T[:, :3]

                    if scale_method == "pairwise_dist":
                        scale = _estimate_scale_pairwise_dist(curr_in_prev[mask], prev_pts[mask])
                    else:
                        scale = estimate_scale_pairwise(curr_in_prev[mask], prev_pts[mask])

            # H_w = H_overlap @ T @ H_scale — matches VGGT-SLAM solver.py:161-162
            # (H_overlap_prev_node @ inv(K_prev) @ K_curr @ H_scale). Note SLAM's
            # submap.proj_mats stores K (the param is misleadingly named
            # intrinsics_inv but solver.py:256 passes K_4x4), so its
            # inv(proj_mats[-1]) @ proj_mats[0] == our inv(K_prev) @ K_curr == T.
            H_scale = np.diag([scale, scale, scale, 1.0])
            prev_submap_last_nid = submap_node_ids[prev_submap.submap_id][-1]
            H_overlap = pg.get_homography(prev_submap_last_nid)
            H_w = H_overlap @ T @ H_scale
            pg.add_node(node_ids_this[0], H_w)

            if debug_out is not None:
                debug_out.append(
                    {
                        "submap_id": submap.submap_id,
                        "T": T.copy(),
                        "scale": float(scale),
                        "H_w": H_w.copy(),
                        "H_overlap": H_overlap.copy(),
                    }
                )

            H_rel_inter = np.linalg.inv(pg.get_homography(prev_submap_last_nid)) @ H_w
            pg.add_sequential_edge(prev_submap_last_nid, node_ids_this[0], H_rel_inter)

            for local_i in range(1, k):
                H_inner = submap.poses[local_i - 1].astype(np.float64) @ np.linalg.inv(
                    submap.poses[local_i].astype(np.float64)
                )
                prev_H = pg.get_homography(node_ids_this[local_i - 1])
                pg.add_node(node_ids_this[local_i], prev_H @ H_inner)
                pg.add_sequential_edge(node_ids_this[local_i - 1], node_ids_this[local_i], H_inner)

        submap_node_ids[submap.submap_id] = node_ids_this
        global_node_id += k
        # Incremental optimization after each submap — matches VGGT-SLAM runner which calls
        # solver.graph.optimize() inside the per-submap loop before processing the next submap.
        # This gives better H_overlap values for the next submap's H_w initialization.
        pg.optimize()

    # Loop edges: VGGT-SLAM 3-constraint chain (solver.py:262-295) per LC submap.
    # Two graph-only nodes for the LC frames; anchor A ties the query frame to
    # LC frame 0 (identical image, scale fold s_a), the inner edge carries the
    # LC relative, anchor B ties LC frame 1 to the detected frame (scale s_b).
    for lc in lc_submaps:
        if lc.poses.shape[0] != 2:
            continue
        path_q, path_d = lc.image_paths[0], lc.image_paths[1]
        loc_q = _resolve_frame_node(frame_to_node, submaps, path_q)
        loc_d = _resolve_frame_node(frame_to_node, submaps, path_d)
        if loc_q is None or loc_d is None:
            continue
        nid_q, sub_q, qi = loc_q
        nid_d, sub_d, di = loc_d

        # Anchor scales on pixel-aligned identical images; fall back to 1.0 when
        # the LC backend supplies poses only (direction fix still applies).
        s_a = _lc_anchor_scale(lc, 0, sub_q, qi, conf_threshold, scale_method)
        s_b = _lc_anchor_scale(sub_d, di, lc, 1, conf_threshold, scale_method)
        if s_a is None or s_b is None:
            log.warning(
                "Loop submap %d → %d: LC world points unavailable or grid-misaligned; " "using anchor scale 1.0",
                sub_q.submap_id,
                sub_d.submap_id,
            )
            s_a = 1.0 if s_a is None else s_a
            s_b = 1.0 if s_b is None else s_b

        # Per-frame intrinsics as 4×4 for the K change across anchors (I for a shared camera)
        K_q, K_lc0, K_lc1, K_d = (np.eye(4, dtype=np.float64) for _ in range(4))
        K_q[:3, :3] = sub_q.intrinsics[qi].astype(np.float64)
        K_lc0[:3, :3] = lc.intrinsics[0].astype(np.float64)
        K_lc1[:3, :3] = lc.intrinsics[1].astype(np.float64)
        K_d[:3, :3] = sub_d.intrinsics[di].astype(np.float64)

        H_rel_a, H_inner_lc, H_rel_b = _loop_chain_relatives(lc.poses[0], lc.poses[1], s_a, s_b, K_q, K_lc0, K_lc1, K_d)

        # LC nodes chained from the query node's current graph state
        # (upstream solver.py:162-166); they map to no output frame.
        nid_lc0, nid_lc1 = global_node_id, global_node_id + 1
        global_node_id += 2
        H_q_state = pg.get_homography(nid_q)
        pg.add_node(nid_lc0, H_q_state @ H_rel_a)
        pg.add_node(nid_lc1, H_q_state @ H_rel_a @ H_inner_lc)

        pg.add_sequential_edge(nid_q, nid_lc0, H_rel_a)
        pg.add_sequential_edge(nid_lc0, nid_lc1, H_inner_lc)
        pg.add_sequential_edge(nid_lc1, nid_d, H_rel_b)

    pg.optimize()

    corrected_per_submap: dict[int, np.ndarray] = {}
    for submap in submaps:
        node_ids = submap_node_ids[submap.submap_id]
        k = len(node_ids)
        poses_out = np.zeros((k, 4, 4), dtype=np.float32)
        # Extraction via K_4x4 — matches VGGT-SLAM: proj_mats[i] @ inv(H_opt[i]).
        # decompose_camera cancels the K factor and returns correct SE3 R, t.
        s_K = np.tile(np.eye(4, dtype=np.float64), (k, 1, 1))
        s_K[:, :3, :3] = submap.intrinsics.astype(np.float64)
        for local_i, nid in enumerate(node_ids):
            H_opt = pg.get_homography(nid)
            local_proj = s_K[local_i]  # camera intrinsics as 4×4
            corrected = local_proj @ np.linalg.inv(H_opt)
            _, R, t, _ = decompose_camera(corrected)
            if debug_out is not None and local_i == 0:
                for entry in debug_out:
                    if entry.get("submap_id") == submap.submap_id and "H_opt" not in entry:
                        entry["H_opt"] = H_opt.copy()
                        entry["corrected_proj"] = corrected.copy()
                        break
            # decompose_camera returns R as the camera-to-world rotation and
            # t = inv(K) @ P[:,3] (VGGT-SLAM's no_inverse=True values). SLAM's
            # camera centre is C = -R @ t, i.e. the world-to-cam pose is
            # [R^T | t]. Store R^T so downstream -R_stored^T @ t recovers C.
            # (Storing R directly gives -R^T @ t — correct only for near-symmetric
            # rotations, which is why single-submap matched but multi-submap bent.)
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R.T.astype(np.float32)
            mat[:3, 3] = t.astype(np.float32)
            poses_out[local_i] = mat
        corrected_per_submap[submap.submap_id] = poses_out

    return dedup_overlap(
        submap_ids=[s.submap_id for s in submaps],
        submap_starts=[s.frame_start for s in submaps],
        corrected=corrected_per_submap,
        total_frames=total_frames,
    )


def _resolve_frame_node(
    frame_to_node: dict[tuple[int, int], int],
    submaps: list[Submap],
    image_path,
) -> tuple[int, Submap, int] | None:
    """Resolve an image path to (node_id, submap, local frame index)."""
    for submap in submaps:
        for local_i, p in enumerate(submap.image_paths):
            if p == image_path:
                nid = frame_to_node.get((submap.submap_id, local_i))
                if nid is not None:
                    return nid, submap, local_i
    return None


def merge_submap_outputs(
    submaps: list[Submap],
    corrected_extrinsics: np.ndarray,
    graph: "PoseGraph | None" = None,
) -> dict:
    """Assemble a unified raw_outputs dict from per-submap outputs with corrected poses.

    Args:
        graph: optimized PoseGraph (from graph.py) whose get_homography() returns the
            SL(4) homography for any node_id. When provided, each submap's world_points
            are reprojected via submap.get_world_points(H=graph.get_homography(frame_start)).
            Without this the concatenated world_points are in mixed submap-local frames.
    """
    merged: dict = {}
    if not submaps or submaps[0].raw_outputs is None:
        return {"extrinsic": corrected_extrinsics}

    # Handle backends (e.g. MapAnything) whose _forward returns list[dict] per frame.
    # Deduplicate overlap frames: each submap contributes only its non-overlap frames
    # (first K frames, where K = len(s.poses) - overlap), except the last submap which
    # contributes all its frames. This produces exactly N unique frames aligned with
    # corrected_extrinsics (N, 4, 4).
    if isinstance(submaps[0].raw_outputs, list):
        merged_list: list = []
        n_submaps = len(submaps)
        for idx, s in enumerate(submaps):
            if not s.raw_outputs:
                continue
            raw_list = s.raw_outputs
            if idx < n_submaps - 1:
                # Infer overlap from next submap's frame_start vs this submap's end
                next_start = submaps[idx + 1].frame_start
                this_end = s.frame_start + len(raw_list)
                overlap = max(0, this_end - next_start)
                keep = len(raw_list) - overlap
                merged_list.extend(raw_list[:keep])
            else:
                merged_list.extend(raw_list)
        merged_list_out: dict = {
            "_raw_list": merged_list,
            "extrinsic": corrected_extrinsics[:, :3, :],
            "extrinsic_global_4x4": corrected_extrinsics,
        }
        return merged_list_out

    sample = submaps[0].raw_outputs
    for key, val in sample.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1:
            try:
                merged[key] = np.concatenate(
                    [s.raw_outputs[key] for s in submaps if s.raw_outputs and key in s.raw_outputs],
                    axis=0,
                )
            except Exception:
                merged[key] = val
        else:
            merged[key] = val

    # raw_outputs["images"] is a torch tensor — generic loop above falls through to
    # merged["images"] = sample["images"] (only first submap, size K not N*K).
    # Rebuild from submap.frames (already CPU) converting to numpy so downstream
    # code (unproject_and_filter_points) receives an array aligned with depth/conf.
    frame_chunks = []
    for s in submaps:
        if s.frames is None:
            continue
        f = s.frames
        if hasattr(f, "cpu"):
            f = f.cpu().float().numpy()
        else:
            f = np.asarray(f, dtype=np.float32)
        frame_chunks.append(f)
    if frame_chunks:
        merged["images"] = np.concatenate(frame_chunks, axis=0)

    # world_points live on Submap.world_points (not in raw_outputs).  Merge them
    # here so downstream code gets a single global-frame point cloud.
    # When graph is provided, reproject each submap's world_points via SL(4)
    # get_world_points(H=graph.get_homography(frame_start)). Without it points
    # are concatenated in their mixed submap-local frames.
    wp_chunks = []
    for s in submaps:
        if s.world_points is None:
            continue
        if graph is not None:
            pts = s.get_world_points(H=graph.get_homography(s.frame_start))
        else:
            pts = s.world_points.reshape(-1, 3).astype(np.float32)
        wp_chunks.append(pts)
    if wp_chunks:
        merged["world_points"] = np.concatenate(wp_chunks, axis=0)

    # corrected_extrinsics is indexed by global frame index (size N).
    # All other keys were concatenated across submaps including overlapping frames,
    # so extrinsic must follow the same scheme: repeat frames from each submap's
    # local window (frame_start : frame_start + len(poses)) to stay aligned.
    # corrected_extrinsics is (N, 4, 4); raw_outputs["extrinsic"] convention is (K, 3, 4).
    # Slice to 3x4 so _raw_to_world_points can append the bottom row without producing (K,5,4).
    # corrected_extrinsics is (N, 4, 4); raw_outputs["extrinsic"] convention is (K, 3, 4).
    # Slice to 3x4 so _raw_to_world_points can append the bottom row without producing (K,5,4).
    # Uses the overlap-expanded scheme (size M >= N) so depth/images stay aligned.
    merged["extrinsic"] = np.concatenate(
        [corrected_extrinsics[s.frame_start : s.frame_start + len(s.poses), :3, :] for s in submaps],
        axis=0,
    )
    # Store the deduped global poses (N, 4, 4) separately so _postprocess can expose exactly
    # N extrinsics in FeedforwardResult (not the overlap-expanded M).
    merged["extrinsic_global_4x4"] = corrected_extrinsics

    # Build a dedup-index array: for each global frame g in 0..N-1, record the first
    # row in the M-expanded concatenated arrays that corresponds to frame g.
    # Used by BundleAdjustment._apply_ba to align intrinsics/images/conf/world_points
    # (all M-expanded) down to N unique frames before running track extraction + BA.
    N_global = corrected_extrinsics.shape[0]
    dedup_rows = np.full(N_global, -1, dtype=np.int64)
    row = 0
    for s in submaps:
        for li in range(len(s.poses)):
            g = s.frame_start + li
            if g < N_global and dedup_rows[g] < 0:
                dedup_rows[g] = row
            row += 1
    merged["_dedup_rows"] = dedup_rows  # (N,) int64 — index into M-expanded arrays

    return merged
