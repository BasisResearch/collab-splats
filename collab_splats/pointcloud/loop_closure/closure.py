"""Loop closure orchestration utilities.

Houses pre-add gates, pose-graph build/dedup, and submap-output merging.
"""
from __future__ import annotations

import heapq
import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

import logging

from scipy.spatial.transform import Rotation as ScipyR

from .graph import PoseGraph as _SL4PoseGraph, decompose_camera, estimate_scale_pairwise, normalize_to_sl4
from .submap import Submap

log = logging.getLogger(__name__)

# Minimum points required for a reliable median scale estimate (matches VGGT-SLAM fallback threshold)
_MIN_CONF_POINTS = 100


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

    scale_src = float(np.sqrt((w * (src_c ** 2).sum(axis=1)).sum()))
    scale_tgt = float(np.sqrt((w * (tgt_c ** 2).sum(axis=1)).sum()))
    s = scale_tgt / scale_src if scale_src > 1e-9 else 1.0

    H = (src_c * s * w[:, None]).T @ tgt_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)

    t = (mu_tgt - s * R @ mu_src).astype(np.float32)
    return s, R, t


########################################
####### Absorbed from pose_graph.py ####
########################################

# SE(3) noise constants — used by eval.py edge classification
_ODOM_INTRA_SIGMA_R, _ODOM_INTRA_SIGMA_T = 0.02, 0.05
_INTER_SIGMA_R, _INTER_SIGMA_T = 0.05, 0.20

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


@dataclass
class LoopClosureConfig:
    submap_size: int = 20
    submap_overlap: int = 1  # 1 = VGGT-SLAM parity; 4 = old default
    lc_cosine_threshold: float = 0.549  # L2 < 0.95 parity with VGGT-SLAM (was 0.75 → L2 < 0.707, too strict)
    max_loops_per_submap: int = 5
    verify_match_ratio: float = 0.85
    nms_frame_distance: int = 25
    min_submap_gap: int = 1
    manifold: Literal["sl4", "se3"] = "sl4"
    max_jump_ratio: float = math.inf  # reject loops where ‖ΔT.t‖/path_length > this; math.inf disables
    conf_threshold: float = 25.0  # confidence gate for scale estimation; matches VGGT-SLAM --conf_threshold 25
    lc_threshold: float | None = None   # deprecated

    def __post_init__(self) -> None:
        if self.lc_threshold is not None:
            warnings.warn(
                "LoopClosureConfig.lc_threshold is deprecated; use lc_cosine_threshold. "
                f"Equivalent: {1 - self.lc_threshold**2 / 2:.4f}",
                DeprecationWarning, stacklevel=2,
            )

    @property
    def lc_threshold_l2(self) -> float:
        return math.sqrt(2 * (1 - self.lc_cosine_threshold))


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
        candidates = sorted(
            [m for _, _, m in self._heap], key=lambda m: m.similarity_score
        )
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
    for q_idx in range(query_submap.retrieval_vectors.shape[0]):
        q_vec = query_submap.retrieval_vectors[q_idx]
        for past in past_submaps:
            dists = torch.cdist(q_vec.unsqueeze(0), past.retrieval_vectors).squeeze(0)
            best_idx = int(dists.argmin())
            best_dist = float(dists[best_idx])
            if best_dist < lc_threshold:
                queue.push(LoopMatch(
                    similarity_score=best_dist,
                    query_submap_id=query_submap.submap_id,
                    detected_submap_id=past.submap_id,
                    query_frame_idx=q_idx,
                    detected_frame_idx=best_idx,
                ))
    return queue.get_matches()


########################################
####### Absorbed from alignment.py #####
########################################


def dedup_overlap(  # noqa: F811 — shadows import; canonical copy lives here
    submap_ids: list[int],
    submap_starts: list[int],
    corrected: dict[int, np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """Reconstruct (total_frames, 4, 4) from per-submap corrected poses, deduplicating overlap."""
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


def run_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
    manifold: Literal["sl4", "se3"] = "sl4",
    conf_threshold: float = 25.0,
    debug_out: list | None = None,
) -> np.ndarray:
    """Build + optimize per-frame SL(4) pose graph; return (total_frames, 4, 4).

    Per-frame node building mirrors vggt_slam/solver.py:add_edge:
    - Inner frames: H_inner = poses[i-1] @ inv(poses[i]); node chained from prev
    - Inter-submap first frame: scale estimated via estimate_scale_pairwise on
      overlapping world_points, H_w = graph.get_homography(overlap_prev) @ T @ H_scale
      where T = inv(P_prev_ov) @ P_curr_ov (full w2c poses, not K-only)
    - Loop edges from lc_submaps (2-frame submaps with verified LC poses)
    """
    if not submaps:
        return np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))

    pg = _SL4PoseGraph(manifold=manifold)
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

        if s_idx == 0:
            H0 = submap.poses[0].astype(np.float64)
            pg.add_node(node_ids_this[0], H0)
            pg.add_prior(node_ids_this[0], H0)
            for local_i in range(1, k):
                H_inner = (
                    submap.poses[local_i - 1].astype(np.float64)
                    @ np.linalg.inv(submap.poses[local_i].astype(np.float64))
                )
                prev_H = pg.get_homography(node_ids_this[local_i - 1])
                pg.add_node(node_ids_this[local_i], prev_H @ H_inner)
                pg.add_sequential_edge(
                    node_ids_this[local_i - 1], node_ids_this[local_i], H_inner
                )
        else:
            prev_submap = submaps[s_idx - 1]
            O = min(overlap_frames, k, len(prev_submap.poses))

            # Always compute T from overlap poses — needed for H_w (Bug 1 fix)
            # and scale estimation. Moved outside world_points block.
            P_curr_overlap = submap.poses[0].astype(np.float64)        # w2c: curr world → cam
            P_prev_overlap = prev_submap.poses[-1].astype(np.float64)  # w2c: prev world → cam
            T = np.linalg.inv(P_prev_overlap) @ P_curr_overlap         # curr world → prev world

            scale = 1.0
            if (
                submap.world_points is not None
                and prev_submap.world_points is not None
                and O > 0
            ):
                curr_pts = submap.world_points[:O].reshape(-1, 3).astype(np.float64)
                prev_pts = prev_submap.world_points[-O:].reshape(-1, 3).astype(np.float64)
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
                        # Fallback: try either-side mask; else use all points
                        either_mask = (curr_conf > conf_threshold) | (prev_conf > conf_threshold)
                        if either_mask.sum() >= _MIN_CONF_POINTS:
                            mask = either_mask

                curr_h = np.hstack([curr_pts, np.ones((n, 1))])
                curr_in_prev = (T @ curr_h.T).T[:, :3]
                scale = estimate_scale_pairwise(curr_in_prev[mask], prev_pts[mask])

            H_scale = np.diag([scale, scale, scale, 1.0])
            prev_submap_last_nid = submap_node_ids[prev_submap.submap_id][-1]
            H_overlap = pg.get_homography(prev_submap_last_nid)
            # Bug 1 fix: use full pose T (captures extrinsic rotation) instead of inv(K_prev)@K_curr
            H_w = H_overlap @ T @ H_scale
            pg.add_node(node_ids_this[0], H_w)

            if debug_out is not None:
                debug_out.append({
                    "submap_id": submap.submap_id,
                    "T": T.copy(),
                    "scale": float(scale),
                    "H_w": H_w.copy(),
                    "H_overlap": H_overlap.copy(),
                })

            H_rel_inter = np.linalg.inv(pg.get_homography(prev_submap_last_nid)) @ H_w
            pg.add_sequential_edge(prev_submap_last_nid, node_ids_this[0], H_rel_inter)

            for local_i in range(1, k):
                H_inner = (
                    submap.poses[local_i - 1].astype(np.float64)
                    @ np.linalg.inv(submap.poses[local_i].astype(np.float64))
                )
                prev_H = pg.get_homography(node_ids_this[local_i - 1])
                pg.add_node(node_ids_this[local_i], prev_H @ H_inner)
                pg.add_sequential_edge(
                    node_ids_this[local_i - 1], node_ids_this[local_i], H_inner
                )

        submap_node_ids[submap.submap_id] = node_ids_this
        global_node_id += k

    for lc in lc_submaps:
        if lc.poses.shape[0] != 2:
            continue
        path_q, path_d = lc.image_paths[0], lc.image_paths[1]
        nid_q = _resolve_frame_node(frame_to_node, submaps, path_q)
        nid_d = _resolve_frame_node(frame_to_node, submaps, path_d)
        if nid_q is None or nid_d is None:
            continue
        H_rel_lc = (
            np.linalg.inv(lc.poses[0].astype(np.float64))
            @ lc.poses[1].astype(np.float64)
        )
        pg.add_loop_edge(nid_q, nid_d, H_rel_lc)

    pg.optimize()

    corrected_per_submap: dict[int, np.ndarray] = {}
    for submap in submaps:
        node_ids = submap_node_ids[submap.submap_id]
        k = len(node_ids)
        poses_out = np.zeros((k, 4, 4), dtype=np.float32)
        for local_i, nid in enumerate(node_ids):
            H_opt = pg.get_homography(nid)
            local_proj = submap.poses[local_i].astype(np.float64)
            corrected = local_proj @ np.linalg.inv(H_opt)
            _, R, t, _ = decompose_camera(corrected)
            if debug_out is not None and local_i == 0:
                for entry in debug_out:
                    if entry.get("submap_id") == submap.submap_id and "H_opt" not in entry:
                        entry["H_opt"] = H_opt.copy()
                        entry["corrected_proj"] = corrected.copy()
                        break
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R.astype(np.float32)
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
) -> int | None:
    for submap in submaps:
        for local_i, p in enumerate(submap.image_paths):
            if p == image_path:
                return frame_to_node.get((submap.submap_id, local_i))
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


