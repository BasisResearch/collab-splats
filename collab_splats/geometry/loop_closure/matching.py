"""Loop-closure candidate matching and pre-add gates.

DINO-SALAD retrieval matching (LoopMatch, LoopMatchQueue, find_loop_closures)
plus the translation-jump alarm that rejects incompatible loop corrections.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import torch

from .submap import Submap

########################################
####### Retrieval matching #############
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
####### Loop-jump pre-add gate #########
########################################


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
