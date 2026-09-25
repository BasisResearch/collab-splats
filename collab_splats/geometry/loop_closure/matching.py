"""
Loop-closure candidates from DINO-SALAD retrieval descriptors.

- a candidate pairs a query frame with its nearest frame in an earlier submap
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch

from .submap import Submap

########################################
####### Retrieval matching #############
########################################


@dataclass
class LoopMatch:
    """
    Loop-closure candidate: a query frame and its nearest frame in an earlier submap.

    - similarity_score: L2 distance between unit DINO-SALAD descriptors; lower is more similar
    - frame indices are local to their submaps
    """

    similarity_score: float
    query_submap_id: int
    detected_submap_id: int
    query_frame_idx: int
    detected_frame_idx: int
    accepted: bool = False
    # None on accepted matches; "verify_ratio" | "no_joint_poses" | "non_finite_pose" on rejects.
    reject_reason: str | None = None


class LoopMatchQueue:
    """
    Top-k lowest-distance candidates, with optional non-maximum suppression.
    """

    def __init__(self, max_size: int, nms_frame_distance: int) -> None:
        self._max_size = max_size
        self._nms = nms_frame_distance
        self._counter: int = 0
        self._heap: list = []

    def push(self, match: LoopMatch) -> None:
        """
        Add a candidate, evicting the highest-distance one once over max_size.

        Args:
            match: candidate to add.
        """
        heapq.heappush(self._heap, (-match.similarity_score, self._counter, match))
        self._counter += 1
        if len(self._heap) > self._max_size:
            heapq.heappop(self._heap)

    def get_matches(self) -> list[LoopMatch]:
        """
        Kept candidates, best first, after suppression.

        - a candidate is dropped when a better one hits the same detected submap
          within nms_frame_distance frames

        Returns:
            Candidates in ascending similarity_score.
        """
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
    nms_frame_distance: int,
) -> list[LoopMatch]:
    """
    Best loop candidates between a query submap and earlier submaps.

    - per query frame and past submap, keeps the nearest frame if under lc_threshold

    Args:
        query_submap: newest submap, with retrieval_vectors set.
        past_submaps: earlier submaps to search.
        lc_threshold: maximum descriptor L2 distance for a candidate.
        max_loops: number of candidates kept.
        nms_frame_distance: suppression radius in frames; 0 disables suppression.

    Returns:
        Candidates in ascending similarity_score; empty when past_submaps is empty.
    """
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

