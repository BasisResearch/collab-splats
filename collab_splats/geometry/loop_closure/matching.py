"""
Loop-closure candidates from DINO-SALAD retrieval descriptors.

- a candidate pairs a query frame with its nearest frame in an earlier submap
- adapted from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): vggt_slam/loop_closure.py
  (LoopMatch, LoopMatchQueue, find_loop_closures), vggt_slam/map.py:retrieve_best_score_frame
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch

from collab_splats.geometry.loop_closure.submap import Submap

########################################################################
# Retrieval matching
########################################################################


@dataclass
class LoopMatch:
    """
    Loop-closure candidate: a query frame and its nearest frame in an earlier submap.

    - frame indices are local to their submaps

    Args:
        similarity_score: L2 distance between unit DINO-SALAD descriptors; lower is more similar.
        query_submap_id: submap holding the query frame.
        detected_submap_id: earlier submap holding the nearest frame.
        query_frame_idx: query frame, local to its submap.
        detected_frame_idx: nearest frame, local to its submap.
        accepted: True once the wrapper verifies the candidate.
        reject_reason: None when accepted; else "verify_ratio", "no_joint_poses" or
            "non_finite_pose".
    """

    similarity_score: float
    query_submap_id: int
    detected_submap_id: int
    query_frame_idx: int
    detected_frame_idx: int
    accepted: bool = False
    reject_reason: str | None = None


class LoopMatchQueue:
    """
    Top-k lowest-distance candidates, with optional non-maximum suppression.
    """

    def __init__(self, max_size: int, nms_frame_distance: int) -> None:
        """
        Empty queue with a size cap and a suppression radius.

        - heap entries are (-score, insertion counter, match), so the root is the worst

        Args:
            max_size: candidates kept.
            nms_frame_distance: suppression radius in frames; 0 disables suppression.
        """
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
        # Best first; no suppression radius means nothing to drop
        candidates = sorted([m for _, _, m in self._heap], key=lambda m: m.similarity_score)
        if self._nms <= 0:
            return candidates

        # Keep a candidate unless a better kept one is on the same submap and nearby
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

    # Nearest past frame per query frame and past submap
    # - L2 distance over DINO-SALAD retrieval vectors
    # - kept only under lc_threshold
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
