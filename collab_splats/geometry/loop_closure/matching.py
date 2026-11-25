"""
Loop-closure candidates from global retrieval descriptors.

- a candidate pairs a query frame with its nearest frame in an earlier submap
- ported from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): vggt_slam/loop_closure.py, map.py:retrieve_best_score_frame
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from collab_splats.geometry.loop_closure.submap import Submap


@dataclass
class LoopMatch:
    """
    Loop-closure candidate: a query frame and its nearest frame in an earlier submap.

    - frame indices are local to their submaps

    Args:
        similarity_score: L2 distance between unit retrieval descriptors; lower is more similar.
        query_submap_id: submap holding the query frame.
        detected_submap_id: earlier submap holding the nearest frame.
        query_frame_idx: query frame, local to its submap.
        detected_frame_idx: nearest frame, local to its submap.
        accepted: True once the wrapper verifies the candidate.
    """

    similarity_score: float
    query_submap_id: int
    detected_submap_id: int
    query_frame_idx: int
    detected_frame_idx: int
    accepted: bool = False


def find_loop_closures(
    query_submap: Submap,
    past_submaps: list[Submap],
    lc_threshold: float,
    max_loops: int,
    nms_frame_distance: int,
) -> list[LoopMatch]:
    """
    Best loop candidates between a query submap and earlier submaps.

    - per query frame and past submap, keeps the nearest frame if under lc_threshold
    - the max_loops lowest distances are kept first, then suppressed: a candidate within
      nms_frame_distance frames of a better kept one on the same detected submap is dropped

    Args:
        query_submap: newest submap, with retrieval_vectors set.
        past_submaps: earlier submaps to search.
        lc_threshold: maximum descriptor L2 distance for a candidate.
        max_loops: candidates kept before suppression.
        nms_frame_distance: suppression radius in frames; <= 0 disables suppression.

    Returns:
        Candidates in ascending similarity_score; empty when past_submaps is empty.
    """
    # Nearest past frame per query frame and past submap, by descriptor L2, kept under lc_threshold
    candidates: list[LoopMatch] = []
    assert query_submap.retrieval_vectors is not None

    for q_idx in range(query_submap.retrieval_vectors.shape[0]):
        q_vec = query_submap.retrieval_vectors[q_idx]

        for past in past_submaps:
            dists = torch.cdist(q_vec.unsqueeze(0), past.retrieval_vectors).squeeze(0)
            best_idx = int(dists.argmin())
            best_dist = float(dists[best_idx])

            if best_dist < lc_threshold:
                candidates.append(
                    LoopMatch(
                        similarity_score=best_dist,
                        query_submap_id=query_submap.submap_id,
                        detected_submap_id=past.submap_id,
                        query_frame_idx=q_idx,
                        detected_frame_idx=best_idx,
                    )
                )

    # The max_loops lowest distances, best first
    ranked = sorted(candidates, key=lambda m: m.similarity_score)[:max_loops]

    # Drop a candidate within nms_frame_distance frames of a better kept one on the same detected submap
    kept: list[LoopMatch] = []

    for cand in ranked:
        suppressed = any(
            k.detected_submap_id == cand.detected_submap_id
            and abs(k.detected_frame_idx - cand.detected_frame_idx) < nms_frame_distance
            for k in kept
        )

        if not suppressed:
            kept.append(cand)

    return kept
