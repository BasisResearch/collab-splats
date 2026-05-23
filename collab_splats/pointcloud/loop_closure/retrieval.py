from __future__ import annotations

import heapq
import math
import warnings
from dataclasses import dataclass


@dataclass
class LoopMatch:
    """Loop closure candidate produced by DINO-SALAD retrieval.

    similarity_score: L2 distance on normalized embeddings — lower = more similar.
    accepted: set to True by the feedforward gate loop when all gates pass.
    """
    similarity_score: float
    query_submap_id: int
    detected_submap_id: int
    query_frame_idx: int
    detected_frame_idx: int
    accepted: bool = False


@dataclass
class LoopClosureConfig:
    submap_size: int = 20
    submap_overlap: int = 4
    lc_cosine_threshold: float = 0.75   # cosine similarity threshold (≥ this → candidate)
    max_loops_per_submap: int = 5       # top-k candidates per submap (was 1)
    verify_match_ratio: float = 0.85    # min image_match_ratio to confirm loop
    nms_frame_distance: int = 25        # suppress candidates within this many frames of accepted
    min_submap_gap: int = 1             # skip this many most-recent submaps from LC search
                                        # (adjacent submap shares overlap frames → trivial self-match)
    sim3_lm_steps: int = 10             # Levenberg-Marquardt steps for Sim3 pose graph optimizer
    lc_threshold: float | None = None   # deprecated: use lc_cosine_threshold

    def __post_init__(self) -> None:
        if self.lc_threshold is not None:
            warnings.warn(
                "LoopClosureConfig.lc_threshold is deprecated; use lc_cosine_threshold instead. "
                f"Equivalent cosine threshold: {1 - self.lc_threshold**2 / 2:.4f}",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def lc_threshold_l2(self) -> float:
        """L2 distance threshold equivalent to lc_cosine_threshold on unit-norm vectors."""
        return math.sqrt(2 * (1 - self.lc_cosine_threshold))


class LoopMatchQueue:
    """Max-heap keeping the top-k lowest-distance LoopMatch candidates, with optional NMS."""

    def __init__(self, max_size: int, nms_frame_distance: int = 0):
        self._max_size = max_size
        self._nms = nms_frame_distance
        self._counter: int = 0
        self._heap: list[tuple[float, int, LoopMatch]] = []

    def push(self, match: LoopMatch) -> None:
        heapq.heappush(self._heap, (-match.similarity_score, self._counter, match))
        self._counter += 1
        if len(self._heap) > self._max_size:
            heapq.heappop(self._heap)  # evict worst (highest distance)

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


import torch
from .submap import Submap

from collab_splats.pointcloud.localization import BaseRetrievalExtractor


def _get_retrieval_extractor(device: str):
    cls = BaseRetrievalExtractor.get("dino-salad")
    return cls(device=device)


class ImageRetrieval:
    """Detects loop closure candidates via DINO-SALAD embedding similarity.

    Ported from MIT-SPARK/VGGT-SLAM vggt_slam/loop_closure.py.
    similarity_score is L2 distance on normalized embeddings (lower = more similar).
    """

    def __init__(self, device: str = "cuda"):
        self.extractor = _get_retrieval_extractor(device)

    def embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Embed (K, 3, H, W) frames → (K, D) normalized descriptors."""
        return self.extractor(frames)

    def find_loop_closures(
        self,
        query_submap: Submap,
        past_submaps: list[Submap],
        lc_threshold: float,
        max_loops: int,
        nms_frame_distance: int = 0,
    ) -> list[LoopMatch]:
        """Return top-k loop closure candidates using pre-computed retrieval_vectors.

        Uses L2 distance on normalized embeddings stored in Submap.retrieval_vectors.
        """
        if not past_submaps:
            return []

        queue = LoopMatchQueue(max_size=max_loops, nms_frame_distance=nms_frame_distance)

        for q_idx in range(query_submap.retrieval_vectors.shape[0]):
            q_vec = query_submap.retrieval_vectors[q_idx]  # (D,)

            for past in past_submaps:
                dists = torch.cdist(
                    q_vec.unsqueeze(0),           # (1, D)
                    past.retrieval_vectors,        # (K, D)
                ).squeeze(0)                      # (K,)

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
