from pathlib import Path

import numpy as np
import torch

from collab_splats.geometry.loop_closure import LoopClosureConfig, Submap
from collab_splats.geometry.loop_closure.matching import LoopMatch, find_loop_closures


def _make_submap(k=4, h=224, w=224, d=128, submap_id=0):
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, h, w),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]], dtype=np.float32), (k, 1, 1)),
        retrieval_vectors=torch.zeros(k, d),
        image_paths=[Path(f"frame_{i:04d}.jpg") for i in range(k)],
    )


def test_submap_creation():
    s = _make_submap(k=4)
    assert s.submap_id == 0
    assert s.frames.shape == (4, 3, 224, 224)
    assert s.poses.shape == (4, 4, 4)
    assert s.intrinsics.shape == (4, 3, 3)
    assert s.retrieval_vectors.shape == (4, 128)
    assert len(s.image_paths) == 4


def test_loop_closure_config_defaults():
    cfg = LoopClosureConfig()
    assert cfg.submap_size == 20
    assert cfg.submap_overlap == 1
    assert cfg.lc_retrieval_threshold == 0.95
    assert cfg.max_loops_per_submap == 5
    # None = resolve to the creator's default_verify_match_ratio at LoopClosure init
    assert cfg.verify_match_ratio is None
    assert cfg.nms_frame_distance == 25
    assert cfg.min_submap_gap == 1


def test_loop_match_dataclass():
    m = LoopMatch(0.3, query_submap_id=0, detected_submap_id=1, query_frame_idx=2, detected_frame_idx=5)
    assert m.similarity_score == 0.3
    assert m.detected_submap_id == 1
    assert m.accepted is False  # default
    m.accepted = True
    assert m.accepted is True


def test_find_loop_closures_detects_similar():
    d = 512
    base_vec = torch.nn.functional.normalize(torch.randn(d), p=2, dim=0)
    similar_vec = torch.nn.functional.normalize(base_vec + torch.randn(d) * 0.01, p=2, dim=0)
    different_vec = torch.nn.functional.normalize(torch.randn(d), p=2, dim=0)

    k = 2
    query = Submap(
        submap_id=2,
        frames=torch.zeros(k, 3, 64, 64),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=similar_vec.unsqueeze(0).expand(k, -1).clone(),
        image_paths=[Path(f"s2_f{i}.jpg") for i in range(k)],
    )
    past_similar = Submap(
        submap_id=0,
        frames=torch.zeros(k, 3, 64, 64),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=base_vec.unsqueeze(0).expand(k, -1).clone(),
        image_paths=[Path(f"s0_f{i}.jpg") for i in range(k)],
    )
    past_different = Submap(
        submap_id=1,
        frames=torch.zeros(k, 3, 64, 64),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=different_vec.unsqueeze(0).expand(k, -1).clone(),
        image_paths=[Path(f"s1_f{i}.jpg") for i in range(k)],
    )

    matches = find_loop_closures(
        query_submap=query,
        past_submaps=[past_similar, past_different],
        lc_threshold=0.5,
        max_loops=1,
        nms_frame_distance=0,
    )
    assert len(matches) == 1
    assert matches[0].detected_submap_id == 0


def test_find_loop_closures_no_match():
    d = 128

    query = Submap(
        submap_id=1,
        frames=torch.zeros(2, 3, 64, 64),
        poses=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.nn.functional.normalize(torch.randn(2, d), p=2, dim=1),
        image_paths=[Path(f"f{i}.jpg") for i in range(2)],
    )
    past = Submap(
        submap_id=0,
        frames=torch.zeros(2, 3, 64, 64),
        poses=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.nn.functional.normalize(torch.randn(2, d), p=2, dim=1),
        image_paths=[Path(f"g{i}.jpg") for i in range(2)],
    )

    matches = find_loop_closures(query, [past], lc_threshold=0.001, max_loops=1, nms_frame_distance=0)
    assert matches == []


def _find_at(frame_scores, max_loops, nms):
    """
    find_loop_closures where query frame i lies exactly frame_scores[i][1] from past frame frame_scores[i][0].

    - past descriptors are 10 * one-hot rows; a query row adds its score on an axis no past row uses
    """
    n_past = 101
    past = _make_submap(k=n_past, h=1, w=1, d=n_past + 1, submap_id=0)
    past.retrieval_vectors = 10.0 * torch.eye(n_past, n_past + 1)
    query = _make_submap(k=len(frame_scores), h=1, w=1, d=n_past + 1, submap_id=1)
    query.retrieval_vectors = torch.zeros(len(frame_scores), n_past + 1)

    for i, (frame_idx, score) in enumerate(frame_scores):
        query.retrieval_vectors[i, frame_idx] = 10.0
        query.retrieval_vectors[i, n_past] = score

    matches = find_loop_closures(query, [past], lc_threshold=1.0, max_loops=max_loops, nms_frame_distance=nms)
    return [m.detected_frame_idx for m in matches]


def test_find_loop_closures_ranks_caps_then_suppresses():
    """Best first; top max_loops kept before NMS drops near neighbors on the same submap."""
    frame_scores = [(10, 0.1), (12, 0.2), (50, 0.15), (53, 0.3), (100, 0.05)]

    # Clusters {10, 12}, {50, 53}, {100}: nms=25 keeps each cluster's best, in score order
    assert _find_at(frame_scores, max_loops=10, nms=25) == [100, 10, 50]

    # No suppression radius: every candidate, ascending distance
    assert _find_at(frame_scores, max_loops=10, nms=0) == [100, 10, 50, 12, 53]

    # The cap applies before NMS: top 3 are 100, 10, 50; 12 never reaches NMS
    assert _find_at(frame_scores, max_loops=3, nms=25) == [100, 10, 50]

    # Top 3 of {10, 12, 13} share one cluster, so NMS leaves one; 50 was capped out
    frame_scores = [(10, 0.1), (12, 0.12), (13, 0.13), (50, 0.15)]
    assert _find_at(frame_scores, max_loops=3, nms=25) == [10]
