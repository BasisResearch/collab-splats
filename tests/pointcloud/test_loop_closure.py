import numpy as np
import torch
from pathlib import Path
from collab_splats.pointcloud.loop_closure import Submap
from collab_splats.pointcloud.loop_closure import LoopClosureConfig, LoopMatch, LoopMatchQueue


def _make_submap(k=4, h=224, w=224, d=128, submap_id=0):
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, h, w),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(
            np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]], dtype=np.float32), (k, 1, 1)
        ),
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
    assert not s.is_lc_submap


def test_submap_lc_flag():
    s = _make_submap()
    s.is_lc_submap = True
    assert s.is_lc_submap


def test_loop_closure_config_defaults():
    cfg = LoopClosureConfig()
    assert cfg.submap_size == 20
    assert cfg.submap_overlap == 4
    assert cfg.lc_cosine_threshold == 0.75
    assert cfg.max_loops_per_submap == 5
    assert cfg.verify_match_ratio == 0.85
    assert cfg.nms_frame_distance == 25
    assert cfg.min_submap_gap == 1
    assert cfg.lc_threshold is None


def test_loop_closure_config_l2_property():
    import math
    cfg = LoopClosureConfig(lc_cosine_threshold=0.85)
    expected = math.sqrt(2 * (1 - 0.85))
    assert abs(cfg.lc_threshold_l2 - expected) < 1e-6


def test_loop_closure_config_deprecated_threshold():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = LoopClosureConfig(lc_threshold=0.95)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "lc_cosine_threshold" in str(w[0].message)


def test_loop_match_queue_keeps_top_k():
    queue = LoopMatchQueue(max_size=2)
    queue.push(LoopMatch(0.9, 0, 1, 0, 0))
    queue.push(LoopMatch(0.5, 0, 2, 0, 0))  # lower distance = better
    queue.push(LoopMatch(0.7, 0, 3, 0, 0))
    matches = queue.get_matches()
    assert len(matches) == 2
    distances = [m.similarity_score for m in matches]
    assert sorted(distances) == distances  # ascending (best first)
    assert 0.5 in distances and 0.7 in distances  # 0.9 evicted


def test_loop_match_dataclass():
    m = LoopMatch(0.3, query_submap_id=0, detected_submap_id=1, query_frame_idx=2, detected_frame_idx=5)
    assert m.similarity_score == 0.3
    assert m.detected_submap_id == 1
    assert m.accepted is False  # default
    m.accepted = True
    assert m.accepted is True


def test_loop_match_queue_equal_score_tiebreak():
    """Equal scores must not raise TypeError (dataclass lacks __lt__)."""
    queue = LoopMatchQueue(max_size=5)
    for i in range(3):
        queue.push(LoopMatch(0.5, query_submap_id=0, detected_submap_id=i, query_frame_idx=0, detected_frame_idx=0))
    matches = queue.get_matches()
    assert len(matches) == 3


from collab_splats.pointcloud.loop_closure import find_loop_closures


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

    matches = find_loop_closures(query, [past], lc_threshold=0.001, max_loops=1)
    assert matches == []


def test_loop_match_queue_nms():
    from collab_splats.pointcloud.loop_closure import LoopMatch
    from collab_splats.pointcloud.loop_closure.closure import LoopMatchQueue
    # frames [10, 12, 50, 53, 100] — 10+12 cluster, 50+53 cluster, 100 alone
    # nms=25: keep best of each cluster by score (lower = better)
    queue = LoopMatchQueue(max_size=10, nms_frame_distance=25)
    for frame_idx, score in [(10, 0.1), (12, 0.2), (50, 0.15), (53, 0.3), (100, 0.05)]:
        queue.push(LoopMatch(
            similarity_score=score,
            query_submap_id=1,
            detected_submap_id=0,
            query_frame_idx=0,
            detected_frame_idx=frame_idx,
        ))
    matches = queue.get_matches()
    detected_frames = [m.detected_frame_idx for m in matches]
    assert 10 in detected_frames
    assert 12 not in detected_frames
    assert 50 in detected_frames
    assert 53 not in detected_frames
    assert 100 in detected_frames
