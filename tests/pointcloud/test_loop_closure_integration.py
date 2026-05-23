"""Smoke tests: loop closure pipeline components integrate without crashing."""
import numpy as np
import torch
from pathlib import Path

from collab_splats.pointcloud.loop_closure import LoopClosureConfig, ImageRetrieval
from collab_splats.pointcloud.loop_closure import PoseGraph, Submap


def _make_submap(submap_id, k, vec_dim=128):
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 64, 64),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.nn.functional.normalize(torch.randn(k, vec_dim), p=2, dim=1),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
    )


def test_pose_graph_two_submaps_no_loop():
    """Two submaps with identity poses — optimizer returns poses without crashing."""
    s0 = _make_submap(0, k=4)
    s1 = _make_submap(1, k=4)
    pg = PoseGraph()
    pg.add_submaps([s0, s1])
    result = pg.optimize()
    assert set(result.keys()) == {0, 1}
    assert result[0].shape == (4, 4, 4)
    assert result[1].shape == (4, 4, 4)


def test_loop_closure_config_passed_through():
    cfg = LoopClosureConfig(submap_size=10, submap_overlap=2, lc_threshold=0.8)
    assert cfg.submap_size == 10
    assert cfg.submap_overlap == 2
    assert 0 < cfg.lc_threshold < 1.0


def test_image_retrieval_detects_identical_submaps():
    """Two submaps with identical retrieval vectors → loop detected at threshold 0.01."""
    d = 128
    base_vecs = torch.nn.functional.normalize(torch.randn(3, d), p=2, dim=1)
    s0 = Submap(
        submap_id=0, frames=torch.zeros(3, 3, 64, 64),
        poses=np.tile(np.eye(4), (3, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (3, 1, 1)).astype(np.float32),
        retrieval_vectors=base_vecs,
        image_paths=[Path(f"f{i}.jpg") for i in range(3)],
    )
    s1 = Submap(
        submap_id=1, frames=torch.zeros(3, 3, 64, 64),
        poses=np.tile(np.eye(4), (3, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (3, 1, 1)).astype(np.float32),
        retrieval_vectors=base_vecs.clone(),  # identical → distance ≈ 0
        image_paths=[Path(f"f{i+3}.jpg") for i in range(3)],
    )
    retrieval = ImageRetrieval.__new__(ImageRetrieval)
    retrieval.extractor = None

    matches = retrieval.find_loop_closures(s1, [s0], lc_threshold=0.01, max_loops=1)
    assert len(matches) == 1
    assert matches[0].detected_submap_id == 0
    assert matches[0].similarity_score < 1e-5


def test_verify_loop_candidate_returns_tuple():
    """F4: _verify_loop_candidate must return (bool, ndarray|None), not bare bool."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    creator = object.__new__(MapAnythingCreator)
    creator._lc_retrieval = None

    result = creator._verify_loop_candidate(
        torch.zeros(3, 64, 64), torch.zeros(3, 64, 64)
    )
    assert isinstance(result, tuple) and len(result) == 2
    accepted, lc_poses = result
    assert accepted is False
    assert lc_poses is None


def test_base_verify_raises_with_tuple_signature():
    """F4/F5: base _verify_loop_candidate raises NotImplementedError."""
    import pytest
    from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator, FeedforwardResult

    class _D(BaseFeedforwardCreator):
        def _load_model(self, device): pass
        def _preprocess(self, image_dir): return None, [], np.zeros((0, 2))
        def _forward(self, model, views, **kw): return {}
        def _postprocess(self, raw, **kw):
            return FeedforwardResult(np.zeros((1, 3)), np.zeros((1, 3)),
                                     np.eye(4)[None], np.eye(3)[None], [], 1, 1)

    dummy = object.__new__(_D)
    with pytest.raises(NotImplementedError, match="_verify_loop_candidate"):
        dummy._verify_loop_candidate(None, None)
