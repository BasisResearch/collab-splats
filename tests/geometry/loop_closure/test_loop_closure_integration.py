"""Smoke tests: loop closure pipeline components integrate without crashing."""

from pathlib import Path

import numpy as np
import torch

from collab_splats.geometry.loop_closure import (
    LoopClosureConfig,
    PoseGraph,
    Submap,
    find_loop_closures,
)


def _make_submap(submap_id, k, vec_dim=128):
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 64, 64),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.nn.functional.normalize(torch.randn(k, vec_dim), p=2, dim=1),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
    )


def test_pose_graph_two_nodes_no_loop():
    """Two nodes with identity homographies — optimizer runs without crashing."""
    pg = PoseGraph()
    H0 = np.eye(4, dtype=np.float64)
    H1 = np.eye(4, dtype=np.float64)
    pg.add_homography(0, H0)
    pg.add_prior_factor(0, H0)
    pg.add_homography(1, H1)
    pg.add_between_factor(0, 1, H1)
    pg.optimize()
    assert pg.get_homography(0).shape == (4, 4)
    assert pg.get_homography(1).shape == (4, 4)


def test_loop_closure_config_passed_through():
    cfg = LoopClosureConfig(submap_size=10, submap_overlap=2, lc_retrieval_threshold=0.8)
    assert cfg.submap_size == 10
    assert cfg.submap_overlap == 2
    assert 0 < cfg.lc_retrieval_threshold < 1.0


def test_image_retrieval_detects_identical_submaps():
    """Two submaps with identical retrieval vectors → loop detected at threshold 0.01."""
    d = 128
    base_vecs = torch.nn.functional.normalize(torch.randn(3, d), p=2, dim=1)
    s0 = Submap(
        submap_id=0,
        frames=torch.zeros(3, 3, 64, 64),
        poses=np.tile(np.eye(4), (3, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (3, 1, 1)).astype(np.float32),
        retrieval_vectors=base_vecs,
        image_paths=[Path(f"f{i}.jpg") for i in range(3)],
    )
    s1 = Submap(
        submap_id=1,
        frames=torch.zeros(3, 3, 64, 64),
        poses=np.tile(np.eye(4), (3, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (3, 1, 1)).astype(np.float32),
        retrieval_vectors=base_vecs.clone(),  # identical → distance ≈ 0
        image_paths=[Path(f"f{i+3}.jpg") for i in range(3)],
    )
    matches = find_loop_closures(s1, [s0], lc_threshold=0.01, max_loops=1)
    assert len(matches) == 1
    assert matches[0].detected_submap_id == 0
    assert matches[0].similarity_score < 1e-5


def test_verify_loop_candidate_returns_tuple():
    """F4: _verify_loop_candidate must return (bool, lc_data dict|None), not bare bool."""
    from unittest.mock import MagicMock

    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    creator = object.__new__(MapAnythingCreator)
    creator._lc_retrieval = None
    # Provide a mock model so _verify_loop_candidate can resolve the device
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    creator.model = mock_model
    # Return orthogonal q/k tensors → cross-frame ratio ≈ 0 → rejected
    B, heads, N, hd = 1, 1, 20, 4
    k = torch.zeros(B, heads, N, hd)
    q = torch.zeros(B, heads, N, hd)
    k[:, :, :10, 0] = 10.0  # frame1 tokens align to dim 0
    q[:, :, :10, 0] = 10.0
    k[:, :, 10:, 1] = 10.0  # frame2 tokens align to dim 1 (orthogonal → low ratio)
    q[:, :, 10:, 1] = 10.0
    creator.extract_intermediate_features = lambda frames, layer_index=-1, **kw: {"q": q, "k": k}

    result = creator._verify_loop_candidate(torch.zeros(3, 64, 64), torch.zeros(3, 64, 64))
    assert isinstance(result, tuple) and len(result) == 2
    accepted, lc_poses = result
    assert accepted is False
    assert lc_poses is None
