# tests/evals/test_reconstruction_quality.py
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))


def _make_submaps(n_submaps=3, K=5, frame_start_step=4):
    """Build minimal Submap-like objects for testing."""
    from dataclasses import dataclass

    @dataclass
    class FakeSubmap:
        submap_id: int
        frame_start: int
        poses: np.ndarray  # (K,4,4)
        world_points: np.ndarray  # (K,P,3)
        is_lc_submap: bool = False

    submaps = []
    for i in range(n_submaps):
        poses = np.tile(np.eye(4, dtype=np.float32), (K, 1, 1))
        poses[:, 0, 3] = np.arange(K) * 0.1 + i * K * 0.1
        world_points = np.random.randn(K, 10, 3).astype(np.float32)
        submaps.append(FakeSubmap(
            submap_id=i,
            frame_start=i * frame_start_step,
            poses=poses,
            world_points=world_points,
        ))
    return submaps


def _make_match(q_sid, q_fidx, d_sid, d_fidx, accepted=True):
    from dataclasses import dataclass

    @dataclass
    class FakeMatch:
        query_submap_id: int
        query_frame_idx: int
        detected_submap_id: int
        detected_frame_idx: int
        accepted: bool
        similarity_score: float = 0.5

    return FakeMatch(q_sid, q_fidx, d_sid, d_fidx, accepted)


def test_loop_match_residual_perfect():
    from reconstruction_quality import loop_match_residual
    submaps = _make_submaps(3)
    N = 12
    pre = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    post = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    # accepted match between frame 0 in submap 0 and frame 0 in submap 2
    matches = [_make_match(2, 0, 0, 0, accepted=True)]
    result = loop_match_residual(matches, submaps, pre, post)
    # camera positions are the same (identity poses) so residual = 0
    assert result["mean_before"] == pytest.approx(0.0, abs=1e-5)
    assert result["mean_after"] == pytest.approx(0.0, abs=1e-5)
    assert result["n_matches"] == 1


def test_loop_match_residual_no_accepted():
    from reconstruction_quality import loop_match_residual
    submaps = _make_submaps(2)
    N = 8
    pre = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    matches = [_make_match(0, 0, 1, 0, accepted=False)]
    result = loop_match_residual(matches, submaps, pre, pre)
    assert result["n_matches"] == 0
    assert result["mean_before"] is None


def test_submap_boundary_gap_zero():
    from reconstruction_quality import submap_boundary_gap
    submaps = _make_submaps(3, K=5, frame_start_step=4)
    N = 12
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    result = submap_boundary_gap(submaps, poses, poses)
    # All identity poses → zero gap
    assert result["mean_before"] == pytest.approx(0.0, abs=1e-5)


def test_pointcloud_chamfer_identical():
    from reconstruction_quality import pointcloud_chamfer
    submaps = _make_submaps(3)
    N = 12
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    matches = [_make_match(2, 0, 0, 0, accepted=True)]
    result = pointcloud_chamfer(matches, submaps, poses, poses)
    assert "mean_before" in result
    assert "mean_after" in result
    assert result["n_pairs"] >= 0


def test_compute_alignment_metrics_missing_attrs():
    from reconstruction_quality import compute_alignment_metrics
    from unittest.mock import MagicMock
    creator = MagicMock()
    creator.base = MagicMock(spec=[])  # no _lc_* attrs
    result = compute_alignment_metrics(creator)
    assert result == {}
