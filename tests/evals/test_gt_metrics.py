"""Tests for evals.gt_metrics."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evals.gt_metrics import ate, auc_at_threshold, depth_error, rpe


def _trajectory(n=20, seed=0):
    """Random-walk camera-to-world poses."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(scale=0.3, size=(n, 3))).as_matrix()
    poses[:, :3, 3] = np.cumsum(rng.normal(scale=0.2, size=(n, 3)), axis=0)
    return poses


def _sim3(poses, s=2.5):
    """Apply a fixed similarity transform to camera-to-world poses."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.1, -0.4, 0.7]).as_matrix()
    T[:3, 3] = [1.0, -2.0, 0.5]
    out = T @ poses
    out[:, :3, 3] = s * (out[:, :3, 3] - T[:3, 3]) + T[:3, 3]
    return out


def test_ate_zero_on_identical():
    gt = _trajectory()
    assert ate(gt.copy(), gt)["rmse"] < 1e-9


def test_ate_sim3_invariant():
    gt = _trajectory()
    assert ate(_sim3(gt), gt)["rmse"] < 1e-6


def test_ate_detects_noise():
    gt = _trajectory()
    pred = gt.copy()
    pred[5, :3, 3] += 0.5
    out = ate(pred, gt)
    assert out["rmse"] > 0.05
    assert int(np.argmax(out["per_frame"])) == 5
    assert out["aligned_positions"].shape == (20, 3)


def test_rpe_zero_under_sim3():
    gt = _trajectory()
    out = rpe(_sim3(gt), gt)
    assert out["trans_rmse"] < 1e-6 and out["rot_rmse_deg"] < 1e-4


def test_rpe_detects_rotation_step():
    gt = _trajectory()
    pred = gt.copy()
    pred[10:, :3, :3] = pred[10:, :3, :3] @ Rotation.from_rotvec([0, 0, np.radians(5)]).as_matrix()
    assert rpe(pred, gt)["rot_rmse_deg"] > 0.5


def test_auc_perfect_is_100():
    gt = _trajectory()
    # 1-degree bins: a perfect pair lands in bin 0, so AUC@30 is exactly 100
    assert auc_at_threshold(_sim3(gt), gt, (30.0,))["auc_30"] == pytest.approx(100.0)


def test_auc_counts_unregistered_pairs_as_failures():
    gt = _trajectory()
    n = len(gt)
    # One of n frames unregistered: (n-1)(n-2) of n(n-1) directed pairs remain, all perfect
    auc = auc_at_threshold(_sim3(gt[1:]), gt[1:], (30.0,), n_frames=n)["auc_30"]
    assert auc == pytest.approx(100.0 * (n - 2) / n)


def test_auc_drops_with_rotation_error():
    gt = _trajectory()
    pred = gt.copy()
    pred[:10, :3, :3] = pred[:10, :3, :3] @ Rotation.from_rotvec([0, 0, np.radians(20)]).as_matrix()
    assert auc_at_threshold(pred, gt, (30.0,))["auc_30"] < 80.0


def test_depth_error_recovers_scale():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 4.0, size=(3, 8, 8)).astype(np.float32)
    gt[0, 0, 0] = 0.0
    out = depth_error(gt / 3.0, gt)
    assert out["scale"] == pytest.approx(3.0, rel=1e-5)
    assert out["median_rel_err"] < 1e-5
    assert out["coverage"] == pytest.approx(1.0)


def test_depth_error_coverage_counts_missing_predictions():
    gt = np.ones((1, 2, 2))
    pred = np.array([[[1.0, 0.0], [1.0, 1.0]]])
    assert depth_error(pred, gt)["coverage"] == pytest.approx(0.75)


def test_depth_error_no_overlap_raises():
    with pytest.raises(ValueError, match="no pixel"):
        depth_error(np.zeros((1, 4, 4)), np.ones((1, 4, 4)))


def test_rpe_accepts_near_rotation_gt():
    gt = _trajectory()
    gt[:, :3, :3] *= 1.0 + 1.7e-4  # 7-Scenes GT: rotations off-orthonormal by ~1e-4
    out = rpe(_trajectory(), gt)
    assert np.isfinite(out["trans_rmse"]) and np.isfinite(out["rot_rmse_deg"])
