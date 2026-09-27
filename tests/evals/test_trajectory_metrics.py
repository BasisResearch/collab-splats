import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
from trajectory_metrics import ate_translation, rpe, umeyama_align  # noqa: E402


def _make_poses(translations: np.ndarray) -> np.ndarray:
    """Build (N, 4, 4) world-to-cam poses with identity rotation and given translations."""
    N = len(translations)
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, :3, 3] = translations.astype(np.float32)
    return poses


def test_umeyama_align_identity():
    """When pred == gt, aligned_pred should equal pred (up to float tolerance)."""
    gt = _make_poses(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32))
    aligned, T = umeyama_align(gt.copy(), gt)
    np.testing.assert_allclose(aligned, gt, atol=1e-4)


def test_umeyama_align_pure_translation():
    """Pred shifted by constant offset; aligned_pred should match gt after alignment."""
    translations = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    gt = _make_poses(translations)
    # Pred is gt shifted by (5, 0, 0) in world space: cam_pos_world = R^T(-t)
    # For identity R, cam_pos_world = -t, so shifting t by -5 shifts cam_pos by +5
    shifted_t = translations.copy()
    shifted_t[:, 0] += 5.0  # shift cam translation component
    pred = _make_poses(-shifted_t)  # world-to-cam with shifted t
    aligned, _ = umeyama_align(pred, gt)
    # After alignment, camera positions should match gt
    def cam_pos(poses):
        R_ = poses[:, :3, :3]
        t_ = poses[:, :3, 3]
        return np.einsum("nij,nj->ni", R_.transpose(0, 2, 1), -t_)
    np.testing.assert_allclose(cam_pos(aligned), cam_pos(gt), atol=1e-3)


def test_ate_translation_perfect():
    """When pred == gt (after alignment), ATE RMSE should be near zero."""
    gt = _make_poses(np.random.RandomState(0).randn(10, 3).astype(np.float32))
    result = ate_translation(gt.copy(), gt)
    assert result["rmse"] < 1e-4
    assert result["mean"] < 1e-4
    assert "per_frame" in result
    assert len(result["per_frame"]) == 10


def test_ate_translation_returns_correct_keys():
    gt = _make_poses(np.zeros((5, 3), dtype=np.float32))
    result = ate_translation(gt.copy(), gt)
    assert set(result.keys()) >= {"rmse", "mean", "median", "max", "per_frame"}


def test_rpe_perfect():
    """When pred == gt, RPE trans and rot RMSE should be near zero."""
    gt = _make_poses(np.linspace([0, 0, 0], [5, 0, 0], 10).astype(np.float32))
    result = rpe(gt.copy(), gt, delta=1)
    assert result["trans_rmse"] < 1e-4
    assert result["rot_rmse_deg"] < 1e-3


def test_rpe_returns_correct_keys():
    gt = _make_poses(np.zeros((5, 3), dtype=np.float32))
    result = rpe(gt.copy(), gt)
    assert set(result.keys()) == {"trans_rmse", "rot_rmse_deg"}


def test_ate_nonzero_error():
    """Pred shifted uniformly; after Umeyama alignment, ATE should be near zero (translation-only drift is correctable)."""
    gt_t = np.linspace([0, 0, 0], [4, 0, 0], 5).astype(np.float32)
    gt = _make_poses(gt_t)
    # Pred has a constant offset — Umeyama removes it, ATE → 0
    pred = _make_poses(gt_t + np.array([10, 0, 0], dtype=np.float32))
    result = ate_translation(pred, gt)
    assert result["rmse"] < 1e-3


def test_rpe_nonzero_translation_error():
    """Pred with accumulated drift has nonzero RPE even though ATE ~0 after alignment."""
    # GT: uniform 1m steps; pred: steps grow by 0.1m each frame (drift)
    gt_steps = np.ones((9, 3), dtype=np.float32)
    gt_steps[:, 1:] = 0
    pred_steps = gt_steps.copy()
    pred_steps[:, 0] += np.arange(9, dtype=np.float32) * 0.1
    gt_t = np.vstack([[0, 0, 0], np.cumsum(gt_steps, axis=0)]).astype(np.float32)
    pred_t = np.vstack([[0, 0, 0], np.cumsum(pred_steps, axis=0)]).astype(np.float32)
    gt = _make_poses(gt_t)
    pred = _make_poses(pred_t)
    result = rpe(pred, gt)
    assert result["trans_rmse"] > 0.05


def _random_c2w(n: int, seed: int) -> np.ndarray:
    """Non-identity camera-to-world poses: random rotations, spread-out centers."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = R.random(n, random_state=seed).as_matrix()
    poses[:, :3, 3] = rng.normal(scale=3.0, size=(n, 3))
    return poses


def test_rpe_is_zero_for_a_sim3_copy_of_gt():
    """w2c input: a pred that is gt under a world Sim3 (scale 2.5, rotated, shifted) has no RPE."""
    gt_c2w = _random_c2w(8, seed=0)
    R_w = R.from_euler("xyz", [30, -50, 70], degrees=True).as_matrix()
    s, t_w = 2.5, np.array([4.0, -1.0, 2.0])
    pred_c2w = gt_c2w.copy()
    pred_c2w[:, :3, :3] = R_w @ gt_c2w[:, :3, :3]
    pred_c2w[:, :3, 3] = s * gt_c2w[:, :3, 3] @ R_w.T + t_w

    result = rpe(np.linalg.inv(pred_c2w), np.linalg.inv(gt_c2w), delta=1)

    assert result["trans_rmse"] < 1e-6
    assert result["rot_rmse_deg"] < 1e-4


def test_rpe_rotation_error_isolated_to_the_perturbed_frame():
    """One frame rotated by theta about its own x axis: pairs (k-1,k) and (k,k+1) each err theta."""
    n, k, theta = 8, 4, 5.0
    gt_c2w = _random_c2w(n, seed=1)
    pred_c2w = gt_c2w.copy()
    R_x = R.from_euler("x", theta, degrees=True).as_matrix()
    pred_c2w[k, :3, :3] = gt_c2w[k, :3, :3] @ R_x

    result = rpe(np.linalg.inv(pred_c2w), np.linalg.inv(gt_c2w), delta=1)

    # Two of the n-1 pairs carry theta, the rest zero
    assert result["rot_rmse_deg"] == pytest.approx(theta * np.sqrt(2 / (n - 1)), rel=1e-6)

    # Translation error lands on pair (k,k+1) only: frame k's center is unchanged
    # - that pair's step, seen from camera k, turns by R_x: |R_x^T u - u|, u = R_k^T (c_{k+1} - c_k)
    # - the wrong-frame formula (relative poses of w2c) gives a different value
    u = gt_c2w[k, :3, :3].T @ (gt_c2w[k + 1, :3, 3] - gt_c2w[k, :3, 3])
    expected_trans = np.linalg.norm(R_x.T @ u - u) / np.sqrt(n - 1)
    assert result["trans_rmse"] == pytest.approx(expected_trans, rel=1e-6)
