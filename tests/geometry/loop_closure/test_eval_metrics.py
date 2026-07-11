import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


def _make_poses(translations: np.ndarray) -> np.ndarray:
    """Build (N, 4, 4) world-to-cam poses with identity rotation and given translations."""
    N = len(translations)
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, :3, 3] = translations.astype(np.float32)
    return poses


def test_umeyama_align_identity():
    """When pred == gt, aligned_pred should equal pred (up to float tolerance)."""
    from collab_splats.geometry.loop_closure.eval import umeyama_align
    gt = _make_poses(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32))
    aligned, T = umeyama_align(gt.copy(), gt)
    np.testing.assert_allclose(aligned, gt, atol=1e-4)


def test_umeyama_align_pure_translation():
    """Pred shifted by constant offset; aligned_pred should match gt after alignment."""
    from collab_splats.geometry.loop_closure.eval import umeyama_align
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
    from collab_splats.geometry.loop_closure.eval import ate_translation
    gt = _make_poses(np.random.RandomState(0).randn(10, 3).astype(np.float32))
    result = ate_translation(gt.copy(), gt)
    assert result["rmse"] < 1e-4
    assert result["mean"] < 1e-4
    assert "per_frame" in result
    assert len(result["per_frame"]) == 10


def test_ate_translation_returns_correct_keys():
    from collab_splats.geometry.loop_closure.eval import ate_translation
    gt = _make_poses(np.zeros((5, 3), dtype=np.float32))
    result = ate_translation(gt.copy(), gt)
    assert set(result.keys()) >= {"rmse", "mean", "median", "max", "per_frame"}


def test_rpe_perfect():
    """When pred == gt, RPE trans and rot RMSE should be near zero."""
    from collab_splats.geometry.loop_closure.eval import rpe
    gt = _make_poses(np.linspace([0, 0, 0], [5, 0, 0], 10).astype(np.float32))
    result = rpe(gt.copy(), gt, delta=1)
    assert result["trans_rmse"] < 1e-4
    assert result["rot_rmse_deg"] < 1e-3


def test_rpe_returns_correct_keys():
    from collab_splats.geometry.loop_closure.eval import rpe
    gt = _make_poses(np.zeros((5, 3), dtype=np.float32))
    result = rpe(gt.copy(), gt)
    assert set(result.keys()) == {"trans_rmse", "rot_rmse_deg"}


def test_ate_nonzero_error():
    """Pred shifted uniformly; after Umeyama alignment, ATE should be near zero (translation-only drift is correctable)."""
    from collab_splats.geometry.loop_closure.eval import ate_translation
    gt_t = np.linspace([0, 0, 0], [4, 0, 0], 5).astype(np.float32)
    gt = _make_poses(gt_t)
    # Pred has a constant offset — Umeyama removes it, ATE → 0
    pred = _make_poses(gt_t + np.array([10, 0, 0], dtype=np.float32))
    result = ate_translation(pred, gt)
    assert result["rmse"] < 1e-3


def test_rpe_nonzero_translation_error():
    """Pred with accumulated drift has nonzero RPE even though ATE ~0 after alignment."""
    from collab_splats.geometry.loop_closure.eval import rpe
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
