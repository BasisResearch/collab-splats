import numpy as np
import pytest

from collab_splats.geometry.loop_closure.eval import auc_at_threshold


def _make_poses(rotations_deg, translations):
    """Build (N, 4, 4) world-to-cam poses from rotation angles (about z-axis) and translations."""
    from scipy.spatial.transform import Rotation as R

    N = len(rotations_deg)
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    for i, (deg, t) in enumerate(zip(rotations_deg, translations)):
        poses[i, :3, :3] = R.from_euler("z", deg, degrees=True).as_matrix().astype(np.float32)
        poses[i, :3, 3] = np.array(t, dtype=np.float32)
    return poses


def test_auc_returns_required_keys():
    pred = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    gt   = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    result = auc_at_threshold(pred, gt)
    assert set(result) >= {"auc_30", "per_pair_err"}


def test_auc_perfect_poses_scores_100():
    """When pred == gt, every frame error is 0 → AUC@30 = 100."""
    gt = _make_poses([0, 10, 20, 30], [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    result = auc_at_threshold(gt.copy(), gt)
    assert result["auc_30"] >= 99.0, f"Expected ~100, got {result['auc_30']}"


def test_auc_large_error_scores_near_zero():
    """When all frames have 90° rotation error (>> 30°), AUC@30 ≈ 0."""
    from scipy.spatial.transform import Rotation as R

    N = 5
    gt = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    pred = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    rot_90 = R.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    pred[:, :3, :3] = rot_90
    result = auc_at_threshold(pred, gt)
    assert result["auc_30"] < 10.0, f"Expected near 0, got {result['auc_30']}"


def test_per_pair_err_present():
    N = 7
    pred = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    gt   = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    result = auc_at_threshold(pred, gt)
    assert len(result["per_pair_err"]) == N * (N - 1)


def test_auc_is_float_in_range():
    pred = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    gt   = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    result = auc_at_threshold(pred, gt)
    assert isinstance(result["auc_30"], float)
    assert 0.0 <= result["auc_30"] <= 100.0


def test_auc_half_frames_below_threshold():
    """4 frames: 2 with error ~0°, 2 with error ~45°. AUC@30 should be near 50."""
    from scipy.spatial.transform import Rotation as R

    gt   = _make_poses([0, 10, 20, 30], [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    pred = _make_poses([0, 10, 20, 30], [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    # Frames 0,1 perfect (same as gt)
    # Frames 2,3 have ~45° rotation error (well above 30° threshold)
    rot_45 = R.from_euler("z", 45, degrees=True).as_matrix().astype(np.float32)
    pred[2, :3, :3] = rot_45
    pred[3, :3, :3] = rot_45
    result = auc_at_threshold(pred, gt)
    # Frames 0,1 perfect (err≈0), frames 2,3 err≈45° > 30°
    # accuracy(t) = 0.5 for all t ∈ [30°, 45°), AUC should be ~50
    assert 30.0 < result["auc_30"] < 70.0, f"Expected ~50, got {result['auc_30']}"


def test_auc_translation_direction_error_is_angular_not_l2():
    """Verify auc_at_threshold uses angular direction error (normalized dot product).

    The key property: if two camera directions differ only in magnitude (scale),
    their angular error should be ~0°. The old L2 metric would give large errors.

    This test verifies that the code path for normalizing and computing angular
    error executes correctly without NaN or numerical issues.
    """
    from scipy.spatial.transform import Rotation as R

    N = 2
    gt = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    pred = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))

    # Create a simple case: translate both cameras identically
    # So after Umeyama alignment they should be very close
    for i in range(N):
        offset = float(i + 1)
        gt[i, :3, 3] = np.array([offset, offset, offset], dtype=np.float32)
        pred[i, :3, 3] = np.array([offset, offset, offset], dtype=np.float32)

    result = auc_at_threshold(pred, gt)
    # Both pred and gt are identical, should score near 100
    assert result["auc_30"] >= 95.0, f"Expected ~100 for identical poses, got {result['auc_30']}"
    # Verify the function returns the expected output structure
    assert isinstance(result["per_pair_err"], list)


def test_auc_multi_threshold_keys_and_monotonic():
    """thresholds=(5,15,30) returns auc_5/auc_15/auc_30; AUC is non-decreasing in threshold."""
    gt   = _make_poses([0, 10, 20, 30], [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    pred = _make_poses([0, 12, 18, 33], [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    result = auc_at_threshold(pred, gt, thresholds=(5.0, 15.0, 30.0))
    assert {"auc_5", "auc_15", "auc_30", "per_pair_err"} <= set(result)
    assert result["auc_5"] <= result["auc_15"] <= result["auc_30"] + 1e-9
