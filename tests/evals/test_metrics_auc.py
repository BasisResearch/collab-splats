"""Tests for compute_auc in evals/metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
from metrics import compute_auc


def _write_tum(path, poses_w2c):
    from scipy.spatial.transform import Rotation

    lines = []
    for i, w2c in enumerate(poses_w2c):
        c2w = np.linalg.inv(w2c.astype(np.float64))
        t = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        lines.append(
            f"{i:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def test_compute_auc_perfect():
    """Perfect prediction = AUC@30 of ~100.

    Poses start away from the origin so no frame has a degenerate (zero-norm)
    camera-centre vector, which would produce a 90° translation-direction error
    even for identical pred/gt.
    """
    import tempfile

    N = 20
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    for i in range(N):
        # offset by 1.0 so no frame sits at the world origin
        poses[i, :3, 3] = [1.0 + i * 0.1, 0, 0]

    with tempfile.TemporaryDirectory() as tmp:
        pred_path = Path(tmp) / "pred.tum"
        gt_path = Path(tmp) / "gt.tum"
        _write_tum(pred_path, poses)
        _write_tum(gt_path, poses)
        result = compute_auc(pred_path, gt_path)

    assert result["auc_30"] == pytest.approx(100.0, abs=1.0)
    assert "per_pair_err" in result


def test_compute_auc_returns_lower_for_noisy():
    """Noisy predictions produce AUC@30 strictly below 100.

    Uses 3D spiral translations so evo alignment is non-degenerate.
    Adds large rotation noise (>30°) to push most frames over the threshold.
    auc_at_threshold performs its own Umeyama Sim3 alignment internally.
    """
    import tempfile

    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(42)
    N = 30
    gt_poses = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
    for i in range(N):
        # 3D spiral so alignment is non-degenerate
        gt_poses[i, :3, 3] = [i * 0.1, np.sin(i * 0.3) * 0.5, np.cos(i * 0.3) * 0.5]

    # Add large rotation noise (40–80°) so errors clearly exceed 30° threshold
    pred_poses = gt_poses.copy()
    for i in range(N):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.7, 1.4)  # 40–80 degrees
        R_noise = Rotation.from_rotvec(axis * angle).as_matrix()
        pred_poses[i, :3, :3] = R_noise @ gt_poses[i, :3, :3]

    with tempfile.TemporaryDirectory() as tmp:
        pred_path = Path(tmp) / "pred.tum"
        gt_path = Path(tmp) / "gt.tum"
        _write_tum(pred_path, pred_poses)
        _write_tum(gt_path, gt_poses)
        result = compute_auc(pred_path, gt_path)

    assert result["auc_30"] < 90.0
    assert "per_pair_err" in result
