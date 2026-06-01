import json
import numpy as np
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))


def _make_poses(N=5):
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, 0, 3] = np.arange(N, dtype=np.float32)
    return poses


def test_prepare_image_dir_creates_symlinks(tmp_path):
    from eval_gt import _prepare_image_dir
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    images = []
    for i in range(3):
        p = src_dir / f"frame-{i:06d}.png"
        p.touch()
        images.append(p)
    result = _prepare_image_dir(images)
    try:
        links = sorted(result.iterdir())
        assert len(links) == 3
        assert all(l.is_symlink() for l in links)
        assert links[0].name == "000000.png"
        assert links[2].name == "000002.png"
    finally:
        shutil.rmtree(result)


def test_prepare_image_dir_ordered_by_index(tmp_path):
    """Symlinks are named 000000.png..N so VGGTXCreator sees correct frame order."""
    from eval_gt import _prepare_image_dir
    images = [tmp_path / f"z_{i}.png" for i in range(5)]
    for p in images:
        p.touch()
    result = _prepare_image_dir(images)
    try:
        names = sorted(p.name for p in result.iterdir())
        assert names == [f"{i:06d}.png" for i in range(5)]
    finally:
        shutil.rmtree(result)


def test_cam_positions_identity():
    """Identity world-to-cam → camera at origin."""
    from eval_gt import _cam_positions
    poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    pos = _cam_positions(poses)
    np.testing.assert_allclose(pos, np.zeros((3, 3)), atol=1e-6)


def test_cam_positions_translation():
    """world-to-cam with t=(1,2,3) → cam at (-1,-2,-3) in world (identity R)."""
    from eval_gt import _cam_positions
    poses = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1))
    poses[0, :3, 3] = [1.0, 2.0, 3.0]
    pos = _cam_positions(poses)
    np.testing.assert_allclose(pos[0], [-1.0, -2.0, -3.0], atol=1e-6)


def test_save_outputs_writes_files(tmp_path):
    from eval_gt import _save_outputs
    gt = _make_poses(10)
    trajectories = {"gt": gt, "baseline": gt.copy(), "ba": gt.copy()}
    metrics = {
        "baseline": {"ate": {"rmse": 0.1, "mean": 0.09, "median": 0.08, "max": 0.2, "per_frame": np.zeros(10)}, "rpe": {"trans_rmse": 0.01, "rot_rmse_deg": 0.5}, "auc": {"auc_30": 75.0}},
        "ba":       {"ate": {"rmse": 0.05, "mean": 0.04, "median": 0.03, "max": 0.1, "per_frame": np.zeros(10)}, "rpe": {"trans_rmse": 0.005, "rot_rmse_deg": 0.2}, "auc": {"auc_30": 90.0}},
    }
    _save_outputs(metrics, trajectories, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "trajectories.npz").exists()
    assert (tmp_path / "plots" / "trajectory.png").exists()
    assert (tmp_path / "plots" / "ate_per_frame.png").exists()
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert "baseline" in loaded
    assert "rmse" in loaded["baseline"]["ate"]
    npz = np.load(tmp_path / "trajectories.npz", allow_pickle=False)
    assert "ate_per_frame_baseline" in npz.files
    assert "ate_per_frame_ba" in npz.files
    assert npz["ate_per_frame_baseline"].shape == (10,)


def test_make_creator_ba_track_density_4096(monkeypatch):
    """ba_track-density-4096 → BA enabled with max_query_pts=4096, query_frame_num=8."""
    import eval_gt
    from collab_splats.pointcloud import BundleAdjustmentConfig
    from unittest.mock import MagicMock
    monkeypatch.setattr(eval_gt, "get_creator", lambda name: lambda: MagicMock())
    creator, ba_cfg = eval_gt._make_creator("ba_track-density-4096")
    assert isinstance(ba_cfg, BundleAdjustmentConfig)  # BA enabled
    assert ba_cfg.max_query_pts == 4096
    assert ba_cfg.query_frame_num == 8  # max(5, 4096 // 512)


def test_make_creator_ba_track_density_2048(monkeypatch):
    """ba_track-density-2048 → query_frame_num=5 (max(5, 2048//512) = max(5,4) = 5)."""
    import eval_gt
    from collab_splats.pointcloud import BundleAdjustmentConfig
    from unittest.mock import MagicMock
    monkeypatch.setattr(eval_gt, "get_creator", lambda name: lambda: MagicMock())
    creator, ba_cfg = eval_gt._make_creator("ba_track-density-2048")
    assert isinstance(ba_cfg, BundleAdjustmentConfig)  # BA enabled
    assert ba_cfg.max_query_pts == 2048
    assert ba_cfg.query_frame_num == 5


def test_validate_condition_accepts_known():
    from eval_gt import _validate_condition
    for cond in ["baseline", "ba", "lc", "ba_track-density-4096", "ba_track-density-1024"]:
        _validate_condition(cond)  # must not raise


def test_validate_condition_rejects_ba_hightrack():
    """ba_hightrack is no longer valid — was replaced by ba_track-density-N."""
    import pytest
    from eval_gt import _validate_condition
    with pytest.raises(ValueError, match="ba_hightrack"):
        _validate_condition("ba_hightrack")


def test_validate_condition_rejects_unknown():
    import pytest
    from eval_gt import _validate_condition
    with pytest.raises(ValueError):
        _validate_condition("mystery_condition")


def test_default_output_dir_structure():
    """Auto path: evals/results/{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}/"""
    import re
    from eval_gt import _default_output_dir
    result = _default_output_dir("co3dv2", Path("/data/co3dv2/apple/seq1"))
    parts = list(result.parts)
    idx = parts.index("results")
    assert parts[idx - 1] == "evals"
    assert parts[idx + 1] == "co3dv2"
    assert parts[idx + 2] == "seq1"
    assert re.fullmatch(r"run-\d{8}-\d{6}", parts[idx + 3])


def test_default_output_dir_uses_basename():
    """seq_name = last component of seq_dir."""
    from eval_gt import _default_output_dir
    r1 = _default_output_dir("7scenes", Path("/long/path/chess"))
    r2 = _default_output_dir("7scenes", Path("/other/chess"))
    assert r1.parent.name == "chess"
    assert r2.parent.name == "chess"


def test_save_outputs_no_per_frame_in_json(tmp_path):
    """per_frame array must not appear in metrics.json (not JSON-serializable)."""
    from eval_gt import _save_outputs
    from collab_splats.pointcloud.loop_closure.eval import ate_translation
    gt = _make_poses(10)
    metrics = {
        "baseline": {
            "ate": ate_translation(gt.copy(), gt),
            "rpe": {"trans_rmse": 0.0, "rot_rmse_deg": 0.0},
            "auc": {"auc_30": 100.0},
        }
    }
    trajectories = {"gt": gt, "baseline": gt}
    _save_outputs(metrics, trajectories, tmp_path)
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert "per_frame" not in loaded["baseline"]["ate"]


def test_save_outputs_persists_all_auc_thresholds(tmp_path):
    """metrics.json keeps the full AUC dict (auc_5/15/30) + RPE rotation."""
    from eval_gt import _save_outputs
    poses = np.tile(np.eye(4), (3, 1, 1))
    metrics = {"baseline": {
        "ate": {"rmse": 0.01, "per_frame": np.zeros(3)},
        "rpe": {"trans_rmse": 0.02, "rot_rmse_deg": 1.5},
        "auc": {"auc_5": 10.0, "auc_15": 50.0, "auc_30": 80.0, "per_pair_err": [0.0]},
        "time_s": 1,
    }}
    _save_outputs(metrics, {"gt": poses, "baseline": poses}, tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text())["baseline"]
    assert saved["auc"] == {"auc_5": 10.0, "auc_15": 50.0, "auc_30": 80.0}
    assert saved["rpe"]["rot_rmse_deg"] == 1.5
