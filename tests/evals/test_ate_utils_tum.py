"""TUM groundtruth support in ate_utils."""
import sys
from pathlib import Path

import numpy as np

# Local `evals/` is shadowed by an installed `evals` pip package; insert the
# evals dir on sys.path and import the module directly (sibling-test convention).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from ate_utils import _load_gt_as_tum_trajectory


def _make_tum_scene(tmp_path: Path) -> Path:
    (tmp_path / "rgb").mkdir()
    # 3 keyframes at timestamps 10.0, 10.5, 11.0
    for ts in ("10.000000", "10.500000", "11.000000"):
        (tmp_path / "rgb" / f"{ts}.png").touch()
    # GT at slightly offset timestamps (nearest-neighbour association required)
    gt_lines = ["# ts tx ty tz qx qy qz qw"]
    for i, ts in enumerate((9.99, 10.51, 11.01)):
        gt_lines.append(f"{ts} {i}.0 0.0 0.0 0.0 0.0 0.0 1.0")
    (tmp_path / "groundtruth.txt").write_text("\n".join(gt_lines))
    return tmp_path


def test_tum_gt_loaded_and_associated(tmp_path):
    seq = _make_tum_scene(tmp_path)
    frames = sorted((seq / "rgb").iterdir())
    traj = _load_gt_as_tum_trajectory(seq, selected_frames=frames)
    assert traj.num_poses == 3
    # x positions 0,1,2 from the associated GT rows
    np.testing.assert_allclose(traj.positions_xyz[:, 0], [0.0, 1.0, 2.0])


def test_tum_gt_timestamps_match_frame_filenames(tmp_path):
    # Trajectory timestamps must mirror the pred file's frame-derived timestamps
    # (VGGT-SLAM writes float(frame_id) parsed from the image filename), not the
    # raw (offset) groundtruth.txt row timestamps, so evo can associate pred/GT.
    seq = _make_tum_scene(tmp_path)
    frames = sorted((seq / "rgb").iterdir())
    traj = _load_gt_as_tum_trajectory(seq, selected_frames=frames)
    np.testing.assert_allclose(traj.timestamps, [10.0, 10.5, 11.0])


def _make_tum_scene_with_gap(tmp_path: Path) -> Path:
    (tmp_path / "rgb").mkdir()
    # 3 keyframes at timestamps 10.0, 10.5, 11.0
    for ts in ("10.000000", "10.500000", "11.000000"):
        (tmp_path / "rgb" / f"{ts}.png").touch()
    # GT only covers frames 1 and 3; the middle frame's nearest GT row (either one)
    # is 0.5s away — well past the 0.02s max-gap threshold — so it must be excluded.
    gt_lines = ["# ts tx ty tz qx qy qz qw"]
    for i, ts in ((0, 9.99), (2, 11.01)):
        gt_lines.append(f"{ts} {i}.0 0.0 0.0 0.0 0.0 0.0 1.0")
    (tmp_path / "groundtruth.txt").write_text("\n".join(gt_lines))
    return tmp_path


def test_tum_gt_gap_frame_excluded(tmp_path):
    # A frame whose nearest GT row is >0.02s away (mocap gap, e.g. fr1_room) must be
    # dropped rather than associated with bogus GT — matches datasets.py:_load_tum's
    # own rgb/GT matching threshold.
    seq = _make_tum_scene_with_gap(tmp_path)
    frames = sorted((seq / "rgb").iterdir())
    traj = _load_gt_as_tum_trajectory(seq, selected_frames=frames)
    assert traj.num_poses == 2
