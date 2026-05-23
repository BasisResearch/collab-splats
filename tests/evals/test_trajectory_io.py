"""Round-trip + format tests for evals/trajectory_io.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))


def _random_w2c(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rots = R.from_rotvec(rng.normal(size=(n, 3)) * 0.3).as_matrix()
    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    poses[:, :3, :3] = rots
    poses[:, :3, 3] = rng.normal(size=(n, 3))
    return poses.astype(np.float64)


def test_round_trip_w2c_through_tum(tmp_path):
    """numpy(N,4,4) world-to-cam → TUM file → numpy(N,4,4); err < 1e-6."""
    from trajectory_io import read_tum, write_tum

    poses = _random_w2c(8)
    p = tmp_path / "traj.tum"
    write_tum(p, poses)
    poses_back, ts = read_tum(p)

    assert poses_back.shape == (8, 4, 4)
    assert ts.shape == (8,)
    err = np.linalg.norm(poses - poses_back, axis=(1, 2)).max()
    assert err < 1e-6, f"max round-trip err {err}"


def test_tum_format_eight_columns(tmp_path):
    from trajectory_io import write_tum

    poses = _random_w2c(3)
    p = tmp_path / "traj.tum"
    write_tum(p, poses)
    lines = [ln for ln in p.read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 3
    for ln in lines:
        cols = ln.split()
        assert len(cols) == 8, f"line has {len(cols)} cols, expected 8: {ln!r}"


def test_tum_default_timestamps_are_frame_indices(tmp_path):
    """No timestamps provided → frame index in seconds (0.0, 1.0, ...)."""
    from trajectory_io import read_tum, write_tum

    poses = _random_w2c(5)
    p = tmp_path / "traj.tum"
    write_tum(p, poses)
    _, ts = read_tum(p)
    np.testing.assert_allclose(ts, np.arange(5, dtype=np.float64))


def test_tum_explicit_timestamps_round_trip(tmp_path):
    from trajectory_io import read_tum, write_tum

    poses = _random_w2c(4)
    ts = np.array([0.0, 0.5, 1.25, 2.5])
    p = tmp_path / "traj.tum"
    write_tum(p, poses, timestamps=ts)
    _, ts_back = read_tum(p)
    np.testing.assert_allclose(ts_back, ts)


def test_kitti_3x4_flat_to_w2c_identity():
    """3x4 identity (c2w=I) → w2c=I."""
    from trajectory_io import kitti_3x4_flat_to_w2c

    line = "1 0 0 0 0 1 0 0 0 0 1 0"
    poses = kitti_3x4_flat_to_w2c(line)
    assert poses.shape == (1, 4, 4)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-9)


def test_kitti_3x4_flat_to_w2c_translation():
    """c2w with t=(1,2,3) → w2c with t=(-1,-2,-3) for identity rotation."""
    from trajectory_io import kitti_3x4_flat_to_w2c

    line = "1 0 0 1  0 1 0 2  0 0 1 3"
    poses = kitti_3x4_flat_to_w2c(line)
    assert poses.shape == (1, 4, 4)
    expected = np.eye(4)
    expected[:3, 3] = [-1.0, -2.0, -3.0]
    np.testing.assert_allclose(poses[0], expected, atol=1e-9)


def test_kitti_file_multi_line(tmp_path):
    from trajectory_io import kitti_file_to_w2c

    p = tmp_path / "00.txt"
    p.write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n"
        "1 0 0 1 0 1 0 0 0 0 1 0\n"
    )
    poses = kitti_file_to_w2c(p)
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-9)
    np.testing.assert_allclose(poses[1, :3, 3], [-1.0, 0.0, 0.0], atol=1e-9)


def test_w2c_to_kitti_round_trip():
    from trajectory_io import kitti_3x4_flat_to_w2c, w2c_to_kitti_3x4_flat

    poses = _random_w2c(6, seed=42)
    text = w2c_to_kitti_3x4_flat(poses)
    poses_back = kitti_3x4_flat_to_w2c(text)
    err = np.linalg.norm(poses - poses_back, axis=(1, 2)).max()
    # KITTI text format is %.9f → ~1e-9 per component; matrix-inverse op stacks
    # numerical noise, so tolerate ~1e-8.
    assert err < 1e-8, f"KITTI round-trip err {err}"


def test_kitti_file_to_tum(tmp_path):
    """End-to-end: KITTI file → TUM file → numpy round-trips."""
    from trajectory_io import kitti_file_to_w2c, read_tum, write_tum

    poses_in = _random_w2c(5, seed=7)
    kitti_path = tmp_path / "00.txt"
    from trajectory_io import w2c_to_kitti_3x4_flat
    kitti_path.write_text(w2c_to_kitti_3x4_flat(poses_in))

    poses_mid = kitti_file_to_w2c(kitti_path)
    tum_path = tmp_path / "00.tum"
    write_tum(tum_path, poses_mid)
    poses_out, _ = read_tum(tum_path)

    err = np.linalg.norm(poses_in - poses_out, axis=(1, 2)).max()
    assert err < 1e-6


def test_write_tum_creates_parent_dirs(tmp_path):
    from trajectory_io import write_tum

    p = tmp_path / "nested" / "dir" / "traj.tum"
    write_tum(p, _random_w2c(2))
    assert p.exists()


def test_read_tum_skips_comment_and_blank_lines(tmp_path):
    from trajectory_io import read_tum

    p = tmp_path / "traj.tum"
    p.write_text(
        "# header comment\n"
        "\n"
        "0.0 0 0 0 0 0 0 1\n"
        "# another\n"
        "1.0 1 0 0 0 0 0 1\n"
    )
    poses, ts = read_tum(p)
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(ts, [0.0, 1.0])


def test_write_tum_raises_on_wrong_shape(tmp_path):
    from trajectory_io import write_tum

    with pytest.raises(ValueError, match="shape"):
        write_tum(tmp_path / "bad.tum", np.zeros((3, 3)))


def test_write_tum_raises_on_timestamp_length_mismatch(tmp_path):
    from trajectory_io import write_tum

    with pytest.raises(ValueError, match="timestamps"):
        write_tum(tmp_path / "bad.tum", _random_w2c(3), timestamps=np.array([0.0, 1.0]))
