import numpy as np
import pytest
from pathlib import Path


def _make_seq(tmp_path: Path, n_frames: int, poses: list[np.ndarray] | None = None) -> Path:
    """Create a minimal synthetic 7-Scenes sequence directory (flat layout)."""
    for i in range(n_frames):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
        p = poses[i] if poses else np.eye(4, dtype=np.float64)
        np.savetxt(tmp_path / f"frame-{i:06d}.pose.txt", p)
    return tmp_path


def _make_tum_seq(
    tmp_path: Path,
    rgb_entries: list[tuple[float, str]],
    gt_entries: list[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]],
) -> Path:
    """Create a minimal synthetic TUM RGB-D sequence directory.

    rgb_entries: (timestamp, filename) pairs.
    gt_entries:  (timestamp, (tx,ty,tz), (qx,qy,qz,qw)) tuples.
    """
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for _, fname in rgb_entries:
        (rgb_dir / fname).touch()

    rgb_lines = ["# color images\n", "# timestamp filename\n"]
    rgb_lines += [f"{ts:.6f} rgb/{fname}\n" for ts, fname in rgb_entries]
    (tmp_path / "rgb.txt").write_text("".join(rgb_lines))

    gt_lines = ["# ground truth trajectory\n", "# timestamp tx ty tz qx qy qz qw\n"]
    for ts, (tx, ty, tz), (qx, qy, qz, qw) in gt_entries:
        gt_lines.append(f"{ts:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
    (tmp_path / "groundtruth.txt").write_text("".join(gt_lines))

    # depth.txt is referenced by TUM convention; create a stub file for completeness.
    (tmp_path / "depth.txt").write_text("# depth\n# timestamp filename\n")
    return tmp_path


def _make_waymo_seq(tmp_path: Path, n_frames: int, poses_c2w: list[np.ndarray] | None = None) -> Path:
    """Synthesize a pre-extracted Waymo segment (images/ + groundtruth.txt)."""
    seq = tmp_path / "seg"
    (seq / "images").mkdir(parents=True)
    for i in range(n_frames):
        (seq / "images" / f"{i:06d}.png").touch()
    from scipy.spatial.transform import Rotation
    lines = []
    for i in range(n_frames):
        T = poses_c2w[i] if poses_c2w else np.eye(4)
        t = T[:3, 3]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        lines.append(
            f"{float(i):.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
    (seq / "groundtruth.txt").write_text("\n".join(lines) + "\n")
    return seq


def test_load_waymo_count(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    seq = _make_waymo_seq(tmp_path, n_frames=8)
    dataset = get_dataset("waymo")(seq, max_frames=8)
    assert len(dataset.images) == 8
    assert dataset.gt_poses.shape == (8, 4, 4)


def test_load_waymo_max_frames(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    seq = _make_waymo_seq(tmp_path, n_frames=8)
    dataset = get_dataset("waymo")(seq, max_frames=3)
    assert len(dataset.images) == 3
    assert dataset.gt_poses.shape == (3, 4, 4)


def test_load_waymo_pose_inverted(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    seq = _make_waymo_seq(tmp_path, n_frames=1, poses_c2w=[c2w])
    dataset = get_dataset("waymo")(seq, max_frames=1)
    expected = np.eye(4, dtype=np.float32)
    expected[:3, 3] = [-1.0, -2.0, -3.0]
    np.testing.assert_allclose(dataset.gt_poses[0], expected, atol=1e-5)


def test_load_waymo_dtype(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    seq = _make_waymo_seq(tmp_path, n_frames=3)
    dataset = get_dataset("waymo")(seq)
    assert dataset.gt_poses.dtype == np.float32


def test_load_waymo_misaligned_raises(tmp_path):
    """If groundtruth has fewer entries than images, fail loud."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    seq = _make_waymo_seq(tmp_path, n_frames=5)
    gt_path = seq / "groundtruth.txt"
    lines = gt_path.read_text().splitlines()
    gt_path.write_text("\n".join(lines[:2]) + "\n")
    with pytest.raises(ValueError, match="misaligned"):
        get_dataset("waymo")(seq)


def test_get_dataset_unknown_raises():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    with pytest.raises(KeyError, match="unknown dataset"):
        get_dataset("nonexistent")


def test_load_7scenes_count(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_seq(tmp_path, n_frames=10)
    dataset = get_dataset("7scenes")(tmp_path, max_frames=10)
    assert len(dataset.images) == 10
    assert dataset.gt_poses.shape == (10, 4, 4)


def test_load_7scenes_max_frames(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_seq(tmp_path, n_frames=10)
    dataset = get_dataset("7scenes")(tmp_path, max_frames=5)
    assert len(dataset.images) == 5
    assert dataset.gt_poses.shape == (5, 4, 4)


def test_load_7scenes_pose_inverted(tmp_path):
    """7-Scenes poses are cam-to-world; loader must invert to world-to-cam."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, 3] = [1.0, 2.0, 3.0]
    _make_seq(tmp_path, n_frames=1, poses=[cam_to_world])
    dataset = get_dataset("7scenes")(tmp_path, max_frames=1)
    expected = np.eye(4, dtype=np.float32)
    expected[:3, 3] = [-1.0, -2.0, -3.0]
    np.testing.assert_allclose(dataset.gt_poses[0], expected, atol=1e-5)


def test_load_7scenes_dtype(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_seq(tmp_path, n_frames=3)
    dataset = get_dataset("7scenes")(tmp_path)
    assert dataset.gt_poses.dtype == np.float32


def test_load_7scenes_works_for_any_scene_name(tmp_path):
    """Loader is scene-name-agnostic: only the flat seq-NN/{frame-*.color.png,frame-*.pose.txt}
    layout matters. Documents the multi-scene contract used by the download script
    (chess/fire/office validated for VGGT-SLAM-comparable evals).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    # Mimic real 7-Scenes layout literally: <scene>/seq-01/frame-*.{color.png,pose.txt}
    seq_dir = tmp_path / "fire" / "seq-01"
    seq_dir.mkdir(parents=True)
    _make_seq(seq_dir, n_frames=4)
    dataset = get_dataset("7scenes")(seq_dir, max_frames=4)
    assert len(dataset.images) == 4
    assert dataset.gt_poses.shape == (4, 4, 4)


# ----------------------------- TUM RGB-D tests ----------------------------- #


def test_load_tum_count(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    rgb = [(float(i) * 0.033, f"{i:06d}.png") for i in range(5)]
    gt = [(float(i) * 0.033, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) for i in range(5)]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path, max_frames=5)
    assert len(dataset.images) == 5
    assert dataset.gt_poses.shape == (5, 4, 4)


def test_load_tum_pose_inverted(tmp_path):
    """TUM gt is cam-to-world; loader must invert to world-to-cam."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    rgb = [(0.0, "000000.png")]
    # identity rotation, translation (1, 0, 0)
    gt = [(0.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path, max_frames=1)
    np.testing.assert_allclose(dataset.gt_poses[0, :3, 3], np.array([-1.0, 0.0, 0.0]), atol=1e-5)


def test_load_tum_max_frames(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    rgb = [(float(i) * 0.033, f"{i:06d}.png") for i in range(10)]
    gt = [(float(i) * 0.033, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) for i in range(10)]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path, max_frames=4)
    assert len(dataset.images) == 4
    assert dataset.gt_poses.shape == (4, 4, 4)


def test_load_tum_dtype(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    rgb = [(float(i) * 0.033, f"{i:06d}.png") for i in range(3)]
    gt = [(float(i) * 0.033, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) for i in range(3)]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path)
    assert dataset.gt_poses.dtype == np.float32


def test_load_tum_associates_rgb_to_nearest_gt(tmp_path):
    """rgb ts 0.51 with gt ts {0.5, 0.55} → nearest is 0.5 (|0.01|<|0.04|)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    rgb = [(0.51, "000000.png")]
    # Two gt entries; nearest should be 0.5 (translation (7,0,0)) over 0.55 (translation (9,0,0))
    gt = [
        (0.50, (7.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (0.55, (9.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path, max_frames=1)
    # w2c translation = -c2w translation for identity rotation
    np.testing.assert_allclose(dataset.gt_poses[0, :3, 3], np.array([-7.0, 0.0, 0.0]), atol=1e-5)


# ----------------------------- KITTI Odometry tests ----------------------------- #


def _make_kitti_seq(
    seq_dir: Path,
    n_frames: int,
    poses_lines: list[str] | None = None,
    write_poses: bool = True,
) -> Path:
    """Create a minimal synthetic KITTI Odometry sequence directory.

    Layout:
        seq_dir/image_2/{000000.png ... 00000{N-1}.png}
        seq_dir/poses.txt   (optional; 12-float 3x4 cam-to-world per line)
    """
    img_dir = seq_dir / "image_2"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        (img_dir / f"{i:06d}.png").touch()
    if write_poses:
        if poses_lines is None:
            poses_lines = ["1 0 0 0 0 1 0 0 0 0 1 0"] * n_frames
        (seq_dir / "poses.txt").write_text("\n".join(poses_lines) + "\n")
    return seq_dir


def test_load_kitti_count(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_kitti_seq(tmp_path, n_frames=5)
    dataset = get_dataset("kitti")(tmp_path, max_frames=10)
    assert len(dataset.images) == 5
    assert dataset.gt_poses.shape == (5, 4, 4)


def test_load_kitti_max_frames(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_kitti_seq(tmp_path, n_frames=10)
    dataset = get_dataset("kitti")(tmp_path, max_frames=4)
    assert len(dataset.images) == 4
    assert dataset.gt_poses.shape == (4, 4, 4)


def test_load_kitti_pose_inverted(tmp_path):
    """KITTI poses are cam-to-world; loader must invert to world-to-cam."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    # cam-to-world identity rotation with translation (1, 0, 0); w2c translation → (-1, 0, 0).
    _make_kitti_seq(tmp_path, n_frames=1, poses_lines=["1 0 0 1 0 1 0 0 0 0 1 0"])
    dataset = get_dataset("kitti")(tmp_path, max_frames=1)
    np.testing.assert_allclose(dataset.gt_poses[0, :3, 3], np.array([-1.0, 0.0, 0.0]), atol=1e-5)


def test_load_kitti_dtype(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_kitti_seq(tmp_path, n_frames=3)
    dataset = get_dataset("kitti")(tmp_path)
    assert dataset.gt_poses.dtype == np.float32


def test_load_kitti_alt_poses_path(tmp_path):
    """Loader falls back to <root>/poses/<NN>.txt when seq_dir/poses.txt is absent."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    seq_dir = tmp_path / "sequences" / "00"
    _make_kitti_seq(seq_dir, n_frames=2, write_poses=False)
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    (poses_dir / "00.txt").write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    dataset = get_dataset("kitti")(seq_dir, max_frames=2)
    assert len(dataset.images) == 2
    assert dataset.gt_poses.shape == (2, 4, 4)


def test_load_kitti_missing_poses_raises(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    seq_dir = tmp_path / "sequences" / "00"
    _make_kitti_seq(seq_dir, n_frames=2, write_poses=False)
    with pytest.raises(FileNotFoundError) as ei:
        get_dataset("kitti")(seq_dir, max_frames=2)
    msg = str(ei.value)
    assert "poses.txt" in msg
    assert "00.txt" in msg
