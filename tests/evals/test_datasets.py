"""
Tests for evals.datasets.
"""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from evals.datasets import get_dataset, load_gt_depth


def _make_seq(tmp_path: Path, n_frames: int, poses: list[np.ndarray] | None = None) -> Path:
    """Create a minimal synthetic 7-Scenes sequence directory (flat layout)."""
    for i in range(n_frames):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
        (tmp_path / f"frame-{i:06d}.depth.png").touch()
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


def test_get_dataset_unknown_raises():
    with pytest.raises(KeyError, match="unknown dataset"):
        get_dataset("nonexistent")


def test_get_dataset_names_valid_types():
    with pytest.raises(KeyError, match="7scenes"):
        get_dataset("kitti")


def test_load_7scenes_count(tmp_path):
    _make_seq(tmp_path, n_frames=10)
    dataset = get_dataset("7scenes")(tmp_path, max_frames=10)
    assert len(dataset.images) == 10
    assert dataset.gt_poses.shape == (10, 4, 4)


def test_load_7scenes_max_frames(tmp_path):
    _make_seq(tmp_path, n_frames=10)
    dataset = get_dataset("7scenes")(tmp_path, max_frames=5)
    assert len(dataset.images) == 5
    assert dataset.gt_poses.shape == (5, 4, 4)


def test_load_7scenes_pose_inverted(tmp_path):
    """7-Scenes poses are cam-to-world; loader must invert to world-to-cam."""
    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, 3] = [1.0, 2.0, 3.0]
    _make_seq(tmp_path, n_frames=1, poses=[cam_to_world])
    dataset = get_dataset("7scenes")(tmp_path, max_frames=1)
    expected = np.eye(4, dtype=np.float32)
    expected[:3, 3] = [-1.0, -2.0, -3.0]
    np.testing.assert_allclose(dataset.gt_poses[0], expected, atol=1e-5)


def test_load_7scenes_dtype(tmp_path):
    _make_seq(tmp_path, n_frames=3)
    dataset = get_dataset("7scenes")(tmp_path)
    assert dataset.gt_poses.dtype == np.float32


def test_load_7scenes_depth_paths(tmp_path):
    _make_seq(tmp_path, n_frames=3)
    dataset = get_dataset("7scenes")(tmp_path)
    assert [p.name for p in dataset.depth_paths] == [f"frame-{i:06d}.depth.png" for i in range(3)]
    assert all(p.exists() for p in dataset.depth_paths)


def test_load_gt_depth_meters_and_invalid(tmp_path):
    raw = np.array([[1000, 65535], [0, 2500]], dtype=np.uint16)
    cv2.imwrite(str(tmp_path / "d.png"), raw)
    np.testing.assert_allclose(load_gt_depth(tmp_path / "d.png"), [[1.0, 0.0], [0.0, 2.5]])


def test_load_7scenes_works_for_any_scene_name(tmp_path):
    """Loader is scene-name-agnostic: only the flat seq-NN/{frame-*.color.png,frame-*.pose.txt}
    layout matters. Documents the multi-scene contract used by the download script
    (chess/fire/office).
    """
    # Mimic real 7-Scenes layout literally: <scene>/seq-01/frame-*.{color.png,pose.txt}
    seq_dir = tmp_path / "fire" / "seq-01"
    seq_dir.mkdir(parents=True)
    _make_seq(seq_dir, n_frames=4)
    dataset = get_dataset("7scenes")(seq_dir, max_frames=4)
    assert len(dataset.images) == 4
    assert dataset.gt_poses.shape == (4, 4, 4)


# ----------------------------- TUM RGB-D tests ----------------------------- #


def test_load_tum_count(tmp_path):
    rgb = [(float(i) * 0.033, f"{i:06d}.png") for i in range(5)]
    gt = [(float(i) * 0.033, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) for i in range(5)]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path, max_frames=5)
    assert len(dataset.images) == 5
    assert dataset.gt_poses.shape == (5, 4, 4)


def test_load_tum_pose_inverted(tmp_path):
    """TUM gt is cam-to-world; loader must invert to world-to-cam."""
    rgb = [(0.0, "000000.png")]
    # identity rotation, translation (1, 0, 0)
    gt = [(0.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path, max_frames=1)
    np.testing.assert_allclose(dataset.gt_poses[0, :3, 3], np.array([-1.0, 0.0, 0.0]), atol=1e-5)


def test_load_tum_max_frames(tmp_path):
    rgb = [(float(i) * 0.033, f"{i:06d}.png") for i in range(10)]
    gt = [(float(i) * 0.033, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) for i in range(10)]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path, max_frames=4)
    assert len(dataset.images) == 4
    assert dataset.gt_poses.shape == (4, 4, 4)


def test_load_tum_dtype(tmp_path):
    rgb = [(float(i) * 0.033, f"{i:06d}.png") for i in range(3)]
    gt = [(float(i) * 0.033, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) for i in range(3)]
    _make_tum_seq(tmp_path, rgb, gt)
    dataset = get_dataset("tum")(tmp_path)
    assert dataset.gt_poses.dtype == np.float32


def test_load_tum_associates_rgb_to_nearest_gt(tmp_path):
    """rgb ts 0.51 with gt ts {0.5, 0.55} → nearest is 0.5 (|0.01|<|0.04|)."""
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


# ----------------------------- CO3Dv2 tests ----------------------------- #


def _make_co3dv2_seq(tmp_path: Path, n_frames: int = 4, seq_name: str = "seq1") -> Path:
    """Category dir holding frame_annotations.jgz and one sequence of images."""
    seq_dir = tmp_path / "apple" / seq_name
    (seq_dir / "images").mkdir(parents=True)
    annotations = []
    for i in range(1, n_frames + 1):
        fname = f"frame{i:06d}.jpg"
        (seq_dir / "images" / fname).write_bytes(b"fake")
        annotations.append(
            {
                "sequence_name": seq_name,
                "image": {"path": f"apple/{seq_name}/images/{fname}", "size": [480, 640]},
                "viewpoint": {
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "T": [i * 0.1, 0.0, 1.0],
                    "focal_length": [1.2, 1.2],
                    "principal_point": [0.0, 0.0],
                },
            }
        )
    with gzip.open(seq_dir.parent / "frame_annotations.jgz", "wt", encoding="utf-8") as f:
        json.dump(annotations, f)
    return seq_dir


def test_load_co3dv2_count_and_intrinsics(tmp_path):
    ds = get_dataset("co3dv2")(_make_co3dv2_seq(tmp_path), max_frames=3)
    assert len(ds.images) == 3
    assert ds.gt_poses.shape == (3, 4, 4) and ds.gt_poses.dtype == np.float32
    assert ds.intrinsics.shape == (3, 3, 3) and ds.intrinsics.dtype == np.float32

    # NDC focal 1.2 in units of min(H, W) / 2 = 240 px; principal point at image center
    np.testing.assert_allclose(ds.intrinsics[0], [[288.0, 0.0, 320.0], [0.0, 288.0, 240.0], [0.0, 0.0, 1.0]])


def test_load_co3dv2_pytorch3d_to_opencv(tmp_path):
    ds = get_dataset("co3dv2")(_make_co3dv2_seq(tmp_path, n_frames=2), max_frames=2)
    np.testing.assert_allclose(ds.gt_poses[0, :3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(ds.gt_poses[1, :3, 3], [-0.2, 0.0, 1.0], atol=1e-6)


def test_load_co3dv2_unknown_sequence_raises(tmp_path):
    seq_dir = _make_co3dv2_seq(tmp_path)
    other = seq_dir.parent / "seq2"
    (other / "images").mkdir(parents=True)
    with pytest.raises(ValueError, match="seq2"):
        get_dataset("co3dv2")(other, max_frames=4)
