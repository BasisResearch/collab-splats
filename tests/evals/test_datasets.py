import sys
from pathlib import Path

import numpy as np
import pytest

# Local `evals/` is shadowed by an installed `evals`/`datasets` package; insert the
# evals dir on sys.path and import the module directly (sibling-test convention,
# mirrors the retired tests/evals/test_lc_parity_common.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from datasets import (  # noqa: E402
    _load_video,
    collect_frames,
    filter_images_to_list,
    list_scene_images,
    write_tum_allowed_frames,
)


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
        lines.append(f"{float(i):.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} " f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
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
    (poses_dir / "00.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0 0 1 0 0 0 0 1 0\n")
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


# ------------------- SLAM frame selection helpers (ex lc_parity_common) ------------------- #


def _make_fake_tum_seq(root: Path) -> Path:
    """Minimal TUM seq: 3 rgb frames, GT covers only the first two (third is a GT gap)."""
    seq = root / "tum" / "rgbd_dataset_freiburg3_long_office_household"
    (seq / "rgb").mkdir(parents=True)
    for ts in ("10.000000", "10.050000", "99.000000"):
        (seq / "rgb" / f"{ts}.png").touch()
    rgb_lines = ["# color images"] + [f"{ts} rgb/{ts}.png" for ts in ("10.000000", "10.050000", "99.000000")]
    (seq / "rgb.txt").write_text("\n".join(rgb_lines))
    gt_lines = ["# gt"] + [f"{ts} 0 0 0 0 0 0 1" for ts in ("10.001", "10.049")]
    (seq / "groundtruth.txt").write_text("\n".join(gt_lines))
    return seq


def test_list_scene_images_7scenes_layout(tmp_path):
    # 7-Scenes: color frames directly in seq dir
    for i in range(3):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
    (tmp_path / "frame-000000.pose.txt").touch()  # must be excluded
    imgs = list_scene_images(tmp_path)
    assert len(imgs) == 3
    assert all(p.suffix == ".png" and "color" in p.name for p in imgs)


def test_list_scene_images_tum_layout(tmp_path):
    # TUM: images under rgb/
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for ts in ("1305031102.175304", "1305031102.211214"):
        (rgb / f"{ts}.png").touch()
    (tmp_path / "groundtruth.txt").touch()
    imgs = list_scene_images(tmp_path)
    assert len(imgs) == 2
    assert imgs[0].parent.name == "rgb"
    assert imgs == sorted(imgs)


def test_filter_images_to_list_keeps_only_listed(tmp_path):
    # Matching is by basename: absolute frame paths vs a bare-basename allow list
    frames = [str(tmp_path / "rgb" / f"frame-{i}.png") for i in range(5)]
    allow = tmp_path / "allowed_frames.txt"
    allow.write_text("frame-0.png\nframe-2.png\nframe-4.png\n")
    kept = filter_images_to_list(frames, allow)
    assert [Path(p).name for p in kept] == ["frame-0.png", "frame-2.png", "frame-4.png"]


def test_write_tum_allowed_frames_drops_gt_gap(tmp_path):
    # The GT-gap frame (no groundtruth within 0.02 s) must not appear in the allow list
    seq = _make_fake_tum_seq(tmp_path)
    out = tmp_path / "allowed_frames.txt"
    write_tum_allowed_frames(seq, out)
    assert out.read_text().splitlines() == ["10.000000.png", "10.050000.png"]


def test_collect_frames_no_list_is_raw_sorted_listing(tmp_path):
    # Without image_list the frame universe is the raw dir listing (7-Scenes path)
    for i in range(4):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
    frames = collect_frames(tmp_path)
    assert [Path(p).name for p in frames] == [f"frame-{i:06d}.color.png" for i in range(4)]


def test_collect_frames_respects_image_list(tmp_path):
    # image_list restricts by basename; order of survivors is preserved
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for name in ("10.0.png", "10.5.png", "99.0.png"):
        (rgb / name).touch()
    allow = tmp_path / "allowed_frames.txt"
    allow.write_text("10.0.png\n10.5.png")
    frames = collect_frames(tmp_path, image_list=allow)
    assert [Path(p).name for p in frames] == ["10.0.png", "10.5.png"]


def test_collect_frames_max_frames_applies_after_filter(tmp_path):
    # max_frames caps the FILTERED sequence
    for i in range(6):
        (tmp_path / f"f{i}.png").touch()
    allow = tmp_path / "allowed_frames.txt"
    allow.write_text("\n".join(f"f{i}.png" for i in (0, 2, 4)))
    frames = collect_frames(tmp_path, image_list=allow, max_frames=2)
    assert [Path(p).name for p in frames] == ["f0.png", "f2.png"]


# ------------------- Video loading via the images/ store (single decode) ------------------- #


def test_load_video_single_decode(tmp_path, monkeypatch):
    """
    _load_video writes the images/ store once (single decode), not extract_frames (re-decode).
    """
    # Create synthetic frames and records
    n_frames = 5
    synthetic_frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(n_frames)]
    synthetic_records = [{"frame_idx": i, "timestamp": float(i) * 0.1} for i in range(n_frames)]

    # Patch the sampler to return synthetic data; count calls. max_frames is the
    # sampler's own contract now, so the double honours it instead of the caller trimming.
    call_count = [0]

    def mock_sample_uniform(video_path, *, max_frames, report, **kwargs):
        call_count[0] += 1
        return synthetic_frames[:max_frames], synthetic_records[:max_frames]

    monkeypatch.setattr("datasets.sample_uniform", mock_sample_uniform)
    monkeypatch.setattr("datasets.load_video_quality", lambda *a, **k: {"available": True, "frames": {}})

    # Create a minimal fake video file to pass to _load_video
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()

    # Call _load_video with max_frames limit
    result = _load_video(video_path, max_frames=3)

    # Exactly one decode pass (extract_frames would have made it two)
    assert call_count[0] == 1, f"sample_uniform called {call_count[0]} times, expected 1"

    # Verify result contains the expected number of images (max_frames=3)
    assert len(result.images) == 3, f"Expected 3 images, got {len(result.images)}"
    assert all(img.exists() for img in result.images), "Not all image paths exist"
    assert all(img.suffix.lower() == ".png" for img in result.images), "Images should be PNGs"

    # Verify GT poses are zeros placeholder
    assert result.gt_poses.shape == (3, 4, 4)
    assert np.allclose(result.gt_poses, 0.0)
