from pathlib import Path

import cv2
import numpy as np
import pytest

from collab_splats.preproc import frames as fr
from collab_splats.reconstructor import Reconstructor


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """Synthesize a 30-frame 320x240 mp4 (same pattern as tests/preproc/test_sampling.py)."""
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.default_rng(0)
    noise = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    for i in range(30):
        frame = noise.copy()
        x = 10 + i * 4
        cv2.rectangle(frame, (x, 60), (x + 60, 140), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


def _make_config(tmp_path, video_path):
    """Minimal valid Reconstructor config for a video input (matches tests/reconstructor/test_reconstructor.py)."""
    return {
        "input_path": str(video_path),
        "output_path": str(tmp_path / "out"),
        "preproc": {
            "frame_selection": "uniform",
            "fps": 1.0,
            "min_frames": None,
            "max_frames": 5,
            "n_workers": 1,
        },
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggtx",
            "bundle_adjustment": False,
            "loop_closure": False,
            "clean": {"enabled": False},
        },
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 512},
        "mesh": {"enabled": False, "voxel_size": 0.01, "depth_trunc": 1.0},
        "localization": {"enabled": False},
    }


def test_preprocess_writes_images_dir(tmp_path, tiny_video):
    cfg = _make_config(tmp_path, tiny_video)
    rec = Reconstructor(cfg)
    rec.preproc()

    images_dir = Path(cfg["output_path"]) / "images"
    assert 0 < len(fr.frame_paths(images_dir)) <= 5

    # images/ is the sole persistent frame store — no frames.zarr is written
    assert not (Path(cfg["output_path"]) / "frames.zarr").exists()


def test_preprocess_images_dir_path_property(tmp_path, tiny_video):
    cfg = _make_config(tmp_path, tiny_video)
    rec = Reconstructor(cfg)
    assert rec.images_dir == Path(cfg["output_path"]) / "images"


def test_preprocess_writes_images_dir_for_image_dir(tmp_path):
    """Dir-input branch also produces an images/ store, indexed by position."""
    img_dir = tmp_path / "images_in"
    img_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / f"src_{i}.jpg"), np.full((16, 16, 3), i * 10, dtype=np.uint8))

    cfg = _make_config(tmp_path, img_dir)
    rec = Reconstructor(cfg)
    rec.preproc()

    assert [fr.frame_idx_from_path(p) for p in fr.frame_paths(rec.images_dir)] == [0, 1, 2]


def test_preprocess_writes_a_quality_report_beside_the_images_dir(tmp_path, tiny_video):
    """
    The report is a first-class scene artefact, reused by existence like images/.
    """
    cfg = _make_config(tmp_path, tiny_video)
    rec = Reconstructor(cfg)

    rec.preproc()

    out = Path(cfg["output_path"])
    assert (out / "video_quality_report.json").exists()
    assert (out / "images").exists()


def test_preprocess_image_dir_writes_no_quality_report(tmp_path):
    """
    An image directory takes every image, so there is nothing to measure or select.
    """
    img_dir = tmp_path / "images_in"
    img_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / f"src_{i}.jpg"), np.full((16, 16, 3), i * 10, dtype=np.uint8))

    cfg = _make_config(tmp_path, img_dir)
    rec = Reconstructor(cfg)

    rec.preproc()

    assert not (Path(cfg["output_path"]) / "video_quality_report.json").exists()


def test_preprocess_image_dir_renames_to_frame_store(tmp_path):
    """
    Image-directory input of any filenames lands as images/frame_NNNNNN.png.
    """
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / f"img_{i}.png"), np.full((8, 12, 3), i * 40 + 5, np.uint8))

    cfg = _make_config(tmp_path, src)
    rec = Reconstructor(cfg)
    rec.preproc()

    assert [p.name for p in fr.frame_paths(rec.images_dir)] == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
    ]
    assert fr.read_frames(rec.images_dir)[1][0, 0].tolist() == [45, 45, 45]
