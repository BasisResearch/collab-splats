from pathlib import Path

import cv2
import numpy as np
import pytest

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import Reconstructor


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
    """Minimal valid Reconstructor config for a video input (matches tests/wrapper/test_reconstructor.py)."""
    return {
        "input_path": str(video_path),
        "output_path": str(tmp_path / "out"),
        "preprocessing": {"frame_selection": "uniform", "fps": 1.0, "min_frames": None, "max_frames": 5},
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggtx",
            "bundle_adjustment": False,
            "loop_closure": False,
            "clean": {"enabled": False},
        },
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 512},
        "mesh": {"enabled": False, "mesher": "tsdf", "voxel_size": 0.01, "sdf_trunc": 0.04},
        "localization": {"enabled": False},
        "nerfstudio": {"sfm_tool": "hloc", "train_method": "rade-features"},
    }


def test_preprocess_writes_frames_zarr(tmp_path, tiny_video):
    cfg = _make_config(tmp_path, tiny_video)
    rec = Reconstructor(cfg)
    rec.preprocess()

    store = FrameStore.open(Path(cfg["output_path"]) / "frames.zarr")
    assert 0 < len(store) <= 5
    # frames.zarr is the sole persistent frame store — no images/ JPG dir is written
    assert not (Path(cfg["output_path"]) / "images").exists()


def test_preprocess_frames_zarr_path_property(tmp_path, tiny_video):
    cfg = _make_config(tmp_path, tiny_video)
    rec = Reconstructor(cfg)
    assert rec.frames_zarr == Path(cfg["output_path"]) / "frames.zarr"


def test_preprocess_writes_frames_zarr_for_image_dir(tmp_path):
    """Dir-input branch also produces a frames.zarr, with nan blur_score records."""
    img_dir = tmp_path / "images_in"
    img_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / f"src_{i}.jpg"), np.full((16, 16, 3), i * 10, dtype=np.uint8))

    cfg = _make_config(tmp_path, img_dir)
    rec = Reconstructor(cfg)
    rec.preprocess()

    store = FrameStore.open(rec.frames_zarr)
    assert len(store) == 3
    assert np.isnan(store.record(0)["blur_score"])
