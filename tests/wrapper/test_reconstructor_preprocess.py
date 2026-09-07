import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from collab_splats.preproc import frames as fr
from collab_splats.wrapper import reconstructor
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
    rec.preprocess()

    images_dir = Path(cfg["output_path"]) / "images"
    assert 0 < len(fr.frame_paths(images_dir)) <= 5
    # images/ is the sole persistent frame store — no frames.zarr is written
    assert not (Path(cfg["output_path"]) / "frames.zarr").exists()


def test_preprocess_images_dir_path_property(tmp_path, tiny_video):
    cfg = _make_config(tmp_path, tiny_video)
    rec = Reconstructor(cfg)
    assert rec.images_dir == Path(cfg["output_path"]) / "images"


def test_preprocess_writes_images_dir_for_image_dir(tmp_path):
    """Dir-input branch also produces an images/ store, with nan blur_score records."""
    img_dir = tmp_path / "images_in"
    img_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / f"src_{i}.jpg"), np.full((16, 16, 3), i * 10, dtype=np.uint8))

    cfg = _make_config(tmp_path, img_dir)
    rec = Reconstructor(cfg)
    rec.preprocess()

    assert len(fr.frame_paths(rec.images_dir)) == 3
    assert fr.read_manifest(rec.images_dir)["frames"][0]["blur_score"] is None


def test_preprocess_writes_a_quality_report_beside_the_images_dir(tmp_path, tiny_video):
    """
    The report is a first-class scene artefact, reused by existence like images/.
    """
    cfg = _make_config(tmp_path, tiny_video)
    rec = Reconstructor(cfg)

    rec.preprocess()

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

    rec.preprocess()

    assert not (Path(cfg["output_path"]) / "video_quality_report.json").exists()


def test_extract_frames_writes_an_images_dir_and_manifest(tmp_path, monkeypatch):
    """
    Image-directory input lands as images/frame_NNNNNN.png + frames.json.
    """
    import cv2
    import numpy as np

    from collab_splats.preproc import frames as fr
    from collab_splats.wrapper.reconstructor import extract_frames

    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / f"img_{i}.png"), np.full((8, 12, 3), i * 40 + 5, np.uint8))

    scene = tmp_path / "scene"
    scene.mkdir()

    n = extract_frames(src, scene / "images", "uniform", None, None, 10)

    assert n == 3
    assert [p.name for p in fr.frame_paths(scene / "images")] == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
    ]
    assert fr.read_manifest(scene / "images")["provenance"]["method"] == "dir"


########################################
# extract_frames branch helpers
########################################


def test_extract_frames_has_no_method_alias():
    """
    Each input branch is its own function and the `method = frame_selection` alias is gone.
    """
    src = inspect.getsource(reconstructor.extract_frames)

    assert "method = frame_selection" not in src
    assert callable(reconstructor._frames_from_dir)
    assert callable(reconstructor._frames_from_video)


def test_frames_from_dir_returns_frames_records_and_provenance(tmp_path):
    """
    The directory branch called directly: every image in filename order, plus provenance.
    """
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / f"img_{i}.png"), np.full((8, 12, 3), i * 40 + 5, np.uint8))

    frame_arrays, records, prov = reconstructor._frames_from_dir(src, max_frames=10)

    assert len(frame_arrays) == 3 and frame_arrays[0].shape == (8, 12, 3)
    assert [r["frame_idx"] for r in records] == [0, 1, 2]
    assert prov["method"] == "dir" and prov["max_frames"] == 10
