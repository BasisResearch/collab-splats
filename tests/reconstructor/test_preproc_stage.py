"""
Tests for the preproc stage: undistort, the in-memory frame handoff and the background PNG write.
"""

import gc
import threading
import weakref
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from collab_splats import reconstructor as R
from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator
from collab_splats.preproc import frames as fr
from collab_splats.preproc.undistort import undistort_frames
from collab_splats.reconstructor import Reconstructor


def _write_textured_sequence(out_dir, *, n, width, height):
    """n frames of a high-frequency pattern translating a few pixels per frame, written to out_dir; returns it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (height + 4 * n, width + 4 * n, 3), dtype=np.uint8)

    for i in range(n):
        crop = canvas[2 * i : 2 * i + height, 2 * i : 2 * i + width]
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), crop)

    return out_dir


def _reconstructor(tmp_path, input_path, *, undistort, backend="vggtx"):
    """A frame-directory input with every later stage off."""
    config = {
        "input_path": str(input_path),
        "output_path": str(tmp_path / "scene"),
        "preproc": {"undistort": undistort},
        "pointcloud": {"backend": backend, "loop_closure": False},
        "semantics": {"enabled": False},
    }
    return Reconstructor(config)


def _stub_result():
    """Pointcloud result stand-in with an empty cloud."""
    result = MagicMock()
    result.points = np.zeros((0, 3), np.float32)
    return result


@dataclass
class _Handoff:
    """What one pointcloud call saw: frames handed over, preprocessed images, which paths read them."""

    frames: dict | None
    images: torch.Tensor
    consulted: bool  # read the handoff at all
    used_arrays: bool  # took the array preprocessor
    on_disk: int  # frame files in images/ when preprocessing ran


def _images(views):
    """Preprocessed images as one tensor: a batch as is, MapAnything view dicts concatenated."""
    if isinstance(views, list):
        return torch.cat([v["img"] for v in views])

    return views


def _spy_creator_cls(backend, seen):
    """Real creator class for backend that appends a _Handoff to seen per create_pointcloud."""
    creator_cls = get_creator(backend)
    has_preprocess = hasattr(creator_cls, "_preprocess_arrays")

    class _HandoffSpy(creator_cls):
        def _reconstruct(self, paths, out_dir):
            # Real preprocessing on create_pointcloud's own paths, spying on the handoff read
            with patch.object(
                BaseFeedforwardCreator,
                "_get_path_frames",
                autospec=True,
                side_effect=BaseFeedforwardCreator._get_path_frames,
            ) as path_frames:
                # Spy on the array preprocessor too, where the backend has one
                if has_preprocess:
                    with patch.object(
                        creator_cls,
                        "_preprocess_arrays",
                        autospec=True,
                        side_effect=creator_cls._preprocess_arrays,
                    ) as preprocess:
                        views, _ = self._preprocess(paths)

                    used_arrays = preprocess.called
                else:
                    views, _ = self._preprocess(paths)
                    used_arrays = False

            # Count the frame files already written when preprocessing ran
            images_dir = paths[0].parent
            on_disk = len(fr.frame_paths(images_dir))

            seen.append(
                _Handoff(
                    self.frames,
                    _images(views),
                    path_frames.called,
                    used_arrays,
                    on_disk,
                )
            )
            return _stub_result()

    return _HandoffSpy


def test_preproc_dir_no_undistort_keeps_native_dims(tmp_path):
    # Default path: frames written as read, native dims kept
    src = tmp_path / "imgs"
    src.mkdir()
    cv2.imwrite(str(src / "000.jpg"), np.zeros((480, 640, 3), np.uint8))

    r = _reconstructor(tmp_path, src, undistort=False)
    r.preproc()

    assert fr.read_frames(r.images_dir)[0].shape == (480, 640, 3)


def test_preproc_dir_undistorts(tmp_path, monkeypatch):
    # Real calibration off the written images/, then a rewrite at the undistorted dims
    src = _write_textured_sequence(tmp_path / "imgs", n=12, width=320, height=240)

    # Spy on undistort_frames to see the cameras the stage calibrated and wrote with
    seen = []

    def spy(rgbs, camera):
        out, undistorted_camera = undistort_frames(rgbs, camera)
        seen.append((camera, undistorted_camera))
        return out, undistorted_camera

    monkeypatch.setattr(R, "undistort_frames", spy)

    r = _reconstructor(tmp_path, src, undistort=True)
    r.preproc()

    [(camera, undistorted_camera)] = seen
    assert camera.model.name == "OPENCV"
    assert (camera.width, camera.height) == (320, 240)
    assert undistorted_camera.model.name == "PINHOLE"

    # What lands on disk is the undistorted framing
    stack = fr.read_frames(r.images_dir)
    assert stack.shape[1:3] == (undistorted_camera.height, undistorted_camera.width)


@pytest.mark.parametrize("background_write", [False, True])
@pytest.mark.parametrize("backend", ["vggt_omega", "vggtx", "mapanything", "loger"])
def test_preproc_frames_handed_off_then_released(
    tmp_path, monkeypatch, backend, background_write
):
    # Same-process preproc -> pointcloud hands a feedforward creator the written pixels, then drops them
    src = _write_textured_sequence(tmp_path / "imgs", n=3, width=64, height=48)

    r = _reconstructor(tmp_path, src, undistort=False, backend=backend)

    # A held background write leaves images/ empty until pointcloud joins it
    if background_write:
        started = _gate_background_write(monkeypatch, r)

    r.preproc(background_write=background_write)

    if background_write:
        assert started.wait(10)

    refs = [weakref.ref(rgb) for rgb in r._frames.values()]
    seen = []

    # Real create_pointcloud path resolution; cleaning and COLMAP export stubbed out
    with (
        patch.object(R, "get_creator", return_value=_spy_creator_cls(backend, seen)),
        patch(
            "collab_splats.pointcloud.base.clean_pointcloud",
            side_effect=lambda result, **kw: result,
        ),
        patch("collab_splats.pointcloud.base.write_colmap_reconstruction"),
        patch.object(BaseFeedforwardCreator, "_colmap_model"),
    ):
        r.pointcloud()

        # A leaf-only re-run has no frames to hand off
        r.pointcloud()

    first, rerun = seen
    paths = fr.frame_paths(r.images_dir)

    # Every feedforward creator was handed every written frame, pixel-equal to its PNG
    assert sorted(first.frames) == [p.name for p in paths]
    handed = np.stack([first.frames[p.name] for p in paths])
    written = fr.read_frames(r.images_dir)
    np.testing.assert_array_equal(handed, written)

    # Either path gives the file path's images; only a handoff backend took the arrays
    creator = get_creator(backend)()
    views, _ = creator._preprocess(paths)
    assert torch.equal(first.images, _images(views))
    assert first.used_arrays is (backend != "loger")
    assert first.consulted is (backend != "loger")

    # A handoff backend preprocessed before the held write landed; loger only after it
    assert first.on_disk == (0 if background_write and backend != "loger" else 3)

    # A leaf-only re-run has no frames, so it reads the files
    assert rerun.frames is None
    assert rerun.used_arrays is False

    # Released: no live reference to any frame once the stage returned
    del first, rerun, seen
    gc.collect()
    assert r._frames is None
    assert all(ref() is None for ref in refs)


def test_run_releases_frames_when_pointcloud_does_not_run(tmp_path):
    # A run that stops after preproc drops the handed-off frames on exit
    src = _write_textured_sequence(tmp_path / "imgs", n=2, width=64, height=48)

    r = _reconstructor(tmp_path, src, undistort=False, backend="vggt_omega")
    r.run(["preproc"])

    assert r._frames is None


def test_run_releases_frames_when_a_stage_raises(tmp_path):
    # A stage that raises after preproc still leaves no frames behind
    src = _write_textured_sequence(tmp_path / "imgs", n=2, width=64, height=48)

    r = _reconstructor(tmp_path, src, undistort=False, backend="vggt_omega")

    with (
        patch.object(Reconstructor, "pointcloud", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        r.run(["preproc", "pointcloud"])

    assert r._frames is None


def _gate_background_write(monkeypatch, r):
    """
    Hold r's background write until r joins it; returns the event set once the write is held.
    """
    started = threading.Event()
    release = threading.Event()
    real_write = fr.write_frames
    real_join = r._join_frames_write

    def write(*args, **kwargs):
        # Only the background thread is held; undistort's synchronous write passes through
        if threading.current_thread() is not threading.main_thread():
            started.set()
            assert release.wait(10)

        return real_write(*args, **kwargs)

    def join():
        if r._frames_write is not None:
            release.set()

        real_join()

    monkeypatch.setattr(fr, "write_frames", write)
    monkeypatch.setattr(r, "_join_frames_write", join)
    return started


def _stub_creator_cls(on_create):
    """Feedforward creator stand-in whose create_pointcloud calls on_create, then returns a stub result."""
    creator_cls = MagicMock()
    create_pointcloud = creator_cls.return_value.create_pointcloud
    create_pointcloud.return_value = _stub_result()

    def create(images_dir, out_dir, model_dir):
        on_create(images_dir)
        return create_pointcloud.return_value

    create_pointcloud.side_effect = create
    return creator_cls


def _fail_write(*args, **kwargs):
    """write_frames stand-in that fails as a full disk would."""
    raise OSError("disk full")


def _png_bytes(images_dir):
    """File name to raw bytes for every PNG in images_dir."""
    return {p.name: p.read_bytes() for p in fr.frame_paths(images_dir)}


def test_run_preproc_alone_writes_every_png_byte_identical(tmp_path, monkeypatch):
    # A preproc-only run returns with images/ complete and byte-equal to the synchronous write
    src = _write_textured_sequence(tmp_path / "imgs", n=4, width=64, height=48)

    sync = _reconstructor(tmp_path / "sync", src, undistort=False)
    sync.preproc()

    background = _reconstructor(tmp_path / "background", src, undistort=False)
    _gate_background_write(monkeypatch, background)
    background.run(["preproc"])

    expected = _png_bytes(sync.images_dir)
    assert len(expected) == 4
    assert _png_bytes(background.images_dir) == expected


def test_run_preproc_undistort_writes_undistorted_frames_last(tmp_path, monkeypatch):
    # Undistort's calibration write stays synchronous; the background write lands the undistorted pixels
    src = _write_textured_sequence(tmp_path / "imgs", n=12, width=320, height=240)

    # Spy on undistort_frames to keep the pixels the final write must hold
    seen = []

    def spy(rgbs, camera):
        out, undistorted_camera = undistort_frames(rgbs, camera)
        seen.append(out)
        return out, undistorted_camera

    monkeypatch.setattr(R, "undistort_frames", spy)

    r = _reconstructor(tmp_path, src, undistort=True)
    _gate_background_write(monkeypatch, r)
    r.run(["preproc"])

    [undistorted] = seen
    np.testing.assert_array_equal(fr.read_frames(r.images_dir), np.asarray(undistorted))


@pytest.mark.parametrize(
    "backend, on_disk_at_create", [("vggt_omega", 0), ("loger", 3)]
)
def test_pointcloud_joins_the_background_write(
    tmp_path, monkeypatch, backend, on_disk_at_create
):
    # A handoff creator runs during the held write; loger reads images/, so its creator waits for it
    src = _write_textured_sequence(tmp_path / "imgs", n=3, width=64, height=48)

    r = _reconstructor(tmp_path, src, undistort=False, backend=backend)
    started = _gate_background_write(monkeypatch, r)
    r.preproc(background_write=True)
    assert started.wait(10)
    assert fr.frame_paths(r.images_dir) == []

    # The creator records the store it sees; pointcloud returns only once every PNG is written
    on_disk = []

    def on_create(images_dir):
        on_disk.append(len(fr.frame_paths(images_dir)))

    with patch.object(R, "get_creator", return_value=_stub_creator_cls(on_create)):
        r.pointcloud()

    assert on_disk == [on_disk_at_create]
    assert len(fr.frame_paths(r.images_dir)) == 3
    assert r._frames_write is None


@pytest.mark.parametrize("stages", [["preproc"], ["preproc", "pointcloud"]])
def test_run_raises_a_failing_background_write(tmp_path, monkeypatch, stages):
    # A write error surfaces from run(), and pointcloud writes no zarr over a broken store
    src = _write_textured_sequence(tmp_path / "imgs", n=2, width=64, height=48)

    monkeypatch.setattr(fr, "write_frames", _fail_write)
    creator_cls = _stub_creator_cls(lambda d: None)
    r = _reconstructor(tmp_path, src, undistort=False, backend="vggt_omega")

    with (
        patch.object(R, "get_creator", return_value=creator_cls),
        pytest.raises(OSError, match="disk full"),
    ):
        r.run(stages)

    # The creator ran during the write, but its result was never saved
    create_pointcloud = creator_cls.return_value.create_pointcloud
    assert create_pointcloud.called is ("pointcloud" in stages)
    assert not create_pointcloud.return_value.save_zarr.called
    assert r._frames_write is None


def test_run_stage_error_wins_over_a_failing_background_write(
    tmp_path, monkeypatch, caplog
):
    # A pointcloud error propagates from run(); the concurrent write's error is only logged
    src = _write_textured_sequence(tmp_path / "imgs", n=2, width=64, height=48)

    def on_create(images_dir):
        raise RuntimeError("model failed")

    monkeypatch.setattr(fr, "write_frames", _fail_write)
    r = _reconstructor(tmp_path, src, undistort=False, backend="vggt_omega")

    with (
        patch.object(R, "get_creator", return_value=_stub_creator_cls(on_create)),
        pytest.raises(RuntimeError, match="model failed"),
    ):
        r.run(["preproc", "pointcloud"])

    assert "background PNG write failed: disk full" in caplog.text
    assert r._frames_write is None
