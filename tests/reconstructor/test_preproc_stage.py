"""
Tests for the Reconstructor preproc stage at the undistort boundary.
"""

import cv2
import numpy as np

from collab_splats import reconstructor as R
from collab_splats.preproc import frames as fr
from collab_splats.preproc.undistort import undistort_frames
from collab_splats.reconstructor import Reconstructor


def _write_textured_sequence(out_dir, *, n, width, height):
    """n frames of a high-frequency pattern translating a few pixels per frame."""
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (height + 4 * n, width + 4 * n, 3), dtype=np.uint8)

    for i in range(n):
        crop = canvas[2 * i : 2 * i + height, 2 * i : 2 * i + width]
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), crop)


def _reconstructor(tmp_path, input_path, *, undistort):
    """A frame-directory input with every later stage off."""
    config = {
        "input_path": str(input_path),
        "output_path": str(tmp_path / "scene"),
        "preproc": {"undistort": undistort},
        "semantics": {"enabled": False},
    }
    return Reconstructor(config)


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
    src = tmp_path / "imgs"
    src.mkdir()
    _write_textured_sequence(src, n=12, width=320, height=240)

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
