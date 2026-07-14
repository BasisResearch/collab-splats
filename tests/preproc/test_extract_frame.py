"""Tests for single-frame ffmpeg decode."""
import shutil
import subprocess

import numpy as np
import pytest

from collab_splats.preproc import extract_frame, load_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """10-frame 64x48 synthetic video whose frame index is encoded in the red channel."""
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    # Each frame's red value = frame_index * 20 → decodable assertion signal
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi",
            "-i", "color=black:size=64x48:rate=10:duration=1",
            "-vf", "geq=r='N*20':g=0:b=0",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def rotated_video(tiny_video, tmp_path_factory):
    """tiny_video re-muxed with a rotate=90 tag — display dims become 48x64."""
    path = tmp_path_factory.mktemp("vid_rot") / "tiny_rot.mp4"
    # Stream copy + rotate tag: same pixels, ffmpeg autorotates on decode
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(tiny_video), "-c", "copy",
         "-metadata:s:v:0", "rotate=90", str(path)],
        check=True,
    )
    return path


def test_extract_frame_shape_and_dtype(tiny_video):
    frame = extract_frame(tiny_video, 0)
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8


def test_extract_frame_selects_correct_index(tiny_video):
    # Red channel encodes frame index * 20; codec noise allows a loose tolerance
    f0 = extract_frame(tiny_video, 0)
    f5 = extract_frame(tiny_video, 5)
    assert abs(int(f0[..., 0].mean()) - 0) < 15
    assert abs(int(f5[..., 0].mean()) - 100) < 15


def test_extract_frame_rotated_video_display_orientation(rotated_video):
    # ffmpeg autorotates on decode: 64x48 coded + rotate=90 → (64, 48, 3) display frame
    frame = extract_frame(rotated_video, 0)
    assert frame.shape == (64, 48, 3)
    # Must agree exactly with the load_frames path for the same file
    assert np.array_equal(frame, load_frames(rotated_video, [0])[0])


def test_extract_frame_out_of_range_raises(tiny_video):
    with pytest.raises(ValueError, match="frame 999"):
        extract_frame(tiny_video, 999)
