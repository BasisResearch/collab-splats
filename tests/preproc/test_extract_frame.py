"""extract_frame: input-seek single-frame decode for previews."""

import shutil
import subprocess

import numpy as np
import pytest

from collab_splats.preproc import extract_frame, get_video_info

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    # 2s of 30fps testsrc — 60 frames, constant frame rate
    path = tmp_path_factory.mktemp("vid") / "synth.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=30",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def rotated_video(synth_video, tmp_path_factory):
    """synth_video re-muxed with a rotate=90 tag — display dims swap to (240, 320)."""
    path = tmp_path_factory.mktemp("vid_rot") / "rotated.mp4"
    # Stream copy + rotate tag: same pixels, ffmpeg autorotates on decode
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(synth_video),
            "-c",
            "copy",
            "-metadata:s:v:0",
            "rotate=90",
            str(path),
        ],
        check=True,
    )
    return path


def test_extract_frame_shape_and_dtype(synth_video):
    frame = extract_frame(synth_video, 30)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_extract_frame_lands_near_target_index(synth_video):
    # testsrc's content changes every frame; a correctly-seeked decode must differ
    # less from its immediate neighbour than from a distant frame (catches gross
    # seek errors, e.g. landing many frames off target).
    frame_30 = extract_frame(synth_video, 30)
    frame_31 = extract_frame(synth_video, 31)
    frame_0 = extract_frame(synth_video, 0)
    d_adjacent = np.abs(frame_30.astype(int) - frame_31.astype(int)).mean()
    d_far = np.abs(frame_30.astype(int) - frame_0.astype(int)).mean()
    assert d_adjacent < d_far


def test_extract_frame_rotated_video_matches_display_dims(rotated_video):
    # ffmpeg autorotates on decode: extract_frame's dims must match get_video_info's
    # display dims (which already account for the rotate tag), consistent with the
    # streamed decode this function replaced.
    info = get_video_info(str(rotated_video))
    frame = extract_frame(rotated_video, 0)
    assert frame.shape[:2] == (info["height"], info["width"])


def test_extract_frame_out_of_range_raises(synth_video):
    with pytest.raises(ValueError, match="out of range"):
        extract_frame(synth_video, 10_000)
