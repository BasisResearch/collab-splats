"""extract_frame_fast: input-seek single-frame decode for previews."""

import shutil
import subprocess

import numpy as np
import pytest

from collab_splats.preproc import extract_frame, extract_frame_fast

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


def test_fast_matches_exact_shape_and_content(synth_video):
    fast = extract_frame_fast(synth_video, 30)
    exact = extract_frame(synth_video, 30)
    assert fast.shape == exact.shape == (240, 320, 3)
    assert fast.dtype == np.uint8
    exact_29 = extract_frame(synth_video, 29)
    exact_31 = extract_frame(synth_video, 31)
    d30 = np.abs(fast.astype(int) - exact.astype(int)).mean()
    # Fast must be closer to frame 30 than to its neighbours (off-by-one would flip this)
    assert d30 < np.abs(fast.astype(int) - exact_29.astype(int)).mean()
    assert d30 < np.abs(fast.astype(int) - exact_31.astype(int)).mean()


def test_fast_out_of_range_raises(synth_video):
    with pytest.raises(ValueError):
        extract_frame_fast(synth_video, 10_000)
