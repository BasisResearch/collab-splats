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
    # Same frame modulo codec noise: testsrc frames differ strongly frame-to-frame,
    # so a small mean error proves we seeked to the right frame.
    assert np.abs(fast.astype(int) - exact.astype(int)).mean() < 5


def test_fast_out_of_range_raises(synth_video):
    with pytest.raises(ValueError):
        extract_frame_fast(synth_video, 10_000)
