########################################################################
# Video metadata and streamed decode
########################################################################

import numpy as np
import pytest

from collab_splats.preproc.video import (
    _require_ffmpeg,
    context_indices,
    extract_frame,
    get_video_info,
    iter_frames,
)


def test_get_video_info_keys(tiny_video):
    info = get_video_info(tiny_video)
    assert set(info) == {"total_frames", "fps", "duration_s", "width", "height"}


def test_get_video_info_values(tiny_video):
    info = get_video_info(tiny_video)
    assert info["total_frames"] == 60
    assert info["fps"] == pytest.approx(30.0)
    assert (info["width"], info["height"]) == (320, 240)
    assert info["duration_s"] == pytest.approx(2.0)


def test_get_video_info_missing_file():
    info = get_video_info("/nonexistent/video.mp4")
    assert info["total_frames"] == 0 and info["fps"] == 0.0


def test_require_ffmpeg_raises_without_binary(monkeypatch):
    # Simulate ffmpeg absent from PATH — the only decode backend must hard-fail
    monkeypatch.setattr("collab_splats.preproc.video.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        _require_ffmpeg()


def test_get_video_info_without_count_frames_matches_dims(tiny_video):
    full = get_video_info(tiny_video)
    cheap = get_video_info(tiny_video, count_frames=False)

    assert (cheap["width"], cheap["height"]) == (full["width"], full["height"])
    assert cheap["fps"] == full["fps"]


def test_iter_frames_yields_index_and_bgr(tiny_video):
    out = list(iter_frames(tiny_video))
    total = get_video_info(tiny_video)["total_frames"]

    assert len(out) == total
    assert [i for i, _ in out] == list(range(total))
    assert out[0][1].ndim == 3 and out[0][1].dtype == np.uint8


def test_iter_frames_indices_yields_only_those(tiny_video):
    wanted = [0, 3, 7]

    out = list(iter_frames(tiny_video, indices=wanted))

    assert [i for i, _ in out] == wanted


def test_iter_frames_indices_matches_full_decode(tiny_video):
    everything = {i: f for i, f in iter_frames(tiny_video)}

    for idx, frame in iter_frames(tiny_video, indices=[1, 4]):
        assert np.array_equal(frame, everything[idx])


def test_iter_frames_range_matches_full_decode(tiny_video):
    everything = {i: f for i, f in iter_frames(tiny_video)}

    out = list(iter_frames(tiny_video, start=2, count=3))

    assert [i for i, _ in out] == [2, 3, 4]
    for idx, frame in out:
        assert np.array_equal(frame, everything[idx])


def test_iter_frames_rejects_indices_with_a_range(tiny_video):
    with pytest.raises(ValueError, match="indices"):
        list(iter_frames(tiny_video, indices=[1], start=2))


def test_extract_frame_accepts_a_preprobed_info(tiny_video):
    info = get_video_info(tiny_video)

    assert np.array_equal(extract_frame(tiny_video, 2, info=info), extract_frame(tiny_video, 2))


########################################################################
# extract_frame: input-seek single-frame decode for previews
########################################################################

import shutil
import subprocess

from collab_splats.preproc import extract_frame

# These four exercise the real ffmpeg seek path (and its rotate handling), so
# they are skipped rather than failed when the binary is absent.
requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


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


@requires_ffmpeg
def test_extract_frame_shape_and_dtype(synth_video):
    frame = extract_frame(synth_video, 30)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


@requires_ffmpeg
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


@requires_ffmpeg
def test_extract_frame_rotated_video_matches_display_dims(rotated_video):
    # ffmpeg autorotates on decode: extract_frame's dims must match get_video_info's
    # display dims (which already account for the rotate tag), consistent with the
    # streamed decode this function replaced.
    info = get_video_info(str(rotated_video))
    frame = extract_frame(rotated_video, 0)
    assert frame.shape[:2] == (info["height"], info["width"])


@requires_ffmpeg
def test_extract_frame_out_of_range_raises(synth_video):
    with pytest.raises(ValueError, match="out of range"):
        extract_frame(synth_video, 10_000)


########################################################################
# context_indices: constant-rate source frame grid
########################################################################


def test_context_indices_matches_sample_fps_stride(tiny_video):
    # tiny_video is 60 frames @ 30 fps -> fps=10 gives stride 3
    grid = context_indices(tiny_video, target_fps=10.0)
    assert grid[:4] == [0, 3, 6, 9]
    assert grid[-1] < 60
    assert len(grid) == 20


def test_context_indices_floors_stride_at_one(tiny_video):
    # A target rate above the source rate cannot sample sub-frame
    grid = context_indices(tiny_video, target_fps=1000.0)
    assert grid == list(range(60))


def test_context_indices_reuses_a_probe(tiny_video):
    info = {"total_frames": 10, "fps": 30.0, "width": 320, "height": 240}
    assert context_indices(tiny_video, target_fps=15.0, info=info) == [0, 2, 4, 6, 8]
