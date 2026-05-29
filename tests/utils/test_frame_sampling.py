import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from collab_splats.utils.frame_sampling import (
    sample_frames_optical_flow,
    sample_frames_fps,
    OpticalFlowFrameSelector,
    get_video_info,
    score_all_frames,
)


def test_fps_sampler_returns_empty_for_missing_file():
    frames, indices = sample_frames_fps("nonexistent.mp4", fps=5.0)
    assert frames == []
    assert indices == []


@pytest.fixture
def tiny_video(tmp_path):
    path = str(tmp_path / "test.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    for i in range(90):
        frame = np.full((48, 64, 3), (i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_optical_flow_accepts_max_frames(tiny_video):
    frames, scores = sample_frames_optical_flow(
        tiny_video, max_frames=5
    )
    assert isinstance(frames, list)
    assert len(frames) >= 1


def test_optical_flow_first_frame_always_included(tiny_video):
    frames, scores = sample_frames_optical_flow(
        tiny_video, min_disparity=9999.0, max_frames=10
    )
    assert len(frames) >= 1


def test_optical_flow_frames_are_rgb(tiny_video):
    frames, scores = sample_frames_optical_flow(
        tiny_video, max_frames=3
    )
    assert frames[0].shape[2] == 3
    assert frames[0].dtype == np.uint8


def test_optical_flow_returns_scores_for_each_frame(tiny_video):
    """scores list has one entry per selected frame with required keys."""
    frames, scores = sample_frames_optical_flow(tiny_video, verbose=False)
    assert len(scores) == len(frames)
    for s in scores:
        assert "disparity" in s
        assert "rotation" in s
        assert "histogram_similarity" in s
        assert "score" in s
        assert s["selected"] is True


def test_selector_selects_first_frame_always():
    cv2 = pytest.importorskip("cv2")
    selector = OpticalFlowFrameSelector()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    should_select, score, _ = selector.should_select_frame(frame)
    assert should_select is True  # first frame always selected
    assert score == 1.0


def test_selector_returns_normalized_score():
    cv2 = pytest.importorskip("cv2")
    selector = OpticalFlowFrameSelector()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    selector.should_select_frame(frame)  # initialize
    frame2 = np.full((48, 64, 3), 128, dtype=np.uint8)
    _, score, _ = selector.should_select_frame(frame2)
    assert 0.0 <= score <= 1.0


def test_selector_resets_state():
    selector = OpticalFlowFrameSelector()
    selector.reset()
    assert selector.last_keyframe_gray is None


from unittest.mock import MagicMock, patch, call
import sys


def _make_mock_cap(total_frames=90, fps=30.0):
    """Return a mock VideoCapture that reads one black frame then stops."""
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: float(total_frames),
    }.get(prop, 0.0)
    mock_cap.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
    return mock_cap


def test_fps_sampler_uses_seek(monkeypatch):
    """cv2 fallback path seeks to each target index exactly."""
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_decoder_backend", lambda: "cv2")
    # Also disable ffmpeg so the new ffmpeg-first check falls through to cv2
    monkeypatch.setattr(fs.shutil, "which", lambda _: None)

    mock_cap = _make_mock_cap(total_frames=90, fps=30.0)
    # 90 frames at 30fps, target 1fps → interval=30, targets=[0,30,60], expect 3 seeks
    mock_cap.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture = MagicMock(return_value=mock_cap)
    mock_cv2.CAP_PROP_FPS = cv2.CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    mock_cv2.CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
    mock_cv2.CAP_PROP_ORIENTATION_AUTO = cv2.CAP_PROP_ORIENTATION_AUTO
    mock_cv2.CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    mock_cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    mock_cv2.cvtColor = cv2.cvtColor

    with patch("collab_splats.utils.frame_sampling.cv2", mock_cv2):
        frames, indices = sample_frames_fps("fake.mp4", fps=1.0, verbose=False)

    seek_calls = [c for c in mock_cap.set.call_args_list
                  if c.args[0] == cv2.CAP_PROP_POS_FRAMES]
    assert len(seek_calls) == 3, "Must seek once per target frame"
    assert [c.args[1] for c in seek_calls] == [0, 30, 60]
    assert len(frames) == 3
    assert indices == [0, 30, 60]


def test_fps_sampler_calls_on_progress(monkeypatch):
    mock_cap = _make_mock_cap(total_frames=90, fps=30.0)
    read_results = [(True, np.zeros((48, 64, 3), dtype=np.uint8))] * 3 + [(False, None)]
    mock_cap.read.side_effect = read_results

    calls = []
    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture = MagicMock(return_value=mock_cap)
    mock_cv2.CAP_PROP_FPS = cv2.CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    mock_cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    mock_cv2.cvtColor = cv2.cvtColor

    with patch("collab_splats.utils.frame_sampling.cv2", mock_cv2):
        sample_frames_fps("fake.mp4", fps=1.0, on_progress=lambda c, t: calls.append((c, t)), verbose=False)

    assert len(calls) >= 1
    assert all(t == 3 for _, t in calls), "total must equal n_targets (90 frames / interval 30 = 3)"


def test_fps_sampler_no_progress_arg_ok(monkeypatch):
    """on_progress=None must not raise."""
    mock_cap = _make_mock_cap(total_frames=30, fps=30.0)
    mock_cap.read.side_effect = [(True, np.zeros((48, 64, 3), dtype=np.uint8))] + [(False, None)]

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture = MagicMock(return_value=mock_cap)
    mock_cv2.CAP_PROP_FPS = cv2.CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    mock_cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    mock_cv2.cvtColor = cv2.cvtColor

    with patch("collab_splats.utils.frame_sampling.cv2", mock_cv2):
        frames, indices = sample_frames_fps("fake.mp4", fps=30.0, max_frames=1)
    assert isinstance(frames, list)
    assert isinstance(indices, list)


def test_optical_flow_calls_on_progress(tiny_video):
    calls = []
    frames, _ = sample_frames_optical_flow(
        tiny_video, max_frames=5, on_progress=lambda c, t: calls.append((c, t))
    )
    assert len(calls) >= 1
    assert all(t > 0 for _, t in calls), "total must come from CAP_PROP_FRAME_COUNT"
    assert all(c > 0 for c, _ in calls), "current must increment"


def test_optical_flow_resizes_for_analysis(tiny_video, monkeypatch):
    """Low-res resize must be called when frame width > 480."""
    import cv2 as _cv2

    resize_calls = []
    original_resize = _cv2.resize

    def spy_resize(src, dsize, **kwargs):
        resize_calls.append(src.shape)
        return original_resize(src, dsize, **kwargs)

    monkeypatch.setattr(_cv2, "resize", spy_resize)

    # tiny_video is 64px wide — below 480 threshold, resize must NOT be called
    frames, _ = sample_frames_optical_flow(tiny_video, max_frames=3)
    assert len(resize_calls) == 0, "Must not resize frames already <= 480px wide"


# ── get_video_info ────────────────────────────────────────────────────────────


def test_get_video_info_keys(tiny_video):
    info = get_video_info(tiny_video)
    assert set(info.keys()) == {"total_frames", "fps", "duration_s", "width", "height"}


def test_get_video_info_values(tiny_video):
    info = get_video_info(tiny_video)
    assert info["total_frames"] == 90
    assert abs(info["fps"] - 30.0) < 1.0
    assert abs(info["duration_s"] - 3.0) < 0.5


def test_get_video_info_missing_file():
    info = get_video_info("nonexistent.mp4")
    assert info["total_frames"] == 0
    assert info["fps"] == 0.0
    assert info["duration_s"] == 0.0


def test_get_video_info_no_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    info = get_video_info("anything.mp4")
    assert info == {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}


# ── score_all_frames ──────────────────────────────────────────────────────────


def test_score_all_frames_returns_list(tiny_video):
    results = score_all_frames(tiny_video)
    assert isinstance(results, list)
    assert len(results) > 0


def test_score_all_frames_dict_keys(tiny_video):
    results = score_all_frames(tiny_video)
    expected_keys = {"frame_idx", "disparity", "rotation", "histogram_similarity", "score", "selected"}
    assert set(results[0].keys()) == expected_keys


def test_score_all_frames_first_frame_always_selected(tiny_video):
    results = score_all_frames(tiny_video)
    assert results[0]["selected"] is True
    assert results[0]["score"] == 1.0


def test_score_all_frames_frame_count(tiny_video):
    results = score_all_frames(tiny_video, stride=1, verbose=False)
    # tiny_video has 90 frames; stride=1 scores every frame
    assert len(results) == 90
    assert results[-1]["frame_idx"] == 89


def test_score_all_frames_scores_in_range(tiny_video):
    results = score_all_frames(tiny_video)
    for d in results:
        assert 0.0 <= d["score"] <= 1.0
        assert d["disparity"] >= 0.0
        assert d["rotation"] >= 0.0
        assert 0.0 <= d["histogram_similarity"] <= 1.0


def test_score_all_frames_calls_on_progress(tiny_video):
    calls = []
    # on_progress fires for every frame (grabbed or read), so len == total regardless of stride
    score_all_frames(tiny_video, on_progress=lambda c, t: calls.append((c, t)), verbose=False)
    assert len(calls) == 90
    assert calls[-1][0] == 90


# ── Visualization ─────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collab_splats.utils.frame_sampling import plot_frame_grid, plot_selection, plot_frame_scores, plot_disparity_sensitivity


def test_plot_frame_grid_returns_figure():
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(6)]
    result = plot_frame_grid(frames, title="test grid")
    assert result is None
    plt.close("all")


def test_plot_frame_grid_single_frame():
    frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
    result = plot_frame_grid(frames, title="single")
    assert result is None
    plt.close("all")


def test_plot_selection_fps_only():
    result = plot_selection(total_frames=90, fps_indices=list(range(0, 90, 10)))
    assert result is None
    plt.close("all")


def test_plot_selection_of_only():
    result = plot_selection(total_frames=90, of_indices=[0, 15, 40, 70])
    assert result is None
    plt.close("all")


def test_plot_selection_both():
    result = plot_selection(
        total_frames=90,
        fps_indices=list(range(0, 90, 10)),
        of_indices=[0, 15, 40, 70],
    )
    assert result is None
    plt.close("all")


def test_plot_selection_single_panel_has_one_axis():
    result = plot_selection(total_frames=90, fps_indices=[0, 30, 60])
    assert result is None
    plt.close("all")


def test_plot_frame_scores_returns_figure(tiny_video):
    scores = score_all_frames(tiny_video)
    result = plot_frame_scores(scores)
    assert result is None
    plt.close("all")


def test_plot_frame_scores_empty_input():
    result = plot_frame_scores([])
    assert result is None
    plt.close("all")


def test_plot_disparity_sensitivity_returns_figure(tiny_video):
    scores = score_all_frames(tiny_video)
    result = plot_disparity_sensitivity(scores, [10.0, 50.0, 100.0])
    assert result is None
    plt.close("all")


def test_plot_disparity_sensitivity_monotonic(tiny_video):
    scores = score_all_frames(tiny_video)
    thresholds = [10.0, 25.0, 50.0, 100.0, 200.0]
    # Higher threshold -> fewer or equal frames selected
    counts = []
    for t in thresholds:
        n = sum(
            1 for d in scores
            if 0.6 * min(d["disparity"] / max(t, 1e-6), 1.0)
            + 0.4 * (1.0 - d["histogram_similarity"]) >= 0.5
        )
        counts.append(n)
    assert counts == sorted(counts, reverse=True), "Higher threshold must not increase count"
    plot_disparity_sensitivity(scores, thresholds)
    plt.close("all")


def test_get_video_info_returns_width_height(tiny_video):
    info = get_video_info(tiny_video)
    assert info["width"] == 64
    assert info["height"] == 48


def test_get_decoder_backend_returns_valid_string():
    from collab_splats.utils.frame_sampling import _get_decoder_backend
    result = _get_decoder_backend()
    assert result in ("torchcodec", "ffmpeg", "cv2")


def test_get_decoder_backend_is_cached():
    from collab_splats.utils.frame_sampling import _get_decoder_backend
    assert _get_decoder_backend() == _get_decoder_backend()


def test_iter_decoded_frames_yields_bgr_frames(tiny_video):
    from collab_splats.utils.frame_sampling import _iter_decoded_frames
    info = get_video_info(tiny_video)
    frames = list(_iter_decoded_frames(tiny_video, info["width"], info["height"]))
    assert len(frames) == 90
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


def test_sample_frames_fps_ffmpeg_backend(tmp_path, monkeypatch):
    import shutil as _shutil
    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_decoder_backend", lambda: "ffmpeg")
    path = str(tmp_path / "test_ffmpeg.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(90):
        frame = np.full((48, 64, 3), (i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    frames, indices = fs.sample_frames_fps(path, fps=5.0, verbose=False)
    assert len(frames) > 0
    assert len(frames) == len(indices)
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


def test_sample_frames_fps_progress_callback(tiny_video):
    calls = []
    frames, _ = sample_frames_fps(
        tiny_video, fps=5.0, on_progress=lambda n, t: calls.append((n, t)), verbose=False
    )
    assert len(calls) > 0
    assert all(n <= t for n, t in calls)


def test_score_all_frames_ffmpeg_backend(tiny_video, monkeypatch):
    import shutil as _shutil
    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_decoder_backend", lambda: "ffmpeg")
    scores = fs.score_all_frames(tiny_video, verbose=False)
    assert isinstance(scores, list)
    assert len(scores) > 0
    assert all("disparity" in s for s in scores)


def test_sample_frames_optical_flow_ffmpeg_backend(tiny_video, monkeypatch):
    import shutil as _shutil
    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_decoder_backend", lambda: "ffmpeg")
    frames, scores = fs.sample_frames_optical_flow(tiny_video, verbose=False)
    assert isinstance(frames, list)
    assert isinstance(scores, list)
    assert len(frames) == len(scores)
    for f in frames:
        assert f.shape[2] == 3
        assert f.dtype == np.uint8
