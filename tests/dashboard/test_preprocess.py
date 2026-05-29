import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np
import panel as pn
import pytest
import zarr

from collab_splats.dashboard.operation_log import OperationLog
import time

from collab_splats.dashboard.panes.preprocess import (
    PreprocessPane,
    ThrottledProgress,
    _build_metrics_sources,
    _frames_to_thumbnails,
    _render_fps_raster,
    _resize_to_max_width,
    _window_frame_indices,
    _write_frames_zarr,
)
from collab_splats.dashboard.state import AppState


@pytest.fixture
def mock_video_server():
    """Minimal stand-in for VideoFileServer — avoids binding a real port in unit tests."""
    srv = mock.MagicMock()
    srv.port = 17863
    return srv


def test_window_frame_indices_full():
    indices = _window_frame_indices(total_frames=100, window_start=0.0, window_end=1.0)
    assert indices == list(range(100))


def test_window_frame_indices_half():
    indices = _window_frame_indices(total_frames=100, window_start=0.25, window_end=0.75)
    assert indices[0] == 25
    assert indices[-1] == 74
    assert len(indices) == 50


def test_window_frame_indices_clamps():
    indices = _window_frame_indices(total_frames=100, window_start=-0.1, window_end=1.5)
    assert indices == list(range(100))


def test_build_metrics_sources_returns_dict():
    frame_scores = {
        "disparity": [float(i) for i in range(59)],
        "rotation": [float(i * 0.1) for i in range(59)],
        "hist_similarity": [0.9 - i * 0.01 for i in range(59)],
    }
    sources = _build_metrics_sources(frame_scores)
    assert isinstance(sources, dict)
    assert len(sources) > 0


def test_build_metrics_sources_empty_scores():
    sources = _build_metrics_sources({})
    assert isinstance(sources, dict)
    assert len(sources) == 0


def test_frames_to_thumbnails_returns_bytes_list():
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    thumbnails = _frames_to_thumbnails(frames)
    assert len(thumbnails) == 5
    assert all(isinstance(t, bytes) for t in thumbnails)
    assert all(len(t) > 100 for t in thumbnails)


def test_frames_to_thumbnails_empty():
    assert _frames_to_thumbnails([]) == []


def test_write_frames_zarr_shape():
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "frames.zarr"
        _write_frames_zarr(frames, zarr_path)
        z = zarr.open(str(zarr_path), mode="r")
        arr = z["frames"]
        assert arr.shape == (5, 48, 64, 3)


def test_write_frames_zarr_roundtrip():
    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "frames.zarr"
        _write_frames_zarr([frame], zarr_path)
        z = zarr.open(str(zarr_path), mode="r")
        np.testing.assert_array_equal(z["frames"][0], frame)


def test_video_path_change_loads_video(tmp_path, mock_video_server):
    """_on_video_path_change must call _load_video when video path is valid."""
    state = AppState()
    pane = PreprocessPane(state=state, op_log=OperationLog(), video_server=mock_video_server)

    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"fake")

    class FakeEvent:
        new = str(video_file)

    with mock.patch.object(pane, "_load_video") as mock_load:
        pane._on_video_path_change(FakeEvent())

    mock_load.assert_called_once_with(video_file)


def test_throttled_progress_rate_limits():
    """ThrottledProgress fires at most max_hz times per second, plus first and last."""
    calls = []
    tp = ThrottledProgress(lambda n, t: calls.append((n, t)), max_hz=10.0)
    for i in range(1, 101):
        tp(i, 100)
    assert calls[0] == (1, 100)
    assert calls[-1] == (100, 100)
    assert len(calls) <= 5


def test_render_fps_raster_returns_panel_column():
    result = _render_fps_raster([0, 15, 30, 45, 60], total_frames=90)
    assert isinstance(result, pn.Column)


def test_render_fps_raster_empty_indices():
    result = _render_fps_raster([], total_frames=90)
    assert isinstance(result, pn.Column)


def test_throttled_progress_always_fires_final():
    """ThrottledProgress always fires when n == total even under heavy throttling."""
    calls = []
    tp = ThrottledProgress(lambda n, t: calls.append(n), max_hz=0.01)
    tp(50, 100)
    tp(100, 100)
    assert 100 in calls


def test_resize_to_max_width_no_op_when_under():
    """Frames at or below max_width are returned unchanged."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = _resize_to_max_width(frame, 1920)
    assert result.shape == (480, 640, 3)


def test_resize_to_max_width_caps_wide_frame():
    """Frames wider than max_width are resized; height scaled proportionally."""
    frame = np.zeros((1080, 2000, 3), dtype=np.uint8)
    result = _resize_to_max_width(frame, 1920)
    assert result.shape[1] == 1920
    assert result.shape[0] == int(round(1080 * 1920 / 2000))
    assert result.shape[2] == 3
    assert result.dtype == np.uint8


def test_write_frames_zarr_wide_frames_get_resized(tmp_path):
    """Frames wider than 1920px are stored at 1920px; height scaled."""
    wide = np.zeros((1080, 2000, 3), dtype=np.uint8)
    zarr_path = tmp_path / "wide.zarr"
    _write_frames_zarr([wide, wide], zarr_path)
    z = zarr.open(str(zarr_path), mode="r")
    assert z["frames"].shape == (2, int(round(1080 * 1920 / 2000)), 1920, 3)


def test_write_frames_zarr_streams_one_at_a_time(tmp_path, monkeypatch):
    """zarr array is written frame-by-frame — np.stack must not be called."""
    stack_calls = []
    import numpy as _np
    original_stack = _np.stack

    def spy_stack(*args, **kwargs):
        stack_calls.append(args)
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(_np, "stack", spy_stack)
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    zarr_path = tmp_path / "stream.zarr"
    _write_frames_zarr(frames, zarr_path)
    assert len(stack_calls) == 0, "np.stack must not be called — streaming write only"
