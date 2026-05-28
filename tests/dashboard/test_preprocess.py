import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np
import panel as pn
import zarr

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes.preprocess import (
    PreprocessPane,
    _build_metrics_sources,
    _frames_to_thumbnails,
    _window_frame_indices,
    _write_frames_zarr,
)
from collab_splats.dashboard.state import AppState


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


def test_video_path_change_uses_pn_state_execute(tmp_path):
    """Watch callback must use pn.state.execute so update runs on Tornado main thread."""
    state = AppState()
    pane = PreprocessPane(state=state, op_log=OperationLog())

    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"fake")

    execute_calls = []

    class FakeEvent:
        new = str(video_file)

    with mock.patch("collab_splats.dashboard.panes.preprocess.pn.state") as mock_state:
        mock_state.execute = lambda fn: execute_calls.append(fn)
        pane._on_video_path_change(FakeEvent())

    assert len(execute_calls) == 1, "pn.state.execute should be called once"
    assert callable(execute_calls[0])
