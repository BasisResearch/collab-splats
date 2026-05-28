import tempfile
from pathlib import Path

import numpy as np
import zarr

from collab_splats.dashboard.panes.preprocess import (
    _window_frame_indices,
    _render_metrics_figure,
    _frames_to_thumbnails,
    _write_frames_zarr,
)


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


def test_render_metrics_figure_returns_bytes():
    frame_scores = {
        "disparity": [float(i) for i in range(59)],
        "rotation": [float(i * 0.1) for i in range(59)],
        "hist_similarity": [0.9 - i * 0.01 for i in range(59)],
    }
    selected = [0, 10, 20, 30, 40, 50]
    png_bytes = _render_metrics_figure(frame_scores, selected_indices=selected, total_frames=60)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000


def test_render_metrics_figure_empty_scores():
    png_bytes = _render_metrics_figure({}, selected_indices=[], total_frames=0)
    assert isinstance(png_bytes, bytes)


def test_frames_to_thumbnails_returns_bytes_list():
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    thumbnails = _frames_to_thumbnails(frames)
    assert len(thumbnails) == 5
    assert all(isinstance(t, bytes) for t in thumbnails)
    assert all(len(t) > 100 for t in thumbnails)


def test_frames_to_thumbnails_empty():
    assert _frames_to_thumbnails([]) == []


def test_write_frames_zarr_shape_and_attrs():
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "frames.zarr"
        result = _write_frames_zarr(frames, zarr_path)
        assert result == zarr_path
        z = zarr.open(str(zarr_path), mode="r")
        arr = z["frames"]
        assert arr.shape == (5, 48, 64, 3)
        assert z.attrs["n_frames"] == 5
        assert z.attrs["height"] == 48
        assert z.attrs["width"] == 64


def test_write_frames_zarr_roundtrip():
    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "frames.zarr"
        _write_frames_zarr([frame], zarr_path)
        z = zarr.open(str(zarr_path), mode="r")
        np.testing.assert_array_equal(z["frames"][0], frame)
