"""Tests for pure helpers in SemanticsPane."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from collab_splats.dashboard.panes.semantics import _score_to_rgb, _load_frame_rgb


def test_score_to_rgb_shape():
    score = np.random.rand(8, 10).astype(np.float32)
    rgb = _score_to_rgb(score)
    assert rgb.shape == (8, 10, 3)
    assert rgb.dtype == np.uint8


def test_score_to_rgb_range():
    score = np.random.rand(4, 4).astype(np.float32)
    rgb = _score_to_rgb(score)
    assert rgb.min() >= 0
    assert rgb.max() <= 255


def test_score_to_rgb_uniform_input():
    score = np.ones((6, 6), dtype=np.float32) * 0.5
    rgb = _score_to_rgb(score)
    assert rgb.shape == (6, 6, 3)


def test_load_frame_rgb_returns_array(tmp_path):
    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    zarr_path = tmp_path / "frames.zarr"
    store = zarr.open(str(zarr_path), mode="w")
    store.attrs.update({"n_frames": 1, "height": 48, "width": 64})
    arr = store.create_array(
        "frames", shape=(1, 48, 64, 3), chunks=(1, 48, 64, 3), dtype="uint8",
    )
    arr[0] = frame
    result = _load_frame_rgb(zarr_path, 0)
    assert result.shape == (48, 64, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, frame)
