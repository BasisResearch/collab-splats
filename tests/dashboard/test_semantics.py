"""Tests for pure helpers and structural layout of SemanticsPane."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import panel as pn
import pytest
import zarr

from collab_splats.dashboard.panes.semantics import _score_to_rgb, _load_frame_rgb
from collab_splats.dashboard.panes.semantics import SemanticsPane
from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.operation_log import OperationLog


def _make_semantics():
    return SemanticsPane(state=AppState(), op_log=OperationLog())


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


def test_semantics_has_single_method_dropdown():
    pane = _make_semantics()
    assert isinstance(pane._method_dd, pn.widgets.Select)


def test_semantics_has_three_image_panes():
    pane = _make_semantics()
    assert isinstance(pane._original_pane, pn.pane.PNG)
    assert isinstance(pane._pca_pane, pn.pane.PNG)
    assert isinstance(pane._sim_pane, pn.pane.PNG)


def test_semantics_no_extractor_columns():
    pane = _make_semantics()
    assert not hasattr(pane, "_columns"), "ExtractorColumn list should be gone"


def test_semantics_query_btn_disabled_by_default():
    pane = _make_semantics()
    assert pane._query_btn.disabled is True


def test_semantics_panel_returns_column():
    pane = _make_semantics()
    result = pane.panel()
    assert isinstance(result, pn.Column)


def test_panel_extractor_row_is_first_content():
    """Extractor controls are col.objects[1] (after H3 header)."""
    pane = _make_semantics()
    col = pane.panel()
    extractor_row = col.objects[1]
    assert isinstance(extractor_row, pn.Row)
    assert pane._method_dd in extractor_row.objects


def test_panel_image_area_has_fixed_height():
    """Image area is a Column with height=270 to prevent reflow."""
    pane = _make_semantics()
    col = pane.panel()
    image_area = col.objects[2]
    assert isinstance(image_area, pn.Column)
    assert image_area.height == 270


def test_panel_frame_slider_below_image_area():
    """Frame slider row is col.objects[3], after the image area."""
    pane = _make_semantics()
    col = pane.panel()
    frame_row = col.objects[3]
    assert isinstance(frame_row, pn.Row)
    assert pane._frame_slider in frame_row.objects


def test_panel_query_row_is_last():
    """Query row is col.objects[4] (last)."""
    pane = _make_semantics()
    col = pane.panel()
    query_row = col.objects[4]
    assert isinstance(query_row, pn.Row)
    assert pane._query_input in query_row.objects


def test_panel_status_html_in_extractor_row():
    """status_html lives inside extractor_row, not as a standalone column child."""
    pane = _make_semantics()
    col = pane.panel()
    extractor_row = col.objects[1]
    assert pane._status_html in extractor_row.objects
    # Must NOT be a top-level child of the column
    assert pane._status_html not in col.objects
