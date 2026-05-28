from unittest.mock import MagicMock, patch
import numpy as np
import panel as pn
import pytest

pn.extension()


def _make_scene_panel():
    from collab_splats.dashboard.panes.localize import LocalizeScenePanel
    pts3d = np.zeros((10, 3), dtype=np.float32)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 4)
    return LocalizeScenePanel(pts3d=pts3d, extrinsics=extrinsics, image_paths=[])


def test_scene_panel_constructs():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
    assert panel is not None


def test_scene_panel_reset_clears_highlight():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
        panel._highlighted_query_idx = 2
        panel._highlighted_ref_idx = 1
        panel.reset()
    assert panel._highlighted_query_idx is None
    assert panel._highlighted_ref_idx is None


def test_scene_panel_highlight_sets_indices():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
        panel.highlight(query_ext=np.eye(4), ref_ext=np.eye(4), query_idx=3, ref_idx=1)
    assert panel._highlighted_query_idx == 3
    assert panel._highlighted_ref_idx == 1


from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes.localize import LocalizePane
from collab_splats.dashboard.state import AppState


def _make_pane():
    state = AppState()
    op_log = OperationLog()
    return LocalizePane(state=state, op_log=op_log), state, op_log


def test_localize_pane_run_btn_disabled_without_output_dir():
    pane, _, _ = _make_pane()
    assert pane._run_btn.disabled is True


def test_localize_pane_run_btn_still_disabled_without_query_image(tmp_path):
    # output_dir set but no query image — run still disabled
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    assert pane._run_btn.disabled is True


def test_localize_pane_method_dropdown_empty_without_output_dir():
    pane, _, _ = _make_pane()
    assert pane._method_dd.options == []


def test_localize_pane_method_dropdown_populated_when_zarr_found(tmp_path):
    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    assert "vggtx" in pane._method_dd.options


def test_localize_pane_run_btn_enabled_when_all_conditions_met(tmp_path):
    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    pane._query_input.value = str(tmp_path / "query.jpg")
    assert pane._run_btn.disabled is False
