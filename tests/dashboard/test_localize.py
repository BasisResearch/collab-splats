# tests/dashboard/test_localize.py
def test_appstate_has_localize_fields():
    from collab_splats.dashboard.state import AppState
    s = AppState()
    assert s.localize_method == ""
    assert s.localize_extractor == "DISK+LightGlue"

def test_appstate_localize_method_is_watchable():
    from collab_splats.dashboard.state import AppState
    s = AppState()
    seen = []
    s.param.watch(lambda e: seen.append(e.new), ["localize_method"])
    s.localize_method = "vggtx"
    assert seen == ["vggtx"]


from unittest.mock import MagicMock
import numpy as np


def _make_state(output_dir=None, localize_method="vggtx", localize_extractor="DISK+LightGlue"):
    from collab_splats.dashboard.state import AppState
    s = AppState()
    s.output_dir = output_dir
    s.localize_method = localize_method
    s.localize_extractor = localize_extractor
    return s


def _make_op_log():
    from collab_splats.dashboard.operation_log import OperationLog
    return OperationLog()


def test_localize_pane_run_btn_disabled_without_method():
    from collab_splats.dashboard.panes.localize import LocalizePane
    state = _make_state(localize_method="")
    pane = LocalizePane(state=state, op_log=_make_op_log())
    assert pane._run_btn.disabled


def test_localize_pane_run_btn_disabled_without_query():
    from collab_splats.dashboard.panes.localize import LocalizePane
    state = _make_state(output_dir="/tmp", localize_method="vggtx")
    pane = LocalizePane(state=state, op_log=_make_op_log())
    # method set, output_dir set, but no query path
    assert pane._run_btn.disabled


def test_localize_pane_cache_invalidated_on_method_change():
    from collab_splats.dashboard.panes.localize import LocalizePane
    state = _make_state(localize_method="vggtx")
    pane = LocalizePane(state=state, op_log=_make_op_log())
    pane._localizer = object()  # simulate cached localizer
    state.localize_method = "mapanything"
    assert pane._localizer is None


def test_localize_pane_cache_invalidated_on_extractor_change():
    from collab_splats.dashboard.panes.localize import LocalizePane
    state = _make_state(localize_method="vggtx")
    pane = LocalizePane(state=state, op_log=_make_op_log())
    pane._localizer = object()
    state.localize_extractor = "XFeat+MNN"
    assert pane._localizer is None


def test_localize_pane_has_inline_method_dd():
    """LocalizePane has a _method_dd dropdown populated from zarr dirs on output_dir change."""
    from collab_splats.dashboard.panes.localize import LocalizePane
    state = _make_state()
    pane = LocalizePane(state=state, op_log=_make_op_log())
    assert hasattr(pane, "_method_dd")


def test_localize_scene_panel_off_screen():
    from collab_splats.dashboard.panes.localize import LocalizeScenePanel
    pts3d = np.zeros((5, 3), dtype=np.float32)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 3)
    panel = LocalizeScenePanel(
        pts3d=pts3d,
        extrinsics=extrinsics,
        image_paths=[],
        _off_screen=True,
    )
    assert panel.panel() is not None
