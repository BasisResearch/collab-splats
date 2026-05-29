import json
from pathlib import Path

import numpy as np
import param
import zarr
import zarr.codecs

from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.panes.visualize import (
    _scan_datasets,
    _scan_backends,
    _scan_extractors,
    _apply_viridis,
    _load_lifted_features,
)


def test_appstate_has_frames_zarr_path():
    state = AppState()
    assert hasattr(state, "frames_zarr_path")
    assert state.frames_zarr_path is None


def test_appstate_no_frames_list():
    state = AppState()
    assert not hasattr(state, "frames"), "frames list removed in Phase 2"


########################################################################
# Discovery helpers
########################################################################


def _make_dataset(tmp_path: Path, name: str, backends=("vggt_x",), extractors=()) -> Path:
    """Create a minimal fake output directory tree."""
    ds = tmp_path / name
    for backend in backends:
        be = ds / backend
        be.mkdir(parents=True)
        store = zarr.open(str(be / "feedforward.zarr"), mode="w")
        store.attrs["image_paths"] = []
        for extractor in extractors:
            feat_dir = be / "semantics" / extractor
            feat_dir.mkdir(parents=True)
            feat_store = zarr.open(str(feat_dir / "features.zarr"), mode="w")
            feat_store.create_array("features", data=np.zeros((10, 64), dtype=np.float32))
    (ds / "run_config.yaml").write_text("backend: vggt_x\n")
    return ds


def test_scan_datasets_finds_run_config(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    _make_dataset(tmp_path, "scene_02")
    (tmp_path / "not_a_dataset").mkdir()
    result = _scan_datasets(tmp_path)
    names = [p.name for p in result]
    assert "scene_01" in names
    assert "scene_02" in names
    assert "not_a_dataset" not in names


def test_scan_backends(tmp_path):
    ds = _make_dataset(tmp_path, "scene", backends=("vggt_x", "mapanything"))
    result = _scan_backends(ds)
    assert set(result) == {"vggt_x", "mapanything"}


def test_scan_extractors(tmp_path):
    ds = _make_dataset(tmp_path, "scene", backends=("vggt_x",), extractors=("talk2dino",))
    result = _scan_extractors(ds, "vggt_x")
    assert result == ["talk2dino"]


def test_scan_extractors_empty(tmp_path):
    ds = _make_dataset(tmp_path, "scene", backends=("vggt_x",))
    result = _scan_extractors(ds, "vggt_x")
    assert result == []


########################################################################
# Viridis colormap
########################################################################


def test_apply_viridis_shape():
    sims = np.array([-1.0, 0.0, 0.5, 1.0])
    colors = _apply_viridis(sims)
    assert colors.shape == (4, 3)
    assert colors.dtype == np.uint8


def test_apply_viridis_constant():
    sims = np.ones(5)
    colors = _apply_viridis(sims)
    assert colors.shape == (5, 3)


########################################################################
# Lifted features
########################################################################


def _make_lifted_zarr(tmp_path: Path, n_points: int = 20, dim: int = 64) -> Path:
    store_path = tmp_path / "features.zarr"
    store = zarr.open(str(store_path), mode="w")
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((n_points, dim)).astype(np.float32)
    store.create_array("features", data=feats)
    return store_path


def test_load_lifted_features_normalized(tmp_path):
    store_path = _make_lifted_zarr(tmp_path)
    normed = _load_lifted_features(store_path)
    assert normed.shape == (20, 64)
    assert normed.dtype == np.float32
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, np.ones(20), atol=1e-5)


def test_load_lifted_features_zero_norm(tmp_path):
    store_path = tmp_path / "features.zarr"
    store = zarr.open(str(store_path), mode="w")
    feats = np.zeros((5, 16), dtype=np.float32)
    feats[1] = 1.0
    store.create_array("features", data=feats)
    normed = _load_lifted_features(store_path)
    assert not np.any(np.isnan(normed))


########################################################################
# ScenePanel smoke tests
########################################################################

import panel as pn
from collab_splats.dashboard.panes.visualize import ScenePanel

pn.extension("vtk")


def _make_op_log():
    """Instantiate OperationLog from its canonical module."""
    from collab_splats.dashboard.operation_log import OperationLog
    return OperationLog()


def test_scene_panel_constructs(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    state = AppState()
    op_log = _make_op_log()
    sp = ScenePanel(
        base_dir=tmp_path,
        state=state,
        op_log=op_log,
        _off_screen=True,
    )
    assert "scene_01" in sp._dataset_dd.options


def test_scene_panel_scan_available_modes_no_result(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._scan_available_modes()
    assert "PCD" not in sp._available_modes


def test_scene_panel_scan_available_modes_with_mesh(tmp_path):
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    mesh_dir = tmp_path / "scene_01" / "vggt_x" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh.ply").write_bytes(b"ply\n")
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._result = mock.MagicMock()  # non-None sentinel
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_mesh_viewer"):
        sp._scan_available_modes()
    assert "Mesh" in sp._available_modes


def test_scan_available_modes_auto_displays_mesh(tmp_path):
    """When mesh.ply exists, _scan_available_modes calls _rebuild_mesh_viewer."""
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    mesh_dir = tmp_path / "scene_01" / "vggt_x" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh.ply").write_bytes(b"ply\n")
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_mesh_viewer") as mock_rebuild:
        sp._scan_available_modes()
    mock_rebuild.assert_called_once()


def test_scan_available_modes_enables_run_btn_when_zarr_exists(tmp_path):
    """When feedforward.zarr exists, _mesh_run_btn is enabled after scan."""
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    mesh_dir = tmp_path / "scene_01" / "vggt_x" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh.ply").write_bytes(b"ply\n")
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_mesh_viewer"):
        sp._scan_available_modes()
    # feedforward.zarr was created by _make_dataset
    assert sp._mesh_run_btn.disabled is False


def test_scan_available_modes_run_btn_disabled_without_zarr(tmp_path):
    """When feedforward.zarr is absent, _mesh_run_btn stays disabled."""
    # Create dataset dir with mesh but NO feedforward.zarr
    ds = tmp_path / "scene_01"
    be = ds / "vggt_x"
    be.mkdir(parents=True)
    (ds / "run_config.yaml").write_text("backend: vggt_x\n")
    mesh_dir = be / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh.ply").write_bytes(b"ply\n")
    # No feedforward.zarr in be/
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_mesh_viewer"):
        sp._scan_available_modes()
    assert sp._mesh_run_btn.disabled is True


def test_scene_panel_scan_available_modes_with_features(tmp_path):
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",), extractors=("talk2dino",))
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._result = mock.MagicMock()
    # Mock plotter + vtk_pane + rebuild so _update_mode_buttons rendering is bypassed
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_sim_viewer"), mock.patch.object(sp, "_rebuild_pcd_viewer"):
        sp._scan_available_modes()
    assert "Similarity" in sp._available_modes
    assert "talk2dino" in sp._available_extractors


########################################################################
# ScenePanel layout / new widget tests
########################################################################

import unittest.mock as mock

from collab_splats.dashboard.operation_log import OperationLog


def _make_scene(tmp_path):
    """Helper: construct a ScenePanel with off-screen rendering."""
    state = AppState()
    return ScenePanel(tmp_path, state, OperationLog(), _off_screen=True)


def test_scene_panel_has_radio_button_group(tmp_path):
    sp = _make_scene(tmp_path)
    assert isinstance(sp._mode_selector, pn.widgets.RadioButtonGroup)


def test_scene_panel_default_mode_is_mesh(tmp_path):
    sp = _make_scene(tmp_path)
    assert sp._mode_selector.value == "Mesh"


def test_scene_panel_frustum_is_checkbox(tmp_path):
    sp = _make_scene(tmp_path)
    assert isinstance(sp._frustum_check, pn.widgets.Checkbox)


def test_scene_panel_has_sim_query_row(tmp_path):
    sp = _make_scene(tmp_path)
    assert hasattr(sp, "_sim_query_row")
    assert sp._sim_query_row.visible is False
    assert isinstance(sp._extractor_dd, pn.widgets.Select)


def test_scene_panel_has_points_options_row(tmp_path):
    sp = _make_scene(tmp_path)
    assert hasattr(sp, "_points_options_row")
    assert sp._points_options_row.visible is False


def test_scene_panel_sim_query_row_visible_in_similarity_mode(tmp_path):
    sp = _make_scene(tmp_path)
    sp._available_modes = {"PCD", "Similarity"}
    sp._result = mock.MagicMock()  # non-None, has any attr accessed
    # Mock plotter + vtk_pane + rebuild to isolate visibility logic from rendering
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_sim_viewer"):
        sp._on_mode_change("Similarity")
    assert sp._sim_query_row.visible is True
    assert sp._points_options_row.visible is False


def test_scene_panel_points_options_visible_in_points_mode(tmp_path):
    sp = _make_scene(tmp_path)
    sp._available_modes = {"PCD"}
    sp._result = mock.MagicMock()  # non-None, has any attr accessed
    # Mock plotter + vtk_pane + rebuild to isolate visibility logic from rendering
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_pcd_viewer"):
        sp._on_mode_change("Points")
    assert sp._points_options_row.visible is True
    assert sp._sim_query_row.visible is False


def test_scene_panel_wire_tabs(tmp_path):
    """wire_tabs fires rescan when the wired tab becomes active."""
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    tabs = pn.Tabs(("Visualize", pn.pane.Str("x")), ("Other", pn.pane.Str("y")))
    sp.wire_tabs(tabs, 0)
    tabs.active = 1  # switch away
    tabs.active = 0  # switch back → triggers rescan
    assert isinstance(sp._available_modes, (set, frozenset))


def test_scene_panel_has_mesh_options_row(tmp_path):
    sp = _make_scene(tmp_path)
    assert hasattr(sp, "_mesh_options_row")
    assert sp._mesh_options_row.visible is False


def test_scene_panel_mesh_options_row_visible_in_mesh_mode(tmp_path):
    sp = _make_scene(tmp_path)
    sp._available_modes = {"Mesh"}
    sp._result = mock.MagicMock()
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_mesh_viewer"):
        sp._on_mode_change("Mesh")
    assert sp._mesh_options_row.visible is True
    assert sp._points_options_row.visible is False
    assert sp._sim_query_row.visible is False


def test_scene_panel_mesh_options_row_hidden_in_points_mode(tmp_path):
    sp = _make_scene(tmp_path)
    sp._available_modes = {"PCD"}
    sp._result = mock.MagicMock()
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_pcd_viewer"):
        sp._on_mode_change("Points")
    assert sp._mesh_options_row.visible is False


########################################################################
# _on_run_mesh / _run_mesh_worker tests
########################################################################


def test_on_run_mesh_uses_existing_result(tmp_path):
    """_run_mesh_worker uses _result if already loaded rather than loading zarr."""
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._current_dataset_dir = tmp_path / "scene_01"
    sp._current_backend = "vggt_x"
    sp._result = mock.MagicMock()  # pre-loaded result

    fake_mesh_result = mock.MagicMock()
    fake_mesh_result.mesh_path = tmp_path / "mesh.ply"

    with mock.patch("collab_splats.mesh.utils.pointcloud_to_mesh", return_value=fake_mesh_result) as mock_ptm, \
         mock.patch("panel.io.state._state.execute"):
        sp._run_mesh_worker()

    # pointcloud_to_mesh called with the pre-loaded _result, not load_zarr
    mock_ptm.assert_called_once()
    call_args = mock_ptm.call_args
    assert call_args[0][0] is sp._result  # first positional arg = result


def test_on_run_mesh_loads_zarr_when_result_none(tmp_path):
    """_run_mesh_worker loads feedforward.zarr if _result is None."""
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._current_dataset_dir = tmp_path / "scene_01"
    sp._current_backend = "vggt_x"
    sp._result = None

    fake_result = mock.MagicMock()
    fake_mesh_result = mock.MagicMock()
    fake_mesh_result.mesh_path = tmp_path / "mesh.ply"

    with mock.patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr",
                    return_value=fake_result) as mock_load, \
         mock.patch("collab_splats.mesh.utils.pointcloud_to_mesh",
                    return_value=fake_mesh_result), \
         mock.patch("panel.io.state._state.execute"):
        sp._run_mesh_worker()

    mock_load.assert_called_once()
