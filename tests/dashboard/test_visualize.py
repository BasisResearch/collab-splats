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
    # No result loaded yet — mode selector should be disabled
    assert sp._mode_selector.disabled is True


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
    def _set_actor(*a, **kw):
        sp._mesh_actor = mock.MagicMock()

    with mock.patch.object(sp, "_rebuild_mesh_viewer", side_effect=_set_actor) as mock_rebuild:
        sp._scan_available_modes()
    mock_rebuild.assert_called_once()


def test_scan_available_modes_adds_mesh_when_ply_exists(tmp_path):
    """When mesh.ply exists, Mesh is added to available modes."""
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
    assert "Mesh" in sp._available_modes


def test_scan_available_modes_no_mesh_without_ply(tmp_path):
    """When mesh.ply is absent, Mesh is not in available modes."""
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    sp._scan_available_modes()
    assert "Mesh" not in sp._available_modes


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


def test_scene_panel_default_mode_is_points(tmp_path):
    sp = _make_scene(tmp_path)
    assert sp._mode_selector.value == "Points"



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


def test_scene_panel_mesh_mode_hides_points_and_sim_rows(tmp_path):
    sp = _make_scene(tmp_path)
    sp._available_modes = {"Mesh"}
    sp._result = mock.MagicMock()
    sp._plotter = mock.MagicMock()
    sp._vtk_pane = mock.MagicMock()
    with mock.patch.object(sp, "_rebuild_mesh_viewer"):
        sp._on_mode_change("Mesh")
    assert sp._points_options_row.visible is False
    assert sp._sim_query_row.visible is False


########################################################################
# _on_run_mesh / _run_mesh_worker tests
########################################################################


def test_on_run_mesh_spawns_subprocess_with_zarr_path(tmp_path):
    """_run_mesh_worker spawns a subprocess using the zarr path."""
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._current_dataset_dir = tmp_path / "scene_01"
    sp._current_backend = "vggt_x"
    sp._mesh_voxel = 0.05
    sp._mesh_sdf = 0.15
    sp._mesh_depth = 3.0
    sp._mesh_clean = True
    sp._mesh_on_done = None

    proc_mock = mock.MagicMock()
    proc_mock.exitcode = 0

    with mock.patch("collab_splats.dashboard.panes.visualize.multiprocessing.Process",
                    return_value=proc_mock) as mock_proc_cls, \
         mock.patch("panel.io.state._state.execute"), \
         mock.patch.object(sp, "_refresh_after_mesh"):
        sp._run_mesh_worker()

    mock_proc_cls.assert_called_once()
    call_kwargs = mock_proc_cls.call_args
    assert str(tmp_path / "scene_01" / "vggt_x" / "feedforward.zarr") in call_kwargs[1]["args"]
    proc_mock.start.assert_called_once()
    proc_mock.join.assert_called_once()


def test_on_run_mesh_reports_failure_on_nonzero_exit(tmp_path):
    """_run_mesh_worker calls on_done with ok=False when subprocess exits non-zero."""
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    sp._current_dataset_dir = tmp_path / "scene_01"
    sp._current_backend = "vggt_x"
    sp._mesh_voxel = 0.05
    sp._mesh_sdf = 0.15
    sp._mesh_depth = 3.0
    sp._mesh_clean = True
    results = []
    sp._mesh_on_done = lambda ok, msg: results.append((ok, msg))

    proc_mock = mock.MagicMock()
    proc_mock.exitcode = -9  # OOM kill

    with mock.patch("collab_splats.dashboard.panes.visualize.multiprocessing.Process",
                    return_value=proc_mock), \
         mock.patch("panel.io.state._state.execute", side_effect=lambda f: f()):
        sp._run_mesh_worker()

    assert results and results[0][0] is False


########################################################################
# Ground plane apply/invert tests
########################################################################

import dataclasses


def _make_fake_result(n_pts: int = 5) -> "FeedforwardResult":  # type: ignore[name-defined]
    """Build a minimal FeedforwardResult-like object for ground plane tests."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((n_pts, 3)).astype(np.float32)
    colors = (rng.random((n_pts, 3)) * 255).astype(np.uint8)
    extrinsics = np.tile(np.eye(4, dtype=np.float64), (n_pts, 1, 1))
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (n_pts, 1, 1))
    return FeedforwardResult(
        points=pts,
        colors=colors,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=[Path(f"frame_{i:04d}.jpg") for i in range(n_pts)],
        original_coords=np.zeros((n_pts, 6), dtype=np.float32),
        model_width=256,
        model_height=256,
    )


def test_apply_ground_plane_passthrough_when_no_r(tmp_path):
    """_apply_ground_plane returns result unchanged when ground_plane_R is None."""
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    result = _make_fake_result()
    out = sp._apply_ground_plane(result)
    # Should be the identical object — no transform applied
    assert out is result


def test_apply_ground_plane_translates_points(tmp_path):
    """Identity rotation with t=[0,0,1] shifts all z coords by +1."""
    state = AppState()
    state.ground_plane_R = np.eye(3, dtype=np.float64)
    state.ground_plane_t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    state.ground_plane_enabled = True
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    result = _make_fake_result()
    pts_orig = result.points.copy()
    out = sp._apply_ground_plane(result)
    np.testing.assert_allclose(out.points[:, :2], pts_orig[:, :2], atol=1e-5)
    np.testing.assert_allclose(out.points[:, 2], pts_orig[:, 2] + 1.0, atol=1e-5)


def test_apply_ground_plane_invert_round_trips(tmp_path):
    """Apply then invert restores original points (round-trip, atol=1e-5)."""
    state = AppState()
    rng = np.random.default_rng(7)
    # Random rotation via QR decomposition
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    t = rng.standard_normal(3)
    state.ground_plane_R = Q.astype(np.float64)
    state.ground_plane_t = t.astype(np.float64)
    state.ground_plane_enabled = True
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    result = _make_fake_result()
    pts_orig = result.points.copy()

    # Apply forward transform
    out_fwd = sp._apply_ground_plane(result)

    # Apply inverse
    state.ground_plane_enabled = False
    out_inv = sp._apply_ground_plane(out_fwd)

    np.testing.assert_allclose(out_inv.points, pts_orig, atol=1e-5)
