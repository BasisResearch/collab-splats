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
        scene_id="A",
        base_dir=tmp_path,
        state=state,
        op_log=op_log,
        _off_screen=True,
    )
    assert sp._scene_id == "A"
    assert "scene_01" in sp._dataset_dd.options


def test_scene_panel_scan_available_modes_no_result(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    state = AppState()
    sp = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)
    sp._scan_available_modes()
    assert "PCD" not in sp._available_modes


def test_scene_panel_scan_available_modes_with_mesh(tmp_path):
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    mesh_dir = tmp_path / "scene_01" / "vggt_x" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh.ply").write_bytes(b"ply\n")
    state = AppState()
    sp = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._result = object()  # non-None sentinel
    sp._scan_available_modes()
    assert "Mesh" in sp._available_modes


def test_scene_panel_scan_available_modes_with_features(tmp_path):
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",), extractors=("talk2dino",))
    state = AppState()
    sp = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)
    sp._dataset_dd.value = "scene_01"
    sp._backend_dd.value = "vggt_x"
    sp._result = object()
    sp._scan_available_modes()
    assert "Similarity" in sp._available_modes
    assert "talk2dino" in sp._available_extractors


########################################################################
# VisualizePane smoke tests
########################################################################

import unittest.mock as mock

from collab_splats.dashboard.panes.visualize import VisualizePane


def test_visualize_pane_constructs(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    state = AppState()
    op_log = _make_op_log()
    # Patch ScenePanel to use off_screen so no display needed
    with mock.patch(
        "collab_splats.dashboard.panes.visualize.ScenePanel",
        lambda *a, **kw: ScenePanel(*a, **{**kw, "_off_screen": True}),
    ):
        vp = VisualizePane(state=state, op_log=op_log, base_dir=tmp_path)
    assert vp._scene_a._scene_id == "A"
    assert vp._scene_b._scene_id == "B"
    assert vp._query_bar.visible is False


def test_visualize_pane_query_bar_visible_on_similarity(tmp_path):
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",), extractors=("talk2dino",))
    state = AppState()
    op_log = _make_op_log()
    with mock.patch(
        "collab_splats.dashboard.panes.visualize.ScenePanel",
        lambda *a, **kw: ScenePanel(*a, **{**kw, "_off_screen": True}),
    ):
        vp = VisualizePane(state=state, op_log=op_log, base_dir=tmp_path)
    # Simulate Scene A entering Similarity mode
    vp._scene_a._available_modes = {"PCD", "Similarity"}
    vp._scene_a._available_extractors = ["talk2dino"]
    vp._scene_a.mode = "Similarity"
    # Manually call the watcher (param.watch fires async in tests)
    vp._on_scene_mode_change(None)
    assert vp._query_bar.visible is True
    assert vp._a_extractor_dd.visible is True
