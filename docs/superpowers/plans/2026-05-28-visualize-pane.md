# VisualizePane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `VisualizePane` (Tab 4) — side-by-side interactive 3D comparison of two independently-selected scenes with PCD, Mesh, and Similarity modes.

**Architecture:** `ScenePanel(param.Parameterized)` is a self-contained per-scene unit owning a `pv.Plotter` + `pn.pane.VTK`. `VisualizePane(param.Parameterized)` composes two `ScenePanel` instances and a shared semantic query bar. Three modes per scene: PCD (always), Mesh (gated on `mesh.ply`), Similarity (gated on lifted features zarr). Text queries are encoded via `BaseQueryableExtractor.encode_text()` in a daemon thread, with results mapped to viridis.

**Tech Stack:** Panel 1.9.2, PyVista 0.48.4, `pn.pane.VTK`, zarr, numpy, matplotlib.cm, `collab_splats.semantics.BaseQueryableExtractor`, `collab_splats.utils.visualization.{pointcloud_to_polydata,create_camera_frustum_pyvista}`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `collab_splats/dashboard/state.py` | Modify | Rename `frames` → `frames_zarr_path` |
| `collab_splats/dashboard/panes/visualize.py` | Create | `ScenePanel` + `VisualizePane` |
| `collab_splats/dashboard/app.py` | Modify | Replace `PlaceholderPane("Visualize", …)` with `VisualizePane`; wire tab-change signal |
| `tests/dashboard/test_visualize.py` | Create | Unit tests for discovery, mode scanning, feature loading, similarity |

---

## Task 1: Update AppState — `frames` → `frames_zarr_path`

**Files:**
- Modify: `collab_splats/dashboard/state.py`
- Test: `tests/dashboard/test_visualize.py`

- [ ] **Step 1: Write failing test**

```python
# tests/dashboard/test_visualize.py
import param
from collab_splats.dashboard.state import AppState


def test_appstate_has_frames_zarr_path():
    state = AppState()
    assert hasattr(state, "frames_zarr_path")
    assert state.frames_zarr_path is None


def test_appstate_no_frames_list():
    state = AppState()
    assert not hasattr(state, "frames"), "frames list removed in Phase 2"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py::test_appstate_has_frames_zarr_path tests/dashboard/test_visualize.py::test_appstate_no_frames_list -v
```

Expected: `FAILED` (frames_zarr_path not found / frames still exists).

- [ ] **Step 3: Update `collab_splats/dashboard/state.py`**

```python
from __future__ import annotations

from pathlib import Path

import param


class AppState(param.Parameterized):
    """Shared data bus passed between all dashboard panes.

    Panes observe fields via param.watch — downstream panes auto-enable
    when upstream data arrives (e.g. output_dir set by PreprocessPane).
    """

    output_dir = param.Parameter(default=None)
    video_path = param.Parameter(default=None)
    frames_zarr_path = param.Parameter(default=None)
    feedforward_result = param.Parameter(default=None)
    feature_maps_path = param.Parameter(default=None)
    lifted_features_path = param.Parameter(default=None)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py::test_appstate_has_frames_zarr_path tests/dashboard/test_visualize.py::test_appstate_no_frames_list -v
```

Expected: both PASS.

- [ ] **Step 5: Grep for `state.frames` usages that need updating**

```bash
grep -rn 'state\.frames\b\|state\.frames\[' collab_splats/ --include='*.py' | grep -v 'frames_zarr'
```

Fix any hits (change to `state.frames_zarr_path`). PreprocessPane is the likely writer — update its write from `self._state.frames = [...]` to `self._state.frames_zarr_path = <path>` (this may already be done in Phase 2; if so, grep returns nothing).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/state.py tests/dashboard/test_visualize.py
git commit -m "refactor(dashboard): rename AppState.frames → frames_zarr_path (Phase 2 schema)"
```

---

## Task 2: Discovery helpers + ScenePanel scaffold

**Files:**
- Create: `collab_splats/dashboard/panes/visualize.py`
- Test: `tests/dashboard/test_visualize.py`

- [ ] **Step 1: Write failing tests for discovery functions**

Add to `tests/dashboard/test_visualize.py`:

```python
import json
from pathlib import Path
import numpy as np
import zarr
from zarr.codecs import BloscCodec

from collab_splats.dashboard.panes.visualize import (
    _scan_datasets,
    _scan_backends,
    _scan_extractors,
)


def _make_dataset(tmp_path: Path, name: str, backends=("vggt_x",), extractors=()) -> Path:
    """Create a minimal fake output directory tree."""
    ds = tmp_path / name
    for backend in backends:
        be = ds / backend
        be.mkdir(parents=True)
        # fake feedforward.zarr (empty zarr store)
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
    (tmp_path / "not_a_dataset").mkdir()  # no run_config.yaml
    result = _scan_datasets(tmp_path)
    names = [p.name for p in result]
    assert "scene_01" in names
    assert "scene_02" in names
    assert "not_a_dataset" not in names


def test_scan_backends(tmp_path):
    ds = _make_dataset(tmp_path, "scene", backends=("vggt_x", "mapanything"))
    result = _scan_backends(ds / "scene")
    assert set(result) == {"vggt_x", "mapanything"}


def test_scan_extractors(tmp_path):
    ds = _make_dataset(tmp_path, "scene", backends=("vggt_x",), extractors=("talk2dino",))
    result = _scan_extractors(ds / "scene", "vggt_x")
    assert result == ["talk2dino"]


def test_scan_extractors_empty(tmp_path):
    ds = _make_dataset(tmp_path, "scene", backends=("vggt_x",))
    result = _scan_extractors(ds / "scene", "vggt_x")
    assert result == []
```

- [ ] **Step 2: Run — expect FAIL (module not found)**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "scan" -v
```

Expected: `ImportError: cannot import name '_scan_datasets'`.

- [ ] **Step 3: Create `collab_splats/dashboard/panes/visualize.py` — discovery helpers only**

```python
from __future__ import annotations

########################################################################
# Imports
########################################################################

import logging
import threading
from pathlib import Path
from typing import Any

import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import panel as pn
import param
import pyvista as pv
import zarr

from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.semantics import BaseQueryableExtractor
from collab_splats.utils.visualization import (
    create_camera_frustum_pyvista,
    pointcloud_to_polydata,
)

logger = logging.getLogger(__name__)

########################################################################
# Discovery helpers
########################################################################


def _scan_datasets(base_dir: Path) -> list[Path]:
    """Return subdirectories of base_dir that contain run_config.yaml."""
    if not base_dir or not base_dir.is_dir():
        return []
    return sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and (p / "run_config.yaml").exists()
    )


def _scan_backends(dataset_dir: Path) -> list[str]:
    """Return backend names (subdirs containing feedforward.zarr)."""
    if not dataset_dir or not dataset_dir.is_dir():
        return []
    return sorted(
        p.name for p in dataset_dir.iterdir()
        if p.is_dir() and (p / "feedforward.zarr").exists()
    )


def _scan_extractors(dataset_dir: Path, backend: str) -> list[str]:
    """Return extractor names with a features.zarr under semantics/."""
    semantics_dir = dataset_dir / backend / "semantics"
    if not semantics_dir.is_dir():
        return []
    return sorted(
        p.name for p in semantics_dir.iterdir()
        if p.is_dir() and (p / "features.zarr").exists()
    )


########################################################################
# Viridis colormap helper
########################################################################


def _apply_viridis(sims: np.ndarray) -> np.ndarray:
    """Map similarity scores (P,) to viridis RGB uint8 (P, 3)."""
    s_min, s_max = sims.min(), sims.max()
    if s_max > s_min:
        normalized = (sims - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(sims)
    rgba = cm.viridis(normalized)        # (P, 4) float [0, 1]
    return (rgba[:, :3] * 255).astype(np.uint8)
```

- [ ] **Step 4: Run discovery tests — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "scan" -v
```

Expected: all PASS.

- [ ] **Step 5: Add viridis test**

Add to `tests/dashboard/test_visualize.py`:

```python
from collab_splats.dashboard.panes.visualize import _apply_viridis


def test_apply_viridis_shape():
    sims = np.array([-1.0, 0.0, 0.5, 1.0])
    colors = _apply_viridis(sims)
    assert colors.shape == (4, 3)
    assert colors.dtype == np.uint8


def test_apply_viridis_constant():
    # All-same input → zero gradient, should not divide by zero
    sims = np.ones(5)
    colors = _apply_viridis(sims)
    assert colors.shape == (5, 3)
```

- [ ] **Step 6: Run — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "viridis" -v
```

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(visualize): discovery helpers + viridis util"
```

---

## Task 3: Lifted features loading + similarity query

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`
- Test: `tests/dashboard/test_visualize.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/dashboard/test_visualize.py`:

```python
from collab_splats.dashboard.panes.visualize import _load_lifted_features


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
    # Each row should have unit norm
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, np.ones(20), atol=1e-5)


def test_load_lifted_features_zero_norm(tmp_path):
    # Row of zeros should not produce NaN
    store_path = tmp_path / "features.zarr"
    store = zarr.open(str(store_path), mode="w")
    feats = np.zeros((5, 16), dtype=np.float32)
    feats[1] = 1.0  # one non-zero row
    store.create_array("features", data=feats)
    normed = _load_lifted_features(store_path)
    assert not np.any(np.isnan(normed))
```

- [ ] **Step 2: Run — expect FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "lifted" -v
```

Expected: `ImportError: cannot import name '_load_lifted_features'`.

- [ ] **Step 3: Add `_load_lifted_features` to `visualize.py`** (after `_apply_viridis`):

```python
def _load_lifted_features(features_zarr_path: Path) -> np.ndarray:
    """Load and L2-normalise per-point features from zarr. Returns (P, D) float32."""
    store = zarr.open(str(features_zarr_path), mode="r")
    feats = store["features"][:]                              # (P, D) float32
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, 1e-8)
```

- [ ] **Step 4: Run — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "lifted" -v
```

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(visualize): _load_lifted_features with L2 normalisation"
```

---

## Task 4: ScenePanel — constructor, widgets, PCD viewer

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`

This task adds the full `ScenePanel` class through its PCD rendering path. Tests for Panel widget construction require a display; use `pv.Plotter(off_screen=True)` in any smoke tests.

- [ ] **Step 1: Add `ScenePanel` class to `visualize.py`** — append after the helpers:

```python
########################################################################
# ScenePanel
########################################################################


class ScenePanel(param.Parameterized):
    """Self-contained per-scene 3D viewer panel.

    Manages dataset/backend selection, mode switching (PCD/Mesh/Similarity),
    and a PyVista plotter embedded via pn.pane.VTK.
    """

    # Observed by VisualizePane to toggle shared query bar
    mode = param.String(default="PCD")

    def __init__(
        self,
        scene_id: str,
        base_dir: Path,
        state: AppState,
        op_log: OperationLog,
        _off_screen: bool = False,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._scene_id = scene_id
        self._base_dir = Path(base_dir) if base_dir else Path("/workspace/outputs")
        self._state = state
        self._op_log = op_log

        # Internal data
        self._result: FeedforwardResult | None = None
        self._lifted_normed: np.ndarray | None = None
        self._extractor_cache: dict[str, Any] = {}
        self._available_modes: set[str] = set()
        self._available_extractors: list[str] = []
        self._load_thread: threading.Thread | None = None
        self._current_dataset_dir: Path | None = None
        self._current_backend: str | None = None

        # PyVista plotter — one per scene, never recreated
        self._plotter = pv.Plotter(off_screen=_off_screen)
        self._vtk_pane = pn.pane.VTK(
            self._plotter.ren_win,
            sizing_mode="stretch_both",
            min_height=500,
        )

        # Dataset / backend discovery
        datasets = _scan_datasets(self._base_dir)
        dataset_names = [p.name for p in datasets]
        self._dataset_dd = pn.widgets.Select(
            name="Dataset", options=dataset_names, width=200
        )
        self._backend_dd = pn.widgets.Select(
            name="Backend", options=[], width=150
        )
        self._load_btn = pn.widgets.Button(
            name="Load", button_type="primary", width=80
        )

        # Mode buttons
        self._pcd_btn = pn.widgets.Button(
            name="PCD", button_type="primary", width=80, disabled=True
        )
        self._mesh_btn = pn.widgets.Button(
            name="Mesh", width=120, disabled=True
        )
        self._sim_btn = pn.widgets.Button(
            name="Similarity", width=140, disabled=True
        )

        # Viewer controls
        self._frustum_toggle = pn.widgets.Toggle(
            name="Show frustums", value=False, width=130
        )
        self._point_size_slider = pn.widgets.IntSlider(
            name="Point size", value=2, start=1, end=10, width=180
        )
        self._reset_btn = pn.widgets.Button(name="↺ Reset camera", width=130)
        self._snapshot_btn = pn.widgets.Button(name="📷 Snapshot", width=100)
        self._status_html = pn.pane.HTML("", width=400)

        # Wire callbacks
        self._dataset_dd.param.watch(self._on_dataset_change, "value")
        self._backend_dd.param.watch(self._on_backend_change, "value")
        self._load_btn.on_click(self._on_load)
        self._pcd_btn.on_click(lambda e: self._on_mode_change("PCD"))
        self._mesh_btn.on_click(lambda e: self._on_mode_change("Mesh"))
        self._sim_btn.on_click(lambda e: self._on_mode_change("Similarity"))
        self._frustum_toggle.param.watch(self._on_frustum_toggle, "value")
        self._point_size_slider.param.watch(self._on_point_size_change, "value")
        self._reset_btn.on_click(lambda e: self._plotter.reset_camera() or self._vtk_pane.synchronize())
        self._snapshot_btn.on_click(self._on_snapshot)

        # Scene A: watch AppState for auto-suggest and rescan
        if scene_id == "A":
            state.param.watch(self._on_feedforward_result, "feedforward_result")
            state.param.watch(self._on_lifted_features_path, "lifted_features_path")

        # Populate backend dropdown for initial dataset selection
        if dataset_names:
            self._on_dataset_change(None)

    ####################################################################
    # Discovery
    ####################################################################

    def _on_dataset_change(self, event: Any) -> None:
        """Repopulate backend dropdown when dataset changes."""
        name = self._dataset_dd.value
        if not name:
            self._backend_dd.options = []
            return
        ds_dir = self._base_dir / name
        backends = _scan_backends(ds_dir)
        self._backend_dd.options = backends
        if backends:
            self._backend_dd.value = backends[0]

    def _on_backend_change(self, event: Any) -> None:
        """Clear loaded result when backend changes."""
        self._result = None
        self._lifted_normed = None
        self._available_modes = set()
        self._update_mode_buttons()

    def _scan_available_modes(self) -> None:
        """Check disk for available modes; update button states."""
        ds_name = self._dataset_dd.value
        backend = self._backend_dd.value
        if not ds_name or not backend:
            self._available_modes = set()
            self._available_extractors = []
            self._update_mode_buttons()
            return

        ds_dir = self._base_dir / ds_name
        modes: set[str] = set()

        if self._result is not None:
            modes.add("PCD")

        mesh_path = ds_dir / backend / "mesh" / "mesh.ply"
        if mesh_path.exists():
            modes.add("Mesh")

        extractors = _scan_extractors(ds_dir, backend)
        if extractors:
            modes.add("Similarity")
        self._available_extractors = extractors

        self._available_modes = modes
        self._current_dataset_dir = ds_dir
        self._current_backend = backend
        self._update_mode_buttons()

    def _update_mode_buttons(self) -> None:
        """Enable/disable mode buttons based on available modes."""
        has_pcd = "PCD" in self._available_modes
        has_mesh = "Mesh" in self._available_modes
        has_sim = "Similarity" in self._available_modes

        self._pcd_btn.disabled = not has_pcd
        self._mesh_btn.disabled = not has_mesh
        self._mesh_btn.name = "Mesh" if has_mesh else "Mesh (no mesh.ply)"
        self._sim_btn.disabled = not has_sim
        self._sim_btn.name = "Similarity" if has_sim else "Similarity (no features)"

        # Fall back to first available mode if current mode unavailable
        if self.mode not in self._available_modes and self._available_modes:
            first_available = next(iter(sorted(self._available_modes)))
            self._on_mode_change(first_available)

    ####################################################################
    # AppState watchers (Scene A only)
    ####################################################################

    def _on_feedforward_result(self, event: Any) -> None:
        """Auto-suggest dataset/backend from AppState (no auto-load)."""
        result = event.new
        if result is None or self._state.output_dir is None:
            return
        output_dir = Path(self._state.output_dir)
        # Pre-fill dataset dropdown if this dataset exists in the list
        ds_name = output_dir.name
        if ds_name in (self._dataset_dd.options or []):
            self._dataset_dd.value = ds_name

    def _on_lifted_features_path(self, event: Any) -> None:
        """Re-scan available modes when semantics pipeline writes new features."""
        self._lifted_normed = None  # Invalidate cached features
        self._scan_available_modes()

    def rescan(self) -> None:
        """Public: re-scan available modes (called on tab activation)."""
        self._scan_available_modes()

    ####################################################################
    # Load
    ####################################################################

    def _on_load(self, event: Any) -> None:
        """Start background load thread."""
        if self._load_thread and self._load_thread.is_alive():
            return
        self._load_btn.disabled = True
        self._status_html.object = "<em>Loading…</em>"
        self._load_thread = threading.Thread(target=self._do_load, daemon=True)
        self._load_thread.start()

    def _do_load(self) -> None:
        """Background: load FeedforwardResult from zarr."""
        try:
            ds_name = self._dataset_dd.value
            backend = self._backend_dd.value
            if not ds_name or not backend:
                self._set_status("No dataset/backend selected.")
                return

            zarr_path = self._base_dir / ds_name / backend / "feedforward.zarr"
            self._result = FeedforwardResult.load_zarr(zarr_path)
            self._lifted_normed = None  # Invalidate on new load
            self._scan_available_modes()

            # Reset camera and render PCD
            self._plotter.reset_camera()
            if "PCD" in self._available_modes:
                self._on_mode_change("PCD")

            n_pts = len(self._result.points)
            if n_pts > 500_000:
                self._op_log.log(
                    f"Scene {self._scene_id}: large PCD ({n_pts:,} points) — may be slow to render"
                )
            self._set_status(f"Loaded {n_pts:,} points.")
        except Exception as exc:
            logger.exception("ScenePanel load failed")
            self._set_status(f"Load failed: {exc}")
        finally:
            self._load_btn.disabled = False

    def _set_status(self, msg: str) -> None:
        self._status_html.object = f"<small>{msg}</small>"

    ####################################################################
    # Mode switching
    ####################################################################

    def _on_mode_change(self, new_mode: str) -> None:
        """Switch viewer mode; update button highlight."""
        if new_mode not in self._available_modes and self._result is not None:
            return
        self.mode = new_mode  # triggers VisualizePane.param.watch

        # Update button styles
        for btn, name in (
            (self._pcd_btn, "PCD"),
            (self._mesh_btn, "Mesh"),
            (self._sim_btn, "Similarity"),
        ):
            btn.button_type = "primary" if name == new_mode else "default"

        self._plotter.clear()
        if new_mode == "PCD":
            self._rebuild_pcd_viewer()
        elif new_mode == "Mesh":
            self._rebuild_mesh_viewer()
        elif new_mode == "Similarity":
            if self._lifted_normed is None:
                self._load_lifted_features_for_current_extractor()
            self._rebuild_sim_viewer(colors=None)

        self._vtk_pane.synchronize()

    ####################################################################
    # PCD viewer
    ####################################################################

    def _rebuild_pcd_viewer(self) -> None:
        """Render points + optional frustums."""
        if self._result is None:
            return
        point_size = self._point_size_slider.value
        cloud = pointcloud_to_polydata(self._result.points, RGB=self._result.colors)
        self._plotter.add_mesh(
            cloud, scalars="RGB", rgb=True, point_size=point_size, render_points_as_spheres=False
        )
        if self._frustum_toggle.value:
            self._add_frustums()

    def _add_frustums(self) -> None:
        """Add camera frustum actors for all extrinsics."""
        if self._result is None:
            return
        for ext in self._result.extrinsics:
            frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
            self._plotter.add_mesh(frustum, color="cornflowerblue", line_width=1)

    def _on_frustum_toggle(self, event: Any) -> None:
        if self.mode != "PCD" or self._result is None:
            return
        self._plotter.clear()
        self._rebuild_pcd_viewer()
        self._vtk_pane.synchronize()

    def _on_point_size_change(self, event: Any) -> None:
        if self.mode != "PCD" or self._result is None:
            return
        self._plotter.clear()
        self._rebuild_pcd_viewer()
        self._vtk_pane.synchronize()

    ####################################################################
    # Mesh viewer
    ####################################################################

    def _rebuild_mesh_viewer(self) -> None:
        """Load and render mesh.ply."""
        if self._current_dataset_dir is None or self._current_backend is None:
            return
        mesh_path = self._current_dataset_dir / self._current_backend / "mesh" / "mesh.ply"
        if not mesh_path.exists():
            self._set_status("mesh.ply not found.")
            return
        mesh = pv.read(str(mesh_path))
        self._plotter.add_mesh(mesh, rgb=True)

    ####################################################################
    # Similarity viewer
    ####################################################################

    def _current_extractor_name(self) -> str | None:
        """Return currently selected extractor name — set externally by VisualizePane."""
        return getattr(self, "_selected_extractor", None) or (
            self._available_extractors[0] if self._available_extractors else None
        )

    def _load_lifted_features_for_current_extractor(self) -> None:
        """Load and L2-normalise lifted features for the current extractor."""
        name = self._current_extractor_name()
        if not name or self._current_dataset_dir is None or self._current_backend is None:
            return
        feat_path = (
            self._current_dataset_dir / self._current_backend / "semantics" / name / "features.zarr"
        )
        if not feat_path.exists():
            self._set_status(f"features.zarr not found for {name}")
            return
        try:
            self._lifted_normed = _load_lifted_features(feat_path)
        except Exception as exc:
            self._set_status(f"Feature load failed: {exc}")

    def _rebuild_sim_viewer(self, colors: np.ndarray | None) -> None:
        """Render PCD with viridis similarity colours (or original RGB if no query yet)."""
        if self._result is None:
            return
        if colors is None:
            # Pre-query: show original colours with status prompt
            rgb = self._result.colors
            self._set_status("Enter a query to colour by similarity.")
        else:
            rgb = colors
        cloud = pointcloud_to_polydata(self._result.points, RGB=rgb)
        self._plotter.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2)

    def do_query(self, text: str, extractor_name: str) -> None:
        """Run similarity query (called from VisualizePane in background thread)."""
        self._selected_extractor = extractor_name

        # Load features if extractor changed or not yet loaded
        if self._lifted_normed is None:
            self._load_lifted_features_for_current_extractor()
        if self._lifted_normed is None:
            return

        # Load / cache extractor
        if extractor_name not in self._extractor_cache:
            try:
                extractor_cls = BaseQueryableExtractor.create(extractor_name)
                self._extractor_cache[extractor_name] = extractor_cls()
            except Exception as exc:
                self._set_status(f"Extractor load failed: {extractor_name} — {exc}")
                return

        extractor = self._extractor_cache[extractor_name]
        try:
            query_tensor = extractor.encode_text([text])         # (1, D) torch.Tensor
            query_vec = query_tensor[0].detach().cpu().numpy()   # (D,)
            # Normalise query vector
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 1e-8:
                query_vec = query_vec / q_norm

            sims = self._lifted_normed @ query_vec               # (P,) cosine similarity
            colors = _apply_viridis(sims)                        # (P, 3) uint8
        except Exception as exc:
            self._set_status(f"Query failed: {exc}")
            return

        self._plotter.clear()
        self._rebuild_sim_viewer(colors=colors)
        self._vtk_pane.synchronize()
        self._set_status(f'Query: "{text}" via {extractor_name}')

    ####################################################################
    # Snapshot
    ####################################################################

    def _on_snapshot(self, event: Any) -> None:
        path = Path(f"scene_{self._scene_id}_snapshot.png")
        self._plotter.screenshot(str(path))
        self._set_status(f"Saved {path.name}")

    ####################################################################
    # Layout
    ####################################################################

    def panel(self) -> pn.Column:
        """Return the full scene panel layout."""
        controls_row = pn.Row(
            self._dataset_dd, self._backend_dd, self._load_btn
        )
        mode_row = pn.Row(
            self._pcd_btn, self._mesh_btn, self._sim_btn,
            self._frustum_toggle, self._point_size_slider
        )
        action_row = pn.Row(self._reset_btn, self._snapshot_btn)
        return pn.Column(
            f"### Scene {self._scene_id}",
            controls_row,
            mode_row,
            self._vtk_pane,
            action_row,
            self._status_html,
            sizing_mode="stretch_both",
        )
```

- [ ] **Step 2: Write smoke test for ScenePanel construction (off-screen)**

Add to `tests/dashboard/test_visualize.py`:

```python
import pytest
import panel as pn
from collab_splats.dashboard.panes.visualize import ScenePanel
from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.operation_log import OperationLog

pn.extension("vtk")


def _make_op_log():
    return OperationLog()


def test_scene_panel_constructs(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    state = AppState()
    op_log = _make_op_log()
    panel = ScenePanel(
        scene_id="A",
        base_dir=tmp_path,
        state=state,
        op_log=op_log,
        _off_screen=True,
    )
    assert panel._scene_id == "A"
    assert "scene_01" in panel._dataset_dd.options


def test_scene_panel_scan_available_modes_no_result(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    state = AppState()
    panel = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)
    # No result loaded → no modes available
    panel._scan_available_modes()
    assert "PCD" not in panel._available_modes


def test_scene_panel_scan_available_modes_with_mesh(tmp_path):
    ds = _make_dataset(tmp_path, "scene_01", backends=("vggt_x",))
    # Add mesh.ply
    mesh_dir = tmp_path / "scene_01" / "vggt_x" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh.ply").write_bytes(b"ply\n")
    state = AppState()
    panel = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)
    panel._dataset_dd.value = "scene_01"
    panel._backend_dd.value = "vggt_x"
    # Inject a fake result so PCD is available
    panel._result = object()  # non-None sentinel
    panel._scan_available_modes()
    assert "Mesh" in panel._available_modes


def test_scene_panel_scan_available_modes_with_features(tmp_path):
    _make_dataset(tmp_path, "scene_01", backends=("vggt_x",), extractors=("talk2dino",))
    state = AppState()
    panel = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)
    panel._dataset_dd.value = "scene_01"
    panel._backend_dd.value = "vggt_x"
    panel._result = object()
    panel._scan_available_modes()
    assert "Similarity" in panel._available_modes
    assert "talk2dino" in panel._available_extractors
```

- [ ] **Step 3: Run tests — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "scene_panel" -v
```

If `OperationLog` import fails, check its actual module path — it may be at `collab_splats/dashboard/operation_log.py` or similar. Adjust import accordingly.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(visualize): ScenePanel — scaffold, discovery, PCD/Mesh/Similarity viewers"
```

---

## Task 5: VisualizePane + shared query bar

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`

- [ ] **Step 1: Append `VisualizePane` class to `visualize.py`**

```python
########################################################################
# VisualizePane
########################################################################


class VisualizePane(param.Parameterized):
    """Compositor: two ScenePanels + shared semantic query bar."""

    def __init__(
        self,
        state: AppState,
        op_log: OperationLog,
        base_dir: Path,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._state = state
        self._op_log = op_log

        self._scene_a = ScenePanel("A", base_dir, state, op_log)
        self._scene_b = ScenePanel("B", base_dir, state, op_log)

        # Shared query bar
        self._query_input = pn.widgets.TextInput(
            placeholder="Enter text query…", width=300
        )
        self._a_extractor_dd = pn.widgets.Select(
            name="Scene A extractor", options=[], width=160
        )
        self._b_extractor_dd = pn.widgets.Select(
            name="Scene B extractor", options=[], width=160
        )
        self._query_btn = pn.widgets.Button(
            name="Query →", button_type="success", width=90
        )
        self._query_bar = pn.Row(
            self._query_input,
            self._a_extractor_dd,
            self._b_extractor_dd,
            self._query_btn,
            visible=False,
        )

        # Wire mode change watches on both scenes
        self._scene_a.param.watch(self._on_scene_mode_change, "mode")
        self._scene_b.param.watch(self._on_scene_mode_change, "mode")

        # Wire extractor dropdown options to scene extractor lists
        self._scene_a.param.watch(self._update_extractor_dropdowns, "mode")
        self._scene_b.param.watch(self._update_extractor_dropdowns, "mode")

        self._query_input.param.watch(self._on_query_enter, "value_input")
        self._query_btn.on_click(self._on_query_click)

    ####################################################################
    # Query bar visibility
    ####################################################################

    def _on_scene_mode_change(self, event: Any) -> None:
        """Show query bar when ≥1 scene is in Similarity mode."""
        a_sim = self._scene_a.mode == "Similarity"
        b_sim = self._scene_b.mode == "Similarity"
        self._query_bar.visible = a_sim or b_sim
        self._a_extractor_dd.visible = a_sim
        self._b_extractor_dd.visible = b_sim
        self._update_extractor_dropdowns(None)

    def _update_extractor_dropdowns(self, event: Any) -> None:
        self._a_extractor_dd.options = self._scene_a._available_extractors or []
        self._b_extractor_dd.options = self._scene_b._available_extractors or []

    ####################################################################
    # Query dispatch
    ####################################################################

    def _on_query_enter(self, event: Any) -> None:
        """Submit on Enter key (value_input fires on each keystroke; filter to Enter)."""
        # TextInput doesn't have an Enter event — use button instead.
        pass

    def _on_query_click(self, event: Any) -> None:
        """Fire query on all active Similarity scenes in parallel daemon threads."""
        text = self._query_input.value
        if not text:
            return
        targets = []
        if self._scene_a.mode == "Similarity" and self._a_extractor_dd.value:
            targets.append((self._scene_a, self._a_extractor_dd.value))
        if self._scene_b.mode == "Similarity" and self._b_extractor_dd.value:
            targets.append((self._scene_b, self._b_extractor_dd.value))
        for scene, extractor in targets:
            t = threading.Thread(
                target=scene.do_query, args=(text, extractor), daemon=True
            )
            t.start()

    ####################################################################
    # Tab activation rescan
    ####################################################################

    def _rescan_both_scenes(self) -> None:
        """Re-scan available modes on both scenes (called on tab activation)."""
        self._scene_a.rescan()
        self._scene_b.rescan()
        self._update_extractor_dropdowns(None)

    def wire_tabs(self, tabs: pn.Tabs, tab_index: int) -> None:
        """Connect tab activation signal so modes rescan when this tab becomes active."""
        def _on_tab_change(event: Any) -> None:
            if event.new == tab_index:
                self._rescan_both_scenes()
        tabs.param.watch(_on_tab_change, "active")

    ####################################################################
    # Layout
    ####################################################################

    def panel(self) -> pn.Column:
        """Return the full VisualizePane layout."""
        scenes_row = pn.Row(
            self._scene_a.panel(),
            self._scene_b.panel(),
            sizing_mode="stretch_both",
        )
        return pn.Column(
            self._query_bar,
            scenes_row,
            sizing_mode="stretch_both",
        )
```

- [ ] **Step 2: Write smoke test for VisualizePane**

Add to `tests/dashboard/test_visualize.py`:

```python
from collab_splats.dashboard.panes.visualize import VisualizePane


def test_visualize_pane_constructs(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    state = AppState()
    op_log = _make_op_log()
    vp = VisualizePane.__new__(VisualizePane)
    # Patch ScenePanel to use off_screen
    import unittest.mock as mock
    with mock.patch(
        "collab_splats.dashboard.panes.visualize.ScenePanel",
        lambda *a, **kw: ScenePanel(*a, **{**kw, "_off_screen": True}),
    ):
        vp.__init__(state=state, op_log=op_log, base_dir=tmp_path)
    assert vp._scene_a._scene_id == "A"
    assert vp._scene_b._scene_id == "B"
    assert vp._query_bar.visible is False
```

- [ ] **Step 3: Run — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py::test_visualize_pane_constructs -v
```

- [ ] **Step 4: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(visualize): VisualizePane compositor + shared query bar"
```

---

## Task 6: Wire into `app.py`

**Files:**
- Modify: `collab_splats/dashboard/app.py`

- [ ] **Step 1: Read current `app.py` import block and Tabs wiring section**

Read lines around imports (top ~15 lines) and around the Tabs construction (~lines 125–145).

- [ ] **Step 2: Add import**

Find the existing `PlaceholderPane` import line and add `VisualizePane` alongside it:

```python
from collab_splats.dashboard.panes.visualize import VisualizePane
```

- [ ] **Step 3: Replace PlaceholderPane for Visualize**

Find:
```python
            "Visualize": PlaceholderPane("Visualize", "Coming in Phase 4 — interactive PyVista 3D comparison"),
```

Replace with:
```python
            "Visualize": VisualizePane(state=self._state, op_log=self._op_log, base_dir=self._base_dir),
```

If `self._base_dir` doesn't exist yet in `DashboardApp`, add it. Check where `app.py` stores the `base_dir` path — it's passed as a CLI argument. Trace the constructor and add `self._base_dir = Path(base_dir)` if missing.

- [ ] **Step 4: Wire tab-change signal**

After `tabs = pn.Tabs(...)` is constructed (around line 130), add:

```python
# Wire tab activation → VisualizePane mode rescan
visualize_tab_index = list(self._panes.keys()).index("Visualize")
self._panes["Visualize"].wire_tabs(tabs, visualize_tab_index)
```

- [ ] **Step 5: Add `pn.extension("vtk")` if not already called**

Find the `pn.extension(...)` call in `app.py` and add `"vtk"` to the list if absent:

```python
pn.extension("vtk", ...)
```

- [ ] **Step 6: Run dashboard smoke test**

```bash
timeout 10 /opt/conda/envs/reconstruction/bin/python -c "
import panel as pn
pn.extension('vtk')
from pathlib import Path
from collab_splats.dashboard.app import DashboardApp
app = DashboardApp(base_dir='/workspace/outputs', port=7999)
print('DashboardApp constructed OK')
" && echo "PASS"
```

Expected: `DashboardApp constructed OK` + `PASS` (or timeout after 10s if it tries to serve).

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/app.py
git commit -m "feat(dashboard): wire VisualizePane into Tab 4"
```

---

## Task 7: Full test run + final commit

- [ ] **Step 1: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/ -v
```

Fix any failures before proceeding.

- [ ] **Step 2: Run broader suite to check for regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q --ignore=tests/integration
```

- [ ] **Step 3: Final commit**

```bash
git add -p  # review any unstaged changes
git commit -m "test(visualize): full test coverage for VisualizePane"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec requirement | Task |
|-----------------|------|
| ScenePanel + VisualizePane architecture | Task 4, 5 |
| Dataset/backend discovery from base_dir | Task 2 |
| Explicit Load required (no auto-load) | Task 4 (`_do_load`) |
| PCD mode: points + colors + frustum toggle + point size | Task 4 (`_rebuild_pcd_viewer`) |
| Mesh mode: gated on mesh.ply existence | Task 4 (`_rebuild_mesh_viewer`, `_scan_available_modes`) |
| Similarity mode: gated on features.zarr | Task 4 (`_scan_available_modes`) |
| Similarity query: BaseQueryableExtractor.encode_text() | Task 4 (`do_query`) |
| Viridis colormap, not configurable | Task 3 (`_apply_viridis`) |
| Pre-normalised features at load time | Task 3 (`_load_lifted_features`) |
| Camera preserved across mode switches, reset on Load | Task 4 |
| pn.pane.VTK synchronize() after actor changes | Task 4 (all `_rebuild_*` callers) |
| Dynamic rescan: tab activation + lifted_features_path watch | Task 4 (`_on_lifted_features_path`), Task 5 (`wire_tabs`) |
| Shared query bar: visible ≥1 scene in Similarity | Task 5 (`_on_scene_mode_change`) |
| Per-scene extractor dropdown in shared bar | Task 5 (`_a_extractor_dd`, `_b_extractor_dd`) |
| Parallel query threads | Task 5 (`_on_query_click`) |
| AppState.frames → frames_zarr_path | Task 1 |
| Wire into app.py | Task 6 |
| Error badges for all failure modes | Task 4 (`_set_status`) |
| Large PCD warning via op_log | Task 4 (`_do_load`) |

### Type Consistency

- `_load_lifted_features(path: Path) -> np.ndarray` — used in Task 3 and Task 4 ✓
- `_apply_viridis(sims: np.ndarray) -> np.ndarray` — used in Task 3 and Task 4 ✓
- `ScenePanel.do_query(text: str, extractor_name: str)` — called in Task 5 ✓
- `VisualizePane.wire_tabs(tabs, tab_index)` — called in Task 6 ✓
- `BaseQueryableExtractor.create(name)` returns class → `()` to instantiate ✓
- `extractor.encode_text([text])` → tensor `[0].detach().cpu().numpy()` ✓
