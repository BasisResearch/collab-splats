# Mesh Viewer Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inline mesh options row + Run Mesh button to `ScenePanel`; auto-display `mesh.ply` on tab activate without requiring Load.

**Architecture:** Single file change (`collab_splats/dashboard/panes/visualize.py`). New `_mesh_options_row` follows the existing `_points_options_row` / `_sim_query_row` pattern — hidden by default, shown when Mesh mode is active. `_scan_available_modes` gains auto-display and zarr-gate logic. New `_run_mesh_worker` runs in a background thread (same pattern as `ReconstructPane`).

**Tech Stack:** Panel (`pn.Row`, `pn.widgets.FloatInput`, `pn.widgets.Checkbox`, `pn.widgets.Button`), PyVista, `collab_splats.mesh.utils.pointcloud_to_mesh`, `FeedforwardResult.load_zarr`, `threading`, `pn.io.state.execute`

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/dashboard/panes/visualize.py` | Add `_mesh_options_row` widgets, update `_on_mode_change`, update `_scan_available_modes`, add `_auto_display_mesh`, add `_on_run_mesh` + `_run_mesh_worker`, update `panel()` layout |
| `tests/dashboard/test_visualize.py` | Add 6 new tests; patch `_rebuild_mesh_viewer` in existing mesh-scan test |

---

### Task 1: Write failing tests for mesh options row widgets

**Files:**
- Modify: `tests/dashboard/test_visualize.py`

The test file already imports `ScenePanel`, `pn`, `mock`, `AppState`, `OperationLog` and has the `_make_scene` / `_make_op_log` / `_make_dataset` helpers. Add these tests at the end of the "ScenePanel layout / new widget tests" section.

- [ ] **Step 1: Add 3 widget tests**

In `tests/dashboard/test_visualize.py`, after `test_scene_panel_wire_tabs`, add:

```python
def test_scene_panel_has_mesh_options_row(tmp_path):
    sp = _make_scene(tmp_path)
    assert hasattr(sp, "_mesh_options_row")
    assert sp._mesh_options_row.visible is False


def test_scene_panel_mesh_options_row_visible_in_mesh_mode(tmp_path):
    sp = _make_scene(tmp_path)
    sp._available_modes = {"Mesh"}
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
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "mesh_options_row" -v 2>&1 | tail -20
```

Expected: `AttributeError: 'ScenePanel' object has no attribute '_mesh_options_row'`

---

### Task 2: Add `_mesh_options_row` widgets, update `_on_mode_change`, update `panel()`

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`

- [ ] **Step 1: Add mesh widgets after `_sim_query_row` block**

In `visualize.py`, find the block ending with `self._sim_query_row = pn.Row(...)`. Directly after the closing `)` of that Row (and before `self._reset_btn = ...`), insert:

```python
        # Mesh options row — TSDF params + Run Mesh button
        self._mesh_voxel_input = pn.widgets.FloatInput(
            name="voxel_size", value=0.01, step=0.005, start=0.001, end=1.0, width=90
        )
        self._mesh_sdf_input = pn.widgets.FloatInput(
            name="sdf_trunc", value=0.04, step=0.01, start=0.001, end=5.0, width=90
        )
        self._mesh_depth_input = pn.widgets.FloatInput(
            name="depth_trunc", value=10.0, step=1.0, start=0.1, end=200.0, width=90
        )
        self._mesh_clean_check = pn.widgets.Checkbox(name="clean", value=True)
        self._mesh_run_btn = pn.widgets.Button(
            name="Run Mesh", button_type="primary", disabled=True, width=100
        )
        self._mesh_options_row = pn.Row(
            self._mesh_voxel_input,
            self._mesh_sdf_input,
            self._mesh_depth_input,
            self._mesh_clean_check,
            self._mesh_run_btn,
            visible=False,
        )
        self._mesh_thread: threading.Thread | None = None
```

- [ ] **Step 2: Wire `_mesh_run_btn` callback**

In the "Wire callbacks" section, after `self._sim_query_btn.on_click(self._on_sim_query_click)`, add:

```python
        self._mesh_run_btn.on_click(self._on_run_mesh)
```

- [ ] **Step 3: Update `_on_mode_change` to show/hide `_mesh_options_row`**

In `_on_mode_change`, replace:

```python
        # Show/hide contextual rows based on active mode
        self._points_options_row.visible = (new_mode == "PCD")
        self._sim_query_row.visible = (new_mode == "Similarity")
```

With:

```python
        # Show/hide contextual rows based on active mode
        self._points_options_row.visible = (new_mode == "PCD")
        self._mesh_options_row.visible = (new_mode == "Mesh")
        self._sim_query_row.visible = (new_mode == "Similarity")
```

- [ ] **Step 4: Update `panel()` to include `_mesh_options_row` between mode selector and vtk_pane**

In `panel()`, replace:

```python
        return pn.Column(
            "### Scene",
            controls_row,
            extractor_row,
            self._mode_selector,
            self._vtk_pane,
            self._points_options_row,
            self._sim_query_row,
            action_row,
            self._status_html,
            sizing_mode="stretch_both",
        )
```

With:

```python
        return pn.Column(
            "### Scene",
            controls_row,
            extractor_row,
            self._mode_selector,
            self._mesh_options_row,
            self._points_options_row,
            self._sim_query_row,
            self._vtk_pane,
            action_row,
            self._status_html,
            sizing_mode="stretch_both",
        )
```

- [ ] **Step 5: Run Task 1 tests — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "mesh_options_row" -v 2>&1 | tail -20
```

Expected: 3 tests PASS.

- [ ] **Step 6: Run full test suite — no regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -v 2>&1 | tail -30
```

Expected: all existing tests PASS.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(dashboard): add _mesh_options_row widgets with TSDF params + Run Mesh button"
```

---

### Task 3: Write failing tests for scan behavior and auto-display

**Files:**
- Modify: `tests/dashboard/test_visualize.py`

- [ ] **Step 1: Patch existing mesh-scan test that will break**

The existing `test_scene_panel_scan_available_modes_with_mesh` calls `_scan_available_modes()` which (after our changes in Task 4) will call `_auto_display_mesh` → `_rebuild_mesh_viewer` → `pv.read("mesh.ply")` on a 4-byte fake file. Patch it now before Task 4 breaks it.

Find `test_scene_panel_scan_available_modes_with_mesh` in `tests/dashboard/test_visualize.py`. Replace its body:

```python
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
```

- [ ] **Step 2: Add 3 new scan/auto-display tests**

After `test_scene_panel_scan_available_modes_with_mesh`, add:

```python
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
```

- [ ] **Step 3: Run new tests — expect FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "auto_displays_mesh or enables_run_btn or disabled_without_zarr" -v 2>&1 | tail -20
```

Expected: 3 FAIL (logic not yet added to `_scan_available_modes`).

---

### Task 4: Update `_scan_available_modes`, add `_auto_display_mesh`

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`

- [ ] **Step 1: Update `_scan_available_modes`**

Find `_scan_available_modes`. Replace its entire body:

```python
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

        # PCD mode requires a loaded result
        if self._result is not None:
            modes.add("PCD")

        # Mesh mode requires mesh.ply on disk; auto-display it when found
        mesh_path = ds_dir / backend / "mesh" / "mesh.ply"
        if mesh_path.exists():
            modes.add("Mesh")
            self._auto_display_mesh()

        # Enable Run Mesh if feedforward.zarr exists (zarr gate for generation)
        zarr_path = ds_dir / backend / "feedforward.zarr"
        self._mesh_run_btn.disabled = not zarr_path.exists()
        if not zarr_path.exists() and mesh_path.exists():
            self._set_status("No feedforward.zarr — cannot regenerate mesh.")

        # Similarity mode requires at least one extractor with features.zarr
        extractors = _scan_extractors(ds_dir, backend)
        if extractors:
            modes.add("Similarity")
        self._available_extractors = extractors

        self._available_modes = modes
        self._current_dataset_dir = ds_dir
        self._current_backend = backend
        self._update_mode_buttons()
```

- [ ] **Step 2: Add `_auto_display_mesh` method**

In the Mesh viewer section (after `_rebuild_mesh_viewer`), add:

```python
    def _auto_display_mesh(self) -> None:
        """Render mesh.ply immediately — called from scan, no Load required."""
        self._plotter.clear()
        self._rebuild_mesh_viewer()
        self._vtk_pane.synchronize()
```

- [ ] **Step 3: Run Task 3 tests — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "auto_displays_mesh or enables_run_btn or disabled_without_zarr" -v 2>&1 | tail -20
```

Expected: 3 PASS.

- [ ] **Step 4: Run full dashboard test suite — no regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -v 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(dashboard): auto-display mesh.ply on scan; gate Run Mesh on feedforward.zarr"
```

---

### Task 5: Write failing tests for `_on_run_mesh`

**Files:**
- Modify: `tests/dashboard/test_visualize.py`

- [ ] **Step 1: Add 2 run-mesh tests**

Add after the Task 3 tests:

```python
def test_on_run_mesh_uses_existing_result(tmp_path):
    """_on_run_mesh uses _result if already loaded rather than loading zarr."""
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
         mock.patch("pn.io.state.execute") as mock_execute:
        sp._run_mesh_worker()

    # pointcloud_to_mesh called with the pre-loaded _result, not load_zarr
    mock_ptm.assert_called_once()
    call_kwargs = mock_ptm.call_args
    assert call_kwargs[0][0] is sp._result  # first positional arg = result


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
         mock.patch("pn.io.state.execute"):
        sp._run_mesh_worker()

    mock_load.assert_called_once()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "on_run_mesh" -v 2>&1 | tail -20
```

Expected: 2 FAIL (`AttributeError: _run_mesh_worker` not defined).

---

### Task 6: Implement `_on_run_mesh` and `_run_mesh_worker`

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`

- [ ] **Step 1: Add `_on_run_mesh` and `_run_mesh_worker` methods**

In the Mesh viewer section (after `_auto_display_mesh`), add:

```python
    def _on_run_mesh(self, event: Any) -> None:
        """Spawn background mesh generation thread on Run Mesh click."""
        if self._mesh_thread and self._mesh_thread.is_alive():
            return
        self._mesh_run_btn.disabled = True
        self._set_status("Running mesh generation…")
        self._mesh_thread = threading.Thread(target=self._run_mesh_worker, daemon=True)
        self._mesh_thread.start()

    def _run_mesh_worker(self) -> None:
        """Background: load zarr if needed → pointcloud_to_mesh → refresh viewer."""
        try:
            if self._result is None:
                from collab_splats.pointcloud.feedforward.base import FeedforwardResult
                zarr_path = (
                    self._current_dataset_dir / self._current_backend / "feedforward.zarr"
                )
                self._result = FeedforwardResult.load_zarr(zarr_path)

            mesh_dir = self._current_dataset_dir / self._current_backend / "mesh"
            from collab_splats.mesh.utils import pointcloud_to_mesh
            mesh_result = pointcloud_to_mesh(
                self._result,
                mesh_dir,
                method="open3d_tsdf",
                voxel_size=self._mesh_voxel_input.value,
                sdf_trunc=self._mesh_sdf_input.value,
                depth_trunc=self._mesh_depth_input.value,
                clean_repair=self._mesh_clean_check.value,
            )
            msg = f"Mesh done — {mesh_result.mesh_path.name}"
            pn.io.state.execute(lambda: self._refresh_after_mesh(msg))
        except Exception as exc:
            logger.exception("Mesh generation failed")
            err = str(exc)
            pn.io.state.execute(lambda: self._set_status(f"Mesh failed: {err}"))
        finally:
            pn.io.state.execute(lambda: setattr(self._mesh_run_btn, "disabled", False))

    def _refresh_after_mesh(self, status_msg: str) -> None:
        """IOLoop-thread: redraw mesh viewer after successful generation."""
        self._plotter.clear()
        self._rebuild_mesh_viewer()
        self._vtk_pane.synchronize()
        self._set_status(status_msg)
```

- [ ] **Step 2: Run Task 5 tests — expect PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -k "on_run_mesh" -v 2>&1 | tail -20
```

Expected: 2 PASS.

- [ ] **Step 3: Run full dashboard test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -v 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 4: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(dashboard): add _on_run_mesh background worker — auto-loads zarr, reruns TSDF"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec requirement | Task |
|-----------------|------|
| `_mesh_options_row` hidden by default | Task 1 test + Task 2 widget |
| Row visible when Mesh mode active | Task 1 test + Task 2 `_on_mode_change` |
| voxel_size, sdf_trunc, depth_trunc, clean_repair widgets | Task 2 |
| Run Mesh button wired | Task 2 wire callbacks |
| Auto-display mesh.ply on scan (no Load required) | Task 3 test + Task 4 |
| Run btn enabled when zarr exists | Task 3 test + Task 4 |
| Run btn disabled when zarr absent | Task 3 test + Task 4 |
| Auto-load zarr if `_result` is None | Task 5 test + Task 6 |
| Uses existing `_result` if loaded | Task 5 test + Task 6 |
| Background thread (non-blocking) | Task 6 `_on_run_mesh` |
| Viewer refreshes after generation | Task 6 `_refresh_after_mesh` |
| Error status on failure | Task 6 `except` block |
| Run btn re-enabled in finally | Task 6 `finally` block |
| `panel()` layout: options between mode selector + vtk_pane | Task 2 |

### Type / Name Consistency

- `_mesh_run_btn` — consistent across Task 2 (creation), Task 3 (assertion), Task 4 (gate), Task 6 (finally)
- `_run_mesh_worker` — called from `_on_run_mesh` in Task 6, tested directly in Task 5
- `pointcloud_to_mesh` import path: `collab_splats.mesh.utils` — matches `mesh/utils.py` line 480
- `FeedforwardResult.load_zarr` — matches `pointcloud/feedforward/base.py`
