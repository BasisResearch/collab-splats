# Localize Tab Redesign + Notebook Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Localize tab to use sidebar-driven controls (matching other tabs), fix the headless VTK display bug, add per-frame progress during index build, and fix the notebook's zarr path.

**Architecture:** `AppState` gains `localize_method`/`localize_extractor` string fields; `App._build_sidebar()` grows a Localize section that writes those fields; `LocalizePane` strips its inline DDs and reads from AppState (watching for changes that invalidate the cached localizer); `CameraLocalizer.__init__` accepts an optional `progress_callback(i, total)` kwarg; `LocalizeScenePanel` is created with `_off_screen=True` in the dashboard.

**Tech Stack:** Panel (param.watch), PyVista (off_screen=True), PyTorch, `CameraLocalizer`, `AppState`.

---

### Task 1: AppState — add localize_method and localize_extractor

**Files:**
- Modify: `collab_splats/dashboard/state.py`
- Create: `tests/dashboard/test_localize.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_localize.py -v
```
Expected: `AttributeError: 'AppState' object has no attribute 'localize_method'`

- [ ] **Step 3: Add two fields to state.py**

After `semantic_extractor = param.String(default="")` add:
```python
    localize_method = param.String(default="")
    localize_extractor = param.String(default="DISK+LightGlue")
```

- [ ] **Step 4: Run to verify pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_localize.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/state.py tests/dashboard/test_localize.py
git commit -m "feat(dashboard): add localize_method, localize_extractor to AppState"
```

---

### Task 2: CameraLocalizer — progress_callback kwarg

**Files:**
- Modify: `collab_splats/pointcloud/localization.py` (lines ~532–602, the `__init__` and frame loop)
- Modify: `tests/pointcloud/test_localization.py`

- [ ] **Step 1: Write failing test**

Add to `tests/pointcloud/test_localization.py` (after the existing `test_localization_result_fields_on_success` function):
```python
def test_camera_localizer_calls_progress_callback():
    """progress_callback(i, total) called once per reference frame, 0-indexed."""
    from collab_splats.pointcloud.localization import CameraLocalizer
    import tempfile, pathlib, cv2 as _cv2

    pts3d, extrinsics, intrinsics = _make_synthetic_scene()
    extractor = _MockExtractor(pts3d, extrinsics, intrinsics)
    calls = []

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(len(extrinsics)):
            p = pathlib.Path(tmp) / f"frame_{i:04d}.png"
            _cv2.imwrite(str(p), np.zeros((480, 640, 3), dtype=np.uint8))
            paths.append(p)

        CameraLocalizer(
            pts3d, extrinsics, intrinsics, paths,
            extractor=extractor,
            progress_callback=lambda i, total: calls.append((i, total)),
        )

    total = len(extrinsics)
    assert len(calls) == total
    assert calls[0] == (0, total)
    assert calls[-1] == (total - 1, total)
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization.py::test_camera_localizer_calls_progress_callback -v
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'progress_callback'`

- [ ] **Step 3: Add import and kwarg to localization.py**

First check if `Callable` is already imported:
```bash
grep -n 'Callable' /workspace/collab-splats/collab_splats/pointcloud/localization.py | head -5
```

If missing, add to the top-of-file imports (near other `from ...` lines):
```python
from collections.abc import Callable
```

In `CameraLocalizer.__init__`, add after `config: dict | None = None,`:
```python
        progress_callback: Callable[[int, int], None] | None = None,
```

Inside the frame loop, after `self._frame_features.append(feats)` and before the `logger.debug(...)` call:
```python
            if progress_callback is not None:
                progress_callback(len(self._frame_features) - 1, len(image_paths))
```

- [ ] **Step 4: Run to verify pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization.py::test_camera_localizer_calls_progress_callback -v
```
Expected: PASS.

- [ ] **Step 5: Run full localization test suite (non-slow)**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization.py -v -k "not slow"
```
Expected: all pass (no regressions — `progress_callback` defaults to None so existing callers unaffected).

- [ ] **Step 6: Add C-note comment in CameraLocalizer.__init__**

Above the frame loop (`for path in image_paths:`), add:
```python
        # TODO(future-C): pre-compute and store these features in feedforward.zarr so index
        # build is a zarr read (~1 s) instead of O(N) GPU inference. See spec 2026-05-29.
```

- [ ] **Step 7: Run full localization test suite (non-slow)**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization.py -v -k "not slow"
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "feat(localization): add progress_callback kwarg to CameraLocalizer.__init__"
```

---

### Task 3: App sidebar — Localize section

**Files:**
- Modify: `collab_splats/dashboard/app.py`

- [ ] **Step 1: Add _scan_recon_methods to the localize import in app.py**

Find the existing import:
```python
from collab_splats.dashboard.panes.localize import LocalizePane
```
Replace with:
```python
from collab_splats.dashboard.panes.localize import LocalizePane, _scan_recon_methods
```

- [ ] **Step 2: Create localize sidebar widgets in App.__init__**

In `App.__init__`, in the block that creates sidebar MODELS widgets (before `self._sidebar = self._build_sidebar()`), add after `self._models_status`:

```python
        # Localize sidebar widgets — created before _build_sidebar
        self._localize_method_dd = pn.widgets.Select(
            name="Localize method", options=[], width=280,
        )
        self._localize_extractor_dd = pn.widgets.Select(
            name="Localize extractor",
            options=["DISK+LightGlue", "XFeat+MNN"],
            value="DISK+LightGlue",
            width=280,
        )
        self._localize_method_dd.param.watch(self._on_localize_method_changed, ["value"])
        self._localize_extractor_dd.param.watch(self._on_localize_extractor_changed, ["value"])
```

- [ ] **Step 3: Add output_dir watcher for localize methods**

After `self._sidebar = self._build_sidebar()` in `App.__init__`, add:
```python
        self._state.param.watch(self._on_output_dir_changed_localize, ["output_dir"])
```

- [ ] **Step 4: Add three handler methods to App**

Add after `_on_sidebar_frustum_toggle`:
```python
    def _on_output_dir_changed_localize(self, event: Any) -> None:
        """Rescan feedforward methods for Localize sidebar when output_dir changes."""
        output_dir = event.new
        methods = _scan_recon_methods(Path(output_dir)) if output_dir else []
        self._localize_method_dd.options = methods
        if methods:
            self._localize_method_dd.value = methods[0]
            self._state.localize_method = methods[0]
        else:
            self._state.localize_method = ""

    def _on_localize_method_changed(self, event: Any) -> None:
        self._state.localize_method = event.new or ""

    def _on_localize_extractor_changed(self, event: Any) -> None:
        self._state.localize_extractor = event.new or "DISK+LightGlue"
```

- [ ] **Step 5: Add Localize section to _build_sidebar()**

In `_build_sidebar()`, after the Models `pn.layout.Divider()` and before `pn.pane.HTML("<h3 ...>View</h3>")`, insert:
```python
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>Localize</h3>"),
            self._localize_method_dd,
            self._localize_extractor_dd,
            pn.layout.Divider(),
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/app.py
git commit -m "feat(dashboard): add Localize section to sidebar; wire method/extractor to AppState"
```

---

### Task 4: LocalizePane — remove inline DDs, read AppState, progress, off_screen fix

**Files:**
- Modify: `collab_splats/dashboard/panes/localize.py`
- Modify: `tests/dashboard/test_localize.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/dashboard/test_localize.py`:
```python
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


def test_localize_pane_no_inline_method_dd():
    """LocalizePane should not have _method_dd or _extractor_dd attributes."""
    from collab_splats.dashboard.panes.localize import LocalizePane
    state = _make_state()
    pane = LocalizePane(state=state, op_log=_make_op_log())
    assert not hasattr(pane, "_method_dd")
    assert not hasattr(pane, "_extractor_dd")


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
```

- [ ] **Step 2: Run to verify failures**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_localize.py -v -k "cache_invalidated or no_inline"
```
Expected: `test_localize_pane_cache_invalidated_*` and `test_localize_pane_no_inline_method_dd` fail.

- [ ] **Step 3: Remove _method_dd and _extractor_dd from LocalizePane.__init__**

In `LocalizePane.__init__`, delete these two widget declarations and their associated wiring:
```python
        self._method_dd = pn.widgets.Select(
            name="Recon",
            options=[],
            width=120,
        )
        self._extractor_dd = pn.widgets.Select(
            name="Extractor",
            options=["DISK+LightGlue", "XFeat+MNN"],
            value="DISK+LightGlue",
            width=140,
        )
```

- [ ] **Step 4: Replace output_dir watch and add localize_state watch**

Replace:
```python
        self._state.param.watch(self._on_output_dir_changed, ["output_dir"])
        self._on_output_dir_changed(None)
```
With:
```python
        self._state.param.watch(self._on_output_dir_changed, ["output_dir"])
        self._state.param.watch(self._on_localize_state_changed, ["localize_method", "localize_extractor"])
        self._on_output_dir_changed(None)
```

- [ ] **Step 5: Replace _on_output_dir_changed and add _on_localize_state_changed**

Replace the existing `_on_output_dir_changed` method body:
```python
    def _on_output_dir_changed(self, event: Any) -> None:
        """Re-evaluate gate when session output_dir changes; invalidate cached localizer."""
        self._localizer = None
        self._ff_result = None
        self._update_run_btn_gate()
```

Add new method after it:
```python
    def _on_localize_state_changed(self, event: Any) -> None:
        """Invalidate cached localizer when method or extractor changes."""
        self._localizer = None
        self._ff_result = None
        self._scene_panel = None
        self._update_run_btn_gate()
```

- [ ] **Step 6: Update _update_run_btn_gate to read AppState**

Replace the full method:
```python
    def _update_run_btn_gate(self) -> None:
        """Enable Run when output_dir set, localize_method set, and query path non-empty."""
        has_dir = bool(self._state.output_dir)
        has_method = bool(self._state.localize_method)
        has_query = bool(self._query_input.value and self._query_input.value.strip())
        self._run_btn.disabled = not (has_dir and has_method and has_query)
        self._batch_run_btn.disabled = not (has_dir and has_method)
```

- [ ] **Step 7: Update _build_localizer to read AppState and accept progress_callback**

Replace the full method:
```python
    def _build_localizer(
        self,
        progress_callback=None,
    ) -> tuple["FeedforwardResult", "CameraLocalizer"]:
        """Load feedforward.zarr and build CameraLocalizer with optional frame progress."""
        output_dir = Path(self._state.output_dir)
        method = self._state.localize_method
        extractor_name = self._state.localize_extractor
        zarr_path = output_dir / method / "feedforward.zarr"
        ff = FeedforwardResult.load_zarr(zarr_path)
        extractor = _EXTRACTOR_CLASSES.get(extractor_name, DiskExtractor)()
        localizer = CameraLocalizer.from_feedforward(
            ff, extractor=extractor, progress_callback=progress_callback
        )
        return ff, localizer
```

- [ ] **Step 8: Update _on_run to drop method/extractor snapshot**

Replace the snapshot block inside `_on_run`:
```python
        # OLD:
        method = self._method_dd.value
        extractor_name = self._extractor_dd.value
        query_path = Path(self._query_input.value.strip())
        warp_corners = self._warp_cb.value
        self._run_btn.disabled = True
        self._status_html.object = "<span style='color:#2596be'>⏳ Localizing…</span>"
        self._loc_thread = threading.Thread(
            target=self._run_localize,
            args=(method, extractor_name, query_path, warp_corners),
            daemon=True,
        )
```
With:
```python
        query_path = Path(self._query_input.value.strip())
        warp_corners = self._warp_cb.value
        self._run_btn.disabled = True
        self._status_html.object = "<span style='color:#2596be'>⏳ Localizing…</span>"
        self._loc_thread = threading.Thread(
            target=self._run_localize,
            args=(query_path, warp_corners),
            daemon=True,
        )
```

- [ ] **Step 9: Update _run_localize — signature, progress, caching, off_screen**

Change signature from:
```python
    def _run_localize(
        self,
        method: str,
        extractor_name: str,
        query_path: Path,
        warp_corners: bool,
    ) -> None:
```
To:
```python
    def _run_localize(
        self,
        query_path: Path,
        warp_corners: bool,
    ) -> None:
```

Replace the localizer-build block (the lines `self._op_log.start_op(...)` → `self._localizer = localizer`):
```python
            self._op_log.start_op(f"Localizing in {self._state.localize_method}")
            # Note: was f"Localizing in {method}" when method was a parameter — now reads AppState

            # Build or reuse cached localizer
            if self._localizer is None:
                def _progress(i: int, total: int) -> None:
                    self._status_html.object = (
                        f"<span style='color:#2596be'>⏳ Indexing frame {i + 1} / {total}…</span>"
                    )
                ff, localizer = self._build_localizer(progress_callback=_progress)
                self._ff_result = ff
                self._localizer = localizer
            else:
                ff = self._ff_result
                localizer = self._localizer
```

Replace `LocalizeScenePanel(...)` construction (add `_off_screen=True`):
```python
            if self._scene_panel is None:
                self._scene_panel = LocalizeScenePanel(
                    pts3d=ff.points,
                    extrinsics=ff.extrinsics,
                    image_paths=ff.image_paths,
                    _off_screen=True,
                )
                self._right_col[:] = [self._scene_panel.panel()]
```

- [ ] **Step 10: Update _on_batch_run and _run_batch**

In `_on_batch_run`, remove the two snapshots and update thread args:
```python
        # Remove:
        method = self._method_dd.value
        extractor_name = self._extractor_dd.value
        # ...
        args=(method, extractor_name, folder_path),

        # Replace thread args with:
        args=(folder_path,),
```

Change `_run_batch` signature from:
```python
    def _run_batch(self, method: str, extractor_name: str, folder_path: Path) -> None:
```
To:
```python
    def _run_batch(self, folder_path: Path) -> None:
```

Replace the `_build_localizer` call inside `_run_batch`:
```python
            # Build or reuse cached localizer
            if self._localizer is None:
                ff, localizer = self._build_localizer()
                self._ff_result = ff
                self._localizer = localizer
            ff = self._ff_result
            localizer = self._localizer
            query_intrinsics = ff.intrinsics.mean(axis=0)
```

- [ ] **Step 11: Update panel() to remove inline DDs from controls_bar**

Replace the `controls_bar` pn.Row:
```python
        controls_bar = pn.Row(
            self._query_input,
            pn.Spacer(sizing_mode="stretch_width"),
            self._run_btn,
            self._warp_cb,
            sizing_mode="stretch_width",
            margin=(4, 0),
        )
```
(Remove `pn.pane.HTML("<b style='color:#8b949e;font-size:12px'>Recon:</b>")`, `self._method_dd`, `self._extractor_dd`.)

- [ ] **Step 12: Run all dashboard tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_localize.py -v
```
Expected: all tests pass.

- [ ] **Step 13: Commit**

```bash
git add collab_splats/dashboard/panes/localize.py tests/dashboard/test_localize.py
git commit -m "refactor(dashboard): localize tab reads sidebar AppState; progress callback; off_screen fix"
```

---

### Task 5: Notebook — fix zarr path

**Files:**
- Modify: `docs/source/tutorials/07_localization/localization.ipynb`

- [ ] **Step 1: Edit the config cell source**

In the notebook JSON, find the cell with `"id": "cell-extract-frames"`. In its `"source"` field, replace:
```
"RECON     = CACHE_DIR / METHOD / \"reconstruction.zarr\"\n",
"\n",
"assert RECON.exists(), (\n",
"    f\"Reconstruction not found at {RECON}. Run 02_pointcloud/feedforward_methods first.\"\n",
")",
```
With:
```
"RECON     = CACHE_DIR / METHOD / \"feedforward.zarr\"\n",
"\n",
"assert RECON.exists(), (\n",
"    f\"feedforward.zarr not found at {RECON}. Run 02_pointcloud/feedforward_methods first.\"\n",
")",
```

- [ ] **Step 2: Add large-dataset timing note to §3 markdown cell**

Find the cell with `"id": "cell-sec4"` (markdown cell `## §3 — Localize query image`). Append to its source:

```
"\n\n> **Note:** Index build reads and extracts local features for every reference frame. Expect ~30 s for 20 frames, several minutes for 100+ frames — one-time cost per scene. (Future: cache features in feedforward.zarr.)"
```

- [ ] **Step 3: Clear stale outputs from the localize cell**

Find the cell with `"id": "cell-localize"`. Set `"outputs": []` to clear the stale weight-loading messages. The cell will re-execute cleanly with the fixed path.

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/07_localization/localization.ipynb
git commit -m "fix(tutorials): localization notebook uses feedforward.zarr; add timing note for large datasets"
```

---

### Task 6: Full test suite

- [ ] **Step 1: Run dashboard + localization tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/ tests/pointcloud/test_localization.py -v -k "not slow"
```
Expected: all pass, no regressions in `test_visualize.py` or other dashboard tests.

- [ ] **Step 2: Verify imports don't break (localize pane import smoke)**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.dashboard.panes.localize import LocalizePane; print('ok')"
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.dashboard.app import App; print('ok')"
```
Expected: both print `ok`.

- [ ] **Step 3: Commit any fixups if needed**

If any files have uncommitted changes from fixing test failures:
```bash
git add collab_splats/dashboard/panes/localize.py collab_splats/dashboard/app.py collab_splats/pointcloud/localization.py
git commit -m "fix(dashboard): localize tab fixups from test run"
```
