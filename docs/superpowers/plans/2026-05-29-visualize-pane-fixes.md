# Visualize Pane Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix camera frustum double-inversion bug, suppress VTK/model-load noise, and collapse the two-scene compositor into a single ScenePanel.

**Architecture:** Four independent changes: (1) remove erroneous `np.linalg.inv` at frustum call sites (function already inverts internally); (2) add VTK + Python warnings suppression at entry point; (3) drop `scene_id` param from ScenePanel and make AppState watchers unconditional; (4) delete `VisualizePane` compositor and wire `app.py` directly to `ScenePanel`.

**Tech Stack:** Python 3.11, Panel/PyVista dashboard, VTK, HuggingFace transformers, pytest.

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/dashboard/panes/visualize.py` | Fix frustum call site (line 456); remove `scene_id` from `ScenePanel`; add `wire_tabs` to `ScenePanel`; delete `VisualizePane` class |
| `collab_splats/dashboard/panes/localize.py` | Fix frustum call sites (lines 122, 166) |
| `collab_splats/utils/notebook.py` | Fix frustum call site (line 94) |
| `collab_splats/dashboard/__main__.py` | Add `_suppress_noise()` — VTK off + warnings filters |
| `collab_splats/dashboard/app.py` | Replace `VisualizePane` import/instantiation with `ScenePanel` |
| `tests/dashboard/test_visualize.py` | Update `ScenePanel` tests to new signature (no `scene_id`) |

---

### Task 1: Fix frustum double-inversion

`create_camera_frustum_pyvista(pose)` expects **w2c** and inverts internally (`c2w = np.linalg.inv(pose)`). Four call sites erroneously pass `np.linalg.inv(ext)` — already c2w — causing a second inversion. Frustums land at `w2c @ camera_origin` instead of the camera centre.

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py:456`
- Modify: `collab_splats/dashboard/panes/localize.py:122`
- Modify: `collab_splats/dashboard/panes/localize.py:166`
- Modify: `collab_splats/utils/notebook.py:94`

- [ ] **Step 1: Confirm all 4 double-inversion sites**

```bash
grep -n "create_camera_frustum_pyvista.*np.linalg.inv" \
  collab_splats/dashboard/panes/visualize.py \
  collab_splats/dashboard/panes/localize.py \
  collab_splats/utils/notebook.py
```

Expected — 4 lines matching, one per file (localize.py has 2):
```
collab_splats/dashboard/panes/visualize.py:456:            frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
collab_splats/dashboard/panes/localize.py:122:            frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
collab_splats/dashboard/panes/localize.py:166:            frustum = create_camera_frustum_pyvista(np.linalg.inv(query_ext))
collab_splats/utils/notebook.py:94:        frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=scale)
```

- [ ] **Step 2: Fix `visualize.py:456`**

In `collab_splats/dashboard/panes/visualize.py`, inside `_add_frustums`:
```python
# Before
frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
# After
frustum = create_camera_frustum_pyvista(ext)
```

- [ ] **Step 3: Fix `localize.py:122` (loop over reference extrinsics)**

In `collab_splats/dashboard/panes/localize.py`, inside the reference frustum loop:
```python
# Before
frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
# After
frustum = create_camera_frustum_pyvista(ext)
```

- [ ] **Step 4: Fix `localize.py:166` (query camera frustum)**

In `collab_splats/dashboard/panes/localize.py`, inside `highlight`:
```python
# Before
frustum = create_camera_frustum_pyvista(np.linalg.inv(query_ext))
# After
frustum = create_camera_frustum_pyvista(query_ext)
```

- [ ] **Step 5: Fix `notebook.py:94`**

In `collab_splats/utils/notebook.py`, inside `add_camera_frustums`:
```python
# Before
frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=scale)
# After
frustum = create_camera_frustum_pyvista(ext, scale=scale)
```

- [ ] **Step 6: Verify no remaining double-inversion sites**

```bash
grep -rn "create_camera_frustum_pyvista.*np.linalg.inv" collab_splats/
```

Expected: no output.

- [ ] **Step 7: Run tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py \
        collab_splats/dashboard/panes/localize.py \
        collab_splats/utils/notebook.py
git commit -m "fix(dashboard): remove double-inversion in camera frustum call sites

create_camera_frustum_pyvista already inverts w2c→c2w internally;
passing np.linalg.inv(ext) caused a second inversion and wrong positions."
```

---

### Task 2: Suppress VTK EGL and model-load noise

**Files:**
- Modify: `collab_splats/dashboard/__main__.py`

VTK emits `vtkEGLRenderWindow: Unable to eglMakeCurrent: 12290` to stderr on every `synchronize()` call in headless env. Torch emits `UserWarning: copying from a non-meta parameter` per-weight during Talk2DINO load. HuggingFace transformers emits "Some weights... were not used". All benign.

- [ ] **Step 1: Read current `__main__.py`**

Open `collab_splats/dashboard/__main__.py`. Confirm current content:
- `from __future__ import annotations`
- `import argparse, importlib, warnings`
- `DASHBOARDS` dict
- `def main():`

- [ ] **Step 2: Add `_suppress_noise` function**

In `collab_splats/dashboard/__main__.py`, add `_suppress_noise()` before `main()` and call it at the top of `main()`:

```python
from __future__ import annotations

import argparse
import importlib
import warnings

DASHBOARDS = {
    "app": "collab_splats.dashboard.app:run_app",
    "semantics": "collab_splats.dashboard.app:run_app",
}


def _suppress_noise() -> None:
    """Suppress known benign warnings from VTK, torch, and transformers."""
    import vtk
    import transformers

    vtk.vtkObject.GlobalWarningDisplayOff()
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=".*non-meta parameter.*")


def main() -> None:
    _suppress_noise()
    parser = argparse.ArgumentParser(
        prog="collab-dashboard",
        description="Launch a collab-splats interactive dashboard.",
    )
    parser.add_argument("mode", choices=list(DASHBOARDS.keys()))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--base-dir", default="/workspace/outputs")
    args = parser.parse_args()

    if args.mode == "semantics":
        warnings.warn(
            "'collab-dashboard semantics' is deprecated — use 'collab-dashboard app'",
            DeprecationWarning,
            stacklevel=2,
        )

    module_path, func_name = DASHBOARDS[args.mode].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    run_fn = getattr(mod, func_name)
    run_fn(host=args.host, port=args.port, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/dashboard/__main__.py
git commit -m "fix(dashboard): suppress VTK EGL and model-load warning noise at entry point"
```

---

### Task 3: Simplify ScenePanel — remove scene_id, add wire_tabs

`ScenePanel` currently takes a `scene_id: str` positional param that gates AppState watchers (`if scene_id == "A":`). With a single scene, this guard is dead. We remove it, wire AppState unconditionally, and move `wire_tabs` from `VisualizePane` onto `ScenePanel` so `app.py` can call it directly.

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`
- Modify: `tests/dashboard/test_visualize.py`

- [ ] **Step 1: Update `ScenePanel.__init__` signature**

In `collab_splats/dashboard/panes/visualize.py`, change the constructor at line 116:

```python
# Before
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

# After
def __init__(
    self,
    base_dir: Path,
    state: AppState,
    op_log: OperationLog,
    _off_screen: bool = False,
    **params: Any,
) -> None:
    super().__init__(**params)
    self._base_dir = Path(base_dir) if base_dir else Path("/workspace/outputs")
    self._state = state
    self._op_log = op_log
```

- [ ] **Step 2: Remove `if scene_id == "A":` guard**

In `collab_splats/dashboard/panes/visualize.py`, in `__init__`, replace:
```python
# Scene A: watch AppState for auto-suggest and rescan
if scene_id == "A":
    state.param.watch(self._on_feedforward_result, "feedforward_result")
    state.param.watch(self._on_lifted_features_path, "lifted_features_path")
```

With:
```python
# Watch AppState for auto-suggest and rescan
state.param.watch(self._on_feedforward_result, "feedforward_result")
state.param.watch(self._on_lifted_features_path, "lifted_features_path")
```

- [ ] **Step 3: Update section heading comment**

In `collab_splats/dashboard/panes/visualize.py`, change:
```python
####################################################################
# AppState watchers (Scene A only)
####################################################################
```

To:
```python
####################################################################
# AppState watchers
####################################################################
```

- [ ] **Step 4: Update `_on_snapshot` filename**

In `collab_splats/dashboard/panes/visualize.py`, in `_on_snapshot`:
```python
# Before
path = Path(f"scene_{self._scene_id}_snapshot.png")
# After
path = Path("scene_snapshot.png")
```

- [ ] **Step 5: Update panel heading**

In `collab_splats/dashboard/panes/visualize.py`, in `panel()`:
```python
# Before
f"### Scene {self._scene_id}",
# After
"### Scene",
```

- [ ] **Step 6: Update class docstring**

In `collab_splats/dashboard/panes/visualize.py`, change:
```python
# Before
class ScenePanel(param.Parameterized):
    """Self-contained per-scene 3D viewer panel.

    Manages dataset/backend selection, mode switching (PCD/Mesh/Similarity),
    and a PyVista plotter embedded via pn.pane.VTK.
    """

# After
class ScenePanel(param.Parameterized):
    """3D viewer panel for a single reconstruction scene.

    Manages dataset/backend selection, mode switching (PCD/Mesh/Similarity),
    and a PyVista plotter embedded via pn.pane.VTK.
    """
```

- [ ] **Step 7: Add `wire_tabs` method to ScenePanel**

In `collab_splats/dashboard/panes/visualize.py`, add `wire_tabs` directly before `rescan`:

```python
def wire_tabs(self, tabs: pn.Tabs, tab_index: int) -> None:
    """Connect tab activation signal so modes rescan when this tab becomes active."""
    def _on_tab_change(event: Any) -> None:
        if event.new == tab_index:
            self.rescan()
    tabs.param.watch(_on_tab_change, "active")

def rescan(self) -> None:
    """Re-scan available modes (called on tab activation)."""
    self._scan_available_modes()
```

- [ ] **Step 8: Update existing tests in `test_visualize.py`**

The existing `ScenePanel` tests pass `scene_id` — update them to the new signature. In `tests/dashboard/test_visualize.py`:

**`test_scene_panel_constructs`** — remove `scene_id`, remove `_scene_id` assertion:
```python
# Before
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

# After
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
```

**`test_scene_panel_scan_available_modes_no_result`** — remove positional `"A"`:
```python
# Before
sp = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)

# After
sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
```

**`test_scene_panel_scan_available_modes_with_mesh`** — same:
```python
# Before
sp = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)

# After
sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
```

**`test_scene_panel_scan_available_modes_with_features`** — same:
```python
# Before
sp = ScenePanel("A", tmp_path, state, _make_op_log(), _off_screen=True)

# After
sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
```

Search for any remaining `ScenePanel("` or `scene_id=` in `test_visualize.py` and update similarly — the pattern is always remove the leading `"A"` positional arg or `scene_id=` keyword arg.

- [ ] **Step 9: Add wire_tabs test**

In `tests/dashboard/test_visualize.py`, add after the existing ScenePanel smoke tests:

```python
def test_scene_panel_wire_tabs(tmp_path):
    """wire_tabs connects tab activation → rescan without error."""
    state = AppState()
    sp = ScenePanel(tmp_path, state, _make_op_log(), _off_screen=True)
    tabs = pn.Tabs(("Visualize", pn.pane.Str("x")))
    sp.wire_tabs(tabs, 0)  # must not raise
    assert True
```

- [ ] **Step 10: Run tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -v
```

Expected: all pass, no `scene_id` errors.

- [ ] **Step 11: Run full suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 12: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py \
        tests/dashboard/test_visualize.py
git commit -m "refactor(dashboard): remove scene_id from ScenePanel; AppState watchers unconditional"
```

---

### Task 4: Remove VisualizePane, update app.py

`VisualizePane` is now unused. Delete it and wire `app.py` directly to `ScenePanel` (which now has `wire_tabs`).

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py` — delete `VisualizePane` class (~60 lines)
- Modify: `collab_splats/dashboard/app.py` — replace `VisualizePane` with `ScenePanel`

- [ ] **Step 1: Delete `VisualizePane` from `visualize.py`**

In `collab_splats/dashboard/panes/visualize.py`, delete the entire block from the section divider through the end of the file:

```python
########################################################################
# VisualizePane
########################################################################


class VisualizePane(param.Parameterized):
    """Compositor: two ScenePanels side by side with a vertical divider."""

    def __init__(
        self,
        state: AppState,
        op_log: "OperationLog",
        base_dir: Path,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._state = state
        self._op_log = op_log

        self._scene_a = ScenePanel("A", base_dir, state, op_log)
        self._scene_b = ScenePanel("B", base_dir, state, op_log)

    ####################################################################
    # Tab activation rescan
    ####################################################################

    def _rescan_both_scenes(self) -> None:
        """Re-scan available modes on both scenes (called on tab activation)."""
        self._scene_a.rescan()
        self._scene_b.rescan()

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
        """Return Panel layout for Tab 4."""
        divider = pn.pane.HTML(
            "<div style='border-left:1px solid #444;height:100%;margin:0 8px'></div>",
            width=18,
            sizing_mode="stretch_height",
        )
        scenes_row = pn.Row(
            self._scene_a.panel(),
            divider,
            self._scene_b.panel(),
            sizing_mode="stretch_both",
        )
        return pn.Column(
            pn.pane.HTML("<h3 style='color:#7ec8e3;margin:0 0 8px 0'>Visualize</h3>"),
            scenes_row,
            sizing_mode="stretch_both",
        )
```

The file should end after `ScenePanel.panel()` closes.

- [ ] **Step 2: Update `app.py` import**

In `collab_splats/dashboard/app.py`, line 17:
```python
# Before
from collab_splats.dashboard.panes.visualize import VisualizePane

# After
from collab_splats.dashboard.panes.visualize import ScenePanel
```

- [ ] **Step 3: Update `App.__init__` instantiation**

In `collab_splats/dashboard/app.py`, line 65:
```python
# Before
"Visualize": VisualizePane(state=self._state, op_log=self._op_log, base_dir=self._base_dir),

# After
"Visualize": ScenePanel(base_dir=self._base_dir, state=self._state, op_log=self._op_log),
```

- [ ] **Step 4: Update wire_tabs comment**

In `collab_splats/dashboard/app.py`, line 216:
```python
# Before
# Wire tab activation → VisualizePane mode rescan

# After
# Wire tab activation → ScenePanel mode rescan
```

The `wire_tabs` call itself (`self._panes["Visualize"].wire_tabs(...)`) is unchanged — `ScenePanel` now owns the method.

- [ ] **Step 5: Verify no remaining VisualizePane references**

```bash
grep -rn "VisualizePane" collab_splats/ tests/
```

Expected: no output.

- [ ] **Step 6: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py \
        collab_splats/dashboard/app.py
git commit -m "refactor(dashboard): remove VisualizePane compositor; app.py wires directly to ScenePanel"
```
