# Dashboard Fixes — Design Spec

**Date:** 2026-05-28
**Status:** Approved for planning
**Scope:** Four targeted fixes to the existing dashboard phases 1–2 implementation

---

## Overview

Four issues identified in the live dashboard after Phases 1–2 shipped. Each fix is scoped to a single pane or the App shell. No new panes or architectural changes.

---

## Fix 1 — base_dir drives the "Load existing" selector

### Problem

`App._on_load_existing` reveals a bare `TextInput` for the output directory path. `base_dir` is stored on `App` but never used to populate options. Users must type a full path manually.

### Design

Replace `_output_dir_input` (`TextInput`) with a `Select` widget (`_output_dir_select`) populated at construction time by scanning `base_dir` for immediate subdirectories that contain a `run_config.yaml` file. Add a small `↻` refresh `Button` next to it that re-scans on click.

```
📁  Load existing results
┌─────────────────────────────┐  [ ↻ ]
│ birds_c0043                 ▼│
└─────────────────────────────┘
[ Load session ]
```

**Implementation notes:**
- `_scan_output_dirs(base_dir: Path) -> list[str]` — returns sorted list of dir names where `(base_dir / name / "run_config.yaml").exists()`.
- Called once at `App.__init__` to build initial options, again on refresh click.
- `_on_confirm_session` reads `_output_dir_select.value` and resolves it against `base_dir` to get the full path.
- If `_scan_output_dirs` returns empty list, show `"(no sessions found)"` as the sole disabled option and disable the Load button.

---

## Fix 2 — Load existing → PreprocessPane displays video

### Problem

Two compounding bugs:

1. `pn.Tabs(dynamic=True)` defers rendering of non-active tabs. If the user is on any tab other than Preprocess when they click "Load session", the `video_path` param watch fires but the `_video_pane` widget has not been mounted, so the update is silently lost.
2. `state.video_path` is only set when `run_config.yaml` contains a `video_path` key pointing to an existing file — many configs may lack this, so the pane never receives the signal at all.

### Design

**Bug 1 — tab activation:** After `_on_confirm_session` succeeds in the load-existing branch, programmatically switch `tabs.active = 0` (Preprocess tab) before writing to `AppState`. This ensures the pane is rendered before the param watch fires.

`App` must hold a reference to `tabs` (currently local to `servable()`). Move it to `self._tabs` and pass it into the confirm handler via closure.

**Bug 2 — thread-safe UI update:** In `PreprocessPane._on_video_path_change`, wrap the `_load_video` call in `pn.state.execute(lambda: self._load_video(Path(event.new)))` so the DOM update runs on the Bokeh/Tornado main thread rather than the param-watch thread.

No change to AppState schema.

---

## Fix 3 — Semantics: single extractor, three-panel display

### Problem

`SemanticsPane` instantiates three `ExtractorColumn` objects — each with its own method dropdown, Run button, and image pane. The spec called for simultaneous multi-extractor comparison, but the actual use case is single-extractor inspection with three views: ground truth, PCA features, and cosine similarity.

### Design

Replace the three-column layout with a single-extractor layout:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ◄  [ Frame  42  ]  ►        Extractor: [ DINOv2 SAM ▼ ]  [ Run ] │
├─────────────────────────────────────────────────────────────────────┤
│   Ground truth frame   │   PCA features (RGB)   │  Cosine sim heat │
│                        │                        │                  │
│   [image]              │   [image]              │   [image]        │
├─────────────────────────────────────────────────────────────────────┤
│  Query: [___________________________]  [ Query ]                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Layout changes:**
- Top row: frame prev/next + slider (unchanged) on the left; `pn.HSpacer()` in the middle; `method_dd` (Select) + `run_btn` (Button) on the right — so extractor controls float right, frame controls stay left.
- Three `pn.pane.PNG` panels in a `pn.Row` with equal sizing: `_original_pane`, `_pca_pane`, `_sim_pane`.
- `_sim_pane` starts blank; populated after a successful query.
- Query bar (`_query_input` + `_query_btn`) lives below the three-panel row, always visible but `_query_btn.disabled = True` until extractor is run and is queryable.

**Class changes:**
- Delete the `ExtractorColumn` class entirely.
- `SemanticsPane` directly holds: `_method_dd`, `_run_btn`, `_extractor`, `_feature_zarr_path`, `_pca_pane`, `_sim_pane`, `_original_pane`, `_query_input`, `_query_btn`.
- `_run_extraction` background thread: instantiate extractor, run `extract_and_cache_from_zarr`, then render PCA for current frame into `_pca_pane`.
- `_on_frame_change`: reload `_original_pane` and (if extractor ready) refresh `_pca_pane` for new frame index.
- `_on_query`: run `score_queries` on current frame features, render similarity heatmap into `_sim_pane`.

**Query gating:** `_query_btn` enabled only when `isinstance(self._extractor, BaseQueryableExtractor)` and `_feature_zarr_path is not None`. When disabled, `_status_html` shows `"Run a queryable extractor (e.g. DINOv2 SAM) to enable text queries"` in dim gray.

---

## Fix 4 — Visualize panel reorganization

### Problems

a. No visual separation between Scene A and Scene B.
b. Load button visually disconnected from dataset/backend dropdowns.
c. PCD/Mesh/Similarity are three loose `Button` widgets — no visual grouping, no default.
d. Query input lives at top of `VisualizePane`; should be at bottom of each scene.
e. Frustum `Toggle` widget is out of context in the mode row.

### Design

#### 4a — Vertical divider between scenes

`VisualizePane.panel()` changes from:
```python
pn.Row(scene_a.panel(), scene_b.panel())
```
to:
```python
pn.Row(
    scene_a.panel(),
    pn.pane.HTML("<div style='border-left:1px solid #444;height:100%;margin:0 8px'></div>"),
    scene_b.panel(),
)
```

#### 4b — Load button aligned with selectors

`controls_row = pn.Row(self._dataset_dd, self._backend_dd, self._load_btn, align="end")`

`align="end"` ensures the button baseline aligns with the bottom of the dropdowns.

#### 4c — RadioButtonGroup replacing three Buttons

Replace `_pcd_btn`, `_mesh_btn`, `_sim_btn` with:
```python
self._mode_selector = pn.widgets.RadioButtonGroup(
    options=["Points", "Mesh", "Similarity"],
    value="Mesh",
    button_type="success",
    width=360,
)
self._mode_selector.param.watch(lambda e: self._on_mode_change(e.new), "value")
```

`_on_mode_change` maps `"Points"→"PCD"`, `"Mesh"→"Mesh"`, `"Similarity"→"Similarity"` for backward compat with internal `self.mode` param string.

Mesh mode is the default (`value="Mesh"`). After wiring the watch, call `self._on_mode_change("Mesh")` explicitly in `__init__` to set the initial viewer state — the watch alone does not fire on construction.

#### 4d — Query moved to bottom of ScenePanel

Remove `_query_bar` from `VisualizePane`. Each `ScenePanel` gets its own:
```python
self._sim_query_input = pn.widgets.TextInput(placeholder="Enter text query…", width=260)
self._sim_query_btn = pn.widgets.Button(name="Query", button_type="success", width=80)
self._sim_query_row = pn.Row(self._sim_query_input, self._sim_query_btn, visible=False)
```

`_sim_query_row.visible` toggled true when mode == "Similarity". `VisualizePane` shared query bar removed.

`VisualizePane._a_extractor_dd` and `_b_extractor_dd` removed (were part of shared bar). Each scene's per-scene extractor dropdown (`_extractor_dd`) stays inside `ScenePanel`.

#### 4e — Frustum as Checkbox in Points options row

Replace `pn.widgets.Toggle(name="Show frustums")` with `pn.widgets.Checkbox(name="Show frustums", value=False)`.

Move it (plus `_point_size_slider`) into a `_points_options_row` that is only `visible` when mode == `"Points"`:
```python
self._points_options_row = pn.Row(
    self._frustum_check,
    self._point_size_slider,
    visible=False,
)
```

#### ScenePanel.panel() layout

```python
def panel(self) -> pn.Column:
    controls_row = pn.Row(self._dataset_dd, self._backend_dd, self._load_btn, align="end")
    return pn.Column(
        f"### Scene {self._scene_id}",
        controls_row,
        self._mode_selector,
        self._vtk_pane,
        self._points_options_row,   # visible only in Points mode
        self._sim_query_row,        # visible only in Similarity mode
        pn.Row(self._reset_btn, self._snapshot_btn),
        self._status_html,
        sizing_mode="stretch_both",
    )
```

---

## Files changed

| File | Change |
|---|---|
| `dashboard/app.py` | Fix 1 (dir scanner + Select widget), Fix 2 (tab switch before state write) |
| `dashboard/panes/preprocess.py` | Fix 2 (`pn.state.execute` wrap in watch handler) |
| `dashboard/panes/semantics.py` | Fix 3 (drop ExtractorColumn, single-extractor three-panel layout) |
| `dashboard/panes/visualize.py` | Fix 4 (divider, alignment, RadioButtonGroup, per-scene query, frustum checkbox) |

---

## Out of scope

- Shared cross-scene query bar (dropped — per-scene query in Fix 4d is sufficient)
- Any changes to `ReconstructPane`, `LocalizePane`, or `OperationLog`
- New AppState fields
