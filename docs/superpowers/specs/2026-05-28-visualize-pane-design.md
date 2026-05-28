# VisualizePane Design Spec
**Date:** 2026-05-28  
**Tab:** 4 — Visualize  
**Status:** Approved, ready for implementation

---

## Purpose

Side-by-side interactive 3D comparison of two independently selected scenes/backends. Each scene controls its representation (PCD / Mesh / Similarity). A single shared query bar at the top handles semantic text queries for all active Similarity-mode scenes.

---

## AppState Changes

Phase 2 refactored `state.frames: list[np.ndarray]` → `state.frames_zarr_path: Path | None`. Update `collab_splats/dashboard/state.py`:

```python
class AppState(param.Parameterized):
    output_dir = param.Parameter(default=None)
    video_path = param.Parameter(default=None)
    frames_zarr_path = param.Parameter(default=None)   # replaces frames
    feedforward_result = param.Parameter(default=None)
    feature_maps_path = param.Parameter(default=None)
    lifted_features_path = param.Parameter(default=None)
```

VisualizePane does not use `frames_zarr_path` directly — this change is noted for completeness.

---

## File Layout

```
collab_splats/dashboard/panes/
  visualize.py    ← ScenePanel + VisualizePane (single file)
```

---

## Architecture

### `ScenePanel(param.Parameterized)`

Self-contained per-scene unit. Two instances composed by `VisualizePane`.

**Constructor:** `ScenePanel(scene_id: str, base_dir: Path, state: AppState, **params)`

`scene_id` is `"A"` or `"B"`. `base_dir` is the root output directory for dataset discovery (e.g. `/workspace/outputs`).

**Internal state:**

| Field | Type | Purpose |
|-------|------|---------|
| `_result` | `FeedforwardResult \| None` | Loaded on explicit Load |
| `_lifted_normed` | `np.ndarray \| None` | (P, D) float32, L2-normalised at load |
| `_extractor_cache` | `dict[str, BaseQueryableExtractor]` | Lazy per-name cache |
| `_plotter` | `pv.Plotter` | Created once at `__init__`, `off_screen=False` |
| `_vtk_pane` | `pn.pane.VTK` | Wraps `_plotter.ren_win`; never recreated |
| `_mode` | `str` | `"PCD"` \| `"Mesh"` \| `"Similarity"` |
| `_available_modes` | `set[str]` | Populated by `_scan_available_modes()` |
| `_available_extractors` | `list[str]` | Extractor names with a features.zarr on disk |
| `_load_thread` | `threading.Thread \| None` | Guards against concurrent loads |

### `VisualizePane(param.Parameterized)`

Thin compositor. Owns two `ScenePanel` instances and the shared query bar.

**Constructor:** `VisualizePane(state: AppState, op_log: OperationLog, base_dir: Path, **params)`

Watches each scene's `_mode` param via `param.watch` to show/hide the shared query bar.

---

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Shared query bar — visible when ≥1 scene in Similarity]   │
│  [text input          ] [Scene A extractor ▾] [Scene B ▾] → │
├──────────────────────────┬──────────────────────────────────┤
│  Scene A                 │  Scene B                         │
│  [Dataset ▾] [Backend ▾] │  [Dataset ▾] [Backend ▾]        │
│  [Load]                  │  [Load]                          │
│  [PCD][Mesh][Similarity] │  [PCD][Mesh][Similarity]         │
│  ┌──────────────────┐    │  ┌──────────────────────┐        │
│  │   VTK viewer     │    │  │   VTK viewer         │        │
│  └──────────────────┘    │  └──────────────────────┘        │
│  [↺ Reset] [📷 Snapshot] │  [↺ Reset] [📷 Snapshot]        │
└──────────────────────────┴──────────────────────────────────┘
```

Scene B extractor slot in shared query bar is hidden when Scene B is not in Similarity mode.

---

## ScenePanel Widget Reference

All widgets follow `self._widget_name` convention.

| Widget | Type | Notes |
|--------|------|-------|
| `_dataset_dd` | `pn.widgets.Select` | Populated from `base_dir` scan |
| `_backend_dd` | `pn.widgets.Select` | Populated from `<dataset>/` scan |
| `_load_btn` | `pn.widgets.Button` | Disabled while load thread running |
| `_pcd_btn` | `pn.widgets.Button` | `button_type="primary"` when active |
| `_mesh_btn` | `pn.widgets.Button` | `disabled=True` + name "Mesh (no mesh.ply)" if unavailable |
| `_sim_btn` | `pn.widgets.Button` | `disabled=True` + name "Similarity (no features)" if unavailable |
| `_vtk_pane` | `pn.pane.VTK` | `sizing_mode="stretch_both"`, min height 500px |
| `_reset_btn` | `pn.widgets.Button` | Calls `_plotter.reset_camera()` |
| `_snapshot_btn` | `pn.widgets.Button` | `_plotter.screenshot()` → PNG download |
| `_status_html` | `pn.pane.HTML` | Load/error status badge |

---

## Data Flow

### Dataset / Backend Discovery

`_scan_datasets()` — lists subdirectories of `base_dir` that contain a `run_config.yaml`.  
`_scan_backends(dataset_dir)` — lists subdirectories of `dataset_dir` that contain a `feedforward.zarr`.

Called when `_dataset_dd` or `_backend_dd` changes.

### Load Path

```
user clicks Load
→ _on_load() [daemon thread]
  → FeedforwardResult.load_zarr(<dataset_dir>/<backend>/feedforward.zarr)
  → _scan_available_modes()
  → _result = loaded result
  → _rebuild_viewer()
  → UI thread: update mode button states
```

Load button is disabled for the duration of the thread. On successful load, `_plotter.reset_camera()` is called — camera always resets when a new dataset/backend is loaded.

### Mode Availability — `_scan_available_modes()`

Runs after each Load, on tab activation, and when `state.lifted_features_path` changes (Scene A only).

| Mode | Enabled when |
|------|-------------|
| PCD | `feedforward.zarr` loaded successfully |
| Mesh | `<dataset_dir>/<backend>/mesh/mesh.ply` exists |
| Similarity | ≥1 path matching `<dataset_dir>/<backend>/semantics/*/features.zarr` exists |

`_available_extractors` = list of extractor names discovered from the semantics glob.

If the currently active mode becomes unavailable after a re-scan, fall back to PCD (or the first available mode).

### Dynamic Re-scan

- **Scene A**: `param.watch` on `state.lifted_features_path` — re-runs `_scan_available_modes()` immediately when semantics pipeline writes a new path.
- **Both scenes**: `VisualizePane` watches the parent `pn.Tabs` `value` param — when Tab 4 becomes active, calls `_rescan_both_scenes()`.

### Mode Switch

```
_on_mode_change(new_mode)
  → _mode = new_mode
  → _plotter.clear()
  → PCD:        _rebuild_pcd_viewer()
  → Mesh:       _rebuild_mesh_viewer()
  → Similarity: _load_lifted_features() if _lifted_normed is None
                _rebuild_sim_viewer()
```

Camera state is preserved across mode switches (plotter not recreated). Only "Reset camera" and a new Load reset it.

After every `plotter.clear()` + actor rebuild, call `_vtk_pane.synchronize()` to push the updated `ren_win` state to the browser (Panel 1.x VTK pane requires this explicit sync after actor changes).

### PCD Viewer

```python
cloud = pointcloud_to_polydata(result.points, RGB=result.colors)
plotter.add_mesh(cloud, scalars="RGB", rgb=True, point_size=point_size)
```

Frustum toggle: iterates `result.extrinsics`, calls `create_camera_frustum_pyvista(np.linalg.inv(ext))` per camera, adds/removes actors. Point size slider (1–10, default 2) calls `_rebuild_pcd_viewer()` on change.

Both `pointcloud_to_polydata` and `create_camera_frustum_pyvista` are in `collab_splats/utils/visualization.py` — reuse directly.

### Mesh Viewer

```python
mesh = pv.read(mesh_ply_path)
plotter.add_mesh(mesh, rgb=True)
```

### Similarity Viewer

At load time (`_load_lifted_features()`):
```python
store = zarr.open(str(features_zarr_path), mode="r")
feats = store["features"][:]           # (P, D) float32
norms = np.linalg.norm(feats, axis=1, keepdims=True)
_lifted_normed = feats / np.maximum(norms, 1e-8)
```

Before first query: render PCD with original RGB colors, status badge "enter a query to colour by similarity".

### Similarity Query Path

```
user types text + clicks Query (shared bar)
→ VisualizePane._on_query() [daemon thread per active Similarity scene]
  → extractor = _extractor_cache.setdefault(
        name, BaseQueryableExtractor.create(name))  # RegistryMixin.create() — verify exact API vs registry dict at impl time
  → query_vec = extractor.encode_text(text)   # (D,) normalised
  → sims = _lifted_normed @ query_vec          # (P,) cosine similarity [-1, 1]
  → colors = viridis_colormap(sims)            # (P, 3) uint8
  → plotter.clear(); _rebuild_sim_viewer(colors=colors)
```

Viridis colormap always used for similarity — not configurable.

---

## Shared Query Bar

Owned by `VisualizePane`. Visible when ≥1 scene is in Similarity mode.

| Widget | Type | Notes |
|--------|------|-------|
| `_query_input` | `pn.widgets.TextInput` | Shared text, submit on Enter or button click |
| `_scene_a_extractor_dd` | `pn.widgets.Select` | Options = `scene_a._available_extractors`; hidden if Scene A not in Similarity |
| `_scene_b_extractor_dd` | `pn.widgets.Select` | Options = `scene_b._available_extractors`; hidden if Scene B not in Similarity |
| `_query_btn` | `pn.widgets.Button` | Fires `_on_query()` for all active Similarity scenes in parallel threads |

---

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| zarr load fails | `_status_html` badge "load failed: {msg}"; mode buttons stay disabled |
| no `feedforward.zarr` at path | badge "no feedforward.zarr found" |
| extractor load fails | badge "extractor load failed: {name}"; Query button re-enabled |
| `encode_text()` raises | badge shows error; similarity coloring reverts to last good state (or original RGB) |
| feature/point shape mismatch | Similarity mode disabled; badge "feature/point count mismatch — re-run semantics" |
| P > 500k points | `op_log.warn("large PCD: {P} points — may be slow to render")`; no hard cap |
| concurrent load attempt | Load button disabled while thread running — not possible |

---

## AppState Interaction

| Direction | Field | When |
|-----------|-------|------|
| Watch (Scene A) | `state.feedforward_result` | Pre-fills dataset/backend dropdowns (no auto-load) |
| Watch (Scene A) | `state.lifted_features_path` | Triggers `_scan_available_modes()` |
| Writes | — | None — VisualizePane is read-only |

---

## Out of Scope

- Point cloud downsampling (no size cap beyond the warning)
- Camera synchronisation between Scene A and Scene B
- Correspondence visualisation (belongs to LocalizePane, Tab 5)
- Mesh quality metrics
