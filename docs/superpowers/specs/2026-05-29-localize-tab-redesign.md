# Localize Tab Redesign + Notebook Fix

**Date:** 2026-05-29  
**Status:** approved

## Problem

Two bugs, one UX mismatch:

1. **Notebook §3 hangs on large datasets** — `CameraLocalizer.__init__` eagerly extracts local features for all N reference frames upfront. Works for birds_c0043 (21 frames); hangs for rats_* (100–300+ frames). Secondary: notebook uses `reconstruction.zarr` but new datasets only have `feedforward.zarr`.

2. **Dashboard localize tab display broken** — `LocalizeScenePanel` created with `off_screen=False` (default). In the headless server, `pn.pane.VTK` init fails silently → `panel()` returns `None` → `_right_col[:] = [None]` → layout broken.

3. **Dashboard page times out on Run** — `_build_localizer` loads a DiskExtractor + extracts DISK features for all N frames in background thread. For 100+ frame datasets this takes minutes; WebSocket drops before the thread finishes.

4. **UX mismatch** — LocalizePane has inline method/extractor dropdowns; all other tabs are sidebar-driven.

## Approach

Approach A: Sidebar-driven controls + lazy-cached index with frame-level progress.  
Approach C (future): Pre-compute and cache features in zarr alongside feedforward.zarr.

---

## Design

### 1. AppState — two new fields

```python
localize_method: str = ""
localize_extractor: str = "DISK+LightGlue"
```

Same pattern as existing `pointcloud_backend` / `semantic_extractor`. Cleared when `output_dir` changes.

### 2. Sidebar — Localize section

`App._build_sidebar()` gains a **Localize** section (below Models, above View):

```
[h3] Localize
[Select] Method          ← populated by _scan_recon_methods(output_dir)
[Select] Extractor       ← ["DISK+LightGlue", "XFeat+MNN"]
```

Populated/cleared on `output_dir` change (same watcher that fires for Visualize methods scan). Both DDs write to `AppState.localize_method` / `AppState.localize_extractor` via `param.watch`.

### 3. LocalizePane — simplified main panel

Remove inline `_method_dd` and `_extractor_dd`. Watch `AppState.localize_method` and `AppState.localize_extractor` — any change invalidates cached localizer (`self._localizer = None`, `self._ff_result = None`).

Main panel layout:

```
[controls_bar]  query path input  |  ▶ Localize  |  warp-corners checkbox
[split]         left: correspondence PNG    |    right: 3D scene
[batch_section] (Card, collapsed)
```

No method/extractor dropdowns in main panel.

### 4. CameraLocalizer — progress callback

```python
def __init__(
    self,
    ...,
    progress_callback: Callable[[int, int], None] | None = None,
):
```

Called `progress_callback(frame_idx, total_frames)` after each frame's features are extracted (inside the existing `for path in image_paths` loop). No other change to the class.

Public API unchanged for callers that don't pass the kwarg.

### 5. LocalizePane — lazy index with progress

`_run_localize` (background thread):
1. If `self._localizer is None`: build index, updating `_status_html` each frame via progress callback → `⏳ Indexing frame 23 / 87…`
2. Cache `self._localizer` and `self._ff_result` — skipped on subsequent queries with same method/extractor
3. Proceed to localize query image as before

Invalidation: `_on_method_extractor_changed` watches both AppState fields, clears `self._localizer = None`.

### 6. LocalizeScenePanel — off_screen fix

`_run_localize` creates `LocalizeScenePanel(..., _off_screen=True)`. Parameter default stays `False` for tests.

### 7. Notebook — zarr path fix

- `RECON = CACHE_DIR / METHOD / "reconstruction.zarr"` → `RECON = CACHE_DIR / METHOD / "feedforward.zarr"`
- Update assert message
- Add markdown note in §3 cell: large datasets (100+ frames) take several minutes to index — normal

---

## Files changed

| File | Change |
|---|---|
| `collab_splats/dashboard/state.py` | Add `localize_method`, `localize_extractor` fields |
| `collab_splats/dashboard/app.py` | Add Localize sidebar section; wire DDs to AppState; rescan on output_dir change |
| `collab_splats/dashboard/panes/localize.py` | Remove inline DDs; watch AppState fields; progress callback; `off_screen=True` |
| `collab_splats/pointcloud/localization.py` | Add `progress_callback` kwarg to `CameraLocalizer.__init__` |
| `docs/source/tutorials/07_localization/localization.ipynb` | Fix zarr path; add large-dataset note |

## Out of scope

- Approach C (zarr-cached features) — noted in `CameraLocalizer.__init__` as future work
- Batch mode UI — unchanged
- DISK model pre-download — separate concern

## Tests

- `tests/dashboard/test_localize.py` — existing tests should pass unchanged (no `progress_callback` arg → works as before; `_off_screen=True` already used in tests via mock)
- No new tests required — the progress callback is trivially correct and covered by existing localize flow tests
