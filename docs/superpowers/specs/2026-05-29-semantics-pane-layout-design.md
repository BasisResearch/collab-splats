# SemanticsPane Layout Fix — Design Spec

**Date:** 2026-05-29
**Scope:** Layout restructure + auto-discover cached features on load

---

## Overview

Three problems with current SemanticsPane layout:

1. Extractor selector sits top-right, buried after frame controls.
2. Image area has no reserved height — panel reflows when features load.
3. No auto-discovery: user must press Run even if features are already cached on disk.

---

## Section 1: Layout Restructure

### New layout (top → bottom)

```
Column (stretch_width):
  H3 "Semantics"
  extractor_row:  [Extractor: | method_dd | ▶ Run | status_html]
  image_area:     Column(height=270, sizing_mode="fixed")
                    Row: [GT (320×240) | PCA (320×240) | Sim (320×240)]
  frame_controls: Row: [◄ | frame_slider | ► | frame_count]
  query_row:      Row: [Query: | text_input | Query btn]
```

### Key constraints

- **Extractor row top-left**: `method_dd` and `_run_btn` are the first controls, before frame navigation.
- **Constant image height**: wrap `image_row` in `pn.Column(height=270, sizing_mode="fixed")`. Panel reserves the space regardless of whether PNG panes are populated.
- **Frame controls below images**: `_frame_slider` row moves underneath the image area (acts as "scrollbar" for frame browsing).
- **Query row last**: unchanged position, just below frame controls.

### What moves vs stays

| Widget | Before | After |
|---|---|---|
| `_method_dd` + `_run_btn` | top-right after spacer | top-left, first in extractor_row |
| `_status_html` | standalone row | inline after `_run_btn` in extractor_row |
| `_frame_slider` + nav | top-left | below image_area |
| `image_row` | unsized | wrapped in fixed-height Column |
| `query_row` | bottom | bottom (unchanged) |

---

## Section 2: Auto-Discover Cached Features

### Trigger conditions

Auto-discover fires whenever either of these changes:

- `state.output_dir` (new session loaded)
- `state.frames_zarr_path` (frames processed)
- `_method_dd.value` (extractor switched)

All three watchers call the shared `_try_discover_cache()` helper.

### Discovery logic (`_try_discover_cache`)

```
1. Guard: both state.output_dir and state.frames_zarr_path must be set.
2. candidate = output_dir / "features" / selected_method / (zarr directory)
3. Valid if: candidate exists AND zarr.open(candidate)["features"] array present.
4. If valid:
   - instantiate extractor_cls() → self._extractor
   - set self._feature_zarr_path = candidate
   - set self._state.feature_maps_path = candidate
   - refresh PCA + sim for current frame
   - set status: "Loaded cached <method>"
   - call _update_query_btn() → enables Query if queryable
5. If not valid:
   - clear self._feature_zarr_path = None
   - clear PCA + sim panes (set to None)
   - disable query btn
   - clear status
```

### Extractor selection

When multiple extractors have caches, auto-load whichever is **currently selected in the dropdown**. Switching the dropdown re-triggers discovery immediately.

### Run button behaviour (unchanged)

`▶ Run` still works as before: forces re-extraction even if cache exists. After run completes, `_feature_zarr_path` is updated (same path it would have discovered).

---

## Section 3: State Wiring Changes

New watches to add in `__init__`:

```python
self._state.param.watch(self._on_output_dir_change, "output_dir")
self._method_dd.param.watch(self._on_method_change, "value")
```

New handlers:

- `_on_output_dir_change` → calls `_try_discover_cache()`
- `_on_method_change` → calls `_try_discover_cache()`

Existing `_on_frames_zarr_change` gains a call to `_try_discover_cache()` at the end.

---

## Out of scope

- Moving extractor to sidebar (rejected: Run button coupling, no cross-tab need, pane state persists in memory across tab switches).
- Multiple simultaneous extractors (existing single-extractor model unchanged).
- Placeholder/grey images in empty panes (fixed height is sufficient).
