# SemanticsPane — Design Spec

**Phase:** 2 (depends on Phase 1 skeleton)
**Date:** 2026-05-28
**Scope:** Tab 2 implementation + AppState/PreprocessPane frame memory refactor

---

## Overview

Two coupled changes:

1. **AppState refactor** — replace `state.frames: list[np.ndarray]` (all frames in RAM) with `state.frames_zarr_path: Path` (lazy zarr on disk). PreprocessPane writes the zarr; all downstream panes load frames on demand.
2. **SemanticsPane** — 4-column feature extraction viewer. Original frame (col 0) + up to 3 independently-selectable extractor columns. Frame slider drives reactive display across all columns. Text query bar for queryable extractors.

---

## Section 1: AppState + PreprocessPane Refactor

### AppState changes

```python
# Remove:
frames = param.List(default=[])

# Add:
frames_zarr_path = param.Parameter(default=None)  # Path to output_dir/frames.zarr
```

`frames_zarr_path` is set by PreprocessPane after writing the zarr. Downstream panes (`SemanticsPane`, etc.) gate on this param being non-None.

### frames.zarr layout

```
output_dir/frames.zarr
  frames: (N, H, W, 3)  uint8
  chunks: (1, H, W, 3)  — one chunk per frame for cheap random access
  compressor: Blosc(cname="lz4", clevel=5)
  attrs: {n_frames: N, height: H, width: W}
```

### PreprocessPane `_run_extraction` change

End of background thread (after thumbnail rendering, before thread exit):

1. Open zarr store at `output_dir/frames.zarr` (mode="w")
2. Write frames array with Blosc compression
3. Set `state.frames_zarr_path = zarr_path`
4. Drop in-memory frame list (local var goes out of scope)

Thumbnail rendering (`_frames_to_thumbnails`) happens before zarr write — still uses in-memory frames. No change to thumbnail logic.

---

## Section 2: BaseFeatureExtractor — `extract_and_cache_from_zarr`

Add to `BaseFeatureExtractor` in `collab_splats/semantics/features/base.py`:

```python
def extract_and_cache_from_zarr(
    self,
    frames_zarr_path: Path,
    cache_dir: Path,
    batch_size: int = 1,
    skip_existing: bool = True,
) -> Path:
```

**Behavior:**
- Output zarr path: `cache_dir / f"{self.name}.zarr"` — identical layout to `extract_and_cache`
- Cache validation: same skip logic (extractor name + n_frames match)
- Iteration: opens `frames_zarr_path`, loads one chunk at a time (`zarr[i]` → PIL.Image → batch), calls `forward()`, writes to output zarr
- Never holds all frames in RAM — one batch at a time

Output zarr layout (unchanged from `extract_and_cache`):
```
{name}.zarr
  features: (N, D, H_p, W_p)  float32
  chunks: (1, D, H_p, W_p)
  attrs: {extractor: name, n_frames: N}
```

---

## Section 3: SemanticsPane

### File

`collab_splats/dashboard/panes/semantics.py`

### Layout

```
[ Frame Selector: slider  ◀ ▶ ]
[ Original frame ] [ Extractor 1 ] [ Extractor 2 ] [ Extractor 3 ]
[ (from zarr)    ] [ dropdown+Run ] [ dropdown+Run ] [  +  add    ]
[ Text Query Bar (visible when ≥1 queryable column active) ]
```

4-column grid using `pn.GridSpec` or `pn.Row`. Initial state: 1 active extractor column + 2 "+" slots.

### ExtractorColumn (inner class or dataclass)

Per-column state:
- `method_dd` — `pn.widgets.Select`, options from `BaseFeatureExtractor._registry.keys()`
- `run_btn` — `pn.widgets.Button`
- `status_html` — `pn.pane.HTML` (idle / running... / done / error)
- `image_pane` — `pn.pane.PNG` (displays PCA RGB or heatmap)
- `_extractor: BaseFeatureExtractor | None` — instantiated on Run, released on dropdown change
- `_feature_zarr_path: Path | None` — set on extraction completion
- `_thread: threading.Thread | None`

### Frame Selector

- `_frame_slider`: `pn.widgets.IntSlider(start=0, end=N-1)` — N from `zarr.attrs["n_frames"]`
- `_prev_btn`, `_next_btn`: `pn.widgets.Button`
- On slider change → `_refresh_display(frame_idx)`:
  - Load `frames_zarr[frame_idx]` → display in col 0 image pane
  - For each column with `_feature_zarr_path` set → load `features_zarr[frame_idx]` → PCA → display

### Run flow (per column)

1. Read method from dropdown
2. If column has existing `_extractor`: `del self._extractor; torch.cuda.empty_cache()`
3. Clear `_feature_zarr_path`, reset status → "running..."
4. Instantiate extractor: `extractor = BaseFeatureExtractor.get(method)()`
5. Launch daemon thread:
   - Calls `extractor.extract_and_cache_from_zarr(state.frames_zarr_path, output_dir/features/{method}/)`
   - On success: set `_feature_zarr_path`, status → "done", call `_refresh_display(current_frame)`
   - Sets `state.feature_maps_path = feature_zarr_path`
   - On error: status → "error: {msg}", log to op_log

### Extractor memory management

On **dropdown change**:
- `del self._extractor; torch.cuda.empty_cache()`
- Clear `_feature_zarr_path`
- Reset `image_pane` to blank
- Status → idle

Extractor is only instantiated at Run click (lazy). At most 3 extractor instances live simultaneously (one per active column).

### Text Query Bar

Visibility: shown when ≥1 column has `_extractor` instance of `BaseQueryableExtractor` AND `_feature_zarr_path` is set.

Controls:
- `_query_input`: `pn.widgets.TextInput(placeholder="chair, table, ...")`
- `_query_btn`: `pn.widgets.Button(name="Query")`

On Query click (background thread):
1. For each column where `isinstance(col._extractor, BaseQueryableExtractor)` and `col._feature_zarr_path` is set:
   - Load `features_zarr[frame_idx]` → torch.Tensor `(D, H_p, W_p)`
   - Call `col._extractor.score_queries(features, positive=[query_text])`
   - Returns `(H_p, W_p)` score map → apply viridis colormap → uint8 RGB
   - Update `col.image_pane`

### Enabled gate

Pane widgets disabled until `state.frames_zarr_path` is set. `SemanticsPane.__init__` registers `state.param.watch(self._on_frames_zarr_ready, "frames_zarr_path")`.

### AppState writes

`state.feature_maps_path` — set to the last completed extractor's zarr path (last-wins).

---

## Data Flow

```
PreprocessPane
  → writes output_dir/frames.zarr
  → state.frames_zarr_path = path

SemanticsPane (on Run)
  → reads frames.zarr lazily via extract_and_cache_from_zarr
  → writes output_dir/features/{name}/{name}.zarr
  → state.feature_maps_path = feature_zarr_path

SemanticsPane (on slider / query)
  → reads frames.zarr[i] for original frame display
  → reads features/{name}.zarr[i] for feature display
```

---

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/dashboard/state.py` | Replace `frames` param with `frames_zarr_path` |
| `collab_splats/dashboard/panes/preprocess.py` | Write frames.zarr at end of `_run_extraction`; set `frames_zarr_path` |
| `collab_splats/semantics/features/base.py` | Add `extract_and_cache_from_zarr` to `BaseFeatureExtractor` |
| `collab_splats/dashboard/panes/semantics.py` | New file — `SemanticsPane` |
| `collab_splats/dashboard/app.py` | Wire `SemanticsPane` into tab 2 (replace placeholder) |
| `collab_splats/dashboard/__init__.py` | Export `SemanticsPane` |

---

## Out of Scope

- Multi-frame batch query (query always operates on selected frame only)
- Debiasing toggle in UI (debias not called by default in SemanticsPane — extractor default applies)
- Saving query heatmaps to disk
- Column reordering
