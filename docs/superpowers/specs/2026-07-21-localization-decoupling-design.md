# Design: decouple localization (localizer + viz) from the frames.zarr artifact

**Date:** 2026-07-21 · **Branch:** `refactor/cu121-uv-migration` (main repo, no worktree; concurrent sessions active)

## Problem

The frame-store refactor (Tasks 7–8) fixed a real bug — `FeedforwardResult.image_paths`
pointed at a deleted temp export dir — but did it by threading a `frames_zarr` parameter
through the *general* localization + viz API. That couples general processing code to a
pipeline storage artifact. It must be undone.

## Architectural principle (from the user, non-negotiable)

General functions operate over **core objects** (an image is an `np.ndarray` RGB). Pipeline
**artifacts** (`frames.zarr` / `FrameStore`) provide efficient storage/access and are adapted
to core objects **at the pipeline boundary** (the caller), never threaded into general functions.

## Hard constraints

- **No new classes.** No `ImageSource` Protocol, no adapter classes. Pixel sources are plain
  lazy generators built inline at the call site.
- **No new IO helper.** `utils/image.py::open_image` already coerces path→image and handles
  RGB conversion; reuse it. Do not add `load_rgb`, do not add a `utils/io` module.
- **No re-materialization.** Do NOT export `frames.zarr` → temp JPGs, do NOT eager-load all
  keyframes into RAM. Read one image at a time (chunked partial-read).
- **Zero IO in the localization module.** After this change the localizer and viz do no image
  reading themselves — all pixels arrive as arrays from the caller.
- **Minimal diff. Reuse what exists.**

## Code surface (verified 2026-07-21)

- `localization/extractors.py`: `BaseLocalExtractor.extract(self, image: np.ndarray) -> LocalFeatures`
  is **already array-based**. NO change.
- `localization/localizer.py`: the coupling. Build loops at `__init__` (~:203-236),
  `update_index` (~:504-518), and the `from_feedforward` classmethod (~:738-824). `localize`
  (query, ~:826-845) is already array-based → NO change.
- `localization/viz.py::plot_correspondences`: leaked `frames_zarr` param (~:27, :104-111).
- `utils/image.py::open_image`: existing path/array/PIL → PIL coercion. Reused at path sources.
- `semantics/features/base.py::extract_and_cache_from_zarr` (:176): existing lazy store-iterating
  extractor. Track B routes through it.

### `image_paths` is dual-role today

`image_paths` currently serves as BOTH pixel source AND persisted identity: it is stored on the
localizer, written to the zarr attr `image_paths`, used for the staleness check, and displayed by
the dashboard (basename only). In the wrapper DB-build path the values are already *synthetic*
(`frame_000123.jpg`, no real file) with pixels coming from the store. Decoupling splits the two
roles: `images` (pixels, lazy, ephemeral) + `ids` (labels, persisted). This is the only reason
the pair exists.

## Track A — localization decoupling

### A1. `CameraLocalizer.__init__`

Replace params `image_paths: list` + `frames_zarr` with:
- `images: Iterable[np.ndarray]` — lazy RGB arrays (pixels)
- `ids: list[str]` — stable per-frame labels (persisted)

Build loop becomes pure processing:
```python
for img, fid in zip(images, ids):
    feats = self._extractor.extract(img)
```
Remove from the module: `cv2`, `FrameStore` import, `FrameStore.frame_idx_from_path`, and the
store-vs-`cv2.imread` branch. `progress_callback` / tqdm behaviour preserved (iterate `ids` for
count/labels). Colour: `images` are RGB (`FrameStore.image` returns RGB; path sources convert via
`open_image(...).convert("RGB")`).

- Rename internal `self._image_paths` → `self._ids` (list[str]).
- Persist ids under the **existing zarr attr key `"image_paths"`** (cache-format compat).
- Staleness check compares `ids` lists.
- `image_paths` property (consumed by dashboard as basenames) returns `self._ids` unchanged in
  behaviour — `frame_000123.jpg`-shaped strings still work.

### A2. `CameraLocalizer.update_index`

Same treatment: `(new_images: Iterable[np.ndarray], new_ids: list[str], ...)`. Drop `frames_zarr`
and the imread branch. Append `new_ids` to in-memory ids and the zarr `image_paths` attr as today.

### A3. `CameraLocalizer.from_feedforward` (boundary adapter, stays in module → NO `frames_zarr`)

Signature `(result, images, ids, ...)`. Caller builds the aligned `(images, ids)` pair. `result`
is used only for geometry (`points`/`extrinsics`/`intrinsics`), `_zarr_path`, and the staleness
compare. On a cache **hit** the `images` genexpr is never iterated → zero cost. On a miss it
constructs `cls(..., images=images, ids=ids, ...)`.

### A4. `viz.plot_correspondences`

Drop `frames_zarr` and the `FrameStore`/`frame_idx_from_path` branch. Replace `image_paths` +
index logic with a single `ref_image: np.ndarray` param — the caller passes the one already-resolved
reference frame. Module loses its `FrameStore`/`cv2` coupling.

## Track A — boundary callers (glue; `frames_zarr` allowed here)

Build the pixel source inline — a plain genexpr, no class:
```python
# from frames.zarr (one partial-read per step; no re-write, no full RAM):
images = (store.image_by_frame_idx(fi) for fi in frame_indices)
ids    = [f"frame_{fi:06d}.jpg" for fi in frame_indices]     # aligned
# from a dir / list of paths (general standalone use):
images = (np.asarray(open_image(p).convert("RGB")) for p in paths)
ids    = [p.name for p in paths]
```

- **`wrapper/reconstructor.py::_build_localization_db`**: open store, build the aligned
  `(images, ids)` from `frame_indices`, pass both to `from_feedforward`. `frames_zarr` stays here.
- **`dashboard/pipeline.py::_build_localizer`** and **`dashboard/localize.py::_build_result_figures`**:
  reconstruction ref frames → `store.image_by_frame_idx`; `localized/` frames (not in the store) →
  `np.asarray(open_image(p).convert("RGB"))`. Pass the resolved ref array into `plot_correspondences`.

**Sanity gate:** `grep -rn "frames_zarr" collab_splats/localization` must be EMPTY afterward.
`frames_zarr` may remain in wrapper/dashboard glue — correct.

## Track B — semantics (approved: do the swap)

`wrapper/reconstructor.py::_extract_2d_features` (~:179-195) currently exports `frames.zarr` →
temp JPGs → `extract_and_cache(paths)` — exactly the re-materialization the principle forbids.
Replace with a direct call to the existing `extractor.extract_and_cache_from_zarr(frames_zarr,
cache_dir)` and delete the temp-export bridge. No ids — semantics features are position-indexed
(0..N-1), no staleness-by-path, no per-frame label. ~10-line diff, reuses existing code.

## Verification

- **Parity (localization is parity-sensitive — see `project_lc_parity_harness`):** before/after
  equivalence on a small fixture — the local-features cache built via the new `(images, ids)` path
  must match the pre-refactor cache (same keypoints/descriptors). frames.zarr yields *lossless*
  pixels vs the old JPG dir → expect a match or improvement; verify extractor output matches.
- **Tests:** update Task-7/8 tests that asserted `frames_zarr` params to the decoupled API
  (`tests/localization`, `tests/preproc/test_viz.py`, `tests/dashboard/test_run_localization.py`).
- **Run:** `/opt/venv/reconstruction/bin/python -m pytest tests/localization tests/preproc/test_viz.py tests/dashboard tests/wrapper -q`
  and dashboard smoke `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke` (must print SMOKE PASS).

## Out of scope / do not touch

- Do NOT reintroduce a persistent `images/` dir.
- Do NOT modify `geometry/loop_closure/merge.py` (concurrent-session file).
- Extractors (`extractors.py`) unchanged — already array-based.

## Implementation principles

Reuse existing funcs (`open_image`, `FrameStore.image_by_frame_idx`, `extract_and_cache_from_zarr`).
Delete the dead coupling this obsoletes (temp-export bridge, imread branches, `frames_zarr` params
in the general API). No new abstractions. Commit logically (conventional commits).
