# Localization Dashboard — Design

**Date:** 2026-07-14
**Status:** Approved (pending user review of this document)
**Related:** 2026-07-08-loma-matcher-integration-design.md, 2026-06-03-splats-dashboard-design.md, 2026-07-09-localization-promotion-design.md

## Goal

Add a localization page to the splats dashboard: select a frame from an rgb_X field-camera video, localize it against an existing reconstruction's camera database, and visualize the result (match lines, camera pose on mesh, inlier distribution). Maximize reuse of existing package code — chain existing parts the way `run_pipeline` does; minimize new scaffolding and gcloud data transfer.

## Background / verified facts

- `CameraLocalizer` (collab_splats/localization/localizer.py) already supports the full loop: `from_feedforward` builds or loads a per-extractor feature DB inside `feedforward.zarr` (`local_features/<extractor>/reconstruction`), `localize(query_image, query_intrinsics)` returns a `LocalizationResult` whose pose is world-to-camera **in the reconstruction frame** (directly plottable next to feedforward cameras), and `add_localized_frame` persists localized frames to a separate `local_features/<extractor>/localized/` group that `load_index` merges on reload.
- `LocalizationResult` exposes `ref_frame_indices` + `inlier_mask` → per-reference-image inlier/total counts fall out of a histogram.
- The zarr store already saves sampled frames (`images`, chunked per frame) — reference pixels for match visualization are selectively pullable; no extra frame storage needed.
- Videos are never copied to `fieldwork_processed`; `run_config.yaml` records `video_ref` + `frame_indices`. Localization follows the same pattern.
- Registered extractors: `disk`, `xfeat`, `loma`, `loma-g`. Default for new DBs: `loma-g`.
- Dashboard is a single-page Panel app (`SplatsApp`, MaterialTemplate, PyVista `SplitViewer`), sessions/videos discovered via rclone (`collab-data` remote, `fieldwork_curated` / `fieldwork_processed`).

## Decisions (from brainstorming)

1. **No separate localization config file in gcloud.** The zarr store is the single source of truth. Provenance/metadata goes in zarr **group attrs** (tiny `zarr.json` files per group under zarr v3), which the dashboard can `rclone copyto` individually to populate the method dropdown without pulling the whole store.
2. **No zarr layout redesign.** Per-frame chunking already enables minimum-pull. Changes are strictly additive (attrs, one small array); existing stores stay valid.
3. **Query camera = different hardware from the reconstruction camera.** Intrinsics are estimated (feedforward single-frame prediction) and refined during PnP (`refine_focal_length`). This path is **experimental**: a follow-up validation task must compare predicted vs known intrinsics across cameras (metrics: focal error %, pose delta, inlier-count delta). A per-camera calibration file, when present, overrides the estimate.
4. **Localized frames grow the DB** (`localized/` group) with full source provenance; appending is on by default (checkbox in UI). `clear_localized_frames` already handles invalidation after BA/LC changes.
5. **Dashboard integration: tabbed shell, not routes.** Two page classes in one Panel session under `pn.Tabs(dynamic=True)` — instant no-reload switching, lazy build of the localize page.
6. **Cache by cost class.** CPU-side expensive loads (mesh polydata, zarr arrays, feature index) persist in a session-level `SceneCache` shared by both tabs. GPU models are freed after runs (`pytorch_gc`), except the extractor stays warm while the localize tab is active and the user iterates.
7. **Focus: localize frames from a *different* video (rgb_X field cameras) against the reconstruction.** Same-video localization works incidentally (the API is source-agnostic) but is not a design target now.

## Architecture

### Package additions (all additive)

**Zarr attrs provenance** (existing `localizer.py` save paths, ~20 lines):
- `local_features/<extractor>` group attrs: `{extractor, extractor_params, backbone, ba, lc, built_at}`.
- Each `localized/` frame: `{video_ref, session, camera, frame_idx, localized_at}` attrs.
- New small `frame_indices` array mapping reconstruction frames → source-video frame indices (enables future re-derivation and time-based coloring). Written at DB build time from `run_config.yaml` provenance.

**`collab_splats/localization/intrinsics.py`** (new, small):
- `estimate_intrinsics(frame: np.ndarray) -> np.ndarray` — single-frame feedforward inference, returns (3,3). Experimental; clear docstring warning.
- `CameraLocalizer.localize` gains `refine_focal_length: bool = False`, passed through to pycolmap pose refinement. This refinement optimizes the query pose only (6-DoF + focal); reference poses and 3D points are fixed — the DB is never modified by localization.

**`collab_splats/localization/viz.py`** (refactor, no new function for matches):
- `plot_correspondences` gains `ref_idx: int | None = None` (override best-frame selection) and returns the `Figure`. Existing callers (tutorial notebook) unaffected — rendering behavior unchanged.
- New `plot_inlier_distribution(loc, frame_times=None) -> Figure`: bars = inliers per reference image (x = image number, y = inliers). Bars colored viridis by reference-frame time (same colormap as the 3D camera plot, so the two panels cross-read). Per-bar tick marks each image's total correspondence count; when totals are uniform across images, collapse to a single dashed horizontal line. Localized-frame references visually distinct from reconstruction frames.

**`collab_splats/dashboard/config.py`** (existing file):
- `LocalizationConfig` dataclass (~6 fields): `extractor` (default `"loma-g"`), `top_k_viz`, `append_to_db`, `refine_focal_length`, `calibration_path` (optional override), `max_pairs`. UI/call state only — **not** serialized to gcloud; localization provenance lives in zarr attrs. `RunConfig` is untouched (it is per-reconstruction provenance; localization runs are many-per-reconstruction).

**`collab_splats/dashboard/pipeline.py`** (existing file):
- `run_localization(recon_ref, query_video, frame_idx, config: LocalizationConfig, ...)` next to `run_pipeline`. Reuses `OperationLog`, gpu_worker, push helpers. Each step below reports explicit progress via `OperationLog.update_progress(pct, msg)` (same mechanism as `run_pipeline`), and `LocalizePage` renders the same progress strip component as the reconstruction page — the user always sees which step is running (pulling, building/loading DB, extracting frame, estimating intrinsics, localizing, appending, pushing). Steps:
  1. Selective pull from `fieldwork_processed` (see Data movement).
  2. Load/build `CameraLocalizer` via `from_feedforward` (build-on-demand only when the user explicitly picked a method with no existing DB — UI warns about GPU cost).
  3. Pull query mp4 from `fieldwork_curated`, extract frame `frame_idx`.
  4. Intrinsics: calibration file if configured, else `estimate_intrinsics` + `refine_focal_length=True`.
  5. `localize()` → `LocalizationResult`.
  6. If `append_to_db`: `add_localized_frame` with provenance attrs.
  7. Incremental push: new `localized/` chunks + updated attrs only.

### Dashboard structure

- **`dashboard/shell.py`** (new, ~50 lines): one `MaterialTemplate` hosting `pn.Tabs(dynamic=True)` with `SplatsPage` and `LocalizePage`; watcher on `Tabs.active` swaps sidebar contents; owns the shared `SceneCache`.
- **`SplatsApp` → `SplatsPage`**: mechanical split of `view()` into sidebar/main parts; behavior unchanged.
- **`dashboard/localize.py`** (new — the one substantial new file): `LocalizePage`.
- **`SceneCache`**: session-level dict keyed by `(session, video)` holding mesh polydata, zarr arrays (points/extrinsics/intrinsics), CPU feature index. Session selection on either page pre-warms the other.
- **GPU lifecycle**: runs execute in the existing gpu_worker pattern; `pytorch_gc()` after each run; extractor model kept warm across consecutive localize runs, released on tab switch.

### LocalizePage layout

**Sidebar:** session dropdown → camera dropdown (rgb_X folders only; thermal deferred) → video dropdown (mp4s via rclone listing of `fieldwork_curated/<YYYY_MM_DD>-session_XXXX/rgb_X/`) → frame slider (default 0) → method dropdown (populated from remotely pulled zarr attrs; method with an existing DB preselected; selecting a method without a DB shows a "will build DB — GPU cost" warning; default `loma-g` when no DB exists) → "append to DB" checkbox (default on) → Run button.

**Progress strip:** same component as the reconstruction page, driven by `OperationLog` — step-labeled progress bar for every run (including DB build-on-demand and background push).

**Main area:**
- Pre-run: left = selected query frame; right = mesh with reconstruction cameras (viridis by time; every 3rd camera when N > ~60, with a corner annotation stating the subsampling); bottom empty.
- Post-run: left = stacked match-pair figures, top-k reference frames sorted by inlier count (each via `plot_correspondences(ref_idx=...)`, green inlier / red outlier lines); right = mesh + red localized camera added to the viridis set; bottom = `plot_inlier_distribution` + one-line summary stats (n_inliers / n_correspondences, inlier ratio, estimated focal + source, runtime).

### Data movement (minimum-transfer)

Pull (selective rclone from `fieldwork_processed`):

| Data | Purpose | When |
|---|---|---|
| zarr group `zarr.json` attrs (few KB) | method dropdown | on session select |
| `local_features/<extractor>/…` groups | matching | on run |
| extrinsics / intrinsics / image_paths / points arrays | pose + assignments | on run |
| top-k `images` chunks only | match-pair viz | after localize |
| `mesh.ply` | right panel | on session select (cached) |

Pull from `fieldwork_curated`: the selected query mp4 only.
Push: new `localized/` chunks + updated attrs (incremental sync). Videos are never re-stored; dense `depth`/`world_points`/`confidence` arrays are never pulled.

Push destination — localized data lives inside the reconstruction's store, under the reconstruction's processed folder:

```
fieldwork_processed/reconstruction/<session>/<video_stem>/feedforward.zarr/
    local_features/<extractor>/{reconstruction/, localized/}
```

No parallel `YYYY_MM_DD-session_XXXX/rgb_X/` tree is created in `fieldwork_processed`. Rationale: a localized pose is only meaningful in that reconstruction's coordinate frame (the same rgb_X frame localized against two reconstructions yields two unrelated poses), and `load_index` requires both groups in one store. Source identity is preserved per frame via `{session, camera, video_ref, frame_idx}` attrs. A per-session manifest for cross-reconstruction lookup is deferred until a consumer exists.

## Error handling

- PnP failure (`pose is None`): summary panel reports failure with n_correspondences; no camera added; match-pair plot still shown (outliers only) for diagnosis.
- Zero inliers / too few correspondences: same degraded-viz path; no DB append.
- Missing DB for selected method: explicit build-on-demand confirmation, never silent GPU work.
- rclone pull/push failures: surfaced via `OperationLog` (existing pattern); push failure is non-fatal (local result remains viewable).
- Intrinsics estimate wildly off (focal outside plausible range): warn in summary, still attempt localization.

## Testing

Flat pytest functions (tests mirror package tree):
- attrs round-trip: write provenance attrs, reload via `load_index`, assert preserved.
- `plot_inlier_distribution` + refactored `plot_correspondences`: figure smoke tests on synthetic `LocalizationResult` (uniform-totals dashed-line branch and per-bar tick branch both covered).
- `run_localization` chaining on a tiny synthetic scene (mocked rclone, no GPU): step order, config plumbing, append-on/off.
- method-dropdown preselect logic against fake remote attrs (existing DB → preselected; none → `loma-g` default).
- `estimate_intrinsics`: shape/range sanity with a mocked backbone.
- Dashboard interaction (tabs, sidebar swap): manual verification, consistent with existing dashboard testing practice.

## Deferred / follow-ups

- Intrinsics-estimation validation harness (known-intrinsics cameras; focal error %, pose delta, inlier delta) — required before trusting estimated-intrinsics poses.
- thermal_X cameras.
- Same-video frame localization as a first-class flow (near-duplicate viewpoint handling for DB growth).
- Per-camera calibration workflow/storage convention (only the override hook ships now).
