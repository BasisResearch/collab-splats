# Localize Page Display Fix + DB Browse — Design

**Date:** 2026-07-18
**Status:** Approved
**Scope:** `collab_splats/dashboard/localize.py` (+ small additions to `pipeline.py` / `sources.py` if needed for browse data loading)

## Problem

The Localize page renders nothing in practice:

1. **Frame preview never appears.** Selecting a query video fetches the full video (op-log shows rclone progress and completion) and decodes frame 0, but the left panel stays on the placeholder. Confirmed symptom: fetch completes, pane stays blank.
2. **Run results never appear.** A localization run completes end-to-end (op-log shows `localize: N/M inliers`, `building result figures` done), but all three result panels (match figures, inlier distribution, 3D scene) stay empty.
3. **Existing DBs are invisible.** When a scene already has a feature DB with localized frames, nothing is shown until a new Run — the stored poses and frames should be browsable immediately.
4. **Unclear DB reuse semantics.** (Answered by code reading, no behavior change needed: the DB is built once per extractor and reused; a new query image only extracts query features and matches — `append_to_db` appends incrementally. Rebuild happens only when the extractor's zarr group is missing.)

## Root cause (issues 1–2)

Both symptoms are the same bug class in the display layer:

- All result panes (`_frame_pane`, `_matches_col`, `_dist_pane`, `_stats`, VTK pane) are constructed in `__init__`, before any server document exists. `main()` embeds these long-lived objects. A re-render or tab switch leaves panes bound to a detached/stale Bokeh document, so subsequent updates are silently dropped. Two prior point-fixes ("fresh pane per frame", "fixed width image pane") patched instances of this class without eliminating it.
- Secondary: `_matches_col` (`stretch_width`) sits beside the VTK pane inside a `stretch_both` Row — flex layout can collapse the column to zero width.

## Design

### 1. Display layer restructure (fixes issues 1 + 2)

Split state from rendering:

- **State on the page object:** last preview frame, last run output (`out`, `figs`, `mesh`), browse data. Nothing display-bound survives from `__init__`.
- **Panes built per document:** `main()` constructs all result panes fresh each time it is called. Watchers that fire before `main()` (e.g. `_show_frame` via `_on_query_video` during `__init__`) write to pending state instead of touching panes; `main()` renders pending state when it builds.
- **Render methods** (`_show_frame`, `_render_result`, `_render_scene`, browse render) resolve the current document's panes and keep the existing fresh-pane-per-update pattern. `pn.pane.Matplotlib` is retained (no PNG conversion).
- **Left column fixed width** (`_PREVIEW_MAX_W` + padding) so it cannot flex-collapse beside the VTK pane.
- **Re-render restores content:** because the last run output / preview frame lives on page state, switching away and back re-renders it instead of showing a blank page.

### 2. First-frame preview policy (issue 1)

Full-video download stays — the video is needed for slider scrubbing and for Run's `extract_frame` at an arbitrary index, and it is cached locally after the first pull. No partial-fetch (`rclone cat` + ffmpeg pipe) path: it adds a moov-atom failure mode for marginal gain. The preview appears when the fetch completes (guaranteed by section 1); rclone transfer progress already streams to the op-log.

### 3. DB browse on scene select (issue 3)

When a scene video is selected, alongside the existing feature-DB listing, a background **non-GPU** job:

1. `pull_processed` the minimal set if `feedforward.zarr` is absent locally (existing excludes already skip dense arrays; progress in op-log).
2. Load reconstruction extrinsics, `localized/` group poses for the selected extractor, and thumbnails from `localized_frames/`.
3. Render immediately (no Run required):
   - **3D pane:** mesh + viridis reconstruction cameras + red localized-frame cameras from stored poses.
   - **Left column:** thumbnail strip of localized-frame images.

Notes:
- Match/correspondence plots are **not** recomputed or persisted for browse — matches are run-time artifacts only (per scoping decision).
- Cached via `SceneCache` (existing `mesh` kind + new `browse` kind).
- A Run's output replaces the browse view; re-selecting the scene re-renders browse.
- Selecting a scene now triggers the minimal pull (user-approved); cached thereafter.

### 4. DB reuse visibility (issue 4)

No pipeline change. UI copy only: `db_note` gains a localized-frame count from browse data, e.g. "DB exists — will reuse (7 localized frames)", making incremental behavior visible.

### 5. Testing

- **Unit:** pending-state render on `main()` build; browse-data loader against a small zarr fixture; pane construction is per-document (two `main()` calls yield distinct pane objects).
- **Gate:** `python -m collab_splats.dashboard --smoke` must print SMOKE PASS before commit.
- **Manual browser checklist:** select query video → frame 0 appears after fetch; Run → all three panels fill; select scene with existing DB → browse view (3D + thumbnails) appears without a Run; tab away and back → last content re-renders.

## Out of scope

- Partial/ranged video fetch for instant first-frame preview.
- Persisting or recomputing match figures for past runs.
- Any change to localizer/zarr DB semantics.
