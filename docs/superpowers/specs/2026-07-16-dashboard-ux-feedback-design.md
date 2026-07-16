# Dashboard UX Feedback Pass — Design

**Date:** 2026-07-16
**Status:** Approved
**Scope:** `collab_splats/dashboard/` — shell, app, localize, gpu_worker, operation_log, pipeline

## Problem

Four user-facing feedback gaps in the dashboard:

1. **Unresponsive tab switching.** Switching Splats ↔ Localize gives no visual response; the Localize page is built lazily and synchronously on the UI thread on first activation (`shell.py:_on_tab`), so the page appears blank/frozen until the build completes.
2. **Accidental concurrent processes.** No global busy state. Each page only disables its own run button, so a user who thinks nothing is happening can trigger a second heavy job from the other tab. All heavy jobs funnel through the single `GpuWorker` thread, so duplicates queue silently.
3. **Localize frame preview dead.** `frame_slider` has no watcher. The frame pane shows frame 0 once when a video is selected; moving the slider produces no visible response until a run uses the value.
4. **Opaque scene loading.** The operation log shows one static line (e.g. "loading c0043") for the entire multi-step load (rclone pull → zarr read → mesh read → render). Localize's scene ply read runs synchronously on the UI thread with no feedback at all (`localize.py:_render_scene`).

## Design

### 1. Deferred Localize build with immediate spinner (`shell.py`)

On first Localize tab activation, `_on_tab`:

- Immediately sets the localize holder to a `pn.indicators.LoadingSpinner` + "Building Localize page…" label — browser paints this before any heavy work.
- Defers the heavy `self._localize.main()` build one tick via `pn.state.curdoc.add_next_tick_callback`, then swaps the built page into the holder.
- Logs `building Localize page… / done (Xs)` to the operation log.

Tab switching itself never blocks. Subsequent activations are unchanged (page cached).

### 2. Global busy lock (`gpu_worker.py`, `shell.py`, `app.py`, `localize.py`)

- Busy state propagates by polling, not push: each page's existing 300 ms op-log poll also mirrors the shared `GpuWorker.busy` flag onto its widgets. (One worker serves many browser sessions; pushing to per-session widgets from a shared object would leak dead sessions — polling matches the op-log architecture.) Pages still disable immediately in their own click handlers.
- `SplatsApp._set_busy` (exists) and a new `LocalizePage.set_busy(flag, label)` disable all run/load/force buttons on their page and show the current-op label near the buttons.
- Because every heavy job goes through `GpuWorker.submit`, one lock covers scene loads, reconstruction runs, and localization runs across both tabs. UI-triggered concurrent jobs become impossible; if a job is somehow submitted while busy, the log shows `queued behind: <op>`.

### 3. Debounced live frame preview (`localize.py`)

- Watcher on `frame_slider` value with ~300 ms debounce (reset timer on each change).
- On fire: extract frame N from the selected query video off the UI thread (reuse the ffmpeg extract path from `_on_query_video` / `_show_frame`), display in `_frame_pane`, log `frame 42 loaded (0.4s)`.
- **Latest-wins:** each extract carries a token; results from superseded slider positions are dropped.
- Disabled while a localization run is busy (guarded by the global busy state).
- Extract failure → warning line in the log window; previous frame stays displayed.

### 4. Step-level load logging (`operation_log.py`, `app.py`, `localize.py`, `pipeline.py`)

- `OperationLog.step(label)` context manager: logs `label…` on enter and `label done (12.3s)` on exit (monotonic clock, thread-safe, exception-safe — failures log `label failed (Xs): <err>`).
- Wrap the load and run paths:
  - `app.py:_load_outputs` job: rclone pull (keeps existing % streaming), `load_zarr` read, cache store.
  - `localize.py` scene load: ply/mesh read — **moved off the UI thread onto `GpuWorker`** (fixes the freeze and gives it a logged step); render marshalled back to the doc.
  - `pipeline.py:run_localization` major stages: feature extraction, retrieval, matching, PnP.
- The shared log window (both tabs poll `OperationLog.render_html`) then always shows the current step inside any long operation, with elapsed times.

### 5. Remaining UI-thread blockers (full-code audit)

Beyond the known items above, these run blocking work on the Bokeh IOLoop and freeze the page. All move to the `GpuWorker` / `run_off_loop`, each wrapped in an `OperationLog.step`:

- **`app.py:425` `_on_run`** — `source.has_processed()` (blocking rclone network list) runs synchronously inside the Run/Force click on the cold path. Move into the worker `job()`. *(Highest impact — this is the "click Run, page freezes" case.)*
- **`app.py:333/356` `_on_video` watcher** — `get_video_info()` (ffprobe subprocess) runs inline when the video is already local. Always run off-loop.
- **`localize.py:422,442` `_render_result`** — matplotlib inlier-distribution + correspondence figures built on the IOLoop after each run. Build figures in the worker; hand finished figs to the doc.
- **`app.py:545` → `viewer.py:145-309` `SplitViewer.load`** — mesh `pv.read`, `vertex_features.npy` load, view transform over the full cloud, and PolyData build for up to 500k points all on the IOLoop. Do geometry prep in the worker; only the final pane sync on the loop.
- **`shell.py:51` `_on_tab`** — `release_gpu()` → `pytorch_gc()` (CUDA sync) runs synchronously on tab switch. Offload to `GpuWorker`.

### 6. Surface silent failures and silent states in the log window

Today many failures land only in the server console; the page just looks dead. All of the following get an op-log line:

- **`async_utils.py:29-31` `run_off_loop`** swallows every fetch exception — all dropdown populations (video list, has-processed, scene-video, camera, query-video) fail silently to empty. Add an `on_error` path that calls `op_log.error_op`.
- **Listing feedback:** every rclone listing/populate path (`app.py:245,303,364`; `localize.py:211,235,247,282,294,306`) logs `listing <what>…` / `done (Xs)` so empty dropdowns are visibly "in flight", not broken.
- **`app.py:420-422` `_on_run`** returns silently when session/video unselected — mirror localize's `error_op` message.
- **`viewer.py:365-371` `score_query`** silently returns plain RGB when scene has no semantic features — log "query: no semantic features for this scene".
- **`viewer.py:268-269`** mesh view mode with no mesh silently falls back to pointcloud; `_status` is set but never shown — push `_status` strings to the op log.
- **`viewer.py:176-178`** feature-lift failure logged to console only — surface to op log.
- **`app.py:177` `max_display_points`** change silently does nothing until scene reselect — log "will apply on next scene load" (or trigger re-render).

### 7. Busy lock covers all mutating widgets

Extends section 2: `_set_busy` currently covers only the three Splats action buttons. While busy, also disable: `view_mode` radio + `normalize_view` checkbox (toggling mid-load fires `set_mode` on a half-loaded viewer and queues another job — `app.py:174,234,566`), and Localize's scene/query selectors (`localize.py:395` disables only `run_btn`). If a job is submitted while busy anyway, log `queued behind: <op>`.

### 8. Efficiency fixes

- **Duplicate video fetch** — `app.py:320+370`: `_update_max_frames_bound` runs twice per video select (`_on_video` then `_autoload_current`); on a remote video the second call can start a duplicate download while the first is mid-flight. Drop one call site.
- **Mesh ply read 3× per scene** — `viewer.py:145` (`pv.read`), `viewer.py:199` (`o3d` re-read for features), `localize.py:474` (`pv.read` again). Cache PolyData in the shared `SceneCache`; derive o3d vertices from it.
- **Eager mesh load** — `viewer.py:145-147` loads mesh + vertex features on every `load()` even in default pointcloud mode. Defer to `set_mode("mesh")`.
- **Idle polling** — two independent 300 ms `render_html()` polls (`app.py:636`, `localize.py:191`) run continuously once Localize is opened, even when idle. Single shared poll; pause when no op running and no recent lines.
- **`SceneCache` mesh eviction** — `localize.py:58`: `"mesh"` entries never evicted; add keep-N LRU like `_loaded_order`. Also purge expired TTL entries in `sources.py:76 _listing_cache`.
- **Frame preview thumbnail** — `localize.py:340-345` pushes a full-resolution base64 frame into the doc; downscale to display width first (also makes the new debounced preview cheaper).
- **Verify ffmpeg seek** — confirm `extract_frame` uses input-seek (`-ss` before `-i`) and `get_video_info` reads header metadata, not full decode; fix in `preproc` if not.
- **Deferred (out of this pass):** `pipeline.py:216-217` writes frames both as `frames.zarr` (in `PULL_EXCLUDES`, never read back) and `frames/*.jpg`, wasting disk + upload — output-format change, handle in its own decision doc.

## Error handling

- Deferred page build failure → error message rendered into the holder + logged; tab remains usable.
- Busy callback always fires the `False` transition in a `finally` so a crashed job never leaves the UI locked.
- Frame extract and step failures logged as warnings, never raised into the UI thread.

## Testing

- Unit: `OperationLog.step` timing/format/exception path; debounce latest-wins logic; busy-state fan-out disables/enables the right widgets on both pages (existing dashboard test patterns, mocked doc callbacks); `run_off_loop` on_error surfaces to op log; SceneCache mesh eviction; single `_update_max_frames_bound` call per video select.
- Manual browser smoke: spinner paints on first Localize activation; buttons lock across tabs during a run; slider updates frame; log shows step timings during a scene load. Extends the already-owed manual smoke checklist.

## Out of scope

- Persistent status bar / toast notifications (approach B — possible later polish).
- Job cancellation, queue management beyond the single-flight lock.
- Any change to `run_localization` computation itself.
