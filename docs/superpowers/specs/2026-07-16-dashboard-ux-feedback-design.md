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

- `GpuWorker` gains a busy-transition callback: `on_busy(busy: bool, label: str)`, fired when a job starts/ends, marshalled onto the UI thread via the job's doc (`add_next_tick_callback`).
- `DashboardShell` registers a single handler that fans out to both pages.
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

## Error handling

- Deferred page build failure → error message rendered into the holder + logged; tab remains usable.
- Busy callback always fires the `False` transition in a `finally` so a crashed job never leaves the UI locked.
- Frame extract and step failures logged as warnings, never raised into the UI thread.

## Testing

- Unit: `OperationLog.step` timing/format/exception path; debounce latest-wins logic; busy-state fan-out disables/enables the right widgets on both pages (existing dashboard test patterns, mocked doc callbacks).
- Manual browser smoke: spinner paints on first Localize activation; buttons lock across tabs during a run; slider updates frame; log shows step timings during a scene load. Extends the already-owed manual smoke checklist.

## Out of scope

- Persistent status bar / toast notifications (approach B — possible later polish).
- Job cancellation, queue management beyond the single-flight lock.
- Any change to `run_localization` computation itself.
