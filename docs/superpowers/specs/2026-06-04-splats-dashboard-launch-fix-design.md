# Splats Dashboard Launch Fix — Design

**Date:** 2026-06-04
**Branch:** `refactor/cu121-uv-migration`
**Predecessor handoff:** `docs/superpowers/handoffs/2026-06-04-splats-dashboard-refinement-handoff.md`
**Supersedes that handoff's root-cause theory** (VTK/GL block — disproven below).

## Problem

The served dashboard accepts HTTP/websocket connections but never renders in the
browser. The user sees the process stop after the `Warp 1.14.0 initialized` banner, and
the browser tab shows `bokeh.protocol.exceptions.ProtocolError: Token is expired`.

### Root cause (verified)

The lazy-import refactor `dc82b9e` moved `FeedforwardResult` (and, via `viewer.py`,
`lift_features`) from `app.py` module-top into the request-time call sites
`_load_outputs` / `load_lifted_normed`. Importing that stack runs `@torch.compile` at
module level in `vggt/layers/mlp.py:19`, which spins up torch-inductor's compile-worker
subprocess pool (the Warp banner, the observed 433 threads, 263% CPU).

A faulthandler dump of a live served process proved the import runs **on the server's
asyncio IOLoop thread**, triggered during `initialize_document`:

```
panel get_session → initialize_document → factory builds SplatsApp
  → param watcher fires _on_video (app.py:148)        # auto-load of a cached video
  → _load_outputs (app.py:224)
  → load_lifted_normed (viewer.py:42)
  → import lift_features → vggt → torch.compile (vggt/layers/mlp.py:19)
  → torch._inductor compile-worker subprocess pool     # warp banner, threads, CPU
```

While the IOLoop is blocked, the browser's `/ws` websocket cannot be serviced. The
bokeh session token expires before the handshake completes → `Token is expired` → blank
page → the browser retries → each retry re-enters document init → sessions and threads
pile up.

**Why it worked before `dc82b9e`:** the import previously ran at module-top (launch
time); the server bound *after* it finished, so the IOLoop was free when a browser
connected and `torch.compile` was already warm. The lazy-import perf change pushed the
cost onto the request path on the IOLoop thread.

### Evidence summary

- `import collab_splats.dashboard.app` alone: 5.7s, EXIT 0, **does not** load warp — module import is not the problem.
- `import collab_splats.pointcloud.feedforward.base`: **15.6s**, loads warp + inductor.
- `import collab_splats.semantics.features.base` (query path): **10.2s**.
- Bare `SplatsApp.__init__` standalone: 1.9s, no heavy import (the cascade only reaches `_load_outputs` in the served doc-init path with a locally-cached first video).
- Served faulthandler dump: heavy import on the IOLoop thread, frame `app.py:148 _on_video → 224 _load_outputs → viewer.py:42`, identical across 3 dump cycles.
- `bad X server connection. DISPLAY=` only **warns**; VTK does not hang. The handoff's VTK/GL-block theory is wrong.
- `torch.compile` + CUDA from a worker thread: **works** (1.7s first compile, cached after); two concurrent CUDA threads both succeed. Async CUDA is functionally safe; the real risk is resource contention, not threading.
- A cached `feedforward.zarr` holds **500,000 points** × 2 panes for WebGL.

## Goals

1. The page renders in the browser; the IOLoop never blocks on heavy/CUDA work.
2. CUDA work is serialized (no parallel model loads → no GPU/RAM OOM under the 46 GB cap; no shared-state races).
3. GPU memory is reclaimed after every job (`pytorch_gc`).
4. The other page-load loop-blockers (rclone session listing, token expiry) are removed.
5. The 500k×2-pane WebGL render risk is bounded by configurable decimation.
6. `websocket_origin` is tightened from the blanket `*`.
7. A regression test prevents heavy imports from ever returning to the IOLoop thread.

## Non-goals

- Removing vggt's vendored module-level `@torch.compile` (left as-is; it is warmed, not eliminated).
- Automated real-browser WebGL verification (manual smoke-test only).
- Any change to the reconstruction pipeline, BA, LC, or eval code.

## Architecture — producer / consumer with one GPU worker

A process-singleton **`GpuWorker`** (one daemon thread + `queue.Queue`) owns *all* heavy
and CUDA work. IOLoop handlers only enqueue jobs and update UI state; they never import
the heavy stack or touch CUDA.

```
[IOLoop] click/auto-load
   → GpuWorker.submit(job_fn, on_done, doc)   # returns immediately, marks busy
[GPU worker thread]
   → job_fn()                                  # heavy import (warm-on-first-use), load / infer / score
   → pytorch_gc()                              # finally — empty_cache + synchronize + gc.collect
   → doc.add_next_tick_callback(on_done, result_or_error)
[IOLoop]
   → on_done: viewer render (synchronize) + re-enable buttons + status
```

- The `doc` reference is **captured at enqueue time** and passed to the worker. The
  worker never calls `pn.state.curdoc` (thread-local; `None` off the IOLoop — the latent
  bug in the current `_dispatch_load`).
- One worker thread ⇒ inherently serialized: no parallel model loads, no
  `_extractor_cache` race, `torch.compile` warms once.
- Created once in `run_app`, shared by every `SplatsApp` factory ⇒ serialization spans
  all tabs/sessions, preventing multi-tab GPU OOM.

## Components

### `dashboard/gpu_worker.py` (new)

- `submit(job_fn, on_done, doc) -> None`: enqueue `(job_fn, on_done, doc)`, set `busy = True`, return immediately.
- Worker loop: pop job → run `job_fn()` → `pytorch_gc()` in `finally` → schedule `on_done(result)` on `doc` via `add_next_tick_callback` → set `busy = False`.
- Job exception is caught; the error object is passed to `on_done`; the worker thread keeps draining (survives failures).
- `busy` exposed as an observable flag (param boolean or callback) so the app can disable buttons and show status.
- `doc is None` (tests / non-served): run `job_fn` + `on_done` inline, synchronously, so handlers stay unit-testable.

### `app.py`

- `_on_run`, the auto-load `_load_outputs`, and `_on_query` are refactored to: build the
  heavy closure, call `gpu_worker.submit(...)`, disable Run / Force / Run-query, set a
  busy status. All viewer mutation moves into the `on_done` callback (runs on the IOLoop).
- **Concurrency UX = disable + status:** while `busy`, the three action buttons are
  disabled; `on_done` re-enables them. No queue, no stacking.
- `_refresh_sessions` (rclone / network) moves **off the IOLoop**: run the listing on a
  light background thread, dispatch the dropdown options back via `add_next_tick_callback`.
  (rclone is not GPU work, so it uses a small thread, not the GPU queue.)
- New sidebar control **`max_display_points`** (IntInput/Slider, default 150 000) feeding
  the viewer's decimation.

### `viewer.py`

- The heavy bits (`FeedforwardResult.load_zarr`, `lift_features`, `score_queries`) move
  into worker jobs. `viewer.load` / `_render_left` / `_render_right` / `query`-recolour
  are only ever **called from the IOLoop** via `on_done`, keeping every `synchronize()` on
  the IOLoop thread.
- `_extractor_cache` is touched only by the single worker thread ⇒ no lock required.
- New **decimation helper**: subsample `points` / `colors` (and the matching
  similarity-score array) to `max_display_points` before building polydata. Full data
  stays on disk; only the display is decimated. Both panes use the same subsample indices
  so RGB and heatmap stay registered.

### `run_app`

- Instantiate one `GpuWorker`; pass it into the `SplatsApp` factory.
- Raise `session_token_expiration` (~1800s) on `pn.serve` as insurance against slow first
  loads.
- Tighten `websocket_origin`: default to localhost plus an explicit `allowed_origins`
  parameter; keep `"*"` available but opt-in (for SSH-tunnel / remote-IP access).

## Data flow

```
IOLoop (click / param watcher)
  → SplatsApp handler: submit(job, on_done, doc); disable buttons; status="running…"
GPU worker thread (serialized)
  → job: heavy import (first time) → load_zarr / lift_features / score_queries
  → pytorch_gc()  (finally)
  → schedule on_done(result) on doc
IOLoop
  → on_done: viewer.load / recolour (synchronize) → decimate to max_display_points
  → re-enable buttons; status="done" (or error)
```

## Error handling

- Per-job `try/except` in the worker; `pytorch_gc()` in `finally` so the CUDA cache is
  cleared even on failure.
- Errors are marshalled to `on_done`; the status line shows the error and buttons are
  re-enabled. The worker thread never dies.
- rclone listing failure keeps the existing behaviour: warn, empty options.
- Background push (`_push_async`) is unchanged — already a detached, non-fatal thread.

## Testing strategy

- **Regression (the gap that let this through):** drive the auto-load and query handlers
  with a fake/instrumented source and assert that `collab_splats.pointcloud.feedforward`
  and `collab_splats.semantics.features` are **not** imported synchronously on the calling
  (IOLoop) thread — the handler enqueues instead. Use an import tracker or a stubbed
  `GpuWorker` that records submissions.
- **`GpuWorker` unit tests:** a submitted job runs and `on_done` fires with its result;
  `pytorch_gc` is called after each job (patch + assert); a raising job is caught, surfaced
  to `on_done`, and the worker survives to run the next job; `busy` toggles around a job;
  `doc is None` runs inline.
- **Decimation:** a result with > budget points is subsampled to exactly the budget;
  points / colors / scores stay index-aligned; ≤ budget is a no-op.
- **Button state:** buttons disable on submit, re-enable in `on_done`.
- Existing `tests/dashboard/` (39) and semantics (121) stay green.

## Open risk (manual verification only)

Even decimated, 2 synced WebGL panes must be confirmed to render in a real browser on the
target host. This is a manual smoke-test after the server-side fixes land; it is not
covered by the automated suite. If a weak client still janks, lower the `max_display_points`
default.

## Rollback

All changes are confined to `collab_splats/dashboard/` (+ one new file) and tests. Revert
the dashboard commit to return to current behaviour; no migrations or data changes.
