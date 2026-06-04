# Splats Dashboard Launch Fix — Handoff

**Date:** 2026-06-04
**Branch:** `refactor/cu121-uv-migration`
**Spec:** `docs/superpowers/specs/2026-06-04-splats-dashboard-launch-fix-design.md`
**Plan:** `docs/superpowers/plans/2026-06-04-splats-dashboard-launch-fix.md`

## What was wrong

The served dashboard accepted connections but never rendered; the browser showed
`bokeh.protocol.exceptions.ProtocolError: Token is expired`. Root cause (verified by a
faulthandler dump of a live hung process): the lazy-import refactor `dc82b9e` moved the
heavy `feedforward`/`semantics` imports onto the **server's asyncio IOLoop thread** during
`initialize_document`. Importing that stack runs `@torch.compile` at module level in
`vggt/layers/mlp.py:19`, which spins up torch-inductor's compile-worker pool — blocking the
IOLoop long enough that the bokeh session token expired before the `/ws` handshake. The
handoff predecessor's VTK/GL-block theory was wrong (`bad X server` only warns).

## What changed

All heavy/CUDA work now runs on a single serialized **`GpuWorker`** (one daemon thread +
queue), off the IOLoop. Results marshal back via `doc.add_next_tick_callback`, where all
VTK/panel mutation happens. Specifics:

- `collab_splats/dashboard/gpu_worker.py` (new) — serialized worker; `pytorch_gc()` after
  every job (`finally`); errors surfaced to `on_done`, worker survives; `doc=None` runs
  inline for tests.
- `app.py` — `_on_run`, `_load_outputs` (auto-load), `_on_query` enqueue jobs and gate the
  action buttons via `_set_busy`; `_refresh_sessions` (rclone) moved off-loop;
  `max_display_points` widget added; one shared `GpuWorker` injected by `run_app`.
- `viewer.py` — `query` split into `score_query` (off-loop compute) + `render_query`
  (on-loop); configurable point decimation (`_decimate_indices`, `load(max_points=...)`).
- `run_app` — shared `GpuWorker`; `session_token_expiration=1800`; `websocket_origin`
  defaults to host:port + localhost:port (was blanket `*`; pass a list or `"*"` for tunnels).

Tests: `tests/dashboard/` **56 pass**, `tests/semantics/` **121 pass**. Black + isort clean.

## Smoke-test results (server-side — done)

Launched `python -m collab_splats.dashboard --port 8081` on this headless host:
- ✅ No Warp banner at launch (heavy stack stays lazy).
- ✅ `GET /` returns **HTTP 200 in 2.77s**; repeat requests 1.57s → 0.68s. The IOLoop is
  no longer blocked — **the freeze is fixed** (previously curl timed out indefinitely).
- ✅ Loop stays responsive across concurrent requests.

Observation: a fresh session shows a high thread count (~290 after 3 page loads). These are
**llvmpipe (software-GL) worker threads** that VTK spawns per `pv.Plotter` per session —
pre-existing VTK behavior under `LIBGL_ALWAYS_SOFTWARE=1`, **not** torch/the freeze (warp
never loaded during these requests). It does mean each session/tab is expensive and
sessions are not disposed — see "Still open" below.

## Still open (needs a real browser — not automated)

1. **Browser WebGL render.** curl fetches only the page HTML; it does not drive a full
   bokeh websocket session, so the auto-load watcher (→ deferred heavy job) and the 2 VTK
   panes were not exercised end-to-end here. The deferral is covered by the unit regression
   test `test_load_outputs_defers_heavy_work_to_worker`, but confirm in a browser:
   - Select a cached session/video → outputs load (buttons disable, then re-enable); pane
     shows the decimated pointcloud; check the terminal shows the Warp banner appearing on
     **first load** (background worker), not at launch.
   - Positive/negative query + Run query recolours the right pane; buttons disable during.
   - Lower **Max display points** → reload → fewer points, more responsive.
2. **Session/Plotter pileup (spec #5).** Each session leaks 2 Plotters + their llvmpipe
   threads; many tabs could grow memory under the 46 GB cap. The freeze fix stops the
   *retry-storm* pileup, but legitimate multi-tab disposal is future work.

## How to run

```bash
pkill -9 -f collab_splats.dashboard; pkill -9 Xvfb   # clear any stale instance first
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --port 7860
```
Expect: serves immediately, no Warp banner at launch; Warp + timm load on the first
Run/load (background worker), not at boot.
