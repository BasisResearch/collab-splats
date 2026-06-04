# Splats Dashboard Refinement — Handoff

**Date:** 2026-06-04
**Branch:** `refactor/cu121-uv-migration`
**Spec:** `docs/superpowers/specs/2026-06-04-splats-dashboard-refinement-design.md`
**Predecessor:** `docs/superpowers/handoffs/2026-06-04-splats-dashboard-handoff.md` (the "working dashboard before")

This refinement has two distinct halves:
1. **Six feature/UX items** from the spec (functional changes to the dashboard).
2. **Four launch/serve infrastructure fixes** discovered while trying to actually run it
   end-to-end — these change *how the app boots and is served*, not what it does.

Commits (oldest→newest):
`890f805` (6 items) · `dc82b9e` (lazy import) · `4170864` (vtk ext + websocket) ·
`696ee51` (Xvfb) · `4db374f` (inline resources).

Files changed vs `5a3ea06`: `app.py`, `config.py`, `operation_log.py`, `pipeline.py`,
`sources.py`, `viewer.py`, `semantics/features/talk2dino.py` (+318 / −95).

---

## ⚠️ Operational lesson — read first (this caused a long debugging detour)

A served instance whose VTK panes can't get an OpenGL context **blocks in a C-level GL
call on the main thread**. Consequences:
- **Ctrl-C does nothing** — SIGINT can't be delivered while stuck in C.
- The process keeps **holding the port**. Every relaunch then dies with
  `OSError: [Errno 98] Address already in use`, and the browser keeps hitting the *old,
  broken* process — so code fixes appear to have no effect.

If a launch shows `Address already in use`, or the page won't change no matter what you
edit: **a stale instance is still bound.** Kill it hard (Ctrl-C won't):
```bash
pkill -9 -f collab_splats.dashboard; pkill -9 Xvfb
# verify free:
ps -eo pid,cmd | grep collab_splats.dashboard | grep -v grep   # expect nothing
```
The Xvfb auto-start (fix #3 below) removes the GL-block at its root, so new instances stop
in the foreground cleanly. But a zombie from before that fix must be `kill -9`'d once.

---

## Part 1 — Six feature/UX items (vs the prior working dashboard)

### 1. Conditional sampling widget
- **Before:** `min_disparity` always visible in the Frame-sampling card, even for
  `balanced` (fps) sampling, which never uses it.
- **Now:** new `_bind_visibility(widget, selector, predicate)` helper in `app.py`;
  `min_disparity` is shown **only** when `sampling == "optical_flow"`. Audit found sampling
  is the only pane with unselected-irrelevant knobs (env `conf` maps to both creators; mesh
  always runs). Helper is reusable for future per-attribute gating.

### 2. Granular progress + per-step timings
- **Before:** coarse `op_log.update_progress` at fixed 25/60/80/95%; no timings, no counts.
- **Now:**
  - **Logging bridge** — `OperationLog.attach_logging("collab_splats")` (context manager,
    `operation_log.py`) installs a `logging.Handler` that forwards module `INFO` records
    (e.g. `extract_and_cache: 12/50 frames written`) into the dashboard log during a run.
    No creator/semantics API change — the modules already log the detail.
  - **Per-step `perf_counter` timings** in `pipeline.py` → log lines like
    `pointcloud (vggt_omega): 42.3s`.
  - **Query stage lines** — `viewer.query` emits `encoding… → scoring P points → recolour
    done` to op_log on every Run query.
  - *Note:* tqdm bars (model inference, `lift_features`) write to stderr, not `logging`, so
    they stay in the tmux console, not the web log. Accepted.

### 3. Mesh defaults
- **Before:** voxel 0.01 / sdf 0.04 / depth 10.0 / clean_repair **on**.
- **Now:** voxel **0.005** / sdf **0.02** / depth **1.0** / clean_repair **off**. Changed in
  both `config.py` (`RunConfig`) and `app.py` widget defaults.

### 4. Semantics query — reuse `score_queries`
- **Before:** single `Query` TextInput firing on value-change; `viewer.query` hand-rolled
  L2-norm + cosine + viridis on one text embedding.
- **Now:** **Positive query** + **Negative query** TextInputs + a **Run query** button.
  `viewer.query(positive, negative, …)` calls the existing
  `BaseQueryableExtractor.score_queries` (contrastive softmax, Talk2DINO paper convention,
  `[0,1]`, native multi-term via comma-split). Blank negative → API default `["object"]`.
  `RunConfig.query` → `query_positive` / `query_negative` (provenance yaml).

### 5. Push robustness — streamed background dir-copy
- **Before:** `push_outputs` called `RcloneClient.copy_local_to_remote`, which runs
  `rclone copyto` (single file→file) with a hardcoded 120s timeout — on a **directory**.
  Live runs failed: `copyto … PXL_… timed out after 120 seconds`.
- **Now:** `push_outputs` runs `rclone copy` (recursive, **idempotent**) via `Popen`,
  streaming `--stats-one-line` to op_log, with rclone-native `--retries 3 --timeout 300
  --contimeout 60 --transfers 8` instead of a python wall-clock cap. Runs in a **detached,
  non-fatal background thread** (`_push_async` in `pipeline.py`): the viewer loads as soon
  as local outputs are ready; push failure only logs (outputs are already on disk).
- *Why not extend collab-data:* its `RcloneClient` is all single-file/blocking/short-timeout
  because the data dashboard only reads/writes small metadata + serves buckets — it never
  bulk-uploads trees. We route around it from our side.

### 6. Suppress talk2dino meta-tensor warnings
- **Before:** every talk2dino load spammed `copying from a non-meta parameter … no-op`
  (`visual.transformer.*`) from its internal CLIP load.
- **Now:** targeted `warnings.filterwarnings(..., message=".*copying from a non-meta
  parameter.*")` scoped to the `from_pretrained` call in `talk2dino.py`. Harmless no-op
  (CLIP visual tower is unused; DINO is the backbone).

---

## Part 2 — Launch / serve infrastructure fixes (NEW boot behavior)

These did **not** exist in the prior working dashboard and are the reason it now actually
serves on a headless host. Each was a separate "page won't load" symptom.

### A. Lazy-import the heavy reconstruction stack (`dc82b9e`)
- **Problem:** `app.py` imported `pipeline` (and `viewer` imported `pointcloud.utils`) at
  module top, eagerly pulling VGGT/MapAnything/TSDF-mesh/warp. `python -X importtime`: the
  app import was **~19.8s** — the server couldn't bind for ~20s, looking like a hang.
- **Fix:** defer `run_pipeline`, `FeedforwardResult`, `lift_features`,
  `BaseQueryableExtractor` to their run/load/query call sites (legit heavy-dep exception per
  CLAUDE.md). **App import 19.8s → 4.2s.** The heavy stack now loads only when a run/query
  fires (you'll see timm/warp load in the terminal on first Run, not at launch).

### B. Load `pn.extension("vtk")` once at startup + allow remote websocket (`4170864`)
- **Before:** `pn.extension("vtk")` was called lazily **inside `view()` per session** with a
  `_VTK_EXT_LOADED` global guard. Panel requires it once at startup in the main thread;
  deferring it fails to inject the VTK JS and the panes hang.
- **Fix:** call it once in `run_app` before `pn.serve`; dropped the guard. Also added
  `websocket_origin="*"` (overridable param) so the app renders when reached via a remote
  host IP / SSH tunnel — bokeh otherwise refuses the websocket and the page stays blank.

### C. Auto-start Xvfb for headless VTK (`696ee51`)
- **Problem:** on a host with no `DISPLAY`, `pn.pane.VTK` builds a
  `vtkXOpenGLRenderWindow` during document creation → `bad X server connection. DISPLAY=`
  → blocks in C (page spins, Ctrl-C swallowed — see Operational lesson above).
- **Fix:** `_ensure_display()` in `app.py`, called at the top of `run_app`: if `DISPLAY` is
  unset and `Xvfb` is on PATH, spin up a virtual display (`:99`, software GL via
  `LIBGL_ALWAYS_SOFTWARE=1`), point `DISPLAY` at it, `atexit`-terminate. Verified headless:
  `pl.render()` and `VTKRenderWindowSynchronized._get_properties()` serialize with no bad-X
  and no hang. **No `xvfb-run` wrapper needed** — plain `python -m collab_splats.dashboard`.

### D. Serve JS/CSS inline, not from CDN (`4db374f`)
- **Problem:** the served HTML pulled ~90 resources from `cdn.holoviz.org` /
  `cdn.bokeh.org`, including the large `vtk.js` bundle. A restricted/offline host can't reach
  the CDN → JS never loads → "untitled" page spins.
- **Fix:** `pn.extension("vtk", inline=True)` bundles/serves all JS/CSS from this bokeh
  server's own static routes — zero external CDN. (Trade-off: heavier first byte; correct
  for headless/air-gapped.)

---

## How to run

```bash
# clear any stale instance first (Ctrl-C may not have worked — see Operational lesson)
pkill -9 -f collab_splats.dashboard; pkill -9 Xvfb

# in tmux (GPU + 46GB cgroup cap; never a notebook)
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --port 7860
```
Expect: `Launching server at http://0.0.0.0:7860`, rclone verify, **no** `bad X server`
line. timm/warp load on the **first Run** (lazy), not at launch. Stop with Ctrl-C (works
now that VTK no longer blocks the main thread).

Tests: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -q` → **39 pass**
(+ semantics 121 pass). Format: `black collab_splats/dashboard tests/dashboard && isort …`.

## Still unverified at runtime (the original smoke-test list still stands)

The suite mocks all heavy work; the **full pipeline has still never completed end-to-end**
against real models + rclone in this environment (every attempt was blocked by the stale
port-holding zombie, now cleared). After a clean launch, smoke-test in priority order:
1. **Page renders** — two viewer panes, no frozen tab. (If the tab still freezes with a
   single trivial VTK pane, the host browser/WebGL is the limit — see `/tmp/paneltest.py`
   bisection: `paneltest.py plain` vs `paneltest.py vtk`.)
2. **End-to-end Run** — real `vggt_omega` + `talk2dino` on a real session video; confirm
   per-step timing lines + `XX/XX` frame counts appear in the web log.
3. **Background push** — confirm `rclone copy` stats stream to the log and a push failure
   doesn't kill the run.
4. **Query** — positive/negative + Run query recolours the right pane; stage lines log.

## Diagnostic helper left on disk

`/tmp/paneltest.py` — minimal panel-serve bisection (`plain` = no VTK, `vtk` = one pane).
Use to separate "panel-serve-to-browser works" from "VTK panes freeze this browser" if the
full app still won't render. Not committed (throwaway).
