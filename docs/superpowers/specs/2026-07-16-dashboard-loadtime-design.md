# Dashboard load-time & display-latency optimization

**Date:** 2026-07-16
**Status:** Design — pending review
**Scope:** Phase 1 + Phase 2 (dashboard-local). Phase 3 (F9 zarr re-chunk) deferred to its own decision doc.

## Context

Launching and using the splats/localize dashboard is slow and opaque:

1. **Slow launch** — ~17s of heavy ML imports; first Run/Localize interaction is cold.
2. **Frozen status bar** — selecting a scene shows no progress; "loading from server" is invisible, so a multi-GB network pull is indistinguishable from a hang.
3. **Hard crash** — the GPU worker thread dies permanently on a dead-document callback, silently bricking all further jobs.
4. **Slow load + display** — scene loads pull gigabytes the viewer discards, then re-read and re-render redundantly on every interaction.

Measured cold import cost (fresh process, venv `/opt/venv/reconstruction/bin/python`):

| module | cost |
|---|---|
| `pointcloud.feedforward.base` | 13.2s |
| `semantics.features.base` (timm + mobile_sam warnings) | 7.7s |
| `dashboard.pipeline` (+ mesh/TSDF) | 17.0s |
| `localization.localizer` | 9.7s |
| `mesh.utils` | 16.1s |

Goal: cut perceived load time, make network activity visible with a real progress bar, and stop the worker crash — without changing the on-disk zarr format (that is Phase 3).

## Root causes (evidence)

- **Worker crash:** `gpu_worker.py:60-67` `_loop` wraps the result marshal in `try/finally` with **no `except`**. When the browser session is gone (token expiry / closed tab / IOLoop blocked long enough to drop it), `doc.add_next_tick_callback` raises `AttributeError: 'DocumentCallbackManager' object has no attribute '_change_callbacks'`, which propagates out of `_loop` and kills the daemon thread. Every subsequent job queues to a queue nobody drains.
- **Over-pull:** `app.py:434` `_load_outputs` calls `pull_processed(session, stem, out)` with **no excludes**, copying the entire processed tree (`frames.zarr` + dense `depth`/`world_points`/`confidence`/`features`/`pixel_indices`/`images`, GBs per scene). Display needs only points+colors+extrinsics (+intrinsics). Commit `0bf9ac4` already defined `_PULL_EXCLUDES` in `pipeline.py` for the localization path — the splats path never got it.
- **Frozen poll:** the 300ms status poll (`app.py:543`, `localize.py:165`) runs on the IOLoop. Several rclone calls also run on the IOLoop (`app.py:_on_session`→`list_videos`; `_autoload_current`/`_on_run`→`has_processed`; `localize.py:_on_scene_session`/`_on_field_session`/`_on_camera`→`list_*`), blocking the loop so the poll can't tick.
- **No pull progress:** `pull_processed`/`fetch_video` (`sources.py`) run rclone with no `--stats`; the load job sets `start_op("loading …")` at 0% then jumps to 100%. `push_outputs` already streams `--stats-one-line` via `on_line` — the pattern to copy.
- **Eager decode:** `FeedforwardResult.load_zarr` (`base.py:206-211`) `[:]`-reads every dense array that exists, even for display; only `images` is gated. `app.py:441` eager-loads `lifted_normed.npy` before any query.
- **Redundant render:** `viewer.py` re-reads the mesh from disk via `pv.read()` in three places (`_render_left:236`, `_render_right:259`, `ensure_mesh_features:174`) — twice per mode switch, again on every normalize/query. `_normalize` (`viewer.py:193-197`) transforms `inplace=True`, so caching the mesh would compound the transform. Query recolor (`_render_right:257-271`) does `clear()` + full `pointcloud_to_polydata` rebuild + re-add actor when only colors changed. 500k points render as spheres (`PCD_KWARGS`) at `point_size=0.5` (invisibly small → sphere cost wasted).
- **No splats cache:** `shell.py:34` constructs `SplatsApp` without the shared `SceneCache`; reselecting an already-displayed scene re-pulls + re-reads fully. Both pages issue duplicate `list_sessions` on startup against the same `SessionSource`.
- **Chatty disk:** `_persist_state` (`app.py:107,190-191`) writes the whole `.dashboard_state.yaml` synchronously on every widget `value` change (per slider tick / keystroke).

## Phase 1 — core (high impact, low risk)

### F5 — GPU worker survives a dead document
`gpu_worker.py:_loop`. Wrap `doc.add_next_tick_callback(...)` in `try/except`. On failure (session destroyed): log at debug, reset `self.busy = False`, `continue` the loop — never let `_loop` exit. Optionally pre-check `doc.session_context` before scheduling. Keeps the worker alive across session churn.

### F6 — splats pull excludes dense arrays
`app.py:_load_outputs` job. Pass the existing `_PULL_EXCLUDES` (defined in `pipeline.py` by `0bf9ac4`) to `pull_processed`. Promote `_PULL_EXCLUDES` to a shared location (e.g. `sources.py` or a small dashboard constants module) so both the localization and splats paths import one definition. Safe: `load_zarr` treats the excluded members as optional (absent → None); display never reads them; mesh/semantics live in separate dirs.

### F3 — real "loading from server" progress bar
`sources.py` + `app.py`. Give `pull_processed` and `fetch_video` the `--stats 2s --stats-one-line` + `on_line` streaming that `push_outputs` already has. Parse rclone's `transferred X / Y, NN%, speed` line; push to `op_log.update_progress(pct, "⬇ pulling from server …")` so the bar shows a live percentage + MB/s, visually distinct from GPU/compute stages. Background video fetches (`_update_max_frames_bound`, `_ensure_local_video`, localize `_ensure_local_query_video`) get `start_op`/`finish_op` so the bar reflects remote fetches instead of sitting Idle.

### F2 — move blocking rclone off the IOLoop
`app.py` + `localize.py`. Wrap the remaining synchronous rclone watchers on daemon threads, setting widget options back via `doc.add_next_tick_callback` (the pattern `_refresh_sessions`/`_refresh_listings` already use): `SplatsApp._on_session`→`list_videos`; `_autoload_current`/`_on_run`→`has_processed`; `LocalizePage._on_scene_session`→`list_videos`, `_on_field_session`→`list_rgb_cameras`, `_on_camera`→`list_camera_videos`.

### F7 — memoize rclone listings + `has_processed`
`sources.py`. Add a per-key TTL cache (e.g. 60s) at the `SessionSource` layer for `list_sessions`, `list_videos`, `list_localization_dbs`, `list_rgb_cameras`, `list_camera_videos`, `has_processed`. Invalidate `(session, stem)` after a run's `push_outputs`. Collapses the two duplicate startup `list_sessions` calls and stops re-hitting GCS on reselection.

### F11 — cache mesh PolyData in memory; fix in-place transform
`viewer.py`. Read the mesh once in `load()`, cache the `pv.PolyData` (mirror `localize.py`'s `SceneCache` mesh caching); reuse it in `_render_left`/`_render_right`/`ensure_mesh_features` instead of `pv.read()` per call. Make `_normalize` non-mutating (apply transform to a copy / `inplace=False`, or apply once at load) so the cached mesh is not re-transformed each render.

### F12 — recolor without geometry rebuild
`viewer.py:_render_right`. On a query recolor, update the existing actor's `RGB` point-data + `Modified()` rather than `clear()` + rebuild polydata + re-add actor. Only rebuild when geometry (mode / point set) actually changes.

### F15 — SceneCache for SplatsApp
`shell.py` + `app.py`. Pass the shared `SceneCache` into `SplatsApp`. Key `(session, stem)` on kinds `result` / `mesh` / `lifted_normed`. In `_load_outputs`, short-circuit if the currently-displayed scene matches; otherwise check cache before enqueuing the worker job. Invalidate `(session, stem)` on Force re-run (`_on_run force=True`) and after a successful `run_pipeline`.

### F1 — warm the full stack
`app.py:_warm_heavy_stack`. Add `collab_splats.dashboard.pipeline` (pulls mesh + feedforward) and `collab_splats.localization.localizer` to the background warm thread so the first Run/Localize isn't a ~10-17s cold start. Stays off the IOLoop; module import-lock means a racing first interaction just waits on the same import.

## Phase 2 — display tuning & decode deferral (low risk)

### F13 — cheaper point rendering
`viz_utils.py` `PCD_KWARGS`. Set `render_points_as_spheres=False` (plain GL points — spheres at `point_size=0.5` are invisible anyway) and/or lower the default `max_display_points`. Keep the even-stride decimation.

### F14 — skip redundant per-render overhead
`viewer.py:_apply_view`. Skip light teardown/rebuild + camera reset when geometry is unchanged (e.g. recolor-only). Combined with F12, a color update pushes far less over `synchronize()`.

### F10 — defer `lifted_normed.npy` load
`app.py:_load_outputs`. Don't `np.load(lifted_normed.npy)` at load; defer to first query, matching the already-lazy `ensure_lifted` path in `viewer.py`.

### F8 — opt-in dense-array decode in `load_zarr`
`pointcloud/feedforward/base.py:load_zarr`. Add per-array load flags (mirroring the existing `load_images`) for `depth`/`world_points`/`confidence`/`features`/`pixel_indices`; default to current behavior (True) for back-compat, and have the dashboard display path pass False. Protects locally-generated / re-run scenes that still carry the dense arrays on disk (rclone-exclude only masks freshly-pulled ones).

### F16 — debounce state persistence
`app.py:_persist_state`. Coalesce writes on a short timer (~500ms) or persist on blur / Run only, instead of a full `yaml.safe_dump` + `write_text` per intermediate widget `value` event.

### F4 — lazy plotter build
`shell.py` + `viewer.py`/`localize.py`. Defer the inactive tab's `pv.Plotter` + `pn.pane.VTK` construction until first tab activation (`_on_tab`). **Modest payoff** — plotter construction is sub-second; the dominant first-render cost is `pn.extension("vtk", inline=True)` streaming the vtk.js bundle, which this does not touch. Included for completeness; lowest priority in the set.

## Non-goals / deferred

- **F9 — re-chunk points/colors in `save_zarr`** (`base.py:136,140` `chunks=arr.shape`). Storing the whole cloud as one lz4 chunk forces a full decompress before decimating to 500k. Row-block chunks (e.g. 100k points) would enable strided/partial reads matched to the display decimation. This changes the on-disk format and affects non-dashboard consumers (eval, localization, re-load) → needs a migration/back-compat story and its own `docs/superpowers/decisions/NNN-*.md`. Out of scope here.

## Verification

- **Launch:** `time` from `python -m collab_splats.dashboard` to first page render; confirm server binds before the warm thread finishes.
- **Cold interaction:** immediately after launch, open the Localize tab and hit Run — no ~10s import stall (F1).
- **Network visibility:** select a remote scene — status bar ticks a live percentage + MB/s during list + pull, distinct from compute stages (F2, F3); was frozen/Idle before.
- **Load speed:** compare wall-clock for a known multi-GB scene before/after F6 (should drop from minutes to seconds for pull); confirm the pointcloud still renders with correct colors/extrinsics and queries + mesh still work (excluded arrays unused).
- **Interaction latency:** toggle pointcloud↔mesh, normalize on/off, run a query — no mesh re-read from disk, recolor is near-instant (F11, F12); reselecting the same scene is instant (F15).
- **Crash:** force a session to expire mid-job (or close the tab during a Run) — worker logs and survives; a fresh browser session still gets working Run/load (F5).
- **Suite:** `/opt/venv/reconstruction/bin/python -m pytest tests/` (dashboard tests green; add/extend tests for F5 dead-doc guard, F6 excludes wiring, F7 memoization, F15 cache short-circuit).

## Risks

- **F8 default:** must default the new flags to current behavior so non-dashboard callers of `load_zarr` are unaffected; only the display path opts out.
- **F7 staleness:** listing cache must invalidate after a push/run so newly-produced outputs appear; keep TTL short and provide an explicit refresh if needed.
- **F11 correctness:** the non-mutating `_normalize` change must be verified across mode/normalize/query permutations (the in-place transform is currently load-bearing for a single render pass).
- **F15 invalidation:** stale scene must not display after a Force re-run — evict `(session, stem)` on force and on successful `run_pipeline`.
