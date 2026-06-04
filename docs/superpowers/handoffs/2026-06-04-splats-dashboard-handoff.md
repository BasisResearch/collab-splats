# Splats Dashboard — Handoff for Refinement

**Date:** 2026-06-04
**Branch:** merged into `refactor/cu121-uv-migration` (feature branch `feat/splats-dashboard` deleted after `--no-ff` merge; local only, not pushed)
**Spec:** `docs/superpowers/specs/2026-06-03-splats-dashboard-design.md`
**Plan:** `docs/superpowers/plans/2026-06-03-splats-dashboard.md`

## What this is

A streamlined single-page Panel dashboard for generating environment primitives from field video. Flow: **select session/video (rclone) → configure sidebar → Run → side-by-side viewer (pointcloud/mesh + text-query similarity)**. Replaces the old 5-tab dashboard. Standalone; depends on collab-data one-way for rclone only.

## Current state: WORKS, but only unit-tested

All modules are unit-tested with **heavy work mocked** (no GPU/models/rclone in the suite). The full pipeline has **never run end-to-end against real models + real rclone** in CI. Treat the happy-path integration as unverified until you do a live smoke run. 34 dashboard tests pass; broader subset (dashboard+semantics+pointcloud+mesh) 573 passed, no new failures vs baseline.

## File map (`collab_splats/dashboard/`)

| File | Responsibility | Notes |
|---|---|---|
| `config.py` | `RunConfig` dataclass — all sidebar knobs + provenance (`frame_indices`, `video_ref`); yaml roundtrip. | |
| `sources.py` | `SessionSource` — wraps collab-data `RcloneClient`. `list_sessions/list_videos/fetch_video/has_processed/pull_processed/push_outputs`. Lazy client: `RcloneClient()` failure → `self._client=None`, `_require_client()` raises on use. | Paths: curated `fieldwork_curated/reconstruction/{session}/*.mp4`; processed mirror `fieldwork_processed/reconstruction/{session}/{stem}/`. |
| `viz_utils.py` | Re-exports `pointcloud_to_polydata` (canonical in `utils/visualization.py`); defines `apply_viridis`. | |
| `pipeline.py` | `run_pipeline(*, video_path, session, stem, config, op_log, source, base_dir)` — sample → pointcloud → mesh → semantics → `run_config.yaml` → push. Background-thread-safe (no Panel calls inside). | Writes `frames.zarr`, `frames/*.jpg`, `feedforward.zarr`, `mesh/mesh.ply`, `semantics/{extractor}.zarr`. Push only on success. No BA/LC. |
| `viewer.py` | `SplitViewer` — two PyVista panes, browser camera jslink. `load/set_mode/query`. `load_lifted_normed(result, semantics_dir)` lifts cached features → normalised (P,D). | Right pane recolor only on query; no rerun. |
| `app.py` | `SplatsApp` single-page assembly + wiring + caching; `run_app(host,port,base_dir)`. | Worker thread → `run_pipeline` → `_dispatch_load` (marshals viewer update onto server doc via `add_next_tick_callback`). |
| `__main__.py` | `collab-dashboard` CLI (`--host/--port/--base-dir`). | |
| `operation_log.py` | `OperationLog` — thread-safe progress (kept, unchanged). | |

Deleted: `panes/`, legacy `App`, `state.py`, `video_server.py`.

## How to run

```bash
rclone listremotes                     # must show collab-data:
# run in tmux (GPU + 46GB cgroup cap; never a notebook)
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --port 7860
```
Open `http://localhost:7860`. Tests: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -q`. Format: `black collab_splats/dashboard tests/dashboard && isort ...`.

## collab-data SessionManager — reuse opportunity (READ THIS)

We reused collab-data's **`RcloneClient`** (`from collab_data.data_dashboard.rclone_client import RcloneClient`) — the real rclone integration. We did **not** use collab-data's **`SessionManager`**; we wrote our own thin `SessionSource` instead. Why, and what the refiner should know:

**`SessionManager`** (`collab_data/data_dashboard/session_manager.py`):
- `SessionManager(rclone_client, curated_path="fieldwork_curated", processed_path="fieldwork_processed")`.
- `discover_sessions(force_refresh=False) -> Dict[str, SessionInfo]`.
- `get_session(name, force_refresh=False) -> SessionInfo | None`.
- `get_file_path(session_name, file_type, relative_path) -> (bucket, full_path)`.
- `SessionInfo`: `name, curated_path, processed_path, curated_files, processed_files` + `has_curated/has_processed/is_complete`. `_load_directory_tree` recursively lists; file dicts have keys `Name, Size, IsDir, RelativePath, FullPath, Bucket`.

**Why we didn't use it:**
1. It pairs a curated+processed root into one `SessionInfo` with recursive file trees — our layout is a single root `fieldwork_curated/reconstruction/{session}` + per-video output dirs. Different model.
2. It only **lists** — no transfer. We need `fetch_video` (rclone `copyto` remote→local), `push_outputs`/`pull_processed` (`copy` / `copy_local_to_remote`), `has_processed`. Those wrap `RcloneClient` directly.
3. Our `fieldwork_processed/reconstruction/{session}/{stem}/` convention is splats-specific.

**Refiner options:**
- If you want tighter reuse, you *can* point `SessionManager(rclone_client, curated_path="fieldwork_curated/reconstruction", processed_path="")` and use `discover_sessions()` for the **listing half** (`list_sessions`/`list_videos`), keeping our `fetch/push/pull` transfer methods. Marginal gain; only worth it if collab-data's listing gains caching/features we want to inherit. The `RcloneClient.list_directory(bucket, path)` we call already does the job in ~6 lines.
- Keep an eye on **data-dashboard parity**: the data dashboard uses `Select`(session) + `Tabulator`(file tree) with a row-index→path map (`_file_tree_paths`) and a `bucket_type_toggle`. We mirrored only the Select+filter; if you want the full Tabulator file browser (subdirs, sizes), lift that pattern from `collab_data/data_dashboard/app.py`.

## Known gaps / suggested refinement work

**Unverified at runtime (mocked in tests) — smoke-test these first:**
1. **End-to-end Run** with real `vggt_omega` + `talk2dino` on a real session video. The creator contract (`creator.reconstruct(image_dir, out_dir)` then `creator.outputs` → `FeedforwardResult.save_zarr`) is mirrored from the old `panes/reconstruct.py` but unrun here.
2. **`load_lifted_normed`** — the cached-feature zarr glob (`semantics/*.zarr` → `["features"]` array `(N,D,Hp,Wp)`) matches `extract_and_cache`, but the lift→query→viridis path never ran with real features. Verify `lift_features` gets all fields it needs on the pipeline's `FeedforwardResult` (it needs `pixel_indices`, `depth`, `confidence`, `extrinsics`, `intrinsics`, `model_height/width`).
3. **Camera jslink** — `_left_pane.jslink(_right_pane, camera="camera", bidirectional=True)` is set up but only verified to construct, not to actually sync live in a browser. Confirm in a real session.
4. **Thread-safe viewer update** — `_dispatch_load` uses `pn.state.curdoc.add_next_tick_callback`; verify no Bokeh document errors during a live Run.

**Functional / UX:**
5. Mesh runs **inline** in `pipeline.py` (not the old subprocess isolation). OOM risk on the 46GB cap — if mesh OOMs, wrap `pointcloud_to_mesh` in a `multiprocessing.Process` (pattern existed in the deleted `panes/visualize.py`).
6. Optical-flow sampling stores **positional** `frame_indices` (`range(len(frames))`), not true source frame numbers (the OF sampler returns score dicts, not indices). Provenance for OF runs is positional only. Fix if exact frame provenance matters.
7. No progress detail beyond coarse percentages (`update_progress` at 25/60/80/95). Consider wiring creator/mesh tqdm callbacks through `OperationLog`.
8. Query fires on `TextInput.value` (Enter/blur), not per keystroke — fine. But the right pane only updates after a successful `load_lifted_normed`; if semantics missing, query silently no-ops. Add user feedback.
9. Single video per run only. Multi-video / whole-session combined reconstruction is a non-goal deferred from spec.

**Cosmetic:**
10. Panel 2.x `PendingDeprecationWarning`: `Button(button_type=...)` → will become `color=`; other widgets' `name=`. Non-blocking; batch-fix on a Panel upgrade.

**Out of scope / deferred (do not assume done):**
- BA/LC toggles (dropped from UI; backend intact).
- Camera localization (Localize tab removed).
- collab-data nav iframe entry (intentionally NOT built — fully separate for now; design stays compatible, just a URL).
- `collab_splats/webapp/` FastAPI prototype — untouched, candidate for removal once this is proven.

## Defaults (intentional)
env model `vggt_omega` (conf 50), semantic `talk2dino` (text-queryable), sampling `balanced` (50 frames), mesh voxel 0.01 / sdf 0.04 / depth 10.0 / clean_repair on. Push = **full** run tree incl. frames.zarr (so reruns reuse identical frames).
