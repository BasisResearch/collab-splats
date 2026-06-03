# Splats Dashboard — Design

**Date:** 2026-06-03
**Status:** Approved design, pending spec review
**Branch:** (TBD — off `refactor/cu121-uv-migration` or `main`)

## Summary

Rework the existing Panel dashboard (`collab_splats/dashboard/`) into a single, streamlined page focused on **generating environment primitives** from field-captured video. A user selects a recorded session/video (browsed live from GCS via collab-data's rclone tooling), configures the pipeline in the sidebar, presses **Run**, watches progress, then inspects the pointcloud, mesh, and a text-queried semantic-similarity map side by side.

The dashboard is a standalone Panel server living entirely in `collab-splats`. It is **not** embedded in collab-data — collab-data is a one-way pip dependency used only for its pure-Python rclone/session code.

## Goals

- Straightforward linear flow: **select → configure → run → visualize**.
- Browse `/reconstruction/` sessions (under `fieldwork_curated/reconstruction/`) the same way the collab-data data dashboard browses sessions (session dropdown + file list).
- One Run executes the full primitives pipeline: frame sampling → pointcloud → TSDF mesh → semantic features.
- Visualize pointcloud / mesh, plus a text-query semantic-similarity heatmap, **side by side** with synced cameras.
- All configuration — including the semantic query — lives in the sidebar, in collapsible per-step sections with sensible defaults.
- Reuse already-computed results when present; auto-push new results back to GCS so reruns share identical inputs.

## Non-Goals (v1)

- Bundle Adjustment / Loop Closure (dropped from the UI; backend code untouched).
- Camera localization (the current Localize tab is removed).
- Embedding the dashboard inside collab-data (no nav entry, no iframe — fully separate for now).
- Touching the FastAPI `collab_splats/webapp/` prototype (left as-is, out of scope; candidate for later removal).
- Multi-video / whole-session combined reconstruction (one video per run).

## Decisions (locked)

| Topic | Decision |
|---|---|
| Base stack | Rework existing Panel `dashboard/` into one page. Drop 5-tab structure, Localize, BA/LC, per-pane UIs. Keep the backend logic those panes called. |
| Session source | Live rclone listing via collab-data `RcloneClient` + `SessionManager`, root `fieldwork_curated/reconstruction`. |
| Cross-repo dep | collab-splats depends on collab-data (pip/uv). One-directional; collab-data has **no** dependency on collab-splats. |
| Selection | Session `Select` → `Tabulator` file list of everything under the session (data-dashboard style); user picks one mp4. |
| Run pipeline | Download mp4 → frame sample → pointcloud (env model) → TSDF mesh → semantic features. No BA/LC. |
| Defaults | env model `vggt_omega`; semantic extractor `talk2dino`; frame sampling = current default. |
| Config UX | Collapsible per-step sections in the sidebar (sampling, mesh params, …). Sensible defaults; expose for override. |
| Similarity | Text query in sidebar → **side-by-side split** viewer: RGB left, viridis cos-sim heatmap right, synced cameras. |
| Viewer modes | Left pane toggles pointcloud \| mesh. |
| Caching | Load if outputs already exist (local or `fieldwork_processed`); explicit **Force re-run** recomputes. |
| Output location | `/workspace/outputs/{session}/{video_stem}/`. |
| Provenance | Write selected `frame_indices` + sampling config + source mp4 ref into `run_config.yaml`. |
| Push | Auto-push after successful Run. **Full run derivatives** (incl. frames.zarr) so reruns reuse identical frames. |
| Processed path | Mirror curated: `fieldwork_processed/reconstruction/{session}/{video_stem}/`. |
| Integration | Standalone Panel server, `collab-dashboard` CLI. No collab-data entry. |

## Architecture

```
┌─ Sidebar ──────────────┐   ┌─ Main viewer (split) ─────────────┐
│ SOURCE                 │   │  [ RGB ]        |  [ Similarity ]  │
│  Session   [2026_05_07▾]│   │  pointcloud/mesh|  talk2dino       │
│  Files (Tabulator)     │   │  (toggle)       |  cos-sim heatmap  │
│   clip_01.mp4          │   │  synced cameras                    │
│   clip_03.mp4   ◀sel    │   └────────────────────────────────────┘
│ CONFIG (collapsible)   │   ┌─ Progress ────────────────────────┐
│  ▸ Frame sampling      │   │ [====      ] step 2/4 pointcloud   │
│  ▸ Env model [omega]   │   │ log…                               │
│  ▸ Semantic [talk2dino]│   └────────────────────────────────────┘
│  Query [ chair      ]  │
│  ▸ Mesh params         │
│  View: (pcd) (mesh)    │
│  [ Run ]  [ Force ↻ ]  │
└────────────────────────┘
```

### Components

**`dashboard/sources.py`** — session/video browsing + transfer.
- Wraps collab-data `RcloneClient` and `SessionManager(curated_path="fieldwork_curated/reconstruction", processed_path="")`.
- `list_sessions()` → session names (YYYY_MM_DD). `list_videos(session)` → mp4 file infos under it.
- `fetch_video(session, name) -> local_path` — rclone-copies the chosen mp4 into a local cache before the pipeline runs.
- `has_processed(session, stem) -> bool` and `pull_processed(...)` — check/pull existing outputs from `fieldwork_processed`.
- `push_outputs(local_dir, session, stem)` — `rclone copy` the full output tree to `fieldwork_processed/reconstruction/{session}/{stem}/`.
- What it does: turn a session/video selection into local files, and move outputs to/from GCS.
- Depends on: collab-data rclone client, configured remote.

**`dashboard/pipeline.py`** — run orchestrator.
- One entry: `run(video_path, config, op_log) -> output_dir`. Executes, in a background daemon thread:
  1. frame sampling (method + params from config) → `frames.zarr`; record `frame_indices`.
  2. pointcloud via env-model creator (`vggt_omega` default) → `feedforward.zarr`.
  3. TSDF mesh → `mesh/mesh.ply`.
  4. semantic features via extractor (`talk2dino` default) → `semantics/{extractor}/features.zarr`.
  5. write `run_config.yaml` (source mp4 ref, sampling config, frame_indices, model choices).
  6. on success → `sources.push_outputs(...)`.
- Reports progress through `OperationLog` (existing). No BA/LC.
- Reuses the existing creator/mesh/semantics backend calls the old panes used.
- Depends on: pointcloud creators, mesh module, semantics extractors, `sources`.

**`dashboard/viewer.py`** — side-by-side 3D.
- Two PyVista VTK panes with linked cameras.
- Left: RGB pointcloud or mesh (mode toggle).
- Right: same geometry recolored by cos-sim between per-point `talk2dino` features and the encoded text query (viridis). Empty query → neutral/uncolored.
- Editing the query recolors the right pane only (no pipeline rerun). Lifts per-frame features to points using existing `utils.lift_features` / reprojection helpers.
- Depends on: `FeedforwardResult`, semantic features, talk2dino text encoder.

**`dashboard/app.py`** — slim single-page assembly.
- Builds sidebar (source + collapsible config + query + Run/Force), the split viewer, and the progress strip.
- Holds `AppState` (reused) and wires watchers: select video → maybe load cached → enable Run; Run → `pipeline.run`; completion → populate viewer.
- Caching: on video select, if outputs exist locally or in `fieldwork_processed`, load into the viewer immediately; Run is only needed for new configs or Force re-run.

### Reused, unchanged
`OperationLog`, `AppState`, the PyVista viewer scaffolding (`ScenePanel` internals), and all pipeline backend functions (frame sampling, creators, mesh, semantics). The video HTTP server may be reused if a preview player is kept.

### Removed
Localize pane, BA/LC toggles, and the separate Preprocess / Reconstruct / Semantics pane UIs and the tab shell. Backend logic they invoked is retained and called from `pipeline.py`.

## Data flow

```
select session ──rclone lsf──▶ file list ──pick mp4──▶ fetch_video (rclone copy) ──▶ local mp4
                                                                                       │
  run_config.yaml ◀── write provenance ◀──────────────────────────────────── Run ◀────┘
        │
        ▼
  frames.zarr ─▶ feedforward.zarr ─▶ mesh.ply ─▶ semantics/features.zarr
        │                                                   │
        └──────────────── viewer (RGB | similarity) ◀───────┘
                                   │
                          auto-push full tree ──rclone copy──▶ fieldwork_processed/reconstruction/{session}/{stem}/
```

On reselect of a video with existing outputs: pull/load from `fieldwork_processed` (or local) straight into the viewer, skipping compute.

## Error handling

- **rclone unavailable / remote not configured:** surface a clear banner; disable Run; the rest of the UI degrades to local-only.
- **Video fetch failure:** report in the progress log; do not start the pipeline.
- **Pipeline step failure:** `OperationLog.error_op` with the failing step; partial outputs are *not* pushed (push only on full success).
- **Missing text-aligned features for a query:** if the active extractor is not queryable, disable the query box with a note (default `talk2dino` is queryable).
- **OOM risk:** pipeline runs in a background thread on the GPU box; viewer rendering is client-side (browser) to avoid competing with inference for GPU/RAM. Mesh generation keeps its existing subprocess isolation.

## Testing

- Flat pytest functions in `tests/dashboard/`, mirroring module layout.
- `sources`: mock `RcloneClient`; assert correct remote paths for list/fetch/push and the curated↔processed mirroring.
- `pipeline`: mock the heavy creators/mesh/semantics; assert step ordering, `run_config.yaml` contents (frame_indices, model choices), push-only-on-success, and progress callbacks.
- `viewer`: assert query→recolor maps to expected cos-sim coloring on a tiny synthetic cloud; camera linkage wiring.
- `app`: cache-hit loads without invoking `pipeline.run`; Force re-run does invoke it.

## Open follow-ups (post-v1)

- Optional collab-data nav iframe entry (deferred; design stays compatible — just a URL).
- Read-back / comparison view of pushed results, analysis-dashboard style.
- Remove or fold in the `webapp/` FastAPI prototype once this lands.
- Multi-video / whole-session reconstruction.
- BA/LC re-introduction behind an advanced section.
