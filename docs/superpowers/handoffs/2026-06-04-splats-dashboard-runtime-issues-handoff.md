# Splats Dashboard — Runtime Issues Handoff

**Date:** 2026-06-04
**Branch:** `refactor/cu121-uv-migration`
**Predecessors:**
- `docs/superpowers/specs/2026-06-04-splats-dashboard-launch-fix-design.md`
- `docs/superpowers/plans/2026-06-04-splats-dashboard-launch-fix.md`
- `docs/superpowers/handoffs/2026-06-04-splats-dashboard-launch-fix-handoff.md`

This handoff covers issues found **after** the launch-fix landed, while the user ran real
end-to-end reconstructions through the dashboard and compared them to the tutorial
notebook path. Read the predecessors first for the off-loop/GpuWorker architecture.

---

## Context

The dashboard now launches, serves, and renders (the token-expired freeze is fixed: heavy
imports + CUDA run on a serialized `GpuWorker` off the IOLoop; see predecessor handoff).
The user then exercised the actual pipeline and surfaced a cluster of correctness/UX
problems. Several were fixed this session; the **most important one (distorted pointcloud)
is diagnosed but NOT yet fixed** — that is the next agent's primary task.

The canonical "correct" path the user compares against is the tutorial notebooks:
- `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`
- `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`

The dashboard must reproduce what those notebooks produce.

---

## PRIMARY OPEN ISSUE — pointcloud is distorted / "destroyed" vs the notebook

**Root cause (confirmed by reading both paths): the dashboard never cleans the pointcloud.**

Notebook (`feedforward_methods.ipynb`, §8 `_clean`):
```python
result_omega = VGGTOmegaCreator().run(IMAGES, device)
pcd = o3d PointCloud(result.points, result.colors)
cleaned, idx = clean_pointcloud(pcd)     # outlier removal + voxel downsample
# prints: Points: 500,000 raw → 55,927 filtered
# renders the CLEANED 55,927 points
```

Dashboard (`collab_splats/dashboard/pipeline.py:158-162`):
```python
creator = _build_creator(config.env_model, config.conf_threshold)
creator.reconstruct(image_dir, out_dir)
result = creator.outputs
result.save_zarr(out_dir / "feedforward.zarr")   # RAW 500k pts, outliers included
```
→ The viewer renders the **raw 500,000 points with all of VGGT-Omega's outlier flyers**.
The notebook strips those (500k → ~56k) via `clean_pointcloud`. That scatter of flyers is
the "destroyed/distorted" look. **conf_threshold is NOT the cause** (notebook uses default
`VGGTOmegaCreator()` = conf 50 and still gets 500k raw → cleans down). The missing step is
`clean_pointcloud`.

### The fix (design notes — not yet implemented)

`clean_pointcloud` is in `collab_splats/pointcloud/utils.py:208`. Signature:
```python
clean_pointcloud(pcd, downsample_kwargs=_UNSET, outlier_kwargs=_UNSET, distance_kwargs=_UNSET)
  -> (cleaned_pcd, index_mapping)   # index_mapping maps each output point back to its ORIGINAL index
```
Steps (each skippable by passing None): voxel downsample → statistical outlier removal →
distance removal. Module defaults `_DEFAULT_DOWNSAMPLE_KWARGS` / `_DEFAULT_OUTLIER_KWARGS` /
`_DEFAULT_DISTANCE_KWARGS` near the top of utils.py.

**Critical constraint — keep query/lift alignment.** `lift_features` and `score_query` map
features to the FULL `result.points` (via per-frame `pixel_indices`/`world_points`). If you
clean by **voxel downsample** the points are averaged → no 1:1 index map → query recolour
would misalign. BUT `clean_pointcloud` returns an `index_mapping` (original indices of the
surviving points). Use that mapping so query scores can be indexed to the displayed set.

Recommended approach (keeps lifting correct, de-distorts the display):
- Do the cleaning in the **viewer** at load time (off-loop, in the GpuWorker job), mirroring
  the notebook which keeps `result` raw and renders a separate cleaned cloud.
- Store `self._clean_idx = index_mapping` and render `result.points[clean_idx]` /
  `colors[clean_idx]`. For query recolour, index the per-point scores by `clean_idx` too, so
  colours stay registered. This REPLACES the current even-stride `_decimate_indices` display
  path (cleaning both de-distorts AND reduces count to ~56k, which is browser-friendly).
- Decide whether to enable voxel-downsample inside `clean_pointcloud` (it averages points but
  `index_mapping` still maps to a representative original index, so query indexing still
  works) or to use outlier-removal-only for an exact subset. Match the notebook output
  (~56k) — i.e. use the default kwargs the notebook uses (it calls `clean_pointcloud(pcd)`
  with all defaults).
- Verify against the notebook: same video + same keyframes should yield ~the same filtered
  count and visual shape.

This likely **supersedes the `max_display_points` even-stride decimation** added earlier
(it was a guess to fix browser stall; cleaning is the correct, notebook-matching reduction).
Keep a cap as a safety only if needed.

---

## SECONDARY OPEN ISSUE — optical-flow frame count differs from the notebook

User: with `min_disparity=50` (the notebook's value) the notebook "reaches the max number
of frames quickly", but the dashboard produced only **36 frames**. (My earlier claim that
"min_disparity=50 is strict so few frames" was WRONG — disregard it.)

The two paths are **different code**:
- **Notebook** (`keyframe_extraction.ipynb` §3-4): `score_all_frames(VIDEO_PATH)` — runs at
  **`stride=5`** (scores every 5th frame), then `of_indices = [d["frame_idx"] for d in
  frame_scores if d["selected"]][:MAX_FRAMES]`.
- **Dashboard** (`collab_splats/dashboard/pipeline.py:_sample` → `sample_frames_optical_flow`,
  `collab_splats/utils/frame_sampling.py:599`): iterates **every** decoded frame (stride=1),
  selects score≥0.5, breaks at `max_frames`.

Caveats when comparing: the notebook ran on `birds_c0043` (2388 frames); the user's dashboard
run was `2024_02_06/C0043` — possibly a different clip. Still, the path divergence (stride=1
vs stride=5, streaming-select vs score-then-slice) is real and likely changes which/how many
keyframes are chosen, hence a different VGGT input set and a different reconstruction.

**Next step:** reconcile the dashboard sampling to the notebook. Either call the same
`score_all_frames(..., stride=5)` + `[selected][:max_frames]` logic, or confirm
`sample_frames_optical_flow` is intended to match and fix the discrepancy. Decide with the
user which sampler is canonical. Also note `_sample` writes JPEGs via `_write_frames_jpegs`
(`{i:05d}.jpg`, selection order) whereas the notebook writes `frame_{idx:06d}.jpg` (original
video index) — both sort temporally, but verify no ordering/resolution difference feeds VGGT
differently.

---

## FIXED THIS SESSION (committed on `refactor/cu121-uv-migration`)

1. **Per-step logging now visible.** `view()` rendered only `current_op` (set once to the
   scene name). Now it renders the live `op_log` panel (stage label + bar + scrolling log);
   `update_progress` sets `current_op` per stage; `append_line` collapses consecutive dupes
   (the per-frame "sampling frames" spam). Files: `operation_log.py`, `app.py` `view()`.
2. **Mesh "not found" fixed.** Viewer looked for `mesh.ply`; TSDF writes `mesh_tsdf.ply`
   (`collab_splats/mesh/tsdf.py:94`). `app.py:_load_outputs` now points at `mesh_tsdf.ply`.
3. **Config defaults:** `max_frames` 50→100, `conf_threshold` 50→35 (`config.py` + widgets).
   (User stated these are the intended defaults.)
4. **Decimation default** raised 50k→500k (no decimation) per user choice — but see the
   PRIMARY issue: this should likely be replaced by `clean_pointcloud`.
5. **Lazy feature-lift** (earlier in session): `lift_features` (≈378s on 500k pts) was run
   eagerly on auto-load, freezing the UI for 6+ min. Now deferred to first query, cached;
   auto-load is `load_zarr` only (~0.7s). Background-warm added in `run_app`.

All dashboard tests pass (58) + semantics (121).

## USER DECISIONS ON RECORD (do not relitigate without asking)

- **Mesh `depth_trunc`: keep 1.0** (user chose). Note: the run's mesh was 12 KB / near-empty
  because TSDF discards depth >1 m; the refinement had dropped it from 10.0. If meshes look
  thin, revisit with the user.
- **No display decimation (500k)** chosen — but the PRIMARY fix (`clean_pointcloud`) changes
  this conversation; re-confirm with the user that cleaning to ~56k is acceptable (it matches
  the notebook).

## DEFERRED / UNRESOLVED

- **rclone push fails (exit 1)** to `fieldwork_processed` (`sources.py:push_outputs` →
  `RuntimeError: rclone copy failed (exit 1)`). Non-fatal (outputs are on local disk, push
  runs in a detached thread). Cause unknown — bucket perms / path / `--gcs-bucket-policy-only`.
  Not yet investigated.
- **"More than one camera is found"** warning during COLMAP build — VGGT-Omega emits per-frame
  intrinsics (36 PINHOLE cameras). Believed benign/expected; not confirmed as related to the
  distortion (the distortion is the missing `clean_pointcloud`).

## Key files

- `collab_splats/dashboard/pipeline.py` — run orchestrator; **add `clean_pointcloud` here or
  in the viewer**; `_sample` sampling path.
- `collab_splats/dashboard/viewer.py` — `load`, `_render_left/right`, `score_query`,
  `ensure_lifted`, `_decimate_indices` (display path to replace with cleaning).
- `collab_splats/dashboard/app.py` — `_load_outputs` job, `_on_run`, `view()`, `run_app`.
- `collab_splats/pointcloud/utils.py:208` — `clean_pointcloud` (returns `(pcd, index_mapping)`).
- `collab_splats/utils/frame_sampling.py` — `sample_frames_optical_flow:599`,
  `score_all_frames:401` (stride=5).
- Notebooks: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`,
  `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`.

## How to run

```bash
pkill -9 -f collab_splats.dashboard; pkill -9 Xvfb
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --port 7860
```
Tests: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -q`
