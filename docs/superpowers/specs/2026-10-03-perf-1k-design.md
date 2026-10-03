# perf-1k — one performance pass over the 1k windowed pipeline

Date: 2026-10-03 · Status: approved design, plan `docs/superpowers/plans/2026-10-03-perf-1k.md`

## Goal

Cut wall time of the ~1000-frame GH010229 run (preproc → windowed LC pointcloud → tracks → BA)
without changing its output.

- baseline: py-spy profile of 2026-10-03, `perf/inplace` (already holds 79089b65, c77b9f19, 54af1274)
- baseline total ~2350 s; estimate after both phases ~900–1000 s

## Baseline (main-thread inclusive seconds, 984 frames)

| Stage | s | Hot spot |
|---|---|---|
| QA | 471 | ProcessPool `n_workers=4` on 96 cores |
| sampling decode | 136 | sequential decode via `iter_frames(indices=...)` |
| model load | 88 | `VGGTOmega()` CPU random init, then `torch.load` + `.to` |
| Omega preprocess | 158 | PNG re-read 62, `Image.open` size probe 21 |
| LC dense unproject | 72 | vggt numpy `unproject_depth_map_to_point_map`, `wrapper.py:287` |
| LC assemble | ~170 | `get_world_grid` float64 numpy, run twice; dense cloud built then capped |
| tracks extract | 285 | per-image vismatch xfeat forward |
| BA load / refine / re-unproject | 51 / 134 / 70 | dense N x P CPU filter 44; re-unproject dead under lean store |

## Phases

| Item | Phase | Where |
|---|---|---|
| 1 QA workers | A | `configs/base.yaml` |
| 3a LC GPU unproject | A | `geometry/loop_closure/wrapper.py` |
| 4 LC assemble once, GPU, cap first | A | `submap.py`, `map.py`, `wrapper.py` |
| 5 parallel seek-decode for sampling | A | `preproc/sampling.py` |
| 6 in-memory frame handoff | A | `reconstructor.py`, `vggt_omega.py` |
| 7 device-init Omega load | A | `vggt_omega.py` |
| 2 batched xfeat extract | vismatch-fork | `basis` branch of `/workspace/vismatch` |
| 3b BA load GPU unproject, drop dead re-lift | B | rgbd-ba |
| 8 sparse/GPU BA observation filter | B | rgbd-ba `bundle_adjustment.py` |
| 9 `match_batch` + vectorized depth filter | B | rgbd-ba `extractors.py` |
| 10 single process for imgs + tracks + BA | B | rgbd-ba driver / `Reconstructor` |
| 11 BA in float32, chess GT check | B | rgbd-ba |

- Phase A: branch `perf/1k` off `clean/final`, worktree `.worktrees/perf-1k`, starts now
- Phase B: rgbd-ba, starts only after the user commits their uncommitted edits there
  (`bundle_adjustment.py`, `submap.py`, `graph.py`, `extractors.py`); rebase rgbd-ba onto
  `clean/final` + `perf/1k` first
- one commit per item, sequential; items 6+7 share `vggt_omega.py`, 3a+4 share LC files

## Precision policy: float32 everywhere it costs time

- every large-array compute runs float32: dense grids, unprojection, BA, observation filtering
- float32 matmuls force full fp32 (`set_float32_matmul_precision("highest")`, restored after);
  TF32, which a mapanything import enables process-wide, would round them
- float64 stays only where it buys nothing to drop it, or a library demands it:
  - library boundaries that take double only: gtsam SL4 (`loop_closure/graph.py`), pycolmap,
    open3d `Vector3dVector` (SOR, TSDF integrate, ply)
  - per-pose 4x4 / 3x3 algebra (N x 16 numbers): no measurable time; homography chains compose
    over ~1000 frames
  - `metrics.py` histogram edges and NCC sums: documented float32 misbinning
- Phase A: items 3a and 4 (LC); Phase B: BA (items 3b, 8, 11)

## Phase A mechanics

### 1. QA workers

- `preproc.n_workers` default 4 → 16 in `configs/base.yaml`
- exactness already covered by `test_video_quality_workers_produce_an_identical_report`
- gate: timing only

### 5. Parallel decode for sampling

- `_decode_selection` splits `chosen` into `n_workers` contiguous groups
- each thread: `iter_frames(start=g[0], count=g[-1] - g[0] + 1)`, keeps frames in `g`
- same range-seek path QA uses; threads because PyAV decode releases the GIL and process
  results would pickle ~6 GB
- `n_workers` comes from the same `preproc.n_workers`
- test: frames bit-exact vs the single-pass decode, fixture video and the 1k selection

### 3a. LC GPU unproject

- `wrapper.py:287`: `unproject_depth_map_to_point_map` → `collab_splats.geometry.projection.unproject`,
  float32 on GPU (TF32 forced off, as `transform_points` does), chunks of 64 frames
- measured on 984 x 384 x 688: 49.8 s → 10.1 s, maxdiff 3.8e-6
- test: rtol/atol 1e-5 vs the vggt numpy path; bit-identical with global TF32 on

### 4. LC assemble once, on GPU, cap first

- world grid per submap computed once (float32 on GPU, TF32 forced off), reused for cloud and per-frame depth;
  today `get_world_grid` runs in `map.get_world_pointcloud` and again in `_assemble_result`
- cap first: `subsample_points` draws from an all-True mask, so the draw depends only on the
  total count; compute the count from skip/conf masks, draw the same mask with the same seed,
  gather only drawn points from each submap's grid (grid still needed whole for depth); no dense
  cloud stack; `GraphMap.get_world_pointcloud` deleted (no caller left)
- drop the dense `world_points` array `_assemble_result` builds; the lean store does not keep it
- test: selected indices bit-exact; points, depth rtol/atol 1e-5 vs the current float64 path

### 6. In-memory frame handoff

- `Reconstructor` keeps preproc's decoded RGB frames; a same-process pointcloud stage passes
  them to the creator beside `image_paths`
- `VGGTOmegaCreator._preprocess` crops/resizes from arrays: no size probe, no PNG re-read
- PNGs are still written; leaf re-runs stay file-based
- handoff holds the final arrays (post-undistort), i.e. the same pixels the PNGs hold
- no handoff when: leaf-only run, other backends
- frames released once the pointcloud stage returns
- test: preprocess tensor and `original_coords` bit-exact vs the file path

### 7. Device-init Omega load

- `with torch.device(device): model = VGGTOmega()`, then
  `load_state_dict(torch.load(ckpt, mmap=True, weights_only=True))`
- chosen over meta-init: meta leaves non-persistent buffers unmaterialized
- test: every param and buffer bit-exact vs the old load path

## Acceptance

- per item: its equivalence test above, plus the module's tests, run from the worktree
  (`cd .worktrees/perf-1k && PYTHONPATH=$PWD`, print `collab_splats.__file__`)
- end to end: one 1k run of `gh_f1000.py` from the worktree
  - output `/workspace/outputs/ocr_viewer/GH010229_f1000_perf` (`/tmp` has 13 G free, store is 16 G);
    deleted after the gate
  - vs `GH010229_f1000`: keyframe indices identical, pose maxdiff ≤ 1e-4, point count equal
  - peak RSS via `/usr/bin/time -v` under 40 GB; over that, add a frame-count cutoff for the handoff
  - report: stage wall table before vs after
- final: `pytest tests/preproc tests/pointcloud tests/geometry`, exit code read directly, no
  concurrent heavy jobs
- never write into `GH010229` or `GH010229_f1000`

## Commits (Phase A, `perf/1k`)

0. `fix(pointcloud): vggt_omega _preprocess` — owed fix, uncommitted in the main checkout
   (`vggt_omega.py` + `tests/pointcloud/test_vggt_omega_creator.py`); carried onto `perf/1k`
   as a patch, main-checkout copies reverted only after the commit lands
1. `perf(preproc): 16 quality-report workers`
2. `perf(preproc): parallel range-seek decode for selection`
3. `perf(geometry): GPU float32 unproject in LC submaps`
4. `perf(geometry): world grid once on GPU, cap before stacking`
5. `perf(pointcloud): in-memory frame handoff to Omega`
6. `perf(pointcloud): device-init Omega load from mmap`

## Phase B outline (not implemented now)

- 3b: BA load uses GPU `unproject`; delete the dead re-unproject in the driver
- 8: `_filter_observations` projects only observed (frame, point) pairs on GPU instead of dense N x P CPU float64
- 9: tracks matching via `LocalMatcher.match_batch`; vectorized depth filter
- 10: imgs + tracks + BA in one process (saves ~2 x 25–30 s of imports)
- 8 runs float32 on GPU (TF32 forced off)
- 11: BA in float32 (rgbd-ba `BundleAdjustmentConfig.dtype` already defaults to it; drivers stop
  forcing float64). Check on chess 1000: ATE vs GT and refine wall time, float32 vs float64;
  an ATE regression beyond noise is reported, not silently reverted

## Out of scope

- batched xfeat extract (vismatch-fork effort)
- changing outputs: any item that cannot meet its equivalence test is dropped, not loosened
