# perf-1k — one performance pass over the 1k windowed pipeline

Date: 2026-10-03 · Status: approved design (A2 + A3 added 2026-10-04), plan `docs/superpowers/plans/2026-10-03-perf-1k.md`

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
| 12 one decode thread per QA worker | A2 | `preproc/qa.py` |
| 13 frame handoff for vggtx + mapanything | A2 | `feedforward/base.py`, `vggtx.py`, `mapanything.py`, `reconstructor.py` |
| 14 PNG write overlaps the pointcloud stage | A3 | `reconstructor.py` |
| 15 Omega forward + LC profile, then overlap | A3 | `geometry/loop_closure/wrapper.py`, `vggt_omega.py` |
| 2 batched xfeat extract | vismatch-fork | `basis` branch of `/workspace/vismatch` |
| 3b BA load GPU unproject, drop dead re-lift | B | rgbd-ba |
| 8 sparse/GPU BA observation filter | B | rgbd-ba `bundle_adjustment.py` |
| 9 `match_batch` + vectorized depth filter | B | rgbd-ba `extractors.py` |
| 10 single process for imgs + tracks + BA | B | rgbd-ba driver / `Reconstructor` |
| 11 BA float32 vs float64 | B | rgbd-ba |

- Phase A: branch `perf/1k` off `clean/final`, worktree `.worktrees/perf-1k`, starts now
- Phase B: rgbd-ba, starts only after the user commits their uncommitted edits there
  (`bundle_adjustment.py`, `submap.py`, `graph.py`, `extractors.py`); rebase rgbd-ba onto
  `clean/final` + `perf/1k` first
- one commit per item, sequential; items 6+7 share `vggt_omega.py`, 3a+4 share LC files

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

## Phase A2 (added 2026-10-04, after the Phase A gate)

The Phase A gate measured preproc + pointcloud at ~530 s plain, not the profile's ~1200 s
(py-spy + load inflated it). Omega 533 → 456 s, vggtx 613 → 575 s, mapanything 599 → 579 s.
Two leftovers:

### 12. One decode thread per QA worker

- measured: QA 16 workers 151 s vs 4 workers 143 s, i.e. no gain
- cause (hypothesis): each worker's `iter_frames` runs PyAV `thread_type="AUTO"`, so 16
  processes x all-core decode threads oversubscribe; the selection-decode sweep showed the same
  (AUTO 74 s vs `threads=1` 49 s at 16 workers)
- change: QA worker passes `threads=1` to `iter_frames`
- gate: sweep workers 4/8/16/32 x AUTO/`threads=1` on GH010229, quiet box; ship `threads=1`
  only if faster, and `preproc.n_workers` follows the sweep's best; report bit-identical
  (`test_video_quality_workers_produce_an_identical_report` + a real-video report compare)
- if `threads=1` gives no gain either: revert `n_workers` to 4 and report the real bottleneck

### 13. Frame handoff for vggtx + mapanything

- measured: preprocess vggtx 134 s, mapanything 160 s on the file path (serial PNG read +
  resize), vs Omega 11.5 s with handoff + 8 threads
- `frames: dict[str, np.ndarray] | None` moves from `VGGTOmegaCreator` to
  `BaseFeedforwardCreator`; same contract: every `p.name` present → array path, otherwise a
  warning and the file path
- vggtx: array path mirrors `vggt.utils.load_fn.load_and_preprocess_images(mode="crop")`
  step for step, minus the file open, thread pool of 8
- mapanything: array path mirrors `mapanything.utils.image.load_images` for the creator's
  resize modes, minus the file open, thread pool of 8
- `Reconstructor` hands frames to vggt_omega, vggtx and mapanything; loger stays file-based
- test per backend: preprocess output (tensor / view dicts) and `original_coords` bit-exact vs
  the file path, plus a test that the array path is taken
- gate: 1k run per backend, A2 tip vs a same-code ref at the Phase A tip `7b2f1f1f`
  (keyframes equal, pose maxdiff ≤ 1e-4, equal point count), preprocess wall reported

## Phase A3 (added 2026-10-04, Omega priority)

Omega after A: 456 s = QA ~150, decode + PNG 80, load 17, preprocess 11.5, forwards + LC 182,
save 10. A3 targets the PNG write and the forwards + LC.

### 14. PNG write overlaps the pointcloud stage

- today `preproc` writes PNGs synchronously (`write_frames`, already compression 1, 8 threads):
  35–50 s, twice when `undistort` is on
- change: in-process runs submit the final `write_frames` to a background thread; `preproc`
  returns once frames are selected (and undistorted); the pointcloud stage consumes the in-memory
  frames; the write is joined before the pointcloud stage returns, and before any stage that
  reads `images/`
- undistort keeps its first synchronous write (`calibrate_camera` reads files)
- a write error re-raises at the join; leaf-only runs unchanged (no handoff, nothing to overlap)
- vggtx / mapanything get the overlap only once item 13 lands; loger keeps the synchronous write
- test: `images/` byte-identical to the synchronous write; a failing write raises from the run;
  the join happens before the pointcloud stage returns
- gate: Omega 1k vs a same-code ref, as A2; report the decode + PNG and pointcloud walls

### 15. Omega forward + LC: profile, then overlap

- step 1, measure only: per-window timeline on the 1k Omega run (GPU busy %, forward, retrieval,
  alignment, unproject/assemble, host<->device copies), py-spy + CUDA events, quiet box
- step 2, only where the profile shows GPU idle between windows: prepare window k+1 (slice,
  pin, H2D copy) and run window k's CPU alignment while window k+1's forward runs
- no dtype change: the Omega aggregator already autocasts internally; params stay fp32
- outputs bit-identical, same order of ops per window; any change that cannot be bit-exact is
  dropped, not loosened
- step 2 scope is set by step 1's findings and comes back for approval before implementation
- gate: Omega 1k vs same-code ref, as A2; forwards + LC wall reported

## Phase B outline (not implemented now)

- 3b: BA load uses GPU `unproject`; delete the dead re-unproject in the driver
- 8: `_filter_observations` projects only observed (frame, point) pairs on GPU instead of dense N x P CPU float64
- 9: tracks matching via `LocalMatcher.match_batch`; vectorized depth filter
- 10: imgs + tracks + BA in one process (saves ~2 x 25–30 s of imports)
- 11: BA `dtype` float32 vs float64 on chess 1000 — ATE vs GT and refine wall time; float32 ships
  only if ATE is within noise of float64

## Out of scope

- batched xfeat extract (vismatch-fork effort)
- changing outputs: any item that cannot meet its equivalence test is dropped, not loosened
