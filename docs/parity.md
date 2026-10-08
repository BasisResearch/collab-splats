# Parity references: VGGT-SPARK and VGGT-SLAM

The one place the two external parity references are described.

- our loop closure (`collab_splats/geometry/loop_closure/`) ports VGGT-SLAM's SL(4) pose graph
- both references were used only to check that port against upstream numbers
- neither was ever part of the shipped pipeline; both are removed from the repo
- `Source:` paths are local run output under `evals/results/` (gitignored); the committed
  copies, then under `evals/baselines/`, are at `2c0b2f06`

---

## The two references

| reference | repo | pinned commit | role |
|---|---|---|---|
| VGGT-SPARK | [MIT-SPARK/VGGT_SPARK](https://github.com/MIT-SPARK/VGGT_SPARK) | `6e6e16107b88e8e76c751826af10d4295d87ecd2` | VGGT-1B fork: `forward(compute_similarity=True)` returns `image_match_ratio`; VGGT-SLAM's backbone |
| VGGT-SLAM | [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) | `fd3fd218d1fc5edb1a7ea3490263b9aec3b96300` | reference SLAM system (VGGT-SLAM 2.0, incl. the `vggt-slam2` update); BSD-2-Clause |

Pin caveats:

- VGGT-SPARK `6e6e161`: an upstream commit (2026-02-13, "small update to readme instructions")
- VGGT-SLAM `fd3fd218` (2026-06-04, "removing unused files and adding bug fix for loop closure verification"): previously pinned to a local commit 604efe8, re-pinned 2026-09-26
- our port attributions cite `MIT-SPARK/VGGT-SLAM @ fd3fd218, <path>:<lines>`; every line range is checked against that commit

---

## What we port from VGGT-SLAM

All paths are at `MIT-SPARK/VGGT-SLAM @ fd3fd218`, used under its BSD-2-Clause license (Copyright (c) 2025, MIT-SPARK).

| our module | upstream source |
|---|---|
| `geometry/loop_closure/graph.py` `PoseGraph` | `vggt_slam/graph.py:PoseGraph` (SL4 backend: `BetweenFactorSL4`, sigma 0.05 edges, 1e-6 first-frame prior, `graph.py:19-23`); edge construction from `vggt_slam/solver.py:118-195` (`add_edge`) |
| `geometry/transforms.py` `decompose_camera` | `vggt_slam/slam_utils.py:45-83` (`decompose_camera`): R from the default branch (`:77`), t from the `no_inverse=True` branch (`:80`) |
| `graph.py` `calculate_pairwise_frame_scale` | `vggt_slam/scale_solver.py:estimate_scale_pairwise` |
| `graph.py` loop-edge chain + confidence fallback | `vggt_slam/solver.py:270-287` (LC submap + `add_edge` pair), `:129-146` (anchor scale), `:129-138` (confidence fallback) |
| `geometry/loop_closure/submap.py` `Submap` | `vggt_slam/submap.py:Submap` (fat: dense points/colors/conf) |
| `geometry/loop_closure/map.py` `GraphMap` | `vggt_slam/map.py:GraphMap` |
| `geometry/loop_closure/matching.py` | `vggt_slam/loop_closure.py` (`LoopMatch`, `LoopMatchQueue` → `find_loop_closures`) + `vggt_slam/map.py:retrieve_best_score_frame`, per decision 015 |
| `geometry/loop_closure/wrapper.py` `LoopClosure` | `main.py:109-130` (window loop) + `vggt_slam/solver.py` `Solver` structure (`run_predictions` / `add_points`); graph construction stays in `PoseGraph` |
| `pointcloud/utils.py` `cross_frame_attention_ratio` | VGGT-SPARK `get_similarity()` + top-quarter mean `r[r >= np.percentile(r, 75)].mean()` |

Config defaults that mirror VGGT-SLAM (`LoopClosureConfig`, `wrapper.py`):

- `submap_overlap=1`, `conf_percentile=25.0`, `lc_retrieval_threshold=0.95`
- `conf_percentile` is upstream's `--conf_threshold 25` (`main.py:32`): each submap's gate is `percentile(depth_conf, 25) + 1e-6` (`vggt_slam/submap.py:36-41`); until 2026-09-26 it was `conf_threshold`, a raw gate on the stride-8 points
- scale inputs follow `vggt_slam/solver.py:129-146` on density and gate: dense per-pixel points, gated by the prior submap's threshold (`:131`)
- kept different, on purpose: the point frame; see [Scale point frame](#scale-point-frame-a-measured-divergence) below
- kept different: `calculate_pairwise_frame_scale` drops points with `|x| < 1e-8` (upstream `scale_solver.py:15-25` has no filter); loop anchors use each frame's own K where upstream uses `proj_mats[-1]` / `proj_mats[0]` (`solver.py:140`)
- inter-submap scale is always `rotation_only`, VGGT-SLAM's method; the `scale_method` field was deleted in geometry-round3 (2026-09-26) along with its `none` alternative

### Scale point frame: a measured divergence

Upstream reads both frames' points in their own submap's frame-0 camera; ours moves each into its own camera first.

- upstream: `t1` / `t2` are `get_frame_pointcloud(...)` as stored (`vggt_slam/solver.py:141-142`), i.e. unprojected in frame 0 (`:222`)
- ours: each frame's points go through that frame's world-to-cam pose (`graph.py` `calculate_pairwise_frame_scale`, via `transforms.transform_points`), both frames, sequential edge and loop anchors alike
- why: in frame 0 the two norms share no origin, so the ratio is `|R p + t| / |p|`, biased by the camera's translation inside the submap
- upstream's default `--submap_size 16` (`main.py:28`) keeps that translation small; ours run 20 (`LoopClosureConfig`), 64 (`configs/loop_closure.yaml`) and 50 (the chess eval below)
- history: the back-transform predates 2026-09-26; commit `65d2def3` removed it to match upstream, the next commit restored it after the ATE below

7-Scenes chess seq-01, 500 frames, `submap_size=50`, ATE (m, Sim3-aligned RMSE), `baseline` / `lc`:

| backbone | before `65d2def3` | frame-0 read (`65d2def3`) | overlap-camera read (current) |
|---|---|---|---|
| vggtx | 0.0457 / 0.0327 | 0.108 / 0.133 | 0.0471 / 0.0324 |
| vggt_omega | 0.0154 / 0.0149 | 0.149 / 0.115 | 0.0152 / 0.0152 |
| mapanything | 0.1306 / 0.0406 | 0.100 / 0.094 | 0.1225 / 0.0367 |

- the frame-0 read regresses vggtx and vggt_omega 2-10x in both conditions, baseline (sequential edges only) included, and mapanything `lc` 2.3x; mapanything `baseline` alone improves (0.131 -> 0.100)
- current differs from "before" only through the dense points and the percentile gate, kept from `65d2def3`

---

## What was compared, and how

Method:

- run VGGT-SLAM (own torch 2.3.1 venv, `setup/vggt_slam.sh`) → TUM trajectory, keyframe list, loop count
- run our pipeline on the **exact keyframe list** VGGT-SLAM selected (removes keyframing as a confounder)
- backbone `vggt_spark` = same weights as upstream → the only apples-to-apples arm
- other backbones (`vggtx`, `vggt_omega`, `mapanything`) run the same keyframes, reported not gated
- ATE = Sim3-aligned RMSE vs GT, meters

Conditions:

- `baseline`: windowed submaps, LC off; `lc`: loop closure on
- matched config: `submap_size=16`, `submap_overlap=1`, `max_loops=1`/submap, `conf_threshold=25` (then a raw gate; see the defaults above), `lc_thres=0.95`
- `min_disparity` = VGGT-SLAM keyframe tracker threshold (paper default 50; 5 = loop-rich probe)
- `@25%` / `@50%` = keyframe-prefix arms (scaling / false-loop negatives)

Gates (parity harness):

- SPARK arm: ATE within 5% or 5 mm of VGGT-SLAM, loop counts equal → `PASS` / `FAIL`
- other arms: `HARMLESS` if `ate_lc ≤ max(1.05 × ate_baseline, ate_baseline + 0.005)`, else `HARMFUL`

Datasets:

- 7-Scenes seq-01: chess, office, redkitchen
- TUM RGB-D: fr3/long_office_household (`tum_fr3_office`)

---

## Headline results

All baseline files cited below were removed from the repo by the commit
`chore(evals): drop VGGT-SPARK / VGGT-SLAM baseline outputs`; recover them with
`git show <that commit>^:<path>`. Numbers are copied verbatim.

### VGGT-SLAM reference runs (chess seq-01)

| source | min_disparity | keyframes | submaps | loops | ATE (m) |
|---|---|---|---|---|---|
| `evals/results/lc_parity/7s_chess/slam/metrics.json` | 50 | 29 | 2 | 0 | 0.0389 |
| `evals/results/lc_parity_d5/7s_chess/slam/metrics.json` | 5 | 384 | 45 | 21 | 0.0455 |
| `evals/results/vggt_slam/chess_seq01/metrics.json` | 5 | 384 | 45 | 21 | 0.0464 |

- ATE rounded to 4 dp from `ate_rmse`
- the two d5 rows are separate runs recorded by different drivers (parity harness vs `run_vggt_slam.py`, `max_frames` null vs 1000)

### Pose-extraction parity (windowed baseline, chess, VGGT-SLAM keyframes)

Source: `evals/results/cross_model/_core_matrix_table.md` (SPARK rows, removed; `Δslam` = ATE minus VGGT-SLAM's)

| backbone | frameset | cond | ATE | RPE-t | RPE-r° | AUC5 | AUC15 | AUC30 | Δslam |
|---|---|---|---|---|---|---|---|---|---|
| vggt_spark | slam_d10 | baseline | 0.0174 | 0.0455 | 0.52 | 46.0 | 79.7 | 89.7 | -0.0003 |
| vggt_spark | slam_d10 | lc | 0.0174 | 0.0455 | 0.52 | 46.0 | 79.7 | 89.7 | -0.0003 |
| vggt_spark | slam_d5_long | baseline | 0.0441 | 0.0239 | 0.47 | 24.1 | 70.5 | 84.9 | -0.0022 |
| vggt_spark | slam_d5_long | lc | 0.0441 | 0.0239 | 0.47 | 24.1 | 70.5 | 84.9 | -0.0022 |

- after the `R` vs `Rᵀ` pose-extraction fix (`1372ac28`), SPARK baseline matches VGGT-SLAM within 3 mm
- `lc` = `baseline` here: LC was a no-op for SPARK before the 2026-07 fixes

### Disparity sweep (chess, `max_frames` 200)

Source: `evals/results/disparity_sweep/{slam,ours}_d*/` (removed; ours = `vggt_spark`, `--lc_scale_method none`; measured before geometry-round3 deleted `scale_method`, so `none` can no longer be rerun)

| min_disparity | SLAM keyframes | SLAM ATE (m) | ours baseline ATE (m) | ours lc ATE (m) |
|---|---|---|---|---|
| 10 | 26 | 0.0176 | — | — |
| 20 | 12 | 0.0187 | 0.0184 | 0.0184 |
| 30 | 8 | 0.0187 | 0.0196 | 0.0196 |
| 50 | 5 | 0.0224 | 0.0238 | 0.0238 |

- all ATE rounded to 4 dp from the JSON files; SLAM closed 0 loops at every level
- ours d10 omitted: baseline only, 17x the SLAM ATE, and the file does not record which code version produced it

### LC parity probe, chess d5 — before and after the loop-edge fixes

Pre-fix source: `evals/results/lc_parity_d5/_parity_table.md`

| scene | kf | submaps | SLAM ATE | SLAM loops | ours ATE (base) | ours ATE (lc) | ours loops | base gate | lc gate |
|---|---|---|---|---|---|---|---|---|---|
| 7s_chess [mapanything] | 384 | 45 | 0.0455 | 21 | 0.1927 | 0.1927 | 0 | — | HARMLESS |
| 7s_chess [vggt_omega] | 384 | 45 | 0.0455 | 21 | 0.0300 | 0.6255 | 20 | — | HARMFUL |
| 7s_chess [vggt_spark] | 384 | 45 | 0.0455 | 21 | 0.0442 | 0.0442 | 0 | PASS | FAIL |

Post-fix source: `evals/results/lc_parity_d5_postfix/_parity_table.md`

| scene | kf | submaps | SLAM ATE | SLAM loops | ours ATE (base) | ours ATE (lc) | ours loops | loop P | loop R | base gate | lc gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7s_chess [mapanything] | 384 | 45 | 0.0455 | 21 | 0.1467 | 0.0555 | 21 | 1.00 | 0.22 | — | HARMLESS |
| 7s_chess [vggt_omega] | 384 | 45 | 0.0455 | 21 | 0.0300 | 0.0189 | 21 | 1.00 | 0.22 | — | HARMLESS |
| 7s_chess [vggt_spark] | 384 | 45 | 0.0455 | 21 | 0.0442 | 0.0421 | 21 | 1.00 | 0.22 | PASS | PASS |
| 7s_chess [vggtx] | 384 | 45 | 0.0455 | 21 | 0.0422 | 0.0417 | 21 | 1.00 | 0.22 | — | HARMLESS |

- pre-fix: SPARK and MapAnything dropped every accepted loop (no joint poses); omega's inverted, unscaled loop edge wrecked the graph
- fixes: verify returns geometry; scale-reconciled 3-edge loop chain (VGGT-SLAM `add_edge`)
- write-up: `docs/superpowers/specs/2026-07-09-lc-parity-probe-results.md`

### Multi-scene matrix, main runs (paper config, `min_disparity` 50)

Source: `evals/results/lc_parity_matrix/_parity_table.md` (removed)

| scene | kf | submaps | SLAM ATE | SLAM loops | ours ATE (base) | ours ATE (lc) | ours loops | loop P | loop R | base gate | lc gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7s_chess [vggt_spark] | 29 | 2 | 0.0389 | 0 | 0.0392 | 0.0392 | 0 | — | — | PASS | PASS |
| 7s_office [vggt_spark] | 58 | 6 | 0.1056 | 2 | 0.1130 | 0.1071 | 2 | 1.00 | 0.67 | FAIL | PASS / scale:BLOWUP |
| 7s_redkitchen [vggt_spark] | 43 | 4 | 0.0543 | 1 | 0.0544 | 0.0517 | 1 | 1.00 | 1.00 | PASS | PASS / scale:BLOWUP |
| tum_fr3_office [mapanything] | 75 | 7 | 0.0319 | 2 | 0.1296 | 0.0623 | 2 | 1.00 | 0.67 | — | HARMLESS / scale:BLOWUP |
| tum_fr3_office [vggt_omega] | 75 | 7 | 0.0319 | 2 | 0.0510 | 0.0377 | 2 | 1.00 | 0.67 | — | HARMLESS / scale:OK |
| tum_fr3_office [vggt_spark] | 75 | 7 | 0.0319 | 2 | 0.0450 | 0.0320 | 2 | 1.00 | 0.67 | FAIL | PASS / scale:OK |
| tum_fr3_office [vggtx] | 75 | 7 | 0.0319 | 2 | 0.0427 | 0.0303 | 2 | 1.00 | 0.67 | — | HARMLESS / scale:OK |

- loop precision 1.00 on every backbone, scene and prefix arm: no false loop accepted
- `@25%` / `@50%` prefix arms (all in the source table): zero false loops accepted
- one HARMFUL main run: 7s_redkitchen [mapanything], 0.0501 → 0.0740 (GT-true loop, scale blow-up on correction)

### SPARK native loop-verify scores (chess d5 positives)

Source: `evals/results/parity_harness/vggt_spark_similarity.json` (removed)

| positives | min `image_match_ratio` | max | ≥ 0.95 |
|---|---|---|---|
| 21 | 0.9819 | 1.0634 | 21 |

---

## Calibration constants that come from VGGT-SPARK

| constant | value | origin | where it lives now |
|---|---|---|---|
| verify layer `_lc_layer_index` | 20 | SPARK `target_layer=20` of 24 global blocks; `-1` scores ~0.66 vs ~1.02 | removed; each LC-capable creator sets its own |
| verify threshold `default_verify_match_ratio` | 0.85 | SPARK's acceptance threshold for the attention ratio on VGGT-1B | removed; each LC-capable creator sets its own |
| similarity aggregation | mean of top-25% | SPARK `get_similarity()` + `r[r >= np.percentile(r, 75)].mean()` | `cross_frame_attention_ratio`, `pointcloud/utils.py` |
| native verify threshold | 0.95 | SPARK `image_match_ratio` (native head); 21/21 positives accepted, AUC 0.92 | removed with the `vggt_spark` creator |

- the `BaseFeedforwardCreator` LC classvars default to `None`; `_verify_loop_candidate` raises when they are unset
- calibrated per creator: `vggtx` (10 / 1.17), `vggt_omega` (13 / 1.55), `mapanything` (4 / 1.46)
- `loger` sets none, so LC refuses it
- per-backbone calibration method: clean-negative sweep (21 SLAM-confirmed positives vs 20 GT-clean negatives, seed 42), `evals/scripts/eval_similarity_calibration.py` (removed in `4b5b02bf`; last at `2c0b2f06`)

---

## Why the references were removed

- never part of the shipped pipeline: SPARK was a diagnostic `vggt_spark` backend, VGGT-SLAM an external run in its own venv
- parity is established: SPARK's `lc` gate is PASS on all four main scenes above
- both were cloned sources outside the lock; VGGT-SLAM also needed its own torch 2.3.1 venv
- to reproduce: check out the pinned commits above, plus a collab-splats commit before this removal
  (it still has `vggt_spark_creator.py`, `evals/scripts/run_vggt_slam.py`, `setup/vggt_slam.sh`)
