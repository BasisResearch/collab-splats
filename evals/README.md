# `evals/` — evaluation & parity harness

Compute scripts for reconstruction accuracy: ground-truth ATE/RPE evaluation of the
collab-splats pipelines (feedforward → BA → LC) and parity checking against upstream
**VGGT-SLAM**. **Compute runs in CLI/tmux only** — notebooks in `docs/` are for
visualization. Heavy inference/eval → tmux, one model at a time (46 GB cgroup cap).

```bash
PY=/opt/conda/envs/reconstruction/bin/python      # py3.11; NOT base conda
$PY evals/eval_gt.py --help
```

> **Note on the `evals` package name:** a same-named `evals` pip package is installed
> and shadows this dir at import time. Tests import runner modules via
> `sys.path.insert(0, ".../evals"); from runners.X import …` (see `tests/evals/`).

---

## Layout

```
evals/
  *.py            entry points + library (see tables below)
  runners/        SLAM wrappers, benchmark drivers, parity/diagnostic tools
  baselines/      committed reference results (frozen; see below)
  results/        gitignored scratch output of eval_gt runs
  data/           datasets (7-Scenes etc; large, gitignored)
  envs/           conda env files
```

---

## Entry points (active)

| file | role |
|---|---|
| `eval_gt.py` | **Main GT eval runner.** Feedforward backbone × condition (`baseline`/`lc`/`ba`/`ba_track-density-N`) → ATE/RPE/AUC vs GT. Each condition runs in its own subprocess (clean GPU). Flags: `--backbone --conditions --submap_size --lc_scale_method --keyframe_list --lc_layer --output_ate`. Writes `metrics.json` (ATE, RPE trans+rot, AUC@{5,15,30}), TUM trajectories, plots. |
| `eval_compare.py` | Phase-2 unified comparison runner over multiple methods/sequences. |
| `eval_suite.sh` | Shell driver bundling a standard eval set. |

## Library (imported by the harness)

| file | role |
|---|---|
| `datasets.py` | Dataset loaders: 7-Scenes, CO3Dv2, TUM association files. `get_dataset(name)`. |
| `ate_utils.py` | ATE vs 7-Scenes GT via `evo` (Sim3-aligned). |
| `metrics.py` | Thin `evo` ATE/RPE wrapper + `compute_auc` (TUM-file pose AUC). |
| `trajectory_io.py` | Trajectory read/write (TUM etc). |
| `reconstruction_quality.py` | Submap-alignment quality metrics for LC validation (used by `eval_gt` subprocess mode). |

## Dataset downloaders (utility)

`download_7scenes.py` · `download_7scenes.sh` · `download_co3dv2.sh` · `download_kitti.sh` · `download_tum.sh` · `download_waymo.sh` · `runners/extract_waymo.py` (tfrecord → flat layout).

## `runners/` — SLAM wrappers, benchmark drivers, parity tools

**Active:**
| file | role |
|---|---|
| `run_vggt_slam.py` | Subprocess wrapper around `third_party/VGGT-SLAM/main.py`. **Use its defaults for the published-matching anchor** (see handoff below). |
| `run_vggt_slam_lc.py` | Run VGGT-SLAM on a sequence → dense TUM + ATE + loop count. Produced the long SLAM ref / loop probe. |
| `run_disparity_sweep.py` | Disparity-sweep parity harness (ours vs SLAM); generated `baselines/disparity_sweep/`. |
| `parity_trace.py` | **Canonical** stage-by-stage parity trace (vggt_spark LC vs VGGT-SLAM). Localizes a diverging stage. Uses the solver-dump helpers below. |
| `our_solver_dump.py` · `vggt_slam_solver_dump.py` | Per-boundary solver-internals dumps consumed by `parity_trace.py` / `run_disparity_sweep.py`. |
| `run_cross_model_benchmark.py` | **(2026-05-31)** Serial `eval_gt` matrix over 4 backbones × framesets. |
| `build_benchmark_table.py` | **(2026-05-31)** Aggregate `cross_model/*/metrics.json` → markdown table. |

**Diagnostic / one-off (investigation-complete — see Deprecation report):**
`compare_slam_ours.py` · `compare_vggt_outputs.py` · `debug_lc_steps.py` · `compare_solver_internals.py` · `diagnose_lc_parity.py`.

## Other eval tools (standalone)

| file | role | status |
|---|---|---|
| `eval_similarity_calibration.py` | Sweep LC verify layer per backbone (DINO-SALAD pairs). Produced the per-model `_lc_layer_index` calibration. | keep (re-runnable tool) |
| `eval_vggt_slam_comparison.py` | 7-Scenes comparison: baseline / lc_se3 / lc_sl4 / vggt_slam_oob. | keep |
| `eval_multiview_conf.py` | Multiview-confidence eval across backbones (chess). | keep |
| `diag_pose_graph.py` | Compare our SL(4) pose-graph init vs VGGT-SLAM. | diagnostic (see report) |
| `check_ate_methods.py` | One-off: evo ATE vs our umeyama on SLAM poses. | diagnostic (see report) |
| `_ba_finding_eval.py` · `sweep_incremental_ba.py` · `plot_incremental_ba_sweep.py` · `incremental_ba_sweep*.png` | Incremental-BA add_size sweep investigation (ATE vs runtime). | one-off (see report) |

---

## `baselines/` — committed reference results

| dir | what |
|---|---|
| `disparity_sweep/{slam,ours}_d{10,20,30,50}/` | VGGT-SLAM vs ours frozen baselines at min_disparity 10–50 (chess, max_frames 200). `slam_*/metrics.json` = SLAM ATE; `selected_frames.txt` = identical-frame lists. **plus** `slam_d5_long/` (384 frames, 21 loops — generated 2026-05-31). |
| `cross_model/` | Cross-model benchmark (2026-05-31). One dir per `<backbone>__<frameset>__<sm>/` with `metrics.json` + `ate.json` + TUM. `_gate/` sanity gate, `_layersweep/` omega LC-layer sweep, `_core_matrix_table.md`. Heavy COLMAP/ply/plots/npz **gitignored** (see `.gitignore`). |
| `vggt_slam/chess_seq01/` | Raw VGGT-SLAM run output (dense TUM, similarity scores). |
| `results/parity_harness/` | Parity-trace intermediate dumps. |

---

## Tests / benchmarks performed

- **LC pose-extraction parity fix** (commit `1372ac2`): vggt_spark now matches VGGT-SLAM
  baseline numerically. Trail: `docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-findings.md`.
- **Cross-model LC benchmark** (chess, 2026-05-31): 4 backbones × {single-pass, windowed
  baseline, lc} × {d10, d5_long}. Results + analysis:
  `docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md`.
  Headline: windowing is free; LC never helps on chess (no-op or catastrophic once real
  loops close — mechanism-level, not layer-tunable); `vggt_omega` best baseline.
- **Unit tests:** `tests/evals/` (runner, table, eval_gt helpers, AUC) and
  `tests/pointcloud/` (AUC metric, LC eval). Run: `$PY -m pytest tests/`.
  Known pre-existing failures: `docs/known-test-failures.md`.

## ⚠️ Before extending the benchmark

This benchmark ran at **off-default** params (`min_disparity` 5/10/20, frame caps).
VGGT-SLAM `main.py` default is `min_disparity=50` on the **full** folder. To compare to
published numbers / issue [VGGT-SLAM#43], **first reproduce VGGT-SLAM defaults with
VGGT-SPARK** (the anchor), then swap backbones on identical config. Full instructions:
`docs/superpowers/specs/2026-05-31-cross-model-benchmark-handoff.md`.

---

## Deprecation report (pending owner decision)

Standalone investigation scripts, work complete, **0 imports** from active code. Listed
for the owner to decide keep / archive / remove — **not yet removed**.

| file | what it did | recommend |
|---|---|---|
| `runners/compare_slam_ours.py` | LC-parity debug: side-by-side our-vs-SLAM trajectory compare. Superseded by `parity_trace.py`. | archive/remove |
| `runners/compare_vggt_outputs.py` | Compared VGGT extrinsics under our vs SLAM image preprocessing. One-off; preprocessing parity confirmed (Δ=0). | archive/remove |
| `runners/debug_lc_steps.py` | Step-by-step PGO trace for the d=10 case. Served the pose-extraction fix; superseded by `parity_trace.py`. | archive/remove |
| `runners/compare_solver_internals.py` | Per-boundary solver-internals diff. Subsumed by `parity_trace.py`. | archive/remove |
| `runners/diagnose_lc_parity.py` | Per-frame trajectory parity vs a SLAM TUM. Subsumed by `parity_trace.py`. | keep-or-archive |
| `diag_pose_graph.py` | SL(4) pose-graph init comparison. One-off during the LC fix. | archive/remove |
| `check_ate_methods.py` | Verified evo ATE == our umeyama on SLAM poses. One-off check, passed. | archive/remove |
| `_ba_finding_eval.py` | Ad-hoc BA A/B harness (vis_thresh, track budget, fine_tracking). | keep-or-archive |
| `sweep_incremental_ba.py` + `plot_incremental_ba_sweep.py` + `incremental_ba_sweep*.png` | Incremental-BA add_size vs runtime sweep + plots. Findings folded into `eval_gt` `ba` conditions. | archive/remove (move PNGs out of source) |

**Note:** `our_solver_dump.py` / `vggt_slam_solver_dump.py` look like dumps but are **used
by** `parity_trace.py` — **keep**.
