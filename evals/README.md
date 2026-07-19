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
| `run_disparity_sweep.py` | Disparity-sweep parity harness (ours vs SLAM) for any backbone (`--backbone`, default `vggt_spark`); generated `baselines/disparity_sweep/`. |
| `run_cross_model_benchmark.py` | Serial `eval_gt` matrix over any backbones/framesets (`--backbones`, `--single_framesets`, `--windowed_framesets`, `--submap_size`); defaults reproduce the 2026-05-31 4-backbone matrix. |
| `build_benchmark_table.py` | Aggregate `cross_model/*/metrics.json` → markdown table (`--reference_backbone`, default `vggt_spark`). |
| `compare_loop_edges.py` | Loop-edge composition diff (ours vs SLAM). Kept: `compose_slam_chain` imported by `tests/geometry/loop_closure/test_loop_edge_chain.py`. |

## Other eval tools (standalone)

| file | role | status |
|---|---|---|
| `eval_similarity_calibration.py` | Sweep LC verify layer per backbone (DINO-SALAD pairs). Produced the per-model `_lc_layer_index` calibration. | keep (re-runnable tool) |
| `eval_multiview_conf.py` | Multiview-confidence eval across backbones (chess). | keep |

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

## Investigations — scripts grouped by what they probed

Grouped by topic so the "why" behind each investigation is discoverable. A and B are
closed investigations (finding fixed / parameter absorbed elsewhere) whose scripts have
been deleted — the outcome and trail are what has ongoing value. C and D's scripts are
still retained and re-runnable; several compute paths from their directory depth
(`Path(__file__).parents[N]`), so moving them would silently break path resolution
(the same class of bug as the `_VGGT_SPARK_ROOT` off-by-one fixed 2026-05-31).

### A. LC ↔ VGGT-SLAM parity (pose-extraction fix, commit `1372ac2`)
Goal: find why our LC trajectory diverged 17× from VGGT-SLAM. Root cause = `R` vs `Rᵀ`
in pose extraction. **Outcome:** fixed; vggt_spark baseline now matches SLAM. The
investigation scripts (`parity_trace.py` and its solver-dump/comparison helpers) are
deleted — bug fixed, no ongoing regen value. Trail:
`docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-findings.md`.

### B. Bundle-adjustment tuning
Goal: pick BA track-density / increment params. **Outcome:** folded into `eval_gt` `ba`
and `ba_track-density-N` conditions — this capability is now native to the main runner,
so the sweep scripts are deleted. CO3Dv2 notes in `EVAL_NOTES.md`.

### C. LC similarity / verify-layer calibration
| script | what it probed |
|---|---|
| `eval_similarity_calibration.py` | Per-backbone LC verify layer (DINO-SALAD pairs) → `_lc_layer_index`. See memory `project_lc_layer_calibration`. |
| `eval_multiview_conf.py` | Multiview-confidence comparison across backbones. |

### D. Cross-model benchmark (2026-05-31)
Goal: rank backbones; separate windowing cost from LC benefit. **Outcome:**
`docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md`. The frozen
2026-05-31 numbers live under `baselines/cross_model/`; to reproduce or extend
them for a different backbone/frameset combination, use the generalized scripts:

| script | what it probed |
|---|---|
| `runners/run_cross_model_benchmark.py` | Serial `eval_gt` matrix — any `--backbones` × `--single_framesets`/`--windowed_framesets` (defaults reproduce the original 4-backbone matrix). |
| `runners/build_benchmark_table.py` | Aggregate `cross_model/*/metrics.json` → markdown (`--reference_backbone`). |
| `runners/run_disparity_sweep.py` | ours-vs-SLAM at min_disparity 10–50 for any `--backbone` → `baselines/disparity_sweep/`. |
| `runners/run_vggt_slam.py` · `runners/run_vggt_slam_lc.py` | VGGT-SLAM wrappers (anchor + long ref / loop probe). |

> **Housekeeping:** if bundle-adjustment tuning is revisited, prefer writing
> plots to the gitignored `results/` rather than the source tree.
