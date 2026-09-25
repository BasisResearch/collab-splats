# `evals/` — evaluation harness

Compute scripts for reconstruction accuracy: ground-truth ATE/RPE evaluation of the
collab-splats pipelines (feedforward → BA → LC). **Compute runs in CLI/tmux only** — notebooks in `docs/` are for
visualization. Heavy inference/eval → tmux, one model at a time (46 GB cgroup cap).

```bash
PY=/opt/conda/envs/reconstruction/bin/python      # py3.11; NOT base conda
$PY evals/scripts/eval.py --help
```

> **Note on the `evals` package name:** a same-named `evals` pip package is installed
> and shadows this dir at import time. Tests import script modules via
> `sys.path.insert(0, ".../evals/scripts"); from scripts.X import …` (see `tests/evals/`).

---

## Running an evaluation

`scripts/eval.py` is the one runner. Interpreter: `/opt/venv/reconstruction/bin/python` (py3.11) — run in **tmux**, one model at a time (heavy GPU, 46 GB cap). `results/` is gitignored scratch.

```bash
PY=/opt/venv/reconstruction/bin/python
```

**1 — one model, given parameters (goal: run a model + params → metrics).**
Each condition runs in its own subprocess (clean GPU). Metrics land in `<output_dir>/metrics.json` (per-condition ATE, RPE trans+rot, AUC@{5,15,30}, `n_loops_applied`) + TUM trajectories + plots.

```bash
$PY scripts/eval.py \
    --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
    --backbone vggt_omega --conditions baseline lc \
    --submap_size 16 --max_frames 500 \
    --output_dir results/omega_chess
```
- `--backbone`: `vggt_omega | vggtx | mapanything | loger`. Per-model LC layer + verify-threshold calibrations are applied automatically.
- `--conditions`: `baseline | ba | lc | ba_track-density-N | incremental_ba-N`.
- `--lc_scale_method` (default **`rotation_only`**; also `none`). **Keep the default for numbers comparable to `baselines/`.**
- `--submap_size` for >100-frame sequences (windowed); omit for single-pass short clips.

**2 — compare across models and/or datasets (config-driven grid).**
Declare the axes in a flat YAML; `eval.py` expands `datasets × backbones × conditions`, runs each cell serially, **resumes** (skips cells with an existing `metrics.json`), and writes `comparison.md` + `comparison.json`.

```bash
$PY scripts/eval.py --config configs/cross_model_chess.yaml            # compare backbones on chess
$PY scripts/eval.py --config configs/7scenes.yaml                      # chess/fire/office × backbones × conditions
$PY scripts/eval.py --config configs/7scenes.yaml --dry_run            # print the per-cell plan, run nothing
```
Presets pin `scale_method: rotation_only` so grid runs reproduce the frozen `baselines/` numbers. Author a new experiment by copying a config in `configs/`.

**3 — compare arbitrary trajectories already on disk.**
Drop each method's `<name>.tum` (+ a `gt.tum`) into a dir; aggregate them into one table:

```bash
$PY scripts/eval_compare.py --results-dir results/omega_chess --gt-path results/omega_chess/gt.tum
```

**Datasets:** `$PY data/download_datasets.py <7scenes|co3dv2|kitti|tum|waymo>`.

---

## Layout

```
evals/
  *.py            library modules (see tables below)
  scripts/        entry points: eval runner, comparison, diagnostic tools
  baselines/      committed reference results (frozen; see below)
  results/        gitignored scratch output of eval_gt runs
  data/           datasets (7-Scenes etc; large, gitignored)
  envs/           conda env files
```

---

## Entry points (active)

| file | role |
|---|---|
| `scripts/eval.py` | **Main GT eval runner.** Single-cell: feedforward backbone × condition (`baseline`/`lc`/`ba`/`ba_track-density-N`) → ATE/RPE/AUC vs GT, each condition in its own subprocess (clean GPU). Config-driven grid: `eval.py --config configs/7scenes.yaml` expands datasets × backbones × conditions, resumes on existing `metrics.json`, and aggregates a `comparison.md`/`comparison.json`. Writes `metrics.json` (ATE, RPE trans+rot, AUC@{5,15,30}, `n_loops_applied`), TUM trajectories, plots. |
| `scripts/eval_compare.py` | Comparison aggregation over multiple methods/cells (`scan_results_dir`/`collect_grid_metrics`/`format_markdown*`); imported by `eval.py` and runnable standalone over a results dir. |

## Library (imported by the harness)

| file | role |
|---|---|
| `datasets.py` | Dataset loaders: 7-Scenes, CO3Dv2, TUM association files. `get_dataset(name)`; GT-TUM/frame helpers (`write_tum_allowed_frames`, `collect_frames`). |
| `metrics.py` | Thin `evo` ATE/RPE wrapper + `compute_auc` (TUM-file pose AUC). ATE/RPE source of truth. |
| `trajectory_io.py` | Trajectory read/write (TUM etc). |
| `trajectory_metrics.py` | In-memory pose-array ATE / RPE / pairwise AUC (`ate_translation`, `rpe`, `auc_at_threshold`, `umeyama_align`); ATE and AUC Umeyama-align first, RPE is alignment-free. |
| `pose_graph_diagnostics.py` | Loop-closure pose-graph diagnostics: `capture_pose_graph_loss` (per-iteration LM cost, per-edge residuals). |

## Dataset downloaders (utility)

`data/download_datasets.py <dataset>` — one consolidated CLI with a subcommand per dataset: `7scenes` (`--parity` for the LC-parity set), `co3dv2`, `kitti`, `tum`, `waymo`. `data/extract_waymo.py` converts a Waymo tfrecord → flat layout.

## Parity references

VGGT-SPARK / VGGT-SLAM parity (method, pinned commits, headline numbers): [`../docs/parity.md`](../docs/parity.md).

## Other eval tools (standalone)

| file | role | status |
|---|---|---|
| `eval_similarity_calibration.py` | Sweep LC verify layer per backbone (DINO-SALAD pairs). Produced the per-model `_lc_layer_index` calibration. | keep (re-runnable tool) |
| `eval_multiview_conf.py` | Multiview-confidence eval across backbones (chess). | keep |

---

## `baselines/` — committed reference results

| dir | what |
|---|---|
| `cross_model/` | Cross-model benchmark (2026-05-31, chess). One dir per `<backbone>__<frameset>__<sm>/` with `metrics.json` + `ate.json` + TUM; `slam_d*` framesets are fixed keyframe lists (the frames are the `gt.tum` timestamps). `_layersweep/` omega LC-layer sweep, `_core_matrix_table.md`. Heavy COLMAP/ply/plots/npz **gitignored** (see `.gitignore`). |
| `lc_parity_d5/`, `lc_parity_d5_postfix/` | LC on chess d5 (384 keyframes), before and after the 2026-07 loop-edge fixes: `metrics.json` + `lc_decisions_*.json` per backbone (+ `loop_pr.json` post-fix). |
| `lc_parity_matrix/` | LC on 7s office / redkitchen + TUM fr3_office (+ 25%/50% keyframe prefixes) per backbone: `loop_pr.json` (loop precision/recall), plus `metrics.json` on TUM. |

---

## Tests / benchmarks performed

- **Cross-model LC benchmark** (chess, 2026-05-31): backbones × {single-pass, windowed
  baseline, lc} × {d10, d5_long}; numbers in `baselines/cross_model/_core_matrix_table.md`.
  Headline then: windowing is free; `vggt_omega` best baseline. LC results predate the
  2026-07 loop-edge fixes (see `lc_parity_d5_postfix/`).
- **Unit tests:** `tests/evals/` (runner, table, eval_gt helpers, AUC) and
  `tests/pointcloud/` (AUC metric, LC eval). Run: `$PY -m pytest tests/`.
  Known pre-existing failures: `docs/known-test-failures.md`.

## Investigations — scripts grouped by what they probed

Grouped by topic so the "why" behind each investigation is discoverable. B is a
closed investigation (parameter absorbed elsewhere) whose scripts have been deleted.
C and D's scripts are still retained and re-runnable; several compute paths from their
directory depth (`Path(__file__).parents[N]`), so moving them would silently break path
resolution.

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
Goal: rank backbones; separate windowing cost from LC benefit. The frozen
2026-05-31 numbers live under `baselines/cross_model/`. The bespoke sweep/table
drivers that generated them have been retired; reproduce or extend the matrix for a
different backbone/frameset via `eval.py --config configs/7scenes.yaml` (or
`configs/cross_model_chess.yaml`).

> **Housekeeping:** if bundle-adjustment tuning is revisited, prefer writing
> plots to the gitignored `results/` rather than the source tree.
