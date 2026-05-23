# Loop Closure Baseline Validation Design

**Date:** 2026-05-08
**Branch:** `refactor/core-modules`
**Spec status:** Draft, approved via plan-mode review

---

## Context

Our `collab_splats/pointcloud/loop_closure/` implements Sim(3) submap pose-graph optimization (2026-05-07 fix), modeled after VGGT-Long. Our `evals/eval_gt.py` runner currently validates only against 7-Scenes chess seq-01 with three internal conditions (`baseline`, `ba`, `lc`).

We have no apples-to-apples comparison against the reference implementations:

- **VGGT-SLAM** (MIT-SPARK) — TUM RGB-D, 7-Scenes, SL(4) manifold via GTSAM (Python 3.11)
- **VGGT-Long** (DengKaiCQ) — KITTI Odometry, Waymo, Sim(3) via pypose+Eigen (Python 3.10 + torch 2.5 + CUDA 11.8)

Goal: extend the eval module so we can (1) load TUM, KITTI, Waymo with the same schema as 7-Scenes; (2) run our methods AND the external methods on identical sequences; (3) report side-by-side ATE/RPE numbers using the standard `evo` toolchain.

VGGT-SLAM is blocked on Python 3.11 / SL(4) manifold dependencies — handled via placeholder until the env upgrade lands. VGGT-Long is runnable today; probe single-env install before committing to a separate conda env.

---

## Decisions (locked via brainstorm)

| Axis | Choice |
|------|--------|
| External-method invocation | Probe single-env install for VGGT-Long; trajectory-only (manual dump) for VGGT-SLAM until Py3.11 |
| Dataset scope | TUM 9 fr1 seqs · 7scenes {chess,fire,office} · KITTI {00,02,05,06,07,09} · Waymo 2–3 segments |
| Runner shape | Two-phase: trajectory dump → metrics computation |
| Trajectory format / metrics | TUM canonical (`timestamp tx ty tz qx qy qz qw`); thin `evo_ape`/`evo_rpe` wrapper |
| VGGT-SLAM during block | `.pending` sentinel; phase-2 skips missing methods |

---

## Architecture

### Two-phase data flow

```
Phase 1 (per method × per seq) — write trajectory file
  evals/eval_gt.py                ── ours_{baseline,ba,lc}.tum
  evals/runners/run_vggt_long.py  ── vggt_long.tum
  (manual dump later)             ── vggt_slam.tum (or .pending)
  evals/datasets.py               ── gt.tum  (loaders write GT to TUM)

Phase 2 (per seq) — ingest dir, emit unified metrics
  evals/eval_compare.py
    inputs:  evals/results/{dataset}_{seq}/*.tum
    outputs: metrics.json (all methods × ATE/RPE), plots/
```

### File layout

```
collab_splats/evals/
├── datasets.py                # +load_tum, +load_kitti, +load_waymo (EvalDataset schema)
├── trajectory_io.py           # NEW: TUM r/w; KITTI 3x4-flat ↔ TUM; numpy(N,4,4) ↔ TUM
├── metrics.py                 # NEW: evo_ape / evo_rpe wrapper; Sim3-aligned for monocular
├── eval_gt.py                 # MOD: also emit per-method .tum (additive)
├── eval_compare.py            # NEW: phase-2 runner
├── download_{7scenes,tum,kitti,waymo}.sh
├── runners/
│   ├── run_vggt_long.py       # subprocess wrapper around third_party/VGGT-Long
│   └── run_vggt_slam.py       # raises EnvBlocked until Py3.11
├── baselines/
│   ├── vggt_long/{seq}.tum
│   └── vggt_slam/{seq}.pending
└── results/{dataset}_{seq}/

third_party/
├── VGGT-Long/                 # git submodule
└── VGGT-SLAM/                 # git submodule (used post-unblock)
```

### Reused utilities

- `evals/datasets.py:_load_7scenes` — pattern for new loaders (line 16)
- `EvalDataset` dataclass — schema unchanged
- `collab_splats/pointcloud/wrappers.py:LoopClosure, BundleAdjustment` — phase-1 entry points
- `collab_splats/pointcloud/loop_closure/eval.py:ate_translation, rpe` — kept for legacy paths

---

## Parallel agent decomposition

| ID | Task | Depends | LOE |
|----|------|---------|-----|
| T1 | `trajectory_io.py` + `metrics.py` + tests | — | M |
| T2 | TUM loader + download script | T1 | S |
| T3 | KITTI loader + download script | T1 | M |
| T4 | Waymo loader + download script | T1 | L |
| T5 | 7-Scenes expansion (fire, office) | T1 | S |
| T6 | VGGT-Long submodule + env probe + runner | T1 | L |
| T7 | `eval_compare.py` phase-2 | T1 | M |
| T8 | VGGT-SLAM submodule + stub + sentinels | T1, T7 | S |

T1 lands first. T2–T7 fan out in parallel. T8 follows T7.

---

## Open issues to resolve in execution

1. **Waymo `.tfrecord` parsing** — `waymo-open-dataset` may collide with our torch/tensorflow pin. T4 fallback: ship a `--export-tum` sidecar that runs in its own env.
2. **VGGT-Long single-env probe** — if torch 2.5 conflicts with gsplat-rade pin, fall back to `vggt-long-py310-cu118` env. Investigate at start of T6.
3. **Sim3 vs SE3 alignment** — `evo_ape -as` for VGGT-Long + our `lc` (Sim3 paths). `evo_ape -a` for `baseline`/`ba`. Document per-condition in `metrics.py`.
4. **Frame timestamps for non-TUM datasets** — convention: frame index as float seconds (`0.000000`, `1.000000`, …). Documented in `trajectory_io.py`.

---

## Verification

```bash
# Phase 1
bash evals/download_tum.sh fr1_desk
bash evals/download_kitti.sh 00
python evals/eval_gt.py --dataset tum --seq_dir evals/data/tum/fr1_desk \
    --conditions baseline ba lc --output_dir evals/results/tum_fr1_desk
python evals/runners/run_vggt_long.py --dataset kitti_00 \
    --output evals/baselines/vggt_long/kitti_00.tum

# Phase 2
python evals/eval_compare.py --results-dir evals/results/tum_fr1_desk
python evals/eval_compare.py --results-dir evals/results/kitti_00
```

| # | Acceptance criterion |
|---|----------------------|
| 1 | All 4 dataset loaders return `EvalDataset` with finite `gt_poses` shape `(N,4,4)` |
| 2 | `trajectory_io` round-trips numpy(N,4,4) ↔ TUM with < 1e-6 max error |
| 3 | `metrics.py` ATE RMSE matches `evo_ape … -as` CLI within 1e-9 (golden file on chess seq-01) |
| 4 | `eval_compare.py` produces single `metrics.json` with all available methods; absent methods recorded as `"status": "pending"` |
| 5 | VGGT-Long runner produces a TUM trajectory file on KITTI 00 — within 2× their published ATE (sanity) |
| 6 | All loaders + runners pass pytest; no regressions in existing `tests/pointcloud/` suite |
| 7 | Spec + WORKLOG entry committed; commits follow `<scope>:` prefix |

---

## Out of scope (future specs)

- VGGT-SLAM full integration post Py3.11 unblock
- Additional KITTI sequences without GT loop closures
- Virtual KITTI
- Replicating their specific hyperparameters
- ATE rotation / KITTI t_err–r_err per-100m
