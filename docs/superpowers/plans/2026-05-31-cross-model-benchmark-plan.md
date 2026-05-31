# Plan: Cross-model LC benchmark on chess vs vggt_spark

**Date:** 2026-05-31 · **Branch:** refactor/cu121
**Prereq:** LC pose-extraction fix landed (commit `1372ac2`). vggt_spark is now at numerical parity with VGGT-SLAM — it is the **reference** for this benchmark.
**Read first:** `docs/superpowers/specs/2026-05-31-lc-parity-benchmarking-handoff.md` (gotchas), `...-vggt-spark-stage-parity-findings.md` (why the fix).

---

## Goal

Benchmark every feedforward backbone on **7-Scenes chess**, sweeping parameters, and rank each (model, params) by ATE/RPE against ground truth. Use **vggt_spark** (= VGGT-SLAM parity) as the reference column. Produce one results table + a short writeup of what wins and where each model breaks.

---

## Backbones (`--backbone`)

| backbone | LC layer (`_lc_layer_index`, auto) | notes |
|---|---|---|
| `vggt_spark` | 20 | **reference** (VGGT-SLAM parity). Needs its `_forward` override (already in repo). |
| `vggtx` | 20 | VGGT-X |
| `vggt_omega` | 16 | prior note: LC was *harmful* — **recheck**, may have changed post-fix |
| `mapanything` | 4 | prior note: windowed LC *broken* — **recheck** |

LC layer is set per-class; no CLI flag needed. To sweep layers, override `_lc_layer_index` (or extend the hook's `layer_index` arg).

---

## Parameter axes

- **condition** (`--conditions`): `baseline` (no LC/BA), `lc` (loop closure), `ba` (bundle adjustment). Also `ba_track-density-N`.
- **lc_scale_method** (`--lc_scale_method`): `se3` (default), `rotation_only` (VGGT-SLAM style), `pairwise_dist`, `none`. *For VGGT-SLAM parity use `rotation_only` or `none`.*
- **submap_size** (`--submap_size`): 16 (parity runs). Try 16/20/50. <16 frames ⇒ single submap ⇒ LC loop skipped.
- **frame set**: either full sequence (`--max_frames N`) or `--keyframe_list .../slam_dD/selected_frames.txt` for VGGT-SLAM-identical frames (D ∈ {50,30,20,10,0}).
- **sequence**: chess/seq-01 (unzipped). seq-02..06 are zipped under `evals/data/7scenes/chess/chess/` — unzip if extending.

Suggested core matrix (start small, expand): `{4 backbones} × {baseline, lc} × {submap_size 16} × {keyframe_list d10, d0/full} × {lc_scale_method rotation_only}`. Add `ba` and more disparities once the core table is clean.

---

## Commands

```bash
PY=/opt/conda/envs/reconstruction/bin/python   # py3.11 env; NOT base conda
SEQ=evals/data/7scenes/chess/chess/seq-01

# One run (windowed, identical frames to VGGT-SLAM via keyframe_list)
$PY evals/eval_gt.py --dataset 7scenes --seq_dir $SEQ \
  --backbone <BACKBONE> --conditions baseline lc \
  --submap_size 16 --lc_scale_method rotation_only \
  --keyframe_list evals/baselines/disparity_sweep/slam_d10/selected_frames.txt \
  --output_ate /tmp/bench/<BACKBONE>_d10.json
```

- Run in **tmux**, one model at a time (46 GB cgroup cap — no parallel GPU jobs).
- Reference baselines already on disk: `evals/baselines/disparity_sweep/slam_dD/metrics.json` (VGGT-SLAM ATE) and the vggt_spark numbers you reproduce.
- Per-stage parity sanity for any backbone: `evals/runners/parity_trace.py --side {slam,ours,diff} --min_disparity D`.

---

## CRITICAL gotchas (do not skip)

1. **Verify the model that actually loads.** `eval_gt --backbone vggt_spark` can silently run **VGGT-X** if `vggt` is import-cached before `VGGTSPARKCreator._load_model` inserts the SPARK path. Log/print the loaded module path before trusting SPARK numbers.
2. **Re-verify per-model LC layers.** The old optimal layers (20/16/4) and the "Omega-harmful / MapAnything-broken" notes were measured **under the R/Rᵀ pose bug**. Multi-submap LC results may have shifted — re-measure, don't assume.
3. **bf16 floor.** SLAM stores homographies bf16; ours float64 (~1.5 mm over a multi-submap chain). Compare via **ATE** (Sim3-aligned), not raw pose diff.
4. **Frame 0 is included**; d10 SLAM TUM has a benign overlap-frame duplicate (evo keeps first). Use `--keyframe_list` for cross-model frame parity.
5. **Env:** `/opt/conda/envs/reconstruction/bin/python` (py3.11). `--submap_size 50` for >100-frame sequences.

---

## Deliverables

1. `docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md` — table: rows = (backbone, condition, params), cols = ATE / RPE / Δ-vs-spark / Δ-vs-SLAM, + which model loaded.
2. Short analysis: best model per condition; where LC helps vs hurts per backbone; any backbone still diverging from spark and why (use `parity_trace.py` to localize).
3. Raw `--output_ate` JSONs under a results dir (gitignored `evals/results/` or a committed `evals/baselines/cross_model/`).

---

## Definition of done

- Core matrix run, ATE/RPE tabulated, vggt_spark reproduces ~0.017 m at d10 baseline (sanity: matches VGGT-SLAM 0.0176).
- Each backbone's LC-vs-baseline delta reported with the loaded-model verified.
- Findings written; any divergence traced to a stage via the harness.
