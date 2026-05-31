# Results: Cross-model LC benchmark on 7-Scenes chess

**Date:** 2026-05-31 · **Branch:** refactor/cu121
**Spec:** `2026-05-31-cross-model-benchmark-design.md` · **Plan:** `../plans/2026-05-31-cross-model-benchmark.md`
**Reference backbone:** `vggt_spark` (VGGT-SLAM parity). **Sequence:** chess/seq-01.

---

## TL;DR

1. **Sanity gate passed.** `vggt_spark` loads the **real SPARK** module and reproduces VGGT-SLAM at d10 (ATE 0.0174 m vs SLAM 0.0176 m).
2. **Windowing costs nothing** (goal 1). Submap windowing matches or slightly *beats* single-pass full-batch inference on every backbone. Submaps are free.
3. **Loop closure never helps on chess** (goal 2). On the long set with 21 real loops, LC is either a **no-op** (spark, mapanything) or **catastrophic** (vggtx 0.86 m, omega 0.58 m — a 17–23× blow-up). It never lowers ATE.
4. **LC failure is mechanism-level, not layer calibration.** A verify-layer sweep (8/12/16/24) leaves omega LC catastrophic at every layer. The layer only gates candidates; its best case is reject-all (no-op), never improvement.
5. **Best backbone: `vggt_omega`** — best baseline ATE at every frameset and by far the best local pose quality (AUC@5 ≈ 67 vs ≈ 46 for spark/vggtx).
6. **Two bugs found and fixed** en route (both pre-existing): the `vggt_spark` loader silently ran VGGT-X (wrong root path), and LC-with-loops crashed on a CPU/CUDA device mismatch.

---

## Core matrix

ATE = Sim3-aligned RMSE (m). RPE-t (m), RPE-r (°). AUC@{5,15,30} = pairwise pose AUC (alignment-free). Δspark / Δslam = ATE minus the spark / VGGT-SLAM reference on the same frameset. All `vggt_spark` runs verified to load `third_party/vggt_spark` (real SPARK).

Framesets: `slam_d10` = 26 frames / 2 submaps / **0 loops**; `slam_d10_single` / `slam_d20_single` = single-pass (no submaps); `slam_d5_long` = 384 frames / 45 submaps / **21 loops** (SLAM-LC ref ATE 0.0464 m).

| backbone | frameset | cond | ATE | RPE-t | RPE-r° | AUC5 | AUC15 | AUC30 | Δspark | Δslam |
|---|---|---|---|---|---|---|---|---|---|---|
| vggt_spark | slam_d10_single | baseline | 0.0182 | 0.0456 | 2.07 | 34.7 | 75.1 | 87.4 | — | — |
| vggt_spark | slam_d20_single | baseline | 0.0187 | 0.0944 | 1.79 | 33.5 | 74.8 | 87.4 | — | — |
| vggt_spark | slam_d10 | baseline | 0.0174 | 0.0455 | 0.52 | 46.0 | 79.7 | 89.7 | 0.0000 | −0.0003 |
| vggt_spark | slam_d10 | lc | 0.0174 | 0.0455 | 0.52 | 46.0 | 79.7 | 89.7 | 0.0000 | −0.0003 |
| vggt_spark | slam_d5_long | baseline | 0.0441 | 0.0239 | 0.47 | 24.1 | 70.5 | 84.9 | 0.0000 | −0.0022 |
| vggt_spark | slam_d5_long | lc | 0.0441 | 0.0239 | 0.47 | 24.1 | 70.5 | 84.9 | 0.0000 | −0.0022 |
| vggtx | slam_d10_single | baseline | 0.0187 | 0.0457 | 0.52 | 43.8 | 78.3 | 89.1 | +0.0005 | — |
| vggtx | slam_d20_single | baseline | 0.0184 | 0.0943 | 0.89 | 44.1 | 78.6 | 89.3 | −0.0003 | — |
| vggtx | slam_d10 | baseline | 0.0173 | 0.0457 | 0.51 | 46.6 | 79.9 | 89.9 | −0.0001 | −0.0004 |
| vggtx | slam_d10 | lc | 0.0173 | 0.0457 | 0.51 | 46.6 | 79.9 | 89.9 | −0.0001 | −0.0004 |
| vggtx | slam_d5_long | baseline | 0.0428 | 0.0239 | 0.47 | 21.7 | 68.8 | 84.0 | −0.0013 | −0.0035 |
| vggtx | slam_d5_long | **lc** | **0.8629** | 0.0363 | 0.80 | 1.2 | 6.0 | 12.1 | +0.8187 | +0.8165 |
| vggt_omega | slam_d10_single | baseline | **0.0101** | 0.0436 | 0.43 | 65.5 | 87.3 | 93.6 | −0.0081 | — |
| vggt_omega | slam_d20_single | baseline | 0.0113 | 0.0890 | 0.74 | 69.4 | 88.4 | 94.2 | −0.0075 | — |
| vggt_omega | slam_d10 | baseline | **0.0101** | 0.0445 | 0.43 | 67.4 | 88.3 | 94.2 | −0.0073 | −0.0075 |
| vggt_omega | slam_d10 | lc | 0.0101 | 0.0445 | 0.43 | 67.4 | 88.3 | 94.2 | −0.0073 | −0.0075 |
| vggt_omega | slam_d5_long | baseline | **0.0308** | 0.0223 | 0.42 | 51.5 | 82.6 | 91.1 | −0.0133 | −0.0156 |
| vggt_omega | slam_d5_long | **lc** | **0.5815** | 0.0441 | 2.00 | 0.1 | 0.6 | 2.7 | +0.5373 | +0.5351 |
| mapanything | slam_d10_single | baseline | 0.0235 | 0.0720 | 0.49 | 36.6 | 72.9 | 86.1 | +0.0053 | — |
| mapanything | slam_d20_single | baseline | 0.0223 | 0.1386 | 0.83 | 35.6 | 73.4 | 86.7 | +0.0035 | — |
| mapanything | slam_d10 | baseline | 0.0204 | 0.0695 | 0.57 | 36.2 | 74.7 | 87.0 | +0.0031 | +0.0028 |
| mapanything | slam_d10 | lc | 0.0204 | 0.0695 | 0.57 | 36.2 | 74.7 | 87.0 | +0.0031 | +0.0028 |
| mapanything | slam_d5_long | baseline | 0.1927 | 0.0478 | 0.61 | 2.8 | 33.3 | 60.8 | +0.1486 | +0.1464 |
| mapanything | slam_d5_long | lc | 0.1927 | 0.0478 | 0.61 | 2.8 | 33.3 | 60.8 | +0.1486 | +0.1464 |

Raw JSONs: `evals/baselines/cross_model/<backbone>__<frameset>__<sm>/{metrics.json,ate.json}` (committed; heavy COLMAP/ply/plots/npz gitignored). Layer sweep: `evals/baselines/cross_model/_layersweep/`. Gate: `_gate/`.

---

## Analysis

### Best model per condition
- **Baseline (every frameset): `vggt_omega`.** d10 0.0101, d5_long 0.0308 — and its AUC@5 (≈67) towers over the others (≈46), so its *local* relative poses are markedly better, not just its global alignment. `vggtx` ≈ `vggt_spark` (0.0173 vs 0.0174 at d10). `mapanything` is weakest, and collapses on the long set (0.1927, AUC@5 2.8).
- **vs VGGT-SLAM:** on the long set, the windowed *no-LC* baselines (omega 0.031, vggtx 0.043, spark 0.044) all **beat the SLAM-LC reference (0.046)** — i.e. not running loop closure beats running it, even relative to SLAM's own LC pipeline.

### Goal 1 — windowing cost (single-pass vs windowed baseline, d10)
| backbone | single-pass | windowed | Δ (windowed − single) |
|---|---|---|---|
| vggt_spark | 0.0182 | 0.0174 | −0.0008 |
| vggtx | 0.0187 | 0.0173 | −0.0014 |
| vggt_omega | 0.0101 | 0.0101 | 0.000 |
| mapanything | 0.0235 | 0.0204 | −0.0031 |

**Windowing never costs accuracy** — it is neutral (omega) or marginally better (others), consistent with AUC being unchanged. Splitting into submaps is free; LC is the only reason to window, and LC does not pay off here (below).

### Goal 2 — loop-closure benefit (long set, 21 real loops)
LC **never lowers ATE**. Two distinct failure modes:
- **No-op** — `vggt_spark` and `mapanything`: lc ≡ baseline to 4 dp. The pipeline closes **0** loop edges (spark's progress bar reads `loops=0` even though the native verify gate "accepts" candidates at ratio ≈ 1.0). This is the **retrieval/verify-gate mismatch** flagged in the handoff: our DINO-SALAD L2 gate / native `image_match_ratio` path registers no usable loop edges where SLAM closed 21.
- **Catastrophic** — `vggtx` (0.043 → 0.863) and `vggt_omega` (0.031 → 0.582): the base cross-frame-attention verify path **does** apply loop edges, and they **wreck** the pose graph (AUC@5 collapses 22 → 1, 51 → 0; RPE-rot jumps to 2°). The applied loop edges are geometrically inconsistent and the SL(4)/Sim3 correction distributes the error across all nodes.

**Conclusion:** on chess seq-01, every backbone is better off **without** loop closure. The pose-fix that achieved VGGT-SLAM *baseline* parity did **not** fix LC-with-loops — that path was explicitly unverified in the handoff and is now shown broken.

### LC layer re-measurement (overturns the old calibration premise)
Old notes (VGGT-X=20, Omega=16, MapAnything=4; "Omega LC harmful / MapAnything LC broken") were measured under the pre-fix pose bug. Re-measured post-fix:
- At default layers, on the **0-loop** d10 set, lc ≡ baseline for all backbones (nothing to close).
- On the **21-loop** long set, omega LC is catastrophic at **every** swept layer (8: 0.547, 12: 0.708, 16: 0.582; 24 invalid/exceeds depth). The verify layer only gates which candidates pass — its best-case outcome is reject-all (LC → no-op), **never** improvement.

So the old "re-tune the per-model LC layer" framing is moot: the LC **mechanism** (loop-edge application / pose-graph correction), not the verify layer, is what fails. The prior "Omega LC harmful" note **still holds** post-fix; "MapAnything LC broken" now reads as **no-op** (0 loops applied).

### Divergence from spark
- **mapanything** is the only backbone that diverges materially from spark on baseline (Δspark +0.0031 at d10, **+0.1486** at d5_long; AUC@5 2.8 on the long set). It degrades far faster than the others as the trajectory lengthens — a model-quality gap, not a pipeline bug (its baseline windowing matches single-pass). `parity_trace.py` is spark-vs-SLAM only, so a mapanything-specific stage trace would need a separate harness; the AUC/RPE signature points to weak long-range pose estimation rather than a stitching error.
- vggtx and omega track spark closely on baseline (|Δspark| ≤ 0.013 everywhere); their only divergence is the LC catastrophe above.

---

## Bugs found and fixed

1. **`vggt_spark` silently ran VGGT-X** (`fix(feedforward): correct _VGGT_SPARK_ROOT path`). The SPARK root was computed as `parents[4]` (`/workspace`) but the tree is at `parents[3]` (`<repo>/third_party/vggt_spark`); the bad path insert fell through to site-packages VGGT-X. This is the root cause behind the handoff's "eval_gt vggt_spark may silently run VGGT-X" warning. A new load-guard (`fix(feedforward): hard-guard vggt_spark…`) now raises rather than returning wrong numbers, and logs the resolved module path for every run.
2. **LC-with-loops device crash** (`fix(loop_closure): move LC verify frames to model device`). The first runs to actually close loops crashed: `_verify_loop_candidate` stacked CPU-held candidate frames and fed them to the CUDA model. Both verify paths now move the batch to the model device (SPARK also matches bf16).

---

## Reproduce

```bash
PY=/opt/conda/envs/reconstruction/bin/python
# Core matrix (serial, tmux, one model at a time):
$PY evals/runners/run_cross_model_benchmark.py
# Rebuild this table:
$PY evals/runners/build_benchmark_table.py
# Loop probe / long ref:
$PY evals/runners/run_vggt_slam_lc.py --seq_dir evals/data/7scenes/chess/chess/seq-01 \
   --max_frames 1000 --min_disparity 5 --max_loops 1
```

## Recommended next work
1. **Debug LC-with-loops** (the real blocker): localise why applied loop edges corrupt the pose graph (vggtx/omega) and why spark/mapanything register 0 loops. Start at `wrappers.py:_run_lc_loop` loop-edge construction and the SL(4)/Sim3 correction; compare our retrieval gate to SLAM's `image_match_ratio` on the 21 known loops.
2. **mapanything long-range degradation** — separate model-quality investigation.
3. Until LC is fixed, **run windowed baseline (no LC)** for best ATE on chess; `vggt_omega` is the backbone of choice.
