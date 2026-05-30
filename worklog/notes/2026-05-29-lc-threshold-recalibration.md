# LC Threshold Recalibration — 2026-05-29

## Context
All prior threshold measurements were pre-stride-fix (stride=K-O=15, K frames stored).
Stride fix (2026-05-29) changed to stride=K=16, K+O=17 frames stored — matching VGGT-SLAM
exactly. Recalibration required because window frame composition changed.

## Method
`eval_similarity_calibration.py --mode retrieved --n_pairs 20` on chess_seq01.
Metric: `mean_top_quarter` (matches VGGT-SPARK's `get_similarity()`).
Threshold formula: `max(mean - 2*std, 0.5)` — captures ~95% of overlapping pairs.

## Results

| Model | mtq_mean | mtq_std | mtq_min | mtq_max | threshold set |
|---|---|---|---|---|---|
| vggtx/spark | 0.8178 | 0.0102 | 0.7993 | 0.8388 | 0.80 |
| mapanything | 1.8066 | 0.0781 | 1.6100 | 1.9545 | 1.65 |
| omega | 1.3280 | 0.0852 | 1.0862 | 1.5063 | 1.16 |

## Reference
VGGT-SPARK reference (VGGT-1B via SPARK, same weights as VGGTXCreator): mtq_mean=1.025.

## Notes
- vggtx: 0.85 → 0.80. Old threshold caused 0 loops to fire; new threshold below score floor.
- omega: 0.99 → 1.16. Stricter than before, reduces false-positive LCs that caused +54% ATE regression pre-fix.
- mapanything: 1.65 (new; LC was architecturally blocked before this session's fix).
- VGGTSPARKCreator inherits vggtx threshold via class inheritance.

## JSON outputs
- `evals/results/lc_calibration_vggtx_postfix_retrieved.json`
- `evals/results/lc_calibration_mapanything_postfix_retrieved.json`
- `evals/results/lc_calibration_omega_postfix_retrieved.json`

## Eval results — post stride=K fix, calibrated thresholds (chess_seq01)

| Backbone | Baseline ATE | LC ATE | Delta | Notes |
|---|---|---|---|---|
| vggt_spark | 0.3189m | 0.3752m | +17.6% | LC fires but hurts; same weights as vggtx |
| vggtx | 0.3189m | 0.3752m | +17.6% | LC fires but hurts |
| vggt_omega | 0.3223m | 0.2503m | −22.4% | LC now helps; was +54% worse pre-fix |
| mapanything | 1.0023m (200f windowed) | 0.0843m (50f) | — | 200f OOM; windowed baseline regresses vs single-pass (0.1103m) |

### Open issues
1. vggt_spark/vggtx: loops fire at 0.80 but pose graph adds error — investigate edge convention or scale estimation
2. mapanything 200-frame: OOM during LC (segfault in torch); windowed baseline worse than single-pass
3. mapanything windowed baseline regression: single-pass was 0.1103m; windowed (submap_size=16) is 1.0023m — likely quality degradation from small window size

## Cross-model eval — 2026-05-30 (corrected method, submap_size fix)

Fix applied: `eval_gt.py` `lc` condition now correctly passes `submap_size` to `LoopClosureConfig`
(previously defaulted to 20 regardless of `--submap_size` CLI arg).

### 29-frame (max_frames=29, first 29 consecutive frames, 2 submaps, 0 loops all backends)

| Backbone | Baseline ATE | LC ATE | Δ |
|---|---|---|---|
| vggt_spark | 0.0105m | 0.0103m | −2% |
| vggtx | 0.0105m | 0.0103m | −2% |
| vggt_omega | 0.0126m | 0.0123m | −2% |
| mapanything | 0.0244m | 0.0244m | 0% |

All 0 loops — improvement is pure windowed stitching, not loop corrections.

### 200-frame (max_frames=200, 13 submaps)

| Backbone | Baseline ATE | LC ATE | Loops | Δ |
|---|---|---|---|---|
| vggt_spark | 0.3191m | 0.3347m | 0 | +4.9% |
| vggtx | 0.3191m | 0.3353m | 0 | +5.1% |
| vggt_omega | 0.3223m | 0.2652m | 12 | −17.7% |
| mapanything | 1.0023m | 1.0023m | 0 | 0% |

Open issues: spark/vggtx LC hurts with 0 loops (windowing overhead); mapanything windowed regression (1.0m vs 0.11m single-pass).
