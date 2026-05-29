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
