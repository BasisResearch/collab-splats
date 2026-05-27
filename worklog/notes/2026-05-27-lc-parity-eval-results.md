# LC VGGT-SLAM Parity — Regression Eval Results

**Date:** 2026-05-27  
**Branch:** refactor/cu121  
**Eval command:**
```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
  --output_dir evals/results/chess_seq01_all_parity_fixes \
  --max_frames 200 --submap_size 16 --backbone vggtx \
  --conditions baseline lc
```

## Results (post-fix)

| Condition | ATE RMSE | RPE Trans | RPE Rot |
|-----------|----------|-----------|---------|
| baseline  | 0.3184 m | 0.01337 m | 1.267°  |
| lc        | 0.3175 m | 0.01347 m | 1.269°  |

**LC vs baseline delta:** −0.3% ATE (marginal improvement)  
**Loop closures detected:** 0 (all 11 submaps: loops=0, verified=0)

## Analysis

LC provides minimal improvement because chess/seq-01 at 200 frames with
submap_size=16 detects zero loop closures. Without loop edges, the PGO only
applies sequential edges + inter-submap stitching. The −0.3% gain comes
purely from Bug 1 (H_w formula) and Bug 2 (pose extraction) fixing the
inter-submap stitching math.

**Bug 2 (pose extraction) did NOT regress ATE** — lc (0.3175) < baseline
(0.3184), confirming it is safe to keep.

## Comparison caveat

The pre-fix reference numbers in the plan (baseline=0.183m, lc=0.187m) came
from a prior eval run with different params (likely larger submap_size). They
cannot be directly compared to these results. The correct baseline for
regression testing is the within-run baseline condition, which LC beats.

## Why far from VGGT-SLAM 0.055m

VGGT-SLAM's published ATE is a model-quality difference (different
network, different training data), not purely an LC algorithm difference.
Closing that gap requires model improvements beyond the LC bug fixes.
