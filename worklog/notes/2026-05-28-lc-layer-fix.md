# LC Similarity Layer Fix — 2026-05-28

## Root Cause

`cross_frame_attention_ratio` was hooked at `global_blocks[-1]` (block 23 of 24). VGGT-SPARK uses `target_layer=20` (block 20 of 24). Same `facebook/VGGT-1B` weights, different attention distribution → systematic score gap.

| Layer | mtq mean (random pairs, chess_seq01) |
|---|---|
| -1 / 23 (old) | 0.663 |
| 20 (VGGT-SPARK parity) | 0.764 |
| VGGT-SPARK baseline (LC candidates) | ~1.025 |

Remaining gap between our layer=20 and VGGT-SPARK baseline is expected: VGGT-SPARK scores were measured on DINO-SALAD-retrieved pairs (strong overlap), our calibration used random temporal pairs (moderate overlap). LC candidates in production score higher than random pairs.

## Normalization audit

- VGGT: `ToTensor()` [0,1] input → model normalizes internally with `_RESNET_MEAN/STD = [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`
- MapAnything DINOv2: same mean/std values → **equivalent normalization**
- No mismatch. Score differences are architectural.

## MapAnything findings

- `extract_intermediate_features` missing `data_norm_type=["dinov2"]` in raw_views → `KeyError` from `preprocess_input_views_for_inference`
- Views built from `frames.cpu()` not moved to model device before forward → CPU/CUDA mismatch
- `info_sharing` has no special tokens → `_lc_token_offset` should be 0 (not 5)
- At layer=-1, MapAnything attention scores ~0.487 (high variance). Gate is weak for MapAnything; deeper calibration needed if MapAnything LC is a priority.

## Aggregation fix

`cross_frame_attention_ratio` now uses `mean_top_quarter` (mean of values ≥ 75th pctl) matching VGGT-SPARK `get_similarity()` exactly. Previously used `np.percentile(90)` → lower scalar.

## Validation

chess_seq01, 200 frames, submap_size=16, vggtx backbone:
- 2 loop closures fired (submaps 7→5 dist=0.575, 10→8 dist=0.650)
- 4 candidates rejected (below 0.85 threshold)
- **ATE RMSE: 0.2664m vs 0.3184m baseline (-16.3%)**

## Changes

- `base.py`: `_lc_layer_index=20`, `_lc_token_offset=5` class attrs; `_verify_loop_candidate` uses them
- `vggtx.py`: removed wrong `default_verify_match_ratio=0.59`
- `mapanything.py`: `_lc_token_offset=0`; fixed `data_norm_type` + device in `extract_intermediate_features`
- `utils.py`: `mean_top_quarter` aggregation
- `evals/eval_similarity_calibration.py`: new calibration script

Commit: `315c312`
