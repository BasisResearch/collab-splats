# Parity Run 3 — Results & Next Steps

**Date:** 2026-05-28  
**Branch:** `refactor/cu121`  
**Commit:** `feat(lc): layer calibration sweep + ATE parity run 3`

---

## Architecture Finding

VGGT-SLAM uses **identical two-gate LC** as ours:
1. DINO-SALAD L2 distance gate (their `lc_thres=0.95`, ours `lc_cosine_threshold=0.75`)
2. Attention verify gate — same 0.85 threshold, same `get_similarity` algorithm

`compute_similarity=True` only in `third_party/vggt_spark` fork. Our hook-based `cross_frame_attention_ratio` is the correct equivalent. Score gap vs 1.025 reference = pair distribution difference (VGGT-SPARK measured on LC candidates, our calibration measured on random temporal pairs). Now fixed with `--mode retrieved`.

---

## Layer Sweep Results

DINO-SALAD retrieved pairs, 20 pairs, chess_seq01.

| Model | Block depth | Swept layers | Optimal | mtq |
|---|---|---|---|---|
| VGGT-X | 24 `global_blocks` | 23, 20, 18, 16, 12 | 20 (keep) | 0.817 |
| VGGT-Omega | 24 `inter_frame_blocks` | 20, 16, 12, 8 | **16** | 1.328 |
| MapAnything | 16 `info_sharing.self_attention_blocks` | 15, 4 | **4** | 1.807 |

VGGT-X layer 12 peaks on calibration (1.426) but produces identical ATE (0.2664m) as layer 20 → keep VGGT-SPARK's validated default.

---

## ATE Results (chess_seq01, 200 frames)

| Condition | ATE RMSE | Δ vs baseline |
|---|---|---|
| VGGT-X baseline (submap_size=16) | 0.3184m | — |
| VGGT-X + LC (layer=20) | **0.2664m** | −16.3% ✓ |
| VGGT-X + LC (layer=12) | 0.2664m | same |
| Omega baseline (submap_size=16) | 0.3225m | — |
| Omega + LC (layer=20) | 0.4959m | +54% ✗ |
| Omega + LC (layer=16, fixed) | 0.4958m | +54% ✗ — open issue |
| **MapAnything baseline (single-pass)** | **0.1103m** | n/a (different mode) |
| VGGT-SLAM LC (vggt_spark model) | 0.0189m | 30 keyframes, scale-corrected |

VGGT-SLAM comparison is not apples-to-apples: 30 keyframes selected from 200, uses `--correct_scale`. Not directly comparable to our 200-frame uniform runs.

---

## Code Changes Landed

| File | Change |
|---|---|
| `feedforward/vggt_omega.py` | `_lc_layer_index=16`; removed `default_verify_match_ratio=0.59` placeholder |
| `feedforward/mapanything.py` | `_lc_layer_index=4`; `_lc_collate_outputs` method |
| `feedforward/base.py` | `_lc_collate_outputs` no-op default |
| `pointcloud/wrappers.py` | `_run_lc_loop` calls `_lc_collate_outputs` for list outputs |
| `evals/eval_similarity_calibration.py` | `--mode retrieved`, `--layer_index`, DINO-SALAD retrieval |
| `evals/eval_gt.py` | `mapanything` backbone + `_BACKBONE_PREFIX` |
| `evals/eval_vggt_slam_comparison.py` | vggt_spark PYTHONPATH injection, `max_loops` param, TUM archiving |

---

## Open Issues for Next Session

### Issue 1: Omega LC regression (priority: high)

LC consistently worsens Omega ATE (0.3225 → 0.496m) regardless of layer index. Layer 16 is the correct attention layer but doesn't fix the regression.

**Hypotheses to investigate:**
- Omega's `camera_poses` coordinate convention differs from VGGT-X — `assert_world_to_cam` might silently accept wrong-convention poses
- Submap world point reprojection for Omega uses different scale/frame than VGGT-X
- Omega's pose quality is already very good (baseline 0.3225m ≈ VGGT-X baseline 0.3184m) — LC candidates might be sparse/poor quality on this sequence

**Debugging path:**
1. Run Omega LC with verbose logging — check how many loop closures fire, which submap pairs
2. Compare LC pair poses before/after graph optimization
3. Check `assert_world_to_cam` on Omega's raw extrinsics

### Issue 2: MapAnything windowed LC (priority: medium)

`MapAnythingCreator._forward` ignores the `window` argument, always processes `self._processed_views` (all frames set during `_preprocess`). Windowed LC is architecturally incompatible.

**Fix path:**
Override `_forward` in `MapAnythingCreator` to:
1. Accept a `views` argument as the window frames (not `self._processed_views`)
2. Run `preprocess_input_views_for_inference` on those frames only
3. Call `model.forward()` on the preprocessed window

This requires separating `_preprocess` (which sets `self._processed_views`) from `_forward` (which should accept arbitrary frames for LC windowed use).

Also: `_lc_collate_outputs` needs `camera_poses` key which only appears after `postprocess_model_outputs_for_inference` — raw model output doesn't have it. Fix: run minimal postprocess to extract poses in `_lc_collate_outputs`.

### Issue 3: VGGT-SLAM LC 0 closures (priority: low)

`lc_thres=0.95` produces 0 closures on chess_seq01. Either threshold is too strict (try 0.80 matching VGGT-SLAM default), or first 200 frames of chess_seq01 has no genuine revisits at VGGT-SLAM's keyframe density.

**Debug:** lower `lc_thres` to 0.80 in `VGGT_SLAM_ARGS` and re-run.

---

## Eval Commands for Next Session

```bash
# Omega LC debug — verbose
python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_omega --submap_size 16 --max_frames 200 \
  --conditions lc

# VGGT-SLAM with lower threshold
# Edit VGGT_SLAM_ARGS lc_thres to 0.80, then:
python evals/eval_vggt_slam_comparison.py \
  --scene chess --seq seq-01 \
  --data_root evals/data/7scenes/chess \
  --conditions vggt_slam_lc

# MapAnything + LC (after _forward windowed fix)
python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone mapanything --submap_size 16 --max_frames 200 \
  --conditions baseline lc
```
