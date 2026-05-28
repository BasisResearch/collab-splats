# Parity Run 3 Handoff — LC Layer Calibration + ATE Comparison

**Date:** 2026-05-28  
**Branch:** `refactor/cu121`  
**Commit:** `315c312`

---

## What Was Solved This Session

### Root cause of zero LC triggers (RESOLVED)

`cross_frame_attention_ratio` was hooked at `global_blocks[-1]` (block 23 of 24). VGGT-SPARK hardcodes `target_layer=20` (block 20 of 24). Same `facebook/VGGT-1B` weights — different attention distribution per layer causes systematic score gap.

**Fix applied:**
- `BaseFeedforwardCreator._lc_layer_index: ClassVar[int] = 20` — all creators inherit this default
- `BaseFeedforwardCreator._lc_token_offset: ClassVar[int] = 5` — skip VGGT's 5 special tokens
- `MapAnythingCreator._lc_token_offset: ClassVar[int] = 0` — info_sharing has NO special tokens
- `cross_frame_attention_ratio`: aggregation changed from `np.percentile(90)` → `mean_top_quarter` (mean of values ≥ 75th pctl, matching VGGT-SPARK exactly)
- `MapAnythingCreator.extract_intermediate_features`: fixed missing `data_norm_type=["dinov2"]` + device transfer bug

**Validation (chess_seq01, 200 frames, submap_size=16, vggtx backbone):**
- 2 loop closures fired (submaps 7→5, 10→8)
- 4 candidates rejected (ratio < 0.85 threshold)
- ATE RMSE: **0.2664m vs 0.3184m baseline (−16.3%)**

### Normalization confirmed identical

VGGT `_RESNET_MEAN/STD = [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]` == MapAnything DINOv2 normalization. Score differences between models are architectural, not normalization-driven.

---

## What the Next Agent Needs to Do

### Task 1: Find correct `_lc_layer_index` for MapAnything and VGGT-Omega

**Context:** We defaulted all models to `_lc_layer_index=20` (the VGGT-1B value). MapAnything and VGGT-Omega have different architectures with different numbers of attention blocks. The correct layer for each model is the one whose Q/K attention best separates overlapping from non-overlapping frame pairs.

**MapAnything:**
- Cross-frame blocks: `model.info_sharing.self_attention_blocks` (list)
- Check `len(creator.model.info_sharing.self_attention_blocks)` to get total count
- Current default `_lc_layer_index=20` may be out of range (likely fewer than 20 blocks)
- Run `evals/eval_similarity_calibration.py --models mapanything` across different `layer_index` values (e.g., -1, -3, -6, half-depth) and compare score distributions

**VGGT-Omega:**
- Cross-frame blocks: `model.aggregator.inter_frame_blocks`
- Check total count similarly
- Run calibration sweep across layers

**Tool:** `evals/eval_similarity_calibration.py` already exists. Add `--layer_index` override argument (currently auto-reads from `creator._lc_layer_index`). Alternatively, temporarily set `_lc_layer_index` on the creator instance.

**Goal:** Find the layer where mean_top_quarter scores on overlapping pairs are highest (most peaked attention → best overlap signal). Compare distributions at different layers.

### Task 2: Understand the score gap between our models and VGGT-SPARK

**The puzzle:** At layer=20, VGGT-X (same `facebook/VGGT-1B` weights) scores:
- Random temporal pairs (gap 5–40 frames): mtq mean = 0.764, max = 0.823
- VGGT-SPARK baseline (DINO-SALAD retrieved LC candidates): mtq mean = 1.025

The gap (~0.26) exists even with same weights + same layer. Likely explanation: DINO-SALAD retrieval selects strongly overlapping pairs (camera revisited same location) which produce much higher attention ratios than random temporal pairs. This is expected — the gate is calibrated on LC candidates, not random pairs.

**To confirm:** Run the calibration script on DINO-SALAD-retrieved pairs specifically (not random temporal pairs). The expected result: VGGT-X scores on retrieved pairs cluster near 1.0+, validating that 0.85 threshold is correct.

**How to get DINO-SALAD retrieved pairs:** Run the full LC pipeline with `lc_cosine_threshold` set very low (e.g., 0.99 to suppress geometric verification) and log which pairs DINO-SALAD retrieves. Then compute `cross_frame_attention_ratio` on those pairs.

**Alternative approach:** Run the full LC pipeline with `verify_match_ratio=0.0` (accept all DINO-SALAD candidates) and log all ratio values. This gives the empirical distribution of ratios on actual LC candidates.

### Task 3: Define heuristic for `verify_match_ratio` per model

**Current state:**
- VGGT-X at layer=20: random pairs 0.687–0.823, LC candidates unknown but some >0.85 (2 fired)
- MapAnything at layer=-1 (wrong layer): random pairs 0.332–0.743 (layer TBD from Task 1)
- VGGT-Omega: not yet measured

**Heuristic goal:** Given a model's attention distribution on overlapping pairs, set `verify_match_ratio` such that:
- True overlapping LC candidates pass with ~90%+ recall
- Non-overlapping pairs are rejected

**Process:**
1. Get distribution on DINO-SALAD retrieved pairs (Task 2 above)
2. Get distribution on non-overlapping pairs (large temporal gap, e.g., gap > 100 frames)
3. Set threshold at the point that separates the two distributions (e.g., mean of retrieved pairs - 2*std)

**Implementation:** Override `default_verify_match_ratio` in each creator class after calibration. This already works — `LoopClosure` wrapper reads from `cfg.verify_match_ratio` which defaults from `LoopClosureConfig(verify_match_ratio=0.85)`. To use per-model defaults, pass `LoopClosureConfig(verify_match_ratio=creator.default_verify_match_ratio)` or add logic to the wrapper.

### Task 4: ATE comparison across all methods + VGGT-SLAM

**Required runs on chess_seq01 (200 frames, submap_size=16 for windowed methods):**

| Condition | Command | Expected baseline |
|---|---|---|
| VGGT-X baseline (no LC) | `--backbone vggtx --conditions baseline` | 0.3184m |
| VGGT-X + LC | `--backbone vggtx --conditions lc` | 0.2664m (confirmed) |
| VGGT-Omega baseline | `--backbone vggt_omega --conditions baseline` | unknown |
| VGGT-Omega + LC | `--backbone vggt_omega --conditions lc` | unknown |
| MapAnything baseline | `--backbone mapanything --conditions baseline` | unknown |
| MapAnything + LC | `--backbone mapanything --conditions lc` | unknown |
| VGGT-SLAM (no LC) | via `evals/eval_vggt_slam_comparison.py` | 0.2959m |
| VGGT-SLAM (with LC) | (need to preserve TUM file — see below) | unknown |

**VGGT-SLAM LC result was lost** in a prior session (seq dir deleted before archiving). Next run: save to `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` before cleanup.

**eval commands:**
```bash
# VGGT-X (all conditions)
python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggtx \
  --submap_size 16 \
  --max_frames 200 \
  --conditions baseline lc

# VGGT-Omega (all conditions)
python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_omega \
  --submap_size 16 \
  --max_frames 200 \
  --conditions baseline lc

# VGGT-SLAM (save LC TUM before cleanup!)
python evals/eval_vggt_slam_comparison.py \
  --scene chess --seq seq-01 \
  --data_root evals/data/7scenes/chess/chess \
  --conditions vggt_slam_oob
```

---

## Key Files

| File | Purpose |
|---|---|
| `collab_splats/pointcloud/feedforward/base.py:550–565` | `_lc_layer_index`, `_lc_token_offset`, `default_verify_match_ratio` class attrs |
| `collab_splats/pointcloud/feedforward/base.py:697–727` | `_verify_loop_candidate` — uses class attrs |
| `collab_splats/pointcloud/feedforward/vggtx.py` | `_lc_layer_index` inherited as 20 (no override needed) |
| `collab_splats/pointcloud/feedforward/mapanything.py:128` | `_lc_token_offset=0` set; `_lc_layer_index` still 20 (may be wrong — Task 1) |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | No `_lc_layer_index` override yet (Task 1) |
| `collab_splats/pointcloud/utils.py:930` | `cross_frame_attention_ratio` — mean_top_quarter aggregation |
| `evals/eval_similarity_calibration.py` | Calibration script — run with `--models vggtx mapanything omega` |
| `evals/eval_gt.py` | Main eval script |
| `evals/eval_vggt_slam_comparison.py` | VGGT-SLAM comparison |
| `worklog/notes/2026-05-28-lc-layer-fix.md` | Full session notes |

## Architecture Quick Reference

- VGGT-X: `model.aggregator.global_blocks` — depth=24; LC taps block 20
- MapAnything: `model.info_sharing.self_attention_blocks` — depth unknown; no special tokens (token_offset=0)
- VGGT-Omega: `model.aggregator.inter_frame_blocks` — depth unknown
- DINO-SALAD retrieval: L2 distance on normalized embeddings (separate from attention gate)
- Two gates: (1) DINO-SALAD L2 gate (`lc_cosine_threshold=0.75`) → (2) attention verify gate (`verify_match_ratio=0.85`)
