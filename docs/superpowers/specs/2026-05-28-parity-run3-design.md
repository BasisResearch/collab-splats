# Parity Run 3 — LC Calibration + ATE Comparison Design

**Date:** 2026-05-28  
**Branch:** `refactor/cu121`  
**Status:** Ready for implementation plan

---

## Context

LC layer fix landed in `315c312`: VGGT-X now fires 2 loop closures on chess_seq01 (200 frames, submap_size=16), ATE 0.2664m vs 0.3184m baseline (−16.3%). The fix:
- `_lc_layer_index=20` (VGGT-SPARK `target_layer` value, not last block)
- `_lc_token_offset=5` for VGGT-X, `=0` for MapAnything
- `mean_top_quarter` aggregation (matching `get_similarity` exactly)

VGGT-SLAM architecture confirmed: two-gate pipeline identical to ours — DINO-SALAD L2 retrieval → attention verify gate (same 0.85 threshold, same `get_similarity` algorithm). Score gap between our calibration (~0.764–0.823) and VGGT-SPARK reference (1.025) is pair distribution, not implementation: VGGT-SPARK measures on DINO-SALAD retrieved pairs (strongly overlapping); our calibration script currently measures random temporal pairs (gap 5–40 frames). Implementations are equivalent.

---

## Architecture Confirmation

Both pipelines are identical:
1. **DINO-SALAD L2 gate** — `lc_cosine_threshold` (ours: 0.75; VGGT-SLAM: 0.95 default, configurable)
2. **Attention verify gate** — `cross_frame_attention_ratio` / `get_similarity` at layer 20, `mean_top_quarter`, threshold 0.85

VGGT-SLAM calls `model(..., compute_similarity=True)` which only exists in the `vggt_spark` fork — not the installed `vggt` package. Our hook-based `extract_intermediate_features` + `cross_frame_attention_ratio` is the correct equivalent approach.

---

## Tasks

### Task 1 — Layer sweep for MapAnything and VGGT-Omega

**Problem:** `_lc_layer_index=20` is calibrated for VGGT-1B (24 `global_blocks`). MapAnything uses `info_sharing.self_attention_blocks` (depth unknown, likely < 20). VGGT-Omega uses `aggregator.inter_frame_blocks` (depth unknown). Layer index 20 may be out of range or non-optimal.

**Changes to `evals/eval_similarity_calibration.py`:**
- Add `--layer_index INT` flag. When set, override `creator._lc_layer_index` on the instance before evaluating. Allows sweeping without modifying class code.
- Add `--mode {random,retrieved}` flag (see Task 2 — these ship together).

**Process:**
1. Print `len(model.info_sharing.self_attention_blocks)` (MapAnything) and `len(model.aggregator.inter_frame_blocks)` (VGGT-Omega) to know valid range.
2. Sweep layer indices: last (`-1`), last-quarter, half-depth, quarter-depth. Use `--mode retrieved` (Task 2) on DINO-SALAD pairs.
3. Pick layer where mean_top_quarter on retrieved pairs is highest (most peaked cross-frame attention = best overlap signal).
4. Set `_lc_layer_index` class var in `mapanything.py` and `vggt_omega.py`.

### Task 2 — Calibration on DINO-SALAD retrieved pairs

**Problem:** Calibration on random temporal pairs validates aggregation method comparison but not threshold correctness — wrong distribution. VGGT-SPARK's 1.025 reference is on DINO-SALAD retrieved pairs (genuine location revisits). Threshold 0.85 is calibrated on retrieved pairs.

**Changes to `evals/eval_similarity_calibration.py`:**

Add `--mode {random,retrieved}` flag:
- `random` (existing behavior): sample temporal pairs with `--min_gap` / `--max_gap`
- `retrieved` (new): run DINO-SALAD retrieval on the scene, take top-N candidate pairs, compute attention ratios on those pairs

**Retrieved mode implementation:**
1. Load `DinoSaladExtractor` (already used in LC pipeline via `localization.py`)
2. Compute embeddings for all frames in `--scene_dir`
3. For each frame, retrieve top-1 match from other frames (L2 distance, skip temporally adjacent frames with gap < `--min_gap`)
4. Deduplicate pairs, take up to `--n_pairs` pairs
5. Compute `cross_frame_attention_ratio` on each pair (same as random mode)

**Validation gate:** If retrieved-pair mean_top_quarter scores cluster near 1.0+ (matching VGGT-SPARK), implementations are confirmed equivalent. Retrieved pairs become the primary calibration mode — `--mode retrieved` should be the default once validated.

**Default change:** After validation, flip script default from `random` to `retrieved`. Random stays available as a diagnostic (shows the score floor for non-overlapping pairs).

### Task 3 — Per-model `verify_match_ratio` thresholds

**Problem:** Current defaults — VGGT-X: 0.85 (VGGT-SPARK calibration, validated). MapAnything: 0.85 (inherited, uncalibrated). VGGT-Omega: 0.59 (placeholder, not measured).

**Process (after Tasks 1 and 2):**
1. Get retrieved-pair score distribution per model (Task 2 output)
2. Get non-overlapping pair distribution: large temporal gap (gap > 100 frames, same script with `--min_gap 100 --max_gap 200 --mode random`)
3. Set threshold = mean(retrieved) − 2×std(retrieved), floored above max(non-overlapping)
4. Override `default_verify_match_ratio` in each creator class

**Note:** VGGT-X at 0.85 is already validated by the 2 LC triggers on chess_seq01. Don't change it unless retrieved-pair calibration contradicts.

### Task 4 — ATE comparison across all methods + VGGT-SLAM

**Required runs:** chess_seq01, 200 frames, submap_size=16

| Condition | Command | Known result |
|---|---|---|
| VGGT-X baseline | `--backbone vggtx --conditions baseline` | 0.3184m |
| VGGT-X + LC | `--backbone vggtx --conditions lc` | 0.2664m ✓ |
| VGGT-Omega baseline | `--backbone vggt_omega --conditions baseline` | unknown |
| VGGT-Omega + LC | `--backbone vggt_omega --conditions lc` | unknown |
| MapAnything baseline | `--backbone mapanything --conditions baseline` | unknown |
| MapAnything + LC | `--backbone mapanything --conditions lc` | unknown |
| VGGT-SLAM (no LC) | `eval_vggt_slam_comparison.py` | 0.2959m |
| VGGT-SLAM + LC | (save TUM before cleanup) | lost — re-run |

**Required code change:** `eval_gt.py` line 267 — add `mapanything` to `choices=["vggtx", "vggt_omega"]`. Also add `"mapanything"` to `_BACKBONE_PREFIX` dict.

**VGGT-SLAM LC result:** Was lost (seq dir deleted before archiving). Re-run and save to `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` before any cleanup.

**Ordering dependency:** Task 4 LC runs for MapAnything and VGGT-Omega are only meaningful after Tasks 1 and 3 set correct layer index and threshold. Can run baseline conditions immediately. Run LC conditions after Tasks 1–3.

---

## Key Files

| File | Change |
|---|---|
| `evals/eval_similarity_calibration.py` | Add `--layer_index`, `--mode retrieved` |
| `evals/eval_gt.py:267` | Add `mapanything` to backbone choices; update `_BACKBONE_PREFIX` |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Set `_lc_layer_index` after Task 1 sweep |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | Set `_lc_layer_index`, update `default_verify_match_ratio` after Tasks 1+3 |
| `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` | Save before cleanup |

---

## Success Criteria

1. Retrieved-pair scores for VGGT-X cluster near 1.0+ (confirms parity with VGGT-SPARK)
2. `_lc_layer_index` set to validated value for MapAnything and VGGT-Omega
3. `default_verify_match_ratio` derived from retrieved-pair distributions for all models
4. Full ATE table populated: VGGT-X / VGGT-Omega / MapAnything × baseline / lc + VGGT-SLAM × no-lc / lc
5. `--mode retrieved` is default in calibration script (after validation)
