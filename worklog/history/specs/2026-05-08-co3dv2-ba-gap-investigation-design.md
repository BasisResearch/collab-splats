# CO3Dv2 BA Gap Investigation Design

**Date:** 2026-05-08  
**Branch:** `refactor/core-modules`  
**Goal:** Close the AUC@30 gap between our results and BAE paper (VGGT+BAE = 90.0), and diagnose why BA currently hurts (70.1 vs 74.4 baseline at 50fr).

---

## Background

| Method | AUC@30 | Notes |
|---|---|---|
| VGGT init (paper) | 88.2 | Vanilla VGGT, full sequences, 10 categories |
| VGGT + BAE (paper) | 90.0 | Same + bundle adjustment |
| VGGT-X baseline, seq1, 50fr | 74.4 | Our best result |
| VGGT-X + BA, seq1, 50fr | 70.1 | BA hurts at 50 frames |

Two HIGH IMPACT hypotheses for the gap:
1. **Preprocessing**: paper uses `load_and_preprocess_images_square`; we use `load_and_preprocess_images_ratio`
2. **Sequence length**: paper uses full sequences (100–300fr); we cap at 50 frames (no drift → BA cannot help)

---

## Investigation Approach

Option A (chosen): preprocessing first (isolated, fast), then scale.

Rationale: preprocessing is a 1-line change + 10-min eval. If square boosts baseline to ~85+, the full-sequence run is better-informed. Avoids running a 200-frame eval only to discover preprocessing was the blocker.

---

## Steps

### Step 1 — Square vs ratio preprocessing, seq1, 50fr (~10 min)

**Change:** In `collab_splats/pointcloud/feedforward/vggtx.py:253`:
```python
# From:
from vggt.utils.load_fn import load_and_preprocess_images_ratio
images, original_coords = load_and_preprocess_images_ratio(...)

# To:
from vggt.utils.load_fn import load_and_preprocess_images_square
images, original_coords = load_and_preprocess_images_square(...)
```

Note: `load_and_preprocess_images_square` may return only `images` (no `original_coords`). Check the function signature and adjust the intrinsics/coords downstream accordingly.

**Run:**
```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/110_13051_23361 \
  --output_dir ./eval_results/co3dv2_apple_seq1_square_50fr \
  --max_frames 50 \
  --conditions baseline ba ba_hightrack
```

**Decision gate:**
- Does baseline AUC@30 increase above 74.4?
- Does BA now help (ba > baseline)?
- Record both ratio (74.4) and square results for direct comparison.

If baseline does NOT improve with square: preprocessing is not the issue. Investigate track extraction params (Step 1b: check `max_reproj_error`, `min_inliers_per_frame` in BA config).

---

### Step 2 — Full sequence, seq1, best preprocessing (~30–60 min)

**Change:** Use winning preprocessing from Step 1. Remove `--max_frames` cap (seq1 has ~200 frames — verify: `ls data/co3dv2/apple/110_13051_23361/images/ | wc -l`).

**Run:**
```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/110_13051_23361 \
  --output_dir ./eval_results/co3dv2_apple_seq1_full \
  --conditions baseline ba ba_hightrack
```

**Decision gate:**
- Does BA help over baseline? (Expected: yes — drift exists at 200fr.)
- If BA still hurts at full sequence → BA code issue (not data). Investigate: Sim3 graph construction, track confidence thresholds, reprojection error weighting.
- If BA helps → sequence length was the root cause of the BA regression at 50fr.

---

### Step 3 — ba_hightrack on full sequence (+15 min on top of Step 2)

Already included in Step 2 command via `--conditions baseline ba ba_hightrack`.

**Decision gate:**
- `ba_hightrack > ba > baseline`? Should mirror paper's improvement pattern.
- If ba_hightrack < ba: track quality settings need review (`extract_tracks_vggsfm` params).

---

### Step 4 — Scale to seq2 (~30–60 min)

Run winning config (preprocessing + no frame cap) on seq2 (`data/co3dv2/apple/189_20393_38136`).

```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/189_20393_38136 \
  --output_dir ./eval_results/co3dv2_apple_seq2_full \
  --conditions baseline ba ba_hightrack
```

Skip seq3 (`540_79043_153212`, 2000×900) unless AR filter is added first.

---

### Step 5 — More categories (batch job)

Download 2–3 additional CO3Dv2 categories. Run winning config on each. Target: reproduce paper's 10-category benchmark for publishable comparison.

---

## Key Files

| File | Role |
|---|---|
| `collab_splats/pointcloud/feedforward/vggtx.py:253` | Preprocessing swap (Step 1) |
| `evals/eval_gt.py` | Runner |
| `evals/datasets.py:_load_co3dv2` | CO3Dv2 loader |
| `collab_splats/pointcloud/bundle_adjustment.py` | BA implementation |
| `eval_results/co3dv2_apple_*/metrics.json` | Results |

---

## Success Criteria

- Baseline AUC@30 on seq1 full sequence ≥ 85 (within ~3 pts of paper's 88.2)
- BA improves over baseline on full sequences (ba > baseline)
- ba_hightrack ≥ ba
- Results on ≥ 3 sequences for publishable comparison

---

## Fallback Paths

| If... | Then... |
|---|---|
| Square preprocessing doesn't improve baseline | Check vggtx.py intrinsics/coords — `load_and_preprocess_images_square` may need different downstream handling |
| BA still hurts at full seq | Debug BA: Sim3 graph, track confidence thresholds, reprojection error weighting |
| seq2 AUC@30 < 50 (hard sequence) | Note as outlier, add seq3 with AR filter instead |
| GPU OOM on 200fr | Add `--submap_size 50` for windowed inference |
