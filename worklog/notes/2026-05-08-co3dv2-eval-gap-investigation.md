# CO3Dv2 Eval Gap Investigation

**Date:** 2026-05-08  
**Branch:** `refactor/core-modules`  
**Goal:** Close the gap between our AUC@30 results and BAE paper Table IV (VGGT+BAE = 90.0).

---

## Current Results vs Paper

| Method | AUC@30 | Notes |
|---|---|---|
| VGGT init (paper) | 88.2 | Vanilla VGGT, full sequences, 10 categories |
| VGGT + BAE (paper) | 90.0 | Same + bundle adjustment |
| VGGT-X baseline, seq1, 50fr | **74.4** | Our best result so far |
| VGGT-X + BA, seq1, 50fr | 70.1 | BA slightly hurts at 50 frames |
| VGGT-X baseline, seq2, 50fr | 54.2 | Worse — likely hard sequence |
| VGGT-X baseline, seq3, 50fr | 0.0 | Aspect ratio failure (2000×900) |

All results: `eval_results/co3dv2_apple_*/metrics.json` on branch `refactor/core-modules`.

---

## Known Differences (in order of likely impact)

### 1. Sequence length — HIGH IMPACT

**Paper:** full sequences (typically 100–300 frames per CO3Dv2 sequence)  
**Ours:** `--max_frames 50` (truncated)

BA cannot improve short sequences — there is no drift to correct. This is why BA hurts or is neutral at 50 frames. The same pattern appeared on 7-Scenes chess (50 frames: baseline ATE 0.055m, BA 0.058m — marginally worse).

**Fix:** Re-run seq1 without `--max_frames` cap:
```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/110_13051_23361 \
  --output_dir ./eval_results/co3dv2_apple_seq1_full \
  --conditions baseline ba ba_hightrack
```
seq1 has ~200 frames available (check: `ls data/co3dv2/apple/110_13051_23361/images/ | wc -l`).

---

### 2. Image preprocessing — HIGH IMPACT

**Paper:** `load_and_preprocess_images_square` (square-crops images, no aspect ratio distortion for the model)  
**Ours:** `load_and_preprocess_images_ratio` (preserves aspect ratio, pads to fixed size)

File: `collab_splats/pointcloud/feedforward/vggtx.py:21,253`

```python
# Current (ours):
from vggt.utils.load_fn import load_and_preprocess_images_ratio
images, original_coords = load_and_preprocess_images_ratio(...)
```

The comment in vggtx.py:30 notes: "VGGT-X processes images at a fixed square resolution. load_and_preprocess_images_ratio" — this was a deliberate choice, but it differs from the paper's evaluation setup.

**Effect:** CO3Dv2 images are typically portrait/landscape shots. Square-cropping vs ratio-padding changes the effective field of view and feature distribution seen by the model. For unusual aspect ratios (seq3: 2000×900), ratio preprocessing makes reconstruction catastrophically worse.

**Fix to investigate:** Try swapping to `load_and_preprocess_images_square` for the CO3Dv2 eval:
```python
from vggt.utils.load_fn import load_and_preprocess_images_square
```
Compare AUC@30 on seq1 with square vs ratio preprocessing.

---

### 3. Number of categories and sequences — MEDIUM IMPACT

**Paper:** 10 categories averaged (apple, ball, banana, bench, book, bottle, bowl, broccoli, car, chair)  
**Ours:** 1 category (apple), 3 sequences only

Single-category results are noisy. seq2 (54.2) and seq3 (0.0) drag the average down significantly. The paper's 88.2 is a macro-average over hundreds of sequences.

**Fix:** Download remaining 9 categories:
```bash
for CAT in ball banana bench book bottle bowl broccoli car chair; do
  bash evals/download_co3dv2.sh $CAT ./data/co3dv2
done
```
Then loop eval over all downloaded sequences.

---

### 4. Track extraction parameters — MEDIUM IMPACT

**Paper:** `max_query_pts=4096, query_frame_num=8`  
**Our `ba`:** `max_query_pts=2048, query_frame_num=5`  
**Our `ba_hightrack`:** `max_query_pts=4096, query_frame_num=8` ← matches paper

We have `ba_hightrack` to match paper defaults. Run it on full sequences to see if it closes the gap:
```bash
--conditions baseline ba ba_hightrack
```

---

### 5. Aspect ratio crash on seq3 (2000×900) — KNOWN BUG

seq3 (`540_79043_153212`) images are 2000×900 (14:1 aspect ratio). Two failures:
- **baseline AUC@30 = 0.0**: VGGT-X rotation predictions completely fail — mean R_err ≈ 124° after Umeyama alignment. Model was not trained on such extreme aspect ratios.
- **BA crashes**: LightGlue SIFT `ValueError: array is not broadcastable` on 2000×900 images.

**Fix:** Pre-crop or resize extreme-aspect-ratio images before inference. Or filter such sequences out of the benchmark.

---

### 6. VGGT-X vs vanilla VGGT backbone — STRUCTURAL DIFFERENCE

**Paper:** vanilla VGGT (no depth/normal heads)  
**Ours:** VGGT-X (adds depth + normal prediction heads, modified architecture)

This means absolute number matching is not expected. The goal is to show the BA improvement trend holds. Current results:
- seq1: BA 70.1 vs baseline 74.4 — BA hurts (likely due to 50-frame limitation, not model)
- seq2: BA 50.2 vs baseline 54.2 — same pattern

---

### 7. CO3Dv2 convention fix — COMPLETED

**Status:** Fixed and committed (`339c46f`).

PyTorch3D stores R in row-major left-handed convention. Conversion to OpenCV:
```python
_S = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
pose[:3, :3] = _S[:, None] * R.T   # R_cv = diag(-1,-1,1) @ R.T
pose[:3, 3] = _S * T                # T_cv = diag(-1,-1,1) @ T
cy = -py_ndc * s + H / 2.0          # NDC y-axis flipped vs image y
```

Before this fix: AUC@30 = 0.0 for all sequences. After: seq1 = 74.4.

File: `evals/datasets.py:_load_co3dv2`

---

### 8. `frame_annotations.jgz` at category level — COMPLETED

**Status:** Fixed and committed (agent in prior session).

The single-sequence-subset download puts `frame_annotations.jgz` at `apple/` (category level), not inside `apple/SEQUENCE_NAME/`. The loader now falls back to `seq_dir.parent / "frame_annotations.jgz"` and filters by sequence name.

File: `evals/datasets.py:_load_co3dv2:166-188`

---

## Recommended Investigation Order

1. **Full sequence run** (seq1, no frame cap) — isolates length effect. Expected: BA should help.
2. **Square vs ratio preprocessing** on seq1 — expected: square gets closer to paper's 88.2 baseline.
3. **ba_hightrack on full sequences** — expected: narrows paper gap further.
4. **More categories** — needed for a publishable comparison.
5. **seq3 aspect ratio fix** — filter or pre-process extreme-AR images.

## Key Files

| File | Role |
|---|---|
| `evals/datasets.py:_load_co3dv2` | CO3Dv2 loader (convention-fixed) |
| `evals/eval_gt.py` | Runner: `--dataset co3dv2 --conditions baseline ba ba_hightrack` |
| `evals/download_co3dv2.sh` | Download single category |
| `collab_splats/pointcloud/feedforward/vggtx.py:253` | `load_and_preprocess_images_ratio` call to investigate |
| `collab_splats/pointcloud/loop_closure/eval.py:auc_at_threshold` | AUC@30 metric |
| `eval_results/co3dv2_apple_*/metrics.json` | All current results |

## Commands to Reproduce

```bash
# Full seq1 eval (primary investigation)
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/110_13051_23361 \
  --output_dir ./eval_results/co3dv2_apple_seq1_full \
  --conditions baseline ba ba_hightrack

# Check frame count
ls data/co3dv2/apple/110_13051_23361/images/ | wc -l

# Download more categories
bash evals/download_co3dv2.sh ball ./data/co3dv2
```
