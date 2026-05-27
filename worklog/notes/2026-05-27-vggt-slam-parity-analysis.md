# VGGT-SLAM Parity Analysis — chess_seq01, 2026-05-27

**Branch:** `refactor/cu121`  
**Related spec:** `docs/superpowers/specs/2026-05-27-vggt-spark-runner-design.md`  
**Artefacts:** `evals/results/parity_harness/`

---

## Executive Summary

Our loop-closure pipeline accepts **0/200 loops** on chess_seq01 while VGGT-SLAM (using VGGT-SPARK) accepts **11**. Root cause is a **similarity score incompatibility** between our VGGT-X attention hooks and the VGGT-1B attention mechanism VGGT-SPARK uses — the same threshold (0.85) is calibrated for VGGT-1B and consistently fails to fire on VGGT-X outputs. Additionally, our solver internals diverge increasingly from VGGT-SLAM's over the sequence, suggesting accumulating scale drift in submap stitching.

---

## Setup

| | Ours | VGGT-SLAM |
|---|---|---|
| Model | VGGT-X (fine-tuned VGGT-1B) | VGGT-1B via VGGT-SPARK |
| Weights | VGGT-X weights | VGGT-1B weights (same base) |
| Seq | chess_seq01, 200 frames | chess_seq01, 200 frames |
| Submap size | 16 | 16 |
| `min_disparity` | **0** (intentional¹) | 50 (paper default) |
| LC threshold | 0.85 | 0.85 |

> ¹ `min_disparity=50` yields 0 LC candidates on chess_seq01 (the scene is near-static chessboard with low camera disparity). We intentionally set 0 to let the similarity gating be the sole gate.

---

## Comparison 1: Similarity Scores

**VGGT-SPARK `image_match_ratio`** (VGGT-1B, 11 LC passes):

| idx | score | accept? |
|---|---|---|
| 0 | 1.0321 | ✓ |
| 1 | 1.0432 | ✓ |
| 2 | 1.0359 | ✓ |
| 3 | 1.0278 | ✓ |
| 4 | 1.0205 | ✓ |
| 5 | 1.0016 | ✓ |
| 6 | 1.0222 | ✓ |
| 7 | 1.0223 | ✓ |
| 8 | 1.0147 | ✓ |
| 9 | 1.0181 | ✓ |
| 10 | 1.0246 | ✓ |
| **mean** | **1.025** | 11/11 |

**Our `cross_frame_attention_ratio`** (VGGT-X, 13 candidate passes):

| range | mean | accept? |
|---|---|---|
| 0.695 – 0.820 | ~0.745 | ✗ (all below 0.85) |

**Root cause:** Both metrics use the same K/Q attention algorithm (our code is a port of VGGT-SPARK's `get_similarity()`). But VGGT-X's fine-tuning changed the attention distribution — its attention scores are systematically lower, placing all candidates ~0.10 below the 0.85 threshold. The threshold was calibrated for VGGT-1B, not VGGT-X.

**Consequence:** 0 loop closures triggered in our pipeline. LC is effectively disabled on this scene.

---

## Comparison 2: Solver Internals (Boundary Diff)

Full diff at `evals/results/parity_harness/boundary_diff.json`.

| Metric | Value |
|---|---|
| Our boundaries | 13 |
| VGGT-SLAM boundaries | 12 |
| mean delta_H_w (Frobenius) | **0.6589** |
| max delta_H_w | **1.5169** (boundary 11) |
| Flagged (> 0.01) | 12/13 |

Divergence is monotonically increasing over the sequence — boundary 0 diff = 0.030, boundary 11 diff = 1.517. This is scale drift accumulation; each submap's world-homography correction compounds the error.

Scale drift (delta_scale) also increases: max 0.141 at boundary 8.

---

## Comparison 3: ATE

| Pipeline | ATE (200 frames) | Notes |
|---|---|---|
| Our baseline (no LC) | 0.3184 m | from `eval_gt.py` |
| Our LC (0 loops triggered) | 0.3184 m | identical to baseline; no correction applied |
| VGGT-SLAM (no LC) | 0.2959 m | from `eval_gt.py` null-hypothesis run |
| VGGT-SLAM (with LC) | n/a | TUM file not preserved |

**Gap analysis:** Even without LC, VGGT-SLAM beats us by 0.023 m. This is a model-quality gap — VGGT-1B produces better per-frame pose estimates than VGGT-X on this scene, independent of loop closure.

---

## Comparison 4: Loop Closure Acceptance

| | Ours | VGGT-SLAM |
|---|---|---|
| LC candidates evaluated | 13 | 11 |
| Accepted | 0 | 11 |
| Acceptance rate | 0% | 100% |
| Threshold | 0.85 | 0.85 |
| Score range | 0.695–0.820 | 1.001–1.043 |

---

## Comparison 5: min_disparity Deviation

VGGT-SLAM default is `min_disparity=50`. We use `min_disparity=0`.

With `min_disparity=50` on chess_seq01, our pipeline finds **0 candidates** because the chessboard scene has limited camera translation — median disparity across frames is below 50px. Using 0 allows candidates through to the similarity gate, but the gate itself then blocks all of them.

Implication: The min_disparity=50 default exists to filter poor-geometry loop candidates. Setting it to 0 without a properly calibrated similarity threshold means we accept all geometry but reject all based on attention similarity. We need either: (a) a lower threshold for VGGT-X, or (b) a scene-adaptive min_disparity.

---

## Action Items

1. **Recalibrate threshold for VGGT-X** — Our scores cluster around 0.74 vs VGGT-1B's ~1.02. A threshold of 0.65–0.70 would likely fire on chess_seq01. Needs validation across scenes to avoid false positives.

2. **Consider score normalization** — VGGT-SPARK's `image_match_ratio` can exceed 1.0 (it's an un-normalized attention ratio). Our port clips differently. Align the normalization.

3. **Investigate scale drift** — The monotonically increasing `delta_H_w` suggests our submap stitching accumulates scale error. Compare world-coordinate reference handling in our `_apply_sim3_correction` vs VGGT-SLAM's equivalent.

4. **Preserve VGGT-SLAM LC TUM** — The `chess_seq01.pending` directory was deleted before the LC TUM was archived. Next run: copy to `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` before cleanup.

5. **Multi-scene validation** — chess_seq01 is a hard case (near-static, low disparity). Run on a scene with larger camera motion to validate LC fires correctly.

---

## Methodology Notes

- **VGGT-SPARK isolation**: Used sys.path shadow (`sys.path.insert(0, third_party/vggt_spark/)`) per-process. No separate conda env needed — VGGT-SPARK is additive-only to VGGT-1B and shares weights.
- **BFloat16 fix**: VGGT-SPARK returns BFloat16 tensors; solver `add_points()` requires float32. Cast applied in `run_vggt_slam_lc.py` before `add_points()`.
- **Similarity hook**: VGGT-SPARK's forward hook fires on all passes; guarded with `if output.get("image_match_ratio") is not None` to skip non-LC passes.
- **Data source**: `evals/results/parity_harness/` — our_internals.json, vggt_spark_similarity.json, boundary_diff.json, our_lc.tum.
