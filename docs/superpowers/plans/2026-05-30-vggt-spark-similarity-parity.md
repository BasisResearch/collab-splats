# Handoff: VGGT-SPARK Similarity Parity Investigation

**Date:** 2026-05-30
**Branch:** refactor/cu121
**Goal:** Make our `vggt_spark` pipeline produce identical ATE results to VGGT-SLAM on the same sequences.

---

## What We Know

### Eval results (2026-05-30, `eval_gt.py`, chess/seq-01)

**29-frame (max_frames=29, submap_size=16, lc_scale_method=none):**

| Pipeline | Baseline ATE | LC ATE |
|---|---|---|
| Our vggt_spark | 0.0105m | 0.0103m (0 loops, windowed stitching only) |
| VGGT-SLAM | 0.0176m (no LC) | 0.0165m (with LC, 11 loops) |

**200-frame (max_frames=200, submap_size=16, lc_scale_method=none):**

| Pipeline | Baseline ATE | LC ATE | Loops |
|---|---|---|---|
| Our vggt_spark | 0.3191m | 0.3347m | 1 (false positive, net harmful) |
| Our vggtx | 0.3191m | 0.3353m | 1 (same false positive) |
| Our vggt_omega | 0.3223m | 0.2652m | 12 (genuine, −17.7%) |
| VGGT-SLAM | 0.0176m | 0.0165m | 11 (all genuine, −6.3%) |

**Target:** Our `vggt_spark` LC ATE should match VGGT-SLAM's ~0.017m on 200f and ~0.016m with LC.

---

## Root Cause Hypothesis

### 1. Similarity score gap (primary issue)

Our `cross_frame_attention_ratio` on vggtx/spark produces scores **~0.818 mean** (range 0.80–0.84).  
VGGT-SLAM's native verification produces scores **~1.02–1.05** on the same model.

Same weights (`facebook/VGGT-1B`), same frames — different scores. This means:
- VGGT-SLAM uses `compute_similarity=True` in `VGGT.forward()`, a **native model path** inside VGGT-SPARK's implementation
- Our pipeline uses `extract_intermediate_features` (register_forward_hook on `aggregator.global_blocks[layer].attn.qkv`) then `cross_frame_attention_ratio`
- These tap different parts of the model or aggregate differently → ~20% score gap

VGGT-SLAM's similarity is computed as `get_similarity()` in `third_party/vggt_spark/vggt/utils/helper.py`.  
Our `cross_frame_attention_ratio` is in `collab_splats/pointcloud/utils.py`.

### 2. Baseline ATE gap (0.3191m vs 0.0176m)

Our vggt_spark and vggtx baselines give **0.3191m** on 200 consecutive frames.  
VGGT-SLAM gives **0.0176m** on the same frames.

Both use `facebook/VGGT-1B` weights. This large gap means either:
- VGGT-SLAM's windowed submap stitching/pose graph construction differs fundamentally from ours (the known hypothesis from the 2026-05-28 parity investigation — `proj_mats` vs SE3 extrinsics)
- Or VGGT-SLAM keyframe-selects differently even at `min_disparity=0`

---

## Key Code Locations

### Our pipeline

| Component | Location |
|---|---|
| similarity gate | `collab_splats/pointcloud/feedforward/base.py:_verify_loop_candidate` (L851) |
| `cross_frame_attention_ratio` | `collab_splats/pointcloud/utils.py` |
| feature extraction hook | `collab_splats/pointcloud/feedforward/base.py:extract_intermediate_features` |
| vggt_spark creator | `collab_splats/pointcloud/feedforward/vggt_spark_creator.py` |
| LC wrapper | `collab_splats/pointcloud/wrappers.py:LoopClosure._run_lc_loop` |
| Pose graph | `collab_splats/pointcloud/loop_closure/closure.py:run_pose_graph_optimization` |

### VGGT-SLAM

| Component | Location |
|---|---|
| solver (pose graph) | `third_party/VGGT-SLAM/vggt_slam/solver.py` |
| native similarity | `third_party/vggt_spark/vggt/utils/helper.py:get_similarity` |
| LC matching | `third_party/VGGT-SLAM/vggt_slam/loop_closure.py` |
| our runner script | `evals/runners/run_vggt_slam_lc.py` |

---

## Investigation Tasks

### Task 1: Compare similarity functions

**Goal:** Understand exactly why our scores are ~0.818 and VGGT-SLAM's are ~1.02.

1. Read `third_party/vggt_spark/vggt/utils/helper.py:get_similarity` — understand how it aggregates Q/K features vs our `cross_frame_attention_ratio`.
2. Read `collab_splats/pointcloud/utils.py:cross_frame_attention_ratio`.
3. Run both on the **same frame pair** and compare scores. Use one of the accepted VGGT-SLAM pairs (they were submap pairs from chess/seq-01 200f run).
4. Identify the exact difference: different layer? different token aggregation? different normalization?

### Task 2: Make VGGTSPARKCreator use native `compute_similarity`

`VGGTSPARKCreator` loads the model via the SPARK import path, which supports `compute_similarity=True` in `VGGT.forward()`. When this flag is True, the model computes `image_match_ratio` as a side output alongside pose/depth predictions.

**Goal:** Override `_verify_loop_candidate` in `VGGTSPARKCreator` to call `VGGT.forward(compute_similarity=True)` on the candidate pair and read `image_match_ratio` directly, matching VGGT-SLAM's native path.

Expected: scores shift from ~0.818 to ~1.02 range, threshold needs recalibration.

Reference: `third_party/VGGT-SLAM/vggt_slam/loop_closure.py` — how VGGT-SLAM calls `compute_similarity`.

### Task 3: Investigate baseline ATE gap (0.3191m vs 0.0176m)

This is the larger gap and separate from similarity. Prior investigation (2026-05-28 handoff) identified the root cause as our SE3→SL4 lift vs VGGT-SLAM's native `proj_mats`. Reference: `docs/superpowers/plans/2026-05-28-pose-graph-parity-handoff.md`.

After fixing similarity (Task 2), re-check baseline ATE. If it's still 0.3191m vs 0.0176m, the pose graph construction is the remaining gap.

### Task 4: Verify parity on both frame configs

After fixes, run and confirm:

```bash
# 29-frame: our pipeline
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_spark --conditions baseline lc \
  --submap_size 16 --lc_scale_method none --max_frames 29

# 29-frame: VGGT-SLAM (reference)
/opt/conda/envs/reconstruction/bin/python evals/runners/run_vggt_slam_lc.py \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --max_frames 29 --min_disparity 0 --max_loops 1

# 200-frame: our pipeline
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_spark --conditions baseline lc \
  --submap_size 16 --lc_scale_method none --max_frames 200

# 200-frame: VGGT-SLAM (reference)
/opt/conda/envs/reconstruction/bin/python evals/runners/run_vggt_slam_lc.py \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --max_frames 200 --min_disparity 0 --max_loops 1
```

**Success criteria:** Our vggt_spark LC ATE within 10% of VGGT-SLAM on both frame counts.

---

## Current Diagnostic Logging

`base.py:_verify_loop_candidate` now has `logger.info` calls logging ratio per candidate.  
**Issue:** Python logging defaults to WARNING — ratio values don't appear in eval output.  
**Fix needed:** Either switch to `console.log` (Rich, always visible) or configure `logging.basicConfig(level=logging.INFO)` in `evals/eval_gt.py`.

---

## Python env / run env

```bash
/opt/conda/envs/reconstruction/bin/python  # always use this
# OOM risk — never run parallel GPU processes
# Run heavy evals in tmux
seq_dir: evals/data/7scenes/chess/chess/seq-01
```

---

## Prior Work

- `docs/superpowers/plans/2026-05-28-pose-graph-parity-handoff.md` — SE3 vs SL4 pose graph root cause
- `docs/superpowers/specs/2026-05-29-pipeline-parity-investigation-design.md` — original parity investigation design
- `worklog/notes/2026-05-29-lc-threshold-recalibration.md` — calibration data and eval results
