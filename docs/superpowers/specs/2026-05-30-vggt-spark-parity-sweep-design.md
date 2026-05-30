# Design: VGGT-SPARK Parity Sweep

**Date:** 2026-05-30
**Branch:** refactor/cu121
**Goal:** Establish numerical parity between our `vggt_spark` pipeline and VGGT-SLAM on chess/seq-01, sweeping `min_disparity` from 50 → 0.

---

## Background

Prior investigation (`2026-05-30-vggt-spark-similarity-parity.md`) identified two gaps:

1. **Similarity score gap:** our `cross_frame_attention_ratio` produces ~0.818; VGGT-SLAM's `image_match_ratio` produces ~1.02 on same frame pairs.
2. **Baseline ATE gap:** our vggt_spark = 0.3191m vs VGGT-SLAM = 0.0176m on 200 consecutive frames — root cause is frame selection, not stitching (VGGT-SLAM keyframe-filters via `min_disparity=50`).

The disparity sweep uses VGGT-SLAM's `selected_frames.txt` to guarantee both pipelines run on **identical frames**, isolating any remaining gap to stitching/graph logic or the similarity gate.

---

## Frame Parity Mechanism

1. `run_vggt_slam_lc.py --min_disparity d` → filters frames via optical flow, writes `selected_frames.txt`
2. `eval_gt.py --keyframe_list <path>` → reads those exact paths, symlinks into temp dir, passes to feedforward creator

Same files, same order. Any remaining ATE gap is purely from pipeline logic.

---

## Sweep Protocol

Disparity levels: **50 → 30 → 20 → 10 → 0**

For each level d:

### Step 1: Baseline parity

```bash
# VGGT-SLAM baseline
python evals/runners/run_vggt_slam_lc.py \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --max_frames 200 --min_disparity <d> --max_loops 0 \
  --out_tum evals/baselines/disparity_sweep/slam_d<d>/baseline.tum

# Our pipeline (same frames)
python evals/eval_gt.py \
  --dataset 7scenes --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_spark --conditions baseline \
  --submap_size 16 --lc_scale_method none \
  --keyframe_list evals/baselines/disparity_sweep/slam_d<d>/selected_frames.txt
```

Gate: `|our_ATE - slam_ATE| / slam_ATE > 10%` → **STOP, report.**

### Step 2: Similarity calibration checkpoint (d=50 only, hard gate)

After baseline parity confirmed at d=50, run both pipelines with LC and compare **per-pair similarity scores** before checking LC ATE:

- VGGT-SLAM: `image_match_ratio` logged per candidate pair (forward hook in runner)
- Our pipeline: `cross_frame_attention_ratio` logged per candidate pair (`_verify_loop_candidate`)

For every pair where VGGT-SLAM ran verification, compare scores directly.

**If scores differ** (expected gap ~0.818 vs ~1.02):
- Apply Track A: override `_verify_loop_candidate` in `VGGTSPARKCreator` to use `model.forward(compute_similarity=True)` + read `image_match_ratio` directly
- Recalibrate threshold (0.85 → ~0.95 to match VGGT-SLAM's `lc_thres`)
- Rerun d=50 LC, confirm scores match numerically and loop counts match
- Only then continue sweep

**Similarity scores must match before proceeding to d=30.**

### Step 3: LC parity (all levels)

```bash
# VGGT-SLAM with LC
python evals/runners/run_vggt_slam_lc.py \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --max_frames 200 --min_disparity <d> --max_loops 1 \
  --out_tum evals/baselines/disparity_sweep/slam_d<d>/lc.tum

# Our pipeline with LC (same frames)
python evals/eval_gt.py \
  --dataset 7scenes --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_spark --conditions lc \
  --submap_size 16 --lc_scale_method none \
  --keyframe_list evals/baselines/disparity_sweep/slam_d<d>/selected_frames.txt
```

Gate: same 10% threshold on LC ATE. Also check loop count parity (same candidate pairs accepted/rejected).

---

## Fix Tracks

### Track A — Native similarity (apply at d=50 similarity gate)

Override `_verify_loop_candidate` in `VGGTSPARKCreator`. Logic:

1. Stack `frame1`, `frame2` into the tensor shape expected by `VGGT.forward()` — verify against `third_party/vggt_spark/vggt/models/vggt.py` at implementation time (likely `(S, C, H, W)` with S=2, no batch dim).
2. Call `self.model(images, compute_similarity=True)`.
3. Read `output["image_match_ratio"]` — scalar float.
4. Log ratio at INFO level.
5. Accept if `ratio >= verify_match_ratio` (default 0.95, matching VGGT-SLAM's `lc_thres`).
6. Return `(accepted, output.get("extrinsic"))` — extrinsic shape `(2, 3, 4)` if present, else None.

Threshold 0.95 matches VGGT-SLAM's `lc_thres` default.

### Track B — Stitching debug (if baseline parity fails on same frames)

Instrument `closure.py:run_pose_graph_optimization` to log H matrices per submap boundary. Compare against `solver.py:add_edge` in VGGT-SLAM. Find first diverging boundary.

Track B is only activated if baseline ATE diverges despite identical input frames.

---

## Deliverables

| # | File | Purpose |
|---|------|---------|
| 1 | `evals/runners/run_disparity_sweep.py` | Sweep harness: loops disparity levels, runs both pipelines, prints comparison table, stops on gate failure |
| 2 | `collab_splats/pointcloud/feedforward/vggt_spark_creator.py` | Track A: `_verify_loop_candidate` override |
| 3 | `collab_splats/pointcloud/loop_closure/closure.py` | Track B (if needed): H-matrix debug logging |

---

## Success Criteria

- `vggt_spark` baseline ATE within 10% of VGGT-SLAM at all disparity levels (same frames)
- Similarity scores numerically match at d=50 before continuing sweep
- `vggt_spark` LC ATE within 10% of VGGT-SLAM at all disparity levels
- Loop counts match (same accept/reject decisions)

Once parity confirmed for `vggt_spark`, apply Track A override to `VGGTXCreator` and `MapAnythingCreator`.

---

## Out of Scope

- Other sequences / datasets (chess/seq-01 only for now)
- Other backbones until `vggt_spark` parity confirmed
- Retrieval-stage parity (SALAD vs our DinoSalad) — only geometry verification gate
