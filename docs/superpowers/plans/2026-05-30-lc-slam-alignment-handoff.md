# Handoff: Loop Closure VGGT-SLAM Alignment Investigation

**Date:** 2026-05-30  
**Branch:** refactor/cu121  
**Goal:** Make our SL4 pose graph / submap stitching match VGGT-SLAM exactly.

---

## Current State

### What works

- Single-submap ATE (d=50, d=30, d=20): **~0.018-0.024m ≈ SLAM** ✓
- Native `compute_similarity=True` in `VGGTSPARKCreator._verify_loop_candidate` ✓
- ATE metric fixed (SE3→Sim3 alignment via `umeyama_sim3`) ✓
- Sweep harness: `evals/runners/run_disparity_sweep.py` ✓

### What fails

- Multi-submap (d=10, 2 submaps): **0.181m vs SLAM's 0.018m** (10× gap)
- Gap appeared exactly when number of submaps > 1
- Single-pass on same 26 frames gives 0.019m ✓ — gap is purely in stitching

---

## Changes Already Applied (closure.py / graph.py)

| Change | File | Effect |
|--------|------|--------|
| Remove `normalize_to_sl4` | `graph.py` | 0.308m → 0.308m (no effect) |
| Remove prior on first node | `closure.py` | 0.308m → 0.308m (no effect) |
| H_w[0] = I (not poses[0]) | `closure.py` | — |
| T = inv(K_prev) @ K_curr | `closure.py` | — |
| Extraction: K_4x4 @ inv(H_opt) | `closure.py` | 0.308m → **0.181m** |
| scale_method default = rotation_only | `closure.py` | 0.181m → 0.181m (no effect) |

**Current default `scale_method` = `"rotation_only"`** (matches SLAM default).

---

## Critical Finding: Our Comment Was Wrong

In `closure.py` the comment says:
```
# No prior — matches VGGT-SLAM which never calls add_prior_factor
```

**This is WRONG.** The Explore agent found that SLAM **DOES** call `add_prior_factor` on the first node:
```python
# solver.py ~line 59-61
H_w_submap = np.eye(4)
self.graph.add_homography(0, H_w_submap)
self.graph.add_prior_factor(0, H_w_submap)  # ← SLAM ADDS PRIOR
```

We removed the prior (change 2 above) thinking SLAM doesn't use it. **We need to add it back.**

---

## Full Step-by-Step Diff (SLAM vs Ours)

### ✅ Match

| Step | SLAM | Ours |
|------|------|------|
| Poses storage | `world_to_cam` SE3 | `submap.poses` SE3 |
| Inner H_inner | `W[i-1] @ inv(W[i])` | same |
| T formula | `inv(K_prev[-1]) @ K_curr[0]` | same (after fix) |
| H_w formula | `H_overlap @ T @ H_scale` | same |
| Extraction | `K[i] @ inv(H_opt[i])` | same (after fix) |
| decompose_camera | from SLAM's slam_utils | copied verbatim |
| estimate_scale_pairwise | from SLAM | imported same fn |
| normalize_to_sl4 | NOT called (commented out) | removed (after fix) |

### ❌ Differ

| Step | SLAM | Ours | Action |
|------|------|------|--------|
| **Prior on node 0** | `add_prior_factor(0, I)` | removed | **RE-ADD** |
| **H_w[0] init** | `np.eye(4)` | `np.eye(4)` (after fix) ✓ | done |
| Confidence fallback | AND → less-AND → all | AND → OR → all | investigate |
| world_points computation | `unproject_depth_map_to_point_map` (SLAM util) | `_raw_to_world_points` (our impl) | verify equivalence |

---

## What to Test Next (in order)

### Test 1: Re-add prior (most likely remaining fix)

In `collab_splats/pointcloud/loop_closure/closure.py`, around line 420:
```python
if s_idx == 0:
    pg.add_node(node_ids_this[0], np.eye(4))
    pg.add_prior(node_ids_this[0], np.eye(4))   # ← ADD THIS BACK, prior = I
```

Run d=10 baseline. Expected: ATE drops from 0.181m toward 0.018m.

### Test 2: Verify world_points equivalence

SLAM: `unproject_depth_map_to_point_map(depth, extrinsics_cam, intrinsics_cam)` — external SPARK util.
Ours: `_raw_to_world_points(raw_lc)` in `collab_splats/pointcloud/feedforward/base.py`.

Read both implementations and confirm:
- Same unprojection formula (pixel → camera → world via extrinsics)
- Same coordinate frame (world frame, not camera frame)
- Same subsampling/indexing of depth pixels

### Test 3: Confidence fallback alignment

SLAM:
```python
good_mask = (prior_conf > thresh) & (current_conf > thresh)  # AND primary
if good_mask.sum() < 100:
    good_mask = (prior_conf > thresh)                          # single-side fallback
if good_mask.sum() < 100:
    good_mask = (prior_conf > 0)                              # permissive fallback
```

Ours (in closure.py scale estimation block):
```python
joint_mask = (curr_conf > conf_threshold) & (prev_conf > conf_threshold)  # AND
if joint_mask.sum() >= _MIN_CONF_POINTS: mask = joint_mask
else:
    either_mask = (curr_conf > conf_threshold) | (prev_conf > conf_threshold)  # OR
    if either_mask.sum() >= _MIN_CONF_POINTS: mask = either_mask
```

SLAM falls back to `prior_conf > thresh` only (not OR). Match SLAM's fallback exactly.

### Test 4: Full sweep after fixes

After Test 1-3, re-run `run_disparity_sweep.py` for all disparity levels [50,30,20,10,0].
Target: all levels within 10% of SLAM.

---

## How to Run Tests

```bash
# d=10 baseline (quick, GPU, ~3 min)
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_spark \
  --conditions baseline \
  --submap_size 16 \
  --lc_scale_method rotation_only \
  --keyframe_list evals/baselines/disparity_sweep/slam_d10/selected_frames.txt \
  --output_ate /tmp/d10_test.json

cat /tmp/d10_test.json
# Target: {"baseline": ~0.0176}

# SLAM reference (for comparison)
cat evals/baselines/disparity_sweep/slam_d10/metrics.json
# ate_rmse: 0.017631
```

---

## Key File Locations

| File | Purpose |
|------|---------|
| `collab_splats/pointcloud/loop_closure/closure.py` | `run_pose_graph_optimization` — graph construction + extraction |
| `collab_splats/pointcloud/loop_closure/graph.py` | `PoseGraph` — GTSAM SL4 wrapper |
| `collab_splats/pointcloud/loop_closure/submap.py` | `Submap` dataclass |
| `collab_splats/pointcloud/wrappers.py` | `LoopClosure._run_windowed_lc` — submap creation loop |
| `collab_splats/pointcloud/feedforward/base.py` | `_raw_to_world_points` |
| `third_party/VGGT-SLAM/vggt_slam/solver.py` | SLAM reference: `add_points`, `add_edge` |
| `third_party/VGGT-SLAM/vggt_slam/graph.py` | SLAM reference: `PoseGraph` |
| `third_party/VGGT-SLAM/vggt_slam/submap.py` | SLAM reference: `get_all_poses_world` |
| `evals/runners/run_disparity_sweep.py` | Sweep harness |
| `evals/baselines/disparity_sweep/` | Sweep results (SLAM and ours per disparity) |

---

## Python Env

```bash
/opt/conda/envs/reconstruction/bin/python  # always use this
```

Data: `evals/data/7scenes/chess/chess/seq-01`
