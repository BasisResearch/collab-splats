# VGGT-SPARK Pipeline Parity Handoff
**Date:** 2026-05-29  
**Goal:** Reproduce VGGT-SLAM's 0.038m Sim3 ATE on chess/seq-01 (29 keyframes, 2 submaps) using our VGGTSPARKCreator pipeline.

---

## Current State

### Confirmed facts
- Both use VGGT-SPARK + `facebook/VGGT-1B` weights ✓
- Both use same 29 keyframes from `evals/baselines/vggt_slam/chess_seq01/selected_frames.txt` ✓
- Both use `submap_size=16`, `overlap=1` ✓
- Both call `pose_encoding_to_extri_intri(pred["pose_enc"], image_shape)` for pose extraction ✓

### Fixes applied this session
1. **poses² extraction bug** — `corrected = local_proj @ inv(H_opt)` → `corrected = inv(H_opt)`. First node now `H0 = I` (not `poses[0]`), so graph nodes store `w2c_norm[i] = w2c[i] @ inv(w2c[0])`.
2. **`scale_method="none"` + T=I** — added to `LoopClosureConfig`, wired through `eval_gt.py` baseline condition (was missing `scale_method` in `_make_creator` for baseline).
3. **`vggt_spark` registry** — added `VGGTSPARKCreator` to `pointcloud/__init__.py`.
4. **`eval_gt.py` choices** — added `"vggt_spark"` to `--backbone`, `"none"` to `--lc_scale_method`.

### Current best result
| Condition | SE3 ATE (eval) | Sim3 ATE | Sim3 scale | Step ratio vs GT |
|---|---|---|---|---|
| Our pipeline (scale=none+T=I+poses²fix) | 0.613m | 0.612m | 0.957 (≈1×) | 1.068× |
| VGGT-SLAM reference | 1.072m | **0.038m** | 2.853 (0.35×) | 0.356× |

**Key observation**: Our Sim3 ≈ SE3 ≈ 0.612m — scale correction doesn't help us (trajectory already at correct absolute scale). SLAM has 2.85× depth stretch but near-perfect shape (0.038m residual after Sim3). The gap is in *structural trajectory shape*, not scale.

---

## Open Hypotheses (priority order)

### H1: VGGT inference window composition differs
VGGT-SLAM runner triggers on `len(window) == submap_size + overlapping_window_size = 17`. Our pipeline runs on `submap_size = 16`. VGGT is a transformer — **all outputs for all frames change when you add/remove 1 frame from the window**. The predictions for frames [0..15] from a 16-frame window differ from predictions for [0..15] from a 17-frame window.

**Test:** Patch `wrappers.py` to run inference on windows of size `submap_size + overlap_frames` and discard the last `overlap_frames` predictions from each window (keeping only the first `submap_size`). Compare Sim3 ATE.

Files: `collab_splats/pointcloud/wrappers.py` line ~194-203 (the `_forward` call in the submap sliding loop).

### H2: Image preprocessing / normalization differs
VGGT-SLAM loads images via `cv2.imread` then passes to `run_predictions`. Our `VGGTXCreator._forward` uses `load_and_preprocess_images` from `vggt.utils.load_fn`. These may apply different resizing, normalization, or color channel ordering. If input tensors differ even slightly, VGGT's attention produces different poses.

**Test:** Log the input tensor shape, mean, std for the SAME frame in both pipelines. Check if `load_and_preprocess_images` output matches `cv2.imread` → whatever VGGT-SLAM does.

Files: `evals/runners/run_vggt_slam_lc.py` lines ~306-340 (`run_predictions`), `collab_splats/pointcloud/feedforward/vggtx.py` `_forward`.

### H3: Coordinate extraction convention — K⁻¹ vs raw w2c
VGGT-SLAM extracts poses as `proj_mat = K⁻¹ @ inv(H_i)`. Our extraction uses `inv(H_i)` directly (raw w2c_norm). The K⁻¹ multiplication changes translation magnitude AND direction (due to principal point offset cx, cy). With focal≈540, cx≈320, cy≈240 and image normalized to [-1,1]: the affine shift from cx/cy in K⁻¹ may systematically shift translation directions.

**Test:** In extraction, apply `corrected = inv(K) @ inv(H_opt)` where K is the per-frame predicted intrinsic from `submap.intrinsics[local_i]`. Compare Sim3 ATE.

Files: `closure.py` lines ~553-557 (extraction loop).

### H4: `tranform_submap_to_canonical` aligns world_points to cam_0 frame
VGGT-SLAM calls `solver.tranform_submap_to_canonical(proj_mat_world_to_cam, world_points)` which transforms world_points to cam_0's local frame: `world_points[i] = P_first_cam @ world_points[i]`. Our `world_points` are in the global submap world frame (camera-to-world unprojected). This means world_points in our Submap are in a DIFFERENT coordinate system than VGGT-SLAM's `pointclouds`. Since world_points are used for scale estimation in non-none modes, this changes `estimate_scale_pairwise` results.

**Note:** With `scale_method=none`, world_points are not used — this hypothesis only matters for non-none scale methods.

### H5: GTSAM graph prior strength / noise model differs
VGGT-SLAM uses `intra_submap_noise` and `inner_submap_noise` GTSAM noise models. Our `_SL4PoseGraph` in `graph.py` uses its own noise models. If noise magnitudes differ, the optimizer weights constraints differently and converges to different solutions.

**Test:** Print the GTSAM noise covariance matrices from both pipelines and compare.

Files: `collab_splats/pointcloud/loop_closure/graph.py` (noise constants), `third_party/VGGT-SLAM/vggt_slam/graph.py`.

### H6: Post-optimization extraction uses wrong coordinate frame for submap 1+
After the poses²-fix, the first submap's first frame is at world origin (H0=I). For submap 1+ (non-none mode), the first node is set to `H_overlap = graph.get_H(prev_last_nid)`. This is `w2c_norm[last of submap0]` (in submap 0's normalized frame). The subsequent submap 1 nodes chain from this: `H_i_sub1 = H_overlap @ w2c_sub1[0] @ inv(w2c_sub1[i])`. When we extract via `inv(H_i)`, we get `w2c_sub1[i] @ inv(w2c_sub1[0]) @ inv(H_overlap)` which is the submap1-normalized pose composed with the inverse of submap0's last pose. This may be correct but needs verification vs VGGT-SLAM's convention.

---

## Key code locations

| File | What |
|---|---|
| `collab_splats/pointcloud/wrappers.py:194-235` | Submap sliding window + `_forward` call |
| `collab_splats/pointcloud/loop_closure/closure.py:412-420` | First node H0=I (newly fixed) |
| `collab_splats/pointcloud/loop_closure/closure.py:553-557` | Extraction `corrected = inv(H_opt)` (newly fixed) |
| `collab_splats/pointcloud/loop_closure/closure.py:449-510` | scale_method=none T=I path |
| `evals/runners/run_vggt_slam_lc.py:119-156` | VGGT-SLAM window building (17-frame windows) |
| `third_party/VGGT-SLAM/vggt_slam/solver.py:95-115` | `tranform_submap_to_canonical` |
| `third_party/VGGT-SLAM/vggt_slam/solver.py:118-203` | `add_edge` — full graph construction |
| `evals/baselines/vggt_slam/chess_seq01/vggt_slam_fullseq.tum` | 0.038m reference trajectory |
| `evals/results/7scenes/seq-01/run-20260529-021303/` | Our best result (0.612m Sim3) |

## Recommended next action

**H1 (window size) is the highest-priority hypothesis** because it directly affects what VGGT sees. The 17 vs 16 frame window means different attention context → different pose predictions for ALL frames in the window. Implement in `wrappers.py` and test before investigating H2/H3.

Specifically: change the sliding window in `LoopClosure._run` to pass `window_size = submap_size + overlap_frames` to `_forward`, then use only the first `submap_size` predictions as the submap poses/world_points (dropping the last `overlap_frames`).

## Python env
Always: `/opt/conda/envs/reconstruction/bin/python`  
Heavy eval: run in tmux, not inline (OOM risk at 46.6 GB cap).
