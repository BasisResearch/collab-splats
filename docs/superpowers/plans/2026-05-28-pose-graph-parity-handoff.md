# Pose Graph Parity Investigation — Handoff

**Date:** 2026-05-28  
**Status:** ready for next agent

---

## What We Know

### Canonical eval setup
- Sequence: chess/seq-01, 29 keyframes (min_disparity=50, full sequence)
- VGGT-SLAM (SPARK): **0.038m** ATE
- Our VGGT-X pipeline (lc condition, submap_size=16): **~1.1m** ATE
- Both use identical `facebook/VGGT-1B` weights
- Same 29 keyframes, same 2 submaps (16+13)

### What has been ruled out
- **Weight differences**: same facebook/VGGT-1B for both ✓
- **Scale estimation method**: tried se3, rotation_only, pairwise_dist — all give ~1.1m, none close to 0.038m
- **Keyframe selection**: same 29 frames confirmed via `--keyframe_list`
- **VGGT-SLAM scale method sensitivity**: full SE3 vs rotation_only on VGGT-SLAM → both 0.038m (chess D << d)

### Root cause hypothesis
The gap is in **how the SL4 pose graph is constructed**, not in scale estimation:

- **VGGT-SLAM** builds its SL4 graph using `proj_mats` (4×4 SL4 camera matrices computed internally from the start of each submap). The inter-submap edge is `inv(proj_mats_prev[-1]) @ proj_mats_curr[0] @ H_scale`.
- **Our pipeline** builds the SL4 graph from VGGT-X's SE3 world-to-cam poses (extrinsics from `pose_encoding_to_extri_intri`). We lift SE3 → SL4 and compute `T = inv(P_prev_ov) @ P_curr_ov` as the inter-submap relative transform.

The SE3→SL4 lift may not preserve the same geometric relationships that VGGT-SLAM's native SL4 pipeline does. The inter-submap relative transform T (from SE3 extrinsics) likely has orientation or translation errors not present in VGGT-SLAM's proj_mats approach.

---

## Next Investigation: VGGT-SPARK as seed model

**Goal**: Use VGGT-SPARK (from `third_party/vggt_spark`) as the feedforward backbone in our pipeline in place of VGGT-X. VGGT-SPARK is the same model VGGT-SLAM uses. If our pipeline gives ~0.038m with VGGT-SPARK, the issue is VGGT-X-specific. If it still gives ~1.1m, the issue is in our pose graph construction regardless of model.

### How to do this

VGGT-SPARK is already installed at `third_party/vggt_spark/`. The runner `evals/runners/run_vggt_slam_lc.py` already shadows the vggt package with VGGT-SPARK via sys.path. What's needed:

1. **Create a `VGGTSPARKCreator`** (or temporarily patch `VGGTXCreator` to use the SPARK forward pass). The SPARK model has the same architecture as VGGT-X but with `compute_similarity=True` support. The key: use `VGGT.from_pretrained("facebook/VGGT-1B")` via the SPARK import, not the VGGT-X import.

2. **Run `eval_gt.py`** with `--backbone vggt_spark --conditions lc --keyframe_list evals/baselines/vggt_slam/chess_seq01/selected_frames.txt --submap_size 16`.

3. **Expected result tree**:
   - If SPARK + our pipeline → ~0.038m → VGGT-X poses are the issue (depth/extrinsic quality from VGGT-X differs from SPARK)
   - If SPARK + our pipeline → ~1.1m → our pose graph construction is the issue regardless of model

### Deeper graph investigation (if SPARK still ~1.1m)

Compare step-by-step what VGGT-SLAM builds vs what our `run_pose_graph_optimization` builds:

1. **Log the actual H matrices** at each submap boundary in both pipelines (scale factor, T, H_w)
2. **Compare `proj_mats`** (VGGT-SLAM) vs `extrinsics` (ours): are the raw pose matrices equivalent?
3. **Instrument `run_pose_graph_optimization`** with `debug_out=[]` and print each intermediate — compare against VGGT-SLAM's `graph.get_homography(...)` values

Key code locations:
- Our pose graph: `collab_splats/pointcloud/loop_closure/closure.py:run_pose_graph_optimization` (lines ~351+)
- VGGT-SLAM graph build: `third_party/VGGT-SLAM/vggt_slam/solver.py:add_edge` (lines ~120+)
- Our submap poses come from: `collab_splats/pointcloud/loop_closure/submap.py`
- VGGT-SLAM submap proj_mats: `third_party/VGGT-SLAM/vggt_slam/submap.py`

---

## File Locations

| What | Where |
|------|-------|
| Canonical VGGT-SLAM baseline | `evals/baselines/vggt_slam/chess_seq01/vggt_slam_fullseq.tum` |
| Selected keyframes | `evals/baselines/vggt_slam/chess_seq01/selected_frames.txt` |
| VGGT-SLAM ATE: 0.038370m | `evals/baselines/vggt_slam/chess_seq01/metrics.json` |
| ATE utility | `evals/ate_utils.py` |
| Eval runner | `evals/eval_gt.py` (has `--keyframe_list`, `--lc_scale_method`) |
| VGGT-SLAM runner | `evals/runners/run_vggt_slam_lc.py` |
| Scale method ablation code | `collab_splats/pointcloud/loop_closure/closure.py` (`scale_method` param in `LoopClosureConfig` + `run_pose_graph_optimization`) |
| Active LC diagnostics spec | `docs/superpowers/specs/2026-05-28-lc-diagnostics-and-model-parity-design.md` |

---

## What NOT to re-investigate

- Scale estimation method: ruled out — not the cause
- Keyframe parity: confirmed ✓
- Weight differences: confirmed identical ✓
- VGGT-SLAM repo correctness: confirmed matches upstream except local dtype fix + scale SE3 ablation flag
