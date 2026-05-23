# Sim3 Loop Closure Degeneracy Fix Design

**Date:** 2026-05-06  
**Branch:** refactor/core-modules  
**Status:** Design

---

## Context

`run_sim3_pose_graph_optimization` was introduced (commit 477b1a6) to replace GTSAM SE(3) with pypose Sim(3), enabling joint scale+pose correction of loop closure trajectories. The eval notebook shows degenerate results: wrong camera poses and wrong scale. Root cause is two critical bugs in `closure.py`, plus a sub-optimal initialization.

---

## Bug 1 (Critical) — Wrong loop edge relative transform

**File:** `collab_splats/pointcloud/loop_closure/closure.py:223–231`

**Current code:**
```python
T_rel = np.linalg.inv(lc.poses[1]) @ lc.poses[0]
R_rel = T_rel[:3, :3]
t_rel = T_rel[:3, 3]
T_list.append(_sim3_data_from_sRt(1.0, R_rel, t_rel))
ii_list.append(di)
jj_list.append(qi)
```

**Problem:**

Between-factor residual is `(T_qi⁻¹ ⊗ T_di ⊗ T_rel).Log()`. At optimum, `T_qi = T_di ⊗ T_rel`. Nodes `T_qi`, `T_di` are Sim3 transforms from *submap local world* → *global world*. So **T_rel must map FROM qi's local world TO di's local world**.

The chain is:
```
qi_local → query_cam → lc_world → detected_cam → di_local
```

Steps:
1. `qi.poses[k_q]` maps qi_local → query_cam (world-to-cam in qi's local world)
2. `inv(lc.poses[0])` maps query_cam → lc_world
3. `lc.poses[1]` maps lc_world → detected_cam
4. `inv(di.poses[k_d])` maps detected_cam → di_local

**Correct formula:**
```python
k_q = _find_frame_in_submap(submaps[qi], lc.image_paths[0])
k_d = _find_frame_in_submap(submaps[di], lc.image_paths[1])

T_rel = (
    np.linalg.inv(submaps[di].poses[k_d].astype(np.float64))
    @ lc.poses[1].astype(np.float64)
    @ np.linalg.inv(lc.poses[0].astype(np.float64))
    @ submaps[qi].poses[k_q].astype(np.float64)
).astype(np.float32)
```

The current code computes `inv(lc.poses[1]) @ lc.poses[0]` — the *inverse* of detected camera's position in lc frame, missing both `qi.poses[k_q]` and `di.poses[k_d]`. This pushes submap nodes in geometrically nonsensical directions, causing optimizer divergence.

**Helper function to add** (near `_find_submap_node`):
```python
def _find_frame_in_submap(submap: Submap, image_path) -> int | None:
    for k, p in enumerate(submap.image_paths):
        if p == image_path:
            return k
    return None
```

---

## Bug 2 (Critical) — Scale dropped in correction step

**File:** `collab_splats/pointcloud/loop_closure/closure.py:250–270`

**Current code:**
```python
t_opt = optimized[i, :3]
q_opt = optimized[i, 3:7]
# s_opt = optimized[i, 7]  ← NEVER EXTRACTED OR USED
...
T_opt_SE3[:3, :3] = R_opt
T_opt_SE3[:3, 3] = t_opt
T_correction = T_opt_SE3 @ np.linalg.inv(T_init_SE3)
corr_poses[local_k] = T_correction @ T_init_SE3 @ T0_inv @ submap.poses[local_k]
```

**Problem:** Sim3 scale correction is silently discarded. The correction is pure SE3.

**Correct formula** for applying Sim3 `(s_i, R_i, t_i)` to frame k with local extrinsic `[R_c | t_c]`:

```
Camera center in qi_local:   c_local = -R_c.T @ t_c
Camera center in global:     c_global = s_i * R_i @ c_local + t_i
Corrected orientation:       R_corrected = R_c @ R_i.T
Corrected translation:       t_corrected = s_i * t_c - R_c @ R_i.T @ t_i
```

Derivation: `t_corrected = -R_corrected @ c_global = -(R_c @ R_i.T) @ (s_i * R_i @ (-R_c.T @ t_c) + t_i) = s_i * t_c - R_c @ R_i.T @ t_i`

**Replacement for the correction block:**
```python
s_opt = float(optimized[i, 7])
t_opt = optimized[i, :3].astype(np.float64)
q_opt = optimized[i, 3:7].astype(np.float64)
R_opt = ScipyR.from_quat(q_opt).as_matrix()  # (3, 3) float64

K = submap.poses.shape[0]
corr_poses = np.zeros((K, 4, 4), dtype=np.float32)
for local_k in range(K):
    R_c = submap.poses[local_k, :3, :3].astype(np.float64)
    t_c = submap.poses[local_k, :3, 3].astype(np.float64)
    R_corr = R_c @ R_opt.T
    t_corr = s_opt * t_c - R_c @ R_opt.T @ t_opt
    corr_poses[local_k] = np.eye(4, dtype=np.float32)
    corr_poses[local_k, :3, :3] = R_corr.astype(np.float32)
    corr_poses[local_k, :3, 3] = t_corr.astype(np.float32)
corrected[submap.submap_id] = corr_poses
```

This eliminates `T_opt_SE3`, `T_init_SE3`, `T_correction`, and `T0_inv` entirely — they are artifacts of the broken SE3 correction pattern.

---

## Bug 3 (Initialization) — SE3 init mismatched with Sim3 sequential edges

**File:** `collab_splats/pointcloud/loop_closure/closure.py:199–213`

**Current code:** Calls `overlap_region_align` (SE3, scale=1) for initialization, then calls `overlap_region_align_sim3` (Sim3, scale≠1) again for sequential edges. All initial node scales = 1.0 even though sequential edges encode scale≠1. Optimizer starts with large residuals.

**Fix:** Compute sequential Sim3 transforms once, use for both initialization (chained) and edges.

Sim3 composition: `(s1, R1, t1) ⊗ (s2, R2, t2)` = `(s1·s2, R1@R2, t1 + s1·R1@t2)`

```python
# Compute sequential Sim3 transforms once
seq_sRt = []
for i in range(N - 1):
    s, R, t = overlap_region_align_sim3(submaps[i], submaps[i + 1], overlap_frames)
    seq_sRt.append((s, R, t))

# Initialize nodes by chaining Sim3 transforms
initial_data = np.zeros((N, 8), dtype=np.float32)
s_acc, R_acc, t_acc = 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
initial_data[0] = _sim3_data_from_sRt(1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

for i, (s_ij, R_ij, t_ij) in enumerate(seq_sRt):
    # T_{i+1} = T_i ⊗ T_{i,i+1}
    t_acc = t_acc + s_acc * R_acc @ t_ij
    R_acc = R_acc @ R_ij
    s_acc = s_acc * s_ij
    initial_data[i + 1] = _sim3_data_from_sRt(float(s_acc), R_acc.astype(np.float32), t_acc.astype(np.float32))

# Build sequential edges from precomputed transforms
ii_list, jj_list, T_list = [], [], []
for i, (s, R, t) in enumerate(seq_sRt):
    T_list.append(_sim3_data_from_sRt(s, R, t))
    ii_list.append(i)
    jj_list.append(i + 1)
```

---

## Minor Bug — Threshold default mismatch

`LoopClosureConfig.lc_cosine_threshold = 0.75` but `tests/pointcloud/test_loop_closure.py:42` expects `0.85`. Fix: update test to expect `0.75` (the value was intentionally relaxed per lc-02-bugs spec, test not updated).

---

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/pointcloud/loop_closure/closure.py` | Add `_find_frame_in_submap`, fix loop edge T_rel, fix initialization, fix scale correction; remove unused `overlap_region_align` (SE3) import from `run_sim3_pose_graph_optimization` |
| `tests/pointcloud/test_loop_closure.py` | Update threshold assertion (0.85 → 0.75) |
| `tests/pointcloud/test_sim3_pose_graph.py` | Add integration test for `run_sim3_pose_graph_optimization` with loop closures + scale |

---

## New Test: `run_sim3_pose_graph_optimization` with loops + scale

Add to `tests/pointcloud/test_sim3_pose_graph.py`:

```python
def test_run_sim3_with_loop_and_scale_correction():
    """3 submaps, 1 loop edge, scale drift injected → scale + drift corrected."""
    # Build 3 submaps with world_points at known scales
    # Inject scale drift in submap 2 (world_points 2x too large)
    # Add loop edge back to submap 0
    # Assert corrected output positions are within tolerance of ground truth
```

Key invariants to assert:
- Output shape `(total_frames, 4, 4)`, dtype float32
- Camera positions (extracted as `-R.T @ t`) within 0.1m of ground truth after correction
- Scale factor in output extrinsics matches injected scale within 5% (verifies Bug 2 fix)
- Confirmed different from uncorrected baseline (verifies optimizer actually fired)

---

## Verification

1. Run `pytest tests/pointcloud/test_sim3_pose_graph.py -v` — all 6 tests pass + new test passes
2. Run `pytest tests/pointcloud/test_loop_closure.py -v` — threshold test passes after update
3. Run full pointcloud suite: `pytest tests/pointcloud/ --ignore=tests/pointcloud/test_feedforward_lc_state.py -q`
4. Run eval notebook `docs/pointcloud/loop_closure_eval.ipynb` — optimizer cost decreases, camera frustums align visually
