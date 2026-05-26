# LC Pipeline — VGGT-SLAM Parity Fixes Design

## Goal

Close the 3× ATE gap between our LC pipeline (0.183m) and VGGT-SLAM (0.055m) by fixing concrete architectural divergences in `closure.py`. Our pipeline was built to emulate VGGT-SLAM; this spec identifies and fixes the remaining divergences.

## Context

Previous session fixed: submap_overlap default (4→1), loop-edge noise model (Huber→Gaussian σ=0.05), and scale estimation (K-only→full w2c poses for H_scale). Those fixes had zero ATE impact because two deeper bugs remain unfixed.

VGGT-SLAM eval environment: `third_party/VGGT-SLAM`, runs via `evals/runners/run_vggt_slam.py`. Baseline TUM: `evals/baselines/vggt_slam/chess_seq01/vggt_slam_dense_nolc.tum` (212 frames, min_disparity=0, no LC).

## Identified Divergences

### Bug 1 — H_w inter-submap initialization formula (critical)

**Location:** `closure.py:404–412`

**Current (wrong):**
```python
K_prev4 = np.eye(4); K_prev4[:3,:3] = prev_submap.intrinsics[-1]
K_curr4 = np.eye(4); K_curr4[:3,:3] = submap.intrinsics[0]
H_w = H_overlap @ np.linalg.inv(K_prev4) @ K_curr4 @ H_scale
```

When intrinsics are identity (common in practice), `inv(K_prev) @ K_curr = I`, so H_w = H_overlap @ H_scale. This ignores the **rotation between the two submap world frames** entirely.

**VGGT-SLAM (solver.py:145):**
```python
H_rel = inv(proj_mats[-1]) @ proj_mats[0] @ H_scale
H_w = graph.get_homography(prev_last_node) @ H_rel
```

Uses full SL4 projection matrices for both overlap frames, capturing the extrinsic rotation+translation between world frames.

**Fix:** Reuse T already computed for scale estimation:
```python
T = np.linalg.inv(P_prev_overlap) @ P_curr_overlap  # curr world → prev world
H_w = H_overlap @ T @ H_scale
```
Drop the now-unused K_prev4/K_curr4 block.

### Bug 2 — Final pose extraction method (critical)

**Location:** `closure.py:454–460`

**Current (wrong):**
```python
H_opt = pg.get_homography(nid)
_, R, t, _ = decompose_camera(H_opt)
```
Decomposes the optimized SL4 node directly into SE3. This loses accuracy because SL4 normalization and PGO manifold constraints distort the projection matrix.

**VGGT-SLAM (submap.py:115–123):**
```python
projection_mat = proj_mats[idx] @ np.linalg.inv(homography_world)
K, R, t, scale = decompose_camera(projection_mat[0:3,:])
```
Uses the **original per-frame VGGT projection matrix** (pre-optimization, local submap world) composed with the inverse of the PGO correction. This preserves the quality of the original VGGT reconstruction and uses PGO only as a correction.

**Fix:** Store `submap.poses[local_i]` as the local proj_mat; compose with H_opt inverse:
```python
H_opt = pg.get_homography(nid)
local_proj = submap.poses[local_i].astype(np.float64)
corrected = local_proj @ np.linalg.inv(H_opt)
_, R, t, _ = decompose_camera(corrected)
```

### Bug 3 — Missing confidence filtering in scale estimation (moderate)

**Location:** `closure.py:390–402`

**Current:** All overlap world points used regardless of confidence.

**VGGT-SLAM (solver.py:132–143):** Filters by `conf > conf_threshold` before scale estimation. Falls back to prior-only mask if <100 points remain, then falls back to unfiltered if still <100.

**Fix:** Add `conf_threshold: float = 25.0` to `LoopClosureConfig` (matching VGGT-SLAM's `--conf_threshold 25` default). In scale estimation, apply `conf > conf_threshold` mask using `submap.world_points_conf`. Fallback: if <100 points pass the joint (curr AND prev) mask, try prior-only mask; if still <100, use all points.

## Diagnostic Instrumentation

### 2a — `run_pose_graph_optimization` debug logging

Add `debug_log: bool = False` parameter. When enabled, return a second value: a list of per-submap-boundary dicts containing:
- `T`: inv(P_prev) @ P_curr at each boundary
- `scale`: estimated scale
- `H_w`: initialized homography for first frame of each submap
- `H_opt`: post-PGO homography for same frame
- `corrected_proj`: local_proj @ inv(H_opt) before decompose

Return type becomes `tuple[np.ndarray, list[dict] | None]`.

### 2b — `evals/runners/diagnose_lc_parity.py`

CLI script for systematic comparison:

```
python evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_dense_nolc.tum \
  --max_frames 200 --submap_size 16
```

Outputs:
1. Per-frame rotation error (°) and translation error (m): our pipeline vs VGGT-SLAM, both aligned to GT via Sim3
2. Per-submap ATE contribution table — identifies which submap boundaries accumulate the most error
3. Per-boundary scale estimate diff (our vs printed from VGGT-SLAM logs if available)

Key diagnostic question answered: **does error spike at submap boundaries or accumulate within-submap?** — boundary spikes = H_w bug; within-submap = pose extraction bug.

## Testing

### New tests

**`tests/pointcloud/test_hw_formula.py`**
- Construct 2-submap scenario with a known 30° rotation between world frames
- Show old K-only formula (inv(K)@K = I) produces H_w with wrong rotation → high frame-1 rotation error
- Show new T-based formula produces H_w with correct rotation → low frame-1 rotation error

**`tests/pointcloud/test_pose_extraction.py`**
- Build minimal 2-submap scenario with known ground-truth poses
- Run PGO (sequential edges only, no LC)
- Assert `local_proj @ inv(H_opt)` → decompose recovers original VGGT poses more accurately than `decompose(H_opt)` directly

**`tests/pointcloud/test_pgo_parity.py`** (existing, extend)
- Add `test_confidence_masking_reduces_scale_noise`: synthetic overlap with high-conf and low-conf (noisy) points; assert masked scale estimate closer to ground truth

### Regression eval

After fixes: run `evals/eval_gt.py --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 --output_dir evals/results/chess_seq01_hw_fix --max_frames 200 --submap_size 16 --backbone vggtx --conditions baseline lc`.

Target: ATE RMSE significantly below 0.183m (pre-fix baseline), approaching VGGT-SLAM's 0.055m.

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/pointcloud/loop_closure/closure.py` | Fix H_w formula, fix pose extraction, add confidence filtering, add debug_log param |
| `collab_splats/pointcloud/loop_closure/closure.py` (`LoopClosureConfig`) | Add `conf_threshold: float = 25.0` field |
| `collab_splats/pointcloud/loop_closure/submap.py` | No change needed — `world_points_conf` field already exists |
| `evals/runners/diagnose_lc_parity.py` | New diagnostic CLI |
| `tests/pointcloud/test_hw_formula.py` | New test |
| `tests/pointcloud/test_pose_extraction.py` | New test |
| `tests/pointcloud/test_pgo_parity.py` | Add confidence masking test |

## Implementation Order

1. Fix Bug 1 (H_w) + test → run eval to measure impact
2. Fix Bug 2 (pose extraction) + test → run eval to measure impact
3. Fix Bug 3 (confidence filtering) + test → run eval
4. Add diagnostic script → run to confirm no remaining boundary divergence
