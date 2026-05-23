# Loop Closure 3D Geometric Gate — Feasibility Notes

**Date:** 2026-05-06  
**Status:** Deferred — not implemented  
**Branch:** refactor/core-modules  
**Context:** Written for a future agent exploring whether/how to add geometric verification to loop closure.

---

## What was considered

After DINO-SALAD retrieval finds a loop closure candidate (two submaps that likely revisit the same scene), we wanted a 3D geometry check to confirm the match before inserting it into the pose graph. The goal: reject embedding-space false positives that happen to have similar DINO features but different 3D geometry.

---

## The coordinate frame problem

`Submap.world_points` is `(K, P, 3)` — dense depth grids, one point cloud per frame, expressed in **submap-local world space**. Each submap runs VGGT/MapAnything independently. Frame 0 of each submap is the identity pose; all other frames' world points are expressed relative to that local origin.

Key code path — `feedforward.py:_raw_to_world_points`:

```python
all_pts[ki] = (cam2world[ki] @ pts_cam.T).T[:, :3]
```

`cam2world[ki]` here is VGGT's output pose for frame `ki` within the current submap window. It has no relationship to any other submap's coordinate system.

**Consequence:** `world_points` from submap A and `world_points` from submap B live in completely different coordinate systems. A KD-tree overlap check between them will return garbage — the Euclidean distances are meaningless across submap boundaries.

**Why adjacent submaps are the exception:** `overlap_region_align` in `alignment.py` works because adjacent submaps share `submap_overlap=4` physical frames. VGGT independently recovers nearly the same poses for those frames in both submaps' local worlds, so the point clouds nearly coincide for the overlap region. Loop closure candidates are distant submaps with zero shared frames — no such coincidence.

---

## What VGGT-Long does and why it isn't portable

VGGT-Long (`loop_utils/loop_refinement.py`) uses `ransac_umeyama` on **matched point correspondences**. It gets those correspondences by running VGGT **jointly** on `[query_frame, detected_frame]` as a two-frame video. VGGT, being a video model, produces 3D point tracks across frames — so both frames' points land in the same coordinate system by construction.

Our pipeline runs VGGT on submap windows of 20 frames sequentially. To replicate VGGT-Long's approach:

1. At gate time, for each loop candidate `(query_frame, detected_frame)`:  
   - Run a second VGGT forward pass on just those two frames.
   - Extract the 3D tracks → matched point pairs in a shared coordinate system.
   - Run RANSAC + Umeyama on those pairs.
   - Accept/reject based on inlier ratio.

2. Problems:
   - **Latency:** One VGGT forward pass per loop candidate, at gate time, during the main feedforward loop. MapAnything is GPU-heavy; adding N extra forward passes (N = candidate count) multiplies inference time.
   - **Wiring:** The gate currently receives only `(q_frame tensor, d_frame tensor)`. Running VGGT requires the full model object, device management, and preprocessing pipeline — a significant refactor of the gate interface.
   - **Quality:** VGGT is optimized for sequential video. Feeding two temporally unrelated frames as a "video" may produce poor depth estimates. VGGT-Long uses frames that are close in time with optical flow; random loop closure pairs may not have this property.

---

## Alternative approaches that avoid a second forward pass

### Option A: Running chained transform

During `_run_loop_closure_inference`, maintain a running global pose by chaining submap-to-submap transforms:

```
T_global_submapN = T_submapN-1_to_submapN @ T_global_submapN-1
```

The inter-submap transform can be computed from the overlap frames (submap N-1 and submap N share `overlap_frames` physical frames whose world_points nearly coincide — use `umeyama_se3` on those). Then:

```python
world_points_global = (T_global_submapN @ world_points_local.T).T
```

Once all submaps' world_points are in a shared global frame, KD-tree overlap between loop closure candidates is meaningful.

**Complexity:** ~25 lines in `_run_loop_closure_inference`. No new model. The `overlap_region_align` function in `alignment.py` already computes the inter-submap transform — it returns an SE(3) that could be accumulated. The main risk is drift in the chained transform (each inter-submap alignment has error; over many submaps this accumulates). Since we're using this only as a coarse gate (not for precise alignment), drift may be acceptable.

**Files to modify:**
- `collab_splats/pointcloud/feedforward.py` — accumulate `T_chain` per submap iteration, transform `world_points` before gate
- `collab_splats/pointcloud/loop_closure/alignment.py` — verify `overlap_region_align` returns (4,4) SE(3) suitable for chaining (it does: `umeyama_se3` returns `(4, 4) float32`)

### Option B: Depth statistics (camera-space)

Compare per-frame depth histograms in camera space — no coordinate frame issue since each frame's depth is expressed relative to its own camera. Reject candidates whose depth distributions are incompatible (e.g., indoor scene matched with outdoor).

**Weakness:** Only catches extreme mismatches. Two different outdoor scenes with similar depth ranges will not be rejected. Not a reliable geometric gate — more of a sanity filter.

### Option C: Store inter-submap transforms during feedforward

Log each inter-submap SE(3) transform in `self._lc_*` inspection state. A post-hoc tool could reconstruct approximate global world_points from these without re-running inference. This defers the cost but enables offline geometric validation of any candidate pair.

---

## What to look at to implement Option A

```
collab_splats/pointcloud/feedforward.py
  _run_loop_closure_inference()                      — main loop, ~line 196
    for wi, start in enumerate(range(0, N, step)):   — per-submap loop
      overlap_region_align(prev_submap, curr_submap) — returns SE(3) inter-submap transform

collab_splats/pointcloud/loop_closure/alignment.py
  overlap_region_align(submap_a, submap_b, overlap_frames) → (4,4) float32
  umeyama_se3(source, target, weights)               → (4,4) float32

collab_splats/pointcloud/loop_closure/retrieval.py
  LoopMatch.accepted                                 — set by gate loop
  LoopMatchQueue                                     — NMS heap
```

The chain would be built once per submap iteration, stored as a list `global_transforms: list[np.ndarray]`, and used to transform `submap.world_points` before the gate check. The gate check itself is a KD-tree `cKDTree` overlap ratio (fraction of query points with a nearest neighbour in detected-submap cloud within threshold `t`).

Threshold `t` is a new `LoopClosureConfig` field (e.g., `pointcloud_overlap_ratio: float = 0.3`). The existing `verify_match_ratio` (DINO cosine) and `translation_jump_check` remain — the KD-tree gate is an additional filter after those.

---

## Open questions for the implementing agent

1. Does `overlap_region_align` return a transform that maps submap_b → submap_a coordinate system, or the reverse? Verify sign convention before chaining.
2. How fast does chain drift accumulate over 50+ submaps? Run a synthetic test: create submaps with known SE(3) transforms, chain them, measure accumulated error.
3. What is a sensible default for `pointcloud_overlap_ratio`? Needs empirical calibration on a real scene — suggest starting at 0.2 and checking acceptance rate on known-good and known-bad candidates.
4. Should world_points be subsampled before KD-tree (P can be large — check `_raw_to_world_points` for P)? Confidence masking via `world_points_conf` may help focus on high-quality points.
