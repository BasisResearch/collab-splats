# Spec 3 — Pose-Graph Rewiring

**Branch:** `lc-03-posegraph` (worktree from `refactor/core-modules` post-spec-2 merge)
**Predecessor:** `lc-02-bugs`
**Successor:** `lc-04-robustness`
**Risk:** high (math correctness, behavior diverges)
**Findings addressed:** F1, F2, F4, F11

---

## Goal

Rewire pose graph for mathematical correctness:
- Inter-submap edges connect submap-N's last frame to submap-N+1's first frame (F1)
- Submap initial poses chained into a common world frame via accumulated overlap transforms (F2)
- LC pair re-run pattern: store fresh same-frame poses for the loop edge (F4)
- Overlap-region Umeyama alignment populates `alignment.py` (F11)

After this spec, LM has both meaningful initial values and a connected graph; loop edges are mathematically valid.

---

## Tasks

### F11 — Umeyama alignment in `alignment.py`

**File:** `collab_splats/pointcloud/loop_closure/alignment.py`.

**Add:**
```python
def umeyama_se3(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Closed-form SE(3) alignment via SVD.

    source, target: (M, 3) point arrays in two frames.
    weights: optional (M,) confidence weights.
    Returns: (4, 4) homogeneous transform mapping source → target.
    """
```

Implement as weighted Umeyama:
1. Compute weighted centroids
2. Center both point sets
3. SVD of weighted cross-covariance `H = sum(w_i · src_i^T · tgt_i)`
4. `R = V · diag(1, 1, det(V·U^T)) · U^T` (handle reflection)
5. `t = mean(target) - R · mean(source)`

**Reference:** VGGT-Long `loop_utils/loop_refinement.py::umeyama_alignment` is closed-form Umeyama (no scale, since we're SE(3)). Adapt their version, drop scale factor.

**Companion:**
```python
def overlap_region_align(submap_a: Submap, submap_b: Submap, overlap_frames: int) -> np.ndarray:
    """Compute SE(3) transform mapping submap_b's frame → submap_a's frame.

    Uses overlap-region 3D points (last `overlap_frames` of A, first `overlap_frames` of B)
    and per-point confidence to weight Umeyama.
    """
```

**Acceptance:**
- Unit test: synthetic two point clouds related by known SE(3); Umeyama recovers it within 1e-6.
- Unit test: with weights, low-confidence outliers don't dominate.

### F2 — Chain submap initial values

**File:** `collab_splats/pointcloud/loop_closure/pose_graph.py:add_submaps`.

**Change:** maintain `accumulated_world_transform: np.ndarray = np.eye(4)`. For each submap N:
1. If N == 0: identity (use raw poses).
2. If N > 0: align submap N's frame to submap N-1's frame via `overlap_region_align` (from F11). Compose with `accumulated_world_transform`. Apply the composed transform to all of submap N's poses before inserting into `_initial`.

This requires `Submap` to carry per-frame point clouds + confidence. Extend `Submap` dataclass with `world_points: np.ndarray | None` and `world_points_conf: np.ndarray | None`. Population happens in `feedforward.py:_run_loop_closure_inference` from `raw["world_points"]` and `raw["world_points_conf"]`.

**Acceptance:** synthetic 2-submap test where submap B is a known SE(3) translation of submap A's overlap region; after `add_submaps`, the inserted initial values for submap B match the expected world-frame poses within 1e-3.

### F1 — Inter-submap edges

**File:** `collab_splats/pointcloud/loop_closure/pose_graph.py:add_submaps`.

**Change:** after intra-submap edges of submap N+1 are added, add a `BetweenFactorPose3` edge between:
- `key(submap_N, last_frame_of_N)`
- `key(submap_{N+1}, first_frame_of_{N+1})`

with relative pose computed from the same `overlap_region_align` transform used in F2.

Use `_inter_noise` (new noise model — placeholder for spec 4 to tune; here use same value as `_intra_noise` for now to isolate this spec's diff).

**Acceptance:** after `add_submaps` on a 3-submap input, count `BetweenFactorPose3` edges. Expected = `sum_n(K_n - 1) + (n_submaps - 1)`. Currently = `sum_n(K_n - 1)`.

### F4 — LC pair re-run pattern

**Files:**
- `collab_splats/pointcloud/feedforward.py:VGGTXCreator._verify_loop_candidate` (lines 786-794)
- `collab_splats/pointcloud/feedforward.py:_loop_close` (lines 219-236)

**Change:** when `_verify_loop_candidate` returns true, the verifier ALSO returns the fresh same-frame poses from the re-run (not just a bool). Modify signature:

```python
def _verify_loop_candidate(self, frame1, frame2) -> tuple[bool, np.ndarray | None]:
    """Returns (accepted, lc_poses_2x4x4) or (False, None)."""
```

`VGGTXCreator` extracts `predictions["pose_enc"]` → `extrinsic_lc` (per VGGT-SLAM pattern, Task D verified) → `(2, 4, 4)` array. `MapAnythingCreator` re-runs MapAnything on the pair similarly.

In `_loop_close`, replace the LC-submap construction:
```python
# OLD
poses=np.stack([submap.poses[match.query_frame_idx], d_submap.poses[match.detected_frame_idx]])
# NEW
poses=lc_poses  # the fresh same-frame poses from _verify_loop_candidate
```

**Acceptance:** synthetic test where two submaps observe a shared scene with known relative pose; LC pair re-run recovers that pose; resulting BetweenFactor constraint is consistent with ground truth.

---

## Files touched

- `collab_splats/pointcloud/loop_closure/alignment.py` (Umeyama + overlap_region_align)
- `collab_splats/pointcloud/loop_closure/submap.py` (add world_points, world_points_conf)
- `collab_splats/pointcloud/loop_closure/pose_graph.py` (chain init + inter-submap edges)
- `collab_splats/pointcloud/feedforward.py` (LC pair re-run, populate Submap.world_points)
- `tests/pointcloud/test_alignment_umeyama.py` (new)
- `tests/pointcloud/test_pose_graph.py` (extend — inter-submap edge count, chained init)
- `tests/pointcloud/test_loop_closure_integration.py` (extend — synthetic 3-submap loop test)

---

## Verification gate

1. **Unit tests pass:**
   ```bash
   pytest tests/pointcloud/test_alignment_umeyama.py -v
   pytest tests/pointcloud/test_pose_graph.py -v
   ```

2. **Integration test pass:**
   ```bash
   pytest tests/pointcloud/test_loop_closure_integration.py -v
   ```

3. **Behavior assertions (concrete):**
   - **F11:** Umeyama recovers known SE(3) within `‖ΔR‖ < 1e-6, ‖Δt‖ < 1e-6`.
   - **F2:** synthetic 2-submap, B = SE(3)·A; chained init matches expected world poses within `‖ΔT‖ < 1e-3`.
   - **F1:** edge count = `sum_n(K_n - 1) + (n_submaps - 1)`.
   - **F4:** LC pair re-run recovers ground-truth relative pose within `‖ΔR‖ < 1e-3, ‖Δt‖ < 1e-3`.
   - **End-to-end ATE drop:** synthetic 3-submap closed-loop scene with seeded drift. Run pre-spec-3 vs post-spec-3. Post-spec-3 ATE < pre-spec-3 ATE by ≥30% (record absolute numbers in PR description).

4. **Regression sweep:**
   ```bash
   pytest tests/pointcloud/ -v
   ```

5. **Reviewer signoff** via `superpowers:requesting-code-review`. **Reviewer must verify F4 against VGGT-SLAM upstream `solver.py` LC pattern (Task D in verification doc).**

6. **Squash-merge** into `refactor/core-modules` with commit:
   ```
   fix(lc): pose-graph rewiring F1 F2 F4 F11

   - umeyama SE(3) alignment in alignment.py (F11)
   - chained submap initial values via accumulated overlap transforms (F2)
   - inter-submap BetweenFactor edges (F1)
   - LC pair re-run stores fresh same-frame poses (F4, mirrors VGGT-SLAM solver.py pattern)
   - Submap dataclass grows world_points + world_points_conf
   ```

7. **Update `worklog/WORKLOG.md`** — record merge SHA, ATE numbers, mark spec 3 done.

---

## Out of scope

- Three-way noise split (F6) — spec 4
- Huber kernel — spec 4
- Translation-jump pre-add gate (Item 15) — spec 4
- Loop-edge down-weighting (Item 16) — spec 4
- ADRs — spec 4
