# VGGT-SPARK / VGGT-SLAM Pipeline Parity Investigation Design

**Date:** 2026-05-29
**Branch:** refactor/cu121
**Reference:** `docs/superpowers/plans/2026-05-29-pipeline-parity-handoff.md`

---

## Goal

Identify and fix every structural difference between our LC pipeline and VGGT-SLAM such
that, with `scale=none` on chess_seq01 29-frame keyframe set, our Sim3 ATE approaches
VGGT-SLAM's 0.038m (vs current ~0.612m).

**Out of scope:** Matching VGGT-SLAM's exact world-point source (`unproject_depth_map_to_point_map`
vs our `pts3d` from postprocess). Acceptable residual divergence from this source is expected.

**Success criterion:** `delta_H_overlap_frob` at boundary 0 drops below 0.001 after H1 fix.
Downstream: Sim3 ATE ≤ 0.1m on chess_seq01 29-frame keyframe set.

---

## Hypothesis Disposition Table

| H | Root cause | Mode affected | Status | Action |
|---|---|---|---|---|
| H1 | 16 vs 17-frame VGGT window | **ALL** incl. scale=none | Confirmed from dump data | Fix — extend window to K+1 |
| H2 | cv2.imread vs load_and_preprocess | ALL | Ruled out — both call `load_and_preprocess_images` for inference | None |
| H3 | SLAM T = inv(K)@K (≈I), ours = actual extrinsic | non-none scale only | Confirmed from dump data | Use `rotation_only` mode for SLAM parity; no code change |
| H4 | world_points canonical frame mismatch | non-none scale only | Deferred | Investigate after H1/H3 confirmed |
| H5 | GTSAM noise σ | ALL | Ruled out — both σ=0.05 | None |
| H6 | SL(4) normalization corrupts extraction for submap 1+ | ALL | Suspected — `normalize_to_sl4` changes `H[3,3]` away from 1.0 | Code walk + unit test |

---

## Evidence Summary (from 2026-05-27 dumps)

Dumps at `evals/results/parity_harness/` were produced by `vggt_slam_solver_dump.py` and
`our_solver_dump.py` with `submap_size=16, max_frames=200, scale_method=rotation_only`.

**H1 evidence:**
- `delta_H_overlap_frob` at boundary 0 = **0.0105** — H_overlap diverges even before T is
  involved. H_overlap is set purely by chaining inner-submap edges from the anchor; the only
  source of divergence is that VGGT produced different extrinsics in a 16-frame window vs
  VGGT-SLAM's 17-frame window.
- `delta_H_overlap` grows monotonically (0.010 → 0.016 → 0.091 → ... → 1.52 at boundary 11)
  — error compounds per boundary.

**H3 evidence:**
- SLAM T at boundary 0: `diag([0.99426, 0.99398, 1.0, 1.0])` — pure diagonal, det=0.988,
  not orthogonal (`|T[:3,:3] @ T[:3,:3].T - I|` = 0.017). Confirmed: T = inv(K_prev) @ K_curr.
- Our T at boundary 0: rotation matrix with small off-diagonal values, T-I norm = 0.011.
  Confirmed: actual extrinsic relative pose.
- At boundary 1: SLAM T-I = 0.001 (K virtually constant), our T-I = 0.099. 7× divergence.

**H6 flag:**
- `normalize_to_sl4(H)` = `H / det(H)^0.25`. After normalization, `H[3,3] ≠ 1.0`.
- In the extraction loop, `corrected = inv(H_opt)`. If `H_opt` accumulated SL(4) normalizations
  across submap boundaries, `inv(H_opt)[3,3]` may drift from 1.0.
- `decompose_camera(corrected)` must correctly handle this or extrinsics will be scaled wrong.

---

## Phase 1: No-GPU investigations (H6, confirm H2)

### H6: SL(4) normalization and extraction correctness

**What to check:**

1. Read `decompose_camera` in `graph.py` — does it account for `H[3,3] ≠ 1.0`?
   Specifically: does it divide out the projective scale before extracting R, t?

2. Trace the scale=none path for submap 1, node i:
   ```
   H_opt[0] = H_overlap                           (last node of submap 0)
   H_opt[i] = H_overlap @ inv(poses[i])            (poses[0] ≈ I, so simplifies)
   corrected  = inv(H_opt[i])
              = poses[i] @ inv(H_overlap)
   ```
   After `normalize_to_sl4` is applied at each `add_node` call:
   - `H_overlap` has det=1 (normalized)
   - `H_opt[i]` has det=1
   - `corrected = inv(H_opt[i])` also has det=1
   - `corrected[3,3]` = last element of inv of a det=1 matrix — may not be 1.0

3. Write unit test: build a 2-submap toy graph (scale=none), check that extracted extrinsics
   for submap 1 match expected values within 1e-5.

**Pass criterion:** `decompose_camera` handles projective scale correctly, or we identify and
fix the bug.

### H2: Confirm preprocessing parity

Verified by code inspection: `solver.run_predictions` calls
`load_and_preprocess_images(image_names)` (same function as our `vggtx._preprocess`). The
`cv2.imread` in the dump scripts is only for disparity gating (bypassed at `min_disparity=0`).
No action required.

---

## Phase 2: GPU — H1 fix and re-validation

### H1 fix: extend window to K+1 frames

**File:** `collab_splats/pointcloud/wrappers.py`, sliding window loop (~line 197).

**Current behavior:**
```python
end = min(start + K, N)
window = views[start:end]           # K frames → _forward
```

**Target behavior:**
```python
end = min(start + K + O, N)         # O = overlap_frames (=1 to match VGGT-SLAM)
window = views[start:end]           # K+O frames → _forward
# trim outputs to first K frames
raw = _trim_forward_outputs(raw, K)
```

Where `_trim_forward_outputs(raw, K)` slices per-frame arrays to `[:K]`. Raw can be:
- `dict` (VGGT-X, VGGT-Omega): slice numpy arrays under keys `extrinsic`, `intrinsics`,
  `depth`, `depth_conf`, `world_points`, `world_points_conf`, `images`. Skip non-array keys.
- `list[dict]` (MapAnything): slice the list to `[:K]`.

The extra O frames give VGGT the same attention context as VGGT-SLAM, but only the first K
predictions are stored in the Submap.

**Edge case:** last window — if `end - start ≤ K` (sequence end reached before K+O), skip
trim (window is already ≤ K frames). Only trim when `end - start > K`.

**Verification steps:**
1. Re-run `vggt_slam_solver_dump.py` (fresh VGGT-SLAM baseline).
2. Re-run `our_solver_dump.py` with H1 fix applied.
3. Run `compare_solver_internals.py` → check `delta_H_overlap_frob` at boundary 0.
4. Target: `delta_H_overlap_frob` at boundary 0 < 0.001.

---

## Phase 3: Sim3 ATE re-evaluation

After H1 fix passes the boundary check:

```bash
# 29-frame Sim3 eval (chess_seq01 keyframe subset)
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
  --backbone vggt_spark --conditions lc \
  --submap_size 16 --scale_method none --max_frames 29

# VGGT-SLAM comparable (already have: evals/baselines/vggt_slam/chess_seq01/vggt_slam_nolc.tum)
```

Target ATE table:

| Condition | Sim3 ATE | Notes |
|---|---|---|
| VGGT-SLAM (official) | 0.038m | target |
| Ours scale=none (before H1 fix) | ~0.612m | baseline |
| Ours scale=none (after H1 fix) | **TBD** | goal ≤ 0.1m |

---

## Phase 4: Follow-up hypotheses (if gap remains after H1)

### H3: T computation for non-none scale modes

VGGT-SLAM effectively uses `scale_method=rotation_only` with T≈I (K constant). Our
`rotation_only` mode applies the actual rotation part of T, not K_ratio. For precise parity
in non-none modes:

- Compare scale values at each boundary: `our_internals["scale"]` vs `slam_internals["scale"]`.
- If they diverge by > 0.01, implement a `slam_compat` T option:
  `T = inv(submap.intrinsics[-1]) @ next_submap.intrinsics[0]` (4×4 K-ratio).
- Expected: for chess_seq01 this is near-I, so delta_scale ≈ 0.014 (boundary 0 already shows this).

### H4: world_points canonical frame

VGGT-SLAM calls `tranform_submap_to_canonical` which transforms `world_points[i]` to
cam_0 frame: `world_points[i] = P_first_cam @ world_points[i]`. Our world_points are in
per-camera local frame. This affects `estimate_scale_pairwise(curr_in_prev, prev_pts)` input.

Test: add `transform_to_canonical=True` flag to scale estimation path; run dump; compare
`delta_scale` at each boundary.

---

## Key code locations

| File | Location | What |
|---|---|---|
| `collab_splats/pointcloud/wrappers.py` | line ~197 | Sliding window loop — H1 fix here |
| `collab_splats/pointcloud/loop_closure/closure.py` | line ~553 | Extraction loop — H6 check |
| `collab_splats/pointcloud/loop_closure/graph.py` | `decompose_camera`, `normalize_to_sl4` | H6 check |
| `evals/runners/vggt_slam_solver_dump.py` | `run_dump()` | Re-run for fresh SLAM baseline |
| `evals/runners/our_solver_dump.py` | `run_dump()` | Re-run after H1 fix |
| `evals/runners/compare_solver_internals.py` | `compare()` | Validation |

---

## Python env

```bash
/opt/conda/envs/reconstruction/bin/python
```

Heavy inference runs: tmux only, not notebook.
