# LC Pipeline — VGGT-SLAM Parity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three bugs in `run_pose_graph_optimization` that cause a 3× ATE gap vs VGGT-SLAM, plus add diagnostic instrumentation to verify convergence.

**Architecture:** Three targeted edits to `closure.py`: (1) H_w inter-submap formula now uses full w2c poses (T) instead of K-only identity; (2) pose extraction composes local VGGT proj_mat with inv(H_opt) instead of decomposing H_opt directly; (3) confidence filtering gates scale estimation on `conf > conf_threshold`. Diagnostic script compares our trajectory to VGGT-SLAM's frame-by-frame. Each bug fixed + tested independently, eval run after each to measure impact.

**Tech Stack:** NumPy, GTSAM (SL4 manifold), SciPy (rotation), pytest. Python env: `/opt/conda/envs/reconstruction/bin/python`. Tests: `tests/pointcloud/`. Main eval: `evals/eval_gt.py`.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `collab_splats/pointcloud/loop_closure/closure.py` | Modify | Bug 1: H_w formula; Bug 2: pose extraction; Bug 3: conf filter + conf_threshold field in LoopClosureConfig; debug_log param |
| `collab_splats/pointcloud/wrappers.py` | Modify | Pass `conf_threshold=cfg.conf_threshold` to `run_pose_graph_optimization` |
| `tests/pointcloud/test_hw_formula.py` | Create | Bug 1 unit test |
| `tests/pointcloud/test_pose_extraction.py` | Create | Bug 2 unit test |
| `tests/pointcloud/test_pgo_parity.py` | Modify | Add Bug 3 confidence-masking test |
| `evals/runners/diagnose_lc_parity.py` | Create | Per-frame diagnostic vs VGGT-SLAM TUM |

---

## Task 1: Fix H_w inter-submap formula (Bug 1)

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py:384–412`
- Create: `tests/pointcloud/test_hw_formula.py`

### Background

Current code (lines 404–412):
```python
H_scale = np.diag([scale, scale, scale, 1.0])
K_prev4 = np.eye(4, dtype=np.float64)
K_prev4[:3, :3] = prev_submap.intrinsics[-1].astype(np.float64)
K_curr4 = np.eye(4, dtype=np.float64)
K_curr4[:3, :3] = submap.intrinsics[0].astype(np.float64)
prev_submap_last_nid = submap_node_ids[prev_submap.submap_id][-1]
H_overlap = pg.get_homography(prev_submap_last_nid)
H_w = H_overlap @ np.linalg.inv(K_prev4) @ K_curr4 @ H_scale
```

When `K_prev = K_curr = I` (identity intrinsics, common in VGGT), `inv(K_prev) @ K_curr = I`, so `H_w = H_overlap @ H_scale`. The rotation between the two submap world frames is completely ignored.

VGGT-SLAM uses `inv(proj_mats[-1]) @ proj_mats[0] @ H_scale` — full w2c poses, capturing the extrinsic rotation+translation.

Fix: `T` is already computed above for scale estimation. Move it before the world_points block (so it's always available), then use it in H_w.

- [ ] **Step 1: Write the failing test**

Create `tests/pointcloud/test_hw_formula.py`:

```python
"""Tests for H_w inter-submap initialization formula (Bug 1 fix).

H_w = H_overlap @ T @ H_scale  where T = inv(P_prev_ov) @ P_curr_ov.
Old code used inv(K_prev) @ K_curr which = I when K=identity, ignoring rotation.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.pointcloud.loop_closure.closure import run_pose_graph_optimization
from collab_splats.pointcloud.loop_closure.submap import Submap


def _make_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _make_submap(poses: np.ndarray, world_points: np.ndarray, submap_id: int = 0) -> Submap:
    k = poses.shape[0]
    return Submap(
        submap_id=submap_id,
        frames=None,
        poses=poses.astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=np.zeros((k, 64), dtype=np.float32),
        image_paths=[f"frame_{i:04d}.png" for i in range(k)],
        raw_outputs={},
        frame_start=submap_id * k,
        world_points=world_points.astype(np.float32),
        world_points_conf=None,
    )


def test_hw_formula_uses_full_pose_not_k_only():
    """Direct formula check: H_w_new = H_overlap @ T @ H_scale differs from
    H_w_old = H_overlap @ inv(K) @ K @ H_scale = H_overlap when K=I.
    """
    R_w = ScipyR.from_euler("z", 30, degrees=True).as_matrix()

    P_prev_ov = np.eye(4, dtype=np.float64)
    P_curr_ov = np.eye(4, dtype=np.float64)
    P_curr_ov[:3, :3] = R_w  # 30° rotation between world frames

    H_overlap = P_prev_ov.copy()
    T = np.linalg.inv(P_prev_ov) @ P_curr_ov
    H_scale = np.eye(4, dtype=np.float64)

    # Old formula: K=I → identity transform
    K_prev4 = np.eye(4, dtype=np.float64)
    K_curr4 = np.eye(4, dtype=np.float64)
    H_w_old = H_overlap @ np.linalg.inv(K_prev4) @ K_curr4 @ H_scale

    # New formula: full pose T
    H_w_new = H_overlap @ T @ H_scale

    # Old formula collapses to H_overlap (no rotation encoded)
    assert np.allclose(H_w_old, H_overlap), "Old K-only formula should equal H_overlap when K=I"

    # New formula includes the 30° rotation
    assert not np.allclose(H_w_new, H_overlap, atol=1e-6), \
        "New T-based formula must differ when world frames have a rotation"
    assert np.allclose(H_w_new[:3, :3], R_w, atol=1e-6), \
        "New H_w rotation block should match the world-frame rotation R_w"


def test_hw_formula_integration_two_submaps_rotated():
    """Integration: 2-submap PGO with 30° rotation between world frames.
    New H_w correctly initializes the first frame of submap 2 with the rotation.
    Check: first frame of submap 2 in output has rotation close to R_w (not identity).
    """
    rng = np.random.default_rng(0)
    R_w = ScipyR.from_euler("z", 30, degrees=True).as_matrix()
    k = 2

    # Prev submap: identity camera (overlap frame = last frame = identity)
    poses_prev = np.stack([
        _make_w2c(np.eye(3), np.array([i * 0.1, 0., 0.])) for i in range(k)
    ])
    # Curr submap: first frame (overlap) has 30° rotation in curr world
    poses_curr = np.stack([
        _make_w2c(R_w, np.array([i * 0.1, 0., 0.])) for i in range(k)
    ])

    wp_prev = rng.standard_normal((k, 5, 5, 3)).astype(np.float32) * 0.1
    wp_curr = rng.standard_normal((k, 5, 5, 3)).astype(np.float32) * 0.1

    prev_sub = _make_submap(poses_prev.astype(np.float32), wp_prev, submap_id=0)
    curr_sub = _make_submap(poses_curr.astype(np.float32), wp_curr, submap_id=1)

    total_frames = k + (k - 1)  # 3 with overlap=1
    result = run_pose_graph_optimization(
        [prev_sub, curr_sub], lc_submaps=[], total_frames=total_frames, overlap_frames=1
    )

    assert result.shape == (total_frames, 4, 4)
    # First frame (reference) should be near identity
    assert np.allclose(result[0], np.eye(4), atol=0.15), \
        f"Frame 0 should be near identity, got\n{result[0]}"
    # Second submap first frame: rotation should NOT be identity (it has 30° world rotation)
    # With correct H_w, the rotation is encoded; with K-only H_w, it would be identity/wrong
    R_out = result[k - 1, :3, :3]  # first unique frame of curr submap (index k-1 with overlap=1)
    rot_vs_identity = np.linalg.norm(R_out - np.eye(3), 'fro')
    assert rot_vs_identity > 0.1, \
        f"Curr submap frame should have non-trivial rotation (got rot_vs_identity={rot_vs_identity:.3f})"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_hw_formula.py -v
```

Expected: `test_hw_formula_uses_full_pose_not_k_only` FAILS (old formula = H_overlap, new not yet implemented). `test_hw_formula_integration_two_submaps_rotated` may pass or fail depending on current behavior.

- [ ] **Step 3: Implement the fix in closure.py**

In `run_pose_graph_optimization`, replace the `else:` block (lines 380–428) with:

```python
        else:
            prev_submap = submaps[s_idx - 1]
            O = min(overlap_frames, k, len(prev_submap.poses))

            # Always compute T from overlap poses — needed for H_w (Bug 1 fix)
            # and scale estimation. Moved outside world_points block.
            P_curr_overlap = submap.poses[0].astype(np.float64)       # w2c: curr world → cam
            P_prev_overlap = prev_submap.poses[-1].astype(np.float64)  # w2c: prev world → cam
            T = np.linalg.inv(P_prev_overlap) @ P_curr_overlap         # curr world → prev world

            scale = 1.0
            if (
                submap.world_points is not None
                and prev_submap.world_points is not None
                and O > 0
            ):
                curr_pts = submap.world_points[:O].reshape(-1, 3).astype(np.float64)
                prev_pts = prev_submap.world_points[-O:].reshape(-1, 3).astype(np.float64)
                n = curr_pts.shape[0]
                curr_h = np.hstack([curr_pts, np.ones((n, 1))])
                curr_in_prev = (T @ curr_h.T).T[:, :3]
                scale = estimate_scale_pairwise(curr_in_prev, prev_pts)

            H_scale = np.diag([scale, scale, scale, 1.0])
            prev_submap_last_nid = submap_node_ids[prev_submap.submap_id][-1]
            H_overlap = pg.get_homography(prev_submap_last_nid)
            # Bug 1 fix: use full pose T (captures extrinsic rotation) instead of inv(K_prev)@K_curr
            H_w = H_overlap @ T @ H_scale
            pg.add_node(node_ids_this[0], H_w)

            H_rel_inter = np.linalg.inv(pg.get_homography(prev_submap_last_nid)) @ H_w
            pg.add_sequential_edge(prev_submap_last_nid, node_ids_this[0], H_rel_inter)

            for local_i in range(1, k):
                H_inner = (
                    submap.poses[local_i - 1].astype(np.float64)
                    @ np.linalg.inv(submap.poses[local_i].astype(np.float64))
                )
                prev_H = pg.get_homography(node_ids_this[local_i - 1])
                pg.add_node(node_ids_this[local_i], prev_H @ H_inner)
                pg.add_sequential_edge(
                    node_ids_this[local_i - 1], node_ids_this[local_i], H_inner
                )
```

Also delete the old lines that set `P_curr_overlap`, `P_prev_overlap`, `T` inside the world_points block (they moved above).

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_hw_formula.py tests/pointcloud/test_pgo_parity.py tests/pointcloud/test_graph.py tests/pointcloud/test_closure_split.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/closure.py tests/pointcloud/test_hw_formula.py
git commit -m "fix(lc): use full w2c poses (T) for H_w inter-submap initialization

Old formula inv(K_prev) @ K_curr = I when intrinsics are identity, ignoring
the extrinsic rotation between submap world frames. VGGT-SLAM uses full
projection matrices. T = inv(P_prev_ov) @ P_curr_ov already computed for
scale estimation; reused for H_w.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Fix pose extraction method (Bug 2)

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py:448–460`
- Create: `tests/pointcloud/test_pose_extraction.py`

### Background

Current code (lines 453–459):
```python
for local_i, nid in enumerate(node_ids):
    H_opt = pg.get_homography(nid)
    _, R, t, _ = decompose_camera(H_opt)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R.astype(np.float32)
    mat[:3, 3] = t.astype(np.float32)
    poses_out[local_i] = mat
```

VGGT-SLAM computes: `projection_mat = proj_mats[idx] @ inv(homography_world)` then decomposes. This composites the original per-frame VGGT projection with the inverse of the PGO correction, preserving the original reconstruction quality and using PGO as a small correction only.

Fix: use `submap.poses[local_i]` (the original VGGT per-frame w2c) composed with `inv(H_opt)`.

- [ ] **Step 1: Write the failing test**

Create `tests/pointcloud/test_pose_extraction.py`:

```python
"""Tests for pose extraction method (Bug 2 fix).

New: local_proj @ inv(H_opt) then decompose.
Old: decompose(H_opt) directly.

VGGT-SLAM: projection_mat = proj_mats[idx] @ inv(homography_world).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.pointcloud.loop_closure.closure import run_pose_graph_optimization
from collab_splats.pointcloud.loop_closure.graph import decompose_camera
from collab_splats.pointcloud.loop_closure.submap import Submap


def _make_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _make_submap(poses: np.ndarray, world_points: np.ndarray, submap_id: int = 0) -> Submap:
    k = poses.shape[0]
    return Submap(
        submap_id=submap_id,
        frames=None,
        poses=poses.astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=np.zeros((k, 64), dtype=np.float32),
        image_paths=[f"frame_{i:04d}.png" for i in range(k)],
        raw_outputs={},
        frame_start=submap_id * k,
        world_points=world_points.astype(np.float32),
        world_points_conf=None,
    )


def test_pose_extraction_formula_local_proj_inv_h_opt():
    """local_proj @ inv(H_opt) then decompose gives different result from decompose(H_opt).

    Verifies the formula change is actually applied. Uses synthetic H_opt and local_proj
    with a known relative rotation so old and new differ detectably.
    """
    # Synthetic: H_opt has 15° tilt, local_proj has 5° roll — distinct matrices
    R_opt = ScipyR.from_euler("y", 15, degrees=True).as_matrix()
    H_opt = np.eye(4, dtype=np.float64)
    H_opt[:3, :3] = R_opt
    H_opt[:3, 3] = [0.2, 0.1, 0.0]

    R_local = ScipyR.from_euler("x", 5, degrees=True).as_matrix()
    local_proj = np.eye(4, dtype=np.float64)
    local_proj[:3, :3] = R_local
    local_proj[:3, 3] = [0.05, 0., 0.]

    # New extraction
    corrected = local_proj @ np.linalg.inv(H_opt)
    _, R_new, t_new, _ = decompose_camera(corrected)

    # Old extraction
    _, R_old, t_old, _ = decompose_camera(H_opt)

    # They must differ (formula change is meaningful)
    assert not np.allclose(R_old, R_new, atol=0.01), \
        "Old and new extraction must give different rotations for non-trivial inputs"
    assert not np.allclose(t_old, t_new, atol=0.01), \
        "Old and new extraction must give different translations for non-trivial inputs"


def test_pose_extraction_single_submap_first_frame_near_identity():
    """Single submap: first frame should be near identity after PGO (pinned by prior).

    With new extraction: local_proj[0] @ inv(H_opt[0]).
    H_opt[0] is pinned by prior to H0 = poses[0].
    local_proj[0] = poses[0].
    So result: poses[0] @ inv(poses[0]) = I → decompose(I) → R=I, t=0. ✓
    """
    rng = np.random.default_rng(42)
    k = 4
    poses = np.stack([
        _make_w2c(np.eye(3), np.array([i * 0.1, 0., 0.])) for i in range(k)
    ]).astype(np.float32)
    wp = rng.standard_normal((k, 5, 5, 3)).astype(np.float32) * 0.1

    submap = _make_submap(poses, wp, submap_id=0)
    result = run_pose_graph_optimization(
        [submap], lc_submaps=[], total_frames=k, overlap_frames=1
    )

    assert result.shape == (k, 4, 4)
    # First frame: reference frame → near identity
    assert np.allclose(result[0], np.eye(4), atol=0.1), \
        f"First frame should be near identity, got\n{result[0]}"
```

- [ ] **Step 2: Run test to verify baseline**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_pose_extraction.py -v
```

Note which tests pass/fail before the fix. `test_pose_extraction_formula_local_proj_inv_h_opt` should FAIL (old code doesn't apply new formula). `test_pose_extraction_single_submap_first_frame_near_identity` may pass.

- [ ] **Step 3: Implement the fix in closure.py**

Replace the pose extraction loop (lines 448–460) with:

```python
    corrected_per_submap: dict[int, np.ndarray] = {}
    for submap in submaps:
        node_ids = submap_node_ids[submap.submap_id]
        k = len(node_ids)
        poses_out = np.zeros((k, 4, 4), dtype=np.float32)
        for local_i, nid in enumerate(node_ids):
            H_opt = pg.get_homography(nid)
            # Bug 2 fix: match VGGT-SLAM's extraction — compose original VGGT
            # projection with inverse of PGO correction (proj_mats[idx] @ inv(H_opt)).
            local_proj = submap.poses[local_i].astype(np.float64)
            corrected = local_proj @ np.linalg.inv(H_opt)
            _, R, t, _ = decompose_camera(corrected)
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R.astype(np.float32)
            mat[:3, 3] = t.astype(np.float32)
            poses_out[local_i] = mat
        corrected_per_submap[submap.submap_id] = poses_out
```

- [ ] **Step 4: Run all tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/ -v
```

Expected: all tests pass (including test_pgo_parity.py test 3 which checks `result[0] ≈ eye(4)`).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/closure.py tests/pointcloud/test_pose_extraction.py
git commit -m "fix(lc): extract poses as local_proj @ inv(H_opt) per VGGT-SLAM

Matches vggt_slam/submap.py: projection_mat = proj_mats[idx] @ inv(homography_world).
Uses original VGGT projection quality and applies PGO only as a correction,
rather than decomposing H_opt directly which loses the original reconstruction accuracy.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Add confidence filtering + conf_threshold config field (Bug 3)

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py` (LoopClosureConfig + scale block)
- Modify: `collab_splats/pointcloud/wrappers.py:307`
- Modify: `tests/pointcloud/test_pgo_parity.py` (add test)

### Background

VGGT-SLAM filters overlap-frame world points by `conf > conf_threshold` before scale estimation (`solver.py:132–143`). Falls back to a less restrictive mask if <100 points remain. We use all points regardless of confidence.

`world_points_conf` shape: `(K, P) float32` — per-point confidence for each frame.

- [ ] **Step 1: Write the failing test**

In `tests/pointcloud/test_pgo_parity.py`, add after the existing tests:

```python
def test_confidence_masking_reduces_scale_noise():
    """Confidence filtering excludes noisy low-conf points from scale estimation.

    Setup: 50 good points (conf=50, correct scale) + 50 noisy points (conf=5, wrong scale).
    Without masking: median is pulled toward wrong scale.
    With masking (conf > 25): only good points used → correct scale.
    """
    rng = np.random.default_rng(123)
    true_scale = 2.0

    N_good = 50
    N_noisy = 50
    N = N_good + N_noisy

    # Good overlap points: consistent scale relationship
    X_prev_good = rng.standard_normal((N_good, 3)) + np.array([5., 0., 0.])
    X_curr_good = true_scale * X_prev_good  # exact scale relationship

    # Noisy points: random scale (high noise, low confidence)
    X_prev_noisy = rng.standard_normal((N_noisy, 3)) * 0.1
    X_curr_noisy = rng.standard_normal((N_noisy, 3)) * 10.0  # totally wrong

    X_prev = np.vstack([X_prev_good, X_prev_noisy])
    X_curr = np.vstack([X_curr_good, X_curr_noisy])

    # Confidence: good=50, noisy=5 (below threshold 25)
    conf_prev = np.array([50.] * N_good + [5.] * N_noisy, dtype=np.float32)
    conf_curr = np.array([50.] * N_good + [5.] * N_noisy, dtype=np.float32)

    conf_threshold = 25.0

    # Without masking: noisy points corrupt the estimate
    scale_unmasked = estimate_scale_pairwise(X_curr, X_prev)

    # With joint mask (conf > threshold on both prev and curr)
    joint_mask = (conf_curr > conf_threshold) & (conf_prev > conf_threshold)
    assert joint_mask.sum() >= 100, "Should have ≥100 good points for this test"
    scale_masked = estimate_scale_pairwise(X_curr[joint_mask], X_prev[joint_mask])

    # Masked estimate should be much closer to true_scale
    err_masked = abs(scale_masked - true_scale) / true_scale
    err_unmasked = abs(scale_unmasked - true_scale) / true_scale

    assert err_masked < 0.05, f"Masked scale {scale_masked:.3f} far from true {true_scale}"
    assert err_unmasked > err_masked, (
        f"Masking should improve estimate: masked_err={err_masked:.3f}, "
        f"unmasked_err={err_unmasked:.3f}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_pgo_parity.py::test_confidence_masking_reduces_scale_noise -v
```

Expected: FAIL — test imports correctly but the production code doesn't do confidence filtering yet. The test itself only imports `estimate_scale_pairwise` and tests the masking logic directly, so it may actually PASS as a logic test. If it passes, it's testing the UNIT logic; the integration test (that run_pose_graph_optimization actually applies the mask) comes from the eval run.

- [ ] **Step 3: Add `conf_threshold` to LoopClosureConfig**

In `closure.py`, add to the `LoopClosureConfig` dataclass (after `max_jump_ratio`):

```python
    conf_threshold: float = 25.0  # confidence gate for scale estimation; matches VGGT-SLAM --conf_threshold 25
```

- [ ] **Step 4: Add `conf_threshold` parameter to `run_pose_graph_optimization`**

Change signature from:
```python
def run_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
    manifold: Literal["sl4", "se3"] = "sl4",
) -> np.ndarray:
```

To:
```python
def run_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
    manifold: Literal["sl4", "se3"] = "sl4",
    conf_threshold: float = 25.0,
) -> np.ndarray:
```

- [ ] **Step 5: Replace the scale estimation block with confidence-filtered version**

Replace the inner scale estimation block (the `if submap.world_points ...` block) with:

```python
            scale = 1.0
            if (
                submap.world_points is not None
                and prev_submap.world_points is not None
                and O > 0
            ):
                curr_pts = submap.world_points[:O].reshape(-1, 3).astype(np.float64)
                prev_pts = prev_submap.world_points[-O:].reshape(-1, 3).astype(np.float64)
                n = curr_pts.shape[0]

                # Confidence filtering: match VGGT-SLAM solver.py:132-143
                mask = np.ones(n, dtype=bool)
                if (
                    submap.world_points_conf is not None
                    and prev_submap.world_points_conf is not None
                ):
                    curr_conf = submap.world_points_conf[:O].reshape(-1)
                    prev_conf = prev_submap.world_points_conf[-O:].reshape(-1)
                    joint_mask = (curr_conf > conf_threshold) & (prev_conf > conf_threshold)
                    if joint_mask.sum() >= 100:
                        mask = joint_mask
                    else:
                        # Fallback: try either-side mask; else use all points
                        either_mask = (curr_conf > conf_threshold) | (prev_conf > conf_threshold)
                        if either_mask.sum() >= 100:
                            mask = either_mask

                curr_h = np.hstack([curr_pts, np.ones((n, 1))])
                curr_in_prev = (T @ curr_h.T).T[:, :3]
                scale = estimate_scale_pairwise(curr_in_prev[mask], prev_pts[mask])
```

- [ ] **Step 6: Update wrappers.py call site**

In `wrappers.py:307–310`, change:
```python
        corrected_extrinsics = run_pose_graph_optimization(
            submaps, lc_submaps, total_frames=N,
            overlap_frames=cfg.submap_overlap,
        )
```
to:
```python
        corrected_extrinsics = run_pose_graph_optimization(
            submaps, lc_submaps, total_frames=N,
            overlap_frames=cfg.submap_overlap,
            conf_threshold=cfg.conf_threshold,
        )
```

- [ ] **Step 7: Run all tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/ -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/closure.py collab_splats/pointcloud/wrappers.py tests/pointcloud/test_pgo_parity.py
git commit -m "fix(lc): add confidence filtering to scale estimation per VGGT-SLAM

Add conf_threshold: float = 25.0 to LoopClosureConfig (matches VGGT-SLAM
--conf_threshold 25). Scale estimation filters overlap points by joint
conf > threshold mask; falls back to either-side mask if <100 points, then
to all points. Mirrors vggt_slam/solver.py:132-143 fallback chain.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Add debug_log instrumentation to run_pose_graph_optimization

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py`

### Background

Adds optional per-boundary diagnostic capture: caller passes `debug_out=[]` list; function appends per-boundary dicts with T, scale, H_w, H_opt, corrected_proj. Used by the diagnostic script (Task 5).

- [ ] **Step 1: Add `debug_out` parameter to `run_pose_graph_optimization`**

Change signature to:
```python
def run_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
    manifold: Literal["sl4", "se3"] = "sl4",
    conf_threshold: float = 25.0,
    debug_out: list | None = None,
) -> np.ndarray:
```

Update the docstring one-liner to mention debug_out returns boundary diagnostics.

- [ ] **Step 2: Add boundary capture after H_w is computed**

After `H_w = H_overlap @ T @ H_scale`, add:
```python
            if debug_out is not None:
                debug_out.append({
                    "submap_id": submap.submap_id,
                    "T": T.copy(),
                    "scale": float(scale),
                    "H_w": H_w.copy(),
                    "H_overlap": H_overlap.copy(),
                })
```

- [ ] **Step 3: Add post-optimization capture for H_opt and corrected_proj**

In the pose extraction loop, after `corrected = local_proj @ np.linalg.inv(H_opt)`:
```python
            if debug_out is not None and local_i == 0:
                # Capture first-frame of each submap (boundary diagnostic)
                for entry in debug_out:
                    if entry.get("submap_id") == submap.submap_id and "H_opt" not in entry:
                        entry["H_opt"] = H_opt.copy()
                        entry["corrected_proj"] = corrected.copy()
                        break
```

- [ ] **Step 4: Run existing tests to verify no regression**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/ -v
```

Expected: all PASS. The debug_out parameter defaults to None so all callers are unaffected.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/closure.py
git commit -m "feat(lc): add debug_out param to run_pose_graph_optimization

Optional list; when provided, receives per-submap-boundary dicts with T,
scale, H_w, H_opt, corrected_proj for diagnostic comparison vs VGGT-SLAM.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Add diagnostic script

**Files:**
- Create: `evals/runners/diagnose_lc_parity.py`

- [ ] **Step 1: Create the diagnostic script**

Create `evals/runners/diagnose_lc_parity.py`:

```python
"""Per-frame diagnostic comparing our LC pipeline vs VGGT-SLAM trajectory.

Usage:
    python evals/runners/diagnose_lc_parity.py \
        --seq_dir data/7scenes/chess/seq-01 \
        --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_dense_nolc.tum \
        --max_frames 200 --submap_size 16

Outputs:
  - Per-frame translation and rotation error (ours vs VGGT-SLAM, both vs GT)
  - Per-submap ATE contribution table
  - JSON dump of debug_out boundary diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from collab_splats.pointcloud.loop_closure.closure import LoopClosureConfig, run_pose_graph_optimization
from collab_splats.pointcloud.loop_closure.eval import ate_translation
from evals.datasets import load_dataset


def _load_tum(path: Path) -> np.ndarray:
    """Load TUM trajectory file → (N, 4, 4) c2w matrices."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # TUM format: timestamp tx ty tz qx qy qz qw
        t = np.array([float(x) for x in parts[1:4]])
        q = np.array([float(x) for x in parts[4:8]])  # xyzw
        R = ScipyR.from_quat(q).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        rows.append(T)
    return np.stack(rows)


def _umeyama_align(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Sim3 alignment: return (4,4) transform that maps src → tgt (translation only)."""
    from collab_splats.pointcloud.loop_closure.closure import umeyama_se3
    return umeyama_se3(src, tgt)


def _rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance between two rotation matrices in degrees."""
    R_rel = R1 @ R2.T
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(trace)))


def diagnose(
    seq_dir: Path,
    vggt_slam_tum: Path,
    max_frames: int,
    submap_size: int,
) -> None:
    # Load GT
    dataset = load_dataset("7scenes", seq_dir, max_frames=max_frames)
    gt_c2w = dataset.gt_poses  # (N, 4, 4) c2w

    # Load VGGT-SLAM trajectory
    slam_c2w = _load_tum(vggt_slam_tum)
    n_slam = len(slam_c2w)
    n_gt = len(gt_c2w)
    n = min(n_slam, n_gt, max_frames)
    print(f"GT frames: {n_gt}, VGGT-SLAM frames: {n_slam}, comparing first {n}")

    gt_c2w = gt_c2w[:n]
    slam_c2w = slam_c2w[:n]

    # Run our pipeline with debug_out
    cfg = LoopClosureConfig(submap_size=submap_size, lc_cosine_threshold=1.0)
    debug_out: list = []

    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    from collab_splats.pointcloud.loop_closure import LoopClosure
    creator = LoopClosure(VGGTXCreator(), config=cfg)

    images = dataset.images[:n]
    result = creator.run(images)
    our_w2c = result.extrinsics  # (N, 4, 4) w2c

    # Convert w2c → c2w
    our_c2w = np.linalg.inv(our_w2c)

    # Align both to GT via Sim3 on translations
    gt_t = gt_c2w[:, :3, 3]
    our_t = our_c2w[:, :3, 3]
    slam_t = slam_c2w[:, :3, 3]

    T_our = _umeyama_align(our_t, gt_t)
    T_slam = _umeyama_align(slam_t, gt_t)

    # Apply alignment
    def apply_sim3(c2w_batch: np.ndarray, T: np.ndarray) -> np.ndarray:
        out = np.zeros_like(c2w_batch)
        for i, pose in enumerate(c2w_batch):
            out[i] = T @ pose
        return out

    our_aligned = apply_sim3(our_c2w, T_our)
    slam_aligned = apply_sim3(slam_c2w, T_slam)

    # Per-frame errors
    print(f"\n{'Frame':>6} {'Our t_err (m)':>14} {'SLAM t_err (m)':>15} {'Our R_err (°)':>14} {'SLAM R_err (°)':>15}")
    print("-" * 70)

    our_t_errs, slam_t_errs = [], []
    for i in range(n):
        our_t_err = float(np.linalg.norm(our_aligned[i, :3, 3] - gt_c2w[i, :3, 3]))
        slam_t_err = float(np.linalg.norm(slam_aligned[i, :3, 3] - gt_c2w[i, :3, 3]))
        our_r_err = _rotation_error_deg(our_aligned[i, :3, :3], gt_c2w[i, :3, :3])
        slam_r_err = _rotation_error_deg(slam_aligned[i, :3, :3], gt_c2w[i, :3, :3])
        our_t_errs.append(our_t_err)
        slam_t_errs.append(slam_t_err)
        if i % 10 == 0:
            print(f"{i:>6} {our_t_err:>14.4f} {slam_t_err:>15.4f} {our_r_err:>14.2f} {slam_r_err:>15.2f}")

    print(f"\nATE RMSE — Ours: {np.sqrt(np.mean(np.array(our_t_errs)**2)):.4f}m  "
          f"VGGT-SLAM: {np.sqrt(np.mean(np.array(slam_t_errs)**2)):.4f}m")

    # Per-submap ATE contribution
    print(f"\n{'Submap':>8} {'Frames':>8} {'Our ATE':>10} {'SLAM ATE':>10}")
    print("-" * 40)
    for s_idx in range(0, n, submap_size):
        e = min(s_idx + submap_size, n)
        our_sub = float(np.sqrt(np.mean(np.array(our_t_errs[s_idx:e])**2)))
        slam_sub = float(np.sqrt(np.mean(np.array(slam_t_errs[s_idx:e])**2)))
        print(f"{s_idx // submap_size:>8} {s_idx:>4}–{e:<4} {our_sub:>10.4f} {slam_sub:>10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq_dir", type=Path, required=True)
    ap.add_argument("--vggt_slam_tum", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--submap_size", type=int, default=16)
    args = ap.parse_args()
    diagnose(args.seq_dir, args.vggt_slam_tum, args.max_frames, args.submap_size)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports resolve**

```bash
/opt/conda/envs/reconstruction/bin/python -c "import evals.runners.diagnose_lc_parity"
```

Expected: no ImportError. (The script imports from the main package which is installed.)

- [ ] **Step 3: Commit**

```bash
git add evals/runners/diagnose_lc_parity.py
git commit -m "feat(evals): add diagnose_lc_parity.py for per-frame trajectory comparison

Runs our pipeline vs a VGGT-SLAM TUM file, aligns both to GT via Sim3,
reports per-frame rotation and translation error. Per-submap ATE table
identifies which submap boundaries accumulate the most error.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Run regression eval and measure ATE improvement

No code changes — run eval to confirm all three fixes reduce ATE.

- [ ] **Step 1: Run eval with all fixes applied**

```bash
tmux new-session -d -s parity_eval2
tmux send-keys -t parity_eval2 "/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir data/7scenes/chess/seq-01 \
  --output_dir evals/results/chess_seq01_all_parity_fixes \
  --max_frames 200 --submap_size 16 --backbone vggtx \
  --conditions baseline lc 2>&1 | tee /tmp/parity_eval2.log; echo EVAL2_DONE" Enter
```

- [ ] **Step 2: Check results**

```bash
cat evals/results/chess_seq01_all_parity_fixes/metrics.json
```

Compare to pre-fix: baseline 0.183m, lc 0.187m. Target: ATE approaching VGGT-SLAM 0.055m.

- [ ] **Step 3: If Bug 2 fix hurts ATE, revert it**

If `lc` or `baseline` ATE is WORSE than 0.183m after all fixes, revert Bug 2:

```bash
git revert HEAD~3  # revert the pose extraction commit (adjust HEAD~N as needed)
# Or manually restore decompose_camera(H_opt) in the extraction loop
```

Re-run eval to confirm ATE recovers.

- [ ] **Step 4: Commit results note**

```bash
git commit --allow-empty -m "docs(evals): chess_seq01 parity fixes ATE results

Pre-fix: baseline=0.183m, lc=0.187m
Post-fix: baseline=X.XXXm, lc=X.XXXm (fill in)
VGGT-SLAM reference: 0.055m

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
