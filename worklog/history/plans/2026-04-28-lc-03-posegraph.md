# Loop Closure Pose-Graph Rewiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire the GTSAM pose graph for mathematical correctness: Umeyama SE(3) alignment, chained submap initial values, inter-submap edges, and fresh LC pair re-run poses.

**Architecture:** Four independent findings (F11, F2, F1, F4) applied in dependency order. F11 adds pure-numpy Umeyama to `alignment.py`. F2+F1 rewire `PoseGraph.add_submaps` to chain initial values and add inter-submap edges using F11's `overlap_region_align`. F4 changes `_verify_loop_candidate` to return `(bool, poses|None)` so VGGTXCreator can supply fresh same-frame poses from the re-run.

**Tech Stack:** numpy, gtsam 4.2 (SE(3) Pose3), Python 3.10. Test runner: `conda run -n nerfstudio python -m pytest`. Worktree: `/workspace/collab-splats/.worktrees/lc-03-posegraph`.

---

## File map

| File | Change |
|------|--------|
| `collab_splats/pointcloud/loop_closure/alignment.py` | Add `umeyama_se3` + `overlap_region_align` (F11) |
| `collab_splats/pointcloud/loop_closure/submap.py` | Add `world_points`, `world_points_conf` fields |
| `collab_splats/pointcloud/loop_closure/pose_graph.py` | Rewire `add_submaps`: chain init (F2) + inter edges (F1); add `_inter_noise` |
| `collab_splats/pointcloud/feedforward.py` | Add `_raw_to_world_points`; populate Submap; change verifier signature (F4); update `_loop_close` call; update `VGGTXCreator._verify_loop_candidate`; update `MapAnythingCreator._verify_loop_candidate` |
| `tests/pointcloud/test_alignment_umeyama.py` | New: Umeyama accuracy tests |
| `tests/pointcloud/test_pose_graph.py` | Extend: inter-submap edge count, chained init with world_points |
| `tests/pointcloud/test_loop_closure_integration.py` | Extend: F4 signature test |

---

## Task 1: F11 — `umeyama_se3` + `overlap_region_align` in `alignment.py`

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/alignment.py`
- Create: `tests/pointcloud/test_alignment_umeyama.py`

### Step 1a — Write failing tests

Write `tests/pointcloud/test_alignment_umeyama.py`:

```python
import numpy as np
import pytest
from collab_splats.pointcloud.loop_closure.alignment import umeyama_se3


def test_umeyama_identity():
    """Identity transform: source == target."""
    np.random.seed(0)
    src = np.random.randn(20, 3).astype(np.float32)
    T = umeyama_se3(src, src)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-5)


def test_umeyama_known_translation():
    """Pure translation: recovers shift within 1e-5."""
    np.random.seed(1)
    src = np.random.randn(50, 3).astype(np.float64)
    shift = np.array([1.0, -2.0, 3.0])
    tgt = src + shift
    T = umeyama_se3(src, tgt)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(T[:3, 3], shift, atol=1e-5)


def test_umeyama_known_rotation_translation():
    """Known SE(3): recover rotation + translation within 1e-4."""
    np.random.seed(2)
    src = np.random.randn(100, 3).astype(np.float64)
    # 90-degree rotation around z
    R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    t_true = np.array([1.0, 2.0, 3.0])
    tgt = (R_true @ src.T).T + t_true
    T = umeyama_se3(src, tgt)
    np.testing.assert_allclose(T[:3, :3], R_true, atol=1e-4)
    np.testing.assert_allclose(T[:3, 3], t_true, atol=1e-4)


def test_umeyama_weighted_ignores_outliers():
    """Weights near zero suppress outlier influence."""
    np.random.seed(3)
    M = 100
    src = np.random.randn(M, 3).astype(np.float64)
    shift = np.array([1.0, 0.0, 0.0])
    tgt = src + shift
    # Corrupt last 20 points heavily; give them near-zero weight
    tgt[-20:] += np.random.randn(20, 3) * 100.0
    weights = np.ones(M, dtype=np.float64)
    weights[-20:] = 1e-6
    T = umeyama_se3(src, tgt, weights=weights)
    np.testing.assert_allclose(T[:3, 3], shift, atol=0.05)


def test_umeyama_no_reflection():
    """Output rotation must have det == 1 (not -1)."""
    np.random.seed(4)
    src = np.random.randn(30, 3).astype(np.float64)
    T = umeyama_se3(src, src * np.array([-1, 1, 1]))
    assert np.linalg.det(T[:3, :3]) > 0
```

- [ ] **Step 1b: Run tests — expect ImportError (function not yet defined)**

```bash
cd /workspace/collab-splats/.worktrees/lc-03-posegraph
conda run -n nerfstudio python -m pytest tests/pointcloud/test_alignment_umeyama.py -v 2>&1 | tail -10
```

Expected: 5 failures with `ImportError` or `cannot import name 'umeyama_se3'`.

### Step 2 — Implement `umeyama_se3` and `overlap_region_align` in `alignment.py`

Append to `collab_splats/pointcloud/loop_closure/alignment.py` (after `dedup_overlap`):

```python

def umeyama_se3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form SE(3) alignment via SVD (no scale).

    Args:
        source: (M, 3) points in source frame.
        target: (M, 3) corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (4, 4) float32 homogeneous T such that target ≈ T @ source.
    """
    M = source.shape[0]
    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        return np.eye(4, dtype=np.float32)
    w = w / w_sum

    src = source.astype(np.float64)
    tgt = target.astype(np.float64)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)

    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    # Weighted cross-covariance H = src_c.T @ diag(w) @ tgt_c
    H = (src_c * w[:, None]).T @ tgt_c  # (3, 3)

    U, _, Vt = np.linalg.svd(H)

    # Handle reflection: ensure det(R) == +1
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T  # (3, 3)
    t = mu_tgt - R @ mu_src  # (3,)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def overlap_region_align(submap_a: "Submap", submap_b: "Submap", overlap_frames: int) -> np.ndarray:
    """SE(3) transform T s.t. pts_a_world ≈ T @ pts_b_local.

    Uses the last `overlap_frames` of submap_a and the first `overlap_frames` of submap_b.
    These represent the same physical frames in two different local coordinate systems.
    Returns identity if either submap has no world_points.

    Args:
        submap_a: reference submap (target frame).
        submap_b: incoming submap (source frame).
        overlap_frames: number of overlap frames to use.

    Returns:
        (4, 4) float32 SE(3) transform mapping submap_b's frame → submap_a's frame.
    """
    from collab_splats.pointcloud.loop_closure.submap import Submap  # noqa: F401 (TYPE_CHECKING)
    if submap_a.world_points is None or submap_b.world_points is None:
        return np.eye(4, dtype=np.float32)

    O = min(overlap_frames, submap_a.world_points.shape[0], submap_b.world_points.shape[0])
    if O == 0:
        return np.eye(4, dtype=np.float32)

    pts_a = submap_a.world_points[-O:].reshape(-1, 3).astype(np.float64)  # (O*P, 3) target
    pts_b = submap_b.world_points[:O].reshape(-1, 3).astype(np.float64)   # (O*P, 3) source

    if submap_a.world_points_conf is not None and submap_b.world_points_conf is not None:
        w = ((submap_a.world_points_conf[-O:] + submap_b.world_points_conf[:O]) / 2).reshape(-1)
    elif submap_a.world_points_conf is not None:
        w = submap_a.world_points_conf[-O:].reshape(-1)
    elif submap_b.world_points_conf is not None:
        w = submap_b.world_points_conf[:O].reshape(-1)
    else:
        w = None

    return umeyama_se3(pts_b, pts_a, weights=w)
```

- [ ] **Step 3: Run Umeyama tests**

```bash
conda run -n nerfstudio python -m pytest tests/pointcloud/test_alignment_umeyama.py -v 2>&1 | tail -10
```

Expected: 5 pass.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/alignment.py \
        tests/pointcloud/test_alignment_umeyama.py
git commit -m "feat(lc): umeyama SE(3) alignment + overlap_region_align (F11)"
```

---

## Task 2: Submap `world_points` fields + feedforward population

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/submap.py`
- Modify: `collab_splats/pointcloud/feedforward.py`

### Step 2a — Add fields to Submap

In `collab_splats/pointcloud/loop_closure/submap.py`, add two optional fields after `raw_outputs`:

```python
    raw_outputs: dict | None = field(default=None, repr=False)  # raw _forward() dict, for merging
    world_points: np.ndarray | None = field(default=None, repr=False)       # (K, P, 3) float32 — per-frame 3D points in local frame
    world_points_conf: np.ndarray | None = field(default=None, repr=False)  # (K, P) float32 — per-point confidence
```

- [ ] **Step 2b: Verify import**

```bash
conda run -n nerfstudio python -c "
from collab_splats.pointcloud.loop_closure import Submap
f = Submap.__dataclass_fields__
assert 'world_points' in f and 'world_points_conf' in f
print('ok')
"
```

Expected: `ok`.

### Step 2c — Add `_raw_to_world_points` helper to `feedforward.py`

Insert this module-level function after the `_enough_frames_for_submaps` method, before `_verify_loop_candidate` (around line 136), as a standalone function:

Find:
```python
    def _verify_loop_candidate(self, frame1: Any, frame2: Any) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError(
```

Insert BEFORE it (as a module-level helper — place it right before the `BaseFeedforwardCreator` class, after all imports, around line 20):

Actually, place it as a module-level function in feedforward.py, after the `FeedforwardResult` dataclass definition. Find the first class definition after imports and add:

```python
def _raw_to_world_points(raw: dict, subsample: int = 8) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract world-space 3D points from a raw _forward output dict.

    Requires 'depth', 'extrinsic', 'intrinsics_downsampled' keys (VGGT-compatible).
    Falls back to (None, None) if keys absent.

    Returns:
        world_points: (K, P, 3) float32 where P = ceil(H/s) * ceil(W/s), or None.
        conf: (K, P) float32, or None if 'depth_conf' absent.
    """
    if not all(k in raw for k in ("depth", "extrinsic", "intrinsics_downsampled")):
        return None, None

    depth = raw["depth"]                   # (K, H, W) float32
    intr = raw["intrinsics_downsampled"]   # (K, 3, 3) float32
    extr_3x4 = raw["extrinsic"]            # (K, 3, 4) float32
    conf_map = raw.get("depth_conf")       # (K, H, W) or None

    K, H, W = depth.shape
    bottom = np.tile([0, 0, 0, 1], (K, 1)).reshape(K, 1, 4).astype(np.float32)
    extr_4x4 = np.concatenate([extr_3x4, bottom], axis=1)                     # (K, 4, 4)
    cam2world = np.linalg.inv(extr_4x4.astype(np.float64)).astype(np.float32)  # (K, 4, 4)

    us = np.arange(0, W, subsample)
    vs = np.arange(0, H, subsample)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()
    P = len(uu)

    all_pts = np.zeros((K, P, 3), dtype=np.float32)
    all_conf = np.zeros((K, P), dtype=np.float32) if conf_map is not None else None

    for ki in range(K):
        z = depth[ki][vv, uu]                                   # (P,)
        fx, fy = intr[ki, 0, 0], intr[ki, 1, 1]
        cx, cy = intr[ki, 0, 2], intr[ki, 1, 2]
        x_c = (uu - cx) * z / fx
        y_c = (vv - cy) * z / fy
        pts_cam = np.stack([x_c, y_c, z, np.ones_like(z)], axis=-1)  # (P, 4)
        all_pts[ki] = (cam2world[ki] @ pts_cam.T).T[:, :3]
        if conf_map is not None and all_conf is not None:
            all_conf[ki] = conf_map[ki][vv, uu]

    return all_pts, all_conf
```

- [ ] **Step 2d: Populate `world_points` when creating Submap in `_run_loop_closure_inference`**

Find (in `_run_loop_closure_inference`):
```python
                submap = Submap(
                    submap_id=wi,
                    frames=frames_cpu,
                    poses=poses_4x4,
                    intrinsics=intrinsics,
                    retrieval_vectors=ret_vecs,
                    image_paths=list(self.image_paths[start:end]),
                    raw_outputs=raw,
                    frame_start=start,
                )
```
Replace with:
```python
                wp, wp_conf = _raw_to_world_points(raw)
                submap = Submap(
                    submap_id=wi,
                    frames=frames_cpu,
                    poses=poses_4x4,
                    intrinsics=intrinsics,
                    retrieval_vectors=ret_vecs,
                    image_paths=list(self.image_paths[start:end]),
                    raw_outputs=raw,
                    frame_start=start,
                    world_points=wp,
                    world_points_conf=wp_conf,
                )
```

- [ ] **Step 2e: Verify baseline tests still pass**

```bash
conda run -n nerfstudio python -m pytest tests/pointcloud/test_loop_closure.py \
  tests/pointcloud/test_loop_closure_integration.py \
  tests/pointcloud/test_alignment_dedup.py \
  tests/pointcloud/test_pose_convention.py -q 2>&1 | tail -5
```

Expected: 21 passed.

- [ ] **Step 2f: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/submap.py \
        collab_splats/pointcloud/feedforward.py
git commit -m "feat(lc): Submap.world_points fields + _raw_to_world_points helper"
```

---

## Task 3: F2 + F1 — Rewire `PoseGraph.add_submaps`

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/pose_graph.py`
- Modify: `collab_splats/pointcloud/feedforward.py` (pass `overlap_frames` to `add_submaps`)
- Modify: `tests/pointcloud/test_pose_graph.py`

### Step 3a — Write failing tests

Append to `tests/pointcloud/test_pose_graph.py`:

```python
def test_inter_submap_edge_count_three_submaps():
    """F1: 3 submaps of k=3 → BetweenFactor count = sum(K_n-1) + (n_submaps-1) = 6+2 = 8."""
    k = 3
    n = 3
    pg = PoseGraph()
    submaps = [_identity_submap(i, k=k) for i in range(n)]
    pg.add_submaps(submaps, overlap_frames=1)
    # 1 PriorFactor + 8 BetweenFactors = 9 total
    expected_between = n * (k - 1) + (n - 1)  # 6 + 2 = 8
    actual_between = pg._graph.size() - 1  # subtract 1 PriorFactor
    assert actual_between == expected_between, \
        f"BetweenFactor count: expected {expected_between}, got {actual_between}"


def test_chained_init_pure_x_translation():
    """F2: pure x-translation between submaps; initial values match expected world-frame poses."""
    import gtsam as _gtsam
    k = 5
    overlap = 4
    P = 20
    shift_x = 2.0
    shift = np.array([shift_x, 0.0, 0.0], dtype=np.float32)

    # Submap A: cameras at (0, 0, i*0.5) in world. W2C t = (0, 0, -i*0.5).
    poses_a = np.tile(np.eye(4), (k, 1, 1)).astype(np.float32)
    for i in range(k):
        poses_a[i, 2, 3] = -i * 0.5

    # A's world_points for overlap region (last `overlap` frames = frames 1..4 in A)
    np.random.seed(42)
    pts_world = np.random.randn(overlap, P, 3).astype(np.float32)
    wp_a = np.zeros((k, P, 3), dtype=np.float32)
    wp_a[-overlap:] = pts_world  # frames 1..4 of A

    # B's local frame = A's world frame shifted +shift_x in x.
    # B's world_points for first `overlap` frames = A's overlap points minus shift.
    pts_b_local = pts_world - shift[None, None, :]
    wp_b = np.zeros((k, P, 3), dtype=np.float32)
    wp_b[:overlap] = pts_b_local  # frames 0..3 of B = A's frames 1..4

    # B's W2C poses in B's local frame:
    # B's frame i corresponds to world frame (i+1), shifted by shift_x in x:
    # cam at (0, 0, (i+1)*0.5) in world = (-shift_x, 0, (i+1)*0.5) in B's local.
    # W2C t = (shift_x, 0, -(i+1)*0.5).
    poses_b = np.tile(np.eye(4), (k, 1, 1)).astype(np.float32)
    for i in range(k):
        poses_b[i, 0, 3] = shift_x
        poses_b[i, 2, 3] = -(i + 1) * 0.5

    s_a = Submap(
        submap_id=0, frames=torch.zeros(k, 3, 4, 4),
        poses=poses_a, intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(k, 16),
        image_paths=[Path(f"a{i}.jpg") for i in range(k)],
        world_points=wp_a,
    )
    s_b = Submap(
        submap_id=1, frames=torch.zeros(k, 3, 4, 4),
        poses=poses_b, intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(k, 16),
        image_paths=[Path(f"b{i}.jpg") for i in range(k)],
        world_points=wp_b,
    )

    pg = PoseGraph()
    pg.add_submaps([s_a, s_b], overlap_frames=overlap)

    # B's frame 0 (global_idx=k=5) should have initial value = A's frame 1 (same physical frame).
    first_b_key = _gtsam.symbol('x', k)
    init_b0 = pg._initial.atPose3(first_b_key).matrix().astype(np.float32)
    expected_b0 = poses_a[1]  # A's frame 1: t = (0, 0, -0.5)
    np.testing.assert_allclose(init_b0, expected_b0, atol=1e-2,
                                err_msg="F2: chained initial value for B's frame 0 should match A's frame 1")


def test_add_submaps_no_world_points_backward_compat():
    """F2 fallback: submaps with world_points=None produce same graph size as before."""
    k = 3
    pg = PoseGraph()
    submaps = [_identity_submap(i, k=k) for i in range(2)]  # world_points=None by default
    pg.add_submaps(submaps, overlap_frames=4)
    result = pg.optimize()
    assert 0 in result and 1 in result
    assert result[0].shape == (k, 4, 4)
    assert result[1].shape == (k, 4, 4)
```

- [ ] **Run failing tests**

```bash
conda run -n nerfstudio python -m pytest \
  tests/pointcloud/test_pose_graph.py::test_inter_submap_edge_count_three_submaps \
  tests/pointcloud/test_pose_graph.py::test_chained_init_pure_x_translation \
  tests/pointcloud/test_pose_graph.py::test_add_submaps_no_world_points_backward_compat \
  -v 2>&1 | tail -15
```

Expected: 3 failures (TypeError: add_submaps() got unexpected kwarg `overlap_frames`, etc.).

### Step 3b — Rewire `PoseGraph.add_submaps`

Replace the existing `add_submaps` method in `collab_splats/pointcloud/loop_closure/pose_graph.py`:

```python
    def add_submaps(self, submaps: list[Submap], overlap_frames: int = 4) -> None:
        """Add submaps: intra-submap edges + inter-submap edges (F1) + chained init values (F2)."""
        from collab_splats.pointcloud.loop_closure.alignment import overlap_region_align

        accumulated_T = np.eye(4, dtype=np.float32)
        accumulated_Ts: list[np.ndarray] = []

        for i, submap in enumerate(submaps):
            if i > 0:
                T_local = overlap_region_align(submaps[i - 1], submap, overlap_frames)
                accumulated_T = accumulated_T @ T_local
            accumulated_Ts.append(accumulated_T.copy())

            start = self._total_frames
            self._frame_offset[submap.submap_id] = start
            self._submaps.append(submap)
            k = submap.poses.shape[0]
            inv_T = np.linalg.inv(accumulated_T).astype(np.float32)  # (4, 4)

            for local_i, pose in enumerate(submap.poses):
                global_idx = start + local_i
                key = gtsam.symbol('x', global_idx)

                world_pose = pose @ inv_T  # F2: express initial value in common world frame
                self._initial.insert(key, _pose3(world_pose))

                if global_idx == 0:
                    self._graph.add(PriorFactorPose3(key, _pose3(world_pose), self._anchor_noise))

                if local_i > 0:
                    prev_world_pose = submap.poses[local_i - 1] @ inv_T
                    prev_key = gtsam.symbol('x', global_idx - 1)
                    rel = _relative_pose3(prev_world_pose, world_pose)
                    self._graph.add(BetweenFactorPose3(prev_key, key, rel, self._intra_noise))

            # F1: inter-submap edge between last frame of previous submap and first frame of this one
            if i > 0:
                last_prev_key = gtsam.symbol('x', start - 1)
                first_curr_key = gtsam.symbol('x', start)
                prev_inv_T = np.linalg.inv(accumulated_Ts[-2]).astype(np.float32)
                last_prev_world = submaps[i - 1].poses[-1] @ prev_inv_T
                first_curr_world = submap.poses[0] @ inv_T
                rel_inter = _relative_pose3(last_prev_world, first_curr_world)
                self._graph.add(BetweenFactorPose3(last_prev_key, first_curr_key, rel_inter, self._inter_noise))

            self._total_frames += k
```

Also add `_inter_noise` to `PoseGraph.__init__` (after `self._anchor_noise = ...`):

```python
        self._inter_noise = _noise(_INTRA_NOISE_SIGMA)  # same sigma for now; tuned in spec 4
```

### Step 3c — Update `_loop_close` call in `feedforward.py`

Find:
```python
        pg.add_submaps(submaps)
```
Replace with:
```python
        pg.add_submaps(submaps, overlap_frames=self.loop_closure_config.submap_overlap)
```

- [ ] **Step 3d: Run F1/F2 tests**

```bash
conda run -n nerfstudio python -m pytest \
  tests/pointcloud/test_pose_graph.py -v 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 3e: Run full regression**

```bash
conda run -n nerfstudio python -m pytest tests/pointcloud/ -q 2>&1 | tail -5
```

Expected: same 4 pre-existing failures, no new failures.

- [ ] **Step 3f: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/pose_graph.py \
        collab_splats/pointcloud/feedforward.py \
        tests/pointcloud/test_pose_graph.py
git commit -m "fix(lc): inter-submap edges F1 + chained init F2

- add_submaps accumulates world transform via overlap_region_align
- inter-submap BetweenFactor between last/first frame of consecutive submaps
- initial values expressed in common world frame"
```

---

## Task 4: F4 — `_verify_loop_candidate` returns `(bool, poses | None)`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py` (base, MapAnythingCreator, VGGTXCreator, call site)
- Modify: `tests/pointcloud/test_loop_closure_integration.py`

### Step 4a — Write failing tests

Append to `tests/pointcloud/test_loop_closure_integration.py`:

```python
def test_verify_loop_candidate_returns_tuple():
    """F4: _verify_loop_candidate must return (bool, ndarray|None), not bare bool."""
    import torch
    from collab_splats.pointcloud.feedforward import MapAnythingCreator, LoopClosureConfig as _LCC
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig

    creator = object.__new__(MapAnythingCreator)
    creator._lc_retrieval = None  # no retrieval model loaded
    creator.loop_closure_config = LoopClosureConfig()

    result = creator._verify_loop_candidate(
        torch.zeros(3, 64, 64), torch.zeros(3, 64, 64)
    )
    # Must be a tuple of exactly 2 elements
    assert isinstance(result, tuple) and len(result) == 2, \
        f"Expected (bool, ndarray|None), got {type(result)}"
    accepted, lc_poses = result
    assert accepted is False
    assert lc_poses is None


def test_base_verify_raises():
    """F4/F5: base _verify_loop_candidate raises NotImplementedError with updated signature."""
    import numpy as np, pytest
    from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator, FeedforwardResult
    from pathlib import Path

    class _D(BaseFeedforwardCreator):
        def _load_model(self, device): pass
        def _preprocess(self, image_dir): return None, [], np.zeros((0, 2))
        def _forward(self, model, views, **kw): return {}
        def _postprocess(self, raw, **kw):
            return FeedforwardResult(np.zeros((1,3)), np.zeros((1,3)),
                                     np.eye(4)[None], np.eye(3)[None], [], 1, 1)

    dummy = object.__new__(_D)
    with pytest.raises(NotImplementedError, match="_verify_loop_candidate"):
        dummy._verify_loop_candidate(None, None)
```

- [ ] **Run failing tests (expect current MapAnythingCreator returns bool, not tuple)**

```bash
conda run -n nerfstudio python -m pytest \
  tests/pointcloud/test_loop_closure_integration.py::test_verify_loop_candidate_returns_tuple \
  tests/pointcloud/test_loop_closure_integration.py::test_base_verify_raises \
  -v 2>&1 | tail -10
```

Expected: failures.

### Step 4b — Update base class in `feedforward.py`

Find:
```python
    def _verify_loop_candidate(self, frame1: Any, frame2: Any) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _verify_loop_candidate. "
            "Loop closure cannot run without per-backend verification."
        )
```

This is already raising — return type annotation just needs updating. Verify it matches `tuple[bool, np.ndarray | None]`. (No code change if annotation already says tuple — check and move on.)

### Step 4c — Update `MapAnythingCreator._verify_loop_candidate`

Find:
```python
    def _verify_loop_candidate(self, frame1: Any, frame2: Any) -> bool:
        """DINO-cosine gate: accept loop if cosine similarity >= verify_match_ratio."""
        import torch
        import torch.nn.functional as F
        retrieval = getattr(self, "_lc_retrieval", None)
        if retrieval is None:
            return False
        frames = torch.stack([frame1, frame2])  # (2, C, H, W)
        with torch.no_grad():
            vecs = retrieval.embed_frames(frames)  # (2, D)
        cos = float(F.cosine_similarity(vecs[0:1], vecs[1:2]).item())
        return cos >= self.loop_closure_config.verify_match_ratio
```

Replace with:
```python
    def _verify_loop_candidate(self, frame1: Any, frame2: Any) -> tuple[bool, np.ndarray | None]:
        """DINO-cosine gate: accept loop if cosine similarity >= verify_match_ratio."""
        import torch
        import torch.nn.functional as F
        retrieval = getattr(self, "_lc_retrieval", None)
        if retrieval is None:
            return False, None
        frames = torch.stack([frame1, frame2])  # (2, C, H, W)
        with torch.no_grad():
            vecs = retrieval.embed_frames(frames)  # (2, D)
        cos = float(F.cosine_similarity(vecs[0:1], vecs[1:2]).item())
        if cos >= self.loop_closure_config.verify_match_ratio:
            return True, None  # MapAnything has no pose_enc output for fresh poses
        return False, None
```

### Step 4d — Update `VGGTXCreator._verify_loop_candidate`

Find (near line 814):
```python
    def _verify_loop_candidate(self, frame1: Any, frame2: Any) -> bool:
        import torch
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        lc_frames = torch.stack([frame1, frame2]).unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            predictions = self.model(lc_frames, compute_similarity=True)
        match_ratio = float(predictions.get("image_match_ratio", 1.0))
        return match_ratio >= self.loop_closure_config.verify_match_ratio
```

Replace with:
```python
    def _verify_loop_candidate(self, frame1: Any, frame2: Any) -> tuple[bool, np.ndarray | None]:
        """Re-run model on (frame1, frame2) pair; return (accepted, fresh_poses_2x4x4)."""
        import torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        lc_frames = torch.stack([frame1, frame2]).unsqueeze(0).to(device, dtype=dtype)  # (1, 2, C, H, W)
        image_shape = lc_frames.shape[-2:]  # (H, W)
        with torch.no_grad():
            predictions = self.model(lc_frames, compute_similarity=True)
        match_ratio = float(predictions.get("image_match_ratio", 1.0))
        if match_ratio < self.loop_closure_config.verify_match_ratio:
            return False, None
        # Extract fresh same-frame poses (mirrors VGGT-SLAM solver.py LC pair re-run pattern)
        ext_3x4, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], image_shape)
        ext_3x4 = ext_3x4.cpu().float().numpy().squeeze(0)  # (2, 3, 4)
        k = ext_3x4.shape[0]
        bottom = np.tile([0, 0, 0, 1], (k, 1)).reshape(k, 1, 4).astype(np.float32)
        lc_poses = np.concatenate([ext_3x4, bottom], axis=1)  # (2, 4, 4)
        return True, lc_poses
```

### Step 4e — Update call site in `_run_loop_closure_inference`

Find:
```python
                    if self._verify_loop_candidate(q_frame, d_frame):
                        verified += 1
                        loops_found += 1
                        console.log(
                            f"  ↩ Loop: submap {match.query_submap_id} → {match.detected_submap_id}"
                            f"  dist={match.similarity_score:.3f}"
                        )
                        lc_submaps.append(Submap(
                            submap_id=len(submaps) + len(lc_submaps),
                            frames=torch.stack([q_frame, d_frame]),
                            poses=np.stack([
                                submap.poses[match.query_frame_idx],
                                d_submap.poses[match.detected_frame_idx],
                            ]),
```

Replace with:
```python
                    accepted, lc_poses = self._verify_loop_candidate(q_frame, d_frame)
                    if accepted:
                        verified += 1
                        loops_found += 1
                        console.log(
                            f"  ↩ Loop: submap {match.query_submap_id} → {match.detected_submap_id}"
                            f"  dist={match.similarity_score:.3f}"
                        )
                        if lc_poses is None:
                            lc_poses = np.stack([
                                submap.poses[match.query_frame_idx],
                                d_submap.poses[match.detected_frame_idx],
                            ])
                        lc_submaps.append(Submap(
                            submap_id=len(submaps) + len(lc_submaps),
                            frames=torch.stack([q_frame, d_frame]),
                            poses=lc_poses,
```

- [ ] **Step 4f: Run F4 tests**

```bash
conda run -n nerfstudio python -m pytest \
  tests/pointcloud/test_loop_closure_integration.py -v 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 4g: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py \
        tests/pointcloud/test_loop_closure_integration.py
git commit -m "fix(lc): _verify_loop_candidate returns (bool, poses|None) (F4)

VGGTXCreator extracts fresh pose_enc from model re-run on LC pair.
MapAnythingCreator returns (bool, None) — no pose_enc output.
Call site uses fresh poses when available; falls back to window poses."
```

---

## Task 5: Full verification gate + squash-merge

- [ ] **Run all LC tests**

```bash
conda run -n nerfstudio python -m pytest \
  tests/pointcloud/test_alignment_umeyama.py \
  tests/pointcloud/test_alignment_dedup.py \
  tests/pointcloud/test_pose_convention.py \
  tests/pointcloud/test_pose_graph.py \
  tests/pointcloud/test_loop_closure.py \
  tests/pointcloud/test_loop_closure_integration.py \
  -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Regression sweep**

```bash
conda run -n nerfstudio python -m pytest tests/pointcloud/ -q 2>&1 | tail -5
```

Expected: same 4 pre-existing failures, no new failures.

- [ ] **Behavior assertions**

```bash
conda run -n nerfstudio python -c "
import numpy as np
from collab_splats.pointcloud.loop_closure.alignment import umeyama_se3

# F11: pure translation recovery
src = np.random.randn(100, 3)
shift = np.array([1.0, -2.0, 3.0])
tgt = src + shift
T = umeyama_se3(src, tgt)
err_R = np.linalg.norm(T[:3, :3] - np.eye(3))
err_t = np.linalg.norm(T[:3, 3] - shift)
assert err_R < 1e-4 and err_t < 1e-4, f'F11 FAIL: R_err={err_R}, t_err={err_t}'
print(f'F11 OK: rotation_err={err_R:.2e}, translation_err={err_t:.2e}')
"
```

- [ ] **Request code review** (invoke `superpowers:requesting-code-review`)

Reviewer must check:
1. F4 VGGTXCreator pose_enc extraction against VGGT-SLAM `solver.py` LC pattern
2. F1 inter-submap edge formula: `rel = inv(last_prev_world) @ first_curr_world`
3. F2 accumulated_T composition: `T_accum_new = T_accum_prev @ overlap_region_align(prev, curr)`

- [ ] **Squash-merge into `refactor/core-modules`**

```bash
cd /workspace/collab-splats
git merge --squash lc-03-posegraph
git commit -m "$(cat <<'EOF'
fix(lc): pose-graph rewiring F1 F2 F4 F11

- umeyama SE(3) alignment + overlap_region_align in alignment.py (F11)
- Submap.world_points + world_points_conf fields; populated from depth
- chained submap initial values via accumulated overlap transforms (F2)
- inter-submap BetweenFactor edges between consecutive submaps (F1)
- _verify_loop_candidate returns (bool, poses|None) (F4)
- VGGTXCreator extracts fresh pose_enc from LC pair re-run
- _raw_to_world_points derives world-space 3D points from depth maps

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Update `worklog/WORKLOG.md`** — record merge SHA, mark spec 3 done, note "Next: spec 4 (lc-04-robustness)".
