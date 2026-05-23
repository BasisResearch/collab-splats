# Loop Closure Sim(3) Pose Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GTSAM SE(3) loop closure optimizer with pypose Sim(3) to fix inter-submap scale drift that causes accordion-fold pointclouds.

**Architecture:** One Sim(3) node per submap; sequential edges encode scale-aware inter-submap alignment via Umeyama-with-scale on overlap world_points; loop closure edges use the relative SE3 from the existing VGGT 2-frame joint forward (s=1.0). pypose LM optimizes all nodes jointly. GTSAM PoseGraph kept untouched as upgrade path to SL(4).

**Tech Stack:** pypose 0.9.5, scipy.spatial.transform.Rotation, numpy, existing collab-splats loop closure infrastructure.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `collab_splats/pointcloud/loop_closure/alignment.py` | Modify | Add `umeyama_sim3`, `overlap_region_align_sim3` |
| `collab_splats/pointcloud/loop_closure/pose_graph.py` | Modify | Add `Sim3PoseGraph(nn.Module)` |
| `collab_splats/pointcloud/loop_closure/closure.py` | Modify | Add `run_sim3_pose_graph_optimization` + private helpers |
| `collab_splats/pointcloud/loop_closure/__init__.py` | Modify | Export new public symbols |
| `collab_splats/pointcloud/feedforward.py` | Modify | Swap `run_pose_graph_optimization` → `run_sim3_pose_graph_optimization` (1 line) |
| `setup.sh` | Modify | Add `pip install pypose` |
| `tests/pointcloud/test_alignment_umeyama.py` | Modify | Add Sim3 Umeyama tests |
| `tests/pointcloud/test_sim3_pose_graph.py` | Create | Sim3PoseGraph + optimization tests |

---

### Task 1: Install pypose

**Files:**
- Modify: `setup.sh`

- [ ] **Step 1: Add pypose to setup.sh**

Find the block in `setup.sh` that runs pip installs inside the nerfstudio conda env. Add after existing installs:

```bash
pip install pypose
```

- [ ] **Step 2: Install pypose in current env**

```bash
/opt/conda/envs/nerfstudio/bin/pip install pypose
```

Expected: `Successfully installed pypose-0.9.5`

- [ ] **Step 3: Smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import pypose as pp; import torch; p = pp.Sim3(torch.zeros(8)); print('pypose ok', p)"
```

Expected: prints `pypose ok` with Sim3 tensor.

- [ ] **Step 4: Commit**

```bash
git add setup.sh
git commit -m "build: add pypose for Sim(3) pose graph"
```

---

### Task 2: `umeyama_sim3` in alignment.py

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/alignment.py`
- Modify: `tests/pointcloud/test_alignment_umeyama.py`

- [ ] **Step 1: Write failing tests**

Open `tests/pointcloud/test_alignment_umeyama.py` and append:

```python
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR
from collab_splats.pointcloud.loop_closure.alignment import umeyama_sim3


def test_umeyama_sim3_known_scale():
    rng = np.random.default_rng(0)
    source = rng.standard_normal((50, 3)).astype(np.float32)
    s_gt, R_gt = 2.5, ScipyR.from_euler("z", 30, degrees=True).as_matrix().astype(np.float32)
    t_gt = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    target = (s_gt * (R_gt @ source.T).T + t_gt)

    s, R, t = umeyama_sim3(source, target)

    assert abs(s - s_gt) < 0.01, f"scale off: {s} vs {s_gt}"
    assert np.allclose(R, R_gt, atol=0.01), f"R off: {R}"
    assert np.allclose(t, t_gt, atol=0.05), f"t off: {t}"


def test_umeyama_sim3_unit_scale():
    rng = np.random.default_rng(1)
    source = rng.standard_normal((30, 3)).astype(np.float32)
    target = source + np.array([0.5, 0.0, -1.0])

    s, R, t = umeyama_sim3(source, target)

    assert abs(s - 1.0) < 0.05
    assert np.allclose(R, np.eye(3), atol=0.05)


def test_umeyama_sim3_weighted():
    rng = np.random.default_rng(2)
    source = rng.standard_normal((100, 3)).astype(np.float32)
    s_gt, t_gt = 1.8, np.array([2.0, 0.0, 0.0], dtype=np.float32)
    target = s_gt * source + t_gt
    weights = np.ones(100, dtype=np.float32)

    s, R, t = umeyama_sim3(source, target, weights=weights)
    assert abs(s - s_gt) < 0.05


def test_umeyama_sim3_insufficient_points_returns_unit():
    source = np.zeros((0, 3), dtype=np.float32)
    target = np.zeros((0, 3), dtype=np.float32)
    s, R, t = umeyama_sim3(source, target)
    assert s == 1.0
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, np.zeros(3))
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_alignment_umeyama.py::test_umeyama_sim3_known_scale -v
```

Expected: `ImportError` or `AttributeError` — `umeyama_sim3` not defined yet.

- [ ] **Step 3: Implement `umeyama_sim3` in alignment.py**

Add after `umeyama_se3` in `collab_splats/pointcloud/loop_closure/alignment.py`:

```python
def umeyama_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3) alignment via Umeyama (with scale).

    Args:
        source: (M, 3) float32/64 points in source frame.
        target: (M, 3) float32/64 corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (s, R, t): float scale, (3,3) float32 rotation, (3,) float32 translation
                   such that target ≈ s * R @ source + t.
    """
    M = source.shape[0]
    if M < 3:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    w = w / w_sum

    src = source.astype(np.float64)
    tgt = target.astype(np.float64)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)

    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    # RMS norms for scale estimation
    scale_src = float(np.sqrt((w * (src_c ** 2).sum(axis=1)).sum()))
    scale_tgt = float(np.sqrt((w * (tgt_c ** 2).sum(axis=1)).sum()))
    s = scale_tgt / scale_src if scale_src > 1e-9 else 1.0

    # Rotation via SVD of cross-covariance
    H = (src_c * s * w[:, None]).T @ tgt_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)

    t = (mu_tgt - s * R @ mu_src).astype(np.float32)
    return s, R, t
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_alignment_umeyama.py -k "sim3" -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/alignment.py tests/pointcloud/test_alignment_umeyama.py
git commit -m "feat(alignment): add umeyama_sim3 with scale estimation"
```

---

### Task 3: `overlap_region_align_sim3` in alignment.py

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/alignment.py`
- Modify: `tests/pointcloud/test_alignment_umeyama.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_alignment_umeyama.py`:

```python
import torch
from dataclasses import dataclass, field
from pathlib import Path
from collab_splats.pointcloud.loop_closure.alignment import overlap_region_align_sim3


def _make_submap(world_points, world_points_conf=None):
    """Minimal Submap stub for alignment tests."""
    from collab_splats.pointcloud.loop_closure.submap import Submap
    K = world_points.shape[0] if world_points is not None else 2
    return Submap(
        submap_id=0,
        frames=torch.zeros(K, 3, 4, 4),
        poses=np.eye(4, dtype=np.float32)[None].repeat(K, axis=0),
        intrinsics=np.eye(3, dtype=np.float32)[None].repeat(K, axis=0),
        retrieval_vectors=torch.zeros(K, 128),
        image_paths=[Path(f"f{i}.jpg") for i in range(K)],
        world_points=world_points,
        world_points_conf=world_points_conf,
    )


def test_overlap_region_align_sim3_no_world_points_returns_unit():
    a = _make_submap(None)
    b = _make_submap(None)
    s, R, t = overlap_region_align_sim3(a, b, overlap_frames=2)
    assert s == 1.0
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, np.zeros(3))


def test_overlap_region_align_sim3_known_scale():
    rng = np.random.default_rng(42)
    # submap_a has 4 frames, submap_b has 4 frames, overlap=2
    # last 2 frames of a and first 2 of b share same physical points at 2x scale
    pts_shared = rng.standard_normal((2, 10, 3)).astype(np.float32)
    pts_b_scaled = pts_shared * 2.0  # s=2.0

    wp_a = np.concatenate([rng.standard_normal((2, 10, 3)).astype(np.float32), pts_shared], axis=0)
    wp_b = np.concatenate([pts_b_scaled, rng.standard_normal((2, 10, 3)).astype(np.float32)], axis=0)

    a = _make_submap(wp_a)
    b = _make_submap(wp_b)

    s, R, t = overlap_region_align_sim3(a, b, overlap_frames=2)
    # s should be ~0.5 (maps b's 2x points back to a's 1x)
    assert 0.3 < s < 0.7, f"Expected s≈0.5, got {s}"
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_alignment_umeyama.py::test_overlap_region_align_sim3_no_world_points_returns_unit -v
```

Expected: `ImportError` — `overlap_region_align_sim3` not defined.

- [ ] **Step 3: Implement `overlap_region_align_sim3` in alignment.py**

Add after `overlap_region_align` in `collab_splats/pointcloud/loop_closure/alignment.py`:

```python
def overlap_region_align_sim3(submap_a, submap_b, overlap_frames: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) transform (s, R, t) s.t. pts_a_world ≈ s * R @ pts_b_local + t.

    Uses last `overlap_frames` of submap_a and first `overlap_frames` of submap_b.
    Returns (1.0, eye(3), zeros(3)) if either submap has no world_points.
    """
    if submap_a.world_points is None or submap_b.world_points is None:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    O = min(overlap_frames, submap_a.world_points.shape[0], submap_b.world_points.shape[0])
    if O == 0:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    pts_a = submap_a.world_points[-O:].reshape(-1, 3).astype(np.float64)
    pts_b = submap_b.world_points[:O].reshape(-1, 3).astype(np.float64)

    if submap_a.world_points_conf is not None and submap_b.world_points_conf is not None:
        w = ((submap_a.world_points_conf[-O:] + submap_b.world_points_conf[:O]) / 2).reshape(-1)
    elif submap_a.world_points_conf is not None:
        w = submap_a.world_points_conf[-O:].reshape(-1)
    elif submap_b.world_points_conf is not None:
        w = submap_b.world_points_conf[:O].reshape(-1)
    else:
        w = None

    return umeyama_sim3(pts_b, pts_a, weights=w)
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_alignment_umeyama.py -k "sim3" -v
```

Expected: all sim3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/alignment.py tests/pointcloud/test_alignment_umeyama.py
git commit -m "feat(alignment): add overlap_region_align_sim3"
```

---

### Task 4: `Sim3PoseGraph` in pose_graph.py

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/pose_graph.py`
- Create: `tests/pointcloud/test_sim3_pose_graph.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pointcloud/test_sim3_pose_graph.py`:

```python
"""Tests for Sim3PoseGraph — pypose Sim(3) pose graph optimizer."""
import numpy as np
import pytest
import torch
import pypose as pp
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.pointcloud.loop_closure.pose_graph import Sim3PoseGraph


def _sim3_data(t=(0, 0, 0), euler_deg=(0, 0, 0), s=1.0) -> np.ndarray:
    """Build (8,) pp.Sim3 data [t(3), q(4,xyzw), s(1)]."""
    q = ScipyR.from_euler("xyz", euler_deg, degrees=True).as_quat()
    return np.array([*t, *q, s], dtype=np.float32)


def test_sim3_pose_graph_forward_residual_zero_at_truth():
    """Between-factor residual is zero when optimizer is at ground truth."""
    # 3 nodes: identity chain
    poses_data = np.stack([
        _sim3_data((0, 0, 0)),   # node 0
        _sim3_data((1, 0, 0)),   # node 1 — 1m in x
        _sim3_data((2, 0, 0)),   # node 2 — 2m in x
    ])
    model = Sim3PoseGraph(poses_data)

    # Edge 0→1: relative T = translate 1m in x, s=1.0
    ii = torch.tensor([0], dtype=torch.long)
    jj = torch.tensor([1], dtype=torch.long)
    T_ij_data = torch.from_numpy(_sim3_data((1, 0, 0))[None])  # (1, 8)
    T_ij = pp.Sim3(T_ij_data)

    residual = model(ii, jj, T_ij)
    assert residual.shape == (1, 7)
    assert residual.abs().max().item() < 1e-4


def test_sim3_pose_graph_optimizes_simple_chain():
    """LM optimizer reduces residual for a 3-node chain with known transforms."""
    import pypose.optim as ppopt

    # True poses: 0=(0,0,0), 1=(1,0,0), 2=(2,0,0)
    true_data = np.stack([
        _sim3_data((0, 0, 0)),
        _sim3_data((1, 0, 0)),
        _sim3_data((2, 0, 0)),
    ])
    # Perturb node 1 position
    perturbed = true_data.copy()
    perturbed[1, 0] += 0.5  # x offset

    model = Sim3PoseGraph(perturbed)

    ii = torch.tensor([0, 1], dtype=torch.long)
    jj = torch.tensor([1, 2], dtype=torch.long)
    T_ij_data = torch.from_numpy(np.stack([
        _sim3_data((1, 0, 0)),  # 0→1
        _sim3_data((1, 0, 0)),  # 1→2
    ]))
    T_ij = pp.Sim3(T_ij_data)

    inp = {"ii": ii, "jj": jj, "T_ij": T_ij}
    residual_before = model(**inp).norm().item()

    optimizer = ppopt.LM(model)
    for _ in range(10):
        optimizer.step(inp)

    residual_after = model(**inp).norm().item()
    assert residual_after < residual_before * 0.1, (
        f"Residual did not shrink: {residual_before:.4f} → {residual_after:.4f}"
    )


def test_sim3_pose_graph_optimized_poses_shape():
    data = np.stack([_sim3_data() for _ in range(5)])
    model = Sim3PoseGraph(data)
    out = model.optimized_poses()
    assert out.shape == (5, 8)
    assert out.dtype == np.float32
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_sim3_pose_graph.py::test_sim3_pose_graph_forward_residual_zero_at_truth -v
```

Expected: `ImportError` — `Sim3PoseGraph` not defined.

- [ ] **Step 3: Implement `Sim3PoseGraph` in pose_graph.py**

Add at the bottom of `collab_splats/pointcloud/loop_closure/pose_graph.py`, after the existing `PoseGraph` class:

```python
# ---------------------------------------------------------------------------
# Sim(3) pose graph — pypose-based, replaces GTSAM SE(3) for LC optimization.
# PoseGraph (GTSAM) kept above as upgrade path to SL(4).
# ---------------------------------------------------------------------------


class Sim3PoseGraph(torch.nn.Module):
    """Pose graph on Sim(3) manifold optimized via pypose LM.

    One node per submap. Sequential and loop edges are BetweenFactor residuals:
    log(T_j^{-1} ⊗ T_i ⊗ T_ij) → 0 at optimum.

    Node format: pp.Sim3 data layout = [t(3), q(4, xyzw), s(1)], shape (8,).
    """

    def __init__(self, initial_poses: np.ndarray) -> None:
        """
        Args:
            initial_poses: (N, 8) float32 in pp.Sim3 data format.
        """
        import torch
        import pypose as pp
        super().__init__()
        self.poses = pp.Parameter(
            pp.Sim3(torch.from_numpy(initial_poses).float()),
        )

    def forward(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        T_ij: "pp.Sim3",
    ) -> torch.Tensor:
        """Compute between-factor residuals.

        Args:
            ii: (E,) long — source node indices.
            jj: (E,) long — target node indices.
            T_ij: (E,) pp.Sim3 — relative transforms.

        Returns:
            (E, 7) residual tensor; zero at optimum.
        """
        return (self.poses[jj].Inv() @ self.poses[ii] @ T_ij).Log()

    def optimized_poses(self) -> np.ndarray:
        """Returns (N, 8) float32 optimized Sim3 data after optimization."""
        return self.poses.data.detach().cpu().numpy()
```

Also add at the top of `pose_graph.py`, after the existing imports:

```python
import torch
```

(If not already present.)

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_sim3_pose_graph.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/pose_graph.py tests/pointcloud/test_sim3_pose_graph.py
git commit -m "feat(pose_graph): add Sim3PoseGraph via pypose"
```

---

### Task 5: `run_sim3_pose_graph_optimization` in closure.py

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py`
- Modify: `tests/pointcloud/test_sim3_pose_graph.py`

- [ ] **Step 1: Write failing test**

Append to `tests/pointcloud/test_sim3_pose_graph.py`:

```python
import torch
from pathlib import Path
from collab_splats.pointcloud.loop_closure.closure import run_sim3_pose_graph_optimization
from collab_splats.pointcloud.loop_closure.submap import Submap


def _make_submap_lc(submap_id, poses, image_paths, world_points=None):
    K = poses.shape[0]
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(K, 3, 4, 4),
        poses=poses,
        intrinsics=np.eye(3, dtype=np.float32)[None].repeat(K, axis=0),
        retrieval_vectors=torch.zeros(K, 128),
        image_paths=image_paths,
        world_points=world_points,
        is_lc_submap=False,
        frame_start=0,
    )


def test_run_sim3_pose_graph_optimization_no_loops():
    """3 submaps, no loops, 4 frames each → returns (12, 4, 4) array."""
    submaps = []
    for i in range(3):
        poses = np.eye(4, dtype=np.float32)[None].repeat(4, axis=0)
        paths = [Path(f"s{i}f{k}.jpg") for k in range(4)]
        s = _make_submap_lc(i, poses, paths)
        s.frame_start = i * 4
        submaps.append(s)

    result = run_sim3_pose_graph_optimization(
        submaps=submaps, lc_submaps=[], total_frames=12, overlap_frames=2
    )
    assert result.shape == (12, 4, 4), result.shape
    assert result.dtype == np.float32


def test_run_sim3_pose_graph_optimization_single_submap():
    """Single submap → returns frames unchanged (no optimization needed)."""
    poses = np.eye(4, dtype=np.float32)[None].repeat(3, axis=0)
    s = _make_submap_lc(0, poses, [Path(f"f{k}.jpg") for k in range(3)])
    s.frame_start = 0
    result = run_sim3_pose_graph_optimization([s], [], total_frames=3, overlap_frames=2)
    assert result.shape == (3, 4, 4)
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_sim3_pose_graph.py::test_run_sim3_pose_graph_optimization_no_loops -v
```

Expected: `ImportError` — `run_sim3_pose_graph_optimization` not defined.

- [ ] **Step 3: Implement helpers + function in closure.py**

Add to `collab_splats/pointcloud/loop_closure/closure.py` (after `run_pose_graph_optimization`):

```python
def _se3_to_sim3_data(mat44: np.ndarray) -> np.ndarray:
    """(4, 4) world-to-cam SE3 → (8,) pp.Sim3 data [t(3), q(4,xyzw), s=1.0]."""
    from scipy.spatial.transform import Rotation as ScipyR
    R = mat44[:3, :3].astype(np.float64)
    t = mat44[:3, 3].astype(np.float32)
    q = ScipyR.from_matrix(R).as_quat().astype(np.float32)  # [x,y,z,w]
    return np.concatenate([t, q, [1.0]]).astype(np.float32)


def _sim3_data_from_sRt(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(s, R_3x3, t_3) → (8,) pp.Sim3 data [t(3), q(4,xyzw), s(1)]."""
    from scipy.spatial.transform import Rotation as ScipyR
    q = ScipyR.from_matrix(R.astype(np.float64)).as_quat().astype(np.float32)
    return np.concatenate([t.astype(np.float32), q, [float(s)]]).astype(np.float32)


def _find_submap_node(submaps: list[Submap], image_path) -> int | None:
    """Return index in `submaps` list that contains `image_path`, or None."""
    for i, submap in enumerate(submaps):
        if image_path in submap.image_paths:
            return i
    return None


def run_sim3_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
    lm_steps: int = 10,
) -> np.ndarray:
    """Sim(3) pose graph optimization via pypose LM.

    One Sim(3) node per submap. Sequential edges encode scale-aware inter-submap
    alignment via umeyama_sim3 on overlap world_points. Loop edges use relative
    SE(3) from lc_submaps.poses with s=1.0. Returns (total_frames, 4, 4) float32.
    """
    import torch
    import pypose as pp
    import pypose.optim as ppopt
    from scipy.spatial.transform import Rotation as ScipyR
    from collab_splats.pointcloud.loop_closure.alignment import (
        dedup_overlap,
        overlap_region_align_sim3,
    )
    from collab_splats.pointcloud.loop_closure.pose_graph import Sim3PoseGraph

    N = len(submaps)
    if N == 0:
        return np.zeros((total_frames, 4, 4), dtype=np.float32)

    # --- Build initial Sim3 nodes (chain SE3 transforms, s=1.0) ---
    initial_data = np.zeros((N, 8), dtype=np.float32)
    accumulated_T = np.eye(4, dtype=np.float32)
    for i, submap in enumerate(submaps):
        initial_data[i] = _se3_to_sim3_data(accumulated_T)
        if i < N - 1:
            from collab_splats.pointcloud.loop_closure.alignment import overlap_region_align
            T_local = overlap_region_align(submap, submaps[i + 1], overlap_frames)
            accumulated_T = accumulated_T @ T_local

    # --- Sequential edges (scale-aware via umeyama_sim3) ---
    ii_list, jj_list, T_list = [], [], []
    for i in range(N - 1):
        s, R, t = overlap_region_align_sim3(submaps[i], submaps[i + 1], overlap_frames)
        T_list.append(_sim3_data_from_sRt(s, R, t))
        ii_list.append(i)
        jj_list.append(i + 1)

    # --- Loop closure edges (s=1.0 from VGGT 2-frame relative pose) ---
    for lc in lc_submaps:
        if lc.poses.shape[0] != 2:
            continue
        qi = _find_submap_node(submaps, lc.image_paths[0])
        di = _find_submap_node(submaps, lc.image_paths[1])
        if qi is None or di is None:
            continue
        # Relative pose: detected_frame → query_frame
        T_rel = (
            np.linalg.inv(lc.poses[1].astype(np.float64))
            @ lc.poses[0].astype(np.float64)
        ).astype(np.float32)
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]
        T_list.append(_sim3_data_from_sRt(1.0, R_rel, t_rel))
        ii_list.append(di)  # from detected submap
        jj_list.append(qi)  # to query submap

    # --- Optimize ---
    if ii_list and N > 1:
        model = Sim3PoseGraph(initial_data)
        all_ii = torch.tensor(ii_list, dtype=torch.long)
        all_jj = torch.tensor(jj_list, dtype=torch.long)
        all_T = pp.Sim3(torch.from_numpy(np.stack(T_list)).float())
        inp = {"ii": all_ii, "jj": all_jj, "T_ij": all_T}
        optimizer = ppopt.LM(model)
        for _ in range(lm_steps):
            optimizer.step(inp)
        optimized = model.optimized_poses()  # (N, 8)
    else:
        optimized = initial_data

    # --- Apply corrected submap poses to all frames ---
    corrected: dict[int, np.ndarray] = {}
    for i, submap in enumerate(submaps):
        t_opt = optimized[i, :3]
        q_opt = optimized[i, 3:7]
        R_opt = ScipyR.from_quat(q_opt.astype(np.float64)).as_matrix().astype(np.float32)

        T_opt_SE3 = np.eye(4, dtype=np.float32)
        T_opt_SE3[:3, :3] = R_opt
        T_opt_SE3[:3, 3] = t_opt

        # Compute correction: T_opt @ inv(T_initial)
        T_init_SE3 = np.eye(4, dtype=np.float32)
        q_init = initial_data[i, 3:7].astype(np.float64)
        T_init_SE3[:3, :3] = ScipyR.from_quat(q_init).as_matrix().astype(np.float32)
        T_init_SE3[:3, 3] = initial_data[i, :3]

        T_correction = T_opt_SE3 @ np.linalg.inv(T_init_SE3).astype(np.float32)

        K = submap.poses.shape[0]
        corr_poses = np.zeros((K, 4, 4), dtype=np.float32)
        for local_k in range(K):
            # Global corrected = T_correction @ (accumulated_T_init @ relative_local)
            # Simplified: correction applied to the initial global of frame-0,
            # then relative-within-submap applied from there.
            T0_inv = np.linalg.inv(submap.poses[0].astype(np.float64)).astype(np.float32)
            corr_poses[local_k] = T_correction @ T_init_SE3 @ T0_inv @ submap.poses[local_k]
        corrected[submap.submap_id] = corr_poses

    return dedup_overlap(
        submap_ids=[s.submap_id for s in submaps],
        submap_starts=[s.frame_start for s in submaps],
        corrected=corrected,
        total_frames=total_frames,
    )
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_sim3_pose_graph.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/closure.py tests/pointcloud/test_sim3_pose_graph.py
git commit -m "feat(closure): add run_sim3_pose_graph_optimization"
```

---

### Task 6: Wire up feedforward + exports

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py`
- Modify: `collab_splats/pointcloud/loop_closure/__init__.py`

- [ ] **Step 1: Update feedforward.py import + call**

In `collab_splats/pointcloud/feedforward.py`, find the block (around line 350):

```python
        from collab_splats.pointcloud.loop_closure.closure import (
            run_pose_graph_optimization,
            merge_submap_outputs,
        )
        corrected_extrinsics = run_pose_graph_optimization(
            submaps, lc_submaps, total_frames=N,
            overlap_frames=self.loop_closure_config.submap_overlap,
        )
```

Replace with:

```python
        from collab_splats.pointcloud.loop_closure.closure import (
            run_sim3_pose_graph_optimization,
            merge_submap_outputs,
        )
        corrected_extrinsics = run_sim3_pose_graph_optimization(
            submaps, lc_submaps, total_frames=N,
            overlap_frames=self.loop_closure_config.submap_overlap,
        )
```

- [ ] **Step 2: Update `__init__.py` exports**

In `collab_splats/pointcloud/loop_closure/__init__.py`, add alongside existing exports:

```python
from collab_splats.pointcloud.loop_closure.pose_graph import Sim3PoseGraph
from collab_splats.pointcloud.loop_closure.closure import run_sim3_pose_graph_optimization
from collab_splats.pointcloud.loop_closure.alignment import umeyama_sim3, overlap_region_align_sim3
```

- [ ] **Step 3: Verify import**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.loop_closure import (
    Sim3PoseGraph, run_sim3_pose_graph_optimization,
    umeyama_sim3, overlap_region_align_sim3,
)
print('all exports ok')
"
```

Expected: `all exports ok`

- [ ] **Step 4: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v --tb=short 2>&1 | tail -30
```

Expected: no regressions. Any pre-existing failures are acceptable; no new failures.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py collab_splats/pointcloud/loop_closure/__init__.py
git commit -m "feat(lc): switch loop closure optimizer to Sim(3) via pypose"
```

---

### Task 7: End-to-end verification

**Files:**
- Modify: `docs/pointcloud/loop_closure_eval.ipynb`

- [ ] **Step 1: Run eval notebook smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=300 \
  docs/pointcloud/loop_closure_eval.ipynb \
  --output /tmp/lc_eval_out.ipynb 2>&1 | tail -20
```

Expected: completes without error (timeout may vary by scene size).

- [ ] **Step 2: Check notebook §3b for Sim3 log token**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import nbformat
nb = nbformat.read('/tmp/lc_eval_out.ipynb', as_version=4)
outputs = [o['text'] for c in nb.cells for o in c.get('outputs', []) if 'text' in o]
text = ''.join(outputs)
print('Pose graph line:', [l for l in text.split('\n') if 'Pose graph' in l])
"
```

Expected: a line like `Pose graph: N frames, K loop edges → X.Xs`

- [ ] **Step 3: Update notebook §3b counter cell**

In `docs/pointcloud/loop_closure_eval.ipynb`, find the cell with `n_loop_rej` or similar rejection counters. Add:

```python
n_sim3_steps = lc_config.sim3_lm_steps if hasattr(lc_config, 'sim3_lm_steps') else 10
print(f"Sim3 LM steps: {n_sim3_steps}")
```

- [ ] **Step 4: Final commit**

```bash
git add docs/pointcloud/loop_closure_eval.ipynb
git commit -m "docs(notebook): note Sim3 optimizer in loop closure eval"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** `umeyama_sim3` ✓, `overlap_region_align_sim3` ✓, `Sim3PoseGraph` ✓, `run_sim3_pose_graph_optimization` ✓, `setup.sh` ✓, `__init__.py` exports ✓, feedforward swap ✓
- [x] **Placeholders:** None — all steps have actual code
- [x] **Type consistency:** `(s, R, t)` tuple used consistently; `_sim3_data_from_sRt` converts to `(8,)` everywhere; `Sim3PoseGraph.optimized_poses()` returns `(N, 8) float32` used in Task 5
- [x] **Node granularity doc:** Spec clarification on submap-level nodes reflected in Task 5 implementation
- [x] **GTSAM PoseGraph untouched:** Task 4 adds `Sim3PoseGraph` without touching existing class
