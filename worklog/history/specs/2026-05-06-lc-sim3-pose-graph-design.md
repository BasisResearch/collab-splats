# Spec: Loop Closure Sim(3) Pose Graph

**Date:** 2026-05-06  
**Branch:** `refactor/core-modules`  
**Status:** Ready for implementation  
**Supersedes:** `2026-05-06-lc-geometric-overlap-gate-design.md` (Gate 3 dropped — not present in either reference)

---

## Context

Current loop closure uses GTSAM `BetweenFactorPose3` (SE(3), 6-DOF). VGGT runs independently per
20-frame submap window; each window has its own internal metric scale. Merging with SE(3) ignores
inter-submap scale drift → accordion-fold / sheared pointclouds.

VGGT-Long fixes this with Sim(3) (7-DOF: rotation + translation + scale). Our GTSAM Python bindings
expose `Similarity3` as an object but do **not** expose `BetweenFactorSimilarity3`, so we cannot use
GTSAM for Sim(3) factors. pypose (`pip install pypose`, 144 kB, installable in nerfstudio env) provides
native `pp.Sim3` and `pp.optim.LM`.

**Decision:** Add pypose. Keep GTSAM untouched (upgrade path to SL(4), which VGGT-SLAM uses).
Replace GTSAM SE(3) optimizer in the loop closure path with pypose Sim(3).

---

## Reference: VGGT-Long approach

VGGT-Long (`DengKaiCQ/VGGT-Long`, `loop_utils/sim3loop.py` + `sim3utils.py`):
1. For each adjacent chunk pair: `estimate_sim3(pts_overlap_b, pts_overlap_a)` on overlap world_points
   → `(s, R, t)` sequential transform
2. For each detected loop: same `estimate_sim3` on 2-frame world_points (shared coordinate system)
   → `(s, R, t)` loop constraint
3. `Sim3LoopOptimizer.optimize(sequential_transforms, loop_constraints)` → optimized poses
4. Apply optimized Sim(3) to world_points → globally scale-consistent cloud

We replicate steps 1–4, using `pp.optim.LM` (from pypose's BA example pattern) as the optimizer
instead of VGGT-Long's `fastloop` dependency (not pip-installable).

---

## Design

### New: `umeyama_sim3` in `alignment.py`

```python
def umeyama_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3) alignment via Umeyama (with scale).

    Args:
        source: (M, 3) points in source frame.
        target: (M, 3) corresponding points in target frame.
        weights: optional (M,) non-negative weights.

    Returns:
        (s, R, t): scalar scale, (3,3) rotation, (3,) translation
                   such that target ≈ s * R @ source + t
    """
```

Math (from VGGT-Long `estimate_sim3`):
- Compute weighted means `μ_src`, `μ_tgt`
- Scale: `s = std(target_centered) / std(source_centered)` (RMS ratio)
- Rotation: SVD of `(s * src_centered)^T @ tgt_centered`
- Translation: `t = μ_tgt - s * R @ μ_src`

### New: `overlap_region_align_sim3` in `alignment.py`

```python
def overlap_region_align_sim3(
    submap_a: Submap,
    submap_b: Submap,
    overlap_frames: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) transform (s, R, t) s.t. pts_a_world ≈ s * R @ pts_b_local + t.

    Parallel to overlap_region_align (SE3). Both kept; old used by GTSAM path.
    Returns (1.0, eye(3), zeros(3)) if world_points unavailable.
    """
```

`umeyama_se3` and `overlap_region_align` are **not changed** — GTSAM path still uses them.

### New: `Sim3PoseGraph` in `pose_graph.py`

```python
import torch
import torch.nn as nn
import pypose as pp
import numpy as np

class Sim3PoseGraph(nn.Module):
    """Pose graph on Sim(3) manifold, optimized via pypose LM.

    Each node is a pp.Sim3 pose (world-to-cam). Edges are BetweenFactor
    residuals: log(T_j^{-1} @ T_i @ T_ij) should be zero.
    """

    def __init__(self, initial_poses: np.ndarray):
        """
        Args:
            initial_poses: (N, 8) float32 — pp.Sim3 data format [t(3), q(4,xyzw), s(1)]
        """
        super().__init__()
        self.poses = pp.Parameter(
            pp.Sim3(torch.from_numpy(initial_poses).float()),
            requires_grad=True,
        )

    def forward(
        self,
        ii: torch.Tensor,   # (E,) edge source indices
        jj: torch.Tensor,   # (E,) edge target indices
        T_ij: pp.Sim3,      # (E,) relative transforms
        weights: torch.Tensor | None = None,  # (E,) per-edge weights
    ) -> torch.Tensor:
        """Returns weighted residuals (E, 7) — should be zero at optimum."""
        residuals = (self.poses[jj].Inv() @ self.poses[ii] @ T_ij).Log()
        if weights is not None:
            residuals = residuals * weights[:, None]
        return residuals

    def optimized_poses(self) -> np.ndarray:
        """Returns (N, 8) optimized Sim3 data after optimization."""
        return self.poses.data.detach().cpu().numpy()
```

`PoseGraph` (GTSAM SE3 class) is **not changed or deleted**.

### Node granularity: submap-level

`Sim3PoseGraph` operates at **submap granularity** — one node per submap (N nodes for N submaps),
not per-frame. This matches VGGT-Long's `Sim3LoopOptimizer` which takes one transform per chunk.
After optimization, the corrected submap-level Sim3 is applied to all frames within each submap:
`corrected_frame_pose = T_submap_corrected @ submap.poses[local_frame_idx]`.

Initial Sim3 node for submap `i` = frame-0 pose converted to Sim3 with `s=1.0`:
```python
R = submap.poses[0, :3, :3]   # (3,3) — world-to-cam rotation
t = submap.poses[0, :3, 3]    # (3,) — world-to-cam translation
q = R_to_quat_xyzw(R)
initial_poses[i] = np.concatenate([t, q, [1.0]])  # (8,) pp.Sim3 format
```

### New: `run_sim3_pose_graph_optimization` in `closure.py`

```python
def run_sim3_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    overlap_frames: int,
    lm_steps: int = 10,
) -> np.ndarray:
    """Sim(3) pose graph optimization replacing GTSAM SE(3) path.

    Returns:
        corrected_extrinsics: (total_frames, 4, 4) float32 world-to-cam.
    """
```

Internal steps:
1. Build initial Sim3 nodes from submap poses (frame 0 of each submap)
2. Build sequential edges via `overlap_region_align_sim3` on adjacent submaps
3. Build loop edges from `lc_submaps.poses` (relative transform, `s=1.0`)
4. Optimize with `pp.optim.LM(model, steps=lm_steps)`
5. Extract corrected poses → apply to all frames in each submap
6. Return `(total_frames, 4, 4)` via `merge_submap_outputs`

### Changes in `feedforward.py`

`_run_loop_closure_inference` currently calls `run_pose_graph_optimization` (GTSAM).
Replace that single call with `run_sim3_pose_graph_optimization`.
No other changes to feedforward.py.

```python
# Before:
corrected_extrinsics = run_pose_graph_optimization(submaps, lc_submaps, O)

# After:
corrected_extrinsics = run_sim3_pose_graph_optimization(submaps, lc_submaps, O)
```

### Changes in `setup.sh`

```bash
# After existing pip installs in nerfstudio env block:
pip install pypose
```

### `LoopClosureConfig` additions (optional, can tune later)

```python
sim3_lm_steps: int = 10          # LM iterations for Sim(3) optimizer
sim3_loop_scale: float = 1.0     # scale assigned to loop closure edges (s=1.0 = trust VGGT relative pose)
```

---

## Sim(3) → SE(3) extraction for COLMAP output

`FeedforwardResult` expects `(N, 3, 4)` extrinsic (R|t, no scale). After Sim(3) optimization,
extract R and t from each pose, discard scale: `R, t = pose[:3, :3], pose[:3, 3]`.
The scale is absorbed into the world_points (which are not re-computed here — deferred to
calibration task ADR-005).

---

## Files changed

| File | Change |
|------|--------|
| `collab_splats/pointcloud/loop_closure/alignment.py` | Add `umeyama_sim3`, `overlap_region_align_sim3` |
| `collab_splats/pointcloud/loop_closure/pose_graph.py` | Add `Sim3PoseGraph(nn.Module)`. Keep `PoseGraph` (GTSAM) |
| `collab_splats/pointcloud/loop_closure/closure.py` | Add `run_sim3_pose_graph_optimization` |
| `collab_splats/pointcloud/feedforward.py` | Swap `run_pose_graph_optimization` → `run_sim3_pose_graph_optimization` (1 line) |
| `collab_splats/pointcloud/loop_closure/__init__.py` | Export `Sim3PoseGraph`, `run_sim3_pose_graph_optimization` |
| `setup.sh` | `pip install pypose` |

---

## Unit tests

New file: `tests/pointcloud/test_sim3_pose_graph.py`

```python
def test_umeyama_sim3_known_transform():
    # Create target = 2.0 * R45 @ source + [1,0,0]
    # Verify recovered (s, R, t) matches ground truth

def test_sim3_pose_graph_no_loop_no_drift():
    # 3 submaps, no loops, known sequential transforms
    # After optimization poses should be close to identity chain

def test_sim3_pose_graph_loop_corrects_drift():
    # 5 submaps with injected drift + 1 loop closing back to submap 0
    # After optimization drift should shrink significantly

def test_overlap_region_align_sim3_returns_unit_without_world_points():
    # Submap with world_points=None → (1.0, eye(3), zeros(3))
```

---

## End-to-end verification

Re-run `docs/pointcloud/loop_closure_eval.ipynb` with `enable_loop_closure=True`.

Check:
- No crash / import error for pypose
- `§6 loss reduction` still present
- Pointcloud visually more coherent (no accordion folds between submaps)
- `overlap=` log tokens absent (Gate 3 never implemented)

---

## Out of scope

- Scale-aware `translation_jump_check` — deferred (check still valid as lower bound)
- GTSAM `BetweenFactorSimilarity3` — not in Python bindings; track if future GTSAM release adds it
- GTSAM SL(4) — VGGT-SLAM approach; keep `PoseGraph` as upgrade path
- World-points re-projection after scale correction (ADR-005 field calibration)
- Loop edge scale estimation from world_points (currently `s=1.0`; refine empirically)
