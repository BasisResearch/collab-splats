# SL(4) Loop Closure + VGGT-SLAM Comparison Eval

**Date:** 2026-05-22  
**Branch:** refactor/cu121 (or new branch from main)  
**ADR trigger:** ADR 002 re-evaluation trigger #2 — Python 3.11 upgrade complete; gtsam-develop with SL(4) bindings now available.

---

## Context

Our loop closure module was built on Python 3.10 with gtsam 4.2 stable (no SL(4)). ADR 002 explicitly deferred SL(4) until a Python 3.11 upgrade. That upgrade is now complete (`gtsam-develop` confirmed installed, `from gtsam import SL4` passes). Simultaneously, the module has accumulated dead code — a per-frame GTSAM SE(3) `PoseGraph` that was always a stepping stone, never the production path, and a PyPose `Sim3PoseGraph` that was a workaround.

This spec covers:
1. Module restructure — collapse dead code, rename files to mirror VGGT-SLAM
2. Unified `PoseGraph` on SL(4) (or SE(3) fallback), ported from `third_party/VGGT-SLAM/vggt_slam/graph.py`
3. Submap reprojection methods for per-submap and integrated visualization
4. 7-Scenes evaluation comparing our SL(4) pipeline against VGGT-SLAM out-of-the-box

---

## Module Restructure

### Before → After

```
loop_closure/                    loop_closure/
  __init__.py                      __init__.py       (exports updated)
  pose_graph.py      →             graph.py          (unified PoseGraph + helpers)
  retrieval.py       →             closure.py        (absorbed: config + types + NMS)
  alignment.py       →             closure.py        (absorbed: dedup_overlap only; rest deleted)
  closure.py         →             closure.py        (consolidated)
  submap.py                        submap.py         (+ reprojection methods)
  eval.py                          eval.py           (unchanged)
```

**Deleted:**
- `pose_graph.py` — replaced by `graph.py`
- `retrieval.py` — `ImageRetrieval` deleted (redundant with `localization.DinoSaladExtractor`); config/types absorbed into `closure.py`
- `alignment.py` — `dedup_overlap` moved to `closure.py`; `umeyama_se3`, `umeyama_sim3`, `overlap_region_align`, `overlap_region_align_sim3` deleted (only used by dead SE(3)/Sim3 paths)

**Dead code removed from `pose_graph.py`:**
- `PoseGraph` (GTSAM SE(3), per-frame nodes) — never used in production; comment said "upgrade path to SL(4)"
- `Sim3PoseGraph` (PyPose, per-submap) — SL(4) 15-DOF strictly subsumes Sim3 7-DOF; was a workaround

**Dead code removed from `closure.py`:**
- `run_pose_graph_optimization` (old SE(3) dispatch)
- `build_pose_graph` (old SE(3) builder)

---

## `graph.py` — Unified PoseGraph

**File description (module docstring):**
> Factor graph over submap projection matrices. Optimizes camera poses on the SL(4) manifold to correct trajectory drift and close loops. Ported and adapted from MIT-SPARK/VGGT-SLAM `vggt_slam/graph.py` (SL(4) backend) and `vggt_slam/slam_utils.py` (`decompose_camera`, `normalize_to_sl4`).

### Module-level helpers (ported from `vggt_slam/slam_utils.py` and `vggt_slam/scale_solver.py`)

```python
def decompose_camera(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """RQ decompose 3x4 or 4x4 projection matrix → (K, R, t, scale).
    Strips to (3,4), RQ via scipy.linalg.rq, sign-fixes K diagonal.
    Source: MIT-SPARK/VGGT-SLAM vggt_slam/slam_utils.py:decompose_camera
    """

def normalize_to_sl4(H: np.ndarray) -> np.ndarray:
    """Normalize 4x4 matrix so det=1 (SL(4) constraint): H / det(H)^(1/4).
    Source: MIT-SPARK/VGGT-SLAM vggt_slam/slam_utils.py:normalize_to_sl4
    """

def estimate_scale_pairwise(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimate scale between two point clouds: median(||Y[i]|| / ||X[i]||).
    Used to initialize inter-submap SL(4) edge H with correct scale.
    Source: MIT-SPARK/VGGT-SLAM vggt_slam/scale_solver.py:estimate_scale_pairwise
    """
```

**`proj_mats` convention:** `raw["proj_mats"]` from `VGGTXCreator` are 4×4 camera **intrinsics** (K_4x4), not extrinsics or their inverse. Inter-submap scale estimation uses `inv(K_prev) @ K_curr` to express points in a common frame before computing scale ratio.

### `PoseGraph` class

**Per-frame nodes** (one node per frame, NOT per submap) — mirrors VGGT-SLAM's `solver.py:add_edge` exactly. Node key = global frame index (`submap_id + frame_id` offset). This matches VGGT-SLAM's actual graph structure as confirmed by audit of `vggt_slam/graph.py` and `solver.py`.

```python
class PoseGraph:
    def __init__(self, manifold: Literal["sl4", "se3"] = "sl4"):
        # SL(4): gtsam.SL4 / BetweenFactorSL4 / PriorFactorSL4 — 15-DOF noise
        # SE(3): gtsam.Pose3 / BetweenFactorPose3 / PriorFactorPose3 — 6-DOF noise (fallback)

    def add_node(self, node_id: int, H: np.ndarray) -> None:
        """Insert per-frame node. node_id = global frame index. H is 4x4 projection matrix.
        normalize_to_sl4(H) applied before insertion."""

    def add_sequential_edge(self, id_i: int, id_j: int, H_rel: np.ndarray) -> None:
        """Sequential frame-to-frame constraint (inner-submap or inter-submap odometry)."""

    def add_loop_edge(self, id_i: int, id_j: int, H_rel: np.ndarray, t_norm: float = 0.0) -> None:
        """Loop closure constraint. Applies Huber kernel + per-edge translation downweighting.
        Our addition over VGGT-SLAM baseline."""

    def add_prior(self, node_id: int, H: np.ndarray) -> None:
        """Anchor first frame's node (tight prior, σ=1e-6)."""

    def optimize(self) -> dict[int, np.ndarray]:
        """Levenberg-Marquardt. Returns {node_id: corrected 4x4 H}."""

    def get_homography(self, node_id: int) -> np.ndarray:
        """Return optimized 4x4 H for a frame node (post-optimize)."""
```

**Noise model (SL(4), 15-DOF):**
- Inner-submap sequential edges: `0.05 * np.ones(15)` (matches VGGT-SLAM `inner_submap_noise`)
- Inter-submap sequential edges: `0.05 * np.ones(15)` (same — VGGT-SLAM makes no distinction)
- Loop edges: `0.15 * np.ones(15)` + Huber(k=1.0) + per-edge scale `1 + 0.1 * ||t||²` (our additions)
- Anchor prior: `1e-6 * np.ones(15)`

**SE(3) fallback noise (6-DOF):** existing tuned sigmas from ADR 005.

---

## `submap.py` — Reprojection Methods

Two methods added to the `Submap` dataclass. No graph coupling — H passed directly.

```python
def get_world_points(self, H: np.ndarray | None = None) -> np.ndarray:
    """Return world_points in global frame.

    H=None: local frame (per-submap visualization, no optimization required).
    H=(4,4): apply SL(4) projective transform H @ [pts|1]^T, dehomogenize /w.
    Requires world_points in per-camera local frame (PR #39 constraint).
    """

def get_poses_world(self, H: np.ndarray | None = None) -> np.ndarray:
    """Return (K, 4, 4) poses in global frame.

    H=None: self.poses (local, first pose ≈ identity).
    H=(4,4): apply corrected anchor H to each frame's local pose.
    """
```

**Node initialization:** `raw["extrinsic"]` from `VGGTXCreator` is `(K, 3, 4)`. Append `[0,0,0,1]` row → `(K, 4, 4)`. Graph inserts one node per frame (not per submap). First frame of first submap gets a prior. Subsequent frames initialized via `H_inner` chain (see `run_pose_graph_optimization`). `normalize_to_sl4(H)` applied before every insertion.

**PR #39 note:** `world_points` must be in per-camera local frame (not world frame of first submap). The existing `assert_world_to_cam` check at submap construction enforces this. This is the convention VGGT-SLAM's `VGGT_SPARK` fork fixes — our `VGGTXCreator` already follows it.

---

## `closure.py` — Consolidated Orchestration

### Absorbed from `retrieval.py`

```python
@dataclass
class LoopClosureConfig:
    submap_size: int = 20
    submap_overlap: int = 4
    lc_cosine_threshold: float = 0.75
    max_loops_per_submap: int = 5
    verify_match_ratio: float = 0.85
    nms_frame_distance: int = 25
    min_submap_gap: int = 1
    manifold: Literal["sl4", "se3"] = "sl4"   # NEW — default sl4; se3 for fallback/ablation

@dataclass
class LoopMatch: ...

class LoopMatchQueue: ...   # NMS heap, unchanged
```

`sim3_lm_steps` removed from `LoopClosureConfig` — Sim3 path deleted.

### Absorbed from `alignment.py`

```python
def dedup_overlap(submap_ids, submap_starts, corrected, total_frames) -> np.ndarray: ...
```

### Updated `run_pose_graph_optimization`

```python
def run_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
    manifold: Literal["sl4", "se3"] = "sl4",
) -> np.ndarray:
    """Build + optimize pose graph; return (total_frames, 4, 4) corrected extrinsics.

    Per-frame node building (mirrors vggt_slam/solver.py:add_edge):
    - First frame of first submap: add_prior + add_node(H=normalize_to_sl4(poses[0]))
    - Subsequent frames within submap: H_inner = poses[i-1] @ inv(poses[i]);
      node_H = graph.get_homography(prev_node) @ H_inner
    - Inter-submap boundary: estimate_scale_pairwise(overlap points) → H_scale=diag(s,s,s,1);
      H_w = graph.get_homography(overlap_prev) @ inv(K_prev) @ K_curr @ H_scale;
      add_node(first frame of new submap, H_w)
    - All inner-submap sequential edges: noise=0.05*np.ones(15)
    - Loop edges from lc_submaps: noise=0.15*np.ones(15) + Huber + translation downweight

    Post-optimize: calls decompose_camera per frame to extract SE(3) poses,
    dedup_overlap to expand to per-frame array.
    """
```

### Updated `merge_submap_outputs`

```python
def merge_submap_outputs(
    submaps: list[Submap],
    corrected_extrinsics: np.ndarray,
    graph: PoseGraph | None = None,   # if provided, uses get_homography for SL4 point reprojection
) -> dict:
    """Assemble unified raw_outputs with corrected poses and globally-consistent world_points.

    If graph provided: calls submap.get_world_points(graph.get_homography(sid)) per submap.
    Else: existing local-frame concatenation (SE(3)/Sim3 path).
    """
```

### Unchanged

```python
def translation_jump_check(...) -> tuple[bool, float]: ...
```

---

## `wrappers.py` — `LoopClosure` Updates

- Import `LoopClosureConfig` from `closure` (not `loop_closure`)
- `_run_lc_loop` passes `cfg.manifold` to `run_pose_graph_optimization`
- `merge_submap_outputs` called with `graph=pg` when `manifold="sl4"` for point reprojection
- Remove import of deleted `ImageRetrieval`; use `BaseRetrievalExtractor.get("dino-salad")` directly

---

## Evaluation

### Script: `evals/eval_vggt_slam_comparison.py`

**Dataset:** 7-Scenes (chess seq-01, office seq-01). Already downloaded.

**Conditions:**

| Condition | Pipeline | Notes |
|---|---|---|
| `baseline` | `VGGTXCreator` only | regression anchor |
| `lc_se3` | `LoopClosure(manifold="se3")` | SE(3) reference |
| `lc_sl4` | `LoopClosure(manifold="sl4")` | new SL(4) |
| `vggt_slam_oob` | subprocess `main.py` + parse log | OOB comparison |

**`vggt_slam_oob` condition:**
1. Rename 7-Scenes images to `%06d.png` format (workaround for VGGT-SLAM regex bug fixed in latest commit — issue #43)
2. Run: `python third_party/VGGT-SLAM/main.py --image_folder <rgb_dir> --log_results --log_path <tmp.txt> --max_loops 1 --min_disparity 50 --conf_threshold 25 --lc_thres 0.95 --submap_size 16 --skip_dense_log`
3. Parse TUM-format pose log → 4×4 matrices via `decompose_camera`
4. Evaluate against 7-Scenes GT using same ATE/RPE metrics as `eval_gt.py`

**Metric:** ATE RMSE (meters) via `evo_ape`. Same as existing `eval_gt.py`.

**Expected result:** `lc_sl4` ≈ VGGT-SLAM OOB on chess/office (Dominic confirmed ~same results as v1.0 with default params).

---

## Testing

### New: `tests/pointcloud/test_graph.py`

- `test_sl4_add_node` — add two per-frame nodes (node_id=0, node_id=1), assert both in initialized set
- `test_sl4_sequential_edge_optimize` — two nodes, one sequential edge, optimize, assert corrected H close to initial
- `test_sl4_loop_edge` — three nodes, sequential + loop edge, optimize, no crash
- `test_decompose_camera_round_trip` — `P = K @ [R|t]`, `decompose_camera(P)` recovers K, R, t within tolerance
- `test_estimate_scale_pairwise` — two synthetic point clouds with known scale ratio, assert median matches
- `test_se3_fallback` — `PoseGraph(manifold="se3")`, same interface, optimize without crash
- `test_get_homography_post_optimize` — verify `get_homography` returns post-optimize values

### New: `tests/pointcloud/test_submap_reprojection.py`

- `test_get_world_points_identity` — `get_world_points(H=np.eye(4))` ≈ `get_world_points(H=None)` for local-frame points (identity H = no-op for Euclidean pts)
- `test_get_world_points_translation` — known translation H, assert pts shift correctly
- `test_get_world_points_projective` — non-trivial SL(4) H, assert dehomogenization `/w` applied
- `test_get_poses_world_none` — returns `submap.poses` unchanged
- `test_get_poses_world_corrected` — known H, assert first pose ≈ H

### Updated: existing LC tests

- `tests/pointcloud/test_localization.py` — no import changes (localization unaffected)
- `tests/pointcloud/test_bundle_adjustment.py` — no changes
- `tests/dashboard/*` — no changes
- Remove: any test importing `alignment.py` symbols directly; update imports to `closure`
- Remove: any test of `Sim3PoseGraph` or `run_sim3_pose_graph_optimization`

---

## Key Constraints

1. **Per-camera local frame** — `Submap.world_points` must be per-camera local (enforced by `assert_world_to_cam`). Violating this causes the PR #39 scale drift bug.
2. **gtsam-develop installed** — `from gtsam import SL4, BetweenFactorSL4, PriorFactorSL4` confirmed passing.
3. **VGGT-SLAM 7-Scenes regex** — issue #43 fixed in latest upstream commit. Pull latest `third_party/VGGT-SLAM` or apply image rename step before running `vggt_slam_oob` condition.
4. **SE(3) fallback** — `manifold="se3"` must remain functional. Existing eval conditions (`lc_se3`) use it; downstream `BundleAdjustment` wrapping `LoopClosure` unaffected.
5. **Attribution** — `graph.py` module docstring cites `MIT-SPARK/VGGT-SLAM`. `decompose_camera` and `normalize_to_sl4` cite `vggt_slam/slam_utils.py`.

---

## What Does NOT Change

- `Submap` dataclass fields — additions only (two methods)
- `assert_world_to_cam` — unchanged
- `eval.py` (`capture_pose_graph_loss`) — unchanged
- `VGGTXCreator`, `MapAnythingCreator` — unchanged
- `BundleAdjustment` wrapper — unchanged
- `evals/eval_gt.py` — unchanged (new script is separate)
- `evals/datasets.py` — unchanged
