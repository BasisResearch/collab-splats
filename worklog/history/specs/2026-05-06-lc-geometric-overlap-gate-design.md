# Spec: Loop Closure 3D Geometric Overlap Gate

**Date:** 2026-05-06  
**Branch:** `refactor/core-modules`  
**Status:** Ready for implementation

---

## ⚠️ AGENT INSTRUCTION — DO THIS FIRST

Before implementing anything, fetch and read the relevant VGGT-SLAM and VGGT-Long source files:

```
VGGT-Long:  https://github.com/Linketic/VGGT-Long
  Key file: loop_utils/loop_refinement.py
  Look for: ransac_umeyama, world_points usage, acceptance criteria

VGGT-SLAM:  https://github.com/MIT-SPARK/VGGT-SLAM (or MIT-SPARK org)
  Key file: loop_closure related modules
  Look for: geometric verification approach, thresholds
```

Confirm:
1. What data structure they use for geometric verification (world_points vs tracks vs depth)
2. Their inlier threshold / overlap ratio default values
3. Whether they use confidence weighting
4. Whether they subsample the point cloud before KD-tree

If the repos confirm a meaningfully different approach than described here, adapt the design accordingly and note the divergence. The goal is alignment with VGGT-Long semantics, not blind copy.

---

## Context

Loop closure verification currently uses two gates:
1. DINO-SALAD L2 retrieval (candidate selection)
2. `image_match_ratio` — cross-frame attention similarity gate added via `_patch_vggtx_compute_similarity`

Gate 2 is appearance-only. False positives that look similar in embedding space but have different 3D geometry slip through. VGGT-Long uses a geometric consistency check on world-space 3D point clouds after a joint 2-frame forward pass to catch these.

We add Gate 3: a 3D overlap check using `world_points` already produced by the existing 2-frame verification forward in `VGGTXCreator._verify_loop_candidate`.

---

## Key finding from codebase exploration

`VGGTXCreator._verify_loop_candidate` (`collab_splats/pointcloud/feedforward.py:1052`) already runs:

```python
predictions = self.model(lc_frames, compute_similarity=True)
```

`predictions` already contains:
- `predictions["world_points"]` — `(B, S, H, W, 3)` — 3D world coords for each pixel, shared coordinate system
- `predictions["world_points_conf"]` — `(B, S, H, W)` — per-pixel confidence scores

Both frames' `world_points` are expressed in the **same shared coordinate system** from the joint 2-frame VGGT forward — no additional transformation needed. Gate 3 adds ~15 lines of numpy + KD-tree to the existing function, with zero extra model inference.

`facebook/VGGT-1B` confirmed to have `TrackHead` (track_head is not None), but track-based correspondences are deferred — see Out of Scope.

---

## Design

### Gate position in `_verify_loop_candidate`

```
[Gate 2] image_match_ratio check      →  return (False, None) if fails
[Gate 3] world_points overlap check   →  return (False, None) if fails   ← NEW
Extract lc_poses
return (True, lc_poses)
```

### Algorithm

```python
# After match_ratio check, before extracting lc_poses:
cfg = self.loop_closure_config

if cfg.geometric_overlap_ratio > 0.0:
    world_pts  = predictions["world_points"].squeeze(0).cpu().float().numpy()   # (2, H, W, 3)
    world_conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy()  # (2, H, W)

    step = cfg.geometric_subsample_step   # default 4
    pts1 = world_pts[0].reshape(-1, 3)
    pts2 = world_pts[1].reshape(-1, 3)
    c1   = world_conf[0].reshape(-1)
    c2   = world_conf[1].reshape(-1)

    pts1 = pts1[c1 > cfg.geometric_conf_threshold][::step]
    pts2 = pts2[c2 > cfg.geometric_conf_threshold][::step]

    if pts1.shape[0] >= 10 and pts2.shape[0] >= 10:
        overlap = point_cloud_overlap_ratio(pts1, pts2, cfg.geometric_distance_threshold)
        if overlap < cfg.geometric_overlap_ratio:
            console.log(
                f"  ✗ Loop rejected (overlap={overlap:.2f} < {cfg.geometric_overlap_ratio}): "
                f"submap {match.query_submap_id} → {match.detected_submap_id}"
            )
            return False, None
```

`point_cloud_overlap_ratio` is a new helper in `alignment.py` (see below). Import it at top of `feedforward.py` alongside existing `alignment` imports.

If fewer than 10 high-confidence points survive filtering, gate is skipped (not rejected) — don't punish scenes where VGGT produces low-confidence depth.

### New helper: `point_cloud_overlap_ratio`

Add to `collab_splats/pointcloud/loop_closure/alignment.py`:

```python
def point_cloud_overlap_ratio(
    pts1: np.ndarray,
    pts2: np.ndarray,
    distance_threshold: float,
) -> float:
    """Fraction of pts1 points with a nearest neighbour in pts2 within distance_threshold.

    Args:
        pts1: (M, 3) float32/64 query cloud.
        pts2: (N, 3) float32/64 reference cloud.
        distance_threshold: accept radius in world units.

    Returns:
        Scalar in [0, 1]. Higher = more overlap.
    """
    from scipy.spatial import cKDTree
    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return 0.0
    dists, _ = cKDTree(pts2).query(pts1, k=1, workers=1)
    return float((dists < distance_threshold).mean())
```

### New `LoopClosureConfig` fields (`retrieval.py`)

```python
geometric_overlap_ratio: float = 0.0        # 0.0 = gate disabled (default-off for safe rollout)
geometric_distance_threshold: float = 0.5   # world-unit accept radius; needs calibration per scene
geometric_conf_threshold: float = 0.5       # world_points_conf cutoff before KD-tree
geometric_subsample_step: int = 4           # stride for subsampling before KD-tree
```

**Default-off** (`geometric_overlap_ratio = 0.0`) preserves existing behavior. Enable via `LoopClosureConfig(geometric_overlap_ratio=0.25)`. Threshold `0.5` world-units is a starting point — VGGT scene scale varies; consult VGGT-Long defaults after fetching repo.

### Acceptance log line update

Also update the acceptance log to include overlap value:

```
↩ Loop: submap 2 → 0  dist=0.638 match_ratio=0.91 overlap=0.41 jump=0.08
```

### MapAnything: no change

`MapAnythingCreator._verify_loop_candidate` uses DINO cosine only; no VGGT forward, no `world_points`. Gate 3 is VGGT-X only.

---

## Files to modify

| File | Change |
|------|--------|
| `collab_splats/pointcloud/loop_closure/retrieval.py` | Add 4 fields to `LoopClosureConfig` |
| `collab_splats/pointcloud/loop_closure/alignment.py` | Add `point_cloud_overlap_ratio` helper |
| `collab_splats/pointcloud/feedforward.py` | Add Gate 3 block in `VGGTXCreator._verify_loop_candidate` (~line 1063); import `point_cloud_overlap_ratio` |
| `docs/pointcloud/loop_closure_eval.ipynb` | §3b: add `n_overlap_rej = _log_text.count('Loop rejected (overlap')` counter |

No new modules. `scipy.spatial.cKDTree` is already a transitive dependency via nerfstudio.

---

## Unit tests

New file: `tests/pointcloud/test_lc_geometric_gate.py`

```python
import numpy as np
from collab_splats.pointcloud.loop_closure.alignment import point_cloud_overlap_ratio

def test_nonoverlapping_clouds():
    pts1 = np.random.rand(200, 3).astype(np.float32)
    pts2 = np.random.rand(200, 3).astype(np.float32) + 10.0
    assert point_cloud_overlap_ratio(pts1, pts2, distance_threshold=0.5) < 0.01

def test_overlapping_clouds():
    pts1 = np.random.rand(200, 3).astype(np.float32)
    pts2 = pts1 + np.random.rand(200, 3).astype(np.float32) * 0.05
    assert point_cloud_overlap_ratio(pts1, pts2, distance_threshold=0.5) > 0.95

def test_empty_cloud_returns_zero():
    pts1 = np.zeros((0, 3), dtype=np.float32)
    pts2 = np.random.rand(100, 3).astype(np.float32)
    assert point_cloud_overlap_ratio(pts1, pts2, distance_threshold=0.5) == 0.0

def test_gate_disabled_at_zero_ratio():
    # LoopClosureConfig(geometric_overlap_ratio=0.0) → gate block not entered
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig
    cfg = LoopClosureConfig()
    assert cfg.geometric_overlap_ratio == 0.0
```

---

## End-to-end verification

Re-run `docs/pointcloud/loop_closure_eval.ipynb` with:

```python
lc_config = LoopClosureConfig(geometric_overlap_ratio=0.25)
```

Confirm:
- No crash
- §3b overlap rejection counter present (value may be 0 on bicycle scene if all accepted candidates are geometrically consistent — that's fine)
- §6 loss reduction still ≥ 50%
- `overlap=X.XX` appears in the acceptance log line

---

## Out of scope

- **Option C (track-based):** `track_head` confirmed present in `facebook/VGGT-1B`. Future path: pass `query_points` grid into 2-frame forward → `predictions["track"]` → true pixel correspondences → KD-tree on tracked `world_points`. Deferred — adds forward-call complexity for marginal gain over overlap check.
- **MapAnything geometric gate:** no `world_points` without a second model.
- **Threshold calibration:** ADR-005 field-tuning trigger. `geometric_distance_threshold=0.5` is a starting point; tune empirically on real scenes.
- **Pre-retrieval chained-submap Option A** from `2026-05-06-loop-closure-3d-gate-feasibility.md` — separate task.
