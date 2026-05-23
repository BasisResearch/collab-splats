# Loop Closure Bug Fixes: VGGT-X Similarity Gate + Translation Jump

**Date:** 2026-05-06  
**Branch:** refactor/core-modules  

---

## Problem

Loop closure performs worse than full VGGT because three bugs cause false loops to flood GTSAM:

### Bug 1: `compute_similarity` TypeError → all candidates auto-accept

`VGGTXCreator._verify_loop_candidate` calls `self.model(lc_frames, compute_similarity=True)` but VGGT-X doesn't have this param → `TypeError` → `except` branch → `match_ratio = 1.0` → every DINO-retrieval candidate passes verification.

### Bug 2: `translation_jump_check` inter-submap boundary is meaningless

```python
for si in range(detected_idx, query_idx):
    last_t = submaps[si].poses[-1][:3, 3]       # local frame of submap si
    first_t = submaps[si + 1].poses[0][:3, 3]   # local frame of submap si+1 — DIFFERENT
    path_length += float(np.linalg.norm(first_t - last_t))  # garbage
```

Each submap's poses are in that submap's local frame (frame 0 = identity). Cross-submap pose difference is geometrically meaningless.

### Bug 3: `MapAnythingCreator._verify_loop_candidate` returns `(True, None)`

Fallback in `_run_loop_closure_inference`:
```python
if lc_poses is None:
    lc_poses = np.stack([
        submap.poses[match.query_frame_idx],      # submap N local frame
        d_submap.poses[match.detected_frame_idx], # submap M local frame — DIFFERENT
    ])
```
These poses are in different local coordinate systems → LC constraint invalid → GTSAM gets garbage edge.

---

## Solution

### Fix 1: Monkey-patch VGGT-X to add `compute_similarity` (VGGT-SPARK parity)

Add `_patch_vggtx_compute_similarity(model)` in `feedforward.py`, called from `VGGTXCreator._load_model`.

**Mechanism**: Register a forward hook on `aggregator.global_blocks[-1].attn.qkv` (last global attention layer) to capture `k` and `q` during the forward pass without extra computation. Port `get_similarity(k, q)` from VGGT-SPARK verbatim. Store result on `aggregator._last_image_match_ratio`. Patch `VGGT.forward` to accept `compute_similarity`, set a flag on aggregator before calling original forward, then inject `image_match_ratio` into predictions dict.

**Why last layer**: VGGT-SPARK uses `target_layer=20` in its architecture (may differ from VGGT-X's layer count). Using `global_blocks[-1]` is equivalent — deepest global attention gives the most semantic cross-frame similarity signal.

**Idempotent**: Guard with `_vggtx_similarity_patched` attr on Aggregator class, like `_torch_compat_patched` on MapAnything.

### Fix 2: `translation_jump_check` — drop inter-submap boundary terms

Remove the inner `for si in range(detected_idx, query_idx)` boundary-jump loop. Replace with intra-submap summation for intermediate submaps:

```python
# Before: boundary jumps (broken — different local frames)
for si in range(detected_idx, query_idx):
    path_length += norm(submaps[si].poses[-1][:3,3] - submaps[si+1].poses[0][:3,3])

# After: sum within each intermediate submap (valid — same local frame)
for si in range(detected_idx + 1, query_idx):
    for f in range(submaps[si].poses.shape[0] - 1):
        path_length += norm(submaps[si].poses[f+1][:3,3] - submaps[si].poses[f][:3,3])
```

This gives a lower bound (misses overlap-zone motion at boundaries), making the check slightly lenient — acceptable tradeoff.

### Fix 3: `MapAnythingCreator` — reject when lc_poses is None

MapAnything has no `pose_enc`, so `_verify_loop_candidate` correctly returns `(True, None)`. The fallback that stacks poses from different local frames is the bug. Fix: in `_run_loop_closure_inference`, skip LC submap creation when `lc_poses is None` (treat as rejection).

---

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/pointcloud/feedforward.py` | Add `_patch_vggtx_compute_similarity`; call from `VGGTXCreator._load_model`; fix `_run_loop_closure_inference` None-lc_poses fallback |
| `collab_splats/pointcloud/loop_closure/closure.py` | Fix `translation_jump_check` inter-submap boundary |

---

## What We Don't Fix (Deferred)

- **Confidence filtering in `_raw_to_world_points`**: Points stored without conf threshold; `world_points_conf` used as weights in Umeyama (correct behavior — all points contribute weighted by confidence). Not a critical bug.
- **3D geometric KD-tree gate**: Requires global coordinate frame for cross-submap point clouds. Still needs inter-submap transforms. Deferred per `2026-05-06-loop-closure-3d-gate-feasibility.md`.
- **MapAnything joint forward for LC poses**: Would require running MapAnything on the 2-frame pair. Out of scope; MapAnything LC is disabled pending this.
