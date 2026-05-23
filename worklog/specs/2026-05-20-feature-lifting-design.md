# Feature Lifting: 2D → 3D via VGGT Pointcloud

**Date:** 2026-05-20
**Status:** approved

## Problem

VGGT-X produces a pointcloud `pts3d (P, 3)` with RGB colors but no feature vectors. To enable downstream semantic queries, segmentation, or mesh coloring, each 3D point needs an associated feature vector from a 2D extractor (DINOv2, MaskCLIP, Talk2DINO, etc.).

## Key Structural Insight

`pts3d` is produced by boolean-indexing `points3d (N, H, W, 3)` through `conf_mask (N, H, W)`:

```python
pts_out = points3d[conf_mask]   # (P, 3)
colors_out = colors_np[conf_mask]  # (P, 3)
```

`np.where(conf_mask)` gives `(frame, row, col)` for every point — free and exact. Each point came from exactly one source pixel. Same physical surface appears once per viewing frame as a duplicate entry; points are not deduplicated in the depth-unprojection path.

## Design

### No new files

`lift_features` is a utility function added to the existing `collab_splats/pointcloud/utils.py`. `extractor_name` field goes on `BaseFeedforwardCreator` (not just `VGGTXCreator`) so MapAnything gets it for free.

### Data Flow

```
conf_mask (N, H, W) bool  [randomly_limit_trues already applied]
  ↓ np.stack(np.where(...), axis=1)
point_source (P, 3) int32  ← [frame, row, col] per point

images (N, 3, H, W)  [sequential, one frame at a time]
  ↓ BaseFeatureExtractor
feats_i (H_p, W_p, D)  ← patch-level (e.g. 37×37×384 for DINOv2-small)

features[point_source[:,0]==i] = feats_i[
    point_source[mask_i,1] // patch_stride,
    point_source[mask_i,2] // patch_stride
]

Output: features (P, D) float32 — index-aligned with pts3d (P, 3)
```

Peak memory per frame: ~2 MB (DINOv2-small). Running `features (P, D)` buffer stays allocated; P=500K, D=384 ≈ 768 MB. Use PCA projection to reduce D if needed.

### `point_source` storage

Must store `point_source (P, 3)` in `FeedforwardResult` (not recompute on demand) because `randomly_limit_trues` is non-deterministic — the same conf_mask re-evaluated yields different random subsets. Storing indices ensures post-BA re-lift uses exactly the same point subset.

### Post-BA re-lift

`depth_conf` is unchanged by BA (only extrinsics/intrinsics are refined), so the stored `point_source` remains valid. `_reproject_after_ba` updates `pts3d` and `colors` using new poses, then re-calls `lift_features` with the same stored `point_source`. No features are dropped.

## Files Changed

| File | Change |
|---|---|
| `collab_splats/pointcloud/utils.py` | Add `lift_features(images, point_source, extractor_name, device)` |
| `collab_splats/pointcloud/feedforward/base.py` | Add `features`, `point_source` fields to `FeedforwardResult`; add `extractor_name: str | None = None` to `BaseFeedforwardCreator` |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Extend `unproject_and_filter_points` → also returns `point_source`; wire `_postprocess` + `_reproject_after_ba` |
| `collab_splats/semantics/features.py` | No changes (consumed via `BaseFeatureExtractor.get(name)`) |

## Not In Scope (V2)

Multi-frame averaging via reprojection: reproject `pts3d[p]` into other frames, depth-consistency gate to exclude occluded frames, average surviving features. Requires occlusion check `|z_reproj - depth[j,v,u]| < tol`. Deferred — source-frame assignment is a meaningful lift on its own.

## Verification

1. `VGGTXCreator(extractor_name="dinov2")` on small image dir:
   - `result.features.shape == (len(result.pts3d), D)`
   - `result.point_source.shape == (len(result.pts3d), 3)`
2. Pixel sanity: `colors[p] ≈ images[point_source[p,0], :, point_source[p,1], point_source[p,2]] * 255` for 10 random `p`
3. `python tests/test_grouping.py` — no regressions
