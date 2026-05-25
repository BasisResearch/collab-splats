# lift_features Bug Fix + Multi-View Aggregation Design

**Date:** 2026-05-24
**Status:** draft
**Scope:** `collab_splats/pointcloud/utils.py`, `collab_splats/pointcloud/feedforward/{base,vggt_omega,vggtx,mapanything}.py`, `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`, `tests/pointcloud/test_feature_lifting.py`

## Problem

Lifted per-point features are spatially smeared — points cluster their sampled features around the top-left of every frame's feature map instead of resolving to their true source pixel. Multiple bugs combine to produce this, and the single-source design also misses information available across frames.

### Bug 1 — coord-frame mismatch in notebook

`semantic_lifting.ipynb:288`:
```python
image_size = (imgs[0].height, imgs[0].width)   # PIL — original-image resolution
```

Passed to `lift_features(feature_maps, out.pixel_indices, image_size)`. But `out.pixel_indices` come from `unproject_and_filter_points` against depth at **model resolution** (e.g. `518 × 518`). Original images are far larger (e.g. `1080 × 1920`).

`lift_features` normalizes coords as
```python
gx = (2 * cols + 1) / W - 1
gy = (2 * rows + 1) / H - 1
```
With `W = orig_w`, `H = orig_h` but `cols, rows < model_w, model_h`, every normalized coordinate clusters near `-1` (top-left). All points sample the top-left strip of every feature map. Dominant cause of the observed smear.

### Bug 2 — broken `lift_features` call in feedforward creators

`vggt_omega.py:219` and `vggtx.py:298`:
```python
features = lift_features(raw_outputs["images"], pixel_indices, self.extractor_name, device)
```

The function takes `(feature_maps, pixel_indices, image_size)` — 3 positional args. The call passes 4, including a `str` (`extractor_name`) where `tuple[int, int]` is expected. Raises `TypeError` whenever `self.extractor_name` is truthy — the branch is effectively dead. Worse, `raw_outputs["images"]` is the raw RGB tensor, not feature maps — even if the signature matched, it would lift colors.

### Bug 3 — FOV mismatch between extractor input and depth-map FOV

Notebook calls `extractor.forward(imgs)` where `imgs` is full-resolution PIL opened from disk. VGGT consumed a different tensor: cropped/resized via `load_and_preprocess_images_square` / `_ratio` / Omega's center-crop. Extractor feature maps therefore cover a different field-of-view than the depth map. Even with the coord frame fixed, normalized coords land at the wrong scene location.

All three creators already populate `FeedforwardResult.images` with the exact post-crop, pre-normalization `(N, 3, model_h, model_w)` tensor in `[0, 1]` — the right tensor exists, it just isn't reaching the extractor.

### Bug 4 — single-source sampling is noisy

Current design assigns each point the feature from its single source frame. Source-frame features are subject to extractor noise, occlusion at view boundaries, and patch-grid quantization. Points seen across many frames carry no benefit from that redundancy. Result: per-point features fluctuate even when the underlying 3D point is well-determined.

A confidence-weighted multi-view aggregation closes this gap and is the natural fix while we're already touching the API.

## Design Principles

- **Single coord frame** — pixel_indices live in `(model_h, model_w)`; everything else must match.
- **Lifting is not a creator concern** — creators emit geometry + the image tensor; semantics consumers wire the lift.
- **Couple to `FeedforwardResult`, decouple from extractor + AE** — aggregation needs 8 fields from result; threading them individually is noisy. Extractor + AE stay in the caller so each transform stays visible at the call site.
- **Reuse existing primitives** — `extractor.forward`, `ae.encode`, `ae.per_point_decode`, `compute_similarity` already cover the rest of the pipeline. No new wrapper classes or functions.
- **Storage policy unchanged** — zarr already persists `images`; expose them on load.
- **Fail loud** — runtime asserts at function entry catch mismatched / missing inputs with clear messages.

## Design

### 1. `lift_features` — `pointcloud/utils.py`

```python
def lift_features(
    feature_maps: "list[torch.Tensor]",
    result: "FeedforwardResult",
    *,
    depth_tol: float = 0.05,
) -> torch.Tensor:
    """Multi-view confidence-weighted lift of dense feature maps to per-point features.

    For each 3D point: reproject into every frame, mask by in-bounds +
    depth-consistency, weight by depth_conf, return weighted-mean feature.
    Points never visible fall back to the source-frame sample at pixel_indices.

    feature_maps: list of (D, H_p, W_p) per-frame dense features. Caller is
        responsible for running the extractor and optionally AE-encoding before
        calling — keeps the pipeline linear at the call site.
    result: FeedforwardResult with pts3d, pixel_indices, depth, conf, extrinsics,
        intrinsics, model_height, model_width populated. Reload zarr with
        load_images=True if features will be re-extracted.
    depth_tol: relative depth-consistency tolerance for visibility test.
        |z_proj - depth[v, u]| / |z_proj| < depth_tol marks point as visible
        in that frame.

    Returns (P, D) float32 — one feature per pts3d row.
    """
```

**Entry asserts** — all required fields must be present:
```python
for name in ("pts3d", "pixel_indices", "depth", "conf", "extrinsics", "intrinsics"):
    assert getattr(result, name) is not None, (
        f"lift_features requires result.{name}; "
        f"load zarr with load_images=True or run pipeline fresh"
    )
assert len(feature_maps) == result.extrinsics.shape[0], (
    f"feature_maps count ({len(feature_maps)}) != frame count ({result.extrinsics.shape[0]})"
)
```

**Aggregation kernel** (vectorized per-frame, single Python loop over N frames):
```python
H, W = result.model_height, result.model_width
P, D = result.pts3d.shape[0], feature_maps[0].shape[0]
features_sum = torch.zeros((P, D), dtype=torch.float32)
weights_sum = torch.zeros((P,), dtype=torch.float32)

# Homogeneous points for batched projection
pts_h = np.concatenate([result.pts3d, np.ones((P, 1), dtype=np.float32)], axis=-1)  # (P, 4)

for i, fmap in enumerate(feature_maps):
    # Project all points into frame i using world-to-cam @ K
    cam = (result.extrinsics[i] @ pts_h.T)[:3]                # (3, P)
    proj = result.intrinsics[i] @ cam                          # (3, P)
    z = proj[2]
    u = proj[0] / z
    v = proj[1] / z

    # Visibility: in-bounds + positive z + depth-consistent
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    u_idx = np.clip(u.astype(np.int32), 0, W - 1)
    v_idx = np.clip(v.astype(np.int32), 0, H - 1)
    z_depth = result.depth[i, v_idx, u_idx]
    depth_ok = np.abs(z - z_depth) / (np.abs(z) + 1e-8) < depth_tol
    mask = in_bounds & depth_ok                                # (P,) bool

    # Conf weights at projected pixel; zero for invisible points
    conf_np = result.conf[i] if isinstance(result.conf, np.ndarray) else result.conf[i].numpy()
    w = torch.from_numpy(conf_np[v_idx, u_idx].astype(np.float32))
    w = w * torch.from_numpy(mask.astype(np.float32))

    # Bilinear sample feature map at normalized coords; works for any (H_p, W_p)
    gx = torch.from_numpy(((2 * u + 1) / W - 1).astype(np.float32))
    gy = torch.from_numpy(((2 * v + 1) / H - 1).astype(np.float32))
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        fmap.unsqueeze(0).float(), grid,
        mode="bilinear", align_corners=False, padding_mode="border",
    ).squeeze(0).squeeze(1).T.cpu()                            # (P, D)

    features_sum += sampled * w.unsqueeze(-1)
    weights_sum += w

# Weighted mean; eps for numerical safety
features = features_sum / (weights_sum.unsqueeze(-1) + 1e-8)

# Fallback: source-frame sample for points with zero accumulated weight
zero_w = weights_sum < 1e-6
if zero_w.any():
    features[zero_w] = _sample_at_source_pixels(
        feature_maps, result.pixel_indices[zero_w.numpy()], (H, W),
    )
return features
```

**Helper `_sample_at_source_pixels`** — module-private fallback. Direct grid_sample at `pixel_indices` for points with zero aggregated weight. Reuses the same normalization logic.

### 2. Opt-in image loading — `pointcloud/feedforward/base.py`

```python
@classmethod
def load_zarr(cls, path: Path, load_images: bool = False) -> "FeedforwardResult":
    """Load from zarr v3 store. Set load_images=True to restore the (N, 3, H, W) tensor."""
    # ... existing body ...
    images = (
        torch.from_numpy(store["images"][:])
        if load_images and "images" in store
        else None
    )
    # ... pass images=images instead of images=None ...
```

Default `False` preserves current behavior. Opt-in `True` for post-load lifting.

### 3. Delete broken creator integration

- `vggt_omega.py:217-221` — drop the `if self.extractor_name: lift_features(...)` block. Always `features=None`.
- `vggtx.py:295-299` — same.
- `mapanything.py` — analogous block if present.
- `BaseFeedforwardCreator.extractor_name` field — delete entirely. Creators no longer know about extractors.

### 4. Delete dead `_extract_and_scatter_features`

`pointcloud/utils.py:746-772`. Zero callers in the codebase. Its `pixel_indices // patch_size` approach was incorrect for non-uniform aspect ratios.

### 5. Notebook fix — `semantic_lifting.ipynb`

Two changes per backend section (MaskCLIP ~288, Talk2DINO ~1450):

```python
# Before
imgs = [Image.open(p).convert("RGB") for p in out.image_paths]
image_size = (imgs[0].height, imgs[0].width)
mc_maps = [F.normalize(fm, dim=0) for fm in maskclip.forward(imgs)]
...
lifted = lift_features(compressed, out.pixel_indices, image_size)
decoded = ae.per_point_decode(lifted)

# After
out = FeedforwardResult.load_zarr(cache_path, load_images=True)
mc_maps = [F.normalize(fm, dim=0) for fm in maskclip.forward(out.images)]
compressed = [ae.encode(fm.to(DEVICE)).detach().cpu() for fm in mc_maps]
lifted = lift_features(compressed, out)                # multi-view aggregated
decoded = ae.per_point_decode(lifted.to(DEVICE))
```

Each step linear and visible. AE encode happens before lift; lift does aggregation; AE decode happens after for query.

Requires `extractor.forward` to accept `(N, 3, H, W)` tensor. Already documented as the contract on `BaseFeatureExtractor.forward(images: list)` — the existing list of PIL is one valid form; tensor is the other. If a backend rejects tensor, add type-dispatch in that extractor's `forward` (small change). Out of scope unless verification surfaces a blocker.

### 6. Test updates — `tests/pointcloud/test_feature_lifting.py`

- Remove `test_base_creator_has_extractor_name` (field deleted).
- Update `lift_features(feature_maps, pixel_indices, image_size=(h, w))` call sites — switch to `lift_features(feature_maps, result)` with a built `FeedforwardResult` containing required fields.
- Add `test_lift_features_aggregates_across_frames` — construct 2 frames where same point projects into both with high conf in both; assert output feature is the weighted mean (not source-only).
- Add `test_lift_features_falls_back_to_source_for_unseen` — point only visible in source frame (occluded elsewhere via depth-tol mismatch); assert output equals source-frame sample.
- Add `test_lift_features_asserts_missing_fields` — `FeedforwardResult` with `depth=None`; expect `AssertionError`.
- Add `test_load_zarr_load_images_flag` — round-trip with images; assert `load_zarr(load_images=True).images is not None` and `load_zarr().images is None`.

## Validation

**Manual:** run updated notebook end-to-end on a small scene. Visualize per-point cosine similarity to a text query ("chair"). Pre-fix: features spatially diffuse. Post-fix: high-similarity points form coherent clusters on relevant surfaces. Compare single-source vs aggregated by toggling: aggregated should be visibly smoother on co-visible regions.

**Automated regression:** tests above + integration test asserting adjacent 3D points have higher mean cosine similarity in feature space than random pairs.

## Out of Scope

- New `query_features` / `extract_dense_features` wrapper functions — `compute_similarity` + `extractor.forward` already exist.
- AE training helpers — `FeatureAutoencoder.fit` already exists; notebook inlines the flatten-and-fit in one line.
- Extractor `forward(tensor)` dispatch — defer unless verification confirms a backend blocks it.
- Persisting `images` in `.npz` — zarr remains the recommended backend for semantics workflows.
- Per-point depth-cone weighting (e.g. angle-based) — start with conf-weight; add later if quality demands.

## Coding Principles Conformance

- Imports at top of every modified file.
- Inline block comments on each logical chunk in the aggregation kernel.
- One-line docstrings except `lift_features` which earns multi-line for Args/Returns.
- No new abstractions, classes, or wrappers — one rewritten function, one new kwarg, deletions.
- Hard imports preserved.
- Tests flat (no class-based tests).
- Conventional commits: `fix(pointcloud):` for utils + creator deletions, `feat(pointcloud):` for the aggregation + `load_images` flag, `fix(semantics):` for the notebook.

## Change Summary

| Change | File | Type | LOC |
|---|---|---|---|
| Rewrite `lift_features` (result-based, multi-view, conf-weighted) | `pointcloud/utils.py` | rewrite | ~70 |
| Delete `_extract_and_scatter_features` | `pointcloud/utils.py` | delete | -30 |
| Add `load_images` param to `load_zarr` | `pointcloud/feedforward/base.py` | modify | +3 |
| Delete `extractor_name` field | `pointcloud/feedforward/base.py` | delete | -3 |
| Delete broken `lift_features` call | `pointcloud/feedforward/vggt_omega.py` | delete | -5 |
| Delete broken `lift_features` call | `pointcloud/feedforward/vggtx.py` | delete | -5 |
| Delete broken `lift_features` call (if present) | `pointcloud/feedforward/mapanything.py` | delete | -5 |
| Switch to `out.images` + `load_images=True` + new lift signature | `docs/.../semantic_lifting.ipynb` | modify | ~6 |
| Replace single-source tests with aggregation + assert + load_images tests | `tests/pointcloud/test_feature_lifting.py` | modify | +40, -20 |

Net: ≈ +50 LOC (mostly the aggregation kernel). Zero new functions exposed (one module-private helper `_sample_at_source_pixels`).
