# Feature Lifting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach a feature vector `(P, D)` to every point in `pts3d (P, 3)` produced by VGGT-X, using any registered `BaseFeatureExtractor`, with the vector sourced from the pixel that produced each 3D point.

**Architecture:** `np.where(conf_mask)` gives the source `(frame, row, col)` for every point in `pts3d` — free and exact. These are stored as `pixel_indices (P, 3) int32` in `FeedforwardResult`. Two new utilities go in `pointcloud/utils.py`: `lift_features` scatters patch-level extractor output into a `(P, D)` array using `pixel_indices`; `reproject_pixels` reprojects those same pixels with updated camera poses post-BA, avoiding the stochastic point-set problem from `randomly_limit_trues`.

**Tech Stack:** numpy, torch, PIL, `BaseFeatureExtractor` registry (`collab_splats/semantics/features.py`), existing `unproject_and_filter_points` in `vggtx.py`.

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/feedforward/base.py` | Add `features`, `pixel_indices` to `FeedforwardResult`; add `extractor_name` to `BaseFeedforwardCreator` |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Extend `unproject_and_filter_points` → also returns `pixel_indices`; update `_postprocess` |
| `collab_splats/pointcloud/utils.py` | Add `lift_features`, `_assign_frame_features`, `reproject_pixels` |
| `tests/pointcloud/test_feature_lifting.py` | New test file — all tests for this feature |

No new source files. `_reproject_after_ba` is **not** modified — post-BA reprojection uses `reproject_pixels` directly in the BA wrapper.

---

### Task 1: Extend `FeedforwardResult` and `BaseFeedforwardCreator`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`
- Test: `tests/pointcloud/test_feature_lifting.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pointcloud/test_feature_lifting.py`:

```python
import dataclasses
import numpy as np
import pytest
from collab_splats.pointcloud.feedforward.base import FeedforwardResult, BaseFeedforwardCreator


def test_feedforward_result_has_new_fields():
    fields = {f.name for f in dataclasses.fields(FeedforwardResult)}
    assert "features" in fields
    assert "pixel_indices" in fields


def test_feedforward_result_new_fields_default_none():
    field_map = {f.name: f for f in dataclasses.fields(FeedforwardResult)}
    assert field_map["features"].default is None
    assert field_map["pixel_indices"].default is None


def test_base_creator_has_extractor_name():
    fields = {f.name for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert "extractor_name" in fields
    field_map = {f.name: f for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert field_map["extractor_name"].default is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_feature_lifting.py -v
```

Expected: `AttributeError` or `AssertionError` — fields don't exist yet.

- [ ] **Step 3: Add fields to `FeedforwardResult` in `base.py`**

After the existing optional fields `images`, `conf`, `world_points`, add:

```python
    features: "np.ndarray | None" = None      # (P, D) float32 — feature vector per point, index-aligned with pts3d
    pixel_indices: "np.ndarray | None" = None  # (P, 3) int32 — [frame_id, row, col] source pixel for each point
```

- [ ] **Step 4: Add `extractor_name` to `BaseFeedforwardCreator` in `base.py`**

With the other user-facing config fields (e.g. `camera_model`), add:

```python
    extractor_name: str | None = None  # registered BaseFeatureExtractor name, e.g. "dinov2"; None = skip
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_feature_lifting.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_feature_lifting.py
git commit -m "feat(pointcloud): add features/pixel_indices to FeedforwardResult and extractor_name to BaseFeedforwardCreator"
```

---

### Task 2: Extend `unproject_and_filter_points` to Return `pixel_indices`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`
- Test: `tests/pointcloud/test_feature_lifting.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_feature_lifting.py`:

```python
import torch
from collab_splats.pointcloud.feedforward.vggtx import unproject_and_filter_points


def _make_depth_inputs(n=3, h=8, w=8):
    """Minimal valid inputs for unproject_and_filter_points."""
    depth = np.ones((n, h, w), dtype=np.float32)
    depth_conf = np.random.rand(n, h, w).astype(np.float32)
    images = torch.zeros(n, 3, h, w)
    extrinsic = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    intrinsic = np.tile(
        np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    return depth, depth_conf, images, extrinsic, intrinsic


def test_unproject_returns_pixel_indices():
    depth, depth_conf, images, extrinsic, intrinsic = _make_depth_inputs(n=3, h=8, w=8)
    pts3d, colors, pixel_indices = unproject_and_filter_points(
        depth, depth_conf, images, extrinsic, intrinsic, conf_threshold=0.5
    )
    assert pixel_indices.shape == (len(pts3d), 3)
    assert pixel_indices.dtype == np.int32
    assert (pixel_indices[:, 0] >= 0).all() and (pixel_indices[:, 0] < 3).all()
    assert (pixel_indices[:, 1] >= 0).all() and (pixel_indices[:, 1] < 8).all()
    assert (pixel_indices[:, 2] >= 0).all() and (pixel_indices[:, 2] < 8).all()


def test_pixel_indices_align_with_colors():
    """colors[p] must come from the same pixel as pixel_indices[p]."""
    n, h, w = 2, 8, 8
    depth = np.ones((n, h, w), dtype=np.float32)
    depth_conf = np.ones((n, h, w), dtype=np.float32)  # all pixels pass
    # Encode pixel identity into image: pixel (r, c) = r*10 + c across all channels
    images = torch.zeros(n, 3, h, w)
    for r in range(h):
        for c in range(w):
            images[:, :, r, c] = (r * 10 + c) / 255.0
    extrinsic = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    intrinsic = np.tile(
        np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    pts3d, colors, pixel_indices = unproject_and_filter_points(
        depth, depth_conf, images, extrinsic, intrinsic, conf_threshold=0.0
    )
    images_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    for p in range(min(20, len(pts3d))):
        fi, ri, ci = pixel_indices[p]
        np.testing.assert_array_equal(
            colors[p], images_np[fi, ri, ci],
            err_msg=f"Point {p}: color mismatch at frame={fi} row={ri} col={ci}",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_feature_lifting.py::test_unproject_returns_pixel_indices tests/pointcloud/test_feature_lifting.py::test_pixel_indices_align_with_colors -v
```

Expected: `ValueError: not enough values to unpack` — function returns 2 values, not 3.

- [ ] **Step 3: Extend `unproject_and_filter_points` in `vggtx.py`**

The function currently ends with:

```python
    pts_out = points3d[conf_mask].astype(np.float32)
    colors_out = (colors_np[conf_mask] * 255).astype(np.uint8)

    return pts_out, colors_out
```

Replace with:

```python
    pts_out = points3d[conf_mask].astype(np.float32)
    colors_out = (colors_np[conf_mask] * 255).astype(np.uint8)
    # np.where on the final conf_mask (after randomly_limit_trues applied) gives the
    # exact (frame_id, row, col) that produced each surviving point.
    pixel_indices = np.stack(np.where(conf_mask), axis=1).astype(np.int32)  # (P, 3)

    return pts_out, colors_out, pixel_indices
```

Update the return type annotation from `tuple[np.ndarray, np.ndarray]` to `tuple[np.ndarray, np.ndarray, np.ndarray]`.

Update the docstring Returns section:

```
    Returns:
        pts3d:         (P, 3) float32 world-space points.
        colors:        (P, 3) uint8 RGB.
        pixel_indices: (P, 3) int32 — [frame_id, row, col] source pixel for each point.
```

- [ ] **Step 4: Fix the two call sites in `vggtx.py`**

Both currently unpack 2 values. Update to 3:

**In `_postprocess`:**
```python
        pts3d, colors, pixel_indices = unproject_and_filter_points(...)
```

**In `_reproject_after_ba`:**
```python
        pts3d, colors, _pixel_indices = unproject_and_filter_points(...)
        return pts3d, colors  # pixel_indices unused here; post-BA uses stored indices
```

- [ ] **Step 5: Pass `pixel_indices` into `FeedforwardResult` in `_postprocess`**

In the `FeedforwardResult(...)` constructor at the end of `_postprocess`, add:

```python
            pixel_indices=pixel_indices,
```

- [ ] **Step 6: Run all tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_feature_lifting.py tests/pointcloud/test_feedforward_shared.py tests/pointcloud/test_vggtx_creator.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_feature_lifting.py
git commit -m "feat(pointcloud): extend unproject_and_filter_points to return pixel_indices"
```

---

### Task 3: Add `lift_features` and `reproject_pixels` to `pointcloud/utils.py`

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`
- Test: `tests/pointcloud/test_feature_lifting.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_feature_lifting.py`:

```python
from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.pointcloud.utils import lift_features, reproject_pixels


@BaseFeatureExtractor.register("_test_lift_extractor")
class _TestLiftExtractor(BaseFeatureExtractor):
    """Returns fixed 4-dim features on a 2×2 patch grid for any image."""
    def forward(self, images: list):
        # arange(1,5) → [1,2,3,4]; reshape to (4,1,1) then expand to (4,2,2)
        return [torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1).expand(4, 2, 2)]


def test_lift_features_shape():
    N, H, W = 2, 8, 8
    images = torch.zeros(N, 3, H, W)
    pixel_indices = np.array([[0, 0, 0], [0, 4, 4], [1, 2, 6]], dtype=np.int32)
    feats = lift_features(images, pixel_indices, extractor_name="_test_lift_extractor", device="cpu")
    assert feats.shape == (3, 4)
    assert feats.dtype == np.float32


def test_lift_features_values():
    """All points should get [1,2,3,4] since _TestLiftExtractor always returns that."""
    N, H, W = 2, 8, 8
    images = torch.zeros(N, 3, H, W)
    pixel_indices = np.array([[0, 0, 0], [1, 7, 7]], dtype=np.int32)
    feats = lift_features(images, pixel_indices, extractor_name="_test_lift_extractor", device="cpu")
    np.testing.assert_allclose(feats[0], [1, 2, 3, 4])
    np.testing.assert_allclose(feats[1], [1, 2, 3, 4])


def test_lift_features_empty_frame_skipped():
    """Points from frame 0 only — frame 1 has no points, must not crash."""
    N, H, W = 2, 8, 8
    images = torch.zeros(N, 3, H, W)
    pixel_indices = np.array([[0, 0, 0], [0, 3, 3]], dtype=np.int32)
    feats = lift_features(images, pixel_indices, extractor_name="_test_lift_extractor", device="cpu")
    assert feats.shape == (2, 4)


def test_reproject_pixels_shape():
    N, H, W = 3, 8, 8
    depth = np.ones((N, H, W), dtype=np.float32)
    pixel_indices = np.array([[0, 2, 3], [1, 5, 6], [2, 0, 1]], dtype=np.int32)
    extrinsics_3x4 = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (N, 1, 1)
    )
    intrinsics = np.tile(
        np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )
    pts3d = reproject_pixels(depth, pixel_indices, extrinsics_3x4, intrinsics)
    assert pts3d.shape == (3, 3)
    assert pts3d.dtype == np.float32


def test_reproject_pixels_identity_camera():
    """With identity extrinsics, reprojected points match direct depth unprojection."""
    N, H, W = 1, 8, 8
    depth = np.full((N, H, W), 2.0, dtype=np.float32)  # constant depth = 2
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)  # center pixel
    extrinsics_3x4 = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (N, 1, 1)
    )
    # fx=fy=1, cx=cy=4 → (col-4)*z/1 = 0 at center pixel
    intrinsics = np.tile(
        np.array([[1, 0, 4], [0, 1, 4], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )
    pts3d = reproject_pixels(depth, pixel_indices, extrinsics_3x4, intrinsics)
    # center pixel with unit focal, identity camera → world point = (0, 0, 2)
    np.testing.assert_allclose(pts3d[0], [0, 0, 2], atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_feature_lifting.py::test_lift_features_shape tests/pointcloud/test_feature_lifting.py::test_reproject_pixels_shape -v
```

Expected: `ImportError: cannot import name 'lift_features'`.

- [ ] **Step 3: Add the three functions to `pointcloud/utils.py`**

Add at the end of `collab_splats/pointcloud/utils.py`. The import for `_extrinsics_3x4_to_4x4` is already available via `from .feedforward.base import ...` — add it if not present, or inline the logic.

```python
def _assign_frame_features(
    feats_out: np.ndarray,       # (P, D) output buffer — written in-place for this frame's points
    feat_i: "torch.Tensor",      # (D, H_p, W_p) patch features extracted for frame i
    pixel_indices: np.ndarray,   # (P, 3) int32 — [frame_id, row, col] for every point
    frame: int,                  # which frame to process in this call
    stride_h: int,               # H // H_p — how many image rows per patch row
    stride_w: int,               # W // W_p — how many image cols per patch col
    H_p: int,                    # number of patch rows in feat_i
    W_p: int,                    # number of patch cols in feat_i
) -> None:
    """Scatter patch features for points sourced from `frame` into feats_out."""
    # Boolean mask: which of the P points came from this frame
    mask_i = pixel_indices[:, 0] == frame            # (P,) bool

    if not mask_i.any():
        return  # no points from this frame — nothing to write

    # Map each surviving pixel's row/col to its patch grid position via integer division.
    # Clip to [0, H_p-1] / [0, W_p-1] to guard against off-by-one at image edges.
    pr = (pixel_indices[mask_i, 1] // stride_h).clip(0, H_p - 1)  # (K,) patch row indices
    pc = (pixel_indices[mask_i, 2] // stride_w).clip(0, W_p - 1)  # (K,) patch col indices

    # feat_i[:, pr, pc] → (D, K); transpose to (K, D) then write into the output buffer
    feats_out[mask_i] = feat_i[:, pr, pc].T.cpu().float().numpy()


def lift_features(
    images: "torch.Tensor",       # (N, 3, H, W) float32 in [0,1] — from FeedforwardResult.images
    pixel_indices: np.ndarray,    # (P, 3) int32 — [frame_id, row, col] from FeedforwardResult.pixel_indices
    extractor_name: str = "dinov2",  # name of a registered BaseFeatureExtractor
    device: str = "cuda",         # torch device for the extractor model
) -> np.ndarray:                  # (P, D) float32 — one feature vector per point
    """Assign a feature vector to every point using its source pixel.

    Processes one frame at a time to keep memory bounded. Peak usage per iteration
    is one frame's patch features (e.g. 37×37×384 ≈ 2 MB for DINOv2-small).
    """
    import torch
    from PIL import Image as PILImage
    from collab_splats.semantics.features import BaseFeatureExtractor

    # Instantiate and move extractor to device
    extractor = BaseFeatureExtractor.get(extractor_name)()
    extractor.eval()
    if hasattr(extractor, "to"):
        extractor = extractor.to(device)

    N, _, H, W = images.shape  # N frames, image spatial dims H×W
    P = len(pixel_indices)      # total number of 3D points

    def _tensor_to_pil(t: "torch.Tensor") -> PILImage.Image:
        # Convert (3, H, W) float [0,1] tensor → uint8 PIL image for the extractor
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return PILImage.fromarray(arr)

    # Run frame 0 first to learn the patch grid dimensions and feature depth D
    with torch.no_grad():
        feat0 = extractor([_tensor_to_pil(images[0])])[0]  # (D, H_p, W_p)
    D, H_p, W_p = feat0.shape
    stride_h = max(1, H // H_p)  # image rows per patch row
    stride_w = max(1, W // W_p)  # image cols per patch col

    # Pre-allocate output; zero-initialised (points with no frame hit stay zero)
    feats_out = np.zeros((P, D), dtype=np.float32)

    # Write features for frame 0 (already computed above)
    _assign_frame_features(feats_out, feat0, pixel_indices, 0, stride_h, stride_w, H_p, W_p)

    # Process remaining frames sequentially — only one frame's features in memory at a time
    for i in range(1, N):
        with torch.no_grad():
            feat_i = extractor([_tensor_to_pil(images[i])])[0]  # (D, H_p, W_p)
        _assign_frame_features(feats_out, feat_i, pixel_indices, i, stride_h, stride_w, H_p, W_p)

    return feats_out


def reproject_pixels(
    depth: np.ndarray,            # (N, H, W) float32 — raw depth maps (unchanged by BA)
    pixel_indices: np.ndarray,    # (P, 3) int32 — [frame_id, row, col] stored at initial unprojection
    extrinsics_3x4: np.ndarray,   # (N, 3, 4) float32 — refined world-to-camera [R|t] after BA
    intrinsics: np.ndarray,       # (N, 3, 3) float32 — refined camera intrinsics K after BA
) -> np.ndarray:                  # (P, 3) float32 — updated world-space XYZ
    """Reproject the exact stored pixel set to world-space under new camera parameters.

    Used post-BA instead of re-running unproject_and_filter_points, which would apply
    randomly_limit_trues again and produce a different (stochastic) point subset.
    By reusing pixel_indices, the same P points are preserved with updated positions.
    """
    from collab_splats.pointcloud.feedforward.base import _extrinsics_3x4_to_4x4

    fi = pixel_indices[:, 0]  # (P,) frame index for each point
    ri = pixel_indices[:, 1]  # (P,) pixel row for each point
    ci = pixel_indices[:, 2]  # (P,) pixel col for each point

    # Look up depth at each point's source pixel using advanced indexing
    z = depth[fi, ri, ci]  # (P,) depth values

    # Per-point intrinsics: select each frame's focal lengths and principal point
    fx = intrinsics[fi, 0, 0]  # (P,) focal length x
    fy = intrinsics[fi, 1, 1]  # (P,) focal length y
    cx = intrinsics[fi, 0, 2]  # (P,) principal point x
    cy = intrinsics[fi, 1, 2]  # (P,) principal point y

    # Unproject pixel (col, row) at depth z into camera-space 3D coords
    x_cam = (ci - cx) * z / fx  # (P,) camera-space x
    y_cam = (ri - cy) * z / fy  # (P,) camera-space y

    # Homogeneous camera-space coords: [x_cam, y_cam, z, 1] for each point
    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)  # (P, 4)

    # Invert world-to-camera extrinsics to get camera-to-world transforms (N, 4, 4)
    cam2world = np.linalg.inv(_extrinsics_3x4_to_4x4(extrinsics_3x4))  # (N, 4, 4)

    # For each point, apply its frame's cam2world matrix: cam2world[fi[p]] @ pts_cam[p]
    # cam2world[fi] → (P, 4, 4); einsum over matrix-vector products → (P, 4)
    pts_world = np.einsum("pij,pj->pi", cam2world[fi], pts_cam)  # (P, 4) homogeneous
    pts3d = pts_world[:, :3]  # drop homogeneous coordinate → (P, 3)

    return pts3d.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_feature_lifting.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/utils.py tests/pointcloud/test_feature_lifting.py
git commit -m "feat(pointcloud): add lift_features and reproject_pixels to pointcloud/utils.py"
```

---

### Task 4: Wire `lift_features` into `VGGTXCreator._postprocess`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`
- Test: `tests/pointcloud/test_feature_lifting.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_feature_lifting.py`:

```python
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator


def _make_raw_outputs(n=2, h=8, w=8):
    """Minimal raw_outputs dict that VGGTXCreator._postprocess accepts."""
    depth = np.ones((n, h, w), dtype=np.float32)
    depth_conf = np.ones((n, h, w), dtype=np.float32)
    images = torch.zeros(n, 3, h, w)
    extrinsic = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    intrinsic = np.tile(
        np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    return {
        "images": images, "extrinsic": extrinsic,
        "intrinsics": intrinsic, "intrinsics_downsampled": intrinsic,
        "depth": depth, "depth_conf": depth_conf,
    }


def _make_creator(tmp_path, n=2, extractor_name=None):
    creator = VGGTXCreator(extractor_name=extractor_name)
    creator.image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(n)]
    creator.original_coords = np.tile(
        np.array([0, 0, 8, 8, 8, 8], dtype=np.float32), (n, 1)
    )
    return creator


def test_postprocess_populates_features_when_extractor_set(tmp_path):
    raw = _make_raw_outputs()
    creator = _make_creator(tmp_path, extractor_name="_test_lift_extractor")
    result = creator._postprocess(raw)
    assert result.features is not None
    assert result.features.shape == (len(result.pts3d), 4)  # _TestLiftExtractor → D=4
    assert result.pixel_indices is not None
    assert result.pixel_indices.shape == (len(result.pts3d), 3)


def test_postprocess_skips_features_when_no_extractor(tmp_path):
    raw = _make_raw_outputs()
    creator = _make_creator(tmp_path)  # extractor_name=None
    result = creator._postprocess(raw)
    assert result.features is None
    assert result.pixel_indices is not None  # always populated
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_feature_lifting.py::test_postprocess_populates_features_when_extractor_set tests/pointcloud/test_feature_lifting.py::test_postprocess_skips_features_when_no_extractor -v
```

Expected: `AssertionError: result.features is not None` — not wired yet.

- [ ] **Step 3: Wire into `_postprocess` in `vggtx.py`**

After the `unproject_and_filter_points` call (now returning `pixel_indices`), add:

```python
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            conf_threshold=self.conf_threshold,
        )

        # Lift 2D features to 3D if an extractor is configured; None otherwise
        features = None
        if self.extractor_name is not None:
            from collab_splats.pointcloud.utils import lift_features
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            features = lift_features(
                raw_outputs["images"], pixel_indices, self.extractor_name, device
            )
```

In the `FeedforwardResult(...)` constructor at the end of `_postprocess`, add:

```python
            features=features,
            pixel_indices=pixel_indices,
```

- [ ] **Step 4: Run all tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_feature_lifting.py
git commit -m "feat(pointcloud): wire lift_features into VGGTXCreator._postprocess"
```

---

### Task 5: Wire `reproject_pixels` into the Bundle Adjustment Wrapper

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py` (find the exact call site — see Step 1)
- Test: `tests/pointcloud/test_feature_lifting.py`

**Context:** After BA refines poses, the BA wrapper calls `_reproject_after_ba` to get updated `pts3d`. We now also call `reproject_pixels` with the stored `pixel_indices` to get a deterministic reprojection of the exact same point set, and re-call `lift_features` if an extractor is set.

- [ ] **Step 1: Find the BA wrapper call site**

```bash
grep -n "_reproject_after_ba\|FeedforwardResult\|\.pts3d\s*=" /workspace/collab-splats/collab_splats/pointcloud/bundle_adjustment.py | head -30
```

Note the line numbers — this tells you exactly where to insert the `reproject_pixels` + `lift_features` calls.

- [ ] **Step 2: Write the failing test**

Append to `tests/pointcloud/test_feature_lifting.py`:

```python
def test_reproject_pixels_used_post_ba(tmp_path):
    """After _reproject_after_ba, result.features must be non-None when extractor set."""
    from collab_splats.pointcloud.utils import reproject_pixels, lift_features

    n, h, w = 2, 8, 8
    depth = np.ones((n, h, w), dtype=np.float32)
    pixel_indices = np.array([[0, 2, 3], [1, 5, 6]], dtype=np.int32)
    extrinsics_3x4 = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    intrinsics = np.tile(
        np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    # Verify reproject_pixels produces same P as stored pixel_indices
    pts3d_new = reproject_pixels(depth, pixel_indices, extrinsics_3x4, intrinsics)
    assert pts3d_new.shape == (2, 3)

    # Verify lift_features works on reprojected set
    images = torch.zeros(n, 3, h, w)
    feats = lift_features(images, pixel_indices, "_test_lift_extractor", device="cpu")
    assert feats.shape == (2, 4)
```

- [ ] **Step 3: Run test to verify it passes already**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_feature_lifting.py::test_reproject_pixels_used_post_ba -v
```

Expected: PASS (utilities already exist from Task 3).

- [ ] **Step 4: Update the BA wrapper call site**

At the point in `bundle_adjustment.py` where `_reproject_after_ba` is called and `result.pts3d` / `result.colors` are updated, add the `reproject_pixels` + `lift_features` calls:

```python
        # Re-derive pts3d using BA-refined poses. Uses stored pixel_indices to
        # reproject the exact same point set deterministically (avoids randomly_limit_trues).
        if result.pixel_indices is not None:
            from collab_splats.pointcloud.utils import reproject_pixels, lift_features
            result.pts3d = reproject_pixels(
                creator.raw_outputs["depth"],
                result.pixel_indices,
                refined_extrinsics_3x4,   # use the variable name from the BA wrapper
                refined_intrinsics,        # use the variable name from the BA wrapper
            )
            # Re-lift features with same pixel_indices under new geometry
            if creator.extractor_name is not None:
                device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
                result.features = lift_features(
                    creator.raw_outputs["images"],
                    result.pixel_indices,
                    creator.extractor_name,
                    device,
                )
        else:
            # Fallback: no pixel_indices stored (older result), use abstract method
            result.pts3d, result.colors = creator._reproject_after_ba(
                creator.raw_outputs, refined_extrinsics_3x4, refined_intrinsics
            )
```

Replace `refined_extrinsics_3x4` and `refined_intrinsics` with the actual variable names found in Step 1.

- [ ] **Step 5: Run full test suite**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/ -v --tb=short
```

Expected: all tests PASS including existing BA tests.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_feature_lifting.py
git commit -m "feat(pointcloud): use reproject_pixels + lift_features post-BA for deterministic point preservation"
```

---

## Self-Review

**Spec coverage:**
- ✅ `lift_features` in `pointcloud/utils.py` — Task 3
- ✅ `features`, `pixel_indices` in `FeedforwardResult` — Task 1
- ✅ `extractor_name` on `BaseFeedforwardCreator` — Task 1
- ✅ `unproject_and_filter_points` returns `pixel_indices` — Task 2
- ✅ `_postprocess` wired — Task 4
- ✅ Post-BA features preserved via `reproject_pixels` — Task 5
- ✅ `_reproject_after_ba` NOT modified — preserved as fallback only
- ✅ No new source files

**Placeholder scan:** Step 4 of Task 5 contains variable name placeholders (`refined_extrinsics_3x4`, `refined_intrinsics`) — these are intentional and resolved by Step 1's grep output.

**Type consistency:**
- `pixel_indices: np.ndarray (P, 3) int32` — consistent across all tasks
- `features: np.ndarray (P, D) float32` — consistent across all tasks
- `lift_features(images, pixel_indices, extractor_name, device)` — signature matches definition (Task 3) and all call sites (Task 4, Task 5)
- `reproject_pixels(depth, pixel_indices, extrinsics_3x4, intrinsics)` — signature matches definition (Task 3) and call site (Task 5)
- `_TestLiftExtractor` registered as `"_test_lift_extractor"` — used consistently across Task 3, 4, 5 tests ✅
