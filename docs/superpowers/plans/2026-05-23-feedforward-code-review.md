# Feedforward Module Code Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the feedforward module (base, vggtx, mapanything, wrappers, __init__) for coding-standard compliance and fix the critical BA+MapAnything crash caused by a missing `_reproject_ba` implementation.

**Architecture:** Five targeted tasks — one per file — each self-contained. wrappers.py and base.py first (rename propagation), then vggtx/mapanything (implementations), then __init__.py, then a full inline-comment audit pass, then test run. Every structural change is TDD-first; comment-only passes commit directly.

**Tech Stack:** Python 3.11, numpy, torch, pycolmap, MapAnything (`mapanything.utils.geometry.closed_form_pose_inverse`), pytest (`/opt/conda/envs/nerfstudio/bin/python -m pytest`)

---

## File Map

| File | Changes |
|---|---|
| `collab_splats/pointcloud/feedforward/base.py` | Drop `List`/`Tuple` typing imports; rename + `@abstractmethod` `_reproject_ba`; inline comment audit |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Rename `_reproject_ba`; rename underscore locals in `_postprocess`; inline comment audit |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Implement `_reproject_ba`; fix double `PILImage.open`; declare `_processed_views` field; inline comment audit |
| `collab_splats/pointcloud/wrappers.py` | Rename `_reproject_ba` at 2 call sites |
| `collab_splats/pointcloud/feedforward/__init__.py` | Add comment on `_raw_to_world_points` re-export |
| `tests/pointcloud/test_mapanything_creator.py` | Add `test_reproject_ba_*` tests |

---

## Task 1: Rename `_reproject_after_ba` in wrappers.py

**Files:**
- Modify: `collab_splats/pointcloud/wrappers.py` (lines ~146 and ~196)

No logic change. Two sites: the `_apply_ba` call and the `LoopClosure` delegation method.

- [ ] **Step 1: Apply rename at both sites**

In `collab_splats/pointcloud/wrappers.py`, make these two changes:

Site 1 — `BundleAdjustment._apply_ba` (around line 146):
```python
# Before:
pts3d, colors = self.base._reproject_after_ba(
    self.base.raw_outputs, refined_ext_3x4, refined_intr
)
# After:
pts3d, colors = self.base._reproject_ba(
    self.base.raw_outputs, refined_ext_3x4, refined_intr
)
```

Site 2 — `LoopClosure._reproject_after_ba` delegation (around line 196):
```python
# Before:
def _reproject_after_ba(self, raw_outputs: Any, ext: Any, intr: Any) -> Any:
    return self.base._reproject_after_ba(raw_outputs, ext, intr)
# After:
def _reproject_ba(self, raw_outputs: Any, ext: Any, intr: Any) -> Any:
    return self.base._reproject_ba(raw_outputs, ext, intr)
```

- [ ] **Step 2: Run smoke test to verify no import errors**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.pointcloud.wrappers import BundleAdjustment, LoopClosure; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py
git commit -m "refactor(wrappers): rename _reproject_after_ba → _reproject_ba"
```

---

## Task 2: base.py — typing cleanup + promote `_reproject_ba` to `@abstractmethod`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`

- [ ] **Step 1: Fix typing imports**

In `collab_splats/pointcloud/feedforward/base.py`, update the typing import line:

```python
# Before:
from typing import Any, List, Tuple

# After:
from typing import Any
```

Then update `_rescale_reconstruction_to_original_dimensions` signature (around line 354):

```python
# Before:
def _rescale_reconstruction_to_original_dimensions(
    reconstruction: Any,
    image_paths: List[Path],
    original_image_sizes: np.ndarray,
    image_size: Tuple[int, int],
    shared_camera: bool = False,
    shift_point2d_to_original_res: bool = False,
    verbose: bool = False,
) -> Any:

# After:
def _rescale_reconstruction_to_original_dimensions(
    reconstruction: Any,
    image_paths: list[Path],
    original_image_sizes: np.ndarray,
    image_size: tuple[int, int],
    shared_camera: bool = False,
    shift_point2d_to_original_res: bool = False,
    verbose: bool = False,
) -> Any:
```

- [ ] **Step 2: Rename `_reproject_after_ba` → `_reproject_ba` and promote to `@abstractmethod`**

Replace the entire `_reproject_after_ba` method (around line 655):

```python
# Before:
def _reproject_after_ba(
    self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive world-space point cloud using bundle-adjusted camera poses.
    ...
    """
    raise NotImplementedError(
        f"{type(self).__name__} must implement _reproject_after_ba. "
        "BundleAdjustment wrapper cannot reproject points without this method."
    )

# After:
@abstractmethod
def _reproject_ba(
    self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive world-space point cloud using bundle-adjusted camera poses.

    Called by the BundleAdjustment wrapper after it refines extrinsics.
    Each backend re-projects its raw depth/point data under the new poses.

    Args:
        raw_outputs:    Raw model outputs stored from _forward().
        extrinsics_3x4: (N, 3, 4) refined world-to-camera matrices.
        intrinsics:     (N, 3, 3) refined camera intrinsics.

    Returns:
        (pts3d, colors) — (P, 3) float32 and (P, 3) uint8.
    """
    ...
```

- [ ] **Step 3: Verify import and class instantiation guard**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator
import inspect
methods = [m for m in dir(BaseFeedforwardCreator) if not m.startswith('__')]
abstract = BaseFeedforwardCreator.__abstractmethods__
print('abstract methods:', sorted(abstract))
assert '_reproject_ba' in abstract, '_reproject_ba not abstract'
assert '_reproject_after_ba' not in abstract, '_reproject_after_ba still present'
print('OK')
"
```
Expected output contains `_reproject_ba` in the abstract methods set.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py
git commit -m "refactor(feedforward): drop List/Tuple typing; _reproject_ba as @abstractmethod"
```

---

## Task 3: vggtx.py — rename + local variable cleanup + inline comment audit

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`

- [ ] **Step 1: Rename `_reproject_after_ba` → `_reproject_ba`**

```python
# Before (method signature ~line 316):
def _reproject_after_ba(
    self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

# After:
def _reproject_ba(
    self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
```

- [ ] **Step 2: Rename underscore locals in `_postprocess`**

In `_postprocess` (around lines 282–292), rename three locals. The underscore prefix implies private class scope but these are just temporaries:

```python
# Before:
world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
if world_pts_flat is not None:
    _world_points = world_pts_flat.reshape(
        world_pts_flat.shape[0], model_h, model_w, 3
    )
else:
    _world_points = None
_conf = torch.from_numpy(raw_outputs["depth_conf"])
_images = raw_outputs["images"]

# After:
world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
if world_pts_flat is not None:
    world_points = world_pts_flat.reshape(
        world_pts_flat.shape[0], model_h, model_w, 3
    )
else:
    world_points = None
conf = torch.from_numpy(raw_outputs["depth_conf"])
images = raw_outputs["images"]
```

Also update the `FeedforwardResult(...)` call at the bottom of `_postprocess` to use the new names:

```python
return FeedforwardResult(
    pts3d=pts3d,
    colors=colors,
    pixel_indices=pixel_indices,
    features=features,
    extrinsics=extrinsic_4x4_out,
    intrinsics=intrinsic,
    image_paths=self.image_paths,
    original_coords=self.original_coords,
    model_width=model_w,
    model_height=model_h,
    images=images,
    conf=conf,
    world_points=world_points,
)
```

- [ ] **Step 3: Inline comment audit across all vggtx.py methods**

Check each logical block in every method has a block-level comment. Add where missing. The following blocks currently lack comments or have comments that could be clearer:

In `_load_model`:
```python
# Choose dtype based on GPU capability: bfloat16 for Ampere+, float16 for older
dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)
# Load pretrained model, set eval mode
model = VGGT.from_pretrained(self.model_name, chunk_size=self.chunk_size)
model.eval()
model = model.to(device, dtype=dtype)
return model
```

In `_preprocess`:
```python
# Collect and sort image paths; reject non-image extensions
image_paths = sorted([
    p for p in image_dir.iterdir()
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
])
if not image_paths:
    raise FileNotFoundError(f"No images found in {image_dir}")

# Load and preprocess images to model resolution; store original crop coordinates
image_names = [str(p) for p in image_paths]
if self.image_preproc == "ratio":
    images, original_coords = load_and_preprocess_images_ratio(
        image_names, VGGTX_IMG_LOAD_RESOLUTION
    )
else:
    images, original_coords = load_and_preprocess_images_square(
        image_names, VGGTX_IMG_LOAD_RESOLUTION
    )

return images, image_paths, original_coords.cpu().float().numpy()
```

In `_forward`:
```python
# Move images to model device and dtype
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
images = images.to(device, dtype=dtype)

width, height = self.original_coords[0, -2:]
image_shape = images.shape[-2:]

# Run forward pass; decode pose encoding at both model and original resolution
with torch.no_grad():
    predictions = model(images.unsqueeze(0))

extrinsic_ds, intrinsic_ds = pose_encoding_to_extri_intri(
    predictions["pose_enc"], image_shape
)
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"], [width, height]
)

# Move predictions to CPU float32 for downstream processing
extrinsic = extrinsic.cpu().float().numpy().squeeze(0)
intrinsic = intrinsic.cpu().float().numpy().squeeze(0)
intrinsic_downsampled = intrinsic_ds.cpu().float().numpy().squeeze(0)
depth_map = predictions["depth"].squeeze(0).cpu().float().numpy()
depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()

return {
    "images": images,
    "extrinsic": extrinsic,
    "intrinsics": intrinsic,
    "intrinsics_downsampled": intrinsic_downsampled,
    "depth": depth_map,
    "depth_conf": depth_conf,
}
```

In `_postprocess` (verify existing block comments are present for global alignment, feature lifting, BA field population, and LC merged extrinsics — add any that are missing).

In `_reproject_ba`:
```python
# Re-run depth unprojection with refined extrinsics and intrinsics
pts3d, colors, _pixel_indices = unproject_and_filter_points(
    depth=raw_outputs["depth"],
    depth_conf=raw_outputs["depth_conf"],
    images=raw_outputs["images"],
    extrinsic=extrinsics_3x4,
    intrinsic=intrinsics,
    conf_threshold=self.conf_threshold,
)
return pts3d, colors  # pixel_indices unused here; post-BA uses stored indices
```

In `extract_intermediate_features`:
```python
# Add batch dimension; move to model device + dtype for the forward pass
batch = frames.unsqueeze(0).to(device, dtype=dtype)

# Register a per-call hook on the QKV projection of the chosen block
block = self.model.aggregator.global_blocks[layer_index]
...

# Run forward pass; hook captures q and k
with torch.no_grad():
    predictions = self.model(batch)

# Decode (2, 4, 4) camera extrinsics from VGGT-X pose encoding
image_shape = (frames.shape[-2], frames.shape[-1])
...
```

- [ ] **Step 4: Smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
import inspect
src = inspect.getsource(VGGTXCreator._reproject_ba)
assert '_reproject_ba' in src
assert '_world_points' not in src, 'underscore local still present'
assert '_conf' not in src, 'underscore local still present'
assert '_images' not in src, 'underscore local still present'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py
git commit -m "refactor(vggtx): rename _reproject_ba, clean local vars, add inline comments"
```

---

## Task 4: mapanything.py — implement `_reproject_ba` + fixes + inline comments

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Modify: `tests/pointcloud/test_mapanything_creator.py`

- [ ] **Step 1: Write failing tests for `_reproject_ba`**

Add to `tests/pointcloud/test_mapanything_creator.py`:

```python
import numpy as np
import torch
import pytest
from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator


def _make_raw_outputs(n_frames: int = 2, h: int = 4, w: int = 4) -> list[dict]:
    """Minimal mock raw_outputs mimicking MapAnything model.forward() output."""
    return [
        {
            "pts3d_cam": torch.zeros(1, h, w, 3, dtype=torch.float32),
            "mask": torch.ones(1, h, w, 1, dtype=torch.bool),
            "depth_z": torch.ones(1, h, w, 1, dtype=torch.float32),
            "img_no_norm": torch.zeros(1, h, w, 3, dtype=torch.float32),
        }
        for _ in range(n_frames)
    ]


def test_reproject_ba_output_shapes():
    """_reproject_ba returns (P,3) float32 pts3d and (P,3) uint8 colors."""
    creator = MapAnythingCreator()
    n, h, w = 2, 4, 4
    raw_outputs = _make_raw_outputs(n, h, w)
    extrinsics_3x4 = np.tile(np.eye(4)[:3, :], (n, 1, 1)).astype(np.float32)
    intrinsics = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)

    pts3d, colors = creator._reproject_ba(raw_outputs, extrinsics_3x4, intrinsics)

    assert pts3d.shape == (n * h * w, 3), f"Expected ({n*h*w}, 3), got {pts3d.shape}"
    assert pts3d.dtype == np.float32, f"Expected float32, got {pts3d.dtype}"
    assert colors.shape == (n * h * w, 3), f"Expected ({n*h*w}, 3), got {colors.shape}"
    assert colors.dtype == np.uint8, f"Expected uint8, got {colors.dtype}"


def test_reproject_ba_depth_mask_filters_zero_depth():
    """Points with depth_z <= 0 are excluded from output."""
    creator = MapAnythingCreator()
    h, w = 4, 4
    raw_outputs = _make_raw_outputs(1, h, w)
    # Set half the pixels to depth_z = 0 → should be filtered out
    raw_outputs[0]["depth_z"][0, :h//2, :, 0] = 0.0
    extrinsics_3x4 = np.eye(4)[:3, :][np.newaxis].astype(np.float32)
    intrinsics = np.eye(3)[np.newaxis].astype(np.float32)

    pts3d, colors = creator._reproject_ba(raw_outputs, extrinsics_3x4, intrinsics)

    expected_count = h * w - (h // 2) * w  # only positive-depth pixels survive
    assert pts3d.shape[0] == expected_count, (
        f"Expected {expected_count} points after depth filter, got {pts3d.shape[0]}"
    )


def test_reproject_ba_identity_extrinsic_preserves_cam_points():
    """Identity extrinsic → world pts == pts3d_cam (no rotation/translation)."""
    creator = MapAnythingCreator()
    h, w = 2, 2
    raw_outputs = _make_raw_outputs(1, h, w)
    # Set recognizable camera-frame points
    raw_outputs[0]["pts3d_cam"][0] = torch.tensor([[1., 2., 3.], [4., 5., 6.],
                                                    [7., 8., 9.], [0., 1., 2.]]).reshape(h, w, 3)
    extrinsics_3x4 = np.eye(4)[:3, :][np.newaxis].astype(np.float32)  # identity world2cam
    intrinsics = np.eye(3)[np.newaxis].astype(np.float32)

    pts3d, _ = creator._reproject_ba(raw_outputs, extrinsics_3x4, intrinsics)

    expected = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [0., 1., 2.]], dtype=np.float32)
    np.testing.assert_allclose(pts3d, expected, atol=1e-5)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_reproject_ba_output_shapes tests/pointcloud/test_mapanything_creator.py::test_reproject_ba_depth_mask_filters_zero_depth tests/pointcloud/test_mapanything_creator.py::test_reproject_ba_identity_extrinsic_preserves_cam_points -v
```
Expected: all 3 FAIL with `NotImplementedError` (from base class fallback) or `TypeError` (abstract method).

- [ ] **Step 3: Implement `_reproject_ba` in MapAnythingCreator**

Add the following method to `MapAnythingCreator` in `collab_splats/pointcloud/feedforward/mapanything.py`, after `extract_intermediate_features`:

```python
def _reproject_ba(
    self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive world-space points using bundle-adjusted camera poses.

    MapAnything predicts pts3d_cam (camera-frame XYZ) directly — not a scalar
    depth map. After BA refines world2cam extrinsics, pts3d (world-frame) is
    stale because it has the original predicted poses baked in. pts3d_cam is
    pose-independent, so we transform it with the refined cam2world instead.

    Args:
        raw_outputs:    list[dict] from _forward(); each dict contains pts3d_cam,
                        mask, depth_z, img_no_norm (float32, unmasked).
        extrinsics_3x4: (N, 3, 4) refined world-to-camera matrices from BA.
        intrinsics:     (N, 3, 3) refined camera intrinsics (unused for point
                        reprojection; stored in FeedforwardResult by wrapper).

    Returns:
        (pts3d, colors) — (P, 3) float32 world-space points and (P, 3) uint8 RGB.
    """
    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for i, pred in enumerate(raw_outputs):
        # Extract camera-frame points and validity components
        pts3d_cam = pred["pts3d_cam"][0].cpu().numpy()          # (H, W, 3)
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)
        depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()         # (H, W)

        # Combine validity mask with positive-depth check
        combined_mask = mask & (depth_z > 0)

        # Refined world2cam → cam2world for re-projection into world frame
        ext_4x4 = np.concatenate([extrinsics_3x4[i], [[0, 0, 0, 1]]], axis=0)  # (4, 4)
        cam2world = closed_form_pose_inverse(ext_4x4[None])[0]                   # (4, 4)

        # Apply mask and transform camera-frame points to world frame
        pts_flat = pts3d_cam[combined_mask]                               # (K, 3)
        pts_world = (cam2world[:3, :3] @ pts_flat.T + cam2world[:3, 3:]).T  # (K, 3)

        # Extract colors for surviving pixels
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()                 # (H, W, 3)
        colors = (img_no_norm[combined_mask] * 255).astype(np.uint8)       # (K, 3)

        all_pts.append(pts_world.astype(np.float32))
        all_colors.append(colors)

    return np.concatenate(all_pts, axis=0), np.concatenate(all_colors, axis=0)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_reproject_ba_output_shapes tests/pointcloud/test_mapanything_creator.py::test_reproject_ba_depth_mask_filters_zero_depth tests/pointcloud/test_mapanything_creator.py::test_reproject_ba_identity_extrinsic_preserves_cam_points -v
```
Expected: all 3 PASS.

- [ ] **Step 5: Fix double `PILImage.open` in `_preprocess`**

In `_preprocess`, the `original_coords` list comprehension opens each image twice:

```python
# Before:
original_coords = np.array(
    [
        [0, 0, model_w, model_h, PILImage.open(p).width, PILImage.open(p).height]
        for p in image_paths
    ],
    dtype=np.float32,
)

# After:
# Open each image once to read original dimensions; avoid double file I/O
original_coords = np.array(
    [
        [0, 0, model_w, model_h, *(img := PILImage.open(p)).width, img.height]
        for p in image_paths
    ],
    dtype=np.float32,
)
```

Wait — walrus operator in a list doesn't quite work that syntax. Use a helper instead:

```python
# Open each image once to read original dimensions; avoid double file I/O
def _dims(p: Path) -> tuple[int, int]:
    img = PILImage.open(p)
    return img.width, img.height

original_coords = np.array(
    [
        [0, 0, model_w, model_h, *_dims(p)]
        for p in image_paths
    ],
    dtype=np.float32,
)
```

Actually, a cleaner inline approach without a nested def:

```python
# Open each image once to read original dimensions; avoid double file I/O
original_coords = np.array(
    [
        [0, 0, model_w, model_h, w, h]
        for p in image_paths
        for img in [PILImage.open(p)]
        for w, h in [(img.width, img.height)]
    ],
    dtype=np.float32,
)
```

- [ ] **Step 6: Declare `_processed_views` as a dataclass field**

Add to the `MapAnythingCreator` class body, after the existing attribute declarations:

```python
# Internal preprocessing state; set by _preprocess, consumed by _forward
_processed_views: Any = field(default=None, init=False, repr=False)
```

Remove the bare assignment `self._processed_views = ...` in `_preprocess` — the field declaration replaces the need for it. The `self._processed_views = preprocess_input_views_for_inference(validated)` assignment in `_preprocess` remains; just the declaration moves to the class body.

- [ ] **Step 7: Inline comment audit across all mapanything.py methods**

Every logical block needs a block-level comment. Verify and add where missing:

In `_load_model`:
```python
# Load pretrained model, move to device, set eval mode
model = MapAnything.from_pretrained(self.model_name)
model = model.to(device)
model.eval()
return model
```

In `_preprocess`:
```python
# Collect and sort image paths; reject non-image extensions
exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix in exts)
if not image_paths:
    raise FileNotFoundError(f"No images found in {image_dir}")

# Load images and derive model resolution from first image
views = load_images([str(p) for p in image_paths])
model_h: int = views[0]["img"].shape[-2]
model_w: int = views[0]["img"].shape[-1]

# Open each image once to read original dimensions; avoid double file I/O
original_coords = np.array(...)

# Validate and preprocess views into model-internal format; keep on CPU
# until _forward to avoid holding GPU memory during image loading
validated = validate_input_views_for_inference(views)
self._processed_views = preprocess_input_views_for_inference(validated)

return views, image_paths, original_coords
```

In `_forward`:
```python
# Transfer preprocessed views to model device; kept on CPU in _preprocess
# to avoid holding GPU memory during image loading and validation
for view in self._processed_views:
    ...

# bf16 autocast scoped to model forward only; postprocessing requires float32
# to avoid F.grid_sample dtype mismatch (torch 2.4 enforces strict matching)
with torch.no_grad():
    with torch.autocast(...):
        return model.forward(...)
```

In `_postprocess`:
```python
# Cast bf16 tensors to float32 before postprocessing — F.grid_sample
# requires matching dtypes under torch 2.4 strict enforcement
for pred in raw_outputs:
    ...

# Apply confidence and edge masking via MapAnything's postprocess utility
processed = postprocess_model_outputs_for_inference(...)

# Extract pts3d, colors, extrinsics, intrinsics from processed per-frame dicts
pts3d, colors, extrinsics, intrinsics = collect_pts3d_from_outputs(processed)

# Populate BA fields: per-frame images, confidence, and world-point grid
_images = torch.stack(...)
...

# Voxel downsample to reduce point cloud density
_pcd = o3d.geometry.PointCloud()
...
```

In `extract_intermediate_features`:
```python
# Wrap frames into MapAnything view dicts and preprocess for forward pass
raw_views = [{"img": f.unsqueeze(0)} for f in frames.cpu()]
views = preprocess_input_views_for_inference(raw_views)

# Register a per-call hook on the QKV projection of the chosen block
block = self.model.info_sharing.self_attention_blocks[layer_index]
...

# Run forward pass; hook captures q and k from the attention block
with torch.no_grad():
    self.model.forward(views, ...)
```

- [ ] **Step 8: Run all mapanything tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py -v
```
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/test_mapanything_creator.py
git commit -m "fix(mapanything): implement _reproject_ba; fix double PILImage.open; declare _processed_views field; add inline comments"
```

---

## Task 5: base.py inline comment audit

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`

- [ ] **Step 1: Audit and add inline block comments**

Check every logical block in base.py. The following blocks need verified/added comments:

In `FeedforwardResult.save_zarr`:
```python
# Store scalar metadata and image paths in attrs
store.attrs["image_paths"] = ...

# Save required arrays with lz4 compression
for name, arr in (...):
    store.create_array(...)

# Save optional dense arrays
if self.features is not None:
    ...

# Save world_points chunked by frame: (1, H, W, 3)
if self.world_points is not None:
    ...

# Save images (tensor → numpy) chunked by frame: (1, 3, H, W)
if self.images is not None:
    ...
```

In `FeedforwardResult.load_zarr`:
```python
# Load required arrays
pts3d = store["pts3d"][:]
...

# Load optional arrays; absent keys → None
features = store["features"][:] if "features" in store else None
...
```

In `_raw_to_world_points`:
```python
# Extract and normalize depth: VGGT-X returns (K,H,W,1), MapAnything (K,H,W)
if depth.ndim == 4:
    depth = depth.squeeze(-1)

# Build subsampled pixel grid for sparse world-point extraction
us = np.arange(0, W, subsample)
...

# Unproject each frame's depth grid to world-space points
for ki in range(K):
    ...
```

In `build_pycolmap_reconstruction`:
```python
# Convert colors to uint8
colors_u8 = ...

# Add points — feedforward has no 2D feature tracks, so Track() is empty
for xyz, rgb in zip(pts3d, colors_u8):
    ...

# Add one camera + image per frame
for i, name in enumerate(image_names):
    ...
```

In `BaseFeedforwardCreator` pipeline methods (`reconstruct`, `load_model`, `setup_inference`, `run_inference`, `postprocess`, `build_colmap`): verify each already has block comments (they do from existing code — review and fill any gaps).

In `_verify_loop_candidate`:
```python
# Run the model once to capture cross-frame activations
features = self.extract_intermediate_features(...)

# Compute the cross-frame attention ratio gate
ratio = cross_frame_attention_ratio(features["k"], features["q"])
if ratio < verify_match_ratio:
    return False, None

# "poses" is optional — VGGTx includes it (pre-decoded), MapAnything does not
return True, features.get("poses")
```

- [ ] **Step 2: Smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator, FeedforwardResult, build_pycolmap_reconstruction; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py
git commit -m "refactor(feedforward/base): add inline block comments throughout"
```

---

## Task 6: `__init__.py` — document `_raw_to_world_points` re-export

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/__init__.py`

- [ ] **Step 1: Add explanatory comment**

```python
# Before:
# ── Internal helpers (re-exported for wrappers and tests) ─────────────────────
from .base import _raw_to_world_points

# After:
# ── Internal helpers (re-exported for wrappers and tests) ─────────────────────
# _raw_to_world_points re-exported so wrappers.py BundleAdjustment can call it
# directly on raw VGGT-X outputs without going through the full pipeline.
from .base import _raw_to_world_points
```

- [ ] **Step 2: Commit**

```bash
git add collab_splats/pointcloud/feedforward/__init__.py
git commit -m "docs(feedforward): document _raw_to_world_points re-export reason"
```

---

## Task 7: Full test run

**Files:** none modified

- [ ] **Step 1: Run the full feedforward-related test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ tests/test_cu121_migration.py -v --tb=short
```
Expected: all tests that were passing before these changes still pass. New `test_reproject_ba_*` tests pass.

- [ ] **Step 2: Verify no `_reproject_after_ba` references remain**

```bash
grep -rn "_reproject_after_ba" collab_splats/ tests/
```
Expected: no output (zero matches).

- [ ] **Step 3: Final commit if any fixups needed**

If step 1 or 2 reveals issues, fix and commit with:
```bash
git commit -m "fix(feedforward): address post-review test failures"
```

---

## Self-Review

**Spec coverage:**
- ✅ Drop `List`/`Tuple` typing imports (Task 2)
- ✅ Rename `_reproject_after_ba` → `_reproject_ba` in all 4 files (Tasks 1, 2, 3, 4)
- ✅ Promote `_reproject_ba` to `@abstractmethod` in base (Task 2)
- ✅ Rename underscore locals in vggtx `_postprocess` (Task 3)
- ✅ Implement `MapAnythingCreator._reproject_ba` with TDD (Task 4)
- ✅ Fix double `PILImage.open` (Task 4)
- ✅ Declare `_processed_views` as dataclass field (Task 4)
- ✅ Inline comment audit — all files (Tasks 3, 4, 5)
- ✅ `__init__.py` re-export comment (Task 6)
- ✅ Full test run (Task 7)

**Type consistency:** `_reproject_ba` signature is `(self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — consistent across base (abstract), vggtx (concrete), mapanything (concrete).

**Placeholder scan:** No TBDs, no "similar to task N", no missing code blocks.
