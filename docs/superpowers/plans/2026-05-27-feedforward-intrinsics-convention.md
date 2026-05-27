# Feedforward Intrinsics Convention — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize `FeedforwardResult.intrinsics` to model-resolution K across all backends, fix TSDF RGB crop, and align VGGTXCreator preprocessing to upstream training mode.

**Architecture:** Drop the redundant original-resolution `encoding_to_camera` decode from VGGTXCreator and VGGTOmegaCreator `_forward`; both `raw["intrinsics"]` keys become model-res. Remove the `original_coords`-based scaling in `reproject()` (now a no-op). Fix `_feedforward_to_tsdf_inputs` to crop RGB when `original_coords` records an original-image-pixel crop window. Switch VGGTXCreator to upstream `load_and_preprocess_images(mode="crop")`.

**Tech Stack:** Python 3.11, numpy, torch, vggt, vggt_omega, PIL, pytest

---

## Files

| File | Change |
|---|---|
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | Drop original-res decode in `_forward`; fix `_postprocess` comment |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Drop original-res decode in `_forward`; replace `_preprocess` with crop mode; add `_compute_vggtx_crop_coords`; remove `resize_mode` |
| `collab_splats/pointcloud/feedforward/base.py` | Drop `original_coords` scaling block in `reproject()` |
| `collab_splats/mesh/utils.py` | Add RGB crop heuristic in `_feedforward_to_tsdf_inputs`; fix error message |
| `tests/pointcloud/test_feedforward_intrinsics.py` | New: unit tests for invariants |
| `tests/mesh/test_adapter.py` | Extend: test RGB crop path |

---

## Task 1: Tests for model-res K invariant

**Files:**
- Create: `tests/pointcloud/test_feedforward_intrinsics.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests: result.intrinsics always at model resolution after _postprocess."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator


def _make_raw_omega(model_h=688, model_w=384, orig_h=1920, orig_w=1080):
    """Minimal raw_outputs dict as VGGTOmega._forward returns after fix."""
    N = 3
    rng = np.random.default_rng(0)
    # model-res intrinsics: cx < model_W, cy < model_H
    intr = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intr[:, 0, 0] = 555.0   # fx model-res
    intr[:, 1, 1] = 555.0   # fy
    intr[:, 0, 2] = model_w / 2  # cx = 192, well inside W=384
    intr[:, 1, 2] = model_h / 2  # cy = 344, well inside H=688
    return {
        "images": MagicMock(shape=(N, 3, model_h, model_w)),
        "extrinsic": np.eye(3, 4, dtype=np.float32)[None].repeat(N, axis=0),
        "intrinsics": intr,
        "intrinsics_downsampled": intr,
        "depth": rng.random((N, model_h, model_w), dtype=np.float32).astype(np.float32) + 0.1,
        "depth_conf": rng.random((N, model_h, model_w)).astype(np.float32),
    }


def test_omega_result_intrinsics_cx_inside_model_width():
    """After _postprocess, result.intrinsics cx must be < model_width."""
    raw = _make_raw_omega()
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.conf_threshold = 50.0
    creator.max_points = 500_000
    creator.image_paths = [MagicMock() for _ in range(3)]
    creator.original_coords = np.array(
        [[0, 0, 1080, 1920, 1080, 1920]] * 3, dtype=np.float32
    )
    result = creator._postprocess(raw)
    model_w = result.model_width
    for i in range(result.intrinsics.shape[0]):
        cx = result.intrinsics[i, 0, 2]
        assert cx < model_w, (
            f"cx={cx} >= model_width={model_w}: intrinsics stored at original-res"
        )


def test_omega_result_intrinsics_cy_inside_model_height():
    """After _postprocess, result.intrinsics cy must be < model_height."""
    raw = _make_raw_omega()
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.conf_threshold = 50.0
    creator.max_points = 500_000
    creator.image_paths = [MagicMock() for _ in range(3)]
    creator.original_coords = np.array(
        [[0, 0, 1080, 1920, 1080, 1920]] * 3, dtype=np.float32
    )
    result = creator._postprocess(raw)
    model_h = result.model_height
    for i in range(result.intrinsics.shape[0]):
        cy = result.intrinsics[i, 1, 2]
        assert cy < model_h, (
            f"cy={cy} >= model_height={model_h}: intrinsics stored at original-res"
        )


def test_reproject_uses_intrinsics_directly_no_scaling():
    """reproject() must use self.intrinsics without applying original_coords scaling."""
    import torch
    from dataclasses import replace as dc_replace
    rng = np.random.default_rng(1)
    N, H, W = 2, 8, 8
    depth = rng.random((N, H, W)).astype(np.float32) + 0.5
    pixel_indices = np.array([[0, 2, 3], [1, 4, 5]], dtype=np.int32)
    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intrinsics[:, 0, 0] = 50.0  # fx at model-res
    intrinsics[:, 0, 2] = W / 2
    intrinsics[:, 1, 2] = H / 2
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
    # original_coords: VGGTOmega style (original-image pixel space)
    original_coords = np.array([[0, 0, 1080, 1920, 1080, 1920]] * N, dtype=np.float32)
    images = torch.zeros(N, 3, H, W)
    result = FeedforwardResult(
        points=rng.random((5, 3)).astype(np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=[],
        original_coords=original_coords,
        model_width=W,
        model_height=H,
        depth=depth,
        pixel_indices=pixel_indices,
        images=images,
    )
    reprojected = result.reproject()
    # Points should be finite and reasonably scaled (not blown up by wrong K)
    assert np.all(np.isfinite(reprojected.points)), "reproject() returned non-finite points"
    # x coords from unprojection at fx=50, depth~1: x = (u - cx)/fx * d ≈ small values
    assert np.all(np.abs(reprojected.points[:, 0]) < 100), (
        "reprojected x values too large — scaling may have been applied incorrectly"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_intrinsics.py -v 2>&1 | tail -20
```

Expected: 2–3 failures. `test_omega_result_intrinsics_*` fail because `_postprocess` still reads original-res K (cx=540 > W=384).

- [ ] **Step 3: Commit baseline tests**

```bash
cd /workspace/collab-splats
git add tests/pointcloud/test_feedforward_intrinsics.py
git commit -m "test(feedforward): add model-res K invariant tests — currently failing"
```

---

## Task 2: Fix `VGGTOmegaCreator._forward` — drop original-res decode

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py:185-217`

- [ ] **Step 1: Replace `_forward` body**

In `collab_splats/pointcloud/feedforward/vggt_omega.py`, replace the `_forward` method (lines 185–217):

```python
    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        """Run VGGT-Omega on the preprocessed image tensor; return raw predictions dict."""
        device = next(model.parameters()).device
        image_shape = views.shape[-2:]  # (H_model, W_model)

        # Move images to model device; VGGTOmega adds the batch dim internally
        images = views.to(device)

        # Model forward handles bf16/f16 autocast internally
        with torch.no_grad():
            predictions = model(images)

        # Decode poses at model resolution only — matches upstream demo_gradio.run_model.
        # Original-res decode removed: no downstream consumer requires it and it was the
        # root cause of cx > model_W in result.intrinsics.
        ext, intr = encoding_to_camera(predictions["pose_enc"], image_shape)

        # Move to CPU float32 for downstream numpy ops; squeeze batch dim (always 1)
        extrinsic = ext.cpu().float().numpy().squeeze(0)   # (N, 3, 4)
        intrinsic  = intr.cpu().float().numpy().squeeze(0) # (N, 3, 3) at model-res
        depth      = predictions["depth"].squeeze(0).cpu().float().numpy()      # (N, H, W)
        depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy() # (N, H, W)

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,             # model-res K
            "intrinsics_downsampled": intrinsic, # alias — _raw_to_world_points expects this key
            "depth": depth,
            "depth_conf": depth_conf,
        }
```

Also remove the now-unused `orig_w, orig_h` read. The `_postprocess` already reads `raw_outputs["intrinsics"]` on line 222 — that line's comment needs updating:

```python
        extrinsic = raw_outputs["extrinsic"]   # (N, 3, 4) at model resolution
        intrinsic = raw_outputs["intrinsics"]  # (N, 3, 3) at model resolution
```

- [ ] **Step 2: Run Task 1 tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_intrinsics.py::test_omega_result_intrinsics_cx_inside_model_width tests/pointcloud/test_feedforward_intrinsics.py::test_omega_result_intrinsics_cy_inside_model_height -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all passing (or same failures as before this task).

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats
git add collab_splats/pointcloud/feedforward/vggt_omega.py
git commit -m "fix(feedforward): VGGTOmega _forward — drop original-res K decode, intrinsics now model-res"
```

---

## Task 3: Fix `VGGTXCreator._forward` — drop original-res decode

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py:233-263`

- [ ] **Step 1: Replace the dual-decode block**

In `collab_splats/pointcloud/feedforward/vggtx.py`, replace the block from `width, height = ...` through `return {...}` (lines 233–263):

```python
        image_shape = images.shape[-2:]

        # bf16/f16 autocast scoped to model forward only; downstream numpy ops need float32.
        with torch.no_grad():
            with torch.autocast(device_type, dtype=dtype):
                predictions = model(images.unsqueeze(0))

        # Decode pose encoding at model resolution only — matches VGGT-SLAM upstream.
        # Original-res decode removed: it fed wrong K to the BA wrapper via raw["intrinsics"].
        extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(
            predictions["pose_enc"], image_shape
        )

        # Move predictions to CPU float32 for downstream processing
        extrinsic  = extrinsic_t.cpu().float().numpy().squeeze(0)  # (N, 3, 4)
        intrinsic  = intrinsic_t.cpu().float().numpy().squeeze(0)  # (N, 3, 3) model-res
        depth_map  = predictions["depth"].squeeze(0).cpu().float().numpy()
        depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,             # model-res K
            "intrinsics_downsampled": intrinsic, # alias — _raw_to_world_points expects this key
            "depth": depth_map,
            "depth_conf": depth_conf,
        }
```

Remove the line `width, height = self.original_coords[0, -2:]` that preceded `image_shape = images.shape[-2:]`.

Update `_forward` docstring Returns line: `dict with keys: images, extrinsic, intrinsics, intrinsics_downsampled (alias), depth, depth_conf.`

- [ ] **Step 2: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all passing.

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats
git add collab_splats/pointcloud/feedforward/vggtx.py
git commit -m "fix(feedforward): VGGTXCreator _forward — drop original-res K decode, intrinsics now model-res"
```

---

## Task 4: Fix `FeedforwardResult.reproject()` — drop scaling block

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:228-258`

- [ ] **Step 1: Replace `reproject()` body**

In `collab_splats/pointcloud/feedforward/base.py`, replace the full `reproject()` method body:

```python
    def reproject(self) -> "FeedforwardResult":
        """Re-project points under current extrinsics using stored source pixels and depth.

        ``intrinsics``, ``depth``, and ``pixel_indices`` all live in model-resolution
        space — no scaling required.
        """
        if self.depth is None or self.pixel_indices is None:
            raise ValueError(
                "reproject() requires depth and pixel_indices; load via load_zarr() "
                "or ensure the creator's _postprocess populated both fields."
            )
        # Reproject stored source pixels under current extrinsics — deterministic,
        # point set stays index-aligned with colors and features.
        pts3d = reproject_pixels(
            self.depth,
            self.pixel_indices,
            self.extrinsics[:, :3, :],
            self.intrinsics,
        )
        return replace(self, points=pts3d)
```

- [ ] **Step 2: Run `test_reproject` test**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_intrinsics.py::test_reproject_uses_intrinsics_directly_no_scaling -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3: Run full suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all passing.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats
git add collab_splats/pointcloud/feedforward/base.py
git commit -m "fix(feedforward): reproject() — drop original_coords scaling, intrinsics already model-res"
```

---

## Task 5: Fix `_feedforward_to_tsdf_inputs` RGB crop + error message

**Files:**
- Modify: `collab_splats/mesh/utils.py:438-469`

- [ ] **Step 1: Add test for RGB crop**

In `tests/mesh/test_adapter.py`, add after the existing `_make_result` helper:

```python
def _make_result_with_crop(N=2, model_H=8, model_W=8, orig_H=32, orig_W=16):
    """FeedforwardResult where original_coords records original-image-pixel crop (VGGTOmega style).
    orig_W > model_W so the heuristic fires and crop is applied before resize.
    """
    rng = np.random.default_rng(7)
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
    world_points = rng.random((N, model_H, model_W, 3)).astype(np.float32)
    world_points[..., 2] = 1.5

    tmpdir = tempfile.mkdtemp()
    image_paths = []
    for i in range(N):
        # Save orig_H × orig_W images (larger than model dims)
        img_arr = (rng.random((orig_H, orig_W, 3)) * 255).astype(np.uint8)
        p = Path(tmpdir) / f"frame_{i:04d}.png"
        PILImage.fromarray(img_arr).save(p)
        image_paths.append(p)

    # VGGTOmega-style: coords in original-image pixel space; cr_x = orig_W > model_W
    original_coords = np.tile(
        [0, 0, float(orig_W), float(orig_H), float(orig_W), float(orig_H)], (N, 1)
    ).astype(np.float32)
    # model-res intrinsics: cx < model_W
    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intrinsics[:, 0, 2] = model_W / 2
    intrinsics[:, 1, 2] = model_H / 2
    return FeedforwardResult(
        points=rng.random((10, 3)).astype(np.float32),
        colors=(rng.random((10, 3)) * 255).astype(np.uint8),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=image_paths,
        original_coords=original_coords,
        model_width=model_W,
        model_height=model_H,
        world_points=world_points,
    )


def test_tsdf_rgb_crop_applied_for_original_pixel_coords(tmp_path):
    """_feedforward_to_tsdf_inputs must crop RGB when original_coords exceed model dims."""
    result = _make_result_with_crop(N=2, model_H=8, model_W=8, orig_H=32, orig_W=16)
    mesh_result = pointcloud_to_mesh(
        result, tmp_path / "mesh",
        method="open3d_tsdf",
        voxel_size=0.05,
        sdf_trunc=0.2,
        clean_repair=False,
    )
    assert isinstance(mesh_result, MeshResult)
```

- [ ] **Step 2: Run new test to see current behavior (may pass or fail)**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/mesh/test_adapter.py::test_tsdf_rgb_crop_applied_for_original_pixel_coords -v 2>&1 | tail -15
```

- [ ] **Step 3: Fix `_feedforward_to_tsdf_inputs` in `collab_splats/mesh/utils.py`**

Replace the RGB loading loop and error message:

```python
    if result.world_points is None:
        raise ValueError(
            "result.world_points is None. Access creator.outputs after reconstruct() — "
            "MapAnythingCreator, VGGTXCreator, and VGGTOmegaCreator all populate "
            "world_points during _postprocess()."
        )
```

Replace the RGB loop (currently lines 462–465):

```python
    rgbs = np.empty((N, H, W, 3), dtype=np.float32)
    for i, path in enumerate(result.image_paths):
        img = PILImage.open(path).convert("RGB")
        if result.original_coords is not None:
            tl_x, tl_y, cr_x, cr_y = result.original_coords[i, :4]
            # VGGTOmega and VGGTXCreator crop-mode store original-image-pixel coords;
            # cr_x = orig_w > model_W signals a crop must be applied before resize.
            if cr_x > W + 1 or cr_y > H + 1:
                img = img.crop((float(tl_x), float(tl_y), float(cr_x), float(cr_y)))
        img = img.resize((W, H), PILImage.BILINEAR)
        rgbs[i] = np.asarray(img, dtype=np.float32) / 255.0
```

- [ ] **Step 4: Run all mesh tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/mesh/ -v 2>&1 | tail -20
```

Expected: all passing including new test.

- [ ] **Step 5: Run full suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats
git add collab_splats/mesh/utils.py tests/mesh/test_adapter.py
git commit -m "fix(mesh): _feedforward_to_tsdf_inputs — crop RGB for VGGTOmega/VGGTX crop-mode coords"
```

---

## Task 6: Fix `VGGTXCreator._preprocess` — switch to upstream crop mode

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`

- [ ] **Step 1: Add `_compute_vggtx_crop_coords` helper and update imports**

In `collab_splats/pointcloud/feedforward/vggtx.py`, update the import line:

```python
from vggt.utils.load_fn import load_and_preprocess_images
```

(Remove `load_and_preprocess_images_square, load_and_preprocess_images_ratio` from the import.)

After the `VGGTX_IMG_LOAD_RESOLUTION` constant, add the helper function:

```python
def _compute_vggtx_crop_coords(
    image_paths: list[Path], target_size: int = VGGTX_IMG_LOAD_RESOLUTION
) -> np.ndarray:
    """Compute original_coords for VGGTX upstream crop mode.

    Upstream ``load_and_preprocess_images(mode="crop")`` resizes width→target_size then
    center-crops height to target_size when height > target_size.  This function computes
    the crop window in original-image pixel space so downstream consumers (TSDF RGB loader,
    COLMAP rescale) can invert the transform.

    Returns:
        (N, 6) float32 array [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h] per image.
        cr_x always equals orig_w (full width used).
        For landscape images (no height crop): tl_y=0, cr_y=orig_h.
        For portrait images (height cropped): tl_y and cr_y mark the kept strip.
    """
    coords = []
    for p in image_paths:
        with PILImage.open(p) as img:
            orig_w, orig_h = img.size

        # Upstream: resize width → target_size, maintain AR; round height to div-by-14
        scale = target_size / orig_w
        new_h_raw = orig_h * scale
        new_h = round(new_h_raw / 14) * 14  # divisible-by-14 rounding

        if new_h > target_size:
            # Height crop applied: find crop strip in resized space, map back to original
            start_y_resized = (new_h - target_size) // 2
            # Map resized-pixel boundary back to original-image pixels
            tl_y = start_y_resized / scale
            cr_y = (start_y_resized + target_size) / scale
        else:
            tl_y = 0.0
            cr_y = float(orig_h)

        coords.append([0.0, tl_y, float(orig_w), cr_y, float(orig_w), float(orig_h)])

    return np.array(coords, dtype=np.float32)
```

Also add `from PIL import Image as PILImage` to imports if not already present. Check first — if present, skip.

- [ ] **Step 2: Replace `_preprocess` and remove `resize_mode` attribute**

Replace the `_preprocess` method:

```python
    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        """Load and preprocess images using upstream VGGT crop mode.

        Resizes width to 518px, then center-crops height to 518px when height > 518px.
        Matches the training preprocessing used by VGGT and VGGT-SLAM/SPARK.
        Stores the crop window in original-image pixel coordinates in ``original_coords``
        so the TSDF RGB loader and COLMAP rescale can invert the transform.

        Args:
            image_dir: Directory containing ``.png``/``.jpg``/``.jpeg`` images.

        Returns:
            (images, image_paths, original_coords) where original_coords is (N, 6)
            float32 ``[0, tl_y, orig_w, br_y, orig_w, orig_h]`` in original-image pixels.

        Raises:
            FileNotFoundError: If no supported images are found in image_dir.
        """
        image_dir = Path(image_dir)
        image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Compute crop window in original-image pixel space for downstream consumers
        original_coords = _compute_vggtx_crop_coords(image_paths, VGGTX_IMG_LOAD_RESOLUTION)

        # Load and preprocess using upstream crop mode — matches VGGT training default
        image_names = [str(p) for p in image_paths]
        images = load_and_preprocess_images(image_names, mode="crop")

        return images, image_paths, original_coords
```

Remove `resize_mode: str = "max_size"` field from the dataclass and its `__post_init__` validation:

```python
    camera_model: str = "SIMPLE_PINHOLE"
    model_name: str = "facebook/VGGT-1B"
    use_global_alignment: bool = False
    chunk_size: int = 256
    conf_threshold: float = 35.0
    max_points: int = 500_000
```

(Keep `max_points` if already present; remove only `resize_mode`.)

Remove `__post_init__` entirely if its only content was the `resize_mode` check; otherwise remove just that check.

Update the class docstring to remove the `resize_mode` attribute entry and update the preprocessing description.

- [ ] **Step 3: Add test for crop coords**

In `tests/pointcloud/test_feedforward_intrinsics.py`, add:

```python
from pathlib import Path
import tempfile
from collab_splats.pointcloud.feedforward.vggtx import _compute_vggtx_crop_coords
from PIL import Image as PILImage


def test_vggtx_crop_coords_portrait():
    """Portrait 1080×1920: height crop expected, cr_x = orig_w."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "img.png"
        PILImage.fromarray(
            (np.zeros((1920, 1080, 3), dtype=np.uint8))
        ).save(p)
        coords = _compute_vggtx_crop_coords([p], target_size=518)
    assert coords.shape == (1, 6)
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    assert tl_x == 0.0
    assert cr_x == 1080.0          # full width always kept
    assert orig_w == 1080.0
    assert orig_h == 1920.0
    assert tl_y > 0                # portrait → crop applied
    assert cr_y < orig_h           # crop ends before image bottom
    assert cr_y - tl_y > 0        # non-empty crop


def test_vggtx_crop_coords_landscape_no_crop():
    """Landscape 1920×1080: no height crop (height stays ≤ 518 after width resize)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "img.png"
        PILImage.fromarray(
            (np.zeros((1080, 1920, 3), dtype=np.uint8))
        ).save(p)
        coords = _compute_vggtx_crop_coords([p], target_size=518)
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    assert tl_y == 0.0
    assert cr_y == orig_h          # no crop


def test_vggtx_crop_coords_cr_x_gt_target_size():
    """cr_x must always equal orig_w > target_size for the TSDF heuristic to fire."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "img.png"
        PILImage.fromarray(np.zeros((1920, 1080, 3), dtype=np.uint8)).save(p)
        coords = _compute_vggtx_crop_coords([p], target_size=518)
    assert coords[0, 2] > 518     # cr_x = 1080 > model_W = 518
```

- [ ] **Step 4: Run new tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_intrinsics.py -v 2>&1 | tail -15
```

Expected: all PASS.

- [ ] **Step 5: Run full suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all passing.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats
git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_feedforward_intrinsics.py
git commit -m "fix(feedforward): VGGTXCreator — switch to upstream crop mode, add _compute_vggtx_crop_coords, remove resize_mode"
```

---

## Task 7: Final validation

- [ ] **Step 1: Run complete test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v 2>&1 | tail -40
```

Expected: all tests passing. Note any pre-existing failures from `worklog/known-test-failures.md`.

- [ ] **Step 2: Verify intrinsics invariant end-to-end with synthetic data**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python - <<'EOF'
import numpy as np
from unittest.mock import MagicMock

# Simulate what _feedforward_to_tsdf_inputs receives with fixed omega result
from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
import tempfile
from pathlib import Path
from PIL import Image as PILImage

N, H, W = 2, 8, 8
rng = np.random.default_rng(42)

# model-res intrinsics — cx/cy inside model dims
intr = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
intr[:, 0, 2] = W / 2   # cx = 4 < W=8  ✓
intr[:, 1, 2] = H / 2   # cy = 4 < H=8  ✓

tmpdir = tempfile.mkdtemp()
paths = []
for i in range(N):
    p = Path(tmpdir) / f"f{i}.png"
    PILImage.fromarray(np.zeros((32, 16, 3), dtype=np.uint8)).save(p)
    paths.append(p)

# VGGTOmega-style coords: orig 16×32, cr_x=16 > W=8 → crop fires
oc = np.array([[0, 0, 16, 32, 16, 32]] * N, dtype=np.float32)
wp = rng.random((N, H, W, 3)).astype(np.float32)
wp[..., 2] = 1.5

result = FeedforwardResult(
    points=rng.random((5,3)).astype(np.float32),
    colors=np.zeros((5,3), np.uint8),
    extrinsics=np.eye(4, dtype=np.float32)[None].repeat(N, axis=0),
    intrinsics=intr,
    image_paths=paths,
    original_coords=oc,
    model_width=W, model_height=H,
    world_points=wp,
)

depths, rgbs, c2w, K = _feedforward_to_tsdf_inputs(result)
print(f"depths shape: {depths.shape}")
print(f"rgbs shape:   {rgbs.shape}")
print(f"K[0] cx={K[0,0,2]:.1f}  cy={K[0,1,2]:.1f}  model W={W} H={H}")
assert K[0,0,2] < W, f"FAIL: cx={K[0,0,2]} >= W={W}"
assert K[0,1,2] < H, f"FAIL: cy={K[0,1,2]} >= H={H}"
print("PASS: principal point inside model image bounds")
EOF
```

Expected output:
```
depths shape: (2, 8, 8)
rgbs shape:   (2, 8, 8, 3)
K[0] cx=4.0  cy=4.0  model W=8 H=8
PASS: principal point inside model image bounds
```

- [ ] **Step 3: Commit spec + plan**

```bash
cd /workspace/collab-splats
git add docs/superpowers/specs/2026-05-27-feedforward-intrinsics-convention-design.md \
        docs/superpowers/plans/2026-05-27-feedforward-intrinsics-convention.md
git commit -m "docs(specs): feedforward intrinsics convention — spec + plan"
```

---

## Self-Review

**Spec coverage:**
- ✅ Change 1 (drop original-res decode VGGTX + VGGTOmega) → Tasks 2 + 3
- ✅ Change 2 (reproject drop scaling) → Task 4
- ✅ Change 3 (BA wrapper — no code change needed) → noted in Task 3 (raw["intrinsics"] now correct)
- ✅ Change 4 (build_colmap — no code change) → validated by suite
- ✅ Change 5 (TSDF RGB crop + error msg) → Task 5
- ✅ Change 6 (depth shape verify) → VGGTOmega `_postprocess` already handles ndim 3/4; existing squeeze logic unchanged
- ✅ Change 7 (VGGTX crop mode preprocess) → Task 6
