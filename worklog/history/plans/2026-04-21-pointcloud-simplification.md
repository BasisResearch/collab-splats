# Pointcloud Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract global alignment into `postproc.py` and expand docstrings across the pointcloud module. `CoordinateFrame` keeps its current name and values (`COLMAP`, `NERFSTUDIO`).

**Architecture:** Three sequential tasks, each committed independently. `postproc.py` goes first (structural); docstrings go last (no functional change). TDD for the structural task.

**Tech Stack:** Python 3.10+, numpy, pycolmap, open3d, pytest

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/pointcloud/utils.py` | Expand docstrings |
| `collab_splats/pointcloud/_vggt.py` | Remove `maybe_run_global_alignment` and `_run_global_alignment` |
| `collab_splats/pointcloud/feedforward.py` | Update import in `_postprocess`, expand docstrings |
| `collab_splats/pointcloud/postproc.py` | **NEW** — `run_global_alignment` |
| `tests/pointcloud/test_vggtx_creator.py` | Update alignment mock patch path |

---

## Task 1: Extract global alignment into `postproc.py`

**Files:**
- Create: `collab_splats/pointcloud/postproc.py`
- Modify: `collab_splats/pointcloud/_vggt.py`
- Modify: `collab_splats/pointcloud/feedforward.py`
- Modify: `tests/pointcloud/test_vggtx_creator.py`

- [ ] **Step 1: Update `test_vggtx_creator.py` to use new mock path**

In `tests/pointcloud/test_vggtx_creator.py`, add a new test for the postprocess alignment call:

```python
def test_vggtx_postprocess_calls_global_alignment(tmp_path):
    creator = VGGTXCreator(use_global_alignment=True)
    creator.image_paths = [Path("frame_0000.jpg"), Path("frame_0001.jpg")]
    creator.original_coords = np.zeros((2, 6), dtype=np.float32)

    raw_outputs = {
        "images": np.zeros((2, 3, 518, 518)),
        "extrinsic": np.eye(4)[None, :3, :].repeat(2, axis=0).astype(np.float32),
        "intrinsics_downsampled": np.eye(3)[None].repeat(2, axis=0).astype(np.float32),
        "depth": np.ones((2, 518, 518), dtype=np.float32),
        "depth_conf": np.ones((2, 518, 518), dtype=np.float32) * 80,
    }

    with patch("collab_splats.pointcloud.feedforward.run_global_alignment",
               return_value=(raw_outputs["extrinsic"], raw_outputs["intrinsics_downsampled"])) as mock_align, \
         patch("collab_splats.pointcloud._vggt.unproject_and_filter_points",
               return_value=(np.zeros((10, 3), dtype=np.float32), np.zeros((10, 3), dtype=np.uint8))):
        result = creator._postprocess(raw_outputs)

        mock_align.assert_called_once()
        assert isinstance(result, FeedforwardResult)
```

Also update the existing `test_vggtx_global_alignment_passed_through` to not patch `maybe_run_global_alignment` — that function is being removed. Replace the `use_global_alignment` assertion to check that `run_global_alignment` is not called when `use_global_alignment=False`:

```python
def test_vggtx_no_global_alignment_when_disabled(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    with patch("collab_splats.pointcloud._vggt.run_vggt",
               return_value=_run_vggt_mock_8tuple(image_dir)) as mock_run, \
         patch("collab_splats.pointcloud.feedforward.build_pycolmap_reconstruction",
               return_value=_mock_recon()), \
         patch("collab_splats.pointcloud.feedforward._rescale_reconstruction_to_original_dimensions",
               return_value=_mock_recon()), \
         patch("pycolmap.Reconstruction", return_value=_mock_recon()), \
         patch("collab_splats.pointcloud.feedforward.run_global_alignment") as mock_align, \
         patch.object(VGGTXCreator, "_write_transforms"):
        creator = VGGTXCreator(use_global_alignment=False)
        creator._run_inference(image_dir, output_dir)
        mock_align.assert_not_called()
```

- [ ] **Step 2: Run tests to confirm new test fails**

```bash
pytest tests/pointcloud/test_vggtx_creator.py::test_vggtx_postprocess_calls_global_alignment -v
```

Expected: FAIL — `collab_splats.pointcloud.feedforward.run_global_alignment` not found

- [ ] **Step 3: Create `postproc.py`**

Create `collab_splats/pointcloud/postproc.py` with this content:

```python
"""Postprocessing operations shared across feedforward pointcloud creators."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from vggt.dependency.global_alignment import extract_matches, pose_optimization  # type: ignore


def run_global_alignment(
    raw_outputs: Dict[str, Any],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_paths: List[Path],
    colmap_dir: Optional[Path] = None,
    max_query_pts: int = 4096,
    shared_camera: bool = False,
    lambda_depth: float = 0.0,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine camera poses via feature matching and bundle adjustment.

    Two-stage pipeline:
      1. extract_matches   — build 2D-3D correspondences across all image pairs.
                             Result is cached to ``colmap_dir/matches.pt``.
                             Set ``overwrite=False`` to reload from cache, useful
                             when tuning ``lambda_depth`` without re-running inference.
      2. pose_optimization — joint BA over extrinsics and intrinsics.

    Args:
        raw_outputs:   Forward-pass output dict. Required keys:
                         'images'     — (N, C, H, W) float tensor
                         'depth'      — (N, H, W) float32 depth map
                         'depth_conf' — (N, H, W) float32 per-pixel confidence
        extrinsic:     (N, 3, 4) or (N, 4, 4) float32 world-to-camera matrices.
        intrinsic:     (N, 3, 3) float32 camera intrinsics K.
        image_paths:   Ordered image Paths; basenames used by COLMAP matchers.
        colmap_dir:    Working directory for ``matches.pt``. Created if absent.
                       Defaults to ``Path(".")``.
        max_query_pts: Max 2D-3D correspondences per image for PnP.
        shared_camera: True if all images share one camera model.
        lambda_depth:  Depth regularization weight in BA. 0 = reprojection only.
        overwrite:     When False, load ``matches.pt`` from disk if it exists.

    Returns:
        Tuple of (refined_extrinsic, refined_intrinsic), same shapes as inputs.
    """
    colmap_dir = Path(colmap_dir) if colmap_dir else Path(".")
    colmap_dir.mkdir(parents=True, exist_ok=True)
    matches_path = colmap_dir / "matches.pt"

    # extract_matches requires bare filenames (no parent paths) for the match table.
    image_basenames = [p.name for p in image_paths]

    if not overwrite and matches_path.exists():
        # Reload cached correspondences — skip the expensive matching step.
        match_outputs = torch.load(matches_path)
    else:
        # Build 2D-3D correspondences across all image pairs and cache to disk.
        match_outputs = extract_matches(
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            base_image_path_list=image_basenames,
            max_query_pts=max_query_pts,
        )
        torch.save(match_outputs, matches_path)

    # Run joint bundle adjustment over extrinsics and intrinsics.
    extrinsic_refined, intrinsic_refined = pose_optimization(
        match_outputs=match_outputs,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        images=raw_outputs["images"],
        depth_conf=raw_outputs["depth_conf"],
        base_image_path_list=image_basenames,
        target_scene_dir=colmap_dir,
        shared_intrinsics=shared_camera,
        depth_maps=raw_outputs["depth"],
        lambda_depth=lambda_depth,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return extrinsic_refined, intrinsic_refined
```

- [ ] **Step 4: Update `feedforward.py` `_postprocess` in `VGGTXCreator`**

In `collab_splats/pointcloud/feedforward.py`, add the top-level import:

```python
from .postproc import run_global_alignment
```

Then update `VGGTXCreator._postprocess` to remove `maybe_run_global_alignment` and call `run_global_alignment` directly:

```python
def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
    from ._vggt import unproject_and_filter_points

    extrinsic = raw_outputs["extrinsic"]
    intrinsic = raw_outputs["intrinsics_downsampled"]

    if self.use_global_alignment:
        extrinsic, intrinsic = run_global_alignment(
            raw_outputs, extrinsic, intrinsic, self.image_paths,
        )

    pts3d, colors = unproject_and_filter_points(
        depth=raw_outputs["depth"],
        depth_conf=raw_outputs["depth_conf"],
        images=raw_outputs["images"],
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        conf_threshold=self.conf_threshold,
    )

    model_h = int(raw_outputs["depth"].shape[1])
    model_w = int(raw_outputs["depth"].shape[2])

    return FeedforwardResult(
        pts3d=pts3d,
        colors=colors,
        extrinsics=extrinsic,
        intrinsics=intrinsic,
        image_paths=self.image_paths,
        original_coords=self.original_coords,
        model_width=model_w,
        model_height=model_h,
    )
```

- [ ] **Step 5: Remove alignment functions from `_vggt.py`**

In `collab_splats/pointcloud/_vggt.py`:

1. Delete the entire `maybe_run_global_alignment` function.
2. Delete the entire `_run_global_alignment` function.
3. Update the module docstring to remove the mention of those functions:

```python
"""VGGT-X utility functions used by VGGTXCreator.

Provides: unproject_and_filter_points.
GPU-only — requires VGGT-X to be installed.
"""
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/pointcloud/test_vggtx_creator.py -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/postproc.py collab_splats/pointcloud/_vggt.py collab_splats/pointcloud/feedforward.py tests/pointcloud/test_vggtx_creator.py
git commit -m "refactor(pointcloud): extract global alignment into postproc.py, remove maybe_run_global_alignment"
```

---

## Task 2: Expand `utils.py` docstrings

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`

No new tests — documentation-only. Run existing tests to confirm nothing broke.

- [ ] **Step 1: Add module docstring and update `_WORLD_TRANSFORM` comment**

At the top of `collab_splats/pointcloud/utils.py`, replace the existing module docstring with:

```python
"""Geometric pointcloud utilities: filtering, downsampling, and coordinate conversion.

Coordinate convention used throughout:
  - Input from pycolmap uses COLMAP world (Y-down) + OpenCV camera axes (X right, Y down, Z forward).
  - All public functions that produce poses output CoordinateFrame.NERFSTUDIO (nerfstudio world frame):
    X right, Y up, Z backward camera axes; Z-up world.
  - colmap_reconstruction_to_result applies the two transforms that nerfstudio's
    ns-process-data applies: OpenCV→OpenGL camera axes (A), then COLMAP→nerfstudio world (B).
"""
```

Replace the `_WORLD_TRANSFORM` constant comment:

```python
# COLMAP world (Y-down, right-handed) → nerfstudio world (Z-up, right-handed).
# Row-swap + sign flip: [X, Y, Z] → [X, Z, -Y].
# Matches transforms.json["applied_transform"] written by nerfstudio's ns-process-data.
_WORLD_TRANSFORM = np.array([
    [1,  0, 0, 0],
    [0,  0, 1, 0],
    [0, -1, 0, 0],
], dtype=np.float32)
```

- [ ] **Step 2: Expand `colmap_reconstruction_to_result` docstring**

Replace the existing docstring on `colmap_reconstruction_to_result` with:

```python
def colmap_reconstruction_to_result(
    recon: pycolmap.Reconstruction,
    confidence: np.ndarray | None = None,
) -> PointcloudResult:
    """Convert pycolmap.Reconstruction to PointcloudResult in the nerfstudio world frame.

    Applies two transforms to each camera-to-world pose:

    Transform A — OpenCV → OpenGL camera axes:
        Flip the Y and Z columns of c2w (equivalent to right-multiplying the
        camera frame by diag(1, −1, −1)).  After this, camera Y points up and
        camera Z points backward (out of the lens).

    Transform B — COLMAP world (Y-down) → nerfstudio world (Z-up):
        Swap rows 1 and 2 of c2w, then negate the new row 2.
        This is left-multiplication by _WORLD_TRANSFORM and matches the
        ``applied_transform`` field written to ``transforms.json`` by nerfstudio.

    Args:
        recon:      pycolmap Reconstruction object (registered images + points).
        confidence: Optional (N,) float32 per-point confidence scores.
                    Pass None for SfM results (no per-point confidence available).

    Returns:
        PointcloudResult with:
          frame = CoordinateFrame.NERFSTUDIO
          world_transform = _WORLD_TRANSFORM (3, 4) — the applied B transform.
          extrinsics — (M, 4, 4) float32 camera-to-world poses in nerfstudio frame.
          intrinsics — (M, 3, 3) float32 K matrices (fx, fy, cx, cy per image).
    """
```

- [ ] **Step 3: Add docstrings to `_radial_mask` and `_bbox_mask`**

Replace the existing (empty) docstrings on both private helpers:

```python
def _radial_mask(
    points: np.ndarray,
    max_distance: float | None,
    n_points: int | None,
    reference: str,
) -> np.ndarray:
    """Compute a boolean keep-mask using Euclidean distance from a reference point.

    Args:
        points:       (N, 3) float32 world-space point coordinates.
        max_distance: Keep points within this Euclidean distance from reference.
                      At least one of max_distance / n_points must be provided.
        n_points:     Keep the N closest points to reference.
        reference:    'centroid' uses points.mean(axis=0); 'origin' uses [0,0,0].

    Returns:
        (N,) boolean array; True = keep.

    Raises:
        ValueError: If both max_distance and n_points are None, or reference is unknown.
    """
```

```python
def _bbox_mask(
    points: np.ndarray,
    percentile_range: tuple[float, float],
    max_extent: float | None,
) -> np.ndarray:
    """Compute a boolean keep-mask using a per-axis percentile bounding box.

    Computes per-axis min/max from percentile_range, then optionally clips the
    box to max_extent around its centre.

    Args:
        points:           (N, 3) float32 world-space point coordinates.
        percentile_range: (min_pct, max_pct) used to derive per-axis bounds.
                          E.g., (1.0, 99.0) removes the outermost 1 % on each axis.
        max_extent:       If set, clips the percentile box to this absolute size
                          around the box centre (world units).

    Returns:
        (N,) boolean array; True = keep.
    """
```

- [ ] **Step 4: Run full test suite for pointcloud**

```bash
pytest tests/pointcloud/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/utils.py
git commit -m "docs(pointcloud): expand utils.py module docstring, colmap_reconstruction_to_result, and private mask helpers"
```

---

## Task 3: Expand feedforward docstrings

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py`

Documentation-only. No new tests needed.

- [ ] **Step 1: Update `FeedforwardResult` field comments**

Replace the `FeedforwardResult` class with:

```python
@dataclass
class FeedforwardResult:
    """Typed output from a feedforward creator's ``_postprocess`` step.

    All arrays are float32 unless noted. Shapes assume N images and P output points.
    """

    pts3d: np.ndarray            # (P, 3) float32 — world-space XYZ points
    colors: np.ndarray           # (P, 3) uint8 — RGB, range [0, 255]
    extrinsics: np.ndarray       # (N, 3, 4) float32 — world-to-camera [R|t]
    intrinsics: np.ndarray       # (N, 3, 3) float32 — camera intrinsics K
    image_paths: list[Path]      # length N — source image paths, ordered to match extrinsics
    original_coords: np.ndarray  # (N, 6) float32 — [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]
                                 #   tl = model crop top-left in original pixels
                                 #   cr = model crop bottom-right in original pixels
                                 #   orig_w/h = full original image dimensions
    model_width: int             # model inference resolution width (pixels)
    model_height: int            # model inference resolution height (pixels)
```

- [ ] **Step 2: Expand `BaseFeedforwardCreator` docstring**

Replace the class docstring on `BaseFeedforwardCreator`:

```python
@dataclass
class BaseFeedforwardCreator(BasePointcloudCreator):
    """Template Method pipeline for feedforward pointcloud creators.

    Runs a fixed 5-step pipeline: load_model → setup_inference → run_inference
    → postprocess → build_colmap.  Subclasses implement the four abstract methods
    below; the base class handles device detection, GPU memory cleanup, state
    storage, and COLMAP reconstruction (shared across all feedforward methods).

    Abstract methods — contract each subclass must satisfy:

    ``_load_model(device: str) -> Any``
        Load the model from a pretrained checkpoint, move to ``device``, set to
        eval mode, and return it.  Do not store GPU state outside the returned
        model object.

    ``_preprocess(image_dir: Path) -> tuple[Any, list[Path], np.ndarray]``
        Load and preprocess images from ``image_dir``.  Return:
          views           — model-specific input batch (tensor or list of dicts)
          image_paths     — ordered list of image Paths (length N)
          original_coords — (N, 6) float32 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]

    ``_forward(model, views, **kwargs) -> Any``
        Run the model forward pass under ``torch.no_grad()``.  Return raw outputs
        as a dict or any structure ``_postprocess`` expects.  The base class calls
        ``torch.cuda.empty_cache()`` immediately after this returns.

    ``_postprocess(raw_outputs, **kwargs) -> FeedforwardResult``
        Convert raw model outputs to a ``FeedforwardResult``.  This is where
        depth unprojection, confidence filtering, and optional postprocessing
        (e.g., global alignment) happen.

    Attributes:
        camera_model: pycolmap camera model string for COLMAP reconstruction.
                      Use ``"PINHOLE"`` (fx, fy, cx, cy) or ``"SIMPLE_PINHOLE"``
                      (f, cx, cy — single focal length).
    """
```

- [ ] **Step 3: Add parameter docstrings to `MapAnythingCreator`**

Replace the `MapAnythingCreator` class docstring:

```python
@dataclass
class MapAnythingCreator(BaseFeedforwardCreator):
    """Pointcloud via MapAnything feedforward depth + pose estimation.

    Uses Facebook's MapAnything model to jointly predict per-image depth maps
    and camera poses in a single forward pass without requiring any SfM.

    Attributes:
        model_name:               HuggingFace model ID to load via
                                  ``MapAnything.from_pretrained``.
        confidence_percentile:    Mask out pixels whose multiview confidence
                                  score falls below this percentile (0–100).
                                  Higher = more aggressive masking, fewer points.
        use_multiview_confidence: When True, uses cross-view consistency scores
                                  to mask unreliable depth predictions.
                                  Set False to keep all pixels regardless of
                                  inter-frame agreement.
        minibatch_size:           Number of images processed per inference step.
                                  Reduce if running out of GPU memory.
    """
```

- [ ] **Step 4: Add parameter docstrings to `VGGTXCreator`**

Replace the `VGGTXCreator` class docstring:

```python
@dataclass
class VGGTXCreator(BaseFeedforwardCreator):
    """Pointcloud via VGGT-X feedforward pose + depth estimation.

    Uses VGGT-X (Video Grounded Gaussian Transformer) to jointly predict camera
    poses and per-frame depth maps.  Depth maps are then unprojected to a 3D
    point cloud.

    Attributes:
        camera_model:         pycolmap camera model. Defaults to
                              ``"SIMPLE_PINHOLE"`` because VGGT-X predicts a
                              single focal length (not separate fx/fy).
        model_name:           HuggingFace model ID loaded via
                              ``VGGT.from_pretrained``.
        use_global_alignment: When True, runs ``run_global_alignment`` after the
                              forward pass to refine poses via feature matching +
                              bundle adjustment.  Adds significant compute time
                              but improves accuracy on long sequences.
        chunk_size:           Attention chunk size for memory-efficient inference.
                              Reduce if OOM on long sequences.
        conf_threshold:       Depth confidence percentile cutoff (0–100).
                              Points whose confidence is below this percentile
                              are discarded.  50.0 = keep the top 50 %.
    """
```

- [ ] **Step 5: Expand `build_pycolmap_reconstruction` docstring**

Replace the existing docstring on `build_pycolmap_reconstruction`:

```python
def build_pycolmap_reconstruction(
    pts3d: np.ndarray,
    colors: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: list[str],
    camera_model: str = "PINHOLE",
) -> pycolmap.Reconstruction:
    """Build a pycolmap Reconstruction from pointcloud + camera data.

    Creates one camera and one image per entry in ``image_names``.  Points are
    added as free 3D points with no ``Point2D`` track observations — feedforward
    methods do not produce feature matches, so there are no 2D-3D correspondences
    to record.  This means the reconstruction is valid for writing to disk and
    converting to ``transforms.json``, but cannot be used as input to COLMAP BA.

    Args:
        pts3d:        (P, 3) float32 or float64 world-space point positions.
        colors:       (P, 3) uint8 or float32 [0,1] RGB colors.
                      Float inputs are clipped and scaled to uint8 automatically.
        extrinsics:   (N, 3, 4) or (N, 4, 4) float32 world-to-camera matrices.
                      The first 3 rows are used; the 4th row is ignored.
        intrinsics:   (N, 3, 3) float32 camera intrinsics K per image.
        image_width:  Width in pixels of the model inference resolution.
        image_height: Height in pixels of the model inference resolution.
        image_names:  Length-N list of image filenames (basename only, no path).
        camera_model: pycolmap camera model string.
                      ``"PINHOLE"`` — params [fx, fy, cx, cy].
                      ``"SIMPLE_PINHOLE"`` — params [f, cx, cy], f = mean(fx, fy).

    Returns:
        pycolmap.Reconstruction with cameras, images, frames, and 3D points.
        Call ``_rescale_reconstruction_to_original_dimensions`` before writing
        to disk if the model resolution differs from the original image size.
    """
```

- [ ] **Step 6: Add inline comments to `_rescale_reconstruction_to_original_dimensions`**

Above the `scale_x`/`scale_y` lines in `_rescale_reconstruction_to_original_dimensions`, add:

```python
        # scale_x/scale_y: ratio of original image size to model inference size.
        # Multiplying camera params by these factors maps from model-resolution
        # pixel coordinates back to original-resolution pixel coordinates.
        scale_x = real_image_size[0] / image_size[0]
        scale_y = real_image_size[1] / image_size[1]
```

- [ ] **Step 7: Run full test suite**

```bash
pytest tests/pointcloud/ -v
```

Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py
git commit -m "docs(pointcloud): expand feedforward docstrings — FeedforwardResult, BaseFeedforwardCreator, creators, build helpers"
```

---

## Final Check

- [ ] **Run full pointcloud test suite one last time**

```bash
pytest tests/pointcloud/ -v
```

Expected: all tests PASS, no import errors for `CoordinateFrame` or `maybe_run_global_alignment`
