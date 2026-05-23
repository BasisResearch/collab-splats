# Pointcloud Simplification Design

**Date:** 2026-04-21
**Branch:** refactor/core-modules
**Status:** Approved

## Summary

Three targeted improvements to `collab_splats/pointcloud/`:

1. Extract global alignment into a new `postproc.py`, removing the `maybe_run_global_alignment` wrapper
2. Expand docstrings in `utils.py`
3. Add documentation to feedforward classes and helpers

`CoordinateFrame` keeps its current name and values (`COLMAP`, `NERFSTUDIO`). The docstring should be expanded to include axis convention details (see `utils.py` improvements).

---

## 1. `postproc.py` — Global Alignment Extraction

**New file:** `collab_splats/pointcloud/postproc.py`

Move `_run_global_alignment` from `_vggt.py` into `postproc.py` as a public function `run_global_alignment`. Remove `maybe_run_global_alignment` entirely — callers guard with `if self.use_global_alignment:` directly.

### Function signature

```python
def run_global_alignment(
    raw_outputs: dict[str, Any],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_paths: list[Path],
    colmap_dir: Path | None = None,
    max_query_pts: int = 4096,
    shared_camera: bool = False,
    lambda_depth: float = 0.0,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine camera poses via global alignment against COLMAP or depth-guided BA.

    Args:
        raw_outputs: Dict containing 'images', 'depth', 'depth_conf' tensors
                     from a feedforward model forward pass.
        extrinsic:   (N, 3, 4) or (N, 4, 4) float32 initial camera extrinsics.
        intrinsic:   (N, 3, 3) float32 initial camera intrinsics.
        image_paths: Ordered list of source image paths (used for COLMAP matching).
        colmap_dir:  Optional path to existing COLMAP reconstruction used as
                     anchor for alignment. If None, uses depth-guided BA only.
        max_query_pts: Maximum 2D-3D correspondences per image for PnP.
        shared_camera: Whether all images share a single camera model.
        verbose:     Print alignment progress.
        lambda_depth: Weight for depth regularization in BA (0 = no depth reg).

    Returns:
        refined_extrinsic: (N, 3, 4) float32 refined camera extrinsics.
        refined_intrinsic: (N, 3, 3) float32 refined camera intrinsics.
    """
```

### `_vggt.py` changes

Remove:
- `maybe_run_global_alignment`
- `_run_global_alignment`

Keep:
- `unproject_and_filter_points`

### `VGGTXCreator._postprocess` changes

```python
def _postprocess(self, raw_outputs, **kwargs):
    from ._vggt import unproject_and_filter_points
    from .postproc import run_global_alignment

    extrinsic = raw_outputs["extrinsic"]
    intrinsic = raw_outputs["intrinsics_downsampled"]

    if self.use_global_alignment:
        extrinsic, intrinsic = run_global_alignment(
            raw_outputs, extrinsic, intrinsic,
            image_paths=self.image_paths,
        )

    pts3d, colors = unproject_and_filter_points(...)
    ...
```

`postproc.py` is the intended home for future postprocessing ops shared across feedforward creators (e.g. per-frame depth refinement, multi-view consistency filtering).

---

## 2. `utils.py` Docstring Improvements

**File:** `collab_splats/pointcloud/utils.py`

### Module docstring

Add module-level docstring:

```python
"""Geometric pointcloud utilities: filtering, downsampling, coordinate conversion.

Coordinate convention: all public functions operate in world space.
colmap_reconstruction_to_result outputs CoordinateFrame.NERFSTUDIO (nerfstudio world frame).
"""
```

### `colmap_reconstruction_to_result`

Expand docstring to name each transform step clearly:

```
Transform A — OpenCV → OpenGL camera axes: flip Y and Z columns of c2w.
              Corresponds to multiplying the camera frame by diag(1, -1, -1).
Transform B — COLMAP world (Y-down) → Nerfstudio world (Z-up): swap rows 1↔2, negate new row 2.
              Corresponds to applying _WORLD_TRANSFORM on the left.
```

### `_WORLD_TRANSFORM` constant

Add comment explaining what the matrix does and where it matches `transforms.json["applied_transform"]`.

### `_radial_mask` and `_bbox_mask`

Add full docstrings (currently have none).

---

## 3. Feedforward Documentation

**File:** `collab_splats/pointcloud/feedforward.py`

### `FeedforwardResult`

Add shape+dtype comments to all fields. Document that `original_coords` has layout `[tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]`.

### `BaseFeedforwardCreator`

Expand class docstring to define the contract each abstract method must satisfy:
- `_load_model(device)` → returns model moved to device, in eval mode
- `_preprocess(image_dir)` → returns `(views, image_paths, original_coords)` where `original_coords` is `(N, 6)` float32
- `_forward(model, views)` → returns raw model outputs dict; caller calls `torch.cuda.empty_cache()` after
- `_postprocess(raw_outputs)` → returns `FeedforwardResult`

### `MapAnythingCreator`

Add parameter docstring: what `confidence_percentile` and `use_multiview_confidence` control.

### `VGGTXCreator`

Add parameter docstring: what `use_global_alignment`, `chunk_size`, `conf_threshold` control. Note that `conf_threshold` is a percentile (0–100), not a raw probability.

### `build_pycolmap_reconstruction`

Expand docstring: explain why no `Point2D` tracks are stored (feedforward methods don't produce feature matches), and what `_rescale_reconstruction_to_original_dimensions` does to the output.

### `_rescale_reconstruction_to_original_dimensions`

Add inline comments explaining the scale_x/scale_y computation and the `shared_camera` path.

---

## Files Affected

| File | Change |
|------|--------|
| `collab_splats/pointcloud/utils.py` | Expand docstrings |
| `collab_splats/pointcloud/_vggt.py` | Remove alignment functions |
| `collab_splats/pointcloud/feedforward.py` | Update import, expand docstrings |
| `collab_splats/pointcloud/postproc.py` | **NEW** — global alignment |
| `tests/pointcloud/test_vggtx_creator.py` | Update alignment mock path |

---

## Out of Scope

- No changes to SfM creators (`sfm.py`)
- No new postprocessing operations added to `postproc.py` beyond alignment
- No changes to `_mapanything.py`
- No API changes to `PointcloudResult` fields
