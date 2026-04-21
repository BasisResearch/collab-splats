# Feedforward Pointcloud Consolidation — Design Spec

**Date:** 2026-04-21
**Branch:** refactor/core-modules

---

## Problem

`stage/mapanything_utils.py` (1,513 lines) and `stage/vggt_utils.py` (2,492 lines) live outside
the package, are imported via a sys.path hack in `feedforward.py`, and duplicate each other
significantly. `_rescale_reconstruction_to_original_dimensions` is literally identical in both files
(mapanything's docstring says "adapted from nerfstudio's vggt_utils module"). Both pipelines follow
the same 4-step pattern but the shared structure is invisible.

---

## Insight: Both Pipelines Are the Same 4 Steps

```
1. Load model
2. Preprocess images → views / image_paths / original_coords
3. Forward pass     → pts3d (world), colors, extrinsics, intrinsics
4. Build COLMAP     → filter pts, build_pycolmap_reconstruction, rescale, write_binary
```

The only model-specific code is steps 1-3 and the filtering logic in step 4.
Steps 4b–4d (`build_pycolmap_reconstruction`, `_rescale_reconstruction`, `write_binary`) are shared.

---

## What Is Shared vs Model-Specific

| Step | MapAnything | VGGT | Shared? |
|------|-------------|------|---------|
| Load model | `load_mapanything_model` | inside `run_vggt_inference` | No |
| Preprocess | `load_and_preprocess_images` | inside `run_vggt_inference` | No |
| Forward pass | `run_mapanything_inference` | `run_vggt_inference` | No |
| Collect pts3d | `collect_pts3d_from_outputs` (from outputs dict) | `unproject_depth_map_to_point_map` (from depth maps) | No |
| Filter pts | spatial/voxel filter | confidence threshold | Partly (filter fns → utils.py) |
| Build pycolmap | `build_colmap_reconstruction` + tracks | `_build_pycolmap_reconstruction_without_tracks` via VGGT-X | **→ unified** |
| Rescale intrinsics | `_rescale_reconstruction_to_original_dimensions` | **identical function** | **Yes** |

---

## Target Architecture

```
collab_splats/pointcloud/
  feedforward.py      ← build_pycolmap_reconstruction (shared, no tracks)
                         _rescale_reconstruction_to_original_dimensions (shared, one copy)
                         MapAnythingCreator._run_inference() (~10 lines, orchestration only)
                         VGGTXCreator._run_inference() (~10 lines, orchestration only)
  _mapanything.py     ← run_mapanything() — public entry point
                         (internal: load_mapanything_model, load_and_preprocess_images,
                          run_mapanything_inference, collect_pts3d, voxel_downsample)
  _vggt.py            ← run_vggt() — public entry point
                         (internal: is_vggt_available, unproject_depth_map_to_point_map,
                          filter_vggt_points, run_global_alignment + helpers)
  utils.py            ← (existing) + filter_points_by_spatial_extent
                                    + voxel_downsample_point_cloud
stage/                ← untouched, kept as reference
```

Both `run_mapanything` and `run_vggt` return the same 8-tuple so `_run_inference` is structurally identical in both subclasses:

```
(pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h)
```

---

## Shared: `build_pycolmap_reconstruction`

Simplified from MapAnything's `build_colmap_reconstruction`. No Point2D tracks — nerfstudio's
`colmap_to_json` reads binary files directly and doesn't use tracks for transforms.json.

```python
def build_pycolmap_reconstruction(
    pts3d: np.ndarray,          # (P, 3) world coords, float32
    colors: np.ndarray,         # (P, 3) uint8 RGB
    extrinsics: np.ndarray,     # (N, 3, 4) or (N, 4, 4) world2cam [R|t]
    intrinsics: np.ndarray,     # (N, 3, 3) K matrices
    image_width: int,
    image_height: int,
    image_names: list[str],
    camera_model: str = "PINHOLE",
) -> pycolmap.Reconstruction:
```

- Slices extrinsics to (N, 3, 4) if (N, 4, 4) passed
- One pycolmap.Camera per frame (or shared — handled by the caller passing repeated intrinsics)
- One pycolmap.Image per frame with Rigid3d pose
- P points with empty tracks
- No backprojection, no Point2D observations

Replaces both `build_colmap_reconstruction` (MA) and `_build_pycolmap_reconstruction_without_tracks` (VGGT).

---

## Shared: `_rescale_reconstruction_to_original_dimensions`

Verbatim from either stage file (identical). Rescales camera intrinsics in a pycolmap.Reconstruction
from model resolution to original image resolution.

```python
def _rescale_reconstruction_to_original_dimensions(
    reconstruction: pycolmap.Reconstruction,
    image_paths: list[Path],
    original_coords: np.ndarray,   # (N, 6): [top_x, top_y, crop_r, crop_b, orig_w, orig_h]
    model_size: tuple[int, int],   # (model_w, model_h)
    shared_camera: bool = True,
) -> pycolmap.Reconstruction:
```

---

## Public API: `run_mapanything` and `run_vggt`

Each module exposes one public function. All model-specific complexity stays inside.

```python
# _mapanything.py
def run_mapanything(
    image_dir: Path,
    model_name: str,
    *,
    confidence_percentile: float = 35.0,
    use_multiview_confidence: bool = True,
    minibatch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path], np.ndarray, int, int]:
    """Load + preprocess + inference + collect pts3d + voxel downsample."""
    ...

# _vggt.py
def run_vggt(
    image_dir: Path,
    colmap_dir: Path,
    model_name: str,
    *,
    use_global_alignment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path], np.ndarray, int, int]:
    """Load + preprocess + inference + unproject + optional global alignment + filter."""
    ...
```

Both return `(pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h)`.

---

## Clean `_run_inference()` — Both Subclasses

```python
# MapAnythingCreator
def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
    from ._mapanything import run_mapanything

    pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h = run_mapanything(
        image_dir, self.model_name,
        confidence_percentile=self.confidence_percentile,
        use_multiview_confidence=self.use_multiview_confidence,
        minibatch_size=self.minibatch_size,
    )
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics, intrinsics, model_w, model_h,
        [p.name for p in image_paths],
    )
    recon = _rescale_reconstruction_to_original_dimensions(
        recon, image_paths, original_coords, (model_w, model_h)
    )
    sparse_dir = output_dir / "colmap" / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    recon.write_binary(str(sparse_dir))
    return pycolmap.Reconstruction(str(sparse_dir))


# VGGTXCreator
def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
    from ._vggt import run_vggt

    colmap_dir = output_dir / "colmap"
    pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h = run_vggt(
        image_dir, colmap_dir, self.model_name,
        use_global_alignment=self.use_global_alignment,
    )
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics, intrinsics, model_w, model_h,
        [p.name for p in image_paths],
        camera_model="SIMPLE_PINHOLE",
    )
    recon = _rescale_reconstruction_to_original_dimensions(
        recon, image_paths, original_coords, (model_w, model_h)
    )
    sparse_dir = colmap_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    recon.write_binary(str(sparse_dir))
    return pycolmap.Reconstruction(str(sparse_dir))
```

---

## Implementation Steps

1. **`pointcloud/utils.py`** — add `filter_points_by_spatial_extent`, `voxel_downsample_point_cloud` from mapanything_utils L313-471
2. **`feedforward.py`** — add `build_pycolmap_reconstruction`, `_rescale_reconstruction_to_original_dimensions`; remove `import sys` and `_add_stage_to_path()`
3. **`_mapanything.py`** — create with public `run_mapanything()` entry point; internal helpers: `load_mapanything_model`, `load_and_preprocess_images` (returns `original_coords` as (N,6) array), inference, collect, voxel downsample
4. **`_vggt.py`** — create with public `run_vggt()` entry point; internal helpers: `is_vggt_available`, `unproject_depth_map_to_point_map`, `filter_vggt_points`, `run_global_alignment` + helpers
5. **Rewrite `_run_inference()` methods** — one import, one model call, build + rescale + write
6. **Update test mock paths** — `"stage.mapanything_utils.*"` → `"collab_splats.pointcloud._mapanything.*"`
7. **Verify** — import check + `pytest tests/pointcloud/ -k "not smoke and not gpu"`
8. **Second-pass review** — after tests pass, review stage files for anything missed

---

## Verification

```bash
python -c "from collab_splats.pointcloud.feedforward import MapAnythingCreator, VGGTXCreator; print('OK')"
python -m pytest tests/pointcloud/ -v -k "not smoke and not gpu"
grep -r "_add_stage_to_path\|stage\.mapanything_utils\|stage\.vggt_utils" collab_splats/ tests/
# Expected: 0 hits
```
