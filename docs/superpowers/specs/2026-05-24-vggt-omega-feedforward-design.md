# VGGT-Omega Feedforward Creator — Design Spec

**Date:** 2026-05-24  
**Branch:** refactor/cu121  
**Status:** Approved

---

## Motivation

`VGGTOmegaCreator` was partially implemented but uninstantiable — it named its reprojection method `_reproject_ba` instead of the abstract `_reproject`, causing `TypeError` on every instantiation. This spec covers the fix, a full audit of gaps vs `VGGTXCreator`, and a codebase-wide rename of preprocessing-mode fields to `resize_mode` across all three feedforward creators.

---

## Scope

### Files modified

| File | Change type |
|------|-------------|
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | Fix + extend |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Rename `image_preproc` → `resize_mode` |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Add `resize_mode` + `resolution` |
| `tests/pointcloud/test_vggt_omega_creator.py` | Rename + new tests |
| `tests/pointcloud/test_vggtx_creator.py` | Rename refs |
| `tests/pointcloud/test_mapanything_creator.py` | New tests |
| `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` | Add VGGT-Omega §§ |

### Files not touched

`base.py`, `wrappers.py`, `feedforward/__init__.py`, `pointcloud/__init__.py` — no changes needed.

---

## Design

### 1. `VGGTOmegaCreator` — fixes + new fields

#### 1a. Critical fix: `_reproject_ba` → `_reproject`

The abstract method on `BaseFeedforwardCreator` is `_reproject`. `VGGTOmegaCreator` implemented it as `_reproject_ba`, preventing instantiation. Rename only — logic is correct.

#### 1b. `max_points` gap

`_postprocess` and `_reproject` both call `unproject_and_filter_points` without passing `max_points`. `BaseFeedforwardCreator` defines `max_points: int = 500_000` (inherited). Add `max_points=self.max_points` to both calls, matching `VGGTXCreator`.

#### 1c. New field: `enable_text_alignment: bool = False`

Exposes VGGT-Omega's text-aligned checkpoint variant (`VGGTOmega(enable_alignment=True)`, 256-res).

- `_load_model` passes `enable_alignment=self.enable_text_alignment` to `VGGTOmega(...)` constructor.
- `__post_init__` resolves `resolution`: if `None`, sets to `256` when `enable_text_alignment=True`, else `512`. Explicit values pass through unchanged. Logs resolved value at `DEBUG` level.
- Text-aligned checkpoint exposes `predictions["text_alignment_embedding"]` — not consumed by the pipeline, available via `raw_outputs` for downstream use.

#### 1d. New field: `resolution: int | None = None`

Replaces `image_resolution`. Sentinel default allows `__post_init__` to auto-select:
- `None` + `enable_text_alignment=False` → resolved to `512` (standard checkpoint)
- `None` + `enable_text_alignment=True` → resolved to `256` (text-aligned checkpoint)
- Explicit value → used as-is regardless of `enable_text_alignment`

Resolved value passed as `image_resolution=self.resolution` to `load_and_preprocess_images`.

#### 1e. New field: `resize_mode: str = "balanced"`

Passed as `mode=self.resize_mode` to `load_and_preprocess_images`.

Valid values:

| Value | Behavior | Notes |
|-------|----------|-------|
| `"balanced"` | Smart resize — crops/pads to preserve AR, targets `resolution` | Default; benchmark standard |
| `"max_size"` | Resize longest side to `resolution`, no crop | Lower VRAM; ~512×336 for 3:2 at res=512 |

`__post_init__` validates: `resize_mode in {"balanced", "max_size"}` → `ValueError` otherwise.

#### Updated dataclass

```python
@dataclass
class VGGTOmegaCreator(BaseFeedforwardCreator):
    camera_model: str = "PINHOLE"
    model_path: str | None = None
    model_repo: str = VGGT_OMEGA_HF_REPO
    model_filename: str = VGGT_OMEGA_DEFAULT_FILENAME
    resolution: int | None = None        # None → auto (512 standard, 256 text-aligned); explicit overrides
    resize_mode: str = "balanced"        # mode= passed to load_and_preprocess_images
    conf_threshold: float = 50.0
    enable_text_alignment: bool = False  # VGGTOmega(enable_alignment=True); sets resolution=256 when None
```

---

### 2. `VGGTXCreator` — rename `image_preproc` → `resize_mode`

**Breaking rename.** No other logic change.

| Before | After |
|--------|-------|
| `image_preproc: str = "ratio"` | `resize_mode: str = "max_size"` |
| `"ratio"` value | `"max_size"` value |
| `"square"` value | `"square"` value (unchanged) |

`__post_init__` validation updated: `resize_mode in {"max_size", "square"}`.

`_preprocess` mapping:
- `"max_size"` → `load_and_preprocess_images_ratio`
- `"square"` → `load_and_preprocess_images_square`

Docstring for `resize_mode`:

> `"max_size"` (default): resize longest side to `VGGTX_IMG_LOAD_RESOLUTION` (518), preserving aspect ratio.  
> `"square"`: center-crop + resize to 518×518.

---

### 3. `MapAnythingCreator` — add `resize_mode` + `resolution`

#### New fields

```python
resize_mode: str = "fixed"   # upstream "fixed_mapping"; see docstring for all options
resolution: int = 518        # resolution_set= for "fixed"; size= for "longest_side"/"square"
```

`__post_init__` validates: `resize_mode in {"fixed", "longest_side", "square"}` → `ValueError`.

#### `_preprocess` wiring

```python
_MODE_MAP = {"fixed": "fixed_mapping", "longest_side": "longest_side", "square": "square"}

upstream_mode = _MODE_MAP[self.resize_mode]
if self.resize_mode == "fixed":
    views = load_images(paths, resize_mode=upstream_mode, resolution_set=self.resolution)
else:
    views = load_images(paths, resize_mode=upstream_mode, size=self.resolution)
```

#### Docstring for `resize_mode`

> `"fixed"` (default): auto-selects the best HxW from a lookup table of patch-size-compatible  
> resolutions based on the batch's average aspect ratio. `resolution` selects the lookup table  
> (518 = DINOv2 patch-size-aligned, 512 = ViT-compatible). Best quality, recommended default.  
>  
> `"longest_side"`: resize so the longest side equals `resolution` px, preserving aspect ratio.  
> Use when GPU memory is constrained.  
>  
> `"square"`: resize all images to `resolution × resolution`. Uniform shape; useful for  
> benchmarks requiring identical spatial dimensions across frames.

---

### 4. `resize_mode` alignment across models

| Model | `resize_mode` values | Notes |
|-------|---------------------|-------|
| `VGGTXCreator` | `"max_size"` (default), `"square"` | Maps to two different load functions |
| `MapAnythingCreator` | `"fixed"` (default), `"longest_side"`, `"square"` | `"fixed"` wraps upstream `"fixed_mapping"` |
| `VGGTOmegaCreator` | `"balanced"` (default), `"max_size"` | Passed directly to Omega's `mode=` kwarg |
| 2D feature models (dino, maskclip, talk2dino) | `"max_size"` (default), `"square"` | Field is `resize_mode` — already aligned |

Value names are not unified (each model's upstream uses different terminology). Field name `resize_mode` is the consistency win.

---

### 5. Notebook — `feedforward_methods.ipynb`

Add three sections after §6 (MapAnything Pointcloud Viewer):

- **§7 — VGGT-Omega Reconstruction**: zarr cache-or-run pattern (identical to §1 / §4); uses `VGGTOmegaCreator()` with defaults.
- **§8 — VGGT-Omega Post-processing and Visualisation**: clean via Open3D, extract pts3d/colors, same pattern as §2 / §5.
- **§9 — Three-way Comparison**: side-by-side PyVista viewer with VGGT-X, MapAnything, VGGT-Omega pointclouds + camera poses.

Update notebook title/intro cell to mention all three methods.

---

## Testing

### `test_vggt_omega_creator.py`

**Renames:**
- All `creator._reproject_ba(...)` → `creator._reproject(...)`
- All test function names referencing `_reproject_ba` → `_reproject`

**New tests:**
- `test_enable_text_alignment_auto_sets_resolution_256` — `resolution=None` + `enable_text_alignment=True` → `resolution` resolved to 256
- `test_enable_text_alignment_auto_sets_resolution_512` — `resolution=None` + `enable_text_alignment=False` → `resolution` resolved to 512
- `test_explicit_resolution_not_overridden` — `resolution=768` with `enable_text_alignment=True` → stays 768
- `test_invalid_resize_mode_raises` — `ValueError` on bad value
- `test_resize_mode_balanced_default` — default is `"balanced"`
- `test_load_model_passes_enable_alignment_true` — mock `VGGTOmega` and assert `enable_alignment=True` passed
- `test_load_model_passes_enable_alignment_false` — default path, `enable_alignment=False`
- `test_postprocess_passes_max_points` — assert `unproject_and_filter_points` called with `max_points=self.max_points`
- `test_reproject_passes_max_points` — same for `_reproject`

### `test_vggtx_creator.py`

- Replace all `image_preproc=` → `resize_mode=`
- Replace all `"ratio"` → `"max_size"` where used as field values
- Update any assertion on `creator.image_preproc` → `creator.resize_mode`

### `test_mapanything_creator.py`

- `test_resize_mode_fixed_default` — default is `"fixed"`
- `test_invalid_resize_mode_raises` — `ValueError` on bad value
- `test_preprocess_fixed_mode_calls_load_images_with_fixed_mapping` — mock `load_images`, assert `resize_mode="fixed_mapping"`, `resolution_set=518`
- `test_preprocess_longest_side_calls_load_images_with_size` — mock `load_images`, assert `resize_mode="longest_side"`, `size=518`

---

## Error handling summary

| Condition | Where | Raised |
|-----------|-------|--------|
| `resize_mode` not in valid set | `__post_init__` | `ValueError` |
| `model_path` set but file missing | `_load_model` | `FileNotFoundError` |
| No images in `image_dir` | `_preprocess` | `FileNotFoundError` |
| `resolution=None` | `__post_init__` | auto-resolved; `DEBUG` log |

---

## Non-goals

- Exposing `predictions["text_alignment_embedding"]` in `FeedforwardResult` — out of scope; available in `raw_outputs` if needed downstream.
- Unifying `resize_mode` value names across all models — upstream APIs differ; field name alignment is sufficient.
- Adding `resolution` field to `VGGTXCreator` — resolution is hardcoded by design (`VGGTX_IMG_LOAD_RESOLUTION = 518`).
- `"fixed_size"` mode for MapAnythingCreator — requires a `(width, height)` tuple, doesn't fit the `resolution: int` pattern.
