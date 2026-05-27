# Feedforward Intrinsics Convention — Design

**Date:** 2026-05-27
**Status:** ready-for-plan
**Author:** Tommy

## Problem

`FeedforwardResult.intrinsics` is consumed by four code paths with conflicting resolution assumptions:

| Consumer | Expected K resolution |
|---|---|
| `_feedforward_to_tsdf_inputs` → `Open3DTSDFFusion` | model-res (matches `depth (N, H_m, W_m)`) |
| `build_colmap` → `_rescale_reconstruction_to_original_dimensions` | model-res (rescaled UP to original) |
| `BundleAdjustment` (`wrappers.py:202`) | model-res (matches depth + world_points grid) |
| `FeedforwardResult.reproject()` | original-res (scales DOWN with `original_coords`) |

Both VGGTX and VGGTOmega decode K twice in `_forward` — once at model-res (`intrinsics_downsampled`) and once at original-res (`intrinsics`). This is unnecessary; downstream only needs model-res K.

| Backend | `raw["intrinsics"]` | `raw["intrinsics_downsampled"]` | `result.intrinsics` |
|---|---|---|---|
| `VGGTXCreator` | **original-res** (from `pose_encoding_to_extri_intri(pose_enc, [orig_W, orig_H])`) | model-res ✓ | model-res ✓ (`_postprocess` prefers `_downsampled`) |
| `VGGTOmegaCreator` | **original-res** (from `encoding_to_camera(pose_enc, (orig_h, orig_w))`) | model-res ✓ | **original-res ✗** (`_postprocess` reads `"intrinsics"`) |
| `MapAnythingCreator` | n/a (list-of-per-frame-dicts raw) | n/a | model-res ✓ (predicted directly) |

VGGTXCreator `_postprocess` correctly picks `intrinsics_downsampled`; VGGTOmegaCreator `_postprocess` mistakenly reads `"intrinsics"` (original-res) — root of the observed TSDF break.

`wrappers.py:202` reads `raw["intrinsics"]` (original-res) for **both** VGGTX and VGGTOmega → BA optimizes with K that does not match world_points grid in either backend.

Empirical break: 7-Scenes input 1080×1920, VGGTOmega depth 688×384, intrinsics cx=540 / cy=960. `Open3DTSDFFusion` receives `PinholeCameraIntrinsic(W=384, H=688, cx=540, cy=960)` — principal point outside image, all 3D geometry corrupted.

Additional breaks:
- `build_colmap` → `_rescale_reconstruction_to_original_dimensions` scales original-res K by `orig_W/model_W` again → ~2.8× too large (VGGTOmega only — VGGTX `result.intrinsics` already correct)
- VGGTXCreator default `resize_mode='max_size'` diverges from training/VGGT-SLAM default (`"crop"` mode)
- VGGTXCreator `resize_mode='square'` pads with black — model never trained on black-padded inputs, degrades depth near borders
- `original_coords` space inconsistent: VGGTX stores model-pixel coords; VGGTOmega stores original-image-pixel coords — requires heuristic in TSDF RGB crop

## Decision

**Standardize: `FeedforwardResult.intrinsics` always holds model-resolution K.**

Rationale:
- All consumers except `reproject()` expect model-res K; `reproject()` currently scales down, but with model-res K it needs no scaling at all
- MapAnything already correct; VGGTX `result.intrinsics` already correct (but `raw["intrinsics"]` still wrong for BA wrapper); VGGTOmega both wrong
- Model-res K mechanically tied to `depth.shape[1:]` and `world_points.shape[1:3]` — invariant
- Original-res K recoverable via `_rescale_reconstruction_to_original_dimensions` at COLMAP output boundary

### Upstream alignment

VGGTOmega `demo_gradio.py`: `encoding_to_camera(pose_enc, predictions["images"].shape[-2:])` — model-res only, decoded once.  
VGGT-SLAM/VGGT-SPARK: `load_and_preprocess_images(image_names)` → default `mode="crop"` → decode at model-res only.  
Both upstream references compute K once at model resolution. Our dual-decode in VGGTX and VGGTOmega `_forward` is unnecessary divergence.

`build_colmap` (base.py:633) passes `o.model_width, o.model_height` to `_rescale_reconstruction_to_original_dimensions`; `_rescale` computes `scale_x = orig_W / model_W` — confirms function expects model-res K input. The original-res decode in `_forward` was dead work from the start.

## Changes

### 1. Drop original-res K decode from `VGGTXCreator._forward` and `VGGTOmegaCreator._forward`

Both backends currently decode K twice — once at model-res, once at original-res. Drop the original-res decode from both.

**`vggt_omega.py` `_forward`** — replace dual-decode with single model-res decode:

```python
# Decode poses at model resolution only (matches upstream demo_gradio.run_model)
extrinsic_t, intrinsic_t = encoding_to_camera(
    predictions["pose_enc"], predictions["images"].shape[-2:]
)
extrinsic = extrinsic_t.cpu().float().numpy().squeeze(0)   # (N, 3, 4)
intrinsic  = intrinsic_t.cpu().float().numpy().squeeze(0)  # (N, 3, 3) at model-res
depth      = predictions["depth"].squeeze(0).cpu().float().numpy()
depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()

return {
    "images": images,
    "extrinsic": extrinsic,
    "intrinsics": intrinsic,              # model-res (was: original-res)
    "intrinsics_downsampled": intrinsic,  # alias kept for _raw_to_world_points compat
    "depth": depth,
    "depth_conf": depth_conf,
}
```

`_postprocess` line 222 reads `raw_outputs["intrinsics"]` — now model-res, no change to that line needed.

**`vggtx.py` `_forward`** — drop the `pose_encoding_to_extri_intri(pose_enc, [orig_W, orig_H])` call; keep only the model-res decode. Update return dict so `"intrinsics"` = `intrinsic_downsampled` (model-res):

```python
extrinsic_ds, intrinsic_ds = pose_encoding_to_extri_intri(
    predictions["pose_enc"], image_shape   # model-res only
)
extrinsic  = extrinsic_ds.cpu().float().numpy().squeeze(0)
intrinsic  = intrinsic_ds.cpu().float().numpy().squeeze(0)  # model-res

return {
    "images": images,
    "extrinsic": extrinsic,
    "intrinsics": intrinsic,              # model-res (was: original-res)
    "intrinsics_downsampled": intrinsic,  # alias kept for _raw_to_world_points compat
    "depth": depth_map,
    "depth_conf": depth_conf,
}
```

`_postprocess` already reads `raw.get("intrinsics_downsampled", raw.get("intrinsics"))` — still correct, no change needed.

### 2. `FeedforwardResult.reproject` (base.py:235-249)

After change 1, `result.intrinsics` is model-res for all backends. No scaling needed: depth, pixel_indices, and intrinsics all live in model-res space.

Drop the `original_coords`-based scaling block (lines 238-249) entirely:

```python
# intrinsics, depth, pixel_indices all live in model-res space
pts3d = reproject_pixels(
    self.depth,
    self.pixel_indices,
    self.extrinsics[:, :3, :],
    self.intrinsics,
)
return replace(self, points=pts3d)
```

Note: VGGTX `result.intrinsics` was already model-res before this fix, so `reproject()` was already broken for it (scaling was applied unnecessarily, shrinking fx/fy). Dropping the block fixes this too.

### 3. `BundleAdjustment` wrapper (`wrappers.py:202-203`)

After change 1, `raw["intrinsics"]` = model-res K for both VGGTX and VGGTOmega. The existing lookup is correct; no code change needed. The bug was in the source (wrong K stored under `"intrinsics"`), not in the lookup.

MapAnything uses a different raw_outputs structure (list of per-frame dicts); this lookup path is not exercised for MapAnything — confirmed safe.

### 4. `build_colmap` → `_rescale_reconstruction_to_original_dimensions` (base.py:411–506)

Already correct for model-res K input (the function name and `scale_x = orig_W / image_size[0]` math both assume model-res). No code change. After change 1, VGGTOmega's input to this function matches expectation.

For square mode (VGGTXCreator with non-1:1 aspect): the existing `_rescale` uses `image_size[0]` = padded model_W (518). `scale_x = orig_W/518`, which scales fx as if image_size[0]=518 corresponds to orig_W. This is wrong when padding was applied (orig content occupies <518 px in the padded image). Fix: use `original_coords` to derive the padded scale. **Defer to follow-up** — current users do not hit square+padding (1080×1920 has AR=1.78 not 1.0). Document the limitation in the docstring.

### 5. `_feedforward_to_tsdf_inputs` (mesh/utils.py:438–469)

Two issues:
- **RGB no-crop for VGGTOmega**: PIL.resize() ignores `original_coords` crop window → RGB/depth pixel misalignment when VGGTOmega applies center-crop (AR > 2.0 or AR < 0.5)
- **Error message** at line 449: missing `VGGTOmegaCreator`

`original_coords` semantics after all changes in this spec:
- **VGGTX** (`"crop"` mode, change 7): `[0, tl_y, orig_w, br_y, orig_w, orig_h]` in **original-image pixel space** — crop window applied before resize
- **VGGTOmega**: `[tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]` in **original-image pixel space** — AR-constrained crop window
- **MapAnything**: `[0, 0, model_w, model_h, orig_w, orig_h]` in **model-pixel space** — never crops

Detection: MapAnything is the only backend where `cr_x ≤ model_W + 1` (coords[2] = model_W exactly). VGGTX and VGGTOmega both have `cr_x = orig_w > model_W`. The heuristic is: **if `cr_x > W + 1`, apply crop before resize**.

RGB loading fix:

```python
rgbs = np.empty((N, H, W, 3), dtype=np.float32)
for i, path in enumerate(result.image_paths):
    img = PILImage.open(path).convert("RGB")
    if result.original_coords is not None:
        tl_x, tl_y, cr_x, cr_y = result.original_coords[i, :4]
        # VGGTX crop mode and VGGTOmega both store coords in original-image
        # pixels (cr_x = orig_w > model_W). Apply crop before resize.
        if cr_x > W + 1 or cr_y > H + 1:
            img = img.crop((float(tl_x), float(tl_y), float(cr_x), float(cr_y)))
    img = img.resize((W, H), PILImage.BILINEAR)
    rgbs[i] = np.asarray(img, dtype=np.float32) / 255.0
```

Update error message line 449 to read: `"MapAnythingCreator, VGGTXCreator, and VGGTOmegaCreator all populate world_points..."`.

### 6. `VGGTOmegaCreator` depth shape verification

`predictions["depth"].squeeze(0)` returns either `(N, H, W)` or `(N, H, W, 1)` depending on Omega head version. Upstream `unproject_depth_map_to_point_map` requires `(N, H, W, 1)` (it indexes `depth_map[..., 0]`). Our `unproject_and_filter_points` (`pointcloud/utils.py`) must accept either, or we must normalize at the boundary.

Audit `unproject_and_filter_points` for accepted depth shape. If it requires 4D, expand at `_postprocess` boundary:

```python
depth_arr = raw_outputs["depth"]
if depth_arr.ndim == 3:
    depth_arr = depth_arr[..., None]   # (N, H, W) → (N, H, W, 1)
pts3d, colors, pixel_indices = unproject_and_filter_points(
    depth=depth_arr, ...
)
```

If it accepts 3D, no change required — but document the contract.

### 7. `VGGTXCreator` — align preprocessing to upstream `"crop"` mode

**Background:**
- VGGTX trained with `load_and_preprocess_images(mode="crop")` — upstream default
- VGGT-SLAM/VGGT-SPARK use same: `load_and_preprocess_images(image_names)` (no mode arg → `"crop"`)
- Our `VGGTXCreator` default `resize_mode='max_size'` → `load_and_preprocess_images_ratio` — diverges from training
- Our `resize_mode='square'` → `load_and_preprocess_images_square` — **pads with black**, never in training data, degrades depth predictions near borders

**What `"crop"` mode does** for 1080×1920:
1. Resize width→518, maintain AR → 518×924 (height ÷14 rounded)
2. Center-crop height to 518 → 518×518
- Content lost: top/bottom `(924-518)/2 = 203` resized-px ≈ `203*(1920/924) ≈ 421` original-px

**`original_coords` under `"crop"` mode** (in original-image pixel space, same convention as VGGTOmega):

```
tl_y_orig = start_y_resized * (orig_h / new_h_resized)
br_y_orig = tl_y_orig + orig_h * (518 / new_h_resized)
original_coords[i] = [0, tl_y_orig, orig_w, br_y_orig, orig_w, orig_h]
```

For landscape/shallow images where `new_h ≤ 518` (no height crop):
```
original_coords[i] = [0, 0, orig_w, orig_h, orig_w, orig_h]
```

Both cases: `cr_x = orig_w > model_W` → TSDF RGB crop heuristic (change 5) applies correctly. **Unifies `original_coords` coordinate space between VGGTX and VGGTOmega** — both now store original-image pixel coords.

**Implementation:**
- Add `_compute_vggtx_crop_coords(image_paths, target_size=518) → np.ndarray` helper in `vggtx.py` (mirrors `_compute_omega_original_coords`)
- Replace `_preprocess` to always use `load_and_preprocess_images(mode="crop")` from upstream
- Remove `resize_mode` attribute from `VGGTXCreator` (no longer needed)
- Update `_rescale_reconstruction_to_original_dimensions` call: `image_size=(518, 518)` always (output is always 518×518 after crop mode for any portrait input; for landscape: `(518, new_h)`)

**Non-goal**: For images where height ≤ 518 after width-resize (wide/landscape, no crop occurs), behavior unchanged vs `max_size` default.

## Non-Goals

- Fixing square-mode COLMAP scaling for padded VGGTX inputs (moot — `square` mode removed)
- Migrating MapAnything to a different K convention
- Auto-deriving TSDF voxel_size from scene depth (separate audit item)

## Invariants

After all changes in this spec:

1. `result.intrinsics` is at model resolution: `result.intrinsics[i, 0, 2] < model_W` and `result.intrinsics[i, 1, 2] < model_H` for any well-formed prediction.
2. `raw_outputs["intrinsics"]` (when present) is at model resolution for all backends.
3. `raw_outputs["intrinsics_downsampled"]` (when present) is at model resolution — alias of `raw_outputs["intrinsics"]` for backward compat with `_raw_to_world_points` (base.py:281).
4. `_rescale_reconstruction_to_original_dimensions` is the single boundary where K scales from model to original; called only from `build_colmap`.
5. `reproject()` uses `result.intrinsics` directly — no `original_coords` scaling.
6. `original_coords[i, 2]` (cr_x): equal to `model_W` for MapAnything (no crop); equal to `orig_w > model_W` for VGGTX crop mode and VGGTOmega — unambiguous crop detection.
7. VGGTXCreator always uses upstream `"crop"` preprocessing — output matches VGGTX training distribution and VGGT-SLAM inference pipeline.

## Testing Strategy

- Unit: synthetic FeedforwardResult with `original_coords` where `cr_x > model_W` (simulating portrait image, VGGTOmega or VGGTX crop mode) → TSDF RGB loading applies crop, principal point cx < W
- Unit: VGGTOmega `_postprocess` returns model-res K — `result.intrinsics[i, 0, 2] < result.model_width`
- Unit: `reproject()` returns sane points with model-res K (no scaling applied)
- Unit: `_compute_vggtx_crop_coords` — portrait 1080×1920 → tl_y_orig ≈ 421, cr_y_orig ≈ 1499; landscape 1920×1080 → no crop → [0, 0, 1920, 1080, 1920, 1080]
- Unit: VGGTXCreator `_preprocess` output shape is (N, 3, 518, 518) for portrait; (N, 3, H≤518, 518) for landscape
- Integration: existing `test_tsdf_mesh_synthetic` continues to pass
- Integration: existing `test_pointcloud_to_mesh_returns_mesh_result` continues to pass

## Risk

- `reproject()` callers depending on old original-res-K behavior break. Audit callers: `wrappers.py:122-125` (LC dedup — uses `result.intrinsics` directly, no reproject call). No other callers of `reproject()` found at time of writing.
- Existing zarr-cached `FeedforwardResult` artifacts with original-res K become inconsistent — invalidate cache or re-run preprocessing.
