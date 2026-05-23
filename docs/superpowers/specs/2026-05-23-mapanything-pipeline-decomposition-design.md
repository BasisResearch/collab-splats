# MapAnything Pipeline Decomposition Design

**Date:** 2026-05-23
**Status:** Draft
**Context:** cu121 migration (torch 2.1.2+cu118 → torch 2.4+cu121)

---

## Problem

Two related bugs surfaced after the cu121 migration:

1. **`conf` NaN / `F.grid_sample` crash.** `MapAnythingCreator` calls `model.infer()`, which runs the model forward under bf16 autocast and immediately calls `postprocess_model_outputs_for_inference`. Inside `compute_multiview_depth_confidence`, `F.grid_sample(bf16_depth_map, float32_grid)` raises `RuntimeError: expected scalar type BFloat16 but found Float` on CUDA. The grid is float32 because `depthmap_to_camera_frame` uses `torch.arange(...).float()` internally, promoting depth inputs to float32 throughout the projection chain — but the depth map tensor itself is never promoted. Torch 2.4 enforces this dtype match strictly; 2.1.2 did not.

2. **`_patch_mapanything_torch_compat` is dead code.** Written to fix `tensor.any(dim=(1, 3))` on torch<2.4; we are now on torch 2.4, which supports tuple-dim `any()` natively. The patch's own docstring says "Delete this helper once the env moves to torch>=2.4."

Both stem from the same structural problem: `run_mapanything` delegates entirely to `model.infer()`, which bundles view validation, preprocessing, model forward, and postprocessing into one call. There is no seam to insert a dtype cast between the bf16 forward and the float32-requiring postprocessing.

---

## Goals

- Fix the `F.grid_sample` dtype mismatch so multiview confidence produces valid float32 scores.
- Remove `_patch_mapanything_torch_compat` and all its supporting imports (`inspect`, `linecache`, `sys`, `textwrap`).
- Split `model.infer()` across `_preprocess`, `_forward`, `_postprocess` so each hook has a single clear purpose and the dtype boundary is explicit.
- Remove the `run_mapanything` standalone function; forward logic lives in `MapAnythingCreator._forward`.
- Preserve bf16 autocast for the model forward (speed); postprocessing runs in float32 (correctness).
- Keep multiview confidence (`use_multiview_confidence=True`) working.

---

## Out of Scope

- MapAnything `model_factory` external backends (DA3, MonST3R, etc.) — future design.
- Changes to BA, loop closure, or any other pipeline stage.
- `_reproject_after_ba` for MapAnything: BA+MapAnything is untested. The current `_reproject_mapanything` implementation depends on postprocessed keys (`mask`, `depth_z`, `img_no_norm`) that `model.forward()` does not produce, making it silently wrong after the pipeline split. Both functions are removed; the base class `NotImplementedError` guards against BA+MapAnything until a correct implementation exists.
- VGGT-X (unaffected).

---

## Design

### Pipeline Split

`BaseFeedforwardCreator` defines a 5-step template method: `_load_model → _preprocess → _forward → _postprocess → build_colmap`. After this design:

| Template hook | Responsibility |
|---|---|
| `_load_model` | Load weights, move to device |
| `_preprocess` | Load images, validate views, run `preprocess_input_views_for_inference`, store `self._processed_views` (CPU) |
| `_forward` | Transfer `_processed_views` to device, call `model.forward()` under bf16 autocast, return raw bf16 outputs |
| `_postprocess` | Cast bf16 tensors to float32, run `postprocess_model_outputs_for_inference`, extract pts3d/colors/conf |

The dtype boundary is explicit: bf16 is confined to `_forward`; everything before and after is float32.

### `_processed_views` lifetime

`_preprocess` stores `self._processed_views` (MapAnything-format view dicts, CPU tensors). `_forward` transfers them to the model device in-place and passes them to `model.forward()`. `_postprocess` passes them as `input_views` to `postprocess_model_outputs_for_inference`, which needs them for `img_no_norm` reconstruction and intrinsics recovery.

### `run_mapanything` removal

`run_mapanything` was added to make inference parameters patchable in tests by wrapping `model.infer()`. With `_forward` calling `model.forward()` directly, tests inject `self._processed_views` and mock `model.forward` without needing a standalone wrapper. The function is removed from the module (it appears in the current module docstring and will be removed there too).

---

## Data Flow

```
_preprocess(image_dir)
  load_images
  validate_input_views_for_inference(views) → validated
  preprocess_input_views_for_inference(validated) → self._processed_views  [CPU]
  return (views, image_paths, original_coords)

_forward(model, views)
  transfer self._processed_views tensors → model.device  [in-place]
  with torch.no_grad():
    with torch.autocast(device_type, dtype=bfloat16, enabled=(device_type == 'cuda')):
      model.forward(self._processed_views, memory_efficient_inference=True,
                    minibatch_size=self.minibatch_size)
  return raw_outputs  [bf16]

_postprocess(raw_outputs)
  for pred in raw_outputs:
    pred["pts3d_cam"] = pred["pts3d_cam"].float()   # fix F.grid_sample dtype mismatch
    pred["pts3d"]     = pred["pts3d"].float()        # consistent float32 for numpy ops
  processed = postprocess_model_outputs_for_inference(
      raw_outputs, self._processed_views,
      apply_mask=True, mask_edges=True, apply_confidence_mask=True,
      use_multiview_confidence=self.use_multiview_confidence,
      confidence_percentile=self.confidence_percentile)
  pts3d, colors, extrinsics, intrinsics = collect_pts3d_from_outputs(processed)
  _conf  = stack([p["conf"][0] ...] for p in processed) if processed[0].get("conf")
  _images, _world_points extracted from processed
  voxel_downsample → build FeedforwardResult
```

---

## Imports

**Removed** (used only by compat patch): `inspect`, `linecache`, `sys`, `textwrap`.

**Added** (top-level per project convention):
```python
from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
```

---

## Error Handling

No new error handling. `model.forward()` raises on bad input; `postprocess_model_outputs_for_inference` raises on missing keys. The float32 cast is unconditional — missing keys propagate as KeyError, same as before.

---

## Testing

**Delete** (compat patch gone):
- `test_patch_mapanything_torch_compat_replaces_multi_dim_any`
- `test_patch_mapanything_torch_compat_is_idempotent`
- `test_patch_mapanything_torch_compat_raises_on_source_drift`
- `test_patch_mapanything_torch_compat_rebinds_aliased_imports`
- `test_load_model_invokes_torch_compat_patch`

**Rewrite**:
- `test_mapanything_forward_passes_inference_params` → `test_mapanything_forward_calls_model_forward`: verify `model.forward()` called with `self._processed_views`, `memory_efficient_inference=True`, `minibatch_size`.

**Add**:
- `test_mapanything_preprocess_sets_processed_views`: assert `validate_input_views_for_inference` and `preprocess_input_views_for_inference` are called; `self._processed_views` is set.
- `test_mapanything_postprocess_casts_bf16_to_float32`: inject raw outputs with bf16 `pts3d_cam`; assert tensor dtype is float32 when `postprocess_model_outputs_for_inference` is called. Regression guard for the torch 2.4 dtype fix.

**Update**:
- `test_mapanything_run_inference_smoke` (`@pytest.mark.gpu`): add assertions that `result.conf is not None` and `result.conf.isnan().any() == False`.

**Unchanged**: `test_mapanything_defaults`, `test_mapanything_is_feedforward_creator`, `test_mapanything_missing_image_dir_raises`, `test_mapanything_reconstruct_smoke`.

---

## File Changes Summary

| File | Change |
|---|---|
| `collab_splats/pointcloud/feedforward/mapanything.py` | Remove compat patch + 4 stdlib imports (`inspect`, `linecache`, `sys`, `textwrap`); remove `run_mapanything`; remove `_reproject_mapanything` and `_reproject_after_ba` (BA+MapAnything untested, implementations broken after pipeline split); add 3 `mapanything.utils.inference` imports; rewrite `_preprocess`, `_forward`, `_postprocess`; update module docstring |
| `tests/pointcloud/test_mapanything_creator.py` | Delete 5 compat-patch tests; rewrite 1 forward test; add 2 new unit tests; update 1 GPU smoke test |
