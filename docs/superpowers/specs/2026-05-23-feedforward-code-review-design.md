# Feedforward Module Code Review

**Date:** 2026-05-23  
**Scope:** `collab_splats/pointcloud/feedforward/` — base.py, vggtx.py, mapanything.py, __init__.py, wrappers.py

## Goals

1. Enforce coding principles: imports at top, inline block comments on every logical block, visual spacing between blocks, clear variable names
2. Remove or fix dead/missing code (unused function signatures, lost implementations)
3. Implement missing `_reproject_ba` for MapAnythingCreator (was lost in refactor, BA+MapAnything currently crashes)
4. Rename `_reproject_after_ba` → `_reproject_ba` across all files for consistency

## Inline Comment Standard

Every logical block of code gets a short comment explaining **what** it does at block level (not line-by-line). This applies to all methods across all files. Examples:

- preprocessing: `# Sort images by filename; reject non-image extensions`
- masking: `# Combine validity mask with positive-depth check`
- pose transform: `# refined world2cam → cam2world for re-projection into world frame`
- BA fields: `# Populate BA fields: subsampled world-point grid for track extraction`

Existing comments that already meet this standard are kept. Missing block comments are added.

---

## Changes by File

### base.py

**Typing imports**
- Remove `List`, `Tuple` from `from typing import ...`
- Replace `List[Path]` and `Tuple[int, int]` in `_rescale_reconstruction_to_original_dimensions` signature with `list[Path]` and `tuple[int, int]`
- `from __future__ import annotations` is already present so lowercase generics work on py3.11

**`_reproject_ba` (was `_reproject_after_ba`)**
- Rename to `_reproject_ba`
- Promote from `raise NotImplementedError` to `@abstractmethod`
- Current pattern fails silently at call time; `@abstractmethod` fails at instantiation — catches missing implementations earlier
- Removes the misleading docstring note that says "base class fallback"

**`_rescale_reconstruction_to_original_dimensions`**
- Keep `shared_camera`, `shift_point2d_to_original_res`, `verbose` — all intentionally useful
- No changes to logic

---

### vggtx.py

**`_reproject_ba` (rename)**
- `_reproject_after_ba` → `_reproject_ba`

**Local variable names in `_postprocess`**
- `_world_points`, `_conf`, `_images` → `world_points`, `conf`, `images`
- Underscore prefix on locals implies private class scope; these are just temporaries before packing into `FeedforwardResult`

---

### mapanything.py

**Implement `_reproject_ba` (critical bugfix)**

MapAnythingCreator has no `_reproject_ba`. This was implemented in commit `8eb4ec3` then lost in a subsequent refactor. BA + MapAnything currently crashes with `NotImplementedError` from the base class, after wasting GPU time on inference, track extraction, and BA itself.

Why `pts3d_cam` — not `pts3d` or depth maps:
- `pts3d` embeds the model's original predicted `camera_poses` (world-frame, pose-baked-in). After BA refines poses, `pts3d` is stale.
- `pts3d_cam` is a direct model prediction in camera frame — pose-independent. Apply new `cam2world` from refined extrinsics → correct world-space points.
- `depth_z` is just `pts3d_cam[..., 2]` (Z slice), used only as a validity mask (`depth_z > 0`). Not a full depth map suitable for unprojection.
- Unlike VGGT-X (which derives world pts from depth + intrinsics + extrinsics and needs to re-unproject when all three change), MapAnything's reprojection after BA is a single matrix multiply per frame.

`closed_form_pose_inverse` is already imported at module level.

Implementation:
```python
def _reproject_ba(
    self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for i, pred in enumerate(raw_outputs):
        pts3d_cam = pred["pts3d_cam"][0].cpu().numpy()          # (H, W, 3) camera-frame
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()
        combined_mask = mask & (depth_z > 0)

        # refined world2cam → cam2world for re-projection into world frame
        ext_4x4 = np.concatenate([extrinsics_3x4[i], [[0, 0, 0, 1]]], axis=0)
        cam2world = closed_form_pose_inverse(ext_4x4[None])[0]

        pts_flat = pts3d_cam[combined_mask]                      # (K, 3)
        pts_world = (cam2world[:3, :3] @ pts_flat.T + cam2world[:3, 3:]).T

        img_no_norm = pred["img_no_norm"][0].cpu().numpy()
        colors = (img_no_norm[combined_mask] * 255).astype(np.uint8)

        all_pts.append(pts_world.astype(np.float32))
        all_colors.append(colors)
    return np.concatenate(all_pts, axis=0), np.concatenate(all_colors, axis=0)
```

Note on masking: `self.base.raw_outputs[i]["pts3d_cam"]` is the float32-cast, **unmasked** tensor (confidence masking is applied to the `processed` shallow copy inside `_postprocess`, not to `raw_outputs`). The `mask & (depth_z > 0)` applied here matches the base validity mask used by `collect_pts3d_from_outputs`.

**`_preprocess` — open each image once**
- Current code: `[0, 0, model_w, model_h, PILImage.open(p).width, PILImage.open(p).height]` opens each image twice
- Fix: open once, read both dimensions

**`self._processed_views` — declare as dataclass field**
- Currently set as bare instance attribute in `_preprocess`, invisible to the dataclass
- Add `_processed_views: Any = field(default=None, init=False, repr=False)` to the class body

---

### wrappers.py

**`_reproject_ba` (rename only)**
- Two sites: `BundleAdjustment._apply_ba` call at line 146, and `LoopClosure._reproject_ba` delegation at line 196

---

### __init__.py

**`_raw_to_world_points` re-export comment**
- Add inline comment: re-exported for `wrappers.py` BundleAdjustment (which calls it directly on raw VGGT-X outputs outside the normal pipeline)

---

## What Is NOT Changed

- `_rescale_reconstruction_to_original_dimensions` logic — `shared_camera`, `shift_point2d_to_original_res`, `verbose` kept intact
- `FeedforwardResult` fields — no additions or removals
- `_verify_loop_candidate` — concrete in base, delegates to `extract_intermediate_features`; correct as-is
- `extract_intermediate_features` — already `@abstractmethod`; no changes
- `build_pycolmap_reconstruction` — no changes
- `collect_pts3d_from_outputs` (MapAnything) — no changes

---

## Key Invariant Preserved

After these changes, wrapping either backend with `BundleAdjustment` must succeed end-to-end:
- `VGGTXCreator`: `_reproject_ba` re-runs depth unprojection with refined extrinsics + intrinsics
- `MapAnythingCreator`: `_reproject_ba` transforms `pts3d_cam` to world via refined `cam2world`

Both follow the same interface: `(raw_outputs, extrinsics_3x4, intrinsics) → (pts3d, colors)`.
