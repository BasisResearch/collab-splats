# Pointcloud Density Alignment: VGGTX ↔ MapAnything

**Date:** 2026-05-23
**Status:** Approved
**Branch:** refactor/cu121

## Problem

VGGTXCreator produces ~500k points; MapAnythingCreator produces >3M. Two compounding causes:

1. **Confidence filtering asymmetry** — VGGTX defaults `conf_threshold=50.0` (top 50% survive); MapAnything defaults `confidence_percentile=35.0` (top 65% survive).
2. **No max_points cap on MapAnything** — VGGTX hard-caps at `max_points=500_000` via `randomly_limit_trues` on the (N, H, W) confidence mask; MapAnything only applies `voxel_downsample(adaptive=False)` with no ceiling.

Goal: both backends produce ~500k points by default. Sparser preferred.

## Root Cause Analysis

**VGGTX** (`unproject_and_filter_points`):
- Builds `conf_mask` (N, H, W) bool from percentile threshold on `depth_conf`
- If `sum(conf_mask) > max_points`: `conf_mask = randomly_limit_trues(conf_mask, max_points)`
- Indexes `pts3d[conf_mask]`, `colors[conf_mask]`; derives `pixel_indices = np.where(conf_mask)`

**MapAnything** (`_postprocess` → `collect_pts3d_from_outputs`):
- `postprocess_model_outputs_for_inference(..., apply_confidence_mask=True, confidence_percentile=35)` bakes confidence filtering into `pred["mask"]` per-frame
- `collect_pts3d_from_outputs` iterates frames, builds `combined_mask = pred["mask"] & (depth_z > 0)` per (H, W) frame, concatenates valid points
- NO cross-frame `randomly_limit_trues` — no ceiling on output count
- `pixel_indices` always `None` (never populated)

The per-frame `combined_mask` in MapAnything is structurally identical to VGGTX's `conf_mask` — same boolean mask pattern, just built per-frame then concatenated rather than as one (N, H, W) array.

## Design

### Shared: `BaseFeedforwardCreator`

Add one field:

```python
max_points: int = 500_000
```

No logic changes in base. Backends read `self.max_points`.

### VGGTX changes (`feedforward/vggtx.py`)

1. Change default: `conf_threshold: float = 50.0` → `conf_threshold: float = 35.0`
2. Both call sites of `unproject_and_filter_points`: replace hardcoded `max_points=500_000` with `max_points=self.max_points`
   - `_postprocess` (line ~292)
   - `_reproject_ba` (line ~365)

### MapAnything changes (`feedforward/mapanything.py`)

Replace `collect_pts3d_from_outputs` + separate `_images`/`_conf`/`_world_points` collection + `voxel_downsample` with a single unified loop that mirrors VGGTX's mask-then-subsample pattern:

```python
# Build per-frame masks and grids — mirrors VGGTX conf_mask pattern
masks, pts3d_grid, colors_grid = [], [], []
images_list, conf_list = [], []
extrinsics_list, intrinsics_list = [], []

for pred in processed:
    m = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)   # (H, W)
    dz = pred["depth_z"][0].squeeze(-1).cpu().numpy()            # (H, W)
    masks.append(m & (dz > 0))
    pts3d_grid.append(pred["pts3d"][0].cpu().numpy())            # (H, W, 3)
    colors_grid.append((pred["img_no_norm"][0].cpu().numpy() * 255).astype(np.uint8))  # (H, W, 3)
    images_list.append(pred["img_no_norm"][0].cpu().permute(2, 0, 1))  # (C, H, W) for BA
    if pred.get("conf") is not None:
        c = pred["conf"][0]
        conf_list.append(c[0] if c.ndim == 3 else c)
    cam2world = pred["camera_poses"][0].cpu().numpy()
    extrinsics_list.append(closed_form_pose_inverse(cam2world[None])[0][:3, :4])
    intrinsics_list.append(pred["intrinsics"][0].cpu().numpy())

combined_mask = np.stack(masks)           # (N, H, W) — cross-frame mask
stacked_pts3d = np.stack(pts3d_grid)      # (N, H, W, 3)
stacked_colors = np.stack(colors_grid)    # (N, H, W, 3)

# Apply cross-frame random subsampling — same as VGGTX randomly_limit_trues
if combined_mask.sum() > self.max_points:
    combined_mask = randomly_limit_trues(combined_mask, self.max_points)

pts3d = stacked_pts3d[combined_mask].astype(np.float32)
colors = stacked_colors[combined_mask]
pixel_indices = np.stack(np.where(combined_mask), axis=1).astype(np.int32)  # (P, 3)

_world_points = stacked_pts3d                    # full (N, H, W, 3) grid for BA
_images = torch.stack(images_list)              # (N, C, H, W)
_conf = torch.stack(conf_list) if conf_list else None
extrinsics = np.stack(extrinsics_list)          # (N, 3, 4)
intrinsics = np.stack(intrinsics_list)          # (N, 3, 3)
```

`FeedforwardResult` gains `pixel_indices` for MapAnything — enables semantic feature lifting (previously blocked by `pixel_indices=None`).

`collect_pts3d_from_outputs`, `voxel_downsample`, and the three separate post-loop passes are all removed.

Add import: `from vggt.utils.helper import randomly_limit_trues`

### What is NOT changed

- `postprocess_model_outputs_for_inference` call — unchanged, still applies confidence + edge masking
- `confidence_percentile=35.0` default on MapAnything — already aligned with revised VGGTX default
- `_reproject_ba` on MapAnything — uses `raw_outputs` directly, unaffected
- `_world_points` semantics — still full unmasked (N, H, W, 3) grid, just sourced from `stacked_pts3d`

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/pointcloud/feedforward/base.py` | Add `max_points: int = 500_000` to `BaseFeedforwardCreator` |
| `collab_splats/pointcloud/feedforward/vggtx.py` | `conf_threshold` default 50→35; hardcoded `500_000`→`self.max_points` (2 sites) |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Replace `collect_pts3d_from_outputs` + voxel path with unified loop + `randomly_limit_trues`; gain `pixel_indices` |

## Testing

- Existing tests pass — no behavioral regression for clouds under cap
- Integration test point-count assertions: update expected range to ~500k
- Manual spot-check: both backends on same scene → `len(result.pts3d)` comparable
- MapAnything `pixel_indices` now non-None — feature lifting path exercisable
