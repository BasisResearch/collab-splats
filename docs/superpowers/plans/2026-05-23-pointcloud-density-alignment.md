# Pointcloud Density Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align VGGTX and MapAnything output density to ~500k points by sharing `max_points` via base class and rewriting MapAnything's postprocess to use `randomly_limit_trues` on a stacked (N, H, W) mask — identical pattern to VGGTX.

**Architecture:** Add `max_points` field to `BaseFeedforwardCreator`. VGGTX reads it at both `unproject_and_filter_points` call sites. MapAnything replaces `collect_pts3d_from_outputs` + voxel path with a unified per-frame loop that stacks masks, applies `randomly_limit_trues`, and extracts `pixel_indices` (previously `None`).

**Tech Stack:** numpy, torch, vggt.utils.helper.randomly_limit_trues, mapanything

---

### Task 1: Add `max_points` to `BaseFeedforwardCreator`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:526-527`
- Test: `tests/pointcloud/test_feedforward_density.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pointcloud/test_feedforward_density.py
from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator

def test_base_has_max_points_field():
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert "max_points" in fields
    assert fields["max_points"].default == 500_000
```

- [ ] **Step 2: Run to verify it fails**

```
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_density.py::test_base_has_max_points_field -v
```
Expected: FAIL — `AssertionError`

- [ ] **Step 3: Add field to `BaseFeedforwardCreator` after `extractor_name`**

In `collab_splats/pointcloud/feedforward/base.py`, change:
```python
    camera_model: str = "PINHOLE"
    extractor_name: str | None = None
```
to:
```python
    camera_model: str = "PINHOLE"
    extractor_name: str | None = None
    max_points: int = 500_000
```

- [ ] **Step 4: Run to verify it passes**

```
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_density.py::test_base_has_max_points_field -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_feedforward_density.py
git commit -m "feat(feedforward): add max_points field to BaseFeedforwardCreator"
```

---

### Task 2: Wire `max_points` into VGGTXCreator + lower `conf_threshold` default

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py:127,286-293,359-366`
- Test: `tests/pointcloud/test_feedforward_density.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/pointcloud/test_feedforward_density.py
import dataclasses
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

def test_vggtx_conf_threshold_default():
    fields = {f.name: f for f in dataclasses.fields(VGGTXCreator)}
    assert fields["conf_threshold"].default == 35.0

def test_vggtx_inherits_max_points():
    fields = {f.name: f for f in dataclasses.fields(VGGTXCreator)}
    assert fields["max_points"].default == 500_000
```

- [ ] **Step 2: Run to verify they fail**

```
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_density.py::test_vggtx_conf_threshold_default tests/pointcloud/test_feedforward_density.py::test_vggtx_inherits_max_points -v
```
Expected: `test_vggtx_conf_threshold_default` FAIL (50.0 ≠ 35.0); `test_vggtx_inherits_max_points` PASS (already inherited).

- [ ] **Step 3: Change `conf_threshold` default**

In `collab_splats/pointcloud/feedforward/vggtx.py`, change:
```python
    conf_threshold: float = 50.0
```
to:
```python
    conf_threshold: float = 35.0
```
Also update the docstring line above it:
```python
        conf_threshold:       Depth confidence percentile cutoff (0–100).
                              Points whose confidence is below this percentile
                              are discarded.  35.0 = keep the top 65 %.
```

- [ ] **Step 4: Add `max_points=self.max_points` to both `unproject_and_filter_points` call sites**

In `_postprocess` (~line 286), change:
```python
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            conf_threshold=self.conf_threshold,
        )
```
to:
```python
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
        )
```

In `_reproject_ba` (~line 359), change:
```python
        pts3d, colors, _pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsics_3x4,
            intrinsic=intrinsics,
            conf_threshold=self.conf_threshold,
        )
```
to:
```python
        pts3d, colors, _pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsics_3x4,
            intrinsic=intrinsics,
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
        )
```

- [ ] **Step 5: Run tests**

```
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_density.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_feedforward_density.py
git commit -m "feat(feedforward): lower vggtx conf_threshold default to 35, wire max_points"
```

---

### Task 3: Rewrite MapAnything `_postprocess` with VGGTX-equivalent mask pattern

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Test: `tests/pointcloud/test_feedforward_density.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/pointcloud/test_feedforward_density.py
import dataclasses
from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

def test_mapanything_inherits_max_points():
    fields = {f.name: f for f in dataclasses.fields(MapAnythingCreator)}
    assert fields["max_points"].default == 500_000

def test_mapanything_no_collect_pts3d_import():
    import collab_splats.pointcloud.feedforward.mapanything as m
    assert not hasattr(m, "collect_pts3d_from_outputs")
```

- [ ] **Step 2: Run to verify they fail**

```
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_density.py::test_mapanything_no_collect_pts3d_import -v
```
Expected: FAIL — `collect_pts3d_from_outputs` still exists

- [ ] **Step 3: Rewrite `mapanything.py`**

**Remove** from imports:
- `import open3d as o3d` (no longer needed)
- `from ..utils import voxel_downsample` (no longer needed)

**Add** to imports:
```python
from vggt.utils.helper import randomly_limit_trues
```

**Remove** the module docstring line `collect_pts3d_from_outputs — extract pts3d/colors/extrinsics/intrinsics from processed outputs` and the entire `collect_pts3d_from_outputs` function (lines 41–82).

Update module docstring to remove the `collect_pts3d_from_outputs` line:
```python
"""MapAnything feedforward backend: inference utilities and creator.

Provides:
  MapAnythingCreator — feedforward creator using MapAnything depth + pose estimation
"""
```

**Replace** `_postprocess` body (lines 197–236) with:

```python
        # Build per-frame masks + point/color grids in one pass — mirrors VGGTX conf_mask pattern.
        # postprocess_model_outputs_for_inference already baked confidence + edge masking into
        # pred["mask"], so combined_mask = mask & (depth_z > 0) is the full validity mask.
        masks, pts3d_grid, colors_grid = [], [], []
        images_list, conf_list = [], []
        extrinsics_list, intrinsics_list = [], []

        for pred in processed:
            m = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)       # (H, W)
            dz = pred["depth_z"][0].squeeze(-1).cpu().numpy()                # (H, W)
            masks.append(m & (dz > 0))
            pts3d_grid.append(pred["pts3d"][0].cpu().numpy())                # (H, W, 3)
            colors_grid.append(
                (pred["img_no_norm"][0].cpu().numpy() * 255).astype(np.uint8)
            )                                                                  # (H, W, 3)
            images_list.append(pred["img_no_norm"][0].cpu().permute(2, 0, 1))  # (C, H, W)
            if pred.get("conf") is not None:
                c = pred["conf"][0]
                conf_list.append(c[0] if c.ndim == 3 else c)
            cam2world = pred["camera_poses"][0].cpu().numpy()
            extrinsics_list.append(closed_form_pose_inverse(cam2world[None])[0][:3, :4])
            intrinsics_list.append(pred["intrinsics"][0].cpu().numpy())

        combined_mask = np.stack(masks)           # (N, H, W) bool
        stacked_pts3d = np.stack(pts3d_grid)      # (N, H, W, 3)
        stacked_colors = np.stack(colors_grid)    # (N, H, W, 3)

        # Apply cross-frame random subsampling — same as VGGTX randomly_limit_trues on conf_mask
        if int(combined_mask.sum()) > self.max_points:
            combined_mask = randomly_limit_trues(combined_mask, self.max_points)

        pts3d = stacked_pts3d[combined_mask].astype(np.float32)
        colors = stacked_colors[combined_mask]
        pixel_indices = np.stack(np.where(combined_mask), axis=1).astype(np.int32)  # (P, 3)

        _world_points = stacked_pts3d                        # full (N, H, W, 3) grid for BA
        _images = torch.stack(images_list)                   # (N, C, H, W)
        _conf = torch.stack(conf_list) if conf_list else None
        extrinsics = np.stack(extrinsics_list)               # (N, 3, 4)
        intrinsics = np.stack(intrinsics_list)               # (N, 3, 3)

        # Convert extrinsics to 4×4 homogeneous form
        extrinsics_4x4 = _extrinsics_3x4_to_4x4(extrinsics)

        return FeedforwardResult(
            pts3d=pts3d,
            colors=colors,
            pixel_indices=pixel_indices,
            extrinsics=extrinsics_4x4,
            intrinsics=intrinsics,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=_images,
            conf=_conf,
            world_points=_world_points,
        )
```

- [ ] **Step 4: Run all density tests**

```
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_density.py -v
```
Expected: all PASS

- [ ] **Step 5: Run full test suite**

```
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration
```
Expected: no regressions

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/test_feedforward_density.py
git commit -m "feat(feedforward): rewrite MapAnything postprocess with VGGTX-equivalent mask + randomly_limit_trues; gain pixel_indices"
```
