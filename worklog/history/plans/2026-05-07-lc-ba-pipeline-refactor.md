# LC/BA Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `LoopClosure` and `BundleAdjustment` into composable wrapper classes, removing `use_ba` / `enable_loop_closure` flags from creator constructors.

**Architecture:** `LoopClosure` is a proxy wrapper — it forwards `load_model/setup_inference/postprocess/build_colmap` to the inner creator and overrides only `run_inference()` with the submap LC loop. `BundleAdjustment` duck-types against this proxy interface (no `isinstance` checks needed). Both live in `collab_splats/pointcloud/wrappers.py`.

**Tech Stack:** Python dataclasses, pycolmap, numpy, torch (lazy imports for heavy deps), pytest with mocks for GPU-free unit tests.

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/pointcloud/feedforward.py` | Add 3 fields to `FeedforwardResult`; add `_reproject_after_ba` to `VGGTXCreator` + `MapAnythingCreator`; remove BA blocks; strip LC fields/methods from `BaseFeedforwardCreator` |
| `collab_splats/pointcloud/bundle_adjustment.py` | Add `BundleAdjustmentConfig` dataclass |
| `collab_splats/pointcloud/wrappers.py` | **New.** `LoopClosure` + `BundleAdjustment` wrapper classes |
| `collab_splats/pointcloud/__init__.py` | Export new classes; add `make_creator()` factory |
| `tests/pointcloud/test_wrappers.py` | **New.** Unit tests for `LoopClosure` and `BundleAdjustment` |
| `tests/pointcloud/test_loop_closure.py` | Remove any `enable_loop_closure=True` usage if present |

---

## Task 1: `BundleAdjustmentConfig` dataclass

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`
- Test: `tests/pointcloud/test_wrappers.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pointcloud/test_wrappers.py
from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig

def test_bundle_adjustment_config_defaults():
    cfg = BundleAdjustmentConfig()
    assert cfg.max_reproj_error == 4.0
    assert cfg.lm_steps == 40
    assert cfg.shared_camera is False
    assert cfg.min_inliers_per_frame == 64

def test_bundle_adjustment_config_custom():
    cfg = BundleAdjustmentConfig(max_reproj_error=2.0, lm_steps=20)
    assert cfg.max_reproj_error == 2.0
    assert cfg.lm_steps == 20
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_bundle_adjustment_config_defaults -v
```
Expected: `ImportError` or `AttributeError`

- [ ] **Step 3: Add `BundleAdjustmentConfig` to `bundle_adjustment.py`**

Add after the module docstring and imports, before `extract_tracks_vggsfm`:

```python
from dataclasses import dataclass

@dataclass
class BundleAdjustmentConfig:
    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
```

Also add `"BundleAdjustmentConfig"` to `__all__`.

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_wrappers.py
git commit -m "feat(pointcloud): add BundleAdjustmentConfig dataclass"
```

---

## Task 2: Extend `FeedforwardResult` with BA fields

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py` (lines 28–47)
- Test: `tests/pointcloud/test_wrappers.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/pointcloud/test_wrappers.py
import numpy as np
from pathlib import Path
from collab_splats.pointcloud.feedforward import FeedforwardResult

def _make_ff_result(**overrides):
    defaults = dict(
        pts3d=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        image_paths=[Path("a.jpg"), Path("b.jpg")],
        original_coords=np.zeros((2, 6), dtype=np.float32),
        model_width=224,
        model_height=224,
    )
    defaults.update(overrides)
    return FeedforwardResult(**defaults)

def test_feedforward_result_new_fields_default_none():
    r = _make_ff_result()
    assert r.images is None
    assert r.conf is None
    assert r.world_points is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_feedforward_result_new_fields_default_none -v
```
Expected: `TypeError` (unexpected keyword argument) or `AttributeError`

- [ ] **Step 3: Add fields to `FeedforwardResult`**

In `feedforward.py`, the `FeedforwardResult` dataclass ends at line ~47 with `model_height: int`. Append three new optional fields after `model_height`:

```python
    # Populated by feedforward creators always; consumed by BundleAdjustment wrapper.
    images: "torch.Tensor | None" = None        # (N, 3, H, W) normalised RGB for track extraction
    conf: "torch.Tensor | None" = None           # (N, H, W) confidence scores
    world_points: "np.ndarray | None" = None     # (N, H, W, 3) world-space points per pixel
```

Note: fields with defaults must come after fields without defaults in a dataclass. `images/conf/world_points` are optional so they go last — this is already correct since `pts3d`, `colors`, `extrinsics`, `intrinsics`, `image_paths`, `original_coords`, `model_width`, `model_height` have no defaults.

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_wrappers.py
git commit -m "feat(feedforward): add images/conf/world_points fields to FeedforwardResult"
```

---

## Task 3: `VGGTXCreator` — populate new fields, add `_reproject_after_ba`, remove `use_ba`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py` (`VGGTXCreator` class, ~lines 942–1155)

This task has three changes to `VGGTXCreator._postprocess()`:
1. Always populate `images`, `conf`, `world_points` in the returned `FeedforwardResult`
2. Add `_reproject_after_ba(raw_outputs, ext_3x4, intr)` method
3. Remove `if self.use_ba:` block and `use_ba: bool` field

- [ ] **Step 1: Write failing test**

```python
# tests/pointcloud/test_wrappers.py — full header (replace placeholder imports at top of file)
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

def test_vggtx_postprocess_populates_ba_fields():
    """VGGTXCreator._postprocess() must populate images/conf/world_points."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    creator = VGGTXCreator.__new__(VGGTXCreator)
    creator.conf_threshold = 1.0
    creator.use_global_alignment = False
    creator.image_paths = [Path("a.jpg"), Path("b.jpg")]
    creator.original_coords = np.zeros((2, 6), dtype=np.float32)

    N, H, W = 2, 8, 8
    raw_outputs = {
        "depth": np.ones((N, H, W, 1), dtype=np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": torch.zeros(N, 3, H, W),
        "extrinsic": np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
    }

    with patch("collab_splats.pointcloud.feedforward.unproject_and_filter_points",
               return_value=(np.zeros((5, 3), dtype=np.float32),
                             np.zeros((5, 3), dtype=np.uint8))):
        result = creator._postprocess(raw_outputs)

    assert result.images is not None
    assert result.conf is not None
    assert result.world_points is not None
    assert result.images.shape[0] == N
    assert result.conf.shape == (N, H, W)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_vggtx_postprocess_populates_ba_fields -v
```
Expected: FAIL (result.images is None)

- [ ] **Step 3: Update `VGGTXCreator._postprocess()`**

In `feedforward.py`, find `VGGTXCreator._postprocess()`. After the `pts3d, colors = unproject_and_filter_points(...)` call and the `model_h`/`model_w` lines (around line 1075), replace the `if self.use_ba:` block (lines ~1078–1106) with population logic:

```python
        # Always populate BA-required fields for BundleAdjustment wrapper.
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)  # (N, H*W, 3)
        if world_pts_flat is not None:
            _world_points = world_pts_flat.reshape(
                world_pts_flat.shape[0], model_h, model_w, 3
            )
        else:
            _world_points = None
        import torch as _torch
        _conf = _torch.from_numpy(raw_outputs["depth_conf"])   # (N, H, W)
        _images = raw_outputs["images"]                         # (N, 3, H, W) tensor
```

Then update the `return FeedforwardResult(...)` at the end to include the new fields:

```python
        return FeedforwardResult(
            pts3d=pts3d,
            colors=colors,
            extrinsics=extrinsic_4x4,
            intrinsics=intrinsic,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=_images,
            conf=_conf,
            world_points=_world_points,
        )
```

Remove `use_ba: bool = False` field from the `VGGTXCreator` dataclass definition.

- [ ] **Step 4: Add `_reproject_after_ba` to `VGGTXCreator`**

Add this method to `VGGTXCreator` (after `_postprocess`):

```python
    def _reproject_after_ba(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from ._vggtx import unproject_and_filter_points
        return unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsics_3x4,
            intrinsic=intrinsics,
            conf_threshold=self.conf_threshold,
        )
```

- [ ] **Step 5: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py -v
```
Expected: all PASSED

- [ ] **Step 6: Run existing tests to catch regressions**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v --ignore=tests/pointcloud/test_wrappers.py
```
Expected: all PASSED (test_sim3_pose_graph, test_loop_closure)

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_wrappers.py
git commit -m "refactor(feedforward): VGGTXCreator populate BA fields, add _reproject_after_ba, remove use_ba"
```

---

## Task 4: `MapAnythingCreator` — populate new fields, add `_reproject_after_ba`, remove `use_ba`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py` (`MapAnythingCreator` class, ~lines 780–941)

- [ ] **Step 1: Write failing test**

`MapAnythingCreator._postprocess()` has deep open3d/torch deps making unit-level mocking fragile. Test the two observable contracts: field removal (`use_ba` gone) and field population (checked via a lightweight dataclass check).

```python
# Append to tests/pointcloud/test_wrappers.py
def test_mapanything_no_use_ba_field():
    """MapAnythingCreator must not have use_ba after refactor."""
    import dataclasses
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    field_names = {f.name for f in dataclasses.fields(MapAnythingCreator)}
    assert "use_ba" not in field_names

def test_mapanything_has_reproject_after_ba():
    """MapAnythingCreator must implement _reproject_after_ba."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    assert hasattr(MapAnythingCreator, "_reproject_after_ba")
```

Note: `MapAnythingCreator._postprocess` has deep open3d/torch deps — the unit test checks field removal. A smoke test via notebook/integration covers field population.

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_mapanything_no_use_ba_field tests/pointcloud/test_wrappers.py::test_mapanything_has_reproject_after_ba -v
```
Expected: FAIL (`use_ba` still in fields, `_reproject_after_ba` missing)
Expected: FAIL (use_ba still in fields)

- [ ] **Step 3: Update `MapAnythingCreator._postprocess()`**

In `feedforward.py`, find `MapAnythingCreator._postprocess()`. Replace the `if self.use_ba:` block (the block that imports `_reproject_mapanything`, extracts `images`, runs BA) with population-only logic:

```python
        # Always populate for BundleAdjustment wrapper.
        import torch as _torch
        _images = _torch.stack([p["img_no_norm"][0].cpu().permute(2, 0, 1) for p in raw_outputs])
        if raw_outputs[0].get("conf") is not None:
            conf_list = [p["conf"][0] for p in raw_outputs]
            _conf = _torch.stack([c[0] if c.ndim == 3 else c for c in conf_list])
        else:
            _conf = None
        _world_points = np.stack(
            [p["pts3d"][0].cpu().numpy() for p in raw_outputs]
        )  # (N, H, W, 3)
```

Keep the voxel downsample block below unchanged. Update the `return FeedforwardResult(...)` to add:

```python
        return FeedforwardResult(
            pts3d=pts3d,
            colors=colors,
            extrinsics=extrinsic_4x4,
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

Remove `use_ba: bool = False` from the `MapAnythingCreator` dataclass.

- [ ] **Step 4: Add `_reproject_after_ba` to `MapAnythingCreator`**

```python
    def _reproject_after_ba(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from ._mapanything import _reproject_mapanything
        return _reproject_mapanything(raw_outputs, extrinsics_3x4)
```

- [ ] **Step 5: Run all tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v
```
Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_wrappers.py
git commit -m "refactor(feedforward): MapAnythingCreator populate BA fields, add _reproject_after_ba, remove use_ba"
```

---

## Task 5: `BundleAdjustment` wrapper class

**Files:**
- Create: `collab_splats/pointcloud/wrappers.py`
- Test: `tests/pointcloud/test_wrappers.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/pointcloud/test_wrappers.py
import dataclasses
from unittest.mock import MagicMock, patch

def _make_mock_creator(ff_result):
    """Returns a mock that duck-types as BaseFeedforwardCreator."""
    m = MagicMock()
    m.outputs = ff_result
    m.raw_outputs = {}
    m._reproject_after_ba.return_value = (ff_result.pts3d, ff_result.colors)
    return m

def test_bundle_adjustment_raises_if_images_none():
    from collab_splats.pointcloud.wrappers import BundleAdjustment
    result = _make_ff_result()  # images=None by default
    mock_creator = _make_mock_creator(result)

    ba = BundleAdjustment(mock_creator)
    with pytest.raises(ValueError, match="images"):
        ba.reconstruct("/fake/dir", "/fake/out")


def test_bundle_adjustment_calls_extract_tracks_and_run_ba():
    from collab_splats.pointcloud.wrappers import BundleAdjustment
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig
    import pytest

    N, H, W = 2, 8, 8
    result = _make_ff_result(
        images=torch.zeros(N, 3, H, W),
        conf=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )
    mock_creator = _make_mock_creator(result)
    mock_creator.build_colmap.return_value = MagicMock()  # PointcloudResult

    fake_tracks = np.zeros((N, 10, 2), dtype=np.float32)
    fake_vis = np.ones((N, 10), dtype=np.float32)
    fake_pts = np.zeros((10, 3), dtype=np.float32)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.wrappers.extract_tracks_vggsfm",
               return_value=(fake_tracks, fake_vis, fake_pts)) as mock_tracks, \
         patch("collab_splats.pointcloud.wrappers.run_bundle_adjustment",
               return_value=(fake_pts, refined_ext, refined_intr)) as mock_ba:
        ba = BundleAdjustment(mock_creator)
        ba.reconstruct("/fake/dir", "/fake/out")

    mock_tracks.assert_called_once()
    mock_ba.assert_called_once()
    mock_creator._reproject_after_ba.assert_called_once_with({}, refined_ext, refined_intr)
    mock_creator.build_colmap.assert_called_once()


def test_bundle_adjustment_config_passed_to_run_ba():
    from collab_splats.pointcloud.wrappers import BundleAdjustment
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig

    N, H, W = 2, 8, 8
    result = _make_ff_result(
        images=torch.zeros(N, 3, H, W),
        conf=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )
    mock_creator = _make_mock_creator(result)
    mock_creator.build_colmap.return_value = MagicMock()

    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.wrappers.extract_tracks_vggsfm",
               return_value=(np.zeros((N,10,2)), np.ones((N,10)), np.zeros((10,3)))), \
         patch("collab_splats.pointcloud.wrappers.run_bundle_adjustment",
               return_value=(np.zeros((10,3)), refined_ext, refined_intr)) as mock_ba:
        cfg = BundleAdjustmentConfig(max_reproj_error=2.0, lm_steps=10)
        BundleAdjustment(mock_creator, config=cfg).reconstruct("/a", "/b")

    _, kwargs = mock_ba.call_args
    assert kwargs["max_reproj_error"] == 2.0
    assert kwargs["lm_steps"] == 10
```

Also add `import pytest` to the top of the test file.

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_bundle_adjustment_raises_if_images_none -v
```
Expected: `ImportError` (module doesn't exist yet)

- [ ] **Step 3: Create `collab_splats/pointcloud/wrappers.py`**

```python
# collab_splats/pointcloud/wrappers.py
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BasePointcloudCreator, PointcloudResult
from .bundle_adjustment import BundleAdjustmentConfig, extract_tracks_vggsfm, run_bundle_adjustment
from .feedforward import FeedforwardResult
from .loop_closure import LoopClosureConfig

__all__ = ["BundleAdjustment", "LoopClosure"]


class BundleAdjustment(BasePointcloudCreator):
    """Wrapper that runs bundle adjustment after any feedforward creator (or LoopClosure).

    Expects the wrapped creator to populate FeedforwardResult.images/conf/world_points.
    Both BaseFeedforwardCreator and LoopClosure satisfy the required duck-type interface.
    """

    def __init__(
        self,
        base: Any,  # BaseFeedforwardCreator | LoopClosure — duck-typed
        config: BundleAdjustmentConfig | None = None,
    ) -> None:
        self.base = base
        self.config = config or BundleAdjustmentConfig()

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        self.base.load_model()
        self.base.setup_inference(image_dir)
        self.base.run_inference()
        self.base.postprocess()
        self._apply_ba()
        return self.base.build_colmap(output_dir)

    def _apply_ba(self) -> None:
        result: FeedforwardResult = self.base.outputs
        if result.images is None:
            raise ValueError(
                f"{type(self.base).__name__} did not populate FeedforwardResult.images. "
                "Creator's _postprocess() must always set images/conf/world_points."
            )

        tracks, vis_scores, pts3d_kp = extract_tracks_vggsfm(
            result.images, result.conf, result.world_points
        )
        extrinsics_3x4 = result.extrinsics[:, :3, :]  # (N, 3, 4)
        _, refined_ext_3x4, refined_intr = run_bundle_adjustment(
            pts3d_kp,
            extrinsics_3x4,
            result.intrinsics,
            tracks,
            vis_scores,
            image_size=(result.model_height, result.model_width),
            **dataclasses.asdict(self.config),
        )

        pts3d, colors = self.base._reproject_after_ba(
            self.base.raw_outputs, refined_ext_3x4, refined_intr
        )

        n = refined_ext_3x4.shape[0]
        bottom = np.tile([[0, 0, 0, 1]], (n, 1, 1)).astype(np.float32)
        refined_ext_4x4 = np.concatenate([refined_ext_3x4, bottom], axis=1)

        self.base.outputs = dataclasses.replace(
            result,
            pts3d=pts3d,
            colors=colors,
            extrinsics=refined_ext_4x4,
            intrinsics=refined_intr,
        )
```

Note: `LoopClosure` class will be added to this file in Task 6.

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py -k "bundle_adjustment" -v
```
Expected: all 3 BA tests PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py tests/pointcloud/test_wrappers.py
git commit -m "feat(pointcloud): add BundleAdjustment wrapper class"
```

---

## Task 6: `LoopClosure` wrapper class

**Files:**
- Modify: `collab_splats/pointcloud/wrappers.py` (append `LoopClosure` class)
- Test: `tests/pointcloud/test_wrappers.py`

`LoopClosure` is a **proxy wrapper**: it forwards `load_model`, `setup_inference`, `postprocess`, `build_colmap`, `outputs`, `raw_outputs`, `_reproject_after_ba` to the inner creator, and overrides `run_inference()` with the submap LC loop. This means `BundleAdjustment` can wrap `LoopClosure` and call the same duck-type interface.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/pointcloud/test_wrappers.py
def test_loop_closure_constructor_defaults():
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    assert lc.base is mock_base
    assert isinstance(lc.config, LoopClosureConfig)


def test_loop_closure_constructor_custom_config():
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig

    cfg = LoopClosureConfig(submap_size=10)
    mock_base = MagicMock()
    lc = LoopClosure(mock_base, config=cfg)
    assert lc.config.submap_size == 10


def test_loop_closure_forwards_load_model():
    from collab_splats.pointcloud.wrappers import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.load_model()
    mock_base.load_model.assert_called_once()


def test_loop_closure_forwards_setup_inference():
    from collab_splats.pointcloud.wrappers import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.setup_inference(Path("/fake"))
    mock_base.setup_inference.assert_called_once_with(Path("/fake"))


def test_loop_closure_forwards_postprocess_and_build_colmap():
    from collab_splats.pointcloud.wrappers import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.postprocess()
    mock_base.postprocess.assert_called_once()
    lc.build_colmap(Path("/out"))
    mock_base.build_colmap.assert_called_once_with(Path("/out"))


def test_loop_closure_run_inference_falls_back_to_base_when_too_few_frames():
    """When fewer frames than submap_size, falls back to base.run_inference()."""
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig
    import torch

    mock_base = MagicMock()
    mock_base.views = torch.zeros(3, 3, 8, 8)  # 3 frames

    cfg = LoopClosureConfig(submap_size=20)
    lc = LoopClosure(mock_base, config=cfg)
    lc.run_inference()
    mock_base.run_inference.assert_called_once()


def test_bundle_adjustment_wraps_loop_closure():
    """BundleAdjustment can wrap LoopClosure without isinstance checks."""
    from collab_splats.pointcloud.wrappers import BundleAdjustment, LoopClosure

    N, H, W = 2, 8, 8
    result = _make_ff_result(
        images=torch.zeros(N, 3, H, W),
        conf=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )
    inner_creator = MagicMock()
    lc = LoopClosure(inner_creator)
    lc.base.outputs = result
    lc.base.raw_outputs = {}
    lc.base._reproject_after_ba.return_value = (result.pts3d, result.colors)
    lc.base.build_colmap.return_value = MagicMock()

    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.wrappers.extract_tracks_vggsfm",
               return_value=(np.zeros((N,10,2)), np.ones((N,10)), np.zeros((10,3)))), \
         patch("collab_splats.pointcloud.wrappers.run_bundle_adjustment",
               return_value=(np.zeros((10,3)), refined_ext, refined_intr)):
        BundleAdjustment(lc).reconstruct("/a", "/b")

    # BA called build_colmap on the creator (via lc proxy)
    lc.base.build_colmap.assert_called_once()
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_loop_closure_constructor_defaults -v
```
Expected: `ImportError` (LoopClosure not defined yet)

- [ ] **Step 3: Add `LoopClosure` to `wrappers.py`**

Append to `collab_splats/pointcloud/wrappers.py` after the `BundleAdjustment` class:

```python
class LoopClosure(BasePointcloudCreator):
    """Proxy wrapper that adds submap-based loop closure to a feedforward creator.

    Forwards load_model / setup_inference / postprocess / build_colmap / outputs /
    raw_outputs / _reproject_after_ba to the inner creator. Overrides run_inference()
    with the submap LC loop so BundleAdjustment can wrap this transparently.
    """

    def __init__(
        self,
        base: Any,  # BaseFeedforwardCreator — duck-typed
        config: LoopClosureConfig | None = None,
    ) -> None:
        self.base = base
        self.config = config or LoopClosureConfig()

    # ------------------------------------------------------------------ proxy

    def load_model(self, device: str | None = None) -> None:
        self.base.load_model(device)

    def setup_inference(self, image_dir: Path) -> None:
        self.base.setup_inference(image_dir)

    def postprocess(self, **kwargs: Any) -> None:
        self.base.postprocess(**kwargs)

    def build_colmap(self, output_dir: Path) -> PointcloudResult:
        return self.base.build_colmap(output_dir)

    @property
    def outputs(self) -> FeedforwardResult:
        return self.base.outputs

    @outputs.setter
    def outputs(self, value: FeedforwardResult) -> None:
        self.base.outputs = value

    @property
    def raw_outputs(self) -> Any:
        return self.base.raw_outputs

    def _reproject_after_ba(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.base._reproject_after_ba(raw_outputs, extrinsics_3x4, intrinsics)

    # ----------------------------------------------------------------- LC core

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        self.load_model()
        self.setup_inference(image_dir)
        self.run_inference()
        self.postprocess()
        return self.build_colmap(output_dir)

    def run_inference(self, **kwargs: Any) -> None:
        if self._enough_frames():
            self._run_lc_loop(**kwargs)
        else:
            self.base.run_inference(**kwargs)

    def _enough_frames(self) -> bool:
        views = self.base.views
        n = views.shape[0] if hasattr(views, "shape") else len(views)
        return n >= self.config.submap_size

    def _run_lc_loop(self, **kwargs: Any) -> None:
        """Submap-based inference with loop closure detection and Sim3 pose graph correction.

        Extracted from BaseFeedforwardCreator._run_loop_closure_inference().
        Sets self.base.raw_outputs on completion.
        """
        import math
        import torch
        import logging
        from rich.console import Console
        from tqdm import tqdm
        from collab_splats.pointcloud.feedforward import _raw_to_world_points
        from collab_splats.pointcloud.loop_closure import Submap, ImageRetrieval
        from collab_splats.pointcloud.loop_closure.submap import assert_world_to_cam
        from collab_splats.pointcloud.loop_closure.closure import (
            run_sim3_pose_graph_optimization,
            merge_submap_outputs,
            translation_jump_check,
        )

        console = Console()
        cfg = self.config
        K, O = cfg.submap_size, cfg.submap_overlap
        step = max(1, K - O)
        views = self.base.views
        N = views.shape[0] if hasattr(views, "shape") else len(views)
        device = str(next(self.base.model.parameters()).device)

        try:
            retrieval = ImageRetrieval(device=device)
            self.base._lc_retrieval = retrieval
        except Exception as e:
            logging.getLogger(__name__).warning(
                "DINO-SALAD failed to load (%s) — skipping loop closure", e
            )
            self.base.raw_outputs = self.base._forward(self.base.model, views, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        submaps: list[Submap] = []
        lc_submaps: list[Submap] = []
        all_loop_candidates: list = []
        n_submaps = math.ceil(max(1, N - O) / step)
        loops_found = 0
        verified = 0

        console.log(f"Loop closure: {N} frames → {n_submaps} submaps (size={K}, overlap={O})")

        with tqdm(total=n_submaps, desc="Loop closure", unit="submap") as pbar:
            for wi, start in enumerate(range(0, N, step)):
                end = min(start + K, N)
                window = views[start:end]
                k = window.shape[0] if hasattr(window, "shape") else len(window)

                with torch.no_grad():
                    raw = self.base._forward(self.base.model, window, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                ext_3x4 = raw["extrinsic"]
                bottom = np.tile([0, 0, 0, 1], (k, 1)).reshape(k, 1, 4).astype(np.float32)
                poses_4x4 = np.concatenate([ext_3x4, bottom], axis=1)

                assert_world_to_cam(poses_4x4)

                intr_key = "intrinsics" if "intrinsics" in raw else "intrinsic"
                intrinsics = raw.get(intr_key, np.tile(np.eye(3), (k, 1, 1)).astype(np.float32))

                frames_cpu = window.cpu() if hasattr(window, "cpu") else torch.zeros(k, 3, 1, 1)
                ret_vecs = retrieval.embed_frames(frames_cpu)

                wp, wp_conf = _raw_to_world_points(raw)
                submap = Submap(
                    submap_id=wi,
                    frames=frames_cpu,
                    poses=poses_4x4,
                    intrinsics=intrinsics,
                    retrieval_vectors=ret_vecs,
                    image_paths=list(self.base.image_paths[start:end]),
                    raw_outputs=raw,
                    frame_start=start,
                    world_points=wp,
                    world_points_conf=wp_conf,
                )

                past_for_lc = submaps[: max(0, len(submaps) - cfg.min_submap_gap)]
                loop_matches = retrieval.find_loop_closures(
                    submap, past_for_lc, cfg.lc_threshold_l2, cfg.max_loops_per_submap,
                    nms_frame_distance=cfg.nms_frame_distance,
                )

                for match in loop_matches:
                    q_frame = frames_cpu[match.query_frame_idx]
                    d_submap = submaps[match.detected_submap_id]
                    d_frame = d_submap.frames[match.detected_frame_idx]
                    verify_ok, lc_poses = self.base._verify_loop_candidate(q_frame, d_frame)
                    if not verify_ok:
                        console.log(
                            f"  ✗ Loop rejected (verify ratio): "
                            f"submap {match.query_submap_id} → {match.detected_submap_id}"
                            f"  dist={match.similarity_score:.3f}"
                        )
                    if verify_ok and lc_poses is None:
                        console.log(
                            f"  ✗ Loop skipped (no joint poses): "
                            f"submap {match.query_submap_id} → {match.detected_submap_id}"
                        )
                        verify_ok = False
                    if verify_ok:
                        lc_rel = (
                            np.linalg.inv(lc_poses[1].astype(np.float64))
                            @ lc_poses[0].astype(np.float64)
                        ).astype(np.float32)
                        jump_ok, jump_ratio = translation_jump_check(
                            submaps + [submap],
                            query_idx=len(submaps),
                            query_frame=match.query_frame_idx,
                            detected_idx=match.detected_submap_id,
                            detected_frame=match.detected_frame_idx,
                            lc_relative_pose=lc_rel,
                        )
                        if not jump_ok:
                            console.log(
                                f"  ✗ Loop rejected (jump ratio={jump_ratio:.2f}): "
                                f"submap {match.query_submap_id} → {match.detected_submap_id}"
                            )
                        else:
                            match.accepted = True
                            verified += 1
                            loops_found += 1
                            console.log(
                                f"  ↩ Loop: submap {match.query_submap_id} → {match.detected_submap_id}"
                                f"  dist={match.similarity_score:.3f} jump={jump_ratio:.2f}"
                            )
                            lc_submaps.append(Submap(
                                submap_id=len(submaps) + len(lc_submaps),
                                frames=torch.stack([q_frame, d_frame]),
                                poses=lc_poses,
                                intrinsics=np.stack([
                                    submap.intrinsics[match.query_frame_idx],
                                    d_submap.intrinsics[match.detected_frame_idx],
                                ]),
                                retrieval_vectors=torch.zeros(2, ret_vecs.shape[-1]),
                                image_paths=[
                                    submap.image_paths[match.query_frame_idx],
                                    d_submap.image_paths[match.detected_frame_idx],
                                ],
                                is_lc_submap=True,
                            ))
                    all_loop_candidates.append(match)

                submaps.append(submap)
                pbar.update(1)
                pbar.set_postfix(loops=loops_found, verified=verified)
                if end >= N:
                    break

        # Inspection state (not stable API — see BaseFeedforwardCreator docstring).
        self.base._lc_submaps = submaps
        self.base._lc_loop_submaps = lc_submaps
        self.base._lc_overlap_frames = O
        self.base._lc_all_matches = all_loop_candidates

        pg_result = run_sim3_pose_graph_optimization(
            submaps, lc_submaps, total_frames=N,
            overlap_frames=cfg.submap_overlap,
            lm_steps=cfg.sim3_lm_steps,
            return_trace=True,
        )
        self.base.raw_outputs = merge_submap_outputs(
            submaps, pg_result["corrected"], sim3_nodes=pg_result["sim3_nodes"]
        )
```

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py tests/pointcloud/test_wrappers.py
git commit -m "feat(pointcloud): add LoopClosure proxy wrapper class"
```

---

## Task 7: Strip LC concerns from `BaseFeedforwardCreator`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py` (`BaseFeedforwardCreator`, lines ~92–418)

- [ ] **Step 1: Run existing tests before touching anything**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v
```
Expected: all PASSED. (Baseline before removal.)

- [ ] **Step 2: Remove LC fields and methods from `BaseFeedforwardCreator`**

From the `BaseFeedforwardCreator` dataclass definition, remove these two field lines (around lines 139–140):

```python
    enable_loop_closure: bool = False
    loop_closure_config: LoopClosureConfig = field(default_factory=LoopClosureConfig)
```

From `run_inference()` (lines ~175–185), replace:

```python
    def run_inference(self, **kwargs: Any) -> None:
        import torch
        t0 = time.perf_counter()
        console.log("Running inference...")
        if self.enable_loop_closure and self._enough_frames_for_submaps():
            self._run_loop_closure_inference(**kwargs)
        else:
            self.raw_outputs = self._forward(self.model, self.views, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")
```

with:

```python
    def run_inference(self, **kwargs: Any) -> None:
        import torch
        t0 = time.perf_counter()
        console.log("Running inference...")
        self.raw_outputs = self._forward(self.model, self.views, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")
```

Delete the methods `_enough_frames_for_submaps()` and `_run_loop_closure_inference()` entirely (lines ~187–372).

Remove the top-level import of `LoopClosureConfig` from `feedforward.py` (line 21: `from .loop_closure import LoopClosureConfig`) since it's no longer used in `feedforward.py`.

- [ ] **Step 3: Run all tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v
```
Expected: all PASSED. (The LC loop now lives only in `LoopClosure._run_lc_loop()`.)

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py
git commit -m "refactor(feedforward): remove enable_loop_closure, loop_closure_config, _run_loop_closure_inference from BaseFeedforwardCreator"
```

---

## Task 8: Update `__init__.py` — exports and `make_creator` factory

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py`
- Test: `tests/pointcloud/test_wrappers.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/pointcloud/test_wrappers.py
def test_make_creator_no_wrappers():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    creator = make_creator("vggtx")
    assert isinstance(creator, VGGTXCreator)


def test_make_creator_with_lc():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.wrappers import LoopClosure

    creator = make_creator("vggtx", use_lc=True)
    assert isinstance(creator, LoopClosure)
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    assert isinstance(creator.base, VGGTXCreator)


def test_make_creator_with_ba():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.wrappers import BundleAdjustment

    creator = make_creator("vggtx", use_ba=True)
    assert isinstance(creator, BundleAdjustment)


def test_make_creator_with_lc_and_ba():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.wrappers import BundleAdjustment, LoopClosure

    creator = make_creator("vggtx", use_lc=True, use_ba=True)
    assert isinstance(creator, BundleAdjustment)
    assert isinstance(creator.base, LoopClosure)


def test_make_creator_unknown_name():
    from collab_splats.pointcloud import make_creator
    import pytest
    with pytest.raises(KeyError):
        make_creator("unknown_backend")
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_make_creator_no_wrappers -v
```
Expected: `ImportError` (make_creator not exported)

- [ ] **Step 3: Update `__init__.py`**

Replace the entire contents of `collab_splats/pointcloud/__init__.py` with:

```python
# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult, _colmap_recon_to_result
from .sfm import ColmapCreator, HlocCreator
from .feedforward import BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator
from .bundle_adjustment import BundleAdjustmentConfig, run_bundle_adjustment
from .loop_closure import LoopClosureConfig
from .wrappers import BundleAdjustment, LoopClosure

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "colmap":      ColmapCreator,
    "hloc":        HlocCreator,
    "mapanything": MapAnythingCreator,
    "vggtx":       VGGTXCreator,
}


def get_creator(name: str) -> type[BasePointcloudCreator]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def make_creator(
    name: str,
    *,
    use_lc: bool = False,
    use_ba: bool = False,
    lc_config: LoopClosureConfig | None = None,
    ba_config: BundleAdjustmentConfig | None = None,
    **kwargs,
) -> BasePointcloudCreator:
    """Construct a pointcloud creator, optionally wrapped with LoopClosure and/or BundleAdjustment.

    Args:
        name: Creator name ('colmap', 'hloc', 'mapanything', 'vggtx').
        use_lc: Wrap with LoopClosure (feedforward creators only).
        use_ba: Wrap with BundleAdjustment (feedforward creators only).
        lc_config: Optional LoopClosureConfig; uses defaults if None.
        ba_config: Optional BundleAdjustmentConfig; uses defaults if None.
        **kwargs: Forwarded to the creator constructor (e.g. model_name, conf_threshold).
    """
    creator: BasePointcloudCreator = get_creator(name)(**kwargs)
    if use_lc:
        creator = LoopClosure(creator, config=lc_config)
    if use_ba:
        creator = BundleAdjustment(creator, config=ba_config)
    return creator


__all__ = [
    "BasePointcloudCreator",
    "BaseFeedforwardCreator",
    "BundleAdjustment",
    "BundleAdjustmentConfig",
    "CoordinateFrame",
    "ColmapCreator",
    "HlocCreator",
    "LoopClosure",
    "LoopClosureConfig",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "get_creator",
    "make_creator",
    "run_bundle_adjustment",
]
```

- [ ] **Step 4: Run all tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/__init__.py tests/pointcloud/test_wrappers.py
git commit -m "feat(pointcloud): export LoopClosure, BundleAdjustment, make_creator from __init__"
```

---

## Task 9: Update existing tests and final regression check

**Files:**
- Modify: `tests/pointcloud/test_loop_closure.py` (remove any `enable_loop_closure=True` usage)

- [ ] **Step 1: Check for stale flag usage**

```bash
grep -rn "enable_loop_closure\|use_ba\|loop_closure_config" \
  /workspace/collab-splats/tests/ \
  /workspace/collab-splats/collab_splats/ \
  /workspace/collab-splats/examples/ 2>/dev/null
```

For each hit:
- In `tests/` — update to use `LoopClosure(creator)` or `BundleAdjustment(creator)` pattern
- In `collab_splats/` — these should be zero (already removed in Tasks 3/4/7); fail if any remain
- In `examples/` — update to use `make_creator(..., use_lc=True, use_ba=True)` or explicit wrapping

- [ ] **Step 2: Fix any hits in `test_loop_closure.py`**

If the existing test file uses `BaseFeedforwardCreator(enable_loop_closure=True)` or similar, replace with mock-based `LoopClosure` tests. Since `test_loop_closure.py` currently tests `Submap`, `LoopClosureConfig`, `ImageRetrieval.find_loop_closures` directly (not through the creator flag), it likely needs no changes — confirm with the grep.

- [ ] **Step 3: Run full test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v
```
Expected: all PASSED

- [ ] **Step 4: Verify `wrappers.py` is importable cleanly**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud import (
    make_creator, get_creator, LoopClosure, BundleAdjustment,
    BundleAdjustmentConfig, LoopClosureConfig, VGGTXCreator, MapAnythingCreator
)
print('imports OK')
creator = make_creator('vggtx', use_lc=True, use_ba=True)
print(type(creator).__name__, type(creator.base).__name__, type(creator.base.base).__name__)
"
```
Expected:
```
imports OK
BundleAdjustment LoopClosure VGGTXCreator
```

- [ ] **Step 5: Final commit**

```bash
git add tests/pointcloud/test_loop_closure.py  # only if changed
git commit -m "refactor(tests): remove stale enable_loop_closure/use_ba flag usage"
```

If no test files changed, skip this commit.
