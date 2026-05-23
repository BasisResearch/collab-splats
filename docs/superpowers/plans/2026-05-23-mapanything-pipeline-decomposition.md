# MapAnything Pipeline Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `conf` NaN regression from cu121 migration by splitting `model.infer()` across `_preprocess`/`_forward`/`_postprocess` so bf16 tensors are cast to float32 before `postprocess_model_outputs_for_inference` runs, eliminating the `F.grid_sample` dtype mismatch.

**Architecture:** `_preprocess` validates and preprocesses views into `self._processed_views` (CPU). `_forward` transfers to device, runs `model.forward()` under bf16 autocast, returns raw bf16 outputs. `_postprocess` casts bf16 tensors to float32, calls `postprocess_model_outputs_for_inference`, builds `FeedforwardResult`. Dead code removed: `_patch_mapanything_torch_compat`, `run_mapanything`, `_reproject_mapanything`, `_reproject_after_ba`.

**Tech Stack:** PyTorch 2.4+cu121, MapAnything (`model.forward()`, `mapanything.utils.inference`), Open3D, pytest with `unittest.mock`.

---

## File Map

| File | Action |
|---|---|
| `collab_splats/pointcloud/feedforward/mapanything.py` | Remove 4 stdlib imports + dead functions; add 3 inference imports; rewrite `_load_model`, `_preprocess`, `_forward`, `_postprocess`; update module docstring |
| `tests/pointcloud/test_mapanything_creator.py` | Delete 5 compat-patch tests; rewrite 1 forward test; add 2 unit tests; update 1 GPU smoke test |

---

### Task 1: Delete dead compat-patch tests

**Files:**
- Modify: `tests/pointcloud/test_mapanything_creator.py`

These five tests cover `_patch_mapanything_torch_compat`, which is dead code on torch 2.4. Deleting them before changing the production code establishes a clean baseline.

- [ ] **Step 1: Delete the five compat-patch tests**

Remove these complete functions from `tests/pointcloud/test_mapanything_creator.py`:
- `test_patch_mapanything_torch_compat_replaces_multi_dim_any` (lines 68–75)
- `test_patch_mapanything_torch_compat_is_idempotent` (lines 78–90)
- `test_patch_mapanything_torch_compat_raises_on_source_drift` (lines 93–105)
- `test_load_model_invokes_torch_compat_patch` (lines 108–121)
- `test_patch_mapanything_torch_compat_rebinds_aliased_imports` (lines 124–132)

Also remove the top-level `import inspect` (line 1) — it was only needed by `test_patch_mapanything_torch_compat_replaces_multi_dim_any`.

The file should now start with:
```python
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from collab_splats.pointcloud.feedforward import MapAnythingCreator, BaseFeedforwardCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame
```

- [ ] **Step 2: Verify remaining tests still pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py -v -k "not gpu"
```

Expected: 3 tests pass (`test_mapanything_defaults`, `test_mapanything_is_feedforward_creator`, `test_mapanything_missing_image_dir_raises`), 1 test still passes but will need updating later (`test_mapanything_forward_passes_inference_params`).

- [ ] **Step 3: Commit**

```bash
git add tests/pointcloud/test_mapanything_creator.py
git commit -m "test(mapanything): delete dead compat-patch tests (torch>=2.4 native support)"
```

---

### Task 2: Write failing tests for new behavior

**Files:**
- Modify: `tests/pointcloud/test_mapanything_creator.py`

Write tests before implementing so they fail for the right reason.

- [ ] **Step 1: Replace the forward test**

Delete `test_mapanything_forward_passes_inference_params` entirely and replace with:

```python
def test_mapanything_forward_calls_model_forward():
    import torch

    n = 2
    param = torch.zeros(1)  # CPU param; provides .device = cpu
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([param])
    mock_raw = [MagicMock() for _ in range(n)]
    mock_model.forward.return_value = mock_raw

    creator = MapAnythingCreator(minibatch_size=2)
    creator._processed_views = [{"img": torch.zeros(1, 3, 64, 64)} for _ in range(n)]

    result = creator._forward(mock_model, views=None)

    mock_model.forward.assert_called_once_with(
        creator._processed_views,
        memory_efficient_inference=True,
        minibatch_size=2,
    )
    assert result is mock_raw
```

- [ ] **Step 2: Add the preprocess test**

Append after `test_mapanything_forward_calls_model_forward`:

```python
def test_mapanything_preprocess_sets_processed_views(tmp_path):
    import torch
    from PIL import Image as PILImage

    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    for i in range(2):
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
            image_dir / f"frame_{i:04d}.jpg"
        )

    fake_views = [{"img": torch.zeros(1, 3, 224, 224), "data_norm_type": "imagenet"}
                  for _ in range(2)]
    fake_validated = fake_views
    fake_processed = [{"img": torch.zeros(1, 3, 224, 224)} for _ in range(2)]

    with patch("collab_splats.pointcloud.feedforward.mapanything.load_images",
               return_value=fake_views), \
         patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
               return_value=fake_validated) as mock_validate, \
         patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=fake_processed) as mock_preprocess:
        creator = MapAnythingCreator()
        creator._preprocess(image_dir)

    mock_validate.assert_called_once_with(fake_views)
    mock_preprocess.assert_called_once_with(fake_validated)
    assert creator._processed_views is fake_processed
```

- [ ] **Step 3: Add the dtype regression test**

Append after `test_mapanything_preprocess_sets_processed_views`:

```python
def test_mapanything_postprocess_casts_bf16_to_float32():
    """Regression guard: pts3d_cam and pts3d must be float32 before postprocess.

    F.grid_sample inside compute_multiview_depth_confidence requires matching
    dtypes. torch 2.4 enforces this strictly; 2.1.2 allowed bf16/float32 mismatch.
    """
    import torch
    import open3d as o3d

    n, h, w = 2, 4, 4
    raw_outputs = [
        {
            "pts3d_cam": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
            "pts3d": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
        }
        for _ in range(n)
    ]

    fake_processed = [
        {
            "pts3d": torch.zeros(1, h, w, 3),
            "pts3d_cam": torch.zeros(1, h, w, 3),
            "mask": torch.ones(1, h, w, 1, dtype=torch.bool),
            "depth_z": torch.ones(1, h, w, 1),
            "img_no_norm": torch.zeros(1, h, w, 3),
            "intrinsics": torch.eye(3).unsqueeze(0),
            "camera_poses": torch.eye(4).unsqueeze(0),
        }
        for _ in range(n)
    ]

    fake_pcd = o3d.geometry.PointCloud()
    fake_pcd.points = o3d.utility.Vector3dVector(np.zeros((5, 3)))
    fake_pcd.colors = o3d.utility.Vector3dVector(np.zeros((5, 3)))

    creator = MapAnythingCreator()
    creator._processed_views = [{"img": torch.zeros(1, 3, h, w)} for _ in range(n)]
    creator.image_paths = [Path(f"/fake/img_{i}.jpg") for i in range(n)]
    creator.original_coords = np.zeros((n, 6), dtype=np.float32)

    with patch("collab_splats.pointcloud.feedforward.mapanything"
               ".postprocess_model_outputs_for_inference",
               return_value=fake_processed) as mock_post, \
         patch("collab_splats.pointcloud.feedforward.mapanything.collect_pts3d_from_outputs",
               return_value=(
                   np.zeros((5, 3), dtype=np.float32),
                   np.zeros((5, 3), dtype=np.uint8),
                   np.zeros((n, 3, 4), dtype=np.float32),
                   np.zeros((n, 3, 3), dtype=np.float32),
               )), \
         patch("collab_splats.pointcloud.feedforward.mapanything.voxel_downsample",
               return_value=(fake_pcd, None)):
        creator._postprocess(raw_outputs)

    # raw_outputs is mutated in-place before postprocess is called;
    # mock captures the reference so we inspect dtype at call time
    called_raw = mock_post.call_args[0][0]
    for pred in called_raw:
        assert pred["pts3d_cam"].dtype == torch.float32, (
            f"pts3d_cam not cast to float32 before postprocess: {pred['pts3d_cam'].dtype}"
        )
        assert pred["pts3d"].dtype == torch.float32, (
            f"pts3d not cast to float32 before postprocess: {pred['pts3d'].dtype}"
        )
```

- [ ] **Step 4: Run to confirm all three new tests fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_forward_calls_model_forward tests/pointcloud/test_mapanything_creator.py::test_mapanything_preprocess_sets_processed_views tests/pointcloud/test_mapanything_creator.py::test_mapanything_postprocess_casts_bf16_to_float32 -v
```

Expected: all 3 FAIL.
- `test_mapanything_forward_calls_model_forward`: `AssertionError` — `model.forward` not called (current impl calls `run_mapanything`).
- `test_mapanything_preprocess_sets_processed_views`: `AttributeError` — `validate_input_views_for_inference` not imported.
- `test_mapanything_postprocess_casts_bf16_to_float32`: `AttributeError` — `postprocess_model_outputs_for_inference` not imported.

- [ ] **Step 5: Commit tests-only**

```bash
git add tests/pointcloud/test_mapanything_creator.py
git commit -m "test(mapanything): add failing tests for pipeline split and bf16 dtype regression"
```

---

### Task 3: Implement the refactor in mapanything.py

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`

All changes to a single file. Steps are ordered to maintain a parseable file at each checkpoint.

- [ ] **Step 1: Replace the module docstring and imports**

Replace from the start of the file through the `from .base import ...` line with:

```python
"""MapAnything feedforward backend: inference utilities and creator.

Provides:
  collect_pts3d_from_outputs — extract pts3d/colors/extrinsics/intrinsics from processed outputs
  MapAnythingCreator         — feedforward creator using MapAnything depth + pose estimation
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

# timm 0.6.x compat: uniception (mapanything dep) imports `from timm.layers import DropPath`
# which does not exist in timm<0.9. Re-export it from timm.models.layers before the import.
import timm.layers as _tl, timm.models.layers as _tml  # noqa: E401
if not hasattr(_tl, "DropPath"):
    _tl.DropPath = _tml.DropPath
del _tl, _tml

from mapanything.models import MapAnything
from mapanything.utils.geometry import closed_form_pose_inverse
from mapanything.utils.image import load_images
from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)

from ..utils import voxel_downsample
from .base import BaseFeedforwardCreator, FeedforwardResult, _extrinsics_3x4_to_4x4, console
```

Removed vs current: `inspect`, `linecache`, `sys`, `textwrap`, `import mapanything.utils.wai.intersection_check as ic`.
Added: `from mapanything.utils.inference import (...)`.

- [ ] **Step 2: Delete the compat-patch section and `run_mapanything`**

Delete the entire block from `# ── MapAnything compat patch ──` through the end of the `run_mapanything` function (lines 43–133 in the current file). This removes `_patch_mapanything_torch_compat` and `run_mapanything`.

The file should now jump directly from the imports to `# ── Inference utilities ──` with only `collect_pts3d_from_outputs`.

- [ ] **Step 3: Delete `_reproject_mapanything`**

Delete the entire `_reproject_mapanything` function (currently lines 180–222). BA+MapAnything is untested; the implementation depends on postprocessed keys that `model.forward()` does not produce. The base class `NotImplementedError` guards against BA+MapAnything until a correct implementation exists.

The `# ── Inference utilities ──` section now contains only `collect_pts3d_from_outputs`.

- [ ] **Step 4: Rewrite `_load_model`**

Replace the body of `_load_model`:

```python
    def _load_model(self, device: str) -> Any:
        model = MapAnything.from_pretrained(self.model_name)
        model = model.to(device)
        model.eval()
        return model
```

(Remove the `_patch_mapanything_torch_compat()` call — dead on torch 2.4.)

- [ ] **Step 5: Rewrite `_preprocess`**

Replace the entire `_preprocess` method:

```python
    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix in exts)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        views = load_images([str(p) for p in image_paths])
        model_h: int = views[0]["img"].shape[-2]
        model_w: int = views[0]["img"].shape[-1]

        original_coords = np.array(
            [
                [0, 0, model_w, model_h, PILImage.open(p).width, PILImage.open(p).height]
                for p in image_paths
            ],
            dtype=np.float32,
        )

        # Validate views meet MapAnything input requirements, then convert to the
        # internal format model.forward() expects (ray directions, metric scale, etc.).
        # Kept on CPU here; transferred to model device in _forward.
        validated = validate_input_views_for_inference(views)
        self._processed_views = preprocess_input_views_for_inference(validated)

        return views, image_paths, original_coords
```

- [ ] **Step 6: Rewrite `_forward`**

Replace the entire `_forward` method:

```python
    def _forward(self, model: Any, views: Any, **kwargs: Any) -> list[dict]:
        console.log(f"  → {len(self._processed_views)} images, minibatch_size={self.minibatch_size}")
        device = next(model.parameters()).device
        device_type = device.type

        # Transfer preprocessed views to model device; kept on CPU in _preprocess
        # to avoid holding GPU memory during image loading and validation.
        for view in self._processed_views:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device)

        # bf16 autocast scoped to model forward only; postprocessing requires float32
        # to avoid F.grid_sample dtype mismatch (torch 2.4 enforces strict matching).
        with torch.no_grad():
            with torch.autocast(device_type, dtype=torch.bfloat16,
                                 enabled=(device_type == "cuda")):
                return model.forward(
                    self._processed_views,
                    memory_efficient_inference=True,
                    minibatch_size=self.minibatch_size,
                )
```

- [ ] **Step 7: Rewrite `_postprocess`**

Replace the entire `_postprocess` method:

```python
    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        model_h: int = self._processed_views[0]["img"].shape[-2]
        model_w: int = self._processed_views[0]["img"].shape[-1]

        # Cast bf16 tensors to float32 before postprocessing. model.forward() runs
        # under bf16 autocast; postprocess_model_outputs_for_inference calls
        # F.grid_sample which requires matching dtypes (torch 2.4 strict enforcement).
        for pred in raw_outputs:
            pred["pts3d_cam"] = pred["pts3d_cam"].float()
            pred["pts3d"] = pred["pts3d"].float()

        processed = postprocess_model_outputs_for_inference(
            raw_outputs,
            self._processed_views,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            use_multiview_confidence=self.use_multiview_confidence,
            confidence_percentile=self.confidence_percentile,
        )

        pts3d, colors, extrinsics, intrinsics = collect_pts3d_from_outputs(processed)

        _images = torch.stack(
            [p["img_no_norm"][0].cpu().permute(2, 0, 1) for p in processed]
        )
        if processed[0].get("conf") is not None:
            conf_list = [p["conf"][0] for p in processed]
            _conf = torch.stack([c[0] if c.ndim == 3 else c for c in conf_list])
        else:
            _conf = None
        _world_points = np.stack(
            [p["pts3d"][0].cpu().numpy() for p in processed]
        )  # (N, H, W, 3)

        # Voxel downsample — runs once regardless of use_ba path
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(pts3d)
        _pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        _pcd, _ = voxel_downsample(_pcd, adaptive=False)
        pts3d = np.asarray(_pcd.points, dtype=np.float32)
        colors = (np.asarray(_pcd.colors) * 255).astype(np.uint8)

        extrinsics_4x4 = _extrinsics_3x4_to_4x4(extrinsics)

        return FeedforwardResult(
            pts3d=pts3d,
            colors=colors,
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

- [ ] **Step 8: Delete `_reproject_after_ba`**

Delete the entire `_reproject_after_ba` method from `MapAnythingCreator`. The base class `NotImplementedError` is sufficient until BA+MapAnything is actively tested.

- [ ] **Step 9: Run the three new unit tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_forward_calls_model_forward tests/pointcloud/test_mapanything_creator.py::test_mapanything_preprocess_sets_processed_views tests/pointcloud/test_mapanything_creator.py::test_mapanything_postprocess_casts_bf16_to_float32 -v
```

Expected: all 3 PASS.

- [ ] **Step 10: Run full non-GPU test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py -v -k "not gpu"
```

Expected: all non-GPU tests PASS. If any fail, fix before proceeding.

- [ ] **Step 11: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py
git commit -m "refactor(mapanything): split model.infer() across _preprocess/_forward/_postprocess

Remove _patch_mapanything_torch_compat (dead on torch 2.4), run_mapanything,
_reproject_mapanything, _reproject_after_ba (BA+MapAnything untested).

_forward calls model.forward() under bf16 autocast; _postprocess casts
pts3d_cam and pts3d to float32 before postprocess_model_outputs_for_inference,
fixing F.grid_sample dtype mismatch (torch 2.4 strict enforcement).

Fixes conf NaN regression from cu121 migration."
```

---

### Task 4: Update GPU smoke test

**Files:**
- Modify: `tests/pointcloud/test_mapanything_creator.py`

- [ ] **Step 1: Update `test_mapanything_run_inference_smoke`**

Replace the body of `test_mapanything_run_inference_smoke` (keep the `@pytest.mark.gpu` decorator and function signature):

```python
@pytest.mark.gpu
def test_mapanything_run_inference_smoke(tmp_path):
    """Verify MapAnything inference completes without dtype errors on torch 2.4+cu121
    and produces valid (non-NaN) confidence scores.

    The pre-cu121 compat patch let F.grid_sample(bf16, float32) silently succeed;
    torch 2.4 raises RuntimeError. This test guards the dtype-cast fix in _postprocess.
    """
    pytest.importorskip("mapanything")
    pytest.importorskip("torch")
    bicycle = Path("/workspace/bicycle/images_4")
    if not bicycle.exists():
        pytest.skip(f"{bicycle} not available on this host")

    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    creator = MapAnythingCreator(camera_model="PINHOLE")
    creator.load_model()
    creator.setup_inference(bicycle)
    creator.run_inference()
    creator.postprocess()

    assert creator.outputs.conf is not None, "conf should be set when use_multiview_confidence=True"
    assert not creator.outputs.conf.isnan().any(), (
        "conf contains NaN — dtype cast regression in _postprocess"
    )
```

- [ ] **Step 2: Run the GPU test (if GPU available)**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_run_inference_smoke -v -m gpu
```

Expected: PASS with no `RuntimeError: expected scalar type BFloat16 but found Float` and `conf` not NaN.

If GPU not available locally, note that this will skip and must be validated in the GPU environment.

- [ ] **Step 3: Commit**

```bash
git add tests/pointcloud/test_mapanything_creator.py
git commit -m "test(mapanything): update GPU smoke test to assert conf is non-NaN"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Fix `F.grid_sample` dtype mismatch | Task 3, Steps 6–7 |
| Remove `_patch_mapanything_torch_compat` + 4 stdlib imports | Task 3, Steps 1–2 |
| Remove `run_mapanything` | Task 3, Step 2 |
| Remove `_reproject_mapanything` + `_reproject_after_ba` | Task 3, Steps 3, 8 |
| `_preprocess` calls validate + preprocess, stores `self._processed_views` | Task 3, Step 5 |
| `_forward` calls `model.forward()` under bf16 autocast | Task 3, Step 6 |
| `_postprocess` casts to float32, calls `postprocess_model_outputs_for_inference` | Task 3, Step 7 |
| Update module docstring | Task 3, Step 1 |
| Delete 5 compat-patch tests | Task 1 |
| Rewrite 1 forward test | Task 2, Step 1 |
| Add `test_mapanything_preprocess_sets_processed_views` | Task 2, Step 2 |
| Add `test_mapanything_postprocess_casts_bf16_to_float32` | Task 2, Step 3 |
| Update GPU smoke test with conf assertion | Task 4 |

All spec requirements covered. No gaps.
