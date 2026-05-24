# BA Pointcloud Reprojection Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three bugs left by the `ba-module-cleanup` refactor: broken import in `eval_gt.py`, API mismatch when constructing `BundleAdjustment`, and missing pts3d reprojection after BA.

**Architecture:** Add `FeedforwardResult.reproject()` as an instance method using stored `depth` + `pixel_indices` fields — all inputs live on the result so no creator state is needed. Fix `eval_gt.py` to use the correct import path and the new standalone BA API (`ba.refine(result).reproject()`). Update the `BaseFeedforwardCreator.reproject()` docstring to point callers to the preferred path.

**Tech Stack:** Python 3.11, numpy, pytest, `unittest.mock.patch`. Python env: `/opt/conda/envs/nerfstudio/bin/python`.

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/pointcloud/feedforward/base.py` | Add `reproject_pixels` to top-level import; add `FeedforwardResult.reproject()` method; update `BaseFeedforwardCreator.reproject()` docstring |
| `evals/eval_gt.py` | Fix import; refactor `_make_creator` to return `(creator, ba_cfg \| None)`; refactor `_run_condition` to use standalone BA flow |
| `tests/pointcloud/test_feedforward_reproject.py` | New — unit tests for `FeedforwardResult.reproject()` |

---

## Task 1: TDD — `FeedforwardResult.reproject()`

**Files:**
- Create: `tests/pointcloud/test_feedforward_reproject.py`
- Modify: `collab_splats/pointcloud/feedforward/base.py`

---

- [ ] **Step 1: Write the failing tests**

Create `tests/pointcloud/test_feedforward_reproject.py`:

```python
"""Unit tests for FeedforwardResult.reproject()."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _make_result(*, depth=None, pixel_indices=None) -> FeedforwardResult:
    """Minimal FeedforwardResult with configurable depth and pixel_indices."""
    N, H, W, P = 2, 4, 4, 3
    return FeedforwardResult(
        pts3d=np.zeros((P, 3), dtype=np.float32),
        colors=np.zeros((P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        image_paths=[Path(f"img_{i}.png") for i in range(N)],
        original_coords=np.zeros((N, 6), dtype=np.float32),
        model_width=W,
        model_height=H,
        depth=depth,
        pixel_indices=pixel_indices,
    )


def test_reproject_calls_reproject_pixels_with_correct_args():
    N, H, W, P = 2, 4, 4, 3
    depth = np.ones((N, H, W), dtype=np.float32)
    pixel_indices = np.array([[0, 1, 1], [1, 2, 2], [0, 3, 3]], dtype=np.int32)
    result = _make_result(depth=depth, pixel_indices=pixel_indices)
    new_pts = np.ones((P, 3), dtype=np.float32) * 99.0

    with patch(
        "collab_splats.pointcloud.feedforward.base.reproject_pixels",
        return_value=new_pts,
    ) as mock_rp:
        reprojected = result.reproject()

    mock_rp.assert_called_once_with(
        depth,
        pixel_indices,
        result.extrinsics[:, :3, :],
        result.intrinsics,
    )
    np.testing.assert_array_equal(reprojected.pts3d, new_pts)


def test_reproject_preserves_colors_and_extrinsics():
    N, H, W, P = 2, 4, 4, 3
    depth = np.ones((N, H, W), dtype=np.float32)
    pixel_indices = np.zeros((P, 3), dtype=np.int32)
    result = _make_result(depth=depth, pixel_indices=pixel_indices)
    new_pts = np.full((P, 3), 7.0, dtype=np.float32)

    with patch(
        "collab_splats.pointcloud.feedforward.base.reproject_pixels",
        return_value=new_pts,
    ):
        reprojected = result.reproject()

    # World positions updated; source-pixel-derived fields unchanged
    np.testing.assert_array_equal(reprojected.colors, result.colors)
    np.testing.assert_array_equal(reprojected.extrinsics, result.extrinsics)
    assert reprojected is not result


def test_reproject_raises_without_depth():
    P = 3
    pixel_indices = np.zeros((P, 3), dtype=np.int32)
    result = _make_result(depth=None, pixel_indices=pixel_indices)
    with pytest.raises(ValueError, match="reproject\\(\\) requires depth"):
        result.reproject()


def test_reproject_raises_without_pixel_indices():
    N, H, W = 2, 4, 4
    depth = np.ones((N, H, W), dtype=np.float32)
    result = _make_result(depth=depth, pixel_indices=None)
    with pytest.raises(ValueError, match="reproject\\(\\) requires depth"):
        result.reproject()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_reproject.py -v
```

Expected: 4 failures with `AttributeError: 'FeedforwardResult' object has no attribute 'reproject'`.

- [ ] **Step 3: Add `reproject_pixels` to the top-level import in `feedforward/base.py`**

In `collab_splats/pointcloud/feedforward/base.py`, find line:
```python
from ..utils import colmap_reconstruction_to_result, cross_frame_attention_ratio
```

Replace with:
```python
from ..utils import colmap_reconstruction_to_result, cross_frame_attention_ratio, reproject_pixels
```

- [ ] **Step 4: Add `FeedforwardResult.reproject()` method**

In `collab_splats/pointcloud/feedforward/base.py`, inside the `FeedforwardResult` dataclass, insert this method after the `load_zarr` classmethod (around line 225, before the blank line that precedes the `# ── Geometry helpers` section divider):

```python
    def reproject(self) -> "FeedforwardResult":
        """Re-project pts3d under current extrinsics using stored source pixels and depth."""
        if self.depth is None or self.pixel_indices is None:
            raise ValueError(
                "reproject() requires depth and pixel_indices; load via load_zarr() "
                "or ensure the creator's _postprocess populated both fields."
            )
        # Reproject stored source pixels under new extrinsics — deterministic,
        # point set stays index-aligned with colors and features.
        pts3d = reproject_pixels(
            self.depth, self.pixel_indices,
            self.extrinsics[:, :3, :], self.intrinsics,
        )
        return replace(self, pts3d=pts3d)
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_reproject.py -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Run full pointcloud test suite — verify no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Format**

```bash
cd /workspace/collab-splats && black collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_feedforward_reproject.py && isort collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_feedforward_reproject.py
```

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_feedforward_reproject.py
git commit -m "feat(feedforward): add FeedforwardResult.reproject() — creator-free pts3d reprojection after BA"
```

---

## Task 2: Fix `eval_gt.py` — import + standalone BA flow

**Files:**
- Modify: `evals/eval_gt.py`

The three problems to fix:
1. Line 43: wrong import location for `BundleAdjustment`
2. `_make_creator` returns a bare creator and uses `BundleAdjustment(creator, config=...)` (old wrapper API)
3. `_run_condition` calls `creator.reconstruct()` with no BA step

---

- [ ] **Step 1: Fix imports in `evals/eval_gt.py`**

Find these lines (around lines 39–43):
```python
from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig
from collab_splats.pointcloud.loop_closure.eval import ate_translation, rpe, auc_at_threshold
from collab_splats.pointcloud.loop_closure import LoopClosureConfig
from collab_splats.pointcloud.wrappers import BundleAdjustment, LoopClosure
```

Replace with:
```python
from collab_splats.pointcloud import BundleAdjustment, BundleAdjustmentConfig, get_creator
from collab_splats.pointcloud.loop_closure.eval import ate_translation, rpe, auc_at_threshold
from collab_splats.pointcloud.loop_closure import LoopClosureConfig
from collab_splats.pointcloud.wrappers import LoopClosure
```

- [ ] **Step 2: Refactor `_make_creator` to return `(creator, ba_cfg | None)`**

Find and replace the entire `_make_creator` function:

```python
def _make_creator(condition: str, submap_size: int | None = None):
    """Build a (creator, ba_config) pair for the given condition.

    Returns (creator, None) when no bundle adjustment is needed.
    Returns (creator, BundleAdjustmentConfig) when BA should run after postprocess.
    """
    base = get_creator("vggtx")()
    if condition == "lc":
        return LoopClosure(base), None
    m = re.fullmatch(r"ba_track-density-(\d+)", condition)
    if m:
        n = int(m.group(1))
        cfg = BundleAdjustmentConfig(
            max_query_pts=n,
            query_frame_num=max(5, n // 512),
        )
        if submap_size is not None:
            _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_cosine_threshold=1.0)
            windowed = LoopClosure(base, config=_no_lc_cfg)
            return windowed, cfg
        return base, cfg
    if submap_size is not None:
        # Windowed mode: LC pipeline with detection disabled so baseline = windowed VGGT-X
        _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_cosine_threshold=1.0)
        windowed = LoopClosure(base, config=_no_lc_cfg)
        if condition == "ba":
            return windowed, BundleAdjustmentConfig()
        return windowed, None  # baseline
    # Default: single-pass (short sequences that fit in GPU memory)
    if condition == "ba":
        return base, BundleAdjustmentConfig()
    return base, None  # baseline
```

- [ ] **Step 3: Refactor `_run_condition` to use standalone BA flow**

Find and replace the entire `_run_condition` function:

```python
def _run_condition(name: str, image_dir: Path, output_dir: Path, submap_size: int | None = None) -> np.ndarray:
    """Wrap VGGTXCreator with BA/LC as appropriate, run, return (N,4,4) extrinsics."""
    creator, ba_cfg = _make_creator(name, submap_size=submap_size)
    if ba_cfg is None:
        creator.reconstruct(image_dir, output_dir)
    else:
        # Run inference, refine poses with BA, reproject pts3d, then write COLMAP output
        ba = BundleAdjustment(ba_cfg)
        creator.load_model()
        creator.setup_inference(image_dir)
        creator.run_inference()
        creator.postprocess()
        creator.outputs = ba.refine(creator.outputs).reproject()
        creator.build_colmap(output_dir)
    if creator.outputs is None:
        raise RuntimeError(f"Condition '{name}' produced no outputs")
    return creator.outputs.extrinsics  # (N, 4, 4) world-to-cam
```

- [ ] **Step 4: Verify eval_gt.py imports cleanly**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "
import sys; sys.path.insert(0, 'evals')
import eval_gt
print('import OK')
"
```

Expected: `import OK` (no `ImportError` or `TypeError`).

- [ ] **Step 5: Run full test suite — verify no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Format**

```bash
cd /workspace/collab-splats && black evals/eval_gt.py && isort evals/eval_gt.py
```

- [ ] **Step 7: Commit**

```bash
git add evals/eval_gt.py
git commit -m "fix(eval): fix BundleAdjustment import + standalone BA API in eval_gt.py"
```

---

## Task 3: Update `BaseFeedforwardCreator.reproject()` docstring

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`

---

- [ ] **Step 1: Update the docstring**

In `collab_splats/pointcloud/feedforward/base.py`, find `BaseFeedforwardCreator.reproject()` (around line 712):

```python
    def reproject(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Re-extract pts3d/colors using refined poses stored in result.

        Uses self.raw_outputs from the last run() or run_inference() call.
        Only call when result.pixel_indices is not None.
        """
```

Replace docstring:

```python
    def reproject(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Re-extract pts3d/colors via full depth unprojection under refined poses.

        Uses self.raw_outputs from the last run() or run_inference() call.
        Prefer result.reproject() when depth and pixel_indices are populated —
        no creator state needed, deterministic point set.
        """
```

- [ ] **Step 2: Format and commit**

```bash
cd /workspace/collab-splats && black collab_splats/pointcloud/feedforward/base.py && isort collab_splats/pointcloud/feedforward/base.py
git add collab_splats/pointcloud/feedforward/base.py
git commit -m "docs(feedforward): note result.reproject() preferred over creator.reproject()"
```
