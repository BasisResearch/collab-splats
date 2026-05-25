# BundleAdjustment Standalone Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `BundleAdjustment` into a plain `refine(result) -> FeedforwardResult` class; remove the old `BasePointcloudCreator` wrapper; make `run_bundle_adjustment` and `extract_tracks_vggsfm` private; thread `device` config throughout.

**Architecture:** `BundleAdjustment` moves to `bundle_adjustment.py` as a plain class with a single `refine()` method. The old wrapper in `wrappers.py` is deleted. Creators gain `reproject()` for re-extracting pts3d post-BA. Pipeline becomes: `creator.run()` → `ba.refine()` → `creator.reproject()` → `creator.build_colmap()`.

**Tech Stack:** numpy, torch, pypose, bae (LM BA), vggt (VGGSfM tracker), pycolmap, dataclasses

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/bundle_adjustment.py` | Add `device` to `_get_default_solver`; rename public functions to private; add `BundleAdjustment` class; update `__all__` |
| `collab_splats/pointcloud/wrappers.py` | Delete `BundleAdjustment` class; add `LoopClosure.run()` + `LoopClosure.reproject()`; update `__all__` |
| `collab_splats/pointcloud/feedforward/base.py` | Add `reproject()` method; rename abstract `_reproject_ba` → `_reproject` |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Rename `_reproject_ba` → `_reproject` |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Rename `_reproject_ba` → `_reproject` |
| `collab_splats/pointcloud/__init__.py` | Remove `run_bundle_adjustment`; import `BundleAdjustment` from `bundle_adjustment`; remove `use_ba` from `make_creator` |
| `tests/pointcloud/test_bundle_adjustment.py` | Fix `._inner` assertion; update private function names; add `BundleAdjustment.refine()` tests |
| `tests/pointcloud/test_wrappers.py` | Remove old BA wrapper tests; update `_reproject_ba` → `_reproject`; add `LoopClosure.run/reproject` tests |
| `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb` | Update to new composition pattern |

---

### Task 1: Fix `_get_default_solver` — add `device` param + fix failing test

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`
- Modify: `tests/pointcloud/test_bundle_adjustment.py`

- [ ] **Step 1: Fix the existing failing test**

In `tests/pointcloud/test_bundle_adjustment.py`, replace line 463:
```python
    assert solver._inner is mock_cudss_instance
```
with:
```python
    assert solver is mock_cudss_instance
```

- [ ] **Step 2: Add test for `device="cpu"` always forces PCG**

Add after `test_get_default_solver_falls_back_to_pcg_cudss_import_error`:
```python
def test_get_default_solver_cpu_device_forces_pcg():
    """device='cpu' must always return PCG, even when CUDA is available."""
    mods = _make_ba_mods()
    mock_pcg_instance = MagicMock(name="pcg_instance")
    mock_pcg_class = MagicMock(name="PCG", return_value=mock_pcg_instance)
    mods["bae.utils.pysolvers"].PCG = mock_pcg_class

    with patch.dict(sys.modules, mods), patch("torch.cuda.is_available", return_value=True):
        ba_mod = _load_ba(mods)
        solver = ba_mod._get_default_solver(device="cpu")

    mock_pcg_class.assert_called_once()
    assert solver is mock_pcg_instance
```

- [ ] **Step 3: Run tests to verify current failure + new failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py::test_get_default_solver_prefers_cudss_when_available tests/pointcloud/test_bundle_adjustment.py::test_get_default_solver_cpu_device_forces_pcg -v
```
Expected: first passes, second fails with `AttributeError: _get_default_solver() takes 0 positional arguments`

- [ ] **Step 4: Add `device` param to `_get_default_solver`**

In `bundle_adjustment.py`, replace the `_get_default_solver` function:
```python
def _get_default_solver(device: str | None = None) -> Any:
    """Return CuDSS when CUDA is requested and available, else PCG."""
    import torch
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in resolved:
        try:
            from bae.sparse.solve import CuDirectSparseSolver
            return CuDirectSparseSolver()
        except (ImportError, RuntimeError):
            pass
    from bae.utils.pysolvers import PCG
    return PCG()
```

- [ ] **Step 5: Run tests to verify both pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py::test_get_default_solver_prefers_cudss_when_available tests/pointcloud/test_bundle_adjustment.py::test_get_default_solver_falls_back_to_pcg_no_cuda tests/pointcloud/test_bundle_adjustment.py::test_get_default_solver_falls_back_to_pcg_cudss_import_error tests/pointcloud/test_bundle_adjustment.py::test_get_default_solver_cpu_device_forces_pcg -v
```
Expected: all 4 PASS

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_bundle_adjustment.py
git commit -m "fix(ba): _get_default_solver accepts device param; fix test ._inner assertion"
```

---

### Task 2: Privatize `run_bundle_adjustment` + `extract_tracks_vggsfm`; add `device` to `_run_bundle_adjustment`

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`
- Modify: `tests/pointcloud/test_bundle_adjustment.py`

- [ ] **Step 1: Update tests to use private names**

In `test_bundle_adjustment.py`, make these replacements throughout the file:
- `ba_mod.extract_tracks_vggsfm(` → `ba_mod._extract_tracks_vggsfm(`
- `ba_mod.run_bundle_adjustment(` → `ba_mod._run_bundle_adjustment(`
- `from collab_splats.pointcloud.bundle_adjustment import run_bundle_adjustment` → `from collab_splats.pointcloud.bundle_adjustment import _run_bundle_adjustment`
- In `test_run_bundle_adjustment_reduces_reproj_error`, change the import and the call site accordingly.

Also update the test function `test_run_bundle_adjustment_early_exit_shape` — call:
```python
ref_pts, ref_ext, ref_intr = ba_mod._run_bundle_adjustment(
    points3d=points3d,
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    tracks=tracks,
    vis_mask=vis_mask,
    image_size=(H, W),
    max_reproj_error=4.0,
    lm_steps=5,
)
```

And `test_run_bundle_adjustment_no_reproj_filter`:
```python
ref_pts, ref_ext, ref_intr = ba_mod._run_bundle_adjustment(
    points3d=points3d,
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    tracks=tracks,
    vis_mask=vis_mask,
    image_size=(H, W),
    max_reproj_error=None,
)
```

And `test_run_bundle_adjustment_reduces_reproj_error`:
```python
from collab_splats.pointcloud.bundle_adjustment import _run_bundle_adjustment
...
_, ext_out, _ = _run_bundle_adjustment(
    points3d.copy(),
    extrinsics_noisy,
    intrinsics,
    tracks,
    vis_mask,
    image_size=(H, W),
    max_reproj_error=None,
    lm_steps=20,
)
```

And `test_extract_tracks_vggsfm_shape`:
```python
tracks, vis_scores, pts3d = ba_mod._extract_tracks_vggsfm(
    images, conf=conf, world_points=None, max_query_pts=512, query_frame_num=2,
)
```

And `test_extract_tracks_vggsfm_conf_4d`:
```python
tracks, vis_scores, pts3d = ba_mod._extract_tracks_vggsfm(
    images, conf=conf_4d, world_points=None
)
```

- [ ] **Step 2: Run tests to see current failures (names don't exist yet)**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -x -v 2>&1 | tail -20
```
Expected: multiple failures — `_extract_tracks_vggsfm` and `_run_bundle_adjustment` not found.

- [ ] **Step 3: Rename functions in `bundle_adjustment.py` and add `device` to `_run_bundle_adjustment`**

In `bundle_adjustment.py`:
1. Rename `extract_tracks_vggsfm` → `_extract_tracks_vggsfm` (the `def` line and docstring)
2. Rename `run_bundle_adjustment` → `_run_bundle_adjustment`
3. Add `device: str | None = None` to `_run_bundle_adjustment` signature (after `min_inliers_per_frame` in kwargs)
4. Replace hardcoded device detection in `_run_bundle_adjustment`:

Change:
```python
    device = "cuda" if torch.cuda.is_available() else "cpu"
```
to:
```python
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
```

5. Pass `device` to `_get_default_solver` in the optimizer line:
```python
        optimizer = LM(model, strategy=strategy, solver=solver if solver is not None else _get_default_solver(device=device), reject=10)
```

6. Update `__all__`:
```python
__all__ = ["BundleAdjustmentConfig"]
```
(Temporarily — `BundleAdjustment` added in Task 3)

- [ ] **Step 4: Run all BA tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v 2>&1 | tail -20
```
Expected: all pass (or only skipped if bae/pypose not available).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_bundle_adjustment.py
git commit -m "refactor(ba): privatize run_bundle_adjustment + extract_tracks_vggsfm; thread device through LM optimizer"
```

---

### Task 3: Add `BundleAdjustment` class to `bundle_adjustment.py`

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`
- Modify: `tests/pointcloud/test_bundle_adjustment.py`

- [ ] **Step 1: Write tests for `BundleAdjustment.refine()`**

Add to end of `test_bundle_adjustment.py`:

```python
# ---------------------------------------------------------------------------
# Tests for BundleAdjustment class
# ---------------------------------------------------------------------------

def _make_ff_result_for_ba(N=2, H=8, W=8):
    """Minimal FeedforwardResult for BA tests."""
    import dataclasses
    from pathlib import Path
    # Import FeedforwardResult without triggering vggt/bae at module level
    import sys
    import types
    # Patch heavy deps so feedforward/base can import
    # (already patched if called after module-level mocks are active)
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    return FeedforwardResult(
        pts3d=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        image_paths=[Path(f"img{i}.jpg") for i in range(N)],
        original_coords=np.zeros((N, 6), dtype=np.float32),
        model_width=W,
        model_height=H,
        images=torch.zeros(N, 3, H, W),
        conf=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )


def test_bundle_adjustment_refine_returns_feedforward_result():
    """refine() must return a FeedforwardResult with updated extrinsics/intrinsics."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))), \
         patch("collab_splats.pointcloud.bundle_adjustment._run_bundle_adjustment",
               return_value=(np.zeros((5, 3)), refined_ext, refined_intr)):
        ba = BundleAdjustment()
        out = ba.refine(result)

    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    assert isinstance(out, FeedforwardResult)
    assert out.extrinsics.shape == (N, 4, 4)
    assert out.intrinsics.shape == (N, 3, 3)


def test_bundle_adjustment_refine_preserves_pts3d_colors():
    """refine() must not change pts3d, colors, or pixel_indices."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    original_pts3d = result.pts3d.copy()
    original_colors = result.colors.copy()

    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))), \
         patch("collab_splats.pointcloud.bundle_adjustment._run_bundle_adjustment",
               return_value=(np.zeros((5, 3)), refined_ext, refined_intr)):
        out = BundleAdjustment().refine(result)

    np.testing.assert_array_equal(out.pts3d, original_pts3d)
    np.testing.assert_array_equal(out.colors, original_colors)
    assert out.pixel_indices is None


def test_bundle_adjustment_refine_threads_device():
    """device from BundleAdjustmentConfig is passed to both private functions."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))) as mock_tracks, \
         patch("collab_splats.pointcloud.bundle_adjustment._run_bundle_adjustment",
               return_value=(np.zeros((5, 3)), refined_ext, refined_intr)) as mock_ba:
        cfg = BundleAdjustmentConfig(device="cpu", lm_steps=5)
        BundleAdjustment(config=cfg).refine(result)

    _, tracks_kwargs = mock_tracks.call_args
    assert tracks_kwargs["device"] == "cpu"
    _, ba_kwargs = mock_ba.call_args
    assert ba_kwargs["device"] == "cpu"
    assert ba_kwargs["lm_steps"] == 5


def test_bundle_adjustment_config_defaults():
    """BundleAdjustment() with no args uses default config."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
    ba = BundleAdjustment()
    assert isinstance(ba.config, BundleAdjustmentConfig)
    assert ba.config.device is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py::test_bundle_adjustment_refine_returns_feedforward_result -v 2>&1 | tail -10
```
Expected: FAIL — `cannot import name 'BundleAdjustment' from 'collab_splats.pointcloud.bundle_adjustment'`

- [ ] **Step 3: Implement `BundleAdjustment` class in `bundle_adjustment.py`**

Add to top-level imports in `bundle_adjustment.py`:
```python
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .feedforward.base import FeedforwardResult
```

(Remove the existing `from dataclasses import dataclass` line and replace with `from dataclasses import dataclass, replace`; add `TYPE_CHECKING` import.)

Then add a new section at the bottom of `bundle_adjustment.py` before `__all__`:

```python
########################################################
########## BundleAdjustment class #####################
########################################################


class BundleAdjustment:
    """Refines camera poses via VGGSfM track extraction + LM bundle adjustment.

    Method-agnostic: works with any FeedforwardResult regardless of source creator.
    Does NOT reproject pts3d — call creator.reproject(result) after if needed.
    """

    def __init__(self, config: "BundleAdjustmentConfig | None" = None) -> None:
        self.config = config or BundleAdjustmentConfig()

    def refine(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Refine poses; return updated FeedforwardResult with new extrinsics/intrinsics.

        Only extrinsics and intrinsics are updated. pts3d, colors, and pixel_indices
        are unchanged — call creator.reproject(result) after to re-extract pts3d.
        """
        cfg = self.config
        extrinsics_3x4 = result.extrinsics[:, :3, :]
        image_size = (result.model_height, result.model_width)

        # Extract 2D tracks via VGGSfM tracker; conf+world_points guide keypoint sampling
        tracks, vis_scores, pts3d_kp = _extract_tracks_vggsfm(
            result.images, result.conf, result.world_points,
            max_query_pts=cfg.max_query_pts,
            query_frame_num=cfg.query_frame_num,
            device=cfg.device,
        )

        # Run LM bundle adjustment to refine poses and intrinsics
        _, refined_ext, refined_intr = _run_bundle_adjustment(
            pts3d_kp, extrinsics_3x4, result.intrinsics,
            tracks, vis_scores, image_size,
            max_reproj_error=cfg.max_reproj_error,
            lm_steps=cfg.lm_steps,
            shared_camera=cfg.shared_camera,
            min_inliers_per_frame=cfg.min_inliers_per_frame,
            device=cfg.device,
        )

        # Pad refined (N, 3, 4) extrinsics to (N, 4, 4) for FeedforwardResult convention
        n = refined_ext.shape[0]
        bottom = np.tile([[0, 0, 0, 1]], (n, 1, 1)).astype(np.float32)
        refined_ext_4x4 = np.concatenate([refined_ext, bottom], axis=1)
        return replace(result, extrinsics=refined_ext_4x4, intrinsics=refined_intr)
```

Update `__all__`:
```python
__all__ = ["BundleAdjustment", "BundleAdjustmentConfig"]
```

- [ ] **Step 4: Run all new tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v 2>&1 | tail -20
```
Expected: all pass (or skipped).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_bundle_adjustment.py
git commit -m "feat(ba): add BundleAdjustment plain class with refine(result) -> FeedforwardResult"
```

---

### Task 4: Add `reproject()` to `BaseFeedforwardCreator`; rename `_reproject_ba` → `_reproject`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Modify: `tests/pointcloud/test_wrappers.py`

- [ ] **Step 1: Update tests to expect `_reproject` not `_reproject_ba`**

In `test_wrappers.py`, make these replacements:
```python
# Change:
def test_vggtx_has_reproject_ba():
    """VGGTXCreator must implement _reproject_ba."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    assert hasattr(VGGTXCreator, "_reproject_ba")

# To:
def test_vggtx_has_reproject():
    """VGGTXCreator must implement _reproject."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    assert hasattr(VGGTXCreator, "_reproject")
```

```python
# Change:
def test_mapanything_has_reproject_ba():
    """MapAnythingCreator must implement _reproject_ba."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    assert hasattr(MapAnythingCreator, "_reproject_ba")

# To:
def test_mapanything_has_reproject():
    """MapAnythingCreator must implement _reproject."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    assert hasattr(MapAnythingCreator, "_reproject")
```

Also add a test for the new `reproject()` public method:
```python
def test_base_creator_reproject_returns_updated_pts3d():
    """creator.reproject(result) re-extracts pts3d/colors using refined poses."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from unittest.mock import patch, MagicMock
    import dataclasses

    creator = VGGTXCreator.__new__(VGGTXCreator)
    creator.max_points = 500_000
    creator.conf_threshold = 1.0
    creator.raw_outputs = {"depth": np.ones((2, 8, 8, 1)), "depth_conf": np.ones((2, 8, 8))}
    creator.image_paths = [Path("a.jpg"), Path("b.jpg")]
    creator.original_coords = np.zeros((2, 6), dtype=np.float32)

    result = _make_ff_result(
        extrinsics=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
    )

    new_pts = np.ones((5, 3), dtype=np.float32)
    new_colors = np.zeros((5, 3), dtype=np.uint8)

    with patch.object(creator, "_reproject", return_value=(new_pts, new_colors)):
        out = creator.reproject(result)

    assert out.pts3d is new_pts
    assert out.colors is new_colors
    # extrinsics/intrinsics unchanged by reproject
    np.testing.assert_array_equal(out.extrinsics, result.extrinsics)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_vggtx_has_reproject tests/pointcloud/test_wrappers.py::test_mapanything_has_reproject tests/pointcloud/test_wrappers.py::test_base_creator_reproject_returns_updated_pts3d -v 2>&1 | tail -10
```
Expected: all FAIL

- [ ] **Step 3: Rename `_reproject_ba` → `_reproject` in `base.py`**

In `collab_splats/pointcloud/feedforward/base.py`:

1. Rename the abstract method:
```python
    @abstractmethod
    def _reproject(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space point cloud using bundle-adjusted camera poses.

        Called by reproject() after BundleAdjustment refines extrinsics.

        Args:
            raw_outputs:     Raw model outputs stored from _forward().
            extrinsics_3x4: (N, 3, 4) refined world-to-camera matrices.
            intrinsics:      (N, 3, 3) refined camera intrinsics.

        Returns:
            (pts3d, colors) — (P, 3) float32 and (P, 3) uint8.
        """
        ...
```

2. Add `reproject()` public method before `_reproject`:
```python
    def reproject(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Re-extract pts3d/colors using refined poses stored in result.

        Uses self.raw_outputs from the last run() or run_inference() call.
        Only call when result.pixel_indices is not None.
        """
        pts3d, colors = self._reproject(
            self.raw_outputs, result.extrinsics[:, :3, :], result.intrinsics
        )
        return replace(result, pts3d=pts3d, colors=colors)
```

3. Add `from dataclasses import replace` to `base.py` imports (if not already present). Check existing imports — `base.py` uses `dataclass` and `field` from `dataclasses`. Change:
```python
from dataclasses import dataclass, field
```
to:
```python
from dataclasses import dataclass, field, replace
```

- [ ] **Step 4: Rename `_reproject_ba` → `_reproject` in `vggtx.py`**

In `collab_splats/pointcloud/feedforward/vggtx.py`, rename:
```python
    def _reproject(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses."""
```
(The body is identical — only the `def` name changes.)

- [ ] **Step 5: Rename `_reproject_ba` → `_reproject` in `mapanything.py`**

In `collab_splats/pointcloud/feedforward/mapanything.py`, rename:
```python
    def _reproject(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses."""
```
(Body is identical — only the `def` name changes.)

- [ ] **Step 6: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_vggtx_has_reproject tests/pointcloud/test_wrappers.py::test_mapanything_has_reproject tests/pointcloud/test_wrappers.py::test_base_creator_reproject_returns_updated_pts3d -v 2>&1 | tail -10
```
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py collab_splats/pointcloud/feedforward/vggtx.py collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/test_wrappers.py
git commit -m "refactor(feedforward): rename _reproject_ba -> _reproject; add BaseFeedforwardCreator.reproject()"
```

---

### Task 5: Update `wrappers.py` — delete old `BundleAdjustment`; add `LoopClosure.run()` + `LoopClosure.reproject()`

**Files:**
- Modify: `collab_splats/pointcloud/wrappers.py`
- Modify: `tests/pointcloud/test_wrappers.py`

- [ ] **Step 1: Write tests for new LoopClosure methods**

Add to `test_wrappers.py`:
```python
def test_loop_closure_reproject_delegates_to_base():
    """LoopClosure.reproject() must delegate to base.reproject()."""
    from collab_splats.pointcloud.wrappers import LoopClosure

    mock_base = MagicMock()
    expected = _make_ff_result()
    mock_base.reproject.return_value = expected

    lc = LoopClosure(mock_base)
    result_in = _make_ff_result()
    out = lc.reproject(result_in)

    mock_base.reproject.assert_called_once_with(result_in)
    assert out is expected


def test_loop_closure_run_returns_feedforward_result():
    """LoopClosure.run() must call the 4-step pipeline and return FeedforwardResult."""
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    mock_base = MagicMock()
    ff = _make_ff_result()
    mock_base.outputs = ff
    mock_base.raw_outputs = {}

    lc = LoopClosure(mock_base)
    with patch.object(lc, "run_inference"):
        out = lc.run(Path("/fake/images"))

    mock_base.load_model.assert_called_once()
    mock_base.setup_inference.assert_called_once_with(Path("/fake/images"))
    mock_base.postprocess.assert_called_once()
    assert out is ff
```

- [ ] **Step 2: Delete old BA tests that reference wrappers.BundleAdjustment as a creator wrapper**

Remove these test functions from `test_wrappers.py` (they test the deleted class):
- `test_bundle_adjustment_raises_if_images_none`
- `test_bundle_adjustment_calls_extract_tracks_and_run_ba`
- `test_bundle_adjustment_outputs_proxies_to_base`
- `test_bundle_adjustment_raw_outputs_proxies_to_base`
- `test_bundle_adjustment_config_passed_to_run_ba`
- `test_bundle_adjustment_uses_reproject_pixels_when_pixel_indices_set`
- `test_bundle_adjustment_wraps_loop_closure`
- `test_make_creator_with_ba`
- `test_make_creator_with_lc_and_ba`

Also remove the `_make_mock_creator` helper if nothing else uses it (check first).

- [ ] **Step 3: Run new tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py::test_loop_closure_reproject_delegates_to_base tests/pointcloud/test_wrappers.py::test_loop_closure_run_returns_feedforward_result -v 2>&1 | tail -10
```
Expected: FAIL — `LoopClosure` has no `reproject` or `run`

- [ ] **Step 4: Update `wrappers.py`**

Delete the entire `BundleAdjustment(BasePointcloudCreator)` class (lines 19–190 in current file).

Update imports at top of `wrappers.py` — remove unused imports. Keep only what `LoopClosure` needs:
```python
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePointcloudCreator, PointcloudResult
from .feedforward import FeedforwardResult
from .loop_closure import LoopClosureConfig
```

Update `__all__`:
```python
__all__ = ["LoopClosure"]
```

Add `run()` and `reproject()` to `LoopClosure` (after `reconstruct()` and before `run_inference()`):

```python
    def run(self, image_dir: Path) -> FeedforwardResult:
        """Run inference pipeline without COLMAP; return FeedforwardResult.

        Mirrors BaseFeedforwardCreator.run(). For LC path, also applies
        _dedup_rows to align M-row overlap arrays with N-row global extrinsics.
        """
        image_dir = Path(image_dir)
        self.load_model()
        self.setup_inference(image_dir)
        self.run_inference()
        self.postprocess()
        result = self.base.outputs

        # LC merge_submap_outputs stores _dedup_rows to map M merged rows → N unique frames.
        # Apply dedup so images/conf/world_points/intrinsics align with global extrinsics.
        raw = self.base.raw_outputs
        dedup = raw.get("_dedup_rows") if isinstance(raw, dict) else None
        if dedup is not None:
            N = result.extrinsics.shape[0]
            kwargs: dict = {}
            if result.images is not None and len(result.images) != N:
                kwargs["images"] = result.images[dedup]
            if result.conf is not None and hasattr(result.conf, "__len__") and len(result.conf) != N:
                kwargs["conf"] = result.conf[dedup]
            if result.world_points is not None and result.world_points.shape[0] != N:
                kwargs["world_points"] = result.world_points[dedup]
            if result.intrinsics is not None and result.intrinsics.shape[0] != N:
                kwargs["intrinsics"] = result.intrinsics[dedup]
            if kwargs:
                result = dataclasses.replace(result, **kwargs)
            self.base.outputs = result

        return result

    def reproject(self, result: FeedforwardResult) -> FeedforwardResult:
        """Re-extract pts3d/colors using refined poses; delegates to base.reproject()."""
        return self.base.reproject(result)
```

Also add `_reproject` delegation (needed for backward compat during transition — `LoopClosure` proxied `_reproject_ba` before):
```python
    def _reproject(self, raw_outputs: Any, ext: Any, intr: Any) -> Any:
        """Delegate to base._reproject."""
        return self.base._reproject(raw_outputs, ext, intr)
```

And remove the old `_reproject_ba` delegation method from `LoopClosure`.

- [ ] **Step 5: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py -v 2>&1 | tail -25
```
Expected: all pass. Some old BA tests were deleted; new ones pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py tests/pointcloud/test_wrappers.py
git commit -m "refactor(wrappers): delete BundleAdjustment wrapper class; add LoopClosure.run() + reproject()"
```

---

### Task 6: Update `collab_splats/pointcloud/__init__.py`

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py`

- [ ] **Step 1: Update imports and `make_creator`**

Replace the current `__init__.py` contents with:

```python
# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult, _colmap_recon_to_result
from .sfm import ColmapCreator, HlocCreator
from .feedforward import BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator

try:
    from .feedforward import VGGTOmegaCreator
    _OMEGA_AVAILABLE = True
except ImportError:
    _OMEGA_AVAILABLE = False

from .bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
from .loop_closure import LoopClosureConfig
from .wrappers import LoopClosure
from .localization import (
    BaseRetrievalExtractor,
    CameraLocalizer,
    DiskExtractor,
    LocalFeatures,
    PECLIPExtractor,
    XFeatExtractor,
)
from .utils import compute_obb_from_points, get_points_in_mask

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "colmap":      ColmapCreator,
    "hloc":        HlocCreator,
    "mapanything": MapAnythingCreator,
    "vggtx":       VGGTXCreator,
}
if _OMEGA_AVAILABLE:
    _REGISTRY["vggt_omega"] = VGGTOmegaCreator


def get_creator(name: str) -> type[BasePointcloudCreator]:
    """Get a pointcloud creator class by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def make_creator(
    name: str,
    *,
    use_lc: bool = False,
    lc_config=None,
    **kwargs,
):
    """Construct a pointcloud creator, optionally wrapped with LoopClosure."""
    creator = get_creator(name)(**kwargs)
    if use_lc:
        creator = LoopClosure(creator, config=lc_config)
    return creator


__all__ = [
    "BasePointcloudCreator",
    "BaseRetrievalExtractor",
    "BaseFeedforwardCreator",
    "BundleAdjustment",
    "BundleAdjustmentConfig",
    "CameraLocalizer",
    "CoordinateFrame",
    "ColmapCreator",
    "DiskExtractor",
    "HlocCreator",
    "PECLIPExtractor",
    "LoopClosure",
    "LoopClosureConfig",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "XFeatExtractor",
    "compute_obb_from_points",
    "get_points_in_mask",
    "get_creator",
    "make_creator",
]
```

- [ ] **Step 2: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v 2>&1 | tail -30
```
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/pointcloud/__init__.py
git commit -m "refactor(pointcloud): import BundleAdjustment from bundle_adjustment; remove use_ba from make_creator"
```

---

### Task 7: Update notebook `bundle_adjustment.ipynb`

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`

- [ ] **Step 1: Find and replace old API calls in the notebook**

Search for cells containing `ba.reconstruct` or `creator.refine` or `BundleAdjustment(creator`. Replace with the new composition pattern.

Old pattern:
```python
ba = BundleAdjustment(creator, config=cfg)
result = ba.reconstruct(image_dir, output_dir)
```

New pattern:
```python
ba = BundleAdjustment(config=cfg)
ff = creator.run(image_dir)
ff = ba.refine(ff)
if ff.pixel_indices is not None:
    ff = creator.reproject(ff)
creator.outputs = ff
result = creator.build_colmap(output_dir)
```

Also update the import cell: remove `run_bundle_adjustment` if present; update `BundleAdjustment` import to come from `collab_splats.pointcloud.bundle_adjustment` or `collab_splats.pointcloud`.

- [ ] **Step 2: Run the updated notebook cells (dry-run / kernel check)**

```bash
/opt/conda/envs/nerfstudio/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb --output /tmp/ba_test.ipynb 2>&1 | tail -5
```
If image data is unavailable the execute will fail — that is expected. Verify it fails only on the data-loading cell, not on import/API cells.

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
git commit -m "docs(notebook): update bundle_adjustment to new BA composition pattern"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full pointcloud test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v 2>&1 | tail -30
```
Expected: all pass (or only skipped for missing CUDA/bae).

- [ ] **Step 2: Verify public API**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud import BundleAdjustment, BundleAdjustmentConfig, LoopClosure, make_creator
from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
ba = BundleAdjustment(config=BundleAdjustmentConfig(device='cpu', lm_steps=5))
print('BundleAdjustment OK:', ba.config)
"
```
Expected: prints config, no import errors.

- [ ] **Step 3: Verify old public names are gone**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud import bundle_adjustment as ba
assert not hasattr(ba, 'run_bundle_adjustment'), 'run_bundle_adjustment should be private'
assert not hasattr(ba, 'extract_tracks_vggsfm'), 'extract_tracks_vggsfm should be private'
assert hasattr(ba, '_run_bundle_adjustment'), '_run_bundle_adjustment missing'
assert hasattr(ba, '_extract_tracks_vggsfm'), '_extract_tracks_vggsfm missing'
print('API surface correct')
"
```
Expected: prints `API surface correct`

- [ ] **Step 4: Commit if any stray changes**

```bash
git status
```
If clean, nothing to do. Otherwise commit remaining changes.
