# Mesh TSDF Adapter Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the two divergent TSDF mesh paths into one adapter so `Reconstructor.mesh()` stops fusing model-resolution depth against original-resolution COLMAP intrinsics and stops dividing already-`[0,1]` RGB by 255.

**Architecture:** `_feedforward_to_tsdf_inputs` (`collab_splats/mesh/utils.py`) becomes the single source of TSDF inputs for both callers. It reads `result.depth` and `result.images` straight off the `FeedforwardResult` — deleting the `world_points → camera Z` projection and the PIL re-read/crop/`/255` branch, both measured redundant. `Reconstructor._run_tsdf_mesh` keeps its signature and call site but stops building inputs itself: it loads the zarr, overrides `ff.extrinsics` with the COLMAP poses (COLMAP stays the pose authority — BA and LC land there), and delegates to `pointcloud_to_mesh`. Two one-line guards make the remaining failure modes loud instead of silent.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), numpy, torch, open3d `ScalableTSDFVolume`, zarr v3, pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-mesh-tsdf-adapter-convergence-design.md`

---

## File Structure

| File | Change | Responsibility after the change |
|---|---|---|
| `collab_splats/mesh/utils.py:411-450` | Modify | `_feedforward_to_tsdf_inputs` — the one adapter. Reads `depth`/`images`, inverts poses, passes `intrinsics` through untouched. No disk IO. |
| `collab_splats/mesh/utils.py:453-482` | Modify (docstring) | `pointcloud_to_mesh` — unchanged behaviour, docstring no longer promises `world_points`. |
| `collab_splats/mesh/utils.py:11` | Delete | `from PIL import Image as PILImage` — only used by the deleted branch. |
| `collab_splats/mesh/tsdf.py:38-53` | Modify | `Open3DTSDFFusion.create` — gains an RGB-range guard. |
| `collab_splats/wrapper/reconstructor.py:317-359` | Modify | `_run_tsdf_mesh` — loads zarr, pins COLMAP poses, guards frame count, delegates. |
| `tests/mesh/test_adapter.py` | Modify | Helpers populate `images`/`depth`; crop test deleted; adapter-contract tests added. |
| `tests/mesh/test_tsdf.py` | Create | RGB-range guard test. |
| `tests/wrapper/test_reconstructor.py:604-630` | Modify | `_run_tsdf_mesh` tests patch `pointcloud_to_mesh`'s creator; K/pose/frame-count regression tests added. |

---

### Task 1: One adapter — `_feedforward_to_tsdf_inputs` reads `depth` and `images`

**Files:**
- Modify: `collab_splats/mesh/utils.py:411-482`
- Test: `tests/mesh/test_adapter.py`

- [ ] **Step 1: Rewrite the test file**

The existing helpers build a `FeedforwardResult` with `world_points` and real PNGs on disk but **no `images` and no `depth`** — every test in the file depends on the derivations being deleted. Replace the whole file with this:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.mesh import pointcloud_to_mesh
from collab_splats.mesh.base import MeshResult
from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _make_result(N=2, H=32, W=32, depth_z=1.5, with_depth=True, with_images=True):
    """Minimal FeedforwardResult carrying the model-res depth + RGB the adapter consumes."""
    rng = np.random.default_rng(42)
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)  # identity w2c

    # depth/images are what the model actually produced — no disk IO, no re-derivation
    depth = np.full((N, H, W), depth_z, dtype=np.float32) if with_depth else None
    images = torch.from_numpy(rng.random((N, 3, H, W)).astype(np.float32)) if with_images else None

    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = float(W)
    intrinsics[:, 0, 2] = W / 2
    intrinsics[:, 1, 2] = H / 2

    return FeedforwardResult(
        points=rng.random((10, 3)).astype(np.float32),
        colors=(rng.random((10, 3)) * 255).astype(np.uint8),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=[Path(f"frame_{i:04d}.png") for i in range(N)],
        original_coords=np.tile([0, 0, W, H, W, H], (N, 1)).astype(np.float32),
        model_width=W,
        model_height=H,
        images=images,
        depth=depth,
    )


def test_adapter_returns_depth_and_intrinsics_unchanged():
    """Depth and K pass through untouched — both are model-resolution and already aligned."""
    result = _make_result(N=2, H=8, W=6, depth_z=2.25)
    depths, _, _, intrinsics = _feedforward_to_tsdf_inputs(result)

    np.testing.assert_array_equal(depths, result.depth)
    np.testing.assert_array_equal(intrinsics, result.intrinsics)


def test_adapter_returns_rgb_in_unit_range_hwc():
    """images is (N, 3, H, W) in [0, 1]; the adapter transposes it, it does not rescale it."""
    result = _make_result(N=2, H=8, W=6)
    _, rgbs, _, _ = _feedforward_to_tsdf_inputs(result)

    assert rgbs.shape == (2, 8, 6, 3)
    assert rgbs.dtype == np.float32
    assert rgbs.max() <= 1.0
    np.testing.assert_allclose(rgbs, result.images.numpy().transpose(0, 2, 3, 1), rtol=0, atol=0)


def test_adapter_inverts_extrinsics_to_c2w():
    """c2w is the inverse of the stored w2c extrinsics."""
    result = _make_result(N=2, H=8, W=6)
    result.extrinsics[:, :3, 3] = [0.5, -1.0, 2.0]
    _, _, c2w, _ = _feedforward_to_tsdf_inputs(result)

    identity = np.eye(4, dtype=np.float32)[None].repeat(len(c2w), axis=0)
    np.testing.assert_allclose(c2w @ result.extrinsics, identity, atol=1e-5)


def test_adapter_raises_on_missing_depth():
    result = _make_result(with_depth=False)
    with pytest.raises(ValueError, match="depth"):
        _feedforward_to_tsdf_inputs(result)


def test_adapter_raises_on_missing_images():
    result = _make_result(with_images=False)
    with pytest.raises(ValueError, match="images"):
        _feedforward_to_tsdf_inputs(result)


def test_pointcloud_to_mesh_returns_mesh_result(tmp_path):
    result = _make_result(N=2, H=32, W=32)
    mesh_result = pointcloud_to_mesh(
        result,
        tmp_path / "mesh",
        method="open3d_tsdf",
        voxel_size=0.05,
        sdf_trunc=0.2,
        clean_repair=False,
    )
    assert isinstance(mesh_result, MeshResult)
    assert mesh_result.mesh_path.exists()


def test_pointcloud_to_mesh_invalid_method(tmp_path):
    result = _make_result()
    with pytest.raises(ValueError, match="Unknown mesh method"):
        pointcloud_to_mesh(result, tmp_path / "mesh", method="nonexistent")
```

Deleted along with the old file: `_make_result_with_crop`, `test_tsdf_rgb_crop_applied_for_original_pixel_coords` (asserts the crop branch — removed with the branch), and `test_pointcloud_to_mesh_raises_on_none_world_points` (the adapter no longer touches `world_points`).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_adapter.py -v -p no:randomly
```

Expected: `ImportError` on `_feedforward_to_tsdf_inputs`? No — the symbol already exists, so expect **collection to pass and failures** in `test_adapter_returns_depth_and_intrinsics_unchanged`, `test_adapter_returns_rgb_in_unit_range_hwc`, `test_adapter_raises_on_missing_depth`, `test_adapter_raises_on_missing_images` with `ValueError: result.world_points is None...` (the old adapter demands `world_points`, which the new helper no longer sets).

- [ ] **Step 3: Rewrite the adapter**

Replace `collab_splats/mesh/utils.py:411-450` (the whole `_feedforward_to_tsdf_inputs` body) with:

```python
def _feedforward_to_tsdf_inputs(
    result: FeedforwardResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a FeedforwardResult into (depths, rgbs, c2w, intrinsics) for TSDF fusion.

    Everything returned is at model resolution and mutually pixel-aligned: depth, images and
    intrinsics all come off the same forward pass. Nothing is re-derived or re-read from disk.
    """
    # depth and images are the model's own outputs — all three backends populate both, so a
    # fallback derivation here would be dead code (and was measurably worse: see the design doc)
    if result.depth is None:
        raise ValueError(
            "result.depth is None — cannot mesh. Access creator.outputs after reconstruct(), "
            "or load the zarr with load_depth=True."
        )
    if result.images is None:
        raise ValueError(
            "result.images is None — cannot mesh. Access creator.outputs after reconstruct(), "
            "or load the zarr with load_images=True."
        )

    depths = np.ascontiguousarray(result.depth, dtype=np.float32)  # (N, H, W)

    # images is (N, 3, H, W) in [0, 1] — torch from a live creator, numpy from load_zarr
    imgs = result.images
    if hasattr(imgs, "numpy"):
        imgs = imgs.detach().cpu().numpy()
    rgbs = np.ascontiguousarray(imgs.transpose(0, 2, 3, 1), dtype=np.float32)  # (N, H, W, 3)

    c2w = invert_poses(result.extrinsics).astype(np.float32)
    return depths, rgbs, c2w, result.intrinsics.copy()
```

- [ ] **Step 4: Fix the `pointcloud_to_mesh` docstring**

`collab_splats/mesh/utils.py:466` and `:475` still promise `world_points`. Change those two lines:

```python
        result:  FeedforwardResult with populated depth + images. Access
                 creator.outputs after reconstruct(), or FeedforwardResult.load_zarr(
                 path, load_images=True).
```

```python
        ValueError: if result.depth/result.images are None or method is not in the registry.
```

- [ ] **Step 5: Delete the now-unused PIL import**

`collab_splats/mesh/utils.py:11` — `from PIL import Image as PILImage` was used only at the deleted `:438`/`:445`. Delete the line, then confirm:

```bash
grep -n "PILImage" collab_splats/mesh/utils.py
```

Expected: no output.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/mesh/ -v -p no:randomly
```

Expected: all PASS.

- [ ] **Step 7: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/mesh/utils.py tests/mesh/test_adapter.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/mesh/utils.py tests/mesh/test_adapter.py
git add collab_splats/mesh/utils.py tests/mesh/test_adapter.py
git commit -m "fix(mesh): read depth and RGB straight off FeedforwardResult

The adapter re-derived depth by projecting world_points into camera Z and
re-read RGB from disk with a crop+resize+/255 pass. Both are redundant:
result.depth matches the projection to 2e-6, and result.images is the
tensor the model actually saw, so it is pixel-aligned with depth by
construction. The disk read also outlived the images/ dir (ab11b37).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Note repo-wide `black .` is forbidden here — the venv's black 26.5.1 is newer than the repo's formatting. Format only the touched files.

---

### Task 2: Guard — `Open3DTSDFFusion.create` rejects `[0, 255]` RGB

**Files:**
- Modify: `collab_splats/mesh/tsdf.py:46-56`
- Test: `tests/mesh/test_tsdf.py` (create)

`create` already documents `rgbs` as `[0, 1]` and multiplies by 255 internally. Passing `[0, 255]` produces a silently black mesh — exactly the second half of this bug. Make it raise.

- [ ] **Step 1: Write the failing test**

Create `tests/mesh/test_tsdf.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from collab_splats.mesh.tsdf import Open3DTSDFFusion


def test_create_rejects_rgb_in_0_255_range(tmp_path):
    """rgbs is documented [0, 1]; [0, 255] silently fuses a black mesh, so refuse it."""
    fusion = Open3DTSDFFusion(output_dir=tmp_path, voxel_size=0.05, sdf_trunc=0.2)
    depths = np.ones((1, 8, 8), dtype=np.float32)
    rgbs = np.full((1, 8, 8, 3), 200.0, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)[None]
    intrinsics = np.eye(3, dtype=np.float32)[None]

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fusion.create(depths=depths, rgbs=rgbs, c2w=c2w, intrinsics=intrinsics)


def test_create_accepts_rgb_in_unit_range(tmp_path):
    fusion = Open3DTSDFFusion(output_dir=tmp_path, voxel_size=0.05, sdf_trunc=0.2)
    depths = np.ones((1, 8, 8), dtype=np.float32)
    rgbs = np.full((1, 8, 8, 3), 0.5, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)[None]
    intrinsics = np.eye(3, dtype=np.float32)[None]

    result = fusion.create(depths=depths, rgbs=rgbs, c2w=c2w, intrinsics=intrinsics)
    assert result.mesh_path.exists()
```

- [ ] **Step 2: Run to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_tsdf.py -v -p no:randomly
```

Expected: `test_create_rejects_rgb_in_0_255_range` FAILS with `DID NOT RAISE <class 'ValueError'>`. `test_create_accepts_rgb_in_unit_range` PASSES.

- [ ] **Step 3: Add the guard**

In `collab_splats/mesh/tsdf.py`, immediately after the docstring of `create` (currently line 53) and before `self.output_dir.mkdir(...)`:

```python
        # rgbs is scaled to uint8 below; [0, 255] input would wrap to a black mesh instead of
        # failing, which is how the Reconstructor path shipped black meshes unnoticed
        if rgbs.size and float(np.nanmax(rgbs)) > 1.5:
            raise ValueError(
                f"rgbs must be in [0, 1], got max {float(np.nanmax(rgbs)):.3f}. "
                "Pass FeedforwardResult.images directly — it is already normalised."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run to verify it passes**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/mesh/ -v -p no:randomly
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/mesh/tsdf.py tests/mesh/test_tsdf.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/mesh/tsdf.py tests/mesh/test_tsdf.py
git add collab_splats/mesh/tsdf.py tests/mesh/test_tsdf.py
git commit -m "fix(mesh): reject [0, 255] RGB in Open3DTSDFFusion.create

create() documents rgbs as [0, 1] and scales to uint8 internally, so a
[0, 255] caller fused a fully black mesh with no error. Raise instead.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `_run_tsdf_mesh` delegates to the adapter

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:317-359`
- Test: `tests/wrapper/test_reconstructor.py:604-630`

This is where the bug lives. `_run_tsdf_mesh` receives a **`PointcloudResult`** (pycolmap-backed, original-resolution K after `_rescale_reconstruction_to_original_dimensions`) and pairs its `.intrinsics` with model-resolution depth from the zarr. Signature and call site stay; the body stops building TSDF inputs.

- [ ] **Step 1: Write the failing tests**

Replace `test_run_tsdf_mesh_passes_clean_repair_to_the_fusion` (`tests/wrapper/test_reconstructor.py:604-630`) with these four tests. They patch `collab_splats.mesh.get_mesh_creator`, which is what `pointcloud_to_mesh` resolves the fusion through.

```python
def _tsdf_mesh_doubles(n_colmap=2, n_zarr=2, model_hw=(8, 8)):
    """(PointcloudResult double with original-res K, FeedforwardResult double with model-res K)."""
    H, W = model_hw
    # COLMAP camera after _rescale_reconstruction_to_original_dimensions: 2x the model grid
    result = MagicMock()
    result.extrinsics = np.eye(4, dtype=np.float32)[None].repeat(n_colmap, axis=0)
    result.extrinsics[:, 0, 3] = 7.0  # distinctive, so we can tell the poses apart
    K_orig = np.eye(3, dtype=np.float32)[None].repeat(n_colmap, axis=0)
    K_orig[:, 0, 0] = K_orig[:, 1, 1] = 2.0 * W
    K_orig[:, 0, 2] = W  # cx = W → outside a W-wide image
    K_orig[:, 1, 2] = H
    result.intrinsics = K_orig

    K_model = np.eye(3, dtype=np.float32)[None].repeat(n_zarr, axis=0)
    K_model[:, 0, 0] = K_model[:, 1, 1] = float(W)
    K_model[:, 0, 2] = W / 2
    K_model[:, 1, 2] = H / 2
    ff = FeedforwardResult(
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        extrinsics=np.eye(4, dtype=np.float32)[None].repeat(n_zarr, axis=0),
        intrinsics=K_model,
        image_paths=[Path(f"frame_{i:04d}.png") for i in range(n_zarr)],
        original_coords=np.tile([0, 0, W, H, W, H], (n_zarr, 1)).astype(np.float32),
        model_width=W,
        model_height=H,
        images=torch.zeros((n_zarr, 3, H, W), dtype=torch.float32),
        depth=np.ones((n_zarr, H, W), dtype=np.float32),
    )
    return result, ff


def test_run_tsdf_mesh_fuses_zarr_intrinsics_not_colmap(tmp_path):
    """Regression: COLMAP K is original-res, zarr depth is model-res. Fuse with the zarr K."""
    from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

    result, ff = _tsdf_mesh_doubles()
    mock_creator = MagicMock()
    with (
        patch(
            "collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr",
            return_value=ff,
        ),
        patch("collab_splats.mesh.get_mesh_creator", return_value=mock_creator),
    ):
        _run_tsdf_mesh(
            result=result,
            feedforward_zarr=tmp_path / "feedforward.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=2.0,
            clean_repair=False,
        )

    fused_K = mock_creator.create.call_args[0][3]
    np.testing.assert_allclose(fused_K, ff.intrinsics)
    assert not np.allclose(fused_K, result.intrinsics)


def test_run_tsdf_mesh_uses_colmap_poses(tmp_path):
    """COLMAP stays the pose authority — BA and LC corrections land there, not in the zarr."""
    from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

    result, ff = _tsdf_mesh_doubles()
    mock_creator = MagicMock()
    with (
        patch(
            "collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr",
            return_value=ff,
        ),
        patch("collab_splats.mesh.get_mesh_creator", return_value=mock_creator),
    ):
        _run_tsdf_mesh(
            result=result,
            feedforward_zarr=tmp_path / "feedforward.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=2.0,
            clean_repair=False,
        )

    fused_c2w = mock_creator.create.call_args[0][2]
    np.testing.assert_allclose(fused_c2w, np.linalg.inv(result.extrinsics), atol=1e-5)


def test_run_tsdf_mesh_raises_on_frame_count_mismatch(tmp_path):
    """Stage re-runs can pair a COLMAP dir with a feedforward.zarr from a different run."""
    from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

    result, ff = _tsdf_mesh_doubles(n_colmap=3, n_zarr=2)
    with (
        patch(
            "collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr",
            return_value=ff,
        ),
        pytest.raises(ValueError, match="feedforward.zarr"),
    ):
        _run_tsdf_mesh(
            result=result,
            feedforward_zarr=tmp_path / "feedforward.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=2.0,
            clean_repair=False,
        )


def test_run_tsdf_mesh_passes_clean_repair_to_the_fusion(tmp_path):
    from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

    result, ff = _tsdf_mesh_doubles()
    mock_creator = MagicMock()
    with (
        patch(
            "collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr",
            return_value=ff,
        ),
        patch("collab_splats.mesh.get_mesh_creator", return_value=mock_creator) as mock_get,
    ):
        _run_tsdf_mesh(
            result=result,
            feedforward_zarr=tmp_path / "feedforward.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=1.0,
            clean_repair=True,
        )

    assert mock_get.call_args.kwargs["clean_repair"] is True
    assert mock_get.call_args.kwargs["depth_trunc"] == 1.0
```

Add these imports at the top of `tests/wrapper/test_reconstructor.py` if not already present (check first — `numpy`, `MagicMock`, `patch`, `pytest`, `Path` almost certainly are):

```python
import torch

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -k "run_tsdf_mesh" -v -p no:randomly
```

Expected: `test_run_tsdf_mesh_fuses_zarr_intrinsics_not_colmap` FAILS (`mock_creator.create` never called — the old body constructs `Open3DTSDFFusion` directly, so `get_mesh_creator` is not reached; `call_args` is `None` → `TypeError: 'NoneType' object is not subscriptable`). Same for the pose test and `test_run_tsdf_mesh_passes_clean_repair_to_the_fusion`. `test_run_tsdf_mesh_raises_on_frame_count_mismatch` FAILS with `DID NOT RAISE`.

- [ ] **Step 3: Rewrite `_run_tsdf_mesh`**

Replace the whole body of `_run_tsdf_mesh` (`collab_splats/wrapper/reconstructor.py:317-359`) with:

```python
def _run_tsdf_mesh(
    result: "PointcloudResult",
    feedforward_zarr: Path,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    clean_repair: bool = False,
) -> Path:
    """Fuse depth + RGB from feedforward.zarr into a TSDF mesh, using COLMAP poses."""
    from collab_splats.mesh.utils import pointcloud_to_mesh
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    # world_points is the largest array in the store and the mesh path no longer reads it
    ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True, load_world_points=False)

    # COLMAP is the pose authority — BA and loop-closure corrections land in the reconstruction,
    # not back in the zarr. Intrinsics stay the zarr's: COLMAP's camera was rescaled to original
    # resolution by build_colmap, and the zarr's depth/RGB are at model resolution.
    if result.extrinsics.shape[0] != ff.depth.shape[0]:
        raise ValueError(
            f"Frame-count mismatch: COLMAP reconstruction has {result.extrinsics.shape[0]} "
            f"images but {feedforward_zarr} has {ff.depth.shape[0]}. They are from different "
            "runs — re-run the pointcloud stage, or point --stages mesh at the matching scene."
        )
    ff.extrinsics = result.extrinsics

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_result = pointcloud_to_mesh(
        ff,
        output_dir,
        method="open3d_tsdf",
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        clean_repair=clean_repair,
    )
    return mesh_result.mesh_path
```

Then check the module's now-unused imports:

```bash
grep -n "Open3DTSDFFusion\|^import numpy\|np\." collab_splats/wrapper/reconstructor.py | head -30
```

`numpy` stays (used elsewhere in the module). If `Open3DTSDFFusion` no longer appears anywhere, nothing to delete — it was a function-local import.

The spec flags `mesh_cfg["enabled"]` as something that must not reach the creator constructor. `Reconstructor.mesh()` (`reconstructor.py:832-840`) forwards four explicit keys and never splats `**mesh_cfg`, and `_run_tsdf_mesh` above keeps that shape — so `enabled` cannot leak. No change needed; do not introduce a `**mesh_cfg` splat.

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -p no:randomly
```

Expected: all PASS, including the four pre-existing mesh tests at `:527`, `:556`, `:573`, `:591`. `test_mesh_skip_check_matches_tsdf_writer_filename` does a genuine `Open3DTSDFFusion.create` write and must still pass — the writer filename is unchanged.

- [ ] **Step 5: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "fix(wrapper): mesh with model-res intrinsics, not COLMAP's rescaled camera

_run_tsdf_mesh held a PointcloudResult for poses and reached for
.intrinsics on it. COLMAP's camera is original-resolution after
build_colmap's _rescale_reconstruction_to_original_dimensions, while the
depth and RGB it fused come from feedforward.zarr at model resolution.
Measured on data/outputs (2.81x scale): 5.06M -> 1.41M vertices. The same
function also divided already-[0,1] RGB by 255, fusing a black mesh.

Both dated to 6bc9c81 and were invisible while mesh.enabled was false and
meshes came from the dashboard path instead.

Reconstructor now delegates to pointcloud_to_mesh, the adapter the
dashboard already used, so there is one mesh path. COLMAP stays the pose
authority; a frame-count guard catches a zarr from a different run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Full suite and dashboard smoke gate

**Files:** none modified — this is a gate.

- [ ] **Step 1: Run the full suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly -q 2>&1 | tail -20
```

Expected: no failures beyond those listed in `docs/known-test-failures.md`. If ~60 unrelated failures appear, another session may be editing `configs/base.yaml` mid-run — re-run before believing it.

- [ ] **Step 2: Run the dashboard smoke gate**

The dashboard is the other caller of `pointcloud_to_mesh` (`collab_splats/dashboard/pipeline.py:443`) and this is mandatory before committing any change that touches its path.

```bash
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: prints `SMOKE PASS`.

- [ ] **Step 3: If either gate fails, fix and re-run before continuing**

No commit for this task unless a fix was needed; then commit the fix with a `fix(...)` message naming what broke.

---

### Task 5: Live verification on `data/outputs/`

**Files:**
- Modify: `docs/superpowers/specs/2026-08-11-mesh-tsdf-adapter-convergence-design.md` (record the measured baseline)
- Modify: `CLAUDE.md` (In-Flight Work → Recently completed)

The unit tests prove the wiring. This proves the mesh.

`data/outputs/` is the tutorial scene: `frames.zarr` + `feedforward.zarr` at the root, no `colmap/` dir and no backend subdir. `Reconstructor.mesh()` cannot run against it (`_resolve_result()` needs `backend_dir/colmap/sparse/0`), so verify through the adapter — after Task 3 that *is* the code path `_run_tsdf_mesh` takes.

- [ ] **Step 1: Fuse through the adapter and compare against case A**

Write `/tmp/claude-0/-workspace-collab-splats/5304829b-6c29-427d-835f-80d56cbe6980/scratchpad/verify_mesh.py`:

```python
"""Post-fix verification: fuse data/outputs through the production adapter."""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from collab_splats.mesh.utils import pointcloud_to_mesh
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

ZARR = Path("data/outputs/feedforward.zarr")
depth_trunc = float(sys.argv[1])
out = Path(sys.argv[2])


def stats(mesh_path):
    m = o3d.io.read_triangle_mesh(str(mesh_path))
    v = np.asarray(m.vertices)
    c = np.asarray(m.vertex_colors)
    bbox = (v.max(0) - v.min(0)).round(2) if len(v) else np.zeros(3)
    return len(v), len(np.asarray(m.triangles)), bbox, (c.mean() if len(c) else 0.0)


# Path 1: the adapter, exactly as the dashboard calls it
ff = FeedforwardResult.load_zarr(ZARR, load_images=True, load_world_points=False)
mesh_result = pointcloud_to_mesh(
    ff,
    out / "adapter",
    method="open3d_tsdf",
    voxel_size=0.005,
    sdf_trunc=0.02,
    depth_trunc=depth_trunc,
    clean_repair=False,
)
nv, nt, bbox, cm = stats(mesh_result.mesh_path)
print(f"adapter        trunc={depth_trunc:<5} verts={nv:>9,} tris={nt:>9,} bbox={bbox} meancolor={cm:.4f}")


# Path 2: _run_tsdf_mesh, handed a PointcloudResult double whose K is the
# 2.81x-rescaled COLMAP camera. Vertex counts must match path 1 — if the
# original-res K still leaks into the fusion, this collapses to ~28% of them.
class _ColmapDouble:
    extrinsics = np.asarray(ff.extrinsics)
    intrinsics = np.asarray(ff.intrinsics).copy()


_ColmapDouble.intrinsics[:, 0, 0] *= 2.812
_ColmapDouble.intrinsics[:, 1, 1] *= 2.791
_ColmapDouble.intrinsics[:, 0, 2] *= 2.812
_ColmapDouble.intrinsics[:, 1, 2] *= 2.791

mesh_path = _run_tsdf_mesh(
    result=_ColmapDouble(),
    feedforward_zarr=ZARR,
    output_dir=out / "reconstructor",
    voxel_size=0.005,
    sdf_trunc=0.02,
    depth_trunc=depth_trunc,
    clean_repair=False,
)
nv2, nt2, bbox2, cm2 = stats(mesh_path)
print(f"_run_tsdf_mesh trunc={depth_trunc:<5} verts={nv2:>9,} tris={nt2:>9,} bbox={bbox2} meancolor={cm2:.4f}")
print(f"MATCH: {nv == nv2 and nt == nt2}")
```

Run it at `depth_trunc=20` — the setting the spec's case A was measured at:

```bash
/opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/5304829b-6c29-427d-835f-80d56cbe6980/scratchpad/verify_mesh.py 20 /tmp/claude-0/-workspace-collab-splats/5304829b-6c29-427d-835f-80d56cbe6980/scratchpad/mesh_verify
```

Expected on both lines (spec case A): ≈**5,060,194** verts, ≈**6,904,081** tris, bbox ≈**9.29 × 3.15 × 6.31**, mean vertex colour ≈**0.53**, and `MATCH: True`.

Anything near case B (75,215 verts / 74,704 tris / mean colour 0.000) means the fix did not land. A `MATCH: False` with the second line much smaller means the COLMAP K is still reaching the fusion — stop and debug.

- [ ] **Step 2: Re-fuse at the shipping `depth_trunc: 2.0` and record it**

```bash
/opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/5304829b-6c29-427d-835f-80d56cbe6980/scratchpad/verify_mesh.py 2.0 /tmp/claude-0/-workspace-collab-splats/5304829b-6c29-427d-835f-80d56cbe6980/scratchpad/mesh_verify_2
```

Record the verts/tris/bbox/mean-colour. This is the new baseline for the shipping config (`configs/base.yaml`: `voxel_size: 0.005`, `sdf_trunc: 0.02`, `depth_trunc: 2.0`, `clean_repair: false`). It will be smaller than case A by design — `depth_trunc` also truncates real geometry, and depth here is non-metric (median 0.71, max 9.5 arbitrary units).

Append the measured table to the spec's `## Verification` section. Actual result:

| config | verts | tris | bbox | mean vertex colour |
|---|---|---|---|---|
| `depth_trunc=20` (matches case A) | 5,060,214 | 6,904,130 | 9.29×3.15×6.31 | 0.5271 |
| `depth_trunc=2.0` (shipping baseline) | 711,079 | 915,039 | 3.24×1.48×2.35 | 0.4509 |

- [ ] **Step 3: Update `CLAUDE.md`**

Add above the `stage-rerun-from-processed` entry in the "Recently completed" list:

```markdown
Recently completed (2026-08-11): **mesh-tsdf-adapter-convergence** — one TSDF adapter. `Reconstructor._run_tsdf_mesh` fused model-resolution depth from `feedforward.zarr` against original-resolution COLMAP intrinsics (`build_colmap` rescales the camera) and divided already-`[0,1]` RGB by 255 — measured 5.06M → 75k vertices and black vertex colours on `data/outputs/`. Both lines dated to 6bc9c81 and were invisible while `mesh.enabled: false`. `_feedforward_to_tsdf_inputs` (`mesh/utils.py`) now reads `result.depth`/`result.images` directly (world_points projection and PIL re-read/crop/`/255` deleted as measured-redundant) and is the single path for both `Reconstructor` and the dashboard. COLMAP stays the pose authority; guards on frame-count mismatch and on `[0, 255]` RGB. **Every `mesh.ply` on disk — local and under `environments-processed/` — was fused with the wrong K and needs `mesh(overwrite=True)`** ([spec](docs/superpowers/specs/2026-08-11-mesh-tsdf-adapter-convergence-design.md) · [plan](docs/superpowers/plans/2026-08-11-mesh-tsdf-adapter-convergence.md)).
```

- [ ] **Step 4: Commit**

`docs/superpowers/` is gitignored here — force-add it.

```bash
git add CLAUDE.md
git add -f docs/superpowers/specs/2026-08-11-mesh-tsdf-adapter-convergence-design.md docs/superpowers/plans/2026-08-11-mesh-tsdf-adapter-convergence.md
git commit -m "docs(mesh): record the post-fix mesh baseline and the re-mesh consequence

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Use an explicit pathspec on every commit in this plan — concurrent sessions share this working tree and a bare `git add -A` has merged unrelated work before.

---

## Out of scope

- `mesh.depth_trunc` stays at `2.0` by decision, despite costing vertices (spec case E). Depth from these backends is non-metric, so a fixed metres value is a proxy; revisiting it is separate work.
- `mesh.clean_repair` stays `false`. `clean_repair: true` has never appeared in `configs/` history — the meshlib rework is not implicated in this regression.
- COLMAP stays original-resolution. `_write_transforms_json` feeds nerfstudio and localization matches full-res query pixels against it; making it model-res would break both.
- Re-meshing the scenes already under `environments-processed/` — flagged in `CLAUDE.md`, run separately per the existing stage-rerun contract.

## Known follow-up

`ff.extrinsics = result.extrinsics` assumes the zarr's frame order matches `PointcloudResult.image_paths` order. Both derive from the same FrameStore order and only the count is checked. If ordering ever diverges the mesh degrades silently — a name-level check is worth a follow-up.
