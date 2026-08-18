# BA Pipeline Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pointcloud.bundle_adjustment: true` refines camera poses in the pipeline; `--stages refine` re-runs BA against a processed scene from disk. One implementation, two triggers.

**Architecture:** New leaf stage `refine` (`_STAGE_DEPS["refine"] = ["pointcloud"]`) implemented by one method `Reconstructor.refine_poses()` that loads `FeedforwardResult` from `feedforward.zarr`, runs the existing `BundleAdjustment.refine(...).reproject()` chain, and rewrites COLMAP/transforms/PLY/zarr through existing writers. Plus one contract fix: all creators now emit model-res K, so `_scale_intrinsics_to_model` gains a K-space guard (prevents double-scaling — this is the omega "BA not" trap, and it now affects every backend).

**Tech Stack:** pycolmap, zarr v3, existing `collab_splats.geometry.bundle_adjustment` (bae LM, CUDA-only optimize — tests mock it).

**Spec:** `docs/superpowers/specs/2026-08-18-ba-pipeline-wiring-design.md`

**Key file facts (verified 2026-08-18):**
- `collab_splats/geometry/bundle_adjustment.py:195` `refine()` — scales K to model space at :207 assuming original-res input, rescales back at :233-238. Stale: creators emit model-res K (vggtx.py:305 "Original-res decode removed: it fed wrong K to the BA wrapper", vggt_omega.py:233 same, mapanything.py:306 aliases model-res).
- `collab_splats/geometry/bundle_adjustment.py:410` `_scale_intrinsics_to_model(intrinsics, images, original_coords)`.
- `collab_splats/wrapper/reconstructor.py:48` `_STAGE_ORDER`, :49 `_STAGE_DEPS`, :62 `LEAF_STAGES` (derived), :553 `validate_config`, :637 `build_pointcloud` (NotImplementedError block at :642-647), :697 `_load_pointcloud_from_disk`, :766 `_export_pointcloud_ply`, :776 `_write_transforms_json`, :1063 `_stage_output_exists`, :1096 `run_pipeline` (config-driven stage list at :1115-1125, leaf refusal at :1145, dispatch at :1150-1163).
- `collab_splats/pointcloud/feedforward/base.py`: `FeedforwardResult.load_zarr` (:202, `load_images=True` needed), `result.reproject()` (:275, needs depth+pixel_indices), `build_pycolmap_reconstruction` (:666), `_rescale_reconstruction_to_original_dimensions` (:748), `unproject_depth_map_to_point_map` imported at :33 from `vggt.utils.geometry`.
- COLMAP image-name contract: creators register `frame_{idx:06d}` (no extension); `_load_pointcloud_from_disk` validates against frames.zarr.
- Tests: `tests/geometry/test_bundle_adjustment.py` (`_make_ff_result_for_ba` fixture at :545), `tests/wrapper/test_verify_stage.py` (stage-test template).
- Run tests with `/opt/venv/reconstruction/bin/python -m pytest`.

---

### Task 1: model-res K guard in `_scale_intrinsics_to_model`

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py:410-451` (guard) and `:206` (stale comment)
- Test: `tests/geometry/test_bundle_adjustment.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/geometry/test_bundle_adjustment.py`)

```python
# ---------------------------------------------------------------------------
# Tests for _scale_intrinsics_to_model K-space guard
# ---------------------------------------------------------------------------

def _guard_intrinsics(cx, cy, f=10.0, N=2):
    """(N, 3, 3) K with the given principal point."""
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.tile(K, (N, 1, 1))


def test_scale_intrinsics_model_res_k_is_identity():
    """K already at model res (cx≈W_model/2) must pass through unscaled — all creators
    decode pose at model resolution now; scaling would double-apply the crop transform."""
    from collab_splats.geometry.bundle_adjustment import _scale_intrinsics_to_model

    images = torch.zeros(2, 3, 8, 8)  # model res 8x8
    # original image 64x64, no crop: [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]
    original_coords = np.tile(np.array([0.0, 0.0, 64.0, 64.0, 64.0, 64.0], dtype=np.float32), (2, 1))
    intr = _guard_intrinsics(cx=4.0, cy=4.0)  # model-res principal point

    out, sx, sy, tl_x, tl_y = _scale_intrinsics_to_model(intr, images, original_coords)

    np.testing.assert_array_equal(out, intr)
    assert (sx, sy, tl_x, tl_y) == (1.0, 1.0, 0.0, 0.0)


def test_scale_intrinsics_original_res_k_still_scaled():
    """Legacy original-res K (cx≈orig_w/2) keeps the crop-aware scaling (regression)."""
    from collab_splats.geometry.bundle_adjustment import _scale_intrinsics_to_model

    images = torch.zeros(2, 3, 8, 8)
    original_coords = np.tile(np.array([0.0, 0.0, 64.0, 64.0, 64.0, 64.0], dtype=np.float32), (2, 1))
    intr = _guard_intrinsics(cx=32.0, cy=32.0)  # original-res principal point

    out, sx, sy, tl_x, tl_y = _scale_intrinsics_to_model(intr, images, original_coords)

    assert sx == pytest.approx(8.0 / 64.0)
    assert sy == pytest.approx(8.0 / 64.0)
    assert out[0, 0, 2] == pytest.approx(32.0 * sx)
```

- [ ] **Step 2: Run tests, verify both fail** (first: identity assertion fails; second may pass — confirm)

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k scale_intrinsics -p no:randomly -v`
Expected: `test_scale_intrinsics_model_res_k_is_identity` FAILS (arrays not equal), `..._still_scaled` PASSES (it pins current behavior).

- [ ] **Step 3: Implement the guard** in `_scale_intrinsics_to_model` — replace the `if original_coords is not None:` branch body (bundle_adjustment.py:431-439) with:

```python
    if original_coords is not None:
        # Crop-aware: original_coords = [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]
        # The cropped region was resized to (W_model, H_model).
        tl_x = float(original_coords[0, 0])
        tl_y = float(original_coords[0, 1])
        cr_x = float(original_coords[0, 2])
        cr_y = float(original_coords[0, 3])
        # All current creators decode K at model resolution (original-res decode removed —
        # see vggtx._forward). Detect which space K lives in by comparing the principal
        # point against the two candidate optical centres: model-res K has 2·cx ≈ W_model,
        # original-res K has 2·cx ≈ tl_x + cr_x (crop centre in original pixels). Scaling
        # a model-res K would double-apply the crop transform and corrupt reprojection.
        cx2 = 2.0 * float(intrinsics[0, 0, 2])
        if abs(cx2 - W_model) <= abs(cx2 - (tl_x + cr_x)):
            return intrinsics, 1.0, 1.0, 0.0, 0.0
        crop_w = cr_x - tl_x
        crop_h = cr_y - tl_y
        sx = W_model / crop_w
        sy = H_model / crop_h
```

- [ ] **Step 4: Fix the stale comment** at bundle_adjustment.py:206 — change

```python
        # Scale intrinsics once — VGGSfM tracks in model-res, intrinsics stored at original-res
```
to
```python
        # Bring intrinsics to model space if needed — VGGSfM tracks are model-res; creators
        # already store model-res K (the guard makes this a no-op), legacy original-res K is scaled
```

- [ ] **Step 5: Run the full BA test file**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -p no:randomly -q`
Expected: all pass (existing refine tests use `original_coords=None` — untouched fallback branch).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/bundle_adjustment.py tests/geometry/test_bundle_adjustment.py
git commit -m "fix(ba): _scale_intrinsics_to_model detects model-res K, skips double-scaling"
```

---

### Task 2: config validation — LC × BA mutual exclusion, drop the NotImplementedError

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:553-582` (validate_config), `:642-647` (delete block)
- Test: `tests/wrapper/test_refine_stage.py` (new file)

- [ ] **Step 1: Write the failing tests** — create `tests/wrapper/test_refine_stage.py`:

```python
"""Tests for the refine stage: config validation, stage registration, refine_poses."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.wrapper.reconstructor import (
    _STAGE_DEPS,
    _STAGE_ORDER,
    LEAF_STAGES,
    Reconstructor,
)


def _cfg(tmp_path, **pointcloud):
    """Minimal valid config dict; pointcloud kwargs merged over base.yaml defaults."""
    return {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": pointcloud,
    }


def test_validate_config_rejects_ba_with_lc_bool(tmp_path):
    """bundle_adjustment + loop_closure=true must fail loud at construction."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))


def test_validate_config_rejects_ba_with_lc_dict(tmp_path):
    """Dict-form loop_closure ({'enabled': ...} implicit true) is rejected too."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure={"submap_size": 16}))


def test_validate_config_allows_ba_without_lc(tmp_path):
    """BA alone constructs fine — the old NotImplementedError is gone."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=False))
    assert r.config["pointcloud"]["bundle_adjustment"] is True
```

- [ ] **Step 2: Run tests, verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_refine_stage.py -p no:randomly -v`
Expected: first two FAIL (no ValueError raised), third PASSES (construction never raised — the NotImplementedError fires in build_pointcloud, not __init__).

- [ ] **Step 3: Add the exclusion to `validate_config`** — insert before `return config` (reconstructor.py:582):

```python
        # BA over LC submaps is unsupported: submaps don't carry the per-frame model tensors
        # track extraction needs. Fail loud at construction instead of silently skipping one.
        lc = pc.get("loop_closure")
        lc_enabled = lc.get("enabled") is not False if isinstance(lc, dict) else bool(lc)
        if pc.get("bundle_adjustment") and lc_enabled:
            raise ValueError(
                "pointcloud.bundle_adjustment and pointcloud.loop_closure are mutually "
                "exclusive — BA needs per-frame model tensors that LC submaps do not carry."
            )
```

- [ ] **Step 4: Delete the NotImplementedError block** in `build_pointcloud` (reconstructor.py:642-647):

```python
        # BA at the Reconstructor level is not wired — fail loud instead of silently no-op'ing
        if pc_cfg["bundle_adjustment"]:
            raise NotImplementedError(
                "pointcloud.bundle_adjustment is not wired at the Reconstructor level. "
                "Pass bundle_adjustment to the creator config directly for now."
            )
```
(remove entirely — refine runs as its own stage, Task 4).

- [ ] **Step 5: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_refine_stage.py tests/wrapper/test_reconstructor.py -p no:randomly -q`
Expected: new tests pass; existing reconstructor tests pass (grep test_reconstructor.py for `bundle_adjustment`/`NotImplementedError` first — if a test pins the old raise, update it to expect the new ValueError path or delete it as obsolete).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_refine_stage.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): validate BA x LC mutual exclusion; drop build_pointcloud NotImplementedError"
```

---

### Task 3: `Reconstructor.refine_poses()`

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (new method after `build_pointcloud` helpers, near `verify`)
- Test: `tests/wrapper/test_refine_stage.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/wrapper/test_refine_stage.py`:

```python
# ---------------------------------------------------------------------------
# Tests for refine_poses
# ---------------------------------------------------------------------------

def _write_ff_zarr(backend_dir, N=2, H=8, W=8, P=10):
    """Synthetic FeedforwardResult persisted to backend_dir/feedforward.zarr."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    K = np.tile(
        np.array([[10.0, 0, W / 2], [0, 10.0, H / 2], [0, 0, 1.0]], dtype=np.float32), (N, 1, 1)
    )
    ff = FeedforwardResult(
        points=np.random.rand(P, 3).astype(np.float32),
        colors=np.zeros((P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)),
        intrinsics=K,
        image_paths=[Path(f"frame_{i:06d}") for i in range(N)],
        original_coords=np.tile(
            np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (N, 1)
        ),
        model_width=W,
        model_height=H,
        images=torch.zeros(N, 3, H, W),
        confidence=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
        depth=np.ones((N, H, W), dtype=np.float32),
        pixel_indices=np.stack(
            [np.zeros(P, dtype=np.int64), np.arange(P) % H, np.arange(P) % W], axis=1
        ),
    )
    backend_dir.mkdir(parents=True, exist_ok=True)
    ff.save_zarr(backend_dir / "feedforward.zarr")
    return ff


def _reconstructor(tmp_path):
    return Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=False))


def test_refine_poses_missing_zarr_raises(tmp_path):
    """No feedforward.zarr → clear FileNotFoundError, not a deep BA stack trace."""
    r = _reconstructor(tmp_path)
    with pytest.raises(FileNotFoundError, match="feedforward.zarr"):
        r.refine_poses()


def test_refine_poses_skips_when_marker_exists(tmp_path):
    """Existing refine.json without overwrite → skip (no BA), matching other stage methods."""
    r = _reconstructor(tmp_path)
    marker = r.backend_dir / "colmap" / "refine.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    with patch.object(Reconstructor, "_resolve_result", return_value=MagicMock()) as mock_resolve, \
         patch("collab_splats.geometry.bundle_adjustment.BundleAdjustment.refine") as mock_refine:
        r.refine_poses(overwrite=False)
    mock_refine.assert_not_called()
    mock_resolve.assert_called_once()


def test_refine_poses_refines_and_persists(tmp_path):
    """refine_poses: BA refine + reproject, COLMAP rewritten, zarr updated, marker written."""
    import zarr as zarr_mod
    from dataclasses import replace

    r = _reconstructor(tmp_path)
    ff = _write_ff_zarr(r.backend_dir)

    # BA output: translate every camera by +1 in x so refinement is observable
    def fake_refine(self, result):
        new_ext = result.extrinsics.copy()
        new_ext[:, 0, 3] += 1.0
        return replace(result, extrinsics=new_ext)

    fake_result = MagicMock()
    with patch("collab_splats.geometry.bundle_adjustment.BundleAdjustment.refine", fake_refine), \
         patch.object(Reconstructor, "_load_pointcloud_from_disk", return_value=fake_result), \
         patch.object(Reconstructor, "_export_pointcloud_ply") as mock_ply, \
         patch.object(Reconstructor, "_write_transforms_json") as mock_tj:
        out = r.refine_poses()

    # COLMAP rewritten with refined poses
    assert (r.backend_dir / "colmap" / "sparse" / "0" / "images.bin").exists()
    # zarr extrinsics updated in place — never diverges from COLMAP
    store = zarr_mod.open(str(r.backend_dir / "feedforward.zarr"), mode="r")
    np.testing.assert_allclose(store["extrinsics"][:, 0, 3], ff.extrinsics[:, 0, 3] + 1.0)
    # standard writers refreshed the derived artifacts
    mock_ply.assert_called_once_with(fake_result)
    mock_tj.assert_called_once_with(fake_result)
    # marker doubles as provenance
    marker = json.loads((r.backend_dir / "colmap" / "refine.json").read_text())
    assert "config" in marker and "loss_history" in marker
    assert out is fake_result
```

- [ ] **Step 2: Run tests, verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_refine_stage.py -p no:randomly -v`
Expected: three new tests FAIL with `AttributeError: ... no attribute 'refine_poses'`.

- [ ] **Step 3: Verify `BundleAdjustmentConfig` field name for loss capture** before writing the method:

Run: `grep -n 'capture_loss\|loss_history' collab_splats/geometry/bundle_adjustment.py | head`
Use the actual field name in Step 4 (expected `capture_loss_history`; if it differs, adjust).

- [ ] **Step 4: Implement `refine_poses`** — add to `Reconstructor` after `build_pointcloud`'s helpers (suggested: right before `extract_semantics`, reconstructor.py:~885). Heavy deps imported lazily, matching `_load_pointcloud_from_disk`'s pattern:

```python
    def refine_poses(self, overwrite: bool = False) -> "PointcloudResult":
        """Refine camera poses via LM bundle adjustment; rewrite pose-derived artifacts.

        One implementation for both triggers: runs inline after the pointcloud stage when
        pointcloud.bundle_adjustment is enabled, and from disk via --stages refine against
        a processed scene. Loads everything from feedforward.zarr — no live creator needed.
        """
        from dataclasses import asdict

        import zarr as zarr_mod
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        from collab_splats.geometry.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
        from collab_splats.pointcloud.feedforward.base import (
            FeedforwardResult,
            _rescale_reconstruction_to_original_dimensions,
            build_pycolmap_reconstruction,
        )

        # Skip when already refined — run_pipeline refuses NAMED re-runs generically, so this
        # mirrors the other stage methods' silent skip for config-driven repeat runs.
        marker = self.backend_dir / "colmap" / "refine.json"
        if marker.exists() and not overwrite:
            logger.info("Poses already refined (%s), skipping refine", marker)
            return self._resolve_result()

        # Load the full FeedforwardResult from zarr: images/confidence/world_points feed track
        # extraction, depth+pixel_indices feed the deterministic creator-free reproject.
        zarr_path = self.backend_dir / "feedforward.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"refine requires {zarr_path}; run the pointcloud stage first.")
        ff = FeedforwardResult.load_zarr(zarr_path, load_images=True)

        # Refine poses with LM BA, then re-derive the point set under the new cameras
        cfg = BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir, capture_loss_history=True)
        ba = BundleAdjustment(cfg)
        ff = ba.refine(ff).reproject()

        # Rewrite COLMAP through the creators' exact write path: build at model res from the
        # refined result, then rescale K + image dims back to original resolution.
        recon = build_pycolmap_reconstruction(
            ff.points,
            ff.colors,
            ff.extrinsics,
            ff.intrinsics,
            ff.model_width,
            ff.model_height,
            [p.name for p in ff.image_paths],
        )
        recon = _rescale_reconstruction_to_original_dimensions(
            recon, ff.image_paths, ff.original_coords, (ff.model_width, ff.model_height)
        )
        sparse_dir = self.backend_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        recon.write_binary(str(sparse_dir))

        # Write pose-derived arrays back to feedforward.zarr so zarr and COLMAP never disagree
        # (localization samples world_points; a later --stages refine re-reads these poses).
        store = zarr_mod.open(str(zarr_path), mode="r+")
        store["extrinsics"][:] = ff.extrinsics
        store["intrinsics"][:] = ff.intrinsics
        store["points"][:] = ff.points
        if "world_points" in store and ff.depth is not None:
            wp = unproject_depth_map_to_point_map(
                ff.depth[..., None], ff.extrinsics[:, :3, :], ff.intrinsics
            ).astype(np.float32)
            store["world_points"][:] = wp

        # Refresh the remaining derived artifacts through the standard writers
        result = self._load_pointcloud_from_disk()
        self._export_pointcloud_ply(result)
        self._write_transforms_json(result)

        # Marker + provenance in one file: BA config and per-step LM loss history
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                    "loss_history": ba._last_loss_history,
                    "n_frames": len(ff.image_paths),
                },
                indent=2,
            )
        )

        self.pointcloud = result
        return result
```

- [ ] **Step 5: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_refine_stage.py -p no:randomly -v`
Expected: all pass. If `save_zarr`/`load_zarr` shape mismatches surface in the fixture, fix the fixture, not the method.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_refine_stage.py
git commit -m "feat(wrapper): refine_poses — BA refine + reproject + COLMAP/zarr/transforms rewrite"
```

---

### Task 4: stage registration + both triggers

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:48-58` (order/deps), `:1063` (_stage_output_exists), `:1096-1163` (run_pipeline)
- Test: `tests/wrapper/test_refine_stage.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/wrapper/test_refine_stage.py`:

```python
# ---------------------------------------------------------------------------
# Tests for stage registration + triggers
# ---------------------------------------------------------------------------

def test_refine_is_a_leaf_stage():
    """refine must be re-runnable on its own from environments-processed."""
    assert "refine" in _STAGE_ORDER
    assert _STAGE_DEPS["refine"] == ["pointcloud"]
    assert "refine" in LEAF_STAGES


def test_run_pipeline_config_driven_appends_refine(tmp_path):
    """bundle_adjustment: true → refine runs right after pointcloud, before dependents."""
    r = _reconstructor(tmp_path)
    calls = []
    with patch.object(Reconstructor, "preprocess", side_effect=lambda **k: calls.append("preproc")), \
         patch.object(Reconstructor, "build_pointcloud", side_effect=lambda **k: calls.append("pointcloud")), \
         patch.object(Reconstructor, "refine_poses", side_effect=lambda **k: calls.append("refine")), \
         patch.object(Reconstructor, "extract_semantics", side_effect=lambda **k: calls.append("semantics")), \
         patch.object(Reconstructor, "mesh", side_effect=lambda **k: calls.append("mesh")), \
         patch.object(Reconstructor, "build_localization_db", side_effect=lambda **k: calls.append("localize")):
        r.run_pipeline()
    assert "refine" in calls
    assert calls.index("refine") == calls.index("pointcloud") + 1


def test_run_pipeline_named_refine_refuses_existing_output(tmp_path):
    """Named refine with existing refine.json and no overwrite → refusal (generic leaf rule)."""
    r = _reconstructor(tmp_path)
    # Satisfy the pointcloud dependency and the refine marker on disk
    (r.backend_dir / "colmap" / "sparse" / "0").mkdir(parents=True)
    (r.backend_dir / "colmap" / "sparse" / "0" / "cameras.bin").touch()
    (r.backend_dir / "feedforward.zarr").mkdir()
    (r.backend_dir / "colmap" / "refine.json").write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        r.run_pipeline(stages=["refine"])
```

- [ ] **Step 2: Run tests, verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_refine_stage.py -p no:randomly -v`
Expected: the three new tests FAIL (`"refine" not in _STAGE_ORDER`, KeyError, no refusal).

- [ ] **Step 3: Register the stage** (reconstructor.py:48-58):

```python
_STAGE_ORDER = ["preproc", "pointcloud", "refine", "semantics", "mesh", "localize", "verify"]
_STAGE_DEPS: dict[str, list[str]] = {
    "preproc": [],
    "pointcloud": ["preproc"],
    # refine rewrites pointcloud outputs in place; deliberately NOT a dependency of the
    # stages below — that would demote them from LEAF_STAGES and break their disk re-run.
    # Staleness contract: after --stages refine, re-run dependents with overwrite
    # (configs/README.md). Inline runs are ordered refine-before-dependents, so never stale.
    "refine": ["pointcloud"],
    "semantics": ["pointcloud"],
    "mesh": ["pointcloud"],
    "localize": ["pointcloud"],
    # verify reuses the localize feature cache but builds it itself when absent, so its
    # only hard dependency is the reconstruction
    "verify": ["pointcloud"],
}
```

- [ ] **Step 4: Add the exists-marker** in `_stage_output_exists` (after the `pointcloud` branch):

```python
        if stage == "refine":
            return (self.backend_dir / "colmap" / "refine.json").exists()
```

- [ ] **Step 5: Wire both triggers in `run_pipeline`** — config-driven append (after `stages = ["preproc", "pointcloud"]`, reconstructor.py:1117):

```python
            if self.config["pointcloud"]["bundle_adjustment"]:
                stages.append("refine")
```

and dispatch (after the `pointcloud` branch at :1155):

```python
            elif stage == "refine":
                result = self.refine_poses(overwrite=overwrite)
```

Also update the `run_pipeline` docstring stage list to include `"refine"`.

- [ ] **Step 6: Run the wrapper suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -p no:randomly -q`
Expected: all pass (check test_reconstructor.py stage-list assertions — any test pinning `_STAGE_ORDER`/`LEAF_STAGES` contents needs the new member added).

- [ ] **Step 7: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_refine_stage.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): refine leaf stage — inline via bundle_adjustment flag + --stages refine"
```

---

### Task 5: config + docs

**Files:**
- Modify: `configs/base.yaml:60-61`, `configs/README.md`

- [ ] **Step 1: base.yaml** — replace

```yaml
  # ── NOT YET IMPLEMENTED (raises NotImplementedError if enabled) ──
  bundle_adjustment: false    # BA at Reconstructor level not wired; pass to creator config directly
```
with
```yaml
  # LM bundle adjustment pose refinement (leaf stage `refine`; runs inline after the
  # pointcloud stage when enabled here, or standalone via --stages refine). Rewrites
  # COLMAP/transforms.json/sparse_pc.ply/feedforward.zarr poses in place. Mutually
  # exclusive with loop_closure. Knobs stay on BundleAdjustmentConfig defaults
  # (global BA; tracks cached in <backend>/).
  bundle_adjustment: false
```

- [ ] **Step 2: configs/README.md** — in the "Re-running one stage against a processed scene" section, add `refine` to the leaf-stage list and append the staleness contract:

```markdown
`refine` (LM bundle adjustment) rewrites the reconstruction's poses in place —
COLMAP, `transforms.json`, `sparse_pc.ply`, and the pose-derived arrays in
`feedforward.zarr`. It does NOT invalidate `mesh/`, lifted semantics, or the
localization DB built under the old poses: after `--stages refine`, re-run those
stages with `overwrite` if pose-sensitive outputs matter. Provenance for the last
refine run (BA config + LM loss history) is in `<backend>/colmap/refine.json`.
```

- [ ] **Step 3: Commit**

```bash
git add configs/base.yaml configs/README.md
git commit -m "docs(configs): bundle_adjustment flag wired — refine stage contract + staleness note"
```

---

### Task 6: full-suite gate

- [ ] **Step 1: Run the full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly -q`
Expected: pass counts consistent with `docs/known-test-failures.md`; zero NEW failures.

- [ ] **Step 2: Update the graph**

Run: `graphify update .`

- [ ] **Step 3: Commit any stragglers; update CLAUDE.md in-flight section** (move BA wiring into "Recently completed" once ATE validation lands — not before).

---

### Task 7: ATE validation (compute — human-gated, tmux only)

Owed after implementation; NOT run automatically:

- 7-Scenes chess/seq-01, 4 backends (vggtx, mapanything, vggt_omega, loger), conditions `baseline` vs `ba` via `evals/scripts/eval.py --conditions baseline ba` (tmux, `--submap_size 50` if >100 frames).
- Reference-free control alongside (LoGeR lesson: ATE-vs-GT has a noise floor).
- Expect: measurable improvement on chess (prior LC-era numbers had windowed+BA beating baseline); no-op on tiny-baseline scenes (C0043) — document, don't fight.
- Numbers append to the spec (`docs/superpowers/specs/2026-08-18-ba-pipeline-wiring-design.md`).
