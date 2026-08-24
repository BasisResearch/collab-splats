# Dense Scale-Aligned VDA Depth Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `pointcloud.zarr` VDA depth to the COLMAP world scale per frame at `_run_sfm` time, so sfm scenes get dense COLMAP-scale splat depth targets, a working `mesh.source: feedforward`, and scale-consistent localization/dashboard data.

**Architecture:** One new alignment function pair in `collab_splats/pointcloud/sfm.py` (`align_depth_to_reconstruction` fits per-frame median-of-ratios scales from track observations; `apply_depth_alignment` mutates a `FeedforwardResult` in place and returns provenance attrs), one 3-line wiring block in `Reconstructor._run_sfm`, and attr-guarded consumer changes in `splats()`/`mesh()`. Everything runs and is validated in a worktree branch before any merge (user directive — spec §Execution).

**Tech Stack:** numpy, pycolmap, `vggt.utils.geometry.unproject_depth_map_to_point_map`, zarr attrs, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-dense-vda-depth-targets-design.md`

**Baseline branch:** `insfm-exec` @ `9b1c1b07` (worktree `/workspace/collab-splats/.claude/worktrees/insfm-exec`). Trunk lacks `45077365`/`9b1c1b07` — do NOT branch from trunk.

**Environment rules (apply to every task):**
- Python: `/opt/venv/reconstruction/bin/python` (py3.11). All pytest/eval commands use it.
- Commit with `git commit --only <files>` (concurrent sessions share the index).
- `docs/superpowers/**` needs `git add -f`.
- Heavy eval (Task 8) runs in tmux; 46.6 GB cgroup; no parallel GPU work.
- Style: imports at top (heavy optional deps may be lazy with clear ImportError), block comments, `########` dividers, one-line docstrings with `"""` on own lines, flat test functions.
- `tests/wrapper/test_reconstructor.py::test_no_inline_defaults_in_source` forbids `.get(key, default)` in reconstructor.py — use `in` membership checks on zarr attrs.

---

### Task 1: Branch setup in the worktree

**Files:** none (git only)

- [ ] **Step 1: Create the work branch off `insfm-exec` @ `9b1c1b07`**

```bash
cd /workspace/collab-splats/.claude/worktrees/insfm-exec
git status --short          # must be clean; if not, stop and report
git switch -c dense-vda-align 9b1c1b07
```

Expected: `Switched to a new branch 'dense-vda-align'`.

- [ ] **Step 2: Verify the baseline has the sparse-targets fix**

```bash
git log --oneline -2
grep -n "points3d_depth_maps" collab_splats/pointcloud/utils.py | head -2
```

Expected: HEAD is `9b1c1b07`; grep finds the function.

- [ ] **Step 3: Sanity-run the existing sparse-target tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_points3d_depth_maps.py -q
```

Expected: all pass.

---

### Task 2: `align_depth_to_reconstruction` (fit core)

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py` (new public function + `MIN_ALIGN_OBS` constant, in a new `######## Depth alignment` section)
- Test: `tests/pointcloud/test_depth_alignment.py` (create)

All work in `/workspace/collab-splats/.claude/worktrees/insfm-exec`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pointcloud/test_depth_alignment.py`. The fixture mirrors `tests/pointcloud/test_points3d_depth_maps.py::_recon` but stores **real projected pixel coords** in each `Point2D` — the alignment reads `p.xy`, unlike `points3d_depth_maps` which reprojects. Trap: `add_point3D` range-checks track elements, so `im.points2D` must be populated before `add_point3D` back-fills the ids.

```python
"""
Tests for align_depth_to_reconstruction / apply_depth_alignment: per-frame VDA-to-COLMAP
depth scale fitting from track observations.
"""

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.sfm import (
    MIN_ALIGN_OBS,
    align_depth_to_reconstruction,
    apply_depth_alignment,
)

W, H = 64, 48  # native camera resolution
K_PARAMS = [100.0, 100.0, 32.0, 24.0]  # fx, fy, cx, cy

# 25-point grid at z=5: projects to u in [20, 44], v in [16, 32] — all in bounds,
# and comfortably above the MIN_ALIGN_OBS=20 per-frame floor
GRID = [(x, y, 5.0) for x in np.linspace(-0.6, 0.6, 5) for y in np.linspace(-0.4, 0.4, 5)]


def _recon(points_xyz, translations):
    """
    One PINHOLE camera, one image per translation (identity rotation), every image
    observing every point3D at its true projected pixel.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=K_PARAMS, camera_id=1)
    recon.add_camera_with_trivial_rig(cam)

    # Point2D carries the REAL projected pixel — the alignment samples depth at p.xy;
    # add_point3D's track back-fills each observation's point3D_id
    fx, fy, cx, cy = K_PARAMS
    for i, t in enumerate(translations):
        im = pycolmap.Image(name=f"frame_{i:06d}", camera_id=1, image_id=i + 1)
        pts2d = []
        for xyz in points_xyz:
            p = np.asarray(xyz, np.float64) + np.asarray(t, np.float64)
            pts2d.append(pycolmap.Point2D(np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])))
        im.points2D = pts2d
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.asarray(t, dtype=np.float64))
        recon.add_image_with_trivial_frame(im, pose)

    for j, xyz in enumerate(points_xyz):
        track = pycolmap.Track()
        for i in range(len(translations)):
            track.add_element(i + 1, j)
        recon.add_point3D(np.asarray(xyz, dtype=np.float64), track, np.zeros(3, dtype=np.uint8))
    return recon


def test_recovers_known_global_scale():
    # COLMAP depth 5.0 everywhere, VDA depth 5/3.2 -> every per-frame scale is 3.2
    recon = _recon(GRID, [(0.0, 0.0, 0.0)])
    depth = np.full((1, H, W), 5.0 / 3.2, np.float32)
    scales, stats = align_depth_to_reconstruction(recon, ["frame_000000"], depth)

    assert scales.shape == (1,)
    assert scales[0] == pytest.approx(3.2, rel=1e-5)
    assert stats["global_scale"] == pytest.approx(3.2, rel=1e-5)
    assert stats["fallback_frames"] == []
    # Post-alignment ratio spread collapses to 1.0
    assert stats["ratio_p10_p50_p90_after"] == pytest.approx([1.0, 1.0, 1.0], rel=1e-5)


def test_recovers_per_frame_jitter():
    # Frame 0: COLMAP 5.0, VDA 2.5 -> 2.0.  Frame 1: COLMAP 6.0 (t_z=1), VDA 2.0 -> 3.0.
    recon = _recon(GRID, [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
    depth = np.stack([np.full((H, W), 2.5, np.float32), np.full((H, W), 2.0, np.float32)])
    scales, stats = align_depth_to_reconstruction(recon, ["frame_000000", "frame_000001"], depth)

    assert scales == pytest.approx([2.0, 3.0], rel=1e-5)
    # A global-median fit would leave the per-frame spread; per-frame collapses it
    assert stats["ratio_p10_p50_p90_after"] == pytest.approx([1.0, 1.0, 1.0], rel=1e-5)


def test_sparse_frame_falls_back_to_global_median():
    # Frame 1's depth map is zero except one observation pixel -> 1 valid pair < MIN_ALIGN_OBS
    recon = _recon(GRID, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    depth = np.stack([np.full((H, W), 2.5, np.float32), np.zeros((H, W), np.float32)])
    depth[1, 16, 20] = 999.0  # grid point (-0.6, -0.4, 5) projects to (u=20, v=16)
    scales, stats = align_depth_to_reconstruction(recon, ["frame_000000", "frame_000001"], depth)

    assert scales[0] == pytest.approx(2.0, rel=1e-5)
    assert scales[1] == pytest.approx(2.0, rel=1e-5)  # inherited, not fit from the 999 outlier
    assert stats["fallback_frames"] == ["frame_000001"]
    assert stats["n_fallback"] == 1


def test_all_frames_sparse_raises():
    recon = _recon(GRID, [(0.0, 0.0, 0.0)])
    depth = np.zeros((1, H, W), np.float32)  # no valid VDA depth anywhere
    with pytest.raises(ValueError, match=str(MIN_ALIGN_OBS)):
        align_depth_to_reconstruction(recon, ["frame_000000"], depth)


def test_unknown_image_name_raises():
    recon = _recon(GRID, [(0.0, 0.0, 0.0)])
    depth = np.full((1, H, W), 1.0, np.float32)
    with pytest.raises(ValueError, match="not in reconstruction"):
        align_depth_to_reconstruction(recon, ["frame_999999"], depth)


def test_native_to_model_res_pixel_rescale():
    # Depth grid at HALF the native camera resolution: native pixel (u, v) must be
    # sampled at (u/2, v/2) — the localization ref_px bug class (92f2e4a)
    recon = _recon(GRID, [(0.0, 0.0, 0.0)])
    depth = np.full((1, H // 2, W // 2), 2.5, np.float32)
    scales, _ = align_depth_to_reconstruction(recon, ["frame_000000"], depth)
    assert scales[0] == pytest.approx(2.0, rel=1e-5)
```

(`apply_depth_alignment` tests come in Task 3 — same file; the import line above already names it, so this file stays red until Task 3 lands. That is fine: run per-test-name until then.)

- [ ] **Step 2: Run the alignment tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_alignment.py -q 2>&1 | tail -3
```

Expected: `ImportError: cannot import name 'MIN_ALIGN_OBS' from 'collab_splats.pointcloud.sfm'`.

- [ ] **Step 3: Implement `align_depth_to_reconstruction` in `collab_splats/pointcloud/sfm.py`**

Add a `######## Depth alignment` section (near the other public helpers). numpy and pycolmap are already imported at the top of sfm.py (verify; add if missing).

```python
########################################################################################
# Depth alignment: fit per-frame scales taking VDA metric depth to the COLMAP world
########################################################################################

MIN_ALIGN_OBS = 20  # per-frame track-observation floor for a trustworthy median


def align_depth_to_reconstruction(
    reconstruction: "pycolmap.Reconstruction",
    image_names: list[str],
    depth: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Per-frame scale factors aligning VDA depth to the reconstruction's world scale.

    - Correspondences are track observations: each points2D with a point3D gives an exact
      pixel plus the point's z in the camera frame (d_colmap); the pixel is rescaled from
      native camera resolution to the depth grid and nearest-sampled into VDA depth (d_vda).
    - s_i = median(d_colmap / d_vda) per frame; frames with fewer than MIN_ALIGN_OBS valid
      pairs inherit the global median of the fitted scales; zero fitted frames raises.
    - Returns (scales, stats): (N,) float64 depth multipliers, and a stats dict with the
      global scale, fallback frames, per-frame obs counts, and the pooled ratio spread
      before/after alignment (the after-spread is the unit-level success check).
    """
    # Row order is the caller's; every name must be registered (mirrors points3d_depth_maps)
    name_to_image = {image.name: image for image in reconstruction.images.values()}
    missing = [name for name in image_names if name not in name_to_image]
    if missing:
        raise ValueError(f"{len(missing)} image names not in reconstruction (first: {missing[0]})")

    n_frames, grid_h, grid_w = depth.shape
    scales = np.full(n_frames, np.nan)
    obs_counts = np.zeros(n_frames, dtype=np.int64)
    pooled_ratios: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []

    for row, name in enumerate(image_names):
        image = name_to_image[name]
        camera = reconstruction.cameras[image.camera_id]

        # Track observations: exact 2D pixel + the observed point's depth in this view
        observations = [p for p in image.points2D if p.has_point3D()]
        if not observations:
            continue
        xyz = np.stack([reconstruction.points3D[p.point3D_id].xyz for p in observations])
        cam_from_world = image.cam_from_world().matrix()
        d_colmap = (xyz @ cam_from_world[:3, :3].T + cam_from_world[:3, 3])[:, 2]

        # Rescale native pixels to the depth grid (the localization ref_px bug class —
        # native-res keypoints indexed into a model-res grid), then nearest-sample
        xy = np.stack([p.xy for p in observations])
        u = np.rint(xy[:, 0] * (grid_w / camera.width)).astype(np.int64)
        v = np.rint(xy[:, 1] * (grid_h / camera.height)).astype(np.int64)
        in_bounds = (u >= 0) & (u < grid_w) & (v >= 0) & (v < grid_h)
        d_vda = np.zeros(len(observations))
        d_vda[in_bounds] = depth[row, v[in_bounds], u[in_bounds]]

        # Keep pairs with positive depth on both sides; fit only above the obs floor
        valid = in_bounds & (d_vda > 0) & (d_colmap > 0)
        obs_counts[row] = int(valid.sum())
        if obs_counts[row] == 0:
            continue
        ratios = d_colmap[valid] / d_vda[valid]
        pooled_ratios.append(ratios)
        pooled_rows.append(np.full(len(ratios), row))
        if obs_counts[row] >= MIN_ALIGN_OBS:
            scales[row] = np.median(ratios)

    fitted = ~np.isnan(scales)
    if not fitted.any():
        raise ValueError(
            f"depth alignment: no frame has >= {MIN_ALIGN_OBS} valid track observations — "
            "the reconstruction is too sparse to align VDA depth to the COLMAP world."
        )

    # Thin frames inherit the scene answer (below the obs floor: don't fit, inherit)
    global_scale = float(np.median(scales[fitted]))
    fallback_frames = [image_names[i] for i in np.flatnonzero(~fitted)]
    for name in fallback_frames:
        logger.warning("depth alignment: %s under %d obs — using global scale", name, MIN_ALIGN_OBS)
    scales[~fitted] = global_scale

    # Pooled spread: before = one global scale for all frames, after = per-frame scales
    ratios_all = np.concatenate(pooled_ratios)
    rows_all = np.concatenate(pooled_rows).astype(np.int64)
    stats = {
        "global_scale": global_scale,
        "n_fallback": len(fallback_frames),
        "fallback_frames": fallback_frames,
        "obs_counts": obs_counts.tolist(),
        "ratio_p10_p50_p90_before": [float(x) for x in np.percentile(ratios_all / global_scale, [10, 50, 90])],
        "ratio_p10_p50_p90_after": [float(x) for x in np.percentile(ratios_all / scales[rows_all], [10, 50, 90])],
    }
    return scales, stats
```

- [ ] **Step 4: Run the six alignment tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_alignment.py -q -k "not apply" 2>&1 | tail -3
```

Expected: collection error persists only if `apply_depth_alignment` import fails — temporarily comment that import? **No**: instead implement Task 3's stub signature is NOT allowed. Run with the import trimmed to what exists is churn — simplest: implement Task 3 immediately after and run the whole file once. For a strict TDD checkpoint here, run:

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
import subprocess, sys
sys.exit(subprocess.call([
    "/opt/venv/reconstruction/bin/python", "-m", "pytest",
    "tests/pointcloud/test_depth_alignment.py", "-q", "--co",
]))
EOF
```

If collection fails only on the `apply_depth_alignment` import, proceed to Task 3 Step 2 and run the full file there. (Do not commit yet.)

---

### Task 3: `apply_depth_alignment` (result mutation + provenance attrs)

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py` (same section as Task 2)
- Test: `tests/pointcloud/test_depth_alignment.py` (append)

- [ ] **Step 1: Append the failing test**

Depth grid = half native res (24×32), so model-res K halves `K_PARAMS`. With identity pose, the unprojected world z equals the scaled depth.

```python
def test_apply_depth_alignment_scales_depth_and_recomputes_world_points():
    recon = _recon(GRID, [(0.0, 0.0, 0.0)])
    grid_h, grid_w = H // 2, W // 2
    depth = np.full((1, grid_h, grid_w), 5.0 / 3.2, np.float32)
    intrinsics = np.array([[[50.0, 0.0, 16.0], [0.0, 50.0, 12.0], [0.0, 0.0, 1.0]]], np.float32)
    result = FeedforwardResult(
        points=np.zeros((1, 3), np.float32),
        colors=np.zeros((1, 3), np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
        intrinsics=intrinsics,
        image_paths=[Path("frame_000000.jpg")],
        original_coords=np.array([[0, 0, W, H, W, H]], np.float32),
        model_width=grid_w,
        model_height=grid_h,
        depth=depth,
        world_points=np.zeros((1, grid_h, grid_w, 3), np.float32),
    )

    attrs = apply_depth_alignment(result, recon)

    # Provenance attrs for save_zarr(extra_attrs=...)
    assert attrs["depth_scale"] == "colmap"
    assert attrs["depth_scales"] == pytest.approx([3.2], rel=1e-5)
    assert attrs["depth_scale_fallback_frames"] == []

    # Depth rescaled in place to the COLMAP world; world_points re-derived from it
    assert result.depth.dtype == np.float32
    assert result.depth[0, 0, 0] == pytest.approx(5.0, rel=1e-5)
    assert result.world_points.shape == (1, grid_h, grid_w, 3)
    assert result.world_points.dtype == np.float32
    # Identity pose: world z at every pixel equals the scaled depth
    assert result.world_points[0, 12, 16, 2] == pytest.approx(5.0, rel=1e-4)
```

Note: `FeedforwardResult` field names/order are in `collab_splats/pointcloud/feedforward/base.py:53-85` — if the constructor call above mismatches (it is keyword-only here, so order is safe), fix the test, not the dataclass.

- [ ] **Step 2: Run to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_alignment.py -q 2>&1 | tail -3
```

Expected: `ImportError: cannot import name 'apply_depth_alignment'` (or the Task 2 tests pass and only this one fails once the import resolves).

- [ ] **Step 3: Implement `apply_depth_alignment`**

Append in the same sfm.py section. `unproject_depth_map_to_point_map` comes from vggt — reconstructor.py imports it at top level, and sfm.py already sits behind heavy imports, so import it at the top of sfm.py (`from vggt.utils.geometry import unproject_depth_map_to_point_map`). Type-only import of `FeedforwardResult` goes under `TYPE_CHECKING` (a runtime import would be circular-risk free but unnecessary).

```python
def apply_depth_alignment(result: "FeedforwardResult", reconstruction: "pycolmap.Reconstruction") -> dict:
    """
    Scale result.depth to the reconstruction's world scale in place; recompute world_points.

    - Fits per-frame scales via align_depth_to_reconstruction over the result's rows, then
      re-derives dense world_points from the scaled depth under the COLMAP poses, so the
      zarr and the COLMAP model share one scale.
    - Returns the provenance attrs to merge into save_zarr's extra_attrs; raises on an
      unalignable scene — never a silent VDA-metric write.
    """
    names = [path.name for path in result.image_paths]
    scales, stats = align_depth_to_reconstruction(reconstruction, names, result.depth)
    logger.info(
        "depth alignment: global scale %.4f, ratio p10/p50/p90 %s -> %s, %d fallback frames",
        stats["global_scale"],
        [round(x, 4) for x in stats["ratio_p10_p50_p90_before"]],
        [round(x, 4) for x in stats["ratio_p10_p50_p90_after"]],
        stats["n_fallback"],
    )

    # Scale depth per frame; re-unproject dense world points (t is not scale-invariant,
    # so world_points cannot be scaled directly — they must be re-derived)
    result.depth = (result.depth * scales[:, None, None]).astype(np.float32)
    result.world_points = unproject_depth_map_to_point_map(
        result.depth[..., None], result.extrinsics[:, :3, :], result.intrinsics
    ).astype(np.float32)

    return {
        "depth_scale": "colmap",
        "depth_scales": [float(s) for s in scales],
        "depth_scale_fallback_frames": stats["fallback_frames"],
    }
```

- [ ] **Step 4: Run the whole test file**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_alignment.py -q 2>&1 | tail -3
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/pointcloud/test_depth_alignment.py collab_splats/pointcloud/sfm.py
git commit --only tests/pointcloud/test_depth_alignment.py collab_splats/pointcloud/sfm.py \
  -m "feat(sfm): per-frame VDA-to-COLMAP depth scale alignment

align_depth_to_reconstruction fits median-of-ratios scales from track
observations (MIN_ALIGN_OBS floor, global-median fallback);
apply_depth_alignment rescales FeedforwardResult depth in place,
re-unprojects world_points, and returns depth_scale provenance attrs.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `_run_sfm` wiring

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:1034-1044` (the `_sfm_result_from_reconstruction` → `save_zarr` block) and the sfm import block at `reconstructor.py:26-32`

- [ ] **Step 1: Add `apply_depth_alignment` to the existing sfm import**

At `reconstructor.py:26`, extend (isort wraps at 88 — keep parenthesized, alphabetical):

```python
from collab_splats.pointcloud.sfm import (
    InstantSfMCreator,
    _pixel_indices_from_reconstruction,
    _tracked_point3d_ids,
    apply_depth_alignment,
    generate_vda_depth,
    vda_depth_complete,
)
```

- [ ] **Step 2: Wire alignment between the builder and `save_zarr`**

Replace `reconstructor.py:1034-1044`:

```python
        # Unified pointcloud.zarr at VDA depth res, with provenance from the installed package
        outputs = self._sfm_result_from_reconstruction(recon, backend_dir, store)

        # Align VDA depth to the COLMAP world before anything persists — the zarr and the
        # model must share one scale (splat depth targets, mesh fusion, localization lookup).
        # Raises rather than writing a VDA-metric zarr; depth_scale attrs mark aligned scenes.
        align_attrs = apply_depth_alignment(outputs, recon)

        zarr_path = backend_dir / "pointcloud.zarr"
        outputs.save_zarr(
            zarr_path,
            extra_attrs={
                "method": "sfm",
                "backend": "instantsfm",
                "instantsfm_version": importlib.metadata.version("instantsfm"),
                **align_attrs,
            },
        )
```

(The builder's own `world_points` unprojection is recomputed by `apply_depth_alignment` — a second unproject over 300 model-res frames costs ~1 s and keeps the builder's contract untouched.)

- [ ] **Step 3: Run the neighboring test files**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_result.py tests/pointcloud/test_zarr_attrs.py tests/wrapper/test_sfm_config.py -q 2>&1 | tail -3
```

Expected: all pass (the wiring sits above `_sfm_result_from_reconstruction`, which these test directly; the attrs test builds its dict independently). The wiring itself is exercised end-to-end by Task 8's eval run — its INFO log line is the check.

- [ ] **Step 4: Commit**

```bash
git commit --only collab_splats/wrapper/reconstructor.py \
  -m "feat(sfm): align pointcloud.zarr depth to COLMAP scale in _run_sfm

depth_scale/depth_scales/depth_scale_fallback_frames stamped into zarr
attrs; absence of depth_scale marks a legacy VDA-metric scene.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: splats() — dense zarr targets on sfm + legacy guard

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:1501-1548` (splats depth-targets block)
- Test: `tests/wrapper/test_splats_stage.py` (append two tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/wrapper/test_splats_stage.py` (reuses its `_stub_reconstructor`; `zarr` is already imported there):

```python
def test_splats_sfm_aligned_zarr_uses_zarr_depth(tmp_path):
    # sfm scene whose zarr carries depth_scale: falls through to the zarr-depth path
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    depth = np.stack([np.full((4, 4), view + 1, np.float32) for view in range(3)])
    feedforward = SimpleNamespace(
        image_paths=[Path(f"frame_{view:06d}.jpg") for view in range(3)],
        depth=depth,
        confidence=None,  # sfm scenes carry no confidence — unmasked targets
    )
    group = zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    group.attrs["depth_scale"] = "colmap"
    with (
        patch("collab_splats.splats.trainer.train") as train,
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=feedforward),
    ):
        recon.splats()

    depth_targets = train.call_args.kwargs["depth_targets"]
    assert depth_targets.shape == (3, 4, 4)
    # Rows reordered to image_paths (reversed): row 0 is frame 2, row 2 is frame 0
    assert depth_targets[0, 0, 0] == 3 and depth_targets[2, 0, 0] == 1


def test_splats_sfm_legacy_zarr_refused(tmp_path):
    # sfm zarr without depth_scale is VDA-metric — feeding it as targets collapsed
    # training once (PSNR 6.15); must refuse with a re-run pointer
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    with patch("collab_splats.splats.trainer.train") as train, pytest.raises(ValueError, match="depth_scale"):
        recon.splats()
    train.assert_not_called()
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_splats_stage.py -q -k sfm 2>&1 | tail -3
```

Expected: 2 failed — the first hits the sparse `points3d_depth_maps` dispatch (AttributeError: `SimpleNamespace` result stub has no `reconstruction`), the second trains instead of raising.

- [ ] **Step 3: Replace the sfm dispatch with the guard**

Replace `reconstructor.py:1503-1522` (from `depth_targets = None` through the `FileNotFoundError` block; delete the whole sparse branch including its inline `from collab_splats.pointcloud.utils import points3d_depth_maps`):

```python
        depth_targets = None
        depth_on = "depth" in cfg.losses and cfg.losses["depth"]["weight"] > 0  # from_dict guarantees weight
        if depth_on:
            pointcloud_zarr = self.pointcloud_zarr
            if not pointcloud_zarr.exists():
                raise FileNotFoundError(
                    f"pointcloud.zarr not found at {pointcloud_zarr}. "
                    "Splats depth loss requires depth maps from the pointcloud stage."
                )

            # SfM scenes: zarr depth is usable only once aligned to the COLMAP world — a
            # legacy VDA-metric store fed as targets collapsed training (measured PSNR 6.15)
            if self.config["pointcloud"]["method"] == "sfm":
                attrs = zarr.open(str(pointcloud_zarr), mode="r").attrs
                if "depth_scale" not in attrs:
                    raise ValueError(
                        f"{pointcloud_zarr} predates depth alignment (no depth_scale attr) — "
                        "re-run the pointcloud stage to align VDA depth to the COLMAP world."
                    )

            feedforward = FeedforwardResult.load_zarr(pointcloud_zarr, load_images=False, load_world_points=False)
```

The rest of the block (`if feedforward.depth is None` onward, lines 1524-1548) stays byte-identical — the row alignment, confidence masking, and absent-confidence log already handle sfm scenes. Also update the block comment above (line 1501-1502) to say "model-res pointcloud.zarr depth" instead of "feedforward depth".

Guard note: `zarr.open(...).attrs` with membership `in` — NOT `.get(key, default)` (`test_no_inline_defaults_in_source`).

- [ ] **Step 4: Run the full splats-stage + absent-confidence test files**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_splats_stage.py tests/mesh/test_absent_confidence.py -q 2>&1 | tail -3
```

Expected: all pass (the pre-existing feedforward tests never take the sfm guard; the absent-confidence splats test uses method feedforward). If `tests/mesh/test_absent_confidence.py`'s module docstring still says "sfm scenes use points3d_depth_maps", update that sentence to "sfm scenes use the same zarr path once depth is aligned (depth_scale attr)".

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_splats_stage.py tests/mesh/test_absent_confidence.py \
  -m "feat(splats): dense aligned zarr depth targets on sfm scenes

sfm dispatch to points3d_depth_maps removed; sfm scenes take the shared
zarr-depth path, gated on the depth_scale attr (legacy VDA-metric zarr
refused with a re-run pointer).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: mesh() — lift the sfm refusal for aligned scenes

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:1329-1338` (the sfm refusal branch)
- Test: `tests/wrapper/test_splats_stage.py` (append two tests — the mesh-source tests already live there)

- [ ] **Step 1: Write the failing tests**

Append to `tests/wrapper/test_splats_stage.py`:

```python
def test_mesh_sfm_legacy_zarr_refused(tmp_path):
    # Legacy VDA-metric zarr against COLMAP poses fused geometry at the wrong scale —
    # keep refusing scenes without the depth_scale attr
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    with pytest.raises(ValueError, match="depth_scale"):
        recon.mesh()


def test_mesh_sfm_aligned_zarr_fuses(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    group = zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    group.attrs["depth_scale"] = "colmap"
    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as fuse:
        fuse.return_value = recon.backend_dir / "mesh" / "mesh.ply"
        out = recon.mesh()
    fuse.assert_called_once()
    assert out == recon.backend_dir / "mesh" / "mesh.ply"
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_splats_stage.py -q -k mesh_sfm 2>&1 | tail -3
```

Expected: 2 failed — both hit the unconditional "not scale-consistent" refusal (first fails on the match pattern, second on the raise).

- [ ] **Step 3: Gate the refusal on the attr**

Replace `reconstructor.py:1329-1338`:

```python
        elif self.config["pointcloud"]["method"] == "sfm":
            # SfM scenes fuse only after zarr depth was aligned to the COLMAP world
            # (depth_scale attr). A legacy VDA-metric store against COLMAP poses produced
            # geometry at the wrong scale in the wrong places (measured 3.2x on GH010229).
            attrs = zarr.open(str(pointcloud_zarr), mode="r").attrs
            if "depth_scale" not in attrs:
                raise ValueError(
                    f"{pointcloud_zarr} predates depth alignment (no depth_scale attr) — "
                    "re-run the pointcloud stage, or set mesh.source: splats."
                )
```

- [ ] **Step 4: Run the file**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_splats_stage.py -q 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_splats_stage.py \
  -m "feat(mesh): allow mesh.source feedforward on aligned sfm scenes

Refusal now keyed on the depth_scale zarr attr: aligned scenes fuse,
legacy VDA-metric scenes still refuse.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Full-suite gate

**Files:** none

- [ ] **Step 1: Run the full suite in the worktree**

```bash
cd /workspace/collab-splats/.claude/worktrees/insfm-exec
/opt/venv/reconstruction/bin/python -m pytest tests/ -q -x --ignore=tests/localization 2>&1 | tail -15
/opt/venv/reconstruction/bin/python -m pytest tests/localization -q 2>&1 | tail -5
```

(Split only if the single full run OOM-risks nothing else is running; otherwise one `pytest tests/ -q` is fine.)

- [ ] **Step 2: Compare failures against the known set**

Allowed failures: exactly those in `docs/known-test-failures.md` plus the 2 documented `tests/examples/test_run_pipeline_remote.py` scene-id failures, plus the 2 pre-existing failures recorded in memory (`project_instantsfm_backend`: suite 6 fail = 2 pre-existing + 4 stub regression fixed in `9b1c1b07`). Any NEW failure blocks Task 8 — fix it first.

---

### Task 8: GH010229 evaluation (the merge gate)

**Files:**
- Create: `/tmp/claude-0/-workspace-collab-splats/f9d55594-0017-4e34-9802-8bc874e6167f/scratchpad/eval_dense_align.py` (driver)
- Scene: `/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229` (backend dir `instantsfm/`)

Constraints: tmux only, 46.6 GB cgroup, no parallel GPU work, /workspace has ~8.7 GB write quota invisible to `df` — `splats.zarr` must live on /tmp via symlink.

- [ ] **Step 1: Back up the baseline artifacts before anything overwrites them**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/f9d55594-0017-4e34-9802-8bc874e6167f/scratchpad
SCENE=/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229/instantsfm
mkdir -p "$SCRATCH/baseline_sparse"
cp -a "$SCENE/splats/splats_quality_report.json" "$SCRATCH/baseline_sparse/" 2>/dev/null || echo "MISSING report"
cp -a "$SCENE/mesh" "$SCRATCH/baseline_sparse/mesh" 2>/dev/null || echo "MISSING mesh"
ls -la "$SCENE/splats/"   # note whether splats.zarr is a symlink and where it points
```

If the baseline `splats_quality_report.json` is missing, stop and report — the gate (18.38 / 0.574) has no artifact to re-verify against.

- [ ] **Step 2: Write the eval driver**

`$SCRATCH/eval_dense_align.py`:

```python
"""GH010229 dense-aligned eval: pointcloud tail + splats + mesh, worktree code."""
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Provenance: this MUST import the worktree, not the installed HEAD
import collab_splats
assert "worktrees" in collab_splats.__file__, f"wrong install: {collab_splats.__file__}"

from collab_splats.wrapper.reconstructor import Reconstructor

config = yaml.safe_load(open("/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229/run_config.yaml"))
config["semantics"]["enabled"] = False  # standing override
config["mesh"].update({
    "source": "splats",       # baseline mesh comparison first; feedforward probe is Step 6
    "voxel_size": 0.2,        # world-scale-derived params (~78x omega reference, NOT 3.2x)
    "sdf_trunc": 0.8,
    "depth_trunc": 100,
    "conf_percentile": 0,     # splat alpha is near-uniform — any percentile empties the mesh
})

recon = Reconstructor(config)
recon.build_pointcloud(overwrite=True)  # SIFT db + VDA npys cached -> warm tail, writes aligned zarr
recon.splats(overwrite=True)            # 3dgs pose_opt 30k
recon.mesh(overwrite=True)
```

- [ ] **Step 3: Run pointcloud + splats + mesh in tmux with worktree PYTHONPATH**

```bash
tmux new-session -d -s dense_eval
tmux send-keys -t dense_eval "cd /workspace/collab-splats/.claude/worktrees/insfm-exec && \
  PYTHONPATH=/workspace/collab-splats/.claude/worktrees/insfm-exec \
  /opt/venv/reconstruction/bin/python $SCRATCH/eval_dense_align.py 2>&1 | tee $SCRATCH/dense_eval.log" Enter
```

Watch `$SCRATCH/dense_eval.log`. **Before the splats stage starts writing**, confirm `$SCENE/splats/splats.zarr` still resolves to a /tmp target (`readlink` it); if the overwrite removed the symlink, kill the run after `build_pointcloud`, recreate it, and resume with a driver that skips the pointcloud call:

```bash
mkdir -p /tmp/claude-0/splats_local_GH010229
rm -rf "$SCENE/splats/splats.zarr"
ln -s /tmp/claude-0/splats_local_GH010229 "$SCENE/splats/splats.zarr"
```

- [ ] **Step 4: Verify the alignment stats (unit-level success check)**

In the log, find the `depth alignment:` INFO line. Required: global scale ≈ 3.2 (1/0.3092), `ratio p10/p50/p90` after ≪ the ±35% before-spread, fallback count small (0 on a fully-registered 300-frame scene). Also confirm the zarr:

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
import zarr
attrs = dict(zarr.open("/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229/instantsfm/pointcloud.zarr", mode="r").attrs)
print(attrs["depth_scale"], len(attrs["depth_scales"]), attrs["depth_scale_fallback_frames"][:5])
EOF
```

Expected: `colmap 300 []`.

- [ ] **Step 5: Score the gate**

```bash
cat "$SCENE/splats/splats_quality_report.json" | /opt/venv/reconstruction/bin/python -c \
  "import json,sys; r=json.load(sys.stdin); print(r)"
```

**Gate: PSNR > 18.38 AND SSIM > 0.574** (sparse baseline; omega reference 18.97 / 0.596 is the stretch). Record both numbers.

- [ ] **Step 6: Mesh renders from scene cameras (the only trusted mesh diagnostic)**

Render views 0/100/200 of the new `mesh/mesh.ply` and the backed-up baseline mesh with the same cameras; compare visually (coverage, truncation, speckle). Vertex counts alone are NOT a verdict — they hid a catastrophic truncation once.

```python
# $SCRATCH/render_mesh.py — run for both the new mesh and $SCRATCH/baseline_sparse/mesh/mesh.ply
import sys
import numpy as np
import open3d as o3d
import pycolmap

mesh_path, out_prefix = sys.argv[1], sys.argv[2]
scene = "/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229/instantsfm"
recon = pycolmap.Reconstruction(f"{scene}/colmap/sparse/0")
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

images = sorted(recon.images.values(), key=lambda im: im.name)
for idx in (0, 100, 200):
    im = images[idx]
    cam = recon.cameras[im.camera_id]
    renderer = o3d.visualization.rendering.OffscreenRenderer(cam.width, cam.height)
    renderer.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())
    intr = o3d.camera.PinholeCameraIntrinsic(cam.width, cam.height, cam.calibration_matrix())
    w2c = np.eye(4)
    w2c[:3, :] = im.cam_from_world().matrix()
    renderer.setup_camera(intr, w2c)
    o3d.io.write_image(f"{out_prefix}_view{idx:03d}.png", renderer.render_to_image())
    print(f"{out_prefix}_view{idx:03d}.png")
```

```bash
/opt/venv/reconstruction/bin/python $SCRATCH/render_mesh.py "$SCENE/mesh/mesh.ply" "$SCRATCH/dense_mesh"
/opt/venv/reconstruction/bin/python $SCRATCH/render_mesh.py "$SCRATCH/baseline_sparse/mesh/mesh.ply" "$SCRATCH/baseline_mesh"
```

- [ ] **Step 7 (bonus probe, non-gating): `mesh.source: feedforward` on the aligned zarr**

Same TSDF params; write to a probe dir so the splats mesh survives:

```python
# append to a copy of the driver, after moving the splats mesh aside
config["mesh"]["source"] = "feedforward"
recon.mesh(overwrite=True)
```

Render it with the same script. This is the first-ever feedforward fusion of an sfm scene — record what it looks like, no pass/fail.

- [ ] **Step 8: Record the verdict**

Write PSNR/SSIM (dense vs 18.38/0.574 vs omega 18.97/0.596), alignment stats, and the render comparison into `$SCRATCH/verdict.md`. This feeds Task 9.

---

### Task 9: Verdict-dependent retirement, docs, and merge

**Files (dense wins):**
- Delete: `tests/pointcloud/test_points3d_depth_maps.py`
- Modify: `collab_splats/pointcloud/utils.py` (remove `points3d_depth_maps`)
- Modify: `docs/superpowers/specs/2026-08-24-dense-vda-depth-targets-design.md` (append "## Measured" appendix)

**Files (dense loses):** spec appendix only — sparse dispatch must then be RESTORED in `splats()` (revert the Task 5 dispatch change but KEEP the legacy `depth_scale` guard and Task 4/6 — zarr rewrite and mesh-lift ship regardless, per spec §3).

- [ ] **Step 1 (only if PSNR/SSIM gate passed): retire the sparse path**

```bash
git rm tests/pointcloud/test_points3d_depth_maps.py
```

In `collab_splats/pointcloud/utils.py`, delete the `points3d_depth_maps` function (whole def at `utils.py:311` through its return). Verify nothing references it:

```bash
grep -rn "points3d_depth_maps" collab_splats/ tests/ docs/examples/ | grep -v pycache
```

Expected: no matches in code (spec/plan mentions are fine).

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper -q 2>&1 | tail -3
git commit --only collab_splats/pointcloud/utils.py tests/pointcloud/test_points3d_depth_maps.py \
  -m "refactor(pointcloud): retire points3d_depth_maps — dense aligned targets won the eval

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2 (only if the gate FAILED): restore the sparse dispatch, keep the rewrite**

Re-insert the sparse branch into `splats()` between the legacy guard and `load_zarr` (the guard stays — a legacy zarr is still refused):

```python
            # Dense aligned targets measured below the sparse baseline (see spec Measured
            # appendix) — keep upstream-style sparse points3D supervision on sfm scenes
            if self.config["pointcloud"]["method"] == "sfm":
                from collab_splats.pointcloud.utils import points3d_depth_maps

                height, width = images.shape[1:3]
                depth_targets = points3d_depth_maps(
                    result.reconstruction, [path.name for path in result.image_paths], height, width
                )
```

and re-add an early `if depth_on and ... == "sfm"` structure equivalent to the pre-Task-5 shape (adapt the Task 5 tests: the aligned-zarr test then asserts sparse targets, not zarr rows). Run `pytest tests/wrapper/test_splats_stage.py -q`, commit as `revert(splats): keep sparse sfm depth targets — dense lost the eval`.

- [ ] **Step 3: Append the Measured appendix to the spec**

Add to `docs/superpowers/specs/2026-08-24-dense-vda-depth-targets-design.md`:

```markdown
## Measured (GH010229, 300 frames, 3dgs pose_opt 30k)

| condition | PSNR | SSIM |
|---|---|---|
| sparse points3D targets (baseline) | 18.38 | 0.574 |
| dense aligned VDA targets | <measured> | <measured> |
| vggt_omega reference | 18.97 | 0.596 |

- Alignment: global scale <measured> (predicted ≈ 3.2), ratio p10/p50/p90 before <...> → after <...>, <n> fallback frames.
- Mesh (source: splats, voxel 0.2 / sdf 0.8 / trunc 100): render comparison verdict <one sentence>.
- Feedforward-fusion probe on the aligned zarr: <one sentence>.
- Verdict: <dense shipped / sparse kept>, consumer state <...>.
```

Fill every `<...>` from Task 8's `$SCRATCH/verdict.md`. Commit:

```bash
git add -f docs/superpowers/specs/2026-08-24-dense-vda-depth-targets-design.md
git commit --only docs/superpowers/specs/2026-08-24-dense-vda-depth-targets-design.md \
  -m "docs(specs): dense VDA depth targets — measured verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Update the CLAUDE.md instantsfm known-limitation lines**

In the worktree `CLAUDE.md` instantsfm paragraph: the "zarr depth/world_points VDA-metric vs COLMAP poses" known limitation and the splats/mesh consequences are now fixed for scenes carrying `depth_scale: colmap` (legacy scenes need a pointcloud re-run). Keep the dashboard `_ensure_lift_inputs` note. Commit with `git commit --only CLAUDE.md -m "docs: sfm depth now COLMAP-scale-aligned (depth_scale attr)"` (+ Co-Authored-By trailer).

- [ ] **Step 5: Merge chain (check `.git/sequencer` and concurrent-session state first)**

```bash
cd /workspace/collab-splats/.claude/worktrees/insfm-exec
git switch insfm-exec && git merge --ff-only dense-vda-align   # ff expected: branched from its tip
cd /workspace/collab-splats
git status --short   # concurrent sessions: do not sweep foreign staged work
git merge insfm-exec  # trunk gets 45077365 + 9b1c1b07 + this feature (the owed insfm-exec merge)
/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -5   # suite on merged trunk
```

If trunk has uncommitted foreign changes, stop and report instead of merging over them.

---

## Self-review notes

- Spec §1 (fit core) → Task 2; §2 (wiring + attrs + raise-not-silent) → Tasks 3-4; §3 consumers (splats fall-through, mesh lift, retirement, lose-branch) → Tasks 5, 6, 9; §Execution order (worktree → unit tests → suite → eval → verdict) → Tasks 1, 2-6, 7, 8, 9. Non-goals untouched.
- `min_obs=20` = `MIN_ALIGN_OBS`, module constant, no config surface (spec §1). Attr names `depth_scale`/`depth_scales`/`depth_scale_fallback_frames` consistent across Tasks 3/4/5/6/8/9.
- The Task 2 Step 4 checkpoint is deliberately soft (file imports Task 3's symbol); the honest red/green boundary is Task 3 Step 4.
- Known unknown: whether `splats(overwrite=True)` deletes the splats.zarr symlink — Task 8 Step 3 carries the detection + recovery procedure rather than an assumption.
