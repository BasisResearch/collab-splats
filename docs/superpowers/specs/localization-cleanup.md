# Spec: localization.py cleanup + localization tutorial notebook

**Date:** 2026-05-21
**Branch:** `refactor/core-modules`
**Status:** In-progress — localization.py changes applied, notebook not yet written

---

## Goal

Two deliverables:
1. Clean up `collab_splats/pointcloud/localization.py` — add missing base class for local extractors, surface RANSAC output as a typed result dataclass.
2. Write `docs/source/tutorials/pointcloud/localization.ipynb` — single tutorial notebook showing how to localize a new camera pose within a scene built by the feedforward pipeline.

---

## Current state (already applied to branch)

These changes are already in `collab_splats/pointcloud/localization.py`:

- `from typing import Dict` removed; built-in `dict` used throughout
- `BaseRetrievalExtractor._registry` type updated to `dict[...]`
- `LocalizationResult` dataclass added after `LocalFeatures`
- `BaseLocalExtractor(RegistryMixin, ABC)` added between Stage 1 / Stage 2 section dividers
- `DiskExtractor` registered as `"disk"`, inherits `BaseLocalExtractor`
- `XFeatExtractor` registered as `"xfeat"`, inherits `BaseLocalExtractor`
- `CameraLocalizer.localize()` updated to return `LocalizationResult` instead of `np.ndarray | None`

These changes are already in `tests/pointcloud/test_localization.py`:

- Imports updated to include `BaseLocalExtractor`, `LocalizationResult`, `DiskExtractor`, `XFeatExtractor`
- `test_base_local_extractor_registry()` added
- `test_camera_localizer_recovers_known_pose` updated to use `result.pose` instead of bare return value

---

## What still needs to be done

### 1. Verify tests pass

Run:
```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -v -m "not slow"
```

Note: the conftest at `tests/pointcloud/conftest.py` imports `collab_splats.pointcloud` which triggers a broken `timm.layers.DropPath` import chain through `mapanything`. Use `--noconftest` if needed, or run tests via direct python import.

### 2. Add `LocalizationResult` failure-path test

Add to `tests/pointcloud/test_localization.py` using `_make_synthetic_scene` + `_MockExtractor` helpers already in the file:

```python
def test_localization_result_fields_on_success():
    from collab_splats.pointcloud.localization import CameraLocalizer, LocalizationResult
    import tempfile, pathlib

    pts3d, extrinsics, intrinsics = _make_synthetic_scene()
    K = intrinsics[0]

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(len(extrinsics)):
            p = pathlib.Path(tmp) / f"frame_{i:04d}.png"
            import cv2 as _cv2
            _cv2.imwrite(str(p), np.zeros((480, 640, 3), dtype=np.uint8))
            paths.append(p)

        extractor = _MockExtractor(pts3d, extrinsics, intrinsics)
        loc = CameraLocalizer(pts3d, extrinsics, intrinsics, paths, extractor=extractor)
        result = loc.localize(np.zeros((480, 640, 3), dtype=np.uint8), K)

        assert isinstance(result, LocalizationResult)
        assert result.pose is not None
        assert result.inlier_mask is not None
        assert result.inlier_mask.dtype == bool
        assert len(result.inlier_mask) == result.n_correspondences
        assert result.n_inliers > 0
```

### 3. Write tutorial notebook

**File:** `docs/source/tutorials/pointcloud/localization.ipynb`

Audience: internal team. Terse prose. Builds directly off `feedforward_exploration.ipynb` — assumes `FeedforwardResult` from VGGT-X on C0043 data is already in memory or re-loadable from pkl.

**Cell outline:**

| # | Type | Content |
|---|------|---------|
| 1 | Markdown | What localization is; when to use it (map built, new camera arrives) |
| 2 | Code | Imports; load C0043 `FeedforwardResult` from pkl (same path as feedforward notebook) |
| 3 | Code | Hold out frame at index N (e.g. 10); slice `pts3d`, `extrinsics`, `intrinsics`, `image_paths` to exclude it; keep query RGB + intrinsics as ground truth |
| 4 | Code | `localizer = CameraLocalizer.from_feedforward(result_trimmed, extractor=XFeatExtractor())` |
| 5 | Code | `loc = localizer.localize(query_image, query_K)`; print `loc.n_correspondences`, `loc.n_inliers` |
| 6 | Code | Scatter `loc.pts2d` on query image — green = `loc.inlier_mask`, red = outliers (matplotlib) |
| 7 | Code | PyVista: point cloud + grey reference frustums + red estimated frustum + green gt frustum |
| 8 | Code | Print rotation error (°) and translation error (cm) vs gt pose |

**Visualization details:**

Cell 6 (keypoint scatter):
```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(query_image)
pts = loc.pts2d
mask = loc.inlier_mask
ax.scatter(pts[~mask, 0], pts[~mask, 1], s=6, c='red', label='outlier')
ax.scatter(pts[mask, 0], pts[mask, 1], s=6, c='lime', label='inlier')
ax.legend()
ax.set_title(f"PnP correspondences — {loc.n_inliers}/{loc.n_correspondences} inliers")
```

Cell 7 (PyVista): use `pointcloud_to_polydata()` from `collab_splats/utils/visualization.py` for the point cloud. Draw camera frustums as 5-point wireframes (4 image corners + principal point, projected to a fixed depth).

Cell 8 (error metrics):
```python
R_est = loc.pose[:3, :3]
R_gt  = gt_extrinsic[:3, :3]
rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(R_est @ R_gt.T) - 1) / 2, -1, 1)))
t_err_cm = np.linalg.norm(loc.pose[:3, 3] - gt_extrinsic[:3, 3]) * 100
print(f"Rotation error: {rot_err_deg:.2f}°  |  Translation error: {t_err_cm:.1f} cm")
```

### 4. Format + commit

```bash
cd /workspace/collab-splats
black collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
isort collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -v -m "not slow"
```

Commit messages:
- `refactor(localization): add BaseLocalExtractor ABC + LocalizationResult`
- `test(localization): update tests for LocalizationResult return type`
- `docs(tutorials): add localization tutorial notebook`

---

## Key design decisions (from conversation)

| Decision | Choice | Reason |
|----------|--------|--------|
| `LocalizationResult` vs bare `np.ndarray` | Add dataclass | Surfaces RANSAC inlier/outlier data already computed — zero extra cost; enables visualization |
| `match_frame()` helper | Skip | Minor convenience only; notebook can use `XFeatExtractor` directly for any frame-pair visualization |
| `backbone`/`aggregator` visibility in `DinoSaladExtractor` | Stay public | Checkpoint key names must match VPRModel (`backbone.*`, `aggregator.*`) — documented in comment |
| `pts3d` parameter naming | Keep as-is | CV convention; mirrors `FeedforwardResult.pts3d` attribute |
| Number of notebooks | One | End-to-end demo preferred over separate comparison/failure-modes/extractor notebooks |

---

## Files modified

| File | Change |
|------|--------|
| `collab_splats/pointcloud/localization.py` | Add `BaseLocalExtractor`, `LocalizationResult`; update `localize()` return type; drop `typing.Dict` |
| `tests/pointcloud/test_localization.py` | Update imports, update pose-recovery test, add registry + result-shape tests |
| `docs/source/tutorials/pointcloud/localization.ipynb` | New file — localization tutorial |

---

## Utilities to reuse

- `pointcloud_to_polydata()` — `collab_splats/utils/visualization.py`
- `CameraLocalizer.from_feedforward()` — `collab_splats/pointcloud/localization.py`
- `XFeatExtractor` — `collab_splats/pointcloud/localization.py`
- C0043 pkl from `feedforward_exploration.ipynb` — same data path, no new downloads needed
