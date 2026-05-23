# Design: localization-cleanup — test + tutorial notebook

**Date:** 2026-05-21
**Branch:** `refactor/core-modules`
**Status:** Approved

---

## Context

`localization.py` refactor is mostly on branch: `BaseLocalExtractor`, `LocalizationResult`,
`DiskExtractor`, `XFeatExtractor` in place. Tests partially updated. Three things remain.

---

## Scope

1. `collab_splats/pointcloud/localization.py` — extend `LocalizationResult` + `localize()` to surface reference-frame 2D coords and frame indices
2. One additional test in `tests/pointcloud/test_localization.py`
3. New tutorial notebook `docs/pointcloud/localization.ipynb`

---

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Tutorial data source | Run VGGTXCreator inline | No pkl in feedforward notebook; self-contained |
| Frame extraction | `score_all_frames` keyframe selection | Same optical-flow approach as feedforward tutorial |
| GT comparison | Feedforward pseudo-reference (`result.extrinsics[10]`) | C0043 has no ground-truth poses; labeled as consistency check |
| Correspondence viz | hloc-style connecting lines between query + best reference frame | Requires storing ref-frame 2D coords — small `LocalizationResult` extension |
| Frustum helper | Reuse `create_camera_frustum_pyvista` from `visualization.py` | Already exists (L310) |
| Failure-path test | Skip | Focus is usage demonstration |

---

## localization.py changes

### `LocalizationResult` — add two fields

```python
pts2d_ref: np.ndarray | None     # (M, 2) reference-frame pixel coords per correspondence
ref_frame_indices: np.ndarray | None  # (M,) int — which reference frame each correspondence came from
```

### `localize()` — track ref coords in matching loop

Currently the loop stores `best_2d[pt3d_idx] = pt2d` (query side only). Also track:

```python
best_ref_2d[pt3d_idx] = db_feats.keypoints[int(db_idx)].numpy().astype(np.float32)
best_ref_frame[pt3d_idx] = i
```

Assemble into arrays alongside `pts2d` and include in returned `LocalizationResult`.

---

## localization.py — visualization helper

New section at bottom of `localization.py` under `########` divider:

```python
########################################################
########## Visualization ###############################
########################################################


def plot_correspondences(
    loc: LocalizationResult,
    query_image: np.ndarray,
    image_paths: list,
    max_pairs: int = 200,
) -> None:
    """Side-by-side query + best reference frame with inlier/outlier connecting lines.

    Best reference frame = the one contributing the most inlier correspondences.
    Lines are green for inliers, red for outliers. White dots mark each keypoint.

    Args:
        loc:         LocalizationResult from CameraLocalizer.localize().
        query_image: HxWx3 uint8 RGB query image.
        image_paths: Reference image paths (same order as CameraLocalizer input).
        max_pairs:   Cap on number of lines drawn (random subsample if exceeded).
    """
    import matplotlib.pyplot as plt

    if loc.pose is None or loc.pts2d_ref is None:
        logger.warning("plot_correspondences: no valid localization result to plot")
        return

    # Pick best reference frame (most inlier correspondences)
    inlier_frames = loc.ref_frame_indices[loc.inlier_mask]
    best_ref_idx = int(np.bincount(inlier_frames).argmax())
    frame_mask = loc.ref_frame_indices == best_ref_idx

    kpts0 = loc.pts2d[frame_mask]           # (K, 2) query
    kpts1 = loc.pts2d_ref[frame_mask]       # (K, 2) reference
    inliers = loc.inlier_mask[frame_mask]   # (K,) bool

    # Subsample if needed
    if len(kpts0) > max_pairs:
        idx = np.random.choice(len(kpts0), max_pairs, replace=False)
        kpts0, kpts1, inliers = kpts0[idx], kpts1[idx], inliers[idx]

    ref_bgr = cv2.imread(str(image_paths[best_ref_idx]))
    ref_image = ref_bgr[..., ::-1].copy()

    W = query_image.shape[1]
    combined = np.concatenate([query_image, ref_image], axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(combined)
    for (x0, y0), (x1, y1), ok in zip(kpts0, kpts1, inliers):
        color = "lime" if ok else "red"
        ax.plot([x0, x1 + W], [y0, y1], color=color, linewidth=0.8, alpha=0.6)
    ax.scatter(kpts0[:, 0], kpts0[:, 1], s=8, c="white", zorder=3, linewidths=0)
    ax.scatter(kpts1[:, 0] + W, kpts1[:, 1], s=8, c="white", zorder=3, linewidths=0)
    ax.axvline(W, color="white", linewidth=1, alpha=0.5)
    ax.axis("off")
    n_shown = inliers.sum()
    ax.set_title(
        f"query ↔ reference frame {best_ref_idx} — "
        f"{n_shown}/{len(inliers)} inliers shown"
    )
    plt.tight_layout()
    plt.show()
```

Notebook Cell 5 becomes a single call:
```python
plot_correspondences(loc, query_image, result_trimmed.image_paths)
```

---

## Test

**File:** `tests/pointcloud/test_localization.py`

Add `test_localization_result_fields_on_success` after `test_camera_localizer_recovers_known_pose`.
Uses existing `_make_synthetic_scene` + `_MockExtractor` helpers.

Asserts on success result:
- `isinstance(result, LocalizationResult)`
- `result.pose is not None`
- `result.inlier_mask.dtype == bool`
- `len(result.inlier_mask) == result.n_correspondences`
- `result.n_inliers > 0`
- `result.pts2d_ref is not None and result.pts2d_ref.shape == result.pts2d.shape`
- `result.ref_frame_indices is not None and len(result.ref_frame_indices) == result.n_correspondences`

Full test skeleton in `worklog/specs/localization-cleanup.md` §2 (update assertions).

---

## Tutorial notebook

**File:** `docs/pointcloud/localization.ipynb`
Symlinked from `docs/source/tutorials/pointcloud/localization.ipynb`.

### Cell outline

| # | Type | Content |
|---|------|---------|
| 1 | MD | One sentence: what localization does and when to use it |
| 2 | Code | Imports + extract C0043 keyframes via `score_all_frames` + run VGGTXCreator |
| 3 | Code | Hold out frame idx=10; build `result_trimmed`; keep `query_image`, `query_K`, `ref_extrinsic` |
| 4 | Code | Build localizer + localize; print n_correspondences + n_inliers |
| 5 | Code | `plot_correspondences(loc, query_image, result_trimmed.image_paths)` |
| 6 | Code | PyVista: cloud + grey reference frustums + red localized frustum |
| 7 | Code | Rotation + translation error vs `ref_extrinsic` (consistency check) |

### Key patterns

**Cell 2 — extract + reconstruct:**
```python
VIDEO_PATH = "/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4"
FRAME_DIR = Path("/tmp/localization_tutorial/frames")
FRAME_DIR.mkdir(parents=True, exist_ok=True)

frame_scores = score_all_frames(VIDEO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
for s in frame_scores:
    if s["selected"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame_idx"])
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(FRAME_DIR / f"frame_{s['frame_idx']:05d}.jpg"), frame)
cap.release()

device = "cuda" if torch.cuda.is_available() else "cpu"
creator = VGGTXCreator()
creator.load_model(device=device)
creator.setup_inference(FRAME_DIR)
creator.run_inference()
creator.postprocess()
result = creator.outputs
```

**Cell 6 — PyVista frustums:**
```python
pl = pv.Plotter()
pl.add_mesh(pointcloud_to_polydata(result_trimmed.pts3d), point_size=2, render_points_as_spheres=True)
for ext in result_trimmed.extrinsics:
    pl.add_mesh(create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05), color="grey", line_width=1)
pl.add_mesh(create_camera_frustum_pyvista(np.linalg.inv(loc.pose), scale=0.05), color="red", line_width=3)
pl.show()
```

**Cell 7 — error:**
```python
R_est = loc.pose[:3, :3]
R_ref  = ref_extrinsic[:3, :3]
rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(R_est @ R_ref.T) - 1) / 2, -1, 1)))
t_err_cm = np.linalg.norm(loc.pose[:3, 3] - ref_extrinsic[:3, 3]) * 100
print(f"vs feedforward reference — rotation: {rot_err_deg:.2f}°  |  translation: {t_err_cm:.1f} cm")
```

---

## Files modified

| File | Change |
|---|---|
| `collab_splats/pointcloud/localization.py` | Add `pts2d_ref` + `ref_frame_indices` to `LocalizationResult`; track in `localize()`; add `plot_correspondences()` under new Visualization section |
| `tests/pointcloud/test_localization.py` | Add `test_localization_result_fields_on_success` (updated assertions) |
| `docs/pointcloud/localization.ipynb` | New file — localization tutorial |
| `docs/source/tutorials/pointcloud/localization.ipynb` | Symlink to above |

---

## Utilities reused

- `create_camera_frustum_pyvista` — `collab_splats/utils/visualization.py:310`
- `pointcloud_to_polydata` — `collab_splats/utils/visualization.py:380`
- `CameraLocalizer.from_feedforward` — `collab_splats/pointcloud/localization.py:521`
- `XFeatExtractor` — `collab_splats/pointcloud/localization.py:277`
- `score_all_frames` — `collab_splats/utils/frame_sampling.py`

---

## Verification

1. `pytest tests/pointcloud/test_localization.py -v -m "not slow"` — all fast tests pass
2. Run notebook cells 1–4; confirm `loc.n_inliers > 0` and `loc.pts2d_ref` not None
3. Cell 5 renders side-by-side with green/red lines
4. Cell 6 renders reference (grey) + localized (red) frustums in PyVista
