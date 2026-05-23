# localization-cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `LocalizationResult` to surface reference-frame correspondence coords, add a `plot_correspondences()` viz helper, and write a 7-cell tutorial notebook demonstrating end-to-end localization on C0043.

**Architecture:** Three targeted edits to `localization.py` (dataclass fields, matching loop tracking, new viz section), one test addition, one new notebook. No new modules.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PyTorch, matplotlib, PyVista, kornia/DISK, XFeat (vendored at `vendor/xfeat/`). Always use `/opt/conda/envs/nerfstudio/bin/python`.

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/localization.py` | Add 2 fields to `LocalizationResult`; update `localize()` matching loop and all 3 return statements; add `plot_correspondences()` in new Visualization section |
| `tests/pointcloud/test_localization.py` | Add `test_localization_result_fields_on_success` |
| `docs/pointcloud/localization.ipynb` | New 7-cell tutorial notebook |
| `docs/source/tutorials/pointcloud/localization.ipynb` | Symlink to above |

---

### Task 1: Extend LocalizationResult + update localize()

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization.py`

- [ ] **Step 1: Add failing test**

Append to `tests/pointcloud/test_localization.py` after `test_camera_localizer_recovers_known_pose`:

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
        assert result.pts2d_ref is not None
        assert result.pts2d_ref.shape == result.pts2d.shape
        assert result.ref_frame_indices is not None
        assert len(result.ref_frame_indices) == result.n_correspondences
```

- [ ] **Step 2: Run test — verify it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py::test_localization_result_fields_on_success -v --noconftest
```

Expected: `AttributeError: 'LocalizationResult' object has no attribute 'pts2d_ref'`

- [ ] **Step 3: Add two fields to LocalizationResult**

In `collab_splats/pointcloud/localization.py`, replace the `LocalizationResult` dataclass (lines 57–72):

```python
@dataclass
class LocalizationResult:
    """Output of CameraLocalizer.localize().

    pts2d and pts3d_matched are the M correspondences fed to PnP.
    inlier_mask[i] is True when correspondence i survived RANSAC.
    pts2d_ref[i] is the matched keypoint position in the reference frame.
    ref_frame_indices[i] is which reference frame the match came from.
    pose is None when PnP fails or fewer than 4 correspondences exist.
    """

    pose: np.ndarray | None                      # (4, 4) world-to-camera, or None
    n_correspondences: int                        # M — total 2D↔3D pairs before RANSAC
    n_inliers: int                                # RANSAC inlier count
    pts2d: np.ndarray | None                      # (M, 2) query pixel coords
    pts3d_matched: np.ndarray | None              # (M, 3) matched world points
    inlier_mask: np.ndarray | None                # (M,) bool
    pts2d_ref: np.ndarray | None = None           # (M, 2) reference-frame pixel coords
    ref_frame_indices: np.ndarray | None = None   # (M,) int — source reference frame
```

- [ ] **Step 4: Add tracking dicts to localize() matching block**

In `localize()`, find (around line 563):

```python
        best_score: dict[int, float] = {}    # pt3d_idx → best confidence so far
        best_2d: dict[int, np.ndarray] = {}  # pt3d_idx → corresponding query 2D point
```

Replace with:

```python
        best_score: dict[int, float] = {}         # pt3d_idx → best confidence so far
        best_2d: dict[int, np.ndarray] = {}        # pt3d_idx → query 2D point
        best_ref_2d: dict[int, np.ndarray] = {}    # pt3d_idx → reference 2D point
        best_ref_frame: dict[int, int] = {}        # pt3d_idx → reference frame index
```

- [ ] **Step 5: Track ref coords in the inner assignment loop**

In `localize()`, find (around line 583):

```python
                if pt3d_idx not in best_score or score > best_score[pt3d_idx]:
                    best_score[pt3d_idx] = score
                    best_2d[pt3d_idx] = pt2d
```

Replace with:

```python
                if pt3d_idx not in best_score or score > best_score[pt3d_idx]:
                    best_score[pt3d_idx] = score
                    best_2d[pt3d_idx] = pt2d
                    best_ref_2d[pt3d_idx] = db_feats.keypoints[int(db_idx)].numpy().astype(np.float32)
                    best_ref_frame[pt3d_idx] = i
```

- [ ] **Step 6: Update early-return (< 4 correspondences)**

Find (around line 588):

```python
            return LocalizationResult(
                pose=None, n_correspondences=len(best_2d), n_inliers=0,
                pts2d=None, pts3d_matched=None, inlier_mask=None,
            )
```

Replace with:

```python
            return LocalizationResult(
                pose=None, n_correspondences=len(best_2d), n_inliers=0,
                pts2d=None, pts3d_matched=None, inlier_mask=None,
                pts2d_ref=None, ref_frame_indices=None,
            )
```

- [ ] **Step 7: Assemble ref arrays before PnP block**

Find (around line 599):

```python
        # Assemble correspondence arrays for PnP
        pts2d = np.array(list(best_2d.values()), dtype=np.float32)
        pts3d_matched = np.array(
            [self._pts3d[idx] for idx in best_2d.keys()], dtype=np.float32
        )
```

Replace with:

```python
        # Assemble correspondence arrays for PnP
        pts2d = np.array(list(best_2d.values()), dtype=np.float32)
        pts3d_matched = np.array(
            [self._pts3d[idx] for idx in best_2d.keys()], dtype=np.float32
        )
        pts2d_ref = np.array(list(best_ref_2d.values()), dtype=np.float32)
        ref_frame_indices = np.array(list(best_ref_frame.values()), dtype=np.int32)
```

- [ ] **Step 8: Update PnP-failure return**

Find (around line 621):

```python
            return LocalizationResult(
                pose=None, n_correspondences=len(pts2d),
                n_inliers=len(inliers) if inliers is not None else 0,
                pts2d=pts2d, pts3d_matched=pts3d_matched, inlier_mask=None,
            )
```

Replace with:

```python
            return LocalizationResult(
                pose=None, n_correspondences=len(pts2d),
                n_inliers=len(inliers) if inliers is not None else 0,
                pts2d=pts2d, pts3d_matched=pts3d_matched, inlier_mask=None,
                pts2d_ref=pts2d_ref, ref_frame_indices=ref_frame_indices,
            )
```

- [ ] **Step 9: Update success return**

Find (around line 639):

```python
        return LocalizationResult(
            pose=pose, n_correspondences=len(pts2d), n_inliers=len(inliers),
            pts2d=pts2d, pts3d_matched=pts3d_matched, inlier_mask=inlier_mask,
        )
```

Replace with:

```python
        return LocalizationResult(
            pose=pose, n_correspondences=len(pts2d), n_inliers=len(inliers),
            pts2d=pts2d, pts3d_matched=pts3d_matched, inlier_mask=inlier_mask,
            pts2d_ref=pts2d_ref, ref_frame_indices=ref_frame_indices,
        )
```

- [ ] **Step 10: Run all fast tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -v -m "not slow" --noconftest
```

Expected: All tests pass including `test_localization_result_fields_on_success`.

- [ ] **Step 11: Format and commit**

```bash
cd /workspace/collab-splats
black collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
isort collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "refactor(localization): surface ref-frame 2D coords in LocalizationResult"
```

---

### Task 2: Add plot_correspondences() visualization function

**Files:**
- Modify: `collab_splats/pointcloud/localization.py` (append new section)

- [ ] **Step 1: Append Visualization section to localization.py**

At the very end of `collab_splats/pointcloud/localization.py`, add:

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

    Best reference frame = one contributing the most inlier correspondences.
    Lines are green for inliers, red for outliers. White dots mark each keypoint.

    Args:
        loc:         LocalizationResult from CameraLocalizer.localize().
        query_image: HxWx3 uint8 RGB query image.
        image_paths: Reference image paths (same order as CameraLocalizer input).
        max_pairs:   Cap on lines drawn — random subsample if exceeded.
    """
    import matplotlib.pyplot as plt

    if loc.pose is None or loc.pts2d_ref is None or loc.ref_frame_indices is None:
        logger.warning("plot_correspondences: no valid localization result to plot")
        return

    # Best reference frame = one with most inlier correspondences
    inlier_frames = loc.ref_frame_indices[loc.inlier_mask]
    best_ref_idx = int(np.bincount(inlier_frames).argmax())
    frame_mask = loc.ref_frame_indices == best_ref_idx

    kpts0 = loc.pts2d[frame_mask]          # (K, 2) query
    kpts1 = loc.pts2d_ref[frame_mask]      # (K, 2) reference
    inliers = loc.inlier_mask[frame_mask]  # (K,) bool

    if len(kpts0) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(kpts0), max_pairs, replace=False)
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
    ax.set_title(
        f"query ↔ reference frame {best_ref_idx} — "
        f"{inliers.sum()}/{len(inliers)} inliers shown"
    )
    plt.tight_layout()
    plt.show()
```

- [ ] **Step 2: Run fast tests to confirm nothing broken**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -v -m "not slow" --noconftest
```

Expected: All pass.

- [ ] **Step 3: Format and commit**

```bash
cd /workspace/collab-splats
black collab_splats/pointcloud/localization.py
isort collab_splats/pointcloud/localization.py
git add collab_splats/pointcloud/localization.py
git commit -m "feat(localization): add plot_correspondences() visualization helper"
```

---

### Task 3: Tutorial notebook

**Files:**
- Create: `docs/pointcloud/localization.ipynb`
- Create: `docs/source/tutorials/pointcloud/localization.ipynb` (symlink)

- [ ] **Step 1: Write notebook**

Create `docs/pointcloud/localization.ipynb`. Use the Write tool with this JSON structure (nbformat 4.4):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Camera Localization\n",
    "\n",
    "Given a 3D scene reconstructed by the feedforward pipeline, find where a new camera is positioned within it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import pyvista as pv\n",
    "import torch\n",
    "from pathlib import Path\n",
    "\n",
    "from collab_splats.pointcloud.feedforward import VGGTXCreator\n",
    "from collab_splats.pointcloud.localization import CameraLocalizer, XFeatExtractor, plot_correspondences\n",
    "from collab_splats.utils.frame_sampling import score_all_frames\n",
    "from collab_splats.utils.visualization import create_camera_frustum_pyvista, pointcloud_to_polydata\n",
    "\n",
    "pv.set_jupyter_backend(\"trame\")\n",
    "\n",
    "VIDEO_PATH = \"/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4\"\n",
    "FRAME_DIR = Path(\"/tmp/localization_tutorial/frames\")\n",
    "FRAME_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Extract keyframes using optical-flow selection\n",
    "frame_scores = score_all_frames(VIDEO_PATH)\n",
    "cap = cv2.VideoCapture(VIDEO_PATH)\n",
    "for s in frame_scores:\n",
    "    if s[\"selected\"]:\n",
    "        cap.set(cv2.CAP_PROP_POS_FRAMES, s[\"frame_idx\"])\n",
    "        ret, frame = cap.read()\n",
    "        if ret:\n",
    "            cv2.imwrite(str(FRAME_DIR / f\"frame_{s['frame_idx']:05d}.jpg\"), frame)\n",
    "cap.release()\n",
    "\n",
    "# Build point cloud\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "creator = VGGTXCreator()\n",
    "creator.load_model(device=device)\n",
    "creator.setup_inference(FRAME_DIR)\n",
    "creator.run_inference()\n",
    "creator.postprocess()\n",
    "result = creator.outputs\n",
    "print(f\"pts3d: {result.pts3d.shape}  frames: {result.extrinsics.shape[0]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "QUERY_IDX = 10\n",
    "\n",
    "# Query frame — the camera we want to localize\n",
    "query_bgr = cv2.imread(str(result.image_paths[QUERY_IDX]))\n",
    "query_image = query_bgr[..., ::-1].copy()\n",
    "query_K = result.intrinsics[QUERY_IDX]\n",
    "ref_extrinsic = result.extrinsics[QUERY_IDX]  # feedforward reference for consistency check\n",
    "\n",
    "# Scene without the query frame\n",
    "keep = [i for i in range(len(result.image_paths)) if i != QUERY_IDX]\n",
    "\n",
    "class _Trimmed:\n",
    "    pts3d = result.pts3d\n",
    "    extrinsics = result.extrinsics[keep]\n",
    "    intrinsics = result.intrinsics[keep]\n",
    "    image_paths = [result.image_paths[i] for i in keep]\n",
    "\n",
    "result_trimmed = _Trimmed()\n",
    "print(f\"Scene: {len(result_trimmed.image_paths)} reference frames, query held out at index {QUERY_IDX}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "localizer = CameraLocalizer.from_feedforward(result_trimmed, extractor=XFeatExtractor())\n",
    "loc = localizer.localize(query_image, query_K)\n",
    "print(f\"correspondences: {loc.n_correspondences}  inliers: {loc.n_inliers}  pose: {'found' if loc.pose is not None else 'FAILED'}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_correspondences(loc, query_image, result_trimmed.image_paths)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pl = pv.Plotter()\n",
    "pl.add_mesh(pointcloud_to_polydata(result_trimmed.pts3d), point_size=2, render_points_as_spheres=True)\n",
    "for ext in result_trimmed.extrinsics:\n",
    "    pl.add_mesh(create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05), color=\"grey\", line_width=1)\n",
    "pl.add_mesh(create_camera_frustum_pyvista(np.linalg.inv(loc.pose), scale=0.05), color=\"red\", line_width=3)\n",
    "pl.add_axes()\n",
    "pl.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "R_est = loc.pose[:3, :3]\n",
    "R_ref  = ref_extrinsic[:3, :3]\n",
    "rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(R_est @ R_ref.T) - 1) / 2, -1, 1)))\n",
    "t_err_cm = np.linalg.norm(loc.pose[:3, 3] - ref_extrinsic[:3, 3]) * 100\n",
    "print(f\"vs feedforward reference — rotation: {rot_err_deg:.2f}°  |  translation: {t_err_cm:.1f} cm\")\n",
    "print(\"(consistency check vs feedforward model — not absolute ground truth)\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "nerfstudio",
   "language": "python",
   "name": "nerfstudio"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Create symlink**

```bash
cd /workspace/collab-splats/docs/source/tutorials/pointcloud
ln -s ../../../../docs/pointcloud/localization.ipynb localization.ipynb
ls -la localization.ipynb
```

Expected: symlink pointing to `../../../../docs/pointcloud/localization.ipynb`

- [ ] **Step 3: Verify symlink target exists**

```bash
ls /workspace/collab-splats/docs/pointcloud/localization.ipynb
```

Expected: file exists, no error.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats
git add docs/pointcloud/localization.ipynb docs/source/tutorials/pointcloud/localization.ipynb
git commit -m "docs(tutorials): add localization tutorial notebook"
```

---

## Verification Checklist

- [ ] `pytest tests/pointcloud/test_localization.py -v -m "not slow" --noconftest` — all pass
- [ ] `python -c "from collab_splats.pointcloud.localization import plot_correspondences, LocalizationResult; print('ok')"` — no import error
- [ ] `LocalizationResult` has `pts2d_ref` and `ref_frame_indices` fields with defaults `None`
- [ ] Notebook Cell 4 prints `pose: found` (not FAILED) when run on real C0043 data
- [ ] Notebook Cell 5 renders side-by-side with green/red lines
- [ ] Notebook Cell 6 renders grey reference frustums + red localized frustum
