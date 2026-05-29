# Localization Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache local feature extraction and localized frame poses in feedforward.zarr to eliminate O(N) GPU inference on session restart and enable a growing reference database for localization.

**Architecture:** Two zarr subgroups per extractor — `local_features/{extractor}/reconstruction/` (built once from FF pipeline) and `local_features/{extractor}/localized/` (appended after each successful localization). `CameraLocalizer` gains `save_index`, `load_index`, `update_index`, `add_localized_frame`, and `clear_localized_frames` methods. `from_feedforward` auto-detects cache and skips GPU on hit.

**Tech Stack:** zarr 3.1.5 (v3 API), numpy, torch, cv2. Python: `/opt/conda/envs/reconstruction/bin/python`. Specs: `docs/superpowers/specs/2026-05-29-localization-feature-cache.md` and `2026-05-29-localization-track-cache.md`.

---

## File Map

| File | Changes |
|---|---|
| `collab_splats/pointcloud/feedforward/base.py` | `FeedforwardResult._zarr_path` field + set in `load_zarr` |
| `collab_splats/pointcloud/localization.py` | `LocalizationResult.query_features`; `CameraLocalizer._frame_sources/_image_paths`; `save_index`, `load_index`, `update_index`, `add_localized_frame`, `clear_localized_frames`, `frame_sources`; extend `localize()` and `from_feedforward` |
| `collab_splats/dashboard/panes/localize.py` | Pass zarr_path to `from_feedforward`; call `add_localized_frame` after success |
| `tests/pointcloud/test_localization_cache.py` | New test file covering all cache operations |

---

## Task 1: Test infrastructure + FeedforwardResult._zarr_path

**Files:**
- Create: `tests/pointcloud/__init__.py`
- Create: `tests/pointcloud/test_localization_cache.py`
- Modify: `collab_splats/pointcloud/feedforward/base.py`

- [ ] **Step 1: Create tests/pointcloud/__init__.py**

```bash
mkdir -p /workspace/collab-splats/tests/pointcloud
touch /workspace/collab-splats/tests/pointcloud/__init__.py
```

- [ ] **Step 2: Write the failing test for _zarr_path**

Create `tests/pointcloud/test_localization_cache.py`:

```python
"""Tests for localization feature + track cache (zarr-backed)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import cv2
from pathlib import Path
from unittest.mock import MagicMock

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.localization import (
    CameraLocalizer,
    LocalFeatures,
    LocalizationResult,
)


# ── Shared fixtures ──────────────────────────────────────────────────────────

def _make_features(n_kpts: int = 10, desc_dim: int = 128) -> LocalFeatures:
    """Synthetic LocalFeatures for mocking — no GPU needed."""
    return LocalFeatures(
        keypoints=torch.rand(n_kpts, 2) * 60,
        descriptors=torch.rand(n_kpts, desc_dim),
        scores=None,
    )


def _make_scene(n_frames: int = 3, n_pts: int = 20):
    """Minimal reconstruction scene: front-facing cameras, random pts in front."""
    rng = np.random.default_rng(42)
    extrinsics = np.zeros((n_frames, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((n_frames, 3, 3), dtype=np.float32)
    for i in range(n_frames):
        extrinsics[i] = np.eye(4)
        extrinsics[i, 0, 3] = i * 0.5
        intrinsics[i] = np.array([[32, 0, 32], [0, 32, 32], [0, 0, 1]], dtype=np.float32)
    pts3d = rng.random((n_pts, 3)).astype(np.float32)
    pts3d[:, 2] += 2.0
    return pts3d, extrinsics, intrinsics


def _make_image_files(tmp_path: Path, n: int = 3, size: int = 64) -> list[Path]:
    """Write tiny solid-colour JPEG images to tmp_path."""
    paths = []
    for i in range(n):
        img = np.full((size, size, 3), fill_value=80 + i * 40, dtype=np.uint8)
        p = tmp_path / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths, desc_dim=128):
    """Build CameraLocalizer backed by a mock extractor (no GPU)."""
    mock_ext = MagicMock()
    mock_ext.extract.return_value = _make_features(desc_dim=desc_dim)
    mock_ext.match.return_value = torch.zeros((0, 2), dtype=torch.long)
    localizer = CameraLocalizer(
        pts3d=pts3d,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=image_paths,
        extractor=mock_ext,
    )
    return localizer, mock_ext


def _empty_zarr(tmp_path: Path) -> Path:
    """Create an empty feedforward.zarr store and return its path."""
    import zarr
    zarr_path = tmp_path / "feedforward.zarr"
    zarr.open(str(zarr_path), mode="w")
    return zarr_path


# ── Task 1 test ──────────────────────────────────────────────────────────────

def test_load_zarr_sets_zarr_path(tmp_path):
    """FeedforwardResult.load_zarr should set _zarr_path on the result."""
    pts3d, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    result = FeedforwardResult(
        points=pts3d,
        colors=np.zeros((len(pts3d), 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=image_paths,
        original_coords=np.zeros((3, 6), dtype=np.float32),
        model_width=64,
        model_height=64,
    )
    zarr_path = tmp_path / "test.zarr"
    result.save_zarr(zarr_path)

    loaded = FeedforwardResult.load_zarr(zarr_path)
    assert hasattr(loaded, "_zarr_path")
    assert loaded._zarr_path == zarr_path
```

- [ ] **Step 3: Run to verify it fails**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_load_zarr_sets_zarr_path -v 2>&1 | tail -20
```

Expected: `FAILED` — `AttributeError: 'FeedforwardResult' object has no attribute '_zarr_path'`

- [ ] **Step 4: Add `_zarr_path` to FeedforwardResult and set it in load_zarr**

In `collab_splats/pointcloud/feedforward/base.py`, add field to the dataclass after `pixel_indices`:

```python
    pixel_indices: "np.ndarray | None" = None  # (P, 3) int32 — [frame_id, row, col] source pixel for each point
    _zarr_path: "Path | None" = field(default=None, init=False, repr=False, compare=False)
```

Make sure `field` is already imported (it is — line 15: `from dataclasses import dataclass, field, replace`).

At the end of `load_zarr`, just before `return cls(...)`, the return statement should instead do:

```python
        result = cls(
            points=pts3d,
            colors=colors,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            original_coords=original_coords,
            image_paths=image_paths,
            model_width=model_width,
            model_height=model_height,
            features=features,
            pixel_indices=pixel_indices,
            world_points=world_points,
            depth=depth,
            images=images,
            confidence=confidence,
        )
        result._zarr_path = Path(path)
        return result
```

Replace the existing `return cls(...)` block (lines 212–227) with the above.

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_load_zarr_sets_zarr_path -v 2>&1 | tail -10
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/__init__.py tests/pointcloud/test_localization_cache.py && git commit -m "feat(feedforward): FeedforwardResult._zarr_path set in load_zarr"
```

---

## Task 2: Data model changes — LocalizationResult.query_features + CameraLocalizer state

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_localization_cache.py`:

```python
def test_localize_populates_query_features(tmp_path):
    """localize() should always set query_features on the result."""
    pts3d, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path, n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    query_img = np.zeros((64, 64, 3), dtype=np.uint8)
    query_K = intrinsics[0]
    result = localizer.localize(query_img, query_K)

    assert result.query_features is not None
    assert hasattr(result.query_features, "keypoints")
    assert hasattr(result.query_features, "descriptors")


def test_camera_localizer_stores_image_paths_and_sources(tmp_path):
    """CameraLocalizer should maintain _image_paths and _frame_sources after build."""
    pts3d, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path, n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    assert len(localizer._image_paths) == 3
    assert len(localizer._frame_sources) == 3
    assert all(s == "reconstruction" for s in localizer._frame_sources)
    assert localizer.frame_sources == ["reconstruction", "reconstruction", "reconstruction"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_localize_populates_query_features tests/pointcloud/test_localization_cache.py::test_camera_localizer_stores_image_paths_and_sources -v 2>&1 | tail -15
```

Expected: `FAILED` — `LocalizationResult` has no `query_features`; `CameraLocalizer` has no `_image_paths`.

- [ ] **Step 3: Add `query_features` to LocalizationResult**

In `localization.py`, the `LocalizationResult` dataclass (lines 62–83), add the new field at the end:

```python
    pts2d_ref: np.ndarray | None = None           # (M, 2) reference-frame pixel coords
    ref_frame_indices: np.ndarray | None = None   # (M,) int32 — source reference frame per correspondence
    query_features: "LocalFeatures | None" = None  # always set by localize(); pass to add_localized_frame
```

- [ ] **Step 4: Add `_image_paths` and `_frame_sources` to `CameraLocalizer.__init__`**

In `__init__`, after `self._extractor = ...` (around line 567), add:

```python
        # Store image paths and provenance for duplicate guard and dashboard display
        self._image_paths: list[Path] = [Path(p) for p in image_paths]
        self._frame_sources: list[str] = []
```

After the feature extraction loop (after line 618, where `self._assignments` is set), add:

```python
        self._frame_sources = ["reconstruction"] * len(self._frame_features)
```

- [ ] **Step 5: Add `frame_sources` property to CameraLocalizer**

After `__init__`, before `from_feedforward`:

```python
    @property
    def frame_sources(self) -> list[str]:
        """Provenance per frame: 'reconstruction' or 'localized'."""
        return list(self._frame_sources)
```

- [ ] **Step 6: Populate `query_features` in `localize()`**

In `localize()`, after `query_feats = self._extractor.extract(query_image)` (line 659), the variable is already computed. In the final return statements, add `query_features=query_feats` to each `LocalizationResult(...)` call.

Early-return (< 4 correspondences, line ~697):
```python
            return LocalizationResult(
                pose=None, n_correspondences=len(best_2d), n_inliers=0,
                pts2d=None, pts3d_matched=None, inlier_mask=None,
                pts2d_ref=None, ref_frame_indices=None,
                query_features=query_feats,
            )
```

PnP-fail return (line ~750):
```python
            return LocalizationResult(
                pose=None, n_correspondences=len(pts2d),
                n_inliers=ret["num_inliers"] if ret is not None else 0,
                pts2d=pts2d, pts3d_matched=pts3d_matched, inlier_mask=None,
                pts2d_ref=pts2d_ref, ref_frame_indices=ref_frame_indices,
                query_features=query_feats,
            )
```

Success return (line ~768):
```python
        return LocalizationResult(
            pose=pose, n_correspondences=len(pts2d), n_inliers=ret["num_inliers"],
            pts2d=pts2d, pts3d_matched=pts3d_matched, inlier_mask=inlier_mask,
            pts2d_ref=pts2d_ref, ref_frame_indices=ref_frame_indices,
            query_features=query_feats,
        )
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_localize_populates_query_features tests/pointcloud/test_localization_cache.py::test_camera_localizer_stores_image_paths_and_sources -v 2>&1 | tail -15
```

Expected: both `PASSED`

- [ ] **Step 8: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization_cache.py && git commit -m "feat(localization): LocalizationResult.query_features; CameraLocalizer._image_paths/_frame_sources"
```

---

## Task 3: save_index

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization_cache.py`

- [ ] **Step 1: Write failing test**

Append to `tests/pointcloud/test_localization_cache.py`:

```python
def test_save_index_creates_reconstruction_group(tmp_path):
    """save_index writes local_features/{name}/reconstruction/ with expected arrays."""
    import zarr
    pts3d, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/reconstruction" in store
    grp = store["local_features/disk/reconstruction"]
    assert "frame_offsets" in grp
    assert "keypoints" in grp
    assert "descriptors" in grp
    assert len(grp.attrs["image_paths"]) == 3
    assert grp["frame_offsets"].shape == (4,)   # N+1 = 3+1
    assert grp["keypoints"].shape[1] == 2
    assert grp["descriptors"].shape[0] == grp["keypoints"].shape[0]
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_save_index_creates_reconstruction_group -v 2>&1 | tail -10
```

Expected: `FAILED` — `AttributeError: 'CameraLocalizer' object has no attribute 'save_index'`

- [ ] **Step 3: Add zarr import to localization.py top-level imports**

After the existing imports (around line 43), add:

```python
import zarr
from zarr.codecs import BloscCodec
```

- [ ] **Step 4: Implement save_index**

Add after the `frame_sources` property (before `from_feedforward`):

```python
    def save_index(self, zarr_path: "str | Path", extractor_name: str) -> None:
        """Persist extracted frame features to feedforward.zarr reconstruction/ subgroup.

        Overwrites any existing reconstruction cache for extractor_name.
        Not automatically invalidated when source images change — caller's responsibility.
        Single-writer assumption; not safe for concurrent calls.
        """
        lz4 = BloscCodec(cname="lz4")
        zarr_path = pathlib.Path(zarr_path)
        store = zarr.open(str(zarr_path), mode="a")

        # Clean overwrite: delete existing reconstruction group if present
        rec_key = f"local_features/{extractor_name}/reconstruction"
        if rec_key in store:
            del store[rec_key]

        rec_group = store.require_group(rec_key)

        # Build CSR frame_offsets from per-frame keypoint counts
        counts = [len(f.keypoints) for f in self._frame_features]
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        # Concatenate all keypoints and descriptors across frames
        if offsets[-1] > 0:
            all_kpts = np.concatenate(
                [f.keypoints.numpy() for f in self._frame_features], axis=0
            ).astype(np.float32)
            all_descs = np.concatenate(
                [f.descriptors.numpy() for f in self._frame_features], axis=0
            ).astype(np.float32)
        else:
            d = self._frame_features[0].descriptors.shape[1] if self._frame_features else 1
            all_kpts = np.zeros((0, 2), dtype=np.float32)
            all_descs = np.zeros((0, d), dtype=np.float32)

        rec_group.attrs["image_paths"] = [str(p) for p in self._image_paths]
        rec_group.attrs["hw"] = list(self._image_hw)

        rec_group.create_array("frame_offsets", data=offsets,
                               chunks=offsets.shape, compressors=lz4)
        rec_group.create_array("keypoints", data=all_kpts,
                               chunks=(max(all_kpts.shape[0], 1), 2), compressors=lz4)
        rec_group.create_array("descriptors", data=all_descs,
                               chunks=(max(all_descs.shape[0], 1), all_descs.shape[1] if all_descs.shape[1] > 0 else 1),
                               compressors=lz4)

        # scores: XFeat only — skip if all None
        has_scores = any(f.scores is not None for f in self._frame_features)
        if has_scores:
            all_scores = np.concatenate([
                f.scores.numpy() if f.scores is not None
                else np.zeros(len(f.keypoints), dtype=np.float32)
                for f in self._frame_features
            ]).astype(np.float32)
            rec_group.create_array("scores", data=all_scores,
                                   chunks=(max(all_scores.shape[0], 1),), compressors=lz4)

        logger.info("CameraLocalizer.save_index: saved %d frames to %s [%s]",
                    len(self._frame_features), zarr_path, extractor_name)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_save_index_creates_reconstruction_group -v 2>&1 | tail -10
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization_cache.py && git commit -m "feat(localization): save_index writes reconstruction/ subgroup to zarr"
```

---

## Task 4: load_index

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization_cache.py`

- [ ] **Step 1: Write failing test**

Append to `tests/pointcloud/test_localization_cache.py`:

```python
def test_load_index_round_trip(tmp_path):
    """save_index + load_index: loaded localizer has same frame count and sources."""
    pts3d, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, mock_ext = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    loaded = CameraLocalizer.load_index(
        zarr_path=zarr_path,
        extractor_name="disk",
        pts3d=pts3d,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )

    assert len(loaded._frame_features) == 3
    assert len(loaded._assignments) == 3
    assert len(loaded._frame_sources) == 3
    assert all(s == "reconstruction" for s in loaded._frame_sources)
    assert loaded.frame_sources == ["reconstruction", "reconstruction", "reconstruction"]
    # Keypoint count preserved
    assert loaded._frame_features[0].keypoints.shape[1] == 2


def test_load_index_missing_extractor_raises(tmp_path):
    """load_index raises KeyError when extractor cache not found."""
    zarr_path = _empty_zarr(tmp_path)
    pts3d, extrinsics, intrinsics = _make_scene()
    with pytest.raises(KeyError, match="disk"):
        CameraLocalizer.load_index(
            zarr_path=zarr_path,
            extractor_name="disk",
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_load_index_round_trip tests/pointcloud/test_localization_cache.py::test_load_index_missing_extractor_raises -v 2>&1 | tail -10
```

Expected: `FAILED` — `AttributeError: type object 'CameraLocalizer' has no attribute 'load_index'`

- [ ] **Step 3: Implement load_index**

Add after `save_index` in `localization.py`:

```python
    @classmethod
    def load_index(
        cls,
        zarr_path: "str | Path",
        extractor_name: str,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        config: "dict | None" = None,
        extractor=None,
        radius: float = 8.0,
    ) -> "CameraLocalizer":
        """Load feature index from zarr; rebuild kpt→3D assignments from current geometry.

        Loads reconstruction/ and localized/ (if present) groups and merges them.
        Raises KeyError if extractor_name reconstruction cache not found.
        """
        zarr_path = pathlib.Path(zarr_path)
        store = zarr.open(str(zarr_path), mode="r")

        rec_key = f"local_features/{extractor_name}/reconstruction"
        if rec_key not in store:
            raise KeyError(
                f"No feature cache for extractor '{extractor_name}' in {zarr_path}. "
                "Rebuild via CameraLocalizer.from_feedforward()."
            )

        # ── Load reconstruction group ────────────────────────────────────────
        rec_group = store[rec_key]
        rec_image_paths = [pathlib.Path(p) for p in rec_group.attrs["image_paths"]]
        hw = tuple(int(x) for x in rec_group.attrs["hw"])
        offsets = rec_group["frame_offsets"][:]
        all_kpts = rec_group["keypoints"][:] if rec_group["keypoints"].shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
        all_descs = rec_group["descriptors"][:] if rec_group["descriptors"].shape[0] > 0 else np.zeros((0, 1), dtype=np.float32)
        all_scores = rec_group["scores"][:] if "scores" in rec_group else None

        rec_features: list[LocalFeatures] = []
        for i in range(len(offsets) - 1):
            s, e = int(offsets[i]), int(offsets[i + 1])
            f_kpts = torch.from_numpy(all_kpts[s:e])
            f_descs = torch.from_numpy(all_descs[s:e])
            f_scores = torch.from_numpy(all_scores[s:e]) if all_scores is not None else None
            rec_features.append(LocalFeatures(keypoints=f_kpts, descriptors=f_descs, scores=f_scores))

        # ── Load localized group (optional) ──────────────────────────────────
        loc_key = f"local_features/{extractor_name}/localized"
        loc_features: list[LocalFeatures] = []
        loc_image_paths: list[pathlib.Path] = []
        loc_extrinsics_list: list[np.ndarray] = []
        loc_intrinsics_list: list[np.ndarray] = []

        if loc_key in store:
            loc_group = store[loc_key]
            loc_image_paths = [pathlib.Path(p) for p in loc_group.attrs.get("image_paths", [])]
            if loc_image_paths:
                loc_offsets = loc_group["frame_offsets"][:]
                loc_kpts = loc_group["keypoints"][:]
                loc_descs = loc_group["descriptors"][:]
                loc_scores = loc_group["scores"][:] if "scores" in loc_group else None
                loc_ext = loc_group["extrinsics"][:]   # (N_loc, 4, 4)
                loc_intr = loc_group["intrinsics"][:]  # (N_loc, 3, 3)
                for i in range(len(loc_offsets) - 1):
                    s, e = int(loc_offsets[i]), int(loc_offsets[i + 1])
                    f_kpts = torch.from_numpy(loc_kpts[s:e])
                    f_descs = torch.from_numpy(loc_descs[s:e])
                    f_scores = torch.from_numpy(loc_scores[s:e]) if loc_scores is not None else None
                    loc_features.append(LocalFeatures(keypoints=f_kpts, descriptors=f_descs, scores=f_scores))
                    loc_extrinsics_list.append(loc_ext[i])
                    loc_intrinsics_list.append(loc_intr[i])

        # ── Build assignments ─────────────────────────────────────────────────
        rec_assignments = _build_frame_assignments(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_keypoints=[f.keypoints for f in rec_features],
            image_hw=hw,
            radius=radius,
        )

        if loc_features:
            loc_ext_arr = np.stack(loc_extrinsics_list, axis=0)
            loc_intr_arr = np.stack(loc_intrinsics_list, axis=0)
            loc_assignments = _build_frame_assignments(
                pts3d=pts3d,
                extrinsics=loc_ext_arr,
                intrinsics=loc_intr_arr,
                frame_keypoints=[f.keypoints for f in loc_features],
                image_hw=hw,
                radius=radius,
            )
        else:
            loc_assignments = []

        # ── Assemble object without running __init__ extraction loop ──────────
        obj = object.__new__(cls)
        obj.config = config or {}
        obj._pts3d = pts3d
        obj._extrinsics = extrinsics
        obj._intrinsics = intrinsics
        obj._extractor = extractor if extractor is not None else DiskExtractor()
        obj._image_hw = hw
        obj._frame_features = rec_features + loc_features
        obj._frame_sources = (["reconstruction"] * len(rec_features) +
                              ["localized"] * len(loc_features))
        obj._image_paths = rec_image_paths + loc_image_paths
        obj._assignments = rec_assignments + loc_assignments

        logger.info(
            "CameraLocalizer.load_index: loaded %d rec + %d loc frames from %s [%s]",
            len(rec_features), len(loc_features), zarr_path, extractor_name,
        )
        return obj
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_load_index_round_trip tests/pointcloud/test_localization_cache.py::test_load_index_missing_extractor_raises -v 2>&1 | tail -10
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization_cache.py && git commit -m "feat(localization): load_index loads reconstruction/ + localized/ groups from zarr"
```

---

## Task 5: from_feedforward cache integration

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization_cache.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_localization_cache.py`:

```python
def _make_ff_result(pts3d, extrinsics, intrinsics, image_paths):
    """Minimal FeedforwardResult for testing from_feedforward."""
    result = MagicMock()
    result.points = pts3d
    result.extrinsics = extrinsics
    result.intrinsics = intrinsics
    result.image_paths = image_paths
    result._zarr_path = None
    return result


def test_from_feedforward_cache_miss_builds_and_saves(tmp_path):
    """Cache miss: from_feedforward runs GPU inference and saves to zarr."""
    import zarr
    pts3d, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    zarr_path = _empty_zarr(tmp_path)

    mock_ext = MagicMock()
    mock_ext.extract.return_value = _make_features()
    mock_ext.match.return_value = torch.zeros((0, 2), dtype=torch.long)

    result = _make_ff_result(pts3d, extrinsics, intrinsics, image_paths)
    result._zarr_path = zarr_path

    localizer = CameraLocalizer.from_feedforward(result, extractor=mock_ext, extractor_name="disk")

    assert mock_ext.extract.call_count == 3  # GPU ran for each frame
    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/reconstruction" in store


def test_from_feedforward_cache_hit_skips_extraction(tmp_path):
    """Cache hit: from_feedforward loads from zarr, does not call extractor.extract."""
    pts3d, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    zarr_path = _empty_zarr(tmp_path)

    # Populate cache via first build
    build_ext = MagicMock()
    build_ext.extract.return_value = _make_features()
    build_ext.match.return_value = torch.zeros((0, 2), dtype=torch.long)
    result = _make_ff_result(pts3d, extrinsics, intrinsics, image_paths)
    CameraLocalizer.from_feedforward(result, extractor=build_ext, extractor_name="disk",
                                      zarr_path=zarr_path)

    # Second build — should hit cache
    load_ext = MagicMock()
    load_ext.extract.return_value = _make_features()
    loaded = CameraLocalizer.from_feedforward(result, extractor=load_ext, extractor_name="disk",
                                               zarr_path=zarr_path)

    assert load_ext.extract.call_count == 0  # no GPU inference on cache hit
    assert len(loaded._frame_features) == 3
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_from_feedforward_cache_miss_builds_and_saves tests/pointcloud/test_localization_cache.py::test_from_feedforward_cache_hit_skips_extraction -v 2>&1 | tail -15
```

Expected: `FAILED` — `from_feedforward` doesn't yet accept `zarr_path` or `extractor_name`.

- [ ] **Step 3: Replace from_feedforward in localization.py**

Replace the existing `from_feedforward` classmethod (lines 620–638):

```python
    @classmethod
    def from_feedforward(
        cls,
        result,
        extractor=None,
        progress_callback=None,
        zarr_path=None,
        extractor_name=None,
        **kwargs,
    ) -> "CameraLocalizer":
        """Construct from a FeedforwardResult. Loads from zarr cache if available.

        Args:
            result:         FeedforwardResult (or duck-typed object with .points,
                            .extrinsics, .intrinsics, .image_paths, ._zarr_path).
            extractor:      Local feature extractor; defaults to DiskExtractor().
            progress_callback: Called as (frame_idx, total) during index build.
            zarr_path:      Override zarr cache path; falls back to result._zarr_path.
            extractor_name: Override extractor registry key; auto-detected if None.
            **kwargs:       Forwarded to CameraLocalizer.__init__ (e.g. radius).

        Returns:
            CameraLocalizer ready to localize query images in the given scene.
        """
        extractor_inst = extractor if extractor is not None else DiskExtractor()

        # Determine extractor_name via registry reverse-lookup
        if extractor_name is None:
            extractor_name = next(
                (k for k, v in BaseLocalExtractor._registry.items()
                 if v is type(extractor_inst)),
                type(extractor_inst).__name__.lower().replace("extractor", ""),
            )

        # Resolve zarr_path: explicit arg > result._zarr_path
        if zarr_path is None:
            zarr_path = getattr(result, "_zarr_path", None)

        # Try cache first
        if zarr_path is not None:
            try:
                store = zarr.open(str(zarr_path), mode="r")
                rec_key = f"local_features/{extractor_name}/reconstruction"
                if rec_key in store:
                    # Staleness check: warn if image_paths differ
                    cached_paths = [pathlib.Path(p) for p in store[rec_key].attrs["image_paths"]]
                    if cached_paths != list(result.image_paths):
                        logger.warning(
                            "CameraLocalizer: cached image_paths differ from result — cache may be stale"
                        )
                    logger.info("CameraLocalizer: cache hit for '%s', loading from zarr", extractor_name)
                    return cls.load_index(
                        zarr_path=zarr_path,
                        extractor_name=extractor_name,
                        pts3d=result.points,
                        extrinsics=result.extrinsics,
                        intrinsics=result.intrinsics,
                        extractor=extractor_inst,
                        **{k: v for k, v in kwargs.items() if k in ("config", "radius")},
                    )
            except KeyError:
                logger.debug("CameraLocalizer: cache miss for '%s', building index", extractor_name)
            except Exception as exc:
                logger.warning("CameraLocalizer: cache load failed (%s), rebuilding", exc)

        # Cache miss — build from GPU inference
        localizer = cls(
            pts3d=result.points,
            extrinsics=result.extrinsics,
            intrinsics=result.intrinsics,
            image_paths=result.image_paths,
            extractor=extractor_inst,
            progress_callback=progress_callback,
            **kwargs,
        )

        # Save for next session
        if zarr_path is not None:
            try:
                localizer.save_index(zarr_path, extractor_name)
            except Exception as exc:
                logger.warning("CameraLocalizer: failed to save index to zarr: %s", exc)

        return localizer
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_from_feedforward_cache_miss_builds_and_saves tests/pointcloud/test_localization_cache.py::test_from_feedforward_cache_hit_skips_extraction -v 2>&1 | tail -10
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization_cache.py && git commit -m "feat(localization): from_feedforward tries zarr cache; build+save on miss"
```

---

## Task 6: update_index

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization_cache.py`

- [ ] **Step 1: Write failing test**

Append to `tests/pointcloud/test_localization_cache.py`:

```python
def test_update_index_appends_new_frames(tmp_path):
    """update_index extracts + appends new reconstruction frames to zarr."""
    import zarr
    pts3d, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    # Create 2 new frames
    new_paths = _make_image_files(tmp_path / "new_imgs", n=2)
    new_ext = MagicMock()
    new_ext.extract.return_value = _make_features()
    new_ext.match.return_value = torch.zeros((0, 2), dtype=torch.long)
    localizer._extractor = new_ext

    localizer.update_index(new_paths, zarr_path, "disk")

    assert new_ext.extract.call_count == 2
    assert len(localizer._frame_features) == 5  # 3 + 2
    assert localizer._frame_sources.count("reconstruction") == 5

    # Verify zarr updated
    store = zarr.open(str(zarr_path), mode="r")
    grp = store["local_features/disk/reconstruction"]
    assert grp["frame_offsets"].shape == (6,)  # 5+1
    assert len(grp.attrs["image_paths"]) == 5
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_update_index_appends_new_frames -v 2>&1 | tail -10
```

Expected: `FAILED` — no `update_index` method.

- [ ] **Step 3: Implement update_index**

Add after `load_index` in `localization.py`:

```python
    def update_index(
        self,
        new_image_paths: list,
        zarr_path: "str | Path",
        extractor_name: str,
        progress_callback: "Callable[[int, int], None] | None" = None,
    ) -> None:
        """Extract features for new reconstruction frames; append to zarr cache.

        Does NOT update pts3d/extrinsics/intrinsics — caller must update those
        and call clear_localized_frames() + load_index() to rebuild assignments.
        """
        zarr_path = pathlib.Path(zarr_path)
        new_features: list[LocalFeatures] = []

        for i, path in enumerate(new_image_paths):
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"CameraLocalizer.update_index: cannot read {path}")
            rgb = bgr[..., ::-1].copy()
            feats = self._extractor.extract(rgb)
            new_features.append(feats)
            if progress_callback is not None:
                progress_callback(i, len(new_image_paths))
            logger.debug("update_index: frame %s: %d kpts", path, len(feats.keypoints))

        # Update in-memory state
        for path, feats in zip(new_image_paths, new_features):
            self._frame_features.append(feats)
            self._frame_sources.append("reconstruction")
            self._image_paths.append(pathlib.Path(path))

        # Append to reconstruction/ zarr group
        store = zarr.open(str(zarr_path), mode="a")
        rec_key = f"local_features/{extractor_name}/reconstruction"

        if rec_key not in store:
            logger.warning("update_index: no existing reconstruction cache — building from scratch")
            self.save_index(zarr_path, extractor_name)
            return

        rec_group = store[rec_key]

        # Update attrs
        existing_paths = list(rec_group.attrs.get("image_paths", []))
        existing_paths.extend([str(p) for p in new_image_paths])
        rec_group.attrs["image_paths"] = existing_paths

        # Append CSR data frame by frame
        for feats in new_features:
            kpts_np = feats.keypoints.numpy().astype(np.float32)
            descs_np = feats.descriptors.numpy().astype(np.float32)

            off_arr = rec_group["frame_offsets"]
            last_off = int(off_arr[-1])
            n_off = off_arr.shape[0]
            off_arr.resize((n_off + 1,))
            off_arr[n_off] = last_off + len(kpts_np)

            kpts_arr = rec_group["keypoints"]
            old_m = kpts_arr.shape[0]
            kpts_arr.resize((old_m + len(kpts_np), kpts_arr.shape[1]))
            kpts_arr[old_m:] = kpts_np

            descs_arr = rec_group["descriptors"]
            descs_arr.resize((old_m + len(descs_np), descs_arr.shape[1]))
            descs_arr[old_m:] = descs_np

            if feats.scores is not None and "scores" in rec_group:
                scores_np = feats.scores.numpy().astype(np.float32)
                sc_arr = rec_group["scores"]
                sc_arr.resize((sc_arr.shape[0] + len(scores_np),))
                sc_arr[old_m:] = scores_np

        logger.info("CameraLocalizer.update_index: appended %d frames to %s [%s]",
                    len(new_image_paths), zarr_path, extractor_name)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py::test_update_index_appends_new_frames -v 2>&1 | tail -10
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization_cache.py && git commit -m "feat(localization): update_index appends new reconstruction frames to zarr"
```

---

## Task 7: add_localized_frame

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization_cache.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_localization_cache.py`:

```python
def test_add_localized_frame_extends_index_in_memory(tmp_path):
    """add_localized_frame appends a localized frame to in-memory reference set."""
    pts3d, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path, n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    assert len(localizer._frame_features) == 3

    new_pose = np.eye(4, dtype=np.float32)
    new_pose[0, 3] = 2.0
    new_intr = intrinsics[0].copy()
    new_feats = _make_features()
    new_path = tmp_path / "query.jpg"

    localizer.add_localized_frame(
        image_path=new_path,
        pose=new_pose,
        intrinsics=new_intr,
        features=new_feats,
    )

    assert len(localizer._frame_features) == 4
    assert localizer._frame_sources[-1] == "localized"
    assert localizer._image_paths[-1] == new_path


def test_add_localized_frame_persists_to_zarr(tmp_path):
    """add_localized_frame with zarr_path writes to localized/ group."""
    import zarr
    pts3d, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    new_pose = np.eye(4, dtype=np.float32)
    new_intr = intrinsics[0].copy()
    new_feats = _make_features()
    new_path = tmp_path / "query.jpg"

    localizer.add_localized_frame(
        image_path=new_path,
        pose=new_pose,
        intrinsics=new_intr,
        features=new_feats,
        zarr_path=zarr_path,
        extractor_name="disk",
    )

    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/localized" in store
    loc_grp = store["local_features/disk/localized"]
    assert loc_grp["extrinsics"].shape == (1, 4, 4)
    assert loc_grp["intrinsics"].shape == (1, 3, 3)
    assert len(loc_grp.attrs["image_paths"]) == 1


def test_add_localized_frame_duplicate_skipped(tmp_path):
    """Duplicate image_path is silently skipped — no double-add."""
    pts3d, extrinsics, intrinsics = _make_scene(n_frames=2)
    image_paths = _make_image_files(tmp_path, n=2)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    new_pose = np.eye(4, dtype=np.float32)
    new_path = tmp_path / "query.jpg"
    new_feats = _make_features()

    localizer.add_localized_frame(new_path, new_pose, intrinsics[0], new_feats)
    localizer.add_localized_frame(new_path, new_pose, intrinsics[0], new_feats)  # duplicate

    assert len(localizer._frame_features) == 3  # 2 rec + 1 loc, not 4


def test_load_index_includes_localized_frames(tmp_path):
    """After add + reload, localized frame is in the loaded index."""
    pts3d, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    new_pose = np.eye(4, dtype=np.float32)
    new_pose[0, 3] = 1.5
    new_path = tmp_path / "query.jpg"
    localizer.add_localized_frame(new_path, new_pose, intrinsics[0], _make_features(),
                                   zarr_path=zarr_path, extractor_name="disk")

    # Reload from zarr
    loaded = CameraLocalizer.load_index(
        zarr_path=zarr_path, extractor_name="disk",
        pts3d=pts3d, extrinsics=extrinsics, intrinsics=intrinsics,
    )

    assert len(loaded._frame_features) == 4
    assert loaded._frame_sources == ["reconstruction"] * 3 + ["localized"]
    assert loaded._image_paths[-1] == new_path
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py -k "localized_frame" -v 2>&1 | tail -15
```

Expected: all `FAILED` — no `add_localized_frame` method.

- [ ] **Step 3: Implement add_localized_frame**

Add after `update_index` in `localization.py`:

```python
    def add_localized_frame(
        self,
        image_path: "str | Path",
        pose: np.ndarray,
        intrinsics: np.ndarray,
        features: LocalFeatures,
        zarr_path: "str | Path | None" = None,
        extractor_name: str | None = None,
    ) -> None:
        """Add a successfully localized frame to the in-memory reference set.

        Rebuilds kpt→3D assignments for the new frame from existing pts3d + given pose.
        If zarr_path and extractor_name are provided, appends to localized/ in zarr.
        Single-writer; not thread-safe across concurrent callers.
        Call clear_localized_frames() after BA/LC updates that invalidate poses.
        """
        image_path = pathlib.Path(image_path)

        # Duplicate guard
        if image_path in self._image_paths:
            logger.warning(
                "CameraLocalizer.add_localized_frame: %s already in index, skipping",
                image_path.name,
            )
            return

        # Build kpt→3D assignment for this frame using existing pts3d
        new_assignments = _build_frame_assignments(
            pts3d=self._pts3d,
            extrinsics=pose[np.newaxis],       # (1, 4, 4)
            intrinsics=intrinsics[np.newaxis],  # (1, 3, 3)
            frame_keypoints=[features.keypoints],
            image_hw=self._image_hw,
        )

        # Append in-memory
        self._frame_features.append(features)
        self._frame_sources.append("localized")
        self._image_paths.append(image_path)
        self._assignments.extend(new_assignments)

        # Persist to zarr if requested
        if zarr_path is not None and extractor_name is not None:
            self._append_localized_to_zarr(
                image_path, pose, intrinsics, features,
                pathlib.Path(zarr_path), extractor_name,
            )

    def _append_localized_to_zarr(
        self,
        image_path: pathlib.Path,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        features: LocalFeatures,
        zarr_path: pathlib.Path,
        extractor_name: str,
    ) -> None:
        """Append one localized frame to the localized/ zarr group."""
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(zarr_path), mode="a")
        loc_key = f"local_features/{extractor_name}/localized"

        kpts_np = features.keypoints.numpy().astype(np.float32)
        descs_np = features.descriptors.numpy().astype(np.float32)
        scores_np = (features.scores.numpy().astype(np.float32)
                     if features.scores is not None else None)

        if loc_key not in store:
            # First localized frame — create group + arrays
            loc_group = store.require_group(loc_key)
            offsets = np.array([0, len(kpts_np)], dtype=np.int64)
            loc_group.attrs["image_paths"] = [str(image_path)]
            loc_group.create_array("frame_offsets", data=offsets,
                                   chunks=(max(offsets.shape[0], 2),), compressors=lz4)
            loc_group.create_array("keypoints", data=kpts_np,
                                   chunks=(max(kpts_np.shape[0], 1), 2), compressors=lz4)
            loc_group.create_array("descriptors", data=descs_np,
                                   chunks=(max(descs_np.shape[0], 1),
                                           max(descs_np.shape[1], 1)), compressors=lz4)
            if scores_np is not None:
                loc_group.create_array("scores", data=scores_np,
                                       chunks=(max(scores_np.shape[0], 1),), compressors=lz4)
            loc_group.create_array("extrinsics", data=pose[np.newaxis],
                                   chunks=(1, 4, 4), compressors=lz4)
            loc_group.create_array("intrinsics", data=intrinsics[np.newaxis],
                                   chunks=(1, 3, 3), compressors=lz4)
        else:
            # Append to existing group
            loc_group = store[loc_key]
            existing = list(loc_group.attrs.get("image_paths", []))
            existing.append(str(image_path))
            loc_group.attrs["image_paths"] = existing

            off_arr = loc_group["frame_offsets"]
            last_off = int(off_arr[-1])
            n_off = off_arr.shape[0]
            off_arr.resize((n_off + 1,))
            off_arr[n_off] = last_off + len(kpts_np)

            kpts_arr = loc_group["keypoints"]
            old_m = kpts_arr.shape[0]
            kpts_arr.resize((old_m + len(kpts_np), kpts_arr.shape[1]))
            kpts_arr[old_m:] = kpts_np

            descs_arr = loc_group["descriptors"]
            descs_arr.resize((old_m + len(descs_np), descs_arr.shape[1]))
            descs_arr[old_m:] = descs_np

            if scores_np is not None and "scores" in loc_group:
                sc_arr = loc_group["scores"]
                sc_arr.resize((sc_arr.shape[0] + len(scores_np),))
                sc_arr[old_m:] = scores_np

            ext_arr = loc_group["extrinsics"]
            n_loc = ext_arr.shape[0]
            ext_arr.resize((n_loc + 1, 4, 4))
            ext_arr[n_loc] = pose

            intr_arr = loc_group["intrinsics"]
            intr_arr.resize((n_loc + 1, 3, 3))
            intr_arr[n_loc] = intrinsics

        logger.debug("CameraLocalizer: appended localized frame %s to zarr", image_path.name)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py -k "localized_frame or includes_localized" -v 2>&1 | tail -15
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization_cache.py && git commit -m "feat(localization): add_localized_frame appends to in-memory index and zarr"
```

---

## Task 8: clear_localized_frames

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization_cache.py`

- [ ] **Step 1: Write failing test**

Append to `tests/pointcloud/test_localization_cache.py`:

```python
def test_clear_localized_frames_removes_zarr_group(tmp_path):
    """clear_localized_frames deletes localized/ group; reconstruction/ untouched."""
    import zarr
    pts3d, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    # Add a localized frame
    localizer.add_localized_frame(
        tmp_path / "q.jpg", np.eye(4, dtype=np.float32),
        intrinsics[0], _make_features(),
        zarr_path=zarr_path, extractor_name="disk",
    )

    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/localized" in store

    # Clear
    CameraLocalizer.clear_localized_frames(zarr_path, "disk")

    store2 = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/localized" not in store2
    assert "local_features/disk/reconstruction" in store2  # untouched


def test_load_index_after_clear_has_only_reconstruction(tmp_path):
    """After clear_localized_frames, load_index returns only reconstruction frames."""
    pts3d, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(pts3d, extrinsics, intrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")
    localizer.add_localized_frame(
        tmp_path / "q.jpg", np.eye(4, dtype=np.float32),
        intrinsics[0], _make_features(),
        zarr_path=zarr_path, extractor_name="disk",
    )

    CameraLocalizer.clear_localized_frames(zarr_path, "disk")

    loaded = CameraLocalizer.load_index(
        zarr_path=zarr_path, extractor_name="disk",
        pts3d=pts3d, extrinsics=extrinsics, intrinsics=intrinsics,
    )
    assert len(loaded._frame_features) == 3
    assert all(s == "reconstruction" for s in loaded._frame_sources)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py -k "clear" -v 2>&1 | tail -10
```

Expected: `FAILED` — no `clear_localized_frames`.

- [ ] **Step 3: Implement clear_localized_frames**

Add after `_append_localized_to_zarr` in `localization.py`:

```python
    @staticmethod
    def clear_localized_frames(zarr_path: "str | Path", extractor_name: str) -> None:
        """Delete the localized/ group for extractor_name from feedforward.zarr.

        Reconstruction data is untouched. Call this after BA/LC updates that
        invalidate previously estimated localized poses, then reload via load_index().
        """
        store = zarr.open(str(pathlib.Path(zarr_path)), mode="a")
        loc_key = f"local_features/{extractor_name}/localized"
        if loc_key in store:
            del store[loc_key]
            logger.info(
                "CameraLocalizer.clear_localized_frames: cleared '%s' from %s",
                extractor_name, zarr_path,
            )
        else:
            logger.debug(
                "CameraLocalizer.clear_localized_frames: no localized group for '%s'",
                extractor_name,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py -k "clear" -v 2>&1 | tail -10
```

Expected: both `PASSED`

- [ ] **Step 5: Run full test file to confirm no regressions**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_localization_cache.py -v 2>&1 | tail -25
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization_cache.py && git commit -m "feat(localization): clear_localized_frames; full cache test suite passing"
```

---

## Task 9: Dashboard integration

**Files:**
- Modify: `collab_splats/dashboard/panes/localize.py`

- [ ] **Step 1: Update `_build_localizer` to pass zarr_path**

In `localize.py`, replace `_build_localizer` (lines 373–387):

```python
    def _build_localizer(
        self,
        progress_callback=None,
    ) -> tuple["FeedforwardResult", "CameraLocalizer"]:
        """Load feedforward.zarr and build CameraLocalizer with optional frame progress."""
        output_dir = Path(self._state.output_dir)
        method = self._state.localize_method
        extractor_label = self._state.localize_extractor
        zarr_path = output_dir / method / "feedforward.zarr"
        ff = FeedforwardResult.load_zarr(zarr_path)
        extractor_cls = _EXTRACTOR_CLASSES.get(extractor_label, DiskExtractor)
        extractor = extractor_cls()
        localizer = CameraLocalizer.from_feedforward(
            ff,
            extractor=extractor,
            progress_callback=progress_callback,
            zarr_path=zarr_path,
        )
        return ff, localizer
```

Also add `self._extractor_name` storage so `add_localized_frame` can pass the correct key. Add to `__init__` after `self._localizer = None`:

```python
        self._extractor_name: str | None = None  # set in _build_localizer
```

Update `_build_localizer` to set it (add after `extractor = extractor_cls()`):

```python
        # Determine registry key for zarr group name
        self._extractor_name = next(
            (k for k, v in _EXTRACTOR_CLASSES.items() if v is extractor_cls),
            extractor_label,
        ).lower().replace("+", "").split("+")[0].lower()
        # Simplify: "disk+lightglue" -> "disk", "xfeat+mnn" -> "xfeat"
        self._extractor_name = "disk" if "disk" in extractor_label.lower() else "xfeat"
```

Wait, that's fragile. Use the registry reverse-lookup via `BaseLocalExtractor`:

```python
        from collab_splats.pointcloud.localization import BaseLocalExtractor
        self._extractor_name = next(
            (k for k, v in BaseLocalExtractor._registry.items() if v is extractor_cls),
            extractor_label.lower(),
        )
```

- [ ] **Step 2: Call add_localized_frame after successful single-image localization**

In `_run_localize`, after `# Highlight cameras in 3D` block (around line 497), add before the `except` block:

```python
                # Add to growing reference database
                output_dir = Path(self._state.output_dir)
                method = self._state.localize_method
                zarr_path = output_dir / method / "feedforward.zarr"
                if self._extractor_name and result.query_features is not None:
                    try:
                        localizer.add_localized_frame(
                            image_path=query_path,
                            pose=loc.pose,
                            intrinsics=query_intrinsics,
                            features=loc.query_features,
                            zarr_path=zarr_path,
                            extractor_name=self._extractor_name,
                        )
                        logger.debug("LocalizePane: added localized frame to reference set")
                    except Exception as exc:
                        logger.warning("LocalizePane: add_localized_frame failed: %s", exc)
```

Note: `result` is named `loc` in `_run_localize`. Use `loc.query_features`.

- [ ] **Step 3: Call add_localized_frame in batch mode**

In `_run_batch`, inside the `else` branch after `rows.append({...status: "✓"...})` (around line 591), add:

```python
                    # Add to growing reference database
                    output_dir = Path(self._state.output_dir)
                    method = self._state.localize_method
                    zarr_path_batch = output_dir / method / "feedforward.zarr"
                    if self._extractor_name and loc.query_features is not None:
                        try:
                            localizer.add_localized_frame(
                                image_path=qp,
                                pose=loc.pose,
                                intrinsics=query_intrinsics,
                                features=loc.query_features,
                                zarr_path=zarr_path_batch,
                                extractor_name=self._extractor_name,
                            )
                        except Exception as exc:
                            logger.debug("batch add_localized_frame failed: %s", exc)
```

- [ ] **Step 4: Run existing dashboard smoke tests to catch regressions**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/ -v 2>&1 | tail -20
```

Expected: all existing dashboard tests still `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/dashboard/panes/localize.py && git commit -m "feat(dashboard): localize pane uses zarr cache; add_localized_frame after success"
```

---

## Self-Review Checklist

- [x] `FeedforwardResult._zarr_path` set in `load_zarr` — Task 1
- [x] `LocalizationResult.query_features` — Task 2
- [x] `_frame_sources`, `_image_paths`, `frame_sources` property — Task 2
- [x] `localize()` populates `query_features` on all return paths — Task 2
- [x] `save_index` writes `reconstruction/` subgroup — Task 3
- [x] `load_index` merges `reconstruction/` + `localized/` — Task 4
- [x] `from_feedforward` tries cache first, saves on miss — Task 5
- [x] Staleness warning logged when paths differ — Task 5 (in from_feedforward)
- [x] `update_index` appends to reconstruction/ — Task 6
- [x] `add_localized_frame` in-memory + zarr — Task 7
- [x] Duplicate guard — Task 7
- [x] `clear_localized_frames` static method — Task 8
- [x] Dashboard passes zarr_path; calls add_localized_frame — Task 9
