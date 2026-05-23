# Notebook Reorganization & Zarr Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate tutorial notebooks from symlinks into a pipeline-stage taxonomy, wire a consistent zarr-based cross-notebook cache, and add `FeedforwardResult` zarr I/O.

**Architecture:** Single canonical notebook location at `docs/source/tutorials/{stage}/`. Cache lives at `docs/.cache/{dataset}/{method}/{variant}/reconstruction.zarr`, mirroring the existing `evals/results/{dataset}/{variant}/` convention. Each notebook reads from the previous stage's zarr output via two config variables (`METHOD`, `VARIANT`).

**Tech Stack:** Python 3.11, zarr>=2.16, numcodecs, numpy, `collab_splats.pointcloud.feedforward.base.FeedforwardResult`, nbsphinx (sphinx build), pyproject.toml deps.

**Spec:** `docs/superpowers/specs/2026-05-23-notebook-reorganization-design.md`

**Python binary:** Always use `/opt/conda/envs/nerfstudio/bin/python`

---

## File Map

**Modified:**
- `pyproject.toml` — add `zarr>=2.16`, `numcodecs`
- `collab_splats/pointcloud/feedforward/base.py` — add `save_zarr()`, `load_zarr()`
- `docs/source/tutorials/index.rst` — rewrite toctree for new structure
- `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` — save `images/` to cache
- `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` — standardize cache config
- `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb` — standardize cache config
- `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb` — standardize cache config
- `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` — load reconstruction from cache
- `docs/source/tutorials/evals/ground_truth_evals.ipynb` — update cache paths

**Created:**
- `tests/pointcloud/test_feedforward_zarr.py` — zarr roundtrip tests
- `docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb` — stub notebook

**Moved (git mv):**
- All notebooks from `docs/{pointcloud,semantics,splats}/` → `docs/source/tutorials/`
- Symlinks in `docs/source/tutorials/` deleted and replaced with real files
- Old directories `docs/pointcloud/`, `docs/semantics/`, `docs/splats/` deleted

**Renamed during move:**
- `feedforward_exploration.ipynb` → `feedforward_methods.ipynb`
- `loop_closure_eval.ipynb` → `slam_loop_closure.ipynb`
- `maskclip_reference_comparison.ipynb` → `maskclip_vs_talk2dino.ipynb`

---

## Task 1: Add zarr dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add zarr + numcodecs to pyproject.toml**

Open `pyproject.toml`, find `dependencies = [` block, add after `"numpy>=1.26"`:
```toml
    "zarr>=2.16",
    "numcodecs",
```

- [ ] **Step 2: Install**

```bash
/opt/conda/envs/nerfstudio/bin/pip install "zarr>=2.16" numcodecs
```

Expected: installs cleanly, no conflicts.

- [ ] **Step 3: Verify**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import zarr, numcodecs; print(zarr.__version__)"
```

Expected: prints version like `2.18.x`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(deps): add zarr and numcodecs for compressed array cache"
```

---

## Task 2: FeedforwardResult zarr backend

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py` (after line 93 — after existing `load()`)
- Create: `tests/pointcloud/test_feedforward_zarr.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pointcloud/test_feedforward_zarr.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _make_result(n=4, p=100, with_world_points=False):
    r = FeedforwardResult(
        pts3d=np.random.randn(p, 3).astype(np.float32),
        colors=np.random.randint(0, 255, (p, 3), dtype=np.uint8),
        extrinsics=np.eye(4, dtype=np.float32)[None].repeat(n, 0),
        intrinsics=np.eye(3, dtype=np.float32)[None].repeat(n, 0),
        image_paths=[Path(f"frame_{i:04d}.jpg") for i in range(n)],
        original_coords=np.zeros((n, 6), dtype=np.float32),
        model_width=224,
        model_height=224,
        pixel_indices=np.zeros((p, 3), dtype=np.int32),
    )
    if with_world_points:
        r.world_points = np.random.randn(n, 8, 8, 3).astype(np.float32)
    return r


def test_zarr_roundtrip_core_fields(tmp_path):
    result = _make_result()
    zarr_path = tmp_path / "result.zarr"
    result.save_zarr(zarr_path)
    loaded = FeedforwardResult.load_zarr(zarr_path)

    np.testing.assert_array_equal(result.pts3d, loaded.pts3d)
    np.testing.assert_array_equal(result.colors, loaded.colors)
    np.testing.assert_array_equal(result.extrinsics, loaded.extrinsics)
    np.testing.assert_array_equal(result.intrinsics, loaded.intrinsics)
    np.testing.assert_array_equal(result.original_coords, loaded.original_coords)
    np.testing.assert_array_equal(result.pixel_indices, loaded.pixel_indices)
    assert loaded.image_paths == result.image_paths
    assert loaded.model_width == result.model_width
    assert loaded.model_height == result.model_height


def test_zarr_includes_world_points(tmp_path):
    result = _make_result(with_world_points=True)
    zarr_path = tmp_path / "result_wp.zarr"
    result.save_zarr(zarr_path)
    loaded = FeedforwardResult.load_zarr(zarr_path)

    assert loaded.world_points is not None
    np.testing.assert_array_almost_equal(result.world_points, loaded.world_points)


def test_zarr_world_points_chunked_by_frame(tmp_path):
    import zarr
    result = _make_result(n=3, with_world_points=True)
    zarr_path = tmp_path / "result_chunks.zarr"
    result.save_zarr(zarr_path)

    store = zarr.open(str(zarr_path), mode="r")
    # Each chunk covers exactly one frame
    assert store["world_points"].chunks[0] == 1


def test_zarr_missing_optional_fields_load_as_none(tmp_path):
    result = _make_result()
    result.pixel_indices = None
    zarr_path = tmp_path / "result_opt.zarr"
    result.save_zarr(zarr_path)
    loaded = FeedforwardResult.load_zarr(zarr_path)

    assert loaded.pixel_indices is None
    assert loaded.world_points is None
    assert loaded.images is None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_zarr.py -v
```

Expected: `AttributeError: 'FeedforwardResult' object has no attribute 'save_zarr'`

- [ ] **Step 3: Add save_zarr and load_zarr to FeedforwardResult**

In `collab_splats/pointcloud/feedforward/base.py`, after the `load()` classmethod (after line 93), add:

```python
    def save_zarr(self, path: Path) -> None:
        """Save to zarr store with lz4 compression. Includes world_points and images chunked by frame."""
        import zarr
        import numcodecs

        path = Path(path)
        store = zarr.open(str(path), mode="w")
        compressor = numcodecs.Blosc(cname="lz4", clevel=5)

        store.attrs["image_paths"] = [str(p) for p in self.image_paths]
        store.attrs["model_width"] = self.model_width
        store.attrs["model_height"] = self.model_height

        for name, arr in [
            ("pts3d", self.pts3d),
            ("colors", self.colors),
            ("extrinsics", self.extrinsics),
            ("intrinsics", self.intrinsics),
            ("original_coords", self.original_coords),
        ]:
            store.create_dataset(name, data=arr, compressor=compressor)

        for name, arr in [("features", self.features), ("pixel_indices", self.pixel_indices)]:
            if arr is not None:
                store.create_dataset(name, data=arr, compressor=compressor)

        if self.world_points is not None:
            wp = self.world_points if isinstance(self.world_points, np.ndarray) else self.world_points.numpy()
            store.create_dataset(
                "world_points", data=wp,
                chunks=(1,) + wp.shape[1:], compressor=compressor,
            )

        if self.images is not None:
            imgs = self.images.numpy() if hasattr(self.images, "numpy") else np.array(self.images)
            store.create_dataset(
                "images", data=imgs,
                chunks=(1,) + imgs.shape[1:], compressor=compressor,
            )

    @classmethod
    def load_zarr(cls, path: Path) -> "FeedforwardResult":
        """Load from zarr store. images kept None (too large); world_points loaded if present."""
        import zarr

        store = zarr.open(str(Path(path)), mode="r")

        def _opt(name):
            return store[name][:] if name in store else None

        return cls(
            pts3d=store["pts3d"][:],
            colors=store["colors"][:],
            extrinsics=store["extrinsics"][:],
            intrinsics=store["intrinsics"][:],
            original_coords=store["original_coords"][:],
            image_paths=[Path(p) for p in store.attrs["image_paths"]],
            model_width=int(store.attrs["model_width"]),
            model_height=int(store.attrs["model_height"]),
            features=_opt("features"),
            pixel_indices=_opt("pixel_indices"),
            world_points=_opt("world_points"),
        )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_zarr.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run full pointcloud test suite — no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v --tb=short
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_feedforward_zarr.py
git commit -m "feat(feedforward): add FeedforwardResult.save_zarr / load_zarr with lz4 compression"
```

---

## Task 3: Remove symlinks — consolidate notebooks

**Files:**
- Delete: 14 symlinks in `docs/source/tutorials/{pointcloud,semantics,splats}/`
- Move: notebooks from `docs/{pointcloud,semantics,splats}/` to `docs/source/tutorials/`
- Delete: `docs/pointcloud/`, `docs/semantics/`, `docs/splats/` directories

- [ ] **Step 1: Delete symlinks**

```bash
find /workspace/collab-splats/docs/source/tutorials -type l -delete
```

- [ ] **Step 2: Move pointcloud notebooks**

```bash
cd /workspace/collab-splats
git mv docs/pointcloud/feedforward_exploration.ipynb docs/source/tutorials/pointcloud/feedforward_exploration.ipynb
git mv docs/pointcloud/bundle_adjustment.ipynb docs/source/tutorials/pointcloud/bundle_adjustment.ipynb
git mv docs/pointcloud/loop_closure_eval.ipynb docs/source/tutorials/pointcloud/loop_closure_eval.ipynb
git mv docs/pointcloud/feedforward_mesh.ipynb docs/source/tutorials/pointcloud/feedforward_mesh.ipynb
git mv docs/pointcloud/localization.ipynb docs/source/tutorials/pointcloud/localization.ipynb
git mv docs/pointcloud/ground-truth-evals.ipynb docs/source/tutorials/pointcloud/ground-truth-evals.ipynb
```

- [ ] **Step 3: Move semantics notebooks**

```bash
git mv docs/semantics/feature_extraction.ipynb docs/source/tutorials/semantics/feature_extraction.ipynb
git mv docs/semantics/segmentation.ipynb docs/source/tutorials/semantics/segmentation.ipynb
git mv docs/semantics/maskclip_reference_comparison.ipynb docs/source/tutorials/semantics/maskclip_reference_comparison.ipynb
git mv docs/semantics/semantic_lifting.ipynb docs/source/tutorials/semantics/semantic_lifting.ipynb
```

- [ ] **Step 4: Move splats notebooks**

```bash
git mv docs/splats/derive_splats.ipynb docs/source/tutorials/splats/derive_splats.ipynb
git mv docs/splats/create_mesh.ipynb docs/source/tutorials/splats/create_mesh.ipynb
git mv docs/splats/visualization.ipynb docs/source/tutorials/splats/visualization.ipynb
git mv docs/splats/compare_maskclip_talk2dino.ipynb docs/source/tutorials/splats/compare_maskclip_talk2dino.ipynb
```

- [ ] **Step 5: Remove now-empty source dirs**

```bash
git rm -rf docs/pointcloud docs/semantics docs/splats
```

- [ ] **Step 6: Verify notebooks present**

```bash
find /workspace/collab-splats/docs/source/tutorials -name "*.ipynb" | sort
```

Expected: 15 notebooks, all real files (no symlinks).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(docs): remove symlinks, consolidate notebooks to docs/source/tutorials/"
```

---

## Task 4: Restructure into numbered pipeline stage folders

**Files:**
- Create dirs: `01_preprocessing/`, `02_pointcloud/`, `03_splats/`, `04_semantics/`, `05_lifting/`, `06_mesh/`, `07_localization/`, `evals/`
- Move + rename notebooks into staged folders

- [ ] **Step 1: Create new directories**

```bash
cd /workspace/collab-splats/docs/source/tutorials
mkdir -p 01_preprocessing 02_pointcloud 03_splats 04_semantics 05_lifting 06_mesh 07_localization evals
```

- [ ] **Step 2: Move preprocessing notebook**

```bash
cd /workspace/collab-splats
git mv docs/source/tutorials/preprocessing/keyframe_extraction.ipynb \
        docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
rmdir docs/source/tutorials/preprocessing
```

- [ ] **Step 3: Move + rename pointcloud notebooks**

```bash
cd /workspace/collab-splats
git mv docs/source/tutorials/pointcloud/feedforward_exploration.ipynb \
        docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git mv docs/source/tutorials/pointcloud/bundle_adjustment.ipynb \
        docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
git mv docs/source/tutorials/pointcloud/loop_closure_eval.ipynb \
        docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
git mv docs/source/tutorials/pointcloud/feedforward_mesh.ipynb \
        docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb
git mv docs/source/tutorials/pointcloud/localization.ipynb \
        docs/source/tutorials/07_localization/localization.ipynb
git mv docs/source/tutorials/pointcloud/ground-truth-evals.ipynb \
        docs/source/tutorials/evals/ground_truth_evals.ipynb
rmdir docs/source/tutorials/pointcloud
```

- [ ] **Step 4: Move splats notebooks**

```bash
cd /workspace/collab-splats
git mv docs/source/tutorials/splats/derive_splats.ipynb \
        docs/source/tutorials/03_splats/derive_splats.ipynb
git mv docs/source/tutorials/splats/create_mesh.ipynb \
        docs/source/tutorials/06_mesh/create_mesh.ipynb
git mv docs/source/tutorials/splats/visualization.ipynb \
        docs/source/tutorials/03_splats/visualization.ipynb
git mv docs/source/tutorials/splats/compare_maskclip_talk2dino.ipynb \
        docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb
rmdir docs/source/tutorials/splats
```

- [ ] **Step 5: Move semantics notebooks**

```bash
cd /workspace/collab-splats
git mv docs/source/tutorials/semantics/feature_extraction.ipynb \
        docs/source/tutorials/04_semantics/feature_extraction.ipynb
git mv docs/source/tutorials/semantics/segmentation.ipynb \
        docs/source/tutorials/04_semantics/segmentation.ipynb
git mv docs/source/tutorials/semantics/maskclip_reference_comparison.ipynb \
        docs/source/tutorials/04_semantics/maskclip_reference_comparison_old.ipynb
git mv docs/source/tutorials/semantics/semantic_lifting.ipynb \
        docs/source/tutorials/05_lifting/semantic_lifting.ipynb
rmdir docs/source/tutorials/semantics
```

Note: `maskclip_reference_comparison_old.ipynb` is the old notebook whose content is now superseded by `maskclip_vs_talk2dino.ipynb`. Delete it after verifying content is captured in the renamed version.

- [ ] **Step 6: Verify final structure**

```bash
find /workspace/collab-splats/docs/source/tutorials -name "*.ipynb" | sort
```

Expected output:
```
docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb
docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
docs/source/tutorials/03_splats/derive_splats.ipynb
docs/source/tutorials/03_splats/visualization.ipynb
docs/source/tutorials/04_semantics/feature_extraction.ipynb
docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb
docs/source/tutorials/04_semantics/segmentation.ipynb
docs/source/tutorials/05_lifting/semantic_lifting.ipynb
docs/source/tutorials/06_mesh/create_mesh.ipynb
docs/source/tutorials/07_localization/localization.ipynb
docs/source/tutorials/evals/ground_truth_evals.ipynb
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(docs): restructure tutorials into numbered pipeline stage folders"
```

---

## Task 5: Update toctree

**Files:**
- Modify: `docs/source/tutorials/index.rst`

- [ ] **Step 1: Rewrite index.rst**

Replace entire contents of `docs/source/tutorials/index.rst` with:

```rst
Tutorials
=========

.. toctree::
   :maxdepth: 1
   :caption: 01 · Preprocessing

   01_preprocessing/keyframe_extraction

.. toctree::
   :maxdepth: 1
   :caption: 02 · Pointcloud

   02_pointcloud/feedforward_methods
   02_pointcloud/bundle_adjustment
   02_pointcloud/slam_loop_closure
   02_pointcloud/feedforward_mesh
   02_pointcloud/colmap_sfm

.. toctree::
   :maxdepth: 1
   :caption: 03 · Splats

   03_splats/derive_splats
   03_splats/visualization

.. toctree::
   :maxdepth: 1
   :caption: 04 · Semantics

   04_semantics/feature_extraction
   04_semantics/segmentation
   04_semantics/maskclip_vs_talk2dino

.. toctree::
   :maxdepth: 1
   :caption: 05 · Lifting

   05_lifting/semantic_lifting

.. toctree::
   :maxdepth: 1
   :caption: 06 · Mesh

   06_mesh/create_mesh

.. toctree::
   :maxdepth: 1
   :caption: 07 · Localization

   07_localization/localization

.. toctree::
   :maxdepth: 1
   :caption: Evaluation

   evals/ground_truth_evals
```

- [ ] **Step 2: Commit**

```bash
git add docs/source/tutorials/index.rst
git commit -m "docs(tutorials): update toctree to numbered pipeline stage structure"
```

---

## Task 6: Create colmap_sfm.ipynb stub

**Files:**
- Create: `docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb`

- [ ] **Step 1: Create stub notebook**

Create `docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb` with this content (valid minimal nbformat v4):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# COLMAP Structure from Motion\n",
    "\n",
    "**Status: stub — content to be filled in.**\n",
    "\n",
    "This notebook demonstrates sparse 3D reconstruction using COLMAP (Structure from Motion).\n",
    "\n",
    "**Pipeline:** `images/` → COLMAP feature extraction → matching → mapping → `colmap/sparse/0/`\n",
    "\n",
    "**Prerequisite:** [Keyframe Extraction](../01_preprocessing/keyframe_extraction.ipynb)\n",
    "\n",
    "**Cache outputs:**\n",
    "- `docs/.cache/{dataset}/colmap/sparse/0/cameras.bin`\n",
    "- `docs/.cache/{dataset}/colmap/sparse/0/images.bin`\n",
    "- `docs/.cache/{dataset}/colmap/sparse/0/points3D.bin`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Configuration ─────────────────────────────────────────────────────────────\n",
    "DATASET = \"birds_c0043\"  # scene identifier — change per dataset\n",
    "\n",
    "from pathlib import Path\n",
    "CACHE  = Path(\"../../.cache\") / DATASET\n",
    "IMAGES = CACHE / \"images\"\n",
    "COLMAP_OUT = CACHE / \"colmap\"\n",
    "COLMAP_OUT.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "if not IMAGES.exists():\n",
    "    raise FileNotFoundError(f\"No images at {IMAGES}. Run 01_preprocessing/keyframe_extraction first.\")"
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
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb
git commit -m "docs(tutorials): add colmap_sfm stub notebook"
```

---

## Task 7: Fix keyframe_extraction — save images/ to cache

**Files:**
- Modify: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`

The notebook currently saves `frame_scores.json` but does not call `extract_video_frames`, so downstream notebooks have no `images/` directory to read from.

- [ ] **Step 1: Update config cell**

Find the cell containing `CACHE_DIR = Path("../.cache/c0043")`. Replace its full source with:

```python
# ── Configuration ─────────────────────────────────────────────────────────────
VIDEO_PATH = "/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4"
DATASET    = "birds_c0043"   # change per dataset

from pathlib import Path
CACHE  = Path("../../.cache") / DATASET
IMAGES = CACHE / "images"
CACHE.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)
print(f"Cache: {CACHE.resolve()}")
```

- [ ] **Step 2: Update frame scoring cell**

Find the cell containing `_scores_path = CACHE_DIR / "frame_scores.json"`. Replace `CACHE_DIR` references with `CACHE`:

```python
_scores_path = CACHE / "frame_scores.json"
if _scores_path.exists():
    frame_scores = load_frame_scores(_scores_path)
    print("Loaded frame scores from cache")
else:
    frame_scores = score_all_frames(VIDEO_PATH)
    save_frame_scores(frame_scores, _scores_path)
plot_frame_scores(frame_scores)
```

- [ ] **Step 3: Add images/ extraction cell**

After the optical-flow selection cell (the one containing `of_frames = sample_frames_optical_flow`), add a new code cell:

```python
# ── Save selected keyframes to cache ──────────────────────────────────────────
# Images named with sequential COLMAP-style indices (000001.jpg, …)
selected_indices = [s["frame_idx"] for s in frame_scores if s["selected"]]

if list(IMAGES.glob("*.jpg")):
    print(f"images/ already populated ({len(list(IMAGES.glob('*.jpg')))} frames) — skipping extraction")
else:
    extract_video_frames(VIDEO_PATH, selected_indices, IMAGES)
    # Rename to sequential COLMAP convention: 000001.jpg, 000002.jpg, ...
    existing = sorted(IMAGES.glob("*.jpg"))
    for new_idx, fpath in enumerate(existing, start=1):
        fpath.rename(IMAGES / f"{new_idx:06d}.jpg")
    print(f"Saved {len(selected_indices)} frames to {IMAGES}")
```

- [ ] **Step 4: Verify imports include extract_video_frames**

Find the imports cell. Confirm `extract_video_frames` is imported from `collab_splats.utils.frame_sampling`. If not present, add it:

```python
from collab_splats.utils.frame_sampling import (
    get_video_info,
    sample_frames_fps,
    sample_frames_optical_flow,
    score_all_frames,
    save_frame_scores,
    load_frame_scores,
    extract_video_frames,   # ← add if missing
    plot_frame_grid,
    plot_selection,
    plot_frame_scores,
    plot_disparity_sensitivity,
)
```

- [ ] **Step 5: Commit**

```bash
git add docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
git commit -m "fix(notebooks): keyframe_extraction saves images/ to cache for downstream notebooks"
```

---

## Task 8: Standardize cache config in feedforward_methods

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`

This was `feedforward_exploration.ipynb`. It already loads frame scores and saves `vggtx_result.npz`. Update it to use zarr and the new standardized paths.

- [ ] **Step 1: Replace config cell**

Find the cell containing `CACHE_DIR = Path("/workspace/collab-splats/docs/.cache")`. Replace it with:

```python
# ── Configuration ─────────────────────────────────────────────────────────────
VIDEO_PATH = "/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4"
DATASET    = "birds_c0043"

from pathlib import Path
import torch
CACHE  = Path("../../.cache") / DATASET
IMAGES = CACHE / "images"

if not IMAGES.exists() or not list(IMAGES.glob("*.jpg")):
    raise FileNotFoundError(f"No images at {IMAGES}. Run 01_preprocessing/keyframe_extraction first.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}  |  images: {len(list(IMAGES.glob('*.jpg')))}")
```

- [ ] **Step 2: Update VGGT-X cache save/load**

Find the cell that saves/loads `vggtx_result.npz`. Replace with:

```python
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

_vggtx_cache = CACHE / "vggtx" / "reconstruction.zarr"
_vggtx_cache.parent.mkdir(parents=True, exist_ok=True)

if _vggtx_cache.exists():
    result_vggt = FeedforwardResult.load_zarr(_vggtx_cache)
    print(f"Loaded VGGT-X result from cache  ({result_vggt.pts3d.shape[0]:,} pts)")
else:
    creator_vggt = VGGTXCreator()
    creator_vggt.load_model(device=device)
    creator_vggt.setup_inference(IMAGES)
    creator_vggt.run_inference()
    creator_vggt.postprocess()
    result_vggt = creator_vggt.outputs
    result_vggt.save_zarr(_vggtx_cache)
    print(f"VGGT-X done  →  saved to {_vggtx_cache}")
```

- [ ] **Step 3: Update MapAnything cache save/load**

Find the cell that saves/loads `mapanything_result.npz`. Replace with:

```python
_ma_cache = CACHE / "mapanything" / "reconstruction.zarr"
_ma_cache.parent.mkdir(parents=True, exist_ok=True)

if _ma_cache.exists():
    result_ma = FeedforwardResult.load_zarr(_ma_cache)
    print(f"Loaded MapAnything result from cache  ({result_ma.pts3d.shape[0]:,} pts)")
else:
    creator_ma = MapAnythingCreator()
    creator_ma.load_model(device=device)
    creator_ma.setup_inference(IMAGES)
    creator_ma.run_inference()
    creator_ma.postprocess()
    result_ma = creator_ma.outputs
    result_ma.save_zarr(_ma_cache)
    print(f"MapAnything done  →  saved to {_ma_cache}")
```

- [ ] **Step 4: Remove old FRAME_DIR / RESULT_PATH variables**

Delete any remaining cells or lines that reference `FRAME_DIR`, `RESULT_PATH`, `CACHE_DIR`, `frame_scores` loading (the new notebook reads directly from `IMAGES/`).

- [ ] **Step 5: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "fix(notebooks): feedforward_methods uses zarr cache and standardized paths"
```

---

## Task 9: Fix semantic_lifting — load reconstruction from cache

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`

Currently re-runs `VGGTXCreator` from scratch using `/tmp/`. Must load from `vggtx/ba/reconstruction.zarr` (or `vggtx/reconstruction.zarr` for raw).

- [ ] **Step 1: Replace config cell**

Find the cell containing `VIDEO_PATH`, `FRAMES_DIR = Path("/tmp/semantic_lifting/frames")`. Replace entire cell with:

```python
# ── Configuration ─────────────────────────────────────────────────────────────
DATASET   = "birds_c0043"
METHOD    = "vggtx"   # "vggtx" | "mapanything"
VARIANT   = "ba"      # "ba" | "lc" | "" (empty = raw baseline)
QUERIES   = ["tree", "bird feeder", "ground"]
NEGATIVES = ["background"]
LATENT_DIM = 13

from pathlib import Path
import torch
CACHE  = Path("../../.cache") / DATASET
IMAGES = CACHE / "images"
RECON  = CACHE / METHOD / VARIANT if VARIANT else CACHE / METHOD
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not (RECON / "reconstruction.zarr").exists():
    raise FileNotFoundError(
        f"No reconstruction at {RECON}/reconstruction.zarr. "
        f"Run 02_pointcloud/feedforward_methods and 02_pointcloud/bundle_adjustment first."
    )
print(f"Device: {DEVICE}  |  Reconstruction: {RECON}")
```

- [ ] **Step 2: Replace VGGTXCreator section with cache load**

Find the cell containing `creator = VGGTXCreator(extractor_name="maskclip")` and all subsequent inference cells. Replace with:

```python
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

out = FeedforwardResult.load_zarr(RECON / "reconstruction.zarr")
print(f"Loaded reconstruction: {out.pts3d.shape[0]:,} pts  |  {len(out.image_paths)} frames")
```

- [ ] **Step 3: Remove frame extraction cell**

Delete the cell containing `frames = sample_frames_fps(VIDEO_PATH, ...)` and any `cv2.imwrite` loop. The images already exist at `IMAGES/`.

- [ ] **Step 4: Update lifted.zarr save**

Find where lifted features are used (after `lift_features()`). After that cell, add a save cell:

```python
_lifted_path = RECON / "lifted.zarr"
import zarr, numcodecs
store = zarr.open(str(_lifted_path), mode="w")
store.create_dataset(
    "features", data=features.numpy() if hasattr(features, "numpy") else features,
    compressor=numcodecs.Blosc(cname="lz4", clevel=5),
)
print(f"Saved lifted features {features.shape} → {_lifted_path}")
```

- [ ] **Step 5: Remove sys.path.insert hack**

Delete `sys.path.insert(0, "/workspace/collab-splats")` — the project is installed in the env.

- [ ] **Step 6: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "fix(notebooks): semantic_lifting loads reconstruction from zarr cache instead of re-running"
```

---

## Task 10: Standardize cache paths in remaining notebooks

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`
- Modify: `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`
- Modify: `docs/source/tutorials/07_localization/localization.ipynb`
- Modify: `docs/source/tutorials/evals/ground_truth_evals.ipynb`

For each notebook, add the standard config block as the first code cell:

```python
# ── Configuration ─────────────────────────────────────────────────────────────
DATASET   = "birds_c0043"
METHOD    = "vggtx"    # "vggtx" | "mapanything"
VARIANT   = "ba"       # "ba" | "lc" | "" (empty = raw baseline)

from pathlib import Path
CACHE  = Path("../../.cache") / DATASET
IMAGES = CACHE / "images"
RECON  = CACHE / METHOD / VARIANT if VARIANT else CACHE / METHOD
RECON.mkdir(parents=True, exist_ok=True)
```

Then update each notebook's `RESULT_PATH`, `CACHE_DIR`, `FRAME_DIR` references to use `RECON` and `IMAGES`.

- [ ] **Step 1: Fix bundle_adjustment.ipynb**

Verify what the notebook currently loads (look for `RESULT_PATH` or `CACHE_DIR`). Add standard config block at top. Update the load call to:

```python
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
result = FeedforwardResult.load_zarr(CACHE / "vggtx" / "reconstruction.zarr")
```

Update any save call to write into `RECON / "reconstruction.zarr"`:

```python
ba_result.save_zarr(RECON / "reconstruction.zarr")
print(f"BA result saved → {RECON}/reconstruction.zarr")
```

- [ ] **Step 2: Fix slam_loop_closure.ipynb**

Add standard config block. Update load to:

```python
result = FeedforwardResult.load_zarr(CACHE / "vggtx" / "reconstruction.zarr")
```

Update save to:

```python
lc_result.save_zarr(RECON / "reconstruction.zarr")
```

- [ ] **Step 3: Fix localization.ipynb**

Find the cell with `CACHE_DIR = Path("/workspace/collab-splats/docs/.cache")`. Replace with standard config block (METHOD=`vggtx`, VARIANT=`""` for raw). Update load:

```python
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
result = FeedforwardResult.load_zarr(RECON / "reconstruction.zarr")
print(f"pts3d: {result.pts3d.shape}  frames: {result.extrinsics.shape[0]}")
```

- [ ] **Step 4: Fix ground_truth_evals.ipynb**

Add config block (VARIANT=`ba`). Update any result load paths to `RECON / "reconstruction.zarr"`. The eval notebook compares BA vs LC — add a second load:

```python
result_ba = FeedforwardResult.load_zarr(CACHE / METHOD / "ba" / "reconstruction.zarr")
result_lc = FeedforwardResult.load_zarr(CACHE / METHOD / "lc" / "reconstruction.zarr")
```

- [ ] **Step 5: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb \
        docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb \
        docs/source/tutorials/07_localization/localization.ipynb \
        docs/source/tutorials/evals/ground_truth_evals.ipynb
git commit -m "fix(notebooks): standardize DATASET/METHOD/VARIANT cache config across remaining notebooks"
```

---

## Task 11: Verify sphinx build

- [ ] **Step 1: Build docs**

```bash
cd /workspace/collab-splats/docs
/opt/conda/envs/nerfstudio/bin/python -m sphinx source _build/html -b html 2>&1 | tail -20
```

Expected: `build succeeded` with 0 errors. Warnings about missing notebook outputs are acceptable (`nbsphinx_execute = "never"`).

- [ ] **Step 2: Check all tutorials appear in HTML**

```bash
grep -r "feedforward_methods\|slam_loop_closure\|01_preprocessing\|05_lifting" \
     /workspace/collab-splats/docs/_build/html/ --include="*.html" -l | head -10
```

Expected: at least one HTML file found for each pattern.

- [ ] **Step 3: Run full test suite — verify no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --tb=short -x \
    --ignore=tests/examples --ignore=tests/evals 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "docs(tutorials): verified sphinx build and test suite clean after notebook reorganization"
```
