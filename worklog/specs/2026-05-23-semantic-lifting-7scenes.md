# Semantic Lifting on 7-Scenes Dataset

**Date:** 2026-05-23  
**Status:** Approved

## Goal

Extend `docs/semantics/semantic_lifting.ipynb` to run the semantic lifting pipeline on
pre-extracted image sequences (7scenes chess) in addition to videos. Add scalable feature
caching via Zarr so extraction, AE training, and lifting are decoupled and re-entrant.

## Dataset Choice

**7scenes chess** — `evals/data/7scenes/chess/chess/seq-01/`

- 1000 frames on disk, `frame-*.color.png` + `frame-*.pose.txt`
- Indoor bounded scene → clean VGGT-X reconstruction
- Rich distinct semantics (chess board, pieces, table, chair) → meaningful CLIP text queries
- GT poses available for future eval integration
- Preferred over bicycle (LLFF format, outdoor, less semantic variety for CLIP)

## Pipeline

```
§0 Config
§1 Frame load          get_dataset(DATASET_TYPE) — uniform interface for all sources
§2 VGGT-X              unchanged — accepts list[Path]
§3 Inspect             unchanged
§4 Extract + cache     extractor.extract_and_cache() → raw.zarr (768D)
§5 Train AE + encode   ae.fit(raw.zarr) → ae.encode_cache(raw.zarr, compressed.zarr)
§6 Lift compressed     lift_features(feature_cache=compressed.zarr) → (P, latent_dim)
§7 Text query + viz    ae.per_point_decode() → visualize_splat(similarity=scores, cmap="viridis")
§8 Gallery             per-query tiled views — visualize_splat per query
```

Sections §2–§3 are unchanged. §0–§1 and §4–§8 change.

## Correct Compression Order

The `FeatureAutoencoder` docstring says "compressing patch features **before** 3D lifting."

**Wrong (current notebook):**
```
lift_features() → (P, 768) → ae.fit() → per_point_encode → (P, latent_dim)
```
Allocates (P, 768) intermediate. At 30 frames, P ≈ 4.8M → **14GB** — unacceptable at scale.

**Correct — three cleanly separated stages:**
```
extract_and_cache() → raw.zarr (D=768, H_p, W_p)
ae.fit(raw.zarr)    → train AE on spatial patch features
ae.encode_cache()   → compressed.zarr (latent_dim, H_p, W_p)
lift_features(feature_cache=compressed.zarr) → (P, latent_dim)  ← pure geometric lift
ae.per_point_decode() → only at query time, for CLIP scoring
```

`lift_features` receives **already-compressed** features from the cache — no transform logic
inside the function. Compression is the AE's responsibility; lifting is geometric-only.


## Why Zarr

- **Fork-safe**: DataLoader `num_workers > 0` works with no workarounds (unlike h5py)
- **Cloud-native**: same API for local and `gs://` paths — relevant given GCS data infrastructure
- **Chunked**: `chunks=(1, D, H_p, W_p)` → reading frame `i` loads one chunk from disk
- **Self-describing**: `.zattrs` stores extractor name, patch_size, resolution → cache validation

**Zarr layout:**
```
CACHE_DIR/{extractor_name}.zarr/
  .zattrs:  {extractor, patch_size, resolution, n_frames, created_at}
  features: (N, D, H_p, W_p) float32   chunks=(1, D, H_p, W_p)
```

## Library Changes

### 1. `evals/datasets.py` — uniform frame loading interface

Add `"bicycle"` and `"video"` loaders to `_REGISTRY`:

```python
def _load_bicycle(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    images = sorted((seq_dir / "images_4").glob("*.png"))[:max_frames]
    return EvalDataset(images=images, gt_poses=np.zeros((len(images), 4, 4), dtype=np.float32))

def _load_video(seq_dir: Path, max_frames: int = 500, fps: float = 1.0) -> EvalDataset:
    from collab_splats.utils.frame_sampling import sample_frames_fps
    frames_dir = seq_dir.parent / (seq_dir.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    images = sample_frames_fps(str(seq_dir), frames_dir, fps=fps)[:max_frames]
    return EvalDataset(images=images, gt_poses=np.zeros((len(images), 4, 4), dtype=np.float32))

_REGISTRY["bicycle"] = _load_bicycle
_REGISTRY["video"]   = _load_video
```

Notebook §1 is a single expression for all source types — no if/else.

### 2. `collab_splats/semantics/features.py`

Add `extract_and_cache()` to `BaseFeatureExtractor`:

```python
def extract_and_cache(
    self,
    image_paths: list[Path],
    cache_dir: Path,
    batch_size: int = 1,
    skip_existing: bool = True,
) -> Path:
    """Extract patch features for all images, write to cache_dir/{name}.zarr.

    Returns the zarr store path. Re-entrant when skip_existing=True.
    Chunks=(1, D, H_p, W_p) so reading frame i loads exactly one chunk.
    """
```

- Namespaces store by extractor `name` (e.g., `"maskclip"`)
- Validates existing cache via `.zattrs` before skipping
- Returns `cache_dir / f"{self.name}.zarr"`

### 3. `collab_splats/semantics/compression.py`

No changes. `FeatureAutoencoder` stays focused: `fit(Tensor)`, `encode`, `decode`,
`per_point_encode`, `per_point_decode`. Cache loading and per-frame encoding are
explicit notebook cells — not hidden inside the AE.

### 4. `collab_splats/pointcloud/utils.py`

Refactor `lift_features` to be a **pure geometric operation** — 2D feature maps → 3D points.
Rename the old extractor-running function to `extract_and_lift_features` for backward compat.

```python
def lift_features(
    feature_maps: list[torch.Tensor],   # list of (D, H_p, W_p) — one per frame, any D
    pixel_indices: np.ndarray,          # (P, 3) int32  [frame_id, row, col] — pixel space
    image_size: tuple[int, int],        # (H, W) — upsample target (from out.images.shape[-2:])
) -> np.ndarray:                        # (P, D) float32
    """Map per-frame 2D feature maps to per-point features via bilinear upsampling.

    Upsamples each (D, H_p, W_p) map to (D, H, W), then indexes at pixel coords.
    Pure geometric operation: no extractor, no cache, no AE.
    Works for any D — raw (768) or compressed (latent_dim).
    Bilinear upsampling gives sub-patch precision over floor-div patch indexing.
    """
```

No `patch_size` needed — upsampling handles the coordinate space conversion.
`extract_and_lift_features` keeps the old signature for any existing callers.

## Notebook Changes (`docs/semantics/semantic_lifting.ipynb`)

### §0 — Configuration

```python
import sys
sys.path.insert(0, "/workspace/collab-splats")
sys.path.insert(0, "/workspace/collab-splats/evals")

import numpy as np
import torch
import pyvista as pv
from pathlib import Path

from datasets import get_dataset
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
from collab_splats.pointcloud.utils import lift_features
from collab_splats.semantics.features import MaskCLIPExtractor
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.utils.visualization import pointcloud_to_polydata, visualize_splat, PCD_KWARGS

DATASET_TYPE = "7scenes"
SEQ_DIR      = Path("/workspace/collab-splats/evals/data/7scenes/chess/chess/seq-01")
N_FRAMES     = 30
CAPTURE_FPS  = 30
SAMPLE_FPS   = 1

CACHE_DIR  = Path("/tmp/semantic_lifting/feature_cache")
QUERIES    = ["chess board", "chess pieces", "table", "chair", "wall"]
NEGATIVES  = ["background"]
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 13

CACHE_DIR.mkdir(parents=True, exist_ok=True)
```

### §1 — Frame Loading (no if/else)

```python
stride = max(1, round(CAPTURE_FPS / SAMPLE_FPS))
frames = get_dataset(DATASET_TYPE)(SEQ_DIR, max_frames=9999).images[::stride][:N_FRAMES]
print(f"Loaded {len(frames)} frames  source={DATASET_TYPE}")
```

### §2–§3 — VGGT-X + Inspect

Unchanged. Produces `out.images (N,3,H,W)`, `out.pixel_indices (P,3)`, `out.pts3d (P,3)`.

### §4 — Extract + Cache

```python
extractor = MaskCLIPExtractor(device=DEVICE)
zarr_path = extractor.extract_and_cache(frames, CACHE_DIR, skip_existing=True)
print(f"Feature cache: {zarr_path}")
```

### §5 — Load from Cache, Train AE, Encode per Frame

```python
import zarr

z = zarr.open(str(CACHE_DIR / "maskclip.zarr"), mode='r')
raw_maps = [torch.from_numpy(z['features'][i]) for i in range(len(frames))]
# raw_maps: list of (768, H_p, W_p) CPU tensors

# Flatten all patches and train AE
all_patches = torch.cat([f.flatten(1).T for f in raw_maps])   # (N_patches, 768)
ae = FeatureAutoencoder(input_dim=768, latent_dim=LATENT_DIM)
ae.fit(all_patches.to(DEVICE))
print(f"AE trained: 768D → {LATENT_DIM}D")

# Encode per frame: list of (latent_dim, H_p, W_p)
compressed_maps = [ae.encode(f.to(DEVICE)).detach().cpu() for f in raw_maps]
```

### §6 — Lift Compressed Features (2D → 3D)

```python
codes = lift_features(
    compressed_maps,                               # list of (latent_dim, H_p, W_p)
    pixel_indices,
    image_size=images.shape[-2:],                  # (H, W) from out.images
)
print(f"codes: {codes.shape}  dtype={codes.dtype}")
# (P, LATENT_DIM) — (P, 768) never allocated
```

### §7 — Text Query + Visualization

```python
feat_decoded = ae.per_point_decode(
    torch.from_numpy(codes).to(DEVICE)
).detach().cpu().numpy()                           # (P, 768) — only at query time

scores = score_queries(feat_decoded, QUERIES, NEGATIVES, extractor)  # (P, Q)

# Visualize combined score (max across queries)
cloud = pointcloud_to_polydata(pts3d, similarity=scores.max(axis=1))
pl = visualize_splat(
    cloud,
    mesh_kwargs={**PCD_KWARGS, "scalars": "similarity", "cmap": "viridis", "rgb": False},
    viz_kwargs=VIZ_KWARGS,
)
pl.show()
```

### §8 — Per-Query Gallery

```python
for query, q_scores in zip(QUERIES, scores.T):    # scores (P, Q) — one col per query
    cloud_q = pointcloud_to_polydata(pts3d, similarity=q_scores)
    pl = visualize_splat(
        cloud_q,
        mesh_kwargs={**PCD_KWARGS, "scalars": "similarity", "cmap": "viridis", "rgb": False},
        viz_kwargs=VIZ_KWARGS,
    )
    # tile renders — reuse existing gallery cell pattern
```

## Files to Modify / Create

1. `evals/datasets.py` — add `_load_bicycle`, `_load_video`, register both
2. `collab_splats/semantics/features.py` — add `extract_and_cache()` to `BaseFeatureExtractor`
3. `collab_splats/semantics/compression.py` — no changes
4. `collab_splats/pointcloud/utils.py` — refactor to pure `lift_features(feature_maps, pixel_indices, patch_size)`; rename old fn to `extract_and_lift_features`
5. `docs/semantics/semantic_lifting.ipynb` — §0, §1, §4–§8

## Acceptance Criteria

1. `extract_and_cache()` writes Zarr; re-run with `skip_existing=True` skips
2. Zarr `.zattrs` contains extractor name, patch_size, n_frames
3. `ae.fit(tensor)` unchanged
4. `lift_features(compressed_maps, pixel_indices, image_size)` returns `(P, latent_dim)` — no `(P, 768)` allocated; bilinear upsample used
5. `extract_and_lift_features()` still works for existing callers (backward compat)
6. `visualize_splat()` called with `cmap="viridis"` and `VIZ_KWARGS` passthrough
7. Notebook runs end-to-end on chess seq-01 with `N_FRAMES=30`
8. `DATASET_TYPE = "video"` path runs (backward compat)
