# Pipeline Framework Briefing — For Dashboard Agent

**Context:** You are being asked to update the dashboard to visualise outputs from the
reconstruction pipeline (`docs/reconstruct.py`). This document describes the full
pipeline, what each stage produces, and where outputs land on disk.

---

## Overview

The pipeline converts raw fieldwork video into a 3D reconstruction with lifted semantic
features, optionally a mesh. Controlled by a YAML config; runs via CLI.

```
Video → [preprocess] → frames
      → [pointcloud]  → 3D points + camera poses
      → [semantics]   → per-point feature vectors
      → [mesh]        → triangle mesh (optional)
```

Entry point: `docs/reconstruct.py`
Core class: `collab_splats.wrapper.reconstructor.Reconstructor`

---

## Stage 1: Preprocess

**What it does:** Extracts keyframes from input video. Writes to `<output_path>/images/`.

**Options:**

| Config key | Values | Default | Effect |
|---|---|---|---|
| `preprocessing.frame_selection` | `fps` \| `optical_flow` | `fps` | Sampling strategy — uniform vs motion-guided |
| `preprocessing.frame_proportion` | float (0–1) | `0.1` | Fraction of total frames to extract |
| `preprocessing.min_frames` | int | `300` | Floor — always extract at least this many |
| `preprocessing.max_frames` | int \| null | `null` | Cap; null = no limit |

**Output:** `<output_path>/images/*.jpg` (or `.png`)

---

## Stage 2: Pointcloud

**What it does:** Runs a feedforward or SFM model to produce:
- 3D point cloud (world XYZ + RGB)
- Camera poses (extrinsics + intrinsics per frame)
- Depth maps (feedforward only)
- Confidence maps (feedforward only)

**Method option:**

| `pointcloud.method` | Description |
|---|---|
| `feedforward` | Neural feedforward: single-pass model produces points + poses + depth. Fast. No COLMAP needed. |
| `sfm` | Structure from Motion (COLMAP/hloc). Slower, more classical. Requires feature matching. |

**Backend option (feedforward only):**

| `pointcloud.backend` | Model | Notes |
|---|---|---|
| `vggt_omega` | VGGTOmegaCreator | Default. Latest architecture. Best quality. |
| `vggtx` | VGGTXCreator | Older VGGT variant. Slightly faster. |
| `mapanything` | MapAnythingCreator | MapAnything-based. Different FOV handling. |

**Post-processing options:**

| Config key | Default | Effect |
|---|---|---|
| `pointcloud.bundle_adjustment` | `false` | Run Levenberg-Marquardt BA to refine poses |
| `pointcloud.loop_closure` | `false` | Run pose-graph loop closure (Sim3 correction) |
| `pointcloud.clean.enabled` | `true` | Remove outlier points (statistical + voxel filter) |
| `pointcloud.clean.outlier_removal` | `true` | Statistical outlier removal |
| `pointcloud.clean.voxel_size` | `null` | Voxel downsampling; null = adaptive |
| `pointcloud.clean.confidence_threshold` | `null` | Drop low-confidence points; null = no filter |

**Output:**

```
<output_path>/<backend>/
  feedforward.zarr          ← primary output (zarr store, see schema below)
  colmap/sparse/0/          ← COLMAP reconstruction (always written)
    cameras.bin
    images.bin
    points3D.bin
  transforms.json           ← nerfstudio-compatible camera poses
```

**`feedforward.zarr` schema** (zarr store, load with `FeedforwardResult.load_zarr(path)`):

| Array key | Shape | Dtype | Description |
|---|---|---|---|
| `points` | (P, 3) | float32 | World-space XYZ for P points |
| `colors` | (P, 3) | uint8 | RGB [0–255] per point |
| `extrinsics` | (N, 4, 4) | float32 | World-to-camera homogeneous transforms |
| `intrinsics` | (N, 3, 3) | float32 | Camera intrinsics K per frame |
| `depth` | (N, H, W) | float32 | Depth maps in metres (model resolution) |
| `confidence` | (N, H, W) | float32 | Per-pixel confidence [0–1] |
| `pixel_indices` | (P, 3) | int32 | [frame_id, row, col] source pixel per point |
| `images` | (N, 3, H, W) | uint8 | RGB frames at model resolution |
| `original_coords` | (N, 6) | float32 | [tl_x, tl_y, br_x, br_y, orig_w, orig_h] |

N = frame count, P = point count, H/W = model resolution (typically 224 or 336px).

**Key classes:**

- `FeedforwardResult` — typed wrapper around zarr; load via `FeedforwardResult.load_zarr(path, load_images=True)`
- `PointcloudResult` — wraps pycolmap.Reconstruction; properties: `.points` (P,3), `.colors` (P,3), `.extrinsics` (N,4,4), `.intrinsics` (N,3,3), `.image_paths`

---

## Stage 3: Semantics

**What it does:** Extracts 2D feature maps from each frame using a registered extractor,
projects them into 3D (weighted by depth + confidence), optionally compresses with PCA.

**Two sub-steps:**
1. **2D extraction** — runs registered `BaseFeatureExtractor` on each frame → `(D, H_p, W_p)` tensor per frame. Cached at `<output_path>/features/<extractor>/<extractor>.zarr`.
2. **Lifting** — `lift_features()` in `collab_splats/pointcloud/utils.py` projects 2D features onto 3D points via multi-view confidence-weighted aggregation.

**Extractor options:**

| `semantics.extractor` | Class | Type | Notes |
|---|---|---|---|
| `talk2dino` | `Talk2DinoExtractor` | `BaseQueryableExtractor` | DINO features + text-query support. Best for semantic segmentation. |
| `dinov2` | `DINOFeatureExtractor` | `BaseFeatureExtractor` | Pure DINOv2 features. Dense, high-D. |
| `maskclip` | `MaskCLIPExtractor` | `BaseQueryableExtractor` | CLIP-based, supports text queries. |

`BaseQueryableExtractor` subclasses support `.query(text)` for text-driven feature retrieval.
`BaseFeatureExtractor` subclasses only support dense feature extraction.

**PCA compression:**

| Config key | Default | Effect |
|---|---|---|
| `semantics.n_components` | `64` | PCA latent dim. null = no compression. |

If `n_components` set, a `FeatureAutoencoder` is fit and saved alongside features.

**Output:**

```
<output_path>/<backend>/semantics/<extractor>/
  features.zarr        ← lifted 3D features: array "features" shape (P, D)
  compressor.pt        ← PCA weights (only if n_components set)

<output_path>/features/<extractor>/
  <extractor>.zarr     ← 2D feature cache: array "features" shape (N, D, H_p, W_p)
```

P = point count (same as pointcloud stage). D = feature dim after optional compression.

---

## Stage 4: Mesh (optional, off by default)

**What it does:** Fuses depth maps from feedforward into a triangle mesh via TSDF or Poisson.

| Config key | Values | Default |
|---|---|---|
| `mesh.enabled` | bool | `false` |
| `mesh.mesher` | `tsdf` \| `poisson` | `tsdf` |
| `mesh.voxel_size` | float | `0.01` (metres) |
| `mesh.sdf_trunc` | float | `0.04` (metres) |

**Output:** `<output_path>/<backend>/mesh/mesh.ply`

---

## Config & Reproducibility

Every run writes a `run_config.yaml` to `<output_path>/` containing the full merged config
(base + dataset + CLI overrides). Load it to know exactly what produced any output.

Config hierarchy:
```
configs/reconstruction/base.yaml          ← defaults
configs/reconstruction/datasets/<name>.yaml  ← per-dataset overrides
+ CLI KEY=VALUE overrides                 ← highest priority
```

---

## Output Directory Tree (complete)

```
/workspace/outputs/<dataset_name>/
  run_config.yaml                         ← full merged config for reproducibility
  images/                                 ← extracted keyframes (jpg/png)
  features/                              ← 2D feature cache (extractor subdir)
    talk2dino/
      talk2dino.zarr                      ← (N, D, H_p, W_p) feature maps
  <backend>/                             ← e.g. vggt_omega/
    feedforward.zarr                      ← full feedforward output (see schema above)
    colmap/sparse/0/                      ← COLMAP reconstruction
    transforms.json                       ← nerfstudio camera poses
    semantics/
      <extractor>/                        ← e.g. talk2dino/
        features.zarr                     ← (P, D) lifted 3D features
        compressor.pt                     ← PCA weights (if n_components set)
    mesh/
      mesh.ply                            ← (if mesh.enabled=true)
```

---

## Current Dashboard State

**Location:** `collab_splats/dashboard/`

**Launch:**
```bash
python -m collab_splats.dashboard semantics --base-dir /workspace/fieldwork-data --port 7860
```

**Current capabilities:**
- Video discovery from `{base_dir}/{species}/{date}/SplatsSD/*.MP4` via `video_discovery.py`
- Config panel (`config_panel.py`) — reads/writes the *old* splatter YAML format (not reconstruction pipeline YAMLs)
- Semantics tab (`semantics.py`) — extracts features interactively via `BaseFeatureExtractor` registry, displays PCA-reduced feature maps as RGB overlay on frames

**What the dashboard does NOT yet do (needs adding):**
- Load and display reconstruction outputs from `feedforward.zarr`
- Display 3D pointcloud (points + colors)
- Load pre-computed lifted features from `<backend>/semantics/<extractor>/features.zarr`
- Read `run_config.yaml` to auto-populate backend/extractor dropdowns
- Browse outputs from `/workspace/outputs/` (not just `/workspace/fieldwork-data/`)

**Key APIs the dashboard can use:**

```python
# Load full feedforward result (pointcloud + depth + images)
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
ff = FeedforwardResult.load_zarr(
    Path("/workspace/outputs/birds_c0043/vggt_omega/feedforward.zarr"),
    load_images=True,
)
# ff.points        (P, 3) float32
# ff.colors        (P, 3) uint8
# ff.extrinsics    (N, 4, 4) float32
# ff.intrinsics    (N, 3, 3) float32
# ff.depth         (N, H, W) float32
# ff.images        (N, 3, H, W) uint8

# Load lifted semantic features
import zarr
store = zarr.open("/workspace/outputs/birds_c0043/vggt_omega/semantics/talk2dino/features.zarr")
features = store["features"][:]  # (P, D) float32

# Load run config
import yaml
cfg = yaml.safe_load(open("/workspace/outputs/birds_c0043/run_config.yaml"))
backend = cfg["pointcloud"]["backend"]    # e.g. "vggt_omega"
extractor = cfg["semantics"]["extractor"] # e.g. "talk2dino"

# Discover available output dirs
from pathlib import Path
output_dirs = [d for d in Path("/workspace/outputs").iterdir()
               if (d / "run_config.yaml").exists()]
```

**Tech stack:** Panel + param (reactive widgets). Uses `pn.serve()`. No JS frameworks.
All heavy computation (model inference) runs in background threads. Avoid blocking the Panel
event loop.

---

## Registered Extractors (for dropdowns)

```python
from collab_splats.semantics.features import BaseFeatureExtractor
available = list(BaseFeatureExtractor._registry.keys())
# ['dinov2', 'talk2dino', 'maskclip', ...]
```

Queryable extractors support `.query(text_prompt)` for text-driven retrieval:

```python
from collab_splats.semantics.features import BaseQueryableExtractor
queryable = list(BaseQueryableExtractor._registry.keys())
# ['talk2dino', 'maskclip']
```

---

## Env / Runtime Notes

- Python: `/opt/conda/envs/reconstruction/bin/python` (3.11). NOT base conda.
- GPU: required for model inference. Don't run inference in main Panel thread.
- Memory: 46.6 GB container cap. Load one dataset at a time; release tensors after use.
- `feedforward.zarr` can be 2-10 GB. Use zarr lazy loading (don't call `[:]` on large arrays unless needed).
