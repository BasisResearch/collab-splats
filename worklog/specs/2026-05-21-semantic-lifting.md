# Spec: Semantic Lifting — Shape-Aware API + Tutorial Notebook

**Date:** 2026-05-21
**Status:** Approved for implementation

---

## Goal

Two deliverables:

1. **Library change** — make `compute_similarity` and `score_queries` on `BaseQueryableExtractor` shape-aware so point arrays `(P, D)` work as a first-class input alongside image feature maps `(C, H, W)`. No new public methods.

2. **Tutorial notebook** — `docs/source/tutorials/semantics/semantic_lifting.ipynb`. Demonstrates lifting MaskCLIP patch features into a VGGT-X pointcloud and querying the 3D scene by text.

---

## Part 1: Shape-Aware API

### Problem

`compute_similarity` and `score_queries` currently use `einsum("chw,nc->nhw", ...)`, which requires a 3D `(C, H, W)` input. Point arrays from `lift_features` are `(P, D)` — 2D — so callers must manually reshape or use `compute_semantic_contrast` directly, which is opaque and undocumented.

### Solution

Detect `features.ndim` at entry. If `ndim == 2` (point array), reshape internally to `(D, P, 1)`, compute using the existing einsum, then squeeze the trailing dim back out before returning.

### Shape Contract

| Input shape | `compute_similarity` output | `score_queries` output |
|-------------|----------------------------|------------------------|
| `(C, H, W)` | `(N_queries, H, W)` | `(H, W)` — **unchanged** |
| `(P, D)`    | `(N_queries, P)`    | `(P,)` — **new** |

### Implementation

**File:** `collab_splats/semantics/features.py`
**Class:** `BaseQueryableExtractor` (~line 255)

```python
def compute_similarity(
    self,
    features: torch.Tensor,
    queries: List[str],
) -> torch.Tensor:
    """Raw cosine similarities between features and text queries.

    Args:
        features: (C, H, W) patch feature map, or (P, D) point feature array.
        queries: text strings to compare against features.

    Returns:
        (N_queries, H, W) when input is (C, H, W).
        (N_queries, P)    when input is (P, D).
    """
    is_points = features.ndim == 2
    if is_points:
        features = features.T.unsqueeze(-1)          # (D, P, 1)
    text_embs = self.encode_text(queries)
    out = torch.einsum("chw,nc->nhw", features, text_embs)
    return out.squeeze(-1) if is_points else out


def score_queries(
    self,
    features: torch.Tensor,
    positive: List[str],
    negative: List[str] = ["object"],
    temperature: float = 0.05,
    reduction: str = "max",
) -> torch.Tensor:
    """Contrastive score. Accepts (C, H, W) → (H, W) or (P, D) → (P,)."""
    is_points = features.ndim == 2
    if is_points:
        features = features.T.unsqueeze(-1)          # (D, P, 1)
    # ... existing body unchanged ...
    result = <existing logic returning (H, W) or (P, 1)>
    return result.squeeze(-1) if is_points else result
```

`score_queries` delegates to `compute_similarity` → `compute_semantic_contrast` (unchanged). Only the guard + squeeze at the boundary changes.

### Tests

**File:** `tests/semantics/test_features.py`

New tests (use `MaskCLIPExtractor` as concrete queryable extractor):

```python
def test_compute_similarity_point_array():
    ext = MaskCLIPExtractor(device="cpu")
    P, D = 100, 768
    features = torch.randn(P, D)
    features = F.normalize(features, dim=-1)
    out = ext.compute_similarity(features, ["tree", "ground"])
    assert out.shape == (2, P)

def test_score_queries_point_array():
    ext = MaskCLIPExtractor(device="cpu")
    P, D = 100, 768
    features = torch.randn(P, D)
    features = F.normalize(features, dim=-1)
    scores = ext.score_queries(features, positive=["tree"], negative=["background"])
    assert scores.shape == (P,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0

def test_compute_similarity_image_map_unchanged():
    """Regression: (C, H, W) path must be unaffected."""
    ext = MaskCLIPExtractor(device="cpu")
    C, H, W = 768, 12, 16
    features = torch.randn(C, H, W)
    out = ext.compute_similarity(features, ["tree", "ground"])
    assert out.shape == (2, H, W)

def test_score_queries_image_map_unchanged():
    ext = MaskCLIPExtractor(device="cpu")
    C, H, W = 768, 12, 16
    features = torch.randn(C, H, W)
    scores = ext.score_queries(features, positive=["tree"], negative=["background"])
    assert scores.shape == (H, W)
```

---

## Part 2: Tutorial Notebook

**File:** `docs/source/tutorials/semantics/semantic_lifting.ipynb`

### Notebook Sections

**§0 — Configuration**
```python
VIDEO_PATH = Path("/path/to/your/video.mp4")
FRAMES_DIR = Path("/tmp/semantic_lifting_frames")
QUERIES    = ["tree", "bird feeder", "ground"]
NEGATIVES  = ["background"]
```

**§1 — Keyframe Extraction**
```python
frames = sample_frames_optical_flow(VIDEO_PATH, max_frames=40)
for i, frame_bgr in enumerate(frames):
    cv2.imwrite(str(FRAMES_DIR / f"frame_{i:04d}.jpg"), frame_bgr)
```

**§2 — VGGT-X Reconstruction**
```python
creator = VGGTXCreator()
creator.load_model()
creator.setup_inference(FRAMES_DIR)
creator.run_inference()
creator.postprocess()
out = creator.outputs
```

**§3 — Inspect Outputs**
```python
pts3d         = out.pts3d          # (P, 3) float32
pixel_indices = out.pixel_indices  # (P, 3) int32
images        = out.images         # (N, 3, H, W) float32 [0,1]
colors        = out.colors         # (P, 3) uint8
```

**§4 — MaskCLIP Feature Lifting**
```python
features = lift_features(images, pixel_indices, extractor_name="maskclip", device=DEVICE)
# → (P, 768) float32, L2-normalized
```

**§5 — Text Queries → Per-Point Scores**
```python
extractor = MaskCLIPExtractor(device=DEVICE)
feat_t    = torch.from_numpy(features)         # (P, 768)

# shape-aware: (P, D) → (P,)
scores    = extractor.score_queries(
    feat_t, positive=QUERIES, negative=NEGATIVES, temperature=0.05
).numpy()

# shape-aware: (P, D) → (N_queries, P) → transpose → (P, Q)
per_query = extractor.compute_similarity(feat_t, QUERIES).T.numpy()
```

**§6 — Interactive 3D Viewer**
```python
cloud = pointcloud_to_polydata(
    pts3d,
    RGB=colors,
    semantic=scores,
    **{q.replace(" ", "_"): per_query[:, i] for i, q in enumerate(QUERIES)},
)
pl = pv.Plotter(title="Semantic Lifting")
pl.add_mesh(cloud, scalars="semantic", cmap="plasma", point_size=2)
pl.add_scalar_bar("semantic score", fmt="%.2f")
pl.add_title("Switch scalars in the side panel", font_size=9)
pl.show()
```

**§7 — Multi-Query Tiled Gallery**
```python
n  = len(QUERIES)
pl = pv.Plotter(shape=(1, n), title="Multi-Query Semantic Gallery")
for i, q in enumerate(QUERIES):
    pl.subplot(0, i)
    pl.add_mesh(cloud.copy(), scalars=q.replace(" ", "_"), cmap="plasma", point_size=2)
    pl.add_title(f'"{q}"', font_size=10)
pl.link_views()
pl.show()
```

---

## Imports (Notebook)

```python
import sys
sys.path.insert(0, "/workspace/collab-splats")

import cv2
import numpy as np
import torch
import pyvista as pv
from pathlib import Path

from collab_splats.utils.frame_sampling import sample_frames_optical_flow
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
from collab_splats.pointcloud.utils import lift_features
from collab_splats.semantics.features import MaskCLIPExtractor
from collab_splats.utils.visualization import pointcloud_to_polydata

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Utilities Referenced

| Utility | Path |
|---------|------|
| `BaseQueryableExtractor.compute_similarity` | `collab_splats/semantics/features.py:278` |
| `BaseQueryableExtractor.score_queries` | `collab_splats/semantics/features.py:297` |
| `compute_semantic_contrast` | `collab_splats/semantics/utils.py:33` |
| `lift_features` | `collab_splats/pointcloud/utils.py:706` |
| `sample_frames_optical_flow` | `collab_splats/utils/frame_sampling.py` |
| `VGGTXCreator` | `collab_splats/pointcloud/feedforward/vggtx.py` |
| `pointcloud_to_polydata` | `collab_splats/utils/visualization.py` |

---

## Verification

```bash
# Library tests
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_features.py -v

# Import check
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.semantics.features import MaskCLIPExtractor
import torch, torch.nn.functional as F
ext = MaskCLIPExtractor(device='cpu')
P, D = 1000, 768
f = F.normalize(torch.randn(P, D), dim=-1)
scores = ext.score_queries(f, positive=['tree'], negative=['background'])
assert scores.shape == (P,) and 0 <= scores.min() and scores.max() <= 1
print('PASS')
"
```

Notebook: run all cells top-to-bottom without error. PyVista viewer opens with colored pointcloud.

---

## Execution Order

1. Write spec (this file) → commit
2. Write failing tests in `tests/semantics/test_features.py`
3. Implement shape-aware `compute_similarity` + `score_queries` in `features.py`
4. Run tests → all pass → commit library change
5. Create notebook `docs/source/tutorials/semantics/semantic_lifting.ipynb`
6. Run notebook end-to-end → commit notebook
