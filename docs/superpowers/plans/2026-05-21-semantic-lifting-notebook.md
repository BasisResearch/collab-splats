# Semantic Lifting Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `docs/source/tutorials/semantics/semantic_lifting.ipynb` — a self-contained tutorial lifting MaskCLIP features into a VGGT-X pointcloud and querying by text.

**Architecture:** Single notebook; no library changes. Reuses `lift_features`, `MaskCLIPExtractor`, `compute_semantic_contrast`, and `pointcloud_to_polydata` from existing modules. Scoring uses manual cosine similarity + `compute_semantic_contrast` (not `score_queries`, which expects 2D feature maps).

**Tech Stack:** PyVista, PyTorch, MaskCLIP (ViT-L/14@336px), VGGT-X

---

## Files

- Create: `docs/source/tutorials/semantics/semantic_lifting.ipynb`
- Reference: `docs/source/tutorials/pointcloud/feedforward_exploration.ipynb` (style model)
- Spec: `docs/superpowers/specs/2026-05-21-semantic-lifting-notebook-design.md`

---

### Task 1: Create the Notebook

**Files:**
- Create: `docs/source/tutorials/semantics/semantic_lifting.ipynb`

- [ ] **Step 1: Write the notebook JSON**

The complete notebook (all cells at once — it is a single JSON artifact):

```python
# Verify FRAMES_DIR exists and notebook directory exists
import os
os.makedirs("/workspace/collab-splats/docs/source/tutorials/semantics", exist_ok=True)
```

Then write `docs/source/tutorials/semantics/semantic_lifting.ipynb` with the following cells in order:

**Cell 1 — Markdown: Title**
```markdown
# Semantic Lifting: Text-Queryable 3D Pointclouds

Lift MaskCLIP patch features from 2D frames into a VGGT-X pointcloud, then query the scene by text.

**Pipeline:**
1. Extract keyframes from video
2. Run VGGT-X → `pts3d` (P,3) + `pixel_indices` (P,3)
3. `lift_features` maps each 3D point to its MaskCLIP patch embedding → (P, 768)
4. `encode_text` + contrastive softmax → per-point semantic scores
5. Color the 3D cloud by score in an interactive PyVista viewer

Builds on the [Feedforward Exploration](../pointcloud/feedforward_exploration.ipynb) tutorial.
```

**Cell 2 — Code: Imports**
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
from collab_splats.semantics.utils import compute_semantic_contrast
from collab_splats.utils.visualization import pointcloud_to_polydata

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
```

**Cell 3 — Markdown: §0 Configuration**
```markdown
## §0  Configuration

Set `VIDEO_PATH` to your video file. `FRAMES_DIR` is where extracted keyframes will be saved.
```

**Cell 4 — Code: Config**
```python
# ── USER CONFIGURATION ──────────────────────────────────────────────────────
VIDEO_PATH = Path("/path/to/your/video.mp4")
FRAMES_DIR = Path("/tmp/semantic_lifting_frames")
# ────────────────────────────────────────────────────────────────────────────

QUERIES   = ["tree", "bird feeder", "ground"]
NEGATIVES = ["background"]

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
```

**Cell 5 — Markdown: §1 Keyframe Extraction**
```markdown
## §1  Keyframe Extraction

Select informative frames via optical-flow scoring (motion + coverage), then save them
to disk for VGGT-X. Targets ~40 frames — enough for a dense reconstruction without OOM.
```

**Cell 6 — Code: Keyframe Extraction**
```python
frames = sample_frames_optical_flow(VIDEO_PATH, max_frames=40)
print(f"Selected {len(frames)} keyframes")

for i, frame_bgr in enumerate(frames):
    cv2.imwrite(str(FRAMES_DIR / f"frame_{i:04d}.jpg"), frame_bgr)

print(f"Saved to {FRAMES_DIR}")
```

**Cell 7 — Markdown: §2 VGGT-X Reconstruction**
```markdown
## §2  VGGT-X Reconstruction

Run feedforward inference to obtain `pts3d` (P, 3), `pixel_indices` (P, 3), and
`images` (N, 3, H, W). `pixel_indices[p]` = (frame, row, col) — the source pixel
for each 3D point, which is the bridge from 2D features to 3D space.
```

**Cell 8 — Code: VGGTXCreator pipeline**
```python
creator = VGGTXCreator()
creator.load_model()
creator.setup_inference(FRAMES_DIR)
creator.run_inference()
creator.postprocess()
out = creator.outputs
```

**Cell 9 — Code: Inspect outputs**
```python
pts3d         = out.pts3d          # (P, 3) float32 world-space XYZ
pixel_indices = out.pixel_indices  # (P, 3) int32  [frame_id, row, col]
images        = out.images         # (N, 3, H, W) float32 in [0, 1]
colors        = out.colors         # (P, 3) uint8  RGB

print(f"pts3d:         {pts3d.shape}  dtype={pts3d.dtype}")
print(f"pixel_indices: {pixel_indices.shape}  dtype={pixel_indices.dtype}")
print(f"images:        {tuple(images.shape)}  dtype={images.dtype}")
```

**Cell 10 — Markdown: §3 MaskCLIP Feature Lifting**
```markdown
## §3  MaskCLIP Feature Lifting

For each 3D point, look up the MaskCLIP patch embedding of its source pixel.
`lift_features` runs the extractor frame-by-frame and maps pixel→patch via integer
division. Expect ~1–2 min on GPU for 40 frames at default 1024px resolution.
```

**Cell 11 — Code: lift_features**
```python
features = lift_features(images, pixel_indices, extractor_name="maskclip", device=DEVICE)
# features: (P, 768) float32, L2-normalized — ready for cosine similarity
print(f"features: {features.shape}  [{features.min():.3f}, {features.max():.3f}]")
```

**Cell 12 — Markdown: §4 Text Queries → Per-Point Scores**
```markdown
## §4  Text Queries → Per-Point Semantic Scores

Encode query strings with MaskCLIP's text encoder, then score each 3D point
via temperature-scaled contrastive softmax against the negative `"background"`.

- `scores` (P,) — contrastive score in [0, 1]; the dominant positive query wins.
- `per_query` (P, Q) — raw cosine similarity per query, for the gallery in §6.

Note: `score_queries` expects 2D `(C, H, W)` feature maps; for point arrays `(P, D)`
we compute raw sims manually and pass to `compute_semantic_contrast` directly.
```

**Cell 13 — Code: encode + score**
```python
extractor = MaskCLIPExtractor(device=DEVICE)

pos_embs = extractor.encode_text(QUERIES).cpu()    # (Q, 768)
neg_embs = extractor.encode_text(NEGATIVES).cpu()  # (1, 768)
all_embs = torch.cat([pos_embs, neg_embs], dim=0)  # (Q+1, 768)

feat_t   = torch.from_numpy(features)              # (P, 768) on CPU
raw_sims = feat_t @ all_embs.T                     # (P, Q+1)

# compute_semantic_contrast expects (N_queries, P)
scores = compute_semantic_contrast(
    raw_sims.T,
    num_positive=len(QUERIES),
    temperature=0.05,
    reduction="max",
).numpy()  # (P,)

per_query = (feat_t @ pos_embs.T).numpy()  # (P, Q) raw cosine sims

for i, q in enumerate(QUERIES):
    print(f"  {q:15s}: mean={per_query[:, i].mean():.3f}  max={per_query[:, i].max():.3f}")
print(f"  contrastive:    mean={scores.mean():.3f}  max={scores.max():.3f}")
```

**Cell 14 — Markdown: §5 Interactive Viewer**
```markdown
## §5  Interactive 3D Viewer

Use the scalar dropdown in the PyVista side-panel to switch between:
- `semantic` — contrastive score (best query wins)
- `tree`, `bird_feeder`, `ground` — per-query raw cosine similarity
```

**Cell 15 — Code: Interactive viewer**
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
pl.add_title("Tip: switch scalars in the side panel", font_size=9)
pl.show()
```

**Cell 16 — Markdown: §6 Multi-Query Gallery**
```markdown
## §6  Multi-Query Gallery

Tiled view: one panel per query, cameras linked. Brighter patches indicate stronger
feature match. Compare spatial footprints of `tree` (tall), `ground` (low horizontal),
and `bird feeder` (small localized cluster).
```

**Cell 17 — Code: Tiled gallery**
```python
n = len(QUERIES)
pl = pv.Plotter(shape=(1, n), title="Multi-Query Semantic Gallery")

for i, q in enumerate(QUERIES):
    key = q.replace(" ", "_")
    pl.subplot(0, i)
    pl.add_mesh(cloud.copy(), scalars=key, cmap="plasma", point_size=2)
    pl.add_title(f'"{q}"', font_size=10)

pl.link_views()
pl.show()
```

- [ ] **Step 2: Commit the notebook and design doc**

```bash
git add docs/source/tutorials/semantics/semantic_lifting.ipynb \
        docs/superpowers/specs/2026-05-21-semantic-lifting-notebook-design.md \
        docs/superpowers/plans/2026-05-21-semantic-lifting-notebook.md
git commit -m "feat(tutorials): add semantic lifting notebook (MaskCLIP → VGGT-X pointcloud)"
```

---

### Task 2: Verification

**Files:**
- Read: `docs/source/tutorials/semantics/semantic_lifting.ipynb`

- [ ] **Step 1: Validate notebook JSON is well-formed**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
with open('docs/source/tutorials/semantics/semantic_lifting.ipynb') as f:
    nb = json.load(f)
print(f'nbformat: {nb[\"nbformat\"]}.{nb[\"nbformat_minor\"]}')
print(f'cells: {len(nb[\"cells\"])}')
for c in nb['cells']:
    print(f'  [{c[\"cell_type\"]}] id={c[\"id\"]}')
"
```

Expected: 17 cells listed, no JSON parse errors.

- [ ] **Step 2: Verify imports resolve**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.utils.frame_sampling import sample_frames_optical_flow
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
from collab_splats.pointcloud.utils import lift_features
from collab_splats.semantics.features import MaskCLIPExtractor
from collab_splats.semantics.utils import compute_semantic_contrast
from collab_splats.utils.visualization import pointcloud_to_polydata
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Verify scoring shapes with synthetic data**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import torch, numpy as np
from collab_splats.semantics.features import MaskCLIPExtractor
from collab_splats.semantics.utils import compute_semantic_contrast

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ext = MaskCLIPExtractor(device=device)

QUERIES   = ['tree', 'bird feeder', 'ground']
NEGATIVES = ['background']

pos_embs = ext.encode_text(QUERIES).cpu()
neg_embs = ext.encode_text(NEGATIVES).cpu()
all_embs = torch.cat([pos_embs, neg_embs], dim=0)

P = 10000
features = torch.randn(P, 768)
features = features / features.norm(dim=-1, keepdim=True)

raw_sims = features @ all_embs.T
scores = compute_semantic_contrast(raw_sims.T, num_positive=len(QUERIES), temperature=0.05).numpy()
per_q  = (features @ pos_embs.T).numpy()

assert scores.shape == (P,),   f'scores shape {scores.shape}'
assert per_q.shape  == (P, 3), f'per_q shape {per_q.shape}'
assert scores.min() >= 0 and scores.max() <= 1, f'scores out of [0,1]: {scores.min():.3f} {scores.max():.3f}'
print(f'shapes OK: scores={scores.shape}, per_query={per_q.shape}')
print(f'score range [{scores.min():.3f}, {scores.max():.3f}]')
"
```

Expected: `shapes OK: scores=(10000,), per_query=(10000, 3)` and score range within [0, 1].
