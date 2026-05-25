# Semantic Lifting Notebook — Design

**Date:** 2026-05-21
**Status:** Approved

## Context

VGGT-X feedforward reconstruction stores `pixel_indices (P, 3)` — the source frame/pixel for every
3D point. `lift_features` already maps those pixels to patch-level MaskCLIP embeddings. This notebook
closes the loop: demonstrate text-queryable 3D semantics in a single interactive tutorial.

MaskCLIP was chosen over Talk2DINO (no model download needed for basic use) and DINOv2 (no text
support). Queries match those in `docs/source/tutorials/semantics/maskclip_reference_comparison.ipynb`
for consistency: `["tree", "bird feeder", "ground"]` with `["background"]` as negative.

**No library changes.** Notebook only.

---

## New File

`docs/source/tutorials/semantics/semantic_lifting.ipynb`

---

## Pipeline

```
video
  └─ sample_frames_optical_flow()  →  N keyframes saved to disk
         └─ VGGTXCreator pipeline   →  pts3d (P,3) + pixel_indices (P,3) + images (N,3,H,W)
                └─ lift_features()  →  features (P, 768)   ← MaskCLIP patch lookup per point
                       └─ encode_text() + compute_semantic_contrast()
                              └─ scores (P,)  →  PyVista colored pointcloud
```

## Scoring Logic

`score_queries` expects `(C, H, W)` (2D feature maps). For point-level `(P, D)` arrays, we
compute raw cosine similarities manually then call `compute_semantic_contrast`:

```python
raw_sims = feat_t @ all_embs.T   # (P, Q+1)
scores = compute_semantic_contrast(raw_sims.T, num_positive=Q, ...)  # (P,)
```

## Sections

| § | Title | Key call |
|---|-------|----------|
| 0 | Configuration | — |
| 1 | Keyframe Extraction | `sample_frames_optical_flow` |
| 2 | VGGT-X Reconstruction | `VGGTXCreator` pipeline |
| 3 | MaskCLIP Feature Lifting | `lift_features(..., "maskclip")` |
| 4 | Text Queries → Per-Point Scores | `encode_text` + `compute_semantic_contrast` |
| 5 | Interactive 3D Viewer | `pointcloud_to_polydata` + `pv.Plotter` |
| 6 | Multi-Query Gallery | `pv.Plotter(shape=(1, Q))` |

## Reused Utilities

| Utility | Path |
|---------|------|
| `sample_frames_optical_flow` | `collab_splats/utils/frame_sampling.py` |
| `VGGTXCreator` | `collab_splats/pointcloud/feedforward/vggtx.py` |
| `lift_features` | `collab_splats/pointcloud/utils.py` |
| `MaskCLIPExtractor` | `collab_splats/semantics/features.py` |
| `compute_semantic_contrast` | `collab_splats/semantics/utils.py` |
| `pointcloud_to_polydata` | `collab_splats/utils/visualization.py` |

## Verification

1. All cells run top-to-bottom without error
2. `features.shape == (len(pts3d), 768)`
3. `scores.shape == (len(pts3d),)` and values in `[0, 1]`
4. PyVista viewer launches; switching scalars recolors the cloud
5. Tiled gallery shows 3 distinct spatial patterns
