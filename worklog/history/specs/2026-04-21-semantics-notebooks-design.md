# Semantics Documentation Notebooks

**Date:** 2026-04-21  
**Branch:** refactor/core-modules  

## Goal

Two Jupyter notebooks under `docs/semantics/` that demonstrate the semantics module end-to-end: feature extraction with text queries, and segmentation with mask-to-feature aggregation.

## Scope

### Code changes
- Rename registry key `"samclip"` → `"maskclip"` in `collab_splats/semantics/features.py`
- Update all test references from `"samclip"` to `"maskclip"`
- Add three functions to `collab_splats/utils/visualization.py` under `## Semantic Visualization` section
- Add `## 3D Visualization` section header to delineate existing splat viz functions

### New notebooks
- `docs/semantics/feature_extraction.ipynb`
- `docs/semantics/segmentation.ipynb`

---

## visualization.py Changes

Add section headers and three new functions:

```
## Semantic Visualization       ← new header
compute_heatmap(...)            ← exists
plot_heatmap(...)               ← exists
query_heatmap(...)              ← exists
pca_to_rgb(features, image)     ← new
compute_masked_image(...)       ← new
overlay_masks(...)              ← new

## 3D Visualization             ← new header
visualize_splat(...)            ← exists
create_camera_frustum_pyvista(...)  ← exists
```

### `pca_to_rgb(features: torch.Tensor, image: np.ndarray) -> np.ndarray`
- Input: `(C, pH, pW)` feature tensor + original image `(H, W, 3)`
- sklearn PCA to 3 components, normalize each to [0,1], resize to image dims, blend over image
- Used in both notebooks for all extractors

### `compute_masked_image(image: np.ndarray, sim_map, threshold: float = 0.5) -> np.ndarray`
- Input: original image, similarity map `(H, W)` or `(H, W, 1)`, threshold
- Pixels below threshold set to black; high-similarity regions show original color
- Used in `feature_extraction.ipynb` as third column per extractor

### `overlay_masks(image: np.ndarray, masks: torch.Tensor, alpha: float = 0.5) -> np.ndarray`
- Input: image `(H, W, 3)`, masks `(N, H, W)` float32 tensor, blend alpha
- Assigns distinct color per mask index via colormap, blends over image
- Used in `segmentation.ipynb` for both strategy cells and aggregate cell

---

## `docs/semantics/feature_extraction.ipynb`

### Setup section
```python
from pathlib import Path
from collab_splats.wrapper import SplatterConfig
from collab_splats.semantics import sample_frames_fps

config = SplatterConfig(file_path=Path("/path/to/video.mp4"))
frames = sample_frames_fps(config["file_path"], fps=1, max_frames=1)
frame = frames[0]  # np.ndarray HxWx3
```
Then display frame. User edits `file_path` only.

Also show registry:
```python
from collab_splats.semantics.features import BaseFeatureExtractor
print(BaseFeatureExtractor._registry.keys())
# → dict_keys(['maskclip', 'dinov2', 'talk2dino'])
```

Note: `dinov2` available for pure patch features only (no text query); not covered in this notebook.

### MaskCLIP section (3 cells, identical block structure)

**Cell 1 — Load:**
```python
from collab_splats.semantics.features import MaskCLIPExtractor
extractor = MaskCLIPExtractor(device="cpu")  # change to "cuda" if available
```

**Cell 2 — Extract:**
```python
from PIL import Image
pil_frame = Image.fromarray(frame)
features = extractor.forward([pil_frame])  # list of (C, pH, pW)
```

**Cell 3 — Visualize (3 columns):**
```python
import matplotlib.pyplot as plt
from collab_splats.utils.visualization import pca_to_rgb, compute_masked_image, compute_heatmap

positive = ["bird", "animal"]
negative = ["background", "sky", "ground"]

sim_map = extractor.compute_similarity(features[0], positive=positive, negative=negative)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(pca_to_rgb(features[0], frame)); axes[0].set_title("PCA → RGB")
axes[1].imshow(compute_heatmap(frame, sim_map)); axes[1].set_title("Similarity Heatmap")
axes[2].imshow(compute_masked_image(frame, sim_map)); axes[2].set_title("Masked Image")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
```

### Talk2DINO section (identical 3-cell block)

Same structure. `Talk2DinoExtractor` requires square-cropped input via `extractor.preprocess(pil_frame)` before `forward`.

---

## `docs/semantics/segmentation.ipynb`

### Setup section
Same `SplatterConfig` + `sample_frames_fps(..., max_frames=1)` pattern.

### Object Strategy section (3 cells)

**Cell 1 — Load:**
```python
from collab_splats.semantics.segmentation import Segmentation
seg = Segmentation(backend="mobilesamv2", strategy="object", device="cpu")
```

**Cell 2 — Segment:**
```python
result = seg.segment(frame)
if result is not None:
    masks, metadata = result  # masks: (N, H, W) float32
    print(f"{len(metadata)} objects detected")
```

**Cell 3 — Visualize:**
```python
from collab_splats.utils.visualization import overlay_masks
plt.imshow(overlay_masks(frame, masks)); plt.axis("off"); plt.show()
```

### Auto Strategy section (identical 3-cell block, `strategy="auto"`)

### Masks → Features section

Shows how segmentation output feeds into feature aggregation — the bridge between the two notebooks.

**Cell 1 — Load extractor:**
```python
# MaskCLIPExtractor used here: returns spatial (C, pH, pW) directly,
# which aggregate_masked_features requires. Talk2DinoExtractor returns flat
# (N_patches, D) and would need an explicit reshape first.
from collab_splats.semantics.features import MaskCLIPExtractor
extractor = MaskCLIPExtractor(device="cpu")
```

**Cell 2 — Aggregate:**
```python
from collab_splats.semantics.segmentation import aggregate_masked_features
from collab_splats.utils.visualization import pca_to_rgb

pil_frame = Image.fromarray(frame)
features = extractor.forward([pil_frame])  # list of (C, pH, pW)
H, W = frame.shape[:2]
agg = aggregate_masked_features(features[0], masks, resolution=(H//4, W//4), final_resolution=(H, W))
```

**Cell 3 — Visualize (3 columns):**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(frame); axes[0].set_title("Original")
axes[1].imshow(overlay_masks(frame, masks)); axes[1].set_title("Segmentation Masks")
axes[2].imshow(pca_to_rgb(agg, frame)); axes[2].set_title("Aggregated Features")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
```

---

## Registry Rename

`collab_splats/semantics/features.py`:
```python
# before
@BaseFeatureExtractor.register("samclip")
class MaskCLIPExtractor(BaseFeatureExtractor):

# after
@BaseFeatureExtractor.register("maskclip")
class MaskCLIPExtractor(BaseFeatureExtractor):
```

Update all test files referencing `"samclip"` → `"maskclip"`.

---

## File Structure

```
docs/
  semantics/
    feature_extraction.ipynb   ← new
    segmentation.ipynb         ← new
collab_splats/
  semantics/
    features.py                ← rename "samclip" → "maskclip"
  utils/
    visualization.py           ← add sections + 3 functions
tests/
  (any test referencing "samclip" → "maskclip")
```
