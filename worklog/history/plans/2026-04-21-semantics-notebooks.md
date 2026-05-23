# Semantics Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pca_to_rgb`, `compute_masked_image`, and `overlay_masks` to `visualization.py`; rename registry key `"samclip"` → `"maskclip"`; create `docs/semantics/feature_extraction.ipynb` and `docs/semantics/segmentation.ipynb`.

**Architecture:** Registry rename propagates through features.py, datamanager, tests, and docstrings. Three new viz helpers are pure numpy/sklearn functions added under a new `## Semantic Visualization` section header. Notebooks use `SplatterConfig` + `sample_frames_fps(..., max_frames=1)` for video input, identical 3-cell blocks per extractor/strategy.

**Tech Stack:** Python, PyTorch, sklearn (PCA), matplotlib, numpy, Jupyter notebooks.

---

## File Map

| File | Action |
|------|--------|
| `collab_splats/semantics/features.py` | Change `@register("samclip")` → `@register("maskclip")` |
| `collab_splats/nerfstudio/datamanagers/features.py` | Change `Literal["samclip"]` → `Literal["maskclip"]` |
| `stage/feedforward.py` | Update docstring references `"samclip"` → `"maskclip"` |
| `tests/test_models.py` | Update `"samclip"` → `"maskclip"` in two places |
| `collab_splats/utils/visualization.py` | Add section headers + 3 new functions |
| `tests/test_visualization.py` | Create — tests for the 3 new viz functions |
| `docs/semantics/feature_extraction.ipynb` | Create — feature extraction notebook |
| `docs/semantics/segmentation.ipynb` | Create — segmentation notebook |

---

### Task 1: Rename registry key `"samclip"` → `"maskclip"`

**Files:**
- Modify: `collab_splats/semantics/features.py:87`
- Modify: `collab_splats/nerfstudio/datamanagers/features.py:41`
- Modify: `tests/test_models.py:27,29`
- Modify: `stage/feedforward.py` (docstrings only)

- [ ] **Step 1: Update registry decorator in features.py**

In `collab_splats/semantics/features.py`, change line 87:
```python
# before
@BaseFeatureExtractor.register("samclip")
class MaskCLIPExtractor(BaseFeatureExtractor):

# after
@BaseFeatureExtractor.register("maskclip")
class MaskCLIPExtractor(BaseFeatureExtractor):
```

- [ ] **Step 2: Update Literal type in datamanager**

In `collab_splats/nerfstudio/datamanagers/features.py`, line 41:
```python
# before
main_features: Literal["samclip"] = "samclip"

# after
main_features: Literal["maskclip"] = "maskclip"
```

- [ ] **Step 3: Update test_models.py**

In `tests/test_models.py`, update lines 27 and 29:
```python
# before (lines 27-29)
"feature_type": "samclip",
...
    "samclip": (channels, height, width),

# after
"feature_type": "maskclip",
...
    "maskclip": (channels, height, width),
```

- [ ] **Step 4: Update stage/feedforward.py docstrings**

In `stage/feedforward.py`, replace all 7 occurrences of `"samclip"` with `"maskclip"` in docstrings (lines 803, 809, 836, 1055, 1081, 1292, 1314, 1331). These are string literals in docstrings, not code — use find-and-replace. Note line 1331 contains a user-facing error message:
```python
# before
f"query_pointcloud requires a CLIP-based extractor (e.g. 'samclip'). "
# after
f"query_pointcloud requires a CLIP-based extractor (e.g. 'maskclip'). "
```

- [ ] **Step 5: Run existing tests to verify rename doesn't break anything**

```bash
cd /workspace/collab-splats && python -m pytest tests/test_models.py -v 2>&1 | head -40
```
Expected: all tests that ran before still pass. If `test_models.py` requires GPU and skips, that's fine.

- [ ] **Step 6: Verify registry key via Python**

```bash
cd /workspace/collab-splats && python -c "
from collab_splats.semantics.features import BaseFeatureExtractor
keys = list(BaseFeatureExtractor._registry.keys())
assert 'maskclip' in keys, f'maskclip not found, got {keys}'
assert 'samclip' not in keys, f'samclip still present'
print('OK:', keys)
"
```
Expected output: `OK: ['maskclip', 'dinov2', 'talk2dino']`

- [ ] **Step 7: Commit**

```bash
git add collab_splats/semantics/features.py \
        collab_splats/nerfstudio/datamanagers/features.py \
        tests/test_models.py \
        stage/feedforward.py
git commit -m "refactor(semantics): rename registry key samclip -> maskclip"
```

---

### Task 2: Add section headers and visualization helpers

**Files:**
- Modify: `collab_splats/utils/visualization.py`
- Create: `tests/test_visualization.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_visualization.py`:
```python
import numpy as np
import pytest
import torch


def make_features(C=32, pH=14, pW=14):
    return torch.randn(C, pH, pW)


def make_image(H=224, W=224):
    return (np.random.rand(H, W, 3) * 255).astype(np.uint8)


def test_pca_to_rgb_output_shape():
    from collab_splats.utils.visualization import pca_to_rgb
    features = make_features()
    image = make_image()
    result = pca_to_rgb(features, image)
    assert result.shape == image.shape, f"Expected {image.shape}, got {result.shape}"
    assert result.dtype == np.uint8


def test_pca_to_rgb_values_in_range():
    from collab_splats.utils.visualization import pca_to_rgb
    result = pca_to_rgb(make_features(), make_image())
    assert result.min() >= 0 and result.max() <= 255


def test_compute_masked_image_shape():
    from collab_splats.utils.visualization import compute_masked_image
    image = make_image()
    sim_map = np.random.rand(image.shape[0], image.shape[1]).astype(np.float32)
    result = compute_masked_image(image, sim_map)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_compute_masked_image_blacks_low_sim():
    from collab_splats.utils.visualization import compute_masked_image
    image = (np.ones((4, 4, 3)) * 200).astype(np.uint8)
    # all zeros sim_map → all pixels below threshold → all black
    sim_map = np.zeros((4, 4), dtype=np.float32)
    result = compute_masked_image(image, sim_map, threshold=0.5)
    assert result.sum() == 0, "All pixels should be black"


def test_compute_masked_image_keeps_high_sim():
    from collab_splats.utils.visualization import compute_masked_image
    image = (np.ones((4, 4, 3)) * 200).astype(np.uint8)
    # all ones sim_map → all pixels above threshold → image unchanged
    sim_map = np.ones((4, 4), dtype=np.float32)
    result = compute_masked_image(image, sim_map, threshold=0.5)
    np.testing.assert_array_equal(result, image)


def test_overlay_masks_output_shape():
    from collab_splats.utils.visualization import overlay_masks
    image = make_image(H=64, W=64)
    masks = torch.zeros(3, 64, 64)
    masks[0, :32, :32] = 1.0
    masks[1, :32, 32:] = 1.0
    masks[2, 32:, :] = 1.0
    result = overlay_masks(image, masks)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_overlay_masks_values_in_range():
    from collab_splats.utils.visualization import overlay_masks
    image = make_image(H=32, W=32)
    masks = torch.ones(1, 32, 32)
    result = overlay_masks(image, masks)
    assert result.min() >= 0 and result.max() <= 255
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/test_visualization.py -v 2>&1 | head -30
```
Expected: all 7 tests FAIL with `ImportError` or `cannot import name`.

- [ ] **Step 3: Add section headers and implement the three functions**

In `collab_splats/utils/visualization.py`, add `# ── Semantic Visualization ──` header before `compute_heatmap` and `# ── 3D Visualization ──` header before `visualize_splat`. Then add the three new functions after `query_heatmap`:

```python
# ── Semantic Visualization ──────────────────────────────────────────────────


def compute_heatmap(   # (already exists — no change)
    ...
```

```python
# ── 3D Visualization ────────────────────────────────────────────────────────


def visualize_splat(   # (already exists — no change)
    ...
```

Add after `query_heatmap` (still in `## Semantic Visualization` section):

```python
def pca_to_rgb(
    features: "torch.Tensor",
    image: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Project (C, pH, pW) feature tensor to RGB via PCA, blend over image.

    Args:
        features: Feature tensor of shape (C, pH, pW).
        image: Original image (H, W, 3) uint8.
        alpha: Blend weight for PCA overlay (0=image only, 1=PCA only).

    Returns:
        Blended image (H, W, 3) uint8.
    """
    import torch
    from sklearn.decomposition import PCA
    import cv2

    if isinstance(features, torch.Tensor):
        feat_np = features.detach().cpu().float().numpy()  # (C, pH, pW)
    else:
        feat_np = features

    C, pH, pW = feat_np.shape
    flat = feat_np.reshape(C, -1).T  # (pH*pW, C)

    pca = PCA(n_components=3)
    projected = pca.fit_transform(flat)  # (pH*pW, 3)

    # Normalize each channel to [0, 1]
    for i in range(3):
        col = projected[:, i]
        col_min, col_max = col.min(), col.max()
        projected[:, i] = (col - col_min) / (col_max - col_min + 1e-8)

    pca_img = projected.reshape(pH, pW, 3)  # (pH, pW, 3)
    pca_img = (pca_img * 255).astype(np.uint8)

    H, W = image.shape[:2]
    pca_resized = cv2.resize(pca_img, (W, H), interpolation=cv2.INTER_LINEAR)

    blended = (1 - alpha) * image.astype(float) + alpha * pca_resized.astype(float)
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_masked_image(
    image: np.ndarray,
    sim_map: Union["torch.Tensor", np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray:
    """Mask image to black where similarity is below threshold.

    Args:
        image: Original image (H, W, 3) uint8.
        sim_map: Similarity map (H, W) or (H, W, 1), float in [0, 1].
        threshold: Pixels with sim < threshold are set to black.

    Returns:
        Masked image (H, W, 3) uint8.
    """
    import torch
    import cv2

    if isinstance(sim_map, torch.Tensor):
        sim_map = sim_map.detach().cpu().numpy()
    sim_map = np.squeeze(sim_map)  # (H, W)

    H, W = image.shape[:2]
    if sim_map.shape != (H, W):
        sim_map = cv2.resize(sim_map, (W, H), interpolation=cv2.INTER_LINEAR)

    mask = sim_map >= threshold  # (H, W) bool
    result = image.copy()
    result[~mask] = 0
    return result


def overlay_masks(
    image: np.ndarray,
    masks: "torch.Tensor",
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay segmentation masks on image with distinct colors per mask.

    Args:
        image: Original image (H, W, 3) uint8.
        masks: Binary masks (N, H, W) float32, one per detected object.
        alpha: Blend weight for mask colors (0=image only, 1=colors only).

    Returns:
        Blended image (H, W, 3) uint8.
    """
    import torch
    import cv2

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()  # (N, H, W)
    else:
        masks_np = masks

    N, mH, mW = masks_np.shape
    H, W = image.shape[:2]

    cmap = plt.get_cmap("tab20")
    overlay = image.astype(float).copy()

    for i in range(N):
        mask = masks_np[i]  # (mH, mW)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        color = np.array(cmap(i % 20)[:3]) * 255  # RGB in [0,255]
        overlay[mask > 0.5] = (
            (1 - alpha) * overlay[mask > 0.5] + alpha * color
        )

    return np.clip(overlay, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && python -m pytest tests/test_visualization.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/visualization.py tests/test_visualization.py
git commit -m "feat(visualization): add pca_to_rgb, compute_masked_image, overlay_masks with section headers"
```

---

### Task 3: Create `docs/semantics/feature_extraction.ipynb`

**Files:**
- Create: `docs/semantics/feature_extraction.ipynb`

- [ ] **Step 1: Create docs/semantics/ directory**

```bash
mkdir -p /workspace/collab-splats/docs/semantics
```

- [ ] **Step 2: Create the notebook**

Create `docs/semantics/feature_extraction.ipynb` with the following cells in order. Use `nbformat` JSON structure (each cell is a dict with `cell_type`, `source`, `metadata`, `outputs`).

**Cell 0 — Markdown: title**
```markdown
# Feature Extraction

Demonstrates patch-level feature extraction and text-conditioned semantic queries using the `collab_splats.semantics` module.

Two extractors are shown:
- **MaskCLIP** (`"maskclip"`) — patch CLIP features via `maskclip_onnx`
- **Talk2DINO** (`"talk2dino"`) — DINOv3 + CLIP projection, text queries via HuggingFace Hub

Each section: load extractor → extract features → visualize (PCA RGB | similarity heatmap | masked image).
```

**Cell 1 — Code: setup**
```python
from pathlib import Path
from collab_splats.wrapper import SplatterConfig
from collab_splats.semantics import sample_frames_fps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Edit this path ──────────────────────────────────────────────────────────
config = SplatterConfig(file_path=Path("/path/to/video.mp4"))
# ────────────────────────────────────────────────────────────────────────────

frames = sample_frames_fps(str(config["file_path"]), fps=1, max_frames=1)
frame = frames[0]  # np.ndarray HxWx3

plt.imshow(frame)
plt.axis("off")
plt.title("First frame")
plt.show()
```

**Cell 2 — Code: show registry**
```python
from collab_splats.semantics.features import BaseFeatureExtractor

print("Registered extractors:", list(BaseFeatureExtractor._registry.keys()))
# → ['maskclip', 'dinov2', 'talk2dino']
# dinov2 provides pure patch features (no text query) — not covered in this notebook.
```

**Cell 3 — Markdown: MaskCLIP section**
```markdown
## MaskCLIP

Patch-level CLIP features. Supports text query via `compute_similarity()`.
```

**Cell 4 — Code: MaskCLIP load**
```python
from collab_splats.semantics.features import MaskCLIPExtractor

extractor_clip = MaskCLIPExtractor(device="cpu")  # change to "cuda" if available
print("MaskCLIP loaded. Patch size:", extractor_clip.patch_size)
```

**Cell 5 — Code: MaskCLIP extract**
```python
pil_frame = Image.fromarray(frame)
features_clip = extractor_clip.forward([pil_frame])  # list of (C, pH, pW)
print("Feature shape:", features_clip[0].shape)  # e.g. (512, 24, 24)
```

**Cell 6 — Code: MaskCLIP visualize**
```python
from collab_splats.utils.visualization import pca_to_rgb, compute_masked_image, compute_heatmap

positive = ["bird", "animal"]
negative = ["background", "sky", "ground"]

sim_map_clip = extractor_clip.compute_similarity(
    features_clip[0], positive=positive, negative=negative
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(pca_to_rgb(features_clip[0], frame))
axes[0].set_title("PCA → RGB")
axes[1].imshow(compute_heatmap(frame, sim_map_clip))
axes[1].set_title(f"Similarity: {positive}")
axes[2].imshow(compute_masked_image(frame, sim_map_clip))
axes[2].set_title("Masked Image")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

**Cell 7 — Markdown: Talk2DINO section**
```markdown
## Talk2DINO

DINOv3 backbone with CLIP text projection. Patch features + text-conditioned similarity.

`preprocess()` center-crops to square before `forward()` — required by Talk2DINO.
```

**Cell 8 — Code: Talk2DINO load**
```python
from collab_splats.semantics.features import Talk2DinoExtractor

extractor_t2d = Talk2DinoExtractor(
    hf_model_id="lorebianchi98/Talk2DINOv3-ViTB",
    device="cpu",  # change to "cuda" if available
)
print("Talk2DINO loaded. Patch size:", extractor_t2d.patch_size)
```

**Cell 9 — Code: Talk2DINO extract**
```python
# preprocess() center-crops to square — required by Talk2DINO
preprocessed = extractor_t2d.preprocess(pil_frame)
features_t2d = extractor_t2d.forward([preprocessed])  # list of (N_patches, D)
print("Feature shape:", features_t2d[0].shape)
```

**Cell 10 — Code: Talk2DINO visualize**
```python
import torch

positive = ["bird", "animal"]
negative = ["background", "sky", "ground"]

sim_t2d = extractor_t2d.compute_similarity(
    features_t2d[0], positive=positive, negative=negative
)

# Talk2DINO returns flat (N_patches,) similarity scores.
# Reshape to (pH, pW) for visualization using the cropped image dimensions.
sq_size = preprocessed.size[0]  # PIL Image .size = (W, H)
pH = pW = sq_size // extractor_t2d.patch_size
sim_t2d_2d = sim_t2d.reshape(pH, pW).unsqueeze(-1)  # (pH, pW, 1)

sq_frame = np.array(preprocessed)  # use cropped frame to match sim_map dims

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# pca_to_rgb expects (C, pH, pW) — reshape from (N_patches, D) → (D, pH, pW)
feat_chw = features_t2d[0].T.reshape(-1, pH, pW)  # (D, pH, pW)
axes[0].imshow(pca_to_rgb(feat_chw, sq_frame))
axes[0].set_title("PCA → RGB")
axes[1].imshow(compute_heatmap(sq_frame, sim_t2d_2d))
axes[1].set_title(f"Similarity: {positive}")
axes[2].imshow(compute_masked_image(sq_frame, sim_t2d_2d))
axes[2].set_title("Masked Image")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Commit**

```bash
git add docs/semantics/feature_extraction.ipynb
git commit -m "docs(semantics): add feature_extraction notebook"
```

---

### Task 4: Create `docs/semantics/segmentation.ipynb`

**Files:**
- Create: `docs/semantics/segmentation.ipynb`

- [ ] **Step 1: Create the notebook**

Create `docs/semantics/segmentation.ipynb` with the following cells:

**Cell 0 — Markdown: title**
```markdown
# Segmentation

Demonstrates object segmentation using `collab_splats.semantics.Segmentation` with two strategies:
- **object** — YOLOv8 bounding boxes → SAM masks (recommended for distinct objects)
- **auto** — SAM automatic mask generator (denser coverage)

Closes with **Masks → Features**: how to aggregate per-segment feature vectors using `aggregate_masked_features`, bridging segmentation and feature extraction.
```

**Cell 1 — Code: setup**
```python
from pathlib import Path
from collab_splats.wrapper import SplatterConfig
from collab_splats.semantics import sample_frames_fps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Edit this path ──────────────────────────────────────────────────────────
config = SplatterConfig(file_path=Path("/path/to/video.mp4"))
# ────────────────────────────────────────────────────────────────────────────

frames = sample_frames_fps(str(config["file_path"]), fps=1, max_frames=1)
frame = frames[0]  # np.ndarray HxWx3

plt.imshow(frame)
plt.axis("off")
plt.title("First frame")
plt.show()
```

**Cell 2 — Markdown: object strategy**
```markdown
## Object Strategy

Uses YOLOv8 to detect bounding boxes, then SAM to produce per-object masks.
Best for scenes with distinct foreground objects.
```

**Cell 3 — Code: object strategy load**
```python
from collab_splats.semantics.segmentation import Segmentation

seg_object = Segmentation(backend="mobilesamv2", strategy="object", device="cpu")
print("Segmentation model loaded.")
```

**Cell 4 — Code: object strategy segment**
```python
result_object = seg_object.segment(frame)
if result_object is not None:
    masks_object, metadata_object = result_object  # masks: (N, H, W) float32
    print(f"{len(metadata_object)} objects detected")
else:
    print("No objects detected.")
    masks_object = None
```

**Cell 5 — Code: object strategy visualize**
```python
from collab_splats.utils.visualization import overlay_masks

if masks_object is not None:
    plt.imshow(overlay_masks(frame, masks_object))
    plt.axis("off")
    plt.title(f"Object Strategy — {len(metadata_object)} masks")
    plt.show()
```

**Cell 6 — Markdown: auto strategy**
```markdown
## Auto Strategy

SAM automatic mask generator without object detection priors.
Produces denser, smaller segments — useful for fine-grained coverage.
```

**Cell 7 — Code: auto strategy load**
```python
seg_auto = Segmentation(backend="mobilesamv2", strategy="auto", device="cpu")
print("Auto segmentation model loaded.")
```

**Cell 8 — Code: auto strategy segment**
```python
result_auto = seg_auto.segment(frame)
if result_auto is not None:
    masks_auto, metadata_auto = result_auto
    print(f"{len(metadata_auto)} segments detected")
else:
    print("No segments detected.")
    masks_auto = None
```

**Cell 9 — Code: auto strategy visualize**
```python
if masks_auto is not None:
    plt.imshow(overlay_masks(frame, masks_auto))
    plt.axis("off")
    plt.title(f"Auto Strategy — {len(metadata_auto)} masks")
    plt.show()
```

**Cell 10 — Markdown: masks → features**
```markdown
## Masks → Features

`aggregate_masked_features` pools feature vectors per mask region.
This bridges segmentation and feature extraction: instead of per-pixel features,
you get one feature vector per detected object.

`MaskCLIPExtractor` is used here because it returns spatial `(C, pH, pW)` features
directly — the required input format for `aggregate_masked_features`.
```

**Cell 11 — Code: load extractor**
```python
from collab_splats.semantics.features import MaskCLIPExtractor

# MaskCLIPExtractor returns spatial (C, pH, pW) — required by aggregate_masked_features.
# Talk2DinoExtractor returns flat (N_patches, D) and would need an explicit reshape first.
extractor = MaskCLIPExtractor(device="cpu")
print("MaskCLIP loaded.")
```

**Cell 12 — Code: aggregate**
```python
from collab_splats.semantics.segmentation import aggregate_masked_features
from collab_splats.utils.visualization import pca_to_rgb

pil_frame = Image.fromarray(frame)
features = extractor.forward([pil_frame])  # list of (C, pH, pW)

H, W = frame.shape[:2]

# Use object strategy masks for aggregation
masks_for_agg = masks_object  # (N, H, W) float32

agg = aggregate_masked_features(
    features[0],
    masks_for_agg,
    resolution=(H // 4, W // 4),
    final_resolution=(H, W),
)
print("Aggregated features shape:", agg.shape)  # (C, H, W)
```

**Cell 13 — Code: visualize 3-column**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(frame)
axes[0].set_title("Original")
axes[1].imshow(overlay_masks(frame, masks_for_agg))
axes[1].set_title("Segmentation Masks")
axes[2].imshow(pca_to_rgb(agg, frame))
axes[2].set_title("Aggregated Features (PCA)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add docs/semantics/segmentation.ipynb
git commit -m "docs(semantics): add segmentation notebook"
```

---

## Self-Review

**Spec coverage:**
- ✅ Registry rename `"samclip"` → `"maskclip"` — Task 1
- ✅ `pca_to_rgb` — Task 2
- ✅ `compute_masked_image` — Task 2
- ✅ `overlay_masks` — Task 2
- ✅ Section headers in visualization.py — Task 2, Step 3
- ✅ `feature_extraction.ipynb` with MaskCLIP + Talk2DINO, 3-col viz, SplatterConfig setup — Task 3
- ✅ `segmentation.ipynb` with object + auto strategy + aggregate bridge — Task 4
- ✅ `dinov2` mentioned as footnote in feature_extraction — Task 3, Cell 2
- ✅ Tests for all 3 new viz functions — Task 2

**Type consistency:**
- `pca_to_rgb(features, image)` — same signature in tests and implementation
- `compute_masked_image(image, sim_map, threshold=0.5)` — consistent across tests and notebook
- `overlay_masks(image, masks, alpha=0.5)` — consistent throughout
- `aggregate_masked_features(features[0], masks, resolution, final_resolution)` — matches existing signature in `collab_splats/semantics/segmentation.py`

**Talk2DINO reshape note:** Cell 10 in `feature_extraction.ipynb` explicitly reshapes `(N_patches, D)` → `(D, pH, pW)` for `pca_to_rgb`. This is documented inline in the cell.
