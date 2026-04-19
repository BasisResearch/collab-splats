# Semantics Module

`collab_splats.semantics` provides feature extraction, segmentation, and text-conditioned semantic queries. Use it to explore image semantics before (or without) fitting a splat model.

---

## Feature Extractors

All extractors share a registry. Look up by name or import directly.

```python
from collab_splats.semantics.features import BaseFeatureExtractor

# list registered extractors
print(BaseFeatureExtractor._registry.keys())
# → ['samclip', 'clip-vit', 'dinov2', 'talk2dino']

# instantiate by name
cls = BaseFeatureExtractor.get("dinov2")
extractor = cls(device="cuda")
```

### MaskCLIPExtractor (`"samclip"`)

Patch-level CLIP features via `maskclip_onnx`.

```python
from collab_splats.semantics.features import MaskCLIPExtractor
from PIL import Image

extractor = MaskCLIPExtractor(model_name="ViT-L/14@336px", device="cuda")

img = Image.open("frame.jpg").convert("RGB")
tensor = extractor.preprocess(img, resolution=1024)          # (3, H, W)
features = extractor.forward(tensor.unsqueeze(0))            # (1, C, H/p, W/p)

# text similarity
text_emb = extractor.encode_text(["a chair", "a table"])    # (2, D)

# similarity heatmap
sim = extractor.compute_similarity(
    features[0],                # (C, H, W)
    positive=["a chair"],
    negative=["background"],
    softmax_temp=0.05,
    method="standard",          # or "pairwise"
)  # → (H, W, 1)
```

### DINOFeatureExtractor (`"dinov2"`)

Patch tokens from DINOv2 via `torch.hub`.

```python
from collab_splats.semantics.features import DINOFeatureExtractor

extractor = DINOFeatureExtractor(model_name="dinov2_vits14", device="cuda")

tensor, H, W = extractor.preprocess(img)   # patches to patch-aligned resolution
tokens = extractor.forward(tensor)         # (N_patches, D)
feat_chw = extractor.reshape(tokens, H, W) # (D, H/p, W/p)
```

### Talk2DinoExtractor (`"talk2dino"`)

DINOv3 + CLIP projection from HuggingFace Hub. Key capability: text-conditioned semantic heatmaps.

```python
from collab_splats.semantics.features import Talk2DinoExtractor

extractor = Talk2DinoExtractor(
    hf_model_id="lorebianchi98/Talk2DINOv3-ViTB",  # or Talk2DINO-ViTB for DINOv2
    device="cuda",
)

# patch features (same interface as DINOFeatureExtractor)
img_sq = extractor.preprocess(img)         # center-crop to square
tokens = extractor.forward(img_sq)         # (N_patches, D)

# text embeddings
text_emb = extractor.encode_text(["a feeder", "a tree"])  # (2, D), normalized

# semantic heatmaps — the unique capability
text_pairs = {
    "feeder": (
        ["feeder", "bird feeder", "wooden feeder"],
        ["background", "sky", "blur"],
    ),
    "tree": (
        ["tree", "trunk", "branch"],
        ["sky", "ground", "feeder"],
    ),
}
heatmaps = extractor.compute_semantic_heatmap(
    img,
    text_pairs,
    softmax_temp=0.05,   # lower = sharper masks
    method="standard",   # "standard" | "pairwise"
)
# heatmaps["feeder"] → np.ndarray (H, W, 3), float32, image masked by similarity
# heatmaps["tree"]   → np.ndarray (H, W, 3), float32
```

**`method` options:**
- `"standard"` — softmax over all queries, sum positive probs. Good default.
- `"pairwise"` — average positive vs each negative independently, take min. Sharper boundaries when negatives are specific.

**Model variants:**
| HF model ID | Backbone | patch_size |
|---|---|---|
| `lorebianchi98/Talk2DINOv3-ViTB` | DINOv3 ViT-B | 14 |
| `lorebianchi98/Talk2DINO-ViTB` | DINOv2 ViT-B | 14 |

Both use `trust_remote_code=True` internally.

---

## Segmentation

```python
from collab_splats.semantics.segmentation import Segmentation
import numpy as np

seg = Segmentation(backend="mobilesamv2", strategy="object", device="cuda")

frame = np.array(img)           # HxWx3 uint8
result = seg.segment(frame)

if result is not None:
    masks, metadata = result    # masks: (N, H, W) float32 tensor
```

**strategies:**
- `"object"` — YOLOv8 bounding boxes → SAM masks. Better for distinct objects.
- `"auto"` — SAM automatic mask generator. Denser coverage.

---

## SupportsTextQuery Protocol

Check if an extractor supports text queries before using `compute_semantic_heatmap`:

```python
from collab_splats.semantics.protocols import SupportsTextQuery

if isinstance(extractor, SupportsTextQuery):
    heatmaps = extractor.compute_semantic_heatmap(img, text_pairs)
```

Currently satisfied by `Talk2DinoExtractor`. Any future extractor that implements `encode_text` + `compute_semantic_heatmap` with matching signatures automatically satisfies it — no inheritance needed.

---

## Dashboard

Interactive 4-tab Gradio app for exploring semantics before training.

**Launch:**
```bash
# after pip install -e .
collab-dashboard semantics

# or without install
python -m collab_splats.dashboard semantics

# custom host/port
collab-dashboard semantics --host 127.0.0.1 --port 8080
```

Opens at `http://localhost:7860`.

### Tab 1 — Load Video

1. Upload a video file
2. Set FPS to extract (1–30; lower = fewer frames)
3. Click **Extract Frames**
4. Use the slider to select the frame to work on

### Tab 2 — Feature Extraction

1. Select extractor from dropdown (all registered extractors appear automatically)
2. Select device (`cpu` / `cuda`)
3. Click **Extract Features**

Output: PCA projection of patch features to RGB, blended over the original frame. Useful for seeing what semantic regions the extractor distinguishes.

### Tab 3 — Segmentation

1. Select strategy (`object` recommended for scenes with distinct items)
2. Select device
3. Click **Segment**

Output: coloured mask overlay. Each detected region gets a distinct colour.

### Tab 4 — Semantic Query

Uses Talk2DINO. For each concept you want to locate, specify positive and negative text descriptions.

**Text pairs format (JSON):**
```json
{
    "feeder": [
        ["feeder", "bird feeder", "wooden feeder"],
        ["background", "sky", "blur", "bokeh"]
    ],
    "tree": [
        ["tree", "trunk", "branch"],
        ["sky", "ground", "feeder"]
    ]
}
```

1. Paste JSON into the text pairs box
2. Select model (`Talk2DINOv3-ViTB` default)
3. Tune **softmax temperature** — lower values (0.001–0.01) give sharper, more confident masks; higher (0.05–0.1) give softer coverage
4. Choose method (`standard` or `pairwise`)
5. Click **Generate Heatmaps**

Output: gallery with one masked image per label. Bright regions = high similarity to positive queries.

**Tips:**
- Start with `standard` method and `temp=0.05`
- Add specific negatives to reduce bleed into similar-looking regions
- `pairwise` helps when you have tight contrast between concepts (e.g. "feeder" vs "tree")
