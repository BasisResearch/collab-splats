# PE-CLIP + SAM3 Open-Label Segmentation Design

**Date:** 2026-05-23
**Status:** Approved

## Goal

Replicate VGGT-SLAM's `--run_os` (open-set semantic search) pipeline within `collab-splats`:

1. PE-CLIP global image embeddings for text-driven frame retrieval
2. SAM3 text-prompted segmentation → 2D masks
3. 3D OBB lifting from mask-filtered pointcloud

Additionally, refactor `Segmentation` string-dispatch class into a proper `BaseSegmentation` registry hierarchy consistent with existing codebase patterns.

---

## Context: VGGT-SLAM Reference Pipeline

```
for each frame:
    img_emb = pe_clip.encode_image(frame)       # (D,) global
    submap.semantic_vectors.append(img_emb)

# at query time:
text_emb = pe_clip.encode_text("red tractor")  # (D,)
best_frame = argmax(cosine_sim(all_img_embs, text_emb))

masks, boxes, scores = sam3.set_text_prompt(best_frame, "red tractor")
pts3d = get_points_in_mask(best_frame, masks[0], graph)
center, extent, rotation = compute_obb_from_points(pts3d)
```

---

## Key Technical Decisions

### PE-CLIP is NOT a patch-level extractor

`timm/PE-Core-L-14-336` open_clip config:

```json
"timm_pool": "map",
"timm_proj": null
```

`pool="map"` = `AttentionPoolLatent`: cross-attention collapses N patch tokens into 1 vector. No separable linear projection exists (`proj=null`). Patch tokens are in ViT latent space — not CLIP text space. Per-patch CLIP-aligned features are not possible.

**PE-CLIP belongs with `BaseRetrievalExtractor`** (global descriptors), not `BaseQueryableExtractor` (patch features). Architecturally identical to `DinoSaladExtractor`, but with text encoding added.

### `Segmentation` refactor

Current `Segmentation` class uses string dispatch — inconsistent with `BaseFeatureExtractor` / `BaseRetrievalExtractor` registry pattern. Refactored to `BaseSegmentation` + registry. No backwards-compat aliases: all callers updated directly.

---

## Component Design

### 1. `PECLIPExtractor` — `collab_splats/pointcloud/localization.py`

```python
@BaseRetrievalExtractor.register("pe-clip")
class PECLIPExtractor(BaseRetrievalExtractor):
    """PE-Core-L/14-336 global image+text encoder for open-label frame retrieval.

    Uses open_clip with hf-hub:timm/PE-Core-L-14-336. Produces (N, 1024)
    normalized descriptors for both images and text — aligned in the same
    CLIP embedding space.
    """
    _MODEL_ID = "hf-hub:timm/PE-Core-L-14-336"

    def __init__(self, model_id: str = _MODEL_ID, device: str = "cuda"):
        # open_clip.create_model_and_transforms(model_id) → model, _, preprocess
        # open_clip.get_tokenizer(model_id) → tokenizer
        # model is TimmModel: trunk=vit_pe_core_large_patch14_336, pool='map', proj=None

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """Return (N, 1024) normalized image descriptors."""
        # preprocess each image → stack → model.encode_image(normalize=True)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return (N, 1024) normalized text descriptors."""
        # tokenizer(texts, context_length=32) → model.encode_text(normalize=True)
```

**Usage:**
```python
retriever = PECLIPExtractor(device="cuda")
img_embs = retriever(frames)                       # (N, 1024)
text_emb = retriever.encode_text(["red tractor"]) # (1, 1024)
scores = img_embs @ text_emb.T                    # (N, 1)
best_idx = scores.argmax().item()
```

**Dependency:** `open-clip-torch` — add to `requirements.txt`.

---

### 2. `BaseSegmentation` hierarchy — `collab_splats/semantics/segmentation.py`

```python
class BaseSegmentation(RegistryMixin, ABC):
    """Abstract base for segmentation backends with name-based registry."""
    _registry: dict[str, type["BaseSegmentation"]] = {}

    @abstractmethod
    def segment(self, image) -> tuple[torch.Tensor, Any]:
        """Class-agnostic segmentation. Returns (masks, metadata)."""

    def segment_with_text(
        self, image, prompt: str, confidence_threshold: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Text-prompted segmentation → (masks, boxes, scores).
        Only SAM3 backend supports this; others raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support text-prompted segmentation. "
            "Use backend='sam3'."
        )


@BaseSegmentation.register("mobilesamv2")
class MobileSAMSegmentation(BaseSegmentation):
    """MobileSAMv2 class-agnostic segmentation (object or auto strategy)."""

    def __init__(
        self,
        strategy: str = "object",
        device: str = "cpu",
        mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2",
    ):
        # load_mobile_sam(mobilesam_encoder_name, device) → seg_model, object_model, predictor

    def segment(self, image) -> tuple[torch.Tensor, Any]:
        # delegates to object_segment_image or auto_segment_image per strategy


@BaseSegmentation.register("sam3")
class SAM3Segmentation(BaseSegmentation):
    """SAM3 segmentation: class-agnostic auto-segment + text-prompted masking."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = "cuda"):
        # build_sam3_image_model() → sam3_model
        # Sam3Processor(sam3_model, confidence_threshold=confidence_threshold)

    def segment(self, image) -> tuple[torch.Tensor, Any]:
        # auto-segment all objects (no text prompt)

    def segment_with_text(
        self, image, prompt: str, confidence_threshold: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # processor.set_image(image)
        # processor.set_text_prompt(state, prompt)
        # returns masks (N,1,H,W), boxes (N,4), scores (N,)
```

**Dependency:** `sam3` — add to `requirements.txt` (or note as optional heavy dep).

---

### 3. `compute_obb_from_points` — `collab_splats/pointcloud/utils.py`

Port directly from `third_party/VGGT-SLAM/vggt_slam/slam_utils.py`. PCA-based oriented bounding box:

```python
def compute_obb_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute oriented bounding box from Nx3 point cloud via PCA.

    Returns:
        center   : (3,) world-space OBB center
        extent   : (3,) box side lengths along principal axes
        rotation : (3,3) rotation matrix, columns = principal axes
    """
```

---

## Callsite Updates

All direct `Segmentation(...)` calls replaced with `MobileSAMSegmentation(...)` or `BaseSegmentation.get("mobilesamv2")(...)`. No backwards-compat aliases.

| File | Change |
|------|--------|
| `collab_splats/semantics/segmentation.py` | Full refactor: `BaseSegmentation`, `MobileSAMSegmentation`, `SAM3Segmentation` |
| `collab_splats/semantics/__init__.py` | Export `BaseSegmentation`, `MobileSAMSegmentation`, `SAM3Segmentation`; remove `Segmentation` |
| `collab_splats/nerfstudio/datamanagers/features.py` | `Segmentation(...)` → `MobileSAMSegmentation(...)` |
| `stage/grouping.py` | `Segmentation(...)` → `MobileSAMSegmentation(...)` |
| `docs/semantics/segmentation.ipynb` | Update imports and instantiation |
| `stage/extract-colmap-vggt.ipynb` | Update `segmentation_backend` usage |
| `collab_splats/pointcloud/localization.py` | Add `PECLIPExtractor` |
| `collab_splats/pointcloud/utils.py` | Add `compute_obb_from_points`, `get_points_in_mask` |
| `collab_splats/pointcloud/__init__.py` | Export `compute_obb_from_points`, `get_points_in_mask` |
| `requirements.txt` | Add `open-clip-torch` |

---

## Tests

### Update existing
- `tests/nerfstudio/test_datamanager_config.py` — patch target changes from `Segmentation` to `MobileSAMSegmentation`; update import path in patch string

### New tests — `tests/semantics/test_segmentation.py`
- `BaseSegmentation.get("mobilesamv2")` returns `MobileSAMSegmentation`
- `BaseSegmentation.get("sam3")` returns `SAM3Segmentation`
- `MobileSAMSegmentation().segment_with_text(...)` raises `NotImplementedError`
- Unknown backend raises `KeyError` from registry

### New tests — `tests/pointcloud/test_utils.py`
- `compute_obb_from_points` on axis-aligned cube → correct center, extent, identity-ish rotation
- Empty/NaN input raises `ValueError`

### New tests — `tests/pointcloud/test_localization.py`
- `BaseRetrievalExtractor.get("pe-clip")` returns `PECLIPExtractor`
- `PECLIPExtractor.forward([img])` → shape `(1, 1024)`, unit norm
- `PECLIPExtractor.encode_text(["cat"])` → shape `(1, 1024)`, unit norm

---

## End-to-End Usage

```python
from collab_splats.pointcloud.localization import PECLIPExtractor
from collab_splats.semantics import SAM3Segmentation
from collab_splats.pointcloud.utils import compute_obb_from_points

# 1. embed frames
retriever = PECLIPExtractor(device="cuda")
img_embs = retriever(frames)                          # (N, 1024)
text_emb = retriever.encode_text(["red tractor"])     # (1, 1024)

# 2. best matching frame
best_idx = (img_embs @ text_emb.T).argmax().item()

# 3. SAM3 text-prompted masks
seg = SAM3Segmentation(device="cuda")
masks, boxes, scores = seg.segment_with_text(frames[best_idx], "red tractor")

# 4. lift top mask to 3D OBB
pts3d = get_points_in_mask(best_idx, masks[0], pointcloud)
center, extent, rotation = compute_obb_from_points(pts3d)
```

---

## Component 4: `get_points_in_mask` — `collab_splats/pointcloud/utils.py`

`PointcloudResult` already stores `points (P, 3)` in world space and `pixel_indices (P, 3)` = [frame_id, row, col] per point. No depth unprojection needed — just a filter:

```python
def get_points_in_mask(
    frame_idx: int,
    mask: np.ndarray,           # (H, W) bool — e.g. from SAM3
    points: np.ndarray,         # (P, 3) world-space
    pixel_indices: np.ndarray,  # (P, 3) int32 [frame_id, row, col]
) -> np.ndarray:
    """Return world-space points whose source pixel falls within a 2D mask.

    Returns (M, 3) subset of points, where M ≤ P.
    """
    frame_mask = pixel_indices[:, 0] == frame_idx
    rows = pixel_indices[frame_mask, 1]
    cols = pixel_indices[frame_mask, 2]
    in_mask = mask[rows, cols]
    return points[frame_mask][in_mask]
```

Generally useful beyond SAM3 — any 2D region → corresponding 3D points from a `PointcloudResult`. Added to `pointcloud/utils.py` alongside `lift_features` / `reproject_pixels`.

---

## Out of Scope

- `overlay_masks` visualization utility (can be added later, trivial)
- Video-mode SAM3 (`build_sam3_video_predictor`) — image-only for now
- Per-patch CLIP features from PE-Core (architecturally not possible without model modification)
