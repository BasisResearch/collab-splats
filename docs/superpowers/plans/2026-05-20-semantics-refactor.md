# Semantics Module Audit & Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unnecessary complexity, fix latent bugs, migrate misplaced code to correct modules, and add section dividers throughout `collab_splats/semantics/`.

**Architecture:** Seven sequential passes — (1) dead code + hard imports in `features.py`, (2) segmentation bugs, (3) dead backends + shim removal, (4) utils scope migration, (5) move retrieval to pointcloud, (6) registry dedup + abstract base, (7) clarity pass.

**Tech Stack:** Python, PyTorch, pytest

**Formatting:** PEP 8 throughout. Two blank lines between top-level defs. Section dividers:

```python

########################################################
########## Section Title ###############################
########################################################

```

---

## Audit Findings

| # | File | Issue |
|---|------|-------|
| B1 | `features.py:29-34,193` | Commented guard block + `if not _MASKCLIP_AVAILABLE:` check — required dep, use hard import |
| B2 | `segmentation.py:259` | `zip(*selected_masks)` crashes when `selected_masks` is empty |
| B3 | `segmentation.py:322` | `uint16 → uint8` cast silently truncates >255 mask IDs |
| D1 | `features.py:37` | Unresolved question comment on `TORCH_HOME` |
| D2 | `features.py:282,372` | `try/except ImportError: raise ImportError(better msg)` around `transformers` import — required dep |
| D3 | `segmentation.py:23-50` | `ultralytics` / `grounded_sam` stub `pass` backends |
| D4 | `semantics/__init__.py:33-37` | `frame_sampling` re-export shim — callers already import from `collab_splats.utils` |
| A1 | `features.py:71` | `forward` raises `NotImplementedError` instead of `@abstractmethod` |
| A2 | `features.py` + `retrieval.py` | Duplicate `register()`/`get()` pattern in two base classes |
| S1 | `semantics/utils.py` | General torch utils (`pytorch_gc`, `infer_batch_size`, etc.) imported by modules outside semantics |
| S2 | `semantics/retrieval.py` | Retrieval extractors used only by `pointcloud/loop_closure/` — wrong module |
| C1 | `features.py` | `print()` in `forward()` / `score_queries()` — use `logging` |
| C2 | `segmentation.py:114` | Stale docstring: lists `obj_results` as param but it's computed internally |
| C3 | `features.py`, `utils.py` | No section dividers — `segmentation.py` has them, normalize |

---

## Critical Files

- `collab_splats/semantics/features.py`
- `collab_splats/semantics/segmentation.py`
- `collab_splats/semantics/utils.py`
- `collab_splats/semantics/retrieval.py` — deleted in Task 5
- `collab_splats/semantics/__init__.py`
- `collab_splats/utils/__init__.py`
- `collab_splats/utils/torch_utils.py` — created in Task 4
- `collab_splats/pointcloud/localization.py` — created in Task 5
- `collab_splats/nerfstudio/datamanagers/features.py`
- `collab_splats/wrapper/splatter.py`
- `collab_splats/pointcloud/loop_closure/retrieval.py`
- `tests/semantics/test_features_guards.py`
- `tests/semantics/test_semantics_utils.py`
- `tests/semantics/test_retrieval.py` — merged into `tests/pointcloud/test_retrieval.py`

---

## Task 1: Clean up `features.py` — hard imports, remove dead code (B1, D1, D2)

**Files:**
- Modify: `collab_splats/semantics/features.py`
- Modify: `tests/semantics/test_features_guards.py`

**Why:** `maskclip_onnx` and `transformers` are required deps — if missing, Python raises `ImportError` naturally. Guards add flags and try/except wrappers for conditions that should never happen. All imports go to the top of the file.

- [ ] **Step 1: Hard-import `maskclip_onnx` at module top**

Delete lines 28–34 (the commented guard block). Replace with:
```python
import maskclip_onnx
```

Remove lines 193–196 (`if not _MASKCLIP_AVAILABLE: raise ImportError(...)`) from `MaskCLIPExtractor.__init__`.

- [ ] **Step 2: Move `transformers` import to module top**

Add to top-level imports:
```python
from transformers import AutoModel
```

In `DINOFeatureExtractor.__init__` (~line 282), delete:
```python
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for DINOFeatureExtractor. "
                "Install via: pip install transformers"
            ) from e
```

Same in `Talk2DinoExtractor.__init__` (~line 372).

- [ ] **Step 3: Fix `TORCH_HOME` comment**

Replace the question comment with an explanatory one:
```python
# TORCH_HOME: respects $TORCH_HOME env var, falls back to ~/.cache/torch (torch default)
TORCH_HOME = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
```

- [ ] **Step 4: Update test**

In `tests/semantics/test_features_guards.py`, delete `test_maskclip_extractor_raises_without_package`. Replace with:
```python
def test_maskclip_onnx_importable():
    """maskclip_onnx must be installed — it is a required dependency."""
    import maskclip_onnx  # raises ImportError if missing
```

- [ ] **Step 5: Verify**
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_features_guards.py -xvs 2>&1 | tail -20
```
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add collab_splats/semantics/features.py tests/semantics/test_features_guards.py
git commit -m "refactor(semantics): hard imports for maskclip_onnx and transformers

Both are required deps. Guards were dead complexity — removed guard block,
_MASKCLIP_AVAILABLE flag, try/except wrappers. All imports at module top.
"
```

---

## Task 2: Fix `segmentation.py` bugs (B2, B3)

**Files:**
- Modify: `collab_splats/semantics/segmentation.py`
- Modify: `tests/semantics/test_segmentation.py`

**Why (B2):** `zip(*[])` raises `ValueError`. Need early return when no masks pass threshold.
**Why (B3):** Intermediate array is `uint16`; final `.astype(np.uint8)` silently truncates IDs >255.

- [ ] **Step 1: Write failing test**

Add to `tests/semantics/test_segmentation.py`:
```python
def test_create_composite_mask_empty_results():
    """create_composite_mask must not crash when no masks pass confidence threshold."""
    from collab_splats.semantics.segmentation import create_composite_mask
    results = [{"predicted_iou": 0.5, "segmentation": np.zeros((4, 4), dtype=np.uint8)}]
    result = create_composite_mask(results, confidence_threshold=0.99)
    assert isinstance(result, np.ndarray)
```

- [ ] **Step 2: Verify it fails**
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_segmentation.py::test_create_composite_mask_empty_results -xvs 2>&1 | tail -10
```
Expected: FAIL with `ValueError`

- [ ] **Step 3: Fix `create_composite_mask` — guard empty list**

At `segmentation.py` line 258, before `zip(*selected_masks)`, add:
```python
    if not selected_masks:
        return np.zeros_like(results[0]["segmentation"], dtype=np.uint8) if results else np.zeros((0, 0), dtype=np.uint8)
    masks, confs = zip(*selected_masks)
```

- [ ] **Step 4: Fix `convert_matched_mask` — preserve `uint16`**

At line 322, remove `.astype(np.uint8)`:
```python
    return matched_mask  # uint16 — preserves IDs > 255
```
Update docstring: `Returns: np.ndarray dtype=uint16`.

- [ ] **Step 5: Verify tests pass**
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_segmentation.py -xvs 2>&1 | tail -20
```

- [ ] **Step 6: Commit**
```bash
git add collab_splats/semantics/segmentation.py tests/semantics/test_segmentation.py
git commit -m "fix(semantics): guard empty masks in create_composite_mask, preserve uint16 in convert_matched_mask"
```

---

## Task 3: Remove stub backends and frame_sampling shim (D3, D4)

**Files:**
- Modify: `collab_splats/semantics/segmentation.py`
- Modify: `collab_splats/semantics/__init__.py`

**Why (D3):** `ultralytics` and `grounded_sam` branches are unreachable `pass` stubs — remove so only implemented backend is exposed.
**Why (D4):** `frame_sampling` moved to `collab_splats.utils` per ADR 009; all callers already import from there.

- [ ] **Step 1: Simplify `Segmentation` class**

Replace `__init__`:
```python
    def __init__(
        self, backend: str = "mobilesamv2", strategy: str = "object", device: str = "cpu"
    ):
        if backend != "mobilesamv2":
            raise ValueError(
                f"Backend '{backend}' not supported. Available: ['mobilesamv2']"
            )
        self.seg_model, self.object_model, self.predictor = load_mobile_sam(device=device)
        self.backend = backend
        self.strategy = strategy
```

Replace `segment()`:
```python
    def segment(self, image):
        if self.strategy == "object":
            return object_segment_image(image, self.seg_model, self.object_model, self.predictor)
        elif self.strategy == "auto":
            return auto_segment_image(image, self.seg_model)
        else:
            raise ValueError(
                f"Strategy '{self.strategy}' not supported. Available: ['object', 'auto']"
            )
```

Note: default changed from `"mobile_sam"` (was invalid — would always raise) to `"mobilesamv2"`.

- [ ] **Step 2: Drop frame_sampling shim from `__init__.py`**

Remove lines 33–37 (`from collab_splats.utils.frame_sampling import ...`) and matching `__all__` entries (lines 65–68).

- [ ] **Step 3: Confirm no callers use the shim**
```bash
grep -rn "from collab_splats.semantics import.*frame\|from collab_splats.semantics import.*OpticalFlow\|from collab_splats.semantics import.*sample_frames" /workspace/collab-splats --include="*.py" | grep -v __pycache__
```
Expected: no output.

- [ ] **Step 4: Run semantics tests**
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ -x 2>&1 | tail -15
```

- [ ] **Step 5: Commit**
```bash
git add collab_splats/semantics/segmentation.py collab_splats/semantics/__init__.py
git commit -m "refactor(semantics): remove stub backends and frame_sampling shim

Segmentation: only mobilesamv2 implemented — remove ultralytics/grounded_sam pass stubs.
frame_sampling shim dropped — ADR 009, callers already use collab_splats.utils.
"
```

---

## Task 4: Extract general utils to `collab_splats/utils/torch_utils.py` (S1)

**Files:**
- Create: `collab_splats/utils/torch_utils.py`
- Modify: `collab_splats/utils/__init__.py`
- Modify: `collab_splats/semantics/utils.py`
- Modify: `collab_splats/nerfstudio/datamanagers/features.py`
- Modify: `collab_splats/wrapper/splatter.py`

**Why:** `nerfstudio/datamanagers/features.py` and `wrapper/splatter.py` import `pytorch_gc`, `infer_batch_size`, `get_device` from `semantics.utils` — violates module layering. These are general torch utilities, not semantics-specific.

- [ ] **Step 1: Create `collab_splats/utils/torch_utils.py`**

```python
"""General-purpose PyTorch and model-loading utilities."""

import gc
from typing import Any, Generator, List

import torch
from huggingface_hub import hf_hub_download


########################################################
########## Device helpers ##############################
########################################################


def get_device() -> str:
    """Return 'cuda' if a CUDA device is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def pytorch_gc():
    """Clear CUDA cache and run Python garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


########################################################
########## Batching ####################################
########################################################


def infer_batch_size(mem_per_image_gb: float, headroom: float = 0.3) -> int:
    """Compute safe batch size from available VRAM.

    Args:
        mem_per_image_gb: Estimated GPU memory per image in GB.
        headroom: Fraction of VRAM reserved for batch data (rest = model weights).

    Returns:
        Batch size >= 1. Returns 1 if CUDA is unavailable.
    """
    if mem_per_image_gb <= 0:
        raise ValueError(f"mem_per_image_gb must be positive, got {mem_per_image_gb}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return max(1, int(vram_gb * headroom / mem_per_image_gb))
    return 1


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """Yield aligned batches of size *batch_size* from one or more parallel sequences.

    Args:
        batch_size: Number of items per batch.
        *args: One or more sequences of equal length.

    Yields:
        A list of sliced sequences, one per input arg.
    """
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args), (
        "Batched iteration must have inputs of all the same size."
    )
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


########################################################
########## Model loading ###############################
########################################################


def load_hf_weights(repo_id: str, filename: str):
    """Download a single file from a Hugging Face Hub repository.

    Args:
        repo_id: HuggingFace repo (e.g. ``"facebook/dinov2-small"``).
        filename: File path within the repo.

    Returns:
        Local path to the downloaded file.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_torchhub_model(repo_id: str, model_name: str):
    """Load a pre-trained model from torch.hub.

    Args:
        repo_id: GitHub repo (e.g. ``"facebookresearch/dinov2"``).
        model_name: Entry-point name registered in the repo's ``hubconf.py``.

    Returns:
        The loaded model (nn.Module).
    """
    return torch.hub.load(repo_id, model_name)


########################################################
########## Registry mixin ##############################
########################################################


class RegistryMixin:
    """Name-based class registry. Declare ``_registry: Dict[str, type] = {}`` in subclass."""

    _registry: dict

    @classmethod
    def register(cls, name: str):
        """Class decorator: register a subclass under *name*."""
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str):
        """Return registered class for *name*, or raise ValueError."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
```

- [ ] **Step 2: Update `collab_splats/utils/__init__.py`**

Add to imports and `__all__`:
```python
from .torch_utils import (
    get_device,
    pytorch_gc,
    infer_batch_size,
    batch_iterator,
    load_hf_weights,
    load_torchhub_model,
    RegistryMixin,
)
```

- [ ] **Step 3: Replace `semantics/utils.py` body**

Keep only semantics-specific functions; re-export the rest for backwards compat:

```python
"""Semantics-specific utilities: contrastive scoring and patch alignment.

General torch utilities live in collab_splats.utils.torch_utils.
Re-exported here so existing callers don't need to update import paths.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


########################################################
########## Re-exports from collab_splats.utils.torch_utils
########################################################

# Canonical location: collab_splats.utils.torch_utils
from collab_splats.utils.torch_utils import (
    get_device,
    pytorch_gc,
    infer_batch_size,
    batch_iterator,
    load_hf_weights,
    load_torchhub_model,
)


########################################################
########## Contrastive scoring #########################
########################################################


def compute_semantic_contrast(
    raw_similarities: torch.Tensor,
    num_positive: int,
    temperature: float = 0.05,
    reduction: str = "max",
) -> torch.Tensor:
    """Contrastive scoring: how strongly positive queries match relative to negatives.

    When no negatives are present (num_positive == raw_similarities.shape[0]),
    falls back to raw reduction over positives — contrastive scoring is undefined
    without a negative to push against.

    Args:
        raw_similarities: (N_queries, N) dot-product similarities per patch.
        num_positive: rows [0:num_positive] are positive queries; rest are negative.
        temperature: scaling parameter τ. Lower = sharper. Ignored when no negatives.
        reduction: aggregation over positive queries:
            "max"  — each positive independently scored against all negatives via
                     binary softmax; max over per-positive scores. Use for distinct
                     concepts where any match counts.
            "pool" — positives averaged in similarity space before softmax; one
                     representative competes against all negatives. Use for synonymous
                     concepts that should be treated as one combined query.

    Returns:
        (N,) contrastive scores in [0, 1].
    """
    if reduction not in ("max", "pool"):
        raise ValueError(f"Unknown reduction '{reduction}'. Choose 'max' or 'pool'.")

    pos = raw_similarities[:num_positive]
    neg = raw_similarities[num_positive:]

    if neg.shape[0] == 0:
        return pos.max(dim=0).values if reduction == "max" else pos.mean(dim=0)

    if reduction == "max":
        scores = []
        for p_i in pos:
            stacked = torch.cat([p_i.unsqueeze(0), neg], dim=0)
            scores.append(stacked.div(temperature).softmax(dim=0)[0])
        return torch.stack(scores).max(dim=0).values

    if reduction == "pool":
        avg_pos = pos.mean(dim=0, keepdim=True)
        stacked = torch.cat([avg_pos, neg], dim=0)
        return stacked.div(temperature).softmax(dim=0)[0]

    raise ValueError(f"Unknown reduction '{reduction}'. Choose 'max' or 'pool'.")


########################################################
########## Patch alignment #############################
########################################################


def interpolate_to_patch_size(
    img_bchw: torch.Tensor, patch_size: int
) -> Tuple[torch.Tensor, int, int]:
    """Interpolate image tensor so H and W are evenly divisible by patch_size.

    Args:
        img_bchw: Image tensor of shape (B, C, H, W).
        patch_size: Patch dimension to align to.

    Returns:
        Tuple of (resized_tensor, target_H, target_W).
    """
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = F.interpolate(
        img_bchw, size=(target_H, target_W), mode="bilinear", align_corners=False
    )
    return img_bchw, target_H, target_W
```

- [ ] **Step 4: Update callers outside semantics**

`collab_splats/nerfstudio/datamanagers/features.py` — change:
```python
from collab_splats.semantics.utils import infer_batch_size, pytorch_gc
```
to:
```python
from collab_splats.utils.torch_utils import infer_batch_size, pytorch_gc
```

`collab_splats/wrapper/splatter.py` — change:
```python
from collab_splats.semantics.utils import get_device
```
to:
```python
from collab_splats.utils.torch_utils import get_device
```

- [ ] **Step 5: Run tests**
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_semantics_utils.py tests/semantics/ -x 2>&1 | tail -20
```

- [ ] **Step 6: Commit**
```bash
git add collab_splats/utils/torch_utils.py collab_splats/utils/__init__.py collab_splats/semantics/utils.py collab_splats/nerfstudio/datamanagers/features.py collab_splats/wrapper/splatter.py
git commit -m "refactor(utils): extract torch utilities to collab_splats.utils.torch_utils

pytorch_gc/infer_batch_size/batch_iterator/load_hf_weights/load_torchhub_model/get_device
moved out of semantics.utils — nerfstudio and wrapper were importing across module boundary.
RegistryMixin co-located here (used by both semantics and pointcloud). Re-exports in
semantics.utils preserve backwards compat for existing callers.
"
```

---

## Task 5: Move retrieval extractors to `collab_splats/pointcloud/localization.py` (S2)

**Files:**
- Create: `collab_splats/pointcloud/localization.py`
- Delete: `collab_splats/semantics/retrieval.py`
- Modify: `collab_splats/pointcloud/loop_closure/retrieval.py`
- Modify: `tests/pointcloud/test_retrieval.py` (merge from `tests/semantics/test_retrieval.py`)

**Why:** Retrieval is stage 1 of camera localization (global retrieval → local matching → PnP) — a pointcloud concern. Never exported from `semantics/__init__.py`, so deleting `semantics/retrieval.py` breaks no public API.

- [ ] **Step 1: Create `collab_splats/pointcloud/localization.py`**

```python
"""Camera localization pipeline: find pose of a query image in a known reconstruction.

Three-stage pipeline (Stage 1 implemented; 2 and 3 reserved):

  Stage 1 — Global retrieval: top-K visually similar reference frames via compact
             global descriptors (DINOv2-SALAD). Current use: loop closure detection.

  Stage 2 — Local feature matching: XFeat / ALIKED+SP / LightGlue (reserved).

  Stage 3 — Pose estimation: PnP via OpenCV / pycolmap (reserved).
"""
from __future__ import annotations

import pathlib
import sys
from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import torchvision.transforms as T

# Vendor path must precede vendored imports.
# serizba/salad is vendored at third_party/salad/ to avoid pytorch_lightning dep.
_SALAD_VENDOR_PATH = str(pathlib.Path(__file__).parents[1] / "vendor" / "salad")
if _SALAD_VENDOR_PATH not in sys.path:
    sys.path.insert(0, _SALAD_VENDOR_PATH)

from models.aggregators.salad import SALAD  # vendored: third_party/salad/
from models.backbones.dinov2 import DINOv2  # vendored: third_party/salad/

from collab_splats.utils.torch_utils import RegistryMixin


########################################################
########## Stage 1: Global retrieval ###################
########################################################


class BaseRetrievalExtractor(RegistryMixin, nn.Module, ABC):
    """Abstract base for global image descriptor extractors with name-based registry.

    Returns (N, D) normalized descriptors — one compact vector per image.
    Used for top-K candidate retrieval before local feature matching.
    """

    _registry: Dict[str, type["BaseRetrievalExtractor"]] = {}

    @abstractmethod
    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """Return (N, D) normalized global descriptors for N images."""


@BaseRetrievalExtractor.register("dino-salad")
class DinoSaladExtractor(BaseRetrievalExtractor):
    """DINO-SALAD global image descriptor for visual place recognition.

    DINOv2 ViT-B/14 backbone + SALAD aggregator, pretrained on GSV-Cities.
    Vendored from serizba/salad at third_party/salad/. Avoids VPRModel to skip
    pytorch_lightning / pytorch_metric_learning at runtime.
    Weights: https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
    """

    _WEIGHTS_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
    _INPUT_SIZE = 224  # divisible by 14 for DINOv2 patch grid

    def __init__(self, device: str | None = None):
        super().__init__()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Attr names match VPRModel so dino_salad.ckpt keys (backbone.* / aggregator.*) load cleanly
        self.backbone = DINOv2(
            model_name="dinov2_vitb14",
            num_trainable_blocks=4,
            norm_layer=True,
            return_token=True,
        ).to(self._device)
        self.aggregator = SALAD(
            num_channels=768,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
        ).to(self._device)
        sd = torch.hub.load_state_dict_from_url(
            self._WEIGHTS_URL, map_location=torch.device("cpu")
        )
        self.load_state_dict(sd, strict=False)
        self.eval()

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """Return (N, D) normalized float32 descriptors on CPU."""
        if isinstance(images, (list, tuple)):
            _t = T.Compose([
                T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            imgs = torch.stack([_t(img) for img in images])
        else:
            imgs = images.float()
            if imgs.shape[-2] != self._INPUT_SIZE or imgs.shape[-1] != self._INPUT_SIZE:
                imgs = torch.nn.functional.interpolate(
                    imgs, size=(self._INPUT_SIZE, self._INPUT_SIZE),
                    mode="bilinear", align_corners=False,
                )
        imgs = imgs.to(self._device)
        with torch.no_grad():
            feats = self.backbone(imgs)
            descriptors = self.aggregator(feats)
        return torch.nn.functional.normalize(descriptors, p=2, dim=-1).cpu()


########################################################
########## Stage 2: Local feature matching (reserved) ##
########################################################

# Future: XFeat / ALIKED+SP / LightGlue matcher classes.
# Interface: match(img_a, img_b) -> Tuple[kpts_a, kpts_b]


########################################################
########## Stage 3: Pose estimation (reserved) #########
########################################################

# Future: PnP solver given 2D-3D correspondences + pointcloud.
# Interface: estimate_pose(kpts_2d, pts_3d, K) -> SE3
```

Note: `_SALAD_VENDOR_PATH` uses `.parents[1]` — one level up from `pointcloud/` to reach project root. Verify path is correct after creating file.

- [ ] **Step 2: Update `pointcloud/loop_closure/retrieval.py` import**

Change line 88:
```python
from collab_splats.semantics.retrieval import BaseRetrievalExtractor
```
to:
```python
from collab_splats.pointcloud.localization import BaseRetrievalExtractor
```

- [ ] **Step 3: Delete `semantics/retrieval.py`**

```bash
git rm collab_splats/semantics/retrieval.py
```

No shim — `semantics/__init__.py` never exported anything from this file.

- [ ] **Step 4: Merge test files**

Read both `tests/semantics/test_retrieval.py` and `tests/pointcloud/test_retrieval.py`. Add any unique tests from the semantics version to the pointcloud version, updating imports to `collab_splats.pointcloud.localization`. Then:
```bash
git rm tests/semantics/test_retrieval.py
```

- [ ] **Step 5: Run tests**
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_retrieval.py -xvs 2>&1 | tail -20
```

- [ ] **Step 6: Commit**
```bash
git add collab_splats/pointcloud/localization.py collab_splats/pointcloud/loop_closure/retrieval.py tests/pointcloud/test_retrieval.py
git rm collab_splats/semantics/retrieval.py tests/semantics/test_retrieval.py
git commit -m "refactor(localization): move retrieval extractors to pointcloud/localization.py

Stage 1 of camera localization pipeline. Belongs with pointcloud not semantics.
localization.py reserves sections for future XFeat local matching and PnP stages.
No shim: semantics/__init__.py never exported retrieval symbols.
"
```

---

## Task 6: Abstract `forward()` + apply `RegistryMixin` (A1, A2)

**Files:**
- Modify: `collab_splats/semantics/features.py`

**Why (A1):** `BaseFeatureExtractor.forward` raises `NotImplementedError` but doesn't prevent instantiation — use `@abstractmethod`.
**Why (A2):** `register()`/`get()` now live in `RegistryMixin` (Task 4); remove the duplicates.

- [ ] **Step 1: Update `BaseFeatureExtractor`**

Add to top-level imports in `features.py`:
```python
from collab_splats.utils.torch_utils import RegistryMixin
```

Change class declaration and remove hand-written `register()`/`get()` methods:
```python
class BaseFeatureExtractor(RegistryMixin, nn.Module, ABC):
    """Abstract base for image feature extractors with a name-based registry.

    Register subclasses via ``@BaseFeatureExtractor.register("name")``.
    Retrieve with ``BaseFeatureExtractor.get("name")``.
    """

    _registry: Dict[str, type["BaseFeatureExtractor"]] = {}
    _FALLBACK_MEM_GB: float = 2.0

    @abstractmethod
    def forward(self, images: list) -> list[torch.Tensor]:
        """Preprocess, run inference, reshape. Returns one feature tensor per image."""
        ...

    def _get_memory_per_image(self, sample_image) -> float:
        """Measure GPU memory consumed per image via a single forward pass.

        Returns GB. Falls back to _FALLBACK_MEM_GB if CUDA unavailable.
        """
        if not torch.cuda.is_available():
            return self._FALLBACK_MEM_GB
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        with torch.no_grad():
            self.forward([sample_image])
        peak = torch.cuda.max_memory_allocated()
        mem_gb = (peak - baseline) / (1024 ** 3)
        return max(mem_gb, 0.01)
```

- [ ] **Step 2: Run tests**
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ tests/pointcloud/test_retrieval.py -x 2>&1 | tail -20
```

- [ ] **Step 3: Commit**
```bash
git add collab_splats/semantics/features.py
git commit -m "refactor(semantics): BaseFeatureExtractor uses RegistryMixin, forward is @abstractmethod"
```

---

## Task 7: Clarity pass — logging, section dividers, docstring fix (C1–C3)

**Files:**
- Modify: `collab_splats/semantics/features.py`
- Modify: `collab_splats/semantics/segmentation.py`

**Why (C1):** `features.py` uses `print()` for progress; `segmentation.py` uses `logging`. Normalize to `logging`.
**Why (C2):** `object_segment_image` docstring lists `obj_results` as a param but it's computed internally.
**Why (C3):** `features.py` has no section dividers; normalize to `########` style used by `segmentation.py`.

- [ ] **Step 1: Add `logger` + replace `print` in `features.py`**

After imports, add:
```python
import logging

logger = logging.getLogger(__name__)
```

In `MaskCLIPExtractor.forward`, `DINOFeatureExtractor.forward`, `Talk2DinoExtractor.forward`, replace:
```python
print(f"[{type(self).__name__}] Extracting features: {len(images)} images...", end=" ", flush=True)
t0 = time.perf_counter()
# ...
print(f"done in {time.perf_counter() - t0:.1f}s")
```
with:
```python
logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))
```

In `BaseQueryableExtractor.score_queries`, replace `print` calls with:
```python
logger.debug(
    "[%s] Scoring queries: %d positive%s (reduction=%s)",
    type(self).__name__, len(positive),
    f", {len(negative)} negative" if negative else "",
    reduction,
)
```

Remove `import time` if no longer used after this change.

- [ ] **Step 2: Fix `object_segment_image` docstring in `segmentation.py`**

Replace `Inputs` section of `object_segment_image` docstring:
```
    Inputs:
        - image: np.ndarray (H, W, 3)
        - mobile_sam: MobileSAM model
        - obj_model: YOLOv8 object detector (bounding boxes computed internally)
        - predictor: SAMPredictor object
        - batch_size: boxes per decode batch (default: 320)
```

- [ ] **Step 3: Add section dividers to `features.py`**

Before `BaseFeatureExtractor` class definition:
```python

########################################################
########## Registry and abstract base ##################
########################################################

```

Before `BaseQueryableExtractor` class definition:
```python

########################################################
########## Queryable extractor base ####################
########################################################

```

Before `@BaseFeatureExtractor.register("maskclip")`:
```python

########################################################
########## Concrete extractors #########################
########################################################

```

- [ ] **Step 4: Run tests**
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ -x 2>&1 | tail -15
```

- [ ] **Step 5: Commit**
```bash
git add collab_splats/semantics/features.py collab_splats/semantics/segmentation.py
git commit -m "refactor(semantics): replace print with logging, add section dividers, fix docstring"
```

---

## Verification

Run full test suite:
```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x --ignore=tests/dashboard -q 2>&1 | tail -30
```
Expected: same pass/skip/fail counts as before.

Spot-check imports:
```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.semantics import BaseFeatureExtractor, Segmentation
from collab_splats.utils.torch_utils import pytorch_gc, get_device, batch_iterator, RegistryMixin
from collab_splats.pointcloud.localization import BaseRetrievalExtractor, DinoSaladExtractor
print('all imports OK')
"
```
