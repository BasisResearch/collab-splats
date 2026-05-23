# Semantics Features Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `collab_splats/semantics/features.py` into focused modules, unify extractor interfaces, migrate DINO to HuggingFace, and remove dead code.

**Architecture:** Extract general utilities into `utils/image.py` (PIL ops) and `semantics/utils.py` (torch/model ops). `features.py` becomes purely extractor classes with unified `compute_similarity` interface. `_get_memory_per_image()` replaces hardcoded `MEM_PER_IMAGE_GB`. DINO migrates from torch.hub to `transformers.AutoModel`.

**Tech Stack:** Python, PyTorch, HuggingFace transformers, PIL, pytest

**Spec:** `worklog/history/specs/2026-04-21-semantics-features-refactor-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `collab_splats/utils/image.py` | `open_image`, `resize_image` (pure PIL) |
| Create | `collab_splats/semantics/utils.py` | `compute_semantic_contrast`, `interpolate_to_patch_size`, `pytorch_gc`, `infer_batch_size`, `load_hf_weights`, `load_torchhub_model`, `batch_iterator` |
| Create | `tests/utils/test_image.py` | Tests for `utils/image.py` |
| Create | `tests/semantics/test_semantics_utils.py` | Tests for `semantics/utils.py` |
| Modify | `collab_splats/semantics/features.py` | Strip utilities, add `_get_memory_per_image`, unify interfaces, migrate DINO |
| Modify | `collab_splats/semantics/protocols.py` | Simplify `SupportsTextQuery` to `encode_text` + `compute_similarity` |
| Modify | `collab_splats/semantics/__init__.py` | Update re-exports to new locations |
| Modify | `collab_splats/utils/features.py` | Update shim re-exports to new locations |
| Modify | `collab_splats/utils/__init__.py` | Add `image.py` exports |
| Modify | `collab_splats/nerfstudio/datamanagers/features.py` | Use `_get_memory_per_image` instead of `MEM_PER_IMAGE_GB` |
| Modify | `collab_splats/dashboard/semantics.py` | Inline multi-label query logic (replaces `compute_semantic_heatmap` call) |
| Modify | `tests/test_models.py:27,29` | `"clip-vit"` → `"samclip"` |

---

### Task 1: Create `collab_splats/utils/image.py`

**Files:**
- Create: `collab_splats/utils/image.py`
- Create: `tests/utils/test_image.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/utils/test_image.py
"""Tests for collab_splats.utils.image — pure PIL image utilities."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from collab_splats.utils.image import open_image, resize_image


class TestOpenImage:
    def test_pil_passthrough(self):
        img = Image.new("RGB", (100, 100))
        assert open_image(img) is img

    def test_from_ndarray(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = open_image(arr)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_from_path(self, tmp_path):
        p = tmp_path / "test.png"
        Image.new("RGB", (50, 50)).save(p)
        result = open_image(str(p))
        assert result.size == (50, 50)

    def test_from_pathlib(self, tmp_path):
        p = tmp_path / "test.png"
        Image.new("RGB", (50, 50)).save(p)
        result = open_image(p)
        assert isinstance(result, Image.Image)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported image type"):
            open_image(42)


class TestResizeImage:
    def test_landscape(self):
        img = Image.new("RGB", (200, 100))
        resized = resize_image(img, longest_edge=100)
        assert resized.size[0] == 100  # width is longest
        assert resized.size[1] == 50

    def test_portrait(self):
        img = Image.new("RGB", (100, 200))
        resized = resize_image(img, longest_edge=100)
        assert resized.size[0] == 50
        assert resized.size[1] == 100  # height is longest

    def test_square(self):
        img = Image.new("RGB", (200, 200))
        resized = resize_image(img, longest_edge=100)
        assert resized.size == (100, 100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/utils/test_image.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'collab_splats.utils.image'`

- [ ] **Step 3: Create `collab_splats/utils/image.py`**

```python
# collab_splats/utils/image.py
"""Pure PIL image utilities — no torch dependency."""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def open_image(image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    """Load an image from path, ndarray, or PIL Image."""
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


def resize_image(image: Image.Image, longest_edge: int) -> Image.Image:
    """Resize maintaining aspect ratio so longest edge equals target length."""
    width, height = image.size
    ratio = longest_edge / max(width, height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return image.resize((new_width, new_height), Image.BILINEAR)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/utils/test_image.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/image.py tests/utils/test_image.py
git commit -m "refactor(utils): create image.py with open_image and resize_image"
```

---

### Task 2: Create `collab_splats/semantics/utils.py`

**Files:**
- Create: `collab_splats/semantics/utils.py`
- Create: `tests/semantics/test_semantics_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/semantics/test_semantics_utils.py
"""Tests for collab_splats.semantics.utils — torch/model utilities."""

import gc
from unittest.mock import patch

import pytest
import torch

from collab_splats.semantics.utils import (
    compute_semantic_contrast,
    interpolate_to_patch_size,
    pytorch_gc,
    infer_batch_size,
    batch_iterator,
)


class TestComputeSemanticContrast:
    def test_standard_method_shape(self):
        raw = torch.randn(5, 100)  # 5 queries, 100 spatial positions
        result = compute_semantic_contrast(raw, num_positive=2, softmax_temp=0.05, method="standard")
        assert result.shape == (100,)

    def test_standard_method_sums_positive_probs(self):
        raw = torch.randn(3, 50)  # 2 positive + 1 negative
        result = compute_semantic_contrast(raw, num_positive=2, softmax_temp=0.05, method="standard")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_pairwise_method_shape(self):
        raw = torch.randn(4, 100)  # 2 positive + 2 negative
        result = compute_semantic_contrast(raw, num_positive=2, softmax_temp=0.05, method="pairwise")
        assert result.shape == (100,)

    def test_unknown_method_raises(self):
        raw = torch.randn(3, 50)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_semantic_contrast(raw, num_positive=1, softmax_temp=0.05, method="invalid")


class TestInterpolateToPatchSize:
    def test_divisible_dimensions(self):
        img = torch.randn(1, 3, 224, 224)
        result, h, w = interpolate_to_patch_size(img, patch_size=14)
        assert h % 14 == 0
        assert w % 14 == 0
        assert result.shape == (1, 3, h, w)

    def test_non_divisible_dimensions(self):
        img = torch.randn(1, 3, 230, 230)
        result, h, w = interpolate_to_patch_size(img, patch_size=14)
        assert h % 14 == 0
        assert w % 14 == 0
        assert h == 224  # 230 // 14 * 14 = 224


class TestPytorchGc:
    def test_runs_without_error(self):
        pytorch_gc()  # should not raise on CPU


class TestInferBatchSize:
    def test_cpu_returns_one(self):
        if not torch.cuda.is_available():
            assert infer_batch_size(3.0) == 1

    def test_negative_mem_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            infer_batch_size(-1.0)

    def test_zero_mem_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            infer_batch_size(0.0)


class TestBatchIterator:
    def test_basic_batching(self):
        items = list(range(10))
        batches = list(batch_iterator(3, items))
        assert len(batches) == 4  # 3+3+3+1
        assert batches[0] == [[0, 1, 2]]
        assert batches[-1] == [[9]]

    def test_multiple_args(self):
        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        batches = list(batch_iterator(2, a, b))
        assert len(batches) == 2
        assert batches[0] == [[1, 2], [5, 6]]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            list(batch_iterator(2, [1, 2], [3]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/semantics/test_semantics_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'collab_splats.semantics.utils'`

- [ ] **Step 3: Create `collab_splats/semantics/utils.py`**

```python
# collab_splats/semantics/utils.py
"""Torch and model-loading utilities for the semantics pipeline."""

import gc
from typing import Any, Generator, List, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


def compute_semantic_contrast(
    raw_similarities: torch.Tensor,
    num_positive: int,
    softmax_temp: float,
    method: str,
) -> torch.Tensor:
    """Score features against positive queries relative to negative queries.

    Applies temperature-scaled softmax across all query dimensions, then reduces
    to a single score per spatial position.

    Methods:
        - "standard": softmax over all queries, sum positive probabilities.
          Higher score = more similar to positive queries relative to all queries.
        - "pairwise": average positive similarities paired against each negative,
          softmax per pair, take minimum across negatives.
          Higher score = consistently more similar to positives than any single negative.

    Args:
        raw_similarities: (num_queries, N) dot-product similarities.
        num_positive: number of rows that are positive queries (rest are negative).
        softmax_temp: temperature for softmax (lower = sharper contrast).
        method: "standard" or "pairwise".

    Returns:
        (N,) contrastive scores in [0, 1].
    """
    if method == "standard":
        probs = (raw_similarities / softmax_temp).softmax(dim=0)
        return probs[:num_positive].sum(dim=0)
    if method == "pairwise":
        pos_similarities = raw_similarities[:num_positive]
        neg_similarities = raw_similarities[num_positive:]
        avg_pos = pos_similarities.mean(dim=0, keepdim=True)
        paired = torch.cat([avg_pos.expand(neg_similarities.shape[0], -1), neg_similarities], dim=0)
        probs = (paired / softmax_temp).softmax(dim=0)
        return torch.nan_to_num(probs[: neg_similarities.shape[0]].min(dim=0)[0], nan=0.0)
    raise ValueError(f"Unknown method: {method}. Choose 'standard' or 'pairwise'")


def interpolate_to_patch_size(
    img_bchw: torch.Tensor, patch_size: int
) -> Tuple[torch.Tensor, int, int]:
    """Interpolate image tensor so H and W are evenly divisible by patch_size."""
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = F.interpolate(
        img_bchw, size=(target_H, target_W), mode="bilinear", align_corners=False
    )
    return img_bchw, target_H, target_W


def pytorch_gc():
    """Clear CUDA cache and run Python garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def infer_batch_size(mem_per_image_gb: float, headroom: float = 0.3) -> int:
    """Compute safe batch size from available VRAM.

    Args:
        mem_per_image_gb: Estimated GPU memory per image in GB.
        headroom: Fraction of VRAM to use for batch data; remainder reserved for model weights.

    Returns:
        Batch size >= 1. Returns 1 if CUDA is unavailable.
    """
    if mem_per_image_gb <= 0:
        raise ValueError(f"mem_per_image_gb must be positive, got {mem_per_image_gb}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return max(1, int(vram_gb * headroom / mem_per_image_gb))
    return 1


def load_hf_weights(repo_id: str, filename: str):
    """Download a file from Hugging Face Hub."""
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_torchhub_model(repo_id: str, model_name: str):
    """Load a model from torch.hub."""
    return torch.hub.load(repo_id, model_name)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """Yield batches of size batch_size from parallel sequences."""
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args), (
        "Batched iteration must have inputs of all the same size."
    )
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/semantics/test_semantics_utils.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/utils.py tests/semantics/test_semantics_utils.py
git commit -m "refactor(semantics): create utils.py with shared torch/model utilities"
```

---

### Task 3: Add `_get_memory_per_image` to `BaseFeatureExtractor`

**Files:**
- Modify: `collab_splats/semantics/features.py:194-216`
- Create: `tests/semantics/test_memory_estimation.py`

- [ ] **Step 1: Write failing test**

```python
# tests/semantics/test_memory_estimation.py
"""Tests for BaseFeatureExtractor._get_memory_per_image."""

import torch
import torch.nn as nn
from PIL import Image

from collab_splats.semantics.features import BaseFeatureExtractor


class DummyExtractor(BaseFeatureExtractor):
    """Minimal extractor for testing _get_memory_per_image."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, images: list) -> list[torch.Tensor]:
        return [torch.randn(10) for _ in images]


class TestGetMemoryPerImage:
    def test_returns_positive_float(self):
        ext = DummyExtractor()
        sample = Image.new("RGB", (64, 64))
        result = ext._get_memory_per_image(sample)
        assert isinstance(result, float)
        assert result > 0.0

    def test_cpu_returns_fallback(self):
        if not torch.cuda.is_available():
            ext = DummyExtractor()
            sample = Image.new("RGB", (64, 64))
            result = ext._get_memory_per_image(sample)
            assert result == 2.0  # fallback default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/semantics/test_memory_estimation.py -v`
Expected: FAIL with `AttributeError: 'DummyExtractor' object has no attribute '_get_memory_per_image'`

- [ ] **Step 3: Add `_get_memory_per_image` to `BaseFeatureExtractor`**

In `collab_splats/semantics/features.py`, add method to `BaseFeatureExtractor` class (after `forward`):

```python
    _FALLBACK_MEM_GB: float = 2.0

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
        return max(mem_gb, 0.01)  # floor to avoid zero
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/semantics/test_memory_estimation.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Remove `MEM_PER_IMAGE_GB` from all three extractors**

Delete these lines from `features.py`:
- `MEM_PER_IMAGE_GB: float = 3.0` from `MaskCLIPExtractor` (line 234)
- `MEM_PER_IMAGE_GB: float = 1.5` from `DINOFeatureExtractor` (line 345)
- `MEM_PER_IMAGE_GB: float = 1.0` from `Talk2DinoExtractor` (line 418)

- [ ] **Step 6: Commit**

```bash
git add collab_splats/semantics/features.py tests/semantics/test_memory_estimation.py
git commit -m "refactor(features): replace hardcoded MEM_PER_IMAGE_GB with _get_memory_per_image"
```

---

### Task 4: Strip utilities from `features.py` and fix cascading imports

**Files:**
- Modify: `collab_splats/semantics/features.py`
- Modify: `collab_splats/semantics/__init__.py`
- Modify: `collab_splats/semantics/segmentation.py`

This task removes all extracted functions from `features.py` and updates all internal imports that would break. `segmentation.py` imports `batch_iterator` and `load_torchhub_model` directly from `semantics.features`, and `semantics/__init__.py` re-exports utilities from `.features` — both must be updated in the same task to keep the package importable.

- [ ] **Step 1: Update imports at top of `features.py`**

Replace the imports section and remove extracted functions. The new top of file:

```python
"""
Feature extraction classes.

Provides a registry-based system (BaseFeatureExtractor) for image feature extractors:
  - MaskCLIPExtractor: patch-level CLIP features via maskclip_onnx
  - DINOFeatureExtractor: patch-level DINOv2 features via HuggingFace transformers
  - Talk2DinoExtractor: patch features + text-conditioned similarity via Talk2DINO (HF Hub)
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import (
    compute_semantic_contrast,
    interpolate_to_patch_size,
)

try:
    import maskclip_onnx
    _MASKCLIP_AVAILABLE = True
except ImportError:
    maskclip_onnx = None  # type: ignore[assignment]

TORCH_HOME = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
```

- [ ] **Step 2: Remove extracted functions from `features.py`**

Delete all of these from `features.py` (they now live in `utils/image.py` or `semantics/utils.py`):
- `_open_image` function (lines 39-58)
- `_apply_similarity_method` function (lines 61-79)
- The `########` general utils section header (lines 82-84)
- `load_hf_weights` function (lines 87-89)
- `load_torchhub_model` function (lines 92-94)
- `pytorch_gc` function (lines 97-101)
- `infer_batch_size` function (lines 104-119)
- `resize_image` function (lines 122-140) — now in `utils/image.py`
- `interpolate_to_patch_size` function (lines 143-165)
- `batch_iterator` function (lines 168-186)
- `_DEFAULT_NEGATIVE` constant (line 36)

- [ ] **Step 3: Update internal references in extractors**

In `MaskCLIPExtractor.preprocess`:
- Change `_open_image(image)` → `open_image(image)`

In `MaskCLIPExtractor.compute_similarity`:
- Change `negative = _DEFAULT_NEGATIVE` → `negative = ["object"]`
- Change `_apply_similarity_method(...)` → `compute_semantic_contrast(...)`

In `DINOFeatureExtractor.preprocess`:
- Change `_open_image(image)` → `open_image(image)`

In `DINOFeatureExtractor.__init__`:
- `load_torchhub_model` is no longer imported at module level. Add local import:
  ```python
  from collab_splats.semantics.utils import load_torchhub_model
  ```
  (This will be removed entirely in Task 6 when DINO migrates to HF)

In `Talk2DinoExtractor.preprocess`:
- Change `_open_image(image)` → `open_image(image)`

- [ ] **Step 4: Delete `clip-vit` alias**

Remove lines 332-334:
```python
# Remove backwards compability tests --> these should be retired
# Register backward-compatible alias expected by tests/metadata
BaseFeatureExtractor._registry["clip-vit"] = MaskCLIPExtractor
```

- [ ] **Step 5: Remove all comment annotations (Tommy's review comments)**

Delete these lines/comments:
- `# These should be manatory installs` (line 24)
- `# Should this be a larger variable than just this script?` (line 32)
- `# Hardcoding` (line 233)
- `# Hardcoding` (line 344)
- `# Is this available via huggingface?...` (line 353)
- `# This seems oddly different from other models...` (line 388)
- `# Why does this need to be a private model associated function?` (line 391)
- `# Weirdly redundant and why not just call it compute similarity?...` (line 484)
- `# We moved things to visualization for this` (line 511)
- All `########` section banners

- [ ] **Step 6: Update `semantics/segmentation.py` imports**

Change:
```python
from collab_splats.semantics.features import batch_iterator, load_torchhub_model
```
To:
```python
from collab_splats.semantics.utils import batch_iterator, load_torchhub_model
```

- [ ] **Step 7: Update `semantics/__init__.py` re-exports**

Replace the features import block so utilities come from `.utils` not `.features`:

```python
"""
collab_splats.semantics — feature extraction, segmentation, and capability protocols.
"""

from .features import (
    BaseFeatureExtractor,
    MaskCLIPExtractor,
    DINOFeatureExtractor,
    Talk2DinoExtractor,
)
from .utils import (
    compute_semantic_contrast,
    interpolate_to_patch_size,
    pytorch_gc,
    infer_batch_size,
    load_hf_weights,
    load_torchhub_model,
    batch_iterator,
)
from .segmentation import (
    Segmentation,
    load_mobile_sam,
    auto_segment_image,
    get_object_masks,
    object_segment_image,
    create_patch_mask,
    create_composite_mask,
    mask_id_to_binary_mask,
    convert_matched_mask,
    aggregate_masked_features,
)
from .protocols import SupportsTextQuery
from collab_splats.utils.frame_sampling import (
    OpticalFlowFrameSelector,
    sample_frames_fps,
    sample_frames_optical_flow,
)

__all__ = [
    # extractors
    "BaseFeatureExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    # semantics utilities
    "compute_semantic_contrast",
    "interpolate_to_patch_size",
    "pytorch_gc",
    "infer_batch_size",
    "load_hf_weights",
    "load_torchhub_model",
    "batch_iterator",
    # segmentation
    "Segmentation",
    "load_mobile_sam",
    "auto_segment_image",
    "get_object_masks",
    "object_segment_image",
    "create_patch_mask",
    "create_composite_mask",
    "mask_id_to_binary_mask",
    "convert_matched_mask",
    "aggregate_masked_features",
    # protocols
    "SupportsTextQuery",
    # frame sampling
    "OpticalFlowFrameSelector",
    "sample_frames_fps",
    "sample_frames_optical_flow",
]
```

Note: old `__all__` had `"resize_image"` in the helpers section — removed since it now lives in `utils.image`, not semantics.

- [ ] **Step 8: Run existing tests**

Run: `pytest tests/semantics/test_features_guards.py tests/semantics/test_semantics_utils.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add collab_splats/semantics/features.py collab_splats/semantics/__init__.py collab_splats/semantics/segmentation.py
git commit -m "refactor(features): extract utilities to utils/image.py and semantics/utils.py"
```

---

### Task 5: Unify `compute_similarity` on Talk2DinoExtractor

**Files:**
- Modify: `collab_splats/semantics/features.py` (Talk2DinoExtractor section)

- [ ] **Step 1: Rename `_compute_similarity` → `compute_similarity`**

In `Talk2DinoExtractor`, rename the method and inline the default negative:

```python
    def compute_similarity(
        self,
        features: torch.Tensor,
        positive: List[str],
        negative: Optional[List[str]] = None,
        softmax_temp: float = 0.05,
        method: str = "standard",
    ) -> torch.Tensor:
        """Compute per-patch similarity between image features and text queries.

        Args:
            features: patch embeddings of shape (N_patches, D).
            positive: positive text queries.
            negative: negative text queries. Defaults to ["object"].
            softmax_temp: temperature for softmax (lower = sharper).
            method: "standard" or "pairwise".

        Returns:
            Contrastive similarity scores of shape (N_patches,).
        """
        if negative is None:
            negative = ["object"]
        queries = positive + negative
        with torch.no_grad():
            text_embeddings = self._model.encode_text(queries)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        features_norm = F.normalize(features, dim=-1)
        raw_similarities = text_embeddings @ features_norm.T
        return compute_semantic_contrast(raw_similarities, len(positive), softmax_temp, method)
```

- [ ] **Step 2: Delete `compute_semantic_heatmap` entirely**

Remove the entire `compute_semantic_heatmap` method (was lines 512-575) and import of `math` at top of file (no longer needed after heatmap removal — verify no other usage first).

- [ ] **Step 3: Update Talk2DinoExtractor docstring**

```python
class Talk2DinoExtractor(BaseFeatureExtractor):
    """
    Wraps Talk2DINO models from HuggingFace Hub for patch-level feature extraction
    and text-conditioned semantic similarity.

    Supports DINOv3 (default) and DINOv2 variants:
      - "lorebianchi98/Talk2DINOv3-ViTB"  (default, cleaner interface)
      - "lorebianchi98/Talk2DINO-ViTB"    (DINOv2, older interface)
    """
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/semantics/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/features.py
git commit -m "refactor(features): unify compute_similarity, delete compute_semantic_heatmap"
```

---

### Task 6: Migrate DINOFeatureExtractor to HuggingFace

**Files:**
- Modify: `collab_splats/semantics/features.py` (DINOFeatureExtractor section)

- [ ] **Step 1: Rewrite `DINOFeatureExtractor.__init__`**

```python
@BaseFeatureExtractor.register("dinov2")
class DINOFeatureExtractor(BaseFeatureExtractor):

    def __init__(
        self, model_name: str = "facebook/dinov2-small", resolution: int = 800, device: str = "cpu"
    ):
        super().__init__()
        self.model_name = model_name

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for DINOFeatureExtractor. "
                "Install via: pip install transformers"
            ) from e

        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size: int = self.model.config.patch_size
        self.resolution = resolution

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
```

- [ ] **Step 2: Update `preprocess` to use `self.patch_size`**

```python
    def preprocess(self, image) -> Tuple[torch.Tensor, int, int]:
        image = open_image(image)
        image = resize_image(image, longest_edge=self.resolution)
        image = self.transform(image)[:3].unsqueeze(0)
        image, target_H, target_W = interpolate_to_patch_size(image, self.patch_size)
        image = image.to(self.device)
        return image, target_H, target_W
```

- [ ] **Step 3: Rewrite `forward` — use HF API, inline `_reshape`**

```python
    def forward(self, images: list) -> list[torch.Tensor]:
        """Preprocess images, extract patch features, return one (C, pH, pW) tensor per image."""
        preprocessed = [self.preprocess(img) for img in images]
        tensors = torch.cat([t for t, _, _ in preprocessed], dim=0).to(self.device)
        with torch.no_grad():
            # last_hidden_state includes CLS token at index 0; skip it for patch tokens
            features = self.model(tensors).last_hidden_state[:, 1:]
        results = []
        for i, (_, H, W) in enumerate(preprocessed):
            patch_features = features[i].cpu()
            reshaped = patch_features.reshape(
                H // self.patch_size, W // self.patch_size, -1
            ).permute(2, 0, 1)  # (C, pH, pW)
            results.append(reshaped)
        return results
```

- [ ] **Step 4: Delete `_reshape` method**

Remove the entire `_reshape` method (was lines 392-396). Its logic is now inlined in `forward`.

- [ ] **Step 5: Remove `load_torchhub_model` local import if added in Task 4**

If Task 4 added a local import of `load_torchhub_model` in `DINOFeatureExtractor.__init__`, remove it — no longer needed.

- [ ] **Step 6: Run tests**

Run: `pytest tests/semantics/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add collab_splats/semantics/features.py
git commit -m "refactor(features): migrate DINOFeatureExtractor from torch.hub to HuggingFace"
```

---

### Task 7: Update `protocols.py`

**Files:**
- Modify: `collab_splats/semantics/protocols.py`

- [ ] **Step 1: Rewrite `SupportsTextQuery` protocol**

Replace the entire file:

```python
# collab_splats/semantics/protocols.py
"""
Capability protocols for feature extractors.

Using typing.Protocol (PEP 544, Python 3.8+) for structural duck typing:
any class that implements the required methods satisfies the protocol without
explicit inheritance. @runtime_checkable enables isinstance() checks at runtime.

Example:
    from collab_splats.semantics.protocols import SupportsTextQuery
    if isinstance(extractor, SupportsTextQuery):
        sim = extractor.compute_similarity(features, ["cat"], ["background"])
"""

from typing import List, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class SupportsTextQuery(Protocol):
    """Extractor capability: text encoding and text-conditioned similarity scoring.

    Satisfied by MaskCLIPExtractor and Talk2DinoExtractor.
    Dashboard uses isinstance(extractor, SupportsTextQuery) to show/hide
    the semantic query tab without hardcoding extractor types.
    """

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings to normalized embeddings (N, D)."""
        ...

    def compute_similarity(
        self,
        features: torch.Tensor,
        positive: List[str],
        negative: Optional[List[str]],
        softmax_temp: float,
        method: str,
    ) -> torch.Tensor:
        """Compute contrastive similarity between features and text queries."""
        ...
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/semantics/ -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add collab_splats/semantics/protocols.py
git commit -m "refactor(protocols): simplify SupportsTextQuery to encode_text + compute_similarity"
```

---

### Task 8: Update backward-compat shims (`utils/features.py`, `utils/__init__.py`)

**Files:**
- Modify: `collab_splats/utils/features.py`
- Modify: `collab_splats/utils/__init__.py`

Note: `semantics/__init__.py` was already updated in Task 4 (required to keep the package importable after utility extraction).

- [ ] **Step 1: Update `utils/features.py` shim**

Update imports to pull from new locations. Keep `TwoLayerMLP` where it is:

```python
"""
Backward-compatibility shim.

Utility functions have moved:
  - Image ops → collab_splats.utils.image
  - Torch/model ops → collab_splats.semantics.utils
  - Extractors → collab_splats.semantics.features

This module re-exports everything so existing callers require no changes.
TwoLayerMLP stays here — it is a splatting decoder layer, not a feature extractor.
"""

from collab_splats.semantics.features import (
    BaseFeatureExtractor,
    MaskCLIPExtractor,
    DINOFeatureExtractor,
    Talk2DinoExtractor,
)
from collab_splats.utils.image import (
    open_image,
    resize_image,
)
from collab_splats.semantics.utils import (
    compute_semantic_contrast,
    interpolate_to_patch_size,
    pytorch_gc,
    infer_batch_size,
    load_hf_weights,
    load_torchhub_model,
    batch_iterator,
)
```

Keep the existing `TORCH_HOME` re-export by importing from features:

```python
from collab_splats.semantics.features import TORCH_HOME
```

Keep `TwoLayerMLP` class and existing `torch`/`torchvision` imports for it — unchanged.

Update `__all__` to include `open_image` (renamed from `_open_image`) and `compute_semantic_contrast`:

```python
__all__ = [
    "BaseFeatureExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    "TwoLayerMLP",
    "open_image",
    "resize_image",
    "compute_semantic_contrast",
    "interpolate_to_patch_size",
    "pytorch_gc",
    "infer_batch_size",
    "load_hf_weights",
    "load_torchhub_model",
    "batch_iterator",
    "TORCH_HOME",
]
```

- [ ] **Step 3: Update `utils/__init__.py`**

Add image.py exports:

```python
from .camera_utils import ColmapCamera, convert_to_colmap_camera, depth_double_to_normal
from .frame_sampling import OpticalFlowFrameSelector, sample_frames_fps, sample_frames_optical_flow
from .image import open_image, resize_image

__all__ = [
    "ColmapCamera",
    "convert_to_colmap_camera",
    "depth_double_to_normal",
    "OpticalFlowFrameSelector",
    "sample_frames_fps",
    "sample_frames_optical_flow",
    "open_image",
    "resize_image",
]
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v --ignore=tests/nerfstudio`
Expected: PASS (nerfstudio tests have pre-existing env failures)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/__init__.py collab_splats/utils/features.py collab_splats/utils/__init__.py
git commit -m "refactor(imports): update re-exports for new utility locations"
```

---

### Task 9: Update `nerfstudio/datamanagers/features.py`

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py:29-30,159`

- [ ] **Step 1: Update imports**

Change:
```python
from collab_splats.semantics.features import BaseFeatureExtractor, infer_batch_size
from collab_splats.utils.features import pytorch_gc, resize_image
```

To:
```python
from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.semantics.utils import infer_batch_size, pytorch_gc
from collab_splats.utils.image import resize_image
```

- [ ] **Step 2: Update batch size inference to use `_get_memory_per_image`**

Find the line (was line 159):
```python
batch_size = infer_batch_size(extractor.MEM_PER_IMAGE_GB)
```

Replace with:
```python
sample_image = Image.open(image_filenames[0])
mem_per_image = extractor._get_memory_per_image(sample_image)
batch_size = infer_batch_size(mem_per_image)
```

Add `from PIL import Image` to imports if not already present (check — it's already imported on line 8).

- [ ] **Step 3: Commit**

```bash
git add collab_splats/nerfstudio/datamanagers/features.py
git commit -m "refactor(datamanager): use _get_memory_per_image for dynamic batch sizing"
```

---

### Task 10: Update `dashboard/semantics.py`

**Files:**
- Modify: `collab_splats/dashboard/semantics.py:650-680`

- [ ] **Step 1: Replace `compute_semantic_heatmap` call with inline logic**

Find the try block around line 656-675. Replace:

```python
        try:
            from collab_splats.semantics.features import Talk2DinoExtractor

            extractor = Talk2DinoExtractor(hf_model_id=self.hf_model_dd.value, device=self.query_device_dd.value)
            heatmaps = extractor.compute_semantic_heatmap(
                Image.fromarray(self._current_frame),
                text_pairs,
                self.temp_slider.value,
                self.method_dd.value,
            )
```

With:

```python
        try:
            import math
            import torch.nn.functional as F
            from collab_splats.semantics.features import Talk2DinoExtractor

            extractor = Talk2DinoExtractor(hf_model_id=self.hf_model_dd.value, device=self.query_device_dd.value)
            image_pil = extractor.preprocess(Image.fromarray(self._current_frame))
            with torch.no_grad():
                img_embed = extractor._model.encode_image([image_pil])[0]

            num_patches = img_embed.shape[0]
            grid_size = int(math.isqrt(num_patches))
            img_size = grid_size * extractor.patch_size

            img_np = np.array(image_pil).transpose(2, 0, 1).astype(np.float32)
            img_np = (
                F.interpolate(
                    torch.tensor(img_np).unsqueeze(0),
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .numpy()
                / 255.0
            )

            heatmaps = {}
            for label, (positive, negative) in text_pairs.items():
                sim = extractor.compute_similarity(img_embed, positive, negative, self.temp_slider.value, self.method_dd.value)
                mask = (
                    F.interpolate(
                        sim.view(1, 1, grid_size, grid_size),
                        size=(img_size, img_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .cpu()
                    .squeeze()
                    .numpy()
                )
                heatmaps[label] = (img_np * mask[np.newaxis, :, :]).transpose(1, 2, 0)
```

The rest of the handler (building `items` list from `heatmaps`) stays unchanged.

- [ ] **Step 2: Verify dashboard imports**

Confirm `torch` and `np` are already imported or available in scope in `dashboard/semantics.py`. If not, add them.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/dashboard/semantics.py
git commit -m "refactor(dashboard): inline semantic query logic, remove compute_semantic_heatmap dependency"
```

---

### Task 11: Update `tests/test_models.py`

**Files:**
- Modify: `tests/test_models.py:27,29`

- [ ] **Step 1: Replace `clip-vit` with `samclip`**

Line 27: `"feature_type": "clip-vit",` → `"feature_type": "samclip",`
Line 29: `"clip-vit": (channels, height, width),` → `"samclip": (channels, height, width),`

- [ ] **Step 2: Run test**

Run: `pytest tests/test_models.py -v`
Expected: PASS (or pre-existing nerfstudio env failure — not caused by this change)

- [ ] **Step 3: Commit**

```bash
git add tests/test_models.py
git commit -m "fix(tests): update clip-vit alias to samclip registry name"
```

---

### Task 12: Final validation

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/nerfstudio 2>&1 | tail -30
```

Expected: All tests pass (excluding pre-existing nerfstudio env failures).

- [ ] **Step 2: Run import smoke test**

```bash
python -c "
from collab_splats.semantics import (
    BaseFeatureExtractor, MaskCLIPExtractor, DINOFeatureExtractor, Talk2DinoExtractor,
    compute_semantic_contrast, interpolate_to_patch_size, pytorch_gc, infer_batch_size,
    load_hf_weights, load_torchhub_model, batch_iterator, SupportsTextQuery,
)
from collab_splats.utils.image import open_image, resize_image
from collab_splats.utils.features import TwoLayerMLP, TORCH_HOME
from collab_splats.semantics.protocols import SupportsTextQuery
print('All imports OK')
print('Registry:', list(BaseFeatureExtractor._registry.keys()))
assert 'clip-vit' not in BaseFeatureExtractor._registry
assert 'samclip' in BaseFeatureExtractor._registry
assert 'dinov2' in BaseFeatureExtractor._registry
assert 'talk2dino' in BaseFeatureExtractor._registry
print('Registry clean — no clip-vit alias')
"
```

Expected: `All imports OK`, `Registry clean — no clip-vit alias`

- [ ] **Step 3: Verify features.py line count**

```bash
wc -l collab_splats/semantics/features.py
```

Expected: ~300-350 lines (down from ~575).

- [ ] **Step 4: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "refactor(semantics): final cleanup for features refactor"
```
