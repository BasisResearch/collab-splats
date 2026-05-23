# PE-CLIP + SAM3 Open-Label Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PE-CLIP global frame retrieval, SAM3 text-prompted segmentation, 3D OBB lifting, and refactor `Segmentation` into a `BaseSegmentation` registry hierarchy.

**Architecture:** `PECLIPExtractor` extends `BaseRetrievalExtractor` (global image+text descriptors, lives in `localization.py`). `BaseSegmentation` + `MobileSAMSegmentation` + `SAM3Segmentation` replace the existing string-dispatch `Segmentation` class. `compute_obb_from_points` and `get_points_in_mask` are pure numpy utilities added to `pointcloud/utils.py`.

**Tech Stack:** `open-clip-torch` (PE-Core via HuggingFace hub), `sam3` (Meta, requires HF login), `numpy` (OBB via PCA), existing `RegistryMixin` pattern.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `collab_splats/pointcloud/utils.py` | Modify | Add `compute_obb_from_points`, `get_points_in_mask` |
| `collab_splats/pointcloud/__init__.py` | Modify | Export new utils, `PECLIPExtractor` |
| `collab_splats/pointcloud/localization.py` | Modify | Add `PECLIPExtractor` |
| `collab_splats/semantics/segmentation.py` | Modify | Replace `Segmentation` with `BaseSegmentation` + `MobileSAMSegmentation` + `SAM3Segmentation` |
| `collab_splats/semantics/__init__.py` | Modify | Update exports |
| `collab_splats/nerfstudio/datamanagers/features.py` | Modify | `Segmentation` → `MobileSAMSegmentation` |
| `stage/grouping.py` | Modify | `Segmentation` → `MobileSAMSegmentation` |
| `requirements.txt` | Modify | Add `open-clip-torch` |
| `tests/pointcloud/test_pointcloud_utils.py` | Modify | Add OBB + mask tests |
| `tests/pointcloud/test_retrieval.py` | Modify | Add `PECLIPExtractor` registry + interface tests |
| `tests/semantics/test_segmentation.py` | Modify | Add `BaseSegmentation` registry + `SAM3Segmentation` interface tests |
| `tests/nerfstudio/test_datamanager_config.py` | Modify | Update patch target `Segmentation` → `MobileSAMSegmentation` |

---

## Task 1: `compute_obb_from_points` and `get_points_in_mask`

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`
- Modify: `collab_splats/pointcloud/__init__.py`
- Modify: `tests/pointcloud/test_pointcloud_utils.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_pointcloud_utils.py`:

```python
########################################################
########## compute_obb_from_points #####################
########################################################

def test_compute_obb_axis_aligned_cube():
    """Axis-aligned cube: center at origin, extents all 2.0, rotation near identity."""
    from collab_splats.pointcloud.utils import compute_obb_from_points
    pts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    ], dtype=np.float32)
    center, extent, rotation = compute_obb_from_points(pts)
    np.testing.assert_allclose(center, [0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(sorted(extent), sorted([2.0, 2.0, 2.0]), atol=1e-4)
    # rotation must be orthonormal
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)


def test_compute_obb_all_nan_raises():
    from collab_splats.pointcloud.utils import compute_obb_from_points
    pts = np.full((10, 3), np.nan)
    with pytest.raises(ValueError, match="empty or invalid"):
        compute_obb_from_points(pts)


def test_compute_obb_empty_raises():
    from collab_splats.pointcloud.utils import compute_obb_from_points
    pts = np.zeros((0, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="empty or invalid"):
        compute_obb_from_points(pts)


########################################################
########## get_points_in_mask ##########################
########################################################

def test_get_points_in_mask_basic():
    from collab_splats.pointcloud.utils import get_points_in_mask
    # 4 points from frame 0, 4 from frame 1
    points = np.arange(24, dtype=np.float32).reshape(8, 3)
    pixel_indices = np.array([
        [0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3],
        [1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3],
    ], dtype=np.int32)
    # mask: only (row=1,col=1) and (row=3,col=3) are True in frame 0
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[3, 3] = True
    result = get_points_in_mask(0, mask, points, pixel_indices)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, points[[1, 3]])


def test_get_points_in_mask_no_match():
    from collab_splats.pointcloud.utils import get_points_in_mask
    points = np.zeros((4, 3), dtype=np.float32)
    pixel_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=np.int32)
    mask = np.zeros((5, 5), dtype=bool)  # nothing selected
    result = get_points_in_mask(0, mask, points, pixel_indices)
    assert result.shape == (0, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py -k "obb or mask" -v 2>&1 | tail -20
```
Expected: ImportError or `FAILED` — functions don't exist yet.

- [ ] **Step 3: Implement `compute_obb_from_points` and `get_points_in_mask`**

Add to the bottom of `collab_splats/pointcloud/utils.py` (before end of file, after `reproject_pixels`):

```python
########################################################
########## Geometry: OBB + mask lifting ################
########################################################

def compute_obb_from_points(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute oriented bounding box for a Nx3 point cloud via PCA.

    Returns:
        center   : (3,) world-space OBB center
        extent   : (3,) box side lengths along principal axes
        rotation : (3,3) rotation matrix, columns = principal axes
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be Nx3"
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty or invalid")

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    rotation = eigvecs

    points_local = centered @ rotation
    min_corner = points_local.min(axis=0)
    max_corner = points_local.max(axis=0)
    extent = max_corner - min_corner

    center_local = 0.5 * (min_corner + max_corner)
    center = centroid + center_local @ rotation.T

    return center, extent, rotation


def get_points_in_mask(
    frame_idx: int,
    mask: np.ndarray,
    points: np.ndarray,
    pixel_indices: np.ndarray,
) -> np.ndarray:
    """Return world-space points whose source pixel falls within a 2D mask.

    Args:
        frame_idx:     Frame to query.
        mask:          (H, W) bool array — True for pixels of interest.
        points:        (P, 3) float32 world-space point positions.
        pixel_indices: (P, 3) int32 [frame_id, row, col] per point.

    Returns:
        (M, 3) float32 — subset of points with source pixel inside mask, M ≤ P.
    """
    frame_mask = pixel_indices[:, 0] == frame_idx
    rows = pixel_indices[frame_mask, 1]
    cols = pixel_indices[frame_mask, 2]
    in_mask = mask[rows, cols]
    return points[frame_mask][in_mask]
```

- [ ] **Step 4: Update `collab_splats/pointcloud/__init__.py`**

Add imports and exports. Find the existing localization import line and add utils imports:

```python
from .utils import (
    colmap_reconstruction_to_result,
    compute_obb_from_points,
    get_points_in_mask,
    lift_features,
    reproject_pixels,
)
```

Add to `__all__`:
```python
"compute_obb_from_points",
"get_points_in_mask",
"lift_features",
"reproject_pixels",
```

- [ ] **Step 5: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py -k "obb or mask" -v 2>&1 | tail -20
```
Expected: all 5 new tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/pointcloud/utils.py collab_splats/pointcloud/__init__.py tests/pointcloud/test_pointcloud_utils.py && git commit -m "feat(pointcloud): add compute_obb_from_points and get_points_in_mask"
```

---

## Task 2: Refactor `Segmentation` → `BaseSegmentation` + `MobileSAMSegmentation`

**Files:**
- Modify: `collab_splats/semantics/segmentation.py`
- Modify: `collab_splats/semantics/__init__.py`
- Modify: `collab_splats/nerfstudio/datamanagers/features.py`
- Modify: `stage/grouping.py`
- Modify: `tests/semantics/test_segmentation.py`
- Modify: `tests/nerfstudio/test_datamanager_config.py`

- [ ] **Step 1: Write failing registry tests**

Append to `tests/semantics/test_segmentation.py`:

```python
########################################################
########## BaseSegmentation registry ##################
########################################################

def test_registry_get_mobilesamv2():
    from collab_splats.semantics.segmentation import BaseSegmentation, MobileSAMSegmentation
    cls = BaseSegmentation.get("mobilesamv2")
    assert cls is MobileSAMSegmentation


def test_registry_unknown_raises():
    from collab_splats.semantics.segmentation import BaseSegmentation
    with pytest.raises((KeyError, ValueError)):
        BaseSegmentation.get("nonexistent-backend")


def test_mobilesamv2_segment_with_text_raises():
    from collab_splats.semantics.segmentation import MobileSAMSegmentation
    from unittest.mock import patch, MagicMock
    with patch("collab_splats.semantics.segmentation.load_mobile_sam",
               return_value=(MagicMock(), MagicMock(), MagicMock())):
        seg = MobileSAMSegmentation.__new__(MobileSAMSegmentation)
        seg.seg_model = MagicMock()
        seg.object_model = MagicMock()
        seg.predictor = MagicMock()
        seg.strategy = "object"
        with pytest.raises(NotImplementedError, match="does not support text-prompted"):
            seg.segment_with_text(MagicMock(), "a dog")


def test_base_segmentation_abstract():
    from collab_splats.semantics.segmentation import BaseSegmentation
    import pytest
    with pytest.raises(TypeError):
        BaseSegmentation()
```

Add `import pytest` at top of test file if not already present.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_segmentation.py -k "registry or abstract or text_raises" -v 2>&1 | tail -20
```
Expected: ImportError or `FAILED` — `BaseSegmentation` doesn't exist yet.

- [ ] **Step 3: Refactor `segmentation.py`**

Replace the `Segmentation` class at the top of `collab_splats/semantics/segmentation.py`. Add the following imports at the top:

```python
from abc import ABC, abstractmethod
from typing import Any
```

Replace the existing `Segmentation` class definition (lines 1 through the end of `def segment`) with:

```python
########################################################
########## Registry and abstract base ##################
########################################################

from collab_splats.utils.torch_utils import RegistryMixin


class BaseSegmentation(RegistryMixin, ABC):
    """Abstract base for segmentation backends with name-based registry.

    Register subclasses via ``@BaseSegmentation.register("name")``.
    Retrieve with ``BaseSegmentation.get("name")``.
    """

    _registry: dict[str, type["BaseSegmentation"]] = {}

    @abstractmethod
    def segment(self, image) -> tuple[torch.Tensor, Any]:
        """Class-agnostic segmentation. Returns (masks, metadata)."""

    def segment_with_text(
        self,
        image,
        prompt: str,
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Text-prompted segmentation → (masks, boxes, scores).

        Raises NotImplementedError for backends that don't support text prompts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support text-prompted segmentation. "
            "Use backend='sam3'."
        )


########################################################
########## MobileSAMv2 backend #########################
########################################################

@BaseSegmentation.register("mobilesamv2")
class MobileSAMSegmentation(BaseSegmentation):
    """MobileSAMv2 class-agnostic segmentation (object or auto strategy)."""

    def __init__(
        self,
        strategy: str = "object",
        device: str = "cpu",
        mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2",
    ):
        self.seg_model, self.object_model, self.predictor = load_mobile_sam(
            mobilesam_encoder_name, device
        )
        self.strategy = strategy

    def segment(self, image) -> tuple[torch.Tensor, Any]:
        if self.strategy == "object":
            return object_segment_image(
                image, self.seg_model, self.object_model, self.predictor
            )
        elif self.strategy == "auto":
            return auto_segment_image(image, self.seg_model)
        else:
            raise ValueError(
                f"Strategy '{self.strategy}' not supported. Available: ['object', 'auto']"
            )
```

The existing free functions (`load_mobile_sam`, `auto_segment_image`, etc.) stay unchanged below the class definitions.

- [ ] **Step 4: Update `collab_splats/semantics/__init__.py`**

Replace the segmentation imports section:

```python
from .segmentation import (
    BaseSegmentation,
    MobileSAMSegmentation,
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
```

Update `__all__` — remove `"Segmentation"`, add:
```python
"BaseSegmentation",
"MobileSAMSegmentation",
```

- [ ] **Step 5: Update `collab_splats/nerfstudio/datamanagers/features.py`**

Change line 32:
```python
# Before:
from collab_splats.semantics.segmentation import Segmentation, aggregate_masked_features
# After:
from collab_splats.semantics.segmentation import MobileSAMSegmentation, aggregate_masked_features
```

Change instantiation (around line 194):
```python
# Before:
segmentation = Segmentation(
    backend=self.config.segmentation_backend,
    strategy=self.config.segmentation_strategy,
    device=device,
)
# After:
segmentation = MobileSAMSegmentation(
    strategy=self.config.segmentation_strategy,
    device=device,
)
```

- [ ] **Step 6: Update `stage/grouping.py`**

Change import:
```python
# Before:
from collab_splats.semantics.segmentation import (
    Segmentation,
    create_patch_mask,
    ...
)
# After:
from collab_splats.semantics.segmentation import (
    MobileSAMSegmentation,
    create_patch_mask,
    create_composite_mask,
    mask_id_to_binary_mask,
    convert_matched_mask,
)
```

Change instantiation (around line 174):
```python
# Before:
self.segmentation = Segmentation(
    backend=self.params.segmentation_backend,
    strategy=self.params.segmentation_strategy,
    device=self.model.device,
)
# After:
self.segmentation = MobileSAMSegmentation(
    strategy=self.params.segmentation_strategy,
    device=self.model.device,
)
```

- [ ] **Step 7: Update `tests/nerfstudio/test_datamanager_config.py`**

Change the patch target string at line 82:
```python
# Before:
patch("collab_splats.nerfstudio.datamanagers.features.Segmentation") as mock_seg_cls,
# After:
patch("collab_splats.nerfstudio.datamanagers.features.MobileSAMSegmentation") as mock_seg_cls,
```

- [ ] **Step 8: Run full test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_segmentation.py tests/nerfstudio/test_datamanager_config.py -v 2>&1 | tail -30
```
Expected: all tests PASS (new registry tests + existing composite mask tests + datamanager test).

- [ ] **Step 9: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/semantics/segmentation.py collab_splats/semantics/__init__.py collab_splats/nerfstudio/datamanagers/features.py stage/grouping.py tests/semantics/test_segmentation.py tests/nerfstudio/test_datamanager_config.py && git commit -m "refactor(semantics): replace Segmentation with BaseSegmentation registry hierarchy"
```

---

## Task 3: `SAM3Segmentation`

**Files:**
- Modify: `collab_splats/semantics/segmentation.py`
- Modify: `collab_splats/semantics/__init__.py`
- Modify: `tests/semantics/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/semantics/test_segmentation.py`:

```python
########################################################
########## SAM3Segmentation ###########################
########################################################

def test_registry_get_sam3():
    from collab_splats.semantics.segmentation import BaseSegmentation, SAM3Segmentation
    cls = BaseSegmentation.get("sam3")
    assert cls is SAM3Segmentation


def test_sam3_segment_with_text_interface():
    """segment_with_text exists and calls SAM3 processor — mocked to avoid loading model."""
    from collab_splats.semantics.segmentation import SAM3Segmentation
    from unittest.mock import patch, MagicMock
    import torch

    mock_masks = torch.zeros(2, 1, 4, 4)
    mock_boxes = torch.zeros(2, 4)
    mock_scores = torch.ones(2)
    mock_output = {"masks": mock_masks, "boxes": mock_boxes, "scores": mock_scores}

    mock_processor = MagicMock()
    mock_processor.set_image.return_value = "state"
    mock_processor.set_text_prompt.return_value = mock_output

    with patch("collab_splats.semantics.segmentation.build_sam3_image_model",
               return_value=MagicMock()):
        with patch("collab_splats.semantics.segmentation.Sam3Processor",
                   return_value=mock_processor):
            seg = SAM3Segmentation(confidence_threshold=0.5)

    from PIL import Image
    fake_img = Image.new("RGB", (4, 4))
    masks, boxes, scores = seg.segment_with_text(fake_img, "a cat")

    mock_processor.set_image.assert_called_once_with(fake_img)
    mock_processor.set_text_prompt.assert_called_once_with(state="state", prompt="a cat")
    assert masks.shape == (2, 1, 4, 4)
    assert boxes.shape == (2, 4)
    assert scores.shape == (2,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_segmentation.py -k "sam3" -v 2>&1 | tail -20
```
Expected: ImportError or `FAILED` — `SAM3Segmentation` doesn't exist yet.

- [ ] **Step 3: Implement `SAM3Segmentation`**

Add to `collab_splats/semantics/segmentation.py`, after `MobileSAMSegmentation`:

```python
########################################################
########## SAM3 backend ################################
########################################################

@BaseSegmentation.register("sam3")
class SAM3Segmentation(BaseSegmentation):
    """SAM3 text-prompted segmentation backend.

    Requires sam3 package (facebook/sam3 on HuggingFace Hub — gated model,
    requires accepting Meta's license).

    Args:
        confidence_threshold: Minimum score for returned masks (default 0.5).
        device: Torch device string (default 'cuda').
    """

    def __init__(self, confidence_threshold: float = 0.5, device: str = "cuda"):
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError:
            raise ImportError(
                "sam3 is not installed. Install from https://github.com/facebookresearch/sam3 "
                "after accepting the Meta license at https://huggingface.co/facebook/sam3"
            )
        sam3_model = build_sam3_image_model()
        self._processor = Sam3Processor(
            sam3_model, confidence_threshold=confidence_threshold
        )
        self._confidence_threshold = confidence_threshold

    def segment(self, image) -> tuple[torch.Tensor, Any]:
        """Auto-segment all objects without a text prompt."""
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt="object")
        return output["masks"], output

    def segment_with_text(
        self,
        image,
        prompt: str,
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Text-prompted segmentation.

        Args:
            image:                PIL Image.
            prompt:               Text prompt, e.g. "red tractor".
            confidence_threshold: Unused (set at construction time via Sam3Processor).

        Returns:
            masks  : (N, 1, H, W) float32
            boxes  : (N, 4) float32
            scores : (N,) float32
        """
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=prompt)
        return output["masks"], output["boxes"], output["scores"]
```

Also add `build_sam3_image_model` and `Sam3Processor` to the imports section at the top as lazy imports (already handled inside `__init__`).

- [ ] **Step 4: Update `collab_splats/semantics/__init__.py`**

Add `SAM3Segmentation` to the import and `__all__`:

```python
from .segmentation import (
    BaseSegmentation,
    MobileSAMSegmentation,
    SAM3Segmentation,
    ...
)
```

Add `"SAM3Segmentation"` to `__all__`.

- [ ] **Step 5: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_segmentation.py -v 2>&1 | tail -20
```
Expected: all tests PASS (no real model loaded — mocked).

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/semantics/segmentation.py collab_splats/semantics/__init__.py tests/semantics/test_segmentation.py && git commit -m "feat(semantics): add SAM3Segmentation backend with text-prompted segmentation"
```

---

## Task 4: `PECLIPExtractor`

**Files:**
- Modify: `requirements.txt`
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `collab_splats/pointcloud/__init__.py`
- Modify: `tests/pointcloud/test_retrieval.py`

- [ ] **Step 1: Add `open-clip-torch` to requirements**

Append to `requirements.txt`:
```
open-clip-torch
```

Install:
```bash
/opt/conda/envs/nerfstudio/bin/pip install open-clip-torch 2>&1 | tail -5
```

- [ ] **Step 2: Write failing tests**

Append to `tests/pointcloud/test_retrieval.py`:

```python
########################################################
########## PECLIPExtractor ############################
########################################################

def test_registry_get_pe_clip():
    from collab_splats.pointcloud.localization import BaseRetrievalExtractor, PECLIPExtractor
    cls = BaseRetrievalExtractor.get("pe-clip")
    assert cls is PECLIPExtractor


def test_pe_clip_forward_shape_and_norm(tmp_path):
    """forward() returns (1, 1024) unit-norm tensor without loading real weights."""
    from collab_splats.pointcloud.localization import PECLIPExtractor
    from unittest.mock import patch, MagicMock
    import torch
    from PIL import Image

    fake_img_emb = torch.randn(1, 1024)
    fake_img_emb = fake_img_emb / fake_img_emb.norm(dim=-1, keepdim=True)

    mock_model = MagicMock()
    mock_model.encode_image.return_value = fake_img_emb
    mock_model.context_length = 32

    mock_preprocess = MagicMock(return_value=torch.zeros(3, 336, 336))

    with patch("collab_splats.pointcloud.localization.open_clip") as mock_oc:
        mock_oc.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_oc.get_tokenizer.return_value = MagicMock(return_value=torch.zeros(1, 32, dtype=torch.long))
        extractor = PECLIPExtractor(device="cpu")

    img = Image.new("RGB", (336, 336))
    result = extractor([img])

    assert result.shape == (1, 1024)
    norms = result.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(1), atol=1e-5, rtol=0)


def test_pe_clip_encode_text_shape_and_norm():
    """encode_text() returns (2, 1024) unit-norm tensor."""
    from collab_splats.pointcloud.localization import PECLIPExtractor
    from unittest.mock import patch, MagicMock
    import torch

    fake_text_emb = torch.randn(2, 1024)
    fake_text_emb = fake_text_emb / fake_text_emb.norm(dim=-1, keepdim=True)

    mock_model = MagicMock()
    mock_model.encode_text.return_value = fake_text_emb
    mock_model.context_length = 32

    with patch("collab_splats.pointcloud.localization.open_clip") as mock_oc:
        mock_oc.create_model_and_transforms.return_value = (mock_model, None, MagicMock())
        mock_tokenizer = MagicMock(return_value=torch.zeros(2, 32, dtype=torch.long))
        mock_oc.get_tokenizer.return_value = mock_tokenizer
        extractor = PECLIPExtractor(device="cpu")

    result = extractor.encode_text(["a cat", "a dog"])

    assert result.shape == (2, 1024)
    norms = result.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(2), atol=1e-5, rtol=0)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_retrieval.py -k "pe_clip" -v 2>&1 | tail -20
```
Expected: ImportError or `FAILED` — `PECLIPExtractor` doesn't exist yet.

- [ ] **Step 4: Implement `PECLIPExtractor` in `localization.py`**

Add `import open_clip` at the top of `collab_splats/pointcloud/localization.py` with the other imports:

```python
import open_clip
```

Add `PECLIPExtractor` after the `DinoSaladExtractor` class:

```python
########################################################
########## PE-CLIP retrieval extractor #################
########################################################

@BaseRetrievalExtractor.register("pe-clip")
class PECLIPExtractor(BaseRetrievalExtractor):
    """PE-Core-L/14-336 global image+text encoder for open-label frame retrieval.

    Uses open_clip with hf-hub:timm/PE-Core-L-14-336. Produces (N, 1024)
    normalized descriptors for both images and text — aligned in the same
    CLIP embedding space. Suitable for text-driven frame retrieval.

    Architecture note: PE-Core uses AttentionPoolLatent (pool='map') with no
    separate linear projection. Patch-level CLIP-aligned features are not
    available; use this extractor for global retrieval only.
    """

    _MODEL_ID = "hf-hub:timm/PE-Core-L-14-336"

    def __init__(self, model_id: str = _MODEL_ID, device: str | None = None):
        super().__init__()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_id)
        self._model = self._model.to(self._device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_id)
        logger.debug("PECLIPExtractor: loaded %s on %s", model_id, self._device)

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """Return (N, 1024) normalized image descriptors.

        Args:
            images: List of PIL Images or (N, 3, H, W) float32 tensor.

        Returns:
            (N, 1024) float32, L2-normalized, on CPU.
        """
        if isinstance(images, (list, tuple)):
            imgs = torch.stack([self._preprocess(img) for img in images])
        else:
            imgs = images
        imgs = imgs.to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(imgs, normalize=True)
        return features.cpu().float()

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return (N, 1024) normalized text descriptors.

        Args:
            texts: List of text strings.

        Returns:
            (N, 1024) float32, L2-normalized, on CPU.
        """
        tokens = self._tokenizer(texts, context_length=self._model.context_length)
        tokens = tokens.to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens, normalize=True)
        return features.cpu().float()
```

- [ ] **Step 5: Update `collab_splats/pointcloud/__init__.py`**

Add `PECLIPExtractor` to the localization import line:

```python
from .localization import (
    BaseRetrievalExtractor,
    CameraLocalizer,
    DiskExtractor,
    LocalFeatures,
    PECLIPExtractor,
    XFeatExtractor,
)
```

Add `"PECLIPExtractor"` to `__all__`.

- [ ] **Step 6: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_retrieval.py -v 2>&1 | tail -20
```
Expected: all tests PASS (mocked, no real model download).

- [ ] **Step 7: Run full suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x --ignore=tests/integration -q 2>&1 | tail -20
```
Expected: all tests pass. Fix any regressions before committing.

- [ ] **Step 8: Commit**

```bash
cd /workspace/collab-splats && git add requirements.txt collab_splats/pointcloud/localization.py collab_splats/pointcloud/__init__.py tests/pointcloud/test_retrieval.py && git commit -m "feat(pointcloud): add PECLIPExtractor for global image+text retrieval"
```

---

## Task 5: Update notebooks

**Files:**
- Modify: `docs/semantics/segmentation.ipynb`
- Modify: `stage/extract-colmap-vggt.ipynb`

- [ ] **Step 1: Update `docs/semantics/segmentation.ipynb`**

Find cells that import/use `Segmentation` and update:

```python
# Before:
from collab_splats.semantics import Segmentation
seg_object = Segmentation(backend="mobilesamv2", strategy="object", device=device)

# After:
from collab_splats.semantics import MobileSAMSegmentation
seg_object = MobileSAMSegmentation(strategy="object", device=device)
```

- [ ] **Step 2: Update `stage/extract-colmap-vggt.ipynb`**

Find any cells referencing `Segmentation` or `segmentation_backend='mobilesamv2'` as a class arg and update to use `MobileSAMSegmentation` directly.

- [ ] **Step 3: Verify notebooks parse**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "
import json
for nb in ['docs/semantics/segmentation.ipynb', 'stage/extract-colmap-vggt.ipynb']:
    try:
        data = json.load(open(nb))
        src = ' '.join(
            ''.join(c['source']) for c in data['cells'] if c['cell_type']=='code'
        )
        assert 'from collab_splats.semantics import Segmentation' not in src, f'{nb}: old import found'
        assert 'Segmentation(' not in src or 'MobileSAM' in src, f'{nb}: old Segmentation() call found'
        print(f'OK: {nb}')
    except FileNotFoundError:
        print(f'SKIP (not found): {nb}')
"
```
Expected: `OK` for each file found.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats && git add docs/semantics/segmentation.ipynb stage/extract-colmap-vggt.ipynb && git commit -m "docs: update notebooks for BaseSegmentation refactor"
```

---

## Task 6: Final validation

- [ ] **Step 1: Run full test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x --ignore=tests/integration -q 2>&1 | tail -30
```
Expected: all tests pass, 0 errors, 0 failures.

- [ ] **Step 2: Verify no old `Segmentation` references remain in production code**

```bash
cd /workspace/collab-splats && grep -rn 'from collab_splats.*import.*\bSegmentation\b\|import Segmentation' collab_splats/ stage/ --include='*.py' | grep -v __pycache__
```
Expected: no output (all old references gone).

- [ ] **Step 3: Smoke-test imports**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.semantics import BaseSegmentation, MobileSAMSegmentation, SAM3Segmentation
from collab_splats.pointcloud import BaseRetrievalExtractor, PECLIPExtractor, compute_obb_from_points, get_points_in_mask
print('BaseSegmentation registry:', list(BaseSegmentation._registry.keys()))
print('BaseRetrievalExtractor registry:', list(BaseRetrievalExtractor._registry.keys()))
print('All imports OK')
"
```
Expected output:
```
BaseSegmentation registry: ['mobilesamv2', 'sam3']
BaseRetrievalExtractor registry: ['dino-salad', 'pe-clip']
All imports OK
```

- [ ] **Step 4: Commit if any cleanup was needed**

```bash
cd /workspace/collab-splats && git status && git add -p && git commit -m "fix: final cleanup after PE-CLIP + SAM3 integration"
```
(Only needed if Step 1-3 revealed issues.)
