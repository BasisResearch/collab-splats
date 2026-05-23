# Unified Extractor Preprocessing Interface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify `preprocess(image) -> Tensor` + single `forward()` path across DINOv2, MaskCLIP, and Talk2DINO; fix normalization stats for MaskCLIP (CLIP stats) and DINOv2 (ImageNet stats); add `resize_mode`/`image_resolution` params to all extractors.

**Architecture:** Each extractor gets a single `preprocess(image) -> (C, H, W) CPU Tensor` that handles both `resize_mode` branches, and a single `forward()` that stacks tensors → batched backbone call → `_tokens_to_feature_map` loop. Module-level `_tokens_to_feature_map(tokens, H, W, patch_size)` is added to `collab_splats/semantics/utils.py` and shared across all three extractors.

**Tech Stack:** PyTorch, torchvision, transformers, maskclip_onnx, PIL

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/semantics/utils.py` | Add `_tokens_to_feature_map` |
| `collab_splats/semantics/features/dino.py` | Fix norm; add `resize_mode`; `preprocess` → tensor; unified `forward` |
| `collab_splats/semantics/features/maskclip.py` | Fix norm (CLIP stats); add `resize_mode`; unified `forward` |
| `collab_splats/semantics/features/talk2dino.py` | Collapse `_forward_max_size`/`_forward_square` → single `preprocess`+`forward` |
| `tests/semantics/test_extractor_preprocessing.py` | New test file |

---

### Task 1: Add `_tokens_to_feature_map` to utils

**Files:**
- Modify: `collab_splats/semantics/utils.py`
- Test: `tests/semantics/test_extractor_preprocessing.py`

- [ ] **Step 1: Write failing test**

Create `tests/semantics/test_extractor_preprocessing.py`:

```python
"""Tests for unified extractor preprocessing utilities and interface."""
import torch
import torch.nn.functional as F
import pytest
from PIL import Image

from collab_splats.semantics.utils import _tokens_to_feature_map


def test_tokens_to_feature_map_shape():
    patch_size = 14
    H, W = 196, 280  # multiples of 14
    D = 8
    ph, pw = H // patch_size, W // patch_size
    tokens = torch.randn(ph * pw, D)
    out = _tokens_to_feature_map(tokens, H, W, patch_size)
    assert out.shape == (D, ph, pw)


def test_tokens_to_feature_map_l2_normalized():
    patch_size = 14
    H, W = 196, 196
    D = 8
    ph, pw = H // patch_size, W // patch_size
    tokens = torch.randn(ph * pw, D) * 10  # large values
    out = _tokens_to_feature_map(tokens, H, W, patch_size)
    # F.normalize(feat, dim=0) normalizes along channel dim per spatial position
    norms = out.norm(dim=0)  # (H_p, W_p)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_tokens_to_feature_map_wrong_count_raises():
    with pytest.raises(AssertionError):
        _tokens_to_feature_map(torch.randn(99, 8), 196, 196, 14)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py -v
```

Expected: `ImportError` or `AttributeError` — `_tokens_to_feature_map` not yet defined.

- [ ] **Step 3: Implement in utils.py**

Open `collab_splats/semantics/utils.py` and add (at end of module, before or after `interpolate_to_patch_size`):

```python
def _tokens_to_feature_map(
    tokens: torch.Tensor, input_h: int, input_w: int, patch_size: int
) -> torch.Tensor:
    """Reshape (N, D) patch tokens to (D, H_p, W_p), L2-normalized along channel dim."""
    ph = input_h // patch_size
    pw = input_w // patch_size
    assert tokens.shape[0] == ph * pw, (
        f"Expected {ph * pw} tokens for {input_h}x{input_w} "
        f"(patch_size={patch_size}), got {tokens.shape[0]}"
    )
    feat = tokens.reshape(ph, pw, -1).permute(2, 0, 1)  # (D, H_p, W_p)
    return F.normalize(feat, dim=0)
```

Also add the import at the top of `utils.py` if not already present:
```python
import torch.nn.functional as F
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/utils.py tests/semantics/test_extractor_preprocessing.py
git commit -m "feat(semantics): add _tokens_to_feature_map shared utility"
```

---

### Task 2: Refactor DINOv2 extractor

**Files:**
- Modify: `collab_splats/semantics/features/dino.py`
- Test: `tests/semantics/test_extractor_preprocessing.py`

Key changes:
- Normalization: `mean=[0.5], std=[0.5]` → ImageNet `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- Rename `resolution` → `image_resolution`
- Add `resize_mode: str = "max_size"` constructor param
- `preprocess(image)` returns `(C, H, W)` CPU tensor (no more tuple)
- `forward()` stacks tensors → single batched backbone call → `_tokens_to_feature_map`

- [ ] **Step 1: Add test for DINOv2 preprocess shape (no model load)**

Append to `tests/semantics/test_extractor_preprocessing.py`:

```python
from unittest.mock import MagicMock, patch


def _make_fake_dino(resize_mode="max_size", image_resolution=224, patch_size=14):
    """Build a DINOFeatureExtractor with a mocked backbone (no weights download)."""
    from collab_splats.semantics.features.dino import DINOFeatureExtractor

    mock_model = MagicMock()
    mock_model.config.patch_size = patch_size
    # last_hidden_state: (B, 1 + H_p*W_p, D) — CLS + patch tokens
    def fake_forward(batch):
        B, _, H, W = batch.shape
        ph, pw = H // patch_size, W // patch_size
        n_tokens = 1 + ph * pw
        result = MagicMock()
        result.last_hidden_state = torch.randn(B, n_tokens, 8)
        return result
    mock_model.side_effect = fake_forward

    with patch("collab_splats.semantics.features.dino.AutoModel.from_pretrained", return_value=mock_model):
        ext = DINOFeatureExtractor(
            resize_mode=resize_mode,
            image_resolution=image_resolution,
            device="cpu",
        )
    ext.model = mock_model
    return ext


def test_dino_preprocess_returns_tensor():
    ext = _make_fake_dino()
    img = Image.new("RGB", (320, 240))
    t = ext.preprocess(img)
    assert isinstance(t, torch.Tensor)
    assert t.dim() == 3  # (C, H, W)
    assert t.shape[0] == 3


def test_dino_preprocess_patch_aligned():
    patch_size = 14
    ext = _make_fake_dino(patch_size=patch_size)
    img = Image.new("RGB", (320, 240))
    t = ext.preprocess(img)
    _, H, W = t.shape
    assert H % patch_size == 0
    assert W % patch_size == 0


def test_dino_preprocess_imagenet_normalization():
    """Mean of normalized tensor should be near 0 (ImageNet centering), not -1 (mean=0.5 centering)."""
    ext = _make_fake_dino(image_resolution=224)
    # Solid grey image (value ~0.5 in [0,1]) → after ImageNet norm ≈ 0, after [0.5] norm ≈ 0 too
    # Use solid white (1.0) → ImageNet norm gives ~(1-0.456)/0.224≈2.4; [0.5] norm gives (1-0.5)/0.5=1.0
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    t = ext.preprocess(img)
    # Green channel: (1.0 - 0.456) / 0.224 ≈ 2.43 under ImageNet; 1.0 under [0.5] norm
    assert t[1].mean().item() > 1.5, "Expected ImageNet normalization (green channel > 1.5 for white img)"


def test_dino_forward_returns_list_of_maps():
    ext = _make_fake_dino(image_resolution=196, patch_size=14)
    imgs = [Image.new("RGB", (196, 196)) for _ in range(2)]

    fake_features = MagicMock()
    # last_hidden_state (B, 1+N, D): 1 CLS + 14*14=196 patches
    fake_features.last_hidden_state = torch.randn(2, 197, 8)
    ext.model.side_effect = None
    ext.model.return_value = fake_features

    out = ext.forward(imgs)
    assert len(out) == 2
    assert out[0].shape == (8, 14, 14)  # (D, H_p, W_p)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py::test_dino_preprocess_returns_tensor -v
```

Expected: FAIL — `DINOFeatureExtractor.__init__` has no `resize_mode` param.

- [ ] **Step 3: Rewrite dino.py**

Replace `collab_splats/semantics/features/dino.py` entirely:

```python
"""DINOv2 feature extractor backend."""
import logging
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device, _tokens_to_feature_map
from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)

# ImageNet normalization — matches DINOv2 training preprocessing
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


@BaseFeatureExtractor.register("dinov2")
class DINOFeatureExtractor(BaseFeatureExtractor):
    """Patch-level DINOv2 feature extractor via HuggingFace transformers.

    Args:
        model_name: HuggingFace model ID. Defaults to ``"facebook/dinov2-small"``.
        resize_mode: ``"max_size"`` (proportional longest-edge) or ``"square"`` (center-crop + resize).
        image_resolution: Longest-edge target (max_size) or square side length (square). Default 800.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        resize_mode: str = "max_size",
        image_resolution: int = 800,
        device: Optional[str] = None,
        **kwargs,
    ):
        if device is None:
            device = get_device()
        super().__init__(**kwargs)
        self.model_name = model_name
        self._resize_mode = resize_mode
        self._image_resolution = image_resolution

        # Load DINOv2 from HuggingFace and move to device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size: int = self.model.config.patch_size

        # ImageNet normalization — correct stats for DINOv2
        self._normalize = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        """Device of the underlying model parameters."""
        return self._device

    def preprocess(self, image) -> torch.Tensor:
        """Resize, normalize, and pad image to patch-aligned dims.

        Returns:
            ``(C, H, W)`` float32 tensor on CPU. H and W are multiples of ``patch_size``.
        """
        img = open_image(image).convert("RGB")

        if self._resize_mode == "square":
            # Center-crop to square, then resize to target resolution
            w, h = img.size
            crop = min(w, h)
            img = img.crop(((w - crop) // 2, (h - crop) // 2,
                             (w + crop) // 2, (h + crop) // 2))
            img = img.resize((self._image_resolution, self._image_resolution), Image.BILINEAR)
        else:
            # Proportional longest-edge resize
            img = resize_image(img, longest_edge=self._image_resolution)

        # Round H and W to nearest patch_size multiple
        w, h = img.size
        ph = round(h / self.patch_size) * self.patch_size
        pw = round(w / self.patch_size) * self.patch_size
        if (ph, pw) != (h, w):
            img = img.resize((pw, ph), Image.BILINEAR)

        return self._normalize(T.ToTensor()(img))

    def forward(self, images: list) -> list[torch.Tensor]:
        """Extract patch-level DINOv2 features from a list of images."""
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess all images and stack into a single batch
        preprocessed = [self.preprocess(img) for img in images]
        batch = torch.stack(preprocessed).to(self._device)

        # Run DINOv2; drop CLS token (index 0), keep patch tokens → (B, N, D)
        with torch.no_grad():
            tokens_all = self.model(batch).last_hidden_state[:, 1:]

        # Reshape each image's flat patch sequence to (D, H_p, W_p)
        results = []
        for i, t in enumerate(preprocessed):
            _, H, W = t.shape
            results.append(
                _tokens_to_feature_map(tokens_all[i].cpu(), H, W, self.patch_size)
            )
        return results
```

- [ ] **Step 4: Run DINOv2 tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py -k "dino" -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run full semantics suite to catch regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ -v
```

Expected: all pass (or pre-existing failures only).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/semantics/features/dino.py tests/semantics/test_extractor_preprocessing.py
git commit -m "refactor(semantics): unify DINOv2 preprocess interface; fix ImageNet normalization"
```

---

### Task 3: Refactor MaskCLIP extractor

**Files:**
- Modify: `collab_splats/semantics/features/maskclip.py`
- Test: `tests/semantics/test_extractor_preprocessing.py`

Key changes:
- Normalization: ImageNet → CLIP stats `(0.48145466, 0.4578275, 0.40821073)` / `(0.26862954, 0.26130258, 0.27577711)`
- Add `resize_mode: str = "max_size"` constructor param
- Rename `resolution` → `image_resolution`
- Remove per-call `resolution` param from `preprocess` and `forward`
- `forward` uses `_tokens_to_feature_map`, drops `list(features)` anti-pattern

- [ ] **Step 1: Add tests for MaskCLIP preprocessing**

Append to `tests/semantics/test_extractor_preprocessing.py`:

```python
def _make_fake_maskclip(resize_mode="max_size", image_resolution=336, patch_size=14):
    """Build MaskCLIPExtractor with mocked model (no weights download)."""
    from collab_splats.semantics.features.maskclip import MaskCLIPExtractor

    mock_model = MagicMock()
    mock_model.visual.patch_size = patch_size

    def fake_patch_encodings(batch):
        B, _, H, W = batch.shape
        ph, pw = H // patch_size, W // patch_size
        return torch.randn(B, ph * pw, 8)

    mock_model.get_patch_encodings.side_effect = fake_patch_encodings

    mock_maskclip_onnx = MagicMock()
    mock_maskclip_onnx.clip.load.return_value = (mock_model, MagicMock())

    with patch.dict("sys.modules", {"maskclip_onnx": mock_maskclip_onnx}):
        with patch("collab_splats.semantics.features.maskclip.maskclip_onnx", mock_maskclip_onnx):
            ext = MaskCLIPExtractor(
                resize_mode=resize_mode,
                image_resolution=image_resolution,
                device="cpu",
            )
    ext.model = mock_model
    return ext


def test_maskclip_preprocess_clip_normalization():
    """White image green channel should be ~(1.0-0.4578275)/0.26130258 ≈ 2.07 under CLIP stats."""
    ext = _make_fake_maskclip(image_resolution=336)
    img = Image.new("RGB", (336, 336), color=(255, 255, 255))
    t = ext.preprocess(img)
    # CLIP green: (1.0 - 0.4578275) / 0.26130258 ≈ 2.075
    # ImageNet green: (1.0 - 0.456) / 0.224 ≈ 2.429
    assert 1.9 < t[1].mean().item() < 2.2, (
        f"Expected CLIP normalization (~2.07), got {t[1].mean().item():.3f}"
    )


def test_maskclip_preprocess_patch_aligned():
    ext = _make_fake_maskclip(patch_size=14)
    img = Image.new("RGB", (400, 300))
    t = ext.preprocess(img)
    _, H, W = t.shape
    assert H % 14 == 0 and W % 14 == 0


def test_maskclip_forward_returns_list_of_maps():
    ext = _make_fake_maskclip(image_resolution=196, patch_size=14)
    imgs = [Image.new("RGB", (196, 196)) for _ in range(3)]
    out = ext.forward(imgs)
    assert len(out) == 3
    assert out[0].shape == (8, 14, 14)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py -k "maskclip" -v
```

Expected: FAIL — `MaskCLIPExtractor.__init__` has no `resize_mode` param.

- [ ] **Step 3: Rewrite maskclip.py**

Replace `collab_splats/semantics/features/maskclip.py` entirely:

```python
"""MaskCLIP feature extractor backend."""
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device, _tokens_to_feature_map
from .base import BaseQueryableExtractor, TORCH_HOME

logger = logging.getLogger(__name__)

# CLIP normalization — from maskclip_onnx/clip.py _transform()
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


@BaseQueryableExtractor.register("maskclip")
class MaskCLIPExtractor(BaseQueryableExtractor):
    """Patch-level MaskCLIP feature extractor.

    Args:
        model_name: CLIP model variant. Defaults to ``"ViT-L/14@336px"``.
        resize_mode: ``"max_size"`` (proportional longest-edge) or ``"square"`` (center-crop + resize).
        image_resolution: Longest-edge target (max_size) or square side length (square). Default 1024.
        cache_dir: Directory to cache model weights. Defaults to TORCH_HOME.
        device: Torch device string. Defaults to auto-detected device.
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14@336px",
        resize_mode: str = "max_size",
        image_resolution: int = 1024,
        cache_dir: str = TORCH_HOME,
        device: Optional[str] = None,
        **kwargs,
    ):
        if device is None:
            device = get_device()
        super().__init__(**kwargs)
        self._resize_mode = resize_mode
        self._image_resolution = image_resolution

        # Lazy import: maskclip_onnx depends on pkg_resources.packaging which was removed
        # in setuptools>=71. Import here so the module is importable even if maskclip_onnx
        # has broken transitive deps — failures only surface when MaskCLIPExtractor is used.
        import maskclip_onnx  # noqa: PLC0415

        # Load the MaskCLIP model; discard the library's default preprocess (square crop)
        # since we apply our own transform with correct CLIP normalization stats
        self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
        self._maskclip_onnx = maskclip_onnx

        self.model = self.model.to(device).eval()
        self.patch_size: int = self.model.visual.patch_size
        self._device = torch.device(device)

        # CLIP normalization — matches maskclip_onnx/clip.py _transform() stats
        self._normalize = T.Normalize(_CLIP_MEAN, _CLIP_STD)

    @property
    def device(self) -> torch.device:
        """Device of the underlying model parameters."""
        return self._device

    def preprocess(self, image) -> torch.Tensor:
        """Resize and normalize image to patch-aligned dims.

        Returns:
            ``(C, H, W)`` float32 tensor on CPU. H and W are multiples of ``patch_size``.
        """
        img = open_image(image).convert("RGB")

        if self._resize_mode == "square":
            # Center-crop to square, then resize to target resolution
            w, h = img.size
            crop = min(w, h)
            img = img.crop(((w - crop) // 2, (h - crop) // 2,
                             (w + crop) // 2, (h + crop) // 2))
            img = img.resize((self._image_resolution, self._image_resolution), Image.BILINEAR)
        else:
            # Proportional longest-edge resize
            img = resize_image(img, longest_edge=self._image_resolution)

        # Round H and W to nearest patch_size multiple
        w, h = img.size
        ph = round(h / self.patch_size) * self.patch_size
        pw = round(w / self.patch_size) * self.patch_size
        if (ph, pw) != (h, w):
            img = img.resize((pw, ph), Image.BILINEAR)

        return self._normalize(T.ToTensor()(img))

    def forward(self, images: list) -> list[torch.Tensor]:
        """Extract patch-level CLIP features from a list of images."""
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess all images and stack into a single batch
        preprocessed = [self.preprocess(img) for img in images]
        batch = torch.stack(preprocessed).to(self._device)

        # get_patch_encodings returns (B, N_patches, D) — no CLS token to skip
        with torch.no_grad():
            tokens_all = F.normalize(
                self.model.get_patch_encodings(batch).to(torch.float32), dim=-1
            )

        # Reshape each image's flat patch sequence to (D, H_p, W_p)
        results = []
        for i, t in enumerate(preprocessed):
            _, H, W = t.shape
            results.append(
                _tokens_to_feature_map(tokens_all[i].cpu(), H, W, self.patch_size)
            )
        return results

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Compute normalized CLIP embeddings for a list of text queries."""
        tokens = self._maskclip_onnx.clip.tokenize(text).to(self._device)
        embed = self.model.encode_text(tokens).float()
        embed /= embed.norm(dim=-1, keepdim=True)
        return embed
```

- [ ] **Step 4: Run MaskCLIP tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py -k "maskclip" -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Run full semantics suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/semantics/features/maskclip.py tests/semantics/test_extractor_preprocessing.py
git commit -m "refactor(semantics): unify MaskCLIP preprocess interface; fix CLIP normalization stats"
```

---

### Task 4: Refactor Talk2DINO extractor

**Files:**
- Modify: `collab_splats/semantics/features/talk2dino.py`
- Test: `tests/semantics/test_extractor_preprocessing.py`

Key changes:
- Remove `_forward_max_size()` and `_forward_square()` entirely
- `preprocess(image) -> Tensor` (CPU) handles both modes — no more PIL return
- `forward()` stacks tensors → `self._model.model.forward_features(batch)[:, 5:]` → `_tokens_to_feature_map`
- Keep normalization from `self._model.image_transforms.transforms[-1]`

- [ ] **Step 1: Add tests for Talk2DINO preprocessing**

Append to `tests/semantics/test_extractor_preprocessing.py`:

```python
def _make_fake_talk2dino(resize_mode="max_size", image_resolution=512, patch_size=14):
    """Build Talk2DinoExtractor with mocked backbone (no weights download)."""
    from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

    mock_backbone = MagicMock()

    def fake_forward_features(batch):
        B, _, H, W = batch.shape
        ph, pw = H // patch_size, W // patch_size
        # Returns (B, 5 + H_p*W_p, D): 5 register/CLS tokens + patch tokens
        return torch.randn(B, 5 + ph * pw, 8)

    mock_backbone.forward_features.side_effect = fake_forward_features

    mock_model = MagicMock()
    mock_model.model = mock_backbone
    mock_model.patch_embed.proj.stride = [patch_size, patch_size]

    # Provide a real Normalize as image_transforms.transforms[-1]
    mock_model.image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    with patch("collab_splats.semantics.features.talk2dino.AutoModel.from_pretrained", return_value=mock_model):
        ext = Talk2DinoExtractor(
            resize_mode=resize_mode,
            image_resolution=image_resolution,
            device="cpu",
        )
    ext._model = mock_model
    return ext


def test_talk2dino_preprocess_returns_tensor():
    ext = _make_fake_talk2dino()
    img = Image.new("RGB", (320, 240))
    t = ext.preprocess(img)
    assert isinstance(t, torch.Tensor)
    assert t.dim() == 3


def test_talk2dino_preprocess_patch_aligned():
    ext = _make_fake_talk2dino(patch_size=14)
    img = Image.new("RGB", (400, 300))
    t = ext.preprocess(img)
    _, H, W = t.shape
    assert H % 14 == 0 and W % 14 == 0


def test_talk2dino_forward_single_path():
    """forward() must work for both resize modes without _forward_max_size/_forward_square."""
    for mode in ("max_size", "square"):
        ext = _make_fake_talk2dino(resize_mode=mode, image_resolution=196, patch_size=14)
        imgs = [Image.new("RGB", (196, 196)) for _ in range(2)]
        out = ext.forward(imgs)
        assert len(out) == 2
        assert out[0].shape[1] == out[0].shape[2]  # square feature map for square input


def test_talk2dino_no_dual_forward_methods():
    """_forward_max_size and _forward_square must not exist after refactor."""
    from collab_splats.semantics.features import talk2dino as t2d_mod
    src = open(t2d_mod.__file__).read()
    assert "_forward_max_size" not in src
    assert "_forward_square" not in src
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py -k "talk2dino" -v
```

Expected: `test_talk2dino_no_dual_forward_methods` FAILS (methods still exist).

- [ ] **Step 3: Rewrite talk2dino.py**

Replace `collab_splats/semantics/features/talk2dino.py` entirely:

```python
"""Talk2DINO feature extractor backend."""
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device, _tokens_to_feature_map
from .base import BaseQueryableExtractor

logger = logging.getLogger(__name__)


@BaseQueryableExtractor.register("talk2dino")
class Talk2DinoExtractor(BaseQueryableExtractor):
    """
    Wraps Talk2DINO models from HuggingFace Hub for patch-level feature extraction
    and text-conditioned semantic similarity.

    Supports DINOv3 (default) and DINOv2 variants:
      - "lorebianchi98/Talk2DINOv3-ViTB"  (default)
      - "lorebianchi98/Talk2DINO-ViTB"    (DINOv2)

    resize_mode controls image preprocessing:
      - "max_size" (default): proportional longest-edge resize, all pixels retained.
        Correct spatial correspondence for lift_features — no crop, no remapping.
      - "square": center-crop to square, then resize to image_resolution.

    In both modes, preprocessing is fully handled by preprocess() before forward().
    forward_features is called directly (bypassing encode_image's internal resize).

    Algorithm from Talk2DINO (https://github.com/lorebianchi98/Talk2DINO).
    """

    def __init__(
        self,
        model_name: str = "lorebianchi98/Talk2DINOv3-ViTB",
        device: Optional[str] = None,
        resize_mode: str = "max_size",
        image_resolution: int = 512,
        **kwargs,
    ):
        """
        Args:
            model_name: HuggingFace Hub model ID.
            device: Torch device string ("cpu" or "cuda").
            resize_mode: "max_size" (proportional, longest-edge) or "square" (center-crop + resize).
            image_resolution: Longest-edge target for "max_size"; square side for "square".
        """
        if device is None:
            device = get_device()
        super().__init__(**kwargs)

        self._resize_mode = resize_mode
        self._image_resolution = image_resolution

        # Load Talk2DINO model from HuggingFace Hub and move to device
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

        # Extract Normalize transform from model's stored image_transforms —
        # correct mean/std regardless of backbone variant
        self._normalize: T.Normalize = self._model.image_transforms.transforms[-1]

        # Derive patch_size from backbone conv layer (more reliable than config)
        try:
            conv = self._model.model.patch_embed.proj
            self.patch_size: int = conv.stride[0]
        except AttributeError:
            self.patch_size = getattr(self._model.config, "patch_size", 14)

        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        """Device of the underlying model."""
        return self._device

    def preprocess(self, image) -> torch.Tensor:
        """Resize, round to patch multiples, and normalize image.

        Both resize_mode branches produce a ``(C, H, W)`` CPU tensor ready for
        ``torch.stack`` and ``forward_features``. No PIL images passed to the backbone.

        Returns:
            ``(C, H, W)`` float32 tensor on CPU. H and W are multiples of ``patch_size``.
        """
        img = open_image(image).convert("RGB")

        if self._resize_mode == "square":
            # Center-crop to square, then resize to target resolution
            w, h = img.size
            crop = min(w, h)
            img = img.crop(((w - crop) // 2, (h - crop) // 2,
                             (w + crop) // 2, (h + crop) // 2))
            img = img.resize((self._image_resolution, self._image_resolution), Image.BILINEAR)
        else:
            # Proportional longest-edge resize — preserves aspect ratio and pixel correspondence
            img = resize_image(img, longest_edge=self._image_resolution)

        # Round H and W to nearest patch_size multiple so ViT pos-embeds align cleanly
        w, h = img.size
        ph = round(h / self.patch_size) * self.patch_size
        pw = round(w / self.patch_size) * self.patch_size
        if (ph, pw) != (h, w):
            img = img.resize((pw, ph), Image.BILINEAR)

        return self._normalize(T.ToTensor()(img))

    def forward(self, images: list) -> list[torch.Tensor]:
        """Extract patch-level Talk2DINO features from a list of images."""
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess all images and stack into a single batch
        preprocessed = [self.preprocess(img) for img in images]
        batch = torch.stack(preprocessed).to(self._device)

        # Call forward_features directly — bypasses encode_image's internal T.Resize((N,N)).
        # [:, 5:] drops CLS + 4 register tokens (DINOv3 convention) → (B, N_patches, D)
        with torch.no_grad():
            tokens_all = self._model.model.forward_features(batch)[:, 5:]

        # Reshape each image's flat patch sequence to (D, H_p, W_p)
        results = []
        for i, t in enumerate(preprocessed):
            _, H, W = t.shape
            results.append(
                _tokens_to_feature_map(tokens_all[i].cpu(), H, W, self.patch_size)
            )
        return results

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text queries to normalized embeddings.

        Returns:
            ``(N, D)`` normalized embeddings.
        """
        with torch.no_grad():
            embeddings = self._model.encode_text(texts)
        return F.normalize(embeddings, dim=-1)
```

- [ ] **Step 4: Run Talk2DINO tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py -k "talk2dino" -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run full semantics suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/semantics/features/talk2dino.py tests/semantics/test_extractor_preprocessing.py
git commit -m "refactor(semantics): unify Talk2DINO preprocess interface; single forward path"
```

---

### Task 5: Final integration check

- [ ] **Step 1: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration -x
```

Expected: all pass (or only pre-existing failures).

- [ ] **Step 2: Verify no old params leak**

```bash
grep -rn "resize_mode\|image_resolution\|_forward_max_size\|_forward_square" \
  collab_splats/semantics/features/
```

Expected: `resize_mode` and `image_resolution` appear in all three files as constructor params. `_forward_max_size` and `_forward_square` appear nowhere.

- [ ] **Step 3: Verify normalization constants**

```bash
grep -n "Normalize\|MEAN\|STD\|0.485\|0.48145" collab_splats/semantics/features/dino.py \
  collab_splats/semantics/features/maskclip.py collab_splats/semantics/features/talk2dino.py
```

Expected:
- `dino.py`: `0.485, 0.456, 0.406` (ImageNet)
- `maskclip.py`: `0.48145466` (CLIP)
- `talk2dino.py`: no hardcoded stats (reads from model)

- [ ] **Step 4: Commit final state if any fixups needed**

```bash
git add -p
git commit -m "fix(semantics): extractor preprocessing integration fixups"
```
