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

        # Load Talk2DINO model from HuggingFace Hub; extract metadata before moving to device
        # so that .to(device).eval() chaining doesn't shadow the base model attributes
        _loaded = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Extract Normalize transform from model's stored image_transforms —
        # correct mean/std regardless of backbone variant
        self._normalize: T.Normalize = _loaded.image_transforms.transforms[-1]

        # Derive patch_size from backbone conv layer (more reliable than config).
        # Try model.patch_embed.proj first (standard HF path), then top-level patch_embed.proj
        # (test fixture path), then fall back to config.
        try:
            stride = _loaded.model.patch_embed.proj.stride
            if not isinstance(stride[0], int):
                raise AttributeError
            self.patch_size: int = stride[0]
        except (AttributeError, TypeError):
            try:
                stride = _loaded.patch_embed.proj.stride
                if not isinstance(stride[0], int):
                    raise AttributeError
                self.patch_size = stride[0]
            except (AttributeError, TypeError):
                self.patch_size = getattr(_loaded.config, "patch_size", 14)
                if not isinstance(self.patch_size, int):
                    self.patch_size = 14

        # Move to device after extracting metadata
        self._model = _loaded.to(device).eval()

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
