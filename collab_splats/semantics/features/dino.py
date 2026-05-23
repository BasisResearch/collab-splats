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

########################################################################
########## Constants ###################################################
########################################################################

# ImageNet normalization — matches DINOv2 training preprocessing
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


########################################################################
########## Extractor ###################################################
########################################################################

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

        # ImageNet normalization — correct stats for DINOv2
        self._normalize = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        """Device of the underlying model parameters."""
        return self._device

    @property
    def patch_size(self) -> int:
        """Patch size read from the model config at call time."""
        return self.model.config.patch_size

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
