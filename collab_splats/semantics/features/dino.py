"""DINOv2 feature extractor backend."""
import logging
from typing import Optional, Tuple

import torch
import torchvision.transforms as T
from transformers import AutoModel

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device, interpolate_to_patch_size
from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


@BaseFeatureExtractor.register("dinov2")
class DINOFeatureExtractor(BaseFeatureExtractor):
    """Patch-level DINOv2 feature extractor via HuggingFace transformers.

    Args:
        model_name: HuggingFace model ID. Defaults to ``"facebook/dinov2-small"``.
        resolution: Longest-edge resize target before patch extraction.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        resolution: int = 800,
        device: Optional[str] = None,
        **kwargs,  # passes svd_components and any future BaseFeatureExtractor params through
    ):
        if device is None:
            device = get_device()
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__
        self.model_name = model_name

        # Load DINOv2 from HuggingFace and move to device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size: int = self.model.config.patch_size
        self.resolution = resolution

        # Standard ViT image normalization transform
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @property
    def device(self) -> torch.device:
        """Device of the underlying model parameters."""
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def preprocess(self, image) -> Tuple[torch.Tensor, int, int]:
        """Resize, normalize, and pad image to be patch-aligned.

        Returns:
            Tuple of (image_tensor, target_H, target_W) where H and W are
            divisible by ``self.patch_size``.
        """
        image = open_image(image)
        image = resize_image(image, longest_edge=self.resolution)
        image = self.transform(image)[:3].unsqueeze(0)
        image, target_H, target_W = interpolate_to_patch_size(image, self.patch_size)
        image = image.to(self.device)

        return image, target_H, target_W

    def forward(self, images: list) -> list[torch.Tensor]:
        """Extract patch-level DINOv2 features from a list of images."""
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess each image: resize, normalize, pad to patch-aligned dimensions
        preprocessed = [self.preprocess(img) for img in images]

        # Batch all images into a single tensor and move to model device
        tensors = torch.cat([t for t, _, _ in preprocessed], dim=0).to(self.device)

        # Run DINOv2 forward; drop CLS token (index 0), keep patch tokens
        with torch.no_grad():
            features = self.model(tensors).last_hidden_state[:, 1:]

        # Reshape each image's flat patch sequence to (C, H_p, W_p) spatial layout
        results = []
        for i, (_, H, W) in enumerate(preprocessed):
            patch_features = features[i].cpu()
            reshaped = patch_features.reshape(
                H // self.patch_size, W // self.patch_size, -1
            ).permute(2, 0, 1)
            results.append(reshaped)
        return results
