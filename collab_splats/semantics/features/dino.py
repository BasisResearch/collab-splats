"""DINOv2 feature extractor backend."""
import logging
from typing import Optional

import torch
import torchvision.transforms as T
from transformers import AutoModel

from collab_splats.semantics.utils import tokens_to_feature_map
from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD
from collab_splats.utils.torch_utils import get_device

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


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
        svd_components: Top singular vectors kept for positional debiasing. Default 500.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        resize_mode: str = "max_size",
        image_resolution: int = 800,
        device: Optional[str] = None,
        svd_components: int = 500,
    ):
        if device is None:
            device = get_device()
        super().__init__(resize_mode, image_resolution, svd_components)
        self.model_name = model_name

        # Load DINOv2 from HuggingFace and move to device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        # ImageNet normalization — correct stats for DINOv2
        self._normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        """
        Device of the underlying model parameters.

        Returns:
            The torch device the model was moved to at construction.
        """
        return self._device

    @property
    def patch_size(self) -> int:
        """
        Patch size read from the model config at call time.

        Returns:
            Side length in pixels of one DINOv2 patch.
        """
        return self.model.config.patch_size

    def forward(self, images: list) -> list[torch.Tensor]:
        """
        Extract patch-level DINOv2 features from a list of images.

        Args:
            images: anything `preprocess` accepts — paths, ndarrays or PIL images.

        Returns:
            One (D, H_p, W_p) float32 CPU tensor per input image. Per-image shape, because
            `preprocess` preserves aspect ratio and the grids differ.
        """
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
            results.append(tokens_to_feature_map(tokens_all[i].cpu(), H, W, self.patch_size))
        return results
