"""
DINOv2 patch-feature backend ("dinov2"), via HuggingFace transformers.
"""
from typing import Optional

import torch
import torchvision.transforms as T
from transformers import AutoModel

from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD
from collab_splats.utils.torch_utils import get_device

from .base import BaseFeatureExtractor


########################################################################
########## Extractor ###################################################
########################################################################

@BaseFeatureExtractor.register("dinov2")
class DINOFeatureExtractor(BaseFeatureExtractor):
    """
    DINOv2 patch features via HuggingFace transformers.

    Args:
        model_name: HuggingFace model id.
        resize_mode: "max_size" (longest edge) or "square" (center-crop, then resize).
        image_resolution: longest-edge target for "max_size", side for "square".
        device: torch device; None picks one with `get_device`.
        svd_components: positional-subspace rank for debias().
    """

    debias_validated = True
    n_prefix_tokens = 1  # CLS

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

        # Load DINOv2 from HuggingFace and move to device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        # ImageNet normalization — correct stats for DINOv2
        self._normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._device = torch.device(device)

    @property
    def patch_size(self) -> int:
        """
        Patch size read from the model config at call time.

        Returns:
            Side length in pixels of one DINOv2 patch.
        """
        return self.model.config.patch_size

    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        DINOv2 last hidden state: CLS token, then patch tokens.

        Args:
            batch: (B, C, H, W) preprocessed images.

        Returns:
            (B, 1 + H_p * W_p, D) tokens.
        """
        return self.model(batch).last_hidden_state
