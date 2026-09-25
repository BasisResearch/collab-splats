"""
Talk2DINO patch-feature backend ("talk2dino"), with a text tower for queries.
"""

import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModel

from collab_splats.utils.torch_utils import get_device

from .base import BaseQueryableExtractor


@BaseQueryableExtractor.register("talk2dino")
class Talk2DinoExtractor(BaseQueryableExtractor):
    """
    Talk2DINO patch features and text embeddings, from the HuggingFace Hub.

    - backbones: Talk2DINOv3-ViTB (DINOv3, default) or Talk2DINO-ViTB (DINOv2 with registers)
    - calls forward_features directly: preprocess() already made patch-aligned tensors
    - source: https://github.com/lorebianchi98/Talk2DINO

    Args:
        model_name: HuggingFace Hub model id.
        device: torch device; None picks one with `get_device`.
        resize_mode: "max_size" (longest edge) or "square" (center-crop, then resize).
        image_resolution: longest-edge target for "max_size", side for "square".
        svd_components: positional-subspace rank for debias().
    """

    debias_validated = True
    n_prefix_tokens = 5  # CLS + 4 registers

    def __init__(
        self,
        model_name: str = "lorebianchi98/Talk2DINOv3-ViTB",
        device: Optional[str] = None,
        resize_mode: str = "max_size",
        image_resolution: int = 512,
        svd_components: int = 500,
    ):
        if device is None:
            device = get_device()
        super().__init__(resize_mode=resize_mode, image_resolution=image_resolution, svd_components=svd_components)

        # low_cpu_mem_usage=False: Talk2DINO's load_state_dict would no-op on meta tensors
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
            _loaded = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=False)

        # Normalize transform from the model's own image_transforms — correct stats per backbone
        self._normalize: T.Normalize = _loaded.image_transforms.transforms[-1]

        # patch_size = backbone conv stride (sets the token grid; kernel_size does not)
        self.patch_size: int = _loaded.model.patch_embed.proj.stride[0]

        # Move to device after extracting metadata
        self._model = _loaded.to(device).eval()
        self._device = torch.device(device)

    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Backbone forward_features, bypassing encode_image's internal square resize.

        Args:
            batch: (B, C, H, W) preprocessed images.

        Returns:
            (B, prefix + H_p * W_p, D) tokens; prefix is CLS plus registers.
        """
        return self._model.model.forward_features(batch)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text queries to normalized embeddings.

        Args:
            texts: query strings, encoded by the Talk2DINO text tower.

        Returns:
            (N, D) normalized embeddings.
        """
        with torch.no_grad():
            embeddings = self._model.encode_text(texts)
        return F.normalize(embeddings, dim=-1)
