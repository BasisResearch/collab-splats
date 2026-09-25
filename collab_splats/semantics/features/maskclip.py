"""
MaskCLIP patch-feature backend ("maskclip"), with CLIP's text tower for queries.
"""
import os
from typing import List, Optional

import torch
import torchvision.transforms as T

from collab_splats.utils.image import CLIP_MEAN, CLIP_STD
from collab_splats.utils.torch_utils import get_device

from .base import BaseQueryableExtractor


@BaseQueryableExtractor.register("maskclip")
class MaskCLIPExtractor(BaseQueryableExtractor):
    """
    MaskCLIP patch features and CLIP text embeddings.

    Args:
        model_name: CLIP model variant.
        resize_mode: "max_size" (longest edge) or "square" (center-crop, then resize).
        image_resolution: longest-edge target for "max_size", side for "square".
        cache_dir: weights cache; None uses $TORCH_HOME, else ~/.cache/torch.
        device: torch device; None picks one with `get_device`.
        svd_components: positional-subspace rank for debias().

    Raises:
        ImportError: when maskclip_onnx is not installed; the message names the install command.
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14@336px",
        resize_mode: str = "max_size",
        image_resolution: int = 1024,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        svd_components: int = 500,
    ):
        if device is None:
            device = get_device()

        # Read $TORCH_HOME at call time, not at import
        cache_dir = cache_dir or os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

        super().__init__(resize_mode=resize_mode, image_resolution=image_resolution, svd_components=svd_components)

        # Imported here: needs setuptools<70 (pkg_resources.packaging) at import
        try:
            import maskclip_onnx  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "MaskCLIPExtractor needs maskclip_onnx: "
                "pip install 'git+https://github.com/RogerQi/maskclip_onnx.git' 'setuptools<70'"
            ) from e

        # Drop the library's square-crop preprocess; preprocess() does CLIP normalization
        self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
        self._maskclip_onnx = maskclip_onnx

        self.patch_size: int = self.model.visual.patch_size
        self.model = self.model.to(device).eval()
        self._device = torch.device(device)

        # CLIP normalization — matches maskclip_onnx/clip.py _transform() stats
        self._normalize = T.Normalize(CLIP_MEAN, CLIP_STD)

    ########################################################################
    # Backbone
    ########################################################################

    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Raw MaskCLIP patch encodings as float32; the base L2-normalizes them.

        Args:
            batch: (B, C, H, W) preprocessed images.

        Returns:
            (B, H_p * W_p, D) tokens; no CLS.
        """
        return self.model.get_patch_encodings(batch).to(torch.float32)

    ########################################################################
    # Text encoding
    ########################################################################

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Compute normalized CLIP embeddings for a list of text queries.

        Args:
            text: query strings, tokenized by CLIP's own tokenizer.

        Returns:
            (N, D) float32 embeddings on this extractor's device, unit-norm per row.
        """
        tokens = self._maskclip_onnx.clip.tokenize(text).to(self._device)
        embed = self.model.encode_text(tokens).float()
        embed /= embed.norm(dim=-1, keepdim=True)
        return embed
