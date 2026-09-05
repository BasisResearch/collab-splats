"""MaskCLIP feature extractor backend."""
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from collab_splats.semantics.utils import _tokens_to_feature_map, get_device
from collab_splats.utils.image import CLIP_MEAN, CLIP_STD

from .base import TORCH_HOME, BaseQueryableExtractor

logger = logging.getLogger(__name__)


@BaseQueryableExtractor.register("maskclip")
class MaskCLIPExtractor(BaseQueryableExtractor):
    """Patch-level MaskCLIP feature extractor.

    Args:
        model_name: CLIP model variant. Defaults to ``"ViT-L/14@336px"``.
        resize_mode: ``"max_size"`` (proportional longest-edge) or ``"square"`` (center-crop + resize).
        image_resolution: Longest-edge target (max_size) or square side length (square). Default 1024.
        cache_dir: Directory to cache model weights. Defaults to TORCH_HOME.
        device: Torch device string. Defaults to auto-detected device.
        svd_components: Top singular vectors kept for positional debiasing. Default 500.
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14@336px",
        resize_mode: str = "max_size",
        image_resolution: int = 1024,
        cache_dir: str = TORCH_HOME,
        device: Optional[str] = None,
        svd_components: int = 500,
    ):
        if device is None:
            device = get_device()
        super().__init__(resize_mode, image_resolution, svd_components)

        # Lazy import: maskclip_onnx depends on pkg_resources.packaging which was removed
        # in setuptools>=71. Import here so the module is importable even if maskclip_onnx
        # has broken transitive deps — failures only surface when MaskCLIPExtractor is used.
        import maskclip_onnx  # noqa: PLC0415

        # Load the MaskCLIP model; discard the library's default preprocess (square crop)
        # since we apply our own transform with correct CLIP normalization stats
        self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
        self._maskclip_onnx = maskclip_onnx

        # Read patch_size before chaining .to().eval() (chained calls return new objects on mocks)
        self.patch_size: int = self.model.visual.patch_size
        self.model = self.model.to(device).eval()
        self._device = torch.device(device)

        # CLIP normalization — matches maskclip_onnx/clip.py _transform() stats
        self._normalize = T.Normalize(CLIP_MEAN, CLIP_STD)

    ########################################################################
    # Properties
    ########################################################################

    @property
    def device(self) -> torch.device:
        """Device of the underlying model parameters."""
        return self._device

    ########################################################################
    # Forward pass
    ########################################################################

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

    ########################################################################
    # Text encoding
    ########################################################################

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Compute normalized CLIP embeddings for a list of text queries."""
        tokens = self._maskclip_onnx.clip.tokenize(text).to(self._device)
        embed = self.model.encode_text(tokens).float()
        embed /= embed.norm(dim=-1, keepdim=True)
        return embed
