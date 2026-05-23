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

########################################################################
# CLIP normalization constants — from maskclip_onnx/clip.py _transform()
########################################################################

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

        # Read patch_size before chaining .to().eval() (chained calls return new objects on mocks)
        self.patch_size: int = self.model.visual.patch_size
        self.model = self.model.to(device).eval()
        self._device = torch.device(device)

        # CLIP normalization — matches maskclip_onnx/clip.py _transform() stats
        self._normalize = T.Normalize(_CLIP_MEAN, _CLIP_STD)

    ########################################################################
    # Properties
    ########################################################################

    @property
    def device(self) -> torch.device:
        """Device of the underlying model parameters."""
        return self._device

    ########################################################################
    # Preprocessing
    ########################################################################

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
