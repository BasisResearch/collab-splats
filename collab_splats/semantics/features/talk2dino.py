"""Talk2DINO feature extractor backend."""
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModel

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device, interpolate_to_patch_size
from .base import BaseQueryableExtractor

logger = logging.getLogger(__name__)


@BaseQueryableExtractor.register("talk2dino")
class Talk2DinoExtractor(BaseQueryableExtractor):
    """
    Wraps Talk2DINO models from HuggingFace Hub for patch-level feature extraction
    and text-conditioned semantic similarity.

    Supports DINOv3 (default) and DINOv2 variants:
      - "lorebianchi98/Talk2DINOv3-ViTB"  (default, cleaner interface)
      - "lorebianchi98/Talk2DINO-ViTB"    (DINOv2, older interface)

    Algorithm from Talk2DINO (https://github.com/lorebianchi98/Talk2DINO).
    """

    def __init__(
        self,
        model_name: str = "lorebianchi98/Talk2DINOv3-ViTB",
        device: Optional[str] = None,
        **kwargs,  # passes svd_components and any future BaseFeatureExtractor params through
    ):
        """
        Args:
            model_name: HuggingFace Hub model ID.
                DINOv3 (default): "lorebianchi98/Talk2DINOv3-ViTB"
                DINOv2: "lorebianchi98/Talk2DINO-ViTB"
            device: Torch device string ("cpu" or "cuda").

        Note: No resolution param. Talk2DINO requires center-crop to square — this
            is a model architecture constraint, not a configurable resolution.
        """
        if device is None:
            device = get_device()
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__

        # Load Talk2DINO model from HuggingFace Hub and move to device
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

        # config.patch_size reflects the ViT token grid, not the pixel stride seen
        # by the caller — encode_image upscales internally (e.g. 224→448) so the
        # effective stride in original-input pixels is smaller. Derive from conv layer.
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

    def preprocess(self, image) -> "Image.Image":
        """
        Center-crop image to square. Required by Talk2DINO (from hf_demo.ipynb).

        Returns PIL Image ready for encode_image / forward.
        """
        from PIL import Image

        image = open_image(image).convert("RGB")
        w, h = image.size
        crop_size = min(w, h)
        image = image.crop(
            (
                (w - crop_size) // 2,
                (h - crop_size) // 2,
                (w + crop_size) // 2,
                (h + crop_size) // 2,
            )
        )
        return image

    def forward(self, images: list) -> list[torch.Tensor]:
        """Extract patch-level Talk2DINO features from a list of images."""
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Center-crop each image to square (Talk2DINO model constraint)
        preprocessed = [self.preprocess(img) for img in images]

        # Run encode_image; model returns patch tokens per image, no CLS token
        with torch.no_grad():
            result = self._model.encode_image(preprocessed)
        patch_tokens = list(result) if isinstance(result, torch.Tensor) else result

        # Reshape flat patch sequence to (C, H_p, W_p) and L2-normalize each patch vector
        outputs = []
        for tokens in patch_tokens:
            n_patches = tokens.shape[0]
            ph = pw = int(n_patches ** 0.5)

            assert ph * pw == n_patches, (
                f"Talk2DinoExtractor.forward: expected square patch grid, got {n_patches} patches "
                f"(sqrt={n_patches**0.5:.3f}). Is a CLS token included in model output?"
            )

            feat = tokens.reshape(ph, pw, -1).permute(2, 0, 1)  # (C, H, W)
            outputs.append(F.normalize(feat, dim=0))  # cosine sim requires unit-norm patches

        return outputs

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text queries to normalized embeddings.

        Args:
            texts: list of text strings.

        Returns:
            (N, D) normalized embeddings.
        """
        with torch.no_grad():
            embeddings = self._model.encode_text(texts)
        return F.normalize(embeddings, dim=-1)
