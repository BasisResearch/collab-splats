"""Stage 1 — Global retrieval: compact global image descriptors for top-K frame retrieval."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T
from salad.models_salad.aggregators.salad import SALAD
from salad.models_salad.backbones.dinov2 import DINOv2

from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD
from collab_splats.utils.torch_utils import RegistryMixin

logger = logging.getLogger(__name__)


class BaseRetrievalExtractor(RegistryMixin, nn.Module, ABC):
    """Abstract base for global image descriptor extractors with name-based registry.

    Returns (N, D) normalized descriptors — one compact vector per image.
    Used for top-K candidate retrieval before local feature matching.
    """

    _registry: dict[str, type["BaseRetrievalExtractor"]] = {}

    @abstractmethod
    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """Return (N, D) normalized global descriptors for N images."""


@BaseRetrievalExtractor.register("dino-salad")
class DinoSaladExtractor(BaseRetrievalExtractor):
    """DINO-SALAD global image descriptor for visual place recognition.

    DINOv2 ViT-B/14 backbone + SALAD aggregator, pretrained on GSV-Cities.
    Installed via pip (Dominic101/salad). Avoids VPRModel to skip pytorch_lightning
    at runtime — imports SALAD and DINOv2 directly.
    Weights: https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
    """

    _WEIGHTS_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
    _INPUT_SIZE = 224  # divisible by 14 for DINOv2 patch grid

    def __init__(self, device: str | None = None):
        super().__init__()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Attr names match VPRModel so dino_salad.ckpt keys (backbone.* / aggregator.*) load cleanly
        self.backbone = DINOv2(
            model_name="dinov2_vitb14",
            num_trainable_blocks=4,
            norm_layer=True,
            return_token=True,
        ).to(self._device)

        self.aggregator = SALAD(
            num_channels=768,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
        ).to(self._device)

        # Load pretrained weights; strict=False tolerates minor key mismatches
        sd = torch.hub.load_state_dict_from_url(
            self._WEIGHTS_URL, map_location=torch.device("cpu")
        )
        self.load_state_dict(sd, strict=False)
        self.eval()

        # Build input transform once — resize + normalize to ImageNet stats
        self._transform = T.Compose([
            T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """Return (N, D) normalized float32 descriptors on CPU."""
        # Preprocess: PIL list → stacked tensor, or resize if already a tensor
        if isinstance(images, (list, tuple)):
            imgs = torch.stack([self._transform(img) for img in images])
        else:
            imgs = images.float()
            if imgs.shape[-2] != self._INPUT_SIZE or imgs.shape[-1] != self._INPUT_SIZE:
                imgs = torch.nn.functional.interpolate(
                    imgs, size=(self._INPUT_SIZE, self._INPUT_SIZE),
                    mode="bilinear", align_corners=False,
                )

        logger.debug("DinoSaladExtractor: embedding batch of %d images", len(imgs))

        # Run backbone → aggregator and L2-normalize output descriptors
        imgs = imgs.to(self._device)
        with torch.no_grad():
            feats = self.backbone(imgs)
            descriptors = self.aggregator(feats)
        return torch.nn.functional.normalize(descriptors, p=2, dim=-1).cpu()


########################################################
########## PE-CLIP retrieval extractor #################
########################################################


@BaseRetrievalExtractor.register("pe-clip")
class PECLIPExtractor(BaseRetrievalExtractor):
    """PE-Core-L/14-336 global image+text encoder for open-label frame retrieval.

    Uses open_clip with hf-hub:timm/PE-Core-L-14-336. Produces (N, 1024)
    normalized descriptors for both images and text — aligned in the same
    CLIP embedding space. Suitable for text-driven frame retrieval.

    Architecture note: PE-Core uses AttentionPoolLatent (pool='map') with no
    separate linear projection. Patch-level CLIP-aligned features are not
    available; use this extractor for global retrieval only.
    """

    _MODEL_ID = "hf-hub:timm/PE-Core-L-14-336"

    def __init__(self, model_id: str = _MODEL_ID, device: str | None = None):
        super().__init__()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_id)
        self._model.to(self._device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_id)
        logger.debug("PECLIPExtractor: loaded %s on %s", model_id, self._device)

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """Return (N, 1024) normalized image descriptors.

        Args:
            images: List of PIL Images or (N, 3, H, W) float32 tensor.

        Returns:
            (N, 1024) float32, L2-normalized, on CPU.
        """
        # Preprocess PIL images if needed; tensors passed through directly
        if isinstance(images, (list, tuple)):
            imgs = torch.stack([self._preprocess(img) for img in images])
        else:
            imgs = images
        imgs = imgs.to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(imgs, normalize=True)
        return features.cpu().float()

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return (N, 1024) normalized text descriptors.

        Args:
            texts: List of text strings.

        Returns:
            (N, 1024) float32, L2-normalized, on CPU.
        """
        tokens = self._tokenizer(texts, context_length=self._model.context_length)
        tokens = tokens.to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens, normalize=True)
        return features.cpu().float()
