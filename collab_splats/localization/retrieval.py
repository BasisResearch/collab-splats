"""
Global retrieval: compact image descriptors that rank reference frames for top-K retrieval.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn as nn
import torchvision.transforms as T

from salad.models_salad.aggregators.salad import SALAD
from salad.models_salad.backbones.dinov2 import DINOv2

from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD
from collab_splats.utils.torch_utils import RegistryMixin, get_device

logger = logging.getLogger(__name__)


class BaseRetrievalExtractor(RegistryMixin, nn.Module, ABC):
    """
    Global image descriptor extractor with a name-based registry.

    - one compact (N, D) normalized vector per image
    - used for top-K candidate retrieval before local feature matching
    """

    _registry: ClassVar[dict[str, type["BaseRetrievalExtractor"]]] = {}

    @abstractmethod
    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """
        Normalized global descriptors, one per image.

        Args:
            images: PIL images, or an (N, 3, H, W) RGB tensor.

        Returns:
            (N, D) L2-normalized descriptors.
        """


@BaseRetrievalExtractor.register("dino-salad")
class DinoSaladExtractor(BaseRetrievalExtractor):
    """
    DINO-SALAD global image descriptor for visual place recognition.

    - DINOv2 ViT-B/14 backbone + SALAD aggregator, pretrained on GSV-Cities
    - pip package Dominic101/salad
    - SALAD + DINOv2 imported directly: VPRModel pulls in pytorch_lightning
    - weights: https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
    """

    _WEIGHTS_URL = (
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
    )
    _INPUT_SIZE = 224  # divisible by 14 for DINOv2 patch grid

    def __init__(self, device: str | None = None):
        super().__init__()
        self._device = device or get_device()

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
        self._transform = T.Compose(
            [
                T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """
        Global DINO-SALAD descriptors, one per image.

        - tensor input: resized to 224 and ImageNet-normalized here, as the PIL transform
        - input raw RGB in [0, 1], never pre-normalized

        Args:
            images: PIL images, or an (N, 3, H, W) RGB tensor in [0, 1].

        Returns:
            (N, D) float32 descriptors, L2-normalized, on CPU.
        """
        # Preprocess: PIL list → stacked tensor, or resize if already a tensor
        if isinstance(images, (list, tuple)):
            imgs = torch.stack([self._transform(img) for img in images])
        else:
            imgs = images.float()
            imgs = T.functional.resize(
                imgs, [self._INPUT_SIZE, self._INPUT_SIZE], antialias=True
            )

            # ImageNet stats, as the PIL path's transform applies
            imgs = T.functional.normalize(imgs, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        logger.debug("DinoSaladExtractor: embedding batch of %d images", len(imgs))

        # Run backbone → aggregator and L2-normalize output descriptors
        imgs = imgs.to(self._device)
        with torch.no_grad():
            feats = self.backbone(imgs)
            descriptors = self.aggregator(feats)
        return torch.nn.functional.normalize(descriptors, p=2, dim=-1).cpu()


########################################################################
# MegaLoc retrieval extractor
########################################################################


@BaseRetrievalExtractor.register("megaloc")
class MegaLocExtractor(BaseRetrievalExtractor):
    """
    MegaLoc global image descriptor for visual place recognition.

    - ported from gmberton/MegaLoc @ 5fe0dd697c4a70ba3e23607f6716ab3c606b16db (MIT): hubconf.py
    - DINOv2 backbone + optimal-transport aggregator; weights from Hugging Face gberton/MegaLoc
    - preprocessing as upstream evaluates: ImageNet normalization, 322 x 322 resize
    """

    _HUB_REPO = "gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
    _INPUT_SIZE = 322  # divisible by 14 for the DINOv2 patch grid

    def __init__(self, device: str | None = None):
        super().__init__()
        self._device = device or get_device()
        self.model = torch.hub.load(
            self._HUB_REPO, "get_trained_model", trust_repo=True
        )
        self.model = self.model.to(self._device).eval()

        # PIL input: tensor in [0, 1], ImageNet stats, upstream's eval size
        self._transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                T.Resize((self._INPUT_SIZE, self._INPUT_SIZE), antialias=True),
            ]
        )

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """
        Global MegaLoc descriptors, one per image.

        - tensor input: ImageNet-normalized then resized to 322 here, as the PIL transform
        - input raw RGB in [0, 1], never pre-normalized

        Args:
            images: PIL images, or an (N, 3, H, W) RGB tensor in [0, 1].

        Returns:
            (N, 8448) float32 descriptors, L2-normalized, on CPU.
        """
        # Preprocess: PIL list through the transform, tensors normalized then resized
        if isinstance(images, (list, tuple)):
            imgs = torch.stack([self._transform(img) for img in images])
        else:
            imgs = images.float()
            imgs = T.functional.normalize(imgs, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            imgs = T.functional.resize(
                imgs, [self._INPUT_SIZE, self._INPUT_SIZE], antialias=True
            )

        logger.debug("MegaLocExtractor: embedding batch of %d images", len(imgs))

        # MegaLoc returns L2-normalized descriptors already
        imgs = imgs.to(self._device)
        with torch.no_grad():
            descriptors = self.model(imgs)

        return descriptors.float().cpu()
