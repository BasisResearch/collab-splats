"""MaskCLIP feature extractor backend."""
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device
from .base import BaseQueryableExtractor, TORCH_HOME

logger = logging.getLogger(__name__)


@BaseQueryableExtractor.register("maskclip")
class MaskCLIPExtractor(BaseQueryableExtractor):
    """
    A module that extracts patch-level features from images using a CLIP model.

    Args:
        model_name (str): Name of the CLIP model to use. Defaults to 'ViT-L/14@336px'.
        resolution (int): Longest-edge resize target. Defaults to 1024.
        cache_dir (str): Directory to cache model weights. Defaults to TORCH_HOME.
        device: Torch device string. Defaults to auto-detected device.
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14@336px",
        resolution: int = 1024,
        cache_dir: str = TORCH_HOME,
        device: Optional[str] = None,
        **kwargs,  # passes svd_components and any future BaseFeatureExtractor params through
    ):
        if device is None:
            device = get_device()
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__
        self.default_resolution = resolution

        # Lazy import: maskclip_onnx depends on pkg_resources.packaging which was removed
        # in setuptools>=71. Import here so the module is importable even if maskclip_onnx
        # has broken transitive deps — failures only surface when MaskCLIPExtractor is used.
        import maskclip_onnx  # noqa: PLC0415

        # Load the MaskCLIP model from the onnx implementation --> loads model + preprocessor
        # Preprocessor here is composed of a square crop, so we implement the transform ourselves
        self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
        self._maskclip_onnx = maskclip_onnx  # retain reference for encode_text

        # Move model to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()

        # Grab the patch size
        self.patch_size = self.model.visual.patch_size

        # Taken from https://github.com/vuer-ai/feature-splatting/blob/main/feature_splatting/feature_extractor.py
        # Transform for object-level CLIP features (used within feature splatting)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @property
    def device(self) -> torch.device:
        """Device of the underlying model parameters."""
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def preprocess(self, image, resolution: Optional[int] = None) -> torch.Tensor:
        """Open, resize, normalize an image and return a (C, H, W) tensor on device.

        Args:
            image: PIL Image or path accepted by open_image().
            resolution: longest-edge resize target. Defaults to self.default_resolution.
        """
        resolution = resolution if resolution is not None else self.default_resolution
        image = open_image(image).convert("RGB")
        image = resize_image(image, longest_edge=resolution)
        return self.transform(image).to(self.device)

    def forward(self, images: list, resolution: Optional[int] = None) -> list[torch.Tensor]:
        """Extract patch-level CLIP features from a list of images."""
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess each image in the list and stack into a batch tensor
        preprocessed = [self.preprocess(img, resolution) for img in images]
        stacked = torch.stack(preprocessed)

        # Grab shapes such that we can reshape the output features back to (C, H, W) per image after extraction
        b, _, H, W = stacked.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # Forward pass getting model features
        with torch.no_grad():
            # MaskCLIP's get_patch_encodings returns (B, N_patches, D) where N_patches = patch_h * patch_w.
            features = self.model.get_patch_encodings(stacked).to(torch.float32)

            # Normalize features to unit length for cosine similarity, then reshape to (B, C, H_p, W_p).
            features = F.normalize(features, dim=-1)
            features = features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

        # Return a list of features for each image
        result = list(features)
        return result

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Compute CLIP embeddings based on a set of queries.

        Args:
            text (List[str]): List of text queries to encode.

        Returns:
            torch.Tensor: Encoded text features of shape (B, D), where B is the number of queries
                         and D is the embedding dimension.
        """
        tokens = self._maskclip_onnx.clip.tokenize(text).to(self.device)
        embed = self.model.encode_text(tokens).float()
        embed /= embed.norm(dim=-1, keepdim=True)
        return embed
