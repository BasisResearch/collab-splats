"""Talk2DINO feature extractor backend."""

import logging
import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModel

from collab_splats.semantics.utils import tokens_to_feature_map
from collab_splats.utils.torch_utils import get_device

from .base import BaseQueryableExtractor

logger = logging.getLogger(__name__)


@BaseQueryableExtractor.register("talk2dino")
class Talk2DinoExtractor(BaseQueryableExtractor):
    """
    Talk2DINO patch features and text-conditioned similarity, from HuggingFace Hub.

    - supports DINOv3 ("lorebianchi98/Talk2DINOv3-ViTB", default) and DINOv2
      ("lorebianchi98/Talk2DINO-ViTB")
    - forward_features is called directly, bypassing encode_image's internal resize:
      preprocess() has already produced patch-aligned tensors
    - algorithm from Talk2DINO (https://github.com/lorebianchi98/Talk2DINO)
    """

    def __init__(
        self,
        model_name: str = "lorebianchi98/Talk2DINOv3-ViTB",
        device: Optional[str] = None,
        resize_mode: str = "max_size",
        image_resolution: int = 512,
        svd_components: int = 500,
    ):
        """
        Args:
            model_name: HuggingFace Hub model ID.
            device: Torch device string ("cpu" or "cuda").
            resize_mode: "max_size" (proportional, longest-edge) or "square" (center-crop + resize).
            image_resolution: Longest-edge target for "max_size"; square side for "square".
            svd_components: top singular vectors kept for positional debiasing.
        """
        if device is None:
            device = get_device()
        super().__init__(resize_mode=resize_mode, image_resolution=image_resolution, svd_components=svd_components)

        # low_cpu_mem_usage=False avoids meta-tensor init: Talk2DINO's HF code calls
        # load_state_dict() without assign=True, so weight copies would silently no-op.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
            _loaded = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=False)

        # Normalize transform from the model's own image_transforms — correct stats per backbone
        self._normalize: T.Normalize = _loaded.image_transforms.transforms[-1]

        # patch_size comes from the backbone conv stride
        # - Conv2d normalizes stride to a tuple
        # - stride, not kernel_size, is what sets the token grid
        # - both supported checkpoints (Talk2DINOv3-ViTB, Talk2DINO-ViTB) expose model.patch_embed.proj
        self.patch_size: int = _loaded.model.patch_embed.proj.stride[0]

        # Move to device after extracting metadata
        self._model = _loaded.to(device).eval()
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        """
        Device of the underlying model parameters.

        Returns:
            The torch device the model was moved to at construction.
        """
        return self._device

    def forward(self, images: list) -> list[torch.Tensor]:
        """
        Extract patch-level Talk2DINO features from a list of images.

        Args:
            images: anything `preprocess` accepts — paths, ndarrays or PIL images.

        Returns:
            One (D, H_p, W_p) float32 CPU tensor per input image.
        """
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess all images and stack into a single batch
        preprocessed = [self.preprocess(img) for img in images]
        batch = torch.stack(preprocessed).to(self._device)

        # Call forward_features directly — bypasses encode_image's internal T.Resize((N,N)).
        # [:, 5:] drops CLS + 4 register tokens (DINOv3 convention) → (B, N_patches, D)
        with torch.no_grad():
            tokens_all = self._model.model.forward_features(batch)[:, 5:]

        # Reshape each image's flat patch sequence to (D, H_p, W_p)
        results = []
        for i, t in enumerate(preprocessed):
            _, H, W = t.shape
            results.append(tokens_to_feature_map(tokens_all[i].cpu(), H, W, self.patch_size))
        return results

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
