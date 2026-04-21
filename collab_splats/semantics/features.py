"""
Feature extraction classes and utilities.

Provides a registry-based system (BaseFeatureExtractor) for image feature extractors:
  - MaskCLIPExtractor: patch-level CLIP features via maskclip_onnx
  - DINOFeatureExtractor: patch-level DINOv2 features via torch.hub
  - Talk2DinoExtractor: patch features + text-conditioned heatmaps via Talk2DINO (HF Hub)
"""

import math
import os
import gc
from pathlib import Path
from typing import Dict, Generator, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

try:
    import maskclip_onnx
    _MASKCLIP_AVAILABLE = True
except ImportError:
    maskclip_onnx = None  # type: ignore[assignment]
    _MASKCLIP_AVAILABLE = False

TORCH_HOME = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
_DEFAULT_NEGATIVE: list[str] = ["object"]


def _open_image(image) -> Image.Image:
    """
    Load an image from various input types into a PIL Image.

    Args:
        image: Input image as str path, Path object, ndarray, or PIL Image.

    Returns:
        PIL Image.Image

    Raises:
        ValueError: If image type is not supported.
    """
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


########################################################
########## General feature extraction utils ############
########################################################


def load_hf_weights(repo_id: str, filename: str):
    """Download a file from Hugging Face."""
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_torchhub_model(repo_id: str, model_name: str):
    """Load a model from torch.hub."""
    return torch.hub.load(repo_id, model_name)


def pytorch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def infer_batch_size(mem_per_image_gb: float, headroom: float = 0.3) -> int:
    """Compute a safe batch size from available VRAM.

    Args:
        mem_per_image_gb: Estimated GPU memory per image in GB (use extractor.MEM_PER_IMAGE_GB).
        headroom: Fraction of VRAM to use for batch data; remainder reserved for model weights.

    Returns:
        Batch size >= 1. Returns 1 if CUDA is unavailable.
    """
    if mem_per_image_gb <= 0:
        raise ValueError(f"mem_per_image_gb must be positive, got {mem_per_image_gb}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return max(1, int(vram_gb * headroom / mem_per_image_gb))
    return 1


def resize_image(image: Image.Image, longest_edge: int) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio so its longest edge equals the specified length.

    Args:
        image (Image.Image): Input PIL image to resize
        longest_edge (int): Target length for the longest edge of the image

    Returns:
        Image.Image: Resized PIL image with longest edge equal to longest_edge
    """
    width, height = image.size
    if width > height:
        ratio = longest_edge / width
    else:
        ratio = longest_edge / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return image.resize((new_width, new_height), Image.BILINEAR)


def interpolate_to_patch_size(
    img_bchw: torch.Tensor, patch_size: int
) -> Tuple[torch.Tensor, int, int]:
    """
    Interpolate an image tensor so its height and width are evenly divisible by patch_size.

    Args:
        img_bchw (torch.Tensor): Input image tensor of shape (B, C, H, W)
        patch_size (int): Size of patches the image will be divided into

    Returns:
        Tuple containing:
            - torch.Tensor: Interpolated image tensor
            - int: New height that is divisible by patch_size
            - int: New width that is divisible by patch_size
    """
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = F.interpolate(
        img_bchw, size=(target_H, target_W), mode="bilinear", align_corners=False
    )
    return img_bchw, target_H, target_W


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """
    Batch iterator for MobileSAM -- helps with memory usage.

    Inputs:
        - batch_size: int
        - *args: List[Any]

    Returns:
        - Generator[List[Any], None, None]

    Taken from feature-splatting
    """
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args), (
        "Batched iteration must have inputs of all the same size."
    )
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


######################################################################
########## Define BaseFeatureExtractor for registration ##############
######################################################################


class BaseFeatureExtractor(nn.Module):
    _registry: Dict[str, type["BaseFeatureExtractor"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise ValueError(
                f"Unknown extractor '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    def forward_batch(self, preprocessed: list) -> torch.Tensor:
        """Run inference on a pre-processed batch. Subclasses must override."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward_batch()")

    def reshape_batch(self, batch: torch.Tensor, idx: int, *args) -> torch.Tensor:
        """Reshape flat patch tokens to spatial feature maps. Subclasses must override."""
        raise NotImplementedError(f"{type(self).__name__} must implement reshape_batch()")


######################################################################
############### CLIP Feature Extraction Utils ########################
######################################################################


@BaseFeatureExtractor.register("samclip")
class MaskCLIPExtractor(BaseFeatureExtractor):
    """
    A module that extracts patch-level features from images using a CLIP model.

    Args:
        clip_model_name (str): Name of the CLIP model to use. Defaults to 'ViT-L/14@336px'.
        cache_dir (str): Directory to cache model weights. Defaults to TORCH_HOME.
    """

    MEM_PER_IMAGE_GB: float = 3.0  # CLIP ViT-L/14@336px

    def forward_batch(self, preprocessed: list) -> torch.Tensor:
        """Batch inference. preprocessed: list of (C,H,W) tensors from preprocess()."""
        images = torch.stack(preprocessed)  # (B, C, H, W)
        return self.forward(images)  # (B, C_feat, pH, pW)

    def reshape_batch(self, batch: torch.Tensor, idx: int, *_) -> torch.Tensor:
        return batch[idx]  # (C_feat, pH, pW) — already correctly shaped

    def __init__(
        self,
        model_name: str = "ViT-L/14@336px",
        cache_dir: str = TORCH_HOME,
        device: str = "cpu",
    ):
        if not _MASKCLIP_AVAILABLE:
            raise ImportError(
                "maskclip_onnx is not installed. Install with: pip install maskclip_onnx"
            )
        super().__init__()

        # Load model
        self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
        self.model = self.model.to(device)
        self.model.eval()

        # Setup preprocessing
        self.patch_size = self.model.visual.patch_size
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @property
    def device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def preprocess(self, image, resolution: int = 1024) -> torch.Tensor:
        image = _open_image(image).convert("RGB")
        image = resize_image(image, longest_edge=resolution)
        return self.transform(image).to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from input images.

        Args:
            img (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Extracted features of shape (B, C, H/patch_size, W/patch_size)

        Seems like it needs to be on GPU otherwise throws an error due to precision (specific to maskclip_onnx)
        """
        b, _, input_size_h, input_size_w = image.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size

        with torch.no_grad():
            features = self.model.get_patch_encodings(image).to(torch.float32)
            features = features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

        return features

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Compute CLIP embeddings based on a set of queries.

        Args:
            text (List[str]): List of text queries to encode.

        Returns:
            torch.Tensor: Encoded text features of shape (B, D), where B is the number of queries
                         and D is the embedding dimension.
        """
        tokens = maskclip_onnx.clip.tokenize(text).to(self.device)
        embed = self.model.encode_text(tokens).float()
        embed /= embed.norm(dim=-1, keepdim=True)
        return embed

    def compute_similarity(
        self,
        features: torch.Tensor,
        positive: List[str],
        negative: Optional[List[str]] = None,
        softmax_temp: float = 0.05,
        method: str = "standard",
    ) -> torch.Tensor:
        """
        Compute similarity probability map between image features and text queries.

        Args:
            features (torch.Tensor): Image features of shape (C, H, W)
            positive (List[str]): List of positive text queries
            negative (List[str], optional): List of negative text queries.
                                                   If None, uses default negatives.
            softmax_temp (float): Temperature parameter for softmax
            method (str): "standard" or "pairwise"

        Returns:
            torch.Tensor: Similarity probability map of shape (H, W, 1)
        """
        if negative is None:
            negative = _DEFAULT_NEGATIVE

        queries = positive + negative
        text_embeddings = self.encode_text(queries)

        raw_similarities = torch.einsum("chw,nc->nhw", features, text_embeddings)
        raw_similarities = raw_similarities.reshape(raw_similarities.shape[0], -1)
        probs = (raw_similarities / softmax_temp).softmax(dim=0)
        num_positive = len(positive)

        if method == "standard":
            similarity = probs[:num_positive].sum(dim=0)
        elif method == "pairwise":
            pos_similarities = raw_similarities[:num_positive]
            neg_similarities = raw_similarities[num_positive:]
            avg_pos_similarity = pos_similarities.mean(dim=0, keepdim=True)
            broadcasted_pos = avg_pos_similarity.expand(neg_similarities.shape[0], -1)
            paired_similarities = torch.cat([broadcasted_pos, neg_similarities], dim=0)
            probs = (paired_similarities / softmax_temp).softmax(dim=0)
            pos_pair_probs = probs[: neg_similarities.shape[0]]
            pos_similarity = pos_pair_probs.min(dim=0)[0]
            similarity = torch.nan_to_num(pos_similarity, nan=0.0)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'standard' or 'pairwise'")

        return similarity.reshape(features.shape[1:] + (1,))  # (H, W, 1)


# Register backward-compatible alias expected by tests/metadata
BaseFeatureExtractor._registry["clip-vit"] = MaskCLIPExtractor

######################################################################
############### DINO Feature Extraction Utils ########################
######################################################################


@BaseFeatureExtractor.register("dinov2")
class DINOFeatureExtractor(BaseFeatureExtractor):
    MEM_PER_IMAGE_GB: float = 1.5  # DINOv2 ViTS14 at 800px

    def __init__(
        self, model_name: str = "dinov2_vits14", resolution=800, device: str = "cpu"
    ):
        super().__init__()
        self.model_name = model_name

        self.model = load_torchhub_model("facebookresearch/dinov2", model_name).to(device)
        self.model.eval()

        self.resolution = resolution

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @property
    def device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def preprocess(self, image) -> Tuple[torch.Tensor, int, int]:
        image = _open_image(image)
        image = resize_image(image, longest_edge=self.resolution)
        image = self.transform(image)[:3].unsqueeze(0)

        # Interpolate overall image to be evenly divisible by patch size
        image, target_H, target_W = interpolate_to_patch_size(image, self.model.patch_size)
        image = image.to(self.device)

        return image, target_H, target_W

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model.forward_features(image)["x_norm_patchtokens"][0]
        return features

    def reshape(self, features: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
        features = features.cpu()
        features_hwc = features.reshape(
            (target_H // self.model.patch_size, target_W // self.model.patch_size, -1)
        )
        features_chw = features_hwc.permute((2, 0, 1))
        return features_chw

    def forward_batch(self, preprocessed: list) -> torch.Tensor:
        """Batch inference. preprocessed: list of (tensor_1CHW, H, W) from preprocess().

        All images in the batch must have the same spatial resolution after preprocessing.
        This holds for video-frame datasets (same camera → same resolution).
        """
        tensors = torch.cat([t for t, _, _ in preprocessed], dim=0).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(tensors)["x_norm_patchtokens"]
        return features

    def reshape_batch(self, batch: torch.Tensor, idx: int, target_H: int, target_W: int) -> torch.Tensor:
        return self.reshape(batch[idx], target_H, target_W)



######################################################################
############### Talk2DINO Feature Extraction Utils ###################
######################################################################


@BaseFeatureExtractor.register("talk2dino")
class Talk2DinoExtractor(BaseFeatureExtractor):
    """
    Wraps Talk2DINO models from HuggingFace Hub for patch-level feature extraction
    and text-conditioned semantic heatmap generation.

    Supports DINOv3 (default) and DINOv2 variants:
      - "lorebianchi98/Talk2DINOv3-ViTB"  (default, cleaner interface)
      - "lorebianchi98/Talk2DINO-ViTB"    (DINOv2, older interface)

    Algorithm from Talk2DINO (https://github.com/lorebianchi98/Talk2DINO).
    """

    MEM_PER_IMAGE_GB: float = 1.0  # Talk2DINO ViTB

    def __init__(
        self,
        hf_model_id: str = "lorebianchi98/Talk2DINOv3-ViTB",
        device: str = "cpu",
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for Talk2DinoExtractor. "
                "Install via: pip install transformers"
            ) from e

        self._model = AutoModel.from_pretrained(hf_model_id, trust_remote_code=True).to(device).eval()
        self.patch_size: int = getattr(self._model.config, "patch_size", 14)
        self._device = torch.device(device)

    def forward_batch(self, preprocessed: list) -> torch.Tensor:
        """Batch inference. preprocessed: list of PIL Images from preprocess()."""
        with torch.no_grad():
            result = self._model.encode_image(preprocessed)
        # encode_image may return tensor (B,N,D) or list of (N,D) tensors
        return result if isinstance(result, torch.Tensor) else torch.stack(result)

    def reshape_batch(self, batch: torch.Tensor, idx: int, *_) -> torch.Tensor:
        return batch[idx]  # (N_patches, D) patch tokens

    @property
    def device(self) -> torch.device:
        return self._device

    def preprocess(self, image) -> Image.Image:
        """
        Center-crop image to square. Required by Talk2DINO (from hf_demo.ipynb).

        Returns PIL Image ready for encode_image / forward.
        """
        image = _open_image(image).convert("RGB")
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

    def forward(self, image: Image.Image) -> torch.Tensor:
        """
        Extract patch tokens from a preprocessed PIL image.

        Args:
            image: PIL Image (should be square, output of preprocess())

        Returns:
            torch.Tensor: patch tokens of shape (N_patches, D)
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Talk2DinoExtractor.forward() expects a PIL Image (use preprocess() first)")
        with torch.no_grad():
            return self._model.encode_image([image])[0]

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text queries to normalized embeddings.

        Args:
            texts: List of text strings

        Returns:
            torch.Tensor: normalized embeddings of shape (N, D)
        """
        with torch.no_grad():
            embeddings = self._model.encode_text(texts)
        return F.normalize(embeddings, dim=-1)

    def _compute_similarity(
        self,
        features: torch.Tensor,
        positive: List[str],
        negative: Optional[List[str]] = None,
        softmax_temp: float = 0.05,
        method: str = "standard",
    ) -> torch.Tensor:
        """
        Compute per-patch similarity between image features and text queries.

        Port of compute_similarity() from hf_demo.ipynb.

        Returns:
            torch.Tensor: similarity scores of shape (N_patches,)
        """
        if negative is None:
            negative = _DEFAULT_NEGATIVE

        queries = positive + negative
        with torch.no_grad():
            text_embeddings = self._model.encode_text(queries)

        text_embeddings = F.normalize(text_embeddings, dim=-1)
        features_norm = F.normalize(features, dim=-1)
        raw_similarities = text_embeddings @ features_norm.T  # (num_queries, N_patches)
        num_positive = len(positive)

        if method == "standard":
            probs = (raw_similarities / softmax_temp).softmax(dim=0)
            return probs[:num_positive].sum(dim=0)
        elif method == "pairwise":
            pos_similarities = raw_similarities[:num_positive]
            neg_similarities = raw_similarities[num_positive:]
            avg_pos = pos_similarities.mean(dim=0, keepdim=True)
            broadcasted_pos = avg_pos.expand(neg_similarities.shape[0], -1)
            paired = torch.cat([broadcasted_pos, neg_similarities], dim=0)
            probs = (paired / softmax_temp).softmax(dim=0)
            pos_pair_probs = probs[: neg_similarities.shape[0]]
            return torch.nan_to_num(pos_pair_probs.min(dim=0)[0], nan=0.0)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'standard' or 'pairwise'")

    def compute_semantic_heatmap(
        self,
        image: Image.Image,
        text_pairs: Dict[str, Tuple[List[str], List[str]]],
        softmax_temp: float = 0.05,
        method: str = "standard",
    ) -> Dict[str, np.ndarray]:
        """
        Generate per-label masked image overlays using text-conditioned similarity.

        Port of segment_image() from hf_demo.ipynb.

        Args:
            image: PIL Image (will be preprocessed internally if not already square)
            text_pairs: dict mapping label → (positive_queries, negative_queries)
                        e.g. {"feeder": (["feeder", "bird feeder"], ["background", "sky"])}
            softmax_temp: softmax temperature (lower = sharper masks)
            method: "standard" or "pairwise"

        Returns:
            dict mapping label → HxWxC float32 array (image masked by similarity)
        """
        # Ensure square crop
        image = self.preprocess(image)

        with torch.no_grad():
            img_embed = self._model.encode_image([image])[0]  # (N_patches, D)

        num_patches = img_embed.shape[0]
        grid_size = int(math.isqrt(num_patches))
        img_size = grid_size * self.patch_size

        # Prepare image array at patch-aligned resolution
        img_np = np.array(image).transpose(2, 0, 1).astype(np.float32)  # (3, H, W)
        img_np = (
            F.interpolate(
                torch.tensor(img_np).unsqueeze(0),
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
            / 255.0
        )

        results: Dict[str, np.ndarray] = {}
        for label, (positive, negative) in text_pairs.items():
            sim = self._compute_similarity(img_embed, positive, negative, softmax_temp, method)
            mask = (
                F.interpolate(
                    sim.view(1, 1, grid_size, grid_size),
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                .cpu()
                .squeeze()
                .numpy()
            )  # (img_size, img_size)
            # Return HxWxC masked image (same as hf_demo.ipynb segment_image output)
            results[label] = (img_np * mask[np.newaxis, :, :]).transpose(1, 2, 0)

        return results
