"""
Module for feature extraction and processing.

This module provides utilities for extracting and processing features from images using CLIP models.
It includes classes and functions for:
- Loading and initializing CLIP models
- Extracting CLIP features from images 
- Aggregating features based on segmentation masks
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from typing import List, Tuple, Any, Generator, Dict, Optional
import numpy as np
from PIL import Image
import maskclip_onnx
from pathlib import Path
import gc
from huggingface_hub import hf_hub_download

TORCH_HOME = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

########################################################
########## General feature extraction utils ############
########################################################

def load_hf_weights(repo_id: str, filename: str):
    """
    Download a file from Hugging Face.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename)

def load_torchhub_model(repo_id: str, model_name: str):
    """
    Load a model from torch.hub.
    """
    return torch.hub.load(repo_id, model_name)

def pytorch_gc():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

def resize_image(image: Image.Image, longest_edge: int) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio so its longest edge equals the specified length.

    Args:
        img (Image.Image): Input PIL image to resize
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

def interpolate_to_patch_size(img_bchw: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
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
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """
    Batch iterator for MobileSAM --> helps with memory usage

    Inputs:
        - batch_size: int
        - *args: List[Any]

    Returns:
        - Generator[List[Any], None, None]

    Taken from feature-splatting
    """
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

######################################################################
########## Define BaseFeatureExtractor for registration ##############
######################################################################

class BaseFeatureExtractor(nn.Module):
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Unknown extractor '{name}'. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

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
    def __init__(self, model_name: str = 'ViT-L/14@336px', cache_dir: str = TORCH_HOME, device: str = "cpu"):
        super().__init__()

        # Load model
        self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
        self.model = self.model.to(device)
        self.model.eval()

        # Setup preprocessing
        self.patch_size = self.model.visual.patch_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = device

    def to(self, device: str):
        self.model = self.model.to(device)
        self.device = device
        return super().to(device)

    def preprocess(self, image, resolution: int = 1024) -> torch.Tensor:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass # Already a PIL image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
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
            torch.Tensor: Encoded text features of shape (B, D), where B is the number of queries and D is the embedding dimension.
        """

        # Tokenize text and compute embeddings
        tokens = maskclip_onnx.clip.tokenize(text).to(self.device)
        embed = self.model.encode_text(tokens).float()

        # Normalize embeddings
        embed /= embed.norm(dim=-1, keepdim=True)
        return embed 

    def compute_similarity(
        self, 
        features: torch.Tensor,
        positive: List[str], 
        negative: Optional[List[str]] = None, 
        softmax_temp: float = 0.05,
        method: str = "standard"
    ) -> torch.Tensor:
        """
        Compute similarity probability map between image features and text queries.
        
        Args:
            features (torch.Tensor): Image features of shape (C, H, W)
            positive (List[str]): List of positive text queries
            negative (List[str], optional): List of negative text queries. 
                                                   If None, uses default negatives.
            softmax_temp (float): Temperature parameter for softmax
            method (str): Method to use for computing similarity.
                          "standard" uses standard softmax.
                          "pairwise" uses pairwise softmax.
        Returns:
            torch.Tensor: Similarity probability map of shape (H, W, 1)
        """
        # Use default negatives if none provided
        if negative is None:
            negative = ["object"]
        
        # Encode all text queries
        queries = positive + negative
        text_embeddings = self.encode_text(queries)
        
        # Compute raw similarities: (N, H, W)
        raw_similarities = torch.einsum("chw,nc->nhw", features, text_embeddings)
        
        # Flatten spatial dimensions: (N, H*W)
        raw_similarities = raw_similarities.reshape(raw_similarities.shape[0], -1)
        
        # Apply softmax with temperature
        probs = (raw_similarities / softmax_temp).softmax(dim=0)
        
        # Sum positive probabilities and reshape
        num_positive = len(positive)

        if method == "standard":
            similarity = probs[:num_positive].sum(dim=0) # Similarity map
        elif method == "pairwise":
            # Pairwise softmax: average positive vs each negative individually
            pos_similarities = raw_similarities[:num_positive]  # (num_positive, num_pixels)
            neg_similarities = raw_similarities[num_positive:]  # (num_negative, num_pixels)
            
            # Average positive similarities
            avg_pos_similarity = pos_similarities.mean(dim=0, keepdim=True)  # (1, num_pixels)
            
            # Broadcast to match negative shape
            broadcasted_pos = avg_pos_similarity.expand(neg_similarities.shape[0], -1)  # (num_negative, num_pixels)
            
            # Create pairs: positive vs each negative
            paired_similarities = torch.cat([broadcasted_pos, neg_similarities], dim=0)  # (2*num_negative, num_pixels)
            
            # Compute pairwise softmax
            probs = (paired_similarities / softmax_temp).softmax(dim=0)
            
            # Extract positive probabilities and take minimum across pairs
            pos_pair_probs = probs[:neg_similarities.shape[0]]  # First half are positive probs
            pos_similarity = pos_pair_probs.min(dim=0)[0]  # Take minimum across all pairs
            
            # Handle NaN values 
            similarity = torch.nan_to_num(pos_similarity, nan=0.0)
            
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'standard' or 'pairwise'")

        return similarity.reshape(features.shape[1:] + (1,))  # (H, W, 1)
    
######################################################################
############### DINO Feature Extraction Utils ########################
######################################################################

@BaseFeatureExtractor.register("dinov2")
class DINOFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model_name: str = "dinov2_vits14", resolution=800, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name

        self.model = load_torchhub_model('facebookresearch/dinov2', model_name).to(device)
        self.model.eval()

        self.resolution = resolution

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.device = device

    def to(self, device: str):
        self.model = self.model.to(device)
        self.device = device
        return super().to(device)

    def preprocess(self, image) -> torch.Tensor:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass # Already a PIL image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        image = resize_image(image, longest_edge=self.resolution)
        image = self.transform(image)[:3].unsqueeze(0)

        # Setup for DINO --> interpolating overall image to be evenly divisible by patch size
        image, target_H, target_W = interpolate_to_patch_size(image, self.model.patch_size)
        image = image.to(self.device)

        return image, target_H, target_W

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            features = self.model.forward_features(image)["x_norm_patchtokens"][0]
        
        return features

    def reshape(self, features: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
        features = features.cpu()
        features_hwc = features.reshape((target_H // self.model.patch_size, target_W // self.model.patch_size, -1))
        features_chw = features_hwc.permute((2, 0, 1))

        return features_chw

######################################################################
################### Decoder utilities ################################
######################################################################

class TwoLayerMLP(nn.Module):
    """
    A two-layer MLP implemented using 1x1 convolutions for reconstructing feature maps.
    The network consists of:
    - A shared hidden 1x1 convolution layer (acts as the intermediate representation).
    - A set of task-specific output branches, each also a 1x1 convolution, producing different feature maps.
    
    Attributes:
        hidden_conv (nn.Conv2d): Shared hidden layer.
        feature_branch_dict (nn.ModuleDict): Dictionary of output branches, one for each model type.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        features_dim_dict: Dict[str, Tuple[int, int, int]]
    ):
        """
        Args:
            input_dim (int): Number of input channels.
            hidden_dim (int): Number of channels in the intermediate hidden layer.
            feature_dim_dict (dict): Dictionary mapping feature names to output shapes (C, H, W). 
                                     Only the channel dimension (C) is used here.
        """
        super().__init__()
        self.hidden_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        # Create a branch for each model type
        self.feature_branch_dict = nn.ModuleDict({
            model: nn.Conv2d(hidden_dim, feat_shape[0], kernel_size=1)
            for model, feat_shape in features_dim_dict.items()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass using 2D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)
        
        Returns:
            dict: Dictionary mapping feature names to output tensors of shape (B, C_out, H, W)
        """
        x = F.relu(self.hidden_conv(x))
        return {model: conv(x) for model, conv in self.feature_branch_dict.items()}

    @torch.no_grad()
    def per_gaussian_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass using fully-connected (linear) layers assuming `x` is a flattened per-Gaussian input.
        This mimics convolution using linear operations by flattening weights.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in), where N is number of Gaussians.
        
        Returns:
            outputs: Dictionary mapping feature names to output tensors of shape (N, C_out)
        """
        # Flatten 1x1 conv weights into (C_out, C_in)
        w_hidden = self.hidden_conv.weight.view(self.hidden_conv.out_channels, -1)
        x = F.relu(F.linear(x, w_hidden, self.hidden_conv.bias))
        
        outputs = {}
        for model, conv in self.feature_branch_dict.items():
            w_out = conv.weight.view(conv.out_channels, -1)
            outputs[model] = F.linear(x, w_out, conv.bias)
        
        return outputs