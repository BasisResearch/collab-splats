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
from tqdm import trange
from typing import List, Tuple, Any, Generator
import numpy as np
import cv2
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

########################################################
########## CLIP Feature Extraction Utils ###############
########################################################

class MaskCLIPExtractor(nn.Module):
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

        # Setup preprocessing information
        self.patch_size = self.model.visual.patch_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = device

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
        image = self.transform(image).to(self.device)

        return image
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from input images.
        
        Args:
            img (torch.Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Extracted features of shape (B, C, H/patch_size, W/patch_size)
        """
        b, _, input_size_h, input_size_w = image.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(image).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

########################################################
########## DINO Feature Extraction Utils ###############
########################################################

class DINOFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14", resolution=800, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.device = device

        self.model = load_torchhub_model('facebookresearch/dinov2', model_name).to(device)
        self.model.eval()

        self.resolution = resolution

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

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