"""
Datamanager for extracting and managing image features for feature splatting.

This module provides functionality to:
1. Extract DINO and CLIP features from images
2. Cache features to disk for faster loading
3. Split features into train/eval sets
4. Provide features during training/evaluation

Based on https://github.com/vuer-ai/feature-splatting/blob/main/feature_splatting/feature_splatting_datamgr.py
"""

import gc
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, List, Optional
from PIL import Image
from tqdm import trange
import numpy as np
import torch
from jaxtyping import Float

from enum import Enum

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from ns_extension.utils.features import BaseFeatureExtractor, pytorch_gc, resize_image
from ns_extension.utils.segmentation import Segmentation, aggregate_masked_features

@dataclass
class FeatureSplattingDataManagerConfig(FullImageDatamanagerConfig):
    """Configuration for the FeatureSplattingDataManager."""

    _target: Type = field(default_factory=lambda: FeatureSplattingDataManager)

    main_features: Literal["samclip"] = "samclip"
    """Type of features to extract - SAMCLIP or CLIP."""

    regularization_features: Literal["dinov2", None] = "dinov2"
    """Type of features to use for regularization."""

    enable_cache: bool = True
    """Whether to cache extracted features to disk."""

    segmentation_backend: str = "mobilesamv2"
    """Segmentation model to use for mask generation."""

    sam_resolution: int = 1024
    """Resolution of SAM features."""

    obj_resolution: int = 100
    """Resolution of object-level features."""

    final_resolution: int = 64
    """Resolution of final features."""


class FeatureSplattingDataManager(FullImageDatamanager):
    """DataManager that handles feature extraction and management for feature splatting."""

    config: FeatureSplattingDataManagerConfig

    def __init__(self, *args, **kwargs):
        """Initialize the data manager and extract/load features."""
        super().__init__(*args, **kwargs)

        # Extract or load cached features for all images
        self.features_dict = self.setup()
        self._set_metadata(self.features_dict)

        # Split features into train and eval sets
        self.train_features, self.eval_features = self.split_train_test_features(self.features_dict)

        # Cleanup
        del self.features_dict
        torch.cuda.empty_cache()
        gc.collect()
    
    def setup(self) -> Dict[str, Float[torch.Tensor, "n h w c"]]:
        """Set up feature extraction or load from cache.
        
        Returns:
            Dict mapping feature types to tensors of extracted features.
        """
        # Get all image paths
        image_filenames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames

        # Set up cache path
        cache_dir = self.config.dataparser.data
        cache_path = cache_dir / f"feature-splatting_{self.config.feature_type.lower()}-features.pt"

        # Try loading from cache if enabled
        if self.config.enable_cache and cache_path.exists():
            cache_dict = torch.load(cache_path)

            if cache_dict.get("image_filenames") != image_filenames:
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            else:
                return cache_dict["features_dict"]
        else:
            CONSOLE.print("Cache does not exist, extracting features...")

        # Extract features
        CONSOLE.print(f"Extracting {self.config.feature_type} features for {len(image_filenames)} images...")
        features_dict = self.extract_features(image_filenames)

        # Cache features if enabled
        if self.config.enable_cache:
            cache_dict = {
                "image_filenames": image_filenames,
                "features_dict": features_dict
            }
            cache_dir.mkdir(exist_ok=True)
            torch.save(cache_dict, cache_path)
            CONSOLE.print(f"Saved {self.config.feature_type} features to cache at {cache_path}")
        
        return features_dict

    def extract_features(self, image_filenames: List[str]) -> Dict[str, Float[torch.Tensor, "n h w c"]]:
        """Extract DINO and CLIP features from images.
        
        Args:
            image_filenames: List of paths to images to process.
            
        Returns:
            Dictionary mapping feature types to lists of feature tensors.
        """

        features_dict = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.config.regularization_features is not None:
            features_dict[self.config.regularization_features] = []

            # Create extractor for regularization features --> extract
            extractor = BaseFeatureExtractor.get(self.config.regularization_features)(device=device)

            for i in trange(len(image_filenames), desc="Extracting DINO features"):
                image, target_H, target_W = extractor.preprocess(image_filenames[i])
                features = extractor.forward(image)
                features = extractor.reshape(features, target_H, target_W)
                features = features.detach().cpu()
                features_dict[self.config.regularization_features].append(features)

            del extractor
            pytorch_gc()

        # Extract main features with segmentation masks
        extractor = BaseFeatureExtractor.get(self.config.main_features)(device=device)
        segmentation = Segmentation(
            backend=self.config.segmentation_backend,
            strategy=self.config.segmentation_strategy,
            device=device,
        )

        for i in trange(len(image_filenames), desc="Extracting CLIP features"):
            # Load and process image
            image = Image.open(image_filenames[i])
            H, W = image.height, image.width

            # Calculate resolutions
            object_W = self.config.obj_resolution
            object_H = H * object_W // W
            final_W = self.config.final_resolution
            final_H = H * final_W // W

            # Extract features
            inputs = extractor.preprocess(image)
            features = extractor.forward(inputs[None])[0]

            # Prepare image for segmentation
            image = resize_image(image, self.config.sam_resolution) # Resize image to SAM resolution
            image = np.asarray(image) # Convert to numpy array

            # Apply segmentation masks over features
            masks = segmentation.segment(image)

            if masks is None:
                # Add an all-zero tensor if no object is detected
                features_dict[self.config.main_features].append(torch.zeros((features.shape[0], final_H, final_W)))
                continue
            
            features = aggregate_masked_features(
                features, 
                masks,
                resolution=(object_H, object_W),
                final_resolution=(final_H, final_W)
            )

            features = features.detach().cpu()
            features_dict[self.config.main_features].append(features)

            # Clear memory after each image
            del features, masks
            torch.cuda.empty_cache()
            gc.collect()

        del extractor, segmentation
        pytorch_gc()

        # Stack features along batch dimension
        for k in features_dict.keys():
            features_dict[k] = torch.stack(features_dict[k], dim=0)  # BCHW

        return features_dict

    def split_train_test_features(self, features_dict: Dict[str, Float[torch.Tensor, "n h w c"]]) -> Tuple[Dict[str, Float[torch.Tensor, "n h w c"]], Dict[str, Float[torch.Tensor, "n h w c"]]]:
        """Split features into training and evaluation sets.
        
        Args:
            features_dict: Dictionary of all extracted features.
            
        Returns:
            Tuple of (train_features, eval_features) dictionaries.
        """
        train_size = len(self.train_dataset)
        eval_size = len(self.eval_dataset)
        total_size = train_size + eval_size
    
        # Validate feature lengths
        for feature_name, features in features_dict.items():
            if len(features) != total_size:
                raise ValueError(f"Feature {feature_name} has length {len(features)}, expected {total_size}")
        
        train_features = {model: features[:train_size] for model, features in features_dict.items()}
        eval_features = {model: features[train_size:] for model, features in features_dict.items()}

        return train_features, eval_features

    def _set_metadata(self, features_dict: Dict[str, Float[torch.Tensor, "n h w c"]]):
        """Set feature metadata in the dataset.
        
        Args:
            features_dict: Dictionary of extracted features.
        """
        feature_dims = {model: features.shape[1:] for model, features in features_dict.items()}
        metadata = {
            "feature_type": self.config.feature_type,
            "feature_dims": feature_dims,
        }
        self.train_dataset.metadata.update(metadata)

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Get next training batch with features.
        
        Args:
            step: Current training step.
            
        Returns:
            Tuple of (camera, data dict with features).
        """
        camera, data = super().next_train(step)
        camera_idx = camera.metadata['cam_idx']
        features_dict = {}
        for model, features in self.train_features.items():
            features_dict[model] = features[camera_idx]
        data["features_dict"] = features_dict
        return camera, data
    
    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Get next evaluation batch with features.
        
        Args:
            step: Current evaluation step.
            
        Returns:
            Tuple of (camera, data dict with features).
        """
        camera, data = super().next_eval(step)
        camera_idx = camera.metadata['cam_idx']
        features_dict = {}
        for model, features in self.eval_features.items():
            features_dict[model] = features[camera_idx]
        data["features_dict"] = features_dict
        return camera, data