"""Datamanager for extracting and managing image features for feature splatting.

Provides:
  FeatureSplattingDataManagerConfig — config dataclass (main_features, regularization, cache settings)
  FeatureSplattingDataManager      — extracts/caches features, splits train/eval, serves per step
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import trange

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.semantics.segmentation import MobileSAMSegmentation, aggregate_masked_features
from collab_splats.utils.image import resize_image
from collab_splats.utils.torch_utils import infer_batch_size, pytorch_gc

########################################################
########## Module-level helpers ########################
########################################################

_EXTRACTOR_NAME: dict[str, str] = {"samclip": "maskclip"}


def _cache_filenames(filenames: list) -> list[str]:
    """Canonical cache key: sorted filename stems, independent of path prefix and downscale folder."""
    return sorted(Path(f).name for f in filenames)


########################################################
########## Config ######################################
########################################################

@dataclass
class FeatureSplattingDataManagerConfig(FullImageDatamanagerConfig):
    """Configuration for the FeatureSplattingDataManager."""

    _target: Type = field(default_factory=lambda: FeatureSplattingDataManager)

    main_features: Literal["maskclip", "samclip", "talk2dino"] = "samclip"
    """Feature extractor for main training features.

    "samclip": patch CLIP features masked by SAM segmentation (default).
        Use with regularization_features="dinov2".
    "maskclip": patch CLIP features, no segmentation.
        Use with regularization_features="dinov2".
    "talk2dino": DINOv3 + CLIP projection — structural grounding is built in;
        set regularization_features=None. Images are center-cropped to
        square during extraction (feature spatial coverage != full frame).
    """

    regularization_features: Literal["dinov2", None] = "dinov2"
    """Type of features to use for regularization."""

    enable_cache: bool = True
    """Whether to cache extracted features to disk."""

    segmentation_backend: str = "mobilesamv2"
    """Segmentation model to use for mask generation."""

    segmentation_strategy: str = "object"
    """Segmentation strategy to use for mask generation."""

    sam_resolution: int = 1024
    """Resolution of SAM features."""

    obj_resolution: int = 100
    """Resolution of object-level features."""

    final_resolution: int = 64
    """Resolution of final features."""


########################################################
########## DataManager #################################
########################################################

class FeatureSplattingDataManager(FullImageDatamanager):
    """DataManager that handles feature extraction and management for feature splatting."""

    config: FeatureSplattingDataManagerConfig

    def __init__(self, *args, **kwargs):
        """Initialize the data manager and extract/load features."""
        super().__init__(*args, **kwargs)

        # Extract or load cached features; set feature metadata on the dataset
        self.features_dict = self.setup()
        self._set_metadata(self.features_dict)

        # Slice into train and eval dicts; each maps feature-type → (N, C, H, W) tensor
        self.train_features, self.eval_features = self.split_train_test_features(
            self.features_dict
        )

        # Free full feature dict; train/eval copies now own the memory
        del self.features_dict
        torch.cuda.empty_cache()
        gc.collect()

    def setup(self) -> Dict[str, torch.Tensor]:
        """Extract features or load from disk cache; return feature dict.

        Returns:
            Dict mapping feature-type name → (N, C, H, W) tensor, train+eval combined.
        """
        # Gather all image paths: train split first, then eval (order must match split_train_test_features)
        image_filenames = (
            self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        )

        # Resolve cache path to absolute so CWD changes don't affect it
        cache_dir = Path(self.config.dataparser.data).resolve()
        cache_path = (
            cache_dir / f"feature-splatting_{self.config.main_features}-features.pt"
        )

        # Return cached features if the cache is valid (same extractor name + same filenames)
        if self.config.enable_cache and cache_path.exists():
            cache_dict = torch.load(cache_path)

            if _cache_filenames(cache_dict.get("image_filenames", [])) != _cache_filenames(image_filenames):
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            else:
                return cache_dict["features_dict"]
        else:
            CONSOLE.print("Cache does not exist, extracting features...")

        # Run full extraction pipeline
        CONSOLE.print(
            f"Extracting {self.config.main_features} features for {len(image_filenames)} images..."
        )
        features_dict = self.extract_features(image_filenames)

        # Persist to disk for future runs
        if self.config.enable_cache:
            cache_dict = {
                "image_filenames": image_filenames,
                "features_dict": features_dict,
            }
            cache_dir.mkdir(exist_ok=True)
            torch.save(cache_dict, cache_path)
            CONSOLE.print(
                f"Saved {self.config.main_features} features to cache at {cache_path}"
            )

        return features_dict

    def extract_features(
        self, image_filenames: List[Union[str, Path]]
    ) -> Dict[str, torch.Tensor]:
        """Extract regularization and main features from all images.

        Args:
            image_filenames: ordered list of image paths (train then eval).

        Returns:
            Dict mapping feature-type name → (N, C, H, W) stacked tensor.
        """
        features_dict: Dict[str, List[torch.Tensor]] = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Extract regularization features (e.g. DINOv2) in batches if configured
        if self.config.regularization_features is not None:
            features_dict[self.config.regularization_features] = []

            extractor = BaseFeatureExtractor.get(self.config.regularization_features)(
                device=device
            )

            # Infer batch size from per-image memory estimate to avoid OOM
            sample_image = Image.open(image_filenames[0])
            mem_per_image = extractor._get_memory_per_image(sample_image)
            batch_size = infer_batch_size(mem_per_image)
            name = self.config.regularization_features
            for start in trange(0, len(image_filenames), batch_size, desc=f"Extracting {name} features"):
                batch_paths = image_filenames[start : start + batch_size]
                for features in extractor.forward(batch_paths):
                    features_dict[name].append(features.detach().cpu())

            del extractor
            pytorch_gc()

        # Initialize main feature extractor; resolve alias (samclip → maskclip)
        extractor_name = _EXTRACTOR_NAME.get(self.config.main_features, self.config.main_features)
        extractor = BaseFeatureExtractor.get(extractor_name)(device=device)

        # samclip mode: apply SAM segmentation masks before aggregating CLIP features
        use_seg = self.config.main_features == "samclip"
        segmentation = None
        if use_seg:
            segmentation = MobileSAMSegmentation(
                strategy=self.config.segmentation_strategy,
                device=device,
            )

        features_dict[self.config.main_features] = []

        for i in trange(
            len(image_filenames),
            desc=f"Extracting {self.config.main_features} features",
        ):
            # Load image and derive aspect-preserving intermediate/final resolutions
            image = Image.open(image_filenames[i])
            H, W = image.height, image.width
            object_W = self.config.obj_resolution
            object_H = H * object_W // W
            final_W = self.config.final_resolution
            final_H = H * final_W // W

            # Run extractor forward pass on single image
            [features] = extractor.forward([image])

            if use_seg:
                # Resize and array-convert image for SAM input
                image = resize_image(image, self.config.sam_resolution)
                image = np.asarray(image)

                # Segment image; fall back to zero tensor if no objects detected
                seg_outputs = segmentation.segment(image)

                if seg_outputs is None:
                    features_dict[self.config.main_features].append(
                        torch.zeros((features.shape[0], final_H, final_W))
                    )
                    del features
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                # Pool CLIP features per SAM mask region
                masks = seg_outputs[0]
                features = aggregate_masked_features(
                    features,
                    masks,
                    resolution=(object_H, object_W),
                    final_resolution=(final_H, final_W),
                )
                del masks

            features = features.detach().cpu()
            features_dict[self.config.main_features].append(features)

            # Release GPU memory after each image to stay within cgroup cap
            del features
            torch.cuda.empty_cache()
            gc.collect()

        del extractor
        if segmentation is not None:
            del segmentation
        pytorch_gc()

        # Stack per-image lists into (N, C, H, W) batch tensors
        for k, v in list(features_dict.items()):
            features_dict[k] = torch.stack(v, dim=0)  # BCHW

        return features_dict

    def split_train_test_features(
        self, features_dict: Dict[str, torch.Tensor]
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Split combined feature tensor into train and eval dicts.

        Args:
            features_dict: Dict of feature-type → (N_train+N_eval, C, H, W) tensors.

        Returns:
            Tuple of (train_features, eval_features) with same keys as input.
        """
        train_size = len(self.train_dataset)
        eval_size = len(self.eval_dataset)
        total_size = train_size + eval_size

        # Guard: feature count must equal total dataset size (extract_features preserves order)
        for model_name, features in features_dict.items():
            if len(features) != total_size:
                raise ValueError(
                    f"Feature {model_name} has length {len(features)}, expected {total_size}"
                )

        # Slice first train_size frames for training; remainder for eval
        train_features = {
            model_name: features[:train_size]
            for model_name, features in features_dict.items()
        }
        eval_features = {
            model_name: features[train_size:]
            for model_name, features in features_dict.items()
        }

        return train_features, eval_features

    def _set_metadata(self, features_dict: Dict[str, torch.Tensor]):
        """Write feature-type and per-type shape metadata onto the training dataset.

        Args:
            features_dict: Dict of feature-type → (N, C, H, W) tensors.
        """
        # Build per-type spatial shape dict: {name: (C, H, W)} — excludes batch dim
        feature_dims = {
            model_name: features.shape[1:]
            for model_name, features in features_dict.items()
        }
        metadata = {
            "feature_type": self.config.main_features,
            "feature_dims": feature_dims,
        }

        # Merge into train_dataset.metadata, initializing the dict if absent
        if getattr(self.train_dataset, "metadata", None) is None:
            self.train_dataset.metadata = {}
        self.train_dataset.metadata.update(metadata)

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Return next training camera and data dict, with pre-extracted features attached.

        Args:
            step: current training step.

        Returns:
            Tuple of (camera, data dict containing ``features_dict``).
        """
        # Fetch base camera + data from parent datamanager
        camera, data = super().next_train(step)

        # Index into pre-extracted train features by camera index
        camera_idx = camera.metadata["cam_idx"]
        features_dict = {
            model_name: features[camera_idx]
            for model_name, features in self.train_features.items()
        }
        data["features_dict"] = features_dict
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Return next eval camera and data dict, with pre-extracted features attached.

        Args:
            step: current eval step.

        Returns:
            Tuple of (camera, data dict containing ``features_dict``).
        """
        # Fetch base camera + data from parent datamanager
        camera, data = super().next_eval(step)

        # Index into pre-extracted eval features by camera index
        camera_idx = camera.metadata["cam_idx"]
        features_dict = {
            model_name: features[camera_idx]
            for model_name, features in self.eval_features.items()
        }
        data["features_dict"] = features_dict
        return camera, data
