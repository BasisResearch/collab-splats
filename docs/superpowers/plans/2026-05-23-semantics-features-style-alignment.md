# Semantics & Features Module Style Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `semantics/features.py` and `nerfstudio/datamanagers/features.py` into full alignment with `segmentation/base.py` style — block comments, docstrings, section dividers, blank lines. Zero logic changes.

**Architecture:** Pure readability pass. Two files touched sequentially. Each file committed separately. Test suite validates no regressions.

**Tech Stack:** Python 3.11, `black`, `isort`, `pytest`

**Spec:** `docs/superpowers/specs/2026-05-23-semantics-features-style-alignment-design.md`

---

## Files Modified

- `collab_splats/semantics/features.py` — docstrings, blank lines, block comments
- `collab_splats/nerfstudio/datamanagers/features.py` — full alignment: header, imports, `from __future__`, dividers, block comments

---

## Task 1: Establish test baseline

**Files:** none modified

- [ ] **Step 1: Run existing test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_semantics_logging.py tests/test_feedforward_logging.py -v 2>&1 | tail -20
```

Expected: all tests pass. Note any pre-existing failures before proceeding.

---

## Task 2: `semantics/features.py` — docstring normalization

**Files:**
- Modify: `collab_splats/semantics/features.py:482-508` (MaskCLIPExtractor.forward)
- Modify: `collab_splats/semantics/features.py:683-695` (Talk2DinoExtractor.encode_text)

- [ ] **Step 1: Fix MaskCLIPExtractor.forward() docstring**

Replace:
```python
    def forward(self, images: list, resolution: Optional[int] = None) -> list[torch.Tensor]:
        """
        Given a list of images, preprocess and extract patch-level features using the CLIP model.
        """
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))
```
With:
```python
    def forward(self, images: list, resolution: Optional[int] = None) -> list[torch.Tensor]:
        """Extract patch-level CLIP features from a list of images."""
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))
```

- [ ] **Step 2: Fix Talk2DinoExtractor.encode_text() docstring**

Replace:
```python
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text queries to normalized embeddings.

        Args:
            texts: List of text strings

        Returns:
            torch.Tensor: normalized embeddings of shape (N, D)
        """
```
With:
```python
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text queries to normalized embeddings.

        Args:
            texts: list of text strings.

        Returns:
            (N, D) normalized embeddings.
        """
```

- [ ] **Step 3: Verify no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_semantics_logging.py -v 2>&1 | tail -10
```

Expected: same pass/fail as Task 1.

---

## Task 3: `semantics/features.py` — blank lines in `__init__` bodies

**Files:**
- Modify: `collab_splats/semantics/features.py:538-558` (DINOFeatureExtractor.__init__)
- Modify: `collab_splats/semantics/features.py:611-639` (Talk2DinoExtractor.__init__)

- [ ] **Step 1: DINOFeatureExtractor.__init__ — add block comments and blank line before transform**

Replace the body after `super().__init__(**kwargs)`:
```python
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__
        self.model_name = model_name

        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size: int = self.model.config.patch_size
        self.resolution = resolution

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
```
With:
```python
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__
        self.model_name = model_name

        # Load DINOv2 from HuggingFace and move to device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size: int = self.model.config.patch_size
        self.resolution = resolution

        # Standard ViT image normalization transform
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
```

- [ ] **Step 2: Talk2DinoExtractor.__init__ — add blank lines between model load / patch_size / device**

Replace:
```python
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__
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
```
With:
```python
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
```

---

## Task 4: `semantics/features.py` — block comments for bare forward() methods

**Files:**
- Modify: `collab_splats/semantics/features.py:582-595` (DINOFeatureExtractor.forward)
- Modify: `collab_splats/semantics/features.py:665-681` (Talk2DinoExtractor.forward)

- [ ] **Step 1: DINOFeatureExtractor.forward() — add docstring and block comments**

Replace:
```python
    def forward(self, images: list) -> list[torch.Tensor]:
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))
        preprocessed = [self.preprocess(img) for img in images]
        tensors = torch.cat([t for t, _, _ in preprocessed], dim=0).to(self.device)
        with torch.no_grad():
            features = self.model(tensors).last_hidden_state[:, 1:]
        results = []
        for i, (_, H, W) in enumerate(preprocessed):
            patch_features = features[i].cpu()
            reshaped = patch_features.reshape(
                H // self.patch_size, W // self.patch_size, -1
            ).permute(2, 0, 1)
            results.append(reshaped)
        return results
```
With:
```python
    def forward(self, images: list) -> list[torch.Tensor]:
        """Extract patch-level DINOv2 features from a list of images."""
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess each image: resize, normalize, pad to patch-aligned dimensions
        preprocessed = [self.preprocess(img) for img in images]

        # Batch all images into a single tensor and move to model device
        tensors = torch.cat([t for t, _, _ in preprocessed], dim=0).to(self.device)

        # Run DINOv2 forward; drop CLS token (index 0), keep patch tokens
        with torch.no_grad():
            features = self.model(tensors).last_hidden_state[:, 1:]

        # Reshape each image's flat patch sequence to (C, H_p, W_p) spatial layout
        results = []
        for i, (_, H, W) in enumerate(preprocessed):
            patch_features = features[i].cpu()
            reshaped = patch_features.reshape(
                H // self.patch_size, W // self.patch_size, -1
            ).permute(2, 0, 1)
            results.append(reshaped)
        return results
```

- [ ] **Step 2: Talk2DinoExtractor.forward() — add docstring and block comments**

Replace:
```python
    def forward(self, images: list) -> list[torch.Tensor]:
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))
        preprocessed = [self.preprocess(img) for img in images]
        with torch.no_grad():
            result = self._model.encode_image(preprocessed)
        patch_tokens = list(result) if isinstance(result, torch.Tensor) else result
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
```
With:
```python
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
```

- [ ] **Step 3: Run tests and commit**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_semantics_logging.py -v 2>&1 | tail -10
```

```bash
git add collab_splats/semantics/features.py
git commit -m "style(semantics): normalize docstrings, add block comments, visual blank lines"
```

---

## Task 5: `nerfstudio/datamanagers/features.py` — header, imports, logger

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py:1-34`

- [ ] **Step 1: Replace file header (docstring + imports + logger)**

Replace lines 1–34 (everything before `_EXTRACTOR_NAME`):
```python
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
from typing import Dict, Literal, Tuple, Type, List, Union
from pathlib import Path
from PIL import Image
from tqdm import trange
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.utils.torch_utils import infer_batch_size, pytorch_gc
from collab_splats.utils.image import resize_image
from collab_splats.semantics.segmentation import MobileSAMSegmentation, aggregate_masked_features
```
With:
```python
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
from tqdm import trange

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
```

- [ ] **Step 2: Verify import smoke-test**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManager; print('ok')"
```
Expected: `ok`

---

## Task 6: `nerfstudio/datamanagers/features.py` — section dividers

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py`

- [ ] **Step 1: Add divider before module-level helpers**

Add before `_EXTRACTOR_NAME = ...`:
```python

########################################################
########## Module-level helpers ########################
########################################################

```

- [ ] **Step 2: Add divider before config class**

Add before `@dataclass`:
```python

########################################################
########## Config ######################################
########################################################

```

- [ ] **Step 3: Add divider before DataManager class**

Add before `class FeatureSplattingDataManager`:
```python

########################################################
########## DataManager #################################
########################################################

```

---

## Task 7: `nerfstudio/datamanagers/features.py` — block comments in all methods

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py`

- [ ] **Step 1: `__init__` — add block comments**

Replace:
```python
    def __init__(self, *args, **kwargs):
        """Initialize the data manager and extract/load features."""
        super().__init__(*args, **kwargs)

        # Extract or load cached features for all images
        self.features_dict = self.setup()
        self._set_metadata(self.features_dict)

        # Split features into train and eval sets
        self.train_features, self.eval_features = self.split_train_test_features(
            self.features_dict
        )

        # Cleanup
        del self.features_dict
        torch.cuda.empty_cache()
        gc.collect()
```
With:
```python
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
```

- [ ] **Step 2: `setup()` — add block comments**

Replace:
```python
    def setup(self) -> Dict[str, torch.Tensor]:
        """Set up feature extraction or load from cache.

        Returns:
            Dict mapping feature types to tensors of extracted features.
        """
        # Get all image paths
        image_filenames = (
            self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        )

        # Set up cache path — resolve to absolute so CWD changes don't affect it
        cache_dir = Path(self.config.dataparser.data).resolve()
        cache_path = (
            cache_dir / f"feature-splatting_{self.config.main_features}-features.pt"
        )

        # Try loading from cache if enabled
        if self.config.enable_cache and cache_path.exists():
            cache_dict = torch.load(cache_path)

            if _cache_filenames(cache_dict.get("image_filenames", [])) != _cache_filenames(image_filenames):
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            else:
                return cache_dict["features_dict"]
        else:
            CONSOLE.print("Cache does not exist, extracting features...")

        # Extract features
        CONSOLE.print(
            f"Extracting {self.config.main_features} features for {len(image_filenames)} images..."
        )
        features_dict = self.extract_features(image_filenames)

        # Cache features if enabled
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
```
With:
```python
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
```

- [ ] **Step 3: `extract_features()` — add missing block comments**

Replace:
```python
    def extract_features(
        self, image_filenames: List[Union[str, Path]]
    ) -> Dict[str, torch.Tensor]:
        """Extract DINO and CLIP features from images.

        Args:
            image_filenames: List of paths to images to process.

        Returns:
            Dictionary mapping feature types to lists of feature tensors.
        """

        features_dict: Dict[str, List[torch.Tensor]] = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.config.regularization_features is not None:
            features_dict[self.config.regularization_features] = []

            # Create extractor for regularization features --> extract
            extractor = BaseFeatureExtractor.get(self.config.regularization_features)(
                device=device
            )

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

        # Extract main features
        extractor_name = _EXTRACTOR_NAME.get(self.config.main_features, self.config.main_features)
        extractor = BaseFeatureExtractor.get(extractor_name)(device=device)
        use_seg = self.config.main_features == "samclip"
        segmentation = None
        if use_seg:
            segmentation = MobileSAMSegmentation(
                strategy=self.config.segmentation_strategy,
                device=device,
            )

        # Add empty list for main features
        features_dict[self.config.main_features] = []

        for i in trange(
            len(image_filenames),
            desc=f"Extracting {self.config.main_features} features",
        ):
            # Load and process image
            image = Image.open(image_filenames[i])
            H, W = image.height, image.width

            # Calculate resolutions
            object_W = self.config.obj_resolution
            object_H = H * object_W // W
            final_W = self.config.final_resolution
            final_H = H * final_W // W

            # Extract features
            [features] = extractor.forward([image])

            if use_seg:
                # Prepare image for segmentation
                image = resize_image(image, self.config.sam_resolution)
                image = np.asarray(image)

                # Apply segmentation masks over features
                seg_outputs = segmentation.segment(image)

                # Add an all-zero tensor if no object is detected
                if seg_outputs is None:
                    features_dict[self.config.main_features].append(
                        torch.zeros((features.shape[0], final_H, final_W))
                    )
                    del features
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

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

            # Clear memory after each image
            del features
            torch.cuda.empty_cache()
            gc.collect()

        del extractor
        if segmentation is not None:
            del segmentation
        pytorch_gc()

        # Stack features along batch dimension
        for k, v in list(features_dict.items()):
            features_dict[k] = torch.stack(v, dim=0)  # BCHW

        return features_dict
```
With:
```python
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
```

- [ ] **Step 4: `split_train_test_features()` — add block comments**

Replace:
```python
    def split_train_test_features(
        self, features_dict: Dict[str, torch.Tensor]
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
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
        for model_name, features in features_dict.items():
            if len(features) != total_size:
                raise ValueError(
                    f"Feature {model_name} has length {len(features)}, expected {total_size}"
                )

        train_features = {
            model_name: features[:train_size]
            for model_name, features in features_dict.items()
        }
        eval_features = {
            model_name: features[train_size:]
            for model_name, features in features_dict.items()
        }

        return train_features, eval_features
```
With:
```python
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
```

- [ ] **Step 5: `_set_metadata()` — add block comments**

Replace:
```python
    def _set_metadata(self, features_dict: Dict[str, torch.Tensor]):
        """Set feature metadata in the dataset.

        Args:
            features_dict: Dictionary of extracted features.
        """
        feature_dims = {
            model_name: features.shape[1:]
            for model_name, features in features_dict.items()
        }
        metadata = {
            "feature_type": self.config.main_features,
            "feature_dims": feature_dims,
        }
        if getattr(self.train_dataset, "metadata", None) is None:
            self.train_dataset.metadata = {}
        self.train_dataset.metadata.update(metadata)
```
With:
```python
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
```

- [ ] **Step 6: `next_train()` and `next_eval()` — add block comments**

Replace:
```python
    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Get next training batch with features.

        Args:
            step: Current training step.

        Returns:
            Tuple of (camera, data dict with features).
        """
        camera, data = super().next_train(step)
        camera_idx = camera.metadata["cam_idx"]
        features_dict = {}
        for model_name, features in self.train_features.items():
            features_dict[model_name] = features[camera_idx]
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
        camera_idx = camera.metadata["cam_idx"]
        features_dict = {}
        for model_name, features in self.eval_features.items():
            features_dict[model_name] = features[camera_idx]
        data["features_dict"] = features_dict
        return camera, data
```
With:
```python
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
```

---

## Task 8: Final checks and commit

**Files:** none modified

- [ ] **Step 1: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: same pass/fail as Task 1 baseline.

- [ ] **Step 2: Import smoke-test both modules**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.semantics.features import (
    BaseFeatureExtractor, MaskCLIPExtractor, DINOFeatureExtractor, Talk2DinoExtractor
)
from collab_splats.nerfstudio.datamanagers.features import (
    FeatureSplattingDataManager, FeatureSplattingDataManagerConfig
)
print('all imports ok')
"
```

Expected: `all imports ok`

- [ ] **Step 3: Commit datamanager changes**

```bash
git add collab_splats/nerfstudio/datamanagers/features.py
git commit -m "style(datamanager): align to semantics module — from __future__, dividers, block comments"
```
