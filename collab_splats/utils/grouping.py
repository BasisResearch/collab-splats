"""
Gaga --> gaussian grouping via multiview association + memory bank

GaGa learns to associate gaussians across views based on a 
memory bank of masks. It functions as a post-processing step
to a trained 3D gaussian splat model. 

Within this implementation, we have separated the gaussian splat
training from the mask association step. The GroupingClassifier
therefore functions to do the following:
    1. Create segmentation masks for each view within the dataset
    2. Associate masks across views 
        - Processes frames sequentially (this is critical for association)
        - Takes a percentage of gaussians within each mask
        - Associates those gaussians across views
    3. Train a classifier to learn identity embeddings for each gaussian
        - Rasterizes the image and aims to reconstruct the associated mask image
        - Assigns an identity embedding to each gaussian (mask ID)

In the original paper, the authors directly rasterize the gaussians
and learn the identity embeddings. 


TLB notes for improvement:
- Could probably side-step rasterization in a smart fashion rather than add a new rasterization step
- Adaptive memory bank: favors large masks currently --> this wont be great for outdoor scenes which are noisy
"""

# Standard library imports
import sys
import random
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Tuple, Optional, Union
import pickle
from pathlib import Path
import shutil
import wandb

# Third party imports
from tqdm import tqdm
import cv2
import numpy as np
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from PIL import Image
from functools import partial

# Silence Python warnings
import os, warnings, logging

warnings.filterwarnings("ignore")
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
logging.getLogger().setLevel(logging.ERROR)

# Nerfstudio imports
from gsplat.strategy import DefaultStrategy

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.data.utils.dataloaders import ImageBatchStream

# Local imports
from collab_splats.utils.segmentation import (
    Segmentation,
    create_composite_mask,
    create_patch_mask,
    convert_matched_mask,
    mask_id_to_binary_mask,
    visualize_mask,
    get_n_different_colors,
)

from collab_splats.utils import project_gaussians, create_fused_features, get_camera_parameters

@staticmethod
def indices_to_bitmask(indices: torch.Tensor, N: int, device=None) -> torch.Tensor:
    """
    Converts a tensor of indices into a bit-packed mask tensor of length ceil(N/32)
    """
    num_ints = (N + 31) // 32
    bitmask = torch.zeros(num_ints, dtype=torch.int32, device=device)
    if indices.numel() == 0:
        return bitmask
    word_idx = indices // 32
    bit_idx = indices % 32
    bitmask[word_idx] |= 1 << bit_idx
    return bitmask


def popcount(x: torch.Tensor) -> torch.Tensor:
    """
    Counts number of set bits (1s) per element using bit shifting.
    Simple and reliable approach for int32 tensors.
    """
    count = torch.zeros_like(x, dtype=torch.int32)
    
    # Check each of the 32 bits
    for i in range(32):
        bit_mask = 1 << i
        count += ((x & bit_mask) != 0).int()
    
    return count

@dataclass
class GroupingConfig:
    segmentation_backend: str
    segmentation_strategy: str
    front_percentage: float = 0.2
    iou_threshold: float = 0.1
    num_patches: int = 32

    # Identity parameters
    # src_feature: str = 'features_rest' # which feature to use as base for the identity
    debug: bool = False

    # Classifier training parameters
    max_steps: int = 10000
    log_every_n_steps: int = 10 # log every n steps
    save_every_n_steps: int = 100 # save a checkpoint every n steps
    accelerator: str = "gpu"
    precision: int = 16 # 16 for mixed precision, 32 for full precision
    identity_dim: int = 13
    lr_classifier: float = 5e-4
    lr_embeddings: float = 2.5e-3

class GroupingClassifier(pl.LightningModule):
    def __init__(self, load_config: str, config: GroupingConfig):
        super().__init__()

        # Config is where the model directory is --> we want one above
        self.output_dir = Path(load_config).parent.parent / "grouping"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save both raw and associated masks as images
        self.raw_mask_dir = self.output_dir / "masks" / "raw"
        self.associated_mask_dir = self.output_dir / "masks" / "associated"
        self.checkpoint_dir = self.output_dir / "checkpoints"

        # Make directories
        self.raw_mask_dir.mkdir(parents=True, exist_ok=True)
        self.associated_mask_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save the hyperparameters (including the output directory)
        self.save_hyperparameters(
            ignore=["pipeline", "model", "segmentation", "datamanager"]
        )

        # Set variables
        self.load_config = load_config
        self.config = config
        self.total_masks = 0

        # Load memory bank and set total masks
        self._memory_bank, self.total_masks = self.load_memory_bank()

    @property
    def identities(self):
        if not hasattr(self, "params"):
            self.setup()

        # Project the identities
        return self.params.identities

    @property
    def objects(self):
        if not hasattr(self, "params"):
            self.setup()

        # Convert identities to classes
        classes = self.per_gaussian_forward(self.identities)
        
        return classes

    @property
    def memory_bank(self) -> list[torch.Tensor]:
        """
        Expose memory bank as list of CPU tensors for compatibility / saving.
        """
        return [
            torch.tensor(list(g), dtype=torch.long, device="cpu")
            for g in self._memory_bank
            ]

    def save_memory_bank(self):
        """
        Save CPU memory bank to disk.
        """
        path = self.output_dir / "memory_bank.pkl"
        with open(path, "wb") as f:
            pickle.dump(self._memory_bank, f)

    def load_memory_bank(self):
        path = self.output_dir / "memory_bank.pkl"

        if path.exists():
            with open(path, "rb") as f:
                memory_bank = pickle.load(f)
            print(f"Memory bank loaded from {path} with {len(memory_bank)} masks")
        else:
            print(f"Memory bank not found at {path}, initializing empty memory bank")
            memory_bank = []

        # Determine device for bit-packed memory bank
        if hasattr(self, "pipeline") and torch.cuda.is_available():
            device = self.pipeline.model.device
            N = int(self.pipeline.model.num_points)
            self._memory_bank_bit = [
                indices_to_bitmask(torch.tensor(list(g), device=device), N, device=device)
                for g in memory_bank
            ]
        else:
            # CPU fallback: just use empty list
            self._memory_bank_bit = []

        return memory_bank, len(memory_bank)

    def _reset_memory_bank(self):
        self._memory_bank = []
        self.total_masks = 0
        self._memory_bank_bit = []

    #########################################################
    ################### Model loading #######################
    #########################################################

    def load_pipeline(self):
        """Load NeRF pipeline only if not already loaded."""

        # Load pipeline and model if not present
        if not hasattr(self, "pipeline"): #or not hasattr(self, "pipeline.model"):
            print("Loading NeRF pipeline and model...")
            _, pipeline, _, _ = eval_setup(self.load_config)
            assert isinstance(pipeline.model, SplatfactoModel)
            self.pipeline = pipeline
            # self.model = pipeline.model

            self.pipeline.datamanager.config.cache_images = "disk"
            self.pipeline.datamanager.setup_train()
        else:
            print("Pipeline and model already loaded. Skipping.")

    def load_segmentation(self):
        """Load segmentation module only if not already loaded."""  
        # Load segmentation if not present
        if not hasattr(self, "segmentation"):
            print("Loading segmentation module...")
            self.segmentation = Segmentation(
                backend=self.config.segmentation_backend,
                strategy=self.config.segmentation_strategy,
                device=self.pipeline.model.device
            )
        else:
            print("Segmentation module already loaded. Skipping.")

    def fixed_indices_dataloader(self, split="train") -> list[tuple]:
        """Returns evaluation data in strict dataset order as a list of (camera, data) tuples.
        Works for both cached and disk modes without loading everything into memory.
        This solution disables the internal shuffling temporarily."""
        print(f"Loading {split} ordered data...")

        if split == "train":
            dataset = self.pipeline.datamanager.train_dataset
        else:
            dataset = self.pipeline.datamanager.eval_dataset


        filenames = dataset.image_filenames

        # --- Disk mode ---
        if self.pipeline.datamanager.config.cache_images == "disk":
            # Temporarily disable internal shuffling
            original_shuffle = random.Random.shuffle

            def no_op_shuffle(self, x):
                pass

            random.Random.shuffle = no_op_shuffle  # type: ignore

            dataloader = DataLoader(
                getattr(self.pipeline.datamanager, f"{split}_imagebatch_stream"),
                batch_size=1,
                num_workers=0,
                collate_fn=lambda x: x[0],
            )
            items = list(islice(dataloader, len(dataset)))

            # Restore original shuffle
            random.Random.shuffle = original_shuffle
            return items, filenames

        else:
            cached_data = getattr(self.pipeline.datamanager, f"{split}_cached")
            # --- Cached mode ---
            data = [d.copy() for d in cached_data]  # copy cached data
            _cameras = deepcopy(dataset.cameras).to(self.pipeline.device)
            cameras = []
            for i in range(len(dataset)):
                data[i]["image"] = data[i]["image"].to(self.pipeline.device)
                cameras.append(_cameras[i : i + 1])  # maintain batch dimension

            assert len(dataset.cameras.shape) == 1, "Assumes single batch dimension"
            return list(zip(cameras, data)), filenames

    #########################################################
    ############### Creation of masks #######################
    #########################################################

    def create_masks(self):
        """Run segmentation and save raw masks for each dataset image."""

        if not hasattr(self, "pipeline"):
            self.load_pipeline()

        if not hasattr(self, "segmentation"):
            self.load_segmentation()
        
        # Get the ordered dataset
        ordered_dataset, filenames = self.fixed_indices_dataloader(split="train")

        for camera, data in tqdm(ordered_dataset, desc=f"Creating masks [train]"):

            # Grab the camera and data for the curret frame
            image = data['image']
            camera_idx = camera.metadata['cam_idx']

            # Get image info
            image_path = filenames[camera_idx]
            image_name = image_path.name

            # Set the path to save the mask
            save_path = self.raw_mask_dir / f"{image_name}"

            if save_path.exists():
                continue  # skip if already exists

            # Run segmentation
            segmentation_results = self.segmentation.segment(image.detach().cpu().numpy())

            # Skip if no objects were found
            if segmentation_results is None:
                print (f"No objects found in {image_name}, creating empty mask")
                # Create empty mask
                composite_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
                cv2.imwrite(str(save_path), composite_mask)
                continue

            # Create composite mask
            _, results = segmentation_results
            composite_mask = create_composite_mask(results).astype(np.uint16)

            # Save the composite mask
            cv2.imwrite(str(save_path), composite_mask)
        
    #########################################################
    ############## Association of gaussians #################
    #########################################################

    def _reset_associations(self):
        # Reset the memory bank
        self._reset_memory_bank()

        # Remove all files within the associated mask directory
        if self.associated_mask_dir.exists():
            shutil.rmtree(self.associated_mask_dir)
        self.associated_mask_dir.mkdir(parents=True, exist_ok=True)

        # Remove memory bank file if it exists
        memory_bank_path = self.output_dir / "memory_bank.pkl"
        if memory_bank_path.exists():
            memory_bank_path.unlink()
        
        # Remove processed images file if it exists
        processed_images_path = self.output_dir / "processed_frames.pkl"
        if processed_images_path.exists():
            processed_images_path.unlink()

    def associate(self):
        """
        Builds a memory bank associating Gaussians across multiple views using segmentation masks. Only matters for train set
        """

        if not hasattr(self, "pipeline"):
            self.load_pipeline()
        
        if not hasattr(self, "segmentation"):
            self.load_segmentation()
        # dataloader = self.fixed_indices_dataloader(split="train")

        progress_path = self.output_dir / "processed_frames.pkl"

        # Load already processed frames if available
        if progress_path.exists():
            with open(progress_path, "rb") as f:
                processed_frames = pickle.load(f)
        else:
            processed_frames = set()

        # Get the ordered dataset
        ordered_dataset, filenames = self.fixed_indices_dataloader(split="train")
  
        with torch.no_grad():
            for camera, data in tqdm(ordered_dataset, desc="Processing frames", total=len(ordered_dataset)):
                # Get image and camera index
                image = data['image']
                camera_idx = camera.metadata['cam_idx']

                # Get image info
                image_path = filenames[camera_idx]
                image_name = image_path.name

                # We require images for training the model so need to check that they exist --> need to check for undistorted images as well
                raw_image_path = self.raw_mask_dir / f"{image_name}"
                associated_image_path = self.associated_mask_dir / f"{image_name}"
                image_exists = associated_image_path.exists()

                # Load the composite mask
                composite_mask = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED) # load as uint16 if provided

                # Skip if no objects were found
                if not np.any(composite_mask):
                    print (f"No objects found in {image_name}")
                    # Copy the raw image to the associated image path
                    shutil.copy(raw_image_path, associated_image_path)
                    continue

                # Skip if already processed
                if camera_idx in processed_frames and image_exists:
                    continue

                # Get model outputs
                _ = self.pipeline.model.get_outputs(camera=camera)

                # Create patch mask
                patch_mask = create_patch_mask(image)

                # Select front gaussians for each mask and assign labels
                start_time = time.time()
                mask_gaussians = self.select_front_gaussians(
                    meta=self.pipeline.model.info,
                    composite_mask=composite_mask,
                    patch_mask=patch_mask,
                )

                if self.config.debug:
                    end_time = time.time()
                    print(f"Time taken to select front gaussians: {end_time - start_time} seconds")

                start_time = time.time()
                labels = self._assign_labels(mask_gaussians)

                if self.config.debug:
                    end_time = time.time()
                    print(f"Time taken to assign labels: {end_time - start_time} seconds")

                # Use the labels to convert the composite mask to show the associated labels
                associated_mask = convert_matched_mask(labels, composite_mask).astype(np.uint16)
                cv2.imwrite(associated_image_path, associated_mask)

                start_time = time.time()
                self._update_memory_bank(labels, mask_gaussians)

                if self.config.debug:
                    end_time = time.time()
                    print(f"Time taken to update memory bank: {end_time - start_time} seconds")

                # Mark frame as processed and save progress + memory bank
                processed_frames.add(camera_idx)
                
                with open(progress_path, "wb") as f:
                    pickle.dump(processed_frames, f)
                
                self.save_memory_bank()

    def _assign_labels(self, mask_gaussians: list[torch.Tensor]) -> torch.Tensor:
        device = self.pipeline.model.device if torch.cuda.is_available() else "cpu"
        N = int(self.pipeline.model.num_points)
        num_masks = len(mask_gaussians)

        if self.total_masks == 0:
            labels = torch.arange(num_masks, dtype=torch.long, device=device)
            self.total_masks = num_masks
            return labels

        # Debug: Check mask_gaussians
        if self.config.debug:
            print(f"Debug: num_masks = {num_masks}")
            for i, g in enumerate(mask_gaussians):
                print(f"  mask_gaussians[{i}] size: {len(g)}, sample indices: {g[:5] if len(g) > 0 else 'empty'}")

        # Prepare mask bits
        mask_bits = torch.stack([
            indices_to_bitmask(g.to(device), N, device=device)
            for g in mask_gaussians
        ])
        
        # Debug: Check bitmask conversion
        if self.config.debug:
            print(f"Debug: mask_bits shape: {mask_bits.shape}")
            for i in range(min(3, len(mask_bits))):
                bit_count = popcount(mask_bits[i:i+1]).sum()
                print(f"  mask_bits[{i}] popcount: {bit_count}, expected: {len(mask_gaussians[i])}")

        labels = torch.zeros(num_masks, dtype=torch.long, device=device)
        
        # Calculate sizes correctly - count actual set bits, not tensor elements
        cur_sizes = torch.tensor([
            popcount(indices_to_bitmask(g.to(device), N, device=device)).sum().float()
            for g in mask_gaussians
        ], dtype=torch.float32, device=device)
        
        if self.config.debug:
            print(f"Debug: cur_sizes: {cur_sizes.tolist()}")

        # Prepare memory bank chunks
        bank_chunks = []
        bank_sizes = []
        if self._memory_bank_bit is not None and len(self._memory_bank_bit) > 0:
            chunk_size = 128
            for i in range(0, len(self._memory_bank_bit), chunk_size):
                chunk = torch.stack(self._memory_bank_bit[i:i + chunk_size], dim=0)
                bank_chunks.append(chunk)
                # Pre-calculate sizes for efficiency
                bank_sizes.append(popcount(chunk).sum(dim=1).float())
            
            if self.config.debug:
                print(f"Debug: Created {len(bank_chunks)} bank chunks")
        else:
            if self.config.debug:
                print("Debug: No memory bank or empty memory bank")

        for i, cur_mask in enumerate(mask_bits):
            max_iou = 0.0
            selected_label = -1

            for chunk_idx, (chunk, ref_sizes) in enumerate(zip(bank_chunks, bank_sizes)):
                # Compute intersection using bitwise AND - fix the broadcasting issue
                intersections = popcount(cur_mask.unsqueeze(0) & chunk).sum(dim=1).float()
                
                # Compute union sizes: |A| + |B| - |A ∩ B|
                union_sizes = cur_sizes[i] + ref_sizes - intersections
                
                # IoU computation with proper numerical stability
                ious = intersections / (union_sizes + 1e-8)

                # --- DEBUG OUTPUT ---
                if self.config.debug and chunk_idx == 0:  # Only show first chunk
                    print(f"\nMask {i}, Chunk {chunk_idx}")
                    print("cur_size (bits):", cur_sizes[i].item())
                    print("ref_sizes:", ref_sizes.tolist()[:5])  # Show first 5 only
                    print("intersections:", intersections.tolist()[:5])
                    print("union_sizes:", union_sizes.tolist()[:5])
                    print("ious:", ious.tolist()[:5])
                # -------------------

                chunk_max, idx = torch.max(ious, dim=0)
                if chunk_max > max_iou:
                    max_iou = chunk_max
                    selected_label = chunk_idx * chunk_size + idx.item()

            if max_iou < self.config.iou_threshold:
                selected_label = self.total_masks
                self.total_masks += 1

            labels[i] = selected_label
        # sys.exit(0)
        return labels

        # return labels

    def _update_memory_bank(self, labels: torch.Tensor, mask_gaussians: list[torch.Tensor]):
        device = self.pipeline.model.device if torch.cuda.is_available() else "cpu"
        N = int(self.pipeline.model.num_points)

        if self._memory_bank_bit is None and torch.cuda.is_available():
            # Initialize GPU bit bank if needed
            self._memory_bank_bit = []

        for label, g in zip(labels.tolist(), mask_gaussians):
            g_tensor = g.to(device) if isinstance(g, torch.Tensor) else torch.tensor(list(g), device=device)
            bitmask = indices_to_bitmask(g_tensor, N, device=device) if self._memory_bank_bit is not None else None

            if self._memory_bank_bit is not None:
                if label >= len(self._memory_bank_bit):
                    self._memory_bank_bit.append(bitmask)
                else:
                    self._memory_bank_bit[label] |= bitmask

            # Always update CPU memory bank
            if label >= len(self._memory_bank):
                self._memory_bank.append(set(g_tensor.tolist()))
            else:
                if not isinstance(self._memory_bank[label], set):
                    self._memory_bank[label] = set(self._memory_bank[label].tolist())
                self._memory_bank[label].update(g_tensor.tolist())

    # def _assign_labels(self, mask_gaussians: list[torch.Tensor]) -> torch.Tensor:
    #     """
    #     Assigns a label to each mask's Gaussian set.
    #     If a mask doesn't sufficiently overlap with any existing label group (via IOCUR),
    #     it is assigned a new label, and total_masks is incremented.

    #     Args:
    #         mask_gaussians (list[Tensor]): Each tensor contains the indices of Gaussians
    #         associated with a single mask in the current view.

    #     Returns:
    #         Tensor: A tensor of shape (num_masks,) containing the assigned label for each mask.
    #     """
    #     num_masks = len(mask_gaussians)

    #     # First view: assign each mask a unique label
    #     if self.total_masks == 0:
    #         labels = torch.arange(num_masks, dtype=torch.long)
    #         self.total_masks = num_masks
    #         return labels

    #     labels = torch.zeros(num_masks, dtype=torch.long)

    #     for i, gaussians in enumerate(mask_gaussians):
    #         n_gaussians = len(gaussians)
    #         overlaps = []

    #         # Compare against each label group in memory_bank
    #         for bank in self.memory_bank:
    #             union = torch.unique(torch.cat([bank, gaussians]))
    #             intersection = len(bank) + n_gaussians - len(union)

    #             # IOCUR: intersection / (intersection + current mask size)
    #             io_cur = intersection / (n_gaussians + intersection + 1e-8)
    #             overlaps.append(io_cur)

    #         overlaps = torch.tensor(overlaps, dtype=torch.float32)
    #         selected = torch.argmax(overlaps)

    #         # If no label matches above the threshold → assign new label
    #         if overlaps[selected] < self.config.iou_threshold:
    #             selected = self.total_masks
    #             self.total_masks += 1

    #         labels[i] = selected

    #     return labels

    # def _update_memory_bank(self, labels: torch.Tensor, mask_gaussians: list[torch.Tensor]):
    #     """
    #     Updates the memory bank with newly assigned or updated Gaussians per label.

    #     The memory bank stores sets of Gaussian indices for each unique label. When updating,
    #     new labels get a new set entry, while existing labels have their sets updated with 
    #     the new Gaussian indices.

    #     Args:
    #         labels (torch.Tensor): Tensor of label assignments for each mask
    #         mask_gaussians (list[torch.Tensor]): List of tensors containing Gaussian indices 
    #                                             belonging to each mask

    #     Note:
    #         The memory bank (_memory_bank) is stored as a list of sets for efficient 
    #         uniqueness checking and updates.
    #     """
    #     for label, gaussians in zip(labels.tolist(), mask_gaussians):
    #         gaussians_set = set(gaussians.tolist())
    #         if label >= len(self._memory_bank):
    #             self._memory_bank.append(gaussians_set)
    #         else:
    #             self._memory_bank[label].update(gaussians_set)

    # ------------------ Saving / Loading ------------------

    # def save_memory_bank(self):
    #     """
    #     Save memory bank to disk using pickle.
    #     """
    #     path = self.output_dir / "memory_bank.pkl"

    #     with open(path, "wb") as f:
    #         pickle.dump(self._memory_bank, f)

    # def load_memory_bank(self):
    #     """
    #     Load memory bank from disk using pickle.
    #     """
    #     path = self.output_dir / "memory_bank.pkl"

    #     if not path.exists():
    #         print(f"Memory bank not found at {path}, initializing empty memory bank")
    #         return [], 0

    #     with open(path, "rb") as f:
    #         memory_bank = pickle.load(f)
        
    #     print (f"Memory bank loaded from {path} with {len(memory_bank)} masks")    
    #     return memory_bank, len(memory_bank)
    
    #########################################################
    ############## Gaussian selection #######################
    #########################################################

    def select_front_gaussians(
        self,
        meta: Dict[str, torch.Tensor],
        composite_mask: torch.Tensor,
        patch_mask: torch.Tensor,
    ):
        """
        JIT-compiled version using torch.compile (PyTorch 2.0+).
        Maintains original structure and comments while adding compilation optimization.
        Now with separated helper functions for better code organization.
        """

        proj_results = project_gaussians(meta)
        
        # Get device from proj_results (should be GPU device)
        device = proj_results["proj_flattened"].device

        # Prepare masks = Decimate the composite mask into individual masks
        binary_masks = mask_id_to_binary_mask(composite_mask)
        flattened_masks = torch.tensor(binary_masks, device=device).flatten(start_dim=1)  # (N, H*W)
        
        # Ensure patch_mask is on the same device
        if isinstance(patch_mask, torch.Tensor):
            patch_mask = patch_mask.to(device)
        else:
            patch_mask = torch.tensor(patch_mask, device=device)

        # Collect front gaussians
        front_gaussians = []
        
        for mask in flattened_masks:
            
            # Use compiled function for main processing
            result = self.process_mask_gaussians(
                proj_results,
                mask,
                patch_mask,
                front_percentage=self.config.front_percentage,
                debug=self.config.debug
            )

            front_gaussians.append(result)

        return front_gaussians

    @staticmethod
    @torch.compile(mode="max-autotune")
    def process_mask_gaussians(
        proj_results: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        patch_mask: torch.Tensor,
        front_percentage: float = 0.5,
        debug: bool = False,
    ):
        """
        JIT-compiled function for processing a single mask.
        Optimized for performance with torch.compile.
        """

        ### TLB THIS SECTION COULD BE REFACTORED OUT FOR MORE FLEXIBILITY (PATCH VS NO PATCH)
        # Find intersection between object mask and patch masks
        patch_intersections = mask.unsqueeze(0).unsqueeze(0) & patch_mask

        # Find non-empty patches
        patch_sums = patch_intersections.sum(dim=2)
        non_empty_patches = (patch_sums > 0).nonzero(as_tuple=False)

        if len(non_empty_patches) == 0:
            return torch.tensor([], dtype=torch.long, device=mask.device)

        # Extract all patches at once
        mask_gaussians = []
        patches_data = patch_intersections[
            non_empty_patches[:, 0], non_empty_patches[:, 1]
        ]

        # Go through each non-empty patch and get the front gaussians
        for _, current_patch in enumerate(patches_data):
            # Projected flattened are the pixel coordinates of each gaussian --> current patch is the pixels of the mask
            # Grab gaussians in the current patch
            patch_gaussians = (
                current_patch[proj_results["proj_flattened"]].nonzero().squeeze(-1)
            )

            if len(patch_gaussians) == 0:
                continue

            # Filter valid gaussians using global valid mask
            overlap_mask = proj_results["valid_mask"][patch_gaussians]

            if not overlap_mask.all() and debug:
                invalid_count = (~overlap_mask).sum()
                print(f"Found {invalid_count} gaussians not in the IDs")
                print("Gaussians not in the IDs: ", patch_gaussians[~overlap_mask])

            # Note: Error checking moved outside compiled function for better performance
            patch_gaussians = patch_gaussians[overlap_mask]

            if len(patch_gaussians) == 0:
                continue

            # Grab the depths of the gaussians in the patch
            num_front_gaussians = max(int(front_percentage * len(patch_gaussians)), 1)

            if num_front_gaussians < len(patch_gaussians):
                # Use partial sorting for better performance
                patch_depths = proj_results["proj_depths"][patch_gaussians]
                _, front_indices = torch.topk(
                    patch_depths, num_front_gaussians, largest=False
                )
                selected_gaussians = patch_gaussians[front_indices]
            else:
                selected_gaussians = patch_gaussians

            mask_gaussians.append(selected_gaussians)

        if len(mask_gaussians) > 0:
            mask_gaussians = torch.cat(mask_gaussians)
            return mask_gaussians
        else:
            return torch.tensor([], dtype=torch.long, device=mask.device)

    #########################################################
    ################# Model rendering ########################
    #########################################################

    def _render_identities(self, camera: Cameras) -> torch.Tensor:
        """
        Rasterizes the identity embeddings -- operates similarly to the get_outputs
        method from the 3DGS model, but doesn't return other properties.

        Args:
            camera (Cameras): Camera object
        Returns:
            torch.Tensor: Output tensor of shape (N, self.config.identity_dim)
        """

        # Link the model locally for convenience
        # model = self.pipeline.model

        if not isinstance(camera, Cameras):
            print("Called _render_identities with not a camera")
            return {}

        # Prep the camera for rasterization --> get parameters
        camera_scale_fac = self.pipeline.model._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        # Get camera parameters for rendering (K, viewmats, etc.)
        camera_params = get_camera_parameters(camera, device=self.pipeline.model.device)

        # Get colors
        features_dc = self.pipeline.model.features_dc
        features_rest = self.pipeline.model.features_rest
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
        
        # We need a hack to get features into model gsplat for rendering
        # Convert the SH coefficients to RGB via gsplat
        # Found here: https://github.com/nerfstudio-project/gsplat/issues/529#issuecomment-2575128309
        if self.pipeline.model.config.sh_degree > 0:
            sh_degree_to_use = min(self.pipeline.model.step // self.pipeline.model.config.sh_degree_interval, self.pipeline.model.config.sh_degree)
        else:
            colors = torch.sigmoid(colors).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None
        
        # # Get the colors to pass into the fused features function
        # fused_features = create_fused_features(
        #     means=model.means,
        #     colors=colors,
        #     features=self.identities, # Identities
        #     camera_params=camera_params,
        #     sh_degree_to_use=sh_degree_to_use,
        # )

        # Rasterize the image using the fused features
        with torch.set_grad_enabled(True):
            render, alpha, _, _, _, _ = self.pipeline.model._render(
                means=self.pipeline.model.means,
                quats=self.pipeline.model.quats,
                scales=self.pipeline.model.scales,
                opacities=self.pipeline.model.opacities,
                # colors=fused_features,
                colors=self.identities,
                render_mode="RGB",
                sh_degree_to_use=sh_degree_to_use,
                camera_params=camera_params,
            )

        # # Grab RGB
        # background = self.pipeline.model._get_background_color()
        # rgb = render[:, ..., :3] + (1 - alpha) * background
        # rgb = torch.clamp(rgb, 0.0, 1.0)

        # Grab identity embeddings
        # preds = render[:, ..., 3:3 + self.config.identity_dim]
        preds = render[:, ..., :self.config.identity_dim]

        return {
            # "rgb": rgb.squeeze(0),
            "identities": preds.squeeze(0),
        }

    #########################################################
    ################# Model training ########################
    #########################################################

    def lift_segmentation(
        self,
        logger: Optional[pl.loggers.Logger] = None,
        ckpt_path: Optional[str] = None,
        use_simulated: bool = False,
    ):
        """
        Train the classifier and identity embeddings with checkpointing.
        """
        
        # Initialize the datamodule for training (no validation module)
        datamodule = GroupingDataModule(
            datamanager=self.pipeline.datamanager,
            mask_dir=self.associated_mask_dir,
            use_simulated=use_simulated,
        )

        # ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            save_top_k=1,
            mode="min",
            filename="grouping-classifier",
            dirpath=self.checkpoint_dir,
            every_n_train_steps=self.config.save_every_n_steps,  # optional: save every 100 steps
            save_last=True,
        )

        # Create Trainer
        trainer = Trainer(
            max_steps=self.config.max_steps,
            accelerator=self.config.accelerator,
            precision=self.config.precision,
            default_root_dir=self.checkpoint_dir,
            callbacks=[checkpoint_callback],
            logger=logger,
        )

        # Resume if checkpoint exists
        last_ckpt = self.checkpoint_dir / "last.ckpt"
        if last_ckpt.exists() and ckpt_path is None:
            ckpt_path = str(last_ckpt)

        print (ckpt_path)
        # Start training
        trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt_path)

        return self, checkpoint_callback.best_model_path

    def setup(self, stage: str = None):
        """
        Initialize trainable parameters and classifier.
        Only creates them if they are not already present (e.g., after loading a checkpoint).
        """
        # Load Splatfacto model if needed
        if not hasattr(self, "pipeline"): # or not hasattr(self, "pipeline.model"):
            self.load_pipeline()

        # Identity embeddings
        if not hasattr(self, "params"):
            n_gaussians = self.pipeline.model.num_points

            # Initialize identities randomly and pass through RGB2SH
            identities = torch.nn.Parameter(
                # self.model.distill_features.clone()
                torch.randn((n_gaussians, self.config.identity_dim), device=self.pipeline.model.device) # don't think i need to pass through RGB2SH here
            )

            # Store identities
            self.params = torch.nn.ParameterDict({"identities": identities})

            self.colors = get_n_different_colors(self.total_masks)

        # Classifier
        if not hasattr(self, "classifier"):
            self.classifier = torch.nn.Conv2d(
                self.config.identity_dim, self.total_masks, kernel_size=1
            ).to(self.pipeline.model.device)

        # Loss function
        if not hasattr(self, "loss_fn"):
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none") #, ignore_index=0)

        # Freeze Splatfacto base model --> change to only gaussian parameter freezing?
        for p in self.pipeline.model.gauss_params.parameters():
            p.requires_grad = False

    def forward(self, camera: Cameras) -> torch.Tensor:
        """
        Forward pass of the classifier --> renders the camera
        viewpoint and creates identity embeddings. Then classifies
        the rasterized identities.
        """
        outputs = self._render_identities(camera)
        
        # Grab identity embeddings
        identities = outputs["identities"]
        identities = identities.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        outputs["logits"] = self.classifier(identities)

        # Classify the identities
        return outputs

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
        # Reshape classifier weights from Conv2d (C_out, C_in, 1, 1) -> (C_out, C_in)
        w = self.classifier.weight.view(self.classifier.out_channels, -1)
        b = self.classifier.bias

        # Apply linear transformation (same as 1x1 conv but on per-Gaussian embeddings)
        logits = F.linear(x, w, b)  # (N, C_out)
        return logits

    def training_step(self, batch, batch_idx: Optional[int] = None):
        """
        Training step of the classifier --> computes the loss
        between the predicted logits and the ground truth segmentation.
        """
        camera, data = batch

        # Put on same device as model (required for rendering)
        camera = camera.to(self.pipeline.model.device)
        seg = data["segmentation"].to(self.pipeline.model.device)

        # Forward pass
        outputs = self(camera)
        logits = outputs["logits"]
        
        # Compute loss
        loss = self.loss_fn(
            logits.unsqueeze(0), 
            seg.unsqueeze(0)
        ).mean()

        # Normalize by number of classes
        loss = loss / torch.log(torch.tensor(self.total_masks))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Log ground truth and predicted segmentation
        if batch_idx % 100 == 0:  # log every 100 batches
            # Assuming seg and logits are HxW or BxHxW
            pred = torch.argmax(logits, dim=0)  # predicted class per pixel
            gt_img = seg.detach().cpu().numpy()
            pred_img = pred.detach().cpu().numpy()
            # rgb_img = outputs["rgb"].detach().cpu().numpy()

            # Visualize the masks
            gt_img = visualize_mask(gt_img, colors=self.colors)
            pred_img = visualize_mask(pred_img, colors=self.colors)

            # Stack side by side: RGB, GT, Pred
            combined = np.concatenate([gt_img, pred_img], axis=1)  # H x (3*W)
            combined = (combined * 255).astype(np.uint8)  # scale if needed

            self.logger.experiment.log({
                "segmentation_comparison": [wandb.Image(combined, caption=f"Batch {batch_idx}")]
            })

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer to train the classifier and identity embeddings.
        """
        # params = list(self.classifier.parameters()) + list(self.params.values())
        # return torch.optim.Adam(params, lr=self.config.lr)

        optimizer = torch.optim.Adam([
            {'params': self.classifier.parameters(), 'lr': self.config.lr_classifier},
            {'params': self.params.values(), 'lr': self.config.lr_embeddings}  # Higher learning rate for embeddings
        ])

        return optimizer

# from dataclasses import fields, is_dataclass

# def setup_datamanager_config(config, datamanager_cls, **kwargs):
#     """
#     Replace the datamanager config in a pipeline while preserving overlapping fields.
#     Works even if config.pipeline.datamanager is a dict (from YAML).
    
#     Args:
#         config: TrainerConfig or dict-loaded config
#         datamanager_cls: The new datamanager config class (e.g., GroupingDataManagerConfig)
#         **kwargs: Any additional fields to override in the new config
#     """
#     old_dm_config = config.pipeline.datamanager

#     # Convert to dict if it's a dataclass, or copy if it's already a dict
#     if is_dataclass(old_dm_config):
#         old_dict = {f.name: getattr(old_dm_config, f.name) for f in fields(old_dm_config)}
#     elif isinstance(old_dm_config, dict):
#         old_dict = old_dm_config.copy()
#     else:
#         raise TypeError(f"Unsupported datamanager type: {type(old_dm_config)}")

#     # Only keep fields that exist in the new datamanager class
#     new_fields = {f.name for f in fields(datamanager_cls)}
#     filtered_dict = {k: v for k, v in old_dict.items() if k in new_fields}

#     # Merge with any explicit overrides
#     filtered_dict.update(kwargs)

#     # Instantiate new datamanager config
#     new_dm_config = datamanager_cls(**filtered_dict)
#     config.pipeline.datamanager = new_dm_config
#     return config


#########################################################
############### Dataloading stuff #######################
#########################################################

class GroupingDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping a Nerfstudio DataManager and associated masks."""

    def __init__(self, datamanager, mask_dir, device="cpu",
                 train_num_workers=0, val_num_workers=0, use_simulated=False):
        super().__init__()
        self.datamanager = datamanager
        self.mask_dir = Path(mask_dir)
        self.device = device
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.use_simulated = use_simulated # for testing

    def _create_simulated_segmentation(self, image_idx: int, h: int = 256, w: int = 256) -> torch.Tensor:
        """Create simple synthetic segmentation patterns for testing."""
        # pattern_type = image_idx % 3
        
        # if pattern_type == 0:
        # Horizontal stripes (3 classes)
        segmentation = torch.arange(h).unsqueeze(1).repeat(1, w) // (h // 10)
        # elif pattern_type == 1:
        #     # Vertical stripes
        #     segmentation = torch.arange(w).unsqueeze(0).repeat(h, 1) // (w // 3)
        # else:
        # Simple center vs border
        # cy, cx = h // 2, w // 2
        # y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        # dist = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5
        # segmentation = (dist > min(h, w) // 4).long()
            
        return segmentation.long()  # Ensure 3 classes: 0, 1, 2

    def _load_segmentation_processor(self, dataset, camera, data: Dict) -> Tuple:
        """Load and attach a segmentation mask for a given camera view."""
        if self.use_simulated:
            segmentation = self._create_simulated_segmentation(
                data["image_idx"], 
                h=data['image'].shape[0], 
                w=data['image'].shape[1]
            )
        else:
            # Original mask loading code
            image_name = dataset.image_filenames[data["image_idx"]].name
            mask_path = self.mask_dir / image_name

            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            segmentation = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            segmentation = segmentation.astype(np.int32)
            segmentation = torch.from_numpy(segmentation).long()
        
        data["segmentation"] = segmentation
        return camera, data

    def _create_dataloader(self, dataset, num_workers: int):
        processor = partial(self._load_segmentation_processor, dataset)
        return DataLoader(
            ImageBatchStream(
                input_dataset=dataset,
                custom_image_processor=processor,
            ),
            batch_size=None,
            num_workers=num_workers,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.datamanager.train_dataset, self.train_num_workers)

    def val_dataloader(self):
        return self._create_dataloader(self.datamanager.eval_dataset, self.val_num_workers)