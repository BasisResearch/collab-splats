"""
Gaga --> gaussian grouping via multiview association + memory bank

Steps:
1. Create masks --> for each view within the dataset, create masks
    - Original implementation saves them out as images, but we could just save them out as tensors

2. Associate masks --> creates the memory bank?
    - Front percentage (0.2)
    - Overlap threshold (0.1)
    - For each camera --> 
        - If no masks, initialize a memory bank for the first view's masks
        - Get gaussian idxs and zcoords (for depth grouping) for the current view
        - Find front gaussians:
            - Create Spatial patch mask --> divides image into patch grid
            - Object masks --> goes through each mask in the image
            - Combines the two masks (i.e., find overlap between patch and object mask)
            - Find frontmost gaussians within each patch for each object
        - Based on this:
            - Stores the indices of the front gaussians
            - Mask ID = tensor of ALL indices of that mask (i.e., all gaussians in that mask)
            - Num masks == number of masks in the memory bank


TLB notes for improvement:
- Adaptive memory bank: favors large masks currently --> this wont be great for outdoor scenes which are noisy
"""

# Standard library imports
import random
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from typing import Dict
import pickle

# Third party imports
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

# Silence Python warnings
import os, warnings, logging

warnings.filterwarnings("ignore")
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
logging.getLogger().setLevel(logging.ERROR)

# Nerfstudio imports
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup

# Local imports
from collab_splats.utils.segmentation import (
    Segmentation,
    create_composite_mask,
    create_patch_mask,
    convert_matched_mask,
    mask_id_to_binary_mask,
)
from collab_splats.utils.utils import project_gaussians

@dataclass
class GroupingConfig:
    segmentation_backend: str
    segmentation_strategy: str
    front_percentage: float = 0.2
    iou_threshold: float = 0.1
    num_patches: int = 32

    # Identity parameters
    src_feature: str = 'features_rest' # which feature to use as base for the identity
    identity_dim: int = 16
    lr: float = 5e-4

    # Lightning training parameters
    max_epochs: int = 10
    val_check_interval: float = 1.0
    log_every_n_steps: int = 10
    precision: int = 16
    accelerator: str = "gpu"
    devices: int = 1
    
    # Callback parameters
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 5

    debug: bool = False

class GroupingDataModule(pl.LightningDataModule):
    """Data module for GroupingClassifier training."""
    
    def __init__(self, train_dataloader, val_dataloader=None, test_dataloader=None, batch_size=1, num_workers=0):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        """Setup data for training/validation/testing."""
        pass
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self.train_dataloader
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return self.val_dataloader
    
    def test_dataloader(self):
        """Return test dataloader."""
        return self.test_dataloader

@dataclass
class GroupingClassifier(pl.LightningModule):
    def __init__(self, load_config: str, config: GroupingConfig):
        super().__init__()

        self.load_config = load_config
        self.config = config
        self.total_masks = 0

        # Config is where the model directory is --> we want one above
        self.output_dir = load_config.parent.parent / 'grouping'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # eval_setup(load_config)
        # self.num_masks = num_masks
        # self.num_gaussians = num_gaussians
        # self.classifier = nn.Conv2d(in_channels=num_masks, out_channels=num_gaussians, kernel_size=1)
        self._memory_bank: list[set[int]] = self.load_memory_bank()

        # Load pipeline and segmentation
        self._load_models()

    @property
    def memory_bank(self) -> list[torch.Tensor]:
        """
        Expose memory bank as list of tensors (on-demand conversion).
        """
        return [
            torch.tensor(list(g), dtype=torch.long, device="cpu")
            for g in self._memory_bank
        ]

    @property
    def identities(self):
        if not hasattr(self, "params"):
            self.setup_train()
        return self.params.identities

    #########################################################
    ################### Model loading #######################
    #########################################################

    def setup(self):
        self.memory_bank = []
        self._load_models()
    
    def setup_train(self):
        n_gaussians = self.model.num_points
        src_dim = self.model.gauss_params[self.config.src_feature].shape[-1]
        identities = torch.nn.Parameter(torch.zeros((n_gaussians, self.config.identity_dim)))

        self.params = torch.nn.ParameterDict(
            {
                "identities": identities,
            }
        )

        # In projection is simple linear projection followed by nonlinearity
        # We piggyback off an existing feature and use that as a method to create identities
        self.in_proj = nn.Sequential(
            nn.Linear(src_dim, self.config.identity_dim),
            nn.ReLU(),
        )

        # Fit identity of gaussians by predicting masks within each view
        self.classifier = torch.nn.Conv2d(self.config.identity_dim, self.total_masks, kernel_size=1)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.config.lr)

    def _load_models(self):
        """Load NeRF pipeline only if not already loaded."""
        
        # Load pipeline and model if not present
        if not hasattr(self, "pipeline") or not hasattr(self, "model"):
            print("Loading NeRF pipeline and model...")
            _, pipeline, _, _ = eval_setup(self.load_config)
            assert isinstance(pipeline.model, SplatfactoModel)
            self.pipeline = pipeline
            self.model: SplatfactoModel = pipeline.model
            

            self.pipeline.datamanager.config.cache_images = "disk"
            self.pipeline.datamanager.setup_train()
        else:
            print("Pipeline and model already loaded. Skipping.")

        # Load segmentation if not present
        if not hasattr(self, "segmentation"):
            print("Loading segmentation module...")
            self.segmentation = Segmentation(
                backend=self.config.segmentation_backend,
                strategy=self.config.segmentation_strategy,
                device=self.model.device
            )
        else:
            print("Segmentation module already loaded. Skipping.")

    def _destroy_models(self):
        """Safely destroy model instances to release memory."""
        del self.model
        del self.segmentation
        del self.pipeline
        torch.cuda.empty_cache()

    def fixed_indices_dataloader(self, split="train") -> list[tuple]:
        """ Returns evaluation data in strict dataset order as a list of (camera, data) tuples.
        Works for both cached and disk modes without loading everything into memory.
        This solution disables the internal shuffling temporarily. """
        print(f"Loading {split} ordered data...")

        if split == "train":
            dataset = self.pipeline.datamanager.train_dataset
        else:
            dataset = self.pipeline.datamanager.eval_dataset

        # --- Disk mode ---
        if self.pipeline.datamanager.config.cache_images == "disk":
            # Temporarily disable internal shuffling
            original_shuffle = random.Random.shuffle
            random.Random.shuffle = lambda self, x: x  # no-op

            dataloader = DataLoader(
                getattr(self.pipeline.datamanager, f"{split}_imagebatch_stream"),
                batch_size=1,
                num_workers=0,
                collate_fn=lambda x: x[0],
            )
            items = list(islice(dataloader, len(dataset)))

            # Restore original shuffle
            random.Random.shuffle = original_shuffle
            return items

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
            return list(zip(cameras, data))
        
    #########################################################
    ############## Association of gaussians #################
    #########################################################

    def _reset_associations(self):
        self._memory_bank = []
        self.total_masks = 0

    def associate(self):
        """
        Builds a memory bank associating Gaussians across multiple views using segmentation masks.
        """

        dataloader = self.fixed_indices_dataloader(split="train")

        # Save both raw and associated masks as images
        raw_dir = self.output_dir / "masks" / "raw"
        associated_dir = self.output_dir / "masks" / "associated"
        progress_path = self.output_dir / "processed_frames.pkl"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        associated_dir.mkdir(parents=True, exist_ok=True)

        # Load already processed frames if available
        if progress_path.exists():
            with open(progress_path, "rb") as f:
                processed_frames = pickle.load(f)
        else:
            processed_frames = set()
  
        with torch.no_grad():
            for camera, data in tqdm(dataloader, desc="Processing frames", total=len(dataloader)):
                # Get image and camera index
                image = data['image']
                camera_idx = camera.metadata['cam_idx']

                # Get image info
                image_path = self.pipeline.datamanager.train_dataset.image_filenames[camera_idx]
                image_name = image_path.name

                # We require images for training the model so need to check that they exist --> need to check for undistorted images as well
                raw_image_path = raw_dir / f"{image_name}"
                associated_image_path = associated_dir / f"{image_name}"
                images_exist = raw_image_path.exists() and associated_image_path.exists()

                # Skip if already processed
                if camera_idx in processed_frames and images_exist:
                    continue

                # Get model outputs
                _ = self.model.get_outputs(camera=camera)

                # Create patch mask
                patch_mask = create_patch_mask(image)
                segmentation_results = self.segmentation.segment(image.detach().cpu().numpy())

                # Ensure objects were found
                if segmentation_results is None:
                    # Add to processed frames
                    processed_frames.add(camera_idx)
                    continue

                _, results = segmentation_results

                # Merge masks into single mask --> save out
                composite_mask = create_composite_mask(results)
                cv2.imwrite(raw_image_path, composite_mask)

                # Select front gaussians for each mask and assign labels
                mask_gaussians = self.select_front_gaussians(
                    meta=self.model.info,
                    composite_mask=composite_mask,
                    patch_mask=patch_mask
                )
                                
                labels = self._assign_labels(mask_gaussians)

                # Use the labels to convert the composite mask to show the associated labels
                associated_mask = convert_matched_mask(labels, composite_mask)
                cv2.imwrite(associated_image_path, associated_mask)

                self._update_memory_bank(labels, mask_gaussians)

                # Mark frame as processed and save progress + memory bank
                processed_frames.add(camera_idx)
                
                with open(progress_path, "wb") as f:
                    pickle.dump(processed_frames, f)
                
                self.save_memory_bank()

    def _assign_labels(self, mask_gaussians: list[torch.Tensor]) -> torch.Tensor:
        """
        Assigns a label to each mask's Gaussian set.
        If a mask doesn't sufficiently overlap with any existing label group (via IOCUR),
        it is assigned a new label, and total_masks is incremented.

        Args:
            mask_gaussians (list[Tensor]): Each tensor contains the indices of Gaussians 
            associated with a single mask in the current view.

        Returns:
            Tensor: A tensor of shape (num_masks,) containing the assigned label for each mask.
        """
        num_masks = len(mask_gaussians)

        # First view: assign each mask a unique label
        if self.total_masks == 0:
            labels = torch.arange(num_masks, dtype=torch.long)
            self.total_masks = num_masks
            return labels

        labels = torch.zeros(num_masks, dtype=torch.long)

        for i, gaussians in enumerate(mask_gaussians):
            n_gaussians = len(gaussians)
            overlaps = []

            # Compare against each label group in memory_bank
            for bank in self.memory_bank:
                union = torch.unique(torch.cat([bank, gaussians]))
                intersection = len(bank) + n_gaussians - len(union)

                # IOCUR: intersection / (intersection + current mask size)
                io_cur = intersection / (n_gaussians + intersection + 1e-8)
                overlaps.append(io_cur)

            overlaps = torch.tensor(overlaps, dtype=torch.float32)
            selected = torch.argmax(overlaps)

            # If no label matches above the threshold → assign new label
            if overlaps[selected] < self.config.iou_threshold:
                selected = self.total_masks
                self.total_masks += 1

            labels[i] = selected

        return labels
        
    def _update_memory_bank(self, labels: torch.Tensor, mask_gaussians: list[torch.Tensor]):
        """
        Updates the memory bank with newly assigned or updated Gaussians per label.

        The memory bank stores sets of Gaussian indices for each unique label. When updating,
        new labels get a new set entry, while existing labels have their sets updated with 
        the new Gaussian indices.

        Args:
            labels (torch.Tensor): Tensor of label assignments for each mask
            mask_gaussians (list[torch.Tensor]): List of tensors containing Gaussian indices 
                                                belonging to each mask

        Note:
            The memory bank (_memory_bank) is stored as a list of sets for efficient 
            uniqueness checking and updates.
        """
        for label, gaussians in zip(labels.tolist(), mask_gaussians):
            gaussians_set = set(gaussians.tolist())
            if label >= len(self._memory_bank):
                self._memory_bank.append(gaussians_set)
            else:
                self._memory_bank[label].update(gaussians_set)

    # ------------------ Saving / Loading ------------------

    def save_memory_bank(self):
        """
        Save memory bank to disk using pickle.
        """
        path = self.output_dir / "memory_bank.pkl"

        with open(path, "wb") as f:
            pickle.dump(self._memory_bank, f)

    def load_memory_bank(self):
        """
        Load memory bank from disk using pickle.
        """
        path = self.output_dir / "memory_bank.pkl"

        if not path.exists():
            print(f"Memory bank not found at {path}")
            return []

        with open(path, "rb") as f:
            bank = pickle.load(f)

        print(f"Memory bank loaded from {path}")
        return bank
    
    #########################################################
    ############## Gaussian selection #######################
    #########################################################

    def select_front_gaussians(self, meta: Dict[str, torch.Tensor], composite_mask: torch.Tensor, patch_mask: torch.Tensor, front_percentage: float = 0.5):
        """
        JIT-compiled version using torch.compile (PyTorch 2.0+).
        Maintains original structure and comments while adding compilation optimization.
        Now with separated helper functions for better code organization.
        """

        proj_results = project_gaussians(meta)
                
        # Prepare masks = Decimate the composite mask into individual masks
        binary_masks = mask_id_to_binary_mask(composite_mask)
        flattened_masks = torch.tensor(binary_masks).flatten(start_dim=1)  # (N, H*W)

        # Collect front gaussians
        front_gaussians = []
        
        for mask in flattened_masks: # total=len(flattened_masks), desc="Processing masks"): # total=len(flattened_masks), desc="Processing masks"):
            
            # Use compiled function for main processing
            result = self.process_mask_gaussians(
                proj_results, 
                mask, 
                patch_mask,
                front_percentage=front_percentage,
                debug=self.config.debug
            )
            
            front_gaussians.append(result)

        return front_gaussians

    @staticmethod
    @torch.compile(mode="max-autotune")
    def process_mask_gaussians(proj_results: Dict[str, torch.Tensor], mask: torch.Tensor, patch_mask: torch.Tensor, front_percentage: float = 0.5, debug: bool = False):
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
        patches_data = patch_intersections[non_empty_patches[:, 0], non_empty_patches[:, 1]]

        # Go through each non-empty patch and get the front gaussians
        for _, current_patch in enumerate(patches_data):
            # Projected flattened are the pixel coordinates of each gaussian --> current patch is the pixels of the mask
            # Grab gaussians in the current patch
            patch_gaussians = current_patch[proj_results['proj_flattened']].nonzero().squeeze(-1)
            
            if len(patch_gaussians) == 0:
                continue

            # Filter valid gaussians using global valid mask
            overlap_mask = proj_results['valid_mask'][patch_gaussians]

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
                patch_depths = proj_results['proj_depths'][patch_gaussians]
                _, front_indices = torch.topk(patch_depths, num_front_gaussians, largest=False)
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
    ################# Model training ########################
    #########################################################

    def train_classifier(self, dataloader=None, epochs: int = 10, use_lightning: bool = True):
        """
        Train the mask classifier with a 

        Args:
            dataloader: torch DataLoader yielding (features, labels).
                        If None, you need to provide it externally.
            epochs (int): number of epochs to train for.
            use_lightning (bool): If True, use Lightning Trainer. If False, use Fabric.
        """
        if dataloader is None:
            raise ValueError("Must provide a dataloader to train_classifier.")

            # Use Lightning Trainer
            from pytorch_lightning import Trainer
            from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
            
            # Setup callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.output_dir / "checkpoints",
                filename="grouping-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min"
            )
            
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min"
            )
            
            # Create trainer
            trainer = Trainer(
                max_epochs=epochs,
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator=self.config.accelerator if torch.cuda.is_available() else "cpu",
                devices=self.config.devices,
                precision=self.config.precision,
                log_every_n_steps=self.config.log_every_n_steps,
                val_check_interval=self.config.val_check_interval
            )
            
            # Train
            trainer.fit(self, dataloader)
    
    #########################################################
    ################# Lightning Methods #####################
    #########################################################
    
    def configure_optimizers(self):
        """Configure optimizers for Lightning training."""
        if self.classifier is None:
            self.setup_train()
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.config.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Single training step for Lightning."""
        if self.classifier is None:
            self.setup_train()
        
        feats, labels = batch
        identities = self.in_proj(feats)
        logits = self.classifier(identities)
        loss = self.loss_fn(logits, labels).mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Single validation step for Lightning."""
        if self.classifier is None:
            self.setup_train()
        
        feats, labels = batch
        identities = self.in_proj(feats)
        logits = self.classifier(identities)
        loss = self.loss_fn(logits, labels).mean()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Single test step for Lightning."""
        if self.classifier is None:
            self.setup_train()
        
        feats, labels = batch
        identities = self.in_proj(feats)
        logits = self.classifier(identities)
        loss = self.loss_fn(logits, labels).mean()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True)
        return loss
    
    def on_train_start(self):
        """Called when training starts."""
        if not hasattr(self, 'classifier') or self.classifier is None:
            self.setup_train()
    
    def on_validation_start(self):
        """Called when validation starts."""
        if not hasattr(self, 'classifier') or self.classifier is None:
            self.setup_train()
    
    def on_test_start(self):
        """Called when testing starts."""
        if not hasattr(self, 'classifier') or self.classifier is None:
            self.setup_train()
    
    def create_trainer(self, **kwargs):
        """Create a Lightning Trainer with default configuration."""
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / "checkpoints",
            filename="grouping-{epoch:02d}-{val_loss:.2f}",
            save_top_k=self.config.save_top_k,
            monitor=self.config.monitor,
            mode=self.config.mode
        )
        
        early_stop_callback = EarlyStopping(
            monitor=self.config.monitor,
            patience=self.config.patience,
            mode=self.config.mode
        )
        
        # Default trainer configuration
        trainer_config = {
            "max_epochs": self.config.max_epochs,
            "callbacks": [checkpoint_callback, early_stop_callback],
            "accelerator": self.config.accelerator if torch.cuda.is_available() else "cpu",
            "devices": self.config.devices,
            "precision": self.config.precision,
            "log_every_n_steps": self.config.log_every_n_steps,
            "val_check_interval": self.config.val_check_interval
        }
        
        # Update with any provided kwargs
        trainer_config.update(kwargs)
        
        return Trainer(**trainer_config)

# def get_n_different_colors(n: int) -> np.ndarray:
#     np.random.seed(0)
#     return np.random.randint(1, 256, (n, 3), dtype=np.uint8)

# def visualize_mask(mask: np.ndarray) -> np.ndarray:
#     color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#     num_masks = np.max(mask)
#     random_colors = get_n_different_colors(num_masks)
#     for i in range(num_masks):
#         color_mask[mask == i+1] = random_colors[i]
#     return color_mask