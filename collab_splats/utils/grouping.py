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
import random
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Tuple
import pickle
from pathlib import Path
import shutil

# Third party imports
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

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

# Local imports
from collab_splats.utils.segmentation import (
    Segmentation,
    create_composite_mask,
    create_patch_mask,
    convert_matched_mask,
    mask_id_to_binary_mask,
)

from collab_splats.utils import project_gaussians, create_fused_features, get_camera_parameters

@dataclass
class GroupingConfig:
    segmentation_backend: str
    segmentation_strategy: str
    front_percentage: float = 0.2
    iou_threshold: float = 0.1
    num_patches: int = 32

    # Identity parameters
    # src_feature: str = 'features_rest' # which feature to use as base for the identity
    identity_dim: int = 13
    lr: float = 5e-4

    debug: bool = False


@dataclass
class GroupingClassifier(pl.LightningModule):
    def __init__(self, load_config: str, config: GroupingConfig):
        super().__init__()

        # Config is where the model directory is --> we want one above
        self.output_dir = Path(load_config).parent.parent / "grouping"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save both raw and associated masks as images
        self.raw_mask_dir = self.output_dir / "masks" / "raw"
        self.associated_mask_dir = self.output_dir / "masks" / "associated"

        # Make directories
        self.raw_mask_dir.mkdir(parents=True, exist_ok=True)
        self.associated_mask_dir.mkdir(parents=True, exist_ok=True)

        # Save the hyperparameters (including the output directory)
        self.save_hyperparameters()

        # Set variables
        self.load_config = load_config
        self.config = config
        self.total_masks = 0

        # Load memory bank and set total masks
        self._memory_bank, self.total_masks = self.load_memory_bank()

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

        # Project the identities
        return self.params.identities

    #########################################################
    ################### Model loading #######################
    #########################################################

    def load_pipeline(self):
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

    def load_segmentation(self):
        """Load segmentation module only if not already loaded."""  
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
                composite_mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
                cv2.imwrite(str(save_path), composite_mask)
                continue

            # Create composite mask
            _, results = segmentation_results
            composite_mask = create_composite_mask(results)

            # Save the composite mask
            cv2.imwrite(str(save_path), composite_mask)
        
    #########################################################
    ############## Association of gaussians #################
    #########################################################

    def _reset_associations(self):
        self._memory_bank = []
        self.total_masks = 0

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
                composite_mask = cv2.imread(raw_image_path)

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
                _ = self.model.get_outputs(camera=camera)

                # Create patch mask
                patch_mask = create_patch_mask(image)

                # Select front gaussians for each mask and assign labels
                mask_gaussians = self.select_front_gaussians(
                    meta=self.model.info,
                    composite_mask=composite_mask,
                    patch_mask=patch_mask,
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
            print(f"Memory bank not found at {path}, initializing empty memory bank")
            return [], 0

        with open(path, "rb") as f:
            memory_bank = pickle.load(f)
        
        print (f"Memory bank loaded from {path} with {len(memory_bank)} masks")    
        return memory_bank, len(memory_bank)
    
    #########################################################
    ############## Gaussian selection #######################
    #########################################################

    def select_front_gaussians(
        self,
        meta: Dict[str, torch.Tensor],
        composite_mask: torch.Tensor,
        patch_mask: torch.Tensor,
        front_percentage: float = 0.5,
    ):
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
    ################# Model training ########################
    #########################################################

    def setup(self):
        # src_dim = self.model.gauss_params[self.config.src_feature].shape[-1]
        n_gaussians = self.model.num_points
        identities = torch.nn.Parameter(torch.zeros((n_gaussians, self.config.identity_dim)))
        identities = identities.to(self.model.device)

        self.params = torch.nn.ParameterDict(
            {
                "identities": identities,
            }
        )

        # Fit identity of gaussians by predicting masks within each view
        self.classifier = torch.nn.Conv2d(self.config.identity_dim, self.total_masks, kernel_size=1).to(self.model.device)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.config.lr)

        # Freeze the base model parameters
        if hasattr(self, 'model'):
            for param in self.model.parameters():
                param.requires_grad = False

        if hasattr(self, 'segmentation'):
            del self.segmentation

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
        model = self.model

        if not isinstance(camera, Cameras):
            print("Called _render_identities with not a camera")
            return {}

        # Prep the camera for rasterization --> get parameters
        camera_scale_fac = model._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        # Get camera parameters for rendering (K, viewmats, etc.)
        camera_params = get_camera_parameters(camera, device=model.device)

        # Get colors
        features_dc = model.features_dc
        features_rest = model.features_rest
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
        
        # We need a hack to get features into model gsplat for rendering
        # Convert the SH coefficients to RGB via gsplat
        # Found here: https://github.com/nerfstudio-project/gsplat/issues/529#issuecomment-2575128309
        if model.config.sh_degree > 0:
            sh_degree_to_use = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)
        else:
            colors = torch.sigmoid(colors).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None
        
        # Get the colors to pass into the fused features function
        fused_features = create_fused_features(
            means=model.means,
            colors=colors,
            features=self.identities, # Identities
            camera_params=camera_params,
            sh_degree_to_use=sh_degree_to_use,
        )

        # Rasterize the image using the fused features
        render, _, _, _, _, _ = self.model._render(
            means=self.model.means,
            quats=self.model.quats,
            scales=self.model.scales,
            opacities=self.model.opacities,
            colors=fused_features,
            render_mode="RGB",
            sh_degree_to_use=sh_degree_to_use,
            camera_params=camera_params,
        )

        preds = render[:, ..., 3:3 + self.config.identity_dim]

        return preds.squeeze(0)

    def forward(self, batch):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, H, W, src_dim)
            Where src_dim is the dimension of the source feature
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, self.total_masks)
        """

        camera, image, features = batch
        # outputs = self.model.get_outputs(camera=camera)

        # identities = self.in_proj(features)
        identities = self._render_identities(camera)
        identities = identities.permute(2, 0, 1) # [H, W, C] -> [C, H, W]

        # Pass through classifier and get raw prediction logits
        logits = self.classifier(identities)
        return logits

    def training_step(self, batch):
        """
        Args:
            batch (tuple): Tuple containing (features, labels)
        Returns:
            torch.Tensor: Loss tensor
        """
        feats, labels = batch
        logits = self(feats)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_params = list(self.classifier.parameters())
        return torch.optim.Adam(trainable_params, lr=self.config.lr)

    # def train_classifier(self, dataloader=None, epochs: int = 10):
    #     """
    #     Train the mask classifier with a lightweight Fabric loop.

    #     Args:
    #         dataloader: torch DataLoader yielding (features, labels).
    #                     If None, you need to provide it externally.
    #         epochs (int): number of epochs to train for.
    #     """
    #     if dataloader is None:
    #         raise ValueError("Must provide a dataloader to train_classifier.")

    #     # Initialize Fabric
    #     fabric = Fabric(accelerator="cuda", devices=1, precision="16-mixed")

    #     # Make sure classifier / proj / loss are initialized
    #     if not hasattr(self, "classifier"):
    #         self.setup_train()

    #     optimizer = self.optimizer

    #     # Fabric handles device placement + DDP wrapping
    #     model, optimizer = fabric.setup(self, optimizer)
    #     dataloader = fabric.setup_dataloaders(dataloader)

    #     # Training loop
    #     model.train()
    #     for epoch in range(epochs):
    #         for batch_idx, (feats, labels) in enumerate(dataloader):
    #             optimizer.zero_grad()

    #             identities = model.in_proj(feats)
    #             logits = model.classifier(identities)

    #             loss = model.loss_fn(logits, labels)
    #             fabric.backward(loss)
    #             optimizer.step()

    #             if batch_idx % 10 == 0:
    #                 fabric.print(f"[epoch {epoch} | batch {batch_idx}] loss={loss.item():.4f}")

    #     fabric.print("Training complete ✅")


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

#     # Convert to dict if it’s a dataclass, or copy if it’s already a dict
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
                 train_num_workers=0, val_num_workers=0):
        super().__init__()
        self.datamanager = datamanager
        self.mask_dir = Path(mask_dir)
        self.device = device
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers

    def _load_segmentation_processor(self, dataset, camera, data: Dict) -> Tuple:
        """Load and attach a segmentation mask for a given camera view."""
        # Mask filename matches the source image filename
        image_name = dataset.image_filenames[data["image_idx"]].name
        mask_path = self.mask_dir / image_name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load and normalize mask
        segmentation = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        segmentation = torch.from_numpy(segmentation).to(data["image"].device)

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