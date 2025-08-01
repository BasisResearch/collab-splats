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

"""

import torch
from dataclasses import dataclass
from typing import Dict
from torch import nn
from tqdm.notebook import tqdm

from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup

from collab_splats.utils.segmentation import Segmentation, create_patch_mask, create_composite_mask, mask_id_to_binary_mask
from collab_splats.utils.utils import project_gaussians

@dataclass
class GroupingParams:
    segmentation_backend: str
    segmentation_strategy: str
    front_percentage: float = 0.2
    iou_threshold: float = 0.1
    num_patches: int = 32

    # Identity parameters
    src_feature: str = 'features_rest' # which feature to use as base for the identity
    identity_dim: int = 16
    lr: float = 5e-4

    debug: bool = False

@dataclass
class GroupingClassifier(nn.Module):
    def __init__(self, load_config: str, params: GroupingParams):
        super().__init__()

        self.load_config = load_config
        self.params = params
        self.total_masks = 0

        # eval_setup(load_config)
        # self.num_masks = num_masks
        # self.num_gaussians = num_gaussians
        # self.classifier = nn.Conv2d(in_channels=num_masks, out_channels=num_gaussians, kernel_size=1)
        self._memory_bank = []

        # Load pipeline and segmentation
        self._load_models()
        

        self.params

    @property
    def memory_bank(self):
        return self._memory_bank

    @property
    def identities(self):
        return self.params.identities

    #########################################################
    ################### Model loading #######################
    #########################################################

    def setup(self):
        self.memory_bank = []
        self._load_models()
    
    def setup_train(self):
        n_gaussians = self.model.num_points
        src_dim = self.model.info[self.src_feature].shape[-1]
        identities = torch.nn.Parameter(n_gaussians, self.identity_dim)

        self.params = torch.nn.ParameterDict(
            {
                "identities": identities,
            }
        )

        # In projection is simple linear projection followed by nonlinearity
        # We piggyback off an existing feature and use that as a method to create identities
        self.in_proj = nn.Sequential(
            nn.Linear(src_dim, self.identity_dim),
            nn.ReLU(),
        )

        # Fit identity of gaussians by predicting masks within each view
        self.classifier = torch.nn.Conv2d(self.identity_dim, self.total_masks, kernel_size=1)

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
                backend=self.params.segmentation_backend,
                strategy=self.params.segmentation_strategy,
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

    #########################################################
    ############## Association of gaussians #################
    #########################################################

    def associate(self):
        """
        Builds a memory bank associating Gaussians across multiple views using segmentation masks.
        """
        
        with torch.no_grad():
            # cameras: Cameras = self.pipeline.datamanager.train_dataset.cameras

            for camera, data in tqdm(
                self.pipeline.datamanager.train_imagebatch_stream, # Need to use cached_train since it undistorts the images
                desc="Processing frames",
                total=len(self.pipeline.datamanager.train_dataset)
            ):
                image = data['image']

                _ = self.model.get_outputs(camera=camera)

                patch_mask = create_patch_mask(image)
                _, results = self.segmentation.segment(image.detach().cpu().numpy())
                composite_mask = create_composite_mask(results)

                mask_gaussians = self.select_front_gaussians(
                    meta=self.model.info,
                    composite_mask=composite_mask,
                    patch_mask=patch_mask
                )

                labels = self._assign_labels(mask_gaussians)
                self._update_memory_bank(labels, mask_gaussians)

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
            if overlaps[selected] < self.params.iou_threshold:
                selected = self.total_masks
                self.total_masks += 1

            labels[i] = selected

        return labels
    
    def _update_memory_bank(self, labels: torch.Tensor, mask_gaussians: list[torch.Tensor]):
        """
        Updates the memory_bank with newly assigned or updated Gaussians per label.

        Assumes self.total_masks is up-to-date and labels are valid.

        Args:
            labels (Tensor): Assigned labels for each mask.
            mask_gaussians (list[Tensor]): Gaussian indices per mask.
        """
        for label, gaussians in zip(labels.tolist(), mask_gaussians):
            if label >= len(self.memory_bank):
                # New label → initialize memory
                self.memory_bank.append(gaussians)
            else:
                # Existing label → merge and deduplicate
                combined = torch.cat([self.memory_bank[label], gaussians])
                self.memory_bank[label] = torch.unique(combined)

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

        for mask in tqdm(flattened_masks, total=len(flattened_masks), desc="Processing masks"):

            # Use compiled function for main processing
            result = self.process_mask_gaussians(
                proj_results, 
                mask, 
                patch_mask,
                front_percentage=front_percentage
            )
            
            front_gaussians.append(result)

        return front_gaussians

    @torch.compile(mode="max-autotune")
    def process_mask_gaussians(self,  proj_results: Dict[str, torch.Tensor], mask: torch.Tensor, patch_mask: torch.Tensor, front_percentage: float = 0.5):
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
        for patch_idx, current_patch in enumerate(patches_data):
            # Projected flattened are the pixel coordinates of each gaussian --> current patch is the pixels of the mask
            # Grab gaussians in the current patch
            patch_gaussians = current_patch[proj_results['proj_flattened']].nonzero().squeeze(-1)
            
            if len(patch_gaussians) == 0:
                continue

            # Filter valid gaussians using global valid mask
            overlap_mask = proj_results['valid_mask'][patch_gaussians]

            if not overlap_mask.all() and self.params.debug:
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

