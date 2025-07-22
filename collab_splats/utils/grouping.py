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
from torch import nn

import numpy as np
import math

from collab_splats.utils.segmentation import create_patch_mask, create_composite_mask, mask_id_to_binary_mask
from collab_splats.utils.utils import project_gaussians

class GroupingClassifier(nn.Module):
    def __init__(self, num_masks: int, num_gaussians: int):
        super(GroupingClassifier, self).__init__()

        # eval_setup(load_config)
        self.num_masks = num_masks
        self.num_gaussians = num_gaussians
        self.classifier = nn.Conv2d(in_channels=num_masks, out_channels=num_gaussians, kernel_size=1)

    #########################################################
    ############## Mask initialization ######################
    #########################################################

    #########################################################
    ############## Gaussian selection #######################
    #########################################################

    def select_front_gaussians(self, model, camera, composite_mask, front_percentage: float = 0.5):
        """
        JIT-compiled version using torch.compile (PyTorch 2.0+).
        Maintains original structure and comments while adding compilation optimization.
        Now with separated helper functions for better code organization.
        """
        
        # Project gaussians onto 2d image
        proj_results = project_gaussians(model, camera)
        
        # Prepare masks = Decimate the composite mask into individual masks
        binary_masks = mask_id_to_binary_mask(composite_mask)
        flattened_masks = torch.tensor(binary_masks).flatten(start_dim=1)  # (N, H*W)

        # Compute the gaussian lookup table
        max_gaussian_id = proj_results['gaussian_ids'].max() if len(proj_results['gaussian_ids']) > 0 else 0
        valid_gaussian_mask = torch.zeros(max_gaussian_id + 1, dtype=torch.bool, device=proj_results['gaussian_ids'].device)
        valid_gaussian_mask[proj_results['gaussian_ids']] = True

        front_gaussians = []

        for mask in tqdm(flattened_masks, total=len(flattened_masks), desc="Processing masks"):
            # Use compiled function for main processing
            result = self.process_mask_gaussians(
                mask, 
                proj_results, 
                valid_gaussian_mask, 
                front_percentage=front_percentage
            )
            
            front_gaussians.append(result)

        return front_gaussians

    @torch.compile(mode="max-autotune")
    def process_mask_gaussians(self, mask, proj_results: Dict[str, torch.Tensor], valid_gaussian_mask: torch.Tensor, front_percentage: float = 0.5):
        """
        JIT-compiled function for processing a single mask.
        Optimized for performance with torch.compile.
        """
        # Find intersection between object mask and patch masks
        patch_intersections = mask.unsqueeze(0).unsqueeze(0) & self.patch_mask

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

            # Filter valid gaussians using pre-computed mask
            overlap_mask = valid_gaussian_mask[patch_gaussians]

            if not overlap_mask.all():
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

    def associate_masks(self):
        pass

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

