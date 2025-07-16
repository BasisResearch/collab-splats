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

from ns_extension.utils.utils import project_gaussians

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

    def set_patch_mask(self, image, num_patches: int = 32):
        """
        Provided an image of given dimensions, create an array of patches.
        """
        # Get image dimensions
        H, W = image.shape[:2]

        # Get patch dimensions
        patch_width = math.ceil(W / num_patches)
        patch_height = math.ceil(H / num_patches)
        
        # Create flattened coordinates
        total_pixels = H * W
        y_coords = torch.arange(H).unsqueeze(1).expand(-1, W).flatten()
        x_coords = torch.arange(W).unsqueeze(0).expand(H, -1).flatten()
        
        # Calculate patch indices for all pixels at once
        patch_y_indices = torch.clamp(y_coords // patch_height, 0, num_patches - 1)
        patch_x_indices = torch.clamp(x_coords // patch_width, 0, num_patches - 1)
        
        # Create sparse representation
        flatten_patch_mask = torch.zeros((num_patches, num_patches, total_pixels), 
                                    dtype=torch.bool)
        
        # Use indexing to set values
        pixel_indices = torch.arange(total_pixels)
        flatten_patch_mask[patch_y_indices, patch_x_indices, pixel_indices] = True
        
        return flatten_patch_mask
    
    def create_composite_mask(self, results, confidence_threshold=0.85):
        """
        Creates a composite mask from the results of the segmentation model.
        
        Inputs:
            results: list of dicts, each containing a mask and a confidence score
            confidence_threshold: float, the minimum confidence score for a mask to be included in the composite mask

        Outputs:
            composite_mask: numpy array, the composite mask
        """

        selected_masks = []
        for mask in results:
            if mask['predicted_iou'] < confidence_threshold:
                continue

            selected_masks.append(
                (mask['segmentation'], mask['predicted_iou'])
            )
        
        # Store the masks and confidences
        masks, confs = zip(*selected_masks)

        # Create empty image to store mask ids
        H, W = masks[0].shape[:2]
        mask_id = np.zeros((H, W), dtype=np.uint8)

        sorted_idxs = np.argsort(confs)
        for i, idx in enumerate(sorted_idxs, start=1):
            current_mask = masks[idx - 1]
            mask_id[current_mask == 1] = i

        # Find mask indices after having calculated overlap based on ranked confidence
        mask_indices = np.unique(mask_id)
        mask_indices = np.setdiff1d(mask_indices, [0]) # remove 0 item

        composite_mask = np.zeros((H, W), dtype=np.uint8)

        for i, idx in enumerate(mask_indices, start=1):
            mask = (mask_id == idx)
            if mask.sum() > 0 and (mask.sum() / masks[idx-1].sum()) > 0.1:
                composite_mask[mask] = i

        return composite_mask

    def mask_id_to_binary_mask(self, composite_mask: np.ndarray) -> np.ndarray:
        """
        Convert an image with integer mask IDs to a binary mask array.

        Args:
            mask_id (np.ndarray): An (H, W) array where each unique positive integer 
                                represents a separate object mask.

        Returns:
            np.ndarray: A (N, H, W) boolean array where N is the number of masks and each 
                        slice contains a binary mask.
        """
        unique_ids = np.unique(composite_mask)
        unique_ids = unique_ids[unique_ids > 0]  # Ignore background (assumed to be 0)

        binary_masks = (composite_mask[None, ...] == unique_ids[:, None, None])
        return binary_masks

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
        binary_masks = self.mask_id_to_binary_mask(composite_mask)
        flattened_masks = torch.tensor(binary_masks).flatten(start_dim=1)  # (N, H*W)

        # Pre-extract proj_results for compiled function
        proj_flattened = proj_results['proj_flattened']
        proj_depths = proj_results['proj_depths']

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

