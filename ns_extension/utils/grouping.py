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

class GroupingClassifier(nn.Module):
    def __init__(self, num_masks: int, num_gaussians: int):
        super(GroupingClassifier, self).__init__()

        # eval_setup(load_config)
        self.num_masks = num_masks
        self.num_gaussians = num_gaussians
        self.classifier = nn.Conv2d(in_channels=num_masks, out_channels=num_gaussians, kernel_size=1)

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

    def mask_id_to_binary_mask(composite_mask: np.ndarray) -> np.ndarray:
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