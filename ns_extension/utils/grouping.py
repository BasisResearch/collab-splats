# """
# Gaga --> gaussian grouping via multiview association + memory bank

# Steps:
# 1. Create masks --> for each view within the dataset, create masks
#     - Original implementation saves them out as images, but we could just save them out as tensors

# 2. Associate masks --> creates the memory bank?
#     - Front percentage (0.2)
#     - Overlap threshold (0.1)
#     - For each camera --> 
#         - If no masks, initialize a memory bank for the first view's masks
#         - Get gaussian idxs and zcoords (for depth grouping) for the current view
#         - Find front gaussians:
#             - Create Spatial patch mask --> divides image into patch grid
#             - Object masks --> goes through each mask in the image
#             - Combines the two masks (i.e., find overlap between patch and object mask)
#             - Find frontmost gaussians within each patch for each object
#         - Based on this:
#             - Stores the indices of the front gaussians
#             - Mask ID = tensor of ALL indices of that mask (i.e., all gaussians in that mask)
#             - Num masks == number of masks in the memory bank

# """
# import torch
# from torch import nn

# class GroupingClassifier(nn.Module):
#     def __init__(self, num_masks: int, num_gaussians: int):
#         super(GroupingClassifier, self).__init__()

#         # eval_setup(load_config)
#         self.num_masks = num_masks
#         self.num_gaussians = num_gaussians
#         self.classifier = nn.Conv2d(in_channels=num_masks, out_channels=num_gaussians, kernel_size=1)

#     def create_masks(self):
#         pass

#     def associate_masks(self):
#         pass