"""
Utils for calculating metrics

Taken from dn-splatter
"""
import numpy as np
from scipy.spatial import cKDTree
import torch

def project_gaussians(meta: dict):
    """
    Project gaussians onto 2D image and prepare lookup data.

    meta = output from fully_fused_projection or rasterization
    """

    W, H = meta["width"], meta["height"]

    # gaussians where the radius is greater than 1.0 can be seen in the camera frustum
    radii = meta['radii'].squeeze()
    gaussian_ids = torch.where(torch.sum(radii > 1.0, axis=1))[0]

    # Convert 2D coords to flat pixel indices
    xy_rounded = torch.round(meta['means2d']).squeeze().long()
    x = torch.clamp(xy_rounded[:, 0], 0, W)
    y = torch.clamp(xy_rounded[:, 1], 0, H)
    projected_flattened = x + y * W                      # (M,)

    return {
        "proj_flattened": projected_flattened.squeeze().detach().cpu(),                      # (M,)
        "proj_depths": meta['depths'].squeeze().detach().cpu(),                                      # (M,)
        "gaussian_ids": gaussian_ids.squeeze().detach().cpu(),                 # (M,)
    }

def calculate_accuracy(reconstructed_points, reference_points, percentile=90):
    """
    Calculate accuracy: How far away 90% of the reconstructed point clouds are from the reference point cloud.
    """
    tree = cKDTree(reference_points)
    distances, _ = tree.query(reconstructed_points)
    return np.percentile(distances, percentile)


def calculate_completeness(reconstructed_points, reference_points, threshold=0.05):
    """
    Calculate completeness: What percentage of the reference point cloud is within
    a specific distance of the reconstructed point cloud.
    """
    tree = cKDTree(reconstructed_points)
    distances, _ = tree.query(reference_points)
    within_threshold = np.sum(distances < threshold) / len(distances)
    return within_threshold * 100


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute the mean angular error between predicted and reference normals

    Args:
        predicted_normals: [B, C, H, W] tensor of predicted normals
        reference_normals : [B, C, H, W] tensor of gt normals

    Returns:
        mae: [B, H, W] mean angular error
    """
    # Dot product of predicted and reference normals
    dot_products = torch.sum(gt * pred, dim=1)  # over the C dimension
    
    # Clamp the dot product to ensure valid cosine values (to avoid nans)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)

    # Calculate the angle between the vectors (in radians)
    mae = torch.acos(dot_products)
    return mae
