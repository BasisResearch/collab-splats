"""
Utils for calculating metrics

Taken from dn-splatter
"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from typing import Dict

def project_gaussians(meta: dict) -> Dict[str, torch.Tensor]:
    """
    Projects Gaussians into 2D image space and prepares full-resolution lookup tensors.
    Returns full-length arrays indexed by global Gaussian ID, and a list of visible IDs.
    """

    W, H = meta["width"], meta["height"]
    N = meta["radii"].shape[0]

    # Visibility based on Gaussian radius threshold
    radii = meta['radii'].squeeze()  # shape (N, 2) or (N, D)
    valid_mask = (radii > 1.0).sum(dim=1) > 0  # shape (N,)
    gaussian_ids = valid_mask.nonzero(as_tuple=False).squeeze()  # global indices of visible Gaussians

    # Compute flat image coordinates for all Gaussians
    xy_rounded = torch.round(meta['means2d']).squeeze().long()  # shape (N, 2)
    x = torch.clamp(xy_rounded[:, 0], 0, W - 1)
    y = torch.clamp(xy_rounded[:, 1], 0, H - 1)
    projected_flattened = x + y * W  # shape (N,)

    return {
        "proj_flattened": projected_flattened.detach().cpu(),     # (N,)
        "proj_depths": meta['depths'].squeeze().detach().cpu(),   # (N,)
        "valid_mask": valid_mask.detach().cpu(),                  # (N,)
        "gaussian_ids": gaussian_ids.detach().cpu(),              # (M,) global indices of valid Gaussians
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
