"""
Utils for calculating metrics

Taken from dn-splatter
"""
import numpy as np
from scipy.spatial import cKDTree
import torch

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
