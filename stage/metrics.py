"""Evaluation metrics (accuracy, completeness, normal error). Not part of production path."""

import numpy as np
import torch
from scipy.spatial import cKDTree


def calculate_accuracy(reconstructed_points, reference_points, percentile=90):
    """How far away percentile% of reconstructed points are from reference."""
    tree = cKDTree(reference_points)
    distances, _ = tree.query(reconstructed_points)
    return np.percentile(distances, percentile)


def calculate_completeness(reconstructed_points, reference_points, threshold=0.05):
    """Percentage of reference points within threshold of reconstructed cloud."""
    tree = cKDTree(reconstructed_points)
    distances, _ = tree.query(reference_points)
    within_threshold = np.sum(distances < threshold) / len(distances)
    return within_threshold * 100


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean angular error between predicted and reference normals (B, C, H, W)."""
    dot_products = torch.sum(gt * pred, dim=1)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    return torch.acos(dot_products)
