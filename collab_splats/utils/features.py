"""
Backward-compatibility shim.

All feature extraction code now lives in collab_splats.semantics.features.
This module re-exports everything so existing callers require no changes.

TwoLayerMLP stays here — it is a splatting decoder layer, not a feature extractor.
"""

from collab_splats.semantics.features import (
    BaseFeatureExtractor,
    MaskCLIPExtractor,
    DINOFeatureExtractor,
    Talk2DinoExtractor,
    load_hf_weights,
    load_torchhub_model,
    pytorch_gc,
    resize_image,
    interpolate_to_patch_size,
    batch_iterator,
    TORCH_HOME,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class TwoLayerMLP(nn.Module):
    """
    A two-layer MLP implemented using 1x1 convolutions for reconstructing feature maps.
    The network consists of:
    - A shared hidden 1x1 convolution layer (acts as the intermediate representation).
    - A set of task-specific output branches, each also a 1x1 convolution, producing different feature maps.

    Attributes:
        hidden_conv (nn.Conv2d): Shared hidden layer.
        feature_branch_dict (nn.ModuleDict): Dictionary of output branches, one for each model type.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        features_dim_dict: Dict[str, Tuple[int, int, int]],
    ):
        """
        Args:
            input_dim (int): Number of input channels.
            hidden_dim (int): Number of channels in the intermediate hidden layer.
            feature_dim_dict (dict): Dictionary mapping feature names to output shapes (C, H, W).
                                     Only the channel dimension (C) is used here.
        """
        super().__init__()
        self.hidden_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        self.feature_branch_dict = nn.ModuleDict(
            {
                model: nn.Conv2d(hidden_dim, feat_shape[0], kernel_size=1)
                for model, feat_shape in features_dim_dict.items()
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass using 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)

        Returns:
            dict: Dictionary mapping feature names to output tensors of shape (B, C_out, H, W)
        """
        x = F.relu(self.hidden_conv(x))
        return {model: conv(x) for model, conv in self.feature_branch_dict.items()}

    @torch.no_grad()
    def per_gaussian_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass using fully-connected (linear) layers assuming `x` is a flattened per-Gaussian input.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in), where N is number of Gaussians.

        Returns:
            outputs: Dictionary mapping feature names to output tensors of shape (N, C_out)
        """
        w_hidden = self.hidden_conv.weight.view(self.hidden_conv.out_channels, -1)
        x = F.relu(F.linear(x, w_hidden, self.hidden_conv.bias))

        outputs = {}
        for model, conv in self.feature_branch_dict.items():
            w_out = conv.weight.view(conv.out_channels, -1)
            outputs[model] = F.linear(x, w_out, conv.bias)

        return outputs


__all__ = [
    "BaseFeatureExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    "TwoLayerMLP",
    "load_hf_weights",
    "load_torchhub_model",
    "pytorch_gc",
    "resize_image",
    "interpolate_to_patch_size",
    "batch_iterator",
    "TORCH_HOME",
]
