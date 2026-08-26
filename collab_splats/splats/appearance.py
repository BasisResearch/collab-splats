"""
Per-image affine colour model: absorbs exposure / white-balance drift so the Gaussians need not.
"""

import torch
from torch import Tensor


class AppearanceModule(torch.nn.Module):
    """
    Learned per-image gain and bias per channel, applied to the rendered rgb.
    """

    def __init__(self, n_images: int):
        super().__init__()

        # Per image: 3 gain deltas + 3 biases, zero = identity
        self.params = torch.nn.Embedding(n_images, 6)
        torch.nn.init.zeros_(self.params.weight)

    def forward(self, rgb: Tensor, camera_ids: Tensor) -> Tensor:
        """
        rgb (B, H, W, 3), camera_ids (B,) -> rgb * (1 + gain) + bias.
        """
        params = self.params(camera_ids)
        gain = 1.0 + params[:, None, None, :3]
        bias = params[:, None, None, 3:]
        return rgb * gain + bias
