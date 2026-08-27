"""
Scaffold-GS anchors on gsplat: anchor parameters, MLP decode heads, and anchor densification.

Reimplemented from the Scaffold-GS paper (Lu et al., CVPR 2024). The reference implementation
(city-super/Scaffold-GS) is under the Inria/MPII Gaussian-Splatting licence and is NOT vendored —
each ported concept cites its upstream site for provenance only.

Anchors carry a feature vector; three MLP heads decode ``n_offsets`` neural Gaussians per anchor per
view, which then go through the same gsplat rasterizer the vanilla representation uses. Densification
operates on anchors (voxel growing / opacity pruning), not on the decoded Gaussians.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import _update_param_with_optimizer
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

logger = logging.getLogger(__name__)

########################################
# Config
########################################


@dataclass
class ScaffoldConfig:
    """
    Anchor-representation knobs; mirrors the ``splats.scaffold:`` yaml block.
    """

    n_offsets: int = 10
    feat_dim: int = 32

    # Anchor voxel size = median kNN spacing of the seed points x voxel_multiplier, so it is
    # scale-free under normalize_scene either way. voxel_size overrides it in world units.
    voxel_multiplier: float = 1.0
    voxel_size: float | None = None

    # Densification window and thresholds. grad_threshold is Scaffold's published value and is
    # directly comparable because AnchorStrategy renormalises gradients the way gsplat's
    # DefaultStrategy does (strategy/default.py:243-249).
    update_from: int = 1500
    update_until: int = 15000
    refine_every: int = 100
    grad_threshold: float = 2e-4
    min_opacity: float = 0.005
    update_depth: int = 3  # coarse-to-fine growing levels
    update_hierarchy_factor: int = 4

    # Scaffold's own per-image appearance embedding, concatenated into the colour MLP input.
    # 0 disables it. Independent of splats.appearance_opt (our per-image affine module): both
    # ship, neither retires the other, and the 2x2 is measured.
    appearance_dim: int = 0

    # Learning rates (anchor / offset lrs are multiplied by scene_scale, like means_lr)
    anchor_lr: float = 1.6e-4
    offset_lr: float = 1e-2
    anchor_feat_lr: float = 7.5e-3
    scaling_lr: float = 7e-3
    rotation_lr: float = 2e-3
    mlp_lr: float = 2e-3

    @classmethod
    def from_dict(cls, block: dict) -> "ScaffoldConfig":
        """
        Build from the yaml block; rejects unknown keys (they are almost always typos).
        """
        unknown = set(block) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"splats.scaffold: unknown keys {sorted(unknown)}; allowed {sorted(cls.__dataclass_fields__)}"
            )
        cfg = cls(**block)

        # Shapes the whole decode depends on: a zero here fails much later and much less clearly
        if cfg.n_offsets < 1:
            raise ValueError(f"splats.scaffold.n_offsets must be >= 1, got {cfg.n_offsets}")
        if cfg.feat_dim < 1:
            raise ValueError(f"splats.scaffold.feat_dim must be >= 1, got {cfg.feat_dim}")
        if cfg.voxel_size is not None and cfg.voxel_size <= 0:
            raise ValueError(f"splats.scaffold.voxel_size must be > 0 when set, got {cfg.voxel_size}")
        if cfg.voxel_multiplier <= 0:
            raise ValueError(f"splats.scaffold.voxel_multiplier must be > 0, got {cfg.voxel_multiplier}")
        return cfg


########################################
# MLP heads
########################################

VIEW_DIM = 4  # unit view direction (3) + view distance (1)


class ScaffoldMLPs(torch.nn.Module):
    """
    The three Scaffold-GS decode heads: opacity, covariance, colour.

    - Input is [anchor_feat, view_dir, view_dist] per visible anchor; every head emits one row of
      ``n_offsets`` outputs per anchor.
    - opacity ends in tanh (its sign is the offset visibility mask), colour in sigmoid (RGB),
      covariance is raw (3 scale factors + 4 quaternion components per offset).
    - ``appearance_dim > 0`` adds Scaffold's per-image embedding to the colour head input only.

    Head shapes follow city-super/Scaffold-GS scene/gaussian_model.py (MLP definitions); no code copied.
    """

    def __init__(self, cfg: ScaffoldConfig, n_views: int = 0):
        super().__init__()
        self.n_offsets = cfg.n_offsets
        width = cfg.feat_dim
        base_dim = cfg.feat_dim + VIEW_DIM

        self.mlp_opacity = torch.nn.Sequential(
            torch.nn.Linear(base_dim, width),
            torch.nn.ReLU(True),
            torch.nn.Linear(width, cfg.n_offsets),
            torch.nn.Tanh(),
        )
        self.mlp_cov = torch.nn.Sequential(
            torch.nn.Linear(base_dim, width),
            torch.nn.ReLU(True),
            torch.nn.Linear(width, 7 * cfg.n_offsets),
        )

        # Scaffold's appearance embedding rides the colour head only
        self.embedding_appearance = None
        colour_dim = base_dim
        if cfg.appearance_dim > 0:
            if n_views < 1:
                raise ValueError("scaffold.appearance_dim > 0 needs n_views >= 1 to size the embedding")
            self.embedding_appearance = torch.nn.Embedding(n_views, cfg.appearance_dim)
            torch.nn.init.zeros_(self.embedding_appearance.weight)
            colour_dim += cfg.appearance_dim
        self.mlp_colour = torch.nn.Sequential(
            torch.nn.Linear(colour_dim, width),
            torch.nn.ReLU(True),
            torch.nn.Linear(width, 3 * cfg.n_offsets),
            torch.nn.Sigmoid(),
        )

    def forward(self, features: Tensor, camera_id: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Decode (opacity, covariance, colour) for every visible anchor. Shapes [A, K], [A, 7K], [A, 3K].
        """
        opacity = self.mlp_opacity(features)
        cov = self.mlp_cov(features)

        # The embedding is per-image, so the colour head needs to know which view is being rendered
        colour_input = features
        if self.embedding_appearance is not None:
            if camera_id is None:
                raise ValueError("scaffold.appearance_dim > 0 requires camera_id at decode time")
            embedding = self.embedding_appearance(camera_id[:1]).expand(len(features), -1)
            colour_input = torch.cat([features, embedding], dim=-1)
        colour = self.mlp_colour(colour_input)
        return opacity, cov, colour
