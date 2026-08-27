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


########################################
# Anchor field
########################################


def median_knn_spacing(points: Tensor, k: int = 3) -> float:
    """
    Median mean-distance to the k nearest neighbours — the seed points' own length scale.

    - Same kNN spacing ``init_gaussians_from_points`` gives each vanilla Gaussian, reduced to one
      number: anchors need a single grid size, not a per-point scale.
    """
    array = points.detach().cpu().numpy()
    neighbour_dists, _ = NearestNeighbors(n_neighbors=k + 1).fit(array).kneighbors(array)
    return float(np.median(np.sqrt((neighbour_dists[:, 1:] ** 2).mean(-1))))


def voxelize(points: Tensor, voxel_size: float) -> Tensor:
    """
    One representative point per occupied voxel: round to the grid, dedup, return grid centres.
    """
    grid_coords = torch.round(points / voxel_size)
    unique_coords = torch.unique(grid_coords, dim=0)
    return unique_coords * voxel_size


class AnchorField:
    """
    Scaffold-GS anchors: the trainable state plus the per-view decode into neural Gaussians.

    - ``params`` is the ParameterDict the densification strategy grows and prunes; the MLP heads live
      in ``mlps`` and are fixed-size, so they are optimised separately and never handed to the strategy.
    - ``decode`` returns the rasterizer inputs plus ``decode_index``, the (anchor * n_offsets + offset)
      slot each emitted Gaussian came from. Anchor densification is built entirely on that index.
    - There is no anchor opacity parameter: opacity is decoded per view by mlp_opacity.
    """

    def __init__(
        self,
        cfg: ScaffoldConfig,
        points: np.ndarray,
        colors: np.ndarray,
        scene_scale: float,
        n_views: int,
        device: str,
    ):
        self.cfg = cfg
        self.device = device

        # Voxel size from the seed points' own spacing: scale-free, so normalize_scene cannot silently
        # change anchor density (Scaffold's absolute 0.001 default assumes an already-normalised scene)
        points_t = torch.from_numpy(np.asarray(points, dtype=np.float32))
        if cfg.voxel_size is not None:
            self.voxel_size = float(cfg.voxel_size)
        else:
            self.voxel_size = float(median_knn_spacing(points_t) * cfg.voxel_multiplier)

        anchors = voxelize(points_t, self.voxel_size).to(device)
        n_anchors = len(anchors)
        logger.info(
            "scaffold: %d seed points -> %d anchors at voxel_size %.6g", len(points_t), n_anchors, self.voxel_size
        )

        # Offsets start at zero (upstream seeds new anchors with zero offsets); scaling holds the offset
        # extent (first 3) and the decoded-Gaussian extent (last 3), both in log space
        log_voxel = math.log(self.voxel_size)
        self.params = torch.nn.ParameterDict(
            {
                "anchors": torch.nn.Parameter(anchors),
                "offsets": torch.nn.Parameter(torch.zeros(n_anchors, cfg.n_offsets, 3, device=device)),
                "anchor_feat": torch.nn.Parameter(torch.zeros(n_anchors, cfg.feat_dim, device=device)),
                "scaling": torch.nn.Parameter(torch.full((n_anchors, 6), log_voxel, device=device)),
                "rotation": torch.nn.Parameter(
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_anchors, 1)
                ),
            }
        )

        self.mlps = ScaffoldMLPs(cfg, n_views=n_views).to(device)

        # One Adam per anchor tensor so the strategy can grow/prune optimizer state per tensor
        learning_rates = {
            "anchors": cfg.anchor_lr * scene_scale,
            "offsets": cfg.offset_lr * scene_scale,
            "anchor_feat": cfg.anchor_feat_lr,
            "scaling": cfg.scaling_lr,
            "rotation": cfg.rotation_lr,
        }
        self.optimizers = {
            name: torch.optim.Adam([{"params": self.params[name], "lr": lr, "name": name}], eps=1e-15)
            for name, lr in learning_rates.items()
        }
        self.mlp_optimizer = torch.optim.Adam(self.mlps.parameters(), lr=cfg.mlp_lr, eps=1e-15)
