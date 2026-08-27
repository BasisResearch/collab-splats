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

    def visible_anchors(self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int) -> Tensor:
        """
        Boolean mask of anchors whose centre projects in front of the camera and inside the frame.

        - Deviation from upstream, which reuses the rasterizer's own prefilter pass: a projection test
          is cheaper and needs no extra rasterization. The whole-frame margin keeps anchors whose
          Gaussians spill into frame from centres just outside it.
        """
        world_to_cam = torch.linalg.inv(cam_to_world)[0]
        anchors_cam = self.params["anchors"] @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        depth = anchors_cam[:, 2]
        in_front = depth > 1e-3

        # Project with the frame's K; the margin keeps off-centre anchors that still splat into frame
        safe_depth = depth.clamp_min(1e-3)
        projected = (anchors_cam[:, :2] / safe_depth[:, None]) @ intrinsics[0, :2, :2].T + intrinsics[0, :2, 2]
        margin_x, margin_y = width * 0.5, height * 0.5
        in_frame = (
            (projected[:, 0] > -margin_x)
            & (projected[:, 0] < width + margin_x)
            & (projected[:, 1] > -margin_y)
            & (projected[:, 1] < height + margin_y)
        )
        return in_front & in_frame

    def decode(
        self,
        primitive: str,
        cam_to_world: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
        camera_id: Tensor | None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """
        Decode this view's neural Gaussians. Returns (rasterizer inputs, decode_index).

        - ``decode_index`` is ``anchor_index * n_offsets + offset_index`` per emitted Gaussian.
        - ``colors`` are post-activation RGB, so the caller rasterizes with ``sh_degree=None``.
        - ``log_scales`` carries the decoded scales in log space for the scale regulariser.

        Follows generate_neural_gaussians in city-super/Scaffold-GS scene/gaussian_model.py
        (reimplemented; no code copied).
        """
        n_offsets = self.cfg.n_offsets
        visible = self.visible_anchors(cam_to_world, intrinsics, width, height)
        anchor_ids = torch.nonzero(visible, as_tuple=False).squeeze(-1)
        anchors = self.params["anchors"][anchor_ids]
        feat = self.params["anchor_feat"][anchor_ids]
        scaling = torch.exp(self.params["scaling"][anchor_ids])
        offsets = self.params["offsets"][anchor_ids]

        # View direction and distance from each anchor to the camera centre feed every head
        camera_centre = cam_to_world[0, :3, 3]
        to_camera = anchors - camera_centre
        view_distance = to_camera.norm(dim=-1, keepdim=True)
        view_direction = to_camera / view_distance.clamp_min(1e-8)
        features = torch.cat([feat, view_direction, view_distance], dim=-1)

        neural_opacity, cov, colour = self.mlps(features, camera_id)

        # Offsets with non-positive opacity contribute nothing: dropping them here is what keeps the
        # decoded count far below anchors x n_offsets
        keep = (neural_opacity > 0).reshape(-1)
        slot_index = (anchor_ids[:, None] * n_offsets + torch.arange(n_offsets, device=anchors.device)).reshape(-1)
        decode_index = slot_index[keep]

        # means = anchor + offset scaled by the anchor's offset extent (scaling[:, :3])
        means = (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep]

        # scales = the anchor's gaussian extent (scaling[:, 3:6]) modulated per offset
        cov = cov.reshape(-1, 7)[keep]
        scales = scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[keep] * torch.sigmoid(cov[:, :3])
        quats = F.normalize(cov[:, 3:7], dim=-1)
        opacities = neural_opacity.reshape(-1)[keep]
        colors = colour.reshape(-1, 3)[keep]

        # 2DGS reads scales[..., :2]; zero the unused third channel so it can never be misread
        if primitive == "2dgs":
            zeros = torch.zeros_like(scales[:, :1])
            log_scales = torch.cat([torch.log(scales[:, :2].clamp_min(1e-12)), zeros], dim=-1)
            scales = torch.cat([scales[:, :2], zeros], dim=-1)
        else:
            log_scales = torch.log(scales.clamp_min(1e-12))

        decoded = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
            "log_scales": log_scales,
        }
        return decoded, decode_index
