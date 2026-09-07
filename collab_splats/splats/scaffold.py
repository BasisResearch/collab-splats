"""
Scaffold-GS anchors on gsplat: anchor parameters, MLP decode heads, and anchor densification.

- anchors carry a feature vector; three MLP heads decode ``n_offsets`` neural Gaussians per
  anchor per view
- rasterizer: the same gsplat one the vanilla representation uses
- densification: operates on anchors (voxel growing / opacity pruning), not on the decoded
  Gaussians
- reimplemented from Scaffold-GS (Lu et al., CVPR 2024)
- city-super/Scaffold-GS @ 59c833b5: license-encumbered and NOT vendored, so sites are cited for
  provenance only
- sites reading `GS-SR`: yanxian-ll/GS-SR @ 566359be, the reimplementation followed for
  densification details
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import fully_fused_projection
from gsplat.strategy.ops import _update_param_with_optimizer
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from collab_splats.splats.gaussian import SH_C0
from collab_splats.splats.rendering import render_gaussians

# Type-only: trainer imports this module, so a runtime import back would be circular
if TYPE_CHECKING:
    from collab_splats.splats.trainer import SplatsConfig

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

    # Anchor voxel size, derived from the seed spacing
    # - median kNN spacing x voxel_multiplier: scale-free under normalize_scene either way
    # - voxel_size: overrides it, in world units
    voxel_multiplier: float = 1.0
    voxel_size: float | None = None

    # Densification window and thresholds
    # - grad_threshold: Scaffold's published value, usable as-is
    # - gradients renormalized like gsplat's DefaultStrategy
    #   (nerfstudio-project/gsplat @ d2f5c0f, gsplat/strategy/default.py:243-249)
    start_stat: int = 500  # statistics start here, before growing does
    update_from: int = 1500
    update_until: int = 15000
    refine_every: int = 100
    grad_threshold: float = 2e-4
    min_opacity: float = 0.005
    update_depth: int = 3  # coarse-to-fine growing levels
    update_hierarchy_factor: int = 4
    update_init_factor: int = 16  # level 0 grows on a grid this many voxels across, level i shrinks it
    success_threshold: float = 0.8  # fraction of the window a slot/anchor must be seen for to count

    # Per-image appearance embedding, color head only
    # - 0: disables it; 32 is upstream's default
    #   (yanxian-ll/GS-SR @ 566359be, gssr/gaussian/scaffold_gaussian.py:41)
    # - independent of splats.appearance_opt, our per-image affine module: both ship
    appearance_dim: int = 32

    # Learning rates
    # - anchor / offset lrs: multiplied by scene_scale, like means_lr
    # - anchors frozen upstream (position_lr_init = position_lr_final = 0): the growing dedup's grid
    #   only stays a grid if the anchors on it do not drift
    anchor_lr: float = 0.0
    offset_lr: float = 1e-2
    offset_lr_final: float = 1e-4
    anchor_feat_lr: float = 7.5e-3
    scaling_lr: float = 7e-3
    rotation_lr: float = 2e-3

    # One lr per head, each with its own exponential decay, as upstream ships them
    mlp_opacity_lr: float = 2e-3
    mlp_opacity_lr_final: float = 2e-5
    mlp_cov_lr: float = 4e-3
    mlp_color_lr: float = 8e-3
    mlp_color_lr_final: float = 5e-5
    appearance_lr: float = 5e-2
    appearance_lr_final: float = 5e-4

    # Schedules key to a FIXED horizon, not the run length
    # - *_lr_max_steps = 30_000 (GS-SR @ 566359be, gssr/gaussian/scaffold_gaussian.py:35-91)
    # - a 12k run stops partway down
    lr_max_steps: int = 30000

    @classmethod
    def from_dict(cls, block: dict) -> "ScaffoldConfig":
        """
        Build from the yaml block; rejects unknown keys (they are almost always typos).

        Args:
            block: the parsed ``splats.scaffold:`` mapping.

        Returns:
            A validated ScaffoldConfig.
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
        if cfg.lr_max_steps < 1:
            raise ValueError(f"splats.scaffold.lr_max_steps must be >= 1, got {cfg.lr_max_steps}")
        if cfg.voxel_multiplier <= 0:
            raise ValueError(f"splats.scaffold.voxel_multiplier must be > 0, got {cfg.voxel_multiplier}")
        return cfg


########################################
# MLP heads
########################################


class ScaffoldMLPs(torch.nn.Module):
    """
    The three Scaffold-GS decode heads: opacity, covariance, color.

    - Input: [anchor_feat, view_dir] per visible anchor; each head emits ``n_offsets`` per anchor.
    - opacity: tanh, its sign the offset visibility mask. color: sigmoid RGB. covariance: raw,
      3 scale factors + 4 quaternion components per offset.
    - ``appearance_dim > 0``: adds Scaffold's per-image embedding, color head input only.
    - head shapes: city-super/Scaffold-GS @ 59c833b5, scene/gaussian_model.py:107-128
    - no code copied
    """

    def __init__(self, cfg: ScaffoldConfig, n_views: int = 0):
        """
        Build the three heads at the widths `cfg` implies, plus the appearance embedding.

        Args:
            cfg: supplies `feat_dim` (head width and anchor feature size), `n_offsets` and
                `appearance_dim`.
            n_views: sizes the per-image appearance embedding; 0 leaves it out, as does
                `appearance_dim == 0`.
        """
        super().__init__()
        self.n_offsets = cfg.n_offsets
        width = cfg.feat_dim

        # Every head sees [anchor_feat, unit view direction]
        # - add_opacity_dist / add_cov_dist / add_color_dist all ship off
        #   (city-super/Scaffold-GS @ 59c833b5, arguments/__init__.py:75-77;
        #   gaussian_renderer/__init__.py:50 takes cat_local_view_wodist)
        # - unit direction: keeps the heads scale-free; distance is a world-unit quantity
        base_dim = cfg.feat_dim + 3

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

        # Scaffold's appearance embedding rides the color head only
        self.embedding_appearance = None
        color_dim = base_dim
        if cfg.appearance_dim > 0:
            if n_views < 1:
                raise ValueError("scaffold.appearance_dim > 0 needs n_views >= 1 to size the embedding")
            self.embedding_appearance = torch.nn.Embedding(n_views, cfg.appearance_dim)
            color_dim += cfg.appearance_dim
        self.mlp_color = torch.nn.Sequential(
            torch.nn.Linear(color_dim, width),
            torch.nn.ReLU(True),
            torch.nn.Linear(width, 3 * cfg.n_offsets),
            torch.nn.Sigmoid(),
        )

    def forward(self, features: Tensor, camera_id: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Decode (opacity, covariance, color) for every visible anchor.

        Args:
            features: (A, feat_dim + 3) anchor features concatenated with the unit view direction.
            camera_id: (1,) view index for the appearance embedding; required when it is on.

        Returns:
            (opacity [A, K], covariance [A, 7K], color [A, 3K]).
        """
        opacity = self.mlp_opacity(features)
        cov = self.mlp_cov(features)

        # The embedding is per-image, so the color head needs to know which view is being rendered
        color_input = features
        if self.embedding_appearance is not None:
            if camera_id is None:
                raise ValueError("scaffold.appearance_dim > 0 requires camera_id at decode time")
            embedding = self.embedding_appearance(camera_id[:1]).expand(len(features), -1)
            color_input = torch.cat([features, embedding], dim=-1)
        color = self.mlp_color(color_input)
        return opacity, cov, color


########################################
# Scaffold
########################################


def _decay_lambda(lr_init: float, lr_final: float, lr_max_steps: int):
    """
    LambdaLR multiplier reproducing 3DGS's log-space lr decay from lr_init to lr_final.

    - LambdaLR scales the optimizer's *initial* lr, so the multiplier is the ratio raised to the
      progress fraction — upstream's ``get_expon_lr_func`` curve (its delay term is inert there).
    - Non-positive endpoints hold the lr flat, which the anchor lr's 0.0 default needs.

    Args:
        lr_init: lr at step 0, already scaled by the caller.
        lr_final: lr at `lr_max_steps` and beyond.
        lr_max_steps: schedule horizon in steps; upstream fixes it at 30k.

    Returns:
        A `step -> multiplier` callable for `torch.optim.lr_scheduler.LambdaLR`.
    """
    if lr_init <= 0.0 or lr_final <= 0.0:
        return lambda step: 1.0
    ratio = lr_final / lr_init
    horizon = max(lr_max_steps, 1)
    return lambda step: ratio ** min(step / horizon, 1.0)


def median_knn_spacing(points: Tensor, k: int = 3) -> float:
    """
    Median mean-distance to the k nearest neighbors — the seed points' own length scale.

    Args:
        points: (N, 3) seed positions.
        k: neighbors per point, excluding the point itself.

    Returns:
        One length scale for the whole cloud; anchors need a grid size, not a per-point scale.
    """
    array = points.detach().cpu().numpy()
    neighbor_dists, _ = NearestNeighbors(n_neighbors=k + 1).fit(array).kneighbors(array)
    return float(np.median(np.sqrt((neighbor_dists[:, 1:] ** 2).mean(-1))))


def voxelize(points: Tensor, voxel_size: float) -> Tensor:
    """
    One representative point per occupied voxel: round to the grid, dedup, return grid centers.

    Args:
        points: (N, 3) positions to quantize.
        voxel_size: grid pitch in world units.

    Returns:
        (M, 3) centers of the occupied voxels.
    """
    grid_coords = torch.round(points / voxel_size)
    unique_coords = torch.unique(grid_coords, dim=0)
    return unique_coords * voxel_size


class Scaffold:
    """
    Scaffold-GS anchors: the trainable state plus the per-view decode into neural Gaussians.

    - ``params``: the ParameterDict the densification strategy grows and prunes
    - ``mlps``: the fixed-size MLP heads, optimized separately and never handed to the strategy
    - ``decode`` returns the rasterizer inputs plus ``log_scales`` (log space, for the scale
      regularizer), ``visible_ids`` and ``decode_index``
    - ``decode_index``: the (anchor * n_offsets + offset) slot each emitted Gaussian came from;
      anchor densification is built entirely on it
    - no anchor opacity parameter and no ``activate``: mlp_opacity decodes it per view, and
      ``decode`` activates per view
    - exposes the rest of the ``Gaussians`` surface, so the trainer needs no branch
    """

    # Anchors, not gaussians: n_primitives counts anchors here and the writer's log line says so.
    primitive_unit: str = "anchors"

    def __init__(
        self,
        cfg: "SplatsConfig",
        points: np.ndarray,
        colors: np.ndarray,
        scene_scale: float,
        n_views: int,
        device: str,
        *,
        lr_decay: float = 0.01,
    ):
        """
        Voxelize the seed cloud into anchors, build MLP heads, optimizers, schedulers and strategy.

        Args:
            cfg: the run's SplatsConfig; `scaffold_config`, `primitive` and `max_steps` are read,
                none kept.
            points: (N, 3) float seed positions in the training frame.
            colors: (N, 3) uint8 seed colors. Unused; the color head learns color from scratch.
            scene_scale: camera extent of the training frame; scales the anchor and offset lrs.
            n_views: number of training views; sizes the appearance embedding.
            device: torch device string.
            lr_decay: total multiplicative decay of the anchor-position lr over the run.
        """
        self.cfg = cfg.scaffold_config
        self.primitive = cfg.primitive
        self.device = device

        # Local alias: every read below is of the scaffold block, never of the run config
        cfg_scaffold = self.cfg

        # Voxel size from the seed points' own spacing
        # - scale-free, so normalize_scene cannot silently change anchor density
        # - Scaffold's absolute 0.001 default assumes an already-normalized scene
        points_t = torch.from_numpy(np.asarray(points, dtype=np.float32))
        if cfg_scaffold.voxel_size is not None:
            self.voxel_size = float(cfg_scaffold.voxel_size)
        else:
            self.voxel_size = float(median_knn_spacing(points_t) * cfg_scaffold.voxel_multiplier)

        anchors = voxelize(points_t, self.voxel_size).to(device)
        n_anchors = len(anchors)
        logger.info(
            "scaffold: %d seed points -> %d anchors at voxel_size %.6g", len(points_t), n_anchors, self.voxel_size
        )

        # Anchor parameter tensors at init
        # - offsets: zero, as upstream seeds new anchors
        # - scaling: offset extent (first 3) and decoded-Gaussian extent (last 3), both log space
        log_voxel = math.log(self.voxel_size)
        self.params = torch.nn.ParameterDict(
            {
                "anchors": torch.nn.Parameter(anchors),
                "offsets": torch.nn.Parameter(torch.zeros(n_anchors, cfg_scaffold.n_offsets, 3, device=device)),
                "anchor_feat": torch.nn.Parameter(torch.zeros(n_anchors, cfg_scaffold.feat_dim, device=device)),
                "scaling": torch.nn.Parameter(torch.full((n_anchors, 6), log_voxel, device=device)),
                "rotation": torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_anchors, 1)),
            }
        )

        self.mlps = ScaffoldMLPs(cfg_scaffold, n_views=n_views).to(device)

        # One Adam per anchor tensor so the strategy can grow/prune optimizer state per tensor
        learning_rates = {
            "anchors": cfg_scaffold.anchor_lr * scene_scale,
            "offsets": cfg_scaffold.offset_lr * scene_scale,
            "anchor_feat": cfg_scaffold.anchor_feat_lr,
            "scaling": cfg_scaffold.scaling_lr,
            "rotation": cfg_scaffold.rotation_lr,
        }
        self.param_optimizers = {
            name: torch.optim.Adam([{"params": self.params[name], "lr": lr, "name": name}], eps=1e-15)
            for name, lr in learning_rates.items()
        }
        # One param group per head: upstream gives each head its own lr and its own decay schedule
        mlp_groups = [
            {"params": self.mlps.mlp_opacity.parameters(), "lr": cfg_scaffold.mlp_opacity_lr, "name": "mlp_opacity"},
            {"params": self.mlps.mlp_cov.parameters(), "lr": cfg_scaffold.mlp_cov_lr, "name": "mlp_cov"},
            {"params": self.mlps.mlp_color.parameters(), "lr": cfg_scaffold.mlp_color_lr, "name": "mlp_color"},
        ]
        if self.mlps.embedding_appearance is not None:
            mlp_groups.append(
                {
                    "params": self.mlps.embedding_appearance.parameters(),
                    "lr": cfg_scaffold.appearance_lr,
                    "name": "embedding_appearance",
                }
            )
        # No lr= here: the group dicts above already carry each head's own.
        self.mlp_optimizer = torch.optim.Adam(mlp_groups, eps=1e-15)

        # Flat optimizer list for the trainer
        # - one per anchor tensor, plus the multi-group MLP one
        # - param_optimizers stays: gsplat's _update_param_with_optimizer needs the name mapping
        self.optimizers = [*self.param_optimizers.values(), self.mlp_optimizer]

        # Anchor positions decay over the RUN, like the vanilla means lr
        self.anchor_scheduler = ExponentialLR(self.param_optimizers["anchors"], gamma=lr_decay ** (1.0 / cfg.max_steps))

        # Offsets and the MLP heads follow upstream's log-space curves
        # - horizon is lr_max_steps, not the run length: a shorter run stops partway
        # - LambdaLR applies lambda(0) at construction, lambda(step) after each step(), so the lr in
        #   force during step k is lambda(k)
        offset_lr = cfg_scaffold.offset_lr * scene_scale
        offset_lr_final = cfg_scaffold.offset_lr_final * scene_scale
        self.offset_scheduler = LambdaLR(
            self.param_optimizers["offsets"],
            lr_lambda=_decay_lambda(offset_lr, offset_lr_final, cfg_scaffold.lr_max_steps),
        )

        # (init, final) per MLP group; mlp_cov is constant upstream, so both endpoints are its lr
        mlp_endpoints = {
            "mlp_opacity": (cfg_scaffold.mlp_opacity_lr, cfg_scaffold.mlp_opacity_lr_final),
            "mlp_cov": (cfg_scaffold.mlp_cov_lr, cfg_scaffold.mlp_cov_lr),
            "mlp_color": (cfg_scaffold.mlp_color_lr, cfg_scaffold.mlp_color_lr_final),
            "embedding_appearance": (cfg_scaffold.appearance_lr, cfg_scaffold.appearance_lr_final),
        }
        self.mlp_scheduler = LambdaLR(
            self.mlp_optimizer,
            lr_lambda=[
                _decay_lambda(*mlp_endpoints[group["name"]], cfg_scaffold.lr_max_steps)
                for group in self.mlp_optimizer.param_groups
            ],
        )
        self.schedulers = [self.anchor_scheduler, self.offset_scheduler, self.mlp_scheduler]

        # Anchor densification: grows into under-covered cells and prunes anchors whose offsets shut
        self.strategy = AnchorStrategy(self.cfg, self.primitive, self.voxel_size, len(self.params["anchors"]), device)

    @torch.no_grad()
    def visible_anchors(self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int) -> Tensor:
        """
        Boolean mask of the anchors this view's projection keeps (radii > 0).

        - Upstream's prefilter_voxel over the anchors' OFFSET extent and rotation (GS-SR @ 566359be,
          gssr/scene/scaffold_scene.py:122-155); gsplat exposes the same kernel, so no extra pass.
        - gsplat ships CUDA kernels only, so off-GPU the analytic frustum test below stands in.

        Args:
            cam_to_world: (1, 4, 4) camera-to-world pose.
            intrinsics: (1, 3, 3) camera matrix in pixels.
            width: render width in pixels.
            height: render height in pixels.

        Returns:
            (A,) bool mask over the anchors; no gradient, it only selects which anchors decode.
        """
        anchors = self.params["anchors"]
        if not anchors.is_cuda:
            return self._frustum_anchors(cam_to_world, intrinsics, width, height)

        radii, *_ = fully_fused_projection(
            anchors,
            None,
            self.params["rotation"],
            torch.exp(self.params["scaling"][:, :3]),
            torch.linalg.inv(cam_to_world),
            intrinsics,
            width,
            height,
        )
        return radii.reshape(len(anchors), -1).amax(dim=-1) > 0

    def _frustum_anchors(self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int) -> Tensor:
        """
        CPU stand-in for the projection prefilter: anchor centers in front of the camera, in frame.

        - The whole-frame margin keeps anchors whose Gaussians spill into frame from centers just
          outside it, which is what the projection kernel's radii do on GPU.

        Returns:
            (A,) bool mask over the anchors.
        """
        world_to_cam = torch.linalg.inv(cam_to_world)[0]
        anchors_cam = self.params["anchors"] @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        depth = anchors_cam[:, 2]
        in_front = depth > 1e-3

        # Project with the frame's K; the margin keeps off-center anchors that still splat into frame
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
        Decode this view's neural Gaussians.

        - Never returns zero Gaussians: gsplat's projection kernel raises SIGFPE on an empty input.
        - Follows generate_neural_gaussians (city-super/Scaffold-GS @ 59c833b5,
          gaussian_renderer/__init__.py:18); reimplemented, no code copied.

        Args:
            primitive: "3dgs" or "2dgs"; 2dgs zeroes the unused third scale channel.
            cam_to_world: (1, 4, 4) camera-to-world pose.
            intrinsics: (1, 3, 3) camera matrix in pixels.
            width: render width in pixels.
            height: render height in pixels.
            camera_id: (1,) view index for the appearance embedding; None when off.

        Returns:
            (decoded, decode_index). decoded: rasterizer inputs — `colors` post-activation RGB, so
            `sh_degree=None` — plus `log_scales` for the scale regularizer and `visible_ids`, the
            anchors visited this view (opacity pruning's denominator). decode_index:
            `anchor * n_offsets + offset` per emitted Gaussian.
        """
        n_offsets = self.cfg.n_offsets
        visible = self.visible_anchors(cam_to_world, intrinsics, width, height)
        anchor_ids = torch.nonzero(visible, as_tuple=False).squeeze(-1)

        # An empty decode is fatal, not an empty image
        # - gsplat's projection kernel divides by the primitive count -> SIGFPE
        # - frustum culled everything: decode every anchor and let the rasterizer cull them
        if len(anchor_ids) == 0:
            anchor_ids = torch.arange(len(self.params["anchors"]), device=self.params["anchors"].device)

        anchors = self.params["anchors"][anchor_ids]
        feat = self.params["anchor_feat"][anchor_ids]
        scaling = torch.exp(self.params["scaling"][anchor_ids])
        offsets = self.params["offsets"][anchor_ids]

        # Unit direction from anchor to camera center feeds every head
        # - distance computed only to normalize it (upstream's ob_dist; add_*_dist ship off)
        camera_center = cam_to_world[0, :3, 3]
        to_camera = anchors - camera_center
        view_distance = to_camera.norm(dim=-1, keepdim=True)
        view_direction = to_camera / view_distance.clamp_min(1e-8)
        features = torch.cat([feat, view_direction], dim=-1)

        neural_opacity, cov, color = self.mlps(features, camera_id)

        # Offsets with non-positive opacity contribute nothing
        # - dropping them here keeps the decoded count far below anchors x n_offsets
        keep = (neural_opacity > 0).reshape(-1)

        # Same kernel constraint: keep the most opaque offset when none is open
        # - renders as good as nothing, and the MLP heads keep receiving gradient
        if not bool(keep.any()):
            keep = torch.zeros_like(keep)
            keep[neural_opacity.reshape(-1).argmax()] = True

        slot_index = (anchor_ids[:, None] * n_offsets + torch.arange(n_offsets, device=anchors.device)).reshape(-1)
        decode_index = slot_index[keep]

        # means = anchor + offset scaled by the anchor's offset extent (scaling[:, :3])
        means = (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep]

        # scales = the anchor's gaussian extent (scaling[:, 3:6]) modulated per offset
        cov = cov.reshape(-1, 7)[keep]
        scales = scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[keep] * torch.sigmoid(cov[:, :3])
        quats = F.normalize(cov[:, 3:7], dim=-1)
        opacities = neural_opacity.reshape(-1)[keep]
        colors = color.reshape(-1, 3)[keep]

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
            "visible_ids": anchor_ids,
        }
        return decoded, decode_index

    @property
    def n_primitives(self) -> int:
        """
        Number of anchors currently in the model. Decoded Gaussian count is per view and varies.
        """
        return len(self.params["anchors"])

    def render(
        self,
        cam_to_world: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
        camera_id: Tensor,
        step: int | None = None,
        render_normals: bool = True,
        render_plane: bool = False,
    ) -> tuple[dict[str, Tensor], dict]:
        """
        Decode this view's neural Gaussians and rasterize them.

        Args:
            cam_to_world: (1, 4, 4) camera-to-world pose in the training frame.
            intrinsics: (1, 3, 3) camera matrix in pixels at this render's resolution.
            width: render width in pixels.
            height: render height in pixels.
            camera_id: (1,) long view index; feeds the appearance embedding when it is on.
            step: current training step. Unused; the SH schedule is vanilla-only.
            render_normals: render the per-Gaussian normal and its finite-differenced partner.
            render_plane: add PGSR's planar signals (3dgs only).

        Returns:
            (render, info). render: adds the decoded `log_scales` and `opacities` the regularizers
            read — no parameter holds them here. info: adds `decode_index`, `decoded_opacities`
            and `visible_ids` for the strategy.
        """
        decoded, decode_index = self.decode(self.primitive, cam_to_world, intrinsics, width, height, camera_id)
        render, info = render_gaussians(
            self.primitive,
            decoded,
            cam_to_world,
            intrinsics,
            width,
            height,
            sh_degree=None,
            absgrad=False,
            render_normals=render_normals,
            render_plane=render_plane,
        )

        # The regularizers read decoded quantities; the strategy reads the decode bookkeeping
        render["log_scales"] = decoded["log_scales"]
        render["opacities"] = decoded["opacities"]
        info["decode_index"] = decode_index
        info["decoded_opacities"] = decoded["opacities"]
        info["visible_ids"] = decoded["visible_ids"]
        return render, info

    def frame_report(self, render: dict[str, Tensor]) -> dict[str, int]:
        """
        Per-view fields this representation adds to splats_quality_report.json.

        Args:
            render: one view's render dict; its `opacities` holds one entry per decoded Gaussian.

        Returns:
            {"n_decoded": count} — primitives handed to the rasterizer. Not inferable from
            `summary.n_gaussians`, which counts ANCHORS.
        """
        return {"n_decoded": int(len(render["opacities"]))}

    def pre_backward(self, step: int, info: dict) -> None:
        """
        Retain the screen-space gradient the anchor strategy accumulates after backward.

        Args:
            step: current training step. Unused; the signature is shared with ``Gaussians``.
            info: the gsplat info dict this step's render returned.
        """
        info[self.strategy.key_for_gradient].retain_grad()

    def post_backward(self, step: int, info: dict) -> None:
        """
        Accumulate anchor statistics, then grow and prune.

        - Retained gradient after the optimizer step is safe to read: `zero_grad(set_to_none=True)`
          clears the *parameters'* gradients, and this is a retained non-leaf

        Args:
            step: current training step.
            info: the gsplat info dict this step's render returned.
        """
        self.strategy.accumulate(step, info)
        self.strategy.refine(self, step)

    def denormalize(self, center: np.ndarray, scale: float) -> None:
        """
        Undo ``utils.scene_normalization`` on the anchors, in place.

        - Both halves of the log ``scaling`` shift by -log(scale); offsets are in units of the
          anchor's own extent, so they are scale-free and untouched.
        - The MLP heads survive because their only view input is a unit direction.

        Args:
            center: (3,) the center `scene_normalization` returned.
            scale: the scale `scene_normalization` returned.
        """
        center_t = torch.as_tensor(center, dtype=torch.float32, device=self.params["anchors"].device)
        with torch.no_grad():
            self.params["anchors"].data = self.params["anchors"].data / scale + center_t
            self.params["scaling"].data = self.params["scaling"].data - math.log(scale)

    @torch.no_grad()
    def export_gaussians(self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int) -> dict[str, Tensor]:
        """
        Decode each anchor once at its mean observed view direction → a static viewer-loadable ply.

        - Anchors no camera saw: fall back to the nearest camera's direction.
        - Lossy by construction: the model is view-dependent, so the ply and a render of the same
          anchors do not match. `rendering.render_views` decodes per view and is unaffected.

        Args:
            cam_to_world: (N, 4, 4) training poses in the output frame.
            intrinsics: (N, 3, 3) camera matrices at the training image size.
            width: training image width in pixels.
            height: training image height in pixels.

        Returns:
            {"means" (M,3), "scales" (M,3) log, "quats" (M,4), "opacities" (M,) logit,
            "sh0" (M,1,3), "shN" (M,0,3)} — the raw forms every ply viewer expects.
        """
        device = self.params["anchors"].device
        anchors = self.params["anchors"].detach()

        # Accumulate the unit direction to every camera that can see each anchor
        direction_sum = torch.zeros_like(anchors)
        seen_count = torch.zeros(len(anchors), device=device)
        for view in range(len(cam_to_world)):
            visible = self.visible_anchors(cam_to_world[view : view + 1], intrinsics[view : view + 1], width, height)
            to_camera = anchors - cam_to_world[view, :3, 3]
            unit = to_camera / to_camera.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            direction_sum[visible] += unit[visible]
            seen_count[visible] += 1

        # Unseen anchors: use the nearest camera's direction rather than dropping them from the ply
        unseen = seen_count == 0
        if bool(unseen.any()):
            camera_centers = cam_to_world[:, :3, 3]
            nearest = torch.cdist(anchors[unseen], camera_centers).argmin(dim=1)
            to_nearest = anchors[unseen] - camera_centers[nearest]
            direction_sum[unseen] = to_nearest / to_nearest.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            seen_count[unseen] = 1

        mean_direction = direction_sum / seen_count[:, None]
        mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # One decode at that direction, bypassing the frustum filter so every anchor is written
        features = torch.cat([self.params["anchor_feat"].detach(), mean_direction], dim=-1)
        camera_id = torch.zeros(1, dtype=torch.long, device=device)
        neural_opacity, cov, color = self.mlps(features, camera_id)

        n_offsets = self.cfg.n_offsets
        scaling = torch.exp(self.params["scaling"].detach())
        offsets = self.params["offsets"].detach()
        keep = (neural_opacity > 0).reshape(-1)

        # An all-closed decode would write an empty ply
        # - gsplat's export_splats cannot serialize it (its shN reshape needs >= 1 splat)
        # - keep the most opaque offset, mirroring the guard in decode
        if not bool(keep.any()):
            keep = torch.zeros_like(keep)
            keep[neural_opacity.reshape(-1).argmax()] = True

        means = (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep]
        cov = cov.reshape(-1, 7)[keep]
        scales = scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[keep] * torch.sigmoid(cov[:, :3])
        quats = F.normalize(cov[:, 3:7], dim=-1)
        opacities = neural_opacity.reshape(-1)[keep]
        colors = color.reshape(-1, 3)[keep]

        # The ply writer wants the raw forms every viewer expects: log scales, logit opacities, SH DC
        return {
            "means": means,
            "scales": torch.log(scales.clamp_min(1e-12)),
            "quats": quats,
            "opacities": torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),
            "sh0": ((colors - 0.5) / SH_C0).unsqueeze(1),
            "shN": torch.zeros(len(means), 0, 3, device=means.device),
        }

    def checkpoint(self) -> dict:
        """
        The model half of ckpt.pt.

        Returns:
            {"splats": ParameterDict, "mlps": state_dict, "voxel_size": float}. The trainer adds
            `config`, cameras and image ids.
        """
        return {"splats": self.params, "mlps": self.mlps.state_dict(), "voxel_size": self.voxel_size}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str) -> "Scaffold":
        """
        Rebuild a render-only model from a checkpoint.

        - No optimizers, schedulers or strategy: renders and exports, does not train.
        - Every attribute ``__init__`` binds is bound here, nulled, or a read raises AttributeError
          where the sibling ``Gaussians`` returns None.

        Args:
            ckpt: a loaded ckpt.pt holding `splats`, `mlps`, `voxel_size` and a plain-dict `config`.
            device: torch device string.

        Returns:
            A Scaffold instance whose parameters and heads are the checkpoint's, on `device`.
        """
        model = cls.__new__(cls)
        config = ckpt["config"]
        model.cfg = ScaffoldConfig.from_dict(config.get("scaffold") or {})
        model.primitive = config["primitive"]
        model.device = device
        model.voxel_size = ckpt["voxel_size"]
        model.params = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(tensor) for name, tensor in dict(ckpt["splats"]).items()}
        ).to(device)

        # n_views comes from the saved appearance embedding, which is the only view-sized head
        appearance_weight = ckpt["mlps"].get("embedding_appearance.weight")
        n_views = 0 if appearance_weight is None else len(appearance_weight)
        model.mlps = ScaffoldMLPs(model.cfg, n_views=n_views).to(device)
        model.mlps.load_state_dict(ckpt["mlps"])

        model.param_optimizers = {}
        model.mlp_optimizer = None
        model.optimizers = []
        model.anchor_scheduler = None
        model.offset_scheduler = None
        model.mlp_scheduler = None
        model.schedulers = []
        model.strategy = None
        return model


########################################
# Densification
########################################


class AnchorStrategy:
    """
    Scaffold-GS anchor densification: grow into under-covered voxels, prune anchors that shut.

    - screen-space gradients: accumulate per (anchor, offset) slot through the decode index,
      averaged by visit count — Scaffold's offset_gradient_accum / offset_denom
    - growing: adds anchors in unoccupied voxels around high-gradient slots
    - pruning: drops anchors whose accumulated decoded opacity stays below min_opacity
    - anchor tensors grow/prune through gsplat's optimizer-state surgery, so Adam moments follow
      the parameters; the fixed-size MLP heads are never handed to this class
    - reimplemented from Scaffold-GS (Lu et al., CVPR 2024); no code copied
    - upstream: training_statis / anchor_growing / adjust_anchor in city-super/Scaffold-GS @
      59c833b5, scene/gaussian_model.py:509, :582 and :681
    """

    def __init__(self, cfg: ScaffoldConfig, primitive: str, voxel_size: float, n_anchors: int, device: str):
        """
        Bind the config and allocate the four densification accumulators.

        Args:
            cfg: the run's ScaffoldConfig; thresholds and cadence, kept.
            primitive: "3dgs" or "2dgs"; picks which gsplat info key carries the gradient.
            voxel_size: anchor grid pitch in world units; growth quantizes to it.
            n_anchors: anchors at initialization; sizes the accumulators.
            device: torch device string; the accumulators live on it.
        """
        self.cfg = cfg
        self.voxel_size = voxel_size
        self.device = device

        # The gradient key must follow the primitive
        # - 2DGS backward writes .absgrad on means2d only, not on gradient_2dgs
        # - wrong key: accumulation is all zeros
        self.key_for_gradient = "means2d" if primitive == "3dgs" else "gradient_2dgs"

        # Growing statistics per slot (n_anchors x n_offsets), pruning per anchor
        # - each is a running sum with its own denominator
        # - a slot that never rendered reads unseen, not zero gradient
        n_slots = n_anchors * cfg.n_offsets
        self.offset_gradient_accum = torch.zeros(n_slots, device=device)
        self.offset_denom = torch.zeros(n_slots, device=device)
        self.opacity_accum = torch.zeros(n_anchors, device=device)
        self.anchor_denom = torch.zeros(n_anchors, device=device)

    def accumulate(self, step: int, info: dict) -> None:
        """
        Fold this step's screen-space gradients and decoded opacities into the running sums.

        - Window: upstream's, both bounds exclusive — counting starts before growing, so the first
          refine reads a full one (GS-SR @ 566359be, densify(),
          gssr/gaussian/scaffold_gaussian.py:710).
        - Gradients: renormalized to [-1, 1] screen space like gsplat's DefaultStrategy
          (gsplat @ d2f5c0f, gsplat/strategy/default.py:243-249)
        - Scaffold's grad_threshold is usable as-is
        - Opacity: summed per anchor, divided by the anchor's VISIT count, not by offset renders
          (upstream training_statis) — a visible anchor with every offset shut must score zero.
        - Gradient half filtered by radii > 0, upstream's update_filter (GS-SR @ 566359be,
          gssr/gaussian/scaffold_gaussian.py:506-508); the opacity half is not.

        Args:
            step: current training step; accumulation runs only inside the statistics window.
            info: the render's gsplat info dict, carrying `decode_index`, `decoded_opacities`,
                `visible_ids` and the retained gradient under `key_for_gradient`.
        """
        if not self.cfg.start_stat < step < self.cfg.update_until:
            return

        # A step whose gradient never reached the tensor (nothing rendered) is skipped, not counted
        grads = info[self.key_for_gradient].grad
        if grads is None:
            return
        grads = grads.detach().clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        grad_norm = grads.reshape(-1, 2).norm(dim=-1)

        index = info["decode_index"].to(self.device)

        # radii is [C, N, 2] (or [N, 2] for one camera); a Gaussian counts if any axis rendered
        rendered = info["radii"].reshape(len(grad_norm), -1).amax(dim=-1) > 0
        grad_index = index[rendered.to(self.device)]
        self.offset_gradient_accum.index_add_(0, grad_index, grad_norm[rendered].to(self.device))
        self.offset_denom.index_add_(0, grad_index, torch.ones_like(grad_index, dtype=self.offset_denom.dtype))

        # Summing the surviving slots gives upstream's per-anchor numerator
        # - negative opacities were dropped at decode; upstream clamps them to zero first
        anchor_index = torch.div(index, self.cfg.n_offsets, rounding_mode="floor")
        self.opacity_accum.index_add_(0, anchor_index, info["decoded_opacities"].detach().to(self.device))
        visible = info["visible_ids"].to(self.device)
        self.anchor_denom.index_add_(0, visible, torch.ones_like(visible, dtype=self.anchor_denom.dtype))

    def grow(self, scaffold: "Scaffold") -> int:
        """
        Add anchors in unoccupied voxels around slots whose mean gradient clears the threshold.

        - ``update_depth`` levels, COARSE to fine: level i raises the threshold by
          ``(update_hierarchy_factor // 2) ** i`` and shrinks the grid from ``update_init_factor``
          voxels towards one, so weak gradients seed coarse anchors and strong ones fine (upstream
          anchor_growing). Level i > 0 runs only if a coarser one added anchors this call.
        - Slots count only after most of a refine window; candidates thinned per level.
        - Candidates: the decoded means, deduped against each other and the anchor grid; one in an
          occupied voxel is dropped.
        - New anchor features: per-element max of its source slots'; blank would leave a grown
          scaffold mostly feature-less.

        Args:
            scaffold: the Scaffold whose params and param_optimizers are grown.

        Returns:
            The number of anchors added.
        """
        n_offsets = self.cfg.n_offsets
        mean_grads = self.offset_gradient_accum / self.offset_denom.clamp_min(1.0)

        # A slot's gradient counts only after most of a window's decodes
        # - upstream offset_denom > check_interval * success_threshold * 0.5
        seen = self.offset_denom > self.cfg.refine_every * self.cfg.success_threshold * 0.5

        added_total = 0
        for level in range(self.cfg.update_depth):
            # Finer levels need a coarser one to have added anchors in THIS call
            # - upstream's `length_inc == 0` continue
            #   (GS-SR @ 566359be, gssr/gaussian/scaffold_gaussian.py:568-573)
            if level > 0 and added_total == 0:
                break

            threshold = self.cfg.grad_threshold * ((self.cfg.update_hierarchy_factor // 2) ** level)
            size_factor = max(self.cfg.update_init_factor // (self.cfg.update_hierarchy_factor**level), 1)
            level_voxel = self.voxel_size * size_factor
            selected = seen & (mean_grads >= threshold)

            # Upstream thins candidates per level (rand > 0.5 ** (i + 1))
            # - one refine cannot claim every free cell around a hot region at once
            selected = selected & (torch.rand_like(mean_grads) > 0.5 ** (level + 1))
            if not bool(selected.any()):
                continue

            # Candidate positions are the slots' decoded means
            # - rebuilt from the current anchors, so each level sees what the last one added
            anchors = scaffold.params["anchors"].detach()
            offset_extent = torch.exp(scaffold.params["scaling"].detach()[:, :3])
            candidates = anchors[:, None, :] + scaffold.params["offsets"].detach() * offset_extent[:, None, :]
            candidates = candidates.reshape(-1, 3)[selected]

            # Each candidate carries its source anchor's feature into the cell it lands in
            slot_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
            source_feat = scaffold.params["anchor_feat"].detach()[torch.div(slot_ids, n_offsets, rounding_mode="floor")]

            # Quantize onto this level's grid; occupied cells are dropped
            # - unique+counts, not a pairwise mask: at 100k anchors the latter is tens of GB
            candidate_cells, cell_of_candidate = torch.unique(
                torch.round(candidates / level_voxel), dim=0, return_inverse=True
            )
            cell_feat = torch.zeros(len(candidate_cells), self.cfg.feat_dim, device=source_feat.device)
            cell_feat.index_reduce_(0, cell_of_candidate, source_feat, "amax", include_self=False)

            occupied_cells = torch.round(anchors / level_voxel)
            combined = torch.cat([occupied_cells, candidate_cells], dim=0)
            _, inverse, counts = torch.unique(combined, dim=0, return_inverse=True, return_counts=True)
            free = counts[inverse[len(occupied_cells) :]] == 1
            new_cells = candidate_cells[free]
            if len(new_cells) == 0:
                continue

            self._append_anchors(scaffold, new_cells * level_voxel, level_voxel, cell_feat[free])
            added_total += len(new_cells)

            # The per-slot views grow with the anchors
            # - new slots have no history and cannot seed the next level
            padding = torch.zeros(len(new_cells) * n_offsets, device=mean_grads.device)
            mean_grads = torch.cat([mean_grads, padding])
            seen = torch.cat([seen, padding.bool()])

        # Only the slots that carried evidence reset
        # - upstream leaves the rest accumulating, so a rarely-visible slot still builds a window
        self.offset_gradient_accum[seen] = 0.0
        self.offset_denom[seen] = 0.0

        if added_total:
            logger.debug("scaffold: grew %d anchors -> %d", added_total, len(scaffold.params["anchors"]))
        return added_total

    def _append_anchors(self, scaffold: "Scaffold", new_anchors: Tensor, level_voxel: float, new_feat: Tensor) -> None:
        """
        Append new anchors (zero offsets, inherited features, level-sized scaling).

        - Extends `params`, their Adam moments and the four accumulators together; resizing only
          the parameters leaves every accumulator indexed one anchor short.

        Args:
            scaffold: the Scaffold whose params and param_optimizers are extended in place.
            new_anchors: (N, 3) world positions of the anchors to add.
            level_voxel: this growth level's voxel edge; sets the new anchors' initial scaling.
            new_feat: (N, feat_dim) features inherited from the anchors the candidates came from.
        """
        n_new = len(new_anchors)
        device = scaffold.params["anchors"].device
        log_voxel = math.log(level_voxel)
        additions = {
            "anchors": new_anchors.to(device),
            "offsets": torch.zeros(n_new, self.cfg.n_offsets, 3, device=device),
            "anchor_feat": new_feat.to(device),
            "scaling": torch.full((n_new, 6), log_voxel, device=device),
            "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_new, 1),
        }

        # gsplat's helper rebuilds each Parameter and its Adam moments together
        # - new rows start at zero momentum, as upstream densification does
        def param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(torch.cat([param.detach(), additions[name]], dim=0))

        def optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return torch.cat([value, torch.zeros((n_new, *value.shape[1:]), device=value.device)], dim=0)

        _update_param_with_optimizer(param_fn, optimizer_fn, scaffold.params, scaffold.param_optimizers)

        # Growing statistics are per slot, pruning statistics per anchor, so they grow by different rows
        slot_padding = torch.zeros(n_new * self.cfg.n_offsets, device=self.device)
        anchor_padding = torch.zeros(n_new, device=self.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, slot_padding])
        self.offset_denom = torch.cat([self.offset_denom, slot_padding.clone()])
        self.opacity_accum = torch.cat([self.opacity_accum, anchor_padding])
        self.anchor_denom = torch.cat([self.anchor_denom, anchor_padding.clone()])

    def prune(self, scaffold: "Scaffold", *, scale_cap: float = 0.05) -> int:
        """
        Drop anchors whose mean decoded opacity stayed below min_opacity across the window.

        - An anchor needs most of a window's visits before its mean opacity is evidence (upstream
          anchor_demon > check_interval * success_threshold); a never-decoded one is kept.
        - Never prunes the scaffold empty: gsplat's projection kernel raises SIGFPE on empty input.

        Args:
            scaffold: the Scaffold whose params and param_optimizers are pruned.
            scale_cap: upper bound on the raw (log-space) gaussian-extent channels of `scaling`,
                applied every refine. Upstream's value (GS-SR @ 566359be,
                gssr/gaussian/scaffold_gaussian.py:530); raising it lets offsets outgrow the cell.

        Returns:
            The number of anchors removed.
        """
        n_offsets = self.cfg.n_offsets
        denom = self.anchor_denom
        mean_opacity = self.opacity_accum / denom.clamp_min(1.0)
        seen = denom > self.cfg.refine_every * self.cfg.success_threshold
        drop = seen & (mean_opacity < self.cfg.min_opacity)

        # The window closes for the anchors that carried evidence, whether or not they were dropped
        self.opacity_accum[seen] = 0.0
        self.anchor_denom[seen] = 0.0

        # Upstream caps the raw gaussian-extent channels on every refine
        # - not only the ones that drop an anchor: the clamp rides inside unconditional optimizer
        #   surgery (GS-SR @ 566359be, gssr/gaussian/scaffold_gaussian.py:530, from adjust_anchor:703)
        with torch.no_grad():
            scaffold.params["scaling"][:, 3:].clamp_(max=scale_cap)

        if not bool(drop.any()):
            return 0

        # An empty scaffold cannot be decoded or rendered, so the most opaque anchor always survives
        if bool(drop.all()):
            drop[mean_opacity.argmax()] = False
            logger.warning(
                "scaffold: every anchor fell below min_opacity %.4g; kept the most opaque one", self.cfg.min_opacity
            )

        keep = ~drop
        keep_slots = keep.repeat_interleave(n_offsets)

        # Mirror of the growing surgery
        # - gsplat's helper reindexes each Parameter and its Adam moments through the same mask
        def param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param.detach()[keep])

        def optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return value[keep]

        _update_param_with_optimizer(param_fn, optimizer_fn, scaffold.params, scaffold.param_optimizers)

        self.offset_gradient_accum = self.offset_gradient_accum[keep_slots]
        self.offset_denom = self.offset_denom[keep_slots]
        self.opacity_accum = self.opacity_accum[keep]
        self.anchor_denom = self.anchor_denom[keep]

        n_dropped = int(drop.sum())
        logger.debug("scaffold: pruned %d anchors -> %d", n_dropped, len(scaffold.params["anchors"]))
        return n_dropped

    def refine(self, scaffold: "Scaffold", step: int) -> None:
        """
        Grow then prune anchors on the refine cadence.

        - Both bounds exclusive. The check reads only `update_from` / `update_until`, matching
          upstream's nested statistics gate whenever `start_stat <= update_from` — true of both
          default sets, 500 vs 1500 (GS-SR @ 566359be, gssr/gaussian/scaffold_gaussian.py:707-723).
        - Each half resets only the slots / anchors whose statistics it consumed
        - A rarely-visible one keeps building history

        Args:
            scaffold: the Scaffold whose params and param_optimizers are grown and pruned.
            step: current training step; refinement runs every `refine_every` steps in the window.
        """
        if step <= self.cfg.update_from or step >= self.cfg.update_until:
            return
        if step % self.cfg.refine_every != 0:
            return

        self.grow(scaffold)
        self.prune(scaffold)
