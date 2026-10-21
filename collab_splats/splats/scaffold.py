"""
Scaffold-GS: anchors whose MLP heads decode neural Gaussians per view.

- `ScaffoldConfig`, `ScaffoldMLPs`: the yaml block and the three decode heads
- `Scaffold`: anchor parameters, per-view decode, render, export, checkpoint
- `AnchorStrategy`: grows and prunes anchors
- reimplements city-super/Scaffold-GS @ 59c833b5: scene/gaussian_model.py, gaussian_renderer/__init__.py,
  arguments/__init__.py, train.py; no code vendored
- Scaffold x 2DGS (two-scale decode) follows yanxian-ll/GS-SR @ 566359be, gssr/scene/scaffold_2dgs_scene.py:8-19
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from gsplat import fully_fused_projection
from gsplat.strategy.ops import _update_param_with_optimizer

from collab_splats.geometry.projection import project
from collab_splats.splats.gaussian import SH_C0
from collab_splats.splats.rendering import render_gaussians
from collab_splats.splats.utils import knn_spacing

# Annotation-only: a runtime import of the trainer would be circular
if TYPE_CHECKING:
    from collab_splats.splats.trainer import SplatsConfig

logger = logging.getLogger(__name__)

########################################
# Config
########################################


@dataclass
class ScaffoldConfig:
    """
    Settings for the anchor representation; the `splats.scaffold:` yaml block.
    """

    n_offsets: int = 10
    feat_dim: int = 32

    # Anchor voxel size: median seed spacing x voxel_multiplier, or voxel_size in world units
    voxel_multiplier: float = 1.0
    voxel_size: float | None = None

    # Densification window and thresholds
    # - grad_threshold: Scaffold's published value
    # - gradients normalized as in gsplat @ d2f5c0f, gsplat/strategy/default.py:243-249
    start_stat: int = 500  # statistics start before growing
    update_from: int = 1500
    update_until: int = 15000
    refine_every: int = 100
    grad_threshold: float = 2e-4
    min_opacity: float = 0.005
    update_depth: int = 3  # coarse-to-fine growing levels
    update_hierarchy_factor: int = 4
    update_init_factor: int = 16  # level-0 growth grid, in voxels
    success_threshold: float = 0.8  # fraction of a window a slot must be seen

    # Per-image appearance embedding for the color head
    # - 0 disables it
    # - independent of splats.appearance_opt
    appearance_dim: int = 32

    # Learning rates
    # - anchor / offset lrs scale with scene_scale
    # - anchor_lr 0: anchors stay on the voxel grid, as upstream
    anchor_lr: float = 0.0
    offset_lr: float = 1e-2
    offset_lr_final: float = 1e-4
    anchor_feat_lr: float = 7.5e-3
    scaling_lr: float = 7e-3
    rotation_lr: float = 2e-3

    # One lr and decay per head, as upstream
    mlp_opacity_lr: float = 2e-3
    mlp_opacity_lr_final: float = 2e-5
    mlp_cov_lr: float = 4e-3
    mlp_color_lr: float = 8e-3
    mlp_color_lr_final: float = 5e-5
    appearance_lr: float = 5e-2
    appearance_lr_final: float = 5e-4

    # Decay horizon, fixed rather than the run length
    lr_max_steps: int = 30000

    # bf16 autocast of the three decode heads; outputs return as float32; false for float32 heads
    mlp_bf16: bool = True

    @classmethod
    def from_dict(cls, block: dict) -> "ScaffoldConfig":
        """
        Build from the yaml block, rejecting unknown keys.

        Args:
            block: the parsed `splats.scaffold:` mapping.

        Returns:
            The validated config.
        """
        unknown = set(block) - set(cls.__dataclass_fields__)

        if unknown:
            raise ValueError(
                f"splats.scaffold: unknown keys {sorted(unknown)}; allowed {sorted(cls.__dataclass_fields__)}"
            )

        cfg = cls(**block)

        # Reject sizes that would break the decode later
        if cfg.n_offsets < 1:
            raise ValueError(
                f"splats.scaffold.n_offsets must be >= 1, got {cfg.n_offsets}"
            )

        if cfg.feat_dim < 1:
            raise ValueError(
                f"splats.scaffold.feat_dim must be >= 1, got {cfg.feat_dim}"
            )

        if cfg.voxel_size is not None and cfg.voxel_size <= 0:
            raise ValueError(
                f"splats.scaffold.voxel_size must be > 0 when set, got {cfg.voxel_size}"
            )

        if cfg.lr_max_steps < 1:
            raise ValueError(
                f"splats.scaffold.lr_max_steps must be >= 1, got {cfg.lr_max_steps}"
            )

        if cfg.voxel_multiplier <= 0:
            raise ValueError(
                f"splats.scaffold.voxel_multiplier must be > 0, got {cfg.voxel_multiplier}"
            )

        return cfg


########################################
# MLP heads
########################################


class ScaffoldMLPs(torch.nn.Module):
    """
    The three decode heads: opacity, covariance, color.

    - input: anchor feature + unit view direction; output: `n_offsets` values per anchor
    - opacity is tanh; a non-positive value hides that offset
    - `mlp_bf16` runs the heads under bf16 autocast, forward and backward
    """

    def __init__(self, cfg: ScaffoldConfig, n_views: int = 0):
        """
        Build the heads and, if enabled, the appearance embedding.

        Args:
            cfg: scaffold config; `feat_dim`, `n_offsets`, `appearance_dim` and `mlp_bf16` are read.
            n_views: number of training views; 0 leaves out the appearance embedding.
        """
        super().__init__()
        self.n_offsets = cfg.n_offsets
        self.bf16 = cfg.mlp_bf16
        width = cfg.feat_dim

        # Head input: anchor feature + unit view direction, no distance
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

        # Appearance embedding feeds the color head only
        self.embedding_appearance = None
        color_dim = base_dim

        if cfg.appearance_dim > 0:
            if n_views < 1:
                raise ValueError(
                    "scaffold.appearance_dim > 0 needs n_views >= 1 to size the embedding"
                )

            self.embedding_appearance = torch.nn.Embedding(n_views, cfg.appearance_dim)
            color_dim += cfg.appearance_dim

        self.mlp_color = torch.nn.Sequential(
            torch.nn.Linear(color_dim, width),
            torch.nn.ReLU(True),
            torch.nn.Linear(width, 3 * cfg.n_offsets),
            torch.nn.Sigmoid(),
        )

    def forward(
        self, features: Tensor, camera_id: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Opacity, covariance and color for each anchor.

        Args:
            features: (A, feat_dim + 3) anchor features + unit view direction.
            camera_id: (1,) view index; required when the appearance embedding is on.

        Returns:
            (opacity (A, K), covariance (A, 7K), color (A, 3K)), K = n_offsets.
        """
        # Optional bf16 autocast; float32 (a no-op cast when off) for everything downstream
        with torch.autocast(
            device_type=features.device.type, dtype=torch.bfloat16, enabled=self.bf16
        ):
            opacity = self.mlp_opacity(features)
            cov = self.mlp_cov(features)

            # Append this view's appearance embedding to the color input
            color_input = features

            if self.embedding_appearance is not None:
                if camera_id is None:
                    raise ValueError(
                        "scaffold.appearance_dim > 0 requires camera_id at decode time"
                    )

                embedding = self.embedding_appearance(camera_id[:1]).expand(
                    len(features), -1
                )
                color_input = torch.cat([features, embedding], dim=-1)

            color = self.mlp_color(color_input)

        return opacity.float(), cov.float(), color.float()


########################################
# Scaffold
########################################


def _decay_lambda(lr_init: float, lr_final: float, lr_max_steps: int):
    """
    LambdaLR multiplier for a log-space decay from lr_init to lr_final.

    - matches upstream's `get_expon_lr_func`; flat past lr_max_steps
    - non-positive endpoints hold the lr flat
    """
    if lr_init <= 0.0 or lr_final <= 0.0:
        return lambda step: 1.0

    ratio = lr_final / lr_init
    horizon = max(lr_max_steps, 1)
    return lambda step: ratio ** min(step / horizon, 1.0)


def voxelize(points: Tensor, voxel_size: float) -> Tensor:
    """
    Centers of the voxels that contain at least one point.

    Args:
        points: (N, 3) positions.
        voxel_size: voxel edge in world units.

    Returns:
        (M, 3) centers of the occupied voxels.
    """
    grid_coords = torch.round(points / voxel_size)
    unique_coords = torch.unique(grid_coords, dim=0)
    return unique_coords * voxel_size


def _offsets_to_gaussians(
    anchors: Tensor,
    scaling: Tensor,
    offsets: Tensor,
    neural_opacity: Tensor,
    cov: Tensor,
    color: Tensor,
    n_offsets: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Neural Gaussians from the open offsets, plus the kept slot indices.

    - offsets with non-positive opacity are dropped
    - the most opaque is always kept: gsplat needs at least one splat; a no-op when any is open
    - one `nonzero` (host sync) gathers every field, not one per boolean mask
    """
    flat_opacity = neural_opacity.reshape(-1)
    keep = flat_opacity > 0
    keep[flat_opacity.argmax(dim=0, keepdim=True)] = True
    kept = torch.nonzero(keep).squeeze(-1)

    # Means: anchor + offset x offset extent
    offset_extent = scaling[:, None, :3]
    means = anchors[:, None, :] + offsets * offset_extent
    means = means.reshape(-1, 3)[kept]

    # Scales: anchor's Gaussian extent x sigmoid of the cov head
    cov = cov.reshape(-1, 7)[kept]
    extent = scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[kept]
    modulation = torch.sigmoid(cov[:, :3])
    scales = extent * modulation

    # Rotation, opacity and color of the open slots
    quats = F.normalize(cov[:, 3:7], dim=-1)
    opacities = flat_opacity[kept]
    colors = color.reshape(-1, 3)[kept]

    gaussians = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
    }
    return kept, gaussians


class Scaffold:
    """
    Anchor-based splat model, a drop-in for `Gaussians` in the trainer.

    - `params`: anchor tensors, grown and pruned by `strategy`
    - `mlps`: decode heads, fixed size
    - Gaussians exist only per view, via `decode`
    """

    # n_primitives counts anchors, not Gaussians
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
        adam_eps: float = 1e-15,
    ):
        """
        Seed anchors from a point cloud and set up heads, optimizers and densification.

        Args:
            cfg: run config; `scaffold_config`, `primitive` and `max_steps` are read.
            points: (N, 3) seed positions in the training frame.
            colors: (N, 3) seed colors; unused.
            scene_scale: camera extent of the training frame; scales anchor and offset lrs.
            n_views: number of training views.
            device: torch device string.
            lr_decay: total decay of the anchor lr over the run.
            adam_eps: Adam epsilon for every optimizer.
        """
        self.cfg = cfg.scaffold_config
        self.primitive = cfg.primitive
        self.device = device

        # Short alias for the scaffold block
        cfg_scaffold = self.cfg

        # Voxel size: explicit, or from the seed points' median spacing
        points_t = torch.from_numpy(np.asarray(points, dtype=np.float32))

        if cfg_scaffold.voxel_size is not None:
            self.voxel_size = float(cfg_scaffold.voxel_size)
        else:
            spacing = knn_spacing(points_t.numpy(), 3)
            self.voxel_size = float(np.median(spacing) * cfg_scaffold.voxel_multiplier)

        anchors = voxelize(points_t, self.voxel_size).to(device)
        n_anchors = len(anchors)
        logger.info(
            "scaffold: %d seed points -> %d anchors at voxel_size %.6g",
            len(points_t),
            n_anchors,
            self.voxel_size,
        )

        # Anchor tensors; scaling = log offset extent (3) + log Gaussian extent (3)
        log_voxel = math.log(self.voxel_size)
        self.params = torch.nn.ParameterDict(
            {
                "anchors": torch.nn.Parameter(anchors),
                "offsets": torch.nn.Parameter(
                    torch.zeros(n_anchors, cfg_scaffold.n_offsets, 3, device=device)
                ),
                "anchor_feat": torch.nn.Parameter(
                    torch.zeros(n_anchors, cfg_scaffold.feat_dim, device=device)
                ),
                "scaling": torch.nn.Parameter(
                    torch.full((n_anchors, 6), log_voxel, device=device)
                ),
                "rotation": torch.nn.Parameter(
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(
                        n_anchors, 1
                    )
                ),
            }
        )

        self.mlps = ScaffoldMLPs(cfg_scaffold, n_views=n_views).to(device)

        # One Adam per anchor tensor, so densification can resize its state
        learning_rates = {
            "anchors": cfg_scaffold.anchor_lr * scene_scale,
            "offsets": cfg_scaffold.offset_lr * scene_scale,
            "anchor_feat": cfg_scaffold.anchor_feat_lr,
            "scaling": cfg_scaffold.scaling_lr,
            "rotation": cfg_scaffold.rotation_lr,
        }
        self.param_optimizers = {
            name: torch.optim.Adam(
                [{"params": self.params[name], "lr": lr, "name": name}], eps=adam_eps
            )
            for name, lr in learning_rates.items()
        }
        # One param group per head, each with its own lr
        mlp_groups = [
            {
                "params": self.mlps.mlp_opacity.parameters(),
                "lr": cfg_scaffold.mlp_opacity_lr,
                "name": "mlp_opacity",
            },
            {
                "params": self.mlps.mlp_cov.parameters(),
                "lr": cfg_scaffold.mlp_cov_lr,
                "name": "mlp_cov",
            },
            {
                "params": self.mlps.mlp_color.parameters(),
                "lr": cfg_scaffold.mlp_color_lr,
                "name": "mlp_color",
            },
        ]

        if self.mlps.embedding_appearance is not None:
            mlp_groups.append(
                {
                    "params": self.mlps.embedding_appearance.parameters(),
                    "lr": cfg_scaffold.appearance_lr,
                    "name": "embedding_appearance",
                }
            )

        # Each group carries its own lr
        self.mlp_optimizer = torch.optim.Adam(mlp_groups, eps=adam_eps)

        # Flat optimizer list for the trainer; param_optimizers keeps the by-name map
        self.optimizers = [*self.param_optimizers.values(), self.mlp_optimizer]

        # Anchor lr decays over the run
        self.anchor_scheduler = ExponentialLR(
            self.param_optimizers["anchors"], gamma=lr_decay ** (1.0 / cfg.max_steps)
        )

        # Offsets and heads decay over lr_max_steps, not the run length
        offset_lr = cfg_scaffold.offset_lr * scene_scale
        offset_lr_final = cfg_scaffold.offset_lr_final * scene_scale
        self.offset_scheduler = LambdaLR(
            self.param_optimizers["offsets"],
            lr_lambda=_decay_lambda(
                offset_lr, offset_lr_final, cfg_scaffold.lr_max_steps
            ),
        )

        # (init, final) lr per head; mlp_cov is constant
        mlp_endpoints = {
            "mlp_opacity": (
                cfg_scaffold.mlp_opacity_lr,
                cfg_scaffold.mlp_opacity_lr_final,
            ),
            "mlp_cov": (cfg_scaffold.mlp_cov_lr, cfg_scaffold.mlp_cov_lr),
            "mlp_color": (cfg_scaffold.mlp_color_lr, cfg_scaffold.mlp_color_lr_final),
            "embedding_appearance": (
                cfg_scaffold.appearance_lr,
                cfg_scaffold.appearance_lr_final,
            ),
        }
        self.mlp_scheduler = LambdaLR(
            self.mlp_optimizer,
            lr_lambda=[
                _decay_lambda(*mlp_endpoints[group["name"]], cfg_scaffold.lr_max_steps)
                for group in self.mlp_optimizer.param_groups
            ],
        )
        self.schedulers = [
            self.anchor_scheduler,
            self.offset_scheduler,
            self.mlp_scheduler,
        ]

        # Anchor densification
        self.strategy = AnchorStrategy(
            self.cfg,
            self.primitive,
            self.voxel_size,
            len(self.params["anchors"]),
            device,
        )

    @torch.no_grad()
    def visible_anchors(
        self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int
    ) -> Tensor:
        """
        Mask of the anchors visible in this view.

        - same anchor prefilter as upstream `prefilter_voxel`
        - CPU falls back to a frustum test

        Args:
            cam_to_world: (1, 4, 4) camera-to-world pose.
            intrinsics: (1, 3, 3) camera matrix in pixels.
            width: render width in pixels.
            height: render height in pixels.

        Returns:
            (A,) bool mask over the anchors.
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

    def _frustum_anchors(
        self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int
    ) -> Tensor:
        """
        CPU visibility test: anchors in front of the camera and near the frame.

        - half-frame margin on each side
        """
        # Project to pixels; depth floor 1e-3 keeps points behind the camera finite
        world_to_cam = torch.linalg.inv(cam_to_world)[0]
        projected, anchors_cam = project(
            self.params["anchors"], world_to_cam, intrinsics[0], min_depth=1e-3
        )
        in_front = anchors_cam[:, 2] > 1e-3

        # The margin keeps anchors just outside the frame
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

        - never empty: gsplat crashes (SIGFPE) on zero Gaussians

        Args:
            primitive: "3dgs" or "2dgs".
            cam_to_world: (1, 4, 4) camera-to-world pose.
            intrinsics: (1, 3, 3) camera matrix in pixels.
            width: render width in pixels.
            height: render height in pixels.
            camera_id: (1,) view index for the appearance embedding; None when off.

        Returns:
            (decoded, decode_index): rasterizer inputs plus `log_scales` and `visible_ids`, and
            the `anchor * n_offsets + offset` slot of each Gaussian.
        """
        n_offsets = self.cfg.n_offsets
        visible = self.visible_anchors(cam_to_world, intrinsics, width, height)
        anchor_ids = torch.nonzero(visible, as_tuple=False).squeeze(-1)

        # Nothing visible: decode every anchor, since an empty decode crashes gsplat
        if len(anchor_ids) == 0:
            anchor_ids = torch.arange(
                len(self.params["anchors"]), device=self.params["anchors"].device
            )

        anchors = self.params["anchors"][anchor_ids]
        feat = self.params["anchor_feat"][anchor_ids]
        scaling = torch.exp(self.params["scaling"][anchor_ids])
        offsets = self.params["offsets"][anchor_ids]

        # Head input: anchor feature + unit direction from the camera
        camera_center = cam_to_world[0, :3, 3]
        to_camera = anchors - camera_center
        view_distance = to_camera.norm(dim=-1, keepdim=True)
        view_direction = to_camera / view_distance.clamp_min(1e-8)
        features = torch.cat([feat, view_direction], dim=-1)

        neural_opacity, cov, color = self.mlps(features, camera_id)

        # Keep open offsets and record which slot each came from
        kept, gaussians = _offsets_to_gaussians(
            anchors, scaling, offsets, neural_opacity, cov, color, n_offsets
        )
        slot_offsets = torch.arange(n_offsets, device=anchors.device)
        slot_index = anchor_ids[:, None] * n_offsets + slot_offsets
        slot_index = slot_index.reshape(-1)
        decode_index = slot_index[kept]
        scales = gaussians["scales"]

        # 2DGS uses two scales; zero the third
        if primitive == "2dgs":
            zeros = torch.zeros_like(scales[:, :1])
            log_scales = torch.cat(
                [torch.log(scales[:, :2].clamp_min(1e-12)), zeros], dim=-1
            )
            scales = torch.cat([scales[:, :2], zeros], dim=-1)
        else:
            log_scales = torch.log(scales.clamp_min(1e-12))

        decoded = {
            "means": gaussians["means"],
            "quats": gaussians["quats"],
            "scales": scales,
            "opacities": gaussians["opacities"],
            "colors": gaussians["colors"],
            "log_scales": log_scales,
            "visible_ids": anchor_ids,
        }
        return decoded, decode_index

    @property
    def n_primitives(self) -> int:
        """
        Number of anchors currently in the model.

        Returns:
            Anchor count.
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
    ) -> tuple[dict[str, Tensor], dict]:
        """
        Decode this view's neural Gaussians and rasterize them.

        Args:
            cam_to_world: (1, 4, 4) camera-to-world pose.
            intrinsics: (1, 3, 3) camera matrix in pixels.
            width: render width in pixels.
            height: render height in pixels.
            camera_id: (1,) view index for the appearance embedding.
            step: current training step; unused.
            render_normals: also render normals.

        Returns:
            (render, info): gsplat outputs plus the decoded Gaussians' fields the losses and
            densification read.
        """
        decoded, decode_index = self.decode(
            self.primitive, cam_to_world, intrinsics, width, height, camera_id
        )
        render, info = render_gaussians(
            self.primitive,
            decoded,
            cam_to_world,
            intrinsics,
            width,
            height,
            sh_degree=None,
            render_normals=render_normals,
        )

        # Expose decoded fields to the losses and the strategy
        render["log_scales"] = decoded["log_scales"]
        render["opacities"] = decoded["opacities"]
        info["decode_index"] = decode_index
        info["decoded_opacities"] = decoded["opacities"]
        info["visible_ids"] = decoded["visible_ids"]
        return render, info

    def frame_report(self, render: dict[str, Tensor]) -> dict[str, int]:
        """
        Per-view fields added to splats_quality_report.json.

        Args:
            render: one view's render dict.

        Returns:
            {"n_decoded": number of Gaussians decoded for this view}.
        """
        return {"n_decoded": int(len(render["opacities"]))}

    def pre_backward(self, step: int, info: dict) -> None:
        """
        Keep the screen-space gradient that densification reads after backward.

        Args:
            step: current training step; unused.
            info: gsplat info dict from this step's render.
        """
        info[self.strategy.key_for_gradient].retain_grad()

    def post_backward(self, step: int, info: dict) -> None:
        """
        Accumulate anchor statistics, then grow and prune.

        Args:
            step: current training step.
            info: gsplat info dict from this step's render.
        """
        self.strategy.accumulate(step, info)
        self.strategy.refine(self, step)

    def denormalize(self, center: np.ndarray, scale: float) -> None:
        """
        Undo `utils.scene_normalization` on the anchors, in place.

        - offsets and MLP heads are scale-free and left as is

        Args:
            center: (3,) center returned by `scene_normalization`.
            scale: scale returned by `scene_normalization`.
        """
        center_t = torch.as_tensor(
            center, dtype=torch.float32, device=self.params["anchors"].device
        )

        with torch.no_grad():
            self.params["anchors"].data = self.params["anchors"].data / scale + center_t
            self.params["scaling"].data = self.params["scaling"].data - math.log(scale)

    @torch.no_grad()
    def export_gaussians(
        self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int
    ) -> dict[str, Tensor]:
        """
        Static Gaussians for a ply: each anchor decoded at its mean view direction.

        - lossy: renders of the ply differ from per-view decodes
        - unseen anchors use the nearest camera's direction

        Args:
            cam_to_world: (N, 4, 4) training poses.
            intrinsics: (N, 3, 3) camera matrices.
            width: image width in pixels.
            height: image height in pixels.

        Returns:
            means, log scales, quats, logit opacities, sh0 and empty shN, as the ply expects.
        """
        device = self.params["anchors"].device
        anchors = self.params["anchors"].detach()

        # Sum unit directions over the cameras that see each anchor
        direction_sum = torch.zeros_like(anchors)
        seen_count = torch.zeros(len(anchors), device=device)

        for view in range(len(cam_to_world)):
            visible = self.visible_anchors(
                cam_to_world[view : view + 1],
                intrinsics[view : view + 1],
                width,
                height,
            )
            to_camera = anchors - cam_to_world[view, :3, 3]
            unit = to_camera / to_camera.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            direction_sum[visible] += unit[visible]
            seen_count[visible] += 1

        # Unseen anchors: use the nearest camera's direction
        unseen = seen_count == 0

        if bool(unseen.any()):
            camera_centers = cam_to_world[:, :3, 3]
            nearest = torch.cdist(anchors[unseen], camera_centers).argmin(dim=1)
            to_nearest = anchors[unseen] - camera_centers[nearest]
            direction_sum[unseen] = to_nearest / to_nearest.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            seen_count[unseen] = 1

        mean_direction = direction_sum / seen_count[:, None]
        mean_direction = mean_direction / mean_direction.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        # Decode every anchor at its mean direction
        features = torch.cat(
            [self.params["anchor_feat"].detach(), mean_direction], dim=-1
        )
        camera_id = torch.zeros(1, dtype=torch.long, device=device)
        neural_opacity, cov, color = self.mlps(features, camera_id)

        n_offsets = self.cfg.n_offsets
        scaling = torch.exp(self.params["scaling"].detach())
        offsets = self.params["offsets"].detach()

        # Same slot-to-Gaussian step as a render
        _, gaussians = _offsets_to_gaussians(
            anchors, scaling, offsets, neural_opacity, cov, color, n_offsets
        )
        means = gaussians["means"]
        scales = gaussians["scales"]
        quats = gaussians["quats"]
        opacities = gaussians["opacities"]
        colors = gaussians["colors"]

        # Convert to raw ply forms: log scales, logit opacities, SH DC
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
        Model state for ckpt.pt.

        Returns:
            {"splats", "mlps", "voxel_size"}; the trainer adds the rest.
        """
        return {
            "splats": self.params,
            "mlps": self.mlps.state_dict(),
            "voxel_size": self.voxel_size,
        }

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str) -> "Scaffold":
        """
        Rebuild a render-only model from a checkpoint.

        - no optimizers, schedulers or strategy

        Args:
            ckpt: loaded ckpt.pt.
            device: torch device string.

        Returns:
            The model on `device`.
        """
        model = cls.__new__(cls)
        config = ckpt["config"]
        model.cfg = ScaffoldConfig.from_dict(config.get("scaffold") or {})
        model.primitive = config["primitive"]
        model.device = device
        model.voxel_size = ckpt["voxel_size"]
        model.params = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(tensor)
                for name, tensor in dict(ckpt["splats"]).items()
            }
        ).to(device)

        # n_views from the saved appearance embedding, if any
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
        model.strategy = None  # type: ignore[assignment]
        return model


########################################
# Densification
########################################


class AnchorStrategy:
    """
    Grows anchors where gradients are high and prunes anchors that stay transparent.

    - statistics are per (anchor, offset) slot for growing, per anchor for pruning
    """

    def __init__(
        self,
        cfg: ScaffoldConfig,
        primitive: str,
        voxel_size: float,
        n_anchors: int,
        device: str,
    ):
        """
        Allocate the densification statistics.

        Args:
            cfg: scaffold config.
            primitive: "3dgs" or "2dgs".
            voxel_size: anchor voxel edge in world units.
            n_anchors: initial anchor count.
            device: torch device string.
        """
        self.cfg = cfg
        self.voxel_size = voxel_size
        self.device = device

        # Gradient key per primitive; the wrong one accumulates zeros
        self.key_for_gradient = "means2d" if primitive == "3dgs" else "gradient_2dgs"

        # Running sums and visit counts: per slot for growing, per anchor for pruning
        n_slots = n_anchors * cfg.n_offsets
        self.offset_gradient_accum = torch.zeros(n_slots, device=device)
        self.offset_denom = torch.zeros(n_slots, device=device)
        self.opacity_accum = torch.zeros(n_anchors, device=device)
        self.anchor_denom = torch.zeros(n_anchors, device=device)

    def accumulate(self, step: int, info: dict) -> None:
        """
        Add this step's gradients and decoded opacities to the running sums.

        - runs only for start_stat < step < update_until
        - gradients normalized as in gsplat @ d2f5c0f, gsplat/strategy/default.py:243-249
        - only rendered Gaussians count

        Args:
            step: current training step.
            info: gsplat info dict from `render`.
        """
        if not self.cfg.start_stat < step < self.cfg.update_until:
            return

        # No gradient (nothing rendered): skip the step
        grads = info[self.key_for_gradient].grad

        if grads is None:
            return

        grads = grads.detach().clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        grad_norm = grads.reshape(-1, 2).norm(dim=-1)

        index = info["decode_index"].to(self.device)

        # Gradient sums per slot, rendered Gaussians only; slots are unique, so unrendered ones add exactly 0
        rendered = (info["radii"].reshape(len(grad_norm), -1).amax(dim=-1) > 0).to(
            self.device
        )
        self.offset_gradient_accum.index_add_(
            0, index, torch.where(rendered, grad_norm.to(self.device), 0.0)
        )
        self.offset_denom.index_add_(0, index, rendered.to(self.offset_denom.dtype))

        # Opacity sums per anchor, divided later by visits
        anchor_index = torch.div(index, self.cfg.n_offsets, rounding_mode="floor")
        self.opacity_accum.index_add_(
            0, anchor_index, info["decoded_opacities"].detach().to(self.device)
        )
        visible = info["visible_ids"].to(self.device)
        self.anchor_denom.index_add_(
            0, visible, torch.ones_like(visible, dtype=self.anchor_denom.dtype)
        )

    def grow(self, scaffold: "Scaffold") -> int:
        """
        Add anchors in empty voxels near high-gradient slots.

        - coarse to fine: finer levels need higher gradients
        - a new anchor takes the max feature of the slots that seeded it

        Args:
            scaffold: model to grow in place.

        Returns:
            The number of anchors added.
        """
        n_offsets = self.cfg.n_offsets
        mean_grads = self.offset_gradient_accum / self.offset_denom.clamp_min(1.0)

        # Slots seen often enough in this window
        seen = (
            self.offset_denom > self.cfg.refine_every * self.cfg.success_threshold * 0.5
        )

        added_total = 0

        for level in range(self.cfg.update_depth):
            # Stop once a level adds nothing
            if level > 0 and added_total == 0:
                break

            threshold = self.cfg.grad_threshold * (
                (self.cfg.update_hierarchy_factor // 2) ** level
            )
            size_factor = max(
                self.cfg.update_init_factor
                // (self.cfg.update_hierarchy_factor**level),
                1,
            )
            level_voxel = self.voxel_size * size_factor
            selected = seen & (mean_grads >= threshold)

            # Randomly thin candidates, as upstream
            selected = selected & (torch.rand_like(mean_grads) > 0.5 ** (level + 1))

            if not bool(selected.any()):
                continue

            # Candidate positions: the slots' decoded means
            anchors = scaffold.params["anchors"].detach()
            offset_extent = torch.exp(scaffold.params["scaling"].detach()[:, :3])
            candidates = (
                anchors[:, None, :]
                + scaffold.params["offsets"].detach() * offset_extent[:, None, :]
            )
            candidates = candidates.reshape(-1, 3)[selected]

            # Feature of each candidate's source anchor
            slot_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
            source_feat = scaffold.params["anchor_feat"].detach()[
                torch.div(slot_ids, n_offsets, rounding_mode="floor")
            ]

            # Snap to this level's grid; drop cells already holding an anchor
            candidate_cells, cell_of_candidate = torch.unique(
                torch.round(candidates / level_voxel), dim=0, return_inverse=True
            )
            cell_feat = torch.zeros(
                len(candidate_cells), self.cfg.feat_dim, device=source_feat.device
            )
            cell_feat.index_reduce_(
                0, cell_of_candidate, source_feat, "amax", include_self=False
            )

            occupied_cells = torch.round(anchors / level_voxel)
            combined = torch.cat([occupied_cells, candidate_cells], dim=0)
            _, inverse, counts = torch.unique(
                combined, dim=0, return_inverse=True, return_counts=True
            )
            free = counts[inverse[len(occupied_cells) :]] == 1
            new_cells = candidate_cells[free]

            if len(new_cells) == 0:
                continue

            self._append_anchors(
                scaffold, new_cells * level_voxel, level_voxel, cell_feat[free]
            )
            added_total += len(new_cells)

            # Pad per-slot stats for the new anchors; new slots are unseen
            padding = torch.zeros(len(new_cells) * n_offsets, device=mean_grads.device)
            mean_grads = torch.cat([mean_grads, padding])
            seen = torch.cat([seen, padding.bool()])

        # Reset only the slots that were used
        self.offset_gradient_accum[seen] = 0.0
        self.offset_denom[seen] = 0.0

        if added_total:
            logger.debug(
                "scaffold: grew %d anchors -> %d",
                added_total,
                len(scaffold.params["anchors"]),
            )

        return added_total

    def _append_anchors(
        self,
        scaffold: "Scaffold",
        new_anchors: Tensor,
        level_voxel: float,
        new_feat: Tensor,
    ) -> None:
        """
        Append anchors to params, Adam state and statistics together.

        - zero offsets, given features, scaling from `level_voxel`
        """
        n_new = len(new_anchors)
        device = scaffold.params["anchors"].device
        log_voxel = math.log(level_voxel)
        additions = {
            "anchors": new_anchors.to(device),
            "offsets": torch.zeros(n_new, self.cfg.n_offsets, 3, device=device),
            "anchor_feat": new_feat.to(device),
            "scaling": torch.full((n_new, 6), log_voxel, device=device),
            "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(
                n_new, 1
            ),
        }

        # Extend each parameter and its Adam state; new rows start at zero
        def param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(
                torch.cat([param.detach(), additions[name]], dim=0)
            )

        def optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [value, torch.zeros((n_new, *value.shape[1:]), device=value.device)],
                dim=0,
            )

        _update_param_with_optimizer(
            param_fn, optimizer_fn, scaffold.params, scaffold.param_optimizers
        )

        # Pad the statistics: per slot and per anchor
        slot_padding = torch.zeros(n_new * self.cfg.n_offsets, device=self.device)
        anchor_padding = torch.zeros(n_new, device=self.device)
        self.offset_gradient_accum = torch.cat(
            [self.offset_gradient_accum, slot_padding]
        )
        self.offset_denom = torch.cat([self.offset_denom, slot_padding.clone()])
        self.opacity_accum = torch.cat([self.opacity_accum, anchor_padding])
        self.anchor_denom = torch.cat([self.anchor_denom, anchor_padding.clone()])

    def prune(self, scaffold: "Scaffold", *, scale_cap: float = 0.05) -> int:
        """
        Remove anchors whose mean opacity stayed below min_opacity.

        - only anchors seen often enough are judged
        - always keeps at least one anchor

        Args:
            scaffold: model to prune in place.
            scale_cap: cap on the log Gaussian extent.

        Returns:
            The number of anchors removed.
        """
        n_offsets = self.cfg.n_offsets
        denom = self.anchor_denom
        mean_opacity = self.opacity_accum / denom.clamp_min(1.0)
        seen = denom > self.cfg.refine_every * self.cfg.success_threshold
        drop = seen & (mean_opacity < self.cfg.min_opacity)

        # Reset the anchors that were judged
        self.opacity_accum[seen] = 0.0
        self.anchor_denom[seen] = 0.0

        # Cap the Gaussian extent every refine
        with torch.no_grad():
            scaffold.params["scaling"][:, 3:].clamp_(max=scale_cap)

        if not bool(drop.any()):
            return 0

        # Never prune to empty: keep the most opaque anchor
        if bool(drop.all()):
            drop[mean_opacity.argmax()] = False
            logger.warning(
                "scaffold: every anchor fell below min_opacity %.4g; kept the most opaque one",
                self.cfg.min_opacity,
            )

        keep = ~drop
        keep_slots = keep.repeat_interleave(n_offsets)

        # Drop rows from each parameter and its Adam state
        def param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param.detach()[keep])

        def optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return value[keep]

        _update_param_with_optimizer(
            param_fn, optimizer_fn, scaffold.params, scaffold.param_optimizers
        )

        self.offset_gradient_accum = self.offset_gradient_accum[keep_slots]
        self.offset_denom = self.offset_denom[keep_slots]
        self.opacity_accum = self.opacity_accum[keep]
        self.anchor_denom = self.anchor_denom[keep]

        n_dropped = int(drop.sum())
        logger.debug(
            "scaffold: pruned %d anchors -> %d",
            n_dropped,
            len(scaffold.params["anchors"]),
        )
        return n_dropped

    def refine(self, scaffold: "Scaffold", step: int) -> None:
        """
        Grow then prune every `refine_every` steps between update_from and update_until.

        Args:
            scaffold: model to refine in place.
            step: current training step.
        """
        if step <= self.cfg.update_from or step >= self.cfg.update_until:
            return

        if step % self.cfg.refine_every != 0:
            return

        self.grow(scaffold)
        self.prune(scaffold)
