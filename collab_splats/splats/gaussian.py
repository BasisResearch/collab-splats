"""
Plain Gaussian splats: one Gaussian per seed point.

- `make_strategy`: the gsplat densification strategy per primitive
- `Gaussians`: parameters, optimizers and rendering, same interface as `Scaffold`
"""

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

from collab_splats.splats.rendering import render_gaussians
from collab_splats.splats.utils import knn_spacing

# Annotation-only: a runtime import of the trainer would be circular
if TYPE_CHECKING:
    from collab_splats.splats.trainer import SplatsConfig

logger = logging.getLogger(__name__)

# Degree-0 SH constant: rgb = SH_C0 * sh0 + 0.5; changing it changes every splats.ply
SH_C0 = 0.5 / math.sqrt(math.pi)


########################################
# Densification strategy
########################################


def make_strategy(
    cfg: "SplatsConfig",
    n_views: int,
    *,
    prune_opa: float = 0.1,
    prune_scale3d: float = 0.5,
    refine_scale2d_stop_iter: int = 4000,
) -> MCMCStrategy | DefaultStrategy:
    """
    Densification strategy: MCMC for 3dgs, Default with splatfacto's settings for 2dgs.

    Args:
        cfg: run config; reads `primitive`, `cap_max`, `grow_grad2d`.
        n_views: number of training views; sets the 2dgs refine pause.
        prune_opa: prune below this opacity (2dgs only).
        prune_scale3d: prune above this world scale (2dgs only).
        refine_scale2d_stop_iter: step after which screen-scale splitting stops (2dgs only).

    Returns:
        The strategy; the caller still calls `initialize_state`.
    """
    if cfg.primitive == "3dgs":
        return MCMCStrategy(cap_max=cfg.cap_max, verbose=False)

    # splatfacto's DefaultStrategy settings
    # - upstream: nerfstudio-project/nerfstudio @ 50e0e3c, splatfacto.py:264-280
    # - absgrad off: 2dgs needs gradient_2dgs, so grow_grad2d is 2e-4 not 8e-4
    defaults = DefaultStrategy()

    # Cap the refine pause, or large view counts never refine
    pause = n_views + 100
    max_pause = defaults.reset_every - defaults.refine_every

    if pause > max_pause:
        logger.warning(
            "pause_refine_after_reset=%d (n_views+100) >= reset_every=%d would never refine; capped to %d",
            pause,
            defaults.reset_every,
            max_pause,
        )
        pause = max_pause

    return DefaultStrategy(
        absgrad=False,
        grow_grad2d=cfg.grow_grad2d,
        key_for_gradient="gradient_2dgs",
        prune_opa=prune_opa,
        prune_scale3d=prune_scale3d,
        refine_scale2d_stop_iter=refine_scale2d_stop_iter,
        pause_refine_after_reset=pause,
        verbose=False,
    )


########################################
# Model
########################################


class Gaussians:
    """
    3DGS / 2DGS Gaussians with their optimizers, strategy and rendering.

    - same members as `Scaffold`, so the trainer treats both alike
    - `activate` exists only here; `Scaffold` decodes per view instead
    """

    # What this model counts, for log lines
    primitive_unit: str = "gaussians"

    def __init__(
        self,
        cfg: "SplatsConfig",
        points: np.ndarray,
        colors: np.ndarray,
        scene_scale: float,
        n_views: int,
        device: str,
        *,
        knn: int = 4,
        adam_eps: float = 1e-15,
        lr_decay: float = 0.01,
    ):
        """
        One Gaussian per seed point, sized by neighbor spacing, one optimizer per parameter.

        - ported from gsplat @ d2f5c0f, examples/simple_trainer.py:285 (`create_splats_with_optimizers`)

        Args:
            cfg: run config; not kept.
            points: (N, 3) seed positions in the training frame; N >= `knn`.
            colors: (N, 3) uint8 seed colors.
            scene_scale: camera extent; scales the means lr and the strategy.
            n_views: number of training views.
            device: torch device string.
            knn: neighbors for the initial scale, counting the point itself.
            adam_eps: Adam epsilon.
            lr_decay: total means-lr decay over the run (0.01 = 100x).
        """
        # Copy the config fields; from_checkpoint rebuilds them from a dict
        self.primitive = cfg.primitive
        self.sh_degree = cfg.sh_degree
        self.sh_degree_interval = cfg.sh_degree_interval
        self.device = device

        n_points = len(points)

        # Initial log-scale from nearest-neighbor spacing
        spacing = torch.from_numpy(knn_spacing(points, knn - 1)).float()
        log_scales = torch.log(spacing).unsqueeze(-1).repeat(1, 3)

        # Color into the degree-0 SH band; higher bands start at zero
        rgb = torch.from_numpy(colors).float() / 255.0
        n_sh_coeffs = (cfg.sh_degree + 1) ** 2
        sh_coeffs = torch.zeros(n_points, n_sh_coeffs, 3)
        sh_coeffs[:, 0, :] = (rgb - 0.5) / SH_C0

        # Raw parameters: random orientation, logit opacity
        initial_opacities = torch.logit(torch.full((n_points,), cfg.init_opacity))
        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.from_numpy(points).float()),
                "scales": torch.nn.Parameter(log_scales),
                "quats": torch.nn.Parameter(torch.rand(n_points, 4)),
                "opacities": torch.nn.Parameter(initial_opacities),
                "sh0": torch.nn.Parameter(sh_coeffs[:, :1, :]),
                "shN": torch.nn.Parameter(sh_coeffs[:, 1:, :]),
            }
        ).to(device)

        # One Adam per parameter, so the strategy can grow and prune each
        learning_rates = {
            "means": cfg.means_lr * scene_scale,
            "scales": cfg.scales_lr,
            "quats": cfg.quats_lr,
            "opacities": cfg.opacities_lr,
            "sh0": cfg.sh0_lr,
            "shN": cfg.shN_lr,
        }
        self.param_optimizers = {
            name: torch.optim.Adam([{"params": self.params[name], "lr": lr, "name": name}], eps=adam_eps)
            for name, lr in learning_rates.items()
        }
        self.optimizers = list(self.param_optimizers.values())

        # Only the means lr decays
        self.means_scheduler = ExponentialLR(self.param_optimizers["means"], gamma=lr_decay ** (1.0 / cfg.max_steps))
        self.schedulers = [self.means_scheduler]

        # Densification strategy and its state
        self.strategy = make_strategy(cfg, n_views)
        self.strategy.check_sanity(self.params, self.param_optimizers)

        if isinstance(self.strategy, MCMCStrategy):
            self.strategy_state = self.strategy.initialize_state()
        else:
            self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)

    @property
    def n_primitives(self) -> int:
        """
        Number of Gaussians in the model.

        Returns:
            The Gaussian count.
        """
        return len(self.params["means"])

    def activate(self) -> dict[str, Tensor]:
        """
        Raw parameters converted to what the rasterizer takes.

        - `colors` are still SH coefficients

        Returns:
            {"means" (N,3), "quats" (N,4), "scales" (N,3), "opacities" (N,), "colors" (N,K,3)}.
        """
        return {
            "means": self.params["means"],
            "quats": self.params["quats"],
            "scales": torch.exp(self.params["scales"]),
            "opacities": torch.sigmoid(self.params["opacities"]),
            "colors": torch.cat([self.params["sh0"], self.params["shN"]], dim=1),
        }

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
        Render one view.

        Args:
            cam_to_world: (1, 4, 4) pose in the training frame.
            intrinsics: (1, 3, 3) camera matrix at the render resolution.
            width: render width in pixels.
            height: render height in pixels.
            camera_id: (1,) view index; unused here.
            step: training step, which unlocks SH bands; None uses all bands (export only).
            render_normals: also render normals and depth normals.

        Returns:
            (render dict, gsplat info dict), as in `rendering.render_gaussians`.
        """
        sh_degree = self.sh_degree if step is None else min(step // self.sh_degree_interval, self.sh_degree)

        return render_gaussians(
            self.primitive,
            self.activate(),
            cam_to_world,
            intrinsics,
            width,
            height,
            sh_degree,
            render_normals=render_normals,
        )

    def frame_report(self, render: dict[str, Tensor]) -> dict[str, int]:
        """
        Per-view fields for the quality report.

        Args:
            render: one view's render dict; unused here.

        Returns:
            An empty dict; plain Gaussians add no per-view fields.
        """
        return {}

    def pre_backward(self, step: int, info: dict) -> None:
        """
        Keep the screen-space gradients DefaultStrategy needs; no-op under MCMC.

        Args:
            step: current training step.
            info: the gsplat info dict this step's render returned.
        """
        # Only DefaultStrategy uses the pre-backward hook
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_pre_backward(self.params, self.param_optimizers, self.strategy_state, step, info)

    def post_backward(self, step: int, info: dict) -> None:
        """
        Densify, prune or relocate Gaussians.

        - call after the optimizer step, or that step is silently skipped

        Args:
            step: current training step.
            info: the gsplat info dict this step's render returned.
        """
        if isinstance(self.strategy, MCMCStrategy):
            means_lr = self.means_scheduler.get_last_lr()[0]
            self.strategy.step_post_backward(
                self.params, self.param_optimizers, self.strategy_state, step, info, lr=means_lr
            )
        else:
            self.strategy.step_post_backward(
                self.params, self.param_optimizers, self.strategy_state, step, info, packed=False
            )

    def denormalize(self, center: np.ndarray, scale: float) -> None:
        """
        Undo scene normalization on the Gaussians, in place.

        Args:
            center: (3,) the center `utils.scene_normalization` returned.
            scale: the scale `utils.scene_normalization` returned.
        """
        center_t = torch.as_tensor(center, dtype=torch.float32, device=self.params["means"].device)

        with torch.no_grad():
            self.params["means"].data = self.params["means"].data / scale + center_t
            self.params["scales"].data = self.params["scales"].data - math.log(scale)

    def export_gaussians(self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int) -> dict[str, Tensor]:
        """
        Raw parameters for the ply writer.

        - the camera arguments are unused here; `Scaffold` needs them

        Args:
            cam_to_world: (N, 4, 4) poses of all training views.
            intrinsics: (N, 3, 3) camera matrices.
            width: image width in pixels.
            height: image height in pixels.

        Returns:
            The parameters themselves (not copies): log scales, logit opacities, SH colors.
        """
        return dict(self.params)

    def checkpoint(self) -> dict:
        """
        The model's part of ckpt.pt.

        Returns:
            {"splats": params}; the trainer adds the rest.
        """
        return {"splats": self.params}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str) -> "Gaussians":
        """
        Rebuild a render-only model from a checkpoint.

        - no optimizers or strategy, so it cannot train

        Args:
            ckpt: loaded ckpt.pt with `splats` and `config`.
            device: torch device string.

        Returns:
            The model on `device`.
        """
        model = cls.__new__(cls)
        config = ckpt["config"]
        model.primitive = config["primitive"]
        model.sh_degree = config["sh_degree"]
        model.sh_degree_interval = config["sh_degree_interval"]
        model.device = device
        model.params = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(tensor) for name, tensor in dict(ckpt["splats"]).items()}
        ).to(device)
        model.param_optimizers = {}
        model.optimizers = []
        model.schedulers = []
        model.means_scheduler = None
        model.strategy = None
        model.strategy_state = None
        return model
