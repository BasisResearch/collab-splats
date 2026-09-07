"""
Vanilla Gaussian primitives: one Gaussian per seed point, densified by a gsplat strategy.

- ``Gaussians`` owns its parameters, optimizers, strategy and rendering, so the trainer drives it
  and ``Scaffold`` through the same members.
- Tunable literals are keyword-only arguments; ``SH_C0`` is the one module-level constant.
"""

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

from collab_splats.splats.rendering import render_gaussians

# Type-only: trainer imports this module, so a runtime import back would be circular
if TYPE_CHECKING:
    from collab_splats.splats.trainer import SplatsConfig

logger = logging.getLogger(__name__)

# Degree-0 spherical harmonic: rgb = SH_C0 * sh0 + 0.5
# - sole definition; bit-identical (0x3fd20dd750429b6d, 0 ULP) to the retired rendering.py copy
# - do not rewrite the literal: it changes the bytes of every existing splats.ply
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
    MCMC for 3dgs (budgeted, no gradient heuristics); Default with splatfacto's args for 2dgs.

    Args:
        cfg: the run's SplatsConfig (reads `primitive`, `cap_max`, `grow_grad2d`).
        n_views: number of training views; sets DefaultStrategy's post-reset refine pause.
        prune_opa: opacity below which a Gaussian is pruned (2dgs only).
        prune_scale3d: world-scale above which a Gaussian is pruned (2dgs only).
        refine_scale2d_stop_iter: step after which screen-scale splitting stops (2dgs only).

    Returns:
        An unstarted gsplat strategy; the caller still calls `initialize_state`.
    """
    if cfg.primitive == "3dgs":
        return MCMCStrategy(cap_max=cfg.cap_max, verbose=False)

    # splatfacto's non-default DefaultStrategy args
    # - upstream: nerfstudio-project/nerfstudio @ 50e0e3c, splatfacto.py:264-280
    # - absgrad=False: 2dgs backward writes .absgrad on means2d, not gradient_2dgs
    # - grow_grad2d 2e-4: measured-good non-absgrad threshold
    defaults = DefaultStrategy()

    # Trap: n_views + 100 never refines once n_views >= reset_every - 100 (2900 default) — cap it
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
    Vanilla 3DGS / 2DGS primitives with their optimizers, strategy and rendering.

    - Exposes the same members as ``Scaffold`` — `params`, `optimizers`, `schedulers`,
      `n_primitives`, `primitive_unit`, `render`, `pre_backward`, `post_backward`, `denormalize`,
      `export_gaussians`, `frame_report`, `checkpoint`, `from_checkpoint` — so the trainer needs
      no branch.
    - `activate` is ``Gaussians``-only: ``Scaffold`` runs the same raw-to-rasterizer step inside
      ``decode``, per view, so it has no view-independent equivalent to expose.
    """

    # What this representation counts, for the writer's log line; a class attribute, not a check
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
        One Gaussian per seed point: scale from kNN spacing, color as SH DC, one Adam per parameter.

        - port of create_splats_with_optimizers: nerfstudio-project/gsplat @ d2f5c0f,
          examples/simple_trainer.py:285

        Args:
            cfg: the run's SplatsConfig; read here, never retained.
            points: (N, 3) float seed positions in the training frame. Needs at least `knn` points.
            colors: (N, 3) uint8 seed colors.
            scene_scale: camera extent; scales the means lr and seeds the strategy.
            n_views: number of training views; sets the 2dgs refine pause.
            device: torch device string.
            knn: neighbors (including self) used for the initial scale; 4 means the 3 nearest.
            adam_eps: Adam epsilon, 1e-15 as upstream.
            lr_decay: total decay of the means lr over the run, 0.01 = 100x down.
        """
        # Plain attributes, not a cfg reference: from_checkpoint has only a dict to match
        self.primitive = cfg.primitive
        self.sh_degree = cfg.sh_degree
        self.sh_degree_interval = cfg.sh_degree_interval
        self.device = device

        n_points = len(points)

        # Initial scale: mean distance to the (knn - 1) nearest neighbors, stored as log-scale
        neighbor_dists, _ = NearestNeighbors(n_neighbors=knn).fit(points).kneighbors(points)
        neighbor_sq_dists = neighbor_dists[:, 1:] ** 2
        mean_spacing = np.sqrt(neighbor_sq_dists.mean(-1))
        spacing = torch.from_numpy(mean_spacing).float()
        log_scales = torch.log(spacing).unsqueeze(-1).repeat(1, 3)

        # Color: RGB goes into the degree-0 SH band, higher bands start at zero
        rgb = torch.from_numpy(colors).float() / 255.0
        n_sh_coeffs = (cfg.sh_degree + 1) ** 2
        sh_coeffs = torch.zeros(n_points, n_sh_coeffs, 3)
        sh_coeffs[:, 0, :] = (rgb - 0.5) / SH_C0

        # Raw parameters: random orientation, logit-opacity so sigmoid gives init_opacity
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

        # One Adam per parameter so the densification strategy can grow/prune optimizer state per tensor
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

        # Only the positions decay; the other groups hold their lr for the whole run
        self.means_scheduler = ExponentialLR(self.param_optimizers["means"], gamma=lr_decay ** (1.0 / cfg.max_steps))
        self.schedulers = [self.means_scheduler]

        # Densification: MCMC is budgeted and stateless, Default carries per-Gaussian statistics
        self.strategy = make_strategy(cfg, n_views)
        self.strategy.check_sanity(self.params, self.param_optimizers)
        if isinstance(self.strategy, MCMCStrategy):
            self.strategy_state = self.strategy.initialize_state()
        else:
            self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)

    @property
    def n_primitives(self) -> int:
        """
        Number of Gaussians currently in the model.

        Returns:
            Row count of ``params["means"]``.
        """
        return len(self.params["means"])

    def activate(self) -> dict[str, Tensor]:
        """
        Activate raw parameters into the tensors the rasterizer takes.

        - log-scales -> scales, logit-opacities -> opacities, SH bands concatenated
        - `colors` stays SH: the caller passes an integer `sh_degree` to the rasterizer

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
        render_plane: bool = False,
    ) -> tuple[dict[str, Tensor], dict]:
        """
        Rasterize one view.

        - re-entrant: the trainer renders more than once per step (PGSR neighbor view)
        - per-render state travels in the returned `info`, never on `self`

        Args:
            cam_to_world: (1, 4, 4) camera-to-world pose in the training frame.
            intrinsics: (1, 3, 3) camera matrix in pixels at this render's resolution.
            width: render width in pixels.
            height: render height in pixels.
            camera_id: (1,) long view index. Unused here; ``Scaffold`` needs it for appearance.
            step: current training step, which unlocks SH bands progressively. `None` renders at
                the full `sh_degree` and is export-only — never pass it as a "don't care".
            render_normals: render the per-Gaussian normal and its finite-differenced partner.
            render_plane: add PGSR's planar signals (3dgs only).

        Returns:
            (render dict, gsplat strategy info dict) — see `rendering.render_gaussians`.
        """
        sh_degree = self.sh_degree if step is None else min(step // self.sh_degree_interval, self.sh_degree)

        # Never absolute gradients
        # - make_strategy pins absgrad=False for 2dgs; MCMC has none; a restored model has none
        # - flipping it there must move this literal: test_render_never_asks_for_absolute_gradients
        absgrad = False

        return render_gaussians(
            self.primitive,
            self.activate(),
            cam_to_world,
            intrinsics,
            width,
            height,
            sh_degree,
            absgrad,
            render_normals=render_normals,
            render_plane=render_plane,
        )

    def frame_report(self, render: dict[str, Tensor]) -> dict[str, int]:
        """
        Per-view fields this representation adds to splats_quality_report.json.

        Args:
            render: one view's render dict, as `render` returned it. Unused here.

        Returns:
            {} — the same Gaussians render every view, and that count is already
            `summary.n_gaussians`; the keys are absent from a vanilla report, not zero.
        """
        return {}

    def pre_backward(self, step: int, info: dict) -> None:
        """
        Retain the screen-space gradients DefaultStrategy densifies on. No-op under MCMC.

        Args:
            step: current training step.
            info: the gsplat info dict this step's render returned.
        """
        # MCMC inherits a no-op pre-backward hook, so the guard is documentation, not correctness
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_pre_backward(self.params, self.param_optimizers, self.strategy_state, step, info)

    def post_backward(self, step: int, info: dict) -> None:
        """
        Densify / prune / relocate, after the optimizer step.

        - step the optimizers first (upstream order): refine ops rebuild the Parameters, so a
          later step sees `.grad=None` and silently skips
        - MCMC reads the post-decay means lr

        Args:
            step: current training step.
            info: the *main* render's gsplat info dict, never a neighbor view's.
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
        Undo ``utils.scene_normalization`` on the Gaussians, in place.

        Args:
            center: (3,) the center `scene_normalization` returned.
            scale: the scale `scene_normalization` returned.

        Returns:
            None — `params` is modified in place.
        """
        center_t = torch.as_tensor(center, dtype=torch.float32, device=self.params["means"].device)
        with torch.no_grad():
            self.params["means"].data = self.params["means"].data / scale + center_t
            self.params["scales"].data = self.params["scales"].data - math.log(scale)

    def export_gaussians(self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int) -> dict[str, Tensor]:
        """
        The raw parameters the ply writer wants.

        - all four arguments unused here: only ``Scaffold`` reads them, and both are called
          through this one signature

        Args:
            cam_to_world: (N, 4, 4) training poses, BATCHED over every view — unlike `render`,
                which takes the single (1, 4, 4) view it rasterizes.
            intrinsics: (N, 3, 3) camera matrices, batched to match.
            width: image width in pixels.
            height: image height in pixels.

        Returns:
            {"means", "scales", "quats", "opacities", "sh0", "shN"} — the parameters themselves,
            not copies. Scales are log, opacities logits, colors SH coefficients.
        """
        return dict(self.params)

    def checkpoint(self) -> dict:
        """
        The model half of ckpt.pt.

        Returns:
            {"splats": ParameterDict}. The trainer adds `config`, cameras and image ids.
        """
        return {"splats": self.params}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str) -> "Gaussians":
        """
        Rebuild a render-only model from a checkpoint.

        - no optimizers, schedulers or strategy: renders and exports, does not train.

        Args:
            ckpt: a loaded ckpt.pt holding `splats` and a plain-dict `config`.
            device: torch device string.

        Returns:
            A Gaussians instance whose `params` are the checkpoint's, on `device`.
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
