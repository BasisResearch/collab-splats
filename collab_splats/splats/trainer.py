"""
Trainer for 3DGS / 2DGS splats on upstream gsplat.

Training runs in the COLMAP world frame by default so poses, depth and the ply line up with
every other stage artifact. ``scene_scale`` — 1.1 x the largest camera distance from the
camera centroid, as in gsplat's simple_trainer — only scales the means learning rate, the
densification thresholds and the depth loss. With ``normalize_scene`` the cameras, seed points
and depth targets are instead Sim3-normalised the splatfacto way (centre on the camera-position
mean, scale so the largest |camera coordinate| is 1) and ``scene_scale`` is fixed to 1.0; the
outputs are mapped back to world units before writing, so downstream stages never see the
normalised frame.

Densification: ``MCMCStrategy`` for 3DGS (fixed budget ``cap_max``, dead Gaussians relocated,
no gradient heuristics — upstream's ``mcmc`` preset), ``DefaultStrategy`` for 2DGS (the only
pairing upstream ships; prunes opacity < 0.005 and oversized Gaussians, resets opacity every 3000).
"""

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

from collab_splats.splats.cameras import CameraOptModule
from collab_splats.splats.losses import OPTIONAL_LOSSES, compute_losses, loss_active
from collab_splats.splats.outputs import write_splat_outputs
from collab_splats.splats.rendering import render_view
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)

PRIMITIVES = ("3dgs", "2dgs")
SH_DC_NORMALISER = 0.28209479177387814  # rgb -> SH degree-0 coefficient (1 / (2 sqrt(pi)))

########################################
# Config — every tunable, with gsplat simple_trainer defaults
########################################


def _default_losses(primitive: str) -> dict[str, dict]:
    """
    Default loss schedule per primitive: MCMC regularisers for 3dgs, distortion for 2dgs.
    """
    losses = {"depth": {"weight": 0.01}, "normal_consistency": {"weight": 0.05, "start": 7000}}
    if primitive == "3dgs":
        losses["opacity_reg"] = {"weight": 0.01}
        losses["scale_reg"] = {"weight": 0.01}
    else:
        losses["distortion"] = {"weight": 0.01, "start": 3000}
    return losses


@dataclass
class SplatsConfig:
    """
    Trainer knobs; mirrors the ``splats:`` yaml block minus ``enabled``. Defaults follow gsplat's examples.
    """

    primitive: str = "3dgs"
    max_steps: int = 30000
    pose_opt: bool = True
    losses: dict[str, dict] | None = None  # None -> _default_losses(primitive)

    # Appearance
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # one more SH band unlocked every this many steps
    init_opacity: float = 0.1

    # Learning rates (means_lr and pose_lr are multiplied by scene_scale; means/pose decay 0.01x over the run)
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 1.25e-4
    pose_lr: float = 1e-5

    # Densification
    cap_max: int = 1_000_000  # 3dgs (MCMC): Gaussian budget
    # 2dgs (Default): 2D-gradient threshold to split/duplicate. gsplat's non-absgrad default;
    # the spec's 8e-4 was calibrated for absgrad=True and starved densification (GH010229:
    # 160k vs 1.4M gaussians, -1.26 dB).
    grow_grad2d: float = 2e-4

    log_every: int = 500

    # Coarse-to-fine (splatfacto): start at 1/2^num_downscales resolution, double
    # every resolution_schedule steps. 0 disables.
    num_downscales: int = 2
    resolution_schedule: int = 3000

    # splatfacto scene normalisation (nerfstudio center_method="poses" + auto_scale_poses):
    # train in a unit-cube frame with scene_scale = 1.0 instead of world units x scene_scale.
    normalize_scene: bool = False

    def __post_init__(self):
        if self.losses is None:
            self.losses = _default_losses(self.primitive)

        # At least one step: lr_gamma divides by max_steps and the report's final_losses needs a last step
        if self.max_steps < 1:
            raise ValueError(f"splats.max_steps must be >= 1, got {self.max_steps}")

    @classmethod
    def from_dict(cls, block: dict) -> "SplatsConfig":
        """
        Build from the yaml block; rejects unknown keys, unknown/ill-formed losses, and distortion with 3dgs.

        - A given ``losses`` mapping REPLACES the per-primitive defaults (no merge).
        """
        # Unknown top-level keys are almost always typos — refuse rather than silently default
        allowed_keys = {"enabled", *cls.__dataclass_fields__}
        unknown_keys = set(block) - allowed_keys
        if unknown_keys:
            raise ValueError(f"splats: unknown keys {sorted(unknown_keys)}; allowed {sorted(allowed_keys)}")
        fields = {key: value for key, value in block.items() if key != "enabled"}
        cfg = cls(**fields)

        # Primitive and loss entries must be ones the trainer knows, each with a weight
        if cfg.primitive not in PRIMITIVES:
            raise ValueError(f"splats.primitive must be one of {PRIMITIVES}, got '{cfg.primitive}'")
        for name, spec in cfg.losses.items():
            if name not in OPTIONAL_LOSSES:
                raise ValueError(f"splats.losses: unknown loss '{name}'; allowed {sorted(OPTIONAL_LOSSES)}")
            unknown_spec_keys = set(spec) - {"weight", "start"}
            if unknown_spec_keys or "weight" not in spec:
                raise ValueError(f"splats.losses.{name}: expected {{weight[, start]}}, got {sorted(spec)}")

        # The distortion map only exists for 2DGS
        distortion_spec = cfg.losses.get("distortion", {})
        distortion_weight = distortion_spec.get("weight", 0.0)
        if cfg.primitive == "3dgs" and distortion_weight > 0:
            raise ValueError("splats.losses.distortion is 2dgs-only; set its weight to 0 or use primitive: 2dgs")
        return cfg


########################################
# Setup helpers
########################################


def compute_scene_scale(cam_to_world: Tensor) -> float:
    """
    1.1 x the largest camera distance from the camera centroid (gsplat's scene-extent proxy).
    """
    positions = cam_to_world[:, :3, 3]
    centroid = positions.mean(0)
    spread = (positions - centroid).norm(dim=-1).max()
    return float(spread) * 1.1


def scene_normalization(cam_to_world: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Splatfacto's Sim3: centre = mean camera position, scale = 1 / max |camera coordinate - centre|.

    - Matches nerfstudio ``center_method="poses"`` + ``auto_scale_poses`` (L-inf, not L2); the
      "up" re-orientation is skipped because no loss or lr depends on the world's rotation.
    """
    positions = cam_to_world[:, :3, 3]
    center = positions.mean(0)
    spread = float(np.abs(positions - center).max())
    if spread <= 0:
        raise ValueError("splats: cannot normalise a scene whose cameras coincide")
    return center.astype(np.float32), 1.0 / spread


def denormalize_outputs(
    gaussians: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    cam_to_world: Tensor,
    center: np.ndarray,
    scale: float,
) -> None:
    """
    Undo ``scene_normalization`` in place on the trained Gaussians, cameras and pose deltas.

    - means / camera translations: p / scale + center; log-scales: - log(scale).
    - Pose-refiner translation deltas live in the camera frame (c2w @ delta), so they only
      need the 1 / scale; rotations are untouched.
    """
    center_t = torch.as_tensor(center, dtype=torch.float32, device=cam_to_world.device)
    with torch.no_grad():
        gaussians["means"].data = gaussians["means"].data / scale + center_t
        gaussians["scales"].data = gaussians["scales"].data - math.log(scale)
        cam_to_world[:, :3, 3] = cam_to_world[:, :3, 3] / scale + center_t
        if pose_refiner is not None:
            pose_refiner.embeds.weight[:, :3] /= scale


def init_gaussians_from_points(
    cfg: SplatsConfig, points: np.ndarray, colors: np.ndarray, scene_scale: float, device: str
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer]]:
    """
    One Gaussian per seed point (scale from kNN spacing, colour as SH DC) plus one Adam per parameter.

    - Requires >= 4 points (kNN over the 3 nearest neighbours).

    Port of create_splats_with_optimizers, gsplat @ d2f5c0f examples/simple_trainer.py.
    """
    n_points = len(points)

    # Initial scale: mean distance to the 3 nearest neighbours, stored as log-scale
    neighbour_dists, _ = NearestNeighbors(n_neighbors=4).fit(points).kneighbors(points)
    neighbour_sq_dists = neighbour_dists[:, 1:] ** 2
    mean_spacing = np.sqrt(neighbour_sq_dists.mean(-1))
    spacing = torch.from_numpy(mean_spacing).float()
    log_scales = torch.log(spacing).unsqueeze(-1).repeat(1, 3)

    # Colour: RGB goes into the degree-0 SH band, higher bands start at zero
    rgb = torch.from_numpy(colors).float() / 255.0
    n_sh_coeffs = (cfg.sh_degree + 1) ** 2
    sh_coeffs = torch.zeros(n_points, n_sh_coeffs, 3)
    sh_coeffs[:, 0, :] = (rgb - 0.5) / SH_DC_NORMALISER
    sh0 = sh_coeffs[:, :1, :]
    shN = sh_coeffs[:, 1:, :]

    # Raw parameters: random orientation, logit-opacity so sigmoid gives init_opacity
    initial_opacities = torch.logit(torch.full((n_points,), cfg.init_opacity))
    means = torch.from_numpy(points).float()
    gaussians = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(log_scales),
            "quats": torch.nn.Parameter(torch.rand(n_points, 4)),
            "opacities": torch.nn.Parameter(initial_opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
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
    optimizers = {}
    for name, lr in learning_rates.items():
        param_group = {"params": gaussians[name], "lr": lr, "name": name}
        optimizers[name] = torch.optim.Adam([param_group], eps=1e-15)
    return gaussians, optimizers


class ViewSampler:
    """
    Splatfacto's view schedule: seeded shuffled permutation, popped until empty, reshuffled.

    - Guarantees every view trains max_steps/n_views (+-1) times, vs +-17%
      spread from torch.randint sampling with replacement.
    - Port of nerfstudio @ 50e0e3c full_images_datamanager (random.Random shuffle + pop).
    """

    def __init__(self, n_views: int, seed: int = 42):
        self._rng = random.Random(seed)
        self._n_views = n_views
        self._pending: list[int] = []

    def next(self) -> int:
        """
        Next view index; reshuffles a fresh permutation when the epoch empties.
        """
        if not self._pending:
            self._pending = list(range(self._n_views))
            self._rng.shuffle(self._pending)
        return self._pending.pop()


def make_strategy(cfg: SplatsConfig, n_views: int) -> MCMCStrategy | DefaultStrategy:
    """
    MCMC for 3dgs (budgeted, no gradient heuristics); Default with splatfacto's args for 2dgs.
    """
    if cfg.primitive == "3dgs":
        return MCMCStrategy(cap_max=cfg.cap_max, verbose=False)

    # splatfacto (nerfstudio @ 50e0e3c) non-default DefaultStrategy args. absgrad stays
    # False: the 2dgs backward writes .absgrad on means2d only, never on the
    # gradient_2dgs densify tensor this strategy reads (see the parity spec's verdict),
    # and grow_grad2d 2e-4 is the measured-good non-absgrad threshold.
    # gsplat gates refine on `step % reset_every >= pause_refine_after_reset`, so
    # splatfacto's n_views + 100 silently disables densification once n_views
    # >= reset_every - 100 (2900 at gsplat defaults). Cap it and say so.
    defaults = DefaultStrategy()
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
        prune_opa=0.1,
        prune_scale3d=0.5,
        refine_scale2d_stop_iter=4000,
        pause_refine_after_reset=pause,
        verbose=False,
    )


def make_pose_refiner(
    cfg: SplatsConfig, n_views: int, scene_scale: float, lr_gamma: float, device: str
) -> tuple[CameraOptModule, torch.optim.Optimizer, ExponentialLR]:
    """
    Zero-initialised CameraOptModule with its Adam optimizer and exponential lr decay.
    """
    refiner = CameraOptModule(n_views).to(device)
    refiner.zero_init()
    pose_lr = cfg.pose_lr * scene_scale
    # weight_decay=1e-6 as in gsplat @ d2f5c0f examples/simple_trainer.py pose_optimizers
    optimizer = torch.optim.Adam(refiner.parameters(), lr=pose_lr, weight_decay=1e-6)
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)
    return refiner, optimizer, scheduler


def downscale_factor(step: int, num_downscales: int, resolution_schedule: int) -> int:
    """
    Coarse-to-fine divisor at a step: 2^max(0, num_downscales - step // resolution_schedule).
    """
    if num_downscales <= 0:
        return 1
    return 2 ** max(0, num_downscales - step // resolution_schedule)


def downscale_view(image: np.ndarray, intrinsics: Tensor, factor: int):
    """
    Image (bilinear) and K scaled by 1/factor; passthrough at factor 1.
    """
    if factor == 1:
        return image, intrinsics

    height, width = image.shape[:2]
    small = cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR)
    K_small = intrinsics.clone()
    K_small[:, :2, :] /= factor
    return small, K_small


def prepare_training_target(image: np.ndarray, depth_target: np.ndarray | None, device: str) -> dict:
    """
    One view's targets as tensors: rgb (1, H, W, 3) in [0, 1]; depth (1, H, W, 1) resized nearest, or None.
    """
    rgb = torch.from_numpy(image).to(device).float()[None] / 255.0
    if depth_target is None:
        return {"rgb": rgb, "depth": None}

    # Depth targets may be at model resolution; nearest resize keeps zeros (no target) as zeros
    height, width = image.shape[:2]
    depth_nchw = torch.from_numpy(depth_target).to(device)[None, None]
    depth_nchw = F.interpolate(depth_nchw, size=(height, width), mode="nearest")
    depth_nhwc = depth_nchw.permute(0, 2, 3, 1)
    return {"rgb": rgb, "depth": depth_nhwc}


########################################
# Training
########################################


def train(
    cfg: SplatsConfig,
    images: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsics: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
    depth_targets: np.ndarray | None = None,
) -> None:
    """
    Train splats and write splats.ply / ckpt.pt / splats.zarr / splats_quality_report.json to out_dir.

    Args:
        images: (n_views, H, W, 3) uint8 frames.
        world_to_cam: (n_views, 4, 4) COLMAP-convention poses; intrinsics: (n_views, 3, 3) at frame resolution.
        points, colors: (P, 3) float32 / uint8 seed points in the same world frame.
        depth_targets: optional (n_views, h, w) float32 depth at any resolution, 0 = no target.
    """
    device = "cuda"
    n_views, height, width = images.shape[:3]
    out_dir = Path(out_dir)

    # Refuse inputs that cannot train: too few seed points, or mismatched per-view arrays
    n_points = len(points)
    if n_points < 100:
        raise ValueError(f"splats: need >= 100 seed points, got {n_points}")
    n_poses = len(world_to_cam)
    n_intrinsics = len(intrinsics)
    n_depth = None if depth_targets is None else len(depth_targets)
    per_view_counts = {n_views, n_poses, n_intrinsics}
    if n_depth is not None:
        per_view_counts.add(n_depth)
    if len(per_view_counts) != 1:
        raise ValueError(
            f"splats: frames mismatch — images {n_views}, world_to_cam {n_poses}, "
            f"intrinsics {n_intrinsics}, depth_targets {n_depth}"
        )

    # Cameras on the GPU; frames stay uint8 on the CPU and move one view at a time.
    # normalize_scene trains in splatfacto's unit-cube frame (scene_scale fixed to 1.0);
    # otherwise the world frame is kept and scene_scale carries the extent.
    cam_to_world_np = np.linalg.inv(world_to_cam)
    if cfg.normalize_scene:
        center, scale = scene_normalization(cam_to_world_np)
        cam_to_world_np[:, :3, 3] = (cam_to_world_np[:, :3, 3] - center) * scale
        points = (points - center) * scale
        if depth_targets is not None:
            depth_targets = depth_targets * scale
        logger.info("splats: normalised scene, centre %s scale %.4g", np.round(center, 3), scale)
    cam_to_world = torch.from_numpy(cam_to_world_np).float().to(device)
    intrinsics_gpu = torch.from_numpy(intrinsics).float().to(device)
    scene_scale = 1.0 if cfg.normalize_scene else compute_scene_scale(cam_to_world)

    # Gaussians, densification strategy, and the lr decay on the means (0.01x over the run)
    gaussians, optimizers = init_gaussians_from_points(cfg, points, colors, scene_scale, device)
    strategy = make_strategy(cfg, n_views)
    strategy.check_sanity(gaussians, optimizers)
    if isinstance(strategy, MCMCStrategy):
        strategy_state = strategy.initialize_state()
    else:
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    lr_gamma = 0.01 ** (1.0 / cfg.max_steps)
    means_optimizer = optimizers["means"]
    means_scheduler = ExponentialLR(means_optimizer, gamma=lr_gamma)
    schedulers = [means_scheduler]

    # Optional joint pose refinement
    pose_refiner, pose_optimizer = None, None
    if cfg.pose_opt:
        pose_refiner, pose_optimizer, pose_scheduler = make_pose_refiner(cfg, n_views, scene_scale, lr_gamma, device)
        schedulers.append(pose_scheduler)

    start_time = time.perf_counter()
    view_sampler = ViewSampler(n_views)
    loss_values: dict[str, float] = {}
    use_pre_backward_hook = isinstance(strategy, DefaultStrategy)
    normal_spec = cfg.losses.get("normal_consistency")
    for step in progress(range(cfg.max_steps), desc=f"splats[{cfg.primitive}]"):
        # Pick one view (splatfacto's permutation schedule) and its (possibly refined) camera
        view = view_sampler.next()
        view_image = images[view]
        view_depth_target = None if depth_targets is None else depth_targets[view]

        # Coarse-to-fine: train at 1/4 -> 1/2 -> native resolution on the splatfacto
        # schedule. Depth targets follow the image automatically — prepare_training_target
        # nearest-resizes them to the (downscaled) image dims.
        factor = downscale_factor(step, cfg.num_downscales, cfg.resolution_schedule)
        view_intrinsics = intrinsics_gpu[view : view + 1]
        view_image, view_intrinsics = downscale_view(view_image, view_intrinsics, factor)
        step_height, step_width = view_image.shape[:2]

        target = prepare_training_target(view_image, view_depth_target, device)
        view_cam_to_world = cam_to_world[view : view + 1]
        if pose_refiner is not None:
            camera_id = torch.tensor([view], device=device)
            view_cam_to_world = pose_refiner(view_cam_to_world, camera_id)

        # Render with the SH bands unlocked so far, over a random background so transparency cannot hide.
        # 3DGS normals cost an extra-signal pass, so they are only rendered once the consistency loss is on.
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        absgrad = use_pre_backward_hook and strategy.absgrad
        render_normals = loss_active(step, normal_spec)
        render, info = render_view(
            cfg.primitive,
            gaussians,
            view_cam_to_world,
            view_intrinsics,
            step_width,
            step_height,
            sh_degree,
            absgrad,
            render_normals=render_normals,
        )
        background = torch.rand(1, 3, device=device)
        transparency = 1.0 - render["alpha"]
        render["rgb"] = render["rgb"] + background * transparency

        # Loss + backward; DefaultStrategy needs a hook before backward to retain 2D-means gradients
        if use_pre_backward_hook:
            strategy.step_pre_backward(gaussians, optimizers, strategy_state, step, info)
        loss, loss_values = compute_losses(step, render, target, gaussians, cfg.losses, scene_scale)
        loss.backward()

        # Optimizer steps for Gaussians (and poses), then lr decay
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if pose_optimizer is not None:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        # Densify / prune / relocate AFTER the optimizer step (upstream simple_trainer order): refine ops
        # rebuild the Parameters, so stepping afterwards would see .grad=None and silently skip them.
        # The two strategies take different extra arguments; MCMC reads the post-decay means lr like upstream.
        if isinstance(strategy, MCMCStrategy):
            means_lr_now = means_scheduler.get_last_lr()[0]
            strategy.step_post_backward(gaussians, optimizers, strategy_state, step, info, lr=means_lr_now)
        else:
            strategy.step_post_backward(gaussians, optimizers, strategy_state, step, info, packed=False)

        if step % cfg.log_every == 0:
            n_gaussians = len(gaussians["means"])
            rounded = {name: round(value, 4) for name, value in loss_values.items()}
            logger.info("splats step %d loss %.4f gaussians %d %s", step, loss.item(), n_gaussians, rounded)

    # loss_values is the LAST step's single-view snapshot, reported as summary.final_losses (not an average)
    train_seconds = time.perf_counter() - start_time
    # Outputs stay in world units: undo the normalisation before anything is written
    if cfg.normalize_scene:
        denormalize_outputs(gaussians, pose_refiner, cam_to_world, center, scale)

    write_splat_outputs(
        cfg, gaussians, pose_refiner, images, cam_to_world, intrinsics_gpu, out_dir, train_seconds, loss_values
    )
