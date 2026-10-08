"""
Gaussian-splat training loop on gsplat.

- `SplatsConfig`: every training knob, parsed from the `splats:` yaml block
- `train`: frames, poses and seed points in; splats.ply, ckpt.pt and a quality report out
- `primitive` picks the rasterizer (3dgs / 2dgs), `representation` the model (vanilla / scaffold)
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from collab_splats.geometry.transforms import (
    invert_poses,
    rescale_intrinsics,
    shift_intrinsics,
)
from collab_splats.splats.cameras import CameraOpt
from collab_splats.splats.checkpoint import (
    MODEL_CLASSES,
    REPRESENTATIONS,
    write_outputs,
)
from collab_splats.splats.losses import (
    compute_losses,
    default_losses,
    loss_active,
    rescale_depth_units,
    validate_schedule,
)
from collab_splats.splats.scaffold import ScaffoldConfig
from collab_splats.splats.utils import (
    cached_target,
    compute_scene_scale,
    denormalize_cameras,
    downscale_factor,
    scene_normalization,
    view_order,
)
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)

PRIMITIVES = ("3dgs", "2dgs")

########################################
# Config
########################################


@dataclass
class SplatsConfig:
    """
    Training settings: the `splats:` yaml block, defaults from gsplat's example trainer.
    """

    primitive: str = "3dgs"
    representation: str = "vanilla"  # vanilla | scaffold
    scaffold: dict | None = None  # only with representation: scaffold
    max_steps: int = 30000
    pose_opt: bool = True
    appearance_opt: bool = False  # per-image color correction
    losses: dict[str, dict] | None = None  # None -> default_losses(primitive)

    # Appearance
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # steps between SH band unlocks
    init_opacity: float = 0.1

    # Learning rates; means_lr and pose_lr are scaled by the scene extent
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 1.25e-4
    pose_lr: float = 1e-5
    appearance_lr: float = 1e-3

    # Densification
    cap_max: int = 1_000_000  # 3dgs: max Gaussian count
    # 2dgs: gradient threshold for split/duplicate (gsplat's non-absgrad default)
    grow_grad2d: float = 2e-4

    log_every: int = 500

    # Coarse-to-fine: start at 1/2^num_downscales, double every resolution_schedule steps; 0 off
    num_downscales: int = 2
    resolution_schedule: int = 3000

    # Train in splatfacto's unit-cube frame; outputs are mapped back to world units
    normalize_scene: bool = False

    def __post_init__(self):
        """
        Fill in default losses, check max_steps, and parse the scaffold block.
        """
        if self.losses is None:
            self.losses = default_losses(self.primitive)

        # At least one step
        if self.max_steps < 1:
            raise ValueError(f"splats.max_steps must be >= 1, got {self.max_steps}")

        # Parsed scaffold block, kept off the fields so asdict(cfg) matches the yaml
        if self.representation == "scaffold":
            self.scaffold_config = ScaffoldConfig.from_dict(self.scaffold or {})
        else:
            self.scaffold_config = None

    @classmethod
    def from_dict(cls, block: dict) -> "SplatsConfig":
        """
        Config from the `splats:` yaml block, validated.

        - a given `losses` block replaces the defaults, it is not merged

        Args:
            block: the `splats:` yaml mapping; `enabled` is ignored.

        Returns:
            The validated config.

        Raises:
            ValueError: unknown key, primitive or representation, SH set on scaffold, or bad losses.
        """
        # Unknown keys are likely typos: refuse them
        allowed_keys = {"enabled", *cls.__dataclass_fields__}
        unknown_keys = set(block) - allowed_keys

        if unknown_keys:
            raise ValueError(f"splats: unknown keys {sorted(unknown_keys)}; allowed {sorted(allowed_keys)}")

        fields = {key: value for key, value in block.items() if key != "enabled"}
        cfg = cls(**fields)

        # Known primitive
        if cfg.primitive not in PRIMITIVES:
            raise ValueError(f"splats.primitive must be one of {PRIMITIVES}, got '{cfg.primitive}'")

        # Known representation; a scaffold block needs representation: scaffold
        if cfg.representation not in REPRESENTATIONS:
            raise ValueError(f"splats.representation must be one of {REPRESENTATIONS}, got '{cfg.representation}'")

        if cfg.scaffold is not None and cfg.representation != "scaffold":
            raise ValueError("splats.scaffold requires representation: scaffold")

        # Scaffold has no SH; only a non-default SH value is an error
        sh_keys = ("sh_degree", "sh_degree_interval")
        sh_overridden = any(key in block and block[key] != cls.__dataclass_fields__[key].default for key in sh_keys)

        if cfg.representation == "scaffold" and sh_overridden:
            raise ValueError(
                "splats.sh_degree / sh_degree_interval are vanilla-only; scaffold decodes RGB from mlp_color"
            )

        # Loss schedule
        validate_schedule(cfg.losses, cfg.primitive)

        return cfg


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
    *,
    image_ids: Sequence[int],
    min_points: int = 100,
    lr_decay: float = 0.01,
) -> None:
    """
    Train a splat model and write splats.ply, ckpt.pt and the quality report to `out_dir`.

    Args:
        cfg: run config.
        images: (n_views, H, W, 3) uint8 frames.
        world_to_cam: (n_views, 4, 4) COLMAP-convention poses.
        intrinsics: (n_views, 3, 3) pixel-center K (pixel i at coordinate i) at frame resolution.
        points: (P, 3) float32 seed points.
        colors: (P, 3) uint8 seed colors.
        out_dir: output directory; created if absent.
        depth_targets: optional (n_views, h, w) depth at any resolution; 0 = no target.
        image_ids: source frame index per row of `images`.
        min_points: minimum seed point count.
        lr_decay: total lr decay over the run; 0.01 = 100x down.

    Returns:
        None; outputs are written to `out_dir`.
    """
    device = "cuda"
    n_views, height, width = images.shape[:3]
    out_dir = Path(out_dir)

    # Refuse too few seed points or mismatched per-view counts
    n_points = len(points)

    if n_points < min_points:
        raise ValueError(f"splats: need >= {min_points} seed points, got {n_points}")

    n_poses = len(world_to_cam)
    n_intrinsics = len(intrinsics)
    n_depth = None if depth_targets is None else len(depth_targets)
    n_ids = len(image_ids)
    per_view_counts = {n_views, n_poses, n_intrinsics, n_ids}

    if n_depth is not None:
        per_view_counts.add(n_depth)

    if len(per_view_counts) != 1:
        raise ValueError(
            f"splats: frames mismatch — images {n_views}, world_to_cam {n_poses}, "
            f"intrinsics {n_intrinsics}, image_ids {n_ids}, depth_targets {n_depth}"
        )

    # Cameras to the GPU, optionally normalized to a unit cube (frames reach the GPU through the target cache)
    cam_to_world_np = invert_poses(world_to_cam)
    world_extent = compute_scene_scale(torch.from_numpy(cam_to_world_np))
    loss_schedule = cfg.losses
    center = normalize_factor = None

    if cfg.normalize_scene:
        center, normalize_factor = scene_normalization(cam_to_world_np)
        cam_to_world_np[:, :3, 3] = (cam_to_world_np[:, :3, 3] - center) * normalize_factor
        points = (points - center) * normalize_factor

        if depth_targets is not None:
            depth_targets = depth_targets * normalize_factor

        logger.info("splats: normalized scene, center %s scale %.4g", np.round(center, 3), normalize_factor)

        # Depth-unit loss settings follow the normalization
        loss_schedule = rescale_depth_units(loss_schedule, normalize_factor)

    # gsplat puts pixel i's center at i + 0.5
    intrinsics = shift_intrinsics(intrinsics, (0.5, 0.5))

    cam_to_world = torch.from_numpy(cam_to_world_np).float().to(device)
    intrinsics_gpu = torch.from_numpy(intrinsics).float().to(device)

    # Intrinsics per coarse-to-fine factor, on the (H // f, W // f) grid
    native_hw = np.array([height, width])
    n_levels = max(cfg.num_downscales, 0) + 1
    intrinsics_by_factor = {}

    for level in range(n_levels):
        factor = 2**level
        scaled = rescale_intrinsics(intrinsics, native_hw, native_hw // factor)
        intrinsics_by_factor[factor] = torch.from_numpy(scaled).float().to(device)

    scene_scale = 1.0 if cfg.normalize_scene else compute_scene_scale(cam_to_world)

    # Model and camera refiner, each with its own optimizers
    model = MODEL_CLASSES[cfg.representation](cfg, points, colors, scene_scale, n_views, device, lr_decay=lr_decay)
    refine = CameraOpt.from_config(cfg, n_views, world_extent, scene_scale, lr_decay ** (1.0 / cfg.max_steps), device)
    logger.info(
        "splats: training %s/%s, %d views, %d primitives at start",
        cfg.primitive,
        cfg.representation,
        n_views,
        model.n_primitives,
    )

    start_time = time.perf_counter()
    views = view_order(n_views)
    loss_values: dict[str, Tensor] = {}
    normal_spec = loss_schedule.get("normal_consistency")

    # GPU targets of the current downscale factor, built on each view's first visit
    target_cache: dict[int, dict[int, dict]] = {}

    # Camera ids on the GPU once: a per-step torch.tensor(..., device) syncs the stream
    camera_ids = torch.arange(n_views, device=device)

    for step in progress(range(cfg.max_steps), desc=f"splats[{cfg.primitive}]"):
        # Next view in shuffled order
        view = next(views)

        # Coarse-to-fine: the view's target at this factor and its refined camera
        factor = downscale_factor(step, cfg.num_downscales, cfg.resolution_schedule)
        target = cached_target(target_cache, images, depth_targets, view, factor, device)
        view_intrinsics = intrinsics_by_factor[factor][view : view + 1]
        step_height, step_width = target["rgb"].shape[1:3]

        camera_id = camera_ids[view : view + 1]
        view_cam_to_world = refine.camera(cam_to_world[view : view + 1], camera_id)

        # Render; normals only once the normal-consistency loss is active
        render_normals = loss_active(step, normal_spec)
        render, info = model.render(
            view_cam_to_world,
            view_intrinsics,
            step_width,
            step_height,
            camera_id,
            step=step,
            render_normals=render_normals,
        )

        # Color correction, then composite over a random background
        render["rgb"] = refine.color(render["rgb"], camera_id)
        appearance_params = refine.color_params(camera_id)

        if appearance_params is not None:
            render["appearance"] = appearance_params

        background = torch.rand(1, 3, device=device)
        transparency = 1.0 - render["alpha"]
        render["rgb"] = render["rgb"] + background * transparency

        # Loss + backward
        model.pre_backward(step, info)
        loss, loss_values = compute_losses(step, render, target, model.params, loss_schedule, scene_scale)
        loss.backward()

        # Optimizer and lr scheduler steps
        for optimizer in model.optimizers + refine.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for scheduler in model.schedulers + refine.schedulers:
            scheduler.step()

        # Densify / prune after the optimizer step: refine rebuilds the Parameters, so .grad is gone
        model.post_backward(step, info)

        if step % cfg.log_every == 0:
            rounded = {name: round(value.item(), 4) for name, value in loss_values.items()}
            logger.info(
                "splats step %d loss %.4f %s %d %s",
                step,
                loss.item(),
                type(model).__name__,
                model.n_primitives,
                rounded,
            )

    # loss_values holds the last step's losses, not an average
    train_seconds = time.perf_counter() - start_time
    final_losses = {name: value.item() for name, value in loss_values.items()}

    # Map model, cameras and pose deltas back to world units
    if cfg.normalize_scene:
        model.denormalize(center, normalize_factor)
        denormalize_cameras(cam_to_world, center, normalize_factor)
        refine.denormalize(normalize_factor)

    # Bake the pose deltas into the saved poses
    with torch.no_grad():
        corrected = torch.cat(
            [refine.camera(cam_to_world[view : view + 1], camera_ids[view : view + 1]) for view in range(n_views)]
        )

    # Write outputs: corrected poses for ckpt and renders, training poses for the ply
    write_outputs(
        cfg,
        model,
        refine,
        images,
        list(image_ids),
        corrected,
        intrinsics_gpu,
        out_dir,
        train_seconds,
        final_losses,
        training_cam_to_world=cam_to_world,
    )
