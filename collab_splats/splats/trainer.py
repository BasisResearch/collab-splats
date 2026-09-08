"""
Gaussian-splat training on upstream gsplat.

- ``primitive``: rasterizer axis — 3dgs / 2dgs
- ``representation``: model axis — vanilla ``Gaussians`` / Scaffold-GS anchors
- both answer one interface, so ``train`` never branches
- in: frames, COLMAP poses, intrinsics, seed point cloud
- out: splats.ply, ckpt.pt, splats_quality_report.json
- default frame: COLMAP world; ``scene_scale`` = 1.1 x the largest camera distance from the
  camera centroid, as in gsplat's simple_trainer
- ``scene_scale`` scales only: the means lr, the densification thresholds, the depth loss
- ``normalize_scene``: splatfacto's unit cube at ``scene_scale`` 1.0; outputs mapped back to
  world units before writing
- pose lrs: rotation x WORLD extent, translation x training-frame ``scene_scale``
- one shared lr: ~79x weaker pose opt (2.3x over-densification by step 2k), or 65x oversized
  translation steps (-1.6 dB)
- 2dgs distortion loss is in depth units, so its weight is rescaled too
"""

import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from collab_splats.splats.cameras import CameraOpt
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.losses import (
    compute_losses,
    default_losses,
    loss_active,
    neighbor_selection,
    rescale_depth_units,
    validate_schedule,
)
from collab_splats.splats.pgsr import render_neighbor, select_near_views
from collab_splats.splats.rendering import write_outputs
from collab_splats.splats.scaffold import Scaffold, ScaffoldConfig
from collab_splats.splats.utils import (
    compute_scene_scale,
    denormalize_cameras,
    downscale_factor,
    downscale_view,
    prepare_target,
    scene_normalization,
    view_order,
)
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)

PRIMITIVES = ("3dgs", "2dgs")

# Representation -> model class; the config validator takes its allow-list from the same mapping
MODEL_CLASSES = {"vanilla": Gaussians, "scaffold": Scaffold}
REPRESENTATIONS = tuple(MODEL_CLASSES)

########################################
# Config — every tunable, with gsplat simple_trainer defaults
########################################


@dataclass
class SplatsConfig:
    """
    Trainer knobs: the ``splats:`` yaml block minus ``enabled``, at gsplat's example defaults.
    """

    primitive: str = "3dgs"
    representation: str = "vanilla"  # vanilla (per-gaussian params) | scaffold (anchors + MLP decode)
    scaffold: dict | None = None  # scaffold block; only with representation: scaffold
    max_steps: int = 30000
    pose_opt: bool = True
    appearance_opt: bool = False  # per-image affine color (CameraOpt); train views only
    losses: dict[str, dict] | None = None  # None -> default_losses(primitive)

    # Appearance
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # one more SH band unlocked every this many steps
    init_opacity: float = 0.1

    # Learning rates: means_lr x scene_scale; pose_lr x world extent (rot) / scene_scale (trans);
    # means and pose decay 0.01x over the run
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 1.25e-4
    pose_lr: float = 1e-5
    appearance_lr: float = 1e-3

    # Densification
    cap_max: int = 1_000_000  # 3dgs (MCMC): Gaussian budget
    # 2dgs (Default): 2D-gradient split/duplicate threshold
    # - gsplat's non-absgrad default
    # - the spec's absgrad-calibrated 8e-4 starved densification: GH010229 160k vs 1.4M, -1.26 dB
    grow_grad2d: float = 2e-4

    log_every: int = 500

    # Coarse-to-fine (splatfacto): start at 1/2^num_downscales, double every schedule step; 0 off
    num_downscales: int = 2
    resolution_schedule: int = 3000

    # splatfacto scene normalization
    # - nerfstudio center_method="poses" + auto_scale_poses
    # - unit-cube frame at scene_scale = 1.0, not world units x scene_scale
    normalize_scene: bool = False

    def __post_init__(self):
        """
        Fill in the default loss schedule, refuse a zero-step run, and parse the scaffold block.
        """
        if self.losses is None:
            self.losses = default_losses(self.primitive)

        # At least one step: lr_gamma divides by max_steps, and final_losses needs a last step
        if self.max_steps < 1:
            raise ValueError(f"splats.max_steps must be >= 1, got {self.max_steps}")

        # Parsed scaffold block, off the dataclass fields so asdict(cfg) stays yaml-shaped
        if self.representation == "scaffold":
            self.scaffold_config = ScaffoldConfig.from_dict(self.scaffold or {})
        else:
            self.scaffold_config = None

    @classmethod
    def from_dict(cls, block: dict) -> "SplatsConfig":
        """
        Build from the yaml block, rejecting unknown keys and ill-formed or incompatible losses.

        - A given ``losses`` mapping REPLACES the per-primitive defaults (no merge).

        Args:
            block: the ``splats:`` yaml mapping; ``enabled`` is accepted and dropped.

        Returns:
            A SplatsConfig; ValueError on unknown keys, an unknown primitive or representation,
            a scaffold/SH clash, or an invalid loss schedule.
        """
        # Unknown top-level keys are almost always typos — refuse rather than silently default
        allowed_keys = {"enabled", *cls.__dataclass_fields__}
        unknown_keys = set(block) - allowed_keys
        if unknown_keys:
            raise ValueError(f"splats: unknown keys {sorted(unknown_keys)}; allowed {sorted(allowed_keys)}")
        fields = {key: value for key, value in block.items() if key != "enabled"}
        cfg = cls(**fields)

        # The primitive must be one the trainer knows
        if cfg.primitive not in PRIMITIVES:
            raise ValueError(f"splats.primitive must be one of {PRIMITIVES}, got '{cfg.primitive}'")

        # Representation and its block: a scaffold block without the representation is a silent no-op
        if cfg.representation not in REPRESENTATIONS:
            raise ValueError(f"splats.representation must be one of {REPRESENTATIONS}, got '{cfg.representation}'")
        if cfg.scaffold is not None and cfg.representation != "scaffold":
            raise ValueError("splats.scaffold requires representation: scaffold")

        # Scaffold decodes RGB from mlp_color, so SH has nothing to act on
        # - only a DELIBERATE override is an error: base.yaml merges both keys at their defaults
        sh_keys = ("sh_degree", "sh_degree_interval")
        sh_overridden = any(key in block and block[key] != cls.__dataclass_fields__[key].default for key in sh_keys)
        if cfg.representation == "scaffold" and sh_overridden:
            raise ValueError(
                "splats.sh_degree / sh_degree_interval are vanilla-only; scaffold decodes RGB from mlp_color"
            )

        # Loss schedule shape and primitive compatibility
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
    Train one splat model and write splats.ply, ckpt.pt and splats_quality_report.json to out_dir.

    Args:
        cfg: the run's SplatsConfig; `representation` picks the model class, `primitive` the
            rasterizer.
        images: (n_views, H, W, 3) uint8 frames.
        world_to_cam: (n_views, 4, 4) COLMAP-convention poses.
        intrinsics: (n_views, 3, 3) camera matrices at frame resolution.
        points: (P, 3) float32 seed points in the same world frame.
        colors: (P, 3) uint8 seed colors.
        out_dir: output directory; created if absent.
        depth_targets: optional (n_views, h, w) float32 depth at any resolution, 0 = no target.
        image_ids: SOURCE frame index per row of `images`, checkpointed so a later stage can
            find each view's file; a row-position default would silently mismatch any scene
            not sampled contiguously from frame 0.
        min_points: fewest seed points that can produce a model.
        lr_decay: total decay of the position lr over the run, 0.01 = 100x down.

    Returns:
        None — everything is written to `out_dir`.
    """
    device = "cuda"
    n_views, height, width = images.shape[:3]
    out_dir = Path(out_dir)

    # Refuse inputs that cannot train: too few seed points, or mismatched per-view arrays
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

    # Cameras on the GPU, frames uint8 on the CPU one view at a time
    # - normalize_scene: unit-cube frame, scene_scale 1.0
    # - else: world frame, scene_scale carries the extent
    cam_to_world_np = np.linalg.inv(world_to_cam)
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

        # Losses that live in depth units mean something different in a unit-cube frame
        loss_schedule = rescale_depth_units(loss_schedule, normalize_factor)
    cam_to_world = torch.from_numpy(cam_to_world_np).float().to(device)
    intrinsics_gpu = torch.from_numpy(intrinsics).float().to(device)
    scene_scale = 1.0 if cfg.normalize_scene else compute_scene_scale(cam_to_world)

    # Model and camera refiner own their own optimizers; nothing below branches on representation
    model = MODEL_CLASSES[cfg.representation](cfg, points, colors, scene_scale, n_views, device, lr_decay=lr_decay)
    refine = CameraOpt.from_config(cfg, n_views, world_extent, scene_scale, lr_decay ** (1.0 / cfg.max_steps), device)
    logger.info(
        "splats: training %s/%s, %d views, %d primitives at start",
        cfg.primitive,
        cfg.representation,
        n_views,
        model.n_primitives,
    )

    # PGSR losses are only defined against the 3dgs kernel
    # - upstream: yanxian-ll/GS-SR @ 566359be, gssr/scene/scaffold_pgsr_scene.py:11
    # - upstream scenes: pgsr / scaffold_pgsr / octree_pgsr, no 2dgs pairing
    pgsr_normal_spec = loss_schedule.get("pgsr_normal")
    pgsr_mv_spec = loss_schedule.get("pgsr_multiview")
    if (pgsr_normal_spec is not None or pgsr_mv_spec is not None) and cfg.primitive != "3dgs":
        raise ValueError(f"pgsr losses need primitive: 3dgs, got {cfg.primitive!r}")

    # Neighbor views for the multi-view losses, scored once: same surface, useful baseline
    near_ids: list[list[int]] = []
    if pgsr_mv_spec is not None:
        near_ids = select_near_views(
            torch.linalg.inv(cam_to_world),
            intrinsics_gpu,
            torch.from_numpy(np.ascontiguousarray(points)).float().to(device),
            height,
            width,
            **neighbor_selection(pgsr_mv_spec),
        )

    start_time = time.perf_counter()
    views = view_order(n_views)
    loss_values: dict[str, float] = {}
    normal_spec = loss_schedule.get("normal_consistency")
    for step in progress(range(cfg.max_steps), desc=f"splats[{cfg.primitive}]"):
        # Pick one view (splatfacto's permutation schedule) and its (possibly refined) camera
        view = next(views)
        view_image = images[view]
        view_depth_target = None if depth_targets is None else depth_targets[view]

        # Coarse-to-fine (splatfacto): 1/4 -> 1/2 -> native; prepare_target resizes depth to match
        factor = downscale_factor(step, cfg.num_downscales, cfg.resolution_schedule)
        view_intrinsics = intrinsics_gpu[view : view + 1]
        view_image, view_intrinsics = downscale_view(view_image, view_intrinsics, factor)
        step_height, step_width = view_image.shape[:2]

        target = prepare_target(view_image, view_depth_target, device)
        camera_id = torch.tensor([view], device=device)
        view_cam_to_world = refine.camera(cam_to_world[view : view + 1], camera_id)

        # Render with the SH bands unlocked so far
        # - 3DGS normals cost an extra pass: rendered only once the consistency loss is on
        render_normals = loss_active(step, normal_spec)
        render_plane = loss_active(step, pgsr_normal_spec) or loss_active(step, pgsr_mv_spec)
        render, info = model.render(
            view_cam_to_world,
            view_intrinsics,
            step_width,
            step_height,
            camera_id,
            step=step,
            render_normals=render_normals,
            render_plane=render_plane,
        )

        # PGSR multi-view: render one co-visible neighbor
        # - through its OWN camera id: appearance and pose deltas are per view
        # - NOT detached: the geometric term pulls both views' plane depths together
        # - upstream: GS-SR @ 566359be, gssr/scene/pgsr_scene.py:214-223
        if render_plane and loss_active(step, pgsr_mv_spec) and near_ids[view]:
            near = near_ids[view][random.randrange(len(near_ids[view]))]
            near_image, near_intrinsics = downscale_view(images[near], intrinsics_gpu[near : near + 1], factor)
            near_camera_id = torch.tensor([near], device=device)
            render["world_to_cam"] = torch.linalg.inv(view_cam_to_world)
            render["intrinsics"] = view_intrinsics
            render["pgsr_neighbor"] = render_neighbor(
                model,
                near_image,
                refine.camera(cam_to_world[near : near + 1], near_camera_id),
                near_intrinsics,
                near_camera_id,
            )

        # Per-image color correction, before compositing the background; params feed appearance_reg
        render["rgb"] = refine.color(render["rgb"], camera_id)
        appearance_params = refine.color_params(camera_id)
        if appearance_params is not None:
            render["appearance"] = appearance_params
        background = torch.rand(1, 3, device=device)
        transparency = 1.0 - render["alpha"]
        render["rgb"] = render["rgb"] + background * transparency

        # Loss + backward; `info` is the MAIN view's, never the neighbor's
        model.pre_backward(step, info)
        loss, loss_values = compute_losses(step, render, target, model.params, loss_schedule, scene_scale)
        loss.backward()

        # Optimizer steps for the model (and the cameras), then lr decay
        for optimizer in model.optimizers + refine.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in model.schedulers + refine.schedulers:
            scheduler.step()

        # Densify / prune / relocate AFTER the optimizer step (upstream order)
        # - refine ops rebuild the Parameters: a later step sees .grad=None and silently skips
        model.post_backward(step, info)

        if step % cfg.log_every == 0:
            rounded = {name: round(value, 4) for name, value in loss_values.items()}
            logger.info(
                "splats step %d loss %.4f %s %d %s",
                step,
                loss.item(),
                type(model).__name__,
                model.n_primitives,
                rounded,
            )

    # loss_values is the LAST step's single-view snapshot, not an average
    train_seconds = time.perf_counter() - start_time

    # Outputs stay in world units: undo the Sim3 on model, cameras and pose deltas before writing
    if cfg.normalize_scene:
        model.denormalize(center, normalize_factor)
        denormalize_cameras(cam_to_world, center, normalize_factor)
        refine.denormalize(normalize_factor)

    # Pose deltas fold into the stored poses, never checkpointed apart — reapplying would double
    with torch.no_grad():
        corrected = torch.cat(
            [
                refine.camera(cam_to_world[view : view + 1], torch.tensor([view], device=device))
                for view in range(n_views)
            ]
        )

    # Writer takes both pose sets: corrected for ckpt/re-renders, uncorrected for the baked ply
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
        loss_values,
        training_cam_to_world=cam_to_world,
    )
