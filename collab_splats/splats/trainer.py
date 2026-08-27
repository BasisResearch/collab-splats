"""
Trainer for 3DGS / 2DGS splats on upstream gsplat.

Training runs in the COLMAP world frame by default so poses, depth and the ply line up with
every other stage artifact. ``scene_scale`` — 1.1 x the largest camera distance from the
camera centroid, as in gsplat's simple_trainer — only scales the means learning rate, the
densification thresholds and the depth loss. With ``normalize_scene`` the cameras, seed points
and depth targets are instead Sim3-normalised the splatfacto way (centre on the camera-position
mean, scale so the largest |camera coordinate| is 1) and ``scene_scale`` is fixed to 1.0; the
outputs are mapped back to world units before writing, so downstream stages never see the
normalised frame. The pose refiner's rotation lr is scaled by the WORLD-frame camera extent
and its translation lr by the training-frame ``scene_scale`` (measured 2026-08-25: one shared
lr x unit-cube scene_scale made pose opt ~79x weaker — 2.3x over-densification by step 2k; one
shared lr x world extent made translation steps 65x too large in world units, -1.6 dB). The
2dgs distortion loss is in depth units, so its weight is rescaled to world units as well.

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

from collab_splats.splats.appearance import AppearanceModule
from collab_splats.splats.cameras import CameraOptModule
from collab_splats.splats.losses import OPTIONAL_LOSSES, compute_losses, loss_active
from collab_splats.splats.outputs import write_splat_outputs
from collab_splats.splats.pgsr import select_near_views, to_gray
from collab_splats.splats.rendering import (
    SH_DC_NORMALISER,
    render_gaussians,
    render_view,
)
from collab_splats.splats.scaffold import AnchorField, AnchorStrategy, ScaffoldConfig
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)

PRIMITIVES = ("3dgs", "2dgs")
REPRESENTATIONS = ("vanilla", "scaffold")

########################################
# Config — every tunable, with gsplat simple_trainer defaults
########################################


def _default_losses(primitive: str) -> dict[str, dict]:
    """
    Default loss schedule per primitive: MCMC regularisers for 3dgs, distortion for 2dgs.
    """
    losses = {
        "depth": {"weight": 0.01},
        "normal_consistency": {"weight": 0.05, "start": 7000},
        "appearance_reg": {"weight": 1e-3},
    }
    if primitive == "3dgs":
        losses["opacity_reg"] = {"weight": 0.01}
        losses["scale_reg"] = {"weight": 0.01}
    else:
        losses["distortion"] = {"weight": 0.01, "start": 3000}
    return losses


# Per-loss tuning keys beyond {weight, start, end, end_weight}. depth_ratio is the RaDe-GS
# median-normal blend; the pgsr_* keys are PGSR's own hyperparameters (GS-SR
# gssr/scene/pgsr_scene.py:52-70), kept on the loss spec rather than promoted to SplatsConfig
# because they mean nothing when the loss is off.
LOSS_SPEC_KEYS = {
    "normal_consistency": {"depth_ratio"},
    "pgsr_normal": {"erode_ksize"},
    "pgsr_multiview": {
        "geo",
        "ncc",
        "pixel_noise_threshold",
        "num_sample",
        "patch_size",
        "num_multi_view",
        "max_points",
    },
}


@dataclass
class SplatsConfig:
    """
    Trainer knobs; mirrors the ``splats:`` yaml block minus ``enabled``. Defaults follow gsplat's examples.
    """

    primitive: str = "3dgs"
    representation: str = "vanilla"  # vanilla (per-gaussian params) | scaffold (anchors + MLP decode)
    scaffold: dict | None = None  # scaffold block; only with representation: scaffold
    max_steps: int = 30000
    pose_opt: bool = True
    appearance_opt: bool = False  # per-image affine colour (AppearanceModule); train views only
    losses: dict[str, dict] | None = None  # None -> _default_losses(primitive)

    # Appearance
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # one more SH band unlocked every this many steps
    init_opacity: float = 0.1

    # Learning rates (means_lr x scene_scale; pose_lr x world extent for rotation, x scene_scale for
    # translation; means/pose decay 0.01x over the run)
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

        # Parsed scaffold block, kept off the dataclass fields so asdict(cfg) — which lands in ckpt.pt
        # and the zarr attrs — stays plain yaml-shaped data
        if self.representation == "scaffold":
            self.scaffold_config = ScaffoldConfig.from_dict(self.scaffold or {})
        else:
            self.scaffold_config = None

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

        # Representation and its block: a scaffold block without the representation is a silent no-op
        if cfg.representation not in REPRESENTATIONS:
            raise ValueError(f"splats.representation must be one of {REPRESENTATIONS}, got '{cfg.representation}'")
        if cfg.scaffold is not None and cfg.representation != "scaffold":
            raise ValueError("splats.scaffold requires representation: scaffold")

        # Scaffold decodes RGB from mlp_colour, so the SH schedule has nothing to act on. Only a
        # DELIBERATE override is an error: base.yaml always carries both keys at their defaults and
        # deep-merges them into every block, so presence alone would make scaffold unrunnable.
        sh_keys = ("sh_degree", "sh_degree_interval")
        sh_overridden = any(key in block and block[key] != cls.__dataclass_fields__[key].default for key in sh_keys)
        if cfg.representation == "scaffold" and sh_overridden:
            raise ValueError(
                "splats.sh_degree / sh_degree_interval are vanilla-only; scaffold decodes RGB from mlp_colour"
            )
        for name, spec in cfg.losses.items():
            if name not in OPTIONAL_LOSSES:
                raise ValueError(f"splats.losses: unknown loss '{name}'; allowed {sorted(OPTIONAL_LOSSES)}")
            # Some losses carry their own tuning keys; each belongs to exactly one loss
            allowed_spec_keys = {"weight", "start", "end", "end_weight"} | LOSS_SPEC_KEYS.get(name, set())
            unknown_spec_keys = set(spec) - allowed_spec_keys
            if unknown_spec_keys or "weight" not in spec:
                # Report the keys legal for THIS loss, so a depth_ratio typo isn't told the key doesn't exist
                optional_keys = ", ".join(sorted(allowed_spec_keys - {"weight"}))
                raise ValueError(f"splats.losses.{name}: expected {{weight[, {optional_keys}]}}, got {sorted(spec)}")

            # Decay entries need both endpoints positive (log-linear) and a non-empty interval
            if "end" in spec:
                end_weight = spec.get("end_weight")
                if end_weight is None or spec["weight"] <= 0 or end_weight <= 0 or spec["end"] <= spec.get("start", 0):
                    raise ValueError(
                        f"splats.losses.{name}: decay needs weight > 0, end_weight > 0 and end > start, got {spec}"
                    )

        # The distortion map only exists for 2DGS
        distortion_spec = cfg.losses.get("distortion", {})
        distortion_weight = distortion_spec.get("weight", 0.0)
        if cfg.primitive == "3dgs" and distortion_weight > 0:
            raise ValueError("splats.losses.distortion is 2dgs-only; set its weight to 0 or use primitive: 2dgs")

        # bool is an int subclass, so `depth_ratio: yes` would coerce to a silent full median blend;
        # reject bools and non-numbers (a quoted '0.6' too) before any coercion, naming the key
        raw_depth_ratio = cfg.losses.get("normal_consistency", {}).get("depth_ratio", 0.0)
        if isinstance(raw_depth_ratio, bool) or not isinstance(raw_depth_ratio, (int, float)):
            raise ValueError(
                f"splats.losses.normal_consistency.depth_ratio must be a number in [0, 1], got {raw_depth_ratio!r}"
            )

        # Median depth only exists for 2DGS, so a non-zero blend on 3dgs is a config error
        depth_ratio = float(raw_depth_ratio)
        if not 0.0 <= depth_ratio <= 1.0:
            raise ValueError(f"splats.losses.normal_consistency.depth_ratio must be in [0, 1], got {depth_ratio}")
        if depth_ratio > 0 and cfg.primitive != "2dgs":
            raise ValueError(
                "splats.losses.normal_consistency.depth_ratio > 0 is 2dgs-only "
                "(median depth is a rasterization_2dgs output); set it to 0 or use primitive: 2dgs"
            )
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
            pose_refiner.translation.weight /= scale


def denormalize_anchors(
    anchors: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    cam_to_world: Tensor,
    center: np.ndarray,
    scale: float,
) -> None:
    """
    Undo ``scene_normalization`` in place on the anchors, cameras and pose deltas.

    - anchors: p / scale + center; both halves of the log ``scaling`` shift by - log(scale).
    - offsets are stored in units of the anchor's own extent, so they are scale-free and untouched.
    - the MLP heads survive this because their only view input is a unit direction: everything the
      outputs are written from decodes AFTER this call, so a scale-dependent head input would render
      a different model than the one that trained.
    """
    center_t = torch.as_tensor(center, dtype=torch.float32, device=cam_to_world.device)
    with torch.no_grad():
        anchors["anchors"].data = anchors["anchors"].data / scale + center_t
        anchors["scaling"].data = anchors["scaling"].data - math.log(scale)
        cam_to_world[:, :3, 3] = cam_to_world[:, :3, 3] / scale + center_t
        if pose_refiner is not None:
            pose_refiner.translation.weight /= scale


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
    cfg: SplatsConfig, n_views: int, rotation_lr_scale: float, translation_lr_scale: float, lr_gamma: float, device: str
) -> tuple[CameraOptModule, torch.optim.Optimizer, ExponentialLR]:
    """
    Zero-initialised CameraOptModule with a two-group Adam (rotation, translation) and exponential lr decay.

    - `rotation_lr_scale`: world-frame camera extent (`compute_scene_scale` of the un-normalised cameras) —
      the value the pose_opt win was measured at; rotation is unit-free so it must not follow the frame.
    - `translation_lr_scale`: training-frame `scene_scale`, so translation steps keep their world-unit size
      whether or not the scene is normalised.
    """
    refiner = CameraOptModule(n_views).to(device)
    refiner.zero_init()
    param_groups = [
        {"params": refiner.rotation.parameters(), "lr": cfg.pose_lr * rotation_lr_scale},
        {"params": refiner.translation.parameters(), "lr": cfg.pose_lr * translation_lr_scale},
    ]
    # weight_decay=1e-6 as in gsplat @ d2f5c0f examples/simple_trainer.py pose_optimizers
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-6)
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)
    return refiner, optimizer, scheduler


def make_appearance_module(
    cfg: SplatsConfig, n_views: int, lr_gamma: float, device: str
) -> tuple[AppearanceModule, torch.optim.Optimizer, ExponentialLR]:
    """
    Identity-initialised AppearanceModule with Adam at `appearance_lr` and the run's exponential lr decay.
    """
    module = AppearanceModule(n_views).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=cfg.appearance_lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)
    return module, optimizer, scheduler


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
    world_extent = compute_scene_scale(torch.from_numpy(cam_to_world_np))
    loss_schedule = cfg.losses
    if cfg.normalize_scene:
        center, scale = scene_normalization(cam_to_world_np)
        cam_to_world_np[:, :3, 3] = (cam_to_world_np[:, :3, 3] - center) * scale
        points = (points - center) * scale
        if depth_targets is not None:
            depth_targets = depth_targets * scale
        logger.info("splats: normalised scene, centre %s scale %.4g", np.round(center, 3), scale)

        # The 2dgs distortion loss is linear in depth units: rescale its weight so the penalty
        # matches what the same weight applies in the world frame (reported value stays unit-cube)
        if "distortion" in loss_schedule:
            loss_schedule = dict(loss_schedule)
            distortion = dict(loss_schedule["distortion"])
            distortion["weight"] = distortion["weight"] / scale
            if "end_weight" in distortion:
                distortion["end_weight"] = distortion["end_weight"] / scale
            loss_schedule["distortion"] = distortion
    cam_to_world = torch.from_numpy(cam_to_world_np).float().to(device)
    intrinsics_gpu = torch.from_numpy(intrinsics).float().to(device)
    scene_scale = 1.0 if cfg.normalize_scene else compute_scene_scale(cam_to_world)

    # Representation: vanilla Gaussians with a gsplat strategy, or Scaffold anchors with AnchorStrategy.
    # Both expose the same (ParameterDict, {name: Adam}) pair, so everything below is shared.
    anchor_field = None
    if cfg.representation == "scaffold":
        anchor_field = AnchorField(cfg.scaffold_config, points, colors, scene_scale, n_views, device)
        gaussians, optimizers = anchor_field.params, anchor_field.optimizers
        strategy = AnchorStrategy(cfg.scaffold_config, cfg.primitive, anchor_field.voxel_size)
        anchor_state = strategy.initialize_state(len(gaussians["anchors"]))
        strategy_state = {name: value.to(device) for name, value in anchor_state.items()}
    else:
        gaussians, optimizers = init_gaussians_from_points(cfg, points, colors, scene_scale, device)
        strategy = make_strategy(cfg, n_views)
        strategy.check_sanity(gaussians, optimizers)
        if isinstance(strategy, MCMCStrategy):
            strategy_state = strategy.initialize_state()
        else:
            strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    # lr decay on the anchor / Gaussian positions (0.01x over the run)
    lr_gamma = 0.01 ** (1.0 / cfg.max_steps)
    means_key = "anchors" if anchor_field is not None else "means"
    means_optimizer = optimizers[means_key]
    means_scheduler = ExponentialLR(means_optimizer, gamma=lr_gamma)
    schedulers = [means_scheduler]

    # Optional joint pose refinement
    pose_refiner, pose_optimizer = None, None
    if cfg.pose_opt:
        pose_refiner, pose_optimizer, pose_scheduler = make_pose_refiner(
            cfg, n_views, world_extent, scene_scale, lr_gamma, device
        )
        schedulers.append(pose_scheduler)

    # Optional per-image appearance (exposure / white balance) model
    appearance, appearance_optimizer = None, None
    if cfg.appearance_opt:
        appearance, appearance_optimizer, appearance_scheduler = make_appearance_module(cfg, n_views, lr_gamma, device)
        schedulers.append(appearance_scheduler)

    # PGSR: the losses are only defined against the 3dgs kernel (GS-SR pairs scaffold-pgsr with the
    # vanilla rasterizer; there is no 2dgs-pgsr upstream)
    pgsr_normal_spec = cfg.losses.get("pgsr_normal")
    pgsr_mv_spec = cfg.losses.get("pgsr_multiview")
    if (pgsr_normal_spec is not None or pgsr_mv_spec is not None) and cfg.primitive != "3dgs":
        raise ValueError(f"pgsr losses need primitive: 3dgs, got {cfg.primitive!r}")

    # Neighbour views for the multi-view losses, scored once over the seed cloud: a partner that sees
    # the same surface from a useful baseline beats both a near-duplicate and a wide one
    near_ids: list[list[int]] = []
    if pgsr_mv_spec is not None:
        near_ids = select_near_views(
            torch.linalg.inv(cam_to_world),
            intrinsics_gpu,
            torch.from_numpy(np.ascontiguousarray(points)).float().to(device),
            height,
            width,
            num_views=int(pgsr_mv_spec.get("num_multi_view", 5)),
            max_points=int(pgsr_mv_spec.get("max_points", 20000)),
        )

    start_time = time.perf_counter()
    view_sampler = ViewSampler(n_views)
    loss_values: dict[str, float] = {}
    # AnchorStrategy is not a DefaultStrategy, so scaffold never takes the pre-backward hook: it retains
    # the screen-space gradient itself, below, and accumulates per anchor slot after backward
    use_pre_backward_hook = isinstance(strategy, DefaultStrategy)
    normal_spec = cfg.losses.get("normal_consistency")
    for step in progress(range(cfg.max_steps), desc=f"splats[{cfg.primitive}]"):
        # Offsets and the MLP heads follow their own exponential schedules, which the shared
        # ExponentialLR below cannot express (it drives one optimizer at one gamma). Their horizon is
        # scaffold.lr_max_steps, not cfg.max_steps — upstream's schedules are run-length free. Upstream
        # sets them at the TOP of the iteration it trains (GS-SR gssr/trainer.py before render), so
        # setting them after the step would run every group one iteration behind.
        if anchor_field is not None:
            anchor_field.update_learning_rate(step)

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
        camera_id = torch.tensor([view], device=device)
        if pose_refiner is not None:
            view_cam_to_world = pose_refiner(view_cam_to_world, camera_id)

        # Render with the SH bands unlocked so far, over a random background so transparency cannot hide.
        # 3DGS normals cost an extra-signal pass, so they are only rendered once the consistency loss is on.
        render_normals = loss_active(step, normal_spec)
        render_plane = loss_active(step, pgsr_normal_spec) or loss_active(step, pgsr_mv_spec)
        if anchor_field is not None:
            # Scaffold decodes this view's Gaussians from the visible anchors (post-activation RGB, so
            # sh_degree=None); the decoded scales and opacities feed the regularisers, which have no
            # parameter to read under this representation
            decoded, decode_index = anchor_field.decode(
                cfg.primitive, view_cam_to_world, view_intrinsics, step_width, step_height, camera_id
            )
            render, info = render_gaussians(
                cfg.primitive,
                decoded,
                view_cam_to_world,
                view_intrinsics,
                step_width,
                step_height,
                sh_degree=None,
                absgrad=False,
                render_normals=render_normals,
                render_plane=render_plane,
            )
            render["log_scales"] = decoded["log_scales"]
            render["opacities"] = decoded["opacities"]
        else:
            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            absgrad = use_pre_backward_hook and strategy.absgrad
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
                render_plane=render_plane,
            )
        # PGSR multi-view: render one co-visible neighbour at this step's resolution. Upstream does NOT
        # detach it — the geometric term pulls both views' plane depths towards each other (GS-SR
        # gssr/scene/pgsr_scene.py, get_train_loss_dict). Costs a second decode and rasterization.
        if render_plane and loss_active(step, pgsr_mv_spec) and near_ids[view]:
            near = near_ids[view][random.randrange(len(near_ids[view]))]
            near_image, near_intrinsics = downscale_view(images[near], intrinsics_gpu[near : near + 1], factor)
            near_cam_to_world = cam_to_world[near : near + 1]
            near_camera_id = torch.tensor([near], device=device)
            if pose_refiner is not None:
                near_cam_to_world = pose_refiner(near_cam_to_world, near_camera_id)

            # Only the neighbour's plane depth and its grey image are read, so it needs no normals,
            # no appearance correction and no background composite
            if anchor_field is not None:
                near_decoded, _ = anchor_field.decode(
                    cfg.primitive, near_cam_to_world, near_intrinsics, step_width, step_height, near_camera_id
                )
                near_render, _ = render_gaussians(
                    cfg.primitive,
                    near_decoded,
                    near_cam_to_world,
                    near_intrinsics,
                    step_width,
                    step_height,
                    sh_degree=None,
                    absgrad=False,
                    render_normals=False,
                    render_plane=True,
                )
            else:
                near_render, _ = render_view(
                    cfg.primitive,
                    gaussians,
                    near_cam_to_world,
                    near_intrinsics,
                    step_width,
                    step_height,
                    min(step // cfg.sh_degree_interval, cfg.sh_degree),
                    False,
                    render_normals=False,
                    render_plane=True,
                )
            render["world_to_cam"] = torch.linalg.inv(view_cam_to_world)
            render["intrinsics"] = view_intrinsics
            render["pgsr_neighbour"] = {
                "plane_depth": near_render["plane_depth"],
                "gray": to_gray(torch.from_numpy(near_image).to(device).float() / 255.0),
                "world_to_cam": torch.linalg.inv(near_cam_to_world),
                "intrinsics": near_intrinsics,
            }

        # Per-image colour correction goes on the splat colour before the background is composited
        # (the background is not part of the photo's exposure); its params feed appearance_reg
        if appearance is not None:
            render["rgb"] = appearance(render["rgb"], camera_id)
            render["appearance"] = appearance.params(camera_id)
        background = torch.rand(1, 3, device=device)
        transparency = 1.0 - render["alpha"]
        render["rgb"] = render["rgb"] + background * transparency

        # Loss + backward; DefaultStrategy needs a hook before backward to retain 2D-means gradients
        if use_pre_backward_hook:
            strategy.step_pre_backward(gaussians, optimizers, strategy_state, step, info)
        if anchor_field is not None:
            info[strategy.key_for_gradient].retain_grad()
        loss, loss_values = compute_losses(step, render, target, gaussians, loss_schedule, scene_scale)
        loss.backward()

        # Scaffold reads the screen-space gradient off the retained tensor BEFORE the optimizers zero it,
        # and only inside the statistics window upstream gathers over
        if anchor_field is not None and strategy.should_accumulate(step):
            strategy.accumulate(strategy_state, info, decode_index, decoded["opacities"], decoded["visible_ids"])

        # Optimizer steps for Gaussians (and poses), then lr decay
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if anchor_field is not None:
            anchor_field.mlp_optimizer.step()
            anchor_field.mlp_optimizer.zero_grad(set_to_none=True)
        if pose_optimizer is not None:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        if appearance_optimizer is not None:
            appearance_optimizer.step()
            appearance_optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        # Densify / prune / relocate AFTER the optimizer step (upstream simple_trainer order): refine ops
        # rebuild the Parameters, so stepping afterwards would see .grad=None and silently skip them.
        # The two strategies take different extra arguments; MCMC reads the post-decay means lr like upstream.
        if anchor_field is not None:
            strategy.step_post_backward(anchor_field, strategy_state, step)
        elif isinstance(strategy, MCMCStrategy):
            means_lr_now = means_scheduler.get_last_lr()[0]
            strategy.step_post_backward(gaussians, optimizers, strategy_state, step, info, lr=means_lr_now)
        else:
            strategy.step_post_backward(gaussians, optimizers, strategy_state, step, info, packed=False)

        if step % cfg.log_every == 0:
            unit = "anchors" if anchor_field is not None else "gaussians"
            n_primitives = len(gaussians[means_key])
            rounded = {name: round(value, 4) for name, value in loss_values.items()}
            logger.info("splats step %d loss %.4f %s %d %s", step, loss.item(), unit, n_primitives, rounded)

    # loss_values is the LAST step's single-view snapshot, reported as summary.final_losses (not an average)
    train_seconds = time.perf_counter() - start_time
    # Outputs stay in world units: undo the normalisation before anything is written
    if cfg.normalize_scene:
        if anchor_field is not None:
            denormalize_anchors(gaussians, pose_refiner, cam_to_world, center, scale)
        else:
            denormalize_outputs(gaussians, pose_refiner, cam_to_world, center, scale)

    write_splat_outputs(
        cfg,
        gaussians,
        pose_refiner,
        appearance,
        images,
        cam_to_world,
        intrinsics_gpu,
        out_dir,
        train_seconds,
        loss_values,
        anchor_field=anchor_field,
    )
