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
from gsplat import fully_fused_projection
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

    # Scaffold's own per-image appearance embedding, concatenated into the colour MLP input.
    # 32 is upstream's shipped default (yanxian-ll/GS-SR gssr/gaussian/scaffold_gaussian.py:41);
    # 0 disables it. Independent of splats.appearance_opt (our per-image affine module): both
    # ship, neither retires the other, and the 2x2 is measured.
    appearance_dim: int = 32

    # Learning rates (anchor / offset lrs are multiplied by scene_scale, like means_lr). Anchors are
    # frozen upstream (position_lr_init = position_lr_final = 0): the voxel grid the growing dedup
    # tests against only stays a grid if the anchors sitting on it do not drift.
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
    mlp_colour_lr: float = 8e-3
    mlp_colour_lr_final: float = 5e-5
    appearance_lr: float = 5e-2
    appearance_lr_final: float = 5e-4

    # Every upstream schedule is keyed to a FIXED horizon (*_lr_max_steps = 30_000 in
    # gssr/gaussian/scaffold_gaussian.py:35-91), not to the run length: a 12k run therefore stops
    # at the lr upstream would hold at step 12k instead of racing to the final lr.
    lr_max_steps: int = 30000

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
        if cfg.lr_max_steps < 1:
            raise ValueError(f"splats.scaffold.lr_max_steps must be >= 1, got {cfg.lr_max_steps}")
        if cfg.voxel_multiplier <= 0:
            raise ValueError(f"splats.scaffold.voxel_multiplier must be > 0, got {cfg.voxel_multiplier}")
        return cfg


########################################
# MLP heads
########################################

# Unit view direction only. Upstream can also concatenate the view DISTANCE, but all three of its
# switches ship off (city-super/Scaffold-GS arguments/__init__.py: add_opacity_dist / add_cov_dist /
# add_color_dist = False), so the shipped heads read [anchor_feat, ob_view] and gaussian_renderer/
# __init__.py takes the cat_local_view_wodist branch. Direction is unit length, which is what keeps
# the heads scale-free: distance is a world-unit quantity, and feeding it would make every decode
# depend on the frame the scene happened to be trained in.
VIEW_DIM = 3

# Upstream clamps the raw (log-space) gaussian-extent channels at every prune
# (GS-SR gssr/gaussian/scaffold_gaussian.py:530), so no decoded Gaussian may exceed exp(0.05)
# world units however far the scaling parameter drifts.
SCALE_CAP = 0.05


class ScaffoldMLPs(torch.nn.Module):
    """
    The three Scaffold-GS decode heads: opacity, covariance, colour.

    - Input is [anchor_feat, view_dir] per visible anchor; every head emits one row of
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


def expon_lr(lr_init: float, lr_final: float, t: float) -> float:
    """
    Log-space interpolation from lr_init to lr_final over t in [0, 1].

    - Same curve as 3DGS's ``get_expon_lr_func``, which upstream Scaffold drives every iteration;
      its delay term is inert there (lr_delay_steps defaults to 0), so it is not reproduced.
    """
    if lr_init <= 0.0 or lr_final <= 0.0:
        return lr_init
    return math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)


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
                "rotation": torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_anchors, 1)),
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
        # One param group per head: upstream gives each head its own lr and its own decay schedule
        mlp_groups = [
            {"params": self.mlps.mlp_opacity.parameters(), "lr": cfg.mlp_opacity_lr, "name": "mlp_opacity"},
            {"params": self.mlps.mlp_cov.parameters(), "lr": cfg.mlp_cov_lr, "name": "mlp_cov"},
            {"params": self.mlps.mlp_colour.parameters(), "lr": cfg.mlp_colour_lr, "name": "mlp_colour"},
        ]
        if self.mlps.embedding_appearance is not None:
            mlp_groups.append(
                {
                    "params": self.mlps.embedding_appearance.parameters(),
                    "lr": cfg.appearance_lr,
                    "name": "embedding_appearance",
                }
            )
        self.mlp_optimizer = torch.optim.Adam(mlp_groups, lr=0.0, eps=1e-15)

        # (init, final) per decaying group; mlp_cov is constant upstream and the anchors are frozen
        self.lr_schedule = {
            "offsets": (cfg.offset_lr * scene_scale, cfg.offset_lr_final * scene_scale),
            "mlp_opacity": (cfg.mlp_opacity_lr, cfg.mlp_opacity_lr_final),
            "mlp_cov": (cfg.mlp_cov_lr, cfg.mlp_cov_lr),
            "mlp_colour": (cfg.mlp_colour_lr, cfg.mlp_colour_lr_final),
            "embedding_appearance": (cfg.appearance_lr, cfg.appearance_lr_final),
        }

    def update_learning_rate(self, step: int) -> None:
        """
        Decay the offset and MLP lrs for this step; the other anchor tensors hold a constant lr.

        - The horizon is ``cfg.lr_max_steps``, not the run length: upstream keys every schedule to a
          fixed 30k and a shorter run simply stops partway down the curve.
        """
        t = min(max(step / max(self.cfg.lr_max_steps, 1), 0.0), 1.0)
        for group in self.optimizers["offsets"].param_groups:
            group["lr"] = expon_lr(*self.lr_schedule["offsets"], t)
        for group in self.mlp_optimizer.param_groups:
            group["lr"] = expon_lr(*self.lr_schedule[group["name"]], t)

    @torch.no_grad()
    def visible_anchors(self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int) -> Tensor:
        """
        Boolean mask of the anchors this view's projection keeps (radii > 0).

        - Upstream's prefilter_voxel, which runs the rasterizer's visible_filter over the anchors with
          their OFFSET extent and rotation (GS-SR gssr/scene/scaffold_scene.py:122-155). gsplat exposes
          the same projection kernel directly, so this needs no extra rasterization pass.
        - gsplat ships CUDA kernels only, so off-GPU the analytic frustum test below stands in — the
          decode path stays exercisable on CPU, and training never takes that branch.
        - No gradient: the mask only selects which anchors decode.
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
        CPU stand-in for the projection prefilter: anchor centres in front of the camera and in frame.

        - The whole-frame margin keeps anchors whose Gaussians spill into frame from centres just
          outside it, which is what the projection kernel's radii do on GPU.
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
        - ``log_scales`` carries the decoded scales in log space for the scale regulariser, and
          ``visible_ids`` the anchors this view decoded — an anchor whose offsets all shut renders
          nothing yet still counts as visited, which is the whole basis of opacity pruning.
        - Never returns zero Gaussians: gsplat's projection kernel raises SIGFPE on an empty input.

        Follows generate_neural_gaussians in city-super/Scaffold-GS scene/gaussian_model.py
        (reimplemented; no code copied).
        """
        n_offsets = self.cfg.n_offsets
        visible = self.visible_anchors(cam_to_world, intrinsics, width, height)
        anchor_ids = torch.nonzero(visible, as_tuple=False).squeeze(-1)

        # gsplat's projection kernel divides by the primitive count, so an empty decode is a fatal
        # floating-point exception, not an empty image: when the frustum test culls everything, decode
        # every anchor instead and let the rasterizer cull them (rare, and the graph stays alive)
        if len(anchor_ids) == 0:
            anchor_ids = torch.arange(len(self.params["anchors"]), device=self.params["anchors"].device)

        anchors = self.params["anchors"][anchor_ids]
        feat = self.params["anchor_feat"][anchor_ids]
        scaling = torch.exp(self.params["scaling"][anchor_ids])
        offsets = self.params["offsets"][anchor_ids]

        # Unit direction from each anchor to the camera centre feeds every head; the distance is
        # computed only to normalise it (upstream's ob_dist, whose add_*_dist switches ship off)
        camera_centre = cam_to_world[0, :3, 3]
        to_camera = anchors - camera_centre
        view_distance = to_camera.norm(dim=-1, keepdim=True)
        view_direction = to_camera / view_distance.clamp_min(1e-8)
        features = torch.cat([feat, view_direction], dim=-1)

        neural_opacity, cov, colour = self.mlps(features, camera_id)

        # Offsets with non-positive opacity contribute nothing: dropping them here is what keeps the
        # decoded count far below anchors x n_offsets
        keep = (neural_opacity > 0).reshape(-1)

        # Same kernel constraint as above: if no offset is open this view, keep the single most
        # opaque one. It renders as good as nothing, and the MLP heads keep receiving gradient.
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
            "visible_ids": anchor_ids,
        }
        return decoded, decode_index


########################################
# Densification
########################################


class AnchorStrategy(Strategy):
    """
    Scaffold-GS anchor growing and pruning, in gsplat's Strategy shape.

    - Screen-space gradients are accumulated per (anchor, offset) slot through the decode index, then
      averaged by visit count — Scaffold's offset_gradient_accum / offset_denom.
    - Growing adds anchors in unoccupied voxels around high-gradient slots; pruning drops anchors whose
      accumulated decoded opacity stays below min_opacity.
    - Anchor tensors are grown/pruned through gsplat's own optimizer-state surgery, so Adam moments
      follow the parameters. The MLP heads are fixed-size and are never handed to this class.

    Reimplemented from Scaffold-GS (Lu et al., CVPR 2024) — adjust_anchor / anchor_growing /
    training_statis in city-super/Scaffold-GS scene/gaussian_model.py. No code copied.
    """

    def __init__(self, cfg: ScaffoldConfig, primitive: str, voxel_size: float = 0.0, verbose: bool = False):
        self.cfg = cfg
        self.primitive = primitive
        self.voxel_size = voxel_size
        self.verbose = verbose

        # 2DGS backward writes .absgrad on means2d only, never on the gradient_2dgs tensor
        # DefaultStrategy reads, so the key must follow the primitive or accumulation is all zeros
        self.key_for_gradient = "means2d" if primitive == "3dgs" else "gradient_2dgs"

    def initialize_state(self, n_anchors: int) -> dict[str, Tensor]:
        """
        Growing statistics are per slot (n_anchors x n_offsets), pruning statistics are per anchor.
        """
        return {
            "grad_accum": torch.zeros(n_anchors * self.cfg.n_offsets),
            "denom": torch.zeros(n_anchors * self.cfg.n_offsets),
            "opacity_accum": torch.zeros(n_anchors),
            "anchor_denom": torch.zeros(n_anchors),
        }

    def accumulate(self, state: dict, info: dict, decode_index: Tensor, opacities: Tensor, visible_ids: Tensor) -> None:
        """
        Scatter this view's screen-space gradient norms into the slot accumulators and its opacities
        into the anchor accumulators.

        - Gradients are renormalised to [-1, 1] screen space exactly as gsplat's DefaultStrategy does
          (strategy/default.py:243-249), which is what makes Scaffold's published grad_threshold
          directly usable here.
        - Opacity is summed per anchor over all its offsets and divided by the anchor's VISIT count,
          not by how often its offsets happened to render (upstream training_statis): an anchor that
          is visible with every offset shut must score zero, or it can never be pruned.
        - Only Gaussians the projection actually kept (radii > 0) carry gradient evidence, so the
          gradient half is filtered by it — upstream's update_filter (training_statis, GS-SR
          gssr/gaussian/scaffold_gaussian.py:506-508). The opacity half is not: an anchor is visited
          whether or not its Gaussians landed on screen.
        - A step whose gradient never reached the tensor (nothing rendered) is skipped, not counted.
        """
        grads = info[self.key_for_gradient].grad
        if grads is None:
            return
        grads = grads.detach().clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        grad_norm = grads.reshape(-1, 2).norm(dim=-1)

        device = state["grad_accum"].device
        index = decode_index.to(device)

        # radii is [C, N, 2] (or [N, 2] for one camera); a Gaussian counts if any axis rendered
        rendered = info["radii"].reshape(len(grad_norm), -1).amax(dim=-1) > 0
        grad_index = index[rendered.to(device)]
        state["grad_accum"].index_add_(0, grad_index, grad_norm[rendered].to(device))
        state["denom"].index_add_(0, grad_index, torch.ones_like(grad_index, dtype=state["denom"].dtype))

        # Negative opacities were dropped at decode, and upstream clamps them to zero before summing,
        # so summing the slots that survived gives the same per-anchor numerator
        anchor_index = torch.div(index, self.cfg.n_offsets, rounding_mode="floor")
        state["opacity_accum"].index_add_(0, anchor_index, opacities.detach().to(device))
        visible = visible_ids.to(device)
        state["anchor_denom"].index_add_(0, visible, torch.ones_like(visible, dtype=state["anchor_denom"].dtype))

    def should_accumulate(self, step: int) -> bool:
        """
        True inside upstream's statistics window (start_stat, update_until).

        - Growing starts later than counting does, so the first refine reads a full window instead of
          a cold one, and nothing is counted after the last refine (GS-SR densify(),
          gssr/gaussian/scaffold_gaussian.py:710).
        """
        return self.cfg.start_stat < step < self.cfg.update_until

    def grow(self, field: "AnchorField", state: dict) -> int:
        """
        Add anchors in unoccupied voxels around slots whose mean gradient clears the threshold.

        - Runs ``update_depth`` levels COARSE to fine: level i raises the threshold by
          ``(update_hierarchy_factor // 2) ** i`` while shrinking the grid from ``update_init_factor``
          voxels down towards one, so weak gradients seed coarse anchors and strong ones seed fine
          anchors (upstream anchor_growing; the two factors move in opposite directions).
        - Only slots decoded for most of the refine window count, and candidates are thinned per level.
        - Candidates are the decoded Gaussian positions, deduped against each other and against the
          existing anchor grid; a candidate landing in an occupied voxel is dropped.
        - A new anchor inherits the per-element max of its source slots' features: starting them blank
          leaves a grown field mostly feature-less, since growing adds far more anchors than seeding.

        Returns the number of anchors added.
        """
        n_offsets = self.cfg.n_offsets
        mean_grads = state["grad_accum"] / state["denom"].clamp_min(1.0)

        # A slot must have been decoded for most of the window before its gradient is evidence
        # (upstream offset_denom > check_interval * success_threshold * 0.5)
        seen = state["denom"] > self.cfg.refine_every * self.cfg.success_threshold * 0.5

        added_total = 0
        for level in range(self.cfg.update_depth):
            threshold = self.cfg.grad_threshold * ((self.cfg.update_hierarchy_factor // 2) ** level)
            size_factor = max(self.cfg.update_init_factor // (self.cfg.update_hierarchy_factor**level), 1)
            level_voxel = self.voxel_size * size_factor
            selected = seen & (mean_grads >= threshold)

            # Upstream thins candidates per level (rand > 0.5 ** (i + 1)), so one refine cannot claim
            # every free cell around a hot region at once
            selected = selected & (torch.rand_like(mean_grads) > 0.5 ** (level + 1))
            if not bool(selected.any()):
                continue

            # Candidate positions are the slots' decoded means, rebuilt from the current anchors so
            # each level sees the ones the previous level added
            anchors = field.params["anchors"].detach()
            offset_extent = torch.exp(field.params["scaling"].detach()[:, :3])
            candidates = anchors[:, None, :] + field.params["offsets"].detach() * offset_extent[:, None, :]
            candidates = candidates.reshape(-1, 3)[selected]

            # Each candidate carries its source anchor's feature into the cell it lands in
            slot_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
            source_feat = field.params["anchor_feat"].detach()[torch.div(slot_ids, n_offsets, rounding_mode="floor")]

            # Quantise onto this level's grid; a candidate cell already holding an anchor is dropped.
            # unique+counts rather than a pairwise mask: at 100k anchors the latter is tens of GB.
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

            self._append_anchors(field, state, new_cells * level_voxel, level_voxel, cell_feat[free])
            added_total += len(new_cells)

            # The accumulators grew with the anchors, so the per-slot views must grow too; new slots
            # have no history and cannot seed the next level
            padding = torch.zeros(len(new_cells) * n_offsets, device=mean_grads.device)
            mean_grads = torch.cat([mean_grads, padding])
            seen = torch.cat([seen, padding.bool()])

        # Only the slots that carried evidence reset; upstream leaves the rest accumulating so a slot
        # that is rarely visible still builds a window's worth of history
        state["grad_accum"][seen] = 0.0
        state["denom"][seen] = 0.0

        if self.verbose and added_total:
            logger.info("scaffold: grew %d anchors -> %d", added_total, len(field.params["anchors"]))
        return added_total

    def _append_anchors(
        self, field: "AnchorField", state: dict, new_anchors: Tensor, level_voxel: float, new_feat: Tensor
    ) -> None:
        """
        Append new anchors (zero offsets, inherited features, level-sized scaling) to params, optimizer
        state and the accumulators.
        """
        n_new = len(new_anchors)
        device = field.params["anchors"].device
        log_voxel = math.log(level_voxel)
        additions = {
            "anchors": new_anchors.to(device),
            "offsets": torch.zeros(n_new, self.cfg.n_offsets, 3, device=device),
            "anchor_feat": new_feat.to(device),
            "scaling": torch.full((n_new, 6), log_voxel, device=device),
            "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_new, 1),
        }

        # gsplat's helper rebuilds each Parameter and its Adam moments together; new rows start at zero
        # momentum, as upstream densification does
        def param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(torch.cat([param.detach(), additions[name]], dim=0))

        def optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return torch.cat([value, torch.zeros((n_new, *value.shape[1:]), device=value.device)], dim=0)

        _update_param_with_optimizer(param_fn, optimizer_fn, field.params, field.optimizers)

        # Growing statistics are per slot, pruning statistics per anchor, so they grow by different rows
        n_new_slots = n_new * self.cfg.n_offsets
        for key in ("grad_accum", "denom"):
            state[key] = torch.cat([state[key], torch.zeros(n_new_slots, device=state[key].device)])
        for key in ("opacity_accum", "anchor_denom"):
            state[key] = torch.cat([state[key], torch.zeros(n_new, device=state[key].device)])

    def prune(self, field: "AnchorField", state: dict) -> int:
        """
        Drop anchors whose mean decoded opacity stayed below min_opacity across the window.

        - An anchor must have been visited for most of the window before its mean opacity is evidence
          (upstream anchor_demon > check_interval * success_threshold); one that was never decoded is
          kept, since no evidence is not evidence of transparency.
        - Never prunes the field empty: gsplat's projection kernel raises SIGFPE on an empty input.

        Returns the number of anchors removed.
        """
        n_offsets = self.cfg.n_offsets
        denom = state["anchor_denom"]
        mean_opacity = state["opacity_accum"] / denom.clamp_min(1.0)
        seen = denom > self.cfg.refine_every * self.cfg.success_threshold
        drop = seen & (mean_opacity < self.cfg.min_opacity)

        # The window closes for the anchors that carried evidence, whether or not they were dropped
        state["opacity_accum"][seen] = 0.0
        state["anchor_denom"][seen] = 0.0

        if not bool(drop.any()):
            return 0

        # An empty field cannot be decoded or rendered, so the most opaque anchor always survives
        if bool(drop.all()):
            drop[mean_opacity.argmax()] = False
            logger.warning(
                "scaffold: every anchor fell below min_opacity %.4g; kept the most opaque one", self.cfg.min_opacity
            )

        keep = ~drop
        keep_slots = keep.repeat_interleave(n_offsets)

        def param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param.detach()[keep])

        def optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return value[keep]

        _update_param_with_optimizer(param_fn, optimizer_fn, field.params, field.optimizers)

        # Upstream caps the raw gaussian-extent channels every time it prunes (SCALE_CAP)
        with torch.no_grad():
            field.params["scaling"][:, 3:].clamp_(max=SCALE_CAP)

        for key in ("grad_accum", "denom"):
            state[key] = state[key][keep_slots]
        for key in ("opacity_accum", "anchor_denom"):
            state[key] = state[key][keep]

        n_dropped = int(drop.sum())
        if self.verbose:
            logger.info("scaffold: pruned %d anchors -> %d", n_dropped, len(field.params["anchors"]))
        return n_dropped

    def step_post_backward(self, field: "AnchorField", state: dict, step: int) -> None:
        """
        Grow then prune on the refine cadence inside [update_from, update_until].

        - Signature deliberately differs from gsplat's (params, optimizers, state, step, info): the
          anchor field owns both params and optimizers, and gradient accumulation happens in
          ``accumulate`` right after backward, before the optimizer step.
        """
        if step < self.cfg.update_from or step > self.cfg.update_until:
            return
        if step % self.cfg.refine_every != 0:
            return

        # Each half resets only the slots / anchors whose statistics it consumed, so a rarely-visible
        # one keeps building history instead of being wiped every window
        self.grow(field, state)
        self.prune(field, state)
