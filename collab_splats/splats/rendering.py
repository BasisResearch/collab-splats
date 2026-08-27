"""
One render call for both splat primitives.

3DGS goes through ``gsplat.rasterization`` (fast kernel, antialiased); per-Gaussian normals are
rendered as an extra signal and the depth normal is finite-differenced from rendered depth.
2DGS goes through ``gsplat.rasterization_2dgs`` which returns rendered normals (world frame), a
distortion map, and the median depth (RaDe-GS's surface depth). Normals are rotated into camera
space and depth normals are finite-differenced at an identity pose for both primitives and for
both depths, so the consistency loss compares like with like.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs
from gsplat.utils import depth_to_normal, normalized_quat_to_rotmat
from torch import Tensor

SH_DC_NORMALISER = 0.28209479177387814  # rgb -> SH degree-0 coefficient (1 / (2 sqrt(pi)))


def gaussian_normals_in_camera_frame(quats: Tensor, scales: Tensor, means: Tensor, world_to_cam: Tensor) -> Tensor:
    """
    Per-Gaussian normal (shortest scale axis), rotated into the camera frame and flipped to face it. (N, 3).

    - `argmin` over scales is non-differentiable, so no gradient reaches `scales` through the normal;
      `quats` and `means` do receive gradient.
    """
    # Shortest axis of each Gaussian is its normal direction in world space
    unit_quats = F.normalize(quats, dim=-1)
    rotations = normalized_quat_to_rotmat(unit_quats)
    shortest_axis = scales.argmin(dim=-1)
    gaussian_index = torch.arange(len(rotations), device=rotations.device)
    normals_world = rotations[gaussian_index, :, shortest_axis]

    # Rotate normals and positions into the camera frame
    rotation_w2c = world_to_cam[:3, :3]
    translation_w2c = world_to_cam[:3, 3]
    normals_cam = normals_world @ rotation_w2c.T
    means_cam = means @ rotation_w2c.T + translation_w2c

    # A normal pointing away from the camera (positive dot with the view ray) is flipped
    faces_away = (normals_cam * means_cam).sum(-1, keepdim=True) > 0
    return torch.where(faces_away, -normals_cam, normals_cam)


def activate_vanilla(gaussians: torch.nn.ParameterDict) -> dict[str, Tensor]:
    """
    Activate raw vanilla parameters into the tensors the rasterizer takes.

    - log-scales -> scales, logit-opacities -> opacities, SH bands concatenated.
    - `colors` here are SH coefficients: the caller passes `sh_degree` to the rasterizer.
    """
    return {
        "means": gaussians["means"],
        "quats": gaussians["quats"],
        "scales": torch.exp(gaussians["scales"]),
        "opacities": torch.sigmoid(gaussians["opacities"]),
        "colors": torch.cat([gaussians["sh0"], gaussians["shN"]], dim=1),
    }


def render_gaussians(
    primitive: str,
    decoded: dict[str, Tensor],
    cam_to_world: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int | None,
    absgrad: bool,
    render_normals: bool = True,
) -> tuple[dict[str, Tensor], dict]:
    """
    Rasterize already-activated Gaussians. Returns ({rgb, alpha, depth[, normal, ...]}, strategy info).

    - `decoded` holds means/quats/scales/opacities/colors post-activation; any other key it carries
      (scaffold's `log_scales`, say) is ignored here.
    - `sh_degree=None` with `colors` of shape (N, 3) rasterizes post-activation RGB — that is the
      scaffold path. Vanilla passes SH coefficients and an integer degree.

    - Normals are camera-frame for both primitives; `depth_normal` is finite-differenced at an identity pose.
    - 3DGS `normal` is unit length (zero where nothing renders). `render_normals=False` skips the extra-signal
      pass and omits both `normal` and `depth_normal` (only the normal_consistency loss reads them); for 2DGS
      it omits only the finite-differenced `depth_normal`/`depth_normal_median`, since the rasterizer's own
      `normal` and `median_depth` come free and other consumers read them.
    - 2DGS `normal` is the alpha-weighted accumulated normal (non-unit), mirroring upstream gsplat's 2DGS
      trainer, so the consistency loss is effectively alpha^2-weighted there. Deliberately not normalized.
    - 2DGS extras: `distortion`, plus `median_depth` and, when `render_normals`, its `depth_normal_median`.
    """
    assert cam_to_world.shape[0] == 1, "render_gaussians renders one camera at a time"

    # The five tensors the rasterizer takes; everything else in `decoded` is for the caller
    means = decoded["means"]
    quats = decoded["quats"]
    scales = decoded["scales"]
    opacities = decoded["opacities"]
    colors = decoded["colors"]
    world_to_cam = torch.linalg.inv(cam_to_world)
    shared_kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_to_cam,
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=absgrad,
        render_mode="RGB+ED",
    )

    # rasterization_2dgs concatenates the depth channel onto precomputed colours per camera, so RGB
    # must arrive as (C, N, 3) there; SH coefficients (N, K, 3) and the 3DGS kernel broadcast on their own
    if primitive == "2dgs" and sh_degree is None and colors.dim() == 2:
        shared_kwargs["colors"] = colors[None].expand(len(intrinsics), -1, -1)

    # Depth normals are finite-differenced in camera space (identity pose) for both primitives
    identity_pose = torch.eye(4, device=cam_to_world.device)[None]

    # 2DGS: the rasterizer returns rgb+depth, alpha, rendered normals (world frame), the distortion
    # map and the median depth; normals are rotated back into the camera frame and both depth normals
    # are recomputed there
    if primitive == "2dgs":
        rgb_depth, alpha, normal_world, _depth_normal_world, distortion, median_depth, info = rasterization_2dgs(
            **shared_kwargs, distloss=True
        )
        rgb = rgb_depth[..., :3]
        depth = rgb_depth[..., 3:4]
        rotation_w2c = world_to_cam[:, :3, :3]
        normal_cam = torch.einsum("cij,chwj->chwi", rotation_w2c, normal_world)

        # RaDe-GS median depth: the depth of the median Gaussian along each ray, rather than
        # the alpha-weighted expectation. Sparse by construction (one Gaussian per ray
        # receives gradient) but sharper across depth discontinuities. Unconditional — the
        # splats.zarr writer persists it and has no render_normals to gate on.
        render = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": depth,
            "median_depth": median_depth,
            "normal": normal_cam,
            "distortion": distortion,
        }

        # Only normal_consistency_loss reads either depth normal (the median one only when its
        # depth_ratio > 0), and the trainer gates render_normals on that same loss_active — so off
        # its schedule both finite differences are dead tensors held live through backward
        if render_normals:
            render["depth_normal"] = depth_to_normal(depth, identity_pose, intrinsics)
            render["depth_normal_median"] = depth_to_normal(median_depth, identity_pose, intrinsics)
        return render, info

    # 3DGS without normals: plain rgb+depth pass (no extra-signal channels through the kernel)
    if not render_normals:
        rgb_depth, alpha, info = rasterization(**shared_kwargs, rasterize_mode="antialiased")
        render = {"rgb": rgb_depth[..., :3], "alpha": alpha, "depth": rgb_depth[..., 3:4]}
        return render, info

    # 3DGS: normals ride along as an extra per-Gaussian signal, zero-padded to 4 channels because the
    # compiled kernel supports 8 total channels (rgb + depth + 4) but not 7
    first_world_to_cam = world_to_cam[0]
    normals_cam = gaussian_normals_in_camera_frame(quats, scales, means, first_world_to_cam)
    padding = torch.zeros_like(normals_cam[:, :1])
    extra_signals = torch.cat([normals_cam, padding], dim=-1)
    rgb_depth, alpha, info = rasterization(**shared_kwargs, rasterize_mode="antialiased", extra_signals=extra_signals)
    rgb = rgb_depth[..., :3]
    depth = rgb_depth[..., 3:4]
    rendered_signals = info["render_extra_signals"]
    rendered_normals = rendered_signals[..., :3]
    render = {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "normal": F.normalize(rendered_normals, dim=-1),
        "depth_normal": depth_to_normal(depth, identity_pose, intrinsics),
    }
    return render, info


def render_view(
    primitive: str,
    gaussians: torch.nn.ParameterDict,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    absgrad: bool,
    render_normals: bool = True,
) -> tuple[dict[str, Tensor], dict]:
    """
    Render one camera from raw vanilla parameters (activate, then rasterize).
    """
    return render_gaussians(
        primitive,
        activate_vanilla(gaussians),
        cam_to_world,
        intrinsics,
        width,
        height,
        sh_degree,
        absgrad,
        render_normals=render_normals,
    )
