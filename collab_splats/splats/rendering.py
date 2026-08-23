"""
One render call for both splat primitives.

3DGS goes through ``gsplat.rasterization`` (fast kernel, antialiased); per-Gaussian normals are
rendered as an extra signal and the depth normal is finite-differenced from rendered depth.
2DGS goes through ``gsplat.rasterization_2dgs`` which returns rendered normals (world frame) plus a
distortion map. Normals are rotated into camera space and the depth normal is finite-differenced at an
identity pose for both primitives, so the consistency loss compares like with like.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs
from gsplat.utils import depth_to_normal, normalized_quat_to_rotmat
from torch import Tensor


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


def render_view(
    primitive: str,
    gaussians: torch.nn.ParameterDict,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    absgrad: bool,
) -> tuple[dict[str, Tensor], dict]:
    """
    Render one camera. Returns ({rgb, alpha, depth, normal, depth_normal[, distortion]}, strategy info).

    - Normals are camera-frame for both primitives; `depth_normal` is finite-differenced at an identity pose.
    - 3DGS `normal` is unit length (zero where nothing renders).
    - 2DGS `normal` is the alpha-weighted accumulated normal (non-unit), mirroring upstream gsplat's 2DGS
      trainer, so the consistency loss is effectively alpha^2-weighted there. Deliberately not normalized.
    """
    assert cam_to_world.shape[0] == 1, "render_view renders one camera at a time"

    # Activate the raw parameters: log-scales -> scales, logit-opacities -> opacities, SH bands concatenated
    means = gaussians["means"]
    quats = gaussians["quats"]
    sh0 = gaussians["sh0"]
    shN = gaussians["shN"]
    scales = torch.exp(gaussians["scales"])
    opacities = torch.sigmoid(gaussians["opacities"])
    sh_coeffs = torch.cat([sh0, shN], dim=1)
    world_to_cam = torch.linalg.inv(cam_to_world)
    shared_kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=world_to_cam,
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=absgrad,
        render_mode="RGB+ED",
    )

    # Depth normals are finite-differenced in camera space (identity pose) for both primitives
    identity_pose = torch.eye(4, device=cam_to_world.device)[None]

    # 2DGS: the rasterizer returns rgb+depth, alpha, rendered normals (world frame), and the distortion
    # map; normals are rotated back into the camera frame and the depth normal recomputed there
    if primitive == "2dgs":
        rgb_depth, alpha, normal_world, _depth_normal_world, distortion, _median_depth, info = rasterization_2dgs(
            **shared_kwargs, distloss=True
        )
        rgb = rgb_depth[..., :3]
        depth = rgb_depth[..., 3:4]
        rotation_w2c = world_to_cam[:, :3, :3]
        normal_cam = torch.einsum("cij,chwj->chwi", rotation_w2c, normal_world)
        render = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": depth,
            "normal": normal_cam,
            "depth_normal": depth_to_normal(depth, identity_pose, intrinsics),
            "distortion": distortion,
        }
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
