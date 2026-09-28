"""
Rendering Gaussians through gsplat.

- `render_gaussians`: rasterize one camera, 3DGS or 2DGS
- `gaussian_normals_in_camera_frame`: per-Gaussian normals for 3DGS
- `render_views`: re-render a trained model one view at a time
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs
from gsplat.utils import depth_to_normal, normalized_quat_to_rotmat
from torch import Tensor

from collab_splats.splats.cameras import CameraOpt

# Annotation-only: a runtime import would be circular
if TYPE_CHECKING:
    from collab_splats.splats.gaussian import Gaussians
    from collab_splats.splats.scaffold import Scaffold


def gaussian_normals_in_camera_frame(quats: Tensor, scales: Tensor, means: Tensor, world_to_cam: Tensor) -> Tensor:
    """
    Camera-frame normal per Gaussian: its shortest axis, flipped to face the camera.

    - no gradient reaches `scales` through the normal (`argmin`)

    Args:
        quats: (N, 4) rotations.
        scales: (N, 3) axis lengths.
        means: (N, 3) world-frame centers.
        world_to_cam: (4, 4) view matrix.

    Returns:
        (N, 3) camera-frame normals.
    """
    # World-frame normal: each Gaussian's shortest axis
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

    # Flip normals that point away from the camera
    faces_away = (normals_cam * means_cam).sum(-1, keepdim=True) > 0
    return torch.where(faces_away, -normals_cam, normals_cam)


def render_gaussians(
    primitive: str,
    decoded: dict[str, Tensor],
    cam_to_world: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int | None,
    render_normals: bool = True,
) -> tuple[dict[str, Tensor], dict]:
    """
    Rasterize activated Gaussians for one camera.

    Args:
        primitive: "3dgs" or "2dgs".
        decoded: activated means/quats/scales/opacities/colors.
        cam_to_world: (1, 4, 4) one camera pose.
        intrinsics: (1, 3, 3) camera matrix.
        width: render width, px.
        height: render height, px.
        sh_degree: SH degree, or None when colors are plain (N, 3) RGB.
        render_normals: add the normal maps (2DGS always returns `normal`).

    Returns:
        (render, gsplat info). Maps are (1, H, W, C) in the camera frame:

        - always: `rgb`, `alpha`, `depth`
        - with normals: `normal`, `depth_normal` (2DGS also `depth_normal_median`)
        - 2DGS only: `distortion`, `median_depth`; its `normal` is not unit length
    """
    assert cam_to_world.shape[0] == 1, "render_gaussians renders one camera at a time"

    # Arguments shared by both rasterizers
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
        render_mode="RGB+ED",
    )

    # 2DGS plain RGB colors must be (C, N, 3); SH and 3DGS broadcast
    if primitive == "2dgs" and sh_degree is None and colors.dim() == 2:
        shared_kwargs["colors"] = colors[None].expand(len(intrinsics), -1, -1)

    # Depth normals are computed in the camera frame (identity pose)
    identity_pose = torch.eye(4, device=cam_to_world.device)[None]

    # 2DGS: normals come back in the world frame; rotate them into the camera
    if primitive == "2dgs":
        rgb_depth, alpha, normal_world, _depth_normal_world, distortion, median_depth, info = rasterization_2dgs(
            **shared_kwargs, distloss=True
        )
        rgb = rgb_depth[..., :3]
        depth = rgb_depth[..., 3:4]
        rotation_w2c = world_to_cam[:, :3, :3]
        normal_cam = torch.einsum("cij,chwj->chwi", rotation_w2c, normal_world)

        # Keep the median depth (RaDe-GS): sharper edges, used by the mesh stage
        render = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": depth,
            "median_depth": median_depth,
            "normal": normal_cam,
            "distortion": distortion,
        }

        # Depth normals only when asked for; they feed the normal consistency loss
        if render_normals:
            render["depth_normal"] = depth_to_normal(depth, identity_pose, intrinsics)
            render["depth_normal_median"] = depth_to_normal(median_depth, identity_pose, intrinsics)

        return render, info

    # 3DGS normals as an extra signal, padded to 4 channels (the kernel rejects 3)
    extra_signals = None

    if render_normals:
        normals_cam = gaussian_normals_in_camera_frame(quats, scales, means, world_to_cam[0])
        extra_signals = torch.cat([normals_cam, torch.zeros_like(normals_cam[:, :1])], dim=-1)

    rgb_depth, alpha, info = rasterization(**shared_kwargs, rasterize_mode="antialiased", extra_signals=extra_signals)
    depth = rgb_depth[..., 3:4]
    render = {"rgb": rgb_depth[..., :3], "alpha": alpha, "depth": depth}

    # Normal maps only when asked for; rasterized normals come back in `info`
    if render_normals:
        render["normal"] = F.normalize(info["render_extra_signals"][..., :3], dim=-1)
        render["depth_normal"] = depth_to_normal(depth, identity_pose, intrinsics)

    return render, info


########################################################################################
# View re-rendering
########################################################################################


# no_grad as a decorator, never a `with` in the body
# - a generator paused inside `with` would leave autograd off process-wide
@torch.no_grad()
def render_views(
    model: "Gaussians | Scaffold",
    camera_opt: CameraOpt,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    height: int,
    width: int,
) -> Iterator[dict[str, Tensor]]:
    """
    Re-render every view from a trained model, one at a time.

    - applies the color correction only; `cam_to_world` must already be pose-corrected

    Args:
        model: a trained `Gaussians` or `Scaffold`.
        camera_opt: the run's per-camera corrections.
        cam_to_world: (N, 4, 4) camera poses.
        intrinsics: (N, 3, 3) camera matrices at (height, width).
        height: render height, px.
        width: render width, px.

    Yields:
        One render dict per view: `rgb` (1, H, W, 3) in [0, 1], `depth`, `alpha`, `normal`
        (2DGS also `median_depth`).
    """
    device = cam_to_world.device

    for view in range(len(cam_to_world)):
        camera_id = torch.tensor([view], device=device)
        render, _ = model.render(
            cam_to_world[view : view + 1],
            intrinsics[view : view + 1],
            width,
            height,
            camera_id,
            step=None,
        )
        render["rgb"] = camera_opt.color(render["rgb"], camera_id).clamp(0, 1)
        yield render
