"""
Loss registry and the scheduled weighted sum over it.

Photometric (0.8 L1 + 0.2 (1 - SSIM)) is always on. Each optional loss is one small function
with the same signature; ``compute_losses`` loops over the yaml schedule ``name: {weight, start}``
and adds a loss iff weight > 0, step >= start, and the function returns a value.
"""

import torch
from gsplat import losses as gsplat_losses
from torch import Tensor

########################################
# Optional losses — each returns None when its input is absent
########################################


def depth_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor | None:
    """
    Disparity L1 against the depth target on the pixels that have one (0 = no target).

    - `render["depth"]` and `target["depth"]` are both `(1, H, W, 1)`; None target -> None.
    """
    target_depth = target.get("depth")
    if target_depth is None:
        return None

    # Only pixels with a target contribute
    rendered_depth = render["depth"]
    assert rendered_depth.shape == target_depth.shape, (rendered_depth.shape, target_depth.shape)
    has_target = target_depth > 0
    rendered_depth = rendered_depth[has_target]
    target_depth = target_depth[has_target]
    return gsplat_losses.depth_l1_loss(rendered_depth, target_depth, scene_scale)


def normal_consistency_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float
) -> Tensor | None:
    """
    Cosine distance between rendered normals and normals finite-differenced from rendered depth.

    - None when the render carries no normals (e.g. a renderer without normal output), so the loss is skipped.
    """
    rendered_normal = render.get("normal")
    depth_normal = render.get("depth_normal")
    if rendered_normal is None or depth_normal is None:
        return None

    # Scaling depth_normal by detached alpha scales the GRADIENT so empty pixels stop pulling; the
    # reported value still carries a (1 - alpha) offset on those pixels. Parity with upstream
    # simple_trainer_2dgs.py. The scaled vector is no longer unit-norm, so GSPLAT_ENFORCE_CONTRACTS=1
    # trips normal_cosine_loss's norm assert here by design.
    alpha = render["alpha"].detach()
    depth_normal = depth_normal * alpha
    cosine_distance = gsplat_losses.normal_cosine_loss(rendered_normal, depth_normal)
    return cosine_distance.mean()


def distortion_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor | None:
    """
    Mean of the 2DGS rasterizer's per-pixel distortion map; None when the primitive has none.
    """
    distortion_map = render.get("distortion")
    if distortion_map is None:
        return None
    return distortion_map.mean()


def opacity_reg_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor:
    """
    Opacity regulariser from gsplat (MCMC); expects raw logit opacities.
    """
    opacities = gaussians["opacities"]
    return gsplat_losses.opacity_reg_loss(opacities)


def scale_reg_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor:
    """
    Scale regulariser from gsplat (MCMC); expects raw log scales.
    """
    log_scales = gaussians["scales"]
    return gsplat_losses.scale_reg_loss(log_scales)


# Name in the yaml `losses:` block -> function. Also the allow-list for config validation.
OPTIONAL_LOSSES = {
    "depth": depth_loss,
    "normal_consistency": normal_consistency_loss,
    "distortion": distortion_loss,
    "opacity_reg": opacity_reg_loss,
    "scale_reg": scale_reg_loss,
}

########################################
# Weighted sum
########################################


def compute_losses(
    step: int,
    render: dict,
    target: dict,
    gaussians: torch.nn.ParameterDict,
    loss_schedule: dict[str, dict],
    scene_scale: float,
) -> tuple[Tensor, dict[str, float]]:
    """
    Weighted sum of the losses active at `step`. Returns (total, {name: value}).

    - The `.item()` per reported loss is one GPU sync per loss per step (upstream does the same).
    """
    # Photometric: L1 + SSIM between rendered and target RGB (ssim_loss wants NCHW)
    rendered_rgb = render["rgb"]
    target_rgb = target["rgb"]
    rendered_nchw = rendered_rgb.permute(0, 3, 1, 2)
    target_nchw = target_rgb.permute(0, 3, 1, 2)
    l1 = gsplat_losses.l1_loss(rendered_rgb, target_rgb).mean()
    ssim = gsplat_losses.ssim_loss(rendered_nchw, target_nchw)
    total = 0.8 * l1 + 0.2 * ssim
    values = {"l1": l1.item(), "ssim": ssim.item()}

    # Optional losses: skip when not started, zero-weighted, or the loss has no input this step
    for name, spec in loss_schedule.items():
        weight = spec["weight"]
        start = spec.get("start", 0)
        if weight <= 0 or step < start:
            continue
        loss_fn = OPTIONAL_LOSSES[name]
        value = loss_fn(render, target, gaussians, scene_scale)
        if value is None:
            continue
        total = total + weight * value
        values[name] = value.item()

    return total, values
