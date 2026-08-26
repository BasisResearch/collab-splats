"""
Loss registry and the scheduled weighted sum over it.

Photometric (0.8 L1 + 0.2 (1 - SSIM)) is always on. Each optional loss is one small function
with the same signature ``(render, target, gaussians, scene_scale, spec)``; ``compute_losses``
loops over the yaml schedule ``name: {weight[, start, end, end_weight]}`` and adds a loss iff its
weight at the step is > 0 and the function returns a value. With ``end`` the weight decays
log-linearly from ``weight`` at ``start`` to ``end_weight`` at ``end`` and holds there.

A loss receives its FULL yaml spec: the scheduling keys (``weight``, ``start``, ``end``,
``end_weight``) are the caller's — ``compute_losses`` has already applied them, so a loss reads
only its own extra keys or it double-applies what the caller handled.
"""

import torch
from gsplat import losses as gsplat_losses
from torch import Tensor

########################################
# Optional losses — each returns None when its input is absent
########################################


def depth_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
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
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    Cosine distance between rendered normals and normals finite-differenced from rendered depth.

    - `spec["depth_ratio"]` (default 0.0) blends in the RaDe-GS median-depth normal, writing
      `d(n, m) = 1 - cos(n, m)` for the per-pixel cosine DISTANCE gsplat minimises:
      `(1 - r) * d(n, dn_expected) + r * d(n, dn_median)`. This is a blend of two LOSSES,
      the RaDe-GS semantics — not upstream-2DGS's `depth_ratio`, which blends the two depths
      into one surf_depth before differencing.
    - RaDe-GS refills background median depth with its max before differencing
      (`.worktrees/streaming/collab_splats/nerfstudio/models/rade_gs.py:254`); omitted here on
      purpose, because the `* alpha` scaling below already zeroes those pixels.
    - Raises when the render carries no normals: the trainer gates `render_normals` on `loss_active`, so an
      active loss without normals is a wiring bug, not a condition to skip silently.
    """
    rendered_normal = render.get("normal")
    depth_normal = render.get("depth_normal")
    if rendered_normal is None or depth_normal is None:
        raise ValueError("normal_consistency is active but the render has no normals; render with render_normals=True")

    # Scaling depth_normal by detached alpha scales the GRADIENT so empty pixels stop pulling; the
    # reported value still carries a (1 - alpha) offset on those pixels. Parity with upstream
    # simple_trainer_2dgs.py. The scaled vector is no longer unit-norm, so GSPLAT_ENFORCE_CONTRACTS=1
    # trips normal_cosine_loss's norm assert here by design.
    alpha = render["alpha"].detach()
    expected_term = gsplat_losses.normal_cosine_loss(rendered_normal, depth_normal * alpha).mean()

    # depth_ratio 0 is the shipped behaviour, bit-for-bit
    ratio = float(spec.get("depth_ratio", 0.0))
    if ratio <= 0.0:
        return expected_term

    # Median depth is a 2DGS rasterizer output; an active ratio without it is a wiring bug
    median_normal = render.get("depth_normal_median")
    if median_normal is None:
        raise ValueError(
            "normal_consistency depth_ratio > 0 needs 'depth_normal_median' in the render "
            "(2dgs only — median depth is a rasterization_2dgs output)"
        )
    median_term = gsplat_losses.normal_cosine_loss(rendered_normal, median_normal * alpha).mean()
    return (1.0 - ratio) * expected_term + ratio * median_term


def distortion_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    Mean of the 2DGS rasterizer's per-pixel distortion map; None when the primitive has none.
    """
    distortion_map = render.get("distortion")
    if distortion_map is None:
        return None
    return distortion_map.mean()


def opacity_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    Opacity regulariser from gsplat (MCMC); expects raw logit opacities.
    """
    opacities = gaussians["opacities"]
    return gsplat_losses.opacity_reg_loss(opacities)


def scale_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    Scale regulariser from gsplat (MCMC); expects raw log scales.
    """
    log_scales = gaussians["scales"]
    return gsplat_losses.scale_reg_loss(log_scales)


def appearance_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    Mean squared per-image appearance params of the rendered view (pull towards identity); None when off.
    """
    params = render.get("appearance")
    if params is None:
        return None
    return params.square().mean()


# Name in the yaml `losses:` block -> function. Also the allow-list for config validation.
OPTIONAL_LOSSES = {
    "depth": depth_loss,
    "normal_consistency": normal_consistency_loss,
    "distortion": distortion_loss,
    "opacity_reg": opacity_reg_loss,
    "scale_reg": scale_reg_loss,
    "appearance_reg": appearance_reg_loss,
}

########################################
# Weighted sum
########################################


def loss_weight(step: int, spec: dict | None) -> float:
    """
    Weight of a schedule entry at `step`: 0 before `start`, `weight` (or its log-linear decay to
    `end_weight` over [start, end], held after) from then on; a missing entry (None) is always 0.
    """
    if spec is None or step < spec.get("start", 0):
        return 0.0
    weight = spec["weight"]
    end = spec.get("end")
    if end is None:
        return weight
    if step >= end:
        return spec["end_weight"]

    # Geometric interpolation: equal ratios per step, so a 100x decay is smooth in log space
    start = spec.get("start", 0)
    fraction = (step - start) / (end - start)
    return weight * (spec["end_weight"] / weight) ** fraction


def loss_active(step: int, spec: dict | None) -> bool:
    """
    Whether a schedule entry contributes at `step`.
    """
    return loss_weight(step, spec) > 0


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
        weight = loss_weight(step, spec)
        if weight <= 0:
            continue
        loss_fn = OPTIONAL_LOSSES[name]
        value = loss_fn(render, target, gaussians, scene_scale, spec)
        if value is None:
            continue
        total = total + weight * value
        values[name] = value.item()

    return total, values
