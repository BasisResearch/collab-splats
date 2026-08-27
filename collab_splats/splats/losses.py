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

from collab_splats.splats.pgsr import (
    flat_region_weight,
    forward_backward_noise,
    patch_ncc,
    pixel_grid,
    to_gray,
)

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

    - Scaffold has no opacity parameter — its Gaussians are decoded per view — so the render carries
      the decoded opacities, which are ALREADY activated and are therefore averaged directly rather
      than pushed through `gsplat_losses.opacity_reg_loss` (that helper sigmoids its argument).
    """
    decoded_opacities = render.get("opacities")
    if decoded_opacities is not None:
        return decoded_opacities.mean()
    return gsplat_losses.opacity_reg_loss(gaussians["opacities"])


def scale_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    Scale regulariser; expects raw log scales.

    - Vanilla uses gsplat's MCMC form (mean of the exponentiated scales) over the scales parameter.
    - Scaffold has no scales parameter — its Gaussians are decoded per view — so the render carries
      `log_scales`, and the penalty is the decoded VOLUME upstream uses: lambda_scaling *
      scaling.prod(dim=1).mean() (GS-SR gssr/scene/scaffold_scene.py:184). Under 2dgs the third
      channel is zeroed at decode, so exp() makes it 1 and the product is the 2-channel area
      upstream's scaffold-2dgs scene penalises (gssr/scene/scaffold_2dgs_scene.py:25).
    """
    log_scales = render.get("log_scales")
    if log_scales is None:
        return gsplat_losses.scale_reg_loss(gaussians["scales"])
    return torch.exp(log_scales).prod(dim=-1).mean()


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


def pgsr_normal_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    PGSR single-view planar loss: L1 between the plane normal and the plane-depth normal, on flat regions.

    - Reproduces GS-SR gssr/scene/pgsr_scene.py, the `# sigle-view loss` block of `get_loss_dict`.
    - The weight is the detached `(1 - image gradient)^5`, eroded: textured pixels contribute ~0, so the
      loss flattens surfaces without straightening the depth discontinuities that edges usually mark.
    - Upstream scales the depth normal by detached alpha inside its renderer; `render_plane` keeps that
      key a pure geometric quantity, so the scaling happens here instead.
    - Raises when the render carries no plane maps: the trainer gates `render_plane` on this same
      schedule, so an active loss without them is a wiring bug, not a condition to skip silently.
    """
    plane_normal = render.get("plane_normal")
    plane_depth_normal = render.get("plane_depth_normal")
    if plane_normal is None or plane_depth_normal is None:
        raise ValueError("pgsr_normal is active but the render has no plane maps; render with render_plane=True")

    # The flat-region weight is read off the TARGET image, so it is data rather than something to fit
    weight = flat_region_weight(target["rgb"][0], ksize=int(spec.get("erode_ksize", 5)))

    # Scaling the depth normal by accumulated alpha stops empty pixels pulling, as upstream does
    alpha = render["alpha"].detach()
    residual = (plane_depth_normal * alpha - plane_normal).abs().sum(dim=-1)[0]
    return (weight * residual).mean()


def pgsr_multiview_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    PGSR multi-view loss: geometric round-trip error plus patch NCC through the plane homography.

    - Reproduces GS-SR gssr/scene/pgsr_scene.py, the `# multi-view loss` block of `get_loss_dict`.
    - Both terms are ONE schedule entry because upstream computes them from a single shared
      correspondence pass: `spec["geo"]` (0.03) and `spec["ncc"]` (0.15) are its `lambda_geo` and
      `lambda_ncc`, and the entry's own `weight` multiplies their sum.
    - Extra spec keys: `pixel_noise_threshold` (1.0 px), `num_sample` (102400 patches), `patch_size`
      (3, the half-width, so 7x7 patches). `num_multi_view` and `max_points` are read by the trainer
      once, for neighbour selection.
    - Returns None when the trainer rendered no neighbour this step — a view with no co-visible
      partner, or a round-trip that kept no pixel.
    """
    neighbour = render.get("pgsr_neighbour")
    if neighbour is None:
        return None

    # Round trip: reference pixel -> its plane depth -> the neighbour's own surface -> back. A pixel
    # whose two views disagree about where the surface is lands away from where it started.
    world_to_cam, intrinsics = render["world_to_cam"], render["intrinsics"]
    pixel_noise, valid = forward_backward_noise(
        render["plane_depth"],
        world_to_cam,
        intrinsics,
        neighbour["plane_depth"],
        neighbour["world_to_cam"],
        neighbour["intrinsics"],
    )
    valid = valid & (pixel_noise < float(spec.get("pixel_noise_threshold", 1.0)))
    if not bool(valid.any()):
        return None

    # exp(-noise) down-weights the pixels that are already nearly consistent; detached, so the weight
    # itself is not a target
    weights = torch.where(valid, (1.0 / torch.exp(pixel_noise)).detach(), torch.zeros_like(pixel_noise))
    geometric = (weights * pixel_noise)[valid].mean()

    # Sample the surviving pixels down to a fixed patch budget; which pixels is not differentiable
    with torch.no_grad():
        indices = valid.nonzero(as_tuple=False)[:, 0]
        num_sample = int(spec.get("num_sample", 102400))
        if len(indices) > num_sample:
            indices = indices[torch.randperm(len(indices), device=indices.device)[:num_sample]]
        sample_weights = weights[indices]

    # Photometric: warp each reference patch into the neighbour through ITS OWN rendered plane and
    # compare structure. The gradient goes to the plane normal and distance, so a wrong plane is what
    # the NCC penalises.
    height, width = render["plane_depth"].shape[1:3]
    pixels = pixel_grid(height, width, render["plane_depth"].device)[indices]
    ncc, keep = patch_ncc(
        to_gray(target["rgb"][0]),
        neighbour["gray"],
        pixels,
        render["plane_normal"].reshape(-1, 3)[indices],
        render["plane_distance"].reshape(-1)[indices],
        world_to_cam,
        intrinsics,
        neighbour["world_to_cam"],
        neighbour["intrinsics"],
        half_patch=int(spec.get("patch_size", 3)),
    )
    keep = keep.reshape(-1)
    total = float(spec.get("geo", 0.03)) * geometric
    if bool(keep.any()):
        total = total + float(spec.get("ncc", 0.15)) * (ncc.reshape(-1) * sample_weights)[keep].mean()
    return total


# Name in the yaml `losses:` block -> function. Also the allow-list for config validation.
OPTIONAL_LOSSES = {
    "depth": depth_loss,
    "normal_consistency": normal_consistency_loss,
    "distortion": distortion_loss,
    "opacity_reg": opacity_reg_loss,
    "scale_reg": scale_reg_loss,
    "appearance_reg": appearance_reg_loss,
    "pgsr_normal": pgsr_normal_loss,
    "pgsr_multiview": pgsr_multiview_loss,
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

    # yaml reads `weight: yes` as True, and True floats to 1.0 — a loss meant to be switched
    # on would silently train at full strength, so refuse the bool instead of coercing it
    for key in ("weight", "end_weight"):
        if isinstance(spec.get(key), bool):
            raise TypeError(f"loss {key} must be a number, got {spec[key]!r} — yaml booleans are not weights")

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
