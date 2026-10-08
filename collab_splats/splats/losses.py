"""
Training losses and their per-step weight schedule.

- schedule: default, validation and per-step weights (`name: {weight[, start, end, end_weight]}`)
- `compute_losses`: photometric L1 + SSIM plus every active optional loss
- optional losses share one signature and return None when their input is missing
- Scaffold scale_reg: city-super/Scaffold-GS @ 59c833b5, train.py:144-145; its 2dgs area form follows
  yanxian-ll/GS-SR @ 566359be, gssr/scene/scaffold_2dgs_scene.py:17-25
"""

import torch
from torch import Tensor

from gsplat import losses as gsplat_losses

########################################
# Schedule
########################################


def default_losses(primitive: str) -> dict[str, dict]:
    """
    Default loss schedule for a primitive.

    Args:
        primitive: "3dgs" or "2dgs".

    Returns:
        A new `{name: spec}`, safe to edit in place.
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


# Extra spec keys a loss accepts beyond the schedule keys
LOSS_SPEC_KEYS = {
    "normal_consistency": {"depth_ratio"},
}


def validate_schedule(losses: dict[str, dict], primitive: str) -> None:
    """
    Check a loss schedule; raise on unknown losses, bad specs or wrong primitive.

    Args:
        losses: the yaml `splats.losses` mapping.
        primitive: "3dgs" or "2dgs".

    Raises:
        ValueError: naming the offending key.
    """
    for name, spec in losses.items():
        if name not in OPTIONAL_LOSSES:
            raise ValueError(f"splats.losses: unknown loss '{name}'; allowed {sorted(OPTIONAL_LOSSES)}")

        # Allowed keys: the schedule keys plus this loss's own
        allowed_spec_keys = {"weight", "start", "end", "end_weight"} | LOSS_SPEC_KEYS.get(name, set())
        unknown_spec_keys = set(spec) - allowed_spec_keys

        if unknown_spec_keys or "weight" not in spec:
            # Report the keys this loss allows
            optional_keys = ", ".join(sorted(allowed_spec_keys - {"weight"}))
            raise ValueError(f"splats.losses.{name}: expected {{weight[, {optional_keys}]}}, got {sorted(spec)}")

        # Decay needs positive weights at both ends and end > start
        if "end" in spec:
            end_weight = spec.get("end_weight")

            if end_weight is None or spec["weight"] <= 0 or end_weight <= 0 or spec["end"] <= spec.get("start", 0):
                raise ValueError(
                    f"splats.losses.{name}: decay needs weight > 0, end_weight > 0 and end > start, got {spec}"
                )

    # Distortion is 2dgs-only
    distortion_weight = losses.get("distortion", {}).get("weight", 0.0)

    if primitive == "3dgs" and distortion_weight > 0:
        raise ValueError("splats.losses.distortion is 2dgs-only; set its weight to 0 or use primitive: 2dgs")

    # depth_ratio must be a real number; yaml booleans are refused
    raw_depth_ratio = losses.get("normal_consistency", {}).get("depth_ratio", 0.0)

    if isinstance(raw_depth_ratio, bool) or not isinstance(raw_depth_ratio, (int, float)):
        raise ValueError(
            f"splats.losses.normal_consistency.depth_ratio must be a number in [0, 1], got {raw_depth_ratio!r}"
        )

    # depth_ratio must lie in [0, 1]
    depth_ratio = float(raw_depth_ratio)

    if not 0.0 <= depth_ratio <= 1.0:
        raise ValueError(f"splats.losses.normal_consistency.depth_ratio must be in [0, 1], got {depth_ratio}")

    # A non-zero depth_ratio needs median depth, which only 2dgs renders
    if depth_ratio > 0 and primitive != "2dgs":
        raise ValueError(
            "splats.losses.normal_consistency.depth_ratio > 0 is 2dgs-only "
            "(median depth is a rasterization_2dgs output); set it to 0 or use primitive: 2dgs"
        )


def loss_weight(step: int, spec: dict | None) -> float:
    """
    Weight of a schedule entry at `step`.

    - 0 before `start`; with `end`, decays log-linearly to `end_weight`, then holds

    Args:
        step: current training step.
        spec: the schedule entry, or None.

    Returns:
        The weight; 0.0 for a missing entry.
    """
    if spec is None or step < spec.get("start", 0):
        return 0.0

    # Refuse yaml booleans as weights
    for key in ("weight", "end_weight"):
        if isinstance(spec.get(key), bool):
            raise TypeError(f"loss {key} must be a number, got {spec[key]!r} — yaml booleans are not weights")

    weight = spec["weight"]
    end = spec.get("end")

    if end is None:
        return weight

    if step >= end:
        return spec["end_weight"]

    # Log-linear interpolation between the two weights
    start = spec.get("start", 0)
    fraction = (step - start) / (end - start)
    return weight * (spec["end_weight"] / weight) ** fraction


def loss_active(step: int, spec: dict | None) -> bool:
    """
    Whether a schedule entry contributes at `step`.

    Args:
        step: current training step.
        spec: the schedule entry, or None.

    Returns:
        True when its weight is > 0.
    """
    return loss_weight(step, spec) > 0


def rescale_depth_units(losses: dict[str, dict], scale: float) -> dict[str, dict]:
    """
    Loss schedule with depth-unit weights adjusted for scene normalization.

    - only distortion scales with depth, so only its weights are divided by `scale`

    Args:
        losses: the loss schedule; not modified.
        scale: the scale `utils.scene_normalization` returned.

    Returns:
        The adjusted schedule.
    """
    if "distortion" not in losses:
        return losses

    rescaled = dict(losses)
    distortion = dict(rescaled["distortion"])
    distortion["weight"] = distortion["weight"] / scale

    if "end_weight" in distortion:
        distortion["end_weight"] = distortion["end_weight"] / scale

    rescaled["distortion"] = distortion
    return rescaled


########################################
# Weighted sum and the optional losses
########################################


def compute_losses(
    step: int,
    render: dict,
    target: dict,
    gaussians: torch.nn.ParameterDict,
    loss_schedule: dict[str, dict],
    scene_scale: float,
    *,
    l1_weight: float = 0.8,
    ssim_weight: float = 0.2,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Weighted sum of the losses active at `step`.

    - values are detached 0-d tensors: no host sync until the caller reads them

    Args:
        step: current training step.
        render: render dict for the view.
        target: ground truth, `rgb` and optionally `depth`.
        gaussians: the model's raw parameters.
        loss_schedule: `{name: spec}` over `OPTIONAL_LOSSES`.
        scene_scale: camera extent of the training frame.
        l1_weight: photometric L1 weight.
        ssim_weight: photometric (1 - SSIM) weight.

    Returns:
        (total loss, {name: detached value}).
    """
    # Photometric: L1 + SSIM (ssim_loss wants NCHW)
    rendered_rgb = render["rgb"]
    target_rgb = target["rgb"]
    rendered_nchw = rendered_rgb.permute(0, 3, 1, 2)
    target_nchw = target_rgb.permute(0, 3, 1, 2)
    l1 = gsplat_losses.l1_loss(rendered_rgb, target_rgb).mean()
    ssim = gsplat_losses.ssim_loss(rendered_nchw, target_nchw)
    total = l1_weight * l1 + ssim_weight * ssim
    values = {"l1": l1.detach(), "ssim": ssim.detach()}

    # Optional losses: skip zero weights and missing inputs
    for name, spec in loss_schedule.items():
        weight = loss_weight(step, spec)

        if weight <= 0:
            continue

        loss_fn = OPTIONAL_LOSSES[name]
        value = loss_fn(render, target, gaussians, scene_scale, spec)

        if value is None:
            continue

        total = total + weight * value
        values[name] = value.detach()

    return total, values


########################################
# Optional losses — each returns None when its input is absent
########################################


def depth_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    Disparity L1 against the target depth, where it exists.

    Args:
        render: needs `depth`, (1, H, W, 1).
        target: `depth`, same shape; 0 = no target.
        gaussians: unused.
        scene_scale: passed to gsplat's depth L1.
        spec: unused.

    Returns:
        Scalar loss, or None when the target has no depth.
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
    Cosine distance between rendered normals and normals from rendered depth.

    - `depth_ratio` blends the losses against expected and median depth normals
    - RaDe-GS semantics (BaowenZ/RaDe-GS @ d72f2079, train.py:166-169), not 2DGS's

    Args:
        render: needs `normal`, `depth_normal`, `alpha`; `depth_normal_median` when blending.
        target: unused.
        gaussians: unused.
        scene_scale: unused.
        spec: optional `depth_ratio` in [0, 1], default 0.

    Returns:
        Scalar loss.

    Raises:
        ValueError: the render has no normals.
    """
    rendered_normal = render.get("normal")
    depth_normal = render.get("depth_normal")

    if rendered_normal is None or depth_normal is None:
        raise ValueError("normal_consistency is active but the render has no normals; render with render_normals=True")

    # Weight depth normals by detached alpha, as gsplat's simple_trainer_2dgs.py
    # - not unit-norm, so GSPLAT_ENFORCE_CONTRACTS=1 asserts here
    alpha = render["alpha"].detach()
    expected_term = gsplat_losses.normal_cosine_loss(rendered_normal, depth_normal * alpha).mean()

    # No blend: expected-depth term only
    ratio = float(spec.get("depth_ratio", 0.0))

    if ratio <= 0.0:
        return expected_term

    # Blend in the median-depth term (2dgs only)
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
    Mean of the 2DGS distortion map.

    Args:
        render: `distortion`, rendered by 2dgs only.
        target: unused.
        gaussians: unused.
        scene_scale: unused.
        spec: unused.

    Returns:
        Scalar loss, or None without a distortion map.
    """
    distortion_map = render.get("distortion")

    if distortion_map is None:
        return None

    return distortion_map.mean()


def opacity_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    Mean opacity penalty (gsplat MCMC).

    - Scaffold's decoded opacities are already activated, so they are averaged directly

    Args:
        render: `opacities` when decoded per view (Scaffold).
        target: unused.
        gaussians: the raw `opacities`, used when the render has none.
        scene_scale: unused.
        spec: unused.

    Returns:
        Scalar penalty.
    """
    decoded_opacities = render.get("opacities")

    if decoded_opacities is not None:
        return decoded_opacities.mean()

    return gsplat_losses.opacity_reg_loss(gaussians["opacities"])


def scale_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    Scale penalty: mean scale for Gaussians, mean volume for Scaffold.

    - under 2dgs the zeroed third log-scale makes the volume an area

    Args:
        render: `log_scales` when decoded per view (Scaffold).
        target: unused.
        gaussians: the raw `scales`, used when the render has none.
        scene_scale: unused.
        spec: unused.

    Returns:
        Scalar penalty.
    """
    log_scales = render.get("log_scales")

    if log_scales is None:
        return gsplat_losses.scale_reg_loss(gaussians["scales"])

    return torch.exp(log_scales).prod(dim=-1).mean()


def appearance_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    Pull the view's appearance correction towards identity.

    Args:
        render: `appearance`, the view's color parameters.
        target: unused.
        gaussians: unused.
        scene_scale: unused.
        spec: unused.

    Returns:
        Their mean square, or None when appearance is off.
    """
    params = render.get("appearance")

    if params is None:
        return None

    return params.square().mean()


########################################
# Registries
########################################

# yaml loss name -> function; also the allow-list for validation
OPTIONAL_LOSSES = {
    "depth": depth_loss,
    "normal_consistency": normal_consistency_loss,
    "distortion": distortion_loss,
    "opacity_reg": opacity_reg_loss,
    "scale_reg": scale_reg_loss,
    "appearance_reg": appearance_reg_loss,
}
