"""
Loss registry and the scheduled weighted sum over it.

- photometric (0.8 L1 + 0.2 (1 - SSIM)): always on
- every optional loss shares the signature ``(render, target, gaussians, scene_scale, spec)``;
  the ``Args:`` below name only the entries a given loss reads
- ``compute_losses`` walks the yaml schedule ``name: {weight[, start, end, end_weight]}`` and adds
  a loss iff its weight at the step is > 0 and the function returns a value
- with ``end``: the weight decays log-linearly from ``weight`` at ``start`` to ``end_weight`` at
  ``end``, and holds there
- a loss gets its FULL yaml spec, but the scheduling keys are the caller's — ``compute_losses`` has
  applied them already, so a loss reading them double-applies
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
# Schedule
########################################


def default_losses(primitive: str) -> dict[str, dict]:
    """
    Default loss schedule per primitive: MCMC regularizers for 3dgs, distortion for 2dgs.

    Args:
        primitive: "3dgs" or "2dgs".

    Returns:
        A fresh `{name: spec}` — mapping and specs both — so a caller may merge its yaml over it
        in place without the next call inheriting the edit.
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


# Per-loss tuning keys beyond {weight, start, end, end_weight}
# - depth_ratio: the RaDe-GS median-normal blend
# - pgsr_*: PGSR's own (yanxian-ll/GS-SR @ 566359be, gssr/scene/pgsr_scene.py:18-26), kept on the
#   spec rather than promoted to SplatsConfig because they mean nothing when the loss is off
# - plain data, so it sits above the validator; OPTIONAL_LOSSES maps to functions and must follow
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


def validate_schedule(losses: dict[str, dict], primitive: str) -> None:
    """
    Reject unknown losses, ill-formed specs, and primitive/loss mismatches.

    - `OPTIONAL_LOSSES` sits at the foot of the module and resolves when this is CALLED, not
      when it is defined

    Args:
        losses: the yaml `splats.losses` mapping, `{name: {weight[, start, end, end_weight, ...]}}`.
        primitive: "3dgs" or "2dgs"; decides the distortion and depth_ratio guards.

    Raises:
        ValueError: naming the offending key. Returning is the pass.
    """
    for name, spec in losses.items():
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
    # - this guard names the primitive it REJECTS, the depth_ratio guard below the one it REQUIRES
    # - the two agree over today's PRIMITIVES; a third would be waved through here, refused there
    distortion_weight = losses.get("distortion", {}).get("weight", 0.0)
    if primitive == "3dgs" and distortion_weight > 0:
        raise ValueError("splats.losses.distortion is 2dgs-only; set its weight to 0 or use primitive: 2dgs")

    # bool is an int subclass: `depth_ratio: yes` would coerce to a silent full median blend
    # - bools and non-numbers (a quoted '0.6' too) are refused before any coercion
    # - the same trap on `weight`/`end_weight` is caught per step in loss_weight, not here
    raw_depth_ratio = losses.get("normal_consistency", {}).get("depth_ratio", 0.0)
    if isinstance(raw_depth_ratio, bool) or not isinstance(raw_depth_ratio, (int, float)):
        raise ValueError(
            f"splats.losses.normal_consistency.depth_ratio must be a number in [0, 1], got {raw_depth_ratio!r}"
        )

    # A value outside [0, 1] is not a blend at all, so it is rejected before the primitive question
    depth_ratio = float(raw_depth_ratio)
    if not 0.0 <= depth_ratio <= 1.0:
        raise ValueError(f"splats.losses.normal_consistency.depth_ratio must be in [0, 1], got {depth_ratio}")

    # Median depth only exists for 2DGS, so a non-zero blend on 3dgs is a config error
    if depth_ratio > 0 and primitive != "2dgs":
        raise ValueError(
            "splats.losses.normal_consistency.depth_ratio > 0 is 2dgs-only "
            "(median depth is a rasterization_2dgs output); set it to 0 or use primitive: 2dgs"
        )


def loss_weight(step: int, spec: dict | None) -> float:
    """
    Weight of a schedule entry at `step`.

    Args:
        step: current training step.
        spec: the schedule entry, or None.

    Returns:
        0.0 before `start`, and for a missing entry. From `start`: `weight`, or its log-linear
        decay to `end_weight` across [start, end], held at `end_weight` from `end` on.
    """
    if spec is None or step < spec.get("start", 0):
        return 0.0

    # yaml reads `weight: yes` as True and True floats to 1.0 — refuse it rather than coerce
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

    Args:
        step: current training step.
        spec: the schedule entry, or None.

    Returns:
        True iff `loss_weight` is > 0.
    """
    return loss_weight(step, spec) > 0


def rescale_depth_units(losses: dict[str, dict], scale: float) -> dict[str, dict]:
    """
    Rescale the depth-unit loss weights for a scene normalized by `scale`.

    - the 2dgs distortion loss is linear in depth: the same yaml weight penalizes `scale` times
      harder in a unit-cube frame, so dividing it through restores the world-frame penalty

    Args:
        losses: the loss schedule. Not mutated — the caller's copy may be the yaml dict
            `asdict(cfg)` writes into ckpt.pt.
        scale: the factor `utils.scene_normalization` returned.

    Returns:
        A fresh mapping, `distortion`'s `weight` (and `end_weight`, when scheduled) divided by
        `scale`; every other entry passed through.
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


def neighbor_selection(spec: dict) -> dict[str, int]:
    """
    The two `pgsr_multiview` keys read once before training, not per step.

    - they configure `pgsr.select_near_views`, so they default here, not at the trainer's call site
    - `select_near_views`' scoring knobs (`theta0`, `sigma_below`, `sigma_above`) are deliberately
      NOT forwarded and no yaml key reaches them — pinned at their upstream defaults

    Args:
        spec: the `pgsr_multiview` schedule entry; its other keys are read per step inside
            `pgsr_multiview_loss`.

    Returns:
        `{"num_views", "max_points"}`, ready to splat into `select_near_views`.
    """
    return {
        "num_views": int(spec.get("num_multi_view", 5)),
        "max_points": int(spec.get("max_points", 20000)),
    }


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
) -> tuple[Tensor, dict[str, float]]:
    """
    Weighted sum of the losses active at `step`.

    Args:
        step: current training step; picks each entry's scheduled weight.
        render: the render dict the losses read.
        target: the ground truth — `rgb`, optionally `depth`.
        gaussians: the model's raw parameters, for the regularizers.
        loss_schedule: `{name: spec}` over `OPTIONAL_LOSSES`.
        scene_scale: scene normalization scale, forwarded to the losses that take it.
        l1_weight: photometric L1 weight; 0.8 is what every existing run trained at.
        ssim_weight: photometric (1 - SSIM) weight; 0.2 likewise.

    Returns:
        (total, {name: value}); the per-loss `.item()` is one GPU sync per loss per step,
        as upstream.
    """
    # Photometric: L1 + SSIM between rendered and target RGB (ssim_loss wants NCHW)
    rendered_rgb = render["rgb"]
    target_rgb = target["rgb"]
    rendered_nchw = rendered_rgb.permute(0, 3, 1, 2)
    target_nchw = target_rgb.permute(0, 3, 1, 2)
    l1 = gsplat_losses.l1_loss(rendered_rgb, target_rgb).mean()
    ssim = gsplat_losses.ssim_loss(rendered_nchw, target_nchw)
    total = l1_weight * l1 + ssim_weight * ssim
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


########################################
# Optional losses — each returns None when its input is absent
########################################


def depth_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    Disparity L1 against the depth target on the pixels that have one.

    Args:
        render: needs `depth`, (1, H, W, 1).
        target: needs `depth`, same shape; 0 marks a pixel with no target.
        scene_scale: forwarded to gsplat's depth L1.

    Returns:
        Scalar loss over the targeted pixels, or None when the target carries no depth.
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

    - `depth_ratio` blends two LOSSES, not two depths: `(1 - r) * d(n, dn_expected) +
      r * d(n, dn_median)` over the per-pixel cosine distance `d(n, m) = 1 - cos(n, m)`
    - upstream: RaDe-GS semantics (BaowenZ/RaDe-GS @ d72f2079, train.py:166-169), NOT
      upstream-2DGS's `depth_ratio`, which mixes the depths into one surf_depth first

    Args:
        render: needs `normal`, `depth_normal`, `alpha`; `depth_normal_median` when blending.
        spec: `depth_ratio` (default 0.0), the median-normal blend factor.

    Returns:
        Scalar loss.

    Raises:
        ValueError: no normals in the render. The trainer gates `render_normals` on
            `loss_active`, so it is a wiring bug, not a skip condition.
    """
    rendered_normal = render.get("normal")
    depth_normal = render.get("depth_normal")
    if rendered_normal is None or depth_normal is None:
        raise ValueError("normal_consistency is active but the render has no normals; render with render_normals=True")

    # depth_normal scaled by detached alpha, for parity with upstream simple_trainer_2dgs.py
    # - scales the GRADIENT so empty pixels stop pulling; the value keeps a (1 - alpha) offset there
    # - the scaled vector is not unit-norm, so GSPLAT_ENFORCE_CONTRACTS=1 trips a norm assert here
    alpha = render["alpha"].detach()
    expected_term = gsplat_losses.normal_cosine_loss(rendered_normal, depth_normal * alpha).mean()

    # depth_ratio 0 is the shipped behavior, bit-for-bit
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
    Mean of the 2DGS rasterizer's per-pixel distortion map.

    Args:
        render: needs `distortion`, which only the 2DGS rasterizer produces.

    Returns:
        Scalar mean, or None for a primitive with no distortion map.
    """
    distortion_map = render.get("distortion")
    if distortion_map is None:
        return None
    return distortion_map.mean()


def opacity_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    Opacity regularizer from gsplat (MCMC); expects raw logit opacities.

    Args:
        render: `opacities` when the model decodes them per view (scaffold).
        gaussians: the `opacities` parameter, read when the render carries none.

    Returns:
        Scalar penalty. Scaffold's decoded opacities are ALREADY activated, so they are averaged
        directly rather than pushed through `gsplat_losses.opacity_reg_loss`, which sigmoids first.
    """
    decoded_opacities = render.get("opacities")
    if decoded_opacities is not None:
        return decoded_opacities.mean()
    return gsplat_losses.opacity_reg_loss(gaussians["opacities"])


def scale_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    Scale regularizer; expects raw log scales.

    Args:
        render: `log_scales` when the model decodes them per view (scaffold).
        gaussians: the `scales` parameter, read when the render carries none.

    Returns:
        Scalar penalty.

        - vanilla: gsplat's MCMC form, the mean of the exponentiated scales
        - scaffold: the decoded VOLUME, `scaling.prod(dim=1).mean()` (GS-SR @ 566359be,
          gssr/scene/scaffold_scene.py:184)
        - scaffold under 2dgs: the third channel is zeroed at decode, so exp() makes it 1 and the
          product is upstream's 2-channel area (GS-SR @ 566359be, scaffold_2dgs_scene.py:25)
    """
    log_scales = render.get("log_scales")
    if log_scales is None:
        return gsplat_losses.scale_reg_loss(gaussians["scales"])
    return torch.exp(log_scales).prod(dim=-1).mean()


def appearance_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    Pull the view's per-image appearance params towards identity.

    Args:
        render: `appearance`, the view's params; absent when appearance opt is off.

    Returns:
        Their mean square, or None when appearance optimization is off.
    """
    params = render.get("appearance")
    if params is None:
        return None
    return params.square().mean()


def pgsr_normal_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    PGSR single-view planar loss: L1 between plane normal and plane-depth normal, on flat regions.

    - upstream: GS-SR @ 566359be, gssr/scene/pgsr_scene.py:107-112, the `# sigle-view loss` block
      of `get_loss_dict` (their typo, kept so the block stays greppable)
    - upstream scales the depth normal by detached alpha inside its renderer; `render_plane` keeps
      that key pure geometry, so the scaling happens here instead

    Args:
        render: needs `plane_normal`, `plane_depth_normal`, `alpha`.
        target: needs `rgb` — the flat-region weight is read off it, so it is data, not a fit.
        spec: `erode_ksize` (default 5), the erosion width of that weight.

    Returns:
        Scalar, weighted by the detached, eroded `(1 - image gradient)^5` — textured pixels
        contribute ~0, so surfaces flatten without straightening real depth discontinuities.

    Raises:
        ValueError: no plane maps in the render. The trainer gates `render_plane` on this same
            schedule, so it is a wiring bug, not a skip condition.
    """
    plane_normal = render.get("plane_normal")
    plane_depth_normal = render.get("plane_depth_normal")
    if plane_normal is None or plane_depth_normal is None:
        raise ValueError("pgsr_normal is active but the render has no plane maps; render with render_plane=True")

    # The flat-region weight is read off the TARGET image, so it is data, not something to fit
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

    - upstream: GS-SR @ 566359be, gssr/scene/pgsr_scene.py:114-198, the `# multi-view loss` block
      of `get_loss_dict`
    - both terms are ONE schedule entry: upstream computes them from a single shared
      correspondence pass, and the entry's own `weight` multiplies their sum

    Args:
        render: needs `pgsr_neighbor`, `plane_depth`, `plane_normal`, `plane_distance`,
            `world_to_cam`, `intrinsics`.
        target: needs `rgb`, the reference image the patches are cut from.
        spec: `geo` (0.03), `ncc` (0.15) — upstream's `lambda_geo`/`lambda_ncc`; plus
            `pixel_noise_threshold` (1.0 px), `num_sample` (102400 patches), `patch_size`
            (3, half-width, so 7x7). `num_multi_view`/`max_points` are the trainer's.

    Returns:
        `geo * geometric + ncc * photometric`, or None when no neighbor was rendered — no
        co-visible partner, or a round-trip that kept no pixel.
    """
    neighbor = render.get("pgsr_neighbor")
    if neighbor is None:
        return None

    # Round trip: reference pixel -> its plane depth -> the neighbor's surface -> back
    # - two views that disagree about the surface land the pixel away from where it started
    world_to_cam, intrinsics = render["world_to_cam"], render["intrinsics"]
    pixel_noise, valid = forward_backward_noise(
        render["plane_depth"],
        world_to_cam,
        intrinsics,
        neighbor["plane_depth"],
        neighbor["world_to_cam"],
        neighbor["intrinsics"],
    )
    valid = valid & (pixel_noise < float(spec.get("pixel_noise_threshold", 1.0)))
    if not bool(valid.any()):
        return None

    # exp(-noise) down-weights already-consistent pixels; detached, so the weight is not a target
    weights = torch.where(valid, (1.0 / torch.exp(pixel_noise)).detach(), torch.zeros_like(pixel_noise))
    geometric = (weights * pixel_noise)[valid].mean()

    # Sample the surviving pixels down to a fixed patch budget; which pixels is not differentiable
    with torch.no_grad():
        indices = valid.nonzero(as_tuple=False)[:, 0]
        num_sample = int(spec.get("num_sample", 102400))
        if len(indices) > num_sample:
            indices = indices[torch.randperm(len(indices), device=indices.device)[:num_sample]]
        sample_weights = weights[indices]

    # Photometric: warp each reference patch into the neighbor through ITS OWN rendered plane
    # - the gradient reaches the plane normal and distance, so a wrong plane is what NCC penalizes
    height, width = render["plane_depth"].shape[1:3]
    pixels = pixel_grid(height, width, render["plane_depth"].device)[indices]
    ncc, keep = patch_ncc(
        to_gray(target["rgb"][0]),
        neighbor["gray"],
        pixels,
        render["plane_normal"].reshape(-1, 3)[indices],
        render["plane_distance"].reshape(-1)[indices],
        world_to_cam,
        intrinsics,
        neighbor["world_to_cam"],
        neighbor["intrinsics"],
        half_patch=int(spec.get("patch_size", 3)),
    )
    keep = keep.reshape(-1)
    total = float(spec.get("geo", 0.03)) * geometric
    if bool(keep.any()):
        total = total + float(spec.get("ncc", 0.15)) * (ncc.reshape(-1) * sample_weights)[keep].mean()
    return total


########################################
# Registries
########################################

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
