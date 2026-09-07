"""
compute_losses: the photometric terms and the optional-loss gates.

- photometric always on
- optional losses gated by weight > 0, start step, and input presence
"""

import pytest
import torch

from collab_splats.splats.losses import (
    OPTIONAL_LOSSES,
    appearance_reg_loss,
    compute_losses,
    default_losses,
    loss_active,
    loss_weight,
    neighbor_selection,
    opacity_reg_loss,
    rescale_depth_units,
    scale_reg_loss,
    validate_schedule,
)


def _render(height=16, width=16, with_distortion=True):
    """
    A synthetic render dict with every optional-loss input present.

    - normals are unit; ``depth_normal`` is the same field perturbed, so consistency is non-zero
    - depth is offset off zero so no pixel reads as "no target"

    Args:
        height: render height.
        width: render width.
        with_distortion: include the 2dgs distortion map.

    Returns:
        Render dict keyed rgb / alpha / depth / normal / depth_normal (+ distortion).
    """
    gen = torch.Generator().manual_seed(0)
    normal = torch.nn.functional.normalize(torch.randn(1, height, width, 3, generator=gen), dim=-1)
    noisy_normal = torch.nn.functional.normalize(normal + 0.1 * torch.randn_like(normal), dim=-1)
    render = {
        "rgb": torch.rand(1, height, width, 3, generator=gen),
        "alpha": torch.rand(1, height, width, 1, generator=gen),
        "depth": torch.rand(1, height, width, 1, generator=gen) + 0.5,
        "normal": normal,
        "depth_normal": noisy_normal,
    }
    if with_distortion:
        render["distortion"] = torch.rand(1, height, width, 1, generator=gen)
    return render


def _target(height=16, width=16, with_depth=True):
    """
    A synthetic target dict whose first four rows carry no depth.

    - the zero rows exercise the depth loss's own no-target mask

    Args:
        height: target height.
        width: target width.
        with_depth: emit a depth target at all.

    Returns:
        Target dict keyed rgb / depth, depth None when not requested.
    """
    gen = torch.Generator().manual_seed(1)
    depth = torch.rand(1, height, width, 1, generator=gen) + 0.5
    depth[:, :4] = 0.0  # rows without a target
    rgb = torch.rand(1, height, width, 3, generator=gen)
    return {"rgb": rgb, "depth": depth if with_depth else None}


def _gaussians(n_points=50):
    """
    The two raw parameters the opacity and scale regularizers read.

    Args:
        n_points: primitive count.

    Returns:
        ParameterDict of opacities and scales, both zeroed.
    """
    opacities = torch.nn.Parameter(torch.zeros(n_points))
    scales = torch.nn.Parameter(torch.zeros(n_points, 3))
    return torch.nn.ParameterDict({"opacities": opacities, "scales": scales})


def test_registry_names():
    assert set(OPTIONAL_LOSSES) == {
        "depth",
        "normal_consistency",
        "distortion",
        "opacity_reg",
        "scale_reg",
        "appearance_reg",
        "pgsr_normal",
        "pgsr_multiview",
    }


def test_photometric_only_when_no_optional_losses():
    total, values = compute_losses(0, _render(), _target(), _gaussians(), {}, 1.0)
    expected = 0.8 * values["l1"] + 0.2 * values["ssim"]
    assert set(values) == {"l1", "ssim"}
    assert total.item() == pytest.approx(expected, rel=1e-5)


def test_loss_waits_for_its_start_step():
    schedule = {"depth": {"weight": 1.0, "start": 100}}
    _, before = compute_losses(99, _render(), _target(), _gaussians(), schedule, 1.0)
    _, after = compute_losses(100, _render(), _target(), _gaussians(), schedule, 1.0)
    assert "depth" not in before and "depth" in after


def test_zero_weight_skips_loss():
    schedule = {"depth": {"weight": 0.0}}
    _, values = compute_losses(0, _render(), _target(), _gaussians(), schedule, 1.0)
    assert "depth" not in values


def test_depth_loss_ignores_zero_targets_and_scales_with_scene_scale():
    schedule = {"depth": {"weight": 1.0}}
    render, target = _render(), _target()
    _, at_scale_1 = compute_losses(0, render, target, _gaussians(), schedule, 1.0)
    _, at_scale_2 = compute_losses(0, render, target, _gaussians(), schedule, 2.0)
    assert at_scale_2["depth"] == pytest.approx(2 * at_scale_1["depth"], rel=1e-5)

    # Perfect prediction on targeted pixels -> zero loss, whatever the untargeted rows hold
    render["depth"] = target["depth"].clone()
    render["depth"][:, :4] = 123.0
    _, perfect = compute_losses(0, render, target, _gaussians(), schedule, 1.0)
    assert perfect["depth"] == pytest.approx(0.0, abs=1e-6)


def test_depth_loss_skipped_without_targets():
    schedule = {"depth": {"weight": 1.0}}
    target = _target(with_depth=False)
    _, values = compute_losses(0, _render(), target, _gaussians(), schedule, 1.0)
    assert "depth" not in values


def test_normal_consistency_is_zero_for_identical_normals():
    render = _render()
    render["depth_normal"] = render["normal"].clone()
    render["alpha"] = torch.ones_like(render["alpha"])
    schedule = {"normal_consistency": {"weight": 1.0}}
    _, values = compute_losses(0, render, _target(), _gaussians(), schedule, 1.0)
    assert values["normal_consistency"] == pytest.approx(0.0, abs=1e-5)


def test_normal_consistency_is_one_when_alpha_is_zero():
    render = _render()
    render["alpha"] = torch.zeros_like(render["alpha"])
    schedule = {"normal_consistency": {"weight": 1.0}}
    _, values = compute_losses(0, render, _target(), _gaussians(), schedule, 1.0)
    assert values["normal_consistency"] == pytest.approx(1.0, abs=1e-5)


def test_distortion_skipped_on_3dgs_render():
    render_3dgs = _render(with_distortion=False)
    schedule = {"distortion": {"weight": 1.0}}
    _, values = compute_losses(0, render_3dgs, _target(), _gaussians(), schedule, 1.0)
    assert "distortion" not in values


def test_normal_consistency_raises_when_active_without_normals():
    render_no_normals = _render(with_distortion=False)
    del render_no_normals["normal"], render_no_normals["depth_normal"]
    schedule = {"normal_consistency": {"weight": 1.0}}
    with pytest.raises(ValueError, match="render_normals"):
        compute_losses(0, render_no_normals, _target(), _gaussians(), schedule, 1.0)


def test_normal_consistency_inactive_tolerates_missing_normals():
    render_no_normals = _render(with_distortion=False)
    del render_no_normals["normal"], render_no_normals["depth_normal"]
    schedule = {"normal_consistency": {"weight": 0.0}}
    _, values = compute_losses(0, render_no_normals, _target(), _gaussians(), schedule, 1.0)
    assert "normal_consistency" not in values
    schedule = {"normal_consistency": {"weight": 1.0, "start": 500}}
    _, values = compute_losses(0, render_no_normals, _target(), _gaussians(), schedule, 1.0)
    assert "normal_consistency" not in values


def test_loss_active_gates_on_weight_start_and_presence():
    assert loss_active(0, {"weight": 1.0})
    assert loss_active(500, {"weight": 1.0, "start": 500})
    assert not loss_active(0, {"weight": 0.0})
    assert not loss_active(499, {"weight": 1.0, "start": 500})
    assert not loss_active(0, None)


def test_yaml_bool_weight_is_refused_not_coerced():
    # `weight: yes` in a config reads as True, and float(True) is 1.0
    # - a loss the author meant to switch on would train at full strength, silently
    with pytest.raises(TypeError, match="yaml booleans"):
        loss_weight(0, {"weight": True})
    with pytest.raises(TypeError, match="end_weight"):
        loss_weight(0, {"weight": 0.1, "end": 100, "end_weight": False})


def test_distortion_skipped_when_render_has_no_map():
    schedule = {"distortion": {"weight": 1.0}}
    render_without_map = _render(with_distortion=False)
    _, with_map = compute_losses(0, _render(), _target(), _gaussians(), schedule, 1.0)
    _, without_map = compute_losses(0, render_without_map, _target(), _gaussians(), schedule, 1.0)
    assert "distortion" in with_map and "distortion" not in without_map


def test_regularizers_read_raw_gaussian_params():
    schedule = {"opacity_reg": {"weight": 1.0}, "scale_reg": {"weight": 1.0}}
    _, values = compute_losses(0, _render(), _target(), _gaussians(), schedule, 1.0)
    # Raw zeros: gsplat's opacity_reg = sigmoid(0).mean() = 0.5, scale_reg = exp(0).mean() = 1.0
    assert values["opacity_reg"] == pytest.approx(0.5)
    assert values["scale_reg"] == pytest.approx(1.0)


def test_total_is_weighted_sum():
    schedule = {"depth": {"weight": 0.3}, "normal_consistency": {"weight": 0.7}}
    total, values = compute_losses(0, _render(), _target(), _gaussians(), schedule, 1.0)
    expected = 0.8 * values["l1"] + 0.2 * values["ssim"] + 0.3 * values["depth"] + 0.7 * values["normal_consistency"]
    assert total.item() == pytest.approx(expected, rel=1e-5)


def test_loss_weight_constant_without_end():
    assert loss_weight(0, {"weight": 0.5}) == 0.5
    assert loss_weight(10**6, {"weight": 0.5, "start": 100}) == 0.5
    assert loss_weight(99, {"weight": 0.5, "start": 100}) == 0.0
    assert loss_weight(0, None) == 0.0


def test_loss_weight_decays_log_linearly_then_holds():
    spec = {"weight": 0.01, "start": 1000, "end": 3000, "end_weight": 0.0001}
    assert loss_weight(999, spec) == 0.0
    assert loss_weight(1000, spec) == pytest.approx(0.01)
    assert loss_weight(2000, spec) == pytest.approx(0.001)  # geometric midpoint
    assert loss_weight(3000, spec) == pytest.approx(0.0001)
    assert loss_weight(10**6, spec) == pytest.approx(0.0001)


def test_compute_losses_uses_decayed_weight():
    schedule = {"depth": {"weight": 0.4, "end": 100, "end_weight": 0.1}}
    total, values = compute_losses(50, _render(), _target(), _gaussians(), schedule, 1.0)
    expected = 0.8 * values["l1"] + 0.2 * values["ssim"] + 0.2 * values["depth"]
    assert total.item() == pytest.approx(expected, rel=1e-5)


def test_depth_ratio_zero_matches_the_expected_only_loss():
    render = _render()  # no depth_normal_median — the 3dgs / explicit-zero config
    target, gaussians = _target(), _gaussians()
    fn = OPTIONAL_LOSSES["normal_consistency"]

    plain = fn(render, target, gaussians, 1.0, {"weight": 1.0})
    ratio_zero = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 0.0})
    assert torch.allclose(plain, ratio_zero)


def test_depth_ratio_one_uses_only_the_median_normal():
    render = _render()
    median = torch.nn.functional.normalize(
        torch.randn(1, 16, 16, 3, generator=torch.Generator().manual_seed(2)), dim=-1
    )
    render["depth_normal_median"] = median
    target, gaussians = _target(), _gaussians()
    fn = OPTIONAL_LOSSES["normal_consistency"]

    blended = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 1.0})

    # Same computation with the median normal in the expected slot
    swapped = dict(render)
    swapped["depth_normal"] = median
    reference = fn(swapped, target, gaussians, 1.0, {"weight": 1.0})
    assert torch.allclose(blended, reference)


def test_depth_ratio_is_a_convex_blend():
    render = _render()
    render["depth_normal_median"] = torch.nn.functional.normalize(
        torch.randn(1, 16, 16, 3, generator=torch.Generator().manual_seed(2)), dim=-1
    )
    target, gaussians = _target(), _gaussians()
    fn = OPTIONAL_LOSSES["normal_consistency"]

    at_zero = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 0.0})
    at_one = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 1.0})
    at_six = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 0.6})
    assert torch.allclose(at_six, 0.4 * at_zero + 0.6 * at_one, atol=1e-6)


def test_depth_ratio_without_a_median_normal_raises():
    render = _render()  # no depth_normal_median
    target, gaussians = _target(), _gaussians()
    with pytest.raises(ValueError, match="depth_normal_median"):
        OPTIONAL_LOSSES["normal_consistency"](render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 0.6})


def test_compute_losses_passes_the_spec_through():
    render = _render()
    render["depth_normal_median"] = torch.nn.functional.normalize(
        torch.randn(1, 16, 16, 3, generator=torch.Generator().manual_seed(2)), dim=-1
    )
    fn = OPTIONAL_LOSSES["normal_consistency"]
    median_only = fn(render, _target(), _gaussians(), 1.0, {"weight": 1.0, "depth_ratio": 1.0}).item()

    schedule = {"normal_consistency": {"weight": 1.0, "depth_ratio": 1.0}}
    _, values = compute_losses(0, render, _target(), _gaussians(), schedule, 1.0)
    assert values["normal_consistency"] == pytest.approx(median_only)


def test_scale_reg_prefers_decoded_log_scales():
    """
    Under scaffold there is no scales parameter: the regularizer reads the decoded scales.
    """
    render = {"log_scales": torch.full((5, 3), -2.0)}
    gaussians = torch.nn.ParameterDict({"scales": torch.nn.Parameter(torch.zeros(5, 3))})
    decoded_value = scale_reg_loss(render, {}, gaussians, 1.0, {"weight": 1.0})
    parameter_value = scale_reg_loss({}, {}, gaussians, 1.0, {"weight": 1.0})
    assert decoded_value.item() == pytest.approx(0.1353352832366127**3, rel=1e-5)
    assert parameter_value.item() == pytest.approx(1.0, rel=1e-5)


def test_scale_reg_on_decoded_scales_is_the_volume_not_the_mean():
    """
    Scaffold penalizes the decoded volume, so the 2dgs zeroed third channel leaves the area.
    """
    flat = {"log_scales": torch.cat([torch.full((4, 2), -2.0), torch.zeros(4, 1)], dim=-1)}
    gaussians = torch.nn.ParameterDict({"scales": torch.nn.Parameter(torch.zeros(4, 3))})
    assert scale_reg_loss(flat, {}, gaussians, 1.0, {"weight": 1.0}).item() == pytest.approx(
        0.1353352832366127**2, rel=1e-5
    )


def test_opacity_reg_prefers_decoded_opacities():
    """
    Scaffold's opacities are decoded and already activated, so they are averaged as-is.
    """
    render = {"opacities": torch.full((4,), 0.25)}
    gaussians = torch.nn.ParameterDict({"opacities": torch.nn.Parameter(torch.zeros(4))})
    assert opacity_reg_loss(render, {}, gaussians, 1.0, {"weight": 1.0}).item() == pytest.approx(0.25)
    assert opacity_reg_loss({}, {}, gaussians, 1.0, {"weight": 1.0}).item() == pytest.approx(0.5)


def test_appearance_reg_reads_render_params_or_skips():
    """
    The regularizer is registered and reads render["appearance"], skipping when it is absent.
    """
    assert OPTIONAL_LOSSES["appearance_reg"] is appearance_reg_loss
    assert appearance_reg_loss({}, {}, None, 1.0, {"weight": 1.0}) is None
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert appearance_reg_loss({"appearance": params}, {}, None, 1.0, {"weight": 1.0}).item() == pytest.approx(1.0 / 6)


def test_default_losses_gives_3dgs_the_mcmc_regularizers():
    losses = default_losses("3dgs")

    assert losses["opacity_reg"] == {"weight": 0.01}
    assert losses["scale_reg"] == {"weight": 0.01}
    assert "distortion" not in losses


def test_default_losses_gives_2dgs_distortion():
    losses = default_losses("2dgs")

    assert losses["distortion"] == {"weight": 0.01, "start": 3000}
    assert "opacity_reg" not in losses


def test_default_losses_shares_depth_normal_and_appearance_across_primitives():
    for primitive in ("3dgs", "2dgs"):
        losses = default_losses(primitive)
        assert losses["depth"] == {"weight": 0.01}
        assert losses["normal_consistency"] == {"weight": 0.05, "start": 7000}
        assert losses["appearance_reg"] == {"weight": 1e-3}


def test_validate_schedule_accepts_the_defaults():
    for primitive in ("3dgs", "2dgs"):
        validate_schedule(default_losses(primitive), primitive)


def test_validate_schedule_rejects_an_unknown_loss():
    with pytest.raises(ValueError, match="unknown loss 'wobble'"):
        validate_schedule({"wobble": {"weight": 1.0}}, "3dgs")


def test_validate_schedule_requires_a_weight():
    with pytest.raises(ValueError, match=r"splats.losses.depth: expected"):
        validate_schedule({"depth": {"start": 100}}, "3dgs")


def test_validate_schedule_names_the_keys_legal_for_that_loss():
    with pytest.raises(ValueError, match="depth_ratio"):
        validate_schedule({"normal_consistency": {"weight": 1.0, "typo": 1}}, "2dgs")


def test_validate_schedule_accepts_a_well_formed_decay():
    # Both other decay tests raise, and the defaults carry no `end`
    # - without this, a guard rejecting every decay entry leaves the rest of the file green
    validate_schedule({"depth": {"weight": 0.01, "start": 1000, "end": 3000, "end_weight": 0.0001}}, "3dgs")


def test_validate_schedule_rejects_a_decay_without_both_endpoints():
    with pytest.raises(ValueError, match="decay needs weight > 0"):
        validate_schedule({"depth": {"weight": 1.0, "end": 100}}, "3dgs")


def test_validate_schedule_rejects_a_decay_that_ends_before_it_starts():
    with pytest.raises(ValueError, match="end > start"):
        validate_schedule({"depth": {"weight": 1.0, "end_weight": 0.1, "start": 500, "end": 100}}, "3dgs")


def test_validate_schedule_rejects_distortion_on_3dgs():
    with pytest.raises(ValueError, match="distortion is 2dgs-only"):
        validate_schedule({"distortion": {"weight": 0.01}}, "3dgs")


def test_validate_schedule_allows_zero_weight_distortion_on_3dgs():
    validate_schedule({"distortion": {"weight": 0.0}}, "3dgs")


def test_validate_schedule_rejects_a_boolean_depth_ratio():
    with pytest.raises(ValueError, match="depth_ratio must be a number"):
        validate_schedule({"normal_consistency": {"weight": 1.0, "depth_ratio": True}}, "2dgs")


def test_validate_schedule_rejects_an_out_of_range_depth_ratio():
    with pytest.raises(ValueError, match=r"depth_ratio must be in \[0, 1\]"):
        validate_schedule({"normal_consistency": {"weight": 1.0, "depth_ratio": 1.5}}, "2dgs")


def test_validate_schedule_rejects_depth_ratio_on_3dgs():
    with pytest.raises(ValueError, match="depth_ratio > 0 is 2dgs-only"):
        validate_schedule({"normal_consistency": {"weight": 1.0, "depth_ratio": 0.5}}, "3dgs")


def test_compute_losses_photometric_weights_are_keyword_arguments():
    # Unequal random images, so l1 and ssim are far apart
    # - swapping the two weights moves the total by ~0.5 * (ssim - l1)
    # - an implementation that ignored them could not reproduce that
    render, target, gaussians = _render(), _target(), _gaussians()

    total, values = compute_losses(0, render, target, gaussians, {}, 1.0, l1_weight=0.3, ssim_weight=0.7)
    default_total, _ = compute_losses(0, render, target, gaussians, {}, 1.0)

    # The passed weights compose the reported terms; the defaults are still 0.8 / 0.2
    # - nothing else pins those, and a later task deletes the duplicate literals elsewhere
    assert float(total) == pytest.approx(0.3 * values["l1"] + 0.7 * values["ssim"], rel=1e-5)
    assert float(default_total) == pytest.approx(0.8 * values["l1"] + 0.2 * values["ssim"], rel=1e-5)
    assert float(total) != pytest.approx(float(default_total))


def test_compute_losses_photometric_weights_are_keyword_only():
    # `scene_scale` is the sixth and last positional; a seventh becomes `l1_weight`
    # - the full count string is pinned: any arity slip in this call also says "positional"
    with pytest.raises(TypeError, match="takes 6 positional arguments but 7 were given"):
        compute_losses(0, _render(), _target(), _gaussians(), {}, 1.0, 0.5)


########################################
# The schedule readers the trainer calls once, before the loop
########################################


def test_rescale_depth_units_divides_the_distortion_weight():
    # The 2dgs distortion loss is linear in depth
    # - a unit-cube frame penalizes `scale` times harder for the same yaml weight
    # - dividing restores the world-frame penalty
    # - multiplying, or leaving it alone: same schedule shape, different training run
    rescaled = rescale_depth_units({"distortion": {"weight": 100.0}, "depth": {"weight": 0.5}}, 4.0)

    assert rescaled["distortion"]["weight"] == pytest.approx(25.0)
    assert rescaled["depth"] == {"weight": 0.5}


def test_rescale_depth_units_rescales_a_scheduled_end_weight_too():
    # A decayed distortion loss carries its endpoint in the same units
    # - forgetting it ramps from a unit-cube weight towards a world-frame one
    rescaled = rescale_depth_units({"distortion": {"weight": 100.0, "end": 5, "end_weight": 10.0}}, 4.0)

    assert rescaled["distortion"]["weight"] == pytest.approx(25.0)
    assert rescaled["distortion"]["end_weight"] == pytest.approx(2.5)
    assert rescaled["distortion"]["end"] == 5


def test_rescale_depth_units_passes_a_schedule_without_distortion_through():
    losses = {"depth": {"weight": 0.5}}

    assert rescale_depth_units(losses, 4.0) == losses


def test_rescale_depth_units_leaves_the_callers_schedule_alone():
    # The caller's schedule is the yaml dict `asdict(cfg)` writes into ckpt.pt and summary.config
    # - aliasing instead of copying divides the reported weight in place
    # - a frozen report then carries the rescaled number
    losses = {"distortion": {"weight": 100.0, "end_weight": 10.0}, "depth": {"weight": 0.5}}

    rescaled = rescale_depth_units(losses, 4.0)

    assert losses == {"distortion": {"weight": 100.0, "end_weight": 10.0}, "depth": {"weight": 0.5}}
    assert rescaled is not losses
    assert rescaled["distortion"] is not losses["distortion"]


def test_neighbor_selection_reads_the_pgsr_multiview_keys():
    spec = {"weight": 0.1, "num_multi_view": 3, "max_points": 500}

    assert neighbor_selection(spec) == {"num_views": 3, "max_points": 500}


def test_neighbor_selection_defaults_are_the_documented_ones():
    # Read once before the loop, not per step
    # - so they default here, not at the trainer's call site
    # - nothing downstream would report a changed default
    assert neighbor_selection({"weight": 0.1}) == {"num_views": 5, "max_points": 20000}


def test_neighbor_selection_casts_to_int():
    # select_near_views indexes and slices with both, and yaml hands back whatever was typed
    selected = neighbor_selection({"weight": 0.1, "num_multi_view": 3.0, "max_points": 500.0})

    assert isinstance(selected["num_views"], int)
    assert isinstance(selected["max_points"], int)
