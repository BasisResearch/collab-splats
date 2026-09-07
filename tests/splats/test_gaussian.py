"""
Vanilla Gaussian primitives: initialization, strategy selection, denormalization, checkpointing.
"""

import math

import numpy as np
import pytest
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from collab_splats.splats.gaussian import SH_C0, Gaussians, make_strategy
from collab_splats.splats.trainer import SplatsConfig


def _seed_cloud(n_points=32):
    """
    A small deterministic seed cloud and its colors.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, (n_points, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (n_points, 3)).astype(np.uint8)
    return points, colors


def _model(*, scene_scale=2.0, n_views=4, **overrides):
    """
    A CPU Gaussians model over the seed cloud; `overrides` go into the config.
    """
    points, colors = _seed_cloud()
    cfg = SplatsConfig.from_dict({"max_steps": 100, **overrides})
    return Gaussians(cfg, points, colors, scene_scale=scene_scale, n_views=n_views, device="cpu")


def _model_with(*, scene_scale=2.0, n_views=4, **kwargs):
    """
    A CPU Gaussians model over the seed cloud, overriding `Gaussians.__init__` keyword args.
    """
    points, colors = _seed_cloud()
    cfg = SplatsConfig.from_dict({"max_steps": 100})
    return Gaussians(cfg, points, colors, scene_scale=scene_scale, n_views=n_views, device="cpu", **kwargs)


def test_sh_c0_is_the_degree_zero_spherical_harmonic():
    assert SH_C0 == pytest.approx(0.5 / math.sqrt(math.pi))
    # The value every existing checkpoint's sh0 was written against
    assert SH_C0 == pytest.approx(0.28209479177387814)


def test_gaussians_start_one_per_seed_point():
    model = _model()

    assert model.n_primitives == 32
    assert model.params["means"].shape == (32, 3)
    assert model.params["quats"].shape == (32, 4)
    assert model.params["opacities"].shape == (32,)


def test_gaussians_store_color_in_the_degree_zero_sh_band():
    # _seed_cloud is deterministic, so these are the same colors the helper's model was built from
    _, colors = _seed_cloud()
    model = _model()

    expected = (torch.from_numpy(colors).float() / 255.0 - 0.5) / SH_C0
    assert torch.allclose(model.params["sh0"][:, 0, :], expected, atol=1e-5)
    # Higher bands start at zero
    assert torch.all(model.params["shN"] == 0)


def test_gaussians_sh_band_count_follows_sh_degree():
    model = _model(sh_degree=2)

    # (degree + 1)^2 coefficients, one held out as the DC band
    assert model.params["sh0"].shape[1] == 1
    assert model.params["shN"].shape[1] == (2 + 1) ** 2 - 1


def test_gaussians_opacities_start_at_the_configured_logit():
    model = _model(init_opacity=0.1)

    assert torch.allclose(torch.sigmoid(model.params["opacities"]), torch.full((32,), 0.1), atol=1e-6)


def test_gaussians_build_one_optimizer_per_tensor():
    model = _model()

    # Six parameters, six Adams, so the strategy can grow and prune optimizer state per tensor
    assert len(model.optimizers) == 6
    assert {id(o) for o in model.optimizers} == {id(o) for o in model.param_optimizers.values()}


def test_gaussians_scale_the_means_learning_rate_by_the_scene():
    cfg = SplatsConfig.from_dict({"max_steps": 100})
    model = _model(scene_scale=3.0)

    assert model.param_optimizers["means"].param_groups[0]["lr"] == pytest.approx(cfg.means_lr * 3.0)


def test_gaussians_expose_one_scheduler_on_the_means_optimizer():
    model = _model()

    assert len(model.schedulers) == 1
    assert model.schedulers[0].optimizer is model.param_optimizers["means"]


def test_gaussians_activate_raw_parameters_for_the_rasterizer():
    model = _model()

    # Spread the seed opacities first: constant ones pin the value 0.1, not sigmoid as a function
    with torch.no_grad():
        model.params["opacities"].data = torch.linspace(-3.0, 3.0, len(model.params["opacities"]))
    activated = model.activate()

    assert torch.allclose(activated["scales"], torch.exp(model.params["scales"]))
    assert torch.allclose(activated["opacities"], torch.sigmoid(model.params["opacities"]))
    assert not torch.allclose(activated["opacities"], model.params["opacities"])

    # means and quats are already in the rasterizer's form and pass through untouched
    assert torch.equal(activated["means"], model.params["means"])
    assert torch.equal(activated["quats"], model.params["quats"])

    # SH bands are concatenated back into one (N, K, 3) tensor
    assert activated["colors"].shape == (32, (model.sh_degree + 1) ** 2, 3)


def test_gaussians_denormalize_inverts_the_sim3():
    model = _model()
    center = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    scale = 0.25
    means_before = model.params["means"].detach().clone()
    scales_before = model.params["scales"].detach().clone()

    model.denormalize(center, scale)

    assert torch.allclose(model.params["means"], means_before / scale + torch.from_numpy(center), atol=1e-5)
    assert torch.allclose(model.params["scales"], scales_before - math.log(scale), atol=1e-5)


def test_gaussians_export_returns_the_raw_parameters():
    model = _model()
    exported = model.export_gaussians(torch.eye(4)[None], torch.eye(3)[None], 64, 64)

    assert set(exported) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    assert exported["means"] is model.params["means"]


def test_gaussians_checkpoint_round_trips():
    model = _model()
    ckpt = model.checkpoint()
    ckpt["config"] = {"primitive": "3dgs", "sh_degree": 3, "sh_degree_interval": 1000}

    restored = Gaussians.from_checkpoint(ckpt, "cpu")

    assert restored.n_primitives == model.n_primitives
    assert torch.allclose(restored.params["means"], model.params["means"])
    # A checkpoint-restored model renders; it does not train
    assert restored.optimizers == []
    assert restored.schedulers == []
    assert restored.strategy is None

    # from_checkpoint builds through cls.__new__, so __init__'s attributes are set by hand there
    # - comparing the attribute sets is what catches one added to only one path
    assert set(vars(restored)) == set(vars(model))


def test_make_strategy_picks_mcmc_for_3dgs():
    cfg = SplatsConfig.from_dict({"primitive": "3dgs", "cap_max": 500})
    strategy = make_strategy(cfg, n_views=4)

    assert isinstance(strategy, MCMCStrategy)
    assert strategy.cap_max == 500


def test_make_strategy_picks_default_with_splatfacto_arguments_for_2dgs():
    cfg = SplatsConfig.from_dict({"primitive": "2dgs", "grow_grad2d": 2e-4})
    strategy = make_strategy(cfg, n_views=4)

    assert isinstance(strategy, DefaultStrategy)
    assert strategy.absgrad is False
    assert strategy.key_for_gradient == "gradient_2dgs"
    assert strategy.grow_grad2d == pytest.approx(2e-4)
    assert strategy.pause_refine_after_reset == 4 + 100


def test_make_strategy_caps_the_refine_pause_so_densification_still_runs(caplog):
    cfg = SplatsConfig.from_dict({"primitive": "2dgs"})
    # n_views + 100 would exceed reset_every, which silently disables refinement forever
    strategy = make_strategy(cfg, n_views=100_000)

    defaults = DefaultStrategy()
    assert strategy.pause_refine_after_reset == defaults.reset_every - defaults.refine_every
    assert "would never refine" in caplog.text


def test_make_strategy_uses_splatfactos_tuning_literals_by_default():
    strategy = make_strategy(SplatsConfig.from_dict({"primitive": "2dgs"}), n_views=4)

    # splatfacto's non-default args (nerfstudio @ 50e0e3c)
    # - gsplat's own defaults are 0.005 / 0.1 / 0, so these three literals are the parity contract
    # - drift shows up as an unexplained dPSNR, never as a failing test
    assert strategy.prune_opa == pytest.approx(0.1)
    assert strategy.prune_scale3d == pytest.approx(0.5)
    assert strategy.refine_scale2d_stop_iter == 4000


def test_make_strategy_tuning_literals_are_keyword_only():
    cfg = SplatsConfig.from_dict({"primitive": "2dgs"})

    # The bare `*` is the point: a third positional must not be accepted
    # - match the full count string: every arity slip in this call also says "positional"
    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
        make_strategy(cfg, 4, 0.2)

    strategy = make_strategy(cfg, n_views=4, prune_opa=0.2, prune_scale3d=0.7, refine_scale2d_stop_iter=1234)
    assert strategy.prune_opa == pytest.approx(0.2)
    assert strategy.prune_scale3d == pytest.approx(0.7)
    assert strategy.refine_scale2d_stop_iter == 1234


########################################
# Ground Rule 9 keyword-only defaults
########################################


def test_gaussians_init_tuning_literals_are_keyword_only():
    points, colors = _seed_cloud()
    cfg = SplatsConfig.from_dict({"max_steps": 100})

    # knn / adam_eps / lr_decay sit behind a bare `*`: a seventh positional must be rejected
    # - `device` is the sixth and last, so the count string pins which slot the eighth hit
    with pytest.raises(TypeError, match="takes 7 positional arguments but 8 were given"):
        Gaussians(cfg, points, colors, 2.0, 4, "cpu", 8)


def test_gaussians_use_upstreams_adam_epsilon_by_default():
    model = _model()

    # 1e-15, not torch's 1e-8 -- upstream simple_trainer's value
    assert all(optimizer.param_groups[0]["eps"] == 1e-15 for optimizer in model.optimizers)
    assert _model_with(adam_eps=1e-8).optimizers[0].param_groups[0]["eps"] == pytest.approx(1e-8)


def test_gaussians_decay_the_means_lr_by_one_hundredth_over_the_run():
    model = _model()

    # lr_decay defaults to 0.01: the means lr ends the run 100x below where it started
    assert model.means_scheduler.gamma == pytest.approx(0.01 ** (1.0 / 100))
    assert _model_with(lr_decay=0.5).means_scheduler.gamma == pytest.approx(0.5 ** (1.0 / 100))


def test_gaussians_initial_scale_uses_the_three_nearest_neighbors_by_default():
    default_scales = _model().params["scales"]

    # knn defaults to 4 -- self plus the 3 nearest; a wider neighborhood gives coarser scales
    assert torch.allclose(default_scales, _model_with(knn=4).params["scales"])
    assert not torch.allclose(default_scales, _model_with(knn=8).params["scales"])


########################################
# Render and strategy dispatch
########################################


def test_render_warms_up_the_sh_degree(monkeypatch):
    model = _model(sh_degree=3, sh_degree_interval=1000)
    captured = {}

    def _capture(primitive, decoded, cam_to_world, intrinsics, width, height, sh_degree, absgrad, **kwargs):
        captured["sh_degree"] = sh_degree
        return {}, {}

    monkeypatch.setattr("collab_splats.splats.gaussian.render_gaussians", _capture)
    view = (torch.eye(4)[None], torch.eye(3)[None], 64, 64, torch.zeros(1, dtype=torch.long))

    # sh_degree is computed here from the step; render_gaussians only takes it as an argument
    model.render(*view, step=0)
    assert captured["sh_degree"] == 0
    model.render(*view, step=2500)
    assert captured["sh_degree"] == 2

    # Clamped at cfg.sh_degree, and step=None (export) renders every band
    model.render(*view, step=99_000)
    assert captured["sh_degree"] == 3
    model.render(*view, step=None)
    assert captured["sh_degree"] == 3


def test_render_never_asks_for_absolute_gradients(monkeypatch):
    captured = {}

    def _capture(primitive, decoded, cam_to_world, intrinsics, width, height, sh_degree, absgrad, **kwargs):
        captured["absgrad"] = absgrad
        return {}, {}

    monkeypatch.setattr("collab_splats.splats.gaussian.render_gaussians", _capture)
    view = (torch.eye(4)[None], torch.eye(3)[None], 64, 64, torch.zeros(1, dtype=torch.long))

    # 2dgs is the only primitive with an absgrad the render could disagree with
    # - `render` sends a literal, so this is the only thing holding it equal to make_strategy's
    default = _model(primitive="2dgs")
    default.render(*view, step=0)
    assert default.strategy.absgrad is False
    assert captured["absgrad"] is default.strategy.absgrad

    # 3dgs is not the same assertion twice: MCMCStrategy has no absgrad, so the literal is all
    captured.clear()
    mcmc = _model(primitive="3dgs", cap_max=500)
    mcmc.render(*view, step=0)
    assert not hasattr(mcmc.strategy, "absgrad")
    assert captured["absgrad"] is False

    # Third construction path: a restored model has no strategy at all, and still renders
    captured.clear()
    ckpt = default.checkpoint()
    ckpt["config"] = {"primitive": "2dgs", "sh_degree": 3, "sh_degree_interval": 1000}
    restored = Gaussians.from_checkpoint(ckpt, "cpu")
    restored.render(*view, step=0)
    assert restored.strategy is None
    assert captured["absgrad"] is False


def test_render_hands_the_rasterizer_activated_params_and_the_configured_primitive(monkeypatch):
    captured = {}

    def _capture(primitive, decoded, cam_to_world, intrinsics, width, height, sh_degree, absgrad, **kwargs):
        captured["primitive"] = primitive
        captured["decoded"] = decoded
        return {}, {}

    monkeypatch.setattr("collab_splats.splats.gaussian.render_gaussians", _capture)
    view = (torch.eye(4)[None], torch.eye(3)[None], 64, 64, torch.zeros(1, dtype=torch.long))

    # The rasterizer takes activated tensors, never the raw log-scales and logit-opacities in params
    model = _model()
    model.render(*view, step=0)
    assert captured["primitive"] == "3dgs"
    assert torch.allclose(captured["decoded"]["scales"], torch.exp(model.params["scales"]))
    assert torch.allclose(captured["decoded"]["opacities"], torch.sigmoid(model.params["opacities"]))

    # The configured primitive is forwarded: a constant would rasterize every 2dgs run as 3dgs
    captured.clear()
    two_dgs = _model(primitive="2dgs")
    two_dgs.render(*view, step=0)
    assert captured["primitive"] == "2dgs"


def test_render_forwards_every_positional_in_its_own_slot(monkeypatch):
    # render_gaussians takes eight positionals, every slot pinned here rather than by shape luck
    # - transposed width/height, swapped cam_to_world/intrinsics or a dropped `decoded` tensor all
    #   render a different image with no exception raised
    model = _model()
    captured = {}

    def _capture(primitive, decoded, cam_to_world, intrinsics, width, height, sh_degree, absgrad, **kwargs):
        captured.update(
            primitive=primitive,
            decoded=decoded,
            cam_to_world=cam_to_world,
            intrinsics=intrinsics,
            width=width,
            height=height,
        )
        return {}, {}

    monkeypatch.setattr("collab_splats.splats.gaussian.render_gaussians", _capture)

    # Distinguishable pose and camera matrix, and width != height
    cam_to_world = torch.eye(4)[None] * 2.0
    intrinsics = torch.eye(3)[None] * 3.0
    model.render(cam_to_world, intrinsics, 64, 32, torch.zeros(1, dtype=torch.long), step=0)

    assert captured["width"] == 64
    assert captured["height"] == 32
    assert torch.equal(captured["cam_to_world"], cam_to_world)
    assert torch.equal(captured["intrinsics"], intrinsics)

    # Every activated tensor reaches the rasterizer, not only the two the activation transforms
    expected = model.activate()
    assert set(captured["decoded"]) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(captured["decoded"][name], tensor), name


def test_render_forwards_the_normal_and_plane_flags(monkeypatch):
    model = _model()
    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return {}, {}

    monkeypatch.setattr("collab_splats.splats.gaussian.render_gaussians", _capture)
    view = (torch.eye(4)[None], torch.eye(3)[None], 64, 64, torch.zeros(1, dtype=torch.long))

    # Defaults: normals on (the mesh path needs them), PGSR's planar signals off
    model.render(*view, step=0)
    assert captured == {"render_normals": True, "render_plane": False}

    # Both are pass-throughs, so a hardcoded default here would silently ignore the caller
    model.render(*view, step=0, render_normals=False, render_plane=True)
    assert captured == {"render_normals": False, "render_plane": True}


def test_gaussians_seed_the_default_strategy_state_with_the_scene_scale():
    # DefaultStrategy prunes on prune_scale3d * scene_scale: a wrong scale prunes a different set
    default = _model(primitive="2dgs")
    assert default.strategy_state["scene_scale"] == pytest.approx(2.0)

    # MCMC's initialize_state takes no scene_scale at all, which is why __init__ has to branch
    mcmc = _model(primitive="3dgs", cap_max=500)
    assert "scene_scale" not in mcmc.strategy_state


def test_pre_backward_dispatches_to_the_default_strategy_and_no_ops_under_mcmc(monkeypatch):
    default = _model(primitive="2dgs")

    calls = []
    monkeypatch.setattr(default.strategy, "step_pre_backward", lambda *args, **kwargs: calls.append(args[3]))
    default.pre_backward(7, {})

    # gsplat does the retain_grad inside step_pre_backward, so dispatch is all this method owns
    assert calls == [7]

    # MCMC inherits a no-op hook from Strategy, so the same call must simply run and record nothing
    mcmc = _model(primitive="3dgs", cap_max=500)
    mcmc.pre_backward(7, {})
    assert calls == [7]


def test_post_backward_dispatches_mcmc_with_the_means_lr_and_default_without(monkeypatch):
    calls = []

    # MCMC relocates against the post-decay means lr, as upstream simple_trainer does
    mcmc = _model(primitive="3dgs", cap_max=500)
    monkeypatch.setattr(mcmc.strategy, "step_post_backward", lambda *args, **kwargs: calls.append((args[3], kwargs)))
    mcmc.post_backward(7, {})

    # args[3] is the step gsplat gates refine_every / refine_start_iter / reset_every on
    assert calls == [(7, {"lr": pytest.approx(mcmc.means_scheduler.get_last_lr()[0])})]

    # Default densifies unpacked and takes no lr
    default = _model(primitive="2dgs")
    monkeypatch.setattr(default.strategy, "step_post_backward", lambda *args, **kwargs: calls.append((args[3], kwargs)))
    default.post_backward(7, {})

    assert calls[-1] == (7, {"packed": False})
