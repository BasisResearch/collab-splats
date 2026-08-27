"""
Scaffold-GS anchors: config, MLP heads, decode, and anchor densification.
"""

import numpy as np
import pytest
import torch

from collab_splats.splats.scaffold import ScaffoldConfig
from collab_splats.splats.trainer import SplatsConfig

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def test_representation_defaults_to_vanilla():
    cfg = SplatsConfig()
    assert cfg.representation == "vanilla"
    assert cfg.scaffold is None
    assert cfg.scaffold_config is None


def test_scaffold_block_parses_into_scaffold_config():
    cfg = SplatsConfig.from_dict({"representation": "scaffold", "scaffold": {"n_offsets": 5, "feat_dim": 16}})
    assert isinstance(cfg.scaffold_config, ScaffoldConfig)
    assert cfg.scaffold_config.n_offsets == 5
    assert cfg.scaffold_config.feat_dim == 16


def test_scaffold_representation_without_a_block_gets_defaults():
    cfg = SplatsConfig.from_dict({"representation": "scaffold"})
    assert cfg.scaffold_config.n_offsets == ScaffoldConfig().n_offsets


def test_unknown_representation_is_rejected():
    with pytest.raises(ValueError, match="representation"):
        SplatsConfig.from_dict({"representation": "octree"})


def test_unknown_scaffold_key_is_rejected():
    with pytest.raises(ValueError, match="n_offset"):
        SplatsConfig.from_dict({"representation": "scaffold", "scaffold": {"n_offset": 5}})


def test_scaffold_block_without_scaffold_representation_is_rejected():
    with pytest.raises(ValueError, match="representation: scaffold"):
        SplatsConfig.from_dict({"scaffold": {"n_offsets": 5}})


def test_sh_degree_is_rejected_under_scaffold():
    """A deliberate SH override is a misunderstanding of the representation, so it raises."""
    with pytest.raises(ValueError, match="sh_degree"):
        SplatsConfig.from_dict({"representation": "scaffold", "sh_degree": 0})
    with pytest.raises(ValueError, match="sh_degree"):
        SplatsConfig.from_dict({"representation": "scaffold", "sh_degree_interval": 500})


def test_scaffold_accepts_the_inherited_sh_defaults():
    """base.yaml deep-merges sh_degree/sh_degree_interval into every block at their defaults."""
    cfg = SplatsConfig.from_dict({"representation": "scaffold", "sh_degree": 3, "sh_degree_interval": 1000})
    assert cfg.representation == "scaffold"


########################################
# MLP heads
########################################


def test_mlp_heads_emit_per_offset_outputs():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    cfg = ScaffoldConfig(n_offsets=4, feat_dim=8, appearance_dim=0)
    mlps = ScaffoldMLPs(cfg, n_views=5)
    features = torch.zeros(6, cfg.feat_dim + 3)  # feat + unit view dir (3)
    opacity, cov, colour = mlps(features, camera_id=None)
    assert opacity.shape == (6, 4)
    assert cov.shape == (6, 4 * 7)
    assert colour.shape == (6, 4 * 3)
    assert opacity.min() >= -1.0 and opacity.max() <= 1.0  # tanh
    assert colour.min() >= 0.0 and colour.max() <= 1.0  # sigmoid


def test_appearance_embedding_changes_colour_only_when_enabled():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    features = torch.zeros(3, 8 + 3)
    camera_id = torch.zeros(3, dtype=torch.long)

    off = ScaffoldMLPs(ScaffoldConfig(n_offsets=2, feat_dim=8, appearance_dim=0), n_views=5)
    assert off.embedding_appearance is None
    off(features, camera_id)  # camera_id is accepted and ignored

    on = ScaffoldMLPs(ScaffoldConfig(n_offsets=2, feat_dim=8, appearance_dim=6), n_views=5)
    assert on.embedding_appearance is not None
    assert on.embedding_appearance.weight.shape == (5, 6)
    with pytest.raises(ValueError, match="camera_id"):
        on(features, camera_id=None)


def test_appearance_embedding_needs_view_count():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    with pytest.raises(ValueError, match="n_views"):
        ScaffoldMLPs(ScaffoldConfig(appearance_dim=6), n_views=0)


########################################
# Anchor field
########################################


def _seed_points(n=500, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    colors = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    return points, colors


def test_anchor_init_voxelizes_seed_points():
    from collab_splats.splats.scaffold import AnchorField

    cfg = ScaffoldConfig(n_offsets=4, feat_dim=8)
    points, colors = _seed_points()
    field = AnchorField(cfg, points, colors, scene_scale=1.0, n_views=3, device="cpu")
    n_anchors = len(field.params["anchors"])
    assert 0 < n_anchors <= len(points)
    assert field.params["offsets"].shape == (n_anchors, 4, 3)
    assert field.params["anchor_feat"].shape == (n_anchors, 8)
    assert field.params["scaling"].shape == (n_anchors, 6)
    assert field.params["rotation"].shape == (n_anchors, 4)
    assert field.voxel_size > 0

    # Opacity is decoded per view by mlp_opacity, so there is no anchor opacity parameter to train
    assert "opacities" not in field.params


def test_anchor_count_is_invariant_to_scene_scale():
    """voxel_size is derived from kNN spacing, so a 10x bigger copy of a scene gets the same anchors."""
    from collab_splats.splats.scaffold import AnchorField

    cfg = ScaffoldConfig(n_offsets=2, feat_dim=8)
    points, colors = _seed_points()
    small = AnchorField(cfg, points, colors, scene_scale=1.0, n_views=1, device="cpu")
    large = AnchorField(cfg, points * 10.0, colors, scene_scale=10.0, n_views=1, device="cpu")
    assert len(large.params["anchors"]) == len(small.params["anchors"])
    assert large.voxel_size == pytest.approx(small.voxel_size * 10.0, rel=1e-5)


def test_explicit_voxel_size_overrides_the_derived_one():
    from collab_splats.splats.scaffold import AnchorField

    points, colors = _seed_points()
    field = AnchorField(
        ScaffoldConfig(n_offsets=2, feat_dim=8, voxel_size=0.5),
        points,
        colors,
        scene_scale=1.0,
        n_views=1,
        device="cpu",
    )
    assert field.voxel_size == 0.5


def test_optimizers_cover_every_anchor_parameter():
    from collab_splats.splats.scaffold import AnchorField

    points, colors = _seed_points()
    field = AnchorField(ScaffoldConfig(n_offsets=2, feat_dim=8), points, colors, 1.0, n_views=1, device="cpu")
    assert set(field.optimizers) == set(field.params)
    for optimizer in field.optimizers.values():
        assert len(optimizer.param_groups) == 1


########################################
# Decode
########################################


def _field(n_offsets=4, appearance_dim=0, n_views=3, device="cpu", lr_max_steps=30000):
    from collab_splats.splats.scaffold import AnchorField

    cfg = ScaffoldConfig(n_offsets=n_offsets, feat_dim=8, appearance_dim=appearance_dim, lr_max_steps=lr_max_steps)
    points, colors = _seed_points(n=200)
    return AnchorField(cfg, points, colors, scene_scale=1.0, n_views=n_views, device=device)


def _cam(device="cpu"):
    cam_to_world = torch.eye(4, device=device)[None]
    cam_to_world[0, 2, 3] = -4.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)[None]
    return cam_to_world, intrinsics


def test_decode_returns_rasterizer_inputs_and_index():
    field = _field()
    cam_to_world, intrinsics = _cam()
    decoded, index = field.decode("3dgs", cam_to_world, intrinsics, width=64, height=64, camera_id=None)
    n = len(decoded["means"])
    assert n > 0
    assert decoded["means"].shape == (n, 3)
    assert decoded["quats"].shape == (n, 4)
    assert decoded["scales"].shape == (n, 3)
    assert decoded["opacities"].shape == (n,)
    assert decoded["colors"].shape == (n, 3)  # post-activation RGB, so sh_degree=None at rasterize
    assert decoded["log_scales"].shape == (n, 3)  # scale_reg reads this instead of a parameter
    assert index.shape == (n,)
    assert index.dtype == torch.int64
    assert int(index.max()) < len(field.params["anchors"]) * field.cfg.n_offsets


def test_decode_index_points_at_the_generating_anchor():
    field = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()
    decoded, index = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    anchor_ids = index // field.cfg.n_offsets

    # Each decoded mean is its anchor plus a scaled offset, so it must sit within the offset extent
    anchors = field.params["anchors"][anchor_ids]
    offset_extent = torch.exp(field.params["scaling"][anchor_ids][:, :3])
    displacement = (decoded["means"] - anchors).abs()
    assert torch.all(displacement <= offset_extent * 1.001 + 1e-6)


def test_decode_drops_offsets_with_non_positive_opacity():
    field = _field(n_offsets=4)
    cam_to_world, intrinsics = _cam()

    # Close offsets 0 and 2, open 1 and 3: the tanh head's last linear bias dominates the feature input
    with torch.no_grad():
        field.mlps.mlp_opacity[-2].weight.zero_()
        field.mlps.mlp_opacity[-2].bias.copy_(torch.tensor([-5.0, 5.0, -5.0, 5.0]))
    decoded, index = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    n_visible = int(field.visible_anchors(cam_to_world, intrinsics, 64, 64).sum())
    assert len(decoded["means"]) == 2 * n_visible
    assert set((index % 4).tolist()) == {1, 3}


def test_decode_is_differentiable_into_the_mlps():
    field = _field()
    cam_to_world, intrinsics = _cam()
    decoded, _ = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    decoded["colors"].sum().backward()
    assert field.mlps.mlp_colour[0].weight.grad is not None
    assert field.params["anchor_feat"].grad is not None


def test_decode_2dgs_zeroes_the_third_scale():
    field = _field()
    cam_to_world, intrinsics = _cam()
    decoded, _ = field.decode("2dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    assert torch.all(decoded["scales"][:, 2] == 0.0)


def test_decode_frustum_filter_drops_anchors_behind_the_camera():
    field = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()
    visible = field.visible_anchors(cam_to_world, intrinsics, 64, 64)

    # Camera sits at z = -4 looking down +z, so a point far behind it can never be visible
    with torch.no_grad():
        field.params["anchors"][0] = torch.tensor([0.0, 0.0, -100.0])
    assert not bool(field.visible_anchors(cam_to_world, intrinsics, 64, 64)[0])
    assert bool(visible.any())


def test_decode_never_returns_zero_gaussians_when_every_offset_is_closed():
    """gsplat's projection kernel raises SIGFPE on an empty input, so decode keeps the best offset."""
    field = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()

    # Drive every neural opacity negative: the tanh head saturates at -1 for a large negative bias
    with torch.no_grad():
        field.mlps.mlp_opacity[-2].bias.fill_(-50.0)
        field.mlps.mlp_opacity[-2].weight.zero_()
    decoded, decode_index = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    assert len(decoded["means"]) == 1
    assert len(decode_index) == 1


def test_decode_falls_back_to_every_anchor_when_the_frustum_is_empty():
    """A camera looking away culls every anchor; decode still emits Gaussians the rasterizer can cull."""
    field = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()

    # Push every anchor far behind the camera, which sits at z = -4 looking down +z
    with torch.no_grad():
        field.params["anchors"].data[:] = torch.tensor([0.0, 0.0, -100.0])
    assert not bool(field.visible_anchors(cam_to_world, intrinsics, 64, 64).any())
    decoded, _ = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    assert len(decoded["means"]) > 0


########################################
# Rasterization
########################################


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_scaffold_decode_renders_through_gsplat(primitive):
    from collab_splats.splats.rendering import render_gaussians

    field = _field(device="cuda")
    cam_to_world, intrinsics = _cam(device="cuda")
    decoded, _ = field.decode(primitive, cam_to_world, intrinsics, 64, 64, camera_id=None)
    render, info = render_gaussians(primitive, decoded, cam_to_world, intrinsics, 64, 64, sh_degree=None, absgrad=False)
    assert render["rgb"].shape == (1, 64, 64, 3)
    assert render["depth"].shape == (1, 64, 64, 1)
    expected_gradient_key = "means2d" if primitive == "3dgs" else "gradient_2dgs"
    assert expected_gradient_key in info


@cuda
def test_render_gradient_reaches_the_anchor_features():
    from collab_splats.splats.rendering import render_gaussians

    field = _field(device="cuda")
    cam_to_world, intrinsics = _cam(device="cuda")
    decoded, _ = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    render, _ = render_gaussians("3dgs", decoded, cam_to_world, intrinsics, 64, 64, sh_degree=None, absgrad=False)
    render["rgb"].sum().backward()
    assert field.params["anchor_feat"].grad is not None
    assert torch.isfinite(field.params["anchor_feat"].grad).all()


########################################
# Densification
########################################


def test_gradient_key_follows_the_primitive():
    from collab_splats.splats.scaffold import AnchorStrategy

    assert AnchorStrategy(ScaffoldConfig(), primitive="3dgs").key_for_gradient == "means2d"
    assert AnchorStrategy(ScaffoldConfig(), primitive="2dgs").key_for_gradient == "gradient_2dgs"


def test_accumulation_renormalises_gradients_like_gsplat():
    """gsplat's DefaultStrategy scales means2d grads to [-1, 1] screen space before thresholding."""
    from collab_splats.splats.scaffold import AnchorStrategy

    strategy = AnchorStrategy(ScaffoldConfig(n_offsets=2, feat_dim=8), primitive="3dgs")
    state = strategy.initialize_state(n_anchors=4)

    means2d = torch.zeros(1, 3, 2)
    means2d.grad = torch.tensor([[[1e-3, 0.0], [0.0, 2e-3], [0.0, 0.0]]])
    info = {"means2d": means2d, "width": 800, "height": 600, "n_cameras": 1, "radii": torch.ones(3, 2)}
    decode_index = torch.tensor([0, 5, 7])

    strategy.accumulate(
        state, info, decode_index, opacities=torch.tensor([0.5, 0.5, 0.5]), visible_ids=torch.tensor([0, 2, 3])
    )
    assert state["grad_accum"][0] == pytest.approx(1e-3 * 400.0)
    assert state["grad_accum"][5] == pytest.approx(2e-3 * 300.0)
    assert state["denom"][0] == 1
    assert state["denom"][1] == 0

    # Opacity is per anchor (slot 5 and 7 belong to anchors 2 and 3), and every visible anchor
    # counts a visit whether or not any of its offsets rendered
    assert state["opacity_accum"][0] == pytest.approx(0.5)
    assert state["opacity_accum"][2] == pytest.approx(0.5)
    assert list(state["anchor_denom"]) == [1, 0, 1, 1]


def test_accumulation_is_additive_over_steps():
    from collab_splats.splats.scaffold import AnchorStrategy

    strategy = AnchorStrategy(ScaffoldConfig(n_offsets=2, feat_dim=8), primitive="3dgs")
    state = strategy.initialize_state(n_anchors=2)
    means2d = torch.zeros(1, 1, 2)
    means2d.grad = torch.tensor([[[1e-3, 0.0]]])
    info = {"means2d": means2d, "width": 800, "height": 600, "n_cameras": 1, "radii": torch.ones(1, 2)}
    for _ in range(3):
        strategy.accumulate(
            state, info, torch.tensor([2]), opacities=torch.tensor([0.25]), visible_ids=torch.tensor([1])
        )
    assert state["denom"][2] == 3
    assert state["grad_accum"][2] == pytest.approx(3 * 1e-3 * 400.0)
    assert state["opacity_accum"][1] == pytest.approx(0.75)
    assert state["anchor_denom"][1] == 3


def test_accumulation_without_a_gradient_is_a_no_op():
    from collab_splats.splats.scaffold import AnchorStrategy

    strategy = AnchorStrategy(ScaffoldConfig(n_offsets=2, feat_dim=8), primitive="3dgs")
    state = strategy.initialize_state(n_anchors=2)
    info = {"means2d": torch.zeros(1, 1, 2), "width": 800, "height": 600, "n_cameras": 1, "radii": torch.ones(1, 2)}
    strategy.accumulate(state, info, torch.tensor([2]), opacities=torch.tensor([0.25]), visible_ids=torch.tensor([1]))
    assert state["denom"].sum() == 0
    assert state["anchor_denom"].sum() == 0


def test_accumulation_skips_gaussians_the_projection_dropped():
    """Upstream counts gradients only for decoded Gaussians that rendered (update_filter)."""
    from collab_splats.splats.scaffold import AnchorStrategy

    strategy = AnchorStrategy(ScaffoldConfig(n_offsets=2, feat_dim=8), primitive="3dgs")
    state = strategy.initialize_state(n_anchors=4)

    means2d = torch.zeros(1, 2, 2)
    means2d.grad = torch.tensor([[[1e-3, 0.0], [1e-3, 0.0]]])
    info = {
        "means2d": means2d,
        "width": 800,
        "height": 600,
        "n_cameras": 1,
        "radii": torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
    }
    strategy.accumulate(
        state, info, torch.tensor([0, 4]), opacities=torch.tensor([0.5, 0.5]), visible_ids=torch.tensor([0, 2])
    )

    # Slot 4 never rendered, so it carries no gradient evidence; both anchors still count a visit
    assert state["denom"][0] == 1
    assert state["denom"][4] == 0
    assert state["grad_accum"][4] == 0.0
    assert list(state["anchor_denom"]) == [1, 0, 1, 0]


def test_statistics_window_opens_before_growing_does():
    """Upstream gathers statistics over (start_stat, update_until), a window wider than growing's."""
    from collab_splats.splats.scaffold import AnchorStrategy

    cfg = ScaffoldConfig(n_offsets=2, feat_dim=8, start_stat=500, update_from=1500, update_until=15000)
    strategy = AnchorStrategy(cfg, primitive="3dgs")

    assert not strategy.should_accumulate(500)
    assert strategy.should_accumulate(501)
    assert strategy.should_accumulate(1000)  # counting starts long before the first refine
    assert strategy.should_accumulate(14999)
    assert not strategy.should_accumulate(15000)


def _strategy_and_state(field):
    from collab_splats.splats.scaffold import AnchorStrategy

    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_anchors=len(field.params["anchors"]))
    return strategy, state


def test_growing_adds_an_anchor_at_a_high_gradient_slot(monkeypatch):
    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy, state = _strategy_and_state(field)

    # Upstream thins candidates at random per level; keep them all so one hot slot is a fixed outcome
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # One slot far above threshold, displaced well clear of every occupied voxel, and seen
    # for most of the window (the gate is a fraction of refine_every, not a single sighting)
    with torch.no_grad():
        field.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    state["grad_accum"][0] = 100.0
    state["denom"][0] = 100.0

    strategy.grow(field, state)
    assert len(field.params["anchors"]) > n_before

    # Growing consumes the window it grew from
    assert state["grad_accum"][0] == 0.0
    assert state["denom"][0] == 0.0


def test_growing_skips_slots_below_threshold():
    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy, state = _strategy_and_state(field)
    with torch.no_grad():
        field.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    state["grad_accum"][0] = field.cfg.grad_threshold * 0.5 * 100.0
    state["denom"][0] = 100.0

    strategy.grow(field, state)
    assert len(field.params["anchors"]) == n_before


def test_fine_levels_need_a_coarse_anchor_in_the_same_call(monkeypatch):
    """A candidate inside an occupied COARSE cell grows nothing, even where the fine grid is free."""
    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy, state = _strategy_and_state(field)
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # Land the candidate on its own anchor's level-0 cell centre: level 0 adds nothing, so upstream
    # never reaches the finer levels that would have placed it
    coarse = field.voxel_size * field.cfg.update_init_factor
    anchor = field.params["anchors"][0]
    with torch.no_grad():
        field.params["offsets"][0, 0] = (torch.round(anchor / coarse) * coarse - anchor) / field.voxel_size
    state["grad_accum"][0] = 100.0
    state["denom"][0] = 100.0

    strategy.grow(field, state)
    assert len(field.params["anchors"]) == n_before


def test_growing_does_not_duplicate_an_occupied_voxel():
    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy, state = _strategy_and_state(field)

    # Zero offset: the candidate lands in its own anchor's voxel, which is already occupied
    torch.manual_seed(0)
    state["grad_accum"][0] = 100.0
    state["denom"][0] = 100.0
    strategy.grow(field, state)
    assert len(field.params["anchors"]) == n_before


def test_growing_extends_optimizer_state_to_match():
    field = _field(n_offsets=2)
    strategy, state = _strategy_and_state(field)

    # Take one Adam step so exp_avg exists and must be grown alongside the parameters
    for name, optimizer in field.optimizers.items():
        field.params[name].grad = torch.ones_like(field.params[name])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.manual_seed(0)
    with torch.no_grad():
        field.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    state["grad_accum"][0] = 100.0
    state["denom"][0] = 100.0
    strategy.grow(field, state)

    n_anchors = len(field.params["anchors"])
    for name, optimizer in field.optimizers.items():
        exp_avg = optimizer.state[field.params[name]]["exp_avg"]
        assert len(exp_avg) == n_anchors, name
    assert len(state["grad_accum"]) == n_anchors * field.cfg.n_offsets
    assert len(state["anchor_denom"]) == n_anchors


def test_grown_anchors_inherit_the_source_feature(monkeypatch):
    field = _field(n_offsets=2)
    strategy, state = _strategy_and_state(field)
    n_before = len(field.params["anchors"])
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # Anchor 0 carries a distinctive feature; only its first slot grows
    with torch.no_grad():
        field.params["anchor_feat"][0] = 5.0
        field.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    state["grad_accum"][0] = 100.0
    state["denom"][0] = 100.0

    strategy.grow(field, state)
    grown = field.params["anchor_feat"][n_before:]
    assert len(grown) > 0
    assert torch.equal(grown, torch.full_like(grown, 5.0))


def test_anchors_are_frozen_by_default():
    field = _field(n_offsets=2)
    assert field.optimizers["anchors"].param_groups[0]["lr"] == 0.0


def test_learning_rates_decay_per_head():
    field = _field(n_offsets=2, appearance_dim=4)

    field.update_learning_rate(0)
    start = {group["name"]: group["lr"] for group in field.mlp_optimizer.param_groups}
    start_offsets = field.optimizers["offsets"].param_groups[0]["lr"]
    field.update_learning_rate(field.cfg.lr_max_steps)
    end = {group["name"]: group["lr"] for group in field.mlp_optimizer.param_groups}
    end_offsets = field.optimizers["offsets"].param_groups[0]["lr"]

    # Heads do not share one rate, and each decays on its own schedule (the cov head is flat upstream)
    assert start["mlp_colour"] > start["mlp_cov"] > start["mlp_opacity"]
    assert end["mlp_colour"] < start["mlp_colour"]
    assert end["mlp_opacity"] < start["mlp_opacity"]
    assert end["embedding_appearance"] < start["embedding_appearance"]
    assert end["mlp_cov"] == pytest.approx(start["mlp_cov"])
    assert end_offsets < start_offsets


def test_learning_rate_horizon_is_the_config_not_the_run_length():
    """Upstream keys every schedule to a fixed 30k horizon, so a short run stops partway down."""
    long_horizon = _field(n_offsets=2)
    short_horizon = _field(n_offsets=2, lr_max_steps=1000)

    long_horizon.update_learning_rate(1000)
    short_horizon.update_learning_rate(1000)
    long_lr = long_horizon.optimizers["offsets"].param_groups[0]["lr"]
    short_lr = short_horizon.optimizers["offsets"].param_groups[0]["lr"]

    assert long_horizon.cfg.lr_max_steps == 30000
    assert short_lr < long_lr
    assert short_lr == pytest.approx(short_horizon.cfg.offset_lr_final)


def test_defaults_match_the_reference_implementation():
    """GS-SR's shipped scaffold defaults, including the appearance embedding."""
    cfg = ScaffoldConfig()

    assert (cfg.feat_dim, cfg.n_offsets, cfg.appearance_dim) == (32, 10, 32)
    assert (cfg.start_stat, cfg.update_from, cfg.update_until, cfg.refine_every) == (500, 1500, 15000, 100)
    assert (cfg.update_depth, cfg.update_init_factor, cfg.update_hierarchy_factor) == (3, 16, 4)
    assert (cfg.grad_threshold, cfg.min_opacity, cfg.success_threshold) == (0.0002, 0.005, 0.8)
    assert (cfg.anchor_lr, cfg.anchor_feat_lr, cfg.scaling_lr, cfg.rotation_lr) == (0.0, 0.0075, 0.007, 0.002)
    assert (cfg.offset_lr, cfg.offset_lr_final) == (0.01, 0.0001)
    assert (cfg.mlp_opacity_lr, cfg.mlp_opacity_lr_final) == (0.002, 0.00002)
    assert (cfg.mlp_cov_lr, cfg.mlp_colour_lr, cfg.mlp_colour_lr_final) == (0.004, 0.008, 0.00005)
    assert (cfg.appearance_lr, cfg.appearance_lr_final, cfg.lr_max_steps) == (0.05, 0.0005, 30000)


def test_pruning_removes_persistently_transparent_anchors():
    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy, state = _strategy_and_state(field)

    # Anchor 0 seen many times at ~zero opacity; anchor 1 seen many times at high opacity
    state["anchor_denom"][0:2] = 100.0
    state["opacity_accum"][0] = 1e-6
    state["opacity_accum"][1] = 50.0

    strategy.prune(field, state)
    assert len(field.params["anchors"]) == n_before - 1
    assert len(state["grad_accum"]) == (n_before - 1) * 2


def test_pruning_caps_the_raw_gaussian_extent():
    """Upstream clamps scaling[:, 3:] to SCALE_CAP at every prune; the offset extent is untouched."""
    from collab_splats.splats.scaffold import SCALE_CAP

    field = _field(n_offsets=2)
    strategy, state = _strategy_and_state(field)
    with torch.no_grad():
        field.params["scaling"][:, :] = 3.0

    state["anchor_denom"][0:2] = 100.0
    state["opacity_accum"][0] = 1e-6
    state["opacity_accum"][1] = 50.0
    strategy.prune(field, state)

    assert float(field.params["scaling"][:, 3:].max()) == pytest.approx(SCALE_CAP)
    assert float(field.params["scaling"][:, :3].max()) == pytest.approx(3.0)


def test_pruning_keeps_anchors_that_were_never_seen():
    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy, state = _strategy_and_state(field)
    strategy.prune(field, state)
    assert len(field.params["anchors"]) == n_before


def test_pruning_shrinks_optimizer_state_to_match():
    field = _field(n_offsets=2)
    strategy, state = _strategy_and_state(field)
    for name, optimizer in field.optimizers.items():
        field.params[name].grad = torch.ones_like(field.params[name])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    state["anchor_denom"][0] = 100.0
    state["opacity_accum"][0] = 1e-6
    strategy.prune(field, state)

    n_anchors = len(field.params["anchors"])
    for name, optimizer in field.optimizers.items():
        assert len(optimizer.state[field.params[name]]["exp_avg"]) == n_anchors, name


def test_step_post_backward_refines_only_inside_the_window():
    from collab_splats.splats.scaffold import AnchorStrategy

    field = _field(n_offsets=2)
    field.cfg = ScaffoldConfig(n_offsets=2, feat_dim=8, update_from=10, update_until=20, refine_every=5)
    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_anchors=len(field.params["anchors"]))
    state["anchor_denom"][0] = 100.0
    state["opacity_accum"][0] = 1e-6
    n_before = len(field.params["anchors"])

    strategy.step_post_backward(field, state, step=5)  # before the window
    assert len(field.params["anchors"]) == n_before
    strategy.step_post_backward(field, state, step=12)  # inside, but not on the cadence
    assert len(field.params["anchors"]) == n_before
    strategy.step_post_backward(field, state, step=15)  # inside and on the cadence
    assert len(field.params["anchors"]) == n_before - 1

    # Each half resets the window it consumed, so the next refine starts clean
    assert state["denom"].sum() == 0
    assert state["anchor_denom"].sum() == 0
    assert state["opacity_accum"].sum() == 0


def test_decode_is_invariant_to_denormalization():
    """
    Everything written after training decodes post-denormalisation, so the heads must be scale-free.
    """
    from collab_splats.splats.trainer import denormalize_anchors

    field = _field(n_offsets=4)
    cam_to_world, intrinsics = _cam()

    # Random heads, so the invariant cannot be carried by a near-constant untrained output
    with torch.no_grad():
        for head in (field.mlps.mlp_opacity, field.mlps.mlp_cov, field.mlps.mlp_colour):
            for layer in head:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, std=0.5)
                    torch.nn.init.normal_(layer.bias, std=0.5)
    trained, _ = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)

    # Undo a normalisation the way the trainer does before writing outputs: anchors and the camera
    # both move to world units, K is unchanged, so the same anchors stay visible
    scale = 0.02
    denormalize_anchors(field.params, None, cam_to_world, np.zeros(3, dtype=np.float32), scale)
    world, _ = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)

    assert len(world["means"]) == len(trained["means"])
    assert torch.allclose(world["opacities"], trained["opacities"], atol=1e-5)
    assert torch.allclose(world["colors"], trained["colors"], atol=1e-5)
    assert torch.allclose(world["means"], trained["means"] / scale, atol=1e-4)
    assert torch.allclose(world["scales"], trained["scales"] / scale, rtol=1e-4)


def test_mlp_input_is_direction_only():
    """
    Upstream's add_opacity_dist / add_cov_dist / add_color_dist all ship off, so no distance rides in.
    """
    field = _field()
    assert field.mlps.mlp_opacity[0].in_features == field.cfg.feat_dim + 3
    assert field.mlps.mlp_cov[0].in_features == field.cfg.feat_dim + 3
    assert field.mlps.mlp_colour[0].in_features == field.cfg.feat_dim + 3
