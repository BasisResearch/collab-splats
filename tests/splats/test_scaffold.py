"""
Scaffold-GS anchors: config, MLP heads, decode, and anchor densification.
"""

import ast
import inspect
import io
import math
import tokenize
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.splats import scaffold as scaffold_module
from collab_splats.splats.gaussian import SH_C0
from collab_splats.splats.scaffold import (
    AnchorStrategy,
    Scaffold,
    ScaffoldConfig,
    ScaffoldMLPs,
)
from collab_splats.splats.trainer import SplatsConfig
from collab_splats.splats.utils import denormalize_cameras

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
    """
    A deliberate SH override is a misunderstanding of the representation, so it raises.
    """
    with pytest.raises(ValueError, match="sh_degree"):
        SplatsConfig.from_dict({"representation": "scaffold", "sh_degree": 0})
    with pytest.raises(ValueError, match="sh_degree"):
        SplatsConfig.from_dict({"representation": "scaffold", "sh_degree_interval": 500})


def test_scaffold_accepts_the_inherited_sh_defaults():
    """
    base.yaml deep-merges sh_degree/sh_degree_interval into every block at their defaults.
    """
    cfg = SplatsConfig.from_dict({"representation": "scaffold", "sh_degree": 3, "sh_degree_interval": 1000})
    assert cfg.representation == "scaffold"


########################################
# MLP heads
########################################


def test_mlp_heads_emit_per_offset_outputs():
    cfg = ScaffoldConfig(n_offsets=4, feat_dim=8, appearance_dim=0)
    mlps = ScaffoldMLPs(cfg, n_views=5)
    features = torch.zeros(6, cfg.feat_dim + 3)  # feat + unit view dir (3)
    opacity, cov, color = mlps(features, camera_id=None)
    assert opacity.shape == (6, 4)
    assert cov.shape == (6, 4 * 7)
    assert color.shape == (6, 4 * 3)
    assert opacity.min() >= -1.0 and opacity.max() <= 1.0  # tanh
    assert color.min() >= 0.0 and color.max() <= 1.0  # sigmoid


def test_appearance_embedding_changes_color_only_when_enabled():
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
    with pytest.raises(ValueError, match="n_views"):
        ScaffoldMLPs(ScaffoldConfig(appearance_dim=6), n_views=0)


########################################
# Scaffold
########################################


def _seed_points(n=500, seed=0):
    """
    A deterministic seed cloud and its colors.

    Args:
        n: point count.
        seed: numpy generator seed.

    Returns:
        ((n, 3) float32 points, (n, 3) uint8 colors).
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    colors = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    return points, colors


def _scaffold_config(run=None, **scaffold):
    """
    The run config a Scaffold takes: the scaffold representation carrying this scaffold block.

    Args:
        run: top-level run keys (`primitive`, `max_steps`); a different level from the block, never
            splatted in with it or an override configures nothing.
        scaffold: the ``scaffold:`` block itself.

    Returns:
        SplatsConfig.
    """
    return SplatsConfig.from_dict({"representation": "scaffold", "scaffold": scaffold, **(run or {})})


def test_anchor_init_voxelizes_seed_points():
    cfg = _scaffold_config(n_offsets=4, feat_dim=8)
    points, colors = _seed_points()
    model = Scaffold(cfg, points, colors, scene_scale=1.0, n_views=3, device="cpu")
    n_anchors = len(model.params["anchors"])
    assert 0 < n_anchors <= len(points)
    assert model.params["offsets"].shape == (n_anchors, 4, 3)
    assert model.params["anchor_feat"].shape == (n_anchors, 8)
    assert model.params["scaling"].shape == (n_anchors, 6)
    assert model.params["rotation"].shape == (n_anchors, 4)
    assert model.voxel_size > 0

    # Opacity is decoded per view by mlp_opacity, so there is no anchor opacity parameter to train
    assert "opacities" not in model.params


def test_anchor_count_is_invariant_to_scene_scale():
    """
    voxel_size is derived from kNN spacing, so a 10x bigger copy of a scene gets the same anchors.
    """
    cfg = _scaffold_config(n_offsets=2, feat_dim=8)
    points, colors = _seed_points()
    small = Scaffold(cfg, points, colors, scene_scale=1.0, n_views=1, device="cpu")
    large = Scaffold(cfg, points * 10.0, colors, scene_scale=10.0, n_views=1, device="cpu")
    assert len(large.params["anchors"]) == len(small.params["anchors"])
    assert large.voxel_size == pytest.approx(small.voxel_size * 10.0, rel=1e-5)


def test_explicit_voxel_size_overrides_the_derived_one():
    points, colors = _seed_points()
    model = Scaffold(
        _scaffold_config(n_offsets=2, feat_dim=8, voxel_size=0.5),
        points,
        colors,
        scene_scale=1.0,
        n_views=1,
        device="cpu",
    )
    assert model.voxel_size == 0.5


def test_optimizers_cover_every_anchor_parameter():
    points, colors = _seed_points()
    model = Scaffold(_scaffold_config(n_offsets=2, feat_dim=8), points, colors, 1.0, n_views=1, device="cpu")
    assert set(model.param_optimizers) == set(model.params)
    for optimizer in model.param_optimizers.values():
        assert len(optimizer.param_groups) == 1


########################################
# Decode
########################################


def _field(n_offsets=4, appearance_dim=0, n_views=3, device="cpu", scene_scale=1.0, run=None, **scaffold):
    """
    The one Scaffold builder in this file: a fixed 200-point seed cloud voxelized into anchors.

    - Deliberately the only builder: a second fixture is how a test stops covering its own name.

    Args:
        run: top-level run keys; other keywords land in the ``scaffold:`` block.

    Returns:
        Scaffold over 200 seed points.
    """
    cfg = _scaffold_config(run, n_offsets=n_offsets, feat_dim=8, appearance_dim=appearance_dim, **scaffold)
    points, colors = _seed_points(n=200)
    return Scaffold(cfg, points, colors, scene_scale=scene_scale, n_views=n_views, device=device)


def _cam(device="cpu"):
    """
    One axis-aligned camera 4 units back down -z.

    Args:
        device: torch device for both tensors.

    Returns:
        ((1, 4, 4) cam_to_world, (1, 3, 3) intrinsics).
    """
    cam_to_world = torch.eye(4, device=device)[None]
    cam_to_world[0, 2, 3] = -4.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)[None]
    return cam_to_world, intrinsics


def _cam_pair(device="cpu"):
    """
    Two poses seeing the whole seed cloud, far enough apart that re-normalizing their mean view
    direction is not a no-op.

    Args:
        device: torch device for both tensors.

    Returns:
        ((2, 4, 4) cam_to_world, (2, 3, 3) intrinsics).
    """
    single, intrinsics = _cam(device)
    cam_to_world = single.repeat(2, 1, 1)
    cam_to_world[1, 0, 3] = 1.5
    return cam_to_world, intrinsics.repeat(2, 1, 1)


def _open_every_offset(model):
    """
    Shift the tanh opacity head clear of zero so decode's ``keep`` mask holds every offset.

    - Untouched head: pre-activations inside +-0.6 (max 0.585 over 400 draws), tanh straddling zero
      — over 600 draws 22.8% opened every offset, 57.0% some, 20.2% none. Armed: 600 of 600.
    - Final bias only, so opacity still spreads: 0.659 to 0.972, per-draw spread >= 3.1e-3, clear of
      both ends of ``export_gaussians``' 1e-4 clamp.
    - Not bigger: at +3.0 the spread collapses to 1.7e-4, invisible to a 2e-4 comparison.

    Args:
        model: Scaffold; its `mlps.mlp_opacity` final bias is shifted in place.
    """
    with torch.no_grad():
        model.mlps.mlp_opacity[-2].bias += 1.5


def test_decode_returns_rasterizer_inputs_and_index():
    model = _field()
    _open_every_offset(model)
    cam_to_world, intrinsics = _cam()
    decoded, index = model.decode("3dgs", cam_to_world, intrinsics, width=64, height=64, camera_id=None)

    # Shapes checked over the whole slot grid, not the all-closed fallback's single row
    n = len(decoded["means"])
    assert n == model.n_primitives * model.cfg.n_offsets
    assert decoded["means"].shape == (n, 3)
    assert decoded["quats"].shape == (n, 4)
    assert decoded["scales"].shape == (n, 3)
    assert decoded["opacities"].shape == (n,)
    assert decoded["colors"].shape == (n, 3)  # post-activation RGB, so sh_degree=None at rasterize
    assert decoded["log_scales"].shape == (n, 3)  # scale_reg reads this instead of a parameter
    assert index.shape == (n,)
    assert index.dtype == torch.int64
    assert int(index.max()) < len(model.params["anchors"]) * model.cfg.n_offsets


def test_decode_index_points_at_the_generating_anchor():
    model = _field(n_offsets=2)
    _open_every_offset(model)
    cam_to_world, intrinsics = _cam()

    # Per-slot ramp arms the displacement bound
    # - `offsets` zero at init: every mean lands ON its anchor, bound holds whatever the index said
    with torch.no_grad():
        model.params["offsets"] += torch.arange(1, model.cfg.n_offsets + 1).float().view(1, -1, 1) * 0.3

    decoded, index = model.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    anchor_ids = index // model.cfg.n_offsets

    # Index is anchor * n_offsets + offset — with every slot open it walks the anchors in order
    # - a transposed encoding stays in range but does not reproduce this
    assert len(decoded["means"]) == model.n_primitives * model.cfg.n_offsets
    assert torch.equal(anchor_ids, decoded["visible_ids"].repeat_interleave(model.cfg.n_offsets))

    # Each decoded mean is its anchor plus a scaled offset, so it must sit within the offset extent
    anchors = model.params["anchors"][anchor_ids]
    offset_extent = torch.exp(model.params["scaling"][anchor_ids][:, :3])
    displacement = (decoded["means"] - anchors).abs()
    assert torch.all(displacement > 0)
    assert torch.all(displacement <= offset_extent * 1.001 + 1e-6)


def test_decode_drops_offsets_with_non_positive_opacity():
    model = _field(n_offsets=4)
    cam_to_world, intrinsics = _cam()

    # Close offsets 0 and 2, open 1 and 3: the tanh head's last linear bias dominates the feature input
    with torch.no_grad():
        model.mlps.mlp_opacity[-2].weight.zero_()
        model.mlps.mlp_opacity[-2].bias.copy_(torch.tensor([-5.0, 5.0, -5.0, 5.0]))
    decoded, index = model.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    n_visible = int(model.visible_anchors(cam_to_world, intrinsics, 64, 64).sum())
    assert len(decoded["means"]) == 2 * n_visible
    assert set((index % 4).tolist()) == {1, 3}


def test_decode_is_differentiable_into_the_mlps():
    model = _field()
    _open_every_offset(model)
    cam_to_world, intrinsics = _cam()
    decoded, _ = model.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    decoded["colors"].sum().backward()

    # `grad is not None` collapses 163 anchors to one boolean
    # - 6 of 200 unarmed draws: no offset opened, decode fell back to one slot, gradient reached 1
    #   anchor, test passed
    # - armed, the same draws reached >= 43 of 163 anchors; the count below fails on a collapse
    assert model.mlps.mlp_color[0].weight.grad is not None
    assert model.params["anchor_feat"].grad is not None
    assert int((model.params["anchor_feat"].grad.abs().sum(dim=-1) > 0).sum()) > 1


def test_decode_2dgs_zeroes_the_third_scale():
    model = _field()
    _open_every_offset(model)
    cam_to_world, intrinsics = _cam()
    decoded, _ = model.decode("2dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)

    # Over the whole slot grid, not the fallback's single row
    # - against a live first pair: an all-zero `scales` would satisfy the third-channel check alone
    assert len(decoded["scales"]) == model.n_primitives * model.cfg.n_offsets
    assert torch.all(decoded["scales"][:, :2] > 0.0)
    assert torch.all(decoded["scales"][:, 2] == 0.0)


def test_decode_frustum_filter_drops_anchors_behind_the_camera():
    model = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()
    visible = model.visible_anchors(cam_to_world, intrinsics, 64, 64)

    # Camera sits at z = -4 looking down +z, so a point far behind it can never be visible
    with torch.no_grad():
        model.params["anchors"][0] = torch.tensor([0.0, 0.0, -100.0])
    assert not bool(model.visible_anchors(cam_to_world, intrinsics, 64, 64)[0])
    assert bool(visible.any())


def test_decode_never_returns_zero_gaussians_when_every_offset_is_closed():
    """
    gsplat's projection kernel raises SIGFPE on an empty input, so decode keeps the best offset.
    """
    model = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()

    # Drive every neural opacity negative: the tanh head saturates at -1 for a large negative bias
    with torch.no_grad():
        model.mlps.mlp_opacity[-2].bias.fill_(-50.0)
        model.mlps.mlp_opacity[-2].weight.zero_()
    decoded, decode_index = model.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    assert len(decoded["means"]) == 1
    assert len(decode_index) == 1


def test_decode_falls_back_to_every_anchor_when_the_frustum_is_empty():
    """
    A camera looking away culls every anchor; decode still emits Gaussians the rasterizer culls.
    """
    model = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()

    # Push every anchor far behind the camera, which sits at z = -4 looking down +z
    with torch.no_grad():
        model.params["anchors"].data[:] = torch.tensor([0.0, 0.0, -100.0])
    assert not bool(model.visible_anchors(cam_to_world, intrinsics, 64, 64).any())
    decoded, _ = model.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    assert len(decoded["means"]) > 0

    # `len(means) > 0` cannot separate this fallback from the all-closed one (also non-empty, one row)
    # - the anchor set is what this fallback restores
    assert torch.equal(decoded["visible_ids"], torch.arange(model.n_primitives))


########################################
# Rasterization
########################################


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_scaffold_decode_renders_through_gsplat(primitive):
    from collab_splats.splats.rendering import render_gaussians

    model = _field(device="cuda")
    cam_to_world, intrinsics = _cam(device="cuda")
    decoded, _ = model.decode(primitive, cam_to_world, intrinsics, 64, 64, camera_id=None)
    render, info = render_gaussians(primitive, decoded, cam_to_world, intrinsics, 64, 64, sh_degree=None, absgrad=False)
    assert render["rgb"].shape == (1, 64, 64, 3)
    assert render["depth"].shape == (1, 64, 64, 1)
    expected_gradient_key = "means2d" if primitive == "3dgs" else "gradient_2dgs"
    assert expected_gradient_key in info


@cuda
def test_render_gradient_reaches_the_anchor_features():
    from collab_splats.splats.rendering import render_gaussians

    model = _field(device="cuda")
    _open_every_offset(model)
    cam_to_world, intrinsics = _cam(device="cuda")
    decoded, _ = model.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    render, _ = render_gaussians("3dgs", decoded, cam_to_world, intrinsics, 64, 64, sh_degree=None, absgrad=False)
    render["rgb"].sum().backward()

    # `grad is not None` and `isfinite` both hold on an all-zero gradient
    # - 7 of 120 unarmed draws reached ZERO anchors and this test passed on all 7 — exactly the draws
    #   where no offset opened and decode fell back to one slot
    # - armed, the same draws reached >= 133 of 163 anchors over 200 trials, never zero
    # - the `any()` below is what fails when the gradient stops arriving
    assert model.params["anchor_feat"].grad is not None
    assert torch.isfinite(model.params["anchor_feat"].grad).all()
    assert bool((model.params["anchor_feat"].grad.abs().sum(dim=-1) > 0).any())


########################################
# Densification
########################################


def _bare_strategy(cfg, n_anchors, primitive="3dgs"):
    """
    An AnchorStrategy with no Scaffold behind it, for the accumulator arithmetic on its own.

    Args:
        cfg: the ScaffoldConfig.
        n_anchors: sizes the four accumulators the strategy owns outright.
        primitive: "3dgs" or "2dgs".

    Returns:
        AnchorStrategy on the CPU, voxel_size 0.
    """
    return AnchorStrategy(cfg, primitive=primitive, voxel_size=0.0, n_anchors=n_anchors, device="cpu")


def _accumulation_info(grad, radii, decode_index, opacities, visible_ids, key="means2d", shape=None):
    """
    The gsplat info dict `accumulate` reads, with the retained gradient already attached.

    Args:
        grad: means2d gradient; None models a backward that never reached the tensor.
        radii, decode_index, opacities, visible_ids: the rest of the dict, verbatim.
        key: gradient key, "means2d" or "gradient_2dgs".
        shape: sizes the stand-in tensor when `grad` is None.

    Returns:
        The info dict.
    """
    means2d = torch.zeros(shape) if grad is None else torch.zeros_like(grad)
    means2d.grad = grad
    return {
        key: means2d,
        "width": 800,
        "height": 600,
        "n_cameras": 1,
        "radii": radii,
        "decode_index": decode_index,
        "decoded_opacities": opacities,
        "visible_ids": visible_ids,
    }


# Any step inside the default (start_stat, update_until) window
# - the accumulation tests exercise the arithmetic, not the window guard folded into `accumulate`
COUNTING_STEP = 1000


def test_gradient_key_follows_the_primitive():
    assert _bare_strategy(ScaffoldConfig(), 4, primitive="3dgs").key_for_gradient == "means2d"
    assert _bare_strategy(ScaffoldConfig(), 4, primitive="2dgs").key_for_gradient == "gradient_2dgs"


def test_accumulation_renormalizes_gradients_like_gsplat():
    """
    gsplat's DefaultStrategy scales means2d grads to [-1, 1] screen space before thresholding.
    """
    strategy = _bare_strategy(ScaffoldConfig(n_offsets=2, feat_dim=8), n_anchors=4)

    info = _accumulation_info(
        grad=torch.tensor([[[1e-3, 0.0], [0.0, 2e-3], [0.0, 0.0]]]),
        radii=torch.ones(3, 2),
        decode_index=torch.tensor([0, 5, 7]),
        opacities=torch.tensor([0.5, 0.5, 0.5]),
        visible_ids=torch.tensor([0, 2, 3]),
    )
    strategy.accumulate(COUNTING_STEP, info)

    assert strategy.offset_gradient_accum[0] == pytest.approx(1e-3 * 400.0)
    assert strategy.offset_gradient_accum[5] == pytest.approx(2e-3 * 300.0)
    assert strategy.offset_denom[0] == 1
    assert strategy.offset_denom[1] == 0

    # Opacity is per anchor (slots 5 and 7 belong to anchors 2 and 3)
    # - every visible anchor counts a visit whether or not any offset rendered
    assert strategy.opacity_accum[0] == pytest.approx(0.5)
    assert strategy.opacity_accum[2] == pytest.approx(0.5)
    assert list(strategy.anchor_denom) == [1, 0, 1, 1]


def test_accumulation_is_additive_over_steps():
    strategy = _bare_strategy(ScaffoldConfig(n_offsets=2, feat_dim=8), n_anchors=2)
    info = _accumulation_info(
        grad=torch.tensor([[[1e-3, 0.0]]]),
        radii=torch.ones(1, 2),
        decode_index=torch.tensor([2]),
        opacities=torch.tensor([0.25]),
        visible_ids=torch.tensor([1]),
    )
    for _ in range(3):
        strategy.accumulate(COUNTING_STEP, info)

    assert strategy.offset_denom[2] == 3
    assert strategy.offset_gradient_accum[2] == pytest.approx(3 * 1e-3 * 400.0)
    assert strategy.opacity_accum[1] == pytest.approx(0.75)
    assert strategy.anchor_denom[1] == 3


def test_accumulation_without_a_gradient_is_a_no_op():
    strategy = _bare_strategy(ScaffoldConfig(n_offsets=2, feat_dim=8), n_anchors=2)
    info = _accumulation_info(
        grad=None,
        shape=(1, 1, 2),
        radii=torch.ones(1, 2),
        decode_index=torch.tensor([2]),
        opacities=torch.tensor([0.25]),
        visible_ids=torch.tensor([1]),
    )

    strategy.accumulate(COUNTING_STEP, info)

    assert strategy.offset_denom.sum() == 0
    assert strategy.anchor_denom.sum() == 0


def test_accumulation_skips_gaussians_the_projection_dropped():
    """
    Upstream counts gradients only for decoded Gaussians that rendered (update_filter).
    """
    strategy = _bare_strategy(ScaffoldConfig(n_offsets=2, feat_dim=8), n_anchors=4)

    info = _accumulation_info(
        grad=torch.tensor([[[1e-3, 0.0], [1e-3, 0.0]]]),
        radii=torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
        decode_index=torch.tensor([0, 4]),
        opacities=torch.tensor([0.5, 0.5]),
        visible_ids=torch.tensor([0, 2]),
    )
    strategy.accumulate(COUNTING_STEP, info)

    # Slot 4 never rendered, so it carries no gradient evidence; both anchors still count a visit
    assert strategy.offset_denom[0] == 1
    assert strategy.offset_denom[4] == 0
    assert strategy.offset_gradient_accum[4] == 0.0
    assert list(strategy.anchor_denom) == [1, 0, 1, 0]


@pytest.mark.parametrize(
    "step, counts",
    [(500, False), (501, True), (1000, True), (14999, True), (15000, False)],
)
def test_statistics_window_opens_before_growing_does(step, counts):
    """
    Upstream gathers statistics over (start_stat, update_until), a window wider than growing's.

    - Both bounds exclusive; counting starts before the first refine at update_from=1500, so that
      refine reads a full window.
    - Asserted through `accumulate`'s own effect: the guard is folded into it.
    """
    cfg = ScaffoldConfig(n_offsets=2, feat_dim=8, start_stat=500, update_from=1500, update_until=15000)
    strategy = _bare_strategy(cfg, n_anchors=4)
    info = _accumulation_info(
        grad=torch.tensor([[[1e-3, 0.0]]]),
        radii=torch.ones(1, 2),
        decode_index=torch.tensor([0]),
        opacities=torch.tensor([0.5]),
        visible_ids=torch.tensor([0]),
    )

    strategy.accumulate(step, info)

    assert bool(strategy.offset_denom.sum() > 0) is counts
    assert bool(strategy.anchor_denom.sum() > 0) is counts


def _strategy(model):
    """
    A fresh strategy sized to `model`'s anchors — the state it densifies from lives on it.

    Args:
        model: the Scaffold whose cfg, voxel_size, anchor count and device are read.

    Returns:
        AnchorStrategy.
    """
    return AnchorStrategy(
        model.cfg,
        primitive="3dgs",
        voxel_size=model.voxel_size,
        n_anchors=len(model.params["anchors"]),
        device=model.device,
    )


def test_growing_adds_an_anchor_at_a_high_gradient_slot(monkeypatch):
    model = _field(n_offsets=2)
    n_before = len(model.params["anchors"])
    strategy = _strategy(model)

    # Upstream thins candidates at random per level; keep them all so one hot slot is a fixed outcome
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # One slot far above threshold, displaced clear of every occupied voxel
    # - seen for most of the window: the gate is a fraction of refine_every, not a single sighting
    with torch.no_grad():
        model.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    strategy.offset_gradient_accum[0] = 100.0
    strategy.offset_denom[0] = 100.0

    strategy.grow(model)
    assert len(model.params["anchors"]) > n_before

    # Growing consumes the window it grew from
    assert strategy.offset_gradient_accum[0] == 0.0
    assert strategy.offset_denom[0] == 0.0


def test_growing_skips_slots_below_threshold():
    model = _field(n_offsets=2)
    n_before = len(model.params["anchors"])
    strategy = _strategy(model)
    with torch.no_grad():
        model.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    strategy.offset_gradient_accum[0] = model.cfg.grad_threshold * 0.5 * 100.0
    strategy.offset_denom[0] = 100.0

    strategy.grow(model)
    assert len(model.params["anchors"]) == n_before


@pytest.mark.parametrize("window_fraction, grows", [(0.4, False), (0.75, True)])
def test_the_seen_gate_opens_at_half_a_refine_window(monkeypatch, window_fraction, grows):
    """
    A slot counts after HALF of `refine_every * success_threshold` decodes, not all of them.
    """
    model = _field(n_offsets=2)
    n_before = len(model.params["anchors"])
    strategy = _strategy(model)
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # The mean gradient clears the threshold in both arms, so only the window decides
    # - 0.4 sits below upstream's 0.5 gate, 0.75 above it and below a whole window
    window = model.cfg.refine_every * model.cfg.success_threshold
    with torch.no_grad():
        model.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    strategy.offset_gradient_accum[0] = 100.0
    strategy.offset_denom[0] = window * window_fraction

    strategy.grow(model)
    assert (len(model.params["anchors"]) > n_before) is grows


def test_growing_leaves_unseen_slots_accumulating(monkeypatch):
    """
    Only slots that cleared the window reset; a rarely-decoded one keeps building towards one.
    """
    model = _field(n_offsets=2)
    strategy = _strategy(model)
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # Slot 0 grows and is consumed; slot 1 has one sighting, far short of the window
    with torch.no_grad():
        model.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    strategy.offset_gradient_accum[0] = 100.0
    strategy.offset_denom[0] = 100.0
    strategy.offset_gradient_accum[1] = 7.0
    strategy.offset_denom[1] = 1.0

    assert strategy.grow(model) > 0

    assert strategy.offset_gradient_accum[0] == 0.0
    assert strategy.offset_denom[0] == 0.0
    assert strategy.offset_gradient_accum[1] == 7.0
    assert strategy.offset_denom[1] == 1.0


def test_scaffold_init_tuning_literals_are_keyword_only():
    cfg = _scaffold_config()
    points, colors = _seed_points(n=200)

    # lr_decay sits behind a bare `*`: an eighth positional must be rejected
    # - `device` is the seventh and last, so the count string pins which slot the eighth hit
    with pytest.raises(TypeError, match="takes 7 positional arguments but 8 were given"):
        Scaffold(cfg, points, colors, 1.0, 3, "cpu", 0.5)


def test_fine_levels_need_a_coarse_anchor_in_the_same_call(monkeypatch):
    """
    A candidate inside an occupied COARSE cell grows nothing, even where the fine grid is free.
    """
    model = _field(n_offsets=2)
    n_before = len(model.params["anchors"])
    strategy = _strategy(model)
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # Land the candidate on its own anchor's level-0 cell center
    # - level 0 adds nothing, so upstream never reaches the finer levels that would have placed it
    coarse = model.voxel_size * model.cfg.update_init_factor
    anchor = model.params["anchors"][0]
    with torch.no_grad():
        model.params["offsets"][0, 0] = (torch.round(anchor / coarse) * coarse - anchor) / model.voxel_size
    strategy.offset_gradient_accum[0] = 100.0
    strategy.offset_denom[0] = 100.0

    strategy.grow(model)
    assert len(model.params["anchors"]) == n_before


def test_growing_does_not_duplicate_an_occupied_voxel():
    model = _field(n_offsets=2)
    n_before = len(model.params["anchors"])
    strategy = _strategy(model)

    # Zero offset: the candidate lands in its own anchor's voxel, which is already occupied
    torch.manual_seed(0)
    strategy.offset_gradient_accum[0] = 100.0
    strategy.offset_denom[0] = 100.0
    strategy.grow(model)
    assert len(model.params["anchors"]) == n_before


def test_growing_extends_optimizer_state_to_match():
    model = _field(n_offsets=2)
    strategy = _strategy(model)

    # Take one Adam step so exp_avg exists and must be grown alongside the parameters
    for name, optimizer in model.param_optimizers.items():
        model.params[name].grad = torch.ones_like(model.params[name])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.manual_seed(0)
    with torch.no_grad():
        model.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    strategy.offset_gradient_accum[0] = 100.0
    strategy.offset_denom[0] = 100.0
    strategy.grow(model)

    n_anchors = len(model.params["anchors"])
    for name, optimizer in model.param_optimizers.items():
        exp_avg = optimizer.state[model.params[name]]["exp_avg"]
        assert len(exp_avg) == n_anchors, name
    assert len(strategy.offset_gradient_accum) == n_anchors * model.cfg.n_offsets
    assert len(strategy.anchor_denom) == n_anchors


def test_grown_anchors_inherit_the_source_feature(monkeypatch):
    model = _field(n_offsets=2)
    strategy = _strategy(model)
    n_before = len(model.params["anchors"])
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # Anchor 0 carries a distinctive feature; only its first slot grows
    with torch.no_grad():
        model.params["anchor_feat"][0] = 5.0
        model.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    strategy.offset_gradient_accum[0] = 100.0
    strategy.offset_denom[0] = 100.0

    strategy.grow(model)
    grown = model.params["anchor_feat"][n_before:]
    assert len(grown) > 0
    assert torch.equal(grown, torch.full_like(grown, 5.0))


def test_anchors_are_frozen_by_default():
    model = _field(n_offsets=2)
    assert model.param_optimizers["anchors"].param_groups[0]["lr"] == 0.0


def test_learning_rates_decay_per_head():
    model = _field(n_offsets=2, appearance_dim=4, lr_max_steps=200)

    # The schedulers are constructed at step 0, so the groups already carry their initial lrs
    start = {group["name"]: group["lr"] for group in model.mlp_optimizer.param_groups}
    start_offsets = model.param_optimizers["offsets"].param_groups[0]["lr"]
    for _ in range(model.cfg.lr_max_steps):
        for scheduler in model.schedulers:
            scheduler.step()
    end = {group["name"]: group["lr"] for group in model.mlp_optimizer.param_groups}
    end_offsets = model.param_optimizers["offsets"].param_groups[0]["lr"]

    # Heads do not share one rate, and each decays on its own schedule (the cov head is flat upstream)
    assert start["mlp_color"] > start["mlp_cov"] > start["mlp_opacity"]
    assert end["mlp_color"] < start["mlp_color"]
    assert end["mlp_opacity"] < start["mlp_opacity"]
    assert end["embedding_appearance"] < start["embedding_appearance"]
    assert end["mlp_cov"] == pytest.approx(start["mlp_cov"])
    assert end_offsets < start_offsets


def test_learning_rate_horizon_is_the_config_not_the_run_length():
    """
    Upstream keys every schedule to a fixed 30k horizon, so a short run stops partway down.
    """
    long_horizon = _field(n_offsets=2)
    short_horizon = _field(n_offsets=2, lr_max_steps=1000)

    for _ in range(1000):
        long_horizon.offset_scheduler.step()
        short_horizon.offset_scheduler.step()
    long_lr = long_horizon.param_optimizers["offsets"].param_groups[0]["lr"]
    short_lr = short_horizon.param_optimizers["offsets"].param_groups[0]["lr"]

    assert long_horizon.cfg.lr_max_steps == 30000
    assert short_lr < long_lr
    assert short_lr == pytest.approx(short_horizon.cfg.offset_lr_final)


def test_defaults_match_the_reference_implementation():
    """
    GS-SR's shipped scaffold defaults, including the appearance embedding.
    """
    cfg = ScaffoldConfig()

    assert (cfg.feat_dim, cfg.n_offsets, cfg.appearance_dim) == (32, 10, 32)
    assert (cfg.start_stat, cfg.update_from, cfg.update_until, cfg.refine_every) == (500, 1500, 15000, 100)
    assert (cfg.update_depth, cfg.update_init_factor, cfg.update_hierarchy_factor) == (3, 16, 4)
    assert (cfg.grad_threshold, cfg.min_opacity, cfg.success_threshold) == (0.0002, 0.005, 0.8)
    assert (cfg.anchor_lr, cfg.anchor_feat_lr, cfg.scaling_lr, cfg.rotation_lr) == (0.0, 0.0075, 0.007, 0.002)
    assert (cfg.offset_lr, cfg.offset_lr_final) == (0.01, 0.0001)
    assert (cfg.mlp_opacity_lr, cfg.mlp_opacity_lr_final) == (0.002, 0.00002)
    assert (cfg.mlp_cov_lr, cfg.mlp_color_lr, cfg.mlp_color_lr_final) == (0.004, 0.008, 0.00005)
    assert (cfg.appearance_lr, cfg.appearance_lr_final, cfg.lr_max_steps) == (0.05, 0.0005, 30000)


def test_pruning_removes_persistently_transparent_anchors():
    model = _field(n_offsets=2)
    n_before = len(model.params["anchors"])
    strategy = _strategy(model)

    # Anchor 0 seen many times at ~zero opacity; anchor 1 seen many times at high opacity
    strategy.anchor_denom[0:2] = 100.0
    strategy.opacity_accum[0] = 1e-6
    strategy.opacity_accum[1] = 50.0

    strategy.prune(model)
    assert len(model.params["anchors"]) == n_before - 1
    assert len(strategy.offset_gradient_accum) == (n_before - 1) * 2


def test_pruning_caps_the_raw_gaussian_extent():
    """
    Upstream clamps scaling[:, 3:] to the 0.05 cap at every prune; the offset extent is untouched.
    """
    model = _field(n_offsets=2)
    strategy = _strategy(model)
    with torch.no_grad():
        model.params["scaling"][:, :] = 3.0

    strategy.anchor_denom[0:2] = 100.0
    strategy.opacity_accum[0] = 1e-6
    strategy.opacity_accum[1] = 50.0
    strategy.prune(model)

    # The literal, not a re-import of the default: a cap that silently moved would still pass that
    assert float(model.params["scaling"][:, 3:].max()) == pytest.approx(0.05)
    assert float(model.params["scaling"][:, :3].max()) == pytest.approx(3.0)


def test_the_scale_cap_is_a_keyword_a_caller_can_move():
    """
    The 0.05 clamp is a default, not a constant welded into prune.
    """
    model = _field(n_offsets=2)
    strategy = _strategy(model)
    with torch.no_grad():
        model.params["scaling"][:, :] = 3.0

    strategy.anchor_denom[0:2] = 100.0
    strategy.opacity_accum[0] = 1e-6
    strategy.opacity_accum[1] = 50.0
    strategy.prune(model, scale_cap=0.5)

    assert float(model.params["scaling"][:, 3:].max()) == pytest.approx(0.5)
    assert inspect.signature(AnchorStrategy.prune).parameters["scale_cap"].default == 0.05


def test_prune_takes_the_scale_cap_as_a_keyword_only_argument():
    """
    Ground rule 9: a tunable is keyword-only, so a caller cannot bind it by position.

    - Dropping the bare `*` from `prune` left all 346 of tests/splats green (225abd6f).
    """
    parameter = inspect.signature(AnchorStrategy.prune).parameters["scale_cap"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_prune_refuses_the_scale_cap_positionally():
    """
    The property above taken from behavior, a second axis over the same `*`.

    - `def prune(self, scaffold, *args, scale_cap=0.05)`: also KEYWORD_ONLY, so the signature
      check cannot see it and this one can
    - that form swallows a positional cap and raises nothing — already red in
      `test_anchor_strategy_calls_do_not_take_a_state_argument`, which compares names
    - `pytest.raises(TypeError)`: what excludes it; `match=` only pins CPython's arity wording
    """
    model = _field(n_offsets=2)
    strategy = _strategy(model)

    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
        strategy.prune(model, 0.5)


def test_pruning_keeps_anchors_that_were_never_seen():
    model = _field(n_offsets=2)
    n_before = len(model.params["anchors"])
    strategy = _strategy(model)
    strategy.prune(model)
    assert len(model.params["anchors"]) == n_before


def test_pruning_shrinks_optimizer_state_to_match():
    model = _field(n_offsets=2)
    strategy = _strategy(model)
    for name, optimizer in model.param_optimizers.items():
        model.params[name].grad = torch.ones_like(model.params[name])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    strategy.anchor_denom[0] = 100.0
    strategy.opacity_accum[0] = 1e-6
    strategy.prune(model)

    n_anchors = len(model.params["anchors"])
    for name, optimizer in model.param_optimizers.items():
        assert len(optimizer.state[model.params[name]]["exp_avg"]) == n_anchors, name


def test_refine_acts_only_inside_the_window():
    """
    Both window bounds are EXCLUSIVE upstream, and both land on the refine cadence here.

    - `update_from` / `update_until` are multiples of `refine_every`, so a bound relaxed to `<=`
      refines on a step it must not; asserting only the interior hides both flips.
    """
    model = _field(n_offsets=2)
    model.cfg = ScaffoldConfig(n_offsets=2, feat_dim=8, update_from=10, update_until=20, refine_every=5)
    strategy = _strategy(model)  # reads the window off model.cfg, so it must be built after it
    n_before = len(model.params["anchors"])

    def arm():
        strategy.anchor_denom[0] = 100.0
        strategy.opacity_accum[0] = 1e-6

    arm()
    strategy.refine(model, step=5)  # before the window
    assert len(model.params["anchors"]) == n_before
    strategy.refine(model, step=10)  # exactly ON update_from, which is exclusive
    assert len(model.params["anchors"]) == n_before
    strategy.refine(model, step=12)  # inside, but not on the cadence
    assert len(model.params["anchors"]) == n_before
    strategy.refine(model, step=15)  # inside and on the cadence
    assert len(model.params["anchors"]) == n_before - 1

    # Each half resets the window it consumed, so the next refine starts clean
    assert strategy.offset_denom.sum() == 0
    assert strategy.anchor_denom.sum() == 0
    assert strategy.opacity_accum.sum() == 0

    # Re-armed
    # - otherwise the upper bound is asserted against an empty window and passes whether or not it fired
    arm()
    strategy.refine(model, step=20)  # exactly ON update_until, which is exclusive
    assert len(model.params["anchors"]) == n_before - 1
    assert strategy.anchor_denom.sum() == 100.0


def test_decode_is_invariant_to_denormalization():
    """
    Everything written after training decodes post-denormalization, so the heads must be scale-free.
    """
    model = _field(n_offsets=4)
    cam_to_world, intrinsics = _cam()

    # Random cov and color heads, so a near-constant untrained output cannot carry the invariant
    # - opacity head keeps its own init and is armed open: at std=0.5 it closes every offset on 4.2%
    #   of draws (17/400), leaving the invariant asserted over a single row
    with torch.no_grad():
        for head in (model.mlps.mlp_cov, model.mlps.mlp_color):
            for layer in head:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, std=0.5)
                    torch.nn.init.normal_(layer.bias, std=0.5)
    _open_every_offset(model)
    trained, _ = model.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)

    # Invariance over the whole slot grid
    # - opacities vary across rows by far more than the 1e-5 tolerance; a constant column would pass
    assert len(trained["means"]) == model.n_primitives * model.cfg.n_offsets
    assert float(trained["opacities"].max() - trained["opacities"].min()) > 1e-3

    # Undo a normalization the way the trainer does before writing outputs
    # - anchors and camera both move to world units, K unchanged, so the same anchors stay visible
    scale = 0.02
    center = np.zeros(3, dtype=np.float32)
    model.denormalize(center, scale)
    denormalize_cameras(cam_to_world, center, scale)
    world, _ = model.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)

    assert len(world["means"]) == len(trained["means"])
    assert torch.allclose(world["opacities"], trained["opacities"], atol=1e-5)
    assert torch.allclose(world["colors"], trained["colors"], atol=1e-5)
    assert torch.allclose(world["means"], trained["means"] / scale, atol=1e-4)
    assert torch.allclose(world["scales"], trained["scales"] / scale, rtol=1e-4)


def test_mlp_input_is_direction_only():
    """
    Upstream's add_opacity_dist / add_cov_dist / add_color_dist ship off, so no distance rides in.
    """
    model = _field()
    assert model.mlps.mlp_opacity[0].in_features == model.cfg.feat_dim + 3
    assert model.mlps.mlp_cov[0].in_features == model.cfg.feat_dim + 3
    assert model.mlps.mlp_color[0].in_features == model.cfg.feat_dim + 3


########################################
# Model interface
########################################


def _window(model, **overrides):
    """
    Re-key the model's strategy to a tight refine window so one post_backward counts AND refines.

    Args:
        model: the Scaffold; its cfg and strategy are replaced in place.
        overrides: ScaffoldConfig fields to override.
    """
    model.cfg = replace(model.cfg, **overrides)
    model.strategy = AnchorStrategy(
        model.cfg,
        model.primitive,
        model.voxel_size,
        len(model.params["anchors"]),
        model.device,
    )


def _backward_info(model, anchors=(0,), opacity=1e-6):
    """
    The gsplat info dict a step's render hands post_backward, for one view that saw `anchors`.

    Args:
        model: supplies the gradient key and n_offsets.
        anchors: anchor ids the view saw; offset 0 of each is the decoded slot.
        opacity: decoded opacity for every slot.

    Returns:
        The info dict.
    """
    slots = torch.tensor([anchor * model.cfg.n_offsets for anchor in anchors])
    means2d = torch.zeros(1, len(slots), 2)
    means2d.grad = torch.zeros(1, len(slots), 2)
    return {
        model.strategy.key_for_gradient: means2d,
        "width": 64,
        "height": 64,
        "n_cameras": 1,
        "radii": torch.ones(len(slots), 2),
        "decode_index": slots,
        "decoded_opacities": torch.full((len(slots),), opacity),
        "visible_ids": torch.tensor(anchors),
    }


def _reachable_state(model):
    """
    Every value reachable from model and strategy, flattened to {path: cloned tensor or repr}.

    - ``vars()`` keys alone miss a write INTO an existing container — the shape a leak would take.
    - Descends into ``vars(model.strategy)`` too: it owns the accumulators, the natural cache site;
      the model-level entry for it is only a repr.

    Args:
        model: the Scaffold to walk.

    Returns:
        {dotted path: cloned tensor or repr}.
    """
    state = {}

    def walk(namespace, prefix=""):
        for name, value in namespace.items():
            if torch.is_tensor(value):
                state[f"{prefix}{name}"] = value.detach().clone()
            elif isinstance(value, dict):
                for key, item in value.items():
                    state[f"{prefix}{name}.{key}"] = item.detach().clone() if torch.is_tensor(item) else repr(item)
            else:
                state[f"{prefix}{name}"] = repr(value)

    walk(vars(model))
    walk(vars(model.strategy), prefix="strategy.")

    # The two containers that are Modules rather than dicts, so the walk above only saw their repr
    for name, tensor in model.params.items():
        state[f"params.{name}"] = tensor.detach().clone()
    for name, tensor in model.mlps.state_dict().items():
        state[f"mlps.{name}"] = tensor.detach().clone()
    return state


def test_scaffold_exposes_anchor_parameters():
    model = _field(n_offsets=2)

    assert set(model.params) == {"anchors", "offsets", "anchor_feat", "scaling", "rotation"}
    assert model.n_primitives == len(model.params["anchors"])


def test_scaffold_optimizers_is_a_flat_list_covering_every_tensor_and_the_mlps():
    model = _field(n_offsets=2)

    # Five anchor tensors plus the one multi-group MLP optimizer
    assert len(model.optimizers) == 6
    assert model.mlp_optimizer in model.optimizers
    assert set(model.param_optimizers) == set(model.params)


def test_scaffold_lambda_schedulers_reproduce_the_exponential_curve():
    model = _field(n_offsets=2, scene_scale=2.0)
    lr_max_steps = model.cfg.lr_max_steps
    offset_optimizer = model.param_optimizers["offsets"]
    lr_init = offset_optimizer.param_groups[0]["lr"]

    # Step 0: the schedule has not moved
    assert lr_init == pytest.approx(model.cfg.offset_lr * 2.0, rel=1e-6)

    # Halfway along the schedule the lr is the geometric mean of the endpoints
    for _ in range(lr_max_steps // 2):
        for scheduler in model.schedulers:
            scheduler.step()
    expected = math.sqrt((model.cfg.offset_lr * 2.0) * (model.cfg.offset_lr_final * 2.0))
    assert offset_optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-4)


def test_scaffold_lambda_schedulers_hold_at_the_final_lr_past_the_horizon():
    model = _field(n_offsets=2, scene_scale=2.0)
    offset_optimizer = model.param_optimizers["offsets"]

    for _ in range(model.cfg.lr_max_steps + 50):
        for scheduler in model.schedulers:
            scheduler.step()

    assert offset_optimizer.param_groups[0]["lr"] == pytest.approx(model.cfg.offset_lr_final * 2.0, rel=1e-4)


def test_anchor_scheduler_decays_over_the_run_not_the_lr_horizon():
    """
    The anchor lr is the one schedule keyed to the RUN length, like the vanilla means lr.
    """
    model = _field(n_offsets=2, run={"max_steps": 100}, anchor_lr=0.01)
    anchors = model.param_optimizers["anchors"]
    lr_init = anchors.param_groups[0]["lr"]

    for _ in range(100):
        model.anchor_scheduler.step()

    # lr_decay is a total 0.01 over max_steps, so a 100-step run lands there after exactly 100 steps
    assert lr_init == pytest.approx(0.01)
    assert anchors.param_groups[0]["lr"] == pytest.approx(lr_init * 0.01, rel=1e-5)
    assert len(model.schedulers) == 3
    assert set(model.schedulers) == {model.anchor_scheduler, model.offset_scheduler, model.mlp_scheduler}


def test_scaffold_denormalize_inverts_the_sim3():
    model = _field(n_offsets=2)
    center = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    scale = 0.25
    anchors_before = model.params["anchors"].detach().clone()
    scaling_before = model.params["scaling"].detach().clone()
    offsets_before = model.params["offsets"].detach().clone()

    model.denormalize(center, scale)

    assert torch.allclose(model.params["anchors"], anchors_before / scale + torch.from_numpy(center), atol=1e-5)
    assert torch.allclose(model.params["scaling"], scaling_before - math.log(scale), atol=1e-5)
    # Offsets are stored in units of the anchor's own extent, so they are scale-free
    assert torch.allclose(model.params["offsets"], offsets_before)


def test_scaffold_checkpoint_carries_the_mlps_and_the_voxel_size():
    model = _field(n_offsets=2, appearance_dim=4, n_views=4)
    ckpt = model.checkpoint()

    assert set(ckpt) == {"splats", "mlps", "voxel_size"}
    assert ckpt["voxel_size"] == pytest.approx(model.voxel_size)
    assert ckpt["mlps"]["embedding_appearance.weight"].shape == (4, 4)


def test_scaffold_checkpoint_round_trips():
    model = _field(n_offsets=2, appearance_dim=4, n_views=4)
    ckpt = model.checkpoint()
    ckpt["config"] = {"primitive": model.primitive, "scaffold": asdict(model.cfg)}

    restored = Scaffold.from_checkpoint(ckpt, "cpu")

    assert restored.n_primitives == model.n_primitives
    assert torch.allclose(restored.params["anchors"], model.params["anchors"])

    # The heads and the voxel grid come back too, not just the anchor tensors
    # - the ply bake reads both; a checkpoint restoring fresh heads renders noise
    assert restored.voxel_size == pytest.approx(model.voxel_size)
    restored_mlps = restored.mlps.state_dict()
    assert set(restored_mlps) == set(model.mlps.state_dict())
    for name, tensor in model.mlps.state_dict().items():
        assert torch.equal(restored_mlps[name], tensor), name

    # "Loaded, not trainable" has to be the WHOLE surface
    # - an unbound attribute reads back as AttributeError where the sibling Gaussians returns None
    assert set(vars(restored)) == set(vars(model))
    assert restored.param_optimizers == {}
    assert restored.mlp_optimizer is None
    assert restored.optimizers == []
    assert restored.schedulers == []
    assert restored.anchor_scheduler is None
    assert restored.offset_scheduler is None
    assert restored.mlp_scheduler is None
    assert restored.strategy is None


def test_pre_backward_retains_the_screen_space_gradient():
    """
    The accumulator reads .grad off a non-leaf, and autograd frees that unless it is retained.
    """
    model = _field(n_offsets=2)
    leaf = torch.zeros(3, 2, requires_grad=True)
    means2d = leaf * 2.0  # non-leaf, exactly like the tensor gsplat hands back in info
    upstream_grad = torch.arange(6.0).reshape(3, 2)

    model.pre_backward(0, {model.strategy.key_for_gradient: means2d})
    (means2d * upstream_grad).sum().backward()

    assert means2d.grad is not None
    assert torch.equal(means2d.grad, upstream_grad)


def test_post_backward_accumulates_the_view_it_is_given():
    """
    Without accumulation the statistics never leave their init and refinement finds nothing.
    """
    model = _field(n_offsets=2)
    _window(model, start_stat=0, update_from=10000, update_until=15000)

    model.post_backward(5, _backward_info(model, anchors=(0, 3), opacity=0.25))

    # Slots 0 and 6 are anchors 0 and 3 at offset 0; the opacity half is keyed per anchor
    strategy = model.strategy
    assert strategy.offset_denom[0] == 1
    assert strategy.offset_denom[6] == 1
    assert list(strategy.anchor_denom[:4]) == [1, 0, 0, 1]
    assert strategy.opacity_accum[0] == pytest.approx(0.25)
    assert strategy.opacity_accum[3] == pytest.approx(0.25)


def test_post_backward_accumulates_before_it_refines():
    """
    Refinement must decide on THIS step's statistics.

    - Accumulating after `refine` would make every refine read the previous step's
      window — silent, suite-green, and a whole refine behind for 30k steps.
    """
    model = _field(n_offsets=2)
    _window(model, start_stat=0, update_from=0, update_until=100, refine_every=1)
    n_before = model.n_primitives

    model.post_backward(2, _backward_info(model, anchors=(0,), opacity=1e-6))

    # One visit at ~zero opacity is exactly what pruning needs
    # - a refine that ran first would have found an empty window and kept the anchor
    assert model.n_primitives == n_before - 1


def test_post_backward_respects_the_statistics_window():
    """
    Counting outside (start_stat, update_until) pollutes the window the first refine reads.
    """
    model = _field(n_offsets=2)
    _window(model, start_stat=500, update_from=1500, update_until=15000, refine_every=100)

    model.post_backward(100, _backward_info(model, anchors=(0,), opacity=1e-6))

    assert float(model.strategy.offset_denom.sum()) == 0.0
    assert float(model.strategy.anchor_denom.sum()) == 0.0


@cuda
def test_post_backward_accumulates_a_cuda_render_inside_the_window():
    """
    The four accumulators are allocated on `device`, so a CUDA step must fold in without a move.

    - Drives the whole production path: render, retain the gradient, backward, accumulate. Every
      other accumulation test is CPU and the CUDA training test runs max_steps=3 against
      start_stat=500, so the window guard returns first. Allocating all four on "cpu" while leaving
      `self.device` intact left all 346 of tests/splats green (225abd6f).
    - `update_from` sits past the step: accumulation, not densification.
    - Seeded: `_field` draws MLP weights from the global RNG, and that draw decides whether any
      decoded Gaussian survives projection with `radii > 0`.
    """
    # Seeded: torch seeds its default generator per process
    # - failed 2 runs in 10 under `pytest <nodeid>` at 2155f673, on the `offset_denom` assertion
    # - over seeds 0-11: offset_denom non-zero at 7 of 12 (zero at 2, 3, 5, 8, 9), anchor_denom 163 at
    #   all 12; seed 0 puts 163 through the gradient half
    torch.manual_seed(0)

    model = _field(n_offsets=2, device="cuda")
    _window(model, start_stat=0, update_from=10000, update_until=15000)
    cam_to_world, intrinsics = _cam(device="cuda")

    render, info = model.render(cam_to_world, intrinsics, 64, 64, torch.tensor([0], device="cuda"))
    model.pre_backward(5, info)
    render["rgb"].sum().backward()
    model.post_backward(5, info)

    # A CPU accumulator raises in `index_add_` before this
    # - catches the variant that moves `self.device` too: nothing raises, statistics leave CUDA
    strategy = model.strategy
    assert strategy.offset_gradient_accum.device.type == "cuda"
    assert strategy.offset_denom.device.type == "cuda"
    assert strategy.opacity_accum.device.type == "cuda"
    assert strategy.anchor_denom.device.type == "cuda"

    # Not vacuous over an empty accumulation
    # - a decode of nothing, or an early window-guard return, satisfies every device assertion above
    # - anchor_denom carries it whatever the draw: 163 at every seed measured
    # - offset_denom is the gradient half, dependable only at the seed above
    assert float(strategy.offset_denom.sum()) > 0.0
    assert float(strategy.anchor_denom.sum()) > 0.0


def test_scaffold_export_gaussians_bakes_every_anchor():
    model = _field(n_offsets=2)
    _open_every_offset(model)
    cam_to_world, intrinsics = _cam_pair()

    baked = model.export_gaussians(cam_to_world, intrinsics, 64, 64)

    assert set(baked) == {"means", "scales", "quats", "opacities", "sh0", "shN"}

    # What the name claims: one row per anchor slot
    # - `> 0` also passes on the all-closed fallback, which bakes a single Gaussian for the scene
    assert len(baked["means"]) == model.n_primitives * model.cfg.n_offsets

    # The ply writer wants raw forms: log scales, logit opacities, a single SH DC band
    assert baked["sh0"].shape[1] == 1
    assert baked["shN"].shape == (len(baked["means"]), 0, 3)


def test_export_gaussians_writes_the_decoded_values_in_the_ply_s_raw_forms():
    """
    One camera that sees every anchor must bake exactly what decode emits for it.

    - The ply reader applies sigmoid and the SH DC transform on load: an opacity written raw, or an
      sh0 without the SH_C0 division, is a broken export with a green suite.
    - Only the open branch is comparable. With no offset open both paths force-keep the most opaque
      one, `decode` returns its NEGATIVE tanh opacity raw and the export clamps to 1e-4 first — the
      comparison failed on 121 of 600 unseeded draws (20.2%), exactly the fallback draws.
      `_open_every_offset` removes the branch; the clamp test below covers it by name.
    """
    model = _field(n_offsets=2)
    _open_every_offset(model)
    cam_to_world, intrinsics = _cam()
    camera_id = torch.zeros(1, dtype=torch.long)

    # Arm both halves of the mean: the offset term and the extent slice
    # - at init `scaling`'s halves are equal and `offsets` is exactly zero, so `means` comes out as
    #   `anchors` whatever multiplies them and `scaling[:, 0:3]` passes for `scaling[:, 3:6]`
    # - perturb `scaling`: arms the extent slice. perturb `offsets`: arms the offset term. Both needed
    # - the bump RAMPS across slots: a uniform one leaves every offset of an anchor identical, and a
    #   permuted decode would be indistinguishable from a correct one
    with torch.no_grad():
        model.params["scaling"][:, 3:] += 0.75
        model.params["offsets"] += torch.arange(1, model.cfg.n_offsets + 1).float().view(1, -1, 1) * 0.3
    assert bool(model.visible_anchors(cam_to_world, intrinsics, 64, 64).all())

    baked = model.export_gaussians(cam_to_world, intrinsics, 64, 64)
    decoded, _ = model.decode(model.primitive, cam_to_world, intrinsics, 64, 64, camera_id)

    # Non-vacuity, three ways the comparison below can be weakened without failing
    # - the whole slot grid, not the fallback's single row
    # - every opacity strictly inside the export's clamp, so the comparison is of the transform
    # - opacities differing across rows by more than the tolerance, or a permuted column compares equal
    assert len(decoded["means"]) == model.n_primitives * model.cfg.n_offsets
    assert bool((decoded["opacities"] > 1e-4).all())
    assert bool((decoded["opacities"] < 1 - 1e-4).all())
    assert float(decoded["opacities"].max() - decoded["opacities"].min()) > 2e-4

    assert len(baked["means"]) == len(decoded["means"])
    assert torch.allclose(baked["means"], decoded["means"], atol=1e-6)
    assert torch.allclose(baked["quats"], decoded["quats"], atol=1e-6)
    assert torch.allclose(torch.exp(baked["scales"]), decoded["scales"], rtol=1e-5)
    assert torch.allclose(torch.sigmoid(baked["opacities"]), decoded["opacities"], atol=2e-4)
    assert torch.allclose(baked["sh0"][:, 0, :] * SH_C0 + 0.5, decoded["colors"], atol=1e-6)


def test_export_gaussians_clamps_the_all_closed_opacity_decode_returns_raw():
    """
    The one branch on which the export and the decode are meant to disagree.

    - `mlp_opacity` ends in tanh, so a closed offset carries a NEGATIVE opacity; with none open both
      paths force-keep the most opaque one anyway (gsplat cannot serialize an empty set).
    - `decode` hands that value over raw. A ply cannot carry it, so the export clamps to 1e-4 first
      — logit -9.2102, not the raw value.
    - Absolute literals, not each other: both sides share `decode`'s arithmetic.
    """
    model = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()
    camera_id = torch.zeros(1, dtype=torch.long)

    # decode reads only the visible anchors
    # - its flat slot index matches one computed over every anchor only while the cloud is all in frustum
    assert bool(model.visible_anchors(cam_to_world, intrinsics, 64, 64).all())

    # Same arming, same reason, as the test above
    # - `offsets` zero and all six `scaling` columns equal at init: `means` reduces to `anchors` and the
    #   offset-extent slice passes for the gaussian one
    # - without it the four round-trip assertions hold on a decode that dropped the offset term
    # - the opacity head reads `anchor_feat` and view direction only: neither bump reopens a closed one
    with torch.no_grad():
        model.params["scaling"][:, 3:] += 0.75
        model.params["offsets"] += torch.arange(1, model.cfg.n_offsets + 1).float().view(1, -1, 1) * 0.3

    # A -3.0 final bias closes every offset on every draw, not most
    # - `anchor_feat` zeros, both Linears U(-1/sqrt(fan_in)): pre-activation bounded by 2.33
    with torch.no_grad():
        model.mlps.mlp_opacity[-2].bias.fill_(-3.0)
        direction = model.params["anchors"] - cam_to_world[0, :3, 3]
        direction = direction / direction.norm(dim=-1, keepdim=True)
        features = torch.cat([model.params["anchor_feat"], direction], dim=-1)
        neural_opacity = model.mlps(features, camera_id)[0]
    assert bool((neural_opacity <= 0).all())

    baked = model.export_gaussians(cam_to_world, intrinsics, 64, 64)
    decoded, decode_index = model.decode(model.primitive, cam_to_world, intrinsics, 64, 64, camera_id)

    # Both fall back to the single most opaque slot, and to the same one
    assert len(decoded["means"]) == 1
    assert len(baked["means"]) == 1
    assert int(decode_index[0]) == int(neural_opacity.reshape(-1).argmax())

    # The divergence itself: raw and negative out of decode, clamped and logit-ed out of the export
    assert float(decoded["opacities"][0]) == pytest.approx(float(neural_opacity.max()), abs=1e-6)
    assert float(decoded["opacities"][0]) < 0.0
    assert float(baked["opacities"][0]) == pytest.approx(-9.2102, abs=1e-3)
    assert not torch.allclose(torch.sigmoid(baked["opacities"]), decoded["opacities"], atol=2e-4)

    # Every other raw form still round-trips on that row, so the clamp is the only difference
    assert torch.allclose(baked["means"], decoded["means"], atol=1e-6)
    assert torch.allclose(baked["quats"], decoded["quats"], atol=1e-6)
    assert torch.allclose(torch.exp(baked["scales"]), decoded["scales"], rtol=1e-5)
    assert torch.allclose(baked["sh0"][:, 0, :] * SH_C0 + 0.5, decoded["colors"], atol=1e-6)


def test_decode_means_are_the_anchor_plus_its_offset_extent():
    """
    An ABSOLUTE guard on decode's means, computed here rather than compared against the export.

    - The parity assertions above share `decode`'s arithmetic on both sides, so a wrong scaling
      slice, a permuted slot or a flipped sign applied to BOTH is invisible; this recomputes the
      means straight from `params` and never calls the export.
    - The three "wrong" forms are asserted materially DIFFERENT first, or a fixture that made them
      coincide turns this into a tautology — how the parity test lost its teeth.
    - Armed open: on the all-closed fallback the comparison collapses to a single row.
    """
    model = _field(n_offsets=2)
    _open_every_offset(model)
    cam_to_world, intrinsics = _cam()
    camera_id = torch.zeros(1, dtype=torch.long)

    # Same two perturbations the export test needs, same reasons
    # - at init `offsets` is zero and `scaling`'s halves are equal: the forms are inseparable
    with torch.no_grad():
        model.params["scaling"][:, 3:] += 0.75
        model.params["offsets"] += torch.arange(1, model.cfg.n_offsets + 1).float().view(1, -1, 1) * 0.3

    decoded, decode_index = model.decode(model.primitive, cam_to_world, intrinsics, 64, 64, camera_id)

    anchors = model.params["anchors"]
    offsets = model.params["offsets"]
    scaling = torch.exp(model.params["scaling"])

    def slots(anchor_term, offset_term, extent):
        # decode_index is anchor * n_offsets + offset, so it indexes this full slot grid directly
        return (anchor_term[:, None, :] + offset_term * extent).reshape(-1, 3)[decode_index]

    expected = slots(anchors, offsets, scaling[:, None, :3])
    wrong_extent = slots(anchors, offsets, scaling[:, None, 3:6])
    rolled_slots = slots(anchors, offsets.roll(1, dims=1), scaling[:, None, :3])
    flipped_sign = slots(anchors, -offsets, scaling[:, None, :3])

    # Non-degeneracy first: each wrong form has to be reachable and distinct from the right one
    assert len(decoded["means"]) == model.n_primitives * model.cfg.n_offsets
    assert not torch.allclose(expected, wrong_extent, atol=1e-6)
    assert not torch.allclose(expected, rolled_slots, atol=1e-6)
    assert not torch.allclose(expected, flipped_sign, atol=1e-6)

    assert torch.allclose(decoded["means"], expected, atol=1e-6)


def test_export_gaussians_decodes_at_the_unit_mean_view_direction():
    """
    Every anchor bakes at the unit mean of the directions to the cameras that saw it, and an anchor
    no camera saw falls back to the nearest camera rather than dividing by a zero visit count.
    """
    model = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam_pair()

    # Anchor 0 goes far behind both cameras, so seen_count is zero for it and the fallback must fire
    with torch.no_grad():
        model.params["anchors"][0] = torch.tensor([0.0, 0.0, -100.0])
    seen = torch.ones(model.n_primitives, dtype=torch.bool)
    seen[0] = False
    for view in range(len(cam_to_world)):
        visible = model.visible_anchors(cam_to_world[view : view + 1], intrinsics[view : view + 1], 64, 64)
        assert torch.equal(visible, seen)

    # Capture what the heads are actually asked to decode
    # - export bypasses the frustum filter, so this covers every anchor including the unseen one
    captured = {}
    heads = model.mlps

    def capture(features, camera_id):
        captured["features"] = features.detach().clone()
        return heads(features, camera_id)

    model.mlps = capture
    model.export_gaussians(cam_to_world, intrinsics, 64, 64)
    directions = captured["features"][:, model.cfg.feat_dim :]

    # Dropping the unseen fallback divides by zero
    # - dropping the re-normalization leaves a short vector: the heads are scale-free only on a unit one
    assert torch.isfinite(directions).all()
    assert torch.allclose(directions.norm(dim=-1), torch.ones(len(directions)), atol=1e-5)

    anchors = model.params["anchors"].detach()
    centers = cam_to_world[:, :3, 3]
    per_camera = torch.stack([(anchors - center) / (anchors - center).norm(dim=-1, keepdim=True) for center in centers])
    mean_direction = per_camera.mean(dim=0)
    assert (mean_direction[seen].norm(dim=-1) < 0.999).all()  # re-normalizing is not a no-op here
    assert torch.allclose(
        directions[seen], mean_direction[seen] / mean_direction[seen].norm(dim=-1, keepdim=True), atol=1e-5
    )

    # The unseen anchor takes the nearest camera's direction
    to_nearest = anchors[0] - centers[int(torch.cdist(anchors[0:1], centers).argmin())]
    assert torch.allclose(directions[0], to_nearest / to_nearest.norm(), atol=1e-5)


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_scaffold_render_rasterizes_with_the_configured_primitive(primitive):
    """
    `render` reads self.primitive: a hardcoded '3dgs' rasterizes 2dgs models with the wrong kernel.
    """
    model = _field(n_offsets=2, device="cuda", run={"primitive": primitive})
    cam_to_world, intrinsics = _cam(device="cuda")

    render, info = model.render(cam_to_world, intrinsics, 64, 64, torch.tensor([0], device="cuda"))

    # The strategy's gradient key and the 2dgs-only signals both follow the configured primitive
    assert model.strategy.key_for_gradient in info
    assert ("distortion" in render) == (primitive == "2dgs")
    assert ("median_depth" in render) == (primitive == "2dgs")


@cuda
def test_scaffold_render_passes_the_extra_signal_flags_through():
    """
    The regularizers ask for normals and PGSR asks for planes; neither can be decided in here.
    """
    model = _field(n_offsets=2, device="cuda")
    cam_to_world, intrinsics = _cam(device="cuda")
    camera_id = torch.tensor([0], device="cuda")

    default, _ = model.render(cam_to_world, intrinsics, 64, 64, camera_id)
    without_normals, _ = model.render(cam_to_world, intrinsics, 64, 64, camera_id, render_normals=False)
    planar, _ = model.render(cam_to_world, intrinsics, 64, 64, camera_id, render_plane=True)

    assert {"normal", "depth_normal"} <= set(default)
    assert not {"normal", "depth_normal"} & set(without_normals)
    assert {"plane_normal", "plane_distance", "plane_depth", "plane_depth_normal"} <= set(planar)


@cuda
def test_scaffold_render_carries_per_view_state_in_info_not_on_self():
    """
    PGSR renders a neighbor view and discards its info, so per-render state cannot live on self.
    """
    model = _field(n_offsets=2, device="cuda")
    cam_to_world, intrinsics = _cam(device="cuda")
    state_before = _reachable_state(model)

    # The main view: everything post_backward and the regularizers read travels in the return values
    render, info = model.render(cam_to_world, intrinsics, 64, 64, torch.tensor([0], device="cuda"))
    assert {"decode_index", "decoded_opacities", "visible_ids"} <= set(info)
    assert {"log_scales", "opacities"} <= set(render)
    main_index = info["decode_index"].clone()
    main_visible = info["visible_ids"].clone()

    # The neighbor must see a genuinely DIFFERENT anchor set
    # - otherwise both comparisons below pass against a render that clobbered the main view's decode
    neighbor = cam_to_world.clone()
    neighbor[0, 0, 3] += 3.0
    _, neighbor_info = model.render(neighbor, intrinsics, 64, 64, torch.tensor([1], device="cuda"))
    assert not torch.equal(neighbor_info["visible_ids"], main_visible)

    # post_backward is handed the MAIN info, so it must still describe the main view's decode
    assert torch.equal(info["decode_index"], main_index)
    assert torch.equal(info["visible_ids"], main_visible)

    # Values, not just names
    # - a decode cached on the strategy's accumulators leaves vars(model) untouched
    state_after = _reachable_state(model)
    assert set(state_after) == set(state_before)
    for path, value in state_before.items():
        if torch.is_tensor(value):
            assert torch.equal(state_after[path], value), path
        else:
            assert state_after[path] == value, path


########################################
# Strategy shape
########################################


def test_anchor_strategy_owns_its_state():
    """
    The four accumulators are attributes on the strategy, not a dict threaded through every call.
    """
    model = _field(n_offsets=2)
    strategy = model.strategy

    assert isinstance(strategy.offset_gradient_accum, torch.Tensor)
    assert isinstance(strategy.offset_denom, torch.Tensor)
    assert isinstance(strategy.opacity_accum, torch.Tensor)
    assert isinstance(strategy.anchor_denom, torch.Tensor)

    # Sized at construction, not lazily on the first accumulate
    # - a strategy that allocated late reads as owning nothing until a step had run
    assert len(strategy.opacity_accum) == model.n_primitives
    assert len(strategy.anchor_denom) == model.n_primitives
    assert len(strategy.offset_gradient_accum) == model.n_primitives * model.cfg.n_offsets
    assert len(strategy.offset_denom) == model.n_primitives * model.cfg.n_offsets

    # The dict they replaced is gone from the model, not merely unused
    assert not hasattr(model, "strategy_state")


def test_anchor_strategy_stores_no_primitive_of_its_own():
    """
    The constructor argument feeds `key_for_gradient` and nothing else; storing it is dead state.

    - `Scaffold.primitive`: a different object that stays — `decode` and `render` both read it
    - measured at 225abd6f: deleting `self.primitive = primitive` left all 346 of tests/splats green
    """
    strategy = _field(n_offsets=2).strategy

    assert not hasattr(strategy, "primitive")


def test_anchor_strategy_calls_do_not_take_a_state_argument():
    """
    The signatures are the contract: a lingering `state` parameter is the refactor half-done.
    """
    model = _field(n_offsets=2)
    strategy = model.strategy

    assert list(inspect.signature(strategy.accumulate).parameters) == ["step", "info"]
    assert list(inspect.signature(strategy.refine).parameters) == ["scaffold", "step"]
    assert list(inspect.signature(strategy.grow).parameters) == ["scaffold"]
    assert list(inspect.signature(strategy.prune).parameters) == ["scaffold", "scale_cap"]

    # The gsplat lifecycle hooks this strategy no longer implements
    assert not hasattr(strategy, "initialize_state")
    assert not hasattr(strategy, "step_post_backward")
    assert not hasattr(strategy, "check_sanity")


def test_anchor_strategy_has_no_abstract_base():
    """
    It inherited gsplat's `Strategy` only for `check_sanity`, which nothing in this codebase calls.
    """
    assert AnchorStrategy.__bases__ == (object,)

    # The optimizer-surgery helper from gsplat.strategy.ops stays; only the base class goes
    source = Path(scaffold_module.__file__).read_text()
    assert "gsplat.strategy.base" not in source
    assert "class AnchorStrategy:" in source


def _identifiers(source):
    """
    Every NAME token in `source`: the identifiers and keywords its code actually spells.

    - Tokenized, not searched: a retired name is also the natural word for prose about its
      retirement, and a comment or string that mentions one binds nothing.

    Args:
        source: module source text.

    Returns:
        Set of NAME token strings.
    """
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return {token.string for token in tokens if token.type == tokenize.NAME}


def test_the_retired_constants_and_the_verbosity_flag_are_absent_by_name():
    """
    The three names that were removed must not come back, at module level or inside a function.

    - Kept alongside the AST check below, which sees less: an identifier anywhere reaches a constant
      reintroduced INSIDE a function, and `verbose` as a parameter or attribute. `VIEW_DIM = 3`
      planted in `AnchorStrategy.__init__` is 1 failed, this test alone.
    - By name, not by substring: a bare COMMENT naming `VIEW_DIM` turned the substring form red
      (1 failed / 359 at 2155f673), a false red over prose that binds nothing.
    """
    identifiers = _identifiers(Path(scaffold_module.__file__).read_text())

    assert "VIEW_DIM" not in identifiers
    assert "SCALE_CAP" not in identifiers
    assert "verbose" not in identifiers


@pytest.mark.parametrize(
    ("planted", "name"),
    (
        ("VIEW_DIM = 3", "VIEW_DIM"),
        ("def grow(self, scaffold):\n    VIEW_DIM = 3\n", "VIEW_DIM"),
        ("def prune(self, scaffold, verbose=False):\n    return verbose\n", "verbose"),
        ("self.verbose = False", "verbose"),
    ),
)
def test_the_by_name_check_still_sees_a_planted_identifier(planted, name):
    # One case per place the check claims to reach, so a token scan cannot be quietly vacuous
    # - module-level binding, name bound inside a function body (invisible to the AST walk below),
    #   keyword parameter, instance attribute
    assert name in _identifiers(planted)


@pytest.mark.parametrize(
    ("planted", "name"),
    (
        ("# VIEW_DIM was retired; its literal 3 now lives on the head that reads it", "VIEW_DIM"),
        ('"""SCALE_CAP is a keyword on prune now, not a module constant."""', "SCALE_CAP"),
        ('logger.info("verbose tracing was dropped")', "verbose"),
    ),
)
def test_the_by_name_check_ignores_a_name_that_is_only_prose(planted, name):
    # The direction that was live: every case names a retired constant and binds nothing
    # - the substring form this replaced fired on the first of them (1 failed / 359 at 2155f673)
    # - the scan drops COMMENT and STRING; both string spellings are here, a docstring and a logged
    #   message
    assert name not in _identifiers(planted)


# Derived by AST from scaffold.py: one name bound outside its defs and classes
# - ground rule 9 puts every tunable on the function that reads it
# - anything else appearing here is a constant that grew back
SCAFFOLD_MODULE_BINDINGS = {"logger"}


def _module_level_bindings(source):
    """
    Every name bound by an assignment statement outside any function or class in `source`.

    - Assignment targets only, descending into the BODIES of module-level compound statements. A
      `with ... as X`, a `for` target, a walrus and an `import m as X` are unreported: none is a way
      a tunable comes back, and imports are not module state.

    Args:
        source: module source text.

    Returns:
        Set of bound names.
    """
    statements = list(ast.parse(source).body)
    names = set()
    while statements:
        node = statements.pop()

        # A def or a class opens its own scope; what it binds is not module state
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        # Both assignment forms bind, and a target can be a tuple of several names
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            names.update(child.id for child in ast.walk(target) if isinstance(child, ast.Name))

        # A module-level if / try / with still runs at import
        # - walk its bodies only, never a `with`'s own `as` target
        for field in ("body", "orelse", "finalbody", "handlers"):
            statements.extend(getattr(node, field, None) or [])
    return names


def test_the_scaffold_module_binds_no_module_level_constants():
    """
    The general property the three names above are only instances of.

    - An unused `GROWTH_EPS = 1e-6` at module level left all 346 of tests/splats green (225abd6f):
      the by-name check only knows the three names it was given.
    - Equality, not a subset: it also fails if the walk stops seeing `logger`.
    """
    bindings = _module_level_bindings(Path(scaffold_module.__file__).read_text())

    unexpected = sorted(bindings - SCAFFOLD_MODULE_BINDINGS)
    assert bindings == SCAFFOLD_MODULE_BINDINGS, f"unexpected module-level constants: {unexpected}"


@pytest.mark.parametrize(
    "planted",
    (
        "GROWTH_EPS = 1e-6",
        "GROWTH_EPS: float = 1e-6",
        "GROWTH_EPS, OTHER_EPS = 1e-6, 1e-9",
        "try:\n    GROWTH_EPS = 1e-6\nexcept ImportError:\n    GROWTH_EPS = 0.0",
    ),
)
def test_the_module_level_walk_sees_a_constant_replanted_under_a_new_name(planted):
    # One case per SPELLING, so each clause of the walk is load-bearing
    # - the check above passes trivially against a walk blind to the form a constant returns in, and a
    #   union of plants hides a lost clause
    # - plain form: ast.Assign. annotated: ast.AnnAssign (also missed by `^[A-Z_]+ *= *`)
    # - tuple form: walking INTO a target rather than reading `target.id`
    # - `try` form: the descent into a module-level compound statement
    assert "GROWTH_EPS" in _module_level_bindings(planted)


def test_anchor_growth_still_adds_anchors(monkeypatch):
    """
    Moving the state must not change what densification does — only where the state lives.

    - Drives the whole path the trainer drives: `refine`, on a real step, off the accumulators the
      strategy now owns. The narrower `grow` tests above call `grow` directly.
    """
    model = _field(n_offsets=2)
    strategy = model.strategy
    before = model.n_primitives

    # Upstream thins candidates at random per level; keep them all so one hot slot is a fixed outcome
    monkeypatch.setattr(torch, "rand_like", torch.ones_like)

    # `offsets` initialize to zero
    # - every candidate lands in its anchor's occupied voxel: growing no-ops for unrelated reasons
    with torch.no_grad():
        model.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])

    # Force the slot over the growth threshold, then refine on a step inside the window
    # - offset_denom must clear refine_every * success_threshold * 0.5, or `grow` reads the slot as
    #   unseen and ignores its gradient however large
    cfg = strategy.cfg
    step = (cfg.update_from // cfg.refine_every + 1) * cfg.refine_every
    strategy.offset_gradient_accum[0] = 100.0
    strategy.offset_denom[0] = float(cfg.refine_every)
    strategy.refine(model, step)

    # Strictly more, not ">=": a refine that grew nothing at all would pass the weaker form
    assert model.n_primitives > before
