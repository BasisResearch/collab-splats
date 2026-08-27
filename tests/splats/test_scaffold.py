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
    with pytest.raises(ValueError, match="sh_degree"):
        SplatsConfig.from_dict({"representation": "scaffold", "sh_degree": 3})


########################################
# MLP heads
########################################


def test_mlp_heads_emit_per_offset_outputs():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    cfg = ScaffoldConfig(n_offsets=4, feat_dim=8)
    mlps = ScaffoldMLPs(cfg)
    features = torch.zeros(6, cfg.feat_dim + 4)  # feat + view dir (3) + view distance (1)
    opacity, cov, colour = mlps(features, camera_id=None)
    assert opacity.shape == (6, 4)
    assert cov.shape == (6, 4 * 7)
    assert colour.shape == (6, 4 * 3)
    assert opacity.min() >= -1.0 and opacity.max() <= 1.0  # tanh
    assert colour.min() >= 0.0 and colour.max() <= 1.0  # sigmoid


def test_appearance_embedding_changes_colour_only_when_enabled():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    features = torch.zeros(3, 8 + 4)
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


def _field(n_offsets=4, appearance_dim=0, n_views=3, device="cpu"):
    from collab_splats.splats.scaffold import AnchorField

    cfg = ScaffoldConfig(n_offsets=n_offsets, feat_dim=8, appearance_dim=appearance_dim)
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

    # Force every opacity negative: the tanh head's last linear bias dominates the feature input
    with torch.no_grad():
        field.mlps.mlp_opacity[-2].weight.zero_()
        field.mlps.mlp_opacity[-2].bias.fill_(-5.0)
    decoded, index = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    assert len(decoded["means"]) == 0
    assert len(index) == 0


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
    render, info = render_gaussians(
        primitive, decoded, cam_to_world, intrinsics, 64, 64, sh_degree=None, absgrad=False
    )
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
