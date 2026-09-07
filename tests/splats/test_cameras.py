"""
CameraOpt: pose deltas, color affine, half selection, and 6D rotations.
"""

import pytest
import torch

from collab_splats.splats.cameras import CameraOpt, rotation_6d_to_matrix
from collab_splats.splats.trainer import SplatsConfig


def test_rotation_6d_gives_proper_rotations():
    torch.manual_seed(0)
    rotations = rotation_6d_to_matrix(torch.randn(5, 6))
    identity = torch.eye(3).expand(5, 3, 3)
    gram = rotations @ rotations.transpose(-1, -2)
    determinants = torch.linalg.det(rotations)
    assert torch.allclose(gram, identity, atol=1e-5)
    assert torch.allclose(determinants, torch.ones(5), atol=1e-5)


def test_construction_is_the_identity_on_both_halves():
    # Embedding defaults to N(0, 1) — the constructor must zero every weight itself
    module = CameraOpt(3, optimize_pose=True, optimize_appearance=True)
    cam_to_world = torch.eye(4).expand(3, 4, 4).clone()
    cam_to_world[:, :3, 3] = torch.arange(3).float()[:, None]
    rgb = torch.rand(3, 4, 4, 3)
    ids = torch.arange(3)

    assert torch.allclose(module.camera(cam_to_world, ids), cam_to_world)
    assert torch.allclose(module.color(rgb, ids), rgb)


def test_camera_applies_the_pose_delta():
    module = CameraOpt(2)
    with torch.no_grad():
        module.translation.weight[1] = torch.tensor([0.1, -0.2, 0.3])
    # 90 degrees about z: under an identity rotation the two composition orders are the same matrix
    cam_to_world = torch.eye(4).expand(2, 4, 4).clone()
    cam_to_world[:, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    refined = module.camera(cam_to_world, torch.arange(2))

    # View 0 has no delta, view 1 is translated in its own camera frame
    # - R @ [0.1, -0.2, 0.3] = [0.2, 0.1, 0.3]; composing on the left leaves the delta unrotated
    assert torch.allclose(refined[0], cam_to_world[0])
    assert torch.allclose(refined[1, :3, 3], torch.tensor([0.2, 0.1, 0.3]))


def test_camera_matches_the_module_dtype():
    module = CameraOpt(2).double()
    with torch.no_grad():
        module.translation.weight.normal_(std=0.1)
    cam_to_world = torch.eye(4, dtype=torch.float64).expand(2, 4, 4).clone()

    assert module.camera(cam_to_world, torch.arange(2)).dtype == torch.float64


def test_color_applies_per_image_gain_and_bias():
    module = CameraOpt(2, optimize_pose=False, optimize_appearance=True)
    with torch.no_grad():
        module.appearance.weight[1] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    rgb = torch.ones(1, 1, 1, 3)

    out = module.color(rgb, torch.tensor([1]))

    # Channel 0 gain 1 + 1 = 2, channel 1 bias +0.5, channel 2 untouched
    assert out[0, 0, 0].tolist() == pytest.approx([2.0, 1.5, 1.0])


def test_both_halves_pass_through_when_neither_is_selected():
    module = CameraOpt(1, optimize_pose=False, optimize_appearance=False)
    cam_to_world = torch.eye(4)[None]
    rgb = torch.rand(1, 2, 2, 3)
    ids = torch.tensor([0])

    # Object identity, not just value equality — nothing is copied on a disabled half
    assert module.camera(cam_to_world, ids) is cam_to_world
    assert module.color(rgb, ids) is rgb


def test_denormalize_rescales_translation_deltas_only():
    module = CameraOpt(2)
    with torch.no_grad():
        module.translation.weight.fill_(2.0)
        module.rotation.weight.fill_(3.0)

    module.denormalize(scale=0.5)

    assert torch.allclose(module.translation.weight, torch.full((2, 3), 4.0))
    assert torch.allclose(module.rotation.weight, torch.full((2, 6), 3.0))


def test_denormalize_is_a_no_op_without_the_pose_half():
    module = CameraOpt(2, optimize_pose=False, optimize_appearance=True)
    with torch.no_grad():
        module.appearance.weight.normal_(std=0.1)
    before = module.appearance.weight.clone()

    module.denormalize(scale=0.5)

    # Scene scale is a pose-side unit only — the color affine must come through bit-identical
    assert torch.equal(module.appearance.weight, before)


def test_from_config_builds_pose_only_when_appearance_is_off():
    cfg = SplatsConfig.from_dict({"pose_opt": True, "appearance_opt": False})

    module = CameraOpt.from_config(cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu")

    assert module.translation is not None and module.rotation is not None
    assert module.appearance is None
    # One optimizer and one scheduler for the pose half, none for appearance
    assert len(module.optimizers) == 1
    assert len(module.schedulers) == 1


def test_from_config_builds_appearance_only_when_pose_is_off():
    cfg = SplatsConfig.from_dict({"pose_opt": False, "appearance_opt": True})

    module = CameraOpt.from_config(cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu")

    # The two halves are selected independently — appearance on with pose off is a legal run
    assert module.translation is None and module.rotation is None
    assert module.appearance is not None
    assert len(module.optimizers) == 1
    assert len(module.schedulers) == 1


def test_from_config_builds_both_when_both_are_on():
    cfg = SplatsConfig.from_dict({"pose_opt": True, "appearance_opt": True})

    module = CameraOpt.from_config(cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu")

    assert module.rotation is not None and module.appearance is not None
    assert len(module.optimizers) == 2
    assert len(module.schedulers) == 2


def test_from_config_builds_nothing_when_both_are_off():
    cfg = SplatsConfig.from_dict({"pose_opt": False, "appearance_opt": False})

    module = CameraOpt.from_config(cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu")

    assert module.translation is None and module.rotation is None and module.appearance is None
    assert module.optimizers == [] and module.schedulers == []


def test_from_config_scales_the_two_pose_learning_rates_differently():
    cfg = SplatsConfig.from_dict({"pose_opt": True})

    module = CameraOpt.from_config(cfg, n_views=4, world_extent=8.0, scene_scale=2.0, lr_gamma=0.999, device="cpu")

    rotation_group, translation_group = module.optimizers[0].param_groups
    # Rotation follows the world extent (unit-free), translation follows the training frame
    assert rotation_group["lr"] == pytest.approx(cfg.pose_lr * 8.0)
    assert translation_group["lr"] == pytest.approx(cfg.pose_lr * 2.0)


########################################
# Ground Rule 9 keyword-only defaults
########################################


def test_camera_opt_half_selection_is_keyword_only():
    # The bare `*` is the point: a second positional would silently become `optimize_pose`
    # - match the full count string: any arity slip in this call also says "positional"
    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
        CameraOpt(3, True)

    module = CameraOpt(3, optimize_pose=False, optimize_appearance=True)
    assert module.has_appearance and not module.has_pose


def test_from_config_weight_decay_is_keyword_only():
    cfg = SplatsConfig.from_dict({"pose_opt": True})

    # `device` is the sixth and last positional; a seventh would silently become `weight_decay`
    with pytest.raises(TypeError, match="takes 7 positional arguments but 8 were given"):
        CameraOpt.from_config(cfg, 4, 2.0, 1.0, 0.999, "cpu", 1e-5)

    module = CameraOpt.from_config(
        cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu", weight_decay=1e-3
    )
    assert module.optimizers[0].param_groups[0]["weight_decay"] == pytest.approx(1e-3)
