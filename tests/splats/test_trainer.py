"""
SplatsConfig validation, scene scale, Gaussian init, strategy choice and training-target preparation.
"""

import numpy as np
import pytest
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy

import collab_splats.splats.trainer as trainer_module
from collab_splats.splats.trainer import (
    SH_DC_NORMALISER,
    SplatsConfig,
    compute_scene_scale,
    init_gaussians_from_points,
    make_strategy,
    prepare_training_target,
    train,
)
from tests.splats.synthetic import make_scene

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def test_config_from_dict_keeps_given_values_and_defaults():
    block = {"enabled": True, "primitive": "2dgs", "max_steps": 10, "losses": {"depth": {"weight": 0.1}}}
    cfg = SplatsConfig.from_dict(block)
    assert (cfg.primitive, cfg.max_steps, cfg.pose_opt, cfg.sh_degree) == ("2dgs", 10, True, 3)
    assert cfg.losses == {"depth": {"weight": 0.1}}


def test_default_losses_match_primitive():
    losses_3dgs = SplatsConfig(primitive="3dgs").losses
    losses_2dgs = SplatsConfig(primitive="2dgs").losses
    assert {"opacity_reg", "scale_reg"} <= set(losses_3dgs)
    assert "distortion" in losses_2dgs and "opacity_reg" not in losses_2dgs


@pytest.mark.parametrize(
    "bad",
    [
        {"primitive": "4dgs"},
        {"losses": {"tv": {"weight": 1.0}}},
        {"losses": {"depth": {"weight": 1.0, "stop": 5}}},
        {"losses": {"depth": {"start": 5}}},
        {"primitive": "3dgs", "losses": {"distortion": {"weight": 0.1}}},
        {"unknown_key": 1},
    ],
)
def test_config_rejects_invalid(bad):
    with pytest.raises(ValueError):
        SplatsConfig.from_dict(bad)


def test_config_rejects_zero_steps():
    with pytest.raises(ValueError, match="max_steps"):
        SplatsConfig(max_steps=0)


def test_scene_scale_is_max_camera_spread_times_margin():
    cam_to_world = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    cam_to_world[:, 0, 3] = [-1.0, 0.0, 1.0]
    scene_scale = compute_scene_scale(torch.from_numpy(cam_to_world))
    assert scene_scale == pytest.approx(1.1)


def test_strategy_follows_primitive():
    mcmc = make_strategy(SplatsConfig(primitive="3dgs", cap_max=1234))
    default = make_strategy(SplatsConfig(primitive="2dgs"))
    assert isinstance(mcmc, MCMCStrategy) and mcmc.cap_max == 1234
    assert isinstance(default, DefaultStrategy) and default.key_for_gradient == "gradient_2dgs"


@cuda
def test_init_gaussians_shapes_and_optimizers():
    points = np.random.default_rng(0).uniform(-1, 1, (200, 3)).astype(np.float32)
    colors = np.full((200, 3), 128, np.uint8)
    cfg = SplatsConfig()
    gaussians, optimizers = init_gaussians_from_points(cfg, points, colors, scene_scale=1.0, device="cuda")
    assert gaussians["means"].shape == (200, 3) and gaussians["sh0"].shape == (200, 1, 3)
    assert gaussians["shN"].shape == (200, 15, 3) and gaussians["opacities"].shape == (200,)

    opacities = torch.sigmoid(gaussians["opacities"])
    expected_opacity = torch.full((200,), cfg.init_opacity, device="cuda")
    assert torch.allclose(opacities, expected_opacity)
    assert set(optimizers) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    means_lr = optimizers["means"].param_groups[0]["lr"]
    assert means_lr == pytest.approx(cfg.means_lr)


def test_prepare_training_target_scales_rgb_and_resizes_depth():
    image = np.full((8, 8, 3), 255, np.uint8)
    depth = np.array([[1.0, 0.0], [2.0, 3.0]], np.float32)
    target = prepare_training_target(image, depth, device="cpu")
    assert target["rgb"].shape == (1, 8, 8, 3) and target["rgb"].max() == 1.0
    assert target["depth"].shape == (1, 8, 8, 1)
    assert (
        target["depth"][0, 0, 0, 0] == 1.0 and target["depth"][0, 0, 7, 0] == 0.0 and target["depth"][0, 7, 7, 0] == 3.0
    )

    no_depth = prepare_training_target(image, None, device="cpu")
    assert no_depth["depth"] is None


########################################
# Short training runs pin the loop order (optimizer step before strategy post_backward)
########################################


def _recorder(monkeypatch):
    """
    Swap write_splat_outputs for a recorder; returns the list of captured call kwargs.
    """
    calls = []

    def record(cfg, gaussians, pose_refiner, images, cam_to_world, intrinsics, out_dir, train_seconds, loss_values):
        calls.append(
            {
                "gaussians": gaussians,
                "pose_refiner": pose_refiner,
                "train_seconds": train_seconds,
                "loss_values": loss_values,
            }
        )

    monkeypatch.setattr(trainer_module, "write_splat_outputs", record)
    return calls


def _assert_trained(calls, points):
    assert len(calls) == 1
    call = calls[0]
    assert call["train_seconds"] > 0
    assert {"l1", "ssim"} <= set(call["loss_values"])
    means = call["gaussians"]["means"].detach().cpu()
    seed = torch.from_numpy(points)
    assert means.shape[0] >= len(points)
    assert not torch.allclose(means[: len(points)], seed)


@cuda
@pytest.mark.parametrize("primitive, pose_opt", [("3dgs", False), ("2dgs", False), ("3dgs", True)])
def test_train_short_run_moves_gaussians(monkeypatch, tmp_path, primitive, pose_opt):
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    calls = _recorder(monkeypatch)
    cfg = SplatsConfig(primitive=primitive, pose_opt=pose_opt, max_steps=5, log_every=1, means_lr=1e-2)
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)
    _assert_trained(calls, points)
    assert (calls[0]["pose_refiner"] is not None) == pose_opt


@cuda
def test_train_refining_every_step_still_moves_gaussians(monkeypatch, tmp_path):
    # Refine on EVERY step (start=-1 since the gate is step > start): with post_backward before optimizer.step,
    # rebuilt params would carry .grad=None and never be optimized
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    calls = _recorder(monkeypatch)
    strategy = MCMCStrategy(cap_max=300, refine_start_iter=-1, refine_every=1, verbose=False)
    monkeypatch.setattr(trainer_module, "make_strategy", lambda cfg: strategy)
    cfg = SplatsConfig(primitive="3dgs", max_steps=5, log_every=1, means_lr=1e-2)
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)
    _assert_trained(calls, points)

    # MCMC noise moves means even without an optimizer step; sh0 only moves through the optimizer
    sh0 = calls[0]["gaussians"]["sh0"].detach().cpu()
    seed_sh0 = (torch.from_numpy(colors).float() / 255.0 - 0.5) / SH_DC_NORMALISER
    assert not torch.allclose(sh0[: len(points), 0, :], seed_sh0)
