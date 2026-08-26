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
    ViewSampler,
    compute_scene_scale,
    downscale_factor,
    downscale_view,
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


@pytest.mark.parametrize(
    "spec",
    [
        {"weight": 0.01, "end": 100},  # end without end_weight
        {"weight": 0.01, "start": 100, "end": 100, "end_weight": 0.001},  # end <= start
        {"weight": 0.0, "end": 100, "end_weight": 0.001},  # log-linear needs positive endpoints
        {"weight": 0.01, "end": 100, "end_weight": 0.0},
    ],
)
def test_config_rejects_bad_decay(spec):
    with pytest.raises(ValueError, match="splats.losses.depth"):
        SplatsConfig.from_dict({"losses": {"depth": spec}})


def test_config_accepts_decay():
    cfg = SplatsConfig.from_dict({"losses": {"depth": {"weight": 0.01, "end": 100, "end_weight": 0.001}}})
    assert cfg.losses["depth"]["end_weight"] == 0.001


def test_config_rejects_zero_steps():
    with pytest.raises(ValueError, match="max_steps"):
        SplatsConfig(max_steps=0)


def test_scene_scale_is_max_camera_spread_times_margin():
    cam_to_world = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    cam_to_world[:, 0, 3] = [-1.0, 0.0, 1.0]
    scene_scale = compute_scene_scale(torch.from_numpy(cam_to_world))
    assert scene_scale == pytest.approx(1.1)


def test_strategy_follows_primitive():
    mcmc = make_strategy(SplatsConfig(primitive="3dgs", cap_max=1234), n_views=300)
    default = make_strategy(SplatsConfig(primitive="2dgs"), n_views=300)
    assert isinstance(mcmc, MCMCStrategy) and mcmc.cap_max == 1234
    assert isinstance(default, DefaultStrategy) and default.key_for_gradient == "gradient_2dgs"


def test_make_strategy_2dgs_splatfacto_args():
    cfg = SplatsConfig(primitive="2dgs")
    strategy = make_strategy(cfg, n_views=300)

    assert isinstance(strategy, DefaultStrategy)
    # splatfacto (nerfstudio @ 50e0e3c) non-default args
    assert strategy.prune_opa == 0.1
    assert strategy.prune_scale3d == 0.5
    assert strategy.refine_scale2d_stop_iter == 4000
    assert strategy.pause_refine_after_reset == 400  # n_views + 100
    # Measured-good pair kept (absgrad unusable on gradient_2dgs — see spec)
    assert strategy.absgrad is False
    assert strategy.grow_grad2d == pytest.approx(2e-4)
    assert strategy.key_for_gradient == "gradient_2dgs"


def test_make_strategy_2dgs_pause_capped_for_many_views():
    # n_views + 100 >= reset_every would gate refine off forever; cap keeps it reachable
    strategy = make_strategy(SplatsConfig(primitive="2dgs"), n_views=3000)
    assert strategy.pause_refine_after_reset == strategy.reset_every - strategy.refine_every
    assert strategy.pause_refine_after_reset < strategy.reset_every


def test_make_strategy_3dgs_untouched():
    strategy = make_strategy(SplatsConfig(primitive="3dgs"), n_views=300)
    assert isinstance(strategy, MCMCStrategy)
    assert strategy.cap_max == 1_000_000


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
    monkeypatch.setattr(trainer_module, "make_strategy", lambda cfg, n_views: strategy)
    cfg = SplatsConfig(primitive="3dgs", max_steps=5, log_every=1, means_lr=1e-2)
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)
    _assert_trained(calls, points)

    # MCMC noise moves means even without an optimizer step; sh0 only moves through the optimizer
    sh0 = calls[0]["gaussians"]["sh0"].detach().cpu()
    seed_sh0 = (torch.from_numpy(colors).float() / 255.0 - 0.5) / SH_DC_NORMALISER
    assert not torch.allclose(sh0[: len(points), 0, :], seed_sh0)


def test_view_sampler_covers_every_view_once_per_epoch():
    sampler = ViewSampler(7, seed=42)
    epoch1 = [sampler.next() for _ in range(7)]
    epoch2 = [sampler.next() for _ in range(7)]

    assert sorted(epoch1) == list(range(7))
    assert sorted(epoch2) == list(range(7))
    # Reshuffle across epochs: identical order for 7 views has p = 1/5040
    assert epoch1 != epoch2


def test_view_sampler_deterministic_for_seed():
    a = ViewSampler(5, seed=42).next()
    runs = [[ViewSampler(5, seed=42).next() for _ in range(15)] for _ in range(2)]
    assert runs[0] == runs[1]
    assert a == runs[0][0]


def test_downscale_factor_boundaries():
    # splatfacto defaults: num_downscales=2, resolution_schedule=3000
    assert downscale_factor(0, 2, 3000) == 4
    assert downscale_factor(2999, 2, 3000) == 4
    assert downscale_factor(3000, 2, 3000) == 2
    assert downscale_factor(5999, 2, 3000) == 2
    assert downscale_factor(6000, 2, 3000) == 1
    assert downscale_factor(29999, 2, 3000) == 1
    # 0 disables the schedule entirely
    assert downscale_factor(0, 0, 3000) == 1


def test_downscale_view_scales_image_and_k():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    K = torch.tensor([[[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]])

    small, K_small = downscale_view(image, K, 4)
    assert small.shape == (120, 160, 3)
    assert torch.allclose(K_small[0, 0, 0], torch.tensor(125.0))
    assert torch.allclose(K_small[0, 0, 2], torch.tensor(80.0))
    assert torch.allclose(K_small[0, 2, 2], torch.tensor(1.0))

    same, K_same = downscale_view(image, K, 1)
    assert same is image and K_same is K


def test_splats_config_accepts_downscale_fields():
    cfg = SplatsConfig.from_dict({"enabled": True, "num_downscales": 1, "resolution_schedule": 100})
    assert cfg.num_downscales == 1 and cfg.resolution_schedule == 100
    # Defaults are splatfacto's
    default = SplatsConfig()
    assert default.num_downscales == 2 and default.resolution_schedule == 3000


def test_scene_normalization_centres_and_unit_cubes_cameras():
    cam_to_world = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    cam_to_world[:, :3, 3] = [[0, 0, 0], [4, 0, 0], [2, 6, -2]]
    center, scale = trainer_module.scene_normalization(cam_to_world)

    # Centre = mean position; scale = 1 / max |coord - centre| (L-inf, splatfacto auto_scale_poses)
    np.testing.assert_allclose(center, [2, 2, -2 / 3], rtol=1e-6)
    normalised = (cam_to_world[:, :3, 3] - center) * scale
    assert np.isclose(np.abs(normalised).max(), 1.0)
    with pytest.raises(ValueError, match="coincide"):
        trainer_module.scene_normalization(np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)))


def test_denormalize_outputs_round_trips_gaussians_cameras_and_pose_deltas():
    from collab_splats.splats.cameras import CameraOptModule

    rng = np.random.default_rng(0)
    cam_to_world = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    cam_to_world[:, :3, 3] = rng.normal(size=(4, 3)) * 5 + 10
    center, scale = trainer_module.scene_normalization(cam_to_world)

    # Gaussians and cameras in the normalised frame; refiner with a known camera-frame translation delta
    world_means = rng.normal(size=(20, 3)).astype(np.float32) * 5 + 10
    world_log_scales = np.log(rng.uniform(0.1, 2, size=(20, 3)).astype(np.float32))
    gaussians = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy((world_means - center) * scale)),
            "scales": torch.nn.Parameter(torch.from_numpy(world_log_scales + np.log(scale))),
        }
    )
    normalised_cams = torch.from_numpy(cam_to_world.copy())
    normalised_cams[:, :3, 3] = (normalised_cams[:, :3, 3] - torch.from_numpy(center)) * scale
    refiner = CameraOptModule(4)
    refiner.zero_init()
    with torch.no_grad():
        refiner.translation.weight[1] = torch.tensor([0.1, -0.2, 0.3])
    refined_normalised = refiner(normalised_cams[1:2], torch.tensor([1]))[0]

    trainer_module.denormalize_outputs(gaussians, refiner, normalised_cams, center, scale)

    # Back in world units; the refined pose maps through the same Sim3 as the raw one
    np.testing.assert_allclose(gaussians["means"].detach().numpy(), world_means, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gaussians["scales"].detach().numpy(), world_log_scales, atol=1e-5)
    np.testing.assert_allclose(normalised_cams[:, :3, 3].numpy(), cam_to_world[:, :3, 3], rtol=1e-4, atol=1e-4)
    refined_world = refiner(normalised_cams[1:2], torch.tensor([1]))[0]
    expected_t = refined_normalised[:3, 3] / scale + torch.from_numpy(center)
    np.testing.assert_allclose(refined_world[:3, 3].detach().numpy(), expected_t.detach().numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        refined_world[:3, :3].detach().numpy(), refined_normalised[:3, :3].detach().numpy(), atol=1e-6
    )


def test_config_normalize_scene_default_off_and_settable():
    assert SplatsConfig().normalize_scene is False
    assert SplatsConfig.from_dict({"enabled": True, "normalize_scene": True}).normalize_scene is True


def test_pose_refiner_rotation_lr_follows_world_extent_translation_lr_follows_frame():
    # Normalised scene: rotation keeps the world-extent lr, translation gets the unit-cube lr
    cfg = SplatsConfig(pose_lr=1e-5, max_steps=100)
    refiner, optimizer, _ = trainer_module.make_pose_refiner(
        cfg, n_views=4, rotation_lr_scale=78.65, translation_lr_scale=1.0, lr_gamma=0.99, device="cpu"
    )
    lrs = {id(group["params"][0]): group["lr"] for group in optimizer.param_groups}
    assert lrs[id(refiner.rotation.weight)] == pytest.approx(1e-5 * 78.65)
    assert lrs[id(refiner.translation.weight)] == pytest.approx(1e-5)
