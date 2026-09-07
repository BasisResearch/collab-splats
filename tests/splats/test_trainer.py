"""
SplatsConfig validation and the training loop.
"""

import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gsplat.strategy import MCMCStrategy

import collab_splats.splats.gaussian as gaussian_module
import collab_splats.splats.trainer as trainer_module
from collab_splats.splats.gaussian import SH_C0
from collab_splats.splats.trainer import (
    MODEL_CLASSES,
    REPRESENTATIONS,
    SplatsConfig,
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


def test_config_densification_defaults_are_the_measured_good_ones():
    # grow_grad2d 2e-4 is load-bearing: 8e-4 (4x gsplat's default) choked 2dgs to PSNR 17.95
    cfg = SplatsConfig()
    assert cfg.grow_grad2d == pytest.approx(2e-4)
    assert cfg.cap_max == 1_000_000


def test_config_rejects_an_unknown_representation():
    # Nothing catches a WIDENED allow-list
    # - narrowing it breaks 61 other tests incidentally; widening breaks none
    # - a rejection test cannot see it either: a third name still rejects "octree"
    # - so pin the allow-list to the two literals it may hold
    assert set(REPRESENTATIONS) == {"vanilla", "scaffold"}
    assert set(MODEL_CLASSES) == set(REPRESENTATIONS)

    with pytest.raises(ValueError, match="representation must be one of"):
        SplatsConfig.from_dict({"representation": "octree"})


def test_config_rejects_zero_steps():
    with pytest.raises(ValueError, match="max_steps"):
        SplatsConfig(max_steps=0)


########################################
# Short training runs pin the loop order (optimizer step before strategy post_backward)
########################################


def _recorder(monkeypatch):
    """
    Swap write_outputs for a recorder; returns the list of captured calls.
    """
    calls = []

    def record(
        cfg,
        model,
        refine,
        images,
        image_ids,
        cam_to_world,
        intrinsics,
        out_dir,
        seconds,
        final_losses,
        *,
        training_cam_to_world,
    ):
        # training_cam_to_world bound but not recorded: the raw-vs-corrected split is pinned at
        # the real writer, in test_rendering.py
        calls.append(
            {
                "model": model,
                "refine": refine,
                "image_ids": image_ids,
                "cam_to_world": cam_to_world,
                "seconds": seconds,
                "loss_values": final_losses,
            }
        )

    monkeypatch.setattr(trainer_module, "write_outputs", record)
    return calls


def _assert_trained(calls, points):
    """
    Assert one recorded run actually trained, not just returned.

    - means must have grown past the seed count and moved off the seed positions

    Args:
        calls: recorded calls from the writer stub.
        points: the seed cloud handed to the trainer.
    """
    assert len(calls) == 1
    call = calls[0]
    assert call["seconds"] > 0
    assert {"l1", "ssim"} <= set(call["loss_values"])
    means = call["model"].params["means"].detach().cpu()
    seed = torch.from_numpy(points)
    assert means.shape[0] >= len(points)
    assert not torch.allclose(means[: len(points)], seed)


@cuda
@pytest.mark.parametrize(
    "primitive, pose_opt, appearance_opt",
    [("3dgs", False, False), ("2dgs", False, False), ("3dgs", True, True)],
)
def test_train_short_run_moves_gaussians(monkeypatch, tmp_path, primitive, pose_opt, appearance_opt):
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    calls = _recorder(monkeypatch)
    cfg = SplatsConfig(
        primitive=primitive,
        pose_opt=pose_opt,
        appearance_opt=appearance_opt,
        max_steps=5,
        log_every=1,
        means_lr=1e-2,
    )
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)
    _assert_trained(calls, points)
    refine = calls[0]["refine"]
    assert refine.has_pose == pose_opt
    assert refine.has_appearance == appearance_opt
    if appearance_opt:
        # render["appearance"] must still be populated, or appearance_reg silently drops out
        assert "appearance_reg" in calls[0]["loss_values"]
        assert refine.appearance.weight.abs().sum() > 0


@cuda
def test_train_refining_every_step_still_moves_gaussians(monkeypatch, tmp_path):
    # Refine on EVERY step (start=-1, the gate is step > start)
    # - post_backward before optimizer.step would leave rebuilt params at .grad=None
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    calls = _recorder(monkeypatch)
    strategy = MCMCStrategy(cap_max=300, refine_start_iter=-1, refine_every=1, verbose=False)
    monkeypatch.setattr(gaussian_module, "make_strategy", lambda cfg, n_views: strategy)
    cfg = SplatsConfig(primitive="3dgs", max_steps=5, log_every=1, means_lr=1e-2)
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)
    _assert_trained(calls, points)

    # MCMC noise moves means even without an optimizer step; sh0 only moves through the optimizer
    sh0 = calls[0]["model"].params["sh0"].detach().cpu()
    seed_sh0 = (torch.from_numpy(colors).float() / 255.0 - 0.5) / SH_C0
    assert not torch.allclose(sh0[: len(points), 0, :], seed_sh0)


def test_splats_config_accepts_downscale_fields():
    cfg = SplatsConfig.from_dict({"enabled": True, "num_downscales": 1, "resolution_schedule": 100})
    assert cfg.num_downscales == 1 and cfg.resolution_schedule == 100
    # Defaults are splatfacto's
    default = SplatsConfig()
    assert default.num_downscales == 2 and default.resolution_schedule == 3000


def test_config_normalize_scene_default_off_and_settable():
    assert SplatsConfig().normalize_scene is False
    assert SplatsConfig.from_dict({"enabled": True, "normalize_scene": True}).normalize_scene is True


def test_config_appearance_fields():
    cfg = SplatsConfig.from_dict({"appearance_opt": True, "appearance_lr": 2e-3})
    assert cfg.appearance_opt and cfg.appearance_lr == 2e-3
    assert SplatsConfig().appearance_opt is False


def test_depth_ratio_accepted_on_normal_consistency_for_2dgs():
    cfg = SplatsConfig.from_dict(
        {"primitive": "2dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": 0.6}}}
    )
    assert cfg.losses["normal_consistency"]["depth_ratio"] == 0.6


def test_depth_ratio_rejected_on_another_loss():
    with pytest.raises(ValueError, match=r"splats\.losses\.depth:"):
        SplatsConfig.from_dict({"primitive": "2dgs", "losses": {"depth": {"weight": 0.01, "depth_ratio": 0.6}}})


def test_bad_spec_key_message_names_the_keys_legal_for_that_loss():
    # normal_consistency also allows depth_ratio, so its message must offer it; depth's must not
    with pytest.raises(ValueError, match=r"expected \{weight\[, depth_ratio, end, end_weight, start\]\}"):
        SplatsConfig.from_dict({"primitive": "2dgs", "losses": {"normal_consistency": {"weight": 0.05, "nope": 1}}})
    with pytest.raises(ValueError, match=r"expected \{weight\[, end, end_weight, start\]\}"):
        SplatsConfig.from_dict({"primitive": "2dgs", "losses": {"depth": {"weight": 0.01, "nope": 1}}})


@pytest.mark.parametrize("depth_ratio", [1.5, -0.5])
def test_depth_ratio_out_of_range_rejected(depth_ratio):
    with pytest.raises(ValueError, match=r"depth_ratio must be in \[0, 1\]"):
        SplatsConfig.from_dict(
            {"primitive": "2dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": depth_ratio}}}
        )


# yaml parses `yes`/`on`/`true` to True, which a bare float() would take as a full median blend
@pytest.mark.parametrize("depth_ratio", [True, False, None, "0.6", [0.6]])
def test_depth_ratio_non_numeric_rejected(depth_ratio):
    with pytest.raises(ValueError, match=r"depth_ratio must be a number in \[0, 1\]"):
        SplatsConfig.from_dict(
            {"primitive": "2dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": depth_ratio}}}
        )


def test_depth_ratio_is_2dgs_only():
    with pytest.raises(ValueError, match="2dgs"):
        SplatsConfig.from_dict(
            {"primitive": "3dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": 0.6}}}
        )


def test_depth_ratio_zero_is_allowed_on_3dgs():
    cfg = SplatsConfig.from_dict(
        {"primitive": "3dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": 0.0}}}
    )
    assert cfg.losses["normal_consistency"]["depth_ratio"] == 0.0


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_train_scaffold_runs_and_writes_outputs(tmp_path, primitive):
    """
    A short scaffold run produces the same artifact set as a vanilla run.
    """
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    cfg = SplatsConfig.from_dict(
        {
            "representation": "scaffold",
            "primitive": primitive,
            "max_steps": 60,
            "log_every": 10,
            "scaffold": {"n_offsets": 4, "feat_dim": 8, "update_from": 20, "update_until": 50, "refine_every": 10},
        }
    )
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)

    for name in ("splats.ply", "ckpt.pt", "splats_quality_report.json"):
        assert (tmp_path / name).exists(), name
    assert not (tmp_path / "splats.zarr").exists()
    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert "anchors" in checkpoint["splats"]
    assert "mlps" in checkpoint
    assert checkpoint["voxel_size"] > 0


@cuda
def test_scaffold_records_anchor_provenance(tmp_path):
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    cfg = SplatsConfig.from_dict(
        {
            "representation": "scaffold",
            "primitive": "3dgs",
            "max_steps": 60,
            "log_every": 10,
            "scaffold": {
                "n_offsets": 4,
                "feat_dim": 8,
                "update_from": 10,
                "update_until": 50,
                "refine_every": 10,
                "min_opacity": 0.5,
            },
        }
    )
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)

    # Provenance used to live in the zarr attrs; ckpt.pt and the report carry it now
    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    report = json.loads((tmp_path / "splats_quality_report.json").read_text())
    assert checkpoint["config"]["representation"] == "scaffold"
    assert report["summary"]["config"]["representation"] == "scaffold"
    assert report["summary"]["n_gaussians"] == len(checkpoint["splats"]["anchors"])


########################################
# The model interface: train() must not ask which representation it is training
########################################


def _branch_test_names(node):
    """
    Every identifier, attribute name and string literal a branch's test condition mentions.
    """
    mentioned = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            mentioned.add(child.id)
        elif isinstance(child, ast.Attribute):
            mentioned.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            mentioned.add(child.value)
    return mentioned


# Every spelling that means "which representation is this"
# - `primitive_unit`: the sanctioned discriminator, so the likeliest future violation
# - lowercase literals: catch a comparison against the config value
# - shared with the meta-test below, so narrowing this set cannot leave it green
REPRESENTATION_NAMES = {
    "Scaffold",
    "Gaussians",
    "representation",
    "primitive_unit",
    "anchor_field",
    "AnchorStrategy",
    "scaffold",
    "vanilla",
}


def _representation_branches(node):
    """
    Line numbers of every branch under `node` whose condition mentions a representation name.
    """
    return [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, (ast.If, ast.IfExp)) and REPRESENTATION_NAMES & _branch_test_names(child.test)
    ]


def test_trainer_has_no_representation_branches():
    """
    The trainer must not ask which model it is training.
    """
    # Over the `train` AST node, not the text after the `def`
    # - a string denylist misses single quotes, `isinstance(model, Scaffold)`, and code above it
    source = Path(trainer_module.__file__).read_text()
    train_node = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == "train"
    )

    branches = _representation_branches(train_node)

    assert branches == [], f"train() branches on the representation at line(s) {branches}"


@pytest.mark.parametrize("structure", ("statement", "ternary"))
@pytest.mark.parametrize("spelling", ("identifier", "attribute", "literal"))
@pytest.mark.parametrize("name", sorted(REPRESENTATION_NAMES))
def test_the_no_branch_check_sees_every_representation_name(name, spelling, structure):
    # Plant a branch and run the SAME predicate over the SAME set, so a narrowed set turns red
    # - one case per (name, spelling): `_branch_test_names` is a flat set of three clauses;
    #   deleting the ast.Attribute clause with a real branch planted still gave 277 passed, RC=0
    # - one case per structure: `_representation_branches` matches (ast.If, ast.IfExp); deleting
    #   ast.IfExp still gave 293 passed, RC=0
    # - the ternary plant sits inside an ast.Assign, so it also pins the walk's depth
    condition = {
        "identifier": f"isinstance(model, {name})",
        "attribute": f"model.{name}",
        "literal": f'cfg.unrelated_attr == "{name}"',
    }[spelling]
    body = {
        "statement": f"    if {condition}:\n        pass\n",
        "ternary": f"    unrelated = 1 if {condition} else 2\n",
    }[structure]
    planted = ast.parse(f"def train():\n{body}")

    assert _representation_branches(planted.body[0]) == [2]


def test_the_representation_name_set_is_not_narrowed():
    # One case PER NAME above, so dropping a name deletes its case instead of failing one
    # - measured: the earlier set of five, with THIS test deleted too, left 77 passed, 0 failed
    #   (the count tracks the axes above: 52 at 8 cases, 77 at 48)
    # - only an absolute literal pins the set; `primitive_unit` is the entry that matters most
    assert REPRESENTATION_NAMES == {
        "AnchorStrategy",
        "Gaussians",
        "Scaffold",
        "anchor_field",
        "primitive_unit",
        "representation",
        "scaffold",
        "vanilla",
    }


@cuda
def test_train_builds_a_gaussians_model_by_default(tmp_path):
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=3, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "cap_max": 500, "losses": {}})

    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)

    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert set(ckpt["splats"]) == {"means", "scales", "quats", "opacities", "sh0", "shN"}


@cuda
def test_train_survives_non_square_frames(tmp_path):
    """
    A non-square frame survives training: 32x32 fixtures hide a transposed (width, height).
    """
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=3, height=24, width=40, n_points=200)
    cfg = SplatsConfig.from_dict({"max_steps": 2, "cap_max": 500, "num_downscales": 0, "losses": {}})

    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)

    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert tuple(ckpt["image_size"]) == (24, 40)


@cuda
def test_train_builds_a_scaffold_model_when_asked(tmp_path):
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=3, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"representation": "scaffold", "scaffold": {}, "max_steps": 3, "losses": {}})

    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)

    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert "anchors" in ckpt["splats"]
    assert "mlps" in ckpt
    assert "voxel_size" in ckpt


def test_train_takes_the_tuning_constants_keyword_only():
    # `min_points` / `lr_decay` sit behind a bare `*`, after seven positionals
    # - a dropped `*` would silently bind a ninth positional to min_points
    # - match the message: the body raises its own TypeError, so a bare raises() passes either way
    with pytest.raises(TypeError, match="takes from 7 to 8 positional arguments"):
        train(None, None, None, None, None, None, None, None, 100)


def test_train_refuses_too_few_seed_points(tmp_path):
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=3, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "losses": {}})

    with pytest.raises(ValueError, match="need >= 100 seed points"):
        train(cfg, images, world_to_cam, intrinsics, points[:50], colors[:50], tmp_path)


def test_train_refuses_mismatched_per_view_arrays(tmp_path):
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=3, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "losses": {}})

    with pytest.raises(ValueError, match="frames mismatch"):
        train(cfg, images, world_to_cam[:2], intrinsics, points, colors, tmp_path)


@cuda
def test_neighbor_render_uses_the_neighbors_own_camera_id(monkeypatch, tmp_path):
    """
    The neighbor renders through its OWN camera id: appearance and pose deltas are per view.
    """
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    calls = _recorder(monkeypatch)

    # Pin both halves of the pair: every step trains view 0, whose only neighbor is view 2
    monkeypatch.setattr(trainer_module, "view_order", lambda n_views: itertools.repeat(0))
    monkeypatch.setattr(trainer_module, "select_near_views", lambda *args, **kwargs: [[2], [3], [0], [1]])

    real_render_neighbor = trainer_module.render_neighbor
    seen = []

    def record_neighbor(model, image, cam_to_world, near_intrinsics, camera_id):
        seen.append(int(camera_id.item()))
        return real_render_neighbor(model, image, cam_to_world, near_intrinsics, camera_id)

    monkeypatch.setattr(trainer_module, "render_neighbor", record_neighbor)
    cfg = SplatsConfig.from_dict(
        {
            "primitive": "3dgs",
            "max_steps": 3,
            "log_every": 1,
            "appearance_opt": True,
            "num_downscales": 0,
            "losses": {"pgsr_multiview": {"weight": 0.1}},
        }
    )
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)

    assert len(calls) == 1
    assert seen and set(seen) == {2}


########################################
# Public API
########################################


def test_public_api_surface():
    import collab_splats.splats as splats

    # Exactly six: two model classes, the config, the entry point, the ckpt reader, the commit
    assert set(splats.__all__) == {
        "GSPLAT_COMMIT",
        "Gaussians",
        "Scaffold",
        "SplatsConfig",
        "load_checkpoint",
        "train",
    }
    for name in splats.__all__:
        assert hasattr(splats, name), name
