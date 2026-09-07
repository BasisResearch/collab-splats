"""
render_gaussians' output contract, plus the splat stage's artifacts.

- output shapes for both primitives; 3DGS normals face the camera
- render_views, write_outputs (through `train`), load_checkpoint
"""

import json
import logging
import random
import types

import numpy as np
import pytest
import torch
from gsplat.exporter import load_ply_to_splats

from collab_splats.splats.cameras import CameraOpt
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.rendering import (
    gaussian_normals_in_camera_frame,
    load_checkpoint,
    render_gaussians,
    render_views,
    write_outputs,
)
from collab_splats.splats.scaffold import Scaffold
from collab_splats.splats.trainer import REPRESENTATIONS, SplatsConfig, train
from tests.splats.synthetic import make_scene

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def _render_only_model(gaussians, primitive="3dgs", device="cuda"):
    """
    Wrap a raw ParameterDict fixture in a render-only Gaussians.

    - the activation the rasterizer needs comes from production code, not a copy in this file

    Args:
        gaussians: raw ParameterDict of splat tensors.
        primitive: "3dgs" or "2dgs".
        device: torch device.

    Returns:
        Gaussians, at sh_degree 0.
    """
    config = {"primitive": primitive, "sh_degree": 0, "sh_degree_interval": 1}
    return Gaussians.from_checkpoint({"splats": gaussians, "config": config}, device)


def _render(primitive, gaussians, cam_to_world, intrinsics, width, height, **kwargs):
    """
    Rasterize a ParameterDict fixture at SH degree 0 with absgrad off.

    Args:
        primitive: "3dgs" or "2dgs".
        gaussians: raw ParameterDict of splat tensors.
        cam_to_world: (1, 4, 4).
        intrinsics: (1, 3, 3).
        width: render width, px.
        height: render height, px.
        kwargs: forwarded to render_gaussians.

    Returns:
        (render, gsplat strategy info).
    """
    model = _render_only_model(gaussians, primitive)
    return render_gaussians(primitive, model.activate(), cam_to_world, intrinsics, width, height, 0, False, **kwargs)


def _gaussians(n_points=200, device="cuda"):
    """
    A deterministic cloud of small splats in the unit cube.

    Args:
        n_points: primitive count.
        device: torch device.

    Returns:
        ParameterDict of raw (unactivated) splat tensors.
    """
    gen = torch.Generator().manual_seed(0)
    gaussians = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.rand(n_points, 3, generator=gen) - 0.5),
            "scales": torch.nn.Parameter(torch.full((n_points, 3), -3.0)),
            "quats": torch.nn.Parameter(torch.rand(n_points, 4, generator=gen)),
            "opacities": torch.nn.Parameter(torch.zeros(n_points)),
            "sh0": torch.nn.Parameter(torch.rand(n_points, 1, 3, generator=gen)),
            "shN": torch.nn.Parameter(torch.zeros(n_points, 15, 3)),
        }
    )
    return gaussians.to(device)


def _camera(device="cuda"):
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


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_render_shapes(primitive):
    cam_to_world, intrinsics = _camera()
    render, info = _render(primitive, _gaussians(), cam_to_world, intrinsics, 64, 64)
    assert render["rgb"].shape == (1, 64, 64, 3)
    assert render["alpha"].shape == (1, 64, 64, 1)
    assert render["depth"].shape == (1, 64, 64, 1)
    assert render["normal"].shape == (1, 64, 64, 3)
    assert render["depth_normal"].shape == (1, 64, 64, 3)
    assert ("distortion" in render) == (primitive == "2dgs")
    expected_gradient_key = "means2d" if primitive == "3dgs" else "gradient_2dgs"
    assert expected_gradient_key in info


@cuda
def test_render_3dgs_without_normals_omits_normal_keys():
    cam_to_world, intrinsics = _camera()
    render, info = _render("3dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, render_normals=False)
    assert set(render) == {"rgb", "alpha", "depth"}
    assert render["rgb"].shape == (1, 64, 64, 3)
    assert "render_extra_signals" not in info or info["render_extra_signals"] is None
    assert "means2d" in info


@cuda
def test_gaussian_normals_face_the_camera():
    gaussians = _gaussians()
    cam_to_world, _ = _camera()
    world_to_cam = torch.linalg.inv(cam_to_world)[0]
    scales = torch.exp(gaussians["scales"])
    normals, means_cam = gaussian_normals_in_camera_frame(gaussians["quats"], scales, gaussians["means"], world_to_cam)

    rotation_w2c = world_to_cam[:3, :3]
    translation_w2c = world_to_cam[:3, 3]
    expected_means_cam = gaussians["means"] @ rotation_w2c.T + translation_w2c
    normal_lengths = normals.norm(dim=-1)
    facing = (normals * means_cam).sum(-1)
    assert normals.shape == (200, 3)
    assert means_cam.shape == (200, 3)
    assert torch.allclose(means_cam, expected_means_cam, atol=1e-6)
    assert torch.allclose(normal_lengths, torch.ones(200, device="cuda"), atol=1e-5)
    assert (facing <= 1e-6).all()


def _flat_disc(device="cuda"):
    """
    One large opaque flat disc at the origin, normal on world +x.

    - local z (2DGS normal axis, thin 3DGS axis) rotated 90 deg about y onto world +x
    - so both primitives agree the normal is world +x

    Args:
        device: torch device.

    Returns:
        Single-primitive ParameterDict.
    """
    gaussians = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(1, 3)),
            "scales": torch.nn.Parameter(torch.tensor([[-1.0, -1.0, -8.0]])),
            "quats": torch.nn.Parameter(torch.tensor([[0.7071068, 0, 0.7071068, 0]])),
            "opacities": torch.nn.Parameter(torch.full((1,), 10.0)),
            "sh0": torch.nn.Parameter(torch.ones(1, 1, 3)),
            "shN": torch.nn.Parameter(torch.zeros(1, 15, 3)),
        }
    )
    return gaussians.to(device)


def _rotated_camera(device="cuda"):
    """
    Camera on the +x axis at distance 4, looking back at the origin.

    - rotated 90 degrees about y so OpenCV +z faces the origin

    Args:
        device: torch device for both tensors.

    Returns:
        ((1, 4, 4) cam_to_world, (1, 3, 3) intrinsics).
    """
    cam_to_world = torch.zeros(1, 4, 4, device=device)
    cam_to_world[0, :3, 0] = torch.tensor([0.0, 0, 1])
    cam_to_world[0, :3, 1] = torch.tensor([0.0, 1, 0])
    cam_to_world[0, :3, 2] = torch.tensor([-1.0, 0, 0])
    cam_to_world[0, :3, 3] = torch.tensor([4.0, 0, 0])
    cam_to_world[0, 3, 3] = 1.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)[None]
    return cam_to_world, intrinsics


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_normals_are_camera_frame_under_rotated_camera(primitive):
    # Disc normal is world +x, which a 90 deg camera rotation maps to camera -z
    # - a world-frame normal leaking through would read as camera +x
    cam_to_world, intrinsics = _rotated_camera()
    render, _ = _render(primitive, _flat_disc(), cam_to_world, intrinsics, 64, 64)
    center_normal = render["normal"][0, 32, 32]
    center_depth_normal = render["depth_normal"][0, 32, 32]
    assert render["alpha"][0, 32, 32, 0] > 0.5
    unit_normal = torch.nn.functional.normalize(center_normal, dim=-1)
    unit_depth_normal = torch.nn.functional.normalize(center_depth_normal, dim=-1)
    expected = torch.tensor([0.0, 0, -1], device="cuda")
    assert unit_normal[2] < -0.9
    assert torch.allclose(unit_normal, expected, atol=0.05)
    assert torch.allclose(unit_depth_normal, expected, atol=0.05)


@cuda
def test_2dgs_render_carries_median_depth_and_its_normal():
    cam_to_world, intrinsics = _camera()
    render, _info = _render("2dgs", _gaussians(), cam_to_world, intrinsics, 64, 64)
    assert render["median_depth"].shape == render["depth"].shape
    assert render["depth_normal_median"].shape == render["depth_normal"].shape
    assert render["median_depth"].requires_grad

    # Content, not just plumbing
    # - over a cloud, median depth is not the alpha-weighted expectation
    # - its finite difference is not a second copy of the expected depth normal
    assert (render["median_depth"] > 0).any()
    assert not torch.equal(render["median_depth"], render["depth"])
    assert not torch.allclose(render["depth_normal_median"], render["depth_normal"])


@cuda
def test_2dgs_median_depth_equals_expected_depth_on_a_single_surface():
    # One opaque disc face-on at distance 4: exactly one Gaussian per ray
    # - so the median depth IS the expected depth, and the distortion map is 0 there
    # - both outputs are (C,H,W,1) and differentiable, so only their values tell them apart
    cam_to_world, intrinsics = _rotated_camera()
    render, _info = _render("2dgs", _flat_disc(), cam_to_world, intrinsics, 64, 64)
    assert render["alpha"][0, 32, 32, 0] > 0.5
    assert render["depth"][0, 32, 32, 0].item() == pytest.approx(4.0, rel=1e-3)
    assert render["median_depth"][0, 32, 32, 0].item() == pytest.approx(4.0, rel=1e-3)
    assert render["distortion"][0, 32, 32, 0].item() == pytest.approx(0.0, abs=1e-6)


@cuda
def test_2dgs_without_normals_omits_the_finite_differenced_depth_normals():
    # The rasterizer's own normal and median depth come free and stay; only depth_to_normal is skipped
    cam_to_world, intrinsics = _camera()
    render, _info = _render("2dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, render_normals=False)
    assert set(render) == {"rgb", "alpha", "depth", "median_depth", "normal", "distortion"}


@cuda
def test_3dgs_render_has_no_median_depth():
    cam_to_world, intrinsics = _camera()
    render, _info = _render("3dgs", _gaussians(), cam_to_world, intrinsics, 64, 64)
    assert "median_depth" not in render


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_model_render_is_the_same_call_as_render_gaussians(primitive):
    """
    Gaussians.render is render_gaussians over the activated params, not a second code path.
    """
    cam_to_world, intrinsics = _camera()
    model = _render_only_model(_gaussians(), primitive)
    camera_id = torch.zeros(1, dtype=torch.long, device="cuda")

    reference, _ = render_gaussians(primitive, model.activate(), cam_to_world, intrinsics, 64, 64, 0, False)
    actual, _ = model.render(cam_to_world, intrinsics, 64, 64, camera_id, step=0)

    assert set(actual) == set(reference)
    for key, expected in reference.items():
        assert torch.equal(actual[key], expected), key


@cuda
def test_render_plane_rejects_2dgs():
    # No 2dgs-pgsr upstream: the plane signals have no 2DGS definition
    cam_to_world, intrinsics = _camera()
    with pytest.raises(ValueError, match="3DGS-only"):
        _render("2dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, render_plane=True)


@cuda
def test_render_plane_adds_exactly_the_four_plane_keys():
    cam_to_world, intrinsics = _camera()
    plane, _info = _render("3dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, render_plane=True)
    assert plane["plane_normal"].shape == (1, 64, 64, 3)
    assert plane["plane_distance"].shape == (1, 64, 64, 1)
    assert plane["plane_depth"].shape == (1, 64, 64, 1)
    assert plane["plane_depth_normal"].shape == (1, 64, 64, 3)

    # Without the flag none of them exist; the ordinary keys are unchanged either way
    ordinary, _info = _render("3dgs", _gaussians(), cam_to_world, intrinsics, 64, 64)
    assert set(ordinary) == {"rgb", "alpha", "depth", "normal", "depth_normal"}


@cuda
def test_render_plane_normal_is_the_raw_accumulated_map():
    # Upstream returns `rendered_normal` un-normalized
    # - plane depth divides one accumulated sum by another, so the missing 1/alpha cancels
    # - normalizing here would break that ratio
    cam_to_world, intrinsics = _rotated_camera()
    render, _info = _render("3dgs", _flat_disc(), cam_to_world, intrinsics, 64, 64, render_plane=True)
    center_alpha = render["alpha"][0, 32, 32, 0]
    center_plane_normal = render["plane_normal"][0, 32, 32]
    assert center_alpha > 0.9
    assert center_alpha < 1.0

    # The disc normal is camera -z, scaled by the accumulated alpha rather than to unit length
    assert center_plane_normal.norm().item() == pytest.approx(center_alpha.item(), rel=1e-4)
    assert render["normal"][0, 32, 32].norm().item() == pytest.approx(1.0, rel=1e-4)


@cuda
def test_plane_depth_matches_rasterized_depth_on_a_fronto_parallel_plane():
    # Two estimators of one surface must coincide on a fronto-parallel plane
    # - `depth`: alpha-weighted expected z; `plane_depth`: ray-plane `distance / -(n . ray)`
    # - median over the confident interior, not elementwise
    # - at the antialiased rim partial coverage separates them for real; allclose there tests the rim
    cam_to_world, intrinsics = _rotated_camera()
    render, _info = _render("3dgs", _flat_disc(), cam_to_world, intrinsics, 64, 64, render_plane=True)
    interior = render["alpha"][..., 0] > 0.9
    assert interior.sum() > 0

    # The disc sits at the origin with the camera 4 away, so both estimators must read 4
    plane_depth = render["plane_depth"][..., 0][interior]
    rasterized_depth = render["depth"][..., 0][interior]
    assert plane_depth.median().item() == pytest.approx(4.0, rel=1e-3)
    assert (plane_depth - rasterized_depth).abs().median().item() < 1e-3

    # And the plane depth's own normal is the unit camera-frame normal, not alpha-scaled
    center_plane_depth_normal = render["plane_depth_normal"][0, 32, 32]
    assert center_plane_depth_normal.norm().item() == pytest.approx(1.0, rel=1e-4)
    assert torch.allclose(center_plane_depth_normal, torch.tensor([0.0, 0, -1], device="cuda"), atol=0.05)


@cuda
def test_render_plane_forces_the_extra_signal_pass_on():
    # Plane signals ride the same four extra channels as the normals
    # - render_normals=False cannot switch that pass off underneath them
    cam_to_world, intrinsics = _camera()
    render, _info = _render(
        "3dgs",
        _gaussians(),
        cam_to_world,
        intrinsics,
        64,
        64,
        render_normals=False,
        render_plane=True,
    )
    assert {"plane_normal", "plane_distance", "plane_depth", "plane_depth_normal"} <= set(render)
    assert render["plane_depth"].shape == (1, 64, 64, 1)

    # Turning the pass on brings the normals back with it rather than half-filling the channels
    assert "normal" in render and "depth_normal" in render


@cuda
def test_plane_path_does_not_perturb_the_ordinary_render():
    # The plane distance only replaces a channel that was a zero pad
    # - so color, depth and the normal_consistency inputs come back bit-for-bit identical
    cam_to_world, intrinsics = _camera()
    gaussians = _gaussians()
    ordinary, _info = _render("3dgs", gaussians, cam_to_world, intrinsics, 64, 64)
    plane, _info = _render("3dgs", gaussians, cam_to_world, intrinsics, 64, 64, render_plane=True)
    for key in ("rgb", "alpha", "depth", "normal", "depth_normal"):
        assert torch.equal(plane[key], ordinary[key]), key


########################################
# Outputs
########################################


def _train_stub(tmp_path, n_views=4, **overrides):
    """
    A small scene trained for 3 steps — enough for the writer to produce every artifact.

    - frames are 40 wide by 24 high on purpose: a square scene hides a transposed (width, height)
      everywhere downstream, including through `load_checkpoint`'s `image_size` read

    Args:
        tmp_path: output directory.
        n_views: views in the synthetic scene.
        overrides: merged into the SplatsConfig block.

    Returns:
        The training frames, so a caller can score the report against what was trained on.
    """
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=n_views, height=24, width=40)
    block = {"max_steps": 3, "cap_max": 500, "losses": {}, **overrides}
    cfg = SplatsConfig.from_dict(block)
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)
    return images


def _seed(value=0):
    """
    Seed every generator a training run draws from, so a comparison sees code and not noise.

    Args:
        value: seed for random, numpy and torch (CPU and CUDA).
    """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _seed_pose_deltas(monkeypatch, shift=0.02):
    """
    Make every CameraOpt `train` builds start with a per-view translation delta.

    - measured: 3 steps move the deltas by 4.1e-05, all of it the normalize/denormalize round trip
    - so a test needing the two pose sets to differ seeds them instead of training for them

    Args:
        monkeypatch: patches CameraOpt.from_config.
        shift: per-view translation step; ramps per view, since one shared shift would
            leave the views indistinguishable.
    """
    build = CameraOpt.from_config.__func__

    def seeded(cls, *args, **kwargs):
        module = build(cls, *args, **kwargs)
        with torch.no_grad():
            ramp = torch.arange(1, len(module.translation.weight) + 1, device=module.translation.weight.device)
            module.translation.weight += ramp[:, None].float() * shift
        return module

    monkeypatch.setattr(CameraOpt, "from_config", classmethod(seeded))


@cuda
def test_write_outputs_writes_three_artifacts_and_no_zarr(tmp_path):
    _train_stub(tmp_path)

    assert (tmp_path / "splats.ply").exists()
    assert (tmp_path / "ckpt.pt").exists()
    assert (tmp_path / "splats_quality_report.json").exists()
    assert not (tmp_path / "splats.zarr").exists()


@cuda
def test_checkpoint_is_self_contained(tmp_path):
    _train_stub(tmp_path)
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)

    assert set(ckpt) >= {"splats", "config", "cam_to_world", "intrinsics", "image_ids", "image_size", "appearance"}
    # The pose deltas are folded into cam_to_world, so there is nothing left to restore separately
    assert "pose_adjust" not in ckpt
    assert ckpt["cam_to_world"].shape == (4, 4, 4)
    assert ckpt["intrinsics"].shape == (4, 3, 3)
    assert ckpt["image_ids"] == [0, 1, 2, 3]
    assert tuple(ckpt["image_size"]) == (24, 40)

    # An ABSOLUTE guard on K, not a round trip
    # - the mesh stage re-renders from this key: a scaled or transposed K ships a wrong mesh
    # - a writer-side scale a reader undoes is invisible to any read-back comparison
    # - literals are make_scene's own: focal 60, principal point at the 40x24 center
    expected_k = torch.tensor([[60.0, 0.0, 20.0], [0.0, 60.0, 12.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(ckpt["intrinsics"][0], expected_k)
    assert torch.equal(ckpt["intrinsics"][0], ckpt["intrinsics"][-1])


@cuda
def test_checkpointed_poses_carry_the_pose_correction(tmp_path):
    # pose_opt is on by default: stored poses are the refined ones, not the raw inputs
    # - three steps is enough to move them off zero
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=4, height=32, width=32)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "cap_max": 500, "losses": {}, "pose_opt": True})
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)
    stored = torch.load(tmp_path / "ckpt.pt", weights_only=False)["cam_to_world"]

    raw = torch.from_numpy(np.linalg.inv(world_to_cam)).float()
    assert not torch.equal(stored, raw)
    assert torch.allclose(stored, raw, atol=1e-2)


@cuda
def test_quality_report_keeps_its_schema(tmp_path):
    _train_stub(tmp_path)
    report = json.loads((tmp_path / "splats_quality_report.json").read_text())

    assert set(report) == {"summary", "per_frame"}
    assert set(report["summary"]) >= {"psnr", "ssim", "n_gaussians", "seconds", "final_losses", "config"}
    assert len(report["per_frame"]) == 4
    assert set(report["per_frame"][0]) == {"image_id", "psnr", "ssim"}

    # A vanilla model's primitive count is static
    # - so the decoded-count keys are absent here: absent, not zero and not null
    assert "n_decoded_mean" not in report["summary"]


@cuda
def test_quality_report_carries_the_decoded_count_for_scaffold(tmp_path):
    # summary.n_gaussians counts ANCHORS for a scaffold
    # - the rendered count is per-view (frustum culling, opacity gate), so it gets its own keys
    _train_stub(tmp_path, n_views=2, representation="scaffold")
    report = json.loads((tmp_path / "splats_quality_report.json").read_text())

    assert set(report["per_frame"][0]) == {"image_id", "psnr", "ssim", "n_decoded"}
    assert "n_decoded_mean" in report["summary"]

    # A count, and json round-trips the distinction
    # - a float would freeze `1009.0` into the report; n_decoded_mean stays a float, it is a mean
    assert isinstance(report["per_frame"][0]["n_decoded"], int)

    # Anchors and decoded primitives are different counts
    # - an assertion that cannot tell them apart passes on a report that copied n_gaussians across
    assert report["summary"]["n_decoded_mean"] > 0
    assert report["summary"]["n_decoded_mean"] != report["summary"]["n_gaussians"]
    assert report["summary"]["n_decoded_mean"] == pytest.approx(
        float(np.mean([frame["n_decoded"] for frame in report["per_frame"]]))
    )


@cuda
def test_the_scaffold_ply_bakes_against_the_uncorrected_training_poses(tmp_path, monkeypatch):
    # The pose set the export receives decides the ply bytes
    # - splats.ply is frozen byte-for-byte; Scaffold bakes each anchor at the mean direction of
    #   the cameras that saw it
    # - ply keeps the raw training poses, ckpt.pt the corrected ones: deliberate, and pinned here
    _seed_pose_deltas(monkeypatch)
    baked_against = []
    export_gaussians = Scaffold.export_gaussians

    def spy(self, cam_to_world, intrinsics, width, height):
        baked_against.append(cam_to_world.detach().clone())
        return export_gaussians(self, cam_to_world, intrinsics, width, height)

    monkeypatch.setattr(Scaffold, "export_gaussians", spy)

    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=3, height=24, width=40)
    cfg = SplatsConfig.from_dict(
        {"max_steps": 3, "cap_max": 500, "losses": {}, "representation": "scaffold", "pose_opt": True}
    )
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)

    stored = torch.load(tmp_path / "ckpt.pt", weights_only=False)["cam_to_world"]
    exported = baked_against[0].cpu()
    raw = torch.from_numpy(np.linalg.inv(world_to_cam)).float()

    # The seeded deltas moved the cameras far enough that the two pose sets are distinguishable
    assert len(baked_against) == 1
    assert (stored - exported).abs().max().item() > 0.01

    # The export saw the training poses; only the checkpoint carries the pose correction
    assert torch.allclose(exported, raw, atol=1e-4)
    assert not torch.allclose(stored, raw, atol=1e-4)


def test_write_outputs_takes_the_training_poses_keyword_only():
    # A dropped bare `*` would swap the two pose sets silently
    # - same shape, adjacent in the call: a caller could pass corrected where raw belongs
    # - the message is matched because the body raises its own TypeError on these arguments
    # - measured: a bare `pytest.raises(TypeError)` passes whether or not the `*` is there
    with pytest.raises(TypeError, match="takes 10 positional arguments"):
        write_outputs(None, None, None, None, [], None, None, None, 0.0, {}, None)


@cuda
def test_load_checkpoint_round_trips_the_model_and_the_cameras(tmp_path):
    _train_stub(tmp_path)

    model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width) = load_checkpoint(
        tmp_path / "ckpt.pt", "cpu"
    )

    assert isinstance(model, Gaussians)
    # appearance_opt defaults false, so the rebuilt module is the identity on both halves
    assert isinstance(camera_opt, CameraOpt)
    assert camera_opt.appearance is None and camera_opt.rotation is None
    assert cam_to_world.shape == (4, 4, 4)
    assert intrinsics.shape == (4, 3, 3)
    assert image_ids == [0, 1, 2, 3]
    assert (height, width) == (24, 40)


@cuda
def test_load_checkpoint_restores_the_color_affine_but_never_the_pose_half(tmp_path):
    _train_stub(tmp_path, pose_opt=True, appearance_opt=True)
    saved = torch.load(tmp_path / "ckpt.pt", weights_only=False)["appearance"]

    _, camera_opt, *_ = load_checkpoint(tmp_path / "ckpt.pt", "cpu")

    # The affine comes back; the pose deltas do not
    # - write_outputs folded them into cam_to_world, so a restored pose half applies them twice
    assert camera_opt.appearance is not None
    assert torch.allclose(camera_opt.appearance.weight, saved["weight"].cpu())
    assert camera_opt.translation is None and camera_opt.rotation is None


@cuda
def test_load_checkpoint_rebuilds_a_scaffold_from_the_representation(tmp_path):
    # The class comes off cfg.representation, not off which keys happen to be in the file
    _train_stub(tmp_path, n_views=2, representation="scaffold")

    model, *_ = load_checkpoint(tmp_path / "ckpt.pt", "cuda")

    assert isinstance(model, Scaffold)
    assert model.n_primitives > 0


@cuda
def test_render_views_yields_one_render_per_view(tmp_path):
    _train_stub(tmp_path)
    model, camera_opt, cam_to_world, intrinsics, _, (height, width) = load_checkpoint(tmp_path / "ckpt.pt", "cuda")

    renders = list(render_views(model, camera_opt, cam_to_world, intrinsics, height, width))

    assert len(renders) == 4
    assert set(renders[0]) >= {"rgb", "depth", "alpha", "normal"}
    assert renders[0]["rgb"].shape == (1, 24, 40, 3)
    # A generator, not a list: the mesh stage streams 300 views through it
    assert isinstance(render_views(model, camera_opt, cam_to_world, intrinsics, height, width), types.GeneratorType)


@cuda
def test_render_views_leaves_grad_enabled_while_suspended(tmp_path):
    # Grad mode is process-global
    # - a `with torch.no_grad():` INSIDE the body stays in force while the generator sits at a yield
    # - any partial consumer (zip, bare next, break) then disables autograd for every later test,
    #   manufacturing failures in unrelated files and corrupting mutation-kill attribution
    # - the decorator form is what keeps this true; a `with` inside the body breaks it
    _train_stub(tmp_path, n_views=2)
    model, camera_opt, cam_to_world, intrinsics, _, (height, width) = load_checkpoint(tmp_path / "ckpt.pt", "cuda")

    assert torch.is_grad_enabled(), "precondition: the suite runs with grad on, or this is vacuous"

    # Suspend the generator mid-stream, exactly as a partial consumer leaves it
    renders = render_views(model, camera_opt, cam_to_world, intrinsics, height, width)
    first = next(renders)

    assert torch.is_grad_enabled(), "render_views leaked no_grad out of a suspended generator"

    # ...and grad really was off INSIDE the body, so the guard above is not just a deleted no_grad
    assert not first["rgb"].requires_grad
    renders.close()
    assert torch.is_grad_enabled()


@cuda
def test_render_views_applies_each_views_own_color_affine(tmp_path):
    # Every downstream artifact is scored or written through this call
    # - a dropped affine shows up only as an unexplained dPSNR against the training-time renders
    # - the two views get DIFFERENT weights: one shared weight makes lookup-by-own-id and
    #   lookup-by-constant-zero the same computation
    _train_stub(tmp_path, n_views=2, appearance_opt=True)
    model, camera_opt, cam_to_world, intrinsics, _, (height, width) = load_checkpoint(tmp_path / "ckpt.pt", "cuda")
    per_view = [(1.0, 0.2), (-0.5, 0.05)]
    with torch.no_grad():
        for view, (gain, bias) in enumerate(per_view):
            camera_opt.appearance.weight[view] = torch.tensor([gain] * 3 + [bias] * 3, device="cuda")

    corrected = list(render_views(model, camera_opt, cam_to_world, intrinsics, height, width))

    assert len(corrected) == len(per_view)
    for view, (gain, bias) in enumerate(per_view):
        camera_id = torch.tensor([view], device="cuda")
        raw, _ = model.render(
            cam_to_world[view : view + 1], intrinsics[view : view + 1], width, height, camera_id, step=None
        )

        # gain = 1 + params[:3], bias = params[3:], then the clamp render_views owns
        assert torch.allclose(corrected[view]["rgb"], (raw["rgb"] * (1.0 + gain) + bias).clamp(0, 1), atol=1e-6)
        assert not torch.allclose(corrected[view]["rgb"], raw["rgb"].clamp(0, 1))

    # And the two views really are told apart: view 1 through view 0's affine is a different image
    view_one_raw, _ = model.render(cam_to_world[1:2], intrinsics[1:2], width, height, torch.tensor([1], device="cuda"))
    assert not torch.allclose(corrected[1]["rgb"], (view_one_raw["rgb"] * 2.0 + 0.2).clamp(0, 1), atol=1e-6)


@cuda
def test_load_checkpoint_refuses_an_unknown_representation(tmp_path):
    # The class comes off the trainer's MODEL_CLASSES
    # - the same mapping the config validator takes its allow-list from
    # - an `else Gaussians` fallback rebuilt a vanilla model for ANY unknown name, silently
    _train_stub(tmp_path, n_views=2)
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    ckpt["config"]["representation"] = "octree"
    torch.save(ckpt, tmp_path / "ckpt.pt")

    # The message must name the allow-list and the offending value, not just fail
    # - a bare `KeyError: 'octree'` says neither which names are valid nor which checkpoint is bad
    with pytest.raises(ValueError, match=r"must be one of .*got 'octree'") as excinfo:
        load_checkpoint(tmp_path / "ckpt.pt", "cpu")

    # Each allowed name matched independently, not in sequence
    # - REPRESENTATIONS is `tuple(MODEL_CLASSES)`, so a `.*vanilla.*scaffold.*` regex pins ORDER
    # - measured: swapping the two entries reddens the sequenced form, and leaves this one green
    # - measured: a partial three-entry message passes the sequenced form and fails this one
    # - the set's width is pinned by its own literal in test_trainer.py
    message = str(excinfo.value)
    assert "ckpt.pt" in message
    for representation in REPRESENTATIONS:
        assert f"'{representation}'" in message, message


@cuda
@pytest.mark.parametrize(
    "representation, unit, other", [("vanilla", "gaussians", "anchors"), ("scaffold", "anchors", "gaussians")]
)
def test_write_outputs_logs_the_models_own_primitive_unit(tmp_path, caplog, representation, unit, other):
    # The summary line names the model's own primitive unit
    # - n_primitives counts anchors for a scaffold, gaussians for a vanilla model
    # - off the model, not type(model).__name__, which falls through to "gaussians" on any rename
    with caplog.at_level(logging.INFO, logger="collab_splats.splats.rendering"):
        _train_stub(tmp_path, n_views=2, representation=representation)

    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.name == "collab_splats.splats.rendering" and record.getMessage().startswith("splats: ")
    ]
    assert len(summaries) == 1
    assert f" {unit}" in summaries[0]
    assert other not in summaries[0]


@cuda
def test_quality_report_scores_each_frame_against_its_own_image(tmp_path):
    # render_views yields views in order
    # - reversing either side of the zip pairs every render with a different view's photo
    # - every psnr/ssim wrong, every image_id mislabelled, and the report still well formed
    images = _train_stub(tmp_path)
    report = json.loads((tmp_path / "splats_quality_report.json").read_text())
    model, camera_opt, cam_to_world, intrinsics, _, (height, width) = load_checkpoint(tmp_path / "ckpt.pt", "cuda")

    # Recompute each psnr from the render at that POSITION against the image at that position
    # - indexing the target by the reported id would move with the mislabeling
    renders = list(render_views(model, camera_opt, cam_to_world, intrinsics, height, width))
    for view, (frame, render) in enumerate(zip(report["per_frame"], renders)):
        target = torch.from_numpy(images[view]).to(render["rgb"].device).float()[None] / 255.0
        mse = torch.nn.functional.mse_loss(render["rgb"], target).item()
        assert frame["image_id"] == view
        assert frame["psnr"] == pytest.approx(10 * np.log10(1.0 / max(mse, 1e-12)), rel=1e-4)

    # The frames are distinguishable, so that comparison is not vacuous
    # - the same render scored against the next view's photo is a materially different number
    first_render = renders[0]
    wrong_target = torch.from_numpy(images[1]).to(first_render["rgb"].device).float()[None] / 255.0
    wrong_mse = torch.nn.functional.mse_loss(first_render["rgb"], wrong_target).item()
    assert abs(10 * np.log10(1.0 / max(wrong_mse, 1e-12)) - report["per_frame"][0]["psnr"]) > 0.1


@cuda
def test_the_scaffold_ply_holds_the_values_exported_at_the_frame_size(tmp_path, monkeypatch):
    # splats.ply is a frozen surface and `.exists()` cannot see into it
    # - swapping `width, height` in the export changes the frustum every anchor is tested against,
    #   and so the baked view direction, decoded colors, and which offsets pass the opacity gate
    # - measured: byte and vertex counts identical either way, 56864 B / 1009 vertices
    # - so a size check passes and a count check passes; only the values catch it
    _seed()
    captured = {}
    export_gaussians = Scaffold.export_gaussians

    def spy(self, cam_to_world, intrinsics, width, height):
        captured["model"] = self
        captured["cam_to_world"] = cam_to_world.detach().clone()
        captured["intrinsics"] = intrinsics.detach().clone()
        return export_gaussians(self, cam_to_world, intrinsics, width, height)

    monkeypatch.setattr(Scaffold, "export_gaussians", spy)

    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=3, height=24, width=40)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "cap_max": 500, "losses": {}, "representation": "scaffold"})
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)

    # Re-export the same model at the size the fixture built: 40 wide, 24 high
    # - the two literals are the independent guard; a swap at the call site cannot move them
    expected = export_gaussians(captured["model"], captured["cam_to_world"], captured["intrinsics"], 40, 24)
    written = load_ply_to_splats(str(tmp_path / "splats.ply"))
    expected_np = {name: value.detach().cpu().numpy() for name, value in expected.items()}

    # Vacuity guard: a swapped frame size must actually change what is exported
    # - measured: 6 of 1009 decoded Gaussians move — 0.40 opacities, 0.066 sh0, 0.037 scales, 0.35 quats
    # - `means` excluded: anchor positions do not depend on the frustum, measured 0 of 1009
    # - if the fixture drifts to orientation-invariant anchors, all five match under the mutant
    #   and the assertion below passes green with nothing to notice
    swapped = export_gaussians(captured["model"], captured["cam_to_world"], captured["intrinsics"], 24, 40)
    assert len(swapped["means"]) == len(expected["means"]), "fixture drifted — re-measure the margins below"
    for name in ("opacities", "sh0", "scales", "quats"):
        margin = (expected[name] - swapped[name]).abs().max().item()
        assert margin > 0.01, f"{name}: a swapped frame size changes nothing, so the check below is vacuous"

    # A single-splat fallback would make every comparison below vacuous
    assert len(written["means"]) > 1
    assert len(written["means"]) == len(expected_np["means"])
    for name in ("means", "opacities", "sh0", "scales", "quats"):
        assert np.allclose(written[name].numpy(), expected_np[name], rtol=1e-4, atol=1e-6), name


@cuda
def test_render_views_yields_median_depth_for_2dgs(tmp_path):
    _train_stub(tmp_path, n_views=2, primitive="2dgs")
    model, camera_opt, cam_to_world, intrinsics, _, (height, width) = load_checkpoint(tmp_path / "ckpt.pt", "cuda")

    render = list(render_views(model, camera_opt, cam_to_world, intrinsics, height, width))[0]

    assert "median_depth" in render
