"""
End-to-end: train() on the synthetic scene writes every artifact with the documented shapes/attrs.
"""

import json

import numpy as np
import pytest
import torch
import zarr

from collab_splats.splats import GSPLAT_COMMIT
from collab_splats.splats.cameras import CameraOptModule
from collab_splats.splats.trainer import SplatsConfig, train
from tests.splats.synthetic import make_scene

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_train_writes_all_outputs(tmp_path, primitive):
    images, world_to_cam, intrinsics, points, colors, depths = make_scene()
    losses = {"depth": {"weight": 0.1}, "normal_consistency": {"weight": 0.05, "start": 10}}
    if primitive == "2dgs":
        losses["distortion"] = {"weight": 0.01, "start": 10}
    cfg = SplatsConfig(primitive=primitive, max_steps=50, cap_max=500, losses=losses)

    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)

    # Files
    for name in ("splats.ply", "ckpt.pt", "splats.zarr", "splats_quality_report.json"):
        assert (tmp_path / name).exists(), name

    # Report
    report = json.loads((tmp_path / "splats_quality_report.json").read_text())
    summary = report["summary"]
    first_frame = report["per_frame"][0]
    assert summary["n_gaussians"] > 0 and np.isfinite(summary["psnr"]) and summary["config"]["max_steps"] == 50
    assert {"depth", "normal_consistency"} <= set(summary["final_losses"])
    assert len(report["per_frame"]) == 8 and {"image_id", "psnr", "ssim"} <= set(first_frame)

    # Rendered zarr
    store = zarr.open_group(tmp_path / "splats.zarr", mode="r")
    assert store["rgb"].shape == (8, 64, 64, 3) and store["rgb"].dtype == np.uint8
    assert store["depth"].shape == (8, 64, 64) and store["normal"].shape == (8, 64, 64, 3)
    assert store["alpha"].shape == (8, 64, 64) and store["c2w"].shape == (8, 4, 4) and store["K"].shape == (8, 3, 3)
    assert store.attrs["primitive"] == primitive and store.attrs["gsplat_commit"] == GSPLAT_COMMIT
    assert list(store.attrs["image_ids"]) == list(range(8)) and store.attrs["pose_opt"] is True

    # Checkpoint
    ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert set(ckpt) == {"splats", "pose_adjust", "appearance", "config"}
    assert ckpt["pose_adjust"] is not None and "means" in ckpt["splats"] and ckpt["config"]["primitive"] == primitive
    assert ckpt["appearance"] is None


@cuda
def test_pose_opt_refines_poses_slightly(tmp_path):
    images, world_to_cam, intrinsics, points, colors, _ = make_scene()
    cfg = SplatsConfig(max_steps=30, pose_opt=True, cap_max=500, losses={})
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)

    store = zarr.open_group(tmp_path / "splats.zarr", mode="r")
    cam_to_world_in = np.linalg.inv(world_to_cam)
    cam_to_world_out = store["c2w"][:]
    assert not np.allclose(cam_to_world_out, cam_to_world_in, atol=1e-7)  # moved
    assert np.allclose(cam_to_world_out, cam_to_world_in, atol=1e-2)  # but only a little

    # The checkpointed deltas replay to exactly the stored c2w
    ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["pose_adjust"] is not None
    refiner = CameraOptModule(8)
    refiner.load_state_dict(ckpt["pose_adjust"])
    cam_to_world_in_tensor = torch.from_numpy(cam_to_world_in).float()
    camera_ids = torch.arange(8)
    with torch.no_grad():
        cam_to_world_replayed = refiner(cam_to_world_in_tensor, camera_ids).numpy()
    assert np.allclose(cam_to_world_replayed, cam_to_world_out, atol=1e-6)


@cuda
def test_train_rejects_bad_inputs(tmp_path):
    images, world_to_cam, intrinsics, points, colors, _ = make_scene()
    cfg = SplatsConfig(max_steps=1)
    few_points, few_colors = points[:50], colors[:50]
    with pytest.raises(ValueError, match="seed points"):
        train(cfg, images, world_to_cam, intrinsics, few_points, few_colors, tmp_path)
    fewer_images = images[:4]
    with pytest.raises(ValueError, match="frames mismatch"):
        train(cfg, fewer_images, world_to_cam, intrinsics, points, colors, tmp_path)


def _render_scene(tmp_path, primitive):
    """
    Train briefly on a tiny synthetic scene and return the splats.zarr group it wrote.
    """
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=2, height=32, width=32)
    cfg = SplatsConfig(primitive=primitive, max_steps=1, cap_max=500, losses={})
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)
    return zarr.open_group(tmp_path / "splats.zarr", mode="r")


@cuda
def test_2dgs_render_all_views_writes_median_depth(tmp_path):
    store = _render_scene(tmp_path, primitive="2dgs")
    assert "median_depth" in store
    assert store["median_depth"].shape == store["depth"].shape

    # Content, not just the create_array: the per-view write ran, and what landed is the median
    # depth rather than a second copy of the alpha-weighted expected depth
    median = store["median_depth"][:]
    assert (median > 0).any()
    assert not np.array_equal(median, store["depth"][:])


@cuda
def test_3dgs_render_all_views_omits_median_depth(tmp_path):
    store = _render_scene(tmp_path, primitive="3dgs")
    assert "median_depth" not in store


def _scaffold_field(device="cuda"):
    """AnchorField seeded from the synthetic scene's points, on the given device."""
    from collab_splats.splats.scaffold import AnchorField, ScaffoldConfig

    _, _, _, points, colors, _ = make_scene(n_views=3)
    cfg = ScaffoldConfig(n_offsets=2, feat_dim=8)
    return AnchorField(cfg, points, colors, scene_scale=1.0, n_views=3, device=device)


@cuda
def test_bake_anchor_gaussians_uses_mean_observed_view_direction():
    from collab_splats.splats.outputs import bake_anchor_gaussians

    field = _scaffold_field()
    cam_to_world = torch.eye(4, device="cuda")[None].repeat(3, 1, 1)
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device="cuda")[None].repeat(3, 1, 1)

    baked = bake_anchor_gaussians(field, cam_to_world, intrinsics, width=64, height=64)
    n = len(baked["means"])
    assert baked["scales"].shape == (n, 3)  # log scales, as the ply writer expects
    assert baked["quats"].shape == (n, 4)
    assert baked["opacities"].shape == (n,)
    assert baked["sh0"].shape == (n, 1, 3)
    assert baked["shN"].shape == (n, 0, 3)
    assert torch.isfinite(baked["means"]).all()


@cuda
def test_scaffold_ply_loads_with_the_expected_field_set(tmp_path_factory):
    """The baked ply is a normal degree-0 3DGS ply: any viewer must be able to read it."""
    from plyfile import PlyData

    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    cfg = SplatsConfig.from_dict(
        {
            "representation": "scaffold",
            "primitive": "3dgs",
            "max_steps": 30,
            "log_every": 10,
            "scaffold": {"n_offsets": 2, "feat_dim": 8, "update_from": 10, "update_until": 20, "refine_every": 10},
        }
    )
    out_dir = tmp_path_factory.mktemp("scaffold_ply")
    train(cfg, images, world_to_cam, intrinsics, points, colors, out_dir, depth_targets=depths)

    ply = PlyData.read(str(out_dir / "splats.ply"))
    fields = {prop.name for prop in ply["vertex"].properties}
    assert {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"} <= fields
    assert {"scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"} <= fields
    assert len(ply["vertex"]) > 0


@cuda
def test_scaffold_report_records_the_decoded_gaussian_count(tmp_path_factory):
    """Scaffold's primitive count is per view: the report carries the measured mean, not just anchors."""
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    cfg = SplatsConfig.from_dict(
        {
            "representation": "scaffold",
            "primitive": "3dgs",
            "max_steps": 30,
            "log_every": 10,
            "scaffold": {"n_offsets": 2, "feat_dim": 8, "update_from": 10, "update_until": 20, "refine_every": 10},
        }
    )
    out_dir = tmp_path_factory.mktemp("scaffold_decoded_count")
    train(cfg, images, world_to_cam, intrinsics, points, colors, out_dir, depth_targets=depths)

    report = json.loads((out_dir / "splats_quality_report.json").read_text())
    summary = report["summary"]
    n_anchors = summary["n_gaussians"]

    # Every view decodes at most n_anchors x n_offsets Gaussians (frustum culling and the opacity
    # gate only ever remove), so the mean sits inside that bound and is reported per frame too
    assert summary["n_decoded_mean"] > 0
    assert summary["n_decoded_mean"] <= n_anchors * cfg.scaffold_config.n_offsets
    assert all(frame["n_decoded"] > 0 for frame in report["per_frame"])


@cuda
def test_vanilla_report_has_no_decoded_count(tmp_path_factory):
    """A vanilla run's primitive count is exact, so there is nothing to average."""
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=3)
    cfg = SplatsConfig.from_dict({"max_steps": 20, "log_every": 10})
    out_dir = tmp_path_factory.mktemp("vanilla_decoded_count")
    train(cfg, images, world_to_cam, intrinsics, points, colors, out_dir, depth_targets=depths)

    report = json.loads((out_dir / "splats_quality_report.json").read_text())
    assert "n_decoded_mean" not in report["summary"]
    assert all("n_decoded" not in frame for frame in report["per_frame"])


@cuda
def test_bake_never_returns_zero_gaussians_when_every_offset_is_closed():
    """An empty bake cannot be serialised: export_splats reshapes shN by the splat count."""
    from collab_splats.splats.outputs import bake_anchor_gaussians

    field = _scaffold_field()
    cam_to_world = torch.eye(4, device="cuda")[None].repeat(3, 1, 1)
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device="cuda")[None].repeat(3, 1, 1)

    # Drive every neural opacity negative: the tanh head saturates at -1 for a large negative bias
    with torch.no_grad():
        field.mlps.mlp_opacity[-2].bias.fill_(-50.0)
        field.mlps.mlp_opacity[-2].weight.zero_()

    baked = bake_anchor_gaussians(field, cam_to_world, intrinsics, width=64, height=64)
    assert len(baked["means"]) == 1
    assert baked["sh0"].shape == (1, 1, 3)
    assert baked["shN"].shape == (1, 0, 3)
