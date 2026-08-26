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
