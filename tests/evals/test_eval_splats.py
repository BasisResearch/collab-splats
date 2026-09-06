"""
eval_splats: pointcloud.zarr -> trainer inputs, and the summary row built from a quality report.
"""

import json

import numpy as np
import pytest
import zarr

from collab_splats.preproc import frames as fr
from evals.scripts.eval_splats import inputs_from_pointcloud_zarr, summarise_run


def _write_ff_zarr(path, n=2, h=4, w=6, scale=255.0, orig_hw=None):
    # orig_hw: recorded original (H, W) per frame; defaults to the model size (no rescale)
    orig_h, orig_w = (h, w) if orig_hw is None else orig_hw
    store = zarr.open_group(path, mode="w")
    images = np.random.rand(n, 3, h, w).astype(np.float32) * scale
    intrinsics = np.tile(np.array([[2.0, 0, 3.0], [0, 4.0, 5.0], [0, 0, 1]], np.float32), (n, 1, 1))
    for name, array in (
        ("images", images),
        ("depth", np.ones((n, h, w), np.float32)),
        ("confidence", np.ones((n, h, w), np.float32)),
        ("extrinsics", np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))),
        ("intrinsics", intrinsics),
        ("points", np.zeros((120, 3), np.float32)),
        ("colors", np.zeros((120, 3), np.uint8)),
        # original_coords + model_* attrs are required by FeedforwardResult.load_zarr
        ("original_coords", np.tile(np.array([0, 0, w, h, orig_w, orig_h], np.float32), (n, 1))),
    ):
        store.create_array(name, data=array)
    store.attrs["image_paths"] = [f"frame_{i:06d}.jpg" for i in range(n)]
    store.attrs["model_width"] = w
    store.attrs["model_height"] = h


def _write_images_dir(path, n, h, w):
    frames = np.random.randint(0, 255, size=(n, h, w, 3), dtype=np.uint8)
    records = [{"frame_idx": i} for i in range(n)]
    fr.write_frames(path, frames, records, {"video_path": "v"})
    return frames


def test_inputs_uint8_hwc_for_both_image_scales(tmp_path):
    for scale in (1.0, 255.0):
        _write_ff_zarr(tmp_path / f"ff_{scale}.zarr", scale=scale)
        inputs = inputs_from_pointcloud_zarr(tmp_path / f"ff_{scale}.zarr")
        assert inputs.images.shape == (2, 4, 6, 3) and inputs.images.dtype == np.uint8
        assert inputs.images.max() > 1  # [0,1] stores are rescaled, [0,255] stores are not squashed
        assert inputs.world_to_cam.shape == (2, 4, 4) and inputs.depth_targets.shape == (2, 4, 6)
        assert inputs.resolution == (4, 6)


def test_inputs_native_from_images_dir(tmp_path):
    # Native frames are 2x the model size; K must scale with them, depth targets stay model-res
    n, h, w = 2, 4, 6
    _write_ff_zarr(tmp_path / "pointcloud.zarr", n=n, h=h, w=w, orig_hw=(2 * h, 2 * w))
    frames = _write_images_dir(tmp_path / "images", n, 2 * h, 2 * w)
    inputs = inputs_from_pointcloud_zarr(tmp_path / "pointcloud.zarr")
    assert inputs.images.shape == (n, 2 * h, 2 * w, 3) and inputs.images.dtype == np.uint8
    np.testing.assert_array_equal(inputs.images, frames)
    assert inputs.resolution == (2 * h, 2 * w)
    assert np.all(inputs.intrinsics[:, 0, 0] == 2 * 2.0)
    assert np.all(inputs.intrinsics[:, 0, 2] == 2 * 3.0)
    assert np.all(inputs.intrinsics[:, 1, 1] == 2 * 4.0)
    assert np.all(inputs.intrinsics[:, 1, 2] == 2 * 5.0)
    assert inputs.depth_targets.shape == (n, h, w)

    # Explicit None forces model res even though a sibling images/ exists
    model_res = inputs_from_pointcloud_zarr(tmp_path / "pointcloud.zarr", images_dir=None)
    assert model_res.images.shape == (n, h, w, 3) and model_res.intrinsics[0, 0, 0] == 2.0


def test_inputs_native_resolution_mismatch_raises(tmp_path):
    n, h, w = 2, 4, 6
    _write_ff_zarr(tmp_path / "pointcloud.zarr", n=n, h=h, w=w, orig_hw=(2 * h, 2 * w))
    _write_images_dir(tmp_path / "images", n, 3 * h, 3 * w)
    with pytest.raises(ValueError, match="images"):
        inputs_from_pointcloud_zarr(tmp_path / "pointcloud.zarr")


def test_summarise_run_reads_report(tmp_path):
    report = {
        "summary": {
            "psnr": 30.0,
            "ssim": 0.9,
            "n_gaussians": 10,
            "seconds": 12.0,
            "final_losses": {"depth": 0.1},
            "config": {"primitive": "3dgs", "max_steps": 5},
        },
        "per_frame": [{"image_id": 0, "psnr": 30.0, "ssim": 0.9}],
    }
    (tmp_path / "splats_quality_report.json").write_text(json.dumps(report))
    row = summarise_run(tmp_path)
    assert row == {
        "primitive": "3dgs",
        "max_steps": 5,
        "psnr": 30.0,
        "ssim": 0.9,
        "n_gaussians": 10,
        "seconds": 12.0,
        "ms_per_step": 2400.0,
        "final_losses": {"depth": 0.1},
    }
