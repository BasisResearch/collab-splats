"""
eval_splats: pointcloud.zarr -> trainer inputs, and the summary row built from a quality report.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from collab_splats.mesh.io import upsample_depths
from collab_splats.preproc import frames as fr
from collab_splats.splats.utils import prepare_target
from evals.scripts.eval_splats import (
    _model_res_images,
    _native_images_and_intrinsics,
    inputs_from_pointcloud_zarr,
    summarise_run,
)


def _write_ff_zarr(path, n=2, h=4, w=6, scale=1.0, orig_hw=None):
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
        ("original_coords", np.tile(np.array([0, 0, orig_w, orig_h, orig_w, orig_h], np.float32), (n, 1))),
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


def test_inputs_uint8_hwc_from_a_0_1_zarr(tmp_path):
    """A [0, 1] zarr is rounded to uint8 HWC; a [0, 255] one raises instead of being guessed at."""
    _write_ff_zarr(tmp_path / "ff.zarr")
    inputs = inputs_from_pointcloud_zarr(tmp_path / "ff.zarr")
    assert inputs.images.shape == (2, 4, 6, 3) and inputs.images.dtype == np.uint8
    stored = zarr.open_group(tmp_path / "ff.zarr", mode="r")["images"][:]
    np.testing.assert_array_equal(inputs.images, np.rint(stored.transpose(0, 2, 3, 1) * 255.0))
    assert inputs.world_to_cam.shape == (2, 4, 4) and inputs.depth_targets.shape == (2, 4, 6)
    assert inputs.resolution == (4, 6)

    # Pre-a157421 [0, 255] store: a named error, not a silent pass-through
    _write_ff_zarr(tmp_path / "ff_255.zarr", scale=255.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        inputs_from_pointcloud_zarr(tmp_path / "ff_255.zarr")


def test_inputs_native_from_images_dir(tmp_path):
    # Native frames are 2x the model size; K and depth targets must both follow them
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
    assert inputs.depth_targets.shape == (n, 2 * h, 2 * w)

    # Explicit None forces model res even though a sibling images/ exists
    model_res = inputs_from_pointcloud_zarr(tmp_path / "pointcloud.zarr", images_dir=None)
    assert model_res.images.shape == (n, h, w, 3) and model_res.intrinsics[0, 0, 0] == 2.0
    assert model_res.depth_targets.shape == (n, h, w)


def test_inputs_native_resolution_mismatch_raises(tmp_path):
    n, h, w = 2, 4, 6
    _write_ff_zarr(tmp_path / "pointcloud.zarr", n=n, h=h, w=w, orig_hw=(2 * h, 2 * w))
    _write_images_dir(tmp_path / "images", n, 3 * h, 3 * w)
    with pytest.raises(ValueError, match="images"):
        inputs_from_pointcloud_zarr(tmp_path / "pointcloud.zarr")


def test_native_intrinsics_undo_the_crop(tmp_path):
    """Cropped box: native K is K_model / s + tl, not K_model * orig / model."""
    # Non-square model grid + a matching non-square crop: a swapped (w, h) model_hw
    # changes sx/sy independently, so this catches an h/w mix-up a square grid can't
    model_w, model_h = 16, 8
    orig_w, orig_h = 32, 64
    box = np.array([[0, 16, 32, 32, orig_w, orig_h]] * 2, dtype=np.float32)  # 32x16 crop, tl_y = 16
    K_model = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1]], dtype=np.float32)
    _write_images_dir(tmp_path / "images", 2, orig_h, orig_w)
    result = SimpleNamespace(
        image_paths=[Path("frame_000000.png"), Path("frame_000001.png")],
        original_coords=box,
        model_width=model_w,
        model_height=model_h,
        intrinsics=np.stack([K_model, K_model]),
    )

    _, K_native = _native_images_and_intrinsics(result, tmp_path / "images", tmp_path / "pointcloud.zarr")

    np.testing.assert_allclose(K_native[0], [[40, 0, 16], [0, 40, 32], [0, 0, 1]], rtol=1e-6)
    assert K_native.dtype == np.float32


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


def test_native_depth_targets_lift_through_the_crop_box(tmp_path):
    """
    Cropped zarr + native frames: depth fills only its crop box of the frame, never the whole frame.

    - model 8x4 into a 20x10 frame through a 16x8 crop at offset (2, 1)
    - column-ramp depth, so a plain stretch and a crop-aware lift disagree on position
    """
    n, h, w = 2, 4, 8
    _write_ff_zarr(tmp_path / "pointcloud.zarr", n=n, h=h, w=w)

    # Overwrite the default full-frame box and unit depth with a real crop and a column ramp
    box = np.array([2, 1, 18, 9, 20, 10], np.float32)
    ramp = np.tile(np.arange(1, w + 1, dtype=np.float32), (n, h, 1))
    store = zarr.open_group(tmp_path / "pointcloud.zarr", mode="a")
    store["original_coords"][:] = np.tile(box, (n, 1))
    store["depth"][:] = ramp
    frames = _write_images_dir(tmp_path / "images", n, 10, 20)

    inputs = inputs_from_pointcloud_zarr(tmp_path / "pointcloud.zarr", conf_percentile=None)

    # What the trainer supervises with: each view's target after train()'s own per-view resize
    seen = np.stack(
        [prepare_target(frames[i], inputs.depth_targets[i], "cpu")["depth"][0, ..., 0].numpy() for i in range(n)]
    )

    # Outside the crop box there is no target; a full-frame stretch fills it
    inside = np.zeros((10, 20), bool)
    inside[1:9, 2:18] = True
    assert np.all(seen[:, ~inside] == 0)
    assert np.all(seen[:, inside] > 0)

    # Inside, the values are the mesh stage's lift through the same box
    np.testing.assert_allclose(seen, upsample_depths(ramp, frames, np.tile(box[:4], (n, 1))))


def test_model_res_images_rejects_a_0_255_zarr():
    """The <=1.0 guess read a [0, 255] zarr as already scaled; now it is a named error."""
    result = SimpleNamespace(images=torch.full((1, 3, 2, 2), 200.0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _model_res_images(result)
