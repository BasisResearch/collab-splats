from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pytest
import torch
from PIL import Image

from collab_splats.geometry import LoopClosure
from collab_splats.pointcloud.base import BasePointcloudCreator, PointcloudResult
from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator
from collab_splats.pointcloud.utils import clean_pointcloud

# Cropped box: a full-frame box would make both Ks proportional and hide a swap
CROPPED_BOX = [0.0, 60.0, 640.0, 420.0, 640.0, 480.0]


def _tiny_result(box: list[float] = CROPPED_BOX, n: int = 3) -> PointcloudResult:
    """
    n frames on a 64x48 model grid sharing one crop box; fx != fy, full-res K derived.
    """
    model_intrinsics = np.array([[40.0, 0.0, 31.5], [0.0, 44.0, 23.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(n, axis=0)
    extrinsics[:, 0, 3] = np.arange(n, dtype=np.float32) * 0.5
    return PointcloudResult(
        points=np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.5]], dtype=np.float32),
        colors=np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=None,
        model_intrinsics=model_intrinsics[None].repeat(n, axis=0),
        image_paths=[Path(f"images/frame_{i:06d}.png") for i in range(n)],
        original_coords=np.array([box] * n, dtype=np.float32),
        model_width=64,
        model_height=48,
    )


def test_select_filters_points_and_keeps_per_frame_fields():
    """
    One mask over points and colors; pixel_indices=None passes through, per-frame fields untouched.
    """
    result = _tiny_result()
    keep = np.array([True, False, True])

    out = result.select_points(keep)

    np.testing.assert_array_equal(out.points, result.points[[0, 2]])
    np.testing.assert_array_equal(out.colors, result.colors[[0, 2]])
    assert out.pixel_indices is None
    assert out.extrinsics is result.extrinsics
    assert out.intrinsics is result.intrinsics
    assert out.model_intrinsics is result.model_intrinsics
    assert out.original_coords is result.original_coords
    assert out.image_paths == result.image_paths


def test_base_creator_is_abstract():
    with pytest.raises(TypeError):
        BasePointcloudCreator()


def test_base_creator_abstract_method_is_reconstruct():
    abstract_methods = BasePointcloudCreator.__abstractmethods__
    assert abstract_methods == {"_reconstruct"}


def test_post_init_sets_full_res_intrinsics_on_cropped_box():
    """
    intrinsics=None derives the full-res K: resize to the crop undone, then the crop origin added.
    """
    result = _tiny_result()

    # Hand-computed for CROPPED_BOX: 64x48 grid -> 640x360 crop at y=60
    # - x scale 640 / 64 = 10: fx 40 -> 400, cx 31.5 -> 315
    # - y scale 360 / 48 = 7.5: fy 44 -> 330, cy 23.5 -> 176.25, + 60 crop origin -> 236.25
    expected = np.array([[400.0, 0.0, 315.0], [0.0, 330.0, 236.25], [0.0, 0.0, 1.0]], dtype=np.float32)
    assert result.intrinsics.dtype == np.float32
    np.testing.assert_array_equal(result.intrinsics[0], expected)


def test_supplied_intrinsics_are_kept():
    """
    A caller-supplied full-res K (sfm, load_zarr) is not re-derived from the model grid.
    """
    base = _tiny_result()
    K = np.full_like(base.intrinsics, 7.0)
    result = PointcloudResult(
        points=base.points,
        colors=base.colors,
        extrinsics=base.extrinsics,
        intrinsics=K,
        model_intrinsics=base.model_intrinsics,
        image_paths=base.image_paths,
        original_coords=base.original_coords,
        model_width=base.model_width,
        model_height=base.model_height,
    )
    np.testing.assert_array_equal(result.intrinsics, K)


def test_to_colmap_cameras_carry_full_res_k():
    """
    Every exported camera is PINHOLE with the full-res K and the original frame size.
    """
    result = _tiny_result()
    recon = result.to_colmap()
    assert len(recon.images) == len(result.image_paths)

    for image_id, image in recon.images.items():
        camera = recon.cameras[image.camera_id]
        row = image_id - 1
        assert camera.model.name == "PINHOLE"
        assert image.name == result.image_paths[row].name
        np.testing.assert_array_equal(camera.calibration_matrix(), result.intrinsics[row].astype(np.float64))
        assert camera.width == int(result.original_coords[row][4])
        assert camera.height == int(result.original_coords[row][5])
    assert recon.num_points3D() == len(result.points)


def test_to_colmap_sizes_each_camera_by_its_own_frame():
    """
    Frames of different original sizes each export their own width, height and K.
    """
    # Frame 0 keeps CROPPED_BOX; frame 1 is the full frame of a smaller 320x240 source
    result = _tiny_result(n=2)
    coords = np.array([CROPPED_BOX, [0.0, 0.0, 320.0, 240.0, 320.0, 240.0]], dtype=np.float32)
    result = replace(result, original_coords=coords, intrinsics=None)
    recon = result.to_colmap()

    # Frame 1 by hand: 64x48 -> 320x240 is x5 on both axes, no crop origin
    # - fx 40 -> 200, fy 44 -> 220, cx 31.5 -> 157.5, cy 23.5 -> 117.5
    assert (recon.cameras[1].width, recon.cameras[1].height) == (640, 480)
    np.testing.assert_array_equal(recon.cameras[1].params, [400.0, 330.0, 315.0, 236.25])
    assert (recon.cameras[2].width, recon.cameras[2].height) == (320, 240)
    np.testing.assert_array_equal(recon.cameras[2].params, [200.0, 220.0, 157.5, 117.5])


def test_to_colmap_accepts_3x4_extrinsics():
    """
    (N, 3, 4) w2c extrinsics export the same poses as their (N, 4, 4) form.
    """
    result = _tiny_result()
    result = replace(result, extrinsics=result.extrinsics[:, :3, :].copy())
    recon = result.to_colmap()

    # Frame 2 sits at x = 1.0 with identity rotation (_tiny_result's 0.5 stride)
    for image_id, image in recon.images.items():
        np.testing.assert_array_equal(image.cam_from_world().matrix(), result.extrinsics[image_id - 1])
    np.testing.assert_array_equal(recon.images[3].cam_from_world().translation, [1.0, 0.0, 0.0])


def test_load_zarr_round_trips_both_intrinsics(tmp_path):
    """
    save_zarr then load_zarr keeps both Ks bit-for-bit.
    """
    path = tmp_path / "pointcloud.zarr"
    result = _tiny_result()
    result.save_zarr(path)

    loaded = PointcloudResult.load_zarr(path)
    np.testing.assert_array_equal(loaded.intrinsics, result.intrinsics)
    np.testing.assert_array_equal(loaded.model_intrinsics, result.model_intrinsics)


def test_write_ply_round_trip(tmp_path):
    """
    write_ply round-trips xyz and uint8 rgb exactly and creates missing parents.
    """
    result = _tiny_result()
    out = tmp_path / "nested" / "sparse_pc.ply"
    result.write_ply(out)

    pcd = o3d.io.read_point_cloud(str(out))
    np.testing.assert_array_equal(np.asarray(pcd.points).astype(np.float32), result.points)
    np.testing.assert_array_equal(np.round(np.asarray(pcd.colors) * 255).astype(np.uint8), result.colors)


def _outlier_result() -> PointcloudResult:
    """
    200 clustered points plus one far outlier as the last row; pixel_indices tag each row.
    """
    cluster = np.random.default_rng(0).normal(scale=0.01, size=(200, 3))
    points = np.vstack([cluster, [[50.0, 50.0, 50.0]]]).astype(np.float32)
    rows = np.arange(len(points))
    return replace(
        _tiny_result(),
        points=points,
        colors=(rows[:, None] % 256 * np.ones(3)).astype(np.uint8),
        pixel_indices=np.stack([rows % 3, rows, rows], axis=-1).astype(np.int32),
    )


def _images_dir(tmp_path: Path, n: int = 2) -> Path:
    """
    n blank keyframes under tmp_path/images.
    """
    images = tmp_path / "images"
    images.mkdir()
    for i in range(n):
        Image.new("RGB", (10, 8)).save(images / f"frame_{i:06d}.png")
    return images


@dataclass
class _EchoCreator(BaseFeedforwardCreator):
    """
    Feedforward creator whose forward returns a canned result and _postprocess passes it on.
    """

    canned: Any = None

    def _load_model(self, device: str) -> Any:
        return object()

    def _preprocess(self, paths: list[Path]) -> Any:
        return None, None

    def _forward(self, model: Any, views: Any) -> Any:
        return self.canned

    def _postprocess(self, raw_outputs: Any) -> PointcloudResult:
        return raw_outputs


def test_create_pointcloud_reads_frames_and_writes_the_model(tmp_path):
    """
    Frame files reach the backend as stems, in order; model_dir gets a COLMAP model.
    """
    creator = _EchoCreator(canned=_tiny_result(), clean=False)
    creator.create_pointcloud(_images_dir(tmp_path, n=3), tmp_path / "out", tmp_path / "model")

    assert [p.name for p in creator.image_paths] == ["frame_000000", "frame_000001", "frame_000002"]
    assert (tmp_path / "model" / "cameras.bin").exists()


def test_create_pointcloud_missing_dir(tmp_path):
    """
    A missing images dir raises before any model load.
    """
    with pytest.raises(FileNotFoundError, match="no images"):
        _EchoCreator().create_pointcloud(tmp_path / "nope", tmp_path / "out")


def test_clean_drops_the_outlier_from_every_per_point_array(tmp_path):
    """
    clean=True removes the far point from points, colors and pixel_indices together.
    """
    raw = _outlier_result()
    out = _EchoCreator(canned=raw, clean=True).create_pointcloud(_images_dir(tmp_path), tmp_path / "out")

    # Exactly the outlier row went; the three per-point arrays stay row-aligned
    assert len(out.points) == len(out.colors) == len(out.pixel_indices) == 200
    np.testing.assert_array_equal(out.points, raw.points[:200])
    np.testing.assert_array_equal(out.colors, raw.colors[:200])
    np.testing.assert_array_equal(out.pixel_indices, raw.pixel_indices[:200])


def test_cap_with_clean_off(tmp_path):
    """
    clean=False skips the outlier removal but still draws the max_points cap.
    """
    creator = _EchoCreator(canned=_outlier_result(), max_points=50, clean=False)
    out = creator.create_pointcloud(_images_dir(tmp_path), tmp_path / "out")
    assert len(out.points) == len(out.pixel_indices) == 50


def test_clean_before_the_cap(tmp_path):
    """
    Outliers never take a max_points slot: the cap lands exactly after the outlier removal.
    """
    # Noisy depth-2 surface on two 16x16 frames; five far pixels at depth 100
    n, h, w = 2, 16, 16
    depth = 2.0 + 0.01 * np.random.default_rng(0).standard_normal((n, h, w, 1))
    depth[0, [1, 4, 7, 10, 13], [2, 5, 8, 11, 14]] = 100.0
    intrinsics = np.array([[16.0, 0.0, 7.5], [0.0, 16.0, 7.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    raw = {
        "images": torch.zeros(n, 3, h, w),
        "extrinsic": np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(intrinsics, (n, 1, 1)),
        "depth": depth.astype(np.float32),
        "depth_conf": np.ones((n, h, w), dtype=np.float32),
    }
    creator = _EchoCreator(max_points=300, conf_threshold=0.0, clean=True)
    creator.image_paths = [Path(f"frame_{i:06d}") for i in range(n)]
    creator.original_coords = np.tile(np.array([0, 0, w, h, w, h], dtype=np.float32), (n, 1))

    # Raw conf cutoff 0.0 keeps all 512 pixels; the cap is well under that
    grid = BaseFeedforwardCreator._postprocess(creator, raw)
    out = clean_pointcloud(grid, remove_outliers=True, max_points=300)

    assert len(out.points) == len(out.colors) == len(out.pixel_indices) == 300
    assert out.points[:, 2].max() < 10.0


def test_loop_closure_assembled_result_is_cleaned(tmp_path):
    """
    LC's assembled result skips _postprocess but still gets the outlier removal.
    """
    creator = _EchoCreator(clean=True)
    creator.outputs = replace(_outlier_result(), pixel_indices=None)

    # The state _run_lc_loop leaves: outputs assembled from the GraphMap
    lc = LoopClosure.__new__(LoopClosure)
    lc.base = creator
    lc._lc_assembled = True
    lc.run_inference = lambda: None
    out = lc.create_pointcloud(_images_dir(tmp_path), tmp_path / "out")

    assert len(out.points) == len(out.colors) == 200
    assert out.pixel_indices is None


########################################################
########## PointcloudResult fields #####################
########################################################


def test_feedforward_result_has_new_fields():
    names = {f.name for f in fields(PointcloudResult)}
    assert "pixel_indices" in names


def test_feedforward_result_new_fields_default_none():
    field_map = {f.name: f for f in fields(PointcloudResult)}
    assert field_map["pixel_indices"].default is None


def test_load_zarr_load_images_flag(tmp_path: Path):
    """load_images=True restores tensor; default (False) returns None."""
    n, h, w = 3, 48, 64
    result = _tiny_result(n=n)
    # Attach images tensor — required for round-trip
    images = torch.rand(n, 3, h, w)
    result = replace(result, images=images)

    store_path = tmp_path / "result.zarr"
    result.save_zarr(store_path)

    # Default: images dropped
    loaded_default = PointcloudResult.load_zarr(store_path)
    assert loaded_default.images is None

    # Opt-in: images restored
    loaded_with_images = PointcloudResult.load_zarr(store_path, load_images=True)
    assert loaded_with_images.images is not None
    assert loaded_with_images.images.shape == (n, 3, h, w)
    np.testing.assert_allclose(loaded_with_images.images.numpy(), images.numpy(), atol=1e-5)
