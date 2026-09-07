import dataclasses
from pathlib import Path
import numpy as np
import pytest
import torch
from collab_splats.pointcloud.feedforward.base import FeedforwardResult, BaseFeedforwardCreator
from collab_splats.pointcloud.feedforward.vggtx import unproject_and_filter_points
from collab_splats.pointcloud.utils import lift_features, reproject_pixels


########################################################
########## FeedforwardResult fields ####################
########################################################


def test_feedforward_result_has_new_fields():
    fields = {f.name for f in dataclasses.fields(FeedforwardResult)}
    assert "features" in fields
    assert "pixel_indices" in fields


def test_feedforward_result_new_fields_default_none():
    field_map = {f.name: f for f in dataclasses.fields(FeedforwardResult)}
    assert field_map["features"].default is None
    assert field_map["pixel_indices"].default is None


def test_base_creator_no_extractor_name():
    """extractor_name field removed; creators no longer own lifting."""
    fields = {f.name for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert "extractor_name" not in fields


########################################################
########## unproject_and_filter_points #################
########################################################


def _make_depth_inputs(n=3, h=8, w=8):
    """Minimal valid inputs for unproject_and_filter_points."""
    depth = np.ones((n, h, w, 1), dtype=np.float32)  # vggt expects (N, H, W, 1)
    depth_conf = np.random.rand(n, h, w).astype(np.float32)
    images = torch.zeros(n, 3, h, w)
    extrinsic = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    intrinsic = np.tile(
        np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    return depth, depth_conf, images, extrinsic, intrinsic


def test_unproject_returns_pixel_indices():
    depth, depth_conf, images, extrinsic, intrinsic = _make_depth_inputs(n=3, h=8, w=8)
    pts3d, colors, pixel_indices = unproject_and_filter_points(
        depth, depth_conf, images, extrinsic, intrinsic, conf_threshold=0.5
    )
    assert pixel_indices.shape == (len(pts3d), 3)
    assert pixel_indices.dtype == np.int32
    assert (pixel_indices[:, 0] >= 0).all() and (pixel_indices[:, 0] < 3).all()
    assert (pixel_indices[:, 1] >= 0).all() and (pixel_indices[:, 1] < 8).all()
    assert (pixel_indices[:, 2] >= 0).all() and (pixel_indices[:, 2] < 8).all()


def test_pixel_indices_align_with_colors():
    """colors[p] must come from the same pixel as pixel_indices[p]."""
    n, h, w = 2, 8, 8
    depth = np.ones((n, h, w, 1), dtype=np.float32)
    depth_conf = np.ones((n, h, w), dtype=np.float32)  # all pixels pass
    # Encode pixel identity into image: pixel (r, c) = r*10 + c across all channels
    images = torch.zeros(n, 3, h, w)
    for r in range(h):
        for c in range(w):
            images[:, :, r, c] = (r * 10 + c) / 255.0
    extrinsic = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    intrinsic = np.tile(
        np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    pts3d, colors, pixel_indices = unproject_and_filter_points(
        depth, depth_conf, images, extrinsic, intrinsic, conf_threshold=0.0
    )
    images_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    for p in range(min(20, len(pts3d))):
        fi, ri, ci = pixel_indices[p]
        np.testing.assert_array_equal(
            colors[p], images_np[fi, ri, ci],
            err_msg=f"Point {p}: color mismatch at frame={fi} row={ri} col={ci}",
        )


########################################################
########## reproject_pixels ############################
########################################################


def _make_extrinsics_intrinsics(n=2):
    extrinsics_3x4 = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    intrinsics = np.tile(
        np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    return extrinsics_3x4, intrinsics


def test_reproject_pixels_shape():
    depth = np.ones((2, 8, 8, 1), dtype=np.float32)
    pixel_indices = np.array([[0, 2, 3], [1, 4, 5], [0, 1, 1]], dtype=np.int32)
    extrinsics_3x4, intrinsics = _make_extrinsics_intrinsics(n=2)
    pts = reproject_pixels(depth, pixel_indices, extrinsics_3x4, intrinsics)
    assert pts.shape == (3, 3)
    assert pts.dtype == np.float32


def test_reproject_pixels_principal_point_zero_xy():
    """Pixel at principal point + identity extrinsics → world X=Y=0, Z=depth."""
    depth = np.ones((1, 8, 8, 1), dtype=np.float32) * 3.0
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    extrinsics_3x4 = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=np.float32)
    intrinsics = np.array([[[100, 0, 4], [0, 100, 4], [0, 0, 1]]], dtype=np.float32)
    pts = reproject_pixels(depth, pixel_indices, extrinsics_3x4, intrinsics)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 3.0], atol=1e-5)


########################################################
########## lift_features (multi-view) ##################
########################################################


def _make_lift_result(
    pts3d: np.ndarray,
    pixel_indices: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    *,
    n: int,
    h: int,
    w: int,
) -> FeedforwardResult:
    """Build a minimal FeedforwardResult sufficient for lift_features.

    Uses identity world-to-cam extrinsics and a pinhole intrinsics centered at (w/2, h/2).
    """
    extrinsics_3x4 = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1, 1))
    extrinsics_4x4 = np.concatenate([extrinsics_3x4, bottom], axis=1)
    intrinsics = np.tile(
        np.array([[100, 0, w / 2], [0, 100, h / 2], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    return FeedforwardResult(
        points=pts3d,
        colors=np.zeros((pts3d.shape[0], 3), dtype=np.uint8),
        extrinsics=extrinsics_4x4,
        intrinsics=intrinsics,
        image_paths=[Path(f"frame_{i}.png") for i in range(n)],
        original_coords=np.zeros((n, 6), dtype=np.float32),
        model_width=w,
        model_height=h,
        pixel_indices=pixel_indices,
        depth=depth,
        confidence=torch.from_numpy(conf),
    )


def test_lift_features_aggregates_across_frames():
    """Point visible in 2 frames with conf (1, 2) → weighted mean of feats (1, 2)."""
    n, h, w, D = 2, 8, 8, 4
    # Point at world (0, 0, 1); identity extrinsics → both frames project to (cx, cy) = (4, 4)
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    # Depth at the projected pixel matches z=1 in both frames → depth-consistent
    depth = np.ones((n, h, w), dtype=np.float32)
    # Distinct conf per frame: 1 in frame 0, 2 in frame 1
    conf = np.zeros((n, h, w), dtype=np.float32)
    conf[0, 4, 4] = 1.0
    conf[1, 4, 4] = 2.0
    # Feature maps: frame 0 all ones, frame 1 all twos
    feature_maps = [
        torch.full((D, h, w), 1.0, dtype=torch.float32),
        torch.full((D, h, w), 2.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps, result)

    # Weighted mean: (1*1 + 2*2) / (1 + 2) = 5/3
    expected = 5.0 / 3.0
    np.testing.assert_allclose(feats[0].numpy(), [expected] * D, atol=1e-4)


def test_lift_features_source_fallback_for_unseen():
    """Point with zero accumulated weight everywhere → source-frame sample fallback."""
    n, h, w, D = 2, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    # Depth far from z=1 in both frames → depth-consistency fails everywhere → zero weight
    depth = np.full((n, h, w), 100.0, dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    # Distinct features per frame; fallback should sample from frame 0 (source)
    feature_maps = [
        torch.full((D, h, w), 7.0, dtype=torch.float32),
        torch.full((D, h, w), 9.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps, result)

    # Source frame is 0 (pixel_indices says so) → expect feature 7.0
    np.testing.assert_allclose(feats[0].numpy(), [7.0] * D, atol=1e-4)


def test_lift_features_asserts_missing_depth():
    """Missing required field (depth) raises AssertionError with clear message."""
    n, h, w, D = 1, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    # Drop depth to trigger assert
    result = dataclasses.replace(result, depth=None)
    feature_maps = [torch.zeros(D, h, w)]

    with pytest.raises(AssertionError, match="depth"):
        lift_features(feature_maps, result)


def test_lift_features_asserts_frame_count_mismatch():
    """feature_maps length must match number of frames in extrinsics."""
    n, h, w, D = 2, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feature_maps = [torch.zeros(D, h, w)]  # only 1 map, 2 frames

    with pytest.raises(AssertionError, match="frame count"):
        lift_features(feature_maps, result)


########################################################
########## load_zarr(load_images=...) ##################
########################################################


def test_load_zarr_load_images_flag(tmp_path: Path):
    """load_images=True restores tensor; default (False) returns None."""
    n, h, w = 2, 8, 8
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    # Attach images tensor — required for round-trip
    images = torch.rand(n, 3, h, w)
    result = dataclasses.replace(result, images=images)

    store_path = tmp_path / "result.zarr"
    result.save_zarr(store_path)

    # Default: images dropped
    loaded_default = FeedforwardResult.load_zarr(store_path)
    assert loaded_default.images is None

    # Opt-in: images restored
    loaded_with_images = FeedforwardResult.load_zarr(store_path, load_images=True)
    assert loaded_with_images.images is not None
    assert loaded_with_images.images.shape == (n, 3, h, w)
    np.testing.assert_allclose(loaded_with_images.images.numpy(), images.numpy(), atol=1e-5)
