import dataclasses
import unittest.mock as mock
import numpy as np
import torch
from collab_splats.pointcloud.feedforward.base import FeedforwardResult, BaseFeedforwardCreator
from collab_splats.pointcloud.feedforward.vggtx import unproject_and_filter_points
from collab_splats.pointcloud.utils import lift_features, reproject_pixels


def test_feedforward_result_has_new_fields():
    fields = {f.name for f in dataclasses.fields(FeedforwardResult)}
    assert "features" in fields
    assert "pixel_indices" in fields


def test_feedforward_result_new_fields_default_none():
    field_map = {f.name: f for f in dataclasses.fields(FeedforwardResult)}
    assert field_map["features"].default is None
    assert field_map["pixel_indices"].default is None


def test_base_creator_has_extractor_name():
    fields = {f.name for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert "extractor_name" in fields
    field_map = {f.name: f for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert field_map["extractor_name"].default is None


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
    depth = np.ones((n, h, w, 1), dtype=np.float32)  # vggt expects (N, H, W, 1)
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
    # principal point is cx=cy=4; pixel (row=4, col=4) → x_cam = y_cam = 0
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    extrinsics_3x4 = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=np.float32)
    intrinsics = np.array([[[100, 0, 4], [0, 100, 4], [0, 0, 1]]], dtype=np.float32)
    pts = reproject_pixels(depth, pixel_indices, extrinsics_3x4, intrinsics)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 3.0], atol=1e-5)


class _FakeExtractor:
    """Minimal extractor stub: returns constant (D=4) feature tensor per image."""
    patch_size = 2

    def eval(self):
        return self

    def forward(self, images):
        D = 4
        results = []
        for img in images:
            w_px, h_px = img.size  # PIL Image: (width, height)
            H_p, W_p = h_px // self.patch_size, w_px // self.patch_size
            results.append(torch.ones(D, H_p, W_p))
        return results


def test_lift_features_shape():
    n, h, w = 2, 8, 8
    images = torch.zeros(n, 3, h, w)
    pixel_indices = np.array([[0, 2, 4], [1, 6, 2], [0, 0, 0]], dtype=np.int32)
    with mock.patch(
        "collab_splats.pointcloud.utils.BaseFeatureExtractor"
    ) as MockBase:
        MockBase.get.return_value = lambda **_: _FakeExtractor()
        feats = lift_features(images, pixel_indices, extractor_name="fake", device="cpu")
    assert feats.shape == (3, 4)
    assert feats.dtype == np.float32


def test_lift_features_frame_alignment():
    """Points from frame 0 get value 1.0; points from frame 1 get value 2.0."""
    n, h, w = 2, 8, 8
    images = torch.zeros(n, 3, h, w)
    pixel_indices = np.array([[0, 2, 4], [1, 6, 2]], dtype=np.int32)

    class _CountingExtractor:
        patch_size = 2
        _count = 0

        def eval(self):
            return self

        def forward(self, imgs):
            D = 4
            results = []
            for img in imgs:
                w_px, h_px = img.size
                H_p, W_p = h_px // self.patch_size, w_px // self.patch_size
                feat = torch.full((D, H_p, W_p), float(self._count + 1))
                self._count += 1
                results.append(feat)
            return results

    instance = _CountingExtractor()
    with mock.patch(
        "collab_splats.pointcloud.utils.BaseFeatureExtractor"
    ) as MockBase:
        MockBase.get.return_value = lambda **_: instance
        feats = lift_features(images, pixel_indices, extractor_name="fake", device="cpu")
    np.testing.assert_allclose(feats[0], [1.0, 1.0, 1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(feats[1], [2.0, 2.0, 2.0, 2.0], atol=1e-5)
