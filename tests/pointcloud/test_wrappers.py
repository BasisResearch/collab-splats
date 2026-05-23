# tests/pointcloud/test_wrappers.py
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from collab_splats.pointcloud.feedforward import FeedforwardResult


def _make_ff_result(**overrides):
    defaults = dict(
        pts3d=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        image_paths=[Path("a.jpg"), Path("b.jpg")],
        original_coords=np.zeros((2, 6), dtype=np.float32),
        model_width=224,
        model_height=224,
    )
    defaults.update(overrides)
    return FeedforwardResult(**defaults)


def test_feedforward_result_new_fields_default_none():
    r = _make_ff_result()
    assert r.images is None
    assert r.conf is None
    assert r.world_points is None


def test_bundle_adjustment_config_defaults():
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig

    cfg = BundleAdjustmentConfig()
    assert cfg.max_reproj_error == 4.0
    assert cfg.lm_steps == 40
    assert cfg.shared_camera is False
    assert cfg.min_inliers_per_frame == 64


def test_bundle_adjustment_config_custom():
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig

    cfg = BundleAdjustmentConfig(max_reproj_error=2.0, lm_steps=20)
    assert cfg.max_reproj_error == 2.0
    assert cfg.lm_steps == 20


def test_vggtx_postprocess_populates_ba_fields():
    """VGGTXCreator._postprocess() must populate images/conf/world_points."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    from unittest.mock import patch

    creator = VGGTXCreator.__new__(VGGTXCreator)
    creator.conf_threshold = 1.0
    creator.use_global_alignment = False
    creator.image_paths = [Path("a.jpg"), Path("b.jpg")]
    creator.original_coords = np.zeros((2, 6), dtype=np.float32)

    N, H, W = 2, 8, 8
    raw_outputs = {
        "depth": np.ones((N, H, W, 1), dtype=np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": torch.zeros(N, 3, H, W),
        "extrinsic": np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        "intrinsics_downsampled": np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
    }

    with patch(
        "collab_splats.pointcloud.feedforward.unproject_and_filter_points",
        return_value=(np.zeros((5, 3), dtype=np.float32), np.zeros((5, 3), dtype=np.uint8)),
    ):
        result = creator._postprocess(raw_outputs)

    assert result.images is not None
    assert result.conf is not None
    assert result.world_points is not None
    assert result.images.shape[0] == N
    assert result.conf.shape == (N, H, W)


def test_vggtx_no_use_ba_field():
    """VGGTXCreator must not have use_ba after refactor."""
    import dataclasses
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    field_names = {f.name for f in dataclasses.fields(VGGTXCreator)}
    assert "use_ba" not in field_names


def test_vggtx_has_reproject_ba():
    """VGGTXCreator must implement _reproject_ba."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    assert hasattr(VGGTXCreator, "_reproject_ba")


def test_mapanything_no_use_ba_field():
    """MapAnythingCreator must not have use_ba after refactor."""
    import dataclasses
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    field_names = {f.name for f in dataclasses.fields(MapAnythingCreator)}
    assert "use_ba" not in field_names


def test_mapanything_has_reproject_ba():
    """MapAnythingCreator must implement _reproject_ba."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    assert hasattr(MapAnythingCreator, "_reproject_ba")


def _make_mock_creator(ff_result):
    """Returns a mock that duck-types as BaseFeedforwardCreator."""
    m = MagicMock()
    m.outputs = ff_result
    m.raw_outputs = {}
    m._reproject_ba.return_value = (ff_result.pts3d, ff_result.colors)
    return m


def test_bundle_adjustment_raises_if_images_none():
    from collab_splats.pointcloud.wrappers import BundleAdjustment
    result = _make_ff_result()  # images=None by default
    mock_creator = _make_mock_creator(result)

    ba = BundleAdjustment(mock_creator)
    with pytest.raises(ValueError, match="images"):
        ba.reconstruct("/fake/dir", "/fake/out")


def test_bundle_adjustment_calls_extract_tracks_and_run_ba():
    from collab_splats.pointcloud.wrappers import BundleAdjustment

    N, H, W = 2, 8, 8
    result = _make_ff_result(
        images=torch.zeros(N, 3, H, W),
        conf=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )
    mock_creator = _make_mock_creator(result)
    mock_creator.build_colmap.return_value = MagicMock()

    fake_tracks = np.zeros((N, 10, 2), dtype=np.float32)
    fake_vis = np.ones((N, 10), dtype=np.float32)
    fake_pts = np.zeros((10, 3), dtype=np.float32)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.wrappers.extract_tracks_vggsfm",
               return_value=(fake_tracks, fake_vis, fake_pts)) as mock_tracks, \
         patch("collab_splats.pointcloud.wrappers.run_bundle_adjustment",
               return_value=(fake_pts, refined_ext, refined_intr)) as mock_ba:
        ba = BundleAdjustment(mock_creator)
        ba.reconstruct("/fake/dir", "/fake/out")

    mock_tracks.assert_called_once()
    mock_ba.assert_called_once()
    mock_creator._reproject_ba.assert_called_once_with({}, refined_ext, refined_intr)
    mock_creator.build_colmap.assert_called_once()


def test_bundle_adjustment_outputs_proxies_to_base():
    """eval_gt.py:66 reads creator.outputs after reconstruct — BundleAdjustment
    must proxy outputs to base, mirroring LoopClosure. Without this, eval harness
    raises AttributeError on the BA condition."""
    from collab_splats.pointcloud.wrappers import BundleAdjustment

    result = _make_ff_result()
    mock_creator = _make_mock_creator(result)

    ba = BundleAdjustment(mock_creator)
    assert ba.outputs is result

    new_result = _make_ff_result()
    ba.outputs = new_result
    assert mock_creator.outputs is new_result


def test_bundle_adjustment_raw_outputs_proxies_to_base():
    """raw_outputs proxy mirrors LoopClosure for consumer symmetry."""
    from collab_splats.pointcloud.wrappers import BundleAdjustment

    mock_creator = _make_mock_creator(_make_ff_result())
    sentinel = {"foo": "bar"}
    mock_creator.raw_outputs = sentinel

    ba = BundleAdjustment(mock_creator)
    assert ba.raw_outputs is sentinel

    new_raw = {"baz": 1}
    ba.raw_outputs = new_raw
    assert mock_creator.raw_outputs is new_raw


def test_bundle_adjustment_config_passed_to_run_ba():
    from collab_splats.pointcloud.wrappers import BundleAdjustment
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig

    N, H, W = 2, 8, 8
    result = _make_ff_result(
        images=torch.zeros(N, 3, H, W),
        conf=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )
    mock_creator = _make_mock_creator(result)
    mock_creator.build_colmap.return_value = MagicMock()

    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.wrappers.extract_tracks_vggsfm",
               return_value=(np.zeros((N,10,2)), np.ones((N,10)), np.zeros((10,3)))), \
         patch("collab_splats.pointcloud.wrappers.run_bundle_adjustment",
               return_value=(np.zeros((10,3)), refined_ext, refined_intr)) as mock_ba:
        cfg = BundleAdjustmentConfig(max_reproj_error=2.0, lm_steps=10)
        BundleAdjustment(mock_creator, config=cfg).reconstruct("/a", "/b")

    _, kwargs = mock_ba.call_args
    assert kwargs["max_reproj_error"] == 2.0
    assert kwargs["lm_steps"] == 10


def test_loop_closure_constructor_defaults():
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    assert lc.base is mock_base
    assert isinstance(lc.config, LoopClosureConfig)


def test_loop_closure_constructor_custom_config():
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig

    cfg = LoopClosureConfig(submap_size=10)
    mock_base = MagicMock()
    lc = LoopClosure(mock_base, config=cfg)
    assert lc.config.submap_size == 10


def test_loop_closure_forwards_load_model():
    from collab_splats.pointcloud.wrappers import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.load_model()
    mock_base.load_model.assert_called_once()


def test_make_creator_no_wrappers():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    creator = make_creator("vggtx")
    assert isinstance(creator, VGGTXCreator)


def test_make_creator_with_lc():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.wrappers import LoopClosure

    creator = make_creator("vggtx", use_lc=True)
    assert isinstance(creator, LoopClosure)
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    assert isinstance(creator.base, VGGTXCreator)


def test_make_creator_with_ba():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.wrappers import BundleAdjustment

    creator = make_creator("vggtx", use_ba=True)
    assert isinstance(creator, BundleAdjustment)


def test_bundle_adjustment_uses_reproject_pixels_when_pixel_indices_set():
    """When pixel_indices is set, BA uses reproject_pixels (deterministic) and
    preserves colors/features from pre-BA result without calling _reproject_ba."""
    from collab_splats.pointcloud.wrappers import BundleAdjustment

    N, H, W = 2, 8, 8
    pixel_indices = np.array([[0, 2, 3], [1, 4, 5], [0, 1, 1]], dtype=np.int32)
    features = np.ones((3, 4), dtype=np.float32) * 7.0
    pre_ba_colors = np.arange(9, dtype=np.uint8).reshape(3, 3)
    depth = np.ones((N, H, W, 1), dtype=np.float32) * 2.0
    result = _make_ff_result(
        pts3d=np.zeros((3, 3), dtype=np.float32),
        colors=pre_ba_colors,
        features=features,
        pixel_indices=pixel_indices,
        images=torch.zeros(N, 3, H, W),
        conf=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )
    mock_creator = _make_mock_creator(result)
    mock_creator.raw_outputs = {"depth": depth}
    mock_creator.build_colmap.return_value = MagicMock()

    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]]), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.wrappers.extract_tracks_vggsfm",
               return_value=(np.zeros((N, 10, 2)), np.ones((N, 10)), np.zeros((10, 3)))), \
         patch("collab_splats.pointcloud.wrappers.run_bundle_adjustment",
               return_value=(np.zeros((10, 3)), refined_ext, refined_intr)):
        BundleAdjustment(mock_creator).reconstruct("/fake/dir", "/fake/out")

    # _reproject_ba must NOT be called when pixel_indices is set
    mock_creator._reproject_ba.assert_not_called()

    # colors and features must be preserved unchanged from pre-BA result
    final = mock_creator.outputs
    np.testing.assert_array_equal(final.colors, pre_ba_colors)
    np.testing.assert_array_equal(final.features, features)
    np.testing.assert_array_equal(final.pixel_indices, pixel_indices)


def test_make_creator_with_lc_and_ba():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.wrappers import BundleAdjustment, LoopClosure

    creator = make_creator("vggtx", use_lc=True, use_ba=True)
    assert isinstance(creator, BundleAdjustment)
    assert isinstance(creator.base, LoopClosure)


def test_make_creator_unknown_name():
    from collab_splats.pointcloud import make_creator
    with pytest.raises(KeyError):
        make_creator("unknown_backend")


def test_loop_closure_forwards_setup_inference():
    from collab_splats.pointcloud.wrappers import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.setup_inference(Path("/fake"))
    mock_base.setup_inference.assert_called_once_with(Path("/fake"))


def test_loop_closure_forwards_postprocess_and_build_colmap():
    from collab_splats.pointcloud.wrappers import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.postprocess()
    mock_base.postprocess.assert_called_once()
    lc.build_colmap(Path("/out"))
    mock_base.build_colmap.assert_called_once_with(Path("/out"))


def test_loop_closure_run_inference_falls_back_to_base_when_too_few_frames():
    """When fewer frames than submap_size, falls back to base.run_inference()."""
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig

    mock_base = MagicMock()
    mock_base.views = torch.zeros(3, 3, 8, 8)  # 3 frames

    cfg = LoopClosureConfig(submap_size=20)
    lc = LoopClosure(mock_base, config=cfg)
    lc.run_inference()
    mock_base.run_inference.assert_called_once()


def test_bundle_adjustment_wraps_loop_closure():
    """BundleAdjustment can wrap LoopClosure without isinstance checks."""
    from collab_splats.pointcloud.wrappers import BundleAdjustment, LoopClosure

    N, H, W = 2, 8, 8
    result = _make_ff_result(
        images=torch.zeros(N, 3, H, W),
        conf=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )
    inner_creator = MagicMock()
    inner_creator.views = torch.zeros(N, 3, H, W)  # N < default submap_size=20 → falls back
    lc = LoopClosure(inner_creator)
    lc.base.outputs = result
    lc.base.raw_outputs = {}
    lc.base._reproject_ba.return_value = (result.pts3d, result.colors)
    lc.base.build_colmap.return_value = MagicMock()

    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.wrappers.extract_tracks_vggsfm",
               return_value=(np.zeros((N, 10, 2)), np.ones((N, 10)), np.zeros((10, 3)))), \
         patch("collab_splats.pointcloud.wrappers.run_bundle_adjustment",
               return_value=(np.zeros((10, 3)), refined_ext, refined_intr)):
        BundleAdjustment(lc).reconstruct("/a", "/b")

    lc.base.build_colmap.assert_called_once()
