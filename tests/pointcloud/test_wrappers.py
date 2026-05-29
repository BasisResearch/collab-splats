# tests/pointcloud/test_wrappers.py
import dataclasses

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from collab_splats.pointcloud.feedforward import FeedforwardResult
from collab_splats.pointcloud.wrappers import _trim_forward_outputs


def _make_ff_result(**overrides):
    defaults = dict(
        points=np.zeros((10, 3), dtype=np.float32),
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


def _make_mock_creator(ff_result):
    """Returns a mock that duck-types as BaseFeedforwardCreator."""
    m = MagicMock()
    m.outputs = ff_result
    m.raw_outputs = {}
    m._reproject.return_value = (ff_result.points, ff_result.colors)
    return m


########################################################
########## FeedforwardResult field tests ##############
########################################################


def test_feedforward_result_new_fields_default_none():
    r = _make_ff_result()
    assert r.images is None
    assert r.confidence is None
    assert r.world_points is None


########################################################
########## BundleAdjustmentConfig tests ###############
########################################################


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


########################################################
########## BaseFeedforwardCreator field tests #########
########################################################


def test_vggtx_postprocess_populates_ba_fields():
    """VGGTXCreator._postprocess() must populate images/confidence/world_points."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator

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
    assert result.confidence is not None
    assert result.world_points is not None
    assert result.images.shape[0] == N
    assert result.confidence.shape == (N, H, W)


def test_vggtx_no_use_ba_field():
    """VGGTXCreator must not have use_ba after refactor."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    field_names = {f.name for f in dataclasses.fields(VGGTXCreator)}
    assert "use_ba" not in field_names


def test_vggtx_has_reproject():
    """VGGTXCreator must implement _reproject."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    assert hasattr(VGGTXCreator, "_reproject")


def test_mapanything_no_use_ba_field():
    """MapAnythingCreator must not have use_ba after refactor."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    field_names = {f.name for f in dataclasses.fields(MapAnythingCreator)}
    assert "use_ba" not in field_names


def test_mapanything_has_reproject():
    """MapAnythingCreator must implement _reproject."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    assert hasattr(MapAnythingCreator, "_reproject")


########################################################
########## make_creator tests #########################
########################################################


def test_make_creator_no_wrappers():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    creator = make_creator("vggtx")
    assert isinstance(creator, VGGTXCreator)


def test_make_creator_with_lc():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    creator = make_creator("vggtx", use_lc=True)
    assert isinstance(creator, LoopClosure)
    assert isinstance(creator.base, VGGTXCreator)


def test_make_creator_unknown_name():
    from collab_splats.pointcloud import make_creator
    with pytest.raises(KeyError):
        make_creator("unknown_backend")


########################################################
########## LoopClosure delegation tests ###############
########################################################


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


########################################################
########## LoopClosure.run() tests ####################
########################################################


def test_loop_closure_run_returns_feedforward_result():
    """lc.run(image_dir) returns FeedforwardResult from base.outputs."""
    from collab_splats.pointcloud.wrappers import LoopClosure

    result = _make_ff_result()
    mock_base = _make_mock_creator(result)
    mock_base.views = torch.zeros(2, 3, 8, 8)

    lc = LoopClosure(mock_base)
    returned = lc.run(Path("/fake/dir"))

    assert returned is result
    mock_base.load_model.assert_called_once()
    mock_base.setup_inference.assert_called_once()
    mock_base.postprocess.assert_called_once()


def test_loop_closure_run_no_dedup_when_raw_outputs_empty():
    """When raw_outputs has no _dedup_rows key, result passes through unchanged."""
    from collab_splats.pointcloud.wrappers import LoopClosure

    N, H, W = 2, 8, 8
    images = torch.zeros(N, 3, H, W)
    result = _make_ff_result(images=images)
    mock_base = _make_mock_creator(result)
    mock_base.raw_outputs = {}
    mock_base.views = torch.zeros(N, 3, H, W)

    lc = LoopClosure(mock_base)
    returned = lc.run(Path("/fake/dir"))

    # images tensor unchanged — same object
    assert returned.images is images


def test_loop_closure_run_dedup_applied_when_dedup_rows_present():
    """When raw_outputs has _dedup_rows, arrays with M rows are sliced to N rows."""
    from collab_splats.pointcloud.wrappers import LoopClosure

    N = 2
    # Extrinsics has N=2 rows; images/world_points have M=3 rows (merge artifact)
    dedup = np.array([0, 2])  # selects frames 0 and 2 from merged M=3
    images_merged = torch.zeros(3, 3, 8, 8)
    world_points_merged = np.zeros((3, 8, 8, 3), dtype=np.float32)

    result = _make_ff_result(
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        images=images_merged,
        world_points=world_points_merged,
    )
    mock_base = _make_mock_creator(result)
    mock_base.raw_outputs = {"_dedup_rows": dedup}
    mock_base.views = torch.zeros(N, 3, 8, 8)

    lc = LoopClosure(mock_base)
    returned = lc.run(Path("/fake/dir"))

    # images and world_points should be sliced to N rows via dedup indices
    assert returned.images.shape[0] == N
    assert returned.world_points.shape[0] == N


########################################################
########## LoopClosure.reproject() tests ##############
########################################################


def test_loop_closure_reproject_delegates_to_base():
    """lc.reproject(result) delegates to base.reproject(result)."""
    from collab_splats.pointcloud.wrappers import LoopClosure

    result = _make_ff_result()
    reprojected = _make_ff_result(points=np.ones((5, 3), dtype=np.float32))
    mock_base = _make_mock_creator(result)
    mock_base.reproject.return_value = reprojected

    lc = LoopClosure(mock_base)
    out = lc.reproject(result)

    mock_base.reproject.assert_called_once_with(result)
    assert out is reprojected


########################################################
########## _trim_forward_outputs tests ###############
########################################################


def test_trim_forward_outputs_dict_trims_arrays():
    raw = {
        "extrinsic": np.zeros((7, 3, 4)),
        "intrinsics": np.zeros((7, 3, 3)),
        "depth": np.zeros((7, 64, 64, 1)),
        "depth_conf": np.zeros((7, 64, 64)),
        "world_points": np.zeros((7, 100, 3)),
        "world_points_conf": np.zeros((7, 100)),
        "images": np.zeros((7, 3, 64, 64)),
    }
    trimmed = _trim_forward_outputs(raw, 6)
    for key, val in trimmed.items():
        assert isinstance(val, np.ndarray)
        assert val.shape[0] == 6, f"{key}: expected 6, got {val.shape[0]}"


def test_trim_forward_outputs_list_trims_list():
    raw = [{"extrinsic": np.zeros((1, 3, 4))} for _ in range(7)]
    trimmed = _trim_forward_outputs(raw, 6)
    assert len(trimmed) == 6


def test_trim_forward_outputs_no_op_when_short():
    raw = {"extrinsic": np.zeros((5, 3, 4)), "scalar": 1.0}
    trimmed = _trim_forward_outputs(raw, 6)
    assert trimmed["extrinsic"].shape[0] == 5
    assert trimmed["scalar"] == 1.0


def test_trim_forward_outputs_preserves_non_array_values():
    raw = {"extrinsic": np.zeros((7, 3, 4)), "label": "keep", "count": 42}
    trimmed = _trim_forward_outputs(raw, 6)
    assert trimmed["label"] == "keep"
    assert trimmed["count"] == 42
