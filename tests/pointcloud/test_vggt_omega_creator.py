"""Tests for VGGTOmegaCreator.

All tests mock vggt_omega.models and checkpoint loading — no GPU or HF download required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator
from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator
from tests.pointcloud.conftest import _frame_files, _omega_boxes

########################################################################
########## Helpers #####################################################
########################################################################


def _make_raw_outputs(n: int = 2, h: int = 4, w: int = 4) -> dict:
    """Minimal raw_outputs dict matching VGGTOmegaCreator._forward output format."""
    return {
        "images": torch.zeros(n, 3, h, w),
        "extrinsic": np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        "depth": np.ones((n, h, w, 1), dtype=np.float32),
        "depth_conf": np.ones((n, h, w), dtype=np.float32) * 100.0,
    }


def _make_mock_model(n_frames: int = 2, num_heads: int = 2, n_blocks: int = 2, h: int = 4, w: int = 4):
    """VGGTOmega-shaped mock with real qkv linears so hooks fire."""
    total_dim = num_heads * 8  # head_dim=8
    n_tokens = 10

    # Build inter_frame_blocks with real qkv linears
    inter_blocks = []
    for _ in range(n_blocks):
        block = MagicMock()
        block.attn.num_heads = num_heads
        block.attn.qkv = nn.Linear(total_dim, total_dim * 3, bias=False)
        inter_blocks.append(block)

    def mock_forward(images):
        # Trigger inter_frame_blocks[-1].attn.qkv so the hook fires
        B = 1
        x = torch.randn(B, n_tokens, total_dim)
        inter_blocks[-1].attn.qkv(x)
        # Verify path decodes pose_enc AND depth/depth_conf from the same forward;
        # depth carries the trailing channel dim: (B, S, H, W, 1)
        return {
            "pose_enc": torch.zeros(B, n_frames, 9),
            "depth": torch.ones(B, n_frames, h, w, 1),
            "depth_conf": torch.ones(B, n_frames, h, w) * 100.0,
            "images": images if images.ndim == 5 else images.unsqueeze(0),
        }

    model = MagicMock()
    model.aggregator.inter_frame_blocks = inter_blocks
    model.side_effect = mock_forward
    param = nn.Parameter(torch.zeros(1))
    model.parameters = lambda: iter([param])
    return model


########################################################################
########## Structural tests ############################################
########################################################################


def test_vggt_omega_creator_is_feedforward_creator():
    """VGGTOmegaCreator inherits from BaseFeedforwardCreator."""
    assert issubclass(VGGTOmegaCreator, BaseFeedforwardCreator)


def test_vggt_omega_creator_defaults():
    """Default params match the 512-res standard checkpoint."""
    c = VGGTOmegaCreator()
    assert c.model_path is None
    assert c.resolution == 512
    assert c.resize_mode == "balanced"
    assert c.conf_threshold == 50.0


def test_vggt_omega_creator_missing_image_dir_raises(tmp_path):
    """create_pointcloud raises FileNotFoundError for nonexistent image dir."""
    c = VGGTOmegaCreator()
    with pytest.raises(FileNotFoundError):
        c.create_pointcloud(tmp_path / "nonexistent", tmp_path / "out", tmp_path / "model")


########################################################################
########## Omega crop boxes (VGGTOmegaCreator._preprocess) #############
########################################################################


def test_omega_crop_box_normal_ar():
    """Square image — no crop; coords are full-frame."""
    coords = _omega_boxes([(64, 64)])
    assert coords.shape == (1, 6)
    assert coords.dtype == np.float32
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    assert tl_x == pytest.approx(0.0)
    assert tl_y == pytest.approx(0.0)
    assert cr_x == pytest.approx(64.0)
    assert cr_y == pytest.approx(64.0)
    assert orig_w == pytest.approx(64.0)
    assert orig_h == pytest.approx(64.0)


def test_omega_crop_box_tall_image_crops_height():
    """Tall image (AR > 2.0) — height is center-cropped."""
    coords = _omega_boxes([(100, 400)])  # AR = 4.0 > 2.0
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    # crop_h = 100 * 2.0 = 200; tl_y = (400 - 200) / 2 = 100; cr_y = 100 + 200 = 300
    assert tl_x == pytest.approx(0.0)
    assert tl_y == pytest.approx(100.0)
    assert cr_x == pytest.approx(100.0)
    assert cr_y == pytest.approx(300.0)
    assert orig_w == pytest.approx(100.0)
    assert orig_h == pytest.approx(400.0)


def test_omega_crop_box_wide_image_crops_width():
    """Wide image (AR < 0.5) — width is center-cropped."""
    coords = _omega_boxes([(400, 100)])  # AR = 0.25 < 0.5
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    # crop_w = 100 / 0.5 = 200; tl_x = (400 - 200) / 2 = 100; cr_x = 100 + 200 = 300
    assert tl_x == pytest.approx(100.0)
    assert tl_y == pytest.approx(0.0)
    assert cr_x == pytest.approx(300.0)
    assert cr_y == pytest.approx(100.0)


def test_omega_crop_box_multiple_images():
    """Multiple images — coords shape (N, 6)."""
    coords = _omega_boxes([(64, 64), (100, 400), (400, 100)])
    assert coords.shape == (3, 6)
    assert coords.dtype == np.float32


@pytest.mark.parametrize(
    "size, box",
    [((1001, 400), [100, 0, 900, 400, 1001, 400]), ((300, 1001), [0, 200, 300, 800, 300, 1001])],
)
def test_omega_crop_box_is_upstream_integer_crop(size, box):
    """An odd margin floors to upstream's integer offset, not a half-pixel float one."""
    np.testing.assert_array_equal(_omega_boxes([size])[0], box)


########################################################################
########## _preprocess #################################################
########################################################################


def test_preprocess_returns_correct_shapes(tmp_path):
    """_preprocess returns (views [N,3,H,W], original_coords [N,6])."""
    paths = _frame_files([np.full((64, 64, 3), i * 80, dtype=np.uint8) for i in range(3)], tmp_path)

    creator = VGGTOmegaCreator(resolution=64)
    with patch(
        "collab_splats.pointcloud.feedforward.vggt_omega.load_and_preprocess_images",
        return_value=torch.zeros(3, 3, 64, 64),
    ):
        views, original_coords = creator._preprocess(paths)

    assert views.shape == (3, 3, 64, 64)
    assert original_coords.shape == (3, 6)
    assert original_coords.dtype == np.float32


########################################################################
########## _forward ####################################################
########################################################################


def test_forward_output_keys(tmp_path):
    """_forward returns dict with required keys for downstream processing."""
    n, h, w = 2, 8, 8

    mock_model = MagicMock()
    mock_model.side_effect = lambda imgs: {
        "pose_enc": torch.zeros(1, n, 9),
        "depth": torch.ones(1, n, h, w),
        "depth_conf": torch.ones(1, n, h, w) * 80.0,
        "images": imgs if imgs.ndim == 5 else imgs.unsqueeze(0),
    }
    param = nn.Parameter(torch.zeros(1))
    mock_model.parameters = lambda: iter([param])

    creator = VGGTOmegaCreator()
    creator.original_coords = np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32)

    views = torch.zeros(n, 3, h, w)
    with patch(
        "collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
        return_value=(torch.zeros(1, n, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, n, 3, 3)),
    ):
        raw = creator._forward(mock_model, views)

    required_keys = {"images", "extrinsic", "intrinsics", "depth", "depth_conf"}
    assert required_keys.issubset(raw.keys())


def test_forward_extrinsic_shape(tmp_path):
    """extrinsic in raw_outputs is (N, 3, 4) float32 numpy."""
    n, h, w = 3, 8, 8

    mock_model = MagicMock()
    mock_model.side_effect = lambda imgs: {
        "pose_enc": torch.zeros(1, n, 9),
        "depth": torch.ones(1, n, h, w),
        "depth_conf": torch.ones(1, n, h, w),
        "images": imgs if imgs.ndim == 5 else imgs.unsqueeze(0),
    }
    param = nn.Parameter(torch.zeros(1))
    mock_model.parameters = lambda: iter([param])

    creator = VGGTOmegaCreator()
    creator.original_coords = np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32)

    ext_tensor = torch.zeros(1, n, 3, 4)
    intr_tensor = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, n, 3, 3).contiguous()
    with patch(
        "collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera", return_value=(ext_tensor, intr_tensor)
    ):
        raw = creator._forward(mock_model, torch.zeros(n, 3, h, w))

    assert raw["extrinsic"].shape == (n, 3, 4)
    assert raw["extrinsic"].dtype == np.float32
    assert raw["intrinsics"].shape == (n, 3, 3)


########################################################################
########## _postprocess ################################################
########################################################################


def test_postprocess_returns_feedforward_result(tmp_path):
    """_postprocess returns a PointcloudResult with correct shapes."""
    n = 2
    image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(n)]
    raw = _make_raw_outputs(n)

    creator = VGGTOmegaCreator()
    creator.image_paths = image_paths
    creator.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n, 1))  # full-frame box
    creator.model = MagicMock()
    creator.model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])

    result = creator._postprocess(raw)

    # Constant confidence: the percentile cutoff keeps every pixel of the 2x4x4 grid
    assert isinstance(result, PointcloudResult)
    assert result.points.shape == (n * 4 * 4, 3)
    assert result.colors.shape == (n * 4 * 4, 3)
    assert result.extrinsics.shape == (n, 4, 4)
    assert result.intrinsics.shape == (n, 3, 3)
    assert result.model_width == 4
    assert result.model_height == 4
    assert result.image_paths == image_paths


def test_postprocess_world_points_populated(tmp_path):
    """world_points always populated (needed by BundleAdjustment wrapper)."""
    n, h, w = 2, 4, 4
    raw = _make_raw_outputs(n, h, w)

    creator = VGGTOmegaCreator()
    creator.image_paths = [tmp_path / f"{i}.jpg" for i in range(n)]
    creator.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n, 1))  # full-frame box
    creator.model = MagicMock()
    creator.model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])

    result = creator._postprocess(raw)

    assert result.world_points is not None
    assert result.world_points.shape == (n, h, w, 3)
    assert result.confidence is not None
    assert result.images is not None


########################################################################
########## _load_model #################################################
########################################################################


def test_load_model_with_explicit_path(tmp_path):
    """_load_model with model_path skips load_hf_weights."""
    ckpt_path = tmp_path / "fake.pt"
    fake_state = {"aggregator.depth": torch.tensor(1.0)}
    torch.save(fake_state, ckpt_path)

    creator = VGGTOmegaCreator(model_path=str(ckpt_path))

    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    with (
        patch(
            "collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega", return_value=mock_model_instance
        ) as mock_cls,
        patch("collab_splats.pointcloud.feedforward.vggt_omega.load_hf_weights") as mock_hf,
    ):
        creator._load_model("cpu")

    mock_hf.assert_not_called()
    mock_cls.assert_called_once()
    mock_model_instance.load_state_dict.assert_called_once()
    mock_model_instance.eval.assert_called_once()


def test_load_model_nonexistent_path_raises(tmp_path):
    """_load_model raises FileNotFoundError for missing checkpoint."""
    creator = VGGTOmegaCreator(model_path=str(tmp_path / "nonexistent.pt"))
    with pytest.raises(FileNotFoundError):
        creator._load_model("cpu")


def test_load_model_without_path_calls_hf_download(tmp_path):
    """_load_model with model_path=None downloads from HuggingFace."""
    ckpt_path = tmp_path / "downloaded.pt"
    torch.save({}, ckpt_path)

    creator = VGGTOmegaCreator()

    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    with (
        patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega", return_value=mock_model_instance),
        patch(
            "collab_splats.pointcloud.feedforward.vggt_omega.load_hf_weights", return_value=str(ckpt_path)
        ) as mock_hf,
    ):
        creator._load_model("cpu")

    mock_hf.assert_called_once_with("facebook/VGGT-Omega", "vggt_omega_1b_512.pt")


def test_load_model_keeps_fp32_params(tmp_path):
    """_load_model must NOT cast the model to bf16/fp16.

    VGGTOmega disables autocast for CameraHead/DenseHead and casts their inputs to fp32,
    so bf16/fp16 params crash the head LayerNorms with
    "expected scalar type Float but found BFloat16".
    """
    ckpt_path = tmp_path / "fake.pt"
    torch.save({}, ckpt_path)
    creator = VGGTOmegaCreator(model_path=str(ckpt_path))

    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega", return_value=mock_model_instance):
        creator._load_model("cpu")

    # No .to() call may pass a low-precision dtype
    for call in mock_model_instance.to.call_args_list:
        dtype = call.kwargs.get("dtype")
        if dtype is None and len(call.args) > 1:
            dtype = call.args[1]
        assert dtype not in (
            torch.bfloat16,
            torch.float16,
        ), f"_load_model cast model to {dtype}; VGGTOmega heads require fp32 params"


########################################################################
########## extract_intermediate_features ###############################
########################################################################


def test_extract_intermediate_features_returns_q_k_poses():
    """Hook fires on inter_frame_blocks[-1]; returns q, k, poses."""
    model = _make_mock_model(n_frames=2, h=16, w=16, num_heads=2, n_blocks=2)
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.model = model

    with patch(
        "collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
        return_value=(torch.zeros(1, 2, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 2, 3, 3)),
    ):
        result = creator.extract_intermediate_features(torch.zeros(2, 3, 16, 16), layer_index=-1)

    assert "q" in result and "k" in result
    assert "poses" in result
    assert result["poses"].shape == (2, 4, 4)
    assert result["poses"].dtype == np.float32
    # Geometry decoded from the same forward: unprojected depth + confidence
    assert result["world_points"].shape == (2, 16, 16, 3)
    assert result["world_points"].dtype == np.float32
    assert result["conf"].shape == (2, 16, 16)
    assert result["conf"].dtype == np.float32


def test_extract_intermediate_features_hook_removed_after_call():
    """Hook is removed after successful call."""
    model = _make_mock_model(n_frames=2, num_heads=2, n_blocks=2)
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.model = model

    block = model.aggregator.inter_frame_blocks[-1]
    assert len(block.attn.qkv._forward_hooks) == 0

    with patch(
        "collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
        return_value=(torch.zeros(1, 2, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 2, 3, 3)),
    ):
        creator.extract_intermediate_features(torch.zeros(2, 3, 4, 4), layer_index=-1)

    assert len(block.attn.qkv._forward_hooks) == 0


def test_extract_intermediate_features_hook_removed_on_error():
    """Hook is removed even when model forward raises."""
    model = _make_mock_model(n_frames=2, num_heads=2, n_blocks=2)
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)

    def boom(images):
        raise RuntimeError("simulated failure")

    model.side_effect = boom
    creator.model = model

    block = model.aggregator.inter_frame_blocks[-1]
    with pytest.raises(RuntimeError, match="simulated failure"):
        with patch(
            "collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
            return_value=(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 3)),
        ):
            creator.extract_intermediate_features(torch.zeros(2, 3, 4, 4), layer_index=-1)

    assert len(block.attn.qkv._forward_hooks) == 0


########################################################################
########## resolution / resize_mode ####################################
########################################################################


def test_invalid_resize_mode_raises():
    """__post_init__ raises ValueError for unknown resize_mode."""
    with pytest.raises(ValueError, match="resize_mode"):
        VGGTOmegaCreator(resize_mode="bogus")


def test_resize_mode_balanced_default():
    """Default resize_mode is 'balanced'."""
    c = VGGTOmegaCreator()
    assert c.resize_mode == "balanced"


def test_load_model_builds_on_upstream_defaults(tmp_path):
    """_load_model constructs VGGTOmega with no overrides (upstream enable_alignment=False)."""
    ckpt_path = tmp_path / "fake.pt"
    torch.save({}, ckpt_path)
    creator = VGGTOmegaCreator(model_path=str(ckpt_path))

    mock_instance = MagicMock()
    mock_instance.eval.return_value = mock_instance
    mock_instance.to.return_value = mock_instance

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega", return_value=mock_instance) as mock_cls:
        creator._load_model("cpu")

    mock_cls.assert_called_once_with()
