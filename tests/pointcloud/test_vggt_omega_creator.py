"""Tests for VGGTOmegaCreator.

All tests mock vggt_omega.models and checkpoint loading — no GPU or HF download required.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator, FeedforwardResult
from collab_splats.pointcloud.feedforward.vggt_omega import (
    VGGTOmegaCreator,
    _compute_omega_original_coords,
)


########################################################################
########## Helpers #####################################################
########################################################################

def _make_raw_outputs(n: int = 2, h: int = 4, w: int = 4) -> dict:
    """Minimal raw_outputs dict matching VGGTOmegaCreator._forward output format."""
    return {
        "images": torch.zeros(n, 3, h, w),
        "extrinsic": np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        "intrinsics_downsampled": np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        "depth": np.ones((n, h, w), dtype=np.float32),
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
    assert c.camera_model == "PINHOLE"
    assert c.model_path is None
    assert c.model_repo == "facebook/VGGT-Omega"
    assert c.model_filename == "vggt_omega_1b_512.pt"
    assert c.resolution == 512  # None → resolved to 512
    assert c.resize_mode == "balanced"
    assert c.enable_text_alignment is False
    assert c.conf_threshold == 50.0


def test_vggt_omega_creator_missing_image_dir_raises(tmp_path):
    """reconstruct raises FileNotFoundError for nonexistent image dir."""
    c = VGGTOmegaCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def test_vggt_omega_has_logger():
    """Module exposes a module-level logger."""
    import collab_splats.pointcloud.feedforward.vggt_omega as mod
    assert hasattr(mod, "logger")
    assert isinstance(mod.logger, logging.Logger)


########################################################################
########## _compute_omega_original_coords ##############################
########################################################################

def test_compute_omega_original_coords_normal_ar(tmp_path):
    """Square image — no crop; coords are full-frame."""
    import PIL.Image
    img_path = tmp_path / "square.png"
    PIL.Image.new("RGB", (64, 64), color=0).save(img_path)

    coords = _compute_omega_original_coords([img_path])
    assert coords.shape == (1, 6)
    assert coords.dtype == np.float32
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    assert tl_x == pytest.approx(0.0)
    assert tl_y == pytest.approx(0.0)
    assert cr_x == pytest.approx(64.0)
    assert cr_y == pytest.approx(64.0)
    assert orig_w == pytest.approx(64.0)
    assert orig_h == pytest.approx(64.0)


def test_compute_omega_original_coords_tall_image_crops_height(tmp_path):
    """Tall image (AR > 2.0) — height is center-cropped."""
    import PIL.Image
    img_path = tmp_path / "tall.png"
    PIL.Image.new("RGB", (100, 400), color=0).save(img_path)  # AR = 4.0 > 2.0

    coords = _compute_omega_original_coords([img_path])
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    # crop_h = 100 * 2.0 = 200; tl_y = (400 - 200) / 2 = 100; cr_y = 100 + 200 = 300
    assert tl_x == pytest.approx(0.0)
    assert tl_y == pytest.approx(100.0)
    assert cr_x == pytest.approx(100.0)
    assert cr_y == pytest.approx(300.0)
    assert orig_w == pytest.approx(100.0)
    assert orig_h == pytest.approx(400.0)


def test_compute_omega_original_coords_wide_image_crops_width(tmp_path):
    """Wide image (AR < 0.5) — width is center-cropped."""
    import PIL.Image
    img_path = tmp_path / "wide.png"
    PIL.Image.new("RGB", (400, 100), color=0).save(img_path)  # AR = 0.25 < 0.5

    coords = _compute_omega_original_coords([img_path])
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    # crop_w = 100 / 0.5 = 200; tl_x = (400 - 200) / 2 = 100; cr_x = 100 + 200 = 300
    assert tl_x == pytest.approx(100.0)
    assert tl_y == pytest.approx(0.0)
    assert cr_x == pytest.approx(300.0)
    assert cr_y == pytest.approx(100.0)


def test_compute_omega_original_coords_multiple_images(tmp_path):
    """Multiple images — coords shape (N, 6)."""
    import PIL.Image
    paths = []
    for i, (w, h) in enumerate([(64, 64), (100, 400), (400, 100)]):
        p = tmp_path / f"img{i}.png"
        PIL.Image.new("RGB", (w, h), color=0).save(p)
        paths.append(p)

    coords = _compute_omega_original_coords(paths)
    assert coords.shape == (3, 6)
    assert coords.dtype == np.float32


########################################################################
########## _preprocess #################################################
########################################################################

def test_preprocess_empty_dir_raises(tmp_path):
    """_preprocess raises FileNotFoundError when no images present."""
    creator = VGGTOmegaCreator()
    with pytest.raises(FileNotFoundError, match="No images found"):
        creator._preprocess(tmp_path)


def test_preprocess_returns_correct_shapes(tmp_path):
    """_preprocess returns (views [N,3,H,W], paths, original_coords [N,6])."""
    import PIL.Image
    for i in range(3):
        PIL.Image.new("RGB", (64, 64), color=i * 80).save(tmp_path / f"frame_{i:04d}.jpg")

    creator = VGGTOmegaCreator(resolution=64)
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.load_and_preprocess_images",
               return_value=torch.zeros(3, 3, 64, 64)):
        views, image_paths, original_coords = creator._preprocess(tmp_path)

    assert views.shape == (3, 3, 64, 64)
    assert len(image_paths) == 3
    assert original_coords.shape == (3, 6)
    assert original_coords.dtype == np.float32
    # Verify image_paths are sorted by name
    assert [p.name for p in image_paths] == sorted(p.name for p in image_paths)


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
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
               return_value=(torch.zeros(1, n, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, n, 3, 3))):
        raw = creator._forward(mock_model, views)

    required_keys = {"images", "extrinsic", "intrinsics", "intrinsics_downsampled", "depth", "depth_conf"}
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
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
               return_value=(ext_tensor, intr_tensor)):
        raw = creator._forward(mock_model, torch.zeros(n, 3, h, w))

    assert raw["extrinsic"].shape == (n, 3, 4)
    assert raw["extrinsic"].dtype == np.float32
    assert raw["intrinsics"].shape == (n, 3, 3)


########################################################################
########## _postprocess ################################################
########################################################################

def test_postprocess_returns_feedforward_result(tmp_path):
    """_postprocess returns a FeedforwardResult with correct shapes."""
    n = 2
    image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(n)]
    raw = _make_raw_outputs(n)

    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator()
    creator.image_paths = image_paths
    creator.original_coords = np.zeros((n, 6), dtype=np.float32)
    creator.model = MagicMock()
    creator.model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)):
        result = creator._postprocess(raw)

    assert isinstance(result, FeedforwardResult)
    assert result.points.shape == (5, 3)
    assert result.colors.shape == (5, 3)
    assert result.extrinsics.shape == (n, 4, 4)
    assert result.intrinsics.shape == (n, 3, 3)
    assert result.model_width == 4
    assert result.model_height == 4
    assert result.image_paths == image_paths


def test_postprocess_world_points_populated(tmp_path):
    """world_points always populated (needed by BundleAdjustment wrapper)."""
    n, h, w = 2, 4, 4
    raw = _make_raw_outputs(n, h, w)
    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator()
    creator.image_paths = [tmp_path / f"{i}.jpg" for i in range(n)]
    creator.original_coords = np.zeros((n, 6), dtype=np.float32)
    creator.model = MagicMock()
    creator.model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)):
        result = creator._postprocess(raw)

    assert result.world_points is not None
    assert result.world_points.shape == (n, h, w, 3)
    assert result.confidence is not None
    assert result.images is not None


########################################################################
########## _load_model #################################################
########################################################################

def test_load_model_with_explicit_path(tmp_path):
    """_load_model with model_path skips hf_hub_download."""
    ckpt_path = tmp_path / "fake.pt"
    fake_state = {"aggregator.depth": torch.tensor(1.0)}
    torch.save(fake_state, ckpt_path)

    creator = VGGTOmegaCreator(model_path=str(ckpt_path))

    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega",
               return_value=mock_model_instance) as mock_cls, \
         patch("collab_splats.pointcloud.feedforward.vggt_omega.hf_hub_download") as mock_hf:
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

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega",
               return_value=mock_model_instance), \
         patch("collab_splats.pointcloud.feedforward.vggt_omega.hf_hub_download",
               return_value=str(ckpt_path)) as mock_hf:
        creator._load_model("cpu")

    mock_hf.assert_called_once_with(
        repo_id="facebook/VGGT-Omega",
        filename="vggt_omega_1b_512.pt",
    )


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

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega",
               return_value=mock_model_instance):
        creator._load_model("cpu")

    # No .to() call may pass a low-precision dtype
    for call in mock_model_instance.to.call_args_list:
        dtype = call.kwargs.get("dtype")
        if dtype is None and len(call.args) > 1:
            dtype = call.args[1]
        assert dtype not in (torch.bfloat16, torch.float16), (
            f"_load_model cast model to {dtype}; VGGTOmega heads require fp32 params"
        )


########################################################################
########## extract_intermediate_features ###############################
########################################################################

def test_extract_intermediate_features_returns_q_k_poses():
    """Hook fires on inter_frame_blocks[-1]; returns q, k, poses."""
    model = _make_mock_model(n_frames=2, h=16, w=16, num_heads=2, n_blocks=2)
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.model = model

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
               return_value=(torch.zeros(1, 2, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 2, 3, 3))):
        result = creator.extract_intermediate_features(
            torch.zeros(2, 3, 16, 16), layer_index=-1
        )

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

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
               return_value=(torch.zeros(1, 2, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 2, 3, 3))):
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
        with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
                   return_value=(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 3))):
            creator.extract_intermediate_features(torch.zeros(2, 3, 4, 4), layer_index=-1)

    assert len(block.attn.qkv._forward_hooks) == 0


########################################################################
########## _reproject ##################################################
########################################################################

def test_reproject_returns_pts3d_colors(tmp_path):
    """_reproject returns (pts3d, colors) tuple."""
    n = 2
    raw = _make_raw_outputs(n)
    extrinsics_3x4 = np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32)
    intrinsics = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)

    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator()
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)):
        result_pts, result_colors = creator._reproject(raw, extrinsics_3x4, intrinsics)

    assert result_pts.shape == (5, 3)
    assert result_colors.shape == (5, 3)


########################################################################
########## resolution / resize_mode / enable_text_alignment ############
########################################################################

def test_enable_text_alignment_auto_sets_resolution_256():
    """resolution=None + enable_text_alignment=True → resolved to 256."""
    c = VGGTOmegaCreator(enable_text_alignment=True)
    assert c.resolution == 256


def test_enable_text_alignment_auto_sets_resolution_512():
    """resolution=None + enable_text_alignment=False → resolved to 512."""
    c = VGGTOmegaCreator(enable_text_alignment=False)
    assert c.resolution == 512


def test_explicit_resolution_not_overridden():
    """Explicit resolution=768 is preserved regardless of enable_text_alignment."""
    c = VGGTOmegaCreator(resolution=768, enable_text_alignment=True)
    assert c.resolution == 768


def test_invalid_resize_mode_raises():
    """__post_init__ raises ValueError for unknown resize_mode."""
    with pytest.raises(ValueError, match="resize_mode"):
        VGGTOmegaCreator(resize_mode="bogus")


def test_resize_mode_balanced_default():
    """Default resize_mode is 'balanced'."""
    c = VGGTOmegaCreator()
    assert c.resize_mode == "balanced"


def test_load_model_passes_enable_alignment_true(tmp_path):
    """_load_model passes enable_alignment=True to VGGTOmega when flag is set."""
    ckpt_path = tmp_path / "fake.pt"
    torch.save({}, ckpt_path)
    creator = VGGTOmegaCreator(model_path=str(ckpt_path), enable_text_alignment=True)

    mock_instance = MagicMock()
    mock_instance.eval.return_value = mock_instance
    mock_instance.to.return_value = mock_instance

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega",
               return_value=mock_instance) as mock_cls:
        creator._load_model("cpu")

    mock_cls.assert_called_once_with(enable_alignment=True)


def test_load_model_passes_enable_alignment_false(tmp_path):
    """_load_model passes enable_alignment=False by default."""
    ckpt_path = tmp_path / "fake.pt"
    torch.save({}, ckpt_path)
    creator = VGGTOmegaCreator(model_path=str(ckpt_path))

    mock_instance = MagicMock()
    mock_instance.eval.return_value = mock_instance
    mock_instance.to.return_value = mock_instance

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega",
               return_value=mock_instance) as mock_cls:
        creator._load_model("cpu")

    mock_cls.assert_called_once_with(enable_alignment=False)


def test_postprocess_passes_max_points():
    """_postprocess passes max_points=self.max_points to unproject_and_filter_points."""
    n, h, w = 2, 4, 4
    raw = _make_raw_outputs(n, h, w)
    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator(max_points=123_456)
    creator.image_paths = [Path(f"/fake/{i}.jpg") for i in range(n)]
    creator.original_coords = np.zeros((n, 6), dtype=np.float32)
    creator.model = MagicMock()
    creator.model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)) as mock_unproj:
        creator._postprocess(raw)

    call_kwargs = mock_unproj.call_args[1]
    assert call_kwargs.get("max_points") == 123_456


def test_reproject_passes_max_points():
    """_reproject passes max_points=self.max_points to unproject_and_filter_points."""
    n = 2
    raw = _make_raw_outputs(n)
    extrinsics_3x4 = np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32)
    intrinsics = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)
    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator(max_points=99_000)

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)) as mock_unproj:
        creator._reproject(raw, extrinsics_3x4, intrinsics)

    call_kwargs = mock_unproj.call_args[1]
    assert call_kwargs.get("max_points") == 99_000


def test_omega_use_multiview_confidence_calls_compute_fn(tmp_path):
    """VGGTOmegaCreator with use_multiview_confidence=True calls compute_multiview_depth_confidence."""
    import numpy as np
    from unittest.mock import patch
    from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator

    N, H, W = 2, 4, 4
    raw_outputs = {
        "depth": np.ones((N, H, W, 1), dtype=np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": np.zeros((N, 3, H, W), dtype=np.float32),
        "extrinsic": np.stack([np.eye(4)[:3, :]] * N).astype(np.float32),
        "intrinsics": np.eye(3, dtype=np.float32)[np.newaxis].repeat(N, axis=0),
        "intrinsics_downsampled": np.eye(3, dtype=np.float32)[np.newaxis].repeat(N, axis=0),
    }

    creator = VGGTOmegaCreator(use_multiview_confidence=True, mv_conf_threshold=0.0)
    creator.image_paths = [tmp_path / f"{i:06d}.jpg" for i in range(N)]
    creator.original_coords = np.zeros((N, 6), dtype=np.float32)
    creator.views = None

    mv_conf_ones = np.ones((N, H, W), dtype=np.float32)

    with patch(
        "collab_splats.pointcloud.feedforward.vggt_omega.compute_multiview_depth_confidence",
        return_value=mv_conf_ones,
    ) as mock_mv:
        result = creator._postprocess(raw_outputs)

    mock_mv.assert_called_once()
    called_depth = mock_mv.call_args[0][0]
    assert called_depth.shape == (N, H, W)
    assert result is not None
