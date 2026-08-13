import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock
from collab_splats.pointcloud.feedforward import VGGTXCreator, BaseFeedforwardCreator, FeedforwardResult
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


def test_vggtx_defaults():
    c = VGGTXCreator()
    assert c.model_name == "facebook/VGGT-1B"


def test_vggtx_is_feedforward_creator():
    assert issubclass(VGGTXCreator, BaseFeedforwardCreator)


def test_vggtx_missing_image_dir_raises(tmp_path):
    c = VGGTXCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


@pytest.mark.gpu
def test_vggtx_reconstruct_smoke(tmp_path):
    from PIL import Image as PILImage

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        PILImage.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")

    c = VGGTXCreator()
    result = c.reconstruct(image_dir, tmp_path / "out")
    assert isinstance(result, PointcloudResult)
    assert result.frame == CoordinateFrame.COLMAP
    assert (tmp_path / "out" / "transforms.json").exists()
    assert (tmp_path / "out" / "colmap" / "sparse" / "0" / "cameras.bin").exists()


# ---------------------------------------------------------------------------
# extract_intermediate_features — hook pattern
# ---------------------------------------------------------------------------


def _make_vggtx_with_mock_model(num_heads=2, head_dim=4, n_blocks=2, n_tokens=10):
    """VGGTXCreator backed by a mock model with real nn.Linear QKV layers.

    The mock model.forward calls each block's qkv linear so any registered
    forward hook fires.  Returns (creator, blocks, n_tokens, num_heads, head_dim).
    """
    total_dim = num_heads * head_dim

    # Real nn.Linear so register_forward_hook actually triggers
    qkv_linears = [nn.Linear(total_dim, total_dim * 3, bias=False) for _ in range(n_blocks)]
    blocks = []
    for qkv in qkv_linears:
        block = MagicMock()
        block.attn.num_heads = num_heads
        block.attn.qkv = qkv
        blocks.append(block)

    def mock_forward(batch):
        # Call every block's qkv so any registered hook fires
        B, S, H, W = batch.shape[0], batch.shape[1], batch.shape[-2], batch.shape[-1]
        x = torch.randn(B, n_tokens, total_dim)
        for qkv in qkv_linears:
            qkv(x)
        # Verify path decodes pose_enc AND depth/depth_conf from the same forward
        return {
            "pose_enc": torch.zeros(B, S, 9),
            "depth": torch.ones(B, S, H, W, 1),
            "depth_conf": torch.ones(B, S, H, W),
        }

    mock_model = MagicMock()
    mock_model.aggregator.global_blocks = blocks
    mock_model.side_effect = mock_forward
    param = nn.Parameter(torch.zeros(1))
    mock_model.parameters = lambda: iter([param])

    creator = VGGTXCreator.__new__(VGGTXCreator)
    creator.model = mock_model
    return creator, blocks, n_tokens, num_heads, head_dim


def test_vggtx_extract_intermediate_features_shapes():
    """Hook fires on last block; q/k shapes correct; poses decoded and present."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_vggtx_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)
    result = creator.extract_intermediate_features(frames, layer_index=-1)
    # Required keys: q and k (activations), poses (decoded)
    assert "q" in result and "k" in result
    assert "pose_enc" not in result, "raw pose_enc must not leak out of extract_intermediate_features"
    assert "poses" in result
    # q/k: (B=1, heads, n_tokens, head_dim)
    assert result["q"].shape == (1, num_heads, n_tokens, head_dim)
    assert result["k"].shape == (1, num_heads, n_tokens, head_dim)
    # poses: (2, 4, 4) float32 numpy array
    assert result["poses"].shape == (2, 4, 4)
    assert result["poses"].dtype == np.float32
    # Geometry decoded from the same forward: unprojected depth + confidence
    assert result["world_points"].shape == (2, 16, 16, 3)
    assert result["world_points"].dtype == np.float32
    assert result["conf"].shape == (2, 16, 16)
    assert result["conf"].dtype == np.float32


def test_vggtx_extract_intermediate_features_hook_removed():
    """Hook removed after call — no accumulated hooks on repeated calls."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_vggtx_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)
    qkv = blocks[-1].attn.qkv
    assert len(qkv._forward_hooks) == 0
    creator.extract_intermediate_features(frames, layer_index=-1)
    # Hook must be removed whether the call succeeded or failed
    assert len(qkv._forward_hooks) == 0


def test_vggtx_extract_intermediate_features_hook_removed_on_error():
    """Hook removed even when model forward raises."""
    total_dim = 2 * 4  # num_heads=2, head_dim=4
    qkv = nn.Linear(total_dim, total_dim * 3, bias=False)
    block = MagicMock()
    block.attn.num_heads = 2
    block.attn.qkv = qkv

    def boom(batch):
        raise RuntimeError("simulated forward failure")

    mock_model = MagicMock()
    mock_model.aggregator.global_blocks = [block]
    mock_model.side_effect = boom
    param = nn.Parameter(torch.zeros(1))
    mock_model.parameters = lambda: iter([param])

    creator = VGGTXCreator.__new__(VGGTXCreator)
    creator.model = mock_model

    with pytest.raises(RuntimeError, match="simulated forward failure"):
        creator.extract_intermediate_features(torch.zeros(2, 3, 16, 16))
    assert len(qkv._forward_hooks) == 0


def test_vggtx_extract_intermediate_features_layer_index():
    """Non-default layer_index taps the correct block."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_vggtx_with_mock_model(n_blocks=3)
    frames = torch.zeros(2, 3, 16, 16)

    # Spy on blocks[1].attn.qkv.register_forward_hook
    hooked_blocks = []
    orig_register = blocks[1].attn.qkv.register_forward_hook

    def spy_register(fn):
        hooked_blocks.append(1)
        return orig_register(fn)

    blocks[1].attn.qkv.register_forward_hook = spy_register
    creator.extract_intermediate_features(frames, layer_index=1)
    assert hooked_blocks == [1]


def test_patch_vggtx_compute_similarity_deleted():
    """_patch_vggtx_compute_similarity must not exist after refactor."""
    import collab_splats.pointcloud.feedforward.vggtx as vggtx_mod

    assert not hasattr(
        vggtx_mod, "_patch_vggtx_compute_similarity"
    ), "_patch_vggtx_compute_similarity still exists — delete it and its _load_model call"


def test_unproject_and_filter_points_extra_mask():
    """extra_mask=False pixels are excluded from the output."""
    import numpy as np
    from collab_splats.pointcloud.feedforward.vggtx import unproject_and_filter_points

    N, H, W = 2, 4, 4
    depth = np.ones((N, H, W, 1), dtype=np.float32)
    depth_conf = np.ones((N, H, W), dtype=np.float32)
    images = np.zeros((N, 3, H, W), dtype=np.float32)
    extrinsic = np.stack([np.eye(4)[:3, :]] * N).astype(np.float32)
    intrinsic = np.stack([np.eye(3)] * N).astype(np.float32)

    # Without extra_mask: all pixels survive (conf_threshold=0.0)
    pts_all, _, _ = unproject_and_filter_points(depth, depth_conf, images, extrinsic, intrinsic, conf_threshold=0.0)

    # extra_mask zeros out frame 0 completely
    extra_mask = np.ones((N, H, W), dtype=bool)
    extra_mask[0] = False
    pts_masked, _, _ = unproject_and_filter_points(
        depth,
        depth_conf,
        images,
        extrinsic,
        intrinsic,
        conf_threshold=0.0,
        extra_mask=extra_mask,
    )

    assert len(pts_masked) < len(
        pts_all
    ), f"Expected fewer points with extra_mask; got {len(pts_masked)} vs {len(pts_all)}"
    # Frame 0 masked → only frame 1's H*W points survive
    assert len(pts_masked) == H * W, f"Expected {H*W}, got {len(pts_masked)}"


def test_vggtx_use_multiview_confidence_calls_compute_fn(tmp_path):
    """VGGTXCreator with use_multiview_confidence=True calls compute_multiview_depth_confidence."""
    import numpy as np
    from unittest.mock import patch
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    N, H, W = 2, 4, 4
    raw_outputs = {
        "depth": np.ones((N, H, W, 1), dtype=np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": np.zeros((N, 3, H, W), dtype=np.float32),
        "extrinsic": np.stack([np.eye(4)[:3, :]] * N).astype(np.float32),
        "intrinsics": np.eye(3, dtype=np.float32)[np.newaxis].repeat(N, axis=0),
        "intrinsics_downsampled": np.eye(3, dtype=np.float32)[np.newaxis].repeat(N, axis=0),
    }

    creator = VGGTXCreator(use_multiview_confidence=True, mv_conf_threshold=0.0)
    creator.image_paths = [tmp_path / f"{i:06d}.jpg" for i in range(N)]
    creator.original_coords = np.zeros((N, 6), dtype=np.float32)
    creator.views = None

    from collab_splats.pointcloud.feedforward.base import MultiviewConfidence

    mv_conf_ones = MultiviewConfidence(
        ratio=np.ones((N, H, W), dtype=np.float32),
        inlier_count=np.ones((N, H, W), dtype=np.int32),
        valid_count=np.ones((N, H, W), dtype=np.int32),
        judged=np.ones(N, dtype=bool),
    )

    with patch(
        "collab_splats.pointcloud.feedforward.vggtx.compute_multiview_depth_confidence",
        return_value=mv_conf_ones,
    ) as mock_mv:
        result = creator._postprocess(raw_outputs)

    mock_mv.assert_called_once()
    called_depth = mock_mv.call_args[0][0]
    assert called_depth.shape == (N, H, W), f"Expected ({N},{H},{W}), got {called_depth.shape}"
    assert result is not None
    assert len(result.points) > 0
