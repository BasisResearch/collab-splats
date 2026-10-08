from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator, VGGTXCreator


def test_vggtx_defaults():
    c = VGGTXCreator()
    assert c.model_name == "facebook/VGGT-1B"


def test_vggtx_is_feedforward_creator():
    assert issubclass(VGGTXCreator, BaseFeedforwardCreator)


def test_vggtx_missing_image_dir_raises(tmp_path):
    c = VGGTXCreator()
    with pytest.raises(FileNotFoundError):
        c.create_pointcloud(
            tmp_path / "nonexistent", tmp_path / "out", tmp_path / "model"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VGGT-X needs CUDA")
def test_vggtx_reconstruct_smoke(tmp_path):
    from PIL import Image as PILImage

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        PILImage.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")

    c = VGGTXCreator()
    result = c.create_pointcloud(image_dir, tmp_path / "out", tmp_path / "model")
    assert isinstance(result, PointcloudResult)
    assert (tmp_path / "model" / "cameras.bin").exists()


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
    qkv_linears = [
        nn.Linear(total_dim, total_dim * 3, bias=False) for _ in range(n_blocks)
    ]
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
    assert "pose_enc" not in result, (
        "raw pose_enc must not leak out of extract_intermediate_features"
    )
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
        creator.extract_intermediate_features(torch.zeros(2, 3, 16, 16), layer_index=-1)
    assert len(qkv._forward_hooks) == 0


def test_vggtx_extract_intermediate_features_layer_index():
    """Non-default layer_index taps the correct block."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_vggtx_with_mock_model(
        n_blocks=3
    )
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

    assert not hasattr(vggtx_mod, "_patch_vggtx_compute_similarity"), (
        "_patch_vggtx_compute_similarity still exists — delete it and its _load_model call"
    )


def _postprocessed_outputs():
    """
    VGGTXCreator._postprocess on a seeded scene with a rotated, translated camera.

    - identity poses would make both unprojection paths bit-identical and hide an ulp mismatch
    """
    N, H, W = 2, 6, 8
    rng = np.random.default_rng(0)
    angle = 0.3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsic = np.stack([np.hstack([rotation, [[0.1], [-0.2], [0.3]]])] * N).astype(
        np.float32
    )
    intrinsic = np.stack(
        [[[7.3, 0.0, 3.9], [0.0, 6.1, 2.7], [0.0, 0.0, 1.0]]] * N
    ).astype(np.float32)
    raw_outputs = {
        "depth": rng.uniform(0.5, 3.0, (N, H, W, 1)).astype(np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": np.zeros((N, 3, H, W), dtype=np.float32),
        "extrinsic": extrinsic,
        "intrinsics": intrinsic,
    }

    creator = VGGTXCreator(conf_threshold=0.0)
    creator.image_paths = [Path(f"{i:06d}.jpg") for i in range(N)]
    creator.original_coords = np.tile(
        np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (N, 1)
    )  # full-frame box
    return creator._postprocess(raw_outputs)


def test_points_are_rows_of_the_world_grid():
    out = _postprocessed_outputs()
    grid = out.world_points.reshape(-1, 3)
    assert len(out.points) > 0
    assert {tuple(p) for p in out.points} <= {tuple(p) for p in grid}
