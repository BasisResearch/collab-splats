import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.pointcloud.base import CoordinateFrame, PointcloudResult
from collab_splats.pointcloud.feedforward import (
    BaseFeedforwardCreator,
    MapAnythingCreator,
)
from collab_splats.pointcloud.feedforward.base import _raw_to_world_points
from tests.pointcloud.feedforward.conftest import _FakeMapAnythingModel


def test_mapanything_defaults():
    c = MapAnythingCreator()
    assert c.model_name == "facebook/map-anything"
    assert c.confidence_percentile == 35.0
    assert c.use_multiview_confidence is True
    assert c.minibatch_size == 1


def test_mapanything_is_feedforward_creator():
    assert issubclass(MapAnythingCreator, BaseFeedforwardCreator)


def test_mapanything_missing_image_dir_raises(tmp_path):
    c = MapAnythingCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def test_mapanything_forward_calls_model_forward():
    import torch

    n = 2
    param = torch.zeros(1)  # CPU param; provides .device = cpu
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([param])
    mock_raw = [MagicMock() for _ in range(n)]
    mock_model.forward.return_value = mock_raw

    creator = MapAnythingCreator(minibatch_size=2)
    creator._processed_views = [{"img": torch.zeros(1, 3, 64, 64)} for _ in range(n)]

    result = creator._forward(mock_model, views=None)

    mock_model.forward.assert_called_once_with(
        creator._processed_views,
        memory_efficient_inference=True,
        minibatch_size=2,
    )
    assert result is mock_raw


def test_mapanything_preprocess_sets_processed_views():
    import torch

    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    frame_idxs = [0, 1]

    fake_views = [{"img": torch.zeros(1, 3, 224, 224), "data_norm_type": "imagenet"} for _ in range(2)]
    fake_validated = fake_views
    fake_processed = [{"img": torch.zeros(1, 3, 224, 224)} for _ in range(2)]

    with (
        patch("collab_splats.pointcloud.feedforward.mapanything.load_images", return_value=fake_views),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
            return_value=fake_validated,
        ) as mock_validate,
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=fake_processed,
        ) as mock_preprocess,
    ):
        creator = MapAnythingCreator()
        creator._preprocess(frames, frame_idxs)

    mock_validate.assert_called_once_with(fake_views)
    mock_preprocess.assert_called_once_with(fake_validated)
    assert creator._processed_views is fake_processed


def test_mapanything_postprocess_casts_bf16_to_float32():
    """Regression guard: pts3d_cam and pts3d must be float32 before postprocess.

    F.grid_sample inside compute_multiview_depth_confidence requires matching
    dtypes. torch 2.4 enforces this strictly; 2.1.2 allowed bf16/float32 mismatch.
    """
    import torch

    n, h, w = 2, 4, 4
    raw_outputs = [
        {
            "pts3d_cam": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
            "pts3d": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
        }
        for _ in range(n)
    ]

    fake_processed = [
        {
            "pts3d": torch.zeros(1, h, w, 3),
            "pts3d_cam": torch.zeros(1, h, w, 3),
            "mask": torch.ones(1, h, w, 1, dtype=torch.bool),
            "depth_z": torch.ones(1, h, w, 1),
            "img_no_norm": torch.zeros(1, h, w, 3),
            "intrinsics": torch.eye(3).unsqueeze(0),
            "camera_poses": torch.eye(4).unsqueeze(0),
        }
        for _ in range(n)
    ]

    creator = MapAnythingCreator()
    creator._processed_views = [{"img": torch.zeros(1, 3, h, w)} for _ in range(n)]
    creator.image_paths = [Path(f"/fake/img_{i}.jpg") for i in range(n)]
    creator.original_coords = np.zeros((n, 6), dtype=np.float32)

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything" ".postprocess_model_outputs_for_inference",
        return_value=fake_processed,
    ) as mock_post:
        creator._postprocess(raw_outputs)

    # raw_outputs is mutated in-place before postprocess is called;
    # mock captures the reference so we inspect dtype at call time
    called_raw = mock_post.call_args[0][0]
    for pred in called_raw:
        assert (
            pred["pts3d_cam"].dtype == torch.float32
        ), f"pts3d_cam not cast to float32 before postprocess: {pred['pts3d_cam'].dtype}"
        assert (
            pred["pts3d"].dtype == torch.float32
        ), f"pts3d not cast to float32 before postprocess: {pred['pts3d'].dtype}"


def test_mapanything_full_pipeline_cpu_mock(tmp_path):
    """Integration test: setup_inference → run_inference → postprocess through mocked model.

    Verifies pipeline wiring end-to-end without GPU or real model weights.
    Mocks are placed at model.forward() and mapanything.utils.inference functions.
    """
    import torch
    from PIL import Image as PILImage

    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    for i in range(2):
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image_dir / f"frame_{i:04d}.jpg")

    n, h, w = 2, 8, 8
    fake_views = [{"img": torch.zeros(1, 3, h, w), "data_norm_type": "imagenet"} for _ in range(n)]
    fake_processed = [{"img": torch.zeros(1, 3, h, w)} for _ in range(n)]

    # Raw outputs from model.forward() — bf16 as produced under autocast
    fake_raw = [
        {
            "pts3d_cam": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
            "pts3d": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
            "ray_directions": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
            "depth_along_ray": torch.ones(1, h, w, 1, dtype=torch.bfloat16),
            "cam_trans": torch.zeros(1, 3, dtype=torch.bfloat16),
            "cam_quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16),
            "metric_scaling_factor": torch.ones(1, dtype=torch.bfloat16),
        }
        for _ in range(n)
    ]

    # postprocess_model_outputs_for_inference output: float32, with derived keys
    fake_post = [
        {
            "pts3d": torch.zeros(1, h, w, 3),
            "pts3d_cam": torch.zeros(1, h, w, 3),
            "mask": torch.ones(1, h, w, 1, dtype=torch.bool),
            "depth_z": torch.ones(1, h, w, 1),
            "img_no_norm": torch.zeros(1, h, w, 3),
            "intrinsics": torch.eye(3).unsqueeze(0),
            "camera_poses": torch.eye(4).unsqueeze(0),
        }
        for _ in range(n)
    ]

    param = torch.zeros(1)
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([param])
    mock_model.forward.return_value = fake_raw

    with (
        patch("collab_splats.pointcloud.feedforward.mapanything.load_images", return_value=fake_views),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
            return_value=fake_views,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=fake_processed,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            return_value=fake_post,
        ),
    ):

        creator = MapAnythingCreator()
        creator.model = mock_model

        creator.setup_inference(image_dir)
        assert hasattr(creator, "_processed_views"), "_preprocess must set _processed_views"
        assert creator._processed_views is fake_processed

        creator.run_inference()
        mock_model.forward.assert_called_once_with(
            fake_processed,
            memory_efficient_inference=True,
            minibatch_size=1,
        )

        creator.postprocess()
        assert creator.outputs is not None
        assert creator.outputs.points.shape[1] == 3
        assert creator.outputs.extrinsics.shape == (n, 4, 4)


def test_reproject_output_shapes():
    """_reproject returns (P,3) float32 pts3d and (P,3) uint8 colors."""
    import torch

    creator = MapAnythingCreator()
    n, h, w = 2, 4, 4
    raw_outputs = [
        {
            "pts3d_cam": torch.zeros(1, h, w, 3, dtype=torch.float32),
            "mask": torch.ones(1, h, w, 1, dtype=torch.bool),
            "depth_z": torch.ones(1, h, w, 1, dtype=torch.float32),
            "img_no_norm": torch.zeros(1, h, w, 3, dtype=torch.float32),
        }
        for _ in range(n)
    ]
    extrinsics_3x4 = np.tile(np.eye(4)[:3, :], (n, 1, 1)).astype(np.float32)
    intrinsics = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)

    pts3d, colors = creator._reproject(raw_outputs, extrinsics_3x4, intrinsics)

    assert pts3d.shape == (n * h * w, 3), f"Expected ({n*h*w}, 3), got {pts3d.shape}"
    assert pts3d.dtype == np.float32
    assert colors.shape == (n * h * w, 3)
    assert colors.dtype == np.uint8


def test_reproject_depth_mask_filters_zero_depth():
    """Points with depth_z <= 0 are excluded from output."""
    import torch

    creator = MapAnythingCreator()
    h, w = 4, 4
    raw_outputs = [
        {
            "pts3d_cam": torch.zeros(1, h, w, 3, dtype=torch.float32),
            "mask": torch.ones(1, h, w, 1, dtype=torch.bool),
            "depth_z": torch.ones(1, h, w, 1, dtype=torch.float32),
            "img_no_norm": torch.zeros(1, h, w, 3, dtype=torch.float32),
        }
    ]
    raw_outputs[0]["depth_z"][0, : h // 2, :, 0] = 0.0
    extrinsics_3x4 = np.eye(4)[:3, :][np.newaxis].astype(np.float32)
    intrinsics = np.eye(3)[np.newaxis].astype(np.float32)

    pts3d, colors = creator._reproject(raw_outputs, extrinsics_3x4, intrinsics)

    expected_count = h * w - (h // 2) * w
    assert pts3d.shape[0] == expected_count


def test_reproject_identity_extrinsic_preserves_cam_points():
    """Identity extrinsic (world2cam=I) → world pts == pts3d_cam."""
    import torch

    creator = MapAnythingCreator()
    h, w = 2, 2
    raw_outputs = [
        {
            "pts3d_cam": torch.zeros(1, h, w, 3, dtype=torch.float32),
            "mask": torch.ones(1, h, w, 1, dtype=torch.bool),
            "depth_z": torch.ones(1, h, w, 1, dtype=torch.float32),
            "img_no_norm": torch.zeros(1, h, w, 3, dtype=torch.float32),
        }
    ]
    raw_outputs[0]["pts3d_cam"][0] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 1.0, 2.0]]
    ).reshape(h, w, 3)
    extrinsics_3x4 = np.eye(4)[:3, :][np.newaxis].astype(np.float32)
    intrinsics = np.eye(3)[np.newaxis].astype(np.float32)

    pts3d, _ = creator._reproject(raw_outputs, extrinsics_3x4, intrinsics)

    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(pts3d, expected, atol=1e-5)


def test_mapanything_forward_transfers_tensors_to_device():
    """_forward must move _processed_views tensors to model device before model.forward()."""
    from contextlib import nullcontext

    import torch

    # meta device is always available without GPU; CPU→meta transfer is observable
    mock_param = MagicMock()
    mock_param.device = torch.device("meta")
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([mock_param])

    captured_views = []

    def capture_and_return(views, **kwargs):
        captured_views.extend(views)
        return []

    mock_model.forward.side_effect = capture_and_return

    cpu_tensor = torch.zeros(1, 3, 4, 4)  # starts on CPU
    creator = MapAnythingCreator()
    creator._processed_views = [{"img": cpu_tensor, "scalar": 1.0}]

    # Mock torch.autocast to avoid "unsupported autocast device_type 'meta'" error.
    # autocast is only for dtype conversion; tensor transfer happens before it.
    with patch("torch.autocast", return_value=nullcontext()):
        creator._forward(mock_model, views=None)

    # Tensor must have been transferred to meta device before model.forward() was called
    assert len(captured_views) == 1
    assert captured_views[0]["img"].device.type == "meta", (
        f"Expected meta device, got {captured_views[0]['img'].device.type} — " "tensor was not transferred in _forward"
    )
    # Non-tensor values must be unchanged
    assert creator._processed_views[0]["scalar"] == 1.0


def test_mapanything_load_model_standard():
    """_load_model calls from_pretrained → to(device) → eval; no compat patch."""
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    with patch("collab_splats.pointcloud.feedforward.mapanything.MapAnything") as mock_cls:
        mock_cls.from_pretrained.return_value = fake_model
        creator = MapAnythingCreator(model_name="test/model")
        result = creator._load_model(device="cpu")

    mock_cls.from_pretrained.assert_called_once_with("test/model")
    fake_model.to.assert_called_once_with("cpu")
    fake_model.eval.assert_called_once()
    assert result is fake_model


@pytest.mark.gpu
def test_mapanything_reconstruct_smoke(tmp_path):
    from PIL import Image as PILImage

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        PILImage.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")

    c = MapAnythingCreator()
    result = c.reconstruct(image_dir, tmp_path / "out")
    assert isinstance(result, PointcloudResult)
    assert result.frame == CoordinateFrame.COLMAP
    assert result.points.shape[1] == 3
    assert (tmp_path / "out" / "transforms.json").exists()
    assert (tmp_path / "out" / "colmap" / "sparse" / "0" / "cameras.bin").exists()


@pytest.mark.gpu
def test_mapanything_run_inference_smoke(tmp_path):
    """Verify MapAnything inference completes without dtype errors on torch 2.4+cu121
    and produces valid (non-NaN) confidence scores.

    The pre-cu121 compat patch let F.grid_sample(bf16, float32) silently succeed;
    torch 2.4 raises RuntimeError. This test guards the dtype-cast fix in _postprocess.
    """
    pytest.importorskip("mapanything")
    pytest.importorskip("torch")
    bicycle = Path("/workspace/bicycle/images_4")
    if not bicycle.exists():
        pytest.skip(f"{bicycle} not available on this host")

    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    creator = MapAnythingCreator(camera_model="PINHOLE")
    creator.load_model()
    creator.setup_inference(bicycle)
    creator.run_inference()
    creator.postprocess()

    assert creator.outputs.confidence is not None, "confidence should be set when use_multiview_confidence=True"
    assert (
        not creator.outputs.confidence.isnan().any()
    ), "confidence contains NaN — dtype cast regression in _postprocess"


# ---------------------------------------------------------------------------
# extract_intermediate_features — hook pattern
# ---------------------------------------------------------------------------


def _make_mapanything_with_mock_model(num_heads=2, head_dim=4, n_blocks=2, n_tokens=10):
    """MapAnythingCreator backed by mock model with real nn.Linear QKV layers.

    The mock model.forward calls each block's qkv linear so any registered
    forward hook fires.  Returns (creator, blocks, n_tokens, num_heads, head_dim).
    """
    import torch.nn as nn

    total_dim = num_heads * head_dim

    # Real nn.Linear so register_forward_hook actually triggers
    qkv_linears = [nn.Linear(total_dim, total_dim * 3, bias=False) for _ in range(n_blocks)]
    blocks = []
    for qkv in qkv_linears:
        block = MagicMock()
        block.attn.num_heads = num_heads
        block.attn.qkv = qkv
        blocks.append(block)

    def mock_forward(views, **kwargs):
        # Simulate info_sharing calling QKV on all blocks
        import torch

        x = torch.randn(1, n_tokens, total_dim)
        for qkv in qkv_linears:
            qkv(x)
        # Two per-view preds with float-castable pointmaps — the post-forward
        # pose recipe float-casts pts3d_cam/pts3d before postprocessing.
        return [{"pts3d_cam": torch.zeros(1, 4, 4, 3), "pts3d": torch.zeros(1, 4, 4, 3)} for _ in range(2)]

    mock_model = MagicMock()
    mock_model.info_sharing.self_attention_blocks = blocks
    mock_model.forward.side_effect = mock_forward

    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator.model = mock_model
    return creator, blocks, n_tokens, num_heads, head_dim


def _patch_mapanything_postprocess():
    """Patch module-level postprocess to yield identity camera_poses for 2 views."""
    import torch

    return patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        return_value=[{"camera_poses": torch.eye(4).unsqueeze(0)} for _ in range(2)],
    )


def test_mapanything_extract_intermediate_features_shapes():
    """Hook fires on last block; q/k shapes correct; poses derived from forward."""
    import numpy as np
    import torch

    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=[{}] * 2,
        ),
        _patch_mapanything_postprocess(),
    ):
        result = creator.extract_intermediate_features(frames, layer_index=-1)

    assert "q" in result and "k" in result
    assert result["q"].shape == (1, num_heads, n_tokens, head_dim)
    assert result["k"].shape == (1, num_heads, n_tokens, head_dim)
    # Poses derived from the same forward via the postprocess → invert recipe
    assert result["poses"].shape == (2, 4, 4)
    assert result["poses"].dtype == np.float32


def test_mapanything_extract_intermediate_features_hook_removed():
    """Hook removed after call — no accumulated hooks on repeated calls."""
    import torch

    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)
    qkv = blocks[-1].attn.qkv
    assert len(qkv._forward_hooks) == 0

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=[{}] * 2,
        ),
        _patch_mapanything_postprocess(),
    ):
        creator.extract_intermediate_features(frames, layer_index=-1)

    assert len(qkv._forward_hooks) == 0


def test_mapanything_extract_intermediate_features_layer_index():
    """Non-default layer_index taps the correct block."""
    import torch

    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_with_mock_model(n_blocks=3)
    frames = torch.zeros(2, 3, 16, 16)

    hooked_blocks = []
    orig_register = blocks[0].attn.qkv.register_forward_hook

    def spy_register(fn):
        hooked_blocks.append(0)
        return orig_register(fn)

    blocks[0].attn.qkv.register_forward_hook = spy_register
    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=[{}] * 2,
        ),
        _patch_mapanything_postprocess(),
    ):
        creator.extract_intermediate_features(frames, layer_index=0)

    assert hooked_blocks == [0]


def test_mapanything_verify_loop_candidate_not_on_class():
    """MapAnythingCreator must not define its own _verify_loop_candidate."""
    assert (
        "_verify_loop_candidate" not in MapAnythingCreator.__dict__
    ), "_verify_loop_candidate still defined on MapAnythingCreator — base class only"


@pytest.mark.gpu
@pytest.mark.xfail(
    reason="LC + MapAnything: _run_loop_closure_inference expects dict, "
    "MapAnythingCreator._forward returns list[dict]. Separate bug, see follow-up.",
    strict=True,
)
def test_mapanything_run_inference_loop_closure_smoke(tmp_path):
    pytest.importorskip("mapanything")
    pytest.importorskip("torch")
    bicycle = Path("/workspace/bicycle/images_4")
    if not bicycle.exists():
        pytest.skip(f"{bicycle} not available on this host")

    from collab_splats.geometry.loop_closure import LoopClosureConfig
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure
    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    base = MapAnythingCreator(camera_model="PINHOLE")
    creator = LoopClosure(base, config=LoopClosureConfig())
    creator.load_model()
    creator.setup_inference(bicycle)
    creator.run_inference()


########################################################################
########## resize_mode + resolution ####################################
########################################################################


def test_mapanything_resize_mode_default():
    """Default resize_mode is 'fixed'."""
    c = MapAnythingCreator()
    assert c.resize_mode == "fixed"
    assert c.resolution == 518


def test_mapanything_invalid_resize_mode_raises():
    """__post_init__ raises ValueError for unknown resize_mode."""
    with pytest.raises(ValueError, match="resize_mode"):
        MapAnythingCreator(resize_mode="bogus")


def test_mapanything_preprocess_fixed_mode_calls_load_images_with_fixed_mapping():
    """resize_mode='fixed' passes resize_mode='fixed_mapping' + resolution_set to load_images."""
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    frame_idxs = [0, 1]

    fake_view = {"img": __import__("torch").zeros(1, 3, 64, 64), "data_norm_type": "imagenet"}
    fake_views = [fake_view, fake_view]

    c = MapAnythingCreator(resize_mode="fixed", resolution=518)
    with (
        patch("collab_splats.pointcloud.feedforward.mapanything.load_images", return_value=fake_views) as mock_li,
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
            return_value=fake_views,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=fake_views,
        ),
    ):
        c._preprocess(frames, frame_idxs)

    call_kwargs = mock_li.call_args[1]
    assert call_kwargs.get("resize_mode") == "fixed_mapping"
    assert call_kwargs.get("resolution_set") == 518
    assert "size" not in call_kwargs


def test_mapanything_preprocess_longest_side_calls_load_images_with_size():
    """resize_mode='longest_side' passes resize_mode='longest_side' + size= to load_images."""
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    frame_idxs = [0, 1]

    fake_view = {"img": __import__("torch").zeros(1, 3, 64, 64), "data_norm_type": "imagenet"}
    fake_views = [fake_view, fake_view]

    c = MapAnythingCreator(resize_mode="longest_side", resolution=512)
    with (
        patch("collab_splats.pointcloud.feedforward.mapanything.load_images", return_value=fake_views) as mock_li,
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
            return_value=fake_views,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=fake_views,
        ),
    ):
        c._preprocess(frames, frame_idxs)

    call_kwargs = mock_li.call_args[1]
    assert call_kwargs.get("resize_mode") == "longest_side"
    assert call_kwargs.get("size") == 512
    assert "resolution_set" not in call_kwargs


def test_mapanything_postprocess_calls_shared_mv_conf(monkeypatch):
    """After refactor, use_multiview_confidence calls compute_multiview_depth_confidence,
    not the upstream postprocess_model_outputs_for_inference with use_multiview_confidence=True."""
    from unittest.mock import patch

    import numpy as np

    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    upstream_calls = []

    def fake_postprocess(raw_outputs, processed_views, **kwargs):
        upstream_calls.append(kwargs.get("use_multiview_confidence", False))
        N = len(raw_outputs)
        H, W = 4, 4
        import torch

        preds = []
        for _ in range(N):
            preds.append(
                {
                    "mask": [torch.ones(1, H, W, 1)],
                    "depth_z": [torch.ones(1, H, W, 1) * 2.0],
                    "pts3d": [torch.zeros(1, H, W, 3)],
                    "img_no_norm": [torch.zeros(1, H, W, 3)],
                    "camera_poses": [torch.eye(4).unsqueeze(0)],
                    "intrinsics": [torch.eye(3).unsqueeze(0)],
                }
            )
        return preds

    mv_conf_return = np.ones((2, 4, 4), dtype=np.float32)

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            side_effect=fake_postprocess,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.compute_multiview_depth_confidence",
            return_value=mv_conf_return,
        ) as mock_mv,
    ):
        creator = MapAnythingCreator(use_multiview_confidence=True)
        H, W = 4, 4
        creator._processed_views = [{"img": np.zeros((1, 3, H, W), dtype=np.float32)} for _ in range(2)]
        creator.image_paths = []
        creator.original_coords = np.zeros((2, 6), dtype=np.float32)

        raw_outputs = [{"dummy": i} for i in range(2)]
        result = creator._postprocess(raw_outputs)

    assert all(
        not v for v in upstream_calls
    ), f"postprocess_model_outputs_for_inference called with use_multiview_confidence=True: {upstream_calls}"
    mock_mv.assert_called_once()


########################################################################
########## _lc_collate_outputs: depth keys for _raw_to_world_points ####
########################################################################


def _make_creator():
    """Bare MapAnythingCreator without __init__ (no model load)."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    return object.__new__(MapAnythingCreator)


def _fake_frames(n_frames: int, h: int, w: int):
    """Build (raw_list, processed) pairs mimicking MapAnything forward/postprocess."""
    # Raw preds only need the keys the float-cast touches; postprocess is patched.
    raw_list = [{"pts3d_cam": torch.zeros(1, h, w, 3), "pts3d": torch.zeros(1, h, w, 3)} for _ in range(n_frames)]
    # Identity-like K (fx=fy=1, cx=cy=0) so unprojection of depth=1 gives (u, v, 1).
    intr = torch.eye(3)
    processed = []
    for i in range(n_frames):
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0]) * i  # frame 0 identity, frame 1 shifted
        processed.append(
            {
                "camera_poses": c2w.unsqueeze(0),
                "intrinsics": intr.unsqueeze(0),
                "depth_z": torch.ones(1, h, w, 1, dtype=torch.bfloat16),
                "conf": torch.full((1, h, w), 0.5 + 0.25 * i, dtype=torch.bfloat16),
            }
        )
    return raw_list, processed


def test_lc_collate_outputs_carries_depth_keys():
    """Collated dict exposes depth/intrinsics_downsampled/depth_conf with correct shapes."""
    h, w, n = 16, 24, 2
    raw_list, processed = _fake_frames(n, h, w)
    creator = _make_creator()
    creator._lc_window_views = [object()] * n
    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        return_value=processed,
    ):
        out = creator._lc_collate_outputs(raw_list)

    # Existing pose keys still present with correct shapes
    assert out["extrinsic"].shape == (n, 3, 4)
    assert out["intrinsics"].shape == (n, 3, 3)

    # New depth keys: shapes, dtypes, and intrinsics_downsampled == intrinsics
    assert out["depth"].shape == (n, h, w, 1)
    assert out["depth"].dtype == np.float32
    assert out["depth_conf"].shape == (n, h, w)
    assert out["depth_conf"].dtype == np.float32
    np.testing.assert_array_equal(out["intrinsics_downsampled"], out["intrinsics"])


def test_lc_collate_outputs_feeds_raw_to_world_points():
    """Collated dict unprojects: identity pose + depth=1 grid → world points at pixel coords."""
    h, w, n = 16, 24, 2
    raw_list, processed = _fake_frames(n, h, w)
    creator = _make_creator()
    creator._lc_window_views = [object()] * n
    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        return_value=processed,
    ):
        out = creator._lc_collate_outputs(raw_list)

    wp, wp_conf = _raw_to_world_points(out, subsample=8)
    assert wp is not None, "world points must not be None with depth keys present"

    # Grid size: strided pixel lattice arange(0, W, 8) x arange(0, H, 8)
    us, vs = np.arange(0, w, 8), np.arange(0, h, 8)
    p = len(us) * len(vs)
    assert wp.shape == (n, p, 3)
    assert wp_conf.shape == (n, p)

    # Frame 0: identity extrinsic + identity K + depth=1 → world point (u, v, 1)
    uu, vv = np.meshgrid(us, vs)
    expected0 = np.stack([uu.ravel(), vv.ravel(), np.ones(p)], axis=-1).astype(np.float32)
    np.testing.assert_allclose(wp[0], expected0, atol=1e-5)

    # Frame 1: cam2world translation (1, 2, 3) shifts every world point
    np.testing.assert_allclose(wp[1], expected0 + np.array([1.0, 2.0, 3.0]), atol=1e-4)

    # Confidence passthrough per frame
    np.testing.assert_allclose(wp_conf[0], 0.5, atol=1e-6)
    np.testing.assert_allclose(wp_conf[1], 0.75, atol=1e-6)


def test_lc_collate_outputs_warns_when_depth_z_missing(caplog):
    """Missing depth_z/conf in postprocess output logs a warning; poses still returned."""
    h, w, n = 16, 24, 2
    raw_list, processed = _fake_frames(n, h, w)
    # Strip the geometry keys — some postprocess variants omit them
    for p in processed:
        del p["depth_z"]
        del p["conf"]
    creator = _make_creator()
    creator._lc_window_views = [object()] * n
    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            return_value=processed,
        ),
        caplog.at_level(logging.WARNING, logger="collab_splats.pointcloud.feedforward.mapanything"),
    ):
        out = creator._lc_collate_outputs(raw_list)

    # Warning names the missing keys and the consequence
    assert any(
        "depth_z" in rec.message and "world_points" in rec.message
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    ), f"expected depth_z warning, got: {[r.message for r in caplog.records]}"

    # Pose keys survive; geometry keys omitted rather than raising
    assert out["extrinsic"].shape == (n, 3, 4)
    assert out["intrinsics"].shape == (n, 3, 3)
    assert "depth" not in out
    assert "depth_conf" not in out


########################################################################
########## extract_intermediate_features: warn on missing pts3d ########
########################################################################


def test_extract_features_warns_when_pts3d_missing(caplog):
    """Missing pts3d in postprocess output logs a loud warning; poses still returned."""
    h = w = 8
    preds = [{"pts3d_cam": torch.zeros(1, h, w, 3), "pts3d": torch.zeros(1, h, w, 3)} for _ in range(2)]
    # Postprocessed output WITHOUT pts3d — geometry unavailable, poses intact.
    processed = [{"camera_poses": torch.eye(4).unsqueeze(0), "conf": torch.rand(1, h, w)} for _ in range(2)]
    creator = _make_creator()
    creator.model = _FakeMapAnythingModel(preds)
    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            side_effect=lambda v: v,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            return_value=processed,
        ),
        caplog.at_level(logging.WARNING, logger="collab_splats.pointcloud.feedforward.mapanything"),
    ):
        out = creator.extract_intermediate_features(torch.rand(2, 3, h, w))

    # Warning names the missing key and the consequence
    assert any(
        "pts3d" in rec.message and "anchor scale" in rec.message
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    ), f"expected pts3d warning, got: {[r.message for r in caplog.records]}"

    # Poses still derived from the same forward; geometry keys absent
    assert out["poses"].shape == (2, 4, 4)
    assert "world_points" not in out
