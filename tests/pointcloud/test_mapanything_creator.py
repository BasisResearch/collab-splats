import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from collab_splats.pointcloud.feedforward import MapAnythingCreator, BaseFeedforwardCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


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


def test_mapanything_preprocess_sets_processed_views(tmp_path):
    import torch
    from PIL import Image as PILImage

    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    for i in range(2):
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
            image_dir / f"frame_{i:04d}.jpg"
        )

    fake_views = [{"img": torch.zeros(1, 3, 224, 224), "data_norm_type": "imagenet"}
                  for _ in range(2)]
    fake_validated = fake_views
    fake_processed = [{"img": torch.zeros(1, 3, 224, 224)} for _ in range(2)]

    with patch("collab_splats.pointcloud.feedforward.mapanything.load_images",
               return_value=fake_views), \
         patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
               return_value=fake_validated) as mock_validate, \
         patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=fake_processed) as mock_preprocess:
        creator = MapAnythingCreator()
        creator._preprocess(image_dir)

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

    with patch("collab_splats.pointcloud.feedforward.mapanything"
               ".postprocess_model_outputs_for_inference",
               return_value=fake_processed) as mock_post:
        creator._postprocess(raw_outputs)

    # raw_outputs is mutated in-place before postprocess is called;
    # mock captures the reference so we inspect dtype at call time
    called_raw = mock_post.call_args[0][0]
    for pred in called_raw:
        assert pred["pts3d_cam"].dtype == torch.float32, (
            f"pts3d_cam not cast to float32 before postprocess: {pred['pts3d_cam'].dtype}"
        )
        assert pred["pts3d"].dtype == torch.float32, (
            f"pts3d not cast to float32 before postprocess: {pred['pts3d'].dtype}"
        )


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
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
            image_dir / f"frame_{i:04d}.jpg"
        )

    n, h, w = 2, 8, 8
    fake_views = [{"img": torch.zeros(1, 3, h, w), "data_norm_type": "imagenet"}
                  for _ in range(n)]
    fake_processed = [{"img": torch.zeros(1, 3, h, w)} for _ in range(n)]

    # Raw outputs from model.forward() — bf16 as produced under autocast
    fake_raw = [
        {
            "pts3d_cam": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
            "pts3d": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
            "ray_directions": torch.zeros(1, h, w, 3, dtype=torch.bfloat16),
            "depth_along_ray": torch.ones(1, h, w, 1, dtype=torch.bfloat16),
            "cam_trans": torch.zeros(1, 3, dtype=torch.bfloat16),
            "cam_quats": torch.tensor([[1., 0., 0., 0.]], dtype=torch.bfloat16),
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

    with patch("collab_splats.pointcloud.feedforward.mapanything.load_images",
               return_value=fake_views), \
         patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
               return_value=fake_views), \
         patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=fake_processed), \
         patch("collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
               return_value=fake_post):

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
        assert creator.outputs.pts3d.shape[1] == 3
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
    raw_outputs[0]["depth_z"][0, :h//2, :, 0] = 0.0
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
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [0., 1., 2.]]
    ).reshape(h, w, 3)
    extrinsics_3x4 = np.eye(4)[:3, :][np.newaxis].astype(np.float32)
    intrinsics = np.eye(3)[np.newaxis].astype(np.float32)

    pts3d, _ = creator._reproject(raw_outputs, extrinsics_3x4, intrinsics)

    expected = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [0., 1., 2.]], dtype=np.float32)
    np.testing.assert_allclose(pts3d, expected, atol=1e-5)



def test_mapanything_forward_transfers_tensors_to_device():
    """_forward must move _processed_views tensors to model device before model.forward()."""
    import torch
    from contextlib import nullcontext

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
        f"Expected meta device, got {captured_views[0]['img'].device.type} — "
        "tensor was not transferred in _forward"
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
    assert result.frame == CoordinateFrame.NERFSTUDIO
    assert result.world_transform is not None
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

    assert creator.outputs.conf is not None, "conf should be set when use_multiview_confidence=True"
    assert not creator.outputs.conf.isnan().any(), (
        "conf contains NaN — dtype cast regression in _postprocess"
    )


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
        return [{}]

    mock_model = MagicMock()
    mock_model.info_sharing.self_attention_blocks = blocks
    mock_model.forward.side_effect = mock_forward

    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator.model = mock_model
    return creator, blocks, n_tokens, num_heads, head_dim


def test_mapanything_extract_intermediate_features_shapes():
    """Hook fires on last block; q/k shapes correct; no poses key."""
    import torch
    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)

    with patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=[{}] * 2):
        result = creator.extract_intermediate_features(frames, layer_index=-1)

    assert "q" in result and "k" in result
    assert "poses" not in result  # MapAnything has no pose_enc to decode
    assert result["q"].shape == (1, num_heads, n_tokens, head_dim)
    assert result["k"].shape == (1, num_heads, n_tokens, head_dim)


def test_mapanything_extract_intermediate_features_hook_removed():
    """Hook removed after call — no accumulated hooks on repeated calls."""
    import torch
    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)
    qkv = blocks[-1].attn.qkv
    assert len(qkv._forward_hooks) == 0

    with patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=[{}] * 2):
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
    with patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=[{}] * 2):
        creator.extract_intermediate_features(frames, layer_index=0)

    assert hooked_blocks == [0]


def test_mapanything_verify_loop_candidate_not_on_class():
    """MapAnythingCreator must not define its own _verify_loop_candidate."""
    assert "_verify_loop_candidate" not in MapAnythingCreator.__dict__, (
        "_verify_loop_candidate still defined on MapAnythingCreator — base class only"
    )


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

    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig
    from collab_splats.pointcloud.wrappers import LoopClosure

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


def test_mapanything_preprocess_fixed_mode_calls_load_images_with_fixed_mapping(tmp_path):
    """resize_mode='fixed' passes resize_mode='fixed_mapping' + resolution_set to load_images."""
    from PIL import Image as PILImage
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(2):
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_dir / f"f{i}.jpg")

    fake_view = {"img": __import__("torch").zeros(1, 3, 64, 64), "data_norm_type": "imagenet"}
    fake_views = [fake_view, fake_view]

    c = MapAnythingCreator(resize_mode="fixed", resolution=518)
    with patch("collab_splats.pointcloud.feedforward.mapanything.load_images",
               return_value=fake_views) as mock_li, \
         patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
               return_value=fake_views), \
         patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=fake_views):
        c._preprocess(img_dir)

    call_kwargs = mock_li.call_args[1]
    assert call_kwargs.get("resize_mode") == "fixed_mapping"
    assert call_kwargs.get("resolution_set") == 518
    assert "size" not in call_kwargs


def test_mapanything_preprocess_longest_side_calls_load_images_with_size(tmp_path):
    """resize_mode='longest_side' passes resize_mode='longest_side' + size= to load_images."""
    from PIL import Image as PILImage
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(2):
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_dir / f"f{i}.jpg")

    fake_view = {"img": __import__("torch").zeros(1, 3, 64, 64), "data_norm_type": "imagenet"}
    fake_views = [fake_view, fake_view]

    c = MapAnythingCreator(resize_mode="longest_side", resolution=512)
    with patch("collab_splats.pointcloud.feedforward.mapanything.load_images",
               return_value=fake_views) as mock_li, \
         patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
               return_value=fake_views), \
         patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=fake_views):
        c._preprocess(img_dir)

    call_kwargs = mock_li.call_args[1]
    assert call_kwargs.get("resize_mode") == "longest_side"
    assert call_kwargs.get("size") == 512
    assert "resolution_set" not in call_kwargs
