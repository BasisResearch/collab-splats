from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from mapanything.utils import cropping

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward import (
    BaseFeedforwardCreator,
    MapAnythingCreator,
)
from tests.pointcloud.conftest import _frame_files
from tests.pointcloud.feedforward.conftest import (
    _assert_views_equal,
    _FakeMapAnythingModel,
    _mapanything_boxes,
)


def _centered_k(h: int, w: int) -> torch.Tensor:
    """Plausible pinhole K for an h x w grid: focal = the long side, principal point centered."""
    f = float(max(h, w))
    return torch.tensor([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def test_mapanything_defaults():
    c = MapAnythingCreator()
    assert c.model_name == "facebook/map-anything"
    assert c.conf_percentile == 35.0
    assert c.min_views == 0
    assert c.mv_rel_thresh == 0.01
    assert c.minibatch_size == 1


def test_mapanything_is_feedforward_creator():
    assert issubclass(MapAnythingCreator, BaseFeedforwardCreator)


def test_mapanything_missing_image_dir_raises(tmp_path):
    c = MapAnythingCreator()
    with pytest.raises(FileNotFoundError):
        c.create_pointcloud(
            tmp_path / "nonexistent", tmp_path / "out", tmp_path / "model"
        )


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
    creator.views = [object()] * n

    with patch.object(
        MapAnythingCreator, "_stack_predictions", return_value="stacked"
    ) as mock_stack:
        result = creator._forward(mock_model, creator.views)

    mock_model.forward.assert_called_once_with(
        creator._processed_views,
        memory_efficient_inference=True,
        minibatch_size=2,
    )
    mock_stack.assert_called_once_with(mock_raw, creator._processed_views, masked=True)
    assert result == "stacked"


def test_mapanything_preprocess_sets_processed_views(tmp_path):
    import torch

    paths = _frame_files(
        [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)], tmp_path
    )

    fake_views = [
        {"img": torch.zeros(1, 3, 224, 224), "data_norm_type": "imagenet"}
        for _ in range(2)
    ]
    fake_validated = fake_views
    fake_processed = [{"img": torch.zeros(1, 3, 224, 224)} for _ in range(2)]

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.load_images",
            return_value=fake_views,
        ),
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
        creator._preprocess(paths)

    mock_validate.assert_called_once_with(fake_views)
    mock_preprocess.assert_called_once_with(fake_validated)
    assert creator._processed_views is fake_processed


def test_mapanything_postprocess_casts_bf16_to_float32():
    """Regression guard: pts3d_cam and pts3d must be float32 before postprocess.

    The shared multiview depth check requires matching
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
            "conf": torch.ones(1, h, w),
            "img_no_norm": torch.zeros(1, h, w, 3),
            # Centered pinhole K at the fixture resolution. Identity K puts the principal point
            # at (0, 0), which the mv resolution contract rejects as a K built for another grid.
            "intrinsics": _centered_k(h, w).unsqueeze(0),
            "camera_poses": torch.eye(4).unsqueeze(0),
        }
        for _ in range(n)
    ]

    creator = MapAnythingCreator()
    views = [{"img": torch.zeros(1, 3, h, w)} for _ in range(n)]

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything"
        ".postprocess_model_outputs_for_inference",
        return_value=fake_processed,
    ) as mock_post:
        creator._stack_predictions(raw_outputs, views, masked=True)

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
    fake_views = [
        {"img": torch.zeros(1, 3, h, w), "data_norm_type": "imagenet"} for _ in range(n)
    ]
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
            "conf": torch.ones(1, h, w),
            "img_no_norm": torch.zeros(1, h, w, 3),
            # Centered pinhole K at the fixture resolution. Identity K puts the principal point
            # at (0, 0), which the mv resolution contract rejects as a K built for another grid.
            "intrinsics": _centered_k(h, w).unsqueeze(0),
            "camera_poses": torch.eye(4).unsqueeze(0),
        }
        for _ in range(n)
    ]

    param = torch.zeros(1)
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([param])
    mock_model.forward.return_value = fake_raw

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.load_images",
            return_value=fake_views,
        ),
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

        creator.setup_inference(sorted(image_dir.glob("*.jpg")))
        assert hasattr(creator, "_processed_views"), (
            "_preprocess must set _processed_views"
        )
        assert creator._processed_views is fake_processed

        creator.run_inference()
        mock_model.forward.assert_called_once_with(
            fake_processed,
            memory_efficient_inference=True,
            minibatch_size=1,
        )

        result = creator._postprocess(creator.raw_outputs)
        assert result.points.shape[1] == 3
        assert result.extrinsics.shape == (n, 4, 4)


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
    creator.views = [object()]

    # Mock torch.autocast to avoid "unsupported autocast device_type 'meta'" error.
    # autocast is only for dtype conversion; tensor transfer happens before it.
    with (
        patch("torch.autocast", return_value=nullcontext()),
        patch.object(MapAnythingCreator, "_stack_predictions"),
    ):
        creator._forward(mock_model, creator.views)

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

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.MapAnything"
    ) as mock_cls:
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
    result = c.create_pointcloud(image_dir, tmp_path / "out", tmp_path / "model")
    assert isinstance(result, PointcloudResult)
    assert result.points.shape[1] == 3
    assert (tmp_path / "model" / "cameras.bin").exists()


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

    creator = MapAnythingCreator()
    result = creator.create_pointcloud(bicycle, tmp_path / "out")

    assert result.confidence is not None, (
        "MapAnything postprocess should populate confidence"
    )
    assert not result.confidence.isnan().any(), (
        "confidence contains NaN — dtype cast regression in _stack_predictions"
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
    qkv_linears = [
        nn.Linear(total_dim, total_dim * 3, bias=False) for _ in range(n_blocks)
    ]
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
        return [
            {"pts3d_cam": torch.zeros(1, 4, 4, 3), "pts3d": torch.zeros(1, 4, 4, 3)}
            for _ in range(2)
        ]

    mock_model = MagicMock()
    mock_model.info_sharing.self_attention_blocks = blocks
    mock_model.forward.side_effect = mock_forward

    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator.model = mock_model
    return creator, blocks, n_tokens, num_heads, head_dim


def _patch_mapanything_postprocess():
    """Patch module-level postprocess to yield identity camera_poses plus geometry for 2 views."""
    import torch

    return patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        return_value=[
            {
                "camera_poses": torch.eye(4).unsqueeze(0),
                "pts3d": torch.zeros(1, 4, 4, 3),
                "conf": torch.ones(1, 4, 4),
            }
            for _ in range(2)
        ],
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

    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_with_mock_model(
        n_blocks=3
    )
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


@pytest.mark.gpu
def test_mapanything_run_inference_loop_closure_smoke(tmp_path):
    pytest.importorskip("mapanything")
    pytest.importorskip("torch")
    bicycle = Path("/workspace/bicycle/images_4")
    if not bicycle.exists():
        pytest.skip(f"{bicycle} not available on this host")

    from collab_splats.geometry.loop_closure import LoopClosureConfig
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure
    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    base = MapAnythingCreator()
    creator = LoopClosure(base, config=LoopClosureConfig())
    result = creator.create_pointcloud(bicycle, tmp_path / "out")
    assert result.points.shape[1] == 3


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


def test_mapanything_preprocess_fixed_mode_calls_load_images_with_fixed_mapping(
    tmp_path,
):
    """resize_mode='fixed' passes resize_mode='fixed_mapping' + resolution_set to load_images."""
    paths = _frame_files(
        [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)], tmp_path
    )

    fake_view = {
        "img": __import__("torch").zeros(1, 3, 64, 64),
        "data_norm_type": "imagenet",
    }
    fake_views = [fake_view, fake_view]

    c = MapAnythingCreator(resize_mode="fixed", resolution=518)
    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.load_images",
            return_value=fake_views,
        ) as mock_li,
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
            return_value=fake_views,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=fake_views,
        ),
    ):
        c._preprocess(paths)

    call_kwargs = mock_li.call_args[1]
    assert call_kwargs.get("resize_mode") == "fixed_mapping"
    assert call_kwargs.get("resolution_set") == 518
    assert "size" not in call_kwargs


def test_mapanything_preprocess_longest_side_calls_load_images_with_size(tmp_path):
    """resize_mode='longest_side' passes resize_mode='longest_side' + size= to load_images."""
    paths = _frame_files(
        [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)], tmp_path
    )

    fake_view = {
        "img": __import__("torch").zeros(1, 3, 64, 64),
        "data_norm_type": "imagenet",
    }
    fake_views = [fake_view, fake_view]

    c = MapAnythingCreator(resize_mode="longest_side", resolution=512)
    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.load_images",
            return_value=fake_views,
        ) as mock_li,
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
            return_value=fake_views,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            return_value=fake_views,
        ),
    ):
        c._preprocess(paths)

    call_kwargs = mock_li.call_args[1]
    assert call_kwargs.get("resize_mode") == "longest_side"
    assert call_kwargs.get("size") == 512
    assert "resolution_set" not in call_kwargs


def test_mapanything_postprocess_calls_shared_mv_conf(monkeypatch):
    """min_views > 0 calls the shared multiview_depth_confidence,
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
                    "mask": torch.ones(1, H, W, 1),
                    "depth_z": torch.ones(1, H, W, 1) * 2.0,
                    "conf": torch.ones(1, H, W),
                    "pts3d": torch.zeros(1, H, W, 3),
                    "img_no_norm": torch.zeros(1, H, W, 3),
                    "camera_poses": torch.eye(4).unsqueeze(0),
                    "intrinsics": torch.eye(3).unsqueeze(0),
                }
            )
        return preds

    ones = np.ones((2, 4, 4), dtype=np.int32)

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            side_effect=fake_postprocess,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.base.multiview_depth_confidence",
            return_value=(ones, ones),
        ) as mock_mv,
    ):
        creator = MapAnythingCreator(min_views=1)
        H, W = 4, 4
        creator._processed_views = [
            {"img": np.zeros((1, 3, H, W), dtype=np.float32)} for _ in range(2)
        ]
        creator.image_paths = []
        creator.original_coords = np.tile(
            np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (2, 1)
        )  # full-frame box

        raw_outputs = [
            {"pts3d_cam": torch.zeros(1), "pts3d": torch.zeros(1)} for _ in range(2)
        ]
        raw = creator._stack_predictions(
            raw_outputs, creator._processed_views, masked=True
        )
        creator._postprocess(raw)

    assert all(not v for v in upstream_calls), (
        f"postprocess_model_outputs_for_inference called with use_multiview_confidence=True: {upstream_calls}"
    )
    mock_mv.assert_called_once()


def test_mapanything_postprocess_skips_the_multiview_pass_when_disabled():
    """min_views=0 is off: the O(N^2) pass never runs."""
    creator = MapAnythingCreator(min_views=0)
    H, W = 4, 4
    n = 2

    def fake_postprocess(raw_outputs, processed_views, **kwargs):
        return [
            {
                "mask": torch.ones(1, H, W, 1),
                "depth_z": torch.ones(1, H, W, 1) * 2.0,
                "conf": torch.ones(1, H, W),
                "pts3d": torch.zeros(1, H, W, 3),
                "img_no_norm": torch.zeros(1, H, W, 3),
                "camera_poses": torch.eye(4).unsqueeze(0),
                "intrinsics": _centered_k(H, W).unsqueeze(0),
            }
            for _ in range(len(raw_outputs))
        ]

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            side_effect=fake_postprocess,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.base.multiview_depth_confidence"
        ) as mock_mv,
    ):
        views = [{"img": np.zeros((1, 3, H, W), dtype=np.float32)} for _ in range(n)]
        creator.image_paths = []
        creator.original_coords = np.tile(
            np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n, 1)
        )  # full-frame box
        preds = [
            {"pts3d_cam": torch.zeros(1), "pts3d": torch.zeros(1)} for _ in range(n)
        ]
        creator._postprocess(creator._stack_predictions(preds, views, masked=True))

    mock_mv.assert_not_called()


########################################################################
########## _stack_predictions: the base raw dict #######################
########################################################################


def _make_creator():
    """Bare MapAnythingCreator without __init__ (no model load)."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    return object.__new__(MapAnythingCreator)


def _fake_frames(n_frames: int, h: int, w: int):
    """Build (raw_list, processed) pairs mimicking MapAnything forward/postprocess."""
    # Raw preds only need the keys the float-cast touches; postprocess is patched.
    raw_list = [
        {"pts3d_cam": torch.zeros(1, h, w, 3), "pts3d": torch.zeros(1, h, w, 3)}
        for _ in range(n_frames)
    ]

    # Upstream mask keeps the left half of every frame
    mask = torch.zeros(1, h, w, 1, dtype=torch.bool)
    mask[:, :, : w // 2] = True

    processed = []
    for i in range(n_frames):
        c2w = torch.eye(4)
        c2w[:3, 3] = (
            torch.tensor([1.0, 2.0, 3.0]) * i
        )  # frame 0 identity, frame 1 shifted
        processed.append(
            {
                "camera_poses": c2w.unsqueeze(0),
                "intrinsics": _centered_k(h, w).unsqueeze(0),
                # postprocess always adds denormalized [0, 1] RGB (used for dense colors)
                "img_no_norm": torch.zeros(1, h, w, 3),
                "depth_z": torch.ones(1, h, w, 1, dtype=torch.bfloat16),
                "conf": torch.full((1, h, w), 0.5 + 0.25 * i, dtype=torch.bfloat16),
                "mask": mask,
            }
        )
    return raw_list, processed


def _stack(masked: bool, n: int = 2, h: int = 16, w: int = 24) -> dict:
    """_stack_predictions over _fake_frames with upstream postprocess patched."""
    raw_list, processed = _fake_frames(n, h, w)
    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        return_value=processed,
    ):
        return _make_creator()._stack_predictions(
            raw_list, [object()] * n, masked=masked
        )


def test_stack_predictions_unmasked_carries_the_base_raw_keys():
    """LC window stack: every base raw key at base shapes, no upstream mask."""
    h, w, n = 16, 24, 2
    out = _stack(masked=False, n=n, h=h, w=w)

    assert out["extrinsic"].shape == (n, 3, 4)
    assert out["intrinsics"].shape == (n, 3, 3)
    assert out["depth"].shape == (n, h, w, 1)
    assert out["depth"].dtype == np.float32
    assert out["depth_conf"].shape == (n, h, w)
    assert out["depth_conf"].dtype == np.float32
    assert out["images"].shape == (n, 3, h, w)
    assert "mask" not in out


def test_stack_predictions_inverts_c2w_to_w2c():
    """Upstream camera_poses are c2w; the stacked extrinsic is w2c."""
    out = _stack(masked=False)

    np.testing.assert_allclose(out["extrinsic"][0], np.eye(4)[:3], atol=1e-6)
    np.testing.assert_allclose(out["extrinsic"][1][:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_stack_predictions_masked_keeps_the_upstream_mask():
    """Full-sequence stack carries upstream's mask as (N, H, W) bool."""
    h, w, n = 16, 24, 2
    out = _stack(masked=True, n=n, h=h, w=w)

    assert out["mask"].shape == (n, h, w)
    assert out["mask"].dtype == bool
    assert out["mask"][:, :, : w // 2].all() and not out["mask"][:, :, w // 2 :].any()


def test_postprocess_mask_key_replaces_the_confidence_cutoff():
    """A raw "mask" key decides the kept pixels; conf_percentile is not applied."""
    h, w, n = 16, 24, 2
    raw = _stack(masked=True, n=n, h=h, w=w)
    creator = MapAnythingCreator(
        min_views=0, conf_percentile=100.0
    )  # nothing is above the 100th percentile: would keep nothing
    creator.image_paths = []
    creator.original_coords = np.tile(
        np.array([0, 0, w, h, w, h], dtype=np.float32), (n, 1)
    )  # full-frame box

    result = creator._postprocess(raw)

    assert len(result.points) == n * h * (w // 2)
    assert (result.pixel_indices[:, 2] < w // 2).all()


def test_stack_predictions_raises_when_depth_z_missing():
    """Missing depth_z/conf in postprocess output raises instead of dropping submap geometry."""
    h, w, n = 16, 24, 2
    raw_list, processed = _fake_frames(n, h, w)

    # Strip the geometry keys
    for p in processed:
        del p["depth_z"]
        del p["conf"]
    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            return_value=processed,
        ),
        pytest.raises(KeyError, match="depth_z"),
    ):
        _make_creator()._stack_predictions(raw_list, [object()] * n, masked=False)


########################################################################
########## Missing upstream keys raise, never degrade ##################
########################################################################


def test_extract_raises_when_pts3d_missing():
    """Postprocess output without pts3d raises instead of dropping LC anchor geometry."""
    h = w = 8
    preds = [
        {"pts3d_cam": torch.zeros(1, h, w, 3), "pts3d": torch.zeros(1, h, w, 3)}
        for _ in range(2)
    ]
    processed = [
        {"camera_poses": torch.eye(4).unsqueeze(0), "conf": torch.rand(1, h, w)}
        for _ in range(2)
    ]
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
        pytest.raises(KeyError, match="pts3d"),
    ):
        creator.extract_intermediate_features(torch.rand(2, 3, h, w), layer_index=-1)


def test_postprocess_raises_when_conf_missing():
    """Postprocess output without conf raises instead of returning confidence=None."""
    h = w = 4
    processed = [
        {
            "mask": torch.ones(1, h, w, 1),
            "depth_z": torch.ones(1, h, w, 1),
            "pts3d": torch.zeros(1, h, w, 3),
            "img_no_norm": torch.zeros(1, h, w, 3),
            "camera_poses": torch.eye(4).unsqueeze(0),
            "intrinsics": _centered_k(h, w).unsqueeze(0),
        }
        for _ in range(2)
    ]
    creator = MapAnythingCreator(min_views=0)
    views = [{"img": torch.zeros(1, 3, h, w)} for _ in range(2)]
    preds = [{"pts3d_cam": torch.zeros(1), "pts3d": torch.zeros(1)} for _ in range(2)]
    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            return_value=processed,
        ),
        pytest.raises(KeyError, match="conf"),
    ):
        creator._stack_predictions(preds, views, masked=True)


########################################################################
########## MapAnything crop box (MapAnythingCreator._preprocess) #######
########################################################################


@pytest.mark.parametrize("hw", [(1000, 1000), (1080, 1920), (1920, 1080), (200, 300)])
def test_mapanything_crop_box_matches_upstream_crop(hw):
    """The box, cropped and resized with PIL, reproduces upstream's model-res image."""
    h, w = hw
    # Runtime target is aspect-matched to the frame: a portrait frame gets a portrait
    # target (294 wide x 518 high), never the landscape 518x294 the creator produces
    # for landscape/square frames.
    model_w, model_h = (294, 518) if h > w else (518, 294)
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack(
        [(xx * 255 // w), (yy * 255 // h), ((xx + yy) % 256)], axis=-1
    ).astype(np.uint8)

    upstream = np.asarray(
        cropping.crop_resize_if_necessary(rgb, resolution=(model_w, model_h))[0],
        dtype=np.float32,
    )
    (box,) = _mapanything_boxes([(w, h)], model_w, model_h)
    ours = np.asarray(
        Image.fromarray(rgb)
        .crop(tuple(float(v) for v in box[:4]))
        .resize((model_w, model_h), Image.LANCZOS),
        dtype=np.float32,
    )

    assert box[4:].tolist() == [w, h]
    # Measured worst case is 0.228 (200x300); 0.3 still catches a 3-px box shift or a
    # dropped centering pass (see the mutant table in the consistency review report).
    assert np.abs(ours[..., :2] - upstream[..., :2]).mean() < 0.3


########################################################################
########## in-memory frame handoff #####################################
########################################################################


@pytest.mark.parametrize(
    "mode, resolution",
    [
        ("fixed", 518),
        ("fixed", 512),
        ("longest_side", 518),
        ("longest_side", 252),
        ("square", 250),
    ],
)
@pytest.mark.parametrize(
    "sizes",
    [
        [(96, 48)] * 9,
        [(96, 48)] * 4 + [(48, 96)] * 3,
        [(200, 50)] * 3,
        [(48, 120)] * 2,
        [(1920, 1080), (1080, 1920), (1920, 1080)],
    ],
)
def test_preprocess_frames_match_files(tmp_path, mode, resolution, sizes):
    """
    Handed-off arrays preprocess to the same view dicts and boxes as the PNG files.
    """
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for w, h in sizes]
    paths = _frame_files(frames, tmp_path)

    from_files = MapAnythingCreator(resize_mode=mode, resolution=resolution)
    views_f, coords_f = from_files._preprocess(paths)

    from_arrays = MapAnythingCreator(resize_mode=mode, resolution=resolution)
    from_arrays.frames = {p.name: f for p, f in zip(paths, frames, strict=True)}
    views_a, coords_a = from_arrays._preprocess(paths)

    _assert_views_equal(views_a, views_f)
    np.testing.assert_array_equal(coords_a, coords_f)

    # The model-ready copy the forward pass reads matches too
    for a, f in zip(
        from_arrays._processed_views, from_files._processed_views, strict=True
    ):
        assert torch.equal(a["img"], f["img"])


########################################################################
########## LC window collate ###########################################
########################################################################


def _make_mock_processed_view(tag: str) -> dict:
    """Processed view carrying only a tag; postprocess is patched out."""
    return {"img": torch.zeros(1, 3, 4, 4), "_tag": tag}


def test_lc_window_forward_uses_window_views_unmasked():
    """A views slice other than self.views is preprocessed here and stacked without the mask."""
    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator.minibatch_size = 1
    creator.views = [object() for _ in range(10)]
    creator._processed_views = [
        _make_mock_processed_view(f"full_{i}") for i in range(10)
    ]
    window_views = [_make_mock_processed_view(f"win_{i}") for i in range(3)]

    # Model on CPU returning one raw pred per window view
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.forward.return_value = [
        {"pts3d_cam": torch.zeros(1, 3), "pts3d": torch.zeros(1, 3)} for _ in range(3)
    ]

    captured = {}

    def fake_postprocess(preds, views, **kwargs):
        captured["views"] = views
        captured["kwargs"] = kwargs
        return [
            {
                "camera_poses": torch.eye(4).unsqueeze(0),
                "intrinsics": torch.eye(3).unsqueeze(0),
                "img_no_norm": torch.full((1, 4, 4, 3), 0.5),
                "depth_z": torch.ones(1, 4, 4, 1),
                "conf": torch.ones(1, 4, 4),
            }
            for _ in preds
        ]

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
            lambda v: v,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            lambda v: v,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            side_effect=fake_postprocess,
        ),
    ):
        out = creator._forward(model, window_views)

    assert [v["_tag"] for v in captured["views"]] == ["win_0", "win_1", "win_2"]
    assert captured["kwargs"] == {"apply_mask": False}
    assert "mask" not in out
    assert out["depth"].shape == (3, 4, 4, 1)
    assert out["depth_conf"].shape == (3, 4, 4)
