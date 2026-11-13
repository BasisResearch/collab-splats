from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from collab_splats.geometry import LoopClosure
from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.base import BasePointcloudCreator, PointcloudResult
from collab_splats.pointcloud.feedforward import (
    BaseFeedforwardCreator,
    LoGeRCreator,
    MapAnythingCreator,
    VGGTXCreator,
)
from collab_splats.preproc import frames as fr
from tests.pointcloud._stubs import vggt_raw_outputs


def _make_result(n=3, p=50, colors=None):
    """
    n PINHOLE frames on a 512x512 full frame, p random points.
    """
    rng = np.random.default_rng(0)
    intrinsics = np.tile(
        np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    return PointcloudResult(
        points=rng.standard_normal((p, 3)).astype(np.float32),
        colors=rng.integers(0, 255, (p, 3), dtype=np.uint8)
        if colors is None
        else colors,
        extrinsics=np.tile(np.eye(4), (n, 1, 1)).astype(np.float32),
        intrinsics=None,
        model_intrinsics=intrinsics,
        image_paths=[Path(f"frame_{i:04d}.jpg") for i in range(n)],
        original_coords=np.array([[0, 0, 512, 512, 512, 512]] * n, dtype=np.float32),
        model_width=512,
        model_height=512,
    )


def test_camera_image_point_counts():
    recon = _make_result(n=3, p=50).to_colmap()
    assert len(recon.cameras) == 3
    assert len(recon.images) == 3
    assert len(recon.points3D) == 50


def test_to_colmap_rejects_float_colors():
    result = _make_result(p=50, colors=np.full((50, 3), 0.5, dtype=np.float32))
    with pytest.raises(TypeError, match="uint8"):
        result.to_colmap()


# ---------------------------------------------------------------------------
# _verify_loop_candidate on BaseFeedforwardCreator
# ---------------------------------------------------------------------------


class _StubCreator(BaseFeedforwardCreator):
    """Minimal concrete subclass of BaseFeedforwardCreator for base-method tests.

    Implements all abstract methods as no-ops.  Instantiate via object.__new__ to
    bypass the dataclass __init__ (no real model or paths needed for these tests).
    Set creator._stubbed_features before calling _verify_loop_candidate.
    """

    def _load_model(self, device):
        return None

    def _preprocess(self, image_paths, **kwargs):
        pass

    def _forward(self, model, views):
        pass

    def _postprocess(self, raw_outputs):
        pass

    # LC calibration an LC-capable backend sets
    _lc_layer_index = 20
    _lc_token_offset = 5

    def extract_intermediate_features(self, frames, layer_index):
        # Return the pre-configured stub features for this test
        return self._stubbed_features


def _make_stub() -> _StubCreator:
    """Bypass the dataclass __init__; no real model or paths needed."""
    from unittest.mock import MagicMock

    creator = object.__new__(_StubCreator)
    # _verify_loop_candidate calls next(self.model.parameters()).device
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    creator.model = mock_model
    return creator


def _high_ratio_features():
    """Build q/k tensors whose cross-frame attention ratio will be > 0.8."""
    torch.manual_seed(0)
    B, heads, hd = 1, 2, 4
    half = torch.randn(B, heads, 10, hd) * 10.0
    k = torch.cat([half, half], dim=2)
    q = torch.cat([half, half], dim=2)
    return k, q


def _low_ratio_features():
    """Build q/k tensors whose cross-frame attention ratio will be < 0.2."""
    B, heads, hd = 1, 1, 4
    N = 20
    k = torch.zeros(B, heads, N, hd)
    q = torch.zeros(B, heads, N, hd)
    # First frame in dim 0, second frame in orthogonal dim 1
    k[:, :, :10, 0] = 10.0
    q[:, :, :10, 0] = 10.0
    k[:, :, 10:, 1] = 10.0
    q[:, :, 10:, 1] = 10.0
    return k, q


def test_verify_loop_candidate_rejected():
    """Ratio below threshold → (False, None) regardless of poses presence."""
    creator = _make_stub()
    k, q = _low_ratio_features()
    creator._stubbed_features = {"q": q, "k": k}
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    accepted, poses = creator._verify_loop_candidate(
        frame1, frame2, verify_match_ratio=0.85
    )
    assert accepted is False
    assert poses is None


def test_verify_raises_when_backend_supplies_no_poses():
    creator = _make_stub()
    k, q = _high_ratio_features()
    creator._stubbed_features = {"q": q, "k": k}
    frame = torch.zeros(3, 16, 16)
    with pytest.raises(KeyError, match="poses"):
        creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.5)


def test_verify_loop_candidate_accepted_with_geometry():
    """'world_points'/'conf' feature keys are folded into lc_data on accept."""
    creator = _make_stub()
    k, q = _high_ratio_features()
    fake_poses = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)  # (2, 4, 4)
    wp = np.zeros((2, 8, 8, 3), dtype=np.float32)  # (2, H, W, 3)
    conf = np.ones((2, 8, 8), dtype=np.float32)  # (2, H, W)
    creator._stubbed_features = {
        "q": q,
        "k": k,
        "poses": fake_poses,
        "world_points": wp,
        "conf": conf,
    }
    accepted, lc_data = creator._verify_loop_candidate(
        torch.zeros(3, 16, 16),
        torch.zeros(3, 16, 16),
        verify_match_ratio=0.5,
    )
    assert accepted is True
    assert lc_data["world_points"].shape == (2, 8, 8, 3)
    assert lc_data["conf"].shape == (2, 8, 8)


def test_verify_taps_the_calibrated_layer():
    creator = _make_stub()
    seen = {}

    def _extract(frames, layer_index):
        seen["layer"] = layer_index
        k, q = _high_ratio_features()
        return {
            "q": q,
            "k": k,
            "poses": np.zeros((2, 4, 4), np.float32),
            "world_points": np.zeros((2, 1, 1, 3), np.float32),
            "conf": np.ones((2, 1, 1), np.float32),
        }

    creator._lc_layer_index = 7
    creator.extract_intermediate_features = _extract
    frame = torch.zeros(3, 16, 16)
    creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.0)
    assert seen["layer"] == 7


def test_loop_closure_raises_without_calibrated_ratio():
    base = _make_stub()
    with pytest.raises(NotImplementedError, match="default_verify_match_ratio"):
        LoopClosure(base)


def test_extract_intermediate_features_base_raises():
    creator = _make_stub()
    with pytest.raises(NotImplementedError):
        BaseFeedforwardCreator.extract_intermediate_features(
            creator, torch.zeros(2, 3, 4, 4), 0
        )


########################################################
########## _postprocess pixel bookkeeping ##############
########################################################


def _postprocess(
    depth_conf: np.ndarray, images: torch.Tensor, conf_percentile: float
) -> PointcloudResult:
    """
    Base _postprocess over flat depth-2 frames with these confidences and images.
    """
    n, _, h, w = images.shape
    intrinsics = np.array(
        [[w, 0.0, (w - 1) / 2], [0.0, w, (h - 1) / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    raw = {
        "images": images,
        "extrinsic": np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1)),
        "intrinsics": np.tile(intrinsics, (n, 1, 1)),
        "depth": np.full((n, h, w), 2.0, dtype=np.float32),
        "depth_conf": depth_conf,
    }
    creator = object.__new__(_StubCreator)
    creator.conf_percentile = conf_percentile
    creator.min_views = 0
    creator.image_paths = [Path(f"frame_{i:06d}") for i in range(n)]
    creator.original_coords = np.tile(
        np.array([0, 0, w, h, w, h], dtype=np.float32), (n, 1)
    )
    return BaseFeedforwardCreator._postprocess(creator, raw)


def test_postprocess_returns_pixel_indices():
    n, h, w = 3, 8, 8
    depth_conf = np.random.default_rng(0).random((n, h, w)).astype(np.float32)
    out = _postprocess(depth_conf, torch.zeros(n, 3, h, w), conf_percentile=0.5)

    assert out.pixel_indices.shape == (len(out.points), 3)
    assert out.pixel_indices.dtype == np.int32
    assert (out.pixel_indices.min(axis=0) >= 0).all()
    assert (out.pixel_indices.max(axis=0) < [n, h, w]).all()


def test_pixel_indices_align_with_colors():
    """colors[p] must come from the same pixel as pixel_indices[p]."""
    n, h, w = 2, 8, 8

    # Encode pixel identity into image: pixel (r, c) = r*10 + c across all channels
    images = torch.zeros(n, 3, h, w)
    for r in range(h):
        for c in range(w):
            images[:, :, r, c] = (r * 10 + c) / 255.0

    out = _postprocess(
        np.ones((n, h, w), dtype=np.float32), images, conf_percentile=0.0
    )

    images_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    fi, ri, ci = out.pixel_indices.T
    np.testing.assert_array_equal(out.colors, images_np[fi, ri, ci])


########################################################################
########## Creator registry ############################################
########################################################################


def test_get_creator_mapanything():
    assert get_creator("mapanything") is MapAnythingCreator


def test_get_creator_vggtx():
    assert get_creator("vggtx") is VGGTXCreator


def test_get_creator_loger():
    # Not guarded on third_party/LoGeR: the vendored import is deferred into _load_model
    assert get_creator("loger") is LoGeRCreator


def test_get_creator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown 'nonexistent'"):
        get_creator("nonexistent")


def test_all_creators_are_instantiable():
    for name in ("mapanything", "vggtx"):
        cls = get_creator(name)
        instance = cls()
        assert isinstance(instance, BasePointcloudCreator)


def test_registry_holds_exactly_the_feedforward_backends():
    assert set(BaseFeedforwardCreator._registry) == {
        "loger",
        "mapanything",
        "vggt_omega",
        "vggtx",
    }


########################################################################
########## Multiview confidence wiring #################################
########################################################################


def _run(creator: VGGTXCreator, raw: dict | None = None) -> int:
    """
    Point count after _postprocess over the stub raw dict.
    """
    raw = vggt_raw_outputs(n=3, h=8, w=10) if raw is None else raw
    creator.image_paths = [f"frame_{i:06d}" for i in range(3)]
    creator.original_coords = np.tile(
        np.array([0, 0, 10, 8, 10, 8], np.float32), (3, 1)
    )
    return len(creator._postprocess(raw).points)


def test_min_views_zero_is_off():
    with patch(
        "collab_splats.pointcloud.feedforward.base.multiview_depth_confidence"
    ) as mock_mv:
        _run(VGGTXCreator(min_views=0))
    mock_mv.assert_not_called()


def test_min_views_filters_on_disagreement():
    assert _run(VGGTXCreator(min_views=1, mv_rel_thresh=1e-9)) < _run(
        VGGTXCreator(min_views=0)
    )


def _disjoint_views() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two frames whose cameras sit 1000 units apart, so neither sees the other's pixels.
    """
    depth = np.ones((2, 8, 10), np.float32)
    intrinsics = np.tile(
        np.array([[10, 0, 5], [0, 10, 4], [0, 0, 1]], np.float32), (2, 1, 1)
    )
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    extrinsics[1, 0, 3] = 1000.0
    return depth, intrinsics, extrinsics


def test_unseen_pixels_are_kept():
    depth, intrinsics, extrinsics = _disjoint_views()
    on = VGGTXCreator(min_views=2)._multiview_mask(depth, intrinsics, extrinsics)
    off = VGGTXCreator(min_views=0)._multiview_mask(depth, intrinsics, extrinsics)
    assert on.sum() == off.sum() == depth.size


def test_zero_depth_is_dropped_with_the_filter_off():
    raw = vggt_raw_outputs(n=3, h=8, w=10)
    raw["depth"][0, 0, 0] = 0.0
    assert _run(VGGTXCreator(min_views=0), raw) == 3 * 8 * 10 - 1


########################################################################
########## Preprocess store ############################################
########################################################################


@dataclass
class _EchoCreator(BaseFeedforwardCreator):
    """Minimal creator whose _preprocess decodes the files and echoes their dims."""

    def _load_model(self, device: str) -> Any:
        return object()

    def _preprocess(self, paths: list[Path]) -> Any:
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        coords = np.array(
            [[0, 0, f.shape[1], f.shape[0], f.shape[1], f.shape[0]] for f in frames],
            dtype=np.float32,
        )
        return frames, coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        return {}

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> PointcloudResult:
        raise NotImplementedError


# Gappy source indices: a quality filter drops frames, so row position and frame_idx
# diverge and a label taken from the wrong one is visible here.
_FRAME_IDXS = [0, 3, 7, 9]


def _make_images_dir(tmp_path) -> Path:
    """Write a scene images/ directory whose frame indices are non-contiguous."""
    frames = [np.full((32, 48, 3), idx, dtype=np.uint8) for idx in _FRAME_IDXS]
    images_dir = tmp_path / "images"
    fr.write_frames(images_dir, frames, _FRAME_IDXS)
    return images_dir


def test_setup_inference_labels_are_file_stems(tmp_path):
    creator = _EchoCreator()

    creator.setup_inference(fr.frame_paths(_make_images_dir(tmp_path)))

    # Labels are the filename stems — never the row position, which would misjoin poses to frames
    assert [p.name for p in creator.image_paths] == [
        f"frame_{i:06d}" for i in _FRAME_IDXS
    ]
    assert [f.shape for f in creator.views] == [(32, 48, 3)] * 4
    assert creator.original_coords.shape == (4, 6)


def test_setup_inference_reads_pixels_in_filename_order(tmp_path):
    """Frame N's pixels land at row N — a reorder would shift every pose against its frame."""
    creator = _EchoCreator()

    creator.setup_inference(fr.frame_paths(_make_images_dir(tmp_path)))

    # _make_images_dir paints each frame with its own source index
    assert [int(f[0, 0, 0]) for f in creator.views] == _FRAME_IDXS


def test_colmap_image_names_stable_extensionless(tmp_path):
    """Extension-less frame_{idx:06d} labels are valid, stable COLMAP image names."""
    names = [f"frame_{i:06d}" for i in (0, 1)]
    result = PointcloudResult(
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        extrinsics=np.stack([np.eye(4, dtype=np.float32)] * 2),
        intrinsics=np.stack([np.eye(3, dtype=np.float32)] * 2),
        model_intrinsics=np.stack([np.eye(3, dtype=np.float32)] * 2),
        image_paths=[tmp_path / name for name in names],
        original_coords=np.array([[0, 0, 64, 64, 64, 64]] * 2, dtype=np.float32),
        model_width=64,
        model_height=64,
    )

    recon = result.to_colmap()

    got = sorted(recon.images[i].name for i in recon.images)
    assert got == names
