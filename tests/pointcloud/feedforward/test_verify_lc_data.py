"""lc_data verify contract: MapAnything pose derivation + wrapper loop application."""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from collab_splats.geometry.loop_closure import LoopClosureConfig
from collab_splats.geometry.loop_closure.matching import LoopMatch
from collab_splats.geometry.loop_closure.wrapper import LoopClosure
from collab_splats.geometry.transforms import invert_poses
from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator
from tests.pointcloud.feedforward.conftest import _FakeMapAnythingModel, _FakeQKV

########################################################################
########## MapAnything: extract_intermediate_features returns poses ####
########################################################################


def _make_mapanything_creator(preds):
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    creator = object.__new__(MapAnythingCreator)
    creator.model = _FakeMapAnythingModel(preds)
    return creator


def test_mapanything_extract_features_derives_w2c_poses():
    """extract_intermediate_features postprocesses the SAME forward into w2c poses."""
    h = w = 8
    # Raw preds only need the keys the float-cast touches; postprocess is patched.
    preds = [{"pts3d_cam": torch.zeros(1, h, w, 3), "pts3d": torch.zeros(1, h, w, 3)} for _ in range(2)]
    # Postprocessed output: frame 0 at identity, frame 1 translated (c2w).
    c2w_1 = torch.eye(4)
    c2w_1[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    processed = [
        {
            "camera_poses": torch.eye(4).unsqueeze(0),
            "pts3d": torch.rand(1, h, w, 3),
            "conf": torch.rand(1, h, w),
        },
        {
            "camera_poses": c2w_1.unsqueeze(0),
            "pts3d": torch.rand(1, h, w, 3),
            "conf": torch.rand(1, h, w),
        },
    ]
    creator = _make_mapanything_creator(preds)
    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
            side_effect=lambda v: v,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            return_value=processed,
        ) as mock_post,
    ):
        features = creator.extract_intermediate_features(torch.zeros(2, 3, h, w))

    # apply_mask=False — LC needs unmasked dense geometry
    assert mock_post.call_args.kwargs.get("apply_mask") is False
    # Poses: (2, 4, 4) float32 w2c, frame 0 at identity, frame 1 = inv(c2w_1)
    assert features["poses"].shape == (2, 4, 4)
    assert features["poses"].dtype == np.float32
    assert np.allclose(features["poses"][0], np.eye(4), atol=1e-6)
    assert np.allclose(features["poses"][1], invert_poses(c2w_1.numpy()), atol=1e-6)
    # Geometry passthrough: (2, H, W, 3) points + (2, H, W) confidence
    assert features["world_points"].shape == (2, h, w, 3)
    assert features["conf"].shape == (2, h, w)


########################################################################
########## VGGT-Omega: verify path emits world_points + conf ##########
########################################################################


class _FakeOmegaModel(torch.nn.Module):
    """Minimal stand-in: aggregator inter-frame block tree + canned predictions."""

    def __init__(self, preds):
        super().__init__()
        self._preds = preds
        self._param = torch.nn.Parameter(torch.zeros(1))
        attn = types.SimpleNamespace(num_heads=2, qkv=_FakeQKV())
        block = types.SimpleNamespace(attn=attn)
        self.aggregator = types.SimpleNamespace(inter_frame_blocks=[block])

    def forward(self, images):
        return self._preds


def test_omega_extract_features_emits_world_points_and_conf():
    """Omega's verify forward yields poses + unprojected world_points + conf."""
    from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator

    h = w = 8
    # Canned full-forward predictions: pose encoding + unit depth + constant conf
    preds = {
        "pose_enc": torch.zeros(1, 2, 9),
        "depth": torch.ones(1, 2, h, w, 1),
        "depth_conf": torch.full((1, 2, h, w), 0.5),
    }
    creator = object.__new__(VGGTOmegaCreator)
    creator.model = _FakeOmegaModel(preds)

    # Frame 0 at identity, frame 1 translated; identity K → cam coords = (x, y, 1)
    ext = torch.eye(4)[:3].repeat(2, 1, 1)
    ext[1, 0, 3] = 1.0
    intr = torch.eye(3).repeat(2, 1, 1)
    with patch(
        "collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
        return_value=(ext.unsqueeze(0), intr.unsqueeze(0)),
    ):
        features = creator.extract_intermediate_features(torch.zeros(2, 3, h, w))

    # Poses: (2, 4, 4) float32 w2c, frame 0 identity, frame 1 carries the translation
    assert features["poses"].shape == (2, 4, 4)
    assert features["poses"].dtype == np.float32
    assert np.allclose(features["poses"][0], np.eye(4), atol=1e-6)
    assert np.isclose(features["poses"][1][0, 3], 1.0)
    # Geometry: (2, H, W, 3) float32 world points from depth unprojection
    assert features["world_points"].shape == (2, h, w, 3)
    assert features["world_points"].dtype == np.float32
    # depth=1, K=I, identity extrinsic → world point at (row y, col x) = (x, y, 1)
    assert np.allclose(features["world_points"][0, 0, 0], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(features["world_points"][0, 1, 2], [2.0, 1.0, 1.0], atol=1e-6)
    # Confidence: (2, H, W) float32 passthrough of depth_conf
    assert features["conf"].shape == (2, h, w)
    assert features["conf"].dtype == np.float32
    assert np.allclose(features["conf"], 0.5)


########################################################################
########## VGGT-X: verify path emits world_points + conf ##############
########################################################################


class _FakeVGGTXModel(torch.nn.Module):
    """Minimal stand-in: aggregator global block tree + canned predictions."""

    def __init__(self, preds):
        super().__init__()
        self._preds = preds
        self._param = torch.nn.Parameter(torch.zeros(1))
        attn = types.SimpleNamespace(num_heads=2, qkv=_FakeQKV())
        block = types.SimpleNamespace(attn=attn)
        self.aggregator = types.SimpleNamespace(global_blocks=[block])

    def forward(self, images):
        return self._preds


def test_vggtx_extract_features_emits_world_points_and_conf():
    """VGGT-X's verify forward yields poses + unprojected world_points + conf."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    h = w = 8
    # Canned full-forward predictions: pose encoding + unit depth + constant conf
    preds = {
        "pose_enc": torch.zeros(1, 2, 9),
        "depth": torch.ones(1, 2, h, w, 1),
        "depth_conf": torch.full((1, 2, h, w), 0.5),
    }
    creator = object.__new__(VGGTXCreator)
    creator.model = _FakeVGGTXModel(preds)

    # Frame 0 at identity, frame 1 translated (w2c); identity K → cam coords = (x, y, 1)
    ext = torch.eye(4)[:3].repeat(2, 1, 1)
    ext[1, 0, 3] = 1.0
    intr = torch.eye(3).repeat(2, 1, 1)
    with patch(
        "collab_splats.pointcloud.feedforward.vggtx.pose_encoding_to_extri_intri",
        return_value=(ext.unsqueeze(0), intr.unsqueeze(0)),
    ):
        features = creator.extract_intermediate_features(torch.zeros(2, 3, h, w))

    # Poses: (2, 4, 4) float32 w2c, frame 0 identity, frame 1 carries the translation
    assert features["poses"].shape == (2, 4, 4)
    assert features["poses"].dtype == np.float32
    assert np.allclose(features["poses"][0], np.eye(4), atol=1e-6)
    assert np.isclose(features["poses"][1][0, 3], 1.0)
    # Geometry: (2, H, W, 3) float32 world points from depth unprojection
    assert features["world_points"].shape == (2, h, w, 3)
    assert features["world_points"].dtype == np.float32
    # depth=1, K=I, identity extrinsic → world point at (row y, col x) = (x, y, 1)
    assert np.allclose(features["world_points"][0, 0, 0], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(features["world_points"][0, 1, 2], [2.0, 1.0, 1.0], atol=1e-6)
    # Frame 1: w2c t=[1,0,0] → c2w t=[-1,0,0]; cam point (x, y, 1) lands at (x-1, y, 1).
    # Catches c2w/w2c convention slips in the unprojection call.
    assert np.allclose(features["world_points"][1, 0, 0], [-1.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(features["world_points"][1, 1, 2], [1.0, 1.0, 1.0], atol=1e-6)
    # Confidence: (2, H, W) float32 passthrough of depth_conf
    assert features["conf"].shape == (2, h, w)
    assert features["conf"].dtype == np.float32
    assert np.allclose(features["conf"], 0.5)


########################################################################
########## Wrapper LC loop: lc_data applied to the loop Submap #########
########################################################################


class _StubCreator(BaseFeedforwardCreator):
    """Minimal creator for driving the LoopClosure LC loop in-process."""

    def _load_model(self, device):
        m = MagicMock()
        m.parameters.return_value = iter([torch.zeros(1)])
        return m

    def _preprocess(self, image_dir):
        return torch.zeros(40, 3, 16, 16), [None] * 40, np.zeros((40, 6))

    def _forward(self, model, views, **kwargs):
        k = views.shape[0]
        h = w = views.shape[-1]
        # Varied positive depth_conf so window submaps carry a non-empty dense cloud
        # (else _assemble_result fails fast on an empty point cloud).
        rows = np.arange(h, dtype=np.float32)[:, None]
        cols = np.arange(w, dtype=np.float32)[None, :]
        depth_conf = np.tile(50.0 + rows + cols, (k, 1, 1)).astype(np.float32)
        return {
            "extrinsic": np.tile(np.eye(4)[:3], (k, 1, 1)).astype(np.float32),
            "intrinsic": np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
            "depth": np.ones((k, h, w, 1), dtype=np.float32),
            "depth_conf": depth_conf,
        }

    def _verify_loop_candidate(self, frame1, frame2, verify_match_ratio=0.85):
        return self._verify_return

    def _postprocess(self, raw_outputs, **kwargs):
        pass

    def extract_intermediate_features(self, frames, layer_index=-1, **kwargs):
        return {}

    def _reproject(self, raw_outputs, extrinsics_3x4, intrinsics):
        return np.zeros((0, 3)), np.zeros((0, 3))

    def build_colmap(self, output_dir):
        pass


def _run_lc_with_verify_return(verify_return):
    """Drive LoopClosure.run_inference with one loop candidate; return the base creator."""
    base = _StubCreator(camera_model="PINHOLE")
    base._verify_return = verify_return
    # submap_size=10 → 4 submaps over 40 frames, so min_submap_gap=1 leaves a
    # non-empty past for later submaps and the fake loop candidate fires.
    creator = LoopClosure(base, config=LoopClosureConfig(submap_size=10, submap_overlap=2))
    creator.load_model()
    base.views = torch.zeros(40, 3, 16, 16)
    base.image_paths = [None] * 40

    def _fake_find_loops(submap, past, *args, **kwargs):
        # Exactly one candidate across the whole run (first submap with a past)
        if len(past) != 1:
            return []
        return [
            LoopMatch(
                similarity_score=0.1,
                query_submap_id=submap.submap_id,
                detected_submap_id=0,
                query_frame_idx=0,
                detected_frame_idx=0,
            )
        ]

    # Output is now assembled from the GraphMap (no batch PGO / merge helpers to
    # patch); n_loops_applied is still set in _run_lc_loop.
    with (
        patch("collab_splats.localization.BaseRetrievalExtractor.get") as mock_get,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=_fake_find_loops),
        patch("collab_splats.geometry.loop_closure.wrapper.translation_jump_check", return_value=(True, 0.0)),
    ):
        mock_get.return_value = MagicMock(return_value=lambda frames: torch.zeros(frames.shape[0], 128))
        creator.run_inference()
    return base


def test_accepted_loop_with_geometry_is_applied():
    """Accepted lc_data with geometry → the loop is applied (n_loops_applied == 1)."""
    h = w = 8
    lc_data = {
        "poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "world_points": np.random.rand(2, h, w, 3).astype(np.float32),
        "conf": np.ones((2, h, w), dtype=np.float32),
    }
    base = _run_lc_with_verify_return((True, lc_data))

    assert base.n_loops_applied == 1


def test_accepted_loop_without_geometry_is_applied():
    """Accepted lc_data without geometry still applies the loop (None-geometry branch)."""
    lc_data = {
        "poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "world_points": None,
        "conf": None,
    }
    base = _run_lc_with_verify_return((True, lc_data))

    assert base.n_loops_applied == 1


def test_accept_without_lc_data_hits_defensive_guard():
    """(True, None) from verify is a contract violation → candidate rejected, no loop applied."""
    base = _run_lc_with_verify_return((True, None))

    assert base.n_loops_applied == 0
