"""lc_data verify contract: MapAnything pose derivation + wrapper loop application."""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator
from collab_splats.pointcloud.loop_closure import LoopClosureConfig
from collab_splats.pointcloud.loop_closure.closure import LoopMatch
from collab_splats.pointcloud.wrappers import LoopClosure
from collab_splats.utils.geometry import invert_poses

########################################################################
########## MapAnything: extract_intermediate_features returns poses ####
########################################################################


class _FakeQKV(torch.nn.Module):
    """Real nn.Module so register_forward_hook works; never actually called."""

    def forward(self, x):
        return x


class _FakeMapAnythingModel(torch.nn.Module):
    """Minimal stand-in: info_sharing block tree + forward returning canned preds."""

    def __init__(self, preds):
        super().__init__()
        self._preds = preds
        self._param = torch.nn.Parameter(torch.zeros(1))
        attn = types.SimpleNamespace(num_heads=2, qkv=_FakeQKV())
        block = types.SimpleNamespace(attn=attn)
        self.info_sharing = types.SimpleNamespace(self_attention_blocks=[block])

    def forward(self, views, memory_efficient_inference=False, minibatch_size=1):
        return self._preds


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
        return {
            "extrinsic": np.tile(np.eye(4)[:3], (k, 1, 1)).astype(np.float32),
            "intrinsic": np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
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

    with (
        patch("collab_splats.localization.BaseRetrievalExtractor.get") as mock_get,
        patch("collab_splats.pointcloud.wrappers.find_loop_closures", side_effect=_fake_find_loops),
        patch("collab_splats.pointcloud.wrappers.translation_jump_check", return_value=(True, 0.0)),
        patch("collab_splats.pointcloud.wrappers.run_pose_graph_optimization") as mock_pg,
        patch("collab_splats.pointcloud.wrappers.merge_submap_outputs") as mock_merge,
    ):
        mock_get.return_value = MagicMock(return_value=lambda frames: torch.zeros(frames.shape[0], 128))
        mock_pg.return_value = np.tile(np.eye(4, dtype=np.float32), (40, 1, 1))
        mock_merge.return_value = {}
        creator.run_inference()
    return base


def test_lc_submap_carries_reshaped_world_points():
    """Accepted lc_data with geometry → LC Submap gets (K, P, 3)/(K, P) points/conf."""
    h = w = 8
    lc_data = {
        "poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "world_points": np.random.rand(2, h, w, 3).astype(np.float32),
        "conf": np.ones((2, h, w), dtype=np.float32),
    }
    base = _run_lc_with_verify_return((True, lc_data))

    assert len(base._lc_loop_submaps) == 1
    lc_submap = base._lc_loop_submaps[0]
    assert lc_submap.world_points.shape == (2, h * w, 3)
    assert lc_submap.world_points_conf.shape == (2, h * w)
    # Reshape preserves values: row-major (H, W) flattening
    assert np.allclose(lc_submap.world_points[0], lc_data["world_points"][0].reshape(-1, 3))
    accepted = [m for m in base._lc_all_matches if m.accepted]
    assert len(accepted) == 1 and accepted[0].reject_reason is None


def test_lc_submap_geometry_none_when_backend_has_none():
    """Accepted lc_data without geometry → LC Submap world_points/conf stay None."""
    lc_data = {
        "poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "world_points": None,
        "conf": None,
    }
    base = _run_lc_with_verify_return((True, lc_data))

    assert len(base._lc_loop_submaps) == 1
    assert base._lc_loop_submaps[0].world_points is None
    assert base._lc_loop_submaps[0].world_points_conf is None


def test_accept_without_lc_data_hits_defensive_guard():
    """(True, None) from verify is a contract violation → rejected as no_joint_poses."""
    base = _run_lc_with_verify_return((True, None))

    assert base._lc_loop_submaps == []
    reasons = [m.reject_reason for m in base._lc_all_matches]
    assert "no_joint_poses" in reasons
