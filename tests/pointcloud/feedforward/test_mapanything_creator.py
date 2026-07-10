"""MapAnything creator: _lc_collate_outputs carries depth for anchor-scale estimation."""

import logging
import types
from unittest.mock import patch

import numpy as np
import torch

from collab_splats.pointcloud.feedforward.base import _raw_to_world_points

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
