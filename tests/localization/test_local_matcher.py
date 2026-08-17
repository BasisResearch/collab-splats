"""Unit tests for LocalMatcher — vismatch mocked throughout; no model downloads."""

import sys

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from collab_splats.localization.extractors import LocalFeatures, LocalMatcher, MatchResult
from collab_splats.localization.localizer import CameraLocalizer


def _fake_vismatch_matcher(n_kpts=8, d=64, stable_indices=True):
    """Mock of a vismatch BaseMatcher: forward(img0, img1) -> result dict."""
    rng = np.random.default_rng(0)
    all_kpts0 = rng.uniform(0, 99, (n_kpts, 2)).astype(np.float32)
    all_kpts1 = rng.uniform(0, 99, (n_kpts, 2)).astype(np.float32)
    if stable_indices:
        matched0, matched1 = all_kpts0[:4], all_kpts1[:4]  # exact rows
    else:
        matched0, matched1 = all_kpts0[:4] + 0.3, all_kpts1[:4] + 0.3  # refined coords
    result = {
        "num_inliers": 4, "H": np.eye(3),
        "all_kpts0": all_kpts0, "all_kpts1": all_kpts1,
        "all_desc0": rng.standard_normal((n_kpts, d)).astype(np.float32),
        "all_desc1": rng.standard_normal((n_kpts, d)).astype(np.float32),
        "matched_kpts0": matched0, "matched_kpts1": matched1,
        "inlier_kpts0": matched0[:3], "inlier_kpts1": matched1[:3],
        "matched_confidences": np.ones(4, dtype=np.float32),
    }
    m = MagicMock()
    m.side_effect = lambda i0, i1: dict(result)  # __call__(img0, img1)
    m.extract.return_value = {"all_kpts0": all_kpts0, "all_desc0": result["all_desc0"]}
    return m


@patch("vismatch.get_matcher")
def test_extract_returns_local_features(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    feats = lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert isinstance(feats, LocalFeatures)
    assert feats.keypoints.shape == (8, 2) and feats.keypoints.dtype == torch.float32
    assert feats.descriptors.shape == (8, 64)
    mock_get.assert_called_once_with("disk-lightglue", device="cpu")


@patch("vismatch.get_matcher")
def test_extract_handles_tensor_outputs(mock_get):
    # Real matchers may return torch tensors (check_types allows tensor or ndarray).
    m = _fake_vismatch_matcher()
    prev = m.extract.return_value
    m.extract.return_value = {
        "all_kpts0": torch.from_numpy(prev["all_kpts0"]),
        "all_desc0": torch.from_numpy(prev["all_desc0"]),
    }
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    feats = lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert feats.keypoints.shape == (8, 2) and feats.keypoints.dtype == torch.float32
    assert feats.descriptors.shape == (8, 64)


@patch("vismatch.get_matcher")
def test_extract_asserts_pixel_frame(mock_get):
    # Keypoints outside the input image bounds = coordinate-frame violation (92f2e4a class)
    m = _fake_vismatch_matcher()
    bad = dict(m.extract.return_value)
    bad["all_kpts0"] = np.array([[500.0, 500.0]], dtype=np.float32)
    bad["all_desc0"] = np.zeros((1, 64), dtype=np.float32)
    m.extract.return_value = bad
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    with pytest.raises(ValueError, match="pixel frame"):
        lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))


@patch("vismatch.get_matcher")
def test_extract_passes_chw_unit_range_tensor(mock_get):
    # vismatch input contract: (3, H, W) float in [0, 1].
    m = _fake_vismatch_matcher()
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    lm.extract(np.full((100, 100, 3), 255, dtype=np.uint8))
    (img,), _ = m.extract.call_args
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 100, 100)
    assert float(img.max()) <= 1.0 and float(img.min()) >= 0.0


def test_blocked_models_raise_without_importing_vismatch():
    # Blocklist check happens before the vismatch import — a None sys.modules entry
    # would make any import attempt raise ImportError, so ValueError proves the order.
    with patch.dict(sys.modules, {"vismatch": None}):
        with pytest.raises(ValueError, match="dependency"):
            LocalMatcher("ufm", device="cpu", probe=False)
        with pytest.raises(ValueError, match="license"):
            LocalMatcher("superglue", device="cpu", probe=False)


@patch("vismatch.get_matcher")
def test_descriptor_level_match_unsupported(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    feats = lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    with pytest.raises(NotImplementedError, match="match_images"):
        lm.match(feats, feats, image_hw=(100, 100))


@patch("vismatch.get_matcher")
def test_match_images_pre_ransac_with_indices(mock_get):
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=True)
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    lm.has_stable_indices = True
    q = np.zeros((100, 100, 3), dtype=np.uint8)
    m = lm.match_images(q, q)
    assert isinstance(m, MatchResult)
    assert len(m) == 4  # pre-RANSAC matched_kpts, NOT the 3 homography inliers
    assert m.idx_q is not None and m.idx_db is not None
    # idx must point at the exact rows of the extract()-visible keypoint table
    fake = mock_get.return_value(q, q)
    np.testing.assert_array_equal(fake["all_kpts0"][m.idx_q], m.query_px)
    np.testing.assert_array_equal(fake["all_kpts1"][m.idx_db], m.ref_px)


@patch("vismatch.get_matcher")
def test_match_images_no_indices_when_unstable(mock_get):
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=False)
    lm = LocalMatcher("roma", device="cpu", probe=False)
    lm.has_stable_indices = False
    m = lm.match_images(np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8))
    assert len(m) == 4 and m.idx_q is None and m.idx_db is None


@patch("vismatch.get_matcher")
def test_match_images_downgrades_on_recovery_failure(mock_get, caplog):
    # Probe said stable, but this pair's coords are refined off the table: both index
    # arrays must be nulled together (never one side indexed, the other not) + warning.
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=False)
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    lm.has_stable_indices = True
    q = np.zeros((100, 100, 3), dtype=np.uint8)
    with caplog.at_level("WARNING", logger="collab_splats.localization.extractors"):
        m = lm.match_images(q, q)
    assert len(m) == 4
    assert m.idx_q is None and m.idx_db is None
    assert any("index recovery failed" in r.message for r in caplog.records)


@patch("vismatch.get_matcher")
def test_probe_sets_stability_flag(mock_get):
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=True)
    lm = LocalMatcher("disk-lightglue", device="cpu")  # probe=True default
    assert lm.has_stable_indices is True
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=False)
    lm2 = LocalMatcher("roma", device="cpu")
    assert lm2.has_stable_indices is False


def _make_pairwise_localizer(n_frames=3, hw=(64, 64)):
    """CameraLocalizer with a mocked LocalMatcher and synthetic world_points."""
    lm = MagicMock(spec=LocalMatcher)
    lm.has_stable_indices = False
    q_px = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], np.float32)
    lm.match_images.return_value = MatchResult(query_px=q_px, ref_px=q_px.copy())
    lm.extract.return_value = LocalFeatures(
        keypoints=torch.zeros((4, 2)), descriptors=torch.zeros((4, 8))
    )
    # world_points: tilted plane so sampled 3D points are valid and non-degenerate
    yy, xx = np.mgrid[0 : hw[0], 0 : hw[1]].astype(np.float32)
    wp = np.stack([xx, yy, 0.01 * xx + 0.02 * yy + 1.0], axis=-1)
    # Assemble instance without running __init__'s extraction loop
    loc = CameraLocalizer.__new__(CameraLocalizer)
    loc.config = {}
    loc._extractor = lm
    loc._world_points = np.stack([wp] * n_frames)
    loc._extrinsics = np.stack([np.eye(4, dtype=np.float32)] * n_frames)
    loc._frame_features = [lm.extract.return_value] * n_frames
    loc._frame_sources = ["reconstruction"] * n_frames
    loc._image_paths = [f"frame_{i}" for i in range(n_frames)]
    loc._localized_extrinsics = []
    # Full-res grid DIFFERENT from the model grid — a rescale bug would show in pts3d
    loc._image_hw = (128, 128)
    loc._ref_images = [np.zeros((*hw, 3), np.uint8)] * n_frames
    loc._ref_global_desc = np.eye(n_frames, 8, dtype=np.float32)  # frame i ~ basis vector i
    loc._retrieval = MagicMock()
    loc._retrieval.forward.return_value = torch.from_numpy(np.eye(1, 8, dtype=np.float32))
    loc._top_k = 2
    return loc, lm


def test_pairwise_localize_matches_topk_only():
    loc, lm = _make_pairwise_localizer(n_frames=3)
    query = np.zeros((64, 64, 3), np.uint8)
    result = loc.localize(query)
    # top_k=2 -> exactly 2 pairwise calls, not 3 (all-refs would be 3)
    assert lm.match_images.call_count == 2
    # 4 matches x 2 frames of correspondences fed to PnP
    assert result.n_correspondences == 8
    assert result.ref_hw == (64, 64)  # model-res grid, not full-res _image_hw
    # No rescale: pts3d sampled at ref_px directly on the model grid, so with the
    # identity-like plane the sampled x/y equal the ref pixel coords exactly.
    q_px = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], np.float32)
    np.testing.assert_allclose(result.pts3d_matched[:4, :2], q_px, atol=1e-4)


def test_pairwise_skips_localized_frames_and_clamps_topk():
    loc, lm = _make_pairwise_localizer(n_frames=3)
    # Append a localized frame: never a match target, even with top_k > n_recon
    loc._frame_sources = ["reconstruction"] * 3 + ["localized"]
    loc._frame_features = loc._frame_features + [lm.extract.return_value]
    loc._image_paths = loc._image_paths + ["query_prev"]
    loc._top_k = 10
    loc.localize(np.zeros((64, 64, 3), np.uint8))
    # Clamped to the 3 reconstruction frames; localized frame (index 3) never matched
    assert lm.match_images.call_count == 3
