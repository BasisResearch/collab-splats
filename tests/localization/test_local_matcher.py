"""Unit tests for LocalMatcher — vismatch mocked throughout; no model downloads."""

import sys

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from collab_splats.localization.extractors import (
    BaseLocalExtractor,
    LocalFeatures,
    LocalMatcher,
    MatchResult,
)


def _fake_vismatch_matcher(n_kpts=8, d=64, stable_indices=True):
    """Mock of a vismatch BaseMatcher: forward(img0, img1) -> result dict."""
    rng = np.random.default_rng(0)
    all_kpts0 = rng.uniform(0, 100, (n_kpts, 2)).astype(np.float32)
    all_kpts1 = rng.uniform(0, 100, (n_kpts, 2)).astype(np.float32)
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
def test_not_registered_under_legacy_names(mock_get):
    # LocalMatcher must not shadow legacy registry entries; resolve-by-name comes later.
    assert LocalMatcher not in BaseLocalExtractor._registry.values()


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
def test_probe_sets_stability_flag(mock_get):
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=True)
    lm = LocalMatcher("disk-lightglue", device="cpu")  # probe=True default
    assert lm.has_stable_indices is True
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=False)
    lm2 = LocalMatcher("roma", device="cpu")
    assert lm2.has_stable_indices is False
