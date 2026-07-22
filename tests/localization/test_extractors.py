"""Tests for local extractor match() pixel-pair contract (MatchResult)."""

import numpy as np
import pytest

from collab_splats.localization.extractors import (
    BaseLocalExtractor,
    DiskExtractor,
    MatchResult,
    XFeatExtractor,
    _empty_match,
)


@pytest.fixture(scope="module")
def image_pair():
    """Two 128x128 random-dot images; second is the first shifted 4 px right."""
    rng = np.random.default_rng(0)
    img0 = (rng.random((128, 128, 1)) > 0.95).astype(np.uint8) * 255
    img0 = np.repeat(img0, 3, axis=2)
    img1 = np.roll(img0, 4, axis=1)
    return img0, img1


def test_match_returns_pixel_pairs_disk(image_pair):
    """match() returns MatchResult whose pixels correspond to detected keypoints."""
    img0, img1 = image_pair
    ex = DiskExtractor(top_k=256)
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert isinstance(m, MatchResult)
    assert m.query_px.shape == m.ref_px.shape and m.query_px.shape[1] == 2
    assert len(m) > 0
    # Matched pixels must be drawn from the detected keypoints on both sides
    kq, kr = f0.keypoints.numpy(), f1.keypoints.numpy()
    assert all(((kq == q).all(axis=1)).any() for q in m.query_px[:5])
    assert all(((kr == r).all(axis=1)).any() for r in m.ref_px[:5])


def test_match_returns_pixel_pairs_xfeat(image_pair):
    """XFeat match() returns MatchResult with pixels drawn from detected keypoints."""
    img0, img1 = image_pair
    ex = XFeatExtractor(top_k=256)
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert isinstance(m, MatchResult)
    assert m.query_px.shape == m.ref_px.shape and m.query_px.shape[1] == 2
    assert len(m) > 0
    # Matched pixels must be drawn from the detected keypoints on both sides
    kq, kr = f0.keypoints.numpy(), f1.keypoints.numpy()
    assert all(((kq == q).all(axis=1)).any() for q in m.query_px[:5])
    assert all(((kr == r).all(axis=1)).any() for r in m.ref_px[:5])


def test_xfeat_star_matches_shifted_image():
    """XFeat* dense extract + refined pairwise match recovers a known shift."""
    rng = np.random.default_rng(1)
    img0 = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    img1 = np.roll(img0, 6, axis=1)
    ex = BaseLocalExtractor.get("xfeat-star")(top_k=2048)
    f0, f1 = ex.extract(img0), ex.extract(img1)
    assert f0.scales is not None  # dense path caches scales
    m = ex.match(f0, f1, img0.shape[:2])
    assert isinstance(m, MatchResult) and len(m) > 50
    dx = m.ref_px[:, 0] - m.query_px[:, 0]
    assert abs(np.median(dx) - 6) < 1.5  # recovers the shift


def test_empty_match_helper():
    """_empty_match returns a zero-length float32 MatchResult."""
    m = _empty_match()
    assert isinstance(m, MatchResult)
    assert len(m) == 0
    assert m.query_px.shape == (0, 2) and m.ref_px.shape == (0, 2)
    assert m.query_px.dtype == np.float32
