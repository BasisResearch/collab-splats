"""Tests for local extractor match() pixel-pair contract (MatchResult)."""
import numpy as np
import pytest


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
    from collab_splats.localization.extractors import DiskExtractor, MatchResult

    img0, img1 = image_pair
    ex = DiskExtractor(top_k=256)
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert isinstance(m, MatchResult)
    assert m.query_px.shape == m.ref_px.shape and m.query_px.shape[1] == 2
    if len(m.query_px):
        kq = f0.keypoints.numpy()
        assert all(((kq == q).all(axis=1)).any() for q in m.query_px[:5])


def test_match_returns_pixel_pairs_xfeat(image_pair):
    """XFeat match() returns MatchResult with pixels drawn from detected keypoints."""
    from collab_splats.localization.extractors import MatchResult, XFeatExtractor

    img0, img1 = image_pair
    ex = XFeatExtractor(top_k=256)
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert isinstance(m, MatchResult)
    assert m.query_px.shape == m.ref_px.shape and m.query_px.shape[1] == 2
    if len(m.query_px):
        kq = f0.keypoints.numpy()
        assert all(((kq == q).all(axis=1)).any() for q in m.query_px[:5])


def test_empty_match_helper():
    """_empty_match returns a zero-length float32 MatchResult."""
    from collab_splats.localization.extractors import MatchResult, _empty_match

    m = _empty_match()
    assert isinstance(m, MatchResult)
    assert len(m) == 0
    assert m.query_px.shape == (0, 2) and m.ref_px.shape == (0, 2)
    assert m.query_px.dtype == np.float32
