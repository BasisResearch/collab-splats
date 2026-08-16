import numpy as np
import pytest

from collab_splats.localization import (
    BaseLocalExtractor,
    LocalFeatures,
    LomaExtractor,
    LomaGExtractor,
)
from collab_splats.localization.extractors import MatchResult


@pytest.fixture(scope="module")
def image_pair():
    """Two 128x128 random-dot images; second is the first shifted 4 px right."""
    rng = np.random.default_rng(0)
    img0 = (rng.random((128, 128, 1)) > 0.95).astype(np.uint8) * 255
    img0 = np.repeat(img0, 3, axis=2)
    img1 = np.roll(img0, 4, axis=1)
    return img0, img1


def test_loma_registry():
    assert BaseLocalExtractor.get("loma") is LomaExtractor
    assert BaseLocalExtractor.get("loma-g") is LomaGExtractor


def test_loma_g_is_distinct_class():
    # from_feedforward reverse-looks-up class -> registry name for zarr cache
    # keying; each variant must be its own class so the lookup is unambiguous.
    assert LomaGExtractor is not LomaExtractor
    assert issubclass(LomaGExtractor, LomaExtractor)


@pytest.fixture(scope="module")
def loma_extractor():
    """Shared instance — construction loads ~723 MB of weights (already cached)."""
    return LomaExtractor(top_k=512)


@pytest.mark.slow
def test_loma_extract_shapes(loma_extractor):
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = loma_extractor.extract(image)
    assert isinstance(feats, LocalFeatures)
    assert feats.keypoints.ndim == 2 and feats.keypoints.shape[1] == 2
    assert feats.descriptors.ndim == 2 and feats.descriptors.shape[1] == 256
    assert len(feats.keypoints) == len(feats.descriptors) == len(feats.scores)
    assert len(feats.keypoints) > 0
    # Keypoints are in ORIGINAL pixel coords (not 784x784 inference coords)
    assert feats.keypoints[:, 0].max() <= 640
    assert feats.keypoints[:, 1].max() <= 480


@pytest.mark.slow
def test_loma_match_returns_pixel_pairs(loma_extractor):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = loma_extractor.extract(img)
    matches = loma_extractor.match(feats, feats, image_hw=(480, 640))
    assert isinstance(matches, MatchResult)
    assert matches.query_px.shape == matches.ref_px.shape
    assert matches.query_px.shape[1] == 2
    assert matches.query_px.dtype == np.float32
    # Self-match: some matches should land on the identical pixel
    if len(matches) > 0:
        diag = (matches.query_px == matches.ref_px).all(axis=1).sum()
        assert diag > 0


@pytest.mark.slow
def test_loma_match_returns_indices(image_pair):
    """LoMa filter_matches indices survive into MatchResult."""
    img0, img1 = image_pair
    ex = LomaExtractor()
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert m.idx_q is not None and m.idx_db is not None
    np.testing.assert_allclose(m.query_px, f0.keypoints.numpy()[m.idx_q])
    np.testing.assert_allclose(m.ref_px, f1.keypoints.numpy()[m.idx_db])


def test_loma_exported_from_localization_package():
    from collab_splats.localization import LomaExtractor as le
    from collab_splats.localization import LomaGExtractor as lge

    assert le is LomaExtractor and lge is LomaGExtractor
