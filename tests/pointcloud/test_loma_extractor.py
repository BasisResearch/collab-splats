import numpy as np
import pytest
import torch

from collab_splats.pointcloud.localization import (
    BaseLocalExtractor,
    LocalFeatures,
    LomaExtractor,
    LomaGExtractor,
)


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
def test_loma_match_returns_index_pairs(loma_extractor):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = loma_extractor.extract(img)
    matches = loma_extractor.match(feats, feats, image_hw=(480, 640))
    assert matches.ndim == 2 and matches.shape[1] == 2
    assert matches.dtype == torch.long
    # Self-match: most matches should be the identity pair
    if len(matches) > 0:
        diag = (matches[:, 0] == matches[:, 1]).sum()
        assert diag > 0


def test_loma_exported_from_pointcloud_package():
    from collab_splats.pointcloud import LomaExtractor as le
    from collab_splats.pointcloud import LomaGExtractor as lge
    assert le is LomaExtractor and lge is LomaGExtractor
