"""
Localization test fixtures: a CPU DINO-SALAD stand-in and a duck-typed matcher.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from collab_splats.localization.extractors import LocalFeatures, MatchResult


class FakeSalad(torch.nn.Module):
    """
    Deterministic global descriptor: per-channel mean and std, L2-normalized.
    """

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = images.float().flatten(2)
        desc = torch.cat([flat.mean(-1), flat.std(-1)], dim=1)
        return torch.nn.functional.normalize(desc, dim=-1).cpu()


class StubMatcher:
    """
    LocalMatcher stand-in: extract replays a callback, match pairs equal descriptor rows.

    - keypoints(image) -> (K, 2) px; descriptors are row ids, so equal ids match
    """

    model_name = "stub"
    max_num_keypoints = 2048

    def __init__(self, keypoints) -> None:
        self._keypoints = keypoints
        self.n_extract = 0

    def _one(self, image: np.ndarray) -> LocalFeatures:
        self.n_extract += 1
        kp = torch.as_tensor(self._keypoints(image), dtype=torch.float32)
        ids = torch.arange(len(kp), dtype=torch.float32)[:, None]
        return LocalFeatures(
            keypoints=kp, descriptors=ids, image_size=(image.shape[1], image.shape[0])
        )

    def extract(self, images):
        if isinstance(images, list):
            return [self._one(im) for im in images]
        return self._one(images)

    def to_device(self, features: LocalFeatures) -> LocalFeatures:
        return features

    def match(self, q: LocalFeatures, db: LocalFeatures) -> MatchResult:
        _, iq, idb = np.intersect1d(
            q.descriptors[:, 0].numpy(),
            db.descriptors[:, 0].numpy(),
            return_indices=True,
        )
        return MatchResult(
            query_px=q.keypoints.numpy()[iq],
            ref_px=db.keypoints.numpy()[idb],
            idx_q=iq.astype(np.int64),
            idx_db=idb.astype(np.int64),
        )


class ReplayMatcher(StubMatcher):
    """
    StubMatcher whose extract hands out the given LocalFeatures in order, one per image.
    """

    def __init__(self, features: list[LocalFeatures]) -> None:
        super().__init__(lambda image: np.zeros((0, 2)))
        self._queue = list(features)

    def _one(self, image: np.ndarray) -> LocalFeatures:
        self.n_extract += 1
        return self._queue.pop(0)


@pytest.fixture(autouse=True)
def fake_salad(monkeypatch):
    """
    Every localizer in these tests embeds with FakeSalad, never the real checkpoint.
    """
    registry = SimpleNamespace(get=lambda name: FakeSalad)
    monkeypatch.setattr(
        "collab_splats.localization.localizer.BaseRetrievalExtractor", registry
    )
    return FakeSalad


@pytest.fixture
def stub_matcher():
    """
    The StubMatcher class, for tests that need a duck-typed matcher.
    """
    return StubMatcher


@pytest.fixture
def replay_matcher():
    """
    The ReplayMatcher class, for tests that feed exact per-frame features.
    """
    return ReplayMatcher
