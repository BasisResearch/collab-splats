"""
The skywater sky backend and its mask cache.

- polarity: the model emits HIGH values for sky, inverted from skyseg's stored 255-is-not-sky
- the output is four-class LOGITS, so the sky channel needs a softmax before it is a probability
- the per-image min-max rescale skyseg's upstream applies is deliberately dropped
- the cache is keyed only by frame_idx, so a warm call must reproduce a cold one exactly
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from collab_splats.semantics.segmentation import sky

######## Fakes


class _FakeSession:
    """
    Stand-in for onnxruntime.InferenceSession returning one canned output tensor.
    """

    def __init__(self, output):
        self._output = output.astype(np.float32)
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, output_names, feed):
        self.calls += 1
        return [self._output]


class _ConstantBackend:
    """
    A BaseSegmentation stand-in whose every mask is one constant value.
    """

    def __init__(self, value):
        self._value = value

    def segment(self, image):
        rgb = np.asarray(sky.open_image(image).convert("RGB"))
        mask = np.full(rgb.shape[:2], self._value, bool)
        return torch.from_numpy(mask), {"raw": mask.astype(np.float32)}


def _logits_for(prob: np.ndarray) -> np.ndarray:
    """
    (4, h, w) logits whose softmax puts exactly `prob` on the sky class.
    """
    rest = np.log(np.clip((1.0 - prob) / 3.0, 1e-12, None))
    return np.stack([rest, np.log(np.clip(prob, 1e-12, None)), rest, rest]).astype(np.float32)


def _wire(monkeypatch, output, threshold=0.5):
    """
    A SkyWaterSegmentation wired to a fake session over `output`; no download, no ONNX.
    """
    session = _FakeSession(output)
    monkeypatch.setattr(sky, "load_hf_weights", lambda repo_id, filename: "model.onnx")
    monkeypatch.setattr(sky.ort, "InferenceSession", lambda path, providers=None: session)
    return sky.SkyWaterSegmentation(threshold=threshold), session


def _backend(monkeypatch, prob, threshold=0.5):
    """
    A SkyWaterSegmentation over a probability map, encoded back into four-class logits.
    """
    return _wire(monkeypatch, _logits_for(prob)[None], threshold=threshold)


######## Polarity and threshold


def test_high_probability_means_sky(monkeypatch):
    # Top half of the model grid is sky; upstream's comment claims the opposite of this
    prob = np.zeros((384, 384), np.float32)
    prob[:192] = 0.9
    backend, _ = _backend(monkeypatch, prob)

    mask, meta = backend.segment(np.zeros((64, 96, 3), np.uint8))

    assert mask.shape == (64, 96)
    assert mask.dtype == torch.bool
    assert mask[:30].all()
    assert not mask[34:].any()
    assert meta["raw"].shape == (64, 96)
    assert meta["raw"].dtype == np.float32


def test_threshold_is_a_probability(monkeypatch):
    # 0.125 is upstream's 32/255; a uniform 0.2 map is sky under it and not under 0.5
    prob = np.full((384, 384), 0.2, np.float32)

    loose, _ = _backend(monkeypatch, prob, threshold=0.125)
    assert loose.segment(np.zeros((16, 16, 3), np.uint8))[0].all()

    strict, _ = _backend(monkeypatch, prob, threshold=0.5)
    assert not strict.segment(np.zeros((16, 16, 3), np.uint8))[0].any()


######## The dropped rescale


def test_a_dim_map_yields_no_sky(monkeypatch):
    # Ramp 0.0 -> 0.3. Upstream's per-image min-max rescale would stretch the top row to
    # 1.0 and manufacture sky from nothing; thresholding the raw probability must not.
    prob = np.linspace(0.0, 0.3, 384, dtype=np.float32)[:, None].repeat(384, axis=1)
    backend, _ = _backend(monkeypatch, prob)

    mask, meta = backend.segment(np.zeros((32, 32, 3), np.uint8))

    assert not mask.any()
    assert meta["raw"].max() <= 0.3 + 1e-6


######## Logits, not probabilities


def test_softmax_picks_the_sky_class(monkeypatch):
    # Top half sky, bottom half background; both halves are decided by which CLASS wins,
    # not by the sky channel's own magnitude, which is identical in the two halves
    logits = np.zeros((4, 384, 384), np.float32)
    logits[1] = 4.0
    logits[0, 192:] = 8.0

    backend, _ = _wire(monkeypatch, logits[None])
    mask, meta = backend.segment(np.zeros((64, 96, 3), np.uint8))

    assert mask[:30].all()
    assert not mask[34:].any()
    assert meta["raw"].shape == (64, 96)


def test_the_output_is_logits_not_probabilities(monkeypatch):
    # Background wins 5.0 to 2.0, so softmax puts sky at ~0.047 and nothing is sky. A
    # backend that thresholded the raw sky channel would see 2.0 > 0.5 and call it all sky.
    logits = np.zeros((4, 384, 384), np.float32)
    logits[0] = 5.0
    logits[1] = 2.0

    backend, _ = _wire(monkeypatch, logits[None])
    mask, meta = backend.segment(np.zeros((32, 32, 3), np.uint8))

    assert not mask.any()
    assert meta["raw"].max() < 0.1


def test_probabilities_sum_to_one_across_classes(monkeypatch):
    # A softmax normalised over the wrong axis still lands in [0, 1] and would pass the two
    # tests above; only rotating each class into the sky slot and summing catches it
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 3.0, (4, 8, 8)).astype(np.float32)
    image = np.zeros((8, 8, 3), np.uint8)

    per_class = []
    for c in range(4):
        backend, _ = _wire(monkeypatch, np.roll(logits, -c, axis=0)[None])
        per_class.append(backend.segment(image)[1]["raw"])

    assert np.allclose(np.sum(per_class, axis=0), 1.0, atol=1e-5)


######## Registration


def test_skywater_is_registered_and_exported():
    from collab_splats.semantics.segmentation import (
        BaseSegmentation,
        SkyWaterSegmentation,
    )

    assert BaseSegmentation.get("skywater") is SkyWaterSegmentation


######## Mask cache and ordering


def _scene(tmp_path, monkeypatch, prob, frame_idxs=(0, 5, 7)):
    """
    An images/ dir plus a registry whose "skywater" entry is the fake-session backend.
    """
    from collab_splats.preproc import frames as fr

    images = [np.full((16, 16, 3), idx, np.uint8) for idx in frame_idxs]
    fr.write_frames(tmp_path / "images", images, [{"frame_idx": i} for i in frame_idxs], {})

    backend, session = _backend(monkeypatch, prob)
    monkeypatch.setattr(sky.BaseSegmentation, "get", classmethod(lambda cls, name: lambda: backend))
    return tmp_path / "images", session


def test_sky_masks_reads_every_frame_in_filename_order(tmp_path, monkeypatch):
    prob = np.zeros((384, 384), np.float32)
    prob[:192] = 0.9
    images_dir, session = _scene(tmp_path, monkeypatch, prob)

    masks = sky.sky_masks(images_dir)

    assert masks.shape == (3, 16, 16)
    assert masks.dtype == bool
    assert masks[:, :6].all() and not masks[:, 10:].any()
    assert session.calls == 3


def test_sky_masks_returns_masks_in_the_order_idxs_names_them(tmp_path, monkeypatch):
    # Per-frame probability so a permutation is detectable: frame_idx 7 is the only sky one
    calls = {"n": 0}

    def _logits_for_call():
        calls["n"] += 1
        return _logits_for(np.full((384, 384), 0.9 if calls["n"] == 3 else 0.0, np.float32))

    images_dir, session = _scene(tmp_path, monkeypatch, np.zeros((384, 384), np.float32))
    monkeypatch.setattr(session, "run", lambda names, feed: [_logits_for_call()[None]])

    # Filename order is (0, 5, 7), so the third segmented frame is frame_idx 7
    sky.sky_masks(images_dir)
    permuted = sky.sky_masks(images_dir, idxs=[7, 0, 5])

    assert permuted[0].all()
    assert not permuted[1].any() and not permuted[2].any()


def test_sky_masks_caches_to_disk_with_255_meaning_sky(tmp_path, monkeypatch):
    import cv2

    prob = np.zeros((384, 384), np.float32)
    prob[:192] = 0.9
    images_dir, session = _scene(tmp_path, monkeypatch, prob)

    sky.sky_masks(images_dir)
    cached = cv2.imread(str(tmp_path / "sky" / "frame_000005.png"), cv2.IMREAD_GRAYSCALE)

    assert cached.shape == (16, 16)
    assert cached[0, 0] == 255 and cached[15, 15] == 0


def test_sky_masks_second_call_runs_the_model_zero_times(tmp_path, monkeypatch):
    # Each frame gets a DIFFERENT sky band, so the warm call is compared and not merely counted
    # - a uniform fixture makes every row identical, and then a cache that returned the wrong
    #   frame, or the rows in the wrong order, would still satisfy the assertions below
    calls = {"n": 0}

    def _logits_for_call():
        calls["n"] += 1
        prob = np.zeros((384, 384), np.float32)
        prob[: 96 * calls["n"]] = 0.9
        return _logits_for(prob)

    images_dir, session = _scene(tmp_path, monkeypatch, np.zeros((384, 384), np.float32))
    monkeypatch.setattr(session, "run", lambda names, feed: [_logits_for_call()[None]])

    first = sky.sky_masks(images_dir)
    assert calls["n"] == 3

    again = sky.sky_masks(images_dir)
    assert calls["n"] == 3
    assert np.array_equal(first, again)

    # The three frames must be mutually distinguishable, or the equality above proves nothing
    assert len({int(frame.sum()) for frame in first}) == 3


def test_sky_masks_rejects_an_unknown_frame_idx(tmp_path, monkeypatch):
    images_dir, _ = _scene(tmp_path, monkeypatch, np.zeros((384, 384), np.float32))

    with pytest.raises(KeyError, match="99"):
        sky.sky_masks(images_dir, idxs=[0, 99])


def test_sky_masks_asks_the_registry_for_skywater(tmp_path, monkeypatch):
    asked = []
    images_dir, _ = _scene(tmp_path, monkeypatch, np.zeros((384, 384), np.float32))
    monkeypatch.setattr(
        sky.BaseSegmentation,
        "get",
        classmethod(lambda cls, name: asked.append(name) or (lambda: _ConstantBackend(False))),
    )

    sky.sky_masks(images_dir)

    assert asked == ["skywater"]


def test_sky_masks_segments_without_stacking_every_uncached_frame(tmp_path, monkeypatch):
    # read_frames returns one np.stack of every miss — ~10 GB for an 853-frame scene, on top
    # of whatever fusion buffers the mesh stage is already holding. Segmenting reads one at a
    # time, so this whole run must never reach read_frames.
    images_dir, _ = _scene(tmp_path, monkeypatch, np.zeros((384, 384), np.float32))
    monkeypatch.setattr(sky.BaseSegmentation, "get", classmethod(lambda cls, name: lambda: _ConstantBackend(True)))
    monkeypatch.setattr(sky.frames, "read_frames", lambda *a, **k: pytest.fail("sky_masks called read_frames"))

    assert sky.sky_masks(images_dir).all()
