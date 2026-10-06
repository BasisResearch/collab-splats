"""Unit tests for LocalMatcher — vismatch mocked throughout; no model downloads."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.localization.extractors import (
    LocalFeatures,
    LocalMatcher,
)
from collab_splats.localization.localizer import CameraLocalizer, read_localization_db


def _fake_vismatch_matcher(n_kpts=8, d=64):
    """Mock of a vismatch BaseMatcher: forward(img0, img1) -> result dict."""
    rng = np.random.default_rng(0)
    all_kpts0 = rng.uniform(0, 99, (n_kpts, 2)).astype(np.float32)
    all_kpts1 = rng.uniform(0, 99, (n_kpts, 2)).astype(np.float32)
    matched0, matched1 = all_kpts0[:4], all_kpts1[:4]
    result = {
        "num_inliers": 4,
        "H": np.eye(3),
        "all_kpts0": all_kpts0,
        "all_kpts1": all_kpts1,
        "all_desc0": rng.standard_normal((n_kpts, d)).astype(np.float32),
        "all_desc1": rng.standard_normal((n_kpts, d)).astype(np.float32),
        "matched_kpts0": matched0,
        "matched_kpts1": matched1,
        "inlier_kpts0": matched0[:3],
        "inlier_kpts1": matched1[:3],
        "matched_confidences": np.ones(4, dtype=np.float32),
    }
    m = MagicMock()
    m.side_effect = lambda i0, i1: dict(result)  # __call__(img0, img1)
    m.supports_batches = False  # a MagicMock attribute would be truthy
    feats = {"all_kpts0": all_kpts0, "all_desc0": result["all_desc0"]}
    # vismatch extract: a list in -> one dict per image; a single image in -> one dict
    m.extract.side_effect = lambda imgs: [dict(feats) for _ in imgs] if isinstance(imgs, list) else dict(feats)
    return m


def _batch_vismatch_matcher(n_kpts: int = 8, d: int = 64) -> MagicMock:
    """
    Mock vismatch matcher that supports cached-feature matching.
    """
    m = _fake_vismatch_matcher(n_kpts=n_kpts, d=d)
    m.supports_batches = True
    return m


@patch("vismatch.get_matcher")
def test_non_batch_model_refused(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()

    with pytest.raises(ValueError, match="supports_batches"):
        LocalMatcher("disk-lightglue", device="cpu")


@patch("vismatch.get_matcher")
def test_max_num_keypoints_forwarded(mock_get):
    mock_get.return_value = _batch_vismatch_matcher()

    matcher = LocalMatcher("xfeat", device="cpu", max_num_keypoints=512)

    mock_get.assert_called_once_with("xfeat", device="cpu", max_num_keypoints=512)
    assert matcher.max_num_keypoints == 512


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))],
)
@patch("vismatch.get_matcher")
def test_to_device_moves_every_tensor(mock_get, device):
    mock_get.return_value = _batch_vismatch_matcher()
    lm = LocalMatcher("xfeat", device=device)
    feats = LocalFeatures(
        keypoints=torch.zeros(3, 2, dtype=torch.float64),
        descriptors=torch.zeros(3, 4),
        keypoints_normalized=torch.zeros(3, 2),
        image_size=(10, 10),
    )

    moved = lm.to_device(feats)

    names = ("keypoints", "descriptors", "keypoints_normalized")

    for name in names:
        before = getattr(feats, name)
        after = getattr(moved, name)
        assert after.device.type == device
        assert after.dtype == before.dtype
        assert before.device.type == "cpu"

    assert moved.image_size == (10, 10)
    assert moved is not feats


@patch("vismatch.get_matcher")
def test_match_returns_native_indices(mock_get):
    m = _batch_vismatch_matcher()
    m.match.return_value = {
        "matched_kpts0": np.ones((2, 2), np.float32),
        "matched_kpts1": np.ones((2, 2), np.float32),
        "matched_idxs0": np.array([0, 3]),
        "matched_idxs1": np.array([1, 2]),
    }
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu")
    f = LocalFeatures(keypoints=torch.zeros(4, 2), descriptors=torch.zeros(4, 8), image_size=(10, 10))

    res = lm.match(f, f)

    np.testing.assert_array_equal(res.idx_q, [0, 3])
    np.testing.assert_array_equal(res.idx_db, [1, 2])
    assert res.idx_q.dtype == np.int64


@patch("vismatch.get_matcher")
def test_extract_returns_local_features(mock_get):
    mock_get.return_value = _batch_vismatch_matcher()
    lm = LocalMatcher("xfeat", device="cpu")
    feats = lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert isinstance(feats, LocalFeatures)
    assert feats.keypoints.shape == (8, 2) and feats.keypoints.dtype == torch.float32
    assert feats.descriptors.shape == (8, 64)
    mock_get.assert_called_once_with("xfeat", device="cpu", max_num_keypoints=2048)


@patch("vismatch.get_matcher")
def test_extract_handles_tensor_outputs(mock_get):
    # Real matchers may return torch tensors (check_types allows tensor or ndarray).
    m = _batch_vismatch_matcher()
    prev = m.extract([None])[0]
    tensor_feats = {"all_kpts0": torch.from_numpy(prev["all_kpts0"]), "all_desc0": torch.from_numpy(prev["all_desc0"])}
    m.extract.side_effect = lambda imgs: [dict(tensor_feats)]
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu")
    feats = lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert feats.keypoints.shape == (8, 2) and feats.keypoints.dtype == torch.float32
    assert feats.descriptors.shape == (8, 64)


@patch("vismatch.get_matcher")
def test_extract_asserts_pixel_frame(mock_get):
    # Keypoints outside the input image bounds = coordinate-frame violation (92f2e4a class)
    m = _batch_vismatch_matcher()
    bad = {"all_kpts0": np.array([[500.0, 500.0]], dtype=np.float32), "all_desc0": np.zeros((1, 64), dtype=np.float32)}
    m.extract.side_effect = lambda imgs: [dict(bad)]
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu")
    with pytest.raises(ValueError, match="pixel frame"):
        lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))


@patch("vismatch.get_matcher")
def test_extract_passes_chw_uint8_list(mock_get):
    # vismatch input: a list of (3, H, W) uint8 tensors; vismatch scales uint8 on device itself
    m = _batch_vismatch_matcher()
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu")
    lm.extract(np.full((100, 120, 3), 255, dtype=np.uint8))
    (imgs,), _ = m.extract.call_args
    assert isinstance(imgs, list) and len(imgs) == 1
    assert imgs[0].shape == (3, 100, 120) and imgs[0].dtype == torch.uint8


@patch("vismatch.get_matcher")
def test_extract_list_returns_one_feature_set_per_image(mock_get):
    m = _batch_vismatch_matcher()
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu")
    feats = lm.extract([np.zeros((100, 120, 3), np.uint8), np.zeros((100, 110, 3), np.uint8)])
    assert m.extract.call_count == 1  # one batched vismatch call
    assert [f.image_size for f in feats] == [(120, 100), (110, 100)]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]
)
@patch("vismatch.get_matcher")
def test_to_tensor_keeps_dtype_on_device(mock_get, device, dtype):
    # Upload as-is (uint8 is 4x fewer bytes); vismatch's to_tensor_image scales uint8 to [0, 1]
    mock_get.return_value = _batch_vismatch_matcher()
    lm = LocalMatcher("xfeat", device=device)
    rgb = np.random.default_rng(0).integers(0, 256, (37, 53, 3)).astype(np.uint8)
    image = rgb if dtype == np.uint8 else rgb.astype(np.float32) / 255.0

    out = lm._to_tensor(image)

    assert out.device.type == device and out.shape == (3, 37, 53)
    torch.testing.assert_close(out.cpu(), torch.from_numpy(image).permute(2, 0, 1), atol=0, rtol=0)


def _one_hot_features(rows, d=8, keypoints_normalized=False):
    """LocalFeatures whose descriptors are one-hot rows — mutual-NN is exactly identity."""
    kpts = np.stack([np.arange(len(rows)), np.arange(len(rows))], axis=1).astype(np.float32) * 10
    desc = np.eye(d, dtype=np.float32)[rows]
    norm = torch.from_numpy(kpts / 100) if keypoints_normalized else None
    return LocalFeatures(
        keypoints=torch.from_numpy(kpts),
        descriptors=torch.from_numpy(desc),
        keypoints_normalized=norm,
        image_size=(100, 80),
    )


@patch("vismatch.get_matcher")
def test_match_passes_features_and_maps_indices(mock_get):
    m = _batch_vismatch_matcher()
    m.match.return_value = {
        "matched_kpts0": np.array([[0.0, 0.0], [20.0, 20.0]], np.float32),
        "matched_kpts1": np.array([[30.0, 30.0], [10.0, 10.0]], np.float32),
        "matched_idxs0": np.array([0, 2]),
        "matched_idxs1": np.array([3, 1]),
    }
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu")
    q = _one_hot_features([0, 1, 2, 3])
    db = _one_hot_features([3, 2, 1, 0], keypoints_normalized=True)
    res = lm.match(q, db)

    (f0, f1), _ = m.match.call_args
    assert f0["image_size"] == (100, 80) and "kpts_normalized" not in f0
    assert f0["all_kpts0"] is q.keypoints and f0["all_desc0"] is q.descriptors
    assert torch.equal(f1["kpts_normalized"], db.keypoints_normalized)
    np.testing.assert_array_equal(res.idx_q, [0, 2])
    np.testing.assert_array_equal(res.idx_db, [3, 1])
    np.testing.assert_array_equal(res.ref_px, [[30.0, 30.0], [10.0, 10.0]])
    assert res.idx_q.dtype == np.int64 and res.query_px.dtype == np.float32


@patch("vismatch.get_matcher")
def test_init_skips_vismatch_ransac(mock_get):
    # Callers verify geometry with pycolmap; vismatch's homography RANSAC is wasted work
    mock_get.return_value = _batch_vismatch_matcher()
    lm = LocalMatcher("xfeat", device="cpu")
    assert lm._matcher.skip_ransac is True


@patch("vismatch.get_matcher")
def test_match_empty_descriptors_returns_empty(mock_get):
    mock_get.return_value = _batch_vismatch_matcher()
    lm = LocalMatcher("xfeat", device="cpu")
    empty = LocalFeatures(keypoints=torch.zeros((0, 2)), descriptors=torch.zeros((0, 8)))
    m = lm.match(empty, _one_hot_features([0, 1]))
    assert len(m) == 0 and m.idx_q is not None  # empty but indexable


########################################
# GPU parity gates (real models) — vismatch match() vs the pair forward
########################################

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-model parity needs CUDA")


@pytest.fixture
def strict_fp32():
    """Pin full-precision fp32 matmul: mapanything flips TF32 on at module import
    (via collab_splats.pointcloud.feedforward.base), and under TF32 the batched vs
    pair-forward paths flip near-threshold matches — parity is byte-equal only at
    'highest'. Any test module importing feedforward.base in the same pytest
    process would otherwise poison these gates at collection time."""
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_prec = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    torch.set_float32_matmul_precision(prev_prec)


def _real_pair(seed=3):
    """Synthetic textured pair: noise image + a horizontally rolled copy."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 255, (240, 320, 3)).astype(np.uint8)
    return a, np.roll(a, 12, axis=1)


@requires_cuda
def test_real_loma_extract_keeps_normalized_payload(strict_fp32):
    """loma extract() fills keypoints_normalized (the payload match() needs) and image_size."""
    lm = LocalMatcher("loma")
    a, _ = _real_pair()
    feats = lm.extract(a)
    assert feats.keypoints_normalized is not None and len(feats.keypoints_normalized) == len(feats.keypoints)
    assert feats.image_size == (320, 240)


@requires_cuda
def test_real_loma_match_parity_after_zarr_roundtrip(tmp_path, strict_fp32, replay_matcher):
    """match() on zarr-roundtripped features == the plain pair forward (float64 pair forward vs float32 cache)."""
    lm = LocalMatcher("loma")
    a, b = _real_pair(seed=4)
    with torch.inference_mode():
        ref = lm._matcher(lm._to_tensor(a), lm._to_tensor(b))
    assert len(ref["matched_kpts0"]) > 0

    # extract -> save via CameraLocalizer -> load -> match()
    feats = [lm.extract(a), lm.extract(b)]
    extractor = replay_matcher(feats)
    loc = CameraLocalizer(
        world_points=np.zeros((2, 8, 8, 3), dtype=np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        images=[a, b],
        ids=["a", "b"],
        extractor=extractor,
    )
    loc.save_index(tmp_path / "ff.zarr", "loma")
    loaded, _, _ = read_localization_db(tmp_path / "ff.zarr", "loma")
    assert all(f.keypoints_normalized is not None and f.image_size == (320, 240) for f in loaded)
    m = lm.match(loaded[0], loaded[1])
    np.testing.assert_allclose(m.query_px, ref["matched_kpts0"], atol=1e-4)
    np.testing.assert_allclose(m.ref_px, ref["matched_kpts1"], atol=1e-4)
    np.testing.assert_array_equal(loaded[0].keypoints.numpy()[m.idx_q], m.query_px)
    np.testing.assert_array_equal(loaded[1].keypoints.numpy()[m.idx_db], m.ref_px)


@requires_cuda
def test_real_xfeat_match_matches_pair_forward(strict_fp32):
    """xfeat match() over extract() features == the plain pair forward."""
    lm = LocalMatcher("xfeat")
    a, b = _real_pair(seed=5)
    fa = lm.extract(a)
    fb = lm.extract(b)
    m = lm.match(fa, fb)
    np.testing.assert_array_equal(fa.keypoints.numpy()[m.idx_q], m.query_px)
    np.testing.assert_array_equal(fb.keypoints.numpy()[m.idx_db], m.ref_px)
    with torch.inference_mode():
        ref = lm._matcher(lm._to_tensor(a), lm._to_tensor(b))
    ref_q = np.asarray(ref["matched_kpts0"])
    ref_db = np.asarray(ref["matched_kpts1"])
    assert len(m) == len(ref_q) and len(m) > 0
    order_m = np.lexsort(m.query_px.T)
    order_ref = np.lexsort(ref_q.T)
    np.testing.assert_array_equal(m.query_px[order_m], ref_q[order_ref])
    np.testing.assert_array_equal(m.ref_px[order_m], ref_db[order_ref])


@requires_cuda
@pytest.mark.parametrize("model", ["xfeat", "loma"])
def test_real_batched_extract_keypoints_overlap_single_extract(model, strict_fp32):
    """
    Batched extract keypoints overlap single-image extract keypoints.

    - the batched forward is numerically close to the single forward, not bit-identical
    """
    lm = LocalMatcher(model)
    overlaps = []

    for seed in (3, 4, 5, 6):
        a, b = _real_pair(seed=seed)
        batched = lm.extract([a, b])
        singles = [lm.extract(a), lm.extract(b)]

        for got, want in zip(batched, singles):
            dist = torch.cdist(want.keypoints, got.keypoints)
            overlaps.append((dist.min(dim=1).values <= 0.5).float().mean().item())

    assert min(overlaps) >= 0.99, overlaps
