"""Unit tests for LocalMatcher — vismatch mocked throughout; no model downloads."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.localization.extractors import (
    LocalFeatures,
    LocalMatcher,
    MatchResult,
)
from collab_splats.localization.localizer import CameraLocalizer, read_localization_db


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
    m.supports_batches = False  # a MagicMock attribute would be truthy
    feats = {"all_kpts0": all_kpts0, "all_desc0": result["all_desc0"]}
    # vismatch extract: a list in -> one dict per image; a single image in -> one dict
    m.extract.side_effect = lambda imgs: [dict(feats) for _ in imgs] if isinstance(imgs, list) else dict(feats)
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
    prev = m.extract([None])[0]
    tensor_feats = {"all_kpts0": torch.from_numpy(prev["all_kpts0"]), "all_desc0": torch.from_numpy(prev["all_desc0"])}
    m.extract.side_effect = lambda imgs: [dict(tensor_feats)]
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    feats = lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert feats.keypoints.shape == (8, 2) and feats.keypoints.dtype == torch.float32
    assert feats.descriptors.shape == (8, 64)


@patch("vismatch.get_matcher")
def test_extract_asserts_pixel_frame(mock_get):
    # Keypoints outside the input image bounds = coordinate-frame violation (92f2e4a class)
    m = _fake_vismatch_matcher()
    bad = {"all_kpts0": np.array([[500.0, 500.0]], dtype=np.float32), "all_desc0": np.zeros((1, 64), dtype=np.float32)}
    m.extract.side_effect = lambda imgs: [dict(bad)]
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    with pytest.raises(ValueError, match="pixel frame"):
        lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))


@patch("vismatch.get_matcher")
def test_extract_passes_chw_uint8_list(mock_get):
    # vismatch input: a list of (3, H, W) uint8 tensors; vismatch scales uint8 on device itself
    m = _fake_vismatch_matcher()
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    lm.extract(np.full((100, 120, 3), 255, dtype=np.uint8))
    (imgs,), _ = m.extract.call_args
    assert isinstance(imgs, list) and len(imgs) == 1
    assert imgs[0].shape == (3, 100, 120) and imgs[0].dtype == torch.uint8


@patch("vismatch.get_matcher")
def test_extract_list_returns_one_feature_set_per_image(mock_get):
    m = _fake_vismatch_matcher()
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    feats = lm.extract([np.zeros((100, 120, 3), np.uint8), np.zeros((100, 110, 3), np.uint8)])
    assert m.extract.call_count == 1  # one batched vismatch call
    assert [f.image_size for f in feats] == [(120, 100), (110, 100)]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])
@patch("vismatch.get_matcher")
def test_to_tensor_keeps_dtype_on_device(mock_get, device, dtype):
    # Upload as-is (uint8 is 4x fewer bytes); vismatch's to_tensor_image scales uint8 to [0, 1]
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device=device, probe=False)
    rgb = np.random.default_rng(0).integers(0, 256, (37, 53, 3)).astype(np.uint8)
    image = rgb if dtype == np.uint8 else rgb.astype(np.float32) / 255.0

    out = lm._to_tensor(image)

    assert out.device.type == device and out.shape == (3, 37, 53)
    torch.testing.assert_close(out.cpu(), torch.from_numpy(image).permute(2, 0, 1), atol=0, rtol=0)


def test_blocked_models_raise_without_importing_vismatch():
    # Blocklist check happens before the vismatch import — a None sys.modules entry
    # would make any import attempt raise ImportError, so ValueError proves the order.
    with patch.dict(sys.modules, {"vismatch": None}):
        with pytest.raises(ValueError, match="dependency"):
            LocalMatcher("ufm", device="cpu", probe=False)
        with pytest.raises(ValueError, match="license"):
            LocalMatcher("superglue", device="cpu", probe=False)


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


def _one_hot_features(rows, d=8, keypoints_normalized=False):
    """LocalFeatures whose descriptors are one-hot rows — mutual-NN is exactly identity."""
    kpts = np.stack([np.arange(len(rows)), np.arange(len(rows))], axis=1).astype(np.float32) * 10
    desc = np.eye(d, dtype=np.float32)[rows]
    norm = torch.from_numpy(kpts / 100) if keypoints_normalized else None
    return LocalFeatures(
        keypoints=torch.from_numpy(kpts), descriptors=torch.from_numpy(desc), keypoints_normalized=norm, image_size=(100, 80)
    )


@patch("vismatch.get_matcher")
def test_match_unsupported_without_vismatch_batching(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()  # supports_batches False
    lm = LocalMatcher("roma", device="cpu", probe=False)
    q = _one_hot_features([0, 1])
    with pytest.raises(NotImplementedError, match="match_images"):
        lm.match(q, q)


@patch("vismatch.get_matcher")
def test_match_passes_features_and_maps_indices(mock_get):
    m = _fake_vismatch_matcher()
    m.supports_batches = True
    m.match.return_value = {
        "matched_kpts0": np.array([[0.0, 0.0], [20.0, 20.0]], np.float32),
        "matched_kpts1": np.array([[30.0, 30.0], [10.0, 10.0]], np.float32),
        "matched_idxs0": np.array([0, 2]),
        "matched_idxs1": np.array([3, 1]),
    }
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu", probe=False)
    q = _one_hot_features([0, 1, 2, 3])
    db = _one_hot_features([3, 2, 1, 0], keypoints_normalized=True)
    res = lm.match(q, db)

    (f0, f1), _ = m.match.call_args
    assert f0["image_size"] == (100, 80) and "kpts_normalized" not in f0
    assert torch.equal(f1["kpts_normalized"], db.keypoints_normalized)
    np.testing.assert_array_equal(res.idx_q, [0, 2])
    np.testing.assert_array_equal(res.idx_db, [3, 1])
    np.testing.assert_array_equal(res.ref_px, [[30.0, 30.0], [10.0, 10.0]])
    assert res.idx_q.dtype == np.int64 and res.query_px.dtype == np.float32


@patch("vismatch.get_matcher")
def test_init_skips_vismatch_ransac(mock_get):
    # Callers verify geometry with pycolmap; vismatch's homography RANSAC is wasted work
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    assert lm._matcher.skip_ransac is True


@patch("vismatch.get_matcher")
def test_match_empty_descriptors_returns_empty(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    mock_get.return_value.supports_batches = True
    lm = LocalMatcher("xfeat", device="cpu", probe=False)
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
    pairwise paths flip near-threshold matches — parity is byte-equal only at
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
    """Synthetic textured pair: noise image + a horizontally rolled copy (probe pattern)."""
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
def test_real_loma_match_parity_after_zarr_roundtrip(tmp_path, strict_fp32):
    """match() on zarr-roundtripped features == the plain pair forward, byte-identical."""
    lm = LocalMatcher("loma")
    a, b = _real_pair(seed=4)
    ref = lm.match_images(a, b)
    assert len(ref) > 0

    # extract -> save via CameraLocalizer -> load -> match()
    feats = [lm.extract(a), lm.extract(b)]
    extractor = MagicMock(spec=LocalMatcher)
    extractor.extract.side_effect = list(feats)
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
    np.testing.assert_array_equal(m.query_px, ref.query_px)
    np.testing.assert_array_equal(m.ref_px, ref.ref_px)
    np.testing.assert_array_equal(m.idx_q, ref.idx_q)  # native == recovered (probe-exact)
    np.testing.assert_array_equal(m.idx_db, ref.idx_db)


@requires_cuda
def test_real_xfeat_match_matches_pairwise(strict_fp32):
    """xfeat match() over extract() features == match_images() on the pair."""
    lm = LocalMatcher("xfeat")
    a, b = _real_pair(seed=5)
    m = lm.match(lm.extract(a), lm.extract(b))
    ref = lm.match_images(a, b)
    assert len(m) == len(ref) and len(m) > 0
    order_m, order_ref = np.argsort(m.idx_q), np.argsort(ref.idx_q)
    np.testing.assert_array_equal(m.idx_q[order_m], ref.idx_q[order_ref])
    np.testing.assert_array_equal(m.idx_db[order_m], ref.idx_db[order_ref])
