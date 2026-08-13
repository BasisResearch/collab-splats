"""Unit tests for compute_multiview_depth_confidence in base.py."""
import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.base import (
    MultiviewConfidence,
    compute_multiview_depth_confidence,
)


def _make_intrinsics(H: int, W: int) -> np.ndarray:
    """Simple pinhole K with focal = W, principal at image centre."""
    return np.array(
        [[float(W), 0.0, W / 2.0], [0.0, float(H), H / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_compute_mv_conf_identical_cameras():
    """Two co-located cameras, same depth → mv_conf = 1.0 for all valid pixels."""
    N, H, W = 2, 8, 8
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)

    mv_conf = compute_multiview_depth_confidence(
        depth, intrinsics, extrinsics, abs_thresh=0.0, rel_thresh=0.1, device="cpu"
    ).ratio

    assert mv_conf.shape == (N, H, W)
    assert mv_conf.dtype == np.float32
    assert np.all(mv_conf > 0.9), f"Expected all >0.9; min={mv_conf.min():.4f}"


def test_compute_mv_conf_depth_disagreement():
    """Same pose, very different depths → mv_conf = 0.0 everywhere."""
    N, H, W = 2, 4, 4
    depth = np.zeros((N, H, W), dtype=np.float32)
    depth[0] = 1.0
    depth[1] = 100.0

    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)

    mv_conf = compute_multiview_depth_confidence(
        depth, intrinsics, extrinsics, abs_thresh=0.0, rel_thresh=0.05, device="cpu"
    ).ratio

    assert np.all(mv_conf == 0.0), f"Expected all 0.0; max={mv_conf.max():.4f}"


def test_compute_mv_conf_depth_masks_source():
    """depth_masks=False on frame 0 → frame 0 source pixels get mv_conf = 0."""
    N, H, W = 2, 4, 4
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)

    depth_masks = np.ones((N, H, W), dtype=bool)
    depth_masks[0] = False

    mv_conf = compute_multiview_depth_confidence(
        depth, intrinsics, extrinsics,
        depth_masks=depth_masks, abs_thresh=0.0, rel_thresh=0.1, device="cpu",
    ).ratio

    assert np.all(mv_conf[0] == 0.0), f"Frame 0 should be 0; got {mv_conf[0]}"
    assert np.any(mv_conf[1] > 0.0), "Frame 1 should have some inliers"


def test_compute_mv_conf_output_shape():
    """Output shape matches (N, H, W) regardless of N."""
    for N in (1, 3, 5):
        H, W = 6, 6
        depth = np.ones((N, H, W), dtype=np.float32) * 3.0
        K = _make_intrinsics(H, W)
        intrinsics = np.stack([K] * N)
        extrinsics = np.stack([np.eye(4, dtype=np.float32)] * N)
        out = compute_multiview_depth_confidence(
            depth, intrinsics, extrinsics, device="cpu"
        ).ratio
        assert out.shape == (N, H, W), f"N={N}: expected {(N,H,W)}, got {out.shape}"


def test_nearest_sampling_no_fabricated_depth():
    """Bilinear across a depth step invents a depth on no surface; nearest cannot.

    Two cameras separated along x view a scene that is 2.0 deep on the left half and
    8.0 on the right. Sampled depth must be one of the two surface depths, never between.
    """
    N, H, W = 2, 16, 16
    depth = np.empty((N, H, W), dtype=np.float32)
    depth[:, :, : W // 2] = 2.0
    depth[:, :, W // 2 :] = 8.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    extr[1, 0, 3] = 0.05  # small baseline so the step lands mid-pixel in view 1

    out = compute_multiview_depth_confidence(
        depth, np.stack([K, K]), extr, abs_thresh=0.0, rel_thresh=0.05, device="cpu"
    )
    # Every pixel here sits on a rigid surface seen by both cameras, so with nearest
    # sampling every comparison is against a real surface depth and all of them agree.
    # Under bilinear the column just left of the step samples 0.6*2.0 + 0.4*8.0 = 3.6 —
    # a depth on no surface — and is scored an outlier. Judged pixels only: the last
    # column projects off the right edge of view 1 and has no partner at all.
    judged_px = out.valid_count[0] > 0
    assert judged_px.sum() > 0
    assert np.all(out.ratio[0][judged_px] == 1.0), (
        f"{int((out.ratio[0][judged_px] < 1.0).sum())} judged px below 1.0 — "
        f"sampler is fabricating depth"
    )


def test_pair_gate_does_not_change_output():
    """The gate is a cost optimisation: gated and ungated results must be identical."""
    N, H, W = 4, 8, 8
    rng = np.random.default_rng(7)
    depth = (rng.random((N, H, W)).astype(np.float32) + 1.0) * 3.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    for i in range(N):
        extr[i, 0, 3] = 0.3 * i
    kw = dict(abs_thresh=0.0, rel_thresh=0.05, device="cpu")
    gated = compute_multiview_depth_confidence(depth, np.stack([K] * N), extr, pair_gate=True, **kw)
    plain = compute_multiview_depth_confidence(depth, np.stack([K] * N), extr, pair_gate=False, **kw)
    np.testing.assert_array_equal(gated.inlier_count, plain.inlier_count)
    np.testing.assert_array_equal(gated.valid_count, plain.valid_count)
    np.testing.assert_array_equal(gated.judged, plain.judged)


def test_pair_gate_skips_disjoint_views():
    """Two views looking at scenes 1000 units apart share no frustum volume."""
    N, H, W = 2, 8, 8
    depth = np.full((N, H, W), 2.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    extr[1, 0, 3] = 1000.0
    out = compute_multiview_depth_confidence(
        depth, np.stack([K, K]), extr, pair_gate=True, device="cpu"
    )
    assert out.valid_count.max() == 0, "disjoint views should contribute nothing"
    assert out.judged.tolist() == [False, False]


def test_returns_multiview_confidence_dataclass():
    """The function returns MultiviewConfidence with ratio, both counts, and judged."""
    N, H, W = 2, 4, 4
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    out = compute_multiview_depth_confidence(
        depth, np.stack([K, K]), np.stack([np.eye(4, dtype=np.float32)] * 2),
        abs_thresh=0.0, rel_thresh=0.1, device="cpu",
    )
    assert isinstance(out, MultiviewConfidence)
    assert out.ratio.shape == (N, H, W)
    assert out.ratio.dtype == np.float32
    assert out.inlier_count.shape == (N, H, W)
    assert out.inlier_count.dtype == np.int32
    assert out.valid_count.shape == (N, H, W)
    assert out.valid_count.dtype == np.int32
    assert out.judged.shape == (N,)
    assert out.judged.dtype == np.bool_


def test_counts_consistent_with_ratio():
    """ratio == inlier_count / valid_count wherever valid_count > 0, else 0."""
    N, H, W = 3, 6, 6
    rng = np.random.default_rng(0)
    depth = (rng.random((N, H, W)).astype(np.float32) + 1.0) * 2.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    for i in range(N):
        extr[i, 0, 3] = 0.1 * i
    out = compute_multiview_depth_confidence(
        depth, np.stack([K] * N), extr, abs_thresh=0.0, rel_thresh=0.1, device="cpu"
    )
    expected = np.where(
        out.valid_count > 0,
        out.inlier_count / np.maximum(out.valid_count, 1),
        0.0,
    ).astype(np.float32)
    np.testing.assert_allclose(out.ratio, expected, rtol=0, atol=0)
    assert np.all(out.inlier_count <= out.valid_count)


def test_single_view_is_unjudged():
    """N=1 has no partners: judged is False and every count is zero."""
    H, W = 4, 4
    depth = np.full((1, H, W), 3.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    out = compute_multiview_depth_confidence(
        depth, K[None], np.eye(4, dtype=np.float32)[None], device="cpu"
    )
    assert out.judged.tolist() == [False]
    assert out.valid_count.max() == 0
    assert out.inlier_count.max() == 0
