"""Unit tests for compute_multiview_depth_confidence in base.py."""
import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.base import compute_multiview_depth_confidence


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
    )

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
    )

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
    )

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
        )
        assert out.shape == (N, H, W), f"N={N}: expected {(N,H,W)}, got {out.shape}"
