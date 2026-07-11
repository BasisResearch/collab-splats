import numpy as np
import pytest
from collab_splats.geometry.loop_closure.closure import dedup_overlap


def _identity_poses(k: int) -> np.ndarray:
    return np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))


def _shifted_poses(k: int, offset: float) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    poses[:, :3, 3] = offset
    return poses


def test_dedup_no_overlap():
    """Two non-overlapping submaps: output shape == total_frames."""
    corrected = {0: _identity_poses(5), 1: _identity_poses(5)}
    result = dedup_overlap([0, 1], [0, 5], corrected, total_frames=10)
    assert result.shape == (10, 4, 4)


def test_dedup_with_overlap_shape():
    """N=100, K=20, O=4: step=16, output must be (100, 4, 4)."""
    N, K, O = 100, 20, 4
    step = K - O  # 16
    corrected = {}
    submap_ids = []
    submap_starts = []
    for wi, start in enumerate(range(0, N, step)):
        end = min(start + K, N)
        k = end - start
        corrected[wi] = _identity_poses(k)
        submap_ids.append(wi)
        submap_starts.append(start)
    result = dedup_overlap(submap_ids, submap_starts, corrected, total_frames=N)
    assert result.shape == (100, 4, 4)


def test_dedup_small():
    """N=21, K=20, O=4: output must be (21, 4, 4)."""
    N, K, O = 21, 20, 4
    step = K - O
    corrected = {}
    submap_ids = []
    submap_starts = []
    for wi, start in enumerate(range(0, N, step)):
        end = min(start + K, N)
        k = end - start
        corrected[wi] = _identity_poses(k)
        submap_ids.append(wi)
        submap_starts.append(start)
    result = dedup_overlap(submap_ids, submap_starts, corrected, total_frames=N)
    assert result.shape == (21, 4, 4)


def test_dedup_canonical_owner():
    """Overlap frames keep the EARLIER submap's pose (canonical-owner rule)."""
    # submap 0: frames 0-4 (5 frames), submap 1: frames 3-7 (5 frames), overlap = frames 3,4
    poses_a = _identity_poses(5)       # all identity
    poses_b = _shifted_poses(5, 99.0)  # all shifted — should NOT appear at frames 3,4
    corrected = {0: poses_a, 1: poses_b}
    result = dedup_overlap([0, 1], [0, 3], corrected, total_frames=8)
    assert result.shape == (8, 4, 4)
    # frames 3 and 4 must come from submap 0 (identity, not shifted)
    assert np.allclose(result[3], np.eye(4))
    assert np.allclose(result[4], np.eye(4))
    # frame 5 = local 2 of submap 1 = shifted
    assert np.allclose(result[5, :3, 3], 99.0)
