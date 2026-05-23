"""Tests for collab_splats.pointcloud.feedforward.mapanything._reproject_mapanything."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Pre-import so patch.dict teardown doesn't evict it from sys.modules and break
# subsequent tests that patch collab_splats.pointcloud.feedforward.mapanything.run_mapanything.
import collab_splats.pointcloud.feedforward.mapanything  # noqa: F401


def _make_tensor(arr: np.ndarray) -> MagicMock:
    """Wrap a numpy array in a minimal tensor mock that supports [0], .cpu(), .numpy()."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, idx: _make_numpy_mock(arr)
    return mock


def _make_numpy_mock(arr: np.ndarray) -> MagicMock:
    """Mock for tensor[0] that supports .cpu().numpy() and .squeeze(-1).cpu().numpy()."""
    m = MagicMock()
    m.cpu.return_value = m
    m.numpy.return_value = arr
    # Support squeeze(-1) — returns a mock that also resolves to arr with last dim removed
    squeezed = MagicMock()
    squeezed.cpu.return_value = squeezed
    squeezed.numpy.return_value = arr[..., 0] if (arr.ndim > 2 and arr.shape[-1] == 1) else arr
    m.squeeze.return_value = squeezed
    return m


def _make_raw_outputs(n_frames: int, H: int = 4, W: int = 4) -> list[dict]:
    """Build fake raw_outputs with deterministic values for n_frames."""
    rng = np.random.default_rng(42)
    outputs = []
    for i in range(n_frames):
        pts3d_cam = rng.standard_normal((H, W, 3)).astype(np.float32)
        mask = np.ones((H, W, 1), dtype=np.float32)
        depth_z = np.ones((H, W, 1), dtype=np.float32) * 2.0  # all positive
        img_no_norm = rng.uniform(0, 1, (H, W, 3)).astype(np.float32)

        outputs.append(
            {
                "pts3d_cam": _make_tensor(pts3d_cam),
                "mask": _make_tensor(mask),
                "depth_z": _make_tensor(depth_z),
                "img_no_norm": _make_tensor(img_no_norm),
            }
        )
    return outputs


def _identity_extrinsics(n: int) -> np.ndarray:
    """Return n identity world2cam matrices (3, 4)."""
    ext = np.zeros((n, 3, 4), dtype=np.float32)
    for i in range(n):
        ext[i, :3, :3] = np.eye(3, dtype=np.float32)
    return ext


# ---------------------------------------------------------------------------
# closed_form_pose_inverse stub: for R|t world2cam → returns R.T | -R.T t cam2world
# ---------------------------------------------------------------------------
def _stub_pose_inverse(ext_4x4_batch: np.ndarray) -> np.ndarray:
    """Minimal closed_form_pose_inverse for testing (batch of 4x4 matrices)."""
    out = []
    for m in ext_4x4_batch:
        R = m[:3, :3]
        t = m[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        inv = np.eye(4, dtype=np.float64)
        inv[:3, :3] = R_inv
        inv[:3, 3] = t_inv
        out.append(inv)
    return np.stack(out)


@pytest.fixture(autouse=True)
def patch_mapanything_import():
    """Patch sys.modules so the local `from mapanything...` import inside the function resolves."""
    mock_geo = MagicMock()
    mock_geo.closed_form_pose_inverse = _stub_pose_inverse
    with patch.dict(
        "sys.modules",
        {
            "mapanything": MagicMock(),
            "mapanything.utils": MagicMock(),
            "mapanything.utils.geometry": mock_geo,
        },
    ):
        yield


def test_reproject_mapanything_shape():
    """Output arrays have correct shapes: (P, 3) pts and (P, 3) uint8 colors."""
    from collab_splats.pointcloud.feedforward.mapanything import _reproject_mapanything

    n_frames = 3
    H, W = 4, 4
    raw_outputs = _make_raw_outputs(n_frames, H, W)
    extrinsics = _identity_extrinsics(n_frames)

    pts3d, colors = _reproject_mapanything(raw_outputs, extrinsics)

    expected_points = n_frames * H * W  # all mask=1, depth>0
    assert pts3d.shape == (expected_points, 3), f"pts3d shape {pts3d.shape}"
    assert colors.shape == (expected_points, 3), f"colors shape {colors.shape}"
    assert pts3d.dtype == np.float32
    assert colors.dtype == np.uint8


def test_reproject_mapanything_uses_refined_extrinsics():
    """Output pts3d must differ when extrinsics change (non-trivial rotation/translation)."""
    from collab_splats.pointcloud.feedforward.mapanything import _reproject_mapanything

    n_frames = 2
    H, W = 3, 3
    raw_outputs = _make_raw_outputs(n_frames, H, W)

    # Extrinsics set A: identity world2cam
    ext_a = _identity_extrinsics(n_frames)

    # Extrinsics set B: translate by [1, 2, 3] in camera space
    ext_b = ext_a.copy()
    ext_b[:, :3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    pts_a, _ = _reproject_mapanything(raw_outputs, ext_a)
    pts_b, _ = _reproject_mapanything(raw_outputs, ext_b)

    assert pts_a.shape == pts_b.shape, "Shape must be consistent across extrinsic sets"
    assert not np.allclose(pts_a, pts_b), "pts3d must differ when extrinsics differ"
