"""Tests for collab_splats.pointcloud.bundle_adjustment.

bae and vggt may not be installed in CI; all heavy imports are mocked.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers — build a minimal mock hierarchy so the module can be imported
# without bae / vggt present.
# ---------------------------------------------------------------------------

def _make_vggt_mock():
    """Return a mock package tree for vggt.dependency.track_predict / projection."""
    vggt = types.ModuleType("vggt")
    vggt.dependency = types.ModuleType("vggt.dependency")
    tp = types.ModuleType("vggt.dependency.track_predict")
    tp.predict_tracks = MagicMock()
    proj = types.ModuleType("vggt.dependency.projection")
    proj.project_3D_points_np = MagicMock()
    vggt.dependency.track_predict = tp
    vggt.dependency.projection = proj
    return vggt, tp, proj


def _make_bae_mock():
    """Return a mock package tree for bae.*."""
    bae = types.ModuleType("bae")
    af = types.ModuleType("bae.autograd")
    af_func = types.ModuleType("bae.autograd.function")
    af_func.TrackingTensor = lambda x: x
    af_func.map_transform = lambda fn: fn  # identity decorator
    bae.autograd = af
    bae.autograd.function = af_func

    utils = types.ModuleType("bae.utils")
    utils_py = types.ModuleType("bae.utils.pysolvers")
    utils_py.PCG = MagicMock
    utils_ba = types.ModuleType("bae.utils.ba")
    utils_ba.rotate_quat = MagicMock(side_effect=lambda pts, _pose: pts)
    bae.utils = utils
    bae.utils.pysolvers = utils_py
    bae.utils.ba = utils_ba

    optim_mod = types.ModuleType("bae.optim")
    optim_mod.LM = MagicMock()
    bae.optim = optim_mod

    return bae


# ---------------------------------------------------------------------------
# Test 1: extract_tracks_vggsfm — shape correctness with mocked predict_tracks
# ---------------------------------------------------------------------------

def test_extract_tracks_vggsfm_shape():
    N, H, W = 6, 64, 64
    P = 50  # vggt's predict_tracks already concatenates per-query-frame results
            # internally and returns single np.ndarrays — mock matches that contract.

    images = torch.zeros(N, 3, H, W)
    conf = torch.ones(N, H, W)

    tracks_arr = np.random.rand(N, P, 2).astype(np.float32)
    vis_arr = np.random.rand(N, P).astype(np.float32)
    confs_arr = np.ones((N, P), dtype=np.float32)
    pts3d_arr = np.random.rand(P, 3).astype(np.float32)
    colors_arr = np.ones((P, 3), dtype=np.float32)

    mock_predict = MagicMock(
        return_value=(tracks_arr, vis_arr, confs_arr, pts3d_arr, colors_arr)
    )

    vggt_mod, tp_mod, _ = _make_vggt_mock()
    tp_mod.predict_tracks = mock_predict

    with patch.dict(
        sys.modules,
        {
            "vggt": vggt_mod,
            "vggt.dependency": vggt_mod.dependency,
            "vggt.dependency.track_predict": tp_mod,
        },
    ):
        # Re-import to pick up mocked module
        import importlib
        import collab_splats.pointcloud.bundle_adjustment as ba_mod
        importlib.reload(ba_mod)

        tracks, vis_scores, pts3d = ba_mod.extract_tracks_vggsfm(
            images,
            conf=conf,
            world_points=None,
            max_query_pts=512,
            query_frame_num=2,
        )

    assert tracks.shape == (N, P, 2), f"expected ({N},{P},2), got {tracks.shape}"
    assert vis_scores.shape == (N, P), f"expected ({N},{P}), got {vis_scores.shape}"
    assert pts3d.shape == (P, 3), f"expected ({P},3), got {pts3d.shape}"
    assert tracks.dtype == np.float32
    assert vis_scores.dtype == np.float32
    assert pts3d.dtype == np.float32

    # Ensure predict_tracks was called with the right images tensor
    mock_predict.assert_called_once()
    call_kwargs = mock_predict.call_args
    assert call_kwargs[0][0] is images


def test_extract_tracks_vggsfm_conf_4d():
    """4-D conf (N,1,H,W) should be squeezed to (N,H,W) before passing."""
    N, H, W = 4, 32, 32
    P = 10
    images = torch.zeros(N, 3, H, W)
    conf_4d = torch.ones(N, 1, H, W)

    mock_predict = MagicMock(
        return_value=(
            np.random.rand(N, P, 2).astype(np.float32),
            np.random.rand(N, P).astype(np.float32),
            np.ones((N, P), dtype=np.float32),
            np.random.rand(P, 3).astype(np.float32),
            np.ones((P, 3), dtype=np.float32),
        )
    )

    vggt_mod, tp_mod, _ = _make_vggt_mock()
    tp_mod.predict_tracks = mock_predict

    with patch.dict(
        sys.modules,
        {
            "vggt": vggt_mod,
            "vggt.dependency": vggt_mod.dependency,
            "vggt.dependency.track_predict": tp_mod,
        },
    ):
        import importlib
        import collab_splats.pointcloud.bundle_adjustment as ba_mod
        importlib.reload(ba_mod)

        tracks, vis_scores, pts3d = ba_mod.extract_tracks_vggsfm(
            images, conf=conf_4d, world_points=None
        )

    # Verify that conf passed to predict_tracks has shape (N,H,W), not (N,1,H,W)
    passed_conf = mock_predict.call_args[1]["conf"]
    assert passed_conf.shape == (N, H, W), f"conf should be squeezed, got {passed_conf.shape}"


# ---------------------------------------------------------------------------
# Test 2: run_bundle_adjustment — synthetic smoke test (no GPU / bae needed;
# we mock bae entirely and verify shapes + no exception).
# ---------------------------------------------------------------------------

def _pypose_available() -> bool:
    try:
        import pypose  # noqa: F401
        return True
    except ImportError:
        return False


def _build_synthetic_scene(N=4, P=50, H=128, W=128, seed=42):
    """Create a simple synthetic scene for BA smoke tests."""
    rng = np.random.default_rng(seed)

    points3d = rng.standard_normal((P, 3)).astype(np.float64)

    # Simple cameras looking along +Z at the origin
    extrinsics = np.zeros((N, 3, 4), dtype=np.float32)
    for i in range(N):
        extrinsics[i, :3, :3] = np.eye(3)
        extrinsics[i, :3, 3] = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 2.0]

    f = 100.0
    cx, cy = W / 2.0, H / 2.0
    intrinsics = np.tile(
        np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )

    # Project points to 2D (trivial — just use K @ (R @ p + t) / z)
    tracks = np.zeros((N, P, 2), dtype=np.float32)
    vis_mask = np.ones((N, P), dtype=bool)
    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        pts_cam = (R @ points3d.T).T + t  # (P, 3)
        z = pts_cam[:, 2]
        u = f * pts_cam[:, 0] / z + cx
        v = f * pts_cam[:, 1] / z + cy
        tracks[i, :, 0] = u
        tracks[i, :, 1] = v
        vis_mask[i] = z > 0.1

    return points3d, extrinsics, intrinsics, tracks, vis_mask


@pytest.mark.skipif(not _pypose_available(), reason="requires pypose (downgrade bae to 0.2)")
def test_run_bundle_adjustment_early_exit_shape():
    """Verify run_bundle_adjustment returns correct shapes when inlier count is below threshold (early-exit path)."""
    N, P, H, W = 4, 50, 128, 128
    points3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    # We need at least 64 inliers per frame and >= 2 observations per point.
    # With P=50 and N=4 that falls below the 64-inlier threshold in the filter step,
    # so we expect the function to return early (unchanged arrays) rather than crash.
    ref_pts, ref_ext, ref_intr = None, None, None

    vggt_mod, _, proj_mod = _make_vggt_mock()
    # project_3D_points_np returns (N, P, 2) projections + dummy cam points
    proj_mod.project_3D_points_np = MagicMock(return_value=(tracks.copy(), None))

    bae_mod = _make_bae_mock()

    extra_mods = {
        "vggt": vggt_mod,
        "vggt.dependency": vggt_mod.dependency,
        "vggt.dependency.projection": proj_mod,
        "bae": bae_mod,
        "bae.autograd": bae_mod.autograd,
        "bae.autograd.function": bae_mod.autograd.function,
        "bae.utils": bae_mod.utils,
        "bae.utils.pysolvers": bae_mod.utils.pysolvers,
        "bae.utils.ba": bae_mod.utils.ba,
        "bae.optim": bae_mod.optim,
    }

    with patch.dict(sys.modules, extra_mods):
        import importlib
        import collab_splats.pointcloud.bundle_adjustment as ba_mod
        importlib.reload(ba_mod)

        ref_pts, ref_ext, ref_intr = ba_mod.run_bundle_adjustment(
            points3d=points3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            tracks=tracks,
            vis_mask=vis_mask,
            image_size=(H, W),
            max_reproj_error=4.0,
            lm_steps=5,
        )

    assert ref_pts.shape == (P, 3), f"points3d shape wrong: {ref_pts.shape}"
    assert ref_ext.shape == (N, 3, 4), f"extrinsics shape wrong: {ref_ext.shape}"
    assert ref_intr.shape == (N, 3, 3), f"intrinsics shape wrong: {ref_intr.shape}"


def _cuda_and_bae_available() -> bool:
    try:
        import torch  # noqa: F401

        if not torch.cuda.is_available():
            return False
        import pypose  # noqa: F401
        import bae  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cuda_and_bae_available(), reason="requires CUDA, pypose, and bae")
def test_run_bundle_adjustment_reduces_reproj_error():
    """With noisy initial poses and clean 2D observations, BA must reduce reprojection error."""
    from collab_splats.pointcloud.bundle_adjustment import run_bundle_adjustment

    rng = np.random.default_rng(0)
    N, P, H, W = 5, 200, 256, 256
    f = 200.0

    # Clean 3D points in front of cameras
    points3d = rng.uniform(-1, 1, (P, 3)).astype(np.float64)
    points3d[:, 2] += 3.0  # z > 0 in front of all cameras

    # Clean extrinsics: identity R, small random t
    extrinsics_clean = np.zeros((N, 3, 4), dtype=np.float32)
    for i in range(N):
        extrinsics_clean[i, :3, :3] = np.eye(3)
        extrinsics_clean[i, :3, 3] = rng.uniform(-0.3, 0.3, 3).astype(np.float32)

    intrinsics = np.tile(
        np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )

    # Project with clean poses → ground-truth 2D observations
    tracks = np.zeros((N, P, 2), dtype=np.float32)
    vis_mask = np.ones((N, P), dtype=bool)
    for i in range(N):
        R, t = extrinsics_clean[i, :3, :3], extrinsics_clean[i, :3, 3]
        pts_cam = (R @ points3d.T).T + t
        z = pts_cam[:, 2]
        tracks[i, :, 0] = f * pts_cam[:, 0] / z + W / 2
        tracks[i, :, 1] = f * pts_cam[:, 1] / z + H / 2
        vis_mask[i] = (
            (z > 0.1)
            & (tracks[i, :, 0] >= 0)
            & (tracks[i, :, 0] < W)
            & (tracks[i, :, 1] >= 0)
            & (tracks[i, :, 1] < H)
        )

    # Perturb extrinsics with Gaussian translation noise
    extrinsics_noisy = extrinsics_clean.copy()
    for i in range(N):
        extrinsics_noisy[i, :3, 3] += rng.normal(0, 0.1, 3).astype(np.float32)

    def mean_reproj_error(ext: np.ndarray) -> float:
        errs = []
        for i in range(N):
            R, t = ext[i, :3, :3], ext[i, :3, 3]
            pts_cam = (R @ points3d.T).T + t
            z = pts_cam[:, 2]
            px = f * pts_cam[:, 0] / z + W / 2
            py = f * pts_cam[:, 1] / z + H / 2
            proj = np.stack([px, py], axis=-1)
            mask = vis_mask[i]
            errs.append(np.linalg.norm(proj[mask] - tracks[i][mask], axis=-1).mean())
        return float(np.mean(errs))

    err_before = mean_reproj_error(extrinsics_noisy)

    _, ext_out, _ = run_bundle_adjustment(
        points3d.copy(),
        extrinsics_noisy,
        intrinsics,
        tracks,
        vis_mask,
        image_size=(H, W),
        max_reproj_error=None,
        lm_steps=20,
    )

    err_after = mean_reproj_error(ext_out)
    assert err_after < err_before, (
        f"BA did not reduce reprojection error: {err_before:.4f} → {err_after:.4f}"
    )


@pytest.mark.skipif(not _pypose_available(), reason="requires pypose (downgrade bae to 0.2)")
def test_run_bundle_adjustment_no_reproj_filter():
    """Passing max_reproj_error=None skips reprojection filtering."""
    N, P, H, W = 4, 50, 128, 128
    points3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    vggt_mod, _, proj_mod = _make_vggt_mock()
    bae_mod = _make_bae_mock()

    extra_mods = {
        "vggt": vggt_mod,
        "vggt.dependency": vggt_mod.dependency,
        "vggt.dependency.projection": proj_mod,
        "bae": bae_mod,
        "bae.autograd": bae_mod.autograd,
        "bae.autograd.function": bae_mod.autograd.function,
        "bae.utils": bae_mod.utils,
        "bae.utils.pysolvers": bae_mod.utils.pysolvers,
        "bae.utils.ba": bae_mod.utils.ba,
        "bae.optim": bae_mod.optim,
    }

    with patch.dict(sys.modules, extra_mods):
        import importlib
        import collab_splats.pointcloud.bundle_adjustment as ba_mod
        importlib.reload(ba_mod)

        ref_pts, ref_ext, ref_intr = ba_mod.run_bundle_adjustment(
            points3d=points3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            tracks=tracks,
            vis_mask=vis_mask,
            image_size=(H, W),
            max_reproj_error=None,  # skip filter
        )

    # project_3D_points_np should NOT have been called
    proj_mod.project_3D_points_np.assert_not_called()

    assert ref_pts.shape == (P, 3)
    assert ref_ext.shape == (N, 3, 4)
    assert ref_intr.shape == (N, 3, 3)
