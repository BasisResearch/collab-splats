"""Tests for collab_splats.pointcloud.bundle_adjustment.

bae and vggt may not be installed in CI; all heavy imports are mocked.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
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
    bae.utils = utils
    bae.utils.pysolvers = utils_py

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

        tracks, vis_scores, pts3d = ba_mod._extract_tracks_vggsfm(
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

    # Ensure predict_tracks was called once with images on the resolved target device
    mock_predict.assert_called_once()
    called_images = mock_predict.call_args[0][0]
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert called_images.shape == images.shape, (
        f"images shape mismatch: {called_images.shape} != {images.shape}"
    )
    assert called_images.device.type == expected_device, (
        f"images device {called_images.device.type!r} != target_device {expected_device!r}"
    )


def test_extract_tracks_vggsfm_tensor_images_reach_target_device():
    """Tensor images (not numpy) must be moved to target_device before predict_tracks.

    Regression guard: the isinstance(np.ndarray) guard previously meant CPU torch.Tensor
    inputs bypassed .to(target_device). predict_tracks uses images.device for tracker
    placement — wrong device means the whole tracker runs on CPU even when CUDA is available.
    """
    N, H, W = 2, 8, 8
    # CPU torch.Tensor — NOT numpy; exercises the non-numpy code path
    images_cpu = torch.zeros(N, 3, H, W)

    # Capture the device of images as seen inside predict_tracks
    received_device: list[str] = []

    def fake_predict(imgs, conf=None, points_3d=None, **kw):
        received_device.append(str(imgs.device))
        P = 4
        return (
            np.zeros((N, P, 2), dtype=np.float32),
            np.zeros((N, P), dtype=np.float32),
            np.zeros((N, P), dtype=np.float32),
            np.zeros((P, 3), dtype=np.float32),
            np.zeros((P, 3), dtype=np.float32),
        )

    with patch(
        "collab_splats.pointcloud.bundle_adjustment.predict_tracks",
        side_effect=fake_predict,
    ):
        from collab_splats.pointcloud.bundle_adjustment import _extract_tracks_vggsfm
        _extract_tracks_vggsfm(images_cpu, conf=None, world_points=None, device="cpu")

    assert len(received_device) == 1, "predict_tracks must be called exactly once"
    # After fix: images.device always matches target_device regardless of input type.
    # For CUDA correctness the real regression is when target_device='cuda' and images are CPU;
    # that scenario requires a GPU — this guard covers the CPU→CPU contract.
    assert received_device[0] == "cpu", (
        f"images.device={received_device[0]!r} != target_device='cpu'; "
        "tensor images are not being relocated to target_device"
    )


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

        tracks, vis_scores, pts3d = ba_mod._extract_tracks_vggsfm(
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


@pytest.mark.skipif(not _pypose_available(), reason="requires pypose")
def test_optimize_early_exit_shape():
    """_optimize returns correct shapes when inlier count is below threshold (early-exit path)."""
    N, P, H, W = 4, 50, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    vggt_mod, _, proj_mod = _make_vggt_mock()
    proj_cam = np.ones((N, 3, P), dtype=np.float32)
    proj_mod.project_3D_points_np = MagicMock(return_value=(tracks.copy(), proj_cam))
    bae_mod = _make_bae_mock()

    extra_mods = {
        "vggt": vggt_mod,
        "vggt.dependency": vggt_mod.dependency,
        "vggt.dependency.projection": proj_mod,
        "vggt.dependency.track_predict": vggt_mod.dependency.track_predict,
        "bae": bae_mod,
        "bae.autograd": bae_mod.autograd,
        "bae.autograd.function": bae_mod.autograd.function,
        "bae.utils": bae_mod.utils,
        "bae.utils.pysolvers": bae_mod.utils.pysolvers,
        "bae.optim": bae_mod.optim,
    }

    with patch.dict(sys.modules, extra_mods):
        import importlib.util
        _ba_path = Path(__file__).parents[2] / "collab_splats" / "pointcloud" / "bundle_adjustment.py"
        _mod_name = "collab_splats.pointcloud.bundle_adjustment"
        spec = importlib.util.spec_from_file_location(_mod_name, _ba_path)
        ba_mod = importlib.util.module_from_spec(spec)
        sys.modules[_mod_name] = ba_mod
        spec.loader.exec_module(ba_mod)

        ba = ba_mod.BundleAdjustment()
        ref_pts, ref_ext, ref_intr = ba._optimize(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            tracks=tracks,
            vis_scores=vis_mask.astype(np.float32),
            max_reproj_error=4.0,
        )

    assert ref_pts.shape == (P, 3)
    assert ref_ext.shape == (N, 3, 4)
    assert ref_intr.shape == (N, 3, 3)


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
def test_optimize_reduces_reproj_error():
    """With noisy initial poses and clean 2D observations, _optimize must reduce reprojection error."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment

    rng = np.random.default_rng(0)
    N, P, H, W = 5, 200, 256, 256
    f = 200.0

    points3d = rng.uniform(-1, 1, (P, 3)).astype(np.float64)
    points3d[:, 2] += 3.0

    extrinsics_clean = np.zeros((N, 3, 4), dtype=np.float32)
    for i in range(N):
        extrinsics_clean[i, :3, :3] = np.eye(3)
        extrinsics_clean[i, :3, 3] = rng.uniform(-0.3, 0.3, 3).astype(np.float32)

    intrinsics = np.tile(
        np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )

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

    extrinsics_noisy = extrinsics_clean.copy()
    for i in range(N):
        extrinsics_noisy[i, :3, 3] += rng.normal(0, 0.1, 3).astype(np.float32)

    def mean_reproj_error(ext):
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

    ba = BundleAdjustment()
    _, ext_out, _ = ba._optimize(
        points3d.copy(),
        extrinsics_noisy,
        intrinsics,
        tracks,
        vis_mask.astype(np.float32),
        max_reproj_error=None,
        lm_steps=20,
    )

    err_after = mean_reproj_error(ext_out)
    assert err_after < err_before, f"BA did not reduce error: {err_before:.4f} → {err_after:.4f}"


@pytest.mark.skipif(not _pypose_available(), reason="requires pypose")
def test_optimize_no_reproj_filter():
    """Passing max_reproj_error=None skips reprojection filtering."""
    N, P, H, W = 4, 50, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    vggt_mod, _, proj_mod = _make_vggt_mock()
    bae_mod = _make_bae_mock()

    extra_mods = {
        "vggt": vggt_mod,
        "vggt.dependency": vggt_mod.dependency,
        "vggt.dependency.projection": proj_mod,
        "vggt.dependency.track_predict": vggt_mod.dependency.track_predict,
        "bae": bae_mod,
        "bae.autograd": bae_mod.autograd,
        "bae.autograd.function": bae_mod.autograd.function,
        "bae.utils": bae_mod.utils,
        "bae.utils.pysolvers": bae_mod.utils.pysolvers,
        "bae.optim": bae_mod.optim,
    }

    with patch.dict(sys.modules, extra_mods):
        import importlib.util
        _ba_path = Path(__file__).parents[2] / "collab_splats" / "pointcloud" / "bundle_adjustment.py"
        _mod_name = "collab_splats.pointcloud.bundle_adjustment"
        spec = importlib.util.spec_from_file_location(_mod_name, _ba_path)
        ba_mod = importlib.util.module_from_spec(spec)
        sys.modules[_mod_name] = ba_mod
        spec.loader.exec_module(ba_mod)

        ba = ba_mod.BundleAdjustment()
        ref_pts, ref_ext, ref_intr = ba._optimize(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            tracks=tracks,
            vis_scores=vis_mask.astype(np.float32),
            max_reproj_error=None,
        )

    proj_mod.project_3D_points_np.assert_not_called()
    assert ref_pts.shape == (P, 3)
    assert ref_ext.shape == (N, 3, 4)
    assert ref_intr.shape == (N, 3, 3)


# ---------------------------------------------------------------------------
# Tests for _get_default_solver — solver auto-selection logic
# ---------------------------------------------------------------------------

def _make_ba_mods(extra=None):
    """Build sys.modules patch dict for loading bundle_adjustment under mocks."""
    bae_mod = _make_bae_mock()
    vggt_mod, _, proj_mod = _make_vggt_mock()
    mods = {
        "vggt": vggt_mod,
        "vggt.dependency": vggt_mod.dependency,
        "vggt.dependency.projection": proj_mod,
        "vggt.dependency.track_predict": vggt_mod.dependency.track_predict,
        "bae": bae_mod,
        "bae.autograd": bae_mod.autograd,
        "bae.autograd.function": bae_mod.autograd.function,
        "bae.utils": bae_mod.utils,
        "bae.utils.pysolvers": bae_mod.utils.pysolvers,
        "bae.optim": bae_mod.optim,
    }
    if extra:
        mods.update(extra)
    return mods


def _load_ba(mods):
    """Exec bundle_adjustment.py under the given sys.modules patch and return the module.

    Must be called inside a patch.dict(sys.modules, mods) context.
    Registers the module in sys.modules before exec so @dataclass can find its own module.
    """
    import importlib.util
    from pathlib import Path
    _ba_path = Path(__file__).parents[2] / "collab_splats" / "pointcloud" / "bundle_adjustment.py"
    _mod_name = "collab_splats.pointcloud.bundle_adjustment"
    spec = importlib.util.spec_from_file_location(_mod_name, _ba_path)
    ba_mod = importlib.util.module_from_spec(spec)
    sys.modules[_mod_name] = ba_mod
    spec.loader.exec_module(ba_mod)
    return ba_mod


def test_get_default_solver_prefers_cudss_when_available():
    """CuDSS instantiated (and wrapped) when CUDA available and bae.sparse.solve importable."""
    mock_cudss_instance = MagicMock(name="cudss_instance")
    mock_cudss_class = MagicMock(name="CuDirectSparseSolver", return_value=mock_cudss_instance)
    mock_solve_mod = types.ModuleType("bae.sparse.solve")
    mock_solve_mod.CuDirectSparseSolver = mock_cudss_class

    mods = _make_ba_mods({"bae.sparse.solve": mock_solve_mod})
    with patch.dict(sys.modules, mods), patch("torch.cuda.is_available", return_value=True):
        ba_mod = _load_ba(mods)
        solver = ba_mod._get_default_solver()

    mock_cudss_class.assert_called_once()
    assert solver is mock_cudss_instance


def test_get_default_solver_falls_back_to_pcg_no_cuda():
    """PCG instantiated when CUDA unavailable."""
    mods = _make_ba_mods()
    mock_pcg_instance = MagicMock(name="pcg_instance")
    mock_pcg_class = MagicMock(name="PCG", return_value=mock_pcg_instance)
    mods["bae.utils.pysolvers"].PCG = mock_pcg_class

    with patch.dict(sys.modules, mods), patch("torch.cuda.is_available", return_value=False):
        ba_mod = _load_ba(mods)
        solver = ba_mod._get_default_solver()

    mock_pcg_class.assert_called_once()
    assert solver is mock_pcg_instance


def test_get_default_solver_cpu_device_forces_pcg():
    """device='cpu' must always return PCG, even when CUDA is available."""
    mods = _make_ba_mods()
    mock_pcg_instance = MagicMock(name="pcg_instance")
    mock_pcg_class = MagicMock(name="PCG", return_value=mock_pcg_instance)
    mods["bae.utils.pysolvers"].PCG = mock_pcg_class

    with patch.dict(sys.modules, mods), patch("torch.cuda.is_available", return_value=True):
        ba_mod = _load_ba(mods)
        solver = ba_mod._get_default_solver(device="cpu")

    mock_pcg_class.assert_called_once()
    assert solver is mock_pcg_instance


def test_get_default_solver_falls_back_to_pcg_cudss_import_error():
    """PCG instantiated when CUDA available but bae.sparse.solve raises ImportError."""
    # Set bae.sparse.solve to None — Python treats None entries as blocked imports,
    # raising ImportError. Without this, a prior test's real import may be cached.
    mods = _make_ba_mods({"bae.sparse.solve": None})
    mock_pcg_instance = MagicMock(name="pcg_instance")
    mock_pcg_class = MagicMock(name="PCG", return_value=mock_pcg_instance)
    mods["bae.utils.pysolvers"].PCG = mock_pcg_class

    with patch.dict(sys.modules, mods), patch("torch.cuda.is_available", return_value=True):
        ba_mod = _load_ba(mods)
        solver = ba_mod._get_default_solver()

    mock_pcg_class.assert_called_once()
    assert solver is mock_pcg_instance


# ---------------------------------------------------------------------------
# Tests for BundleAdjustment class
# ---------------------------------------------------------------------------

def _make_ff_result_for_ba(N=2, H=8, W=8):
    """Minimal FeedforwardResult for BundleAdjustment tests."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    return FeedforwardResult(
        points=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        image_paths=[Path(f"img{i}.jpg") for i in range(N)],
        original_coords=None,
        model_width=W,
        model_height=H,
        images=torch.zeros(N, 3, H, W),
        confidence=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )


def test_bundle_adjustment_refine_returns_feedforward_result():
    """refine() must return a FeedforwardResult with updated extrinsics/intrinsics."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))), \
         patch.object(BundleAdjustment, "_optimize",
                      return_value=(np.zeros((5, 3)), refined_ext, refined_intr)):
        out = BundleAdjustment().refine(result)

    assert isinstance(out, FeedforwardResult)
    assert out.extrinsics.shape == (N, 4, 4)
    assert out.intrinsics.shape == (N, 3, 3)


def test_bundle_adjustment_refine_preserves_pts3d_colors():
    """refine() must not change points, colors, or pixel_indices."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    original_points = result.points.copy()
    original_colors = result.colors.copy()
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))), \
         patch.object(BundleAdjustment, "_optimize",
                      return_value=(np.zeros((5, 3)), refined_ext, refined_intr)):
        out = BundleAdjustment().refine(result)

    np.testing.assert_array_equal(out.points, original_points)
    np.testing.assert_array_equal(out.colors, original_colors)
    assert out.pixel_indices is None


def test_bundle_adjustment_refine_threads_config():
    """Config params and device are passed through to both private functions."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))) as mock_tracks, \
         patch.object(BundleAdjustment, "_optimize",
                      return_value=(np.zeros((5, 3)), refined_ext, refined_intr)) as mock_opt:
        cfg = BundleAdjustmentConfig(device="cuda:1", lm_steps=5, max_reproj_error=2.0)
        BundleAdjustment(config=cfg).refine(result)

    _, tracks_kw = mock_tracks.call_args
    assert tracks_kw["device"] == "cuda:1"
    mock_opt.assert_called_once()


@pytest.mark.skipif(not _pypose_available(), reason="requires pypose")
def test_optimize_rejects_cpu_device():
    """_optimize must raise a clear error for a non-CUDA device — bae LM is CUDA-only."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    # min_inliers_per_frame lowered so frames survive the filter and reach the device check
    ba = BundleAdjustment(config=BundleAdjustmentConfig(device="cpu", min_inliers_per_frame=10))
    with pytest.raises(RuntimeError, match="CUDA"):
        ba._optimize(
            pts3d, extrinsics, intrinsics,
            tracks, vis_mask.astype(np.float32),
            max_reproj_error=None,
        )


def test_ba_config_new_fields_default():
    """BundleAdjustmentConfig has increment_size=0 and tracks_cache_dir=None by default."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig
    cfg = BundleAdjustmentConfig()
    assert cfg.increment_size == 0
    assert cfg.tracks_cache_dir is None


def test_bundle_adjustment_default_config():
    """BundleAdjustment() with no args uses default BundleAdjustmentConfig."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
    ba = BundleAdjustment()
    assert isinstance(ba.config, BundleAdjustmentConfig)
    assert ba.config.device is None
    assert ba.config.lm_steps == 40
    # capture_loss_history defaults to False; _last_loss_history always starts empty
    assert ba.config.capture_loss_history is False
    assert ba._last_loss_history == []


@pytest.mark.skipif(not _cuda_and_bae_available(), reason="requires CUDA, pypose, and bae")
def test_optimize_captures_loss_history_when_flag_set():
    """_optimize appends one inner list to _last_loss_history when capture_loss_history=True."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    n_steps = 5
    cfg = BundleAdjustmentConfig(capture_loss_history=True, lm_steps=n_steps, min_inliers_per_frame=10)
    ba = BundleAdjustment(config=cfg)
    assert ba._last_loss_history == []

    ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), max_reproj_error=None)

    hist = ba._last_loss_history
    assert isinstance(hist, list)
    assert len(hist) == 1, f"one _optimize call → one inner list; got {len(hist)}"
    assert len(hist[0]) == n_steps
    assert all(isinstance(v, float) for v in hist[0])
    assert all(v >= 0 for v in hist[0])


@pytest.mark.skipif(not _cuda_and_bae_available(), reason="requires CUDA, pypose, and bae")
def test_optimize_no_loss_history_by_default():
    """Without capture_loss_history, _last_loss_history stays empty after _optimize."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    ba = BundleAdjustment(config=BundleAdjustmentConfig(min_inliers_per_frame=10))
    ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), max_reproj_error=None)
    assert ba._last_loss_history == []


def test_tracks_cache_save_load(tmp_path):
    """_load_or_extract_tracks saves to zarr; second call returns cached arrays without extracting."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_tracks = np.ones((N, 5, 2), dtype=np.float32)
    fake_vis = np.ones((N, 5), dtype=np.float32) * 0.9
    fake_pts3d = np.ones((5, 3), dtype=np.float32) * 2.0

    cfg = BundleAdjustmentConfig(tracks_cache_dir=tmp_path)
    ba = BundleAdjustment(config=cfg)

    extract_calls = []

    def fake_extract(images, confidence, world_points, max_query_pts, query_frame_num, device=None):
        extract_calls.append(1)
        return fake_tracks, fake_vis, fake_pts3d

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        t1, v1, p1 = ba._load_or_extract_tracks(result)
        t2, v2, p2 = ba._load_or_extract_tracks(result)

    assert len(extract_calls) == 1, "second call should use cache, not re-extract"
    np.testing.assert_array_equal(t1, fake_tracks)
    np.testing.assert_array_equal(t2, fake_tracks)
    np.testing.assert_array_equal(p1, fake_pts3d)
    np.testing.assert_array_equal(p2, fake_pts3d)


def test_tracks_cache_invalidates_on_config_change(tmp_path):
    """Cache is invalidated when query_frame_num changes; extraction runs again."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_a = (np.ones((N, 5, 2), dtype=np.float32),
              np.ones((N, 5), dtype=np.float32),
              np.ones((5, 3), dtype=np.float32))
    fake_b = (np.zeros((N, 5, 2), dtype=np.float32),
              np.zeros((N, 5), dtype=np.float32),
              np.zeros((5, 3), dtype=np.float32))
    extractions = [fake_a, fake_b]

    def fake_extract(*args, **kwargs):
        return extractions.pop(0)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        ba1 = BundleAdjustment(config=BundleAdjustmentConfig(query_frame_num=5, tracks_cache_dir=tmp_path))
        ba1._load_or_extract_tracks(result)

        # Change query_frame_num — different key → cache miss
        ba2 = BundleAdjustment(config=BundleAdjustmentConfig(query_frame_num=10, tracks_cache_dir=tmp_path))
        t2, _, _ = ba2._load_or_extract_tracks(result)

    np.testing.assert_array_equal(t2, fake_b[0])


def test_incremental_ba_increment_size_n_matches_allonce():
    """increment_size >= N dispatches to all-at-once path: _optimize called exactly once with all N frames."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 4, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    P = 5
    fake_tracks = np.zeros((N, P, 2), dtype=np.float32)
    fake_vis = np.ones((N, P), dtype=np.float32)
    fake_pts3d = np.zeros((P, 3), dtype=np.float32)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    optimize_frame_counts = []

    def mock_optimize(pts3d, extrinsics, intrinsics, tracks, vis_scores, **kwargs):
        optimize_frame_counts.append(len(tracks))
        return (fake_pts3d, refined_ext[:len(tracks)], refined_intr[:len(tracks)])

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(fake_tracks, fake_vis, fake_pts3d)), \
         patch.object(BundleAdjustment, "_optimize", side_effect=mock_optimize):

        BundleAdjustment(BundleAdjustmentConfig(increment_size=0)).refine(result)
        BundleAdjustment(BundleAdjustmentConfig(increment_size=N)).refine(result)
        BundleAdjustment(BundleAdjustmentConfig(increment_size=N + 10)).refine(result)

    assert optimize_frame_counts == [N, N, N], (
        f"increment_size=0/N/N+10 should all call _optimize once with N frames; got {optimize_frame_counts}"
    )


def test_incremental_ba_warm_start_updates_registered_frames():
    """_refine_incremental updates extrinsics[:k] after each step (warm start propagates)."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 6, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    P = 5
    fake_tracks = np.zeros((N, P, 2), dtype=np.float32)
    fake_vis = np.ones((N, P), dtype=np.float32)
    fake_pts3d = np.zeros((P, 3), dtype=np.float32)

    step_counter = [0]
    received_extrinsics = []

    def mock_optimize(pts3d, extrinsics, intrinsics, tracks, vis_scores, **kwargs):
        k = len(tracks)
        received_extrinsics.append(extrinsics.copy())
        refined = extrinsics.copy()
        refined[:, 0, 0] += float(step_counter[0] + 1)
        step_counter[0] += 1
        return (fake_pts3d, refined, intrinsics.copy())

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(fake_tracks, fake_vis, fake_pts3d)), \
         patch.object(BundleAdjustment, "_optimize", side_effect=mock_optimize):

        ba = BundleAdjustment(BundleAdjustmentConfig(increment_size=2))
        ba.refine(result)

    # N=6, increment_size=2 → steps k=2,4,6 → 3 _optimize calls
    assert len(received_extrinsics) == 3, f"expected 3 steps for N=6 increment_size=2, got {len(received_extrinsics)}"

    # Warm start: step-2 extrinsics[:2] should be step-1 refined output (diagonal+1), not original
    assert received_extrinsics[1][:2, 0, 0].mean() > 1.0, (
        "warm start failed: step-2 extrinsics[:2] should be step-1 refined output, not original feedforward"
    )


def test_incremental_ba_loss_history_has_one_entry_per_step():
    """_last_loss_history contains one inner list per k-step when capture_loss_history=True."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 6, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    P = 5
    fake_tracks = np.zeros((N, P, 2), dtype=np.float32)
    fake_vis = np.ones((N, P), dtype=np.float32)
    fake_pts3d = np.zeros((P, 3), dtype=np.float32)

    def mock_optimize_with_hist(self_ba, pts3d, extrinsics, intrinsics, tracks, vis_scores, **kwargs):
        k = len(tracks)
        if self_ba.config.capture_loss_history:
            self_ba._last_loss_history.append([float(k) * 0.1])
        return (fake_pts3d, extrinsics.copy(), intrinsics.copy())

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(fake_tracks, fake_vis, fake_pts3d)), \
         patch.object(BundleAdjustment, "_optimize",
                      lambda self_ba, *a, **kw: mock_optimize_with_hist(self_ba, *a, **kw)):

        ba = BundleAdjustment(BundleAdjustmentConfig(increment_size=2, capture_loss_history=True))
        ba.refine(result)

    # N=6, increment_size=2 → steps k=2,4,6 → 3 _optimize calls → 3 inner lists
    assert len(ba._last_loss_history) == 3, (
        f"expected 3 inner lists for 3 steps; got {len(ba._last_loss_history)}"
    )
    assert all(isinstance(entry, list) for entry in ba._last_loss_history)
