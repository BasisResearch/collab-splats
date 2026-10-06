"""
Tests for collab_splats.geometry.bundle_adjustment.

- module skipped without bae or pypose; bundle_adjustment imports both at load
- selected tests swap vggt for a mock via patch.dict(sys.modules) and reload
- CUDA-dependent solves are skipped without CUDA
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# BA solver deps: bundle_adjustment imports them at load, so their absence skips the module
pp = pytest.importorskip("pypose")
bae_graph = pytest.importorskip("bae.autograd.graph")
bae_optim = pytest.importorskip("bae.optim")

from collab_splats.geometry import bundle_adjustment
from collab_splats.geometry.bundle_adjustment import (
    BundleAdjustment,
    BundleAdjustmentConfig,
    _align_to_input_poses,
    _BAModel,
    _filter_observations,
    _lm_step,
)
from collab_splats.geometry.photometric import photometric_samples
from collab_splats.geometry.projection import project

# Reprojection-only solve: these tests exercise the track path, which takes no depth
REPROJ_ONLY = {"use_photometric": False, "use_depth": False}


def _refine(ba, result):
    """Old-style call: refine arrays, return the result with refined cameras."""
    ext, K = ba.refine(
        result.images, result.confidence, result.world_points, result.extrinsics, result.intrinsics, result.image_paths
    )
    return replace(result, extrinsics=ext, intrinsics=K)


def _tracks(ba, result):
    """Track load with the result's arrays."""
    return ba.extract_tracks(result.images, result.confidence, result.world_points, result.image_paths)


########################################################################
# Helpers — mock vggt package tree that tests swap into sys.modules
########################################################################


def _make_vggt_mock():
    """Return a mock package tree for vggt.dependency.track_predict / projection."""
    vggt = types.ModuleType("vggt")
    vggt.dependency = types.ModuleType("vggt.dependency")
    tp = types.ModuleType("vggt.dependency.track_predict")
    tp.predict_tracks = MagicMock()
    proj = types.ModuleType("vggt.dependency.projection")
    vggt.dependency.track_predict = tp
    vggt.dependency.projection = proj
    return vggt, tp, proj


########################################################################
# Test 1: extract_tracks_vggsfm — shape correctness with mocked predict_tracks
########################################################################


def test_extract_tracks_shape():
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

    mock_predict = MagicMock(return_value=(tracks_arr, vis_arr, confs_arr, pts3d_arr, colors_arr))

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

        import collab_splats.geometry.bundle_adjustment as ba_mod

        importlib.reload(ba_mod)

        tracks, vis_scores, pts3d = ba_mod.extract_tracks_vggsfm(
            images,
            conf=conf,
            world_points=None,
            cfg=ba_mod.BundleAdjustmentConfig(max_query_pts=512, query_frame_num=2, device=None),
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
    assert called_images.shape == images.shape, f"images shape mismatch: {called_images.shape} != {images.shape}"
    assert (
        called_images.device.type == expected_device
    ), f"images device {called_images.device.type!r} != target_device {expected_device!r}"


def test_extract_tracks_tensor_images_reach_target_device():
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
        "collab_splats.geometry.bundle_adjustment.predict_tracks",
        side_effect=fake_predict,
    ):
        from collab_splats.geometry.bundle_adjustment import (
            BundleAdjustmentConfig,
            extract_tracks_vggsfm,
        )

        extract_tracks_vggsfm(
            images_cpu,
            conf=None,
            world_points=None,
            cfg=BundleAdjustmentConfig(max_query_pts=2048, query_frame_num=5, device="cpu"),
        )

    assert len(received_device) == 1, "predict_tracks must be called exactly once"
    # After fix: images.device always matches target_device regardless of input type.
    # For CUDA correctness the real regression is when target_device='cuda' and images are CPU;
    # that scenario requires a GPU — this guard covers the CPU→CPU contract.
    assert received_device[0] == "cpu", (
        f"images.device={received_device[0]!r} != target_device='cpu'; "
        "tensor images are not being relocated to target_device"
    )


def test_extract_tracks_conf_4d():
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

        import collab_splats.geometry.bundle_adjustment as ba_mod

        importlib.reload(ba_mod)

        tracks, vis_scores, pts3d = ba_mod.extract_tracks_vggsfm(
            images,
            conf=conf_4d,
            world_points=None,
            cfg=ba_mod.BundleAdjustmentConfig(max_query_pts=2048, query_frame_num=5, device=None),
        )

    # Verify that conf passed to predict_tracks has shape (N,H,W), not (N,1,H,W)
    passed_conf = mock_predict.call_args[1]["conf"]
    assert passed_conf.shape == (N, H, W), f"conf should be squeezed, got {passed_conf.shape}"


########################################################################
# Test 2: BundleAdjustment.refine — synthetic smoke tests on real bae, CUDA-gated where they solve
########################################################################


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
    intrinsics = np.tile(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32), (N, 1, 1))

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


def test_optimize_raises_below_inlier_threshold():
    """_optimize raises when every frame falls below the inlier threshold after the reproj filter."""
    N, P, H, W = 4, 50, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    ba = BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=4.0, **REPROJ_ONLY))

    with (
        patch.object(bundle_adjustment, "project", wraps=bundle_adjustment.project) as project_spy,
        pytest.raises(ValueError, match="too few active frames/points"),
    ):
        ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), None, None, None)

    # The reprojection filter projects all gated pairs in one device batch
    assert project_spy.call_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_optimize_reduces_reproj_error(dtype):
    """With noisy initial poses and clean 2D observations, _optimize must reduce reprojection error."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    rng = np.random.default_rng(0)
    N, P, H, W = 5, 200, 256, 256
    f = 200.0

    points3d = rng.uniform(-1, 1, (P, 3)).astype(np.float64)
    points3d[:, 2] += 3.0

    extrinsics_clean = np.zeros((N, 3, 4), dtype=np.float32)
    for i in range(N):
        extrinsics_clean[i, :3, :3] = np.eye(3)
        extrinsics_clean[i, :3, 3] = rng.uniform(-0.3, 0.3, 3).astype(np.float32)

    intrinsics = np.tile(np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32), (N, 1, 1))

    tracks = np.zeros((N, P, 2), dtype=np.float32)
    vis_mask = np.ones((N, P), dtype=bool)
    for i in range(N):
        R, t = extrinsics_clean[i, :3, :3], extrinsics_clean[i, :3, 3]
        pts_cam = (R @ points3d.T).T + t
        z = pts_cam[:, 2]
        tracks[i, :, 0] = f * pts_cam[:, 0] / z + W / 2
        tracks[i, :, 1] = f * pts_cam[:, 1] / z + H / 2
        vis_mask[i] = (
            (z > 0.1) & (tracks[i, :, 0] >= 0) & (tracks[i, :, 0] < W) & (tracks[i, :, 1] >= 0) & (tracks[i, :, 1] < H)
        )

    extrinsics_noisy = extrinsics_clean.copy()
    for i in range(N):
        extrinsics_noisy[i, :3, 3] += rng.normal(0, 0.1, 3).astype(np.float32)

    def sq_reproj_error(ext):
        errs = []
        for i in range(N):
            R, t = ext[i, :3, :3], ext[i, :3, 3]
            pts_cam = (R @ points3d.T).T + t
            z = pts_cam[:, 2]
            px = f * pts_cam[:, 0] / z + W / 2
            py = f * pts_cam[:, 1] / z + H / 2
            proj = np.stack([px, py], axis=-1)
            mask = vis_mask[i]
            errs.append(np.square(proj[mask] - tracks[i][mask]).sum())
        return float(np.sum(errs))

    # Squared reprojection error at the start, the units of losses
    err_before = sq_reproj_error(extrinsics_noisy)

    ba = BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=None, lm_steps=20, dtype=dtype, **REPROJ_ONLY))
    ba._optimize(
        points3d.copy(),
        extrinsics_noisy,
        intrinsics,
        tracks,
        vis_mask.astype(np.float32),
        None,
        None,
        None,
    )

    # Final solve loss: gauge-free, unlike re-scoring the aligned cameras against the input points
    err_after = ba.losses["reprojection"]
    assert err_after < err_before, f"BA did not reduce error: {err_before:.4f} → {err_after:.4f}"


def test_optimize_no_reproj_filter():
    """A config with max_reproj_error=None skips reprojection filtering."""
    N, P, H, W = 4, 50, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    ba = BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=None, **REPROJ_ONLY))

    # P=50 is below min_inliers_per_frame, so the solve is refused after the (skipped) filter
    with (
        patch.object(bundle_adjustment, "project", wraps=bundle_adjustment.project) as project_spy,
        pytest.raises(ValueError, match="too few active frames/points"),
    ):
        ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), None, None, None)

    project_spy.assert_not_called()


########################################################################
# Tests for BundleAdjustment class
########################################################################


def _make_ff_result_for_ba(N=2, H=8, W=8):
    """Minimal PointcloudResult for BundleAdjustment tests."""
    from collab_splats.pointcloud.base import PointcloudResult

    return PointcloudResult(
        points=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=None,
        model_intrinsics=np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        image_paths=[Path(f"img{i}.jpg") for i in range(N)],
        original_coords=np.tile(np.array([0, 0, W, H, W, H], np.float32), (N, 1)),
        model_width=W,
        model_height=H,
        images=torch.zeros(N, 3, H, W),
        confidence=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
    )


def test_bundle_adjustment_refine_returns_arrays():
    """refine() takes arrays and returns (N, 4, 4) extrinsics and (N, 3, 3) intrinsics."""
    from collab_splats.geometry.bundle_adjustment import BundleAdjustment

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_ext[:, 0, 3] = 1.0
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with (
        patch(
            "collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm",
            return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3))),
        ),
        patch.object(BundleAdjustment, "_optimize", return_value=(refined_ext, refined_intr)),
    ):
        ext, K = BundleAdjustment(BundleAdjustmentConfig(**REPROJ_ONLY)).refine(
            result.images, result.confidence, result.world_points, result.extrinsics, result.intrinsics
        )

    assert ext.shape == (N, 4, 4)
    assert K.shape == (N, 3, 3)
    np.testing.assert_array_equal(ext[:, :3, :], refined_ext)
    np.testing.assert_array_equal(ext[:, 3], np.tile([0, 0, 0, 1], (N, 1)))
    np.testing.assert_array_equal(K, refined_intr)


def test_bundle_adjustment_refine_threads_config():
    """Config params and device are passed through to both private functions."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with (
        patch(
            "collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm",
            return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3))),
        ) as mock_tracks,
        patch.object(BundleAdjustment, "_optimize", return_value=(refined_ext, refined_intr)) as mock_opt,
    ):
        cfg = BundleAdjustmentConfig(device="cuda:1", lm_steps=5, max_reproj_error=2.0, **REPROJ_ONLY)
        _refine(BundleAdjustment(config=cfg), result)

    tracks_args, _ = mock_tracks.call_args
    assert tracks_args[3].device == "cuda:1"
    mock_opt.assert_called_once()


def test_optimize_rejects_cpu_device():
    """_optimize must raise a clear error for a non-CUDA device — bae LM is CUDA-only."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    # min_inliers_per_frame lowered so frames survive the filter and reach the device check
    ba = BundleAdjustment(
        config=BundleAdjustmentConfig(device="cpu", min_inliers_per_frame=10, max_reproj_error=None, **REPROJ_ONLY)
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), None, None, None)


def test_ba_config_new_fields_default():
    """BundleAdjustmentConfig has increment_size=0 and tracks_cache_dir=None by default."""
    from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig

    cfg = BundleAdjustmentConfig()
    assert cfg.increment_size == 0
    assert cfg.tracks_cache_dir is None


def test_ba_config_rejects_unknown_dtype():
    """
    A dtype other than float32 or float64 fails at construction.
    """
    with pytest.raises(ValueError, match="dtype must be"):
        BundleAdjustmentConfig(dtype="float16")


def test_ba_config_rejects_photometric_with_incremental_solve():
    """
    use_photometric with increment_size > 0 fails at construction; the incremental solve carries no images.
    """
    with pytest.raises(ValueError, match="use_photometric needs increment_size 0"):
        BundleAdjustmentConfig(use_photometric=True, increment_size=4)


def test_ba_config_track_quality_defaults():
    """Track-quality defaults: vis gate, coarse tracking (fine OOMs at 200 frames), shared camera."""
    from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig

    cfg = BundleAdjustmentConfig()
    assert cfg.vis_thresh == 0.2
    assert cfg.fine_tracking is False
    assert cfg.shared_camera is True
    assert cfg.max_query_pts == 4096
    assert cfg.query_frame_num == 8


def test_bundle_adjustment_default_config():
    """BundleAdjustment() with no args uses default BundleAdjustmentConfig."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    ba = BundleAdjustment()
    assert isinstance(ba.config, BundleAdjustmentConfig)
    assert ba.config.device is None
    assert ba.config.lm_steps == 40
    assert ba.loss_history == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimize_captures_loss_history_unconditionally():
    """_optimize always records one inner list of per-step losses (no flag gates it)."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    n_steps = 5
    cfg = BundleAdjustmentConfig(
        lm_steps=n_steps, lm_tol=0.0, min_inliers_per_frame=10, max_reproj_error=None, **REPROJ_ONLY
    )
    ba = BundleAdjustment(config=cfg)
    assert ba.loss_history == []

    ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), None, None, None)

    hist = ba.loss_history
    assert isinstance(hist, list)
    assert len(hist) == 1, f"one _optimize call → one inner list; got {len(hist)}"
    assert len(hist[0]) == n_steps, (
        f"expected all {n_steps} LM steps; got {len(hist[0])}. A short history means the "
        "StopOnPlateau reject_count abort is back."
    )
    assert all(isinstance(v, float) for v in hist[0])
    assert all(v >= 0 for v in hist[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimize_warns_when_no_lm_step_runs(caplog):
    """Every photometric scale short of samples: no LM step runs, and the solve says so."""
    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    images = np.zeros((N, 3, H, W), np.float32)
    depth = np.ones((N, H, W), np.float32)
    cfg = BundleAdjustmentConfig(min_inliers_per_frame=10, max_reproj_error=None, use_depth=False)
    ba = BundleAdjustment(config=cfg)

    with patch.object(bundle_adjustment, "photometric_samples", return_value=None):
        ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), images, depth, depth)

    assert ba.loss_history == [[]]
    assert "no LM step ran" in caplog.text


def test_lm_step_undoes_a_loss_increase():
    """A step bae kept after exhausting its rejects is undone; params and loss return to before."""
    param = torch.nn.Parameter(torch.ones(3))
    optimizer = SimpleNamespace(model=torch.nn.Module(), last=torch.tensor(1.0), reject_count=10)
    optimizer.model.p = param

    # bae keeps a bad step: params move and loss rises past last
    def bad_step(input):
        param.data += 5.0
        optimizer.loss = torch.tensor(9.0)
        return optimizer.loss

    optimizer.step = bad_step
    assert _lm_step(optimizer, {}) == (1.0, 0.0)
    assert torch.equal(param.data, torch.ones(3))
    assert float(optimizer.loss) == 1.0


def test_lm_step_keeps_a_loss_decrease():
    """A step that lowers the loss is kept."""
    param = torch.nn.Parameter(torch.ones(3))
    optimizer = SimpleNamespace(model=torch.nn.Module(), last=torch.tensor(1.0), reject_count=0)
    optimizer.model.p = param

    def good_step(input):
        param.data -= 0.5
        return torch.tensor(0.5)

    optimizer.step = good_step
    assert _lm_step(optimizer, {}) == (0.5, 0.5)
    assert torch.equal(param.data, torch.full((3,), 0.5))


def test_lm_step_drop_is_zero_without_a_prior_loss():
    """
    A zero loss before the step gives a zero drop, not a division error.
    """
    optimizer = SimpleNamespace(model=torch.nn.Module(), last=torch.tensor(0.0), reject_count=0)
    optimizer.model.p = torch.nn.Parameter(torch.ones(1))
    optimizer.step = lambda input: torch.tensor(0.0)

    assert _lm_step(optimizer, {}) == (0.0, 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimize_stops_after_patience_stalled_steps():
    """
    An unreachable lm_tol stalls every step, so the solve ends after lm_patience steps.
    """
    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    cfg = BundleAdjustmentConfig(
        lm_steps=20, lm_tol=1.0, lm_patience=3, min_inliers_per_frame=10, max_reproj_error=None, **REPROJ_ONLY
    )
    ba = BundleAdjustment(config=cfg)

    ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), None, None, None)

    assert len(ba.loss_history[0]) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_photometric_stall_ends_each_resample_not_the_scale():
    """
    With every step stalled, each of the 6 re-samples (3 + 2 + 1 over 3 scales) runs lm_patience steps.
    """
    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    # Textured frames over one shared plane: every camera sits 2 m above it, facing it
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    texture = 0.5 + 0.2 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    images = np.broadcast_to(texture, (N, 3, H, W)).astype(np.float32)
    depth = np.full((N, H, W), 2.0, np.float32)
    confidence = np.ones((N, H, W), np.float32)

    cfg = BundleAdjustmentConfig(
        lm_tol=1.0, lm_patience=2, use_depth=False, min_inliers_per_frame=10, max_reproj_error=None
    )
    ba = BundleAdjustment(config=cfg)

    ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), images, depth, confidence)

    assert len(ba.loss_history[0]) == 6 * 2


def _textured_plane(N, H, W):
    """
    Textured frames over one shared plane 2 m in front of every camera, with unit confidence.
    """
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    texture = 0.5 + 0.2 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    images = np.broadcast_to(texture, (N, 3, H, W)).astype(np.float32)
    depth = np.full((N, H, W), 2.0, np.float32)
    confidence = np.ones((N, H, W), np.float32)
    return images, depth, confidence


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_photometric_level_K_is_exactly_one_over_factor():
    """
    A 130 px frame pools to 32 px at 1/4, yet the level K stays exactly K/4; the floor only crops the edge.
    """
    N, P, H, W = 4, 60, 130, 130
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    images, depth, confidence = _textured_plane(N, H, W)
    cfg = BundleAdjustmentConfig(use_depth=False, min_inliers_per_frame=10, max_reproj_error=None)
    seen = []

    # Record each scale's K; None skips the scale, so no LM step moves the focal
    def record(w2c, gray, depth_l, K, seed):
        seen.append(K.cpu().numpy())

    with patch.object(bundle_adjustment, "photometric_samples", side_effect=record):
        BundleAdjustment(cfg)._optimize(
            pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), images, depth, confidence
        )

    assert len(seen) == 3

    for K, factor in zip(seen, (4, 2, 1)):
        np.testing.assert_allclose(K[:, :2], intrinsics[:, :2] / factor, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_photometric_resample_reads_live_per_camera_focal():
    """
    With shared_camera False each re-sample's K carries the model's current per-camera focal.
    """
    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    images, depth, confidence = _textured_plane(N, H, W)

    # Start 5% off the true focal so the solve moves it
    intrinsics[:, [0, 1], [0, 1]] *= 1.05
    cfg = BundleAdjustmentConfig(shared_camera=False, use_depth=False, min_inliers_per_frame=10, max_reproj_error=None)
    models = []
    seen = []
    real_model = bundle_adjustment._BAModel
    real_samples = bundle_adjustment.photometric_samples

    # Keep the model the solve builds, to read its focal at each re-sample
    def build(*args, **kwargs):
        models.append(real_model(*args, **kwargs))
        return models[-1]

    # Pair each re-sample's K focal with the model's focal at that moment, per scale factor
    def record(w2c, gray, depth_l, K, seed):
        factor = H // gray.shape[-2]
        live = models[0].pose.data[:, 7].detach().cpu().numpy() / factor
        seen.append((K[:, 0, 0].cpu().numpy(), live))
        return real_samples(w2c, gray, depth_l, K, seed=seed)

    with (
        patch.object(bundle_adjustment, "_BAModel", side_effect=build),
        patch.object(bundle_adjustment, "photometric_samples", side_effect=record),
    ):
        BundleAdjustment(cfg)._optimize(
            pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), images, depth, confidence
        )

    # The focal moved by the last (full-res) re-sample, and every re-sample saw the live value
    assert not np.allclose(seen[-1][1], intrinsics[:, 0, 0], rtol=1e-4)

    for K_focal, live in seen:
        np.testing.assert_allclose(K_focal, live, rtol=1e-5)


def test_ba_config_has_no_capture_loss_history_field():
    """capture_loss_history is deleted: history is always captured, so the flag is dead."""
    import dataclasses

    from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig

    names = {f.name for f in dataclasses.fields(BundleAdjustmentConfig)}
    assert "capture_loss_history" not in names


def test_tracks_cache_save_load(tmp_path):
    """extract_tracks saves to zarr; second call returns cached arrays without extracting."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_tracks = np.ones((N, 5, 2), dtype=np.float32)
    fake_vis = np.ones((N, 5), dtype=np.float32) * 0.9
    fake_pts3d = np.ones((5, 3), dtype=np.float32) * 2.0

    cfg = BundleAdjustmentConfig(tracks_cache_dir=tmp_path)
    ba = BundleAdjustment(config=cfg)

    extract_calls = []

    def fake_extract(images, confidence, world_points, cfg):
        extract_calls.append(1)
        return fake_tracks, fake_vis, fake_pts3d

    with patch("collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm", side_effect=fake_extract):
        t1, v1, p1 = _tracks(ba, result)
        t2, v2, p2 = _tracks(ba, result)

    assert len(extract_calls) == 1, "second call should use cache, not re-extract"
    np.testing.assert_array_equal(t1, fake_tracks)
    np.testing.assert_array_equal(t2, fake_tracks)
    np.testing.assert_array_equal(p1, fake_pts3d)
    np.testing.assert_array_equal(p2, fake_pts3d)


def test_tracks_cache_invalidates_on_config_change(tmp_path):
    """Cache is invalidated when query_frame_num changes; extraction runs again."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_a = (
        np.ones((N, 5, 2), dtype=np.float32),
        np.ones((N, 5), dtype=np.float32),
        np.ones((5, 3), dtype=np.float32),
    )
    fake_b = (
        np.zeros((N, 5, 2), dtype=np.float32),
        np.zeros((N, 5), dtype=np.float32),
        np.zeros((5, 3), dtype=np.float32),
    )
    extractions = [fake_a, fake_b]

    def fake_extract(*args, **kwargs):
        return extractions.pop(0)

    with patch("collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm", side_effect=fake_extract):
        ba1 = BundleAdjustment(config=BundleAdjustmentConfig(query_frame_num=5, tracks_cache_dir=tmp_path))
        _tracks(ba1, result)

        # Change query_frame_num — different key → cache miss
        ba2 = BundleAdjustment(config=BundleAdjustmentConfig(query_frame_num=10, tracks_cache_dir=tmp_path))
        t2, _, _ = _tracks(ba2, result)

    np.testing.assert_array_equal(t2, fake_b[0])


def test_tracks_cache_invalidates_on_fine_tracking_change(tmp_path):
    """fine_tracking is part of the cache key — flipping it must re-extract."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_a = (
        np.ones((N, 5, 2), dtype=np.float32),
        np.ones((N, 5), dtype=np.float32),
        np.ones((5, 3), dtype=np.float32),
    )
    fake_b = (
        np.zeros((N, 5, 2), dtype=np.float32),
        np.zeros((N, 5), dtype=np.float32),
        np.zeros((5, 3), dtype=np.float32),
    )
    extractions = [fake_a, fake_b]

    def fake_extract(*args, **kwargs):
        return extractions.pop(0)

    with patch("collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm", side_effect=fake_extract):
        ba1 = BundleAdjustment(config=BundleAdjustmentConfig(fine_tracking=True, tracks_cache_dir=tmp_path))
        _tracks(ba1, result)

        # Flip fine_tracking — different key → cache miss, re-extract
        ba2 = BundleAdjustment(config=BundleAdjustmentConfig(fine_tracking=False, tracks_cache_dir=tmp_path))
        t2, _, _ = _tracks(ba2, result)

    np.testing.assert_array_equal(t2, fake_b[0])


def test_tracks_cache_hit_on_vis_thresh_change(tmp_path):
    """vis_thresh is applied post-extraction — changing it must reuse the cache."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_tracks = np.ones((N, 5, 2), dtype=np.float32)
    fake_vis = np.ones((N, 5), dtype=np.float32) * 0.9
    fake_pts3d = np.ones((5, 3), dtype=np.float32) * 2.0

    extract_calls = []

    def fake_extract(*args, **kwargs):
        extract_calls.append(1)
        return fake_tracks, fake_vis, fake_pts3d

    with patch("collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm", side_effect=fake_extract):
        ba1 = BundleAdjustment(config=BundleAdjustmentConfig(vis_thresh=0.2, tracks_cache_dir=tmp_path))
        _tracks(ba1, result)
        ba2 = BundleAdjustment(config=BundleAdjustmentConfig(vis_thresh=0.5, tracks_cache_dir=tmp_path))
        _tracks(ba2, result)

    assert len(extract_calls) == 1, "vis_thresh change must NOT invalidate the track cache"


def test_tracks_cache_key_changes_with_world_points():
    """world_points enter the cache key: two backbones over the same images must not share tracks."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustmentConfig,
        _compute_tracks_cache_key,
    )

    cfg = BundleAdjustmentConfig()
    paths = [Path("a/frame_000000.png"), Path("a/frame_000001.png")]
    wp = np.zeros((2, 4, 4, 3), np.float32)
    k0 = _compute_tracks_cache_key(paths, wp, cfg)
    wp2 = wp.copy()
    wp2[0, 0, 0, 0] = 1.0
    assert _compute_tracks_cache_key(paths, wp2, cfg) != k0
    assert _compute_tracks_cache_key(paths, wp.copy(), cfg) == k0


def test_extract_receives_fine_tracking_kwarg(tmp_path):
    """extract_tracks passes cfg (with fine_tracking) through to extract_tracks_vggsfm."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    seen_kwargs = {}

    def fake_extract(images, confidence, world_points, cfg):
        seen_kwargs["fine_tracking"] = cfg.fine_tracking
        return (
            np.ones((N, 5, 2), dtype=np.float32),
            np.ones((N, 5), dtype=np.float32),
            np.ones((5, 3), dtype=np.float32),
        )

    with patch("collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm", side_effect=fake_extract):
        ba = BundleAdjustment(config=BundleAdjustmentConfig(fine_tracking=False))
        _tracks(ba, result)

    assert seen_kwargs.get("fine_tracking") is False


def test_incremental_ba_increment_size_n_matches_global():
    """increment_size >= N dispatches to the global path: _optimize called exactly once with all N frames."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

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
        return (refined_ext[: len(tracks)], refined_intr[: len(tracks)])

    with (
        patch(
            "collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm",
            return_value=(fake_tracks, fake_vis, fake_pts3d),
        ),
        patch.object(BundleAdjustment, "_optimize", side_effect=mock_optimize),
    ):

        _refine(BundleAdjustment(BundleAdjustmentConfig(increment_size=0, **REPROJ_ONLY)), result)
        _refine(BundleAdjustment(BundleAdjustmentConfig(increment_size=N, **REPROJ_ONLY)), result)
        _refine(BundleAdjustment(BundleAdjustmentConfig(increment_size=N + 10, **REPROJ_ONLY)), result)

    assert optimize_frame_counts == [
        N,
        N,
        N,
    ], f"increment_size=0/N/N+10 should all call _optimize once with N frames; got {optimize_frame_counts}"


def test_incremental_ba_warm_start_updates_registered_frames():
    """_refine_incremental updates extrinsics[:k] after each step (warm start propagates)."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

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
        return (refined, intrinsics.copy())

    with (
        patch(
            "collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm",
            return_value=(fake_tracks, fake_vis, fake_pts3d),
        ),
        patch.object(BundleAdjustment, "_optimize", side_effect=mock_optimize),
    ):

        ba = BundleAdjustment(BundleAdjustmentConfig(increment_size=2, **REPROJ_ONLY))
        _refine(ba, result)

    # N=6, increment_size=2 → steps k=2,4,6 → 3 _optimize calls
    assert len(received_extrinsics) == 3, f"expected 3 steps for N=6 increment_size=2, got {len(received_extrinsics)}"

    # Warm start: step-2 extrinsics[:2] should be step-1 refined output (diagonal+1), not original
    assert (
        received_extrinsics[1][:2, 0, 0].mean() > 1.0
    ), "warm start failed: step-2 extrinsics[:2] should be step-1 refined output, not original feedforward"


def test_incremental_ba_loss_history_has_one_entry_per_step():
    """loss_history contains one inner list per incremental k-step."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, H, W = 6, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    P = 5
    fake_tracks = np.zeros((N, P, 2), dtype=np.float32)
    fake_vis = np.ones((N, P), dtype=np.float32)
    fake_pts3d = np.zeros((P, 3), dtype=np.float32)

    def mock_optimize_with_hist(self_ba, pts3d, extrinsics, intrinsics, tracks, vis_scores, **kwargs):
        k = len(tracks)
        self_ba.loss_history.append([float(k) * 0.1])
        return (extrinsics.copy(), intrinsics.copy())

    with (
        patch(
            "collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm",
            return_value=(fake_tracks, fake_vis, fake_pts3d),
        ),
        patch.object(
            BundleAdjustment, "_optimize", lambda self_ba, *a, **kw: mock_optimize_with_hist(self_ba, *a, **kw)
        ),
    ):

        ba = BundleAdjustment(BundleAdjustmentConfig(increment_size=2, **REPROJ_ONLY))
        _refine(ba, result)

    # N=6, increment_size=2 → steps k=2,4,6 → 3 _optimize calls → 3 inner lists
    assert len(ba.loss_history) == 3, f"expected 3 inner lists for 3 steps; got {len(ba.loss_history)}"
    assert all(isinstance(entry, list) for entry in ba.loss_history)


########################################################################
# Tests for _filter_observations — vis-threshold gate + upstream filter order
########################################################################


def test_filter_observations_vis_threshold():
    """Observations under vis_thresh are dropped; landmarks left with <2 obs die with them."""
    from collab_splats.geometry.bundle_adjustment import _filter_observations

    vis_scores = np.array([[0.9, 0.1], [0.9, 0.9]], dtype=np.float32)
    tracks = np.zeros((2, 2, 2), dtype=np.float32)
    pts3d = np.zeros((2, 3), dtype=np.float64)
    ext = np.zeros((2, 3, 4), dtype=np.float32)
    intr = np.zeros((2, 3, 3), dtype=np.float32)

    vis = _filter_observations(
        vis_scores,
        tracks,
        pts3d,
        ext,
        intr,
        vis_thresh=0.2,
        max_reproj=None,
        min_inliers_per_frame=1,
    )
    # (0,1) fails the 0.2 gate; landmark 1 then has a single obs -> dropped everywhere
    assert not vis[0, 1] and not vis[1, 1]
    assert vis[0, 0] and vis[1, 0]


def test_filter_observations_no_single_obs_landmark_after_frame_drop():
    """Upstream order: frames drop BEFORE the >=2-obs landmark check, so no landmark
    can survive on observations from dropped frames (old code kept single-obs landmarks)."""
    from collab_splats.geometry.bundle_adjustment import _filter_observations

    vis_scores = np.array(
        [
            [0.9, 0.9, 0.9],  # frame 0: 3 obs
            [0.0, 0.0, 0.9],  # frame 1: 1 obs -> under min_inliers=2, whole frame drops
            [0.9, 0.9, 0.0],  # frame 2: 2 obs
        ],
        dtype=np.float32,
    )
    tracks = np.zeros((3, 3, 2), dtype=np.float32)
    pts3d = np.zeros((3, 3), dtype=np.float64)
    ext = np.zeros((3, 3, 4), dtype=np.float32)
    intr = np.zeros((3, 3, 3), dtype=np.float32)

    vis = _filter_observations(
        vis_scores,
        tracks,
        pts3d,
        ext,
        intr,
        vis_thresh=0.2,
        max_reproj=None,
        min_inliers_per_frame=2,
    )
    # Landmark 2 was seen only by frames 0 and (dropped) 1 -> single obs -> fully dropped
    assert not vis[:, 2].any()
    # Invariant: every surviving landmark has >=2 observations
    assert (vis.sum(0)[vis.any(0)] >= 2).all()


def _dense_filter_reference(vis_scores, tracks, pts3d, extrinsics, intrinsics, *, vis_thresh, max_reproj, min_inliers):
    """Pre-perf dense N x P CPU float64 _filter_observations, kept verbatim as the equivalence reference."""
    vis = vis_scores > vis_thresh

    # Dense projection of every point into every frame
    if max_reproj is not None:
        points = torch.as_tensor(pts3d, dtype=torch.float64)
        world_to_cam = torch.as_tensor(extrinsics, dtype=torch.float64)
        K = torch.as_tensor(intrinsics, dtype=torch.float64)
        projected = [project(points, world_to_cam[i], K[i]) for i in range(len(K))]
        proj2d = torch.stack([pixels for pixels, _ in projected])
        proj2d = proj2d.numpy()
        depth = torch.stack([cam[:, 2] for _, cam in projected])
        depth = depth.numpy()
        proj2d[depth <= 0] = 1e6
        reproj_err = np.linalg.norm(proj2d - tracks, axis=-1)
        vis[~(reproj_err <= max_reproj)] = False

    # Frame and landmark drops
    vis[vis.sum(1) < min_inliers] = False
    seen_enough = vis.sum(0) >= 2
    in_range = (np.abs(pts3d) < 3000).all(axis=-1)
    vis[:, ~(seen_enough & in_range)] = False
    return vis


def _filter_case(seed):
    """Random cameras, points and tracks: behind-camera, NaN, out-of-range points, NaN tracks, mixed vis."""
    rng = np.random.default_rng(seed)
    N, P = 7, 600

    # Cameras: small random rotations about identity, translations near the origin
    angles = rng.normal(scale=0.1, size=(N, 3))
    R = np.stack([_rotation_from_angles(a) for a in angles])
    t = rng.normal(scale=0.2, size=(N, 3))
    ext = np.concatenate([R, t[..., None]], axis=-1).astype(np.float32)
    K = np.tile(np.array([[500.0, 0, 320], [0, 520.0, 240], [0, 0, 1]], dtype=np.float32), (N, 1, 1))
    K[:, 0, 0] += rng.normal(scale=5.0, size=N).astype(np.float32)

    # Points mostly in front, some behind or at the camera plane, some NaN, some out of range
    pts = rng.normal(size=(P, 3))
    pts[:, 2] = rng.uniform(-2.0, 8.0, size=P)
    pts[:20, 2] = 0.0
    pts[20:30] = np.nan
    pts[30:40, 0] = 5000.0

    # Tracks: true projection plus noise spanning the threshold, some NaN
    pts_t = torch.as_tensor(pts)
    pts_batch = pts_t[None].expand(N, -1, -1)
    ext_t = torch.as_tensor(ext, dtype=torch.float64)
    K_t = torch.as_tensor(K, dtype=torch.float64)
    ref_pix, _ = project(pts_batch, ext_t, K_t)
    noise = rng.normal(size=(N, P, 2)) * rng.uniform(0.0, 12.0, size=(N, P, 1))
    tracks = (ref_pix.numpy() + noise).astype(np.float32)
    tracks[:, 40:45] = np.nan
    vis_scores = rng.uniform(size=(N, P)).astype(np.float32)
    return vis_scores, tracks, pts, ext, K


def _rotation_from_angles(angles):
    """Rotation matrix from xyz Euler angles."""
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_filter_observations_matches_dense_reference(monkeypatch, device, seed):
    """Gated per-pair projection returns the same mask as the dense N x P float64 reference."""
    monkeypatch.setattr(bundle_adjustment, "get_device", lambda: device)
    vis_scores, tracks, pts, ext, K = _filter_case(seed)
    kwargs = {"vis_thresh": 0.3, "max_reproj": 8.0}

    expected = _dense_filter_reference(vis_scores, tracks, pts, ext, K, min_inliers=50, **kwargs)
    got = _filter_observations(vis_scores, tracks, pts, ext, K, min_inliers_per_frame=50, batch_size=97, **kwargs)

    assert np.array_equal(got, expected)

    # The case exercises both outcomes of the reprojection gate
    gated = vis_scores > kwargs["vis_thresh"]
    assert 0 < (gated & ~got).sum() and got.sum() > 0


########################################################################
# Tests for _align_to_input_poses (post-solve drift undo)
########################################################################


def _w2c_from_rt(R, t):
    """Stack (N,3,3) rotations and (N,3) translations into (N,3,4) world-to-cam extrinsics."""
    return np.concatenate([R, t[..., None]], axis=-1).astype(np.float32)


def _random_w2c(N, seed=7):
    """N random valid world-to-cam poses."""
    rng = np.random.default_rng(seed)
    R0 = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(N)])
    R0[np.linalg.det(R0) < 0] *= -1.0
    return R0, rng.normal(size=(N, 3))


def test_align_to_input_poses_undoes_sim3():
    """Refined = input under a world Sim(3): active frames map back onto the input; dropped frames keep theirs."""
    N = 5
    R0, t0 = _random_w2c(N)
    original = _w2c_from_rt(R0, t0)

    # Known whole-scene Sim(3): 90 deg about z, scale 1.5, translation (0.3, -0.7, 2.0)
    R_g = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    s_g, t_g = 1.5, np.array([0.3, -0.7, 2.0])
    R_ref = R0 @ R_g.T
    refined = _w2c_from_rt(R_ref, s_g * t0 - np.einsum("nij,j->ni", R_ref, t_g))
    refined[3] = original[3]
    active = np.array([0, 1, 2, 4])

    out, scale = _align_to_input_poses(refined, original, active, fix_scale=False)

    assert np.allclose(out, original, atol=1e-4)
    assert np.isclose(scale, 1 / s_g, atol=1e-4)


def test_align_to_input_poses_fix_scale():
    """fix_scale keeps scale 1: a uniformly scaled refined set stays scaled, rotation stays identity."""
    R0, t0 = _random_w2c(4)
    original = _w2c_from_rt(R0, t0)
    refined = _w2c_from_rt(R0, 2.0 * t0)

    out, scale = _align_to_input_poses(refined, original, np.arange(4), fix_scale=True)

    assert scale == 1.0
    assert np.allclose(out[..., :3], original[..., :3], atol=1e-5)
    assert not np.allclose(out, original, atol=1e-2)


def test_align_to_input_poses_undoes_roll_about_collinear_centers():
    """A roll about a straight trajectory leaves centers fixed; the orientation mean still undoes it."""
    N = 6
    R0, _ = _random_w2c(N)
    centers = np.stack([np.linspace(0, 1, N), np.zeros(N), np.zeros(N)], 1)
    original = _w2c_from_rt(R0, -np.einsum("nij,nj->ni", R0, centers))

    # World roll of 10 deg about the x-axis trajectory line: centers unchanged, orientations rotated
    a = np.deg2rad(10.0)
    R_g = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    R_ref = R0 @ R_g.T
    refined = _w2c_from_rt(R_ref, -np.einsum("nij,nj->ni", R_ref, centers))

    out, _ = _align_to_input_poses(refined, original, np.arange(N), fix_scale=False)

    assert np.allclose(out, original, atol=1e-4)


def test_align_to_input_poses_needs_three_active_frames():
    """Fewer than 3 active frames cannot fix a Sim(3): the input is returned with no alignment."""
    refined = np.tile(np.eye(4, dtype=np.float32)[:3], (4, 1, 1))
    original = refined.copy()
    refined[:2, :, 3] += 5.0  # move only the active pair
    out, scale = _align_to_input_poses(refined, original, np.array([0, 1]), fix_scale=False)
    assert scale is None
    assert np.allclose(out, refined)


def test_align_to_input_poses_static_centers_keep_scale_one():
    """
    Refined cameras that share one center give no scale to fit: scale stays 1, poses stay finite.
    """
    R0, _ = _random_w2c(4)
    original = _w2c_from_rt(R0, np.zeros((4, 3)))
    refined = original.copy()

    out, scale = _align_to_input_poses(refined, original, np.arange(4), fix_scale=False)

    assert scale == 1.0
    assert np.isfinite(out).all()
    assert np.allclose(out, original, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimize_keeps_dropped_frame_and_shares_focal():
    """A frame dropped by the inlier gate keeps its input pose and takes the shared focal."""
    from collab_splats.geometry.bundle_adjustment import (
        BundleAdjustment,
        BundleAdjustmentConfig,
    )

    N, P, H, W = 5, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    vis = vis_mask.astype(np.float32)
    vis[0] = 0.0  # frame 0 has no admissible observations -> dropped by the inlier gate
    intrinsics = intrinsics.copy()
    intrinsics[0, 0, 0] = intrinsics[0, 1, 1] = 50.0  # distinct focal on the dropped frame

    cfg = BundleAdjustmentConfig(
        lm_steps=3, min_inliers_per_frame=10, shared_camera=True, max_reproj_error=None, **REPROJ_ONLY
    )
    ba = BundleAdjustment(config=cfg)
    ref_ext, ref_K = ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis, None, None, None)

    # shared_camera=True: the solved focal reaches the dropped frame too
    assert ref_K[0, 0, 0] == pytest.approx(ref_K[1, 0, 0], rel=1e-6)
    assert ref_K[0, 0, 0] != pytest.approx(50.0, rel=1e-6), "dropped frame kept its stale focal"

    # The dropped frame sits at its input pose, which is the frame the active set is aligned to
    assert np.allclose(ref_ext[0], extrinsics[0, :3], atol=1e-6)
    assert ba.alignment_scale is not None


def test_optimize_raises_when_too_few_observations_survive():
    """All-zero visibility leaves no active frame: _optimize must raise, not return input poses."""
    N, P = 3, 8
    ba = BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=None, **REPROJ_ONLY))
    ext = np.tile(np.eye(3, 4, dtype=np.float32), (N, 1, 1))
    intr = np.tile(np.diag([100.0, 100.0, 1.0]).astype(np.float32), (N, 1, 1))
    with pytest.raises(ValueError, match=r"0 frames, 0 points"):
        ba._optimize(
            np.zeros((P, 3)), ext, intr, np.zeros((N, P, 2), np.float32), np.zeros((N, P), np.float32), None, None, None
        )


def test_incremental_steps_skip_single_frame_windows(monkeypatch):
    """increment_size=1 must not hand _optimize a 1-frame window (it would now raise)."""
    seen = []

    def fake_optimize(self, pts3d, ext, intr, tracks, vis, **kwargs):
        seen.append(len(ext))
        return ext, intr

    monkeypatch.setattr(BundleAdjustment, "_optimize", fake_optimize)
    N = 4
    ba = BundleAdjustment(BundleAdjustmentConfig(increment_size=1, **REPROJ_ONLY))
    result = SimpleNamespace(
        images=np.zeros((N, 3, 8, 8), np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)),
    )
    intr = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    ba._refine_incremental(
        result.extrinsics, np.zeros((N, 5, 2)), np.zeros((N, 5)), np.zeros((5, 3)), intr, 1, None, np.ones((N, 8, 8))
    )
    assert seen == [2, 3, 4]


def _cache_ba(tmp_path):
    """A cache-enabled BA whose extractor is counted, plus a result to feed it."""
    result = _make_ff_result_for_ba(3, 8, 8)
    ba = BundleAdjustment(config=BundleAdjustmentConfig(tracks_cache_dir=tmp_path))
    calls = []

    def fake_extract(images, confidence, world_points, cfg):
        calls.append(1)
        return np.ones((3, 5, 2), np.float32), np.ones((3, 5), np.float32), np.ones((5, 3), np.float32)

    return ba, result, calls, fake_extract


def test_unreadable_track_cache_is_re_extracted(tmp_path):
    """A cache dir that is not a zarr store is rebuilt, not fatal."""
    ba, result, calls, fake_extract = _cache_ba(tmp_path)
    (tmp_path / "tracks.zarr").mkdir()
    (tmp_path / "tracks.zarr" / "zarr.json").write_text("not json")
    with patch("collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm", side_effect=fake_extract):
        _tracks(ba, result)
    assert calls == [1]


def test_unexpected_track_cache_error_propagates(tmp_path, monkeypatch):
    """A bug inside the cache read is not an unreadable cache: it must surface."""
    ba, result, calls, fake_extract = _cache_ba(tmp_path)
    (tmp_path / "tracks.zarr").mkdir()

    def broken_open(*args, **kwargs):
        raise TypeError("bug")

    monkeypatch.setattr("collab_splats.geometry.bundle_adjustment.zarr.open", broken_open)
    with patch("collab_splats.geometry.bundle_adjustment.extract_tracks_vggsfm", side_effect=fake_extract):
        with pytest.raises(TypeError, match="bug"):
            _tracks(ba, result)

    # zarr.open also backs the cache write, so only a zero extract count proves the read raised
    assert calls == []


########################################################################
# Residual Jacobians and gauge freedom per term set
########################################################################


def _term_problem(config, seed=0):
    """6 frames, 200 points, noisy tracks and depth, textured images: model and input dict."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N, P, H, W, f = 6, 200, 48, 64, 50.0
    dev, f64 = "cuda", torch.float64

    # Non-collinear camera path looking at a 3-5 m slab of points
    pts = torch.tensor(rng.uniform([-1, -1, 3], [1, 1, 5], (P, 3)), device=dev)
    rot = pp.so3(torch.tensor([[0.02 * i, 0.08 * (i - N / 2), 0.01] for i in range(N)], device=dev, dtype=f64)).Exp()
    trans = torch.tensor([[0.15 * i, 0.05 * (i % 2), 0.03 * i] for i in range(N)], device=dev, dtype=f64)
    poses = torch.cat([trans, rot.tensor()], 1)
    pc = torch.tensor([[W / 2, H / 2]] * N, device=dev, dtype=f64)

    # Every point seen by every frame: noisy pixels and 1% noisy depth
    cam_idx, pt_idx = (t.reshape(-1) for t in torch.meshgrid(torch.arange(N), torch.arange(P), indexing="ij"))
    cam_idx, pt_idx = cam_idx.to(dev), pt_idx.to(dev)
    x_cam = pp.SE3(poses)[cam_idx].Act(pts[pt_idx])
    uv = x_cam[:, :2] / x_cam[:, 2:] * f + pc[cam_idx] + 0.3 * torch.randn(len(cam_idx), 2, device=dev, dtype=f64)
    D = x_cam[:, 2] * (1 + 0.01 * torch.randn(len(cam_idx), device=dev, dtype=f64))
    w_z = 1 / (D * config.depth_sigma) if config.use_depth else torch.zeros_like(D)
    inputs = {
        "camera_indices": cam_idx,
        "point_indices": pt_idx,
        "principal_points": pc,
        "target": torch.cat([uv, D[:, None]], 1),
        "weight": torch.cat([torch.ones_like(uv), w_z[:, None]], 1),
    }

    # Photometric rows over a smooth texture and a wavy depth map
    if config.use_photometric:
        yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        gray = torch.stack([0.5 + 0.2 * torch.sin(xx / 3.0 + 0.1 * i) * torch.cos(yy / 4.0) for i in range(N)])
        depth = (4.0 + 0.3 * torch.sin(xx / 7.0) + 0.2 * torch.cos(yy / 5.0)).expand(N, H, W).contiguous()
        K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev, dtype=f64).expand(N, 3, 3)
        inputs["photometric"] = photometric_samples(
            pp.SE3(poses).matrix(), gray.double(), depth.double(), K.contiguous(), n_samples=256
        )

    # Parameter layout as _optimize builds it
    focal = torch.full((N, 1), f, device=dev, dtype=f64)
    cam_params = poses if config.shared_camera else torch.cat([poses, focal], 1)
    shared = focal[:1] if config.shared_camera else None
    return _BAModel(cam_params, pts, shared, refine_focal=True), inputs


TERM_SETS = [
    (REPROJ_ONLY, 7),
    ({"use_depth": True, "use_photometric": False}, 6),
    ({"use_depth": True, "use_photometric": True}, 6),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shared_camera", [True, False])
@pytest.mark.parametrize("terms,n_gauge", TERM_SETS)
def test_term_jacobian_matches_fd_and_gauge_count(terms, n_gauge, shared_camera):
    """bae Jacobian equals central differences under LM's own update; null space is the expected gauge."""
    model, inputs = _term_problem(BundleAdjustmentConfig(shared_camera=shared_camera, **terms))
    params = list(model.parameters())

    # bae sparse Jacobian, densified
    with torch.enable_grad():
        J = torch.cat([j.to_dense() for j in bae_graph.jacobian(model(**inputs), params)], 1).detach()

    # Central differences through LM.update_parameter (SE3 retraction, additive elsewhere)
    eps, cols = 1e-6, []
    with torch.no_grad():
        for k in range(J.shape[1]):
            step = torch.zeros(J.shape[1], 1, device=J.device, dtype=J.dtype)
            step[k] = eps
            bae_optim.LM.update_parameter(None, params, step)
            plus = model(**inputs).tensor().reshape(-1).clone()
            bae_optim.LM.update_parameter(None, params, -2 * step)
            minus = model(**inputs).tensor().reshape(-1).clone()
            bae_optim.LM.update_parameter(None, params, step)
            cols.append((plus - minus) / (2 * eps))
    J_fd = torch.stack(cols, 1)
    assert (J - J_fd).abs().max() <= 1e-6 * J_fd.abs().max()

    # Near-zero singular values count the unobservable whole-scene motions
    sv = torch.linalg.svdvals(J)
    assert int((sv < 1e-9 * sv[0]).sum()) == n_gauge


########################################################################
# Track source dispatch
########################################################################


def test_extract_tracks_vggsfm_source_unchanged(monkeypatch):
    calls = []

    def fake_vggsfm(images, conf, world_points, cfg):
        calls.append((images, conf, world_points, cfg))
        return np.zeros((2, 1, 2), np.float32), np.ones((2, 1), np.float32), np.zeros((1, 3), np.float32)

    monkeypatch.setattr(bundle_adjustment, "extract_tracks_vggsfm", fake_vggsfm)
    ba = BundleAdjustment(BundleAdjustmentConfig())

    ba.extract_tracks("imgs", "conf", "wp", None, extrinsics="ext", intrinsics="K", frame_paths=None)

    assert calls == [("imgs", "conf", "wp", ba.config)]


def test_extract_tracks_matcher_source_dispatches(monkeypatch):
    seen = {}

    def fake_build(matcher, images, frame_paths, world_points, extrinsics, intrinsics):
        seen.update(
            matcher=matcher,
            images=images,
            frame_paths=frame_paths,
            world_points=world_points,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        return np.zeros((2, 1, 2), np.float32), np.ones((2, 1), np.float32), np.zeros((1, 3), np.float32)

    monkeypatch.setattr(bundle_adjustment, "build_tracks", fake_build)
    monkeypatch.setattr(bundle_adjustment, "LocalMatcher", lambda name: f"matcher:{name}")
    monkeypatch.setattr(bundle_adjustment, "pytorch_gc", lambda: None)
    ba = BundleAdjustment(BundleAdjustmentConfig(track_source="xfeat"))

    ba.extract_tracks("imgs", "conf", "wp", None, extrinsics="ext", intrinsics="K", frame_paths=["a.png"])

    assert seen == {
        "matcher": "matcher:xfeat",
        "images": "imgs",
        "frame_paths": ["a.png"],
        "world_points": "wp",
        "extrinsics": "ext",
        "intrinsics": "K",
    }


def test_extract_tracks_matcher_source_needs_frame_paths():
    ba = BundleAdjustment(BundleAdjustmentConfig(track_source="xfeat"))

    with pytest.raises(ValueError, match="frame_paths"):
        ba.extract_tracks("imgs", "conf", "wp", None, extrinsics="ext", intrinsics="K", frame_paths=None)


def test_refine_threads_frame_paths_and_input_poses_to_extract_tracks():
    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    extrinsics = result.extrinsics.copy()
    extrinsics[:, 0, 3] = 0.5
    intrinsics = result.model_intrinsics.copy()
    intrinsics[:, 0, 0] = 7.0
    refined = (np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32), intrinsics.copy())
    tracks = (np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))

    with (
        patch.object(BundleAdjustment, "extract_tracks", return_value=tracks) as mock_extract,
        patch.object(BundleAdjustment, "_optimize", return_value=refined),
    ):
        BundleAdjustment(BundleAdjustmentConfig(**REPROJ_ONLY)).refine(
            result.images,
            result.confidence,
            result.world_points,
            extrinsics,
            intrinsics,
            frame_paths=[Path("a.png"), Path("b.png")],
        )

    kwargs = mock_extract.call_args.kwargs
    assert kwargs["frame_paths"] == [Path("a.png"), Path("b.png")]
    np.testing.assert_array_equal(kwargs["extrinsics"], extrinsics)
    np.testing.assert_array_equal(kwargs["intrinsics"], intrinsics)


def test_track_source_rejects_unknown():
    with pytest.raises(ValueError, match="track_source"):
        BundleAdjustmentConfig(track_source="disk")


def test_tracks_cache_key_includes_source():
    wp = np.zeros((2, 2, 2, 3), np.float32)
    k_v = bundle_adjustment._compute_tracks_cache_key(["a"], wp, BundleAdjustmentConfig())
    k_x = bundle_adjustment._compute_tracks_cache_key(["a"], wp, BundleAdjustmentConfig(track_source="xfeat"))

    assert k_v != k_x
