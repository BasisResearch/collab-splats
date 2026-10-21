"""
Tests for collab_splats.geometry.bundle_adjustment.

- module skipped without bae or pypose; bundle_adjustment imports both at load
- track extraction is patched at bundle_adjustment.extract_tracks; its own tests live in test_tracks.py
- CUDA-dependent solves are skipped without CUDA
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

# BA solver deps: bundle_adjustment imports them at load, so their absence skips the module
pp = pytest.importorskip("pypose")
bae_graph = pytest.importorskip("bae.autograd.graph")
bae_optim = pytest.importorskip("bae.optim")

from collab_splats.geometry import bundle_adjustment  # noqa: E402
from collab_splats.geometry.bundle_adjustment import (  # noqa: E402
    BundleAdjustment,
    BundleAdjustmentConfig,
    _align_to_input_poses,
    _BAModel,
    _filter_observations,
)
from collab_splats.geometry.photometric import photometric_samples  # noqa: E402
from collab_splats.geometry.projection import project  # noqa: E402

# Reprojection-only solve: these tests exercise the track path, which takes no depth
REPROJ_ONLY = {"use_photometric": False, "use_depth": False}


def _refine_inputs(N=2, H=8, W=8):
    """
    Identity-camera refine() arguments; extract_tracks and _optimize are patched in the tests that use it.
    """
    return {
        "images": torch.zeros(N, 3, H, W),
        "confidence": torch.ones(N, H, W),
        "world_points": np.zeros((N, H, W, 3), dtype=np.float32),
        "extrinsics": np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        "depth": np.ones((N, H, W), dtype=np.float32),
    }


def _flat(tracks, vis):
    """
    Dense (N, P) tracks and visibility as flat frame-major (frame, track, xy, score) observations.
    """
    vis = np.asarray(vis, dtype=np.float32)
    frame, track = np.nonzero(vis > 0)
    return (
        frame.astype(np.int32),
        track.astype(np.int32),
        tracks[frame, track],
        vis[frame, track],
    )


def _flat_tracks(N, P):
    """
    extract_tracks return value: every frame sees every one of P zero tracks.
    """
    return *_flat(
        np.zeros((N, P, 2), np.float32), np.ones((N, P), np.float32)
    ), np.zeros((P, 3), np.float32)


def _refine_with(cfg, optimize, N=2, P=5):
    """
    refine() over zero tracks with _optimize replaced by `optimize`; returns the BundleAdjustment.
    """
    ba = BundleAdjustment(cfg)
    tracks = _flat_tracks(N, P)

    with (
        patch.object(bundle_adjustment, "extract_tracks", return_value=tracks),
        patch.object(BundleAdjustment, "_optimize", optimize),
    ):
        ba.refine(**_refine_inputs(N))

    return ba


########################################################################
# BundleAdjustment.refine — synthetic smoke tests on real bae, CUDA-gated where they solve
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
    intrinsics = np.tile(
        np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )

    # Project points to 2D: K @ (R @ p + t) / z
    tracks = np.zeros((N, P, 2), dtype=np.float32)
    vis_mask = np.ones((N, P), dtype=bool)

    for i in range(N):
        pts_cam = (extrinsics[i, :3, :3] @ points3d.T).T + extrinsics[
            i, :3, 3
        ]  # (P, 3)
        z = pts_cam[:, 2]
        tracks[i, :, 0] = f * pts_cam[:, 0] / z + cx
        tracks[i, :, 1] = f * pts_cam[:, 1] / z + cy
        vis_mask[i] = z > 0.1

    return points3d, extrinsics, intrinsics, tracks, vis_mask


def test_optimize_raises_below_inlier_threshold():
    """_optimize raises when every frame falls below the inlier threshold after the reproj filter."""
    N, P, H, W = 4, 50, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    ba = BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=4.0, **REPROJ_ONLY))

    with (
        patch.object(
            bundle_adjustment,
            "reprojection_error",
            wraps=bundle_adjustment.reprojection_error,
        ) as spy,
        pytest.raises(ValueError, match="too few active frames/points"),
    ):
        ba._optimize(
            pts3d, extrinsics, intrinsics, *_flat(tracks, vis_mask), None, None, None
        )

    # The reprojection filter projects all gated pairs in one device batch
    assert spy.call_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_optimize_reduces_reproj_error(dtype):
    """With noisy initial poses and clean 2D observations, _optimize must reduce reprojection error."""

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

    ba = BundleAdjustment(
        BundleAdjustmentConfig(
            max_reproj_error=None, lm_steps=20, dtype=dtype, **REPROJ_ONLY
        )
    )
    ba._optimize(
        points3d,
        extrinsics_noisy,
        intrinsics,
        *_flat(tracks, vis_mask),
        None,
        None,
        None,
    )

    # Final solve loss: gauge-free, unlike re-scoring the aligned cameras against the input points
    err_after = ba.losses["reprojection"]
    assert err_after < err_before, (
        f"BA did not reduce error: {err_before:.4f} → {err_after:.4f}"
    )


def test_optimize_no_reproj_filter():
    """A config with max_reproj_error=None skips reprojection filtering."""
    N, P, H, W = 4, 50, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    ba = BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=None, **REPROJ_ONLY))

    # P=50 is below min_inliers_per_frame, so the solve is refused after the (skipped) filter
    with (
        patch.object(
            bundle_adjustment,
            "reprojection_error",
            wraps=bundle_adjustment.reprojection_error,
        ) as spy,
        pytest.raises(ValueError, match="too few active frames/points"),
    ):
        ba._optimize(
            pts3d, extrinsics, intrinsics, *_flat(tracks, vis_mask), None, None, None
        )

    spy.assert_not_called()


########################################################################
# Tests for BundleAdjustment class
########################################################################


def test_bundle_adjustment_refine_returns_arrays():
    """
    refine() takes arrays and returns (N, 4, 4) extrinsics and (N, 3, 3) intrinsics.
    """
    N = 2
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_ext[:, 0, 3] = 1.0
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with (
        patch.object(
            bundle_adjustment,
            "extract_tracks",
            return_value=_flat_tracks(N, 5),
        ),
        patch.object(
            BundleAdjustment, "_optimize", return_value=(refined_ext, refined_intr)
        ),
    ):
        ext, K = BundleAdjustment(
            BundleAdjustmentConfig(track_source="vggsfm", **REPROJ_ONLY)
        ).refine(**_refine_inputs(N))

    assert ext.shape == (N, 4, 4)
    assert K.shape == (N, 3, 3)
    np.testing.assert_array_equal(ext[:, :3, :], refined_ext)
    np.testing.assert_array_equal(ext[:, 3], np.tile([0, 0, 0, 1], (N, 1)))
    np.testing.assert_array_equal(K, refined_intr)


def test_bundle_adjustment_refine_threads_config():
    """
    Track source and track_kwargs reach extract_tracks, the solve device does not, and _optimize runs once.
    """
    N = 2
    refined = (
        np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32),
        np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
    )

    with (
        patch.object(
            bundle_adjustment,
            "extract_tracks",
            return_value=_flat_tracks(N, 5),
        ) as mock_tracks,
        patch.object(BundleAdjustment, "_optimize", return_value=refined) as mock_opt,
    ):
        cfg = BundleAdjustmentConfig(
            device="cuda:1",
            lm_steps=5,
            max_reproj_error=2.0,
            track_source="loma",
            track_kwargs={"retrieval": "megaloc"},
            **REPROJ_ONLY,
        )
        BundleAdjustment(config=cfg).refine(**_refine_inputs(N))

    assert mock_tracks.call_args.kwargs["source"] == "loma"
    assert mock_tracks.call_args.kwargs["retrieval"] == "megaloc"
    assert "device" not in mock_tracks.call_args.kwargs
    mock_opt.assert_called_once()


def test_optimize_rejects_cpu_device():
    """_optimize must raise a clear error for a non-CUDA device — bae LM is CUDA-only."""

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    # min_inliers_per_frame lowered so frames survive the filter and reach the device check
    ba = BundleAdjustment(
        config=BundleAdjustmentConfig(
            device="cpu", min_inliers_per_frame=10, max_reproj_error=None, **REPROJ_ONLY
        )
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        ba._optimize(
            pts3d, extrinsics, intrinsics, *_flat(tracks, vis_mask), None, None, None
        )


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


def test_ba_config_defaults():
    """
    Defaults: Schur with a frozen shared focal, global solve, xfeat tracks at their own defaults, no cache.
    """
    cfg = BundleAdjustmentConfig()
    assert cfg.solver == "schur"
    assert cfg.refine_focal is False
    assert cfg.increment_size == 0
    assert cfg.tracks_cache_dir is None
    assert cfg.vis_thresh == 0.2
    assert cfg.shared_camera is True
    assert cfg.track_source == "xfeat"
    assert cfg.track_kwargs == {}
    assert cfg.max_reproj_error == 4.0
    assert cfg.lm_steps == 40
    assert cfg.min_inliers_per_frame == 64


def test_ba_config_rejects_schur_with_refined_shared_focal():
    """
    Schur splits cameras and points only; a refined shared focal would be a third block.
    """
    with pytest.raises(ValueError, match="cannot refine a shared focal"):
        BundleAdjustmentConfig(refine_focal=True)

    BundleAdjustmentConfig(refine_focal=True, solver="lm")
    BundleAdjustmentConfig(refine_focal=True, shared_camera=False)


def test_ba_config_rejects_unknown_solver():
    """
    A solver other than schur or lm fails at construction.
    """
    with pytest.raises(ValueError, match="solver must be"):
        BundleAdjustmentConfig(solver="cholesky")


def test_bundle_adjustment_default_config():
    """
    BundleAdjustment() with no args uses the default config and starts with no loss history.
    """

    ba = BundleAdjustment()
    assert isinstance(ba.config, BundleAdjustmentConfig)
    assert ba.config.device is None
    assert ba.config.lm_steps == 40
    assert ba.loss_history == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimize_captures_loss_history_unconditionally():
    """_optimize always records one inner list of per-step losses (no flag gates it)."""

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    n_steps = 5
    cfg = BundleAdjustmentConfig(
        lm_steps=n_steps,
        lm_tol=0.0,
        min_inliers_per_frame=10,
        max_reproj_error=None,
        **REPROJ_ONLY,
    )
    ba = BundleAdjustment(config=cfg)
    assert ba.loss_history == []

    ba._optimize(
        pts3d, extrinsics, intrinsics, *_flat(tracks, vis_mask), None, None, None
    )

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
    cfg = BundleAdjustmentConfig(
        min_inliers_per_frame=10, max_reproj_error=None, use_depth=False
    )
    ba = BundleAdjustment(config=cfg)

    with patch.object(bundle_adjustment, "photometric_samples", return_value=None):
        ba._optimize(
            pts3d,
            extrinsics,
            intrinsics,
            *_flat(tracks, vis_mask),
            images,
            depth,
            depth,
        )

    assert ba.loss_history == [[]]
    assert "no LM step ran" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimize_loss_history_never_increases():
    """
    Every recorded LM loss is at or below the one before it.
    """
    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    extrinsics[:, :3, 3] += (
        np.random.default_rng(3).normal(0, 0.05, (N, 3)).astype(np.float32)
    )
    cfg = BundleAdjustmentConfig(
        lm_steps=20, min_inliers_per_frame=10, max_reproj_error=None, **REPROJ_ONLY
    )
    ba = BundleAdjustment(config=cfg)

    ba._optimize(
        pts3d, extrinsics, intrinsics, *_flat(tracks, vis_mask), None, None, None
    )

    hist = np.array(ba.loss_history[0])
    assert len(hist) > 1
    assert (np.diff(hist) <= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimize_undoes_a_step_that_raises_the_loss(caplog):
    """
    A step bae keeps although the loss rose is undone: loss and poses stay at their start.
    """
    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    cfg = BundleAdjustmentConfig(
        lm_steps=3,
        lm_patience=3,
        min_inliers_per_frame=10,
        max_reproj_error=None,
        **REPROJ_ONLY,
    )
    ba = BundleAdjustment(config=cfg)
    real_step = bundle_adjustment.Schur.step

    # bae keeps a bad step: poses jump and the loss lands far above the last one
    def bad_step(self, input):
        real_step(self, input=input)

        for param in self.model.parameters():
            param.data += 0.1

        return self.last * 10

    with patch.object(bundle_adjustment.Schur, "step", bad_step):
        refined, _ = ba._optimize(
            pts3d, extrinsics, intrinsics, *_flat(tracks, vis_mask), None, None, None
        )

    hist = ba.loss_history[0]
    assert len(hist) == 3
    assert hist[0] == hist[1] == hist[2]
    assert "undone" in caplog.text
    np.testing.assert_allclose(refined, extrinsics[:, :3], atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimize_stops_after_patience_stalled_steps():
    """
    An unreachable lm_tol stalls every step, so the solve ends after lm_patience steps.
    """
    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    cfg = BundleAdjustmentConfig(
        lm_steps=20,
        lm_tol=1.0,
        lm_patience=3,
        min_inliers_per_frame=10,
        max_reproj_error=None,
        **REPROJ_ONLY,
    )
    ba = BundleAdjustment(config=cfg)

    ba._optimize(
        pts3d, extrinsics, intrinsics, *_flat(tracks, vis_mask), None, None, None
    )

    assert len(ba.loss_history[0]) == 3


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
def test_photometric_stall_ends_each_resample_not_the_scale():
    """
    With every step stalled, each of the 6 re-samples (3 + 2 + 1 over 3 scales) runs lm_patience steps.
    """
    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    images, depth, confidence = _textured_plane(N, H, W)
    cfg = BundleAdjustmentConfig(
        lm_tol=1.0,
        lm_patience=2,
        use_depth=False,
        min_inliers_per_frame=10,
        max_reproj_error=None,
    )
    ba = BundleAdjustment(config=cfg)

    ba._optimize(
        pts3d,
        extrinsics,
        intrinsics,
        *_flat(tracks, vis_mask),
        images,
        depth,
        confidence,
    )

    assert len(ba.loss_history[0]) == 6 * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_photometric_level_K_is_pixel_center_rescale():
    """
    A 130 px frame pools to 32 px at 1/4, yet the level K stays the exact 1/4 rescale; the floor only crops the edge.

    - focal is K/factor; principal point is the pixel-center rescale (c + 0.5)/factor - 0.5
    """
    N, P, H, W = 4, 60, 130, 130
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)
    images, depth, confidence = _textured_plane(N, H, W)
    cfg = BundleAdjustmentConfig(
        use_depth=False, min_inliers_per_frame=10, max_reproj_error=None
    )
    seen = []

    # Record each scale's K; None skips the scale, so no LM step moves the focal
    def record(w2c, gray, depth_l, K, seed):
        seen.append(K.cpu().numpy())

    with patch.object(bundle_adjustment, "photometric_samples", side_effect=record):
        BundleAdjustment(cfg)._optimize(
            pts3d,
            extrinsics,
            intrinsics,
            *_flat(tracks, vis_mask),
            images,
            depth,
            confidence,
        )

    assert len(seen) == 3

    for K, factor in zip(seen, (4, 2, 1)):
        np.testing.assert_allclose(
            K[:, [0, 1], [0, 1]], intrinsics[:, [0, 1], [0, 1]] / factor, rtol=1e-6
        )
        np.testing.assert_allclose(
            K[:, :2, 2], (intrinsics[:, :2, 2] + 0.5) / factor - 0.5, rtol=1e-6
        )


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
    cfg = BundleAdjustmentConfig(
        shared_camera=False,
        refine_focal=True,
        use_depth=False,
        min_inliers_per_frame=10,
        max_reproj_error=None,
    )
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
            pts3d,
            extrinsics,
            intrinsics,
            *_flat(tracks, vis_mask),
            images,
            depth,
            confidence,
        )

    # The focal moved by the last (full-res) re-sample, and every re-sample saw the live value
    assert not np.allclose(seen[-1][1], intrinsics[:, 0, 0], rtol=1e-4)

    for K_focal, live in seen:
        np.testing.assert_allclose(K_focal, live, rtol=1e-5)


def test_incremental_ba_increment_size_n_matches_global():
    """
    increment_size 0, N or above N is one global solve: _optimize called once with all N frames.
    """
    N = 4
    optimize_frame_counts = []

    def mock_optimize(
        self, pts3d, extrinsics, intrinsics, frame, track, xy, score, **kwargs
    ):
        optimize_frame_counts.append(len(np.unique(frame)))
        return extrinsics, intrinsics

    for increment_size in (0, N, N + 10):
        cfg = BundleAdjustmentConfig(
            increment_size=increment_size, track_source="vggsfm", **REPROJ_ONLY
        )
        _refine_with(cfg, mock_optimize, N=N)

    assert optimize_frame_counts == [N, N, N], (
        f"expected one N-frame solve each; got {optimize_frame_counts}"
    )


def test_incremental_ba_warm_start_updates_registered_frames():
    """
    Each incremental step starts from the previous step's refined poses.
    """
    N = 6
    received_extrinsics = []

    def mock_optimize(
        self, pts3d, extrinsics, intrinsics, frame, track, xy, score, **kwargs
    ):
        received_extrinsics.append(extrinsics.copy())
        refined = extrinsics.copy()
        refined[:, 0, 0] += float(len(received_extrinsics))
        return refined, intrinsics.copy()

    _refine_with(
        BundleAdjustmentConfig(increment_size=2, track_source="vggsfm", **REPROJ_ONLY),
        mock_optimize,
        N=N,
    )

    # N=6, increment_size=2 → steps k=2,4,6 → 3 _optimize calls
    assert len(received_extrinsics) == 3, (
        f"expected 3 steps for N=6 increment_size=2, got {len(received_extrinsics)}"
    )

    # Warm start: step-2 extrinsics[:2] should be step-1 refined output (diagonal+1), not original
    assert received_extrinsics[1][:2, 0, 0].mean() > 1.0, (
        "warm start failed: step-2 extrinsics[:2] should be step-1 refined output, not original feedforward"
    )


def test_incremental_ba_loss_history_has_one_entry_per_step():
    """
    loss_history contains one inner list per incremental k-step.
    """

    def mock_optimize(
        self, pts3d, extrinsics, intrinsics, frame, track, xy, score, **kwargs
    ):
        self.loss_history.append([len(np.unique(frame)) * 0.1])
        return extrinsics, intrinsics

    cfg = BundleAdjustmentConfig(increment_size=2, track_source="vggsfm", **REPROJ_ONLY)
    ba = _refine_with(cfg, mock_optimize, N=6)

    # N=6, increment_size=2 → steps k=2,4,6 → 3 _optimize calls → 3 inner lists
    assert len(ba.loss_history) == 3, (
        f"expected 3 inner lists for 3 steps; got {len(ba.loss_history)}"
    )
    assert all(isinstance(entry, list) for entry in ba.loss_history)


def test_incremental_steps_skip_single_frame_windows():
    """
    increment_size=1 never hands _optimize a 1-frame window (it would raise).
    """
    seen = []

    def mock_optimize(
        self, pts3d, extrinsics, intrinsics, frame, track, xy, score, **kwargs
    ):
        seen.append(len(extrinsics))
        return extrinsics, intrinsics

    _refine_with(
        BundleAdjustmentConfig(increment_size=1, **REPROJ_ONLY), mock_optimize, N=4
    )

    assert seen == [2, 3, 4]


########################################################################
# Tests for _filter_observations — vis-threshold gate + upstream filter order
########################################################################


def test_filter_observations_vis_threshold():
    """Observations under vis_thresh are dropped; landmarks left with <2 obs die with them."""

    vis_scores = np.array([[0.9, 0.1], [0.9, 0.9]], dtype=np.float32)
    tracks = np.zeros((2, 2, 2), dtype=np.float32)
    pts3d = np.zeros((2, 3), dtype=np.float64)
    ext = np.zeros((2, 3, 4), dtype=np.float32)
    intr = np.zeros((2, 3, 3), dtype=np.float32)

    keep = _filter_observations(
        *_flat(tracks, vis_scores),
        pts3d,
        ext,
        intr,
        vis_thresh=0.2,
        max_reproj=None,
        min_inliers_per_frame=1,
    )

    vis = keep.reshape(2, 2)

    # (0,1) fails the 0.2 gate; landmark 1 then has a single obs -> dropped everywhere
    assert not vis[0, 1] and not vis[1, 1]
    assert vis[0, 0] and vis[1, 0]


def test_filter_observations_no_single_obs_landmark_after_frame_drop():
    """
    Frames drop before the >=2-obs landmark check, so no landmark survives on a dropped frame's observations.
    """

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
    frame, track, xy, score = _flat(tracks, vis_scores)

    keep = _filter_observations(
        frame,
        track,
        xy,
        score,
        pts3d,
        ext,
        intr,
        vis_thresh=0.2,
        max_reproj=None,
        min_inliers_per_frame=2,
    )

    vis = np.zeros((3, 3), bool)
    vis[frame, track] = keep

    # Landmark 2 was seen only by frames 0 and (dropped) 1 -> single obs -> fully dropped
    assert not vis[:, 2].any()
    # Invariant: every surviving landmark has >=2 observations
    assert (vis.sum(0)[vis.any(0)] >= 2).all()


def _dense_filter_reference(
    vis_scores,
    tracks,
    pts3d,
    extrinsics,
    intrinsics,
    *,
    vis_thresh,
    max_reproj,
    min_inliers,
):
    """
    Pre-perf dense N x P CPU float64 _filter_observations, kept verbatim as the equivalence reference.
    """
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
    K = np.tile(
        np.array([[500.0, 0, 320], [0, 520.0, 240], [0, 0, 1]], dtype=np.float32),
        (N, 1, 1),
    )
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


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_filter_observations_matches_dense_reference(monkeypatch, device, seed):
    """Flat per-observation filter returns the same mask as the dense N x P float64 reference."""
    monkeypatch.setattr(bundle_adjustment, "get_device", lambda: device)
    vis_scores, tracks, pts, ext, K = _filter_case(seed)
    kwargs = {"vis_thresh": 0.3, "max_reproj": 8.0}

    # Every (frame, point) cell as an observation, so the flat mask reshapes onto the grid
    frame, track = np.nonzero(np.ones(vis_scores.shape, bool))
    flat = (frame, track, tracks[frame, track], vis_scores[frame, track])

    expected = _dense_filter_reference(
        vis_scores, tracks, pts, ext, K, min_inliers=50, **kwargs
    )
    got = _filter_observations(
        *flat, pts, ext, K, min_inliers_per_frame=50, batch_size=97, **kwargs
    )
    got = got.reshape(vis_scores.shape)

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
    R_g = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]]
    )
    R_ref = R0 @ R_g.T
    refined = _w2c_from_rt(R_ref, -np.einsum("nij,nj->ni", R_ref, centers))

    out, _ = _align_to_input_poses(refined, original, np.arange(N), fix_scale=False)

    assert np.allclose(out, original, atol=1e-4)


def test_align_to_input_poses_needs_three_active_frames():
    """Fewer than 3 active frames cannot fix a Sim(3): the input is returned with no alignment."""
    refined = np.tile(np.eye(4, dtype=np.float32)[:3], (4, 1, 1))
    original = refined.copy()
    refined[:2, :, 3] += 5.0  # move only the active pair

    out, scale = _align_to_input_poses(
        refined, original, np.array([0, 1]), fix_scale=False
    )

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

    N, P, H, W = 5, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    vis = vis_mask.astype(np.float32)
    vis[0] = 0.0  # frame 0 has no admissible observations -> dropped by the inlier gate
    intrinsics = intrinsics.copy()
    intrinsics[0, 0, 0] = intrinsics[0, 1, 1] = (
        50.0  # distinct focal on the dropped frame
    )

    cfg = BundleAdjustmentConfig(
        lm_steps=3,
        min_inliers_per_frame=10,
        shared_camera=True,
        max_reproj_error=None,
        **REPROJ_ONLY,
    )
    ba = BundleAdjustment(config=cfg)
    ref_ext, ref_K = ba._optimize(
        pts3d, extrinsics, intrinsics, *_flat(tracks, vis), None, None, None
    )

    # shared_camera=True: the solved focal reaches the dropped frame too
    assert ref_K[0, 0, 0] == pytest.approx(ref_K[1, 0, 0], rel=1e-6)
    assert ref_K[0, 0, 0] != pytest.approx(50.0, rel=1e-6), (
        "dropped frame kept its stale focal"
    )

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
            np.zeros((P, 3)),
            ext,
            intr,
            *_flat(np.zeros((N, P, 2), np.float32), np.zeros((N, P))),
            None,
            None,
            None,
        )


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
    rot = pp.so3(
        torch.tensor(
            [[0.02 * i, 0.08 * (i - N / 2), 0.01] for i in range(N)],
            device=dev,
            dtype=f64,
        )
    ).Exp()
    trans = torch.tensor(
        [[0.15 * i, 0.05 * (i % 2), 0.03 * i] for i in range(N)], device=dev, dtype=f64
    )
    poses = torch.cat([trans, rot.tensor()], 1)
    pc = torch.tensor([[W / 2, H / 2]] * N, device=dev, dtype=f64)

    # Every point seen by every frame: noisy pixels and 1% noisy depth
    cam_idx, pt_idx = (
        t.reshape(-1)
        for t in torch.meshgrid(torch.arange(N), torch.arange(P), indexing="ij")
    )
    cam_idx, pt_idx = cam_idx.to(dev), pt_idx.to(dev)
    x_cam = pp.SE3(poses)[cam_idx].Act(pts[pt_idx])
    uv = (
        x_cam[:, :2] / x_cam[:, 2:] * f
        + pc[cam_idx]
        + 0.3 * torch.randn(len(cam_idx), 2, device=dev, dtype=f64)
    )
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
        yy, xx = torch.meshgrid(
            torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij"
        )
        gray = torch.stack(
            [
                0.5 + 0.2 * torch.sin(xx / 3.0 + 0.1 * i) * torch.cos(yy / 4.0)
                for i in range(N)
            ]
        )
        depth = (
            (4.0 + 0.3 * torch.sin(xx / 7.0) + 0.2 * torch.cos(yy / 5.0))
            .expand(N, H, W)
            .contiguous()
        )
        K = torch.tensor(
            [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev, dtype=f64
        ).expand(N, 3, 3)
        inputs["photometric"] = photometric_samples(
            pp.SE3(poses).matrix(),
            gray.double(),
            depth.double(),
            K.contiguous(),
            n_samples=256,
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
    model, inputs = _term_problem(
        BundleAdjustmentConfig(shared_camera=shared_camera, **terms)
    )
    params = list(model.parameters())

    # bae sparse Jacobian, densified
    with torch.enable_grad():
        J = torch.cat(
            [j.to_dense() for j in bae_graph.jacobian(model(**inputs), params)], 1
        ).detach()

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
# Track settings
########################################################################


def test_refine_threads_frame_paths_and_input_poses_to_extract_tracks():
    """
    refine() hands extract_tracks the frame paths, the input poses and K unchanged, and track_kwargs.
    """
    N = 2
    inputs = _refine_inputs(N)
    extrinsics = inputs["extrinsics"]
    extrinsics[:, 0, 3] = 0.5
    intrinsics = inputs["intrinsics"]
    intrinsics[:, 0, 0] = 7.0
    refined = (np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32), intrinsics.copy())
    tracks = _flat_tracks(N, 5)

    with (
        patch.object(
            bundle_adjustment, "extract_tracks", return_value=tracks
        ) as mock_extract,
        patch.object(BundleAdjustment, "_optimize", return_value=refined),
    ):
        BundleAdjustment(
            BundleAdjustmentConfig(**REPROJ_ONLY, track_kwargs={"seed_fraction": 0.5})
        ).refine(**inputs, frame_paths=[Path("a.png"), Path("b.png")])

    kwargs = mock_extract.call_args.kwargs
    assert kwargs["frame_paths"] == [Path("a.png"), Path("b.png")]
    assert kwargs["seed_fraction"] == 0.5
    np.testing.assert_array_equal(kwargs["extrinsics"], extrinsics)
    np.testing.assert_array_equal(kwargs["intrinsics"], intrinsics)


def test_track_source_rejects_unknown():
    """
    A track_source outside vggsfm, xfeat and loma fails at construction.
    """
    with pytest.raises(ValueError, match="track_source"):
        BundleAdjustmentConfig(track_source="disk")
