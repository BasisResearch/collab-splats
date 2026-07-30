"""Tests for lift_point_features module-level helper."""

from unittest.mock import patch

import numpy as np
import torch
import zarr

from collab_splats.dashboard.viewer import lift_point_features
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

EXTRACTOR = "talk2dino"


def _write_2d_cache(semantics_dir, extractor=EXTRACTOR):
    """Put the 2D patch cache in place — the write paths read the extractor off its stem.

    Every real caller of the on-demand lift has it: the lift itself reads its feature maps out
    of this very store, so a semantics dir without one cannot reach the save.
    """
    semantics_dir.mkdir(parents=True, exist_ok=True)
    zarr.open(str(semantics_dir / f"{extractor}.zarr"), mode="w")["features"] = np.zeros((1, 4, 2, 2), dtype=np.float32)


def test_lift_point_features_normalises():
    fake_lifted = torch.tensor([[3.0, 4.0], [0.0, 2.0]])  # norms 5, 2

    def fake_lift(feature_maps, result):
        # Proves the load_feature_maps patch is still intercepting: a patch that stopped
        # biting would hand the real (globbing) loader an object() and never reach here.
        assert feature_maps == ["sentinel-map"]
        return fake_lifted

    with (
        patch("collab_splats.pointcloud.utils.lift_features", fake_lift),
        patch("collab_splats.dashboard.pipeline.load_feature_maps", return_value=["sentinel-map"]),
    ):
        out = lift_point_features(result=object(), semantics_dir=object())
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_viewer_has_no_private_feature_map_loader():
    """One shared definition (item 5) — the byte-identical viewer copy is gone."""
    import collab_splats.dashboard.viewer as viewer_mod

    assert not hasattr(viewer_mod, "_load_feature_maps")


def test_lift_point_features_reloads_dense_on_demand(tmp_path):
    """A lean display result must be re-hydrated from its zarr before lifting."""
    # Dense-bearing store: depth + confidence + pixel_indices persisted on disk.
    n, p, h, w = 2, 5, 4, 4
    full = FeedforwardResult(
        points=np.zeros((p, 3), dtype=np.float32),
        colors=np.zeros((p, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        image_paths=[tmp_path / f"{i:05d}.jpg" for i in range(n)],
        original_coords=np.zeros((n, 6), dtype=np.float32),
        model_width=w,
        model_height=h,
        depth=np.ones((n, h, w), dtype=np.float32),
        confidence=torch.ones((n, h, w)),
        pixel_indices=np.zeros((p, 3), dtype=np.int32),
    )
    store = tmp_path / "feedforward.zarr"
    full.save_zarr(store)
    lean = FeedforwardResult.load_zarr(
        store,
        load_depth=False,
        load_world_points=False,
        load_confidence=False,
        load_features=False,
        load_pixel_indices=False,
    )

    seen = {}

    def fake_lift(feature_maps, result):
        # Record whether the lift saw the re-hydrated dense members.
        seen["dense"] = all(getattr(result, f) is not None for f in ("depth", "confidence", "pixel_indices"))
        return torch.zeros((p, 2))

    with (
        patch("collab_splats.pointcloud.utils.lift_features", fake_lift),
        patch("collab_splats.dashboard.pipeline.load_feature_maps", return_value=[]),
    ):
        lift_point_features(lean, tmp_path)
    assert seen["dense"] is True


def test_ensure_lifted_self_upgrades_legacy_scene(tmp_path, monkeypatch):
    """A successful on-demand lift persists the canonical artifact pair so the scene upgrades."""
    import collab_splats.dashboard.viewer as viewer_mod
    from collab_splats.dashboard.operation_log import OperationLog
    from collab_splats.dashboard.viewer import SplitViewer

    lifted = np.ones((5, 4), dtype=np.float32)
    monkeypatch.setattr(viewer_mod, "lift_point_features", lambda result, sem: lifted)
    _write_2d_cache(tmp_path)
    op_log = OperationLog()
    viewer = SplitViewer(off_screen=True, op_log=op_log)
    viewer._result = object()  # anything non-None; the lift itself is stubbed
    viewer._semantics_dir = tmp_path
    viewer.ensure_lifted(op_log)
    # Both halves of the pair — codes alone are unreadable without the weights
    assert (tmp_path / f"{EXTRACTOR}_lifted.zarr").exists()
    assert (tmp_path / f"{EXTRACTOR}_ae.pt").is_file()
    codes = np.asarray(zarr.open(str(tmp_path / f"{EXTRACTOR}_lifted.zarr"), mode="r")["features"])
    assert codes.shape == (5, 4)  # 5 points; latent capped at the 4-D input width
    assert any("scene upgraded" in line for line in op_log.log_lines)


def test_ensure_lifted_relifts_when_cached_codes_lack_weights(tmp_path, monkeypatch):
    """Orphaned codes are not a cache: re-lift and rewrite the complete pair (self-heal).

    Reading them back would raise (or, worse, hand back undecoded codes) on every query with
    no path out — the scene must recover by redoing the lift it never finished caching.
    """
    import collab_splats.dashboard.viewer as viewer_mod
    from collab_splats.dashboard.viewer import SplitViewer
    from collab_splats.semantics.compression import (
        FeatureAutoencoder,
        write_point_features,
    )

    # Half-written pair: latent codes on disk, the weights that decode them missing.
    _write_2d_cache(tmp_path)
    write_point_features(
        tmp_path, EXTRACTOR, np.zeros((4, 2), dtype=np.float32), FeatureAutoencoder(input_dim=5, latent_dim=2)
    )
    (tmp_path / f"{EXTRACTOR}_ae.pt").unlink()

    monkeypatch.setattr(viewer_mod, "lift_point_features", lambda result, sem: np.ones((5, 4), dtype=np.float32))
    viewer = SplitViewer(off_screen=True)
    viewer._result = object()
    viewer._semantics_dir = tmp_path
    viewer.ensure_lifted()

    assert viewer._point_features.shape == (5, 4)  # the fresh lift, not the stale (4, 2) codes
    assert (tmp_path / f"{EXTRACTOR}_ae.pt").is_file()  # pair completed
    assert np.asarray(zarr.open(str(tmp_path / f"{EXTRACTOR}_lifted.zarr"), mode="r")["features"]).shape == (5, 4)


def test_save_point_features_gates_fit_on_the_shared_policy(tmp_path, monkeypatch):
    """The self-upgrade fit is gated on reconstruction fidelity, not just an epoch ceiling.

    Both dashboard fit paths now read ONE policy from configs/base.yaml (item 7): the
    self-upgrade must not be held to a stricter (or looser) bar than a fresh reconstruction.
    """
    import collab_splats.dashboard.viewer as viewer_mod
    from collab_splats.dashboard.pipeline import semantics_ae_policy
    from collab_splats.semantics import compression

    seen = {}
    real_fit = compression.FeatureAutoencoder.fit

    def spy_fit(self, features, **kwargs):
        seen.update(kwargs)
        return real_fit(self, features, **kwargs)

    monkeypatch.setattr(compression.FeatureAutoencoder, "fit", spy_fit)
    _write_2d_cache(tmp_path)
    viewer_mod._save_point_features(tmp_path, np.random.rand(8, 4).astype(np.float32))

    policy = semantics_ae_policy()
    assert seen["target_cosine"] == policy.target_cosine
    assert seen["epochs"] == policy.max_epochs


def test_save_point_features_takes_no_latent_dim_argument(tmp_path):
    """item 12: the always-default param is gone; the width comes from the shared policy."""
    import inspect

    import collab_splats.dashboard.viewer as viewer_mod

    assert "latent_dim" not in inspect.signature(viewer_mod._save_point_features).parameters


def test_save_point_features_latent_width_comes_from_the_policy(tmp_path):
    """A wider-than-latent input must actually compress to the configured width."""
    import collab_splats.dashboard.viewer as viewer_mod
    from collab_splats.dashboard.pipeline import semantics_ae_policy

    latent = semantics_ae_policy().latent_dim
    feats = np.random.rand(32, latent * 2).astype(np.float32)
    _write_2d_cache(tmp_path)
    viewer_mod._save_point_features(tmp_path, feats)

    codes = np.asarray(zarr.open(str(tmp_path / f"{EXTRACTOR}_lifted.zarr"), mode="r")["features"])
    assert codes.shape == (32, latent)


def test_save_point_features_warns_when_the_latent_clamp_binds(tmp_path, caplog):
    """item 10: a <=latent-width input gets no compression and a lossy round-trip."""
    import logging

    import collab_splats.dashboard.viewer as viewer_mod

    _write_2d_cache(tmp_path)
    with caplog.at_level(logging.WARNING, logger="collab_splats.dashboard.pipeline"):
        viewer_mod._save_point_features(tmp_path, np.random.rand(8, 4).astype(np.float32))
    assert "no-op" in caplog.text


def test_ensure_lifted_failure_does_not_write_artifacts(tmp_path, monkeypatch):
    import collab_splats.dashboard.viewer as viewer_mod
    from collab_splats.dashboard.viewer import SplitViewer

    def boom(result, sem):
        raise RuntimeError("missing pixel_indices")

    monkeypatch.setattr(viewer_mod, "lift_point_features", boom)
    _write_2d_cache(tmp_path)
    viewer = SplitViewer(off_screen=True)
    viewer._result = object()
    viewer._semantics_dir = tmp_path
    viewer.ensure_lifted(None)
    assert viewer._point_features is None
    assert not (tmp_path / f"{EXTRACTOR}_lifted.zarr").exists()
    assert not (tmp_path / f"{EXTRACTOR}_ae.pt").exists()
