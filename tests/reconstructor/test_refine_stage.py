"""Tests for the refine stage: config validation, stage registration, the refine body."""

import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import open3d as o3d
import pycolmap
import pytest
import torch
import zarr

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.reconstructor import LEAF_STAGES, STAGES, Reconstructor
from tests.reconstructor._stubs import stub_creator_cls

# BA on vggsfm tracks: refine tests mock BA and write no images/ store
VGGSFM_BA = {"enabled": True, "track_source": "vggsfm"}


def _cfg(tmp_path, **pointcloud):
    """Minimal valid config dict; pointcloud kwargs merged over base.yaml defaults."""
    return {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": pointcloud,
    }


def test_validate_config_accepts_ba_with_lc_bool(tmp_path):
    """bundle_adjustment + loop_closure=true is per-window BA, accepted at construction."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))
    assert r.config["pointcloud"]["bundle_adjustment"]["enabled"] is True


def test_validate_config_accepts_ba_with_lc_dict(tmp_path):
    """Dict-form loop_closure with BA on is accepted too."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure={"submap_size": 16}))
    assert r.config["pointcloud"]["loop_closure"]["enabled"] is True


@pytest.mark.parametrize("source", ["xfeat", "loma"])
def test_validate_config_accepts_matcher_tracks_with_lc(tmp_path, source):
    """Window BA reads the window's full-res frames, so a matcher track_source runs with LC."""
    ba = {"enabled": True, "track_source": source}
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=ba, loop_closure=True))
    assert r.config["pointcloud"]["bundle_adjustment"]["track_source"] == source


def test_validate_config_refuses_sfm_before_matcher_tracks_with_lc(tmp_path):
    """An sfm config with BA, LC and a matcher track_source sees the sfm refusal first."""
    ba = {"enabled": True, "track_source": "xfeat"}
    with pytest.raises(ValueError, match="not supported with method: sfm"):
        Reconstructor(_cfg(tmp_path, method="sfm", backend="instantsfm", bundle_adjustment=ba, loop_closure=True))


def test_run_config_driven_skips_refine_under_lc(tmp_path):
    """BA + LC: BA runs inside the pointcloud stage, so the default stage set drops refine."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))
    calls = []
    with (
        patch.object(Reconstructor, "preproc", side_effect=lambda **kwargs: calls.append("preproc")),
        patch.object(Reconstructor, "pointcloud", side_effect=lambda: calls.append("pointcloud")),
        patch.object(Reconstructor, "refine", side_effect=lambda: calls.append("refine")),
        patch.object(Reconstructor, "semantics", side_effect=lambda: calls.append("semantics")),
        patch.object(Reconstructor, "mesh", side_effect=lambda: calls.append("mesh")),
        patch.object(Reconstructor, "localize", side_effect=lambda: calls.append("localize")),
        patch.object(
            Reconstructor,
            "reconstruction_quality_report",
            side_effect=lambda: calls.append("reconstruction_quality_report"),
        ),
    ):
        r.run()
    assert "pointcloud" in calls
    assert "refine" not in calls


def test_refine_refuses_under_lc(tmp_path):
    """An explicit refine under LC would re-solve a store BA already ran in."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))
    with pytest.raises(ValueError, match="refine is not supported with pointcloud.loop_closure"):
        r.refine()


def test_validate_config_allows_ba_without_lc(tmp_path):
    """BA alone constructs fine — the old NotImplementedError is gone."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=False))
    assert r.config["pointcloud"]["bundle_adjustment"] == {"enabled": True}


# ---------------------------------------------------------------------------
# Tests for the refine stage body
# ---------------------------------------------------------------------------


def _write_ff_zarr(backend_dir, N=2, H=8, W=8, P=10):
    """Synthetic PointcloudResult persisted to backend_dir/pointcloud.zarr."""
    K = np.tile(np.array([[10.0, 0, W / 2], [0, 10.0, H / 2], [0, 0, 1.0]], dtype=np.float32), (N, 1, 1))
    ff = PointcloudResult(
        points=np.random.rand(P, 3).astype(np.float32),
        colors=np.zeros((P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)),
        intrinsics=None,
        model_intrinsics=K,
        image_paths=[Path(f"frame_{i:06d}") for i in range(N)],
        original_coords=np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (N, 1)),
        model_width=W,
        model_height=H,
        images=torch.zeros(N, 3, H, W),
        confidence=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
        depth=np.ones((N, H, W), dtype=np.float32),
        pixel_indices=np.stack([np.zeros(P, dtype=np.int64), np.arange(P) % H, np.arange(P) % W], axis=1),
    )
    backend_dir.mkdir(parents=True, exist_ok=True)
    ff.save_zarr(backend_dir / "pointcloud.zarr")
    return ff


def _reconstructor(tmp_path):
    return Reconstructor(_cfg(tmp_path, bundle_adjustment=VGGSFM_BA, loop_closure=False))


def fake_refine(self, images, confidence, world_points, extrinsics, intrinsics, depth=None, **kwargs):
    """BA output: translate every camera by +1 in x so refinement is observable."""
    new_ext = extrinsics.copy()
    new_ext[:, 0, 3] += 1.0
    return new_ext, intrinsics


def test_refine_missing_zarr_raises(tmp_path):
    """No pointcloud.zarr → clear FileNotFoundError, not a deep BA stack trace."""
    r = _reconstructor(tmp_path)
    with pytest.raises(FileNotFoundError, match="pointcloud.zarr"):
        r.refine()


def test_refine_skips_in_config_run_when_marker_exists(tmp_path):
    """Existing refine.json in a config-driven run → skip (no BA), like every done stage."""
    r = _reconstructor(tmp_path)
    r.images_dir.mkdir(parents=True)
    r.colmap_model_dir.mkdir(parents=True)
    (r.backend_dir / "pointcloud.zarr").mkdir()
    (r.backend_dir / "colmap" / "refine.json").write_text("{}")
    with (
        patch.object(Reconstructor, "semantics"),
        patch.object(Reconstructor, "mesh"),
        patch.object(Reconstructor, "reconstruction_quality_report"),
        patch("collab_splats.reconstructor.BundleAdjustment.refine") as mock_refine,
    ):
        r.run()
    mock_refine.assert_not_called()


def test_refine_refines_and_persists(tmp_path):
    """refine: BA refine + reproject, COLMAP rewritten, zarr rewritten with its attrs, marker written."""
    r = _reconstructor(tmp_path)
    ff = _write_ff_zarr(r.backend_dir)
    zarr.open_group(str(r.pointcloud_zarr), mode="r+").attrs.update({"method": "feedforward", "backend": "vggt_omega"})

    with patch("collab_splats.reconstructor.BundleAdjustment.refine", fake_refine):
        r.refine()

    # COLMAP rewritten with refined poses
    assert (r.backend_dir / "colmap" / "sparse" / "0" / "images.bin").exists()

    # zarr extrinsics rewritten — never diverges from COLMAP — and the provenance attrs survive
    store = zarr.open(str(r.pointcloud_zarr), mode="r")
    np.testing.assert_allclose(store["extrinsics"][:, 0, 3], ff.extrinsics[:, 0, 3] + 1.0)
    assert store.attrs["backend"] == "vggt_omega"

    # The reloaded result and the PLY hold the cleaned set the zarr holds
    assert len(r.result.points) == store["points"].shape[0]
    ply = o3d.io.read_point_cloud(str(r.backend_dir / "sparse_pc.ply"))
    assert len(ply.points) == store["points"].shape[0]

    # marker doubles as provenance
    marker = json.loads((r.backend_dir / "colmap" / "refine.json").read_text())
    assert {"config", "loss_history", "losses", "alignment_scale"} <= set(marker)


def test_refine_vggsfm_source_passes_no_frame_paths(tmp_path):
    """The vggsfm source never reads images/: refine gets frame_paths=None."""
    r = _reconstructor(tmp_path)
    _write_ff_zarr(r.backend_dir)
    seen = {}

    def capture(self, *args, frame_paths=None, **kwargs):
        seen["frame_paths"] = frame_paths
        return fake_refine(self, *args, **kwargs)

    with patch("collab_splats.reconstructor.BundleAdjustment.refine", capture):
        r.refine()

    assert seen == {"frame_paths": None}


def test_refine_matcher_source_passes_store_frames_in_zarr_order(tmp_path):
    """A matcher track source gets the full-res images/ frame per zarr frame, joined on frame index."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment={"enabled": True, "track_source": "xfeat"}, loop_closure=False))
    _write_ff_zarr(r.backend_dir)
    r.images_dir.mkdir(parents=True)

    for idx in (0, 1, 5):
        (r.images_dir / f"frame_{idx:06d}.png").write_bytes(b"")

    seen = {}

    def capture(self, *args, frame_paths=None, **kwargs):
        seen["frame_paths"] = frame_paths
        return fake_refine(self, *args, **kwargs)

    with patch("collab_splats.reconstructor.BundleAdjustment.refine", capture):
        r.refine()

    assert seen["frame_paths"] == [r.images_dir / "frame_000000.png", r.images_dir / "frame_000001.png"]


def test_refine_matcher_source_without_images_dir_raises(tmp_path):
    """A matcher track source with no images/ store fails with FileNotFoundError, not a KeyError."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment={"enabled": True, "track_source": "xfeat"}, loop_closure=False))
    _write_ff_zarr(r.backend_dir)

    with patch("collab_splats.reconstructor.BundleAdjustment.refine", fake_refine):
        with pytest.raises(FileNotFoundError, match="images/"):
            r.refine()


def test_refine_matcher_source_checks_images_dir_before_loading_zarr(tmp_path):
    """The images/ guard fires before the zarr load, so a missing store is reported even with no zarr."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment={"enabled": True, "track_source": "xfeat"}, loop_closure=False))

    with patch("collab_splats.reconstructor.PointcloudResult.load_zarr") as load_zarr:
        with pytest.raises(FileNotFoundError, match="images/"):
            r.refine()

    load_zarr.assert_not_called()


def test_refine_reexports_pinhole(tmp_path):
    """A vggtx scene is re-exported as PINHOLE after BA."""
    recon = Reconstructor(_cfg(tmp_path, backend="vggtx", bundle_adjustment=VGGSFM_BA, loop_closure=False))
    _write_ff_zarr(recon.backend_dir)
    with (
        patch("collab_splats.reconstructor.BundleAdjustment.refine", fake_refine),
        patch.object(Reconstructor, "result", new_callable=PropertyMock),
    ):
        recon.refine()
    sparse = recon.backend_dir / "colmap" / "sparse" / "0"
    cams = list(pycolmap.Reconstruction(str(sparse)).cameras.values())
    assert {c.model.name for c in cams} == {"PINHOLE"}

    # Full-res K: center cx=4 rescales 8x to 35.5; COLMAP corner export adds 0.5
    assert all(np.allclose(c.params, [80, 80, 36, 36]) for c in cams)


def test_pointcloud_rerun_clears_a_stale_refine_marker(tmp_path):
    """A re-run pointcloud stage removes refine.json, so refine is not reported done on an unrefined zarr."""
    r = _reconstructor(tmp_path)
    marker = r.backend_dir / "colmap" / "refine.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")

    with patch("collab_splats.reconstructor.get_creator", return_value=stub_creator_cls(MagicMock())):
        r.pointcloud()

    assert not marker.exists()


def _write_outlier_zarr(r):
    """
    Refine fixture with 200 distinct depth-1 pixels, one of them pushed to depth 1000.

    - rows written in place: every array keeps the shape _write_ff_zarr gave it
    """
    P, H, W = 200, 16, 16
    _write_ff_zarr(r.backend_dir, H=H, W=W, P=P)
    store = zarr.open(str(r.backend_dir / "pointcloud.zarr"), mode="r+")

    # 200 distinct pixels of frame 0, so no two points coincide
    idx = np.arange(P)
    store["pixel_indices"][:] = np.stack([np.zeros(P, dtype=np.int64), idx // W, idx % W], axis=1)

    # One far depth pixel: an outlier only once reproject rebuilds the points from depth
    depth = np.ones((2, H, W), dtype=np.float32)
    depth[0, 5, 5] = 1000.0
    store["depth"][:] = depth
    return P


def test_refine_recleans_after_reproject(tmp_path):
    """A far pixel reprojected by refine is cleaned from the zarr, COLMAP model and PLY alike."""
    r = _reconstructor(tmp_path)
    P = _write_outlier_zarr(r)
    with patch("collab_splats.reconstructor.BundleAdjustment.refine", fake_refine):
        r.refine()

    # zarr per-point arrays are resized together and the far point is gone
    store = zarr.open(str(r.backend_dir / "pointcloud.zarr"), mode="r")
    points = store["points"][:]
    assert len(points) < P
    assert np.abs(points).max() < 10.0
    assert store["colors"].shape[0] == len(points)
    assert store["pixel_indices"].shape[0] == len(points)

    # COLMAP model and sparse_pc.ply hold the same cleaned set
    recon = pycolmap.Reconstruction(str(r.colmap_model_dir))
    assert recon.num_points3D() == len(points)
    ply = o3d.io.read_point_cloud(str(r.backend_dir / "sparse_pc.ply"))
    assert len(ply.points) == len(points)


def test_refine_keeps_every_point_when_clean_disabled(tmp_path):
    """pointcloud.clean.enabled: false leaves the reprojected set whole, outlier included."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=VGGSFM_BA, loop_closure=False, clean={"enabled": False}))
    P = _write_outlier_zarr(r)
    with patch("collab_splats.reconstructor.BundleAdjustment.refine", fake_refine):
        r.refine()

    store = zarr.open(str(r.backend_dir / "pointcloud.zarr"), mode="r")
    assert store["points"].shape[0] == P
    assert np.abs(store["points"][:]).max() > 100.0


def test_refine_recaps_to_max_points(tmp_path):
    """A max_points lowered since the pointcloud stage caps the refined set on every artifact."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=VGGSFM_BA, loop_closure=False, max_points=50))
    _write_outlier_zarr(r)
    with patch("collab_splats.reconstructor.BundleAdjustment.refine", fake_refine):
        r.refine()

    store = zarr.open(str(r.backend_dir / "pointcloud.zarr"), mode="r")
    assert store["points"].shape[0] == 50
    assert store["pixel_indices"].shape[0] == 50
    assert pycolmap.Reconstruction(str(r.colmap_model_dir)).num_points3D() == 50


# ---------------------------------------------------------------------------
# Tests for stage registration + triggers
# ---------------------------------------------------------------------------


def test_refine_is_a_leaf_stage():
    """refine must be re-runnable on its own from environments-processed."""
    assert STAGES["refine"] == ("pointcloud",)
    assert "refine" in LEAF_STAGES


def test_run_config_driven_appends_refine(tmp_path):
    """bundle_adjustment: true → refine runs right after pointcloud, before dependents."""
    r = _reconstructor(tmp_path)
    calls = []
    with (
        patch.object(Reconstructor, "preproc", side_effect=lambda **kwargs: calls.append("preproc")),
        patch.object(Reconstructor, "pointcloud", side_effect=lambda: calls.append("pointcloud")),
        patch.object(Reconstructor, "refine", side_effect=lambda: calls.append("refine")),
        patch.object(Reconstructor, "semantics", side_effect=lambda: calls.append("semantics")),
        patch.object(Reconstructor, "mesh", side_effect=lambda: calls.append("mesh")),
        patch.object(Reconstructor, "localize", side_effect=lambda: calls.append("localize")),
        patch.object(
            Reconstructor,
            "reconstruction_quality_report",
            side_effect=lambda: calls.append("reconstruction_quality_report"),
        ),
    ):
        r.run()
    assert "refine" in calls
    assert calls.index("refine") == calls.index("pointcloud") + 1


def test_run_named_refine_refuses_existing_output(tmp_path):
    """Named refine with existing refine.json and no overwrite → refusal (generic leaf rule)."""
    r = _reconstructor(tmp_path)
    # Satisfy the pointcloud dependency and the refine marker on disk
    r.colmap_model_dir.mkdir(parents=True)
    (r.backend_dir / "pointcloud.zarr").mkdir()
    (r.backend_dir / "colmap" / "refine.json").write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        r.run(["refine"])


def test_refine_refuses_sfm_method(tmp_path):
    """refine refuses outright when pointcloud.method is sfm; the guard is its first line."""
    r = Reconstructor(_cfg(tmp_path, method="sfm", backend="instantsfm"))
    with pytest.raises(ValueError, match="refine is not supported"):
        r.refine()


@pytest.mark.parametrize(
    "block,match",
    [
        ({"use_photometric": True, "increment_size": 16}, "pointcloud.bundle_adjustment: .*use_photometric needs"),
        ({"fit_depth_scale": True}, "unknown keys"),
        ({"dtype": "float16"}, "pointcloud.bundle_adjustment: .*dtype must be"),
    ],
)
def test_validate_config_refuses_bad_ba_terms(tmp_path, block, match):
    """Each invalid BA term combination fails at construction."""
    with pytest.raises(ValueError, match=match):
        Reconstructor(_cfg(tmp_path, bundle_adjustment=block, loop_closure=False))
