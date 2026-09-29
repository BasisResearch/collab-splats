"""Tests for the refine stage: config validation, stage registration, refine_poses."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pycolmap
import pytest
import torch
import zarr

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.wrapper.reconstructor import (
    _STAGE_DEPS,
    _STAGE_ORDER,
    LEAF_STAGES,
    Reconstructor,
)


def _cfg(tmp_path, **pointcloud):
    """Minimal valid config dict; pointcloud kwargs merged over base.yaml defaults."""
    return {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": pointcloud,
    }


def test_validate_config_rejects_ba_with_lc_bool(tmp_path):
    """bundle_adjustment + loop_closure=true must fail loud at construction."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))


def test_validate_config_rejects_ba_with_lc_dict(tmp_path):
    """Dict-form loop_closure ({'enabled': ...} implicit true) is rejected too."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure={"submap_size": 16}))


def test_validate_config_allows_ba_without_lc(tmp_path):
    """BA alone constructs fine — the old NotImplementedError is gone."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=False))
    assert r.config["pointcloud"]["bundle_adjustment"] is True


# ---------------------------------------------------------------------------
# Tests for refine_poses
# ---------------------------------------------------------------------------

def _write_ff_zarr(backend_dir, N=2, H=8, W=8, P=10):
    """Synthetic PointcloudResult persisted to backend_dir/pointcloud.zarr."""
    K = np.tile(
        np.array([[10.0, 0, W / 2], [0, 10.0, H / 2], [0, 0, 1.0]], dtype=np.float32), (N, 1, 1)
    )
    ff = PointcloudResult(
        points=np.random.rand(P, 3).astype(np.float32),
        colors=np.zeros((P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)),
        intrinsics=None,
        model_intrinsics=K,
        image_paths=[Path(f"frame_{i:06d}") for i in range(N)],
        original_coords=np.tile(
            np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (N, 1)
        ),
        model_width=W,
        model_height=H,
        images=torch.zeros(N, 3, H, W),
        confidence=torch.ones(N, H, W),
        world_points=np.zeros((N, H, W, 3), dtype=np.float32),
        depth=np.ones((N, H, W), dtype=np.float32),
        pixel_indices=np.stack(
            [np.zeros(P, dtype=np.int64), np.arange(P) % H, np.arange(P) % W], axis=1
        ),
    )
    backend_dir.mkdir(parents=True, exist_ok=True)
    ff.save_zarr(backend_dir / "pointcloud.zarr")
    return ff


def _reconstructor(tmp_path):
    return Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=False))


def fake_refine(self, images, confidence, world_points, extrinsics, intrinsics, image_paths=None):
    """BA output: translate every camera by +1 in x so refinement is observable."""
    new_ext = extrinsics.copy()
    new_ext[:, 0, 3] += 1.0
    return new_ext, intrinsics


def test_refine_poses_missing_zarr_raises(tmp_path):
    """No pointcloud.zarr → clear FileNotFoundError, not a deep BA stack trace."""
    r = _reconstructor(tmp_path)
    with pytest.raises(FileNotFoundError, match="pointcloud.zarr"):
        r.refine_poses()


def test_refine_poses_skips_when_marker_exists(tmp_path):
    """Existing refine.json without overwrite → skip (no BA), matching other stage methods."""
    r = _reconstructor(tmp_path)
    marker = r.backend_dir / "colmap" / "refine.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    with patch.object(Reconstructor, "_resolve_result", return_value=MagicMock()) as mock_resolve, \
         patch("collab_splats.wrapper.reconstructor.BundleAdjustment.refine") as mock_refine:
        r.refine_poses(overwrite=False)
    mock_refine.assert_not_called()
    mock_resolve.assert_called_once()


def test_refine_poses_refines_and_persists(tmp_path):
    """refine_poses: BA refine + reproject, COLMAP rewritten, zarr updated, marker written."""
    r = _reconstructor(tmp_path)
    ff = _write_ff_zarr(r.backend_dir)

    fake_result = MagicMock()
    with patch("collab_splats.wrapper.reconstructor.BundleAdjustment.refine", fake_refine), \
         patch.object(Reconstructor, "_load_pointcloud_from_disk", return_value=fake_result):
        out = r.refine_poses()

    # COLMAP rewritten with refined poses
    assert (r.backend_dir / "colmap" / "sparse" / "0" / "images.bin").exists()
    # zarr extrinsics updated in place — never diverges from COLMAP
    store = zarr.open(str(r.backend_dir / "pointcloud.zarr"), mode="r")
    np.testing.assert_allclose(store["extrinsics"][:, 0, 3], ff.extrinsics[:, 0, 3] + 1.0)
    # the standard writer refreshed the derived artifact
    fake_result.write_ply.assert_called_once_with(r.backend_dir / "sparse_pc.ply")
    # marker doubles as provenance
    marker = json.loads((r.backend_dir / "colmap" / "refine.json").read_text())
    assert "config" in marker and "loss_history" in marker
    assert out is fake_result


def test_refine_poses_reexports_pinhole(tmp_path):
    """A vggtx scene is re-exported as PINHOLE after BA."""
    recon = Reconstructor(_cfg(tmp_path, backend="vggtx", bundle_adjustment=True, loop_closure=False))
    _write_ff_zarr(recon.backend_dir)
    with patch("collab_splats.wrapper.reconstructor.BundleAdjustment.refine", fake_refine), \
         patch.object(Reconstructor, "_load_pointcloud_from_disk"):
        recon.refine_poses()
    sparse = recon.backend_dir / "colmap" / "sparse" / "0"
    assert {c.model.name for c in pycolmap.Reconstruction(str(sparse)).cameras.values()} == {"PINHOLE"}


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


def test_refine_poses_recleans_after_reproject(tmp_path):
    """A far pixel reprojected by refine is cleaned from the zarr, COLMAP model and PLY alike."""
    r = _reconstructor(tmp_path)
    P = _write_outlier_zarr(r)
    with patch("collab_splats.wrapper.reconstructor.BundleAdjustment.refine", fake_refine):
        r.refine_poses()

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


def test_refine_poses_keeps_every_point_when_clean_disabled(tmp_path):
    """pointcloud.clean.enabled: false leaves the reprojected set whole, outlier included."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=False, clean={"enabled": False}))
    P = _write_outlier_zarr(r)
    with patch("collab_splats.wrapper.reconstructor.BundleAdjustment.refine", fake_refine):
        r.refine_poses()

    store = zarr.open(str(r.backend_dir / "pointcloud.zarr"), mode="r")
    assert store["points"].shape[0] == P
    assert np.abs(store["points"][:]).max() > 100.0


def test_refine_poses_recaps_to_max_points(tmp_path):
    """A max_points lowered since the pointcloud stage caps the refined set on every artifact."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=False, max_points=50))
    _write_outlier_zarr(r)
    with patch("collab_splats.wrapper.reconstructor.BundleAdjustment.refine", fake_refine):
        r.refine_poses()

    store = zarr.open(str(r.backend_dir / "pointcloud.zarr"), mode="r")
    assert store["points"].shape[0] == 50
    assert store["pixel_indices"].shape[0] == 50
    assert pycolmap.Reconstruction(str(r.colmap_model_dir)).num_points3D() == 50


# ---------------------------------------------------------------------------
# Tests for stage registration + triggers
# ---------------------------------------------------------------------------

def test_refine_is_a_leaf_stage():
    """refine must be re-runnable on its own from environments-processed."""
    assert "refine" in _STAGE_ORDER
    assert _STAGE_DEPS["refine"] == ["pointcloud"]
    assert "refine" in LEAF_STAGES


def test_run_pipeline_config_driven_appends_refine(tmp_path):
    """bundle_adjustment: true → refine runs right after pointcloud, before dependents."""
    r = _reconstructor(tmp_path)
    calls = []
    with patch.object(Reconstructor, "preprocess", side_effect=lambda **k: calls.append("preproc")), \
         patch.object(Reconstructor, "build_pointcloud", side_effect=lambda **k: calls.append("pointcloud")), \
         patch.object(Reconstructor, "refine_poses", side_effect=lambda **k: calls.append("refine")), \
         patch.object(Reconstructor, "extract_semantics", side_effect=lambda **k: calls.append("semantics")), \
         patch.object(Reconstructor, "mesh", side_effect=lambda **k: calls.append("mesh")), \
         patch.object(Reconstructor, "build_localization_db", side_effect=lambda **k: calls.append("localize")), \
         patch.object(Reconstructor, "reconstruction_quality_report",
                      side_effect=lambda **k: calls.append("reconstruction_quality_report")):
        r.run_pipeline()
    assert "refine" in calls
    assert calls.index("refine") == calls.index("pointcloud") + 1


def test_run_pipeline_named_refine_refuses_existing_output(tmp_path):
    """Named refine with existing refine.json and no overwrite → refusal (generic leaf rule)."""
    r = _reconstructor(tmp_path)
    # Satisfy the pointcloud dependency and the refine marker on disk
    r.colmap_model_dir.mkdir(parents=True)
    (r.backend_dir / "pointcloud.zarr").mkdir()
    (r.backend_dir / "colmap" / "refine.json").write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        r.run_pipeline(stages=["refine"])
