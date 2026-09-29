"""
Reconstructor wiring for the sfm stage and the frame subset it leaves.

- creator: built from the config block plus pointcloud.clean.enabled and max_points, seed forwarded
- create_pointcloud: handed the scene's images/, backend dir and colmap/sparse/0
- attrs: the backend plus the creator's own, stamped on pointcloud.zarr
- downstream: reload, semantics lift, localization DB and mesh see only the zarr's frames
"""

import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import zarr

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.preproc import frames as fr
from collab_splats.wrapper.reconstructor import (
    Reconstructor,
    _build_localization_db,
    _lift_and_save,
    _run_tsdf_mesh,
)
from tests.wrapper._stubs import minimal_feedforward_result, minimal_pose_result

RECONSTRUCTOR = "collab_splats.wrapper.reconstructor"

# _run_sfm never opens the video: input_path only reaches config validation (a presence check)
# and the store's provenance stamp, so a stand-in path is enough — no encoded fixture needed.
VIDEO_PATH = "/nonexistent/tiny.mp4"

# Source frame indices the fake store is built on. Non-contiguous on purpose: a leg that
# confused source frame indices with store rows would show up.
FRAME_IDX = (0, 9, 30, 57)


########################################################################
# Fixtures and helpers
########################################################################


def _sfm_reconstructor(tmp_path, *, random_seed=None, backend="instantsfm", clean=True):
    """
    A method:sfm Reconstructor backed by a real images/ store holding FRAME_IDX keyframes.
    """
    config = {
        "input_path": VIDEO_PATH,
        "output_path": str(tmp_path / "out"),
        "pointcloud": {
            "method": "sfm",
            "backend": backend,
            "instantsfm": {"random_seed": random_seed},
            "clean": {"enabled": clean},
        },
    }
    recon = Reconstructor(config)

    # Real store: the downstream-stage tests read the PNGs off it
    frames = [np.full((8, 8, 3), i, np.uint8) for i in FRAME_IDX]
    records = [{"frame_idx": int(i), "blur_score": 1.0} for i in FRAME_IDX]
    fr.write_frames(
        recon.images_dir,
        frames,
        records,
        {"video_path": VIDEO_PATH, "method": "uniform"},
    )
    return recon


def _run_sfm_with_mocks(recon, create=None, attrs=None):
    """
    Execute _run_sfm with a mock creator class for the recon's backend; returns that class.

    - create: side effect for creator.create_pointcloud; default returns a light MagicMock result
    - attrs: the creator's attrs after create_pointcloud
    """
    backend = recon.config["pointcloud"]["backend"]
    outputs = MagicMock()
    outputs.points = np.zeros((3, 3), np.float32)
    creator_cls = MagicMock()
    creator_cls.return_value.create_pointcloud.side_effect = create or (lambda *a: outputs)
    creator_cls.return_value.attrs = attrs or {"method": "sfm"}

    # patch.dict restores SFM_CREATORS in place on exit; the mock class is kept by name
    with patch.dict(f"{RECONSTRUCTOR}.SFM_CREATORS", {backend: creator_cls}):
        recon._run_sfm()
    return creator_cls


def test_run_sfm_frees_the_dense_arrays_after_save(tmp_path):
    """
    The sfm result is light once pointcloud.zarr is written: no dense array stays alive.
    """
    recon = _sfm_reconstructor(tmp_path)
    refs = {}
    out = {}

    # A real dense result per call, reached only through weakrefs from here
    def _dense_result(*args):
        result = minimal_feedforward_result()
        refs["depth"] = weakref.ref(result.depth)
        refs["images"] = weakref.ref(result.images)
        out["result"] = result
        return result

    _run_sfm_with_mocks(recon, create=_dense_result)

    # Freed by refcount alone; the zarr keeps the depth for the downstream stages
    assert refs["depth"]() is None and refs["images"]() is None
    assert out["result"].depth is None and out["result"].points.shape == (5, 3)
    assert zarr.open(str(recon.pointcloud_zarr), mode="r")["depth"].shape == (2, 8, 8)


########################################################################
# _run_sfm: config pass-through
########################################################################


def test_run_sfm_forwards_random_seed_from_the_config(tmp_path):
    recon = _sfm_reconstructor(tmp_path, random_seed=7)
    creator_cls = _run_sfm_with_mocks(recon)
    assert creator_cls.call_args.kwargs["random_seed"] == 7


@pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
@pytest.mark.parametrize("clean", [True, False])
def test_run_sfm_builds_the_creator_from_the_block_and_the_clean_switch(tmp_path, backend, clean):
    recon = _sfm_reconstructor(tmp_path, backend=backend, clean=clean)
    creator_cls = _run_sfm_with_mocks(recon)
    pc_cfg = recon.config["pointcloud"]
    assert creator_cls.call_args.kwargs == {"clean": clean, "max_points": pc_cfg["max_points"], **pc_cfg[backend]}
    assert creator_cls.call_args.kwargs["min_registered_frac"] == 0.5


def test_run_sfm_hands_create_the_scene_dirs(tmp_path):
    recon = _sfm_reconstructor(tmp_path)
    creator_cls = _run_sfm_with_mocks(recon)
    creator_cls.return_value.create_pointcloud.assert_called_once_with(
        recon.images_dir, recon.backend_dir, recon.colmap_model_dir
    )


def test_run_sfm_stamps_the_backend_and_the_creator_attrs(tmp_path):
    recon = _sfm_reconstructor(tmp_path, backend="colmap")
    attrs = {"method": "sfm", "registered_frames": 3, "total_frames": 4}
    saved = {}

    # Capture the attrs save_zarr receives
    def _result(*args):
        result = MagicMock()
        result.points = np.zeros((3, 3), np.float32)
        result.save_zarr.side_effect = lambda path, extra_attrs: saved.update(extra_attrs)
        return result

    _run_sfm_with_mocks(recon, create=_result, attrs=attrs)
    assert saved == {"backend": "colmap", **attrs}


########################################################################
# Downstream stages on a registered subset: images/ holds 4, the model 3
########################################################################

# The frames a colmap / hloc model kept; images/ still holds all of FRAME_IDX
KEPT_IDX = (0, 30, 57)


def _seed_subset_scene(tmp_path):
    """
    A finished colmap run on disk that registered KEPT_IDX out of the FRAME_IDX store.

    - pointcloud.zarr holds only the KEPT_IDX rows, stems as names — what _run_sfm saves
    - colmap/sparse/0 exists, as the stage-exists check requires; nothing reads it back
    """
    recon = _sfm_reconstructor(tmp_path, backend="colmap")
    _subset_result().save_zarr(recon.pointcloud_zarr)
    recon.colmap_model_dir.mkdir(parents=True, exist_ok=True)
    return recon


def _subset_result():
    """
    A PointcloudResult double over the KEPT_IDX rows, as load_zarr would return it.
    """
    result = minimal_feedforward_result(n=len(KEPT_IDX))
    result.image_paths = [Path(f"frame_{i:06d}") for i in KEPT_IDX]
    return result


def test_load_pointcloud_from_disk_reads_the_subset_the_zarr_holds(tmp_path):
    recon = _seed_subset_scene(tmp_path)

    # A re-run without overwrite takes the load-from-disk branch: the zarr's rows, not images/'s
    result = recon.build_pointcloud()
    assert [p.name for p in result.image_paths] == [f"frame_{i:06d}" for i in KEPT_IDX]
    assert result.extrinsics.shape == (3, 4, 4)


def test_resolve_result_reads_the_subset_the_zarr_holds(tmp_path):
    # A leaf stage run on its own resolves the result through the same loader
    result = _seed_subset_scene(tmp_path)._resolve_result()
    assert [p.name for p in result.image_paths] == [f"frame_{i:06d}" for i in KEPT_IDX]


def test_semantics_lifts_only_the_rows_the_pointcloud_holds(tmp_path):
    recon = _seed_subset_scene(tmp_path)

    # Scene-level 2D cache over all four images/ frames; row r is constant FRAME_IDX[r]
    cache = tmp_path / "dinov2.zarr"
    zarr.open(str(cache), mode="w")["features"] = np.stack([np.full((4, 2, 2), i, np.float32) for i in FRAME_IDX])

    lifted = {}

    def spy_lift(feature_maps, result):
        lifted["rows"] = [float(fm[0, 0, 0]) for fm in feature_maps]
        return torch.zeros(5, 4)

    with (
        patch.object(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: _subset_result())),
        patch(f"{RECONSTRUCTOR}.lift_features", side_effect=spy_lift),
    ):
        _lift_and_save(
            "dinov2",
            cache,
            recon.pointcloud_zarr,
            tmp_path / "semantics",
            None,
            target_cosine=None,
            max_epochs=1,
            images_dir=recon.images_dir,
        )

    # The real lift_features pairs map i with zarr row i: three maps, in the zarr's order
    assert lifted["rows"] == [float(i) for i in KEPT_IDX]


def test_localization_db_pairs_zarr_rows_with_their_own_frames(tmp_path):
    recon = _seed_subset_scene(tmp_path)
    seen = {}

    def spy_from_feedforward(result, *, images, ids, **kwargs):
        seen["ids"] = ids
        seen["pixels"] = [int(image[0, 0, 0]) for image in images]

    with (
        patch.object(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: _subset_result())),
        patch("collab_splats.localization.extractors.LocalMatcher"),
        patch(
            "collab_splats.localization.localizer.CameraLocalizer.from_feedforward", side_effect=spy_from_feedforward
        ),
    ):
        _build_localization_db(recon.pointcloud_zarr, "loma", recon.images_dir)

    # from_feedforward pairs image i with geometry row i: M ids, M frames, zarr order
    assert seen["ids"] == [f"frame_{i:06d}.png" for i in KEPT_IDX]
    assert seen["pixels"] == list(KEPT_IDX)


def test_mesh_fuses_the_frames_the_zarr_rows_came_from(tmp_path):
    recon = _seed_subset_scene(tmp_path)
    fused = {}

    def spy_fuse(depths, rgbs, c2w, K, out_dir, **kwargs):
        fused["rgbs"] = rgbs
        return tmp_path / "mesh.ply"

    with (
        patch.object(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: _subset_result())),
        patch(f"{RECONSTRUCTOR}.upsample_depths", side_effect=lambda d, r, b: d),
        patch(f"{RECONSTRUCTOR}.create_tsdf_mesh", side_effect=spy_fuse),
        patch(f"{RECONSTRUCTOR}.clean_repair_mesh"),
    ):
        _run_tsdf_mesh(
            result=minimal_pose_result(n=len(KEPT_IDX)),
            pointcloud_zarr=recon.pointcloud_zarr,
            output_dir=tmp_path,
            images_dir=recon.images_dir,
            voxel_size=0.01,
            depth_trunc=2.0,
        )

    # Depth row i fuses with the RGB of the frame it was predicted on, not images/ row i
    assert fused["rgbs"][:, 0, 0, 0].tolist() == list(KEPT_IDX)
