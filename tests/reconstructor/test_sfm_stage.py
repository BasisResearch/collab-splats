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
from collab_splats.reconstructor import Reconstructor, _build_localization_db
from tests.reconstructor._stubs import minimal_feedforward_result

RECONSTRUCTOR = "collab_splats.reconstructor"

# The sfm stage never opens the video, so a stand-in input_path is enough
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
    fr.write_frames(recon.images_dir, frames, FRAME_IDX)
    return recon


def _run_stage_with_sfm_mock(recon, create=None, attrs=None):
    """
    Run the pointcloud stage with a mock sfm creator class for the recon's backend; returns that class.

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
        recon.pointcloud()
    return creator_cls


def test_sfm_stage_frees_the_dense_arrays_after_save(tmp_path):
    """
    The sfm result is light once pointcloud.zarr is written: no dense array stays alive.
    """
    recon = _sfm_reconstructor(tmp_path)
    refs = {}

    # A real dense result per call, reached only through weakrefs from here
    def _dense_result(*args):
        result = minimal_feedforward_result()
        refs["depth"] = weakref.ref(result.depth)
        refs["images"] = weakref.ref(result.images)
        return result

    _run_stage_with_sfm_mock(recon, create=_dense_result)

    # Freed by refcount alone; the zarr keeps the depth for the downstream stages
    assert refs["depth"]() is None and refs["images"]() is None
    assert recon.result.points.shape == (5, 3)
    assert zarr.open(str(recon.pointcloud_zarr), mode="r")["depth"].shape == (2, 8, 8)


########################################################################
# sfm stage: config pass-through
########################################################################


def test_sfm_stage_forwards_random_seed_from_the_config(tmp_path):
    recon = _sfm_reconstructor(tmp_path, random_seed=7)
    creator_cls = _run_stage_with_sfm_mock(recon)
    assert creator_cls.call_args.kwargs["random_seed"] == 7


@pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
@pytest.mark.parametrize("clean", [True, False])
def test_sfm_stage_builds_the_creator_from_the_block_and_the_clean_switch(tmp_path, backend, clean):
    recon = _sfm_reconstructor(tmp_path, backend=backend, clean=clean)
    creator_cls = _run_stage_with_sfm_mock(recon)
    pc_cfg = recon.config["pointcloud"]
    assert creator_cls.call_args.kwargs == {"clean": clean, "max_points": pc_cfg["max_points"], **pc_cfg[backend]}
    assert creator_cls.call_args.kwargs["min_registered_frac"] == 0.5


def test_sfm_stage_hands_create_the_scene_dirs(tmp_path):
    recon = _sfm_reconstructor(tmp_path)
    creator_cls = _run_stage_with_sfm_mock(recon)
    creator_cls.return_value.create_pointcloud.assert_called_once_with(
        recon.images_dir, recon.backend_dir, recon.colmap_model_dir
    )


def test_sfm_stage_stamps_the_backend_and_the_creator_attrs(tmp_path):
    recon = _sfm_reconstructor(tmp_path, backend="colmap")
    attrs = {"method": "sfm", "registered_frames": 3, "total_frames": 4}
    saved = {}

    # Capture the attrs save_zarr receives
    def _result(*args):
        result = MagicMock()
        result.points = np.zeros((3, 3), np.float32)
        result.save_zarr.side_effect = lambda path, extra_attrs: saved.update(extra_attrs)
        return result

    _run_stage_with_sfm_mock(recon, create=_result, attrs=attrs)
    assert saved == {"backend": "colmap", **attrs}


########################################################################
# Downstream stages on a registered subset: images/ holds 4, the model 3
########################################################################

# The frames a colmap / hloc model kept; images/ still holds all of FRAME_IDX
KEPT_IDX = (0, 30, 57)


def _seed_subset_scene(tmp_path):
    """
    A finished colmap run on disk that registered KEPT_IDX out of the FRAME_IDX store.

    - pointcloud.zarr holds only the KEPT_IDX rows, stems as names — what the sfm stage saves
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


def test_result_reads_the_subset_the_zarr_holds(tmp_path):
    recon = _seed_subset_scene(tmp_path)

    # A leaf stage run on its own loads the result from disk: the zarr's rows, not images/'s
    result = recon.result
    assert [p.name for p in result.image_paths] == [f"frame_{i:06d}" for i in KEPT_IDX]
    assert result.extrinsics.shape == (3, 4, 4)


def test_semantics_lifts_only_the_rows_the_pointcloud_holds(tmp_path):
    recon = _seed_subset_scene(tmp_path)

    # Scene-level 2D cache over all four images/ frames; row r is constant FRAME_IDX[r]
    cache = tmp_path / "dinov2.zarr"
    zarr.open(str(cache), mode="w")["features"] = np.stack([np.full((4, 2, 2), i, np.float32) for i in FRAME_IDX])

    lifted = {}

    def spy_lift(feature_maps, result):
        lifted["rows"] = [float(feature_maps(i)[0, 0, 0]) for i in range(len(KEPT_IDX))]
        return torch.zeros(5, 4)

    # Uncompressed, extractor and writer stubbed: only the row pick and the lift run
    recon.config["semantics"]["n_components"] = None

    with (
        patch.object(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: _subset_result())),
        patch(f"{RECONSTRUCTOR}.BaseFeatureExtractor"),
        patch(f"{RECONSTRUCTOR}.extract_feature_cache", return_value=cache),
        patch(f"{RECONSTRUCTOR}.lift_features", side_effect=spy_lift),
        patch(f"{RECONSTRUCTOR}.write_point_features"),
    ):
        recon.semantics()

    # The real lift_features pairs map i with zarr row i: three maps, in the zarr's order
    assert lifted["rows"] == [float(i) for i in KEPT_IDX]


def test_localization_db_pairs_zarr_rows_with_their_own_frames(tmp_path):
    recon = _seed_subset_scene(tmp_path)
    seen = {}

    def spy_from_pointcloud(result, *, zarr_path, images, **kwargs):
        seen["pixels"] = [int(image[0, 0, 0]) for image in images]

    with (
        patch.object(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: _subset_result())),
        patch(f"{RECONSTRUCTOR}.LocalMatcher"),
        patch("collab_splats.localization.localizer.CameraLocalizer.from_pointcloud", side_effect=spy_from_pointcloud),
    ):
        _build_localization_db(recon.pointcloud_zarr, "loma", recon.images_dir)

    # from_pointcloud pairs image i with geometry row i: M frames, in zarr order
    assert seen["pixels"] == list(KEPT_IDX)


def test_localization_db_reads_full_res_store_frames_not_model_images(tmp_path):
    recon = _seed_subset_scene(tmp_path)
    seen = {}

    # Model grid 4x6 against 8x8 store frames: only a store read yields 8x8
    model_res = minimal_feedforward_result(n=len(KEPT_IDX), h=4, w=6)
    model_res.image_paths = [Path(f"frame_{i:06d}") for i in KEPT_IDX]
    load_zarr = MagicMock(return_value=model_res)

    def spy_from_pointcloud(result, *, zarr_path, images, **kwargs):
        seen["shapes"] = [image.shape for image in images]

    with (
        patch.object(PointcloudResult, "load_zarr", load_zarr),
        patch(f"{RECONSTRUCTOR}.LocalMatcher"),
        patch("collab_splats.localization.localizer.CameraLocalizer.from_pointcloud", side_effect=spy_from_pointcloud),
    ):
        _build_localization_db(recon.pointcloud_zarr, "loma", recon.images_dir)

    assert seen["shapes"] == [(8, 8, 3)] * len(KEPT_IDX)
    assert not load_zarr.call_args.kwargs.get("load_images", False)


def test_mesh_fuses_the_frames_the_zarr_rows_came_from(tmp_path):
    recon = _seed_subset_scene(tmp_path)
    fused = {}

    def spy_fuse(depths, rgbs, c2w, K, out_dir, **kwargs):
        fused["rgbs"] = rgbs
        return tmp_path / "mesh.ply"

    with (
        patch.object(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: _subset_result())),
        patch("collab_splats.pointcloud.utils.upsample_depths", side_effect=lambda d, r, b: d),
        patch(f"{RECONSTRUCTOR}.create_tsdf_mesh", side_effect=spy_fuse),
        patch(f"{RECONSTRUCTOR}.clean_repair_mesh"),
        patch(f"{RECONSTRUCTOR}.prepare_mesh", side_effect=lambda mesh, **kw: mesh),
    ):
        recon.mesh()

    # Depth row i fuses with the RGB of the frame it was predicted on, not images/ row i
    assert fused["rgbs"][:, 0, 0, 0].tolist() == list(KEPT_IDX)
