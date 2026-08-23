"""Tests for Reconstructor._sfm_result_from_reconstruction (InstantSfM model → FeedforwardResult)."""

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import Reconstructor

ORIG_W, ORIG_H = 64, 48  # staged-jpg / store resolution
DEPTH_W, DEPTH_H = 16, 12  # VDA depth grid (4x downscale)
K_PARAMS = [50.0, 50.0, 32.0, 24.0]  # fx, fy, cx, cy at ORIG res


def _recon(names):
    """
    One PINHOLE camera at ORIG res, one image per name (identity-ish poses), one point3D
    observed in every image — enough structure for the builder's every branch.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=ORIG_W, height=ORIG_H, params=K_PARAMS, camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    track = pycolmap.Track()
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.array([40.0, 20.0]))]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
        track.add_element(i + 1, 0)
    recon.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], dtype=np.uint8))
    return recon


def _store(tmp_path, frame_idxs):
    """
    frames.zarr with one flat-coloured uint8 frame per source index.
    """
    frames = [np.full((ORIG_H, ORIG_W, 3), 40 * (i + 1), dtype=np.uint8) for i in range(len(frame_idxs))]
    records = [{"frame_idx": fi} for fi in frame_idxs]
    return FrameStore.create(tmp_path / "frames.zarr", frames, records, provenance={})


def _write_depths(backend_dir, names):
    """
    depth_vda/images/npy/<name>.npy at the DEPTH grid, constant depth 2.0.
    """
    npy_dir = backend_dir / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for name in names:
        np.save(npy_dir / f"{name}.npy", np.full((DEPTH_H, DEPTH_W), 2.0, dtype=np.float32))


def _reconstructor(tmp_path):
    r = Reconstructor.__new__(Reconstructor)
    r.config = {"output_path": str(tmp_path), "pointcloud": {"method": "sfm", "backend": "instantsfm"}}
    return r


def test_sfm_result_builder_shapes_and_scaling(tmp_path):
    names = ["frame_000000", "frame_000003"]
    recon = _recon(names)
    store = _store(tmp_path, [0, 3])
    _write_depths(tmp_path, names)

    out = _reconstructor(tmp_path)._sfm_result_from_reconstruction(recon, tmp_path, store)

    # Depth grid defines the model resolution
    assert out.depth.shape == (2, DEPTH_H, DEPTH_W)
    assert out.depth.dtype == np.float32
    assert (out.model_width, out.model_height) == (DEPTH_W, DEPTH_H)

    # K rescaled from staged-jpg res to the depth grid
    assert out.intrinsics.shape == (2, 3, 3)
    assert out.intrinsics[0, 0, 0] == pytest.approx(K_PARAMS[0] * DEPTH_W / ORIG_W)
    assert out.intrinsics[0, 1, 2] == pytest.approx(K_PARAMS[3] * DEPTH_H / ORIG_H)

    # Poses homogeneous w2c, rows in store order
    assert out.extrinsics.shape == (2, 4, 4)
    assert np.allclose(out.extrinsics[:, 3], [0, 0, 0, 1])
    assert out.extrinsics[1, 2, 3] == pytest.approx(1.0)

    # Images resized to the depth grid, CHW float in [0, 1]
    assert out.images.shape == (2, 3, DEPTH_H, DEPTH_W)
    assert out.images.dtype == np.float32
    assert 0.0 <= out.images.min() and out.images.max() <= 1.0
    assert out.images[1].mean() == pytest.approx(80 / 255.0, abs=1e-3)

    # Dense world points + sparse points / colors / pixel_indices
    assert out.world_points.shape == (2, DEPTH_H, DEPTH_W, 3)
    assert out.points.shape == (1, 3) and out.points.dtype == np.float32
    assert out.colors.shape == (1, 3) and out.colors.dtype == np.uint8
    assert out.pixel_indices.shape == (1, 3)
    # keypoint (40, 20) at ORIG res -> (col 10, row 5) on the depth grid, observed in row 0
    assert out.pixel_indices.tolist() == [[0, 5, 10]]

    # Contract names, no crop, absent confidence
    assert out.image_paths == [Path("frame_000000"), Path("frame_000003")]
    assert out.original_coords.tolist()[0] == [0, 0, DEPTH_W, DEPTH_H, ORIG_W, ORIG_H]
    assert out.confidence is None and out.mv_ratio is None


def test_sfm_result_builder_refuses_partial_registration(tmp_path):
    # 3 store frames, only 2 registered
    names = ["frame_000000", "frame_000003"]
    recon = _recon(names)
    store = _store(tmp_path, [0, 3, 6])
    _write_depths(tmp_path, names)
    with pytest.raises(RuntimeError, match="partial registration"):
        _reconstructor(tmp_path)._sfm_result_from_reconstruction(recon, tmp_path, store)


def test_sfm_result_builder_refuses_name_mismatch(tmp_path):
    # Same count, but the registered names are not the store's frame indices
    names = ["frame_000000", "frame_000004"]
    recon = _recon(names)
    store = _store(tmp_path, [0, 3])
    _write_depths(tmp_path, names)
    with pytest.raises(ValueError, match="different runs"):
        _reconstructor(tmp_path)._sfm_result_from_reconstruction(recon, tmp_path, store)
