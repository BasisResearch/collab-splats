"""
BaseSfmCreator.create_pointcloud: depth, registered subset, clean, COLMAP write and depth align, once.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.sfm import base as base_mod
from collab_splats.pointcloud.sfm.base import BaseSfmCreator
from collab_splats.preproc import frames as fr
from collab_splats.utils.colmap import read_colmap_reconstruction

# Source frame indices of the store; non-contiguous so a row/index mix-up shows
FRAME_IDX = (0, 9, 30)
NAMES = [f"frame_{i:06d}.png" for i in FRAME_IDX]

# Store frames and camera share one grid, as align_depth requires
WIDTH, HEIGHT = 64, 48


########################################
# Fakes
########################################


@dataclass
class _Fake(BaseSfmCreator):
    """
    Backend whose mapper returns a model built in-test.
    """

    model: pycolmap.Reconstruction | None = None

    def _map(
        self, images_dir: Path, out_dir: Path, names: list[str]
    ) -> pycolmap.Reconstruction:
        """
        The in-test model, or a mapper failure when there is none.
        """
        if self.model is None:
            raise RuntimeError("mapper failed")
        return self.model


def _model(*, unregistered=(), outlier=False, trackless=False, radial=0.0):
    """
    A mapper-shaped in-memory model over NAMES: 30 grid points seen by every registered frame.

    - names keep their extension, as a mapper's do; the write reduces them to stems
    - unregistered: rows added then deregistered, as an incremental mapper leaves them
    - outlier: one far point3D observed once; trackless: one point3D with no observations
    - radial: the shared SIMPLE_RADIAL camera's k; 0 is undistorted
    """
    model = pycolmap.Reconstruction()
    camera = pycolmap.Camera(
        model="SIMPLE_RADIAL",
        width=WIDTH,
        height=HEIGHT,
        params=[50.0, 32.0, 24.0, radial],
    )
    camera.camera_id = 1
    model.add_camera_with_trivial_rig(camera)

    # 6 x 5 grid on the z = 5 plane, well inside every view
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-0.8, 0.8, 5))
    grid = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 5.0)], axis=1)

    # One image per name, shifted along x; keypoints are the exact projections plus one spare
    for row, name in enumerate(NAMES):
        shift = np.array([0.1 * row, 0.0, 0.0])
        cam_pts = grid + shift
        xy = 50.0 * cam_pts[:, :2] / cam_pts[:, 2:] + np.array([32.0, 24.0])
        image = pycolmap.Image(name=name, camera_id=1, image_id=row + 1)
        image.points2D = [pycolmap.Point2D(p) for p in np.vstack([xy, [[10.0, 10.0]]])]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), shift)
        model.add_image_with_trivial_frame(image, pose)

    # Grid points tracked through every registered image
    registered = [row + 1 for row in range(len(NAMES)) if row not in unregistered]
    for j, xyz in enumerate(grid):
        track = pycolmap.Track()
        for image_id in registered:
            track.add_element(image_id, j)
        model.add_point3D(xyz, track, np.array([10, 20, 30], np.uint8))

    # Optional far outlier (spare keypoint of the first image) and a track-less point
    if outlier:
        track = pycolmap.Track()
        track.add_element(1, len(grid))
        model.add_point3D(np.array([50.0, 50.0, 50.0]), track, np.zeros(3, np.uint8))
    if trackless:
        model.add_point3D(
            np.array([0.0, 0.0, 6.0]), pycolmap.Track(), np.zeros(3, np.uint8)
        )

    # Unregistered rows keep their image, as a mapper's in-memory model does
    for row in unregistered:
        model.deregister_frame(model.images[row + 1].frame_id)
    return model


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """
    Store of NAMES (pixel value = frame index), constant VDA depth, and the create_pointcloud() dirs.
    """
    images_dir = tmp_path / "images"
    fr.write_frames(
        images_dir,
        [np.full((HEIGHT, WIDTH, 3), i, np.uint8) for i in FRAME_IDX],
        FRAME_IDX,
    )
    calls = []

    # Constant metric depth, no VDA model
    def fake_depth(frames, out_dir, names):
        calls.append((frames.shape, Path(out_dir), list(names)))
        return np.full((len(names), 12, 16), 5.0, np.float32)

    monkeypatch.setattr(base_mod, "estimate_depth", fake_depth)
    out_dir = tmp_path / "backend"
    return images_dir, out_dir, out_dir / "colmap" / "sparse" / "0", calls


########################################
# create_pointcloud
########################################


def test_create_writes_model_and_returns_aligned_result(scene):
    images_dir, out_dir, model_dir, depth_calls = scene
    creator = _Fake(model=_model())

    result = creator.create_pointcloud(images_dir, out_dir, model_dir)

    # Depth over the whole store, in store order
    assert depth_calls == [((3, HEIGHT, WIDTH, 3), out_dir, NAMES)]

    # One model on disk, stem names; the result covers every registered frame
    written = read_colmap_reconstruction(model_dir)
    assert sorted(im.name for im in written.images.values()) == [
        Path(n).stem for n in NAMES
    ]
    assert len(result.image_paths) == written.num_reg_images() == 3
    assert creator.attrs["method"] == "sfm"
    assert (creator.attrs["registered_frames"], creator.attrs["total_frames"]) == (3, 3)
    assert "depth_scales" in creator.attrs


def test_registered_subset_floor(scene, caplog):
    images_dir, out_dir, model_dir, _ = scene

    # 2 of 3 above a 0.5 floor: subset in store order, with a warning
    creator = _Fake(min_registered_frac=0.5, model=_model(unregistered=(1,)))
    with caplog.at_level(logging.WARNING):
        result = creator.create_pointcloud(images_dir, out_dir, model_dir)
    assert "registered 2/3" in caplog.text
    assert [p.name for p in result.image_paths] == ["frame_000000", "frame_000030"]
    assert np.round(result.images[:, 0, 0, 0] * 255).tolist() == [0.0, 30.0]
    assert (creator.attrs["registered_frames"], creator.attrs["total_frames"]) == (2, 3)

    # 1 of 3 is below the floor
    creator = _Fake(min_registered_frac=0.5, model=_model(unregistered=(0, 1)))
    with pytest.raises(RuntimeError, match=r"1/3.*frame_000000.*frame_000009"):
        creator.create_pointcloud(images_dir, out_dir, model_dir)

    # The floor is a share in (0, 1]
    with pytest.raises(ValueError, match="min_registered_frac"):
        _Fake(min_registered_frac=1.5)


@pytest.mark.parametrize("clean, n_far", [(False, 1), (True, 0)])
def test_outlier_removed_from_result_and_export_when_clean(scene, clean, n_far):
    images_dir, out_dir, model_dir, _ = scene

    creator = _Fake(clean=clean, model=_model(outlier=True))
    result = creator.create_pointcloud(images_dir, out_dir, model_dir)

    # The COLMAP model and the zarr points hold one point set
    written = read_colmap_reconstruction(model_dir)
    assert written.num_points3D() == len(result.points)
    far = [p for p in written.points3D.values() if p.xyz[0] > 10]
    assert len(far) == n_far
    assert (result.points[:, 0] > 10).sum() == n_far


def test_cap_trims_the_export_to_the_result_points(scene):
    images_dir, out_dir, model_dir, _ = scene

    result = _Fake(clean=False, max_points=12, model=_model()).create_pointcloud(
        images_dir, out_dir, model_dir
    )

    # Exactly the capped points survive in the export, tracks intact
    written = read_colmap_reconstruction(model_dir)
    exported = sorted(
        tuple(p.xyz.astype(np.float32).tolist()) for p in written.points3D.values()
    )
    assert exported == sorted(tuple(xyz) for xyz in result.points.tolist())
    assert len(exported) == 12
    assert all(len(p.track.elements) == len(NAMES) for p in written.points3D.values())


def test_trackless_points3d_dropped_before_write(scene):
    images_dir, out_dir, model_dir, _ = scene

    result = _Fake(clean=False, model=_model(trackless=True)).create_pointcloud(
        images_dir, out_dir, model_dir
    )

    assert (
        read_colmap_reconstruction(model_dir).num_points3D() == len(result.points) == 30
    )


def test_create_removes_a_stale_model_before_mapping(scene):
    images_dir, out_dir, model_dir, _ = scene
    model_dir.mkdir(parents=True)
    (model_dir / "cameras.bin").write_bytes(b"old")

    with pytest.raises(RuntimeError, match="mapper failed"):
        _Fake().create_pointcloud(images_dir, out_dir, model_dir)
    assert not model_dir.exists()


def test_create_refuses_a_missing_image_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no images"):
        _Fake(model=_model()).create_pointcloud(
            tmp_path / "images", tmp_path / "backend", tmp_path / "model"
        )


def test_export_keeps_the_mapper_tracks(scene):
    images_dir, out_dir, model_dir, _ = scene

    _Fake(model=_model()).create_pointcloud(images_dir, out_dir, model_dir)

    written = read_colmap_reconstruction(model_dir)
    assert all(len(p.track.elements) == len(NAMES) for p in written.points3D.values())


def test_export_keeps_a_distorted_mapper_camera(scene):
    images_dir, out_dir, model_dir, _ = scene

    _Fake(model=_model(radial=0.1)).create_pointcloud(images_dir, out_dir, model_dir)

    camera = next(iter(read_colmap_reconstruction(model_dir).cameras.values()))
    assert camera.model.name == "SIMPLE_RADIAL"
    assert camera.params[3] == pytest.approx(0.1)
