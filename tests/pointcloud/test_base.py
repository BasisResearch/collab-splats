import inspect
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
import pytest

from collab_splats.pointcloud.base import (
    BasePointcloudCreator,
    CoordinateFrame,
    PointcloudResult,
)
from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction
from collab_splats.pointcloud.utils import clean_pcd, density_filter, remove_far_points


def test_result_fields():
    """PointcloudResult constructed with reconstruction exposes points/extrinsics/intrinsics."""
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=4, height=4, params=[2.0, 2.0, 2.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    img = pycolmap.Image(name="frame_0000.jpg", camera_id=1, image_id=1)
    recon.add_image_with_trivial_frame(img, pycolmap.Rigid3d())
    recon.add_point3D(
        xyz=np.array([0.0, 0.0, 1.0]),
        track=pycolmap.Track(),
        color=np.array([128, 128, 128], dtype=np.uint8),
    )

    r = PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=[Path("frame_0000.jpg")],
    )
    assert r.points.shape == (1, 3)
    assert r.extrinsics.shape == (1, 4, 4)
    assert r.intrinsics.shape == (1, 3, 3)


def test_base_creator_is_abstract():
    with pytest.raises(TypeError):
        BasePointcloudCreator()


def test_result_colmap_frame():
    """PointcloudResult stores and returns frame as COLMAP when set explicitly."""
    recon = pycolmap.Reconstruction()
    r = PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=None,
    )
    assert r.frame == CoordinateFrame.COLMAP
    assert r.world_transform is None


def test_result_nerfstudio_frame():
    """PointcloudResult stores NERFSTUDIO frame + world_transform."""
    recon = pycolmap.Reconstruction()
    wt = np.eye(3, 4, dtype=np.float32)
    r = PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.NERFSTUDIO,
        image_paths=None,
        world_transform=wt,
    )
    assert r.frame == CoordinateFrame.NERFSTUDIO
    assert r.world_transform is not None
    assert r.world_transform.shape == (3, 4)


def test_utils_clean_pcd_returns_tuple():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    result_pcd, indices = clean_pcd(pcd)
    assert isinstance(result_pcd, o3d.geometry.PointCloud)
    assert isinstance(indices, np.ndarray)


def test_coordinate_frame_values():
    assert CoordinateFrame.COLMAP == "colmap"
    assert CoordinateFrame.NERFSTUDIO == "nerfstudio"
    assert isinstance(CoordinateFrame.COLMAP, str)


def test_base_creator_abstract_method_is_reconstruct():
    abstract_methods = BasePointcloudCreator.__abstractmethods__
    assert "reconstruct" in abstract_methods
    assert "create" not in abstract_methods


def _recon_with_images(names=("frame_000000",)):
    """
    PINHOLE camera + one image per name (translation z = index) + one coloured point3D.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=8, height=6, params=[4.0, 4.0, 4.0, 3.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
    recon.add_point3D(
        xyz=np.array([1.0, 2.0, 3.0]),
        track=pycolmap.Track(),
        color=np.array([10, 20, 30], dtype=np.uint8),
    )
    return recon


def _write_model(tmp_path, recon):
    """
    Write recon to <tmp_path>/colmap/sparse/0 and return the <tmp_path>/colmap dir.
    """
    sparse = tmp_path / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True)
    recon.write_binary(str(sparse))
    return tmp_path / "colmap"


def test_write_ply_roundtrips_through_open3d(tmp_path):
    """
    write_ply exports xyz+rgb from the reconstruction and creates missing parents.
    """
    result = PointcloudResult(
        reconstruction=_recon_with_images(),
        frame=CoordinateFrame.COLMAP,
        image_paths=[Path("frame_000000")],
    )
    out = tmp_path / "nested" / "sparse_pc.ply"

    result.write_ply(out)

    pcd = o3d.io.read_point_cloud(str(out))
    np.testing.assert_allclose(np.asarray(pcd.points), [[1.0, 2.0, 3.0]], atol=1e-6)
    np.testing.assert_allclose(np.asarray(pcd.colors), [[10 / 255, 20 / 255, 30 / 255]], atol=1e-6)


def test_from_colmap_reads_the_model_and_keeps_caller_order(tmp_path):
    """
    from_colmap loads <colmap_dir>/sparse/0 and orders extrinsics by the caller's image_paths.
    """
    colmap_dir = _write_model(tmp_path, _recon_with_images(["frame_000000", "frame_000001"]))

    r = PointcloudResult.from_colmap(colmap_dir, [Path("frame_000001"), Path("frame_000000")])

    assert r.image_paths == [Path("frame_000001"), Path("frame_000000")]
    assert r.extrinsics.shape == (2, 4, 4)
    # extrinsics are w2c: image frame_000001 was placed at translation z = 1.0
    np.testing.assert_allclose(r.extrinsics[0][2, 3], 1.0, atol=1e-6)
    np.testing.assert_allclose(r.extrinsics[1][2, 3], 0.0, atol=1e-6)


def test_from_colmap_rejects_names_missing_from_the_model(tmp_path):
    """
    A requested image the model never registered is a hard error, not a silent KeyError later.
    """
    colmap_dir = _write_model(tmp_path, _recon_with_images(["frame_000000"]))

    with pytest.raises(ValueError, match="frame_000001"):
        PointcloudResult.from_colmap(colmap_dir, [Path("frame_000000"), Path("frame_000001")])


def test_from_colmap_raises_when_the_model_was_never_written(tmp_path):
    """
    A colmap_dir with no sparse/0 is an error, not an empty reconstruction.
    """
    with pytest.raises(ValueError):
        PointcloudResult.from_colmap(tmp_path / "colmap", [Path("frame_000000")])
