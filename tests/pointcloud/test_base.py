import numpy as np
import pytest
import pycolmap
import open3d as o3d
from collab_splats.pointcloud.base import (
    CoordinateFrame,
    PointcloudResult,
    BasePointcloudCreator,
    _colmap_recon_to_result,
)
from collab_splats.pointcloud.utils import clean_pcd, remove_far_points, density_filter


def test_result_fields():
    r = PointcloudResult(
        points=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        confidence=None,
        camera_poses=np.eye(4, dtype=np.float32)[None],
        camera_intrinsics=np.eye(3, dtype=np.float32)[None],
        colmap_reconstruction=None,
    )
    assert r.points.shape == (10, 3)
    assert r.camera_poses.shape == (1, 4, 4)


def test_base_creator_is_abstract():
    with pytest.raises(TypeError):
        BasePointcloudCreator()


def test_colmap_recon_to_result_convention():
    """_colmap_recon_to_result applies both OpenCV→OpenGL (A) and COLMAP→nerfstudio world (B).

    Identity w2c → expected c2w after both transforms:
        [[1,  0,  0, 0],
         [0,  0, -1, 0],
         [0,  1,  0, 0],
         [0,  0,  0, 1]]
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=64, height=64, params=[50.0, 32.0, 32.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)

    img = pycolmap.Image(name="frame_0000.jpg", camera_id=1, image_id=1)
    recon.add_image_with_trivial_frame(img, pycolmap.Rigid3d())  # identity w2c

    recon.add_point3D(
        xyz=np.array([0.0, 0.0, 1.0]),
        track=pycolmap.Track(),
        color=np.array([128, 128, 128], dtype=np.uint8),
    )

    result = _colmap_recon_to_result(recon)

    assert result.frame == CoordinateFrame.NERFSTUDIO
    assert result.world_transform is not None
    assert result.world_transform.shape == (3, 4)

    pose = result.camera_poses[0]
    expected = np.array([
        [1,  0,  0, 0],
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1],
    ], dtype=np.float32)
    np.testing.assert_allclose(pose, expected, atol=1e-5)

    expected_world_transform = np.array([
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0, -1, 0, 0],
    ], dtype=np.float32)
    np.testing.assert_allclose(result.world_transform, expected_world_transform, atol=1e-5)


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


def test_result_has_frame_and_world_transform():
    r = PointcloudResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        confidence=None,
        camera_poses=None,
        camera_intrinsics=None,
        colmap_reconstruction=None,
    )
    assert r.frame == CoordinateFrame.NERFSTUDIO
    assert r.world_transform is None


def test_result_explicit_frame():
    r = PointcloudResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        confidence=None,
        camera_poses=None,
        camera_intrinsics=None,
        colmap_reconstruction=None,
        frame=CoordinateFrame.COLMAP,
    )
    assert r.frame == CoordinateFrame.COLMAP


def test_base_creator_abstract_method_is_reconstruct():
    import inspect
    abstract_methods = BasePointcloudCreator.__abstractmethods__
    assert "reconstruct" in abstract_methods
    assert "create" not in abstract_methods
