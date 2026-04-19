import numpy as np
import pytest
import pycolmap
import open3d as o3d
from collab_splats.pointcloud.base import (
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
    """_colmap_recon_to_result must output cam2world OpenGL poses.
    Build a minimal Reconstruction with identity w2c, verify flipped Y/Z output.
    """
    recon = pycolmap.Reconstruction()

    cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=64,
        height=64,
        params=[50.0, 32.0, 32.0],  # f, cx, cy
    )
    cam.camera_id = 1
    recon.add_camera(cam)

    img = pycolmap.Image(name="frame_0000.jpg", camera_id=1)
    img.image_id = 1
    img.cam_from_world = pycolmap.Rigid3d()  # identity w2c
    recon.add_image(img)
    recon.register_image(img.image_id)  # mark as registered so _colmap_recon_to_result includes it

    recon.add_point3D(
        xyz=np.array([0.0, 0.0, 1.0]),
        track=pycolmap.Track(),
        color=np.array([128, 128, 128], dtype=np.uint8),
    )

    result = _colmap_recon_to_result(recon)

    assert result.camera_poses is not None
    c2w = result.camera_poses[0]
    assert c2w.shape == (4, 4)
    np.testing.assert_array_almost_equal(c2w[3], [0.0, 0.0, 0.0, 1.0])
    # identity w2c → c2w = I (OpenCV) → flip Y,Z cols → cols 1,2 negated
    expected = np.eye(4, dtype=np.float32)
    expected[:3, 1:3] *= -1
    np.testing.assert_array_almost_equal(c2w, expected, decimal=5)


def test_utils_clean_pcd_returns_tuple():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    result_pcd, indices = clean_pcd(pcd)
    assert isinstance(result_pcd, o3d.geometry.PointCloud)
    assert isinstance(indices, np.ndarray)
