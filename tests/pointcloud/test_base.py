import numpy as np
import pytest
import pycolmap
import open3d as o3d
from pathlib import Path
from collab_splats.pointcloud.base import (
    CoordinateFrame,
    PointcloudResult,
    BasePointcloudCreator,
)
from collab_splats.pointcloud.utils import clean_pcd, remove_far_points, density_filter
from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction


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
    import inspect
    abstract_methods = BasePointcloudCreator.__abstractmethods__
    assert "reconstruct" in abstract_methods
    assert "create" not in abstract_methods
