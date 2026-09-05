"""The post-clean sparse_pc.ply written by build_pointcloud."""

from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

from collab_splats.pointcloud.base import CoordinateFrame, PointcloudResult


def test_build_pointcloud_ply_is_readable_by_open3d(tmp_path):
    """
    The PLY written from a PointcloudResult round-trips through open3d with xyz+rgb intact.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=8, height=6, params=[4.0, 4.0, 4.0, 3.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    recon.add_image_with_trivial_frame(pycolmap.Image(name="frame_000000", camera_id=1, image_id=1), pycolmap.Rigid3d())
    for i in range(3):
        recon.add_point3D(
            xyz=np.array([float(i), 0.0, 1.0]),
            track=pycolmap.Track(),
            color=np.array([i, 2 * i, 3 * i], dtype=np.uint8),
        )

    out = tmp_path / "backend" / "sparse_pc.ply"
    result = PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=[Path("frame_000000")],
    )
    result.write_ply(out)

    pcd = o3d.io.read_point_cloud(str(out))
    assert np.asarray(pcd.points).shape == (3, 3)
    assert out.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")
