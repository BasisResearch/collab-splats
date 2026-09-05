"""build_pointcloud writes a real sparse_pc.ply into backend_dir."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pycolmap

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.wrapper.reconstructor import Reconstructor


def _pointcloud_result(n_points):
    """
    A real PointcloudResult over one registered frame and n_points tracked points.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=8, height=6, params=[4.0, 4.0, 4.0, 3.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    recon.add_image_with_trivial_frame(pycolmap.Image(name="frame_000000", camera_id=1, image_id=1), pycolmap.Rigid3d())
    for i in range(n_points):
        recon.add_point3D(
            xyz=np.array([float(i), 0.0, 1.0]),
            track=pycolmap.Track(),
            color=np.array([i, 2 * i, 3 * i], dtype=np.uint8),
        )
    return PointcloudResult(
        reconstruction=recon,
        image_paths=[Path("frame_000000")],
    )


def test_build_pointcloud_writes_sparse_pc_ply(tmp_path):
    """
    The pointcloud stage lands a binary PLY of the final result at backend_dir/sparse_pc.ply.
    """
    # clean disabled so the result reaching the writer is the one the backend returned;
    # transforms.json is a separate writer with its own tests.
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "clean": {"enabled": False}},
    }
    rec = Reconstructor(config)
    result = _pointcloud_result(3)

    with (
        patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=(result, None)),
        patch.object(Reconstructor, "_write_transforms_json"),
    ):
        rec.build_pointcloud(overwrite=True)

    # The path is the contract: downstream stages and the remote sync both look here by name
    out = rec.backend_dir / "sparse_pc.ply"
    assert out.exists()

    # Binary little-endian, and carrying THIS result's points — not an empty or stub file.
    # xyz/rgb fidelity is PointcloudResult.write_ply's own contract (tests/pointcloud/test_base.py).
    header = out.read_bytes()[:200]
    assert header.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 3\n" in header
