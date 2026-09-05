"""build_pointcloud writes a real sparse_pc.ply into backend_dir, and its clean step deletes in place."""

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
    # clean disabled so the result reaching the writer is the one the backend returned.
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "clean": {"enabled": False}},
    }
    rec = Reconstructor(config)
    result = _pointcloud_result(3)

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=(result, None)):
        rec.build_pointcloud(overwrite=True)

    # The path is the contract: downstream stages and the remote sync both look here by name
    out = rec.backend_dir / "sparse_pc.ply"
    assert out.exists()

    # Binary little-endian, and carrying THIS result's points — not an empty or stub file.
    # xyz/rgb fidelity is PointcloudResult.write_ply's own contract (tests/pointcloud/test_base.py).
    header = out.read_bytes()[:200]
    assert header.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 3\n" in header


def _sorted_rows(xyz):
    """
    Rows of an (N, 3) array in lexicographic order — for comparing point sets order-independently.
    """
    return np.asarray(xyz, dtype=np.float64)[np.lexsort(np.asarray(xyz, dtype=np.float64).T[::-1])]

def _clustered_pointcloud_result():
    """
    A PointcloudResult whose tracked set is a tight cluster plus one planted far outlier.

    Returns (result, cluster_xyz) so a caller can assert WHICH points survived cleaning.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=8, height=6, params=[4.0, 4.0, 4.0, 3.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    recon.add_image_with_trivial_frame(pycolmap.Image(name="frame_000000", camera_id=1, image_id=1), pycolmap.Rigid3d())

    # 60 cluster points clears clean_pointcloud's nb_neighbors=20 guard; the outlier sits far
    # enough out that statistical removal rejects it and nothing else.
    cluster = np.random.default_rng(0).normal(scale=0.01, size=(60, 3))
    for xyz in np.vstack([cluster, [[50.0, 50.0, 50.0]]]):
        recon.add_point3D(xyz=xyz, track=pycolmap.Track(), color=np.array([1, 2, 3], dtype=np.uint8))

    return PointcloudResult(reconstruction=recon, image_paths=[Path("frame_000000")]), cluster


def test_build_pointcloud_clean_deletes_only_the_outlier(tmp_path):
    """
    clean.enabled deletes the rejected point3D ids from the reconstruction in place.

    Asserting the surviving XYZs — not just the count — is what pins the id snapshot
    (points3D.keys()) to the mask, which is computed over .points (built from .values()).
    """
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "clean": {"enabled": True}},
    }
    rec = Reconstructor(config)
    result, cluster = _clustered_pointcloud_result()

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=(result, None)):
        rec.build_pointcloud(overwrite=True)

    # Exactly one point went, and the survivors are the cluster itself — a keys()/values()
    # misalignment would keep the far outlier and drop a cluster point instead, failing the
    # second assert. Compared as a multiset of rows: points3D iteration is by id, not insertion.
    assert result.reconstruction.num_points3D() == len(cluster)
    np.testing.assert_allclose(_sorted_rows(result.points), _sorted_rows(cluster), atol=1e-6)

    # The re-exported PLY carries the cleaned set, not the pre-clean one.
    header = (rec.backend_dir / "sparse_pc.ply").read_bytes()[:200]
    assert b"element vertex 60\n" in header
