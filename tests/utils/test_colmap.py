"""
Tests for utils/colmap.py: stem names, atomic swap, round trip.
"""

import numpy as np
import pycolmap

from collab_splats.utils.colmap import write_colmap_reconstruction


def _recon(names: list[str]) -> pycolmap.Reconstruction:
    recon = pycolmap.Reconstruction()
    for i, name in enumerate(names, start=1):
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=64,
            height=48,
            params=[50.0, 50.0, 32.0, 24.0],
            camera_id=i,
        )
        recon.add_camera_with_trivial_rig(camera)
        image = pycolmap.Image(name=name, camera_id=i, image_id=i)
        recon.add_image_with_trivial_frame(image, pycolmap.Rigid3d())
    recon.add_point3D(
        np.array([0.0, 0.0, 1.0]), pycolmap.Track(), np.array([1, 2, 3], dtype=np.uint8)
    )
    return recon


def test_write_names_are_stems(tmp_path):
    model_dir = tmp_path / "colmap" / "sparse" / "0"
    write_colmap_reconstruction(
        _recon(["frame_000001.png", "frame_000002.jpg"]), model_dir
    )
    names = sorted(
        img.name for img in pycolmap.Reconstruction(str(model_dir)).images.values()
    )
    assert names == ["frame_000001", "frame_000002"]


def test_write_replaces_whole_model(tmp_path):
    model_dir = tmp_path / "m"
    write_colmap_reconstruction(_recon(["a.png", "b.png"]), model_dir)
    write_colmap_reconstruction(_recon(["c.png"]), model_dir)
    assert [
        img.name for img in pycolmap.Reconstruction(str(model_dir)).images.values()
    ] == ["c"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m"]


def test_write_clears_crash_leftovers(tmp_path):
    model_dir = tmp_path / "m"
    (tmp_path / ".m.tmp").mkdir()
    (tmp_path / ".m.old").mkdir()
    write_colmap_reconstruction(_recon(["a.png"]), model_dir)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m"]
