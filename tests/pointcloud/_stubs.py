"""
Shared pycolmap/frame-store/raw-output builders for pointcloud tests.
"""

from pathlib import Path

import numpy as np
import pycolmap

from collab_splats.preproc import frames as fr


def make_recon(
    names: list[str],
    cam_w: int = 64,
    cam_h: int = 48,
    model: str = "PINHOLE",
    params: tuple[float, ...] = (50.0, 50.0, 32.0, 24.0),
) -> pycolmap.Reconstruction:
    """
    One shared camera, one image per name, one point3D observed in every image.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model=model, width=cam_w, height=cam_h, params=list(params), camera_id=1)
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


def make_scene(tmp_path: Path, names: list[str]) -> tuple[Path, Path]:
    """
    A backend data dir plus an images/ store holding names as real PNG keyframes.
    """
    images_dir = tmp_path / "images"
    fr.write_frames(images_dir, [np.zeros((8, 8, 3), np.uint8)] * len(names), [int(n[6:12]) for n in names])
    return tmp_path / "backend", images_dir


def vggt_raw_outputs(n: int, h: int, w: int) -> dict[str, np.ndarray]:
    """
    A VGGT-family _forward raw dict: random depth in [1, 2], near-identity poses, centered K.

    - seeded, so point counts are reproducible
    - small x translation per frame, so views overlap without coinciding
    """
    rng = np.random.default_rng(0)
    extrinsic = np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1))
    extrinsic[:, 0, 3] = 0.05 * np.arange(n)
    intrinsics = np.array([[10.0, 0, (w - 1) / 2], [0, 10.0, (h - 1) / 2], [0, 0, 1]], np.float32)
    return {
        "depth": rng.uniform(1.0, 2.0, (n, h, w, 1)).astype(np.float32),
        "depth_conf": np.ones((n, h, w), np.float32),
        "images": rng.random((n, 3, h, w)).astype(np.float32),
        "extrinsic": extrinsic,
        "intrinsics": np.tile(intrinsics, (n, 1, 1)),
    }
