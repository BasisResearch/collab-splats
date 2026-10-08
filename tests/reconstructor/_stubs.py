"""
Shared Reconstructor stubs for the reconstructor tests.

This module deliberately imports nothing from `collab_splats.splats`. It used to live in
`test_splats_stage.py`, which imports `SplatsConfig` at module level; `test_vda_context.py`
imported the helper from there and so inherited the dependency, and when the environment's
gsplat stopped matching the pinned commit BOTH files became uncollectible — taking with them
the only end-to-end exercise of the `Reconstructor.pointcloud` sfm branch, which needs no gsplat at all.
"""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.preproc import frames as fr
from collab_splats.reconstructor import Reconstructor


def _stub_reconstructor(tmp_path, n_views=3, height=8, width=8):
    """
    Reconstructor with config + images/ store + a fake PointcloudResult; image_paths reversed to prove index lookup.
    """
    recon = Reconstructor.__new__(Reconstructor)
    recon.config = {
        "output_path": str(tmp_path),
        "pointcloud": {"method": "feedforward", "backend": "vggtx"},
        "mesh": {
            "enabled": True,
            "source": "feedforward",
            "voxel_depth_px": 4.0,
            "voxel_ref_percentile": 50,
            "sdf_trunc_mult": 4.0,
            "depth_trunc_percentile": 95,
            "max_faces": None,
            "conf_percentile": 20,
            "mask_sky": False,
            "texture": False,
            "use_convex_hull": False,
            "smooth_iterations": 0,
        },
        "semantics": {"extractor": "dinov2"},
        "splats": {"enabled": True, "max_steps": 1, "losses": {"depth": {"weight": 0.1}}},
    }

    frames = np.stack([np.full((height, width, 3), view * 10, np.uint8) for view in range(n_views)])
    fr.write_frames(recon.images_dir, frames, list(range(n_views)))
    image_paths = [Path(f"frame_{view:06d}.jpg") for view in reversed(range(n_views))]
    recon._result = SimpleNamespace(
        image_paths=image_paths,
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n_views, 1, 1)),
        points=np.zeros((200, 3), np.float32),
        colors=np.zeros((200, 3), np.uint8),
    )
    return recon


def minimal_feedforward_result(n=2, h=8, w=8):
    """
    PointcloudResult with unit depth and confidence=None; the smallest thing the mesh stage fuses.
    """
    return PointcloudResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=None,
        model_intrinsics=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)),
        image_paths=[f"frame_{i:06d}.jpg" for i in range(n)],
        original_coords=np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32),
        model_width=w,
        model_height=h,
        images=np.zeros((n, 3, h, w), dtype=np.float32),
        depth=np.ones((n, h, w), dtype=np.float32),
    )


def stub_creator_cls(result):
    """
    Creator class stand-in for a patched get_creator: records its kwargs, returns `result`.
    """
    creator_cls = MagicMock()
    creator_cls.return_value.create_pointcloud.return_value = result
    return creator_cls


@contextmanager
def stub_mesh_cleanup():
    """
    clean_repair_mesh and prepare_mesh stubbed to one tetrahedron, so the stage still writes mesh.ply.

    - yields the clean_repair_mesh mock
    """
    tet = o3d.geometry.TriangleMesh.create_tetrahedron()

    with (
        patch("collab_splats.reconstructor.clean_repair_mesh", return_value=(tet, tet)) as clean,
        patch("collab_splats.reconstructor.prepare_mesh", return_value=tet),
    ):
        yield clean
