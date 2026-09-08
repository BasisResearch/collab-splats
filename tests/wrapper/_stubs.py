"""
Shared Reconstructor stubs for the wrapper tests.

This module deliberately imports nothing from `collab_splats.splats`. It used to live in
`test_splats_stage.py`, which imports `SplatsConfig` at module level; `test_vda_context.py`
imported the helper from there and so inherited the dependency, and when the environment's
gsplat stopped matching the pinned commit BOTH files became uncollectible — taking with them
the only end-to-end exercise of `Reconstructor._run_sfm`, which needs no gsplat at all.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.preproc import frames as fr
from collab_splats.wrapper.reconstructor import Reconstructor


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
            "voxel_size": 0.01,
            "sdf_trunc_mult": 4.0,
            "bands": None,
            "depth_trunc": 1.0,
            "conf_percentile": 20,
            "mask_sky": False,
            "texture": False,
        },
        "splats": {"enabled": True, "max_steps": 1, "losses": {"depth": {"weight": 0.1}}},
    }
    recon._stage_output_exists = lambda stage: False

    frames = np.stack([np.full((height, width, 3), view * 10, np.uint8) for view in range(n_views)])
    records = [{"frame_idx": view} for view in range(n_views)]
    fr.write_frames(recon.images_dir, frames, records, {"video_path": "v"})
    image_paths = [Path(f"frame_{view:06d}.jpg") for view in reversed(range(n_views))]
    recon._resolve_result = lambda: SimpleNamespace(
        image_paths=image_paths,
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n_views, 1, 1)),
        points=np.zeros((200, 3), np.float32),
        colors=np.zeros((200, 3), np.uint8),
    )
    return recon


def minimal_feedforward_result(n=2, h=8, w=8):
    """
    FeedforwardResult with unit depth and confidence=None; the smallest thing _run_tsdf_mesh accepts.
    """
    return FeedforwardResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)),
        image_paths=[f"frame_{i:06d}.jpg" for i in range(n)],
        original_coords=np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32),
        model_width=w,
        model_height=h,
        images=np.zeros((n, 3, h, w), dtype=np.float32),
        depth=np.ones((n, h, w), dtype=np.float32),
    )


def minimal_pose_result(n=2):
    """
    The PointcloudResult stand-in _run_tsdf_mesh reads poses and original-res K from.
    """
    return SimpleNamespace(
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], np.float32), (n, 1, 1)),
    )
