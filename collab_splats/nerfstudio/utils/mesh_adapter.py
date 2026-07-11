from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup

from collab_splats.geometry.transforms import OPENGL_TO_OPENCV, extrinsics_to_homogeneous

def extract_mesh_inputs(
    load_config: Path,
    depth_name: str = "depth",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-frame (depths, rgbs, c2w, intrinsics) from a trained nerfstudio pipeline.

    Args:
        load_config: Path to nerfstudio config yaml (e.g. outputs/.../config.yml)

    Returns:
        depths:     (N, H, W) float32, metres
        rgbs:       (N, H, W, 3) float32, [0, 1]
        c2w:        (N, 4, 4) float32, cam-to-world OpenCV convention
        intrinsics: (N, 3, 3) float32
    """
    _, pipeline, _, _ = eval_setup(Path(load_config))

    cameras = pipeline.datamanager.train_dataset.cameras
    N = len(pipeline.datamanager.train_dataset)

    all_depths: list[np.ndarray] = []
    all_rgbs: list[np.ndarray] = []
    all_c2w: list[np.ndarray] = []
    all_intrinsics: list[np.ndarray] = []

    with torch.no_grad():
        for image_idx, data in enumerate(pipeline.datamanager.train_dataset):
            camera = cameras[image_idx : image_idx + 1]
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)

            rgb = outputs["rgb"].cpu().numpy().astype(np.float32)          # (H, W, 3)
            depth = outputs[depth_name].squeeze(-1).cpu().numpy().astype(np.float32)  # (H, W)

            # nerfstudio camera_to_worlds is (N, 3, 4); take [0] to get (3, 4), then pad to (4, 4)
            c2w_34 = camera.camera_to_worlds[0].cpu().numpy()
            # nerfstudio stores c2w in OpenGL convention (Y-up, Z-back); TSDF expects OpenCV (Y-down, Z-forward)
            c2w_44 = (extrinsics_to_homogeneous(c2w_34) @ OPENGL_TO_OPENCV).astype(np.float32)

            K = np.eye(3, dtype=np.float32)
            K[0, 0] = camera.fx.item()
            K[1, 1] = camera.fy.item()
            K[0, 2] = camera.cx.item()
            K[1, 2] = camera.cy.item()

            all_rgbs.append(rgb)
            all_depths.append(depth)
            all_c2w.append(c2w_44)
            all_intrinsics.append(K)

    return (
        np.stack(all_depths),
        np.stack(all_rgbs),
        np.stack(all_c2w),
        np.stack(all_intrinsics),
    )
