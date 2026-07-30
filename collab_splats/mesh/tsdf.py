from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm.auto import tqdm

from collab_splats.geometry.transforms import extract_intrinsics, invert_poses
from collab_splats.mesh.base import BaseMeshCreator, MeshResult
from collab_splats.mesh.utils import clean_repair_mesh

logger = logging.getLogger(__name__)


@dataclass
class Open3DTSDFFusion(BaseMeshCreator):
    """TSDF fusion via Open3D ScalableTSDFVolume.

    Accepts rendered depth + RGB frames as numpy arrays — no nerfstudio dependency.
    """

    output_dir: Path
    voxel_size: float = 0.01
    sdf_trunc: float = 0.04
    depth_trunc: float = 20.0
    depth_scale: float = 1.0
    # Opt-in: cleanup rewrites the fused mesh and costs a second pass over it, so a caller asks
    # for it rather than getting it silently. Config-driven runs reach this default — that is why
    # it is the value that works, not the one that raises.
    clean_repair: bool = False
    clean_max_hole_size: float = 3.0
    clean_max_edge_splits: int = 10000
    clean_use_largest: bool = False

    def create(
        self,
        depths: np.ndarray,
        rgbs: np.ndarray,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        **kwargs,
    ) -> MeshResult:
        """Fuse depth+RGB frames into a mesh via TSDF.

        Args:
            depths:     (N, H, W) float32, metres
            rgbs:       (N, H, W, 3) float32, [0, 1]
            c2w:        (N, 4, 4) float32, cam-to-world OpenCV
            intrinsics: (N, 3, 3) float32
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        N, H, W = depths.shape

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        w2c = invert_poses(c2w)  # (N, 4, 4) — precompute all at once

        for i in tqdm(range(N), desc="TSDF integration"):
            rgb_u8 = (np.ascontiguousarray(rgbs[i]) * 255).astype(np.uint8)
            depth_f32 = np.ascontiguousarray(depths[i]).astype(np.float32)

            rgb_o3d = o3d.geometry.Image(rgb_u8)
            depth_o3d = o3d.geometry.Image(depth_f32)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_scale=self.depth_scale,
                depth_trunc=self.depth_trunc,
                convert_rgb_to_intensity=False,
            )

            fx, fy, cx, cy = extract_intrinsics(intrinsics[i])
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            extrinsic = w2c[i]

            volume.integrate(rgbd, intrinsic=intrinsic_o3d, extrinsic=extrinsic)

        mesh = volume.extract_triangle_mesh()

        # Filename matches Reconstructor.mesh()'s skip-check and the dashboard's mesh lookup
        mesh_path = self.output_dir / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        # Opt-in cleanup, rewriting in place: mesh.ply is the one name every reader in this repo
        # uses (Reconstructor's skip-check, splatter, dashboard, the remote push), and a separate
        # mesh_clean.ply would mean each of them needs a second probe and a precedence rule.
        if self.clean_repair:
            clean_repair_mesh(
                mesh_path,
                max_hole_size=self.clean_max_hole_size,
                max_edge_splits=self.clean_max_edge_splits,
                use_largest=self.clean_use_largest,
            )

        return MeshResult(mesh_path=mesh_path)
