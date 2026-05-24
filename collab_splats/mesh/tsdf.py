from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm.auto import tqdm

try:
    import meshlib.mrmeshpy as mm
    _MM_AVAILABLE = True
except ImportError:
    mm = None
    _MM_AVAILABLE = False

from collab_splats.mesh.base import BaseMeshCreator, MeshResult
from collab_splats.mesh.utils import clean_repair_mesh
from collab_splats.utils.geometry import extract_intrinsics, invert_poses

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
    clean_repair: bool = True
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

        raw_path = self.output_dir / "mesh_tsdf.ply"
        o3d.io.write_triangle_mesh(str(raw_path), mesh)
        final_path = raw_path

        if self.clean_repair:
            if _MM_AVAILABLE:
                temp_path = self.output_dir / "mesh_tsdf_temp.ply"
                shutil.copy(raw_path, temp_path)
                cleaned = clean_repair_mesh(
                    str(temp_path),
                    max_hole_size=self.clean_max_hole_size,
                    max_edge_splits=self.clean_max_edge_splits,
                    use_largest=self.clean_use_largest,
                )
                clean_path = self.output_dir / "mesh_tsdf_clean.ply"
                mm.saveMesh(cleaned, str(clean_path))
                temp_path.unlink(missing_ok=True)
                final_path = clean_path

                # meshlib strips vertex colors — transfer from raw mesh via KNN
                raw_mesh = o3d.io.read_triangle_mesh(str(raw_path))
                if raw_mesh.has_vertex_colors():
                    clean_mesh = o3d.io.read_triangle_mesh(str(clean_path))
                    raw_pcd = o3d.geometry.PointCloud(raw_mesh.vertices)
                    raw_pcd.colors = raw_mesh.vertex_colors
                    kdtree = o3d.geometry.KDTreeFlann(raw_pcd)
                    raw_colors = np.asarray(raw_mesh.vertex_colors)
                    clean_verts = np.asarray(clean_mesh.vertices)
                    new_colors = np.empty_like(clean_verts)
                    for j in range(len(clean_verts)):
                        _, idx, _ = kdtree.search_knn_vector_3d(clean_verts[j], 1)
                        new_colors[j] = raw_colors[idx[0]]
                    clean_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
                    o3d.io.write_triangle_mesh(str(clean_path), clean_mesh)
            else:
                logger.warning(
                    "clean_repair=True but meshlib not installed; skipping. "
                    "Install with: pip install meshlib"
                )

        return MeshResult(mesh_path=final_path)
