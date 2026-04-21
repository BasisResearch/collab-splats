# collab_splats/pointcloud/sfm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pycolmap

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result


@dataclass
class ColmapCreator(BasePointcloudCreator):
    """Pointcloud via pycolmap SIFT feature extraction + exhaustive matching."""

    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = False

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        db_path = output_dir / "colmap" / "database.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        camera_mode = (
            pycolmap.CameraMode.SINGLE if self.single_camera else pycolmap.CameraMode.AUTO
        )
        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(image_dir),
            camera_mode=camera_mode,
            camera_model=self.camera_model,
        )
        pycolmap.match_exhaustive(str(db_path))
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(image_dir),
            output_path=str(sparse_dir.parent),  # colmap/sparse/ → creates 0/ inside
        )
        if not reconstructions:
            raise RuntimeError("reconstruction failed — pycolmap incremental_mapping returned no results")

        recon = reconstructions[0]
        recon.write_binary(str(sparse_dir))
        self._write_transforms(sparse_dir, output_dir)
        return _colmap_recon_to_result(recon)
