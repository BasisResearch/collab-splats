"""
Classical COLMAP SfM backend: SIFT, exhaustive matching, incremental mapping.

- pycolmap end to end; no learned components
- unwired — nothing dispatches to ColmapCreator, see sfm/__init__.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from ..base import BasePointcloudCreator, PointcloudResult

logger = logging.getLogger(__name__)


@dataclass
class ColmapCreator(BasePointcloudCreator):
    """
    Pointcloud via pycolmap: SIFT extraction -> exhaustive matching -> incremental mapping.

    - camera_model: COLMAP camera model string (SIMPLE_PINHOLE, SIMPLE_RADIAL, OPENCV, ...).
    - single_camera: one shared camera for every image (video from one device) vs one per image.
    - Writes the binary model to output_dir/colmap/sparse/0; the SIFT DB to colmap/database.db.
    """

    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = False

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        """
        SIFT extraction, exhaustive matching and incremental mapping over image_dir.

        Args:
            image_dir:  Directory of input images; must exist.
            output_dir: Run directory. The binary model lands in colmap/sparse/0, the SIFT
                database in colmap/database.db.

        Returns:
            PointcloudResult wrapping the largest reconstructed model.
        """
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        db_path = output_dir / "colmap" / "database.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # pycolmap >=4.0: camera_model lives in ImageReaderOptions, not as a
        # top-level kwarg of extract_features.
        camera_mode = pycolmap.CameraMode.SINGLE if self.single_camera else pycolmap.CameraMode.AUTO
        reader_opts = pycolmap.ImageReaderOptions(camera_model=self.camera_model)
        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(image_dir),
            camera_mode=camera_mode,
            reader_options=reader_opts,
        )
        pycolmap.match_exhaustive(str(db_path))
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(image_dir),
            output_path=str(sparse_dir.parent),  # colmap/sparse/ -> creates 0/ inside
        )
        if not reconstructions:
            raise RuntimeError("reconstruction failed — pycolmap incremental_mapping returned no results")

        # Cluster 0 is the model; image_paths follow filename order, the pipeline's row order
        recon = reconstructions[0]
        recon.write_binary(str(sparse_dir))
        logger.info("COLMAP: %d registered images, %d points3D", len(recon.images), len(recon.points3D))
        image_paths = sorted(
            [image_dir / img.name for img in recon.images.values()],
            key=lambda p: p.name,
        )
        return PointcloudResult(reconstruction=recon, image_paths=image_paths)
