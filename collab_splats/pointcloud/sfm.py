# collab_splats/pointcloud/sfm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pycolmap

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result


@dataclass
class NerfstudioSfmCreator(BasePointcloudCreator):
    """Pointcloud via NerfStudio SfM (hloc or pycolmap SIFT).

    use_hloc=True: SuperPoint+SuperGlue via nerfstudio hloc_utils.
    use_hloc=False: classical SIFT via pycolmap directly.
    """

    use_hloc: bool = True
    feature_type: str = "superpoint_aachen"
    matcher_type: str = "superglue"
    num_matched: int = 50
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = False

    def create(self, image_dir: Path, output_dir: Path, **kwargs) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recon = self._run_hloc(image_dir, output_dir) if self.use_hloc else self._run_pycolmap(image_dir, output_dir)
        return _colmap_recon_to_result(recon)

    def _run_hloc(self, image_dir: Path, output_dir: Path):
        from nerfstudio.process_data.hloc_utils import run_hloc
        recon = run_hloc(
            image_dir=image_dir,
            output_dir=output_dir,
            sfm_tool="colmap",
            feature_type=self.feature_type,
            matcher_type=self.matcher_type,
            num_matched=self.num_matched,
            verbose=False,
        )
        if recon is None:
            raise RuntimeError("reconstruction failed — hloc returned None")
        return recon

    def _run_pycolmap(self, image_dir: Path, output_dir: Path):
        db_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        camera_mode = pycolmap.CameraMode.SINGLE if self.single_camera else pycolmap.CameraMode.AUTO
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
            output_path=str(sparse_dir),
        )
        if not reconstructions:
            raise RuntimeError("reconstruction failed — pycolmap incremental_mapping returned no results")
        return reconstructions[0]
