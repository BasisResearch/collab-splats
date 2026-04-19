# collab_splats/pointcloud/feedforward.py
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result

if TYPE_CHECKING:
    from collab_splats.semantics.extractors import BaseExtractor


@dataclass
class MapAnythingCreator(BasePointcloudCreator):
    """Pointcloud via MapAnything feedforward reconstruction.

    No COLMAP feature matching — depth + pose estimated directly by the network.
    Unique among backends: populates PointcloudResult.confidence.

    extractor: BaseExtractor from collab_splats.semantics.extractors (Agent 1).
               None = no semantic feature extraction, pointcloud only.
    """

    model_name: str = "mapanything"
    conf_threshold: float = 1.5
    subsample_factor: int = 1
    # String literal avoids NameError at runtime when semantics module not installed.
    extractor: "BaseExtractor | None" = field(default=None, repr=False)

    def create(self, image_dir: Path, output_dir: Path, **kwargs) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # stage/ is not an installed package — add repo root to sys.path if needed
        repo_root = Path(__file__).parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from stage.feedforward import Reconstructor
        from stage.mapanything_utils import build_colmap_reconstruction

        config = {
            "file_path": str(image_dir),
            "model_name": self.model_name,
            "output_path": str(output_dir),
        }
        reconstructor = Reconstructor(config)
        reconstructor.preprocess()
        reconstructor.setup_inference()
        reconstructor.infer()

        # Points, colors, confidence from feedforward network.
        # extractor wiring: pass name if extractor exposes one; full wiring awaits Agent 1 API.
        extractor_name = getattr(self.extractor, "name", None)
        pcd_dict = reconstructor.feedforward_to_pointcloud(
            confidence_threshold=self.conf_threshold,
            subsample_factor=self.subsample_factor,
            feature_extractor=extractor_name,
        )

        # Raw outputs for camera poses
        raw = reconstructor.prepare_outputs_for_export()

        # build_colmap_reconstruction expects (N, 3, 4) extrinsics; handle both (S,4,4) and (S,3,4)
        ext = raw["extrinsic"]
        if ext.ndim != 3 or ext.shape[1] not in (3, 4):
            raise RuntimeError(f"unexpected extrinsic shape {ext.shape} from prepare_outputs_for_export")
        extrinsics_3x4 = ext[:, :3, :]
        H, W = raw["images"].shape[1], raw["images"].shape[2]

        colors = pcd_dict["colors"]
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)

        recon = build_colmap_reconstruction(
            points_3d=pcd_dict["points"].astype(np.float64),
            points_rgb=colors,
            extrinsics=extrinsics_3x4,
            intrinsics=raw["intrinsic"],
            image_width=W,
            image_height=H,
            skip_point2d=True,
            verbose=False,
        )
        pose_result = _colmap_recon_to_result(recon)

        return PointcloudResult(
            points=pcd_dict["points"].astype(np.float32),
            colors=colors,
            confidence=pcd_dict.get("confidences"),
            camera_poses=pose_result.camera_poses,
            camera_intrinsics=pose_result.camera_intrinsics,
            colmap_reconstruction=recon,
        )
