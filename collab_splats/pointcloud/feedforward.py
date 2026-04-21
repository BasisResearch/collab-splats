# collab_splats/pointcloud/feedforward.py
from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result


@dataclass
class BaseFeedforwardCreator(BasePointcloudCreator):
    """Template for feedforward (depth-estimation) pointcloud creators.

    Subclasses implement _run_inference() which writes binary COLMAP files to
    output_dir/colmap/sparse/0/ and returns the pycolmap.Reconstruction.
    Base class calls _write_transforms() once after _run_inference() completes.
    """

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        recon = self._run_inference(image_dir, output_dir)
        self._write_transforms(sparse_dir, output_dir)
        return _colmap_recon_to_result(recon)

    @abstractmethod
    def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
        """Run model inference, write binary to output_dir/colmap/sparse/0/, return Reconstruction."""
        ...


def _add_stage_to_path() -> None:
    repo_root = Path(__file__).parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@dataclass
class MapAnythingCreator(BaseFeedforwardCreator):
    """Pointcloud via MapAnything feedforward depth + pose estimation.

    No feature matching — depth and pose estimated directly by the network.
    Populates PointcloudResult.confidence via inference filtering.
    """

    model_name: str = "facebook/map-anything"
    confidence_percentile: float = 35.0   # keep points above this percentile (top 65%)
    use_multiview_confidence: bool = True  # multi-view depth consistency filter
    minibatch_size: int = 1               # frames processed at once (1 = most memory-efficient)

    def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
        _add_stage_to_path()
        from stage.mapanything_utils import (
            load_mapanything_model,
            load_and_preprocess_images,
            run_mapanything_inference,
            export_to_colmap,
            rescale_to_original_dimensions,
        )

        model = load_mapanything_model(model_name=self.model_name)
        views, image_paths = load_and_preprocess_images(image_dir)
        image_names = [p.name for p in image_paths]

        model_width = views[0]["img"].shape[-1]
        model_height = views[0]["img"].shape[-2]

        outputs = run_mapanything_inference(
            model,
            views,
            memory_efficient_inference=True,
            minibatch_size=self.minibatch_size,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            use_multiview_confidence=self.use_multiview_confidence,
            confidence_percentile=self.confidence_percentile,
        )

        # Export at model resolution, then rescale intrinsics to original dims
        sparse_dir = export_to_colmap(
            outputs, views, image_names, output_dir=output_dir, model=model
        )
        rescaled_sparse_dir = rescale_to_original_dimensions(
            sparse_dir, image_paths, model_width, model_height, output_dir=output_dir
        )

        return pycolmap.Reconstruction(str(rescaled_sparse_dir))
