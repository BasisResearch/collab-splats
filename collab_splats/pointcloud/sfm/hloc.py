"""
hloc SfM backend: retrieval-gated learned features and matching into COLMAP's mapper.

- hloc is a third_party clone, not a locked dependency — imported inside reconstruct
- unwired — nothing dispatches to HlocCreator, see sfm/__init__.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..base import BasePointcloudCreator, PointcloudResult

logger = logging.getLogger(__name__)


@dataclass
class HlocCreator(BasePointcloudCreator):
    """
    Pointcloud via hloc: retrieval -> learned features -> learned matching -> COLMAP mapper.

    - retrieval_conf: hloc retrieval config key; see `hloc.extract_features.confs`.
    - feature_conf: hloc local-feature config key; see `hloc.extract_features.confs`.
    - matcher_conf: hloc matcher config key; see `hloc.match_features.confs`.
    - Writes the binary model to output_dir/colmap/sparse/0, hloc intermediates to colmap/hloc/.
    """

    retrieval_conf: str = "netvlad"
    feature_conf: str = "superpoint_aachen"
    matcher_conf: str = "superglue"

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        """
        Retrieval-gated learned matching and COLMAP incremental mapping over image_dir.

        Args:
            image_dir:  Directory of input images; must exist.
            output_dir: Run directory. The binary model lands in colmap/sparse/0, hloc
                intermediates in colmap/hloc/.

        Returns:
            PointcloudResult wrapping the reconstructed model.

        Raises:
            ImportError: hloc is not installed (a third_party clone, not a locked dependency).
        """
        # hloc is a third_party clone, not a locked dependency — import inside the one method
        # that needs it so the module (and the registry) import without it installed
        try:
            from hloc import (
                extract_features,
                match_features,
                pairs_from_retrieval,
                reconstruction,
            )
        except ImportError as err:
            raise ImportError(
                "hloc is not installed — run `bash setup/hloc.sh` to clone and install it"
            ) from err

        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        hloc_dir = output_dir / "colmap" / "hloc"
        hloc_dir.mkdir(parents=True, exist_ok=True)

        # Retrieval first: O(N) candidate pairs instead of the O(N^2) exhaustive set
        retrieval_path = extract_features.main(extract_features.confs[self.retrieval_conf], image_dir, hloc_dir)
        pairs_path = hloc_dir / "pairs.txt"
        pairs_from_retrieval.main(retrieval_path, pairs_path)

        # Learned features + matcher over those pairs, then the COLMAP incremental mapper
        feature_path = extract_features.main(extract_features.confs[self.feature_conf], image_dir, hloc_dir)
        match_path = match_features.main(
            match_features.confs[self.matcher_conf],
            pairs_path,
            features=feature_path,
            matches=hloc_dir / "matches.h5",
        )
        recon = reconstruction.main(
            sfm_dir=sparse_dir,
            image_dir=image_dir,
            pairs=pairs_path,
            features=feature_path,
            matches=match_path,
        )
        if recon is None:
            raise RuntimeError("reconstruction failed — hloc returned None")

        logger.info("hloc: %d registered images, %d points3D", len(recon.images), len(recon.points3D))
        image_paths = sorted(
            [image_dir / img.name for img in recon.images.values()],
            key=lambda p: p.name,
        )
        return PointcloudResult(reconstruction=recon, image_paths=image_paths)
