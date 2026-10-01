"""
COLMAP incremental SfM: pycolmap SIFT, then pycolmap incremental mapping.

- config: `pointcloud: {method: sfm, backend: colmap}`
- SIFT DB at <out_dir>/colmap/colmap.db, reused while its images and matching params match
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from collab_splats.pointcloud.sfm.base import BaseSfmCreator
from collab_splats.pointcloud.sfm.sift_db import (
    ensure_sift_database,
    fetch_vocab_tree,
)

logger = logging.getLogger(__name__)


########################################################################
# Creator
########################################################################


@dataclass
class ColmapCreator(BaseSfmCreator):
    """
    Classical incremental SfM on a scene directory.

    - features, matches and mapping all run in pycolmap
    - one shared SIMPLE_RADIAL camera, refined by the mapper
    - num_retrieved neighbors come from a vocab tree (9.5 MB, fetched once)
    """

    def _map(self, images_dir: Path, out_dir: Path, names: list[str]) -> pycolmap.Reconstruction:
        """
        SIFT and incremental mapping over the keyframes; returns the largest model, in memory.
        """
        # Create a working folder for COLMAP files
        colmap_dir = out_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)

        # Build the SIFT feature database, or reuse it if it already matches
        db_path = colmap_dir / "colmap.db"
        ensure_sift_database(
            images_dir,
            db_path,
            names,
            pairing=self.pairing,
            overlap=self.overlap,
            num_retrieved=self.num_retrieved,
            vocab_tree=fetch_vocab_tree if "retrieval" in self.pairing else None,  # downloaded only on a rebuild
            num_threads=self.num_threads,
        )

        # Run incremental mapping in a fresh folder
        mapper_dir = colmap_dir / "mapper"
        shutil.rmtree(mapper_dir, ignore_errors=True)
        mapper_dir.mkdir()
        recons = pycolmap.incremental_mapping(
            str(db_path), str(images_dir), str(mapper_dir), options={"num_threads": self.num_threads}
        )

        # Keep the model with the most registered images
        if not recons:
            raise RuntimeError("incremental mapping produced no model — too little overlap between frames")

        recon = max(recons.values(), key=lambda r: r.num_reg_images())

        if len(recons) > 1:
            sizes = sorted((r.num_reg_images() for r in recons.values()), reverse=True)
            logger.warning(
                "incremental mapping split the scene into %d models %s — keeping the largest", len(recons), sizes
            )

        # Delete the working folder, since the model is already in memory
        shutil.rmtree(mapper_dir)

        return recon
