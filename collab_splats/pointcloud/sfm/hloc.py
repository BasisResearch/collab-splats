"""
hloc incremental SfM: learned features and matches, then pycolmap incremental mapping.

- config: `pointcloud: {method: sfm, backend: hloc}`
- needs the optional `hloc` extra; install with setup/hloc.sh
- intermediates under <out_dir>/colmap/hloc/
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from collab_splats.pointcloud.sfm.base import BaseSfmCreator
from collab_splats.pointcloud.sfm.sift_db import PAIRINGS

logger = logging.getLogger(__name__)

########################################################################
# Creator
########################################################################


@dataclass
class HlocCreator(BaseSfmCreator):
    """
    Learned-feature incremental SfM via hloc on a scene directory.

    - one shared SIMPLE_RADIAL camera, refined by the mapper
    - h5 features and matches are reused across runs by hloc itself

    Attributes:
        pairing: sequential | retrieval | sequential+retrieval | exhaustive.
        overlap: sequential neighbors per frame.
        num_retrieved: global-descriptor neighbors per frame when pairing retrieves.
        retrieval_conf: hloc.extract_features.confs key for global descriptors.
        feature_conf: hloc.extract_features.confs key for local features.
        matcher_conf: hloc.match_features.confs key.
    """

    pairing: str = "sequential+retrieval"
    overlap: int = 10
    num_retrieved: int = 20
    retrieval_conf: str = "netvlad"
    feature_conf: str = "superpoint_max"
    matcher_conf: str = "superpoint+lightglue"

    def __post_init__(self) -> None:
        """
        Refuse an unknown pairing before any feature extraction.
        """
        super().__post_init__()

        if self.pairing not in PAIRINGS:
            raise ValueError(f"pairing must be one of {PAIRINGS}, got {self.pairing!r}")

    def _map(self, images_dir: Path, out_dir: Path, names: list[str]) -> pycolmap.Reconstruction:
        """
        Extract, pair, match and map the keyframes with hloc; returns the largest model, in memory.
        """
        # Import hloc here so the other backends work without it installed
        try:
            from hloc import (
                extract_features,
                match_features,
                pairs_from_exhaustive,
                pairs_from_retrieval,
                reconstruction,
            )
        except ImportError as err:
            raise ImportError(
                "hloc is not installed — run `bash setup/hloc.sh`, then `uv lock` and `bash setup.sh` "
                "(the `hloc` extra is an editable path source on third_party/hloc)"
            ) from err

        # Working dir for features, pairs, matches and the mapper
        hloc_dir = out_dir / "colmap" / "hloc"
        hloc_dir.mkdir(parents=True, exist_ok=True)

        # Extract local features for every keyframe
        feature_conf = extract_features.confs[self.feature_conf]
        features = extract_features.main(feature_conf, images_dir, hloc_dir, image_list=names)

        # Build the list of image pairs to match
        pairs_path = hloc_dir / f"pairs-{self.pairing}.txt"

        if self.pairing == "exhaustive":
            pairs_from_exhaustive.main(pairs_path, image_list=names)
        else:
            # Start with sequential pairs, then add retrieval pairs (hloc removes duplicates)
            lines = (
                [f"{a} {b}\n" for a, b in sequential_pairs(names, self.overlap)] if "sequential" in self.pairing else []
            )

            if "retrieval" in self.pairing:
                retrieval_conf = extract_features.confs[self.retrieval_conf]
                descriptors = extract_features.main(retrieval_conf, images_dir, hloc_dir, image_list=names)
                retrieval_path = hloc_dir / "pairs-retrieval.txt"
                pairs_from_retrieval.main(
                    descriptors,
                    retrieval_path,
                    num_matched=min(self.num_retrieved, len(names) - 1),  # topk fails if k exceeds the image count
                    query_list=names,
                    db_list=names,
                )
                lines.append(retrieval_path.read_text())

            pairs_path.write_text("".join(lines))

        # Match features across each image pair
        matches = match_features.main(
            match_features.confs[self.matcher_conf], pairs_path, feature_conf["output"], hloc_dir
        )

        # Run incremental mapping in a fresh folder
        sfm_dir = hloc_dir / "sfm"
        shutil.rmtree(sfm_dir, ignore_errors=True)  # old model folders would skew the split count below
        recon = reconstruction.main(
            sfm_dir,
            images_dir,
            pairs_path,
            features,
            matches,
            camera_mode=pycolmap.CameraMode.SINGLE,
            image_list=names,
            image_options={"camera_model": "SIMPLE_RADIAL"},
            mapper_options={"num_threads": self.num_threads},
        )

        if recon is None:
            raise RuntimeError("hloc produced no model — too little overlap between frames")

        # Warn if the scene split into several separate models
        models = [p for p in (sfm_dir / "models").iterdir() if p.is_dir()]

        if len(models) > 1:
            logger.warning("hloc split the scene into %d models — keeping the largest", len(models))

        return recon


########################################################################
# Pair lists
########################################################################


def sequential_pairs(names: list[str], overlap: int) -> list[tuple[str, str]]:
    """
    Pair each frame with the next `overlap` frames.

    - same pairs as COLMAP's sequential matcher without quadratic overlap

    Args:
        names: image names in capture order.
        overlap: forward neighbors per frame.

    Returns:
        (earlier, later) pairs, ordered by the earlier frame.
    """
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1 : i + 1 + overlap]]
