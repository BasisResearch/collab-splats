"""
SfM skeleton shared by every sfm backend: depth, mapper, registered subset, align.

- backends implement _map only
- input: the scene images/ store, read in place
- output: a PointcloudResult, plus the mapper's COLMAP model at model_dir with stem image names
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

from collab_splats.pointcloud.base import BasePointcloudCreator, PointcloudResult
from collab_splats.pointcloud.depth import align_depth, estimate_depth
from collab_splats.pointcloud.sfm.sift_db import PAIRINGS
from collab_splats.preproc import frames
from collab_splats.utils.torch_utils import pytorch_gc

logger = logging.getLogger(__name__)


########################################################################
# Skeleton
########################################################################


def _require_positive_int(name: str, value: object) -> None:
    """
    Refuse anything but an int >= 1.

    - bool is an int subclass, so it is refused explicitly
    - ValueError names the argument
    """
    if isinstance(value, bool) or not (isinstance(value, int) and value >= 1):
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")


@dataclass
class BaseSfmCreator(BasePointcloudCreator):
    """
    SfM skeleton: VDA depth, backend mapper, registered subset, depth align.

    - backends implement _map only
    - the result holds only the frames the mapper registered
    - the exported COLMAP model is the mapper's own, tracks and camera model kept

    Attributes:
        pairing: sequential | retrieval | sequential+retrieval | exhaustive.
        overlap: sequential neighbors per frame.
        num_retrieved: retrieval neighbors per frame when pairing retrieves.
        min_registered_frac: minimum share of frames the mapper must register, in (0, 1].
        num_threads: thread cap for CPU SIFT and the mapper.
        attrs: pointcloud.zarr attrs of the last run (method, registered subset, depth alignment).
    """

    pairing: str = "sequential+retrieval"
    overlap: int = 10
    num_retrieved: int = 20
    min_registered_frac: float = 0.5
    num_threads: int = 8
    attrs: dict = field(default_factory=dict, init=False, repr=False)

    # COLMAP model from the last run, kept so it can be exported with its tracks
    _recon: pycolmap.Reconstruction | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Refuse an unknown pairing, a non-positive count, or a floor outside (0, 1].
        """
        # Pairing must name one of sift_db's modes
        if self.pairing not in PAIRINGS:
            raise ValueError(f"pairing must be one of {PAIRINGS}, got {self.pairing!r}")

        # Pair counts are positive ints
        _require_positive_int("overlap", self.overlap)
        _require_positive_int("num_retrieved", self.num_retrieved)

        # Floor must be a real number in (0, 1]
        frac = self.min_registered_frac

        if isinstance(frac, bool) or not (
            isinstance(frac, (int, float)) and 0 < frac <= 1
        ):
            raise ValueError(f"min_registered_frac must be in (0, 1], got {frac!r}")

        _require_positive_int("num_threads", self.num_threads)

    def _colmap_model(self, result: PointcloudResult) -> pycolmap.Reconstruction:
        """
        The mapper's model, trimmed to the points the clean and cap kept.

        - tracks and the mapper's camera model survive, distortion included
        - the export and the result hold one point set
        """
        # Take the model off the creator so its memory is freed with the export
        recon, self._recon = self._recon, None
        assert recon is not None

        # Delete the points3D the clean or cap removed, matched on the float32 xyz the result holds
        kept = {tuple(xyz) for xyz in result.points.tolist()}
        dropped = [
            pid
            for pid, p in recon.points3D.items()
            if tuple(p.xyz.astype(np.float32).tolist()) not in kept
        ]

        for pid in dropped:
            recon.delete_point3D(pid)

        return recon

    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        """
        VDA depth, the backend mapper, the registered subset, then the depth align.

        - keeps the mapper's model on self._recon for _colmap_model
        """
        # Estimate metric depth for every keyframe, then free the loaded frames
        images_dir = paths[0].parent
        names = [p.name for p in paths]
        store = frames.read_frames(images_dir)
        depths = estimate_depth(store, out_dir, names)
        del store

        # Run the backend's mapper, then free GPU memory
        recon = self._map(images_dir, out_dir, names)
        pytorch_gc()

        # Find which input frames the mapper registered, matching on file name without extension
        backend = type(self).__name__
        registered = {Path(recon.images[i].name).stem for i in recon.reg_image_ids()}
        rows = [row for row, name in enumerate(names) if Path(name).stem in registered]

        # Fail if too few frames were registered, and warn if only some were
        if len(rows) < self.min_registered_frac * len(names):
            missing = [
                Path(name).stem for name in names if Path(name).stem not in registered
            ]
            raise RuntimeError(
                f"{backend} registered {len(rows)}/{len(names)} frames, below min_registered_frac "
                f"{self.min_registered_frac} (unregistered: {missing[:10]})"
            )

        if len(rows) < len(names):
            logger.warning(
                "%s registered %d/%d frames — continuing on the registered subset",
                backend,
                len(rows),
                len(names),
            )

        # Remove unregistered images from the model
        recon.tear_down()
        logger.info("%s: %d/%d registered", backend, len(rows), len(names))

        # Delete 3D points that no image sees anymore
        for pid in [
            pid
            for pid, point in recon.points3D.items()
            if len(point.track.elements) == 0
        ]:
            recon.delete_point3D(pid)

        # Rename images to their file name without extension
        for image in recon.images.values():
            image.name = Path(image.name).stem

        # Keep the model so it can be exported later
        self._recon = recon

        # Build the dense point cloud by aligning the depth maps to the COLMAP model
        registered_idxs = [frames.frame_idx_from_path(paths[row]) for row in rows]
        keyframes = frames.read_frames(images_dir, registered_idxs)
        registered_names = [names[row] for row in rows]
        result, align_attrs = align_depth(
            recon, depths[rows], keyframes, registered_names
        )

        # Record the method, frame counts and depth alignment stats
        self.attrs = {
            "method": "sfm",
            "registered_frames": len(rows),
            "total_frames": len(names),
            **align_attrs,
        }

        return result

    @abstractmethod
    def _map(
        self, images_dir: Path, out_dir: Path, names: list[str]
    ) -> pycolmap.Reconstruction:
        """
        Backend mapper: one in-memory COLMAP model over `names`.

        - image names as in images_dir, extensions kept
        """
