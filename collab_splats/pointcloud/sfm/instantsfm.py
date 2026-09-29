"""
InstantSfM global SfM through its python API.

- upstream: https://github.com/cre185/InstantSfM @ d3e599e (0.3.0)
- call pattern follows upstream scripts/sfm.py::run_sfm
- _to_pycolmap mirrors controllers/reconstruction_writer.py:102-141 (WriteGlomapReconstruction)
- SIFT database built with pycolmap (sift_db.py), not upstream's DB step
- three upstream compat monkeypatches (_patch_*), applied on each _map call
- license CC-BY-NC-4.0 (non-commercial): cleared for research use, revisit before commercial use
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pycolmap

from collab_splats.pointcloud.sfm.base import BaseSfmCreator
from collab_splats.pointcloud.sfm.sift_db import ensure_sift_database

if TYPE_CHECKING:
    from instantsfm.controllers.config import Config
    from instantsfm.scene.defs import Cameras, Images, Tracks

logger = logging.getLogger(__name__)


########################################################################
# Creator
########################################################################


@dataclass
class InstantSfMCreator(BaseSfmCreator):
    """
    Global SfM via InstantSfM on a scene directory.

    - SIFT DB at <out_dir>/colmap/instantsfm.db, exhaustive pairing
    - depth priors read from <out_dir>/depth_vda/

    Attributes:
        use_depths: feed depth_vda/ maps into the solve as depth priors.
        retriangulation: retriangulate and re-run BA after the global solve.
        random_seed: seed for the solve; None leaves it unseeded, so runs differ.
        min_num_view_per_track: drop tracks seen in fewer views; None keeps upstream's 3.
    """

    use_depths: bool = True
    retriangulation: bool = False
    random_seed: int | None = None
    min_num_view_per_track: int | None = None

    def _build_config(self) -> Config:
        """
        Upstream Config built from the creator fields.

        - Config shares module-level dicts, so each dict is copied before it is written
        """
        # Import InstantSfM here since it is an optional install with CUDA extensions
        from instantsfm.controllers.config import Config

        # InstantSfM ignores the handler name, so any value works
        config = Config("colmap")
        config.OPTIONS = dict(config.OPTIONS)
        config.RUNTIME_OPTIONS = dict(config.RUNTIME_OPTIONS)

        # Turn retriangulation after the global solve on or off
        config.OPTIONS["skip_retriangulation"] = not self.retriangulation

        # Set a random seed so runs are repeatable
        if self.random_seed is not None:
            config.RUNTIME_OPTIONS["random_seed"] = self.random_seed

        # Drop tracks seen in too few views, which lowers bundle adjustment memory
        if self.min_num_view_per_track is not None:
            config.TRACK_ESTABLISHMENT_OPTIONS = dict(config.TRACK_ESTABLISHMENT_OPTIONS)
            config.TRACK_ESTABLISHMENT_OPTIONS["min_num_view_per_track"] = self.min_num_view_per_track

        return config

    def _map(self, images_dir: Path, out_dir: Path, names: list[str]) -> pycolmap.Reconstruction:
        """
        InstantSfM over the keyframes; returns the model in memory.

        - depth priors from out_dir/depth_vda/ when use_depths
        - SIFT DB at out_dir/colmap/instantsfm.db, reused on re-runs
        """
        # Import InstantSfM here since it is an optional install with CUDA extensions
        from instantsfm.controllers.data_reader import (
            ReadColmapDatabase,
            ReadData,
            ReadDepthsIntoFeatures,
        )
        from instantsfm.controllers.global_mapper import SolveGlobalMapper

        # Patch known bugs in InstantSfM and its dependencies
        _patch_instantsfm_track_ids()
        _patch_pypose_robustmodel_target()
        _patch_bae_pcg_column_shape()

        # Find the depth maps folder, and fail if depths are needed but missing
        path_info = ReadData(str(out_dir))

        if self.use_depths and not path_info.depth_path:
            raise RuntimeError(f"no depth_vda/ under {out_dir} — estimate_depth must run first")

        colmap_dir = out_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)

        # Build or reuse the SIFT DB for this image set
        db_path = colmap_dir / "instantsfm.db"
        ensure_sift_database(
            images_dir,
            db_path,
            names,
            pairing="exhaustive",
            num_threads=self.num_threads,
        )

        # Load the database into InstantSfM's own data types
        view_graph, cameras, images, _feature_name, _rig = ReadColmapDatabase(str(db_path))

        if view_graph is None or cameras is None or images is None:
            raise RuntimeError(f"InstantSfM could not read {db_path}")

        # Configure the solver, and load the depth maps as priors when enabled
        config = self._build_config()
        config.RUNTIME_OPTIONS["use_depths"] = self.use_depths

        if self.use_depths:
            logger.info("InstantSfM: loading depths from %s", path_info.depth_path)

            for idx in range(len(images)):
                camera = cameras[images[idx].cam_id]
                images.features[idx] = _nudge_edge_keypoints(images.features[idx], camera.width, camera.height)

            ReadDepthsIntoFeatures(path_info.depth_path, cameras, images)

        # Run global mapping, turning its unhelpful IndexError into a clear error
        try:
            cameras, images, tracks = SolveGlobalMapper(view_graph, cameras, images, config, visualizer=None)
        except IndexError as err:
            logger.exception("InstantSfM SolveGlobalMapper raised IndexError")
            raise RuntimeError(
                "InstantSfM global mapping raised IndexError — usually every track was filtered out "
                "(sparse/low-overlap frames, upstream numpy-2 empty-mask path) or depth priors failed "
                "to load; see chained cause"
            ) from err

        if not tracks:
            raise RuntimeError("InstantSfM produced zero tracks — reconstruction is empty")

        # Convert to a pycolmap model in memory
        return _to_pycolmap(cameras, images, tracks, images_dir)


########################################################################
# Upstream compat patches
########################################################################


def _nudge_edge_keypoints(features: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Pull keypoints sitting exactly on the far image edge (x == width or y == height) inward.

    - upstream depth sampling raises IndexError on such keypoints
    - coords beyond the edge are left alone; upstream marks them as having no depth
    """
    features = np.asarray(features)

    if features.size == 0:
        return features

    # Move keypoints on the right or bottom edge slightly inside the image
    nudged = features.copy()
    nudged[nudged[:, 0] == width, 0] = width - 1e-3
    nudged[nudged[:, 1] == height, 1] = height - 1e-3

    return nudged


def _patch_instantsfm_track_ids() -> None:
    """
    Renumber track ids to sequential ints before they reach int32 storage.

    - upstream packs 64-bit track ids but stores them as int32; numpy 2 raises OverflowError
    - track ids are opaque labels, so renumbering is lossless
    - idempotent
    """
    # Import InstantSfM here since it is an optional install with CUDA extensions
    from instantsfm.processors.track_establishment import TrackEngine

    if getattr(TrackEngine.FindTracksForProblem, "_collab_splats_renumber", False):
        return

    # Wrap the original method so it renumbers track ids first
    upstream_find_tracks = TrackEngine.FindTracksForProblem

    def find_tracks_renumbered(self, tracks_full, TRACK_ESTABLISHMENT_OPTIONS):
        renumbered = dict(enumerate(tracks_full.values()))

        return upstream_find_tracks(self, renumbered, TRACK_ESTABLISHMENT_OPTIONS)

    find_tracks_renumbered._collab_splats_renumber = True
    TrackEngine.FindTracksForProblem = find_tracks_renumbered


def _patch_pypose_robustmodel_target() -> None:
    """
    Default target=None on pypose RobustModel.forward for bae's LM.

    - bae LM.step passes no target; pypose 0.7.5 requires one, so every step raises TypeError
    - class-level, since instantsfm builds its optimizers internally
    - target=None: residuals are the raw model output, the intended objective
    - idempotent
    """
    # Import here since it is only needed for InstantSfM
    from pypose.optim.optimizer import RobustModel

    if getattr(RobustModel.forward, "_collab_splats_default_target", False):
        return

    # Wrap the original method so target defaults to None
    upstream_forward = RobustModel.forward

    def forward_default_target(self, input, target=None):
        return upstream_forward(self, input, target)

    forward_default_target._collab_splats_default_target = True
    RobustModel.forward = forward_default_target


def _patch_bae_pcg_column_shape() -> None:
    """
    Make bae's PCG solver return a column vector for a column-vector rhs.

    - pypose 0.7.5 CG returns 1-D for an (n, 1) rhs
    - bae LM.step passes a column rhs, so pypose TrustRegion.update then fails
    - a column result matches the pre-0.7.5 CG contract
    - idempotent
    """
    # Import here since it is only needed for InstantSfM
    from bae.utils.pysolvers import PCG

    if getattr(PCG.forward, "_collab_splats_column_shape", False):
        return

    # Wrap the original method to restore the column shape of the result
    upstream_pcg_forward = PCG.forward

    def forward_keep_column(self, A, b, x=None, M=None):
        res = upstream_pcg_forward(self, A, b, x, M)

        if b.dim() == 2 and res.dim() == 1:
            res = res[:, None]

        return res

    forward_keep_column._collab_splats_column_shape = True
    PCG.forward = forward_keep_column


########################################################################
# Model conversion
########################################################################


def _to_pycolmap(cameras: Cameras, images: Images, tracks: Tracks, image_dir: Path) -> pycolmap.Reconstruction:
    """
    InstantSfM solve as an in-memory pycolmap model, matching upstream's sparse/0 export.

    - keeps registered, rig-complete images; a split solve keeps cluster 0 only (ADR 018)
    - ids are upstream array indices; every image keeps its full keypoint list
    - correspondences built for tracks of length 3 or more
    - a track keeps only observations that link back to it
    - reads upstream private attrs (_selected_indices, _point3d_ids), which have no accessor
    """
    # Import InstantSfM here since it is an optional install with CUDA extensions
    from instantsfm.controllers.reconstruction_writer import FilterRigCompleteness
    from instantsfm.scene.reconstruction import Reconstruction as InsfmReconstruction

    # Keep registered images, and only cluster 0 if the scene split
    keep = FilterRigCompleteness(images)

    if not keep.any():
        raise RuntimeError("InstantSfM registered no images")

    clusters = np.unique(images.cluster_ids[keep]).tolist()

    if len(clusters) > 1:
        if 0 not in clusters:
            raise RuntimeError(f"InstantSfM has no cluster 0 among {clusters}: use more frames or overlap")

        logger.warning("InstantSfM split the scene into clusters %s — keeping cluster 0 only", clusters)
        keep &= images.cluster_ids == 0

    # Let InstantSfM compute point ids and colors for the kept images
    insfm = InsfmReconstruction(cameras, images, tracks)
    insfm._selected_indices = np.flatnonzero(keep)
    insfm.build_correspondences(min_track_length=3)
    insfm.extract_colors_batch(str(image_dir))

    # Add each camera to the model
    recon = pycolmap.Reconstruction()

    for i, cam in enumerate(cameras):
        camera = pycolmap.Camera(camera_id=i, model=cam.model, width=cam.width, height=cam.height, params=cam.params)
        recon.add_camera_with_trivial_rig(camera)

    # Add each kept image with its pose and keypoints
    for idx in insfm._selected_indices.tolist():
        world2cam = images.world2cams[idx][:3].astype(np.float64)
        cam_from_world = pycolmap.Rigid3d(world2cam)
        name = images.filenames[idx]
        keypoints = images.features[idx]
        cam_id = int(images.cam_ids[idx])
        image = pycolmap.Image(name=name, keypoints=keypoints, camera_id=cam_id, image_id=idx)
        recon.add_image_with_trivial_frame(image, cam_from_world)

    # Add each track as a 3D point, keeping only observations that point back to it
    for track_id in range(len(tracks)):
        track = pycolmap.Track()

        for image_id, feat_idx in tracks.observations[track_id].tolist():
            ids = insfm._point3d_ids[image_id]

            if ids is not None and feat_idx < len(ids) and ids[feat_idx] == track_id:
                track.add_element(image_id, feat_idx)

        point = pycolmap.Point3D(xyz=tracks.xyzs[track_id], color=tracks.colors[track_id], error=0.0, track=track)
        recon.add_point3D_with_id(track_id, point)

    return recon
