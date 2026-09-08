"""
InstantSfM global SfM: the one backend `pointcloud.method: sfm` dispatches to.

- drives the upstream python API (cre185/InstantSfM @ 0.3.0), never their CLI
- SIFT database is built here with system colmap; upstream's own DB step is bypassed
- four upstream monkeypatches live here (_patch_*), each documented at its definition
- output contract: image names are filename stems, model at <data_dir>/colmap/sparse/0
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import torch

from collab_splats.preproc.frames import frame_paths

logger = logging.getLogger(__name__)


########################################
# SIFT feature database (system colmap)
########################################


def _nudge_edge_keypoints(features: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Pull keypoints sitting exactly on the far image edge (x == width or y == height) inward.

    - Upstream ``sample_depth_at_pixel`` rejects coords with ``x / width > 1`` but lets
      ``== 1`` through, then indexes ``depth_map[:, W]`` -> IndexError. SIFT emits such
      keypoints rarely (2 of 3M on GH010229 undistorted); coords beyond the edge are left
      alone so upstream still marks them depth-unavailable.
    """
    features = np.asarray(features)
    if features.size == 0:
        return features
    nudged = features.copy()
    nudged[nudged[:, 0] == width, 0] = width - 1e-3
    nudged[nudged[:, 1] == height, 1] = height - 1e-3
    return nudged


def _sfm_image_dir(images_dir: Path) -> Path:
    """
    Image directory InstantSfM reads.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        The same directory — the store IS the COLMAP image layout, so nothing is staged.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"InstantSfM expects an image directory at {images_dir} — run preprocess first")
    return images_dir


def _sift_database_valid(database_path: Path, image_names: list[str]) -> bool:
    """
    True when the SIFT DB holds extraction and matching output for exactly `image_names`.

    Args:
        database_path: the scene's colmap/instantsfm.db.
        image_names: filenames this run is about to reconstruct, in any order.

    Returns:
        False when the DB is missing, partial, or keyed on a different image set.

    - A crashed colmap subprocess (e.g. OOM-killed under the cgroup cap) leaves a partial or
      unreadable DB behind; an existence-only check would cache-hit on it and feed
      ReadColmapDatabase zero tracks.
    - Database.open CREATES the file when absent, so the exists() pre-check must stay; it
      raises RuntimeError (not sqlite3.Error) on a file no registered factory can open.
    - Nothing stages a per-run image copy any more, so the DB's own image table is what says
      which selection it was extracted from. Without this the caller would reuse SIFT features
      for frames that are no longer in the scene.
    """
    if not database_path.exists():
        return False
    try:
        db = pycolmap.Database.open(str(database_path))
    except RuntimeError:
        return False
    try:
        registered = {image.name for image in db.read_all_images()}
        if registered != set(image_names):
            logger.info(
                "SIFT database %s holds %d images against this run's %d — rebuilding",
                database_path,
                len(registered),
                len(image_names),
            )
            return False
        return db.num_keypoints() > 0 and db.num_verified_image_pairs() > 0
    finally:
        db.close()


def _generate_sift_database(image_path: Path, database_path: Path, *, num_threads: int = 8) -> None:
    """
    Build the COLMAP SIFT feature database: extraction + exhaustive matching.

    - num_threads: CPU SIFT thread cap. colmap's default (-1) spawns one thread per HOST core
      — 96 here — and per-thread RAM blows past the 46.6 GB container cgroup cap (measured:
      OOM-kill at default, clean 1.5 min run at 8 threads on 100 frames of 1920x1080).
    - Drives the colmap CLI, not pycolmap: the system binary is the CUDA build (GPU SIFT,
      measured 100x1920x1080 extraction 7 s vs 90 s, matching 55 s vs ~816 s) while the wheel
      is CPU-only. Reimplements upstream GenerateDatabase (cre185/InstantSfM
      instantsfm/controllers/feature_handler.py:18-57 @ d3e599e), which forces CPU with no
      thread cap and swallows CalledProcessError — a colmap crash there surfaces only as an
      empty-tracks IndexError much later.
    - On failure the partial DB is unlinked so a re-run rebuilds from scratch.
    """
    env = os.environ.copy()
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    # One shared SIMPLE_RADIAL camera over the whole set
    # - the sfm path stages frames from a single video/scene, so single_camera holds by
    #   construction (it was a creator field once and was never set False)
    # - per-image cameras would leave every intrinsic solved from one view
    extractor_cmd = [
        "colmap",
        "feature_extractor",
        "--image_path",
        str(image_path),
        "--database_path",
        str(database_path),
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        "1" if use_gpu else "0",
    ]
    matcher_cmd = [
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--SiftMatching.use_gpu",
        "1" if use_gpu else "0",
    ]
    if not use_gpu:
        extractor_cmd += ["--SiftExtraction.num_threads", str(num_threads)]
        matcher_cmd += ["--SiftMatching.num_threads", str(num_threads)]

    try:
        for cmd in (extractor_cmd, matcher_cmd):
            logger.info("InstantSfM: running %s %s (%s)", cmd[0], cmd[1], "gpu" if use_gpu else "cpu")
            subprocess.run(cmd, check=True, env=env)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        database_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"COLMAP SIFT database build failed ({err}) — is the `colmap` binary installed "
            "(CUDA-built for the GPU path) and is there enough memory?"
        ) from err


def _patch_instantsfm_track_ids() -> None:
    """
    Renumber track ids to sequential ints before they reach int32 storage.

    - Upstream keys tracks by packed 64-bit global point ids ((image_id << 32) | feature_idx,
      cre185/InstantSfM @ d3e599e instantsfm/processors/track_establishment.py:56) but stores
      them in an int32 array (instantsfm/scene/defs.py:339) — numpy 1.x wrapped these silently
      (with collision risk), numpy 2 raises OverflowError for any track rooted past image 0.
    - Track ids are opaque labels downstream (dict keys in track_retriangulation, max()+1
      allocation), so a compact renumber is lossless. Idempotent across creator instances.
    """
    # Lazy heavy import — instantsfm is an optional dep (CUDA extensions)
    from instantsfm.processors.track_establishment import TrackEngine

    if getattr(TrackEngine.FindTracksForProblem, "_collab_splats_renumber", False):
        return
    upstream_find_tracks = TrackEngine.FindTracksForProblem

    def find_tracks_renumbered(self, tracks_full, TRACK_ESTABLISHMENT_OPTIONS):
        renumbered = dict(enumerate(tracks_full.values()))
        return upstream_find_tracks(self, renumbered, TRACK_ESTABLISHMENT_OPTIONS)

    find_tracks_renumbered._collab_splats_renumber = True
    TrackEngine.FindTracksForProblem = find_tracks_renumbered


def _patch_pypose_robustmodel_target() -> None:
    """
    Default target=None on pypose RobustModel.forward for bae's LM.

    - instantsfm's global positioning/BA drive `bae.optim.LM`, whose step calls
      self.model(input) with no target; pypose 0.7.5 RobustModel.forward requires
      target positionally, so every LM step raises TypeError (same incompatibility
      geometry/bundle_adjustment.py:401 binds away per-instance — instantsfm builds
      its optimizers internally, so the class-level default is the only reachable fix).
    - target=None makes residuals fall back to the raw model output, the intended
      objective. Idempotent.
    """
    # Lazy heavy import — instantsfm is an optional dep (CUDA extensions)
    from pypose.optim.optimizer import RobustModel

    if getattr(RobustModel.forward, "_collab_splats_default_target", False):
        return
    upstream_forward = RobustModel.forward

    def forward_default_target(self, input, target=None):
        return upstream_forward(self, input, target)

    forward_default_target._collab_splats_default_target = True
    RobustModel.forward = forward_default_target


def _patch_bae_pcg_column_shape() -> None:
    """
    Make bae's PCG solver return a column vector for a column-vector rhs.

    - pypose 0.7.5 CG.forward squeezes an (n, 1) rhs to 1-D and returns 1-D;
      bae's PCG wrapper (bae/utils/pysolvers.py:25-38) only restores the shape
      when the CALLER passed 1-D. bae LM.step passes -J_T @ R.view(-1, 1), gets a
      1-D step back, and pypose TrustRegion.update then dies on (J @ D).mT
      ("tensor.mT is only supported on matrices..."). instantsfm hardcodes
      PCG(tol=1e-5); our own BA avoids this only because it prefers CuDSS.
    - Restoring the column dimension matches the pre-0.7.5 CG contract
      ("layout is the same as the layout of b"). Idempotent.
    """
    # Lazy heavy import — instantsfm is an optional dep (CUDA extensions)
    from bae.utils.pysolvers import PCG

    if getattr(PCG.forward, "_collab_splats_column_shape", False):
        return
    upstream_pcg_forward = PCG.forward

    def forward_keep_column(self, A, b, x=None, M=None):
        res = upstream_pcg_forward(self, A, b, x, M)
        if b.dim() == 2 and res.dim() == 1:
            res = res[:, None]
        return res

    forward_keep_column._collab_splats_column_shape = True
    PCG.forward = forward_keep_column


def _patch_instantsfm_colmap_write() -> None:
    """
    Make the upstream COLMAP binary writer internally consistent so pycolmap can read it.

    - Upstream `_write_images_binary` (cre185/InstantSfM @ d3e599e instantsfm/scene/
      reconstruction.py:214-253) compresses each image's points2D list to the
      valid-track subset, while `_write_points3d_binary` (:255-275) writes track
      observations carrying ORIGINAL SIFT feature indices — pycolmap range-checks
      the pair and refuses the model (`vector::_M_range_check`). points3D.bin also
      keeps observations on unregistered images and on sub-min-track-length tracks,
      which exist in no written image.
    - Fix: images.bin gets the FULL per-image keypoint list (point3D id -1 = COLMAP
      invalid where no surviving track), so original feature indices stay valid;
      points3D.bin drops observations that don't round-trip through the per-image
      correspondence table built by build_correspondences.
    - Binary writers only (our path never exports text). In-memory mapping untouched;
      idempotent.
    """
    # Lazy heavy import — instantsfm is an optional dep (CUDA extensions)
    from instantsfm.scene.reconstruction import Reconstruction as InsfmReconstruction
    from instantsfm.utils.read_write_model import write_next_bytes
    from scipy.spatial.transform import Rotation

    if getattr(InsfmReconstruction._write_images_binary, "_collab_splats_consistent", False):
        return

    def write_images_full_points2d(self, filepath):
        # Upstream body with one change: no valid_mask compression of the keypoint list
        if self.images is None or self._selected_indices is None:
            return
        with open(filepath, "wb") as fid:
            write_next_bytes(fid, len(self._selected_indices), "Q")
            for idx in self._selected_indices:
                world2cam = self.images.world2cams[idx]
                tvec = world2cam[:3, 3]
                qvec = Rotation.from_matrix(world2cam[:3, :3]).as_quat()  # xyzw
                write_next_bytes(fid, int(idx), "i")  # index-as-id, as upstream
                write_next_bytes(fid, [float(qvec[3]), float(qvec[0]), float(qvec[1]), float(qvec[2])], "dddd")
                write_next_bytes(fid, tvec.tolist(), "ddd")
                write_next_bytes(fid, int(self.images.cam_ids[idx]), "i")
                filename = self.images.filenames[idx] if hasattr(self.images, "filenames") else f"{idx}.jpg"
                for char in filename:
                    write_next_bytes(fid, char.encode("utf-8"), "c")
                write_next_bytes(fid, b"\x00", "c")

                # FULL keypoint list keeps track observation indices valid; "q" packs
                # -1 as 0xFF..FF, COLMAP's invalid point3D id
                point3d_ids = self._point3d_ids[idx]
                features = self.images.features[idx]
                write_next_bytes(fid, len(features), "Q")
                for xy, p3d_id in zip(features, point3d_ids):
                    write_next_bytes(fid, [float(xy[0]), float(xy[1]), int(p3d_id)], "ddq")

    def write_points3d_consistent(self, filepath):
        # Upstream body with one change: observations filtered through the
        # correspondence table (drops unregistered images + sub-min-length tracks)
        if self.tracks is None:
            return
        with open(filepath, "wb") as fid:
            write_next_bytes(fid, len(self.tracks), "Q")
            for track_id in range(len(self.tracks)):
                obs = self.tracks.observations[track_id]
                kept = [
                    (int(image_id), int(feat_idx))
                    for image_id, feat_idx in obs
                    if self._point3d_ids[image_id] is not None
                    and feat_idx < len(self._point3d_ids[image_id])
                    and self._point3d_ids[image_id][feat_idx] == track_id
                ]
                write_next_bytes(fid, track_id, "Q")
                write_next_bytes(fid, self.tracks.xyzs[track_id].tolist(), "ddd")
                write_next_bytes(fid, [int(c) for c in self.tracks.colors[track_id]], "BBB")
                write_next_bytes(fid, 0.0, "d")  # error, as upstream
                write_next_bytes(fid, len(kept), "Q")
                for image_id, feat_idx in kept:
                    write_next_bytes(fid, [image_id, feat_idx], "ii")

    write_images_full_points2d._collab_splats_consistent = True
    write_points3d_consistent._collab_splats_consistent = True
    InsfmReconstruction._write_images_binary = write_images_full_points2d
    InsfmReconstruction._write_points3d_binary = write_points3d_consistent


########################################
# Output contract
########################################


def _rename_images_to_stems(recon: pycolmap.Reconstruction, sparse_dir: Path) -> None:
    """
    Rename COLMAP images to their filename stems and rewrite the binary model in place.

    - InstantSfM registers images under their filenames (frame_000000.png); the pipeline
      contract is frame_{source_idx:06d} with NO extension (see PointcloudResult.from_colmap).
    - pycolmap.Image.name is settable by reference, so the rename lands on the model itself.
    """
    for im in recon.images.values():
        im.name = Path(im.name).stem
    recon.write_binary(str(sparse_dir))


@dataclass
class InstantSfMCreator:
    """
    Global SfM via InstantSfM's python API on a scene directory.

    - Upstream: https://github.com/cre185/InstantSfM (IROS 2026), pinned at 0.3.0 in setup.sh.
      Drives the upstream python API directly (never their CLI); the call pattern follows
      instantsfm/scripts/sfm.py::run_sfm at that version.
    - use_depths: feed depth_vda/ maps into the solve as depth priors.
    - retriangulation: GLOMAP-style retriangulate + re-BA after the global solve.
    - min_num_view_per_track: track-establishment cut; None = upstream default (3).
    - random_seed: seeds InstantSfM's RUNTIME_OPTIONS; None = unseeded (upstream draws initial
      camera translations and track xyzs from an unseeded uniform, so runs differ).
    - reconstruct(data_dir): data_dir/images/ (+ depth_vda/) -> pycolmap.Reconstruction whose
      image names are filename stems (frame_NNNNNN); model written to data_dir/colmap/sparse/0,
      SIFT DB at data_dir/colmap/instantsfm.db.
    - Not a BasePointcloudCreator on purpose: a scene dir in, a Reconstruction out; the
      Reconstructor wraps it.
    - License: CC-BY-NC-4.0 (non-commercial) — cleared for this repo's research use; revisit
      before any commercial deployment.
    """

    use_depths: bool = True
    retriangulation: bool = False
    random_seed: int | None = None
    min_num_view_per_track: int | None = None

    def _build_config(self):
        """
        Upstream Config with the option dicts copied before mutation.

        - Config.__init__ aliases module-level dicts, so in-place mutation leaks across
          instances; every dict this method writes to is copied first.
        """
        # Lazy heavy import — instantsfm is an optional dep (CUDA extensions)
        from instantsfm.controllers.config import Config

        # The handler name is inert at v0.3.0
        # - upstream's DB step ignores it and always runs colmap SIFT + exhaustive matching
        #   (_generate_sift_database above)
        # - "colmap" is the only value that means anything today: a constant, not a knob
        config = Config("colmap")
        config.OPTIONS = dict(config.OPTIONS)
        config.RUNTIME_OPTIONS = dict(config.RUNTIME_OPTIONS)

        # Optional GLOMAP-style refinement
        # - retriangulate from the full pre-filter track set, then up to
        #   ba_global_max_refinements (5) further BA rounds
        # - upstream defaults skip_retriangulation True; this is their only post-BA knob
        config.OPTIONS["skip_retriangulation"] = not self.retriangulation

        # InstantSfM is nondeterministic unless random_seed is set
        # - InitializeRandomPositions draws camera translations and track xyzs from an
        #   unseeded np.random.uniform(-1, 1) (cre185/InstantSfM @ 0.3.0,
        #   instantsfm/processors/global_positioning.py:229-243), so two runs differ
        # - random_seed is an upstream RUNTIME_OPTION read by SolveGlobalMapper
        #   (instantsfm/controllers/global_mapper.py:25) that seeds numpy/random/torch/cuda
        # - neither we nor upstream's CLI sets it by default, so an absent key stays absent
        if self.random_seed is not None:
            config.RUNTIME_OPTIONS["random_seed"] = int(self.random_seed)

        # Track-establishment cut — the memory lever for large image sets
        # - FindTracksForProblem drops any track seen by fewer views
        #   (cre185/InstantSfM @ 0.3.0, instantsfm/processors/track_establishment.py:109)
        # - the BA normal equations scale with the surviving track count: 875 images at the
        #   upstream default of 3 produced 486k tracks and exhausted 44 GiB in cuDSS (after a
        #   20 GiB J^T J); 6 is the measured setting that fits
        # - retriangulation is unaffected: it rebuilds density from the PRE-filter track set
        #   through TRIANGULATOR_OPTIONS, whose own min_num_view_per_track stays 2
        if self.min_num_view_per_track is not None:
            config.TRACK_ESTABLISHMENT_OPTIONS = dict(config.TRACK_ESTABLISHMENT_OPTIONS)
            config.TRACK_ESTABLISHMENT_OPTIONS["min_num_view_per_track"] = int(self.min_num_view_per_track)

        return config

    def reconstruct(self, data_dir: Path, images_dir: Path | None = None) -> pycolmap.Reconstruction:
        """
        Run InstantSfM over data_dir (depth_vda/ from generate_vda_depth when use_depths).

        - works in the contract layout directly: SIFT DB at data_dir/colmap/instantsfm.db (reused on
          re-runs; GCS push excludes it; its own name so geometry/verification.py's colmap/database.db
          can never be mistaken for it), COLMAP binary at data_dir/colmap/sparse/0

        Args:
            data_dir:   Working directory — colmap/ and depth_vda/ live here.
            images_dir: COLMAP-shaped image directory to read; defaults to data_dir/images.
                The pipeline passes the scene's own images/ so nothing is staged.

        Returns:
            The pycolmap.Reconstruction read back from the written model.
        """
        # Lazy heavy import — instantsfm is an optional dep (CUDA extensions)
        from instantsfm.controllers.data_reader import (
            ReadColmapDatabase,
            ReadData,
            ReadDepthsIntoFeatures,
        )
        from instantsfm.controllers.global_mapper import SolveGlobalMapper
        from instantsfm.controllers.reconstruction_writer import (
            WriteGlomapReconstruction,
        )

        # Upstream compat fixes
        # - packed 64-bit track ids vs int32 storage (numpy 2)
        # - bae LM.step vs pypose-0.7.5 RobustModel.forward(target)
        # - PCG 1-D step vs TrustRegion.update
        # - COLMAP writer emitting a model pycolmap cannot read
        _patch_instantsfm_track_ids()
        _patch_pypose_robustmodel_target()
        _patch_bae_pcg_column_shape()
        _patch_instantsfm_colmap_write()

        # Point ReadData's image dir straight at the keyframe store
        # - ReadData derives every path from data_dir, falling back to data_dir itself as
        #   the image dir when images/ is absent
        # - the keyframe store is already a COLMAP image directory
        # - same PathInfo override the database and output paths get below; it removes the
        #   staged JPEG copy
        data_dir = Path(data_dir)
        image_dir = _sfm_image_dir(data_dir / "images" if images_dir is None else images_dir)
        path_info = ReadData(str(data_dir))
        path_info.image_path = str(image_dir)

        # Depth is the shipped mode; the nodepth path exists only for the eval ablation
        if self.use_depths and not path_info.depth_path:
            raise RuntimeError(f"no depth_vda/ under {data_dir} — generate_vda_depth must run first")

        # Redirect upstream's flat data_dir/{database.db,sparse} into the contract colmap/
        # - PathInfo is a plain mutable class, so the paths are simply reassigned
        # - the DB then survives for re-runs instead of being re-extracted, and no post-hoc
        #   moves are needed
        # - named instantsfm.db: colmap/database.db belongs to the verify stage (loma/xfeat
        #   matches), which unlinks and rewrites it, and reusing that as the SIFT DB would
        #   feed ReadColmapDatabase the wrong features
        # - the whole stale sparse/ tree is removed, not just 0/, so a re-run never mixes
        #   models and a leftover sibling cluster dir (sparse/1) cannot trip the
        #   multi-cluster warning below
        colmap_dir = data_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        path_info.database_path = str(colmap_dir / "instantsfm.db")
        path_info.output_path = str(colmap_dir / "sparse")
        shutil.rmtree(Path(path_info.output_path), ignore_errors=True)
        sparse_dst = Path(path_info.output_path) / "0"

        # SIFT database: reuse a complete one extracted from THIS image set
        # - that is what makes re-runs idempotent
        # - a partial DB from a crashed colmap run, or one keyed on a previous frame
        #   selection, is rebuilt from scratch
        # - build failures raise RuntimeError
        db_path = Path(path_info.database_path)
        if not _sift_database_valid(db_path, [p.name for p in frame_paths(image_dir)]):
            db_path.unlink(missing_ok=True)
            logger.info("InstantSfM: building COLMAP feature database (SIFT, exhaustive)")
            _generate_sift_database(Path(path_info.image_path), db_path)

        view_graph, cameras, images, _feature_name, _rig = ReadColmapDatabase(path_info.database_path)
        if view_graph is None or cameras is None or images is None:
            raise RuntimeError(f"InstantSfM could not read {path_info.database_path}")

        # Config with copied dicts; depth-aware mode per creator field
        config = self._build_config()
        config.RUNTIME_OPTIONS["use_depths"] = self.use_depths
        if self.use_depths:
            logger.info("InstantSfM: loading depths from %s", path_info.depth_path)
            for idx in range(len(images)):
                camera = cameras[images[idx].cam_id]
                images.features[idx] = _nudge_edge_keypoints(images.features[idx], camera.width, camera.height)
            ReadDepthsIntoFeatures(path_info.depth_path, cameras, images)

        # Global mapping; upstream raises a raw IndexError on several failure paths
        # - numpy-2 empty float64 mask in scene/defs.py filter_by_mask once every track
        #   is filtered
        # - empty images.depths when depth priors did not load
        # - log the traceback and re-raise with an honest pointer to the chained cause
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

        # Upstream writes output_path/0 for a single cluster, output_path/<id> per cluster
        # otherwise, and returns WITHOUT creating output_path when no image is registered
        WriteGlomapReconstruction(str(path_info.output_path), cameras, images, tracks, str(path_info.image_path))
        output_path = Path(path_info.output_path)
        if not output_path.exists():
            raise RuntimeError("InstantSfM wrote no reconstruction (no registered images)")
        clusters = sorted(p.name for p in output_path.iterdir() if p.is_dir())
        if not sparse_dst.is_dir():
            raise RuntimeError(
                f"InstantSfM wrote no sparse/0 model (clusters: {clusters}) — the scene split "
                "into disconnected components; use more frames or higher overlap."
            )
        if len(clusters) > 1:
            logger.warning("InstantSfM split the scene into clusters %s — keeping cluster 0 only", clusters)

        # Read back the written model, then rename its images to the pipeline's stem contract
        recon = pycolmap.Reconstruction(str(sparse_dst))
        _rename_images_to_stems(recon, sparse_dst)
        logger.info("InstantSfM: %d registered images, %d points3D", recon.num_reg_images(), recon.num_points3D())
        return recon
