# collab_splats/pointcloud/sfm.py
from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch

from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult

logger = logging.getLogger(__name__)


@dataclass
class ColmapCreator(BasePointcloudCreator):
    """Pointcloud via pycolmap SIFT feature extraction + exhaustive matching.

    Runs a three-stage classical SfM pipeline:

    1. **Feature extraction** — SIFT keypoints and descriptors are detected in
       every image.  ``camera_model`` selects the distortion model; ``single_camera``
       controls whether all images share one camera or each gets its own.
    2. **Exhaustive matching** — every image pair is compared (O(N²)).  Suitable
       for small-to-medium datasets (< ~500 images).
    3. **Incremental mapping** — COLMAP initialises from a two-view seed, then
       registers remaining images one-by-one with PnP+RANSAC and periodic
       bundle adjustment.

    Args:
        camera_model: COLMAP camera model string.  Common choices:

            * ``"SIMPLE_PINHOLE"`` — fx, cx, cy (no distortion, 3 params)
            * ``"SIMPLE_RADIAL"`` — fx, cx, cy, k1 (default, 4 params)
            * ``"OPENCV"`` — fx, fy, cx, cy, k1, k2, p1, p2 (8 params)

        single_camera: If ``True``, all images share one camera model
            (``CameraMode.SINGLE``).  Use for video frames from a single
            physical device.  If ``False``, each image gets an independent
            camera (``CameraMode.AUTO``).
    """

    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = False

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        db_path = output_dir / "colmap" / "database.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        camera_mode = (
            pycolmap.CameraMode.SINGLE if self.single_camera else pycolmap.CameraMode.AUTO
        )
        # pycolmap >=4.0: camera_model lives in ImageReaderOptions, not as a
        # top-level kwarg of extract_features.
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
            output_path=str(sparse_dir.parent),  # colmap/sparse/ → creates 0/ inside
        )
        if not reconstructions:
            raise RuntimeError("reconstruction failed — pycolmap incremental_mapping returned no results")

        recon = reconstructions[0]
        recon.write_binary(str(sparse_dir))
        image_paths = sorted(
            [image_dir / img.name for img in recon.images.values()],
            key=lambda p: p.name,
        )
        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=image_paths,
        )


@dataclass
class HlocCreator(BasePointcloudCreator):
    """Pointcloud via hloc (SuperPoint+SuperGlue feature matching).

    hloc is imported lazily inside :meth:`reconstruct` because it has heavy
    transitive dependencies (torch, kornia, etc.) that are not required by
    other creators, and may not be installed in all environments.  The lazy
    import also avoids GPU initialisation at module load time.

    Runs a four-stage learned SfM pipeline:

    1. **Image retrieval** (NetVLAD) — finds candidate matching pairs without
       exhaustive comparison.  O(N) retrieval vs O(N²) exhaustive.
    2. **Feature extraction** (SuperPoint) — learned keypoint detector and
       descriptor, more robust than SIFT under challenging lighting or
       texture-poor conditions.
    3. **Feature matching** (SuperGlue) — graph-neural-network matcher that
       uses attention to establish correspondences across wide baselines.
    4. **Reconstruction** — COLMAP incremental mapper driven by the hloc
       matches instead of SIFT.

    Args:
        retrieval_conf: hloc retrieval config key — controls how candidate
            image pairs are selected before matching.  Valid values:

            * ``"netvlad"`` *(default)* — global descriptor trained for
              place recognition; robust across lighting and viewpoint changes.
            * ``"openibl"`` — OpenIBL global descriptor, similar to NetVLAD.
            * ``"cosplace"`` — CoSPlace retrieval, strong on large-scale scenes.
            * ``"eigenplaces"`` — EigenPlaces descriptor, good for urban scenes.

        feature_conf: hloc feature extraction config key — selects the local
            feature detector and descriptor used for matching.  Valid values:

            * ``"superpoint_aachen"`` *(default)* — SuperPoint weights tuned
              on the Aachen Day-Night benchmark; best all-round choice.
            * ``"superpoint_max"`` — SuperPoint with higher max keypoints
              (8192 vs 1024); better for large textureless scenes.
            * ``"superpoint_inloc"`` — SuperPoint weights tuned for indoor
              localisation (InLoc benchmark).
            * ``"d2net-ss"`` — D2-Net single-scale; slower but handles
              texture-poor and day/night changes well.
            * ``"sift"`` — classical SIFT; no GPU required, good baseline.
            * ``"sosnet"`` — SIFT keypoints with SOS-Net descriptors.
            * ``"disk"`` — DISK detector+descriptor; strong on wide baselines.

        matcher_conf: hloc matcher config key — selects how descriptors are
            matched across image pairs.  Valid values:

            * ``"superglue"`` *(default)* — SuperGlue graph-neural-network
              matcher; handles wide baselines and occlusion well.  Requires GPU.
            * ``"superglue-fast"`` — SuperGlue with reduced iterations; faster
              inference at slight accuracy cost.
            * ``"NN-superpoint"`` — nearest-neighbour matching tuned for
              SuperPoint descriptors; no learned parameters, CPU-friendly.
            * ``"NN-ratio"`` — nearest-neighbour with Lowe's ratio test;
              works with any descriptor, fastest option.
            * ``"NN-mutual"`` — mutual nearest-neighbour (cross-check);
              more precise than ratio test, slightly slower.
            * ``"adalam"`` — AdaLAM local affine matcher; good for planar
              or repeated-structure scenes.
            * ``"disk+lightglue"`` — LightGlue matcher optimised for DISK
              features; use with ``feature_conf="disk"``.
            * ``"superpoint+lightglue"`` — LightGlue matcher optimised for
              SuperPoint; faster than SuperGlue with comparable accuracy.
    """

    retrieval_conf: str = "netvlad"
    feature_conf: str = "superpoint_aachen"
    matcher_conf: str = "superglue"

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction

        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        hloc_dir = output_dir / "colmap" / "hloc"
        hloc_dir.mkdir(parents=True, exist_ok=True)

        retrieval_path = extract_features.main(
            extract_features.confs[self.retrieval_conf], image_dir, hloc_dir
        )
        pairs_path = hloc_dir / "pairs.txt"
        pairs_from_retrieval.main(retrieval_path, pairs_path)

        feature_path = extract_features.main(
            extract_features.confs[self.feature_conf], image_dir, hloc_dir
        )
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

        image_paths = sorted(
            [image_dir / img.name for img in recon.images.values()],
            key=lambda p: p.name,
        )
        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=image_paths,
        )


########################################################################
# Video Depth Anything — metric depth for the SfM path
########################################################################

# Repo root -> third_party clone (setup.sh owns creation); module-level so tests can monkeypatch
VDA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Video-Depth-Anything"
VDA_CHECKPOINT = "metric_video_depth_anything_vitl.pth"

# Upstream encoder table (Video-Depth-Anything metric_depth/run.py `model_configs`); vitl only —
# VDA_CHECKPOINT is the vitl metric weight, so `encoder` must be a key here
_VDA_MODEL_CONFIGS = {
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def generate_vda_depth(
    frames: np.ndarray,
    fps: float,
    out_dir: Path,
    names: list[str],
    *,
    encoder: str = "vitl",
    input_size: int = 518,
    depth_width: int = 518,
    device: str = "cuda",
) -> Path:
    """
    Run Video Depth Anything metric depth over keyframes; write InstantSfM's depth layout.

    - frames: (N, H, W, 3) uint8 RGB (frames.zarr order).
    - fps: effective keyframe rate — VDA is temporal.
    - names: staged image filenames (e.g. frame_000000.jpg), one per frame, same order.
    - out_dir: parent dir; one float32 map per frame lands at
      out_dir/depth_vda/images/npy/<stem>.npy — the layout instantsfm's
      ReadDepthsIntoFeatures single-camera branch consumes (data_reader.py:404-407 ->
      ReadDepthsWithFilenames(depth_vda/images) -> npy/<stem>.npy matched by image stem).
    - depth_width: VDA returns depth at the input frame resolution (300 x 1080p = 2.5 GB),
      too heavy for pointcloud.zarr; each map is nearest-resized to this width (no depth
      blending across discontinuities). 518 matches the feedforward model-res convention
      so downstream stages see the same resolution class. Any depth res is valid for SfM —
      instantsfm's sample_depth_at_pixel normalises keypoints by camera w/h.
    - Returns out_dir/depth_vda. Skips inference when npy/ already holds exactly the
      stems in `names`.

    Attribution: inference pattern follows
    https://github.com/DepthAnything/Video-Depth-Anything metric_depth/run.py.
    """
    if len(names) != len(frames):
        raise ValueError(f"names ({len(names)}) and frames ({len(frames)}) must align one-to-one")
    depth_dir = Path(out_dir) / "depth_vda"
    npy_dir = depth_dir / "images" / "npy"

    # Idempotent: the exact per-frame stem set is authoritative (overwrite = delete upstream);
    # compared as sets so a wrong-named or leftover file never satisfies the gate
    if npy_dir.is_dir() and {p.stem for p in npy_dir.glob("*.npy")} == {Path(n).stem for n in names}:
        logger.info("VDA depth exists at %s (%d maps) — skipping inference", npy_dir, len(names))
        return depth_dir

    # Lazy heavy import — VDA lives in a third_party clone, not site-packages
    metric_dir = VDA_ROOT / "metric_depth"
    if not metric_dir.exists():
        raise ImportError(
            f"Video-Depth-Anything clone not found at {VDA_ROOT} — run setup.sh "
            "(clones the repo and downloads the metric vitl checkpoint)"
        )
    if str(metric_dir) not in sys.path:
        sys.path.insert(0, str(metric_dir))
    from video_depth_anything.video_depth import VideoDepthAnything

    ckpt = VDA_ROOT / "checkpoints" / VDA_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"VDA metric checkpoint missing: {ckpt} — run setup.sh")

    # Load the metric model on the target device
    model = VideoDepthAnything(**_VDA_MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    # Metric inference over the whole keyframe sequence (returns input-res depth)
    logger.info("VDA metric inference: %d frames @ %.2f fps (encoder=%s)", len(frames), fps, encoder)
    depths, _fps = model.infer_video_depth(frames, fps, input_size=input_size, device=device, fp32=False)
    depths = np.asarray(depths, dtype=np.float32)

    # Nearest-resize to depth_width and write one map per frame, keyed by image stem
    h, w = depths.shape[1:3]
    depth_hw = (int(round(depth_width * h / w)), depth_width)
    npy_dir.mkdir(parents=True, exist_ok=True)
    for name, depth in zip(names, depths):
        small = cv2.resize(depth, (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST)
        np.save(npy_dir / f"{Path(name).stem}.npy", small.astype(np.float32))
    logger.info("VDA depths written: %s (%d maps @ %dx%d)", npy_dir, len(names), depth_hw[1], depth_hw[0])
    return depth_dir


########################################################################
# InstantSfM
########################################################################


def _pixel_indices_from_reconstruction(
    recon: pycolmap.Reconstruction,
    point3d_ids: list[int],
    name_to_row: dict[str, int],
    scale_x: float,
    scale_y: float,
    depth_hw: tuple[int, int],
) -> np.ndarray:
    """
    Synthesize (P, 3) int32 [frame_row, row, col] pixel indices from COLMAP tracks.

    - First track observation per point3D; keypoint xy is original-res, scaled to
      the depth grid and clamped in-bounds.
    - lift_features requires pixel_indices; SfM results have no dense source pixel,
      so the observing keypoint is the honest substitute.
    """
    h, w = depth_hw
    out = np.zeros((len(point3d_ids), 3), dtype=np.int32)

    # One observation per point: the first track element's keypoint, scaled + clamped
    for i, pid in enumerate(point3d_ids):
        elem = recon.points3D[pid].track.elements[0]
        image = recon.images[elem.image_id]
        xy = image.points2D[elem.point2D_idx].xy
        col = min(max(int(xy[0] * scale_x), 0), w - 1)
        row = min(max(int(xy[1] * scale_y), 0), h - 1)
        out[i] = (name_to_row[image.name], row, col)

    return out


@dataclass
class InstantSfMCreator:
    """
    Global SfM via InstantSfM (https://github.com/cre185/InstantSfM, IROS 2026).

    - License: CC-BY-NC-4.0 (non-commercial) — cleared for this repo's research use;
      revisit before any commercial deployment. Install pinned in setup.sh.
    - Drives the upstream Python API directly (never their CLI): ReadData ->
      GenerateDatabase (system colmap binary, CPU SIFT, exhaustive) ->
      ReadColmapDatabase -> Config -> ReadDepthsIntoFeatures (VDA metric depth) ->
      SolveGlobalMapper -> WriteGlomapReconstruction. Call pattern follows
      instantsfm/scripts/sfm.py::run_sfm at the installed version (0.3.0).
    - Not a BasePointcloudCreator: its contract is reconstruct(data_dir) ->
      pycolmap.Reconstruction over a staged scene dir, not (image_dir, output_dir) ->
      PointcloudResult; the Reconstructor wraps the result.
    """

    features: str = "colmap"
    single_camera: bool = True
    use_depths: bool = True

    def _build_config(self):
        """
        Upstream Config with OPTIONS/RUNTIME_OPTIONS copied — Config.__init__
        aliases module-level dicts, so in-place mutation leaks across instances.
        """
        # Lazy heavy import — instantsfm is an optional dep (CUDA extensions)
        from instantsfm.controllers.config import Config

        config = Config(self.features)
        config.OPTIONS = dict(config.OPTIONS)
        config.RUNTIME_OPTIONS = dict(config.RUNTIME_OPTIONS)
        return config

    def reconstruct(self, data_dir: Path) -> pycolmap.Reconstruction:
        """
        Run InstantSfM over data_dir (must hold images/; depth_vda/ from generate_vda_depth when use_depths).

        - Works in the contract layout directly: SIFT DB at data_dir/colmap/database.db
          (reused on re-runs; GCS push excludes it), COLMAP binary at data_dir/colmap/sparse/0.
        - Returns the pycolmap.Reconstruction read back from the written model.
        """
        # Lazy heavy import — instantsfm is an optional dep (CUDA extensions)
        from instantsfm.controllers.data_reader import (
            ReadColmapDatabase,
            ReadData,
            ReadDepthsIntoFeatures,
        )
        from instantsfm.controllers.feature_handler import GenerateDatabase
        from instantsfm.controllers.global_mapper import SolveGlobalMapper
        from instantsfm.controllers.reconstruction_writer import (
            WriteGlomapReconstruction,
        )

        # ReadData falls back to data_dir itself as the image dir when images/ is
        # absent — refuse that silently-wrong layout up front
        data_dir = Path(data_dir)
        if not (data_dir / "images").is_dir():
            raise FileNotFoundError(f"InstantSfM expects {data_dir / 'images'} — stage keyframes first")
        path_info = ReadData(str(data_dir))

        # Depth is the shipped mode; the nodepth path exists only for the eval ablation
        if self.use_depths and not path_info.depth_path:
            raise RuntimeError(f"no depth_vda/ under {data_dir} — generate_vda_depth must run first")

        # Redirect upstream's flat data_dir/{database.db,sparse} into the contract layout
        # colmap/ (PathInfo is a plain mutable class) — the DB then survives for re-runs
        # instead of being re-extracted, and no post-hoc moves are needed. The whole stale
        # sparse/ tree is removed (not just 0/) so a re-run never mixes models and leftover
        # sibling cluster dirs (sparse/1) cannot trip the multi-cluster warning below.
        colmap_dir = data_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        path_info.database_path = str(colmap_dir / "database.db")
        path_info.database_exists = Path(path_info.database_path).exists()
        path_info.output_path = str(colmap_dir / "sparse")
        shutil.rmtree(Path(path_info.output_path), ignore_errors=True)
        sparse_dst = Path(path_info.output_path) / "0"

        # SIFT database: reuse an existing one (idempotent re-runs), else build via the
        # system colmap binary (upstream subprocesses it; CPU SIFT, exhaustive). Upstream
        # swallows CalledProcessError, so the existence check below is the real gate.
        if not path_info.database_exists:
            logger.info("InstantSfM: building COLMAP feature database (CPU SIFT, exhaustive)")
            GenerateDatabase(
                str(path_info.image_path),
                str(path_info.database_path),
                self.features,
                None,
                single_camera=self.single_camera,
            )
        if not Path(path_info.database_path).exists():
            raise RuntimeError(
                "COLMAP database missing after GenerateDatabase — is the `colmap` binary installed?"
            )

        view_graph, cameras, images, _feature_name, _rig = ReadColmapDatabase(path_info.database_path)
        if view_graph is None or cameras is None or images is None:
            raise RuntimeError(f"InstantSfM could not read {path_info.database_path}")

        # Config with copied dicts; depth-aware mode per creator field
        config = self._build_config()
        config.RUNTIME_OPTIONS["use_depths"] = self.use_depths
        if self.use_depths:
            logger.info("InstantSfM: loading depths from %s", path_info.depth_path)
            ReadDepthsIntoFeatures(path_info.depth_path, cameras, images)

        # Global mapping. Upstream raises a raw IndexError on several failure paths
        # (numpy-2 empty float64 mask in scene/defs.py filter_by_mask once every track is
        # filtered; empty images.depths when depth priors did not load) — log the
        # traceback and re-raise with an honest pointer to the chained cause.
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

        # Read back the written model as the return value
        recon = pycolmap.Reconstruction(str(sparse_dst))
        logger.info(
            "InstantSfM: %d registered images, %d points3D", recon.num_reg_images(), recon.num_points3D()
        )
        return recon
