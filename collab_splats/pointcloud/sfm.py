# collab_splats/pointcloud/sfm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pycolmap

from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult


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
