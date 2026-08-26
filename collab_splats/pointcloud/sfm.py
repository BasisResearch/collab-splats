# collab_splats/pointcloud/sfm.py
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from vggt.utils.geometry import unproject_depth_map_to_point_map

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

        camera_mode = pycolmap.CameraMode.SINGLE if self.single_camera else pycolmap.CameraMode.AUTO
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
        from hloc import (
            extract_features,
            match_features,
            pairs_from_retrieval,
            reconstruction,
        )

        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        hloc_dir = output_dir / "colmap" / "hloc"
        hloc_dir.mkdir(parents=True, exist_ok=True)

        retrieval_path = extract_features.main(extract_features.confs[self.retrieval_conf], image_dir, hloc_dir)
        pairs_path = hloc_dir / "pairs.txt"
        pairs_from_retrieval.main(retrieval_path, pairs_path)

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

# Upstream encoder table — DepthAnything/Video-Depth-Anything @ 4f5ae23, run.py:45-49
# `model_configs`; vitl only — VDA_CHECKPOINT is the vitl metric weight, so `encoder` must be a
# key here. Metric vs relative is a constructor flag (`metric=True`), not a separate subdir.
_VDA_MODEL_CONFIGS = {
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def vda_depth_complete(out_dir: Path, names: list[str]) -> bool:
    """
    True when out_dir/depth_vda/images/npy holds exactly one .npy per stem in `names`.
    """
    npy_dir = Path(out_dir) / "depth_vda" / "images" / "npy"
    return npy_dir.is_dir() and {p.stem for p in npy_dir.glob("*.npy")} == {Path(n).stem for n in names}


def _load_vda_model(*, encoder: str, device: str):
    """
    Construct the VDA metric model on `device` from the pinned checkpoint.

    - Split out from generate_vda_depth so the write path is testable without a GPU or
      the third_party clone.
    """
    # Lazy heavy import — VDA lives in a third_party clone (repo root on sys.path), not
    # site-packages. Upstream HEAD (4f5ae23) has no metric_depth/ subdir: `video_depth_anything/`
    # sits at the clone root and `video_depth.py:27` imports a TOP-LEVEL `utils` namespace
    # package (`utils/util.py`) from the same root. Probed 2026-08-23: no foreign top-level
    # `utils` in the venv, and importing collab_splats.wrapper.reconstructor leaves none in
    # sys.modules — a regular `utils` package anywhere on sys.path would shadow VDA's namespace
    # one regardless of insert order, so re-probe if a dependency ever ships one.
    if not (VDA_ROOT / "video_depth_anything").is_dir():
        raise ImportError(
            f"Video-Depth-Anything clone not found at {VDA_ROOT} — run setup.sh "
            "(clones the repo at 4f5ae23 and downloads the metric vitl checkpoint)"
        )
    if str(VDA_ROOT) not in sys.path:
        sys.path.insert(0, str(VDA_ROOT))
    from video_depth_anything.video_depth import VideoDepthAnything

    ckpt = VDA_ROOT / "checkpoints" / VDA_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"VDA metric checkpoint missing: {ckpt} — run setup.sh")

    # metric=True loads the metric head AND disables infer_video_depth's cross-window
    # scale-and-shift chaining (video_depth.py:135), so consecutive windows are stitched
    # on the head's own absolute output rather than fitted to each other. Measured
    # 2026-08-26: this is why a full-video pass does not improve metric contiguity.
    model = VideoDepthAnything(**_VDA_MODEL_CONFIGS[encoder], metric=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.to(device).eval()


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
    keep_rows: Sequence[int] | None = None,
) -> Path:
    """
    Run Video Depth Anything metric depth over keyframes; write InstantSfM's depth layout.

    - frames: (N, H, W, 3) uint8 RGB (frames.zarr order).
    - keep_rows: when set, `frames` is a CONTEXT stream (a contiguous constant-rate grid)
      and only these rows are written, one per entry of `names`, in order. VDA is temporal,
      so inference sees the whole stream and only the write is filtered.
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
    https://github.com/DepthAnything/Video-Depth-Anything @ 4f5ae23 run.py:45-57
    (construct with `metric=`, `load_state_dict(strict=True)`, `infer_video_depth`).
    """
    if keep_rows is None:
        if len(names) != len(frames):
            raise ValueError(f"names ({len(names)}) and frames ({len(frames)}) must align one-to-one")
    else:
        keep_rows = [int(r) for r in keep_rows]
        if len(names) != len(keep_rows):
            raise ValueError(f"names ({len(names)}) and keep_rows ({len(keep_rows)}) must align one-to-one")
        out_of_range = [r for r in keep_rows if not 0 <= r < len(frames)]
        if out_of_range:
            raise ValueError(f"keep_rows out of range for {len(frames)} context frames (first: {out_of_range[0]})")

    depth_dir = Path(out_dir) / "depth_vda"
    npy_dir = depth_dir / "images" / "npy"

    # Idempotent: the exact per-frame stem set is authoritative (overwrite = delete upstream);
    # callers may check vda_depth_complete first to avoid decoding frames at all
    if vda_depth_complete(out_dir, names):
        logger.info("VDA depth exists at %s (%d maps) — skipping inference", npy_dir, len(names))
        return depth_dir

    model = _load_vda_model(encoder=encoder, device=device)

    # Metric inference over the whole sequence (returns input-res depth). With keep_rows
    # the stream is the context grid and `fps` is the CONTEXT rate, not the keyframe rate.
    logger.info(
        "VDA metric inference: %d frames @ %.2f fps (encoder=%s, writing %d maps)",
        len(frames),
        fps,
        encoder,
        len(names),
    )
    depths, _fps = model.infer_video_depth(frames, fps, input_size=input_size, device=device, fp32=False)
    depths = np.asarray(depths, dtype=np.float32)

    # Free the GPU before the caller's InstantSfM CUDA step — resize/write below is CPU-only
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Keep only the rows the caller asked for (all of them when keep_rows is None)
    if keep_rows is not None:
        depths = depths[np.asarray(keep_rows, dtype=np.int64)]

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
# Depth alignment: fit per-frame scales taking VDA metric depth to the
# COLMAP world
########################################################################

MIN_ALIGN_OBS = 20  # per-frame track-observation floor for a trustworthy median
MIN_AFFINE_OBS = 50  # per-frame floor for the 2-parameter disparity fit (scale needs only 20)
AFFINE_REJECT_ROUNDS = 2  # MAD-3sigma rejection passes over the least-squares fit
AFFINE_EPS = 1e-9  # positivity floor on the fitted disparity at the frame's far end


def _depth_correspondences(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Per-frame (d_colmap, d_vda) pairs from track observations, both positive and in bounds.

    - Each points2D carrying a point3D gives an exact pixel plus that point's z in the camera
      frame; the pixel is rescaled from native camera resolution to the depth grid and
      nearest-sampled into VDA depth.
    - Returns one (d_colmap, d_vda) tuple per row of `depth`, in `image_names` order; a frame
      with no usable observation gets a pair of empty arrays.
    """
    # Row order is the caller's; every name must be registered
    name_to_image = {image.name: image for image in reconstruction.images.values()}
    missing = [name for name in image_names if name not in name_to_image]
    if missing:
        raise ValueError(f"{len(missing)} image names not in reconstruction (first: {missing[0]})")

    _n_frames, grid_h, grid_w = depth.shape
    empty = (np.zeros(0), np.zeros(0))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for row, name in enumerate(image_names):
        image = name_to_image[name]
        camera = reconstruction.cameras[image.camera_id]

        # Track observations: exact 2D pixel + the observed point's depth in this view
        observations = [p for p in image.points2D if p.has_point3D()]
        if not observations:
            pairs.append(empty)
            continue
        xyz = np.stack([reconstruction.points3D[p.point3D_id].xyz for p in observations])
        cam_from_world = image.cam_from_world().matrix()
        d_colmap = (xyz @ cam_from_world[:3, :3].T + cam_from_world[:3, 3])[:, 2]

        # Rescale native pixels to the depth grid (the localization ref_px bug class —
        # native-res keypoints indexed into a model-res grid), then nearest-sample
        xy = np.stack([p.xy for p in observations])
        u = np.rint(xy[:, 0] * (grid_w / camera.width)).astype(np.int64)
        v = np.rint(xy[:, 1] * (grid_h / camera.height)).astype(np.int64)
        in_bounds = (u >= 0) & (u < grid_w) & (v >= 0) & (v < grid_h)
        d_vda = np.zeros(len(observations))
        d_vda[in_bounds] = depth[row, v[in_bounds], u[in_bounds]]

        # Keep pairs with positive depth on both sides
        valid = in_bounds & (d_vda > 0) & (d_colmap > 0)
        pairs.append((d_colmap[valid], d_vda[valid]))

    return pairs


def align_depth_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Per-frame scale factors aligning VDA depth to the reconstruction's world scale.

    - Correspondences are track observations: each points2D with a point3D gives an exact
      pixel plus the point's z in the camera frame (d_colmap); the pixel is rescaled from
      native camera resolution to the depth grid and nearest-sampled into VDA depth (d_vda).
    - s_i = median(d_colmap / d_vda) per frame; frames with fewer than MIN_ALIGN_OBS valid
      pairs inherit the global median of the fitted scales; zero fitted frames raises.
    - Returns (scales, stats): (N,) float64 depth multipliers, and a stats dict with the
      global scale, fallback frames, per-frame obs counts, and the pooled ratio spread
      before/after alignment (the after-spread is the unit-level success check).
    """
    n_frames = depth.shape[0]
    scales = np.full(n_frames, np.nan)
    obs_counts = np.zeros(n_frames, dtype=np.int64)
    pooled_ratios: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []

    # One robust scale per frame from its track observations; below the obs floor, don't fit
    for row, (d_colmap, d_vda) in enumerate(_depth_correspondences(reconstruction, image_names, depth)):
        obs_counts[row] = len(d_colmap)
        if obs_counts[row] == 0:
            continue
        ratios = d_colmap / d_vda
        pooled_ratios.append(ratios)
        pooled_rows.append(np.full(len(ratios), row))
        if obs_counts[row] >= MIN_ALIGN_OBS:
            scales[row] = np.median(ratios)

    fitted = ~np.isnan(scales)
    if not fitted.any():
        raise ValueError(
            f"depth alignment: no frame has >= {MIN_ALIGN_OBS} valid track observations — "
            "the reconstruction is too sparse to align VDA depth to the COLMAP world."
        )

    # Thin frames inherit the scene answer (below the obs floor: don't fit, inherit)
    global_scale = float(np.median(scales[fitted]))
    fallback_frames = [image_names[i] for i in np.flatnonzero(~fitted)]
    if fallback_frames:
        logger.warning(
            "depth alignment: %d frames under %d obs (first: %s) — using global scale",
            len(fallback_frames),
            MIN_ALIGN_OBS,
            fallback_frames[0],
        )
    scales[~fitted] = global_scale

    # Pooled spread: before = one global scale for all frames, after = per-frame scales
    ratios_all = np.concatenate(pooled_ratios)
    rows_all = np.concatenate(pooled_rows).astype(np.int64)
    stats = {
        "global_scale": global_scale,
        "n_fallback": len(fallback_frames),
        "fallback_frames": fallback_frames,
        "obs_counts": obs_counts.tolist(),
        "ratio_p10_p50_p90_before": [float(x) for x in np.percentile(ratios_all / global_scale, [10, 50, 90])],
        "ratio_p10_p50_p90_after": [float(x) for x in np.percentile(ratios_all / scales[rows_all], [10, 50, 90])],
    }
    return scales, stats


def _fit_affine_disparity(d_colmap: np.ndarray, d_vda: np.ndarray) -> tuple[float, float] | None:
    """
    Least-squares 1/d_colmap ~= a*(1/d_vda) + b with two MAD-3sigma rejection rounds.

    - Disparity, not depth: gsplat's depth_l1_loss is L1 on 1/d, so the fit is done in the
      space the loss is paid in. Measured 2026-08-26: the offset term b carries the entire
      -18.9% loss-floor improvement over a scale-only fit.
    - Returns None when the surviving inlier set is too small or degenerate to solve.
    """
    q_vda = 1.0 / d_vda
    q_colmap = 1.0 / d_colmap
    inliers = np.ones(len(q_vda), dtype=bool)

    for _round in range(AFFINE_REJECT_ROUNDS + 1):
        if inliers.sum() < MIN_AFFINE_OBS:
            return None

        # Solve the 2-parameter normal equations over the current inliers
        design = np.stack([q_vda[inliers], np.ones(int(inliers.sum()))], axis=1)
        solution, _residuals, rank, _sv = np.linalg.lstsq(design, q_colmap[inliers], rcond=None)
        if rank < 2:
            return None
        a, b = float(solution[0]), float(solution[1])

        # Reject at 3 MAD (scaled to sigma) and refit; a zero MAD means an exact fit
        residual = q_colmap - (a * q_vda + b)
        mad = float(np.median(np.abs(residual[inliers] - np.median(residual[inliers]))))
        if mad <= 0:
            return a, b
        inliers = np.abs(residual) <= 3.0 * 1.4826 * mad

    return a, b


def _apply_affine_depth(depth_row: np.ndarray, a: float, b: float, far_limit: float) -> np.ndarray:
    """
    Map one VDA depth map through the fitted affine disparity: d_new = d / (a + b*d).

    - Applied in depth form so a zero-depth pixel never divides; zeros stay zero.
    - Pixels where the denominator collapses (b < 0 saturates past a horizon) and pixels
      beyond `far_limit` (no track evidence at that range) are written as 0 = no target.
    """
    denominator = a + b * depth_row
    out = np.zeros_like(depth_row, dtype=np.float32)
    supported = (depth_row > 0) & (denominator > AFFINE_EPS) & (depth_row <= far_limit)
    out[supported] = depth_row[supported] / denominator[supported]
    return out


def align_depth_affine(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Per-frame affine-in-disparity alignment of VDA depth to the reconstruction's world.

    - Returns (coeffs (N,2) [a, b], far_limits (N,) metres, stats). Apply with
      `_apply_affine_depth`; a scale-only frame is returned as (1/s, 0.0), which is the
      same mapping the scale path applies.
    - Falls back to scale-only when a frame has fewer than MIN_AFFINE_OBS observations, the
      fit is unsolvable, a <= 0, or the fitted disparity is non-positive at the frame's far
      end. The far end is p99 of the depth map, not its max: a single sky pixel rejects
      78/300 frames, p99 rejects 14/300 (measured 2026-08-26).
    - far_limits is the furthest observed track depth per frame. Beyond it the fit is
      extrapolating, and held-out observations there are 2.2x worse. The bound is ONE-SIDED
      by design: SIFT tracks do not cover close surfaces, so a near-side bound would delete
      the closest ~4% of every frame — the near-field geometry this exists to sharpen.
    """
    n_frames = depth.shape[0]
    coeffs = np.zeros((n_frames, 2), dtype=np.float64)
    far_limits = np.full(n_frames, np.inf)
    fitted = np.zeros(n_frames, dtype=bool)
    scale_only_rows: list[int] = []
    scales = np.full(n_frames, np.nan)

    pairs = _depth_correspondences(reconstruction, image_names, depth)
    for row, (d_colmap, d_vda) in enumerate(pairs):
        if len(d_colmap) == 0:
            continue

        # Evidence bound first: it applies whether or not the affine fit survives
        far_limits[row] = float(d_vda.max())
        if len(d_colmap) >= MIN_ALIGN_OBS:
            scales[row] = float(np.median(d_colmap / d_vda))

        # Affine needs its own, higher observation floor
        if len(d_colmap) < MIN_AFFINE_OBS:
            continue
        solution = _fit_affine_disparity(d_colmap, d_vda)
        if solution is None:
            continue
        a, b = solution

        # Reject a fit that inverts depth or saturates inside the frame's own range
        far_depth = float(np.percentile(depth[row][depth[row] > 0], 99)) if (depth[row] > 0).any() else 0.0
        if a <= 0 or far_depth <= 0 or (a / far_depth + b) <= AFFINE_EPS:
            continue
        coeffs[row] = (a, b)
        fitted[row] = True

    # Scale-only frames: a = 1/s, b = 0 reproduces the scale path exactly
    global_scale = float(np.median(scales[~np.isnan(scales)])) if (~np.isnan(scales)).any() else None
    for row in np.flatnonzero(~fitted):
        scale = scales[row] if not np.isnan(scales[row]) else global_scale
        if scale is None or scale <= 0:
            raise ValueError(
                "depth alignment: no frame has enough valid track observations to fit even a "
                "scale — the reconstruction is too sparse to align VDA depth to the COLMAP world."
            )
        coeffs[row] = (1.0 / scale, 0.0)
        scale_only_rows.append(int(row))

    stats = {
        "n_fitted": int(fitted.sum()),
        "n_fallback": len(scale_only_rows),
        "fallback_frames": [image_names[i] for i in scale_only_rows],
        "global_scale": global_scale,
        "far_limit_p10_p50_p90": (
            [float(x) for x in np.percentile(far_limits[np.isfinite(far_limits)], [10, 50, 90])]
            if np.isfinite(far_limits).any()
            else []
        ),
    }
    return coeffs, far_limits, stats


def apply_depth_alignment(result: "FeedforwardResult", reconstruction: pycolmap.Reconstruction) -> dict:
    """
    Scale result.depth to the reconstruction's world scale in place; recompute world_points.

    - Fits per-frame scales via align_depth_to_reconstruction over the result's rows, then
      re-derives dense world_points from the scaled depth under the COLMAP poses, so the
      zarr and the COLMAP model share one scale.
    - Returns the provenance attrs to merge into save_zarr's extra_attrs; raises on an
      unalignable scene — never a silent VDA-metric write.
    """
    # SfM image_paths are extension-less stems (Path(im.name) from COLMAP, whose image
    # names ARE stems) — path.name is the COLMAP image name, the splats-branch convention
    names = [path.name for path in result.image_paths]
    scales, stats = align_depth_to_reconstruction(reconstruction, names, result.depth)
    logger.info(
        "depth alignment: global scale %.4f, ratio p10/p50/p90 %s -> %s, %d fallback frames",
        stats["global_scale"],
        [round(x, 4) for x in stats["ratio_p10_p50_p90_before"]],
        [round(x, 4) for x in stats["ratio_p10_p50_p90_after"]],
        stats["n_fallback"],
    )

    # Scale depth per frame; re-unproject dense world points (t is not scale-invariant,
    # so world_points cannot be scaled directly — they must be re-derived)
    result.depth = (result.depth * scales[:, None, None]).astype(np.float32)
    result.world_points = unproject_depth_map_to_point_map(
        result.depth[..., None], result.extrinsics[:, :3, :], result.intrinsics
    ).astype(np.float32)

    return {
        "depth_scale": "colmap",
        "depth_scales": [float(s) for s in scales],
        "depth_scale_fallback_frames": stats["fallback_frames"],
    }


########################################################################
# InstantSfM
########################################################################


def _tracked_point3d_ids(recon: pycolmap.Reconstruction) -> list[int]:
    """
    Sorted point3D ids that carry at least one observation.

    - InstantSfM exports sub-min-track-length points with EMPTY tracks (the writer
      consistency patch drops their unverifiable observations); no observation means
      no pixel provenance, so the result tail excludes them.
    """
    return sorted(pid for pid, p in recon.points3D.items() if len(p.track.elements) > 0)


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


########################################
# SIFT feature database (system colmap)
########################################

# CPU SIFT threads: colmap's default (-1) spawns one thread per HOST core — 96 on
# this machine — and per-thread RAM blows past the 46.6 GB container cgroup cap
# (measured: OOM-kill at default, clean 1.5 min run at 8 threads on 100 frames
# of 1920x1080).
_SIFT_NUM_THREADS = 8


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


def _sift_database_valid(database_path: Path) -> bool:
    """
    True when the SIFT DB holds both extraction and matching output.

    - A crashed colmap subprocess (e.g. OOM-killed under the cgroup cap) leaves a
      partial DB behind; an existence-only check would cache-hit on it and feed
      ReadColmapDatabase zero tracks.
    """
    if not database_path.exists():
        return False
    try:
        with sqlite3.connect(database_path) as conn:
            keypoints = conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
            geometries = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    except sqlite3.Error:
        return False
    return keypoints > 0 and geometries > 0


def _generate_sift_database(image_path: Path, database_path: Path, single_camera: bool) -> None:
    """
    Build the COLMAP SIFT feature database: extraction + exhaustive matching.

    - Reimplements upstream GenerateDatabase (cre185/InstantSfM
      instantsfm/controllers/feature_handler.py:18-57 @ d3e599e): upstream
      hardcodes CPU SIFT with no thread cap (OOM-kill under the cgroup, see
      _SIFT_NUM_THREADS) and swallows CalledProcessError, so a colmap crash
      there surfaces only as an empty-tracks IndexError much later.
    - GPU SIFT when CUDA is available (CUDA-built colmap runs SiftGPU headless;
      measured 100x1920x1080: extraction 7 s vs 90 s CPU, exhaustive matching
      55 s vs ~816 s CPU); CPU fallback keeps upstream's CUDA_VISIBLE_DEVICES=""
      plus the thread cap.
    - On failure the partial DB is unlinked so a re-run rebuilds from scratch.
    """
    env = os.environ.copy()
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

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
        "1" if single_camera else "0",
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
        extractor_cmd += ["--SiftExtraction.num_threads", str(_SIFT_NUM_THREADS)]
        matcher_cmd += ["--SiftMatching.num_threads", str(_SIFT_NUM_THREADS)]

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


@dataclass
class InstantSfMCreator:
    """
    Global SfM via InstantSfM (https://github.com/cre185/InstantSfM, IROS 2026).

    - License: CC-BY-NC-4.0 (non-commercial) — cleared for this repo's research use;
      revisit before any commercial deployment. Install pinned in setup.sh.
    - Drives the upstream Python API directly (never their CLI): ReadData ->
      SIFT DB build (_generate_sift_database, reimplemented from upstream
      GenerateDatabase; system colmap binary, SIFT + exhaustive, GPU when
      CUDA is available) -> ReadColmapDatabase -> Config -> ReadDepthsIntoFeatures
      (VDA metric depth) -> SolveGlobalMapper -> WriteGlomapReconstruction.
      Call pattern follows instantsfm/scripts/sfm.py::run_sfm at the installed
      version (0.3.0).
    - Not a BasePointcloudCreator: its contract is reconstruct(data_dir) ->
      pycolmap.Reconstruction over a staged scene dir, not (image_dir, output_dir) ->
      PointcloudResult; the Reconstructor wraps the result.
    """

    features: str = "colmap"
    single_camera: bool = True
    use_depths: bool = True
    retriangulation: bool = False

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

        # Optional GLOMAP-style refinement: retriangulate from the full pre-filter track
        # set, then up to ba_global_max_refinements (5) further BA rounds. Upstream
        # defaults skip_retriangulation True; this is their only post-BA refinement knob.
        config.OPTIONS["skip_retriangulation"] = not self.retriangulation
        return config

    def reconstruct(self, data_dir: Path) -> pycolmap.Reconstruction:
        """
        Run InstantSfM over data_dir (must hold images/; depth_vda/ from generate_vda_depth when use_depths).

        - Works in the contract layout directly: SIFT DB at data_dir/colmap/instantsfm.db
          (reused on re-runs; GCS push excludes it; its own name so geometry/verification.py's
          colmap/database.db can never be mistaken for it), COLMAP binary at
          data_dir/colmap/sparse/0.
        - Returns the pycolmap.Reconstruction read back from the written model.
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

        # Upstream compat fixes: packed 64-bit track ids vs int32 storage (numpy 2),
        # bae LM.step vs pypose-0.7.5 RobustModel.forward(target), PCG 1-D step vs
        # TrustRegion.update, COLMAP writer emitting a model pycolmap can't read
        _patch_instantsfm_track_ids()
        _patch_pypose_robustmodel_target()
        _patch_bae_pcg_column_shape()
        _patch_instantsfm_colmap_write()

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
        # instead of being re-extracted, and no post-hoc moves are needed. The DB is named
        # instantsfm.db: colmap/database.db belongs to the verify stage (loma/xfeat matches),
        # which unlinks and rewrites it, and reusing that as the SIFT DB would feed
        # ReadColmapDatabase the wrong features. The whole stale sparse/ tree is removed (not
        # just 0/) so a re-run never mixes models and leftover sibling cluster dirs (sparse/1)
        # cannot trip the multi-cluster warning below.
        colmap_dir = data_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        path_info.database_path = str(colmap_dir / "instantsfm.db")
        path_info.output_path = str(colmap_dir / "sparse")
        shutil.rmtree(Path(path_info.output_path), ignore_errors=True)
        sparse_dst = Path(path_info.output_path) / "0"

        # SIFT database: reuse a complete one (idempotent re-runs); a partial DB left by
        # a crashed colmap run is rebuilt from scratch (build failures raise RuntimeError)
        db_path = Path(path_info.database_path)
        if not _sift_database_valid(db_path):
            db_path.unlink(missing_ok=True)
            logger.info("InstantSfM: building COLMAP feature database (SIFT, exhaustive)")
            _generate_sift_database(Path(path_info.image_path), db_path, self.single_camera)

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
        logger.info("InstantSfM: %d registered images, %d points3D", recon.num_reg_images(), recon.num_points3D())
        return recon
