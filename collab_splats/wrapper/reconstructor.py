"""5-stage reconstruction pipeline wrapper."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import warnings
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pycolmap
import yaml
import zarr
from mergedeep import merge

from collab_splats.geometry.bundle_adjustment import (
    BundleAdjustment,
    BundleAdjustmentConfig,
    check_model_resolution,
)
from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig
from collab_splats.geometry.metrics import compute_reconstruction_quality
from collab_splats.geometry.transforms import invert_poses
from collab_splats.localization.extractors import LocalMatcher
from collab_splats.localization.localizer import CameraLocalizer
from collab_splats.mesh import (
    clean_repair_mesh,
    create_texture_mesh,
    create_tsdf_mesh,
)
from collab_splats.pointcloud import BaseFeedforwardCreator, get_creator
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.sfm import SFM_CREATORS
from collab_splats.pointcloud.sfm.sift_db import PAIRINGS as _SFM_PAIRINGS
from collab_splats.pointcloud.utils import clean_pointcloud, confidence_mask
from collab_splats.preproc import frames, get_video_info
from collab_splats.preproc import viz as preproc_viz
from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.sampling import (
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
from collab_splats.preproc.undistort import calibrate_camera, undistort_frames
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.lifting import lift_features
from collab_splats.semantics.segmentation import sky_masks
from collab_splats.semantics.utils import (
    extract_feature_cache,
    lifted_store_path,
    load_feature_maps,
    write_point_features,
)
from collab_splats.utils.colmap import write_colmap_reconstruction
from collab_splats.utils.image import upsample_depths
from collab_splats.utils.io import LZ4, write_json
from collab_splats.utils.torch_utils import get_device, pytorch_gc, to_numpy

if TYPE_CHECKING:
    from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# base.yaml is the single source of defaults; __init__ merges any passed config over it.
DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"

_FEEDFORWARD_BACKENDS = set(BaseFeedforwardCreator._registry)
# Every sfm creator _run_sfm can dispatch; config load rejects anything else
_SFM_BACKENDS = set(SFM_CREATORS)

# Every sfm sub-block key: the creator's init fields
# - clean comes from pointcloud.clean.enabled, never the sub-block
# - instantsfm use_depths stays at its creator default: not a config knob
_SFM_BLOCK_KEYS = {
    backend: {f.name for f in dataclasses.fields(creator) if f.init} - {"clean"}
    for backend, creator in SFM_CREATORS.items()
}
_SFM_BLOCK_KEYS["instantsfm"] -= {"use_depths"}

# PointcloudResult's dense per-frame fields, dropped from the in-session result once saved
_DENSE_FIELDS = ("images", "confidence", "world_points", "depth")

_VALID_METHODS = {"feedforward", "sfm"}

# The removed verify stage and pointcloud.geometric_verification flag both point here
_VERIFY_REMOVED = "geometric verification was removed (2026-09-27): drop the verify stage and geometric_verification"

_STAGE_ORDER = [
    "preproc",
    "pointcloud",
    "refine",
    "semantics",
    "splats",
    "mesh",
    "localize",
    "reconstruction_quality_report",
]
_STAGE_DEPS: dict[str, list[str]] = {
    "preproc": [],
    "pointcloud": ["preproc"],
    # refine rewrites pointcloud outputs in place, point count included
    # - NOT a dependency of the stages below: that would demote them from LEAF_STAGES
    # - staleness: after --stages refine, re-run dependents with overwrite (configs/README.md)
    # - semantics MUST re-run: lifted rows index the pre-refine points, now row-misaligned
    # - inline runs order refine before dependents, so never stale
    "refine": ["pointcloud"],
    "semantics": ["pointcloud"],
    # splats: trains on COLMAP poses/points + images/; leaf — nothing reads it yet
    "splats": ["pointcloud"],
    "mesh": ["pointcloud"],
    "localize": ["pointcloud"],
    "reconstruction_quality_report": ["pointcloud"],
}
# A stage is re-runnable on its own iff nothing depends on it → {refine, semantics, splats, mesh,
# localize, reconstruction_quality_report}.
# Derived from the graph above rather than hardcoded: a future stage that depends on mesh drops
# mesh from this set automatically, so callers gating on it can never disagree with _STAGE_DEPS.
LEAF_STAGES = frozenset(s for s in _STAGE_ORDER if not any(s in deps for deps in _STAGE_DEPS.values()))


########################################
# Helpers
########################################


def _store_rows(images_dir: Path, names: Sequence[Path | str]) -> list[int]:
    """
    Rows of the images/ store holding each named frame, in the order named.

    - pointcloud.zarr may hold fewer frames than images/: incremental sfm drops unregistered ones
    - joined on the source frame index in each name, never on row position or extension

    Args:
        images_dir: the scene's images/ directory.
        names: frame names (frame_NNNNNN with any extension or none), e.g. a result's image_paths.

    Returns:
        Indices into `frames.frame_paths(images_dir)`, one per name.

    Raises:
        KeyError: when a named frame is not in images/.
    """
    rows_by_frame_idx = {frames.frame_idx_from_path(p): row for row, p in enumerate(frames.frame_paths(images_dir))}
    frame_indices = [frames.frame_idx_from_path(name) for name in names]

    # A frame the store never selected means the two artifacts come from different runs
    unknown = [fi for fi in frame_indices if fi not in rows_by_frame_idx]
    if unknown:
        raise KeyError(
            f"{len(unknown)} reconstruction frames are not in {images_dir} "
            f"(frame_idx {unknown[:5]}); the images/ store and the reconstruction describe "
            "different runs."
        )
    return [rows_by_frame_idx[fi] for fi in frame_indices]


def _camera_provenance(camera: pycolmap.Camera) -> dict:
    """
    A pycolmap.Camera as a json.dumps-able dict that Camera(**d) reads back.

    Args:
        camera: any pycolmap.Camera.

    Returns:
        {"model", "width", "height", "params"} — model as its name and params as floats,
        because Camera.todict() hands back a CameraModelId enum and an ndarray, neither
        of which json.dumps can write.
    """
    return {
        "model": camera.model.name,
        "width": camera.width,
        "height": camera.height,
        "params": [float(p) for p in camera.params],
    }


def _apply_undistortion(frame_arrays: np.ndarray, images_dir: Path, prov: dict) -> np.ndarray:
    """
    Calibrate from the written frames and undistort them onto COLMAP's framing.

    Args:
        frame_arrays: (N, H, W, 3) uint8 RGB, as selected.
        images_dir: where those frames were written — calibration reads them from here.
        prov: provenance dict, stamped with both cameras under "undistort".

    Returns:
        (N, H', W', 3) uint8 RGB on the undistorted framing.
    """
    camera = calibrate_camera(images_dir)
    undistorted, new_camera = undistort_frames(frame_arrays, camera)

    # Both cameras, as COLMAP writes them — no local mirror of the same numbers
    prov["undistort"] = {
        "camera": _camera_provenance(camera),
        "undistorted_camera": _camera_provenance(new_camera),
    }
    return undistorted


def _frames_from_dir(input_path: Path, *, max_frames: int | None) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    Every image in a directory, in filename order.

    Args:
        input_path: directory of images.
        max_frames: recorded in provenance only — a directory is always taken whole.

    Returns:
        (frames, records, provenance) — records carry a NaN blur score, since a directory
        has no quality report to read one from.
    """
    # Read source images from directory; reject non-image extensions
    source_paths = frames.frame_paths(input_path)
    if not source_paths:
        raise ValueError(f"No images ({list(frames.IMAGE_EXTS)}) found in directory {input_path}")

    frame_arrays = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in source_paths]
    records = [{"frame_idx": i, "blur_score": float("nan")} for i in range(len(frame_arrays))]
    prov = {
        "video_path": str(input_path),
        "video_mtime": None,
        "method": "dir",
        "fps": None,
        "max_frames": max_frames,
    }
    return frame_arrays, records, prov


def _frames_from_video(
    input_path: Path,
    *,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
    report: dict,
    quality: dict | None = None,
    on_empty_slot: str = "rescue",
) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    Frames selected from a video by one of the three sampling methods.

    Args:
        input_path: source video.
        frame_selection: "fps" | "uniform" | "optical_flow".
        fps: target rate for frame_selection="fps".
        min_frames: floor for the fps re-spread band.
        max_frames: cap; the contract for frame_selection="uniform".
        report: quality report, already measured by load_video_quality.
        quality: overrides for filter_frame_quality's thresholds; None takes its defaults.
        on_empty_slot: empty-slot policy for frame_selection="fps".

    Returns:
        (frames, records, provenance).
    """
    # 'fps' samples at a constant wall-clock rate (band-bounded), 'uniform' spreads exactly
    # max_frames over the eligible pool (the quality mask), 'optical_flow' picks
    # high-motion frames. Each method gets only its own knobs.
    if frame_selection == "fps":
        frame_arrays, records = sample_fps(
            str(input_path),
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
            report=report,
            quality=quality,
            on_empty_slot=on_empty_slot,
        )
    elif frame_selection == "uniform":
        frame_arrays, records = sample_uniform(
            str(input_path),
            max_frames=max_frames,
            report=report,
            quality=quality,
        )
    elif frame_selection == "optical_flow":
        frame_arrays, records = sample_optical_flow(
            str(input_path), max_frames=max_frames, report=report, quality=quality
        )
    else:
        raise ValueError(f"preproc.frame_selection must be 'fps', 'uniform' or 'optical_flow', got {frame_selection!r}")

    prov = {
        "video_path": str(input_path),
        "video_mtime": input_path.stat().st_mtime,
        "method": frame_selection,
        "fps": fps,
        "max_frames": max_frames,
        "quality": quality,
        "on_empty_slot": on_empty_slot,
    }
    return frame_arrays, records, prov


def extract_frames(
    input_path: Path,
    images_dir: Path,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
    n_workers: int = 1,
    undistort: bool = False,
    quality: dict | None = None,
    on_empty_slot: str = "rescue",
) -> int:
    """
    Extract frames from video or image dir into images/ (sole persistent store).

    Two steps for video input: measure the whole video into
    video_quality_report.json, then select from it. An image directory takes
    every image and needs no report. images/frame_NNNNNN.png plus frames.json
    beside it is the canonical decode-once keyframe store. Returns the number
    of frames stored.

    - quality overrides filter_frame_quality's thresholds (the eligibility gate every
      sampler selects from); None takes its defaults
    - on_empty_slot reaches sample_fps only: "rescue" or "drop" for an all-ineligible slot
    - undistort=True self-calibrates one shared OPENCV camera from the written frames
      and rewrites them undistorted (COLMAP's framing resizes the canvas, so frame dims
      change; both cameras are stamped into provenance["undistort"]).
    """
    input_path = Path(input_path)
    report = None

    if input_path.is_dir():
        frame_arrays, records, prov = _frames_from_dir(input_path, max_frames=max_frames)
    else:
        # Probe first: a bad path or undecodable file raises here, before any measuring
        total_frames = get_video_info(str(input_path))["total_frames"]

        # Measure before selecting. The report lands beside images/ and is reused by
        # existence, so a re-run never re-measures. Report-only: it carries no verdicts —
        # filter_frame_quality turns its columns into a keep mask inside the samplers,
        # with a robust MAD cut on log(laplacian) and an absolute clipping cut.
        report = load_video_quality(
            input_path,
            images_dir.parent / "video_quality_report.json",
            workers=n_workers,
        )

        frame_arrays, records, prov = _frames_from_video(
            input_path,
            frame_selection=frame_selection,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
            report=report,
            quality=quality,
            on_empty_slot=on_empty_slot,
        )

        # Every candidate failed the quality filter (or the selector rejected all) — refuse
        # to write an empty store that would only surface as a downstream FileNotFound.
        if not frame_arrays:
            raise ValueError(
                f"0 frames selected from {input_path} ({total_frames} decoded) with "
                f"frame_selection={frame_selection!r}. Every frame failed the quality filter — "
                f"check {images_dir.parent / 'video_quality_report.json'} for the measurements."
            )

    # Write once so calibration has images to read, then rewrite the undistorted stack
    frames.write_frames(images_dir, frame_arrays, records, prov)
    if undistort:
        frame_arrays = _apply_undistortion(frame_arrays, images_dir, prov)
        frames.write_frames(images_dir, frame_arrays, records, prov)

    # Render the report beside images/ with the kept frames marked. Written whenever frames
    # are, so the PNGs never go stale against the store. Video only — a directory has no
    # report to plot.
    if report is not None:
        out_dir = images_dir.parent
        selected = [r["frame_idx"] for r in records]
        written = [
            preproc_viz.plot_photometric(report, out_dir, selected=selected),
            preproc_viz.plot_motion(report, out_dir, selected=selected),
        ]
        logger.info("video quality: wrote %d plots to %s", sum(p is not None for p in written), out_dir)

    return len(frame_arrays)


def _run_feedforward(
    backend: str,
    images_dir: Path,
    output_dir: Path,
    loop_closure: bool | dict,
    viz_enabled: bool,
    viz_port: int,
    max_points: int,
    min_views: int,
    mv_rel_thresh: float,
    # Keyword-only: model_dir is required, the others optional
    # - a positional caller must not silently bind one Path or int to another's slot
    *,
    model_dir: Path,
    clean: bool = True,
    max_frames: int | None = None,
    creator_kwargs: dict[str, Any] | None = None,
) -> tuple[PointcloudResult, "Viewer | None"]:
    """
    Run a feedforward creator, optionally wrapped in loop closure, and save pointcloud.zarr.

    - pointcloud.zarr feeds the semantics lift and mesh stages
    - the Viewer is returned so the viser server stays reachable after this returns

    Args:
        backend: feedforward registry key.
        images_dir: the scene's images/ frame store, read in place.
        output_dir: backend directory; pointcloud.zarr lands here.
        loop_closure: bool, or a LoopClosureConfig knob dict whose `enabled` key defaults to True.
        viz_enabled: attach a viser Viewer; only with loop closure.
        viz_port: Viewer port.
        max_points: final point cap, drawn after the SOR clean.
        min_views: other views that must agree to keep a pixel; 0 turns the multiview filter off.
        mv_rel_thresh: multiview agreement tolerance as a fraction of depth.
        model_dir: where the binary COLMAP model is written.
        clean: pointcloud.clean.enabled; the creator SOR-cleans before any write.
        max_frames: preproc.max_frames; only gates the LoGeR frame-count advisory.
        creator_kwargs: the pointcloud.<backend> block; max_points / min_views / mv_rel_thresh / clean are
            reserved.

    Returns:
        (PointcloudResult, Viewer | None); the Viewer only with loop closure + viz.
    """
    # Normalize the bool|dict loop_closure config into (enabled, LoopClosureConfig|None)
    if isinstance(loop_closure, dict):
        # A knobs dict enables LC unless it explicitly sets enabled: false.
        lc_enabled = loop_closure.get("enabled") is not False
        lc_knobs = {k: v for k, v in loop_closure.items() if k != "enabled"}
        try:
            lc_config = LoopClosureConfig(**lc_knobs) if lc_knobs else None
        except TypeError as e:
            valid = [f.name for f in dataclasses.fields(LoopClosureConfig)]
            raise ValueError(f"Invalid pointcloud.loop_closure knob ({e}); valid keys: {valid}") from e
    else:
        lc_enabled = bool(loop_closure)
        lc_config = None

    # LoGeR refuses loop closure
    # - no LC verify thresholds are calibrated for LoGeR
    # - refused before any filesystem read: a config error must not surface as an IO error
    # - reads the normalized lc_enabled: {"enabled": False} is truthy with falsy intent
    if backend == "loger" and lc_enabled:
        raise ValueError(
            "pointcloud.loop_closure is not supported with backend 'loger'. LoGeR's windowed "
            "TTT memory already carries state across frames, and loop closure verification "
            "thresholds are calibrated per backbone. Use vggt_omega, vggtx, or mapanything."
        )

    # LoGeR under Omega's frame ceiling buys nothing: warn, don't change behavior
    # - preproc.max_frames is VGGT-Omega's GPU limit, already applied upstream
    # - None: no ceiling configured, no advice due
    if backend == "loger":
        n_frames = len(frames.frame_paths(images_dir))
        if max_frames is not None and n_frames <= max_frames:
            logger.warning(
                "LoGeR is running on %d frames, at or under the preproc.max_frames "
                "ceiling of %d. That ceiling is VGGT-Omega's GPU limit, not LoGeR's — LoGeR "
                "uses sliding-window inference and is built for longer sequences. Raise "
                "preproc.max_frames to use it.",
                n_frames,
                max_frames,
            )

    # Reserved creator kwargs: passed explicitly, so a clash in the backend block is refused by name
    # - otherwise an opaque TypeError naming neither key nor config path
    # - unknown keys are left to the constructor's TypeError, which names them
    extra = dict(creator_kwargs or {})
    explicit = {"max_points": max_points, "min_views": min_views, "mv_rel_thresh": mv_rel_thresh, "clean": clean}
    clash = sorted(extra.keys() & explicit.keys())
    if clash:
        raise ValueError(f"pointcloud.{backend}.{clash[0]} is not settable; use pointcloud.{clash[0]}")
    creator = get_creator(backend)(**explicit, **extra)

    # Wrap with loop closure if requested; viz has nothing to show without it, so only
    # attach the viser Viewer (also a heavy/websocket dep) when both are enabled
    viewer = None
    if lc_enabled:
        creator = LoopClosure(base=creator, config=lc_config)
        if viz_enabled:
            from collab_splats.viewer import Viewer

            viewer = Viewer(port=viz_port)
            creator.viz = viewer
            # Force live loop-edge timing so the viewer shows each loop correcting the
            # scene as it fires; deferred would defer the snap to a single end-of-run PGO.
            # ATE is identical either way — this is purely the live-build experience.
            creator.config.loop_edge_timing = "live"

    # Creators read the scene's images/ directory in place — no staging, no temp export
    result = creator.create_pointcloud(images_dir, output_dir, model_dir)

    # Persist the result to pointcloud.zarr — required by semantics lift + mesh stages
    zarr_path = output_dir / "pointcloud.zarr"
    result.save_zarr(zarr_path, extra_attrs={"method": "feedforward", "backend": backend})
    logger.info("pointcloud.zarr saved: %s  (%s pts)", zarr_path, f"{len(result.points):,}")

    # Drop the dense per-frame arrays: they are on disk now
    # - the returned result becomes Reconstructor.pointcloud, alive for the whole pipeline
    # - stages that need depth / world_points / images load the zarr themselves
    for name in _DENSE_FIELDS:
        setattr(result, name, None)

    # Explicitly release model + GPU memory before next stage (semantics) loads its model
    del creator
    pytorch_gc()
    logger.info("Pointcloud model released from GPU")

    return result, viewer


def _get_extractor(name: str):
    """Instantiate feature extractor by registry name."""
    from collab_splats.semantics.features import BaseFeatureExtractor

    return BaseFeatureExtractor.get(name)()


def _extract_2d_features(
    extractor_name: str,
    images_dir: Path,
    cache_dir: Path,
) -> Path:
    """
    Extract 2D features for all frames straight from the scene's images/ directory.

    Args:
        extractor_name: registry key of the extractor to run.
        images_dir: the scene's images/ directory of keyframes.
        cache_dir: directory the `<extractor>.zarr` patch cache is written into.

    Returns:
        Path of the written 2D patch cache.
    """
    # extract_feature_cache reads the keyframe JPGs from images/ one at a time — no full-RAM
    # load of the frame set.
    cache_dir.mkdir(parents=True, exist_ok=True)
    return extract_feature_cache(_get_extractor(extractor_name), images_dir, cache_dir)


def _lift_and_save(
    extractor_name: str,
    zarr_path: Path,
    pointcloud_zarr: Path,
    output_dir: Path,
    n_components: int | None,
    target_cosine: float | None,
    max_epochs: int,
    *,
    images_dir: Path,
) -> Path:
    """Load 2D feature cache + PointcloudResult, lift to 3D, compress, save.

    Writes output_dir/{extractor}_lifted.zarr (latent codes) and, when compressing,
    output_dir/{extractor}_ae.pt (weights + fit metrics) — the pair a consumer needs
    to recover full-dimensionality features.

    target_cosine/max_epochs are required, not defaulted: they are the config's single
    autoencoder policy, and a default here would be a fourth copy of it to drift from.

    images_dir is the store the 2D cache was extracted from, one row per images/ frame; the
    rows the pointcloud holds are picked out of it by frame index before lifting.
    """
    # Validate pointcloud zarr exists before attempting load
    if not pointcloud_zarr.exists():
        raise FileNotFoundError(
            f"pointcloud.zarr not found at {pointcloud_zarr}. "
            "Run build_pointcloud() with a feedforward backend first."
        )

    # Load feature maps from the 2D cache: one (D, H_p, W_p) tensor per images/ frame
    feature_maps = load_feature_maps(zarr_path)

    # Load PointcloudResult with depth/pixel data for lifting
    ff_result = PointcloudResult.load_zarr(pointcloud_zarr)

    # The cache is scene-level (every images/ frame); lift only the pointcloud's rows, in its order
    feature_maps = [feature_maps[row] for row in _store_rows(images_dir, ff_result.image_paths)]

    # Lift 2D features to 3D: (P, D)
    lifted = lift_features(feature_maps, ff_result)

    # Optional autoencoder compression → latent codes persisted alongside the weights
    ae = None
    if n_components is not None:
        # Move lifted to GPU for autoencoder training; lift_features returns CPU tensor
        lifted = lifted.to(get_device())
        ae = FeatureAutoencoder(input_dim=lifted.shape[-1], latent_dim=n_components)
        ae.fit(lifted, epochs=max_epochs, target_cosine=target_cosine)
        lifted = ae.per_point_encode(lifted)

    # One writer for both halves of the pair — it also stamps input_dim/latent_dim on the
    # zarr attrs, which is what tells a reader whether the weights are required at all
    # (n_components=None writes full-dim codes and no weights, legitimately).
    write_point_features(output_dir, extractor_name, to_numpy(lifted), ae)
    return output_dir


def _run_tsdf_mesh(
    result: PointcloudResult,
    pointcloud_zarr: Path,
    output_dir: Path,
    images_dir: Path,
    voxel_size: float,
    depth_trunc: float,
    sdf_trunc_mult: float = 4.0,
    conf_percentile: float | None = None,
    mask_sky: bool = False,
    source: str = "feedforward",
    splats_ckpt: Path | None = None,
    texture: bool = False,
    use_convex_hull: bool = False,
) -> Path:
    """
    Fuse depth and RGB into a TSDF mesh, clean it, and optionally texture it.

    Args:
        result: PointcloudResult; supplies poses and full-res K on the feedforward path.
        pointcloud_zarr: the scene's pointcloud.zarr; read on the feedforward path only.
        output_dir: receives mesh.ply, and texture/ when texture is set.
        images_dir: the scene's images/ directory of original-resolution keyframes.
        voxel_size: TSDF voxel edge, world units.
        depth_trunc: ignore depth beyond this, world units.
        sdf_trunc_mult: truncation band as a multiple of voxel_size; sets the thin-structure floor.
        conf_percentile: drop depth below this confidence percentile (None = off); feedforward only.
        mask_sky: zero out depth where the sky segmenter fires; applies to both sources.
        source: "feedforward" (zarr depth lifted to frame resolution) or "splats" (checkpoint renders).
        splats_ckpt: the splats stage's ckpt.pt; required when source is "splats".
        texture: also decimate, unwrap and project the fused views into output_dir/texture/.
        use_convex_hull: trim the ragged outer edge and patch the ground out to a rounded convex hull.
    Returns:
        Path to output_dir/mesh.ply.
    """
    # Splats source: renders come out at frame resolution carrying the poses they were rendered
    # with, pose-opt deltas included, so nothing here has to be lifted or re-posed.
    if source == "splats":
        # gsplat is CUDA-only; import lazily so Reconstructor stays importable without it
        from collab_splats.splats.checkpoint import render_tsdf_inputs

        depths, rgbs, c2w, intrinsics, image_ids = render_tsdf_inputs(splats_ckpt, images_dir)
    else:
        ff = PointcloudResult.load_zarr(pointcloud_zarr, load_images=False, load_world_points=False)
        if ff.depth is None:
            raise ValueError(f"{pointcloud_zarr} has no depth — cannot mesh.")
        if result.extrinsics.shape[0] != ff.depth.shape[0]:
            raise ValueError(
                f"Frame-count mismatch: the pointcloud result has {result.extrinsics.shape[0]} "
                f"images but {pointcloud_zarr} has {ff.depth.shape[0]}. They are from different "
                "runs — re-run the pointcloud stage, or point --stages mesh at the matching scene."
            )

        # Confidence masking first, on the model grid the confidence was predicted on.
        # Absent confidence is a property of the method (sfm, and any backend that ships none),
        # not an error — fuse unmasked and say so.
        depth = np.asarray(ff.depth)
        if conf_percentile is not None:
            if ff.confidence is None:
                logger.info(
                    "mesh.conf_percentile=%s but %s has no confidence array — fusing unmasked",
                    conf_percentile,
                    pointcloud_zarr,
                )
            else:
                keep = confidence_mask(np.asarray(ff.confidence), conf_percentile)
                depth = np.where(keep, depth, 0.0)

        # Lift model-res depth onto the original frame grid so the full-res K is the
        # right one to fuse with. Pairing one grid's depth with the other grid's K is the
        # 2026-08-11 collapse bug (5.06M -> 75k vertices).
        # Frames by the zarr's own rows: images/ may hold frames an incremental sfm model dropped
        image_ids = [frames.frame_idx_from_path(path) for path in ff.image_paths]
        rgbs = frames.read_frames(images_dir, image_ids)
        depths = upsample_depths(depth, rgbs, np.asarray(ff.original_coords)[:, :4])
        c2w = invert_poses(result.extrinsics)
        intrinsics = result.intrinsics

    # Sky fuses as a backdrop and seeds floaters; drop its depth after the arms converge
    # - idxs follows the arm: the checkpoint's ids for splats, the zarr's rows for feedforward
    # - the report is a fraction of pixels that HAD depth; a splats stack is mostly zero
    #   already, so dividing by depths.size would understate the cut several-fold
    if mask_sky:
        sky = sky_masks(images_dir, idxs=image_ids)
        if sky.shape != depths.shape:
            raise ValueError(
                f"Sky masks are {sky.shape} but depths are {depths.shape} — "
                f"{images_dir} does not match the depth source."
            )

        dropped = np.count_nonzero(sky & (depths > 0)) / max(np.count_nonzero(depths), 1)
        depths = np.where(sky, 0.0, depths)
        logger.info("mesh.mask_sky: dropped %.2f%% of valid depth pixels as sky", 100 * dropped)

    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = create_tsdf_mesh(
        depths,
        rgbs,
        c2w,
        intrinsics,
        output_dir,
        voxel_size=voxel_size,
        depth_trunc=depth_trunc,
        sdf_trunc=sdf_trunc_mult * voxel_size,
    )
    clean_repair_mesh(mesh_path, use_convex_hull=use_convex_hull)
    if texture:
        create_texture_mesh(mesh_path, output_dir / "texture", rgbs, c2w, intrinsics, voxel_size=voxel_size)
    return mesh_path


def _localization_db_exists(pointcloud_zarr: Path, extractor_name: str) -> bool:
    """True if the local-feature DB group already exists in pointcloud.zarr."""
    import zarr as zarr_lib

    try:
        store = zarr_lib.open_group(str(pointcloud_zarr), mode="r")
        return (
            "local_features" in store
            and extractor_name in store["local_features"]
            and "reconstruction" in store["local_features"][extractor_name]
        )
    except Exception:
        return False


def _build_localization_db(
    pointcloud_zarr: Path,
    extractor_name: str,
    images_dir: Path,
    top_k: int = 8,
    overwrite: bool = False,
) -> Path:
    """Build the per-frame local-feature localization cache into pointcloud.zarr.

    Loads the PointcloudResult, runs the local matcher over every DB frame, and persists
    keypoints/descriptors to group local_features/{extractor_name}/reconstruction. top_k
    is the pairwise (vismatch) matching fan-out; the descriptor path ignores it.
    """
    # overwrite: drop the stale reconstruction group so from_feedforward's cache check
    # misses and the index is re-extracted + re-saved (from_feedforward has no
    # overwrite notion of its own — an existing group always cache-hits).
    if overwrite:
        store = zarr.open_group(str(pointcloud_zarr), mode="a")
        rec_key = f"local_features/{extractor_name}/reconstruction"
        if rec_key in store:
            del store[rec_key]

    ff = PointcloudResult.load_zarr(pointcloud_zarr, load_images=True, load_world_points=True)
    extractor = LocalMatcher(extractor_name)

    # Boundary adapter: images/ directory → (images, ids) core objects
    # - lazy genexpr: zero reads on a cache hit, one imread per frame on a miss
    # - the zarr's image_paths fix the order: from_feedforward pairs row i with ff's geometry row i
    # - images/ may hold frames an incremental sfm model dropped; those are never read
    all_paths = frames.frame_paths(images_dir)
    paths = [all_paths[row] for row in _store_rows(images_dir, ff.image_paths)]
    images = (cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths)

    # Localization ids name the store's own files
    ids = [p.name for p in paths]
    CameraLocalizer.from_feedforward(
        ff,
        images=images,
        ids=ids,
        extractor=extractor,
        extractor_name=extractor_name,
        zarr_path=pointcloud_zarr,
        top_k=top_k,
    )
    logger.info("Localization DB built: %s :: local_features/%s", pointcloud_zarr, extractor_name)
    return pointcloud_zarr


@lru_cache(maxsize=1)
def _scene_frames(images_dir: Path) -> np.ndarray:
    """
    Every frame of a scene as one RGB stack, cached per directory.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        (N, H, W, 3) uint8 RGB.
    """
    return frames.read_frames(images_dir)


########################################
# Reconstructor
########################################


class Reconstructor:
    """5-stage environment reconstruction pipeline: preproc → pointcloud → semantics / mesh / localize."""

    def __init__(self, config: dict[str, Any], config_dir: str | Path = DEFAULT_CONFIG_DIR) -> None:
        """Merge config over base.yaml defaults, validate, and store."""
        # Load base defaults; deep-merge the caller's config over them so every key is present
        base_path = Path(config_dir) / "base.yaml"
        with open(base_path) as f:
            defaults = yaml.safe_load(f) or {}
        merged = merge({}, defaults, config)

        # Validate shape, then store the fully-populated config
        self.config = self.validate_config(merged)
        self.pointcloud: PointcloudResult | None = None
        # Set by build_pointcloud() when pointcloud.viz.enabled + loop_closure; lets callers
        # (e.g. run_pipeline.py --keep-viewer) reach the viser server after run() returns.
        self.viewer: "Viewer | None" = None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Validate required fields and method/backend consistency.

        Raises:
            ValueError: If required fields missing or backend invalid for method.
        """
        # Required top-level fields
        for field in ("input_path", "output_path"):
            if field not in config or config[field] is None:
                raise ValueError(f"Reconstructor config missing required field: '{field}'")

        # `preprocessing` was renamed to `preproc` (2026-08-22) to match the module
        # and the stage name. Refuse a stale block rather than silently ignoring it
        # and substituting base.yaml defaults — every run_config.yaml already under
        # environments-processed/ carries the old name.
        if "preprocessing" in config:
            raise ValueError(
                "config key 'preprocessing' was renamed to 'preproc' (2026-08-22); " "rename the section in your config"
            )

        # Removed geometric verification
        # - truthy only: published run_config.yaml files carry `geometric_verification: false`,
        #   and a leaf re-run keeps their pointcloud section
        if config.get("pointcloud", {}).get("geometric_verification"):
            raise ValueError(_VERIFY_REMOVED)

        # Single-arg .get() returns None if absent — the membership checks below reject None,
        # so no inline value defaults are needed (base.yaml is the sole default source).
        pc = config.get("pointcloud", {})
        method = pc.get("method")
        backend = pc.get("backend")

        if method not in _VALID_METHODS:
            raise ValueError(f"pointcloud.method must be one of {_VALID_METHODS}, got '{method}'")

        if method == "feedforward" and backend not in _FEEDFORWARD_BACKENDS:
            raise ValueError(
                f"pointcloud.backend must be one of {_FEEDFORWARD_BACKENDS} "
                f"for method='feedforward', got '{backend}'"
            )
        if method == "sfm" and backend not in _SFM_BACKENDS:
            raise ValueError(f"pointcloud.backend must be one of {_SFM_BACKENDS} " f"for method='sfm', got '{backend}'")

        # BA over LC submaps is unsupported: submaps don't carry the per-frame model tensors
        # track extraction needs. Fail loud at construction instead of silently skipping one.
        lc = pc.get("loop_closure")
        lc_enabled = lc.get("enabled") is not False if isinstance(lc, dict) else bool(lc)
        if pc.get("bundle_adjustment") and lc_enabled:
            raise ValueError(
                "pointcloud.bundle_adjustment and pointcloud.loop_closure are mutually "
                "exclusive — BA needs per-frame model tensors that LC submaps do not carry."
            )

        # SfM path: every sfm mapper runs its own BA — refuse the flag
        if method == "sfm" and pc.get("bundle_adjustment"):
            raise ValueError(
                "pointcloud.bundle_adjustment is not supported with method: sfm — "
                "every sfm backend runs its own bundle adjustment"
            )

        # SfM path: LC wraps a feedforward creator in sequential submaps — nothing to wrap here
        if method == "sfm" and lc_enabled:
            raise ValueError(
                "pointcloud.loop_closure is not supported with method: sfm — "
                "sfm backends map the whole frame set at once, not in sequential submaps"
            )

        # sfm sub-block bounds: every key is read after SIFT or the mapper has started
        if method == "sfm":
            Reconstructor._validate_sfm_block(pc, backend)

        # Mesh truncation band: a band narrower than a voxel leaves gaps between adjacent
        # voxels' zero crossings, so the extracted surface is punctured rather than thin
        sdf_mult = config.get("mesh", {}).get("sdf_trunc_mult")
        if isinstance(sdf_mult, bool) or not isinstance(sdf_mult, (int, float)) or sdf_mult < 1.0:
            raise ValueError(f"mesh.sdf_trunc_mult must be a number >= 1.0, got {sdf_mult!r}")

        return config

    @staticmethod
    def _validate_sfm_block(pc: dict, backend: str) -> None:
        """
        Bounds-check an sfm sub-block at config load.

        - every key is consumed after SIFT or the mapper has started: a typo would cost a whole run
        - hloc conf names NOT checked against hloc.*.confs: that would import the optional extra

        Args:
            pc: the config's pointcloud section.
            backend: a key of SFM_CREATORS.
        """
        block = pc[backend]

        # Unknown keys would reach the creator constructor as a TypeError, mid-run
        unknown = set(block) - _SFM_BLOCK_KEYS[backend]
        if unknown:
            raise ValueError(f"pointcloud.{backend} has unknown keys {sorted(unknown)}")

        # Registered-frame floor: a share in (0, 1], shared by every backend
        frac = block["min_registered_frac"]
        if isinstance(frac, bool) or not (isinstance(frac, (int, float)) and 0 < frac <= 1):
            raise ValueError(f"pointcloud.{backend}.min_registered_frac must be a number in (0, 1], got {frac!r}")

        # Thread cap: an int >= 1 for every backend; bool is an int subclass, so it is rejected explicitly
        threads = block["num_threads"]
        if isinstance(threads, bool) or not (isinstance(threads, int) and threads >= 1):
            raise ValueError(f"pointcloud.{backend}.num_threads must be an int >= 1, got {threads!r}")

        # instantsfm: seed in np.random.seed's domain; a track needs two views to triangulate
        if backend == "instantsfm":
            random_seed = block["random_seed"]
            if random_seed is not None and not (isinstance(random_seed, int) and 0 <= random_seed < 2**32):
                raise ValueError(
                    f"pointcloud.instantsfm.random_seed must be null or an int in [0, 2**32), got {random_seed!r}"
                )
            min_views = block["min_num_view_per_track"]
            if min_views is not None and not (isinstance(min_views, int) and min_views >= 2):
                raise ValueError(
                    f"pointcloud.instantsfm.min_num_view_per_track must be null or an int >= 2, got {min_views!r}"
                )
            return

        # Pairing: one of sift_db's modes, shared by both backends
        pairing = block["pairing"]
        if pairing not in _SFM_PAIRINGS:
            raise ValueError(f"pointcloud.{backend}.pairing must be one of {_SFM_PAIRINGS}, got {pairing!r}")

        # Counts: bool is an int subclass, so it is rejected explicitly
        for key in ("overlap", "num_retrieved"):
            value = block[key]
            if isinstance(value, bool) or not (isinstance(value, int) and value >= 1):
                raise ValueError(f"pointcloud.{backend}.{key} must be an int >= 1, got {value!r}")

        # hloc conf names: non-empty strings; hloc itself resolves them at run time
        if backend == "hloc":
            for key in ("retrieval_conf", "feature_conf", "matcher_conf"):
                value = block[key]
                if not (isinstance(value, str) and value):
                    raise ValueError(f"pointcloud.hloc.{key} must be a non-empty string, got {value!r}")

    ########################################
    # Path properties
    ########################################

    @property
    def backend_dir(self) -> Path:
        """output_path / backend — e.g. out/vggtx/. Backend subdir for all stage 2+ artifacts."""
        return Path(self.config["output_path"]) / self.config["pointcloud"]["backend"]

    @property
    def images_dir(self) -> Path:
        """output_path / images — canonical decode-once keyframe store for this run."""
        return Path(self.config["output_path"]) / "images"

    @property
    def pointcloud_zarr(self) -> Path:
        """
        Unified reconstruction zarr for this backend (all pointcloud methods).
        """
        return self.backend_dir / "pointcloud.zarr"

    @property
    def colmap_model_dir(self) -> Path:
        """
        COLMAP binary model for this backend; present only as a whole model.
        """
        return self.backend_dir / "colmap" / "sparse" / "0"

    @property
    def semantics_cache_dir(self) -> Path:
        """output_path / semantics/ — 2D patch cache, one {extractor}.zarr per extractor.

        Scene-level, not backend-level: the 2D features depend only on the frames, so every
        backend lifts from the same cache. The per-backend lift lands under backend_dir.
        """
        return Path(self.config["output_path"]) / "semantics"

    ########################################
    # Pipeline stages
    ########################################

    def preprocess(self, overwrite: bool = False) -> Path:
        """Extract frames from input video/dir into images/ (sole persistent store)."""
        # Skip if the store already exists and overwrite not requested
        if not overwrite and self.images_dir.exists():
            logger.info("Frames already extracted at %s, skipping preprocess", self.images_dir)
            return self.images_dir

        if overwrite and self.images_dir.exists():
            shutil.rmtree(self.images_dir)

        pre_cfg = self.config["preproc"]
        n_frames = extract_frames(
            input_path=Path(self.config["input_path"]),
            images_dir=self.images_dir,
            frame_selection=pre_cfg["frame_selection"],
            fps=pre_cfg["fps"],
            min_frames=pre_cfg["min_frames"],
            max_frames=pre_cfg["max_frames"],
            n_workers=pre_cfg["n_workers"],
            undistort=pre_cfg["undistort"],
            quality=pre_cfg["quality"],
            on_empty_slot=pre_cfg["on_empty_slot"],
        )
        logger.info(
            "Preprocessing complete: %d frames at %s",
            n_frames,
            self.images_dir,
        )
        return self.images_dir

    def build_pointcloud(self, overwrite: bool = False) -> PointcloudResult:
        """Run pointcloud stage. Sets self.pointcloud, returns PointcloudResult."""
        pc_cfg = self.config["pointcloud"]
        method = pc_cfg["method"]

        # Skip when pointcloud.zarr and the COLMAP export both exist and overwrite is off
        # - the result is reloaded from the zarr; the COLMAP model is only a done-marker here
        # - either one missing means a partial run, so inference runs again
        if not overwrite and self._stage_output_exists("pointcloud"):
            logger.info("Pointcloud exists at %s, loading from disk", self.pointcloud_zarr)
            self.pointcloud = self._load_pointcloud_from_disk()
            return self.pointcloud

        # Dispatch to the appropriate reconstruction method
        if method == "sfm":
            warnings.warn(
                "pointcloud.method='sfm' is experimental and not production-tested.",
                UserWarning,
                stacklevel=2,
            )
            result = self._run_sfm()
        else:
            result, viewer = _run_feedforward(
                backend=pc_cfg["backend"],
                images_dir=self.images_dir,
                output_dir=self.backend_dir,
                loop_closure=pc_cfg["loop_closure"],
                viz_enabled=pc_cfg["viz"]["enabled"],
                viz_port=pc_cfg["viz"]["port"],
                max_points=pc_cfg["max_points"],
                min_views=pc_cfg["min_views"],
                mv_rel_thresh=pc_cfg["mv_rel_thresh"],
                model_dir=self.colmap_model_dir,
                clean=pc_cfg["clean"]["enabled"],
                max_frames=self.config["preproc"]["max_frames"],
                creator_kwargs=pc_cfg.get(pc_cfg["backend"], {}),
            )
            self.viewer = viewer

        # The PLY carries the same cleaned point set as pointcloud.zarr and the COLMAP model
        result.write_ply(self.backend_dir / "sparse_pc.ply")

        self.pointcloud = result
        return result

    def _load_pointcloud_from_disk(self) -> PointcloudResult:
        """
        Load pointcloud.zarr's points and cameras; the dense per-frame arrays stay on disk.

        - stages that need depth / world_points / images load the zarr themselves
        """
        return PointcloudResult.load_zarr(
            self.pointcloud_zarr,
            load_depth=False,
            load_world_points=False,
            load_confidence=False,
            load_pixel_indices=False,
        )

    def _run_sfm(self) -> PointcloudResult:
        """
        SfM path: BaseSfmCreator.create_pointcloud over the keyframes -> pointcloud.zarr.

        - images/ is read in place: it is already the COLMAP layout
        - the creator writes colmap/sparse/0; the zarr carries its subset and depth-alignment attrs

        Returns:
            The PointcloudResult for the shared tail.
        """
        pc_cfg = self.config["pointcloud"]
        backend = pc_cfg["backend"]

        # Creator from the whole block plus the shared clean switch and cap; one call does the rest
        creator = SFM_CREATORS[backend](
            clean=pc_cfg["clean"]["enabled"], max_points=pc_cfg["max_points"], **pc_cfg[backend]
        )
        outputs = creator.create_pointcloud(self.images_dir, self.backend_dir, self.colmap_model_dir)
        outputs.save_zarr(self.pointcloud_zarr, extra_attrs={"backend": backend, **creator.attrs})
        logger.info("pointcloud.zarr saved: %s  (%s pts)", self.pointcloud_zarr, f"{len(outputs.points):,}")

        # Drop the dense per-frame arrays: they are on disk now, and self.pointcloud outlives the stage
        for name in _DENSE_FIELDS:
            setattr(outputs, name, None)

        return outputs

    def refine_poses(self, overwrite: bool = False) -> PointcloudResult:
        """
        Refine camera poses via LM bundle adjustment; rewrite pose-derived artifacts.

        - one implementation for both triggers: inline after the pointcloud stage when
          pointcloud.bundle_adjustment is enabled, and from disk via --stages refine
        - loads everything from pointcloud.zarr — no live creator needed
        - after reproject: SOR re-clean (pointcloud.clean.enabled), then re-cap to max_points;
          the point count can change, so lifted semantics must be re-run
        """
        # SfM results are already globally bundle-adjusted; LM re-refinement is undefined here
        pc_cfg = self.config["pointcloud"]
        if pc_cfg["method"] == "sfm":
            raise ValueError("refine_poses is not supported for pointcloud.method: sfm")

        # Skip when already refined — run_pipeline refuses NAMED re-runs generically, so this
        # mirrors the other stage methods' silent skip for config-driven repeat runs.
        marker = self.backend_dir / "colmap" / "refine.json"
        if marker.exists() and not overwrite:
            logger.info("Poses already refined (%s), skipping refine", marker)
            return self._resolve_result()

        # Load the full PointcloudResult from zarr: images/confidence/world_points feed track
        # extraction, depth+pixel_indices feed the deterministic creator-free reproject.
        zarr_path = self.pointcloud_zarr
        if not zarr_path.exists():
            raise FileNotFoundError(f"refine requires {zarr_path}; run the pointcloud stage first.")
        ff = PointcloudResult.load_zarr(zarr_path, load_images=True)

        # Refine poses with LM BA, then re-derive the point set under the new cameras
        # - VGGSfM tracks live on the model grid, so K must too; checked before the slow extraction
        # - intrinsics=None: __post_init__ re-derives the full-res K from the refined model-grid K
        check_model_resolution(ff.model_intrinsics, ff.images, ff.original_coords)
        cfg = BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir)
        ba = BundleAdjustment(cfg)
        extrinsics, intrinsics = ba.refine(
            ff.images, ff.confidence, ff.world_points, ff.extrinsics, ff.model_intrinsics, ff.image_paths
        )
        ff = dataclasses.replace(ff, extrinsics=extrinsics, model_intrinsics=intrinsics, intrinsics=None).reproject()

        # Re-clean under the refined cameras, then re-cap to max_points
        # - a moved pose can throw a pixel into a new outlier
        # - the cap binds only when the config lowered it since the pointcloud stage
        n_before = len(ff.points)
        ff = clean_pointcloud(ff, remove_outliers=pc_cfg["clean"]["enabled"], max_points=pc_cfg["max_points"])
        logger.info("refine: %d of %d pts kept after clean + cap", len(ff.points), n_before)

        # Re-export COLMAP under the refined cameras
        recon = ff.to_colmap()
        write_colmap_reconstruction(recon, self.colmap_model_dir)

        # Write pose-derived arrays back to pointcloud.zarr so zarr and COLMAP never disagree
        # (localization samples world_points; a later --stages refine re-reads these poses).
        store = zarr.open(str(zarr_path), mode="r+")
        store["extrinsics"][:] = ff.extrinsics
        store["intrinsics"][:] = ff.intrinsics
        store["model_intrinsics"][:] = ff.model_intrinsics

        # Per-point arrays changed length with the clean: rewrite them, a slice-assign cannot resize
        # - pixel_indices first: reproject rebuilds the points from it
        # - a mid-loop crash leaves mixed lengths; refine.json is written last, so the
        #   done-check still refuses to treat that state as refined
        for name, arr in (("pixel_indices", ff.pixel_indices), ("points", ff.points), ("colors", ff.colors)):
            store.create_array(name, data=arr, chunks=arr.shape, compressors=LZ4, overwrite=True)

        if "world_points" in store:
            store["world_points"][:] = ff.world_points

        # Reload the refined zarr light — it is both the returned result and the source for
        # the refreshed sparse_pc.ply
        result = self._load_pointcloud_from_disk()
        result.write_ply(self.backend_dir / "sparse_pc.ply")

        # Marker + provenance in one file: BA config and per-step LM loss history
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            marker,
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()},
                "loss_history": ba.loss_history,
                "n_frames": len(ff.image_paths),
            },
        )

        self.pointcloud = result
        return result

    def extract_semantics(
        self,
        result: PointcloudResult | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Extract 2D features (cached), lift to 3D, compress. Returns the lifted-pair dir."""
        sem_cfg = self.config["semantics"]
        extractor_name = sem_cfg["extractor"]
        n_components = sem_cfg["n_components"]

        lifted_dir = self.backend_dir / "semantics"

        # Skip if this extractor's lifted features are already on disk. The extractor is in the
        # filename, so two extractors coexist here instead of overwriting each other.
        if not overwrite and self._stage_output_exists("semantics"):
            logger.info("Lifted features exist at %s, skipping", lifted_store_path(lifted_dir, extractor_name))
            return lifted_dir

        result = result or self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # Stage 1: 2D feature extraction (cached at semantics/{extractor}.zarr, scene-level)
        zarr_path = self.semantics_cache_dir / f"{extractor_name}.zarr"
        if overwrite or not zarr_path.exists():
            logger.info("Extracting 2D features with %s", extractor_name)
            zarr_path = _extract_2d_features(extractor_name, self.images_dir, self.semantics_cache_dir)
        else:
            logger.info("2D feature cache hit: %s", zarr_path)

        # Stage 2: Lift to 3D and save
        pointcloud_zarr = self.pointcloud_zarr
        logger.info("Lifting 2D features to 3D pointcloud")
        out_dir = _lift_and_save(
            extractor_name,
            zarr_path,
            pointcloud_zarr,
            lifted_dir,
            n_components,
            target_cosine=sem_cfg["target_cosine"],
            max_epochs=sem_cfg["max_epochs"],
            images_dir=self.images_dir,
        )
        return out_dir

    def mesh(
        self,
        result: PointcloudResult | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Build a TSDF mesh from `mesh.source` depth: pointcloud.zarr (default) or splats ckpt.pt renders.

        pointcloud.zarr is the pose authority on the feedforward path; the splats path fuses the
        checkpoint's own renders — the poses the splats were rendered with (pose-opt deltas
        included), with alpha as confidence — so splats.zarr is not an input. The splats stage
        is never auto-run — `source: splats` requires ckpt.pt on disk.
        Returns path to mesh.ply.
        """
        mesh_path = self.backend_dir / "mesh.ply"

        # Skip if mesh already on disk
        if not overwrite and self._stage_output_exists("mesh"):
            logger.info("Mesh exists at %s, skipping", mesh_path)
            return mesh_path

        result = result or self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # Source-specific input check: each branch needs only its own artifact on disk
        mesh_cfg = self.config["mesh"]
        source = mesh_cfg["source"]
        if source not in ("feedforward", "splats"):
            raise ValueError(f"mesh.source must be 'feedforward' or 'splats', got {source!r}")
        pointcloud_zarr = self.pointcloud_zarr
        splats_ckpt = None
        if source == "splats":
            splats_ckpt = self.backend_dir / "splats" / "ckpt.pt"
            if not splats_ckpt.exists():
                raise ValueError(
                    f"mesh.source: splats needs {splats_ckpt} — run the splats stage first " "(it is never auto-run)"
                )
        elif not pointcloud_zarr.exists():
            raise FileNotFoundError(
                f"pointcloud.zarr not found at {pointcloud_zarr}. "
                "Mesh requires pointcloud.zarr depth maps — run the pointcloud stage first."
            )

        out = _run_tsdf_mesh(
            result=result,
            pointcloud_zarr=pointcloud_zarr,
            output_dir=self.backend_dir,
            images_dir=self.images_dir,
            voxel_size=mesh_cfg["voxel_size"],
            depth_trunc=mesh_cfg["depth_trunc"],
            sdf_trunc_mult=mesh_cfg["sdf_trunc_mult"],
            conf_percentile=mesh_cfg["conf_percentile"],
            mask_sky=mesh_cfg["mask_sky"],
            source=source,
            splats_ckpt=splats_ckpt,
            texture=mesh_cfg["texture"],
            use_convex_hull=mesh_cfg["use_convex_hull"],
        )
        logger.info("Mesh saved to %s", out)
        return out

    def build_localization_db(self, overwrite: bool = False) -> Path:
        """Build/refresh the per-frame local-feature localization cache in pointcloud.zarr."""
        loc_cfg = self.config["localization"]
        extractor_name = loc_cfg["matcher"]

        pointcloud_zarr = self.pointcloud_zarr
        if not pointcloud_zarr.exists():
            raise FileNotFoundError(
                f"pointcloud.zarr not found at {pointcloud_zarr}. "
                "Localization DB requires a feedforward pointcloud stage first."
            )

        # Skip if the DB group already exists and overwrite not requested
        if not overwrite and self._stage_output_exists("localize"):
            logger.info(
                "Localization DB exists at %s :: local_features/%s, skipping",
                pointcloud_zarr,
                extractor_name,
            )
            return pointcloud_zarr

        return _build_localization_db(
            pointcloud_zarr,
            extractor_name,
            self.images_dir,
            top_k=loc_cfg["top_k"],
            overwrite=overwrite,
        )

    def splats(self, overwrite: bool = False) -> Path:
        """
        Train Gaussian splats from the pointcloud stage. Returns path to splats/ckpt.pt.
        """
        out_dir = self.backend_dir / "splats"
        splats_ckpt = out_dir / "ckpt.pt"
        if not overwrite and self._stage_output_exists("splats"):
            logger.info("Splats exist at %s, skipping", out_dir)
            return splats_ckpt

        result = self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # gsplat is CUDA-only; import lazily so Reconstructor stays importable without it
        from collab_splats.splats.trainer import SplatsConfig, train

        cfg = SplatsConfig.from_dict(self.config["splats"])

        # Frames in the result's row order, looked up in images/ by the frame index in each name.
        # _scene_frames caches the whole-directory stack, so an earlier read in the same
        # process re-reads nothing; rows are then picked by position out of that stack.
        frame_indices = [frames.frame_idx_from_path(path) for path in result.image_paths]
        # CPU-resident by design: train() moves one view to the GPU at a time
        images = _scene_frames(self.images_dir)[_store_rows(self.images_dir, result.image_paths)]

        # Depth targets: pointcloud.zarr depth masked and lifted like the mesh stage does it (0 = no
        # target); train() only resizes them further for its coarse-to-fine schedule
        depth_targets = None
        depth_on = "depth" in cfg.losses and cfg.losses["depth"]["weight"] > 0  # from_dict guarantees weight
        if depth_on:
            pointcloud_zarr = self.pointcloud_zarr
            if not pointcloud_zarr.exists():
                raise FileNotFoundError(
                    f"pointcloud.zarr not found at {pointcloud_zarr}. "
                    "Splats depth loss requires depth maps from the pointcloud stage."
                )

            feedforward = PointcloudResult.load_zarr(pointcloud_zarr, load_images=False, load_world_points=False)
            if feedforward.depth is None:
                raise ValueError(f"{pointcloud_zarr} has no depth — cannot build splats depth targets.")

            # Feedforward rows follow the zarr's own image_paths order; align to result.image_paths
            # by frame index so depth (and confidence, before masking) match the frames above
            feedforward_rows = {
                frames.frame_idx_from_path(path): row for row, path in enumerate(feedforward.image_paths)
            }
            missing = [frame_idx for frame_idx in frame_indices if frame_idx not in feedforward_rows]
            if missing:
                raise ValueError(
                    f"splats depth loss: {len(missing)} reconstruction frames have no depth in "
                    f"{pointcloud_zarr} (frame_idx {missing[:5]}) — re-run the pointcloud stage."
                )
            rows = [feedforward_rows[frame_idx] for frame_idx in frame_indices]
            depth_targets = np.ascontiguousarray(feedforward.depth[rows], dtype=np.float32)
            conf_percentile = self.config["mesh"]["conf_percentile"]
            if conf_percentile is not None and feedforward.confidence is not None:
                confidence = to_numpy(feedforward.confidence)[rows]
                keep = confidence_mask(confidence, conf_percentile)
                depth_targets = np.where(keep, depth_targets, 0.0).astype(np.float32)
            elif conf_percentile is not None:
                # No confidence channel, so mesh.conf_percentile cannot apply — report the
                # share of targets that are already zero (= no target) rather than drop the
                # setting silently.
                zero_fraction = 100.0 * float((depth_targets <= 0).mean())
                logger.info(
                    "splats depth targets: mesh.conf_percentile=%s not applied (no confidence "
                    "channel); %.2f%% of target pixels are zero",
                    conf_percentile,
                    zero_fraction,
                )

            # Lift model-res depth onto the frame grid through each row's crop box, as the mesh
            # stage does — a plain resize stretches a cropped/padded model grid over the frame
            crop_boxes = np.asarray(feedforward.original_coords)[rows, :4]
            depth_targets = upsample_depths(depth_targets, images, crop_boxes)

        train(
            cfg,
            images,
            result.extrinsics,
            result.intrinsics,
            result.points,
            result.colors,
            out_dir,
            depth_targets=depth_targets,
            image_ids=frame_indices,
        )
        logger.info("Splats saved to %s", out_dir)
        return splats_ckpt

    def reconstruction_quality_report(self, overwrite: bool = False) -> Path:
        """
        Reference-free error report: columnar tables, one reconstruction_quality_report.json.

        - named for the artifact it writes; distinct from the video quality report (capture)
        - null photometric table without images; columns listed in geometry/metrics.py
        - a failing measurement, or a stale report on disk, raises
        - runs no model and no matcher; reads the zarr and images/ only

        Args:
            overwrite: rebuild the report even when it already exists on disk.

        Returns:
            Path to reconstruction_quality_report.json.
        """
        out_json = self.backend_dir / "reconstruction_quality_report.json"
        if not overwrite and self._stage_output_exists("reconstruction_quality_report"):
            if "frames" not in json.loads(out_json.read_text()):
                raise ValueError(
                    f"{out_json} is a stale reconstruction quality report (no 'frames'); delete it and re-run"
                )
            logger.info("Reconstruction quality report exists at %s, skipping", out_json)
            return out_json
        if self._resolve_result() is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # Load the result; the report runs its own cross-view pass
        ff = PointcloudResult.load_zarr(self.pointcloud_zarr)

        # Optional keyframes for photometric, picked by the zarr's own rows
        # - images/ may hold frames an incremental sfm model dropped; filename order would mispair
        images = None
        if frames.frame_paths(self.images_dir):
            images = _scene_frames(self.images_dir)[_store_rows(self.images_dir, ff.image_paths)].astype(np.float32)

        tables = compute_reconstruction_quality(
            ff.depth,
            ff.model_intrinsics,
            ff.intrinsics,
            ff.extrinsics,
            ff.original_coords,
            [Path(str(p)).name for p in ff.image_paths],
            ff.confidence,
            images,
        )
        report = {
            "scene": {
                "backend": self.config["pointcloud"]["backend"],
                "n_frames": len(ff.depth),
                "model_resolution": f"{ff.model_width}x{ff.model_height}",
                "image_width": int(ff.original_coords[0][4]),
                "zarr": str(self.pointcloud_zarr),
            },
            **tables,
        }

        # Atomic write: a crash mid-dump must not leave a half file that reuse-by-existence trusts
        out_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_json, report)
        logger.info("Reconstruction quality report written to %s", out_json)
        return out_json

    def _stage_output_exists(self, stage: str) -> bool:
        """True if `stage`'s on-disk output is already present (lets deps be reused across runs)."""
        if stage == "preproc":
            return self.images_dir.exists()
        if stage == "pointcloud":
            return self.pointcloud_zarr.exists() and self.colmap_model_dir.exists()
        if stage == "refine":
            return (self.backend_dir / "colmap" / "refine.json").exists()
        # Leaf-stage markers. Only preproc/pointcloud are ever depended on, but run_pipeline also
        # needs these to refuse a named stage whose output already exists — and each leaf stage's
        # own skip-check reads them, so they live here once instead of three times.
        if stage == "splats":
            return (self.backend_dir / "splats" / "ckpt.pt").exists()
        if stage == "mesh":
            return (self.backend_dir / "mesh.ply").exists()
        if stage == "semantics":
            lifted = lifted_store_path(self.backend_dir / "semantics", self.config["semantics"]["extractor"])
            return lifted.exists()
        if stage == "localize":
            pointcloud_zarr = self.pointcloud_zarr
            return pointcloud_zarr.exists() and _localization_db_exists(
                pointcloud_zarr, self.config["localization"]["matcher"]
            )
        if stage == "reconstruction_quality_report":
            return (self.backend_dir / "reconstruction_quality_report.json").exists()
        return False

    def _resolve_result(self) -> PointcloudResult | None:
        """PointcloudResult for a stage-2+ run, loading pointcloud.zarr if not in memory."""
        # A stage run on its own never calls build_pointcloud(), so self.pointcloud is None even
        # when a complete reconstruction is already sitting in backend_dir.
        if self.pointcloud is None and self._stage_output_exists("pointcloud"):
            self.pointcloud = self._load_pointcloud_from_disk()
        return self.pointcloud

    def run_pipeline(
        self,
        stages: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Run named stages in dependency order.

        Args:
            stages: Subset of ["preproc", "pointcloud", "refine", "semantics", "splats", "mesh",
                    "localize", "reconstruction_quality_report"].
                    Default: all enabled stages from config.
            overwrite: Re-run stages even if output exists.

        Raises:
            ValueError: If stages list violates dependency ordering, names the removed verify
                stage, or names a LEAF_STAGES stage whose output is on disk and overwrite is False.
        """
        # The removed verify stage: say so, not an unknown-stage no-op
        if stages is not None and "verify" in stages:
            raise ValueError(_VERIFY_REMOVED)

        # Naming a stage means asking for it; inheriting it from config does not. Capture the
        # distinction before `stages` is reassigned below.
        named = stages is not None
        if stages is None:
            # Build from config enabled flags; preproc + pointcloud always included
            stages = ["preproc", "pointcloud"]
            if self.config["pointcloud"]["bundle_adjustment"]:
                stages.append("refine")
            if self.config["semantics"]["enabled"]:
                stages.append("semantics")
            if self.config["splats"]["enabled"]:
                stages.append("splats")
            if self.config["mesh"]["enabled"]:
                stages.append("mesh")
            if self.config["localization"]["enabled"]:
                stages.append("localize")
            # Report always on, no enable boolean
            # - runs no model and no matcher, only reads the zarr just written
            stages.append("reconstruction_quality_report")

        # Validate stage dependencies before starting any work. A dependency is
        # satisfied when it's in this run's stages OR its output already exists on
        # disk — so `--stages pointcloud` reuses a prior preprocess's images/.
        stages_set = set(stages)
        for stage in stages:
            for dep in _STAGE_DEPS.get(stage, []):
                if dep not in stages_set and not self._stage_output_exists(dep):
                    raise ValueError(
                        f"Stage '{stage}' requires '{dep}', but '{dep}' is neither in "
                        f"stages={stages} nor already on disk. Add '{dep}' to the stages list "
                        f"(or run it first)."
                    )
            # Refuse a LEAF stage the caller NAMED whose output already exists, instead of
            # silently no-op'ing. A remote re-run would otherwise pull the whole scene, skip
            # every stage, push nothing and report success. Scoped to leaves because every
            # re-run set is leaf-only by construction, while a named non-leaf stage is how the
            # local drivers resume: `--stages preproc,pointcloud,localize` after a localize
            # failure must skip the two completed upstream stages, not refuse them.
            if named and stage in LEAF_STAGES and not overwrite and self._stage_output_exists(stage):
                raise ValueError(f"Stage '{stage}' output already exists; pass overwrite=True to replace it.")

        # Execute stages in canonical order
        result = None
        for stage in [s for s in _STAGE_ORDER if s in stages_set]:
            logger.info("=== Stage: %s ===", stage)
            if stage == "preproc":
                self.preprocess(overwrite=overwrite)
            elif stage == "pointcloud":
                result = self.build_pointcloud(overwrite=overwrite)
            elif stage == "refine":
                result = self.refine_poses(overwrite=overwrite)
            elif stage == "semantics":
                self.extract_semantics(result=result, overwrite=overwrite)
            elif stage == "splats":
                self.splats(overwrite=overwrite)
            elif stage == "mesh":
                self.mesh(result=result, overwrite=overwrite)
            elif stage == "localize":
                self.build_localization_db(overwrite=overwrite)
            elif stage == "reconstruction_quality_report":
                self.reconstruction_quality_report(overwrite=overwrite)

    def launch_dashboard(self) -> None:
        """Launch interactive dashboard for current reconstruction state."""
        from collab_splats.dashboard.__main__ import (
            main as dashboard_main,  # optional heavy dep; lazy load
        )

        dashboard_main()
