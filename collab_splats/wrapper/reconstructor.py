"""5-stage reconstruction pipeline wrapper."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import shutil
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pycolmap
import torch
import yaml
import zarr
from mergedeep import merge
from vggt.utils.geometry import unproject_depth_map_to_point_map

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh import (
    clean_repair_mesh,
    fuse_tsdf,
    fuse_tsdf_bands,
    texture_mesh,
)
from collab_splats.mesh.io import render_tsdf_inputs, upsample_depths
from collab_splats.mesh.tsdf import check_bands
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.depth_align import result_from_reconstruction
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.sfm import InstantSfMCreator
from collab_splats.pointcloud.utils import (
    clean_pointcloud,
    confidence_mask,
    lift_features,
)
from collab_splats.pointcloud.vda import generate_vda_depth, vda_depth_complete
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
from collab_splats.semantics.segmentation import sky_masks
from collab_splats.semantics.utils import (
    extract_feature_cache,
    lifted_store_path,
    load_feature_maps,
    write_point_features,
)

if TYPE_CHECKING:
    from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# base.yaml is the single source of defaults; __init__ merges any passed config over it.
DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"

_FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega", "loger"}
# Only instantsfm is wired into _run_sfm. ColmapCreator/HlocCreator exist and work, but
# nothing dispatches to them — listing them here would let a config load cleanly and then
# die mid-run, after preproc had already burned its time.
_SFM_BACKENDS = {"instantsfm"}
_VALID_METHODS = {"feedforward", "sfm"}
_STAGE_ORDER = [
    "preproc",
    "pointcloud",
    "refine",
    "semantics",
    "splats",
    "mesh",
    "localize",
    "verify",
    "reconstruction_quality_report",
]
_STAGE_DEPS: dict[str, list[str]] = {
    "preproc": [],
    "pointcloud": ["preproc"],
    # refine rewrites pointcloud outputs in place; deliberately NOT a dependency of the
    # stages below — that would demote them from LEAF_STAGES and break their disk re-run.
    # Staleness contract: after --stages refine, re-run dependents with overwrite
    # (configs/README.md). Inline runs are ordered refine-before-dependents, so never stale.
    "refine": ["pointcloud"],
    "semantics": ["pointcloud"],
    # splats: trains on COLMAP poses/points + images/; leaf — nothing reads it yet
    "splats": ["pointcloud"],
    "mesh": ["pointcloud"],
    "localize": ["pointcloud"],
    # verify reuses the localize feature cache but builds it itself when absent, so its
    # only hard dependency is the reconstruction
    "verify": ["pointcloud"],
    # reconstruction_quality_report loads verification.json when verify has produced it and
    # reports the epipolar channel unavailable when it has not, so its only hard dependency is
    # the reconstruction
    "reconstruction_quality_report": ["pointcloud"],
}
# A stage is re-runnable on its own iff nothing depends on it → {refine, semantics, splats, mesh,
# localize, verify, reconstruction_quality_report}.
# Derived from the graph above rather than hardcoded: a future stage that depends on mesh drops
# mesh from this set automatically, so callers gating on it can never disagree with _STAGE_DEPS.
LEAF_STAGES = frozenset(s for s in _STAGE_ORDER if not any(s in deps for deps in _STAGE_DEPS.values()))


########################################
# Helpers
########################################


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
        )
    elif frame_selection == "uniform":
        frame_arrays, records = sample_uniform(
            str(input_path),
            max_frames=max_frames,
            report=report,
        )
    elif frame_selection == "optical_flow":
        frame_arrays, records = sample_optical_flow(str(input_path), max_frames=max_frames, report=report)
    else:
        raise ValueError(f"preproc.frame_selection must be 'fps', 'uniform' or 'optical_flow', got {frame_selection!r}")

    prov = {
        "video_path": str(input_path),
        "video_mtime": input_path.stat().st_mtime,
        "method": frame_selection,
        "fps": fps,
        "max_frames": max_frames,
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
) -> int:
    """
    Extract frames from video or image dir into images/ (sole persistent store).

    Two steps for video input: measure the whole video into
    video_quality_report.json, then select from it. An image directory takes
    every image and needs no report. images/frame_NNNNNN.png plus frames.json
    beside it is the canonical decode-once keyframe store. Returns the number
    of frames stored.

    - undistort=True self-calibrates one shared OPENCV camera from the written frames
      and rewrites them undistorted (COLMAP's framing grows the canvas, so frame dims
      change; both cameras are stamped into provenance["undistort"]).
    """
    input_path = Path(input_path)
    report = None

    if input_path.is_dir():
        frame_arrays, records, prov = _frames_from_dir(input_path, max_frames=max_frames)
    else:
        # Fail loud on an unreadable/empty video — 0 total frames means a bad path or a
        # codec ffmpeg can't decode, which otherwise silently yields an empty store.
        total_frames = get_video_info(str(input_path))["total_frames"]
        if total_frames == 0:
            raise ValueError(
                f"No frames decoded from {input_path} (0 total frames). "
                "Check the path exists and is a video ffmpeg can read."
            )

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
    use_multiview_confidence: bool,
    # Keyword-only: these two are optional and order-independent, so a positional caller
    # must not be able to silently bind one to the other.
    *,
    max_frames: int | None = None,
    creator_kwargs: dict[str, Any] | None = None,
) -> tuple[PointcloudResult, "Viewer | None"]:
    """Instantiate feedforward creator, optionally wrap with LoopClosure, run reconstruct.

    Saves pointcloud.zarr to output_dir after inference so downstream stages
    (semantics lift, mesh) can load depth/confidence/pixel data. Returns the
    PointcloudResult and the created Viewer (None unless loop_closure + viz_enabled),
    so callers can keep the viser server reachable after this function returns.

    ``loop_closure`` is either a bool (enable with all LoopClosureConfig defaults) or
    a dict of knobs (submap_size, submap_overlap, scale_method, …); an ``enabled`` key
    in the dict toggles it, defaulting to True when any knobs are given.

    ``creator_kwargs`` is the per-backend ``pointcloud.<backend>`` config block, forwarded
    verbatim to the creator's constructor. ``max_points`` and ``use_multiview_confidence``
    are reserved — both are passed explicitly, so redeclaring either raises ValueError.

    ``max_frames`` is ``preproc.max_frames``, used only to decide whether the LoGeR
    frame-count advisory fires. None means no ceiling was configured, so no advice is due.
    """
    # Heavy dep imports — kept inline so module loads without GPU/model deps
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )
    from collab_splats.pointcloud.feedforward import (
        LoGeRCreator,
        MapAnythingCreator,
        VGGTOmegaCreator,
        VGGTXCreator,
    )

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

    # LoGeR refuses loop closure in this cut. Refuse here rather than in the creator:
    # _verify_loop_candidate is concrete on BaseFeedforwardCreator, so an LC run would
    # otherwise complete a full forward pass before dying inside the LC loop. LC verify
    # thresholds are calibrated per backbone and none exists for LoGeR. Read the normalised
    # lc_enabled, not the raw arg — loop_closure={"enabled": False} is a truthy object with
    # falsy intent, and refusing an explicit disable would be wrong.
    #
    # Keep this ahead of any filesystem read, and do NOT merge it into the advisory block
    # below: validate config before touching the filesystem. A config error is the user's to
    # fix, an IO error is environmental, and reporting the environmental one first sends them
    # to the wrong place. Reachable with --stages pointcloud when preproc has not run.
    if backend == "loger" and lc_enabled:
        raise ValueError(
            "pointcloud.loop_closure is not supported with backend 'loger'. LoGeR's windowed "
            "TTT memory already carries state across frames, and loop closure verification "
            "thresholds are calibrated per backbone. Use vggt_omega, vggtx, or mapanything."
        )

    # preproc.max_frames is a VGGT-Omega GPU property applied in the preproc stage,
    # which has already run by the time we get here. Flipping to loger under that same
    # ceiling therefore processes exactly as many frames as Omega would, and LoGeR appears
    # to buy nothing. Warn rather than change behaviour — LoGeR's true ceiling is unmeasured.
    # None means no ceiling was configured, so there is no advice to give. Unlike the refusal
    # above, this one reads the store, so it stays below the config validation.
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

    # Select creator class by backend name
    creator_map = {
        "vggtx": VGGTXCreator,
        "mapanything": MapAnythingCreator,
        "vggt_omega": VGGTOmegaCreator,
        "loger": LoGeRCreator,
    }
    # max_points caps the confidence mask during inference — a memory guard, not a preference.
    # use_multiview_confidence is the only mv knob exposed: rel_thresh and min_views stay as
    # calibrated creator field defaults so nobody hand-tunes bare floats in YAML.
    # Both are passed explicitly, so a duplicate in the per-backend config block would surface
    # as an opaque TypeError naming neither the key nor its config path. Reject clashes by
    # name; unknown keys are left to the constructor's own TypeError, which names them.
    # The reserved set is derived from the explicit kwargs rather than restated, so the two
    # cannot drift apart.
    extra = dict(creator_kwargs or {})
    explicit = {"max_points": max_points, "use_multiview_confidence": use_multiview_confidence}
    clash = sorted(extra.keys() & explicit.keys())
    if clash:
        raise ValueError(f"pointcloud.{backend}.{clash[0]} is not settable; use pointcloud.{clash[0]}")
    creator = creator_map[backend](**explicit, **extra)

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

    # Creators read the scene's images/ directory in place — no staging, no temp export;
    # build_colmap appends colmap/sparse/0 under output_dir internally
    result = creator.reconstruct(images_dir, output_dir)

    # Persist FeedforwardResult to pointcloud.zarr — required by semantics lift + mesh stages
    ff_outputs = getattr(creator, "outputs", None)
    if ff_outputs is not None:
        zarr_path = output_dir / "pointcloud.zarr"
        ff_outputs.save_zarr(
            zarr_path,
            extra_attrs={"method": "feedforward", "backend": backend},
        )
        logger.info("pointcloud.zarr saved: %s  (%s pts)", zarr_path, f"{len(ff_outputs.points):,}")
    else:
        logger.warning("Creator has no outputs after reconstruct — pointcloud.zarr not saved")

    # Explicitly release model + GPU memory before next stage (semantics) loads its model
    import torch as _torch

    del creator
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()
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
) -> Path:
    """Load 2D feature cache + FeedforwardResult, lift to 3D, compress, save.

    Writes output_dir/{extractor}_lifted.zarr (latent codes) and, when compressing,
    output_dir/{extractor}_ae.pt (weights + fit metrics) — the pair a consumer needs
    to recover full-dimensionality features.

    target_cosine/max_epochs are required, not defaulted: they are the config's single
    autoencoder policy, and a default here would be a fourth copy of it to drift from.
    """
    # Kept local to match this file's per-stage-method import convention — not because it
    # defers load cost. feedforward.base is already resident by the time this module finishes
    # importing (depth_align pulls it in at module scope), so this local import saves nothing.

    # Validate pointcloud zarr exists before attempting load
    if not pointcloud_zarr.exists():
        raise FileNotFoundError(
            f"pointcloud.zarr not found at {pointcloud_zarr}. "
            "Run build_pointcloud() with a feedforward backend first."
        )

    # Load feature maps from the 2D cache: one (D, H_p, W_p) tensor per frame
    feature_maps = load_feature_maps(zarr_path)

    # Load FeedforwardResult with depth/pixel data for lifting
    ff_result = FeedforwardResult.load_zarr(pointcloud_zarr)

    # Lift 2D features to 3D: (P, D)
    lifted = lift_features(feature_maps, ff_result)

    # Optional autoencoder compression → latent codes persisted alongside the weights
    ae = None
    if n_components is not None:
        # Move lifted to GPU for autoencoder training; lift_features returns CPU tensor
        if torch.cuda.is_available():
            lifted = lifted.cuda()
        ae = FeatureAutoencoder(input_dim=lifted.shape[-1], latent_dim=n_components)
        ae.fit(lifted, epochs=max_epochs, target_cosine=target_cosine)
        lifted = ae.per_point_encode(lifted)

    # One writer for both halves of the pair — it also stamps input_dim/latent_dim on the
    # zarr attrs, which is what tells a reader whether the weights are required at all
    # (n_components=None writes full-dim codes and no weights, legitimately).
    write_point_features(output_dir, extractor_name, lifted.detach().cpu().numpy(), ae)
    return output_dir


def _run_tsdf_mesh(
    result: PointcloudResult,
    pointcloud_zarr: Path,
    output_dir: Path,
    images_dir: Path,
    voxel_size: float,
    depth_trunc: float,
    sdf_trunc_mult: float = 4.0,
    bands: list[dict] | None = None,
    conf_percentile: float | None = None,
    mask_sky: bool = False,
    source: str = "feedforward",
    splats_ckpt: Path | None = None,
    texture: bool = False,
) -> Path:
    """
    Fuse depth and RGB into a TSDF mesh, clean it, and optionally texture it.

    Args:
        result: PointcloudResult; supplies COLMAP poses and original-res K on the feedforward path.
        pointcloud_zarr: the scene's pointcloud.zarr; read on the feedforward path only.
        output_dir: receives mesh.ply, and texture/ when texture is set.
        images_dir: the scene's images/ directory of original-resolution keyframes.
        voxel_size: TSDF voxel edge, world units; ignored when bands is set.
        depth_trunc: ignore depth beyond this, world units; ignored when bands is set.
        sdf_trunc_mult: truncation band as a multiple of voxel_size; sets the thin-structure floor.
        bands: list of {depth_min, depth_trunc, voxel_size} to fuse per band and merge (None = one volume).
        conf_percentile: drop depth below this confidence percentile (None = off); feedforward only.
        mask_sky: zero out depth where the sky segmenter fires; applies to both sources.
        source: "feedforward" (zarr depth lifted to frame resolution) or "splats" (checkpoint renders).
        splats_ckpt: the splats stage's ckpt.pt; required when source is "splats".
        texture: also decimate, unwrap and project the fused views into output_dir/texture/.
    Returns:
        Path to output_dir/mesh.ply.
    """
    # Splats source: renders come out at frame resolution carrying the poses they were rendered
    # with, pose-opt deltas included, so nothing here has to be lifted or re-posed.
    if source == "splats":
        depths, rgbs, c2w, intrinsics, image_ids = render_tsdf_inputs(splats_ckpt, images_dir)
    else:
        ff = FeedforwardResult.load_zarr(pointcloud_zarr, load_images=False, load_world_points=False)
        if ff.depth is None:
            raise ValueError(f"{pointcloud_zarr} has no depth — cannot mesh.")
        if result.extrinsics.shape[0] != ff.depth.shape[0]:
            raise ValueError(
                f"Frame-count mismatch: COLMAP reconstruction has {result.extrinsics.shape[0]} "
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

        # Lift model-res depth onto the original frame grid so COLMAP's original-res K is the
        # right one to fuse with. Pairing one grid's depth with the other grid's K is the
        # 2026-08-11 collapse bug (5.06M -> 75k vertices).
        rgbs = frames.read_frames(images_dir)
        depths = upsample_depths(depth, rgbs, np.asarray(ff.original_coords)[:, :4])
        c2w = invert_poses(result.extrinsics)
        intrinsics = result.intrinsics

        # read_frames above took filename order, which sky_masks defaults to
        image_ids = None

    # Sky fuses as a backdrop and seeds floaters; drop its depth after the arms converge
    # - idxs follows the arm: the checkpoint's ids for splats, filename order for feedforward
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

    # Banded fusion runs one volume per depth range so the near field can be finer than the
    # far field; the scalar voxel_size/depth_trunc are the single-volume case of the same thing
    if bands:
        mesh_path = fuse_tsdf_bands(
            depths,
            rgbs,
            c2w,
            intrinsics,
            output_dir,
            bands=bands,
            sdf_trunc_mult=sdf_trunc_mult,
        )
        texture_voxel = min(band["voxel_size"] for band in bands)
    else:
        mesh_path = fuse_tsdf(
            depths,
            rgbs,
            c2w,
            intrinsics,
            output_dir,
            voxel_size=voxel_size,
            depth_trunc=depth_trunc,
            sdf_trunc=sdf_trunc_mult * voxel_size,
        )
        texture_voxel = voxel_size

    clean_repair_mesh(mesh_path)
    if texture:
        texture_mesh(mesh_path, output_dir / "texture", rgbs, c2w, intrinsics, voxel_size=texture_voxel)
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

    Loads the FeedforwardResult, runs the local matcher over every DB frame, and persists
    keypoints/descriptors to group local_features/{extractor_name}/reconstruction. top_k
    is the pairwise (vismatch) matching fan-out; the descriptor path ignores it.
    """
    # Heavy deps kept inline so the module imports without GPU/model libs
    from collab_splats.localization.extractors import LocalMatcher
    from collab_splats.localization.localizer import CameraLocalizer

    # overwrite: drop the stale reconstruction group so from_feedforward's cache check
    # misses and the index is re-extracted + re-saved (from_feedforward has no
    # overwrite notion of its own — an existing group always cache-hits).
    if overwrite:
        store = zarr.open_group(str(pointcloud_zarr), mode="a")
        rec_key = f"local_features/{extractor_name}/reconstruction"
        if rec_key in store:
            del store[rec_key]

    ff = FeedforwardResult.load_zarr(pointcloud_zarr, load_images=True, load_world_points=True)
    extractor = LocalMatcher(extractor_name)

    # Boundary adapter: images/ directory → (images, ids) core objects. Lazy genexpr → zero
    # reads on a cache hit; one imread per frame on a miss. frame_paths fixes the order, and
    # both lists are built from that one call so they stay index-aligned.
    paths = frames.frame_paths(images_dir)
    frame_indices = [frames.frame_idx_from_path(p) for p in paths]
    images = (cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths)

    # The .jpg suffix is deliberate and stays even though the store writes .png. These ids are
    # opaque labels: CameraLocalizer stores them verbatim in the zarr attrs, every consumer
    # joins on the frame_NNNNNN stem (verify() here, geometry/metrics.py), and nothing reads
    # the extension or resolves an id to a file. Rewriting it to .png would make
    # from_feedforward's whole-string staleness check miss against every localization DB
    # already on disk and under environments-processed/ — a migration, not a rename.
    ids = [f"frame_{int(fi):06d}.jpg" for fi in frame_indices]
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

        # SfM path: BA re-refinement is InstantSfM's own job — refuse the flag
        if method == "sfm" and pc.get("bundle_adjustment"):
            raise ValueError(
                "pointcloud.bundle_adjustment is not supported with method: sfm — "
                "InstantSfM runs its own global bundle adjustment"
            )

        # SfM path: LC wraps a feedforward creator in sequential submaps — nothing to wrap here
        if method == "sfm" and lc_enabled:
            raise ValueError(
                "pointcloud.loop_closure is not supported with method: sfm — "
                "InstantSfM is a global mapper, not a sequential submap pipeline"
            )

        # InstantSfM sub-block bounds check
        if method == "sfm" and backend == "instantsfm":
            instantsfm = pc.get("instantsfm", {})

            # random_seed is consumed long after the run starts — InstantSfM reads it at
            # _build_config, after the SIFT + exhaustive-matching pass — so a bad value is
            # rejected here, where a typo costs a config load and not a whole run. The
            # bound is np.random.seed's domain; InstantSfM passes the value through
            random_seed = instantsfm.get("random_seed")
            if random_seed is not None and not (isinstance(random_seed, int) and 0 <= random_seed < 2**32):
                raise ValueError(
                    f"pointcloud.instantsfm.random_seed must be null or an int in [0, 2**32), got {random_seed!r}"
                )

            # min_num_view_per_track is read at the same late point, and a track needs two
            # views to triangulate at all — a sub-2 value cannot produce geometry
            min_views = instantsfm.get("min_num_view_per_track")
            if min_views is not None and not (isinstance(min_views, int) and min_views >= 2):
                raise ValueError(
                    f"pointcloud.instantsfm.min_num_view_per_track must be null or an int >= 2, got {min_views!r}"
                )

        # Mesh truncation band: a band narrower than a voxel leaves gaps between adjacent
        # voxels' zero crossings, so the extracted surface is punctured rather than thin
        sdf_mult = config.get("mesh", {}).get("sdf_trunc_mult")
        if isinstance(sdf_mult, bool) or not isinstance(sdf_mult, (int, float)) or sdf_mult < 1.0:
            raise ValueError(f"mesh.sdf_trunc_mult must be a number >= 1.0, got {sdf_mult!r}")

        # Mesh bands: null keeps the single volume built from voxel_size/depth_trunc. A list has
        # to partition depth — a gap loses the geometry inside it, an overlap double-surfaces it
        bands = config.get("mesh", {}).get("bands")
        if bands is not None:
            try:
                check_bands(bands)
            except ValueError as exc:
                raise ValueError(f"mesh.bands invalid: {exc}") from exc

        return config

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

        # Skip if COLMAP + pointcloud.zarr both exist and overwrite not requested.
        # Require pointcloud.zarr too — if a previous run was partial (zarr missing),
        # we must re-run inference rather than loading stale COLMAP.
        if not overwrite and self._stage_output_exists("pointcloud"):
            logger.info("Pointcloud exists at %s, loading from disk", self.backend_dir / "colmap")
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
                use_multiview_confidence=pc_cfg["use_multiview_confidence"],
                max_frames=self.config["preproc"]["max_frames"],
                creator_kwargs=pc_cfg.get(pc_cfg["backend"], {}),
            )
            self.viewer = viewer

        # Statistical outlier removal on the final sparse set. Rejected point3D IDs are deleted
        # from the reconstruction in place, so the re-exported PLY matches the model.
        # (pointcloud.zarr was already written from the dense FeedforwardResult and is unaffected.)
        if pc_cfg["clean"]["enabled"] and result.reconstruction.points3D:
            point3d_ids = list(result.reconstruction.points3D.keys())
            keep = clean_pointcloud(result.points)
            for pid, keep_this in zip(point3d_ids, keep, strict=True):
                if not keep_this:
                    result.reconstruction.delete_point3D(pid)
            logger.info("Pointcloud after cleaning: %d points", result.reconstruction.num_points3D())

        # Re-export the PLY from the FINAL result — clean may have dropped points since
        # the creator wrote its own copy.
        result.write_ply(self.backend_dir / "sparse_pc.ply")

        self.pointcloud = result
        return result

    def _load_pointcloud_from_disk(self) -> PointcloudResult:
        """
        Load the written COLMAP model into a PointcloudResult in images/ order.
        """
        # Rebuild image_paths from images/ in filename order, so it lines up with the per-frame
        # arrays the downstream stages index. The creators register COLMAP images as
        # frame_{source_idx:06d} with NO extension — the frame_*.jpg spelling elsewhere is the
        # localization id namespace, not this one.
        frame_indices = [frames.frame_idx_from_path(p) for p in frames.frame_paths(self.images_dir)]
        image_paths = [Path(f"frame_{int(fi):06d}") for fi in frame_indices]
        return PointcloudResult.from_colmap(self.backend_dir / "colmap", image_paths)

    def _run_sfm(self) -> PointcloudResult:
        """
        SfM pointcloud path: staged keyframes -> VDA metric depth -> InstantSfM global mapping.

        - Points InstantSfM at the scene's own images/ directory — nothing is staged.
        - Generates depth_vda/images/npy/<stem>.npy (skipped when present), runs
          InstantSfMCreator, then builds the dense result already rescaled to the COLMAP world.
        - Returns the PointcloudResult for the shared tail; writes pointcloud.zarr with the
          alignment and version provenance on the way.
        """
        pc_cfg = self.config["pointcloud"]
        backend = pc_cfg["backend"]

        # Unreachable through a constructed Reconstructor — validate_config rejects any sfm
        # backend outside _SFM_BACKENDS at config load. Kept as defence in depth for direct
        # _run_sfm calls and for whoever wires ColmapCreator/HlocCreator up later.
        if backend != "instantsfm":
            raise NotImplementedError(f"sfm backend {backend!r} is not implemented — only 'instantsfm' is")
        backend_dir = self.backend_dir
        backend_dir.mkdir(parents=True, exist_ok=True)

        # The scene's images/ is already the COLMAP image layout InstantSfM wants, so it is read
        # in place — no JPEG copy is staged, no SIFT database is dropped on a re-stage (the DB
        # keys itself on the image set now), and COLMAP sees the lossless PNGs the store holds.
        # `names` are those filenames, taken from the directory rather than reconstructed from a
        # hardcoded extension; only their stems reach VDA (depth_vda/images/npy/<stem>.npy).
        images_dir = self.images_dir
        names = [p.name for p in frames.frame_paths(images_dir)]

        # VDA metric depth for every keyframe, cached across runs by stem. On a gate miss drop
        # depth_vda/ first — generate_vda_depth only ever adds maps, so a stem left over from a
        # different keyframe set would keep the set-equality gate false forever and re-run the
        # full GPU inference on every subsequent run.
        if not vda_depth_complete(backend_dir, names):
            shutil.rmtree(backend_dir / "depth_vda", ignore_errors=True)

        # The full-res keyframe stack is built twice on purpose, and freed in between. It is
        # ~1.87 GB at 300 frames of 1080p and ~5.4 GB at the 875-frame configuration this repo has
        # run, against a 46.6 GB cgroup cap — holding it live across creator.reconstruct() would
        # stack that on top of InstantSfM's own peak, in the phase this pipeline has previously
        # been OOM-killed in. Dropping it here means the two peaks never overlap; the second
        # decode after the solve is the deliberate cost of that. On a complete depth cache
        # generate_vda_depth runs no inference and touches this stack only for its len(), so the
        # first build is wasted there (~3-4 s at 875 frames); skipping it needs generate_vda_depth
        # to accept frames=None, a signature change deliberately left out of this commit.
        keyframes = np.ascontiguousarray(frames.read_frames(images_dir))
        depths = generate_vda_depth(keyframes, backend_dir, names)
        del keyframes

        # Global SfM via the upstream python API; writes colmap/instantsfm.db + colmap/sparse/0
        # and returns a model whose image names are the frame_NNNNNN stems
        creator = InstantSfMCreator(
            retriangulation=pc_cfg["instantsfm"]["retriangulation"],
            random_seed=pc_cfg["instantsfm"]["random_seed"],
            min_num_view_per_track=pc_cfg["instantsfm"]["min_num_view_per_track"],
        )
        recon = creator.reconstruct(backend_dir, images_dir=images_dir)
        del creator
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Re-read rather than held: the solve's peak is gone by here, so this is the cheapest
        # point to pay the decode back.
        keyframes = np.ascontiguousarray(frames.read_frames(images_dir))

        # Dense result at VDA depth resolution, already rescaled into the COLMAP world — the zarr
        # and the model must share one scale (splat depth targets, mesh fusion, localization)
        outputs, align_attrs = result_from_reconstruction(recon, depths, keyframes, names)

        zarr_path = backend_dir / "pointcloud.zarr"
        outputs.save_zarr(
            zarr_path,
            extra_attrs={
                "method": "sfm",
                "backend": "instantsfm",
                "instantsfm_version": importlib.metadata.version("instantsfm"),
                **align_attrs,
            },
        )
        logger.info("pointcloud.zarr saved: %s  (%s pts)", zarr_path, f"{len(outputs.points):,}")

        return PointcloudResult(reconstruction=recon, image_paths=outputs.image_paths)

    def refine_poses(self, overwrite: bool = False) -> PointcloudResult:
        """Refine camera poses via LM bundle adjustment; rewrite pose-derived artifacts.

        One implementation for both triggers: runs inline after the pointcloud stage when
        pointcloud.bundle_adjustment is enabled, and from disk via --stages refine against
        a processed scene. Loads everything from pointcloud.zarr — no live creator needed.
        """
        # SfM results are already globally bundle-adjusted; LM re-refinement is undefined here
        if self.config["pointcloud"]["method"] == "sfm":
            raise ValueError("refine_poses is not supported for pointcloud.method: sfm")

        # Heavy deps imported lazily, matching the other stage methods
        from collab_splats.geometry.bundle_adjustment import (
            BundleAdjustment,
            BundleAdjustmentConfig,
        )
        from collab_splats.pointcloud.feedforward.base import (
            FeedforwardResult,
            _rescale_reconstruction_to_original_dimensions,
            build_pycolmap_reconstruction,
        )

        # Skip when already refined — run_pipeline refuses NAMED re-runs generically, so this
        # mirrors the other stage methods' silent skip for config-driven repeat runs.
        marker = self.backend_dir / "colmap" / "refine.json"
        if marker.exists() and not overwrite:
            logger.info("Poses already refined (%s), skipping refine", marker)
            return self._resolve_result()

        # Load the full FeedforwardResult from zarr: images/confidence/world_points feed track
        # extraction, depth+pixel_indices feed the deterministic creator-free reproject.
        zarr_path = self.pointcloud_zarr
        if not zarr_path.exists():
            raise FileNotFoundError(f"refine requires {zarr_path}; run the pointcloud stage first.")
        ff = FeedforwardResult.load_zarr(zarr_path, load_images=True)

        # Refine poses with LM BA, then re-derive the point set under the new cameras
        cfg = BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir)
        ba = BundleAdjustment(cfg)
        ff = ba.refine(ff).reproject()

        # Rewrite COLMAP through the creators' exact write path: build at model res from the
        # refined result, then rescale K + image dims back to original resolution.
        recon = build_pycolmap_reconstruction(
            ff.points,
            ff.colors,
            ff.extrinsics,
            ff.intrinsics,
            ff.model_width,
            ff.model_height,
            [p.name for p in ff.image_paths],
        )
        recon = _rescale_reconstruction_to_original_dimensions(
            recon, ff.image_paths, ff.original_coords, (ff.model_width, ff.model_height)
        )
        sparse_dir = self.backend_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        recon.write_binary(str(sparse_dir))

        # Write pose-derived arrays back to pointcloud.zarr so zarr and COLMAP never disagree
        # (localization samples world_points; a later --stages refine re-reads these poses).
        store = zarr.open(str(zarr_path), mode="r+")
        store["extrinsics"][:] = ff.extrinsics
        store["intrinsics"][:] = ff.intrinsics
        store["points"][:] = ff.points
        if "world_points" in store and ff.depth is not None:
            wp = unproject_depth_map_to_point_map(ff.depth[..., None], ff.extrinsics[:, :3, :], ff.intrinsics).astype(
                np.float32
            )
            store["world_points"][:] = wp

        # Reload the refined model from disk — it is both the returned result and the source
        # for the refreshed sparse_pc.ply
        result = self._load_pointcloud_from_disk()
        result.write_ply(self.backend_dir / "sparse_pc.ply")

        # Marker + provenance in one file: BA config and per-step LM loss history
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "config": {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()},
                    "loss_history": ba._last_loss_history,
                    "n_frames": len(ff.image_paths),
                },
                indent=2,
            )
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
        )
        return out_dir

    def mesh(
        self,
        result: PointcloudResult | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Build a TSDF mesh from `mesh.source` depth: pointcloud.zarr (default) or splats ckpt.pt renders.

        COLMAP is the pose authority on the feedforward path; the splats path fuses the
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
        elif self.config["pointcloud"]["method"] == "sfm":
            # SfM scenes fuse only after zarr depth was aligned to the COLMAP world
            # (depth_scale attr). A legacy VDA-metric store against COLMAP poses produced
            # geometry at the wrong scale in the wrong places (measured 3.2x on GH010229).
            attrs = zarr.open(str(pointcloud_zarr), mode="r").attrs
            if "depth_scale" not in attrs:
                raise ValueError(
                    f"{pointcloud_zarr} predates depth alignment (no depth_scale attr) — "
                    "re-run the pointcloud stage to align VDA depth to the COLMAP world, "
                    "or set mesh.source: splats."
                )

        out = _run_tsdf_mesh(
            result=result,
            pointcloud_zarr=pointcloud_zarr,
            output_dir=self.backend_dir,
            images_dir=self.images_dir,
            voxel_size=mesh_cfg["voxel_size"],
            depth_trunc=mesh_cfg["depth_trunc"],
            sdf_trunc_mult=mesh_cfg["sdf_trunc_mult"],
            bands=mesh_cfg["bands"],
            conf_percentile=mesh_cfg["conf_percentile"],
            mask_sky=mesh_cfg["mask_sky"],
            source=source,
            splats_ckpt=splats_ckpt,
            texture=mesh_cfg["texture"],
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

    def verify(self, overwrite: bool = False) -> Path:
        """Geometrically verify poses/points: pycolmap triangulation over feature tracks.

        Reuses the localization extractor's zarr feature cache (building it if absent) and
        writes colmap/{verified/, verification.json, database.db}. Reports only — nothing
        upstream is mutated.
        """
        out_json = self.backend_dir / "colmap" / "verification.json"
        if not overwrite and self._stage_output_exists("verify"):
            logger.info("Verification exists at %s, skipping", out_json)
            return out_json

        result = self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # One extractor serves localization and verification by design — the cache is
        # keyed by extractor, so sharing it means one extraction pass, zero drift.
        t = time.perf_counter()
        self.build_localization_db()
        logger.info("verify(): build_localization_db took %.1f s", time.perf_counter() - t)
        extractor_name = self.config["localization"]["matcher"]

        # Heavy deps kept inline so the module imports without GPU/model libs
        from collab_splats.geometry.verification import verify_reconstruction
        from collab_splats.localization.extractors import LocalMatcher
        from collab_splats.localization.localizer import load_localization_db

        t = time.perf_counter()
        matcher = LocalMatcher(extractor_name)
        logger.info("verify(): LocalMatcher construction took %.1f s", time.perf_counter() - t)
        t = time.perf_counter()
        features, ids, _ = load_localization_db(self.pointcloud_zarr, extractor_name)
        logger.info("verify(): load_localization_db took %.1f s", time.perf_counter() - t)
        # Loma matches from stored features (keypoints_normalized). A cache from before
        # that array existed gets rebuilt once (~1 min) — no degraded fallback path.
        # getattr: duck-typed test stubs and non-split matchers lack the attribute.
        if getattr(matcher, "_split_loma_forward", False) and any(f.keypoints_normalized is None for f in features):
            logger.info("verify(): feature cache lacks keypoints_normalized — rebuilding localization DB")
            t = time.perf_counter()
            self.build_localization_db(overwrite=True)
            logger.info("verify(): build_localization_db(overwrite=True) took %.1f s", time.perf_counter() - t)
            t = time.perf_counter()
            features, ids, _ = load_localization_db(self.pointcloud_zarr, extractor_name)
            logger.info("verify(): load_localization_db (rebuilt) took %.1f s", time.perf_counter() - t)
        # The cache ids are frame_XXXXXX.jpg, the reconstruction registers frame_XXXXXX
        # (no extension) — compare stems so a reordered/rebuilt cache cannot slip through.
        recon = result.reconstruction
        recon_names = [recon.images[i].name for i in sorted(recon.images)]
        if [Path(n).stem for n in ids] != [Path(n).stem for n in recon_names]:
            raise ValueError(
                "Feature cache and reconstruction disagree on frame order/naming — "
                "rebuild the localization DB (overwrite=True)."
            )
        # Pairwise matchers re-match images, not cached descriptors. The images MUST be
        # the exact frames the cache was extracted from — _build_localization_db feeds
        # images/ frames to extract() — because match-time index recovery lands on the
        # extract-time keypoint tables (the probe's cross-call condition holds for
        # identical inputs only), and those tables are what verification exports to the
        # COLMAP DB. Model-res ff.images would index a different table entirely.
        # verify_reconstruction itself hard-refuses index-incapable pairwise matchers.
        # _scene_frames caches the stack per directory, so a second verify() in the same
        # process re-reads nothing. (The descriptor branch in verify_reconstruction
        # ignores `images`, so the read is wasted work only on that path.)
        images = _scene_frames(self.images_dir)
        # Sequential pairs only in v1 (pycolmap SequentialPairGenerator inside).
        # Loop pairs are a follow-on: COLMAP's own loop_detection needs a SIFT vocab
        # tree (unusable with learned descriptors) and retrieval descriptors are not
        # cached anywhere a processed scene guarantees.
        verify_reconstruction(
            recon=recon,
            features=features,
            matcher=matcher,
            output_dir=self.backend_dir / "colmap",
            images=images,
        )
        logger.info("Verification written to %s", out_json)
        return out_json

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

        # Frames in COLMAP image order, looked up in images/ by the frame index in each name.
        # _scene_frames caches the whole-directory stack, so a verify() earlier in the same
        # process re-reads nothing; rows are then picked by position out of that stack.
        frame_indices = [frames.frame_idx_from_path(path) for path in result.image_paths]
        rows_by_frame_idx = {
            frames.frame_idx_from_path(p): row for row, p in enumerate(frames.frame_paths(self.images_dir))
        }
        unknown = [fi for fi in frame_indices if fi not in rows_by_frame_idx]
        if unknown:
            raise KeyError(
                f"{len(unknown)} reconstruction frames are not in {self.images_dir} "
                f"(frame_idx {unknown[:5]}); the images/ store and the reconstruction describe "
                "different runs."
            )
        # CPU-resident by design: train() moves one view to the GPU at a time
        images = _scene_frames(self.images_dir)[[rows_by_frame_idx[fi] for fi in frame_indices]]

        # Depth targets: model-res pointcloud.zarr depth masked like the mesh stage masks it (0 = no
        # target); train() resizes each view to the frame's resolution with nearest sampling
        depth_targets = None
        depth_on = "depth" in cfg.losses and cfg.losses["depth"]["weight"] > 0  # from_dict guarantees weight
        if depth_on:
            pointcloud_zarr = self.pointcloud_zarr
            if not pointcloud_zarr.exists():
                raise FileNotFoundError(
                    f"pointcloud.zarr not found at {pointcloud_zarr}. "
                    "Splats depth loss requires depth maps from the pointcloud stage."
                )

            # SfM scenes: zarr depth is usable only once aligned to the COLMAP world — a
            # legacy VDA-metric store fed as targets collapsed training (measured PSNR 6.15)
            if self.config["pointcloud"]["method"] == "sfm":
                attrs = zarr.open(str(pointcloud_zarr), mode="r").attrs
                if "depth_scale" not in attrs:
                    raise ValueError(
                        f"{pointcloud_zarr} predates depth alignment (no depth_scale attr) — "
                        "re-run the pointcloud stage to align VDA depth to the COLMAP world."
                    )

            feedforward = FeedforwardResult.load_zarr(pointcloud_zarr, load_images=False, load_world_points=False)
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
                confidence = feedforward.confidence.cpu().numpy()[rows]
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
        """Reference-free error report: three measurements, one reconstruction_quality_report.json.

        Named for the artefact it produces, and to stay distinct from the video quality
        report, which scores capture rather than reconstruction.

        Never fails a reconstruction — a measurement that cannot run records
        {"available": false, "reason": ...} and the rest still emit.

        Not side-effect free. With `pointcloud.geometric_verification` true and no
        verification.json on disk, this runs the verify stage, which writes
        colmap/verification.json, colmap/verified/, colmap/database.db and populates
        local_features in the zarr. With the flag false — the default — nothing outside
        reconstruction_quality_report.json is written and the call is cheap: no model, no
        matcher, one zarr read.
        """
        out_json = self.backend_dir / "reconstruction_quality_report.json"
        if not overwrite and self._stage_output_exists("reconstruction_quality_report"):
            logger.info("Reconstruction quality report exists at %s, skipping", out_json)
            return out_json
        if self._resolve_result() is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # The epipolar rows are loaded when verify has produced them, and reported unavailable
        # when it has not. Building them here regardless would reach around an explicit
        # `geometric_verification: false` and charge every default run verify's cost (measured
        # 47.6 min and +6.29 GB RSS at 300 frames) for a stage that is always on. Degrading to
        # {"available": false, "reason": ...} is the report-only outcome, not a failure.
        verification_json = self.backend_dir / "colmap" / "verification.json"
        if self.config["pointcloud"]["geometric_verification"] and not verification_json.exists():
            try:
                self.verify()
            except Exception:  # noqa: BLE001 — a report must never fail a reconstruction
                logger.warning("verify failed; epipolar rows will be unavailable", exc_info=True)

        # Heavy deps inline so the module imports without GPU/model libs
        from collab_splats.geometry.metrics import build_reconstruction_quality_report

        build_reconstruction_quality_report(
            zarr_path=self.pointcloud_zarr,
            verification_json=verification_json,
            images_dir=self.images_dir,
            output_path=out_json,
            backend=self.config["pointcloud"]["backend"],
        )
        logger.info("Reconstruction quality report written to %s", out_json)
        return out_json

    def _stage_output_exists(self, stage: str) -> bool:
        """True if `stage`'s on-disk output is already present (lets deps be reused across runs)."""
        if stage == "preproc":
            return self.images_dir.exists()
        if stage == "pointcloud":
            colmap_done = (self.backend_dir / "colmap" / "sparse" / "0" / "cameras.bin").exists()
            zarr_done = self.pointcloud_zarr.exists()
            return colmap_done and zarr_done
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
        if stage == "verify":
            return (self.backend_dir / "colmap" / "verification.json").exists()
        if stage == "reconstruction_quality_report":
            return (self.backend_dir / "reconstruction_quality_report.json").exists()
        return False

    def _resolve_result(self) -> PointcloudResult | None:
        """PointcloudResult for a stage-2+ run, loading from COLMAP on disk if not in memory."""
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
                    "localize", "verify", "reconstruction_quality_report"].
                    Default: all enabled stages from config.
            overwrite: Re-run stages even if output exists.

        Raises:
            ValueError: If stages list violates dependency ordering, or if a named LEAF_STAGES
                stage already has output on disk and overwrite is False.
        """
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
            if self.config["pointcloud"]["geometric_verification"]:
                stages.append("verify")
            # Always on, no config boolean. Every other diagnostic ships behind a
            # default-false flag, and the one boolean this would have had is the boolean that
            # keeps it off. Affordable because it runs no model and no matcher — it reads the
            # zarr the reconstruction just wrote — and because verify, when enabled, is appended
            # just above and has already run by then: the report loads its output rather than
            # triggering it. A direct reconstruction_quality_report() call can still run verify;
            # see its docstring.
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
            elif stage == "verify":
                self.verify(overwrite=overwrite)
            elif stage == "reconstruction_quality_report":
                self.reconstruction_quality_report(overwrite=overwrite)

    def launch_dashboard(self) -> None:
        """Launch interactive dashboard for current reconstruction state."""
        from collab_splats.dashboard.__main__ import (
            main as dashboard_main,  # optional heavy dep; lazy load
        )

        dashboard_main()
