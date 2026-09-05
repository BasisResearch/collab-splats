"""5-stage reconstruction pipeline wrapper."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import shutil
import time
import warnings
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
import yaml
import zarr
from mergedeep import merge
from vggt.utils.geometry import unproject_depth_map_to_point_map

from collab_splats.pointcloud.export import write_pointcloud_ply
from collab_splats.pointcloud.sfm import (
    DEPTH_ALIGN_MODELS,
    InstantSfMCreator,
    _pixel_indices_from_reconstruction,
    _tracked_point3d_ids,
    apply_depth_alignment,
    generate_vda_depth,
    vda_depth_complete,
)
from collab_splats.preproc import get_video_info
from collab_splats.preproc import viz as preproc_viz
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.sampling import (
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
from collab_splats.preproc.undistort import (
    DistortionProfile,
    estimate_camera_distortion,
    undistort_frames,
)
from collab_splats.preproc.video import context_indices, decode_context
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import (
    extract_feature_cache,
    lifted_store_path,
    load_feature_maps,
    write_point_features,
)

if TYPE_CHECKING:
    import pycolmap

    from collab_splats.pointcloud.base import PointcloudResult
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# base.yaml is the single source of defaults; __init__ merges any passed config over it.
DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"

_FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega", "loger"}
_SFM_BACKENDS = {"colmap", "hloc", "instantsfm"}
# InstantSfM v0.3.0's DB step ignores the feature-handler name it's given and always runs
# colmap SIFT + exhaustive matching (our _generate_sift_database, GPU when CUDA is available),
# so "colmap" is the only value that means anything today. Key kept (not hardcoded) so a
# future feature handler (e.g. loma) has somewhere to land.
_INSTANTSFM_FEATURES = {"colmap"}
_VALID_METHODS = {"feedforward", "sfm"}
_STAGE_ORDER = ["preproc", "pointcloud", "refine", "semantics", "splats", "mesh", "localize", "verify",
                "reconstruction_quality_report"]
_STAGE_DEPS: dict[str, list[str]] = {
    "preproc": [],
    "pointcloud": ["preproc"],
    # refine rewrites pointcloud outputs in place; deliberately NOT a dependency of the
    # stages below — that would demote them from LEAF_STAGES and break their disk re-run.
    # Staleness contract: after --stages refine, re-run dependents with overwrite
    # (configs/README.md). Inline runs are ordered refine-before-dependents, so never stale.
    "refine": ["pointcloud"],
    "semantics": ["pointcloud"],
    # splats: trains on COLMAP poses/points + frames.zarr; leaf — nothing reads it yet
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


def _apply_undistortion(frame_arrays: list[np.ndarray], prov: dict) -> list[np.ndarray]:
    """
    Self-calibrate + undistort selected frames in place of the raw ones; stamp provenance.
    """
    profile = estimate_camera_distortion(frame_arrays)
    frame_arrays, K_new, roi = undistort_frames(frame_arrays, profile)
    logger.info(
        "undistort: k1=%.4f k2=%.4f p1=%.4f p2=%.4f; alpha=0 crop roi=%s (frames now %dx%d)",
        profile.k1, profile.k2, profile.p1, profile.p2, roi, roi[2], roi[3],
    )
    prov["undistort"] = {
        "profile": profile.to_dict(),
        "K_new": K_new.tolist(),
        "roi": list(roi),
    }
    return frame_arrays


def _context_keep_rows(grid: Sequence[int], keyframe_indices: Sequence[int]) -> list[int] | None:
    """
    Positions of each keyframe within the context grid, or None when they do not line up.

    - Keyframes selected through preproc's candidate grid are grid members by construction;
      a scene whose frames.zarr predates that change is not, and gets the keyframe-only VDA
      path rather than a silently misaligned depth stack.
    """
    # np.unique, matching decode_context's sorted({...}): searchsorted needs a sorted grid, and
    # the two must agree on row order or positions point at the wrong frames
    grid_array = np.unique(np.asarray(grid, dtype=np.int64))
    if grid_array.size == 0:
        logger.warning("VDA context grid is empty — falling back to keyframe-only VDA")
        return None

    # searchsorted clips past-the-end keyframes onto the last member, so the equality check
    # below — not the search — is what decides whether a keyframe is really on the grid
    keyframes = np.asarray(keyframe_indices, dtype=np.int64)
    positions = np.clip(np.searchsorted(grid_array, keyframes), 0, grid_array.size - 1)
    off_grid = grid_array[positions] != keyframes
    if off_grid.any():
        logger.warning(
            "%d of %d keyframes are off the context grid (first: %d) — falling back to "
            "keyframe-only VDA. Re-run preprocess with preproc.vda_context_fps set to align them.",
            int(off_grid.sum()), len(keyframes), int(keyframes[off_grid][0]),
        )
        return None

    return [int(p) for p in positions]


def _video_unchanged(video_path: Path, provenance: dict) -> bool:
    """
    Check that the video on disk is still the one frames.zarr was built from.

    - The context grid is indexed against that specific decode; a re-encoded or replaced clip
      at the same path shifts every index, pairing keyframes with other frames' depth.
    - Both grids start at 0, so a mismatch is not self-announcing — nothing downstream errors.
    - Missing provenance (a store written before the stamp existed) is treated as unchanged.
    """
    recorded_mtime = provenance.get("video_mtime")
    if recorded_mtime is not None and abs(float(recorded_mtime) - video_path.stat().st_mtime) > 1.0:
        logger.warning(
            "%s was modified since frames.zarr was written (mtime %s -> %s)",
            video_path, recorded_mtime, video_path.stat().st_mtime,
        )
        return False

    return True


def extract_frames(
    input_path: Path,
    frames_zarr: Path,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
    n_workers: int = 1,
    undistort: bool = False,
    search_radius: int = 3,
    vda_context_fps: float | None = None,
) -> int:
    """
    Extract frames from video or image dir into frames.zarr (sole persistent store).

    Two steps for video input: measure the whole video into
    video_quality_report.json, then select from it. An image directory takes
    every image and needs no report. frames.zarr is the canonical decode-once
    keyframe store; no JPEG dir is written. Returns the number of frames stored.

    - undistort=True self-calibrates one shared OPENCV camera and undistorts every
      selected frame before writing (alpha=0 crop changes frame dims; profile,
      K_new and roi are stamped into provenance["undistort"]).
    - vda_context_fps restricts every selected frame (target and blur substitute) to the
      constant-rate grid the sfm stage runs VDA over, so the keyframes are grid members by
      construction and their depth rows map back by position. Video input only.
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        # Read source images from directory; reject non-image extensions
        exts = {".jpg", ".jpeg", ".png"}
        frames = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
        if not frames:
            raise ValueError(f"No images ({sorted(exts)}) found in directory {input_path}")
        frame_arrays = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frames]
        records = [{"frame_idx": i, "blur_score": float("nan")} for i in range(len(frame_arrays))]
        prov = {
            "video_path": str(input_path),
            "video_mtime": None,
            "method": "dir",
            "fps": None,
            "max_frames": max_frames,
            "vda_context_fps": None,
        }

        # An image directory has no frame rate to build a grid on, and _run_sfm gates the
        # context stream on a video FILE, so the knob is inert here — say so rather than
        # letting it look honoured
        if vda_context_fps:
            logger.warning(
                "preproc.vda_context_fps=%s ignored for image-directory input %s — "
                "the context stream needs a video to decode",
                vda_context_fps, input_path,
            )
        if undistort:
            frame_arrays = _apply_undistortion(frame_arrays, prov)
        FrameStore.create(frames_zarr, frame_arrays, records, provenance=prov)
        return len(frame_arrays)

    # Config error, not a path error: check the combination before the probe so a bad
    # frame_selection does not surface as an ffprobe failure
    if vda_context_fps and frame_selection == "optical_flow":
        raise ValueError(
            "preproc.vda_context_fps requires frame_selection 'fps' or 'uniform' — "
            "optical_flow picks frames by motion and cannot be restricted to a grid."
        )

    # Fail loud on an unreadable/empty video — 0 total frames means a bad path or a
    # codec ffmpeg can't decode, which otherwise silently yields an empty store.
    total_frames = get_video_info(str(input_path))["total_frames"]
    if total_frames == 0:
        raise ValueError(
            f"No frames decoded from {input_path} (0 total frames). "
            "Check the path exists and is a video ffmpeg can read."
        )

    # Measure before selecting. The report lands beside frames.zarr and is reused by
    # existence, so a re-run never re-measures. Report-only: it carries no verdicts —
    # filter_frame_quality applies the thresholds inside the samplers.
    report = load_video_quality(
        input_path,
        frames_zarr.parent / "video_quality_report.json",
        workers=n_workers,
    )

    # Context grid: when the VDA context stream is enabled the keyframes must be grid
    # members, so the depth rows map back to them by position (see _context_keep_rows).
    # context_indices is range(0, total, step), so the grid always spans the whole video.
    candidates = None
    if vda_context_fps:
        candidates = context_indices(str(input_path), target_fps=vda_context_fps)
        logger.info(
            "VDA context grid: %d frames at %.2f fps; keyframes will be drawn from it",
            len(candidates), vda_context_fps,
        )

    # Video — 'fps' samples at a constant wall-clock rate (band-bounded), 'uniform'
    # spreads exactly max_frames over the whole video, 'optical_flow' picks high-motion
    # frames. Each method gets only its own knobs.
    if frame_selection == "fps":
        frame_arrays, records = sample_fps(
            str(input_path),
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
            report=report,
            search_radius=search_radius,
            candidates=candidates,
        )
    elif frame_selection == "uniform":
        frame_arrays, records = sample_uniform(
            str(input_path), max_frames=max_frames, report=report, search_radius=search_radius,
            candidates=candidates,
        )
    elif frame_selection == "optical_flow":
        frame_arrays, records = sample_optical_flow(str(input_path), max_frames=max_frames, report=report)
    else:
        raise ValueError(f"preproc.frame_selection must be 'fps', 'uniform' or 'optical_flow', got {frame_selection!r}")

    method = frame_selection

    # Every candidate failed the quality filter (or the selector rejected all) — refuse
    # to write an empty store that would only surface as a downstream FileNotFound.
    if not frame_arrays:
        raise ValueError(
            f"0 frames selected from {input_path} ({total_frames} decoded) with "
            f"frame_selection={frame_selection!r}. Every frame failed the quality filter — "
            f"check {frames_zarr.parent / 'video_quality_report.json'} for the measurements."
        )

    # Write the canonical frames.zarr (decode-once keyframe store)
    prov = {
        "video_path": str(input_path),
        "video_mtime": input_path.stat().st_mtime,
        "method": method,
        "fps": fps,
        "max_frames": max_frames,
        "vda_context_fps": vda_context_fps,
    }
    if undistort:
        frame_arrays = _apply_undistortion(frame_arrays, prov)
    FrameStore.create(frames_zarr, frame_arrays, records, provenance=prov)

    # Render the report beside frames.zarr with the kept frames marked. Written
    # whenever frames are, so the PNGs never go stale against the store.
    out_dir = frames_zarr.parent
    selected = [r["frame_idx"] for r in records]
    written = [
        preproc_viz.plot_photometric(report, out_dir, selected=selected),
        preproc_viz.plot_motion(report, out_dir, selected=selected),
    ]
    logger.info("video quality: wrote %d plots to %s", sum(p is not None for p in written), out_dir)

    return len(frame_arrays)


def _run_feedforward(
    backend: str,
    frames_zarr: Path,
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
) -> tuple["PointcloudResult", "Viewer | None"]:
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
    # Keep this ahead of the store open, and do NOT merge it into the advisory block below:
    # validate config before touching the filesystem. A config error is the user's to fix,
    # an IO error is environmental, and reporting the environmental one first sends them to
    # the wrong place. Reachable with --stages pointcloud when preproc has not run.
    if backend == "loger" and lc_enabled:
        raise ValueError(
            "pointcloud.loop_closure is not supported with backend 'loger'. LoGeR's windowed "
            "TTT memory already carries state across frames, and loop closure verification "
            "thresholds are calibrated per backbone. Use vggt_omega, vggtx, or mapanything."
        )

    # Open the canonical decode-once keyframe store once and reuse it: the LoGeR advisory
    # below needs the frame count and inference needs the store itself. The handle is zarr
    # mode="r" — immutable, lazy, and cheap to hold across the span.
    store = FrameStore.open(frames_zarr)

    # preproc.max_frames is a VGGT-Omega GPU property applied in the preproc stage,
    # which has already run by the time we get here. Flipping to loger under that same
    # ceiling therefore processes exactly as many frames as Omega would, and LoGeR appears
    # to buy nothing. Warn rather than change behaviour — LoGeR's true ceiling is unmeasured.
    # None means no ceiling was configured, so there is no advice to give. Unlike the refusal
    # above, this genuinely needs the store, so it belongs after the open.
    if backend == "loger":
        n_frames = len(store)
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

    # Feed inference frames from the canonical decode-once store (temp-exported for path-locked
    # model preprocessing); build_colmap appends colmap/sparse/0 under output_dir internally
    result = creator.reconstruct(store, output_dir)

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


def _rename_images_to_stems(recon: "pycolmap.Reconstruction", sparse_dir: Path) -> None:
    """
    Rename COLMAP images to their filename stems and rewrite the binary model in place.

    - InstantSfM registers images under their filenames (frame_000000.jpg); the pipeline
      contract is frame_{source_idx:06d} with NO extension (see _load_pointcloud_from_disk).
    - pycolmap.Image.name is settable by reference, so the rename lands on the model itself.
    """
    for im in recon.images.values():
        im.name = Path(im.name).stem
    recon.write_binary(str(sparse_dir))


def _get_extractor(name: str):
    """Instantiate feature extractor by registry name."""
    from collab_splats.semantics.features import BaseFeatureExtractor

    return BaseFeatureExtractor.get(name)()


def _extract_2d_features(
    extractor_name: str,
    frames_zarr: Path,
    cache_dir: Path,
) -> Path:
    """Extract 2D features for all frames straight from the canonical store.

    Args:
        extractor_name: registry key of the extractor to run.
        frames_zarr: path to the scene's canonical frames.zarr.
        cache_dir: directory the `<extractor>.zarr` patch cache is written into.

    Returns:
        Path of the written 2D patch cache.
    """
    # extract_feature_cache iterates the frames zarr lazily, one chunk at a time — no temp
    # JPG export, no full-RAM load.
    cache_dir.mkdir(parents=True, exist_ok=True)
    return extract_feature_cache(_get_extractor(extractor_name), frames_zarr, cache_dir)


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
    # Heavy optional stack — pointcloud.utils pulls the feedforward extra
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.pointcloud.utils import lift_features

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
    result: "PointcloudResult",
    pointcloud_zarr: Path,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    clean_repair: bool = False,
    conf_percentile: float | None = None,
    native_resolution: bool = False,
    color_map_iterations: int = 0,
    frames_zarr: Path | None = None,
    source: str = "feedforward",
    splats_zarr: Path | None = None,
    splat_depth: str = "expected",
    splat_max_depth_frac: float | None = None,
    splat_max_depth_grad: float | None = None,
) -> Path:
    """Fuse depth + RGB from pointcloud.zarr (or splats.zarr renders) into a TSDF mesh, using COLMAP poses."""
    from collab_splats.mesh.utils import pointcloud_to_mesh
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    # Splats source: rendered depth/RGB/alpha + the poses actually rendered; no native path
    if source == "splats":
        from collab_splats.mesh.utils import (
            _splats_to_tsdf_inputs,
            mesh_from_tsdf_inputs,
        )

        if native_resolution:
            logger.info(
                "mesh.native_resolution ignored: splats renders are already at frame resolution"
            )
        depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(
            splats_zarr,
            conf_percentile=conf_percentile,
            splat_depth=splat_depth,
            max_depth_frac=splat_max_depth_frac,
            max_depth_grad=splat_max_depth_grad,
        )
        n_views = depths.shape[0]
        n_poses = result.extrinsics.shape[0]
        if n_views != n_poses:
            raise ValueError(
                f"Frame-count mismatch: COLMAP reconstruction has {n_poses} images but "
                f"{splats_zarr} has {n_views}. They are from different runs — re-run the "
                "splats stage, or point --stages mesh at the matching scene."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_result = mesh_from_tsdf_inputs(
            depths,
            rgbs,
            c2w,
            intrinsics,
            output_dir,
            method="open3d_tsdf",
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc,
            clean_repair=clean_repair,
            color_map_iterations=color_map_iterations,
        )
        return mesh_result.mesh_path

    # world_points is the largest array in the store and the mesh path no longer reads it.
    # native_resolution skips the zarr's model-res RGB too — it comes from frames.zarr instead.
    ff = FeedforwardResult.load_zarr(
        pointcloud_zarr, load_images=not native_resolution, load_world_points=False
    )

    # COLMAP is the pose authority — BA and loop-closure corrections land in the reconstruction,
    # not back in the zarr. On the model-res path intrinsics stay the zarr's (build_colmap
    # rescaled COLMAP's camera to original resolution, while the zarr's depth and RGB are at
    # model resolution); the native path swaps in COLMAP's original-res K below.
    if ff.depth is None:
        raise ValueError(f"{pointcloud_zarr} has no depth — cannot mesh.")
    if result.extrinsics.shape[0] != ff.depth.shape[0]:
        raise ValueError(
            f"Frame-count mismatch: COLMAP reconstruction has {result.extrinsics.shape[0]} "
            f"images but {pointcloud_zarr} has {ff.depth.shape[0]}. They are from different "
            "runs — re-run the pointcloud stage, or point --stages mesh at the matching scene."
        )
    ff.extrinsics = result.extrinsics

    # Native path: original-res RGB from frames.zarr, COLMAP's original-res K as intrinsics
    frame_store = None
    native_intrinsics = None
    if native_resolution:
        if frames_zarr is None or not frames_zarr.exists():
            raise FileNotFoundError(
                f"native_resolution requires frames.zarr (looked at {frames_zarr})"
            )
        frame_store = FrameStore.open(frames_zarr)
        native_intrinsics = result.intrinsics  # original-res by contract (build_colmap)

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_result = pointcloud_to_mesh(
        ff,
        output_dir,
        method="open3d_tsdf",
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        clean_repair=clean_repair,
        conf_percentile=conf_percentile,
        frame_store=frame_store,
        native_intrinsics=native_intrinsics,
        color_map_iterations=color_map_iterations,
    )
    return mesh_result.mesh_path


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
    frames_zarr: Path,
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
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.preproc.frame_store import FrameStore

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

    # Boundary adapter: canonical store → (images, ids) core objects. Lazy genexpr → zero
    # reads on a cache hit; one partial-read per frame on a miss.
    store = FrameStore.open(frames_zarr)
    frame_indices = store.frame_indices()
    images = (store.image_by_frame_idx(fi) for fi in frame_indices)
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


class _LazyFrames(Sequence):
    """Lazy len/indexable view over FrameStore frames — never materializes the whole video."""

    def __init__(self, store: FrameStore, frame_indices):
        self._frame_indices = list(frame_indices)
        # Sequential pairing touches each frame ~2x overlap times with strong locality —
        # a small LRU keeps peak memory at a handful of frames, not N full-res images.
        self._get = lru_cache(maxsize=32)(store.image_by_frame_idx)

    def __len__(self) -> int:
        return len(self._frame_indices)

    def __getitem__(self, i: int) -> np.ndarray:
        return self._get(self._frame_indices[i])


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
                "config key 'preprocessing' was renamed to 'preproc' (2026-08-22); "
                "rename the section in your config"
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

        # InstantSfM feature-handler allowlist (v0.3.0 supports only colmap)
        if method == "sfm" and backend == "instantsfm":
            instantsfm = pc.get("instantsfm", {})
            features = instantsfm.get("features")
            if features not in _INSTANTSFM_FEATURES:
                raise ValueError(f"pointcloud.instantsfm.features={features!r} not in {sorted(_INSTANTSFM_FEATURES)}")

            # Both knobs below are consumed long after the run starts — random_seed at
            # InstantSfM's _build_config (after the SIFT + exhaustive-matching pass) and
            # depth_align only once the model is solved. Reject a bad value here so a typo
            # costs a config load, not a whole reconstruction.
            depth_align = instantsfm.get("depth_align")
            if depth_align not in DEPTH_ALIGN_MODELS:
                raise ValueError(
                    f"pointcloud.instantsfm.depth_align={depth_align!r} not in {sorted(DEPTH_ALIGN_MODELS)}"
                )

            # np.random.seed's domain; InstantSfM passes the value straight through
            random_seed = instantsfm.get("random_seed")
            if random_seed is not None and not (isinstance(random_seed, int) and 0 <= random_seed < 2**32):
                raise ValueError(
                    f"pointcloud.instantsfm.random_seed must be null or an int in [0, 2**32), got {random_seed!r}"
                )

        return config

    ########################################
    # Path properties
    ########################################

    @property
    def backend_dir(self) -> Path:
        """output_path / backend — e.g. out/vggtx/. Backend subdir for all stage 2+ artifacts."""
        return Path(self.config["output_path"]) / self.config["pointcloud"]["backend"]

    @property
    def frames_zarr(self) -> Path:
        """output_path / frames.zarr — canonical decode-once keyframe store for this run."""
        return Path(self.config["output_path"]) / "frames.zarr"

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
        """Extract frames from input video/dir into frames.zarr (sole persistent store)."""
        # Skip if the store already exists and overwrite not requested
        if not overwrite and self.frames_zarr.exists():
            logger.info("Frames already extracted at %s, skipping preprocess", self.frames_zarr)
            return self.frames_zarr

        if overwrite and self.frames_zarr.exists():
            shutil.rmtree(self.frames_zarr)

        pre_cfg = self.config["preproc"]
        n_frames = extract_frames(
            input_path=Path(self.config["input_path"]),
            frames_zarr=self.frames_zarr,
            frame_selection=pre_cfg["frame_selection"],
            fps=pre_cfg["fps"],
            min_frames=pre_cfg["min_frames"],
            max_frames=pre_cfg["max_frames"],
            n_workers=pre_cfg["n_workers"],
            undistort=pre_cfg["undistort"],
            search_radius=pre_cfg["search_radius"],
            vda_context_fps=pre_cfg["vda_context_fps"],
        )
        logger.info(
            "Preprocessing complete: %d frames at %s",
            n_frames,
            self.frames_zarr,
        )
        return self.frames_zarr

    def build_pointcloud(self, overwrite: bool = False) -> "PointcloudResult":
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
                frames_zarr=self.frames_zarr,
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

        # Apply cleaning step if enabled
        clean_cfg = pc_cfg["clean"]
        if clean_cfg["enabled"]:
            result = self._clean_pointcloud(result, clean_cfg)

        # Re-export the PLY from the FINAL result — clean may have dropped points since
        # the creator wrote its copy. Density is opt-in via pointcloud.export_max_points.
        self._export_pointcloud_ply(result)

        # Write the pose+intrinsics transforms.json alongside the COLMAP model
        self._write_transforms_json(result)

        self.pointcloud = result
        return result

    def _load_pointcloud_from_disk(self) -> "PointcloudResult":
        """Load PointcloudResult from COLMAP reconstruction on disk."""
        import pycolmap

        from collab_splats.pointcloud.base import CoordinateFrame, PointcloudResult

        colmap_dir = self.backend_dir / "colmap" / "sparse" / "0"
        recon = pycolmap.Reconstruction()
        recon.read(str(colmap_dir))
        # Rebuild image_paths from frames.zarr in store order, so it lines up with the per-frame
        # arrays the downstream stages index. The feedforward creators register COLMAP images as
        # frame_{source_idx:06d} with NO extension (vggt_omega.py, vggtx.py, mapanything.py) — the
        # frame_*.jpg spelling elsewhere is the zarr/localization id namespace, not this one.
        frame_indices = FrameStore.open(self.frames_zarr).frame_indices()
        image_paths = [Path(f"frame_{int(fi):06d}") for fi in frame_indices]
        # That naming is a contract between the store and the reconstruction, and nothing enforces
        # it at write time. Check it here: unchecked, a mismatch surfaces as a bare KeyError from
        # PointcloudResult.extrinsics, several frames into a stage and — on the remote path — after
        # a multi-GB pull that says nothing about which two artifacts disagree.
        registered = {img.name for img in recon.images.values()}
        missing = [p.name for p in image_paths if p.name not in registered]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(image_paths)} frames in {self.frames_zarr} are not "
                f"registered in {colmap_dir} (first: {missing[0]}); the frame store and the "
                f"reconstruction describe different runs."
            )
        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=image_paths,
        )

    def _clean_pointcloud(self, result: "PointcloudResult", cfg: dict) -> "PointcloudResult":
        """Apply open3d outlier removal to PointcloudResult.

        Removes outlier point3D IDs from reconstruction in-place.
        """
        import open3d as o3d

        # Skip cleaning if reconstruction has no 3D points
        if not result.reconstruction.points3D:
            return result

        # Build ordered list of point3D IDs that matches result.points ordering
        point3d_ids = list(result.reconstruction.points3D.keys())
        pts = result.points  # (P, 3)
        colors = result.colors  # (P, 3) uint8

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(float))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)

        # Statistical outlier removal — remove outlier point3D IDs from reconstruction in-place
        if cfg["outlier_removal"]:
            _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            inlier_set = set(inlier_idx)
            for i, pid in enumerate(point3d_ids):
                if i not in inlier_set:
                    result.reconstruction.delete_point3D(pid)

        # Optional voxel downsampling (affects visualization/density; no structural change)
        voxel_size = cfg["voxel_size"]
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size)

        logger.info("Pointcloud after cleaning: %d points", result.reconstruction.num_points3D())
        return result

    def _export_pointcloud_ply(self, result: "PointcloudResult") -> Path:
        """Write backend_dir/sparse_pc.ply (binary) from the post-clean result."""
        self.backend_dir.mkdir(parents=True, exist_ok=True)
        return write_pointcloud_ply(
            result.points,
            result.colors,
            self.backend_dir / "sparse_pc.ply",
            self.config["pointcloud"]["export_max_points"],
        )

    def _write_transforms_json(self, result: "PointcloudResult") -> None:
        """Write pose+intrinsics frames to backend_dir/transforms.json.

        Frames are frame_idx-keyed against frames.zarr, not file_path-keyed against an
        images/ dir, so a stock file_path-keyed dataparser cannot load this file as-is.
        """
        # Nothing to write if there are no registered frames
        extrinsics = result.extrinsics  # (N, 4, 4) w2c
        if extrinsics is None or len(extrinsics) == 0:
            return

        image_paths = result.image_paths

        intrinsics = result.intrinsics  # (N, 3, 3)

        # c2w = inv(w2c): invert each 4x4 pose into camera-to-world convention
        c2w = np.linalg.inv(extrinsics)  # (N, 4, 4)

        # Frames live in the canonical frames.zarr store (no images/ dir); pose+intrinsics only.
        frames = []
        for img_path, K, pose in zip(image_paths, intrinsics, c2w):
            frames.append(
                {
                    "frame_idx": FrameStore.frame_idx_from_path(img_path),
                    "fl_x": float(K[0, 0]),
                    "fl_y": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "transform_matrix": pose.tolist(),
                }
            )

        self.backend_dir.mkdir(parents=True, exist_ok=True)
        out = self.backend_dir / "transforms.json"

        # Poses are OpenCV c2w straight from the COLMAP model; no applied_transform
        payload = {"camera_model": "PINHOLE", "frames": frames}
        out.write_text(json.dumps(payload, indent=2))
        logger.info("transforms.json written to %s", out)

    def _run_sfm(self) -> "PointcloudResult":
        """
        SfM pointcloud path (backend: instantsfm) — VDA metric depth + InstantSfM global mapping.

        - Stages frames.zarr keyframes to backend_dir/images/ (InstantSfM reads a dir).
        - Generates depth_vda/images/npy/<stem>.npy (skipped when present), runs InstantSfMCreator,
          renames COLMAP images to the frame_NNNNNN contract, builds a FeedforwardResult at VDA
          depth resolution → pointcloud.zarr with provenance attrs, returns the PointcloudResult
          for the shared tail.
        """
        from collab_splats.pointcloud.base import CoordinateFrame, PointcloudResult

        pc_cfg = self.config["pointcloud"]
        backend = pc_cfg["backend"]
        if backend != "instantsfm":
            raise NotImplementedError(f"sfm backend {backend!r} is not implemented — only 'instantsfm' is")
        backend_dir = self.backend_dir
        backend_dir.mkdir(parents=True, exist_ok=True)
        store = FrameStore.open(self.frames_zarr)
        names = [f"frame_{int(fi):06d}.jpg" for fi in store.frame_indices()]

        # Stage keyframes as jpgs — exactly what FrameStore.export writes, so a complete staged set
        # is reused as-is. Any other set (partial, or from a different selection) is re-staged,
        # and the SIFT database keyed on it is dropped so InstantSfM cannot reuse stale features.
        image_dir = backend_dir / "images"
        staged = sorted(p.name for p in image_dir.iterdir()) if image_dir.is_dir() else []
        if staged != names:
            shutil.rmtree(image_dir, ignore_errors=True)
            (backend_dir / "colmap" / "instantsfm.db").unlink(missing_ok=True)
            store.export(image_dir, ext="jpg")
            logger.info("Staged %d keyframes to %s", len(names), image_dir)

        # VDA metric depth for every keyframe — cached across runs, stamped with what made it
        self._ensure_vda_depth(backend_dir, store, names)

        # Global SfM via the upstream python API; writes colmap/instantsfm.db + colmap/sparse/0
        creator = InstantSfMCreator(
            features=pc_cfg["instantsfm"]["features"],
            retriangulation=pc_cfg["instantsfm"]["retriangulation"],
            random_seed=pc_cfg["instantsfm"]["random_seed"],
        )
        recon = creator.reconstruct(backend_dir)
        del creator
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Rename to the frame_NNNNNN contract and rewrite the model in place
        _rename_images_to_stems(recon, backend_dir / "colmap" / "sparse" / "0")

        # Unified pointcloud.zarr at VDA depth res, with provenance from the installed package
        outputs = self._sfm_result_from_reconstruction(recon, backend_dir, store)

        # Align VDA depth to the COLMAP world before anything persists — the zarr and the
        # model must share one scale (splat depth targets, mesh fusion, localization lookup).
        # Raises rather than writing a VDA-metric zarr; depth_scale attrs mark aligned scenes.
        align_attrs = apply_depth_alignment(outputs, recon, model=pc_cfg["instantsfm"]["depth_align"])

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

        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=outputs.image_paths,
        )

    def _ensure_vda_depth(self, backend_dir: Path, store: FrameStore, names: list[str]) -> None:
        """
        Write depth_vda/images/npy/<stem>.npy for every keyframe, reusing what is already there.

        - Regenerates when the recorded generating inputs differ from this run's; an absent
          sidecar means the maps predate the stamp and are trusted.
        - Runs VDA over the contiguous context grid when preproc.vda_context_fps is set and the
          source video still matches, keeping only the keyframe rows; falls back to keyframes.
        """
        # VDA metric depth — the only shipped mode (use_depths=True). Gate on the npy set BEFORE
        # decoding anything (300 x 1080p is ~1.9 GB).
        context_fps = self.config["preproc"]["vda_context_fps"]

        # vda_depth_complete keys on `names` alone, but the depth CONTENT also depends on the
        # context grid, and `names` is always the same sequential frame_NNNNNN set — so changing
        # vda_context_fps leaves the stem set identical and would silently reuse stale depth. A
        # sidecar records the generating inputs; an ABSENT one means the maps predate this stamp
        # and are trusted, so only a present-and-different stamp invalidates.
        depth_sidecar = backend_dir / "depth_vda" / "inputs.json"
        keyframe_fps = self.config["preproc"]["fps"]
        signature = {
            "context_fps": float(context_fps) if context_fps else None,
            "keyframe_fps": float(keyframe_fps) if keyframe_fps else None,
            "n_names": len(names),
        }

        # write_text is not atomic and runs here get OOM-killed, so a half-written stamp is
        # realistic; an unreadable one is treated as a mismatch because regenerating is always safe
        cached_signature, unreadable = None, False
        if depth_sidecar.exists():
            try:
                cached_signature = json.loads(depth_sidecar.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("VDA depth sidecar %s is unreadable — regenerating depth", depth_sidecar)
                unreadable = True
        stale = unreadable or (cached_signature is not None and cached_signature != signature)
        if stale:
            logger.info(
                "VDA depth cache invalidated: generating inputs changed %s -> %s", cached_signature, signature,
            )

            # generate_vda_depth early-returns on a complete stem set, and none of these inputs
            # change that set — so the superseded maps must be REMOVED, not merely out-stamped
            shutil.rmtree(backend_dir / "depth_vda", ignore_errors=True)

        if stale or not vda_depth_complete(backend_dir, names):
            keep_rows, context_frames = None, None

            # Context stream: VDA is temporal, so run it over a contiguous constant-rate grid and
            # keep only the keyframe rows. Falls back to the keyframe path whenever the source
            # video is gone (rerun-from-processed, image-dir input) or the keyframes are off-grid.
            if context_fps:
                provenance = store.provenance()
                video_path = provenance.get("video_path")
                same_video = bool(video_path) and Path(video_path).is_file() and _video_unchanged(
                    Path(video_path), provenance,
                )
                if same_video:
                    # The keyframes were drawn from the grid recorded at preproc time; a
                    # different rate here is only caught when it happens to push them off-grid
                    sampled_at = provenance.get("vda_context_fps")
                    if sampled_at is not None and float(sampled_at) != float(context_fps):
                        logger.warning(
                            "preproc.vda_context_fps is %.2f but frames.zarr was sampled against a "
                            "%.2f fps grid — keyframes may not be members of the grid VDA runs on",
                            float(context_fps), float(sampled_at),
                        )

                    grid = context_indices(video_path, target_fps=float(context_fps))
                    keep_rows = _context_keep_rows(grid, [int(fi) for fi in store.frame_indices()])
                    if keep_rows is not None:
                        # Decode with the same distortion profile frames.zarr was written with,
                        # or the context frames and the keyframes disagree on K_new and the crop
                        profile = None
                        if provenance.get("undistort"):
                            profile = DistortionProfile.from_dict(provenance["undistort"]["profile"])
                        logger.info(
                            "VDA context stream: decoding %d frames at %.2f fps from %s",
                            len(grid), float(context_fps), video_path,
                        )
                        context_frames = decode_context(video_path, grid, profile=profile)
                else:
                    logger.warning(
                        "preproc.vda_context_fps is set but the source video is unavailable or no longer "
                        "matches the one frames.zarr was built from (%s) — falling back to keyframe-only VDA",
                        video_path,
                    )

            # One VDA pass either way; the context branch writes only the keyframe rows out.
            # used_context_fps is bound HERE, not read back after the branch: `del context_frames`
            # unbinds the name, and it is the resolved rate the sidecar has to record anyway.
            if context_frames is not None:
                used_context_fps = float(context_fps)
                generate_vda_depth(
                    context_frames, fps=float(context_fps), out_dir=backend_dir, names=names, keep_rows=keep_rows,
                )
                del context_frames
            else:
                used_context_fps = None
                frames = np.ascontiguousarray(store.images())
                generate_vda_depth(frames, fps=float(self.config["preproc"]["fps"]), out_dir=backend_dir, names=names)
                del frames
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Stamp what actually produced these maps: both fallbacks above land here, so a
            # requested context rate that never ran must not be recorded as if it had
            depth_sidecar.parent.mkdir(parents=True, exist_ok=True)
            depth_sidecar.write_text(json.dumps({**signature, "context_fps": used_context_fps}))

    def _sfm_result_from_reconstruction(
        self, recon: "pycolmap.Reconstruction", backend_dir: Path, store: FrameStore
    ) -> "FeedforwardResult":
        """
        Build a FeedforwardResult from an InstantSfM COLMAP model + VDA depth maps.

        - Rows follow store order; image names must already be the frame_NNNNNN contract.
        - Depth maps define the (h, w) grid; K, images and pixel_indices are scaled to it.
        - confidence / mv_* stay absent — SfM has no learned per-pixel confidence.
        """
        from collab_splats.pointcloud.feedforward.base import FeedforwardResult

        # Every store frame must be registered — a partial model would leave rows without poses
        if len(recon.images) != len(store):
            raise RuntimeError(
                f"InstantSfM registered {len(recon.images)}/{len(store)} frames — partial "
                "registration is not supported; re-run with more overlap"
            )
        images_sorted = sorted(recon.images.values(), key=lambda im: im.name)
        expected = [f"frame_{int(fi):06d}" for fi in store.frame_indices()]
        registered = [im.name for im in images_sorted]
        if registered != expected:
            raise ValueError(
                f"registered image names do not match {self.frames_zarr} frame indices "
                f"(first registered: {registered[0]}, first expected: {expected[0]}); the frame "
                "store and the reconstruction describe different runs."
            )
        name_to_row = {name: row for row, name in enumerate(registered)}

        # Per-frame VDA depth, aligned by name; its grid is the model resolution of this result
        depth_dir = backend_dir / "depth_vda" / "images" / "npy"
        depths = np.stack([np.load(depth_dir / f"{im.name}.npy") for im in images_sorted]).astype(np.float32)
        n, h, w = depths.shape

        # Poses: cam_from_world (w2c) as homogeneous 4x4
        extrinsics = np.stack(
            [np.vstack([im.cam_from_world().matrix(), [0.0, 0.0, 0.0, 1.0]]) for im in images_sorted]
        ).astype(np.float32)

        # COLMAP K is at staged-jpg (original) resolution; the depth grid is model-res, so K is
        # rescaled to it — pairing original-res K with model-res depth is the 2026-08-11
        # mesh-regression class (see _feedforward_to_tsdf_inputs). The COLMAP cameras must be at
        # the store's resolution, else the staged set / SIFT DB came from a different store.
        orig_h, orig_w = store.image(0).shape[:2]
        cam_dims = {(recon.cameras[im.camera_id].width, recon.cameras[im.camera_id].height) for im in images_sorted}
        if cam_dims != {(orig_w, orig_h)}:
            raise ValueError(
                f"COLMAP camera resolution {sorted(cam_dims)} does not match {self.frames_zarr} "
                f"({orig_w}x{orig_h}); the staged images / SIFT database came from a different store."
            )
        sx, sy = w / orig_w, h / orig_h
        intrinsics = np.stack([recon.cameras[im.camera_id].calibration_matrix() for im in images_sorted])
        intrinsics = intrinsics.astype(np.float32)
        intrinsics[:, 0, :] *= sx
        intrinsics[:, 1, :] *= sy

        # Sparse points in point3D-id order; pixel_indices from each point's first
        # observation. Observation-less points (InstantSfM's sub-min-track-length
        # exports) have no pixel provenance and are dropped.
        point3d_ids = _tracked_point3d_ids(recon)
        points = np.array([recon.points3D[pid].xyz for pid in point3d_ids], dtype=np.float32).reshape(-1, 3)
        colors = np.array([recon.points3D[pid].color for pid in point3d_ids], dtype=np.uint8).reshape(-1, 3)
        pixel_indices = _pixel_indices_from_reconstruction(
            recon, point3d_ids, name_to_row, scale_x=sx, scale_y=sy, depth_hw=(h, w)
        )

        # RGB at depth res as (N, 3, H, W) float32 in [0, 1] — the feedforward images convention
        images_arr = np.stack([cv2.resize(store.image(i), (w, h), interpolation=cv2.INTER_AREA) for i in range(n)])
        images_arr = images_arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        # Dense world points by unprojecting depth through the rescaled K and w2c poses
        world_points = unproject_depth_map_to_point_map(depths[..., None], extrinsics[:, :3, :], intrinsics)
        world_points = world_points.astype(np.float32)

        # No crop: the depth grid is a full-frame resize, so the crop box is the whole original
        # frame in ORIGINAL pixels — [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], the loger convention
        # (consumers read [:4] as original-res coordinates, not depth-grid ones)
        original_coords = np.array([[0, 0, orig_w, orig_h, orig_w, orig_h]] * n, dtype=np.float32)

        return FeedforwardResult(
            points=points,
            colors=colors,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_paths=[Path(im.name) for im in images_sorted],
            original_coords=original_coords,
            model_width=w,
            model_height=h,
            images=images_arr,  # numpy float32 on purpose — save_zarr accepts it; no torch tensor needed
            world_points=world_points,
            depth=depths,
            pixel_indices=pixel_indices,
        )

    def refine_poses(self, overwrite: bool = False) -> "PointcloudResult":
        """Refine camera poses via LM bundle adjustment; rewrite pose-derived artifacts.

        One implementation for both triggers: runs inline after the pointcloud stage when
        pointcloud.bundle_adjustment is enabled, and from disk via --stages refine against
        a processed scene. Loads everything from pointcloud.zarr — no live creator needed.
        """
        # SfM results are already globally bundle-adjusted; LM re-refinement is undefined here
        if self.config["pointcloud"]["method"] == "sfm":
            raise ValueError("refine_poses is not supported for pointcloud.method: sfm")

        # Heavy deps imported lazily, matching the other stage methods
        from vggt.utils.geometry import unproject_depth_map_to_point_map

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
            wp = unproject_depth_map_to_point_map(
                ff.depth[..., None], ff.extrinsics[:, :3, :], ff.intrinsics
            ).astype(np.float32)
            store["world_points"][:] = wp

        # Refresh the remaining derived artifacts through the standard writers
        result = self._load_pointcloud_from_disk()
        self._export_pointcloud_ply(result)
        self._write_transforms_json(result)

        # Marker + provenance in one file: BA config and per-step LM loss history
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in dataclasses.asdict(cfg).items()
                    },
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
        result: "PointcloudResult | None" = None,
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
            zarr_path = _extract_2d_features(extractor_name, self.frames_zarr, self.semantics_cache_dir)
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
        result: "PointcloudResult | None" = None,
        overwrite: bool = False,
    ) -> Path:
        """Build a TSDF mesh from `mesh.source` depth: pointcloud.zarr (default) or splats.zarr renders.

        COLMAP is the pose authority on the feedforward path; the splats path fuses the poses
        the splats were actually rendered with (including pose-opt deltas) and uses alpha as
        confidence. The splats stage is never auto-run — `source: splats` requires it on disk.
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

        # Source-specific input check: each branch needs only its own zarr on disk
        mesh_cfg = self.config["mesh"]
        source = mesh_cfg["source"]
        if source not in ("feedforward", "splats"):
            raise ValueError(
                f"mesh.source must be 'feedforward' or 'splats', got {source!r}"
            )
        pointcloud_zarr = self.pointcloud_zarr
        splats_zarr = None
        if source == "splats":
            splats_zarr = self.backend_dir / "splats" / "splats.zarr"
            if not splats_zarr.exists():
                raise ValueError(
                    f"mesh.source: splats needs {splats_zarr} — run the splats stage first "
                    "(it is never auto-run)"
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
            voxel_size=mesh_cfg["voxel_size"],
            sdf_trunc=mesh_cfg["sdf_trunc"],
            depth_trunc=mesh_cfg["depth_trunc"],
            clean_repair=mesh_cfg["clean_repair"],
            conf_percentile=mesh_cfg["conf_percentile"],
            native_resolution=mesh_cfg["native_resolution"],
            color_map_iterations=mesh_cfg["color_map_iterations"],
            splat_depth=mesh_cfg["splat_depth"],
            splat_max_depth_frac=mesh_cfg["splat_max_depth_frac"],
            splat_max_depth_grad=mesh_cfg["splat_max_depth_grad"],
            frames_zarr=self.frames_zarr,
            source=source,
            splats_zarr=splats_zarr,
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
            self.frames_zarr,
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
        features, ids, _ = load_localization_db(
            self.pointcloud_zarr, extractor_name
        )
        logger.info("verify(): load_localization_db took %.1f s", time.perf_counter() - t)
        # Loma matches from stored features (keypoints_normalized). A cache from before
        # that array existed gets rebuilt once (~1 min) — no degraded fallback path.
        # getattr: duck-typed test stubs and non-split matchers lack the attribute.
        if getattr(matcher, "_split_loma_forward", False) and any(
            f.keypoints_normalized is None for f in features
        ):
            logger.info("verify(): feature cache lacks keypoints_normalized — rebuilding localization DB")
            t = time.perf_counter()
            self.build_localization_db(overwrite=True)
            logger.info("verify(): build_localization_db(overwrite=True) took %.1f s", time.perf_counter() - t)
            t = time.perf_counter()
            features, ids, _ = load_localization_db(
                self.pointcloud_zarr, extractor_name
            )
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
        # FrameStore frames to extract() — because match-time index recovery lands on the
        # extract-time keypoint tables (the probe's cross-call condition holds for
        # identical inputs only), and those tables are what verification exports to the
        # COLMAP DB. Model-res ff.images would index a different table entirely.
        # verify_reconstruction itself hard-refuses index-incapable pairwise matchers.
        # Handed over lazily (_LazyFrames): frames decode on access under a small LRU,
        # so peak memory stays bounded instead of N full-res frames at once. (The
        # descriptor branch in verify_reconstruction ignores `images` — passing them
        # unconditionally is free until a frame is actually accessed.)
        store = FrameStore.open(self.frames_zarr)
        images = _LazyFrames(store, store.frame_indices())
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
        Train Gaussian splats from the pointcloud stage. Returns path to splats/splats.zarr.
        """
        out_dir = self.backend_dir / "splats"
        splats_zarr = out_dir / "splats.zarr"
        if not overwrite and self._stage_output_exists("splats"):
            logger.info("Splats exist at %s, skipping", out_dir)
            return splats_zarr

        result = self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # gsplat is CUDA-only; import lazily so Reconstructor stays importable without it
        from collab_splats.pointcloud.feedforward.base import FeedforwardResult
        from collab_splats.pointcloud.utils import confidence_mask
        from collab_splats.splats.trainer import SplatsConfig, train

        cfg = SplatsConfig.from_dict(self.config["splats"])

        # Frames in COLMAP image order, looked up in frames.zarr by the frame index in each name
        store = FrameStore.open(self.frames_zarr)
        frame_indices = [FrameStore.frame_idx_from_path(path) for path in result.image_paths]
        # CPU-resident by design: train() moves one view to the GPU at a time
        images = np.stack([store.image_by_frame_idx(frame_idx) for frame_idx in frame_indices])

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
                FrameStore.frame_idx_from_path(path): row for row, path in enumerate(feedforward.image_paths)
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
                # No confidence channel, so mesh.conf_percentile cannot apply. On the sfm path
                # reliability is enforced upstream instead: affine alignment writes 0 for
                # saturated and beyond-evidence pixels, and 0 means "no target" — while scale
                # alignment masks nothing, which is why the model is named rather than implied.
                # The branch is reachable off the sfm path too, hence the guarded lookup.
                instantsfm_cfg = self.config["pointcloud"].get("instantsfm")
                zero_fraction = 100.0 * float((depth_targets <= 0).mean())
                logger.info(
                    "splats depth targets: mesh.conf_percentile=%s not applied (no confidence "
                    "channel); depth_align=%s masks %.2f%% of target pixels",
                    conf_percentile,
                    instantsfm_cfg.get("depth_align") if instantsfm_cfg else None,
                    zero_fraction,
                )

        train(
            cfg, images, result.extrinsics, result.intrinsics, result.points, result.colors, out_dir,
            depth_targets=depth_targets,
        )
        logger.info("Splats saved to %s", out_dir)
        return splats_zarr

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
            frames_zarr=self.frames_zarr,
            output_path=out_json,
            backend=self.config["pointcloud"]["backend"],
        )
        logger.info("Reconstruction quality report written to %s", out_json)
        return out_json

    def _stage_output_exists(self, stage: str) -> bool:
        """True if `stage`'s on-disk output is already present (lets deps be reused across runs)."""
        if stage == "preproc":
            return self.frames_zarr.exists()
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
            return (self.backend_dir / "splats" / "splats.zarr").exists()
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

    def _resolve_result(self) -> "PointcloudResult | None":
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
        # disk — so `--stages pointcloud` reuses a prior preprocess's frames.zarr.
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
