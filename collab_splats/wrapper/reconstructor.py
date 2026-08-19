"""5-stage reconstruction pipeline wrapper."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import subprocess
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

from collab_splats.pointcloud.export import write_pointcloud_ply
from collab_splats.preproc import get_video_info, sample_frames
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.semantics.compression import (
    FeatureAutoencoder,
    lifted_store_path,
    write_point_features,
)

if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult
    from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# base.yaml is the single source of defaults; __init__ merges any passed config over it.
DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"

_FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega", "loger"}
_SFM_BACKENDS = {"colmap", "hloc"}
_VALID_METHODS = {"feedforward", "sfm", "nerfstudio"}
_STAGE_ORDER = ["preproc", "pointcloud", "refine", "semantics", "mesh", "localize", "verify"]
_STAGE_DEPS: dict[str, list[str]] = {
    "preproc": [],
    "pointcloud": ["preproc"],
    # refine rewrites pointcloud outputs in place; deliberately NOT a dependency of the
    # stages below — that would demote them from LEAF_STAGES and break their disk re-run.
    # Staleness contract: after --stages refine, re-run dependents with overwrite
    # (configs/README.md). Inline runs are ordered refine-before-dependents, so never stale.
    "refine": ["pointcloud"],
    "semantics": ["pointcloud"],
    "mesh": ["pointcloud"],
    "localize": ["pointcloud"],
    # verify reuses the localize feature cache but builds it itself when absent, so its
    # only hard dependency is the reconstruction
    "verify": ["pointcloud"],
}
# A stage is re-runnable on its own iff nothing depends on it → {refine, semantics, mesh, localize, verify}.
# Derived from the graph above rather than hardcoded: a future stage that depends on mesh drops
# mesh from this set automatically, so callers gating on it can never disagree with _STAGE_DEPS.
LEAF_STAGES = frozenset(s for s in _STAGE_ORDER if not any(s in deps for deps in _STAGE_DEPS.values()))


########################################
# Helpers
########################################


def _extract_frames(
    input_path: Path,
    frames_zarr: Path,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
) -> int:
    """Extract frames from video or image dir into frames.zarr (sole persistent store).

    frames.zarr is the canonical decode-once keyframe store; no JPEG dir is written.
    Returns the number of frames stored.
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
        }
        FrameStore.create(frames_zarr, frame_arrays, records, provenance=prov)
        return len(frame_arrays)

    # Fail loud on an unreadable/empty video — 0 total frames means a bad path or a
    # codec ffmpeg can't decode, which otherwise silently yields an empty store.
    total_frames = get_video_info(str(input_path))["total_frames"]
    if total_frames == 0:
        raise ValueError(
            f"No frames decoded from {input_path} (0 total frames). "
            "Check the path exists and is a video ffmpeg can read."
        )

    # Video — 'fps' samples at a constant wall-clock rate (band-bounded), 'uniform'
    # spreads exactly max_frames over the whole video, 'optical_flow' picks high-motion
    # frames. Each method gets only its own knobs; sample_frames rejects the others.
    if frame_selection == "fps":
        frame_arrays, records = sample_frames(
            str(input_path),
            method="fps",
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
    elif frame_selection == "uniform":
        frame_arrays, records = sample_frames(str(input_path), method="uniform", max_frames=max_frames)
    elif frame_selection == "optical_flow":
        frame_arrays, records = sample_frames(str(input_path), method="optical_flow", max_frames=max_frames)
    else:
        raise ValueError(
            f"preprocessing.frame_selection must be 'fps', 'uniform' or 'optical_flow', got {frame_selection!r}"
        )
    method = frame_selection

    # Every candidate failed the quality gate (or selector rejected all) — refuse to
    # write an empty store that would only surface as a downstream FileNotFound.
    if not frame_arrays:
        raise ValueError(
            f"0 frames selected from {input_path} ({total_frames} decoded) with "
            f"frame_selection={frame_selection!r}. All frames failed the quality gate — "
            "loosen the blur threshold or use a sharper video."
        )

    # Write the canonical frames.zarr (decode-once keyframe store)
    prov = {
        "video_path": str(input_path),
        "video_mtime": input_path.stat().st_mtime,
        "method": method,
        "fps": fps,
        "max_frames": max_frames,
    }
    FrameStore.create(frames_zarr, frame_arrays, records, provenance=prov)

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

    Saves feedforward.zarr to output_dir after inference so downstream stages
    (semantics lift, mesh) can load depth/confidence/pixel data. Returns the
    PointcloudResult and the created Viewer (None unless loop_closure + viz_enabled),
    so callers can keep the viser server reachable after this function returns.

    ``loop_closure`` is either a bool (enable with all LoopClosureConfig defaults) or
    a dict of knobs (submap_size, submap_overlap, scale_method, …); an ``enabled`` key
    in the dict toggles it, defaulting to True when any knobs are given.

    ``creator_kwargs`` is the per-backend ``pointcloud.<backend>`` config block, forwarded
    verbatim to the creator's constructor. ``max_points`` and ``use_multiview_confidence``
    are reserved — both are passed explicitly, so redeclaring either raises ValueError.

    ``max_frames`` is ``preprocessing.max_frames``, used only to decide whether the LoGeR
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

    # preprocessing.max_frames is a VGGT-Omega GPU property applied in the preproc stage,
    # which has already run by the time we get here. Flipping to loger under that same
    # ceiling therefore processes exactly as many frames as Omega would, and LoGeR appears
    # to buy nothing. Warn rather than change behaviour — LoGeR's true ceiling is unmeasured.
    # None means no ceiling was configured, so there is no advice to give. Unlike the refusal
    # above, this genuinely needs the store, so it belongs after the open.
    if backend == "loger":
        n_frames = len(store)
        if max_frames is not None and n_frames <= max_frames:
            logger.warning(
                "LoGeR is running on %d frames, at or under the preprocessing.max_frames "
                "ceiling of %d. That ceiling is VGGT-Omega's GPU limit, not LoGeR's — LoGeR "
                "uses sliding-window inference and is built for longer sequences. Raise "
                "preprocessing.max_frames to use it.",
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

    # Persist FeedforwardResult to feedforward.zarr — required by semantics lift + mesh stages
    ff_outputs = getattr(creator, "outputs", None)
    if ff_outputs is not None:
        zarr_path = output_dir / "feedforward.zarr"
        ff_outputs.save_zarr(zarr_path)
        logger.info("feedforward.zarr saved: %s  (%s pts)", zarr_path, f"{len(ff_outputs.points):,}")
    else:
        logger.warning("Creator has no outputs after reconstruct — feedforward.zarr not saved")

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
    frames_zarr: Path,
    cache_dir: Path,
) -> Path:
    """Extract 2D features for all frames straight from the canonical store.

    Delegates to BaseFeatureExtractor.extract_and_cache_from_zarr, which iterates the
    frames zarr lazily (one chunk at a time) and writes cache_dir/{name}.zarr — no temp
    JPG export, no full-RAM load.
    """
    extractor = _get_extractor(extractor_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)


def _lift_and_save(
    extractor_name: str,
    zarr_path: Path,
    feedforward_zarr: Path,
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

    # Validate feedforward zarr exists before attempting load
    if not feedforward_zarr.exists():
        raise FileNotFoundError(
            f"feedforward.zarr not found at {feedforward_zarr}. "
            "Run build_pointcloud() with a feedforward backend first."
        )

    # Load feature maps from zarr cache: (N, D, H_p, W_p)
    store = zarr.open(str(zarr_path), mode="r")
    features_arr = store["features"]
    feature_maps = [torch.from_numpy(np.array(features_arr[i])) for i in range(features_arr.shape[0])]

    # Load FeedforwardResult with depth/pixel data for lifting
    ff_result = FeedforwardResult.load_zarr(feedforward_zarr)

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
    feedforward_zarr: Path,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    clean_repair: bool = False,
    conf_percentile: float | None = None,
    native_resolution: bool = False,
    color_map_iterations: int = 0,
    frames_zarr: Path | None = None,
) -> Path:
    """Fuse depth + RGB from feedforward.zarr into a TSDF mesh, using COLMAP poses."""
    from collab_splats.mesh.utils import pointcloud_to_mesh
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    # world_points is the largest array in the store and the mesh path no longer reads it.
    # native_resolution skips the zarr's model-res RGB too — it comes from frames.zarr instead.
    ff = FeedforwardResult.load_zarr(
        feedforward_zarr, load_images=not native_resolution, load_world_points=False
    )

    # COLMAP is the pose authority — BA and loop-closure corrections land in the reconstruction,
    # not back in the zarr. On the model-res path intrinsics stay the zarr's (build_colmap
    # rescaled COLMAP's camera to original resolution, while the zarr's depth and RGB are at
    # model resolution); the native path swaps in COLMAP's original-res K below.
    if ff.depth is None:
        raise ValueError(f"{feedforward_zarr} has no depth — cannot mesh.")
    if result.extrinsics.shape[0] != ff.depth.shape[0]:
        raise ValueError(
            f"Frame-count mismatch: COLMAP reconstruction has {result.extrinsics.shape[0]} "
            f"images but {feedforward_zarr} has {ff.depth.shape[0]}. They are from different "
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


def _localization_db_exists(feedforward_zarr: Path, extractor_name: str) -> bool:
    """True if the local-feature DB group already exists in feedforward.zarr."""
    import zarr as zarr_lib

    try:
        store = zarr_lib.open_group(str(feedforward_zarr), mode="r")
        return (
            "local_features" in store
            and extractor_name in store["local_features"]
            and "reconstruction" in store["local_features"][extractor_name]
        )
    except Exception:
        return False


def _build_localization_db(
    feedforward_zarr: Path, extractor_name: str, frames_zarr: Path, top_k: int = 8
) -> Path:
    """Build the per-frame local-feature localization cache into feedforward.zarr.

    Loads the FeedforwardResult, runs the local matcher over every DB frame, and persists
    keypoints/descriptors to group local_features/{extractor_name}/reconstruction. top_k
    is the pairwise (vismatch) matching fan-out; the descriptor path ignores it.
    """
    # Heavy deps kept inline so the module imports without GPU/model libs
    from collab_splats.localization.extractors import LocalMatcher
    from collab_splats.localization.localizer import CameraLocalizer
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.preproc.frame_store import FrameStore

    ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True, load_world_points=True)
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
        zarr_path=feedforward_zarr,
        top_k=top_k,
    )
    logger.info("Localization DB built: %s :: local_features/%s", feedforward_zarr, extractor_name)
    return feedforward_zarr


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

        pre_cfg = self.config["preprocessing"]
        n_frames = _extract_frames(
            input_path=Path(self.config["input_path"]),
            frames_zarr=self.frames_zarr,
            frame_selection=pre_cfg["frame_selection"],
            fps=pre_cfg["fps"],
            min_frames=pre_cfg["min_frames"],
            max_frames=pre_cfg["max_frames"],
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

        # Skip if COLMAP + feedforward.zarr both exist and overwrite not requested.
        # Require feedforward.zarr too — if a previous run was partial (zarr missing),
        # we must re-run inference rather than loading stale COLMAP.
        if not overwrite and self._stage_output_exists("pointcloud"):
            logger.info("Pointcloud exists at %s, loading from disk", self.backend_dir / "colmap")
            self.pointcloud = self._load_pointcloud_from_disk()
            return self.pointcloud

        # Dispatch to the appropriate reconstruction method
        if method == "nerfstudio":
            result = self._run_nerfstudio()
        elif method == "sfm":
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
                max_frames=self.config["preprocessing"]["max_frames"],
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

        # Write nerfstudio-compatible transforms.json
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
        """Merge pose+intrinsics frames into backend_dir/transforms.json.

        Frames are frame_idx-keyed against frames.zarr, not file_path-keyed against an
        images/ dir, so stock nerfstudio dataparsers cannot load this file as-is.
        """
        # Nothing to write if reconstruction has no registered images
        if not result.reconstruction.images:
            return

        extrinsics = result.extrinsics  # (N, 4, 4) w2c
        image_paths = result.image_paths

        # Nothing to write if there are no registered frames
        if extrinsics is None or len(extrinsics) == 0:
            return

        intrinsics = result.intrinsics  # (N, 3, 3)

        # c2w = inv(w2c): invert each 4x4 pose for nerfstudio convention
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

        # Merge over whatever colmap_to_json already wrote: it owns ply_file_path and
        # applied_transform (splatfacto needs both), we own camera_model + frames.
        payload: dict = {}
        if out.exists():
            payload = json.loads(out.read_text())
        payload["camera_model"] = "PINHOLE"
        payload["frames"] = frames

        out.write_text(json.dumps(payload, indent=2))
        logger.info("transforms.json written to %s", out)

    def _run_sfm(self) -> "PointcloudResult":
        """Run SfM pointcloud stage (colmap/hloc). Experimental."""
        raise NotImplementedError("SfM path not yet implemented — use method: feedforward")

    def _run_nerfstudio(self) -> "PointcloudResult":
        """Run full nerfstudio pipeline via ns-process-data + ns-train subprocesses.

        output_path/nerfstudio/ acts as nerfstudio data dir.
        """
        import pycolmap

        from collab_splats.pointcloud.base import CoordinateFrame, PointcloudResult

        ns_cfg = self.config["nerfstudio"]
        sfm_tool = ns_cfg["sfm_tool"]
        train_method = ns_cfg["train_method"]

        ns_data_dir = Path(self.config["output_path"]) / "nerfstudio"
        input_path = Path(self.config["input_path"])

        # ns-process-data: frame extraction + SfM
        data_type = "video" if input_path.suffix.lower() in {".mp4", ".mov", ".avi"} else "images"
        process_cmd = [
            "ns-process-data",
            data_type,
            "--data",
            str(input_path),
            "--output-dir",
            str(ns_data_dir),
            "--sfm-tool",
            sfm_tool,
        ]
        logger.info("Running ns-process-data: %s", " ".join(process_cmd))
        subprocess.run(process_cmd, check=True)

        # ns-train: train nerfstudio model
        train_cmd = [
            "ns-train",
            train_method,
            "--data",
            str(ns_data_dir),
            "--output-dir",
            str(ns_data_dir / "outputs"),
        ]
        logger.info("Running ns-train: %s", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)

        # Load COLMAP sparse model produced by ns-process-data
        colmap_dir = ns_data_dir / "colmap" / "sparse" / "0"
        if not colmap_dir.exists():
            raise RuntimeError(f"ns-process-data did not produce COLMAP sparse model at {colmap_dir}")

        recon = pycolmap.Reconstruction()
        recon.read(str(colmap_dir))
        image_paths = sorted((ns_data_dir / "images").glob("*.jpg")) + sorted((ns_data_dir / "images").glob("*.png"))
        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=image_paths,
        )

    def refine_poses(self, overwrite: bool = False) -> "PointcloudResult":
        """Refine camera poses via LM bundle adjustment; rewrite pose-derived artifacts.

        One implementation for both triggers: runs inline after the pointcloud stage when
        pointcloud.bundle_adjustment is enabled, and from disk via --stages refine against
        a processed scene. Loads everything from feedforward.zarr — no live creator needed.
        """
        # Heavy deps imported lazily, matching the other stage methods
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        from collab_splats.geometry.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
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
        zarr_path = self.backend_dir / "feedforward.zarr"
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

        # Write pose-derived arrays back to feedforward.zarr so zarr and COLMAP never disagree
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
        feedforward_zarr = self.backend_dir / "feedforward.zarr"
        logger.info("Lifting 2D features to 3D pointcloud")
        out_dir = _lift_and_save(
            extractor_name,
            zarr_path,
            feedforward_zarr,
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
        """Build mesh from pointcloud depth maps. Returns path to mesh.ply."""
        mesh_path = self.backend_dir / "mesh.ply"

        # Skip if mesh already on disk
        if not overwrite and self._stage_output_exists("mesh"):
            logger.info("Mesh exists at %s, skipping", mesh_path)
            return mesh_path

        result = result or self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        feedforward_zarr = self.backend_dir / "feedforward.zarr"
        if not feedforward_zarr.exists():
            raise FileNotFoundError(
                f"feedforward.zarr not found at {feedforward_zarr}. "
                "Mesh requires depth maps from a feedforward backend."
            )

        mesh_cfg = self.config["mesh"]
        out = _run_tsdf_mesh(
            result=result,
            feedforward_zarr=feedforward_zarr,
            output_dir=self.backend_dir,
            voxel_size=mesh_cfg["voxel_size"],
            sdf_trunc=mesh_cfg["sdf_trunc"],
            depth_trunc=mesh_cfg["depth_trunc"],
            clean_repair=mesh_cfg["clean_repair"],
            conf_percentile=mesh_cfg["conf_percentile"],
            native_resolution=mesh_cfg["native_resolution"],
            color_map_iterations=mesh_cfg["color_map_iterations"],
            frames_zarr=self.frames_zarr,
        )
        logger.info("Mesh saved to %s", out)
        return out

    def build_localization_db(self, overwrite: bool = False) -> Path:
        """Build/refresh the per-frame local-feature localization cache in feedforward.zarr."""
        loc_cfg = self.config["localization"]
        extractor_name = loc_cfg["matcher"]

        feedforward_zarr = self.backend_dir / "feedforward.zarr"
        if not feedforward_zarr.exists():
            raise FileNotFoundError(
                f"feedforward.zarr not found at {feedforward_zarr}. "
                "Localization DB requires a feedforward pointcloud stage first."
            )

        # Skip if the DB group already exists and overwrite not requested
        if not overwrite and self._stage_output_exists("localize"):
            logger.info(
                "Localization DB exists at %s :: local_features/%s, skipping",
                feedforward_zarr,
                extractor_name,
            )
            return feedforward_zarr

        return _build_localization_db(
            feedforward_zarr, extractor_name, self.frames_zarr, top_k=loc_cfg["top_k"]
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
        self.build_localization_db()
        extractor_name = self.config["localization"]["matcher"]

        # Heavy deps kept inline so the module imports without GPU/model libs
        from collab_splats.geometry.verification import verify_reconstruction
        from collab_splats.localization.extractors import LocalMatcher
        from collab_splats.localization.localizer import load_reconstruction_features

        features, ids, _ = load_reconstruction_features(
            self.backend_dir / "feedforward.zarr", extractor_name
        )
        # The cache ids are frame_XXXXXX.jpg, the reconstruction registers frame_XXXXXX
        # (no extension) — compare stems so a reordered/rebuilt cache cannot slip through.
        recon = result.reconstruction
        recon_names = [recon.images[i].name for i in sorted(recon.images)]
        if [Path(n).stem for n in ids] != [Path(n).stem for n in recon_names]:
            raise ValueError(
                "Feature cache and reconstruction disagree on frame order/naming — "
                "rebuild the localization DB (overwrite=True)."
            )
        matcher = LocalMatcher(extractor_name)
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

    def _stage_output_exists(self, stage: str) -> bool:
        """True if `stage`'s on-disk output is already present (lets deps be reused across runs)."""
        if stage == "preproc":
            return self.frames_zarr.exists()
        if stage == "pointcloud":
            colmap_done = (self.backend_dir / "colmap" / "sparse" / "0" / "cameras.bin").exists()
            zarr_done = (self.backend_dir / "feedforward.zarr").exists()
            return colmap_done and zarr_done
        if stage == "refine":
            return (self.backend_dir / "colmap" / "refine.json").exists()
        # Leaf-stage markers. Only preproc/pointcloud are ever depended on, but run_pipeline also
        # needs these to refuse a named stage whose output already exists — and each leaf stage's
        # own skip-check reads them, so they live here once instead of three times.
        if stage == "mesh":
            return (self.backend_dir / "mesh.ply").exists()
        if stage == "semantics":
            lifted = lifted_store_path(self.backend_dir / "semantics", self.config["semantics"]["extractor"])
            return lifted.exists()
        if stage == "localize":
            feedforward_zarr = self.backend_dir / "feedforward.zarr"
            return feedforward_zarr.exists() and _localization_db_exists(
                feedforward_zarr, self.config["localization"]["matcher"]
            )
        if stage == "verify":
            return (self.backend_dir / "colmap" / "verification.json").exists()
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
            stages: Subset of ["preproc", "pointcloud", "refine", "semantics", "mesh", "localize", "verify"].
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
            if self.config["mesh"]["enabled"]:
                stages.append("mesh")
            if self.config["localization"]["enabled"]:
                stages.append("localize")
            if self.config["pointcloud"]["geometric_verification"]:
                stages.append("verify")

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
            elif stage == "mesh":
                self.mesh(result=result, overwrite=overwrite)
            elif stage == "localize":
                self.build_localization_db(overwrite=overwrite)
            elif stage == "verify":
                self.verify(overwrite=overwrite)

    def launch_dashboard(self) -> None:
        """Launch interactive dashboard for current reconstruction state."""
        from collab_splats.dashboard.__main__ import (
            main as dashboard_main,  # optional heavy dep; lazy load
        )

        dashboard_main()
