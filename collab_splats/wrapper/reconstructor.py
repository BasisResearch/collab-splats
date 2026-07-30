"""5-stage reconstruction pipeline wrapper."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import subprocess
import warnings
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
from collab_splats.semantics.compression import FeatureAutoencoder, write_point_features

if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult
    from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# base.yaml is the single source of defaults; __init__ merges any passed config over it.
DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"

_FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega"}
_SFM_BACKENDS = {"colmap", "hloc"}
_VALID_METHODS = {"feedforward", "sfm", "nerfstudio"}
_STAGE_ORDER = ["preproc", "pointcloud", "semantics", "mesh", "localize"]
_STAGE_DEPS: dict[str, list[str]] = {
    "preproc": [],
    "pointcloud": ["preproc"],
    "semantics": ["pointcloud"],
    "mesh": ["pointcloud"],
    "localize": ["pointcloud"],
}


########################################
# Helpers
########################################


def _extract_frames(
    input_path: Path,
    frames_zarr: Path,
    frame_selection: str,
    frame_proportion: float,
    min_frames: int,
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
        prov = {"video_path": str(input_path), "video_mtime": None, "method": "dir", "max_frames": max_frames}
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

    # Video — 'optical_flow' picks high-motion frames; 'uniform' (default) spreads evenly
    if frame_selection == "optical_flow":
        method = "optical_flow"
        frame_arrays, records = sample_frames(
            str(input_path),
            method="optical_flow",
            max_frames=max_frames if max_frames is not None else 200,
        )
    else:  # uniform
        # Derive target count from proportion, clamped to [min_frames, max_frames];
        # the uniform sampler spreads that count over the video itself
        method = "uniform"
        target_count = max(min_frames, int(total_frames * frame_proportion))
        if max_frames is not None:
            target_count = min(target_count, max_frames)
        frame_arrays, records = sample_frames(
            str(input_path),
            method="uniform",
            max_frames=target_count,
        )

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
) -> tuple["PointcloudResult", "Viewer | None"]:
    """Instantiate feedforward creator, optionally wrap with LoopClosure, run reconstruct.

    Saves feedforward.zarr to output_dir after inference so downstream stages
    (semantics lift, mesh) can load depth/confidence/pixel data. Returns the
    PointcloudResult and the created Viewer (None unless loop_closure + viz_enabled),
    so callers can keep the viser server reachable after this function returns.

    ``loop_closure`` is either a bool (enable with all LoopClosureConfig defaults) or
    a dict of knobs (submap_size, submap_overlap, scale_method, …); an ``enabled`` key
    in the dict toggles it, defaulting to True when any knobs are given.
    """
    # Heavy dep imports — kept inline so module loads without GPU/model deps
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )
    from collab_splats.pointcloud.feedforward import (
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

    # Select creator class by backend name
    creator_map = {
        "vggtx": VGGTXCreator,
        "mapanything": MapAnythingCreator,
        "vggt_omega": VGGTOmegaCreator,
    }
    # max_points caps the confidence mask during inference — a memory guard, not a preference
    creator = creator_map[backend](max_points=max_points)

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
    result = creator.reconstruct(FrameStore.open(frames_zarr), output_dir)

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
    features_dir: Path,
) -> Path:
    """Extract 2D features for all frames straight from the canonical store.

    Delegates to BaseFeatureExtractor.extract_and_cache_from_zarr, which iterates the
    frames zarr lazily (one chunk at a time) and writes cache_dir/{name}.zarr — no temp
    JPG export, no full-RAM load.
    """
    extractor = _get_extractor(extractor_name)
    cache_dir = features_dir / extractor_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)


def _lift_and_save(
    zarr_path: Path,
    feedforward_zarr: Path,
    output_dir: Path,
    n_components: int | None,
    target_cosine: float | None,
    max_epochs: int,
) -> Path:
    """Load 2D feature cache + FeedforwardResult, lift to 3D, compress, save.

    Writes output_dir/features.zarr (latent codes) and, when compressing,
    output_dir/autoencoder.pt (weights + fit metrics) — the pair a consumer needs
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
    # zarr attrs, which is what tells a reader whether autoencoder.pt is required at all
    # (n_components=None writes full-dim codes and no weights, legitimately).
    write_point_features(output_dir, lifted.detach().cpu().numpy(), ae)
    return output_dir


def _run_tsdf_mesh(
    result: "PointcloudResult",
    feedforward_zarr: Path,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
) -> Path:
    """Fuse depth + RGB from FeedforwardResult into TSDF mesh."""
    from collab_splats.mesh.tsdf import Open3DTSDFFusion
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    # Load depth and RGB from feedforward zarr
    ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)

    depths = ff.depth  # (N, H, W) float32 metres
    if depths is None:
        raise ValueError("FeedforwardResult has no depth — cannot mesh.")

    # images is (N, 3, H, W) torch tensor; convert to (N, H, W, 3) float32 numpy
    if ff.images is None:
        raise ValueError("FeedforwardResult has no images — cannot mesh.")
    imgs = ff.images
    if hasattr(imgs, "numpy"):
        imgs = imgs.numpy()
    rgbs = np.ascontiguousarray(imgs.transpose(0, 2, 3, 1)).astype(np.float32) / 255.0

    # c2w from PointcloudResult extrinsics (w2c → c2w)
    c2w = np.linalg.inv(result.extrinsics)  # (N, 4, 4)
    intrinsics = result.intrinsics  # (N, 3, 3)

    # Run TSDF fusion
    output_dir.mkdir(parents=True, exist_ok=True)
    mesher = Open3DTSDFFusion(
        output_dir=output_dir,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
    )
    mesh_result = mesher.create(depths=depths, rgbs=rgbs, c2w=c2w, intrinsics=intrinsics)
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


def _build_localization_db(feedforward_zarr: Path, extractor_name: str, frames_zarr: Path) -> Path:
    """Build the per-frame local-feature localization cache into feedforward.zarr.

    Loads the FeedforwardResult, runs the local matcher over every DB frame, and persists
    keypoints/descriptors to group local_features/{extractor_name}/reconstruction.
    """
    # Heavy deps kept inline so the module imports without GPU/model libs
    from collab_splats.localization.extractors import BaseLocalExtractor
    from collab_splats.localization.localizer import CameraLocalizer
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.preproc.frame_store import FrameStore

    ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True, load_world_points=True)
    extractor = BaseLocalExtractor.get(extractor_name)()

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
    )
    logger.info("Localization DB built: %s :: local_features/%s", feedforward_zarr, extractor_name)
    return feedforward_zarr


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
    def features_dir(self) -> Path:
        """output_path / features/ — shared 2D feature cache, extractor-scoped subdirs."""
        return Path(self.config["output_path"]) / "features"

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
            frame_proportion=pre_cfg["frame_proportion"],
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

        # BA at the Reconstructor level is not wired — fail loud instead of silently no-op'ing
        if pc_cfg["bundle_adjustment"]:
            raise NotImplementedError(
                "pointcloud.bundle_adjustment is not wired at the Reconstructor level. "
                "Pass bundle_adjustment to the creator config directly for now."
            )

        # Skip if COLMAP + feedforward.zarr both exist and overwrite not requested.
        # Require feedforward.zarr too — if a previous run was partial (zarr missing),
        # we must re-run inference rather than loading stale COLMAP.
        colmap_done = (self.backend_dir / "colmap" / "sparse" / "0" / "cameras.bin").exists()
        zarr_done = (self.backend_dir / "feedforward.zarr").exists()
        if not overwrite and colmap_done and zarr_done:
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
        # Rebuild image_paths from frames.zarr: COLMAP registered names as frame_{source_idx:06d}.jpg
        # (build_pycolmap_reconstruction takes names from FrameStore.export, which is 06d source-idx
        # named), so derive the same names in store order to line up with the reconstruction.
        frame_indices = FrameStore.open(self.frames_zarr).frame_indices()
        image_paths = [Path(f"frame_{int(fi):06d}.jpg") for fi in frame_indices]
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

    def extract_semantics(
        self,
        result: "PointcloudResult | None" = None,
        overwrite: bool = False,
    ) -> Path:
        """Extract 2D features (cached), lift to 3D, compress. Returns lifted zarr dir."""
        sem_cfg = self.config["semantics"]
        extractor_name = sem_cfg["extractor"]
        n_components = sem_cfg["n_components"]

        lifted_dir = self.backend_dir / "semantics" / extractor_name

        # Skip if lifted features already on disk
        if not overwrite and (lifted_dir / "features.zarr").exists():
            logger.info("Lifted features exist at %s, skipping", lifted_dir)
            return lifted_dir

        result = result or self.pointcloud
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # Stage 1: 2D feature extraction (cached at features_dir/extractor)
        zarr_path = self.features_dir / extractor_name / f"{extractor_name}.zarr"
        if overwrite or not zarr_path.exists():
            logger.info("Extracting 2D features with %s", extractor_name)
            zarr_path = _extract_2d_features(extractor_name, self.frames_zarr, self.features_dir)
        else:
            logger.info("2D feature cache hit: %s", zarr_path)

        # Stage 2: Lift to 3D and save
        feedforward_zarr = self.backend_dir / "feedforward.zarr"
        logger.info("Lifting 2D features to 3D pointcloud")
        out_dir = _lift_and_save(
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
        mesh_path = self.backend_dir / "mesh" / "mesh.ply"

        # Skip if mesh already on disk
        if not overwrite and mesh_path.exists():
            logger.info("Mesh exists at %s, skipping", mesh_path)
            return mesh_path

        result = result or self.pointcloud
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
            output_dir=self.backend_dir / "mesh",
            voxel_size=mesh_cfg["voxel_size"],
            sdf_trunc=mesh_cfg["sdf_trunc"],
            depth_trunc=mesh_cfg["depth_trunc"],
        )
        logger.info("Mesh saved to %s", out)
        return out

    def build_localization_db(self, result: "PointcloudResult | None" = None, overwrite: bool = False) -> Path:
        """Build/refresh the per-frame local-feature localization cache in feedforward.zarr."""
        loc_cfg = self.config["localization"]
        extractor_name = loc_cfg["extractor"]

        feedforward_zarr = self.backend_dir / "feedforward.zarr"
        if not feedforward_zarr.exists():
            raise FileNotFoundError(
                f"feedforward.zarr not found at {feedforward_zarr}. "
                "Localization DB requires a feedforward pointcloud stage first."
            )

        # Skip if the DB group already exists and overwrite not requested
        if not overwrite and _localization_db_exists(feedforward_zarr, extractor_name):
            logger.info(
                "Localization DB exists at %s :: local_features/%s, skipping",
                feedforward_zarr,
                extractor_name,
            )
            return feedforward_zarr

        return _build_localization_db(feedforward_zarr, extractor_name, self.frames_zarr)

    def _stage_output_exists(self, stage: str) -> bool:
        """True if `stage`'s on-disk output is already present (lets deps be reused across runs)."""
        # Only preproc/pointcloud are ever depended on; others have no reusable marker.
        if stage == "preproc":
            return self.frames_zarr.exists()
        if stage == "pointcloud":
            colmap_done = (self.backend_dir / "colmap" / "sparse" / "0" / "cameras.bin").exists()
            zarr_done = (self.backend_dir / "feedforward.zarr").exists()
            return colmap_done and zarr_done
        return False

    def run_pipeline(
        self,
        stages: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Run named stages in dependency order.

        Args:
            stages: Subset of ["preproc", "pointcloud", "semantics", "mesh"].
                    Default: all enabled stages from config.
            overwrite: Re-run stages even if output exists.

        Raises:
            ValueError: If stages list violates dependency ordering.
        """
        if stages is None:
            # Build from config enabled flags; preproc + pointcloud always included
            stages = ["preproc", "pointcloud"]
            if self.config["semantics"]["enabled"]:
                stages.append("semantics")
            if self.config["mesh"]["enabled"]:
                stages.append("mesh")
            if self.config["localization"]["enabled"]:
                stages.append("localize")

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

        # Execute stages in canonical order
        result = None
        for stage in [s for s in _STAGE_ORDER if s in stages_set]:
            logger.info("=== Stage: %s ===", stage)
            if stage == "preproc":
                self.preprocess(overwrite=overwrite)
            elif stage == "pointcloud":
                result = self.build_pointcloud(overwrite=overwrite)
            elif stage == "semantics":
                self.extract_semantics(result=result, overwrite=overwrite)
            elif stage == "mesh":
                self.mesh(result=result, overwrite=overwrite)
            elif stage == "localize":
                self.build_localization_db(result=result, overwrite=overwrite)

    def launch_dashboard(self) -> None:
        """Launch interactive dashboard for current reconstruction state."""
        from collab_splats.dashboard.__main__ import (
            main as dashboard_main,  # optional heavy dep; lazy load
        )

        dashboard_main()
