"""5-stage reconstruction pipeline wrapper."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from collab_splats.preproc.frame_store import FrameStore

if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

_FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega"}
_SFM_BACKENDS = {"colmap", "hloc"}
_VALID_METHODS = {"feedforward", "sfm", "nerfstudio"}
_VALID_MESHERS = {"tsdf", "poisson"}
_STAGE_ORDER = ["preprocess", "pointcloud", "semantics", "mesh", "localize"]
_STAGE_DEPS: dict[str, list[str]] = {
    "preprocess": [],
    "pointcloud": ["preprocess"],
    "semantics": ["pointcloud"],
    "mesh": ["pointcloud"],
    "localize": ["pointcloud"],
}


########################################
# Helpers
########################################


def _extract_frames(
    input_path: Path,
    output_dir: Path,
    frame_selection: str,
    frame_proportion: float,
    min_frames: int,
    max_frames: int | None,
    frames_zarr: Path | None = None,
) -> list[Path]:
    """Extract frames from video or copy from image dir into output_dir.

    Also writes frames_zarr — the canonical decode-once keyframe store — alongside
    the JPEGs when a path is given.

    Returns sorted list of extracted frame paths.
    """
    import cv2

    from collab_splats.preproc import get_video_info, sample_frames

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)

    if input_path.is_dir():
        # Copy images from directory; reject non-image extensions
        exts = {".jpg", ".jpeg", ".png"}
        frames = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
        for i, src in enumerate(frames):
            shutil.copy(src, output_dir / f"frame_{i:04d}{src.suffix}")
        if frames_zarr is not None and frames:
            # Read the source images back as RGB arrays for the canonical store
            frame_arrays = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frames]
            records = [{"frame_idx": i, "blur_score": float("nan")} for i in range(len(frame_arrays))]
            prov = {"video_path": str(input_path), "video_mtime": None, "method": "dir", "max_frames": max_frames}
            FrameStore.create(frames_zarr, frame_arrays, records, provenance=prov)
        return sorted(output_dir.iterdir())

    # Video — dispatch to uniform or optical-flow sampling based on frame_selection
    if frame_selection == "optical_flow":
        method = "optical_flow"
        frame_arrays, records = sample_frames(
            str(input_path),
            method="optical_flow",
            max_frames=max_frames if max_frames is not None else 200,
        )
    else:
        # Derive target count from proportion, clamped to [min_frames, max_frames];
        # the uniform sampler spreads that count over the video itself
        method = "uniform"
        total_frames = get_video_info(str(input_path))["total_frames"]
        target_count = max(min_frames, int(total_frames * frame_proportion))
        if max_frames is not None:
            target_count = min(target_count, max_frames)
        frame_arrays, records = sample_frames(
            str(input_path),
            method="uniform",
            max_frames=target_count,
        )

    # Save extracted frames as JPEG files
    paths: list[Path] = []
    for i, frame in enumerate(frame_arrays):
        dest = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(dest), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(dest)

    # Write the canonical frames.zarr alongside the JPEGs (decode-once keyframe store)
    if frames_zarr is not None and frame_arrays:
        prov = {
            "video_path": str(input_path),
            "video_mtime": input_path.stat().st_mtime,
            "method": method,
            "max_frames": max_frames,
        }
        FrameStore.create(frames_zarr, frame_arrays, records, provenance=prov)

    return paths


def _run_feedforward(
    backend: str,
    images_dir: Path,
    output_dir: Path,
    bundle_adjustment: bool,
    loop_closure: bool,
) -> "PointcloudResult":
    """Instantiate feedforward creator, optionally wrap with LoopClosure, run reconstruct.

    Saves feedforward.zarr to output_dir after inference so downstream stages
    (semantics lift, mesh) can load depth/confidence/pixel data.
    """
    # Heavy dep imports — kept inline so module loads without GPU/model deps
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure
    from collab_splats.pointcloud.feedforward import (
        MapAnythingCreator,
        VGGTOmegaCreator,
        VGGTXCreator,
    )

    # Select creator class by backend name
    creator_map = {
        "vggtx": VGGTXCreator,
        "mapanything": MapAnythingCreator,
        "vggt_omega": VGGTOmegaCreator,
    }
    creator = creator_map[backend]()

    # Wrap with loop closure if requested
    if loop_closure:
        creator = LoopClosure(base=creator)

    # reconstruct() expects the root backend dir; build_colmap appends colmap/sparse/0 internally
    result = creator.reconstruct(images_dir, output_dir)

    # Persist FeedforwardResult to feedforward.zarr — required by semantics lift + mesh stages
    ff_outputs = getattr(creator, "outputs", None)
    if ff_outputs is not None:
        zarr_path = output_dir / "feedforward.zarr"
        ff_outputs.save_zarr(zarr_path)
        logger.info("feedforward.zarr saved: %s  (%s pts)", zarr_path, f"{len(ff_outputs.points):,}")
    else:
        logger.warning("Creator has no outputs after reconstruct — feedforward.zarr not saved")

    # Bundle adjustment operates on FeedforwardResult before COLMAP build;
    # at this stage we have PointcloudResult — BA at creator level is not applicable here.
    if bundle_adjustment:
        logger.warning(
            "bundle_adjustment=True is not yet wired at the Reconstructor level; "
            "pass bundle_adjustment to the creator config directly for now."
        )

    # Explicitly release model + GPU memory before next stage (semantics) loads its model
    import torch as _torch

    del creator
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()
    logger.info("Pointcloud model released from GPU")

    return result


def _get_extractor(name: str):
    """Instantiate feature extractor by registry name."""
    from collab_splats.semantics.features import BaseFeatureExtractor

    return BaseFeatureExtractor.get(name)()


def _extract_2d_features(
    extractor_name: str,
    image_paths: list[Path],
    features_dir: Path,
) -> Path:
    """Extract 2D features for all frames, cache to features_dir/{name}/{name}.zarr.

    Delegates to BaseFeatureExtractor.extract_and_cache which handles PIL loading,
    zarr layout (N, D, H_p, W_p), re-entrancy, and progress logging.
    """
    extractor = _get_extractor(extractor_name)
    cache_dir = features_dir / extractor_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    # extract_and_cache returns cache_dir/{name}.zarr
    return extractor.extract_and_cache(image_paths, cache_dir)


def _lift_and_save(
    zarr_path: Path,
    feedforward_zarr: Path,
    output_dir: Path,
    n_components: int | None,
) -> Path:
    """Load 2D feature cache + FeedforwardResult, lift to 3D, compress, save."""
    import numpy as np
    import torch
    import zarr as zarr_lib

    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.pointcloud.utils import lift_features

    # Validate feedforward zarr exists before attempting load
    if not feedforward_zarr.exists():
        raise FileNotFoundError(
            f"feedforward.zarr not found at {feedforward_zarr}. "
            "Run build_pointcloud() with a feedforward backend first."
        )

    # Load feature maps from zarr cache: (N, D, H_p, W_p)
    store = zarr_lib.open(str(zarr_path), mode="r")
    features_arr = store["features"]
    feature_maps = [torch.from_numpy(np.array(features_arr[i])) for i in range(features_arr.shape[0])]

    # Load FeedforwardResult with depth/pixel data for lifting
    ff_result = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)

    # Lift 2D features to 3D: (P, D)
    lifted = lift_features(feature_maps, ff_result)

    # Optional PCA compression via autoencoder
    if n_components is not None:
        import torch as _torch

        from collab_splats.semantics.compression import FeatureAutoencoder

        # Move lifted to GPU for autoencoder training; lift_features returns CPU tensor
        if _torch.cuda.is_available():
            lifted = lifted.cuda()
        ae = FeatureAutoencoder(input_dim=lifted.shape[-1], latent_dim=n_components)
        ae.fit(lifted)  # (N, input_dim) flat tensor
        lifted = ae.per_point_encode(lifted)
        ae.save(output_dir / "compressor.pt")

    # Save lifted features as zarr
    output_dir.mkdir(parents=True, exist_ok=True)
    out_store = zarr_lib.open(str(output_dir / "features.zarr"), mode="w")
    out_store["features"] = lifted.detach().cpu().numpy()
    return output_dir


def _run_tsdf_mesh(
    result: "PointcloudResult",
    feedforward_zarr: Path,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
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


def _build_localization_db(feedforward_zarr: Path, extractor_name: str, radius: float) -> Path:
    """Build the per-frame local-feature localization cache into feedforward.zarr.

    Loads the FeedforwardResult, runs the local matcher over every DB frame, and persists
    keypoints/descriptors to group local_features/{extractor_name}/reconstruction.
    """
    # Heavy deps kept inline so the module imports without GPU/model libs
    from collab_splats.localization.extractors import BaseLocalExtractor
    from collab_splats.localization.localizer import CameraLocalizer
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)
    extractor = BaseLocalExtractor.get(extractor_name)()

    # from_feedforward is cache-first: on miss it runs GPU extraction + save_index
    CameraLocalizer.from_feedforward(
        ff,
        extractor=extractor,
        extractor_name=extractor_name,
        zarr_path=feedforward_zarr,
        radius=radius,
    )
    logger.info("Localization DB built: %s :: local_features/%s", feedforward_zarr, extractor_name)
    return feedforward_zarr


########################################
# Reconstructor
########################################


class Reconstructor:
    """5-stage environment reconstruction pipeline: preprocess → pointcloud → semantics / mesh / localize."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with validated config dict."""
        self.config = self.validate_config(config)
        self.pointcloud: PointcloudResult | None = None

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

        pc = config.get("pointcloud", {})
        method = pc.get("method", "feedforward")
        backend = pc.get("backend", "vggtx")

        if method not in _VALID_METHODS:
            raise ValueError(f"pointcloud.method must be one of {_VALID_METHODS}, got '{method}'")

        if method == "feedforward" and backend not in _FEEDFORWARD_BACKENDS:
            raise ValueError(
                f"pointcloud.backend must be one of {_FEEDFORWARD_BACKENDS} "
                f"for method='feedforward', got '{backend}'"
            )
        if method == "sfm" and backend not in _SFM_BACKENDS:
            raise ValueError(f"pointcloud.backend must be one of {_SFM_BACKENDS} " f"for method='sfm', got '{backend}'")

        mesh_cfg = config.get("mesh", {})
        mesher = mesh_cfg.get("mesher", "tsdf")
        if mesher not in _VALID_MESHERS:
            raise ValueError(f"mesh.mesher must be one of {_VALID_MESHERS}, got '{mesher}'")

        return config

    @classmethod
    def from_config_file(
        cls,
        dataset: str,
        config_dir: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> Reconstructor:
        """Create Reconstructor from YAML config hierarchy.

        Args:
            dataset: Dataset name (matches datasets/<dataset>.yaml).
            config_dir: Directory containing base.yaml and datasets/.
            overrides: Optional runtime overrides applied after merge.
        """
        from collab_splats.wrapper.config import (
            ConfigLoader,  # optional heavy dep; lazy load
        )

        loader = ConfigLoader(config_dir)
        config = loader.load(dataset=dataset, overrides=overrides)
        return cls(config)

    ########################################
    # Path properties
    ########################################

    @property
    def backend_dir(self) -> Path:
        """output_path / backend — e.g. out/vggtx/. Backend subdir for all stage 2+ artifacts."""
        backend = self.config["pointcloud"].get("backend", "nerfstudio")
        return Path(self.config["output_path"]) / backend

    @property
    def images_dir(self) -> Path:
        """output_path / images/ — shared frame store across all backends."""
        return Path(self.config["output_path"]) / "images"

    @property
    def frames_zarr(self) -> Path:
        """output_path / frames.zarr — canonical decode-once keyframe store for this run."""
        return Path(self.config["output_path"]) / "frames.zarr"

    @property
    def features_dir(self) -> Path:
        """output_path / features/ — shared 2D feature cache, extractor-scoped subdirs."""
        return Path(self.config["output_path"]) / "features"

    ########################################
    # Stage stubs (implemented in later tasks)
    ########################################

    def preprocess(self, overwrite: bool = False) -> Path:
        """Extract frames from input video/dir into images_dir and frames.zarr."""
        # Skip if frames already exist and overwrite not requested
        if not overwrite and self.images_dir.exists() and any(self.images_dir.iterdir()):
            logger.info("Frames already extracted at %s, skipping preprocess", self.images_dir)
            return self.images_dir

        if overwrite and self.images_dir.exists():
            shutil.rmtree(self.images_dir)

        pre_cfg = self.config.get("preprocessing", {})
        extracted = _extract_frames(
            input_path=Path(self.config["input_path"]),
            output_dir=self.images_dir,
            frame_selection=pre_cfg.get("frame_selection", "fps"),
            frame_proportion=pre_cfg.get("frame_proportion", 0.1),
            min_frames=pre_cfg.get("min_frames", 300),
            max_frames=pre_cfg.get("max_frames"),
            frames_zarr=self.frames_zarr,
        )
        logger.info(
            "Preprocessing complete: %d frames at %s",
            len(extracted),
            self.images_dir,
        )
        return self.images_dir

    def build_pointcloud(self, overwrite: bool = False) -> "PointcloudResult":
        """Run pointcloud stage. Sets self.pointcloud, returns PointcloudResult."""
        pc_cfg = self.config.get("pointcloud", {})
        method = pc_cfg.get("method", "feedforward")

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
            result = _run_feedforward(
                backend=pc_cfg.get("backend", "vggtx"),
                images_dir=self.images_dir,
                output_dir=self.backend_dir,
                bundle_adjustment=pc_cfg.get("bundle_adjustment", False),
                loop_closure=pc_cfg.get("loop_closure", False),
            )

        # Apply cleaning step if enabled
        clean_cfg = pc_cfg.get("clean", {})
        if clean_cfg.get("enabled", True):
            result = self._clean_pointcloud(result, clean_cfg)

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
        # Gather image paths from images_dir matching the reconstruction
        image_paths = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
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
        if cfg.get("outlier_removal", True):
            _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            inlier_set = set(inlier_idx)
            for i, pid in enumerate(point3d_ids):
                if i not in inlier_set:
                    result.reconstruction.delete_point3D(pid)

        # Optional voxel downsampling (affects visualization/density; no structural change)
        voxel_size = cfg.get("voxel_size")
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size)

        logger.info("Pointcloud after cleaning: %d points", result.reconstruction.num_points3D())
        return result

    def _write_transforms_json(self, result: "PointcloudResult") -> None:
        """Write nerfstudio-compatible transforms.json from PointcloudResult."""
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

        frames = []
        for img_path, K, pose in zip(image_paths, intrinsics, c2w):
            frames.append(
                {
                    "file_path": f"../images/{img_path.name}",
                    "fl_x": float(K[0, 0]),
                    "fl_y": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "transform_matrix": pose.tolist(),
                }
            )

        self.backend_dir.mkdir(parents=True, exist_ok=True)
        out = self.backend_dir / "transforms.json"
        out.write_text(json.dumps({"camera_model": "PINHOLE", "frames": frames}, indent=2))
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

        ns_cfg = self.config.get("nerfstudio", {})
        sfm_tool = ns_cfg.get("sfm_tool", "hloc")
        train_method = ns_cfg.get("train_method", "rade-features")

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
        sem_cfg = self.config.get("semantics", {})
        extractor_name = sem_cfg.get("extractor", "dinov2")
        n_components = sem_cfg.get("n_components", 64)

        lifted_dir = self.backend_dir / "semantics" / extractor_name

        # Skip if lifted features already on disk
        if not overwrite and (lifted_dir / "features.zarr").exists():
            logger.info("Lifted features exist at %s, skipping", lifted_dir)
            return lifted_dir

        result = result or self.pointcloud
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        image_paths = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))

        # Stage 1: 2D feature extraction (cached at features_dir/extractor)
        zarr_path = self.features_dir / extractor_name / f"{extractor_name}.zarr"
        if overwrite or not zarr_path.exists():
            logger.info("Extracting 2D features with %s", extractor_name)
            zarr_path = _extract_2d_features(extractor_name, image_paths, self.features_dir)
        else:
            logger.info("2D feature cache hit: %s", zarr_path)

        # Stage 2: Lift to 3D and save
        feedforward_zarr = self.backend_dir / "feedforward.zarr"
        logger.info("Lifting 2D features to 3D pointcloud")
        out_dir = _lift_and_save(zarr_path, feedforward_zarr, lifted_dir, n_components)
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

        mesh_cfg = self.config.get("mesh", {})
        out = _run_tsdf_mesh(
            result=result,
            feedforward_zarr=feedforward_zarr,
            output_dir=self.backend_dir / "mesh",
            voxel_size=mesh_cfg.get("voxel_size", 0.01),
            sdf_trunc=mesh_cfg.get("sdf_trunc", 0.04),
        )
        logger.info("Mesh saved to %s", out)
        return out

    def build_localization_db(self, result: "PointcloudResult | None" = None, overwrite: bool = False) -> Path:
        """Build/refresh the per-frame local-feature localization cache in feedforward.zarr."""
        loc_cfg = self.config.get("localization", {})
        extractor_name = loc_cfg.get("extractor", "loma")
        radius = loc_cfg.get("radius", 8.0)

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

        return _build_localization_db(feedforward_zarr, extractor_name, radius)

    def run_pipeline(
        self,
        stages: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Run named stages in dependency order.

        Args:
            stages: Subset of ["preprocess", "pointcloud", "semantics", "mesh"].
                    Default: all enabled stages from config.
            overwrite: Re-run stages even if output exists.

        Raises:
            ValueError: If stages list violates dependency ordering.
        """
        if stages is None:
            # Build from config enabled flags; preprocess + pointcloud always included
            stages = ["preprocess", "pointcloud"]
            if self.config.get("semantics", {}).get("enabled", False):
                stages.append("semantics")
            if self.config.get("mesh", {}).get("enabled", False):
                stages.append("mesh")
            if self.config.get("localization", {}).get("enabled", False):
                stages.append("localize")

        # Validate stage dependencies before starting any work
        stages_set = set(stages)
        for stage in stages:
            for dep in _STAGE_DEPS.get(stage, []):
                if dep not in stages_set:
                    raise ValueError(
                        f"Stage '{stage}' requires '{dep}' but '{dep}' is not in stages={stages}. "
                        f"Add '{dep}' to the stages list."
                    )

        # Execute stages in canonical order
        result = None
        for stage in [s for s in _STAGE_ORDER if s in stages_set]:
            logger.info("=== Stage: %s ===", stage)
            if stage == "preprocess":
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
