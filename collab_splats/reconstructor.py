"""
Config-driven reconstruction pipeline, one output tree per scene.

- stages: preproc, pointcloud, refine, semantics, splats, mesh, localize, quality report
- configs/base.yaml holds every default; a run config merges over it
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import warnings
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d
import torch
import yaml
import zarr
from mergedeep import merge

from collab_splats.geometry.bundle_adjustment import (
    BundleAdjustment,
    BundleAdjustmentConfig,
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
    prepare_mesh,
)
from collab_splats.pointcloud import BaseFeedforwardCreator, get_creator
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.sfm import SFM_CREATORS
from collab_splats.pointcloud.utils import clean_pointcloud, frame_depths
from collab_splats.preproc import frames, get_video_info
from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.sampling import (
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
from collab_splats.preproc.undistort import calibrate_camera, undistort_frames
from collab_splats.preproc.viz import plot_motion, plot_photometric
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.semantics.lifting import lift_features
from collab_splats.semantics.segmentation import sky_masks
from collab_splats.semantics.store import (
    extract_feature_cache,
    valid_feature_cache,
    write_point_features,
)
from collab_splats.utils.colmap import write_colmap_reconstruction
from collab_splats.utils.io import read_image, write_json
from collab_splats.utils.torch_utils import (
    get_device,
    load_features,
    pytorch_gc,
    to_numpy,
)

if TYPE_CHECKING:
    from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# Stage -> stages it needs; dict order is run order
STAGES: dict[str, tuple[str, ...]] = {
    "preproc": (),
    "pointcloud": ("preproc",),
    "refine": ("pointcloud",),
    "semantics": ("pointcloud",),
    "splats": ("pointcloud",),
    "mesh": ("pointcloud",),
    "localize": ("pointcloud",),
    "reconstruction_quality_report": ("pointcloud",),
}

# Stages nothing depends on; only these re-run alone against processed outputs
LEAF_STAGES = frozenset(s for s in STAGES if not any(s in deps for deps in STAGES.values()))


########################################
# Helpers
########################################


def store_rows(images_dir: Path, names: Sequence[Path | str]) -> list[int]:
    """
    Rows of the images/ store holding each named frame, in the order named.

    - pointcloud.zarr may hold fewer frames than images/: incremental sfm drops unregistered ones
    - joined on the source frame index in each name, never on row position or extension
    - the 2D feature cache shares these rows: it is extracted over `frames.frame_paths(images_dir)`

    Args:
        images_dir: the scene's images/ keyframe store.
        names: frame_NNNNNN with any extension or none, e.g. a result's image_paths.

    Returns:
        Indices into `frames.frame_paths(images_dir)`, one per name.

    Raises:
        KeyError: when a named frame is not in images/.
    """
    # Map each stored frame's source index to its row
    store_paths = frames.frame_paths(images_dir)
    rows_by_frame_idx = {frames.frame_idx_from_path(p): row for row, p in enumerate(store_paths)}
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


def _load_frame(
    features: zarr.Array,
    rows: list[int],
    ae: FeatureAutoencoder | None,
    i: int,
) -> torch.Tensor:
    """
    Pointcloud frame i's patch map from the scene-level cache, AE-encoded when given.

    - pointcloud frame i is store row rows[i]
    - returns (D, H_p, W_p) raw fp16 CPU, or (latent_dim, H_p, W_p) float32 codes on the AE's device
    """
    fmap = torch.from_numpy(features[rows[i]])
    if ae is None:
        return fmap

    # Encode on the AE's device; lift_features moves it on as float32
    device = next(ae.parameters()).device
    fmap = fmap.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        return ae.encode(fmap)


def _localization_db_exists(pointcloud_zarr: Path, extractor_name: str) -> bool:
    """
    True if the local-feature DB group already exists in pointcloud.zarr.

    - a missing store, or one that is not a group, reads as absent
    """
    # Open read-only; a missing or non-group store has no DB
    try:
        store = zarr.open_group(str(pointcloud_zarr), mode="r")
    except (FileNotFoundError, zarr.errors.NodeNotFoundError, zarr.errors.ContainsArrayError):
        return False

    return (
        "local_features" in store
        and extractor_name in store["local_features"]
        and "reconstruction" in store["local_features"][extractor_name]
    )


def _build_localization_db(
    pointcloud_zarr: Path,
    extractor_name: str,
    images_dir: Path,
    top_k: int,
) -> None:
    """
    Build the per-frame local-feature localization cache into pointcloud.zarr.

    - persists keypoints/descriptors to local_features/{extractor_name}/reconstruction
    - always rebuilds: an existing group for the extractor is dropped first
    - top_k is the pairwise (vismatch) fan-out; the descriptor path ignores it
    - a missing pointcloud.zarr raises FileNotFoundError; nothing is created in its place
    """
    # Drop the stale group; from_feedforward has no overwrite and always cache-hits one
    store = zarr.open_group(str(pointcloud_zarr), mode="r+")
    rec_key = f"local_features/{extractor_name}/reconstruction"

    if rec_key in store:
        del store[rec_key]

    # Load the reconstruction and the matcher
    ff = PointcloudResult.load_zarr(pointcloud_zarr, load_images=True, load_world_points=True)
    extractor = LocalMatcher(extractor_name)

    # Lazily read the store's frames in the zarr's image_paths order
    all_paths = frames.frame_paths(images_dir)
    rows = store_rows(images_dir, ff.image_paths)
    paths = [all_paths[row] for row in rows]
    images = (read_image(p) for p in paths)

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


########################################
# Reconstructor
########################################


class Reconstructor:
    """
    Runs the pipeline stages for one scene from a merged, validated config.

    - preproc writes images/; pointcloud writes the backend's zarr and COLMAP model
    - every later stage reads the pointcloud outputs under backend_dir
    - path properties are the single spelling of the output layout
    """

    def __init__(self, config: dict[str, Any], base_config: Path | None = None) -> None:
        """
        Merge config over base.yaml defaults, validate, and store.

        Args:
            config: run config; any key it omits comes from base_config.
            base_config: defaults YAML to merge over; None reads configs/base.yaml.
        """
        # Default to the repo's configs/base.yaml
        if base_config is None:
            base_config = Path(__file__).parents[1] / "configs" / "base.yaml"

        # Deep-merge the caller's config over base.yaml, the single source of defaults
        text = Path(base_config).read_text()
        defaults = yaml.safe_load(text) or {}
        merged = merge({}, defaults, config)

        # Validate, then store the fully-populated config
        self.config = self.validate_config(merged)
        self._result: PointcloudResult | None = None

        # Viser server kept by pointcloud() under viz + loop closure, reachable after run() returns
        self.viewer: Viewer | None = None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """
        Reject a config that would fail mid-run, before any stage starts.

        - only cross-field checks: required paths, method/backend pairing, BA/LC/sfm/LoGeR exclusions
        - per-argument bounds live in the creators and create_tsdf_mesh
        - pointcloud.loop_closure is normalized in place to a dict carrying `enabled`

        Args:
            config: config already merged over base.yaml.

        Returns:
            The same config, loop_closure normalized.

        Raises:
            ValueError: a required field is missing or two settings cannot run together.
        """
        # Required top-level fields
        for field in ("input_path", "output_path"):
            if config.get(field) is None:
                raise ValueError(f"Reconstructor config missing required field: '{field}'")

        # Method and backend must name a registered creator
        pc = config["pointcloud"]
        method = pc["method"]
        backend = pc["backend"]
        backends = {"feedforward": set(BaseFeedforwardCreator._registry), "sfm": set(SFM_CREATORS)}

        if method not in backends:
            raise ValueError(f"pointcloud.method must be one of {sorted(backends)}, got {method!r}")

        if backend not in backends[method]:
            raise ValueError(
                f"pointcloud.backend must be one of {sorted(backends[method])} for method {method!r}, got {backend!r}"
            )

        # Normalize loop_closure to a dict carrying `enabled`
        lc = pc["loop_closure"]
        lc = dict(lc) if isinstance(lc, dict) else {"enabled": bool(lc)}
        lc.setdefault("enabled", True)
        unknown = set(lc) - {"enabled"} - {f.name for f in dataclasses.fields(LoopClosureConfig)}

        if unknown:
            raise ValueError(f"pointcloud.loop_closure has unknown keys {sorted(unknown)}")

        pc["loop_closure"] = lc

        # Refuse BA with LC: submaps lack the per-frame model tensors BA tracks need
        if pc["bundle_adjustment"] and lc["enabled"]:
            raise ValueError(
                "pointcloud.bundle_adjustment and pointcloud.loop_closure are mutually "
                "exclusive — BA needs per-frame model tensors that LC submaps do not carry."
            )

        # Refuse BA with sfm: every sfm mapper runs its own
        if method == "sfm" and pc["bundle_adjustment"]:
            raise ValueError(
                "pointcloud.bundle_adjustment is not supported with method: sfm — "
                "every sfm backend runs its own bundle adjustment"
            )

        # Refuse LC with sfm: LC wraps a feedforward creator in sequential submaps
        if method == "sfm" and lc["enabled"]:
            raise ValueError(
                "pointcloud.loop_closure is not supported with method: sfm — "
                "sfm backends map the whole frame set at once, not in sequential submaps"
            )

        # Refuse LC with LoGeR: no LC verify thresholds are calibrated for it
        if backend == "loger" and lc["enabled"]:
            raise ValueError(
                "pointcloud.loop_closure is not supported with backend 'loger' — its windowed TTT "
                "memory already carries state across frames; use vggt_omega, vggtx or mapanything"
            )

        return config

    ########################################
    # Path properties
    ########################################

    @staticmethod
    def run_config_path(output_path: Path | str, backend: str) -> Path:
        """
        Recorded config of one backend's run; static so a re-run can find it before construction.

        Args:
            output_path: scene output directory.
            backend: pointcloud backend the run used.

        Returns:
            <output_path>/<backend>/run_config.yaml.
        """
        return Path(output_path) / backend / "run_config.yaml"

    @property
    def backend_dir(self) -> Path:
        """
        Per-backend output directory; every artifact after preproc lands here.

        Returns:
            <output_path>/<pointcloud.backend>.
        """
        return Path(self.config["output_path"]) / self.config["pointcloud"]["backend"]

    @property
    def images_dir(self) -> Path:
        """
        Keyframe store written by preproc; shared by every backend.

        Returns:
            <output_path>/images.
        """
        return Path(self.config["output_path"]) / "images"

    @property
    def pointcloud_zarr(self) -> Path:
        """
        Reconstruction store written by the pointcloud stage.

        Returns:
            <backend_dir>/pointcloud.zarr.
        """
        return self.backend_dir / "pointcloud.zarr"

    @property
    def colmap_model_dir(self) -> Path:
        """
        Binary COLMAP model written beside the zarr.

        Returns:
            <backend_dir>/colmap/sparse/0.
        """
        return self.backend_dir / "colmap" / "sparse" / "0"

    @property
    def sparse_ply(self) -> Path:
        """
        Point cloud PLY written by pointcloud and rewritten by refine.

        Returns:
            <backend_dir>/sparse_pc.ply.
        """
        return self.backend_dir / "sparse_pc.ply"

    @property
    def semantics_cache_dir(self) -> Path:
        """
        Scene-level 2D feature cache, one `<extractor>.zarr` each.

        - frames alone determine it, so every backend lifts from the same cache

        Returns:
            <output_path>/semantics.
        """
        return Path(self.config["output_path"]) / "semantics"

    @property
    def outputs(self) -> dict[str, Path]:
        """
        Marker file per stage; a stage is done when its marker exists.

        - localize writes into pointcloud.zarr, so done() checks it separately

        Returns:
            Stage name to its marker path.
        """
        extractor = self.config["semantics"]["extractor"]
        return {
            "preproc": self.images_dir,
            "pointcloud": self.pointcloud_zarr,
            "refine": self.backend_dir / "colmap" / "refine.json",
            "semantics": self.backend_dir / "semantics" / f"{extractor}_lifted.zarr",
            "splats": self.backend_dir / "splats" / "ckpt.pt",
            "mesh": self.backend_dir / "mesh.ply",
            "reconstruction_quality_report": self.backend_dir / "reconstruction_quality_report.json",
        }

    ########################################
    # Stage dispatch
    ########################################

    def done(self, stage: str) -> bool:
        """
        Whether a stage's output is on disk.

        Args:
            stage: a key of STAGES.

        Returns:
            True when the marker exists; pointcloud also needs the COLMAP model.
        """
        # localize lives inside pointcloud.zarr; pointcloud also needs its COLMAP model
        if stage == "localize":
            matcher = self.config["localization"]["matcher"]
            return self.pointcloud_zarr.exists() and _localization_db_exists(self.pointcloud_zarr, matcher)

        if stage == "pointcloud":
            return self.pointcloud_zarr.exists() and self.colmap_model_dir.exists()

        return self.outputs[stage].exists()

    @property
    def result(self) -> PointcloudResult:
        """
        Points and cameras from pointcloud.zarr, loaded once per stage write.

        - dense per-frame arrays stay on disk; stages needing them load the zarr themselves

        Returns:
            The cached PointcloudResult without depth, world points, confidence or pixel indices.
        """
        if self._result is None:
            self._result = PointcloudResult.load_zarr(
                self.pointcloud_zarr,
                load_depth=False,
                load_world_points=False,
                load_confidence=False,
                load_pixel_indices=False,
            )

        return self._result

    def run(self, stages: list[str] | None = None, overwrite: bool = False) -> None:
        """
        Run stages in table order.

        - None takes every stage the config enables; preproc, pointcloud and the report always run
        - a dependency is met by this run or by its output on disk
        - a done stage is skipped, except a named leaf, which raises unless overwrite

        Args:
            stages: stage names, any order.
            overwrite: rebuild stages whose output exists.

        Raises:
            ValueError: on an unknown stage, an unmet dependency, or a named leaf already done.
        """
        named = stages is not None

        # Default stage set from the config's enable flags
        if stages is None:
            enabled = {
                "refine": self.config["pointcloud"]["bundle_adjustment"],
                "semantics": self.config["semantics"]["enabled"],
                "splats": self.config["splats"]["enabled"],
                "mesh": self.config["mesh"]["enabled"],
                "localize": self.config["localization"]["enabled"],
            }
            stages = [s for s in STAGES if enabled.get(s, True)]

        # Refuse unknown stages and unmet dependencies before any work
        unknown = sorted(set(stages) - set(STAGES))

        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; valid: {list(STAGES)}")

        for stage in stages:
            for dep in STAGES[stage]:
                if dep not in stages and not self.done(dep):
                    raise ValueError(f"stage '{stage}' requires '{dep}', which is neither in this run nor on disk")

            # A named leaf that is already done refuses before any stage runs
            if named and stage in LEAF_STAGES and self.done(stage) and not overwrite:
                raise ValueError(f"stage '{stage}' output already exists; pass overwrite=True to replace it")

        # Run in table order; skip done stages
        for stage in [s for s in STAGES if s in stages]:
            if self.done(stage) and not overwrite:
                logger.info("stage %s done, skipping", stage)
                continue

            logger.info("=== Stage: %s ===", stage)
            getattr(self, stage)()

    ########################################
    # Pipeline stages
    ########################################

    def preproc(self) -> None:
        """
        Select keyframes from the input into images/.

        - video: measure quality, sample, write, plot the report with the kept frames marked
        - directory: every image, in filename order, source index = position
        - undistort self-calibrates from the written frames, then rewrites them

        Raises:
            ValueError: an unknown preproc.frame_selection, or no frame survives selection.
            FileNotFoundError: an input directory with no images, or a missing input video.
        """
        cfg = self.config["preproc"]
        input_path = Path(self.config["input_path"])

        # Clear a previous store; the stage always rebuilds
        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)

        # No quality report unless the input is a video
        report = None

        # Directory input: take every image
        if input_path.is_dir():
            rgbs = frames.read_frames(input_path)
            records = [{"frame_idx": i} for i in range(len(rgbs))]

        # Video input: probe (fails fast on a bad path), measure, then sample
        else:
            total = get_video_info(str(input_path))["total_frames"]
            report_path = self.images_dir.parent / "video_quality_report.json"
            report = load_video_quality(input_path, report_path, workers=cfg["n_workers"])
            common = {"report": report, "quality": cfg["quality"], "max_frames": cfg["max_frames"]}

            # Each selection method gets only its own knobs
            if cfg["frame_selection"] == "fps":
                rgbs, records = sample_fps(
                    str(input_path),
                    fps=cfg["fps"],
                    min_frames=cfg["min_frames"],
                    on_empty_slot=cfg["on_empty_slot"],
                    **common,
                )
            elif cfg["frame_selection"] == "uniform":
                rgbs, records = sample_uniform(str(input_path), **common)
            elif cfg["frame_selection"] == "optical_flow":
                rgbs, records = sample_optical_flow(str(input_path), **common)
            else:
                raise ValueError(
                    f"preproc.frame_selection must be fps, uniform or optical_flow, got {cfg['frame_selection']!r}"
                )

            # Refuse an empty store; it would only surface downstream as a missing file
            if not len(rgbs):
                raise ValueError(f"0 of {total} frames selected from {input_path}; see {report_path}")

        # Write the selected frames under their source indices
        idxs = [r["frame_idx"] for r in records]
        frames.write_frames(self.images_dir, rgbs, idxs)

        # Undistort from the written frames, then rewrite them on the new framing
        if cfg["undistort"]:
            camera = calibrate_camera(self.images_dir)
            rgbs, _ = undistort_frames(rgbs, camera)
            frames.write_frames(self.images_dir, rgbs, idxs)

        # Plot the quality report with the kept frames marked
        if report is not None:
            plot_photometric(report, self.images_dir.parent, selected=idxs)
            plot_motion(report, self.images_dir.parent, selected=idxs)

        logger.info("preproc: %d frames in %s", len(idxs), self.images_dir)

    def pointcloud(self) -> None:
        """
        Reconstruct images/ into pointcloud.zarr, the COLMAP model and sparse_pc.ply.

        - feedforward optionally wraps the creator in loop closure, with a live viewer when viz is on
        - sfm is experimental; its creator writes its own subset and alignment attrs
        """
        cfg = self.config["pointcloud"]
        backend = cfg["backend"]

        # Build the feedforward creator, optionally inside loop closure
        if cfg["method"] == "feedforward":
            creator_cls = get_creator(backend)
            creator = creator_cls(
                max_points=cfg["max_points"],
                min_views=cfg["min_views"],
                mv_rel_thresh=cfg["mv_rel_thresh"],
                clean=cfg["clean"]["enabled"],
                **cfg[backend],
            )

            # Wrap in loop closure when enabled; the remaining keys are its config
            lc = dict(cfg["loop_closure"])

            if lc.pop("enabled"):
                lc_config = LoopClosureConfig(**lc) if lc else None
                creator = LoopClosure(base=creator, config=lc_config)

                # Viewer shows each loop edge live; heavy websocket dep, so imported here
                if cfg["viz"]["enabled"]:
                    try:
                        from collab_splats.viewer import Viewer
                    except ImportError as exc:
                        raise ImportError("pointcloud.viz needs viser; install it or set viz.enabled: false") from exc

                    self.viewer = Viewer(port=cfg["viz"]["port"])
                    creator.viz = self.viewer
                    creator.config.loop_edge_timing = "live"

        # Build the sfm creator from its block
        else:
            warnings.warn(
                "pointcloud.method='sfm' is experimental and not production-tested.", UserWarning, stacklevel=2
            )
            creator = SFM_CREATORS[backend](clean=cfg["clean"]["enabled"], max_points=cfg["max_points"], **cfg[backend])

        # Drop a stale refine marker before touching the artifacts it describes
        self.outputs["refine"].unlink(missing_ok=True)

        # Reconstruct; the creator writes the COLMAP model itself
        result = creator.create_pointcloud(self.images_dir, self.backend_dir, self.colmap_model_dir)

        # Provenance attrs; an sfm creator supplies its own subset and alignment attrs
        if cfg["method"] == "feedforward":
            attrs = {"method": "feedforward", "backend": backend}
        else:
            attrs = {"backend": backend, **creator.attrs}

        # Persist the zarr and the PLY
        result.save_zarr(self.pointcloud_zarr, extra_attrs=attrs)
        result.write_ply(self.sparse_ply)
        logger.info("pointcloud: %d pts in %s", len(result.points), self.pointcloud_zarr)

        # Free the model before the next stage loads its own
        del creator, result
        self._result = None
        pytorch_gc()

    def refine(self) -> None:
        """
        Bundle-adjust the feedforward poses, then rewrite every pose-derived artifact.

        - points are re-derived, re-cleaned and re-capped, so their count can change
        - the zarr is rewritten whole, which drops any localization DB built on the old points

        Raises:
            ValueError: pointcloud.method is sfm.
        """
        cfg = self.config["pointcloud"]

        if cfg["method"] == "sfm":
            raise ValueError("refine is not supported for pointcloud.method: sfm")

        # Refine poses, then re-derive the points under the new cameras
        ff = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=True)
        ba_cfg = BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir)
        ba = BundleAdjustment(ba_cfg)
        extrinsics, intrinsics = ba.refine(
            ff.images, ff.confidence, ff.world_points, ff.extrinsics, ff.model_intrinsics, ff.image_paths
        )
        ff = dataclasses.replace(ff, extrinsics=extrinsics, model_intrinsics=intrinsics, intrinsics=None)
        ff = ff.reproject()

        # Re-clean and re-cap under the refined cameras
        n_before = len(ff.points)
        ff = clean_pointcloud(ff, remove_outliers=cfg["clean"]["enabled"], max_points=cfg["max_points"])
        logger.info("refine: %d of %d pts kept after clean + cap", len(ff.points), n_before)

        # Rewrite the zarr with its provenance attrs, then the COLMAP model and PLY
        store = zarr.open_group(str(self.pointcloud_zarr), mode="r")
        attrs = dict(store.attrs)
        ff.save_zarr(self.pointcloud_zarr, extra_attrs=attrs)
        recon = ff.to_colmap()
        write_colmap_reconstruction(recon, self.colmap_model_dir)
        ff.write_ply(self.sparse_ply)
        self._result = None

        # Marker last: BA config and loss history
        marker = self.outputs["refine"]
        marker.parent.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(ba_cfg).items()}
        write_json(marker, {"config": config, "loss_history": ba.loss_history, "n_frames": len(ff.image_paths)})

    def semantics(self) -> None:
        """
        Extract 2D features into the scene cache, compress them per frame, lift the codes onto the points.

        - the 2D cache is reused when valid (name, frame count, extractor_kwargs); a hit skips the model load
        - AE stored as `<extractor>.zarr/autoencoder.pt`; re-extraction wipes it, a new n_components refits
        - lift reads one frame at a time; no all-frames RAM or full-width (P, D)
        - lifted rows follow the zarr's own frames, which may be a subset of images/
        - lifted store written atomically with its own autoencoder.pt (pushed; the 2D cache is not)
        """
        cfg = self.config["semantics"]
        extractor_kwargs = cfg["extractor_kwargs"]

        # 2D features for every images/ frame; the model loads only when the cache is invalid
        self.semantics_cache_dir.mkdir(parents=True, exist_ok=True)
        cache = valid_feature_cache(self.semantics_cache_dir, cfg["extractor"], self.images_dir, extractor_kwargs)

        if cache is None:
            extractor_cls = BaseFeatureExtractor.get(cfg["extractor"])
            extractor = extractor_cls(**extractor_kwargs)
            cache = extract_feature_cache(extractor, self.images_dir, self.semantics_cache_dir, extractor_kwargs)

            # Free the extractor's GPU memory now; a forward hook can hold it in a reference cycle
            del extractor
            pytorch_gc()

        # Scene cache read lazily; pick the zarr's frames, in the zarr's order
        features = zarr.open(str(cache), mode="r")["features"]
        ff = PointcloudResult.load_zarr(self.pointcloud_zarr, load_world_points=False)
        rows = store_rows(self.images_dir, ff.image_paths)

        # Reuse the AE stored with the 2D cache when its width matches, else fit and store it
        ae = None

        if cfg["n_components"] is not None:
            ae_path = cache / "autoencoder.pt"

            if ae_path.exists():
                ae = FeatureAutoencoder.load(ae_path)

            if ae is not None and ae.latent_dim == cfg["n_components"]:
                logger.info("autoencoder cache hit: %s", ae_path)
            else:
                samples = load_features(features, get_device())
                ae = FeatureAutoencoder(input_dim=samples.shape[1], latent_dim=cfg["n_components"])
                ae.fit(samples, epochs=cfg["max_epochs"], target_cosine=cfg["target_cosine"])
                ae.save(ae_path)

                # Free the samples before the lift
                del samples
                pytorch_gc()

            ae.to(get_device())

        # Lift per-frame codes (or full-width maps when uncompressed) to points
        frame_features = partial(_load_frame, features, rows, ae)
        lifted = lift_features(frame_features, ff)

        # Write the per-point codes beside the lifted-store marker
        codes = to_numpy(lifted)
        write_point_features(self.outputs["semantics"], codes, ae)

    def mesh(self) -> None:
        """
        Fuse depth and RGB into a TSDF mesh.ply, clean and prepare it, and optionally texture it.

        - feedforward source: pointcloud.zarr depth, confidence-masked and lifted to the frame grid
        - splats source: the checkpoint's own renders and poses; the splats stage is never auto-run
        - mask_sky zeroes sky depth on either source before fusion
        - mesh.ply is the prepared mesh (filled, decimated, manifold); texture/ is its UV bake

        Raises:
            FileNotFoundError: the splats source has no ckpt.pt on disk.
            ValueError: mesh.source is neither feedforward nor splats, or sky masks mismatch the depths.
        """
        cfg = self.config["mesh"]
        source = cfg["source"]

        # Splats source: renders at frame resolution, with the poses they were rendered from
        if source == "splats":
            # gsplat is CUDA-only; import lazily so Reconstructor stays importable without it
            try:
                from collab_splats.splats.checkpoint import render_tsdf_inputs
            except ImportError as exc:
                raise ImportError("mesh.source: splats needs gsplat; see setup.sh") from exc

            # Refuse a missing checkpoint; the splats stage is never auto-run
            ckpt = self.outputs["splats"]

            if not ckpt.exists():
                raise FileNotFoundError(f"mesh.source: splats needs {ckpt}; run the splats stage first")

            depths, rgbs, c2w, intrinsics, image_ids = render_tsdf_inputs(ckpt, self.images_dir)

        # Feedforward source: the zarr's depth on the zarr's own frames
        elif source == "feedforward":
            ff = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=False, load_world_points=False)
            image_ids = [frames.frame_idx_from_path(p) for p in ff.image_paths]
            rgbs = frames.read_frames(self.images_dir, image_ids)
            depths = frame_depths(ff, rgbs, conf_percentile=cfg["conf_percentile"])
            c2w = invert_poses(ff.extrinsics)
            intrinsics = ff.intrinsics

        else:
            raise ValueError(f"mesh.source must be 'feedforward' or 'splats', got {source!r}")

        # Sky fuses as a backdrop and seeds floaters; the share is of pixels that had depth
        if cfg["mask_sky"]:
            sky = sky_masks(self.images_dir, idxs=image_ids)

            if sky.shape != depths.shape:
                raise ValueError(
                    f"Sky masks are {sky.shape} but depths are {depths.shape}; "
                    f"{self.images_dir} does not match the depth source."
                )

            n_valid = max(np.count_nonzero(depths), 1)
            dropped = np.count_nonzero(sky & (depths > 0)) / n_valid
            depths = np.where(sky, 0.0, depths)
            logger.info("mesh.mask_sky: dropped %.2f%% of valid depth pixels as sky", 100 * dropped)

        # Fuse, then clean in place
        mesh_path = create_tsdf_mesh(
            depths,
            rgbs,
            c2w,
            intrinsics,
            self.backend_dir,
            voxel_size=cfg["voxel_size"],
            depth_trunc=cfg["depth_trunc"],
            sdf_trunc=cfg["sdf_trunc_mult"] * cfg["voxel_size"],
        )
        clean_repair_mesh(mesh_path, use_convex_hull=cfg["use_convex_hull"])

        # Prepare mesh.ply; keep the cleaned mesh as the texture occluder
        mesh_file = str(mesh_path)
        cleaned = o3d.io.read_triangle_mesh(mesh_file)
        prepared = prepare_mesh(cleaned, voxel_size=cfg["voxel_size"], smooth_iterations=cfg["smooth_iterations"])
        o3d.io.write_triangle_mesh(mesh_file, prepared)

        # Optionally UV-unwrap mesh.ply and project the fused views onto it
        if cfg["texture"]:
            texture_dir = self.backend_dir / "texture"
            create_texture_mesh(prepared, cleaned, texture_dir, rgbs, c2w, intrinsics, voxel_size=cfg["voxel_size"])

        logger.info("Mesh saved to %s", mesh_path)

    def localize(self) -> None:
        """
        Build the per-frame local-feature localization cache into pointcloud.zarr.

        - always rebuilds: an existing group for the matcher is dropped first
        """
        cfg = self.config["localization"]
        _build_localization_db(self.pointcloud_zarr, cfg["matcher"], self.images_dir, top_k=cfg["top_k"])

    def splats(self) -> None:
        """
        Train Gaussian splats on the pointcloud.zarr frames, points and cameras.

        - frames are read from images/ by the frame index in each zarr row name
        - depth targets, when the depth loss is on, are the zarr depth masked and lifted like the mesh stage's
        """
        # gsplat is CUDA-only; import lazily so Reconstructor stays importable without it
        try:
            from collab_splats.splats.trainer import SplatsConfig, train
        except ImportError as exc:
            raise ImportError("the splats stage needs gsplat; see setup.sh") from exc

        # Training config and the light reconstruction
        cfg = SplatsConfig.from_dict(self.config["splats"])
        result = self.result

        # CPU-resident frames in the zarr's row order; train() moves one view to the GPU at a time
        image_ids = [frames.frame_idx_from_path(p) for p in result.image_paths]
        rgbs = frames.read_frames(self.images_dir, image_ids)

        # Depth targets on the frame grid, 0 = no target; the light result carries no depth, so reload it
        depth_targets = None

        if "depth" in cfg.losses and cfg.losses["depth"]["weight"] > 0:
            conf_percentile = self.config["mesh"]["conf_percentile"]
            ff = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=False, load_world_points=False)
            depth_targets = frame_depths(ff, rgbs, conf_percentile=conf_percentile)
            logger.info("splats depth targets use mesh.conf_percentile=%s", conf_percentile)

        train(
            cfg,
            rgbs,
            result.extrinsics,
            result.intrinsics,
            result.points,
            result.colors,
            self.outputs["splats"].parent,
            depth_targets=depth_targets,
            image_ids=image_ids,
        )
        logger.info("Splats saved to %s", self.outputs["splats"])

    def reconstruction_quality_report(self) -> None:
        """
        Reference-free error report: columnar tables, one reconstruction_quality_report.json.

        - named for the artifact it writes; distinct from the video quality report (capture)
        - null photometric table without images; columns listed in geometry/metrics.py
        - a failing measurement raises
        - runs no model and no matcher; reads the zarr and images/ only
        """
        ff = PointcloudResult.load_zarr(self.pointcloud_zarr)

        # Optional frames for photometric, read by the zarr's frame ids; images/ may hold frames sfm dropped
        images = None

        if frames.frame_paths(self.images_dir):
            image_ids = [frames.frame_idx_from_path(p) for p in ff.image_paths]
            images = frames.read_frames(self.images_dir, image_ids)

        # Every table from arrays, keyed by frame name
        names = [Path(str(p)).name for p in ff.image_paths]

        # Optional pair pruning; 0.0 (base.yaml) keeps every ordered pair
        min_pair_overlap = self.config["reconstruction_quality_report"]["min_pair_overlap"]

        tables = compute_reconstruction_quality(
            ff.depth,
            ff.model_intrinsics,
            ff.intrinsics,
            ff.extrinsics,
            ff.original_coords,
            names,
            ff.confidence,
            images,
            min_pair_overlap=min_pair_overlap,
        )

        # Atomic write beside the zarr: reuse-by-existence never sees a half file
        scene = {
            "backend": self.config["pointcloud"]["backend"],
            "n_frames": len(ff.depth),
            "model_resolution": f"{ff.model_width}x{ff.model_height}",
            "image_width": int(ff.original_coords[0][4]),
            "zarr": str(self.pointcloud_zarr),
            "min_pair_overlap": min_pair_overlap,
        }
        write_json(self.outputs["reconstruction_quality_report"], {"scene": scene, **tables})
        logger.info("Reconstruction quality report written to %s", self.outputs["reconstruction_quality_report"])
