"""
Config-driven reconstruction pipeline, one output tree per scene.

- stages: preproc, pointcloud, refine, splats, mesh, semantics, localize, quality report
- a run config merges over configs/base.yaml; keys it omits take the module defaults
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import logging
import shutil
import warnings
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
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
from collab_splats.geometry.viz import plot_photometric_ncc
from collab_splats.localization.extractors import LocalMatcher
from collab_splats.localization.localizer import (
    CameraLocalizer,
    localization_db_exists,
)
from collab_splats.mesh import (
    clean_repair_mesh,
    compute_tsdf_voxel_size,
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
from collab_splats.semantics.features import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
)
from collab_splats.semantics.features.ocr_lens import (
    load_decoder,
    load_processor,
    word_probabilities,
    word_vocabulary,
)
from collab_splats.semantics.lifting import lift_features
from collab_splats.semantics.segmentation import sky_masks
from collab_splats.semantics.store import (
    valid_feature_cache,
    write_feature_cache,
    write_point_features,
)
from collab_splats.utils.colmap import write_colmap_reconstruction
from collab_splats.utils.io import read_image, to_json_safe, write_json
from collab_splats.utils.torch_utils import (
    batch_iterator,
    get_device,
    pytorch_gc,
    to_numpy,
)

if TYPE_CHECKING:
    from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# Stage -> stages it needs; dict order is run order; semantics after mesh lifts onto its vertices
STAGES: dict[str, tuple[str, ...]] = {
    "preproc": (),
    "pointcloud": ("preproc",),
    "refine": ("pointcloud",),
    "splats": ("pointcloud",),
    "mesh": ("pointcloud",),
    "semantics": ("pointcloud",),
    "localize": ("pointcloud",),
    "reconstruction_quality_report": ("pointcloud",),
}

# Stages nothing depends on; only these re-run alone against processed outputs
LEAF_STAGES = frozenset(s for s in STAGES if not any(s in deps for deps in STAGES.values()))


########################################
# Helpers
########################################


def backends() -> dict[str, list[str]]:
    """
    Registered pointcloud backend names per method.

    Returns:
        Method ("feedforward" or "sfm") to its sorted backend names.
    """
    return {"feedforward": sorted(BaseFeedforwardCreator._registry), "sfm": sorted(SFM_CREATORS)}


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


def _build_localization_db(
    pointcloud_zarr: Path, extractor_name: str, images_dir: Path, retrieval: str = "dino-salad"
) -> None:
    """
    Build the per-frame local-feature localization cache into pointcloud.zarr.

    - persists keypoints/descriptors to local_features/{extractor_name}/reconstruction
    - always rebuilds: save_index replaces the extractor's whole group
    - a missing pointcloud.zarr, images/ or frame raises before anything is written
    - a failed extraction leaves the old group in place
    - retrieval: registry name of the global-descriptor model stored with the DB
    """
    # Load the reconstruction; raises on a missing pointcloud.zarr without creating one
    pointcloud = PointcloudResult.load_zarr(pointcloud_zarr, load_world_points=True)

    # The localizer reads full-res frames from images/, not model-res images
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir}: scene has no images/ store")

    # Full-res frames read lazily; KeyError here when a zarr frame is missing from images/
    idxs = [frames.frame_idx_from_path(p) for p in pointcloud.image_paths]
    images = frames.read_frames_chunked(images_dir, idxs)
    ids = [str(p) for p in pointcloud.image_paths]

    # Extract every reference frame, then replace the stored DB
    extractor = LocalMatcher(extractor_name)
    localizer = CameraLocalizer(
        pointcloud.world_points,
        pointcloud.extrinsics,
        images,
        ids,
        extractor=extractor,
        original_coords=pointcloud.original_coords,
        retrieval=retrieval,
    )
    localizer.save_index(pointcloud_zarr, extractor_name)
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

        # Deep-merge the caller's config over base.yaml
        text = Path(base_config).read_text()
        defaults = yaml.safe_load(text) or {}
        merged = merge({}, defaults, config)

        # Validate, then store the fully-populated config
        self.config = self.validate_config(merged)
        self._result: PointcloudResult | None = None

        # Preproc's final RGB frames by file name, handed to an in-process feedforward pointcloud
        self._frames: dict[str, np.ndarray] | None = None

        # Preproc's background PNG write of those frames, joined before anything reads images/
        self._frames_write: Future | None = None

        # Viser server kept by pointcloud() under viz + loop closure, reachable after run() returns
        self.viewer: Viewer | None = None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """
        Reject a config that would fail mid-run, before any stage starts.

        - only cross-field checks: required paths, method/backend pairing, LC/sfm/LoGeR exclusions
        - per-argument bounds live in the creators and create_tsdf_mesh
        - pointcloud.loop_closure and bundle_adjustment are normalized in place to dicts carrying `enabled`

        Args:
            config: config already merged over base.yaml.

        Returns:
            The same config, loop_closure and bundle_adjustment normalized.

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
        registered = backends()

        if method not in registered:
            raise ValueError(f"pointcloud.method must be one of {sorted(registered)}, got {method!r}")

        if backend not in registered[method]:
            raise ValueError(
                f"pointcloud.backend must be one of {registered[method]} for method {method!r}, got {backend!r}"
            )

        # Normalize loop_closure to a dict carrying `enabled`
        lc = pc["loop_closure"]
        lc = dict(lc) if isinstance(lc, dict) else {"enabled": bool(lc)}
        lc.setdefault("enabled", True)
        unknown = set(lc) - {"enabled"} - {f.name for f in dataclasses.fields(LoopClosureConfig)}

        if unknown:
            raise ValueError(f"pointcloud.loop_closure has unknown keys {sorted(unknown)}")

        pc["loop_closure"] = lc

        # Normalize bundle_adjustment to a dict carrying `enabled`
        ba = pc["bundle_adjustment"]
        ba = dict(ba) if isinstance(ba, dict) else {"enabled": bool(ba)}
        ba.setdefault("enabled", True)
        # Unknown keys: neither `enabled` nor a BundleAdjustmentConfig field (tracks_cache_dir is a field, so it passes)
        unknown = (
            set(ba) - {"enabled", "tracks_cache_dir"} - {f.name for f in dataclasses.fields(BundleAdjustmentConfig)}
        )

        if unknown:
            raise ValueError(f"pointcloud.bundle_adjustment has unknown keys {sorted(unknown)}")

        pc["bundle_adjustment"] = ba

        # Build the BA config so its own checks refuse values the solver cannot run
        try:
            BundleAdjustmentConfig(**{k: v for k, v in ba.items() if k != "enabled"})
        except ValueError as e:
            raise ValueError(f"pointcloud.bundle_adjustment: {e}") from e

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

    def write_run_config(self) -> Path:
        """
        Record this run's config beside its outputs.

        - always rewritten: the recorded config must be the one that ran

        Returns:
            The written run_config.yaml path.
        """
        path = self.run_config_path(self.config["output_path"], self.config["pointcloud"]["backend"])
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        return path

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
        Scene-level 2D feature cache: `<extractor>_codes.zarr`.

        - plus a temporary `<extractor>_features.zarr` of full-width features while the AE trains
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
            True when the marker exists; pointcloud also needs the COLMAP model; semantics with a recorded mesh hash
            also needs mesh.ply to match it.
        """
        # localize lives inside pointcloud.zarr; pointcloud also needs its COLMAP model
        if stage == "localize":
            matcher = self.config["localization"]["matcher"]
            return self.pointcloud_zarr.exists() and localization_db_exists(self.pointcloud_zarr, matcher)

        if stage == "pointcloud":
            return self.pointcloud_zarr.exists() and self.colmap_model_dir.exists()

        # Semantics with vertex arrays is stale when mesh.ply changed since their lift
        if stage == "semantics" and self.outputs["semantics"].exists() and self.outputs["mesh"].exists():
            recorded = zarr.open(str(self.outputs["semantics"]), mode="r").attrs.get("mesh_sha256")
            return recorded is None or recorded == hashlib.sha256(self.outputs["mesh"].read_bytes()).hexdigest()

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
        - preproc writes images/ in the background; the next stage or the run's exit joins it, re-raising a write error
          (only logged when a stage error is already propagating)

        Args:
            stages: stage names, any order.
            overwrite: rebuild stages whose output exists.

        Raises:
            ValueError: on an unknown stage, an unmet dependency, or a named leaf already done.
        """
        named = stages is not None

        # Default stage set from the enable flags
        if stages is None:
            pc = self.config["pointcloud"]
            # BA is skipped for sfm (its mapper runs its own) and with LC (it runs inside each window)
            enabled = {
                "refine": pc["bundle_adjustment"]["enabled"]
                and pc["method"] == "feedforward"
                and not pc["loop_closure"]["enabled"],
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

        # Run in table order; skip done stages; finish the PNG write and drop preproc's frames on any exit
        try:
            for stage in [s for s in STAGES if s in stages]:
                if self.done(stage) and not overwrite:
                    logger.info("stage %s done, skipping", stage)
                    continue

                logger.info("=== Stage: %s ===", stage)

                # Pointcloud joins the PNG write itself; every other stage starts after it
                if stage != "pointcloud":
                    self._join_frames_write()

                # Preproc writes its PNGs in the background so pointcloud overlaps them
                if stage == "preproc":
                    self.preproc(background_write=True)
                else:
                    getattr(self, stage)()
        except BaseException:
            # Let the stage error win; wait for the write and only log its own failure
            write = self._frames_write
            self._frames_write = None
            write_error = write.exception() if write is not None else None

            if write_error is not None:
                logger.error("background PNG write failed: %s", write_error)

            raise
        finally:
            self._frames = None
            self._join_frames_write()

    def _join_frames_write(self) -> None:
        """
        Wait for preproc's background PNG write, re-raising its error; a no-op when none is pending.
        """
        if self._frames_write is None:
            return

        write = self._frames_write
        self._frames_write = None
        write.result()

    ########################################
    # Pipeline stages
    ########################################

    def preproc(self, background_write: bool = False) -> None:
        """
        Select keyframes from the input into images/.

        - video: measure quality, sample, write, plot the report with the kept frames marked
        - directory: every image, in filename order, source index = position
        - undistort self-calibrates from the written frames, then rewrites them
        - background_write leaves the final write pending; the next stage or run() joins it

        Args:
            background_write: submit the final PNG write to a thread instead of waiting for it.

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
            common = {"report": report, "max_frames": cfg["max_frames"]}

            # Optional sampler knobs; unset ones take the sampler defaults
            if "quality" in cfg:
                common["quality"] = cfg["quality"]

            fps_opts = {k: cfg[k] for k in ("min_frames", "on_empty_slot") if k in cfg}

            # Each selection method gets only its own knobs
            if cfg["frame_selection"] == "fps":
                rgbs, records = sample_fps(
                    str(input_path),
                    fps=cfg["fps"],
                    workers=cfg["n_workers"],
                    **fps_opts,
                    **common,
                )
            elif cfg["frame_selection"] == "uniform":
                rgbs, records = sample_uniform(str(input_path), workers=cfg["n_workers"], **common)
            elif cfg["frame_selection"] == "optical_flow":
                rgbs, records = sample_optical_flow(str(input_path), **common)
            else:
                raise ValueError(
                    f"preproc.frame_selection must be fps, uniform or optical_flow, got {cfg['frame_selection']!r}"
                )

            # Refuse an empty store; it would only surface downstream as a missing file
            if not len(rgbs):
                raise ValueError(f"0 of {total} frames selected from {input_path}; see {report_path}")

        # Source frame index of each selected frame
        idxs = [r["frame_idx"] for r in records]

        # Undistort calibrates from written frames, then undistorts them for the final write
        if cfg["undistort"]:
            frames.write_frames(self.images_dir, rgbs, idxs)
            camera = calibrate_camera(self.images_dir)
            rgbs, _ = undistort_frames(rgbs, camera)

        # Write the final frames under their source indices, in the background when asked
        if background_write:
            executor = ThreadPoolExecutor(1)
            self._frames_write = executor.submit(frames.write_frames, self.images_dir, rgbs, idxs)
            executor.shutdown(wait=False)
        else:
            frames.write_frames(self.images_dir, rgbs, idxs)

        # Plot the quality report with the kept frames marked
        if report is not None:
            plot_photometric(report, self.images_dir.parent, selected=idxs)
            plot_motion(report, self.images_dir.parent, selected=idxs)

        # Keep the final frames, by write_frames' file names, for an in-process pointcloud stage
        names = [frames.frame_name(idx) for idx in idxs]
        self._frames = dict(zip(names, rgbs, strict=True))

        logger.info("preproc: %d frames in %s", len(idxs), self.images_dir)

    def pointcloud(self) -> None:
        """
        Reconstruct images/ into pointcloud.zarr, the COLMAP model and sparse_pc.ply.

        - feedforward optionally wraps the creator in loop closure, with a live viewer when viz is on
        - BA on with LC: bundle adjustment runs inside each window; records land in the zarr attrs as window_ba
        - sfm is experimental; its creator writes its own subset and alignment attrs
        """
        cfg = self.config["pointcloud"]
        backend = cfg["backend"]

        # Backends that read images/ (sfm, loger) wait for preproc's PNG write
        if cfg["method"] == "sfm" or backend == "loger":
            self._join_frames_write()

        # Window BA config; set only for feedforward with LC and BA on
        ba_cfg = None

        # Build the feedforward creator, optionally inside loop closure
        if cfg["method"] == "feedforward":
            creator_cls = get_creator(backend)
            creator = creator_cls(
                max_points=cfg["max_points"],
                clean=cfg["clean"]["enabled"],
                frames=self._frames,
                **cfg.get(backend, {}),
            )

            # Wrap in loop closure when enabled; the remaining keys are its config
            lc = dict(cfg["loop_closure"])

            if lc.pop("enabled"):
                lc_config = LoopClosureConfig(**lc) if lc else None

                # BA with LC runs inside each window, with the refine stage's terms
                if cfg["bundle_adjustment"]["enabled"]:
                    terms = {k: v for k, v in cfg["bundle_adjustment"].items() if k != "enabled"}
                    ba_cfg = BundleAdjustmentConfig(**terms)

                creator = LoopClosure(base=creator, config=lc_config, ba=ba_cfg)

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
            creator = SFM_CREATORS[backend](
                clean=cfg["clean"]["enabled"], max_points=cfg["max_points"], **cfg.get(backend, {})
            )

        # Drop a stale refine marker before touching the artifacts it describes
        self.outputs["refine"].unlink(missing_ok=True)

        # Reconstruct; the creator writes the COLMAP model itself
        result = creator.create_pointcloud(self.images_dir, self.backend_dir, self.colmap_model_dir)

        # Finish preproc's PNG write before the zarr marks this stage done
        self._join_frames_write()

        # Provenance attrs; an sfm creator supplies its own subset and alignment attrs
        if cfg["method"] == "feedforward":
            attrs = {"method": "feedforward", "backend": backend}
        else:
            attrs = {"backend": backend, **creator.attrs}

        # Per-window BA records; zarr attrs need strict JSON
        if ba_cfg is not None:
            attrs["window_ba"] = to_json_safe(creator.window_ba)

        # Persist the zarr and the PLY
        result.save_zarr(self.pointcloud_zarr, extra_attrs=attrs)
        result.write_ply(self.sparse_ply)
        logger.info("pointcloud: %d pts in %s", len(result.points), self.pointcloud_zarr)

        # Free the model before the next stage loads its own
        del creator, result
        self._result = None
        self._frames = None
        pytorch_gc()

    def refine(self) -> None:
        """
        Bundle-adjust the feedforward poses, then rewrite every pose-derived artifact.

        - points are re-derived, re-cleaned and re-capped, so their count can change
        - the zarr is rewritten whole, which drops any localization DB built on the old points
        - a matcher track_source (xfeat / loma) reads the full-res frames in the images/ store

        Raises:
            ValueError: pointcloud.method is sfm, or loop closure is on.
            FileNotFoundError: a matcher track_source and no images/ store.
        """
        cfg = self.config["pointcloud"]

        if cfg["method"] == "sfm":
            raise ValueError("refine is not supported for pointcloud.method: sfm")

        if cfg["loop_closure"]["enabled"]:
            raise ValueError("refine is not supported with pointcloud.loop_closure — BA already ran inside each window")

        # Build the BA config; a matcher track_source needs the images/ store before any load
        terms = {k: v for k, v in cfg["bundle_adjustment"].items() if k != "enabled"}
        ba_cfg = BundleAdjustmentConfig(**terms)

        if ba_cfg.track_source != "vggsfm" and not self.images_dir.is_dir():
            raise FileNotFoundError(f"{self.images_dir}: scene has no images/ store")

        # Refine poses, then re-derive the points under the new cameras
        pointcloud = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=True)
        ba = BundleAdjustment(ba_cfg)

        # Full-res store frames in zarr order: matcher track input and track-cache key; vggsfm needs no store
        frame_paths = None

        if ba_cfg.track_source != "vggsfm":
            idxs = [frames.frame_idx_from_path(p) for p in pointcloud.image_paths]
            frame_paths = frames.frame_paths(self.images_dir, idxs)

        # Bundle-adjust, then reproject the points under the refined cameras
        extrinsics, intrinsics = ba.refine(
            pointcloud.images,
            pointcloud.confidence,
            pointcloud.world_points,
            pointcloud.extrinsics,
            pointcloud.model_intrinsics,
            depth=pointcloud.depth,
            frame_paths=frame_paths,
        )
        pointcloud = dataclasses.replace(
            pointcloud, extrinsics=extrinsics, model_intrinsics=intrinsics, intrinsics=None
        )
        pointcloud = pointcloud.reproject()

        # Re-clean and re-cap under the refined cameras
        n_before = len(pointcloud.points)
        pointcloud = clean_pointcloud(pointcloud, remove_outliers=cfg["clean"]["enabled"], max_points=cfg["max_points"])
        logger.info("refine: %d of %d pts kept after clean + cap", len(pointcloud.points), n_before)

        # Rewrite the zarr with its provenance attrs, then the COLMAP model and PLY
        store = zarr.open_group(str(self.pointcloud_zarr), mode="r")
        attrs = dict(store.attrs)
        pointcloud.save_zarr(self.pointcloud_zarr, extra_attrs=attrs)
        recon = pointcloud.to_colmap()
        write_colmap_reconstruction(recon, self.colmap_model_dir)
        pointcloud.write_ply(self.sparse_ply)
        self._result = None

        # Marker last: BA config and loss history
        marker = self.outputs["refine"]
        marker.parent.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(ba_cfg).items()}
        report = {
            "config": config,
            "loss_history": ba.loss_history,
            "losses": ba.losses,
            "alignment_scale": ba.alignment_scale,
            "n_frames": len(pointcloud.image_paths),
        }
        write_json(marker, report)

    def semantics(self) -> None:
        """
        Extract 2D features, compress every frame to fp16 codes, lift the codes onto the points.

        - codes store `<extractor>_codes.zarr` reused when valid (name, frames, kwargs, latent_dim)
        - a miss extracts full-width features to a temporary `<extractor>_features.zarr`, trains the AE on all
          of them, encodes every frame, then deletes them; n_components null keeps full width, no AE
        - lift reads one frame of codes at a time; rows follow the zarr's frames (maybe a subset)
        - lifted store written atomically with its own autoencoder.pt; codes and lifted stores are pushed
        - with mesh.ply: ocr_lens also stores each vertex's top-64 words (decode, then lift), queryable
          extractors `vertex_features` (codes on vertices); both record the mesh's sha256
        """
        cfg = self.config["semantics"]
        name = cfg["extractor"]
        extractor_kwargs = cfg["extractor_kwargs"]
        latent_dim = cfg["n_components"]
        codes_path = self.semantics_cache_dir / f"{name}_codes.zarr"
        features_path = self.semantics_cache_dir / f"{name}_features.zarr"
        self.semantics_cache_dir.mkdir(parents=True, exist_ok=True)

        # 2D codes for every images/ frame; the model loads only on a miss
        if valid_feature_cache(codes_path, name, self.images_dir, extractor_kwargs, latent_dim) is None:
            extractor = BaseFeatureExtractor.get(name)(**extractor_kwargs)
            paths = frames.frame_paths(self.images_dir)
            n_frames = len(paths)
            attrs = {
                "extractor": name,
                "patch_size": extractor.patch_size,
                "n_frames": n_frames,
                "extractor_kwargs": extractor_kwargs,
                "latent_dim": latent_dim,
            }

            # Extractor maps over every frame, decoded and run 4 at a time
            batches = (extractor.forward([read_image(p) for p in batch]) for (batch,) in batch_iterator(4, paths))
            maps = itertools.chain.from_iterable(batches)

            # Uncompressed: full-width maps are the codes; else they are temporary features
            target = codes_path if latent_dim is None else features_path

            with torch.no_grad():
                write_feature_cache(target, maps, n_frames, attrs if latent_dim is None else {})

            # Free the extractor's GPU memory now; a forward hook can hold it in a reference cycle
            del extractor
            pytorch_gc()

            # Train the AE on every frame's features, encode each frame into the codes store, drop the features
            if latent_dim is not None:
                features = zarr.open(str(features_path), mode="r")["features"]
                ae = FeatureAutoencoder(input_dim=features.shape[1], latent_dim=latent_dim)
                ae.fit(features, epochs=cfg["max_epochs"], target_cosine=cfg["target_cosine"])
                ae.to(get_device())
                encoded = map(partial(_load_frame, features, range(n_frames), ae), range(n_frames))
                write_feature_cache(codes_path, encoded, n_frames, attrs, ae=ae)
                shutil.rmtree(features_path)

        # Codes read lazily; the AE only rides along into the lifted store
        codes = zarr.open(str(codes_path), mode="r")["features"]
        ae = None

        if latent_dim is not None:
            ae = FeatureAutoencoder.load(codes_path / "autoencoder.pt")

        # Pick the zarr's frames, in the zarr's order, and lift the stored codes onto the points
        pointcloud = PointcloudResult.load_zarr(self.pointcloud_zarr, load_world_points=False)
        rows = store_rows(self.images_dir, pointcloud.image_paths)
        lifted = lift_features(partial(_load_frame, codes, rows, None), pointcloud)

        # Store attrs; the viewer rebuilds a queryable extractor from them
        attrs = {"extractor": name, "extractor_kwargs": extractor_kwargs}
        vertex_arrays = {}
        mesh_path = self.outputs["mesh"]

        # Mesh vertices as points over the same cameras; no source pixel, unseen vertices stay zero
        if mesh_path.exists():
            vertices = np.asarray(o3d.io.read_triangle_mesh(str(mesh_path)).vertices, dtype=np.float32)
            colors = np.zeros((len(vertices), 3), dtype=np.uint8)
            vertex_cloud = dataclasses.replace(pointcloud, points=vertices, colors=colors, pixel_indices=None)

            # ocr_lens: each frame's top-64 words per patch, lifted as indexed maps; each vertex keeps its top-64
            if name == "ocr_lens":
                model_id = extractor_kwargs.get("model_id", "llava-hf/llava-v1.6-vicuna-7b-hf")
                vocab = word_vocabulary(load_processor(model_id).tokenizer)
                decoder = load_decoder(model_id)

                if ae is not None:
                    ae.to(get_device())

                maps = []

                for i in range(len(rows)):
                    # Patch codes as rows, then word probabilities per patch, top-64 kept as an indexed map
                    fmap = _load_frame(codes, rows, None, i)
                    channels, height, width = fmap.shape
                    states = fmap.reshape(channels, -1).T
                    probs = torch.cat([p for p, _ in word_probabilities(states, decoder, vocab, ae=ae)])
                    top = probs.topk(64, dim=1)
                    maps.append((top.indices.T.reshape(64, height, width), top.values.T.reshape(64, height, width)))

                del decoder
                pytorch_gc()

                # One lift, no chunking: host RAM plus GPU bound it, ~14 KB/vertex for the (V, n_words) float32 on CPU
                lifted_words = lift_features(maps.__getitem__, vertex_cloud, num_classes=len(vocab.words))
                top = lifted_words.to(get_device()).topk(64, dim=1)
                vertex_arrays["vertex_word_ids"] = to_numpy(top.indices).astype(np.int16)
                vertex_arrays["vertex_word_probs"] = to_numpy(top.values).astype(np.float16)
                attrs["words"] = vocab.words
                del maps, lifted_words

            # Queryable: codes lifted like the points, decoded at read
            elif issubclass(BaseFeatureExtractor.get(name), BaseQueryableExtractor):
                vertex_codes = lift_features(partial(_load_frame, codes, rows, None), vertex_cloud)
                vertex_arrays["vertex_features"] = to_numpy(vertex_codes).astype(np.float16)

            # Record the mesh the vertex arrays index into
            if vertex_arrays:
                attrs["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()

        # Write points, vertex arrays and attrs together; the lifted store is the stage's done marker
        write_point_features(self.outputs["semantics"], to_numpy(lifted), ae, vertex_arrays=vertex_arrays, attrs=attrs)

    def mesh(self) -> None:
        """
        Fuse depth and RGB into a TSDF mesh.ply, clean and prepare it, and optionally texture it.

        - feedforward source: pointcloud.zarr depth, confidence-masked and lifted to the frame grid
        - splats source: the checkpoint's own renders and poses; the splats stage is never auto-run
        - mask_sky zeroes sky depth on either source before fusion
        - mesh.ply is the prepared mesh (filled, decimated, manifold); texture/ is its UV bake

        Raises:
            FileNotFoundError: the splats source has no ckpt.pt on disk.
            ValueError: mesh.source is neither feedforward nor splats, mesh.depth_trunc_percentile is outside
                (0, 99], or sky masks mismatch the depths.
            RuntimeError: mesh.texture is on but no CUDA device is available.
        """
        cfg = self.config["mesh"]
        source = cfg["source"]

        # Uncut far depth crashes Open3D extraction
        trunc_pct = cfg["depth_trunc_percentile"]

        if trunc_pct is None or not 0 < trunc_pct <= 99:
            raise ValueError(f"mesh.depth_trunc_percentile must be in (0, 99], got {trunc_pct!r}")

        # Texturing is CUDA-only; refuse before fusion rather than after it
        if cfg["texture"] and not torch.cuda.is_available():
            raise RuntimeError("mesh.texture needs CUDA (nvdiffrast); set mesh.texture: false on CPU-only machines")

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
            depth_fx = float(np.median(intrinsics[:, 0, 0]))

        # Feedforward source: the zarr's depth on the zarr's own frames
        elif source == "feedforward":
            pointcloud = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=False, load_world_points=False)
            image_ids = [frames.frame_idx_from_path(p) for p in pointcloud.image_paths]
            rgbs = frames.read_frames(self.images_dir, image_ids)
            depths = frame_depths(pointcloud, rgbs, conf_percentile=cfg["conf_percentile"])
            c2w = invert_poses(pointcloud.extrinsics)
            intrinsics = pointcloud.intrinsics

            # Depth was predicted on the model grid; its pixels, not the frame's, set the voxel
            depth_fx = float(np.median(pointcloud.model_intrinsics[:, 0, 0]))

            # Free the zarr's model-grid arrays; only poses and intrinsics are used from here
            del pointcloud

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
            del sky

        # Cut far depth before voxel sizing
        sub = depths[:, ::8, ::8]
        depth_trunc = float(np.percentile(sub[sub > 0], trunc_pct))
        depths = np.where(depths > depth_trunc, 0.0, depths)

        # Voxel from this scene's own depth; the world has no fixed scale
        voxel_size = compute_tsdf_voxel_size(
            depths,
            c2w,
            intrinsics,
            depth_fx=depth_fx,
            depth_px=cfg["voxel_depth_px"],
            ref_percentile=cfg["voxel_ref_percentile"],
        )

        # mesh.ply marks the stage done, so a stale one must not outlive a failed re-run
        mesh_path = self.outputs["mesh"]
        mesh_path.unlink(missing_ok=True)

        # Fuse every view into one TSDF mesh
        mesh = create_tsdf_mesh(
            depths,
            rgbs,
            c2w,
            intrinsics,
            voxel_size=voxel_size,
            sdf_trunc=cfg["sdf_trunc_mult"] * voxel_size,
        )

        # Free the fused depth before cleanup; nothing below reads it
        del depths

        # Clean; the floater-cut real surface comes back as the texture occluder
        cleaned, real = clean_repair_mesh(mesh, use_convex_hull=cfg["use_convex_hull"])

        # Fill, decimate and repair; the full-density cleaned mesh is freed before texturing
        prepared = prepare_mesh(
            cleaned, voxel_size=voxel_size, smooth_iterations=cfg["smooth_iterations"], max_faces=cfg["max_faces"]
        )
        del cleaned

        # Optionally UV-unwrap the prepared mesh and project the fused views onto it
        if cfg["texture"]:
            texture_dir = self.backend_dir / "texture"
            create_texture_mesh(prepared, real, texture_dir, rgbs, c2w, intrinsics, voxel_size=voxel_size)

        # Write mesh.ply last, so it exists only when every step above finished
        o3d.io.write_triangle_mesh(str(mesh_path), prepared)
        logger.info("Mesh saved to %s", mesh_path)

    def localize(self) -> None:
        """
        Build the per-frame local-feature localization cache into pointcloud.zarr.

        - always rebuilds: an existing group for the matcher is dropped first
        """
        cfg = self.config["localization"]
        _build_localization_db(self.pointcloud_zarr, cfg["matcher"], self.images_dir, cfg["retrieval"])

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

        # CPU-resident frames in the zarr's row order; train() caches each factor's targets on the GPU
        image_ids = [frames.frame_idx_from_path(p) for p in result.image_paths]
        rgbs = frames.read_frames(self.images_dir, image_ids)

        # Depth targets on the frame grid, 0 = no target; the light result carries no depth, so reload it
        depth_targets = None

        if "depth" in cfg.losses and cfg.losses["depth"]["weight"] > 0:
            conf_percentile = self.config["mesh"]["conf_percentile"]
            pointcloud = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=False, load_world_points=False)
            depth_targets = frame_depths(pointcloud, rgbs, conf_percentile=conf_percentile)
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
        - reconstruction_quality_ncc.png beside it: photometric NCC by frame gap
        """
        pointcloud = PointcloudResult.load_zarr(self.pointcloud_zarr)

        # Optional frames for photometric, read by the zarr's frame ids; images/ may hold frames sfm dropped
        images = None

        if frames.frame_paths(self.images_dir):
            image_ids = [frames.frame_idx_from_path(p) for p in pointcloud.image_paths]
            images = frames.read_frames(self.images_dir, image_ids)

        # Every table from arrays, keyed by frame name
        names = [Path(str(p)).name for p in pointcloud.image_paths]

        # Optional pair pruning; 0.0 keeps every ordered pair
        min_pair_overlap = self.config["reconstruction_quality_report"]["min_pair_overlap"]

        tables = compute_reconstruction_quality(
            pointcloud.depth,
            pointcloud.model_intrinsics,
            pointcloud.intrinsics,
            pointcloud.extrinsics,
            pointcloud.original_coords,
            names,
            pointcloud.confidence,
            images,
            min_pair_overlap=min_pair_overlap,
        )

        # Atomic write beside the zarr: reuse-by-existence never sees a half file
        scene = {
            "backend": self.config["pointcloud"]["backend"],
            "n_frames": len(pointcloud.depth),
            "model_resolution": f"{pointcloud.model_width}x{pointcloud.model_height}",
            "image_width": int(pointcloud.original_coords[0][4]),
            "zarr": str(self.pointcloud_zarr),
            "min_pair_overlap": min_pair_overlap,
        }
        write_json(self.outputs["reconstruction_quality_report"], {"scene": scene, **tables})
        logger.info("Reconstruction quality report written to %s", self.outputs["reconstruction_quality_report"])

        # NCC-by-gap plot beside the report; skipped without images
        plot_path = self.backend_dir / "reconstruction_quality_ncc.png"
        plot_photometric_ncc(tables["photometric_pairs"], plot_path, title=f"{scene['backend']}: cross-view NCC by gap")
