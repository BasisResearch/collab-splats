"""Run the primitives pipeline for one video: sample -> pointcloud -> mesh -> semantics."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from PIL import Image

from collab_splats.dashboard.config import LocalizationConfig, RunConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import PULL_EXCLUDES, SessionSource
from collab_splats.mesh.utils import persist_mesh_vertex_features, pointcloud_to_mesh
from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator,
    VGGTXCreator,
)
from collab_splats.pointcloud.utils import lift_features
from collab_splats.preproc import extract_frame, sample_frames
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features.base import BaseFeatureExtractor
from collab_splats.utils.image import open_image

# VGGTOmegaCreator requires the vggt-omega submodule; only available when installed.
try:
    from collab_splats.pointcloud.feedforward import VGGTOmegaCreator
except ImportError:
    VGGTOmegaCreator = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

########
# Helpers
########


def _write_frames_zarr(
    frames: list[np.ndarray], records: list[dict], path: Path, *, video_path: Path, method: str, max_frames: int
) -> None:
    """Write the canonical frames.zarr (FrameStore schema) — feeds CameraLocalizer.from_feedforward
    so pixel reads bypass ff.image_paths (which may point at a directory this session doesn't own)."""
    prov = {
        "video_path": str(video_path),
        "video_mtime": Path(video_path).stat().st_mtime,
        "method": method,
        "max_frames": max_frames,
    }
    FrameStore.create(path, frames, records, provenance=prov)


def _write_frames_jpegs(frames: list[np.ndarray], records: list[dict], frames_dir: Path) -> Path:
    """Write frames as source-frame_idx-named JPEGs for path-locked creators (setup_inference,
    semantics extraction) that require a real on-disk image directory.

    Filenames must match FrameStore.frame_idx_from_path's convention (frame_{idx:06d}.jpg,
    source video index — not list position) so that consumers reading this dir alongside
    frames.zarr (e.g. localization ref thumbnails) resolve the same frame from both.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f, r in zip(frames, records):
        Image.fromarray(f).save(frames_dir / f"frame_{int(r['frame_idx']):06d}.jpg")
    return frames_dir


def _build_creator(env_model: str, conf: float):
    """Instantiate the selected feedforward creator with its confidence arg."""
    if env_model == "vggt_omega":
        if VGGTOmegaCreator is None:
            raise ImportError("vggt-omega is not installed; run setup.sh to enable this model")
        return VGGTOmegaCreator(conf_threshold=conf)
    if env_model == "vggtx":
        return VGGTXCreator(conf_threshold=conf)
    if env_model == "mapanything":
        return MapAnythingCreator(confidence_percentile=conf)
    raise ValueError(f"unknown env_model: {env_model}")


def _extract_semantics(extractor_name: str, image_dir: Path, out_dir: Path) -> None:
    """Extract + cache patch features for the sampled frames."""
    extractor = BaseFeatureExtractor.get(extractor_name)()
    image_paths = sorted(image_dir.glob("*.jpg"))
    extractor.extract_and_cache(image_paths, out_dir)


# Feature-compression autoencoder defaults (matches the semantic_lifting tutorial).
_AE_LATENT_DIM = 64
_AE_EPOCHS = 10


def _load_feature_maps(semantics_dir: Path) -> list[torch.Tensor]:
    """Load per-frame dense feature maps (D, H_p, W_p) from the cached semantics zarr."""
    store_path = next(Path(semantics_dir).glob("*.zarr"))
    arr = zarr.open(str(store_path), mode="r")["features"]  # (N, D, H_p, W_p)
    return [torch.from_numpy(np.asarray(arr[i])) for i in range(arr.shape[0])]


def _lift_and_compress(result, semantics_dir: Path, op_log: OperationLog) -> None:
    """Train a feature-compression autoencoder, lift COMPRESSED maps to points, cache decoded.

    Order matches the semantic_lifting tutorial and is what makes this fast: train the AE on the
    2D patch features, encode each map (D→latent) on GPU, then lift the small latent maps to
    points — lift_features cost scales with channel count, so lifting `latent` (e.g. 64) instead
    of the full D (e.g. 768) is ~D/latent× cheaper. Decode per-point back to D, L2-normalise, and
    cache so the first query is instant. Everything except the lift runs on the GPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_maps = _load_feature_maps(semantics_dir)  # list of (D, H_p, W_p) on CPU
    input_dim = feature_maps[0].shape[0]

    # Train the AE on flattened 2D patch features (all frames) — GPU; stream loss to the log
    op_log.update_progress(80, "semantics: fitting autoencoder")
    t = time.perf_counter()
    ae = FeatureAutoencoder(input_dim=input_dim, latent_dim=_AE_LATENT_DIM).to(device)
    patches = torch.cat([fm.flatten(1).T for fm in feature_maps]).to(device)  # (N*H_p*W_p, D)
    ae.fit(
        patches,
        epochs=_AE_EPOCHS,
        on_epoch=lambda e, t_, loss: op_log.append_line(f"semantics: autoencoder epoch {e}/{t_}  loss={loss:.4f}"),
    )
    op_log.append_line(f"semantics: autoencoder fit in {time.perf_counter() - t:.1f}s")

    # Encode maps (GPU), lift the small latent maps to points (cheap), decode per-point (GPU)
    op_log.update_progress(90, "semantics: lifting compressed features")
    t = time.perf_counter()
    with torch.no_grad():
        compressed_maps = [ae.encode(fm.to(device)).detach().cpu() for fm in feature_maps]
    compressed_pts = lift_features(compressed_maps, result)  # (P, latent) — fast
    with torch.no_grad():
        decoded = ae.per_point_decode(compressed_pts.to(device))  # (P, D)
        normed = torch.nn.functional.normalize(decoded, dim=1)
    op_log.update_progress(94, "semantics: caching lifted features")
    np.save(Path(semantics_dir) / "lifted_normed.npy", normed.detach().cpu().numpy())
    ae.save(Path(semantics_dir))
    op_log.append_line(f"semantics: lift + decode + cache in {time.perf_counter() - t:.1f}s")


def _transfer_mesh_features(result, out_dir: Path, *, k: int = 5, sdf_trunc: float = 0.03) -> None:
    """Transfer cached point features onto the TSDF mesh vertices and persist vertex_features.npy.

    No-op (logged) if the mesh or the lifted point features are missing — neither is fatal
    to the run.
    """
    mesh_path = Path(out_dir) / "mesh" / "mesh_tsdf.ply"
    lifted_path = Path(out_dir) / "semantics" / "lifted_normed.npy"
    if not mesh_path.exists() or not lifted_path.exists():
        logger.warning("mesh feature transfer skipped: mesh=%s lifted=%s", mesh_path.exists(), lifted_path.exists())
        return
    point_features = np.load(lifted_path)
    persist_mesh_vertex_features(mesh_path, result.points, point_features, k=k, sdf_trunc=sdf_trunc)


def _sample(video_path: Path, config: RunConfig, op_log: OperationLog):
    """Sample frames per the configured method; return (frames, records)."""

    # Live label shows images processed / total; log=False so per-frame pings don't flood the log.
    # Throttle to ~100 writes total (every 1% of frames) — the UI polls at 300ms regardless.
    def on_progress(done: int, total: int) -> None:
        step = max(1, total // 100)
        if done % step and done != total:
            return
        op_log.update_progress(
            int(5 + 15 * done / max(total, 1)),
            f"sampling: frame {done}/{total}",
            log=False,
        )

    op_log.update_progress(5, f"sampling: {config.sampling_method}")
    # One call; records carry true source frame indices for both methods
    method = "optical_flow" if config.sampling_method == "optical_flow" else "uniform"
    frames, records = sample_frames(
        str(video_path),
        method=method,
        min_disparity=config.min_disparity,
        max_frames=config.max_frames,
        on_progress=on_progress,
    )
    return frames, records


########
# Orchestrator
########


def _push_async(source: SessionSource, out_dir: Path, session: str, stem: str, op_log: OperationLog) -> None:
    """Push the output tree to fieldwork_processed in a detached, non-fatal thread."""

    def _worker() -> None:
        t0 = time.perf_counter()
        op_log.append_line("push: uploading to fieldwork_processed")
        try:
            source.push_outputs(out_dir, session, stem, on_line=op_log.append_line)
            op_log.append_line(f"push: done in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:  # non-fatal: outputs already on local disk
            logger.exception("push failed")
            op_log.append_line(f"push: FAILED ({exc})")

    threading.Thread(target=_worker, daemon=True).start()


def run_pipeline(
    *,
    video_path: Path,
    session: str,
    stem: str,
    config: RunConfig,
    op_log: OperationLog,
    source: SessionSource,
    base_dir: Path,
) -> Path:
    """Execute the full pipeline; write outputs under base_dir/session/stem; push in background."""
    out_dir = Path(base_dir) / session / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    op_log.start_op(f"{session}/{stem}")
    # Bridge collab_splats module logs (e.g. creator/semantics '%d/%d frames') into the dashboard log.
    try:
        with op_log.attach_logging("collab_splats"):
            # Sample frames from video and persist for creator + viewer
            t = time.perf_counter()
            frames, records = _sample(Path(video_path), config, op_log)
            sampling_method = "optical_flow" if config.sampling_method == "optical_flow" else "uniform"
            _write_frames_zarr(
                frames,
                records,
                out_dir / "frames.zarr",
                video_path=video_path,
                method=sampling_method,
                max_frames=config.max_frames,
            )
            # frames/ jpgs feed path-locked consumers only (setup_inference, semantics);
            # frames.zarr is the canonical store for pixel reads (localization ref thumbnails
            # included — see _build_result_figures's frames_zarr threading).
            image_dir = _write_frames_jpegs(frames, records, out_dir / "frames")
            config.frame_indices = [r["frame_idx"] for r in records]
            op_log.append_line(f"sample ({len(frames)} frames): {time.perf_counter() - t:.1f}s")

            # Feedforward pointcloud reconstruction. Decompose run() into its 4 substeps so each
            # shows in the dashboard (run() = load→preprocess→infer→postprocess, no COLMAP — the
            # build_colmap export nothing here consumes; this keeps the cloud identical to the notebook).
            creator = _build_creator(config.env_model, config.conf_threshold)
            t = time.perf_counter()
            op_log.update_progress(25, f"pointcloud: loading model ({config.env_model})")
            creator.load_model()
            op_log.append_line(f"pointcloud: model loaded in {time.perf_counter() - t:.1f}s")
            t = time.perf_counter()
            op_log.update_progress(32, "pointcloud: preprocessing images")
            creator.setup_inference(image_dir)
            op_log.append_line(f"pointcloud: preprocessed in {time.perf_counter() - t:.1f}s")
            t = time.perf_counter()
            op_log.update_progress(42, "pointcloud: running inference")
            creator.run_inference()
            op_log.append_line(f"pointcloud: inference in {time.perf_counter() - t:.1f}s")
            t = time.perf_counter()
            op_log.update_progress(52, "pointcloud: postprocessing")
            creator.postprocess()
            result = creator.outputs
            op_log.append_line(
                f"pointcloud: postprocessed ({len(result.points):,} pts) in {time.perf_counter() - t:.1f}s"
            )
            result.save_zarr(out_dir / "feedforward.zarr")

            # Mesh from TSDF depth fusion
            op_log.update_progress(60, "mesh: tsdf fusion")
            t = time.perf_counter()
            pointcloud_to_mesh(
                result,
                out_dir / "mesh",
                method="open3d_tsdf",
                voxel_size=config.mesh_voxel_size,
                sdf_trunc=config.mesh_sdf_trunc,
                depth_trunc=config.mesh_depth_trunc,
                clean_repair=config.mesh_clean_repair,
            )
            op_log.append_line(f"mesh: tsdf in {time.perf_counter() - t:.1f}s")

            # Extract and cache semantic patch features
            op_log.update_progress(72, f"semantics: extracting features ({config.semantic_extractor})")
            t = time.perf_counter()
            _extract_semantics(config.semantic_extractor, image_dir, out_dir / "semantics")
            op_log.append_line(f"semantics: extracted in {time.perf_counter() - t:.1f}s")

            # Lift features to points + train compression autoencoder eagerly (instant queries later);
            # _lift_and_compress emits its own 'semantics: ...' substep labels.
            _lift_and_compress(result, out_dir / "semantics", op_log)

            # Transfer lifted point features onto the mesh vertices (same feature space -> mesh is queryable).
            op_log.update_progress(94, "mesh: transferring features to vertices")
            _transfer_mesh_features(result, out_dir)

            # Persist provenance: frame indices + video ref baked into run_config.yaml
            config.to_yaml(
                out_dir / "run_config.yaml",
                video_ref=f"reconstruction/{session}/{stem}/{Path(video_path).name}",
            )

        # Local outputs ready: mark complete and push in the background (non-fatal).
        op_log.update_progress(95, "pushing to fieldwork_processed (background)")
        _push_async(source, out_dir, session, stem, op_log)
        op_log.finish_op()
        return out_dir
    except Exception as exc:
        logger.exception("pipeline failed")
        op_log.error_op(str(exc))
        raise


########
# Localization
########


@dataclass
class LocalizationRunOutput:
    """Everything the localize page needs to render one run."""

    result: "object"  # LocalizationResult
    query_frame: np.ndarray  # (H, W, 3) uint8 RGB
    query_intrinsics: np.ndarray  # (3, 3) — proportions seed or calibrated
    intrinsics_source: str  # "calibration file" | "proportions seed"
    ref_image_paths: list  # local paths, index-aligned with ref_frame_indices
    ref_extrinsics: np.ndarray  # (N, 4, 4) world-to-camera
    frame_sources: list  # per-frame 'reconstruction' | 'localized'


def _load_feedforward_result(out_dir: Path, load_world_points: bool = False):
    """Load the reconstruction result from the local zarr (lazy heavy import)."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    # Localization reads only the required member set (remote pulls already exclude the
    # dense arrays); skip decoding them for locally-generated scenes too. world_points is
    # opted in by the localizer path — CameraLocalizer.from_feedforward requires it.
    return FeedforwardResult.load_zarr(
        out_dir / "feedforward.zarr",
        load_depth=False,
        load_world_points=load_world_points,
        load_confidence=False,
        load_features=False,
        load_pixel_indices=False,
    )


def _stamp_db_provenance(zarr_path: Path, extractor_name: str, out_dir: Path) -> None:
    """Write build provenance from run_config.yaml onto the extractor's zarr group.

    Idempotent — safe to call on every run; older stores gain attrs on first touch.
    """
    cfg_path = Path(out_dir) / "run_config.yaml"
    attrs: dict = {"extractor": extractor_name}
    if cfg_path.exists():
        run_cfg = RunConfig.from_yaml(cfg_path)
        attrs.update(
            {
                "backbone": run_cfg.env_model,
                "frame_indices": list(run_cfg.frame_indices),
                "video_ref": run_cfg.video_ref,
            }
        )
    store = zarr.open(str(zarr_path), mode="a")
    group = store.require_group(f"local_features/{extractor_name}")
    for k, v in attrs.items():
        group.attrs[k] = v


def _build_localizer(
    result,
    config: LocalizationConfig,
    zarr_path: Path,
    op_log: OperationLog,
    cache=None,
    scene_key=None,
    frames_zarr: "Path | None" = None,
):
    """Load (or build, with progress) the feature DB; keep the localizer warm in the
    SceneCache so consecutive runs skip index reload and extractor model load."""
    from collab_splats.localization import CameraLocalizer
    from collab_splats.localization.extractors import BaseLocalExtractor

    if cache is not None and scene_key is not None:
        cached = cache.get(scene_key, f"localizer:{config.extractor}")
        if cached is not None:
            return cached

    extractor = BaseLocalExtractor.get(config.extractor)()

    def on_progress(done: int, total: int) -> None:
        # Only fires on a cache miss (DB build); scale into the 25→55% band
        op_log.update_progress(
            int(25 + 30 * (done + 1) / max(total, 1)),
            f"localize: building DB {done + 1}/{total}",
            log=False,
        )

    # Boundary adapter: build (images, ids) from the canonical store when present, else from
    # the result's export paths. Lazy genexpr → zero reads on a cache hit.
    if frames_zarr is not None:
        store = FrameStore.open(frames_zarr)
        frame_indices = store.frame_indices()
        images = (store.image_by_frame_idx(fi) for fi in frame_indices)
        ids = [f"frame_{int(fi):06d}.jpg" for fi in frame_indices]
    else:
        paths = [Path(p) for p in result.image_paths]
        images = (np.asarray(open_image(p).convert("RGB")) for p in paths)
        ids = [p.name for p in paths]

    localizer = CameraLocalizer.from_feedforward(
        result,
        images=images,
        ids=ids,
        extractor=extractor,
        extractor_name=config.extractor,
        zarr_path=zarr_path,
        progress_callback=on_progress,
    )
    if cache is not None and scene_key is not None:
        cache.put(scene_key, f"localizer:{config.extractor}", localizer)
    return localizer


def _resolve_query_intrinsics(config: LocalizationConfig) -> np.ndarray | None:
    """User-supplied YAML calibration when configured; else None → proportions seed."""
    if config.calibration_path:
        data = yaml.safe_load(Path(config.calibration_path).read_text())
        return np.asarray(data["K"], dtype=np.float32).reshape(3, 3)
    # No calibration → let CameraLocalizer.localize seed K from image proportions.
    return None


def _local_ref_paths(localizer, out_dir: Path) -> list:
    """Remap DB image paths (recorded on the machine that built the DB) to local files."""
    paths = []
    for p, src in zip(localizer.image_paths, localizer.frame_sources):
        sub = "frames" if src == "reconstruction" else "localized_frames"
        paths.append(Path(out_dir) / sub / Path(p).name)
    return paths


@dataclass
class BrowseData:
    """Stored-DB view of a scene: reconstruction cameras + previously localized poses."""

    extractor: str
    ref_extrinsics: np.ndarray  # (N, 4, 4) world-to-camera reconstruction cameras
    localized_extrinsics: np.ndarray  # (L, 4, 4) stored localized poses (L may be 0)
    localized_image_paths: list  # local localized_frames/ paths (existence not guaranteed)
    mesh_path: Path  # scene mesh (may not exist)


def read_localized_group(zarr_path: Path, extractor: str, out_dir: Path) -> "tuple[np.ndarray, list]":
    """Read stored localized poses + local image paths for one extractor (read-only, no GPU)."""
    store = zarr.open(str(zarr_path), mode="r")
    key = f"local_features/{extractor}/localized"
    if key not in store:
        return np.zeros((0, 4, 4), dtype=np.float32), []
    group = store[key]
    poses = np.asarray(group["extrinsics"])
    # image_paths attrs were recorded on the building machine — only basenames are portable
    names = [Path(p).name for p in group.attrs.get("image_paths", [])]
    return poses, [Path(out_dir) / "localized_frames" / n for n in names]


def load_browse_data(
    *,
    session: str,
    stem: str,
    extractor: str,
    source: SessionSource,
    base_dir: Path,
    op_log: OperationLog,
) -> BrowseData:
    """Non-GPU DB browse load: minimal pull if absent, then ref extrinsics + stored localized poses."""
    out_dir = Path(base_dir) / session / stem
    # Minimal pull only when the zarr is not yet local (same excludes as run_localization)
    if not (out_dir / "feedforward.zarr").exists():
        with op_log.step("browse: pulling reconstruction"):
            source.pull_processed(session, stem, out_dir, excludes=PULL_EXCLUDES)
    result = _load_feedforward_result(out_dir)
    loc_ext, loc_paths = read_localized_group(out_dir / "feedforward.zarr", extractor, out_dir)
    return BrowseData(
        extractor=extractor,
        ref_extrinsics=np.asarray(result.extrinsics),
        localized_extrinsics=loc_ext,
        localized_image_paths=loc_paths,
        mesh_path=out_dir / "mesh" / "mesh_tsdf.ply",
    )


def run_localization(
    *,
    query_video: Path,
    frame_idx: int,
    session: str,
    stem: str,
    config: LocalizationConfig,
    op_log: OperationLog,
    source: SessionSource,
    base_dir: Path,
    provenance: "dict | None" = None,
    cache=None,
) -> LocalizationRunOutput:
    """Localize one query-video frame against an existing reconstruction; optionally
    append the result to the localized/ DB group and push incrementally."""
    out_dir = Path(base_dir) / session / stem
    op_log.start_op(f"localize {Path(query_video).name}#{frame_idx}")
    try:
        with op_log.attach_logging("collab_splats"):
            # Reconstruction data: pull once (minimal set), then load from local zarr
            op_log.update_progress(5, "localize: pulling reconstruction")
            if not (out_dir / "feedforward.zarr").exists():
                source.pull_processed(session, stem, out_dir, excludes=PULL_EXCLUDES)
            op_log.update_progress(15, "localize: loading reconstruction")
            with op_log.step("localize: loading reconstruction"):
                result = _load_feedforward_result(out_dir, load_world_points=True)

            # Feature DB: warm-cache hit skips reload; zarr hit is fast; miss builds on GPU
            op_log.update_progress(25, f"localize: loading DB ({config.extractor})")
            with op_log.step(f"localize: DB ({config.extractor})"):
                # frames.zarr is the sole persistent frame store and is now pulled for processed
                # scenes too; guard defensively so any legacy scene without it falls back to image_paths.
                frames_zarr = out_dir / "frames.zarr"
                localizer = _build_localizer(
                    result,
                    config,
                    out_dir / "feedforward.zarr",
                    op_log,
                    cache=cache,
                    scene_key=(session, stem),
                    frames_zarr=frames_zarr if frames_zarr.exists() else None,
                )
            _stamp_db_provenance(out_dir / "feedforward.zarr", config.extractor, out_dir)

            # Query frame + intrinsics
            op_log.update_progress(55, f"localize: extracting frame {frame_idx}")
            frame = extract_frame(query_video, frame_idx)
            op_log.update_progress(60, "localize: resolving query intrinsics")
            K = _resolve_query_intrinsics(config)
            intr_source = "calibration file" if config.calibration_path else "proportions seed"

            # Pose: single-pose PnP + refinement — the DB is never modified here
            op_log.update_progress(70, "localize: matching + solving pose")
            with op_log.step("localize: matching + solving pose"):
                loc = localizer.localize(frame, K)
            # localize() seeds K from proportions when K is None — use what it actually used.
            K = loc.query_intrinsics if K is None else K
            op_log.append_line(
                f"localize: {loc.n_inliers}/{loc.n_correspondences} inliers"
                + ("" if loc.pose is not None else " — POSE FAILED")
            )

            # Persist: save the query frame locally, append to localized/, push new chunks
            if loc.pose is not None and config.append_to_db:
                op_log.update_progress(85, "localize: appending to DB")
                img_dir = out_dir / "localized_frames"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"{Path(query_video).stem}_f{frame_idx:06d}.jpg"
                Image.fromarray(frame).save(img_path)
                localizer.add_localized_frame(
                    img_path,
                    loc.pose,
                    K,
                    loc.query_features,
                    zarr_path=out_dir / "feedforward.zarr",
                    extractor_name=config.extractor,
                    provenance=provenance,
                )
                op_log.update_progress(92, "localize: pushing to fieldwork_processed (background)")
                _push_async(source, out_dir, session, stem, op_log)

            output = LocalizationRunOutput(
                result=loc,
                query_frame=frame,
                query_intrinsics=K,
                intrinsics_source=intr_source,
                ref_image_paths=_local_ref_paths(localizer, out_dir),
                ref_extrinsics=localizer.extrinsics,
                frame_sources=localizer.frame_sources,
            )
        op_log.finish_op()
        return output
    except Exception as exc:
        logger.exception("localization failed")
        op_log.error_op(str(exc))
        raise
