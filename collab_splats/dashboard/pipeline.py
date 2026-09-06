"""Run the primitives pipeline for one video: sample -> pointcloud -> mesh -> semantics."""

from __future__ import annotations

import functools
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
from collab_splats.mesh.utils import persist_mesh_vertex_features, pointcloud_to_mesh
from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator,
    VGGTXCreator,
)
from collab_splats.pointcloud.utils import lift_features
from collab_splats.preproc import extract_frame
from collab_splats.preproc import frames as fr
from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.sampling import (
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
from collab_splats.remote import PULL_EXCLUDES, SceneSource
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features.base import BaseFeatureExtractor
from collab_splats.semantics.utils import (
    cache_store_path,
    extract_feature_cache,
    load_feature_maps,
    load_point_features,
    point_features_cached,
    write_point_features,
)
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


def _write_images_dir(
    frames: list[np.ndarray],
    records: list[dict],
    images_dir: Path,
    *,
    video_path: Path,
    method: str,
    fps: float | None,
    max_frames: int,
) -> None:
    """
    Write the scene's canonical images/ directory and its frames.json manifest.

    Args:
        frames: RGB uint8 keyframes, one per record.
        records: selection records carrying the source 'frame_idx'.
        images_dir: <scene>/images, the sole persistent frame store.
        video_path: source video, stamped into provenance with its mtime.
        method: sampling method name.
        fps: target fps when method is 'fps', else None.
        max_frames: selection cap the sampler ran under.
    """
    prov = {
        "video_path": str(video_path),
        "video_mtime": Path(video_path).stat().st_mtime,
        "method": method,
        "fps": fps,
        "max_frames": max_frames,
    }
    fr.write_frames(images_dir, frames, records, prov)


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


def _extract_semantics(extractor_name: str, images_dir: Path, out_dir: Path) -> None:
    """Extract + cache patch features straight from the scene's images/ directory.

    Args:
        extractor_name: registry key of the extractor to run.
        images_dir: the scene's images/ directory of keyframes.
        out_dir: the scene's semantics dir.
    """
    # The returned cache path is deliberately dropped: every consumer resolves the store by
    # glob (cache_store_path), including the viewer's legacy lift, which has no handle to thread.
    extract_feature_cache(BaseFeatureExtractor.get(extractor_name)(), images_dir, out_dir)


########
# Feature-compression autoencoder policy — ONE gate, shared by both dashboard fit paths
########

# base.yaml is the single source of defaults for the whole project (see configs/README.md);
# the dashboard reads the same semantics: block rather than carrying its own numbers.
_CONFIG_DIR = Path(__file__).parents[2] / "configs"


@dataclass(frozen=True)
class AutoencoderPolicy:
    """Latent width + fit gate applied to every dashboard feature-compression autoencoder."""

    latent_dim: int | None  # None == semantics.n_components: null == no compression
    target_cosine: float
    max_epochs: int


@functools.lru_cache(maxsize=1)
def semantics_ae_policy() -> AutoencoderPolicy:
    """Read the one autoencoder policy from configs/base.yaml's semantics: block.

    Shared by the fresh-reconstruction lift (_lift_and_compress) and the legacy self-upgrade
    (viewer._save_point_features) so a scene is never held to two different fidelity bars.
    """
    semantics = yaml.safe_load((_CONFIG_DIR / "base.yaml").read_text())["semantics"]
    return AutoencoderPolicy(
        latent_dim=semantics["n_components"],
        target_cosine=float(semantics["target_cosine"]),
        max_epochs=int(semantics["max_epochs"]),
    )


def resolve_latent_dim(input_dim: int, latent_dim: "int | None") -> int:
    """Clamp the configured latent width to the input width, warning when the clamp binds.

    The clamp is a real guard (tiny fixtures, and n_components: null asks for full width),
    but an identity-width autoencoder buys no compression AND still costs a lossy
    encode/decode round-trip — strictly worse than storing the features directly.
    """
    resolved = input_dim if latent_dim is None else min(int(latent_dim), input_dim)
    if resolved >= input_dim:
        logger.warning(
            "autoencoder latent width clamped to the %d-D input (requested %s) — compression is a "
            "no-op and the encode/decode round-trip is lossy",
            input_dim,
            latent_dim,
        )
    return resolved


########
# Semantics layout resolution (read path)
########


def resolve_semantics_dir(scene_dir: Path) -> "Path | None":
    """A scene's flat `{scene}/semantics/` dir, or None when it has none.

    Flat only, deliberately. The dashboard is a browser over its OWN scenes: its loader gates on
    and reads flat `{scene}/pointcloud.zarr` and flat `{scene}/mesh.ply`, so a published
    (Reconstructor) scene — everything one level deeper under `{scene}/{backend}/` — fails on the
    pointcloud before semantics is ever consulted. Resolving a backend-keyed semantics dir would
    only serve a hybrid tree (flat pointcloud + nested semantics) that no writer produces.
    Migrating the dashboard to the published layout instead would orphan every dashboard scene
    already on disk, which is why the read path stays flat.

    A flat dir holding only the 2D patch cache still resolves: that is exactly the legacy scene
    viewer.ensure_lifted lifts on demand, and None would strand it with no semantics forever.
    """
    flat = Path(scene_dir) / "semantics"
    return flat if flat.is_dir() else None


def _lift_and_compress(result, semantics_dir: Path, op_log: OperationLog) -> None:
    """Train a feature-compression autoencoder, lift COMPRESSED maps to points, cache latent codes.

    Order matches the semantic_lifting tutorial and is what makes this fast: train the AE on the
    2D patch features, encode each map (D→latent) on GPU, then lift the small latent maps to
    points — lift_features cost scales with channel count, so lifting `latent` (e.g. 64) instead
    of the full D (e.g. 768) is ~D/latent× cheaper. The latent codes are cached as they are; the
    decode to D happens on read (load_point_features). Everything except the lift runs on the GPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_maps = load_feature_maps(cache_store_path(semantics_dir))  # list of (D, H_p, W_p) on CPU
    input_dim = feature_maps[0].shape[0]

    # Train the AE on flattened 2D patch features (all frames) — GPU; stream loss to the log.
    # Width + fit gate come from the shared config policy, same as the legacy self-upgrade.
    op_log.update_progress(80, "semantics: fitting autoencoder")
    t = time.perf_counter()
    policy = semantics_ae_policy()
    ae = FeatureAutoencoder(input_dim=input_dim, latent_dim=resolve_latent_dim(input_dim, policy.latent_dim)).to(device)
    patches = torch.cat([fm.flatten(1).T for fm in feature_maps]).to(device)  # (N*H_p*W_p, D)
    ae.fit(
        patches,
        epochs=policy.max_epochs,
        target_cosine=policy.target_cosine,
        on_epoch=lambda e, t_, loss: op_log.append_line(f"semantics: autoencoder epoch {e}/{t_}  loss={loss:.4f}"),
    )
    op_log.append_line(f"semantics: autoencoder fit in {time.perf_counter() - t:.1f}s")

    # Encode maps (GPU), lift the small latent maps to points (cheap), decode per-point (GPU)
    op_log.update_progress(90, "semantics: lifting compressed features")
    t = time.perf_counter()
    with torch.no_grad():
        compressed_maps = [ae.encode(fm.to(device)).detach().cpu() for fm in feature_maps]
    compressed_pts = lift_features(compressed_maps, result)  # (P, latent) — fast
    op_log.update_progress(94, "semantics: caching lifted features")
    # Persist LATENT codes + weights (not decoded 768-D): same artifact pair the
    # Reconstructor path writes, ~12x smaller, and decodable on read.
    write_point_features(
        Path(semantics_dir),
        cache_store_path(semantics_dir).stem,
        compressed_pts.detach().cpu().numpy(),
        ae,
    )
    op_log.append_line(f"semantics: lift + encode + cache in {time.perf_counter() - t:.1f}s")


def _transfer_mesh_features(result, out_dir: Path, *, k: int = 5, sdf_trunc: float = 0.03) -> None:
    """Transfer cached point features onto the TSDF mesh vertices and persist vertex_features.npy.

    No-op (logged) if the mesh or the lifted point features are missing — neither is fatal
    to the run.
    """
    mesh_path = Path(out_dir) / "mesh.ply"
    sem_dir = Path(out_dir) / "semantics"
    has_features = point_features_cached(sem_dir)
    if not mesh_path.exists() or not has_features:
        logger.warning("mesh feature transfer skipped: mesh=%s features=%s", mesh_path.exists(), has_features)
        return
    # DECODED features, not latent codes: the only reader of vertex_features.npy
    # (viewer.load_mesh_vertex_features) does no decode and feeds score_queries, which
    # compares against full-dim text embeddings. Matches viewer.ensure_mesh_features,
    # which derives the same array from decoded point features on legacy scenes.
    point_features = load_point_features(sem_dir)
    persist_mesh_vertex_features(mesh_path, result.points, point_features, k=k, sdf_trunc=sdf_trunc)


def _sample(video_path: Path, config: RunConfig, out_dir: Path, op_log: OperationLog):
    """
    Sample frames per the configured method; return (frames, records).
    """

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

    # Measure first, select second. The report is reused by existence, so a re-run is free.
    op_log.update_progress(4, "sampling: measuring video quality")
    report = load_video_quality(video_path, out_dir / "video_quality_report.json")

    # Records carry true source frame indices for every method. Each method has ONE
    # density knob, so each branch passes only its own.
    op_log.update_progress(5, f"sampling: {config.sampling_method}")
    method = config.sampling_method

    if method == "fps":
        return sample_fps(
            str(video_path),
            fps=config.fps,
            max_frames=config.max_frames,
            report=report,
            on_progress=on_progress,
        )

    if method == "optical_flow":
        return sample_optical_flow(
            str(video_path),
            min_disparity=config.min_disparity,
            max_frames=config.max_frames,
            report=report,
            on_progress=on_progress,
        )

    return sample_uniform(
        str(video_path),
        max_frames=config.max_frames,
        report=report,
        on_progress=on_progress,
    )


########
# Orchestrator
########


def _push_async(source: SceneSource, out_dir: Path, scene: str, op_log: OperationLog) -> None:
    """Push the output tree to the processed bucket in a detached, non-fatal thread."""

    def _worker() -> None:
        t0 = time.perf_counter()
        op_log.append_line("push: uploading to environments-processed")
        try:
            source.push_outputs(out_dir, scene, on_line=op_log.append_line)
            op_log.append_line(f"push: done in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:  # non-fatal: outputs already on local disk
            logger.exception("push failed")
            op_log.append_line(f"push: FAILED ({exc})")

    threading.Thread(target=_worker, daemon=True).start()


def run_pipeline(
    *,
    video_path: Path,
    scene: str,
    config: RunConfig,
    op_log: OperationLog,
    source: SceneSource,
    base_dir: Path,
) -> Path:
    """Execute the full pipeline; write outputs under base_dir/scene; push in background."""
    out_dir = Path(base_dir) / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    op_log.start_op(scene)
    # Bridge collab_splats module logs (e.g. creator/semantics '%d/%d frames') into the dashboard log.
    try:
        with op_log.attach_logging("collab_splats"):
            # Sample frames from video and persist for creator + viewer
            t = time.perf_counter()
            frames, records = _sample(Path(video_path), config, out_dir, op_log)
            sampling_method = config.sampling_method
            images_dir = out_dir / "images"
            _write_images_dir(
                frames,
                records,
                images_dir,
                video_path=video_path,
                method=sampling_method,
                fps=config.fps if sampling_method == "fps" else None,
                max_frames=config.max_frames,
            )
            # images/ is the sole frame store: setup_inference and semantics extraction both
            # read it directly, and localization ref thumbnails resolve pixels through it (see
            # _build_result_figures's images_dir threading). No second staged copy.
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
            creator.setup_inference(images_dir)
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
            result.save_zarr(
                out_dir / "pointcloud.zarr",
                extra_attrs={"method": "feedforward", "backend": config.env_model},
            )

            # Mesh from TSDF depth fusion
            op_log.update_progress(60, "mesh: tsdf fusion")
            t = time.perf_counter()
            pointcloud_to_mesh(
                result,
                out_dir,
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
            _extract_semantics(config.semantic_extractor, images_dir, out_dir / "semantics")
            op_log.append_line(f"semantics: extracted in {time.perf_counter() - t:.1f}s")

            # Lift features to points + train compression autoencoder eagerly (instant queries later);
            # _lift_and_compress emits its own 'semantics: ...' substep labels.
            _lift_and_compress(result, out_dir / "semantics", op_log)

            # Transfer lifted point features onto the mesh vertices (same feature space -> mesh is queryable).
            op_log.update_progress(94, "mesh: transferring features to vertices")
            _transfer_mesh_features(result, out_dir)

            # Persist provenance: frame indices + video ref baked into run_config.yaml. The ref is
            # scene-relative — the flat curated layout has no path prefix above the scene id.
            config.to_yaml(
                out_dir / "run_config.yaml",
                video_ref=f"{scene}/{Path(video_path).name}",
            )

        # Local outputs ready: mark complete and push in the background (non-fatal).
        op_log.update_progress(95, "pushing to environments-processed (background)")
        _push_async(source, out_dir, scene, op_log)
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


def _load_feedforward_result(out_dir: Path, load_world_points: bool = False, load_images: bool = False):
    """Load the reconstruction result from the local zarr (lazy heavy import)."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    # Localization reads only the required member set (remote pulls already exclude the
    # dense arrays); skip decoding them for locally-generated scenes too. world_points and
    # images are opted in by the localizer path — CameraLocalizer.from_feedforward requires
    # world_points, and the pairwise LocalMatcher additionally needs model-res ref images.
    return FeedforwardResult.load_zarr(
        out_dir / "pointcloud.zarr",
        load_depth=False,
        load_world_points=load_world_points,
        load_images=load_images,
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
    images_dir: "Path | None" = None,
):
    """Load (or build, with progress) the feature DB; keep the localizer warm in the
    SceneCache so consecutive runs skip index reload and extractor model load."""
    from collab_splats.localization import CameraLocalizer
    from collab_splats.localization.extractors import LocalMatcher

    if cache is not None and scene_key is not None:
        cached = cache.get(scene_key, f"localizer:{config.matcher}")
        if cached is not None:
            return cached

    extractor = LocalMatcher(config.matcher)

    def on_progress(done: int, total: int) -> None:
        # Only fires on a cache miss (DB build); scale into the 25→55% band
        op_log.update_progress(
            int(25 + 30 * (done + 1) / max(total, 1)),
            f"localize: building DB {done + 1}/{total}",
            log=False,
        )

    # Boundary adapter: build (images, ids) from the scene's images/ directory when present,
    # else from the result's export paths. Lazy genexpr → zero reads on a cache hit.
    paths = fr.frame_paths(images_dir) if images_dir is not None else [Path(p) for p in result.image_paths]
    images = (np.asarray(open_image(p).convert("RGB")) for p in paths)
    ids = [p.name for p in paths]

    localizer = CameraLocalizer.from_feedforward(
        result,
        images=images,
        ids=ids,
        extractor=extractor,
        extractor_name=config.matcher,
        zarr_path=zarr_path,
        progress_callback=on_progress,
    )
    if cache is not None and scene_key is not None:
        cache.put(scene_key, f"localizer:{config.matcher}", localizer)
    return localizer


def _resolve_query_intrinsics(config: LocalizationConfig) -> np.ndarray | None:
    """User-supplied YAML calibration when configured; else None → proportions seed."""
    if config.calibration_path:
        data = yaml.safe_load(Path(config.calibration_path).read_text())
        return np.asarray(data["K"], dtype=np.float32).reshape(3, 3)
    # No calibration → let CameraLocalizer.localize seed K from image proportions.
    return None


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
    scene: str,
    extractor: str,
    source: SceneSource,
    base_dir: Path,
    op_log: OperationLog,
) -> BrowseData:
    """Non-GPU DB browse load: minimal pull if absent, then ref extrinsics + stored localized poses."""
    out_dir = Path(base_dir) / scene
    # Minimal pull only when the zarr is not yet local (same excludes as run_localization)
    if not (out_dir / "pointcloud.zarr").exists():
        with op_log.step("browse: pulling reconstruction"):
            source.pull_processed(scene, out_dir, excludes=PULL_EXCLUDES)
    result = _load_feedforward_result(out_dir)
    loc_ext, loc_paths = read_localized_group(out_dir / "pointcloud.zarr", extractor, out_dir)
    return BrowseData(
        extractor=extractor,
        ref_extrinsics=np.asarray(result.extrinsics),
        localized_extrinsics=loc_ext,
        localized_image_paths=loc_paths,
        mesh_path=out_dir / "mesh.ply",
    )


def run_localization(
    *,
    query_video: Path,
    frame_idx: int,
    scene: str,
    config: LocalizationConfig,
    op_log: OperationLog,
    source: SceneSource,
    base_dir: Path,
    provenance: "dict | None" = None,
    cache=None,
) -> LocalizationRunOutput:
    """Localize one query-video frame against an existing reconstruction; optionally
    append the result to the localized/ DB group and push incrementally."""
    out_dir = Path(base_dir) / scene
    op_log.start_op(f"localize {Path(query_video).name}#{frame_idx}")
    try:
        with op_log.attach_logging("collab_splats"):
            # Reconstruction data: pull once (minimal set), then load from local zarr
            op_log.update_progress(5, "localize: pulling reconstruction")
            if not (out_dir / "pointcloud.zarr").exists():
                source.pull_processed(scene, out_dir, excludes=PULL_EXCLUDES)
            op_log.update_progress(15, "localize: loading reconstruction")
            with op_log.step("localize: loading reconstruction"):
                result = _load_feedforward_result(out_dir, load_world_points=True, load_images=True)

            # Feature DB: warm-cache hit skips reload; zarr hit is fast; miss builds on GPU
            op_log.update_progress(25, f"localize: loading DB ({config.matcher})")
            with op_log.step(f"localize: DB ({config.matcher})"):
                # images/ is the sole persistent frame store and is now pulled for processed
                # scenes too; guard defensively so any legacy scene without it falls back to image_paths.
                images_dir = out_dir / "images"
                localizer = _build_localizer(
                    result,
                    config,
                    out_dir / "pointcloud.zarr",
                    op_log,
                    cache=cache,
                    scene_key=scene,
                    images_dir=images_dir if images_dir.is_dir() else None,
                )
            _stamp_db_provenance(out_dir / "pointcloud.zarr", config.matcher, out_dir)

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
                    zarr_path=out_dir / "pointcloud.zarr",
                    extractor_name=config.matcher,
                    provenance=provenance,
                )
                op_log.update_progress(92, "localize: pushing to environments-processed (background)")
                _push_async(source, out_dir, scene, op_log)

            # DB ids are basenames recorded on the machine that built the DB; resolve each to a
            # local file — reconstruction frames live in images/, localized ones in localized_frames/.
            ref_image_paths = [
                out_dir / ("images" if src == "reconstruction" else "localized_frames") / Path(p).name
                for p, src in zip(localizer.image_paths, localizer.frame_sources)
            ]

            output = LocalizationRunOutput(
                result=loc,
                query_frame=frame,
                query_intrinsics=K,
                intrinsics_source=intr_source,
                ref_image_paths=ref_image_paths,
                ref_extrinsics=localizer.extrinsics,
                frame_sources=localizer.frame_sources,
            )
        op_log.finish_op()
        return output
    except Exception as exc:
        logger.exception("localization failed")
        op_log.error_op(str(exc))
        raise
