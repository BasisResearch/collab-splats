"""Run the primitives pipeline for one video: sample -> pointcloud -> mesh -> semantics."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image
from zarr.codecs import BloscCodec

from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource
from collab_splats.mesh.utils import persist_mesh_vertex_features, pointcloud_to_mesh
from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator,
    VGGTXCreator,
)
from collab_splats.pointcloud.utils import lift_features
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features.base import BaseFeatureExtractor
from collab_splats.utils.frame_sampling import (
    get_video_info,
    sample_frames_fps,
    sample_frames_optical_flow,
)

# VGGTOmegaCreator requires the vggt-omega submodule; only available when installed.
try:
    from collab_splats.pointcloud.feedforward import VGGTOmegaCreator
except ImportError:
    VGGTOmegaCreator = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

########
# Helpers
########


def _write_frames_zarr(frames: list[np.ndarray], path: Path) -> None:
    """Write RGB frames (N, H, W, 3) uint8 to a zarr group at path/frames."""
    arr = np.stack(frames).astype(np.uint8)
    lz4 = BloscCodec(cname="lz4")
    store = zarr.open(str(path), mode="w")
    store.create_array("frames", data=arr, chunks=(1,) + arr.shape[1:], compressors=lz4)


def _write_frames_jpegs(frames: list[np.ndarray], frames_dir: Path) -> Path:
    """Write frames as zero-padded JPEGs for creators that consume an image dir."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(frames_dir / f"{i:05d}.jpg")
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
    """Sample frames per the configured method; return (frames, indices)."""
    info = get_video_info(str(video_path))

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
    if config.sampling_method == "optical_flow":
        frames, _ = sample_frames_optical_flow(
            str(video_path),
            min_disparity=config.min_disparity,
            max_frames=config.max_frames,
            on_progress=on_progress,
            verbose=False,
        )
        # optical-flow sampler returns score dicts, not source frame numbers; indices are positional
        indices = list(range(len(frames)))
    else:
        duration_s = info.get("duration_s") or (info["total_frames"] / (info.get("fps") or 30.0))
        target_fps = config.max_frames / max(duration_s, 1.0)
        frames, indices = sample_frames_fps(
            str(video_path),
            fps=target_fps,
            max_frames=config.max_frames,
            on_progress=on_progress,
            verbose=False,
        )
    return frames, indices


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
            frames, indices = _sample(Path(video_path), config, op_log)
            _write_frames_zarr(frames, out_dir / "frames.zarr")
            image_dir = _write_frames_jpegs(frames, out_dir / "frames")
            config.frame_indices = list(indices)
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
