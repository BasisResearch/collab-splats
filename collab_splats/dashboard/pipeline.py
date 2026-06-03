"""Run the primitives pipeline for one video: sample -> pointcloud -> mesh -> semantics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from PIL import Image

from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource
from collab_splats.mesh.utils import pointcloud_to_mesh
from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator,
    VGGTXCreator,
)
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
    lz4 = BloscCodec(cname="lz4", clevel=3)
    root = zarr.open_group(str(path), mode="w")
    root.create_array("frames", data=arr, chunks=(1,) + arr.shape[1:], compressors=[lz4])


def _write_frames_jpegs(frames: list[np.ndarray], frames_dir: Path) -> Path:
    """Write frames as zero-padded JPEGs for creators that consume an image dir."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(frames_dir / f"{i:05d}.jpg")
    return frames_dir


def _build_creator(env_model: str, conf: float):
    """Instantiate the selected feedforward creator with its confidence arg."""
    if env_model == "vggt_omega":
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


def _sample(video_path: Path, config: RunConfig, op_log: OperationLog):
    """Sample frames per the configured method; return (frames, indices)."""
    info = get_video_info(str(video_path))

    def on_progress(done: int, total: int) -> None:
        op_log.update_progress(int(5 + 15 * done / max(total, 1)), "sampling frames")

    if config.sampling_method == "optical_flow":
        frames, _ = sample_frames_optical_flow(
            str(video_path),
            min_disparity=config.min_disparity,
            max_frames=config.max_frames,
            on_progress=on_progress,
            verbose=False,
        )
        indices = list(range(len(frames)))
    else:
        duration_s = info.get("duration_s") or (
            info["total_frames"] / (info.get("fps") or 30.0)
        )
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
    """Execute the full pipeline; write outputs under base_dir/session/stem; push on success."""
    out_dir = Path(base_dir) / session / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    op_log.start_op(f"{session}/{stem}")
    try:
        # Sample frames from video and persist for creator + viewer
        frames, indices = _sample(Path(video_path), config, op_log)
        _write_frames_zarr(frames, out_dir / "frames.zarr")
        image_dir = _write_frames_jpegs(frames, out_dir / "frames")
        config.frame_indices = list(indices)

        # Run feedforward pointcloud reconstruction
        op_log.update_progress(25, f"pointcloud: {config.env_model}")
        creator = _build_creator(config.env_model, config.conf_threshold)
        creator.reconstruct(image_dir, out_dir)
        result = creator.outputs
        result.save_zarr(out_dir / "feedforward.zarr")

        # Mesh from TSDF depth fusion
        op_log.update_progress(60, "mesh (TSDF)")
        pointcloud_to_mesh(
            result,
            out_dir / "mesh",
            method="open3d_tsdf",
            voxel_size=config.mesh_voxel_size,
            sdf_trunc=config.mesh_sdf_trunc,
            depth_trunc=config.mesh_depth_trunc,
            clean_repair=config.mesh_clean_repair,
        )

        # Extract and cache semantic patch features
        op_log.update_progress(80, f"semantics: {config.semantic_extractor}")
        _extract_semantics(config.semantic_extractor, image_dir, out_dir / "semantics")

        # Persist provenance: frame indices + video ref baked into run_config.yaml
        config.to_yaml(
            out_dir / "run_config.yaml",
            video_ref=f"reconstruction/{session}/{stem}/{Path(video_path).name}",
        )

        # Push full output tree on success only
        op_log.update_progress(95, "pushing to fieldwork_processed")
        source.push_outputs(out_dir, session, stem)
        op_log.finish_op()
        return out_dir
    except Exception as exc:
        logger.exception("pipeline failed")
        op_log.error_op(str(exc))
        raise
