# collab_splats/pointcloud/vda.py
"""
Video-Depth-Anything metric depth for the SfM path.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Repo root -> third_party clone (setup.sh owns creation); module-level so tests can monkeypatch
VDA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Video-Depth-Anything"


########################################################################
# Model loading
########################################################################


def _load_vda_model(device: str):
    """
    Construct the VDA metric vitl model on `device` from the hub checkpoint.

    - Split out from generate_vda_depth so the write path is testable without a GPU or the clone.
    """
    # Lazy heavy import — VDA lives in a third_party clone (its root on sys.path), not
    # site-packages. Upstream HEAD (4f5ae23) has no metric_depth/ subdir: `video_depth_anything/`
    # sits at the clone root and `video_depth.py:27` imports a TOP-LEVEL `utils` namespace package
    # (`utils/util.py`) from the same root. Probed 2026-08-23: no foreign top-level `utils` in the
    # venv — a regular `utils` package anywhere on sys.path would shadow VDA's namespace one
    # regardless of insert order, so re-probe if a dependency ever ships one.
    if not (VDA_ROOT / "video_depth_anything").is_dir():
        raise ImportError(
            f"Video-Depth-Anything clone not found at {VDA_ROOT} — run setup.sh (clones the repo at 4f5ae23)"
        )
    if str(VDA_ROOT) not in sys.path:
        sys.path.insert(0, str(VDA_ROOT))
    from video_depth_anything.video_depth import VideoDepthAnything

    # Upstream weights live on the hub, not in the clone — the clone carries source only.
    # hub failures (no network, or offline with a cold cache) name neither VDA nor a remedy,
    # so re-raise with both; the download is ~1.5 GB and lands in the HF_HOME cache.
    try:
        ckpt = hf_hub_download(
            repo_id="depth-anything/Metric-Video-Depth-Anything-Large",
            filename="metric_video_depth_anything_vitl.pth",
        )
    except Exception as exc:
        raise RuntimeError(
            "VDA metric checkpoint unavailable (depth-anything/Metric-Video-Depth-Anything-Large, "
            f"~1.5 GB, cached under HF_HOME): {exc}"
        ) from exc

    # metric=True loads the metric head AND disables infer_video_depth's cross-window
    # scale-and-shift chaining (video_depth.py:135), so consecutive windows are stitched on the
    # head's own absolute output rather than fitted to each other. Measured 2026-08-26: this is
    # why a full-video pass does not improve metric contiguity.
    # Constructor values: run.py:45-49 model_configs["vitl"].
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.to(device).eval()


########################################################################
# Depth generation
########################################################################


def vda_depth_complete(out_dir: Path, names: list[str]) -> bool:
    """
    True when out_dir/depth_vda/images/npy holds exactly one .npy per stem in `names`.

    - Public so a caller can prune a stale depth_vda/ BEFORE generating: generate_vda_depth only
      ever ADDS maps, so a stem left over from a different keyframe set would keep this
      set-equality gate false forever and re-run the full GPU inference on every subsequent run.
      That is what Reconstructor._run_sfm uses it for; it does NOT gate on this before
      materialising `frames`, which it builds unconditionally. evals/scripts/eval.py does gate
      its own image decode on it.
    - Never needed to skip redundant inference: generate_vda_depth applies this same check itself
      and returns the cached stack.
    """
    npy_dir = Path(out_dir) / "depth_vda" / "images" / "npy"
    return npy_dir.is_dir() and {p.stem for p in npy_dir.glob("*.npy")} == {Path(n).stem for n in names}


def generate_vda_depth(
    frames: np.ndarray,
    out_dir: Path,
    names: list[str],
    *,
    depth_width: int = 518,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run VDA metric depth over the keyframes; write InstantSfM's depth layout and return the stack.

    - frames: (N, H, W, 3) uint8 RGB in images/ order; names: one filename per frame.
    - Writes out_dir/depth_vda/images/npy/<stem>.npy — the layout instantsfm's
      ReadDepthsIntoFeatures single-camera branch consumes (data_reader.py:404-407).
    - depth_width: VDA returns depth at input resolution (300 x 1080p = 2.5 GB), too heavy for
      pointcloud.zarr; each map is nearest-resized to this width (no blending across depth
      discontinuities). 518 matches the feedforward model-res convention so downstream stages
      see the same resolution class. Any depth resolution is valid for SfM — instantsfm's
      sample_depth_at_pixel normalises keypoints by camera w/h.
    - Returns (N, h, depth_width) float32, the same maps that were written.
    - Idempotent: an exact per-stem npy set is loaded and returned without running inference.
      A complete cached set wins outright — depth_width is NOT re-applied to it, so the returned
      width is whatever the run that wrote the cache used.

    Attribution: inference pattern follows
    https://github.com/DepthAnything/Video-Depth-Anything @ 4f5ae23 run.py:45-57.
    """
    # Names are consumed positionally against the frame stack, so the two must be one-to-one
    if len(names) != len(frames):
        raise ValueError(f"names ({len(names)}) and frames ({len(frames)}) must align one-to-one")

    npy_dir = Path(out_dir) / "depth_vda" / "images" / "npy"
    stems = [Path(n).stem for n in names]

    # Idempotent skip: same exact-stem-set gate the callers use, so the two cannot disagree
    if vda_depth_complete(out_dir, names):
        logger.info("VDA depth exists at %s (%d maps) — loading", npy_dir, len(stems))
        return np.stack([np.load(npy_dir / f"{s}.npy") for s in stems]).astype(np.float32)

    model = _load_vda_model(device)

    # Metric inference over the whole sequence, at upstream's own 518 input resolution (the same
    # model-res convention depth_width follows). target_fps reaches nothing: infer_video_depth
    # only echoes it back (4f5ae23 video_depth.py:70 signature, :162 return) and resamples
    # nothing, so any value does — named here so that is obvious at the call site.
    logger.info("VDA metric inference: %d frames (writing %d maps)", len(frames), len(names))
    depths, _fps = model.infer_video_depth(frames, target_fps=1.0, input_size=518, device=device, fp32=False)
    depths = np.asarray(depths, dtype=np.float32)

    # Free the GPU before the caller's InstantSfM CUDA step — resize/write below is CPU-only
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Nearest-resize to depth_width and write one map per frame, keyed by image stem
    h, w = depths.shape[1:3]
    depth_hw = (int(round(depth_width * h / w)), depth_width)
    npy_dir.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(names), depth_hw[0], depth_hw[1]), dtype=np.float32)
    for i, (stem, depth) in enumerate(zip(stems, depths, strict=True)):
        small = cv2.resize(depth, (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        np.save(npy_dir / f"{stem}.npy", small)
        out[i] = small

    logger.info("VDA depths written: %s (%d maps @ %dx%d)", npy_dir, len(names), depth_hw[1], depth_hw[0])
    return out
