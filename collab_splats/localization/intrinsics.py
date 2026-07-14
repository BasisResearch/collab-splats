"""EXPERIMENTAL — query-camera intrinsics estimation via single-frame feedforward inference.

Feedforward backbones (VGGT-X, MapAnything) predict per-frame intrinsics; running one on a
single query frame yields an approximate pinhole K when the query camera is uncalibrated.
This path is under validation (see spec 2026-07-14): errors of a few percent in focal are
typical and are partially absorbed by pycolmap's focal refinement during PnP (enabled by
default in CameraLocalizer). Prefer a real calibration when one exists.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def estimate_intrinsics(frame: np.ndarray, creator=None) -> np.ndarray:
    """Estimate a (3, 3) pinhole K for one RGB frame, rescaled to the frame's resolution.

    Args:
        frame:   (H, W, 3) uint8 RGB query frame.
        creator: Feedforward creator instance (load_model/setup_inference/run_inference/
                 postprocess/outputs surface). Defaults to VGGTXCreator — imported lazily
                 because the feedforward stack is a heavy optional dependency.
    """
    if creator is None:
        # Heavy import kept inside the function: pulls the full reconstruction stack
        from collab_splats.pointcloud.feedforward import VGGTXCreator

        creator = VGGTXCreator()

    # Stage the frame as a one-image directory — the creator API consumes image dirs
    with tempfile.TemporaryDirectory() as td:
        Image.fromarray(frame).save(Path(td) / "00000.jpg")
        creator.load_model()
        creator.setup_inference(Path(td))
        creator.run_inference()
        creator.postprocess()

    result = creator.outputs
    K = np.asarray(result.intrinsics[0], dtype=np.float64).copy()

    # Rescale from the model's inference resolution to the query frame's resolution.
    # result.images is (N, 3, H, W) channel-first (torch.Tensor in the real pipeline,
    # per FeedforwardResult) — read .shape directly, no np.asarray (would fail on CUDA).
    proc = result.images
    h_proc, w_proc = int(proc.shape[2]), int(proc.shape[3])
    h_q, w_q = frame.shape[:2]
    K[0, :] *= w_q / w_proc
    K[1, :] *= h_q / h_proc

    logger.info(
        "estimate_intrinsics (EXPERIMENTAL): fx=%.1f fy=%.1f cx=%.1f cy=%.1f @ %dx%d",
        K[0, 0], K[1, 1], K[0, 2], K[1, 2], w_q, h_q,
    )
    return K.astype(np.float32)
