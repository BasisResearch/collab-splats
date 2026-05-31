"""VGGT-SPARK feedforward backend: diagnostic creator for parity testing.

Provides:
  VGGTSPARKCreator  — feedforward creator using VGGT-SPARK weights/architecture

Purpose: isolate whether the ATE parity gap vs VGGT-SLAM is model-specific
(VGGT-X vs VGGT-SPARK weights/architecture) or graph-construction-specific.
VGGT-SLAM uses VGGT-SPARK as its feedforward backbone; this creator runs the
same SPARK module through our pipeline unchanged.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .vggtx import VGGTXCreator

logger = logging.getLogger(__name__)

# Absolute path to the VGGT-SPARK third-party source tree.
# Inserted into sys.path only during _load_model, then removed.
_VGGT_SPARK_ROOT: str = str(
    Path(__file__).resolve().parents[4] / "third_party" / "vggt_spark"
)


########################################################################
########## Creator #####################################################
########################################################################


@dataclass
class VGGTSPARKCreator(VGGTXCreator):
    """Pointcloud via VGGT-SPARK feedforward pose + depth estimation.

    Identical pipeline to VGGTXCreator but loads the model from the SPARK
    source tree (``third_party/vggt_spark/``), which ships with native
    ``compute_similarity=True`` support in ``VGGT.forward()``.

    Use this creator to test whether the ATE parity gap vs VGGT-SLAM is
    caused by model differences (VGGT-X vs VGGT-SPARK) or by our
    graph-construction logic.

    Attributes:
        Inherits all attributes from VGGTXCreator.  ``chunk_size`` is NOT
        forwarded to ``from_pretrained`` because VGGT-SPARK's ``__init__``
        does not accept it.
    """

    # Override class identity — name used in eval output prefixes and registry
    name: ClassVar[str] = "vggt_spark"
    registry_name: ClassVar[str] = "vggt_spark"

    def _load_model(self, device: str) -> Any:
        """Load VGGT-SPARK from HuggingFace via the SPARK import path.

        Temporarily prepends ``third_party/vggt_spark`` to sys.path so that
        ``import vggt`` resolves to the SPARK implementation.  The path is
        removed immediately after import, keeping the session sys.path clean.

        SPARK's ``__init__`` does not accept ``chunk_size``, so it is not
        forwarded.  The ``from_pretrained`` call uses the same HuggingFace
        model ID as VGGTXCreator (``facebook/VGGT-1B``).

        Args:
            device: Target device string (e.g. ``"cuda"`` or ``"cpu"``).

        Returns:
            VGGT-SPARK model in eval mode on the requested device.
        """
        # Choose dtype based on GPU capability: bfloat16 for Ampere+, float16 otherwise
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # Patch sys.path so `import vggt` resolves to VGGT-SPARK, not VGGT-X.
        # Scoped: prepend → import → remove. Any subsequent `import vggt` in
        # the session still resolves to VGGT-X (already cached in sys.modules).
        _patched = _VGGT_SPARK_ROOT not in sys.path
        if _patched:
            sys.path.insert(0, _VGGT_SPARK_ROOT)
        try:
            # Import SPARK VGGT; may already be cached from run_vggt_slam_lc.py
            from vggt.models.vggt import VGGT as VGGT_SPARK  # noqa: PLC0415
        finally:
            # Always remove the path injection — do not pollute session sys.path
            if _patched and _VGGT_SPARK_ROOT in sys.path:
                sys.path.remove(_VGGT_SPARK_ROOT)

        logger.info(
            "VGGTSPARKCreator: loading %s from SPARK source tree (%s)",
            self.model_name,
            _VGGT_SPARK_ROOT,
        )

        # SPARK VGGT does not expose chunk_size — load without it
        model = VGGT_SPARK.from_pretrained(self.model_name)
        model.eval()
        model = model.to(device, dtype=dtype)
        return model

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        """Run VGGT-SPARK forward, mirroring VGGT-SLAM's call exactly.

        SPARK's ``VGGT.forward`` casts inputs to bf16 and wraps its heads in
        ``torch.cuda.amp.autocast(enabled=False)`` — it expects to run with NO
        outer autocast. The inherited VGGTX ``_forward`` wraps the call in
        ``torch.autocast``, which leaks fp32 aggregator tokens into SPARK's
        bf16 ``camera_head.token_norm`` (which, unlike VGGT-X, does not cast its
        input) → ``expected Float but found BFloat16``. This override drops the
        outer autocast and feeds the bf16 tensor directly, identical to
        ``solver.run_predictions`` → guarantees stage-2 forward parity.

        Args:
            model: Loaded VGGT-SPARK model (bf16 weights on Ampere+).
            views: ``(N, 3, H, W)`` preprocessed image tensor from _preprocess.

        Returns:
            dict with keys ``images``, ``extrinsic``, ``intrinsics``,
            ``intrinsics_downsampled`` (alias), ``depth``, ``depth_conf`` —
            same schema as VGGTXCreator._forward so downstream is unchanged.
        """
        device = next(model.parameters()).device
        dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )

        # Feed bf16 tensor directly — SPARK adds the batch dim and disables
        # autocast for its heads internally. No outer torch.autocast wrapper.
        images = views.to(device, dtype=dtype)
        with torch.no_grad():
            predictions = model(images)

        # Decode pose encoding at model resolution (matches VGGT-SLAM upstream).
        extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )

        # Move to CPU float32 for downstream numpy ops.
        extrinsic  = extrinsic_t.cpu().float().numpy().squeeze(0)   # (N, 3, 4)
        intrinsic  = intrinsic_t.cpu().float().numpy().squeeze(0)   # (N, 3, 3)
        depth_map  = predictions["depth"].squeeze(0).cpu().float().numpy()        # (N, H, W, 1)
        depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()   # (N, H, W)

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,
            "intrinsics_downsampled": intrinsic,
            "depth": depth_map,
            "depth_conf": depth_conf,
        }

    def _verify_loop_candidate(
        self,
        frame1: Any,
        frame2: Any,
        verify_match_ratio: float = 0.95,
        **kwargs: Any,
    ) -> tuple[bool, Any]:
        """Verify a loop closure candidate via VGGT-SPARK native similarity.

        Calls ``VGGT.forward(compute_similarity=True)`` on the candidate pair
        and reads ``image_match_ratio`` directly — matching VGGT-SLAM's native
        verification path.  This produces scores in the ~1.02–1.05 range on
        accepted pairs (vs ~0.818 from our cross_frame_attention_ratio hook),
        so the default threshold is recalibrated to 0.95.

        Args:
            frame1, frame2:      Preprocessed frames (C, H, W).
            verify_match_ratio:  Accept threshold (default 0.95 for native scores).
            **kwargs:            Ignored (no hook layer to select).

        Returns:
            (accepted, None) — poses are not extracted on this path; caller uses
            submap poses.
        """
        # Stack frames into (2, C, H, W) batch expected by VGGT.forward
        images = torch.stack([frame1, frame2])
        # Native similarity path — model returns image_match_ratio as a side output
        with torch.no_grad():
            outputs = self.model(images, compute_similarity=True)
        ratio = float(outputs["image_match_ratio"])
        if ratio < verify_match_ratio:
            logger.info(
                "LC verify (native): ratio=%.4f < threshold=%.4f → rejected",
                ratio,
                verify_match_ratio,
            )
            return False, None
        logger.info(
            "LC verify (native): ratio=%.4f >= threshold=%.4f → accepted",
            ratio,
            verify_match_ratio,
        )
        return True, None
