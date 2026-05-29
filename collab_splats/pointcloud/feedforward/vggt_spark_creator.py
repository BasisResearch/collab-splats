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
