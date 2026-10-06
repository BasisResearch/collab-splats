"""
Pointcloud reconstruction: feedforward creator lookup and the shared result type.

- get_creator resolves the feedforward backbones by name
- every creator returns a PointcloudResult; see base.py
- needs a CUDA GPU: VGGT-X runs CUDA kernels at import, so refuse before it loads
"""

import torch

# Clear error instead of VGGT-X's import-time CUDA traceback
if not torch.cuda.is_available():
    raise ImportError("collab_splats.pointcloud needs a CUDA GPU; VGGT-X runs CUDA kernels at import")

from collab_splats.pointcloud.base import (  # noqa: E402
    BasePointcloudCreator,
    PointcloudResult,
)
from collab_splats.pointcloud.feedforward import (  # noqa: E402
    BaseFeedforwardCreator,
    MapAnythingCreator,
    VGGTXCreator,
)


def get_creator(name: str) -> type[BaseFeedforwardCreator]:
    """
    Look up a feedforward creator class by registry name.

    Args:
        name: mapanything, vggtx, vggt_omega or loger.

    Returns:
        The creator class.

    Raises:
        ValueError: unknown name; the message lists the available ones.
    """
    return BaseFeedforwardCreator.get(name)


__all__ = [
    "BasePointcloudCreator",
    "BaseFeedforwardCreator",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "get_creator",
]
