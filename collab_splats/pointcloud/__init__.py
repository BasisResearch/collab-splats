"""
Pointcloud reconstruction: feedforward creator lookup and the shared result type.

- get_creator resolves the feedforward backbones by name
- every creator returns a PointcloudResult; see base.py
"""

from collab_splats.pointcloud.base import BasePointcloudCreator, PointcloudResult
from collab_splats.pointcloud.feedforward import (
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
