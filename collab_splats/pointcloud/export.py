"""Point-cloud export to binary PLY.

Single place that knows the on-disk PLY byte layout. Written as
``format binary_little_endian 1.0`` with float32 xyz + uchar rgb = 15 bytes per
vertex, ~1.5x smaller than the ASCII file nerfstudio's ``create_ply_from_colmap``
emits and lossless in the coordinates (ASCII truncates to 6 significant digits).
Read back byte-exact by open3d, which is what every nerfstudio dataparser uses.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from collab_splats.pointcloud.utils import subsample_points

logger = logging.getLogger(__name__)

########
# Constants
########

# Packed (no alignment padding) => itemsize 15, matching the header below.
_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)

_HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
)

# Colour for clouds with no RGB (SfM tracks without image colour, synthetic tests).
_DEFAULT_GREY = 128


########
# Export
########


def write_pointcloud_ply(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    path: Path,
    max_points: Optional[int] = None,
) -> Path:
    """Write a (P, 3) cloud + optional (P, 3) uint8 colors to path as a binary PLY.

    Args:
        points: (P, 3) float coordinates.
        colors: (P, 3) uint8 RGB, or None for uniform mid-grey.
        path: destination file; parents are created.
        max_points: optional density cap. None (default) writes every point —
            thinning is opt-in so the exported cloud matches the reconstruction.

    Returns:
        The path written.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (P, 3), got {points.shape}")

    # Validate colours before any thinning so the error names the caller's own arrays
    if colors is not None:
        colors = np.ascontiguousarray(colors)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"colors must be (P, 3), got {colors.shape}")
        if len(colors) != len(points):
            raise ValueError(f"colors length {len(colors)} != points length {len(points)}")
        if colors.dtype != np.uint8:
            raise ValueError(f"colors must be uint8 0-255, got {colors.dtype}")

    # Optional density cap — reuses the pipeline's exact-budget sampler (seeded, reproducible)
    if max_points is not None and len(points) > max_points:
        points, colors = subsample_points(points, colors, max_points=max_points)
        points = np.ascontiguousarray(points, dtype=np.float32)

    # Fill the packed vertex record; colours are already validated uint8 (P, 3)
    verts = np.empty(len(points), dtype=_VERTEX_DTYPE)
    verts["x"] = points[:, 0]
    verts["y"] = points[:, 1]
    verts["z"] = points[:, 2]
    if colors is None:
        verts["red"] = verts["green"] = verts["blue"] = _DEFAULT_GREY
    else:
        verts["red"] = colors[:, 0]
        verts["green"] = colors[:, 1]
        verts["blue"] = colors[:, 2]

    # Header is ASCII, payload is raw little-endian records
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_HEADER.format(n=len(points)).encode("ascii"))
        f.write(verts.tobytes())

    logger.info("wrote %s (%d points)", path, len(points))
    return path
