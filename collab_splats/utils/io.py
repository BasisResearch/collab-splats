"""
Torch-free IO helpers: images, JSON reports, zarr stores.

- images: read_image decodes RGB; to_uint8_hwc turns [0, 1] floats into uint8 HWC
- JSON: to_json_safe makes strict-JSON values; write_json writes them atomically
- zarr: one LZ4 codec, the unreadable-store error set, open_valid for cache checks
- meshes: write_textured_obj writes a UV-textured OBJ + MTL + albedo PNG
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
import zarr
from PIL import Image
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)

########################################################################
# JSON
########################################################################


def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert a payload into values json.dumps writes as strict JSON.

    - dicts, lists and tuples are walked; tuples become lists, as json.dumps writes them
    - numpy arrays become lists, numpy scalars python scalars
    - non-finite floats (nan, inf) become None: json.dumps would write a bare NaN/Infinity
    - any other value passes through untouched

    Args:
        obj: nested dicts, lists, tuples, numpy values and scalars.

    Returns:
        The same structure built from plain python values.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """
    Write obj as strict JSON, atomically.

    - written to <name>.json.tmp, then os.replace onto path
    - a reader that trusts the file by its existence never sees a half-written one

    Args:
        path: destination JSON file; its parent must exist.
        obj: payload, converted with to_json_safe.
        indent: json.dumps indent.

    Returns:
        The written path.
    """
    path = Path(path)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(to_json_safe(obj), indent=indent))
    os.replace(tmp_path, path)
    return path


########################################################################
# Images
########################################################################


def read_image(path: str | Path) -> np.ndarray:
    """
    Decode one image file as RGB.

    - cv2 decode then BGR -> RGB, the same pixels read_frames has always returned
    - cv2.imread answers None for a missing or undecodable file; raised here by name
    - EXIF orientation ignored: matches PIL/feedforward loaders

    Args:
        path: image file.

    Returns:
        (H, W, 3) uint8 RGB.

    Raises:
        FileNotFoundError: the file is missing or cv2 cannot decode it.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_uint8_hwc(images: np.ndarray, *, channels_first: bool) -> np.ndarray:
    """
    Float [0, 1] images as contiguous uint8, channels last.

    - np.rint (half to even), then clip: truncation biases every channel down half a level
    - a [0, 255] input raises: the images convention is [0, 1], never a per-call guess
    - integer input raises: a uint8 mask or dark image would pass the range check and saturate
    - non-finite input raises: a NaN max slips past the range check

    Args:
        images: float images in [0, 1], (..., 3, H, W) or (..., H, W, 3).
        channels_first: True when the channel axis is third from last.

    Returns:
        (..., H, W, 3) uint8, C-contiguous.

    Raises:
        ValueError: non-float dtype, a non-finite value, a max above 1.5 (a [0, 255] array),
            or a channel axis that is not 3 wide.
    """
    arr = np.asarray(images)

    # Only finite floats carry the [0, 1] convention
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"images must be float in [0, 1], got dtype {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError("images must be finite, got NaN or inf")

    # Slack above 1.0 for resize overshoot; anything past it is a [0, 255] array
    if arr.size and float(arr.max()) > 1.5:
        raise ValueError(f"images must be in [0, 1], got max {float(arr.max()):.1f} (a [0, 255] array?)")

    # Channel axis last; a misdeclared layout shows up as a non-3 last axis
    if channels_first:
        arr = np.moveaxis(arr, -3, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"images must have 3 channels, got shape {arr.shape} (channels_first={channels_first})")

    # Round, clip, cast
    return np.ascontiguousarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8))


########################################################################
# Zarr
########################################################################

# One Blosc LZ4 codec for every array a pipeline store writes
LZ4 = BloscCodec(cname="lz4")

# Errors a missing, corrupt or half-written zarr store raises; anything else is a bug
UNREADABLE_STORE: tuple[type[Exception], ...] = (OSError, ValueError, KeyError, TypeError)


def open_valid(path: str | Path, expected: dict[str, Any]) -> zarr.Group | zarr.Array | None:
    """
    Open a zarr store read-only when its validity attrs match, else None.

    - writers stamp validity attrs last, so a crash mid-write leaves a store this rejects
    - missing path: None; mismatched attrs: None, logged at info
    - unreadable store (UNREADABLE_STORE): None with a warning; any other error propagates

    Args:
        path: zarr store directory.
        expected: attrs the store must carry, each compared with == after to_json_safe.

    Returns:
        The opened store, or None when it must be rebuilt.
    """
    path = Path(path)
    if not path.exists():
        return None

    # Open and read attrs; only store-shaped errors mean "rebuild"
    try:
        store = zarr.open(str(path), mode="r")
        attrs = dict(store.attrs)
    except UNREADABLE_STORE:
        logger.warning("Store at %s is corrupt or unreadable, treating it as absent", path)
        return None

    # Every expected attr must match exactly; attrs round-trip JSON, so compare JSON-safe values
    stale = {k: attrs.get(k) for k, v in to_json_safe(expected).items() if attrs.get(k) != v}
    if stale:
        logger.info("Store at %s is stale (%s), treating it as absent", path, stale)
        return None
    return store


########################################################################
# Meshes
########################################################################


def write_textured_obj(
    out_dir: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    uvs: np.ndarray,
    albedo: np.ndarray,
) -> Path:
    """
    Textured mesh as out_dir/mesh.obj + mesh.mtl + albedo.png.

    - one vertex per face corner: a seam vertex carries a different UV in each face
    - normals are written per corner (vn), so shading stays smooth across UV seams
    - diffuse is white (Kd 1 1 1): trimesh's 0.4 default darkens the texture in every viewer

    Args:
        out_dir: directory to create.
        vertices: (V, 3) positions.
        faces: (F, 3) vertex indices.
        vertex_normals: (V, 3) per-vertex normals.
        uvs: (F, 3, 2) per-corner texture coordinates in [0, 1].
        albedo: (S, S, 3) uint8 texture.

    Returns:
        Path to out_dir/mesh.obj.
    """
    # Split every face corner into its own vertex so each carries its own UV
    faces = np.asarray(faces)
    corners = np.asarray(vertices)[faces].reshape(-1, 3)
    corner_normals = np.asarray(vertex_normals)[faces].reshape(-1, 3)
    material = trimesh.visual.material.SimpleMaterial(
        image=Image.fromarray(albedo), name="albedo", diffuse=[255, 255, 255, 255]
    )
    textured = trimesh.Trimesh(
        corners,
        np.arange(len(corners)).reshape(-1, 3),
        vertex_normals=corner_normals,
        visual=trimesh.visual.TextureVisuals(uv=np.asarray(uvs).reshape(-1, 2), material=material),
        process=False,
    )

    # trimesh names the texture after the material: albedo.png beside mesh.obj + mesh.mtl
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mesh.obj"
    textured.export(out, mtl_name="mesh.mtl")
    return out
