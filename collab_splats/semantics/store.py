"""
Semantics stores: the 2D patch cache and the lifted per-point store.

- `<extractor>.zarr`: `features` (N, D, H_p, W_p) float16, one chunk per frame
- `<extractor>_lifted.zarr`: `features` (P, latent), plus `autoencoder.pt` when compressed
- paths come from `Reconstructor`
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from collab_splats.preproc.frames import IMAGE_EXTS, frame_paths
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features.base import BaseFeatureExtractor
from collab_splats.utils.io import open_valid, read_image, to_json_safe
from collab_splats.utils.torch_utils import batch_iterator

logger = logging.getLogger(__name__)

__all__ = [
    "extract_feature_cache",
    "read_point_features",
    "valid_feature_cache",
    "write_point_features",
]


########################################################
########## 2D patch cache (<extractor>.zarr) ###########
########################################################


def valid_feature_cache(
    cache_dir: Path, name: str, images_dir: Path, extractor_kwargs: Optional[dict[str, Any]] = None
) -> Optional[Path]:
    """
    Path of a reusable `<name>.zarr` patch cache, or None.

    - valid: extractor name, frame count and kwargs match the store attrs

    Args:
        cache_dir: directory holding the patch caches.
        name: extractor registry name.
        images_dir: scene images/ directory.
        extractor_kwargs: constructor kwargs the cache must match.

    Returns:
        The store path, or None.
    """
    zarr_path = Path(cache_dir) / f"{name}.zarr"
    n_frames = len(frame_paths(images_dir))
    expected = {"extractor": name, "n_frames": n_frames, "extractor_kwargs": extractor_kwargs or {}}

    if open_valid(zarr_path, expected) is None:
        return None

    return zarr_path


def extract_feature_cache(
    extractor: BaseFeatureExtractor,
    images_dir: Path,
    cache_dir: Path,
    extractor_kwargs: Optional[dict[str, Any]] = None,
    overwrite: bool = False,
    batch_size: int = 4,
) -> Path:
    """
    Extract patch features from a scene's images/ into `cache_dir/<extractor>.zarr`.

    - a valid store is returned untouched unless overwrite
    - validity attrs written last, so a crashed run is re-extracted

    Args:
        extractor: feature extractor to run.
        images_dir: scene images/ directory.
        cache_dir: directory to write the store into.
        extractor_kwargs: constructor kwargs; they key the cache.
        overwrite: re-extract even when the store is valid.
        batch_size: frames per `forward` call.

    Returns:
        Path of the store.

    Raises:
        FileNotFoundError: `images_dir` holds no frame images.
    """
    zarr_path = Path(cache_dir) / f"{extractor.name}.zarr"

    # Kwargs as JSON-safe values; zarr serializes them into the store attrs
    extractor_kwargs = to_json_safe(extractor_kwargs or {})

    # Frame paths; decoded one batch at a time below
    paths = frame_paths(images_dir)

    if not paths:
        raise FileNotFoundError(f"No frame images ({list(IMAGE_EXTS)}) in {images_dir}")

    N = len(paths)

    # A cache is valid when the extractor name, frame count and kwargs all match
    if not overwrite and valid_feature_cache(cache_dir, extractor.name, images_dir, extractor_kwargs) is not None:
        logger.info("Feature cache valid, skipping extraction: %s", zarr_path)
        return zarr_path

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Open the store; the array is created once the first batch fixes its shape
    store = zarr.open(str(zarr_path), mode="w")
    arr = None
    i = 0

    for (batch,) in batch_iterator(batch_size, paths):
        # Decode and extract one batch of frames
        frames = [read_image(path) for path in batch]

        with torch.no_grad():
            feats = extractor.forward(frames)

        # First batch fixes (D, H_p, W_p); one chunk per frame, so reading frame i loads 1 chunk
        if arr is None:
            D, H_p, W_p = feats[0].shape
            arr = store.create_array(
                "features",
                shape=(N, D, H_p, W_p),
                chunks=(1, D, H_p, W_p),
                dtype="float16",
                fill_value=0,
            )

        # Write each frame's map into its own chunk
        for feat in feats:
            feat = feat.cpu()
            arr[i] = feat.half().numpy()
            i += 1

        logger.debug("extract_feature_cache: %d/%d frames written", i, N)

    # Validity attrs last: a crash mid-loop leaves a store the check above rejects
    store.attrs.update(
        {
            "extractor": extractor.name,
            "patch_size": extractor.patch_size,
            "n_frames": N,
            "extractor_kwargs": extractor_kwargs,
        }
    )

    logger.info("Feature cache written: %s  shape=%s", zarr_path, tuple(arr.shape))
    return zarr_path


########################################################
########## Lifted store (<extractor>_lifted.zarr) ######
########################################################


def write_point_features(store_path: Path, codes: np.ndarray, ae: Optional[FeatureAutoencoder]) -> None:
    """
    Write the lifted store atomically via a `.tmp` dir renamed into place.

    Args:
        store_path: the lifted store path.
        codes: (P, latent) codes, or (P, D) when `ae` is None.
        ae: autoencoder that decodes `codes`, or None.
    """
    store_path = Path(store_path)
    tmp = store_path.with_name(f"{store_path.name}.tmp")
    codes = np.asarray(codes)
    width = int(codes.shape[1])

    store_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # Codes, weights, then attrs into the tmp dir; a failure removes it
    try:
        store = zarr.open(str(tmp), mode="w")
        store["features"] = codes

        if ae is not None:
            ae.save(tmp / "autoencoder.pt")

        input_dim = int(ae.input_dim) if ae is not None else width
        store.attrs.update({"input_dim": input_dim, "latent_dim": width})
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Swap the finished store in under its real name
    shutil.rmtree(store_path, ignore_errors=True)
    tmp.rename(store_path)


def read_point_features(store_path: Path, batch_size: int = 65_536) -> np.ndarray:
    """
    Read the lifted store, decoding latent codes back to full dim.

    Args:
        store_path: the lifted store path.
        batch_size: points per decode chunk.

    Returns:
        (P, D) float32, L2-normalized per row.

    Raises:
        FileNotFoundError: latent codes have no `autoencoder.pt`.
    """
    store_path = Path(store_path)
    store = zarr.open(str(store_path), mode="r")
    codes = np.asarray(store["features"])
    codes = torch.from_numpy(codes)
    weights = store_path / "autoencoder.pt"

    # No weights: full-dim codes are returned normalized; latent codes cannot be read
    if not weights.exists():
        if store.attrs["latent_dim"] < store.attrs["input_dim"]:
            raise FileNotFoundError(f"{store_path} holds latent codes but no autoencoder.pt; re-run semantics")

        return F.normalize(codes, dim=1).numpy()

    # Decode in chunks into one preallocated array; row-wise ops, so chunking is exact
    ae = FeatureAutoencoder.load(weights)
    decoded = torch.empty((codes.shape[0], ae.input_dim), dtype=torch.float32)

    start = 0

    for chunk in ae.iter_decode(codes, batch_size):
        end = start + len(chunk)
        decoded[start:end] = F.normalize(chunk, dim=1)
        start = end

    return decoded.numpy()
