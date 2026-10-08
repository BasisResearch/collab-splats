"""
Semantics stores: the 2D patch cache and the lifted per-point store.

- `<extractor>_codes.zarr` (`_features.zarr` while extracting): `features` (N, D, H_p, W_p) float16, one chunk per frame
- `<extractor>_lifted.zarr`: `features` (P, latent), plus `autoencoder.pt` when compressed, plus mesh-vertex
  arrays (`vertex_word_ids` / `vertex_word_probs` or `vertex_features`) when the mesh existed
- paths come from `Reconstructor`
"""

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from collab_splats.preproc.frames import frame_paths
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.utils.io import open_valid, to_json_safe
from collab_splats.utils.torch_utils import get_device

logger = logging.getLogger(__name__)

__all__ = [
    "read_point_features",
    "valid_feature_cache",
    "write_feature_cache",
    "write_point_features",
]


########################################################
########## 2D feature cache (<extractor>_codes.zarr) ###
########################################################


def valid_feature_cache(
    store_path: Path,
    name: str,
    images_dir: Path,
    extractor_kwargs: Optional[dict[str, Any]],
    latent_dim: Optional[int],
) -> Optional[Path]:
    """
    The 2D feature cache at store_path when reusable, else None.

    - valid: extractor name, frame count, kwargs and code width match the store attrs
    - attrs are written last, so a crashed write reads invalid

    Args:
        store_path: the `<extractor>_codes.zarr` path.
        name: extractor registry name.
        images_dir: scene images/ directory.
        extractor_kwargs: constructor kwargs the cache must match.
        latent_dim: AE code width the cache must hold; None for uncompressed features.

    Returns:
        store_path, or None.
    """
    n_frames = len(frame_paths(images_dir))
    expected = {
        "extractor": name,
        "n_frames": n_frames,
        "extractor_kwargs": extractor_kwargs or {},
        "latent_dim": latent_dim,
    }

    if open_valid(store_path, expected) is None:
        return None

    return Path(store_path)


def write_feature_cache(
    store_path: Path,
    maps: Iterable[torch.Tensor],
    n_frames: int,
    attrs: dict[str, Any],
    ae: Optional[FeatureAutoencoder] = None,
) -> None:
    """
    Write per-frame (D, H_p, W_p) maps to a zarr store as fp16, one chunk per frame.

    - serves extraction (full-width features) and encoding (AE codes)
    - ae saved as `autoencoder.pt` before the attrs; attrs last, so a crash reads invalid
    - overwrites whatever is at store_path

    Args:
        store_path: the store to write.
        maps: frame maps in store-row order, any float dtype and device.
        n_frames: number of maps expected.
        attrs: store attrs, written last.
        ae: autoencoder that decodes the maps, saved inside the store; None for none.

    Raises:
        ValueError: when n_frames < 1 or maps yields other than n_frames maps.
    """
    if n_frames < 1:
        raise ValueError(f"write_feature_cache: no frames to write for {store_path}")

    store = zarr.open(str(store_path), mode="w")
    arr = None
    n_written = 0

    for fmap in maps:
        # Too many maps: stop before zarr's out-of-bounds write
        if n_written == n_frames:
            raise ValueError(
                f"write_feature_cache: more than {n_frames} frame maps for {store_path}"
            )

        # First map fixes (D, H_p, W_p); one chunk per frame, so reading frame i loads 1 chunk
        if arr is None:
            arr = store.create_array(
                "features",
                shape=(n_frames, *fmap.shape),
                chunks=(1, *fmap.shape),
                dtype="float16",
                fill_value=0,
            )

        fmap = fmap.detach().cpu()
        arr[n_written] = fmap.half().numpy()
        n_written += 1

    # Too few maps: leave the store without attrs, so it reads invalid
    if n_written != n_frames:
        raise ValueError(
            f"write_feature_cache: got {n_written} of {n_frames} frame maps for {store_path}"
        )

    assert arr is not None

    # Decoder weights beside the codes, before the attrs mark the store valid
    if ae is not None:
        ae.save(Path(store_path) / "autoencoder.pt")

    # Validity attrs last
    store.attrs.update(to_json_safe(attrs))
    logger.info("feature cache written: %s  shape=%s", store_path, tuple(arr.shape))


########################################################
########## Lifted store (<extractor>_lifted.zarr) ######
########################################################


def write_point_features(
    store_path: Path,
    codes: np.ndarray,
    ae: Optional[FeatureAutoencoder],
    vertex_arrays: Optional[dict[str, np.ndarray]] = None,
    attrs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Write the lifted store atomically via a `.tmp` dir renamed into place.

    - codes stored fp16; read_point_features returns them float32
    - vertex arrays stored in the dtype they arrive in (caller picks fp16 / int16)

    Args:
        store_path: the lifted store path.
        codes: (P, latent) codes, or (P, D) when `ae` is None.
        ae: autoencoder that decodes `codes`, or None.
        vertex_arrays: mesh-vertex arrays by name, e.g. `vertex_features`; None for none.
        attrs: extra store attrs, e.g. `extractor`, `mesh_sha256`; None for none.
    """
    store_path = Path(store_path)
    tmp = store_path.with_name(f"{store_path.name}.tmp")
    codes = np.asarray(codes)
    width = int(codes.shape[1])

    store_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # Codes, vertex arrays, weights, then attrs into the tmp dir; a failure removes it
    try:
        store = zarr.open(str(tmp), mode="w")
        store["features"] = codes.astype(np.float16)

        for key, array in (vertex_arrays or {}).items():
            store[key] = array

        if ae is not None:
            ae.save(tmp / "autoencoder.pt")

        input_dim = int(ae.input_dim) if ae is not None else width
        store.attrs.update(
            {"input_dim": input_dim, "latent_dim": width, **to_json_safe(attrs or {})}
        )
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Swap the finished store in under its real name
    shutil.rmtree(store_path, ignore_errors=True)
    tmp.rename(store_path)


def read_point_features(
    store_path: Path, batch_size: int = 65_536, name: str = "features"
) -> np.ndarray:
    """
    Read the lifted store, decoding latent codes back to full dim.

    - decodes on get_device(); the result is copied back to host

    Args:
        store_path: the lifted store path.
        batch_size: points per decode chunk.
        name: array to read, `features` (points) or `vertex_features`.

    Returns:
        (P, D) float32, L2-normalized per row.

    Raises:
        FileNotFoundError: latent codes have no `autoencoder.pt`.
    """
    store_path = Path(store_path)
    store = zarr.open(str(store_path), mode="r")
    codes = np.asarray(store[name], dtype=np.float32)
    codes = torch.from_numpy(codes)
    weights = store_path / "autoencoder.pt"

    # No weights: full-dim codes are returned normalized; latent codes cannot be read
    if not weights.exists():
        if store.attrs["latent_dim"] < store.attrs["input_dim"]:
            raise FileNotFoundError(
                f"{store_path} holds latent codes but no autoencoder.pt; re-run semantics"
            )

        return F.normalize(codes, dim=1).numpy()

    # Decode in chunks into one preallocated array; row-wise ops, so chunking is exact
    ae = FeatureAutoencoder.load(weights).to(get_device())
    decoded = torch.empty((codes.shape[0], ae.input_dim), dtype=torch.float32)

    start = 0

    for chunk in ae.iter_decode(codes, batch_size):
        end = start + len(chunk)
        decoded[start:end] = F.normalize(chunk, dim=1).cpu()
        start = end

    return decoded.numpy()
