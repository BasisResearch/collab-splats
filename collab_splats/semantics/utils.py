"""
Semantic feature helpers and the on-disk layout of semantic artifacts.

Layout inside a scene's semantics dir:
- `<extractor>.zarr`: 2D patch cache, `features` (N, D, H_p, W_p) float32, one chunk per frame;
  attrs `extractor`, `patch_size`, `n_frames`.
- `<extractor>_lifted.zarr` + `<extractor>_ae.pt`: per-point codes `features` (P, latent) and the
  autoencoder that decodes them; attrs `input_dim`, `latent_dim`.
- `latent_dim == input_dim` with no `_ae.pt` is full-dim by design (`n_components: null`);
  `latent_dim < input_dim` with no `_ae.pt` is an interrupted write and raises on read.
- The `_lifted` suffix is the only thing separating the two stores in one flat dir.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from PIL import Image

from collab_splats.preproc.frames import IMAGE_EXTS, frame_paths
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.utils.torch_utils import batch_iterator

logger = logging.getLogger(__name__)

__all__ = [
    "ae_path",
    "cache_store_path",
    "compute_semantic_contrast",
    "extract_feature_cache",
    "find_lifted_extractor",
    "lifted_store_path",
    "load_feature_maps",
    "load_point_features",
    "point_features_cached",
    "write_point_features",
]


########################################################
########## Artifact paths ##############################
########################################################

# `_lifted` is load-bearing, not decorative: the flat layout puts the 2D patch cache
# (`<extractor>.zarr`) and the lifted per-point codes in the SAME dir, so the filename is the
# only thing that can tell them apart there.


def lifted_store_path(out_dir: Path, extractor: str) -> Path:
    """
    Path of one extractor's per-point latent store.

    Args:
        out_dir: the scene's semantics dir.
        extractor: extractor name.

    Returns:
        `out_dir/<extractor>_lifted.zarr`.
    """
    return Path(out_dir) / f"{extractor}_lifted.zarr"


def ae_path(out_dir: Path, extractor: str) -> Path:
    """
    Path of the autoencoder that decodes `lifted_store_path`'s codes.

    Args:
        out_dir: the scene's semantics dir.
        extractor: extractor name.

    Returns:
        `out_dir/<extractor>_ae.pt`.
    """
    return Path(out_dir) / f"{extractor}_ae.pt"


def find_lifted_extractor(out_dir: Path) -> Optional[str]:
    """
    Name of the single lifted extractor in a dir, or None when there is none.

    Args:
        out_dir: the scene's semantics dir.

    Returns:
        The extractor name, or None.

    Raises:
        ValueError: when the dir holds more than one lifted store — guessing would pair one
            extractor's codes with another's decoder.
    """
    suffix = "_lifted.zarr"
    stems = sorted(p.name.removesuffix(suffix) for p in Path(out_dir).glob(f"*{suffix}"))
    if not stems:
        return None
    if len(stems) > 1:
        raise ValueError(f"{out_dir} holds several lifted stores {stems} — pass the extractor explicitly")
    return stems[0]


########################################################
########## Re-exports from collab_splats.utils.torch_utils
########################################################

# Canonical location: collab_splats.utils.torch_utils
from collab_splats.utils.torch_utils import (  # noqa: E402,F401
    get_device,
    infer_batch_size,
    load_hf_weights,
    load_torchhub_model,
    pytorch_gc,
)

########################################################
########## Contrastive scoring #########################
########################################################


def compute_semantic_contrast(
    raw_similarities: torch.Tensor,
    num_positive: int,
    temperature: float = 0.05,
    reduction: str = "max",
) -> torch.Tensor:
    """Contrastive scoring: how strongly positive queries match relative to negatives.

    When no negatives are present (num_positive == raw_similarities.shape[0]),
    falls back to raw reduction over positives — contrastive scoring is undefined
    without a negative to push against.

    Args:
        raw_similarities: (N_queries, N) dot-product similarities per patch.
        num_positive: rows [0:num_positive] are positive queries; rest are negative.
        temperature: scaling parameter τ. Lower = sharper. Ignored when no negatives.
        reduction: aggregation over positive queries:
            "max"  — each positive independently scored against all negatives via
                     binary softmax; max over per-positive scores. Use for distinct
                     concepts where any match counts.
            "pool" — positives averaged in similarity space before softmax; one
                     representative competes against all negatives. Use for synonymous
                     concepts that should be treated as one combined query.

    Returns:
        (N,) contrastive scores in [0, 1].
    """
    if reduction not in ("max", "pool"):
        raise ValueError(f"Unknown reduction '{reduction}'. Choose 'max' or 'pool'.")

    pos = raw_similarities[:num_positive]
    neg = raw_similarities[num_positive:]

    if neg.shape[0] == 0:
        return pos.max(dim=0).values if reduction == "max" else pos.mean(dim=0)

    if reduction == "max":
        scores = []
        for p_i in pos:
            stacked = torch.cat([p_i.unsqueeze(0), neg], dim=0)
            scores.append(stacked.div(temperature).softmax(dim=0)[0])
        return torch.stack(scores).max(dim=0).values

    avg_pos = pos.mean(dim=0, keepdim=True)
    stacked = torch.cat([avg_pos, neg], dim=0)
    return stacked.div(temperature).softmax(dim=0)[0]


########################################################################
# Shared token utilities
########################################################################


def _tokens_to_feature_map(tokens: torch.Tensor, input_h: int, input_w: int, patch_size: int) -> torch.Tensor:
    """Reshape (N, D) patch tokens to (D, H_p, W_p), L2-normalized along channel dim."""
    ph = input_h // patch_size
    pw = input_w // patch_size
    assert tokens.shape[0] == ph * pw, (
        f"Expected {ph * pw} tokens for {input_h}x{input_w} " f"(patch_size={patch_size}), got {tokens.shape[0]}"
    )
    feat = tokens.reshape(ph, pw, -1).permute(2, 0, 1)  # (D, H_p, W_p)
    return F.normalize(feat, dim=0)


########################################################
########## Patch alignment #############################
########################################################


def interpolate_to_patch_size(img_bchw: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
    """Interpolate image tensor so H and W are evenly divisible by patch_size.

    Args:
        img_bchw: Image tensor of shape (B, C, H, W).
        patch_size: Patch dimension to align to.

    Returns:
        Tuple of (resized_tensor, target_H, target_W).
    """
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = F.interpolate(img_bchw, size=(target_H, target_W), mode="bilinear", align_corners=False)
    return img_bchw, target_H, target_W


########################################################
########## 2D patch cache (<extractor>.zarr) ###########
########################################################


def cache_store_path(semantics_dir: Path) -> Path:
    """
    Find the 2D patch cache store in semantics_dir — `<extractor>.zarr`.

    Args:
        semantics_dir: the scene's semantics dir.

    Returns:
        Path of the cache store. Its `.stem` is the extractor name.

    Raises:
        FileNotFoundError: when the dir holds no cache store.
    """
    sem_dir = Path(semantics_dir)
    # `*.zarr` also matches the lifted store next door; the suffix is the only thing separating them
    store = next((p for p in sem_dir.glob("*.zarr") if not p.name.endswith("_lifted.zarr")), None)
    if store is None:
        raise FileNotFoundError(f"no 2D feature cache (*.zarr) in {sem_dir} — extract this scene's semantics first")
    return store


def extract_feature_cache(extractor, images_dir: Path, cache_dir: Path) -> Path:
    """
    Extract patch features from a scene's images/ directory into `cache_dir/<extractor>.zarr`.

    Re-entrant: a cache whose extractor name and frame count both match is returned untouched.
    The attrs that make a store look valid are written LAST, after every frame is on disk, so a
    run that dies mid-extraction leaves an attr-less store that the next run re-extracts.

    Args:
        extractor: a BaseFeatureExtractor instance — supplies `.name`, `.patch_size`, `.forward`.
        images_dir: the scene's images/ directory of frame_NNNNNN.<ext> keyframes.
        cache_dir: directory to write the 2D patch cache into.

    Returns:
        Path of the store; `features` is (N, D, H_p, W_p) float32, one chunk per frame.
    """
    zarr_path = Path(cache_dir) / f"{extractor.name}.zarr"

    # One path at a time, never one decoded stack: a 300-frame scene at original resolution must
    # not sit in RAM while the model runs.
    paths = frame_paths(images_dir)
    if not paths:
        raise FileNotFoundError(f"No frame images ({list(IMAGE_EXTS)}) in {images_dir}")
    N = len(paths)

    # A cache is valid when the extractor name and frame count both match
    if zarr_path.exists():
        try:
            z = zarr.open(str(zarr_path), mode="r")
            if z.attrs.get("extractor") == extractor.name and z.attrs.get("n_frames") == N:
                logger.info("Feature cache valid, skipping extraction: %s", zarr_path)
                return zarr_path
        except Exception:
            logger.warning("Cache at %s is corrupt or unreadable, re-extracting", zarr_path)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Probe the first frame to learn (D, H_p, W_p) before allocating the store
    first_frame = Image.open(paths[0]).convert("RGB")
    with torch.no_grad():
        [first_feat] = extractor.forward([first_frame])
    D, H_p, W_p = first_feat.shape

    store = zarr.open(str(zarr_path), mode="w")
    # One chunk per frame: reading frame i loads exactly 1 disk chunk
    arr = store.create_array(
        "features",
        shape=(N, D, H_p, W_p),
        chunks=(1, D, H_p, W_p),
        dtype="float32",
        fill_value=0,
    )
    arr[0] = first_feat.cpu().float().numpy()

    # Iterate the rest lazily — never more than one frame in RAM
    for i in range(1, N):
        pil_img = Image.open(paths[i]).convert("RGB")
        with torch.no_grad():
            [feat] = extractor.forward([pil_img])
        arr[i] = feat.cpu().float().numpy()
        if i % 10 == 0:
            logger.info("extract_feature_cache: %d/%d frames written", i + 1, N)

    # Marker last: until these attrs land the store cannot pass the validity check above, so a
    # crash mid-loop can never leave zero-filled planes behind a store that claims to be complete.
    store.attrs.update({"extractor": extractor.name, "patch_size": extractor.patch_size, "n_frames": N})

    logger.info("Feature cache written: %s  shape=%s", zarr_path, tuple(arr.shape))
    return zarr_path


def load_feature_maps(store_path: Path) -> list[torch.Tensor]:
    """
    Load per-frame dense feature maps from a 2D patch cache store.

    Args:
        store_path: path of an `<extractor>.zarr` store (see `cache_store_path`).

    Returns:
        One (D, H_p, W_p) CPU tensor per frame.
    """
    arr = zarr.open(str(store_path), mode="r")["features"]  # (N, D, H_p, W_p)
    return [torch.from_numpy(np.asarray(arr[i])) for i in range(arr.shape[0])]


########################################################
########## Per-point pair (<extractor>_lifted.zarr) ####
########################################################


def write_point_features(
    out_dir: Path,
    extractor: str,
    codes: np.ndarray,
    ae: Optional[FeatureAutoencoder] = None,
) -> Path:
    """
    Write the per-point pair: `<extractor>_lifted.zarr` (+ `_ae.pt` when compressed).

    Args:
        out_dir: the scene's semantics dir.
        extractor: extractor name — both halves of the pair carry it.
        codes: (P, latent) per-point codes, or (P, D) when `ae` is None.
        ae: the autoencoder that decodes `codes`, or None for full-dim codes.

    Returns:
        Path of the written lifted store.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = np.asarray(codes)
    lifted_zarr = lifted_store_path(out_dir, extractor)

    # input_dim == latent_dim marks full-dim-by-design codes that need no weights; unequal
    # widths mark codes the weights are REQUIRED to decode. Without the marker a missing
    # _ae.pt is ambiguous — uncompressed vs orphaned by a crash — and a reader must guess.
    # Codes, then attrs, then weights; any failure after the store exists removes it again
    # so no half-pair (unreadable codes) is ever left behind on disk.
    try:
        store = zarr.open(str(lifted_zarr), mode="w")
        store["features"] = codes
        store.attrs.update(
            {
                "input_dim": int(ae.input_dim) if ae is not None else int(codes.shape[1]),
                "latent_dim": int(ae.latent_dim) if ae is not None else int(codes.shape[1]),
            }
        )
        if ae is not None:
            ae.save(ae_path(out_dir, extractor))
    except Exception:
        shutil.rmtree(lifted_zarr, ignore_errors=True)
        raise
    return lifted_zarr


def point_features_cached(semantics_dir: Path) -> bool:
    """
    Report whether the lifted store exists and is readable.

    Args:
        semantics_dir: the scene's semantics dir.

    Returns:
        True when the codes are present and either the weights are too or the codes are full-dim.
    """
    sem_dir = Path(semantics_dir)
    extractor = find_lifted_extractor(sem_dir)
    if extractor is None:
        return False
    if ae_path(sem_dir, extractor).exists():
        return True
    # No weights: usable only if the codes describe themselves as full-dim. Anything else is a
    # half-written pair (crash between the two writes) — report NOT cached so the caller re-lifts.
    try:
        attrs = zarr.open(str(lifted_store_path(sem_dir, extractor)), mode="r").attrs
        return int(attrs["latent_dim"]) >= int(attrs["input_dim"])
    except Exception:
        logger.warning("Lifted store for %s in %s is corrupt or unreadable, reporting not cached", extractor, sem_dir)
        return False


def load_point_features(semantics_dir: Path, batch_size: int = 65_536) -> np.ndarray:
    """
    Read the lifted per-point store, decoding latent codes back to full dim.

    Args:
        semantics_dir: the scene's semantics dir.
        batch_size: points per decode chunk — a one-shot decode of a 500k-point scene
            materialises ~2.3 GB of float32 at 768-D, against a 46.6 GB shared container cap.

    Returns:
        (P, D) float32, L2-normalized per row.

    Raises:
        FileNotFoundError: when no lifted store exists, or when latent codes have no weights.
    """
    sem_dir = Path(semantics_dir)
    extractor = find_lifted_extractor(sem_dir)
    if extractor is None:
        raise FileNotFoundError(f"no *_lifted.zarr store in {sem_dir} — lift this scene's features first")
    lifted_zarr = lifted_store_path(sem_dir, extractor)
    weights = ae_path(sem_dir, extractor)
    store = zarr.open(str(lifted_zarr), mode="r")
    codes = np.asarray(store["features"])

    # Weights present -> always decode, even at equal widths: an equal-width autoencoder still
    # encodes, so its codes are not full-dim features. Weights absent is the ambiguous case.
    if not weights.exists():
        try:
            full_dim = int(store.attrs["latent_dim"]) >= int(store.attrs["input_dim"])
        except (KeyError, TypeError, ValueError):
            full_dim = False
        if full_dim:
            return F.normalize(torch.from_numpy(codes), dim=1).cpu().numpy()
        raise FileNotFoundError(
            f"{lifted_zarr} holds {codes.shape[1]}-D per-point codes but the autoencoder that "
            f"decodes them ({weights}) is missing — the pair was written only halfway "
            "(interrupted run). Returning the raw codes would be silent garbage; re-lift this "
            f"scene's semantic features instead (delete {lifted_zarr} and re-run the semantics step)."
        )

    ae = FeatureAutoencoder.load(weights)
    # Streamed decode into a preallocated output: peak stays at (result + one chunk). Row-wise
    # normalize and the decoder's linear layers are both row-independent, so chunking is exact.
    codes_t = torch.from_numpy(codes)
    decoded = torch.empty((codes_t.shape[0], ae.input_dim), dtype=torch.float32)
    with torch.no_grad():
        start = 0
        for (chunk,) in batch_iterator(batch_size, codes_t):
            end = start + len(chunk)
            decoded[start:end] = F.normalize(ae.per_point_decode(chunk), dim=1)
            start = end
    return decoded.numpy()
