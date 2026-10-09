"""
Stage 3 pose estimation: cached-feature matches, world-point lookup, pycolmap absolute pose.

- feature DB: pointcloud.zarr local_features/<matcher>/{reconstruction,localized}
- refs: DINO-SALAD top-k reconstruction frames, or frames the caller chooses
- ref px map through the preprocess crop onto the world_points grid before lookup
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import pycolmap
import torch
import zarr
from tqdm.auto import tqdm

from collab_splats.geometry.projection import sample_world_points
from collab_splats.geometry.transforms import rescale_intrinsics
from collab_splats.localization.extractors import LocalFeatures, LocalMatcher
from collab_splats.localization.retrieval import BaseRetrievalExtractor
from collab_splats.utils.io import LZ4
from collab_splats.utils.torch_utils import to_numpy

if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult

logger = logging.getLogger(__name__)


########################################
# Helpers
########################################


def seed_intrinsics(height: int, width: int) -> np.ndarray:
    """
    Pinhole K seed from image proportions, COLMAP's f = 1.2 * max(W, H) rule.

    - centered principal point, square pixels; pycolmap SIMPLE_PINHOLE refinement solves the true focal

    Args:
        height: image height, px.
        width: image width, px.

    Returns:
        (3, 3) float32 K.
    """
    f = 1.2 * max(width, height)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _crop_to_model_grid(
    px: np.ndarray, box: np.ndarray, model_hw: tuple[int, int]
) -> np.ndarray:
    """
    Full-res pixels on the model grid: inverse of PointcloudResult.__post_init__'s crop map.

    - box is one original_coords row [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], pixel-corner
    - px and the result are pixel-center: (px - tl + 0.5) * scale - 0.5
    - pixels outside the grid come back outside it; sample_world_points marks them invalid
    """
    H, W = model_hw
    crop_wh = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float32)
    scale = np.array([W, H], dtype=np.float32) / crop_wh
    shifted = px - box[:2] + 0.5
    return (shifted * scale - 0.5).astype(np.float32)


def _to_query_grid(
    result: LocalizationResult, small_hw: tuple[int, int], full_hw: tuple[int, int]
) -> LocalizationResult:
    """
    Result with K and query px mapped from the shrunk query grid back to the original one.

    - per-axis full / small scale, pixel-corner convention as rescale_intrinsics; pose unchanged
    """
    if small_hw == full_hw:
        return result

    K = rescale_intrinsics(result.query_intrinsics, small_hw, full_hw)
    K = K.astype(np.float32)
    pts2d = result.pts2d

    if pts2d is not None:
        scale = np.array(
            [full_hw[1] / small_hw[1], full_hw[0] / small_hw[0]], dtype=np.float32
        )
        pts2d = pts2d * scale

    return dataclasses.replace(result, query_intrinsics=K, pts2d=pts2d)


def _chunked(items: Iterable[np.ndarray], size: int) -> Iterator[list[np.ndarray]]:
    """
    Consecutive lists of up to size items, drawn lazily.
    """
    it = iter(items)

    while True:
        window = itertools.islice(it, size)
        chunk = list(window)

        if not chunk:
            return

        yield chunk


def _full_frame_coords(n: int, height: int, width: int) -> np.ndarray:
    """
    original_coords rows for uncropped frames whose model grid is the image itself.
    """
    row = np.array([0, 0, width, height, width, height], dtype=np.float32)
    return np.tile(row, (n, 1))


def localization_db_exists(zarr_path: Path | str, extractor_name: str) -> bool:
    """
    True when pointcloud.zarr holds a complete reconstruction feature DB for the extractor.

    - a missing store, or one that is not a group, reads as absent
    - a group without the image_paths commit marker is a crashed build and reads as absent

    Args:
        zarr_path: pointcloud.zarr path.
        extractor_name: feature-cache key.

    Returns:
        Whether local_features/<extractor_name>/reconstruction exists and is committed.
    """
    # Open read-only; a missing or non-group store has no DB
    try:
        store = zarr.open_group(str(zarr_path), mode="r")
    except (
        FileNotFoundError,
        zarr.errors.NodeNotFoundError,
        zarr.errors.ContainsArrayError,
    ):
        return False

    # save_index writes image_paths last; without it the build never finished
    key = f"local_features/{extractor_name}/reconstruction"

    if key not in store:
        return False

    return "image_paths" in store[key].attrs


def _append_rows(group: zarr.Group, name: str, rows: np.ndarray) -> None:
    """
    Append rows to an existing array in one resize and one write.
    """
    arr = group[name]
    n = arr.shape[0]
    arr.resize((n + len(rows), *arr.shape[1:]))
    arr[n:] = rows


def _features_from_csr(
    group: zarr.Group, image_sizes: list[tuple[int, int]]
) -> list[LocalFeatures]:
    """
    Per-frame LocalFeatures from the reconstruction CSR feature group.
    """
    offsets = group["frame_offsets"][:]
    kpts = group["keypoints"][:]
    descs = group["descriptors"][:]
    norm = group["keypoints_normalized"][:] if "keypoints_normalized" in group else None
    feats = []

    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        norm_i = None if norm is None else torch.from_numpy(norm[s:e])
        feats.append(
            LocalFeatures(
                keypoints=torch.from_numpy(kpts[s:e]),
                descriptors=torch.from_numpy(descs[s:e]),
                keypoints_normalized=norm_i,
                image_size=image_sizes[i],
            )
        )

    return feats


def _write_csr(
    group: zarr.Group, feats: list[LocalFeatures], chunk_rows: int = 65536
) -> None:
    """
    Create the CSR arrays of a feature group: offsets, keypoints, descriptors, normalized keypoints.

    - keypoints_normalized: written only when every frame has it; zeros would be wrong data
    - per-keypoint arrays chunk by chunk_rows rows, so a cache-hit load decodes chunks in parallel
    """
    # Row offsets per frame
    counts = [len(f.keypoints) for f in feats]
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    # Flat keypoint and descriptor tables
    d = feats[0].descriptors.shape[1] if feats else 1
    kpts = [to_numpy(f.keypoints) for f in feats]
    descs = [to_numpy(f.descriptors) for f in feats]
    all_kpts = (
        np.concatenate(kpts).astype(np.float32)
        if offsets[-1]
        else np.zeros((0, 2), np.float32)
    )
    all_descs = (
        np.concatenate(descs).astype(np.float32)
        if offsets[-1]
        else np.zeros((0, d), np.float32)
    )

    # Row chunks no taller than the table, so a small DB stays one chunk
    n_rows = int(offsets[-1])
    rows = min(chunk_rows, max(n_rows, 1))

    group.create_array(
        "frame_offsets", data=offsets, chunks=offsets.shape, compressors=LZ4
    )
    group.create_array("keypoints", data=all_kpts, chunks=(rows, 2), compressors=LZ4)
    group.create_array(
        "descriptors", data=all_descs, chunks=(rows, max(d, 1)), compressors=LZ4
    )

    # Normalized keypoints only when every frame carries them
    if feats and offsets[-1] and all(f.keypoints_normalized is not None for f in feats):
        parts = [to_numpy(f.keypoints_normalized) for f in feats]
        data = np.concatenate(parts).astype(np.float32)
        group.create_array(
            "keypoints_normalized", data=data, chunks=(rows, 2), compressors=LZ4
        )


def _append_csr(group: zarr.Group, feats: list[LocalFeatures]) -> None:
    """
    Append frames to an existing CSR feature group, one resize and write per array.
    """
    counts = [len(f.keypoints) for f in feats]
    last = int(group["frame_offsets"][-1])
    offsets = last + np.cumsum(counts, dtype=np.int64)
    _append_rows(group, "frame_offsets", offsets)
    kpts = [to_numpy(f.keypoints) for f in feats]
    descs = [to_numpy(f.descriptors) for f in feats]
    kpts = np.concatenate(kpts, dtype=np.float32)
    descs = np.concatenate(descs, dtype=np.float32)
    _append_rows(group, "keypoints", kpts)
    _append_rows(group, "descriptors", descs)

    # Normalized keypoints must exist for every appended frame, or the array would misalign
    if "keypoints_normalized" in group:
        if any(f.keypoints_normalized is None for f in feats):
            raise ValueError(
                "feature DB stores keypoints_normalized; every appended frame must carry it"
            )

        parts = [to_numpy(f.keypoints_normalized) for f in feats]
        data = np.concatenate(parts, dtype=np.float32)
        _append_rows(group, "keypoints_normalized", data)


########################################
# Result
########################################


@dataclass
class LocalizationResult:
    """
    Output of CameraLocalizer.localize: pose plus the correspondences PnP saw.

    - pose None when PnP fails or fewer than 4 correspondences exist
    - pts2d_ref lives in reference-image pixels of size ref_hw; rescale before drawing elsewhere
    - ref_frame_indices: source reference frame per correspondence
    """

    pose: np.ndarray | None  # (4, 4) world-to-camera
    n_correspondences: int  # M, 2D-3D pairs before RANSAC
    n_inliers: int
    pts2d: np.ndarray | None  # (M, 2) query px
    pts3d_matched: np.ndarray | None  # (M, 3) world
    inlier_mask: np.ndarray | None  # (M,) bool
    pts2d_ref: np.ndarray | None = None  # (M, 2) reference px
    ref_frame_indices: np.ndarray | None = None  # (M,) int32
    query_intrinsics: np.ndarray | None = (
        None  # (3, 3) refined K when a pose is found, else the input K
    )
    ref_hw: tuple[int, int] | None = None  # (H, W) of the reference images

    @property
    def ranked_ref_frames(self) -> list[int]:
        """
        Reference frames ordered by inlier count, most first; zero-inlier frames dropped.

        - ties broken by lowest frame index; empty when pose failed

        Returns:
            Reference-frame indices.
        """
        if self.inlier_mask is None or self.ref_frame_indices is None:
            return []

        # Inliers per reference frame, descending, stable tie-break
        frames = self.ref_frame_indices[self.inlier_mask]
        frames = frames.astype(np.intp)
        counts = np.bincount(frames)
        order = np.argsort(-counts, kind="stable")
        return [int(i) for i in order if counts[i] > 0]


def read_localization_db(
    zarr_path: Path | str, extractor_name: str
) -> tuple[list[LocalFeatures], list[str], tuple[int, int]]:
    """
    Read the reconstruction feature DB of one extractor.

    - the read half of CameraLocalizer.save_index; load_index builds a localizer on it
    - image_paths is written last, so a DB cut short mid-write reads as missing

    Args:
        zarr_path: pointcloud.zarr path.
        extractor_name: feature-cache key.

    Returns:
        Per-frame LocalFeatures, frame ids, and the reference images' (H, W).

    Raises:
        KeyError: no DB for the extractor, or one that is incomplete (rebuild it).
    """
    store = zarr.open(str(zarr_path), mode="r")
    rec_key = f"local_features/{extractor_name}/reconstruction"

    if rec_key not in store:
        raise KeyError(
            f"No feature DB for '{extractor_name}' in {zarr_path}; build via CameraLocalizer.from_pointcloud()"
        )

    # Completeness: the image_paths commit marker, global_desc, and agreeing frame counts
    group = store[rec_key]

    if "image_paths" not in group.attrs or "global_desc" not in group:
        raise KeyError(
            f"feature DB for '{extractor_name}' is incomplete (no image_paths or global_desc); rebuild it"
        )

    ids = [str(p) for p in group.attrs["image_paths"]]
    n_frames = group["frame_offsets"].shape[0] - 1
    n_desc = group["global_desc"].shape[0]

    if not n_frames == len(ids) == n_desc:
        raise KeyError(
            f"feature DB for '{extractor_name}' is inconsistent "
            f"({n_frames} frames, {len(ids)} ids, {n_desc} global_desc); rebuild it"
        )

    # Bulk decode is the slow phase of a cache-hit load; time it
    hw = tuple(int(x) for x in group.attrs["hw"])
    t0 = time.perf_counter()
    feats = _features_from_csr(group, [(hw[1], hw[0])] * len(ids))
    logger.info(
        "CameraLocalizer: read feature DB (%d frames) in %.1fs",
        len(feats),
        time.perf_counter() - t0,
    )
    return feats, ids, cast(tuple[int, int], hw)


########################################
# Localizer
########################################


class CameraLocalizer:
    """
    Locates a query camera in a known scene from cached reference-frame features.

    - reference features stay on the CPU; each chosen ref moves to the matcher's device per localize
    - localized frames are id and pose only, never match sources
    """

    def __init__(
        self,
        world_points: np.ndarray,
        extrinsics: np.ndarray,
        images: Iterable[np.ndarray],
        ids: list[str],
        *,
        extractor: LocalMatcher | None = None,
        config: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        original_coords: np.ndarray | None = None,
        top_k: int = 8,
        batch_size: int = 32,
        retrieval: str = "dino-salad",
    ) -> None:
        """
        Extract and embed every reference frame, one chunk at a time.

        Args:
            world_points: (N, H, W, 3) model-grid world points, one map per reference frame.
            extrinsics: (N, 4, 4) world-to-camera.
            images: (h, w, 3) uint8 RGB reference frames, drawn lazily; no image IO here.
            ids: stable per-frame labels, index-aligned with images.
            extractor: local matcher; None builds LocalMatcher("loma").
            config: pycolmap options: "estimation" {"ransac": {"max_error"}} (50),
                "refinement" {"refine_focal_length", "refine_extra_params"} (True, False).
            progress_callback: called (frame_index, total) as frames are indexed.
            original_coords: (N, 6) preprocess crop per frame; None means images are the uncropped grid source.
            top_k: retrieved reference frames per localize(refs=None).
            batch_size: frames per extract and embed call.
            retrieval: retrieval registry name, e.g. "dino-salad" or "megaloc".

        Raises:
            ValueError: images is empty, or a frame's size disagrees with its original_coords row.
        """
        self._setup(
            world_points,
            extrinsics,
            ids,
            extractor=extractor,
            config=config,
            original_coords=original_coords,
            top_k=top_k,
            retrieval=retrieval,
            batch_size=batch_size,
        )
        descs = []

        # Extract and embed per chunk; tqdm only without an external progress sink
        bar = tqdm(
            total=len(ids),
            desc="Indexing frames",
            unit="frame",
            leave=False,
            disable=progress_callback is not None,
        )

        for chunk in _chunked(images, batch_size):
            start = len(self._frame_features)

            # Ref pixels map through original_coords; each frame must be its row's full-res (orig_w, orig_h)
            if original_coords is not None:
                for k, image in enumerate(chunk, start):
                    frame_wh = (image.shape[1], image.shape[0])
                    orig_wh = original_coords[k, 4:6].tolist()

                    if frame_wh != tuple(orig_wh):
                        raise ValueError(
                            f"CameraLocalizer: reference frame {k} is {frame_wh[0]}x{frame_wh[1]} "
                            f"but original_coords expects {orig_wh}; pass full-res frames"
                        )

            feats = self._extractor.extract(chunk)
            assert isinstance(feats, list)
            self._frame_features += feats
            descs.append(self._embed(chunk))
            bar.update(len(chunk))

            if progress_callback is not None:
                for k in range(start, len(self._frame_features)):
                    progress_callback(k, len(ids))

        bar.close()

        if not self._frame_features:
            raise ValueError("CameraLocalizer: no reference frames")

        # Reference image size, and the crop each frame went through
        image_size = self._frame_features[0].image_size
        assert image_size is not None
        w, h = image_size
        self._set_frames(self._frame_features, np.concatenate(descs), (h, w))
        logger.info("CameraLocalizer: indexed %d frames", len(self._frame_features))

    def _setup(
        self,
        world_points: np.ndarray,
        extrinsics: np.ndarray,
        ids: list[str],
        extractor: LocalMatcher | None,
        config: dict | None,
        original_coords: np.ndarray | None,
        top_k: int,
        retrieval: str,
        batch_size: int = 32,
    ) -> None:
        """
        State shared by __init__ and load_index, before any reference frame is known.
        """
        self.config = config or {}
        self._world_points = world_points
        self._extrinsics = extrinsics
        self._extractor = extractor if extractor is not None else LocalMatcher("loma")
        self._retrieval_name = retrieval
        self._retrieval: BaseRetrievalExtractor | None = None
        self._top_k = top_k
        self._batch_size = batch_size
        self._original_coords = original_coords
        self._ids = [str(i) for i in ids]
        self._localized_ids: list[str] = []
        self._localized_extrinsics: list[np.ndarray] = []
        self._frame_features: list[LocalFeatures] = []

    def _set_frames(
        self,
        features: list[LocalFeatures],
        global_desc: np.ndarray,
        hw: tuple[int, int],
    ) -> None:
        """
        Reference features, descriptors and image size; uncropped coords when none were given.
        """
        self._frame_features = features
        self._global_desc = global_desc
        self._image_hw = hw
        coords = self._original_coords

        if coords is None:
            coords = _full_frame_coords(len(features), *hw)

        self._coords = coords

    ########################################
    # Views
    ########################################

    @property
    def frame_sources(self) -> list[str]:
        """
        Provenance per frame, index-aligned with image_paths.

        Returns:
            'reconstruction' or 'localized' per frame.
        """
        return ["reconstruction"] * len(self._ids) + ["localized"] * len(
            self._localized_ids
        )

    @property
    def image_paths(self) -> list[str]:
        """
        Frame ids, index-aligned with frame_sources and extrinsics.

        Returns:
            Frame ids: reconstruction then localized.
        """
        return self._ids + self._localized_ids

    @property
    def extrinsics(self) -> np.ndarray:
        """
        World-to-camera pose of every frame: reconstruction then localized.

        Returns:
            (N, 4, 4) poses.
        """
        if not self._localized_extrinsics:
            return self._extrinsics

        localized = np.stack(self._localized_extrinsics)
        return np.concatenate([self._extrinsics, localized], axis=0)

    ########################################
    # Retrieval
    ########################################

    def _embed(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Retrieval descriptors of uint8 RGB frames; the registry model is built on first use.
        """
        if self._retrieval is None:
            retrieval_cls = BaseRetrievalExtractor.get(self._retrieval_name)
            self._retrieval = retrieval_cls()

        # Upload uint8, convert on device
        device = next(self._retrieval.parameters()).device
        stack = np.stack(images)
        tensor = torch.from_numpy(stack).to(device)
        tensor = tensor.permute(0, 3, 1, 2).float() / 255
        desc = self._retrieval(tensor)
        return to_numpy(desc).astype(np.float32)

    def _rank_refs(self, query_image: np.ndarray) -> list[int]:
        """
        Top-k reconstruction frames by cosine similarity to the query.
        """
        query_desc = self._embed([query_image])[0]
        sims = self._global_desc @ query_desc
        order = np.argsort(-sims, kind="stable")
        return [int(i) for i in order[: self._top_k]]

    ########################################
    # Persistence
    ########################################

    def save_index(
        self, zarr_path: Path | str, extractor_name: str, attrs: dict | None = None
    ) -> None:
        """
        Persist the reconstruction features and global descriptors to pointcloud.zarr.

        - replaces the extractor's whole group, localized frames included
        - image_paths is written last: it marks the DB complete

        Args:
            zarr_path: pointcloud.zarr path.
            extractor_name: feature-cache key.
            attrs: build provenance (backbone, ba, lc, built_at, ...).
        """
        store = zarr.open(str(zarr_path), mode="a")
        ext_key = f"local_features/{extractor_name}"

        # Clean overwrite; localized poses belong to the reconstruction being replaced
        if ext_key in store:
            del store[ext_key]

        ext_group = store.require_group(ext_key)
        provenance = dict(attrs) if attrs is not None else {}
        provenance["extractor"] = extractor_name
        ext_group.attrs.put(provenance)

        # Arrays first, then the cache stamps, then the image_paths commit marker
        group = store.require_group(f"{ext_key}/reconstruction")
        _write_csr(group, self._frame_features)
        group.create_array(
            "global_desc",
            data=self._global_desc,
            chunks=self._global_desc.shape,
            compressors=LZ4,
        )
        group.attrs["hw"] = list(self._image_hw)
        group.attrs["max_num_keypoints"] = self._extractor.max_num_keypoints
        group.attrs["retrieval"] = self._retrieval_name
        group.attrs["image_paths"] = list(self._ids)
        logger.info(
            "CameraLocalizer.save_index: %d frames to %s [%s]",
            len(self._ids),
            zarr_path,
            extractor_name,
        )

    @classmethod
    def load_index(
        cls,
        zarr_path: Path | str,
        extractor_name: str,
        world_points: np.ndarray,
        extrinsics: np.ndarray,
        config: dict | None = None,
        extractor: LocalMatcher | None = None,
        *,
        original_coords: np.ndarray | None = None,
        top_k: int = 8,
    ) -> CameraLocalizer:
        """
        Localizer from the zarr feature DB, attached to the current scene geometry.

        - localized frames load as ids and poses only; the retrieval model loads on first localize
        - the retrieval model is the one stored with the DB, so queries embed like the references

        Args:
            zarr_path: pointcloud.zarr path.
            extractor_name: feature-cache key.
            world_points: (N, H, W, 3) model-grid world points.
            extrinsics: (N, 4, 4) world-to-camera.
            config: pycolmap options, as __init__.
            extractor: local matcher; None builds LocalMatcher("loma").
            original_coords: (N, 6) preprocess crop per frame; None means uncropped.
            top_k: retrieved reference frames per localize(refs=None).

        Returns:
            The localizer.

        Raises:
            KeyError: no DB for the extractor, or an incomplete one (rebuild it).
            ValueError: the geometry does not cover exactly the DB's frames.
        """
        rec_features, rec_ids, hw = read_localization_db(zarr_path, extractor_name)

        # Geometry must cover exactly the DB's frames
        n = len(rec_ids)

        if (
            len(world_points) != n
            or len(extrinsics) != n
            or (original_coords is not None and len(original_coords) != n)
        ):
            raise ValueError(
                f"load_index: DB holds {n} frames; world_points, extrinsics and original_coords must match"
            )

        store = zarr.open(str(zarr_path), mode="r")
        rec_group = store[f"local_features/{extractor_name}/reconstruction"]

        # Same state as __init__, with the stored features and retrieval model in place of extraction
        retrieval = rec_group.attrs["retrieval"]
        obj = cls.__new__(cls)
        obj._setup(
            world_points,
            extrinsics,
            rec_ids,
            extractor=extractor,
            config=config,
            original_coords=original_coords,
            top_k=top_k,
            retrieval=retrieval,
        )
        global_desc = rec_group["global_desc"][:]
        obj._set_frames(rec_features, global_desc, hw)

        # Localized frames, if any: ids and the poses image_paths commits
        loc_key = f"local_features/{extractor_name}/localized"

        if loc_key in store:
            loc_group = store[loc_key]
            obj._localized_ids = [
                str(p) for p in loc_group.attrs.get("image_paths", [])
            ]
            obj._localized_extrinsics = list(
                loc_group["extrinsics"][: len(obj._localized_ids)]
            )

        logger.info(
            "CameraLocalizer.load_index: %d rec + %d loc frames [%s]",
            len(rec_ids),
            len(obj._localized_ids),
            extractor_name,
        )
        return obj

    def update_index(
        self,
        new_images: Iterable[np.ndarray],
        new_ids: list[str],
        zarr_path: Path | str,
        extractor_name: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Extract and embed new reconstruction frames; append them to the zarr DB in one write per array.

        - persist only: this localizer is unchanged, since the new frames have no world_points yet
        - reload via load_index with geometry that covers them

        Args:
            new_images: (h, w, 3) uint8 RGB frames, drawn lazily.
            new_ids: labels, index-aligned with new_images.
            zarr_path: pointcloud.zarr path.
            extractor_name: feature-cache key.
            progress_callback: called (frame_index, total) as frames are extracted.
        """
        new_features: list[LocalFeatures] = []
        descs = []

        # Extract and embed per chunk
        for chunk in _chunked(new_images, self._batch_size):
            start = len(new_features)
            feats = self._extractor.extract(chunk)
            assert isinstance(feats, list)
            new_features += feats
            descs.append(self._embed(chunk))

            if progress_callback is not None:
                for k in range(start, len(new_features)):
                    progress_callback(k, len(new_ids))

        new_desc = np.concatenate(descs)

        # No DB yet: write this localizer's frames first, then append
        if not localization_db_exists(zarr_path, extractor_name):
            logger.warning("update_index: no reconstruction DB, writing it first")
            self.save_index(zarr_path, extractor_name)

        store = zarr.open(str(zarr_path), mode="a")
        group = store[f"local_features/{extractor_name}/reconstruction"]

        # Arrays first, then the image_paths commit marker
        _append_csr(group, new_features)
        _append_rows(group, "global_desc", new_desc)
        paths = [str(p) for p in group.attrs["image_paths"]]
        paths += [str(i) for i in new_ids]
        group.attrs["image_paths"] = paths
        logger.info(
            "CameraLocalizer.update_index: appended %d frames [%s]",
            len(new_ids),
            extractor_name,
        )

    def add_localized_frame(
        self,
        image_path: Path | str,
        pose: np.ndarray,
        zarr_path: Path | str | None = None,
        extractor_name: str | None = None,
        provenance: dict | None = None,
    ) -> None:
        """
        Record a localized frame's pose; it is never a match source.

        - a repeat of an existing id is logged and skipped
        - persisted to the localized group when zarr_path and extractor_name are given
        - call clear_localized_frames after BA / LC updates that invalidate poses

        Args:
            image_path: frame id.
            pose: (4, 4) world-to-camera.
            zarr_path: pointcloud.zarr path; None keeps id and pose in memory only.
            extractor_name: feature-cache key; None keeps id and pose in memory only.
            provenance: opaque per-frame metadata stored beside the frame.
        """
        frame_id = str(image_path)

        if frame_id in self._ids or frame_id in self._localized_ids:
            logger.warning(
                "CameraLocalizer.add_localized_frame: %s already in index, skipping",
                frame_id,
            )
            return

        self._localized_ids.append(frame_id)
        self._localized_extrinsics.append(np.asarray(pose))

        if zarr_path is not None and extractor_name is not None:
            self._append_localized_to_zarr(
                frame_id, pose, zarr_path, extractor_name, provenance
            )

    @staticmethod
    def _append_localized_to_zarr(
        frame_id: str,
        pose: np.ndarray,
        zarr_path: Path | str,
        extractor_name: str,
        provenance: dict | None,
    ) -> None:
        """
        Write one localized frame's pose at row len(image_paths), then commit its id.

        - rows past image_paths, left by a crashed append, are overwritten
        """
        store = zarr.open(str(zarr_path), mode="a")
        loc_key = f"local_features/{extractor_name}/localized"
        pose_row = np.asarray(pose)[None]
        meta = provenance if provenance is not None else {}

        # First frame creates the group
        if loc_key not in store:
            group = store.require_group(loc_key)
            group.create_array(
                "extrinsics", data=pose_row, chunks=(1, 4, 4), compressors=LZ4
            )
            group.attrs.update({"provenance": [meta], "image_paths": [frame_id]})
            return

        # Later frames write row n, then commit provenance and image_paths together
        group = store[loc_key]
        paths = list(group.attrs["image_paths"])
        metas = list(group.attrs["provenance"])
        n = len(paths)
        group["extrinsics"].resize((n + 1, 4, 4))
        group["extrinsics"][n] = pose_row[0]
        group.attrs.update(
            {"provenance": metas[:n] + [meta], "image_paths": paths + [frame_id]}
        )

    @staticmethod
    def clear_localized_frames(zarr_path: Path | str, extractor_name: str) -> None:
        """
        Delete the extractor's localized group; reconstruction data is untouched.

        - call after BA / LC updates that invalidate localized poses, then reload via load_index

        Args:
            zarr_path: pointcloud.zarr path.
            extractor_name: feature-cache key.
        """
        store = zarr.open(str(zarr_path), mode="a")
        loc_key = f"local_features/{extractor_name}/localized"

        if loc_key in store:
            del store[loc_key]
            logger.info(
                "CameraLocalizer.clear_localized_frames: cleared '%s' from %s",
                extractor_name,
                zarr_path,
            )

    @classmethod
    def from_pointcloud(
        cls,
        result: PointcloudResult,
        *,
        zarr_path: Path | str,
        images: Iterable[np.ndarray] | None = None,
        ids: list[str] | None = None,
        extractor: LocalMatcher | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        top_k: int = 8,
        config: dict | None = None,
        retrieval: str = "dino-salad",
    ) -> CameraLocalizer:
        """
        Localizer for a reconstruction: loads the zarr feature DB, or builds and saves it.

        - cache hit: ids, max_num_keypoints, retrieval name and full-res (H, W) all match the DB
        - a hit draws nothing from images; an incomplete DB rebuilds
        - a rebuild replaces the DB and drops its localized frames

        Args:
            result: the reconstruction; reads world_points, extrinsics, original_coords, image_paths.
            zarr_path: pointcloud.zarr path holding the DB.
            images: (h, w, 3) uint8 RGB full-res reference frames, aligned to ids; drawn only on a rebuild.
            ids: frame labels, index-aligned with result's frames; None labels them str(result.image_paths).
            extractor: local matcher; None builds LocalMatcher("loma"). Its model_name keys the DB.
            progress_callback: called (frame_index, total) during a rebuild.
            top_k: retrieved reference frames per localize(refs=None).
            config: pycolmap options, as __init__.
            retrieval: retrieval registry name; keys the DB's global descriptors.

        Returns:
            The localizer.

        Raises:
            ValueError: result's per-frame arrays or ids disagree in length, or a rebuild is needed without images,
                or a rebuild's frame size disagrees with original_coords.
        """
        coords = result.original_coords
        assert result.world_points is not None
        n = len(result.world_points)

        # Default labels: the reconstruction's own image paths
        if ids is None:
            ids = [str(p) for p in result.image_paths]

        if (
            len(result.extrinsics) != n
            or len(ids) != n
            or (coords is not None and len(coords) != n)
        ):
            raise ValueError(
                "from_pointcloud: world_points, extrinsics, ids and original_coords must align per frame"
            )

        extractor = extractor if extractor is not None else LocalMatcher("loma")
        name = extractor.model_name

        # Cache key: what the DB was built from
        expected = {
            "image_paths": [str(i) for i in ids],
            "max_num_keypoints": extractor.max_num_keypoints,
            "retrieval": retrieval,
        }

        if coords is not None:
            expected["hw"] = [int(coords[0, 5]), int(coords[0, 4])]

        cached = {}

        if localization_db_exists(zarr_path, name):
            store = zarr.open_group(str(zarr_path), mode="r")
            group = store[f"local_features/{name}/reconstruction"]
            cached = dict(group.attrs)

        stale = [key for key, value in expected.items() if cached.get(key) != value]

        # Cache hit, unless the DB turns out incomplete
        if not stale:
            try:
                return cls.load_index(
                    zarr_path,
                    name,
                    result.world_points,
                    result.extrinsics,
                    config=config,
                    extractor=extractor,
                    original_coords=coords,
                    top_k=top_k,
                )
            except KeyError as exc:
                logger.info("CameraLocalizer: %s, rebuilding", exc)
        else:
            logger.info("CameraLocalizer: feature DB stale on %s, rebuilding", stale)

        # Rebuild from the caller's frames
        if images is None:
            raise ValueError("from_pointcloud: building the feature DB needs images")

        localizer = cls(
            result.world_points,
            result.extrinsics,
            images,
            ids,
            extractor=extractor,
            config=config,
            progress_callback=progress_callback,
            original_coords=coords,
            top_k=top_k,
            retrieval=retrieval,
        )
        localizer.save_index(zarr_path, name)
        return localizer

    ########################################
    # Localization
    ########################################

    def localize(
        self,
        query_image: np.ndarray,
        query_intrinsics: np.ndarray | None = None,
        refs: Sequence[int] | None = None,
    ) -> LocalizationResult:
        """
        World-to-camera pose of a query image.

        - refs None: DINO-SALAD ranks reconstruction frames, top_k are matched
        - refs given: exactly those reconstruction frames (align to a chosen scene image)
        - query shrunk to the reference long side; K and px returned on the original grid

        Args:
            query_image: HxWx3 uint8 RGB.
            query_intrinsics: (3, 3) K; None seeds from proportions; its focal is refined, fx == fy.
            refs: reconstruction frame indices to match against.

        Returns:
            LocalizationResult; pose None when PnP fails; query_intrinsics is the refined K on success.

        Raises:
            ValueError: refs names a frame with no world_points (not a reconstruction frame).
        """
        # Chosen frames must carry world points
        n_rec = len(self._world_points)

        if refs is not None and any(not 0 <= i < n_rec for i in refs):
            raise ValueError(
                f"localize: refs {list(refs)} must index reconstruction frames [0, {n_rec})"
            )

        # Match on a query no larger than the references; K follows the resize
        full_hw = query_image.shape[:2]
        small = self._shrink_query(query_image)
        small_hw = small.shape[:2]

        if query_intrinsics is None:
            query_intrinsics = seed_intrinsics(*small_hw)
        else:
            query_intrinsics = rescale_intrinsics(query_intrinsics, full_hw, small_hw)

        query_feats = self._extractor.extract(small)
        assert isinstance(query_feats, LocalFeatures)
        query_feats = self._extractor.to_device(query_feats)
        refs = self._rank_refs(small) if refs is None else list(refs)
        model_hw = self._world_points.shape[1:3]
        all_q, all_3d, all_ref, all_frame = [], [], [], []

        # Match each ref and lift its matched px through the world map
        for i in refs:
            ref_feats = self._extractor.to_device(self._frame_features[i])
            m = self._extractor.match(query_feats, ref_feats)

            if len(m) == 0:
                continue

            px_model = _crop_to_model_grid(m.ref_px, self._coords[i], model_hw)
            pts3d, valid = sample_world_points(self._world_points[i], px_model)
            n_valid = int(valid.sum())

            if n_valid == 0:
                continue

            all_q.append(m.query_px[valid])
            all_3d.append(pts3d[valid])
            all_ref.append(m.ref_px[valid])
            all_frame.append(np.full(n_valid, i, dtype=np.int32))

        result = self._solve_pnp(
            all_q, all_3d, all_ref, all_frame, small_hw, query_intrinsics
        )
        return _to_query_grid(result, small_hw, full_hw)

    def _shrink_query(self, query_image: np.ndarray) -> np.ndarray:
        """
        Query resized so its long side is the reference long side; never upscaled.
        """
        ref_long = max(self._image_hw)
        h, w = query_image.shape[:2]
        scale = ref_long / max(h, w)

        if scale >= 1:
            return query_image

        size = (round(w * scale), round(h * scale))
        return cv2.resize(query_image, size, interpolation=cv2.INTER_AREA)

    def _solve_pnp(
        self,
        all_q: list[np.ndarray],
        all_3d: list[np.ndarray],
        all_ref: list[np.ndarray],
        all_frame: list[np.ndarray],
        query_hw: tuple[int, int],
        query_intrinsics: np.ndarray,
    ) -> LocalizationResult:
        """
        LO-RANSAC + refinement PnP over the accumulated correspondences.

        - no pose: correspondences are still returned for display when any exist
        - SIMPLE_PINHOLE query camera: pycolmap refines the focal, the principal point stays fixed
        """
        ref_hw = self._image_hw
        n_corr = sum(len(a) for a in all_q)

        # Assemble correspondence arrays, when there are any
        pts2d = pts3d_matched = pts2d_ref = ref_frame_indices = None

        if n_corr > 0:
            pts2d = np.concatenate(all_q).astype(np.float32)
            pts3d_matched = np.concatenate(all_3d).astype(np.float32)
            pts2d_ref = np.concatenate(all_ref).astype(np.float32)
            ref_frame_indices = np.concatenate(all_frame).astype(np.int32)

        failed = LocalizationResult(
            pose=None,
            n_correspondences=n_corr,
            n_inliers=0,
            pts2d=pts2d,
            pts3d_matched=pts3d_matched,
            inlier_mask=None,
            pts2d_ref=pts2d_ref,
            ref_frame_indices=ref_frame_indices,
            query_intrinsics=query_intrinsics,
            ref_hw=ref_hw,
        )

        # PnP needs at least 4 correspondences
        if n_corr < 4:
            logger.warning(
                "CameraLocalizer: only %d 2D-3D correspondences, need >= 4 for PnP",
                n_corr,
            )
            return failed

        assert pts2d is not None and pts3d_matched is not None

        # Single-focal pinhole camera from the query intrinsics
        H, W = query_hw
        params = [
            float(query_intrinsics[0, 0]),
            float(query_intrinsics[0, 2]),
            float(query_intrinsics[1, 2]),
        ]
        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE", width=int(W), height=int(H), params=params
        )

        # Solver options from the config dict
        est_cfg = self.config.get("estimation", {})
        ransac_cfg = est_cfg.get("ransac", {})
        estimation_options = pycolmap.AbsolutePoseEstimationOptions()
        estimation_options.ransac.max_error = ransac_cfg.get("max_error", 50)
        ref_cfg = self.config.get("refinement", {})
        refinement_options = pycolmap.AbsolutePoseRefinementOptions()
        refinement_options.refine_focal_length = ref_cfg.get(
            "refine_focal_length", True
        )
        refinement_options.refine_extra_params = ref_cfg.get(
            "refine_extra_params", False
        )

        # Solve with LO-RANSAC + refinement
        pts2d_f64 = pts2d.astype(np.float64)
        pts3d_f64 = pts3d_matched.astype(np.float64)
        ret = pycolmap.estimate_and_refine_absolute_pose(
            pts2d_f64,
            pts3d_f64,
            camera,
            estimation_options=estimation_options,
            refinement_options=refinement_options,
        )
        n_inliers = 0 if ret is None else int(ret["num_inliers"])

        # Too few inliers: no pose, but keep the correspondences for display
        if n_inliers < 4:
            logger.warning(
                "CameraLocalizer: pycolmap failed (inliers=%d / %d correspondences)",
                n_inliers,
                n_corr,
            )
            failed.n_inliers = n_inliers
            return failed

        logger.info("CameraLocalizer: localized, %d / %d inliers", n_inliers, n_corr)

        # Inlier mask and 4x4 world-to-camera transform
        cam_from_world = ret["cam_from_world"]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = cam_from_world.rotation.matrix()
        pose[:3, 3] = cam_from_world.translation

        # Refined K from the camera pycolmap updated in place
        refined_K = camera.calibration_matrix()
        refined_K = refined_K.astype(np.float32)
        return dataclasses.replace(
            failed,
            pose=pose,
            n_inliers=n_inliers,
            inlier_mask=ret["inlier_mask"],
            query_intrinsics=refined_K,
        )
