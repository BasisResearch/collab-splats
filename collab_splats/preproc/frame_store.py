"""
Canonical decode-once store of selected keyframes (chunked zarr).

The preprocess stage decodes a video exactly once and writes frames.zarr:
chunked-per-frame RGB images, columnar selection records, and provenance
attrs. All pixel consumers read from here instead of re-decoding the video.
Path-locked consumers (model preprocessing) use export() for a transient dir.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import zarr
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)


class FrameStore:
    """
    Persist and serve selected keyframes from a chunked frames.zarr store.
    """

    def __init__(self, path: Path, store):
        self.path = Path(path)
        self._store = store

        # frame_idx -> row position, for source-index lookups
        self._idx_to_row = {int(fi): row for row, fi in enumerate(store["frame_idx"][:])}

    @classmethod
    def create(cls, path, frames, records, *, provenance) -> "FrameStore":
        """
        Write frames + records + provenance to a new frames.zarr and return it open.
        """
        if not records or "frame_idx" not in records[0]:
            raise ValueError("FrameStore.create: every record must contain 'frame_idx' (source video index)")

        path = Path(path)
        imgs = np.stack(frames).astype(np.uint8)  # (N, H, W, 3) RGB
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(path), mode="w")

        # Chunk one frame per chunk so a consumer reads a single keyframe alone
        store.create_array("images", data=imgs, chunks=(1, *imgs.shape[1:]), compressors=[lz4])

        # Columnar records: every key present on any record becomes an array
        keys = sorted({k for r in records for k in r})
        for k in keys:
            col = np.array([r.get(k, np.nan) for r in records])
            store.create_array(k, data=col, chunks=col.shape, compressors=[lz4])

        # Provenance is descriptive only — reuse is by existence, never by comparison
        store.attrs["record_keys"] = keys
        store.attrs["provenance"] = {k: provenance.get(k) for k in provenance}
        store.attrs["schema_version"] = 1

        return cls(path, zarr.open(str(path), mode="r"))

    @classmethod
    def open(cls, path) -> "FrameStore":
        """
        Open an existing frames.zarr read-only.
        """
        return cls(path, zarr.open(str(path), mode="r"))

    @classmethod
    def frame_idx_from_path(cls, path) -> int:
        """
        Source frame index encoded in a frame_{idx:06d}.<ext> filename.
        """
        return int(Path(path).stem.split("_")[-1])

    def __len__(self) -> int:
        return int(self._store["images"].shape[0])

    def image(self, i: int) -> np.ndarray:
        """
        i-th selected frame (H, W, 3) uint8 RGB — single-chunk partial read.
        """
        return self._store["images"][i]

    def image_by_frame_idx(self, frame_idx: int) -> np.ndarray:
        """
        Frame by SOURCE video index; KeyError if that index was not selected.
        """
        if frame_idx not in self._idx_to_row:
            raise KeyError(f"frame_idx {frame_idx} not in store {self.path}")
        return self.image(self._idx_to_row[frame_idx])

    def has_frame_idx(self, frame_idx: int) -> bool:
        """
        True if that SOURCE video index is among the selected frames.
        """
        return int(frame_idx) in self._idx_to_row

    def images(self, idxs=None) -> np.ndarray:
        """
        Stack of selected frames (all, or the given row positions).
        """
        if idxs is None:
            return self._store["images"][:]
        return np.stack([self._store["images"][i] for i in idxs])

    def record(self, i: int) -> dict:
        """
        Selection record dict for the i-th selected frame.
        """
        keys = list(self._store.attrs["record_keys"])
        return {k: self._store[k][i] for k in keys}

    def frame_indices(self) -> np.ndarray:
        """
        Source video indices of the selected frames, in order.
        """
        return self._store["frame_idx"][:].astype(int)

    def provenance(self) -> dict:
        """
        Provenance attrs this store was written with (video_path, method, undistort, ...).
        """
        return dict(self._store.attrs.get("provenance", {}))

    def export(self, out_dir, *, ext="jpg") -> list[Path]:
        """
        Write frames to out_dir as frame_NNNNNN.<ext> (source-idx named); return paths.

        Transient bridge for path-locked consumers (model preprocessing); the
        caller deletes out_dir after use. Derived from the store, no re-decode.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for row, fi in enumerate(self.frame_indices()):
            p = out_dir / f"frame_{int(fi):06d}.{ext}"

            # Store holds RGB; cv2 writes BGR
            cv2.imwrite(str(p), cv2.cvtColor(self.image(row), cv2.COLOR_RGB2BGR))
            paths.append(p)

        return paths
