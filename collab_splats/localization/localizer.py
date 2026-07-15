"""Stage 3 — Pose estimation: 2D→3D assignment + absolute pose via LO-RANSAC (pycolmap)."""
from __future__ import annotations

import logging
import pathlib
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
import pycolmap
import torch
import zarr
from zarr.codecs import BloscCodec

from .extractors import BaseLocalExtractor, DiskExtractor, LocalFeatures

logger = logging.getLogger(__name__)


@dataclass
class LocalizationResult:
    """Output of CameraLocalizer.localize().

    pts2d and pts3d_matched are the M correspondences fed to PnP.
    inlier_mask[i] is True when correspondence i survived RANSAC.
    pose is None when PnP fails or fewer than 4 correspondences exist.

    pts2d_ref holds the reference-frame pixel coordinates for each correspondence
    (same ordering as pts2d), enabling visualisation of matched keypoint pairs
    across query and reference images.
    ref_frame_indices records which reference frame each correspondence came from.
    """

    pose: np.ndarray | None            # (4, 4) world-to-camera, or None
    n_correspondences: int             # M — total 2D↔3D pairs before RANSAC
    n_inliers: int                     # RANSAC inlier count
    pts2d: np.ndarray | None           # (M, 2) query pixel coords
    pts3d_matched: np.ndarray | None   # (M, 3) matched world points
    inlier_mask: np.ndarray | None     # (M,) bool
    pts2d_ref: np.ndarray | None = None           # (M, 2) reference-frame pixel coords
    ref_frame_indices: np.ndarray | None = None   # (M,) int32 — source reference frame per correspondence
    query_features: "LocalFeatures | None" = None  # always set by localize(); pass to add_localized_frame


def _build_frame_assignments(
    pts3d: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    frame_keypoints: list[torch.Tensor],
    image_hw: tuple[int, int],
    radius: float = 8.0,
) -> list[dict[int, int]]:
    """For each reference frame, map keypoint indices to 3D point indices.

    Projects all 3D points into each frame and assigns each keypoint to its
    nearest visible projected point within `radius` pixels via torch.cdist.

    Args:
        pts3d:           (P, 3) world-space points.
        extrinsics:      (N, 4, 4) world-to-camera transforms.
        intrinsics:      (N, 3, 3) camera intrinsics K per frame.
        frame_keypoints: list of N tensors, each (K_i, 2) keypoints for frame i.
        image_hw:        (H, W) image dimensions for bounds filtering.
        radius:          max pixel distance for a keypoint to claim a 3D point.

    Returns:
        List of N dicts: {kpt_idx: pt3d_idx} for each frame.
    """
    H, W = image_hw
    assignments: list[dict[int, int]] = []

    for i in range(len(frame_keypoints)):
        R = extrinsics[i, :3, :3]  # (3, 3)
        t = extrinsics[i, :3, 3]   # (3,)
        K = intrinsics[i]           # (3, 3)

        # Project pts3d into frame i — world-to-camera: p_cam = R @ p_world + t
        pts_cam = pts3d @ R.T + t                          # (P, 3)
        visible_mask = pts_cam[:, 2] > 0                   # in front of camera
        visible_idx = np.where(visible_mask)[0]

        frame_assignment: dict[int, int] = {}
        kpts = frame_keypoints[i]

        if len(visible_idx) == 0 or len(kpts) == 0:
            assignments.append(frame_assignment)
            continue

        # Perspective projection: p_2d = K @ p_cam, then divide by depth
        pts_cam_vis = pts_cam[visible_mask]                # (Q, 3)
        pts_proj = pts_cam_vis @ K.T                       # (Q, 3)
        pts_proj_2d = pts_proj[:, :2] / pts_proj[:, 2:3]  # (Q, 2)  pixel coords

        # Filter to image bounds
        in_bounds = (
            (pts_proj_2d[:, 0] >= 0) & (pts_proj_2d[:, 0] < W) &
            (pts_proj_2d[:, 1] >= 0) & (pts_proj_2d[:, 1] < H)
        )
        if not in_bounds.any():
            assignments.append(frame_assignment)
            continue

        pts_proj_valid = torch.from_numpy(pts_proj_2d[in_bounds]).float()  # (V, 2)
        valid_pt_idx = visible_idx[in_bounds]                               # (V,) — original pt3d indices

        # Nearest-neighbour via torch.cdist: shape (K_i, V)
        dists = torch.cdist(kpts.float(), pts_proj_valid)   # (K_i, V)
        min_dists, nearest = dists.min(dim=1)               # (K_i,)

        # One-to-one: each 3D point is claimed by at most the closest keypoint
        claimed: dict[int, tuple[int, float]] = {}  # valid_pt_local_idx → (kpt_idx, dist)
        for kpt_idx in range(len(kpts)):
            d = min_dists[kpt_idx].item()
            if d < radius:
                pt_local = int(nearest[kpt_idx])
                if pt_local not in claimed or d < claimed[pt_local][1]:
                    claimed[pt_local] = (kpt_idx, d)

        for pt_local, (kpt_idx, _) in claimed.items():
            frame_assignment[kpt_idx] = int(valid_pt_idx[pt_local])

        assignments.append(frame_assignment)

    return assignments


class CameraLocalizer:
    """Locates a query camera within a known 3D scene.

    Matches the query image against all N reference frames via local feature
    matching (exhaustive — no global retrieval) then solves absolute pose via
    LO-RANSAC + Ceres refinement (pycolmap). Build once per scene; call
    localize() for each query image.

    Output convention matches FeedforwardResult.extrinsics: (4, 4) float32
    world-to-camera homogeneous transform.
    """

    def __init__(
        self,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        image_paths: list,
        extractor=None,
        radius: float = 8.0,
        config: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        """Build feature index from scene data.

        Args:
            pts3d:       (P, 3) float32 world-space 3D points.
            extrinsics:  (N, 4, 4) float32 world-to-camera transforms.
            intrinsics:  (N, 3, 3) float32 camera intrinsics K per frame.
            image_paths: length-N list of source image paths.
            extractor:   local feature extractor; defaults to DiskExtractor().
            radius:      pixel radius for 2D→3D keypoint assignment.
            config:      solver options dict. Keys:
                           "estimation" → pycolmap estimation_options
                             (default: {"ransac": {"max_error": 50}})
                           "refinement" → pycolmap refinement_options
                             (default: {"refine_focal_length": True,
                                        "refine_extra_params": True})
        """
        self.config = config or {}

        # Store scene geometry for use in localize()
        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._intrinsics = intrinsics

        self._extractor = extractor if extractor is not None else DiskExtractor()

        # Store image paths and provenance for duplicate guard and dashboard display.
        # _extrinsics stays reconstruction-only (assignment building depends on that); poses
        # for appended localized frames accumulate here and are joined by the extrinsics property.
        self._image_paths: list[Path] = [Path(p) for p in image_paths]
        self._frame_sources: list[str] = []
        self._localized_extrinsics: list[np.ndarray] = []

        logger.info(
            "CameraLocalizer: building index for %d frames, %d 3D points",
            len(image_paths), len(pts3d),
        )

        # TODO(future-C): pre-compute and store these features in feedforward.zarr so index
        # build is a zarr read (~1 s) instead of O(N) GPU inference. See spec 2026-05-29.

        # Extract local features for all reference frames
        self._frame_features: list[LocalFeatures] = []
        first_hw: tuple[int, int] | None = None
        # Use tqdm in notebook/terminal when no external progress_callback is wired
        if progress_callback is None:
            try:
                from tqdm.auto import tqdm as _tqdm
                _paths_iter = _tqdm(image_paths, desc="Indexing frames", unit="frame", leave=False)
            except ImportError:
                _paths_iter = image_paths
        else:
            _paths_iter = image_paths
        for path in _paths_iter:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"CameraLocalizer: cannot read {path}")
            rgb = bgr[..., ::-1].copy()
            if first_hw is None:
                first_hw = (rgb.shape[0], rgb.shape[1])
            feats = self._extractor.extract(rgb)
            self._frame_features.append(feats)
            if progress_callback is not None:
                progress_callback(len(self._frame_features) - 1, len(image_paths))
            logger.debug(
                "  frame %s: %d keypoints",
                path.name if hasattr(path, "name") else path,
                len(feats.keypoints),
            )

        self._image_hw: tuple[int, int] = first_hw or (480, 640)

        # Build kpt→3D assignment maps (one per frame)
        self._assignments = _build_frame_assignments(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_keypoints=[f.keypoints for f in self._frame_features],
            image_hw=self._image_hw,
            radius=radius,
        )

        self._frame_sources = ["reconstruction"] * len(self._frame_features)

        logger.info("CameraLocalizer: index built")

    @property
    def frame_sources(self) -> list[str]:
        """Provenance per frame: 'reconstruction' or 'localized'."""
        return list(self._frame_sources)

    @property
    def image_paths(self) -> list:
        """Reference image paths, index-aligned with frame_sources and extrinsics."""
        return list(self._image_paths)

    @property
    def extrinsics(self) -> np.ndarray:
        """(N, 4, 4) world-to-camera for ALL reference frames — reconstruction then localized.

        Index-aligned with image_paths / frame_sources / LocalizationResult.ref_frame_indices.
        """
        if not self._localized_extrinsics:
            return self._extrinsics
        return np.concatenate([self._extrinsics, np.stack(self._localized_extrinsics)], axis=0)

    def save_index(self, zarr_path: "str | Path", extractor_name: str,
                   attrs: "dict | None" = None) -> None:
        """Persist extracted frame features to feedforward.zarr reconstruction/ subgroup.

        Overwrites any existing reconstruction cache for extractor_name.
        Not automatically invalidated when source images change — caller's responsibility.
        Single-writer assumption; not safe for concurrent calls.

        Args:
            zarr_path:      feedforward.zarr store path.
            extractor_name: registry key naming the local_features/ subgroup.
            attrs:          optional build provenance (e.g. backbone, ba, lc, built_at)
                            written to the extractor-level group. Replaces any prior
                            attrs wholesale; extractor_name is always stamped.
        """
        lz4 = BloscCodec(cname="lz4")
        zarr_path = pathlib.Path(zarr_path)
        store = zarr.open(str(zarr_path), mode="a")

        # Clean overwrite: delete existing reconstruction group if present
        rec_key = f"local_features/{extractor_name}/reconstruction"
        if rec_key in store:
            del store[rec_key]

        rec_group = store.require_group(rec_key)

        # Stamp build provenance on the extractor group — wholesale replacement so a
        # rebuild never inherits stale attrs from a previous build
        ext_group = store.require_group(f"local_features/{extractor_name}")
        ext_group.attrs.put({"extractor": extractor_name, **(attrs or {})})

        # Build CSR frame_offsets from per-frame keypoint counts
        counts = [len(f.keypoints) for f in self._frame_features]
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        # Concatenate all keypoints and descriptors across frames
        if offsets[-1] > 0:
            all_kpts = np.concatenate(
                [f.keypoints.numpy() for f in self._frame_features], axis=0
            ).astype(np.float32)
            all_descs = np.concatenate(
                [f.descriptors.numpy() for f in self._frame_features], axis=0
            ).astype(np.float32)
        else:
            d = self._frame_features[0].descriptors.shape[1] if self._frame_features else 1
            all_kpts = np.zeros((0, 2), dtype=np.float32)
            all_descs = np.zeros((0, d), dtype=np.float32)

        rec_group.attrs["image_paths"] = [str(p) for p in self._image_paths]
        rec_group.attrs["hw"] = list(self._image_hw)

        rec_group.create_array("frame_offsets", data=offsets,
                               chunks=offsets.shape, compressors=lz4)
        rec_group.create_array("keypoints", data=all_kpts,
                               chunks=(max(all_kpts.shape[0], 1), 2), compressors=lz4)
        d_dim = all_descs.shape[1] if all_descs.shape[1] > 0 else 1
        rec_group.create_array("descriptors", data=all_descs,
                               chunks=(max(all_descs.shape[0], 1), d_dim),
                               compressors=lz4)

        # scores: XFeat only — skip if all None
        has_scores = any(f.scores is not None for f in self._frame_features)
        if has_scores:
            all_scores = np.concatenate([
                f.scores.numpy() if f.scores is not None
                else np.zeros(len(f.keypoints), dtype=np.float32)
                for f in self._frame_features
            ]).astype(np.float32)
            rec_group.create_array("scores", data=all_scores,
                                   chunks=(max(all_scores.shape[0], 1),), compressors=lz4)

        logger.info("CameraLocalizer.save_index: saved %d frames to %s [%s]",
                    len(self._frame_features), zarr_path, extractor_name)

    @classmethod
    def load_index(
        cls,
        zarr_path: "str | Path",
        extractor_name: str,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        config: "dict | None" = None,
        extractor=None,
        radius: float = 8.0,
    ) -> "CameraLocalizer":
        """Load feature index from zarr; rebuild kpt→3D assignments from current geometry.

        Loads reconstruction/ and localized/ (if present) groups and merges them.
        Raises KeyError if extractor_name reconstruction cache not found.
        """
        zarr_path = pathlib.Path(zarr_path)
        store = zarr.open(str(zarr_path), mode="r")

        rec_key = f"local_features/{extractor_name}/reconstruction"
        if rec_key not in store:
            raise KeyError(
                f"No feature cache for extractor '{extractor_name}' in {zarr_path}. "
                "Rebuild via CameraLocalizer.from_feedforward()."
            )

        # ── Load reconstruction group ────────────────────────────────────────
        rec_group = store[rec_key]
        rec_image_paths = [pathlib.Path(p) for p in rec_group.attrs["image_paths"]]
        hw = tuple(int(x) for x in rec_group.attrs["hw"])
        offsets = rec_group["frame_offsets"][:]
        all_kpts = (rec_group["keypoints"][:]
                    if rec_group["keypoints"].shape[0] > 0
                    else np.zeros((0, 2), dtype=np.float32))
        all_descs = (rec_group["descriptors"][:]
                     if rec_group["descriptors"].shape[0] > 0
                     else np.zeros((0, 1), dtype=np.float32))
        all_scores = rec_group["scores"][:] if "scores" in rec_group else None

        rec_features: list[LocalFeatures] = []
        for i in range(len(offsets) - 1):
            s, e = int(offsets[i]), int(offsets[i + 1])
            f_kpts = torch.from_numpy(all_kpts[s:e])
            f_descs = torch.from_numpy(all_descs[s:e])
            f_scores = torch.from_numpy(all_scores[s:e]) if all_scores is not None else None
            rec_features.append(LocalFeatures(keypoints=f_kpts, descriptors=f_descs, scores=f_scores))

        # ── Load localized group (optional) ──────────────────────────────────
        loc_key = f"local_features/{extractor_name}/localized"
        loc_features: list[LocalFeatures] = []
        loc_image_paths: list[pathlib.Path] = []
        loc_extrinsics_list: list[np.ndarray] = []
        loc_intrinsics_list: list[np.ndarray] = []

        if loc_key in store:
            loc_group = store[loc_key]
            loc_image_paths = [pathlib.Path(p) for p in loc_group.attrs.get("image_paths", [])]
            if loc_image_paths:
                loc_offsets = loc_group["frame_offsets"][:]
                loc_kpts = loc_group["keypoints"][:]
                loc_descs = loc_group["descriptors"][:]
                loc_scores = loc_group["scores"][:] if "scores" in loc_group else None
                loc_ext = loc_group["extrinsics"][:]   # (N_loc, 4, 4)
                loc_intr = loc_group["intrinsics"][:]  # (N_loc, 3, 3)
                for i in range(len(loc_offsets) - 1):
                    s, e = int(loc_offsets[i]), int(loc_offsets[i + 1])
                    f_kpts = torch.from_numpy(loc_kpts[s:e])
                    f_descs = torch.from_numpy(loc_descs[s:e])
                    f_scores = torch.from_numpy(loc_scores[s:e]) if loc_scores is not None else None
                    loc_features.append(LocalFeatures(keypoints=f_kpts, descriptors=f_descs, scores=f_scores))
                    loc_extrinsics_list.append(loc_ext[i])
                    loc_intrinsics_list.append(loc_intr[i])

        # ── Build assignments ─────────────────────────────────────────────────
        rec_assignments = _build_frame_assignments(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_keypoints=[f.keypoints for f in rec_features],
            image_hw=hw,
            radius=radius,
        )

        if loc_features:
            loc_ext_arr = np.stack(loc_extrinsics_list, axis=0)
            loc_intr_arr = np.stack(loc_intrinsics_list, axis=0)
            loc_assignments = _build_frame_assignments(
                pts3d=pts3d,
                extrinsics=loc_ext_arr,
                intrinsics=loc_intr_arr,
                frame_keypoints=[f.keypoints for f in loc_features],
                image_hw=hw,
                radius=radius,
            )
        else:
            loc_assignments = []

        # ── Assemble object without running __init__ extraction loop ──────────
        obj = object.__new__(cls)
        obj.config = config or {}
        obj._pts3d = pts3d
        obj._extrinsics = extrinsics
        obj._intrinsics = intrinsics
        obj._extractor = extractor if extractor is not None else DiskExtractor()
        obj._image_hw = hw
        obj._frame_features = rec_features + loc_features
        obj._frame_sources = (["reconstruction"] * len(rec_features) +
                              ["localized"] * len(loc_features))
        obj._image_paths = rec_image_paths + loc_image_paths
        obj._assignments = rec_assignments + loc_assignments
        # Keep the localized poses so extrinsics stays aligned with image_paths/frame_sources
        obj._localized_extrinsics = list(loc_extrinsics_list)

        logger.info(
            "CameraLocalizer.load_index: loaded %d rec + %d loc frames from %s [%s]",
            len(rec_features), len(loc_features), zarr_path, extractor_name,
        )
        return obj

    def update_index(
        self,
        new_image_paths: list,
        zarr_path: "str | Path",
        extractor_name: str,
        progress_callback: "Callable[[int, int], None] | None" = None,
    ) -> None:
        """Extract features for new reconstruction frames; append to zarr cache.

        Does NOT update pts3d/extrinsics/intrinsics — caller must update those
        and call clear_localized_frames() + load_index() to rebuild assignments.
        """
        zarr_path = pathlib.Path(zarr_path)
        new_features: list[LocalFeatures] = []

        for i, path in enumerate(new_image_paths):
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"CameraLocalizer.update_index: cannot read {path}")
            rgb = bgr[..., ::-1].copy()
            feats = self._extractor.extract(rgb)
            new_features.append(feats)
            if progress_callback is not None:
                progress_callback(i, len(new_image_paths))
            logger.debug("update_index: frame %s: %d kpts", path, len(feats.keypoints))

        # Update in-memory state
        for path, feats in zip(new_image_paths, new_features):
            self._frame_features.append(feats)
            self._frame_sources.append("reconstruction")
            self._image_paths.append(pathlib.Path(path))

        # Append to reconstruction/ zarr group
        store = zarr.open(str(zarr_path), mode="a")
        rec_key = f"local_features/{extractor_name}/reconstruction"

        if rec_key not in store:
            logger.warning("update_index: no existing reconstruction cache — building from scratch")
            self.save_index(zarr_path, extractor_name)
            return

        rec_group = store[rec_key]

        # Update attrs
        existing_paths = list(rec_group.attrs.get("image_paths", []))
        existing_paths.extend([str(p) for p in new_image_paths])
        rec_group.attrs["image_paths"] = existing_paths

        # Append CSR data frame by frame
        for feats in new_features:
            kpts_np = feats.keypoints.numpy().astype(np.float32)
            descs_np = feats.descriptors.numpy().astype(np.float32)

            off_arr = rec_group["frame_offsets"]
            last_off = int(off_arr[-1])
            n_off = off_arr.shape[0]
            off_arr.resize((n_off + 1,))
            off_arr[n_off] = last_off + len(kpts_np)

            kpts_arr = rec_group["keypoints"]
            old_m = kpts_arr.shape[0]
            kpts_arr.resize((old_m + len(kpts_np), kpts_arr.shape[1]))
            kpts_arr[old_m:] = kpts_np

            descs_arr = rec_group["descriptors"]
            descs_arr.resize((old_m + len(descs_np), descs_arr.shape[1]))
            descs_arr[old_m:] = descs_np

            if feats.scores is not None and "scores" in rec_group:
                scores_np = feats.scores.numpy().astype(np.float32)
                sc_arr = rec_group["scores"]
                old_sc = sc_arr.shape[0]
                sc_arr.resize((old_sc + len(scores_np),))
                sc_arr[old_sc:] = scores_np

        logger.info("CameraLocalizer.update_index: appended %d frames to %s [%s]",
                    len(new_image_paths), zarr_path, extractor_name)

    def add_localized_frame(
        self,
        image_path: "str | Path",
        pose: np.ndarray,
        intrinsics: np.ndarray,
        features: "LocalFeatures",
        zarr_path: "str | Path | None" = None,
        extractor_name: "str | None" = None,
        provenance: "dict | None" = None,
    ) -> None:
        """Add a successfully localized frame to the in-memory reference set.

        Rebuilds kpt→3D assignments for the new frame from existing pts3d + given pose.
        If zarr_path and extractor_name are provided, appends to localized/ in zarr.
        provenance: optional per-frame metadata (e.g. video_ref, session, camera,
        frame_idx) recorded alongside the localized frame in zarr.
        Single-writer; not thread-safe across concurrent callers.
        Call clear_localized_frames() after BA/LC updates that invalidate poses.
        """
        image_path = pathlib.Path(image_path)

        # Duplicate guard
        if image_path in self._image_paths:
            logger.warning(
                "CameraLocalizer.add_localized_frame: %s already in index, skipping",
                image_path.name,
            )
            return

        # Build kpt→3D assignment for this frame using existing pts3d
        new_assignments = _build_frame_assignments(
            pts3d=self._pts3d,
            extrinsics=pose[np.newaxis],       # (1, 4, 4)
            intrinsics=intrinsics[np.newaxis],  # (1, 3, 3)
            frame_keypoints=[features.keypoints],
            image_hw=self._image_hw,
        )

        # Append in-memory; the pose keeps extrinsics aligned with image_paths/frame_sources
        self._frame_features.append(features)
        self._frame_sources.append("localized")
        self._image_paths.append(image_path)
        self._assignments.extend(new_assignments)
        self._localized_extrinsics.append(np.asarray(pose))

        # Persist to zarr if requested
        if zarr_path is not None and extractor_name is not None:
            self._append_localized_to_zarr(
                image_path, pose, intrinsics, features,
                pathlib.Path(zarr_path), extractor_name,
                provenance=provenance,
            )

    def _append_localized_to_zarr(
        self,
        image_path: "pathlib.Path",
        pose: np.ndarray,
        intrinsics: np.ndarray,
        features: "LocalFeatures",
        zarr_path: "pathlib.Path",
        extractor_name: str,
        provenance: "dict | None" = None,
    ) -> None:
        """Append one localized frame to the localized/ zarr group."""
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(zarr_path), mode="a")
        loc_key = f"local_features/{extractor_name}/localized"

        kpts_np = features.keypoints.numpy().astype(np.float32)
        descs_np = features.descriptors.numpy().astype(np.float32)
        scores_np = (features.scores.numpy().astype(np.float32)
                     if features.scores is not None else None)

        if loc_key not in store:
            # First localized frame — create group + arrays
            loc_group = store.require_group(loc_key)
            offsets = np.array([0, len(kpts_np)], dtype=np.int64)
            loc_group.attrs["image_paths"] = [str(image_path)]
            loc_group.attrs["provenance"] = [provenance or {}]
            loc_group.create_array("frame_offsets", data=offsets,
                                   chunks=(max(offsets.shape[0], 2),), compressors=lz4)
            loc_group.create_array("keypoints", data=kpts_np,
                                   chunks=(max(kpts_np.shape[0], 1), 2), compressors=lz4)
            loc_group.create_array("descriptors", data=descs_np,
                                   chunks=(max(descs_np.shape[0], 1),
                                           max(descs_np.shape[1], 1)), compressors=lz4)
            if scores_np is not None:
                loc_group.create_array("scores", data=scores_np,
                                       chunks=(max(scores_np.shape[0], 1),), compressors=lz4)
            loc_group.create_array("extrinsics", data=pose[np.newaxis],
                                   chunks=(1, 4, 4), compressors=lz4)
            loc_group.create_array("intrinsics", data=intrinsics[np.newaxis],
                                   chunks=(1, 3, 3), compressors=lz4)
        else:
            # Append to existing group
            loc_group = store[loc_key]
            existing = list(loc_group.attrs.get("image_paths", []))
            existing.append(str(image_path))
            loc_group.attrs["image_paths"] = existing

            prov_list = list(loc_group.attrs.get("provenance", []))
            # Backfill empty provenance for frames appended before provenance existed
            while len(prov_list) < len(existing) - 1:
                prov_list.append({})
            prov_list.append(provenance or {})
            loc_group.attrs["provenance"] = prov_list

            off_arr = loc_group["frame_offsets"]
            last_off = int(off_arr[-1])
            n_off = off_arr.shape[0]
            off_arr.resize((n_off + 1,))
            off_arr[n_off] = last_off + len(kpts_np)

            kpts_arr = loc_group["keypoints"]
            old_m = kpts_arr.shape[0]
            kpts_arr.resize((old_m + len(kpts_np), kpts_arr.shape[1]))
            kpts_arr[old_m:] = kpts_np

            descs_arr = loc_group["descriptors"]
            descs_arr.resize((old_m + len(descs_np), descs_arr.shape[1]))
            descs_arr[old_m:] = descs_np

            if scores_np is not None and "scores" in loc_group:
                sc_arr = loc_group["scores"]
                old_sc = sc_arr.shape[0]
                sc_arr.resize((old_sc + len(scores_np),))
                sc_arr[old_sc:] = scores_np

            ext_arr = loc_group["extrinsics"]
            n_loc = ext_arr.shape[0]
            ext_arr.resize((n_loc + 1, 4, 4))
            ext_arr[n_loc] = pose

            intr_arr = loc_group["intrinsics"]
            intr_arr.resize((n_loc + 1, 3, 3))
            intr_arr[n_loc] = intrinsics

        logger.debug("CameraLocalizer: appended localized frame %s to zarr", image_path.name)

    @staticmethod
    def clear_localized_frames(zarr_path: "str | Path", extractor_name: str) -> None:
        """Delete the localized/ group for extractor_name from feedforward.zarr.

        Reconstruction data is untouched. Call this after BA/LC updates that
        invalidate previously estimated localized poses, then reload via load_index().
        """
        store = zarr.open(str(pathlib.Path(zarr_path)), mode="a")
        loc_key = f"local_features/{extractor_name}/localized"
        if loc_key in store:
            del store[loc_key]
            logger.info(
                "CameraLocalizer.clear_localized_frames: cleared '%s' from %s",
                extractor_name, zarr_path,
            )
        else:
            logger.debug(
                "CameraLocalizer.clear_localized_frames: no localized group for '%s'",
                extractor_name,
            )

    @classmethod
    def from_feedforward(
        cls,
        result,
        extractor=None,
        progress_callback=None,
        zarr_path=None,
        extractor_name=None,
        **kwargs,
    ) -> "CameraLocalizer":
        """Construct from a FeedforwardResult. Loads from zarr cache if available.

        Args:
            result:            FeedforwardResult (or duck-typed object with .points,
                               .extrinsics, .intrinsics, .image_paths, ._zarr_path).
            extractor:         Local feature extractor; defaults to DiskExtractor().
            progress_callback: Called as (frame_idx, total) during index build.
            zarr_path:         Override zarr cache path; falls back to result._zarr_path.
            extractor_name:    Override extractor registry key; auto-detected if None.
            **kwargs:          Forwarded to CameraLocalizer.__init__ (e.g. radius).

        Returns:
            CameraLocalizer ready to localize query images in the given scene.
        """
        extractor_inst = extractor if extractor is not None else DiskExtractor()

        # Determine extractor_name via registry reverse-lookup
        if extractor_name is None:
            extractor_name = next(
                (k for k, v in BaseLocalExtractor._registry.items()
                 if v is type(extractor_inst)),
                type(extractor_inst).__name__.lower().replace("extractor", ""),
            )

        # Resolve zarr_path: explicit arg > result._zarr_path
        if zarr_path is None:
            zarr_path = getattr(result, "_zarr_path", None)

        # Try cache first
        if zarr_path is not None:
            try:
                store = zarr.open(str(zarr_path), mode="r")
                rec_key = f"local_features/{extractor_name}/reconstruction"
                if rec_key in store:
                    # Staleness check: warn if image_paths differ
                    cached_paths = [pathlib.Path(p) for p in store[rec_key].attrs["image_paths"]]
                    if cached_paths != list(result.image_paths):
                        logger.warning(
                            "CameraLocalizer: cached image_paths differ from result — cache may be stale"
                        )
                    logger.info(
                        "CameraLocalizer: cache hit for '%s', loading from zarr", extractor_name
                    )
                    return cls.load_index(
                        zarr_path=zarr_path,
                        extractor_name=extractor_name,
                        pts3d=result.points,
                        extrinsics=result.extrinsics,
                        intrinsics=result.intrinsics,
                        extractor=extractor_inst,
                        **{k: v for k, v in kwargs.items() if k in ("config", "radius")},
                    )
            except KeyError:
                logger.debug(
                    "CameraLocalizer: cache miss for '%s', building index", extractor_name
                )
            except Exception as exc:
                logger.warning("CameraLocalizer: cache load failed (%s), rebuilding", exc)

        # Cache miss — build from GPU inference
        localizer = cls(
            pts3d=result.points,
            extrinsics=result.extrinsics,
            intrinsics=result.intrinsics,
            image_paths=result.image_paths,
            extractor=extractor_inst,
            progress_callback=progress_callback,
            **kwargs,
        )

        # Save for next session
        if zarr_path is not None:
            try:
                localizer.save_index(zarr_path, extractor_name)
            except Exception as exc:
                logger.warning("CameraLocalizer: failed to save index to zarr: %s", exc)

        return localizer

    def localize(
        self,
        query_image: np.ndarray,
        query_intrinsics: np.ndarray,
    ) -> LocalizationResult:
        """Estimate world-to-camera pose for a query image.

        Matches query against all N reference frames, collects 2D↔3D
        correspondences, solves absolute pose via LO-RANSAC + Ceres refinement.

        Args:
            query_image:      HxWx3 uint8 RGB image.
            query_intrinsics: (3, 3) float32 or float64 camera matrix K.

        Returns:
            LocalizationResult with pose (4, 4) and inlier data.
            pose is None if solver fails or fewer than 4 correspondences exist.
        """
        # Extract local features from query image
        query_feats = self._extractor.extract(query_image)

        logger.debug("CameraLocalizer.localize: query has %d keypoints", len(query_feats.keypoints))

        # Match query against all reference frames; deduplicate via best score per 3D point
        best_score: dict[int, float] = {}    # pt3d_idx → best confidence so far
        best_2d: dict[int, np.ndarray] = {}  # pt3d_idx → corresponding query 2D point
        best_ref_2d: dict[int, np.ndarray] = {}    # pt3d_idx → reference 2D point
        best_ref_frame: dict[int, int] = {}        # pt3d_idx → reference frame index

        for i, db_feats in enumerate(self._frame_features):
            if len(db_feats.keypoints) == 0:
                continue
            matches = self._extractor.match(query_feats, db_feats, self._image_hw)
            if len(matches) == 0:
                continue

            # Resolve matched db keypoints to world 3D points via assignment map
            assignment = self._assignments[i]
            for q_idx, db_idx in matches.numpy():
                db_idx = int(db_idx)
                if db_idx not in assignment:
                    continue
                pt3d_idx = assignment[db_idx]
                pt2d = query_feats.keypoints[int(q_idx)].numpy().astype(np.float32)
                score = 1.0  # uniform confidence; match_lighterglue idx does not carry per-match scores

                if pt3d_idx not in best_score or score > best_score[pt3d_idx]:
                    best_score[pt3d_idx] = score
                    best_2d[pt3d_idx] = pt2d
                    best_ref_2d[pt3d_idx] = db_feats.keypoints[int(db_idx)].numpy().astype(np.float32)
                    best_ref_frame[pt3d_idx] = i

        if len(best_2d) < 4:
            logger.warning(
                "CameraLocalizer: only %d 2D↔3D correspondences — need ≥4 for PnP",
                len(best_2d),
            )
            return LocalizationResult(
                pose=None, n_correspondences=len(best_2d), n_inliers=0,
                pts2d=None, pts3d_matched=None, inlier_mask=None,
                pts2d_ref=None, ref_frame_indices=None,
                query_features=query_feats,
            )

        # Assemble correspondence arrays for PnP
        pts2d = np.array(list(best_2d.values()), dtype=np.float32)
        pts3d_matched = np.array(
            [self._pts3d[idx] for idx in best_2d.keys()], dtype=np.float32
        )
        pts2d_ref = np.array(list(best_ref_2d.values()), dtype=np.float32)
        ref_frame_indices = np.array(list(best_ref_frame.values()), dtype=np.int32)

        # Build pycolmap camera from query intrinsics
        H, W = query_image.shape[:2]
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=int(W),
            height=int(H),
            params=[
                float(query_intrinsics[0, 0]),  # fx
                float(query_intrinsics[1, 1]),  # fy
                float(query_intrinsics[0, 2]),  # cx
                float(query_intrinsics[1, 2]),  # cy
            ],
        )

        # Build pycolmap solver options from config dict
        est_cfg = self.config.get("estimation", {})
        estimation_options = pycolmap.AbsolutePoseEstimationOptions()
        estimation_options.ransac.max_error = est_cfg.get("ransac", {}).get("max_error", 50)

        ref_cfg = self.config.get("refinement", {})
        refinement_options = pycolmap.AbsolutePoseRefinementOptions()
        refinement_options.refine_focal_length = ref_cfg.get("refine_focal_length", True)
        refinement_options.refine_extra_params = ref_cfg.get("refine_extra_params", True)

        # Solve with LO-RANSAC + Ceres refinement
        ret = pycolmap.estimate_and_refine_absolute_pose(
            pts2d.astype(np.float64),
            pts3d_matched.astype(np.float64),
            camera,
            estimation_options=estimation_options,
            refinement_options=refinement_options,
        )

        if ret is None or ret["num_inliers"] < 4:
            logger.warning(
                "CameraLocalizer: pycolmap failed (inliers=%d / %d correspondences)",
                ret["num_inliers"] if ret is not None else 0,
                len(pts2d),
            )
            return LocalizationResult(
                pose=None, n_correspondences=len(pts2d),
                n_inliers=ret["num_inliers"] if ret is not None else 0,
                pts2d=pts2d, pts3d_matched=pts3d_matched, inlier_mask=None,
                pts2d_ref=pts2d_ref, ref_frame_indices=ref_frame_indices,
                query_features=query_feats,
            )

        logger.info(
            "CameraLocalizer: localized — %d / %d inliers",
            ret["num_inliers"], len(pts2d),
        )

        # Build inlier mask and 4×4 world-to-camera transform
        inlier_mask = ret["inlier_mask"]
        cam_from_world = ret["cam_from_world"]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = cam_from_world.rotation.matrix()
        pose[:3, 3] = cam_from_world.translation
        return LocalizationResult(
            pose=pose, n_correspondences=len(pts2d), n_inliers=ret["num_inliers"],
            pts2d=pts2d, pts3d_matched=pts3d_matched, inlier_mask=inlier_mask,
            pts2d_ref=pts2d_ref, ref_frame_indices=ref_frame_indices,
            query_features=query_feats,
        )
