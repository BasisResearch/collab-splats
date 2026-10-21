"""
Stage 2 local matching: vismatch features, cached once, matched in batches of pairs.

- xfeat and loma only: vismatch supports_batches, so match() reads cached features
- LocalFeatures is one cache row; MatchResult carries native keypoint-table indices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import torch

import vismatch

from collab_splats.utils.torch_utils import get_device, to_numpy

logger = logging.getLogger(__name__)


########################################
# Containers
########################################


@dataclass
class LocalFeatures:
    """
    Keypoints and descriptors of one image, as the feature cache stores them.

    - keypoints_normalized: loma's model-grid coords, which its matcher consumes
    - image_size: (W, H) of the image the keypoints were detected in
    """

    keypoints: torch.Tensor  # (N, 2) float32 pixel xy
    descriptors: torch.Tensor  # (N, D) float32
    keypoints_normalized: torch.Tensor | None = None  # (N, 2)
    image_size: tuple[int, int] | None = None  # (W, H)


@dataclass
class MatchResult:
    """
    Matched pixels between a query and one reference image, with their keypoint rows.

    - idx_q / idx_db index the two keypoint tables, COLMAP's match format
    """

    query_px: np.ndarray  # (K, 2) float32 xy in the query image
    ref_px: np.ndarray  # (K, 2) float32 xy in the reference image
    idx_q: np.ndarray  # (K,) int64 into the query keypoint table
    idx_db: np.ndarray  # (K,) int64 into the reference keypoint table

    def __len__(self) -> int:
        """
        Number of matches.
        """
        return len(self.query_px)


def _empty_match() -> MatchResult:
    """
    Zero-length MatchResult with empty index arrays.
    """
    query_px = np.zeros((0, 2), dtype=np.float32)
    ref_px = np.zeros((0, 2), dtype=np.float32)
    idx_q = np.zeros(0, dtype=np.int64)
    idx_db = np.zeros(0, dtype=np.int64)
    return MatchResult(query_px=query_px, ref_px=ref_px, idx_q=idx_q, idx_db=idx_db)


########################################
# Matcher
########################################


class LocalMatcher:
    """
    One vismatch model: extract() fills the feature cache, match() pairs two cached frames.

    - the model name is data, not a subclass
    - models without vismatch supports_batches are refused: they cannot match cached features
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        *,
        max_num_keypoints: int = 2048,
    ) -> None:
        """
        Load the vismatch model; refuse one that cannot match cached features.

        Args:
            model_name: vismatch model name, "xfeat" or "loma".
            device: torch device; None picks CUDA when available.
            max_num_keypoints: per-image keypoint cap, forwarded to vismatch unchanged.

        Raises:
            ValueError: the model lacks vismatch supports_batches.
        """
        self._model_name = model_name
        self._device = device or get_device()
        self._max_num_keypoints = max_num_keypoints
        self._matcher = vismatch.get_matcher(
            model_name, device=self._device, max_num_keypoints=max_num_keypoints
        )

        # Cached-feature matching runs on vismatch's batch path only
        if not self._matcher.supports_batches:
            raise ValueError(
                f"LocalMatcher: vismatch '{model_name}' cannot match cached features "
                "(supports_batches is False); use xfeat or loma"
            )

        # Callers verify geometry with pycolmap; vismatch's homography RANSAC is wasted work
        self._matcher.skip_ransac = True

    @property
    def model_name(self) -> str:
        """
        vismatch model name; also the feature-cache key.

        Returns:
            The name passed at construction.
        """
        return self._model_name

    @property
    def max_num_keypoints(self) -> int:
        """
        Per-image keypoint cap; stamped into the feature cache so a changed cap rebuilds it.

        Returns:
            The cap passed at construction.
        """
        return self._max_num_keypoints

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        HxWx3 RGB to (3, H, W) on device, dtype kept; vismatch scales uint8 itself.
        """
        array = np.ascontiguousarray(image)
        tensor = torch.from_numpy(array).to(self._device)
        return tensor.permute(2, 0, 1)

    @staticmethod
    def _check_pixel_frame(kpts: np.ndarray, hw: tuple[int, int], what: str) -> None:
        """
        Refuse keypoints outside the input image's pixel frame.
        """
        if len(kpts) and (
            kpts.min() < -0.5
            or kpts[:, 0].max() > hw[1] - 0.5
            or kpts[:, 1].max() > hw[0] - 0.5
        ):
            raise ValueError(
                f"vismatch '{what}' keypoints outside input pixel frame {hw}: "
                f"x range [{kpts[:, 0].min():.1f}, {kpts[:, 0].max():.1f}], "
                f"y range [{kpts[:, 1].min():.1f}, {kpts[:, 1].max():.1f}] — "
                "model likely returns coords at its internal resolution"
            )

    def extract(
        self, images: np.ndarray | list[np.ndarray]
    ) -> LocalFeatures | list[LocalFeatures]:
        """
        Keypoints and descriptors for one image, or for every image of a list in one vismatch call.

        Args:
            images: HxWx3 RGB uint8 (or float in [0, 1]), or a list of them.

        Returns:
            One LocalFeatures for one image; a list, in order, for a list. Tensors on CPU.

        Raises:
            ValueError: the model returned keypoints outside an input image.
        """
        batch = isinstance(images, list)
        images = images if isinstance(images, list) else [images]
        tensors = [self._to_tensor(image) for image in images]

        with torch.inference_mode():
            outs = self._matcher.extract(tensors)

        # vismatch returns numpy; to_numpy also tolerates tensors
        feats = []

        for image, out in zip(images, outs):
            kpts = to_numpy(out["all_kpts0"])
            kpts = kpts.astype(np.float32, copy=False)
            self._check_pixel_frame(kpts, image.shape[:2], self._model_name)
            desc = to_numpy(out["all_desc0"])
            desc = desc.astype(np.float32, copy=False)
            norm = out.get("kpts_normalized")

            if norm is not None:
                norm = to_numpy(norm)
                norm = norm.astype(np.float32, copy=False)
                norm = torch.from_numpy(norm)

            feats.append(
                LocalFeatures(
                    keypoints=torch.from_numpy(kpts),
                    descriptors=torch.from_numpy(desc),
                    keypoints_normalized=norm,
                    image_size=(image.shape[1], image.shape[0]),
                )
            )

        return feats if batch else feats[0]

    def to_device(self, features: LocalFeatures) -> LocalFeatures:
        """
        The same features on the matcher's device, so vismatch's per-match upload is a no-op.

        Args:
            features: features from extract() or the cache.

        Returns:
            A copy whose tensors sit on the matcher's device.
        """
        moved = {}

        for name in ("keypoints", "descriptors", "keypoints_normalized"):
            value = getattr(features, name)
            moved[name] = None if value is None else value.to(self._device)

        return replace(features, **moved)

    def match(self, query: LocalFeatures, db: LocalFeatures) -> MatchResult:
        """
        Match two cached frames; index rows are native keypoint-table indices.

        Args:
            query: query image features.
            db: reference image features.

        Returns:
            Pre-RANSAC matches; empty when either side has no keypoints.
        """
        return self.match_batch([(query, db)])[0]

    def match_batch(
        self, pairs: list[tuple[LocalFeatures, LocalFeatures]]
    ) -> list[MatchResult]:
        """
        Match many cached frame pairs in one vismatch match_batch call.

        - xfeat matches every pair in one forward; loma loops pairs inside vismatch
        - caller sizes the batch: xfeat holds two B x N x N similarity tensors

        Args:
            pairs: (query, db) feature pairs.

        Returns:
            One match() result per pair, in order; empty where either side has no keypoints.
        """
        results = [_empty_match() for _ in pairs]
        live = [
            i
            for i, (q, d) in enumerate(pairs)
            if len(q.descriptors) and len(d.descriptors)
        ]
        inputs = [
            (self._vismatch_features(pairs[i][0]), self._vismatch_features(pairs[i][1]))
            for i in live
        ]

        with torch.inference_mode():
            outs = self._matcher.match_batch(inputs)

        # MatchResult dtypes: float32 pixels, int64 keypoint indices
        for i, out in zip(live, outs):
            if len(out["matched_idxs0"]) == 0:
                continue

            query_px = out["matched_kpts0"].astype(np.float32, copy=False)
            ref_px = out["matched_kpts1"].astype(np.float32, copy=False)
            idx_q = out["matched_idxs0"].astype(np.int64, copy=False)
            idx_db = out["matched_idxs1"].astype(np.int64, copy=False)
            results[i] = MatchResult(
                query_px=query_px, ref_px=ref_px, idx_q=idx_q, idx_db=idx_db
            )

        return results

    @staticmethod
    def _vismatch_features(feats: LocalFeatures) -> dict:
        """
        LocalFeatures as a vismatch extract() dict; keypoints_normalized feeds loma.
        """
        out = {
            "all_kpts0": feats.keypoints,
            "all_desc0": feats.descriptors,
            "image_size": feats.image_size,
        }

        if feats.keypoints_normalized is not None:
            out["kpts_normalized"] = feats.keypoints_normalized

        return out
