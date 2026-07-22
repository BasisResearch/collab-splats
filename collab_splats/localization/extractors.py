"""Stage 2 — Local feature extraction + matching: each extractor owns its detect→match logic."""

from __future__ import annotations

import copy
import logging
import pathlib
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from kornia.feature import DISK, LightGlue
from loma import LoMa, LoMaB, LoMaG
from loma.device import device as loma_device
from loma.loma import filter_matches

# XFeat cloned at third_party/xfeat/ — no pip package available.
_XFEAT_PATH = str(pathlib.Path(__file__).parents[2] / "third_party" / "xfeat")
if _XFEAT_PATH not in sys.path:
    sys.path.insert(0, _XFEAT_PATH)

from modules.xfeat import XFeat  # third_party/xfeat/modules/

from collab_splats.utils.torch_utils import RegistryMixin

logger = logging.getLogger(__name__)


@dataclass
class LocalFeatures:
    """Container for local feature extraction output.

    `scores` is populated by extractors that produce per-keypoint saliency
    (e.g. XFeatExtractor). Extractors that don't produce scores (e.g. DiskExtractor)
    leave it as None — their matchers don't require it. `scales` is populated only
    by the dense XFeat* path, which needs per-keypoint extraction scale to apply
    subpixel refinement offsets at match time.
    """

    keypoints: torch.Tensor  # (N, 2) float32 pixel [x, y]
    descriptors: torch.Tensor  # (N, D) float32
    scores: torch.Tensor | None = None  # (N,) float32 — XFeat only
    scales: torch.Tensor | None = None  # (N,) — dense XFeat* only


@dataclass
class MatchResult:
    """Matched pixel coordinates between a query and one reference image."""

    query_px: np.ndarray  # (K, 2) float32 xy in query image
    ref_px: np.ndarray  # (K, 2) float32 xy in reference image

    def __len__(self) -> int:
        return len(self.query_px)


def _empty_match() -> MatchResult:
    """Zero-length MatchResult."""
    z = np.zeros((0, 2), dtype=np.float32)
    return MatchResult(query_px=z, ref_px=z)


class BaseLocalExtractor(RegistryMixin, ABC):
    """Abstract base for local feature extractors with name-based registry.

    Both extract() and match() must be implemented. Registry enables
    instantiation by name: BaseLocalExtractor.create("xfeat").
    """

    _registry: dict[str, type["BaseLocalExtractor"]] = {}

    @abstractmethod
    def extract(self, image: np.ndarray) -> LocalFeatures:
        """Return LocalFeatures for an HxWx3 uint8 RGB image."""

    @abstractmethod
    def match(
        self,
        query: LocalFeatures,
        db: LocalFeatures,
        image_hw: tuple[int, int],
    ) -> MatchResult:
        """Return MatchResult of matched (query_px, ref_px) pixel pairs."""


@BaseLocalExtractor.register("disk")
class DiskExtractor(BaseLocalExtractor):
    """DISK local feature extractor with LightGlue matcher.

    Detects and describes keypoints using DISK (kornia pretrained on depth),
    matches pairs with LightGlue. DISK and LightGlue are a matched pair —
    LightGlue was trained specifically for DISK descriptors.

    Weights: downloaded automatically to torch hub cache on first use (~4 MB).
    """

    def __init__(self, top_k: int = 1024, device: str | None = None):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._top_k = top_k

        # Load DISK detector and LightGlue matcher
        self._disk = DISK.from_pretrained("depth").to(self._device).eval()
        self._lightglue = LightGlue(features="disk").to(self._device).eval()

        logger.debug("DiskExtractor: loaded DISK + LightGlue on %s", self._device)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        """Extract DISK keypoints and descriptors.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            LocalFeatures with keypoints (N,2), descriptors (N,128), scores=None.
        """
        # Convert HxWx3 uint8 to 1xCxHxW float tensor in [0, 1]
        img_t = (torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0).to(self._device)

        # Detect keypoints and compute descriptors
        with torch.no_grad():
            features = self._disk(img_t, self._top_k, pad_if_not_divisible=True)
        kpts = features[0].keypoints.cpu()  # (N, 2)
        descs = features[0].descriptors.cpu()  # (N, 128)

        return LocalFeatures(keypoints=kpts, descriptors=descs)

    def match(
        self,
        query: LocalFeatures,
        db: LocalFeatures,
        image_hw: tuple[int, int],
    ) -> MatchResult:
        """Match query features against database features using LightGlue.

        Args:
            query:    LocalFeatures from the query image.
            db:       LocalFeatures from the database image.
            image_hw: (H, W) — required for LightGlue coordinate normalisation.

        Returns:
            MatchResult of matched (query_px, ref_px) pixel pairs.
        """
        # LightGlue normalize_keypoints expects image_size as [W, H] (not [H, W])
        wh = torch.tensor([[image_hw[1], image_hw[0]]], dtype=torch.float32)
        data = {
            "image0": {
                "keypoints": query.keypoints.unsqueeze(0).to(self._device),
                "descriptors": query.descriptors.unsqueeze(0).to(self._device),
                "image_size": wh.to(self._device),
            },
            "image1": {
                "keypoints": db.keypoints.unsqueeze(0).to(self._device),
                "descriptors": db.descriptors.unsqueeze(0).to(self._device),
                "image_size": wh.to(self._device),
            },
        }

        # matches0[i] = index in image1 for kpt i in image0, or -1 if unmatched
        with torch.no_grad():
            result = self._lightglue(data)

        matches0 = result["matches0"][0].cpu()  # (N,)
        valid = matches0 > -1
        idx_q = torch.where(valid)[0]
        idx_db = matches0[valid]
        if len(idx_q) == 0:
            return _empty_match()
        return MatchResult(
            query_px=query.keypoints[idx_q].numpy().astype(np.float32),
            ref_px=db.keypoints[idx_db].numpy().astype(np.float32),
        )


@BaseLocalExtractor.register("xfeat")
class XFeatExtractor(BaseLocalExtractor):
    """XFeat local feature extractor with mutual nearest-neighbour matching.

    Lightweight learned features from verlab/accelerated_features, vendored at
    third_party/xfeat/. Faster than DISK; suitable for CPU or real-time use.
    Paired with kornia match_mnn for descriptor matching.
    """

    def __init__(self, top_k: int = 1024, device: str | None = None):
        self._top_k = top_k
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load XFeat detector
        self._xfeat = XFeat()

        logger.debug("XFeatExtractor: loaded on %s", self._device)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        """Extract XFeat keypoints, scores, and descriptors.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            LocalFeatures with keypoints (N,2), descriptors (N,64), scores (N,).
        """
        # Convert HxWx3 uint8 to 1xCxHxW float tensor in [0, 1]
        img_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Detect keypoints and compute descriptors
        out = self._xfeat.detectAndCompute(img_t, top_k=self._top_k)
        kpts = out[0]["keypoints"].cpu()  # (N, 2)
        scores = out[0]["scores"].cpu()  # (N,)
        descs = out[0]["descriptors"].cpu()  # (N, 64)

        return LocalFeatures(keypoints=kpts, descriptors=descs, scores=scores)

    def match(
        self,
        query: LocalFeatures,
        db: LocalFeatures,
        image_hw: tuple[int, int],
    ) -> MatchResult:
        """Match query features against database features using XFeat LighterGlue.

        Canonical XFeat+LG pipeline: detectAndCompute output dict (with image_size
        added) passed to match_lighterglue. Returns pixel pairs, same contract as
        DiskExtractor.match().

        Args:
            query:    LocalFeatures from the query image (must have scores).
            db:       LocalFeatures from the database image (must have scores).
            image_hw: (H, W) — required for LighterGlue coordinate normalisation.

        Returns:
            MatchResult of matched (query_px, ref_px) pixel pairs.
        """
        # match_lighterglue expects image_size as (W, H)
        W, H = image_hw[1], image_hw[0]
        d0 = {
            "keypoints": query.keypoints.to(self._device),
            "scores": query.scores.to(self._device),
            "descriptors": query.descriptors.to(self._device),
            "image_size": (W, H),
        }
        d1 = {
            "keypoints": db.keypoints.to(self._device),
            "scores": db.scores.to(self._device),
            "descriptors": db.descriptors.to(self._device),
            "image_size": (W, H),
        }

        # XFeat's LighterGlue (built lazily on the first match) rebinds the kornia class
        # attribute LightGlue.default_conf to a 96-dim config — a global, permanent clobber
        # that would corrupt any later DISK LightGlue (256-dim) built in the same process.
        # Snapshot and restore it around the call so XFeat stays self-contained.
        saved_conf = copy.deepcopy(LightGlue.default_conf)
        try:
            # Returns mkpts_0, mkpts_1, idx — idx is (K, 2) index pairs
            _, _, idx = self._xfeat.match_lighterglue(d0, d1)
        finally:
            LightGlue.default_conf = saved_conf
        if len(idx) == 0:
            return _empty_match()
        idx_q = torch.from_numpy(np.asarray(idx[:, 0])).long()
        idx_db = torch.from_numpy(np.asarray(idx[:, 1])).long()
        return MatchResult(
            query_px=query.keypoints[idx_q].numpy().astype(np.float32),
            ref_px=db.keypoints[idx_db].numpy().astype(np.float32),
        )


@BaseLocalExtractor.register("xfeat-star")
class XFeatStarExtractor(BaseLocalExtractor):
    """XFeat* semi-dense matcher: cached dense features + pairwise match refinement.

    Mirrors XFeat.match_xfeat_star (accelerated_features xfeat_matching notebook
    flow) but splits it around the per-frame cache: detectAndComputeDense output
    (keypoints/descriptors/scales) is stored per frame, then batch_match +
    refine_matches run pairwise on the cached dicts — no image needed at match
    time. Output pixel pairs carry subpixel-refined query coordinates.
    """

    def __init__(self, top_k: int = 4096, device: str | None = None):
        self._top_k = top_k
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load XFeat backbone (also hosts the fine_matcher head used at match time)
        self._xfeat = XFeat()

        logger.debug("XFeatStarExtractor: loaded on %s", self._device)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        """Extract dense multiscale XFeat* keypoints, descriptors, and scales.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            LocalFeatures with keypoints (K,2), descriptors (K,64), scales (K,);
            K = top_k, sorted most→least reliable.
        """
        # Convert HxWx3 uint8 to 1xCxHxW float tensor in [0, 1]
        img_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Dense dual-scale extraction; batch dim of 1 is squeezed for caching
        out = self._xfeat.detectAndComputeDense(img_t, top_k=self._top_k)
        kpts = out["keypoints"][0].cpu()  # (K, 2)
        descs = out["descriptors"][0].cpu()  # (K, 64)
        scales = out["scales"][0].cpu()  # (K,)

        return LocalFeatures(keypoints=kpts, descriptors=descs, scales=scales)

    def match(
        self,
        query: LocalFeatures,
        db: LocalFeatures,
        image_hw: tuple[int, int],
    ) -> MatchResult:
        """Match cached dense features pairwise with subpixel refinement.

        Composes XFeat.batch_match + refine_matches exactly as match_xfeat_star
        does, but on two cached feature dicts instead of raw images. Query is
        passed as d0, so refine_matches applies subpixel offsets to the QUERY
        keypoints; ref_px stays on the db coarse grid (sampled for 2D→3D lookup).

        Args:
            query:    LocalFeatures from the query image (must have scales).
            db:       LocalFeatures from the database image.
            image_hw: (H, W) — unused; kept for the extractor interface.

        Returns:
            MatchResult of matched (query_px, ref_px) pixel pairs.
        """
        # Rebuild the batched dicts refine_matches expects (B=1)
        d0 = {
            "keypoints": query.keypoints.unsqueeze(0).to(self._device),
            "descriptors": query.descriptors.unsqueeze(0).to(self._device),
            "scales": query.scales.unsqueeze(0).to(self._device),
        }
        d1 = {
            "keypoints": db.keypoints.unsqueeze(0).to(self._device),
            "descriptors": db.descriptors.unsqueeze(0).to(self._device),
        }

        # Mutual-NN coarse match, then fine_matcher subpixel refinement (upstream
        # wraps both in inference_mode via match_xfeat_star — replicate that)
        with torch.inference_mode():
            idxs_list = self._xfeat.batch_match(d0["descriptors"], d1["descriptors"])
            pairs = self._xfeat.refine_matches(d0, d1, matches=idxs_list, batch_idx=0)

        # (K, 4) rows are [x_query, y_query, x_ref, y_ref]
        if len(pairs) == 0:
            return _empty_match()
        pairs = pairs.cpu().numpy().astype(np.float32)
        return MatchResult(query_px=pairs[:, :2], ref_px=pairs[:, 2:])


@BaseLocalExtractor.register("loma")
class LomaExtractor(BaseLocalExtractor):
    """LoMa-B local feature extractor and matcher (ECCV 2026).

    DaD keypoint detector + DeDoDe-G (DINOv2 ViT-L) descriptors + LoMa
    transformer matcher, from the `lomatch` package. Detection and description
    run once per image at a fixed 784x784 inference resolution; keypoints are
    stored in original-image pixel coordinates so the zarr feature cache and
    the 2D->3D depth-lookup stage work unchanged.

    Weights: auto-downloaded to torch hub cache on first use (~723 MB for
    LoMa-B, plus the DaD detector).
    """

    _cfg_factory = LoMaB
    _inference_hw = 784  # 14 * 56 — divisible by the DINOv2 patch size

    def __init__(self, top_k: int = 2048, filter_threshold: float = 0.1):
        # LoMa pins all modules and inputs to the module-level loma.device
        # global (cuda > cpu, chosen at import) — adopt it rather than fight it.
        self._device = loma_device
        self._top_k = top_k
        self._filter_threshold = filter_threshold

        # Load matcher + frozen detector/descriptor; downloads weights on first use
        self._loma = LoMa(self._cfg_factory()).eval()

        logger.debug("%s: loaded on %s", type(self).__name__, self._device)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        """Extract DaD keypoints and DeDoDe descriptors.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            LocalFeatures with keypoints (N,2) in original pixel coords,
            descriptors (N,256), scores (N,).
        """
        H, W = image.shape[:2]

        # HxWx3 uint8 -> (1,3,784,784) float [0,1]; LoMa normalizes internally.
        # Square resize is safe: keypoints come back normalized [-1,1], which is
        # resolution-invariant, and are denormalized against the ORIGINAL (W,H).
        img_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_t = F.interpolate(
            img_t,
            size=(self._inference_hw, self._inference_hw),
            mode="bilinear",
            align_corners=False,
        ).to(self._device)

        # Detect (normalized [-1,1] xy) then describe at those keypoints.
        # _descriptor is private, but it is the only per-image describe path
        # lomatch exposes (public match() is pair-composed) — re-check this on
        # any lomatch upgrade.
        with torch.inference_mode():
            det = self._loma.detect(img_t, num_keypoints=self._top_k)
            kpts_n = det["keypoints"]  # (1, N, 2) normalized xy
            probs = det["keypoint_probs"]  # (1, N)
            descs = self._loma._descriptor.describe_keypoints(img_t, kpts_n)["descriptions"]  # (1, N, 256)

        # Denormalize to original pixel coords: x_px = W * (x + 1) / 2
        wh = torch.tensor([W, H], dtype=torch.float32)
        kpts_px = (kpts_n[0].cpu().float() + 1.0) * wh / 2.0

        return LocalFeatures(
            keypoints=kpts_px,
            descriptors=descs[0].cpu().float(),
            scores=probs[0].cpu().float(),
        )

    def match(
        self,
        query: LocalFeatures,
        db: LocalFeatures,
        image_hw: tuple[int, int],
    ) -> MatchResult:
        """Match query features against database features with the LoMa matcher.

        Args:
            query:    LocalFeatures from the query image.
            db:       LocalFeatures from the database image.
            image_hw: (H, W) — used to re-normalize pixel coords to [-1, 1].

        Returns:
            MatchResult of matched (query_px, ref_px) pixel pairs.
        """
        # Pixel -> normalized [-1,1] (inverse of loma.loma.to_pixel_coords)
        wh = torch.tensor([image_hw[1], image_hw[0]], dtype=torch.float32)
        k0 = (2.0 * query.keypoints / wh - 1.0).unsqueeze(0).to(self._device)
        k1 = (2.0 * db.keypoints / wh - 1.0).unsqueeze(0).to(self._device)
        d0 = query.descriptors.unsqueeze(0).to(self._device)
        d1 = db.descriptors.unsqueeze(0).to(self._device)

        # Matcher forward autocasts internally (cfg.mp); returns assignment scores
        with torch.inference_mode():
            scores = self._loma(k0, k1, d0, d1)["scores"]
        m0, _, _, _ = filter_matches(scores, self._filter_threshold)

        # m0[i] = index in db for query kpt i, or -1 if unmatched
        m0 = m0[0].cpu()
        valid = m0 > -1
        idx_q = torch.where(valid)[0]
        idx_db = m0[valid].long()
        if len(idx_q) == 0:
            return _empty_match()
        return MatchResult(
            query_px=query.keypoints[idx_q].numpy().astype(np.float32),
            ref_px=db.keypoints[idx_db].numpy().astype(np.float32),
        )


@BaseLocalExtractor.register("loma-g")
class LomaGExtractor(LomaExtractor):
    """LoMa-G variant: larger matcher (embed_dim=1024), best accuracy, ~1.4 GB weights."""

    _cfg_factory = LoMaG
