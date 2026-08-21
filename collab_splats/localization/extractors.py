"""Stage 2 — Local feature matching: vismatch-backed LocalMatcher is the sole provider."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import numpy as np
import torch
from kornia.feature import match_mnn

logger = logging.getLogger(__name__)


@dataclass
class LocalFeatures:
    """Container for local feature extraction output.

    `scores`/`scales` are optional per-keypoint extras some models produce
    (saliency scores, extraction scales); consumers that don't need them
    leave them as None.
    """

    keypoints: torch.Tensor  # (N, 2) float32 pixel [x, y]
    descriptors: torch.Tensor  # (N, D) float32
    scores: torch.Tensor | None = None  # (N,) float32 — optional saliency
    scales: torch.Tensor | None = None  # (N,) — optional extraction scale
    keypoints_normalized: torch.Tensor | None = None  # (N, 2) pre-transform coords for the loma split


@dataclass
class MatchResult:
    """Matched pixel coordinates between a query and one reference image.

    idx_q/idx_db are the keypoint-table indices behind the pixel pairs — COLMAP's match
    format, consumed by geometry/verification.py. None when the matcher cannot provide
    stable indices (per-pair subpixel refinement moves the same keypoint to different
    coordinates in different pairs, so no single table row describes it).
    """

    query_px: np.ndarray  # (K, 2) float32 xy in query image
    ref_px: np.ndarray  # (K, 2) float32 xy in reference image
    idx_q: np.ndarray | None = None  # (K,) int64 into the query keypoint table
    idx_db: np.ndarray | None = None  # (K,) int64 into the reference keypoint table

    def __len__(self) -> int:
        return len(self.query_px)


def _empty_match() -> MatchResult:
    """Zero-length MatchResult (with empty index arrays — a zero match is indexable)."""
    z = np.zeros((0, 2), dtype=np.float32)
    zi = np.zeros(0, dtype=np.int64)
    return MatchResult(query_px=z, ref_px=z, idx_q=zi, idx_db=zi)


########################################
# VisMatch-backed matcher
########################################

# Models match() can serve from precomputed features. "xfeat": its own match stage IS
# descriptor mutual-NN. "loma" (added with its split): pair forward split into per-image /
# per-pair halves. Membership is licensed by GPU parity tests in the suite — extending this
# set requires a new passing parity test. Never silently substitute NN for a learned matcher.
FEATURE_MATCH_MODELS = {"xfeat", "loma"}

# Models whose base deps we override away — vismatch would crash at model load.
_VISMATCH_DEP_BLOCKLIST = {
    "ufm": "requires uniception==0.1.1; this project pins 0.1.7 for MapAnything",
    "edm": "requires lightning==2.3.3; this project resolves lightning>=2.6",
}

# Upstream model licenses that forbid commercial use. Verified 2026-08-17 against
# upstream LICENSE files / READMEs (GitHub); vismatch's own wrapper is BSD-3 but does
# not relicense the models it wraps. Names match vismatch.available_models exactly.
_VISMATCH_LICENSE_BLOCKLIST = {
    "superglue": "Magic Leap academic/non-profit research-only license",
    "superpoint-lightglue": "SuperPoint weights: Magic Leap research-only license",
    "superpoint-lightglue-subpx": "SuperPoint weights: Magic Leap research-only license",
    "superpoint-sphereglue": "SuperPoint weights: Magic Leap research-only license",
    "minima-superpoint-lightglue": "SuperPoint weights: Magic Leap research-only license",
    "lisrd": "default detector is SuperPoint (Magic Leap research-only weights)",
    "lisrd-superpoint": "SuperPoint weights: Magic Leap research-only license",
    "duster": "DUSt3R: CC BY-NC-SA 4.0 (non-commercial)",
    "master": "MASt3R: CC BY-NC-SA 4.0 (non-commercial)",
    "gim-lightglue": "GIM: repo MIT but README restricts model/content to research use only",
    "gim-dkm": "GIM: repo MIT but README restricts model/content to research use only",
    "r2d2": "R2D2: CC BY-NC-SA 3.0 (non-commercial)",
    "aspanformer": "ASpanFormer: Apple license, non-commercial purposes only",
    "ripe": "RIPE: Fraunhofer Software Copyright License for Academic Use",
    "silk": "SiLK: GPL-3.0 (copyleft) — legal review before commercial use",
    "liftfeat": "LiftFeat: no license published (all rights reserved)",
    "matchanything-eloftr": "MatchAnything: license unverified — audit before commercial use",
    "matchanything-roma": "MatchAnything: license unverified — audit before commercial use",
}


def _to_numpy(x) -> np.ndarray:
    """torch.Tensor (any device) or array-like -> float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


class LocalMatcher:
    """Stage-2 local matcher backed by the vismatch model zoo.

    One class for every vismatch model; the model name is data, not a subclass.
    extract() fills the zarr feature cache; match_images() is the pairwise path.
    """

    def __init__(self, model_name: str, device: str | None = None, probe: bool = True):
        # Refuse blocked models before touching vismatch (dep conflicts / licenses).
        for blocklist, kind in ((_VISMATCH_DEP_BLOCKLIST, "dependency"), (_VISMATCH_LICENSE_BLOCKLIST, "license")):
            if model_name in blocklist:
                raise ValueError(f"vismatch model '{model_name}' blocked ({kind}): {blocklist[model_name]}")
        # Heavy optional dep: vismatch pulls the full model zoo machinery.
        import vismatch

        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._matcher = vismatch.get_matcher(model_name, device=self._device)
        # Loma split: extract-once/match-from-features fast path (spec §2). One wrapper
        # class, one boolean — byte-parity with the plain forward is enforced by GPU suite tests.
        self._split_loma_forward = type(self._matcher).__name__ == "LoMaMatcher"
        # Set by _probe_index_stability(); None until probed.
        self.has_stable_indices: bool | None = None
        if probe:
            self._probe_index_stability()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 RGB -> (3,H,W) float [0,1] on device (vismatch input contract)."""
        t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        if t.max() > 1.5:  # uint8-scale input
            t = t / 255.0
        return t.to(self._device)

    @staticmethod
    def _check_pixel_frame(kpts: np.ndarray, hw: tuple[int, int], what: str) -> None:
        """Guard the 92f2e4a bug class: keypoints must be in the INPUT image's pixel frame."""
        if len(kpts) and (kpts.min() < -0.5 or kpts[:, 0].max() > hw[1] - 0.5 or kpts[:, 1].max() > hw[0] - 0.5):
            raise ValueError(
                f"vismatch '{what}' keypoints outside input pixel frame {hw}: "
                f"x range [{kpts[:, 0].min():.1f}, {kpts[:, 0].max():.1f}], "
                f"y range [{kpts[:, 1].min():.1f}, {kpts[:, 1].max():.1f}] — "
                "model likely returns coords at its internal resolution"
            )

    def extract(self, image: np.ndarray) -> LocalFeatures:
        """Extract keypoints+descriptors (vismatch runs a self-pair forward internally)."""
        hw = image.shape[:2]
        # Loma split: per-image half of LoMaMatcher._forward — detect_and_describe once,
        # replay the wrapper's coord chain (to_pixel_coords -> rescale_coords -> the -0.5
        # COLMAP offset) over the FULL table, and keep the pre-transform coords the learned
        # matcher consumes. Indexing a chained table equals chaining an indexed table
        # (elementwise ops), so match-time pixel coords stay byte-identical. Also skips the
        # wrapper's self-pair match stage. Sandbox entered explicitly — vismatch only wraps
        # __init__/_forward.
        if self._split_loma_forward:
            from vismatch.import_sandbox import ImportSandbox

            m = self._matcher
            mod = sys.modules[type(m).__module__]  # wrapper module: to_pixel_coords
            with ImportSandbox.get(type(m).__module__), torch.inference_mode():
                img, orig_shape = m.preprocess(self._to_tensor(image))
                H, W = img.shape[-2:]
                kpts, desc, _, _ = m.matcher.detect_and_describe(img, m.max_num_keypoints)
                px = m.rescale_coords(mod.to_pixel_coords(kpts[0], H, W), *orig_shape, H, W) - 0.5
            feats = LocalFeatures(
                keypoints=torch.from_numpy(_to_numpy(px)),
                descriptors=torch.from_numpy(_to_numpy(desc[0])),
                keypoints_normalized=torch.from_numpy(_to_numpy(kpts[0])),
            )
            self._check_pixel_frame(feats.keypoints.numpy(), hw, self._model_name)
            return feats
        with torch.inference_mode():
            out = self._matcher.extract(self._to_tensor(image))
        # vismatch may hand back numpy or on-device tensors depending on the model.
        kpts = _to_numpy(out["all_kpts0"])
        descs = _to_numpy(out["all_desc0"])
        self._check_pixel_frame(kpts, hw, self._model_name)
        return LocalFeatures(keypoints=torch.from_numpy(kpts), descriptors=torch.from_numpy(descs))

    def match(self, query: LocalFeatures, db: LocalFeatures) -> MatchResult:
        """Feature-level match over precomputed features (FEATURE_MATCH_MODELS only).

        Match rows ARE keypoint-table indices by construction — no _recover_indices.
        """
        if self._model_name not in FEATURE_MATCH_MODELS:
            raise NotImplementedError(
                f"LocalMatcher('{self._model_name}') has no feature-level matching "
                "(not in FEATURE_MATCH_MODELS — its match stage is not parity-proven "
                "on precomputed features). Use match_images()."
            )
        if len(query.descriptors) == 0 or len(db.descriptors) == 0:
            return _empty_match()
        if self._split_loma_forward:
            # Per-pair half of LoMaMatcher._forward: learned matcher on the stored
            # pre-transform coords + descriptors, wrapper's filter, native indices.
            # float32 in is fine — the matcher's autocast recasts at op boundaries
            # either way (parity-gated).
            if query.keypoints_normalized is None or db.keypoints_normalized is None:
                raise ValueError(
                    "loma feature-level match needs keypoints_normalized. If this cache "
                    "predates the payload, rebuild the localization DB "
                    "(build_localization_db(overwrite=True)); if a rebuild already ran, "
                    "the payload was not persisted — check save_index's all-or-none "
                    "keypoints_normalized write."
                )
            from vismatch.import_sandbox import ImportSandbox

            m = self._matcher
            mod = sys.modules[type(m).__module__]  # wrapper module: filter_matches
            k0 = query.keypoints_normalized.to(self._device).unsqueeze(0)
            k1 = db.keypoints_normalized.to(self._device).unsqueeze(0)
            d0 = query.descriptors.to(self._device).unsqueeze(0)
            d1 = db.descriptors.to(self._device).unsqueeze(0)
            with ImportSandbox.get(type(m).__module__), torch.inference_mode():
                scores = m.matcher(k0, k1, d0, d1)["scores"]
                m0, _, _, _ = mod.filter_matches(scores, m.matcher.cfg.filter_threshold)
                valid = m0[0] > -1
                if not bool(valid.any()):
                    return _empty_match()
                idx_q = torch.where(valid)[0].cpu().numpy().astype(np.int64)
                idx_db = m0[0][valid].cpu().numpy().astype(np.int64)
            return MatchResult(
                query_px=query.keypoints.numpy()[idx_q],
                ref_px=db.keypoints.numpy()[idx_db],
                idx_q=idx_q,
                idx_db=idx_db,
            )
        # L2 mutual-NN on unit vectors == cosine mutual-NN; match_mnn admits every mutual pair
        d0 = torch.nn.functional.normalize(query.descriptors.to(self._device), dim=1)
        d1 = torch.nn.functional.normalize(db.descriptors.to(self._device), dim=1)
        _, idxs = match_mnn(d0, d1)
        if len(idxs) == 0:
            return _empty_match()
        # idxs[:, 0] indexes query (desc1), idxs[:, 1] indexes db (desc2) — kornia match_mnn contract
        idx_q = idxs[:, 0].cpu().numpy().astype(np.int64)
        idx_db = idxs[:, 1].cpu().numpy().astype(np.int64)
        return MatchResult(
            query_px=query.keypoints.numpy()[idx_q],
            ref_px=db.keypoints.numpy()[idx_db],
            idx_q=idx_q,
            idx_db=idx_db,
        )

    @staticmethod
    def _recover_indices(matched: np.ndarray, table: np.ndarray) -> np.ndarray | None:
        """Map matched coordinates to exact rows of the keypoint table; None if any miss.

        Exact float equality on purpose: a coordinate a matcher refined off its table
        row must fail here, not silently map to the nearest row. Duplicate table rows:
        argmax picks the FIRST matching row, so two matches at the same coordinate map
        to the same index.
        """
        if len(table) == 0:
            return None
        # (K, N) exact row-equality; argmax over N gives the row per match
        eq = (matched[:, None, :] == table[None, :, :]).all(axis=2)
        if not eq.any(axis=1).all():
            return None
        return eq.argmax(axis=1).astype(np.int64)

    def match_images(self, query_image: np.ndarray, ref_image: np.ndarray) -> MatchResult:
        """Pairwise match two HxWx3 uint8 RGB images. Pre-RANSAC matches.

        vismatch's own RANSAC fits a homography — a planar-scene model that is the
        wrong geometric filter for 3D localization. We take matched_kpts (pre-RANSAC)
        and let PnP LO-RANSAC / epipolar verification do the filtering.
        """
        q_hw, r_hw = query_image.shape[:2], ref_image.shape[:2]
        with torch.inference_mode():
            out = self._matcher(self._to_tensor(query_image), self._to_tensor(ref_image))
        q_px = _to_numpy(out["matched_kpts0"])
        r_px = _to_numpy(out["matched_kpts1"])
        if len(q_px) == 0:
            return _empty_match()
        self._check_pixel_frame(q_px, q_hw, self._model_name)
        self._check_pixel_frame(r_px, r_hw, self._model_name)

        # Recover COLMAP keypoint-table row indices when the probe proved stability
        idx_q = idx_db = None
        if self.has_stable_indices:
            idx_q = self._recover_indices(q_px, _to_numpy(out["all_kpts0"]))
            idx_db = self._recover_indices(r_px, _to_numpy(out["all_kpts1"]))
            if idx_q is None or idx_db is None:
                logger.warning(
                    "LocalMatcher(%s): index recovery failed on a pair despite passing the "
                    "probe — treating this pair as index-less",
                    self._model_name,
                )
                idx_q = idx_db = None
        return MatchResult(query_px=q_px, ref_px=r_px, idx_q=idx_q, idx_db=idx_db)

    def _probe_index_stability(self) -> None:
        """One synthetic pair through the model: are matched kpts exact keypoint-table rows?

        Two conditions must BOTH hold for verify-compatibility:
          (a) within-call: matched_kpts are exact rows of all_kpts (no per-pair refinement);
          (b) cross-call: extract() keypoints reproduce (deterministic detection),
              so cache-time and match-time tables agree.
        """
        rng = np.random.default_rng(7)
        img = rng.uniform(0, 255, (256, 320, 3)).astype(np.uint8)
        img2 = np.roll(img, 8, axis=1)  # shifted copy — guarantees some matches for most models
        with torch.inference_mode():
            out = self._matcher(self._to_tensor(img), self._to_tensor(img2))
            ext = self._matcher.extract(self._to_tensor(img))
        # (a) within-call index recovery on both sides
        within = (
            len(out["matched_kpts0"]) > 0
            and self._recover_indices(_to_numpy(out["matched_kpts0"]), _to_numpy(out["all_kpts0"])) is not None
            and self._recover_indices(_to_numpy(out["matched_kpts1"]), _to_numpy(out["all_kpts1"])) is not None
        )
        # (b) cross-call detection determinism: extract table must equal pair-call table
        # (np.array_equal covers the shape mismatch case)
        cross = np.array_equal(_to_numpy(out["all_kpts0"]), _to_numpy(ext["all_kpts0"]))
        self.has_stable_indices = bool(within and cross)
        logger.info(
            "LocalMatcher(%s): index probe — within-call %s, cross-call %s -> stable_indices=%s",
            self._model_name,
            within,
            cross,
            self.has_stable_indices,
        )
