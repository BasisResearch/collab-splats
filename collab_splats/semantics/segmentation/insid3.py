"""
INSID3 in-context segmentation backend.

Provides:
  INSID3Segmentation — training-free in-context segmentation via frozen DINOv2 features
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from collab_splats.semantics.features.dino import DINOFeatureExtractor

from .base import BaseSegmentation

logger = logging.getLogger(__name__)


########################################################
########## Clustering utilities ########################
########################################################


def _agglomerative_clustering(X: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Partition N patches into clusters via cosine-distance agglomerative clustering.

    Args:
        X: (N, D) L2-normalized patch features.
        tau: similarity threshold — clusters are merged above this cosine similarity.

    Returns:
        (N,) long tensor of integer cluster labels on the same device as X.
    """
    # Compute cosine similarity matrix; clamp for numerical stability
    S = (X @ X.T).clamp(-1, 1)
    D = (1.0 - S).cpu().numpy()
    ac = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=float(1.0 - tau),
    )
    labels = ac.fit_predict(D)
    return torch.from_numpy(labels).long().to(X.device)


def _cluster_prototypes(X: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """
    Compute an L2-normalized prototype (mean) for each cluster.

    Args:
        X: (N, D) patch features.
        labels: (N,) integer cluster assignments.
        K: number of clusters.

    Returns:
        (K, D) L2-normalized prototypes.
    """
    protos = []
    # Compute L2-normalized mean per cluster; fall back to zero vector for empty clusters
    for k in range(K):
        idx = labels == k
        mu = X[idx].mean(dim=0) if idx.any() else torch.zeros(X.shape[1], device=X.device)
        protos.append(F.normalize(mu, p=2, dim=0).unsqueeze(0))
    return torch.cat(protos, dim=0)


########################################################
########## Mask and image helpers ######################
########################################################


def _downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Downsample (H, W) bool mask to (h, w) with fallback for tiny masks.

    Tries bilinear → nearest → single center pixel to ensure non-empty output.
    """
    m = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # Bilinear downsample — works for masks covering multiple patches
    down = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0.5
    if down.sum() == 0:
        # Nearest fallback for masks smaller than one patch
        down = F.interpolate(m, size=(h, w), mode="nearest")[0, 0] > 0.5
    if down.sum() == 0:
        # Center-pixel fallback for single-pixel masks; skip if mask is entirely empty
        if mask.any():
            center = torch.argwhere(mask).float().mean(dim=0)
            scale = torch.tensor([h / mask.shape[0], w / mask.shape[1]], device=mask.device)
            cy, cx = (center * scale).long()
            cy = cy.clamp(0, h - 1)
            cx = cx.clamp(0, w - 1)
            down = torch.zeros(h, w, dtype=torch.bool, device=mask.device)
            down[cy, cx] = True
        else:
            down = torch.zeros(h, w, dtype=torch.bool, device=mask.device)
    return down


def _upsample_mask(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Bilinear upsample (h, w) bool mask to (H, W).
    """
    return F.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0, 0] > 0.5


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    Convert (C, H, W) float tensor in [0, 1] to PIL Image.
    """
    arr = (t.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


########################################################
########## Candidate localization ######################
########################################################


def _locate_candidates(
    tgt_feat_deb: torch.Tensor,   # (D, Ht, Wt) debiased target features
    ref_feat_deb: torch.Tensor,   # (D, Hr, Wr) debiased ref features
    ref_mask_down: torch.Tensor,  # (Hr, Wr) bool — ref mask at patch resolution
    prototype: torch.Tensor,      # (D,) L2-normalized ref prototype
) -> torch.Tensor:                # (Ht, Wt) bool candidate mask
    """
    Forward+backward candidate localization.

    Forward: target patches with positive cosine sim to prototype.
    Backward: for each target patch, its nearest ref patch must be inside ref_mask_down.
    Returns intersection of both masks.
    """
    D, Ht, Wt = tgt_feat_deb.shape
    _, Hr, Wr = ref_feat_deb.shape

    # Forward: cosine similarity of each target patch to the ref prototype
    sim_fwd = torch.einsum("dhw,d->hw", tgt_feat_deb, prototype)  # (Ht, Wt)
    forward_mask = sim_fwd > 0
    if forward_mask.sum() == 0:
        # Fallback: top-10% patches by prototype similarity
        thresh = float(torch.quantile(sim_fwd.float(), 0.9))
        forward_mask = sim_fwd > thresh

    # Backward: for each target patch, find its nearest ref patch; check if inside mask
    tgt_flat = tgt_feat_deb.reshape(D, -1).T          # (Ht*Wt, D)
    ref_flat = ref_feat_deb.reshape(D, -1).T           # (Hr*Wr, D)
    sim_t_to_r = tgt_flat @ ref_flat.T                 # (Ht*Wt, Hr*Wr)
    best_idx = sim_t_to_r.argmax(dim=1)                # (Ht*Wt,)
    rows = (best_idx // Wr).clamp(0, Hr - 1)
    cols = (best_idx % Wr).clamp(0, Wr - 1)
    backward_mask = ref_mask_down[rows, cols].reshape(Ht, Wt)

    return forward_mask & backward_mask


########################################################
########## Seed selection and cluster aggregation ######
########################################################


def _seed_and_aggregate(
    candidate_mask: torch.Tensor,   # (H_p, W_p) bool
    tgt_feat: torch.Tensor,         # (D, H_p, W_p) raw (non-debiased) target features
    tgt_feat_deb: torch.Tensor,     # (D, H_p, W_p) debiased target features
    prototype: torch.Tensor,        # (D,) debiased ref prototype
    cluster_labels: torch.Tensor,   # (H_p, W_p) long
    K: int,
    merge_threshold: float,
) -> torch.Tensor:                  # (H_p, W_p) bool
    """
    Select seed cluster and aggregate remaining clusters by combined similarity score.

    Returns empty mask if no clusters overlap the candidate region.
    """
    D, H, W = tgt_feat.shape
    matched_mask = candidate_mask & (cluster_labels >= 0)
    if matched_mask.sum() == 0:
        return torch.zeros(H, W, dtype=torch.bool, device=candidate_mask.device)

    matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)

    # Build per-cluster area weights from candidate pixel overlap
    all_unique, all_counts = cluster_labels[cluster_labels >= 0].unique(return_counts=True)
    all_areas = torch.zeros(K, device=cluster_labels.device)
    all_areas[all_unique] = all_counts.float()
    area_weights = torch.zeros(K, device=cluster_labels.device)
    area_weights[matched_ids] = n_pixels.float() / all_areas[matched_ids].clamp(min=1)

    # Cluster prototypes in debiased space for cross-image similarity
    tgt_deb_flat = tgt_feat_deb.reshape(D, -1).T   # (N, D)
    deb_protos = _cluster_prototypes(tgt_deb_flat, cluster_labels.view(-1), K)  # (K, D)

    # Seed: matched cluster with highest cross-image similarity to ref prototype
    cross_sim_matched = deb_protos[matched_ids] @ prototype  # (n_matched,)
    seed_idx = int(torch.argmax(cross_sim_matched).item())
    seed_id = int(matched_ids[seed_idx].item())

    # Raw cluster prototypes for intra-image similarity to seed
    tgt_flat = tgt_feat.reshape(D, -1).T           # (N, D)
    raw_protos = _cluster_prototypes(tgt_flat, cluster_labels.view(-1), K)  # (K, D)
    intra_sim = raw_protos @ raw_protos[seed_id]   # (K,)

    # Per-patch cross-image similarity map
    fg_sim = torch.einsum("dhw,d->hw", tgt_feat_deb, prototype)  # (H_p, W_p)
    cross_sim = torch.zeros(K, device=fg_sim.device)
    for k in range(K):
        idx = cluster_labels == k
        cross_sim[k] = fg_sim[idx].mean() if idx.any() else 0.0

    # Combined score; seed cluster always included (area_weight = 1.0)
    combined = cross_sim * intra_sim
    area_weights[seed_id] = 1.0
    combined = combined * area_weights

    final_mask = torch.zeros(H, W, dtype=torch.bool, device=cluster_labels.device)
    valid = cluster_labels >= 0
    final_mask[valid] = combined[cluster_labels[valid]] > merge_threshold
    return final_mask


########################################################
########## INSID3Segmentation ##########################
########################################################


@BaseSegmentation.register("insid3")
class INSID3Segmentation(BaseSegmentation):
    """
    Training-free in-context segmentation using frozen DINOv2 features.

    Call set_context(ref_image, ref_mask) once per semantic category, then segment()
    for each target frame. Context is cached — ref features are extracted only once.

    Args:
        svd_components: SVD rank for positional debiasing (default 500, matches INSID3).
        tau: Agglomerative clustering similarity threshold (default 0.6).
        merge_threshold: Minimum combined score to include a cluster (default 0.2).
        device: Torch device string.
    """

    def __init__(
        self,
        svd_components: int = 500,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        device: str = "cuda",
    ) -> None:
        self._extractor = DINOFeatureExtractor(svd_components=svd_components, device=device)
        self._tau = tau
        self._merge_threshold = merge_threshold
        self._prototype: torch.Tensor | None = None
        self._ref_feat_deb: torch.Tensor | None = None
        self._ref_mask_down: torch.Tensor | None = None

    def set_context(
        self,
        ref_image: "Image.Image | torch.Tensor",
        ref_mask: "np.ndarray | torch.Tensor",
    ) -> None:
        """
        Extract and cache ref features + prototype. Must call before segment().

        Args:
            ref_image: Reference image as PIL Image or (C, H, W) float tensor in [0, 1].
            ref_mask: Binary context mask as numpy bool array or torch bool tensor, (H, W).
        """
        if isinstance(ref_image, torch.Tensor):
            ref_image = _tensor_to_pil(ref_image)

        # Extract and debias ref features
        [feat] = self._extractor.forward([ref_image])  # (D, H_p, W_p)
        feat_norm = F.normalize(feat, p=2, dim=0)
        [feat_deb] = self._extractor.debias([feat_norm])  # (D, H_p, W_p)

        H_p, W_p = feat_deb.shape[1:]
        device = feat_deb.device

        # Normalize mask to bool tensor on same device
        if isinstance(ref_mask, np.ndarray):
            ref_mask = torch.from_numpy(ref_mask)
        ref_mask = ref_mask.bool().to(device)

        # Downsample mask to patch resolution with small-mask fallback
        mask_down = _downsample_mask(ref_mask, H_p, W_p)

        # Guard: empty mask produces NaN prototype — log warning and abort
        if not mask_down.any():
            logger.warning(
                "set_context: ref_mask downsamples to empty patch grid; context not set. "
                "Ensure ref_mask covers at least one patch-sized region."
            )
            return

        # Prototype: L2-normalized mean of debiased features inside the masked region
        fg = feat_deb[:, mask_down]              # (D, N_fg)
        prototype = F.normalize(fg.mean(dim=1), p=2, dim=0)  # (D,)

        self._prototype = prototype
        self._ref_feat_deb = feat_deb
        self._ref_mask_down = mask_down

    def clear_context(self) -> None:
        """
        Reset cached context state.
        """
        self._prototype = None
        self._ref_feat_deb = None
        self._ref_mask_down = None

    def segment(self, image: "Image.Image | torch.Tensor") -> tuple[torch.Tensor, dict]:
        """
        Segment image using cached context.

        Raises RuntimeError if no context is set — call set_context() first.
        Returns (pred_mask (H, W) bool, metadata dict).
        Metadata keys: candidate_mask (H_p, W_p), cluster_labels (H_p, W_p), n_clusters (int).
        """
        if self._prototype is None or self._ref_feat_deb is None or self._ref_mask_down is None:
            raise RuntimeError(
                "No context set. Call set_context(ref_image, ref_mask) before segment()."
            )

        # Capture original size before the extractor resizes the image
        if isinstance(image, torch.Tensor):
            orig_H, orig_W = int(image.shape[-2]), int(image.shape[-1])
            image = _tensor_to_pil(image)
        else:
            orig_H, orig_W = image.height, image.width

        # Extract and debias target features
        [feat] = self._extractor.forward([image])   # (D, H_p, W_p)
        D, H_p, W_p = feat.shape
        feat_norm = F.normalize(feat, p=2, dim=0)
        [feat_deb] = self._extractor.debias([feat_norm])   # (D, H_p, W_p)

        # Candidate localization: forward similarity + backward NN matching
        candidate_mask = _locate_candidates(
            feat_deb, self._ref_feat_deb, self._ref_mask_down, self._prototype
        )  # (H_p, W_p) bool

        # Early exit: no candidates — return empty mask
        if candidate_mask.sum() == 0:
            empty = torch.zeros(H_p, W_p, dtype=torch.bool, device=feat.device)
            return _upsample_mask(empty, orig_H, orig_W), {
                "candidate_mask": candidate_mask,
                "cluster_labels": torch.full((H_p, W_p), -1, dtype=torch.long),
                "n_clusters": 0,
            }

        # Agglomerative clustering on raw (non-debiased) L2-normalized target features
        feat_flat = feat_norm.reshape(D, -1).T   # (H_p*W_p, D)
        cluster_labels = _agglomerative_clustering(feat_flat, self._tau)  # (H_p*W_p,)
        K = int(cluster_labels.max().item()) + 1
        cluster_labels_2d = cluster_labels.reshape(H_p, W_p)

        # Seed cluster selection and cluster aggregation
        pred_mask = _seed_and_aggregate(
            candidate_mask, feat_norm, feat_deb, self._prototype,
            cluster_labels_2d, K, self._merge_threshold,
        )  # (H_p, W_p) bool

        pred_mask_up = _upsample_mask(pred_mask, orig_H, orig_W)
        return pred_mask_up, {
            "candidate_mask": candidate_mask,
            "cluster_labels": cluster_labels_2d,
            "n_clusters": K,
        }

    def segment_with_mask(
        self,
        image: "Image.Image | torch.Tensor",
        ref_image: "Image.Image | torch.Tensor",
        ref_mask: "np.ndarray | torch.Tensor",
    ) -> tuple[torch.Tensor, dict]:
        """
        One-shot in-context segmentation: set_context → segment → clear_context.

        Args:
            image: Target image to segment.
            ref_image: Reference image containing the context category.
            ref_mask: Binary mask on ref_image indicating the context region.

        Returns:
            (pred_mask (H, W) bool, metadata dict) — same as segment().
        """
        # Set context, segment target, then clear so state doesn't persist
        self.set_context(ref_image, ref_mask)
        result = self.segment(image)
        self.clear_context()
        return result
