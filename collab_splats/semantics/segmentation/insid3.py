"""
INSID3 in-context segmentation backend ("insid3").

- training-free: frozen DINOv2 features, one reference image and mask per category
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from collab_splats.semantics.features.dino import DINOFeatureExtractor

from .base import BaseSegmentation


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
    # Unit-norm mean per cluster; every id 0..K-1 has members
    for k in range(K):
        protos.append(F.normalize(X[labels == k].mean(dim=0), p=2, dim=0).unsqueeze(0))
    return torch.cat(protos, dim=0)


########################################################
########## Mask and image helpers ######################
########################################################


def _downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Downsample a (H, W) bool mask to (h, w), keeping a tiny mask non-empty.

    - bilinear, then nearest, then the patch under the mask's centroid
    - each step covers a smaller mask than the last; an empty mask stays empty

    Args:
        mask: (H, W) bool.
        h: output rows.
        w: output columns.

    Returns:
        (h, w) bool.
    """
    m = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # Bilinear downsample — works for masks covering multiple patches
    down = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0.5
    if down.sum() == 0:
        # Nearest fallback for masks smaller than one patch
        down = F.interpolate(m, size=(h, w), mode="nearest")[0, 0] > 0.5
    if down.sum() == 0:
        # Centroid fallback for masks both interpolations miss; an empty mask stays empty
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
    Bilinear upsample of a bool mask.

    Args:
        mask: (h, w) bool.
        H: output rows.
        W: output columns.

    Returns:
        (H, W) bool, thresholded at 0.5.
    """
    return F.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0, 0] > 0.5


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    (C, H, W) float tensor in [0, 1] to a PIL image.

    Args:
        t: (C, H, W) float tensor in [0, 1].

    Returns:
        uint8 PIL image.
    """
    arr = (t.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


########################################################
########## Candidate localization ######################
########################################################


def _locate_candidates(
    tgt_feat_deb: torch.Tensor,
    ref_feat_deb: torch.Tensor,
    ref_mask_down: torch.Tensor,
    prototype: torch.Tensor,
    fallback_quantile: float = 0.9,
) -> torch.Tensor:
    """
    Target patches that match the reference both ways.

    - forward: positive cosine to the prototype (else the top (1 - fallback_quantile) share)
    - backward: the patch's nearest reference patch lies inside the reference mask

    Args:
        tgt_feat_deb: (D, Ht, Wt) debiased target features.
        ref_feat_deb: (D, Hr, Wr) debiased reference features.
        ref_mask_down: (Hr, Wr) bool reference mask at patch resolution.
        prototype: (D,) unit-norm reference prototype.
        fallback_quantile: share of patches below the fallback cut when no patch is positive.

    Returns:
        (Ht, Wt) bool candidate mask.
    """
    D, Ht, Wt = tgt_feat_deb.shape
    _, Hr, Wr = ref_feat_deb.shape

    # Forward: cosine similarity of each target patch to the ref prototype
    sim_fwd = torch.einsum("dhw,d->hw", tgt_feat_deb, prototype)  # (Ht, Wt)
    forward_mask = sim_fwd > 0
    if forward_mask.sum() == 0:
        # Fallback: top (1 - fallback_quantile) share by prototype similarity
        thresh = float(torch.quantile(sim_fwd.float(), fallback_quantile))
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
    candidate_mask: torch.Tensor,
    tgt_feat: torch.Tensor,
    tgt_feat_deb: torch.Tensor,
    prototype: torch.Tensor,
    cluster_labels: torch.Tensor,
    K: int,
    merge_threshold: float,
) -> torch.Tensor:
    """
    Pick the seed cluster, then merge clusters whose combined score clears the threshold.

    - seed: the candidate-overlapping cluster most similar to the reference prototype
    - score: cross-image similarity x similarity to the seed x candidate overlap share

    Args:
        candidate_mask: (H_p, W_p) bool from `_locate_candidates`.
        tgt_feat: (D, H_p, W_p) target features, not debiased.
        tgt_feat_deb: (D, H_p, W_p) debiased target features.
        prototype: (D,) debiased reference prototype.
        cluster_labels: (H_p, W_p) cluster id per patch.
        K: number of clusters.
        merge_threshold: minimum combined score to join the mask.

    Returns:
        (H_p, W_p) bool; empty when candidate_mask is empty or no cluster clears merge_threshold.
    """
    D, H, W = tgt_feat.shape
    if candidate_mask.sum() == 0:
        return torch.zeros(H, W, dtype=torch.bool, device=candidate_mask.device)

    matched_ids, n_pixels = cluster_labels[candidate_mask].unique(return_counts=True)

    # Build per-cluster area weights from candidate pixel overlap
    all_unique, all_counts = cluster_labels.unique(return_counts=True)
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
        cross_sim[k] = fg_sim[cluster_labels == k].mean()

    # Combined score; the seed's area weight is 1, but it too must clear merge_threshold
    combined = cross_sim * intra_sim
    area_weights[seed_id] = 1.0
    combined = combined * area_weights

    return combined[cluster_labels] > merge_threshold


########################################################
########## INSID3Segmentation ##########################
########################################################


@BaseSegmentation.register("insid3")
class INSID3Segmentation(BaseSegmentation):
    """
    Training-free in-context segmentation using frozen DINOv2 features.

    - call set_context(ref_image, ref_mask) once per semantic category, then segment()
      for each target frame
    - context is cached: reference features are extracted only once

    Args:
        svd_components: SVD rank for positional debiasing. 500 matches INSID3.
        tau: agglomerative clustering similarity threshold.
        merge_threshold: minimum combined score for a cluster to be included.
        fallback_quantile: forwarded to candidate localization; see _locate_candidates.
        device: torch device; None picks one with get_device.
    """

    def __init__(
        self,
        svd_components: int = 500,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        fallback_quantile: float = 0.9,
        device: str | None = None,
    ) -> None:
        self._extractor = DINOFeatureExtractor(svd_components=svd_components, device=device)
        self._tau = tau
        self._merge_threshold = merge_threshold
        self._fallback_quantile = fallback_quantile
        self._prototype: torch.Tensor | None = None
        self._ref_feat_deb: torch.Tensor | None = None
        self._ref_mask_down: torch.Tensor | None = None

    def set_context(
        self,
        ref_image: "Image.Image | torch.Tensor",
        ref_mask: "np.ndarray | torch.Tensor",
    ) -> None:
        """
        Cache the reference features and prototype; call before segment().

        - a mask with no True pixel clears any earlier context, then raises

        Args:
            ref_image: reference image as PIL Image or (C, H, W) float tensor in [0, 1].
            ref_mask: binary context mask as numpy bool array or torch bool tensor, (H, W).

        Raises:
            ValueError: when ref_mask has no True pixel.
        """
        if isinstance(ref_image, torch.Tensor):
            ref_image = _tensor_to_pil(ref_image)

        # Mask to bool tensor; an empty one has no prototype, so reject it before the backbone runs
        if isinstance(ref_mask, np.ndarray):
            ref_mask = torch.from_numpy(ref_mask)
        ref_mask = ref_mask.bool()
        if not ref_mask.any():
            self.clear_context()
            raise ValueError("set_context: ref_mask has no True pixel")

        # Extract and debias ref features
        [feat] = self._extractor.forward([ref_image])  # (D, H_p, W_p), unit-norm per patch
        [feat_deb] = self._extractor.debias([feat])  # (D, H_p, W_p)
        H_p, W_p = feat_deb.shape[1:]

        # Downsample mask to patch resolution; the small-mask fallback keeps it non-empty
        mask_down = _downsample_mask(ref_mask.to(feat_deb.device), H_p, W_p)

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
        Segment an image against the cached context.

        Args:
            image: the frame to segment, resized by the extractor.

        Returns:
            (pred_mask, metadata) — pred_mask (H, W) bool at the input resolution; metadata
            carries candidate_mask (H_p, W_p), cluster_labels (H_p, W_p) and n_clusters.

        Raises:
            RuntimeError: when no context is set — call set_context() first.
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
        [feat] = self._extractor.forward([image])   # (D, H_p, W_p), unit-norm per patch
        D, H_p, W_p = feat.shape
        [feat_deb] = self._extractor.debias([feat])   # (D, H_p, W_p)

        # Candidate localization: forward similarity + backward NN matching
        candidate_mask = _locate_candidates(
            feat_deb, self._ref_feat_deb, self._ref_mask_down, self._prototype,
            fallback_quantile=self._fallback_quantile,
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
        feat_flat = feat.reshape(D, -1).T   # (H_p*W_p, D)
        cluster_labels = _agglomerative_clustering(feat_flat, self._tau)  # (H_p*W_p,)
        K = int(cluster_labels.max().item()) + 1
        cluster_labels_2d = cluster_labels.reshape(H_p, W_p)

        # Seed cluster selection and cluster aggregation
        pred_mask = _seed_and_aggregate(
            candidate_mask, feat, feat_deb, self._prototype,
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
        One-shot segmentation: set_context, segment, clear_context.

        - context is cleared on return and when segment() raises

        Args:
            image: target image to segment.
            ref_image: reference image containing the context category.
            ref_mask: binary mask on ref_image indicating the context region.

        Returns:
            (pred_mask (H, W) bool, metadata dict) — same as segment().

        Raises:
            ValueError: when ref_mask has no True pixel.
        """
        # Context lives for this call only, even when segment() raises
        self.set_context(ref_image, ref_mask)
        try:
            return self.segment(image)
        finally:
            self.clear_context()
