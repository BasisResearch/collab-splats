"""
Lift per-frame 2D masks to point groups that stay consistent across views.

- each mask becomes the set of points it covers in its frame (lift_features' visibility test)
- masks from different frames merge when they cover the same points both ways; strongest first
- cannot-link: a group never holds two masks of one frame unless they are near-duplicates
- optional features veto merges whose groups disagree and weight the fill of unclaimed points
- each point keeps its smallest group; a group's parent is the smallest group that contains it
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from collab_splats.semantics.lifting import frame_visibility
from collab_splats.utils.torch_utils import get_device

if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult

logger = logging.getLogger(__name__)


########################################################
########## Result type #################################
########################################################


@dataclass
class SegmentGroups:
    """
    View-consistent point groups.

    - labels: (P,) int32 smallest group per point; -1 only when no mask claimed any point
    - labeled: (P,) bool, True where masks claimed the point, False where the fill chose it
    - parent: (G,) int32 smallest containing group, -1 for a top-level group
    - n_frames: (G,) int32 distinct frames whose masks formed the group
    """

    labels: np.ndarray
    labeled: np.ndarray
    parent: np.ndarray
    n_frames: np.ndarray


########################################################
########## Helpers #####################################
########################################################


def _mask_point_sets(
    frame_masks: Callable[[int], np.ndarray | torch.Tensor],
    result: PointcloudResult,
    depth_tol: float,
    min_points: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Sparse mask-by-point membership, frame-by-point visibility and each mask's frame.

    - masks are nearest-resized to the model grid, as lift_features assumes for feature maps
    - a mask covering fewer than min_points visible points is dropped
    """
    device = get_device()
    N = result.extrinsics.shape[0]
    H, W = result.model_height, result.model_width
    P = result.points.shape[0]

    # Cameras and depth on one device in float32
    pts = torch.as_tensor(result.points, dtype=torch.float32, device=device)
    ext = torch.as_tensor(result.extrinsics, dtype=torch.float32, device=device)
    intr = torch.as_tensor(result.model_intrinsics, dtype=torch.float32, device=device)
    depth = torch.as_tensor(result.depth, dtype=torch.float32, device=device)

    mask_rows, mask_cols, mask_frame = [], [], []
    vis_rows, vis_cols = [], []

    for i in range(N):
        # Visible points of frame i and their rounded pixels
        visible, u, v = frame_visibility(
            pts, ext[i], intr[i], depth[i], depth_tol=depth_tol
        )
        idx = torch.nonzero(visible).squeeze(1)
        cols = torch.round(u[idx]).clamp(0, W - 1).long()
        rows = torch.round(v[idx]).clamp(0, H - 1).long()
        vis_rows.append(np.full(len(idx), i))
        vis_cols.append(idx.cpu().numpy())

        # Frame i's masks on the model grid
        masks = torch.as_tensor(frame_masks(i), device=device)

        if masks.shape[0] == 0 or len(idx) == 0:
            continue

        masks = F.interpolate(masks[:, None].float(), size=(H, W), mode="nearest")
        masks = masks[:, 0].bool()

        # Points each mask covers: the mask's value at every visible point's pixel
        covered = masks[:, rows, cols]
        counts = covered.sum(1).cpu().numpy()

        for m in np.flatnonzero(counts >= min_points):
            hit = idx[covered[m]].cpu().numpy()
            mask_rows.append(np.full(len(hit), len(mask_frame)))
            mask_cols.append(hit)
            mask_frame.append(i)

    # Assemble both sparse matrices
    M = len(mask_frame)
    A = sp.csr_matrix(
        (
            np.ones(sum(map(len, mask_cols)), dtype=np.float32),
            (
                np.concatenate(mask_rows or [np.zeros(0, int)]),
                np.concatenate(mask_cols or [np.zeros(0, int)]),
            ),
        ),
        shape=(M, P),
    )
    V = sp.csr_matrix(
        (
            np.ones(sum(map(len, vis_cols)), dtype=np.float32),
            (np.concatenate(vis_rows), np.concatenate(vis_cols)),
        ),
        shape=(N, P),
    )

    return A, V, np.asarray(mask_frame, dtype=int)


def _merge_masks(
    A: sp.csr_matrix,
    V: sp.csr_matrix,
    mask_frame: np.ndarray,
    mask_features: np.ndarray | None,
    tau: float,
    dup_iou: float,
    veto_cos: float,
    min_points: int,
) -> np.ndarray:
    """
    Group id per mask from greedy merging of mutual-overlap edges, strongest first.

    - overlap counts only co-visible points: a's points that b's frame sees, and vice versa
    - an edge needs both overlap ratios at least tau; its score is the smaller one
    - blocked: two same-frame masks with IoU below dup_iou, or mean-feature cosine below veto_cos
    """
    M = A.shape[0]

    # Pairwise shared points, and each mask's visible size in every frame
    inter = (A @ A.T).tocoo()
    covis = (A @ V.T).toarray()
    a, b, o = inter.row, inter.col, inter.data

    # Candidate pairs once each, with enough co-visible points both ways
    keep = a < b
    a, b, o = a[keep], b[keep], o[keep]
    size_a = covis[a, mask_frame[b]]
    size_b = covis[b, mask_frame[a]]
    keep = (size_a >= min_points) & (size_b >= min_points)
    a, b, o = a[keep], b[keep], o[keep]
    ra = o / size_a[keep]
    rb = o / size_b[keep]

    # Same-object edges, strongest first
    same = (ra >= tau) & (rb >= tau)
    a, b = a[same], b[same]
    score = np.minimum(ra, rb)[same]
    order = np.argsort(-score)

    # IoU of each same-frame mask pair, for the cannot-link test
    msize = np.asarray(A.sum(1)).ravel()
    sf = (mask_frame[inter.row] == mask_frame[inter.col]) & (inter.row != inter.col)
    iou = {
        (x, y): o_ / (msize[x] + msize[y] - o_)
        for x, y, o_ in zip(inter.row[sf], inter.col[sf], inter.data[sf])
    }

    # Union-find state: each root's masks by frame, and its feature sum
    parent = np.arange(M)
    members = {m: {mask_frame[m]: [m]} for m in range(M)}
    feature_sum = None if mask_features is None else mask_features.copy()

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    blocked_link = 0
    blocked_veto = 0

    for e in order:
        rx, ry = find(a[e]), find(b[e])

        if rx == ry:
            continue

        # Cannot-link: frames both groups hold must pair near-duplicate masks only
        X, Y = members[rx], members[ry]
        shared = X.keys() & Y.keys()

        if any(
            iou.get((min(x, y), max(x, y)), 0.0) < dup_iou
            for f in shared
            for x in X[f]
            for y in Y[f]
        ):
            blocked_link += 1
            continue

        # Feature veto: the two groups' mean features must agree
        if feature_sum is not None:
            fx, fy = feature_sum[rx], feature_sum[ry]
            cos = fx @ fy / (np.linalg.norm(fx) * np.linalg.norm(fy) + 1e-8)

            if cos < veto_cos:
                blocked_veto += 1
                continue

            feature_sum[ry] += fx

        # Merge rx into ry
        parent[rx] = ry

        for f, ms in X.items():
            Y.setdefault(f, []).extend(ms)

        del members[rx]

    logger.info(
        "segments: %d edges, %d blocked by cannot-link, %d by feature veto",
        len(order),
        blocked_link,
        blocked_veto,
    )

    # Contiguous group id per mask
    roots = np.array([find(x) for x in range(M)])
    _, gid = np.unique(roots, return_inverse=True)

    return gid


def _fill(
    points: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray | None,
    k: int,
) -> np.ndarray:
    """
    Labels for unclaimed points by a vote of their k nearest claimed points.

    - each vote weighs max(cosine(features), 0) when features are given, else 1
    - processed in chunks so the (chunk, G) vote table stays small
    """
    labels = labels.copy()
    claimed = np.flatnonzero(labels >= 0)
    unclaimed = np.flatnonzero(labels < 0)

    if len(claimed) == 0 or len(unclaimed) == 0:
        return labels

    # Nearest claimed points of every unclaimed one
    k = min(k, len(claimed))
    _, nn = cKDTree(points[claimed]).query(points[unclaimed], k=k, workers=-1)
    nn = nn.reshape(len(unclaimed), k)
    G = labels.max() + 1

    for start in range(0, len(unclaimed), 65536):
        rows = unclaimed[start : start + 65536]
        neighbors = claimed[nn[start : start + 65536]]
        votes = labels[neighbors]

        # Vote weights: feature agreement with each neighbor, or uniform
        if features is None:
            weights = np.ones(votes.shape, dtype=np.float32)
        else:
            weights = np.einsum("nd,nkd->nk", features[rows], features[neighbors])
            weights = np.maximum(weights, 0.0) + 1e-6

        # Weighted label histogram per point, then its argmax
        table = np.zeros((len(rows), G), dtype=np.float32)
        np.add.at(table, (np.arange(len(rows))[:, None], votes), weights)
        labels[rows] = table.argmax(1)

    return labels


########################################################
########## Public API ##################################
########################################################


def lift_segments(
    frame_masks: Callable[[int], np.ndarray | torch.Tensor],
    result: PointcloudResult,
    *,
    features: np.ndarray | None = None,
    tau: float = 0.6,
    dup_iou: float = 0.7,
    veto_cos: float = 0.5,
    min_frames: int = 8,
    min_group_frac: float = 0.001,
    min_points: int = 40,
    contain: float = 0.8,
    depth_tol: float = 0.05,
    k: int = 8,
) -> SegmentGroups:
    """
    Merge per-frame masks into point groups consistent across views.

    - 3D overlap decides merges; features only veto them and weight the fill
    - a point joins a group when at least half the group's frames that see it cover it, twice or more
    - groups seen in fewer than min_frames frames or holding under min_group_frac of points are dropped
    - parent: smallest group holding at least `contain` of the group's points and 1.5x its size

    Args:
        frame_masks: frame i's masks, (M_i, h, w) bool at any resolution; called once per frame.
        result: points, depth, extrinsics, model_intrinsics and model grid size.
        features: (P, D) L2-normalized point features, e.g. lifted Talk2DINO; None disables the
            veto and fills by distance alone.
        tau: minimum co-visible overlap, both ways, for two masks to be one object.
        dup_iou: IoU above which two masks of one frame may share a group.
        veto_cos: minimum cosine between two groups' mean features for them to merge.
        min_frames: distinct frames a group needs.
        min_group_frac: share of all points a group needs.
        min_points: visible points a mask, and a co-visible overlap, needs.
        contain: share of a group's points its parent must hold.
        depth_tol: relative depth tolerance of the visibility test.
        k: claimed neighbors voting on each unclaimed point.

    Returns:
        The groups; group ids are contiguous from 0.

    Raises:
        ValueError: when features do not have one row per point.
    """
    P = result.points.shape[0]

    if features is not None and features.shape[0] != P:
        raise ValueError(f"features has {features.shape[0]} rows for {P} points")

    # Mask point sets, then mask groups
    A, V, mask_frame = _mask_point_sets(frame_masks, result, depth_tol, min_points)
    mask_features = None if features is None else np.asarray(A @ features)
    gid = _merge_masks(
        A, V, mask_frame, mask_features, tau, dup_iou, veto_cos, min_points
    )
    G = int(gid.max()) + 1 if len(gid) else 0
    logger.info("segments: %d masks merged into %d raw groups", len(gid), G)

    # Keep groups seen in enough distinct frames
    group_frame = sp.csr_matrix(
        (np.ones(len(gid)), (gid, mask_frame)), shape=(G, V.shape[0])
    )
    group_frame.data[:] = 1
    n_frames = np.asarray(group_frame.sum(1)).ravel().astype(int)
    good = np.flatnonzero(n_frames >= min_frames)

    # Per kept group and point: covering masks, and the group's frames that see the point
    pick = np.isin(gid, good)
    remap = np.full(G, -1)
    remap[good] = np.arange(len(good))
    group_mask = sp.csr_matrix(
        (np.ones(pick.sum()), (remap[gid[pick]], np.flatnonzero(pick))),
        shape=(len(good), len(gid)),
    )
    hits = (group_mask @ A).tocsr()
    seen = (group_frame[good] @ V).multiply(hits > 0).tocsr()
    hits.sort_indices()
    seen.sort_indices()

    # Membership: covered in at least half the seeing frames, at least twice
    member = hits.copy()
    member.data = (hits.data / seen.data >= 0.5) & (hits.data >= 2)
    member.eliminate_zeros()
    member = member.astype(bool).tocsr()

    # Drop groups too small after the vote
    size = np.asarray(member.sum(1)).ravel()
    kept = np.flatnonzero((size > 0) & (size >= min_group_frac * P))
    member = member[kept]
    size = size[kept]
    n_frames = n_frames[good[kept]]

    # Leaf: smallest group holding the point; larger groups are written first
    labels = np.full(P, -1, dtype=np.int32)

    for g in np.argsort(-size, kind="stable"):
        labels[member.indices[member.indptr[g] : member.indptr[g + 1]]] = g

    # Parent: smallest group containing enough of this one's points and clearly larger
    shared = (member.astype(np.float32) @ member.T.astype(np.float32)).toarray()
    parent = np.full(len(kept), -1, dtype=np.int32)

    for g in range(len(kept)):
        cand = np.flatnonzero((shared[g] / size[g] >= contain) & (size > 1.5 * size[g]))

        if len(cand):
            parent[g] = cand[np.argmin(size[cand])]

    # Fill unclaimed points from their claimed neighbors
    labeled = labels >= 0
    labels = _fill(result.points, labels, features, k)
    logger.info(
        "segments: %d groups, %.1f%% of points claimed by masks",
        len(kept),
        100 * labeled.mean(),
    )

    return SegmentGroups(
        labels=labels.astype(np.int32),
        labeled=labeled,
        parent=parent,
        n_frames=n_frames.astype(np.int32),
    )


def nearest_labels(
    queries: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    *,
    max_spacing: float = 5.0,
) -> np.ndarray:
    """
    Each query takes its nearest point's group, or -1 when no point is close.

    - cutoff is scale-relative: max_spacing x the median nearest-neighbor spacing of `points`
    - used to carry point groups onto mesh vertices

    Args:
        queries: (Q, 3) positions to label, e.g. mesh vertices.
        points: (P, 3) labeled points.
        labels: (P,) group per point.
        max_spacing: cutoff in units of the median point spacing.

    Returns:
        (Q,) int32 group per query, -1 beyond the cutoff.
    """
    tree = cKDTree(points)

    # Median spacing from a point subsample; k=2 skips each point itself
    sample = points[:: max(1, len(points) // 100000)]
    spacing, _ = tree.query(sample, k=2, workers=-1)
    cutoff = max_spacing * float(np.median(spacing[:, 1]))

    dist, nn = tree.query(queries, k=1, workers=-1)
    out = labels[nn].astype(np.int32)
    out[dist > cutoff] = -1

    return out
