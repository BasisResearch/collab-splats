#!/usr/bin/env python
"""GT-verified loop precision/recall for LC parity runs (post-hoc from artifacts).

    /opt/venv/reconstruction/bin/python evals/runners/lc_loop_pr.py \
        evals/baselines/lc_parity_d5_postfix/7s_chess/ours_vggt_spark

Inputs per run dir (an ``ours_<backbone>`` arm), no pipeline re-runs:
``lc_decisions_lc.json`` (all post-NMS loop candidates with accepted /
reject_reason), ``gt.tum`` (GT keyframe poses; rows are 1:1 with the SLAM
keyframe order the run consumed — eval_gt.py writes ``dataset.gt_poses`` after
the ``--keyframe_list`` filter), and ``metrics.json``'s ``_config.submap_size``.
Decision frame indices are submap-local; the global keyframe index is
``submap_id * submap_size + frame_idx`` (wrappers.py windows stride by
submap_size with a +1 overlap frame, so the boundary frame carries local index
submap_size and is attributed to the next submap by ``index // submap_size``).

GT covisibility label (scale-aware heuristic mirroring the clean-negative
threshold-calibration methodology): a keyframe pair is POSITIVE when GT camera
centers are within POS_DIST_FRAC x scene_diameter AND viewing directions agree
within POS_VIEW_DEG; NEGATIVE when centers are farther apart than
NEG_DIST_FRAC x scene_diameter OR viewing directions differ by more than
NEG_VIEW_DEG; anything in between is AMBIGUOUS and EXCLUDED from both metrics
(counts reported). Scene diameter = max pairwise GT camera-center distance.

Precision — of ACCEPTED loops, the fraction whose (query, detected) keyframe
pair is GT-positive: ``accepted_positive / (accepted_positive +
accepted_negative)``. Ambiguous accepted loops leave both numerator and
denominator. None when no accepted loop gets a definite label.

Recall — over GT loop opportunities at SUBMAP-PAIR granularity:

1. Eligible frame pair (i < j): GT-positive AND in different submaps AND
   submap gap STRICTLY greater than min_submap_gap — the pipeline queries
   ``submaps[:len(submaps) - min_submap_gap]`` before appending the current
   submap (wrappers.py), so the nearest reachable partner is
   min_submap_gap + 1 back — AND global frame distance
   ``j - i >= nms_frame_distance``.
2. Opportunity = a distinct submap pair (s_i, s_j) containing at least one
   eligible frame pair; redundant frame pairs between the same two submaps
   collapse to one opportunity (else recall is dominated by them).
3. Per query submap s_j, opportunities are capped at max_loops_per_submap (the
   pipeline's per-submap loop budget): ``den_j = min(|partners(s_j)|, cap)``;
   the matched numerator is capped the same way:
   ``num_j = min(|applied GT-positive partners(s_j) ∩ partners(s_j)|, den_j)``.
4. ``recall = sum_j num_j / sum_j den_j``; None when the denominator is zero.

"Applied" = decisions with ``accepted == true`` (in the parity runs
loops_applied equals the accepted count).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


########################################
###### Heuristic thresholds/config #####
########################################

# Scale-aware GT covisibility thresholds. These are heuristics (mirroring the
# clean-negative calibration methodology), not ground truth of covisibility —
# hence the ambiguous band and the CLI overrides below.
POS_DIST_FRAC = 0.35  # positive: center distance <= 0.35 x scene diameter ...
POS_VIEW_DEG = 75.0  # ... AND view directions within 75°
NEG_DIST_FRAC = 0.50  # negative: distance > 0.5 x diameter ...
NEG_VIEW_DEG = 90.0  # ... OR view angle > 90°

# Pipeline defaults mirrored from LoopClosureConfig (collab_splats/pointcloud/
# loop_closure/closure.py); metrics.json's _config does not record them.
MIN_SUBMAP_GAP = 1
NMS_FRAME_DISTANCE = 25
MAX_LOOPS_PER_SUBMAP = 5


########################################
########## GT pose ingestion ###########
########################################


def _load_gt(gt_tum: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse gt.tum into (centers (N,3), view_dirs (N,3)); rows are keyframe order."""
    rows = []
    for line in gt_tum.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(v) for v in line.split()])
    data = np.asarray(rows, dtype=np.float64)
    centers = data[:, 1:4]
    # View direction = camera z-axis in world = third column of R(qx,qy,qz,qw).
    x, y, z, w = data[:, 4], data[:, 5], data[:, 6], data[:, 7]
    dirs = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return centers, dirs


def scene_diameter(centers: np.ndarray) -> float:
    """Max pairwise GT camera-center distance (metric scale reference)."""
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    return float(dists.max())


########################################
######## Covisibility labeling #########
########################################


def label_pair(
    center_a: np.ndarray,
    dir_a: np.ndarray,
    center_b: np.ndarray,
    dir_b: np.ndarray,
    diameter: float,
    *,
    pos_dist_frac: float = POS_DIST_FRAC,
    pos_view_deg: float = POS_VIEW_DEG,
    neg_dist_frac: float = NEG_DIST_FRAC,
    neg_view_deg: float = NEG_VIEW_DEG,
) -> int:
    """Label one GT pose pair: +1 covisible, -1 not covisible, 0 ambiguous."""
    dist = float(np.linalg.norm(center_a - center_b))
    cos_ang = float(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))
    # Positive and negative bands are disjoint by construction (0.35 < 0.5, 75 < 90).
    if dist > neg_dist_frac * diameter or cos_ang < np.cos(np.radians(neg_view_deg)):
        return -1
    if dist <= pos_dist_frac * diameter and cos_ang >= np.cos(np.radians(pos_view_deg)):
        return 1
    return 0


def scene_covisibility(
    centers: np.ndarray,
    view_dirs: np.ndarray,
    *,
    pos_dist_frac: float = POS_DIST_FRAC,
    pos_view_deg: float = POS_VIEW_DEG,
    neg_dist_frac: float = NEG_DIST_FRAC,
    neg_view_deg: float = NEG_VIEW_DEG,
) -> np.ndarray:
    """(N, N) int8 covisibility label matrix over all keyframe pairs (+1/-1/0)."""
    diameter = scene_diameter(centers)
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    cos_ang = np.clip(view_dirs @ view_dirs.T, -1.0, 1.0)
    # Vectorized version of label_pair's disjoint bands.
    neg = (dists > neg_dist_frac * diameter) | (cos_ang < np.cos(np.radians(neg_view_deg)))
    pos = (dists <= pos_dist_frac * diameter) & (cos_ang >= np.cos(np.radians(pos_view_deg)))
    labels = np.zeros(dists.shape, dtype=np.int8)
    labels[neg] = -1
    labels[pos] = 1
    return labels


########################################
######## Precision/recall metric #######
########################################


def loop_precision_recall(
    run_dir: Path,
    *,
    pos_dist_frac: float = POS_DIST_FRAC,
    pos_view_deg: float = POS_VIEW_DEG,
    neg_dist_frac: float = NEG_DIST_FRAC,
    neg_view_deg: float = NEG_VIEW_DEG,
    min_submap_gap: int = MIN_SUBMAP_GAP,
    nms_frame_distance: int = NMS_FRAME_DISTANCE,
    max_loops_per_submap: int = MAX_LOOPS_PER_SUBMAP,
) -> dict:
    """Compute GT-verified loop precision/recall for one arm dir (see module docstring)."""
    run_dir = Path(run_dir)

    # Load decisions, GT poses, and the run's submap_size for frame-index mapping.
    decisions = json.loads((run_dir / "lc_decisions_lc.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    submap_size = metrics.get("_config", {}).get("submap_size")
    if not submap_size:
        raise ValueError(f"{run_dir}/metrics.json has no _config.submap_size")
    centers, dirs = _load_gt(run_dir / "gt.tum")
    n = len(centers)

    # Full covisibility label matrix + scale reference.
    diameter = scene_diameter(centers)
    thresholds = dict(
        pos_dist_frac=pos_dist_frac,
        pos_view_deg=pos_view_deg,
        neg_dist_frac=neg_dist_frac,
        neg_view_deg=neg_view_deg,
    )
    labels = scene_covisibility(centers, dirs, **thresholds)

    # Map accepted loops to global keyframe pairs and label them (precision).
    accepted = [d for d in decisions if d["accepted"]]
    counts = {1: 0, -1: 0, 0: 0}
    applied_positive_pairs: set[tuple[int, int]] = set()
    for d in accepted:
        gq = d["query_submap"] * submap_size + d["query_frame"]
        gd = d["detected_submap"] * submap_size + d["detected_frame"]
        if gq >= n or gd >= n:
            raise ValueError(
                f"decision maps out of range: frames ({gq}, {gd}) vs {n} GT rows "
                f"(submap_size={submap_size}) — frame-index mapping is wrong"
            )
        lab = int(labels[gq, gd])
        counts[lab] += 1
        if lab == 1:
            applied_positive_pairs.add((d["detected_submap"], d["query_submap"]))
    labeled = counts[1] + counts[-1]
    precision = counts[1] / labeled if labeled else None

    # Enumerate eligible GT-positive frame pairs and collapse to submap-pair partners.
    iu, ju = np.triu_indices(n, k=1)
    si, sj = iu // submap_size, ju // submap_size
    eligible = (labels[iu, ju] == 1) & (sj - si > min_submap_gap) & (ju - iu >= nms_frame_distance)
    partners: dict[int, set[int]] = {}
    for a, b in zip(si[eligible], sj[eligible]):
        partners.setdefault(int(b), set()).add(int(a))

    # Per query submap, cap denominator and matched numerator at the loop budget.
    den = num = 0
    for query_submap, part in partners.items():
        d_j = min(len(part), max_loops_per_submap)
        hits = {p for (p, q) in applied_positive_pairs if q == query_submap} & part
        den += d_j
        num += min(len(hits), d_j)
    recall = num / den if den else None

    return {
        "precision": precision,
        "recall": recall,
        "accepted": len(accepted),
        "accepted_positive": counts[1],
        "accepted_negative": counts[-1],
        "accepted_ambiguous": counts[0],
        "opportunities": den,
        "opportunities_uncapped": sum(len(p) for p in partners.values()),
        "recall_numerator": num,
        "applied_positive_pairs": len(applied_positive_pairs),
        "scene_diameter": diameter,
        "n_frames": n,
        "submap_size": submap_size,
        "params": {
            **thresholds,
            "min_submap_gap": min_submap_gap,
            "nms_frame_distance": nms_frame_distance,
            "max_loops_per_submap": max_loops_per_submap,
        },
    }


########################################
################# CLI ##################
########################################


def main() -> int:
    """CLI: compute loop P/R for one arm dir and write loop_pr.json next to its inputs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, help="ours_<backbone> arm dir with lc_decisions_lc.json + gt.tum")
    ap.add_argument("--pos-dist-frac", type=float, default=POS_DIST_FRAC)
    ap.add_argument("--pos-view-deg", type=float, default=POS_VIEW_DEG)
    ap.add_argument("--neg-dist-frac", type=float, default=NEG_DIST_FRAC)
    ap.add_argument("--neg-view-deg", type=float, default=NEG_VIEW_DEG)
    ap.add_argument("--min-submap-gap", type=int, default=MIN_SUBMAP_GAP)
    ap.add_argument("--nms-frame-distance", type=int, default=NMS_FRAME_DISTANCE)
    ap.add_argument("--max-loops-per-submap", type=int, default=MAX_LOOPS_PER_SUBMAP)
    args = ap.parse_args()

    result = loop_precision_recall(
        args.run_dir,
        pos_dist_frac=args.pos_dist_frac,
        pos_view_deg=args.pos_view_deg,
        neg_dist_frac=args.neg_dist_frac,
        neg_view_deg=args.neg_view_deg,
        min_submap_gap=args.min_submap_gap,
        nms_frame_distance=args.nms_frame_distance,
        max_loops_per_submap=args.max_loops_per_submap,
    )
    out = args.run_dir / "loop_pr.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
