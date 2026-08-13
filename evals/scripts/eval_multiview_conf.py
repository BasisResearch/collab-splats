"""Sweep rel_thresh x min_views against 7-Scenes GT depth, using the SHIPPING function.

Metric: relative depth error of RETAINED pixels vs GT, against retention fraction. A win is
lower retained-pixel error at comparable retention than the learned-confidence baseline.

VGGT depth is non-metric, so each scene is aligned by the median ratio
s = median(d_gt) / median(d_pred) over pixels with valid GT before any error is computed.

CLI/tmux only. Results under evals/results/ (gitignored).

Usage:
  python evals/scripts/eval_multiview_conf.py --zarr data/outputs/feedforward.zarr \
      --seq data/7scenes/chess/seq-01 --max-frames 60 --out evals/results/mv_sweep_chess01.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

from collab_splats.pointcloud.feedforward.base import (
    compute_multiview_depth_confidence,
    multiview_mask,
)

logger = logging.getLogger(__name__)

REL_THRESHOLDS = (0.01, 0.02, 0.05, 0.10)
MIN_VIEWS = (1, 2, 4, 8, 16)


def load_7scenes_depth(color_paths: list[Path]) -> np.ndarray:
    """GT depth in metres for each frame. 7-Scenes stores uint16 mm; 65535 = invalid."""
    out = []
    for p in color_paths:
        dpath = p.with_name(p.name.replace(".color.png", ".depth.png"))
        raw = np.asarray(Image.open(dpath)).astype(np.float32)
        d = raw / 1000.0
        d[raw == 65535] = 0.0
        out.append(d)
    return np.stack(out)


def median_align(pred: np.ndarray, gt: np.ndarray, both_valid: np.ndarray) -> float:
    """Scale factor putting non-metric predicted depth into GT units."""
    return float(np.median(gt[both_valid]) / np.median(pred[both_valid]))


def retained_error(pred_s: np.ndarray, gt: np.ndarray, keep: np.ndarray) -> dict:
    """Error stats over pixels that are kept AND have GT, plus the retention fraction.

    The median alone cannot judge this filter: mv drops a small tail of gross outliers, which
    moves p90/p95 and the >10% outlier fraction while leaving the median flat.
    """
    gt_ok = (gt > 0) & (pred_s > 0)
    sel = keep & gt_ok
    if not sel.any():
        return {
            "median_rel_err": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "frac_over_10pct": float("nan"),
            "retention": 0.0,
        }
    rel = np.abs(pred_s[sel] - gt[sel]) / gt[sel]
    return {
        "median_rel_err": float(np.median(rel)),
        "p90": float(np.percentile(rel, 90)),
        "p95": float(np.percentile(rel, 95)),
        "frac_over_10pct": float((rel > 0.10).mean()),
        "retention": float(sel.sum() / gt_ok.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", type=Path, required=True, help="feedforward.zarr from the backend run")
    ap.add_argument("--seq", type=Path, required=True, help="7-Scenes sequence dir with .depth.png")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Load prediction and GT, then put GT on the prediction's pixel grid
    z = zarr.open(str(args.zarr), mode="r")
    depth, K, E = z["depth"][:], z["intrinsics"][:], z["extrinsics"][:]
    color_paths = sorted(args.seq.glob("*.color.png"))[: args.max_frames][: depth.shape[0]]
    gt = load_7scenes_depth(color_paths)
    if gt.shape[1:] != depth.shape[1:]:
        gt = np.stack(
            [np.asarray(Image.fromarray(g).resize((depth.shape[2], depth.shape[1]), Image.NEAREST)) for g in gt]
        )
    logger.info("pred %s  gt %s", depth.shape, gt.shape)

    both = (gt > 0) & (depth > 0)
    s = median_align(depth, gt, both)
    depth_s = depth * s
    logger.info("median alignment scale = %.4f", s)

    # Reference row: no filtering at all, so every filtered row can be read as a delta
    rows = []
    stats = retained_error(depth_s, gt, depth > 0)
    rows.append({"variant": "unfiltered", "rel_thresh": None, "min_views": None, **stats})
    logger.info(
        "unfiltered: med=%.4f p90=%.4f p95=%.4f out10=%.4f retention=%.4f",
        stats["median_rel_err"],
        stats["p90"],
        stats["p95"],
        stats["frac_over_10pct"],
        stats["retention"],
    )

    # Baseline: learned confidence percentile only, matching the creators' conf_threshold
    if "confidence" in z:
        conf = z["confidence"][:]
        thr = float(np.percentile(conf, 50.0))
        stats = retained_error(depth_s, gt, (conf >= thr) & (depth > 0))
        rows.append({"variant": "learned_conf_p50", "rel_thresh": None, "min_views": None, **stats})
        logger.info(
            "baseline learned_conf_p50: med=%.4f p90=%.4f p95=%.4f out10=%.4f retention=%.4f",
            stats["median_rel_err"],
            stats["p90"],
            stats["p95"],
            stats["frac_over_10pct"],
            stats["retention"],
        )

    # Sweep the shipping function
    for rel in REL_THRESHOLDS:
        mv = compute_multiview_depth_confidence(depth, K, E, abs_thresh=0.0, rel_thresh=rel)
        for k in MIN_VIEWS:
            keep = multiview_mask(mv, depth > 0, min_views=k)
            stats = retained_error(depth_s, gt, keep)
            rows.append({"variant": "mv", "rel_thresh": rel, "min_views": k, **stats})
            logger.info(
                "rel=%.2f K=%d: med=%.4f p90=%.4f p95=%.4f out10=%.4f retention=%.4f",
                rel,
                k,
                stats["median_rel_err"],
                stats["p90"],
                stats["p95"],
                stats["frac_over_10pct"],
                stats["retention"],
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"zarr": str(args.zarr), "seq": str(args.seq), "scale": s, "rows": rows}, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
