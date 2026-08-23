#!/usr/bin/env python
"""Localization benchmark across vismatch LocalMatcher models.

(Historical: originally compared the legacy in-repo extractors against their
vismatch counterparts; the legacy extractors were retired 2026-08-17 after the
parity gate passed. Every spec now constructs a vismatch LocalMatcher.)

For each matcher, builds a CameraLocalizer index over one reconstructed scene and
localizes N held-out reconstruction frames (leave-one-out: the query frame is
masked out of the index while it is being localized, so it never matches itself).
The frame's own reconstruction pose is the ground truth.

Per query: correspondences, inliers, inlier ratio, wall time, and rotation (deg) /
camera-center translation deltas vs the reconstruction pose (translation is in
reconstruction units — the backbones are non-metric). Reports median AND p90 per
matcher (median-only hides tails), prints a compact table, writes JSON.

Usage (tmux, never a notebook — GPU compute):
  /opt/venv/reconstruction/bin/python evals/scripts/eval_localization_parity.py \
      --scene data/outputs/<scene>/vggt_omega \
      --matchers disk-lightglue,xfeat,loma
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from collab_splats.localization.extractors import LocalMatcher
from collab_splats.localization.localizer import CameraLocalizer
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.utils.torch_utils import pytorch_gc

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

########################################
# Matcher specs
########################################

# 'vismatch:<name>' now only controls cache-key NAMESPACING: prefixed specs cache
# under 'vismatch-<name>', bare specs under '<name>'. Both construct the same
# LocalMatcher — the prefix exists so old parity-era caches keep resolving and new
# runs can be isolated from a pre-retirement cache of the same bare name.
_VISMATCH_PREFIX = "vismatch:"

# Default sweep: the three shipping vismatch models.
_DEFAULT_MATCHERS = "disk-lightglue,xfeat,loma"

# 2026-07-22 reference-scene measurements (correspondences/inliers) — historical
# sanity anchors from the retired legacy matchers, not exact targets (scene-dependent).
_ANCHORS = "Sanity anchors (2026-07-22, reference scene): disk 3314/2099, xfeat 2603/1211 (corr/inliers)"

_EPILOG = f"""\
Every spec is a vismatch model name, constructed as LocalMatcher(name).
An optional 'vismatch:' prefix only changes the zarr feature-cache key
('vismatch-<name>' instead of '<name>') — useful to keep a run's cache
separate from an older cache written under the bare name.
{_ANCHORS}
"""


def _build_matcher(spec: str):
    """Matcher spec -> (LocalMatcher instance, zarr feature-cache key).

    Both arms construct a vismatch LocalMatcher; the 'vismatch:' prefix only
    namespaces the cache key ('vismatch-<name>' vs bare '<name>').
    """
    if spec.startswith(_VISMATCH_PREFIX):
        model = spec[len(_VISMATCH_PREFIX) :]
        return LocalMatcher(model), f"vismatch-{model}"
    return LocalMatcher(spec), spec


########################################
# Pose deltas + stats
########################################


def _rotation_error_deg(pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    """Geodesic rotation distance (degrees) between two w2c poses."""
    R = pose_est[:3, :3] @ pose_gt[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _translation_error(pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    """Camera-center distance between two w2c poses (reconstruction units)."""
    c_est = -pose_est[:3, :3].T @ pose_est[:3, 3]
    c_gt = -pose_gt[:3, :3].T @ pose_gt[:3, 3]
    return float(np.linalg.norm(c_est - c_gt))


def _stats(vals: list[float]) -> dict | None:
    """median/p90/n over finite values; None when empty."""
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}


def _model_res_images(ff_images) -> list[np.ndarray]:
    """ff.images (N,3,H,W) tensor -> list of HWC uint8 RGB (mirrors _build_pairwise_refs)."""
    imgs = ff_images.detach().cpu().numpy() if torch.is_tensor(ff_images) else np.asarray(ff_images)
    if imgs.ndim == 4 and imgs.shape[1] == 3 and imgs.shape[-1] != 3:
        imgs = imgs.transpose(0, 2, 3, 1)
    if imgs.max() <= 1.5:  # MapAnything stores [0, 1]; VGGT stores [0, 255]
        imgs = imgs * 255.0
    return [im.astype(np.uint8) for im in np.round(imgs)]


########################################
# Per-matcher benchmark
########################################


def _run_matcher(spec, ff, images, ids, query_idx, top_k) -> dict:
    """Build index for one matcher, localize each held-out query, return records + stats."""
    extractor, extractor_name = _build_matcher(spec)
    localizer = CameraLocalizer.from_feedforward(
        ff, images=images, ids=ids, extractor=extractor, extractor_name=extractor_name, top_k=top_k
    )

    records = []
    for q in query_idx:
        # Leave-one-out: both matching paths only consider frames whose source is
        # "reconstruction" (descriptor loop and the pairwise retrieval gate), so
        # flipping the provenance masks the query frame without disturbing the
        # index-aligned world_points/_frame_features arrays.
        localizer._frame_sources[q] = "held-out"
        t0 = time.perf_counter()
        res = localizer.localize(images[q], query_intrinsics=ff.intrinsics[q] if ff.intrinsics is not None else None)
        dt = time.perf_counter() - t0
        localizer._frame_sources[q] = "reconstruction"

        # Pose deltas vs the frame's own reconstruction pose (GT by construction)
        rot = trans = None
        if res.pose is not None:
            rot = _rotation_error_deg(res.pose, ff.extrinsics[q])
            trans = _translation_error(res.pose, ff.extrinsics[q])
        records.append(
            {
                "query_frame": int(q),
                "query_id": ids[q],
                "n_correspondences": int(res.n_correspondences),
                "n_inliers": int(res.n_inliers),
                "inlier_ratio": float(res.n_inliers / res.n_correspondences) if res.n_correspondences else 0.0,
                "time_s": float(dt),
                "pose_found": res.pose is not None,
                "rot_error_deg": rot,
                "trans_error": trans,
            }
        )
        logger.info(
            "%s frame %d: %d corr / %d inl, %.2fs, rot=%s trans=%s",
            spec,
            q,
            res.n_correspondences,
            res.n_inliers,
            dt,
            f"{rot:.3f}°" if rot is not None else "FAIL",
            f"{trans:.4f}" if trans is not None else "FAIL",
        )

    # Aggregate median + p90 (median-only hides tails)
    summary = {
        "n_queries": len(records),
        "n_failed": sum(1 for r in records if not r["pose_found"]),
        "n_correspondences": _stats([r["n_correspondences"] for r in records]),
        "n_inliers": _stats([r["n_inliers"] for r in records]),
        "inlier_ratio": _stats([r["inlier_ratio"] for r in records]),
        "time_s": _stats([r["time_s"] for r in records]),
        "rot_error_deg": _stats([r["rot_error_deg"] for r in records]),
        "trans_error": _stats([r["trans_error"] for r in records]),
    }

    # Release model weights + CUDA cache before the next matcher loads
    del localizer, extractor
    pytorch_gc()
    return {"spec": spec, "extractor_name": extractor_name, "summary": summary, "per_query": records}


########################################
# Reporting
########################################


def _fmt(s: dict | None, prec: int = 0) -> str:
    """'median/p90' cell, '—' when no data."""
    if s is None:
        return "—"
    return f"{s['median']:.{prec}f}/{s['p90']:.{prec}f}"


def _print_table(results: list[dict]) -> None:
    """Compact per-matcher table: median/p90 columns."""
    header = f"{'matcher':<22} {'corr':>12} {'inliers':>12} {'ratio':>12} {'time_s':>12} {'rot_deg':>14} {'trans':>16} {'fail':>5}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        s = r["summary"]
        print(
            f"{r['spec']:<22} {_fmt(s['n_correspondences']):>12} {_fmt(s['n_inliers']):>12} "
            f"{_fmt(s['inlier_ratio'], 3):>12} {_fmt(s['time_s'], 2):>12} "
            f"{_fmt(s['rot_error_deg'], 3):>14} {_fmt(s['trans_error'], 4):>16} {s['n_failed']:>5}"
        )
    print(f"(cells are median/p90 over queries)\n{_ANCHORS}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", type=Path, required=True, help="scene dir containing pointcloud.zarr")
    ap.add_argument("--matchers", default=_DEFAULT_MATCHERS, help="comma list; 'vismatch:<name>' forces vismatch")
    ap.add_argument("--n_queries", type=int, default=10, help="held-out frames to localize per matcher")
    ap.add_argument("--top_k", type=int, default=8, help="retrieval-ranked refs per query (pairwise path only)")
    ap.add_argument("--output", type=Path, default=None, help="results JSON (default under evals/results/)")
    args = ap.parse_args()

    # Load reconstruction: model-res images double as reference AND query pixels —
    # matched ref pixels then live on the world_points grid with no rescale, and
    # every matcher sees identical inputs (the parity condition).
    ff = FeedforwardResult.load_zarr(args.scene / "pointcloud.zarr", load_images=True)
    if ff.images is None:
        raise ValueError(f"{args.scene / 'pointcloud.zarr'} has no images array — required for queries/pairwise refs")
    images = _model_res_images(ff.images)
    ids = [str(p) for p in ff.image_paths]

    # Evenly spaced query frames across the trajectory
    n_q = min(args.n_queries, len(images))
    query_idx = sorted(set(np.linspace(0, len(images) - 1, n_q).round().astype(int).tolist()))
    logger.info("Scene %s: %d frames, %d queries at %s", args.scene, len(images), len(query_idx), query_idx)

    # Benchmark each matcher sequentially (one model resident at a time)
    results = [_run_matcher(spec, ff, images, ids, query_idx, args.top_k) for spec in args.matchers.split(",")]

    _print_table(results)

    # Write JSON report
    out = args.output or Path("evals/results/localization_parity") / (
        f"{args.scene.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scene": str(args.scene),
        "n_queries": len(query_idx),
        "query_frames": [int(q) for q in query_idx],
        "top_k": args.top_k,
        "anchors": _ANCHORS,
        "matchers": results,
    }
    out.write_text(json.dumps(report, indent=2, default=float))
    logger.info("Report: %s", out)


if __name__ == "__main__":
    main()
