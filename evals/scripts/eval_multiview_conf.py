"""Multiview confidence eval — VGGT-X, VGGTOmega, MapAnything on chess seq-01.

Measures whether MapAnything's geometric multiview depth confidence filter
improves point cloud quality for VGGT-X and VGGTOmega.

Run in tmux:
    /opt/conda/envs/reconstruction/bin/python evals/scripts/eval_multiview_conf.py

Outputs saved to evals/results/mv_conf_eval/:
    point_count_comparison.png
    mv_conf_distributions.png
    scatter_comparison.png
    summary.txt
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import get_dataset
from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator,
    VGGTOmegaCreator,
    VGGTXCreator,
)
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.geometry.transforms import invert_poses
from mapanything.utils.multiview_confidence import compute_multiview_depth_confidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

########################################################################
# Config
########################################################################

SEQ_DIR = Path("evals/data/7scenes/chess/chess/seq-01")
ZARR_BASE = Path("evals/results/mv_conf_eval")
OUT_DIR = ZARR_BASE
PERCENTILE = 35.0
MAX_FRAMES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


########################################################################
# Helpers
########################################################################


def _make_image_dir(image_paths: list[Path]) -> Path:
    """Copy images to a temp dir with sequential names for creators."""
    tmp = Path(tempfile.mkdtemp())
    for i, src in enumerate(image_paths):
        shutil.copy(src, tmp / f"{i:06d}{src.suffix}")
    return tmp


def compute_mv_conf(result: FeedforwardResult) -> np.ndarray:
    """Compute geometric multiview confidence from FeedforwardResult.

    Returns (N, H, W) float32 numpy array, values in [0, 1].
    """
    depth_np = result.depth.astype(np.float32)  # (N, H, W)
    N = depth_np.shape[0]

    # depth_z: List[(1, H, W, 1)] on DEVICE
    depth_z = [torch.from_numpy(depth_np[i]).unsqueeze(0).unsqueeze(-1).to(DEVICE) for i in range(N)]

    # intrinsics: List[(1, 3, 3)] on DEVICE
    intrs_np = result.intrinsics.astype(np.float32)
    intrinsics = [torch.from_numpy(intrs_np[i]).unsqueeze(0).to(DEVICE) for i in range(N)]

    # extrinsics are world2cam (N, 4, 4); invert to cam2world
    cam2world_np = invert_poses(result.extrinsics.astype(np.float32))  # (N, 4, 4)
    camera_poses = [torch.from_numpy(cam2world_np[i]).unsqueeze(0).to(DEVICE) for i in range(N)]

    with torch.no_grad():
        mv_conf_list = compute_multiview_depth_confidence(depth_z, intrinsics, camera_poses)

    mv_conf = torch.stack([c.squeeze(0).cpu() for c in mv_conf_list]).numpy()
    return mv_conf.astype(np.float32)


def apply_mask(conf: np.ndarray, percentile: float = PERCENTILE) -> np.ndarray:
    """Return (N, H, W) bool mask at given percentile threshold."""
    thresh = float(np.percentile(conf, percentile))
    return conf >= thresh


def point_count(world_points: np.ndarray, mask: np.ndarray) -> int:
    """Count surviving points in (N, H, W, 3) grid given (N, H, W) bool mask."""
    return int(mask.sum())


def run_and_save(creator_cls, creator_kwargs: dict, image_dir: Path, zarr_path: Path, label: str) -> FeedforwardResult:
    """Run creator if zarr absent, save zarr, load + return FeedforwardResult."""
    if zarr_path.exists():
        log.info("%s zarr exists, loading from cache: %s", label, zarr_path)
        return FeedforwardResult.load_zarr(zarr_path)

    log.info("Running %s inference...", label)
    creator = creator_cls(**creator_kwargs)
    result = creator.run(image_dir)
    result.save_zarr(zarr_path)
    log.info("%s: %d points saved → %s", label, result.points.shape[0], zarr_path)
    del creator
    torch.cuda.empty_cache()
    return FeedforwardResult.load_zarr(zarr_path)


########################################################################
# Diagnostic
########################################################################


def diagnose_mapanything_mv_conf(
    image_paths: list[Path],
    result_ma: FeedforwardResult,
    result_vggtx: FeedforwardResult,
    diagnose_h2: bool = False,
) -> None:
    """Section 7: MapAnything multiview confidence diagnostic.

    Tests three hypotheses for why use_multiview_confidence=True yields 0 points.
      H1 — depth_z scale mismatch (metric_scaling_factor not applied)
      H2 — confidence_percentile=35 too aggressive vs upstream default=10
      H3 — non_ambiguous_mask near-empty at 50 frames → all target pixels invalid
    """
    import collab_splats.pointcloud.feedforward.mapanything as _ma_mod
    import mapanything.utils.inference as _ma_inference_mod
    from mapanything.utils.inference import postprocess_model_outputs_for_inference as _upstream_pp

    SEP = "═" * 70
    print(f"\n{SEP}")
    print("  MapAnything mv_conf diagnostic  (50 frames, chess seq-01)")
    print(SEP)

    ################################################################
    # H1 — External mv_conf reproduce + depth scale comparison
    ################################################################
    print("\n── H1: external mv_conf + depth scale ──────────────────────────")
    log.info("H1: computing external mv_conf on MapAnything mv_off result...")
    mv_conf_ma = compute_mv_conf(result_ma)

    ptiles = [0, 5, 10, 25, 35, 50, 75, 95, 100]
    pvals = [float(np.percentile(mv_conf_ma, p)) for p in ptiles]
    print(f"  mv_conf: mean={mv_conf_ma.mean():.4f}  std={mv_conf_ma.std():.4f}")
    print("  " + "  ".join(f"p{p}={v:.3f}" for p, v in zip(ptiles, pvals)))

    n_p35 = int(apply_mask(mv_conf_ma, 35).sum())
    n_p10 = int(apply_mask(mv_conf_ma, 10).sum())
    print(f"  surviving: p35={n_p35:,}  p10={n_p10:,}  total_pixels={int(mv_conf_ma.size):,}")
    print(f"  mv_off baseline: {result_ma.points.shape[0]:,} pts")

    # Side-by-side depth_z stats — valid pixels only
    def _depth_stats(d: np.ndarray) -> dict:
        d = d[d > 0].ravel().astype(np.float64)
        return {
            "min": d.min(),
            "max": d.max(),
            "mean": d.mean(),
            "p5": float(np.percentile(d, 5)),
            "p95": float(np.percentile(d, 95)),
        }

    ma_s = _depth_stats(result_ma.depth)
    vx_s = _depth_stats(result_vggtx.depth)
    print(f"\n  {'stat':<8}  {'MapAnything':>14}  {'VGGT-X':>14}")
    print(f"  {'-'*40}")
    for k in ["min", "max", "mean", "p5", "p95"]:
        print(f"  {k:<8}  {ma_s[k]:>14.4f}  {vx_s[k]:>14.4f}")

    scale_ratio = (ma_s["p95"] - ma_s["p5"]) / ((vx_s["p95"] - vx_s["p5"]) + 1e-9)
    if scale_ratio < 0.05 or scale_ratio > 20.0:
        print(f"\n  ⚠  H1 LIKELY — scale ratio={scale_ratio:.3f}  (depth units mismatch)")
    elif n_p35 == 0 and n_p10 == 0:
        print(f"\n  H1 POSSIBLE — scale OK (ratio={scale_ratio:.3f}) but external mv_conf also 0")
    else:
        print(f"\n  H1 CLEARED — scale ratio={scale_ratio:.3f}  p35 surviving={n_p35:,}")

    ################################################################
    # H2 — Percentile rerun (optional, --diagnose-h2)
    ################################################################
    print("\n── H2: percentile mismatch (confidence_percentile=10 vs 35) ────")
    if diagnose_h2:
        image_dir_h2 = _make_image_dir(image_paths)
        try:
            log.info("H2: running MapAnything mv_on with confidence_percentile=10...")
            creator_h2 = MapAnythingCreator(
                use_multiview_confidence=True,
                confidence_percentile=10,
                max_points=500_000,
            )
            result_h2 = creator_h2.run(image_dir_h2)
            n_h2 = result_h2.points.shape[0]
            del creator_h2
            torch.cuda.empty_cache()
            print(f"  mv_on p10={n_h2:,} pts  |  mv_on p35=0 pts  |  mv_off={result_ma.points.shape[0]:,} pts")
            if n_h2 > 0:
                print("  ⚠  H2 CONFIRMED — p10 restores points; percentile=35 is the culprit")
            else:
                print("  H2 CLEARED — p10 also 0 pts; percentile not primary cause")
        finally:
            shutil.rmtree(image_dir_h2, ignore_errors=True)
    else:
        print("  (skipped — rerun with --diagnose-h2 to test; requires second model load)")

    ################################################################
    # H3 — non_ambiguous_mask density via monkey-patch
    ################################################################
    print("\n── H3: non_ambiguous_mask density ──────────────────────────────")
    log.info("H3: running MapAnything mv_on with postprocess patch to capture mask density...")

    captured_masks: list[float] = []

    def _patched_pp(*args, **kwargs):
        out = _upstream_pp(*args, **kwargs)
        for p in out:
            if "non_ambiguous_mask" in p:
                m = p["non_ambiguous_mask"]
                density = (
                    float(m.float().mean().item()) if isinstance(m, torch.Tensor) else float(m.astype(float).mean())
                )
                captured_masks.append(density)
        return out

    _ma_inference_mod.postprocess_model_outputs_for_inference = _patched_pp
    _ma_mod.postprocess_model_outputs_for_inference = _patched_pp

    image_dir_h3 = _make_image_dir(image_paths)
    try:
        creator_h3 = MapAnythingCreator(
            use_multiview_confidence=True,
            confidence_percentile=PERCENTILE,
            max_points=500_000,
        )
        creator_h3.run(image_dir_h3)
        del creator_h3
        torch.cuda.empty_cache()
    finally:
        shutil.rmtree(image_dir_h3, ignore_errors=True)
        _ma_inference_mod.postprocess_model_outputs_for_inference = _upstream_pp
        _ma_mod.postprocess_model_outputs_for_inference = _upstream_pp

    if captured_masks:
        arr = np.array(captured_masks, dtype=np.float32)
        print(f"  non_ambiguous_mask over {len(arr)} frames:")
        print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  " f"min={arr.min():.4f}  max={arr.max():.4f}")
        if arr.mean() < 0.2:
            print("  ⚠  H3 LIKELY — mean density <20%; most target pixels invalid → all outliers")
        else:
            print(f"  H3 CLEARED — mean density={arr.mean():.1%} (sufficient coverage)")
    else:
        print("  WARNING: no non_ambiguous_mask captured — key absent from processed outputs")
        print("  H3 INCONCLUSIVE")

    print(f"\n{SEP}\n")


########################################################################
# Main
########################################################################


def main(diagnose: bool = False, diagnose_h2: bool = False) -> None:
    log.info("Device: %s | MAX_FRAMES: %d | PERCENTILE: %.1f | diagnose=%s", DEVICE, MAX_FRAMES, PERCENTILE, diagnose)

    ZARR_BASE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset — first MAX_FRAMES images
    log.info("Loading dataset from %s", SEQ_DIR)
    dataset = get_dataset("7scenes")(SEQ_DIR, max_frames=MAX_FRAMES)
    image_paths = dataset.images[:MAX_FRAMES]
    log.info("%d frames loaded", len(image_paths))
    image_dir = _make_image_dir(image_paths)
    log.info("Temp image dir: %s", image_dir)

    try:
        ################################################################
        # 1. VGGT-X inference
        ################################################################
        result_vggtx = run_and_save(
            VGGTXCreator,
            {"conf_threshold": PERCENTILE, "max_points": 500_000},
            image_dir,
            ZARR_BASE / "vggtx",
            "VGGT-X",
        )
        assert result_vggtx.depth is not None, "VGGT-X depth missing"
        assert result_vggtx.world_points is not None, "VGGT-X world_points missing"
        log.info(
            "VGGT-X loaded: depth%s conf%s wp%s",
            result_vggtx.depth.shape,
            tuple(result_vggtx.confidence.shape) if result_vggtx.confidence is not None else None,
            result_vggtx.world_points.shape,
        )

        ################################################################
        # 2. VGGTOmega inference
        ################################################################
        result_omega = run_and_save(
            VGGTOmegaCreator,
            {"conf_threshold": PERCENTILE, "max_points": 500_000},
            image_dir,
            ZARR_BASE / "vggt_omega",
            "VGGTOmega",
        )
        assert result_omega.depth is not None, "VGGTOmega depth missing"
        assert result_omega.world_points is not None, "VGGTOmega world_points missing"
        log.info(
            "VGGTOmega loaded: depth%s conf%s wp%s",
            result_omega.depth.shape,
            tuple(result_omega.confidence.shape) if result_omega.confidence is not None else None,
            result_omega.world_points.shape,
        )

        ################################################################
        # 3. MapAnything — mv_off (cached) and mv_on (point count sanity check)
        ################################################################
        result_ma_off = run_and_save(
            MapAnythingCreator,
            {"use_multiview_confidence": False, "confidence_percentile": PERCENTILE, "max_points": 500_000},
            image_dir,
            ZARR_BASE / "mapanything_off",
            "MapAnything mv_off",
        )
        assert result_ma_off.depth is not None, "MapAnything mv_off depth missing"
        n_ma_off = result_ma_off.points.shape[0]
        log.info("MapAnything mv_off: %d points", n_ma_off)

        log.info("Running MapAnything mv_on...")
        creator_on = MapAnythingCreator(
            use_multiview_confidence=True, confidence_percentile=PERCENTILE, max_points=500_000
        )
        n_ma_on = creator_on.run(image_dir).points.shape[0]
        del creator_on
        torch.cuda.empty_cache()
        log.info("MapAnything mv_on: %d points", n_ma_on)

        ################################################################
        # 4. Compute multiview confidence for VGGT-X and VGGTOmega
        ################################################################
        log.info("Computing mv_conf for VGGT-X (%d frames)...", result_vggtx.depth.shape[0])
        mv_conf_vggtx = compute_mv_conf(result_vggtx)
        log.info(
            "VGGT-X mv_conf: mean=%.3f std=%.3f min=%.3f max=%.3f",
            mv_conf_vggtx.mean(),
            mv_conf_vggtx.std(),
            mv_conf_vggtx.min(),
            mv_conf_vggtx.max(),
        )

        log.info("Computing mv_conf for VGGTOmega (%d frames)...", result_omega.depth.shape[0])
        mv_conf_omega = compute_mv_conf(result_omega)
        log.info(
            "VGGTOmega mv_conf: mean=%.3f std=%.3f min=%.3f max=%.3f",
            mv_conf_omega.mean(),
            mv_conf_omega.std(),
            mv_conf_omega.min(),
            mv_conf_omega.max(),
        )

        ################################################################
        # 5. Filtering variants
        ################################################################
        # Confidence from zarr is a torch.Tensor; convert to numpy
        def _conf_np(result: FeedforwardResult) -> np.ndarray:
            c = result.confidence
            if isinstance(c, torch.Tensor):
                return c.cpu().numpy().astype(np.float32)
            return c.astype(np.float32)

        conf_vggtx = _conf_np(result_vggtx)  # (N, H, W) learned depth_conf
        conf_omega = _conf_np(result_omega)  # (N, H, W) learned depth_conf
        wp_vggtx = result_vggtx.world_points  # (N, H, W, 3)
        wp_omega = result_omega.world_points  # (N, H, W, 3)

        # VGGT-X variants
        mask_l_vggtx = apply_mask(conf_vggtx)
        mask_m_vggtx = apply_mask(mv_conf_vggtx)
        mask_i_vggtx = mask_l_vggtx & mask_m_vggtx
        n_l_vggtx = point_count(wp_vggtx, mask_l_vggtx)
        n_m_vggtx = point_count(wp_vggtx, mask_m_vggtx)
        n_i_vggtx = point_count(wp_vggtx, mask_i_vggtx)

        # VGGTOmega variants
        mask_l_omega = apply_mask(conf_omega)
        mask_m_omega = apply_mask(mv_conf_omega)
        mask_i_omega = mask_l_omega & mask_m_omega
        n_l_omega = point_count(wp_omega, mask_l_omega)
        n_m_omega = point_count(wp_omega, mask_m_omega)
        n_i_omega = point_count(wp_omega, mask_i_omega)

        ################################################################
        # 6. Summary table
        ################################################################
        rows = [
            ("VGGT-X", "learned_only", n_l_vggtx, mv_conf_vggtx.mean()),
            ("VGGT-X", "mv_only", n_m_vggtx, mv_conf_vggtx.mean()),
            ("VGGT-X", "intersect", n_i_vggtx, mv_conf_vggtx.mean()),
            ("VGGTOmega", "learned_only", n_l_omega, mv_conf_omega.mean()),
            ("VGGTOmega", "mv_only", n_m_omega, mv_conf_omega.mean()),
            ("VGGTOmega", "intersect", n_i_omega, mv_conf_omega.mean()),
            ("MapAnything", "mv_off", n_ma_off, float("nan")),
            ("MapAnything", "mv_on", n_ma_on, float("nan")),
        ]

        header = f"{'Model':<14} {'Variant':<14} {'N_points':>10}  {'mv_conf_mean':>12}"
        sep = "-" * len(header)
        lines = [header, sep]
        for model, variant, n, mv_mean in rows:
            mv_str = f"{mv_mean:.3f}" if not np.isnan(mv_mean) else "  n/a"
            lines.append(f"{model:<14} {variant:<14} {n:>10,}  {mv_str:>12}")

        summary = "\n".join(lines)
        print("\n" + summary + "\n")
        (OUT_DIR / "summary.txt").write_text(summary + "\n")
        log.info("Summary saved → %s", OUT_DIR / "summary.txt")

        ################################################################
        # 7. Point count bar chart
        ################################################################
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = [f"{r[0]}\n{r[1]}" for r in rows]
        values = [r[2] for r in rows]
        color_map = {
            "learned_only": "#4477AA",
            "mv_only": "#EE7733",
            "intersect": "#AA3377",
            "mv_off": "#4477AA",
            "mv_on": "#EE7733",
        }
        bar_colors = [color_map[r[1]] for r in rows]
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt=lambda v: f"{int(v):,}", padding=3, fontsize=8)
        ax.set_ylabel("Surviving points")
        ax.set_title(
            f"Point cloud density by model × filtering variant\n"
            f"(chess seq-01, {MAX_FRAMES} frames, p{int(PERCENTILE)} threshold)"
        )
        ax.grid(axis="y", alpha=0.3)
        from matplotlib.patches import Patch

        legend_elems = [
            Patch(facecolor="#4477AA", label="learned_only / mv_off"),
            Patch(facecolor="#EE7733", label="mv_only / mv_on"),
            Patch(facecolor="#AA3377", label="intersect"),
        ]
        ax.legend(handles=legend_elems, loc="upper right")
        plt.tight_layout()
        out_bar = OUT_DIR / "point_count_comparison.png"
        plt.savefig(out_bar, dpi=150)
        plt.close()
        log.info("Bar chart saved → %s", out_bar)

        ################################################################
        # 8. mv_conf distribution histograms
        ################################################################
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for ax, (label, mv_conf, conf_learned) in zip(
            axes,
            [
                ("VGGT-X", mv_conf_vggtx, conf_vggtx),
                ("VGGTOmega", mv_conf_omega, conf_omega),
            ],
        ):
            ax.hist(mv_conf.ravel(), bins=60, color="#EE7733", alpha=0.7, label="mv_conf (geometric)", density=True)
            # Normalise learned conf to [0,1] for overlay
            lc = conf_learned.ravel().astype(np.float64)
            lc_norm = (lc - lc.min()) / ((lc.max() - lc.min()) + 1e-9)
            ax.hist(lc_norm, bins=60, color="#4477AA", alpha=0.5, label="depth_conf normalised", density=True)
            thresh_mv = float(np.percentile(mv_conf, PERCENTILE))
            ax.axvline(thresh_mv, color="#EE7733", linestyle="--", label=f"mv p{int(PERCENTILE)}={thresh_mv:.3f}")
            ax.set_xlabel("Confidence score")
            ax.set_ylabel("Density")
            ax.set_title(f"{label} — confidence distributions")
            ax.legend(fontsize=8)
        plt.tight_layout()
        out_hist = OUT_DIR / "mv_conf_distributions.png"
        plt.savefig(out_hist, dpi=150)
        plt.close()
        log.info("Histograms saved → %s", out_hist)

        ################################################################
        # 9. 3D scatter — VGGT-X learned vs mv_only vs intersect
        ################################################################
        rng = np.random.default_rng(42)
        SCATTER_N = 4_000

        def _sample(wp, mask, n):
            idx = np.stack(np.where(mask), axis=1)
            if len(idx) > n:
                idx = idx[rng.choice(len(idx), n, replace=False)]
            return wp[idx[:, 0], idx[:, 1], idx[:, 2]]

        variants_vggtx = [
            (mask_l_vggtx, f"learned_only\n{n_l_vggtx:,} pts", "#4477AA"),
            (mask_m_vggtx, f"mv_only\n{n_m_vggtx:,} pts", "#EE7733"),
            (mask_i_vggtx, f"intersect\n{n_i_vggtx:,} pts", "#AA3377"),
        ]

        fig = plt.figure(figsize=(16, 5))
        for i, (mask, title, color) in enumerate(variants_vggtx, 1):
            pts = _sample(wp_vggtx, mask, SCATTER_N)
            ax = fig.add_subplot(1, 3, i, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4, alpha=0.5, c=pts[:, 2], cmap="viridis")
            ax.set_title(f"VGGT-X {title}", fontsize=9)
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.set_zlabel("Z", fontsize=7)
            ax.tick_params(labelsize=6)
        plt.suptitle("VGGT-X point clouds: filtering variant comparison", fontsize=11)
        plt.tight_layout()
        out_scatter = OUT_DIR / "scatter_comparison.png"
        plt.savefig(out_scatter, dpi=150)
        plt.close()
        log.info("Scatter saved → %s", out_scatter)

        log.info("Done. All outputs in %s", OUT_DIR)

        ################################################################
        # 10. MapAnything mv_conf diagnostic (--diagnose)
        ################################################################
        if diagnose:
            diagnose_mapanything_mv_conf(
                image_paths=image_paths,
                result_ma=result_ma_off,
                result_vggtx=result_vggtx,
                diagnose_h2=diagnose_h2,
            )

    finally:
        shutil.rmtree(image_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiview confidence eval — VGGT-X, VGGTOmega, MapAnything on chess seq-01"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run MapAnything mv_conf diagnostic (H1 + H3; requires second MapAnything run for H3)",
    )
    parser.add_argument(
        "--diagnose-h2",
        action="store_true",
        help="Also run H2 percentile rerun (--diagnose must also be set; requires third MapAnything run)",
    )
    args = parser.parse_args()
    main(diagnose=args.diagnose, diagnose_h2=args.diagnose_h2)
