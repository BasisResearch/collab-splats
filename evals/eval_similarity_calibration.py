"""Similarity score calibration across feedforward backends.

Runs cross_frame_attention_ratio on frame pairs using each available FF model
(VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator) and reports scores under
both aggregation methods — old np.percentile(90) vs new mean_top_quarter — to
confirm the aggregation fix closes the gap with VGGT-SPARK (VGGT-1B, mean=1.025).

Usage (tmux — GPU required):
    /opt/conda/envs/nerfstudio/bin/python evals/eval_similarity_calibration.py \
        --scene_dir /data/7scenes/chess/seq-01 \
        --n_pairs 10 \
        --models vggtx mapanything

Output: summary table + evals/results/similarity_calibration.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


########################################
####### Aggregation helpers ############
########################################

def percentile_90(ratio_np: np.ndarray) -> float:
    """Old aggregation: 90th percentile (pre-fix)."""
    return float(np.percentile(ratio_np, 90))


def mean_top_quarter(ratio_np: np.ndarray) -> float:
    """New aggregation matching VGGT-SPARK get_similarity() / mean_top_quarter()."""
    thresh = float(np.percentile(ratio_np, 75))
    top_vals = ratio_np[ratio_np >= thresh]
    return float(top_vals.mean()) if top_vals.size > 0 else 0.0


########################################
####### Frame loading ##################
########################################

def load_7scenes_images(scene_dir: Path, n: int = 200) -> list[Path]:
    frames = sorted(scene_dir.glob("frame-*.color.png"))
    if not frames:
        raise FileNotFoundError(f"No frame-*.color.png in {scene_dir}")
    return frames[:n]


def sample_pairs(
    frames: list[Path],
    n_pairs: int,
    min_gap: int = 5,
    max_gap: int = 40,
) -> list[tuple[Path, Path]]:
    """Reproducibly sample frame pairs with temporal gap in [min_gap, max_gap]."""
    rng = random.Random(42)
    pairs: list[tuple[Path, Path]] = []
    for _ in range(n_pairs * 20):
        i = rng.randint(0, len(frames) - max_gap - 1)
        gap = rng.randint(min_gap, max_gap)
        j = i + gap
        if j < len(frames):
            pairs.append((frames[i], frames[j]))
        if len(pairs) >= n_pairs:
            break
    return pairs[:n_pairs]


########################################
####### Raw ratio computation ##########
########################################

def _compute_ratio_np(k: torch.Tensor, q: torch.Tensor, token_offset: int = 5) -> np.ndarray:
    """Mirrors cross_frame_attention_ratio internals up to final aggregation."""
    tokens_per_img = q.shape[2] // 2
    k_first = k[:, :, token_offset:tokens_per_img, :]
    if k_first.shape[2] == 0:
        return np.array([0.0])
    attn = q @ k_first.transpose(-2, -1)
    attn = attn.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = attn.mean(dim=1)
    attn_to_first = attn[..., :tokens_per_img]
    attn_to_second = attn[..., tokens_per_img:]
    max_self = attn_to_first.max(dim=-1)[0]
    normalized = attn_to_second / (max_self.unsqueeze(-1) + 1e-8)
    ratio = normalized.max(dim=1)[0]
    return ratio.cpu().float().numpy().ravel()


########################################
####### Per-model eval #################
########################################

def eval_model(
    creator_cls,
    creator_kwargs: dict,
    pairs: list[tuple[Path, Path]],
    device: str,
    label: str,
) -> dict:
    """Instantiate a FF creator and compute scores for all pairs via temp-dir preprocessing."""
    log.info("Loading model: %s", label)
    creator = creator_cls(**creator_kwargs)
    # _load_model is only called inside run(); call explicitly so self.model is set.
    creator.model = creator._load_model(device)

    scores_old: list[float] = []
    scores_new: list[float] = []

    for i, (p1, p2) in enumerate(pairs):
        log.info("  pair %d/%d: %s ↔ %s", i + 1, len(pairs), p1.name, p2.name)
        try:
            # Copy 2 frames into a temp dir; _preprocess() expects a directory.
            # Names must sort so frame1 < frame2 (alphanumeric order = temporal order).
            # Use shutil.copy2 (not symlinks) — symlinks with relative paths break
            # when accessed from the temp dir at a different cwd.
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                shutil.copy2(p1, tmp_dir / f"a_{p1.name}")
                shutil.copy2(p2, tmp_dir / f"b_{p2.name}")

                with torch.no_grad():
                    views, _, _ = creator._preprocess(tmp_dir)
                    # views: (2, 3, H, W) tensor (VGGT-X/Omega) or list of dicts (MapAnything)
                    if isinstance(views, torch.Tensor):
                        images = views.to(device)
                    else:
                        # MapAnything: each v["img"] is (1, C, H, W)
                        images = torch.cat([v["img"] for v in views], dim=0).to(device)

                    # Use the model's calibrated layer and token offset.
                    layer_idx = getattr(creator, "_lc_layer_index", -1)
                    tok_off   = getattr(creator, "_lc_token_offset", 5)
                    features = creator.extract_intermediate_features(
                        images, layer_index=layer_idx
                    )

            k, q = features["k"], features["q"]
            ratio_np = _compute_ratio_np(k, q, token_offset=tok_off)

            scores_old.append(percentile_90(ratio_np))
            scores_new.append(mean_top_quarter(ratio_np))

        except Exception as exc:
            log.warning("  pair %d failed: %s", i + 1, exc)
            scores_old.append(float("nan"))
            scores_new.append(float("nan"))

    valid_old = [s for s in scores_old if not np.isnan(s)]
    valid_new = [s for s in scores_new if not np.isnan(s)]

    return {
        "model": label,
        "n_pairs": len(pairs),
        "n_valid": len(valid_old),
        "percentile90": {
            "mean": float(np.mean(valid_old)) if valid_old else float("nan"),
            "min":  float(np.min(valid_old))  if valid_old else float("nan"),
            "max":  float(np.max(valid_old))  if valid_old else float("nan"),
            "scores": scores_old,
        },
        "mean_top_quarter": {
            "mean": float(np.mean(valid_new)) if valid_new else float("nan"),
            "min":  float(np.min(valid_new))  if valid_new else float("nan"),
            "max":  float(np.max(valid_new))  if valid_new else float("nan"),
            "scores": scores_new,
        },
    }


########################################
####### Main ###########################
########################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Similarity calibration across FF backends")
    parser.add_argument("--scene_dir", type=Path, required=True,
                        help="7-Scenes sequence dir (frame-*.color.png)")
    parser.add_argument("--n_pairs", type=int, default=10)
    parser.add_argument("--min_gap", type=int, default=5,
                        help="Min frame gap for pairs (default 5)")
    parser.add_argument("--max_gap", type=int, default=40,
                        help="Max frame gap for pairs (default 40)")
    parser.add_argument("--models", nargs="+",
                        choices=["vggtx", "mapanything", "omega"],
                        default=["vggtx", "mapanything"])
    parser.add_argument("--out", type=Path,
                        default=Path("evals/results/similarity_calibration.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s | models: %s | n_pairs: %d", device, args.models, args.n_pairs)

    frames = load_7scenes_images(args.scene_dir)
    pairs = sample_pairs(frames, args.n_pairs, args.min_gap, args.max_gap)
    log.info("Sampled %d pairs from %d frames", len(pairs), len(frames))

    results = []

    # VGGT-SPARK reference scores (VGGT-1B, mean_top_quarter, from parity harness)
    results.append({
        "model": "vggt_spark_reference",
        "note": "VGGT-1B via VGGT-SPARK, get_similarity = mean_top_quarter, chess_seq01",
        "mean_top_quarter": {
            "scores": [1.0321, 1.0432, 1.0359, 1.0278, 1.0205,
                       1.0016, 1.0222, 1.0223, 1.0147, 1.0181, 1.0246],
            "mean": 1.025,
            "min":  1.0016,
            "max":  1.0432,
        },
    })

    if "vggtx" in args.models:
        try:
            from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
            results.append(eval_model(VGGTXCreator, {}, pairs, device, "vggtx"))
        except Exception as exc:
            log.error("vggtx failed: %s", exc)

    if "mapanything" in args.models:
        try:
            from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator
            results.append(eval_model(MapAnythingCreator, {}, pairs, device, "mapanything"))
        except Exception as exc:
            log.error("mapanything failed: %s", exc)

    if "omega" in args.models:
        try:
            from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator
            results.append(eval_model(VGGTOmegaCreator, {}, pairs, device, "vggt_omega"))
        except Exception as exc:
            log.error("vggt_omega failed: %s", exc)

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Model':<22} {'pct90 mean':>11} {'mtq mean':>10} {'mtq min':>9} {'mtq max':>9}")
    print("-" * 72)
    for r in results:
        mtq = r.get("mean_top_quarter", {})
        p90 = r.get("percentile90", {})
        print(
            f"{r['model']:<22}"
            f" {p90.get('mean', float('nan')):>11.4f}"
            f" {mtq.get('mean', float('nan')):>10.4f}"
            f" {mtq.get('min', float('nan')):>9.4f}"
            f" {mtq.get('max', float('nan')):>9.4f}"
        )
    print("=" * 72)
    print(f"\nVGGT-SPARK threshold = 0.85  |  gap (mtq mean − 0.85):")
    for r in results:
        if r["model"] == "vggt_spark_reference":
            continue
        mean = r.get("mean_top_quarter", {}).get("mean", float("nan"))
        gap = mean - 0.85
        status = "✓ above" if gap >= 0 else f"✗ below by {abs(gap):.4f}"
        print(f"  {r['model']:<22} {status}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results → %s", args.out)


if __name__ == "__main__":
    main()
