"""Drive the cross-model chess benchmark: one eval_gt.py call per (backbone, frameset).

Serial by design — heavy GPU runs, 46 GB cgroup cap, no parallel jobs. Each run's
metrics.json lands under evals/baselines/cross_model/ for build_benchmark_table.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PY = "/opt/conda/envs/reconstruction/bin/python"
EVAL_GT = Path(__file__).resolve().parents[1] / "eval.py"


@dataclass
class RunSpec:
    """One eval_gt invocation: a backbone over a frameset at a submap size."""

    backbone: str
    frameset: str
    submap_size: int | None
    conditions: tuple[str, ...]
    keyframe_list: Path
    lc_scale_method: str = "rotation_only"


def _out_dir(out_root: Path, spec: RunSpec) -> Path:
    """Per-run output dir: <backbone>__<frameset>__<sm|single>."""
    sm = "single" if spec.submap_size is None else f"sm{spec.submap_size}"
    return out_root / f"{spec.backbone}__{spec.frameset}__{sm}"


def build_commands(specs: list[RunSpec], seq_dir: Path, out_root: Path) -> list[list[str]]:
    """Build the argv list for each RunSpec."""
    cmds: list[list[str]] = []
    for s in specs:
        out_dir = _out_dir(out_root, s)
        cmd = [
            PY, str(EVAL_GT),
            "--dataset", "7scenes",
            "--seq_dir", str(seq_dir),
            "--backbone", s.backbone,
            "--conditions", *s.conditions,
            "--lc_scale_method", s.lc_scale_method,
            "--keyframe_list", str(s.keyframe_list),
            "--output_dir", str(out_dir),
            "--output_ate", str(out_dir / "ate.json"),
        ]
        if s.submap_size is not None:
            cmd += ["--submap_size", str(s.submap_size)]
        cmds.append(cmd)
    return cmds


def build_matrix(
    kf_dir: Path,
    backbones: list[str],
    single_framesets: list[str],
    windowed_framesets: list[str],
    submap_size: int,
) -> list[RunSpec]:
    """Build the benchmark matrix: backbones × framesets × conditions.

    Each frameset name is a subdirectory of kf_dir containing selected_frames.txt
    (e.g. "slam_d10" -> kf_dir/slam_d10/selected_frames.txt). single_framesets get
    a single-pass baseline-only run each; windowed_framesets get a windowed
    baseline+lc run each (a name may appear in both lists). Framesets whose
    keyframe file doesn't exist are skipped.
    """
    specs: list[RunSpec] = []
    for b in backbones:
        # Single-pass baseline-only runs (windowing-cost comparison).
        for fs in single_framesets:
            kf = kf_dir / fs / "selected_frames.txt"
            if kf.exists():
                specs.append(RunSpec(b, f"{fs}_single", None, ("baseline",), kf))
        # Windowed baseline+lc runs (LC-benefit comparison).
        for fs in windowed_framesets:
            kf = kf_dir / fs / "selected_frames.txt"
            if kf.exists():
                specs.append(RunSpec(b, fs, submap_size, ("baseline", "lc"), kf))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path,
                    default=Path("evals/data/7scenes/chess/chess/seq-01"))
    ap.add_argument("--kf_dir", type=Path,
                    default=Path("evals/baselines/disparity_sweep"))
    ap.add_argument("--out_root", type=Path,
                    default=Path("evals/baselines/cross_model"))
    ap.add_argument("--backbones", nargs="+",
                    default=["vggt_spark", "vggtx", "vggt_omega", "mapanything"],
                    help="Backbones to benchmark. Default: the 2026-05-31 4-backbone matrix.")
    ap.add_argument("--single_framesets", nargs="+",
                    default=["slam_d10", "slam_d20"],
                    help="Frameset subdirs under --kf_dir to run single-pass baseline-only "
                         "(windowing-cost comparison). Missing ones are skipped. "
                         "Default: the 2026-05-31 frameset list.")
    ap.add_argument("--windowed_framesets", nargs="+",
                    default=["slam_d10", "slam_d5_long"],
                    help="Frameset subdirs under --kf_dir to run windowed baseline+lc "
                         "(LC-benefit comparison). Missing ones are skipped. "
                         "Default: the 2026-05-31 frameset list.")
    ap.add_argument("--submap_size", type=int, default=16,
                    help="Submap size for the windowed baseline+lc runs. Default 16.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without running.")
    args = ap.parse_args()

    specs = build_matrix(
        args.kf_dir, args.backbones, args.single_framesets, args.windowed_framesets, args.submap_size
    )
    cmds = build_commands(specs, args.seq_dir, args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for i, cmd in enumerate(cmds, 1):
        print(f"\n[{i}/{len(cmds)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}) — continuing", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
