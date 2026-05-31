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
EVAL_GT = Path(__file__).resolve().parents[1] / "eval_gt.py"


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


def core_matrix(kf_dir: Path) -> list[RunSpec]:
    """Core matrix: 4 backbones × framesets × conditions (spec §Execution step 2)."""
    backbones = ["vggt_spark", "vggtx", "vggt_omega", "mapanything"]
    d10_kf = kf_dir / "slam_d10" / "selected_frames.txt"
    d20_kf = kf_dir / "slam_d20" / "selected_frames.txt"
    specs: list[RunSpec] = []
    for b in backbones:
        # Goal 1 — windowing cost: single-pass vs windowed baseline on short sets.
        specs.append(RunSpec(b, "slam_d10_single", None, ("baseline",), d10_kf))
        specs.append(RunSpec(b, "slam_d20_single", None, ("baseline",), d20_kf))
        # Windowed baseline + lc on d10 (2 submaps).
        specs.append(RunSpec(b, "slam_d10", 16, ("baseline", "lc"), d10_kf))
        # Goal 2 — LC benefit on the long, loop-closing set (only if it exists).
        long_kf = kf_dir / "slam_d5_long" / "selected_frames.txt"
        if long_kf.exists():
            specs.append(RunSpec(b, "slam_d5_long", 16, ("baseline", "lc"), long_kf))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path,
                    default=Path("evals/data/7scenes/chess/chess/seq-01"))
    ap.add_argument("--kf_dir", type=Path,
                    default=Path("evals/baselines/disparity_sweep"))
    ap.add_argument("--out_root", type=Path,
                    default=Path("evals/baselines/cross_model"))
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without running.")
    args = ap.parse_args()

    specs = core_matrix(args.kf_dir)
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
