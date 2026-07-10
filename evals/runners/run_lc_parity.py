#!/usr/bin/env python
"""LC parity matrix driver — Levels 0/1 + scaling prefixes, one scene at a time.

Serial by design (46 GB cgroup cap). Resumable: a run whose metrics.json exists
is skipped. Run inside tmux, never a notebook:

    /opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py            # full matrix
    /opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py --scenes 7s_chess
    /opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py --dry_run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from lc_parity_common import (
    PREFIX_FRACTIONS,
    SCENES,
    slam_max_frames_for_prefix,
    slice_keyframes,
    write_tum_allowed_frames,
)

PY = "/opt/venv/reconstruction/bin/python"
RUNNERS = Path(__file__).resolve().parent
REPO_ROOT = RUNNERS.parents[1]

# Pin THIS checkout for the driver's own imports too (mirrors _run_one's subprocess
# pin): write_tum_allowed_frames loads evals/datasets.py, which imports collab_splats —
# the venv's editable install would resolve that to the primary checkout instead.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SLAM_RUNNER = RUNNERS / "run_vggt_slam_lc.py"
EVAL_GT = RUNNERS.parent / "eval_gt.py"

# Paper/parity config — spec §Constraints. Every value explicit, none left to
# runner defaults, so a default drift can't silently break parity.
PARITY = ["--submap_size", "16", "--max_loops", "1", "--min_disparity", "50",
          "--conf_threshold", "25.0", "--lc_thres", "0.95"]

# Same choices as eval_gt.py's --backbone; spark is the only apples-to-apples upstream
# comparison (spec §Level 1), so it's first and always on by default. vggtx is gated
# behind explicit --backbones opt-in (omega/mapanything cover the cross-model sweep).
BACKBONE_CHOICES = ["vggtx", "vggt_omega", "mapanything", "vggt_spark"]
DEFAULT_BACKBONES = ["vggt_spark", "vggt_omega", "mapanything"]


def _slam_cmd(seq_dir: Path, out_dir: Path, max_frames: int | None = None,
              image_list: Path | None = None) -> list[str]:
    """Level-0 upstream reference run (full sequence unless prefix-capped)."""
    cmd = [PY, str(SLAM_RUNNER), "--seq_dir", str(seq_dir),
           "--out_tum", str(out_dir / "slam.tum"), *PARITY]
    if max_frames is not None:
        cmd += ["--max_frames", str(max_frames)]
    if image_list is not None:
        cmd += ["--image_list", str(image_list)]
    return cmd


def _scene_image_list(scene_key: str, seq_dir: Path, out_root: Path,
                      write_files: bool) -> Path | None:
    """Allowed-frames file for the Level-0 SLAM run, or None for 7-Scenes.

    TUM scenes: eval_gt's dataset loader drops GT-gap frames, and its parity guard
    aborts if selected_frames.txt names any of them — so the SLAM reference must be
    restricted to the loader's surviving frames. 7-Scenes has no such filter; those
    scenes get no --image_list and behave byte-identically to before.
    """
    if SCENES[scene_key].dataset != "tum":
        return None
    image_list = out_root / scene_key / "allowed_frames.txt"
    if write_files:
        write_tum_allowed_frames(seq_dir, image_list)
    return image_list


def _ours_cmd(scene_key: str, seq_dir: Path, kf_list: Path, out_dir: Path, backbone: str) -> list[str]:
    """Level-1 run: our pipeline on the exact SLAM keyframes, given backbone.

    --lc_scale_method is passed explicitly: eval_gt.py's CLI default is "se3", but
    VGGT-SLAM's own scale estimation is rotation-only (LoopClosureConfig's class
    default), so parity requires overriding the CLI default to match upstream.
    --max_loops_per_submap=1 matches upstream, which caps at 1 loop per submap.
    This protocol (submap_size, scale method, loop cap, conditions) is identical
    across backbones (spec §Cross-model sweep) — only --backbone and --output_dir vary.
    --slam_tum points at this scene's (or prefix's) Level-0 output — out_dir is always
    an `ours_<backbone>/` dir next to `slam/` (see build_scene_commands/build_prefix_commands)
    — so the trajectory overlay plot shows GT + ours + SLAM. Safe because Level 0 always
    runs (and writes slam.tum) before Level 1 in build_scene_commands/main()'s per-scene
    ordering, so the file exists by the time this command executes.
    """
    dataset = SCENES[scene_key].dataset
    slam_tum = out_dir.parent / "slam" / "slam.tum"
    return [PY, str(EVAL_GT), "--dataset", dataset, "--seq_dir", str(seq_dir),
            "--backbone", backbone, "--conditions", "baseline", "lc",
            "--submap_size", "16", "--lc_scale_method", "rotation_only",
            "--max_loops_per_submap", "1", "--keyframe_list", str(kf_list),
            "--output_dir", str(out_dir), "--output_ate", str(out_dir / "ate.json"),
            "--slam_tum", str(slam_tum)]


def build_scene_commands(scene_key: str, data_root: Path, out_root: Path,
                         backbones: list[str] = DEFAULT_BACKBONES,
                         write_files: bool = True) -> list[list[str]]:
    """Level 0 (one shared SLAM reference) + one Level 1 run per backbone, for one scene.

    write_files=False (dry runs) skips writing the TUM allowed-frames list to disk.
    """
    spec = SCENES[scene_key]
    seq_dir = data_root / spec.rel_seq_dir
    slam_dir = out_root / scene_key / "slam"
    kf_list = slam_dir / "selected_frames.txt"
    image_list = _scene_image_list(scene_key, seq_dir, out_root, write_files)
    cmds = [_slam_cmd(seq_dir, slam_dir, image_list=image_list)]
    for backbone in backbones:
        ours_dir = out_root / scene_key / f"ours_{backbone}"
        cmds.append(_ours_cmd(scene_key, seq_dir, kf_list, ours_dir, backbone))
    return cmds


def build_prefix_commands(scene_key: str, data_root: Path, out_root: Path,
                          backbones: list[str] = DEFAULT_BACKBONES,
                          write_files: bool = True) -> list[list[str]]:
    """Scaling-sweep argv (25/50%) for sweep scenes; 100% is the main run.

    Per fraction: one shared SLAM prefix run + one ours run per backbone (flat list,
    grouped in chunks of 1 + len(backbones); see main()'s consumption of this list).
    write_files=False (dry runs) skips slicing prefix keyframes.txt to disk and
    uses a max_frames=0 placeholder instead.
    """
    spec = SCENES[scene_key]
    if not spec.scaling_sweep:
        return []
    seq_dir = data_root / spec.rel_seq_dir
    full_kf = out_root / scene_key / "slam" / "selected_frames.txt"
    # Prefix SLAM re-runs share the scene-level allowed list (idempotent re-write);
    # max_frames indices are within the filtered sequence, matching the runner.
    image_list = _scene_image_list(scene_key, seq_dir, out_root, write_files)
    cmds: list[list[str]] = []
    for frac in PREFIX_FRACTIONS[:-1]:  # (0.25, 0.50) — 1.0 is the main run
        tag = f"prefix_{int(frac * 100)}"
        pdir = out_root / scene_key / tag
        prefix_kf = pdir / "keyframes.txt"
        if write_files and full_kf.exists():  # slicing needs the Level-0 output
            slice_keyframes(full_kf, frac, prefix_kf)
            max_frames = slam_max_frames_for_prefix(prefix_kf, seq_dir, image_list=image_list)
        else:
            max_frames = 0  # placeholder before Level 0 has produced keyframes / in dry runs
        cmds.append(_slam_cmd(seq_dir, pdir / "slam", max_frames=max_frames, image_list=image_list))
        for backbone in backbones:
            cmds.append(_ours_cmd(scene_key, seq_dir, prefix_kf, pdir / f"ours_{backbone}", backbone))
    return cmds


def _done(cmd: list[str]) -> bool:
    """Skip completed runs: metrics.json already present in the run's output dir."""
    for flag in ("--out_tum", "--output_dir"):
        if flag in cmd:
            out = Path(cmd[cmd.index(flag) + 1])
            out_dir = out.parent if flag == "--out_tum" else out
            return (out_dir / "metrics.json").exists()
    return False


def _run_one(cmd: list[str], dry_run: bool) -> None:
    if _done(cmd):
        print(f"[skip] {' '.join(cmd[2:6])} — metrics.json exists")
        return
    print(f"[run ] {' '.join(cmd)}")
    if not dry_run:
        # Pin subprocess imports to THIS checkout: the venv's editable collab_splats
        # install points at the primary checkout, so a worktree probe would silently
        # exercise the wrong code without an explicit PYTHONPATH prepend. xfeat rides
        # along for collab_splats.localization's `from modules.xfeat import XFeat`.
        env = dict(os.environ)
        pin = [str(REPO_ROOT), str(REPO_ROOT / "third_party" / "xfeat")]
        prior = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(pin + ([prior] if prior else []))
        subprocess.run(cmd, check=True, env=env)


def _has_zero_max_frames(cmd: list[str]) -> bool:
    """True if cmd carries a literal `--max_frames 0` (build_prefix_commands' placeholder)."""
    return any(cmd[i] == "--max_frames" and cmd[i + 1] == "0" for i in range(len(cmd) - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=REPO_ROOT / "evals" / "data")
    ap.add_argument("--out_root", type=Path, default=REPO_ROOT / "evals" / "baselines" / "lc_parity")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--backbones", nargs="*", default=list(DEFAULT_BACKBONES), choices=BACKBONE_CHOICES,
                    help="Ours-side backbones to sweep per scene (spec §Cross-model sweep). "
                         "One shared Level-0 SLAM run per scene regardless of backbone count.")
    ap.add_argument("--min_disparity", type=float, default=50.0,
                    help="Keyframe optical-flow disparity threshold for the Level-0 SLAM run. "
                         "50 = paper config; lower (e.g. 5) densifies keyframes and produces "
                         "loop-rich sequences for LC-focused probes. Use a distinct --out_root "
                         "for non-default values to keep configs separate.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Thread the disparity override into the shared PARITY flags (builders read it at call time)
    PARITY[PARITY.index("--min_disparity") + 1] = str(args.min_disparity)

    group_size = 1 + len(args.backbones)  # 1 slam cmd + 1 ours cmd per backbone, per prefix fraction
    for scene in args.scenes:
        # Prefix sweeps depend on the Level-0 keyframe list, so run after Levels 0/1
        for cmd in build_scene_commands(scene, args.data_root, args.out_root, args.backbones,
                                        write_files=not args.dry_run):
            _run_one(cmd, args.dry_run)
        prefix_cmds = build_prefix_commands(scene, args.data_root, args.out_root, args.backbones,
                                            write_files=not args.dry_run)
        # cmds come in (slam, ours_b1, ..., ours_bN) groups per prefix fraction. A slam
        # cmd carrying the --max_frames 0 placeholder means Level-0 hasn't produced
        # selected_frames.txt yet — on a stale resume (e.g. selected_frames.txt got
        # deleted after a run was marked done) that placeholder could otherwise reach a
        # real, zero-frame SLAM run. Skip the whole group instead of executing it.
        # Dry-run printing is unaffected — the 0 placeholder is still shown via the
        # normal _run_one path.
        for i in range(0, len(prefix_cmds), group_size):
            slam_cmd, ours_cmds = prefix_cmds[i], prefix_cmds[i + 1:i + group_size]
            if not args.dry_run and _has_zero_max_frames(slam_cmd):
                print("[warn] skipping prefix run (no Level-0 keyframes yet)")
                continue
            _run_one(slam_cmd, args.dry_run)
            for ours_cmd in ours_cmds:
                _run_one(ours_cmd, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
