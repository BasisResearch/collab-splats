"""Shared helpers for the LC parity harness (spec 2026-07-08-lc-parity-validation-design).

Scene registry, keyframe prefix slicing for scaling sweeps, Level-1 gate checks,
and Level-2 per-candidate LC decision serialization (used by evals/eval_gt.py).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

########## Scene registry ##########

# Gates from spec §Level 1
ATE_REL_TOL = 0.05  # <=5% relative
ATE_ABS_TOL = 0.005  # or <=5 mm absolute
PREFIX_FRACTIONS = (0.25, 0.50, 1.0)

_IMG_EXTS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class SceneSpec:
    """One validation scene: dataset type, seq path relative to evals/data/, sweep flag."""

    dataset: str  # eval_gt --dataset value ("7scenes" | "tum")
    rel_seq_dir: str  # relative to evals/data/
    scaling_sweep: bool = False  # include in 25/50/100% prefix stress (spec §scaling)


SCENES: dict[str, SceneSpec] = {
    "7s_chess": SceneSpec("7scenes", "7scenes/chess/chess/seq-01"),
    "7s_fire": SceneSpec("7scenes", "7scenes/fire/fire/seq-01"),
    "7s_heads": SceneSpec("7scenes", "7scenes/heads/heads/seq-01"),
    "7s_office": SceneSpec("7scenes", "7scenes/office/office/seq-01", scaling_sweep=True),
    "7s_pumpkin": SceneSpec("7scenes", "7scenes/pumpkin/pumpkin/seq-01"),
    "7s_redkitchen": SceneSpec("7scenes", "7scenes/redkitchen/redkitchen/seq-01", scaling_sweep=True),
    "7s_stairs": SceneSpec("7scenes", "7scenes/stairs/stairs/seq-01"),
    "tum_fr1_desk": SceneSpec("tum", "tum/rgbd_dataset_freiburg1_desk"),
    "tum_fr1_room": SceneSpec("tum", "tum/rgbd_dataset_freiburg1_room", scaling_sweep=True),
    "tum_fr2_xyz": SceneSpec("tum", "tum/rgbd_dataset_freiburg2_xyz"),
    "tum_fr3_office": SceneSpec("tum", "tum/rgbd_dataset_freiburg3_long_office_household", scaling_sweep=True),
}


########## Image listing (7-Scenes flat vs TUM rgb/) ##########


def list_scene_images(seq_dir: Path) -> list[Path]:
    """Sorted source images for a scene; handles 7-Scenes flat and TUM rgb/ layouts."""
    rgb = seq_dir / "rgb"
    root = rgb if rgb.is_dir() else seq_dir
    return sorted(p for p in root.iterdir() if p.suffix.lower() in _IMG_EXTS and ".depth" not in p.name)


def filter_images_to_list(images: list, list_file: Path) -> list:
    """Keep only images whose basename appears in list_file (one path/basename per line)."""
    allowed = {Path(line).name for line in Path(list_file).read_text().splitlines() if line.strip()}
    return [p for p in images if Path(str(p)).name in allowed]


def collect_frames(seq_dir: Path, image_list: Path | None = None,
                   max_frames: int | None = None) -> list[str]:
    """Source frames for a SLAM run: sorted scene images, optionally restricted to
    image_list (basename match — TUM GT-gap parity), then capped at max_frames."""
    images = list_scene_images(seq_dir)
    if image_list is not None:
        images = filter_images_to_list(images, image_list)
    frames = [str(p) for p in images]
    return frames[:max_frames] if max_frames is not None else frames


def write_tum_allowed_frames(seq_dir: Path, out_file: Path) -> Path:
    """Write GT-filtered TUM frame basenames (one per line) for --image_list restriction.

    Uses evals/datasets.py's `_load_tum` (max_frames=100_000, mirroring eval_gt.py's
    --keyframe_list load path) so the SLAM reference only ever sees frames the ours-side
    dataset loader keeps — eval_gt's parity guard aborts otherwise (TUM GT-gap filter).
    """
    # Spec-load datasets.py by file path: an installed `datasets` package (HF) would
    # shadow a plain import, same problem as ate_utils in run_vggt_slam_lc.py. Register
    # in sys.modules before exec — EvalDataset resolves its (stringified) field
    # annotations through sys.modules[cls.__module__]. Load stays inside the function:
    # datasets.py pulls numpy/scipy, which light importers of this module don't need.
    spec = importlib.util.spec_from_file_location(
        "_eval_datasets", Path(__file__).resolve().parents[1] / "datasets.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_eval_datasets"] = mod
    spec.loader.exec_module(mod)
    ds = mod._load_tum(seq_dir, max_frames=100_000)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(Path(p).name for p in ds.images))
    return out_file


########## Keyframe prefix slicing (scaling sweep) ##########


def slice_keyframes(kf_file: Path, fraction: float, out_file: Path) -> Path:
    """Write the first round(fraction*N) keyframe paths of kf_file to out_file."""
    lines = kf_file.read_text().splitlines()
    n = max(1, round(fraction * len(lines)))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines[:n]))
    return out_file


def slam_max_frames_for_prefix(prefix_file: Path, seq_dir: Path,
                               image_list: Path | None = None) -> int:
    """max_frames for a SLAM prefix re-run: source index of the last prefix keyframe + 1.

    Keyframe selection is deterministic (optical-flow tracker state depends only on
    frames seen so far), so SLAM on the first max_frames source frames selects
    exactly the prefix keyframes. When image_list is set, indices are within the
    FILTERED sequence — matching run_vggt_slam_lc.py, which filters before capping.
    """
    all_imgs = list_scene_images(seq_dir)
    if image_list is not None:
        all_imgs = filter_images_to_list(all_imgs, image_list)
    last_kf = prefix_file.read_text().splitlines()[-1]
    return [str(p) for p in all_imgs].index(last_kf) + 1


########## Level-1 gates ##########


def check_gates(slam_metrics: dict, ours_metrics: dict, check_loops: bool = True) -> dict:
    """Spec Level 1 gates: ATE |delta| <=5% rel or <=5mm abs; loop counts equal.

    check_loops=False skips the loop-count check (all loops_pass=True regardless of
    counts) — used for the baseline condition, where LC is always off, so comparing
    its (always-zero) loop count against SLAM's is meaningless. Loop-count equality
    is an LC-condition-only gate.
    """
    slam_ate = slam_metrics.get("ate_rmse")
    ours_ate = ours_metrics.get("ate_rmse")
    if slam_ate is None or ours_ate is None:
        ate_delta, ate_pass = None, False
    else:
        ate_delta = abs(ours_ate - slam_ate)
        ate_pass = ate_delta <= max(ATE_REL_TOL * slam_ate, ATE_ABS_TOL)
    slam_loops = slam_metrics.get("loop_closures")
    ours_loops = ours_metrics.get("loops_applied")
    loops_pass = True if not check_loops else (slam_loops is not None and slam_loops == ours_loops)
    return {
        "ate_delta": ate_delta,
        "ate_pass": ate_pass,
        "slam_loops": slam_loops,
        "ours_loops": ours_loops,
        "loops_pass": loops_pass,
        "all_pass": ate_pass and loops_pass,
    }


def check_scaling_gate(ate_25: float | None, ate_100: float | None) -> bool | None:
    """Spec scaling gate: ATE(100%) <= ~1.5x ATE(25%) — bounded growth as loop count
    increases with sequence length (no blow-up). None when either ATE is missing
    (scaling sweep not applicable or not yet run for this scene).
    """
    if ate_25 is None or ate_100 is None:
        return None
    return ate_100 <= 1.5 * ate_25


def check_lc_harmless(ate_baseline: float | None, ate_lc: float | None) -> bool | None:
    """Cross-model gate: LC must not degrade a backbone vs its own baseline.

    ate_lc <= max(1.05 * ate_baseline, ate_baseline + 0.005). None when either missing.
    Backbones without an upstream reference (vggt_omega, mapanything, vggtx) are gated
    against their own baseline instead of SLAM — same tolerances as the spark parity
    gate, applied one-sided (LC must not make things worse).
    """
    if ate_baseline is None or ate_lc is None:
        return None
    return ate_lc <= max((1 + ATE_REL_TOL) * ate_baseline, ate_baseline + ATE_ABS_TOL)


########## Level-2 LC decision trace ##########


def _serialize_lc_decisions(matches: list) -> list[dict]:
    """Flatten LoopMatch inspection objects to JSON rows for the parity harness."""
    return [
        {
            "l2_score": float(m.similarity_score),
            "query_submap": int(m.query_submap_id),
            "detected_submap": int(m.detected_submap_id),
            "query_frame": int(m.query_frame_idx),
            "detected_frame": int(m.detected_frame_idx),
            "accepted": bool(getattr(m, "accepted", False)),
            # reject_reason is a declared LoopMatch field now; getattr stays as a
            # defensive fallback for LoopMatch objects pickled before the field existed.
            "reject_reason": getattr(m, "reject_reason", None),
        }
        for m in matches
    ]
