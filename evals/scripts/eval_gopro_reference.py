"""Write a GoPro telemetry reference and per-backend trajectories for `eval_compare.py`.

This script computes **no** trajectory metric.  It produces the files the existing phase-2
runner already consumes — a shared `gt.tum` plus one `<backend>.tum` per reconstruction — so
ATE, RPE and AUC come from `evals/metrics.py` unchanged:

    python evals/scripts/eval_gopro_reference.py \\
        --telemetry GH010230_telemetry.parquet \\
        --recon omega=data/outputs/gopro-compare/omega/GH010230/vggt_omega/colmap/sparse/0 \\
        --recon loger=data/outputs/gopro-compare/loger/GH010230/loger/colmap/sparse/0 \\
        --results-dir evals/results/gopro_GH010230
    python evals/scripts/eval_compare.py --results-dir evals/results/gopro_GH010230

What it does supply is the one piece of maths those functions cannot: the constant offset
between each reconstruction's camera frame and the GoPro's, which does not cancel out of
relative poses and would otherwise make RPE report a large rotation error for a perfect
reconstruction.  See `evals/rotation_alignment.py` for why.

Two decisions are made by measurement rather than assumption:

* **Which CORI/IORI composition describes the stabilised image** is undocumented, so all five
  candidates are scored on the fit-free invariant turn-angle difference — scoring them on a
  fitted quantity would be circular, since the fit is what the choice feeds.  A convention
  chosen by fitting one backend would not be expected to also win for another, so agreement
  across backends is the validity check, and disagreement is reported loudly.
* **Whether the offset fit succeeded** is checked against the invariant floor, which the
  fitted error cannot fall below.  Landing far above it means the fit failed rather than the
  reconstruction being bad.

Used for `docs/benchmarks/2026-08-14-loger-vs-omega-gopro.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gopro_telemetry import (
    ORIENTATION_MODES,
    reference_rotations_c2w,
    sample_reference_at_frames,
)
from rotation_alignment import (
    invariant_turn_angle_error_deg,
    relative_rotation_error_deg,
)
from trajectory_io import write_tum

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collab_splats.geometry.transforms import invert_poses

logger = logging.getLogger(__name__)


########################################################################
# Reconstruction loading
########################################################################


def load_colmap_frame_poses(sparse_dir: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a COLMAP model as (frame_idx, camera centres, c2w rotations), frame-index sorted.

    Image names carry the source frame index (`frame_000063`), which is what makes telemetry
    alignment exact rather than interpolated — the pipeline's frame sampling is irregular
    enough (a blur gate substitutes a sharper neighbour) that assuming uniform spacing would
    misalign the reference.
    """
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    rows = []
    for image in reconstruction.images.values():
        frame_idx = int(image.name.split("_")[1])
        # cam_from_world is a METHOD in this pycolmap build, matching the repo's own idiom in
        # collab_splats/pointcloud/base.py (`img.cam_from_world().rotation.matrix()`).
        rigid = image.cam_from_world()
        rot_c2w = np.asarray(rigid.rotation.matrix()).T
        centre = -rot_c2w @ np.asarray(rigid.translation)
        rows.append((frame_idx, centre, rot_c2w))
    rows.sort(key=lambda row: row[0])
    return (
        np.array([row[0] for row in rows]),
        np.stack([row[1] for row in rows]),
        np.stack([row[2] for row in rows]),
    )


def _poses_w2c(centres: np.ndarray, rotations_c2w: np.ndarray) -> np.ndarray:
    """Assemble (N, 4, 4) world-to-cam poses, the form `trajectory_io.write_tum` expects."""
    poses_c2w = np.tile(np.eye(4), (len(centres), 1, 1))
    poses_c2w[:, :3, :3] = rotations_c2w
    poses_c2w[:, :3, 3] = centres
    return invert_poses(poses_c2w)


########################################################################
# Orientation convention selection
########################################################################


def select_orientation_mode(
    telemetry: Path, loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> str:
    """Pick the CORI/IORI composition that best matches the reconstructions, without fitting."""
    scores: dict[str, dict[str, float]] = {}
    for mode in ORIENTATION_MODES:
        for name, (frame_idx, _, rot_est) in loaded.items():
            rot_ref = reference_rotations_c2w(telemetry, frame_idx, mode)
            median = float(np.median(invariant_turn_angle_error_deg(rot_est, rot_ref)))
            scores.setdefault(mode, {})[name] = median
        row = " ".join(f"{name}={scores[mode][name]:.3f}deg" for name in loaded)
        logger.info("orientation %-16s %s", mode, row)

    best = {name: min(scores, key=lambda mode: scores[mode][name]) for name in loaded}
    if len(set(best.values())) > 1:
        # Each backend voting for a different convention means the winner is fitting backend
        # error rather than a physical property of the camera.
        logger.warning(
            "backends disagree on the orientation convention (%s) — the reference is not "
            "measuring a physical convention; treat rotation numbers as indicative only",
            best,
        )
    mode = next(iter(best.values()))
    logger.info("selected orientation mode: %s", mode)
    return mode


########################################################################
# Entry point
########################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--telemetry", type=Path, required=True, help="GPMF telemetry parquet")
    parser.add_argument(
        "--recon",
        action="append",
        required=True,
        metavar="NAME=SPARSE_DIR",
        help="named COLMAP model, repeatable; NAME becomes the method name in metrics.json",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--orientation",
        default="auto",
        choices=("auto", *ORIENTATION_MODES),
        help="'auto' selects by the fit-free invariant (recommended)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load every reconstruction first: the reference is shared, so the frame sets must match.
    loaded = {}
    for spec in args.recon:
        name, _, sparse_dir = spec.partition("=")
        if not sparse_dir:
            raise ValueError(f"--recon expects NAME=SPARSE_DIR, got {spec!r}")
        loaded[name] = load_colmap_frame_poses(sparse_dir)
        logger.info("%s: %d poses from %s", name, len(loaded[name][0]), sparse_dir)

    # A single gt.tum serves all methods, so an unequal frame set would silently score some
    # backend against the wrong frames. Refuse rather than emit a misaligned reference.
    frame_sets = {name: value[0] for name, value in loaded.items()}
    reference_name, reference_frames = next(iter(frame_sets.items()))
    for name, frames in frame_sets.items():
        if not np.array_equal(frames, reference_frames):
            raise ValueError(
                f"{name} and {reference_name} reconstructed different frames "
                f"({len(frames)} vs {len(reference_frames)}) — a shared gt.tum would misalign; "
                "re-run the backends over one identical frame set"
            )

    mode = select_orientation_mode(args.telemetry, loaded) if args.orientation == "auto" \
        else args.orientation

    # The reference is identical for every method, so write it once.
    args.results_dir.mkdir(parents=True, exist_ok=True)
    times, positions_ref, _ = sample_reference_at_frames(args.telemetry, reference_frames, mode)
    rot_ref = reference_rotations_c2w(args.telemetry, reference_frames, mode)
    write_tum(args.results_dir / "gt.tum", _poses_w2c(positions_ref, rot_ref), times)

    for name, (frame_idx, centres, rot_est) in loaded.items():
        angles, offset = relative_rotation_error_deg(rot_est, rot_ref)
        # Apply the fitted offset so the written poses are in the reference's camera frame
        # and evo's RPE compares like with like.
        rot_corrected = np.einsum("nij,jk->nik", rot_est, offset)
        write_tum(args.results_dir / f"{name}.tum", _poses_w2c(centres, rot_corrected), times)

        # Self-check: the fitted error cannot fall below the fit-free floor.
        floor = float(np.median(invariant_turn_angle_error_deg(rot_est, rot_ref)))
        median = float(np.median(angles))
        fit_ok = median < 3 * floor + 0.5
        if not fit_ok:
            logger.warning(
                "%s: camera-offset fit SUSPECT — %.3f deg against a %.3f deg invariant floor; "
                "the rotation numbers describe the fit, not the reconstruction",
                name, median, floor,
            )

        # eval_compare.py reads <name>_alignment.json back into metrics.json verbatim, so the
        # offset and the floor travel with the numbers they produced.
        sidecar = {
            "orientation_mode": mode,
            "camera_offset_rotvec_deg": np.degrees(
                Rotation.from_matrix(offset).as_rotvec()
            ).tolist(),
            "invariant_turn_angle_deg": {
                "median": floor,
                "p95": float(np.percentile(invariant_turn_angle_error_deg(rot_est, rot_ref), 95)),
            },
            "relative_rotation_deg": {
                "median": median,
                "p95": float(np.percentile(angles, 95)),
                "max": float(angles.max()),
            },
            "fit_ok": fit_ok,
            # Path length in arbitrary reconstruction units against GPS metres: the Sim(3)
            # scale is fitted away by ATE, so record it here or the scale question is lost.
            "path_length_recon": float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()),
            "path_length_gps_m": float(
                np.linalg.norm(np.diff(positions_ref, axis=0), axis=1).sum()
            ),
        }
        (args.results_dir / f"{name}_alignment.json").write_text(json.dumps(sidecar, indent=2))
        logger.info(
            "%s: rel-rot median %.3f deg (floor %.3f, %s), path %.2f vs %.2f m GPS",
            name, median, floor, "fit OK" if fit_ok else "FIT SUSPECT",
            sidecar["path_length_recon"], sidecar["path_length_gps_m"],
        )

    logger.info(
        "\nWrote %s — now run:\n  python evals/scripts/eval_compare.py --results-dir %s",
        args.results_dir, args.results_dir,
    )


if __name__ == "__main__":
    main()
