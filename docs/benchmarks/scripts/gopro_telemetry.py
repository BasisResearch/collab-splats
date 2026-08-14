"""Turn a GoPro's GPMF telemetry into a pose reference sampled at reconstructed frames.

Benchmark evidence, not package code — see `docs/benchmarks/scripts/README.md`.  Nothing in
`collab_splats/` or `evals/` imports this module and nothing runs it in CI.

Rotation comes from the camera's orientation quaternion stream, translation from GPS
projected onto a local ENU plane.  Nothing here computes a metric: the output feeds
`evals/trajectory_io.write_tum` and from there the existing `evals/metrics.py` functions.

Two properties of this reference are not clean and are handled explicitly rather than
assumed away:

* **Electronic stabilisation is on and is not negligible.**  A GoPro records two orientation
  streams: `CORI` (the physical attitude of the camera body) and `IORI` (the HyperSmooth warp
  applied to each image).  The model sees the *stabilised* image, so scoring against raw CORI
  compares a reconstruction to an orientation no frame actually shows.  On the 2026-08-14
  scene IORI reached 25.7 deg, median 4.6 deg.  Which composition of the two describes the
  stabilised image is undocumented, so all five candidates are exposed via `mode=` and the
  caller picks one by measurement — see the sibling `rotation_alignment.py` for the fit-free
  quantity that makes that choice honest.
* **GPS is consumer grade.**  Relative precision over a few hundred metres is fair; altitude
  is markedly worse than horizontal and absolute position is metre-scale.  Translation
  metrics from this reference are indicative, not ground truth — on the 2026-08-14 scene two
  reconstructions agreed with *each other* five times more tightly than either agreed with
  GPS.  Rotation is the solid half.

Measured with this module: `docs/benchmarks/2026-08-14-loger-vs-omega-gopro.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

########################################################################
# Constants
########################################################################

# NTSC 60p.  GoPro emits one CORI/IORI sample per video frame, so a frame index maps to
# telemetry time through the video's own rate — no resampling, and no guessing at alignment.
GOPRO_FPS = 60000.0 / 1001.0

# WGS-84 equatorial radius, for the local equirectangular projection below.
EARTH_RADIUS_M = 6378137.0

# The five ways CORI and IORI can compose into "the orientation of the image the model saw".
# Only measurement can say which is right; see module docstring.
ORIENTATION_MODES = ("cori", "cori_iori", "cori_iori_conj", "iori_cori", "iori_conj_cori")


########################################################################
# Quaternion helpers (wxyz convention, matching GPMF)
########################################################################


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (N, 4) wxyz quaternion arrays."""
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of an (N, 4) wxyz quaternion array."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


########################################################################
# Stream loading
########################################################################


def load_image_orientation(
    parquet: Path | str, mode: str = "cori"
) -> tuple[np.ndarray, np.ndarray]:
    """Load one candidate image-orientation convention as (times_s, quats_wxyz).

    Args:
        parquet: telemetry table with `source_time` and `cori_*` / `iori_*` columns.
        mode: one of `ORIENTATION_MODES`; see the module docstring on why this is a choice.
    """
    if mode not in ORIENTATION_MODES:
        raise ValueError(f"unknown orientation mode {mode!r}; expected one of {ORIENTATION_MODES}")
    df = pd.read_parquet(parquet)
    # CORI and IORI share a sampling grid, so a single dropna across both keeps them paired.
    sub = df[
        ["source_time", "cori_w", "cori_x", "cori_y", "cori_z",
         "iori_w", "iori_x", "iori_y", "iori_z"]
    ].dropna()
    times = sub["source_time"].to_numpy()
    cori = sub[["cori_w", "cori_x", "cori_y", "cori_z"]].to_numpy()
    iori = sub[["iori_w", "iori_x", "iori_y", "iori_z"]].to_numpy()
    quats = {
        "cori": cori,
        "cori_iori": _quat_multiply(cori, iori),
        "cori_iori_conj": _quat_multiply(cori, _quat_conjugate(iori)),
        "iori_cori": _quat_multiply(iori, cori),
        "iori_conj_cori": _quat_multiply(_quat_conjugate(iori), cori),
    }[mode]
    # Composing two streams leaves small norm drift that downstream rotation conversion rejects.
    return times, quats / np.linalg.norm(quats, axis=1, keepdims=True)


def load_gps_enu(parquet: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load GPS as (times_s, ENU metres) with the origin at the track's mean lat/lon."""
    df = pd.read_parquet(parquet)
    sub = df[["source_time", "gps_lat", "gps_lon", "gps_alt"]].dropna()
    times = sub["source_time"].to_numpy()
    lat, lon, alt = (sub[c].to_numpy() for c in ("gps_lat", "gps_lon", "gps_alt"))
    # Equirectangular projection about the mean latitude — the curvature error over a track
    # of a few hundred metres is far below the GPS noise this reference is already limited by.
    lat0, lon0 = lat.mean(), lon.mean()
    east = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    north = np.radians(lat - lat0) * EARTH_RADIUS_M
    return times, np.stack([east, north, alt - alt[0]], axis=1)


def sample_reference_at_frames(
    parquet: Path | str, frame_idx: np.ndarray, mode: str = "cori"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the telemetry reference at the video frames a reconstruction actually kept.

    Args:
        parquet: telemetry table.
        frame_idx: source frame indices, e.g. parsed from COLMAP image names.
        mode: orientation convention, one of `ORIENTATION_MODES`.

    Returns:
        (times_s, positions_enu (N, 3), quats_wxyz (N, 4)).
    """
    times = np.asarray(frame_idx, dtype=np.float64) / GOPRO_FPS
    t_quat, quats = load_image_orientation(parquet, mode)
    # Nearest sample rather than slerp: the stream runs at ~60 Hz, so the worst-case error is
    # half a sample (~8 ms) — far below the rotation differences being measured, and slerp
    # would add interpolation machinery for accuracy that is already there.
    nearest = np.abs(t_quat[None, :] - times[:, None]).argmin(axis=1)
    t_gps, enu = load_gps_enu(parquet)
    # GPS is a smooth position signal sampled well below the orientation rate, so per-axis
    # linear interpolation is the right treatment here where nearest would quantise visibly.
    positions = np.stack([np.interp(times, t_gps, enu[:, k]) for k in range(3)], axis=1)
    return times, positions, quats[nearest]


def reference_rotations_c2w(
    parquet: Path | str, frame_idx: np.ndarray, mode: str = "cori"
) -> np.ndarray:
    """Telemetry orientations as (N, 3, 3) camera-to-world matrices.

    The transpose here is **measured, not assumed**: optimising the camera-frame offset over
    all four transpose pairings of estimate and reference gave 0.59 deg for `est` against
    `ref^T`, and 0.76 / 7.1 / 7.4 deg for the other three, against a 0.29 deg invariant floor.
    So the GPMF quaternion is world-from-camera where a reconstruction's camera-to-world
    rotation is its transpose.
    """
    _, _, quats_wxyz = sample_reference_at_frames(parquet, frame_idx, mode)
    # scipy takes xyzw; GPMF is wxyz.
    matrices = Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    return np.transpose(matrices, (0, 2, 1))
