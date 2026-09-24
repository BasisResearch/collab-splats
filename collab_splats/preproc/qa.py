"""
Capture quality report: per-frame photometry and per-pair motion.

- frames: blur, laplacian, exposure_{mean,median,std}, clipped_{low,high}_frac
- pairs: n_matches, translation_px, parallax
- report-only: thresholds live in preproc.sampling.filter_frame_quality
- column evidence: docs/superpowers/specs/2026-08-20-video-quality-report-measured.md
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import blur_effect

from collab_splats.preproc.video import get_video_info, iter_frames

logger = logging.getLogger(__name__)


########################################################################
# Shared by the frame and the pair measurements
########################################################################


def analysis_gray(frame_bgr: np.ndarray, *, width: int = 480) -> np.ndarray:
    """
    Grayscale copy downscaled to `width` for scoring.

    - bounds LK flow and Laplacian cost regardless of source resolution

    Args:
        frame_bgr: (H, W, 3) uint8 BGR frame.
        width: target width in pixels; a narrower frame is passed through untouched.

    Returns:
        (h, w) uint8 grayscale, aspect ratio preserved, w <= width.
    """
    scale = min(1.0, width / frame_bgr.shape[1])
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame_bgr

    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


########################################################################
# Per frame
########################################################################


def compute_blur(gray: np.ndarray, *, h_size: int = 11) -> dict:
    """
    Blur two ways: Crete-Roffet perceptual blur and Laplacian variance.

    - opposite directions on purpose: blur higher = blurrier, laplacian higher = sharper
    - blur SATURATES at 1.0 wherever high-frequency content is sparse — flat field, smooth
      gradient and one perfectly sharp edge all score exactly 1.0
    - so high blur beside high laplacian = sparse detail, NOT a soft frame
    - that is why laplacian ships next to blur, not instead of it

    Args:
        gray: (h, w) uint8 single-channel frame.
        h_size: re-blur kernel width Crete-Roffet compares against; larger reports less
            blur. 11 is skimage's default.

    Returns:
        {'blur': float in [0, 1], 'laplacian': float unbounded}.

    Raises:
        ValueError: when `gray` is not 2-D.
    """
    # A color frame is a caller mistake, not a measurement
    # - skimage reads the channel axis as spatial and returns nan
    # - cv2.Laplacian returns a perfectly plausible number anyway
    # - so the row would read as a failed capture
    if gray.ndim != 2:
        raise ValueError(f"compute_blur expects a 2-D single-channel frame, got shape {gray.shape}")

    # Crete-Roffet re-blurs the image and measures how little changes. A frame
    # that is already blurred barely moves, so its score rises toward 1.
    perceptual = float(blur_effect(gray, h_size=h_size))

    # Laplacian variance: unbounded, higher = sharper. Runs in the opposite
    # direction to `blur` on purpose — see the docstring.
    return {"blur": perceptual, "laplacian": float(cv2.Laplacian(gray, cv2.CV_64F).var())}


def compute_exposure(gray: np.ndarray) -> dict:
    """
    Brightness distribution plus the fraction of pixels pinned at either end.

    - one 256-bin histogram serves all five numbers

    Args:
        gray: (h, w) uint8 single-channel frame, NATIVE resolution — see compute_frame_quality.

    Returns:
        {'exposure_mean', 'exposure_median', 'exposure_std', 'clipped_low_frac',
        'clipped_high_frac'}, all float.
    """
    # float64: cv2.calcHist returns float32, and a 1080x1920 frame's 2.07M counts
    # do not survive it — the clipping fractions come out ~3e-8 off.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    n = hist.sum()
    levels = np.arange(256.0)
    cumulative = np.cumsum(hist)

    # Mean and median together: they separate when a small bright region
    # (a window, a lamp) drags the mean while most of the scene stays dark.
    exposure_mean = float((hist * levels).sum() / n)

    # Histogram median must match np.median on an even pixel count
    # - np.median averages the two central values; searchsorted alone gives the lower
    k_lo, k_hi = (int(n) - 1) // 2, int(n) // 2
    exposure_median = float(
        (np.searchsorted(cumulative, k_lo, side="right") + np.searchsorted(cumulative, k_hi, side="right")) / 2
    )

    # Contrast. A low std is a flat, textureless frame regardless of brightness.
    exposure_std = float(np.sqrt((hist * (levels - exposure_mean) ** 2).sum() / n))

    # Clipped pixels are destroyed data, not merely dark or bright data: 0 and 255
    # are the two values where the sensor recorded nothing recoverable.
    clipped_low_frac = float(hist[0] / n)
    clipped_high_frac = float(hist[255] / n)

    return {
        "exposure_mean": exposure_mean,
        "exposure_median": exposure_median,
        "exposure_std": exposure_std,
        "clipped_low_frac": clipped_low_frac,
        "clipped_high_frac": clipped_high_frac,
    }


def compute_frame_quality(bgr: np.ndarray, *, analysis_width: int = 480, blur_h_size: int = 11) -> dict:
    """
    Photometry for one BGR frame: blur and exposure together.

    - blur and laplacian read a gray downscaled to analysis_width; exposure reads native
    - so blur and laplacian are not comparable across videos whose width straddles it

    Args:
        bgr: (H, W, 3) uint8 BGR frame.
        analysis_width: width blur and laplacian are computed at; exposure ignores it.
        blur_h_size: forwarded to compute_blur as `h_size`.

    Returns:
        compute_blur's keys merged with compute_exposure's, one flat dict.
    """
    # Exposure reads the NATIVE gray: downscaling averages scattered saturated
    # pixels out of existence, so clipping fractions read 0.0 on a blown capture.
    gray_native = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    exposure = compute_exposure(gray_native)

    # Blur and laplacian read the analysis-width gray: cheaper, tied to analysis_width
    gray_small = analysis_gray(bgr, width=analysis_width)
    blur = compute_blur(gray_small, h_size=blur_h_size)

    return {**blur, **exposure}


########################################################################
# Per pair
########################################################################


def detect_orb(gray: np.ndarray, *, n_features: int = 1000) -> tuple[tuple, np.ndarray | None]:
    """
    ORB keypoints and descriptors for one grayscale frame.

    - split from matching so a video run detects each frame once

    Args:
        gray: (h, w) uint8 single-channel frame.
        n_features: ORB feature cap.

    Returns:
        (keypoints, descriptors). descriptors is None on a frame with no detectable
        features — a fact about the frame, not an error.
    """
    return cv2.ORB_create(nfeatures=n_features).detectAndCompute(gray, None)


def compute_pair_motion(feat_a: tuple, feat_b: tuple, *, ransac_thresh_px: float = 3.0) -> dict:
    """
    Match two frames' ORB features and measure the motion between them.

    - pixels are those of the grid detect_orb ran on (analysis_gray in the report)
    - unmeasurable is nan, never 0.0, which would read as a still camera
    - parallax in [0, 1]: one minus the homography/fundamental inlier ratio; ~0 = no depth
    - crossCheck cannot detect unrelated frames: a scene cut reads as large motion

    Args:
        feat_a: (keypoints, descriptors) from detect_orb for the earlier frame.
        feat_b: (keypoints, descriptors) for the later frame.
        ransac_thresh_px: inlier threshold for both fits; looser lowers parallax.

    Returns:
        {'n_matches': int, 'translation_px': float, 'parallax': float}. translation_px is the
        median match displacement, nan with no matches; parallax is nan below 8 matches or
        when either fit fails.
    """
    kp_a, desc_a = feat_a
    kp_b, desc_b = feat_b

    # A featureless frame yields no descriptors at all. Report zero matches rather
    # than raise: an unmatchable pair is a fact about the video, not an error.
    unmeasured = {"n_matches": 0, "translation_px": float("nan"), "parallax": float("nan")}
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return unmeasured

    # ORB descriptors are binary, hence Hamming. crossCheck keeps only mutual best
    # matches, which removes the need for a Lowe ratio test.
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_a, desc_b)
    if not matches:
        return unmeasured

    # Pull the pixel coordinates behind each match into two aligned Nx2 arrays
    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2)

    # Robust translation, fail-closed parallax
    # - translation is a median over per-match displacement, so bad matches cannot drag it
    # - parallax starts nan, filled in only if every fit below succeeds
    row = {
        "n_matches": len(matches),
        "translation_px": float(np.median(np.linalg.norm(pts_b - pts_a, axis=1))),
        "parallax": float("nan"),
    }
    if len(pts_a) < 8:
        return row

    # Fit both models to the same correspondences
    # - H can only explain a plane or a pure rotation
    # - F can additionally explain translation through depth
    # - so the gap between their inlier counts IS the depth information in the pair
    try:
        _, h_inliers = cv2.findHomography(pts_a, pts_b, cv2.USAC_MAGSAC, ransac_thresh_px)
        _, f_inliers = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, ransac_thresh_px)
    except cv2.error:
        return row

    n_h = int(h_inliers.sum()) if h_inliers is not None else 0
    n_f = int(f_inliers.sum()) if f_inliers is not None else 0

    # A degenerate pair must not produce negative parallax
    # - no F inliers = the pair is unexplained by any two-view geometry
    # - min() guards the case where H out-fits F on a degenerate pair
    if n_f == 0:
        return row

    row["parallax"] = float(1.0 - min(1.0, n_h / n_f))

    return row


########################################################################
# Whole video
########################################################################


def _measure_photometry_and_motion(
    args: tuple,
    *,
    analysis_width: int,
    blur_h_size: int,
    n_features: int,
    ransac_thresh_px: float,
) -> tuple[list[dict], list[dict]]:
    """
    Measure one contiguous frame range; returns (frame rows, pair rows).

    - decodes `stride` lead-in frames before `emit_from` so boundary pairs have a partner
    - emits rows from `emit_from` on only, so ranges tile the video exactly once
    - module-level so ProcessPoolExecutor can pickle it

    Args:
        args: (video_path, start, count, emit_from, stride) for this range.
        analysis_width: width blur, laplacian and ORB run at.
        blur_h_size: Crete-Roffet re-blur kernel width.
        n_features: ORB feature cap per frame.
        ransac_thresh_px: inlier threshold for the parallax fits.

    Returns:
        (frame rows, pair rows) for the frames this range owns.
    """
    video_path, start, count, emit_from, stride = args

    # Pin cv2 and BLAS to one thread per worker
    # - unpinned, process fan-out oversubscribes the cores and runs slower than serial
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)

    frame_rows: list[dict] = []
    pair_rows: list[dict] = []

    # Hold only the ORB features still owed a partner: stride + 1 frames at a
    # time, so memory does not track range length.
    pending: dict[int, tuple] = {}

    for idx, bgr in iter_frames(video_path, start=start, count=count):
        # Lead-in frames belong to the previous range
        # - that range is already measuring them
        # - decoded here only to be somebody's partner
        # - so skip the photometry rather than compute a row and drop it
        if idx >= emit_from:
            frame_rows.append(
                {"frame_idx": idx, **compute_frame_quality(bgr, analysis_width=analysis_width, blur_h_size=blur_h_size)}
            )

        # Motion against the frame one stride back, once one exists
        # - first pair this can fire on is (emit_from - stride, emit_from)
        # - that is exactly the boundary pair the lead-in exists to reach
        # - so no pair owned by the previous range is emitted twice
        gray_small = analysis_gray(bgr, width=analysis_width)
        pending[idx] = detect_orb(gray_small, n_features=n_features)
        partner = idx - stride

        if partner in pending:
            pair_rows.append(
                {
                    "frame_idx_a": partner,
                    "frame_idx_b": idx,
                    **compute_pair_motion(pending[partner], pending[idx], ransac_thresh_px=ransac_thresh_px),
                }
            )

            del pending[partner]

    return frame_rows, pair_rows


def _ranges(total: int, *, workers: int, stride: int) -> list[tuple[int, int | None, int]]:
    """
    (start, count, emit_from) per worker; each range decodes a stride-frame lead-in.

    Args:
        total: frames in the video.
        workers: ranges wanted; a short video gets one.
        stride: pair spacing, the lead-in length.

    Returns:
        Ascending ranges that tile [0, total) exactly once by emit_from.
        count is None on the single whole-video range: decode to the end.
    """
    if workers == 1 or total <= stride * 2:
        return [(0, None, 0)]

    per = total // workers
    out = []
    for k in range(workers):
        emit_from = k * per
        start = max(emit_from - stride, 0)
        end = total if k == workers - 1 else (k + 1) * per
        out.append((start, end - start, emit_from))
    return out


def compute_video_quality(
    video_path: str | Path,
    *,
    motion_stride: int | None = None,
    workers: int = 1,
    analysis_width: int = 480,
    blur_h_size: int = 11,
    n_features: int = 1000,
    ransac_thresh_px: float = 3.0,
) -> dict:
    """
    Per-frame photometry and per-pair motion across a whole video.

    - report-only: filter_frame_quality turns these columns into a keep mask

    Args:
        video_path: source video; every frame is decoded and scored.
        motion_stride: frames between the two members of each pair, >= 1; None = round(fps).
        workers: contiguous frame ranges processed in parallel; 1 is serial.
        analysis_width: width blur, laplacian and ORB run at.
        blur_h_size: Crete-Roffet re-blur kernel width.
        n_features: ORB feature cap per frame.
        ransac_thresh_px: inlier threshold for the parallax fits.

    Returns:
        {"video": {path, mtime, **get_video_info}, "params": {motion_stride and the four tuning kwargs},
        "frames": {column: list per frame}, "pairs": {column: list per pair}}.

    Raises:
        FileNotFoundError: the video does not exist.
        ValueError: the video cannot be probed or decodes no frames.
    """
    # `is not None`, not truthiness
    # - 0 is an explicit value; truthiness falls through to the fps default, stride 30
    # - a negative stride is worse than wrong: the partner index runs forward
    # - nothing is ever retired from `pending`, so it grows with the video
    if motion_stride is not None and motion_stride < 1:
        raise ValueError(f"motion_stride must be >= 1, got {motion_stride}")

    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    video_path = Path(video_path)
    info = get_video_info(str(video_path))
    stride = int(motion_stride) if motion_stride is not None else max(1, round(info["fps"]))

    # Announce the work before the first decode
    # - a multi-minute silent run is indistinguishable from a hung one
    logger.info(
        "video quality: %s — %s frames @ %.2f fps, %sx%s, stride %d",
        video_path.name,
        info["total_frames"],
        info["fps"],
        info["width"],
        info["height"],
        stride,
    )
    # One range per worker
    # - ranges tile the video exactly once
    # - each decodes a `stride`-frame lead-in from its predecessor for the boundary pairs
    # - nothing is emitted from the lead-in
    total = info["total_frames"]
    ranges = [
        (str(video_path), start, count, emit_from, stride)
        for start, count, emit_from in _ranges(total, workers=workers, stride=stride)
    ]

    # Bind the tuning once; partial of a module-level function still pickles
    tuning = {
        "analysis_width": analysis_width,
        "blur_h_size": blur_h_size,
        "n_features": n_features,
        "ransac_thresh_px": ransac_thresh_px,
    }
    measure = partial(_measure_photometry_and_motion, **tuning)

    started = time.perf_counter()

    if len(ranges) == 1:
        results = [measure(ranges[0])]
    else:
        with ProcessPoolExecutor(len(ranges)) as pool:
            results = list(pool.map(measure, ranges))

    # pool.map yields in submission order, and the ranges were built in ascending
    # order, so concatenating is the whole merge.
    frame_rows = [row for rows, _ in results for row in rows]
    pair_rows = [row for _, rows in results for row in rows]

    if not frame_rows:
        raise ValueError(f"video quality: no frames decoded from {video_path}")

    # Contiguity check: a bad seek silently shifts every index
    # - int comparison, so rounding cannot false-alarm
    idxs = [row["frame_idx"] for row in frame_rows]

    if len(ranges) > 1 and idxs != list(range(idxs[0], idxs[0] + len(idxs))):
        raise ValueError(
            f"video quality: frame indices are not contiguous across ranges "
            f"({len(idxs)} rows spanning {idxs[0]}..{idxs[-1]}) — input seek is "
            "unreliable on this file. Re-run with workers=1."
        )

    # Columnar report: one list per measurement, aligned by row order
    frame_keys = (
        "frame_idx",
        "blur",
        "laplacian",
        "exposure_mean",
        "exposure_median",
        "exposure_std",
        "clipped_low_frac",
        "clipped_high_frac",
    )
    frames = {k: [row[k] for row in frame_rows] for k in frame_keys}

    # Pair columns; nan -> null on the only two that can be non-finite
    # - json.dumps writes a bare NaN that no strict parser accepts
    # - np.nan_to_num is not the fix: its 0.0 fill reads as "no motion"
    pairs = {k: [r[k] for r in pair_rows] for k in ("frame_idx_a", "frame_idx_b", "n_matches")}
    for k in ("translation_px", "parallax"):
        pairs[k] = [None if np.isnan(r[k]) else r[k] for r in pair_rows]

    report = {
        "video": {"path": str(video_path), "mtime": video_path.stat().st_mtime, **info},
        "params": {"motion_stride": stride, **tuning},
        "frames": frames,
        "pairs": pairs,
    }

    # Throughput, not just a count: it is the number that tells a reader
    # whether a long run is progressing or degrading.
    elapsed = max(time.perf_counter() - started, 1e-9)
    logger.info(
        "video quality: %d frames, %d pairs, stride %d — %.1fs (%.1f frames/s)",
        len(frames["frame_idx"]),
        len(pair_rows),
        stride,
        elapsed,
        len(frames["frame_idx"]) / elapsed,
    )

    return report


def load_video_quality(
    video_path: str | Path,
    report_path: str | Path,
    *,
    workers: int = 1,
    motion_stride: int | None = None,
) -> dict:
    """
    The quality report at report_path, measuring and writing it first if absent.

    - reuse is by existence, the same rule images/ follows
    - an existing file is reused, never recomputed; one without "frames" raises

    Args:
        video_path: source video, used only when report_path is missing.
        report_path: the JSON report; read as-is when it already exists.
        workers: parallel decode ranges, forwarded to compute_video_quality.
        motion_stride: pair spacing, forwarded to compute_video_quality.

    Returns:
        The report dict compute_video_quality produces.

    Raises:
        FileNotFoundError: no cached report and the video does not exist.
        ValueError: a cached report without "frames", or compute_video_quality rejects the input.
    """
    report_path = Path(report_path)

    if report_path.exists():
        logger.info("video quality: reusing %s", report_path)
        report = json.loads(report_path.read_text())

        if "frames" not in report:
            raise ValueError(f"{report_path} is a stale video-quality report (no 'frames'); delete it and re-run")
        return report

    report = compute_video_quality(video_path, motion_stride=motion_stride, workers=workers)

    # Write via a temp file so an interrupted run leaves no partial report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(report, indent=2))
    os.replace(tmp_path, report_path)
    logger.info("video quality: wrote %s (%.1f kB)", report_path, report_path.stat().st_size / 1000)
    return report
