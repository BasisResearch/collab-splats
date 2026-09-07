"""
Video capture quality: how good is the source footage, per frame and per pair.

Two measurement families:

    Photometry — per frame: is this frame sharp and correctly exposed?
                 blur, laplacian, exposure_{mean,median,std}, clipped_{low,high}_frac
    Motion     — per pair:  how far did the camera move between two frames?
                 n_matches, translation_px, parallax

REPORT-ONLY. Nothing here selects, rejects, ranks or scores a frame against a
threshold. Selection policy lives in preproc.sampling.filter_frame_quality.
Measured evidence for every column is in
docs/superpowers/specs/2026-08-20-video-quality-report-measured.md.
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
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
    # A colour frame is a caller mistake, not a measurement
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
    - ~98x faster than the numpy path it replaces (0.34 ms vs 33.5 ms on 1920x1080), equal output

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
    # - np.median averages the two central order statistics
    # - searchsorted(cumulative, n/2) alone returns only the lower one
    # - measured 127.5 off on a two-pixel frame, so take both and average
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

    - **blur is NOT comparable across videos whose source width straddles `analysis_width`**
    - wider sources are downscaled further and read sharper
    - measured on data/tutorial: blur 0.2128 at 480 px against 0.2772 at 1024 px, 30% spread
      on identical frames
    - laplacian and exposure are native-resolution, so unaffected

    Args:
        bgr: (H, W, 3) uint8 BGR frame.
        analysis_width: width blur is measured at; exposure ignores it.
        blur_h_size: forwarded to compute_blur as `h_size`.

    Returns:
        compute_blur's keys merged with compute_exposure's, one flat dict.
    """
    # Exposure reads the NATIVE gray: downscaling averages scattered saturated
    # pixels out of existence, so clipping fractions read 0.0 on a blown capture.
    gray_native = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    exposure = compute_exposure(gray_native)

    # Blur reads the analysis-width gray, not native
    # - 7.6x cheaper: 300.4 ms native vs 39.4 ms at 853x480, one 1080x1920 tutorial frame
    # - paid for in cross-video comparability — see the docstring
    gray_small = analysis_gray(bgr, width=analysis_width)
    blur = compute_blur(gray_small, h_size=blur_h_size)

    return {**blur, **exposure}


########################################################################
# Per pair
########################################################################


def detect_orb(gray: np.ndarray, *, n_features: int = 1000) -> tuple[tuple, np.ndarray | None]:
    """
    ORB keypoints and descriptors for one grayscale frame.

    - split from matching so a video run detects each frame ONCE
    - every frame is one pair's partner and the next pair's current frame, so a combined
      detect-and-match ran ORB twice per frame
    - measured 1.34x on the pair loop over 400 tutorial frames at stride 24, same matches

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

    - both measures are in the pixels of whatever grid detect_orb ran on
    - the report feeds analysis_gray output, so shipped columns are analysis-grid pixels:
      comparable within a report, NOT across videos of differing width
    - rescaling does NOT convert them — ORB detects different keypoints per resolution
      (data/tutorial 1080x1920 at factor 2.25: native-over-analysis 2.37 on one pair, 3.28 on another)
    - unmeasurable is nan, never 0.0 — 0.0 reads as "camera held perfectly still", the opposite
      conclusion
    - **crossCheck does not bound whether the two frames show the same scene.** Mutual-best returns
      a full match set on unrelated frames: two independent noise images gave 373 matches at 92 px
      median displacement against 539 at 17 px for a true 17 px shift, so a scene cut reads as large
      CONFIDENT motion and neither n_matches nor a nan reveals it. Descriptor distance separates the
      cases (median Hamming 80 against 32) and is not returned here.

    Args:
        feat_a: (keypoints, descriptors) from detect_orb for the EARLIER frame.
        feat_b: (keypoints, descriptors) for the later frame.
        ransac_thresh_px: inlier threshold shared by the homography and fundamental fits behind
            parallax. A first-order lever, not a detail — over 99 tutorial pairs, 1.0 against 3.0
            moves parallax by 0.19 on average and reorders the pairs (Spearman 0.708), while 3.0
            against 5.0 barely does (0.053, 0.945). Looser lets a homography explain more, so
            parallax falls monotonically.

    Returns:
        {'n_matches': int, 'translation_px': float, 'parallax': float}. translation_px is the
        median match displacement; parallax is one minus the homography/fundamental inlier ratio,
        so near 0 means the pair carries no depth information. translation_px is nan with no
        matches; parallax is additionally nan below 8 correspondences (the linear 8-point
        minimum), when either fit raises (USAC asserts on configurations it cannot estimate at any
        size — measured on a real 720-correspondence pair whose matches were 97.5%
        zero-displacement), and when the fundamental fit finds no inliers at all.
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


def _measure_photometry_and_motion(args: tuple) -> tuple[list[dict], list[dict]]:
    """
    Measure one contiguous frame range in its own process; returns (frame rows, pair rows).

    - Decoding starts `stride` frames before `emit_from` so the pairs straddling
      the range boundary have their partner, but only rows from `emit_from` on
      are returned. Ranges therefore tile the video exactly once and the caller
      concatenates them — nothing to deduplicate.
    - Runs at module scope because ProcessPoolExecutor pickles by qualified name.
    """
    video_path, start, count, emit_from, stride = args

    # THE THREAD PIN IS LOAD-BEARING
    # - cv2 and numpy each fan out over every core
    # - so the "serial" baseline is already parallel, and process fan-out oversubscribes
    # - unpinned, this measured 0.67x — SLOWER than serial
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
            frame_rows.append({"frame_idx": idx, **compute_frame_quality(bgr)})

        # Motion against the frame one stride back, once one exists
        # - first pair this can fire on is (emit_from - stride, emit_from)
        # - that is exactly the boundary pair the lead-in exists to reach
        # - so no pair owned by the previous range is emitted twice
        gray_small = analysis_gray(bgr)
        pending[idx] = detect_orb(gray_small)
        partner = idx - stride

        if partner in pending:
            pair_rows.append(
                {
                    "frame_idx_a": partner,
                    "frame_idx_b": idx,
                    **compute_pair_motion(pending[partner], pending[idx]),
                }
            )

            del pending[partner]

    return frame_rows, pair_rows


def compute_video_quality(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    motion_stride: int | None = None,
    workers: int = 1,
) -> dict:
    """
    Measure per-frame photometry and per-pair motion across a whole video.

    - report-only: carries measurements, never verdicts
    - filter_frame_quality is what turns these columns into a keep mask
    - HYPOTHESIS, not a measurement: the shortfall against the standalone harness's 3.04x
      prediction is likely ffmpeg — each worker pins cv2, OMP and OpenBLAS to one thread but
      not ffmpeg's own decode threads, so the serial baseline already fans out over cores and
      has less headroom to win back

    Args:
        video_path: source video, decoded in full — every frame is measured.
        output_path: where to write the report as JSON; None returns it without
            touching disk.
        motion_stride: frames between the two members of each measured pair, >= 1.
            None means round(fps) — one second of video, the pair spacing a
            reconstruction sees under the shipping fps: 1.0 sampling rate.
        workers: decode and measure this many contiguous frame ranges in parallel;
            1 is serial. Measured end to end on a 2388-frame 1080x1920 video
            (96-core host): 180.1 s at 1 against 93.9 s at 4, so 1.92x, with
            byte-identical reports. NOT auto-derived — see the design doc, 5.2.

    Returns:
        {"available": True, "video": {path, mtime, **get_video_info},
        "params": {"motion_stride": int}, "frames": {column -> list, one entry per
        source frame}, "pairs": {column -> list, one entry per measured pair}}.
        An unreadable video gives {"available": False, "reason": str} instead.
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
    stride = int(motion_stride) if motion_stride is not None else max(1, round(info["fps"] or 1))

    # Announce the work before the first decode
    # - a multi-minute silent run is indistinguishable from a hung one
    # - %s on the ints so a None from the probe cannot crash the log line
    logger.info(
        "video quality: %s — %s frames @ %.2f fps, %sx%s, stride %d",
        video_path.name,
        info["total_frames"],
        info["fps"] or 0.0,
        info["width"],
        info["height"],
        stride,
    )
    # One range per worker
    # - ranges tile the video exactly once
    # - each decodes a `stride`-frame lead-in from its predecessor for the boundary pairs
    # - nothing is emitted from the lead-in
    total = info["total_frames"]

    if workers == 1 or total <= stride * 2:
        ranges = [(str(video_path), 0, None, 0, stride)]
    else:
        per = total // workers
        ranges = []
        for k in range(workers):
            emit_from = k * per
            start = max(emit_from - stride, 0)
            end = total if k == workers - 1 else (k + 1) * per
            ranges.append((str(video_path), start, end - start, emit_from, stride))

    started = time.perf_counter()

    if len(ranges) == 1:
        results = [_measure_photometry_and_motion(ranges[0])]
    else:
        with ProcessPoolExecutor(len(ranges)) as pool:
            results = list(pool.map(_measure_photometry_and_motion, ranges))

    # pool.map yields in submission order, and the ranges were built in ascending
    # order, so concatenating is the whole merge.
    frame_rows = [row for rows, _ in results for row in rows]
    pair_rows = [row for _, rows in results for row in rows]

    # Cheap integrity check on the seeks
    # - input seek (-ss before -i) lands on a keyframe and counts forward
    # - broken timestamps land it off by a few frames -> every report index quietly wrong
    # - contiguity catches that as a gap, a repeat or a reordering
    # - compared as ints, not measured floats, so rounding cannot false-alarm
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

    if not frames["frame_idx"]:
        # Name the actual condition. "no frames decoded" against a path that is
        # not there sends a reader hunting a codec problem instead of a typo.
        missing = "file does not exist" if not video_path.exists() else "no frames decoded"
        report = {"available": False, "reason": f"{missing}: {video_path}"}
    else:
        report = {
            "available": True,
            "video": {"path": str(video_path), "mtime": video_path.stat().st_mtime, **info},
            "params": {"motion_stride": stride},
            "frames": frames,
            "pairs": {
                "frame_idx_a": [r["frame_idx_a"] for r in pair_rows],
                "frame_idx_b": [r["frame_idx_b"] for r in pair_rows],
                "n_matches": [r["n_matches"] for r in pair_rows],
                # nan -> null on the only two columns that can be non-finite
                # - json.dumps writes a bare NaN that no strict parser accepts
                # - np.nan_to_num is not the fix: its 0.0 fill reads as "no motion"
                # - that is the opposite of "this pair failed to match"
                "translation_px": [None if np.isnan(r["translation_px"]) else r["translation_px"] for r in pair_rows],
                "parallax": [None if np.isnan(r["parallax"]) else r["parallax"] for r in pair_rows],
            },
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

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        logger.info("video quality: wrote %s (%.1f kB)", output_path, output_path.stat().st_size / 1000)
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
    - so a re-run of a scene never re-measures, and nothing needs a staleness check

    Args:
        video_path: source video, measured only when report_path is missing.
        report_path: the JSON report; read as-is when it already exists.
        workers: parallel decode ranges, forwarded to compute_video_quality.
        motion_stride: pair spacing, forwarded to compute_video_quality.

    Returns:
        The report dict compute_video_quality produces.
    """
    report_path = Path(report_path)

    if report_path.exists():
        logger.info("video quality: reusing %s", report_path)
        return json.loads(report_path.read_text())

    return compute_video_quality(
        video_path,
        output_path=report_path,
        motion_stride=motion_stride,
        workers=workers,
    )
