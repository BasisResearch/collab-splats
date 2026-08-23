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

    - Bounds LK flow and Laplacian cost regardless of source resolution.
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

    - `blur` is [0, 1], higher = blurrier. `laplacian` is unbounded, higher =
      sharper. Opposite directions on purpose.
    - `blur` SATURATES at 1.0 on any frame with little high-frequency content to
      destroy — a flat field, a smooth gradient and a single perfectly sharp edge
      all score exactly 1.0 — so a high blur beside a high laplacian means detail
      is sparse, not that the frame is soft. That is why laplacian ships next to
      it, not instead.
    - h_size: width of the re-blur kernel Crete-Roffet compares against. Larger
      reports less blur; 11 is skimage's default.
    """
    # A colour frame is a caller mistake, not a measurement: skimage reads the
    # channel axis as spatial and returns nan while cv2.Laplacian returns a
    # perfectly plausible number, so the row would read as a failed capture.
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

    - One 256-bin histogram serves all five numbers: ~98x faster than the numpy
      path it replaces (0.34 ms vs 33.5 ms on 1920x1080), and equal to it.
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

    # np.median averages the two central order statistics on an even pixel count.
    # searchsorted(cumulative, n/2) alone returns only the lower one — measured
    # 127.5 off on a two-pixel frame — so take both and average.
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

    - **The blur column is not comparable across videos whose source width
      straddles `analysis_width`.** Wider sources are downscaled further and read
      sharper — measured on data/tutorial, blur is 0.2128 at 480 px against
      0.2772 at 1024 px, a 30% spread on identical frames. laplacian and exposure
      are native-resolution and unaffected.
    """
    # Exposure reads the NATIVE gray: downscaling averages scattered saturated
    # pixels out of existence, so clipping fractions read 0.0 on a blown capture.
    gray_native = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    exposure = compute_exposure(gray_native)

    # Blur reads the analysis-width gray: 7.6x cheaper (300.4 ms native against
    # 39.4 ms at 853x480 on one 1080x1920 tutorial frame), paid for in
    # cross-video comparability — see the docstring.
    gray_small = analysis_gray(bgr, width=analysis_width)
    blur = compute_blur(gray_small, h_size=blur_h_size)

    return {**blur, **exposure}


########################################################################
# Per pair
########################################################################


def detect_orb(gray: np.ndarray, *, n_features: int = 1000) -> tuple[tuple, np.ndarray | None]:
    """
    ORB keypoints and descriptors for one grayscale frame.

    - Detection is split from matching so a video run detects each frame ONCE:
      every frame is the partner of one pair and the current frame of the next,
      so a combined detect-and-match call ran ORB twice per frame. Measured 1.34x
      on the pair loop over 400 tutorial frames at stride 24, same matches.
    - desc is None on a frame with no detectable features; that is a fact about
      the frame, not an error.
    """
    return cv2.ORB_create(nfeatures=n_features).detectAndCompute(gray, None)


def match_descriptors(feat_a: tuple, feat_b: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Mutually-matched keypoint coordinates between two detect_orb results, as Nx2 float32.

    - crossCheck makes both sides injective, which is what the downstream RANSAC
      wants, but it bounds nothing about whether the frames show the same scene:
      mutual-best still returns a full set of matches on unrelated frames.
    - Measured on two independent noise images: 373 matches at 92 px median
      displacement, against 539 at 17 px for a true 17 px shift. A scene cut
      therefore reads as large CONFIDENT motion, and neither the match count nor
      a nan reveals it. Descriptor distance is what separates the two cases
      (median Hamming 80 against 32) and this function does not return it.
    """
    kp_a, desc_a = feat_a
    kp_b, desc_b = feat_b

    # A featureless frame yields no descriptors at all. Return empty rather than
    # raise: zero matches is a fact about the video, not an error.
    empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if desc_a is None or desc_b is None:
        return empty

    # ORB descriptors are binary, hence Hamming. crossCheck keeps only mutual
    # best matches, which removes the need for a Lowe ratio test.
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_a, desc_b)
    if not matches:
        return empty

    # Pull the pixel coordinates behind each match into two aligned Nx2 arrays
    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2)

    return pts_a, pts_b


def compute_translation(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Median match displacement in pixels — how far image content moved between the pair.

    - Pixels of whatever grid detect_orb ran on. The report feeds it
      analysis_gray output, so the shipped column is analysis-grid pixels.
    - **It does not convert to source pixels by scaling**: ORB detects different
      keypoints at different resolutions, so the ratio is not the resize factor.
      Measured on data/tutorial (1080x1920 portrait, factor 2.25),
      native-over-analysis is 2.37 on one pair and 3.28 on another.
    - Comparable within a report, not across videos of differing width — the same
      caveat compute_frame_quality carries for blur, for the same reason.
    """
    # nan, not 0.0: with no matches the displacement is unknown, and 0.0 would
    # read as "the camera held perfectly still", the opposite conclusion.
    if len(pts_a) == 0:
        return float("nan")

    # Median over per-match displacement, so a handful of bad matches cannot
    # drag the number the way a mean would.
    return float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))


def compute_parallax(pts_a: np.ndarray, pts_b: np.ndarray, *, ransac_thresh_px: float = 3.0) -> float:
    """
    One minus the homography/fundamental inlier ratio — how far the pair departs from a plane.

    - A homography explains rotation-only motion and planar scenes exactly, so a
      ratio near 1 (parallax near 0) means the pair carries no depth information.
      Read it alongside translation: a flat scene under real translation also
      reads 0.
    - ransac_thresh_px is a first-order lever, not a detail: measured over 99
      tutorial pairs, 1.0 against 3.0 moves parallax by 0.19 on average and
      reorders the pairs (Spearman 0.708), while 3.0 against 5.0 barely does
      (0.053, 0.945). Loosening it lets a homography explain more, so parallax
      falls monotonically. Same grid as the points, i.e. the analysis grid for
      the report.
    - nan below 8 correspondences — the linear 8-point algorithm's minimum.
      (MAGSAC's 7-point solver does return an F at exactly 7, and OpenCV raises
      below that, but 8 is the floor this reports against.)
    - nan when either fit raises: USAC asserts on configurations it cannot
      estimate rather than returning an empty model, and it does so at any size —
      measured on a real 720-correspondence pair whose matches were 97.5%
      zero-displacement. That is a fact about the pair, so it is a nan, not an
      exception that throws away every other frame in a multi-minute run.
    """
    if len(pts_a) < 8:
        return float("nan")

    # Fit both models to the same correspondences. H can only explain a plane or
    # a pure rotation; F can additionally explain translation through depth, so
    # the gap between their inlier counts IS the depth information in the pair.
    try:
        _, h_inliers = cv2.findHomography(pts_a, pts_b, cv2.USAC_MAGSAC, ransac_thresh_px)
        _, f_inliers = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, ransac_thresh_px)
    except cv2.error:
        return float("nan")
    n_h = int(h_inliers.sum()) if h_inliers is not None else 0
    n_f = int(f_inliers.sum()) if f_inliers is not None else 0

    # No F inliers means the pair is unexplained by any two-view geometry
    if n_f == 0:
        return float("nan")

    # min() guards the case where H outfits F on a degenerate pair, which would
    # otherwise push the complement negative.
    return float(1.0 - min(1.0, n_h / n_f))


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
    video_path, start, count, emit_from, stride, info = args

    # THE THREAD PIN IS LOAD-BEARING. cv2 and numpy each fan out over every core,
    # so the "serial" baseline is already parallel and naive process fan-out
    # oversubscribes: unpinned, this measured 0.67x — SLOWER than serial.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)

    frame_rows: list[dict] = []
    pair_rows: list[dict] = []

    # Hold only the ORB features still owed a partner: stride + 1 frames at a
    # time, so memory does not track range length.
    pending: dict[int, tuple] = {}

    for idx, bgr in iter_frames(video_path, start=start, count=count, info=info):
        # The lead-in frames belong to the previous range, which is already
        # measuring them. They are decoded here only to be somebody's partner,
        # so skip the photometry rather than compute a row and drop it.
        if idx >= emit_from:
            frame_rows.append({"frame_idx": idx, **compute_frame_quality(bgr)})

        # Motion against the frame one stride back, once one exists. The first
        # pair this can fire on is (emit_from - stride, emit_from), which is
        # exactly the boundary pair the lead-in exists to reach — so no pair
        # owned by the previous range is ever emitted twice.
        gray_small = analysis_gray(bgr)
        pending[idx] = detect_orb(gray_small)
        partner = idx - stride

        if partner in pending:
            pts_a, pts_b = match_descriptors(pending[partner], pending[idx])

            pair_rows.append(
                {
                    "frame_idx_a": partner,
                    "frame_idx_b": idx,
                    "n_matches": int(len(pts_a)),
                    "translation_px": compute_translation(pts_a, pts_b),
                    "parallax": compute_parallax(pts_a, pts_b),
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

    - video_path is decoded in full; every frame is measured. output_path=None
      returns the report without touching disk.
    - motion_stride: frames between the two members of each measured pair, >= 1.
      None means round(fps) — one second of video, the pair spacing a
      reconstruction sees under the shipping fps: 1.0 sampling rate.
    - workers: decode+measure this many contiguous frame ranges in parallel.
      1 = serial. Measured end to end on a 2388-frame 1080x1920 video (96-core
      host): 180.1 s at 1 against 93.9 s at 4, so 1.92x, with byte-identical
      reports. NOT auto-derived — see the design doc, section 5.2.
    - HYPOTHESIS, not a measurement: the shortfall against the standalone
      harness's 3.04x prediction is likely ffmpeg. Each worker pins cv2, OMP and
      OpenBLAS to one thread but not ffmpeg's own decode threads, so the serial
      baseline already fans out over cores and has less headroom to win back.
    """
    # `is not None`, not truthiness: 0 is an explicit value, and letting it fall
    # through to the fps default silently measures a stride of 30 instead. A
    # negative stride is worse than wrong — the partner index runs forward, so
    # no entry is ever retired from `pending` and it grows with the video.
    if motion_stride is not None and motion_stride < 1:
        raise ValueError(f"motion_stride must be >= 1, got {motion_stride}")

    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    video_path = Path(video_path)
    info = get_video_info(str(video_path))
    stride = int(motion_stride) if motion_stride is not None else max(1, round(info["fps"] or 1))

    # Announce the work before the first decode — a multi-minute silent run is
    # indistinguishable from a hung one. %s on the ints so a probe that came
    # back with None does not crash the log line itself.
    logger.info(
        "video quality: %s — %s frames @ %.2f fps, %sx%s, stride %d",
        video_path.name,
        info["total_frames"],
        info["fps"] or 0.0,
        info["width"],
        info["height"],
        stride,
    )
    # One range per worker. Ranges tile the video exactly once; each also decodes
    # a `stride`-frame lead-in from its predecessor so the boundary pairs have a
    # partner, but emits nothing from it.
    total = info["total_frames"]

    if workers == 1 or total <= stride * 2:
        ranges = [(str(video_path), 0, None, 0, stride, info)]
    else:
        per = total // workers
        ranges = []
        for k in range(workers):
            emit_from = k * per
            start = max(emit_from - stride, 0)
            end = total if k == workers - 1 else (k + 1) * per
            ranges.append((str(video_path), start, end - start, emit_from, stride, info))

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

    # Cheap integrity check on the seeks. Input seek (-ss before -i) lands on a
    # keyframe and counts forward; on a file with broken timestamps it can land
    # off by a few frames, and then every index in the report is quietly wrong.
    # Contiguity catches that as a gap, a repeat or a reordering. Compared as
    # ints, not as measured floats, so it cannot false-alarm on rounding.
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
                # nan -> null on the only two columns that can be non-finite.
                # json.dumps writes a bare NaN that no strict parser accepts, and
                # np.nan_to_num is not the fix: its 0.0 fill would read as "no
                # motion", the opposite of "this pair failed to match".
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

    Reuse is by existence, the same rule frames.zarr follows — so a re-run of a
    scene never re-measures, and nothing needs a staleness check.
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
