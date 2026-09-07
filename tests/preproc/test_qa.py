import json
import logging

import cv2
import numpy as np
import pytest

from collab_splats.preproc import qa
from collab_splats.preproc.qa import (
    _measure_photometry_and_motion,
    analysis_gray,
    compute_blur,
    compute_exposure,
    compute_frame_quality,
    compute_pair_motion,
    compute_video_quality,
    detect_orb,
    load_video_quality,
)


def _raise_usac_error(*args, **kwargs):
    """
    Stand in for the USAC assert OpenCV throws on a pair it cannot estimate.
    """
    raise cv2.error("USAC: bad model")


def _descriptors(n, seed=7):
    """
    n ORB-shaped 32-byte descriptors, random so mutual-best matching is the identity.
    """
    return np.random.default_rng(seed).integers(0, 256, (n, 32), dtype=np.uint8)


def _pair_motion(pts_a, pts_b, **kwargs):
    """
    compute_pair_motion over two synthetic point sets that correspond by index.

    Both sides share one descriptor table, so crossCheck pairs row i with row i and
    the correspondences the function sees are exactly the inputs, in order. That is
    what lets the geometry assertions below drive projected points, which no real
    image pair produces on demand.
    """
    desc = _descriptors(len(pts_a))
    feat_a = (tuple(cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts_a), desc)
    feat_b = (tuple(cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts_b), desc)

    return compute_pair_motion(feat_a, feat_b, **kwargs)


########################################################################
# Per frame
########################################################################


def test_analysis_gray_downscales_to_width_and_never_upscales(clipped_bgr):
    # 640-wide source: 480 downscales, and a width above the source is a no-op
    # rather than an upscale, so a narrow video is measured natively.
    assert analysis_gray(clipped_bgr).shape == (360, 480)
    assert analysis_gray(clipped_bgr, width=320).shape == (240, 320)
    assert analysis_gray(clipped_bgr, width=1024).shape == clipped_bgr.shape[:2]


def test_compute_blur_keys(noise_gray):
    assert set(compute_blur(noise_gray)) == {"blur", "laplacian"}


def test_compute_blur_moves_in_opposite_directions(noise_gray):
    # blur is Crete-Roffet (high = blurrier); laplacian is variance (high = sharper).
    # Progressive Gaussian blur must raise one and lower the other, monotonically.
    ladder = [compute_blur(cv2.GaussianBlur(noise_gray, (0, 0), s) if s else noise_gray) for s in (0, 1, 3, 6)]
    blur = [r["blur"] for r in ladder]
    laplacian = [r["laplacian"] for r in ladder]
    assert blur == sorted(blur), blur
    assert laplacian == sorted(laplacian, reverse=True), laplacian


def test_compute_blur_measured_values(noise_gray):
    # Pinned to measured values so a library swap that silently rescales either
    # metric fails loudly rather than shifting every report already on disk.
    sharp = compute_blur(noise_gray)
    blurred = compute_blur(cv2.GaussianBlur(noise_gray, (0, 0), 3))
    assert sharp["blur"] == pytest.approx(0.1202, abs=0.01)
    assert sharp["laplacian"] == pytest.approx(108108.3, rel=0.05)
    assert blurred["blur"] == pytest.approx(0.4798, abs=0.01)
    assert blurred["laplacian"] == pytest.approx(3.6, rel=0.2)


def test_compute_blur_laplacian_is_the_variance_of_the_laplacian(noise_gray):
    # The laplacian column is exactly cv2's Laplacian variance, nothing rescaled
    assert compute_blur(noise_gray)["laplacian"] == float(cv2.Laplacian(noise_gray, cv2.CV_64F).var())


def test_compute_blur_saturates_on_sparse_detail(noise_gray):
    # blur cannot fail a bounds check — |M1 - M2| / M1 with 0 <= M2 <= M1 is
    # bounded by construction. What is worth pinning is the top of that range:
    # blur hits exactly 1.0 whenever there is little high-frequency content to
    # destroy, and a single perfectly sharp edge qualifies. Only laplacian
    # separates that from a genuinely soft frame.
    flat = np.full((240, 320), 128, np.uint8)
    edge = np.zeros((240, 320), np.uint8)
    edge[:, 160:] = 255
    assert compute_blur(flat) == {"blur": 1.0, "laplacian": 0.0}
    assert compute_blur(edge)["blur"] == 1.0
    assert compute_blur(edge)["laplacian"] > 100.0


def test_compute_blur_rejects_color_input(noise_gray):
    # skimage returns nan on a 3-channel array while cv2.Laplacian returns a
    # plausible number, so the row would be half-valid and read as a real
    # failed measurement rather than a bad call. Refuse instead.
    with pytest.raises(ValueError, match="2-D single-channel"):
        compute_blur(cv2.cvtColor(noise_gray, cv2.COLOR_GRAY2BGR))


def test_compute_blur_h_size_is_tunable(noise_gray):
    # h_size is the metric's only tuning value; the default matches skimage's.
    assert compute_blur(noise_gray, h_size=11) == compute_blur(noise_gray)
    wide = compute_blur(noise_gray, h_size=21)["blur"]
    narrow = compute_blur(noise_gray, h_size=3)["blur"]
    assert narrow > wide
    # laplacian does not depend on h_size at all
    assert compute_blur(noise_gray, h_size=3)["laplacian"] == compute_blur(noise_gray)["laplacian"]


def test_compute_exposure_keys():
    keys = set(compute_exposure(np.full((10, 10), 128, np.uint8)))
    assert keys == {
        "exposure_mean",
        "exposure_median",
        "exposure_std",
        "clipped_low_frac",
        "clipped_high_frac",
    }


def test_compute_exposure_flat_image():
    result = compute_exposure(np.full((10, 10), 128, np.uint8))
    assert result["exposure_mean"] == pytest.approx(128.0)
    assert result["exposure_median"] == pytest.approx(128.0)
    assert result["exposure_std"] == pytest.approx(0.0)
    assert result["clipped_low_frac"] == 0.0
    assert result["clipped_high_frac"] == 0.0


def test_compute_exposure_counts_clipping_at_both_ends():
    # 5 of 100 pixels crushed to black, 5 of 100 blown to white
    gray = np.full((10, 10), 128, np.uint8)
    gray[0, :5] = 0
    gray[1, :5] = 255
    result = compute_exposure(gray)
    assert result["clipped_low_frac"] == pytest.approx(0.05)
    assert result["clipped_high_frac"] == pytest.approx(0.05)


def test_compute_exposure_clipping_rises_only_at_saturation():
    # Scale a uniform mid-bright frame up and down: the mean tracks the scale,
    # but the clipping fractions stay 0 until pixels actually reach 255 or 0.
    base = np.full((10, 10), 200, np.uint8)
    brighter = [compute_exposure(np.clip(base * f, 0, 255).astype(np.uint8)) for f in (1.0, 1.2, 1.3)]
    assert [r["exposure_mean"] for r in brighter] == pytest.approx([200.0, 240.0, 255.0])
    assert [r["clipped_high_frac"] for r in brighter] == [0.0, 0.0, 1.0]
    darker = [compute_exposure((base * f).astype(np.uint8)) for f in (0.1, 0.0)]
    assert [r["exposure_mean"] for r in darker] == pytest.approx([20.0, 0.0])
    assert [r["clipped_low_frac"] for r in darker] == [0.0, 1.0]


def test_compute_exposure_median_separates_from_mean():
    # A dark scene with a bright window: the mean is dragged up, the median is not
    gray = np.full((100, 100), 30, np.uint8)
    gray[:10, :] = 250
    result = compute_exposure(gray)
    assert result["exposure_median"] == pytest.approx(30.0)
    assert result["exposure_mean"] > 50.0


@pytest.fixture(scope="module")
def clipped_bgr():
    """640x480 mid-gray BGR with 300 scattered saturated pixels.

    Scattered, not a block: a saturated block survives downscaling because the
    interpolation window is entirely white, so it would not exercise the bug.
    Shared across the module, so treat it as read-only.
    """
    rng = np.random.default_rng(0)
    bgr = rng.integers(64, 192, (480, 640, 3)).astype(np.uint8)
    ys, xs = rng.integers(0, 480, 300), rng.integers(0, 640, 300)
    bgr[ys, xs] = 255
    return bgr


def test_compute_frame_quality_merges_both_measurements(clipped_bgr):
    blank = np.zeros((8, 8), np.uint8)
    merged = compute_frame_quality(clipped_bgr)
    assert set(merged) == set(compute_blur(blank)) | set(compute_exposure(blank))
    # Count as well as membership: a set union is identical under a key
    # collision, so without this a renamed key silently overwritten by the
    # {**blur, **exposure} merge would still pass.
    assert len(merged) == 7


def test_compute_frame_quality_reads_exposure_at_native_resolution(clipped_bgr):
    # The contract that keeps clipping measurable: exposure must NOT go through
    # analysis_gray, which erases scattered saturated pixels completely.
    native = compute_frame_quality(clipped_bgr)
    downscaled = compute_exposure(analysis_gray(clipped_bgr))
    assert native["clipped_high_frac"] == pytest.approx(300 / (480 * 640), rel=0.05)
    assert downscaled["clipped_high_frac"] == 0.0


def test_compute_frame_quality_reads_blur_at_analysis_resolution(clipped_bgr):
    # Blur must go through analysis_gray; assert by equality with the explicit path
    expected = compute_blur(analysis_gray(clipped_bgr))["blur"]
    assert compute_frame_quality(clipped_bgr)["blur"] == pytest.approx(expected)


def test_compute_frame_quality_forwards_its_tuning(clipped_bgr):
    # analysis_width and blur_h_size are pass-throughs, and analysis_width is the
    # cross-video comparability lever — it must actually reach analysis_gray.
    tuned = compute_frame_quality(clipped_bgr, analysis_width=240, blur_h_size=3)
    expected = compute_blur(analysis_gray(clipped_bgr, width=240), h_size=3)
    assert tuned["blur"] == pytest.approx(expected["blur"])
    assert tuned["laplacian"] == pytest.approx(expected["laplacian"])
    # Exposure is native-resolution, so no tuning can move it
    assert tuned["clipped_high_frac"] == compute_frame_quality(clipped_bgr)["clipped_high_frac"]


########################################################################
# Per pair
########################################################################


def test_compute_pair_motion_returns_all_three_measures():
    """
    One call per pair, replacing match + translation + parallax.
    """
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (300, 400), dtype=np.uint8)
    a, b = canvas[:240, :320], canvas[10:250, 8:328]

    row = compute_pair_motion(detect_orb(a), detect_orb(b))

    assert set(row) == {"n_matches", "translation_px", "parallax"}
    assert row["n_matches"] > 20
    assert 8.0 < row["translation_px"] < 20.0
    # int, not np.int64: the report is serialized to JSON straight from this dict
    assert isinstance(row["n_matches"], int)


def test_compute_pair_motion_respects_n_features(noise_gray):
    shifted = np.roll(noise_gray, 5, axis=1)
    few = compute_pair_motion(detect_orb(noise_gray, n_features=50), detect_orb(shifted, n_features=50))
    many = compute_pair_motion(detect_orb(noise_gray, n_features=1000), detect_orb(shifted, n_features=1000))
    assert few["n_matches"] < many["n_matches"]


def test_compute_pair_motion_recovers_known_shift(noise_gray):
    # Roll the image 17 px right; the median match displacement must be 17 px
    row = compute_pair_motion(detect_orb(noise_gray), detect_orb(np.roll(noise_gray, 17, axis=1)))
    assert row["translation_px"] == pytest.approx(17.0, abs=1.0)


def test_compute_pair_motion_on_an_unmatchable_pair():
    """
    Two unrelated frames give zero matches and nan measures, not an exception.
    """
    # A flat frame has no corners, so ORB returns no descriptors at all. nan, not
    # 0.0: with no matches the displacement is unknown, and 0.0 would read as "the
    # camera held perfectly still", the opposite conclusion. compute_video_quality
    # serializes these two columns as JSON null, which 0.0 would silently replace.
    a = np.zeros((240, 320), np.uint8)
    b = np.full((240, 320), 255, np.uint8)

    row = compute_pair_motion(detect_orb(a), detect_orb(b))

    assert row["n_matches"] == 0
    assert np.isnan(row["translation_px"])
    assert np.isnan(row["parallax"])


def _project(points_3d):
    """Pinhole-project Nx3 world points with fx=fy=500, cx=320, cy=240."""
    x = 500.0 * points_3d[:, 0] / points_3d[:, 2] + 320.0
    y = 500.0 * points_3d[:, 1] / points_3d[:, 2] + 240.0
    return np.stack([x, y], axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_scenes():
    """A depth-varying point cloud and a planar one, both 300 points."""
    rng = np.random.default_rng(3)
    volume = np.stack([rng.uniform(-3, 3, 300), rng.uniform(-3, 3, 300), rng.uniform(4, 12, 300)], axis=1)
    plane = np.stack([rng.uniform(-3, 3, 300), rng.uniform(-3, 3, 300), np.full(300, 8.0)], axis=1)
    return volume, plane


def test_parallax_high_when_depth_varies(synthetic_scenes):
    # Translation across a scene with real depth spread: a homography cannot
    # explain the pair, so most H inliers are lost relative to F.
    volume, _ = synthetic_scenes
    row = _pair_motion(_project(volume), _project(volume - np.array([0.8, 0.0, 0.0])))
    assert row["parallax"] > 0.5


def test_parallax_zero_for_rotation_only(synthetic_scenes):
    # A pure rotation is exactly a homography no matter how much depth exists
    volume, _ = synthetic_scenes
    theta = np.deg2rad(5.0)
    rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    row = _pair_motion(_project(volume), _project(volume @ rot.T))
    assert row["parallax"] < 0.1
    # ...and the image content really did move, so translation alone cannot tell
    # this case apart from the planar one below.
    assert row["translation_px"] > 10.0


def test_parallax_zero_for_translating_over_a_plane(synthetic_scenes):
    # THE TRAP: a flat scene reads parallax 0.0 even under real translation,
    # because a plane is also exactly a homography. parallax alone cannot
    # distinguish "camera did not move" from "scene has no depth".
    _, plane = synthetic_scenes
    row = _pair_motion(_project(plane), _project(plane - np.array([0.8, 0.0, 0.0])))
    assert row["parallax"] < 0.1
    assert row["translation_px"] > 10.0


def test_parallax_is_nan_when_opencv_cannot_fit(monkeypatch):
    # USAC asserts instead of returning an empty model on configurations it cannot
    # estimate, and it does so at any size — not only below the 8-point floor.
    # Measured on frames 40/41 of tiny_video: 720 matches, 97.5% of them
    # zero-displacement, findHomography fine, findFundamentalMat raising at
    # estimator.cpp:353. Which pairs trip it moves with the OpenCV build (4.10.0
    # fits that same pair), so the raise is injected rather than provoked.
    # Unhandled, one such pair discards every other measurement in a long run.
    pts = (np.random.default_rng(0).random((50, 2)) * 100).astype(np.float32)
    monkeypatch.setattr(cv2, "findFundamentalMat", _raise_usac_error)

    row = _pair_motion(pts, pts + 1.0)

    # The pair still reports its matches and its translation — only the fit is lost
    assert np.isnan(row["parallax"])
    assert row["n_matches"] == 50
    assert row["translation_px"] == pytest.approx(np.sqrt(2.0), abs=1e-4)


def test_parallax_is_nan_below_eight_matches():
    # Eight is the fundamental matrix minimum; fewer is not a small sample, it is undefined
    pts = (np.random.default_rng(0).random((7, 2)) * 100).astype(np.float32)
    row = _pair_motion(pts, pts + 1.0)
    assert np.isnan(row["parallax"])
    assert row["n_matches"] == 7


def test_parallax_falls_as_the_ransac_threshold_loosens(synthetic_scenes):
    # Bounds are not worth asserting — 1 - min(1, n_h/n_f) is in [0, 1] by
    # construction, and this fixture's >0.5 case already pins it harder. What is
    # worth pinning is that ransac_thresh_px reaches both fits and moves the
    # number the way the docstring says: a looser inlier test lets a homography
    # explain more of the pair, so parallax falls. Measured 0.93 / 0.8067 /
    # 0.7033 / 0.0333 at 1 / 3 / 5 / 50 px, with zero spread across MAGSAC draws.
    volume, _ = synthetic_scenes
    pts_a, pts_b = _project(volume), _project(volume - np.array([0.8, 0.0, 0.0]))
    ladder = [_pair_motion(pts_a, pts_b, ransac_thresh_px=t)["parallax"] for t in (1.0, 3.0, 5.0, 50.0)]
    assert ladder == sorted(ladder, reverse=True), ladder
    assert ladder[0] > 0.9 and ladder[-1] < 0.1
    # The default is 3.0, so the keyword-free call sits on the second rung
    assert _pair_motion(pts_a, pts_b)["parallax"] == pytest.approx(ladder[1])


def test_the_three_collapsed_helpers_are_gone():
    """
    match_descriptors/compute_translation/compute_parallax had one caller between them.
    """
    for dead in ("match_descriptors", "compute_translation", "compute_parallax"):
        assert not hasattr(qa, dead), dead


########################################################################
# Whole video
########################################################################


def test_compute_video_quality_top_level_keys(tiny_video):
    report = compute_video_quality(tiny_video)
    assert set(report) == {"available", "video", "params", "frames", "pairs"}
    assert report["available"] is True


def test_compute_video_quality_video_block(tiny_video):
    video = compute_video_quality(tiny_video)["video"]
    assert set(video) == {"path", "mtime", "total_frames", "fps", "duration_s", "width", "height"}
    assert video["total_frames"] == 60
    assert (video["width"], video["height"]) == (320, 240)


def test_compute_video_quality_frame_columns_are_equal_length(tiny_video):
    frames = compute_video_quality(tiny_video)["frames"]
    assert set(frames) == {
        "frame_idx",
        "blur",
        "laplacian",
        "exposure_mean",
        "exposure_median",
        "exposure_std",
        "clipped_low_frac",
        "clipped_high_frac",
    }
    assert {len(v) for v in frames.values()} == {60}
    # frame_idx is the source video index, not a row position
    assert frames["frame_idx"] == list(range(60))


def test_compute_video_quality_pairs_use_the_default_stride(tiny_video):
    report = compute_video_quality(tiny_video)
    pairs = report["pairs"]
    assert set(pairs) == {"frame_idx_a", "frame_idx_b", "translation_px", "parallax", "n_matches"}
    # 30 fps rounds to a stride of 30, leaving 60 - 30 = 30 pairs
    assert report["params"] == {"motion_stride": 30}
    assert {len(v) for v in pairs.values()} == {30}
    assert pairs["frame_idx_a"][:3] == [0, 1, 2]
    assert pairs["frame_idx_b"][:3] == [30, 31, 32]


def test_compute_video_quality_honours_motion_stride(tiny_video):
    report = compute_video_quality(tiny_video, motion_stride=5)
    assert report["params"]["motion_stride"] == 5
    assert len(report["pairs"]["frame_idx_a"]) == 55
    assert report["pairs"]["frame_idx_b"][0] - report["pairs"]["frame_idx_a"][0] == 5


def test_compute_video_quality_keeps_n_matches_integral(tiny_video):
    n_matches = compute_video_quality(tiny_video, motion_stride=5)["pairs"]["n_matches"]
    assert all(isinstance(v, int) for v in n_matches)
    assert min(n_matches) > 0


def test_compute_video_quality_writes_json(tiny_video, tmp_path):
    out = tmp_path / "nested" / "video_quality_report.json"
    report = compute_video_quality(tiny_video, motion_stride=5, output_path=out)
    assert json.loads(out.read_text()) == report


def test_compute_video_quality_serializes_unmatched_pairs_as_null(tmp_path):
    # A featureless video is the case that produces nan: ORB finds no corners,
    # so every pair has 0 matches and nan translation/parallax. nan is not valid
    # JSON, and it is also the interesting measurement — it must survive as null.
    path = tmp_path / "flat.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for _ in range(20):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.release()

    out = tmp_path / "flat.json"
    report = compute_video_quality(path, motion_stride=5, output_path=out)
    assert report["pairs"]["n_matches"] == [0] * 15
    assert report["pairs"]["translation_px"] == [None] * 15
    assert report["pairs"]["parallax"] == [None] * 15
    # A frame of pure black is fully clipped low
    assert report["frames"]["clipped_low_frac"][0] == 1.0
    assert "NaN" not in out.read_text()
    assert json.loads(out.read_text()) == report


def test_compute_video_quality_reports_unavailable_for_an_undecodable_file(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"")
    report = compute_video_quality(broken)
    assert report["available"] is False
    assert "broken.mp4" in report["reason"]


def test_compute_video_quality_logs_before_and_after_the_decode(tiny_video, caplog):
    # A silent multi-minute run is indistinguishable from a hung one, so the
    # announce-then-summarize pair is a contract, not a nicety.
    with caplog.at_level(logging.INFO, logger="collab_splats.preproc.qa"):
        compute_video_quality(tiny_video, motion_stride=5)
    messages = [r.getMessage() for r in caplog.records]
    assert any("60 frames @" in m for m in messages), "no line logged before the decode"
    assert any("frames/s" in m for m in messages), "no elapsed/throughput line logged after"


def test_compute_video_quality_survives_a_pair_opencv_cannot_fit(tiny_video, monkeypatch):
    # The end-to-end half of the same failure: before the guard, an unfittable pair
    # raised cv2.error and lost all 60 frames of photometry along with the other 57
    # pairs. The first two pairs are made unfittable here; tiny_video met two of them
    # on its own until OpenCV 4.10.0 started fitting that configuration.
    real_find_f = cv2.findFundamentalMat
    seen = []

    def fail_first_two_pairs(*args, **kwargs):
        seen.append(1)
        if len(seen) <= 2:
            _raise_usac_error()
        return real_find_f(*args, **kwargs)

    monkeypatch.setattr(cv2, "findFundamentalMat", fail_first_two_pairs)

    report = compute_video_quality(tiny_video, motion_stride=1)
    assert len(report["frames"]["frame_idx"]) == 60
    parallax = report["pairs"]["parallax"]
    assert len(parallax) == 59
    # The unfittable pairs survive as null, and everything else is still measured.
    # Degenerate pairs null out on their own too (5 of them under 4.10.0), so the
    # count is a floor, not an equality.
    assert parallax[0] is None and parallax[1] is None
    assert sum(v is not None for v in parallax) >= 50


def test_compute_video_quality_rejects_a_stride_below_one(tiny_video):
    # 0 is an explicit value, not "unset": under a truthiness check it silently
    # became round(fps). A negative stride is worse — the partner index runs
    # forward, so nothing is ever retired from the pending dict and it grows
    # with the video instead of staying at stride + 1 frames.
    for bad in (0, -1, -5):
        with pytest.raises(ValueError, match="must be >= 1"):
            compute_video_quality(tiny_video, motion_stride=bad)


def test_compute_video_quality_names_a_missing_file_as_missing(tmp_path):
    report = compute_video_quality(tmp_path / "nope.mp4")
    assert report["available"] is False
    assert "file does not exist" in report["reason"]


@pytest.mark.parametrize("kind", ["uniform", "constant", "bimodal", "narrow"])
def test_compute_exposure_matches_the_numpy_path(kind):
    """
    Histogram exposure must agree with the numpy definitions it replaced.
    """
    rng = np.random.default_rng(0)
    shapes = [(2, 2), (3, 5), (64, 64), (17, 31)]

    for shape in shapes:
        if kind == "uniform":
            gray = rng.integers(0, 256, shape, dtype=np.uint8)
        elif kind == "constant":
            gray = np.full(shape, 137, np.uint8)
        elif kind == "bimodal":
            gray = rng.integers(0, 2, shape, dtype=np.uint8) * 255
        else:
            gray = rng.integers(100, 140, shape, dtype=np.uint8)

        out = compute_exposure(gray)

        # Exact: mean, median and both clipping fractions. The float64 cast on the
        # histogram and the two-order-statistic median are what make them exact —
        # see the design doc, section 5.1.
        assert out["exposure_mean"] == float(gray.mean())
        assert out["exposure_median"] == float(np.median(gray))
        assert out["clipped_low_frac"] == float((gray == 0).mean())
        assert out["clipped_high_frac"] == float((gray == 255).mean())

        # std differs only in summation order; measured max deviation 4.3e-14
        assert out["exposure_std"] == pytest.approx(float(gray.std()), abs=1e-9)


def test_detect_orb_is_deterministic_so_the_cached_pair_loop_is_safe(noise_gray):
    """
    The pair loop detects each frame once and reuses it as the next pair's
    partner, which only reproduces a detect-per-pair run if detection repeats.
    """
    other = np.roll(noise_gray, 5, axis=1)

    once = compute_pair_motion(detect_orb(noise_gray), detect_orb(other))
    twice = compute_pair_motion(detect_orb(noise_gray), detect_orb(other))

    assert once["n_matches"] == twice["n_matches"]
    assert once["translation_px"] == twice["translation_px"]


def test_detect_orb_on_a_featureless_frame_returns_no_descriptors():
    kp, desc = detect_orb(np.zeros((64, 64), np.uint8))

    assert desc is None or len(kp) == 0


########################################################################
# Range parallelism and the cached report
########################################################################


def test_video_quality_workers_produce_an_identical_report(tiny_video):
    """
    Range-parallel measurement must not change a single number.
    """
    serial = compute_video_quality(tiny_video, motion_stride=2)
    parallel = compute_video_quality(tiny_video, motion_stride=2, workers=3)

    assert parallel["frames"] == serial["frames"]
    assert parallel["pairs"] == serial["pairs"]


def test_video_quality_rejects_a_bad_worker_count(tiny_video):
    with pytest.raises(ValueError, match="workers"):
        compute_video_quality(tiny_video, workers=0)


def test_measure_photometry_and_motion_emits_only_the_frames_it_owns(tiny_video):
    """
    A range's lead-in frames are decoded to be somebody's partner, not measured.
    """
    # A range owning frames 20-59, decoding from 18 so the pair straddling the
    # boundary has a partner: start=18, count=42, emit_from=20, stride=2.
    frames, pairs = _measure_photometry_and_motion((tiny_video, 18, 42, 20, 2))

    # The two lead-in frames produce no photometry row
    assert [row["frame_idx"] for row in frames] == list(range(20, 60))

    # But they do produce the boundary pair, which no other range can emit
    assert (pairs[0]["frame_idx_a"], pairs[0]["frame_idx_b"]) == (18, 20)
    assert [row["frame_idx_b"] for row in pairs] == list(range(20, 60))


def test_video_quality_refuses_non_contiguous_ranges(tiny_video, monkeypatch):
    """
    Ranges tile the video exactly once, so a gap means a seek landed wrong.
    """

    class InlinePool:
        # Runs map in-process, so the fake below never has to pickle
        def __init__(self, n_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def map(self, fn, items):
            return [fn(item) for item in items]

    # One row per range instead of the whole range: indices 0 and 30, a gap
    def one_row_per_range(args):
        emit_from = args[3]
        return [{"frame_idx": emit_from, "blur": 1.0}], []

    monkeypatch.setattr(qa, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(qa, "_measure_photometry_and_motion", one_row_per_range)

    with pytest.raises(ValueError, match="not contiguous"):
        compute_video_quality(tiny_video, motion_stride=2, workers=2)


def test_load_video_quality_writes_then_reuses(tiny_video, tmp_path):
    report_path = tmp_path / "video_quality_report.json"

    first = load_video_quality(tiny_video, report_path, motion_stride=2)

    assert report_path.exists() and first["available"]

    # Second call must read the file, not re-measure it
    stamp = report_path.stat().st_mtime_ns
    second = load_video_quality(tiny_video, report_path, motion_stride=2)

    assert report_path.stat().st_mtime_ns == stamp
    assert second["frames"] == first["frames"]
