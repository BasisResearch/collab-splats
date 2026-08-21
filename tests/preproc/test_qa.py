import cv2
import numpy as np
import pytest

from collab_splats.preproc.qa import (
    _analysis_gray,
    check_frame_quality,
    compute_blur,
    compute_blur_score,
    compute_exposure,
    compute_frame_quality,
)

########################################################################
# Quality gate
########################################################################


def test_compute_blur_score_sharp_exceeds_blurred(noise_gray):
    sharp = noise_gray
    blurred = cv2.GaussianBlur(sharp, (25, 25), 0)
    assert compute_blur_score(sharp) > compute_blur_score(blurred) * 10


def test_check_frame_quality_accepts_sharp_frame(noise_gray):
    ok, metrics = check_frame_quality(noise_gray)
    assert ok is True
    assert metrics["reject_reason"] is None


def test_check_frame_quality_rejects_blurred_frame(noise_gray):
    sharp = noise_gray
    blurred = cv2.GaussianBlur(sharp, (25, 25), 0)
    # Threshold between the two measured scores makes the test threshold-robust
    threshold = (compute_blur_score(sharp) + compute_blur_score(blurred)) / 2
    ok, metrics = check_frame_quality(blurred, blur_threshold=threshold)
    assert ok is False and metrics["reject_reason"] == "blur"
    ok, _ = check_frame_quality(sharp, blur_threshold=threshold)
    assert ok is True


def test_check_frame_quality_rejects_bad_exposure():
    # Near-black and near-white frames fail regardless of sharpness
    dark = np.zeros((240, 320), dtype=np.uint8)
    bright = np.full((240, 320), 255, dtype=np.uint8)
    for gray in (dark, bright):
        ok, metrics = check_frame_quality(gray, blur_threshold=0.0)
        assert ok is False and metrics["reject_reason"] == "exposure"


def test_check_frame_quality_metrics_fields(noise_gray):
    _, metrics = check_frame_quality(noise_gray)
    assert set(metrics) == {"blur_score", "exposure_mean", "exposure_std", "reject_reason"}


def test_check_frame_quality_uses_precomputed_blur_score(noise_gray):
    # Passing blur_score short-circuits the Laplacian recompute
    ok, metrics = check_frame_quality(noise_gray, blur_threshold=100.0, blur_score=50.0)
    assert ok is False and metrics["blur_score"] == 50.0


########################################################################
# Per frame
########################################################################


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


def test_compute_blur_reuses_the_gate_metric(noise_gray):
    # One implementation of the Laplacian in the repo, not two
    assert compute_blur(noise_gray)["laplacian"] == compute_blur_score(noise_gray)


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


def test_compute_blur_rejects_colour_input(noise_gray):
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
    """640x480 mid-grey BGR with 300 scattered saturated pixels.

    Scattered, not a block: a saturated block survives downscaling because the
    interpolation window is entirely white, so it would not exercise the bug.
    """
    rng = np.random.default_rng(0)
    bgr = rng.integers(64, 192, (480, 640, 3)).astype(np.uint8)
    ys, xs = rng.integers(0, 480, 300), rng.integers(0, 640, 300)
    bgr[ys, xs] = 255
    return bgr


def test_compute_frame_quality_merges_both_measurements(clipped_bgr):
    blank = np.zeros((8, 8), np.uint8)
    assert set(compute_frame_quality(clipped_bgr)) == set(compute_blur(blank)) | set(compute_exposure(blank))


def test_compute_frame_quality_reads_exposure_at_native_resolution(clipped_bgr):
    # The contract that keeps clipping measurable: exposure must NOT go through
    # _analysis_gray, which erases scattered saturated pixels completely.
    native = compute_frame_quality(clipped_bgr)
    downscaled = compute_exposure(_analysis_gray(clipped_bgr))
    assert native["clipped_high_frac"] == pytest.approx(300 / (480 * 640), rel=0.05)
    assert downscaled["clipped_high_frac"] == 0.0


def test_compute_frame_quality_reads_blur_at_analysis_resolution(clipped_bgr):
    # Blur must go through _analysis_gray; assert by equality with the explicit path
    expected = compute_blur(_analysis_gray(clipped_bgr))["blur"]
    assert compute_frame_quality(clipped_bgr)["blur"] == pytest.approx(expected)
