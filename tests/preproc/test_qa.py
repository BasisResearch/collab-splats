import cv2
import numpy as np

from collab_splats.preproc.qa import check_frame_quality, compute_blur_score

########################################################################
# Quality gate
########################################################################


def _sharp_gray():
    """High-frequency noise — very high Laplacian variance."""
    rng = np.random.default_rng(1)
    return (rng.random((240, 320)) * 255).astype(np.uint8)


def test_compute_blur_score_sharp_exceeds_blurred():
    sharp = _sharp_gray()
    blurred = cv2.GaussianBlur(sharp, (25, 25), 0)
    assert compute_blur_score(sharp) > compute_blur_score(blurred) * 10


def test_check_frame_quality_accepts_sharp_frame():
    ok, metrics = check_frame_quality(_sharp_gray())
    assert ok is True
    assert metrics["reject_reason"] is None


def test_check_frame_quality_rejects_blurred_frame():
    sharp = _sharp_gray()
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


def test_check_frame_quality_metrics_fields():
    _, metrics = check_frame_quality(_sharp_gray())
    assert set(metrics) == {"blur_score", "exposure_mean", "exposure_std", "reject_reason"}


def test_check_frame_quality_uses_precomputed_blur_score():
    # Passing blur_score short-circuits the Laplacian recompute
    ok, metrics = check_frame_quality(_sharp_gray(), blur_threshold=100.0, blur_score=50.0)
    assert ok is False and metrics["blur_score"] == 50.0
