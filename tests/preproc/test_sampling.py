import logging

import cv2
import numpy as np
import pytest

from collab_splats.preproc.sampling import _fps_targets, _uniform_targets
from collab_splats.preproc.video import get_video_info


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """Synthesize a 60-frame 320x240 mp4: static noise texture + moving square.

    Noise gives LK flow corners to track; the moving square creates motion.
    """
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.default_rng(0)
    noise = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    for i in range(60):
        frame = noise.copy()
        x = 10 + i * 4
        cv2.rectangle(frame, (x, 60), (x + 60, 140), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return str(path)


########################################################################
# Quality gate
########################################################################

from collab_splats.preproc.sampling import check_frame_quality, compute_blur_score


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


########################################################################
# Selector
########################################################################

from collab_splats.preproc.sampling import OpticalFlowFrameSelector, _combine_scores


def test_selector_first_frame_scores_one():
    selector = OpticalFlowFrameSelector()
    score, components = selector.score_frame(_sharp_gray())
    assert score == 1.0
    assert components == {"disparity": 0.0, "rotation": 0.0, "histogram_similarity": 1.0}


def test_selector_scores_are_normalized():
    selector = OpticalFlowFrameSelector()
    rng = np.random.default_rng(2)
    for _ in range(5):
        gray = (rng.random((240, 320)) * 255).astype(np.uint8)
        score, _ = selector.score_frame(gray)
        assert 0.0 <= score <= 1.0


def test_selector_identical_frame_scores_low():
    selector = OpticalFlowFrameSelector()
    gray = _sharp_gray()
    selector.score_frame(gray)  # seeds keyframe
    score, components = selector.score_frame(gray)
    # No motion, near-identical histogram → low combined score
    assert score < 0.3
    assert components["disparity"] < 1.0


def test_selector_has_no_stats_attr():
    from collab_splats.preproc.sampling import OpticalFlowFrameSelector

    sel = OpticalFlowFrameSelector(min_disparity=50.0)
    assert not hasattr(sel, "stats")


def test_combine_scores_monotonic_in_disparity():
    lo = _combine_scores(10.0, 0.5, min_disparity=50.0)
    hi = _combine_scores(60.0, 0.5, min_disparity=50.0)
    assert hi > lo
    assert 0.0 <= lo <= hi <= 1.0


########################################################################
# Samplers
########################################################################

from collab_splats.preproc.sampling import sample_frames, score_frames
from collab_splats.preproc.video import _iter_frames


@pytest.fixture(scope="module")
def blur_pattern_video(tmp_path_factory):
    """40-frame video where every even frame is heavily blurred.

    Lets tests verify sharpest-in-window picks odd (sharp) frames.
    """
    path = tmp_path_factory.mktemp("vid") / "blurry.mp4"
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.default_rng(3)
    noise = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    for i in range(40):
        frame = noise.copy()
        cv2.rectangle(frame, (10 + i * 4, 60), (70 + i * 4, 140), (0, 255, 0), -1)
        if i % 2 == 0:
            frame = cv2.GaussianBlur(frame, (31, 31), 0)
        writer.write(frame)
    writer.release()
    return str(path)


def test_sample_frames_unknown_method_raises(tiny_video):
    with pytest.raises(ValueError, match="Unknown method"):
        sample_frames(tiny_video, method="dso")


def test_sample_frames_params_are_keyword_only(tiny_video):
    with pytest.raises(TypeError):
        sample_frames(tiny_video, "uniform")  # positional method must fail


def test_fps_returns_frames_and_records(tiny_video):
    frames, records = sample_frames(tiny_video, method="fps", fps=10.0)
    # 60 frames @30fps sampled at 10fps → stride 3 → 20
    assert len(frames) == len(records) == 20
    assert frames[0].shape == (240, 320, 3)
    assert set(records[0]) == {"frame_idx", "blur_score"}


def test_fps_respects_max_frames(tiny_video):
    frames, records = sample_frames(tiny_video, method="fps", fps=10.0, max_frames=5)
    assert len(frames) == len(records) == 5


def test_uniform_derives_window_from_max_frames(tiny_video):
    # 60 frames / max_frames 6 → interval 10 → 6 windows
    frames, _ = sample_frames(tiny_video, method="uniform", max_frames=6)
    assert len(frames) == 6


def test_uniform_derives_count_from_max_frames_exact(tiny_video):
    # 60-frame video, request 6 → exactly 6 evenly-spaced frames
    frames, records = sample_frames(tiny_video, method="uniform", max_frames=6)
    assert len(frames) == len(records) == 6
    assert set(records[0]) == {"frame_idx", "blur_score"}
    # indices are non-decreasing source-video indices within range
    idxs = [r["frame_idx"] for r in records]
    assert idxs == sorted(idxs) and idxs[0] >= 0 and idxs[-1] < 60


def test_uniform_best_effort_keeps_count_when_gate_rejects_all(tiny_video):
    # Impossible blur threshold → every frame fails the gate, but best-effort
    # keeps one per position, so the count still hits the target.
    frames, _ = sample_frames(tiny_video, method="uniform", max_frames=8, blur_threshold=1e12)
    assert len(frames) == 8


def test_uniform_selects_in_one_pass_not_whole_video(tiny_video, monkeypatch):
    # Single ffmpeg select pass over a bounded index set — NOT a whole-video
    # Python decode (_iter_frames), and not one subprocess per frame.
    import collab_splats.preproc.sampling as s

    monkeypatch.setattr(s, "_iter_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("full decode")))
    passes = {"n": 0, "wanted": 0}
    real = s._iter_selected_frames

    def counting(video_path, indices, w, h):
        passes["n"] += 1
        passes["wanted"] = len(indices)
        return real(video_path, indices, w, h)

    monkeypatch.setattr(s, "_iter_selected_frames", counting)
    frames, _ = sample_frames(tiny_video, method="uniform", max_frames=4)
    assert len(frames) == 4
    assert passes["n"] == 1  # one ffmpeg pass, not one per frame
    assert passes["wanted"] < 60  # decodes a subset, not the whole 60-frame video


def test_fps_frames_are_rgb(tiny_video):
    frames, records = sample_frames(tiny_video, method="fps", fps=10.0)
    bgr = list(_iter_frames(tiny_video))
    # RGB return means channel order is reversed vs the BGR decode
    np.testing.assert_array_equal(frames[0], bgr[records[0]["frame_idx"]][:, :, ::-1])


def test_optical_flow_first_frame_selected(tiny_video):
    frames, records = sample_frames(tiny_video, method="optical_flow", blur_threshold=0.0)
    assert len(frames) >= 1
    assert records[0]["frame_idx"] == 0


def test_optical_flow_records_have_source_indices(tiny_video):
    _, records = sample_frames(tiny_video, method="optical_flow", blur_threshold=0.0)
    idxs = [r["frame_idx"] for r in records]
    # Source video indices: strictly increasing, within range — NOT list positions
    assert idxs == sorted(idxs) and idxs[-1] < 60
    assert set(records[0]) == {
        "frame_idx",
        "blur_score",
        "score",
        "selected",
        "disparity",
        "rotation",
        "histogram_similarity",
    }


def test_optical_flow_respects_max_frames(tiny_video):
    frames, _ = sample_frames(tiny_video, method="optical_flow", max_frames=2, blur_threshold=0.0)
    assert len(frames) <= 2


def test_optical_flow_gate_rejects_all_blurred(tiny_video):
    # Threshold above any real Laplacian variance → every frame gated out
    frames, records = sample_frames(tiny_video, method="optical_flow", blur_threshold=1e12)
    assert frames == [] and records == []


def test_sample_frames_on_progress_called(tiny_video):
    # Progress is reported over the selected count (fps=10 on a 2s video = 20),
    # not the whole source-frame count.
    calls = []
    frames, _ = sample_frames(
        tiny_video, method="fps", fps=10.0, on_progress=lambda done, total: calls.append((done, total))
    )
    assert calls and calls[-1] == (len(frames), len(frames)) == (20, 20)


def test_score_frames_one_record_per_frame(tiny_video):
    records = score_frames(tiny_video, blur_threshold=0.0)
    assert len(records) == 60
    assert set(records[0]) == {
        "frame_idx",
        "blur_score",
        "exposure_mean",
        "exposure_std",
        "reject_reason",
        "score",
        "selected",
        "disparity",
        "rotation",
        "histogram_similarity",
    }
    assert records[0]["selected"] is True  # first usable frame always selected
    assert all(0.0 <= r["score"] <= 1.0 for r in records)


def test_score_frames_reject_reason_blur(tiny_video):
    # Threshold above any real Laplacian variance → every frame blur-rejected
    records = score_frames(tiny_video, blur_threshold=1e12)
    assert len(records) == 60
    assert all(r["selected"] is False and r["reject_reason"] == "blur" for r in records)


def test_score_frames_accepted_have_no_reject_reason(tiny_video):
    # tiny_video's noise fixture is well-exposed, so with blur disabled (threshold=0.0)
    # neither reject branch can fire — every record should have reject_reason=None
    records = score_frames(tiny_video, blur_threshold=0.0)
    assert all(r["reject_reason"] is None for r in records)


def test_sample_frames_missing_file_returns_empty():
    frames, records = sample_frames("/nonexistent/video.mp4", method="uniform", max_frames=6)
    assert frames == [] and records == []


########################################################################
# Target builders (pure — no video IO)
########################################################################


def test_uniform_targets_spans_endpoints():
    # "N frames spanning the video": first and last source frames are both included
    t = _uniform_targets(60, 5)
    assert t[0] == 0 and t[-1] == 59
    assert len(t) == 5
    assert t == sorted(t)


def test_uniform_targets_caps_at_total():
    # Asking for more frames than exist yields every frame, not duplicates
    assert _uniform_targets(4, 10) == [0, 1, 2, 3]


def test_uniform_targets_degenerate_inputs():
    assert _uniform_targets(0, 5) == []
    assert _uniform_targets(60, 0) == []


def test_fps_targets_uses_constant_stride():
    # "a frame every 1/fps seconds": 30 fps source at 10 fps → stride 3
    t = _fps_targets(60, 30.0, 10.0)
    assert t == list(range(0, 60, 3))
    assert len(t) == 20


def test_fps_targets_does_not_stretch_to_last_frame():
    # Stride-anchored, NOT endpoint-anchored: spacing is the contract, so the last
    # target is wherever the stride lands — this is what distinguishes fps from uniform.
    t = _fps_targets(50, 30.0, 10.0)
    assert t[-1] == 48  # not 49


def test_fps_targets_clamps_stride_to_one():
    # Requesting a rate above the source rate cannot sample sub-frame; stride floors at 1
    assert _fps_targets(10, 30.0, 120.0) == list(range(10))


def test_fps_targets_degenerate_inputs():
    assert _fps_targets(0, 30.0, 1.0) == []
    assert _fps_targets(60, 30.0, 0.0) == []


########################################################################
# fps sampler — band behaviour and cross-method validation
########################################################################


def test_fps_ceiling_decimates_rather_than_truncating(tiny_video):
    # THE regression test for `targets = targets[:max_frames]`: over-ceiling must
    # re-spread across the WHOLE video, not keep the first N and drop the tail.
    # 60 frames @30fps at 30 fps = 60 targets, capped to 10.
    frames, records = sample_frames(tiny_video, method="fps", fps=30.0, max_frames=10)
    assert len(frames) == 10
    # Truncation would put the last frame near index 9; re-spreading puts it near 59.
    assert records[-1]["frame_idx"] >= 50


def test_fps_floor_respreads_over_whole_video(tiny_video):
    # 1 fps on a 2s video = 2 targets, below the floor → re-spread at min_frames
    frames, records = sample_frames(tiny_video, method="fps", fps=1.0, min_frames=10)
    assert len(frames) == 10
    assert records[-1]["frame_idx"] >= 50


def test_fps_within_band_is_untouched(tiny_video):
    # 20 targets sits inside [5, 40] → neither guard fires, stride is preserved
    frames, records = sample_frames(tiny_video, method="fps", fps=10.0, min_frames=5, max_frames=40)
    assert len(frames) == 20
    # Stride 3 preserved: successive picks stay one validation window (radius 1) off it
    idxs = [r["frame_idx"] for r in records]
    assert all(abs(idx - 3 * i) <= 1 for i, idx in enumerate(idxs))


def test_fps_band_warns_when_it_binds(tiny_video, caplog):
    # A bound band silently changes the effective rate — it must be logged
    with caplog.at_level(logging.WARNING, logger="collab_splats.preproc.sampling"):
        sample_frames(tiny_video, method="fps", fps=30.0, max_frames=10)
    assert any("effective" in r.getMessage().lower() for r in caplog.records)


def test_fps_requires_fps_value(tiny_video):
    with pytest.raises(ValueError, match="requires fps"):
        sample_frames(tiny_video, method="fps")


def test_uniform_rejects_fps_kwarg(tiny_video):
    # The exact combination this design removes: uniform's count IS max_frames
    with pytest.raises(ValueError, match="takes no fps"):
        sample_frames(tiny_video, method="uniform", max_frames=6, fps=10.0)


def test_uniform_requires_max_frames(tiny_video):
    with pytest.raises(ValueError, match="requires max_frames"):
        sample_frames(tiny_video, method="uniform")


def test_optical_flow_rejects_fps_kwarg(tiny_video):
    with pytest.raises(ValueError, match="takes no fps"):
        sample_frames(tiny_video, method="optical_flow", fps=10.0)


def test_min_frames_rejected_by_non_fps_methods(tiny_video):
    # min_frames is an fps-only floor: uniform is pinned at its count, and
    # optical_flow cannot be forced to yield frames its selector rejected.
    with pytest.raises(ValueError, match="min_frames"):
        sample_frames(tiny_video, method="uniform", max_frames=6, min_frames=3)
    with pytest.raises(ValueError, match="min_frames"):
        sample_frames(tiny_video, method="optical_flow", min_frames=3)


def test_unknown_method_raises(tiny_video):
    with pytest.raises(ValueError, match="Unknown method"):
        sample_frames(tiny_video, method="nonsense")


########################################################################
# Public API
########################################################################


def test_public_api_surface():
    import collab_splats.preproc as preproc

    # Exactly the 7 public names — viz is opt-in and must NOT be re-exported
    assert set(preproc.__all__) == {
        "FrameStore",
        "sample_frames",
        "score_frames",
        "get_video_info",
        "extract_frame",
        "compute_blur_score",
        "check_frame_quality",
    }
    assert not hasattr(preproc, "plot_frame_scores")


def test_importing_preproc_does_not_import_matplotlib():
    # Fresh subprocess: pipeline-level import must not pull matplotlib
    import subprocess as sp
    import sys

    code = "import sys; import collab_splats.preproc; " "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    result = sp.run([sys.executable, "-c", code])
    assert result.returncode == 0
