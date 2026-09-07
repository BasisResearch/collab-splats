import inspect
import logging
import subprocess as sp
import sys

import numpy as np
import pytest

from collab_splats.preproc import sampling
from collab_splats.preproc.qa import compute_video_quality
from collab_splats.preproc.sampling import (
    OpticalFlowFrameSelector,
    _eligible,
    context_indices,
    filter_frame_quality,
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
from collab_splats.preproc.video import iter_frames

########################################################################
# Fixtures
########################################################################


def _synthetic_report(n=20, bad=()):
    """
    A report shaped like compute_video_quality's, with `bad` frames failing the gate.
    """
    frames = {
        "frame_idx": list(range(n)),
        "laplacian": [200.0] * n,
        "exposure_mean": [128.0] * n,
        "exposure_std": [40.0] * n,
        "blur": [0.3] * n,
        "clipped_low_frac": [0.0] * n,
        "clipped_high_frac": [0.0] * n,
    }
    for i in bad:
        frames["laplacian"][i] = 1.0

    return {"available": True, "frames": frames}


def _report(laplacian, *, clipped_low=None, clipped_high=None):
    """
    Minimal quality report carrying only the columns the filter reads.
    """
    n = len(laplacian)
    return {
        "frames": {
            "laplacian": list(laplacian),
            "clipped_low_frac": list(clipped_low if clipped_low is not None else [0.0] * n),
            "clipped_high_frac": list(clipped_high if clipped_high is not None else [0.0] * n),
        }
    }


@pytest.fixture
def clean_report():
    """
    A 60-frame report matching tiny_video, with every frame passing the filter.
    """
    return _synthetic_report(60)


########################################################################
# Selector
########################################################################


def test_selector_first_frame_scores_one(noise_gray):
    selector = OpticalFlowFrameSelector()
    score, components = selector.score_frame(noise_gray)
    assert score == 1.0
    assert components == {"disparity": 0.0, "rotation": 0.0, "histogram_similarity": 1.0}


def test_selector_scores_are_normalized():
    selector = OpticalFlowFrameSelector()
    rng = np.random.default_rng(2)
    for _ in range(5):
        gray = (rng.random((240, 320)) * 255).astype(np.uint8)
        score, _ = selector.score_frame(gray)
        assert 0.0 <= score <= 1.0


def test_selector_identical_frame_scores_low(noise_gray):
    selector = OpticalFlowFrameSelector()
    gray = noise_gray
    selector.score_frame(gray)  # seeds keyframe
    score, components = selector.score_frame(gray)
    # No motion, near-identical histogram → low combined score
    assert score < 0.3
    assert components["disparity"] < 1.0


def test_selector_has_no_stats_attr():
    sel = OpticalFlowFrameSelector(min_disparity=50.0)
    assert not hasattr(sel, "stats")


def test_selector_combine_is_monotonic_in_disparity():
    # combine() is the scoring formula viz re-thresholds through — it must rank
    # more motion above less at a fixed threshold.
    selector = OpticalFlowFrameSelector(min_disparity=50.0)
    lo = selector.combine(10.0, 0.5)
    hi = selector.combine(60.0, 0.5)
    assert hi > lo
    assert 0.0 <= lo <= hi <= 1.0


########################################################################
# Quality filter
########################################################################


def test_filter_cuts_the_soft_frame_in_an_otherwise_sharp_run():
    """
    One frame two orders of magnitude softer than its neighbors is cut.
    """
    mask = filter_frame_quality(_report([400.0] * 20 + [3.0] + [400.0] * 20))

    assert mask[20] == False  # noqa: E712 — the soft frame
    assert mask.sum() == 40


def test_filter_is_scale_free():
    """
    Multiplying every laplacian by a constant cannot change the mask.

    - That invariance is the whole point of a robust z-score on the log.
    """
    lap = [400.0, 380.0, 410.0, 3.0, 395.0, 405.0] * 8
    a = filter_frame_quality(_report(lap))
    b = filter_frame_quality(_report([x * 1000.0 for x in lap]))

    assert np.array_equal(a, b)


def test_filter_keeps_everything_when_sharpness_is_uniform():
    """
    Zero spread must not divide-by-zero into an all-False mask.
    """
    assert filter_frame_quality(_report([250.0] * 30)).all()


def test_filter_cuts_a_clipped_frame():
    """
    Clipping is an absolute rule: >25% destroyed pixels is out regardless of sharpness.
    """
    lap = [400.0] * 10
    low = [0.0] * 9 + [0.30]
    mask = filter_frame_quality(_report(lap, clipped_low=low))

    assert mask[9] == False  # noqa: E712
    assert mask[:9].all()


def test_filter_clipping_is_the_sum_of_both_tails():
    """
    0.15 crushed + 0.15 blown is 0.30 destroyed, over the 0.25 ceiling.
    """
    mask = filter_frame_quality(
        _report([400.0] * 4, clipped_low=[0.0, 0.0, 0.0, 0.15], clipped_high=[0.0, 0.0, 0.0, 0.15])
    )

    assert mask[3] == False  # noqa: E712


def test_filter_handles_an_empty_report():
    """
    A report with no rows returns an empty mask, not an exception.
    """
    assert filter_frame_quality(_report([])).shape == (0,)


def test_filter_no_longer_takes_the_deleted_thresholds():
    """
    laplacian_min et al are gone; passing one is a TypeError, not a silent no-op.
    """
    for dead in ("laplacian_min", "exposure_mean_range", "exposure_min_std", "blur_max"):
        with pytest.raises(TypeError):
            filter_frame_quality(_report([400.0] * 5), **{dead: 1})


########################################################################
# Target positions — uniform spreads evenly over the eligible pool, and fps snaps
# its targets to that same pool, so on an all-usable report both assert the position
# arithmetic with nothing on top of it.
########################################################################


def test_uniform_spans_endpoints(tiny_video, clean_report):
    # "N frames spanning the video": first and last source frames are both included
    _, records = sample_uniform(tiny_video, max_frames=5, report=clean_report)
    idxs = [r["frame_idx"] for r in records]

    assert len(idxs) == 5
    assert idxs[0] == 0 and idxs[-1] == 59
    assert idxs == sorted(idxs)


def test_uniform_caps_at_total(tiny_video, clean_report):
    # Asking for more frames than exist yields every frame, not duplicates
    _, records = sample_uniform(tiny_video, max_frames=100, report=clean_report)

    assert [r["frame_idx"] for r in records] == list(range(60))


def test_uniform_zero_max_frames_returns_empty(tiny_video, clean_report):
    assert sample_uniform(tiny_video, max_frames=0, report=clean_report) == ([], [])


def test_fps_uses_constant_stride(tiny_video, clean_report):
    # "a frame every 1/fps seconds": 30 fps source at 10 fps → stride 3
    _, records = sample_fps(tiny_video, fps=10.0, report=clean_report)

    assert [r["frame_idx"] for r in records] == list(range(0, 60, 3))


def test_fps_does_not_stretch_to_last_frame(tiny_video, clean_report):
    # Stride-anchored, NOT endpoint-anchored: spacing is the contract, so the last
    # target is wherever the stride lands — this is what distinguishes fps from uniform.
    _, records = sample_fps(tiny_video, fps=10.0, report=clean_report)

    assert records[-1]["frame_idx"] == 57  # not 59


def test_fps_clamps_stride_to_one(tiny_video, clean_report):
    # Requesting a rate above the source rate cannot sample sub-frame; stride floors at 1
    _, records = sample_fps(tiny_video, fps=120.0, report=clean_report)

    assert [r["frame_idx"] for r in records] == list(range(60))


########################################################################
# Samplers
########################################################################


def test_sample_uniform_takes_the_report(tiny_video):
    report = compute_video_quality(tiny_video, motion_stride=2)

    frames, records = sample_uniform(tiny_video, max_frames=4, report=report)

    assert len(frames) == len(records) == 4
    assert frames[0].shape == (240, 320, 3)
    assert set(records[0]) == {"frame_idx", "blur_score"}
    assert [r["frame_idx"] for r in records] == sorted(r["frame_idx"] for r in records)


def test_sample_uniform_avoids_the_frames_the_report_condemns(tiny_video):
    """
    A frame the report condemns leaves the pool, so the picks move and the count survives.
    """
    report = compute_video_quality(tiny_video, motion_stride=2)
    clean = [r["frame_idx"] for r in sample_uniform(tiny_video, max_frames=4, report=report)[1]]

    # Condemn every frame that clean selection picked; the count must survive
    for i in clean:
        report["frames"]["laplacian"][i] = 0.0

    frames, records = sample_uniform(tiny_video, max_frames=4, report=report)

    assert len(frames) == 4
    assert [r["frame_idx"] for r in records] != clean


def test_sample_uniform_raises_when_the_report_condemns_everything(tiny_video, clean_report):
    # An empty pool is a config error, not an empty scene written silently — there is no
    # window left to take a best-effort argmax over. Condemnation is spelled with the
    # clipping rule because the sharpness rule is relative: a uniformly soft video has no
    # outlier to cut.
    for i in range(60):
        clean_report["frames"]["clipped_low_frac"][i] = 1.0

    with pytest.raises(ValueError, match="no eligible frames"):
        sample_uniform(tiny_video, max_frames=8, report=clean_report)


def test_sample_uniform_decodes_in_one_select_pass(tiny_video, clean_report, monkeypatch):
    """
    Uniform sampling must use the select filter, in exactly one ffmpeg call.
    """
    import collab_splats.preproc.sampling as s

    real = s.iter_frames
    calls = []

    def counting(path, **kwargs):
        calls.append(kwargs.get("indices"))
        return real(path, **kwargs)

    monkeypatch.setattr(s, "iter_frames", counting)

    frames, _ = sample_uniform(tiny_video, max_frames=4, report=clean_report)

    assert len(frames) == 4
    # One call, and it named the frames it wanted rather than decoding everything
    assert len(calls) == 1 and calls[0] is not None


def test_sample_uniform_missing_file_returns_empty(clean_report):
    assert sample_uniform("/nonexistent/video.mp4", max_frames=6, report=clean_report) == ([], [])


def test_sampled_frames_are_rgb(tiny_video, clean_report):
    frames, records = sample_fps(tiny_video, fps=10.0, report=clean_report)
    bgr = [f for _, f in iter_frames(tiny_video)]

    # RGB return means channel order is reversed vs the BGR decode
    np.testing.assert_array_equal(frames[0], bgr[records[0]["frame_idx"]][:, :, ::-1])


def test_sample_fps_reports_progress(tiny_video, clean_report):
    # Progress is reported over the selected count (fps=10 on a 2s video = 20),
    # not the whole source-frame count.
    calls = []

    frames, _ = sample_fps(
        tiny_video, fps=10.0, report=clean_report, on_progress=lambda done, total: calls.append((done, total))
    )

    assert calls and calls[-1] == (len(frames), len(frames)) == (20, 20)


def test_sample_fps_respects_the_frame_band(tiny_video):
    report = compute_video_quality(tiny_video, motion_stride=2)

    frames, _ = sample_fps(tiny_video, fps=30.0, max_frames=5, report=report)

    assert len(frames) <= 5


def test_sample_fps_rejects_a_missing_rate(tiny_video, clean_report):
    with pytest.raises(ValueError, match="positive fps"):
        sample_fps(tiny_video, fps=None, report=clean_report)


def test_sample_fps_ceiling_decimates_rather_than_truncating(tiny_video, clean_report):
    # THE regression test for `targets = targets[:max_frames]`: over-ceiling must
    # re-spread across the WHOLE video, not keep the first N and drop the tail.
    # 60 frames @30fps at 30 fps = 60 targets, capped to 10.
    frames, records = sample_fps(tiny_video, fps=30.0, max_frames=10, report=clean_report)

    assert len(frames) == 10
    # Truncation would put the last frame near index 9; re-spreading puts it near 59.
    assert records[-1]["frame_idx"] >= 50


def test_sample_fps_floor_respreads_over_whole_video(tiny_video, clean_report):
    # 1 fps on a 2s video = 2 targets, below the floor → re-spread at min_frames
    frames, records = sample_fps(tiny_video, fps=1.0, min_frames=10, report=clean_report)

    assert len(frames) == 10
    assert records[-1]["frame_idx"] >= 50


def test_sample_fps_within_band_is_untouched(tiny_video, clean_report):
    # 20 targets sits inside [5, 40] → neither guard fires, stride is preserved
    frames, records = sample_fps(tiny_video, fps=10.0, min_frames=5, max_frames=40, report=clean_report)

    assert len(frames) == 20
    # Stride 3 preserved: every frame is eligible, so each target snaps onto itself
    idxs = [r["frame_idx"] for r in records]
    assert idxs == [3 * i for i in range(20)]


def test_sample_fps_warns_when_the_band_binds(tiny_video, clean_report, caplog):
    # A bound band silently changes the effective rate — it must be logged
    with caplog.at_level(logging.WARNING, logger="collab_splats.preproc.sampling"):
        sample_fps(tiny_video, fps=30.0, max_frames=10, report=clean_report)

    assert any("effective" in r.getMessage().lower() for r in caplog.records)


def test_sample_optical_flow_selects_the_first_frame(tiny_video, clean_report):
    frames, records = sample_optical_flow(tiny_video, report=clean_report)

    assert len(frames) >= 1
    assert records[0]["frame_idx"] == 0


def test_sample_optical_flow_records_have_source_indices(tiny_video, clean_report):
    _, records = sample_optical_flow(tiny_video, report=clean_report)
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


def test_sample_optical_flow_respects_max_frames(tiny_video, clean_report):
    frames, _ = sample_optical_flow(tiny_video, max_frames=2, report=clean_report)

    assert len(frames) <= 2


def test_sample_optical_flow_skips_frames_outside_the_pool(tiny_video):
    # The first ten frames leave the pool, so the selector never sees them — not as picks
    # and not as the reference the frames after them are scored against.
    report = compute_video_quality(tiny_video, motion_stride=2)
    for i in range(10):
        report["frames"]["clipped_low_frac"][i] = 1.0

    _frames, records = sample_optical_flow(tiny_video, report=report)

    assert records and all(r["frame_idx"] >= 10 for r in records)


def test_sample_optical_flow_empty_pool_is_a_hard_error(tiny_video):
    # Every sampler shares the pool, so an all-condemned video is the same config error
    # here as it is for uniform — never an empty scene written silently.
    report = compute_video_quality(tiny_video, motion_stride=2)
    for i in range(len(report["frames"]["frame_idx"])):
        report["frames"]["clipped_low_frac"][i] = 1.0

    with pytest.raises(ValueError, match="no eligible frames"):
        sample_optical_flow(tiny_video, report=report)


def test_samplers_require_a_report(tiny_video):
    with pytest.raises(TypeError):
        sample_uniform(tiny_video, max_frames=4)


########################################################################
# Public API
########################################################################


def test_public_api_surface():
    import collab_splats.preproc as preproc

    # Exactly the 17 public names — viz is opt-in and must NOT be re-exported.
    # The surface is the two-step contract: qa measures the whole video into a
    # report (compute_video_quality / load_video_quality), filter_frame_quality
    # turns it into a usability mask, and the three samplers select from it.
    # frames.py's five flat functions are the images/ keyframe store itself.
    # iter_frames is public because it is the one streamed-decode entry point.
    assert set(preproc.__all__) == {
        "analysis_gray",
        "calibrate_camera",
        "compute_video_quality",
        "extract_frame",
        "filter_frame_quality",
        "frame_idx_from_path",
        "frame_paths",
        "get_video_info",
        "iter_frames",
        "load_video_quality",
        "read_frames",
        "read_manifest",
        "sample_fps",
        "sample_optical_flow",
        "sample_uniform",
        "undistort_frames",
        "write_frames",
    }
    assert not hasattr(preproc, "plot_frame_scores")
    # The report is the deliverable; the primitives that build it stay behind
    # collab_splats.preproc.qa. Re-exporting them all would grow this surface for
    # callers who only ever want the report.
    for primitive in (
        "compute_blur",
        "compute_exposure",
        "compute_frame_quality",
        "compute_pair_motion",
        "detect_orb",
    ):
        assert not hasattr(preproc, primitive), primitive


def test_importing_preproc_does_not_import_matplotlib():
    # Fresh subprocess: pipeline-level import must not pull matplotlib
    code = "import sys; import collab_splats.preproc; sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    result = sp.run([sys.executable, "-c", code])

    assert result.returncode == 0


def test_sample_fps_targets_lie_on_the_context_grid(tiny_video, clean_report):
    # The guarantee the VDA context stream rests on: keyframes picked at a given rate are
    # all members of the context grid built at that same rate. Every frame is usable here,
    # so the snap is the identity and this tests the stride rule and nothing else.
    grid = set(context_indices(tiny_video, target_fps=2.0))
    _frames, records = sample_fps(tiny_video, fps=2.0, report=clean_report)
    assert {r["frame_idx"] for r in records} <= grid


########################################################################
# The eligible pool — the quality mask
########################################################################


def test_eligible_drops_the_frames_the_mask_condemns():
    """
    The pool is the quality mask's keep-set, ascending.
    """
    report = _report([400.0] * 20 + [3.0] + [400.0] * 9)
    pool = _eligible(report, quality=None)

    # 20 is the only frame the mask condemns
    assert np.array_equal(pool, np.array([i for i in range(30) if i != 20]))


def test_eligible_raises_when_the_pool_is_empty():
    """
    An empty pool is a config error, not an empty scene written silently.
    """
    # Every frame fully clipped — sharp, but nothing recoverable in the pixels
    with pytest.raises(ValueError, match="no eligible frames"):
        _eligible(_report([400.0] * 10, clipped_high=[0.9] * 10), quality=None)


def test_sample_uniform_spans_the_eligible_pool(monkeypatch, tmp_path):
    """
    Picks are evenly spaced in POOL index, and never land on a condemned frame.
    """
    # 30 frames, the middle 10 blurred out
    report = _report([400.0] * 10 + [2.0] * 10 + [400.0] * 10)

    # Only the decode is stubbed: the pool comes from the report, so uniform sampling
    # never probes the video. A re-added probe would see a path that does not exist.
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i, np.uint8)) for i in indices],
    )

    frames, records = sampling.sample_uniform(str(tmp_path / "v.mp4"), max_frames=4, report=report)

    picked = [r["frame_idx"] for r in records]
    assert len(frames) == 4
    assert all(i < 10 or i >= 20 for i in picked), picked
    assert picked == sorted(picked)


def test_sample_uniform_returns_the_whole_pool_when_it_is_short(monkeypatch, tmp_path, caplog):
    """
    A pool smaller than max_frames returns the pool and logs the shortfall.
    """
    report = _report([400.0] * 3)
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i, np.uint8)) for i in indices],
    )

    with caplog.at_level("WARNING"):
        frames, records = sampling.sample_uniform(str(tmp_path / "v.mp4"), max_frames=10, report=report)

    assert [r["frame_idx"] for r in records] == [0, 1, 2]
    assert "eligible" in caplog.text


########################################################################
# fps snapping — targets land on the pool, and the band re-spreads over it
########################################################################


def test_sample_fps_snaps_targets_to_the_nearest_eligible_frame(monkeypatch, tmp_path):
    """
    Constant-rate targets land on the closest eligible index, never on a condemned one.
    """
    # 60 frames at 30 fps; frames 10-14 blurred out. fps=3 targets 0, 10, 20, ...
    report = _report([400.0] * 10 + [2.0] * 5 + [400.0] * 45)

    monkeypatch.setattr(sampling, "get_video_info", lambda p, **k: {"total_frames": 60, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i % 251, np.uint8)) for i in indices],
    )

    frames, records = sampling.sample_fps(str(tmp_path / "v.mp4"), fps=3.0, report=report)
    picked = [r["frame_idx"] for r in records]

    # Target 10 is condemned; 9 is one frame away and 15 is five, so 9 wins
    assert 10 not in picked
    assert 9 in picked
    assert picked == sorted(set(picked))


def test_sample_fps_respreads_outside_the_band(monkeypatch, tmp_path, caplog):
    """
    A count outside [min_frames, max_frames] re-spreads over the whole pool, never truncates.
    """
    report = _report([400.0] * 60)
    monkeypatch.setattr(sampling, "get_video_info", lambda p, **k: {"total_frames": 60, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i % 251, np.uint8)) for i in indices],
    )

    with caplog.at_level("WARNING"):
        frames, records = sampling.sample_fps(str(tmp_path / "v.mp4"), fps=15.0, report=report, max_frames=6)

    picked = [r["frame_idx"] for r in records]
    assert len(picked) == 6
    assert picked[-1] >= 55, "re-spread must still span the video, not truncate at frame 6"
    assert "re-spread" in caplog.text


def test_context_indices_lives_in_sampling():
    """
    It computes a selection grid, so it belongs to this module — and a supplied probe
    short-circuits the real one, which a nonexistent path proves.
    """
    assert context_indices("x.mp4", target_fps=2.0, info={"total_frames": 10, "fps": 10.0}) == [0, 5]


def test_context_indices_matches_sample_fps_stride(tiny_video):
    # tiny_video is 60 frames @ 30 fps -> fps=10 gives stride 3
    grid = context_indices(tiny_video, target_fps=10.0)

    assert grid[:4] == [0, 3, 6, 9]
    assert len(grid) == 20


def test_context_indices_floors_stride_at_one(tiny_video):
    # A target rate above the source rate cannot sample sub-frame
    assert context_indices(tiny_video, target_fps=1000.0) == list(range(60))


def test_context_indices_rejects_a_nonpositive_fps(tiny_video):
    with pytest.raises(ValueError, match="positive target_fps"):
        context_indices(tiny_video, target_fps=0)


def test_context_indices_empty_video_returns_no_indices(tiny_video):
    # A probe reporting zero frames short-circuits before any stride arithmetic
    assert context_indices(tiny_video, target_fps=2.0, info={"total_frames": 0, "fps": 30.0}) == []


def test_samplers_no_longer_take_search_radius():
    """
    The window search is gone; the pool replaced it.
    """
    for fn in (sampling.sample_uniform, sampling.sample_fps, sampling.sample_optical_flow):
        assert "search_radius" not in inspect.signature(fn).parameters, fn.__name__
