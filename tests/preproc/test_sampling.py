import logging
import subprocess as sp
import sys

import numpy as np
import pytest

from collab_splats.preproc.qa import compute_video_quality
from collab_splats.preproc.sampling import (
    OpticalFlowFrameSelector,
    filter_frame_quality,
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
from collab_splats.preproc.video import context_indices, iter_frames

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
    }
    for i in bad:
        frames["laplacian"][i] = 1.0

    return {"available": True, "frames": frames}


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


def test_filter_frame_quality_flags_soft_frames():
    mask = filter_frame_quality(_synthetic_report(10, bad=(3, 7)))

    assert mask.tolist() == [True, True, True, False, True, True, True, False, True, True]


def test_filter_frame_quality_flags_bad_exposure():
    report = _synthetic_report(4)
    report["frames"]["exposure_mean"][1] = 250.0  # blown
    report["frames"]["exposure_std"][2] = 2.0  # no contrast

    assert filter_frame_quality(report).tolist() == [True, False, False, True]


def test_filter_frame_quality_blur_max_is_off_by_default():
    report = _synthetic_report(3)
    report["frames"]["blur"] = [1.0, 1.0, 1.0]  # Crete-Roffet saturated

    assert filter_frame_quality(report).all()
    assert not filter_frame_quality(report, blur_max=0.5).any()


########################################################################
# Target positions — search_radius=0 pins each pick to its target, so these
# assert the position arithmetic without the window substitution on top of it.
########################################################################


def test_uniform_spans_endpoints(tiny_video, clean_report):
    # "N frames spanning the video": first and last source frames are both included
    _, records = sample_uniform(tiny_video, max_frames=5, report=clean_report, search_radius=0)
    idxs = [r["frame_idx"] for r in records]

    assert len(idxs) == 5
    assert idxs[0] == 0 and idxs[-1] == 59
    assert idxs == sorted(idxs)


def test_uniform_caps_at_total(tiny_video, clean_report):
    # Asking for more frames than exist yields every frame, not duplicates
    _, records = sample_uniform(tiny_video, max_frames=100, report=clean_report, search_radius=0)

    assert [r["frame_idx"] for r in records] == list(range(60))


def test_uniform_zero_max_frames_returns_empty(tiny_video, clean_report):
    assert sample_uniform(tiny_video, max_frames=0, report=clean_report) == ([], [])


def test_fps_uses_constant_stride(tiny_video, clean_report):
    # "a frame every 1/fps seconds": 30 fps source at 10 fps → stride 3
    _, records = sample_fps(tiny_video, fps=10.0, report=clean_report, search_radius=0)

    assert [r["frame_idx"] for r in records] == list(range(0, 60, 3))


def test_fps_does_not_stretch_to_last_frame(tiny_video, clean_report):
    # Stride-anchored, NOT endpoint-anchored: spacing is the contract, so the last
    # target is wherever the stride lands — this is what distinguishes fps from uniform.
    _, records = sample_fps(tiny_video, fps=10.0, report=clean_report, search_radius=0)

    assert records[-1]["frame_idx"] == 57  # not 59


def test_fps_clamps_stride_to_one(tiny_video, clean_report):
    # Requesting a rate above the source rate cannot sample sub-frame; stride floors at 1
    _, records = sample_fps(tiny_video, fps=120.0, report=clean_report, search_radius=0)

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


def test_sample_uniform_substitutes_a_neighbour_for_a_bad_frame(tiny_video):
    """
    A frame the report condemns must be replaced from within its window, keeping the count.
    """
    report = compute_video_quality(tiny_video, motion_stride=2)
    clean = [r["frame_idx"] for r in sample_uniform(tiny_video, max_frames=4, report=report)[1]]

    # Condemn every frame that clean selection picked; the count must survive
    for i in clean:
        report["frames"]["laplacian"][i] = 0.0

    frames, records = sample_uniform(tiny_video, max_frames=4, report=report)

    assert len(frames) == 4
    assert [r["frame_idx"] for r in records] != clean


def test_sample_uniform_keeps_count_when_the_report_condemns_everything(tiny_video, clean_report):
    # Best-effort: with no usable frame anywhere the window still yields its
    # sharpest candidate, so the count stays exact rather than collapsing.
    for i in range(60):
        clean_report["frames"]["laplacian"][i] = 0.0

    frames, _ = sample_uniform(tiny_video, max_frames=8, report=clean_report)

    assert len(frames) == 8


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
    # Stride 3 preserved: successive picks stay one validation window (radius 1) off it
    idxs = [r["frame_idx"] for r in records]
    assert all(abs(idx - 3 * i) <= 1 for i, idx in enumerate(idxs))


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


def test_sample_optical_flow_skips_report_rejected_frames(tiny_video):
    report = compute_video_quality(tiny_video, motion_stride=2)
    for i in range(len(report["frames"]["frame_idx"])):
        report["frames"]["laplacian"][i] = 0.0

    frames, records = sample_optical_flow(tiny_video, report=report)

    assert frames == [] and records == []


def test_samplers_require_a_report(tiny_video):
    with pytest.raises(TypeError):
        sample_uniform(tiny_video, max_frames=4)


########################################################################
# Public API
########################################################################


def test_public_api_surface():
    import collab_splats.preproc as preproc

    # Exactly the 14 public names — viz is opt-in and must NOT be re-exported.
    # The surface is the two-step contract: qa measures the whole video into a
    # report (compute_video_quality / load_video_quality), filter_frame_quality
    # turns it into a usability mask, and the three samplers select from it.
    # iter_frames is public because it is the one streamed-decode entry point.
    assert set(preproc.__all__) == {
        "DistortionProfile",
        "FrameStore",
        "analysis_gray",
        "compute_video_quality",
        "estimate_camera_distortion",
        "extract_frame",
        "filter_frame_quality",
        "get_video_info",
        "iter_frames",
        "load_video_quality",
        "sample_fps",
        "sample_optical_flow",
        "sample_uniform",
        "undistort_frames",
    }
    assert not hasattr(preproc, "plot_frame_scores")
    # The report is the deliverable; the primitives that build it stay behind
    # collab_splats.preproc.qa. Re-exporting them all would grow this surface for
    # callers who only ever want the report.
    for primitive in (
        "compute_blur",
        "compute_exposure",
        "compute_frame_quality",
        "compute_translation",
        "compute_parallax",
    ):
        assert not hasattr(preproc, primitive), primitive


def test_importing_preproc_does_not_import_matplotlib():
    # Fresh subprocess: pipeline-level import must not pull matplotlib
    code = "import sys; import collab_splats.preproc; sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    result = sp.run([sys.executable, "-c", code])

    assert result.returncode == 0


########################################################################
# search_radius: window half-width, capped below half the target spacing
########################################################################


def test_search_radius_capped_by_spacing(tiny_video):
    # Stride 3 (10 fps of 30) caps the window at +-1 however large the radius: frame 2
    # is claimed by target 3 only; an uncapped +-7 would hand it to targets 0 and 6 too
    report = _synthetic_report(n=60)
    report["frames"]["laplacian"][2] = 900.0
    _, records = sample_fps(tiny_video, fps=10.0, report=report, search_radius=7)
    idxs = [r["frame_idx"] for r in records]
    assert idxs[:2] == [0, 2] and len(idxs) == len(set(idxs))


def test_search_radius_picks_sharpest_in_wide_window(tiny_video):
    # Stride 15 (2 fps): radius 7 reaches a sharp frame 6 away, radius 3 does not
    report = _synthetic_report(n=60)
    report["frames"]["laplacian"][15] = 300.0
    report["frames"]["laplacian"][21] = 900.0
    _, wide = sample_fps(tiny_video, fps=2.0, report=report, search_radius=7)
    _, narrow = sample_fps(tiny_video, fps=2.0, report=report, search_radius=3)
    assert [r["frame_idx"] for r in wide][:2] == [0, 21]
    assert [r["frame_idx"] for r in narrow][:2] == [0, 15]


def test_sample_fps_targets_lie_on_the_context_grid(tiny_video, clean_report):
    # The guarantee the VDA context stream rests on: keyframes picked at a given rate are
    # all members of the context grid built at that same rate. search_radius=0 pins the
    # picks to the targets, so this tests the stride rule and nothing else.
    grid = set(context_indices(tiny_video, target_fps=2.0))
    _frames, records = sample_fps(tiny_video, fps=2.0, report=clean_report, search_radius=0)
    assert {r["frame_idx"] for r in records} <= grid


########################################################################
# candidates: keyframes and their blur substitutes drawn from a fixed grid
########################################################################


def test_candidates_restrict_chosen_frames_to_the_grid(tiny_video, clean_report):
    # 60-frame video, grid every 3rd frame, 10 keyframes -> every pick is a grid member
    grid = list(range(0, 60, 3))
    _frames, records = sample_uniform(tiny_video, max_frames=10, report=clean_report, search_radius=7, candidates=grid)
    chosen = [r["frame_idx"] for r in records]
    assert set(chosen) <= set(grid)
    assert len(chosen) == len(set(chosen))


def test_candidates_substitute_a_blurry_target_within_the_grid(tiny_video):
    # Grid every 3rd frame (20 members), 5 targets -> grid spacing 4 -> radius 1, so each
    # window is 3 grid members wide and substitution is actually possible. Target 30 is an
    # exact grid member and unusable, so the pick must move to 27 or 33 — never to 29 or 31.
    report = _synthetic_report(60, bad=(30,))
    grid = list(range(0, 60, 3))
    _frames, records = sample_uniform(tiny_video, max_frames=5, report=report, search_radius=7, candidates=grid)
    chosen = [r["frame_idx"] for r in records]
    assert 30 not in chosen
    assert {27, 33} & set(chosen)
    assert set(chosen) <= set(grid)


def test_candidates_none_is_byte_identical_to_today(tiny_video, clean_report):
    _f1, r1 = sample_uniform(tiny_video, max_frames=10, report=clean_report, search_radius=3)
    _f2, r2 = sample_uniform(tiny_video, max_frames=10, report=clean_report, search_radius=3, candidates=None)
    assert [r["frame_idx"] for r in r1] == [r["frame_idx"] for r in r2]


def test_sample_fps_accepts_candidates(tiny_video, clean_report):
    grid = list(range(0, 60, 3))
    _frames, records = sample_fps(tiny_video, fps=5.0, report=clean_report, search_radius=7, candidates=grid)
    assert set(r["frame_idx"] for r in records) <= set(grid)


def test_candidates_keep_a_target_that_is_already_a_grid_member(tiny_video, clean_report):
    # A target that IS a grid member must snap to itself, not forward to the next one —
    # sample_fps's targets are grid members by construction, so a forward-biased snap
    # would shift every keyframe one grid step later.
    grid = list(range(0, 60, 3))
    _frames, records = sample_uniform(tiny_video, max_frames=2, report=clean_report, search_radius=0, candidates=grid)
    assert [r["frame_idx"] for r in records] == [0, 57]


def test_fps_targets_survive_the_grid_unchanged(tiny_video, clean_report):
    # The property the whole context stream rests on, end to end: keyframes at 2 fps drawn
    # from a 6 fps grid are the SAME frames as without the grid.
    grid = context_indices(tiny_video, target_fps=6.0)
    _f1, plain = sample_fps(tiny_video, fps=2.0, report=clean_report, search_radius=0)
    _f2, gridded = sample_fps(tiny_video, fps=2.0, report=clean_report, search_radius=0, candidates=grid)
    assert [r["frame_idx"] for r in gridded] == [r["frame_idx"] for r in plain]


def test_candidates_coarser_than_the_budget_dedups_and_warns(tiny_video, clean_report, caplog):
    # 5 grid members, 10 keyframes: targets collapse. The store must never see a frame twice —
    # a duplicate makes len(store) != len(_idx_to_row) and hands out a zero-baseline pair.
    grid = list(range(0, 60, 12))
    with caplog.at_level(logging.WARNING, logger="collab_splats.preproc.sampling"):
        _frames, records = sample_uniform(
            tiny_video, max_frames=10, report=clean_report, search_radius=0, candidates=grid
        )

    chosen = [r["frame_idx"] for r in records]
    assert chosen == sorted(set(chosen))
    assert len(chosen) <= len(grid)
    assert any("collapsed" in r.getMessage() for r in caplog.records)


def test_candidates_empty_grid_is_a_hard_error(tiny_video, clean_report):
    # An empty grid has no member to snap to; returning nothing would look like a short video.
    with pytest.raises(ValueError, match="candidates is empty"):
        sample_uniform(tiny_video, max_frames=5, report=clean_report, candidates=[])


def test_candidates_need_not_be_sorted_or_unique(tiny_video, clean_report):
    # The grid is a SET of source indices, so caller order and duplicates must not move a pick.
    grid = list(range(0, 60, 3))
    _f1, a = sample_uniform(tiny_video, max_frames=5, report=clean_report, search_radius=7, candidates=grid)
    _f2, b = sample_uniform(
        tiny_video, max_frames=5, report=clean_report, search_radius=7, candidates=list(reversed(grid)) + grid
    )
    assert [r["frame_idx"] for r in a] == [r["frame_idx"] for r in b]


def test_candidates_outside_the_video_are_rejected(tiny_video, clean_report):
    # The grid indexes the report's per-frame columns, so a member past `total` is a bug
    # in the caller's grid, not a frame to clamp. The error must name the offending span.
    with pytest.raises(ValueError, match="outside the video's 60 frames"):
        sample_uniform(
            tiny_video, max_frames=5, report=clean_report, search_radius=7, candidates=list(range(-6, 90, 3))
        )


def test_candidates_below_zero_are_rejected(tiny_video, clean_report):
    # The silent-corruption half: usable[-3] wraps round to frame 57, so frame 57's sharpness
    # is attributed to index -3 and frame_idx -3 reaches frames.zarr, matching no context row.
    # A grid running past `total` at least raised an IndexError; a negative one raised nothing.
    with pytest.raises(ValueError, match=r"candidates span \[-6, 57\], outside the video's 60 frames"):
        sample_uniform(
            tiny_video, max_frames=5, report=clean_report, search_radius=7, candidates=list(range(-6, 60, 3))
        )
