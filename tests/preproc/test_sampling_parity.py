"""
Sampler parity: the report-driven rewrite must select the frames the inline gate did.

The baseline in data/parity_baseline.json was captured from the pre-cleanup
sample_frames on the tutorial video (plan Task 1). Sharpness is bit-identical
across the rewrite; exposure moves from the 480-wide analysis gray to the native
gray, so a diff here is either that row or a real regression — see the design
doc, section 3.3. Do not relax this test to make it pass.
"""

import json
from pathlib import Path

import pytest

BASELINE = Path(__file__).parent / "data" / "parity_baseline.json"
VIDEO = Path("data/tutorial/tutorial_example-video.mp4")

pytestmark = pytest.mark.skipif(not VIDEO.exists(), reason="tutorial video not present")


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def report():
    from collab_splats.preproc.qa import compute_video_quality

    return compute_video_quality(str(VIDEO), workers=4)


def test_uniform_matches_baseline(baseline, report):
    from collab_splats.preproc.sampling import sample_uniform

    _, records = sample_uniform(str(VIDEO), max_frames=30, report=report)

    assert [int(r["frame_idx"]) for r in records] == baseline["cases"]["uniform_max30"]


def test_fps_matches_baseline(baseline, report):
    from collab_splats.preproc.sampling import sample_fps

    _, records = sample_fps(str(VIDEO), fps=0.5, report=report)

    assert [int(r["frame_idx"]) for r in records] == baseline["cases"]["fps_0.5"]
