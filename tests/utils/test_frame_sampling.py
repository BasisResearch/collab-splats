import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from collab_splats.utils.frame_sampling import (
    sample_frames_optical_flow,
    sample_frames_fps,
    OpticalFlowFrameSelector,
)


def test_fps_sampler_returns_empty_for_missing_file():
    frames = sample_frames_fps("nonexistent.mp4", fps=5.0)
    assert frames == []


@pytest.fixture
def tiny_video(tmp_path):
    path = str(tmp_path / "test.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    for i in range(90):
        frame = np.full((48, 64, 3), (i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_optical_flow_accepts_max_frames(tiny_video):
    frames = sample_frames_optical_flow(
        tiny_video, max_frames=5
    )
    assert isinstance(frames, list)
    assert len(frames) >= 1


def test_optical_flow_first_frame_always_included(tiny_video):
    # min_disparity=9999 means only the forced first frame is emitted
    frames = sample_frames_optical_flow(
        tiny_video, min_disparity=9999.0, max_frames=10
    )
    assert len(frames) >= 1


def test_optical_flow_frames_are_rgb(tiny_video):
    frames = sample_frames_optical_flow(
        tiny_video, max_frames=3
    )
    assert frames[0].shape[2] == 3
    assert frames[0].dtype == np.uint8


def test_selector_selects_first_frame_always():
    cv2 = pytest.importorskip("cv2")
    selector = OpticalFlowFrameSelector()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    should_select, score, _ = selector.should_select_frame(frame)
    assert should_select is True  # first frame always selected
    assert score == 1.0


def test_selector_returns_normalized_score():
    cv2 = pytest.importorskip("cv2")
    selector = OpticalFlowFrameSelector()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    selector.should_select_frame(frame)  # initialize
    frame2 = np.full((48, 64, 3), 128, dtype=np.uint8)
    _, score, _ = selector.should_select_frame(frame2)
    assert 0.0 <= score <= 1.0


def test_selector_resets_state():
    selector = OpticalFlowFrameSelector()
    selector.reset()
    assert selector.last_keyframe_gray is None


from unittest.mock import MagicMock, patch, call
import sys


def _make_mock_cap(total_frames=90, fps=30.0):
    """Return a mock VideoCapture that reads one black frame then stops."""
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: float(total_frames),
    }.get(prop, 0.0)
    mock_cap.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
    return mock_cap


def test_fps_sampler_uses_seek(monkeypatch):
    mock_cap = _make_mock_cap(total_frames=90, fps=30.0)
    # After 3 seeks (frames 0, 30, 60) make read return False to stop
    read_results = [(True, np.zeros((48, 64, 3), dtype=np.uint8))] * 3 + [(False, None)]
    mock_cap.read.side_effect = read_results

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture = MagicMock(return_value=mock_cap)
    mock_cv2.CAP_PROP_FPS = cv2.CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    mock_cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    mock_cv2.cvtColor = cv2.cvtColor

    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        frames = sample_frames_fps("fake.mp4", fps=1.0)

    seek_calls = [c for c in mock_cap.set.call_args_list
                  if c.args[0] == cv2.CAP_PROP_POS_FRAMES]
    assert len(seek_calls) >= 1, "Must use CAP_PROP_POS_FRAMES seeks"
    assert len(frames) <= 3


def test_fps_sampler_calls_on_progress(monkeypatch):
    mock_cap = _make_mock_cap(total_frames=90, fps=30.0)
    read_results = [(True, np.zeros((48, 64, 3), dtype=np.uint8))] * 3 + [(False, None)]
    mock_cap.read.side_effect = read_results

    calls = []
    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture = MagicMock(return_value=mock_cap)
    mock_cv2.CAP_PROP_FPS = cv2.CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    mock_cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    mock_cv2.cvtColor = cv2.cvtColor

    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        sample_frames_fps("fake.mp4", fps=1.0, on_progress=lambda c, t: calls.append((c, t)))

    assert len(calls) >= 1
    assert all(t == 90 for _, t in calls), "total must equal CAP_PROP_FRAME_COUNT"


def test_fps_sampler_no_progress_arg_ok(monkeypatch):
    """on_progress=None must not raise."""
    mock_cap = _make_mock_cap(total_frames=30, fps=30.0)
    mock_cap.read.side_effect = [(True, np.zeros((48, 64, 3), dtype=np.uint8))] + [(False, None)]

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture = MagicMock(return_value=mock_cap)
    mock_cv2.CAP_PROP_FPS = cv2.CAP_PROP_FPS
    mock_cv2.CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    mock_cv2.CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    mock_cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    mock_cv2.cvtColor = cv2.cvtColor

    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        frames = sample_frames_fps("fake.mp4", fps=30.0)
    assert isinstance(frames, list)


def test_optical_flow_calls_on_progress(tiny_video):
    calls = []
    sample_frames_optical_flow(
        tiny_video, max_frames=5, on_progress=lambda c, t: calls.append((c, t))
    )
    assert len(calls) >= 1
    assert all(t > 0 for _, t in calls), "total must come from CAP_PROP_FRAME_COUNT"
    assert all(c > 0 for c, _ in calls), "current must increment"


def test_optical_flow_resizes_for_analysis(tiny_video, monkeypatch):
    """Low-res resize must be called when frame width > 480."""
    # Patch cv2.resize at the point where it's imported/used
    import cv2 as _cv2

    resize_calls = []
    original_resize = _cv2.resize

    def spy_resize(src, dsize, **kwargs):
        resize_calls.append(src.shape)
        return original_resize(src, dsize, **kwargs)

    # Patch resize globally in cv2 module
    monkeypatch.setattr(_cv2, "resize", spy_resize)

    # tiny_video is 64px wide — below 480 threshold, resize must NOT be called
    sample_frames_optical_flow(tiny_video, max_frames=3)
    assert len(resize_calls) == 0, "Must not resize frames already <= 480px wide"
