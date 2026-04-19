import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from collab_splats.semantics.frame_sampling import (
    _rotation_map,
    sample_frames_optical_flow,
    sample_frames_fps,
    OpticalFlowFrameSelector,
)


def test_rotation_90_maps_to_clockwise():
    assert _rotation_map()[90] == cv2.ROTATE_90_CLOCKWISE


def test_rotation_270_maps_to_counterclockwise():
    assert _rotation_map()[270] == cv2.ROTATE_90_COUNTERCLOCKWISE


def test_rotation_180_unchanged():
    assert _rotation_map()[180] == cv2.ROTATE_180


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
