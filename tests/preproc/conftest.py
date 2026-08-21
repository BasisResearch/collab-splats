"""Shared fixtures for preproc tests."""

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def noise_gray():
    """240x320 uniform noise — maximum high-frequency content, the sharp end of any blur ladder.

    Shared across the session, so treat it as read-only: score it, blur a copy,
    never write into it.
    """
    rng = np.random.default_rng(0)
    return (rng.random((240, 320)) * 255).astype(np.uint8)
