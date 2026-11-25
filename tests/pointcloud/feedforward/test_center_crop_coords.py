"""center_crop_coords: the shared crop-box helper behind every backend's resize rule."""

import numpy as np

from collab_splats.pointcloud.feedforward.base import center_crop_coords


def test_no_crop_is_full_frame():
    box = center_crop_coords((1920, 1080), (1920, 1080), (1920, 1080), (1, 1))
    assert box == [0, 0, 1920, 1080, 1920, 1080]


def test_crop_offsets_are_integer_in_the_resized_grid():
    # 1000x800 -> 300x240 at s = 0.3, crop 279x224: left (300 - 279) // 2 = 10, not 10.5
    s = 0.3
    box = center_crop_coords((1000, 800), (300, 240), (279, 224), (s, s))
    np.testing.assert_allclose(box, [10 / s, 8 / s, 289 / s, 232 / s, 1000, 800])
