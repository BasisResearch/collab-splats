"""estimate_intrinsics: single-frame feedforward inference + rescale to query resolution."""
import numpy as np

from collab_splats.localization.intrinsics import estimate_intrinsics


class _FakeCreator:
    """Mimics the BaseFeedforwardCreator run surface at inference resolution 96x128."""

    def __init__(self):
        self.outputs = None
        self.seen_dir = None

    def load_model(self):
        pass

    def setup_inference(self, image_dir):
        self.seen_dir = image_dir

    def run_inference(self):
        pass

    def postprocess(self):
        class _R:
            intrinsics = np.array([[[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]]], np.float32)
            images = np.zeros((1, 3, 96, 128), np.uint8)  # (N, 3, H, W) channel-first, per FeedforwardResult

        self.outputs = _R()


def test_estimate_intrinsics_rescales_to_query_resolution():
    frame = np.zeros((192, 256, 3), dtype=np.uint8)  # 2x the fake inference res
    K = estimate_intrinsics(frame, creator=_FakeCreator())
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K[0, 0], 200.0)  # fx * (256/128)
    np.testing.assert_allclose(K[1, 1], 200.0)  # fy * (192/96)
    np.testing.assert_allclose(K[0, 2], 128.0)  # cx scaled
    np.testing.assert_allclose(K[2], [0, 0, 1])


def test_estimate_intrinsics_writes_frame_for_creator(tmp_path):
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    creator = _FakeCreator()
    estimate_intrinsics(frame, creator=creator)
    assert creator.seen_dir is not None  # creator consumed a staged image dir
