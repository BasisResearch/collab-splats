"""Verify n_loops_applied is set on the base creator after LoopClosure.run_inference()."""

import numpy as np
import torch
from unittest.mock import MagicMock, patch

from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator, FeedforwardResult
from collab_splats.geometry.loop_closure import LoopClosureConfig
from collab_splats.geometry.loop_closure.wrapper import LoopClosure


class _StubCreator(BaseFeedforwardCreator):
    """Minimal creator for unit-testing LoopClosure LC state exposure."""

    def _load_model(self, device):
        m = MagicMock()
        m.parameters.return_value = iter([torch.zeros(1)])
        return m

    def _preprocess(self, image_dir):
        return torch.zeros(40, 3, 224, 224), [None] * 40, np.zeros((40, 6))

    def _forward(self, model, views, **kwargs):
        k = views.shape[0]
        return {
            "extrinsic": np.tile(np.eye(4)[:3], (k, 1, 1)).astype(np.float32),
            "intrinsic": np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        }

    def _verify_loop_candidate(self, frame1, frame2, verify_match_ratio=0.85):
        return False, None  # no loops; tests only need submap exposure

    def _postprocess(self, raw_outputs, **kwargs):
        pass

    def extract_intermediate_features(self, frames, layer_index=-1, **kwargs):
        return getattr(self, "_stubbed_features", {})

    def _reproject(self, raw_outputs, extrinsics_3x4, intrinsics):
        return np.zeros((0, 3)), np.zeros((0, 3))

    def build_colmap(self, output_dir):
        pass


def test_n_loops_applied_set_after_run_inference():
    base = _StubCreator(camera_model="PINHOLE")
    creator = LoopClosure(base, config=LoopClosureConfig(submap_size=20, submap_overlap=4))
    creator.load_model()
    base.views = torch.zeros(40, 3, 224, 224)
    base.image_paths = [None] * 40

    # Patch retrieval extractor and pose graph optimization
    with (
        patch("collab_splats.localization.retrieval.BaseRetrievalExtractor.get") as mock_get,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", return_value=[]),
        patch("collab_splats.geometry.loop_closure.wrapper.run_pose_graph_optimization") as mock_pg,
        patch("collab_splats.geometry.loop_closure.wrapper.merge_submap_outputs") as mock_merge,
    ):
        mock_extractor = MagicMock()
        mock_extractor.return_value = torch.zeros(20, 128)
        mock_get.return_value = MagicMock(return_value=mock_extractor)
        mock_pg.return_value = np.zeros((40, 4, 4))
        mock_merge.return_value = {}
        creator.run_inference()

    # The LC loop ran end-to-end (2 submaps over 40 frames); _verify_loop_candidate
    # returns False, so no loops are accepted → n_loops_applied == 0.
    assert hasattr(base, "n_loops_applied")
    assert base.n_loops_applied == 0
