import numpy as np

from tests.geometry.loop_closure._helpers import drive_pose_graph


def test_incremental_pose_graph_drive_shape(identity_submap_factory):
    """Batch per-submap cadence via the incremental PoseGraph returns (N, 4, 4)."""
    submaps = [identity_submap_factory(i, k=3) for i in range(2)]
    corrected = drive_pose_graph(
        submaps=submaps,
        lc_submaps=[],
        total_frames=5,  # 2*3 - 1 overlap
        overlap_frames=1,
    )
    assert corrected.shape == (5, 4, 4)
