import numpy as np

from collab_splats.geometry.loop_closure.closure import run_pose_graph_optimization


def test_run_pose_graph_optimization_unchanged_signature(identity_submap_factory):
    """Backward compat: existing callers still work."""
    submaps = [identity_submap_factory(i, k=3) for i in range(2)]
    corrected = run_pose_graph_optimization(
        submaps=submaps,
        lc_submaps=[],
        total_frames=5,  # 2*3 - 1 overlap
        overlap_frames=1,
    )
    assert corrected.shape == (5, 4, 4)
