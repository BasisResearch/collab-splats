import numpy as np
import pytest
import torch
from pathlib import Path

from collab_splats.pointcloud.loop_closure import PoseGraph, Submap
from collab_splats.pointcloud.loop_closure.eval import (
    _classify_edges,
    _per_edge_error,
    capture_pose_graph_loss,
    ate_translation,
    rpe,
    umeyama_align,
)


@pytest.fixture
def pg_with_loop():
    pg = PoseGraph()
    H0 = np.eye(4, dtype=np.float64)
    H1 = np.eye(4, dtype=np.float64)
    H2 = np.eye(4, dtype=np.float64)
    H3 = np.eye(4, dtype=np.float64)
    # nodes
    pg.add_node(0, H0); pg.add_node(1, H1); pg.add_node(2, H2); pg.add_node(3, H3)
    pg.add_prior(0, H0)
    # sequential edges (3 total)
    pg.add_sequential_edge(0, 1, H1); pg.add_sequential_edge(1, 2, H2)
    pg.add_sequential_edge(2, 3, H3)
    # loop edge (1 total)
    pg.add_loop_edge(0, 3, H3)
    return pg


def test_classify_edges_returns_two_keys(pg_with_loop):
    groups = _classify_edges(pg_with_loop._graph)
    assert set(groups) == {"sequential", "loop"}


def test_classify_edges_sequential_count_matches_expected(pg_with_loop):
    """All 4 BetweenFactors classify as 'sequential' under VGGT-SLAM noise parity.

    Since 011c56f ("match VGGT-SLAM PGO noise exactly"), add_loop_edge uses the
    same plain Gaussian (Diagonal.Sigmas) noise as add_sequential_edge — neither
    is Robust(Huber). _classify_edges keys off isinstance(nm, noiseModel.Robust),
    so with no Robust factors all 3 sequential + 1 loop BetweenFactors land in
    'sequential' (4 total); the prior is skipped.
    """
    groups = _classify_edges(pg_with_loop._graph)
    assert len(groups["sequential"]) == 4


def test_classify_edges_loop_count_matches_expected(pg_with_loop):
    """No edge is Robust-noised post-VGGT-SLAM-parity, so 'loop' is empty.

    See test_classify_edges_sequential_count_matches_expected — loop edges are no
    longer noise-distinguishable from sequential ones (011c56f), so _classify_edges
    cannot separate them and 'loop' is empty.
    """
    groups = _classify_edges(pg_with_loop._graph)
    assert len(groups["loop"]) == 0


def test_classify_edges_skips_prior_factor(pg_with_loop):
    """The PriorFactor on frame 0 should not appear in any group."""
    groups = _classify_edges(pg_with_loop._graph)
    total_classified = sum(len(v) for v in groups.values())
    # 3 sequential + 1 loop = 4; total factors = 5 (incl. prior)
    assert total_classified == pg_with_loop._graph.size() - 1


def test_per_edge_error_keys_match_classifier(pg_with_loop):
    groups = _classify_edges(pg_with_loop._graph)
    errors = _per_edge_error(pg_with_loop._graph, pg_with_loop._initial, groups)
    assert set(errors) == {"sequential", "loop"}


def test_per_edge_error_returns_floats(pg_with_loop):
    groups = _classify_edges(pg_with_loop._graph)
    errors = _per_edge_error(pg_with_loop._graph, pg_with_loop._initial, groups)
    for v in errors.values():
        assert isinstance(v, float)
        assert v >= 0.0


def test_per_edge_error_sum_matches_classified_total(pg_with_loop):
    """Sum of per-edge errors == sum of individual classified factor errors."""
    groups = _classify_edges(pg_with_loop._graph)
    errors = _per_edge_error(pg_with_loop._graph, pg_with_loop._initial, groups)
    summed = sum(errors.values())
    expected = sum(
        pg_with_loop._graph.at(i).error(pg_with_loop._initial)
        for indices in groups.values()
        for i in indices
    )
    assert abs(summed - expected) < 1e-9


def test_capture_loss_returns_required_keys(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert set(trace) >= {"iterations", "per_edge_initial", "per_edge_final", "optimized_values"}


def test_capture_loss_iterations_is_nonempty_list_of_floats(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert isinstance(trace["iterations"], list)
    assert len(trace["iterations"]) >= 1
    assert all(isinstance(x, float) for x in trace["iterations"])


def test_capture_loss_curve_decreases_or_holds(pg_with_loop):
    """LM should never increase total error step-to-step; final ≤ initial."""
    trace = capture_pose_graph_loss(pg_with_loop)
    assert trace["iterations"][-1] <= trace["iterations"][0] + 1e-9


def test_capture_loss_does_not_mutate_input_graph(pg_with_loop):
    """Capture must not mutate pg._initial or pg._graph."""
    keys_before = set(pg_with_loop._initial.keys())
    poses_before = {
        k: pg_with_loop._initial.atSL4(k).matrix().copy() for k in keys_before
    }
    graph_size_before = pg_with_loop._graph.size()

    capture_pose_graph_loss(pg_with_loop)

    assert set(pg_with_loop._initial.keys()) == keys_before
    assert pg_with_loop._graph.size() == graph_size_before
    for k in keys_before:
        assert np.allclose(poses_before[k], pg_with_loop._initial.atSL4(k).matrix())


def test_capture_loss_per_edge_keys_match_classifier(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert set(trace["per_edge_initial"]) == {"sequential", "loop"}
    assert set(trace["per_edge_final"]) == {"sequential", "loop"}


def test_capture_loss_returns_converged_flag(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert "converged" in trace
    assert isinstance(trace["converged"], bool)


def test_capture_loss_returns_n_iterations(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert "n_iterations" in trace
    assert trace["n_iterations"] == len(trace["iterations"]) - 1
    assert trace["n_iterations"] >= 0


def test_capture_loss_max_iterations_respected(pg_with_loop):
    """If max_iterations=1, iterations should have at most 2 entries (initial + 1 step)."""
    trace = capture_pose_graph_loss(pg_with_loop, max_iterations=1)
    assert len(trace["iterations"]) <= 2


def test_capture_loss_rejects_max_iterations_zero(pg_with_loop):
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        capture_pose_graph_loss(pg_with_loop, max_iterations=0)


def test_umeyama_align_returns_aligned_poses_and_transform():
    pred = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    gt = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    aligned, T_align = umeyama_align(pred, gt)
    assert aligned.shape == (2, 4, 4)
    assert T_align.shape == (4, 4)


def test_ate_translation_returns_error_dict():
    pred = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    gt = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    result = ate_translation(pred, gt)
    assert set(result) >= {"rmse", "mean", "median", "max", "per_frame"}
    assert result["rmse"] >= 0.0


def test_rpe_returns_error_dict():
    pred = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    gt = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    result = rpe(pred, gt)
    assert set(result) >= {"trans_rmse", "rot_rmse_deg"}
    assert result["trans_rmse"] >= 0.0


def test_capture_pose_graph_loss_importable_from_package():
    from collab_splats.pointcloud.loop_closure import capture_pose_graph_loss as cpl
    assert callable(cpl)


# ── merge_submap_outputs dedup index ──────────────────────────────────────────

def _make_submap_with_raw(submap_id, frame_start, k, h=4, w=4):
    """Build a minimal Submap with raw_outputs carrying intrinsics/depth/depth_conf."""
    poses = np.tile(np.eye(4), (k, 1, 1)).astype(np.float32)
    intr = np.tile(np.eye(3), (k, 1, 1)).astype(np.float32)
    raw = {
        "intrinsics": intr.copy(),
        "extrinsic": poses[:, :3, :].copy(),
        "depth": np.zeros((k, h, w), dtype=np.float32),
        "depth_conf": np.ones((k, h, w), dtype=np.float32),
    }
    s = Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, h, w),
        poses=poses,
        intrinsics=intr,
        retrieval_vectors=torch.zeros(k, 16),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
        raw_outputs=raw,
        frame_start=frame_start,
    )
    return s


def test_merge_submap_outputs_dedup_rows_length_and_no_missing():
    """_dedup_rows must have length N (total unique frames) with no -1 entries."""
    from collab_splats.pointcloud.loop_closure.closure import merge_submap_outputs

    # submap_size=5, overlap=2, step=3 → submaps at [0..4], [3..7], [6..9]
    # N=10, M=5+5+4=14
    submaps = [
        _make_submap_with_raw(0, frame_start=0, k=5),
        _make_submap_with_raw(1, frame_start=3, k=5),
        _make_submap_with_raw(2, frame_start=6, k=4),
    ]
    N = 10
    corrected = np.tile(np.eye(4), (N, 1, 1)).astype(np.float32)

    merged = merge_submap_outputs(submaps, corrected)

    dedup = merged["_dedup_rows"]
    assert dedup.shape == (N,), f"expected ({N},), got {dedup.shape}"
    assert (dedup >= 0).all(), "some global frames have no M-row mapping"
    M = sum(len(s.poses) for s in submaps)
    assert (dedup < M).all(), "dedup index out of bounds for M-expanded arrays"


def test_merge_submap_outputs_dedup_rows_first_occurrence():
    """For overlap frames, _dedup_rows should point to the FIRST submap occurrence."""
    from collab_splats.pointcloud.loop_closure.closure import merge_submap_outputs

    # submap 0 covers frames 0-2, submap 1 covers frames 1-3 (overlap at 1,2)
    # M-expanded rows: [0,1,2] from s0 (rows 0,1,2) then [1,2,3] from s1 (rows 3,4,5)
    # Frame 1 first appears at M-row 1 (submap 0), NOT row 3 (submap 1).
    submaps = [
        _make_submap_with_raw(0, frame_start=0, k=3),
        _make_submap_with_raw(1, frame_start=1, k=3),
    ]
    N = 4
    corrected = np.tile(np.eye(4), (N, 1, 1)).astype(np.float32)

    merged = merge_submap_outputs(submaps, corrected)
    dedup = merged["_dedup_rows"]

    # Frame 0 → row 0 (only in s0)
    assert dedup[0] == 0
    # Frame 1 → row 1 (first occurrence in s0, not row 3 in s1)
    assert dedup[1] == 1
    # Frame 2 → row 2 (first occurrence in s0, not row 4 in s1)
    assert dedup[2] == 2
    # Frame 3 → row 5 (only in s1, local index 2 → M-row 3+2=5)
    assert dedup[3] == 5


def test_run_dedup_aligns_intrinsics_to_unique_frames():
    """LoopClosure.run() aligns M-expanded intrinsics down to N unique frames.

    The old _apply_ba wrapper API is gone; the M->N alignment now lives in
    LoopClosure.run() via raw_outputs["_dedup_rows"]. After run(), the merged
    M-row intrinsics must be remapped to N rows == intrinsics[dedup].
    """
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.feedforward import FeedforwardResult

    N = 4   # unique global frames (extrinsics rows)
    M = 6   # overlap-expanded merged rows (e.g. 2 submaps of 4 with overlap 2)
    H, W = 8, 8

    # FeedforwardResult where extrinsics=N but intrinsics/images/etc are M-rows.
    extrinsics_N = np.tile(np.eye(4), (N, 1, 1)).astype(np.float32)
    # Distinct per-row intrinsics so dedup remap is content-verifiable.
    intrinsics_M = np.tile(np.eye(3), (M, 1, 1)).astype(np.float32)
    intrinsics_M[:, 0, 0] = np.arange(M, dtype=np.float32) + 1.0

    result = FeedforwardResult(
        points=np.zeros((N * H * W, 3), dtype=np.float32),
        colors=np.zeros((N * H * W, 3), dtype=np.uint8),
        extrinsics=extrinsics_N,
        intrinsics=intrinsics_M,
        image_paths=[Path(f"frame_{i:04d}.png") for i in range(N)],
        original_coords=np.zeros((N, 6), dtype=np.float32),
        model_width=W,
        model_height=H,
        images=np.zeros((M, 3, H, W), dtype=np.float32),
        confidence=np.ones((M, H, W), dtype=np.float32),
        world_points=np.zeros((M, H, W, 3), dtype=np.float32),
    )

    # _dedup_rows maps each of the N unique frames to a merged M-row index.
    dedup_rows = np.array([0, 1, 3, 5], dtype=np.int64)

    # Minimal fake base: no-op pipeline stages; outputs/raw_outputs settable.
    class FakeBase:
        def __init__(self) -> None:
            self.outputs = result
            self.raw_outputs = {"_dedup_rows": dedup_rows}

        def load_model(self) -> None:
            pass

        def setup_inference(self, image_dir) -> None:
            pass

        def run_inference(self, **kwargs) -> None:
            pass

        def postprocess(self, **kwargs) -> None:
            pass

    base = FakeBase()
    # Bypass the LC loop override so run_inference stays the no-op base stage.
    wrapper = LoopClosure(base)
    wrapper.run_inference = base.run_inference

    out = wrapper.run(Path("unused"))

    # Intrinsics realigned to N rows and equal to the deduped selection.
    assert out.intrinsics.shape[0] == N
    np.testing.assert_array_equal(out.intrinsics, intrinsics_M[dedup_rows])
    # Other M-row arrays are realigned too.
    assert out.images.shape[0] == N
    assert out.confidence.shape[0] == N
    assert out.world_points.shape[0] == N
    # Wrapper writes the realigned result back onto the base.
    assert base.outputs is out
