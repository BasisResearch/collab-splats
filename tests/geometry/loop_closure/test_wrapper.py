# tests/pointcloud/test_wrappers.py
import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.loop_closure.wrapper import _trim_forward_outputs
from collab_splats.pointcloud.feedforward import FeedforwardResult


def _make_ff_result(**overrides):
    defaults = dict(
        points=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        image_paths=[Path("a.jpg"), Path("b.jpg")],
        original_coords=np.zeros((2, 6), dtype=np.float32),
        model_width=224,
        model_height=224,
    )
    defaults.update(overrides)
    return FeedforwardResult(**defaults)


def _make_mock_creator(ff_result):
    """Returns a mock that duck-types as BaseFeedforwardCreator."""
    m = MagicMock()
    m.outputs = ff_result
    m.raw_outputs = {}
    m._reproject.return_value = (ff_result.points, ff_result.colors)
    return m


########################################################
########## FeedforwardResult field tests ##############
########################################################


def test_feedforward_result_new_fields_default_none():
    r = _make_ff_result()
    assert r.images is None
    assert r.confidence is None
    assert r.world_points is None


########################################################
########## BundleAdjustmentConfig tests ###############
########################################################


def test_bundle_adjustment_config_defaults():
    from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig

    cfg = BundleAdjustmentConfig()
    assert cfg.max_reproj_error == 4.0
    assert cfg.lm_steps == 40
    assert cfg.shared_camera is True  # track-quality parity default (2026-08-19)
    assert cfg.min_inliers_per_frame == 64


def test_bundle_adjustment_config_custom():
    from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig

    cfg = BundleAdjustmentConfig(max_reproj_error=2.0, lm_steps=20)
    assert cfg.max_reproj_error == 2.0
    assert cfg.lm_steps == 20


########################################################
########## BaseFeedforwardCreator field tests #########
########################################################


def test_vggtx_postprocess_populates_ba_fields():
    """VGGTXCreator._postprocess() must populate images/confidence/world_points."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    creator = VGGTXCreator.__new__(VGGTXCreator)
    creator.conf_threshold = 1.0
    creator.image_paths = [Path("a.jpg"), Path("b.jpg")]
    creator.original_coords = np.zeros((2, 6), dtype=np.float32)

    N, H, W = 2, 8, 8
    raw_outputs = {
        "depth": np.ones((N, H, W, 1), dtype=np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": torch.zeros(N, 3, H, W),
        "extrinsic": np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        "intrinsics_downsampled": np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
    }

    with patch(
        "collab_splats.pointcloud.feedforward.unproject_and_filter_points",
        return_value=(np.zeros((5, 3), dtype=np.float32), np.zeros((5, 3), dtype=np.uint8)),
    ):
        result = creator._postprocess(raw_outputs)

    assert result.images is not None
    assert result.confidence is not None
    assert result.world_points is not None
    assert result.images.shape[0] == N
    assert result.confidence.shape == (N, H, W)


def test_vggtx_no_use_ba_field():
    """VGGTXCreator must not have use_ba after refactor."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    field_names = {f.name for f in dataclasses.fields(VGGTXCreator)}
    assert "use_ba" not in field_names


def test_vggtx_has_reproject():
    """VGGTXCreator must implement _reproject."""
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    assert hasattr(VGGTXCreator, "_reproject")


def test_mapanything_no_use_ba_field():
    """MapAnythingCreator must not have use_ba after refactor."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    field_names = {f.name for f in dataclasses.fields(MapAnythingCreator)}
    assert "use_ba" not in field_names


def test_mapanything_has_reproject():
    """MapAnythingCreator must implement _reproject."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    assert hasattr(MapAnythingCreator, "_reproject")


########################################################
########## make_creator tests #########################
########################################################


def test_make_creator_no_wrappers():
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    creator = make_creator("vggtx")
    assert isinstance(creator, VGGTXCreator)


def test_make_creator_with_lc():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    creator = make_creator("vggtx", use_lc=True)
    assert isinstance(creator, LoopClosure)
    assert isinstance(creator.base, VGGTXCreator)


def test_make_creator_unknown_name():
    from collab_splats.pointcloud import make_creator

    with pytest.raises(KeyError):
        make_creator("unknown_backend")


########################################################
########## LoopClosure delegation tests ###############
########################################################


def test_loop_closure_constructor_defaults():
    from collab_splats.geometry.loop_closure import LoopClosureConfig
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    assert lc.base is mock_base
    assert isinstance(lc.config, LoopClosureConfig)


def test_loop_closure_constructor_custom_config():
    from collab_splats.geometry.loop_closure import LoopClosureConfig
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    cfg = LoopClosureConfig(submap_size=10)
    mock_base = MagicMock()
    lc = LoopClosure(mock_base, config=cfg)
    assert lc.config.submap_size == 10


def test_loopclosure_holds_map_and_graph():
    from collab_splats.geometry.loop_closure.graph import PoseGraph
    from collab_splats.geometry.loop_closure.map import GraphMap
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    class _StubBase:
        default_verify_match_ratio = 0.85

    lc = LoopClosure(_StubBase())
    assert isinstance(lc.map, GraphMap)
    assert isinstance(lc.graph, PoseGraph)


def test_loop_closure_forwards_load_model():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.load_model()
    mock_base.load_model.assert_called_once()


def test_loop_closure_forwards_setup_inference():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.setup_inference(Path("/fake"))
    mock_base.setup_inference.assert_called_once_with(Path("/fake"))


def test_loop_closure_forwards_postprocess_and_build_colmap():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    mock_base = MagicMock()
    lc = LoopClosure(mock_base)
    lc.postprocess()
    mock_base.postprocess.assert_called_once()
    lc.build_colmap(Path("/out"))
    mock_base.build_colmap.assert_called_once_with(Path("/out"))


def test_loop_closure_run_inference_falls_back_to_base_when_too_few_frames():
    """When fewer frames than submap_size, falls back to base.run_inference()."""
    from collab_splats.geometry.loop_closure import LoopClosureConfig
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    mock_base = MagicMock()
    mock_base.views = torch.zeros(3, 3, 8, 8)  # 3 frames

    cfg = LoopClosureConfig(submap_size=20)
    lc = LoopClosure(mock_base, config=cfg)
    lc.run_inference()
    mock_base.run_inference.assert_called_once()


########################################################
########## LoopClosure.run() tests ####################
########################################################


def test_loop_closure_run_returns_feedforward_result():
    """lc.run(image_dir) returns FeedforwardResult from base.outputs."""
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    result = _make_ff_result()
    mock_base = _make_mock_creator(result)
    mock_base.views = torch.zeros(2, 3, 8, 8)

    lc = LoopClosure(mock_base)
    returned = lc.run(Path("/fake/dir"))

    assert returned is result
    mock_base.load_model.assert_called_once()
    mock_base.setup_inference.assert_called_once()
    mock_base.postprocess.assert_called_once()


def test_loop_closure_run_no_dedup_when_raw_outputs_empty():
    """When raw_outputs has no _dedup_rows key, result passes through unchanged."""
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    N, H, W = 2, 8, 8
    images = torch.zeros(N, 3, H, W)
    result = _make_ff_result(images=images)
    mock_base = _make_mock_creator(result)
    mock_base.raw_outputs = {}
    mock_base.views = torch.zeros(N, 3, H, W)

    lc = LoopClosure(mock_base)
    returned = lc.run(Path("/fake/dir"))

    # images tensor unchanged — same object
    assert returned.images is images


########################################################
########## LoopClosure.reproject() tests ##############
########################################################


def test_loop_closure_reproject_delegates_to_base():
    """lc.reproject(result) delegates to base.reproject(result)."""
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    result = _make_ff_result()
    reprojected = _make_ff_result(points=np.ones((5, 3), dtype=np.float32))
    mock_base = _make_mock_creator(result)
    mock_base.reproject.return_value = reprojected

    lc = LoopClosure(mock_base)
    out = lc.reproject(result)

    mock_base.reproject.assert_called_once_with(result)
    assert out is reprojected


########################################################
########## _trim_forward_outputs tests ###############
########################################################


def test_trim_forward_outputs_dict_trims_arrays():
    raw = {
        "extrinsic": np.zeros((7, 3, 4)),
        "intrinsics": np.zeros((7, 3, 3)),
        "depth": np.zeros((7, 64, 64, 1)),
        "depth_conf": np.zeros((7, 64, 64)),
        "world_points": np.zeros((7, 100, 3)),
        "world_points_conf": np.zeros((7, 100)),
        "images": np.zeros((7, 3, 64, 64)),
    }
    trimmed = _trim_forward_outputs(raw, 6)
    for key, val in trimmed.items():
        assert isinstance(val, np.ndarray)
        assert val.shape[0] == 6, f"{key}: expected 6, got {val.shape[0]}"


def test_trim_forward_outputs_list_trims_list():
    raw = [{"extrinsic": np.zeros((1, 3, 4))} for _ in range(7)]
    trimmed = _trim_forward_outputs(raw, 6)
    assert len(trimmed) == 6


def test_trim_forward_outputs_no_op_when_short():
    raw = {"extrinsic": np.zeros((5, 3, 4)), "scalar": 1.0}
    trimmed = _trim_forward_outputs(raw, 6)
    assert trimmed["extrinsic"].shape[0] == 5
    assert trimmed["scalar"] == 1.0


def test_trim_forward_outputs_preserves_non_array_values():
    raw = {"extrinsic": np.zeros((7, 3, 4)), "label": "keep", "count": 42}
    trimmed = _trim_forward_outputs(raw, 6)
    assert trimmed["label"] == "keep"
    assert trimmed["count"] == 42


########################################################
########## LoopClosure window extension test ##########
########################################################


def _make_raw(k: int) -> dict:
    # All frames use identity 3x4 extrinsic; first frame must be identity for assert_world_to_cam
    extrinsic = np.tile(np.eye(3, 4, dtype=np.float32), (k, 1, 1))
    # Varied positive confidence over the 4x4 grid so the per-submap 25th-percentile
    # conf mask keeps a non-empty dense cloud (else _assemble_result fails fast).
    rows = np.arange(4, dtype=np.float32)[:, None]
    cols = np.arange(4, dtype=np.float32)[None, :]
    depth_conf = np.tile(50.0 + rows + cols, (k, 1, 1)).astype(np.float32)
    return {
        "extrinsic": extrinsic,
        "intrinsics": np.eye(3, dtype=np.float32)[None].repeat(k, axis=0),
        "depth": np.ones((k, 4, 4, 1), dtype=np.float32),
        "depth_conf": depth_conf,
    }


def test_lc_loop_passes_k_plus_overlap_to_forward():
    """_run_lc_loop must pass submap_size+overlap_frames frames to _forward."""
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    submap_size = 3
    overlap = 1
    n_frames = 9

    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=overlap,
        min_submap_gap=0,
        max_loops_per_submap=0,
        lc_retrieval_threshold=999.0,
    )

    captured_sizes: list[int] = []

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        captured_sizes.append(sz)
        return _make_raw(sz)

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    base.views = torch.zeros(n_frames, 3, 4, 4)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._lc_retrieval = None

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = cfg
    wrapper.viz = None  # no live viewer: hooks are guarded no-ops

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", return_value=[]),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        wrapper._run_lc_loop()

    # Non-final full windows should be submap_size + overlap = 4
    non_final = captured_sizes[:-1]
    assert all(
        sz == submap_size + overlap for sz in non_final
    ), f"Expected windows of size {submap_size + overlap}, got {non_final}"


def test_lc_loop_populates_dense_points_and_map():
    """Window submaps carry dense points/colors/conf/conf_threshold and register in self.map."""
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    submap_size = 3
    overlap = 1
    n_frames = 9
    H = W = 4

    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=overlap,
        min_submap_gap=0,
        max_loops_per_submap=0,
        lc_retrieval_threshold=999.0,
    )

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw(sz)

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    # Frames in [0, 1] (0.5) so the *255 color path yields 127.
    base.views = torch.full((n_frames, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = cfg
    wrapper.viz = None  # no live viewer: hooks are guarded no-ops

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", return_value=[]),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        wrapper._run_lc_loop()

    # self.map holds the window submaps (reset at loop start, populated in add_points).
    assert len(wrapper.map.submaps) > 0
    sm = next(iter(wrapper.map.submaps.values()))
    k = sm.frames.shape[0]
    assert sm.points is not None and sm.points.shape == (k, H, W, 3)
    assert sm.colors is not None and sm.colors.dtype == np.uint8
    assert sm.colors.shape == (k, H, W, 3)
    assert bool((sm.colors == 127).all())  # 0.5 frames -> *255 -> 127
    assert sm.conf is not None and sm.conf.shape == (k, H, W)
    assert sm.conf_threshold is not None


def _rot_z(theta: float) -> np.ndarray:
    """3x3 rotation about z by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _make_raw_nontrivial(k: int, H: int, W: int) -> dict:
    """Non-degenerate stub _forward output: non-identity poses/K + varied positive depth.

    Frame 0 stays identity (assert_world_to_cam), frames 1..k-1 carry real rotation +
    translation so optimize() actually moves nodes; intrinsics_downsampled + high-conf
    depth make _raw_to_world_points return points, so the inter-submap scale path runs.
    """
    # Extrinsics (world2cam): frame 0 identity; others rotated + translated.
    extrinsic = np.zeros((k, 3, 4), dtype=np.float32)
    extrinsic[0] = np.eye(3, 4, dtype=np.float32)
    for i in range(1, k):
        extrinsic[i, :, :3] = _rot_z(0.15 * i)
        extrinsic[i, :, 3] = np.array([0.1 * i, -0.05 * i, 0.2 * i], dtype=np.float32)

    # Non-identity intrinsics: fx=fy=200, off-center principal point.
    K = np.array([[200.0, 0.0, W * 0.5 + 3.0], [0.0, 200.0, H * 0.5 - 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    intr = np.tile(K, (k, 1, 1))

    # Varied positive depth + high confidence (> conf_threshold) so scale path engages.
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    base_depth = 1.5 + 0.01 * (rows + cols)
    depth = np.stack([base_depth + 0.1 * i for i in range(k)], axis=0)[..., None].astype(np.float32)
    # Spatially varied confidence, all > 25 (graph scale gate keeps every point, so the
    # driven-graph == batch-shim parity is untouched) but with a real 25th percentile so
    # the per-submap conf mask keeps ~75% of pixels — a non-empty dense cloud.
    conf_grid = 50.0 + 0.5 * (rows + cols)
    depth_conf = np.tile(conf_grid, (k, 1, 1)).astype(np.float32)

    return {
        "extrinsic": extrinsic,
        "intrinsics": intr,
        "intrinsics_downsampled": intr,
        "depth": depth,
        "depth_conf": depth_conf,
    }


def test_self_graph_matches_incremental_drive():
    """self.graph (driven by _run_lc_loop) == a manual incremental drive at 1e-5.

    Strong parity anchor: non-identity poses/intrinsics + varied depth (so optimize()
    and the inter-submap scale path do real work) AND at least one driven loop edge.
    A per-submap-vs-batched cadence bug in the driving would diverge here.
    """
    from collab_splats.geometry.loop_closure.graph import PoseGraph
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    submap_size = 3
    overlap = 1
    n_frames = 9
    H = W = 32

    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=overlap,
        min_submap_gap=0,
        max_loops_per_submap=5,
    )

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    # Verified-loop payload: two non-identity poses + real per-pixel world points/conf.
    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        pose0 = np.eye(4, dtype=np.float32)
        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, :3] = _rot_z(0.1)
        pose1[:3, 3] = np.array([0.05, 0.02, 0.1], dtype=np.float32)
        wp = np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32)
        conf = np.full((2, H, W), 100.0, dtype=np.float32)
        return True, {"poses": np.stack([pose0, pose1]), "world_points": wp, "conf": conf}

    # Return a loop candidate against submap 0 once a prior submap exists; interior
    # (non-overlap) frame indices keep the resolved image paths unambiguous.
    def fake_find(submap, past, *args, **kwargs):
        if len(past) >= 1:
            return [
                LoopMatch(
                    similarity_score=0.1,
                    query_submap_id=submap.submap_id,
                    detected_submap_id=0,
                    query_frame_idx=1,
                    detected_frame_idx=1,
                )
            ]
        return []

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    base.views = torch.full((n_frames, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._verify_loop_candidate = fake_verify

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = cfg
    wrapper.map = None
    wrapper.graph = None
    wrapper.viz = None  # no live viewer: hooks are guarded no-ops

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        wrapper._run_lc_loop()

    # Capture the exact submaps + loop submaps the run drove self.graph with.
    submaps = wrapper._last_submaps
    lc_submaps = wrapper._last_lc_submaps
    N = base.views.shape[0]

    # The strengthened harness must actually exercise a loop edge, else it has no teeth.
    assert len(lc_submaps) >= 1, "expected >=1 driven loop edge"

    # Golden: same submaps driven through a fresh PoseGraph in the batch per-submap
    # cadence (add_submap+optimize per submap, then deferred loop edges, final solve).
    pg_golden = PoseGraph()
    for s in submaps:
        pg_golden.add_submap(
            s,
            wrapper.config.submap_overlap,
            wrapper.config.conf_threshold,
            wrapper.config.scale_method,
        )
        pg_golden.optimize()
    for lc in lc_submaps:
        pg_golden.add_loop_edge(lc, submaps, wrapper.config.conf_threshold, wrapper.config.scale_method)
    pg_golden.optimize()
    golden = pg_golden.extract_extrinsics(N)
    np.testing.assert_allclose(wrapper.graph.extract_extrinsics(N), golden, atol=1e-5)


def test_lc_output_assembled_from_graphmap():
    """LC output (base.outputs) is assembled from the GraphMap dense cloud, not the old merge.

    Uses the strengthened harness (non-identity geometry, dense depth, >=1 driven loop).
    Asserts points/colors == map.get_world_pointcloud, extrinsics == map.get_corrected_extrinsics,
    intrinsics (N,3,3), and len(image_paths) == N.
    """
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )
    from collab_splats.pointcloud.feedforward import FeedforwardResult

    submap_size = 3
    overlap = 1
    n_frames = 9
    H = W = 32

    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=overlap,
        min_submap_gap=0,
        max_loops_per_submap=5,
    )

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        pose0 = np.eye(4, dtype=np.float32)
        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, :3] = _rot_z(0.1)
        pose1[:3, 3] = np.array([0.05, 0.02, 0.1], dtype=np.float32)
        wp = np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32)
        conf = np.full((2, H, W), 100.0, dtype=np.float32)
        return True, {"poses": np.stack([pose0, pose1]), "world_points": wp, "conf": conf}

    def fake_find(submap, past, *args, **kwargs):
        if len(past) >= 1:
            return [
                LoopMatch(
                    similarity_score=0.1,
                    query_submap_id=submap.submap_id,
                    detected_submap_id=0,
                    query_frame_idx=1,
                    detected_frame_idx=1,
                )
            ]
        return []

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    base.views = torch.full((n_frames, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.zeros((n_frames, 6), dtype=np.float32)
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    # A loop edge must have actually driven the graph, else the harness has no teeth.
    assert len(lc._last_lc_submaps) >= 1

    out = base.outputs
    assert isinstance(out, FeedforwardResult)

    # points/colors are exactly the GraphMap dense cloud (float32/uint8, aligned, non-empty).
    # overlap-deduped, matching _assemble_result's call (leading overlap frames dropped).
    exp_pts, exp_cols = lc.map.get_world_pointcloud(lc.graph, overlap=lc.config.submap_overlap)
    assert out.points.shape[0] > 0
    assert out.points.dtype == np.float32 and out.points.shape[1] == 3
    assert out.colors.dtype == np.uint8 and out.colors.shape[1] == 3
    assert out.points.shape[0] == out.colors.shape[0]
    np.testing.assert_allclose(out.points, exp_pts)
    np.testing.assert_array_equal(out.colors, exp_cols)

    # extrinsics == the graph-corrected (N,4,4); intrinsics/image_paths sized to N.
    np.testing.assert_allclose(out.extrinsics, lc.map.get_corrected_extrinsics(lc.graph, n_frames))
    assert out.extrinsics.shape == (n_frames, 4, 4)
    assert out.intrinsics.shape == (n_frames, 3, 3)
    assert len(out.image_paths) == n_frames
    assert out.model_width == W and out.model_height == H

    # postprocess() is now a no-op — must not overwrite the GraphMap-sourced outputs.
    lc.postprocess()
    assert base.outputs is out
    base.postprocess.assert_not_called()


def _dense_submap_with_fx(sid, frame_start, fx_values, H=3, W=4):
    """Dense submap whose per-frame intrinsics encode a distinct fx marker per frame."""
    S = len(fx_values)
    rng = np.random.default_rng(sid)
    intr = np.tile(np.eye(3, dtype=np.float32), (S, 1, 1))
    for i, fx in enumerate(fx_values):
        intr[i, 0, 0] = fx
    poses = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
    for i in range(S):
        poses[i, :3, 3] = [0.0, 0.0, 0.1 * (frame_start + i)]
    return Submap(
        submap_id=sid,
        frames=None,
        poses=poses,
        intrinsics=intr,
        retrieval_vectors=np.zeros((S, 8), dtype=np.float32),
        image_paths=[f"s{sid}_{i}.jpg" for i in range(S)],
        points=rng.standard_normal((S, H, W, 3)).astype(np.float32),
        colors=(rng.random((S, H, W, 3)) * 255).astype(np.uint8),
        # Varied conf so the per-submap 25th-percentile mask keeps a non-empty cloud.
        conf=rng.random((S, H, W)).astype(np.float32),
        frame_start=frame_start,
    )


def test_assemble_result_intrinsics_first_occurrence_dedup():
    """_assemble_result dedups per-frame intrinsics by FIRST occurrence, not last-wins.

    Two overlapping submaps share global frame 2. s0 tags fx=100+global on frames
    [0,1,2]; s1 tags fx=200+global on frames [2,3,4]. The overlap frame 2 must keep
    s0's intrinsic (102), NOT s1's (202) — a last-wins dedup would land 202 here.
    """
    from collab_splats.geometry.loop_closure.graph import PoseGraph
    from collab_splats.geometry.loop_closure.map import GraphMap
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    n_frames = 5
    # s0 → frames [0,1,2] fx {100,101,102}; s1 → frames [2,3,4] fx {202,203,204}.
    s0 = _dense_submap_with_fx(0, frame_start=0, fx_values=[100.0, 101.0, 102.0])
    s1 = _dense_submap_with_fx(1, frame_start=2, fx_values=[202.0, 203.0, 204.0])

    pg = PoseGraph()
    for s in (s0, s1):
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    base.image_paths = [f"img_{i}.png" for i in range(n_frames)]
    base.original_coords = np.zeros((n_frames, 6), dtype=np.float32)

    lc = LoopClosure(base)
    lc.map = GraphMap()
    for s in (s0, s1):
        lc.map.add_submap(s)
    lc.graph = pg
    lc._last_submaps = [s0, s1]

    out = lc._assemble_result(n_frames)
    fx = out.intrinsics[:, 0, 0]

    # Boundary + interior spot-checks per submap.
    assert fx[0] == 100.0  # s0 boundary (first frame)
    assert fx[1] == 101.0  # s0 interior
    # Overlap frame → EARLIER submap's intrinsic (first-occurrence, not last-wins 202).
    assert fx[2] == 102.0
    assert fx[3] == 203.0  # s1 interior
    assert fx[4] == 204.0  # s1 boundary (last frame)
    # Model dims come from the dense point grid (H=3, W=4).
    assert out.model_height == 3 and out.model_width == 4


def test_assemble_result_caps_cloud_to_max_points():
    """_assemble_result caps the dense cloud to base.max_points before build_colmap.

    Regression for the 50 GB cgroup OOM: the LC path makes postprocess() a no-op, so
    without this cap the full dense per-pixel cloud (~50M pts for 500 frames) reaches
    build_colmap and materializes as pycolmap Point3D objects → SIGKILL. Cap is
    ATE-neutral (extrinsics untouched); here max_points=4 forces the cap on a tiny cloud.
    """
    from collab_splats.geometry.loop_closure.graph import PoseGraph
    from collab_splats.geometry.loop_closure.map import GraphMap
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    n_frames = 5
    s0 = _dense_submap_with_fx(0, frame_start=0, fx_values=[100.0, 101.0, 102.0])
    s1 = _dense_submap_with_fx(1, frame_start=2, fx_values=[202.0, 203.0, 204.0])

    pg = PoseGraph()
    for s in (s0, s1):
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()

    base = MagicMock()
    base.max_points = 4  # low budget → cap fires on the tiny dense cloud
    base.image_paths = [f"img_{i}.png" for i in range(n_frames)]
    base.original_coords = np.zeros((n_frames, 6), dtype=np.float32)

    lc = LoopClosure(base)
    lc.map = GraphMap()
    for s in (s0, s1):
        lc.map.add_submap(s)
    lc.graph = pg
    lc._last_submaps = [s0, s1]

    # The uncapped cloud must exceed the budget, else the test proves nothing.
    full_pts, _ = lc.map.get_world_pointcloud(lc.graph)
    assert full_pts.shape[0] > 4

    out = lc._assemble_result(n_frames)
    assert out.points.shape[0] == 4
    assert out.colors.shape[0] == 4  # colors stay index-aligned with points


def test_lc_frees_raw_outputs_after_unproject_keeps_frames():
    """Per-submap raw_outputs is freed once dense points are unprojected (VGGT-SLAM
    memory profile: raw_outputs — full depth+images+conf per frame — is the OOM
    driver). frames must stay resident for loop verify, and freeing raw_outputs
    must not change the assembled LC output.
    """
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    submap_size = 3
    overlap = 1
    n_frames = 9
    H = W = 32

    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=overlap,
        min_submap_gap=0,
        max_loops_per_submap=5,
    )

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        pose0 = np.eye(4, dtype=np.float32)
        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, :3] = _rot_z(0.1)
        pose1[:3, 3] = np.array([0.05, 0.02, 0.1], dtype=np.float32)
        wp = np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32)
        conf = np.full((2, H, W), 100.0, dtype=np.float32)
        return True, {"poses": np.stack([pose0, pose1]), "world_points": wp, "conf": conf}

    def fake_find(submap, past, *args, **kwargs):
        if len(past) >= 1:
            return [
                LoopMatch(
                    similarity_score=0.1,
                    query_submap_id=submap.submap_id,
                    detected_submap_id=0,
                    query_frame_idx=1,
                    detected_frame_idx=1,
                )
            ]
        return []

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    base.views = torch.full((n_frames, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.zeros((n_frames, 6), dtype=np.float32)
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    # A loop edge must have actually driven the graph, else the harness has no teeth.
    assert len(lc._last_lc_submaps) >= 1

    # raw_outputs freed on every submap (window + loop-closure); frames retained.
    for s in lc.map.ordered_submaps_by_key():
        assert s.raw_outputs is None, f"submap {s.submap_id} still retains raw_outputs"
        assert s.frames is not None, f"submap {s.submap_id} lost frames (needed for loop verify)"

    # Freeing raw_outputs must not break output assembly.
    assert lc.base.outputs.points.shape[0] > 0


def test_assemble_result_raises_on_empty_cloud():
    """_assemble_result fails fast (clear ValueError) when no submap carries dense points."""
    from collab_splats.geometry.loop_closure.graph import PoseGraph
    from collab_splats.geometry.loop_closure.map import GraphMap
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    n_frames = 4
    # Two non-LC submaps WITHOUT dense points (fully-degraded backend path).
    degraded = [
        Submap(
            submap_id=sid,
            frames=None,
            poses=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            retrieval_vectors=np.zeros((2, 8), dtype=np.float32),
            image_paths=[f"s{sid}_{i}.jpg" for i in range(2)],
            frame_start=sid * 2,
        )
        for sid in (0, 1)
    ]
    pg = PoseGraph()
    for s in degraded:
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    base.image_paths = [f"img_{i}.png" for i in range(n_frames)]
    base.original_coords = np.zeros((n_frames, 6), dtype=np.float32)
    base.model_width = 0
    base.model_height = 0

    lc = LoopClosure(base)
    lc.map = GraphMap()
    for s in degraded:
        lc.map.add_submap(s)
    lc.graph = pg

    with pytest.raises(ValueError, match="empty point cloud"):
        lc._assemble_result(n_frames)


########################################################
########## Live viewer hook tests #####################
########################################################


class _RecordingViz:
    """Duck-typed viewer stub that records add_points/add_frustum/add_lines calls."""

    def __init__(self):
        self.point_names = []
        self.frustum_names = []
        self.line_names = []

    def add_points(self, name, points, colors, **kwargs):
        self.point_names.append(name)

    def add_frustum(self, name, pose, intrinsic, **kwargs):
        self.frustum_names.append(name)

    def add_lines(self, name, segments, **kwargs):
        self.line_names.append(name)


def test_lc_loop_pushes_to_viz_when_set():
    """With a viewer attached, the LC loop pushes points/frusta per submap + a loop line."""
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    submap_size = 3
    overlap = 1
    n_frames = 9
    H = W = 32

    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=overlap,
        min_submap_gap=0,
        max_loops_per_submap=5,
    )

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        pose0 = np.eye(4, dtype=np.float32)
        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, :3] = _rot_z(0.1)
        pose1[:3, 3] = np.array([0.05, 0.02, 0.1], dtype=np.float32)
        wp = np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32)
        conf = np.full((2, H, W), 100.0, dtype=np.float32)
        return True, {"poses": np.stack([pose0, pose1]), "world_points": wp, "conf": conf}

    def fake_find(submap, past, *args, **kwargs):
        if len(past) >= 1:
            return [
                LoopMatch(
                    similarity_score=0.1,
                    query_submap_id=submap.submap_id,
                    detected_submap_id=0,
                    query_frame_idx=1,
                    detected_frame_idx=1,
                )
            ]
        return []

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    base.views = torch.full((n_frames, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.zeros((n_frames, 6), dtype=np.float32)
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)
    viz = _RecordingViz()
    lc.viz = viz

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    # At least one accepted loop drove the graph (else the harness has no teeth).
    assert len(lc._last_lc_submaps) >= 1

    # add_points fired at least once per non-LC submap; frusta + >=1 loop line drawn.
    n_submaps = len([s for s in lc.map.ordered_submaps_by_key() if not s.is_lc_submap])
    assert n_submaps > 0
    assert len(viz.point_names) >= n_submaps
    assert len(viz.frustum_names) > 0
    assert len(viz.line_names) >= 1


def _run_lc_harness_with_timing(loop_edge_timing):
    """Drive the stubbed LC loop (>=1 loop) under a given loop_edge_timing; return (result, n_loops)."""
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    submap_size = 3
    overlap = 1
    n_frames = 9
    H = W = 32

    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=overlap,
        min_submap_gap=0,
        max_loops_per_submap=5,
        loop_edge_timing=loop_edge_timing,
    )

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        pose0 = np.eye(4, dtype=np.float32)
        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, :3] = _rot_z(0.1)
        pose1[:3, 3] = np.array([0.05, 0.02, 0.1], dtype=np.float32)
        wp = np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32)
        conf = np.full((2, H, W), 100.0, dtype=np.float32)
        return True, {"poses": np.stack([pose0, pose1]), "world_points": wp, "conf": conf}

    def fake_find(submap, past, *args, **kwargs):
        if len(past) >= 1:
            return [
                LoopMatch(
                    similarity_score=0.1,
                    query_submap_id=submap.submap_id,
                    detected_submap_id=0,
                    query_frame_idx=1,
                    detected_frame_idx=1,
                )
            ]
        return []

    base = MagicMock()
    base.max_points = 500_000  # real int so _assemble_result's subsample cap runs (no-op on tiny test clouds)
    base.views = torch.full((n_frames, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.zeros((n_frames, 6), dtype=np.float32)
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    return base.outputs, len(lc._last_lc_submaps), n_frames


def test_loop_edge_timing_default_is_deferred():
    """LoopClosureConfig defaults loop_edge_timing to 'deferred' (repo-validated behavior)."""
    from collab_splats.geometry.loop_closure.wrapper import LoopClosureConfig

    assert LoopClosureConfig().loop_edge_timing == "deferred"


def test_loop_edge_timing_deferred_vs_live_both_valid():
    """deferred and live both complete, drive >=1 loop, and yield internally-valid finite results.

    This is the A/B knob under test: the two modes MAY produce different extrinsics
    (live inserts each loop edge during the window loop, deferred batches them after),
    so we assert each is independently valid + finite, NOT that they match.
    """
    result_def, n_loops_def, n_frames = _run_lc_harness_with_timing("deferred")
    result_live, n_loops_live, _ = _run_lc_harness_with_timing("live")

    for result, n_loops in ((result_def, n_loops_def), (result_live, n_loops_live)):
        assert isinstance(result, FeedforwardResult)
        # >=1 loop edge actually drove the graph.
        assert n_loops >= 1
        # Non-empty dense cloud + finite geometry.
        assert result.points.shape[0] > 0
        assert np.isfinite(result.points).all()
        # Extrinsics are (N, 4, 4) and finite.
        assert result.extrinsics.shape == (n_frames, 4, 4)
        assert np.isfinite(result.extrinsics).all()
