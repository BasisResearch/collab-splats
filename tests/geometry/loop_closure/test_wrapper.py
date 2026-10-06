# tests/pointcloud/test_wrappers.py
import dataclasses
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.map import GraphMap
from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.loop_closure.wrapper import (
    LoopClosure,
    LoopClosureConfig,
)
from collab_splats.geometry.transforms import transform_points
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.utils import subsample_points
from collab_splats.preproc.frames import frame_paths
from tests.geometry.loop_closure._helpers import record_driven_submaps
from tests.pointcloud.conftest import _frame_files, _frames


def _make_ff_result(**overrides):
    defaults = dict(
        points=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=None,
        model_intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        image_paths=[Path("a.jpg"), Path("b.jpg")],
        original_coords=np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (2, 1)),  # full-frame box
        model_width=224,
        model_height=224,
    )
    defaults.update(overrides)
    return PointcloudResult(**defaults)


def _make_mock_creator(ff_result):
    """Returns a mock that duck-types as BaseFeedforwardCreator."""
    m = MagicMock()
    m.raw_outputs = {}
    return m


########################################################
########## PointcloudResult field tests ##############
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
    creator.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (2, 1))  # full-frame box

    N, H, W = 2, 8, 8
    raw_outputs = {
        "depth": np.ones((N, H, W, 1), dtype=np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": torch.zeros(N, 3, H, W),
        "extrinsic": np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
    }

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


def test_mapanything_no_use_ba_field():
    """MapAnythingCreator must not have use_ba after refactor."""
    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    field_names = {f.name for f in dataclasses.fields(MapAnythingCreator)}
    assert "use_ba" not in field_names


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


def test_loop_closure_run_inference_falls_back_to_base_when_too_few_frames():
    """When fewer frames than submap_size, one whole-scene forward sets base.raw_outputs."""
    from collab_splats.geometry.loop_closure import LoopClosureConfig
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    mock_base = MagicMock()
    mock_base.views = torch.zeros(3, 3, 8, 8)  # 3 frames

    cfg = LoopClosureConfig(submap_size=20)
    lc = LoopClosure(mock_base, config=cfg)
    lc.run_inference()
    mock_base._forward.assert_called_once_with(mock_base.model, mock_base.views)
    assert mock_base.raw_outputs is mock_base._forward.return_value
    mock_base.run_inference.assert_not_called()


########################################################
########## LoopClosure.create_pointcloud() tests #######
########################################################


def test_loop_closure_create_pointcloud_returns_the_base_postprocess(tmp_path):
    """Too few frames for LC: create_pointcloud returns the base's own _postprocess result."""
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    result = _make_ff_result()
    mock_base = _make_mock_creator(result)
    mock_base.views = torch.zeros(2, 3, 8, 8)
    mock_base._postprocess.return_value = result
    mock_base.clean = False
    mock_base.max_points = len(result.points)

    # The delegated frame listing reads the directory, as a creator without a handoff does
    mock_base._list_frames.side_effect = frame_paths
    paths = _frame_files(_frames([(8, 8), (8, 8)]), tmp_path / "images")

    lc = LoopClosure(mock_base)
    returned = lc.create_pointcloud(tmp_path / "images", tmp_path / "out")

    np.testing.assert_array_equal(returned.points, result.points)
    mock_base.load_model.assert_called_once()
    mock_base.setup_inference.assert_called_once_with(paths)
    mock_base._postprocess.assert_called_once_with(mock_base.raw_outputs)


########################################################
########## LoopClosure window extension test ##########
########################################################


def _make_raw(k: int) -> dict:
    # All frames use identity 3x4 extrinsic; the wrapper requires frame 0 at identity
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
    # Full-frame box per frame
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (len(base.image_paths), 1))
    base._forward = fake_forward
    base._lc_retrieval = None

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = cfg
    wrapper.viz = None  # no live viewer: hooks are guarded no-ops
    wrapper.ba = None

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


def test_lc_loop_rejects_a_window_not_normalized_to_frame_0():
    """A window whose frame-0 pose is not identity raises before any submap is built."""
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    def fake_forward(model, views, **kwargs):
        raw = _make_raw(views.shape[0])
        raw["extrinsic"][0, :3, 3] = [1.0, 2.0, 3.0]
        return raw

    base = MagicMock()
    base.views = torch.zeros(4, 3, 4, 4)
    base.image_paths = [f"img_{i:03d}.png" for i in range(4)]
    # Full-frame box per frame
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (len(base.image_paths), 1))
    base._forward = fake_forward

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = LoopClosureConfig(submap_size=3, submap_overlap=1)
    wrapper.viz = None
    wrapper.ba = None

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        pytest.raises(ValueError, match="not normalized to frame 0"),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        wrapper._run_lc_loop()


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
    # Full-frame box per frame
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (len(base.image_paths), 1))
    base._forward = fake_forward

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = cfg
    wrapper.viz = None  # no live viewer: hooks are guarded no-ops
    wrapper.ba = None

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", return_value=[]),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        wrapper._run_lc_loop()

    # self.map holds the window submaps (reset at loop start, populated in _add_window_submap).
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

    Frame 0 stays identity (the wrapper's frame-0 check), frames 1..k-1 carry real rotation +
    translation so optimize() actually moves nodes; depth + depth_conf give every window
    dense points, so the inter-submap scale path runs.
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

    # Varied positive depth + confidence so the scale path engages.
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    base_depth = 1.5 + 0.01 * (rows + cols)
    depth = np.stack([base_depth + 0.1 * i for i in range(k)], axis=0)[..., None].astype(np.float32)
    # Spatially varied confidence with a real 25th percentile
    # - the per-submap percentile gate keeps ~75% of pixels, for scale and the dense cloud
    conf_grid = 50.0 + 0.5 * (rows + cols)
    depth_conf = np.tile(conf_grid, (k, 1, 1)).astype(np.float32)

    return {
        "extrinsic": extrinsic,
        "intrinsics": intr,
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
    # Full-frame box per frame
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (len(base.image_paths), 1))
    base._forward = fake_forward
    base._verify_loop_candidate = fake_verify

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = cfg
    wrapper.map = None
    wrapper.graph = None
    wrapper.viz = None  # no live viewer: hooks are guarded no-ops
    wrapper.ba = None

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
        record_driven_submaps() as driven,
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        wrapper._run_lc_loop()

    # Capture the exact submaps + loop submaps the run drove self.graph with.
    submaps = driven["submaps"]
    lc_submaps = driven["lc_submaps"]
    N = base.views.shape[0]

    # The strengthened harness must actually exercise a loop edge, else it has no teeth.
    assert len(lc_submaps) >= 1, "expected >=1 driven loop edge"

    # Golden: same submaps driven through a fresh PoseGraph in the batch per-submap
    # cadence (add_submap+optimize per submap, then deferred loop edges, final solve).
    pg_golden = PoseGraph()
    for s in submaps:
        pg_golden.add_submap(s, wrapper.config.submap_overlap)
        pg_golden.optimize()
    for lc in lc_submaps:
        pg_golden.add_loop_edge(lc, submaps)
    pg_golden.optimize()
    golden = pg_golden.extract_extrinsics(N)
    np.testing.assert_allclose(wrapper.graph.extract_extrinsics(N), golden, atol=1e-5)


def test_conf_percentile_reaches_window_and_loop_submaps():
    """
    LoopClosureConfig.conf_percentile sets every driven submap's percentile threshold.

    - window submaps: percentile(depth_conf, 60) + 1e-6
    - loop carriers: dense (2, H, W, 3) points and (2, H, W) conf, same percentile
    """
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    H = W = 16
    cfg = LoopClosureConfig(submap_size=3, submap_overlap=1, min_submap_gap=0, conf_percentile=60.0)
    lc_conf = np.random.RandomState(1).rand(2, H, W).astype(np.float32)

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, 3] = [0.05, 0.02, 0.1]
        wp = np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32)
        return True, {"poses": np.stack([np.eye(4, dtype=np.float32), pose1]), "world_points": wp, "conf": lc_conf}

    def fake_find(submap, past, *args, **kwargs):
        if not past:
            return []
        return [LoopMatch(0.1, submap.submap_id, 0, query_frame_idx=1, detected_frame_idx=1)]

    base = MagicMock()
    base.max_points = 500_000
    base.views = torch.full((9, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(9)]
    # Full-frame box per frame
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (len(base.image_paths), 1))
    base._forward = fake_forward
    base._verify_loop_candidate = fake_verify

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = cfg
    wrapper.map = None
    wrapper.graph = None
    wrapper.viz = None
    wrapper.ba = None

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
        record_driven_submaps() as driven,
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        wrapper._run_lc_loop()

    assert driven["lc_submaps"], "expected >=1 driven loop edge"
    for sm in driven["submaps"]:
        assert sm.conf_threshold == pytest.approx(float(np.percentile(sm.conf, 60.0)) + 1e-6)
    for lc in driven["lc_submaps"]:
        assert lc.points.shape == (2, H, W, 3)
        assert lc.conf.shape == (2, H, W)
        assert lc.conf_threshold == pytest.approx(float(np.percentile(lc_conf, 60.0)) + 1e-6)


def test_lc_output_assembled_from_graphmap():
    """LC output (lc.outputs) is assembled from the graph-corrected submap reads, not the old merge.

    Uses the strengthened harness (non-identity geometry, dense depth, >=1 driven loop).
    Asserts points/colors == the submaps' masked world reads, extrinsics == graph.extract_extrinsics,
    intrinsics (N,3,3), and len(image_paths) == N.
    """
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )
    from collab_splats.pointcloud.base import PointcloudResult

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
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))  # full-frame box
    base._forward = fake_forward
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
        record_driven_submaps() as driven,
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    # A loop edge must have actually driven the graph, else the harness has no teeth.
    assert len(driven["lc_submaps"]) >= 1

    out = lc.outputs
    assert isinstance(out, PointcloudResult)

    # points/colors are each window submap's masked world read, a non-first submap's overlap frames dropped
    subs = [s for s in lc.map.ordered_submaps_by_key() if not s.is_lc_submap and s.points is not None]
    skips = [lc.config.submap_overlap if s.frame_start > 0 else 0 for s in subs]
    exp_pts = np.vstack([s.get_points_in_world_frame(lc.graph, skip_first=k) for s, k in zip(subs, skips)])
    exp_cols = np.vstack([s.get_points_colors(skip_first=k) for s, k in zip(subs, skips)])
    assert out.points.shape[0] > 0
    assert out.points.dtype == np.float32 and out.points.shape[1] == 3
    assert out.colors.dtype == np.uint8 and out.colors.shape[1] == 3
    assert out.points.shape[0] == out.colors.shape[0]
    np.testing.assert_allclose(out.points, exp_pts, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out.colors, exp_cols)

    # extrinsics == the graph-corrected (N,4,4); intrinsics/image_paths sized to N.
    np.testing.assert_allclose(out.extrinsics, lc.graph.extract_extrinsics(n_frames))
    assert out.extrinsics.shape == (n_frames, 4, 4)
    assert out.intrinsics.shape == (n_frames, 3, 3)
    assert len(out.image_paths) == n_frames
    assert out.model_width == W and out.model_height == H

    # _reconstruct returns the GraphMap-sourced outputs as is, never the base _postprocess
    with patch.object(lc, "run_inference"):
        assert lc._reconstruct([], Path("/unused")) is out
    base._postprocess.assert_not_called()


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
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))  # full-frame box

    lc = LoopClosure(base)
    lc.map = GraphMap()
    for s in (s0, s1):
        lc.map.add_submap(s)
    lc.graph = pg

    out = lc._assemble_result(n_frames)
    fx = out.model_intrinsics[:, 0, 0]

    # Boundary + interior spot-checks per submap.
    assert fx[0] == 100.0  # s0 boundary (first frame)
    assert fx[1] == 101.0  # s0 interior
    # Overlap frame → EARLIER submap's intrinsic (first-occurrence, not last-wins 202).
    assert fx[2] == 102.0
    assert fx[3] == 203.0  # s1 interior
    assert fx[4] == 204.0  # s1 boundary (last frame)
    # Model dims come from the dense point grid (H=3, W=4).
    assert out.model_height == 3 and out.model_width == 4


def test_assemble_result_carries_depth_and_confidence_without_world_points(tmp_path):
    """
    LC result carries per-frame depth and confidence consistent with its poses, and no world points.

    - depth is the z of the corrected world grid under the corrected extrinsics
    - overlap frame 2 comes from s0, the first submap to cover it
    - depth survives a zarr round trip
    """
    n_frames = 5
    s0 = _dense_submap_with_fx(0, frame_start=0, fx_values=[100.0, 101.0, 102.0])
    s1 = _dense_submap_with_fx(1, frame_start=2, fx_values=[202.0, 203.0, 204.0])

    # Two-submap graph, one optimize per add as the wrapper does
    pg = PoseGraph()
    for s in (s0, s1):
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()

    base = MagicMock()
    base.max_points = 500_000
    base.image_paths = [f"img_{i}.png" for i in range(n_frames)]
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))

    lc = LoopClosure(base)
    lc.map = GraphMap()
    for s in (s0, s1):
        lc.map.add_submap(s)
    lc.graph = pg
    out = lc._assemble_result(n_frames)

    # Per-pixel fields cover every frame on the model grid
    assert out.depth.shape == (n_frames, 3, 4)
    assert tuple(out.confidence.shape) == (n_frames, 3, 4)
    assert out.world_points is None

    # Depth is camera-frame z of the corrected world grid; overlap frame 2 takes s0's grid
    grid0, grid1 = s0.get_world_grid(pg), s1.get_world_grid(pg)

    for g, frame in enumerate([grid0[0], grid0[1], grid0[2], grid1[1], grid1[2]]):
        cam = transform_points(frame.astype(np.float64), out.extrinsics[g].astype(np.float64))
        np.testing.assert_allclose(out.depth[g], cam[..., 2], rtol=1e-5, atol=1e-5)

    # Overlap frame takes s0's confidence, not s1's
    np.testing.assert_array_equal(np.asarray(out.confidence[2]), s0.conf[2])

    # Depth round-trips through zarr
    out.save_zarr(tmp_path / "pc.zarr")
    loaded = PointcloudResult.load_zarr(tmp_path / "pc.zarr")
    np.testing.assert_allclose(loaded.depth, out.depth)


def test_assemble_result_caps_cloud_to_max_points():
    """
    _assemble_result caps the dense cloud to base.max_points before the COLMAP export.

    - regression for the 50 GB cgroup OOM: an uncapped dense cloud (~50M pts for 500 frames)
      becomes pycolmap Point3D objects in the export → SIGKILL
    - ATE-neutral (extrinsics untouched); max_points=4 forces the cap on a tiny cloud
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
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))  # full-frame box

    lc = LoopClosure(base)
    lc.map = GraphMap()
    for s in (s0, s1):
        lc.map.add_submap(s)
    lc.graph = pg

    # The uncapped cloud must exceed the budget, else the test proves nothing.
    full_pts = np.vstack([s0.get_points_in_world_frame(lc.graph), s1.get_points_in_world_frame(lc.graph, 1)])
    full_cols = np.vstack([s0.get_points_colors(), s1.get_points_colors(1)])
    assert full_pts.shape[0] > 4

    out = lc._assemble_result(n_frames)
    assert out.points.shape[0] == 4
    assert out.colors.shape[0] == 4  # colors stay index-aligned with points

    # The kept rows are subsample_points' default-seed draw over the whole overlap-deduped cloud
    all_kept = np.ones(full_pts.shape[0], dtype=bool)
    keep = subsample_points(all_kept, 4)
    np.testing.assert_array_equal(out.points, full_pts[keep])
    np.testing.assert_array_equal(out.colors, full_cols[keep])


def _assemble_two_submaps_plus(extra, cap):
    """
    _assemble_result over s0, s1 and an optional extra submap; graph built from s0, s1 only.
    """
    n_frames = 5
    s0 = _dense_submap_with_fx(0, frame_start=0, fx_values=[100.0, 101.0, 102.0])
    s1 = _dense_submap_with_fx(1, frame_start=2, fx_values=[202.0, 203.0, 204.0])

    # Two-submap graph, one optimize per add as the wrapper does
    pg = PoseGraph()

    for s in (s0, s1):
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()

    base = MagicMock()
    base.max_points = cap
    base.image_paths = [f"img_{i}.png" for i in range(n_frames)]
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))
    lc = LoopClosure(base)
    lc.map = GraphMap()

    for s in (s0, s1) if extra is None else (s0, s1, extra):
        lc.map.add_submap(s)

    lc.graph = pg
    return lc._assemble_result(n_frames)


@pytest.mark.parametrize("cap", [500_000, 7])
def test_assemble_result_adds_no_points_from_loop_carriers(cap):
    """
    A dense loop-carrier submap leaves the assembled cloud and its capped draw unchanged.

    - the carrier covers graph frames 0-1, so lifting it would succeed and add points
    """
    carrier = _dense_submap_with_fx(2, frame_start=0, fx_values=[300.0, 301.0])
    carrier.is_lc_submap = True

    without = _assemble_two_submaps_plus(None, cap)
    with_carrier = _assemble_two_submaps_plus(carrier, cap)

    assert with_carrier.points.shape == without.points.shape
    np.testing.assert_array_equal(with_carrier.points, without.points)
    np.testing.assert_array_equal(with_carrier.colors, without.colors)


@pytest.mark.parametrize("cap", [500_000, 7])
def test_assemble_result_adds_no_points_from_pointless_submaps(cap):
    """
    A submap without dense points (MapAnything degraded path) leaves the cloud unchanged.
    """
    degraded = _dense_submap_with_fx(2, frame_start=3, fx_values=[300.0, 301.0])
    degraded.points = degraded.colors = degraded.conf = None

    without = _assemble_two_submaps_plus(None, cap)
    with_degraded = _assemble_two_submaps_plus(degraded, cap)

    assert with_degraded.points.shape == without.points.shape
    np.testing.assert_array_equal(with_degraded.points, without.points)
    np.testing.assert_array_equal(with_degraded.colors, without.colors)


def test_assemble_result_raises_on_dense_submap_without_conf():
    """
    A window submap with dense points but no conf raises instead of failing on a None index.
    """
    broken = _dense_submap_with_fx(2, frame_start=3, fx_values=[300.0, 301.0])
    broken.conf = None

    with pytest.raises(ValueError, match="Submap 2 has no conf"):
        _assemble_two_submaps_plus(broken, 500_000)


def test_lc_submaps_keep_frames_after_unproject():
    """frames stay resident on every submap after dense unprojection (loop verify reads them)."""
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
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))  # full-frame box
    base._forward = fake_forward
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
        record_driven_submaps() as driven,
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    # A loop edge must have actually driven the graph, else the harness has no teeth.
    assert len(driven["lc_submaps"]) >= 1

    # frames retained on every submap (window + loop-closure)
    for s in lc.map.ordered_submaps_by_key():
        assert s.frames is not None, f"submap {s.submap_id} lost frames (needed for loop verify)"

    # Output assembly still runs
    assert lc.outputs.points.shape[0] > 0


def _run_one_window(raw: dict, window, extractor=None):
    """
    Drive LoopClosure.run_predictions over one window's forward outputs.

    - extractor: retrieval stub; None returns zero descriptors
    """
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    base = MagicMock()
    base.image_paths = [f"img_{i}.png" for i in range(3)]
    # Full-frame box per frame
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (len(base.image_paths), 1))
    lc = LoopClosure(base)
    extractor = extractor or (lambda frames: torch.zeros(frames.shape[0], 8))
    return lc.run_predictions(window, raw, 0, 0, [], [], extractor, MagicMock())


def test_run_predictions_requires_intrinsics_key():
    raw = _make_raw_nontrivial(3, 8, 8)
    raw["intrinsic"] = raw.pop("intrinsics")
    with pytest.raises(KeyError, match="intrinsics"):
        _run_one_window(raw, torch.zeros(3, 3, 8, 8))


def test_run_predictions_rejects_unknown_window_shape():
    with pytest.raises(TypeError, match="LC window"):
        _run_one_window(_make_raw_nontrivial(3, 8, 8), [0, 1, 2])


def _recording_extractor(seen: list):
    """
    Retrieval stub that records each input batch and returns zero descriptors.
    """

    def extract(frames: torch.Tensor) -> torch.Tensor:
        seen.append(frames)
        return torch.zeros(frames.shape[0], 8)

    return extract


def test_run_predictions_retrieves_on_a_tensor_window_as_given():
    """
    A tensor window reaches the retrieval extractor unchanged.
    """
    seen = []
    window = torch.rand(3, 3, 8, 8)
    _run_one_window(_make_raw_nontrivial(3, 8, 8), window, _recording_extractor(seen))

    assert len(seen) == 1
    assert torch.equal(seen[0], window)


def test_run_predictions_retrieves_on_raw_images_for_a_view_dict_window():
    """
    A view-dict window retrieves on the forward's raw [0, 1] images, not the normalized views.
    """
    # dinov2-normalized views fall outside [0, 1]; the forward's raw images are the RGB
    seen = []
    images = np.random.default_rng(0).random((3, 3, 8, 8), dtype=np.float32)
    window = [{"img": torch.from_numpy((images[i : i + 1] - 0.45) / 0.225)} for i in range(3)]
    raw = _make_raw_nontrivial(3, 8, 8)
    raw["images"] = images
    _run_one_window(raw, window, _recording_extractor(seen))

    assert len(seen) == 1
    assert torch.equal(seen[0], torch.from_numpy(images))
    assert seen[0].min() >= 0.0 and seen[0].max() <= 1.0


def test_assemble_result_raises_on_empty_cloud():
    """
    Pins the empty-cloud raise when no submap carries dense points.

    - the `model_height is None` arm has no independent trigger: no dense points is an empty cloud
    """
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
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))  # full-frame box

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
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))  # full-frame box
    base._forward = fake_forward
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)
    viz = _RecordingViz()
    lc.viz = viz

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
        record_driven_submaps() as driven,
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    # At least one accepted loop drove the graph (else the harness has no teeth).
    assert len(driven["lc_submaps"]) >= 1

    # add_points fired at least once per non-LC submap; frusta + >=1 loop line drawn.
    n_submaps = len([s for s in lc.map.ordered_submaps_by_key() if not s.is_lc_submap])
    assert n_submaps > 0
    assert len(viz.point_names) >= n_submaps
    assert len(viz.frustum_names) > 0
    assert len(viz.line_names) >= 1


def _make_real_submap_for_viz() -> Submap:
    """Two-frame dense submap whose confidence mask keeps about 75% of its 32x32 pixels."""
    H = W = 32
    sm = Submap(
        submap_id=0,
        frames=torch.zeros(2, 3, H, W),
        poses=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        retrieval_vectors=np.zeros((2, 8), dtype=np.float32),
        image_paths=["a.jpg", "b.jpg"],
        frame_start=0,
    )
    sm.set_dense_points(
        np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32),
        np.zeros((2, H, W, 3), np.uint8),
        # varied conf: a constant one puts every point at the 25th-percentile cutoff, masking all
        np.random.RandomState(1).uniform(1.0, 100.0, (2, H, W)).astype(np.float32),
    )
    return sm


def test_viz_max_points_caps_the_viewer_push():
    """LoopClosureConfig.viz_max_points is the per-submap viewer point cap."""
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    sm = _make_real_submap_for_viz()

    class _SizedViz(_RecordingViz):
        def add_points(self, name, points, colors, **kwargs):
            self.n_points = points.shape[0]

    lc = LoopClosure(MagicMock(), config=LoopClosureConfig(viz_max_points=100))
    lc.graph.add_submap(sm, 1)
    lc.graph.optimize()
    lc.viz = _SizedViz()
    lc._viz_push_submap(sm)
    assert lc.viz.n_points == 100


class _FailingViz(_RecordingViz):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def add_points(self, name, points, colors, **kwargs):
        raise self.exc

    def add_lines(self, name, segments, **kwargs):
        raise self.exc


def _accepted_match():
    from collab_splats.geometry.loop_closure.matching import LoopMatch

    m = LoopMatch(
        similarity_score=0.1, query_submap_id=0, detected_submap_id=0, query_frame_idx=0, detected_frame_idx=0
    )
    m.accepted = True
    return m


def _lc_with_one_submap():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    sm = _make_real_submap_for_viz()
    lc = LoopClosure(MagicMock())
    lc.graph.add_submap(sm, 1)
    lc.graph.optimize()
    return lc, sm


def test_viz_draw_loops_swallows_viewer_errors():
    lc, sm = _lc_with_one_submap()
    lc.viz = _FailingViz(OSError("socket closed"))
    lc._viz_draw_loops([_accepted_match()], [sm])  # logged, not raised


def test_viz_draw_loops_propagates_programming_errors():
    lc, sm = _lc_with_one_submap()
    lc.viz = _FailingViz(AttributeError("no such method"))
    with pytest.raises(AttributeError):
        lc._viz_draw_loops([_accepted_match()], [sm])


def test_viz_draw_loops_skips_rejected_matches():
    """A loop match that failed verification draws no line."""
    lc, sm = _lc_with_one_submap()
    lc.viz = _RecordingViz()
    rejected = _accepted_match()
    rejected.accepted = False
    lc._viz_draw_loops([rejected], [sm])
    assert lc.viz.line_names == []


def test_viz_push_submap_logs_viewer_io_errors(caplog):
    lc, sm = _lc_with_one_submap()
    lc.viz = _FailingViz(OSError("socket closed"))
    with caplog.at_level("WARNING", logger="collab_splats.geometry.loop_closure.wrapper"):
        lc._viz_push_submap(sm)  # logged, not raised
    assert "viewer submap push failed" in caplog.text
    assert "socket closed" in caplog.text


def test_viz_push_submap_propagates_programming_errors():
    lc, sm = _lc_with_one_submap()
    lc.viz = _FailingViz(TypeError("bad argument"))
    with pytest.raises(TypeError, match="bad argument"):
        lc._viz_push_submap(sm)


def test_dino_salad_load_failure_falls_back_to_full_inference():
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    base = MagicMock()
    base.views = torch.zeros(6, 3, 8, 8)
    lc = LoopClosure(base, config=LoopClosureConfig(submap_size=3))
    with patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as r:
        r.get.side_effect = OSError("weights missing")
        lc.run_inference()
    base._forward.assert_called_once()


def test_dino_salad_unexpected_error_propagates():
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    base = MagicMock()
    base.views = torch.zeros(6, 3, 8, 8)
    lc = LoopClosure(base, config=LoopClosureConfig(submap_size=3))
    with patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as r:
        r.get.side_effect = TypeError("bad kwarg")
        with pytest.raises(TypeError, match="bad kwarg"):
            lc.run_inference()


def test_lc_rejects_non_finite_loop_pose():
    """A NaN loop relative is rejected before it reaches the graph."""
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import (
        LoopClosure,
        LoopClosureConfig,
    )

    H = W = 32
    n_frames = 9
    cfg = LoopClosureConfig(submap_size=3, submap_overlap=1, min_submap_gap=0)

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        poses[1, 0, 3] = np.nan
        wp = np.zeros((2, H, W, 3), np.float32)
        return True, {"poses": poses, "world_points": wp, "conf": np.full((2, H, W), 100.0, np.float32)}

    def fake_find(submap, past, *args, **kwargs):
        if past:
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
    base.max_points = 500_000
    base.views = torch.full((n_frames, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))  # full-frame box
    base._forward = fake_forward
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)
    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
        record_driven_submaps() as driven,
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    assert driven["lc_submaps"] == []
    assert np.isfinite(lc.outputs.extrinsics).all()


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
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))  # full-frame box
    base._forward = fake_forward
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
        record_driven_submaps() as driven,
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    return lc.outputs, len(driven["lc_submaps"]), n_frames


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
        assert isinstance(result, PointcloudResult)
        # >=1 loop edge actually drove the graph.
        assert n_loops >= 1
        # Non-empty dense cloud + finite geometry.
        assert result.points.shape[0] > 0
        assert np.isfinite(result.points).all()
        # Extrinsics are (N, 4, 4) and finite.
        assert result.extrinsics.shape == (n_frames, 4, 4)
        assert np.isfinite(result.extrinsics).all()


def _lc_wrapper(n_frames: int = 9) -> LoopClosure:
    """Bare LoopClosure over n_frames tensor views with a stub forward and no viewer."""
    cfg = LoopClosureConfig(submap_size=3, submap_overlap=1, lc_retrieval_threshold=0.0, verify_match_ratio=0.5)

    base = MagicMock()
    base.max_points = 500_000
    base.views = torch.zeros(n_frames, 3, 4, 4)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))
    base._forward = lambda model, views, **kwargs: _make_raw(views.shape[0])

    wrapper = LoopClosure.__new__(LoopClosure)
    wrapper.base = base
    wrapper.config = cfg
    wrapper.viz = None
    wrapper.ba = None
    return wrapper


def test_lc_loop_skips_retrieval_at_zero_threshold():
    wrapper = _lc_wrapper()

    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures") as mock_find,
    ):
        wrapper._run_lc_loop()

    assert wrapper.base.n_loops_applied == 0
    mock_retrieval.get.assert_not_called()
    mock_find.assert_not_called()
    assert wrapper.outputs is not None


def test_lc_loop_launches_next_forward_before_recording_the_window():
    wrapper = _lc_wrapper(n_frames=12)
    events, second_started = [], threading.Event()

    # Forward on the worker: under no_grad; the 2nd call flags that it has started
    def fake_forward(model, views, **kwargs):
        assert not torch.is_grad_enabled()
        events.append("fwd")

        if events.count("fwd") == 2:
            second_started.set()

        return _make_raw(views.shape[0])

    wrapper.base._forward = fake_forward
    add_window_submap = LoopClosure._add_window_submap

    # Window 0 is recorded only once forward 2 runs; a serial loop times out here
    def spy(self, submap, *args):
        if submap.submap_id == 0:
            assert second_started.wait(5)

        events.append(f"add{submap.submap_id}")
        return add_window_submap(self, submap, *args)

    with patch.object(LoopClosure, "_add_window_submap", spy):
        wrapper._run_lc_loop()

    assert [e for e in events if e.startswith("add")] == ["add0", "add1", "add2", "add3"]
    assert events.count("fwd") == 4


def test_lc_loop_reraises_a_pipelined_forward_error():
    wrapper = _lc_wrapper(n_frames=12)
    calls = []

    # The 3rd forward raises on the worker thread
    def fake_forward(model, views, **kwargs):
        calls.append(views.shape[0])

        if len(calls) == 3:
            raise RuntimeError("forward 2 failed")

        return _make_raw(views.shape[0])

    wrapper.base._forward = fake_forward

    with pytest.raises(RuntimeError, match="forward 2 failed"):
        wrapper._run_lc_loop()
