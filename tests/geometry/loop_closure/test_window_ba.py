"""
Bundle adjustment inside each loop-closure window (LoopClosure ba=...).
"""

import inspect
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.geometry.bundle_adjustment import (
    BundleAdjustment,
    BundleAdjustmentConfig,
)
from collab_splats.geometry.loop_closure.matching import LoopMatch
from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig

WRAPPER = "collab_splats.geometry.loop_closure.wrapper"


def _raw(k: int, channel: bool = True) -> dict:
    """
    Forward output for k frames on a 4x4 grid: identity poses, constant depth, varied confidence.
    """
    rows = np.arange(4, dtype=np.float32)[:, None]
    cols = np.arange(4, dtype=np.float32)[None, :]
    K = np.array([[100.0, 0, 2], [0, 100.0, 2], [0, 0, 1]], dtype=np.float32)
    depth = (
        np.ones((k, 4, 4, 1), dtype=np.float32)
        if channel
        else np.ones((k, 4, 4), dtype=np.float32)
    )
    return {
        "extrinsic": np.tile(np.eye(3, 4, dtype=np.float32), (k, 1, 1)),
        "intrinsics": np.tile(K, (k, 1, 1)),
        "depth": depth,
        "depth_conf": np.tile(50.0 + rows + cols, (k, 1, 1)).astype(np.float32),
        "images": np.zeros((k, 3, 4, 4), dtype=np.float32),
    }


def _wrapper(
    n_frames: int, submap_size: int, ba: BundleAdjustmentConfig | None
) -> LoopClosure:
    """
    LoopClosure over a MagicMock creator with n_frames tensor views and a stub forward.
    """
    base = MagicMock()
    base.max_points = 500_000
    base.views = torch.zeros(n_frames, 3, 4, 4)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.tile(
        np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1)
    )
    base._forward = lambda model, views: _raw(views.shape[0])
    base.default_verify_match_ratio = 1.0
    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=1,
        min_submap_gap=0,
        max_loops_per_submap=0,
        lc_retrieval_threshold=999.0,
    )
    wrapper = LoopClosure(base, config=cfg, ba=ba)
    wrapper.frame_paths = [Path(f"/images/frame_{i:06d}.png") for i in range(n_frames)]
    return wrapper


class _FakeBA:
    """
    BundleAdjustment stand-in: records each call, returns poses shifted by +1 in x and focal 77.
    """

    calls: list[dict] = []

    def __init__(self, config: BundleAdjustmentConfig) -> None:
        self.config = config
        self.alignment_scale = 1.0
        self.loss_history = [[3.0, 2.0]]

    def refine(self, *args, **kwargs):
        # Bind to the real signature, so a call the real refine would refuse fails here too
        call = (
            inspect.signature(BundleAdjustment.refine)
            .bind(self, *args, **kwargs)
            .arguments
        )
        extrinsics, intrinsics = call["extrinsics"], call["intrinsics"]
        type(self).calls.append(
            {
                "config": self.config,
                "images": call["images"],
                "intrinsics": intrinsics.copy(),
                "frame_paths": call.get("frame_paths"),
                "depth": call["depth"],
                "world_points": call["world_points"],
            }
        )
        refined = extrinsics.astype(np.float64).copy()
        refined[:, 0, 3] += 1.0
        K = intrinsics.astype(np.float64).copy()
        K[:, 0, 0] = 77.0
        K[:, 1, 1] = 77.0
        return refined, K


@pytest.fixture
def fake_ba():
    """
    Patch the wrapper's BundleAdjustment with _FakeBA; yields the call list.
    """
    _FakeBA.calls = []

    with patch(f"{WRAPPER}.BundleAdjustment", _FakeBA):
        yield _FakeBA.calls


def _run_loop(wrapper: LoopClosure, matches: Callable[..., list] | None = None) -> None:
    """
    run_inference with retrieval stubbed out; loop detection returns no matches unless given.
    """
    with (
        patch(f"{WRAPPER}.BaseRetrievalExtractor") as retrieval,
        patch(f"{WRAPPER}.find_loop_closures", side_effect=matches, return_value=[]),
    ):
        retrieval.get.return_value = lambda device: (
            lambda frames: torch.zeros(frames.shape[0], 128)
        )
        wrapper.run_inference()


def test_ba_runs_once_per_window_over_that_windows_frames(fake_ba, tmp_path):
    # 9 frames, submap 3 + overlap 1: windows start at 0, 3, 6
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(tracks_cache_dir=tmp_path))
    _run_loop(wrapper)

    assert [r["start"] for r in wrapper.window_ba] == [0, 3, 6]
    assert [r["ok"] for r in wrapper.window_ba] == [True, True, True]
    assert [len(c["images"]) for c in fake_ba] == [4, 4, 3]
    assert [len(c["depth"]) for c in fake_ba] == [4, 4, 3]


def test_ba_gets_that_windows_full_res_frame_paths(fake_ba):
    # Matcher tracks read the window's full-res frames from the images/ store
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig())
    _run_loop(wrapper)

    assert [c["frame_paths"] for c in fake_ba] == [
        [Path(f"/images/frame_{i:06d}.png") for i in range(0, 4)],
        [Path(f"/images/frame_{i:06d}.png") for i in range(3, 7)],
        [Path(f"/images/frame_{i:06d}.png") for i in range(6, 9)],
    ]


def test_reconstruct_keeps_full_frame_paths():
    # The base creator keeps stems only; the wrapper keeps the paths it was given
    wrapper = _wrapper(3, 3, BundleAdjustmentConfig())
    paths = [Path(f"/data/images/frame_{i:06d}.png") for i in range(3)]

    with patch.object(LoopClosure, "run_inference"):
        wrapper._reconstruct(paths, Path("/out"))

    assert wrapper.frame_paths == paths


def test_ba_gets_squeezed_depth_and_window_frames(fake_ba):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig())
    wrapper.base.views = torch.full((9, 3, 4, 4), 0.25)
    _run_loop(wrapper)

    call = fake_ba[0]
    assert call["depth"].shape == (4, 4, 4)
    np.testing.assert_array_equal(
        call["images"], np.full((4, 3, 4, 4), 0.25, np.float32)
    )
    assert call["world_points"].shape == (4, 4, 4, 3)


def test_mapanything_window_uses_raw_images(fake_ba):
    # View-dict window: frames come from raw["images"], depth is already (k, H, W)
    raw = _raw(3, channel=False)
    raw["images"] = np.full((3, 3, 4, 4), 0.5, dtype=np.float32)
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())

    with patch.object(wrapper.base, "_forward", return_value=raw):
        out, _ = wrapper._forward_window([{"img": None}] * 3, 0)

    np.testing.assert_array_equal(fake_ba[0]["images"], raw["images"])
    assert fake_ba[0]["depth"].shape == (3, 4, 4)
    assert out is raw


def test_later_windows_hold_window_zero_focal(fake_ba):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(refine_focal=True, solver="lm"))
    _run_loop(wrapper)

    assert fake_ba[0]["config"].refine_focal is True
    assert fake_ba[0]["intrinsics"][0, 0, 0] == 100.0

    for call in fake_ba[1:]:
        assert call["config"].refine_focal is False
        assert call["intrinsics"][0, 0, 0] == 77.0
        assert call["intrinsics"][0, 1, 1] == 77.0


def test_refine_focal_false_never_holds(fake_ba):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(refine_focal=False))
    _run_loop(wrapper)

    for call in fake_ba:
        assert call["intrinsics"][0, 0, 0] == 100.0


def test_matcher_tracks_without_frame_paths_raise(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig(track_source="xfeat"))
    wrapper.frame_paths = []

    # A config error, not a failed window: it is not swallowed into feedforward poses
    with pytest.raises(ValueError, match="frame_paths"):
        wrapper._forward_window(wrapper.base.views, 0)

    assert fake_ba == []


def test_failed_window_keeps_feedforward_poses(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    raw = _raw(3)
    expected = raw["extrinsic"].copy()

    with (
        patch.object(_FakeBA, "refine", side_effect=ValueError("too few points")),
        patch.object(wrapper.base, "_forward", return_value=raw),
    ):
        out, _ = wrapper._forward_window(wrapper.base.views, 0)

    np.testing.assert_array_equal(out["extrinsic"], expected)
    assert wrapper.window_ba[0]["ok"] is False
    assert wrapper.window_ba[0]["focal"] == 100.0
    assert wrapper._ba_focal is None


def test_refined_poses_are_anchored_to_frame_zero(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())

    # Per-frame distinct 4x4 poses: frame i rotated (i+1)*10 deg about z, shifted i+1 in x
    poses = np.tile(np.eye(4), (3, 1, 1))

    for i in range(3):
        angle = np.deg2rad(10.0 * (i + 1))
        poses[i, :2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        poses[i, 0, 3] = float(i + 1)

    K = np.tile(np.eye(3), (3, 1, 1))
    K[:, 0, 0] = 77.0

    with patch.object(_FakeBA, "refine", return_value=(poses, K)):
        out, _ = wrapper._forward_window(wrapper.base.views, 0)

    expected = poses @ np.linalg.inv(poses[0])

    for i in range(3):
        np.testing.assert_allclose(out["extrinsic"][i], expected[i, :3], atol=1e-6)

    np.testing.assert_allclose(out["extrinsic"][0], np.eye(3, 4), atol=1e-6)
    assert out["extrinsic"].dtype == np.float32
    assert out["intrinsics"].dtype == np.float32
    assert out["intrinsics"][0, 0, 0] == 77.0


def test_window_record_fields(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    wrapper._forward_window(wrapper.base.views, 0)

    record = wrapper.window_ba[0]
    assert set(record) == {
        "start",
        "n_frames",
        "ok",
        "seconds",
        "focal",
        "alignment_scale",
        "loss_final",
        "gpu_max_mib",
    }
    assert record["ok"] is True
    assert record["n_frames"] == 3
    assert record["focal"] == 77.0
    assert record["alignment_scale"] == 1.0
    assert record["loss_final"] == 2.0


def test_track_cache_dir_per_window(fake_ba, tmp_path):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(tracks_cache_dir=tmp_path))
    _run_loop(wrapper)

    dirs = [Path(c["config"].tracks_cache_dir) for c in fake_ba]
    assert dirs == [tmp_path / "w000000", tmp_path / "w000003", tmp_path / "w000006"]


def test_no_cache_dir_stays_none(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    wrapper._forward_window(wrapper.base.views, 0)

    assert fake_ba[0]["config"].tracks_cache_dir is None


def test_too_few_frames_fallback_refines_once(fake_ba):
    # 3 frames < submap_size 20: one whole-scene forward, refined as window 0
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    wrapper.run_inference()

    assert len(fake_ba) == 1
    assert wrapper.window_ba[0]["start"] == 0
    assert wrapper.base.raw_outputs["intrinsics"][0, 0, 0] == 77.0


def test_dino_salad_failure_fallback_refines_once(fake_ba):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig())

    with patch(f"{WRAPPER}.BaseRetrievalExtractor") as retrieval:
        retrieval.get.side_effect = OSError("no weights")
        wrapper.run_inference()

    assert len(fake_ba) == 1
    assert wrapper.window_ba[0]["n_frames"] == 9
    assert wrapper.outputs is None


def test_verify_pairs_are_never_refined(fake_ba):
    # One candidate on the third window reaches verify; its pair forward must not be refined
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig())
    base = wrapper.base

    def find_loops(submap, past, threshold, max_loops, nms_frame_distance):
        if submap.submap_id != 2:
            return []

        return [LoopMatch(0.5, 2, 0, 0, 0)]

    def verify(q_frame, d_frame, verify_match_ratio):
        base._forward(base.model, torch.stack([q_frame, d_frame]))
        return False, None

    base._verify_loop_candidate.side_effect = verify
    _run_loop(wrapper, matches=find_loops)

    assert base._verify_loop_candidate.call_count == 1
    assert len(fake_ba) == 3


def test_failed_first_window_lets_next_window_set_focal(fake_ba):
    # The failing call raises before recording, so fake_ba holds windows 1 and 2 at [0] and [1]
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(refine_focal=True, solver="lm"))
    real_refine = _FakeBA.refine
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1

        if calls["n"] == 1:
            raise ValueError("too few points")

        return real_refine(self, *args, **kwargs)

    with patch.object(_FakeBA, "refine", flaky):
        _run_loop(wrapper)

    assert [r["ok"] for r in wrapper.window_ba] == [False, True, True]
    assert len(fake_ba) == 2
    assert fake_ba[0]["config"].refine_focal is True
    assert fake_ba[1]["intrinsics"][0, 0, 0] == 77.0
    assert fake_ba[1]["config"].refine_focal is False


def test_run_inference_resets_window_state(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig(refine_focal=True, solver="lm"))
    wrapper.run_inference()
    wrapper.run_inference()

    assert len(wrapper.window_ba) == 1
    assert fake_ba[1]["config"].refine_focal is True


def test_ba_none_leaves_forward_untouched():
    raw = _raw(3)
    expected = {key: value.copy() for key, value in raw.items()}
    wrapper = _wrapper(3, 20, None)

    with (
        patch.object(wrapper.base, "_forward", return_value=raw),
        patch(f"{WRAPPER}.BundleAdjustment") as ba_cls,
    ):
        out, _ = wrapper._forward_window(wrapper.base.views, 0)

    ba_cls.assert_not_called()

    for key, value in expected.items():
        np.testing.assert_array_equal(out[key], value)

    assert wrapper.window_ba == []
