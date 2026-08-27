"""
Reconstructor wiring for the VDA context stream, the depth alignment model and the SfM seed.

- `_context_keep_rows` maps keyframes onto the context grid, or refuses (None) when they
  do not line up — the fallback that keeps a misaligned depth stack from being written.
- `extract_frames` draws keyframes from the grid so that mapping holds by construction.
- `_run_sfm` decodes the context stream, stamps a sidecar so a rate change invalidates
  cached depth, and forwards `depth_align` / `random_seed` from the config.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.undistort import DistortionProfile
from collab_splats.preproc.video import context_indices
from collab_splats.wrapper.reconstructor import (
    Reconstructor,
    _context_keep_rows,
    extract_frames,
)
from tests.wrapper.test_splats_stage import _stub_reconstructor

RECONSTRUCTOR = "collab_splats.wrapper.reconstructor"


########################################################################
# Fixtures and helpers
########################################################################


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """
    A 60-frame 320x240 30 fps mp4 — the same shape tests/preproc's fixture builds.
    """
    path = tmp_path_factory.mktemp("t11_vid") / "tiny.mp4"
    width, height = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    rng = np.random.default_rng(0)
    noise = (rng.random((height, width, 3)) * 255).astype(np.uint8)

    for i in range(60):
        frame = noise.copy()
        cv2.rectangle(frame, (10 + i * 4, 60), (70 + i * 4, 140), (0, 255, 0), -1)
        writer.write(frame)

    writer.release()
    return str(path)


def _clean_report(n=60):
    """
    A quality report shaped like compute_video_quality's, with every frame passing the gate.
    """
    return {
        "available": True,
        "frames": {
            "frame_idx": list(range(n)),
            "laplacian": [200.0] * n,
            "exposure_mean": [128.0] * n,
            "exposure_std": [40.0] * n,
            "blur": [0.3] * n,
        },
    }


def _sfm_reconstructor(
    tmp_path,
    *,
    video_path,
    vda_context_fps=None,
    depth_align="scale",
    random_seed=None,
    frame_idx=(0, 9, 30, 57),
    video_mtime=None,
    undistort=None,
):
    """
    A method:sfm Reconstructor backed by a real frames.zarr holding `frame_idx` keyframes.
    """
    config = {
        "input_path": str(video_path),
        "output_path": str(tmp_path / "out"),
        "preproc": {"fps": 2.0, "vda_context_fps": vda_context_fps},
        "pointcloud": {
            "method": "sfm",
            "backend": "instantsfm",
            "instantsfm": {"depth_align": depth_align, "random_seed": random_seed},
        },
    }
    recon = Reconstructor(config)

    # Real store: _run_sfm reads frame_indices(), provenance() and images() off it
    frames = [np.full((8, 8, 3), i, np.uint8) for i in frame_idx]
    records = [{"frame_idx": int(i), "blur_score": 1.0} for i in frame_idx]
    FrameStore.create(
        recon.frames_zarr,
        frames,
        records,
        provenance={
            "video_path": str(video_path),
            "video_mtime": video_mtime,
            "method": "uniform",
            "vda_context_fps": vda_context_fps,
            **({"undistort": undistort} if undistort else {}),
        },
    )
    return recon


def _patched_sfm(recon, *, depth_complete=True, on_generate=None):
    """
    Patch every heavy leg of _run_sfm; returns the name -> patcher dict, unstarted.

    - on_generate: called with no arguments when generate_vda_depth fires, so a test can
      assert on state at that moment rather than after the run has finished.
    """
    outputs = MagicMock()
    outputs.points = np.zeros((3, 3), np.float32)
    outputs.image_paths = [Path("frame_000000.jpg")]

    patches = {
        "vda_depth_complete": patch(f"{RECONSTRUCTOR}.vda_depth_complete", return_value=depth_complete),
        "generate_vda_depth": patch(
            f"{RECONSTRUCTOR}.generate_vda_depth",
            side_effect=(lambda *_args, **_kwargs: on_generate()) if on_generate else None,
        ),
        "decode_context": patch(f"{RECONSTRUCTOR}.decode_context", return_value=np.zeros((20, 4, 4, 3), np.uint8)),
        "InstantSfMCreator": patch(f"{RECONSTRUCTOR}.InstantSfMCreator"),
        "_rename_images_to_stems": patch(f"{RECONSTRUCTOR}._rename_images_to_stems"),
        "apply_depth_alignment": patch(f"{RECONSTRUCTOR}.apply_depth_alignment", return_value={}),
        "torch": patch(f"{RECONSTRUCTOR}.torch", SimpleNamespace(cuda=MagicMock(is_available=lambda: False))),
        "_sfm_result_from_reconstruction": patch.object(
            Reconstructor, "_sfm_result_from_reconstruction", return_value=outputs
        ),
    }
    return patches


def _run_sfm_with_mocks(recon, *, depth_complete=True, on_generate=None):
    """
    Execute _run_sfm against the patched heavy legs; returns the mock dict for assertions.
    """
    patches = _patched_sfm(recon, depth_complete=depth_complete, on_generate=on_generate)

    # ExitStack, not start()-in-a-comprehension: a patch that fails to apply must not leak
    # the ones already started into the rest of the session
    with ExitStack() as stack:
        started = {name: stack.enter_context(patcher) for name, patcher in patches.items()}
        recon._run_sfm()

    return started


########################################################################
# _context_keep_rows
########################################################################


def test_keep_rows_map_keyframes_onto_the_grid():
    grid = list(range(0, 60, 3))
    assert _context_keep_rows(grid, [0, 9, 30, 57]) == [0, 3, 10, 19]


def test_keep_rows_returns_none_when_a_keyframe_is_off_grid(caplog):
    grid = list(range(0, 60, 3))
    with caplog.at_level("WARNING"):
        assert _context_keep_rows(grid, [0, 10, 30]) is None
    assert "off the context grid" in caplog.text


def test_keep_rows_returns_none_when_a_keyframe_is_past_the_grid(caplog):
    # searchsorted clips past-the-end keyframes onto the last member; that must read as
    # off-grid, not as a spurious match on the tail
    with caplog.at_level("WARNING"):
        assert _context_keep_rows([0, 3, 6], [0, 99]) is None
    assert "off the context grid" in caplog.text


def test_keep_rows_returns_none_for_an_empty_grid(caplog):
    with caplog.at_level("WARNING"):
        assert _context_keep_rows([], [0, 3]) is None
    assert "empty" in caplog.text


########################################################################
# The context grid itself
########################################################################


def test_context_grid_contains_a_keyframe_grid_at_a_multiple_rate():
    # A 10 fps context grid and a 5 fps keyframe grid on the same video: strict subset
    info = {"total_frames": 60, "fps": 30.0}
    context = context_indices("unused.mp4", target_fps=10.0, info=info)
    keyframes = context_indices("unused.mp4", target_fps=5.0, info=info)
    assert set(keyframes) <= set(context)


def test_context_grid_always_spans_the_whole_video():
    # context_indices is range(0, total, step), so a partial-coverage grid — the failure
    # mode _sample_by_quality accepts silently — cannot come out of this call site
    info = {"total_frames": 601, "fps": 30.0}
    grid = context_indices("unused.mp4", target_fps=8.0, info=info)
    assert grid[0] == 0
    assert grid[-1] > info["total_frames"] - (grid[1] - grid[0]) - 1


########################################################################
# extract_frames
########################################################################


def test_vda_context_fps_rejects_optical_flow(tmp_path):
    # The guard is a config error and must fire before the video is probed, so the probe is
    # patched to explode: a passing test then proves the raise came from the guard. Match the
    # guard's own wording, never the bare knob name — pytest names tmp_path after the test, so
    # "vda_context_fps" appears in every path this test prints and would match either way.
    with patch(f"{RECONSTRUCTOR}.get_video_info", side_effect=AssertionError("video was probed")):
        with pytest.raises(ValueError, match="requires frame_selection 'fps' or 'uniform'"):
            extract_frames(
                input_path=tmp_path / "missing.mp4",
                frames_zarr=tmp_path / "frames.zarr",
                frame_selection="optical_flow",
                fps=None,
                min_frames=None,
                max_frames=10,
                vda_context_fps=8.0,
            )


def test_extract_frames_draws_keyframes_from_the_context_grid(tiny_video, tmp_path):
    (tmp_path / "video_quality_report.json").write_text(json.dumps(_clean_report()))
    frames_zarr = tmp_path / "frames.zarr"

    with patch(f"{RECONSTRUCTOR}.preproc_viz"):
        extract_frames(
            input_path=Path(tiny_video),
            frames_zarr=frames_zarr,
            frame_selection="uniform",
            fps=None,
            min_frames=None,
            max_frames=6,
            search_radius=7,
            vda_context_fps=10.0,
        )

    store = FrameStore.open(frames_zarr)
    grid = context_indices(tiny_video, target_fps=10.0)
    chosen = [int(i) for i in store.frame_indices()]

    assert set(chosen) <= set(grid)
    assert _context_keep_rows(grid, chosen) is not None
    assert store.provenance()["vda_context_fps"] == 10.0


def test_extract_frames_without_context_fps_records_none(tiny_video, tmp_path):
    (tmp_path / "video_quality_report.json").write_text(json.dumps(_clean_report()))
    frames_zarr = tmp_path / "frames.zarr"

    with patch(f"{RECONSTRUCTOR}.preproc_viz"):
        extract_frames(
            input_path=Path(tiny_video),
            frames_zarr=frames_zarr,
            frame_selection="fps",
            fps=5.0,
            min_frames=None,
            max_frames=300,
            search_radius=7,
        )

    assert FrameStore.open(frames_zarr).provenance()["vda_context_fps"] is None


########################################################################
# _run_sfm: the context stream
########################################################################


def test_run_sfm_decodes_the_context_stream_and_keeps_the_keyframe_rows(tiny_video, tmp_path):
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=10.0)
    mocks = _run_sfm_with_mocks(recon, depth_complete=False)

    grid = context_indices(tiny_video, target_fps=10.0)
    assert mocks["decode_context"].call_args.args[1] == grid

    kwargs = mocks["generate_vda_depth"].call_args.kwargs
    assert kwargs["keep_rows"] == [0, 3, 10, 19]
    assert kwargs["fps"] == 10.0


def test_run_sfm_falls_back_to_keyframes_when_the_video_is_gone(tmp_path, caplog):
    recon = _sfm_reconstructor(tmp_path, video_path=tmp_path / "gone.mp4", vda_context_fps=10.0)
    with caplog.at_level("WARNING"):
        mocks = _run_sfm_with_mocks(recon, depth_complete=False)

    assert mocks["decode_context"].call_count == 0
    assert mocks["generate_vda_depth"].call_args.kwargs.get("keep_rows") is None
    assert mocks["generate_vda_depth"].call_args.kwargs["fps"] == 2.0
    assert "source video is unavailable" in caplog.text


def test_run_sfm_falls_back_when_keyframes_are_off_the_grid(tiny_video, tmp_path, caplog):
    # 10 is not a multiple of 3, so the 10 fps grid cannot hold this keyframe set
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=10.0, frame_idx=(0, 10, 30))
    with caplog.at_level("WARNING"):
        mocks = _run_sfm_with_mocks(recon, depth_complete=False)

    assert mocks["decode_context"].call_count == 0
    assert mocks["generate_vda_depth"].call_args.kwargs.get("keep_rows") is None
    assert "off the context grid" in caplog.text


def test_run_sfm_without_context_fps_never_decodes(tiny_video, tmp_path):
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=None)
    mocks = _run_sfm_with_mocks(recon, depth_complete=False)

    assert mocks["decode_context"].call_count == 0
    assert mocks["generate_vda_depth"].call_args.kwargs["fps"] == 2.0


def test_run_sfm_decodes_the_context_stream_with_the_stores_distortion_profile(tiny_video, tmp_path):
    # Context frames must be undistorted with the SAME profile frames.zarr was written with,
    # or the two disagree on K_new and the alpha=0 crop and the depth lands on a shifted grid
    profile = DistortionProfile(
        k1=-0.21,
        k2=0.04,
        p1=0.001,
        p2=-0.002,
        fx=900.0,
        fy=900.0,
        cx=640.0,
        cy=360.0,
        width=1280,
        height=720,
    )
    recon = _sfm_reconstructor(
        tmp_path,
        video_path=tiny_video,
        vda_context_fps=10.0,
        undistort={"profile": profile.to_dict(), "K_new": np.eye(3).tolist(), "roi": [0, 0, 1280, 720]},
    )

    mocks = _run_sfm_with_mocks(recon, depth_complete=False)

    assert mocks["decode_context"].call_args.kwargs["profile"] == profile


def test_run_sfm_warns_when_the_context_rate_differs_from_the_sampled_grid(tiny_video, tmp_path, caplog):
    # frames.zarr records the grid its keyframes came from; a different rate here means VDA
    # runs on a grid the keyframes were not selected against
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=10.0)
    recon.config["preproc"]["vda_context_fps"] = 20.0

    with caplog.at_level("WARNING"):
        _run_sfm_with_mocks(recon, depth_complete=False)

    assert "was sampled against a 10.00 fps grid" in caplog.text


########################################################################
# _run_sfm: the depth cache sidecar
########################################################################


def test_run_sfm_reuses_complete_depth_when_no_sidecar_exists(tiny_video, tmp_path):
    # Scenes that predate the stamp are trusted, not force-regenerated
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=None)
    mocks = _run_sfm_with_mocks(recon, depth_complete=True)

    assert mocks["generate_vda_depth"].call_count == 0
    assert not (recon.backend_dir / "depth_vda" / "inputs.json").exists()


def test_run_sfm_reuses_complete_depth_when_the_sidecar_matches(tiny_video, tmp_path):
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=10.0)
    sidecar = recon.backend_dir / "depth_vda" / "inputs.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"context_fps": 10.0, "keyframe_fps": 2.0, "n_names": 4}))

    mocks = _run_sfm_with_mocks(recon, depth_complete=True)
    assert mocks["generate_vda_depth"].call_count == 0


def _seed_depth_maps(recon, signature):
    """
    Write a sidecar plus one real depth map, so invalidation can be checked by effect.
    """
    npy_dir = recon.backend_dir / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    stale_map = npy_dir / "frame_000000.npy"
    np.save(stale_map, np.ones((4, 4), dtype=np.float32))

    sidecar = recon.backend_dir / "depth_vda" / "inputs.json"
    sidecar.write_text(json.dumps(signature))

    return stale_map


def test_run_sfm_regenerates_when_the_context_rate_changed(tiny_video, tmp_path, caplog):
    # Same keyframes and the same npy stems — only the context rate moved, which
    # vda_depth_complete cannot see. generate_vda_depth early-returns on a complete stem
    # set, so the superseded maps must be gone by the time it is called, not merely re-stamped.
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=10.0)
    stale_map = _seed_depth_maps(recon, {"context_fps": None, "keyframe_fps": 2.0, "n_names": 4})

    surviving = []
    with caplog.at_level("INFO"):
        mocks = _run_sfm_with_mocks(
            recon,
            depth_complete=True,
            on_generate=lambda: surviving.append(stale_map.exists()),
        )

    assert mocks["generate_vda_depth"].call_count == 1
    assert surviving == [False], "stale depth maps outlived the invalidation that should have deleted them"
    assert "VDA depth cache invalidated" in caplog.text


def test_run_sfm_keeps_depth_maps_when_the_sidecar_matches(tiny_video, tmp_path):
    # Mirror of the test above: a matching stamp must not delete anything
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=10.0)
    kept = _seed_depth_maps(recon, {"context_fps": 10.0, "keyframe_fps": 2.0, "n_names": 4})

    mocks = _run_sfm_with_mocks(recon, depth_complete=True)

    assert mocks["generate_vda_depth"].call_count == 0
    assert kept.exists()


def test_run_sfm_regenerates_when_the_sidecar_is_unreadable(tiny_video, tmp_path, caplog):
    # An OOM-killed run can leave a truncated stamp; a cache check is the least explicable
    # place to die, and regenerating is always safe
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=10.0)
    _seed_depth_maps(recon, {"context_fps": 10.0, "keyframe_fps": 2.0, "n_names": 4})
    (recon.backend_dir / "depth_vda" / "inputs.json").write_text('{"context_fps": 10.')

    with caplog.at_level("WARNING"):
        mocks = _run_sfm_with_mocks(recon, depth_complete=True)

    assert mocks["generate_vda_depth"].call_count == 1
    assert "unreadable" in caplog.text


def test_run_sfm_stamps_the_sidecar_after_generating(tiny_video, tmp_path):
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, vda_context_fps=10.0)
    _run_sfm_with_mocks(recon, depth_complete=False)

    sidecar = recon.backend_dir / "depth_vda" / "inputs.json"
    assert json.loads(sidecar.read_text()) == {"context_fps": 10.0, "keyframe_fps": 2.0, "n_names": 4}


def test_run_sfm_stamps_no_context_rate_when_the_context_stream_did_not_run(tmp_path, caplog):
    # Video gone (rerun-from-processed): the run falls back to keyframe-only VDA, so recording
    # the requested rate would let a later run with the video present trust a pass that never happened
    recon = _sfm_reconstructor(tmp_path, video_path=tmp_path / "gone.mp4", vda_context_fps=10.0)

    with caplog.at_level("WARNING"):
        mocks = _run_sfm_with_mocks(recon, depth_complete=False)

    assert mocks["decode_context"].call_count == 0
    sidecar = recon.backend_dir / "depth_vda" / "inputs.json"
    assert json.loads(sidecar.read_text())["context_fps"] is None


def test_run_sfm_falls_back_when_the_video_changed_since_frames_zarr(tiny_video, tmp_path, caplog):
    # A re-encoded clip at the same path shifts every grid index, pairing keyframes with
    # other frames' depth — and both grids start at 0, so nothing downstream would error
    recon = _sfm_reconstructor(
        tmp_path,
        video_path=tiny_video,
        vda_context_fps=10.0,
        video_mtime=Path(tiny_video).stat().st_mtime - 3600.0,
    )

    with caplog.at_level("WARNING"):
        mocks = _run_sfm_with_mocks(recon, depth_complete=False)

    assert mocks["decode_context"].call_count == 0
    assert "was modified since frames.zarr was written" in caplog.text


########################################################################
# _run_sfm: config pass-through
########################################################################


def test_run_sfm_forwards_depth_align_from_the_config(tiny_video, tmp_path):
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, depth_align="affine")
    mocks = _run_sfm_with_mocks(recon)
    assert mocks["apply_depth_alignment"].call_args.kwargs["model"] == "affine"


def test_run_sfm_defaults_depth_align_to_scale(tiny_video, tmp_path):
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video)
    mocks = _run_sfm_with_mocks(recon)
    assert mocks["apply_depth_alignment"].call_args.kwargs["model"] == "scale"


def test_ensure_vda_depth_stamps_only_the_rate_that_actually_ran(tiny_video, tmp_path):
    # The extracted method stands on its own: three patches and no SfM, where _run_sfm needs
    # eight. Frame 5 is off the 10 fps grid, so the context stream never runs — and a rate
    # that never ran must not be stamped as if it had.
    recon = _sfm_reconstructor(
        tmp_path, video_path=tiny_video, vda_context_fps=10.0, frame_idx=(0, 5, 30, 57)
    )
    store = FrameStore.open(recon.frames_zarr)
    names = [f"frame_{int(fi):06d}.jpg" for fi in store.frame_indices()]

    with ExitStack() as stack:
        stack.enter_context(patch(f"{RECONSTRUCTOR}.vda_depth_complete", return_value=False))
        generate = stack.enter_context(patch(f"{RECONSTRUCTOR}.generate_vda_depth"))
        stack.enter_context(
            patch(f"{RECONSTRUCTOR}.torch", SimpleNamespace(cuda=MagicMock(is_available=lambda: False)))
        )
        recon._ensure_vda_depth(recon.backend_dir, store, names)

    assert generate.call_count == 1
    assert generate.call_args.kwargs.get("keep_rows") is None
    sidecar = json.loads((recon.backend_dir / "depth_vda" / "inputs.json").read_text())
    assert sidecar == {"context_fps": None, "keyframe_fps": 2.0, "n_names": 4}


def test_run_sfm_forwards_random_seed_from_the_config(tiny_video, tmp_path):
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video, random_seed=7)
    mocks = _run_sfm_with_mocks(recon)
    assert mocks["InstantSfMCreator"].call_args.kwargs["random_seed"] == 7


def test_run_sfm_random_seed_defaults_to_none(tiny_video, tmp_path):
    recon = _sfm_reconstructor(tmp_path, video_path=tiny_video)
    mocks = _run_sfm_with_mocks(recon)
    assert mocks["InstantSfMCreator"].call_args.kwargs["random_seed"] is None


########################################################################
# The splats stage's conf_percentile report
########################################################################


def test_splats_conf_percentile_log_reports_the_masked_fraction(tmp_path, caplog):
    """
    An SfM zarr has no confidence channel; the log must say so AND quantify the masking
    that depth alignment already applied, rather than claiming the depth is unmasked.
    """
    recon = _stub_reconstructor(tmp_path)

    # Quarter of the target pixels zeroed, exactly what affine alignment writes past its
    # evidence horizon
    depth = np.ones((3, 4, 4), np.float32)
    depth[:, 0, :] = 0.0
    feedforward = SimpleNamespace(
        image_paths=[Path(f"frame_{view:06d}.jpg") for view in range(3)],
        depth=depth,
        confidence=None,
    )
    (recon.backend_dir / "pointcloud.zarr").mkdir(parents=True)

    with (
        patch("collab_splats.splats.trainer.train"),
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=feedforward),
        caplog.at_level("INFO"),
    ):
        recon.splats()

    assert "conf_percentile=20 not applied (no confidence channel)" in caplog.text
    assert "masks 25.00% of target pixels" in caplog.text


########################################################################
# Config-load validation for the late-consumed instantsfm knobs
########################################################################


def _validated_sfm_config(**instantsfm):
    """
    base.yaml merged into a method:sfm config with the given instantsfm overrides, validated.
    """
    config = {
        "input_path": "dummy.mp4",
        "output_path": "dummy_out",
        "pointcloud": {"method": "sfm", "backend": "instantsfm", "instantsfm": dict(instantsfm)},
    }
    return Reconstructor(config).config


def test_random_seed_out_of_range_is_rejected_at_config_load():
    # np.random.seed rejects this, but InstantSfM only reads the value after the SIFT +
    # exhaustive-matching pass — the whole run would burn first
    with pytest.raises(ValueError, match=r"random_seed must be null or an int"):
        _validated_sfm_config(random_seed=-1)

    with pytest.raises(ValueError, match=r"random_seed must be null or an int"):
        _validated_sfm_config(random_seed=2**32)


def test_random_seed_accepts_null_and_an_in_range_int():
    assert _validated_sfm_config(random_seed=None)["pointcloud"]["instantsfm"]["random_seed"] is None
    assert _validated_sfm_config(random_seed=2**32 - 1)["pointcloud"]["instantsfm"]["random_seed"] == 2**32 - 1


def test_depth_align_rejects_an_unknown_model_at_config_load():
    # apply_depth_alignment raises too, but only once the model is solved
    with pytest.raises(ValueError, match=r"depth_align='affinne'"):
        _validated_sfm_config(depth_align="affinne")


def test_depth_align_accepts_both_shipped_models():
    for model in ("scale", "affine"):
        assert _validated_sfm_config(depth_align=model)["pointcloud"]["instantsfm"]["depth_align"] == model
