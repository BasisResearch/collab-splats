"""
Reconstructor wiring for the sfm stage: the depth alignment model and the SfM seed.

- `_run_sfm` forwards `depth_align` / `random_seed` from the config to the legs that
  consume them long after the run has started.
- Both knobs are also validated at config load, so a typo fails before the SIFT pass.
"""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import Reconstructor
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
    depth_align="scale",
    random_seed=None,
    frame_idx=(0, 9, 30, 57),
):
    """
    A method:sfm Reconstructor backed by a real frames.zarr holding `frame_idx` keyframes.
    """
    config = {
        "input_path": str(video_path),
        "output_path": str(tmp_path / "out"),
        "preproc": {"fps": 2.0},
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
            "method": "uniform",
        },
    )
    return recon


def _patched_sfm(recon):
    """
    Patch every heavy leg of _run_sfm; returns the name -> patcher dict, unstarted.
    """
    outputs = MagicMock()
    outputs.points = np.zeros((3, 3), np.float32)
    outputs.image_paths = [Path("frame_000000.jpg")]

    patches = {
        "vda_depth_complete": patch(f"{RECONSTRUCTOR}.vda_depth_complete", return_value=True),
        "generate_vda_depth": patch(f"{RECONSTRUCTOR}.generate_vda_depth"),
        "InstantSfMCreator": patch(f"{RECONSTRUCTOR}.InstantSfMCreator"),
        "_rename_images_to_stems": patch(f"{RECONSTRUCTOR}._rename_images_to_stems"),
        "apply_depth_alignment": patch(f"{RECONSTRUCTOR}.apply_depth_alignment", return_value={}),
        "torch": patch(f"{RECONSTRUCTOR}.torch", SimpleNamespace(cuda=MagicMock(is_available=lambda: False))),
        "_sfm_result_from_reconstruction": patch.object(
            Reconstructor, "_sfm_result_from_reconstruction", return_value=outputs
        ),
    }
    return patches


def _run_sfm_with_mocks(recon):
    """
    Execute _run_sfm against the patched heavy legs; returns the mock dict for assertions.
    """
    patches = _patched_sfm(recon)

    # ExitStack, not start()-in-a-comprehension: a patch that fails to apply must not leak
    # the ones already started into the rest of the session
    with ExitStack() as stack:
        started = {name: stack.enter_context(patcher) for name, patcher in patches.items()}
        recon._run_sfm()

    return started


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
