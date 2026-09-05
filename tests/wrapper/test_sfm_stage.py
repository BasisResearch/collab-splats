"""
Reconstructor wiring for the sfm stage: the VDA depth cache and the late-consumed knobs.

- `_run_sfm` gates VDA inference on the written map set, and drops `depth_vda/` on a gate
  miss so a stem from a previous keyframe set cannot hold the gate false forever.
- It forwards `depth_align` / `random_seed` from the config to the legs that consume them
  long after the run has started.
- Both knobs are also validated at config load, so a typo fails before the SIFT pass.
"""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import Reconstructor

RECONSTRUCTOR = "collab_splats.wrapper.reconstructor"

# _run_sfm never opens the video: input_path only reaches config validation (a presence check)
# and the store's provenance stamp, so a stand-in path is enough — no encoded fixture needed.
VIDEO_PATH = "/nonexistent/tiny.mp4"

# Source frame indices the fake store is built on. Non-contiguous on purpose: a leg that
# confused source frame indices with store rows would show up.
FRAME_IDX = (0, 9, 30, 57)


########################################################################
# Fixtures and helpers
########################################################################


def _sfm_reconstructor(tmp_path, *, depth_align="scale", random_seed=None):
    """
    A method:sfm Reconstructor backed by a real frames.zarr holding FRAME_IDX keyframes.
    """
    config = {
        "input_path": VIDEO_PATH,
        "output_path": str(tmp_path / "out"),
        # 3.0, not base.yaml's own 2.0 default, so an fps assertion pins the config read
        # rather than a value the merged defaults would have supplied anyway
        "preproc": {"fps": 3.0},
        "pointcloud": {
            "method": "sfm",
            "backend": "instantsfm",
            "instantsfm": {"depth_align": depth_align, "random_seed": random_seed},
        },
    }
    recon = Reconstructor(config)

    # Real store: _run_sfm reads frame_indices() off it and exports its images to backend_dir
    frames = [np.full((8, 8, 3), i, np.uint8) for i in FRAME_IDX]
    records = [{"frame_idx": int(i), "blur_score": 1.0} for i in FRAME_IDX]
    FrameStore.create(
        recon.frames_zarr,
        frames,
        records,
        provenance={
            "video_path": VIDEO_PATH,
            "method": "uniform",
        },
    )
    return recon


def _patched_sfm(recon, *, depth_complete=True):
    """
    Patch every heavy leg of _run_sfm; returns the name -> patcher dict, unstarted.
    """
    outputs = MagicMock()
    outputs.points = np.zeros((3, 3), np.float32)
    outputs.image_paths = [Path("frame_000000.jpg")]

    patches = {
        "vda_depth_complete": patch(f"{RECONSTRUCTOR}.vda_depth_complete", return_value=depth_complete),
        "generate_vda_depth": patch(f"{RECONSTRUCTOR}.generate_vda_depth"),
        "InstantSfMCreator": patch(f"{RECONSTRUCTOR}.InstantSfMCreator"),
        "_rename_images_to_stems": patch(f"{RECONSTRUCTOR}._rename_images_to_stems"),
        "apply_depth_alignment": patch(f"{RECONSTRUCTOR}.apply_depth_alignment", return_value={}),
        "torch": patch(f"{RECONSTRUCTOR}.torch", SimpleNamespace(cuda=MagicMock(is_available=lambda: False))),
        # The zarr provenance stamp reads instantsfm's installed version. It is a hard dependency
        # of the real path but absent from some dev venvs, and none of these tests assert on it —
        # stub it so a missing package cannot take the wiring coverage down with it.
        "importlib": patch(
            f"{RECONSTRUCTOR}.importlib",
            SimpleNamespace(metadata=SimpleNamespace(version=lambda _package: "0.0.0")),
        ),
        "_sfm_result_from_reconstruction": patch.object(
            Reconstructor, "_sfm_result_from_reconstruction", return_value=outputs
        ),
    }
    return patches


def _run_sfm_with_mocks(recon, *, depth_complete=True):
    """
    Execute _run_sfm against the patched heavy legs; returns the mock dict for assertions.
    """
    patches = _patched_sfm(recon, depth_complete=depth_complete)

    # ExitStack, not start()-in-a-comprehension: a patch that fails to apply must not leak
    # the ones already started into the rest of the session
    with ExitStack() as stack:
        started = {name: stack.enter_context(patcher) for name, patcher in patches.items()}
        recon._run_sfm()

    return started


def _npy_dir(recon):
    """
    The VDA map directory the completeness gate reads.
    """
    return recon.backend_dir / "depth_vda" / "images" / "npy"


########################################################################
# _run_sfm: the VDA depth cache gate
########################################################################


def test_run_sfm_skips_vda_inference_when_the_map_set_is_complete(tmp_path):
    recon = _sfm_reconstructor(tmp_path)

    # A cached map on the hit path must survive untouched — the prune belongs to the miss branch
    npy_dir = _npy_dir(recon)
    npy_dir.mkdir(parents=True)
    (npy_dir / "frame_000000.npy").write_bytes(b"cached")

    mocks = _run_sfm_with_mocks(recon, depth_complete=True)

    mocks["generate_vda_depth"].assert_not_called()
    assert (npy_dir / "frame_000000.npy").read_bytes() == b"cached"


def test_run_sfm_regenerates_vda_depth_and_drops_stale_maps_on_a_gate_miss(tmp_path):
    recon = _sfm_reconstructor(tmp_path)

    # A stem from a previous keyframe set. generate_vda_depth has no delete path, so leaving
    # this behind would keep vda_depth_complete's set equality false on every subsequent run
    # and re-run the full GPU inference forever.
    npy_dir = _npy_dir(recon)
    npy_dir.mkdir(parents=True)
    stale = npy_dir / "stale.npy"
    stale.write_bytes(b"stale")

    mocks = _run_sfm_with_mocks(recon, depth_complete=False)

    assert mocks["generate_vda_depth"].call_count == 1
    kwargs = mocks["generate_vda_depth"].call_args.kwargs
    assert kwargs["fps"] == 3.0  # float(config["preproc"]["fps"]), not a source literal
    assert kwargs["out_dir"] == recon.backend_dir
    assert kwargs["names"] == [f"frame_{i:06d}.jpg" for i in FRAME_IDX]
    assert not stale.exists()


########################################################################
# _run_sfm: config pass-through
########################################################################


def test_run_sfm_forwards_depth_align_from_the_config(tmp_path):
    recon = _sfm_reconstructor(tmp_path, depth_align="affine")
    mocks = _run_sfm_with_mocks(recon)
    assert mocks["apply_depth_alignment"].call_args.kwargs["model"] == "affine"


def test_run_sfm_defaults_depth_align_to_scale(tmp_path):
    recon = _sfm_reconstructor(tmp_path)
    mocks = _run_sfm_with_mocks(recon)
    assert mocks["apply_depth_alignment"].call_args.kwargs["model"] == "scale"


def test_run_sfm_forwards_random_seed_from_the_config(tmp_path):
    recon = _sfm_reconstructor(tmp_path, random_seed=7)
    mocks = _run_sfm_with_mocks(recon)
    assert mocks["InstantSfMCreator"].call_args.kwargs["random_seed"] == 7


def test_run_sfm_random_seed_defaults_to_none(tmp_path):
    recon = _sfm_reconstructor(tmp_path)
    mocks = _run_sfm_with_mocks(recon)
    assert mocks["InstantSfMCreator"].call_args.kwargs["random_seed"] is None


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
