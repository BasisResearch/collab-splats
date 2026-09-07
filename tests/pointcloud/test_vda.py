"""
Video-Depth-Anything metric depth generation.
"""

import numpy as np
import pytest

from collab_splats.pointcloud import vda

_NAMES = ["frame_000000.jpg", "frame_000001.jpg"]


########################################################################
# vda_depth_complete: the cache gate callers check before materialising
########################################################################


def test_vda_depth_complete_is_an_exact_stem_set_check(tmp_path):
    """
    The gate compares stem sets: no dir, partial and leftover-extra are all incomplete.
    """
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    assert not vda.vda_depth_complete(tmp_path, _NAMES)  # no dir

    npy_dir.mkdir(parents=True)
    np.save(npy_dir / "frame_000000.npy", np.ones((4, 4), dtype=np.float32))
    assert not vda.vda_depth_complete(tmp_path, _NAMES)  # partial

    np.save(npy_dir / "frame_000001.npy", np.ones((4, 4), dtype=np.float32))
    assert vda.vda_depth_complete(tmp_path, _NAMES)  # exact

    np.save(npy_dir / "frame_000009.npy", np.ones((4, 4), dtype=np.float32))
    assert not vda.vda_depth_complete(tmp_path, _NAMES)  # leftover extra file


########################################################################
# generate_vda_depth
########################################################################


def test_missing_clone_raises_actionable_import_error(tmp_path, monkeypatch):
    """
    No third_party clone -> ImportError naming setup.sh, before any weight download.
    """
    monkeypatch.setattr(vda, "VDA_ROOT", tmp_path / "nope")

    # The ~1.5 GB fetch must not start when the clone that consumes it is absent
    monkeypatch.setattr(vda, "hf_hub_download", lambda **kw: pytest.fail("downloaded before the clone guard"))
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)

    with pytest.raises(ImportError, match="setup.sh"):
        vda.generate_vda_depth(frames, tmp_path, _NAMES)


def test_existing_depth_set_is_loaded_not_recomputed(tmp_path, monkeypatch):
    """
    An exact per-stem npy set short-circuits inference and comes back as the stacked array.
    """
    monkeypatch.setattr(vda, "VDA_ROOT", tmp_path / "nope")  # inference would raise
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for i, name in enumerate(_NAMES):
        np.save(npy_dir / f"{name[:-4]}.npy", np.full((4, 6), float(i + 1), dtype=np.float32))

    depths = vda.generate_vda_depth(np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES)

    assert depths.shape == (2, 4, 6)
    assert depths.dtype == np.float32
    np.testing.assert_allclose(depths[0], 1.0)
    np.testing.assert_allclose(depths[1], 2.0)


def test_partial_depth_set_does_not_short_circuit(tmp_path, monkeypatch):
    """
    One map for two names is treated as missing -> the inference path -> missing clone raises.
    """
    monkeypatch.setattr(vda, "VDA_ROOT", tmp_path / "nope")
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    np.save(npy_dir / "frame_000000.npy", np.ones((4, 4), dtype=np.float32))

    with pytest.raises(ImportError, match="setup.sh"):
        vda.generate_vda_depth(np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES)


def test_wrong_named_depth_set_does_not_short_circuit(tmp_path, monkeypatch):
    """
    Right count, wrong stems (a stale selection): the gate compares stem sets, not counts.
    """
    monkeypatch.setattr(vda, "VDA_ROOT", tmp_path / "nope")
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for stem in ("frame_000000", "frame_000007"):
        np.save(npy_dir / f"{stem}.npy", np.ones((4, 4), dtype=np.float32))

    with pytest.raises(ImportError, match="setup.sh"):
        vda.generate_vda_depth(np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES)


def test_names_and_frames_must_align(tmp_path):
    """
    Names are consumed positionally against frames — a length mismatch is a hard error.
    """
    with pytest.raises(ValueError, match="one-to-one"):
        vda.generate_vda_depth(np.zeros((3, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES)


def test_inference_result_is_resized_and_written(tmp_path, monkeypatch):
    """
    A stubbed model's (N, H, W) output is nearest-resized to depth_width and written per stem.
    """

    class _StubModel:
        def infer_video_depth(self, frames, target_fps, input_size, device, fp32):
            # One step edge per frame, mirrored between them: a two-valued map makes the
            # interpolation mode observable (bilinear would emit a blended third value), and
            # the mirroring makes the two frames distinguishable so stem/content pairing is
            # checkable. The edge sits off-center on purpose: a centered one lands between two
            # same-valued bilinear taps, so even INTER_LINEAR would blend nothing.
            n, h, w = frames.shape[:3]
            maps = np.full((n, h, w), 1.0, dtype=np.float32)
            for i in range(n):
                if i % 2 == 0:
                    maps[i, :, 38:] = 100.0
                else:
                    maps[i, :, :42] = 100.0
            return maps, target_fps

    monkeypatch.setattr(vda, "_load_vda_model", lambda device: _StubModel())
    frames = np.zeros((2, 40, 80, 3), dtype=np.uint8)

    depths = vda.generate_vda_depth(frames, tmp_path, _NAMES, depth_width=20)

    assert depths.shape == (2, 10, 20)  # 80x40 -> 20x10 keeps the aspect ratio

    # 80 -> 20 is a 4x nearest decimation: dst col d samples src col 4d, so frame 0's edge at
    # src 38 lands between dst 9 and 10 (bilinear would blend 50.5 into dst 9). Pins the
    # sampling positions AND which frame's map came back under which stem.
    np.testing.assert_array_equal(depths[0], np.tile(np.where(np.arange(20) >= 10, 100.0, 1.0), (10, 1)))
    np.testing.assert_array_equal(depths[1], np.tile(np.where(np.arange(20) >= 11, 1.0, 100.0), (10, 1)))

    # Each map lands under its own frame's stem — content, not just the filename set
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    assert sorted(p.name for p in npy_dir.glob("*.npy")) == ["frame_000000.npy", "frame_000001.npy"]
    for i, name in enumerate(_NAMES):
        np.testing.assert_array_equal(np.load(npy_dir / f"{name[:-4]}.npy"), depths[i])
