"""
Directory-backed keyframe store: images/frame_NNNNNN.png.
"""

import numpy as np
import pytest

from collab_splats.preproc import frames as fr


def _frames(n=3, h=8, w=12):
    """
    n deterministic RGB frames, each a different flat color.
    """
    return [np.full((h, w, 3), i * 40 + 5, dtype=np.uint8) for i in range(n)]


def test_frame_idx_from_path_reads_the_numeric_tail():
    assert fr.frame_idx_from_path("images/frame_000042.png") == 42
    assert fr.frame_idx_from_path("IMG_1234.jpg") == 1234


def test_write_then_read_round_trips_rgb(tmp_path):
    images = tmp_path / "images"
    written = fr.write_frames(images, _frames(3), [0, 5, 11])

    assert [p.name for p in written] == ["frame_000000.png", "frame_000005.png", "frame_000011.png"]

    out = fr.read_frames(images)
    assert out.shape == (3, 8, 12, 3)
    assert out.dtype == np.uint8

    # PNG is lossless and the store is RGB at both boundaries
    np.testing.assert_array_equal(out, np.stack(_frames(3)))


def test_read_frames_selects_by_frame_idx_not_position(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), [0, 5, 11])

    out = fr.read_frames(images, idxs=[11, 0])
    np.testing.assert_array_equal(out[0], _frames(3)[2])
    np.testing.assert_array_equal(out[1], _frames(3)[0])


def test_read_frames_raises_on_an_index_the_directory_does_not_hold(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), [0, 5, 11])

    with pytest.raises(KeyError, match="7"):
        fr.read_frames(images, idxs=[7])


def test_frame_paths_is_sorted_and_extension_filtered(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), [11, 0, 5])
    (images / "notes.txt").write_text("ignore me")

    assert [p.name for p in fr.frame_paths(images)] == [
        "frame_000000.png",
        "frame_000005.png",
        "frame_000011.png",
    ]


def test_frame_paths_on_a_missing_directory_is_empty(tmp_path):
    assert fr.frame_paths(tmp_path / "nope") == []


def test_write_frames_clears_a_previous_longer_run(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), [0, 5, 11])
    fr.write_frames(images, _frames(2), [0, 5])

    assert [p.name for p in fr.frame_paths(images)] == ["frame_000000.png", "frame_000005.png"]


def test_write_frames_rejects_a_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="against"):
        fr.write_frames(tmp_path / "images", _frames(3), [0])


def test_write_frames_png_compression_is_a_kwarg(tmp_path):
    def size(sub, **kw):
        return fr.write_frames(tmp_path / sub / "images", _frames(1, 64, 64), [0], **kw)[0].stat().st_size

    assert size("l0", png_compression=0) > size("default") > size("l9", png_compression=9)


def test_read_frames_names_an_undecodable_frame(tmp_path):
    """A zero-byte PNG must raise by path, not as an opaque cvtColor assert."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_000000.png").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="frame_000000.png"):
        fr.read_frames(images)


def test_frame_paths_selects_by_source_index_in_requested_order(tmp_path):
    for idx in (4, 10, 7):
        (tmp_path / f"frame_{idx:06d}.png").touch()

    paths = fr.frame_paths(tmp_path, [10, 4])

    assert [p.name for p in paths] == ["frame_000010.png", "frame_000004.png"]


def test_frame_paths_raises_on_an_index_the_directory_lacks(tmp_path):
    (tmp_path / "frame_000004.png").touch()

    with pytest.raises(KeyError, match=r"frame_paths: frame_idx \[5\]"):
        fr.frame_paths(tmp_path, [4, 5])


def test_write_frames_names_an_empty_selection(tmp_path):
    with pytest.raises(ValueError, match="no frames selected"):
        fr.write_frames(tmp_path / "images", [], [])
