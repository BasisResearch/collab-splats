import numpy as np
import pytest

from collab_splats.preproc.frame_store import FrameStore


def _frames_and_records(n=3, h=8, w=12):
    # Distinct per-frame content so index mixups are caught
    frames = [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)]
    records = [{"frame_idx": i * 10, "blur_score": float(i)} for i in range(n)]
    return frames, records


def test_create_open_roundtrip(tmp_path):
    frames, records = _frames_and_records()
    prov = {"video_path": "v.mp4", "video_mtime": 1.0, "method": "uniform", "max_frames": None}
    FrameStore.create(tmp_path / "frames.zarr", frames, records, provenance=prov)
    store = FrameStore.open(tmp_path / "frames.zarr")
    assert len(store) == 3
    np.testing.assert_array_equal(store.image(1), frames[1])
    assert store.record(1)["frame_idx"] == 10
    np.testing.assert_array_equal(store.frame_indices(), np.array([0, 10, 20]))


def test_image_by_frame_idx(tmp_path):
    frames, records = _frames_and_records()
    FrameStore.create(tmp_path / "s.zarr", frames, records, provenance={"video_path": "v"})
    store = FrameStore.open(tmp_path / "s.zarr")
    np.testing.assert_array_equal(store.image_by_frame_idx(20), frames[2])
    with pytest.raises(KeyError):
        store.image_by_frame_idx(999)


def test_images_subset(tmp_path):
    frames, records = _frames_and_records()
    FrameStore.create(tmp_path / "s.zarr", frames, records, provenance={"video_path": "v"})
    store = FrameStore.open(tmp_path / "s.zarr")
    out = store.images([0, 2])
    assert out.shape == (2, 8, 12, 3)
    np.testing.assert_array_equal(out[1], frames[2])


def test_export_writes_jpgs(tmp_path):
    frames, records = _frames_and_records()
    FrameStore.create(tmp_path / "s.zarr", frames, records, provenance={"video_path": "v"})
    store = FrameStore.open(tmp_path / "s.zarr")
    out = store.export(tmp_path / "exported")
    assert len(out) == 3
    assert all(p.exists() for p in out)
    import cv2

    back = cv2.cvtColor(cv2.imread(str(out[2])), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(back, frames[2])


def test_has_frame_idx(tmp_path):
    """
    Public membership check — viz needs it, and it replaces a private reach-in.
    """
    store = FrameStore.create(
        tmp_path / "frames.zarr",
        [np.zeros((4, 4, 3), np.uint8)],
        [{"frame_idx": 7, "blur_score": 1.0}],
        provenance={"video_path": "v.mp4"},
    )

    assert store.has_frame_idx(7)
    assert not store.has_frame_idx(8)


def test_is_stale_is_gone(tmp_path):
    """
    Reuse is by existence; a staleness check was unreachable and is deleted.
    """
    store = FrameStore.create(
        tmp_path / "frames.zarr",
        [np.zeros((4, 4, 3), np.uint8)],
        [{"frame_idx": 0}],
        provenance={"video_path": "v.mp4"},
    )

    assert not hasattr(store, "is_stale")
    assert not hasattr(store, "records")
