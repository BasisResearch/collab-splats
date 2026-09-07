"""
Directory-backed keyframe store: images/frame_NNNNNN.png + frames.json.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from collab_splats.preproc import frames as fr

# scripts/migrate_frames_zarr.py is a CLI, not an installed module, and the name `scripts`
# is contested: `tests/evals/test_datasets.py` puts `<repo>/evals` on sys.path[0], which makes
# `evals/scripts/` win `import scripts` for the rest of the session, and MapAnything's
# `uniception` ships its own top-level `scripts` in site-packages. Neither is winnable by path
# order, so load the file directly — no package, no collision.
_MIGRATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_frames_zarr.py"
_spec = importlib.util.spec_from_file_location("_migrate_frames_zarr", _MIGRATE_PATH)
_migrate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_migrate_mod)
migrate_scene = _migrate_mod.migrate_scene


def _frames(n=3, h=8, w=12):
    """
    n deterministic RGB frames, each a different flat color.
    """
    return [np.full((h, w, 3), i * 40 + 5, dtype=np.uint8) for i in range(n)]


def _records(idxs):
    return [{"frame_idx": int(i), "blur_score": float(i) * 1.5} for i in idxs]


def test_frame_idx_from_path_reads_the_padded_stem():
    assert fr.frame_idx_from_path("images/frame_000042.png") == 42


def test_write_then_read_round_trips_rgb(tmp_path):
    images = tmp_path / "images"
    written = fr.write_frames(images, _frames(3), _records([0, 5, 11]), {"method": "uniform"})

    assert [p.name for p in written] == ["frame_000000.png", "frame_000005.png", "frame_000011.png"]

    out = fr.read_frames(images)
    assert out.shape == (3, 8, 12, 3)
    assert out.dtype == np.uint8

    # PNG is lossless and the store is RGB at both boundaries
    np.testing.assert_array_equal(out, np.stack(_frames(3)))


def test_read_frames_selects_by_frame_idx_not_position(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})

    out = fr.read_frames(images, idxs=[11, 0])
    np.testing.assert_array_equal(out[0], _frames(3)[2])
    np.testing.assert_array_equal(out[1], _frames(3)[0])


def test_read_frames_raises_on_an_index_the_directory_does_not_hold(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})

    with pytest.raises(KeyError, match="7"):
        fr.read_frames(images, idxs=[7])


def test_frame_paths_is_sorted_and_extension_filtered(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([11, 0, 5]), {})
    (images / "notes.txt").write_text("ignore me")

    assert [p.name for p in fr.frame_paths(images)] == [
        "frame_000000.png",
        "frame_000005.png",
        "frame_000011.png",
    ]


def test_frame_paths_on_a_missing_directory_is_empty(tmp_path):
    assert fr.frame_paths(tmp_path / "nope") == []


def test_manifest_carries_records_and_provenance(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(2), _records([0, 5]), {"method": "fps", "fps": 2.0})

    manifest = fr.read_manifest(images)
    assert manifest["schema_version"] == 2
    assert manifest["provenance"] == {"method": "fps", "fps": 2.0}
    assert [r["frame_idx"] for r in manifest["frames"]] == [0, 5]

    # frames.json sits beside images/, not inside it
    assert (tmp_path / "frames.json").exists()
    assert not (images / "frames.json").exists()


def test_manifest_converts_nan_to_null(tmp_path):
    images = tmp_path / "images"
    records = [{"frame_idx": 0, "blur_score": float("nan")}]
    fr.write_frames(images, _frames(1), records, {})

    raw = (tmp_path / "frames.json").read_text()
    assert "NaN" not in raw
    assert json.loads(raw)["frames"][0]["blur_score"] is None


def test_write_frames_clears_a_previous_longer_run(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})
    fr.write_frames(images, _frames(2), _records([0, 5]), {})

    assert [p.name for p in fr.frame_paths(images)] == ["frame_000000.png", "frame_000005.png"]


def test_write_frames_rejects_records_without_frame_idx(tmp_path):
    with pytest.raises(ValueError, match="frame_idx"):
        fr.write_frames(tmp_path / "images", _frames(1), [{"blur_score": 1.0}], {})


def test_write_frames_rejects_a_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="against"):
        fr.write_frames(tmp_path / "images", _frames(3), _records([0]), {})


def test_read_manifest_names_the_migration_script_when_absent(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="migrate_frames_zarr"):
        fr.read_manifest(tmp_path / "images")


def test_migrate_converts_a_zarr_store_without_decoding(tmp_path):
    """
    The migration script reads frames.zarr and writes images/ + frames.json.
    """
    # Build a minimal frames.zarr by hand — the same shape the retired zarr store wrote
    scene = tmp_path / "scene"
    scene.mkdir()
    imgs = np.stack(_frames(3))
    store = zarr.open(str(scene / "frames.zarr"), mode="w")
    store.create_array("images", data=imgs, chunks=(1, *imgs.shape[1:]))
    store.create_array("frame_idx", data=np.array([0, 5, 11]))
    store.create_array("blur_score", data=np.array([1.0, 2.0, 3.0]))
    store.attrs["record_keys"] = ["blur_score", "frame_idx"]
    store.attrs["provenance"] = {"method": "fps", "fps": 2.0}
    store.attrs["schema_version"] = 1

    migrate_scene(scene)

    out = fr.read_frames(scene / "images")
    np.testing.assert_array_equal(out, imgs)

    manifest = fr.read_manifest(scene / "images")
    assert manifest["provenance"] == {"method": "fps", "fps": 2.0}
    assert [r["frame_idx"] for r in manifest["frames"]] == [0, 5, 11]
    assert manifest["frames"][1]["blur_score"] == 2.0
