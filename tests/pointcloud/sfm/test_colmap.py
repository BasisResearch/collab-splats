"""
ColmapCreator: pycolmap SIFT DB + pycolmap incremental mapping, all heavy legs mocked.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from collab_splats.pointcloud.sfm import colmap as colmap_mod
from collab_splats.pointcloud.sfm import sift_db
from collab_splats.pointcloud.sfm.colmap import ColmapCreator
from tests.pointcloud._stubs import make_recon, make_scene

NAMES = ["frame_000000.png", "frame_000009.png", "frame_000030.png"]


@pytest.fixture
def mocked(monkeypatch):
    """
    Stub the DB build, vocab fetch and mapper; records their calls.
    """
    calls = {"build": [], "map": []}

    def build(image_path, db_path, **kw):
        calls["build"].append(kw)
        Path(db_path).touch()

    def mapping(db, image_dir, out, options):
        calls["map"].append({"db": db, "image_dir": image_dir, "out": out, "options": options})
        # Largest model at key 0, so picking the last key fails the size check
        return {0: make_recon(NAMES), 1: make_recon(NAMES[:1])}

    monkeypatch.setattr(sift_db, "build_sift_database", build)
    monkeypatch.setattr(colmap_mod, "fetch_vocab_tree", lambda: Path("/vt.bin"))
    monkeypatch.setattr(sift_db, "_database_holds", lambda *a: False)
    monkeypatch.setattr(colmap_mod.pycolmap, "incremental_mapping", mapping)
    return calls


def test_map_returns_the_largest_model_and_drops_the_mapper_scratch(tmp_path, mocked):
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    recon = ColmapCreator()._map(images_dir, data_dir, NAMES)

    assert recon.num_reg_images() == len(NAMES)
    assert not (data_dir / "colmap" / "mapper").exists()


def test_reconstruct_passes_pairing_params_and_thread_cap(tmp_path, mocked):
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    ColmapCreator(pairing="retrieval", overlap=3, num_retrieved=4, num_threads=2)._map(images_dir, data_dir, NAMES)
    assert mocked["build"] == [
        {"pairing": "retrieval", "overlap": 3, "num_retrieved": 4, "vocab_tree": Path("/vt.bin"), "num_threads": 2}
    ]
    assert mocked["map"][0]["options"] == {"num_threads": 2}
    assert mocked["map"][0]["db"] == str(data_dir / "colmap" / "colmap.db")


def test_sequential_pairing_never_fetches_the_vocab_tree(tmp_path, mocked, monkeypatch):
    monkeypatch.setattr(colmap_mod, "fetch_vocab_tree", lambda: pytest.fail("fetched"))
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    ColmapCreator(pairing="sequential")._map(images_dir, data_dir, NAMES)
    assert mocked["build"][0]["vocab_tree"] is None


def test_no_model_raises(tmp_path, mocked, monkeypatch):
    monkeypatch.setattr(colmap_mod.pycolmap, "incremental_mapping", lambda *a, **k: {})
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    with pytest.raises(RuntimeError, match="no model"):
        ColmapCreator()._map(images_dir, data_dir, NAMES)


def test_reused_database_does_not_fetch_the_vocab_tree(tmp_path, mocked, monkeypatch):
    """
    A reused SIFT database must never resolve vocab_tree — no network call on reuse.
    """
    monkeypatch.setattr(sift_db, "_database_holds", lambda *a: True)
    monkeypatch.setattr(colmap_mod, "fetch_vocab_tree", lambda: pytest.fail("fetched"))
    data_dir, images_dir = make_scene(tmp_path, NAMES)

    # Pre-seed a DB whose params row ensure_sift_database's reuse check will accept
    db_path = data_dir / "colmap" / "colmap.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    params = json.dumps({"pairing": "retrieval", "num_retrieved": 20}, sort_keys=True)
    conn = sqlite3.connect(db_path)
    with closing(conn), conn:
        conn.execute("CREATE TABLE collab_params (json TEXT)")
        conn.execute("INSERT INTO collab_params VALUES (?)", (params,))

    ColmapCreator(pairing="retrieval")._map(images_dir, data_dir, NAMES)
    assert mocked["build"] == []
