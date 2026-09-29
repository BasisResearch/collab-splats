"""
Shared pycolmap SIFT database helpers (sfm/sift_db.py).
"""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.sfm import sift_db

########################################################
########## SIFT database build: pycolmap calls #########
########################################################


_STEPS = ("extract_features", "match_exhaustive", "match_sequential", "match_vocabtree")


def _build(image_path, database_path, **overrides):
    """
    build_sift_database with every setting explicit; overrides replace the exhaustive defaults.
    """
    kwargs = {"pairing": "exhaustive", "overlap": 10, "num_retrieved": 20, "vocab_tree": None, "num_threads": 8}
    kwargs.update(overrides)
    sift_db.build_sift_database(image_path, database_path, **kwargs)


def _record_sift_calls(monkeypatch, *, has_cuda=False, device="cpu"):
    """
    Patch the pycolmap SIFT steps and the device probes; returns {step: (args, kwargs)} in call order.
    """
    calls = {}
    for step in _STEPS:
        monkeypatch.setattr(pycolmap, step, lambda *args, _step=step, **kw: calls.update({_step: (args, kw)}))
    monkeypatch.setattr(pycolmap, "has_cuda", has_cuda)
    monkeypatch.setattr(sift_db, "get_device", lambda: device)
    return calls


def test_extraction_shares_one_simple_radial_camera(monkeypatch, tmp_path):
    calls = _record_sift_calls(monkeypatch)
    _build(tmp_path / "img", tmp_path / "db")

    args, kw = calls["extract_features"]
    assert args == (tmp_path / "db", tmp_path / "img")
    assert kw["camera_mode"] == pycolmap.CameraMode.SINGLE
    assert kw["reader_options"].camera_model == "SIMPLE_RADIAL"


@pytest.mark.parametrize(
    "pairing, matcher, options",
    [
        ("exhaustive", "match_exhaustive", {}),
        ("sequential", "match_sequential", {"overlap": 5, "quadratic_overlap": False, "loop_detection": False}),
        (
            "sequential+retrieval",
            "match_sequential",
            {
                "overlap": 5,
                "quadratic_overlap": False,
                "loop_detection": True,
                "loop_detection_num_images": 7,
                "vocab_tree_path": Path("VT"),
            },
        ),
        ("retrieval", "match_vocabtree", {"num_images": 7, "vocab_tree_path": Path("VT")}),
    ],
)
def test_matcher_and_pairing_options_per_pairing(monkeypatch, tmp_path, pairing, matcher, options):
    calls = _record_sift_calls(monkeypatch)
    _build(tmp_path, tmp_path / "db", pairing=pairing, overlap=5, num_retrieved=7, vocab_tree="VT")

    # Extraction first, then exactly the one matcher the pairing names
    assert list(calls) == ["extract_features", matcher]
    args, kw = calls[matcher]
    assert args == (tmp_path / "db",)
    assert {k: getattr(kw["pairing_options"], k) for k in options} == options


@pytest.mark.parametrize(
    "has_cuda, torch_device, expected",
    [(True, "cuda", pycolmap.Device.cuda), (False, "cuda", pycolmap.Device.cpu), (True, "cpu", pycolmap.Device.cpu)],
)
def test_gpu_needs_a_cuda_wheel_and_a_cuda_host(monkeypatch, tmp_path, has_cuda, torch_device, expected):
    """
    The CPU wheel on a GPU host must stay on the CPU: torch seeing a GPU is not enough.
    """
    calls = _record_sift_calls(monkeypatch, has_cuda=has_cuda, device=torch_device)
    _build(tmp_path, tmp_path / "db")
    assert [kw["device"] for _, kw in calls.values()] == [expected, expected]


@pytest.mark.parametrize("pairing", ["retrieval", "sequential+retrieval"])
def test_cpu_path_caps_every_thread_pool(monkeypatch, tmp_path, pairing):
    """
    Uncapped vocab-tree pairing on a 96-core host overruns faiss's OpenBLAS and segfaults.
    """
    calls = _record_sift_calls(monkeypatch)
    _build(tmp_path, tmp_path / "db", pairing=pairing, vocab_tree="VT", num_threads=5)

    (_, extract), (_, match) = calls.values()
    assert extract["extraction_options"].num_threads == 5
    assert match["matching_options"].num_threads == 5
    assert match["pairing_options"].num_threads == 5


def test_gpu_path_caps_only_pairing_threads(monkeypatch, tmp_path):
    """
    faiss pairing search is CPU work on either device; extraction and matching keep colmap's defaults.
    """
    calls = _record_sift_calls(monkeypatch, has_cuda=True, device="cuda")
    _build(tmp_path, tmp_path / "db", pairing="sequential+retrieval", vocab_tree="VT")

    (_, extract), (_, match) = calls.values()
    assert extract["extraction_options"].num_threads == -1
    assert match["matching_options"].num_threads == -1
    assert match["pairing_options"].num_threads == 8


@pytest.mark.parametrize("step", ["extract_features", "match_exhaustive"])
@pytest.mark.parametrize("error", [RuntimeError, ValueError])
def test_failure_unlinks_the_partial_database(monkeypatch, tmp_path, step, error):
    _record_sift_calls(monkeypatch)
    db = tmp_path / "db"
    db.write_bytes(b"partial")

    def fail(*args, **kw):
        raise error("boom")

    monkeypatch.setattr(pycolmap, step, fail)
    with pytest.raises(RuntimeError, match="SIFT database build failed on cpu"):
        _build(tmp_path, db)
    assert not db.exists()


@pytest.mark.parametrize("pairing, match", [("retrieval", "vocab_tree"), ("spatial", "pairing")])
def test_bad_pairing_is_refused_before_any_pycolmap_call(monkeypatch, tmp_path, pairing, match):
    calls = _record_sift_calls(monkeypatch)
    with pytest.raises(ValueError, match=match):
        _build(tmp_path, tmp_path / "db", pairing=pairing)
    assert calls == {}


########################################################
########## SIFT database validity ######################
########################################################


# The image set a cached DB is checked against; reuse is gated on it matching exactly
_DB_NAMES = ["000001.png", "000002.png"]


def test_database_holds_is_false_for_a_missing_file(tmp_path):
    """
    No DB at all is the first-run case, not a corruption.
    """
    assert sift_db._database_holds(tmp_path / "nope.db", _DB_NAMES) is False


def test_database_holds_is_false_for_a_non_database_file(tmp_path):
    """
    A crashed colmap can leave a truncated/garbage file; pycolmap raises rather than returning.
    """
    db = tmp_path / "garbage.db"
    db.write_bytes(b"not a sqlite file")

    assert sift_db._database_holds(db, _DB_NAMES) is False


def test_database_holds_is_false_for_an_empty_database(tmp_path):
    """
    An OOM-killed extractor leaves a well-formed but empty DB — an existence check would
    cache-hit on it and feed ReadColmapDatabase zero tracks.
    """
    db = tmp_path / "empty.db"
    pycolmap.Database.open(str(db)).close()

    assert sift_db._database_holds(db, _DB_NAMES) is False


def _sift_db(path, names=_DB_NAMES, *, verified_pair):
    """
    Minimal colmap SIFT database: one camera, one image per name with keypoints, optional verified pair.
    """
    db = pycolmap.Database.open(str(path))
    cam = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
    db.write_camera(cam, True)
    keypoints = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    for image_id, name in enumerate(names, start=1):
        db.write_image(pycolmap.Image(name=name, camera_id=1, image_id=image_id), True)
        db.write_keypoints(image_id, keypoints)

    # Verified pairs land in two_view_geometries — what exhaustive_matcher writes, and the
    # only evidence that matching ran to completion rather than crashing after extraction
    if verified_pair:
        tvg = pycolmap.TwoViewGeometry()
        tvg.config = 2
        tvg.inlier_matches = np.array([[0, 0], [1, 1]], dtype=np.uint32)
        db.write_two_view_geometry(1, 2, tvg)
    db.close()


def test_database_holds_is_true_for_a_complete_database(tmp_path):
    """
    Keypoints plus a verified pair is the cache-hit case — a full re-extraction is skipped.
    """
    db = tmp_path / "complete.db"
    _sift_db(db, verified_pair=True)

    assert sift_db._database_holds(db, _DB_NAMES) is True


def test_database_holds_is_false_when_matching_never_finished(tmp_path):
    """
    Extraction finished, matching crashed: keypoints but zero verified pairs, which
    ReadColmapDatabase turns into an empty-tracks failure much later.
    """
    db = tmp_path / "unmatched.db"
    _sift_db(db, verified_pair=False)

    assert sift_db._database_holds(db, _DB_NAMES) is False


def test_sift_database_is_reused_for_the_same_image_set(tmp_path):
    """
    Same images in a different order is the same set — reordering must not force a re-extract.
    """
    db = tmp_path / "reused.db"
    _sift_db(db, verified_pair=True)

    assert sift_db._database_holds(db, list(reversed(_DB_NAMES))) is True


def test_sift_database_is_rebuilt_when_the_image_set_changed(tmp_path):
    """
    Nothing stages a per-run image copy any more, so the DB's own images table is the only
    record of which selection its features came from.
    """
    db = tmp_path / "stale.db"
    _sift_db(db, verified_pair=True)

    assert sift_db._database_holds(db, _DB_NAMES + ["000003.png"]) is False


########################################################
########## Ensure: reuse or rebuild ###################
########################################################


ENSURE_KW = dict(pairing="exhaustive", overlap=10, num_retrieved=20, vocab_tree=None, num_threads=1)


def _ensure(tmp_path, monkeypatch, **overrides):
    """
    Run ensure_sift_database with the build stubbed to write a valid DB; return the build calls.
    """
    db = tmp_path / "colmap.db"
    calls = []

    def fake_build(image_dir, db_path, **kw):
        calls.append(kw)
        _sift_db(db_path, verified_pair=True)

    monkeypatch.setattr(sift_db, "build_sift_database", fake_build)
    sift_db.ensure_sift_database(tmp_path, db, _DB_NAMES, **{**ENSURE_KW, **overrides})
    return calls


def test_ensure_builds_then_reuses(tmp_path, monkeypatch):
    first = _ensure(tmp_path, monkeypatch)
    second = _ensure(tmp_path, monkeypatch)
    assert len(first) == 1 and second == []


def test_ensure_rebuilds_when_pairing_changes(tmp_path, monkeypatch):
    _ensure(tmp_path, monkeypatch)
    calls = _ensure(tmp_path, monkeypatch, pairing="sequential")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "pairing, knob, value, rebuilds",
    [
        ("sequential", "overlap", 3, True),
        ("sequential", "num_retrieved", 3, False),
        ("retrieval", "num_retrieved", 3, True),
        ("retrieval", "overlap", 3, False),
        ("exhaustive", "overlap", 3, False),
        ("exhaustive", "num_retrieved", 3, False),
        ("sequential+retrieval", "overlap", 3, True),
        ("sequential+retrieval", "num_retrieved", 3, True),
    ],
)
def test_ensure_params_hold_only_live_knobs(tmp_path, monkeypatch, pairing, knob, value, rebuilds):
    _ensure(tmp_path, monkeypatch, pairing=pairing)
    calls = _ensure(tmp_path, monkeypatch, pairing=pairing, **{knob: value})
    assert (len(calls) == 1) is rebuilds


def test_ensure_stores_params_in_the_db_not_a_sidecar(tmp_path, monkeypatch):
    _ensure(tmp_path, monkeypatch, pairing="sequential", overlap=3)

    conn = sqlite3.connect(tmp_path / "colmap.db")
    with closing(conn):
        rows = conn.execute("SELECT json FROM collab_params").fetchall()
    assert [json.loads(row[0]) for row in rows] == [{"pairing": "sequential", "overlap": 3}]
    assert not (tmp_path / "colmap.db.json").exists()


def test_ensure_rebuilds_a_legacy_db_without_params_table(tmp_path, monkeypatch):
    _sift_db(tmp_path / "colmap.db", verified_pair=True)
    calls = _ensure(tmp_path, monkeypatch)
    assert len(calls) == 1


########################################################
########## Vocab tree ##################################
########################################################


def test_fetch_vocab_tree_uses_a_cached_file_with_the_pinned_hash(tmp_path, monkeypatch):
    cached = tmp_path / sift_db.VOCAB_TREE_NAME
    cached.write_bytes(b"x")
    monkeypatch.setattr(sift_db, "VOCAB_TREE_SHA256", hashlib.sha256(b"x").hexdigest())
    monkeypatch.setattr(sift_db.urllib.request, "urlretrieve", lambda *a, **k: pytest.fail("downloaded"))
    assert sift_db.fetch_vocab_tree(tmp_path) == cached


def test_fetch_vocab_tree_rejects_a_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(sift_db.urllib.request, "urlretrieve", lambda url, dst: Path(dst).write_bytes(b"bad"))
    with pytest.raises(RuntimeError, match="sha256"):
        sift_db.fetch_vocab_tree(tmp_path)
    assert not (tmp_path / sift_db.VOCAB_TREE_NAME).exists()
