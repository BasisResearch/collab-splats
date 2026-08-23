"""
Unit tests for the instantsfm / instantsfm_nodepth eval conditions (pure helpers, no GPU/colmap).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "scripts"))
import eval as eval_gt  # noqa: E402


class _FakeImage:
    """
    pycolmap.Image stand-in: name + cam_from_world().matrix() -> (3,4) with tx = index.
    """

    def __init__(self, name, idx):
        self.name = name
        self._idx = idx

    def cam_from_world(self):
        m = np.hstack([np.eye(3), [[self._idx], [0.0], [0.0]]])
        return type("Rigid", (), {"matrix": lambda _self: m})()


class _FakeRecon:
    def __init__(self, names):
        self.images = {i + 1: _FakeImage(n, i) for i, n in enumerate(names)}


def _write_pngs(image_dir, n):
    image_dir.mkdir()
    paths = []
    for i in range(n):
        p = image_dir / f"{i:06d}.png"
        Image.fromarray(np.full((8, 6, 3), i * 10, dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def test_validate_condition_accepts_instantsfm():
    for cond in ["instantsfm", "instantsfm_nodepth"]:
        eval_gt._validate_condition(cond)


def test_run_condition_rejects_submap_size_for_instantsfm(tmp_path):
    with pytest.raises(ValueError, match="submap_size"):
        eval_gt._run_condition("instantsfm", tmp_path, tmp_path / "out", submap_size=50)


def test_run_instantsfm_nodepth_stages_symlinks_and_orders_by_name(tmp_path, monkeypatch):
    paths = _write_pngs(tmp_path / "imgs", 3)
    seen = {}

    class _Creator:
        def __init__(self, **kw):
            seen["kw"] = kw

        def reconstruct(self, data_dir):
            seen["data_dir"] = Path(data_dir)
            # Registration order differs from name order — the result must follow names
            return _FakeRecon(["000002.png", "000000.png", "000001.png"])

    monkeypatch.setattr(eval_gt, "InstantSfMCreator", _Creator)
    monkeypatch.setattr(eval_gt, "generate_vda_depth", lambda *a, **k: pytest.fail("VDA must not run"))

    out = tmp_path / "out"
    extr, n_loops = eval_gt._run_condition("instantsfm_nodepth", tmp_path / "imgs", out)

    # use_depths False, data_dir = output_dir, images staged as symlinks to the sources
    assert seen["kw"] == {"use_depths": False}
    assert seen["data_dir"] == out
    staged = sorted((out / "images").iterdir())
    assert [p.name for p in staged] == [p.name for p in paths]
    assert all(p.is_symlink() and p.resolve() == src.resolve() for p, src in zip(staged, paths))

    # (N,4,4) float32 w2c, tx encodes registration idx: sorted by name -> 000000 first (idx 1)
    assert n_loops is None
    assert extr.shape == (3, 4, 4) and extr.dtype == np.float32
    assert extr[:, 0, 3].tolist() == [1.0, 2.0, 0.0]
    assert np.allclose(extr[:, 3], [0, 0, 0, 1])


def test_run_instantsfm_depth_generates_vda_then_reconstructs(tmp_path, monkeypatch):
    paths = _write_pngs(tmp_path / "imgs", 2)
    calls = []

    def _fake_vda(frames, fps, out_dir, names):
        calls.append(("vda", frames.shape, frames.dtype, fps, Path(out_dir), list(names)))

    class _Creator:
        def __init__(self, **kw):
            calls.append(("creator", kw))

        def reconstruct(self, data_dir):
            return _FakeRecon([p.name for p in paths])

    monkeypatch.setattr(eval_gt, "generate_vda_depth", _fake_vda)
    monkeypatch.setattr(eval_gt, "InstantSfMCreator", _Creator)

    out = tmp_path / "out"
    extr = eval_gt._run_instantsfm("instantsfm", tmp_path / "imgs", out)

    # VDA ran first over uint8 RGB frames keyed by the staged names, then the depth-aware creator
    assert calls[0] == ("vda", (2, 8, 6, 3), np.uint8, 1.0, out, ["000000.png", "000001.png"])
    assert calls[1] == ("creator", {"use_depths": True})
    assert extr.shape == (2, 4, 4)


def test_run_instantsfm_partial_registration_names_missing(tmp_path, monkeypatch):
    _write_pngs(tmp_path / "imgs", 3)

    class _Creator:
        def __init__(self, **kw):
            pass

        def reconstruct(self, data_dir):
            return _FakeRecon(["000000.png", "000002.png"])

    monkeypatch.setattr(eval_gt, "InstantSfMCreator", _Creator)
    with pytest.raises(RuntimeError, match="000001.png"):
        eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs", tmp_path / "out")


def test_run_instantsfm_changed_name_set_drops_sift_db(tmp_path, monkeypatch):
    _write_pngs(tmp_path / "imgs", 2)
    out = tmp_path / "out"
    db = out / "colmap" / "instantsfm.db"
    db.parent.mkdir(parents=True)
    db.touch()

    class _Creator:
        def __init__(self, **kw):
            pass

        def reconstruct(self, data_dir):
            return _FakeRecon(["000000.png", "000001.png"])

    monkeypatch.setattr(eval_gt, "InstantSfMCreator", _Creator)

    # Stale staged set (different names) -> DB dropped; re-run with the same set keeps it
    (out / "images").mkdir()
    (out / "images" / "999999.png").touch()
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs", out)
    assert not db.exists()
    db.touch()
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs", out)
    assert db.exists()


def test_run_instantsfm_same_names_different_source_drops_caches(tmp_path, monkeypatch):
    # Eval names are positional (000000.png...), so two sources with the same frame
    # count collide on names — link targets must key the SIFT DB and VDA depth cache
    _write_pngs(tmp_path / "imgs_a", 2)
    _write_pngs(tmp_path / "imgs_b", 2)
    out = tmp_path / "out"

    class _Creator:
        def __init__(self, **kw):
            pass

        def reconstruct(self, data_dir):
            return _FakeRecon(["000000.png", "000001.png"])

    monkeypatch.setattr(eval_gt, "InstantSfMCreator", _Creator)

    # First run stages imgs_a; plant a DB and depth cache as if it completed
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs_a", out)
    db = out / "colmap" / "instantsfm.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.touch()
    depth_dir = out / "depth_vda"
    depth_dir.mkdir()

    # Same names, different source -> both caches dropped
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs_b", out)
    assert not db.exists()
    assert not depth_dir.exists()

    # Same source again -> caches kept
    db.parent.mkdir(parents=True, exist_ok=True)
    db.touch()
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs_b", out)
    assert db.exists()
