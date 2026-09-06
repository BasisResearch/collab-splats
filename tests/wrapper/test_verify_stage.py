"""Verify-stage wiring: leaf-stage registration + config default + matcher/image plumbing."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import yaml

from collab_splats.localization.extractors import LocalMatcher
from collab_splats.wrapper.reconstructor import _STAGE_DEPS, _STAGE_ORDER, LEAF_STAGES, Reconstructor

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_verify_is_a_leaf_stage():
    """verify is registered, depends only on pointcloud, and is re-runnable on its own."""
    assert "verify" in _STAGE_ORDER
    assert _STAGE_DEPS["verify"] == ["pointcloud"]
    assert "verify" in LEAF_STAGES


def test_geometric_verification_defaults_off():
    """Ships off until the first measured report (spec: Validation gates the default)."""
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    assert cfg["pointcloud"]["geometric_verification"] is False


def test_database_db_not_pushed():
    """database.db is a rebuildable local artifact — excluded from GCS pushes."""
    from collab_splats.remote.sources import PUSH_EXCLUDES

    assert "/*/colmap/database.db" in PUSH_EXCLUDES


def _stub_verify_call(tmp_path, monkeypatch, matcher, load_fn=None):
    """Run Reconstructor.verify() with all heavy collaborators stubbed; return captured kwargs."""
    # 2-frame reconstruction stub — verify() only reads .images[id].name for the stem check
    recon = SimpleNamespace(
        images={1: SimpleNamespace(name="frame_000003"), 2: SimpleNamespace(name="frame_000007")}
    )
    r = Reconstructor.__new__(Reconstructor)
    # backend_dir / images_dir are config-derived properties — stub via config
    r.config = {
        "output_path": str(tmp_path),
        "pointcloud": {"backend": "vggtx"},
        "localization": {"matcher": "stub"},
    }
    r._stage_output_exists = lambda stage: False
    r._resolve_result = lambda: SimpleNamespace(reconstruction=recon)
    rebuilds = []
    r.build_localization_db = lambda **kw: rebuilds.append(kw)

    # Collaborators: feature cache read, matcher resolution, images/ read, verification entry
    if load_fn is None:
        load_fn = lambda path, name: (  # noqa: E731
            ["f0", "f1"],
            ["frame_000003.jpg", "frame_000007.jpg"],
            (4, 4),
        )
    monkeypatch.setattr("collab_splats.localization.localizer.load_localization_db", load_fn)
    # verify() imports LocalMatcher from the extractors module inline — patch it with a
    # factory returning the stub (no isinstance dispatch remains in verify()).
    monkeypatch.setattr("collab_splats.localization.extractors.LocalMatcher", lambda name: matcher)
    reads = []

    def _scene_frames(images_dir):
        reads.append(Path(images_dir))
        return np.stack([np.full((4, 4, 3), fi, dtype=np.uint8) for fi in (3, 7)])

    monkeypatch.setattr("collab_splats.wrapper.reconstructor._scene_frames", _scene_frames)
    captured = {}
    monkeypatch.setattr(
        "collab_splats.geometry.verification.verify_reconstruction",
        lambda **kw: captured.update(kw),
    )
    r.verify()
    return captured, reads, rebuilds


def test_verify_passes_scene_images_to_pairwise_matcher(tmp_path, monkeypatch):
    """For a LocalMatcher, verify() hands over the scene's images/ frames as images=."""
    matcher = MagicMock(spec=LocalMatcher)
    matcher.has_stable_indices = True
    captured, reads, _ = _stub_verify_call(tmp_path, monkeypatch, matcher)
    assert captured["matcher"] is matcher
    # The scene's own images/ is the source, read once
    assert reads == [Path(tmp_path) / "images"]
    # Frames arrive in filename order — the order the cache was extracted in
    images = captured["images"]
    assert len(images) == 2
    assert [int(images[i][0, 0, 0]) for i in range(2)] == [3, 7]


def test_verify_hands_images_for_any_matcher(tmp_path, monkeypatch):
    """images= is always the images/ stack now, matcher kind notwithstanding.

    (Pre-retirement, descriptor matchers got images=None; verify_reconstruction's
    descriptor branch ignores `images`, so the unconditional handoff is harmless.)"""
    matcher = SimpleNamespace()  # duck-typed descriptor stub, not a LocalMatcher
    captured, reads, _ = _stub_verify_call(tmp_path, monkeypatch, matcher)
    assert captured["matcher"] is matcher
    assert captured["images"] is not None and len(captured["images"]) == 2
    assert reads == [Path(tmp_path) / "images"]


def test_verify_rebuilds_db_when_loma_payload_missing(tmp_path, monkeypatch):
    """Split-capable matcher + payload-less cache: one rebuild + reload before verification."""
    matcher = MagicMock(spec=LocalMatcher)
    matcher.has_stable_indices = True
    matcher._split_loma_forward = True  # assignment is allowed on spec mocks; get-after-set works
    stale = SimpleNamespace(keypoints_normalized=None)
    fresh = SimpleNamespace(keypoints_normalized=np.zeros((1, 2), np.float32))
    loads = []

    def _load(path, name):
        loads.append(name)
        feats = [stale, stale] if len(loads) == 1 else [fresh, fresh]
        return feats, ["frame_000003.jpg", "frame_000007.jpg"], (4, 4)

    captured, _, rebuilds = _stub_verify_call(tmp_path, monkeypatch, matcher, load_fn=_load)
    assert rebuilds == [{}, {"overwrite": True}]  # up-front build, then the payload rebuild
    assert len(loads) == 2  # reloaded after the rebuild
    assert captured["features"] == [fresh, fresh]
