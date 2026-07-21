import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MOD = REPO_ROOT / "docs/source/tutorials/notebook_utils.py"


def _load():
    spec = importlib.util.spec_from_file_location("notebook_utils", MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_keyframe_paths_returns_sorted_frame_paths(tmp_path):
    from collab_splats.preproc import FrameStore

    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    records = [{"frame_idx": i} for i in (5, 1, 3)]
    zpath = tmp_path / "frames.zarr"
    FrameStore.create(zpath, frames, records, provenance={"video_path": "x"})

    nu = _load()
    paths = nu.load_keyframe_paths(zpath, tmp_path / "export")

    assert len(paths) == 3
    assert all(p.exists() and p.suffix == ".jpg" for p in paths)
    assert paths == sorted(paths)  # deterministic order


def test_set_notebook_backend_is_callable(monkeypatch):
    nu = _load()
    called = {}
    import pyvista as pv
    monkeypatch.setattr(pv, "set_jupyter_backend", lambda b: called.setdefault("b", b))
    nu.set_notebook_backend()
    assert called["b"] in ("static", "trame")
