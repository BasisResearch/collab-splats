import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MOD = REPO_ROOT / "docs/source/tutorials/notebook_utils.py"


def _load():
    spec = importlib.util.spec_from_file_location("notebook_utils", MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_set_notebook_backend_is_callable(monkeypatch):
    nu = _load()
    called = {}
    import pyvista as pv
    monkeypatch.setattr(pv, "set_jupyter_backend", lambda b: called.setdefault("b", b))
    nu.set_notebook_backend()
    assert called["b"] in ("static", "trame")
