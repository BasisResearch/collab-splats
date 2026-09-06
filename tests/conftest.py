"""pytest configuration and session-level fixtures.

Shims and environment fixes that must run before any test module is imported.
"""

# ── sys.path: make top-level project dir importable (for evals.* etc.) ───────
import pathlib as _pathlib
import sys as _sys

_project_root = _pathlib.Path(__file__).parents[1]
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

# ── pkg_resources.packaging shim ─────────────────────────────────────────────
# setuptools>=71 removed pkg_resources.packaging as a submodule. maskclip_onnx
# (and some other old packages) do `from pkg_resources import packaging`, which
# raises ImportError on modern setuptools. Inject the standalone `packaging`
# library into the pkg_resources namespace so those imports resolve cleanly.
# This is safe: pkg_resources previously just re-exported packaging, so the API
# is identical.
import importlib
import sys

import pkg_resources

if not hasattr(pkg_resources, "packaging"):
    import packaging as _packaging

    pkg_resources.packaging = _packaging
    # Also make `from pkg_resources import packaging` work via the module dict
    sys.modules["pkg_resources.packaging"] = _packaging
    for _submod in ("version", "specifiers", "requirements", "markers", "utils"):
        _full = f"packaging.{_submod}"
        if _full in sys.modules:
            sys.modules[f"pkg_resources.packaging.{_submod}"] = sys.modules[_full]
        else:
            try:
                sys.modules[f"pkg_resources.packaging.{_submod}"] = importlib.import_module(_full)
            except ImportError:
                pass

# ── shared video fixture ─────────────────────────────────────────────────────
# Lives here rather than in tests/preproc: tests/pointcloud decodes it too.
import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    """Synthesize a 60-frame 320x240 mp4: static noise texture + moving square.

    Noise gives LK flow corners to track; the moving square creates motion.
    """
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.default_rng(0)
    noise = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    for i in range(60):
        frame = noise.copy()
        x = 10 + i * 4
        cv2.rectangle(frame, (x, 60), (x + 60, 140), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return str(path)
