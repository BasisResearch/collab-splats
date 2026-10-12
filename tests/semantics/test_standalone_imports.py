"""
Semantics-only install contract: heavy reconstruction deps stay out of the semantics import graph.

- fresh-process checks: importing a module must not load the named heavy dependency
- AST walk: every third-party module-level import maps to base or the semantics extra
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# (module imported in a fresh process, top-level module that must stay unloaded)
LAZY = [
    ("collab_splats.geometry", "bae"),
    ("collab_splats.preproc", "pycolmap"),
    ("collab_splats.semantics.segmentation", "mobile_sam"),
    ("collab_splats.geometry.transforms", "open3d"),
    ("collab_splats.semantics.lifting", "open3d"),
]


@pytest.mark.parametrize(("module", "heavy"), LAZY, ids=lambda v: str(v))
def test_fresh_import_skips_heavy_dep(module, heavy):
    code = f"import sys, {module}; print({heavy!r} in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", f"import {module} loaded {heavy}"


# Heavy reconstruction deps no semantics module may load
HEAVY = ("bae", "pypose", "pycolmap", "mobile_sam", "open3d", "gsplat", "collab_data")


def test_every_semantics_module_skips_heavy_deps():
    code = (
        "import pkgutil, sys, collab_splats.semantics as s\n"
        "for m in pkgutil.walk_packages(s.__path__, s.__name__ + '.'):\n"
        "    __import__(m.name)\n"
        f"print(sorted(set({HEAVY!r}) & set(sys.modules)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "[]", proc.stdout


ROOT = Path(__file__).resolve().parents[2]

# Import name -> distribution name where they differ (dash/underscore/case are normalized)
IMPORT_TO_DIST = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}


def _normalize(dist: str) -> str:
    """
    PEP 503 normalized distribution name.
    """
    return re.sub(r"[-_.]+", "-", dist).lower()


def _module_path(module: str) -> Path | None:
    """
    Source file for a dotted collab_splats module, or None if it is not one.
    """
    base = ROOT.joinpath(*module.split("."))
    if (base / "__init__.py").exists():
        return base / "__init__.py"

    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")

    return None


def _module_level_imports(path: Path) -> list[str]:
    """
    Absolute names imported at module level.

    - skips function bodies (lazy imports) and `if TYPE_CHECKING:` blocks
    - `from a.b import c` yields both a.b and a.b.c (c may be a submodule)
    """
    found = []

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                walk(node.orelse)
                continue

            if isinstance(node, ast.Import):
                found.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                found.append(node.module)
                found.extend(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, (ast.ClassDef, ast.If, ast.Try, ast.With)):
                walk(getattr(node, "body", []))
                walk(getattr(node, "orelse", []))
                walk(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    walk(handler.body)

    walk(ast.parse(path.read_text(encoding="utf-8")).body)

    return found


def _reachable_third_party() -> dict[str, str]:
    """
    Third-party top-level import name -> first collab_splats module importing it, from semantics.
    """
    semantics_dir = ROOT / "collab_splats" / "semantics"
    pending = [
        ".".join(p.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
        for p in sorted(semantics_dir.rglob("*.py"))
    ]
    seen = set()
    third_party = {}

    while pending:
        module = pending.pop()
        if module in seen:
            continue

        seen.add(module)
        for name in _module_level_imports(_module_path(module)):
            parts = name.split(".")

            # Importing a.b.c runs a, a.b and a.b.c
            if parts[0] == "collab_splats":
                for i in range(1, len(parts) + 1):
                    prefix = ".".join(parts[:i])
                    if _module_path(prefix) is not None:
                        pending.append(prefix)
            elif parts[0] not in sys.stdlib_module_names:
                third_party.setdefault(parts[0], module)

    return third_party


def _allowed_dists() -> set[str]:
    """
    Normalized distribution names in base dependencies plus the semantics extra.
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = (
        project["dependencies"] + project["optional-dependencies"]["semantics"]
    )

    return {_normalize(re.match(r"[A-Za-z0-9._-]+", r).group()) for r in requirements}


def test_walk_reaches_known_imports():
    reachable = _reachable_third_party()

    # Sanity: the walk follows collab_splats edges into preproc and geometry
    assert {"torch", "transformers", "nltk", "av", "scipy"} <= set(reachable)
    assert "open3d" not in reachable and "bae" not in reachable


def test_semantics_imports_stay_in_semantics_extra():
    allowed = _allowed_dists()
    outside = {
        name: module
        for name, module in _reachable_third_party().items()
        if _normalize(IMPORT_TO_DIST.get(name, name)) not in allowed
    }

    assert not outside, f"semantics imports outside base + [semantics]: {outside}"
