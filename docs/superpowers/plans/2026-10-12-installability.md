# Installability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `bash setup.sh` in the runtime container finishes with no errors, false warnings or churn, and `collab_splats[semantics]` installs and imports on its own (CPU, no CUDA, no bae).

**Architecture:**
- split pyproject base deps into a light core plus `semantics`, `reconstruction`, `remote` and `instantsfm` extras
- defer four heavy imports (bae, pycolmap, mobile_sam, open3d) so `collab_splats.semantics` never loads them
- fix setup.sh and the Dockerfile around the new lock

**Tech Stack:** uv (lock, extras, `tool.uv` overrides/sources/dependency-metadata), PEP 562 module `__getattr__`, pytest, bash, Docker.

**Spec:** [docs/superpowers/specs/2026-10-12-installability-design.md](../specs/2026-10-12-installability-design.md)

---

## Ground rules (read before Task 0)

- Work directly on `clean/final` in `/workspace/collab-splats`:
  - every `uv sync` / `setup.sh` here re-points the shared venv's editable install at this checkout, so a worktree would hijack the venv
  - other sessions share the git index: commit with `git commit --only <paths>`, never `git add -A`, never `--amend`
  - `docs/superpowers/` is gitignored: `git add -f` for spec/plan/CHANGELOG files
- Python: `/opt/venv/reconstruction/bin/python` (py3.11). `$SP` = `/tmp/claude-0/-workspace-collab-splats/96ff9778-1e39-4ed8-9440-2a86dad9271d/scratchpad`.
- Comments: one plain line per block comment (user feedback overrides the CLAUDE.md header+bullet rule). Docstrings: CLAUDE.md shape (`"""` on own lines, summary line, bullets, `Args:` / `Returns:` / `Raises:`).
- After every `uv sync` or `setup.sh`: `/opt/venv/reconstruction/bin/python -c "import gsplat; print(gsplat.__version__)"` must print `1.5.3` (shared-venv downgrade trap).
- pytest: never pipe into `tail`/`head` (eats the exit code); long runs go in tmux with the exit code written to a file.
- Redirected shell output can silently write 0 bytes on `/workspace`: use tool-native `-o` flags and check file sizes.
- Lint before each commit: `ruff check <files> && ruff format --check <files>` and `isort --check-only <files>` for touched Python.

## File map

| file | change |
|---|---|
| `CLAUDE.md` | In-Flight entry (Task 0), removed on landing (Task 12) |
| `pyproject.toml` | Talk2DINO deps (Task 1); extras split, instantsfm locked, platform-gated torch (Task 7) |
| `uv.lock` | re-locked (Tasks 1, 7) |
| `collab_splats/geometry/__init__.py` | BundleAdjustment(Config) lazy (Task 2) |
| `collab_splats/preproc/__init__.py` | calibrate_camera / undistort_frames lazy (Task 3) |
| `collab_splats/semantics/segmentation/mobile_sam.py` | mobile_sam import inside `_segment_auto` (Task 4) |
| `collab_splats/geometry/transforms.py` | open3d import inside `fit_dominant_plane` (Task 5) |
| `tests/conftest.py`, `tests/semantics/features/test_maskclip.py` | ports from `37f7af2d` (Task 6) |
| `tests/semantics/features/test_talk2dino_deps.py` | new (Task 1) |
| `tests/semantics/test_standalone_imports.py` | new: fresh-process checks (Tasks 2-5) + AST allowlist (Task 7) |
| `setup.sh` | uv from PATH, InstantSfM block gone, NLTK_DATA, creator smoke (Task 8) |
| `Dockerfile` | jq, pinned uv + rclone, NLTK_DATA (Task 9) |
| `README.md`, `docs/source/api/pointcloud.rst` | install docs (Task 10) |
| `docs/superpowers/CHANGELOG.md` | entry on landing (Task 12) |

---

### Task 0: In-Flight entry

**Files:** Modify `CLAUDE.md` (the `## In-Flight Work` list)

- [ ] **Step 1: Add the entry** as the last bullet of `## In-Flight Work`:

```markdown
- **installability** — container setup fixes + standalone `collab_splats[semantics]` extra (Talk2DINO `clip` regression fixed first) ([spec](docs/superpowers/specs/2026-10-12-installability-design.md) · [plan](docs/superpowers/plans/2026-10-12-installability.md))
```

- [ ] **Step 2: Commit**

```bash
git commit --only CLAUDE.md -m "docs: installability in-flight entry"
```

---

### Task 1: Talk2DINO remote-code deps (regression from `47633e39`)

Talk2DINOv3's HF remote code imports `clip`, `einops`, `omegaconf`; transformers' `check_imports` refuses the model if any is missing. They go into base now; Task 7 moves them into `semantics`.

**Files:**
- Create: `tests/semantics/features/test_talk2dino_deps.py`
- Modify: `pyproject.toml` (`dependencies`), `uv.lock`

- [ ] **Step 1: Write the failing test**

```python
"""
Talk2DINOv3 remote-code imports: transformers' check_imports refuses the model without them.
"""

import importlib.util

import pytest

from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor


@pytest.mark.parametrize("name", ["clip", "einops", "omegaconf"])
def test_remote_code_import_installed(name):
    assert importlib.util.find_spec(name) is not None, f"{name} missing: Talk2DINOv3 cannot load"


def test_clip_is_openai_clip():
    import clip

    # PyPI `clip` is an unrelated package without load/tokenize
    assert callable(clip.load) and callable(clip.tokenize)


@pytest.mark.gpu
def test_talk2dino_constructs_on_cpu():
    extractor = Talk2DinoExtractor(device="cpu")

    assert extractor is not None
```

The `import clip` inside the test body is deliberate: a module-level import would error the whole file at collection instead of failing one test. Add `# noqa: PLC0415` if ruff flags it.

- [ ] **Step 2: Run it, expect FAIL**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/features/test_talk2dino_deps.py -v`
Expected: `test_remote_code_import_installed[clip]` FAIL (`clip missing`) and `test_clip_is_openai_clip` FAIL (`ModuleNotFoundError: No module named 'clip'`); einops/omegaconf may already pass via mapanything.

- [ ] **Step 3: Add the deps** to `pyproject.toml` `dependencies`, right after `"timm>=0.9,<2.0",`:

```toml
    # Talk2DINOv3 HF remote code imports clip, einops, omegaconf; transformers refuses the model without them
    # clip as a PEP 508 direct URL: git-dependency consumers never read [tool.uv.sources], PyPI `clip` is unrelated
    "clip @ git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
    "einops",
    "omegaconf",
```

- [ ] **Step 4: Re-lock and sync**

Sync through setup.sh, not a bare `uv sync`: until Task 8, setup.sh reinstalls InstantSfM, which a bare sync prunes.

```bash
cd /workspace/collab-splats
uv lock
git diff --stat uv.lock
bash setup.sh > "$SP/setup_task1.log" 2>&1; echo "exit=$?"
ls -la "$SP/setup_task1.log"
grep -n "clip\|einops\|omegaconf\|WARN" "$SP/setup_task1.log"
/opt/venv/reconstruction/bin/python -c "import gsplat; print(gsplat.__version__)"
```

Expected: `exit=0`, `+ clip==1.0` (and einops/omegaconf if not already present) in the log, gsplat `1.5.3`. The `VGGTOmegaCreator` `[WARN]` is still expected (fixed in Task 8).

- [ ] **Step 5: Run the tests, expect PASS** (including the gpu-marked load, opted in explicitly)

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/features/test_talk2dino_deps.py -v -m "gpu or not gpu"
```

Expected: 5 passed. `test_talk2dino_constructs_on_cpu` reads the cached `lorebianchi98/Talk2DINOv3-ViTB` snapshot under `/workspace/models/hub`.

- [ ] **Step 6: Commit**

```bash
git commit --only pyproject.toml uv.lock tests/semantics/features/test_talk2dino_deps.py \
  -m "fix(semantics): restore Talk2DINOv3 remote-code deps clip, einops, omegaconf

47633e39 dropped them as unused; transformers' check_imports refuses
Talk2DINOv3 without them, so Talk2DinoExtractor raised ImportError: clip."
```

---

### Task 2: Lazy BundleAdjustment in `geometry/__init__`

**Files:**
- Create: `tests/semantics/test_standalone_imports.py`
- Modify: `collab_splats/geometry/__init__.py`

- [ ] **Step 1: Write the failing test** (`tests/semantics/test_standalone_imports.py`)

```python
"""
Semantics-only install contract: heavy reconstruction deps stay out of the semantics import graph.

- fresh-process checks: importing a module must not load the named heavy dependency
"""

import subprocess
import sys

import pytest

# (module imported in a fresh process, top-level module that must stay unloaded)
LAZY = [
    ("collab_splats.geometry", "bae"),
]


@pytest.mark.parametrize(("module", "heavy"), LAZY, ids=lambda v: str(v))
def test_fresh_import_skips_heavy_dep(module, heavy):
    code = f"import sys, {module}; print({heavy!r} in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", f"import {module} loaded {heavy}"
```

- [ ] **Step 2: Run, expect FAIL**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_standalone_imports.py -v`
Expected: FAIL `import collab_splats.geometry loaded bae`.

- [ ] **Step 3: Implement** — `collab_splats/geometry/__init__.py`: delete the eager `from collab_splats.geometry.bundle_adjustment import (...)` block and extend `__getattr__`:

```python
from collab_splats.geometry.projection import reprojection_error


def __getattr__(name: str) -> type:
    """
    Lazy BundleAdjustment and LoopClosure classes.

    - BundleAdjustment(Config): bundle_adjustment imports bae, a CUDA extension semantics never needs
    - LoopClosure(Config): breaks the cycle loop_closure.wrapper -> collab_splats.pointcloud -> geometry.transforms

    Args:
        name: attribute requested from the package.

    Returns:
        The requested class.

    Raises:
        AttributeError: any other name.
    """
    if name in ("BundleAdjustment", "BundleAdjustmentConfig"):
        # Import on first access: keeps bae out of `import collab_splats.geometry`
        from collab_splats.geometry import bundle_adjustment

        return getattr(bundle_adjustment, name)

    if name in ("LoopClosure", "LoopClosureConfig"):
        # Import on first access to break the load-time cycle
        from collab_splats.geometry import loop_closure

        return getattr(loop_closure, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

`__all__` is unchanged. `from collab_splats.geometry import BundleAdjustment` (reconstructor) keeps working through `__getattr__`.

- [ ] **Step 4: Run, expect PASS**, plus the existing geometry/reconstructor/cycle tests

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_standalone_imports.py tests/test_import_cycles.py tests/geometry tests/reconstructor/test_refine_stage.py tests/test_docstring_contract.py -q
```

Expected: all pass (`test_refine_stage.py` patches `collab_splats.reconstructor.BundleAdjustment.refine`; it still resolves).

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/geometry/__init__.py tests/semantics/test_standalone_imports.py \
  -m "refactor(geometry): load BundleAdjustment lazily so the package import skips bae"
```

---

### Task 3: Lazy undistort in `preproc/__init__`

**Files:** Modify `collab_splats/preproc/__init__.py`, `tests/semantics/test_standalone_imports.py`

- [ ] **Step 1: Add the failing row** to `LAZY`:

```python
    ("collab_splats.preproc", "pycolmap"),
```

- [ ] **Step 2: Run, expect FAIL** `import collab_splats.preproc loaded pycolmap`.

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_standalone_imports.py -v`

- [ ] **Step 3: Implement** — in `collab_splats/preproc/__init__.py` delete `from collab_splats.preproc.undistort import calibrate_camera, undistort_frames` and add after the last eager import (`video`):

```python
def __getattr__(name: str) -> object:
    """
    Lazy calibrate_camera and undistort_frames.

    - undistort imports pycolmap, which semantics-only installs do not carry

    Args:
        name: attribute requested from the package.

    Returns:
        The requested function.

    Raises:
        AttributeError: any other name.
    """
    if name in ("calibrate_camera", "undistort_frames"):
        # Import on first access: keeps pycolmap out of `import collab_splats.preproc`
        from collab_splats.preproc import undistort

        return getattr(undistort, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

`__all__` is unchanged.

- [ ] **Step 4: Run, expect PASS**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_standalone_imports.py tests/preproc tests/test_docstring_contract.py -q
```

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/preproc/__init__.py tests/semantics/test_standalone_imports.py \
  -m "refactor(preproc): load undistort lazily so the package import skips pycolmap"
```

---

### Task 4: `mobile_sam` imported inside the auto strategy

**Files:** Modify `collab_splats/semantics/segmentation/mobile_sam.py`, `tests/semantics/test_standalone_imports.py`

- [ ] **Step 1: Add the failing row**:

```python
    ("collab_splats.semantics.segmentation", "mobile_sam"),
```

- [ ] **Step 2: Run, expect FAIL** `import collab_splats.semantics.segmentation loaded mobile_sam`.

- [ ] **Step 3: Implement** — delete line 16 `from mobile_sam import SamAutomaticMaskGenerator` (and the blank line that separated its import group). In `_segment_auto`, before `mask_generator = ...`:

```python
        # Imported here: mobile_sam is a git-only dep the semantics extra does not carry
        try:
            from mobile_sam import SamAutomaticMaskGenerator  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "mobilesamv2 strategy 'auto' needs mobile_sam: "
                "pip install 'git+https://github.com/ChaoningZhang/MobileSAM.git'"
            ) from e

        mask_generator = SamAutomaticMaskGenerator(model=self.seg_model)
```

Add to `_segment_auto`'s docstring, after `Returns:`:

```
        Raises:
            ImportError: mobile_sam is not installed.
```

(`_segment_auto` is private; if the docstring contract rejects a `Raises:` block on private defs, put `- raises ImportError when mobile_sam is missing` as a bullet instead.)

- [ ] **Step 4: Run, expect PASS**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_standalone_imports.py tests/semantics/test_segmentation.py tests/test_docstring_contract.py tests/test_import_style.py -q
```

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/semantics/segmentation/mobile_sam.py tests/semantics/test_standalone_imports.py \
  -m "refactor(semantics): import mobile_sam inside the auto strategy"
```

---

### Task 5: open3d imported inside `fit_dominant_plane`

`semantics.lifting` → `geometry.projection` → `geometry.transforms`, which imports open3d at top. open3d is ~400 MB and has no wheels past Python 3.12; only `fit_dominant_plane` uses it.

**Files:** Modify `collab_splats/geometry/transforms.py`, `tests/semantics/test_standalone_imports.py`

- [ ] **Step 1: Add the failing rows**, including the whole-package check that closes the contract:

```python
    ("collab_splats.geometry.transforms", "open3d"),
    ("collab_splats.semantics.lifting", "open3d"),
```

and a second test that imports every semantics module at once:

```python
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
```

- [ ] **Step 2: Run, expect FAIL** — the two open3d rows fail; the whole-package test prints `['open3d']`.

- [ ] **Step 3: Implement** — `transforms.py`: delete `import open3d as o3d` (line 16). In `fit_dominant_plane`, first statement of the body:

```python
    # Imported here: open3d ships no wheels past Python 3.12 and only this fit needs it
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("fit_dominant_plane needs open3d: pip install open3d") from e

    # RANSAC plane normal · x + d_norm = 0, with a unit normal
    pcd = o3d.geometry.PointCloud()
```

Docstring gains, after `Returns:`:

```
    Raises:
        ImportError: open3d is not installed.
```

- [ ] **Step 4: Run, expect PASS**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_standalone_imports.py tests/geometry tests/test_docstring_contract.py tests/test_import_style.py -q
```

Also `grep -rn "transforms.o3d\|geometry.transforms import.*o3d" collab_splats tests` → no hits (nobody reached `o3d` through the module).

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/geometry/transforms.py tests/semantics/test_standalone_imports.py \
  -m "refactor(geometry): import open3d inside fit_dominant_plane"
```

---

### Task 6: Test-suite ports from `37f7af2d`

**Files:** Modify `tests/conftest.py:24-42`, `tests/semantics/features/test_maskclip.py`

- [ ] **Step 1: conftest** — replace `import pkg_resources  # noqa: E402` and the `if not hasattr(...)` header with:

```python
# Best-effort: a semantics-only env may carry setuptools >= 81, which has no pkg_resources
try:
    import pkg_resources  # noqa: E402
except ImportError:
    pkg_resources = None

if pkg_resources is not None and not hasattr(pkg_resources, "packaging"):
```

The body under the `if` is unchanged.

- [ ] **Step 2: test_maskclip.py** — whole file:

```python
"""
maskclip_onnx imports when installed; a semantics-only env skips it.
"""

import pytest

maskclip_onnx = pytest.importorskip("maskclip_onnx")


def test_maskclip_onnx_importable():
    assert callable(maskclip_onnx.clip.load)
```

- [ ] **Step 3: Run**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/features/test_maskclip.py tests/semantics/features/test_talk2dino_deps.py -v
```

Expected: pass (not skipped — the container has maskclip_onnx).

- [ ] **Step 4: Commit**

```bash
git commit --only tests/conftest.py tests/semantics/features/test_maskclip.py \
  -m "test: tolerate missing pkg_resources and maskclip_onnx (semantics-only env)"
```

---

### Task 7: pyproject extras split + AST allowlist test

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `tests/semantics/test_standalone_imports.py`

- [ ] **Step 1: Baseline the lock contents**

```bash
cd /workspace/collab-splats
uv export --frozen --all-extras --no-hashes --no-emit-project -o "$SP/export_before.txt"
ls -la "$SP/export_before.txt"    # must be non-zero
```

- [ ] **Step 2: Write the failing AST test** — append to `tests/semantics/test_standalone_imports.py` (add `import ast`, `import re`, `import tomllib`, `from pathlib import Path` to the stdlib group, and a docstring bullet `- AST walk: every third-party module-level import maps to base or the semantics extra`):

```python
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

    walk(ast.parse(path.read_text()).body)

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
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = project["dependencies"] + project["optional-dependencies"]["semantics"]

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
```

- [ ] **Step 3: Run, expect FAIL**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_standalone_imports.py -v`
Expected: `test_walk_reaches_known_imports` PASS (Tasks 2-5 landed); `test_semantics_imports_stay_in_semantics_extra` FAIL with `KeyError: 'semantics'`.

- [ ] **Step 4: Rewrite `[project] dependencies` and `[project.optional-dependencies]`** in `pyproject.toml`. Base becomes:

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pillow",
    "pyyaml",
    "tqdm",
    "matplotlib",
    "mergedeep",
    "python-dotenv",
    "zarr>=2.16",
    "huggingface_hub",
]
```

New extras at the top of `[project.optional-dependencies]` (existing `feedforward`, `hloc`, `gpu`, `dashboard`, `docs`, `dev` stay):

```toml
# 2D features + segmentation: CPU or GPU, no CUDA build, no bae (collab-data mesh_semantics installs this)
semantics = [
    # Plain torch: in-repo syncs get cu121 from [tool.uv] overrides; consumers keep their own build
    "torch",
    "torchvision",
    "transformers>=4.30,<5.0",
    "timm>=0.9,<2.0",
    "opencv-python",
    "scikit-learn",
    "scikit-image",
    "kornia",
    "safetensors",
    # av — PyAV, the decode path in preproc/video.py (semantics.store reaches preproc)
    "av>=17.0",
    # WordNet lemmas for the ocr_lens decode vocabulary
    "nltk>=3.8",
    # Parquet reader for score_ocr_heads' mini-imagenet backgrounds
    "pyarrow",
    # Pinned: unpinned resolves an old release that needs pretrainedmodels
    "segmentation-models-pytorch==0.5.0",
    # Talk2DINOv3 HF remote code imports clip, einops, omegaconf; transformers refuses the model without them
    # clip as a PEP 508 direct URL: git-dependency consumers never read [tool.uv.sources], PyPI `clip` is unrelated
    "clip @ git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
    "einops",
    "omegaconf",
]

# Full pipeline: pointcloud, BA, splats, mesh, localization; needs the CUDA toolchain (setup.sh)
reconstruction = [
    "collab_splats[semantics]",
    # setuptools<70.0 required: maskclip_onnx uses `from pkg_resources import packaging`,
    # which setuptools 70+ removed. See [build-system] comment.
    "setuptools<70.0",
    # wheel + ninja must be IN the venv (not build-extras): bae/gsplat build with
    # no-build-isolation, which uses the project venv as the build env. Without wheel,
    # setuptools<70 has no bdist_wheel command. ninja gives fast parallel CUDA compiles.
    "wheel",
    "ninja",
    # cu121 wheels have no darwin build; darwin resolves plain torch through semantics
    "torch==2.5.1+cu121; sys_platform != 'darwin'",
    "torchvision==0.20.1+cu121; sys_platform != 'darwin'",
    # ftfy: maskclip_onnx imports it without declaring it
    "ftfy",
    "seaborn",
    "ultralytics",
    "pyvista[jupyter]",
    # viser — websocket 3D viewer for live scene visualization (collab_splats/viewer.py)
    "viser>=1.0",
    # Texture bake: nvdiffrast rasterizes the UV atlas and each view's occlusion depth,
    # meshlib decimates and patches holes.
    # nvdiffrast ships no wheel; at this rev it compiles its CUDA extension at INSTALL (AOT),
    # so it needs nvcc at build time only (see setup.sh) and no build isolation (below).
    "meshlib==3.1.4.297",
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast@253ac4fcea7de5f396371124af597e6cc957bfae",
    "trimesh",
    "pydantic==2.10.6",
    # pycolmap-cuda12, pinned at 4.1.1
    # - GPU SIFT wheel: replaces the colmap CLI (docs/superpowers/specs/2026-09-26-pycolmap-cuda-docker-design.md)
    # - not 4.2: shared-focal two-view estimator, ~14x verification CPU for single-camera scenes
    "pycolmap-cuda12==4.1.1",
    # BasisResearch fork, `basis` branch: batched extract + match() + match_batch() (spec 2026-10-03-vismatch-feature-matching)
    "vismatch @ git+https://github.com/BasisResearch/vismatch@865d91a52ff23f60096188e0e0a748911768535d",
    "mobile_sam",
    # torchpack: implicit dep of mobilesamv2 (loaded via torch.hub for SAM segmentation).
    # mobilesamv2/efficientvit/models/nn/drop.py imports torchpack.distributed at module level,
    # so it must be installed before any import of mobilesamv2.
    "torchpack",
    # gsplat: upstream, built from source against the cu121 toolchain (see [tool.uv.sources])
    "gsplat",
    # fused-ssim: CUDA fast path for gsplat.losses.ssim_loss (else a slow torch fallback); needs nvcc
    "fused-ssim",
    "maskclip_onnx",
    <the pypose<0.9 entry with its 5-line comment, the nvidia-cudss-cu12==0.6.0.5 entry with its
     3-line comment, and the bae entry with its 4-line comment — copied byte-for-byte from the old
     base list; they are long and must not be paraphrased>
    # open3d: fit_dominant_plane (imported lazily by geometry.transforms)
    "open3d",
]

# reconstruct CLI and dashboard GCS access: private collab-data (path dep at /workspace/collab-data)
remote = [
    "collab_data @ file:///workspace/collab-data",
]

# InstantSfM sfm backend + VDA runtime dep; instantsfm's own requires-dist is blanked below
instantsfm = [
    "instantsfm",
    "pyceres==2.3",
    # scikit-sparse builds against libsuitesparse-dev (Dockerfile apt lists)
    "scikit-sparse==0.4.15",
    # easydict: Video-Depth-Anything's only runtime dep missing from the lock (dpt_temporal.py)
    "easydict==1.13",
]
```

The one `<...>` line is a cut-and-paste instruction for three long verbatim comment blocks, not literal text. Nothing in the old base list may be left unplaced: check with `git diff pyproject.toml` that every removed `-` dep line reappears as a `+` line. The old base entries `transformers`, `timm`, `opencv-python`, `scikit-learn`, `kornia`, `av`, `nltk`, `pyarrow`, `segmentation-models-pytorch`, and Task 1's `clip`/`einops`/`omegaconf` go to `semantics` (above). `collab_data` goes to `remote`. The torch marker is `!= 'darwin'` (spec §1: gated to linux/win32, complementary darwin entries).

In `feedforward`, delete the 3-line comment beginning `# InstantSfM backend deps (instantsfm, pyceres, ...` (now false).

- [ ] **Step 5: `[tool.uv]` edits**

`override-dependencies`: replace the two torch lines with complementary pairs (an override replaces the requirement outright, so a darwin-false marker alone would drop torch on macOS):

```toml
    "torch==2.5.1+cu121; sys_platform != 'darwin'",
    "torch==2.5.1; sys_platform == 'darwin'",
    "torchvision==0.20.1+cu121; sys_platform != 'darwin'",
    "torchvision==0.20.1; sys_platform == 'darwin'",
```

`[tool.uv.sources]`: replace the torch/torchvision lines and add instantsfm:

```toml
torch = [{ index = "pytorch-cu121", marker = "sys_platform != 'darwin'" }]
torchvision = [{ index = "pytorch-cu121", marker = "sys_platform != 'darwin'" }]
# InstantSfM pinned commit; upstream pins numpy==1.26.4 and more, so its requires-dist is blanked (dependency-metadata)
instantsfm = { git = "https://github.com/cre185/InstantSfM", rev = "d3e599e1a42b4c5a806a84d9f383e1005d25f61b" }
```

New `[[tool.uv.dependency-metadata]]` block next to the others (the lock equivalent of setup.sh's old `--no-deps`):

```toml
[[tool.uv.dependency-metadata]]
name = "instantsfm"
version = "0.3.0"
requires-dist = []
```

Update the comment above the dependency-metadata blocks with one line: `# - instantsfm: upstream pins numpy 1.26 / opencv / scipy / viser exactly; our lock owns those`.

- [ ] **Step 6: Re-lock and compare**

```bash
cd /workspace/collab-splats
uv lock
uv export --frozen --all-extras --no-hashes --no-emit-project -o "$SP/export_after.txt"
ls -la "$SP/export_after.txt"
rtk proxy diff "$SP/export_before.txt" "$SP/export_after.txt"
```

Then the spec's lock gate: `uv sync --locked --all-extras --dry-run` must list only the InstantSfM four as changes.

Expected diff: additions `instantsfm @ git+...d3e599e1...`, `pyceres==2.3`, `scikit-sparse==0.4.15`, `easydict==1.13`; torch/torchvision lines may gain markers; NO version change on any other package. Any other change → stop and investigate. If `uv lock` fails resolving darwin, add `environments = ["sys_platform == 'linux'", "sys_platform == 'darwin'"]` to `[tool.uv]` and re-run; if it still fails, report back before going further.

- [ ] **Step 7: Run the AST test, expect PASS**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_standalone_imports.py tests/test_env.py -v`
Expected: all pass (`test_env.py` reads `tool.uv.sources.gsplat`, unchanged).

- [ ] **Step 8: Sync the venv** — setup.sh still has its InstantSfM block until Task 8; that is fine (it reinstalls the same pin).

```bash
cd /workspace/collab-splats && bash setup.sh > "$SP/setup_task7.log" 2>&1; echo "exit=$?"
ls -la "$SP/setup_task7.log"
/opt/venv/reconstruction/bin/python -c "import gsplat, instantsfm, pyceres, sksparse, easydict, clip; print(gsplat.__version__)"
```

Expected: `exit=0`, gsplat `1.5.3`, imports OK.

- [ ] **Step 9: Commit**

```bash
git commit --only pyproject.toml uv.lock tests/semantics/test_standalone_imports.py \
  -m "build: split deps into semantics / reconstruction / remote / instantsfm extras

Base keeps a light cross-platform core; collab_splats[semantics] installs
without CUDA, bae or the private collab-data. InstantSfM enters the lock
with blanked requires-dist (was a --no-deps install outside it). Torch
cu121 pins and the pytorch index are gated off darwin."
```

---

### Task 8: setup.sh

**Files:** Modify `setup.sh`

- [ ] **Step 1: uv from PATH** — after `export UV_PROJECT_ENVIRONMENT="$VENV"` (line 12):

```bash

# uv from PATH: the Docker image puts it in /root/.local/bin, other installs elsewhere
command -v uv >/dev/null || { echo "setup.sh: uv not on PATH — install it first (README §1)." >&2; exit 1; }
```

Then replace every `/root/.local/bin/uv` with `uv` (lines 59, 105, 111). Check: `grep -n "/root/.local/bin/uv" setup.sh` → no hits.

- [ ] **Step 2: Sync-failure hint** — add after `cuda_toolkit_help()`'s closing `}`:

```bash

# uv sync failed: CUDA toolkit recipe when nvcc is missing, plus the InstantSfM system lib
sync_failed() {
    [ "$HAVE_NVCC" = 0 ] && cuda_toolkit_help
    echo "setup.sh: uv sync failed. scikit-sparse (instantsfm extra) also needs libsuitesparse-dev: apt-get install -y libsuitesparse-dev" >&2
    exit 1
}
```

Replace both `|| { [ "$HAVE_NVCC" = 0 ] && cuda_toolkit_help; exit 1; }` with `|| sync_failed`.

- [ ] **Step 3: WordNet to NLTK_DATA** — replace lines 114-116 with:

```bash
# WordNet corpus for the ocr_lens vocabulary, on the persistent models volume; best-effort
export NLTK_DATA="${NLTK_DATA:-/workspace/models/nltk_data}"
"$PYTHON" -c "import nltk, os; nltk.download('wordnet', download_dir=os.environ['NLTK_DATA'], quiet=True)" \
    || echo "WARN: WordNet download failed — run: python -c \"import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA')\""
```

(Deliberate deviation from spec §3's `-m nltk.downloader -d`: `nltk.download(download_dir=...)` has the same explicit target and avoids runpy's `found in sys.modules` RuntimeWarning from the startup log.)

- [ ] **Step 4: Delete the InstantSfM block** — lines 133-145 (`# --- InstantSfM backend ...` through the closing `} || echo "WARN: InstantSfM ..."`). The `instantsfm` extra now installs it in `uv sync --all-extras`.

- [ ] **Step 5: Creator smoke** — replace the `try:` block (lines 179-183) with:

```python
try:
    from collab_splats.pointcloud import get_creator
    for name in ("vggtx", "mapanything", "vggt_omega", "loger"):
        get_creator(name)
    print("[OK] creators resolve: vggtx, mapanything, vggt_omega, loger")
except Exception as e:  # OSError: missing X11/GL; RuntimeError: vggt touches CUDA at import
    print(f"[WARN] creator import deferred to runtime (no GPU/system lib in build stage): {e}")
```

- [ ] **Step 6: Run it**

```bash
bash -n setup.sh && echo syntax-ok
cd /workspace/collab-splats && bash setup.sh > "$SP/setup_task8.log" 2>&1; echo "exit=$?"
ls -la "$SP/setup_task8.log"
grep -n "WARN\|Uninstalled\|instantsfm\|creators resolve" "$SP/setup_task8.log"
ls /workspace/models/nltk_data/corpora | grep wordnet
```

Expected: `exit=0`; only WARN line is `setup.sh: WARN no nvcc at /usr/local/cuda/bin/nvcc` (runtime pod has no nvcc by design); no `Uninstalled` / `- instantsfm` churn; `[OK] creators resolve: ...`; `wordnet.zip` (or `wordnet/`) present. Run it a second time and confirm `uv sync` reports no changes (`Audited` / no `Installed` lines).

- [ ] **Step 7: Commit**

```bash
git commit --only setup.sh -m "fix(setup): uv from PATH, locked InstantSfM, WordNet on the models volume, real creator smoke"
```

---

### Task 9: Dockerfile

**Files:** Modify `Dockerfile` (builder uv line 45; runtime apt list ~88-95, rclone line 97, uv line 100, ENV block ~111-118, bashrc block ~141-149)

- [ ] **Step 1: Pin uv in both stages** — replace both `RUN curl -LsSf https://astral.sh/uv/install.sh | sh` with:

```dockerfile
# uv pinned: setup.sh and uv.lock are tested against this release
RUN curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh
```

- [ ] **Step 2: jq in the runtime apt list** — change the line `wget curl unzip xz-utils git vim htop tmux less \` to `wget curl unzip xz-utils git vim htop tmux less jq \`. (collab-data's `setup_local_rclone.sh` reads the service-account json with jq.)

- [ ] **Step 3: Pin rclone** — replace `RUN curl https://rclone.org/install.sh | bash` with:

```dockerfile
# rclone pinned from the release zip: install.sh takes no version argument, apt's rclone is too old
ARG RCLONE_VERSION=1.75.2
RUN curl -fsSLo /tmp/rclone.zip https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-amd64.zip \
 && unzip -q /tmp/rclone.zip -d /tmp \
 && install -m 0755 /tmp/rclone-v${RCLONE_VERSION}-linux-amd64/rclone /usr/local/bin/rclone \
 && rm -rf /tmp/rclone.zip /tmp/rclone-v${RCLONE_VERSION}-linux-amd64
```

- [ ] **Step 4: NLTK_DATA** — in the runtime `ENV` block add `NLTK_DATA=/workspace/models/nltk_data \` after `HF_HOME=/workspace/models \`; in the bashrc block add `echo 'export NLTK_DATA="/workspace/models/nltk_data"'; \` after the `HF_HOME` echo.

- [ ] **Step 5: Verify what can be verified here** (images build on the Mac only — memory `project_docker_rebuild`):

```bash
grep -n "astral.sh/uv\|rclone\|jq\|NLTK_DATA" Dockerfile
curl -sIL https://astral.sh/uv/0.12.19/install.sh | grep "^HTTP"        # final 200
curl -sI https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip | head -1   # 200
```

Then replay Step 3's commands in a scratch dir to prove them: `cd "$SP" && mkdir -p rclone-proof && cd rclone-proof` and run the curl/unzip with `-d .`, then `./rclone-v1.75.2-linux-amd64/rclone version | head -1` → `rclone v1.75.2`. Note in the CHANGELOG entry that the image rebuild is owed on the Mac.

- [ ] **Step 6: Commit**

```bash
git commit --only Dockerfile -m "build(docker): jq, pinned uv 0.12.19 and rclone 1.75.2, NLTK_DATA on the models volume"
```

---

### Task 10: Install docs

**Files:** Modify `README.md` (§2-§3, ~lines 102-140), `docs/source/api/pointcloud.rst` (~155-162)

- [ ] **Step 1: README §2** — after the `bash setup.sh` block, add:

```markdown
On RunPod the `/workspace` network volume hides the image's own checkout: clone into `/workspace` (as above), then run `bash setup.sh` there.
```

Replace the `uv sync` extras block with:

```sh
uv sync                                         # light core only (numpy, scipy, zarr, ...)
uv sync --extra semantics                       # + 2D features and segmentation (CPU or GPU, no CUDA build)
uv sync --extra reconstruction --extra remote   # + full pipeline and the reconstruct CLI (CUDA toolkit)
uv sync --extra feedforward                     # + feedforward models (VGGT-X, MapAnything, ...)
uv sync --extra instantsfm                      # + InstantSfM backend (needs libsuitesparse-dev)
uv sync --extra gpu                             # + CUDA wheels used to compile bae / gsplat
uv sync --extra dashboard                       # + dashboard
uv sync --extra dev                             # + lint / test tooling
uv sync --all-extras                            # everything (what setup.sh does)
```

Then a new paragraph:

```markdown
**Semantics only** (another project, CPU or GPU, no CUDA toolkit): run this from outside this repo, so uv does not apply our cu121 overrides:

    uv pip install 'collab_splats[semantics] @ git+https://github.com/BasisResearch/collab-splats@<commit>'
```

In the `> **Use setup.sh ...` callout, replace the bullet `` `uv pip install` ignores this project's `[tool.uv]` config and resolves the wrong package sources. `` with:

```markdown
> - Inside this repo, `uv pip install` applies our `[tool.uv]` overrides; outside it, git dependencies resolve without them (no cu121 torch, no pinned sources).
```

- [ ] **Step 2: README §3** — replace the Linux rclone line with:

```sh
curl -fsSLo rclone.zip https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip \
  && unzip -q rclone.zip && sudo install -m 0755 rclone-v1.75.2-linux-amd64/rclone /usr/local/bin/ \
  && sudo apt-get install -y jq                                         # Linux (apt's rclone is too old)
```

and add after it: `The Docker image ships both, so skip this step there.`

- [ ] **Step 3: pointcloud.rst** — replace the `**Install.**` paragraph (from `**Install.** ``setup.sh`` installs` through `re-run that ``setup.sh`` block afterwards.`) with:

```rst
**Install.** The ``instantsfm`` extra (in ``setup.sh``'s ``uv sync --all-extras``) locks
``instantsfm`` at a pinned git commit (``d3e599e``) with its upstream requirements blanked
(upstream pins ``numpy==1.26.4``; the lock runs numpy 2.x), plus ``pyceres==2.3``,
``scikit-sparse==0.4.15`` (needs ``libsuitesparse-dev``) and ``easydict==1.13``. ``setup.sh``
clones Video-Depth-Anything into ``third_party/Video-Depth-Anything`` at ``4f5ae23`` (source
only). The metric checkpoint (``depth-anything/Metric-Video-Depth-Anything-Large``, ~1.5 GB) is
pulled from the Hugging Face hub on first use and cached under ``HF_HOME``; with no network then,
the first sfm run fails with a ``RuntimeError`` naming the repo.
```

- [ ] **Step 4: Verify docs tests**

```bash
grep -rn "no-deps\|InstantSfM block\|rclone.org/install.sh" README.md docs/source configs/README.md
/opt/venv/reconstruction/bin/python -m pytest tests/docs -q
```

Expected: grep finds no stale install claims; docs tests pass.

- [ ] **Step 5: Commit**

```bash
git commit --only README.md docs/source/api/pointcloud.rst -m "docs: install extras, semantics-only install, pinned rclone, locked InstantSfM"
```

---

### Task 11: Proofs

- [ ] **Step 1: Clean CPU venv, outside the repo cwd** (inside it uv applies our overrides)

```bash
cd "$SP"
rm -rf "$SP/sem-proof"
uv venv "$SP/sem-proof" --python 3.11
uv pip install --python "$SP/sem-proof/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" \
  "collab_splats[semantics] @ /workspace/collab-splats"
uv pip list --python "$SP/sem-proof/bin/python" > "$SP/sem-proof-list.txt"; ls -la "$SP/sem-proof-list.txt"
grep -Ei "^(bae|pypose|pycolmap|pycolmap-cuda12|open3d|mobile.sam|gsplat|collab.data|nvidia-) " "$SP/sem-proof-list.txt"
```

Expected: install succeeds; the grep prints nothing (exit 1).

- [ ] **Step 2: Import every semantics module + load Talk2DINO on CPU**

```bash
cd "$SP" && HF_HOME=/workspace/models "$SP/sem-proof/bin/python" - <<'EOF'
import pkgutil, collab_splats, collab_splats.semantics as s
print("from:", collab_splats.__file__)
for m in pkgutil.walk_packages(s.__path__, s.__name__ + "."):
    __import__(m.name)
print("all semantics modules import")
from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor
Talk2DinoExtractor(device="cpu")
print("Talk2DinoExtractor OK")
EOF
```

Expected: `from:` points into `$SP/sem-proof/lib/python3.11/site-packages/...` (not the checkout), then both OK lines. Record the outputs in the CHANGELOG entry.

- [ ] **Step 3: Container regression check** — `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/features/test_talk2dino_deps.py -m "gpu or not gpu" -v` → 5 passed.

- [ ] **Step 4: Full suite in tmux**

```bash
tmux new-session -d -s inst-gate "cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:cacheprovider > $SP/pytest_full.log 2>&1; echo \$? > $SP/pytest_full.exit"
```

Wait for `$SP/pytest_full.exit` to exist (use Monitor with an until-loop, not sleep). Expected exit `0`. Any failure: rerun that test alone on `c0e6e9d3` (pre-work HEAD) (`git stash` is NOT safe with concurrent sessions — use `git worktree add $SP/base c0e6e9d3` with the PYTHONPATH recipe) to classify pre-existing vs caused; fix every caused failure.

- [ ] **Step 5: Fresh setup.sh** — `bash setup.sh > "$SP/setup_final.log" 2>&1; echo "exit=$?"`; same expectations as Task 8 Step 6 (exit 0, only the nvcc WARN, no churn).

---

### Task 12: Land

- [ ] **Step 1: graphify** — `cd /workspace/collab-splats && graphify update .`

- [ ] **Step 2: CHANGELOG** — add at the top of the entries in `docs/superpowers/CHANGELOG.md`:

```markdown
Recently completed (2026-10-12): **installability** — container setup fixes and a standalone `collab_splats[semantics]` extra ([spec](specs/2026-10-12-installability-design.md) · [plan](plans/2026-10-12-installability.md)). **What changed.** Talk2DINOv3 remote-code deps (`clip` direct URL @ `d05afc43`, `einops`, `omegaconf`) restored — `47633e39` had broken `Talk2DinoExtractor`. Base deps are a light core; extras `semantics` (CPU-installable, no bae/pycolmap/open3d/mobile_sam), `reconstruction` (includes `semantics`), `remote` (collab-data), `instantsfm` (locked, upstream requires-dist blanked). Lazy: `geometry.BundleAdjustment(Config)`, `preproc.calibrate_camera/undistort_frames`, mobile_sam in the auto strategy, open3d in `fit_dominant_plane`. Torch cu121 overrides/sources gated off darwin. setup.sh: uv from PATH, no InstantSfM reinstall churn, WordNet in `$NLTK_DATA` (`/workspace/models/nltk_data`), creator smoke via `get_creator`. Dockerfile: jq, uv 0.12.19, rclone 1.75.2 zip, `NLTK_DATA`. **Gates.** <AST + fresh-import tests; clean CPU venv proof output; full tests/ counts; setup.sh exit 0 with only the nvcc WARN — fill in from Task 11 logs>. **Owed.** Docker image rebuild (Mac); collab-data re-pins `collab_splats[semantics]` from `@db-semantics-package` to this commit; user drops the apt line in `/workspace/setup_profile.sh`.
```

Fill the `<...>` Gates text from the Task 11 logs before committing (no placeholder may land).

- [ ] **Step 3: CLAUDE.md** — remove the `installability` In-Flight bullet; add `- **installability** (2026-10-12)` as the first line of the `## Recently Completed` five-newest list and drop its last line.

- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/CHANGELOG.md
git commit --only CLAUDE.md docs/superpowers/CHANGELOG.md -m "docs(changelog): installability"
```

- [ ] **Step 5: Report** to the user: commit range, gate numbers, owed items. Do not push unless the user asks.
