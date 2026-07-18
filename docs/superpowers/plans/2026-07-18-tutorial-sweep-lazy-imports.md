# Tutorial Sweep + Lazy-Import Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut reconstruction-stack first-import time roughly in half via library-level lazy imports, then align all remaining tutorial notebooks to the `FRAMES`/`TUTORIAL_CACHE` layout with VGGT-Omega as the sole executed backend and LoMa as the primary localization matcher.

**Architecture:** Phase 1 makes three package `__init__`s lazy (PEP 562, the pattern already used by `collab_splats/dashboard/__init__.py` and partially by `geometry/__init__.py`) and demotes one annotation-only import. Phase 2 rewrites/executes notebooks in pipeline order against `2024_02_06/C0043`. Phase 3 migrates the fieldwork-data holdouts and retires dead config.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest, jupyter nbconvert (tmux for heavy runs), zarr v3, VGGT-Omega, LoMa.

**Spec:** `docs/superpowers/specs/2026-07-18-tutorial-sweep-lazy-imports-design.md`

**Measured baseline (2026-07-18, warm cache, `-X importtime`):**
- `import collab_splats` — 11 ms (already lazy, leave alone)
- `from collab_splats.pointcloud.feedforward import VGGTXCreator` — ~14.9 s
  - `geometry/__init__` eager `bundle_adjustment` → pypose + bae: ~2.5 s
  - `semantics/__init__` eager `segmentation` → mobile_sam → timm: ~3.0 s (pulled by `pointcloud/utils.py`)
  - `vggt.models.vggt` (via `vggtx.py`): ~3.7 s (external `vggt.layers.mlp` 3.6 s self-time — deferrable, not removable)
  - torch: ~1.5 s (unavoidable)

**Shared-branch caution:** the working tree carries uncommitted edits from the in-flight feedforward-mesh session (`collab_splats/mesh/*`, `tests/mesh/*`, `02_pointcloud/feedforward_mesh.ipynb`, `02_pointcloud/feedforward_methods.ipynb`, `README.md`). NEVER `git add -A` / `git add .`. Stage only the exact files each task touches. Before editing `feedforward_methods.ipynb` (Task 8), diff the working-tree version against HEAD and preserve any feedforward-mesh hunks.

---

## Phase 1 — Library lazy-import pass

### Task 1: Lazy `geometry/__init__`

BA (`pypose`+`bae`) loads today whenever anything imports `collab_splats.geometry.transforms` (package `__init__` runs first). Keep `transforms` eager (cheap, no heavy deps); make everything else resolve via `__getattr__`.

**Files:**
- Modify: `collab_splats/geometry/__init__.py`
- Test: `tests/test_import_time.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_import_time.py`:

```python
"""Lazy-import regression guards: importing light modules must not load heavy deps.

Each test runs a subprocess so sys.modules state is clean regardless of test order.
"""

import subprocess
import sys

PY = sys.executable


def _forbidden_after(import_stmt: str, forbidden: list[str]) -> list[str]:
    """Run import_stmt in a fresh interpreter; return which forbidden modules loaded."""
    code = (
        f"import sys; {import_stmt}; "
        f"print(','.join(m for m in {forbidden!r} if m in sys.modules))"
    )
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, check=True)
    return [m for m in out.stdout.strip().split(",") if m]


def test_geometry_transforms_does_not_load_ba():
    loaded = _forbidden_after(
        "import collab_splats.geometry.transforms", ["pypose", "bae"]
    )
    assert not loaded, f"geometry.transforms pulled heavy BA deps: {loaded}"


def test_geometry_ba_still_importable():
    # Lazy attrs must still resolve
    code = (
        "from collab_splats.geometry import BundleAdjustment, BundleAdjustmentConfig, "
        "LoopClosureConfig, PoseGraph, Submap; print('ok')"
    )
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"
```

- [ ] **Step 2: Run tests to verify the guard fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_import_time.py -v`
Expected: `test_geometry_transforms_does_not_load_ba` FAILS (pypose/bae listed); `test_geometry_ba_still_importable` passes.

- [ ] **Step 3: Rewrite `collab_splats/geometry/__init__.py`**

```python
"""Geometry backend: loop closure, bundle adjustment, and SE(3)/pose transforms."""

from .transforms import (
    OPENGL_TO_OPENCV,
    extract_intrinsics,
    extrinsics_to_homogeneous,
    invert_poses,
    rotation_align_vectors,
)

# name -> (submodule, attr). Resolved on first access — bundle_adjustment pulls
# pypose+bae (~2.5s) and loop_closure pulls the retrieval stack; neither may load
# just because someone imported geometry.transforms.
_LAZY_ATTRS = {
    "BundleAdjustment": (".bundle_adjustment", "BundleAdjustment"),
    "BundleAdjustmentConfig": (".bundle_adjustment", "BundleAdjustmentConfig"),
    "LoopClosure": (".loop_closure", "LoopClosure"),
    "LoopClosureConfig": (".loop_closure", "LoopClosureConfig"),
    "PoseGraph": (".loop_closure", "PoseGraph"),
    "Submap": (".loop_closure", "Submap"),
}


def __getattr__(name):
    # Lazy import (PEP 562) — also breaks the loop_closure.wrapper -> pointcloud
    # -> geometry.transforms import cycle that an eager import would create.
    if name in _LAZY_ATTRS:
        from importlib import import_module

        module_path, attr = _LAZY_ATTRS[name]
        value = getattr(import_module(module_path, __name__), attr)
        globals()[name] = value  # cache — __getattr__ runs once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OPENGL_TO_OPENCV",
    "BundleAdjustment",
    "BundleAdjustmentConfig",
    "LoopClosure",
    "LoopClosureConfig",
    "PoseGraph",
    "Submap",
    "extract_intrinsics",
    "extrinsics_to_homogeneous",
    "invert_poses",
    "rotation_align_vectors",
]
```

- [ ] **Step 4: Run the guard + geometry tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_import_time.py tests/geometry/ -v`
Expected: all PASS. If any module does `from collab_splats.geometry import X` at top level for a lazy name, that still works (module-level `__getattr__` serves plain imports too).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/__init__.py tests/test_import_time.py
git commit -m "perf(geometry): lazy BA/LC exports — transforms importers no longer pay pypose+bae"
```

### Task 2: Drop runtime `semantics` import from `pointcloud/utils.py`

`BaseFeatureExtractor` is referenced only in annotations (file has `from __future__ import annotations`); the try/except runtime import exists solely for that and costs ~3 s (mobile_sam→timm via `semantics/__init__`).

**Files:**
- Modify: `collab_splats/pointcloud/utils.py:28-31`
- Test: `tests/test_import_time.py`

- [ ] **Step 1: Verify annotation-only usage**

Run: `grep -n "BaseFeatureExtractor" collab_splats/pointcloud/utils.py`
Expected: only the import at line 29 and the `= None` fallback at line 31, plus (possibly) type-hint occurrences inside function signatures. If any occurrence is a runtime use (`isinstance`, call, attribute access), STOP and move the import inside that function instead — then adapt Step 3 accordingly.

- [ ] **Step 2: Add the failing guard test**

Append to `tests/test_import_time.py`:

```python
def test_pointcloud_utils_does_not_load_sam():
    loaded = _forbidden_after(
        "import collab_splats.pointcloud.utils", ["timm", "mobile_sam"]
    )
    assert not loaded, f"pointcloud.utils pulled segmentation deps: {loaded}"
```

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_import_time.py::test_pointcloud_utils_does_not_load_sam -v`
Expected: FAIL (timm, mobile_sam loaded).

- [ ] **Step 3: Move the import under TYPE_CHECKING**

In `collab_splats/pointcloud/utils.py`, replace:

```python
if TYPE_CHECKING:
    from .feedforward.base import FeedforwardResult

try:
    from collab_splats.semantics.features import BaseFeatureExtractor
except ImportError:
    BaseFeatureExtractor = None  # type: ignore[assignment]
```

with:

```python
if TYPE_CHECKING:
    from collab_splats.semantics.features import BaseFeatureExtractor

    from .feedforward.base import FeedforwardResult
```

- [ ] **Step 4: Run guard + pointcloud tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_import_time.py tests/pointcloud/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/utils.py tests/test_import_time.py
git commit -m "perf(pointcloud): BaseFeatureExtractor import is annotation-only — drop 3s SAM/timm tax"
```

### Task 3: Lazy `pointcloud/__init__` + `feedforward/__init__`

Backends resolve on demand: importing `collab_splats.pointcloud` must not load vggt/mapanything/hloc; `get_creator("vggt_omega")` on a machine without the submodule raises a clear `ImportError` naming `setup/feedforward.sh`.

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py`
- Modify: `collab_splats/pointcloud/feedforward/__init__.py`
- Test: `tests/test_import_time.py`, `tests/pointcloud/test_registry_lazy.py` (new)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_import_time.py`:

```python
def test_pointcloud_package_does_not_load_backends():
    loaded = _forbidden_after(
        "import collab_splats.pointcloud",
        ["vggt", "mapanything", "hloc", "mobile_sam", "pypose"],
    )
    assert not loaded, f"pointcloud/__init__ pulled backends eagerly: {loaded}"
```

Create `tests/pointcloud/test_registry_lazy.py`:

```python
"""get_creator/make_creator behavior with the lazy registry."""

import pytest

from collab_splats.pointcloud import get_creator


def test_get_creator_unknown_name():
    with pytest.raises(KeyError, match="unknown pointcloud backend"):
        get_creator("nope")


def test_get_creator_resolves_vggtx():
    cls = get_creator("vggtx")
    assert cls.__name__ == "VGGTXCreator"


def test_get_creator_missing_optional_backend(monkeypatch):
    # Simulate an uninstalled optional backend: registry entry present, module import fails
    import collab_splats.pointcloud as pc

    monkeypatch.setitem(
        pc._CREATORS, "vggt_omega", ("collab_splats.pointcloud.feedforward._nonexistent", "VGGTOmegaCreator")
    )
    with pytest.raises(ImportError, match="setup/feedforward.sh"):
        get_creator("vggt_omega")
```

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_import_time.py::test_pointcloud_package_does_not_load_backends tests/pointcloud/test_registry_lazy.py -v`
Expected: import-time guard FAILS; registry tests fail on `_CREATORS` (doesn't exist yet).

- [ ] **Step 2: Rewrite `collab_splats/pointcloud/__init__.py`**

```python
"""Pointcloud creation: classical SfM and feedforward backends.

Creator classes resolve lazily (PEP 562) — importing this package costs
milliseconds; the heavy model stacks load on first attribute access or
get_creator() call.
"""

from importlib import import_module

from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult

# ── Lazy backend registry ─────────────────────────────────────────────────────
# name -> (module_path, class_name); the module imports on first get_creator()
_CREATORS: dict[str, tuple[str, str]] = {
    "colmap":      ("collab_splats.pointcloud.sfm", "ColmapCreator"),
    "hloc":        ("collab_splats.pointcloud.sfm", "HlocCreator"),
    "mapanything": ("collab_splats.pointcloud.feedforward.mapanything", "MapAnythingCreator"),
    "vggtx":       ("collab_splats.pointcloud.feedforward.vggtx", "VGGTXCreator"),
    "vggt_omega":  ("collab_splats.pointcloud.feedforward.vggt_omega", "VGGTOmegaCreator"),
    "vggt_spark":  ("collab_splats.pointcloud.feedforward.vggt_spark_creator", "VGGTSPARKCreator"),
}

# Class-name access (collab_splats.pointcloud.VGGTXCreator etc.) — same lazy path
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    class_name: (module_path, class_name) for module_path, class_name in _CREATORS.values()
}
_LAZY_ATTRS.update(
    {
        "BaseFeedforwardCreator": ("collab_splats.pointcloud.feedforward", "BaseFeedforwardCreator"),
        "compute_obb_from_points": ("collab_splats.pointcloud.utils", "compute_obb_from_points"),
        "get_points_in_mask": ("collab_splats.pointcloud.utils", "get_points_in_mask"),
    }
)


def __getattr__(name):
    """Lazy resolve creators and utils so importing the package stays cheap."""
    if name in _LAZY_ATTRS:
        module_path, attr = _LAZY_ATTRS[name]
        value = getattr(import_module(module_path), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_creator(name: str) -> type[BasePointcloudCreator]:
    """Get a pointcloud creator class by registry name.

    Raises:
        KeyError: Unknown backend name.
        ImportError: Optional backend (vggt_omega / vggt_spark) not installed.
    """
    if name not in _CREATORS:
        raise KeyError(
            f"unknown pointcloud backend '{name}'. Available: {sorted(_CREATORS)}"
        )
    module_path, class_name = _CREATORS[name]
    try:
        return getattr(import_module(module_path), class_name)
    except ImportError as e:
        raise ImportError(
            f"backend '{name}' is not installed ({e}). "
            "Optional backends require setup/feedforward.sh with the matching submodule."
        ) from e


def make_creator(
    name: str,
    *,
    use_lc: bool = False,
    lc_config=None,
    **kwargs,
):
    """Construct a pointcloud creator, optionally wrapped with LoopClosure."""
    creator = get_creator(name)(**kwargs)
    if use_lc:
        # Deferred import — geometry.loop_closure.wrapper imports this package;
        # an eager import would cycle at load time.
        from collab_splats.geometry import LoopClosure

        creator = LoopClosure(creator, config=lc_config)
    return creator


__all__ = [
    "BasePointcloudCreator",
    "BaseFeedforwardCreator",
    "CoordinateFrame",
    "ColmapCreator",
    "HlocCreator",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "VGGTOmegaCreator",
    "VGGTSPARKCreator",
    "compute_obb_from_points",
    "get_points_in_mask",
    "get_creator",
    "make_creator",
]
```

Note the availability probes (`_OMEGA_AVAILABLE`/`_SPARK_AVAILABLE`) are gone: omega and spark are always registered; absence surfaces as `ImportError` at resolve time, per spec.

- [ ] **Step 3: Rewrite `collab_splats/pointcloud/feedforward/__init__.py`**

Keep `base` eager (FeedforwardResult is the ubiquitous cheap type); lazy-resolve every creator and the vggtx re-exports (vggtx pulls `vggt.models` ~3.7 s):

```python
"""Feedforward pointcloud creators: VGGT-X, MapAnything, VGGT-Omega, and VGGT-SPARK backends.

Import from here — submodule structure is an implementation detail. Creator
classes resolve lazily (PEP 562); FeedforwardResult and the base creator stay
eager because every consumer needs them.
"""
from __future__ import annotations

from importlib import import_module

# ── Public types and utilities ────────────────────────────────────────────────
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    build_pycolmap_reconstruction,
    compute_multiview_depth_confidence,
    _raw_to_world_points,  # re-exported for geometry/loop_closure/wrapper.py
)

# ── Lazy creators + vggtx re-exports ──────────────────────────────────────────
# vggtx pulls vggt.models (~3.7s); omega/spark are optional installs.
# unproject_and_filter_points stays patchable as
# collab_splats.pointcloud.feedforward.unproject_and_filter_points — mock.patch
# triggers __getattr__ on lookup, then swaps the cached module attribute.
_LAZY_ATTRS = {
    "VGGTXCreator": (".vggtx", "VGGTXCreator"),
    "MapAnythingCreator": (".mapanything", "MapAnythingCreator"),
    "VGGTOmegaCreator": (".vggt_omega", "VGGTOmegaCreator"),
    "VGGTSPARKCreator": (".vggt_spark_creator", "VGGTSPARKCreator"),
    "unproject_and_filter_points": (".vggtx", "unproject_and_filter_points"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_path, attr = _LAZY_ATTRS[name]
        value = getattr(import_module(module_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeedforwardResult",
    "BaseFeedforwardCreator",
    "build_pycolmap_reconstruction",
    "VGGTXCreator",
    "MapAnythingCreator",
    "VGGTOmegaCreator",
    "VGGTSPARKCreator",
    "unproject_and_filter_points",
    "compute_multiview_depth_confidence",
]
```

If `_raw_to_world_points` actually lives in `vggtx.py` rather than `base.py` (check: `grep -n "_raw_to_world_points" collab_splats/pointcloud/feedforward/base.py`), put it in `_LAZY_ATTRS` pointing at its real home instead of the eager block.

- [ ] **Step 4: Run the focused tests, then the full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_import_time.py tests/pointcloud/ tests/test_feedforward_logging.py tests/test_cu121_migration.py tests/wrapper/ -v`
Expected: all PASS. Watch specifically the tests that `mock.patch` `collab_splats.pointcloud.feedforward.unproject_and_filter_points` (`tests/pointcloud/test_vggtx_preproc.py`) and anything patching `collab_splats.pointcloud.*` in `tests/wrapper/test_reconstructor.py`. A `patch` failure with "does not have the attribute" means a name is missing from `_LAZY_ATTRS`.

Then full suite: `/opt/venv/reconstruction/bin/python -m pytest tests/ -x -q`
Expected: same pass/fail counts as `docs/known-test-failures.md` baseline.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/__init__.py tests/pointcloud/test_registry_lazy.py tests/test_import_time.py
git commit -m "perf(pointcloud): PEP 562 lazy creator registry — package import no longer loads backends"
```

### Task 4: Measure, smoke, format

**Files:** none new (measurements go in this plan's completion notes)

- [ ] **Step 1: Measure importtime after**

```bash
cd /workspace/collab-splats
/opt/venv/reconstruction/bin/python -X importtime -c 'import collab_splats.pointcloud' 2>&1 | tail -1
/opt/venv/reconstruction/bin/python -X importtime -c 'from collab_splats.pointcloud.feedforward import FeedforwardResult' 2>&1 | tail -1
/opt/venv/reconstruction/bin/python -X importtime -c 'from collab_splats.pointcloud import get_creator; get_creator("vggt_omega")' 2>&1 | tail -1
```

Expected: `import collab_splats.pointcloud` cumulative drops from ~14.9 s to torch-dominated ~2-3 s; record the three numbers at the bottom of this plan.

- [ ] **Step 2: Dashboard smoke gate (mandatory — dashboard imports get_creator/make_creator)**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: prints `SMOKE PASS`.

- [ ] **Step 3: Format and commit any fixes**

```bash
black collab_splats/geometry/__init__.py collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/__init__.py collab_splats/pointcloud/utils.py tests/test_import_time.py tests/pointcloud/test_registry_lazy.py
isort collab_splats/geometry/__init__.py collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/__init__.py collab_splats/pointcloud/utils.py tests/test_import_time.py tests/pointcloud/test_registry_lazy.py
git add -u collab_splats/geometry/__init__.py collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/__init__.py collab_splats/pointcloud/utils.py tests/test_import_time.py tests/pointcloud/test_registry_lazy.py
git diff --cached --quiet || git commit -m "style: format lazy-import pass"
```

---

## Phase 2 — Notebook sweep (pipeline order, each executed vs C0043)

**Common conventions for every notebook task below:**
- Config comes from `%run ../tutorial_config.py`: read frames from `FRAMES`, write ALL notebook output under `TUTORIAL_CACHE` (canonical scene dir is rclone-synced and read-only for tutorials).
- The shared omega cache is `TUTORIAL_CACHE / "vggt_omega.zarr"` — `feedforward_methods` produces it; every downstream notebook loads it with a fail-loud cell (pattern below).
- Execute headless in tmux (46.6 GB cgroup cap; never run two heavy notebooks in parallel):
  ```bash
  tmux new-session -d -s nb 'source /opt/venv/reconstruction/bin/activate && \
    jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 <notebook-path> 2>&1 | tee /tmp/nb.log'
  ```
  Poll with `tmux capture-pane -pt nb | tail -20`; done when nbconvert exits 0.
- Downstream loader cell (used verbatim wherever a notebook consumes the reconstruction):
  ```python
  # Load the VGGT-Omega reconstruction produced by 02_pointcloud/feedforward_methods.ipynb
  from collab_splats.pointcloud.feedforward import FeedforwardResult

  _omega_cache = TUTORIAL_CACHE / "vggt_omega.zarr"
  assert _omega_cache.exists(), (
      f"missing {_omega_cache} — run 02_pointcloud/feedforward_methods.ipynb first"
  )
  result = FeedforwardResult.load_zarr(_omega_cache, load_images=True)
  print(f"loaded {result.points.shape[0]:,} points, {len(result.image_paths)} frames")
  ```
- Commit per notebook, staging ONLY that notebook (shared-branch caution above).

### Task 5: `feedforward_methods.ipynb` — omega-only

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`

- [ ] **Step 1: Reconcile working-tree state**

Run: `git diff docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb | head -100`
Understand what the feedforward-mesh session changed. Keep those semantics (do not revert their cells); rework on top of the working-tree version.

- [ ] **Step 2: Rework the notebook**

Structure (edit via NotebookEdit, cell by cell):
1. **Title markdown** — rewrite: notebook runs **VGGT-Omega** on the pipeline keyframes and writes the zarr cache all downstream notebooks (`semantic_lifting`, `localization`, `bundle_adjustment`) read. Prerequisite: keyframes present in `FRAMES` (pipeline-written `frames/`, or run tutorial 01). Remove `images/` wording.
2. **Setup code** — keep `%load_ext autoreload`; imports trimmed to what the omega-only flow uses (no open3d unless a kept cell uses it); `%run ../tutorial_config.py`.
3. **Frame list cell**:
   ```python
   # Pipeline keyframes: read-only input from the canonical scene dir
   image_paths = sorted(FRAMES.glob("*.jpg")) + sorted(FRAMES.glob("*.png"))
   assert image_paths, f"no keyframes in {FRAMES} — run the dashboard pipeline or tutorial 01"
   image_paths = image_paths[:MAX_FRAMES]
   print(f"{len(image_paths)} keyframes from {FRAMES}")
   ```
4. **Run/reload cell** (keep the existing omega cache-or-run pattern, retargeted at `TUTORIAL_CACHE`):
   ```python
   # Run VGGT-Omega once; later executions reload the zarr cache
   from collab_splats.pointcloud import make_creator
   from collab_splats.pointcloud.feedforward import FeedforwardResult

   _omega_cache = TUTORIAL_CACHE / "vggt_omega.zarr"
   _omega_cache.parent.mkdir(parents=True, exist_ok=True)
   if _omega_cache.exists():
       result = FeedforwardResult.load_zarr(_omega_cache)
       print(f"loaded cached reconstruction ({result.points.shape[0]:,} pts)")
   else:
       creator = make_creator("vggt_omega")
       result = creator.run(image_paths)
       result.save_zarr(_omega_cache)
       print(f"reconstructed {result.points.shape[0]:,} pts → {_omega_cache}")
   ```
   (Match the exact creator-run call signature used by the current notebook's omega cell — reuse its code, only retarget paths.)
5. **Visualization cells** — keep the pyvista pointcloud + frustum rendering for the omega result only; delete VGGT-X and MapAnything run/compare cells.
6. **"Other backends" markdown** (replaces the deleted comparison):
   ```markdown
   ## Other backends

   `make_creator(<name>)` swaps the reconstruction backend with no other code changes:

   | name | model | notes |
   |------|-------|-------|
   | `"vggtx"` | VGGT-X | fastest baseline |
   | `"mapanything"` | MapAnything | metric-scale output |
   | `"vggt_spark"` | VGGT-SPARK | optional install |
   | `"colmap"` / `"hloc"` | classical SfM | no GPU model, slower |

   Optional backends require `setup/feedforward.sh` with the matching submodule.
   See the [API docs](../../api/pointcloud.rst) for per-backend parameters.
   ```

- [ ] **Step 3: Execute headless (tmux, pattern above)**

Expected: exits 0; zarr exists at `/workspace/outputs/tutorial_cache/2024_02_06/C0043/vggt_omega.zarr`.

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(tutorials): feedforward_methods — VGGT-Omega only, FRAMES/TUTORIAL_CACHE layout"
```

### Task 6: `bundle_adjustment.ipynb` — omega backbone + layout

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`

- [ ] **Step 1: Rework** — replace `IMAGES` references with `FRAMES`; reconstruction input = the shared omega loader cell (verbatim from the Phase 2 preamble); primary backbone `vggt_omega`; BA outputs under `TUTORIAL_CACHE`; markdown notes other backbones work via `make_creator`.
- [ ] **Step 2: Execute headless (tmux)** — exits 0.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
git commit -m "docs(tutorials): bundle_adjustment on VGGT-Omega + new layout"
```

### Task 7: `slam_loop_closure.ipynb` — omega backbone + LC calibration

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`

- [ ] **Step 1: Rework** — same layout changes; primary backbone omega with its calibrated LC settings (`target_layer=13`, `similarity_threshold=1.55` — match the exact LoopClosureConfig field names used in the current notebook cells); markdown table of per-backbone calibrations (spark 0.95 native, vggtx L10/1.17, omega L13/1.55, mapanything L4/1.46).
- [ ] **Step 2: Execute headless (tmux)** — exits 0.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
git commit -m "docs(tutorials): slam_loop_closure on VGGT-Omega (L13/1.55) + new layout"
```

### Task 8: `04_semantics` triple

**Files:**
- Modify: `docs/source/tutorials/04_semantics/feature_extraction.ipynb`
- Modify: `docs/source/tutorials/04_semantics/segmentation.ipynb`
- Modify: `docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb`

- [ ] **Step 1: Rework all three** — `IMAGES` → `FRAMES` for input images; any reconstruction load uses the shared omega loader cell; any written artifact (features zarr, masks, figures) goes under `TUTORIAL_CACHE`. No method changes.
- [ ] **Step 2: Execute each headless (tmux, sequentially)** — each exits 0.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/04_semantics/feature_extraction.ipynb docs/source/tutorials/04_semantics/segmentation.ipynb docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb
git commit -m "docs(tutorials): 04_semantics aligned to FRAMES/TUTORIAL_CACHE + omega cache"
```

### Task 9: `05_lifting/semantic_lifting.ipynb`

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`

- [ ] **Step 1: Rework** — shared omega loader cell (with `load_images=True` — `lift_features` needs pixel_indices/depth/confidence); lifted-feature output under `TUTORIAL_CACHE`; layout alignment.
- [ ] **Step 2: Execute headless (tmux)** — exits 0.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "docs(tutorials): semantic_lifting on omega cache + new layout"
```

### Task 10: `07_localization/localization.ipynb` — LoMa primary

**Files:**
- Modify: `docs/source/tutorials/07_localization/localization.ipynb`

- [ ] **Step 1: Rework**

1. **Title markdown** — pipeline description updated: local feature matching with **LoMa** (dashboard default); XFeat/DISK available as drop-in alternatives.
2. **§1 load** — shared omega loader cell replaces the current zarr path.
3. **§3 localize** — the current notebook already has a working LoMa cell (its "second localizer with the LoMa-B extractor" section). Promote that construction to be THE localizer:
   ```python
   from collab_splats.localization import CameraLocalizer, LomaExtractor, plot_correspondences

   localizer = CameraLocalizer(result_trimmed, extractor=LomaExtractor())
   ```
   (Reuse the exact constructor call from the existing LoMa section — including any cache-dir argument, which must point under `TUTORIAL_CACHE`.)
4. **Delete** the XFeat run and the second-localizer comparison section.
5. **"Other matchers" markdown**:
   ```markdown
   ## Other matchers

   Swap the extractor to change the matching frontend — everything else is unchanged:

   - `XFeatExtractor()` — lighter/faster, fewer correspondences
   - `DiskExtractor()` — DISK + LightGlue
   - `LomaGExtractor()` — LoMa with global refinement
   ```
6. Query/trim/3D-view cells keep their current logic (paths already come from the loaded result).

- [ ] **Step 2: Execute headless (tmux)** — exits 0; localization finds a pose (`loc.pose is not None` cell passes).
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/07_localization/localization.ipynb
git commit -m "docs(tutorials): localization — LoMa primary matcher, omega cache, new layout"
```

### Task 11: layout-only pair — `colmap_sfm.ipynb`, `feedforward_mesh.ipynb`

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb`
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb`

- [ ] **Step 1: Rework paths only** — `FRAMES`/`TUTORIAL_CACHE` alignment. `feedforward_mesh.ipynb` is owned by the in-flight feedforward-mesh effort: diff working tree vs HEAD first, change path cells only, do not restructure.
- [ ] **Step 2: Execute both headless (tmux, sequentially)** — exit 0. If `colmap_sfm` exceeds the 1800 s cell timeout on 30 frames, cap its frame list (e.g. `image_paths[:15]`) in the notebook with a markdown note.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb
git commit -m "docs(tutorials): colmap_sfm + feedforward_mesh path alignment"
```

---

## Phase 3 — Data-migration tail

### Task 12: `03_splats` / `06_mesh` fieldwork-data migration

**Files:**
- Modify: `docs/source/tutorials/03_splats/derive_splats.ipynb`
- Modify: `docs/source/tutorials/03_splats/visualization.ipynb`
- Modify: `docs/source/tutorials/06_mesh/create_mesh.ipynb`

- [ ] **Step 1: Gate — verify C0043 splat training data exists**

Run: `ls /workspace/outputs/2024_02_06/C0043/` and check for the artifacts these notebooks consume (trained splat checkpoint / nerfstudio outputs — read each notebook's load cells to get the exact expected paths).
If ABSENT: keep the `BASE_DIR` override, add a dated markdown TODO cell at the top of each notebook ("2026-07-18: awaiting C0043 splat training outputs; still on fieldwork-data"), record the blocker in this plan's completion notes, commit that, and skip Steps 2-3.

- [ ] **Step 2: Migrate** — remove `BASE_DIR = /workspace/fieldwork-data/` overrides; standard `%run ../tutorial_config.py`; outputs under `TUTORIAL_CACHE`.
- [ ] **Step 3: Execute headless (tmux; derive_splats is the heaviest — no parallel work)** — exit 0.
- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/03_splats/derive_splats.ipynb docs/source/tutorials/03_splats/visualization.ipynb docs/source/tutorials/06_mesh/create_mesh.ipynb
git commit -m "docs(tutorials): 03_splats/06_mesh migrated to canonical C0043 layout"
```

### Task 13: `evals/ground_truth_evals.ipynb` results path

**Files:**
- Modify: `docs/source/tutorials/evals/ground_truth_evals.ipynb`

- [ ] **Step 1: Rework** — replace the hardcoded results path with a top-of-notebook constant:
   ```python
   # Results written by evals/eval_gt.py (CLI/tmux only — never run compute here)
   RESULTS_DIR = Path("../../../../evals/results").resolve()
   ```
   All downstream cells read via `RESULTS_DIR`. Visualization-only notebook — do NOT execute compute; run headless only if results exist locally, otherwise leave outputs as-is and note it.
- [ ] **Step 2: Commit**

```bash
git add docs/source/tutorials/evals/ground_truth_evals.ipynb
git commit -m "docs(tutorials): ground_truth_evals configurable results path"
```

### Task 14: `tutorial_config.py` cleanup + final gates

**Files:**
- Modify: `docs/source/tutorials/tutorial_config.py`

- [ ] **Step 1: Verify no notebook reads `CACHE_DIR`**

Run: `grep -l "CACHE_DIR" docs/source/tutorials/*/*.ipynb`
Expected: no matches (Phase 2/3 folded them into `OUTPUT_DIR`/`TUTORIAL_CACHE`). If matches remain, fix those notebooks first.

- [ ] **Step 2: Edit `tutorial_config.py`**

Remove the `CACHE_DIR = OUTPUT_DIR` line. In `_infer_video_path`, drop the dead first candidate (rclone-relative `video_ref` never resolves as an absolute path):

```python
            if raw:
                candidate = output_dir / Path(raw).name
                if candidate.exists():
                    return candidate
```

- [ ] **Step 3: Final gates**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q          # suite matches known-failures baseline
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke   # SMOKE PASS
black --check collab_splats/ && isort --check collab_splats/
```

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/tutorial_config.py
git commit -m "docs(tutorials): retire CACHE_DIR alias; drop dead video_ref candidate"
```

---

## Completion notes (fill during execution)

- Importtime before: `collab_splats.pointcloud` ≈ 14.9 s (feedforward creator import)
- Importtime after: _record here_
- Blockers: _record here_
