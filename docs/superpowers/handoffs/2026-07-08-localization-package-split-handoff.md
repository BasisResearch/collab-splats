# Handoff: split `pointcloud/localization.py` into a package

**Date:** 2026-07-08
**From:** LoMa-integration session
**Status:** approved direction (user asked for this); not started
**Branch:** `refactor/cu121-uv-migration` (sole working branch — commit there, never create a new branch)

## Goal

Mechanically split `collab_splats/pointcloud/localization.py` (1,516 lines) into a
package with one module per pipeline stage. **Pure file move — no rewrites, no
renames, no cleanup, no API changes.** Every existing import path must keep working.

## Why

The file is three modules stapled together (the `########` stage dividers literally
mark the seams), it just grew +122 lines from the LoMa integration (squash commit
`5878b4d`), a future RoMaV2 dense-matcher path would grow it further, and the
retrieval extractors are cross-cutting (used by loop closure via `wrappers.py`).
Precedent: `collab_splats/semantics/features/` is already a package.

## Current file map (line numbers at commit `5878b4d`)

| Lines | Content |
|---|---|
| 1–53 | module docstring, all imports, `_XFEAT_PATH` sys.path hack, `logger` |
| 55–67 | `@dataclass LocalFeatures` |
| 69–92 | `@dataclass LocalizationResult` |
| 94–244 | **Stage 1 — retrieval:** `BaseRetrievalExtractor` (~95), `DinoSaladExtractor` (@register "dino-salad", 113), `PECLIPExtractor` (@register "pe-clip", 187) |
| 246–567 | **Stage 2 — local extractors:** `BaseLocalExtractor` (251), `DiskExtractor` ("disk", 274), `XFeatExtractor` ("xfeat", 361), `LomaExtractor` ("loma", 451), `LomaGExtractor` ("loma-g", 562) |
| 569–1395 | **Stage 3 — pose:** `_build_frame_assignments` (574), `CameraLocalizer` (~656; includes zarr cache save/load_index, `from_feedforward`, `localize`) |
| 1397–1516 | **Visualization:** `plot_correspondences` (1402) |

## Target layout

```
collab_splats/pointcloud/localization/
  __init__.py      # re-exports EVERYTHING (see below)
  retrieval.py     # Stage 1 + its imports (open_clip, salad, torchvision.transforms)
  extractors.py    # LocalFeatures + Stage 2 + _XFEAT_PATH hack + kornia/loma imports
  localizer.py     # LocalizationResult + _build_frame_assignments + CameraLocalizer
                   #   + zarr/BloscCodec/pycolmap imports
  viz.py           # plot_correspondences (+ matplotlib import)
```

Internal dependencies (no cycles):
- `localizer.py` imports `LocalFeatures`, `BaseLocalExtractor`, `DiskExtractor` (the
  `from_feedforward` default) from `.extractors`.
- `viz.py` imports `LocalizationResult` from `.localizer`.
- `retrieval.py` is self-contained.
- Each submodule gets its own `logger = logging.getLogger(__name__)`.

`__init__.py` re-exports every public name currently importable from
`collab_splats.pointcloud.localization`: `LocalFeatures`, `LocalizationResult`,
`BaseRetrievalExtractor`, `DinoSaladExtractor`, `PECLIPExtractor`,
`BaseLocalExtractor`, `DiskExtractor`, `XFeatExtractor`, `LomaExtractor`,
`LomaGExtractor`, `CameraLocalizer`, `plot_correspondences` (grep the file for any
other public names before finalizing — this list is from memory of the section map).
Registration happens via decorators at import time, so `__init__.py` importing all
submodules keeps all registry names ("disk", "xfeat", "loma", "loma-g",
"dino-salad", "pe-clip") available.

## Known call sites (verified by grep, 2026-07-08)

Work unchanged via the `__init__.py` shim:
- `collab_splats/pointcloud/__init__.py:20` — `from .localization import (...)`
- `collab_splats/pointcloud/wrappers.py:19` — `BaseRetrievalExtractor`
- `collab_splats/webapp/routers/localize.py:79` — inline import
- `evals/runners/compare_slam_ours.py:45`
- `tests/pointcloud/test_loma_extractor.py`, `test_localization_cache.py`,
  `test_retrieval.py` (imports), `tests/test_cu121_migration.py:110` (module string)

## ⚠️ The one real hazard: `unittest.mock.patch` targets

Two test files patch attributes **on the localization module object**:
- `tests/pointcloud/test_retrieval.py:47,73` —
  `patch("collab_splats.pointcloud.localization.open_clip")`
- `tests/pointcloud/test_feedforward_lc_state.py:53` —
  `patch("collab_splats.pointcloud.localization.BaseRetrievalExtractor.get")`

After the split, `open_clip` lives in `retrieval.py`'s namespace. Patching the
`__init__` re-export does NOT affect the name `retrieval.py` actually resolves at
call time. **Update these patch targets** to
`collab_splats.pointcloud.localization.retrieval.open_clip` (the
`BaseRetrievalExtractor.get` patch patches the class attribute, so it keeps working
regardless — verify anyway). Grep tests for any other
`patch("collab_splats.pointcloud.localization.` occurrences before starting.

## Constraints

- **Move, don't rewrite.** Keep class bodies byte-identical (git should show the
  move as deletions from one file + identical additions; reviewers will diff).
  Keep the `########` dividers where they still make sense within each file.
- Module docstring at the top of the old file (the 3-stage pipeline description)
  → move to `__init__.py`.
- Repo style: imports at top of each submodule; no inline imports.
- **Concurrent sessions:** other Claude sessions commit to this branch. Working
  tree carries their dirty files (mesh/, some notebooks) — stage ONLY the files you
  touch. Confirm `git log -1` is not mid-change by someone else before committing.
  If `collab_splats/mesh/` or unrelated notebooks appear modified, that is
  expected — leave them.
- **Timing:** confirm no other session has `localization.py` dirty
  (`git status --porcelain collab_splats/pointcloud/localization.py`) before starting.
- `docs/superpowers/` is gitignored — commit docs there with `git add -f`.
- Python is `/opt/venv/reconstruction/bin/python` (py3.11). Never bare `python`.

## Verification

1. `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/ -m "not slow"` — green
   (compare against a pre-change baseline run; some unrelated failures are catalogued
   in `docs/known-test-failures.md`).
2. `/opt/venv/reconstruction/bin/python -m pytest tests/test_cu121_migration.py` — the
   module-import sweep must still pass with the package.
3. Registry smoke:
   `python -c "from collab_splats.pointcloud.localization import BaseLocalExtractor, BaseRetrievalExtractor; print(sorted(BaseLocalExtractor._registry), sorted(BaseRetrievalExtractor._registry))"`
   → `['disk', 'loma', 'loma-g', 'xfeat'] ['dino-salad', 'pe-clip']`
4. Slow path (weights cached at /workspace/models/hub, GPU):
   `pytest tests/pointcloud/test_loma_extractor.py -m slow` — 2 pass.
5. Webapp import: `python -c "import collab_splats.webapp.routers.localize"`.

## Deliverable

Single commit: `refactor(pointcloud): split localization.py into stage modules`
containing the package + updated patch targets in the two test files. Nothing else.

## Context pointers

- LoMa integration (why the file just grew): spec
  `docs/superpowers/specs/2026-07-08-loma-matcher-integration-design.md`, plan
  `docs/superpowers/plans/2026-07-08-loma-matcher-integration.md`, decision
  `docs/superpowers/decisions/014-defer-romav2.md`.
- `LomaExtractor` deliberately reaches into `loma`'s private `_descriptor` (commented
  at the call site) — keep that comment with the code.
- The two-classes-per-variant pattern (LomaExtractor/LomaGExtractor) exists because
  `CameraLocalizer.from_feedforward` reverse-looks-up `type(extractor)` in the
  registry for zarr cache keys — do not merge them during the move.
