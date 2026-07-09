# Promote localization to a top-level package

**Date:** 2026-07-09
**Status:** approved direction (promote, clean break — no compat shim)
**Branch:** `refactor/cu121-uv-migration` (sole working branch — commit there, never create a new branch)
**Supersedes:** `docs/superpowers/handoffs/2026-07-08-localization-package-split-handoff.md` (that handoff split in place under `pointcloud/`; this spec promotes to top level instead. Its file map, mock-patch hazard analysis, and constraints carry over.)

## Goal

Move `collab_splats/pointcloud/localization.py` (1,516 lines) out of `pointcloud/`
into a new top-level package `collab_splats/localization/` with one module per
pipeline stage. Clean break: the old import path dies, every call site is updated.
Class and function bodies move byte-identical — no rewrites, no renames, no API
changes beyond the import path.

## Why promotion (not split-in-place)

- **Zero code dependency on pointcloud.** The file's only in-repo import is
  `collab_splats.utils.torch_utils.RegistryMixin`. `CameraLocalizer.from_feedforward`
  duck-types its input ("FeedforwardResult or duck-typed object") — no import, no
  cycle risk.
- **Dependency arrow points the wrong way for nesting.** `pointcloud/wrappers.py`
  (loop closure) *consumes* localization's `BaseRetrievalExtractor`. After promotion
  the graph is clean and one-way: `pointcloud → localization`, `webapp → localization`,
  `evals → localization`.
- **Distinct capability.** pointcloud builds the map; localization registers new
  query images against an existing map. Peer of `semantics/` and `mesh/` in the
  capability-based top-level tree.
- **Growth direction.** Future RoMaV2 dense matcher and more retrieval variants grow
  localization as its own subsystem.
- **No shim** because this is an overall refactor: only ~6 in-repo call sites plus
  tests and one tutorial notebook import the old path. A shim would preserve a path
  we want dead.

## Target layout

```
collab_splats/localization/
  __init__.py      # 3-stage pipeline docstring (moved from old file top) +
                   # re-exports every public name (see below)
  retrieval.py     # Stage 1: BaseRetrievalExtractor, DinoSaladExtractor ("dino-salad"),
                   # PECLIPExtractor ("pe-clip") + open_clip/salad/torchvision imports
  extractors.py    # LocalFeatures + Stage 2: BaseLocalExtractor, DiskExtractor ("disk"),
                   # XFeatExtractor ("xfeat"), LomaExtractor ("loma"),
                   # LomaGExtractor ("loma-g") + _XFEAT_PATH sys.path hack + kornia/loma imports
  localizer.py     # LocalizationResult + _build_frame_assignments + CameraLocalizer
                   # + zarr/BloscCodec/pycolmap imports
  viz.py           # plot_correspondences + matplotlib import
```

Source file map (line numbers at commit `5878b4d`): docstring/imports 1–53,
`LocalFeatures` 55–67, `LocalizationResult` 69–92, Stage 1 94–244, Stage 2 246–567,
Stage 3 569–1395, viz 1397–1516. The `########` stage dividers mark the seams.

Internal dependencies (no cycles):
- `localizer.py` imports `LocalFeatures`, `BaseLocalExtractor`, `DiskExtractor`
  (the `from_feedforward` default) from `.extractors`.
- `viz.py` imports `LocalizationResult` from `.localizer`.
- `retrieval.py` is self-contained.
- Each submodule gets its own `logger = logging.getLogger(__name__)`.

`__init__.py` imports all submodules (decorator registration happens at import time,
keeping registry names "disk", "xfeat", "loma", "loma-g", "dino-salad", "pe-clip"
available) and re-exports: `LocalFeatures`, `LocalizationResult`,
`BaseRetrievalExtractor`, `DinoSaladExtractor`, `PECLIPExtractor`,
`BaseLocalExtractor`, `DiskExtractor`, `XFeatExtractor`, `LomaExtractor`,
`LomaGExtractor`, `CameraLocalizer`, `plot_correspondences`. Grep the old file for
any other public names before finalizing. `_build_frame_assignments` stays private —
tests import it from `collab_splats.localization.localizer`.

## Deletions

- `collab_splats/pointcloud/localization.py` — deleted, no shim.
- `collab_splats/pointcloud/__init__.py:20-29` — the `from .localization import (...)`
  block removed. `collab_splats.pointcloud` no longer re-exports localization names.

## Call-site updates (complete inventory, verified by grep 2026-07-09)

Source:
- `collab_splats/pointcloud/wrappers.py:19` —
  `from .localization import BaseRetrievalExtractor` →
  `from collab_splats.localization import BaseRetrievalExtractor`
- `collab_splats/webapp/routers/localize.py:79` — inline import → new path
- `evals/runners/compare_slam_ours.py:45` → new path

Docs:
- `docs/source/tutorials/07_localization/localization.ipynb` — TWO cells:
  line ~101 `from collab_splats.pointcloud.localization import CameraLocalizer,
  XFeatExtractor, plot_correspondences` and line ~501
  `from collab_splats.pointcloud import LomaExtractor` — both →
  `collab_splats.localization`
- `third_party/README.md:24,32` — prose references to
  `collab_splats/pointcloud/localization.py` (xfeat pattern pointer) → new path
- `CLAUDE.md` architecture tree — move the `localization.py` line out of
  `pointcloud/` into a top-level `localization/` package entry
- If `docs/` has a module page for pointcloud localization, move/retitle it to
  mirror the new tree (check `docs/source/` toctree)

Tests — move to mirror the new package (`tests/` mirrors `collab_splats/`):
- `tests/pointcloud/test_localization.py` → `tests/localization/test_localizer.py`
- `tests/pointcloud/test_localization_cache.py` → `tests/localization/test_localization_cache.py`
- `tests/pointcloud/test_retrieval.py` → `tests/localization/test_retrieval.py`
- `tests/pointcloud/test_loma_extractor.py` → `tests/localization/test_loma_extractor.py`

Test content updates:
- All `from collab_splats.pointcloud.localization import ...` → `from collab_splats.localization import ...`
- `test_loma_extractor.py:59-60` — package-level `from collab_splats.pointcloud import LomaExtractor` → `from collab_splats.localization import LomaExtractor`
- `test_localization.py` `test_module_has_logger` — asserts `logger` on the module
  object; after the split assert each submodule (`retrieval`, `extractors`,
  `localizer`, `viz`) has its own `logging.Logger`
- `_build_frame_assignments` imports → `from collab_splats.localization.localizer import _build_frame_assignments`
- `tests/test_cu121_migration.py:110` — module string
  `"collab_splats.pointcloud.localization"` → `"collab_splats.localization"`

Confirmed NOT needing updates (repo-wide sweep 2026-07-09, all file types):
- `collab_splats/__init__.py` — no localization re-exports
- `dashboard/`, `nerfstudio/`, `scripts/`, `setup/` — zero references
- `evals/results/*.txt` — gitignored test logs (historical tracebacks)
- `docs/_build/` — Sphinx-generated, rebuilt on next docs build
- `collab_splats.egg-info/SOURCES.txt` — generated
- `.claude/worktrees/lc-parity/` — separate worktree on another branch; gets the
  change via merge, do not touch
- `docs/superpowers/plans/*.md` — historical planning docs, stay as written

## ⚠️ mock.patch targets (the one real hazard, carried over from the handoff)

Patching a re-export does NOT affect the name the defining module resolves at call
time. Update to the defining submodule:
- `tests/pointcloud/test_retrieval.py:47,73` —
  `patch("collab_splats.pointcloud.localization.open_clip")` →
  `patch("collab_splats.localization.retrieval.open_clip")`
- `tests/pointcloud/test_feedforward_lc_state.py:53` —
  `patch("collab_splats.pointcloud.localization.BaseRetrievalExtractor.get")` →
  `patch("collab_splats.localization.retrieval.BaseRetrievalExtractor.get")`
  (class-attribute patch keeps working regardless — update for consistency; this
  file stays in `tests/pointcloud/` since it tests LC state, not localization)

Grep tests for any other `patch("collab_splats.pointcloud.localization` before
starting.

## Constraints (carried over from the handoff)

- **Move, don't rewrite.** Class/function bodies byte-identical; keep `########`
  dividers where they still make sense; keep the `LomaExtractor` `_descriptor`
  private-access comment with the code; do NOT merge LomaExtractor/LomaGExtractor
  (registry reverse-lookup for zarr cache keys needs distinct classes).
- Repo style: imports at top of each submodule; no inline imports.
- **Concurrent sessions:** other Claude sessions commit to this branch. Stage ONLY
  files this refactor touches; dirty `collab_splats/mesh/` + notebook files are
  another session's — leave them. Confirm `git status --porcelain
  collab_splats/pointcloud/localization.py` is clean before starting.
- `docs/superpowers/` is gitignored — commit docs there with `git add -f`.
- Python is `/opt/venv/reconstruction/bin/python` (py3.11). Never bare `python`.

## Verification

1. `/opt/venv/reconstruction/bin/python -m pytest tests/localization/ tests/pointcloud/ -m "not slow"`
   — green vs pre-change baseline (known failures: `docs/known-test-failures.md`).
2. `/opt/venv/reconstruction/bin/python -m pytest tests/test_cu121_migration.py` —
   module-import sweep passes with the new package.
3. Registry smoke:
   `python -c "from collab_splats.localization import BaseLocalExtractor, BaseRetrievalExtractor; print(sorted(BaseLocalExtractor._registry), sorted(BaseRetrievalExtractor._registry))"`
   → `['disk', 'loma', 'loma-g', 'xfeat'] ['dino-salad', 'pe-clip']`
4. Stale-path guard:
   `grep -rn "pointcloud.localization" collab_splats/ tests/ evals/ docs/source/` → no hits.
5. Webapp import: `python -c "import collab_splats.webapp.routers.localize"`.
6. Slow path (weights cached at /workspace/models/hub, GPU):
   `pytest tests/localization/test_loma_extractor.py -m slow` — 2 pass.

## Deliverable

Single commit: `refactor(localization): promote pointcloud/localization.py to top-level stage package`
containing the new package, deletions, all call-site/test/notebook updates, and the
CLAUDE.md tree edit. Nothing else.
