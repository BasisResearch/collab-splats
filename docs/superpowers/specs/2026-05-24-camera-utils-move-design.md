# camera_utils Move to nerfstudio Submodule

**Date:** 2026-05-24
**Status:** Design approved

## Problem

`collab_splats/utils/camera_utils.py` lives at the top-level `utils/` package but is consumed only by nerfstudio-side code. Its API operates on `nerfstudio.cameras.cameras.Cameras`, and its source comments cite gaussian-splatting / scaffold-gs-nerfstudio / dn-splatter / splatfacto-360 lineage. Placement misleads readers into believing it is general infrastructure.

## Investigation

Real importers (excluding `.claude/worktrees/`, `docs/_build/`, `stage/` build/staging artifacts):

| File | Symbol(s) |
|---|---|
| `collab_splats/nerfstudio/models/rade_gs.py:26` | `build_rotation` |
| `collab_splats/utils/__init__.py:1` | `ColmapCamera`, `convert_to_colmap_camera`, `depth_double_to_normal` (re-export) |
| `collab_splats/__init__.py:14` | `ColmapCamera` (lazy `__getattr__` re-export) |
| `tests/test_cu121_migration.py:124` | module path string (importability smoke test) |

No consumer outside the `nerfstudio/` submodule uses any symbol. `pointcloud/`, `mesh/`, `semantics/`, `dashboard/`, `wrapper/` are clean.

## Decision

Move file into the nerfstudio submodule. Drop top-level re-exports (class is nerfstudio-bound; external callers should import from the canonical location).

## Changes

1. `git mv collab_splats/utils/camera_utils.py collab_splats/nerfstudio/utils/camera_utils.py`
2. `collab_splats/nerfstudio/models/rade_gs.py:26` — update import:
   ```python
   from collab_splats.nerfstudio.utils.camera_utils import build_rotation
   ```
3. `collab_splats/utils/__init__.py` — delete line 1 (`from .camera_utils import ColmapCamera, convert_to_colmap_camera, depth_double_to_normal`); prune from any `__all__`.
4. `collab_splats/__init__.py` — remove the `ColmapCamera` branch from `__getattr__` and the `"ColmapCamera"` entry from `__all__`.
5. `tests/test_cu121_migration.py:124` — update path string to `"collab_splats.nerfstudio.utils.camera_utils"`.

## Verification

- `grep -rn camera_utils collab_splats/ tests/ docs/source/` — every hit references `collab_splats.nerfstudio.utils.camera_utils`.
- `/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_cu121_migration.py`
- Import smoke test:
  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.nerfstudio.utils.camera_utils import ColmapCamera, build_rotation, depth_double_to_normal"
  ```
- `/opt/conda/envs/nerfstudio/bin/python -m pytest tests/` (full suite, no regressions).

## Out of Scope

- `stage/*.ipynb` — staging notebooks; not part of shipped surface.
- `docs/_build/**` — Sphinx build artifacts; regenerated from sources.
- `.claude/worktrees/docs-site/**` — agent worktree; isolated branch.

## Risk

Breaking public-API change: `from collab_splats import ColmapCamera` no longer resolves. Accepted — `ColmapCamera` is nerfstudio-bound and consumers should import from the canonical submodule path.

## Follow-up

After this lands, audit remaining `collab_splats/utils/` modules with the same lens (nerfstudio-only vs. shared) — out of scope here.
