# Feedforward Module Refactor — Design Spec
_2026-05-08_

## Context

`collab_splats/pointcloud/feedforward.py` is a 968-line flat file that mixes unrelated concerns: data types, abstract pipeline, COLMAP reconstruction utilities, two large monkey-patches, and two concrete creator classes. There is no internal organization, missing docstrings on abstract methods, repeated boilerplate (3×4→4×4 extrinsic conversion in both creators), `_verify_loop_candidate` raises `NotImplementedError` instead of being declared `@abstractmethod`, and a magic number (`518`) is inlined without a name.

Additionally, `_vggt.py` and `_mapanything.py` are orphaned siblings at `pointcloud/` level despite being feedforward-specific inference utilities.

The goal is to convert `feedforward.py` into a well-structured package where each file has a single purpose, every code block is headed by a comment explaining what it accomplishes, and the abstract interface is correctly declared.

**Zero behavioral changes.** All existing import paths remain valid via `__init__.py` re-exports. All tests pass unchanged.

---

## Final Layout

```
collab_splats/pointcloud/
  feedforward/                  ← NEW package (replaces feedforward.py)
    __init__.py                 re-exports only — preserves all public import paths
    base.py                     shared types · geometry helpers · COLMAP builders · abstract pipeline
    vggtx.py                    VGGT-X compat patch · inference utilities · VGGTXCreator
    mapanything.py              MapAnything compat patch · inference utilities · MapAnythingCreator

  # Deleted (contents moved into feedforward/)
  # _vggt.py
  # _mapanything.py
  # feedforward.py

  # Unchanged
  loop_closure/                 pipeline wrapper — not feedforward-specific
  bundle_adjustment.py          pipeline wrapper — not feedforward-specific
  wrappers.py
  utils.py                      pointcloud-level shared utilities (colmap_reconstruction_to_result lives here)
  sfm.py
  base.py
  __init__.py
```

---

## File-by-File Design

### `feedforward/__init__.py`
Re-exports everything that external code currently imports from `collab_splats.pointcloud.feedforward`. No logic.

```python
from .base import FeedforwardResult, BaseFeedforwardCreator
from .vggtx import VGGTXCreator, unproject_and_filter_points   # unproject re-exported for test patching
from .mapanything import MapAnythingCreator
```

**Why explicit re-export of `unproject_and_filter_points`:** tests patch it at
`collab_splats.pointcloud.feedforward.unproject_and_filter_points`. Re-exporting via `__init__` preserves that patch path regardless of which file defines the function.

---

### `feedforward/base.py`
Single purpose: shared machinery used by all feedforward creators.

Block structure (top to bottom, each block headed by a `# ──` comment):

```
# ── Imports ─────────────────────────────────────────────────────────────────

# ── Output type ─────────────────────────────────────────────────────────────
  FeedforwardResult        (dataclass — already well-documented, keep as-is)

# ── Geometry helpers ─────────────────────────────────────────────────────────
  _extrinsics_3x4_to_4x4  (NEW — extracted from both _postprocess methods)
  _raw_to_world_points     (existing — add docstring explaining subsample param)

# ── COLMAP reconstruction builders ───────────────────────────────────────────
  build_pycolmap_reconstruction             (moved from feedforward.py)
  _rescale_reconstruction_to_original_dimensions  (moved; deduplicate stage/ copies)

# ── Abstract pipeline ────────────────────────────────────────────────────────
  BaseFeedforwardCreator   (existing — fix abstract method declarations)
```

**Interface fixes in `BaseFeedforwardCreator`:**
- `_verify_loop_candidate` → add `@abstractmethod` decorator (currently raises `NotImplementedError` without being declared abstract — inconsistency)
- `_reproject_after_ba` → add to base with `raise NotImplementedError(...)` (currently duck-typed; undocumented interface)
- Both get proper docstrings explaining their contract

---

### `feedforward/vggtx.py`
Single purpose: everything VGGT-X specific.

Block structure:

```
# ── Imports ─────────────────────────────────────────────────────────────────

# ── Constants ────────────────────────────────────────────────────────────────
  VGGTX_IMG_LOAD_RESOLUTION: int = 518    (replaces magic number in _preprocess)

# ── VGGT-X compat patch ──────────────────────────────────────────────────────
  _patch_vggtx_compute_similarity         (moved from feedforward.py — already documented)
  # Inline comment on forward hook mechanism
  # Inline comment on why compute_similarity flag is needed (VGGT-X missing gate)

# ── Inference utilities ───────────────────────────────────────────────────────
  unproject_and_filter_points             (moved from pointcloud/_vggt.py)
  # Inline comment on confidence percentile filtering logic

# ── Creator ──────────────────────────────────────────────────────────────────
  VGGTXCreator                            (moved from feedforward.py)
  # Inline comments in _postprocess on extrinsic_global_4x4 LC override
```

---

### `feedforward/mapanything.py`
Single purpose: everything MapAnything specific.

Block structure:

```
# ── Imports ─────────────────────────────────────────────────────────────────

# ── MapAnything compat patch ──────────────────────────────────────────────────
  _patch_mapanything_torch_compat         (moved from feedforward.py — already documented)
  # Inline comment on linecache injection (why inspect.getsource works on patched fn)
  # Inline comment on sys.modules walk (why rebinding ic.fn alone is insufficient)

# ── Inference utilities ───────────────────────────────────────────────────────
  run_mapanything                         (moved from pointcloud/_mapanything.py)
  collect_pts3d_from_outputs              (moved from pointcloud/_mapanything.py)
  _reproject_mapanything                  (moved from pointcloud/_mapanything.py)

# ── Creator ──────────────────────────────────────────────────────────────────
  MapAnythingCreator                      (moved from feedforward.py)
```

---

## Inline Comment Policy

Each non-trivial block gets a `# ──` section header. Within blocks, inline comments explain **why**, not **what**, for non-obvious lines only:

| Location | Comment explains |
|----------|-----------------|
| `_raw_to_world_points` | `subsample=8` is a fast approximation for BA track extraction, not the final point cloud |
| `VGGTXCreator._postprocess` | `extrinsic_global_4x4` key: LC merged outputs carry deduped poses so result has one entry per input frame, not per submap window frame |
| `_patch_mapanything_torch_compat` | linecache injection makes stack traces / debuggers see the rewritten source |
| `_patch_mapanything_torch_compat` | sys.modules walk needed because `from ... import frustum_intersection_check` copies the reference into callers' namespaces |
| `_patch_vggtx_compute_similarity` | forward hook on last global attention block's QKV — captures k and q without extra forward pass |
| Both patches | `TEMPORARY BRIDGE` header already present — keep it |

---

## Stage Directory Deduplication

`stage/vggt_utils.py` and `stage/mapanything_utils.py` both contain verbatim copies of `_rescale_reconstruction_to_original_dimensions`. After moving the authoritative copy to `feedforward/base.py`, update both stage files to import from there:

```python
from collab_splats.pointcloud.feedforward.base import _rescale_reconstruction_to_original_dimensions
```

---

## What Does NOT Change

- `pointcloud/loop_closure/` — not feedforward-specific; stays at `pointcloud/` level
- `pointcloud/bundle_adjustment.py` — same
- `pointcloud/wrappers.py` — same
- `pointcloud/utils.py` — `colmap_reconstruction_to_result` stays here (shared by SfM + feedforward)
- All public import paths (preserved via `__init__.py`)
- All behavior (pure restructure + documentation)

---

## Verification

After implementation:

```bash
# All existing tests pass
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v

# Import paths work (test patch path preserved)
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward import (
    FeedforwardResult, BaseFeedforwardCreator,
    VGGTXCreator, MapAnythingCreator,
    unproject_and_filter_points,
)
print('all imports OK')
"

# No orphaned references to old sibling files
grep -r "from \._vggt import\|from \._mapanything import" collab_splats/pointcloud/*.py
# → should return nothing (those files deleted)
```

---

## Files Modified / Deleted / Created

| Action | Path |
|--------|------|
| **Create** | `collab_splats/pointcloud/feedforward/__init__.py` |
| **Create** | `collab_splats/pointcloud/feedforward/base.py` |
| **Create** | `collab_splats/pointcloud/feedforward/vggtx.py` |
| **Create** | `collab_splats/pointcloud/feedforward/mapanything.py` |
| **Delete** | `collab_splats/pointcloud/feedforward.py` |
| **Delete** | `collab_splats/pointcloud/_vggt.py` |
| **Delete** | `collab_splats/pointcloud/_mapanything.py` |
| **Modify** | `stage/vggt_utils.py` — import `_rescale_reconstruction_to_original_dimensions` from feedforward |
| **Modify** | `stage/mapanything_utils.py` — same |
