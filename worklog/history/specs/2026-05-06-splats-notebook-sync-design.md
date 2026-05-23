# docs/splats notebook sync design

**Date:** 2026-05-06

## Problem

`docs/splats/` notebooks have two broken imports and one stale method name. These accumulated during the `refactor/core-modules` restructuring:
- `collab_splats/utils/pointcloud.py` was deleted; `clean_pcd` moved to `collab_splats/pointcloud/utils.py`
- `collab_splats/utils/mesh.py` was deleted; `mesh_clustering` moved to `collab_splats/mesh/utils.py`

## Scope

Fix notebooks to match current package layout. No shims, no re-exports — notebooks are authoritative consumers and should import from canonical paths.

## Changes

### 1. `docs/splats/visualization.ipynb` — cell 1

```python
# Before (broken)
from collab_splats.utils.pointcloud import clean_pcd

# After
from collab_splats.pointcloud.utils import clean_pcd
```

### 2. `docs/splats/create_mesh.ipynb` — cell 13

```python
# Before (broken)
from collab_splats.utils.mesh import mesh_clustering

# After
from collab_splats.mesh.utils import mesh_clustering
```

### 3. `docs/splats/derive_splats.ipynb` — cell 0 (markdown)

```
# Before (stale method name)
3. **visualize:** visualize splats via [ns-viewer](...)

# After
3. **viewer:** visualize splats via [ns-viewer](...)
```

## Non-changes

- `CAMERA_KWARGS`, `MESH_KWARGS`, `VIZ_KWARGS`, `visualize_splat` imports in `visualization.ipynb` are correct — still in `collab_splats.utils.visualization`
- `extract_features` docs are accurate — method runs `ns-train` internally
- `from_config_file` / `pointcloud_method` patterns deferred (out of scope)

## Verification

After edits: `jupyter nbconvert --to script` each notebook and verify import lines match.
