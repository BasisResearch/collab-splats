# Feature Transfer: Pointcloud → Mesh Vertices

**Date:** 2026-05-29
**Status:** Approved

## Problem

`lift_features` lifts 2D semantic features to per-point `(P, D)` features on the pointcloud. `features2vertex` KNN-transfers those to mesh vertices `(M, D)`. Both functions exist. Neither is wired into the feedforward mesh notebook, so similarity queries can only run on the raw pointcloud — not on the mesh.

## Goal

Apply the existing feature transfer pipeline in the feedforward mesh notebook so that similarity queries work against either the pointcloud or the mesh.

## Approach

**Path A (now):** pointcloud → mesh via KNN (`features2vertex`). Both representations share world-space, so KNN proximity is valid.

**Path B (future migration):** direct image projection onto mesh vertices — reimplement `lift_features` logic targeting vertices instead of pointcloud points. Better for dense meshes where vertices outnumber cloud points.

## Design

### New function: `transfer_features_to_mesh` in `mesh/utils.py`

```python
def transfer_features_to_mesh(
    result: FeedforwardResult,
    mesh: o3d.geometry.TriangleMesh,
    *,
    k: int = 5,
    sdf_trunc: float = 0.03,
) -> np.ndarray:
    """Transfer per-point features from a FeedforwardResult to mesh vertices via KNN.

    Args:
        result:    FeedforwardResult with features (P, D) populated.
        mesh:      Open3D mesh whose vertices receive the features.
        k:         Neighbors for Gaussian-weighted aggregation.
        sdf_trunc: Truncation distance — points farther than this from their
                   nearest vertex are excluded.

    Returns:
        (M, D) float32 array of per-vertex features, index-aligned with mesh.vertices.
    """
    assert result.features is not None, (
        "result.features is None — call lift_features() first"
    )
    return features2vertex(
        np.asarray(mesh.vertices),
        result.points,
        result.features,
        k=k,
        sdf_trunc=sdf_trunc,
    )
```

Placed immediately after `features2vertex` in `mesh/utils.py`. No new imports needed — `o3d` and `np` already at module top.

### Data flow

```
result.image_paths
    → extractor.extract(images)           # List[(D, H_p, W_p)]
    → lift_features(feature_maps, result) # (P, D) tensor
    → result.features = lifted.numpy()    # mutate in notebook

mesh_result = pointcloud_to_mesh(result, output_dir)
mesh = o3d.io.read_triangle_mesh(str(mesh_result.mesh_path))

vertex_features = transfer_features_to_mesh(result, mesh)  # (M, D) ndarray
```

### Notebook additions: `feedforward_mesh.ipynb`

**§2 — Feature lifting** (new section, after existing mesh creation):
1. Instantiate extractor: `extractor = BaseFeatureExtractor.get("dinov2")()`
2. Extract per-frame maps from `result.image_paths`
3. `lifted = lift_features(feature_maps, result)` → `(P, D) tensor`
4. `result.features = lifted.numpy()`

**§3 — Transfer to mesh** (new section):
1. `mesh = o3d.io.read_triangle_mesh(str(mesh_result.mesh_path))`
2. `vertex_features = transfer_features_to_mesh(result, mesh)` → `(M, D)`

**§4 — Similarity demo** (new section):
1. Text or image query → encode → cosine similarity against `vertex_features`
2. Color mesh by similarity score, visualize with PyVista
3. Same query on `lifted` to show pointcloud and mesh give equivalent results

### No changes to

- `MeshResult` — no `features` field added
- `pointcloud_to_mesh` — no new params
- `FeedforwardResult` — field already exists, populated by caller
- `VGGTXCreator._postprocess` — still sets `features=None`

## Testing

New file: `tests/mesh/test_feature_transfer.py`

| Test | What it checks |
|------|---------------|
| `test_transfer_features_to_mesh_basic` | Synthetic result + toy mesh → output shape `(M, D)`, non-zero values |
| `test_transfer_features_to_mesh_none_raises` | `result.features = None` → `AssertionError` |
| `test_transfer_features_to_mesh_zero_points_in_range` | All points outside `sdf_trunc` → output all zeros |

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/mesh/utils.py` | Add `transfer_features_to_mesh` after `features2vertex` |
| `docs/source/tutorials/06_mesh/stage/feedforward_mesh.ipynb` | Add §2 feature lifting, §3 mesh transfer, §4 similarity demo |
| `tests/mesh/test_feature_transfer.py` | New — 3 tests |

## Future: Path B Migration

When mesh vertex density exceeds pointcloud density, the KNN aggregation in `features2vertex` will smear features. At that point, implement `lift_features_to_vertices(feature_maps, result, mesh)` in `mesh/utils.py` — same multi-view projection logic as `lift_features` but targeting `mesh.vertices` instead of `result.points`. `transfer_features_to_mesh` can then dispatch on a `method` param: `"knn"` (current) vs `"projection"` (Path B).
