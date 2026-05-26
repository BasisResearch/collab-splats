# Spec: semantic_lifting Notebook Cleanup

**Date:** 2026-05-26  
**File:** `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`

---

## Goals

1. Fix broken API calls (`lift_features`, `zarr.open`)
2. Store compressed (LATENT_DIM) features + AE weights instead of full-dim decoded features
3. Replace raw `pv.Plotter` with `visualize_splat` + kwargs dicts
4. Drop the side-by-side subplot cell; replace with separate Talk2DINO viewer

---

## Issues Fixed

| # | Cell | Problem | Fix |
|---|------|---------|-----|
| 1 | §3, §6 | `lift_features(compressed, out.pixel_indices, image_size)` — old 3-arg API | `lift_features(compressed, out)` |
| 2 | §3, §6 | `zarr.open(str(path), mode)` — zarr v3 broke positional args | `zarr.open(store=str(path), mode=mode)` |
| 3 | §3, §6 | Storing decoded full-dim features (wasteful) | Store compressed (LATENT_DIM) + AE weights |
| 4 | §5, §8 | Raw `pv.Plotter` inline instead of module visualizer | `visualize_splat` with kwargs dicts |
| 5 | §0, §3, §6 | `image_size` variable computed + passed — now unused | Remove |

---

## Cache Layout

AE is trained on scene images + extractor only — not on the pointcloud. It lives at scene level, method-independent. Lifted features are reconstruction-specific (P = num points) and live next to the reconstruction.

```
CACHE_DIR/                              ← get_cache_dir(DATASET), scene-level
  semantics/
    maskclip/autoencoder.pt             ← AE, reusable across methods/reconstructions
    talk2dino/autoencoder.pt
  METHOD/
    reconstruction.zarr
    lifted_maskclip.zarr/features       ← (P, LATENT_DIM) float32, lz4
    lifted_talk2dino.zarr/features
    ba/
      reconstruction.zarr
      lifted_maskclip.zarr/features
      lifted_talk2dino.zarr/features
```

§0 path vars:
```python
SEMANTICS_DIR    = CACHE_DIR / "semantics"
AE_MASKCLIP      = SEMANTICS_DIR / "maskclip"
AE_TALK2DINO     = SEMANTICS_DIR / "talk2dino"
LIFTED_MASKCLIP  = _base / "lifted_maskclip.zarr"   # unchanged name
LIFTED_TALK2DINO = _base / "lifted_talk2dino.zarr"  # unchanged name
```

Cache-hit check: `LIFTED_MASKCLIP.exists() and AE_MASKCLIP.exists()`.

---

## §0 Configuration Changes

- Remove `image_size` variable
- Add `SEMANTIC_MESH_KWARGS` at top (derived from imported `PCD_KWARGS`, overriding scalars/cmap); `VIZ_KWARGS` imported directly from visualization module

```python
# Semantic visualization kwargs — PCD_KWARGS base, override scalars + colormap
SEMANTIC_MESH_KWARGS = {**PCD_KWARGS, "scalars": "semantic", "cmap": "viridis", "rgb": False}
```

---

## §3 MaskCLIP: Extract, Compress & Lift

### Cache miss path
```python
# Compress each frame map, lift to 3D
compressed = [ae.encode(fm.to(DEVICE)).detach().cpu() for fm in mc_maps]
compressed_pt = lift_features(compressed, out)          # (P, LATENT_DIM)
feat_mc = ae.per_point_decode(compressed_pt.to(DEVICE)).detach().cpu()

# Save compressed features + AE inside the zarr dir
g = zarr.open(store=str(LIFTED_MASKCLIP), mode="w")
g.create_array("features", data=compressed_pt.numpy(), chunks=compressed_pt.shape, compressors=BloscCodec(cname="lz4"))
ae.save(LIFTED_MASKCLIP)
```

### Cache hit path
```python
ae = FeatureAutoencoder.load(LIFTED_MASKCLIP).to(DEVICE)
compressed_pt = torch.from_numpy(np.asarray(zarr.open(store=str(LIFTED_MASKCLIP), mode="r")["features"][:]))
feat_mc = ae.per_point_decode(compressed_pt.to(DEVICE)).detach().cpu()
```

Comment update: "Compress each frame map, lift to 3D" (drop "decode to full-dim" — decode happens for scoring, not storage).

---

## §5 MaskCLIP Visualization

Replace raw `pv.Plotter` block with:

```python
pl = visualize_splat(cloud_mc, mesh_kwargs=SEMANTIC_MESH_KWARGS, viz_kwargs=VIZ_KWARGS)
pl.show()
```

Import `visualize_splat` in the imports cell.

---

## §6 Talk2DINO: Extract, Compress & Lift

Same changes as §3:
- `lift_features(compressed, out)` (drop old 3-arg form)
- zarr write/read use `store=`, `mode=` kwargs
- Store compressed + `ae_t2d.save(LIFTED_TALK2DINO)`
- Cache hit: `FeatureAutoencoder.load(LIFTED_TALK2DINO)`

---

## §8 Talk2DINO Visualization (replaces Side-by-Side)

Drop the `pv.Plotter(shape=(1,2))` subplot cell. Replace with single Talk2DINO viewer:

```python
cloud_t2d = pointcloud_to_polydata(
    pts3d,
    RGB=colors,
    semantic=scores_t2d,
    **{q.replace(" ", "_"): per_query_t2d[:, i] for i, q in enumerate(QUERY_POSITIVE)},
)

pl = visualize_splat(cloud_t2d, mesh_kwargs=SEMANTIC_MESH_KWARGS, viz_kwargs=VIZ_KWARGS)
pl.show()
```

Update markdown cell: "§8 — Talk2DINO: Interactive 3D Viewer".

---

## Imports Cell

```python
from collab_splats.utils.visualization import (
    pointcloud_to_polydata,
    visualize_splat,
    PCD_KWARGS,
    VIZ_KWARGS,
)
```

---

## Out of Scope

- No changes to `FeatureAutoencoder`, `lift_features`, or visualization module
- No changes to §1, §2, §4, §7 (load, inspect, scoring cells — unchanged)
- No new tests (notebook-only change)
