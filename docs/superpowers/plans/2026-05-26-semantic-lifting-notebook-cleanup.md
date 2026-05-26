# Semantic Lifting Notebook Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix broken API calls, switch to compressed feature storage, and use the visualization module in `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`.

**Architecture:** Pure notebook edit — no library changes. Fix zarr v3 API, update `lift_features` call to new signature, split cache into scene-level AE and reconstruction-level lifted features, replace raw `pv.Plotter` blocks with `visualize_splat` using imported kwargs.

**Tech Stack:** zarr v3, `FeatureAutoencoder` (save/load), `lift_features`, `visualize_splat`, `PCD_KWARGS`/`VIZ_KWARGS` from `collab_splats.utils.visualization`.

---

## File Map

- **Modify:** `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`
  - `cell-imports` — add `visualize_splat`, `PCD_KWARGS`, `VIZ_KWARGS`
  - `cell-s0-code` — add `SEMANTICS_DIR`, `AE_MASKCLIP`, `AE_TALK2DINO`, `SEMANTIC_MESH_KWARGS`; remove `image_size`
  - `cell-s3-code` — new `lift_features` API, zarr v3 API, compressed storage + AE save/load
  - `cell-s5-code` — `visualize_splat` with kwargs (drop raw `pv.Plotter`)
  - `cell-s6-code` — same fixes as §3
  - `cell-s8-md` — rename to "§8 — Talk2DINO: Interactive 3D Viewer"
  - `cell-s8-code` — single `visualize_splat` for Talk2DINO (drop side-by-side subplot)

> **No unit tests:** all changes are in notebook cells. Verification = running each cell and confirming output. The library code (`lift_features`, `FeatureAutoencoder`, `visualize_splat`) is unchanged.

---

### Task 1: Update imports cell

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` — `cell-imports`

- [ ] **Step 1: Edit the imports cell**

Replace the visualization import line:
```python
from collab_splats.utils.visualization import pointcloud_to_polydata
```
with:
```python
from collab_splats.utils.visualization import (
    pointcloud_to_polydata,
    visualize_splat,
    PCD_KWARGS,
    VIZ_KWARGS,
)
```

Use `NotebookEdit` with `cell_id="cell-imports"` and the full updated cell source.

- [ ] **Step 2: Verify cell runs without ImportError**

Run `cell-imports`. Expected: no output, no errors.

---

### Task 2: Update §0 configuration cell

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` — `cell-s0-code`

- [ ] **Step 1: Edit §0 config cell**

Replace the full cell with:
```python
%run ../tutorial_config.py

# ── Configuration ─────────────────────────────────────────────────────────────
METHOD         = "vggtx"    # "vggtx" | "mapanything"
LATENT_DIM     = 13
QUERY_POSITIVE = ["tree"]
QUERY_NEGATIVE = ["ground"]
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Prefer BA reconstruction if available; fall back to raw feedforward result
_ba   = CACHE_DIR / METHOD / "ba" / "reconstruction.zarr"
_raw  = CACHE_DIR / METHOD / "reconstruction.zarr"
RECON = _ba if _ba.exists() else _raw
_base = RECON.parent  # directory that holds lifted.zarr outputs for this reconstruction

# Scene-level semantics dir — AE weights are method-independent, reused across reconstructions
SEMANTICS_DIR    = CACHE_DIR / "semantics"
AE_MASKCLIP      = SEMANTICS_DIR / "maskclip"
AE_TALK2DINO     = SEMANTICS_DIR / "talk2dino"

# Extractor-named cache paths — never collide when switching extractors or methods
LIFTED_MASKCLIP  = _base / "lifted_maskclip.zarr"
LIFTED_TALK2DINO = _base / "lifted_talk2dino.zarr"

# Semantic visualization kwargs — extend PCD_KWARGS with scalar coloring
SEMANTIC_MESH_KWARGS = {**PCD_KWARGS, "scalars": "semantic", "cmap": "viridis", "rgb": False}

assert RECON.exists(), (
    f"Reconstruction not found at {RECON}. "
    "Run 02_pointcloud/feedforward_methods first."
)
print(f"Device: {DEVICE}  |  RECON: {RECON}")
print(f"BA: {'yes' if _ba.exists() else 'no — using raw feedforward result'}")
print(f"MaskCLIP cache:  {LIFTED_MASKCLIP}")
print(f"Talk2DINO cache: {LIFTED_TALK2DINO}")
print(f"AE MaskCLIP:     {AE_MASKCLIP}")
print(f"AE Talk2DINO:    {AE_TALK2DINO}")
```

- [ ] **Step 2: Run §0 and verify**

Expected output: device, RECON path, BA status, all four cache paths printed. No errors.

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "refactor(lifting): update imports and config — add semantics dir layout"
```

---

### Task 3: Rewrite §3 MaskCLIP extract/compress/lift cell

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` — `cell-s3-code`

Key changes:
- Remove `image_size` line
- Cache hit: `LIFTED_MASKCLIP.exists() and AE_MASKCLIP.exists()`
- Cache miss: `lift_features(compressed, out)` (new 2-arg API), store compressed (LATENT_DIM) not decoded
- zarr write: `zarr.open_group(store=str(...), mode="w")`
- zarr read: `zarr.open_group(store=str(...), mode="r")`
- AE saved to `AE_MASKCLIP`, loaded from `AE_MASKCLIP`
- Rename `ae` → `ae_mc` throughout (avoids name collision with §6's `ae_t2d`)

- [ ] **Step 1: Edit §3 cell**

```python
# Load frames once — reused in §6
imgs = [Image.open(p).convert("RGB") for p in tqdm(out.image_paths, desc="Loading frames")]

# Init MaskCLIP — used for both extraction and scoring in §4
maskclip = MaskCLIPExtractor(device=DEVICE)

########################################################################

if LIFTED_MASKCLIP.exists() and AE_MASKCLIP.exists():
    ae_mc = FeatureAutoencoder.load(AE_MASKCLIP).to(DEVICE)
    compressed_mc = torch.from_numpy(np.asarray(zarr.open_group(store=str(LIFTED_MASKCLIP), mode="r")["features"][:]))
    feat_mc = ae_mc.per_point_decode(compressed_mc.to(DEVICE)).detach().cpu()
    print(f"Loaded from cache: {compressed_mc.shape} → decoded {feat_mc.shape}")
else:
    # DINOv2 only needed for AE regularization; init here to skip on cache hit
    dinov2 = DINOFeatureExtractor(device=DEVICE)

    mc_maps = maskclip.forward(imgs)  # list of (D, H_p, W_p)
    dv_maps = dinov2.forward(imgs)    # list of (384, H_p, W_p)

    # Train AE: MaskCLIP reconstruction + DINOv2 regularization branch
    D = mc_maps[0].shape[0]
    ae_mc = FeatureAutoencoder(D, LATENT_DIM, regularization_kwargs={"branches": {"dinov2": 384}, "weight": 0.1})
    ae_mc.fit(
        torch.cat([fm.flatten(1).T for fm in mc_maps]).to(DEVICE),
        reg_targets={"dinov2": torch.cat([fm.flatten(1).T for fm in dv_maps]).to(DEVICE)},
    )

    # Compress each frame map, lift to 3D
    compressed = [ae_mc.encode(fm.to(DEVICE)).detach().cpu() for fm in mc_maps]
    compressed_mc = lift_features(compressed, out)          # (P, LATENT_DIM)
    feat_mc = ae_mc.per_point_decode(compressed_mc.to(DEVICE)).detach().cpu()

    # Save compressed features and AE weights
    g = zarr.open_group(store=str(LIFTED_MASKCLIP), mode="w")
    g.create_array("features", data=compressed_mc.numpy(), chunks=compressed_mc.shape, compressors=BloscCodec(cname="lz4"))
    ae_mc.save(AE_MASKCLIP)
    print(f"Saved: compressed {compressed_mc.shape} → {LIFTED_MASKCLIP}")

print(f"feat_mc: {feat_mc.shape}")
```

- [ ] **Step 2: Run §3 and verify**

Cache miss expected on first run. Expected output: AE training progress bars, then `Saved: compressed (P, 13) → ...`. Shape should be `(P, D_maskclip)` for `feat_mc` after decode. No `TypeError`.

- [ ] **Step 3: Run §3 again (cache hit path)**

Delete nothing. Re-run cell. Expected: `Loaded from cache: torch.Size([P, 13]) → decoded torch.Size([P, D_maskclip])`.

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "refactor(lifting): §3 — compressed storage, zarr v3 API, lift_features new signature"
```

---

### Task 4: Rewrite §5 MaskCLIP visualization cell

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` — `cell-s5-code`

- [ ] **Step 1: Edit §5 cell**

```python
cloud_mc = pointcloud_to_polydata(
    pts3d,
    RGB=colors,
    semantic=scores_mc,
    **{q.replace(" ", "_"): per_query_mc[:, i] for i, q in enumerate(QUERY_POSITIVE)},
)

pl = visualize_splat(cloud_mc, mesh_kwargs=SEMANTIC_MESH_KWARGS, viz_kwargs=VIZ_KWARGS)
pl.show()
```

- [ ] **Step 2: Run §4 then §5**

Run §4 (scoring) first, then §5. Expected: interactive 3D viewer renders with viridis semantic coloring. No `pv.Plotter` instantiation inline.

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "refactor(lifting): §5 — use visualize_splat with SEMANTIC_MESH_KWARGS"
```

---

### Task 5: Rewrite §6 Talk2DINO extract/compress/lift cell

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` — `cell-s6-code`

Same pattern as Task 3 but for Talk2DINO (no DINOv2 regularization branch).

- [ ] **Step 1: Edit §6 cell**

```python
# Init Talk2DINO — used for both extraction and scoring in §7
talk2dino = Talk2DinoExtractor(device=DEVICE)

########################################################################

if LIFTED_TALK2DINO.exists() and AE_TALK2DINO.exists():
    ae_t2d = FeatureAutoencoder.load(AE_TALK2DINO).to(DEVICE)
    compressed_t2d = torch.from_numpy(np.asarray(zarr.open_group(store=str(LIFTED_TALK2DINO), mode="r")["features"][:]))
    feat_t2d = ae_t2d.per_point_decode(compressed_t2d.to(DEVICE)).detach().cpu()
    print(f"Loaded from cache: {compressed_t2d.shape} → decoded {feat_t2d.shape}")
else:
    t2d_maps = talk2dino.forward(imgs)  # list of (D_t, H_p, W_p)

    # Train plain AE — no regularization needed (CLIP-grounded)
    D_t = t2d_maps[0].shape[0]
    ae_t2d = FeatureAutoencoder(D_t, LATENT_DIM)
    ae_t2d.fit(torch.cat([fm.flatten(1).T for fm in t2d_maps]).to(DEVICE))

    # Compress each frame map, lift to 3D
    compressed = [ae_t2d.encode(fm.to(DEVICE)).detach().cpu() for fm in t2d_maps]
    compressed_t2d = lift_features(compressed, out)         # (P, LATENT_DIM)
    feat_t2d = ae_t2d.per_point_decode(compressed_t2d.to(DEVICE)).detach().cpu()

    # Save compressed features and AE weights
    g = zarr.open_group(store=str(LIFTED_TALK2DINO), mode="w")
    g.create_array("features", data=compressed_t2d.numpy(), chunks=compressed_t2d.shape, compressors=BloscCodec(cname="lz4"))
    ae_t2d.save(AE_TALK2DINO)
    print(f"Saved: compressed {compressed_t2d.shape} → {LIFTED_TALK2DINO}")

print(f"feat_t2d: {feat_t2d.shape}")
```

- [ ] **Step 2: Run §6 and verify**

Expected: training progress, `Saved: compressed (P, 13) → ...`. Re-run for cache hit.

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "refactor(lifting): §6 — compressed storage, zarr v3 API, lift_features new signature"
```

---

### Task 6: Replace §8 side-by-side with Talk2DINO single viewer

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` — `cell-s8-md`, `cell-s8-code`

- [ ] **Step 1: Update §8 markdown cell**

Replace cell source with:
```markdown
## §8 — Talk2DINO: Interactive 3D Viewer

Renders Talk2DINO semantic scores on the point cloud. Use the scalar selector to switch
between RGB and per-query views. Compare visually with the MaskCLIP result in §5.
```

- [ ] **Step 2: Update §8 code cell**

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

- [ ] **Step 3: Run §7 then §8**

Run §7 (Talk2DINO scoring) then §8. Expected: single interactive 3D viewer with Talk2DINO viridis coloring. No subplot, no `pv.Plotter(shape=...)`.

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "refactor(lifting): §8 — replace side-by-side with Talk2DINO visualize_splat"
```

---

### Task 7: End-to-end verification

- [ ] **Step 1: Clear all outputs and restart kernel**

Kernel → Restart & Clear Output.

- [ ] **Step 2: Run all cells top-to-bottom**

Kernel → Run All. Monitor for any errors.

Expected checkpoints:
- §0: prints device, RECON path, all 4 cache paths
- §1: `Loaded reconstruction: N pts | M frames`
- §2: shapes printed for pts3d, pixel_indices, colors
- §3: cache miss → AE trains → `Saved: compressed (P, 13)`
- §4: `scores_mc` and `per_query_mc` shapes printed
- §5: 3D viewer renders
- §6: cache miss → AE trains → `Saved: compressed (P, 13)`
- §7: `scores_t2d` and `per_query_t2d` shapes printed
- §8: 3D viewer renders

- [ ] **Step 3: Confirm on-disk layout**

```python
# Run in a scratch cell to verify layout
import os
print(list((CACHE_DIR / "semantics").iterdir()))         # maskclip/, talk2dino/
print(list(AE_MASKCLIP.iterdir()))                        # autoencoder.pt
print(list(AE_TALK2DINO.iterdir()))                       # autoencoder.pt
print(list(LIFTED_MASKCLIP.iterdir()))                    # features (zarr array)
```

- [ ] **Step 4: Final commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "docs(lifting): verify semantic_lifting notebook end-to-end after cleanup"
```
