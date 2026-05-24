# Semantic Lifting Notebook — Redesign Spec

**Date:** 2026-05-24
**Branch:** `refactor/cu121`
**Supersedes:** `2026-05-21-semantic-lifting-notebook-design.md`

---

## Goal

Rewrite `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` to:

1. Run end-to-end without errors
2. Fix structural bugs in the current notebook (missing extractor init, broken cache guard)
3. Show two extractors (MaskCLIP + Talk2DINO) in one notebook, with motivated contrast
4. Encode extractor name in cache paths so outputs never collide
5. Remove dead `REG_EXTRACTOR` branching from `tutorial_config.py`

---

## Bugs Fixed

| Bug | Root cause | Fix |
|-----|-----------|-----|
| `NameError: extractor` in §3 | Extractor never instantiated | Add dedicated init cell |
| `NameError: feature_maps` in §4/§5 on cache hit | Cache guard only wraps §3; §4/§5 unguarded | Single `if/else` block over all extraction+compression+lift |
| "Jump to §6" instruction | Non-linear flow that doesn't actually work | Eliminated — replaced with real `if/else` |
| `%run tutorial_config.py` injects vars invisibly | `REG_EXTRACTOR`/`REG_DIM`/`REG_WEIGHT` defined off-screen | Inline config; drop `%run` |

---

## Cache Structure

Extractor name encoded in filename — outputs never collide when switching extractors or methods:

```
CACHE_DIR / METHOD / "ba" / "lifted_maskclip.zarr"
CACHE_DIR / METHOD / "ba" / "lifted_talk2dino.zarr"
```

Where `CACHE_DIR / METHOD / "ba"` is used when BA reconstruction exists, otherwise `CACHE_DIR / METHOD`.

---

## tutorial_config.py Changes

Remove `REG_EXTRACTOR`, `REG_DIM`, `REG_WEIGHT` — never used now that reg is explicit in the notebook. No replacement needed (extractors are hardcoded in the notebook for clarity).

---

## Notebook Structure

### Cell 1 — autoreload
```python
%load_ext autoreload
%autoreload 2
```

### Cell 2 — imports (all at top, no exceptions)
```python
import os
import numpy as np
from pathlib import Path

import torch
import zarr
from zarr.codecs import BloscCodec
from PIL import Image
from tqdm.auto import tqdm
import pyvista as pv
import matplotlib
matplotlib.use("Agg") if os.environ.get("PYVISTA_OFF_SCREEN") else None
%matplotlib inline

pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.semantics.features import MaskCLIPExtractor, DINOFeatureExtractor, Talk2DinoExtractor
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.pointcloud.utils import lift_features
from collab_splats.utils.visualization import pointcloud_to_polydata
```

### Cell 3 — §0 Configuration (pure assignments, zero imports)
```python
# ── Configuration ─────────────────────────────────────────────────────────────
METHOD         = "vggtx"       # "vggtx" | "mapanything"
LATENT_DIM     = 13
QUERY_POSITIVE = ["tree"]
QUERY_NEGATIVE = ["ground"]
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Prefer BA reconstruction if available, fall back to raw feedforward result
%run ../tutorial_config.py   # provides CACHE_DIR
_ba   = CACHE_DIR / METHOD / "ba" / "reconstruction.zarr"
_raw  = CACHE_DIR / METHOD / "reconstruction.zarr"
RECON = _ba if _ba.exists() else _raw
_base = RECON.parent

LIFTED_MASKCLIP  = _base / "lifted_maskclip.zarr"
LIFTED_TALK2DINO = _base / "lifted_talk2dino.zarr"

assert RECON.exists(), f"Reconstruction not found at {RECON}. Run 02_pointcloud/feedforward_methods first."
print(f"Device: {DEVICE}  |  RECON: {RECON}")
print(f"BA: {'yes' if _ba.exists() else 'no'}")
```

Note: `%run tutorial_config.py` is retained here only to get `CACHE_DIR` — the `REG_*` vars it no longer exports are not used.

---

## Section Map

| § | Title | Key code | Cache writes |
|---|-------|----------|-------------|
| 0 | Configuration | path resolution | — |
| 1 | Load Reconstruction | `FeedforwardResult.load_zarr` | — |
| 2 | Inspect Outputs | shape prints | — |
| 3 | MaskCLIP — Extract, Compress & Lift | cache-or-run block | `lifted_maskclip.zarr` |
| 4 | MaskCLIP — Text Queries | `score_queries` / `compute_similarity` | — |
| 5 | MaskCLIP — Interactive Viewer | `pv.Plotter` | — |
| 6 | Talk2DINO — Extract, Compress & Lift | cache-or-run block | `lifted_talk2dino.zarr` |
| 7 | Talk2DINO — Text Queries | `score_queries` / `compute_similarity` | — |
| 8 | Side-by-Side Comparison | `pv.Plotter(shape=(1, 2))` | — |

---

## §3 — MaskCLIP Detail

Preceding markdown cell (required prose):

> *MaskCLIP patch features are purely appearance-based — they lack structural grounding. A DINOv2 regularization branch during AE training improves the geometry of the latent space by pulling structurally similar patches together. Talk2DINO (§6) already carries CLIP grounding, so no regularization is needed there.*

Code structure (cache-or-run):
```python
imgs       = [Image.open(p).convert("RGB") for p in tqdm(out.image_paths, desc="Loading frames")]
image_size = (imgs[0].height, imgs[0].width)

if LIFTED_MASKCLIP.exists():
    _s      = zarr.open(str(LIFTED_MASKCLIP), mode="r")
    feat_mc = torch.from_numpy(np.asarray(_s["features"][:]))
    print(f"Loaded from cache: {LIFTED_MASKCLIP}  shape={feat_mc.shape}")
else:
    maskclip = MaskCLIPExtractor(device=DEVICE)
    dinov2   = DINOFeatureExtractor(device=DEVICE)

    mc_maps = maskclip.forward(imgs)   # list of (D_mc, H_p, W_p)
    dv_maps = dinov2.forward(imgs)     # list of (384, H_p, W_p)  — reg targets only

    D, H_p, W_p = mc_maps[0].shape
    mc_patches = torch.cat([fm.permute(1, 2, 0).reshape(-1, D) for fm in mc_maps])
    dv_patches = torch.cat([fm.permute(1, 2, 0).reshape(-1, 384) for fm in dv_maps])

    ae = FeatureAutoencoder(
        input_dim=D, latent_dim=LATENT_DIM,
        regularization_kwargs={"branches": {"dinov2": 384}, "weight": 0.1},
    )
    ae.fit(mc_patches.to(DEVICE), reg_targets={"dinov2": dv_patches.to(DEVICE)})

    compressed = [
        ae.per_point_encode(fm.permute(1, 2, 0).reshape(-1, D).to(DEVICE))
           .detach().cpu().reshape(H_p, W_p, LATENT_DIM).permute(2, 0, 1)
        for fm in mc_maps
    ]
    codes  = lift_features(compressed, out.pixel_indices, image_size=image_size)
    feat_mc = ae.per_point_decode(codes.to(DEVICE)).detach().cpu()

    _lz4 = BloscCodec(cname="lz4")
    _s   = zarr.open(str(LIFTED_MASKCLIP), mode="w")
    _s.create_array("features", data=feat_mc.numpy(), chunks=feat_mc.shape, compressors=_lz4)
    print(f"Lifted & saved: {feat_mc.shape}  →  {LIFTED_MASKCLIP}")
```

---

## §6 — Talk2DINO Detail

Preceding markdown cell (required prose):

> *Talk2DINO is CLIP-grounded — its patch features already carry structural and semantic information. No regularization branch is needed; the AE trains on raw Talk2DINO patches directly.*

Code structure mirrors §3 exactly, with:
- `Talk2DinoExtractor` only (no DINOv2)
- `FeatureAutoencoder(input_dim=D, latent_dim=LATENT_DIM)` — no `regularization_kwargs`
- `ae.fit(t2d_patches.to(DEVICE))` — no `reg_targets`
- Cache path: `LIFTED_TALK2DINO`

This visual brevity (shorter than §3) signals "same pattern, simpler case" without saying so explicitly.

---

## §8 — Side-by-Side Comparison

```python
pl = pv.Plotter(shape=(1, 2), title="MaskCLIP vs Talk2DINO — Semantic Lifting")
pl.subplot(0, 0)
pl.add_mesh(cloud_mc.copy(), scalars="semantic", cmap="plasma", point_size=2)
pl.add_title("MaskCLIP + DINOv2 reg", font_size=10)
pl.subplot(0, 1)
pl.add_mesh(cloud_t2d.copy(), scalars="semantic", cmap="plasma", point_size=2)
pl.add_title("Talk2DINO", font_size=10)
pl.link_views()
pl.show()
```

---

## Mandatory Markdown Cells

Every code section has a preceding markdown cell with 1–3 sentences covering:
- What the code does
- What it produces
- Why (role in pipeline)

§3 and §6 cells are specified above. All others follow the notebook-polish spec standard.

---

## Notebook Polish Conformance

Follows `2026-05-23-notebook-polish-design.md`:
- `## §N — Title` section headers throughout
- All imports in Cell 2 — zero imports elsewhere
- `########` dividers inside long code cells
- `python3` kernelspec (`/opt/conda/envs/nerfstudio/`)
- Headless execution: `PYVISTA_OFF_SCREEN=true jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800`

---

## Success Criteria

- Notebook runs top-to-bottom without error, cold start and warm cache
- `feat_mc.shape == (P, D_mc)` and `feat_t2d.shape == (P, D_t2d)` where P = point count
- `scores_mc` and `scores_t2d` both in `[0, 1]`
- `lifted_maskclip.zarr` and `lifted_talk2dino.zarr` both written to `_base/`
- Re-run loads both from cache (no recompute)
- §8 comparison viewer renders two linked panels
- No `REG_EXTRACTOR` / `REG_DIM` / `REG_WEIGHT` in `tutorial_config.py`
