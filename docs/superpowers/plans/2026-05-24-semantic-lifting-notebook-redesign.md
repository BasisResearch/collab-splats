# Semantic Lifting Notebook Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` to run end-to-end, showing MaskCLIP (DINOv2-regularized) and Talk2DINO semantic lifting side-by-side with a clean cache structure.

**Architecture:** Single notebook with two extractor pipelines. Each has its own cache-or-run block that writes to an extractor-named zarr file. A final comparison viewer links both point clouds. The broken multi-section cache guard is replaced with one `if/else` per extractor.

**Tech Stack:** Python 3.11, PyTorch, zarr+BloscCodec, PyVista, PIL, tqdm, `FeatureAutoencoder`, `MaskCLIPExtractor`, `DINOFeatureExtractor`, `Talk2DinoExtractor`, `lift_features`, `pointcloud_to_polydata`

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `docs/source/tutorials/tutorial_config.py` | Drop `REG_EXTRACTOR`, `REG_DIM`, `REG_WEIGHT` |
| Rewrite | `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` | Full notebook — dual extractor pipeline |

---

## Task 1: Clean tutorial_config.py

**Files:**
- Modify: `docs/source/tutorials/tutorial_config.py`

Current file has three lines to remove: `REG_EXTRACTOR`, `REG_DIM`, `REG_WEIGHT` and their comment block. They are never used now that reg is explicit in the notebook.

- [ ] **Step 1: Read the file**

Read `docs/source/tutorials/tutorial_config.py`. Confirm it contains `REG_EXTRACTOR`, `REG_DIM`, `REG_WEIGHT`.

- [ ] **Step 2: Remove the REG_* block**

The file should become exactly:

```python
from pathlib import Path

from collab_splats.utils.paths import get_cache_dir

DATASET = "birds_c0043"
MAX_FRAMES = 30  # cap for light tutorial runs

CACHE_DIR = get_cache_dir(DATASET)
IMAGES = CACHE_DIR / "images"
```

- [ ] **Step 3: Verify no other notebook uses REG_EXTRACTOR**

```bash
grep -r "REG_EXTRACTOR" /workspace/collab-splats/docs/
```

Expected: zero matches (only `tutorial_config.py` defined it, only `semantic_lifting.ipynb` consumed it — both are being changed).

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/tutorial_config.py
git commit -m "refactor(tutorials): drop REG_EXTRACTOR/REG_DIM/REG_WEIGHT from tutorial_config"
```

---

## Task 2: Rewrite notebook — opening cells (autoreload, imports, config)

**Files:**
- Rewrite: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`

This task rewrites the entire notebook from scratch using the Write tool. It is faster and cleaner than editing 19 cells individually. The full notebook JSON is written here; Tasks 3–7 verify section correctness and make targeted fixes if needed.

- [ ] **Step 1: Write the complete notebook**

Write the following content to `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`:

```json
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-autoreload",
   "metadata": {},
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-imports",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "\n",
    "import torch\n",
    "import zarr\n",
    "from zarr.codecs import BloscCodec\n",
    "from PIL import Image\n",
    "from tqdm.auto import tqdm\n",
    "import pyvista as pv\n",
    "import matplotlib\n",
    "matplotlib.use(\"Agg\") if os.environ.get(\"PYVISTA_OFF_SCREEN\") else None\n",
    "%matplotlib inline\n",
    "\n",
    "pv.set_jupyter_backend(\"static\" if os.environ.get(\"PYVISTA_OFF_SCREEN\") else \"trame\")\n",
    "\n",
    "from collab_splats.pointcloud.feedforward.base import FeedforwardResult\n",
    "from collab_splats.semantics.features import MaskCLIPExtractor, DINOFeatureExtractor, Talk2DinoExtractor\n",
    "from collab_splats.semantics.compression import FeatureAutoencoder\n",
    "from collab_splats.pointcloud.utils import lift_features\n",
    "from collab_splats.utils.visualization import pointcloud_to_polydata"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s0-md",
   "metadata": {},
   "source": [
    "## §0 — Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s0-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "%run ../tutorial_config.py\n",
    "\n",
    "# ── Configuration ─────────────────────────────────────────────────────────────\n",
    "METHOD         = \"vggtx\"    # \"vggtx\" | \"mapanything\"\n",
    "LATENT_DIM     = 13\n",
    "QUERY_POSITIVE = [\"tree\"]\n",
    "QUERY_NEGATIVE = [\"ground\"]\n",
    "DEVICE         = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "\n",
    "# Prefer BA reconstruction if available; fall back to raw feedforward result\n",
    "_ba   = CACHE_DIR / METHOD / \"ba\" / \"reconstruction.zarr\"\n",
    "_raw  = CACHE_DIR / METHOD / \"reconstruction.zarr\"\n",
    "RECON = _ba if _ba.exists() else _raw\n",
    "_base = RECON.parent  # directory that holds all lifted.zarr outputs for this variant\n",
    "\n",
    "# Extractor-named cache paths — never collide when switching extractors or methods\n",
    "LIFTED_MASKCLIP  = _base / \"lifted_maskclip.zarr\"\n",
    "LIFTED_TALK2DINO = _base / \"lifted_talk2dino.zarr\"\n",
    "\n",
    "assert RECON.exists(), (\n",
    "    f\"Reconstruction not found at {RECON}. \"\n",
    "    \"Run 02_pointcloud/feedforward_methods first.\"\n",
    ")\n",
    "print(f\"Device: {DEVICE}  |  RECON: {RECON}\")\n",
    "print(f\"BA: {'yes' if _ba.exists() else 'no — using raw feedforward result'}\")\n",
    "print(f\"MaskCLIP cache:  {LIFTED_MASKCLIP}\")\n",
    "print(f\"Talk2DINO cache: {LIFTED_TALK2DINO}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s1-md",
   "metadata": {},
   "source": [
    "## §1 — Load Reconstruction\n",
    "\n",
    "Loads the cached feedforward reconstruction from zarr. Produces a `FeedforwardResult` with\n",
    "world-space 3D points, per-point colors, pixel source indices, and source image paths.\n",
    "Run `02_pointcloud/feedforward_methods` first if the assertion above failed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s1-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load reconstruction from zarr cache\n",
    "out = FeedforwardResult.load_zarr(RECON)\n",
    "print(f\"Loaded reconstruction: {out.pts3d.shape[0]:,} pts  |  {len(out.image_paths)} frames\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s2-md",
   "metadata": {},
   "source": [
    "## §2 — Inspect Outputs\n",
    "\n",
    "Confirms the three arrays needed for feature lifting: `pts3d` (3D positions), `pixel_indices`\n",
    "(source frame/row/col per point), and `colors` (RGB). `pixel_indices` is the bridge between\n",
    "the 2D feature maps and the 3D point cloud."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s2-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "pts3d         = out.pts3d          # (P, 3) float32\n",
    "pixel_indices = out.pixel_indices  # (P, 3) int32  [frame_id, row, col]\n",
    "colors        = out.colors         # (P, 3) uint8\n",
    "\n",
    "print(f\"pts3d:         {pts3d.shape}\")\n",
    "print(f\"pixel_indices: {pixel_indices.shape}\")\n",
    "print(f\"colors:        {colors.shape}\")\n",
    "print(f\"image_paths:   {len(out.image_paths)} frames\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s3-md",
   "metadata": {},
   "source": [
    "## §3 — MaskCLIP: Extract, Compress & Lift\n",
    "\n",
    "Extracts MaskCLIP patch features from every source frame, then fits a `FeatureAutoencoder`\n",
    "to compress them from full-dim to `LATENT_DIM` before lifting onto the point cloud.\n",
    "\n",
    "MaskCLIP patch features are purely appearance-based — they lack structural grounding.\n",
    "A DINOv2 regularization branch during AE training improves the latent space geometry\n",
    "by pulling structurally similar patches together. Talk2DINO (§6) already carries CLIP\n",
    "grounding, so no regularization is needed there.\n",
    "\n",
    "**Cache hit:** `feat_mc` is loaded directly — extraction and compression are skipped."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s3-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load all source frames once — used by both extractor paths\n",
    "imgs       = [Image.open(p).convert(\"RGB\") for p in tqdm(out.image_paths, desc=\"Loading frames\")]\n",
    "image_size = (imgs[0].height, imgs[0].width)\n",
    "\n",
    "########################################################################\n",
    "\n",
    "if LIFTED_MASKCLIP.exists():\n",
    "    # Cache hit — load compressed-decoded features directly\n",
    "    _s     = zarr.open(str(LIFTED_MASKCLIP), mode=\"r\")\n",
    "    feat_mc = torch.from_numpy(np.asarray(_s[\"features\"][:]))\n",
    "    print(f\"Loaded from cache: {LIFTED_MASKCLIP}  shape={feat_mc.shape}\")\n",
    "else:\n",
    "    # Init MaskCLIP (semantics) and DINOv2 (AE regularization only)\n",
    "    maskclip = MaskCLIPExtractor(device=DEVICE)\n",
    "    dinov2   = DINOFeatureExtractor(device=DEVICE)\n",
    "\n",
    "    # Extract per-frame feature maps\n",
    "    mc_maps = maskclip.forward(imgs)  # list of (D_mc, H_p, W_p)\n",
    "    dv_maps = dinov2.forward(imgs)    # list of (384, H_p, W_p)\n",
    "\n",
    "    # Stack all patch vectors for AE training\n",
    "    D, H_p, W_p = mc_maps[0].shape\n",
    "    mc_patches = torch.cat([fm.permute(1, 2, 0).reshape(-1, D)   for fm in mc_maps])\n",
    "    dv_patches = torch.cat([fm.permute(1, 2, 0).reshape(-1, 384) for fm in dv_maps])\n",
    "\n",
    "    # Fit AE: MaskCLIP reconstruction + DINOv2 regularization branch\n",
    "    ae = FeatureAutoencoder(\n",
    "        input_dim=D, latent_dim=LATENT_DIM,\n",
    "        regularization_kwargs={\"branches\": {\"dinov2\": 384}, \"weight\": 0.1},\n",
    "    )\n",
    "    ae.fit(mc_patches.to(DEVICE), reg_targets={\"dinov2\": dv_patches.to(DEVICE)})\n",
    "\n",
    "    # Compress each frame map to LATENT_DIM\n",
    "    compressed_mc = [\n",
    "        ae.per_point_encode(fm.permute(1, 2, 0).reshape(-1, D).to(DEVICE))\n",
    "           .detach().cpu()\n",
    "           .reshape(H_p, W_p, LATENT_DIM)\n",
    "           .permute(2, 0, 1)\n",
    "        for fm in mc_maps\n",
    "    ]\n",
    "\n",
    "    # Lift compressed codes to 3D points, then decode to full-dim\n",
    "    codes_mc = lift_features(compressed_mc, out.pixel_indices, image_size=image_size)\n",
    "    feat_mc  = ae.per_point_decode(codes_mc.to(DEVICE)).detach().cpu()  # (P, D)\n",
    "\n",
    "    # Save to cache\n",
    "    LIFTED_MASKCLIP.parent.mkdir(parents=True, exist_ok=True)\n",
    "    _lz4 = BloscCodec(cname=\"lz4\")\n",
    "    _s   = zarr.open(str(LIFTED_MASKCLIP), mode=\"w\")\n",
    "    _s.create_array(\"features\", data=feat_mc.numpy(), chunks=feat_mc.shape, compressors=_lz4)\n",
    "    print(f\"Lifted & saved: {feat_mc.shape}  →  {LIFTED_MASKCLIP}\")\n",
    "\n",
    "print(f\"feat_mc: {feat_mc.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s4-md",
   "metadata": {},
   "source": [
    "## §4 — MaskCLIP: Text Queries → Per-Point Scores\n",
    "\n",
    "Scores each 3D point against the configured text queries. `score_queries` returns a\n",
    "contrastive softmax score in [0, 1] — higher means the point matches the positive queries.\n",
    "`compute_similarity` returns raw cosine similarities per query for multi-query inspection."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s4-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Init MaskCLIP extractor for text scoring (needed even on cache hit)\n",
    "maskclip_scorer = MaskCLIPExtractor(device=DEVICE)\n",
    "\n",
    "# Move features to device\n",
    "feat_mc_dev = feat_mc.to(DEVICE)\n",
    "\n",
    "# Contrastive score across all positive queries vs negative queries\n",
    "scores_mc = maskclip_scorer.score_queries(\n",
    "    feat_mc_dev, positive=QUERY_POSITIVE, negative=QUERY_NEGATIVE, temperature=0.05\n",
    ").detach().cpu().numpy()\n",
    "\n",
    "# Per-query cosine similarities (P, Q)\n",
    "per_query_mc = maskclip_scorer.compute_similarity(feat_mc_dev, QUERY_POSITIVE).detach().T.cpu().numpy()\n",
    "\n",
    "print(f\"scores_mc:    {scores_mc.shape}  min={scores_mc.min():.3f}  max={scores_mc.max():.3f}\")\n",
    "print(f\"per_query_mc: {per_query_mc.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s5-md",
   "metadata": {},
   "source": [
    "## §5 — MaskCLIP: Interactive 3D Viewer\n",
    "\n",
    "Attaches semantic scores and per-query arrays to the point cloud. Use the scalar selector\n",
    "in the side panel to switch between the RGB view and each semantic query."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s5-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "cloud_mc = pointcloud_to_polydata(\n",
    "    pts3d,\n",
    "    RGB=colors,\n",
    "    semantic=scores_mc,\n",
    "    **{q.replace(\" \", \"_\"): per_query_mc[:, i] for i, q in enumerate(QUERY_POSITIVE)},\n",
    ")\n",
    "\n",
    "pl = pv.Plotter(title=\"MaskCLIP Semantic Lifting\")\n",
    "pl.add_mesh(cloud_mc, scalars=\"semantic\", cmap=\"plasma\", point_size=2)\n",
    "pl.add_scalar_bar(\"semantic score\", fmt=\"%.2f\")\n",
    "pl.add_title(\"MaskCLIP + DINOv2 reg — switch scalars in side panel\", font_size=9)\n",
    "pl.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s6-md",
   "metadata": {},
   "source": [
    "## §6 — Talk2DINO: Extract, Compress & Lift\n",
    "\n",
    "Runs the same extract → compress → lift pipeline with Talk2DINO patch features.\n",
    "Talk2DINO is CLIP-grounded — its patch features already carry structural and semantic\n",
    "information from CLIP pre-training. No DINOv2 regularization branch is needed;\n",
    "the AE trains on Talk2DINO patches directly.\n",
    "\n",
    "**Cache hit:** `feat_t2d` is loaded directly — extraction and compression are skipped."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s6-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "if LIFTED_TALK2DINO.exists():\n",
    "    # Cache hit — load compressed-decoded features directly\n",
    "    _s      = zarr.open(str(LIFTED_TALK2DINO), mode=\"r\")\n",
    "    feat_t2d = torch.from_numpy(np.asarray(_s[\"features\"][:]))\n",
    "    print(f\"Loaded from cache: {LIFTED_TALK2DINO}  shape={feat_t2d.shape}\")\n",
    "else:\n",
    "    # Init Talk2DINO — no DINOv2 needed (already CLIP-grounded)\n",
    "    talk2dino = Talk2DinoExtractor(device=DEVICE)\n",
    "\n",
    "    # Extract per-frame feature maps\n",
    "    t2d_maps = talk2dino.forward(imgs)  # list of (D_t2d, H_p, W_p)\n",
    "\n",
    "    # Stack all patch vectors for AE training\n",
    "    D_t, H_p, W_p = t2d_maps[0].shape\n",
    "    t2d_patches = torch.cat([fm.permute(1, 2, 0).reshape(-1, D_t) for fm in t2d_maps])\n",
    "\n",
    "    # Fit AE: plain reconstruction, no regularization branch\n",
    "    ae_t2d = FeatureAutoencoder(input_dim=D_t, latent_dim=LATENT_DIM)\n",
    "    ae_t2d.fit(t2d_patches.to(DEVICE))\n",
    "\n",
    "    # Compress each frame map to LATENT_DIM\n",
    "    compressed_t2d = [\n",
    "        ae_t2d.per_point_encode(fm.permute(1, 2, 0).reshape(-1, D_t).to(DEVICE))\n",
    "              .detach().cpu()\n",
    "              .reshape(H_p, W_p, LATENT_DIM)\n",
    "              .permute(2, 0, 1)\n",
    "        for fm in t2d_maps\n",
    "    ]\n",
    "\n",
    "    # Lift compressed codes to 3D points, then decode to full-dim\n",
    "    codes_t2d = lift_features(compressed_t2d, out.pixel_indices, image_size=image_size)\n",
    "    feat_t2d  = ae_t2d.per_point_decode(codes_t2d.to(DEVICE)).detach().cpu()  # (P, D_t)\n",
    "\n",
    "    # Save to cache\n",
    "    LIFTED_TALK2DINO.parent.mkdir(parents=True, exist_ok=True)\n",
    "    _lz4 = BloscCodec(cname=\"lz4\")\n",
    "    _s   = zarr.open(str(LIFTED_TALK2DINO), mode=\"w\")\n",
    "    _s.create_array(\"features\", data=feat_t2d.numpy(), chunks=feat_t2d.shape, compressors=_lz4)\n",
    "    print(f\"Lifted & saved: {feat_t2d.shape}  →  {LIFTED_TALK2DINO}\")\n",
    "\n",
    "print(f\"feat_t2d: {feat_t2d.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s7-md",
   "metadata": {},
   "source": [
    "## §7 — Talk2DINO: Text Queries → Per-Point Scores\n",
    "\n",
    "Same query scoring as §4 but using the Talk2DINO text encoder. Both extractors implement\n",
    "`score_queries` and `compute_similarity` via `BaseQueryableExtractor`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s7-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Init Talk2DINO scorer (needed even on cache hit)\n",
    "talk2dino_scorer = Talk2DinoExtractor(device=DEVICE)\n",
    "\n",
    "feat_t2d_dev = feat_t2d.to(DEVICE)\n",
    "\n",
    "scores_t2d = talk2dino_scorer.score_queries(\n",
    "    feat_t2d_dev, positive=QUERY_POSITIVE, negative=QUERY_NEGATIVE, temperature=0.05\n",
    ").detach().cpu().numpy()\n",
    "\n",
    "per_query_t2d = talk2dino_scorer.compute_similarity(feat_t2d_dev, QUERY_POSITIVE).detach().T.cpu().numpy()\n",
    "\n",
    "print(f\"scores_t2d:    {scores_t2d.shape}  min={scores_t2d.min():.3f}  max={scores_t2d.max():.3f}\")\n",
    "print(f\"per_query_t2d: {per_query_t2d.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-s8-md",
   "metadata": {},
   "source": [
    "## §8 — Side-by-Side Comparison\n",
    "\n",
    "Renders both semantic point clouds in linked panels. Orbiting one view orbits both.\n",
    "The contrast between MaskCLIP (DINOv2-regularized) and Talk2DINO highlights the\n",
    "effect of extractor choice on 3D semantic quality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-s8-code",
   "metadata": {},
   "outputs": [],
   "source": [
    "cloud_t2d = pointcloud_to_polydata(\n",
    "    pts3d,\n",
    "    RGB=colors,\n",
    "    semantic=scores_t2d,\n",
    "    **{q.replace(\" \", \"_\"): per_query_t2d[:, i] for i, q in enumerate(QUERY_POSITIVE)},\n",
    ")\n",
    "\n",
    "pl = pv.Plotter(shape=(1, 2), title=\"MaskCLIP vs Talk2DINO — Semantic Lifting\")\n",
    "pl.subplot(0, 0)\n",
    "pl.add_mesh(cloud_mc.copy(), scalars=\"semantic\", cmap=\"plasma\", point_size=2)\n",
    "pl.add_title(\"MaskCLIP + DINOv2 reg\", font_size=10)\n",
    "pl.subplot(0, 1)\n",
    "pl.add_mesh(cloud_t2d.copy(), scalars=\"semantic\", cmap=\"plasma\", point_size=2)\n",
    "pl.add_title(\"Talk2DINO\", font_size=10)\n",
    "pl.link_views()\n",
    "pl.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "python3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify JSON is valid**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import json; json.load(open('docs/source/tutorials/05_lifting/semantic_lifting.ipynb')); print('valid')"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "refactor(tutorials): rewrite semantic_lifting notebook — dual extractor, fixed cache guard"
```

---

## Task 3: Verify §4 and §7 scorer init pattern

**Files:**
- Verify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`

The notebook initializes `maskclip_scorer` and `talk2dino_scorer` in the query cells (§4, §7) rather than inside the cache-or-run block. This is correct — they are always needed for scoring, cache hit or miss. But it means two initializations of each model if §3 was a cache miss (once in the `else` block, once in §4). This is acceptable for a tutorial (models are small) but should be verified intentional.

- [ ] **Step 1: Confirm no double-load on first run**

Read cells §3 and §4 of the written notebook and confirm:
- §3 `else` block initializes `maskclip` (used for `forward` only)
- §4 initializes `maskclip_scorer` (used for `score_queries`/`compute_similarity`)
- Both are `MaskCLIPExtractor(device=DEVICE)` — same instance type, separate variables

This is intentional: `maskclip` in §3 is ephemeral (goes out of scope after the block), `maskclip_scorer` in §4 is retained for scoring. If this is a concern (memory), the scorer can be hoisted to a single init cell. For now, tutorial clarity takes priority.

- [ ] **Step 2: No action needed unless memory is a concern**

No commit — verification only.

---

## Task 4: Execute notebook headlessly and verify

**Files:**
- Execute: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`

- [ ] **Step 1: Execute headlessly**

```bash
cd /workspace/collab-splats && PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=1800 \
  docs/source/tutorials/05_lifting/semantic_lifting.ipynb
```

Expected: exits 0, no `ERROR` or traceback in output.

- [ ] **Step 2: Verify cell outputs exist**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/05_lifting/semantic_lifting.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
empty = [i for i, c in enumerate(code_cells) if not c.get('outputs')]
print(f'{len(code_cells)} code cells, {len(empty)} empty outputs: {empty}')
"
```

Expected: 0 empty outputs.

- [ ] **Step 3: Verify cache files were written**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.utils.paths import get_cache_dir
base = get_cache_dir('birds_c0043')
for p in ['vggtx/ba/lifted_maskclip.zarr', 'vggtx/ba/lifted_talk2dino.zarr',
          'vggtx/lifted_maskclip.zarr', 'vggtx/lifted_talk2dino.zarr']:
    full = base / p
    if full.exists():
        print(f'FOUND: {full}')
"
```

Expected: at least one `lifted_maskclip.zarr` and one `lifted_talk2dino.zarr` printed.

- [ ] **Step 4: Verify re-run hits cache**

Run the execute command from Step 1 again. Should be faster (no GPU compute). Both §3 and §6 should print `Loaded from cache:` in their outputs.

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/05_lifting/semantic_lifting.ipynb'))
for c in nb['cells']:
    for o in c.get('outputs', []):
        text = ''.join(o.get('text', []))
        if 'Loaded from cache' in text:
            print(text.strip())
"
```

Expected: two `Loaded from cache:` lines — one for maskclip, one for talk2dino.

- [ ] **Step 5: Commit executed notebook**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "chore(tutorials): execute semantic_lifting notebook — outputs rendered"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Run end-to-end without errors | Task 4 |
| Fix missing extractor init | Task 2 (extractors init in §4/§7, always present) |
| Fix broken cache guard | Task 2 (single if/else per extractor) |
| MaskCLIP + DINOv2 reg pipeline | Task 2 (§3) |
| Talk2DINO plain pipeline | Task 2 (§6) |
| Extractor-named cache paths | Task 2 (§0 config) |
| Remove REG_* from tutorial_config.py | Task 1 |
| All imports at top | Task 2 (Cell 2) |
| §N section headers | Task 2 (all markdown cells) |
| Prose markdown cells for each section | Task 2 (all markdown cells) |
| Side-by-side comparison viewer | Task 2 (§8) |
| Cache-or-run re-run loads from cache | Task 4 Step 4 |
| python3 kernelspec | Task 2 (metadata block) |

**Placeholder scan:** No TBDs, no TODOs, no "implement later" patterns.

**Type consistency:**
- `feat_mc` defined in §3, used in §4 ✓
- `feat_t2d` defined in §6, used in §7 ✓  
- `cloud_mc` defined in §5, used in §8 ✓
- `cloud_t2d` defined in §8 ✓
- `imgs` and `image_size` defined before both cache-or-run blocks ✓
- `out.pixel_indices` used in both lift calls ✓
