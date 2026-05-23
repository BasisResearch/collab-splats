# Notebook Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 15 tutorial notebooks for style consistency, zarr cache chain, and rendered outputs — user opens any notebook and sees results without re-running anything.

**Architecture:** Approach C — fix + execute each notebook before moving to the next, in dependency order. Chain notebooks write zarr artifacts consumed by later notebooks (keyframe_extraction → feedforward_methods → semantic_lifting → localization). Splatter-based notebooks (derive_splats, visualization, create_mesh, maskclip_vs_talk2dino) use existing trained model at `/workspace/fieldwork-data/birds/2024-02-06/SplatsSD`. `feature_extraction` and `segmentation` are ported from Splatter/ConfigLoader to direct cache-based loading.

**Tech Stack:** `/opt/conda/envs/nerfstudio/bin/python`, `jupyter nbconvert`, `zarr v3`, `pyvista` (conditional static/trame backend), `matplotlib` inline, `collab_splats` package.

---

## Hard Prerequisites

**Do not begin until all in-flight agent work has landed on `refactor/cu121`.**

- [ ] Run `git log --oneline -5` and confirm `vggtx-feedforward-refactor` and `cu121` migration work are committed.
- [ ] Confirm `/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4` exists.
- [ ] Confirm `docs/.cache/` dir exists or is creatable: `mkdir -p /workspace/collab-splats/docs/.cache/birds_c0043/images`.
- [ ] Confirm nerfstudio env is active: `/opt/conda/envs/nerfstudio/bin/python -c "import collab_splats; print('ok')"`.

---

## Canonical Cell Structure Reference

Every notebook is updated to match this structure. Refer back to this section for every task.

**Cell 0: Title markdown** — `# Title` with 2–4 sentence intro explaining the notebook's purpose and its place in the pipeline. Includes prerequisite links where applicable.

**Cell 1: Autoreload**
```python
%load_ext autoreload
%autoreload 2
```

**Cell 2: All imports** — one cell, all stdlib/third-party/collab_splats imports. `from pathlib import Path` goes here. PyVista backend conditional. `%matplotlib inline`. Nothing else.
```python
import os
import time
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv
%matplotlib inline

pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

from collab_splats.X import Y  # all collab_splats imports here
```

**Cell 3: Config block** — pure variable assignments, zero imports.
```python
# ── Configuration ─────────────────────────────────────────────────────────────
DATASET  = "birds_c0043"
METHOD   = "vggtx"       # "vggtx" | "mapanything"
VARIANT  = "ba"          # "ba" | "lc" | "" (raw baseline)

CACHE  = Path("../../.cache") / DATASET
IMAGES = CACHE / "images"
RECON  = CACHE / METHOD / VARIANT if VARIANT else CACHE / METHOD
```

**Every subsequent code section:**
- Preceded by a markdown cell: 1–3 sentences explaining what runs, what it produces, why.
- Uses `## §N — Title` section headers.
- Uses `########` dividers in code for major sub-sections.
- Has inline block comment per logical code block.

**Execution command** (same for every notebook, fill in path):
```bash
PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=1800 \
  docs/source/tutorials/<path/to/notebook.ipynb>
```

**Verification** (same pattern every task):
```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/<notebook>.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
with_output = [c for c in code_cells if c.get('outputs')]
errors = [c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
print(f'{len(with_output)}/{len(code_cells)} cells have output, {len(errors)} errors')
"
```
Expected: all (or all non-stub) code cells have output, 0 errors.

---

## Task 1: `01_preprocessing/keyframe_extraction`

**Notebook path:** `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`
**Current state:** 6/10 cells rendered. Imports in cell 1 (no autoreload first), config in cell 4 (mixed with code), no pyvista backend conditional, some markdown headers missing explanatory text.
**Cache writes:** `docs/.cache/birds_c0043/images/`, `docs/.cache/birds_c0043/frame_scores.json`

- [ ] **Fix: insert autoreload as cell 0 (new first code cell)**

  The current cell 1 has imports but no autoreload. Add a new code cell at position 1:
  ```python
  %load_ext autoreload
  %autoreload 2
  ```

- [ ] **Fix: consolidate imports into one cell**

  Current cell 1 has partial imports (`from pathlib import Path`, frame_sampling imports). Move ALL imports to one cell immediately after autoreload:
  ```python
  import os
  import json
  import numpy as np
  from pathlib import Path

  import matplotlib.pyplot as plt
  import pyvista as pv
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.utils.frame_sampling import (
      extract_video_frames,
      get_video_info,
      sample_frames_fps,
      sample_frames_optical_flow,
      score_all_frames,
  )
  from collab_splats.utils.visualization import plot_frame_grid, plot_selection, plot_frame_scores
  ```
  Remove any `from pathlib import Path` or other imports from later cells.

- [ ] **Fix: config cell — pure variable assignments only**

  The config block (currently in cell 4) has no imports, but ensure it reads:
  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  VIDEO_PATH = "/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4"
  DATASET    = "birds_c0043"

  CACHE  = Path("../../.cache") / DATASET
  IMAGES = CACHE / "images"
  FRAME_SCORES = CACHE / "frame_scores.json"
  CACHE.mkdir(parents=True, exist_ok=True)
  IMAGES.mkdir(parents=True, exist_ok=True)
  ```

- [ ] **Fix: section headers → `## §N — Title` format**

  Replace existing headers:
  - `## 1. Load Video` → `## §1 — Load Video`
  - `## 2. Score Frames` → `## §2 — Score Frames`
  - `## 3. Select Keyframes` → `## §3 — Select Keyframes`
  - `## 4. Visualize` → `## §4 — Visualize Selection`
  - `## 5. Export` → `## §5 — Export to Cache`

- [ ] **Fix: add markdown explanation cells before every code section**

  For each code section lacking a prose explanation (just a header), add 1–3 sentences. Example for the frame-scoring section:
  > *Scores every frame in the video using optical flow disparity and histogram similarity. Returns a `FrameScores` object with per-frame quality metrics used to select the most informative subset. Scores are saved to `frame_scores.json` in the cache so this step is skipped on re-open.*

- [ ] **Fix: cache-or-run guard for frame extraction**

  The `extract_video_frames` call should be guarded:
  ```python
  if not any(IMAGES.glob("*.jpg")):
      extract_video_frames(VIDEO_PATH, IMAGES)
      print(f"Extracted frames to {IMAGES}")
  else:
      print(f"Loaded {len(list(IMAGES.glob('*.jpg')))} cached frames from {IMAGES}")
  ```

- [ ] **Fix: cache-or-run guard for frame scoring**

  ```python
  if FRAME_SCORES.exists():
      import json
      scores = json.load(open(FRAME_SCORES))
      print(f"Loaded frame scores from cache ({len(scores['scores'])} frames)")
  else:
      scores = score_all_frames(VIDEO_PATH)
      import json
      json.dump(scores, open(FRAME_SCORES, "w"))
      print(f"Scored {len(scores['scores'])} frames → saved to {FRAME_SCORES}")
  ```

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=600 \
    docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
  ```

- [ ] **Verify**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json
  nb = json.load(open('docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb'))
  code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
  with_output = [c for c in code_cells if c.get('outputs')]
  errors = [c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f'{len(with_output)}/{len(code_cells)} cells have output, {len(errors)} errors')
  "
  ```
  Also verify cache was written: `ls docs/.cache/birds_c0043/images/ | head -5`

- [ ] **Commit**

  ```bash
  git add docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
  git commit -m "docs(notebooks): polish keyframe_extraction — style, cache guards, executed"
  ```

---

## Task 2: `02_pointcloud/feedforward_methods`

**Notebook path:** `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`
**Current state:** 0/11 cells rendered. Cache-or-run guards exist for vggtx + mapanything zarr. Style mostly good but imports need consolidation, no autoreload, PyVista backend missing.
**Cache reads:** `docs/.cache/birds_c0043/images/`
**Cache writes:** `docs/.cache/birds_c0043/vggtx/reconstruction.zarr`, `docs/.cache/birds_c0043/mapanything/reconstruction.zarr`

- [ ] **Fix: insert autoreload as cell 0**

  Add code cell at position 0 (before current first code cell):
  ```python
  %load_ext autoreload
  %autoreload 2
  ```

- [ ] **Fix: consolidate all imports into one cell**

  Find every `import` / `from X import Y` across all cells. Collect them into one cell immediately after autoreload:
  ```python
  import os
  import time
  import numpy as np
  from pathlib import Path

  import matplotlib.pyplot as plt
  import open3d as o3d
  import pyvista as pv
  import torch
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator
  from collab_splats.pointcloud.feedforward.base import FeedforwardResult
  from collab_splats.utils.visualization import pointcloud_to_polydata, visualize_splat
  ```
  Remove these imports from all other cells.

- [ ] **Fix: config cell — ensure no imports, add IMAGES guard**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  VIDEO_PATH = "/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4"
  DATASET    = "birds_c0043"

  CACHE  = Path("../../.cache") / DATASET
  IMAGES = CACHE / "images"

  assert IMAGES.exists() and any(IMAGES.glob("*.jpg")), (
      f"No images found in {IMAGES}. Run 01_preprocessing/keyframe_extraction first."
  )
  ```

- [ ] **Fix: section headers → `## §N — Title` format**

  Update all section headers to use `## §N — Title`.

- [ ] **Fix: add/improve markdown explanation cells**

  Ensure every code section has a prose explanation cell. The cache-or-run cells should explain:
  > *Loads a cached VGGT-X reconstruction from zarr if available, skipping GPU inference. If no cache exists, runs VGGT-X on the keyframes (~3–5 min on GPU) and saves the result. The zarr file is chunked by frame for efficient per-frame random access by downstream notebooks.*

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=1800 \
    docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
  ```

- [ ] **Verify**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json
  nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
  code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
  with_output = [c for c in code_cells if c.get('outputs')]
  errors = [c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f'{len(with_output)}/{len(code_cells)} cells have output, {len(errors)} errors')
  "
  # Also verify zarr caches written:
  ls docs/.cache/birds_c0043/vggtx/ && ls docs/.cache/birds_c0043/mapanything/
  ```

- [ ] **Commit**

  ```bash
  git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
  git commit -m "docs(notebooks): polish feedforward_methods — style, executed, zarr cache written"
  ```

---

## Task 3: `02_pointcloud/bundle_adjustment`

**Notebook path:** `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`
**Current state:** 3/5 cells rendered. Uses 7-scenes chess canned subset (symlinked from `data/7scenes/chess/seq-01/`). No PyVista. Config block present but `from pathlib import Path` is missing from top-level imports.
**Data requirement:** `data/7scenes/chess/seq-01/` must exist (download: `bash evals/download_7scenes.sh chess data/7scenes`).

- [ ] **Check 7-scenes data exists**

  ```bash
  ls /workspace/collab-splats/data/7scenes/chess/ 2>/dev/null | head -3 || echo "MISSING — run: bash evals/download_7scenes.sh chess data/7scenes"
  ```
  If missing, download before proceeding.

- [ ] **Fix: insert autoreload as cell 0, consolidate imports**

  Add autoreload cell, then one imports cell:
  ```python
  import os
  import shutil
  from pathlib import Path

  import matplotlib.pyplot as plt
  import numpy as np
  import torch
  %matplotlib inline

  from collab_splats.pointcloud.feedforward import VGGTXCreator
  from collab_splats.pointcloud.feedforward.base import FeedforwardResult
  from collab_splats.pointcloud.bundle_adjustment import (
      extract_tracks_vggsfm,
      run_bundle_adjustment,
  )
  ```

- [ ] **Fix: config block — zero imports, add CACHE for consistency**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET   = "birds_c0043"
  DEMO_DIR  = Path("/tmp/ba_demo_chess")   # temp dir with 5 canned chess frames

  CACHE = Path("../../.cache") / DATASET   # declared for consistency; BA uses DEMO_DIR
  ```

- [ ] **Fix: section headers and markdown explanations**

  Current headers `## Setup — ...`, `## Section A — ...`, `## Section B — ...` are good structure but need `§` prefix:
  - `## §0 — Setup: Canned Image Set`
  - `## §1 — Section A: High-Level Wrapper API`
  - `## §2 — Section B: Manual API`
  - `## §3 — Where to go next`

  Add prose explanation before each code block.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=600 \
    docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} have output\")
  "
  git add docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
  git commit -m "docs(notebooks): polish bundle_adjustment — style, fully executed"
  ```

---

## Task 4: `02_pointcloud/slam_loop_closure`

**Notebook path:** `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`
**Current state:** 12/14 cells rendered. Uses bicycle scene + MapAnything. Prerequisites section references external data — check what the 2 missing cells need.

- [ ] **Identify the 2 unrendered cells**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json
  nb=json.load(open('docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb'))
  for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and not c.get('outputs'):
      print(f'Cell {i}: {repr(\"\".join(c[\"source\"])[:300])}')
  "
  ```

- [ ] **Fix: insert autoreload, consolidate imports, add backend conditional**

  Add autoreload cell, then one imports cell with:
  ```python
  import os
  import numpy as np
  from pathlib import Path

  import matplotlib.pyplot as plt
  import pyvista as pv
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  # collab_splats imports — all here
  ```
  Move any scattered imports up.

- [ ] **Fix: config block, section headers, markdown explanations**

  Apply canonical config block. Update all section headers to `## §N — Title`. Add markdown explanation cells for any code section missing one.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=1800 \
    docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
  git commit -m "docs(notebooks): polish slam_loop_closure — style, fully executed"
  ```

---

## Task 5: `02_pointcloud/feedforward_mesh`

**Notebook path:** `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb`
**Current state:** 0/5 cells rendered (small notebook, 116 lines). Runs MapAnything inference directly with no cache guard. Should load from `mapanything/reconstruction.zarr` written by Task 2.

- [ ] **Fix: insert autoreload, consolidate imports**

  ```python
  import os
  import shutil
  import tempfile
  from pathlib import Path

  import open3d as o3d
  import pyvista as pv
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.pointcloud.feedforward import MapAnythingCreator
  from collab_splats.pointcloud.feedforward.base import FeedforwardResult
  from collab_splats.mesh.tsdf import TSDFMesher
  ```

- [ ] **Fix: config block + cache-load guard**

  Replace the current direct-inference cell with a cache-or-run guard:
  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET   = "birds_c0043"
  METHOD    = "mapanything"
  VARIANT   = ""

  CACHE  = Path("../../.cache") / DATASET
  IMAGES = CACHE / "images"
  RECON  = CACHE / METHOD

  _recon_cache = RECON / "reconstruction.zarr"
  assert _recon_cache.exists(), (
      f"No reconstruction cache at {_recon_cache}. Run 02_pointcloud/feedforward_methods first."
  )

  result = FeedforwardResult.load_zarr(_recon_cache)
  print(f"Loaded MapAnything reconstruction: {result.pts3d.shape[0]:,} pts, {result.extrinsics.shape[0]} frames")
  ```

- [ ] **Fix: section headers, markdown explanations for each step**

  Add `## §1 — Load Reconstruction`, `## §2 — Direct TSDF Meshing`, `## §3 — Inspect Mesh`.
  Prose before each code cell.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=600 \
    docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} have output\")
  "
  git add docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb
  git commit -m "docs(notebooks): polish feedforward_mesh — cache-load guard, style, executed"
  ```

---

## Task 6: `02_pointcloud/colmap_sfm` (stub — style only, no execution)

**Notebook path:** `docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb`
**Current state:** 58-line stub. No compute defined. Style fixes only; do not execute.

- [ ] **Fix: apply canonical structure**

  Add autoreload cell, imports cell (with `from pathlib import Path`, `import os`), config block with `DATASET`/`CACHE`/`IMAGES`. Add a clear markdown cell explaining this is a stub:
  > *This notebook demonstrates COLMAP structure-from-motion. It is a stub — COLMAP integration is in progress. Run `02_pointcloud/feedforward_methods` for the recommended feedforward reconstruction path.*

- [ ] **Fix: section headers and any existing headers**

  Apply `## §N — Title` format to any existing sections.

- [ ] **Commit (no execution)**

  ```bash
  git add docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb
  git commit -m "docs(notebooks): polish colmap_sfm stub — canonical structure, no execution"
  ```

---

## Task 7: `03_splats/derive_splats`

**Notebook path:** `docs/source/tutorials/03_splats/derive_splats.ipynb`
**Current state:** 5/7 cells rendered. Splatter-based (COLMAP + training). Uses `Splatter` / `SplatterConfig` correctly — this is NOT ported. Cells 3 and 7 have no output. Cell 13 (`splatter.viewer()`) opens an interactive browser — must be replaced with a static screenshot cell for headless execution.

- [ ] **Identify unrendered cells**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json
  nb=json.load(open('docs/source/tutorials/03_splats/derive_splats.ipynb'))
  for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and not c.get('outputs'):
      print(f'Cell {i}: {repr(\"\".join(c[\"source\"])[:200])}')
  "
  ```

- [ ] **Fix: title markdown, autoreload, consolidate imports**

  Add title markdown as cell 0 if missing. Autoreload cell 1. Imports cell 2:
  ```python
  import os
  from pathlib import Path

  import pyvista as pv
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.wrapper import Splatter, SplatterConfig
  ```

- [ ] **Fix: config block**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET    = "birds_c0043"
  BASE_DIR   = Path("/workspace/fieldwork-data/")
  SESSION_DIR = BASE_DIR / "birds/2024-02-06/SplatsSD"
  VIDEO_PATH  = SESSION_DIR / "C0043.MP4"

  CACHE = Path("../../.cache") / DATASET   # declared for consistency
  ```

- [ ] **Fix: replace `splatter.viewer()` with screenshot**

  `splatter.viewer()` starts a web server and blocks — cannot run headlessly. Replace that cell with:
  ```python
  # viewer() opens an interactive nerfstudio web viewer — run manually for exploration.
  # In headless mode, we screenshot the trained mesh instead.
  from collab_splats.utils.visualization import visualize_splat
  import pyvista as pv

  mesh_fn = splatter.config["mesh_info"]["mesh"].as_posix()
  pl = visualize_splat(mesh=mesh_fn)
  pl.show()
  ```
  Add markdown before it:
  > *Displays the trained Gaussian splat model. In live Jupyter sessions `splatter.viewer()` launches an interactive web viewer; here we render a static screenshot via PyVista instead.*

- [ ] **Fix: section headers and markdown explanations**

  Current headers like `"Set paths..."`, `"Initialize..."` are just one-liners. Replace with `## §N — Title` + prose explanation cells. Example:
  - `## §1 — Configure Paths` + prose
  - `## §2 — Run COLMAP Preprocessing` + prose: *Runs COLMAP on the extracted video frames to produce camera poses and a sparse 3D point cloud. This is the most time-consuming step (~10–30 min depending on frame count). Output is cached in the SplatsSD directory.*
  - `## §3 — Train Gaussian Splat Model` + prose
  - `## §4 — Visualize Result` + prose

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=1800 \
    docs/source/tutorials/03_splats/derive_splats.ipynb
  ```
  If COLMAP/training cells error due to data already existing, add guards: `if not splatter.config["mesh_info"]["mesh"].exists(): splatter.preprocess()`.

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/03_splats/derive_splats.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/03_splats/derive_splats.ipynb
  git commit -m "docs(notebooks): polish derive_splats — viewer→screenshot, style, executed"
  ```

---

## Task 8: `03_splats/visualization`

**Notebook path:** `docs/source/tutorials/03_splats/visualization.ipynb`
**Current state:** All cells rendered (verified earlier via _build copy). Splatter-based. Uses `pv.start_xvfb()` — replace with conditional backend. Imports partially scattered (cell 9 has `import plyfile`, cell 17 has `import torch`).

- [ ] **Fix: insert autoreload, consolidate ALL imports**

  Move `import plyfile`, `import torch`, and any other mid-notebook imports to the imports cell:
  ```python
  import os
  import sys
  from pathlib import Path

  import numpy as np
  import open3d as o3d
  import plyfile
  import pyvista as pv
  import torch
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.wrapper import Splatter, SplatterConfig
  from collab_splats.utils.visualization import visualize_splat, load_colored_splat
  ```
  Remove `pv.start_xvfb()` — replaced by the backend conditional.

- [ ] **Fix: config block**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET     = "birds_c0043"
  BASE_DIR    = Path("/workspace/fieldwork-data/")
  SESSION_DIR = BASE_DIR / "birds/2024-02-06/SplatsSD"

  CACHE = Path("../../.cache") / DATASET   # declared for consistency
  ```

- [ ] **Fix: section headers and markdown explanations**

  Current headers (`### Load the information of a given splat`, `### Visualize a mesh`, etc.) use `###` not `##`. Update to `## §N — Title` and add prose explanation cells.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=600 \
    docs/source/tutorials/03_splats/visualization.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/03_splats/visualization.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/03_splats/visualization.ipynb
  git commit -m "docs(notebooks): polish visualization — remove xvfb, consolidate imports, executed"
  ```

---

## Task 9: `04_semantics/feature_extraction` (PORT from Splatter)

**Notebook path:** `docs/source/tutorials/04_semantics/feature_extraction.ipynb`
**Current state:** 9/9 cells rendered but uses old `Splatter`/`ConfigLoader`/`sample_frames_fps` to load first frame. Port to load from `IMAGES` cache (birds_c0043). Add zarr cache guard for feature extraction.
**Cache reads:** `docs/.cache/birds_c0043/images/`
**Cache writes:** `docs/.cache/birds_c0043/semantics/features.zarr`

- [ ] **Fix: remove Splatter/ConfigLoader imports, add direct path imports**

  Replace the imports cell entirely:
  ```python
  import os
  from pathlib import Path

  import numpy as np
  import matplotlib.pyplot as plt
  from PIL import Image
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.semantics.features import (
      BaseFeatureExtractor,
      MaskCLIPExtractor,
      Talk2DinoExtractor,
  )
  from collab_splats.utils.visualization import pca_to_rgb, compute_heatmap, compute_masked_image
  ```

- [ ] **Fix: config block — load frame from IMAGES cache**

  Replace the `Splatter`/`ConfigLoader`/`sample_frames_fps` setup with:
  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET  = "birds_c0043"
  QUERIES  = ["tree", "bark"]
  NEGATIVES = []

  CACHE    = Path("../../.cache") / DATASET
  IMAGES   = CACHE / "images"
  FEATURES = CACHE / "semantics" / "features.zarr"
  FEATURES.parent.mkdir(parents=True, exist_ok=True)

  assert IMAGES.exists() and any(IMAGES.glob("*.jpg")), (
      f"No images in {IMAGES}. Run 01_preprocessing/keyframe_extraction first."
  )

  # Load first frame for feature extraction demo
  frame_path = sorted(IMAGES.glob("*.jpg"))[0]
  frame = np.array(Image.open(frame_path))
  print(f"Loaded frame: {frame_path.name}  shape={frame.shape}")
  ```

- [ ] **Fix: section headers and markdown explanations**

  Update `## MaskCLIP` → `## §1 — MaskCLIP`, `## Talk2DINO` → `## §2 — Talk2DINO`, etc.
  Add prose before each code block.

- [ ] **Fix: add zarr cache-or-run guard for feature extraction (if extract_and_cache used)**

  If the notebook calls `extract_and_cache`, wrap it:
  ```python
  if FEATURES.exists():
      print(f"Feature cache exists at {FEATURES} — skip extraction")
  else:
      extractor = MaskCLIPExtractor()
      extractor.extract_and_cache([frame], FEATURES)
      print(f"Saved features to {FEATURES}")
  ```

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=600 \
    docs/source/tutorials/04_semantics/feature_extraction.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/04_semantics/feature_extraction.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  # Confirm no Splatter references remain:
  src=''.join(''.join(c.get('source','')) for c in nb['cells'])
  print('Splatter references:', src.count('Splatter'))
  "
  git add docs/source/tutorials/04_semantics/feature_extraction.ipynb
  git commit -m "docs(notebooks): port feature_extraction to cache-based loading, executed"
  ```

---

## Task 10: `04_semantics/segmentation` (PORT from Splatter)

**Notebook path:** `docs/source/tutorials/04_semantics/segmentation.ipynb`
**Current state:** 8/10 cells rendered. Uses `Splatter`/`ConfigLoader`/`sample_frames_fps` for first frame. Port to `IMAGES` cache. Two cells have no output — identify and fix.
**Cache reads:** `docs/.cache/birds_c0043/images/`

- [ ] **Fix: remove Splatter/ConfigLoader imports**

  Replace imports cell with:
  ```python
  import os
  from pathlib import Path

  import numpy as np
  import matplotlib.pyplot as plt
  import torch
  from PIL import Image
  %matplotlib inline

  from collab_splats.semantics import MobileSAMSegmentation
  from collab_splats.semantics.features import MaskCLIPExtractor
  from collab_splats.utils.visualization import overlay_masks, pca_to_rgb
  ```

- [ ] **Fix: config block — load frame from IMAGES**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET  = "birds_c0043"

  CACHE    = Path("../../.cache") / DATASET
  IMAGES   = CACHE / "images"

  assert IMAGES.exists() and any(IMAGES.glob("*.jpg")), (
      f"No images in {IMAGES}. Run 01_preprocessing/keyframe_extraction first."
  )

  device = "cuda" if torch.cuda.is_available() else "cpu"
  frame_path = sorted(IMAGES.glob("*.jpg"))[0]
  frame = np.array(Image.open(frame_path))
  print(f"Device: {device}  |  Frame: {frame_path.name}  shape={frame.shape}")
  ```

- [ ] **Fix: identify 2 unrendered cells and fix them**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json
  nb=json.load(open('docs/source/tutorials/04_semantics/segmentation.ipynb'))
  for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and not c.get('outputs'):
      print(f'Cell {i}: {repr(\"\".join(c[\"source\"])[:200])}')
  "
  ```
  Fix any import errors or stale references.

- [ ] **Fix: section headers and markdown explanations**

  Update `## Object Strategy`, `## Auto Strategy`, `## Masks → Features` → `## §1 — ...`, `## §2 — ...`, `## §3 — ...`. Add prose.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=600 \
    docs/source/tutorials/04_semantics/segmentation.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/04_semantics/segmentation.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/04_semantics/segmentation.ipynb
  git commit -m "docs(notebooks): port segmentation to cache-based loading, executed"
  ```

---

## Task 11: `04_semantics/maskclip_vs_talk2dino`

**Notebook path:** `docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb`
**Current state:** 8/10+ cells rendered. Uses Splatter to load existing trained models. Uses `ipywidgets` for interactive query runner — replace with static calls for headless execution. `%load_ext autoreload` already present but embedded with other imports.

- [ ] **Fix: separate autoreload from imports, add backend conditional**

  Move autoreload to its own cell. Move `import warnings`, `import numpy as np`, `import pyvista as pv`, `import ipywidgets as widgets`, `from pathlib import Path` etc. into one clean imports cell with backend conditional. Add `%matplotlib inline`.

- [ ] **Fix: replace ipywidgets interactive cell with static execution**

  The query runner cell uses `ipywidgets` — this will not render headlessly. Replace with a static call:
  ```python
  # Static query run (ipywidgets interactive version available in live Jupyter session)
  query_name = "feeder"
  positive_queries = ["feeder"]
  negative_queries = ["ground", "bark", "sky"]
  temperature = 100.0
  compare_interactive(splatter_mc, splatter_t2d,
                      load_query_colors(splatter_mc, query_name),
                      load_query_colors(splatter_t2d, query_name),
                      query_name)
  ```
  Add markdown:
  > *Runs a static feature query comparison between MaskCLIP and Talk2DINO for the "feeder" preset. In a live Jupyter session, replace this cell with the ipywidgets interactive version to explore queries dynamically.*

- [ ] **Fix: config block, section headers, markdown explanations**

  Apply canonical config block (no imports). Update headers to `## §N — Title`. Add prose cells.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=900 \
    docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb
  git commit -m "docs(notebooks): polish maskclip_vs_talk2dino — widgets→static, style, executed"
  ```

---

## Task 12: `05_lifting/semantic_lifting`

**Notebook path:** `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`
**Current state:** 0/10 cells rendered. Already has cache-load logic (reads `vggtx/ba/reconstruction.zarr`), config block, and `§N` headers. Main gaps: autoreload missing, mid-cell imports, no pyvista backend.
**Cache reads:** `docs/.cache/birds_c0043/vggtx/ba/reconstruction.zarr` (written by: run bundle_adjustment on vggtx output — verify this exists or use base `vggtx/reconstruction.zarr` with `VARIANT=""`)
**Cache writes:** `docs/.cache/birds_c0043/vggtx/ba/lifted.zarr`

- [ ] **Fix: verify which RECON path to use**

  Check which zarr exists:
  ```bash
  ls docs/.cache/birds_c0043/vggtx/ 2>/dev/null
  ```
  If `ba/reconstruction.zarr` doesn't exist (BA not yet run), set `VARIANT = ""` to use the base reconstruction. Update the config block accordingly.

- [ ] **Fix: insert autoreload, consolidate imports**

  Add autoreload cell. Move all imports to one cell:
  ```python
  import os
  from pathlib import Path

  import numpy as np
  import torch
  import torch.nn.functional as F
  import pyvista as pv
  import matplotlib.pyplot as plt
  from PIL import Image
  from tqdm.auto import tqdm
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.pointcloud.feedforward.base import FeedforwardResult
  from collab_splats.pointcloud.utils import lift_features
  from collab_splats.semantics.features import MaskCLIPExtractor
  from collab_splats.semantics.compression import FeatureAutoencoder
  ```

- [ ] **Fix: config block — no imports, path guard**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET   = "birds_c0043"
  METHOD    = "vggtx"
  VARIANT   = "ba"    # "ba" | "lc" | "" — set to "" if ba cache not yet available
  QUERIES   = ["tree", "bird feeder", "ground"]
  NEGATIVES = ["background"]

  CACHE  = Path("../../.cache") / DATASET
  IMAGES = CACHE / "images"
  RECON  = CACHE / METHOD / VARIANT if VARIANT else CACHE / METHOD

  assert (RECON / "reconstruction.zarr").exists(), (
      f"No reconstruction at {RECON}. Run 02_pointcloud/feedforward_methods (and bundle_adjustment if VARIANT='ba') first."
  )
  ```

- [ ] **Fix: markdown explanations — every existing code section**

  Current sections `§1`, `§2`, etc. likely have headers but may lack prose. Ensure every section has 1–3 sentence explanation.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=1200 \
    docs/source/tutorials/05_lifting/semantic_lifting.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/05_lifting/semantic_lifting.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
  git commit -m "docs(notebooks): polish semantic_lifting — style, fully executed"
  ```

---

## Task 13: `06_mesh/create_mesh`

**Notebook path:** `docs/source/tutorials/06_mesh/create_mesh.ipynb`
**Current state:** 6/12 cells rendered. Splatter-based. Uses `pv.start_xvfb()`. Mid-notebook imports (`import numpy as np`, `import open3d as o3d`, `import copy`). The unrendered cells include clustering/visualization steps.

- [ ] **Fix: remove `pv.start_xvfb()`, insert autoreload, consolidate imports**

  ```python
  import os
  import copy
  from pathlib import Path

  import numpy as np
  import open3d as o3d
  import pyvista as pv
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.wrapper import Splatter, SplatterConfig
  from collab_splats.mesh.utils import mesh_clustering
  ```

- [ ] **Fix: config block — no imports**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET     = "birds_c0043"
  BASE_DIR    = Path("/workspace/fieldwork-data/")
  SESSION_DIR = BASE_DIR / "birds/2024-02-06/SplatsSD"

  CACHE = Path("../../.cache") / DATASET
  ```

- [ ] **Fix: add title markdown and `## §N — Title` headers**

  Current cell 0 is a markdown with `## Creating a mesh` — promote to `# Creating a Mesh` title. Add `## §N — Title` headers throughout.

- [ ] **Fix: add markdown explanation cells**

  Each step (create mesh, plot similarity, cluster, select largest cluster) needs a prose explanation.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=900 \
    docs/source/tutorials/06_mesh/create_mesh.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/06_mesh/create_mesh.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/06_mesh/create_mesh.ipynb
  git commit -m "docs(notebooks): polish create_mesh — remove xvfb, consolidate imports, executed"
  ```

---

## Task 14: `07_localization/localization`

**Notebook path:** `docs/source/tutorials/07_localization/localization.ipynb`
**Current state:** 0/9 cells rendered. Has cache-load guard, config block, `§N` headers. Missing autoreload, pyvista backend, mid-notebook imports.
**Cache reads:** `docs/.cache/birds_c0043/vggtx/reconstruction.zarr`

- [ ] **Fix: insert autoreload, consolidate imports**

  Move all imports to one cell:
  ```python
  import os
  from pathlib import Path

  import cv2
  import numpy as np
  import pyvista as pv
  import torch
  %matplotlib inline

  pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

  from collab_splats.pointcloud.feedforward import VGGTXCreator
  from collab_splats.pointcloud.feedforward.base import FeedforwardResult
  from collab_splats.pointcloud.localization import CameraLocalizer, XFeatExtractor
  ```

- [ ] **Fix: config block — verify RECON path**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET  = "birds_c0043"
  METHOD   = "vggtx"
  VARIANT  = ""    # base reconstruction

  CACHE  = Path("../../.cache") / DATASET
  IMAGES = CACHE / "images"
  RECON  = CACHE / METHOD / VARIANT if VARIANT else CACHE / METHOD

  assert (RECON / "reconstruction.zarr").exists(), (
      f"No reconstruction at {RECON}. Run 02_pointcloud/feedforward_methods first."
  )
  ```

- [ ] **Fix: markdown explanations for each section**

  Verify every `§N` section has prose.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=900 \
    docs/source/tutorials/07_localization/localization.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/07_localization/localization.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/07_localization/localization.ipynb
  git commit -m "docs(notebooks): polish localization — style, fully executed"
  ```

---

## Task 15: `evals/ground_truth_evals`

**Notebook path:** `docs/source/tutorials/evals/ground_truth_evals.ipynb`
**Current state:** 0 cells rendered. Utility/viz notebook — defines helper functions and demonstrates loading results. The `run_eval()` example cell uses a dummy co3dv2 path that doesn't exist; that cell should be commented out. `compare_runs()` reads from `evals/results/` — will work if results exist.

- [ ] **Fix: insert autoreload, consolidate imports**

  Current cell 3 has `import subprocess, sys, from pathlib import Path`; cell 6 has `import json, import pandas as pd, from pathlib import Path`; cell 9 has `import numpy as np, import matplotlib.pyplot as plt`. Consolidate:
  ```python
  import json
  import subprocess
  import sys
  from pathlib import Path

  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
  %matplotlib inline
  ```

- [ ] **Fix: config block**

  ```python
  # ── Configuration ─────────────────────────────────────────────────────────────
  DATASET      = "birds_c0043"
  RESULTS_ROOT = Path("../../evals/results")
  CACHE        = Path("../../.cache") / DATASET   # for consistency
  ```

- [ ] **Fix: comment out the `run_eval()` example cell**

  Cell 4 calls `run_eval(dataset="co3dv2", seq_dir="/data/co3dv2/apple/...")` with a non-existent path. Replace with:
  ```python
  # Example — edit these values for your sequence, then uncomment to run:
  # out = run_eval(
  #     dataset="co3dv2",
  #     seq_dir="/data/co3dv2/apple/110_13051_23361",
  #     conditions=["baseline", "ba", "lc"],
  # )
  print("run_eval() example commented out — edit path above and uncomment to run an evaluation.")
  ```

- [ ] **Fix: `compare_runs()` cell — guard on results existing**

  ```python
  if RESULTS_ROOT.exists() and any(RESULTS_ROOT.rglob("metrics.json")):
      df = compare_runs(results_root=RESULTS_ROOT)
      display(df)
  else:
      print(f"No results found in {RESULTS_ROOT}. Run an evaluation first.")
  ```

- [ ] **Fix: section headers and markdown explanations**

  Current `## 2.`, `## 3.` etc. → `## §1 — Run an Evaluation`, `## §2 — Load and Compare Results`, etc. Add prose.

- [ ] **Execute notebook**

  ```bash
  PYVISTA_OFF_SCREEN=true /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=300 \
    docs/source/tutorials/evals/ground_truth_evals.ipynb
  ```

- [ ] **Verify and commit**

  ```bash
  /opt/conda/envs/nerfstudio/bin/python -c "
  import json; nb=json.load(open('docs/source/tutorials/evals/ground_truth_evals.ipynb'))
  code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
  errors=[c for c in code_cells if any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
  print(f\"{sum(bool(c.get('outputs')) for c in code_cells)}/{len(code_cells)} output, {len(errors)} errors\")
  "
  git add docs/source/tutorials/evals/ground_truth_evals.ipynb
  git commit -m "docs(notebooks): polish ground_truth_evals — consolidate imports, guard cells, executed"
  ```

---

## Task 16: Collect Additional Functions List

After all notebooks are polished and executed, collect repeated patterns observed during the pass into a deliverable list.

- [ ] **Write the additional functions list**

  Scan finished notebooks for repeated patterns. The expected candidates (confirm during pass):

  | Function | Signature | Why |
  |----------|-----------|-----|
  | `load_first_frame(images_dir)` | `(Path) → np.ndarray` | Replace repeated `sorted(IMAGES.glob("*.jpg"))[0]` + `Image.open` across notebooks 9, 10, 11 |
  | `assert_cache_exists(path, prereq_notebook)` | `(Path, str) → None` | Replace ad-hoc `assert path.exists(), f"Run X first"` across all chain notebooks |
  | `display_stats_table(models, results)` | `(dict) → None` | Stats table (pts raw/filtered, conf mean/std, timing) duplicated in feedforward_methods and bundle_adjustment |
  | `save_figure(fig, name, cache_dir)` | `(fig, str, Path) → Path` | Save matplotlib/pyvista figures to `.cache/{dataset}/figures/` for reuse |
  | `check_gpu(min_gb)` | `(float) → None` | Warn if GPU memory below threshold before heavy inference cells |

  Write final list as a markdown comment at the end of the spec file `docs/superpowers/specs/2026-05-23-notebook-polish-design.md`.

- [ ] **Commit the updated spec**

  ```bash
  git add docs/superpowers/specs/2026-05-23-notebook-polish-design.md
  git commit -m "docs(spec): add post-pass additional functions list"
  ```

---

## Self-Review Notes

- `colmap_sfm`: style-only, no execution — correctly handled in Task 6.
- `derive_splats` / `visualization` / `create_mesh` / `maskclip_vs_talk2dino`: remain Splatter-based (intentional — they demonstrate the Splatter API). Only `feature_extraction` and `segmentation` are ported.
- `splatter.viewer()` in `derive_splats`: replaced with PyVista screenshot — headless-safe.
- `ipywidgets` in `maskclip_vs_talk2dino`: replaced with static call — headless-safe.
- `run_eval()` example in `ground_truth_evals`: commented out — no dummy-path crash.
- Cache chain: Task 1 writes `images/` → Tasks 2, 9, 10 read `images/`. Task 2 writes `reconstruction.zarr` → Tasks 5, 12, 14 read it. All dependencies respected in order.
