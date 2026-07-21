# Tutorial example data + self-contained frames.zarr chain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tutorial runnable end-to-end by a new user with no gcloud access, from a single committed example video, and clean up the tutorial (frames.zarr unification, config declutter, notebook hygiene) in the process.

**Architecture:** Ship a re-encoded example video + one external query frame under `data/tutorial/` (committed). Rewrite `tutorial_config.py` to repo-relative paths with generic asset names, no `DATASET`/inference. Every notebook reads keyframes from one canonical `OUTPUT_DIR/frames.zarr` via `FrameStore`, and the reconstruction from one canonical `OUTPUT_DIR/feedforward.zarr`; the 01→07 chain regenerates every artifact from the video. `07` localizes an external frame via `estimate_intrinsics`. Shared notebook boilerplate moves to `docs/source/tutorials/notebook_utils.py`.

**Tech Stack:** Python 3.11, Jupyter notebooks (nbsphinx), `FrameStore` (zarr v3), `CameraLocalizer`/`estimate_intrinsics`, ffmpeg, pytest.

**Interim note (current setup API):** The feedforward creator is fed via a path-based dir today. Task 5 uses `FrameStore.export()` to a transient dir as the sanctioned bridge. A concurrent effort makes `_preprocess` accept preloaded arrays; swapping `export()` → `store.images()` is an explicit **follow-up**, not in this plan.

**Env:** use `/opt/venv/reconstruction/bin/python` for all pytest/python invocations.

---

## File structure

- Create: `data/tutorial/tutorial_example-video.mp4` (~75 MB, committed) — reconstruction input.
- Create: `data/tutorial/tutorial_example-frame.jpg` (2.2 MB, committed) — external localization query.
- Create: `data/tutorial/README.md` — asset provenance + regeneration commands.
- Modify: `docs/source/tutorials/tutorial_config.py` — repo-relative, decluttered.
- Create: `docs/source/tutorials/notebook_utils.py` — `set_notebook_backend`, `load_keyframe_paths`.
- Modify: `.gitignore` — ignore `data/outputs/`.
- Modify notebooks: `01_preprocessing/keyframe_extraction.ipynb`, `02_pointcloud/feedforward_methods.ipynb`, `04_semantics/{feature_extraction,segmentation,maskclip_vs_talk2dino}.ipynb`, `05_lifting/semantic_lifting.ipynb`, `07_localization/localization.ipynb`, `03_splats/visualization.ipynb`.
- Create tests: `tests/docs/__init__.py`, `tests/docs/test_tutorial_config.py`, `tests/docs/test_notebook_utils.py`.

**Out of scope:** `02_pointcloud/bundle_adjustment.ipynb`, `02_pointcloud/slam_loop_closure.ipynb` (no edits).

---

## Task 1: Commit example assets

**Files:**
- Create: `data/tutorial/tutorial_example-video.mp4`
- Create: `data/tutorial/tutorial_example-frame.jpg`
- Create: `data/tutorial/README.md`

- [ ] **Step 1: Re-encode the video under the size cap**

Source is `/workspace/outputs/2024_02_06/C0043/C0043.MP4` (1080p, 99.6 s, 738 MB).

```bash
mkdir -p /workspace/collab-splats/data/tutorial
ffmpeg -y -i /workspace/outputs/2024_02_06/C0043/C0043.MP4 \
       -c:v libx264 -b:v 6M -maxrate 6M -bufsize 12M -an \
       /workspace/collab-splats/data/tutorial/tutorial_example-video.mp4
```

- [ ] **Step 2: Verify the encode is under GitHub's 100 MB/file cap**

```bash
python - <<'PY'
from pathlib import Path
p = Path("/workspace/collab-splats/data/tutorial/tutorial_example-video.mp4")
mb = p.stat().st_size / 1e6
print(f"{p.name}: {mb:.1f} MB")
assert mb < 95, f"too big for git ({mb:.1f} MB) — lower -b:v"
PY
```
Expected: prints ~70–80 MB, no assertion error. If ≥95 MB, drop `-b:v` to `5M` and re-run.

- [ ] **Step 3: Copy the external query frame**

```bash
cp /workspace/outputs/2024_02_06/C0043/localized_frames/GX010119_f000000.jpg \
   /workspace/collab-splats/data/tutorial/tutorial_example-frame.jpg
```

- [ ] **Step 4: Write `data/tutorial/README.md`**

```markdown
# Tutorial example data

Assets consumed by the notebooks in `docs/source/tutorials/`. Both are committed so
a fresh clone can run the full tutorial with no gcloud access.

## `tutorial_example-video.mp4`
Reconstruction input (nb 01–06). Re-encode of session `2024_02_06` video `C0043`
(1080p, ~100 s), bitrate-reduced to fit GitHub's 100 MB/file limit:

    ffmpeg -i C0043.MP4 -c:v libx264 -b:v 6M -maxrate 6M -bufsize 12M -an \
           tutorial_example-video.mp4

## `tutorial_example-frame.jpg`
External localization query (nb 07). Frame 0 of GoPro video `GX010119` — a *different*
video from the reconstruction — localized against the C0043 map.

## Outputs
Notebooks write regenerated artifacts to `data/outputs/` (gitignored).
```

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git add -f data/tutorial/tutorial_example-video.mp4 data/tutorial/tutorial_example-frame.jpg data/tutorial/README.md
git commit -m "feat(tutorial): commit example video + external query frame"
```

---

## Task 2: Rewrite `tutorial_config.py` (TDD)

**Files:**
- Create: `tests/docs/__init__.py`
- Create: `tests/docs/test_tutorial_config.py`
- Modify: `docs/source/tutorials/tutorial_config.py`

- [ ] **Step 1: Create the test package init**

```bash
: > /workspace/collab-splats/tests/docs/__init__.py
```

- [ ] **Step 2: Write the failing test**

`tests/docs/test_tutorial_config.py`:

```python
import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "docs/source/tutorials/tutorial_config.py"


def _load():
    """Exec tutorial_config.py in a fresh namespace and return its globals."""
    return runpy.run_path(str(CONFIG))


def test_paths_are_repo_relative_and_correct():
    ns = _load()
    assert ns["VIDEO_PATH"] == REPO_ROOT / "data/tutorial/tutorial_example-video.mp4"
    assert ns["QUERY_IMAGE"] == REPO_ROOT / "data/tutorial/tutorial_example-frame.jpg"
    assert ns["OUTPUT_DIR"] == REPO_ROOT / "data/outputs"
    assert ns["FRAMES_ZARR"] == REPO_ROOT / "data/outputs/frames.zarr"
    assert ns["RECON"] == REPO_ROOT / "data/outputs/feedforward.zarr"
    assert ns["TUTORIAL_CACHE"] == REPO_ROOT / "data/outputs/tutorial_cache"
    assert ns["MAX_FRAMES"] == 30


def test_no_retired_names():
    ns = _load()
    for gone in ("DATASET", "BASE_DIR", "FRAMES", "_infer_video_path"):
        assert gone not in ns, f"{gone} should be removed"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_config.py -v`
Expected: FAIL — current config defines `DATASET`/`BASE_DIR`/`FRAMES`, no `FRAMES_ZARR`/`RECON`/`QUERY_IMAGE`.

- [ ] **Step 4: Replace `tutorial_config.py` with the decluttered version**

Full new file contents:

```python
"""Shared paths for the tutorial notebooks. `%run ../tutorial_config.py` to load.

Inputs (VIDEO_PATH, QUERY_IMAGE) are committed under data/tutorial/. Outputs are
regenerated by the notebooks under data/outputs/ (gitignored). The 01->07 chain is
self-contained: no gcloud, no pre-baked artifacts.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # docs/source/tutorials/ -> repo root
_TUT_DATA = REPO_ROOT / "data" / "tutorial"
MAX_FRAMES = 30

# ── Committed inputs (read-only) ──────────────────────────────────────────────
VIDEO_PATH = _TUT_DATA / "tutorial_example-video.mp4"
QUERY_IMAGE = _TUT_DATA / "tutorial_example-frame.jpg"

# ── Generated outputs (gitignored; notebooks write here) ──────────────────────
OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
FRAMES_ZARR = OUTPUT_DIR / "frames.zarr"          # canonical keyframes (nb 01 writes)
RECON = OUTPUT_DIR / "feedforward.zarr"           # canonical reconstruction (nb 02 writes)
CACHE_DIR = OUTPUT_DIR                             # alias kept for notebook readability
TUTORIAL_CACHE = OUTPUT_DIR / "tutorial_cache"    # notebook scratch (plots, ae caches)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"missing {VIDEO_PATH} — see data/tutorial/README.md (git clone should provide it)")
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_config.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/source/tutorials/tutorial_config.py tests/docs/__init__.py tests/docs/test_tutorial_config.py
git commit -m "refactor(tutorial): repo-relative config, drop DATASET/FRAMES/inference"
```

---

## Task 3: Add `notebook_utils.py` (TDD)

**Files:**
- Create: `docs/source/tutorials/notebook_utils.py`
- Create: `tests/docs/test_notebook_utils.py`

- [ ] **Step 1: Write the failing test**

`tests/docs/test_notebook_utils.py`:

```python
import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MOD = REPO_ROOT / "docs/source/tutorials/notebook_utils.py"


def _load():
    spec = importlib.util.spec_from_file_location("notebook_utils", MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_keyframe_paths_returns_sorted_frame_paths(tmp_path):
    from collab_splats.preproc import FrameStore

    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    records = [{"frame_idx": i} for i in (5, 1, 3)]
    zpath = tmp_path / "frames.zarr"
    FrameStore.create(zpath, frames, records, provenance={"video_path": "x"})

    nu = _load()
    paths = nu.load_keyframe_paths(zpath, tmp_path / "export")

    assert len(paths) == 3
    assert all(p.exists() and p.suffix == ".jpg" for p in paths)
    assert paths == sorted(paths)  # deterministic order


def test_set_notebook_backend_is_callable(monkeypatch):
    nu = _load()
    called = {}
    import pyvista as pv
    monkeypatch.setattr(pv, "set_jupyter_backend", lambda b: called.setdefault("b", b))
    nu.set_notebook_backend()
    assert called["b"] in ("static", "trame")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_notebook_utils.py -v`
Expected: FAIL — `notebook_utils.py` does not exist.

- [ ] **Step 3: Create `notebook_utils.py`**

```python
"""Presentation helpers shared by the tutorial notebooks. `%run ../notebook_utils.py`.

Notebook-only concerns (kept out of collab_splats/): pyvista backend selection and a
canonical keyframe loader. Import via `%run` so the functions land in the notebook namespace.
"""

import os
from pathlib import Path

import pyvista as pv

from collab_splats.preproc import FrameStore


def set_notebook_backend() -> None:
    """Static pyvista backend when headless (nbconvert), interactive trame otherwise."""
    pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")


def load_keyframe_paths(frames_zarr, export_dir) -> list[Path]:
    """Open the canonical frames.zarr and export its keyframes as sorted jpg paths.

    Transient bridge for path-locked model preprocessing: pixels come from the
    tutorial-owned frames.zarr, not a pipeline jpg dir. Returns frame_NNNNNN.jpg paths.
    """
    store = FrameStore.open(frames_zarr)
    return sorted(store.export(export_dir, ext="jpg"))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_notebook_utils.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/source/tutorials/notebook_utils.py tests/docs/test_notebook_utils.py
git commit -m "feat(tutorial): notebook_utils with backend + keyframe-path helpers"
```

---

## Task 4: NB 01 — write canonical `frames.zarr` to `OUTPUT_DIR`

**Files:**
- Modify: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` (cell 3, cell 21)

Use the NotebookEdit tool for each cell edit.

- [ ] **Step 1: Repoint the config cell (cell 3)**

Replace the `FRAME_SCORES`/`KEYFRAMES` block:

```python
%run ../tutorial_config.py

# ── Configuration ─────────────────────────────────────────────────────────────
# Scratch (plots/score cache) under TUTORIAL_CACHE; the canonical keyframe store
# (FRAMES_ZARR) is what every downstream notebook reads.
TUTORIAL_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_SCORES = TUTORIAL_CACHE / "frame_scores.json"
```

- [ ] **Step 2: Repoint the store-write cell (cell 21)**

Replace the `FrameStore.create(KEYFRAMES, ...)` block:

```python
# Re-sample with the OF method to get pixel data alongside the selection records
of_frames, of_records = sample_frames(VIDEO_PATH, method="optical_flow", max_frames=MAX_FRAMES)
of_indices = [r["frame_idx"] for r in of_records]

# Persist the canonical frames.zarr — the sole keyframe source for notebooks 02-07
store = FrameStore.create(
    FRAMES_ZARR,
    of_frames,
    of_records,
    provenance={"video_path": str(VIDEO_PATH), "method": "optical_flow", "max_frames": MAX_FRAMES},
)
print(f"Extracted {len(of_indices)} OF keyframes → {store.path}")
plot_selection(info["total_frames"], of_indices=of_indices)
```

- [ ] **Step 3: Execute the notebook headless to verify it writes the store**

```bash
cd /workspace/collab-splats/docs/source/tutorials/01_preprocessing
rm -rf /workspace/collab-splats/data/outputs
PYVISTA_OFF_SCREEN=1 /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1200 --output /tmp/_01_out.ipynb keyframe_extraction.ipynb
python -c "from collab_splats.preproc import FrameStore; s=FrameStore.open('/workspace/collab-splats/data/outputs/frames.zarr'); print('frames:', len(s))"
```
Expected: notebook executes without error; prints `frames: <N>` (N ≤ 30).

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
git commit -m "refactor(tutorial): nb01 writes canonical OUTPUT_DIR/frames.zarr"
```

---

## Task 5: NB 02 — read keyframes from `frames.zarr`, write canonical `RECON`

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` (cell 3, cell 5)

- [ ] **Step 1: Replace the keyframe-source cell (cell 3)**

Swap the `FRAMES.glob` block for a `frames.zarr` export (current path-based setup API):

```python
%run ../tutorial_config.py
%run ../notebook_utils.py

# ── Configuration ─────────────────────────────────────────────────────────────
# Keyframes come from the canonical frames.zarr (nb 01). export() is a transient
# jpg bridge for the path-based creator setup API; swap to preloaded arrays once
# _preprocess accepts them (follow-up).
image_paths = load_keyframe_paths(FRAMES_ZARR, TUTORIAL_CACHE / "_frames_export")[:MAX_FRAMES]
assert image_paths, f"no keyframes in {FRAMES_ZARR} — run 01_preprocessing first"
print(f"{len(image_paths)} keyframes from {FRAMES_ZARR}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
```

- [ ] **Step 2: Repoint the reconstruction cache to `RECON` (cell 5)**

Replace `_omega_cache = TUTORIAL_CACHE / "vggt_omega.zarr"` with the canonical `RECON`:

```python
# Run VGGT-Omega once; later executions reload the zarr cache
_omega_cache = RECON
_omega_cache.parent.mkdir(parents=True, exist_ok=True)

if _omega_cache.exists():
    result = FeedforwardResult.load_zarr(_omega_cache)
    print(f"loaded cached reconstruction ({result.points.shape[0]:,} pts)")
    # Stale-cache guard: frame count baked into the cache vs current MAX_FRAMES selection
    if len(result.image_paths) != len(image_paths):
        print("cache frame-count mismatch — delete RECON and re-run to rebuild")
```

Keep the rest of cell 5 (the `else:` branch that runs the creator and saves to `_omega_cache`) unchanged.

- [ ] **Step 3: Execute headless to verify RECON is written**

```bash
cd /workspace/collab-splats/docs/source/tutorials/02_pointcloud
PYVISTA_OFF_SCREEN=1 /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 --output /tmp/_02_out.ipynb feedforward_methods.ipynb
python -c "from collab_splats.pointcloud.feedforward.base import FeedforwardResult; r=FeedforwardResult.load_zarr('/workspace/collab-splats/data/outputs/feedforward.zarr'); print('pts:', r.points.shape)"
```
Expected: executes on GPU; prints a non-empty `pts:` shape.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "refactor(tutorial): nb02 reads frames.zarr, writes canonical RECON"
```

---

## Task 6: NB 04 — read keyframes from `frames.zarr` (3 notebooks)

**Files:**
- Modify: `docs/source/tutorials/04_semantics/feature_extraction.ipynb` (cell 3)
- Modify: `docs/source/tutorials/04_semantics/segmentation.ipynb` (cell 3)
- Modify: `docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb` (cell 4)

Each currently asserts on `FRAMES.glob("*.jpg")`. Replace with an export from the canonical store, keeping the downstream jpg-path interface identical.

- [ ] **Step 1: `feature_extraction.ipynb` cell 3 — replace the FRAMES block**

```python
%run ../tutorial_config.py
%run ../notebook_utils.py

# ── Configuration ─────────────────────────────────────────────────────────────
QUERY_POSITIVE = ["tree"]
QUERY_NEGATIVE = ["ground"]

FEATURES = TUTORIAL_CACHE / "semantics" / "features.zarr"
FEATURES.parent.mkdir(parents=True, exist_ok=True)

# Keyframes from the canonical frames.zarr (nb 01)
frame_paths = load_keyframe_paths(FRAMES_ZARR, TUTORIAL_CACHE / "_frames_export")
assert frame_paths, f"no keyframes in {FRAMES_ZARR} — run 01_preprocessing first"
```

Then, in the cell that consumes the first frame, replace `sorted(FRAMES.glob("*.jpg"))[0]` with `frame_paths[0]`.

- [ ] **Step 2: `segmentation.ipynb` cell 3 — replace the FRAMES assert**

```python
%run ../tutorial_config.py
%run ../notebook_utils.py

# ── Configuration ─────────────────────────────────────────────────────────────
# Keyframes from the canonical frames.zarr (nb 01)
frame_paths = load_keyframe_paths(FRAMES_ZARR, TUTORIAL_CACHE / "_frames_export")
assert frame_paths, f"no keyframes in {FRAMES_ZARR} — run 01_preprocessing first"

device = "cuda" if torch.cuda.is_available() else "cpu"
```

Replace any later `sorted(FRAMES.glob("*.jpg"))` / `FRAMES.glob(...)` use in this notebook with `frame_paths`.

- [ ] **Step 3: `maskclip_vs_talk2dino.ipynb` cell 4 — replace the FRAMES assert**

```python
%run ../tutorial_config.py
%run ../notebook_utils.py

# ── Configuration ─────────────────────────────────────────────────────────────
QUERIES = ["tree", "bark", "sky"]
NEGATIVES = []

# Keyframes from the canonical frames.zarr (nb 01)
frame_paths = load_keyframe_paths(FRAMES_ZARR, TUTORIAL_CACHE / "_frames_export")
assert frame_paths, f"no keyframes in {FRAMES_ZARR} — run 01_preprocessing first"
```

Replace any later `FRAMES.glob(...)` use with `frame_paths`.

- [ ] **Step 4: Execute the three notebooks headless**

```bash
cd /workspace/collab-splats/docs/source/tutorials/04_semantics
for nb in feature_extraction segmentation maskclip_vs_talk2dino; do
  PYVISTA_OFF_SCREEN=1 /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=2400 --output /tmp/_04_$nb.ipynb $nb.ipynb || { echo "FAILED $nb"; break; }
done
```
Expected: all three execute without error (they consume `frame_paths` produced from `frames.zarr`).

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/source/tutorials/04_semantics/feature_extraction.ipynb \
             docs/source/tutorials/04_semantics/segmentation.ipynb \
             docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb
git commit -m "refactor(tutorial): nb04 reads keyframes from frames.zarr"
```

---

## Task 7: NB 05 — RECON from config, frames via FrameStore

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` (cell 1, cell 3, cell 9)

- [ ] **Step 1: Use the shared backend helper (cell 1)**

Replace the inline `pv.set_jupyter_backend(...)` line with a `%run` of notebook_utils and a call. Add near the top of cell 1 (after imports), replacing the existing backend line:

```python
%run ../notebook_utils.py
set_notebook_backend()
```

- [ ] **Step 2: Repoint RECON to config (cell 3)**

Replace `RECON = TUTORIAL_CACHE / "vggt_omega.zarr"` with the config value (delete the local reassignment — `RECON` now comes from `%run ../tutorial_config.py`):

```python
%run ../tutorial_config.py

# ── Configuration ─────────────────────────────────────────────────────────────
LATENT_DIM = 13
QUERY_POSITIVE = ["tree"]
QUERY_NEGATIVE = ["ground"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RECON comes from tutorial_config (OUTPUT_DIR/feedforward.zarr, written by nb 02)
assert RECON.exists(), f"missing {RECON} — run 02_pointcloud/feedforward_methods.ipynb first"

_base = TUTORIAL_CACHE / "lifted"
```

Keep the remaining `_base`-derived paths in this cell unchanged.

- [ ] **Step 3: Load frames via FrameStore instead of `image_paths` (cell 9)**

Replace `imgs = [Image.open(p).convert("RGB") for p in tqdm(out.image_paths, ...)]` with a read from the canonical store (the `07` pattern), since `out.image_paths` reference a transient export dir:

```python
# Load frames from the canonical frames.zarr — out.image_paths reference a transient
# per-run export dir that no longer exists; read pixels from the persistent store.
_fs = FrameStore.open(FRAMES_ZARR)
imgs = [
    Image.fromarray(_fs.image_by_frame_idx(FrameStore.frame_idx_from_path(p))).convert("RGB")
    for p in tqdm(out.image_paths, desc="Loading frames")
]
```

Add `from collab_splats.preproc import FrameStore` to the cell-1 import block if not already present.

- [ ] **Step 4: Execute headless**

```bash
cd /workspace/collab-splats/docs/source/tutorials/05_lifting
PYVISTA_OFF_SCREEN=1 /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 --output /tmp/_05_out.ipynb semantic_lifting.ipynb
```
Expected: executes without error (loads RECON + frames from `frames.zarr`).

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "refactor(tutorial): nb05 uses config RECON + frames.zarr pixels"
```

---

## Task 8: NB 07 — external-frame localization

**Files:**
- Modify: `docs/source/tutorials/07_localization/localization.ipynb` (cell 3, cell 4, cell 7, cell 9)

- [ ] **Step 1: Add `estimate_intrinsics` to imports + shared backend (cell 3)**

In the import cell, extend the localization import and replace the inline backend line:

```python
from collab_splats.localization import (
    CameraLocalizer,
    LomaExtractor,
    estimate_intrinsics,
    plot_correspondences,
)
```
And add after imports:
```python
%run ../notebook_utils.py
set_notebook_backend()
```

- [ ] **Step 2: RECON from config (cell 4)**

```python
%run ../tutorial_config.py

# ── Configuration ─────────────────────────────────────────────────────────────
# Reconstruction produced by 02_pointcloud/feedforward_methods.ipynb
assert RECON.exists(), f"missing {RECON} — run 02_pointcloud/feedforward_methods.ipynb first"
print(f"RECON: {RECON}")
```

- [ ] **Step 3: Replace the in-video query with the external frame (cell 7)**

Swap the `QUERY_IDX`/`frames.zarr` self-query block for the committed external query + estimated intrinsics:

```python
# External query: a frame from a DIFFERENT video (not in the reconstruction).
query_image = cv2.cvtColor(cv2.imread(str(QUERY_IMAGE)), cv2.COLOR_BGR2RGB)

# Estimate query intrinsics (experimental; refined by pycolmap focal refinement in PnP).
query_K = estimate_intrinsics(query_image)
print(f"query: {QUERY_IMAGE.name}  {query_image.shape}  fx≈{query_K[0,0]:.0f}")
```

- [ ] **Step 4: Build the localizer from the FULL reconstruction, no trim (cell 9)**

Replace the `result_trimmed` construction + localizer build with the full-map version, reading reference pixels from `frames.zarr`:

```python
# Reference pixels from the canonical frames.zarr; localize against the full map
# (the external query is not part of it, so no reference-set trim is needed).
frame_store = FrameStore.open(FRAMES_ZARR)
ref_images = (
    frame_store.image_by_frame_idx(FrameStore.frame_idx_from_path(p)) for p in result.image_paths
)
ref_ids = [Path(p).name for p in result.image_paths]
localizer = CameraLocalizer.from_feedforward(result, images=ref_images, ids=ref_ids, extractor=LomaExtractor())

loc = localizer.localize(query_image, query_K)
print(f"pose:\n{loc.pose}\n inliers: {loc.n_inliers}/{loc.n_correspondences}")
```

Remove the downstream `ref_extrinsic`/`rot_err_deg`/`t_err_cm` consistency-check cell (cell 15) — it compared against the in-video reference pose, which no longer exists for an external query. Replace it with a short note cell or delete it; keep the point-cloud + localized-pose viz cell.

- [ ] **Step 5: Execute headless**

```bash
cd /workspace/collab-splats/docs/source/tutorials/07_localization
PYVISTA_OFF_SCREEN=1 /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=2400 --output /tmp/_07_out.ipynb localization.ipynb
```
Expected: executes; `loc.pose` is non-None with ≥4 correspondences.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/source/tutorials/07_localization/localization.ipynb
git commit -m "refactor(tutorial): nb07 localizes external query via estimate_intrinsics"
```

---

## Task 9: NB 03 — remove dead import, shared backend

**Files:**
- Modify: `docs/source/tutorials/03_splats/visualization.ipynb` (cell 2)

- [ ] **Step 1: Drop `import sys` and use the shared backend helper (cell 2)**

Remove the unused `import sys` line, and replace the inline `pv.set_jupyter_backend(...)` with:

```python
%run ../notebook_utils.py
set_notebook_backend()
```

- [ ] **Step 2: Execute headless to verify**

```bash
cd /workspace/collab-splats/docs/source/tutorials/03_splats
PYVISTA_OFF_SCREEN=1 /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 --output /tmp/_03viz_out.ipynb visualization.ipynb
```
Expected: executes without error.

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/source/tutorials/03_splats/visualization.ipynb
git commit -m "refactor(tutorial): nb03 drop dead import, use shared backend helper"
```

---

## Task 10: `.gitignore` + full-chain smoke run

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Ignore generated outputs**

Append to `/workspace/collab-splats/.gitignore`:

```
# Tutorial-regenerated artifacts
data/outputs/
```

- [ ] **Step 2: Confirm outputs are untracked**

```bash
cd /workspace/collab-splats
git status --porcelain data/outputs | head
git check-ignore data/outputs/frames.zarr
```
Expected: `git status` shows nothing under `data/outputs`; `check-ignore` echoes the path.

- [ ] **Step 3: Full clean-slate chain run (01→07)**

Simulates a new user: wipe outputs, run each stage in order, confirming every handoff regenerates.

```bash
cd /workspace/collab-splats/docs/source/tutorials
rm -rf /workspace/collab-splats/data/outputs
for nb in 01_preprocessing/keyframe_extraction 02_pointcloud/feedforward_methods \
          04_semantics/feature_extraction 04_semantics/segmentation \
          04_semantics/maskclip_vs_talk2dino 05_lifting/semantic_lifting \
          03_splats/visualization 07_localization/localization; do
  echo "=== $nb ==="
  PYVISTA_OFF_SCREEN=1 /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 --output /tmp/_chain_$(basename $nb).ipynb $nb.ipynb \
    || { echo "CHAIN FAILED at $nb"; break; }
done
```
Expected: every notebook executes with no missing-artifact error; final line is `07_localization` success. (06_mesh/03 splat notebooks that need splat data are outside this chain — see spec.)

- [ ] **Step 4: Run the full unit suite (no regressions)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/docs/ -v`
Expected: PASS (Task 2 + Task 3 tests).

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git add -f .gitignore
git commit -m "chore(tutorial): gitignore data/outputs (tutorial-regenerated)"
```

---

## Self-review notes

- **Spec coverage:** assets (T1), config declutter (T2), notebook_utils (T3), frames.zarr chain + canonical RECON (T4/T5/T6/T7), 07 external query (T8), dead code (T9), .gitignore + smoke verification (T10). BA/loop-closure excluded per spec.
- **Type consistency:** `FRAMES_ZARR` and `RECON` are defined once in `tutorial_config.py` (T2) and referenced by that name in T4–T8. `load_keyframe_paths(frames_zarr, export_dir)` / `set_notebook_backend()` signatures defined in T3 match all call sites.
- **Deferred:** swap `FrameStore.export()` → preloaded arrays once the concurrent `_preprocess` array interface lands (interim note above). Manual nbsphinx site build + 06_mesh/03_splats splat-data notebooks remain owed (unchanged by this plan).
