# Tutorial Notebook Rework Implementation Plan

> **SUPERSEDED after Task 5** by `docs/superpowers/plans/2026-10-02-tutorial-rework.md`
> (2026-10-02). Tasks 6–15 below target a pre-release API and must not be executed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `docs/source/tutorials` as nine self-contained notebooks that each run top-to-bottom from a clean checkout into their own temp directory, calling the `clean/final` API and adding no new library code.

**Architecture:** Delete the shared `data/outputs/` chain. `tutorial_config.py` keeps only committed inputs. `notebook_utils.py` gains exactly two helpers — `work_dir` (a tempdir) and `tutorial_scene` (a `Reconstructor` pointed at one). Every page builds its upstream inputs by calling `Reconstructor` stages, then drops to the low-level API only for the one stage it is teaching. A parametrized pytest gate enforces isolation and API truth per notebook, so it goes green one page at a time.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), Jupyter + nbformat + nbconvert, pytest, Sphinx + nbsphinx (`nbsphinx_execute = "never"`), PyVista, gsplat, Open3D.

**Spec:** `docs/superpowers/specs/2026-09-09-tutorial-rework-design.md`
**Branch:** `clean/tutorials` · **Worktree:** `.worktrees/tutorial-rework`

---

## Deviation from the approved spec — read first

The spec locked four bootstrap helpers, including a `bootstrap_splats` that reproduced the
~40 lines of trainer glue in `Reconstructor.splats()`. **This plan does not build them.**

`Reconstructor` already exposes every upstream stage as a method — `preprocess()`,
`build_pointcloud()`, `refine_poses()`, `extract_semantics()`, `splats()`, `mesh()`,
`build_localization_db()` — over a config deep-merged onto `configs/base.yaml`. Copying that
glue into `notebook_utils.py` would add library code the tutorial then teaches *instead of*
the real entry point, and it would drift the moment production changes.

So the helper surface is two functions and ~12 lines total:

```python
def work_dir(name: str) -> Path
def tutorial_scene(name: str, input_path: Path, **overrides) -> Reconstructor
```

The rule this serves: **a page calls `Reconstructor` for everything upstream of its subject,
and drops to the low-level API only for the subject itself.** No notebook defines a helper
function; where a plot needs one, the package already has it
(`collab_splats/utils/visualization.py`, `preproc/viz.py`, `localization/viz.py`).

---

## Environment preamble

Every command runs from the worktree root. The venv's editable finder hardcodes
`/workspace/collab-splats`, and `sys.path[0]`=cwd wins over `PYTHONPATH`, so a bare `python`
here silently tests the **main** tree and reports false green.

**The only safe invocation form:**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python <args>
```

Verify once before starting:

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
```

Expected exactly:
```
/workspace/collab-splats/.worktrees/tutorial-rework/collab_splats/__init__.py
```

`third_party/` is gitignored and already symlinked here. If it is empty, guarded tests
**skip instead of fail** and the gate lies:

```bash
ls /workspace/collab-splats/.worktrees/tutorial-rework/third_party | wc -l   # expect 10 (9 symlinked deps + README.md; .vda_fetch_done is hidden)
```

GPU work is serial — the cgroup cap is 46.6 GB and two concurrent feedforward runs OOM.

---

## Verified API surface

Every signature below was checked against `clean/tutorials` while writing this plan. Six
differ from what the spec's prose assumed; those are marked ⚠.

| Symbol | Signature |
|---|---|
| `Reconstructor` | `(config: dict, config_dir=DEFAULT_CONFIG_DIR)` — deep-merges over `configs/base.yaml`; requires `input_path`, `output_path` |
| `Reconstructor` props | `.images_dir`, `.backend_dir`, `.pointcloud_zarr`, `.semantics_cache_dir` |
| `Reconstructor` stages | `preprocess(overwrite=False) -> Path`, `build_pointcloud(overwrite=False) -> PointcloudResult`, `refine_poses(overwrite=False) -> PointcloudResult`, `extract_semantics(result=None, overwrite=False) -> Path`, `splats(overwrite=False) -> Path`, `mesh(result=None, overwrite=False) -> Path`, `build_localization_db(overwrite=False) -> Path` |
| `frames.write_frames` | `(dir, frames, records, provenance) -> list[Path]` |
| `frames.read_frames` ⚠ | `(dir, idxs=None) -> np.ndarray` — returns a stacked `(N,H,W,3)` array, not an iterable |
| `frames.frame_paths` / `frame_idx_from_path` | `(dir) -> list[Path]` / `(path) -> int` |
| `compute_video_quality` | `(video_path, *, output_path=None, motion_stride=None, workers=1) -> dict` |
| `filter_frame_quality` | `(report, *, sharpness_k=2.0, max_clipped_frac=0.25)` |
| `sample_uniform` / `sample_fps` / `sample_optical_flow` | see `preproc/sampling.py`; all take `report=` and optional `quality=` |
| `get_video_info` / `iter_frames` / `extract_frame` | `preproc/video.py` |
| `pointcloud_to_polydata` ⚠ | `collab_splats.utils.visualization` — **not** `pointcloud.viz`, which does not exist |
| `create_camera_frustum_pyvista` | `(pose, scale=0.02, aspect_ratio=1.33, fov=60)` |
| `load_checkpoint` ⚠ | `(path, device) -> (model, camera_opt, cam_to_world, intrinsics, image_ids, (H, W))` — a **6-tuple** |
| `render_views` ⚠ | `(model, camera_opt, cam_to_world, intrinsics, height, width) -> Iterator[dict[str, Tensor]]` — an **iterator**, one dict per view |
| `render_tsdf_inputs` ⚠ | `(ckpt_path, images_dir=None, device='cuda', depth_source='expected')` -> **5-tuple** `(depths, rgbs, cam_to_world, intrinsics, image_ids)` |
| `BaseFeatureExtractor.forward` | `(images: list) -> list[torch.Tensor]` |
| `.features_to_rgb` | `(feat) -> np.ndarray` |
| `score_queries` ⚠ | `(features, positive, negative=None, temperature=0.05, reduction='max')` — takes a `(C, H, W)` map **or** a `(P, D)` point array, returns `(H, W)` / `(P,)`. It **reduces over the positive prompts** (`reduction='max'`), so N prompts do not give N maps — call it once per prompt for a per-prompt map |
| `FeatureAutoencoder` ⚠ | `__init__(input_dim, latent_dim)` — `latent_dim` is **required**; there is no `recon_cosine` attribute, fidelity is reported through `fit(..., on_epoch=...)` / `target_cosine=` |
| `MobileSAMSegmentation` ⚠ | `collab_splats.semantics.segmentation` (module `mobile_sam`, not `mobilesamv2`); `__init__(strategy='object', device='cpu', ...)`; `segment(image) -> tuple[Tensor, Any] \| None` |
| `aggregate_masked_features` ⚠ | `(features, masks, resolution, final_resolution) -> (C, H, W)` — a mask-pooled **feature map**, not per-object vectors: each pixel gets the mean feature of the masks covering it |
| `load_point_features` | `(semantics_dir, batch_size=65_536) -> (P, D)` float32, L2-normalized — reads `*_lifted.zarr`, loads `*_ae.pt`, decodes in chunks. Exported from `collab_splats.semantics` |
| `CameraLocalizer` ⚠ | public API is `from_feedforward`, `localize`, `load_index`, `save_index`, `update_index`, `extrinsics`, `image_paths`, `frame_sources`, `add_localized_frame`, `clear_localized_frames` — **no `ranked_ref_frames`** |
| `LocalizationResult` | fields `pose, n_correspondences, n_inliers, pts2d, pts3d_matched, inlier_mask, pts2d_ref, ref_frame_indices, query_features, query_intrinsics, ref_hw` |
| `correspondences_for_ref` | `(loc, ref_idx, ref_image_hw=None) -> (query_px, ref_px, inlier_mask)`. Pass `ref_image_hw` when the display image differs from `loc.ref_hw`, or `ref_px` lands in the wrong pixel space |
| `plot_correspondences` | `(query_image, ref_image, query_px, ref_px, inlier_mask=None, max_pairs=200, warp_corners=False, show=True)` |
| `BundleAdjustmentConfig` ⚠ | fields are `max_reproj_error, lm_steps, shared_camera, vis_thresh, fine_tracking, min_inliers_per_frame, max_query_pts, query_frame_num, device, increment_size, tracks_cache_dir` — **there is no loss-history capture**, so no convergence plot is possible |
| `clean_repair_mesh` | `(mesh_path, min_area_frac=6e-6, max_gap_frac=0.01, max_hole_frac=0.0045)` |
| `invert_poses` | `collab_splats.geometry.transforms` |

---

## File Structure

| File | Responsibility |
|---|---|
| `docs/source/tutorials/tutorial_config.py` | Committed inputs only: `REPO_ROOT`, `VIDEO_PATH`, `QUERY_IMAGE` |
| `docs/source/tutorials/notebook_utils.py` | pyvista backend + `work_dir` + `tutorial_scene` |
| `docs/source/tutorials/index.rst` | Toctree over the nine pages |
| `tests/docs/test_tutorial_contract.py` | Gate: no shared cache, no cross-notebook deps, imports resolve, no notebook-defined helpers |
| `01_preprocessing/keyframe_extraction.ipynb` | Measure the video, then select from it |
| `02_pointcloud/reconstruction.ipynb` | Feedforward (VGGT-Omega) vs global SfM (InstantSfM) |
| `02_pointcloud/refinement.ipynb` | Bundle adjustment and loop closure |
| `03_splats/train_splats.ipynb` | Scaffold-2DGS |
| `04_semantics/feature_extraction.ipynb` | MaskCLIP vs Talk2DINO |
| `04_semantics/segmentation.ipynb` | Masks, and masks → per-object features |
| `04_semantics/lifting_and_query.ipynb` | 2D features → 3D points → text query |
| `05_mesh/tsdf_mesh.ipynb` | TSDF from feedforward depth vs splat renders |
| `06_localization/localization.ipynb` | Query image → pose |

**Deleted:** `02_pointcloud/{colmap_sfm,feedforward_mesh,slam_loop_closure}.ipynb`, `04_semantics/maskclip_vs_talk2dino.ipynb`, `evals/ground_truth_evals.ipynb`, and the `05_lifting/`, `06_mesh/`, `07_localization/`, `evals/` directories.

**Authoring convention for every notebook task:** notebooks are written by a Python heredoc
using `nbformat`, never hand-edited as JSON:

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
/opt/venv/reconstruction/bin/python - <<'PY'
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell("..."), nbf.v4.new_code_cell("...")]
nb.metadata = {
    # `name` is the REGISTERED kernel; the venv registers only "python3". "reconstruction"
    # is the display name the existing notebooks carry, and naming it here fails with
    # NoSuchKernel at execute time.
    "kernelspec": {"display_name": "reconstruction", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.15"},
}
nbf.write(nb, "docs/source/tutorials/<dir>/<name>.ipynb")
PY
```

Every notebook opens with these two cells, **not repeated** in the per-task cell lists:

```python
# cell 1
%load_ext autoreload
%autoreload 2

# cell 2
%run ../tutorial_config.py
%run ../notebook_utils.py
set_notebook_backend()
```

---

### Task 1: Prune the notebook set and rewrite the toctree

**Files:**
- Delete: `02_pointcloud/colmap_sfm.ipynb`, `02_pointcloud/feedforward_mesh.ipynb`, `02_pointcloud/slam_loop_closure.ipynb`, `04_semantics/maskclip_vs_talk2dino.ipynb`, `evals/ground_truth_evals.ipynb`
- Rename: `02_pointcloud/feedforward_methods.ipynb` → `02_pointcloud/reconstruction.ipynb`; `02_pointcloud/bundle_adjustment.ipynb` → `02_pointcloud/refinement.ipynb`; `05_lifting/semantic_lifting.ipynb` → `04_semantics/lifting_and_query.ipynb`; `06_mesh/splats_mesh.ipynb` → `05_mesh/tsdf_mesh.ipynb`; `07_localization/localization.ipynb` → `06_localization/localization.ipynb`
- Modify: `docs/source/tutorials/index.rst`

Renames use `git mv` so history follows. Renamed notebooks still hold their old broken
content after this task — each is rewritten later. This task fixes only the *set*.

- [ ] **Step 1: Delete and rename**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git rm -q docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb \
          docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb \
          docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb \
          docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb \
          docs/source/tutorials/evals/ground_truth_evals.ipynb
mkdir -p docs/source/tutorials/05_mesh docs/source/tutorials/06_localization
git mv docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb docs/source/tutorials/02_pointcloud/reconstruction.ipynb
git mv docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb   docs/source/tutorials/02_pointcloud/refinement.ipynb
git mv docs/source/tutorials/05_lifting/semantic_lifting.ipynb       docs/source/tutorials/04_semantics/lifting_and_query.ipynb
git mv docs/source/tutorials/06_mesh/splats_mesh.ipynb               docs/source/tutorials/05_mesh/tsdf_mesh.ipynb
git mv docs/source/tutorials/07_localization/localization.ipynb      docs/source/tutorials/06_localization/localization.ipynb
rmdir docs/source/tutorials/05_lifting docs/source/tutorials/06_mesh \
      docs/source/tutorials/07_localization docs/source/tutorials/evals 2>/dev/null || true
```

- [ ] **Step 2: Verify the set is exactly nine**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
find docs/source/tutorials -name '*.ipynb' | sort
```

Expected exactly:
```
docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
docs/source/tutorials/02_pointcloud/reconstruction.ipynb
docs/source/tutorials/02_pointcloud/refinement.ipynb
docs/source/tutorials/03_splats/train_splats.ipynb
docs/source/tutorials/04_semantics/feature_extraction.ipynb
docs/source/tutorials/04_semantics/lifting_and_query.ipynb
docs/source/tutorials/04_semantics/segmentation.ipynb
docs/source/tutorials/05_mesh/tsdf_mesh.ipynb
docs/source/tutorials/06_localization/localization.ipynb
```

- [ ] **Step 3: Rewrite the toctree**

Replace the entire contents of `docs/source/tutorials/index.rst`:

```rst
Tutorials
=========

Each page is self-contained: it builds everything it needs into its own temporary
directory and runs top to bottom. There is no shared cache and no required order —
start anywhere.

Every page drives the pipeline through ``Reconstructor`` for the stages upstream of its
subject, then calls the underlying API directly for the stage it is teaching. The frame
and step counts are a small fast profile; each one sits in the notebook's first cell with
the production value beside it.

.. toctree::
   :maxdepth: 1
   :caption: 01 · Preprocessing

   01_preprocessing/keyframe_extraction

.. toctree::
   :maxdepth: 1
   :caption: 02 · Pointcloud

   02_pointcloud/reconstruction
   02_pointcloud/refinement

.. toctree::
   :maxdepth: 1
   :caption: 03 · Splats

   03_splats/train_splats

.. toctree::
   :maxdepth: 1
   :caption: 04 · Semantics

   04_semantics/feature_extraction
   04_semantics/segmentation
   04_semantics/lifting_and_query

.. toctree::
   :maxdepth: 1
   :caption: 05 · Mesh

   05_mesh/tsdf_mesh

.. toctree::
   :maxdepth: 1
   :caption: 06 · Localization

   06_localization/localization
```

- [ ] **Step 4: Verify every toctree entry resolves**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
grep -E '^   [0-9]{2}_' docs/source/tutorials/index.rst | tr -d ' ' | \
  while read -r e; do test -f "docs/source/tutorials/$e.ipynb" && echo "OK   $e" || echo "MISS $e"; done
```

Expected: nine `OK`, zero `MISS`.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add -A docs/source/tutorials
git commit -m "docs(tutorials): prune to the nine-page set and renumber

Deletes colmap_sfm (4-cell stub on a dead kernel), feedforward_mesh
(duplicate of the mesh page), slam_loop_closure (merges into refinement),
maskclip_vs_talk2dino (already a section of feature_extraction) and
evals/ground_truth_evals (runs a script that does not exist). Renames the
survivors into 01-06 and rewrites the toctree.

Content is still the old content; each page is rewritten in a later commit.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The contract gate

**Files:**
- Create: `tests/docs/__init__.py` (empty)
- Create: `tests/docs/test_tutorial_contract.py`

Fails today for every page, greens one page at a time. It encodes the three properties the
rework is for — isolation, API truth, and no notebook-defined library code.

- [ ] **Step 1: Write the failing test**

Create empty `tests/docs/__init__.py`, then `tests/docs/test_tutorial_contract.py`:

```python
"""
Contract gate for the tutorial notebooks.

- isolation: no notebook may reference the retired shared output cache
- independence: no notebook may depend on another notebook having run
- API truth: every collab_splats symbol a notebook imports must resolve
- no new code: a notebook demonstrates the package, it does not define helpers
"""

import ast
import importlib
import json
import re
from pathlib import Path

import pytest

TUTORIALS = Path(__file__).resolve().parents[2] / "docs" / "source" / "tutorials"
NOTEBOOKS = sorted(TUTORIALS.rglob("*.ipynb"))

# Names that only existed to thread state between notebooks. Their absence is the
# isolation property: a page that cannot name the shared cache cannot read from it.
BANNED_TOKENS = {
    "data/outputs": "the retired shared output directory",
    "OUTPUT_DIR": "shared output dir from the old tutorial_config",
    "IMAGES_DIR": "shared keyframe dir from the old tutorial_config",
    "TUTORIAL_CACHE": "cross-notebook scratch cache",
    "sys.path.insert": "import hack; the tutorial imports installed packages only",
}

# "run 02_pointcloud/feedforward_methods.ipynb first" and friends
RUN_FIRST = re.compile(r"run\s+\S*\d{2}[_/]\S*\.ipynb", re.IGNORECASE)

FIRST_PARTY = {"collab_splats", "evals"}


def _code_sources(nb_path: Path) -> list[str]:
    """
    Source text of every code cell in a notebook.
    """
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _strip_magics(src: str) -> str:
    """
    Blank out IPython magic and shell lines so the cell parses as Python.
    """
    return "\n".join("" if ln.lstrip().startswith(("%", "!", "?")) else ln for ln in src.split("\n"))


def _parsed_cells(nb_path: Path):
    """
    Every code cell that parses as Python, as an AST module.
    """
    for src in _code_sources(nb_path):
        try:
            yield ast.parse(_strip_magics(src))
        except SyntaxError:
            continue


def test_notebook_set_is_the_nine_pages():
    """
    The set itself is part of the contract — a stray notebook is an ungated page.
    """
    names = sorted(p.relative_to(TUTORIALS).as_posix() for p in NOTEBOOKS)
    assert names == [
        "01_preprocessing/keyframe_extraction.ipynb",
        "02_pointcloud/reconstruction.ipynb",
        "02_pointcloud/refinement.ipynb",
        "03_splats/train_splats.ipynb",
        "04_semantics/feature_extraction.ipynb",
        "04_semantics/lifting_and_query.ipynb",
        "04_semantics/segmentation.ipynb",
        "05_mesh/tsdf_mesh.ipynb",
        "06_localization/localization.ipynb",
    ]


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_shared_cache(nb: Path):
    """
    A page may not name the retired cross-notebook cache.
    """
    text = "\n".join(_code_sources(nb))
    hits = [f"{tok} ({why})" for tok, why in BANNED_TOKENS.items() if tok in text]
    assert not hits, f"{nb.name} still references: {'; '.join(hits)}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_dependency_on_another_notebook(nb: Path):
    """
    A page may not tell the reader to go run a different page first.
    """
    match = RUN_FIRST.search("\n".join(_code_sources(nb)))
    assert match is None, f"{nb.name} depends on another notebook: {match.group(0)!r}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_defines_no_helpers(nb: Path):
    """
    Notebooks demonstrate the package; helper code belongs in collab_splats or notebook_utils.
    """
    defined = [
        node.name
        for tree in _parsed_cells(nb)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert not defined, f"{nb.name} defines {defined} — call the package instead"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_imports_resolve(nb: Path):
    """
    Every first-party symbol a page imports exists on this branch.
    """
    missing: list[str] = []
    for tree in _parsed_cells(nb):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if not node.module or node.module.split(".")[0] not in FIRST_PARTY:
                    continue
                try:
                    mod = importlib.import_module(node.module)
                except ImportError as exc:
                    missing.append(f"{node.module} ({exc})")
                    continue
                missing += [f"{node.module}.{a.name}" for a in node.names if not hasattr(mod, a.name)]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in FIRST_PARTY:
                        continue
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as exc:
                        missing.append(f"{alias.name} ({exc})")
    assert not missing, f"{nb.name} imports names that do not exist: {missing}"
```

- [ ] **Step 2: Run it and confirm it fails for the right reasons**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py -v 2>&1 | tail -45
```

Expected: `test_notebook_set_is_the_nine_pages` PASSES (Task 1 established the set); many
parametrized cases FAIL. These specific failures prove the gate has teeth:

- `test_imports_resolve[keyframe_extraction]` — `collab_splats.preproc.sample_frames` / `score_frames`
- `test_no_shared_cache[reconstruction]`, `[tsdf_mesh]`, `[localization]`, `[lifting_and_query]` — `OUTPUT_DIR` / `data/outputs`
- `test_no_shared_cache[train_splats]` — `sys.path.insert`
- `test_no_dependency_on_another_notebook[tsdf_mesh]` — the `run 03_splats/train_splats.ipynb first` assert
- `test_defines_no_helpers[...]` — several pages define local plotting functions

If collection errors on a torch/CUDA import instead of failing, the environment is wrong —
re-run the provenance check.

- [ ] **Step 3: Commit the gate red**

Committing red is deliberate: it is the specification of the remaining work.

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add tests/docs/__init__.py tests/docs/test_tutorial_contract.py
git commit -m "test(docs): gate tutorial isolation, API truth and no-new-code

Parametrized per notebook, so it greens one page at a time. Currently red: it
reproduces the API breaks, the shared-cache references, and the notebook-local
helper functions the rework removes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `tutorial_config.py` — inputs only

**Files:**
- Modify: `docs/source/tutorials/tutorial_config.py` (full rewrite)
- Modify: `tests/docs/test_tutorial_config.py` (full rewrite)

`tests/docs/` already exists and already covers this module — its current test asserts
exactly the five names this task deletes, so it is rewritten here rather than left to fail.

- [ ] **Step 1: Replace the test with one that asserts the new contract**

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
    assert ns["REPO_ROOT"] == REPO_ROOT
    assert ns["VIDEO_PATH"] == REPO_ROOT / "data/tutorial/tutorial_example-video.mp4"
    assert ns["QUERY_IMAGE"] == REPO_ROOT / "data/tutorial/tutorial_example-frame.jpg"


def test_exposes_inputs_only():
    """
    The retired names are the shared output cache. Their absence is the isolation property.
    """
    ns = _load()
    for gone in ("OUTPUT_DIR", "IMAGES_DIR", "RECON", "TUTORIAL_CACHE", "MAX_FRAMES",
                 "DATASET", "BASE_DIR", "FRAMES", "FRAMES_ZARR", "_infer_video_path"):
        assert gone not in ns, f"{gone} should be removed"
```

- [ ] **Step 2: Replace the file contents**

```python
"""Committed inputs for the tutorial notebooks. `%run ../tutorial_config.py` to load.

Inputs only. Every notebook writes its own outputs into its own temporary directory
(see notebook_utils.work_dir) — there is no shared output location and no notebook
depends on another having run.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # docs/source/tutorials/ -> repo root

# ── Committed inputs (read-only) ──────────────────────────────────────────────
VIDEO_PATH = REPO_ROOT / "data/tutorial/tutorial_example-video.mp4"
QUERY_IMAGE = REPO_ROOT / "data/tutorial/tutorial_example-frame.jpg"

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"missing {VIDEO_PATH} — see data/tutorial/README.md")
```

- [ ] **Step 3: Verify it loads and the retired names are gone**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
/opt/venv/reconstruction/bin/python -c "
ns = {'__file__': 'docs/source/tutorials/tutorial_config.py'}
exec(open('docs/source/tutorials/tutorial_config.py').read(), ns)
print('VIDEO_PATH :', ns['VIDEO_PATH'].exists())
print('QUERY_IMAGE:', ns['QUERY_IMAGE'].exists())
gone = [n for n in ('OUTPUT_DIR','IMAGES_DIR','RECON','TUTORIAL_CACHE','MAX_FRAMES') if n in ns]
assert not gone, gone
print('retired names:', gone)
"
```

Expected: `True`, `True`, `retired names: []`.

- [ ] **Step 4: Run the test**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_config.py -q 2>&1 | tail -5
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add docs/source/tutorials/tutorial_config.py tests/docs/test_tutorial_config.py
git commit -m "docs(tutorials): reduce tutorial_config to committed inputs

Drops OUTPUT_DIR, IMAGES_DIR, RECON, TUTORIAL_CACHE and MAX_FRAMES — the
cross-notebook cache. Frame counts are per-page now.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `notebook_utils.py` — two helpers, no glue

**Files:**
- Modify: `docs/source/tutorials/notebook_utils.py`
- Modify: `tests/docs/test_notebook_utils.py` — it already exists and already covers
  `set_notebook_backend`; that test is kept and the new ones are added beside it

- [ ] **Step 1: Write the failing test**

Rewrite `tests/docs/test_notebook_utils.py`, keeping its existing backend test:

```python
"""
Unit tests for the tutorial helpers.

Only the cheap properties are asserted: the tempdir contract and that tutorial_scene
returns a Reconstructor whose paths point inside that tempdir. The GPU stages are the
package's own, tested in tests/wrapper — the tutorial does not re-test them.
"""

import importlib.util
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[2] / "docs" / "source" / "tutorials" / "notebook_utils.py"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def utils():
    """
    Load notebook_utils.py as a module.
    """
    spec = importlib.util.spec_from_file_location("tutorial_notebook_utils", UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_set_notebook_backend_is_callable(utils, monkeypatch):
    import pyvista as pv

    called = {}
    monkeypatch.setattr(pv, "set_jupyter_backend", lambda b: called.setdefault("b", b))
    utils.set_notebook_backend()
    assert called["b"] in ("static", "trame")


def test_work_dir_is_fresh_each_call(utils):
    a, b = utils.work_dir("demo"), utils.work_dir("demo")
    assert a.is_dir() and b.is_dir()
    assert a != b, "two runs of a page must not collide"
    assert "demo" in a.name


def test_work_dir_is_outside_the_repo(utils):
    """
    A page's outputs must never land in the working tree, or they get committed by accident.
    """
    assert REPO not in utils.work_dir("demo").parents


def test_tutorial_scene_points_every_path_at_its_tempdir(utils):
    """
    tutorial_scene is the isolation boundary: every artifact path must sit under the tempdir.
    """
    video = REPO / "data/tutorial/tutorial_example-video.mp4"
    r = utils.tutorial_scene("demo", video, preproc={"max_frames": 4})

    work = Path(r.config["output_path"])
    assert REPO not in work.parents
    for path in (r.images_dir, r.backend_dir, r.pointcloud_zarr, r.semantics_cache_dir):
        assert work in path.parents or path == work


def test_tutorial_scene_merges_overrides_over_base_yaml(utils):
    """
    Overrides are deep-merged, so a partial block keeps the base.yaml defaults beside it.
    """
    video = REPO / "data/tutorial/tutorial_example-video.mp4"
    r = utils.tutorial_scene("demo", video, preproc={"max_frames": 4})

    assert r.config["preproc"]["max_frames"] == 4          # overridden
    assert r.config["preproc"]["frame_selection"] == "fps"  # from base.yaml
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_notebook_utils.py -v 2>&1 | tail -20
```

Expected: FAIL — `module 'tutorial_notebook_utils' has no attribute 'work_dir'`.

- [ ] **Step 3: Write the implementation**

Replace the contents of `docs/source/tutorials/notebook_utils.py`:

```python
"""Shared helpers for the tutorial notebooks. `%run ../notebook_utils.py` to load.

Three notebook-only concerns, kept out of collab_splats/:

- pyvista backend selection (static under nbconvert, interactive otherwise)
- work_dir, so a page's outputs land in a fresh temporary directory
- tutorial_scene, a Reconstructor pointed at one

There is deliberately no bootstrap glue here. Reconstructor already exposes every
stage — preprocess, build_pointcloud, refine_poses, extract_semantics, splats, mesh,
build_localization_db — so a page calls those for everything upstream of its subject
and drops to the underlying API only for the stage it is teaching.
"""

import os
import tempfile
from pathlib import Path

import pyvista as pv

from collab_splats.wrapper.reconstructor import Reconstructor


def set_notebook_backend() -> None:
    """
    Static pyvista backend when headless (nbconvert), interactive trame otherwise.
    """
    pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")


def work_dir(name: str) -> Path:
    """
    A fresh temporary directory for one notebook run, printed and left behind.

    - honours $TMPDIR; never inside the repo, so outputs cannot be committed by accident
    - deliberately not cleaned up: the reader opens it afterwards to inspect artifacts
    - the OS reaps it, so nothing accumulates

    Args:
        name: short label for the page, used in the directory name.

    Returns:
        The created directory.
    """
    path = Path(tempfile.mkdtemp(prefix=f"collab_splats_tutorial_{name}_"))
    print(f"work dir: {path}")
    return path


def tutorial_scene(name: str, input_path: Path, **overrides) -> Reconstructor:
    """
    A Reconstructor writing into this page's own temporary directory.

    - overrides are deep-merged over configs/base.yaml by Reconstructor itself
    - every artifact path (images_dir, backend_dir, pointcloud_zarr) lands under the tempdir

    Args:
        name: short label for the page.
        input_path: source video or image directory.
        **overrides: config blocks to merge over base.yaml, e.g. preproc={"max_frames": 16}.

    Returns:
        A configured Reconstructor; call its stage methods to build what the page needs.
    """
    work = work_dir(name)
    return Reconstructor({"input_path": str(input_path), "output_path": str(work), **overrides})
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_notebook_utils.py -v 2>&1 | tail -20
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add docs/source/tutorials/notebook_utils.py tests/docs/test_notebook_utils.py
git commit -m "docs(tutorials): add work_dir and tutorial_scene helpers

Two functions, no pipeline glue. Reconstructor already exposes every stage over
a config deep-merged onto base.yaml, so tutorial_scene just points one at a
tempdir and the notebooks call its stage methods. The spec's bootstrap_keyframes
/ bootstrap_reconstruction / bootstrap_splats would have duplicated
Reconstructor.splats() into the docs and drifted from it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notebook tasks (5-13) — shared procedure

Each has the same four steps: write the notebook, run its gate slice, execute it, commit.
Written out per task so tasks can be read out of order.

---

### Task 5: `01_preprocessing/keyframe_extraction.ipynb`

**Files:** Modify `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` (full rewrite)

Subject: measure the video, then select from it. This page *is* preproc, so it calls the
`qa`/`sampling` API directly and needs no `Reconstructor`. It repairs the two dead imports
(`sample_frames`, `score_frames`).

Verified contracts this task is written against — all six differ from the first draft:

| Call | Real contract |
| --- | --- |
| `get_video_info` | key is `total_frames`, not `frame_count` |
| `filter_frame_quality(report, **thresholds)` | returns `(N,) bool` ndarray, not a dict with `["eligible"]` |
| samplers' `quality=` | the *threshold overrides* (`sharpness_k`, `max_clipped_frac`, `on_empty_slot`) — **not** the mask. The samplers re-derive the pool through `_eligible`. |
| samplers | return `(frames, records)` — RGB `(H, W, 3)` uint8 **and** their rows, so nothing needs to re-decode |
| `plot_selection` | `(total_frames, fps_indices=None, of_indices=None)` — no report, no out_dir, no `selected=` |
| report columns | `blur`, `laplacian`, `exposure_{mean,median,std}`, `clipped_{low,high}_frac`, `frame_idx`. There is no `sharpness` and no `mean_luma`. |

Two consequences for the cells: the thresholds live in one `QUALITY` dict passed to both
`filter_frame_quality` and the samplers, so the notebook shows the gate the samplers
actually apply; and §5 writes the frames the sampler already handed back, so the page
never re-decodes.

The report plotters (`plot_photometric`, `plot_motion`, `plot_frame_extremes`,
`plot_correlation`) save a PNG and return its `Path` — headless by design, they never call
`plt.show()`. So the notebook displays the returned path. The notebook-only plotters
(`plot_selection`, `plot_frame_grid`) do call `plt.show()` and are used bare.

`plot_frame_scores` is **not** used: it wants a per-frame score series, and
`sample_optical_flow` returns rows for the selected frames only (every one `selected: True`).

**Committed-size rule, measured here and binding on every later page.** `plot_frame_extremes`
saves a lossless PNG at `_THUMB_DPI = 150` with `figsize=(3.3 * n, 14)`, drawing full-resolution
video frames. Measured on `tutorial_example-video.mp4`: 3.15 MB at `n=2`, 4.90 at `n=3`, 6.65 at
`n=4`, 10.07 at `n=6`. The DPI is a deliberate production choice for spot checks and is **not**
changed for the docs; the notebook passes `n=2` instead. For reference the whole previous
14-notebook set was 9.8 MB, and the page this replaces was 2.08 MB, so any cell whose output
runs to megabytes gets checked against those numbers before it is committed.

- [ ] **Step 1: Write the notebook**

Cells after the two standard openers:

**md**
```markdown
# Keyframe Extraction

A reconstruction never sees the whole video. This page covers the two steps that decide
what it does see: **measure** the clip, then **select** from it.

They are deliberately separate. `compute_video_quality` produces measurements and no
verdicts; `filter_frame_quality` is the only place a frame is judged. Keeping them apart
means you can re-threshold without re-decoding.

In a pipeline run these are one call — `Reconstructor.preprocess()`. This page opens it up.
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
N_FRAMES = 16        # production runs hundreds; base.yaml caps at 300
TARGET_FPS = 2.0     # base.yaml default: samples per second of video

# The gate's thresholds, in one place: the samplers re-derive the eligible pool
# from these, so passing the same dict is what makes §3 and §4 agree.
QUALITY = {"sharpness_k": 2.0, "max_clipped_frac": 0.25}

WORK = work_dir("preprocessing")
```

**md** — `## §1 — Measure`: one decode of the whole clip, photometry per frame and motion
per pair. Everything below reads the report; nothing re-decodes.

**code**
```python
from collab_splats.preproc.qa import compute_video_quality
from collab_splats.preproc.video import get_video_info

info = get_video_info(VIDEO_PATH)
print(f"{info['total_frames']} frames  {info['width']}x{info['height']}  {info['fps']:.2f} fps")

report = compute_video_quality(VIDEO_PATH, output_path=WORK / "video_quality.json")
print("frame columns:", sorted(report["frames"]))
print("pair columns :", sorted(report["pairs"]))
```

**md** — `## §2 — Photometry and motion`: the plotters draw every frame and every pair as
measured — raw columns, no thresholds and no verdicts. Each saves a PNG and returns its
path.

**code**
```python
from IPython.display import Image, display

from collab_splats.preproc.viz import plot_correlation, plot_motion, plot_photometric

display(Image(str(plot_photometric(report, WORK))))
display(Image(str(plot_motion(report, WORK))))

# blur is per-frame, translation_px per-pair: plot_correlation reads the frame column
# at each pair's first frame so the two line up sample for sample
display(Image(str(plot_correlation(report, "blur", "translation_px", WORK))))
```

**md** — `## §3 — The quality gate`: `sharpness_k` is a per-video MAD z-score on
`log(laplacian)` — larger keeps more. It is relative because a static blur threshold cut
zero frames on real footage. `max_clipped_frac` is absolute: a pixel at 0 or 255 recorded
nothing recoverable.

**code**
```python
from collab_splats.preproc.sampling import filter_frame_quality

eligible = filter_frame_quality(report, **QUALITY)
print(f"eligible: {eligible.sum()} / {eligible.size}  ({eligible.mean():.1%})")
```

**md** — `### Blur extremes` — the frames the gate is reacting to.

**code**
```python
from collab_splats.preproc.viz import plot_frame_extremes

# n=2, not the default 6: the montage is a 150-DPI lossless PNG of full-resolution
# video frames, so it dominates the committed page — 3.1 MB at n=2 against 10.1 MB at n=6.
display(Image(str(plot_frame_extremes(report, VIDEO_PATH, WORK, column="blur", n=2))))
```

**md** — `## §4 — Three samplers`: `sample_uniform` (count is the contract),
`sample_fps` (spacing is the contract — it fixes the baseline between frames regardless of
clip length, which a count cannot do), `sample_optical_flow` (motion and coverage, so a
slow pan yields fewer frames). All three select only from the eligible pool and hand back
the decoded frames alongside their records.

**code**
```python
from collab_splats.preproc.sampling import sample_fps, sample_optical_flow, sample_uniform

uni_frames, uni_records = sample_uniform(str(VIDEO_PATH), max_frames=N_FRAMES, report=report, quality=QUALITY)
fps_frames, fps_records = sample_fps(
    str(VIDEO_PATH), fps=TARGET_FPS, report=report, max_frames=N_FRAMES, quality=QUALITY
)
of_frames, of_records = sample_optical_flow(str(VIDEO_PATH), report=report, max_frames=N_FRAMES, quality=QUALITY)

for label, records in [("uniform", uni_records), ("fps", fps_records), ("optical flow", of_records)]:
    idx = [r["frame_idx"] for r in records]
    print(f"{label:13s} n={len(idx):3d}  first={idx[:4]}  last={idx[-4:]}")
```

**code**
```python
from collab_splats.preproc.viz import plot_selection

plot_selection(
    info["total_frames"],
    fps_indices=[r["frame_idx"] for r in fps_records],
    of_indices=[r["frame_idx"] for r in of_records],
)
```

**md** — `## §5 — Write the keyframe store`: `frame_NNNNNN.png` named by *source* index plus
`frames.json` carrying records and provenance. Every downstream stage reads exactly this.
The sampler already returned the decoded frames, so this writes them straight out.

**code**
```python
from collab_splats.preproc.frames import write_frames
from collab_splats.preproc.viz import plot_frame_grid

images_dir = WORK / "images"
paths = write_frames(images_dir, fps_frames, fps_records, {"source": str(VIDEO_PATH), "sampler": "fps"})
print(f"wrote {len(paths)} frames to {images_dir}")

plot_frame_grid(fps_frames[:12], title=f"fps keyframes (n={len(fps_frames)}, showing first 12)")
```

**md** — `## §6 — Choosing a sampler`: fps for steady handheld capture (the base.yaml
default); uniform when you need an exact count; optical flow for variable-speed capture,
where spacing in time is not spacing in space.

Plus the caveat the §4 output makes visible: `sample_optical_flow` walks the video in order
and breaks the moment it holds `max_frames`, so at N_FRAMES=16 every index it returns comes
from the first ten seconds (measured: `first=[0, 15, 36, 51] last=[226, 230, 240, 246]`
against 2388 total frames). The section says so rather than leaving the reader to explain
the printed indices.

- [ ] **Step 2: Run the gate slice**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k keyframe_extraction -v 2>&1 | tail -15
```

Expected: 4 passed.

- [ ] **Step 3: Execute**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
```

Expected: exits 0. A `CellExecutionError` names the failing cell — fix it and re-run.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
git commit -m "docs(tutorials): rewrite keyframe extraction on the qa/sampling API

Replaces the dead preproc.sample_frames and preproc.score_frames imports with
the real two-step API: compute_video_quality measures, filter_frame_quality
judges, the three samplers select. Plots come from preproc.viz rather than
notebook-local helpers. Writes into its own tempdir.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `02_pointcloud/reconstruction.ipynb`

**Files:** Modify `docs/source/tutorials/02_pointcloud/reconstruction.ipynb` (full rewrite)

Subject: keyframes in, sparse reconstruction out, by both routes. The routes are a config
choice, so the page shows the config, then opens up the feedforward creator underneath it.

- [ ] **Step 1: Write the notebook**

**md**
```markdown
# Pointcloud Reconstruction

Two routes from keyframes to a sparse reconstruction, selected by `pointcloud.method`:

- **feedforward** — one network pass predicts points, poses and depth together. Seconds,
  no incremental solve, arbitrary scale. Shown here with VGGT-Omega; `vggtx`,
  `mapanything` and `loger` are the same call with a different `backend`.
- **sfm** — correspondences and a global solve, with monocular metric depth as a prior.
  Slower, metrically scaled. Shown here with InstantSfM.

Both write the same `pointcloud.zarr`, so every downstream page consumes either.
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
N_FRAMES = 16        # production runs hundreds

ff = tutorial_scene(
    "reconstruction-ff",
    VIDEO_PATH,
    preproc={"max_frames": N_FRAMES},
    pointcloud={"method": "feedforward", "backend": "vggt_omega"},
)
ff.preprocess()
```

**md** — `## §1 — Feedforward`: `build_pointcloud` runs the creator, cleans the cloud
(`pointcloud.clean`), writes the COLMAP model, `sparse_pc.ply` and `pointcloud.zarr`.

**code**
```python
ff_result = ff.build_pointcloud()
print(f"points     : {ff_result.points.shape}")
print(f"extrinsics : {ff_result.extrinsics.shape}")
print(f"zarr       : {ff.pointcloud_zarr}")
```

**md** — `### What the zarr holds`: `PointcloudResult` carries the COLMAP model at original
resolution; the zarr holds the richer `FeedforwardResult` — depth, confidence and the
model-resolution intrinsics the mesh and semantics stages need.

**code**
```python
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

ff_zarr = FeedforwardResult.load_zarr(ff.pointcloud_zarr)
print(f"depth      : {ff_zarr.depth.shape}")
print(f"confidence : {ff_zarr.confidence.shape}")
print(f"model res  : {ff_zarr.model_width}x{ff_zarr.model_height}")
```

**md** — `### The cloud and its cameras`

**code**
```python
import pyvista as pv
from collab_splats.geometry.transforms import invert_poses
from collab_splats.utils.visualization import create_camera_frustum_pyvista, pointcloud_to_polydata

pl = pv.Plotter()
pl.add_mesh(
    pointcloud_to_polydata(ff_result.points, RGB=ff_result.colors),
    scalars="RGB", rgb=True, point_size=2, render_points_as_spheres=True,
)
for pose in invert_poses(ff_result.extrinsics):
    pl.add_mesh(create_camera_frustum_pyvista(pose, scale=0.05), color="lightblue")
pl.show()
```

**md** — `## §2 — Global SfM`: the same two calls with `method: sfm`. ⚠ There is **no**
`use_depths` flag — `_run_sfm` generates Video-Depth-Anything metric depth unconditionally,
then aligns the COLMAP model to it, so this is the full production path with nothing to
switch off. ⚠ `build_pointcloud` also raises a `UserWarning` that `method='sfm'` is
experimental and not production-tested; the page states that rather than letting a stray
warning in the output speak for it.

**code**
```python
sfm = tutorial_scene(
    "reconstruction-sfm",
    VIDEO_PATH,
    preproc={"max_frames": N_FRAMES},
    pointcloud={"method": "sfm", "backend": "instantsfm"},
)
sfm.preprocess()
sfm_result = sfm.build_pointcloud()
print(f"registered images: {len(sfm_result.image_paths)}")
print(f"points           : {sfm_result.points.shape}")
```

**md** — `### Retriangulation`, as a note, not a run:
````markdown
`pointcloud.instantsfm.retriangulation: true` adds a GLOMAP-style pass: after the global
solve, tracks are re-triangulated from the full pre-filter set and up to five further BA
rounds run. It buys accuracy on longer sequences at real runtime cost and is off by
default.

It belongs here rather than on the refinement page — it is a stage *inside* the SfM solve,
not a pass applied to a finished reconstruction. The refinement page's two passes are
feedforward-only, and the pipeline refuses them on this path.
````

**md** — `## §3 — Comparing the two`: the clouds are not in the same frame — feedforward
scale is arbitrary, SfM is metric — so this compares density and coverage, not position.

**code**
```python
import numpy as np

for label, res in [("vggt_omega", ff_result), ("instantsfm", sfm_result)]:
    extent = res.points.max(axis=0) - res.points.min(axis=0)
    print(f"{label:12s} points={len(res.points):>9,}  frames={len(res.image_paths):3d}  extent={np.round(extent, 2)}")
```

- [ ] **Step 2: Run the gate slice**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k "reconstruction and not refinement" -v 2>&1 | tail -15
```

Expected: 4 passed.

- [ ] **Step 3: Execute** (two reconstructions plus VDA depth — allow ~25 min)

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=5400 \
  docs/source/tutorials/02_pointcloud/reconstruction.ipynb
```

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add docs/source/tutorials/02_pointcloud/reconstruction.ipynb
git commit -m "docs(tutorials): rewrite reconstruction around both pointcloud methods

Drives both routes through Reconstructor.build_pointcloud, which is how the
pipeline selects them, then opens the zarr to show what feedforward carries
beyond the COLMAP model. Drops the cache branch and the hand-rolled pyvista
glue — utils.visualization already has it. Notes retriangulation as an
SfM-internal stage rather than a refinement pass.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `02_pointcloud/refinement.ipynb`

**Files:** Modify `docs/source/tutorials/02_pointcloud/refinement.ipynb` (full rewrite; merges the deleted `slam_loop_closure.ipynb`)

Subject: improving a first pass. Both passes are **feedforward-only** — the config refuses
them on SfM, and refuses them together. This page needs 24 frames, not 16:
`LoopClosureConfig.submap_size` defaults to 20 and `LoopClosure._enough_frames()`
(`collab_splats/geometry/loop_closure/wrapper.py:207-210`) falls back to plain inference
below that, *silently*. At 16 frames the page would render a loop-closure section that
never ran loop closure.

**Correction — there IS a convergence plot, and it needs no new code.** The earlier note
here said `BundleAdjustmentConfig` exposes no loss history. True of the config, but beside
the point: `BundleAdjustment` records `_last_loss_history` (`list[list[float]]` — per-step LM
losses, one list per increment) and `refine_poses` writes it to
`<backend_dir>/colmap/refine.json` alongside the config and frame count
(`wrapper/reconstructor.py:1216-1229`). The page reads that file. The page also measures
what moved, which stays the honest read: BA's objective is reprojection error, not pose error.

- [ ] **Step 1: Write the notebook**

**md**
```markdown
# Refinement — Bundle Adjustment and Loop Closure

A feedforward pass gives poses in one shot. Two refinements improve them:

- **Bundle adjustment** re-optimises poses and points jointly against reprojection error
  (Levenberg-Marquardt), then rebuilds the point set against the refined poses.
- **Loop closure** splits the sequence into submaps, finds revisits with a DINO-SALAD
  retrieval gate, and applies a pose-graph correction — which is what removes drift.

Both are leaf stages selected by config, and both are feedforward-only: a global SfM solve
has already done this work, so the pipeline refuses the combination. They are also mutually
exclusive with each other — BA needs per-frame model tensors that LC submaps do not carry.
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
# 24 frames, not 16: LoopClosure silently falls back to plain inference when the frame
# count is below submap_size. 24 at submap_size=8 gives three submaps and a section that
# actually demonstrates loop closure.
N_FRAMES = 24
SUBMAP_SIZE = 8      # base.yaml default is 20; evals pass 50 for >100-frame sequences

base = tutorial_scene("refinement-base", VIDEO_PATH, preproc={"max_frames": N_FRAMES})
base.preprocess()
plain = base.build_pointcloud()
print(f"{len(plain.image_paths)} frames, {len(plain.points):,} points")
```

**md** — `## §1 — Bundle adjustment`: `pointcloud.bundle_adjustment: true` makes `refine`
run inline after the pointcloud stage; standalone it is `refine_poses()`. It rewrites the
COLMAP model, `sparse_pc.ply` and `pointcloud.zarr` poses in place.

**code**
```python
ba = tutorial_scene(
    "refinement-ba",
    VIDEO_PATH,
    preproc={"max_frames": N_FRAMES},
    pointcloud={"bundle_adjustment": True},
)
ba.preprocess()
ba.build_pointcloud()
ba_result = ba.refine_poses()
print(f"points: {len(plain.points):,} -> {len(ba_result.points):,}")
```

**md** — `### Convergence`: `refine.json` is the stage marker and its provenance in one
file. `loss_history` is one list per BA increment, each the per-step LM loss.

**code**
```python
import json

import matplotlib.pyplot as plt

refine = json.loads((ba.backend_dir / "colmap" / "refine.json").read_text())
fig, ax = plt.subplots(figsize=(6, 3))
for i, curve in enumerate(refine["loss_history"]):
    ax.plot(curve, marker="o", ms=3, label=f"increment {i}")
ax.set(xlabel="LM step", ylabel="loss", yscale="log", title="Bundle adjustment convergence")
ax.legend(fontsize=8)
plt.show()
print(f"{len(refine['loss_history'])} increment(s) over {refine['n_frames']} frames")
```

**md** — `### What BA moved`: camera centres are the translation column of the inverted
extrinsics, so per-frame displacement is the honest read on a refinement whose objective is
reprojection error, not pose error.

**code**
```python
import numpy as np

from collab_splats.geometry.transforms import invert_poses

shift = np.linalg.norm(
    invert_poses(ba_result.extrinsics)[:, :3, 3] - invert_poses(plain.extrinsics)[:, :3, 3], axis=1
)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(shift, marker="o", ms=3)
ax.set(xlabel="frame", ylabel="camera shift", title="Per-frame displacement from BA")
plt.show()
print(f"median {np.median(shift):.4f}   max {shift.max():.4f}")
```

**md** — `### When BA does nothing`: on a small-baseline clip the reprojection filter can
admit no frames, leaving the poses untouched. That is a real outcome on short handheld
footage, not a failure — a flat line above means the scene did not give BA anything to
solve.

**md** — `## §2 — Loop closure`: `pointcloud.loop_closure` takes the `LoopClosureConfig`
block. Below `submap_size` frames the wrapper silently does plain inference, which is why
this page raises the frame count and lowers the submap size together.

**code**
```python
lc = tutorial_scene(
    "refinement-lc",
    VIDEO_PATH,
    preproc={"max_frames": N_FRAMES},
    pointcloud={"loop_closure": {"submap_size": SUBMAP_SIZE}},
)
lc.preprocess()
lc_result = lc.build_pointcloud()
print(f"{len(lc_result.image_paths)} frames, {len(lc_result.points):,} points")
```

**md** — `### What the correction moved`

**code**
```python
n = min(len(lc_result.extrinsics), len(plain.extrinsics))
lc_shift = np.linalg.norm(
    invert_poses(lc_result.extrinsics)[:n, :3, 3] - invert_poses(plain.extrinsics)[:n, :3, 3], axis=1
)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(lc_shift, marker="o", ms=3, color="tab:orange")
ax.set(xlabel="frame", ylabel="camera shift", title="Per-frame displacement from loop closure")
plt.show()
print(f"median {np.median(lc_shift):.4f}   max {lc_shift.max():.4f}")
```

**md** — `## §3 — Choosing`: BA sharpens a locally-consistent pass and needs baseline; loop
closure fixes global drift and needs an actual revisit to fire. They do not compose — the
config rejects both at once.

- [ ] **Step 2: Run the gate slice**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k refinement -v 2>&1 | tail -15
```

Expected: 4 passed. `imports_resolve` in particular — the deleted
`collab_splats.geometry.loop_closure.closure` module is the break this page fixes.

- [ ] **Step 3: Execute** (three reconstructions — allow ~30 min)

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=5400 \
  docs/source/tutorials/02_pointcloud/refinement.ipynb
```

- [ ] **Step 4: Confirm loop closure actually ran**

The silent fallback is the specific failure this page exists to avoid, so check it.

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
grep -c "submap" docs/source/tutorials/02_pointcloud/refinement.ipynb
/opt/venv/reconstruction/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/refinement.ipynb'))
txt = ''.join(''.join(o.get('text','')) for c in nb['cells'] for o in c.get('outputs',[]))
print([l for l in txt.split(chr(10)) if 'median' in l])
"
```

Expected: the loop-closure `median`/`max` line is non-zero. An all-zero displacement means
LC fell back to plain inference — lower `SUBMAP_SIZE` or raise `N_FRAMES`, re-execute, and
only then commit.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add docs/source/tutorials/02_pointcloud/refinement.ipynb
git commit -m "docs(tutorials): merge BA and loop closure into one refinement page

Replaces the deleted geometry.loop_closure.closure import; both passes now run
the way the pipeline runs them, as config-selected stages. 24 frames at
submap_size=8 so loop closure actually fires — below submap_size the wrapper
silently falls back to plain inference and the section would show nothing.

Plots BA convergence from colmap/refine.json, which refine_poses already
writes with the per-increment LM loss history — no library change needed — and
measures per-frame camera displacement beside it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `03_splats/train_splats.ipynb`

**Files:** Modify `docs/source/tutorials/03_splats/train_splats.ipynb` (full rewrite)

Subject: Scaffold-2DGS, the configuration in production use. `Reconstructor.splats()` does
all the input assembly, so the page configures and calls it — this removes both the missing
`image_ids=` break and the `sys.path.insert` into `evals/`.

- [ ] **Step 1: Write the notebook**

**md**
```markdown
# Gaussian Splats — Scaffold-2DGS

Two independent axes:

- `representation` — `vanilla` (explicit per-Gaussian parameters) or `scaffold` (anchors
  plus an MLP that decodes offsets, color and opacity per view)
- `primitive` — `3dgs` (volumetric ellipsoids, MCMC densification against a fixed budget)
  or `2dgs` (surface-aligned disks, gsplat's DefaultStrategy, a better mesh source)

This page trains `scaffold` × `2dgs`. Depth targets come from `pointcloud.zarr`, masked by
`mesh.conf_percentile`; the trainer takes it from there.
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
N_FRAMES = 16
MAX_STEPS = 1000     # base.yaml runs 30000; renders here are visibly softer

scene = tutorial_scene(
    "splats",
    VIDEO_PATH,
    preproc={"max_frames": N_FRAMES},
    splats={
        "enabled": True,
        "representation": "scaffold",
        "primitive": "2dgs",
        "max_steps": MAX_STEPS,
        # base.yaml's losses block is 3dgs-shaped and always wins over default_losses(),
        # so a 2dgs run needs both MCMC regularizers turned off and distortion turned on
        # - opacity_reg must be 0 under scaffold regardless: opacity is decoded and its
        #   sign is the offset visibility mask, so regularizing it shuts offsets off
        "losses": {
            "opacity_reg": {"weight": 0.0},
            "scale_reg": {"weight": 0.0},
            "distortion": {"weight": 0.01, "start": 300},
        },
    },
)
scene.preprocess()
scene.build_pointcloud()
```

**md** — `## §1 — Train`: `splats()` loads the zarr, orders frames to match
`result.image_paths`, masks depth targets by confidence and calls the trainer. The two rules
it enforces: `sh_degree`/`sh_degree_interval` are vanilla-only (scaffold decodes color from
`mlp_color`, so a deliberate override raises), and `opacity_reg` must be 0 under scaffold.

**code**
```python
ckpt = scene.splats()
print("checkpoint:", ckpt)
```

**md** — `## §2 — Outputs`

**code**
```python
import json

out_dir = ckpt.parent
for name in ("ckpt.pt", "splats.ply", "splats_quality_report.json"):
    p = out_dir / name
    print(f"{name:28s} {p.stat().st_size / 1e6:8.2f} MB" if p.exists() else f"{name:28s} MISSING")

report = json.loads((out_dir / "splats_quality_report.json").read_text())
print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2)[:600])
```

**md** — `## §3 — Render gallery`: `load_checkpoint` returns the model, the pose-opt
`CameraOpt`, and the cameras the run was trained with — so the renders below use the
refined poses, not the originals. `render_views` yields one dict per view, each `rgb` carrying a leading batch axis
of 1 — hence the `[0]`.

**code**
```python
import matplotlib.pyplot as plt
import numpy as np
from collab_splats.preproc import frames
from collab_splats.splats.rendering import load_checkpoint, render_views

model, camera_opt, cam_to_world, intrinsics, image_ids, (H, W) = load_checkpoint(ckpt, device="cuda")
picks = [0, len(image_ids) // 2, len(image_ids) - 1]

renders = list(render_views(model, camera_opt, cam_to_world[picks], intrinsics[picks], H, W))
truth = frames.read_frames(scene.images_dir, [image_ids[i] for i in picks])

fig, axes = plt.subplots(len(picks), 2, figsize=(9, 3.2 * len(picks)))
for row, out in enumerate(renders):
    axes[row, 0].imshow(truth[row])
    axes[row, 0].set_title(f"frame {image_ids[picks[row]]} — ground truth")
    axes[row, 1].imshow(np.clip(out["rgb"][0].detach().cpu().numpy(), 0, 1))
    axes[row, 1].set_title("render")
    for ax in axes[row]:
        ax.axis("off")
plt.tight_layout()
plt.show()
```

**md** — `## §4 — A vanilla run`, **not executed**:
````markdown
The same stage runs classic 3DGS — only the config changes:

```python
splats={
    "enabled": True,
    "representation": "vanilla",
    "primitive": "3dgs",
    "max_steps": MAX_STEPS,
    "sh_degree": 3,             # available here; scaffold rejects it
    "sh_degree_interval": 1000, # SH bands unlock evenly over the run
}
```

What changes: Gaussians carry explicit spherical-harmonic color instead of an MLP decode,
so the model is larger and view-dependent color is baked per primitive; `opacity_reg`
becomes usable and `distortion` does not apply; and `3dgs` densifies with MCMC against
`cap_max` rather than `grow_grad2d`, giving a denser but less surface-aligned cloud —
which is why `2dgs` is the better mesh source.
````

- [ ] **Step 2: Run the gate slice**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k train_splats -v 2>&1 | tail -15
```

Expected: 4 passed — `no_shared_cache` now passes because `sys.path.insert` is gone.

- [ ] **Step 3: Execute**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=5400 \
  docs/source/tutorials/03_splats/train_splats.ipynb
```

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add docs/source/tutorials/03_splats/train_splats.ipynb
git commit -m "docs(tutorials): rewrite splats page on Scaffold-2DGS

Fixes two breaks: train() was called without the now-required image_ids
keyword, and trainer inputs came from evals.scripts.eval_splats behind a
sys.path.insert. The page now configures splats and calls
Reconstructor.splats(), which is the assembly production uses — so the notebook
cannot drift from it.

Unpacks load_checkpoint's 6-tuple and iterates render_views, both of which the
old page had wrong.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: `04_semantics/feature_extraction.ipynb`

**Files:** Modify `docs/source/tutorials/04_semantics/feature_extraction.ipynb` (full rewrite; absorbs the deleted `maskclip_vs_talk2dino.ipynb`)

Subject: dense 2D features over one frame, from two extractors. Cheapest page — one decoded
frame, no reconstruction.

- [ ] **Step 1: Write the notebook**

**md**
```markdown
# Feature Extraction

Both extractors produce a dense per-patch feature map in a space shared with CLIP text
embeddings, so a text query scores every patch directly.

- **MaskCLIP** — CLIP's last attention layer reworked to keep spatial structure.
- **Talk2DINO** — DINOv2/v3 patch features mapped into CLIP's text space.

`dinov2` is registered too, but it has no text space — it is for similarity, not queries.
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
FRAME_IDX = 0        # features are per-frame; one is enough
POSITIVE = ["a building", "a tree"]
NEGATIVE = ["sky", "road", "ground"]
```

**code**
```python
from collab_splats.preproc.video import extract_frame

frame = extract_frame(VIDEO_PATH, FRAME_IDX)
print("frame:", frame.shape, frame.dtype)
```

**md** — `## §1 — MaskCLIP`: `forward` takes a list of images and returns one `(C, pH, pW)`
tensor each; `features_to_rgb` is a PCA projection for looking at them.

**code**
```python
import matplotlib.pyplot as plt
from collab_splats.semantics.features.maskclip import MaskCLIPExtractor

clip = MaskCLIPExtractor()
feat_clip = clip.forward([frame])[0]
print("maskclip features:", tuple(feat_clip.shape))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(frame); axes[0].set_title("frame")
axes[1].imshow(clip.features_to_rgb(feat_clip)); axes[1].set_title("MaskCLIP features (PCA -> RGB)")
for ax in axes: ax.axis("off")
plt.show()
```

**md** — `## §2 — Talk2DINO`

**code**
```python
from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

dino = Talk2DinoExtractor()
feat_dino = dino.forward([frame])[0]
print("talk2dino features:", tuple(feat_dino.shape))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(frame); axes[0].set_title("frame")
axes[1].imshow(dino.features_to_rgb(feat_dino)); axes[1].set_title("Talk2DINO features (PCA -> RGB)")
for ax in axes: ax.axis("off")
plt.show()
```

**md** — `## §3 — Text queries`: `score_queries` is a contrastive softmax of the positives
against the negatives, so the negatives do as much work as the positives. Two behaviours to
know: it **reduces over the positive list** (`reduction="max"`) and returns a single `(H, W)`
map, so a per-prompt map means one call per prompt; and `negative=[]` skips the contrast
entirely and returns raw cosine similarity, which is not comparable between prompts. Left
unset, it defaults to `["object"]`.

**code**
```python
fig, axes = plt.subplots(len(POSITIVE), 3, figsize=(12, 4 * len(POSITIVE)))
axes = axes.reshape(len(POSITIVE), 3)
for r, q in enumerate(POSITIVE):
    sim_clip = clip.score_queries(feat_clip, positive=[q], negative=NEGATIVE)
    sim_dino = dino.score_queries(feat_dino, positive=[q], negative=NEGATIVE)
    axes[r, 0].imshow(frame); axes[r, 0].set_title("frame")
    axes[r, 1].imshow(sim_clip.detach().cpu().numpy(), cmap="viridis"); axes[r, 1].set_title(f"MaskCLIP · {q}")
    axes[r, 2].imshow(sim_dino.detach().cpu().numpy(), cmap="viridis"); axes[r, 2].set_title(f"Talk2DINO · {q}")
    for ax in axes[r]: ax.axis("off")
plt.tight_layout(); plt.show()
```

**md** — `## §4 — Which to use`: MaskCLIP is stronger on stuff-like categories and coarse
regions; Talk2DINO inherits DINO's crisper object boundaries and localises things better.
`semantics.extractor` selects either by name — base.yaml defaults to `talk2dino`.

- [ ] **Step 2: Gate, execute, commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k feature_extraction -v 2>&1 | tail -10

PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  docs/source/tutorials/04_semantics/feature_extraction.ipynb

git add docs/source/tutorials/04_semantics/feature_extraction.ipynb
git commit -m "docs(tutorials): rewrite feature extraction, absorbing the comparison page

One decoded frame, no reconstruction, no shared cache. Folds in the
side-by-side MaskCLIP/Talk2DINO comparison from the deleted
maskclip_vs_talk2dino notebook, which duplicated this page's last section.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Expected: 4 passed, nbconvert exits 0.

---

### Task 10: `04_semantics/segmentation.ipynb`

**Files:** Modify `docs/source/tutorials/04_semantics/segmentation.ipynb` (full rewrite)

Subject: instance masks, and turning masks plus dense features into per-object features.

- [ ] **Step 1: Write the notebook**

**md**
```markdown
# Segmentation

Dense features describe every patch. Masks group patches into objects, and
`aggregate_masked_features` turns the two into one feature vector per object — which is
what makes object-level retrieval possible.

Registered segmenters: `mobilesamv2`, `insid3`, `sam3`, and `skywater` (used by
`mesh.mask_sky` to zero sky depth before fusion).
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
FRAME_IDX = 0
```

**code**
```python
from collab_splats.preproc.video import extract_frame

frame = extract_frame(VIDEO_PATH, FRAME_IDX)
print("frame:", frame.shape)
```

**md** — `## §1 — Masks`: `strategy` picks what the segmenter proposes — `object` gives
instance masks. `segment` returns a `(M, H, W)` mask tensor and the raw backend output, or
`None` when nothing is found.

**code**
```python
from collab_splats.semantics.segmentation import MobileSAMSegmentation

seg = MobileSAMSegmentation(strategy="object", device="cuda")
masks, _ = seg.segment(frame)
print("masks:", tuple(masks.shape))
```

**md** — `### The masks`: `segment` returns the masks already stacked, so they are drawn
directly. (`create_composite_mask` in the same module flattens SAM's *raw result dicts* into
one integer-ID image, and `mask_id_to_binary_mask` reverses that — they are the auto-strategy
path, not this one.)

**code**
```python
import matplotlib.pyplot as plt
import numpy as np

n_show = min(5, len(masks))
fig, axes = plt.subplots(1, n_show + 1, figsize=(3 * (n_show + 1), 3.2))
axes[0].imshow(frame); axes[0].set_title(f"frame — {len(masks)} masks")
for i, ax in enumerate(axes[1:]):
    ax.imshow(frame)
    ax.imshow(masks[i].cpu().numpy(), alpha=0.55, cmap="autumn")
    ax.set_title(f"mask {i}")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
```

**md** — `## §2 — Pooling features over masks`: `aggregate_masked_features` averages the
dense feature map within each mask and writes the result back per pixel, giving another
`(C, H, W)` map — mask-aligned, not per-object vectors. That is what makes the features
snap to object boundaries instead of patch boundaries.

It needs the *spatial* `(C, pH, pW)` map, since a pooled embedding has nowhere to apply a
mask. The two resolution arguments are the grid it pools on and the grid it returns; it
resamples between them.

**code**
```python
import torch
from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
from collab_splats.semantics.segmentation import aggregate_masked_features

clip = MaskCLIPExtractor()
feat = clip.forward([frame])[0]

pooled = aggregate_masked_features(
    feat,
    masks.float(),
    resolution=tuple(feat.shape[-2:]),
    final_resolution=tuple(masks.shape[-2:]),
)
print(f"dense  {tuple(feat.shape)}  ->  mask-pooled {tuple(pooled.shape)}")
```

**md** — `### Before and after`: the same PCA projection applied to both, so the change is
the structure and not the palette. The pooled map is piecewise-constant — one color per
object — where the dense map drifts across each surface.

**code**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].imshow(frame); axes[0].set_title("frame")
axes[1].imshow(clip.features_to_rgb(feat)); axes[1].set_title("dense features")
axes[2].imshow(clip.features_to_rgb(pooled)); axes[2].set_title("pooled over masks")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
```

- [ ] **Step 2: Gate, execute, commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k segmentation -v 2>&1 | tail -10

PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  docs/source/tutorials/04_semantics/segmentation.ipynb

git add docs/source/tutorials/04_semantics/segmentation.ipynb
git commit -m "docs(tutorials): rewrite segmentation on a self-contained frame

Decodes its own frame instead of reading the shared keyframe dir, and unpacks
segment()'s (masks, raw) tuple. Keeps the masks-to-features section, which is
the reason the page exists.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: `04_semantics/lifting_and_query.ipynb`

**Files:** Modify `docs/source/tutorials/04_semantics/lifting_and_query.ipynb` (full rewrite)

Subject: 2D features → 3D points → text query. `extract_semantics()` is the whole chain, so
the page runs it, then opens the artifacts to show what each step produced.

- [ ] **Step 1: Write the notebook**

**md**
```markdown
# Semantic Lifting and Query

Features live on 2D patches; the reconstruction lives in 3D. Lifting projects each point
into the frames that see it and gathers the features there, giving one feature vector per
point — which makes a text query a 3D operation.

Raw features are wide (384-1024D per point), so an autoencoder compresses them to
`semantics.n_components` first. `extract_semantics()` runs all three steps and writes
`semantics/<extractor>.zarr` (the 2D cache, shared by every backend),
`<backend>/<extractor>_lifted.zarr` and `<extractor>_ae.pt`.
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
N_FRAMES = 16
POSITIVE = ["a building"]
NEGATIVE = ["sky", "road", "ground", "tree"]

scene = tutorial_scene(
    "lifting",
    VIDEO_PATH,
    preproc={"max_frames": N_FRAMES},
    semantics={"enabled": True, "extractor": "maskclip", "n_components": 64, "max_epochs": 20},
)
scene.preprocess()
result = scene.build_pointcloud()
print(f"points {result.points.shape}   frames {len(result.image_paths)}")
```

**md** — `## §1 — Extract, lift, compress`

**code**
```python
lifted_dir = scene.extract_semantics(result)
print("lifted:", lifted_dir)
for p in sorted(scene.semantics_cache_dir.iterdir()):
    print("  cache:", p.name)
```

**md** — `## §2 — What lifting produced`: on disk the lift is `(P, latent)` — one
*compressed* vector per point. `load_point_features` is the read side of that pair: it finds
the lifted store, loads the matching autoencoder, decodes back to the extractor's width in
chunks and L2-normalizes each row. Chunking is what keeps a 500k-point scene off the 46.6 GB
cap, and normalizing is what makes the scores below cosine similarities.

`target_cosine` is the fidelity knob on the write side — training stops once the round trip
preserves that much direction, which is the metric that matters because the query compares
directions. It is measured in-sample, so on a scene this small read a pass as "the fit
completed", not "the features are trustworthy".

**code**
```python
from collab_splats.semantics import load_point_features

point_features = load_point_features(lifted_dir)
print(f"decoded per-point features: {point_features.shape}")
```

**md** — `## §3 — Query`: `score_queries` takes the `(P, D)` point array directly and returns
one score per point — the same call the 2D pages made on a `(C, H, W)` map, and the same call
the dashboard makes. Nothing here is point-cloud-specific.

**code**
```python
import torch
from collab_splats.semantics.features.maskclip import MaskCLIPExtractor

clip = MaskCLIPExtractor()
scores = clip.score_queries(torch.from_numpy(point_features), positive=POSITIVE, negative=NEGATIVE)
scores = scores.detach().cpu().numpy()
print(f"scores {scores.shape}   range {scores.min():.3f} .. {scores.max():.3f}   above 0.5: {(scores > 0.5).mean():.1%}")
```

**md** — `## §4 — The scored cloud`: the scores are index-aligned with the **zarr**'s points,
not with `result.points`. `build_pointcloud` returns the COLMAP-tracked sparse set; the lift
runs over `FeedforwardResult.points` out of `pointcloud.zarr`, which is a different and much
larger array. Reload it and colour that.

**code**
```python
import pyvista as pv
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.utils.visualization import pointcloud_to_polydata

ff = FeedforwardResult.load_zarr(scene.pointcloud_zarr, load_depth=False, load_world_points=False)
print(f"lifted points {len(ff.points):,}   scores {scores.shape[0]:,}   "
      f"colmap-tracked points {len(result.points):,}")

pl = pv.Plotter()
pl.add_mesh(
    pointcloud_to_polydata(ff.points, score=scores),
    scalars="score", cmap="viridis", point_size=2, render_points_as_spheres=True,
)
pl.show()
```

- [ ] **Step 2: Gate, execute, commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k lifting_and_query -v 2>&1 | tail -10

PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  docs/source/tutorials/04_semantics/lifting_and_query.ipynb

git add docs/source/tutorials/04_semantics/lifting_and_query.ipynb
git commit -m "docs(tutorials): rewrite semantic lifting as a self-contained page

Moves 05_lifting/semantic_lifting into 04_semantics and deletes the
cache-hit/cache-miss branching. Runs the real chain — extract_semantics — then
opens the artifacts it wrote, rather than re-implementing extract/lift/compress
cell by cell.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: `05_mesh/tsdf_mesh.ipynb`

**Files:** Modify `docs/source/tutorials/05_mesh/tsdf_mesh.ipynb` (full rewrite)

Subject: processed data in, mesh out, from both depth sources. Most expensive page — it
pays a splat train. `mesh.source` selects the depth source, so both are config runs.

- [ ] **Step 1: Write the notebook**

**md**
```markdown
# Meshing — TSDF from Two Depth Sources

TSDF fusion builds a voxel grid of **truncated signed distances**: for each voxel, the
signed distance to the nearest surface along the viewing ray, clamped to ±`sdf_trunc`. Each
view writes its own field into the grid and they are averaged — that averaging is what
cancels per-view depth noise. The mesh is the zero level set.

Two consequences worth holding onto:

- `sdf_trunc` — not `voxel_size` — sets the thinnest structure that survives, because a
  structure thinner than the truncation band has both its surfaces write into the same
  voxels. base.yaml's `sdf_trunc_mult: 4.0` makes the band 8 voxels wide, tuned for noisy
  RGBD; rendered splat depth is far cleaner, so 1.5-2.0 recovers thin geometry at the same
  voxel size.
- Cancellation needs **both** surfaces observed. A fence seen only from the front is
  fattened rather than resolved, then dies in floater cleanup.

`mesh.source` picks the depth: `feedforward` fuses `pointcloud.zarr` depth with COLMAP as
the pose authority; `splats` fuses the checkpoint's own renders, with the poses those
splats were rendered with (pose-opt deltas included) and alpha as confidence.
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
N_FRAMES = 16
MAX_STEPS = 1000     # base.yaml runs 30000
VOXEL_SIZE = 0.02    # base.yaml runs 0.0025; coarser keeps this page quick
DEPTH_TRUNC = 3.0

scene = tutorial_scene(
    "mesh",
    VIDEO_PATH,
    preproc={"max_frames": N_FRAMES},
    mesh={"source": "feedforward", "voxel_size": VOXEL_SIZE, "depth_trunc": DEPTH_TRUNC},
    splats={
        "enabled": True, "representation": "scaffold", "primitive": "2dgs",
        "max_steps": MAX_STEPS,
        "losses": {
            "opacity_reg": {"weight": 0.0},
            "scale_reg": {"weight": 0.0},
            "distortion": {"weight": 0.01, "start": 300},
        },
    },
)
scene.preprocess()
result = scene.build_pointcloud()
```

**md** — `## §1 — Source A: feedforward depth`: `mesh.conf_percentile` drops depth below
that confidence percentile before fusion — on this scene most stray components come from
the tail it removes.

**code**
```python
ff_mesh = scene.mesh(result)
print("feedforward mesh:", ff_mesh)
```

**md** — `## §2 — Source B: splat renders`: the splats stage is never auto-run, so train
first. Then re-fuse with `mesh.source: splats`.

**code**
```python
ckpt = scene.splats()
print("checkpoint:", ckpt)
```

**code**
```python
import shutil

# Keep the feedforward mesh — the splats run writes to the same mesh.ply
ff_kept = ff_mesh.with_name("mesh_feedforward.ply")
shutil.move(ff_mesh, ff_kept)

scene.config["mesh"]["source"] = "splats"
sp_mesh = scene.mesh(result, overwrite=True)
print("splats mesh:", sp_mesh)
```

**md** — `## §3 — Compare`: largest-component fraction is the useful number — a mesh in one
piece is a mesh you can texture.

**code**
```python
import numpy as np
import open3d as o3d

for path, label in [(ff_kept, "feedforward"), (sp_mesh, "splats")]:
    m = o3d.io.read_triangle_mesh(str(path))
    _, counts, _ = m.cluster_connected_triangles()
    counts = np.asarray(counts)
    print(f"{label:12s} verts={len(m.vertices):>8,}  tris={len(m.triangles):>8,}  "
          f"components={len(counts):>5,}  largest={counts.max() / counts.sum():.1%}")
```

**md** — `### Side by side`

**code**
```python
import pyvista as pv

pl = pv.Plotter(shape=(1, 2))
for col, (path, title) in enumerate([(ff_kept, "feedforward depth"), (sp_mesh, "splat renders")]):
    pl.subplot(0, col)
    pl.add_mesh(pv.read(path), color="lightgray")
    pl.add_title(title)
pl.link_views()
pl.show()
```

**md** — `## §4 — What the stage already did`, prose only, **no cell**. Both meshes above
are post-cleanup: the stage calls `clean_repair_mesh` on the fused PLY before returning it
(`reconstructor.py:713`), dropping floaters and filling small holes. Its thresholds are
**scale-relative** fractions of the mesh's own extent rather than absolute distances, which
is why the same numbers transfer between scenes. Re-running it here at the same defaults
would rewrite the files in place and change nothing, so the page states the behaviour and
leaves the artifacts alone.

- [ ] **Step 2: Gate, execute, commit** (longest page, ~45 min)

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k tsdf_mesh -v 2>&1 | tail -10

PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  docs/source/tutorials/05_mesh/tsdf_mesh.ipynb

git add docs/source/tutorials/05_mesh/tsdf_mesh.ipynb
git commit -m "docs(tutorials): rewrite meshing to compare both TSDF depth sources

Moves 06_mesh/splats_mesh to 05_mesh and drops the assert that the splats
notebook had been run — the page trains its own. Both fusions go through
Reconstructor.mesh via mesh.source, so the notebook cannot drift from the
stage. Explains what TSDF does with the distance field, and why sdf_trunc
rather than voxel_size sets the thin-structure floor.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: `06_localization/localization.ipynb`

**Files:** Modify `docs/source/tutorials/06_localization/localization.ipynb` (full rewrite)

Subject: one query image → a pose in a known map.

- [ ] **Step 1: Write the notebook**

**md**
```markdown
# Visual Localization

Given a reconstruction and a photo that was not part of it, recover the photo's pose.

Three stages: **retrieval** narrows the reference frames to a handful (`localization.top_k`),
**local matching** finds pixel correspondences against those (`localization.matcher`, over
the vismatch zoo), and **PnP with RANSAC** solves for the pose using the reference frames'
known 3D points.

Feature extraction dominates the per-pair cost, so `build_localization_db()` extracts once
and caches into `pointcloud.zarr` — every later query reuses it.
```

**code**
```python
# ── Budget ────────────────────────────────────────────────────────────────────
N_FRAMES = 16

scene = tutorial_scene(
    "localization",
    VIDEO_PATH,
    preproc={"max_frames": N_FRAMES},
    localization={"enabled": True, "matcher": "loma", "top_k": 8},
)
scene.preprocess()
result = scene.build_pointcloud()
scene.build_localization_db()
print("index:", scene.pointcloud_zarr)
```

**md** — `## §1 — Load the index`: `from_feedforward` reads the cached features back, so
this costs nothing beyond IO.

**code**
```python
from collab_splats.localization.localizer import CameraLocalizer
from collab_splats.preproc import frames
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

ff = FeedforwardResult.load_zarr(scene.pointcloud_zarr)
images = frames.read_frames(scene.images_dir)
loc = CameraLocalizer.from_feedforward(
    ff, images=images, ids=[p.stem for p in ff.image_paths], zarr_path=scene.pointcloud_zarr
)
print(f"{len(loc.image_paths)} reference frames")
```

**md** — `## §2 — The query`: `QUERY_IMAGE` comes from a **different** video. Localizing a
frame that was in the reconstruction would prove nothing.

**code**
```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

query = np.asarray(Image.open(QUERY_IMAGE).convert("RGB"))
plt.figure(figsize=(6, 4)); plt.imshow(query); plt.axis("off")
plt.title("query (different video)"); plt.show()
```

**md** — `## §3 — Localize`: with no `query_intrinsics`, K is seeded from the image
proportions — which is why an uncalibrated phone photo still localizes.

**code**
```python
res = loc.localize(query)
print(f"correspondences {res.n_correspondences}   inliers {res.n_inliers}")
print(f"ref frames by inlier count  {res.ranked_ref_frames[:8]}")
print("pose:\n", np.round(res.pose, 4))
```

**md** — `## §4 — Correspondences`: the inlier mask is RANSAC's verdict — the outliers are
matches the geometry rejected. `ref_frame_indices` is per-*correspondence*, so the frame to
draw comes from `ranked_ref_frames`, which counts inliers per frame and drops the ones that
contributed none.

**code**
```python
from collab_splats.localization.viz import correspondences_for_ref, plot_correspondences

ref_idx = res.ranked_ref_frames[0]
ref_image = images[ref_idx]

# ref_px comes back in loc.ref_hw space; ref_image_hw rescales it onto the image being drawn
query_px, ref_px, inliers = correspondences_for_ref(res, ref_idx, ref_image_hw=ref_image.shape[:2])
plot_correspondences(query, ref_image, query_px, ref_px, inlier_mask=inliers)
```

Skipping `ref_image_hw` is the quiet failure mode here: the localizer indexes at its own
resolution, so the lines land on the right image at the wrong coordinates and the plot looks
plausibly wrong rather than obviously broken.

**md** — `## §5 — The pose in the scene`: reference cameras in blue, the localized query in
red. Both frustums are drawn from world-to-camera matrices — `result.extrinsics` and
`res.pose` are already in that convention, which is what
`create_camera_frustum_pyvista` takes, so neither is inverted.

**code**
```python
import pyvista as pv
from collab_splats.utils.visualization import create_camera_frustum_pyvista, pointcloud_to_polydata

pl = pv.Plotter()
pl.add_mesh(
    pointcloud_to_polydata(result.points, RGB=result.colors),
    scalars="RGB", rgb=True, point_size=2, render_points_as_spheres=True,
)
for pose in result.extrinsics:
    pl.add_mesh(create_camera_frustum_pyvista(pose, scale=0.04), color="lightblue")
pl.add_mesh(create_camera_frustum_pyvista(res.pose, scale=0.08), color="red")
pl.show()
```

**md** — `## §6 — Notes`: other matchers are selected by `localization.matcher`; some are
blocked for license or dependency reasons (see `_VISMATCH_*_BLOCKLIST` in
`localization/extractors.py`). `add_localized_frame` folds a solved query back into the
index, so a localized photo can serve as a reference for the next one.

- [ ] **Step 2: Gate, execute, commit**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/test_tutorial_contract.py \
  -k localization -v 2>&1 | tail -10

PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  docs/source/tutorials/06_localization/localization.ipynb

git add docs/source/tutorials/06_localization/localization.ipynb
git commit -m "docs(tutorials): rewrite localization as a self-contained page

Builds its own reconstruction and localization index rather than reading the
shared zarr cache. Uses LocalizationResult's own fields and
correspondences_for_ref instead of the removed ranked_ref_frames.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: Full green gate and a clean re-execution

Each page was executed in its own task, but never all nine against the final helpers. A late
change to `notebook_utils.py` silently invalidates an early page's committed outputs, so the
set is re-executed once at the end.

**Files:** Modify all nine `.ipynb` (outputs only)

- [ ] **Step 1: Run the whole gate**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/docs/ -v 2>&1 | tail -50
```

Expected: 1 set test + 9×4 contract cases + 5 notebook_utils cases + 2 tutorial_config
cases = **44 passed, 0 failed, 0 skipped**. A skip means `third_party/` is missing — fix it and re-run rather than
accepting the count.

- [ ] **Step 2: Re-execute all nine, serially**

One at a time — two concurrent feedforward runs OOM against the 46.6 GB cap.

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
for nb in \
  01_preprocessing/keyframe_extraction \
  02_pointcloud/reconstruction \
  02_pointcloud/refinement \
  03_splats/train_splats \
  04_semantics/feature_extraction \
  04_semantics/segmentation \
  04_semantics/lifting_and_query \
  05_mesh/tsdf_mesh \
  06_localization/localization ; do
  echo "=== $nb"
  PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework PYVISTA_OFF_SCREEN=1 \
  /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 \
    "docs/source/tutorials/$nb.ipynb" || echo "FAILED: $nb"
done
```

Expected: nine `===` lines, zero `FAILED:`. Budget several hours.

- [ ] **Step 3: Confirm outputs landed and nothing was written into the repo**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
/opt/venv/reconstruction/bin/python -c "
import json, pathlib
for p in sorted(pathlib.Path('docs/source/tutorials').rglob('*.ipynb')):
    nb = json.load(open(p))
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
    n = sum(1 for c in code if c['outputs'])
    err = sum(1 for c in code for o in c['outputs'] if o.get('output_type') == 'error')
    print(f'{str(p.relative_to(\"docs/source/tutorials\")):50s} {n:3d}/{len(code):3d} with output  errors={err}')
"
git status --porcelain | grep -v '^ M docs/source/tutorials' || echo "nothing written outside the tutorials"
```

Expected: every page has most cells carrying output, `errors=0`, and nothing modified
outside `docs/source/tutorials`.

- [ ] **Step 4: Check the committed size**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
du -sh docs/source/tutorials
```

Expected: at or below the 32 MB the old 14-notebook set carried. Materially larger usually
means an interactive PyVista scene got serialised into a cell — confirm
`PYVISTA_OFF_SCREEN=1` was set and re-execute that page.

- [ ] **Step 5: Commit the outputs**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add docs/source/tutorials
git commit -m "docs(tutorials): execute all nine notebooks and commit outputs

nbsphinx_execute is 'never', so committed outputs are what the docs site
renders. Re-executed as a set against the final helpers, serially — two
concurrent feedforward runs OOM against the 46.6 GB cap.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 15: Docs build and final verification

- [ ] **Step 1: Build the Sphinx site**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m sphinx -b html docs/source /tmp/tutorial-docs-build 2>&1 | tail -30
```

Expected: `build succeeded`. Toctree-reference warnings are failures here — they mean a
renamed page is still referenced somewhere.

- [ ] **Step 2: Confirm all nine rendered**

```bash
ls /tmp/tutorial-docs-build/tutorials/*/*.html | sort
```

Expected: nine HTML files.

- [ ] **Step 3: Grep for surviving references to deleted pages**

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
grep -rn "feedforward_methods\|slam_loop_closure\|colmap_sfm\|feedforward_mesh\|maskclip_vs_talk2dino\|semantic_lifting\|splats_mesh\|ground_truth_evals\|05_lifting\|06_mesh/\|07_localization" \
  docs/ --include='*.rst' --include='*.py' --include='*.md' --include='*.ipynb' \
  | grep -v 'docs/superpowers/' || echo "no stale references"
```

Expected: `no stale references`. Hits under `docs/superpowers/` are the spec and this plan
describing the deletions — correct, do not edit them away.

- [ ] **Step 4: Run the repository suite for regressions**

The rework touches `docs/` and adds `tests/docs/`, so nothing else should move.

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework && \
PYTHONPATH=/workspace/collab-splats/.worktrees/tutorial-rework \
/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -20
```

Expected: the inherited known-failure count from `docs/known-test-failures.md` (19 on
`clean/final`: 17 missing `vismatch`, 2 scene-id), plus the 44 new passes. Compare against
that document — do not expect zero failures.

- [ ] **Step 5: Update CLAUDE.md and commit**

Change the `tutorial-rework` line in **In-Flight Work** to add the plan link:

```markdown
- **tutorial-rework** — rebuild the tutorial as nine self-contained notebooks on the clean/final API: no shared `data/outputs/` cache, each page builds its inputs into its own tempdir ([spec](docs/superpowers/specs/2026-09-09-tutorial-rework-design.md) · [plan](docs/superpowers/plans/2026-09-09-tutorial-rework.md))
```

```bash
cd /workspace/collab-splats/.worktrees/tutorial-rework
git add CLAUDE.md
git commit -m "docs: link the tutorial-rework plan from CLAUDE.md

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Definition of done

- [ ] Exactly nine notebooks under `docs/source/tutorials`, matching the toctree
- [ ] `tests/docs/` green: 44 passed, 0 failed, 0 skipped
- [ ] No notebook defines a function or class — the gate enforces it
- [ ] `notebook_utils.py` is two helpers and a backend switch; no pipeline glue was added
- [ ] Every notebook executed with outputs committed, `errors=0`
- [ ] Sphinx builds and renders all nine pages
- [ ] No reference outside `docs/superpowers/` to a deleted notebook or to `data/outputs`
- [ ] Repository suite shows no regressions beyond the inherited known failures
