# Tutorial Rework (shared scene, ten pages) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase `clean/tutorials` onto `clean/final`, cut the tutorial to ten readable pages that share one scene, execute them all, and land the result on `clean/final`.

**Architecture:** `tutorial.py` owns one gitignored scene dir (`data/tutorial_scene/`) and one config. `tutorial_scene(*stages)` runs only the named stages that are not on disk yet. `work_dir(page)` gives a page a scratch dir for longhand experiments. Splats are a terminal stage, named only by `train_splats`. The contract gate (`tests/docs/test_tutorial_contract.py`) enforces the page set, the isolation tokens and the readability caps.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), nbformat, nbconvert, pytest, Sphinx + nbsphinx (`nbsphinx_execute = "never"`), pyvista (static backend when headless).

**Spec:** `docs/superpowers/specs/2026-09-09-tutorial-rework-design.md` (revision 2026-10-08).

---

## Conventions (read before any task)

- **Worktree:** `W=/workspace/collab-splats/.worktrees/tutorial-rework`. Run every command from `$W` with `PYTHONPATH=$W`. Check before each session:
  ```bash
  cd $W && PYTHONPATH=$W /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
  ```
  Expected: a path under `$W/collab_splats/`. Anything else means the main checkout is being tested; stop and fix.
- **`third_party`:** the worktree needs `ln -s /workspace/collab-splats/third_party/* $W/third_party/` for any missing entry, or model tests skip.
- **Python:** `PY=/opt/venv/reconstruction/bin/python`. Never bare `python`.
- **Commits:** the git index is shared with other sessions. Commit with `git commit --only <paths>`. Never `git add -A`, never stash, never amend a commit you did not make in this task. Files under `docs/superpowers/` and `data/` are gitignored but tracked: `git add -f <path>` first.
- **Serial GPU:** one executing notebook at a time; nothing heavy alongside (cgroup cap 46.6 GB).
- **Execute a page** (`<page>` relative to `docs/source/tutorials/`):
  ```bash
  cd $W/docs/source/tutorials/$(dirname <page>) && HF_HUB_OFFLINE=1 PYVISTA_OFF_SCREEN=1 PYTHONPATH=$W \
    xvfb-run -a -s "-screen 0 1280x1024x24" \
    /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 $(basename <page>)
  ```
  Run with the Bash tool's `run_in_background: true`; wait for the notification. Never pipe into `| tail` (hides the exit code). Exit status must be 0. `xvfb-run` is required: without it a headless VTK render kills the kernel.
- **Page contract check** (after executing a page):
  ```bash
  cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -k "<stem>" -p no:randomly -q
  ```
  Expected: all selected tests pass.

## Page rules (apply on every page task)

Shape:

1. Markdown cell: `# Title`, then one paragraph: what goes in, what comes out.
2. One setup code cell: imports (stdlib / third-party / ours, blank line between groups), `%run ../tutorial.py`, then `scene = tutorial_scene(...)` with the stages listed in the task. No other config.
3. Numbered sections: `## 1. Title`, `## 2. Title`, ...
4. Last section `## In a pipeline run`: the config keys that do the same thing as a stage, as a short YAML block.

Prose:

- Plain statements. No slogans ("X is the contract"), no aphorisms, no development history, no "Package gap", no `§`.
- At most 2 `**bold**` phrases and at most 3 em-dashes (`—`) per page, counting code comments.
- One sentence after each figure saying what to look for.

Code:

- Lines ≤ 100 characters. Long imports parenthesized.
- One call per line; never `f(g(x))`. Example: `depths = np.array(values)` then `median = np.nanmedian(depths)`, not `round(float(np.nanmedian(np.array(values))))`.
- Block comments are one plain line.
- No `%load_ext autoreload`, no `mkdtemp`, no `overwrite=True`, no function or class definitions.
- Delete a cell unless it teaches the page's subject: no decorative prints, no plot that repeats another.

---

## Task 0: Preflight, backup, rebase

**Files:** none changed except via rebase.

- [ ] **Step 1: Venv preflight**

```bash
cd $W && PYTHONPATH=$W $PY - <<'EOF'
import gsplat, torch
print("gsplat", gsplat.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF
ls ~/.cache/huggingface/hub | grep -iE "llava|loma"
```

Expected: `gsplat 1.5.3` (if `1.4.0`, repair with the gsplat recipe in `setup.sh`), `cuda True`, both LLaVA-1.6 and LoMa directories listed. If a model cache is missing, stop and report it.

- [ ] **Step 2: Worktree clean, backup ref**

```bash
cd $W && git status --short && git update-ref refs/backup/tutorial-release/pre-rebase HEAD && git rev-parse refs/backup/tutorial-release/pre-rebase
```

Expected: empty status, then the HEAD sha (`1b8aacec` or later).

- [ ] **Step 3: Rebase**

```bash
cd $W && git rebase clean/final
```

Expected: success (the trial merge was clean). On a conflict: stop, report the file list, and do not resolve by picking a side blind.

- [ ] **Step 4: Confirm the two broken pages are the only import failures**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -k imports_resolve -p no:randomly -q
```

Expected: fails for `lifting_and_query` and `ocr_lens` only (`extract_feature_cache`). Anything else failing: note it; the page task for that page must fix it.

- [ ] **Step 5: Remove the trial scratch worktree**

```bash
cd $W && git worktree remove --force /tmp/claude-0/-workspace-collab-splats/e566daea-b71e-409d-83bc-0323a030bbb2/scratchpad/trial && git worktree prune
```

---

## Task 1: Shared scene helper

**Files:**
- Modify: `docs/source/tutorials/tutorial.py`
- Test: `tests/docs/test_tutorial_helpers.py`

- [ ] **Step 1: Replace the tempdir tests with shared-scene tests**

Replace everything below `test_backend_static_when_headless` in `tests/docs/test_tutorial_helpers.py`, update the module docstring, and drop `"work_dir"` from the banned-name list in `test_exposes_no_shared_cache_names`:

```python
"""
Unit tests for docs/source/tutorials/tutorial.py.

- inputs: repo-relative committed paths, and none of the retired shared-cache names
- backend: pyvista backend set at load
- tutorial_scene: one shared scene dir; runs only the named stages not yet on disk
- work_dir: a page's scratch dir under the scene, emptied on each call
"""
```

```python
def test_paths_are_repo_relative(ns):
    assert ns["REPO_ROOT"] == REPO
    assert ns["VIDEO_PATH"] == REPO / "data/tutorial/tutorial_example-video.mp4"
    assert ns["QUERY_IMAGE"] == REPO / "data/tutorial/tutorial_example-frame.jpg"
    assert ns["REF_FRAME"] == REPO / "data/tutorial/tutorial_ref-frame.jpg"
    assert ns["SCENE_DIR"] == REPO / "data/tutorial_scene"


@pytest.fixture
def scene_ns(ns, monkeypatch, tmp_path):
    """
    tutorial.py namespace with SCENE_DIR pointed at tmp_path and Reconstructor.run recorded.
    """
    monkeypatch.setitem(ns["tutorial_scene"].__globals__, "SCENE_DIR", tmp_path)
    runs = []
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages=None, overwrite=False: runs.append(stages))
    ns["_runs"] = runs
    return ns


def test_tutorial_scene_uses_the_shared_dir(scene_ns, tmp_path):
    scene = scene_ns["tutorial_scene"]()
    assert Path(scene.config["output_path"]) == tmp_path
    assert Path(scene.config["input_path"]) == scene_ns["VIDEO_PATH"]
    assert scene.config["pointcloud"]["backend"] == "vggt_omega"
    assert scene.config["mesh"]["source"] == "feedforward"
    assert scene.config["preproc"]["max_frames"] == 96


def test_tutorial_scene_runs_only_missing_stages(scene_ns, monkeypatch):
    monkeypatch.setattr(Reconstructor, "done", lambda self, stage: stage == "preproc")
    scene_ns["tutorial_scene"]("preproc", "pointcloud", "mesh")
    assert scene_ns["_runs"] == [["pointcloud", "mesh"]]


def test_tutorial_scene_skips_run_when_all_done(scene_ns, monkeypatch):
    monkeypatch.setattr(Reconstructor, "done", lambda self, stage: True)
    scene_ns["tutorial_scene"]("preproc", "pointcloud")
    assert scene_ns["_runs"] == []


def test_tutorial_scene_extractor_selects_the_store(scene_ns):
    default = scene_ns["tutorial_scene"]()
    ocr = scene_ns["tutorial_scene"](extractor="ocr_lens")
    assert default.outputs["semantics"].name == "maskclip_lifted.zarr"
    assert ocr.outputs["semantics"].name == "ocr_lens_lifted.zarr"
    assert ocr.config["semantics"]["max_epochs"] == 20


def test_tutorial_scene_does_not_mutate_scene_config(scene_ns):
    scene_ns["tutorial_scene"](extractor="ocr_lens")
    assert scene_ns["SCENE_CONFIG"]["semantics"]["extractor"] == "maskclip"


def test_work_dir_is_emptied_on_each_call(scene_ns, tmp_path):
    work_dir = scene_ns["work_dir"]
    first = work_dir("refinement")
    (first / "stale.txt").write_text("x")
    second = work_dir("refinement")
    assert second == tmp_path / "work" / "refinement"
    assert list(second.iterdir()) == []
```

Add `from collab_splats.reconstructor import Reconstructor` to the imports (ours group, blank line after `pyvista`).

- [ ] **Step 2: Run, expect failure**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_helpers.py -p no:randomly -q
```

Expected: FAIL (`KeyError: 'REF_FRAME'`, `tutorial_scene()` missing `name`, no `work_dir`).

- [ ] **Step 3: Rewrite `tutorial.py`**

```python
"""
Committed inputs and the shared scene every tutorial page builds on; load with `%run ../tutorial.py`.

- one scene dir for all pages: a page runs only the stages it needs that are not on disk yet
- delete data/tutorial_scene/ after pulling code changes; done() checks existence only
- pyvista backend set at load: static when headless (nbconvert), interactive trame otherwise
"""

import copy
import os
import shutil
from pathlib import Path

import pyvista as pv

from collab_splats.reconstructor import Reconstructor

########################################
# Committed inputs (read-only)
########################################

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO_PATH = REPO_ROOT / "data/tutorial/tutorial_example-video.mp4"
QUERY_IMAGE = REPO_ROOT / "data/tutorial/tutorial_example-frame.jpg"
REF_FRAME = REPO_ROOT / "data/tutorial/tutorial_ref-frame.jpg"

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"missing {VIDEO_PATH}; see data/tutorial/README.md")

# Static renders under nbconvert, interactive in a live kernel
pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

########################################
# Shared scene
########################################

SCENE_DIR = REPO_ROOT / "data/tutorial_scene"

# Small profile merged over configs/base.yaml (vggt_omega, feedforward mesh)
SCENE_CONFIG = {
    "preproc": {"max_frames": 96},
    "mesh": {"texture": True},
    "splats": {"representation": "scaffold", "primitive": "2dgs", "max_steps": 1000},
    "semantics": {"extractor": "maskclip", "max_epochs": 20},
}


def tutorial_scene(*stages: str, extractor: str | None = None) -> Reconstructor:
    """
    The shared tutorial scene, with the named stages built if they are not on disk yet.

    - stages already done are dropped before run, so a named leaf never hits the overwrite refusal
    - dependencies already on disk are reused by Reconstructor.run

    Args:
        stages: stage names this page needs, e.g. "preproc", "pointcloud", "mesh".
        extractor: semantics extractor; each one writes its own store, so pages never collide.

    Returns:
        A Reconstructor over SCENE_DIR.
    """
    config = copy.deepcopy(SCENE_CONFIG)
    config["input_path"] = str(VIDEO_PATH)
    config["output_path"] = str(SCENE_DIR)

    if extractor is not None:
        config["semantics"]["extractor"] = extractor

    scene = Reconstructor(config)
    missing = [s for s in stages if not scene.done(s)]

    if missing:
        scene.run(stages=missing)

    return scene


def work_dir(page: str) -> Path:
    """
    An empty scratch dir for one page's longhand experiments, under the shared scene.

    Args:
        page: short page label, used as the directory name.

    Returns:
        SCENE_DIR/work/<page>, emptied and recreated.
    """
    path = SCENE_DIR / "work" / page
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path
```

- [ ] **Step 4: Run, expect pass**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_helpers.py -p no:randomly -q
```

Expected: all pass (`REF_FRAME` is checked as a path only; the file lands in Task 3). If `outputs["semantics"]` is not keyed that way on the rebased branch, read `Reconstructor.outputs` and fix the test, not the reconstructor.

- [ ] **Step 5: Commit**

```bash
cd $W && git commit --only docs/source/tutorials/tutorial.py tests/docs/test_tutorial_helpers.py \
  -m "docs(tutorials): shared tutorial scene and per-page work dir

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 2: Contract gate for ten pages and readability

**Files:**
- Modify: `tests/docs/test_tutorial_contract.py`

The gate goes red on the old pages now; each page task turns its own rows green.

- [ ] **Step 1: Update the module docstring, tokens, page list**

Docstring bullets: replace `- isolation: ...` with `- shared scene: pages reach the scene only through tutorial_scene`, add `- readability: prose and line-length caps`.

Add to `BANNED_TOKENS`:

```python
    "mkdtemp": "pages share data/tutorial_scene through tutorial_scene()",
    "overwrite=True": "would rewrite a stage other pages read",
    "autoreload": "development magic, not tutorial content",
```

Replace `PAGES` and rename the set test:

```python
PAGES = [
    "01_preprocessing/preprocessing.ipynb",
    "02_pointcloud/reconstruction.ipynb",
    "02_pointcloud/refinement.ipynb",
    "03_splats/train_splats.ipynb",
    "04_mesh/mesh.ipynb",
    "05_semantics/feature_extraction.ipynb",
    "05_semantics/lifting_and_query.ipynb",
    "05_semantics/ocr_lens.ipynb",
    "05_semantics/segmentation.ipynb",
    "06_localization/localization.ipynb",
]


def test_notebook_set_is_the_ten_pages():
    """
    The set itself is part of the contract; a stray notebook is an ungated page.
    """
    assert sorted(_rel(p) for p in NOTEBOOKS) == PAGES
```

- [ ] **Step 2: Add the readability tests**

Below the existing constants:

```python
# Prose markers of plan or draft text
BANNED_PROSE = {"§": "use '## 1. Title' headings", "Package gap": "plan language, not tutorial text"}

EM_DASH_CAP = 3
BOLD_CAP = 2
LINE_CAP = 100
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
```

Helper beside `_code_sources`:

```python
def _markdown_sources(nb_path: Path) -> list[str]:
    """
    Source text of every markdown cell in a notebook.
    """
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]
```

Tests:

```python
@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_draft_prose(nb: Path):
    """
    Section marks and plan language do not belong in a tutorial.
    """
    text = "\n".join(_markdown_sources(nb) + _code_sources(nb))
    hits = [f"{tok} ({why})" for tok, why in BANNED_PROSE.items() if tok in text]
    assert not hits, f"{nb.name}: {'; '.join(hits)}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_em_dash_and_bold_caps(nb: Path):
    """
    Em-dashes and bold phrases are capped per page, prose and code comments both.
    """
    text = "\n".join(_markdown_sources(nb) + _code_sources(nb))
    dashes = text.count("—")
    bold = len(BOLD.findall("\n".join(_markdown_sources(nb))))
    assert dashes <= EM_DASH_CAP, f"{nb.name} has {dashes} em-dashes (cap {EM_DASH_CAP})"
    assert bold <= BOLD_CAP, f"{nb.name} has {bold} bold phrases (cap {BOLD_CAP})"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_code_line_length(nb: Path):
    """
    Code lines stay readable in the rendered docs.
    """
    lines = "\n".join(_code_sources(nb)).split("\n")
    long = [ln for ln in lines if len(ln) > LINE_CAP]
    assert not long, f"{nb.name} has {len(long)} code lines over {LINE_CAP} chars: {long[0]!r}"
```

- [ ] **Step 3: Run; expect the old pages to fail**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -p no:randomly -q
```

Expected: failures on the page-set test and on readability rows of most old pages. The test file itself must import and collect without error.

- [ ] **Step 4: Commit**

```bash
cd $W && git commit --only tests/docs/test_tutorial_contract.py \
  -m "test(docs): tutorial gate for ten shared-scene pages and readability caps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 3: Commit the reference frame

**Files:**
- Create: `data/tutorial/tutorial_ref-frame.jpg`
- Modify: `data/tutorial/README.md`

- [ ] **Step 1: Copy, verify, add**

```bash
cd $W && SRC=/workspace/collab-splats/docs/source/tutorials/07_localization/ref_image.jpg && \
  cp "$SRC" data/tutorial/tutorial_ref-frame.jpg && sha256sum "$SRC" data/tutorial/tutorial_ref-frame.jpg
```

Expected: identical hashes.

- [ ] **Step 2: Document it**

Add to `data/tutorial/README.md`, in the file list:

```markdown
- `tutorial_ref-frame.jpg`: frame 0 of `tutorial_example-video.mp4` (1080×1920), the in-video localization query; its pose is known from the reconstruction.
```

- [ ] **Step 3: Commit (data/ is ignored; force-add like the existing assets)**

```bash
cd $W && git add -f data/tutorial/tutorial_ref-frame.jpg data/tutorial/README.md && \
  git commit --only data/tutorial/tutorial_ref-frame.jpg data/tutorial/README.md \
  -m "docs(tutorials): commit frame 0 as the in-video localization query

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Remove the stray untracked copy in the main checkout**

The user approved this delete in the design review. Confirm it is untracked, then remove:

```bash
cd /workspace/collab-splats && git status --short docs/source/tutorials/07_localization/ && rm -r docs/source/tutorials/07_localization
```

Expected: one `??` line before the delete.

---

## Task 4: Retire the quickstart page; index and README

**Files:**
- Delete: `docs/source/tutorials/00_quickstart/`
- Modify: `docs/source/tutorials/index.rst`, `README.md` ("Getting Started" section only)

- [ ] **Step 1: Delete the quickstart**

```bash
cd $W && git rm -r -q docs/source/tutorials/00_quickstart
```

- [ ] **Step 2: Rewrite the intro and toctrees of `index.rst`**

Intro (replace everything above the first toctree):

```rst
Tutorials
=========

The pages share one scene, built from ``data/tutorial/tutorial_example-video.mp4`` into
``data/tutorial_scene/``. Each page runs top to bottom on its own: it builds the pipeline stages
it needs that are not on disk yet, then shows its subject. Opened in order, later pages reuse what
earlier pages built.

After pulling code changes, delete ``data/tutorial_scene/`` so the stages are rebuilt.

The scene uses a small profile (96 frames, short training); production values are in
``configs/base.yaml``.

=============================  ============================================
Stage                          Page
=============================  ============================================
``preproc``                    01 · Preprocessing
``pointcloud``                 02 · Reconstruction
``refine``                     02 · Refinement
``reconstruction_quality_report``  02 · Reconstruction
``splats``                     03 · Train splats
``mesh``                       04 · Mesh
``semantics``                  05 · Lifting and query, OCR lens
``localize``                   06 · Localization
=============================  ============================================
```

Toctrees: drop `00 · Quickstart`; `01` lists `01_preprocessing/preprocessing`; `02` lists `reconstruction`, `refinement`; `04` lists `04_mesh/mesh`; others unchanged. Remove the "Pages marked *(pending)*" paragraph.

- [ ] **Step 3: Rewrite README "Getting Started"**

The main checkout has an uncommitted user edit at the top of `README.md` (the module list). Edit only the `## Getting Started` section on the branch; never touch that block. Replace the section body (down to `## Dashboard`) with:

````markdown
## Getting Started

```python
from collab_splats.reconstructor import Reconstructor

scene = Reconstructor({"input_path": "video.mp4", "output_path": "out/"})
scene.run()
print(scene.outputs)
```

Same run from the shell: `python -m collab_splats local --input video.mp4 --output out/`.
Browse the result: `python -m collab_splats.viewer out/<backend>`.

Tutorials in `docs/source/tutorials/` share one scene and build on each other:

| Stage | Tutorial |
|-------|----------|
| preproc | [Preprocessing](docs/source/tutorials/01_preprocessing/preprocessing.ipynb) |
| pointcloud, quality report | [Reconstruction](docs/source/tutorials/02_pointcloud/reconstruction.ipynb) |
| refine | [Refinement](docs/source/tutorials/02_pointcloud/refinement.ipynb) |
| splats | [Train splats](docs/source/tutorials/03_splats/train_splats.ipynb) |
| mesh | [Mesh](docs/source/tutorials/04_mesh/mesh.ipynb) |
| semantics | [Features](docs/source/tutorials/05_semantics/feature_extraction.ipynb) · [Segmentation](docs/source/tutorials/05_semantics/segmentation.ipynb) · [Lifting and query](docs/source/tutorials/05_semantics/lifting_and_query.ipynb) · [OCR lens](docs/source/tutorials/05_semantics/ocr_lens.ipynb) |
| localize | [Localization](docs/source/tutorials/06_localization/localization.ipynb) |
````

Before writing it, verify the CLI flags: `PYTHONPATH=$W $PY -m collab_splats local --help`. Use the real flag names.

- [ ] **Step 4: Commit**

```bash
cd $W && git commit --only docs/source/tutorials/00_quickstart docs/source/tutorials/index.rst README.md \
  -m "docs(tutorials): quickstart moves to README; index describes the shared scene

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Page tasks (5–14)

Execute pages in this order; each builds the scene a little further. Every page task has the same closing steps:

- **Execute** with the command in Conventions.
- **Read the outputs** in the executed notebook: every number and figure the prose refers to must match. Fix prose that disagrees with outputs.
- **Contract check** with `-k <stem>`. All rows pass.
- **Commit** `git commit --only <notebook paths, including deleted ones>` with `docs(tutorials): <page> on the shared scene`.

Edit notebooks with an nbformat script in the scratchpad (load, replace `cells`, write) or with NotebookEdit. Keep the existing kernelspec metadata.

## Task 5: `01_preprocessing/preprocessing.ipynb`

**Files:** Create from `video_quality.ipynb` (`git mv`), fold in `keyframe_extraction.ipynb`, then `git rm` the latter.

- Setup: `scene = tutorial_scene("preproc")`.
- `## 1. Capture quality`: `load_video_quality` on the stage's `video_quality_report.json` (in `scene.backend_dir.parent` or wherever `Reconstructor` writes it; check `scene.outputs` / the preproc code). No second decode. `plot_photometric`, `plot_motion`, `plot_correlation`, `plot_frame_extremes`.
- `## 2. Choosing keyframes`: `filter_frame_quality`, then `sample_fps`, `sample_uniform`, `sample_optical_flow` on the same eligible pool, `plot_selection` once comparing them.
- `## 3. The keyframe store`: list `scene.images_dir`, read `frames.json`, show a 4-frame montage.
- `## 4. Lens distortion`: two sentences and the `preproc.undistort` key. No calibration run.
- `## In a pipeline run`: `preproc: {max_frames, frame_selection, undistort}`.
- Target: the 18 em-dashes of the old keyframe page are gone (cap 3 page-wide).

## Task 6: `02_pointcloud/reconstruction.ipynb`

**Files:** Modify `reconstruction.ipynb`; fold in `reconstruction_quality_report.ipynb`, then `git rm` it.

- Setup: `scene = tutorial_scene("preproc", "pointcloud", "reconstruction_quality_report")`.
- `## 1. Feedforward reconstruction`: `get_creator("vggt_omega")` shown as the call the stage makes (not run again); `result = scene.result`; `PointcloudResult` fields; `intrinsics` (full-res) vs `model_intrinsics` (model grid), printed side by side; `clean_pointcloud`; cloud + frustums. One line naming the other backends.
- `## 2. Structure from motion`: `InstantSfMCreator` longhand, output to `work_dir("reconstruction")`. One run.
- `## 3. Judging a reconstruction`: the stage's `reconstruction_quality_report.json` tables for feedforward; `compute_reconstruction_quality` on the SfM result in memory; one comparison figure; one paragraph on reading it.
- `## In a pipeline run`: `pointcloud: {method, backend}`, `reconstruction_quality_report` stage.

## Task 7: `02_pointcloud/refinement.ipynb`

**Files:** Modify `refinement.ipynb`.

- Setup: `scene = tutorial_scene("preproc", "pointcloud")`. Remove the `SUBMAP_SIZE` knob and any frame-count knob.
- `## 1. Bundle adjustment`: `BundleAdjustment(...).refine` on `scene.result` (read the `refine` stage in `reconstructor.py` and copy its call shape exactly: one call per line). Loss terms, focal before/after, camera shift.
- `## 2. Loop closure`: `LoopClosure(creator, LoopClosureConfig(), ba=BundleAdjustmentConfig())` into `work_dir("refinement")`, production `submap_size` (default 20). Trajectories before and after.
- `## In a pipeline run`: `pointcloud: {bundle_adjustment, loop_closure}`; the LC × BA rule in one sentence.

## Task 8: `03_splats/train_splats.ipynb`

**Files:** Modify `train_splats.ipynb`.

- Setup: `scene = tutorial_scene("preproc", "pointcloud", "splats")`. This is the only page that names `splats`; no other page reads its output.
- `## 1. Configuration`: the scaffold-2DGS keys from `scene.config["splats"]`, and the two rules (no `sh_degree*` with scaffold; `opacity_reg` 0). State the production `max_steps` 30000 vs 1000 here.
- `## 2. Training report`: the stage's quality report (PSNR / SSIM table, loss curve).
- `## 3. Renders`: `load_checkpoint` + `render_views`, downscaled gallery (≤ 4 views).
- Remove the longhand `train` call and its frame / depth-target glue.
- `## In a pipeline run`: `splats: {enabled: true, representation, primitive, max_steps}`; say splats feed no other stage.

## Task 9: `04_mesh/mesh.ipynb`

**Files:** Create from `tsdf_mesh.ipynb` (`git mv`), fold in `texturing.ipynb`, `git rm` it.

- Setup: `scene = tutorial_scene("preproc", "pointcloud", "mesh")`.
- `## 1. TSDF fusion`: prose: `sdf_trunc`, not `voxel_size`, sets the thinnest structure that survives. Longhand into `work_dir("mesh")`: `frame_depths` from the VGGT-Omega zarr, `sky_masks`, `create_tsdf_mesh`.
- `## 2. Cleaning`: `clean_repair_mesh`, `prepare_mesh`; before/after face counts and one render.
- `## 3. Texture`: the stage's `mesh.ply` and `texture/` atlas (shown at 1024 px), textured render; occluder / color gain / view charts explanation kept from the old texturing page, rewritten to the prose rules.
- No splat source anywhere; delete any `mesh.source: splats` comparison.
- `## In a pipeline run`: `mesh: {source: feedforward, mask_sky, sdf_trunc, texture}`.

## Task 10: `05_semantics/feature_extraction.ipynb`

**Files:** Modify.

- No scene. `QUERY_IMAGE` only. Content unchanged; apply the page rules (readability, line length, one call per line, numbered headings, no autoreload).

## Task 11: `05_semantics/segmentation.ipynb`

**Files:** Modify. Same as Task 10.

## Task 12: `05_semantics/lifting_and_query.ipynb`

**Files:** Modify.

- Setup: `scene = tutorial_scene("preproc", "pointcloud", "mesh", "semantics")` (maskclip).
- `## 1. Features per frame`: `MaskCLIPExtractor` forward over `scene.images_dir` keyframes, kept in memory (no `extract_feature_cache`).
- `## 2. Compressing`: `FeatureAutoencoder` fit; `recon_cosine`.
- `## 3. Lifting onto points`: `lift_features` onto `scene.result` points; `write_point_features` then `read_point_features` in `work_dir("lifting_and_query")`.
- `## 4. Querying the mesh`: open `scene.outputs["semantics"]` (`maskclip_lifted.zarr`), read `vertex_features`; `score_queries` for two text prompts; `transfer_features` smoothing; plot heat on the mesh.
- `## In a pipeline run`: `semantics: {enabled, extractor, max_epochs}`; then `python -m collab_splats.viewer <backend_dir>`.
- Confirm the import locations of `score_queries` and `transfer_features` on the rebased branch before writing (`grep -rn "def score_queries\|def transfer_features" collab_splats`).

## Task 13: `05_semantics/ocr_lens.ipynb`

**Files:** Modify.

- Setup: `scene = tutorial_scene("preproc", "pointcloud", "mesh", "semantics", extractor="ocr_lens")`.
- `## 1. What the lens stores`: open `scene.outputs["semantics"]` (`ocr_lens_lifted.zarr`): `vertex_word_ids` int16 (V, 64), `vertex_word_probs` fp16, attr `words`. Shapes printed once.
- `## 2. Top words`: the most frequent top-1 words over vertices.
- `## 3. Probing a word`: one word's probability as heat on the mesh.
- Prose keeps why the stage decodes per frame before lifting (the decoder's RMSNorm is non-linear, so decoding averaged features is wrong). Delete `OCRLensExtractor`, `load_decoder`, `word_vocabulary`, `word_probabilities` longhand and the chunked vertex lift.
- `## In a pipeline run`: `semantics: {extractor: ocr_lens}`; the viewer command. Note the LLaVA-1.6 weights must be cached.

## Task 14: `06_localization/localization.ipynb`

**Files:** Modify.

- [ ] **Step 1: Check frame 0 is a keyframe**

```bash
cd $W && PYTHONPATH=$W $PY - <<'EOF'
import json
from pathlib import Path
path = next(Path("data/tutorial_scene").rglob("frames.json"))
frames = json.loads(path.read_text())
print(type(frames), list(frames)[:3] if isinstance(frames, dict) else frames[:3])
EOF
```

Find the source-frame index field and confirm 0 is selected. If frame 0 is not a keyframe, stop and report: the page needs a different ground-truth plan.

- [ ] **Step 2: Write the page**

- Setup: `scene = tutorial_scene("preproc", "pointcloud")`; `localizer = CameraLocalizer.from_pointcloud(..., extractor=LocalMatcher("loma"))` with the zarr path in `work_dir("localization")`.
- `## 1. A frame from the video`: localize `REF_FRAME`; look up the reconstruction's pose for frame 0; rotation error in degrees and translation error divided by the trajectory extent (max camera-center distance). Use `geometry.transforms` helpers for the rotation angle; one call per line.
- `## 2. A frame from another video`: localize `QUERY_IMAGE`; `plot_correspondences`, `plot_inlier_distribution`, pose in the scene, `plot_reprojection`. Visual check only; say so.
- `## In a pipeline run`: `localization: {enabled, matcher: loma, query}`.

- [ ] **Step 3: Execute, read outputs, contract check, commit** (as in "Page tasks").

Expected outcome for section 1: rotation error of a few degrees or less. If it is large, report the numbers rather than tuning the page.

---

## Task 15: Full sweep and cold start

- [ ] **Step 1: Fresh scene**

```bash
cd $W && rm -rf data/tutorial_scene
```

- [ ] **Step 2: Execute all ten pages serially, in PAGES order of the index** (01 → 06; within 05: feature_extraction, segmentation, lifting_and_query, ocr_lens). Each must exit 0. Re-read outputs only where numbers in the prose could have moved.

- [ ] **Step 3: Cold start**

```bash
cd $W && rm -rf data/tutorial_scene
```

Execute `06_localization/localization.ipynb` alone. Exit 0 proves a page builds its own inputs. Its outputs come from the same code as the sweep, so commit them as they are.

- [ ] **Step 4: Commit all executed notebooks**

```bash
cd $W && git commit --only docs/source/tutorials -m "docs(tutorials): full sweep on the shared scene

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 16: Gates

- [ ] **Step 1: Tests**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs tests/test_docstring_contract.py tests/test_import_style.py -p no:randomly -q
echo "exit=$?"
```

Expected: `exit=0`. Compare any failure with `docs/known-test-failures.md` before calling it known.

- [ ] **Step 2: Sphinx**

```bash
cd $W && PYTHONPATH=$W /opt/venv/reconstruction/bin/sphinx-build -b html -q docs/source /tmp/claude-0/-workspace-collab-splats/e566daea-b71e-409d-83bc-0323a030bbb2/scratchpad/html
echo "exit=$?"
```

Expected: `exit=0`, no warning naming a tutorial page.

- [ ] **Step 3: Size**

```bash
cd $W && du -cb docs/source/tutorials/*/*.ipynb | tail -1
```

Expected: ≤ 33554432 bytes (32 MB). If over, shrink the largest page's galleries and re-execute it.

- [ ] **Step 4: Readability spot read.** Open three pages at random in the rendered HTML and read them through. Any slogan, wall of prose or unexplained figure: fix, re-execute, recommit.

---

## Task 17: Land on clean/final

Ask the user before Step 4: the main checkout has `clean/final` checked out with uncommitted edits to `README.md` and `docs/superpowers/specs/2026-09-09-clean-final-dead-code-design.md`. The README edit touches the same file as this branch.

- [ ] **Step 1: Re-sync if clean/final moved**

```bash
cd $W && git merge-base --is-ancestor clean/final HEAD && echo up-to-date
```

If not up to date: `git rebase clean/final`, re-run Task 16 Step 1, re-execute only pages whose imports changed.

- [ ] **Step 2: Backup refs**

```bash
cd $W && git update-ref refs/backup/tutorial-release/pre-squash HEAD && \
  git update-ref refs/backup/tutorial-release/clean-final-before $(git rev-parse clean/final)
```

- [ ] **Step 3: Build the landing commits off-tree**

```bash
cd $W && BASE=$(git rev-parse clean/final) && git checkout -q --detach $BASE && \
  git cherry-pick 7a03db99 0fd93dbb && \
  TREE=$(git rev-parse refs/backup/tutorial-release/pre-squash^{tree}) && \
  SQ=$(git commit-tree $TREE -p HEAD -F - <<'EOF'
docs(tutorials): ten pages on one shared scene

- tutorial.py owns data/tutorial_scene; pages run only missing stages
- quickstart moves to README Getting Started
- mesh from VGGT-Omega depth only; splats a terminal stage on its own page
- semantics pages read the lifted vertex stores
- localization compares frame 0 against its fitted pose
- contract gate: ten pages, readability caps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
) && echo $SQ && git diff --stat $SQ refs/backup/tutorial-release/pre-squash
```

Expected: the cherry-picks apply (they may already be on `clean/final`; if `git cherry-pick` reports empty, `git cherry-pick --skip`). The final diff stat is empty (tree equality).

- [ ] **Step 4: CHANGELOG, CLAUDE.md, spec** (on top of `$SQ`, detached)

- Append to `docs/superpowers/CHANGELOG.md` a `tutorial-rework (2026-10-08)` entry: shared scene, ten pages, readability gate, the two fixes.
- `CLAUDE.md`: remove the `tutorial-rework` In-Flight bullet; add `tutorial-rework (2026-10-08)` at the top of "Recently Completed" and drop the oldest of the five.
- The spec on `$SQ` already is the revised one; check `git show $SQ:docs/superpowers/specs/2026-09-09-tutorial-rework-design.md | head -3` shows "ten pages".

```bash
cd $W && git add -f docs/superpowers/CHANGELOG.md && git commit --only docs/superpowers/CHANGELOG.md CLAUDE.md \
  -m "docs(changelog): tutorial-rework landed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Fast-forward clean/final (after the user answers the README question)**

`clean/final` is checked out in the main checkout, so move it there; a bare `update-ref` would leave that checkout's index stale.

```bash
LAND=$(git -C $W rev-parse HEAD) && cd /workspace/collab-splats &&   test "$(git rev-parse HEAD)" = "$(git rev-parse refs/backup/tutorial-release/clean-final-before)" &&   git merge --ff-only $LAND && git status --short
```

Expected: fast-forward, then only the user's own uncommitted edits in `git status`. If git refuses because the uncommitted `README.md` would be overwritten, stop and ask the user how to proceed; never stash or discard their edit. Then `cd $W && git checkout -q clean/tutorials` to leave the detached HEAD.

- [ ] **Step 6: Report, then clean up on confirmation.** Tell the user: landed sha, gate results, sweep wall-clock, size. Ask before deleting `clean/tutorials`, the worktree, and the stale `clean/localization` branch. Do not push.
