# Tutorial Rework (14 pages on clean/final) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the tutorial as fourteen self-contained notebooks on the post-release `clean/final` API, twelve executed now and two gated on unfinished branches.

**Architecture:** Rebase `clean/tutorials` onto `clean/final`, then merge the two helper files into one `tutorial.py` (inputs, pyvista backend, `tutorial_scene`). Next, extend the contract gate: a 14-page set, a matplotlib cap, and an outputs check that skips the GATED pages. Then author each page with an nbformat builder script and execute it serially with nbconvert. Every page builds its upstream stages through `Reconstructor.run(stages=[...])` and writes the stage it teaches out in full.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), nbformat, nbconvert, pytest, Sphinx + nbsphinx (`nbsphinx_execute = "never"`), pyvista (static backend when headless).

**Spec:** `docs/superpowers/specs/2026-09-09-tutorial-rework-design.md` (revised 2026-10-02).
**Supersedes:** `docs/superpowers/plans/2026-09-09-tutorial-rework.md` after its Task 5.

**Status (2026-10-06):** rebased onto clean/final `4781e72c` (joint RGB-D BA, BA per LC window, localization on vismatch). Both GATED pages now executed; `GATED` is empty, so all 14 pages carry outputs.

---

## Conventions every task follows

- **Worktree:** `W=/workspace/collab-splats/.worktrees/tutorial-rework`. Run every command from `$W`, with `PYTHONPATH=$W`:
  ```bash
  cd $W && PYTHONPATH=$W /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
  ```
  Expected: a path under `$W/collab_splats/`. Any other path means you are testing the main checkout. Stop and fix it before going on.
- **Python:** always `/opt/venv/reconstruction/bin/python`, never bare `python`. Write it as `$PY` in the steps below.
- **Commits:** a shared git index means other sessions run concurrently. Always commit with `git commit --only <paths>`. Never `git add -A`, never amend, never stash. `docs/superpowers/` is gitignored but tracked, so commit files there with `git add -f <path>` then `git commit --only <path>`.
- **Serial GPU:** run one executing notebook at a time. The cgroup cap is 46.6 GB, and a concurrent feedforward run or splat train OOMs. Run nothing else heavy alongside.
- **Execution command:** used for every write-now page. `<page>` is the notebook path relative to `docs/source/tutorials/`.
  ```bash
  cd $W/docs/source/tutorials/$(dirname <page>) && PYVISTA_OFF_SCREEN=1 PYTHONPATH=$W xvfb-run -a -s "-screen 0 1280x1024x24" \
    /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 $(basename <page>)
  ```
  `xvfb-run` is required: the container has no EGL or OSMesa, so a headless VTK render segfaults the kernel (`DeadKernelError`) without a virtual X server.
  Use the Bash tool with `run_in_background: true` and wait for its completion notification. Never `| tail` the command, since a pipe hides the exit code. Confirm the exit status is 0 before going on.
- **Notebook builder prologue:** every page task writes its notebook with a heredoc of this shape (`OUT` and `cells` vary):
  ```python
  import nbformat as nbf
  from pathlib import Path

  md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
  OUT = Path("docs/source/tutorials/<dir>/<page>.ipynb")
  cells = [...]
  meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
  OUT.parent.mkdir(parents=True, exist_ok=True)
  nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
  ```
- **Notebook rules:** these are enforced by `tests/docs/test_tutorial_contract.py`.
  - No `def` or `class` in any cell.
  - No `data/outputs`, `OUTPUT_DIR`, `IMAGES_DIR`, `TUTORIAL_CACHE` or `sys.path.insert`.
  - No "run NN_x.ipynb first".
  - At most 10 hand-written `plt.` / `ax.` / `axes[` calls.
  - Every `collab_splats` import resolves.
- **Comments in cells:** follow the user's single-line block-comment preference (memory `feedback_single_line_comments`). Use one plain line per block, saying what the code does.
- **Standing rule:** semantics stays off in every run (`semantics: {enabled: false}`). It is only implied when a page names its stages explicitly, and only the quickstart uses a bare `run()`.

## File structure

| Path | Responsibility | Action |
|---|---|---|
| `docs/source/tutorials/tutorial.py` | committed inputs, pyvista backend at load, `tutorial_scene` | create (replaces two files) |
| `docs/source/tutorials/tutorial_config.py` | old inputs file | delete |
| `docs/source/tutorials/notebook_utils.py` | old helpers; imports dead `collab_splats.wrapper` | delete |
| `tests/docs/test_tutorial_helpers.py` | unit tests for `tutorial.py` | create (merges two files) |
| `tests/docs/test_notebook_utils.py`, `tests/docs/test_tutorial_config.py` | old helper tests | delete |
| `tests/docs/test_tutorial_contract.py` | per-notebook contract gate | modify: 14-page set, matplotlib cap, outputs/GATED |
| `docs/source/tutorials/index.rst` | toctree | rewrite to 14 pages |
| `docs/source/tutorials/00_quickstart/pipeline.ipynb` | Reconstructor end to end | create |
| `docs/source/tutorials/01_preprocessing/video_quality.ipynb` | measure the clip | create |
| `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` | select from the clip | modify (drop QA §1–2, fix drift) |
| `docs/source/tutorials/02_pointcloud/reconstruction.ipynb` | feedforward + SfM | rewrite |
| `docs/source/tutorials/02_pointcloud/reconstruction_quality_report.ipynb` | reference-free quality | create |
| `docs/source/tutorials/02_pointcloud/refinement.ipynb` | BA + LC (GATED) | rewrite, not executed |
| `docs/source/tutorials/03_splats/train_splats.ipynb` | Scaffold-2DGS train + render | rewrite |
| `docs/source/tutorials/04_mesh/tsdf_mesh.ipynb` | TSDF from feedforward vs splats | move from `05_mesh/`, rewrite |
| `docs/source/tutorials/04_mesh/texturing.ipynb` | UV atlas bake (GATED) | create, not executed |
| `docs/source/tutorials/05_semantics/{feature_extraction,segmentation,lifting_and_query}.ipynb` | semantics | move from `04_semantics/`, rewrite |
| `docs/source/tutorials/05_semantics/ocr_lens.ipynb` | OCR lens words on the mesh | create |
| `docs/source/tutorials/06_localization/localization.ipynb` | query image → pose | rewrite |

## Package-first table (per page)

Every page must use these package functions. A cell that re-derives one of them is a review failure.

| Page | Package calls |
|---|---|
| pipeline | `tutorial_scene` → `Reconstructor`, `validate_config`, `run`, `done`, `outputs`, path properties |
| video_quality | `get_video_info`, `load_video_quality`, `plot_photometric`, `plot_motion`, `plot_correlation`, `plot_frame_extremes` |
| keyframe_extraction | `load_video_quality`, `filter_frame_quality`, `sample_uniform` / `sample_fps` / `sample_optical_flow`, `plot_selection`, `plot_frame_grid`, `write_frames`, `calibrate_camera`, `undistort_frames` |
| reconstruction | `get_creator`, `PointcloudResult`, `clean_pointcloud`, `InstantSfMCreator`, `pointcloud_to_polydata`, `create_camera_frustum_pyvista` |
| reconstruction_quality_report | `run([... "reconstruction_quality_report"])`, `compute_reconstruction_quality`, `InstantSfMCreator` |
| refinement (GATED) | `BundleAdjustment`, `BundleAdjustmentConfig`, `LoopClosure`, `LoopClosureConfig`, `get_creator`, `clean_pointcloud`, `create_camera_frustum_pyvista` |
| train_splats | `SplatsConfig.from_dict`, `train`, `load_checkpoint`, `render_views`, `read_frames`, `frame_idx_from_path` |
| tsdf_mesh | `frame_depths`, `render_tsdf_inputs`, `sky_masks`, `create_tsdf_mesh`, `clean_repair_mesh`, `prepare_mesh`, `invert_poses` |
| texturing (GATED) | `create_texture_mesh`, `frame_depths`, `invert_poses`, `read_frames` |
| feature_extraction | `MaskCLIPExtractor`, `Talk2DinoExtractor`, `score_queries`, `feature_viz_row`, `read_frames` |
| segmentation | `MobileSAMSegmentation`, `overlay_masks`, `aggregate_masked_features`, `MaskCLIPExtractor` |
| lifting_and_query | `extract_feature_cache`, `store_rows`, `FeatureAutoencoder`, `lift_features`, `write_point_features`, `read_point_features`, `transfer_features`, `score_queries` |
| ocr_lens | `OCRLensExtractor`, `extract_feature_cache`, `store_rows`, `load_processor`, `word_vocabulary`, `load_decoder`, `word_probabilities`, `lift_features` |
| localization | `CameraLocalizer.from_feedforward`, `LocalMatcher`, `localize`, `correspondences_for_ref`, `plot_correspondences`, `plot_inlier_distribution`, `create_camera_frustum_pyvista` |

## Package gaps (flagged for the user, not implemented here)

1. **`plot_reconstruction_quality`**: nothing plots the report tables. The `reconstruction_quality_report` page uses ≤10 inline matplotlib calls until the user decides.
2. **Chunked vertex lift**: `lift_features` onto mesh vertices in chunks exists only as `_lift_onto` in `docs/examples/ocr_lens_viewer.py`. Pages `lifting_and_query` and `ocr_lens` inline a 5-line loop.
3. **Per-frame word-probability maps**: likewise, this is `_probability_maps` in the same example. The `ocr_lens` page inlines an 8-line loop.

Report all three to the user at the end (Task 20). Do not add them to `collab_splats` without approval.

---

### Task 1: Rebase `clean/tutorials` onto `clean/final`

**Files:**
- Modify: whole branch history (rebase)
- Conflict: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`, `docs/source/tutorials/03_splats/train_splats.ipynb`, `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`, `docs/source/tutorials/evals/ground_truth_evals.ipynb`

- [ ] **Step 1: Check for an in-progress sequencer, then commit the uncommitted keyframe edit as WIP**

```bash
cd $W && ls .git 2>/dev/null; cat .git   # worktree: .git is a file pointing at the gitdir
GITDIR=$(git rev-parse --git-dir); ls $GITDIR/rebase-merge $GITDIR/rebase-apply $GITDIR/sequencer 2>&1 | head
git status --short
git commit --only docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb \
  -m "wip(tutorials): keyframe_extraction pre-rebase edit"
```
Expected: no rebase or sequencer directories, and one commit created. If `git status` shows other modified files, stop and ask: they belong to someone else.

- [ ] **Step 2: Back up the tip**

```bash
git update-ref refs/backup/tutorial-rework/clean-tutorials-pre-rebase HEAD
git rev-parse refs/backup/tutorial-rework/clean-tutorials-pre-rebase HEAD
```
Expected: two identical SHAs.

- [ ] **Step 3: Rebase**

```bash
git rebase clean/final
```
Expected: stops on conflicts. Resolve each one as it appears:
- `bundle_adjustment.ipynb`, `evals/ground_truth_evals.ipynb` (modify/delete): keep the delete with `git rm <path>`.
- `03_splats/train_splats.ipynb`: keep the branch side with `git checkout --theirs <path> && git add <path>`. During a rebase, `--theirs` is the commit being replayed, i.e. the branch. Task 12 rewrites this page anyway.
- `01_preprocessing/keyframe_extraction.ipynb`: keep the branch side the same way, then re-apply `e20b9f31`'s change by hand:
  ```bash
  git show e20b9f31 --stat; git show e20b9f31 -- docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb | head -80
  ```
  Apply the cell-source change it makes to the matching cell of the branch notebook (edit with an nbformat script, not by hand-editing the JSON). If its change is to a QA cell that Task 6 deletes anyway, note that in the commit message and skip it.
- Then run `git rebase --continue`, and repeat until the rebase is done.

- [ ] **Step 4: Verify the rebase**

```bash
git log --oneline clean/final..HEAD | cat
git merge-base --is-ancestor clean/final HEAD && echo REBASED
ls docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb docs/source/tutorials/evals 2>&1
```
Expected: the branch's commits appear on top, `REBASED` prints, and both deleted paths are absent.

- [ ] **Step 5: Run the current gate to see the drift baseline**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs -q 2>&1 | tail -30
```
Expected: failures in `test_imports_resolve` for the draft pages (`FeedforwardResult`, `splats.rendering.load_checkpoint`) and in `test_notebook_utils.py` (`collab_splats.wrapper`). Record the counts in the Task 1 report. They are the baseline that Tasks 4–5 and the page tasks remove.

No commit for this step: the rebase is the change.

---

### Task 2: Re-verify the API every page uses

**Files:**
- Create: `$SCRATCH/api_check.py`, where `$SCRATCH` is the session scratchpad. Do not commit this file.

- [ ] **Step 1: Write the check**

```python
# $SCRATCH/api_check.py — every call the page tasks make, as signature assertions
import inspect

from collab_splats.geometry.metrics import compute_reconstruction_quality
from collab_splats.geometry.transforms import invert_poses
from collab_splats.localization import (
    CameraLocalizer,
    LocalMatcher,
    correspondences_for_ref,
    plot_correspondences,
    plot_inlier_distribution,
)
from collab_splats.mesh import clean_repair_mesh, create_texture_mesh, create_tsdf_mesh, prepare_mesh
from collab_splats.pointcloud import PointcloudResult, get_creator
from collab_splats.pointcloud.sfm import InstantSfMCreator
from collab_splats.pointcloud.utils import clean_pointcloud, frame_depths
from collab_splats.preproc import frames
from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.sampling import filter_frame_quality, sample_fps, sample_optical_flow, sample_uniform
from collab_splats.preproc.undistort import calibrate_camera, undistort_frames
from collab_splats.preproc.video import get_video_info
from collab_splats.preproc.viz import (
    plot_correlation,
    plot_frame_extremes,
    plot_frame_grid,
    plot_motion,
    plot_photometric,
    plot_selection,
)
from collab_splats.reconstructor import Reconstructor, store_rows
from collab_splats.semantics import (
    FeatureAutoencoder,
    MaskCLIPExtractor,
    MobileSAMSegmentation,
    Talk2DinoExtractor,
    aggregate_masked_features,
    extract_feature_cache,
    read_point_features,
    sky_masks,
    write_point_features,
)
from collab_splats.semantics.features.ocr_lens import (
    OCRLensExtractor,
    load_decoder,
    load_processor,
    word_probabilities,
    word_vocabulary,
)
from collab_splats.semantics.lifting import lift_features, transfer_features
from collab_splats.splats import SplatsConfig, load_checkpoint, render_views, train
from collab_splats.splats.checkpoint import render_tsdf_inputs
from collab_splats.utils.notebook import feature_viz_row
from collab_splats.utils.visualization import create_camera_frustum_pyvista, overlay_masks, pointcloud_to_polydata

EXPECT = {
    load_video_quality: ["video_path", "report_path"],
    filter_frame_quality: ["report", "sharpness_k", "max_clipped_frac"],
    sample_fps: ["video_path", "fps", "report", "max_frames", "quality"],
    frames.write_frames: ["dir", "frames", "idxs"],
    frames.read_frames: ["dir", "idxs"],
    calibrate_camera: ["images_dir"],
    undistort_frames: ["frames_in", "camera"],
    plot_photometric: ["report", "out_dir", "selected"],
    plot_frame_extremes: ["report", "video_path", "out_dir", "column", "n"],
    plot_correlation: ["report", "x", "y", "out_dir"],
    clean_pointcloud: ["result", "remove_outliers", "max_points"],
    frame_depths: ["result", "rgbs", "conf_percentile"],
    InstantSfMCreator.create_pointcloud: ["self", "images_dir", "out_dir", "model_dir"],
    compute_reconstruction_quality: [
        "depth", "model_intrinsics", "intrinsics", "extrinsics", "original_coords",
        "image_names", "confidence", "images",
    ],
    train: ["cfg", "images", "world_to_cam", "intrinsics", "points", "colors", "out_dir", "depth_targets", "image_ids"],
    load_checkpoint: ["path", "device"],
    render_views: ["model", "camera_opt", "cam_to_world", "intrinsics", "height", "width"],
    render_tsdf_inputs: ["ckpt_path", "images_dir"],
    create_tsdf_mesh: ["depths", "rgbs", "c2w", "K", "out_dir", "voxel_size", "depth_trunc", "sdf_trunc"],
    clean_repair_mesh: ["mesh_path"],
    prepare_mesh: ["mesh", "voxel_size"],
    create_texture_mesh: ["mesh", "occluder", "out_dir", "rgbs", "c2w", "K", "voxel_size", "tex_size"],
    sky_masks: ["images_dir", "idxs"],
    extract_feature_cache: ["extractor", "images_dir", "cache_dir", "extractor_kwargs"],
    store_rows: ["images_dir", "names"],
    lift_features: ["frame_features", "result"],
    transfer_features: ["targets", "points", "features", "k", "max_dist"],
    write_point_features: ["store_path", "codes", "ae"],
    read_point_features: ["store_path"],
    word_probabilities: ["features", "decoder", "vocab", "chunk"],
    word_vocabulary: ["tokenizer", "words"],
    feature_viz_row: ["axes", "frame", "features", "sim_map", "title_prefix", "query_label"],
    overlay_masks: ["image", "masks", "alpha"],
    aggregate_masked_features: ["features", "masks", "resolution", "final_resolution"],
    CameraLocalizer.from_feedforward: ["result", "images", "ids", "extractor"],
    correspondences_for_ref: ["loc", "ref_idx"],
    plot_correspondences: ["query_image", "ref_image", "query_px", "ref_px", "inlier_mask"],
    plot_inlier_distribution: ["ref_frame_indices", "inlier_mask", "n_frames"],
    create_camera_frustum_pyvista: ["pose", "scale"],
    Reconstructor.run: ["self", "stages", "overwrite"],
}

bad = []
for fn, params in EXPECT.items():
    have = list(inspect.signature(fn).parameters)
    missing = [p for p in params if p not in have]
    if missing:
        bad.append(f"{fn.__qualname__}: missing {missing}; has {have}")

# Shapes the page cells index into
assert get_creator("vggt_omega").__name__, "get_creator returns a class"
assert {"frames", "depth_pairs", "depth_residual_histogram", "photometric_pairs"}
assert hasattr(SplatsConfig, "from_dict")
assert hasattr(CameraLocalizer, "localize")
assert hasattr(FeatureAutoencoder, "fit") and hasattr(FeatureAutoencoder, "encode")
assert hasattr(MaskCLIPExtractor, "score_queries") and hasattr(Talk2DinoExtractor, "score_queries")
assert hasattr(MobileSAMSegmentation, "segment")
assert hasattr(OCRLensExtractor, "forward") and callable(load_decoder) and callable(load_processor)
assert callable(invert_poses) and callable(get_video_info)
assert callable(sample_uniform) and callable(sample_optical_flow) and callable(plot_motion)
assert callable(plot_selection) and callable(plot_frame_grid) and callable(LocalMatcher)
assert callable(pointcloud_to_polydata) and PointcloudResult

print("\n".join(bad) if bad else "API OK")
```

- [ ] **Step 2: Run it**

```bash
cd $W && PYTHONPATH=$W $PY $SCRATCH/api_check.py 2>&1 | grep -v -i warn | tail -20
```
Expected: `API OK`. If any line reports a missing parameter, the page cells below that use it are stale. Fix them in this plan file (one commit, `docs(plans): tutorial rework — API re-verify fixes`) before starting the page tasks. Do not change `collab_splats`.

- [ ] **Step 3: Check the feedforward creator kwargs the reconstruction page passes**

```bash
cd $W && sed -n '/^pointcloud:/,/^[a-z_]*:$/p' configs/base.yaml | head -40
cd $W && PYTHONPATH=$W $PY -c "
import inspect; from collab_splats.pointcloud import get_creator
print(inspect.signature(get_creator('vggt_omega').__init__))"
```
Expected: the signature accepts `clean` and `max_points`, and `configs/base.yaml` has a `pointcloud.vggt_omega` block (possibly empty). If the block is missing, use `{}` in the Task 8 cell.

---

### Task 3: Mark the old plan superseded; fix the spec's two stale mentions

**Files:**
- Modify: `docs/superpowers/plans/2026-09-09-tutorial-rework.md:1-10`
- Modify: `docs/superpowers/specs/2026-09-09-tutorial-rework-design.md` (§01 `video_quality`, §01 `keyframe_extraction`)

- [ ] **Step 1: Add the superseded banner to the old plan**

Insert after its first heading line:

```markdown
> **SUPERSEDED after Task 5** by `docs/superpowers/plans/2026-10-02-tutorial-rework.md`
> (2026-10-02). Tasks 6–15 below target a pre-release API and must not be executed.
```

- [ ] **Step 2: Fix the spec**

Make these two replacements in the spec:
- In §01 `video_quality`, replace `` `compute_video_quality(VIDEO_PATH, output_path=…)`, `load_video_quality`; `` with `` `load_video_quality(VIDEO_PATH, report_path)` (measures and writes once, then reuses — `compute_video_quality` itself returns a dict and writes nothing); ``.
- In §01 `keyframe_extraction`, replace `` `write_frames` into `images/` + `frames.json`. `` with `` `write_frames(images_dir, frames, idxs)` into `images/frame_NNNNNN.png` (no `frames.json` on `clean/final`). ``

- [ ] **Step 3: Commit**

```bash
cd $W && git add -f docs/superpowers/plans/2026-09-09-tutorial-rework.md docs/superpowers/specs/2026-09-09-tutorial-rework-design.md
git commit --only docs/superpowers/plans/2026-09-09-tutorial-rework.md docs/superpowers/specs/2026-09-09-tutorial-rework-design.md \
  -m "docs(specs): tutorial rework — old plan superseded, video report + frames.json drift fixed"
```

---

### Task 4: `tutorial.py`, which replaces `tutorial_config.py` and `notebook_utils.py`

**Files:**
- Create: `docs/source/tutorials/tutorial.py`
- Create: `tests/docs/test_tutorial_helpers.py`
- Delete: `docs/source/tutorials/tutorial_config.py`, `docs/source/tutorials/notebook_utils.py`, `tests/docs/test_notebook_utils.py`, `tests/docs/test_tutorial_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/docs/test_tutorial_helpers.py
"""
Unit tests for docs/source/tutorials/tutorial.py.

- inputs: repo-relative committed paths, and none of the retired shared-cache names
- backend: pyvista backend set at load
- tutorial_scene: a Reconstructor whose every artifact path sits in a fresh tempdir outside the repo
"""

import runpy
from pathlib import Path

import pytest
import pyvista as pv

REPO = Path(__file__).resolve().parents[2]
TUTORIAL = REPO / "docs" / "source" / "tutorials" / "tutorial.py"


@pytest.fixture
def ns(monkeypatch):
    """
    tutorial.py executed in a fresh namespace, with the pyvista backend call recorded.
    """
    calls = []
    monkeypatch.setattr(pv, "set_jupyter_backend", calls.append)
    namespace = runpy.run_path(str(TUTORIAL))
    namespace["_backend_calls"] = calls
    return namespace


def test_paths_are_repo_relative(ns):
    assert ns["REPO_ROOT"] == REPO
    assert ns["VIDEO_PATH"] == REPO / "data/tutorial/tutorial_example-video.mp4"
    assert ns["QUERY_IMAGE"] == REPO / "data/tutorial/tutorial_example-frame.jpg"


def test_exposes_no_shared_cache_names(ns):
    """
    The retired names threaded state between pages; their absence is the isolation property.
    """
    for gone in ("OUTPUT_DIR", "IMAGES_DIR", "RECON", "TUTORIAL_CACHE", "work_dir", "set_notebook_backend"):
        assert gone not in ns, f"{gone} should be gone"


def test_backend_set_on_load(ns, monkeypatch):
    assert ns["_backend_calls"] in (["static"], ["trame"])


def test_backend_static_when_headless(monkeypatch):
    calls = []
    monkeypatch.setattr(pv, "set_jupyter_backend", calls.append)
    monkeypatch.setenv("PYVISTA_OFF_SCREEN", "1")
    runpy.run_path(str(TUTORIAL))
    assert calls == ["static"]


def test_tutorial_scene_paths_live_in_a_fresh_tempdir(ns):
    a = ns["tutorial_scene"]("demo", preproc={"max_frames": 4})
    b = ns["tutorial_scene"]("demo", preproc={"max_frames": 4})

    work = Path(a.config["output_path"])
    assert work != Path(b.config["output_path"]), "two runs of a page must not collide"
    assert REPO not in work.parents
    assert "collab_splats_tutorial_demo_" in work.name

    for path in (a.images_dir, a.backend_dir, a.pointcloud_zarr, a.semantics_cache_dir):
        assert work in path.parents


def test_tutorial_scene_defaults_input_to_the_tutorial_video(ns):
    scene = ns["tutorial_scene"]("demo")
    assert Path(scene.config["input_path"]) == ns["VIDEO_PATH"]


def test_tutorial_scene_deep_merges_over_base_yaml(ns):
    scene = ns["tutorial_scene"]("demo", preproc={"max_frames": 4})
    assert scene.config["preproc"]["max_frames"] == 4
    assert scene.config["preproc"]["frame_selection"] == "fps"
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_helpers.py -q 2>&1 | tail -5
```
Expected: errors with `FileNotFoundError` / `No such file` for `tutorial.py`.

- [ ] **Step 3: Write `tutorial.py`**

```python
# docs/source/tutorials/tutorial.py
"""
Committed inputs and the one helper every tutorial page shares; load with `%run ../tutorial.py`.

- inputs only: no shared output location, no page depends on another having run
- pyvista backend set at load: static when headless (nbconvert), interactive trame otherwise
- tutorial_scene: a Reconstructor writing into this page's own fresh temporary directory
"""

import os
import tempfile
from pathlib import Path

import pyvista as pv

from collab_splats.reconstructor import Reconstructor

########################################
# Committed inputs (read-only)
########################################

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO_PATH = REPO_ROOT / "data/tutorial/tutorial_example-video.mp4"
QUERY_IMAGE = REPO_ROOT / "data/tutorial/tutorial_example-frame.jpg"

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"missing {VIDEO_PATH} — see data/tutorial/README.md")

# Static renders under nbconvert, interactive in a live kernel
pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

########################################
# Scene helper
########################################


def tutorial_scene(name: str, **overrides) -> Reconstructor:
    """
    A Reconstructor writing into a fresh temporary directory for this page.

    - mkdtemp honours $TMPDIR and is never inside the repo; the path is printed and left for inspection
    - overrides are deep-merged over configs/base.yaml by Reconstructor itself

    Args:
        name: short page label, used in the directory name.
        **overrides: config blocks merged over base.yaml; `input_path` defaults to VIDEO_PATH.

    Returns:
        A configured Reconstructor; build upstream stages with `run(stages=[...])`.
    """
    work = Path(tempfile.mkdtemp(prefix=f"collab_splats_tutorial_{name}_"))
    print(f"work dir: {work}")
    config = {"input_path": str(VIDEO_PATH), "output_path": str(work), **overrides}
    return Reconstructor(config)
```

- [ ] **Step 4: Delete the old files and run the test**

```bash
cd $W && git rm -q docs/source/tutorials/tutorial_config.py docs/source/tutorials/notebook_utils.py \
  tests/docs/test_notebook_utils.py tests/docs/test_tutorial_config.py
rm -rf docs/source/tutorials/__pycache__
PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_helpers.py -q 2>&1 | tail -5
```
Expected: `7 passed`. If `test_tutorial_scene_paths_live_in_a_fresh_tempdir` fails because `Reconstructor` validates `input_path` at construction, that is fine: `VIDEO_PATH` exists. If it fails because `semantics_cache_dir` sits under `output_path` but not under `work`, print the four paths and adjust only the assertion's path list to the ones that do live under `output_path`. Do not change `Reconstructor`.

- [ ] **Step 5: Commit**

```bash
cd $W && git add docs/source/tutorials/tutorial.py tests/docs/test_tutorial_helpers.py
git commit --only docs/source/tutorials/tutorial.py tests/docs/test_tutorial_helpers.py \
  docs/source/tutorials/tutorial_config.py docs/source/tutorials/notebook_utils.py \
  tests/docs/test_notebook_utils.py tests/docs/test_tutorial_config.py \
  -m "refactor(tutorials): one tutorial.py replaces tutorial_config + notebook_utils"
```

---

### Task 5: Contract gate — 14-page set, matplotlib cap, outputs/GATED; move the directories; rewrite `index.rst`

**Files:**
- Modify: `tests/docs/test_tutorial_contract.py`
- Move: `docs/source/tutorials/05_mesh/tsdf_mesh.ipynb` → `04_mesh/tsdf_mesh.ipynb`; `04_semantics/*.ipynb` → `05_semantics/`
- Rewrite: `docs/source/tutorials/index.rst`

- [ ] **Step 1: Write the failing gate changes**

In `tests/docs/test_tutorial_contract.py`, replace `test_notebook_set_is_the_nine_pages` with the block below, and add the new constants after `FIRST_PARTY`:

```python
# Pages authored against today's API but executed only once their blocker lands on clean/final
GATED = {
    "02_pointcloud/refinement.ipynb": "feat/rgbd-ba + LC world-grid fix",
    "04_mesh/texturing.ipynb": "texturing decision",
}

# Hand-written matplotlib calls allowed per page; layout around package plots only
MPL_CAP = 10
MPL_CALL = re.compile(r"\bplt\.|\bax\.|\baxes\[")
MPL_OVERRIDES: dict[str, int] = {}

PAGES = [
    "00_quickstart/pipeline.ipynb",
    "01_preprocessing/keyframe_extraction.ipynb",
    "01_preprocessing/video_quality.ipynb",
    "02_pointcloud/reconstruction.ipynb",
    "02_pointcloud/reconstruction_quality_report.ipynb",
    "02_pointcloud/refinement.ipynb",
    "03_splats/train_splats.ipynb",
    "04_mesh/texturing.ipynb",
    "04_mesh/tsdf_mesh.ipynb",
    "05_semantics/feature_extraction.ipynb",
    "05_semantics/lifting_and_query.ipynb",
    "05_semantics/ocr_lens.ipynb",
    "05_semantics/segmentation.ipynb",
    "06_localization/localization.ipynb",
]


def _rel(nb: Path) -> str:
    """
    Notebook path relative to the tutorials root, posix.
    """
    return nb.relative_to(TUTORIALS).as_posix()


def test_notebook_set_is_the_fourteen_pages():
    """
    The set itself is part of the contract — a stray notebook is an ungated page.
    """
    assert sorted(_rel(p) for p in NOTEBOOKS) == PAGES


def test_index_lists_every_page():
    """
    A page missing from the toctree is unreachable in the built docs.
    """
    index = (TUTORIALS / "index.rst").read_text()
    missing = [p for p in PAGES if p.removesuffix(".ipynb") not in index]
    assert not missing, f"index.rst does not list {missing}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_matplotlib_cap(nb: Path):
    """
    Figures come from package plotters; hand-written matplotlib is layout only.
    """
    n = len(MPL_CALL.findall("\n".join(_code_sources(nb))))
    cap = MPL_OVERRIDES.get(_rel(nb), MPL_CAP)
    assert n <= cap, f"{nb.name} has {n} hand-written matplotlib calls (cap {cap}); use a package plotter"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_outputs_present(nb: Path):
    """
    A write-now page is committed executed; a gated page waits for its blocker.
    """
    if _rel(nb) in GATED:
        pytest.skip(f"gated on {GATED[_rel(nb)]}")

    cells = [c for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
    unexecuted = [i for i, c in enumerate(cells) if c.get("execution_count") is None]
    assert not unexecuted, f"{nb.name}: code cells {unexecuted} were never executed"
    assert any(c.get("outputs") for c in cells), f"{nb.name}: no cell has outputs"
```

Also update the module docstring's bullet list: add `- figures: at most MPL_CAP hand-written matplotlib calls per page` and `- executed: outputs committed, except the GATED pages`. In `test_defines_no_helpers`, change the assertion message's `collab_splats or notebook_utils` to `collab_splats or tutorial.py`.

- [ ] **Step 2: Run the gate to make sure it fails**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q 2>&1 | tail -8
```
Expected: `test_notebook_set_is_the_fourteen_pages` and `test_index_lists_every_page` fail, and `test_outputs_present` fails for the unexecuted drafts.

- [ ] **Step 3: Move the directories**

```bash
cd $W/docs/source/tutorials && mkdir -p 04_mesh 05_semantics
git mv 05_mesh/tsdf_mesh.ipynb 04_mesh/tsdf_mesh.ipynb
for f in feature_extraction lifting_and_query segmentation; do git mv 04_semantics/$f.ipynb 05_semantics/$f.ipynb; done
rmdir 05_mesh 04_semantics
```

- [ ] **Step 4: Rewrite `index.rst`**

```rst
Tutorials
=========

Each page is self-contained: it builds everything it needs into its own temporary
directory and runs top to bottom. There is no shared cache and no required order —
start anywhere. Start with the quickstart for the whole pipeline in one call.

Every page builds the stages upstream of its subject with ``Reconstructor.run(stages=[...])``,
then calls the underlying API directly for the stage it is teaching. Frame and step counts
are a small fast profile; each sits in the page's first cell beside its production value.

Pages marked *(pending)* are written against today's API and executed once the work they
depend on lands.

.. toctree::
   :maxdepth: 1
   :caption: 00 · Quickstart

   00_quickstart/pipeline

.. toctree::
   :maxdepth: 1
   :caption: 01 · Preprocessing

   01_preprocessing/video_quality
   01_preprocessing/keyframe_extraction

.. toctree::
   :maxdepth: 1
   :caption: 02 · Pointcloud

   02_pointcloud/reconstruction
   02_pointcloud/reconstruction_quality_report
   02_pointcloud/refinement

.. toctree::
   :maxdepth: 1
   :caption: 03 · Splats

   03_splats/train_splats

.. toctree::
   :maxdepth: 1
   :caption: 04 · Mesh

   04_mesh/tsdf_mesh
   04_mesh/texturing

.. toctree::
   :maxdepth: 1
   :caption: 05 · Semantics

   05_semantics/feature_extraction
   05_semantics/segmentation
   05_semantics/lifting_and_query
   05_semantics/ocr_lens

.. toctree::
   :maxdepth: 1
   :caption: 06 · Localization

   06_localization/localization
```

- [ ] **Step 5: Run the gate; expect only the missing-page and outputs failures**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q 2>&1 | tail -8
```
Expected:
- `test_notebook_set_is_the_fourteen_pages` still fails. It lists the 5 pages not yet created (pipeline, video_quality, reconstruction_quality_report, texturing, ocr_lens), and it turns green at Task 18.
- `test_index_lists_every_page` passes.
- `test_outputs_present` fails for the drafts.

- [ ] **Step 6: Commit**

```bash
cd $W && git commit --only tests/docs/test_tutorial_contract.py docs/source/tutorials/index.rst \
  docs/source/tutorials/04_mesh docs/source/tutorials/05_semantics \
  docs/source/tutorials/05_mesh docs/source/tutorials/04_semantics \
  -m "test(tutorials): 14-page contract, matplotlib cap, outputs gate; mesh before semantics"
```

---

### Task 6: Page `01_preprocessing/video_quality` (write-now)

**Files:**
- Create: `docs/source/tutorials/01_preprocessing/video_quality.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/01_preprocessing/video_quality.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Video Quality

Before any frame is chosen, the clip is measured. `load_video_quality` decodes the video once
and records photometry per frame (blur, exposure, clipping) and motion per frame pair
(translation, parallax, match count). It writes `video_quality_report.json` and reuses it on
every later call.

The report **measures only** — it carries no thresholds and no verdicts. Judging frames is
`filter_frame_quality`'s job, on the keyframe extraction page. In a pipeline run this is the
first thing the `preproc` stage does."""),
    code("""import tempfile
from pathlib import Path

from IPython.display import Image, display

from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.video import get_video_info
from collab_splats.preproc.viz import plot_correlation, plot_frame_extremes, plot_motion, plot_photometric

# This page's own output directory
WORK = Path(tempfile.mkdtemp(prefix="collab_splats_tutorial_video_quality_"))
print(f"work dir: {WORK}")"""),
    md("## §1 — Probe and measure\n\n`get_video_info` reads the container header only. The measurement is the one expensive cell on this page."),
    code("""info = get_video_info(VIDEO_PATH)
print(f"{info['total_frames']} frames  {info['width']}x{info['height']}  {info['fps']:.2f} fps")

# Measure once and write the report; a second call reads the file back
report_path = WORK / "video_quality_report.json"
report = load_video_quality(VIDEO_PATH, report_path)
print(f"report: {report_path} ({report_path.stat().st_size / 1000:.0f} kB)")
print("frame columns:", sorted(report["frames"]))
print("pair columns :", sorted(report["pairs"]))"""),
    md("## §2 — Photometry\n\nEvery frame as measured. Each plotter saves a PNG into the work dir and returns its path."),
    code("display(Image(str(plot_photometric(report, WORK))))"),
    md("## §3 — Motion\n\nPer pair: how far the camera moved and how well the pair's features agreed."),
    code("display(Image(str(plot_motion(report, WORK))))"),
    md("""## §4 — Correlating two columns

`blur` is per frame and `translation_px` per pair; `plot_correlation` reads the frame column at
each pair's first frame so the two line up sample for sample. Fast motion blurs frames — this is
where you see how much."""),
    code("display(Image(str(plot_correlation(report, \"blur\", \"translation_px\", WORK))))"),
    md("## §5 — Extremes\n\nThe sharpest and blurriest frames, side by side. `n=2` keeps the committed page small."),
    code("display(Image(str(plot_frame_extremes(report, VIDEO_PATH, WORK, column=\"blur\", n=2))))"),
    md("""## Where this goes next

- **Keyframe extraction** turns this report into a selection: `filter_frame_quality` decides
  which frames are eligible, and a sampler picks from them.
- In a pipeline run, `preproc` writes the same report beside `images/` and plots it with the
  kept frames marked."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Run the static gate on this page**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k "video_quality" 2>&1 | tail -5
```
Expected: everything passes except `test_outputs_present[video_quality]`.

- [ ] **Step 3: Execute** (Conventions: execution command, `<page>` = `01_preprocessing/video_quality.ipynb`)

Expected: exit 0. Then check the committed size: `ls -la docs/source/tutorials/01_preprocessing/video_quality.ipynb`, which should be under 5 MB. If it is larger, lower `dpi` on `plot_frame_extremes` (`dpi=90`) and re-execute.

- [ ] **Step 4: Gate green for this page, then commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k "video_quality" 2>&1 | tail -3
git add docs/source/tutorials/01_preprocessing/video_quality.ipynb
git commit --only docs/source/tutorials/01_preprocessing/video_quality.ipynb -m "docs(tutorials): video_quality page"
```

---

### Task 7: Page `01_preprocessing/keyframe_extraction` (write-now, edit existing)

**Files:**
- Modify: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`

The existing page (cells 0–17) carries the old helpers, the QA plots that now live on `video_quality`, and two drift bugs: `compute_video_quality(output_path=)` and `write_frames(dir, frames, records, provenance)`.

- [ ] **Step 1: Rewrite the cells with a script**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb")
nb = nbf.read(OUT, as_version=4)
old = nb.cells

# Cells 8-10, 12-14, 17 (gate prose, filter, extremes heading, samplers, plot_selection, choosing) carry over
assert "filter_frame_quality" in "".join(old[9].source) and "sample_fps" in "".join(old[13].source)

cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Keyframe Extraction

A reconstruction never sees the whole video. This page **selects** the frames it does see,
from the measurements the video quality page explains.

Measuring and judging are separate on purpose: the report carries no verdicts, and
`filter_frame_quality` is the only place a frame is judged — so you can re-threshold without
re-decoding.

In a pipeline run this is the `preproc` stage. This page opens it up."""),
    code("""import tempfile
from pathlib import Path

from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.video import get_video_info

# Budget: production runs hundreds of frames; base.yaml caps at 300
N_FRAMES = 16
TARGET_FPS = 2.0  # base.yaml default

# The gate's thresholds in one place; every sampler re-derives the eligible pool from them
QUALITY = {"sharpness_k": 2.0, "max_clipped_frac": 0.25}

# This page's own output directory
WORK = Path(tempfile.mkdtemp(prefix="collab_splats_tutorial_keyframes_"))
print(f"work dir: {WORK}")

# The quality report: the expensive cell, one decode of the whole clip
info = get_video_info(VIDEO_PATH)
report = load_video_quality(VIDEO_PATH, WORK / "video_quality_report.json")
print(f"{info['total_frames']} frames measured")"""),
    old[8],
    old[9],
    old[12],
    old[13],
    old[14],
    md("""## §3 — Undistort (optional)

Action-camera lenses bend straight lines. `calibrate_camera` runs a small SfM to fit the lens;
`undistort_frames` resamples frames onto a pinhole framing and returns the **new** camera — use
that K downstream, never the calibrated one. Base config: `preproc.undistort: false`.

SfM needs overlapping views. Sixteen keyframes spread across 100 s barely overlap — only 2 of 24
register — so this page fits the lens on a short dense clip (every 6th frame of the first 10 s)
and applies it to the keyframes. One lens, one camera: the fit transfers."""),
    code("""import numpy as np

from collab_splats.preproc.frames import write_frames
from collab_splats.preproc.undistort import calibrate_camera, undistort_frames
from collab_splats.preproc.video import extract_frame

# Dense calibration clip: every 6th frame of the first 240, so neighbors overlap
calib_idxs = list(range(0, 240, 6))
calib_frames = np.stack([extract_frame(str(VIDEO_PATH), i) for i in calib_idxs])
write_frames(WORK / "calib", calib_frames, calib_idxs)

# Fit the lens on the clip, then resample the fps keyframes onto the undistorted framing
camera = calibrate_camera(WORK / "calib")
undistorted, pinhole = undistort_frames(fps_frames, camera)
print("calibrated:", camera.model, camera.params.round(3))
print("pinhole   :", pinhole.model, pinhole.params.round(3), undistorted.shape)"""),
    md("""## §4 — The keyframe store

`images/frame_NNNNNN.png`, named by *source* frame index — the COLMAP-style store every
downstream stage reads."""),
    code("""from collab_splats.preproc.viz import plot_frame_grid

# Write the undistorted fps selection under its source frame indices
images_dir = WORK / "images"
idxs = [r["frame_idx"] for r in fps_records]
paths = write_frames(images_dir, undistorted, idxs)
print(f"wrote {len(paths)} frames, first: {paths[0].name}")
plot_frame_grid(undistorted[:12], title=f"undistorted fps keyframes (n={len(idxs)}, first 12)")"""),
    old[17],
]
nb.cells = cells
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs, c.execution_count = [], None
nbf.write(nb, OUT)
EOF
```

- [ ] **Step 2: Check the carried-over cells**

```bash
cd $W && $PY - <<'EOF'
import json
nb = json.load(open("docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb"))
for i, c in enumerate(nb["cells"]):
    print(f"--- [{i}] {c['cell_type']}\n" + "".join(c["source"])[:300])
EOF
```
Expected: the headings run §1 gate, §2 samplers, §3 undistort, §4 store, then choosing a sampler. Edit the carried markdown so its heading numbers match: `## §3 — The quality gate` → `## §1 — The quality gate` and `## §4 — Three samplers` → `## §2 — Three samplers`. No `frames.json`, `work_dir` or `compute_video_quality` should remain. Do this with a short nbformat script that string-replaces in those two cells.

- [ ] **Step 3: Execute** (`<page>` = `01_preprocessing/keyframe_extraction.ipynb`)

Expected: exit 0. (Executed 2026-10-02: calibrating on the fps keyframes failed — 2/24 and 3/31 registered, too little overlap — so §3 calibrates on a dense 40-frame clip; that registers, fx≈1546.)

- [ ] **Step 4: Gate, size check, commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k keyframe 2>&1 | tail -3
ls -la docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
git commit --only docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb \
  -m "docs(tutorials): keyframe_extraction on clean/final — QA split out, undistort, write_frames(idxs)"
```
Expected: the gate passes and the size is under 6 MB.

---

### Task 8: Page `02_pointcloud/reconstruction` (write-now)

**Files:**
- Rewrite: `docs/source/tutorials/02_pointcloud/reconstruction.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/02_pointcloud/reconstruction.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Reconstruction

Keyframes in, a `PointcloudResult` out — by two routes:

- **Feedforward** (VGGT-Omega shown; VGGT-X, MapAnything and LoGeR share the interface): one
  network pass predicts depth and cameras for every frame at once.
- **SfM** (InstantSfM shown; COLMAP and hloc share the interface): match features, solve
  poses globally, then densify with metric depth priors.

Both return the same dataclass, so everything downstream — mesh, splats, semantics,
localization — reads either."""),
    code("""import contextlib
import io
from pathlib import Path

import numpy as np

from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.sfm import InstantSfMCreator
from collab_splats.pointcloud.utils import clean_pointcloud
from collab_splats.utils.visualization import create_camera_frustum_pyvista, pointcloud_to_polydata

# Budget: production base.yaml max_frames is 300; SfM matching needs ~1 s spacing (16 frames registered 7)
MAX_FRAMES = 96

# Keyframes through the pipeline's own preproc stage
scene = tutorial_scene("reconstruction", preproc={"max_frames": MAX_FRAMES})
scene.run(stages=["preproc"])
WORK = Path(scene.config["output_path"])"""),
    md("""## §1 — Feedforward

`get_creator` returns the creator **class** for a backend name; the pointcloud stage instantiates
it with the shared filter knobs plus the backend's own config block. `clean=False` here so §2 can
show the cleaning step on its own."""),
    code("""creator_cls = get_creator("vggt_omega")
creator = creator_cls(clean=False, **scene.config["pointcloud"].get("vggt_omega") or {})

# One forward pass over every keyframe
ff = creator.create_pointcloud(scene.images_dir, WORK / "vggt_omega")
print(type(ff).__name__, ff.points.shape, "points over", len(ff.image_paths), "frames")"""),
    md("""### What a `PointcloudResult` carries

Two intrinsics on purpose: `intrinsics` is the **full-resolution** camera (what the source frames
need), `model_intrinsics` is the **model grid** (what `depth`, `confidence` and `world_points` live
on). Mixing them is the classic bug — a mesh fused with full-res K on model-res depth collapses."""),
    code("""print("model grid     :", ff.model_width, "x", ff.model_height)
print("depth          :", ff.depth.shape, " confidence:", ff.confidence.shape)
print("extrinsics     :", ff.extrinsics.shape, "(world-to-camera, OpenCV)")
print("intrinsics[0]  :\\n", np.round(ff.intrinsics[0], 1))
print("model_K[0]     :\\n", np.round(ff.model_intrinsics[0], 1))"""),
    md("## §2 — Cleaning\n\n`clean_pointcloud` drops statistical outliers and caps the count; result in, result out."),
    code("""cleaned = clean_pointcloud(ff, remove_outliers=True, max_points=200_000)
print(f"{len(ff.points):,} -> {len(cleaned.points):,} points")"""),
    md("## §3 — Look at it\n\nPoints colored from the frames, one frustum per camera."),
    code("""pl = pv.Plotter()
pl.add_mesh(pointcloud_to_polydata(cleaned.points, rgb=cleaned.colors), scalars="rgb", rgb=True, point_size=2)

# One frustum per world-to-camera pose
for w2c in cleaned.extrinsics:
    pl.add_mesh(create_camera_frustum_pyvista(w2c, scale=0.05), color="red", line_width=2)

pl.show()"""),
    md("""## §4 — SfM with depth priors

InstantSfM solves poses globally from feature matches. A sparse SfM model is too thin to mesh,
so the creator then estimates per-frame metric depth with Video-Depth-Anything
(`estimate_depth`) and aligns it to the SfM scale (`align_depth`) — both run inside
`create_pointcloud`. SfM refuses bundle adjustment and loop closure; those are feedforward-only.

SfM only registers frames it can match to a neighbor, so it needs overlap the feedforward model
does not: at 16 keyframes over this 100 s clip (~6 s apart) InstantSfM registered 7 of 16,
which is why this page samples 96."""),
    code("""sfm = InstantSfMCreator(random_seed=0)

# InstantSfM prints every solver phase; keep the page to the summary below
with contextlib.redirect_stdout(io.StringIO()):
    sfm_result = sfm.create_pointcloud(scene.images_dir, WORK / "instantsfm")

# Registration, then the per-frame VDA -> SfM depth scales
scales = np.asarray(sfm.attrs["depth_scales"])
print(f"{len(sfm_result.image_paths)} of {len(ff.image_paths)} frames registered, {len(sfm_result.points):,} points")
print(f"depth scale: median {np.median(scales):.2f}, range {scales.min():.2f}-{scales.max():.2f}")
print("global-scale fallback frames:", sfm.attrs["depth_scale_fallback_frames"])"""),
    code("""pl = pv.Plotter(shape=(1, 2))

# Feedforward left, SfM right, same view
for col, (label, res) in enumerate([("vggt_omega", cleaned), ("instantsfm", sfm_result)]):
    pl.subplot(0, col)
    pl.add_text(label, font_size=10)
    pl.add_mesh(pointcloud_to_polydata(res.points, rgb=res.colors), scalars="rgb", rgb=True, point_size=2)

pl.link_views()
pl.show()"""),
    md("""## In a pipeline run

`scene.run(stages=["pointcloud"])` does §1–§2 with `pointcloud.backend` from the config and
writes `pointcloud.zarr`, a COLMAP model and `sparse_pc.ply` under `scene.backend_dir`.
`PointcloudResult.load_zarr(scene.pointcloud_zarr)` reads it back. The reconstruction quality
report page judges a result like these without ground truth."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Static gate**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k "reconstruction and not quality" 2>&1 | tail -3
```
Expected: only `test_outputs_present` fails.

- [ ] **Step 3: Execute** (`<page>` = `02_pointcloud/reconstruction.ipynb`)

Expected: exit 0. Possible failures and fixes:
- `pointcloud_to_polydata` names its color array differently: print `pointcloud_to_polydata(cleaned.points, rgb=cleaned.colors).array_names` and use that name as `scalars`.
- `sfm.attrs` does not exist until after `create_pointcloud`: it is read after, as written.

- [ ] **Step 4: Gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k "reconstruction and not quality" 2>&1 | tail -3
git commit --only docs/source/tutorials/02_pointcloud/reconstruction.ipynb -m "docs(tutorials): reconstruction page on clean/final"
```

---

### Task 9: Page `02_pointcloud/reconstruction_quality_report` (write-now)

**Files:**
- Create: `docs/source/tutorials/02_pointcloud/reconstruction_quality_report.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/02_pointcloud/reconstruction_quality_report.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Reconstruction Quality Report

No ground truth exists for a handheld capture, so how good is a reconstruction? The views
check each other. Every frame's depth is projected into every other frame; where two views see
the same surface, their depths should agree. The `reconstruction_quality_report` stage writes
those comparisons as columnar tables in `reconstruction_quality_report.json`.

It runs no model and no matcher: it reads `pointcloud.zarr` and `images/` only."""),
    code("""import contextlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collab_splats.geometry.metrics import compute_reconstruction_quality
from collab_splats.pointcloud.sfm import InstantSfMCreator

# Budget: production base.yaml max_frames is 300; §2's SfM needs ~1 s spacing (16 frames registered 5)
MAX_FRAMES = 96

# Keyframes, the feedforward pointcloud and its report through the pipeline
scene = tutorial_scene("quality_report", preproc={"max_frames": MAX_FRAMES})
scene.run(stages=["preproc", "pointcloud", "reconstruction_quality_report"])
report = json.loads(scene.outputs["reconstruction_quality_report"].read_text())
print(json.dumps(report["scene"], indent=2))"""),
    md("""## §1 — The four tables

- **`frames`** — one row per frame: `multiview_agreement` (share of its seen pixels another view
  agrees with) and `median_abs_rel_depth_error` over the pairs touching it.
- **`depth_pairs`** — one row per ordered pair that overlaps: median and IQR relative depth error,
  parallax, overlap in pixels.
- **`depth_residual_histogram`** — every overlapping pixel's relative residual, pre-binned.
- **`photometric_pairs`** — NCC between overlapping views' pixels; present only when frames are
  available.

Agreement uses `rel_thresh=0.05`, looser than the creators' multiview filter: the report
grades the surviving cloud and should not reject what the filter already kept."""),
    code("""frames_df = pd.DataFrame(report["frames"])
pairs_df = pd.DataFrame(report["depth_pairs"])
display(frames_df.round(3))
display(pairs_df.sort_values("median_rel_depth_error", ascending=False).head(5).round(3))"""),
    md("""## §2 — The stage is a thin IO layer

`compute_reconstruction_quality` takes arrays, not files. Call it directly on an in-memory
InstantSfM result — no zarr, no frames, so no photometric table."""),
    code("""# InstantSfM prints every solver phase; keep the page to the tables
with contextlib.redirect_stdout(io.StringIO()):
    sfm = InstantSfMCreator(random_seed=0).create_pointcloud(scene.images_dir, Path(scene.config["output_path"]) / "instantsfm")

names = [Path(str(p)).name for p in sfm.image_paths]

sfm_tables = compute_reconstruction_quality(
    sfm.depth,
    sfm.model_intrinsics,
    sfm.intrinsics,
    sfm.extrinsics,
    sfm.original_coords,
    names,
    sfm.confidence,
    None,
)
print("photometric:", sfm_tables["photometric_pairs"])"""),
    md("""## §3 — Feedforward vs SfM

Per-frame agreement, the pooled residual histogram, and the parallax of each pair against
its error. The histogram's bins are on a bounded transform of the residual, so its tails never
fall off the axis."""),
    code("""sfm_frames = pd.DataFrame(sfm_tables["frames"])
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Per-frame multiview agreement, both routes
axes[0].plot(frames_df["multiview_agreement"].to_numpy(), "o-", label="vggt_omega")
axes[0].plot(sfm_frames["multiview_agreement"].to_numpy(), "s-", label="instantsfm")

# Pooled residual histograms on the shared bins
for label, hist in [("vggt_omega", report["depth_residual_histogram"]), ("instantsfm", sfm_tables["depth_residual_histogram"])]:
    axes[1].stairs(np.asarray(hist["counts"][: len(hist["bin_edges"]) - 1]), hist["bin_edges"], label=label)

# Parallax against error, one dot per pair direction
axes[2].scatter(pairs_df["median_parallax_deg"], pairs_df["median_rel_depth_error"], s=8, label="vggt_omega")
axes[2].scatter(pd.DataFrame(sfm_tables["depth_pairs"])["median_parallax_deg"], pd.DataFrame(sfm_tables["depth_pairs"])["median_rel_depth_error"], s=8, label="instantsfm")

for a, title in zip(axes, ["multiview agreement per frame", "relative depth residual", "pair parallax (deg) vs error"]):
    a.set_title(title)
    a.legend()

plt.tight_layout()"""),
    md("""## Reading it

- Agreement near 1 with low error: the views corroborate each other.
- A frame with low agreement is a frame the others do not see the same way — often a pose error.
- Large error at **low parallax** is expected noise (depth is ill-conditioned when the views
  barely moved); large error at high parallax is a real disagreement.

**Package gap:** nothing in `collab_splats` plots these tables yet; the cell above is the
stand-in until a `plot_reconstruction_quality` lands."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

The `a.set_title` / `a.legend` loop variable is `a`, so the cap regex counts only `axes[` (5) + `plt.` (2) = 7, which is under 10. The counts slice drops the NaN overflow bin, which `compute_reconstruction_quality` documents as one extra bin.

- [ ] **Step 2: Static gate**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k quality 2>&1 | tail -3
```
Expected: only `test_outputs_present` fails.

- [ ] **Step 3: Execute** (`<page>` = `02_pointcloud/reconstruction_quality_report.ipynb`)

Expected: exit 0. If `multiview_agreement` holds `None` entries (frames no other view sees), the `plot` call still works because pandas maps None to NaN. If `stairs` raises on a length mismatch, print `len(hist["counts"]), len(hist["bin_edges"])` and fix the slice to `len(edges) - 1`.

- [ ] **Step 4: Gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k quality 2>&1 | tail -3
git add docs/source/tutorials/02_pointcloud/reconstruction_quality_report.ipynb
git commit --only docs/source/tutorials/02_pointcloud/reconstruction_quality_report.ipynb -m "docs(tutorials): reconstruction_quality_report page"
```

---

### Task 10: Page `02_pointcloud/refinement` (GATED: authored, not executed)

**Files:**
- Rewrite: `docs/source/tutorials/02_pointcloud/refinement.ipynb`

- [ ] **Step 1: Build the notebook against today's API**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/02_pointcloud/refinement.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Refinement — bundle adjustment and loop closure

> **Pending.** This page is written against today's API and executed once joint RGB-D bundle
> adjustment (`feat/rgbd-ba`) and the loop-closure world-grid fix land.

A feedforward pass is a first guess. Two tools improve it, both **feedforward-only** — SfM
refuses them, because it already solves poses globally:

- **Bundle adjustment** tracks points across frames and jointly refines cameras to minimize
  reprojection error.
- **Loop closure** splits a long sequence into overlapping submaps, finds revisits by
  retrieval, and solves a pose graph so the trajectory closes on itself."""),
    code("""import dataclasses
from pathlib import Path

import numpy as np

from collab_splats.geometry.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig
from collab_splats.geometry.transforms import invert_poses
from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.utils import clean_pointcloud
from collab_splats.utils.visualization import create_camera_frustum_pyvista, pointcloud_to_polydata

# Budget: 24 frames over submaps of 8 gives three submaps; below submap_size LC silently runs plain inference
MAX_FRAMES = 24
SUBMAP_SIZE = 8  # production base.yaml: 20

# Keyframes through the pipeline
scene = tutorial_scene("refinement", preproc={"max_frames": MAX_FRAMES})
scene.run(stages=["preproc"])
WORK = Path(scene.config["output_path"])

# The first pass both refinements start from
creator = get_creator("vggt_omega")(clean=False)
first = creator.create_pointcloud(scene.images_dir, WORK / "first_pass")"""),
    md("""## §1 — Bundle adjustment

`refine` takes the model-grid arrays and returns refined extrinsics and model-grid intrinsics.
The refined result drops the stale full-resolution K and reprojects its points from the new
cameras — exactly what the `refine` stage does."""),
    code("""ba = BundleAdjustment(BundleAdjustmentConfig(query_frame_num=8))
extrinsics, model_K = ba.refine(
    first.images,
    first.confidence,
    first.world_points,
    first.extrinsics,
    first.model_intrinsics,
    image_paths=first.image_paths,
)

# Reproject from the refined cameras, then clean as the stage does
refined = dataclasses.replace(first, extrinsics=extrinsics, model_intrinsics=model_K, intrinsics=None).reproject()
refined = clean_pointcloud(refined, remove_outliers=True, max_points=200_000)
print("loss first -> last:", ba.loss_history[0], "->", ba.loss_history[-1])"""),
    code("""pl = pv.Plotter()

# Camera centers before (grey) and after (red) bundle adjustment
for color, res in [("grey", first), ("red", refined)]:
    for w2c in res.extrinsics:
        pl.add_mesh(create_camera_frustum_pyvista(w2c, scale=0.05), color=color, line_width=2)

pl.add_mesh(pointcloud_to_polydata(refined.points, rgb=refined.colors), scalars="rgb", rgb=True, point_size=2)
pl.show()"""),
    md("""## §2 — Loop closure

`LoopClosure` wraps a creator: same `create_pointcloud` call, but it runs the model per submap
and stitches the submaps through a pose graph. `lc_retrieval_threshold` gates which frame pairs
count as revisits."""),
    code("""lc_creator = LoopClosure(get_creator("vggt_omega")(clean=False), LoopClosureConfig(submap_size=SUBMAP_SIZE))
looped = lc_creator.create_pointcloud(scene.images_dir, WORK / "loop_closure")

# Camera centers per route; a closed loop brings the last cameras back to the first
centers = {name: invert_poses(r.extrinsics)[:, :3, 3] for name, r in [("first", first), ("looped", looped)]}
for name, c in centers.items():
    print(f"{name:7s} end-to-start gap: {np.linalg.norm(c[-1] - c[0]):.3f}")"""),
    code("""pl = pv.Plotter()

# Trajectories before (grey) and after (red) loop closure
for color, c in [("grey", centers["first"]), ("red", centers["looped"])]:
    pl.add_mesh(pv.lines_from_points(c), color=color, line_width=3)

pl.show()"""),
    md("""## In a pipeline run

- `pointcloud.bundle_adjustment: true` turns on the `refine` stage, which writes
  `colmap/refine.json` and rewrites `pointcloud.zarr`.
- `pointcloud.loop_closure: {submap_size: N}` wraps the pointcloud stage's creator.
- Both refuse an sfm backend at `Reconstructor.validate_config`."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Static gate only (outputs check skips)**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k refinement -rs 2>&1 | tail -4
```
Expected: pass, with `test_outputs_present[refinement]` SKIPPED and the reason `gated on feat/rgbd-ba + LC world-grid fix`. Do **not** execute.

- [ ] **Step 3: Commit**

```bash
cd $W && git commit --only docs/source/tutorials/02_pointcloud/refinement.ipynb \
  -m "docs(tutorials): refinement page authored (gated on feat/rgbd-ba + LC world-grid fix)"
```

---

### Task 11: Page `00_quickstart/pipeline` (write-now)

**Files:**
- Create: `docs/source/tutorials/00_quickstart/pipeline.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/00_quickstart/pipeline.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Quickstart — the whole pipeline

`Reconstructor` runs every stage from one config: video → keyframes → pointcloud → mesh, plus
a quality report. Every later page opens one stage up; this one runs them all."""),
    md("""## §1 — Config

A config is a few overrides deep-merged over `configs/base.yaml`. `tutorial_scene` adds
`input_path` (the tutorial clip) and `output_path` (a fresh tempdir). Semantics is switched off
here; it has its own pages."""),
    code("""import json
from pathlib import Path

import numpy as np
import yaml
from IPython.display import Image, display

from collab_splats.reconstructor import Reconstructor

# Budget: production base.yaml max_frames is 300
MAX_FRAMES = 16

scene = tutorial_scene("quickstart", preproc={"max_frames": MAX_FRAMES}, semantics={"enabled": False})

# Cross-field checks run before any stage; a bad pairing fails here, not an hour in
Reconstructor.validate_config(scene.config)
print(yaml.safe_dump({k: scene.config[k] for k in ("preproc", "pointcloud", "mesh")}, sort_keys=False)[:1500])"""),
    md("""## §2 — Run

A bare `run()` takes every stage the config enables. `preproc`, `pointcloud` and the
reconstruction quality report always run; `mesh` is on in base.yaml; `splats`, `semantics`,
`localize` and `refine` are off."""),
    code("scene.run()"),
    md("## §3 — What it wrote\n\n`outputs` maps each stage to its artifact; `done(stage)` is whether it exists."),
    code("""for stage, path in scene.outputs.items():
    print(f"{stage:32s} {'done' if scene.done(stage) else '-':5s} {path.relative_to(scene.config['output_path'])}")"""),
    code("""# The capture report preproc wrote beside images/, plotted with the kept frames marked
for png in sorted(scene.images_dir.parent.glob("*.png")):
    display(Image(str(png)))

quality = json.loads(scene.outputs["reconstruction_quality_report"].read_text())
print("median multiview agreement:", round(float(np.nanmedian(np.array(quality["frames"]["multiview_agreement"], dtype=float))), 3))"""),
    code("pv.read(scene.outputs[\"mesh\"]).plot(rgb=True)"),
    md("""## §4 — Named stages and re-runs

`run(stages=[...])` ignores the enable flags and runs exactly what you name, plus nothing
else — a dependency is met by this run or by its output on disk. A named **leaf** stage that is
already done refuses, so a finished mesh is never silently overwritten."""),
    code("""try:
    scene.run(stages=["mesh"])
except ValueError as err:
    print("refused:", err)

# Overrides change the next run; overwrite=True rebuilds the named stage
scene.config["mesh"]["voxel_size"] *= 2
scene.run(stages=["mesh"], overwrite=True)
print("coarser mesh:", scene.outputs["mesh"])"""),
    md("""## §5 — The same run from the shell

```bash
reconstruct local data/tutorial/tutorial_example-video.mp4 \\
    --output-root /tmp/scenes --config my_overrides.yaml
```

`reconstruct remote` does the same over the curated GCS scenes (needs credentials).

## Where each stage is explained

| Stage | Page |
|---|---|
| `preproc` | 01 · video_quality, keyframe_extraction |
| `pointcloud` | 02 · reconstruction |
| `reconstruction_quality_report` | 02 · reconstruction_quality_report |
| `refine` | 02 · refinement |
| `splats` | 03 · train_splats |
| `mesh` | 04 · tsdf_mesh, texturing |
| `semantics` | 05 · feature_extraction, segmentation, lifting_and_query, ocr_lens |
| `localize` | 06 · localization |"""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Static gate**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k pipeline 2>&1 | tail -3
```

- [ ] **Step 3: Verify the CLI line before executing**

```bash
cd $W && PYTHONPATH=$W $PY -m collab_splats local --help | head -20
```
Expected: the help shows positional inputs, `--output-root` and `--config`. If the flags differ, fix the §5 markdown to match exactly.

- [ ] **Step 4: Execute** (`<page>` = `00_quickstart/pipeline.ipynb`)

Expected: exit 0. If `mesh.voxel_size` is absent from `scene.config["mesh"]`, print `scene.config["mesh"]` and use the real key name.

- [ ] **Step 5: Gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k pipeline 2>&1 | tail -3
git add docs/source/tutorials/00_quickstart/pipeline.ipynb
git commit --only docs/source/tutorials/00_quickstart/pipeline.ipynb -m "docs(tutorials): quickstart pipeline page"
```

---

### Task 12: Page `03_splats/train_splats` (write-now)

**Files:**
- Rewrite: `docs/source/tutorials/03_splats/train_splats.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/03_splats/train_splats.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Training Splats

Gaussian splats turn a pointcloud and its cameras into a photorealistic, renderable scene. This
page trains **Scaffold-2DGS**: anchors with small MLPs that spawn surface-aligned 2D Gaussians —
the configuration whose renders also make the best mesh source (tsdf_mesh page)."""),
    code("""import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from collab_splats.pointcloud import PointcloudResult
from collab_splats.pointcloud.utils import frame_depths
from collab_splats.preproc import frames
from collab_splats.splats import SplatsConfig, load_checkpoint, render_views, train

# Budget: production base.yaml max_frames 300, splats.max_steps 30000
MAX_FRAMES = 16
MAX_STEPS = 1000

# Keyframes and the feedforward pointcloud through the pipeline
scene = tutorial_scene("train_splats", preproc={"max_frames": MAX_FRAMES})
scene.run(stages=["preproc", "pointcloud"])
result = scene.result"""),
    md("""## §1 — Config

Two scaffold rules: no `sh_degree` / `sh_degree_interval` (the MLP predicts color, so setting
either raises), and `opacity_reg` weight 0 (anchors manage opacity themselves). 2DGS swaps the
scale/opacity regularizers for a `distortion` loss. Pose optimization is on by default."""),
    code("""cfg = SplatsConfig.from_dict({
    "representation": "scaffold",
    "primitive": "2dgs",
    "max_steps": MAX_STEPS,
    "losses": {
        "depth": {"weight": 0.01},
        "normal_consistency": {"weight": 0.05, "start": MAX_STEPS // 4},
        "distortion": {"weight": 0.01, "start": MAX_STEPS // 10},
    },
})

# The vanilla 3DGS equivalent, for reference only
vanilla = SplatsConfig.from_dict({"representation": "vanilla", "primitive": "3dgs", "max_steps": MAX_STEPS})
print(cfg)"""),
    md("""## §2 — Train

`train` takes frames, world-to-camera poses, full-resolution K, and the seed points. Frames
stay on the CPU in the zarr's row order; one view at a time goes to the GPU. The depth loss
needs targets: `frame_depths` puts the feedforward depth on the frame grid, low-confidence
pixels zeroed (0 = no target)."""),
    code("""image_ids = [frames.frame_idx_from_path(p) for p in result.image_paths]
rgbs = frames.read_frames(scene.images_dir, image_ids)

# Depth targets from the full zarr (the light result carries no depth), masked as the stage does
ff = PointcloudResult.load_zarr(scene.pointcloud_zarr, load_images=False, load_world_points=False)
depth_targets = frame_depths(ff, rgbs, conf_percentile=scene.config["mesh"]["conf_percentile"])

out_dir = scene.outputs["splats"].parent
train(
    cfg, rgbs, result.extrinsics, result.intrinsics, result.points, result.colors, out_dir,
    depth_targets=depth_targets, image_ids=image_ids,
)
print(sorted(p.name for p in out_dir.iterdir()))"""),
    md("## §3 — Quality report\n\nPSNR / SSIM on the training views, written beside the checkpoint: `summary` holds the means,\nfinal losses and the config it trained with; `per_frame` has one row per view."),
    code("""splat_report = json.loads((out_dir / "splats_quality_report.json").read_text())
summary = splat_report["summary"]

# Headline scalars, final losses, then the per-view spread
print({k: summary[k] for k in ("psnr", "ssim", "n_gaussians", "seconds")})
print("final losses:", {k: round(v, 4) for k, v in summary["final_losses"].items()})
print("per-view PSNR:", [round(f["psnr"], 1) for f in splat_report["per_frame"]])"""),
    md("## §4 — Render it back\n\n`load_checkpoint` returns a render-only model with the pose-corrected cameras it trained."),
    code("""model, camera_opt, c2w, K, ids, (height, width) = load_checkpoint(scene.outputs["splats"], "cuda")

# Every fourth view, render beside the source frame
views = list(range(0, len(ids), 4))
renders = [r["rgb"][0].cpu().numpy() for i, r in enumerate(render_views(model, camera_opt, c2w, K, height, width)) if i in views]

# Portrait frames: narrow columns, source row over render row
fig, axes = plt.subplots(2, len(views), figsize=(1.8 * len(views), 6.5))
for col, (view, render) in enumerate(zip(views, renders)):
    axes[0, col].imshow(rgbs[view])
    axes[1, col].imshow(render)

for a in axes.flat:
    a.axis("off")

plt.tight_layout()"""),
    md("""## In a pipeline run

`splats: {enabled: true, representation: scaffold, primitive: 2dgs}` runs §2 as the `splats`
stage into `splats/ckpt.pt`, `splats.ply` and the quality report. At `MAX_STEPS = 1000` the
renders are soft; production trains 30000 steps."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

Matplotlib count: `plt.` ×3 (`import matplotlib.pyplot as plt` does not match `plt\.`, but `plt.subplots` and `plt.tight_layout` do) plus `axes[` ×2 gives 4.

- [ ] **Step 2: Static gate; verify the config is accepted**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k train_splats 2>&1 | tail -3
cd $W && PYTHONPATH=$W $PY -c "
from collab_splats.splats import SplatsConfig
SplatsConfig.from_dict({'representation':'scaffold','primitive':'2dgs','max_steps':1000,'losses':{'depth':{'weight':0.01},'normal_consistency':{'weight':0.05,'start':250},'distortion':{'weight':0.01,'start':100}}}); print('cfg OK')"
```
Expected: `cfg OK`. If `from_dict` rejects missing keys, base it on `scene.config["splats"]` minus `sh_degree` / `sh_degree_interval` / `enabled`, and say so in a code comment.

- [ ] **Step 3: Execute** (`<page>` = `03_splats/train_splats.ipynb`). Run alone; it trains on the GPU.

Expected: exit 0. If `render_views` wants tensors and `c2w` comes back as numpy, it comes from `load_checkpoint` as a `Tensor` per its signature, so no change is needed.

- [ ] **Step 4: Gate, size, commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k train_splats 2>&1 | tail -3
ls -la docs/source/tutorials/03_splats/train_splats.ipynb
git commit --only docs/source/tutorials/03_splats/train_splats.ipynb -m "docs(tutorials): train_splats page on clean/final (Scaffold-2DGS)"
```

---

### Task 13: Page `04_mesh/tsdf_mesh` (write-now)

**Files:**
- Rewrite: `docs/source/tutorials/04_mesh/tsdf_mesh.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/04_mesh/tsdf_mesh.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# TSDF Mesh

A truncated signed distance function (TSDF) fuses per-view depth maps into a voxel grid: each
voxel stores its signed distance to the nearest surface, averaged over the views that see it.
Marching cubes then extracts the zero crossing as a mesh.

Two numbers matter:

- **`voxel_size`** — the grid resolution.
- **`sdf_trunc`** — how far from a surface a view still writes. This, not `voxel_size`, sets the
  thinnest structure that survives: a surface thinner than the truncation band is cancelled when
  **both** of its sides are seen, and fattened when only one is."""),
    code("""from pathlib import Path

import numpy as np
import open3d as o3d

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh import clean_repair_mesh, create_tsdf_mesh, prepare_mesh
from collab_splats.pointcloud import PointcloudResult
from collab_splats.pointcloud.utils import frame_depths
from collab_splats.preproc import frames
from collab_splats.semantics import sky_masks
from collab_splats.splats.checkpoint import render_tsdf_inputs

# Budget: production base.yaml max_frames 300, splats.max_steps 30000
MAX_FRAMES = 16
MAX_STEPS = 1000

# Keyframes, pointcloud and a splat model through the pipeline
scene = tutorial_scene("tsdf_mesh", preproc={"max_frames": MAX_FRAMES},
                       splats={"representation": "scaffold", "primitive": "2dgs", "max_steps": MAX_STEPS})
scene.run(stages=["preproc", "pointcloud", "splats"])
mesh_cfg = scene.config["mesh"]
WORK = Path(scene.config["output_path"])"""),
    md("""## §1 — Depth source A: feedforward

`frame_depths` lifts the zarr's model-grid depth to frame resolution, masking pixels below the
`conf_percentile` confidence. Pair it with the **full-resolution** K."""),
    code("""ff = PointcloudResult.load_zarr(scene.pointcloud_zarr, load_images=False, load_world_points=False)
ids = [frames.frame_idx_from_path(p) for p in ff.image_paths]
rgbs = frames.read_frames(scene.images_dir, ids)
ff_depths = frame_depths(ff, rgbs, conf_percentile=mesh_cfg["conf_percentile"])
print("depth", ff_depths.shape, f"valid {np.count_nonzero(ff_depths) / ff_depths.size:.1%}")"""),
    md("""## §2 — Depth source B: splat renders

Splats render dense depth at frame resolution from the pose-corrected cameras they trained —
smoother than per-frame network depth, which is why `mesh.source: splats` usually wins."""),
    code("""sp_depths, sp_rgbs, sp_c2w, sp_K, sp_ids = render_tsdf_inputs(scene.outputs["splats"], scene.images_dir)
print("depth", sp_depths.shape, f"valid {np.count_nonzero(sp_depths) / sp_depths.size:.1%}")"""),
    md("""## §3 — Sky masking

Sky has no depth worth fusing — it seeds a backdrop and floaters. `sky_masks` segments it per
frame; zero the masked depth before fusion (`mesh.mask_sky: true` does this in the stage)."""),
    code("""sky = sky_masks(scene.images_dir, idxs=ids)
print(f"sky: {sky.mean():.1%} of pixels")
ff_depths = np.where(sky, 0.0, ff_depths)"""),
    md("## §4 — Fuse both\n\nSame grid, same truncation; only the depth source differs."),
    code("""meshes = {}
sdf_trunc = mesh_cfg["sdf_trunc_mult"] * mesh_cfg["voxel_size"]

# One fusion per depth source, each into its own directory
for name, (depths, colors, c2w, K) in {
    "feedforward": (ff_depths, rgbs, invert_poses(ff.extrinsics), ff.intrinsics),
    "splats": (sp_depths, sp_rgbs, sp_c2w, sp_K),
}.items():
    out = WORK / f"mesh_{name}"
    out.mkdir()
    meshes[name] = create_tsdf_mesh(depths, colors, c2w, K, out, voxel_size=mesh_cfg["voxel_size"],
                                    depth_trunc=mesh_cfg["depth_trunc"], sdf_trunc=sdf_trunc)

for name, path in meshes.items():
    m = o3d.io.read_triangle_mesh(str(path))
    print(f"{name:12s} {len(m.vertices):>9,} vertices  {len(m.triangles):>9,} triangles")"""),
    md("""## §5 — Clean and prepare

`clean_repair_mesh` removes floaters and fills holes with **scale-relative** thresholds
(fractions of the mesh's own extent), so one setting works across scene sizes. `prepare_mesh`
fills the remaining small holes, decimates and optionally smooths — the mesh the stage writes."""),
    code("""clean_repair_mesh(meshes["splats"], use_convex_hull=mesh_cfg["use_convex_hull"])
cleaned = o3d.io.read_triangle_mesh(str(meshes["splats"]))
prepared = prepare_mesh(cleaned, voxel_size=mesh_cfg["voxel_size"], smooth_iterations=mesh_cfg["smooth_iterations"])
print(f"cleaned {len(cleaned.triangles):,} -> prepared {len(prepared.triangles):,} triangles")"""),
    code("""pl = pv.Plotter(shape=(1, 2))

# Raw feedforward fusion left, cleaned splat fusion right
for col, (label, path) in enumerate([("feedforward, raw", meshes["feedforward"]), ("splats, cleaned", meshes["splats"])]):
    pl.subplot(0, col)
    pl.add_text(label, font_size=10)
    pl.add_mesh(pv.read(str(path)), rgb=True)

pl.link_views()
pl.show()"""),
    md("""## In a pipeline run

`scene.run(stages=["mesh"])` does §1 or §2 (by `mesh.source`), §3 when `mesh.mask_sky`, then
§4–§5 into `mesh.ply`. `mesh.texture: true` adds a UV-textured mesh — the texturing page."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Static gate**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k tsdf 2>&1 | tail -3
```

- [ ] **Step 3: Execute** (`<page>` = `04_mesh/tsdf_mesh.ipynb`). Run alone (splat train plus sky ONNX).

Expected: exit 0. Possible failures and fixes:
- `create_tsdf_mesh` writes a fixed filename, so separate `out` dirs are required. They are.
- `sky.shape != ff_depths.shape`: print both and stop. That is the stage's own `ValueError` condition and means the frames and depth disagree.

- [ ] **Step 4: Gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k tsdf 2>&1 | tail -3
git commit --only docs/source/tutorials/04_mesh/tsdf_mesh.ipynb -m "docs(tutorials): tsdf_mesh page on clean/final — feedforward vs splat depth, sky, clean"
```

---

### Task 14: Page `04_mesh/texturing` (GATED: authored, not executed)

> **2026-10-03 — un-gated.** The texturing decision landed on `clean/final` (`232bad46`, view-chart
> unwrap + per-view color gains). The page now bakes through the pipeline (`mesh.texture: true`),
> which passes the cleaned pre-decimation mesh as occluder; the code below used `mesh.ply` as its
> own occluder, which the gain solve would sample on filled patches. The notebook is the source of
> truth; `04_mesh/texturing.ipynb` leaves `GATED`.

**Files:**
- Create: `docs/source/tutorials/04_mesh/texturing.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/04_mesh/texturing.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Mesh Texturing

> **Pending.** Written against today's `create_texture_mesh`; executed once the texturing
> decision lands.

Vertex colors are limited by vertex density. A texture atlas decouples color from geometry:
the mesh is UV-unwrapped into charts, packed into one image, and each texel is filled by
projecting the source frames onto the surface — skipping views where something else occludes it."""),
    code("""from pathlib import Path

import open3d as o3d
from IPython.display import Image, display

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh import create_texture_mesh
from collab_splats.pointcloud import PointcloudResult
from collab_splats.preproc import frames

# Budget: production base.yaml max_frames 300; texture edge 8192
MAX_FRAMES = 16
TEX_SIZE = 4096

# Keyframes, pointcloud and an untextured feedforward mesh through the pipeline
scene = tutorial_scene("texturing", preproc={"max_frames": MAX_FRAMES}, mesh={"source": "feedforward", "texture": False})
scene.run(stages=["preproc", "pointcloud", "mesh"])
mesh_cfg = scene.config["mesh"]"""),
    md("""## §1 — Inputs

The views to project are the ones the mesh was fused from. The **occluder** is the cleaned,
undecimated mesh: visibility is tested against it, so a decimated surface never lets a view
paint through a wall."""),
    code("""ff = PointcloudResult.load_zarr(scene.pointcloud_zarr, load_images=False, load_world_points=False)
ids = [frames.frame_idx_from_path(p) for p in ff.image_paths]
rgbs = frames.read_frames(scene.images_dir, ids)
c2w = invert_poses(ff.extrinsics)

# mesh.ply is already prepared; it serves as both the surface and, here, the occluder
prepared = o3d.io.read_triangle_mesh(str(scene.outputs["mesh"]))
occluder = prepared
print(f"{len(prepared.triangles):,} triangles, {len(rgbs)} views")"""),
    md("## §2 — Bake\n\nUV-unwrap, pack and project in one call. Output is an OBJ with its PNG atlas."),
    code("""texture_dir = scene.backend_dir / "texture"
textured = create_texture_mesh(prepared, occluder, texture_dir, rgbs, c2w, ff.intrinsics,
                               voxel_size=mesh_cfg["voxel_size"], tex_size=TEX_SIZE)
print(textured, sorted(p.name for p in texture_dir.iterdir()))"""),
    md("""## §3 — The atlas

Grey texels are surface no camera saw. A **watertight** mesh pays for its sealed hull in atlas
area that no view can fill; an open mesh spends the atlas only on observed surface."""),
    code("""atlas = sorted(texture_dir.glob("*.png"))[0]
display(Image(str(atlas), width=600))"""),
    code("pv.read(str(textured)).plot(texture=pv.read_texture(str(atlas)))"),
    md("""## In a pipeline run

`mesh.texture: true` runs §2 inside the `mesh` stage, with the cleaned pre-decimation mesh as
the occluder, into `texture/`."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Static gate (outputs skip)**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k texturing -rs 2>&1 | tail -4
```
Expected: pass, with `test_outputs_present[texturing]` SKIPPED and the reason `gated on texturing decision`.

- [ ] **Step 3: Commit**

```bash
cd $W && git add docs/source/tutorials/04_mesh/texturing.ipynb
git commit --only docs/source/tutorials/04_mesh/texturing.ipynb -m "docs(tutorials): texturing page authored (gated on texturing decision)"
```

---

### Task 15: Page `05_semantics/feature_extraction` (write-now)

**Files:**
- Rewrite: `docs/source/tutorials/05_semantics/feature_extraction.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/05_semantics/feature_extraction.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Feature Extraction

A feature extractor turns an image into a grid of patch embeddings. **Queryable** extractors
(MaskCLIP, Talk2DINO) embed text in the same space, so a word scores every patch. DINOv2
(not queryable) and the OCR lens (its own page) share the same interface."""),
    code("""import matplotlib.pyplot as plt
import numpy as np

from collab_splats.semantics import MaskCLIPExtractor, Talk2DinoExtractor
from collab_splats.utils.io import read_image
from collab_splats.utils.notebook import feature_viz_row

# One committed frame, no pipeline needed
frame = np.asarray(read_image(QUERY_IMAGE))
QUERY = "tree"
print(frame.shape)"""),
    md("""## §1 — Forward and score

`forward` takes a list of images and returns one `(D, H_p, W_p)` map per image.
`score_queries` reduces over its positives, so call it **once per prompt** to compare prompts."""),
    code("""extractors = {"MaskCLIP": MaskCLIPExtractor(), "Talk2DINO": Talk2DinoExtractor()}
maps, scores = {}, {}

# Patch features and one similarity map per extractor
for name, ex in extractors.items():
    maps[name] = ex.forward([frame])[0]
    scores[name] = ex.score_queries(maps[name], positive=[QUERY]).squeeze().cpu().numpy()
    print(f"{name:10s} features {tuple(maps[name].shape)}")"""),
    md("## §2 — PCA, heatmap, masked\n\n`feature_viz_row` draws one row per extractor: PCA of the features, the query heatmap, and the image masked by it."),
    code("""fig, axes = plt.subplots(len(extractors), 3, figsize=(12, 4 * len(extractors)))

for row, name in enumerate(extractors):
    feature_viz_row(axes[row], frame, maps[name], scores[name], title_prefix=f"{name} · ", query_label=QUERY)

plt.tight_layout()"""),
    md("""## In a pipeline run

The `semantics` stage runs one extractor (`semantics.extractor`) over every keyframe into a
per-scene cache, then lifts it onto the mesh — the lifting_and_query page."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Check `read_image` and `feature_viz_row`'s axes count**

```bash
cd $W && PYTHONPATH=$W $PY -c "
import inspect; from collab_splats.utils.io import read_image; from collab_splats.utils import notebook
print(inspect.signature(read_image)); print(inspect.getsource(notebook.feature_viz_row)[:1500])"
```
Expected: `read_image(path)` and the number of axes `feature_viz_row` fills. Set the `plt.subplots(..., N, ...)` column count to that number (the cell assumes 4). If `read_image` returns a PIL image, the `np.asarray` already handles it.

- [ ] **Step 3: Execute** (`<page>` = `05_semantics/feature_extraction.ipynb`)

Expected: exit 0. Talk2DINO's weights must be cached locally. If the download fails offline, stop and report it: do not drop the extractor silently.

- [ ] **Step 4: Gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k feature_extraction 2>&1 | tail -3
git commit --only docs/source/tutorials/05_semantics/feature_extraction.ipynb -m "docs(tutorials): feature_extraction page on clean/final"
```

---

### Task 16: Page `05_semantics/segmentation` (write-now)

**Files:**
- Rewrite: `docs/source/tutorials/05_semantics/segmentation.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/05_semantics/segmentation.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Segmentation

Segmenters split an image into masks. Paired with a feature map, masks turn noisy patch
features into one clean feature per object. `MobileSAMSegmentation` is shown; SAM3 (text
prompts), INSID3 (in-context, from reference masks) and SkyWater (sky masks for the mesh
stage) share the `BaseSegmentation` interface."""),
    code("""import matplotlib.pyplot as plt
import numpy as np

from collab_splats.semantics import MaskCLIPExtractor, MobileSAMSegmentation, aggregate_masked_features
from collab_splats.utils.image import open_image, resize_image
from collab_splats.utils.visualization import overlay_masks, pca_to_rgb

# One committed frame at 1024 px; SAM's auto mode upsamples every candidate mask to full size
image = open_image(QUERY_IMAGE)
image = resize_image(image, 1024)
frame = np.asarray(image.convert("RGB"))
H, W = frame.shape[:2]"""),
    md("""## §1 — Two strategies

- `object`: detect objects, then one mask per box — fewer, whole-object masks.
- `auto`: a dense point grid — every region, including background."""),
    code("""masks = {}

# One segmentation per strategy
for strategy in ("object", "auto"):
    masks[strategy], _ = MobileSAMSegmentation(strategy=strategy, device="cuda").segment(frame)
    print(f"{strategy:6s} {len(masks[strategy])} masks")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for a, strategy in zip(axes, masks):
    a.imshow(overlay_masks(frame, masks[strategy]))
    a.set_title(strategy)

plt.tight_layout()"""),
    md("""## §2 — Mask-pooled features

`aggregate_masked_features` averages the patch features inside each mask and paints the mean
back — a feature map that is constant per object."""),
    code("""features = MaskCLIPExtractor().forward([frame])[0]
pooled = aggregate_masked_features(features, masks["object"], resolution=tuple(features.shape[1:]), final_resolution=(H, W))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(pca_to_rgb(features, frame))
axes[1].imshow(pca_to_rgb(pooled, frame))
plt.tight_layout()"""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

Matplotlib count: `plt.subplots` ×2 and `plt.tight_layout` ×2 give 4 `plt.`, plus `axes[` ×2, giving 6 under the cap. The `a.` calls do not match (`\bax\.`).

- [ ] **Step 2: Execute** (`<page>` = `05_semantics/segmentation.ipynb`)

Expected: exit 0. If `aggregate_masked_features` expects `masks` at feature resolution or a different dtype, read its docstring (`$PY -c "from collab_splats.semantics import aggregate_masked_features as f; print(f.__doc__)"`) and adapt only the call arguments.

- [ ] **Step 3: Gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k segmentation 2>&1 | tail -3
git commit --only docs/source/tutorials/05_semantics/segmentation.ipynb -m "docs(tutorials): segmentation page on clean/final"
```

---

### Task 17: Page `05_semantics/lifting_and_query` (write-now)

**Files:**
- Rewrite: `docs/source/tutorials/05_semantics/lifting_and_query.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/05_semantics/lifting_and_query.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Lifting and Query

2D features become 3D: every point (or mesh vertex) projects into every frame, keeps the views
where its depth agrees with the frame's depth, and averages their features weighted by
confidence. A point no view sees stays **zero** — unobserved, never inpainted."""),
    code("""import dataclasses
from pathlib import Path

import numpy as np
import torch
import trimesh
import zarr

from collab_splats.pointcloud import PointcloudResult
from collab_splats.reconstructor import store_rows
from collab_splats.semantics import FeatureAutoencoder, MaskCLIPExtractor, extract_feature_cache, read_point_features, write_point_features
from collab_splats.semantics.lifting import lift_features, transfer_features

# Budget: production base.yaml max_frames is 300
MAX_FRAMES = 16
QUERY = "tree"

# Keyframes, pointcloud and a feedforward mesh through the pipeline; no splat train
scene = tutorial_scene("lifting", preproc={"max_frames": MAX_FRAMES}, mesh={"source": "feedforward"})
scene.run(stages=["preproc", "pointcloud", "mesh"])
cloud = PointcloudResult.load_zarr(scene.pointcloud_zarr, load_world_points=False)"""),
    md("""## §1 — Per-frame feature cache

`extract_feature_cache` runs an extractor over every keyframe once into a zarr; `store_rows`
maps the pointcloud's frames to that cache's rows (SfM may have dropped some)."""),
    code("""extractor = MaskCLIPExtractor()
cache = extract_feature_cache(extractor, scene.images_dir, scene.semantics_cache_dir)
features = zarr.open(str(cache), mode="r")["features"]
rows = store_rows(scene.images_dir, cloud.image_paths)
# Cache is fp16 on disk; the autoencoder and lift run in fp32
maps = [torch.from_numpy(features[r]).float() for r in rows]
print(f"{len(maps)} maps of {tuple(maps[0].shape)}")"""),
    md("""## §2 — Compress

ViT features are 768-D; a small autoencoder compresses them for storage. Fit on a sample of
patches; `target_cosine` stops early once reconstructions are close enough."""),
    code("""samples = torch.cat([m.flatten(1).T for m in maps])
ae = FeatureAutoencoder(input_dim=samples.shape[1], latent_dim=64)
ae.fit(samples[torch.randperm(len(samples))[:50_000]], epochs=10, target_cosine=0.95)"""),
    md("## §3 — Lift onto points\n\n`lift_features` takes a per-frame loader (`maps.__getitem__`) and the result."),
    code("""point_features = lift_features(maps.__getitem__, cloud)
observed = point_features.abs().sum(1) > 0
print(f"{observed.float().mean():.1%} of {len(point_features):,} points observed")

# Compressed per-point store, read back
store = scene.backend_dir / "semantics" / "maskclip_points.zarr"
write_point_features(store, ae.per_point_encode(point_features).detach().numpy(), ae)
print("stored", read_point_features(store).shape)"""),
    md("""## §4 — Lift onto mesh vertices

Vertices have no source pixel, so `pixel_indices=None` turns off the fallback: an unseen
vertex stays zero. Lift in chunks to bound memory."""),
    code("""mesh = trimesh.load(scene.outputs["mesh"], process=False)
vertices = np.asarray(mesh.vertices, dtype=np.float32)
colors = np.asarray(mesh.visual.vertex_colors[:, :3])

# Chunked vertex lift; the same loop as docs/examples/ocr_lens_viewer.py
blocks = []
for start in range(0, len(vertices), 131_072):
    part = dataclasses.replace(cloud, points=vertices[start:start + 131_072], colors=colors[start:start + 131_072], pixel_indices=None)
    blocks.append(lift_features(maps.__getitem__, part))
vertex_features = torch.cat(blocks)
print(f"{(vertex_features.abs().sum(1) > 0).float().mean():.1%} of {len(vertices):,} vertices observed")"""),
    md("""## §5 — Query

Score every vertex against a prompt, smooth over neighbours with `transfer_features`
(a Gaussian-weighted k-NN scatter), and color the mesh."""),
    code("""score = extractor.score_queries(vertex_features, positive=[QUERY]).cpu().numpy()
smoothed = transfer_features(vertices, vertices, score[:, None], k=8, max_dist=0.05)[:, 0]

pl = pv.Plotter(shape=(1, 2))
for col, (label, values) in enumerate([("raw", score), ("smoothed", smoothed)]):
    pl.subplot(0, col)
    pl.add_text(f"{QUERY!r} — {label}", font_size=10)
    pl.add_mesh(pv.read(str(scene.outputs["mesh"])), scalars=values, cmap="turbo")

pl.link_views()
pl.show()"""),
    md("""## In a pipeline run

`semantics: {enabled: true, extractor: maskclip}` runs §1–§4 as the `semantics` stage into
`semantics/maskclip_lifted.zarr`. **Package gap:** the chunked vertex lift in §4 lives only in
the example script."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Check the store path convention and `score_queries` input shape**

```bash
cd $W && PYTHONPATH=$W $PY -c "
from collab_splats.semantics import MaskCLIPExtractor as M; print(M.score_queries.__doc__)"
cd $W && sed -n '/def semantics/,/def mesh/p' collab_splats/reconstructor.py | grep -n "write_point_features\|_lifted\|score\|store"
```
Expected: `score_queries` takes `(D, H, W)` (so the `(D, V, 1)` reshape is right) or `(N, D)`. If it takes `(N, D)`, pass `vertex_features` directly. Match the store filename to the stage's `{extractor}_lifted.zarr` only if `write_point_features` requires that exact name.

- [ ] **Step 3: Execute** (`<page>` = `05_semantics/lifting_and_query.ipynb`)

Expected: exit 0. If the mesh has no vertex colors (`mesh.visual.kind != "vertex"`), use `np.full((len(vertices), 3), 200, np.uint8)` as the example does.

- [ ] **Step 4: Gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k lifting 2>&1 | tail -3
git commit --only docs/source/tutorials/05_semantics/lifting_and_query.ipynb -m "docs(tutorials): lifting_and_query page — points and mesh vertices (decision 021)"
```

---

### Task 18: Page `05_semantics/ocr_lens` (write-now)

**Files:**
- Create: `docs/source/tutorials/05_semantics/ocr_lens.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/05_semantics/ocr_lens.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# OCR Lens — words without a query

CLIP-style features need a prompt. The OCR lens reads a vision-language model's (LLaVA-1.6)
own hidden states through the attention heads that do OCR, then decodes each patch through the
language head into a distribution over **words**. Every pixel gets words with no query at all.

Needs the model cached locally; the page sets `HF_HUB_OFFLINE=1`."""),
    code("""import dataclasses
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import trimesh
import zarr

from collab_splats.pointcloud import PointcloudResult
from collab_splats.reconstructor import store_rows
from collab_splats.semantics import extract_feature_cache
from collab_splats.semantics.features.ocr_lens import OCRLensExtractor, load_decoder, load_processor, word_probabilities, word_vocabulary
from collab_splats.semantics.lifting import lift_features

# Budget: production base.yaml max_frames is 300
MAX_FRAMES = 16
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"

# Keyframes, pointcloud and a feedforward mesh through the pipeline
scene = tutorial_scene("ocr_lens", preproc={"max_frames": MAX_FRAMES}, mesh={"source": "feedforward"})
scene.run(stages=["preproc", "pointcloud", "mesh"])
cloud = PointcloudResult.load_zarr(scene.pointcloud_zarr, load_world_points=False, load_pixel_indices=False)"""),
    md("## §1 — Lens features\n\nThe expensive cell: a 7B model over every keyframe, cached once per scene."),
    code("""extractor = OCRLensExtractor(model_id=MODEL_ID)
cache = extract_feature_cache(extractor, scene.images_dir, scene.semantics_cache_dir)
features = zarr.open(str(cache), mode="r")["features"]
rows = store_rows(scene.images_dir, cloud.image_paths)

# Free the vision tower before loading the decoder
del extractor
torch.cuda.empty_cache()"""),
    md("""## §2 — Decode per frame, then lift

The decoder ends in an RMSNorm, so decoding is **non-linear**: averaging embeddings across
views and then decoding is not averaging words. Decode each frame's patches to word
probabilities first, then lift the probabilities."""),
    code("""vocab = word_vocabulary(load_processor(MODEL_ID).tokenizer)
decoder = load_decoder(MODEL_ID)
maps = []

# Each frame's patches decoded to (n_words, H_p, W_p) probabilities
for row in rows:
    fmap = torch.from_numpy(features[row])
    dim, height, width = fmap.shape
    probs = torch.cat([p.half().cpu() for p, _ in word_probabilities(fmap.reshape(dim, -1).T, decoder, vocab)])
    maps.append(probs.T.reshape(-1, height, width))

del decoder
torch.cuda.empty_cache()
print(f"{len(maps)} frames x {len(vocab.words):,} words")"""),
    code("""mesh = trimesh.load(scene.outputs["mesh"], process=False)
vertices = np.asarray(mesh.vertices, dtype=np.float32)
colors = np.asarray(mesh.visual.vertex_colors[:, :3])

# Chunked vertex lift; unseen vertices stay zero
blocks = []
for start in range(0, len(vertices), 131_072):
    part = dataclasses.replace(cloud, points=vertices[start:start + 131_072], colors=colors[start:start + 131_072], pixel_indices=None)
    blocks.append(lift_features(maps.__getitem__, part).half())
words = torch.cat(blocks)
observed = words.float().sum(1) > 0
print(f"{observed.float().mean():.1%} of {len(vertices):,} vertices observed")"""),
    md("## §3 — Words on the mesh\n\nThe scene's most common top-1 words, and a probe: color the mesh by one word's probability."),
    code("""top1 = words[observed].float().argmax(1)
counts = torch.bincount(top1, minlength=len(vocab.words))
for row in counts.argsort(descending=True)[:10].tolist():
    print(f"{vocab.words[row]:15s} {counts[row].item():>8,} vertices")

probe = vocab.words[counts.argmax().item()]
values = words[:, vocab.words.index(probe)].float().numpy()
pv.read(str(scene.outputs["mesh"])).plot(scalars=values, cmap="magma", text=f"p({probe!r})")"""),
    md("""## Interactive viewer

`python docs/examples/ocr_lens_viewer.py <backend_dir>` serves the same lift in a browser:
top-1 labels, a word query, a click probe and neighbour smoothing.
**Package gap:** the per-frame decode loop (§2) and the chunked vertex lift live only in that
example script."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Check that the cache uses the extractor's registered name, and that the model is cached**

```bash
ls ~/.cache/huggingface/hub | grep -i llava
cd $W && PYTHONPATH=$W $PY -c "
from collab_splats.semantics.features.ocr_lens import WordVocab; import dataclasses; print([f.name for f in dataclasses.fields(WordVocab)])"
```
Expected: a `models--llava-hf--llava-v1.6-vicuna-7b-hf` directory, and `WordVocab` has `words`. If `words` is a tuple rather than a list, `.index` still works. If the model is not cached, stop and report it: the page cannot execute offline.

- [ ] **Step 3: Static gate, which now covers all 14 pages**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q 2>&1 | tail -5
```
Expected: `test_notebook_set_is_the_fourteen_pages` passes now. `test_outputs_present` fails only for pages not yet executed (ocr_lens, localization).

- [ ] **Step 4: Execute** (`<page>` = `05_semantics/ocr_lens.ipynb`). Run alone; the 7B model nearly fills the A40.

Expected: exit 0. If it OOMs in §2, lower `word_probabilities(..., chunk=1024)` and re-execute.

- [ ] **Step 5: Gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs/test_tutorial_contract.py -q -k ocr_lens 2>&1 | tail -3
git add docs/source/tutorials/05_semantics/ocr_lens.ipynb
git commit --only docs/source/tutorials/05_semantics/ocr_lens.ipynb -m "docs(tutorials): ocr_lens page — per-frame decode then lift onto mesh"
```

---

### Task 19: Page `06_localization/localization` (write-now)

**Files:**
- Rewrite: `docs/source/tutorials/06_localization/localization.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd $W && $PY - <<'EOF'
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
OUT = Path("docs/source/tutorials/06_localization/localization.ipynb")
cells = [
    code("%load_ext autoreload\n%autoreload 2"),
    code("%run ../tutorial.py"),
    md("""# Localization

Given a reconstruction and a new photo, where was the camera? Three stages:

1. **Retrieve** the keyframes that look most like the query (global descriptors).
2. **Match** local features between the query and each retrieved frame (LoMa here).
3. **Solve** PnP + RANSAC from the matched pixels' 3D points.

The query's intrinsics come from its image proportions when none are given."""),
    code("""from pathlib import Path

import numpy as np

from collab_splats.localization import CameraLocalizer, LocalMatcher, correspondences_for_ref, plot_correspondences, plot_inlier_distribution
from collab_splats.pointcloud import PointcloudResult
from collab_splats.preproc import frames
from collab_splats.utils.io import read_image
from collab_splats.utils.visualization import create_camera_frustum_pyvista, pointcloud_to_polydata

# Budget: production base.yaml max_frames is 300
MAX_FRAMES = 16

# Keyframes and the feedforward pointcloud through the pipeline
scene = tutorial_scene("localization", preproc={"max_frames": MAX_FRAMES})
scene.run(stages=["preproc", "pointcloud"])
result = scene.result"""),
    md("## §1 — Build the localizer\n\nOne retrieval descriptor and one set of local features per keyframe, from the result's own frames and world points."),
    code("""# The localizer needs model-res frames and dense world points, which scene.result leaves on disk
dense = PointcloudResult.load_zarr(scene.pointcloud_zarr, load_images=True, load_world_points=True)

# Keyframe images in result order, then the feature database
ids = [frames.frame_idx_from_path(p) for p in result.image_paths]
images = frames.read_frames(scene.images_dir, ids)
loc = CameraLocalizer.from_feedforward(dense, images=images, ids=[str(i) for i in ids], extractor=LocalMatcher("loma"))"""),
    md("## §2 — Localize one photo"),
    code("""query = np.asarray(read_image(QUERY_IMAGE))
pose = loc.localize(query)
print(f"{pose.n_inliers} / {pose.n_correspondences} correspondences are RANSAC inliers")"""),
    md("## §3 — Why it worked\n\nThe matches against the best reference frame, inliers highlighted, and which frames contributed inliers."),
    code("""ref = int(np.bincount(pose.ref_frame_indices[pose.inlier_mask]).argmax())
query_px, ref_px, inliers = correspondences_for_ref(pose, ref, ref_image_hw=images[ref].shape[:2])
plot_correspondences(query, images[ref], query_px, ref_px, inlier_mask=inliers)
plot_inlier_distribution(pose.ref_frame_indices, pose.inlier_mask, n_frames=len(ids))"""),
    md("## §4 — The pose in the scene\n\nKeyframe cameras grey, the query red."),
    code("""pl = pv.Plotter()
pl.add_mesh(pointcloud_to_polydata(result.points, rgb=result.colors), scalars="rgb", rgb=True, point_size=2)

# Keyframe frustums, then the localized query
for w2c in result.extrinsics:
    pl.add_mesh(create_camera_frustum_pyvista(w2c, scale=0.05), color="grey")
pl.add_mesh(create_camera_frustum_pyvista(pose.pose, scale=0.08), color="red", line_width=3)

pl.show()"""),
    md("""## In a pipeline run

`localization: {enabled: true, matcher: loma}` runs the `localize` stage, caching each
keyframe's local features in `pointcloud.zarr` so a later `CameraLocalizer` starts warm. Other
matchers from the vismatch zoo (`xfeat`, `superpoint-lightglue`, …) are a config string away."""),
]
meta = {"kernelspec": {"name": "python3", "display_name": "reconstruction", "language": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), OUT)
EOF
```

- [ ] **Step 2: Check `LocalizationResult`'s field names and `correspondences_for_ref`'s return**

```bash
cd $W && PYTHONPATH=$W $PY -c "
import dataclasses, inspect
from collab_splats.localization import localizer, correspondences_for_ref
print([f.name for f in dataclasses.fields(localizer.LocalizationResult)])
print(correspondences_for_ref.__doc__)"
```
Expected: fields include a pose (`extrinsics` or similar), `ref_frame_indices` and `inlier_mask`. Rename the attribute accesses in §3–§4 to the real field names, and unpack `correspondences_for_ref`'s return to match its docstring. Change nothing else.

- [ ] **Step 3: Execute** (`<page>` = `06_localization/localization.ipynb`)

Expected: exit 0.

- [ ] **Step 4: Full gate and commit**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs -q -rs 2>&1 | tail -8
git commit --only docs/source/tutorials/06_localization/localization.ipynb -m "docs(tutorials): localization page on clean/final"
```
Expected: all pass, with exactly two skips (`refinement`, `texturing`) and their blocker reasons.

---

### Task 20: Docs build, size budget, report

**Files:**
- Modify (only if the build warns about it): `docs/source/tutorials/index.rst`

- [ ] **Step 1: Sphinx build**

```bash
cd $W/docs && PYTHONPATH=$W /opt/venv/reconstruction/bin/python -m sphinx -b html source _build/html 2>&1 | grep -E "WARNING|ERROR|build succeeded" | head -30
```
Expected: `build succeeded`, with no warning that names a tutorial page (`document isn't included in any toctree` or `toctree contains reference to nonexisting document`). Fix any such warning in `index.rst`.

- [ ] **Step 2: Size budget**

```bash
cd $W && du -ch docs/source/tutorials/*/*.ipynb | tail -1
```
Expected: 32 MB total or less. If over, re-execute the largest page with fewer gallery views or lower-DPI plots and re-commit it.

- [ ] **Step 3: Full test gates**

```bash
cd $W && PYTHONPATH=$W $PY -m pytest tests/docs tests/test_import_style.py -q -rs 2>&1 | tail -6
```
Expected: green, with exactly two skips.

- [ ] **Step 4: Graph update and changelog-free status**

```bash
cd $W && graphify update . 2>&1 | tail -2
```
`clean/tutorials` is not merged yet, so do not append to `CHANGELOG.md` and do not remove the CLAUDE.md in-flight entry. Both happen when the branch lands, after the two gated pages are done.

- [ ] **Step 5: Report to the user**

Report:
1. The 12 executed pages, with their commits.
2. The 2 gated pages and their blockers (`feat/rgbd-ba` + LC world-grid fix; the texturing decision). Each is done once its blocker lands on `clean/final`: rebase, re-run `$SCRATCH/api_check.py`, execute, commit.
3. The three package gaps, for a decision: `plot_reconstruction_quality`, the chunked vertex lift, and per-frame word-probability maps.
4. Any cell that Steps 2–3 of a page task had to adapt, with the reason.
