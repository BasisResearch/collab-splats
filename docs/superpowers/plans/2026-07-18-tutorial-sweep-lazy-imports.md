# Tutorial Notebook Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align all remaining tutorial notebooks to the `FRAMES`/`TUTORIAL_CACHE` layout with VGGT-Omega as the sole executed backend and LoMa as the primary localization matcher, simplifying each notebook in the process.

**Architecture:** Notebook-only changes — no library code modified. Notebooks are reworked and executed in pipeline order against `2024_02_06/C0043`; `feedforward_methods` produces the shared omega zarr cache all downstream notebooks read.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), jupyter nbconvert (tmux for heavy runs), zarr v3, VGGT-Omega, LoMa.

**Spec:** `docs/superpowers/specs/2026-07-18-tutorial-sweep-lazy-imports-design.md`

**Scope revision (2026-07-18):** the library lazy-import pass (former Phase 1) is DEFERRED by user decision. The ~55 s first-import cost stands until that pass runs; the spec's Phase 1 section is its reference design. Nothing in this plan touches `collab_splats/`.

**Shared-branch caution:** the working tree carries uncommitted edits from the in-flight feedforward-mesh session (`collab_splats/mesh/*`, `tests/mesh/*`, `02_pointcloud/feedforward_mesh.ipynb`, `02_pointcloud/feedforward_methods.ipynb`, `README.md`). NEVER `git add -A` / `git add .`. Stage only the exact notebook each task touches. Before editing `feedforward_methods.ipynb` (Task 1), diff the working-tree version against HEAD and preserve any feedforward-mesh hunks.

---

**Common conventions for every task:**
- Config comes from `%run ../tutorial_config.py`: read frames from `FRAMES`, write ALL notebook output under `TUTORIAL_CACHE` (canonical scene dir is rclone-synced and read-only for tutorials).
- The shared omega cache is `TUTORIAL_CACHE / "vggt_omega.zarr"` — `feedforward_methods` produces it; every downstream notebook loads it with a fail-loud cell (pattern below).
- Simplify while touching: delete dead cells, collapse redundant setup, trim imports to what the notebook actually uses, keep block comments per code style.
- Execute headless in tmux (46.6 GB cgroup cap; never run two heavy notebooks in parallel):
  ```bash
  tmux new-session -d -s nb 'source /opt/venv/reconstruction/bin/activate && \
    jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 <notebook-path> 2>&1 | tee /tmp/nb.log'
  ```
  Poll with `tmux capture-pane -pt nb | tail -20`; done when nbconvert exits 0.
- Downstream loader cell (used verbatim wherever a notebook consumes the reconstruction):
  ```python
  # Load the VGGT-Omega reconstruction produced by 02_pointcloud/feedforward_methods.ipynb
  from collab_splats.pointcloud.feedforward import FeedforwardResult

  _omega_cache = TUTORIAL_CACHE / "vggt_omega.zarr"
  assert _omega_cache.exists(), (
      f"missing {_omega_cache} — run 02_pointcloud/feedforward_methods.ipynb first"
  )
  result = FeedforwardResult.load_zarr(_omega_cache, load_images=True)
  print(f"loaded {result.points.shape[0]:,} points, {len(result.image_paths)} frames")
  ```
- Commit per notebook, staging ONLY that notebook (shared-branch caution above).

### Task 1: `feedforward_methods.ipynb` — omega-only

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`

- [ ] **Step 1: Reconcile working-tree state**

Run: `git diff docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb | head -100`
Understand what the feedforward-mesh session changed. Keep those semantics (do not revert their cells); rework on top of the working-tree version.

- [ ] **Step 2: Rework the notebook**

Structure (edit via NotebookEdit, cell by cell):
1. **Title markdown** — rewrite: notebook runs **VGGT-Omega** on the pipeline keyframes and writes the zarr cache all downstream notebooks (`semantic_lifting`, `localization`, `bundle_adjustment`) read. Prerequisite: keyframes present in `FRAMES` (pipeline-written `frames/`, or run tutorial 01). Remove `images/` wording.
2. **Setup code** — keep `%load_ext autoreload`; imports trimmed to what the omega-only flow uses (no open3d unless a kept cell uses it); `%run ../tutorial_config.py`.
3. **Frame list cell**:
   ```python
   # Pipeline keyframes: read-only input from the canonical scene dir
   image_paths = sorted(FRAMES.glob("*.jpg")) + sorted(FRAMES.glob("*.png"))
   assert image_paths, f"no keyframes in {FRAMES} — run the dashboard pipeline or tutorial 01"
   image_paths = image_paths[:MAX_FRAMES]
   print(f"{len(image_paths)} keyframes from {FRAMES}")
   ```
4. **Run/reload cell** (keep the existing omega cache-or-run pattern, retargeted at `TUTORIAL_CACHE`):
   ```python
   # Run VGGT-Omega once; later executions reload the zarr cache
   from collab_splats.pointcloud import make_creator
   from collab_splats.pointcloud.feedforward import FeedforwardResult

   _omega_cache = TUTORIAL_CACHE / "vggt_omega.zarr"
   _omega_cache.parent.mkdir(parents=True, exist_ok=True)
   if _omega_cache.exists():
       result = FeedforwardResult.load_zarr(_omega_cache)
       print(f"loaded cached reconstruction ({result.points.shape[0]:,} pts)")
   else:
       creator = make_creator("vggt_omega")
       result = creator.run(image_paths)
       result.save_zarr(_omega_cache)
       print(f"reconstructed {result.points.shape[0]:,} pts → {_omega_cache}")
   ```
   (Match the exact creator-run call signature used by the current notebook's omega cell — reuse its code, only retarget paths.)
5. **Visualization cells** — keep the pyvista pointcloud + frustum rendering for the omega result only; delete VGGT-X and MapAnything run/compare cells.
6. **"Other backends" markdown** (replaces the deleted comparison):
   ```markdown
   ## Other backends

   `make_creator(<name>)` swaps the reconstruction backend with no other code changes:

   | name | model | notes |
   |------|-------|-------|
   | `"vggtx"` | VGGT-X | fastest baseline |
   | `"mapanything"` | MapAnything | metric-scale output |
   | `"vggt_spark"` | VGGT-SPARK | optional install |
   | `"colmap"` / `"hloc"` | classical SfM | no GPU model, slower |

   Optional backends require `setup/feedforward.sh` with the matching submodule.
   See the [API docs](../../api/pointcloud.rst) for per-backend parameters.
   ```

- [ ] **Step 3: Execute headless (tmux, pattern above)**

Expected: exits 0; zarr exists at `/workspace/outputs/tutorial_cache/2024_02_06/C0043/vggt_omega.zarr`.

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(tutorials): feedforward_methods — VGGT-Omega only, FRAMES/TUTORIAL_CACHE layout"
```

### Task 2: `bundle_adjustment.ipynb` — omega backbone + layout

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`

- [ ] **Step 1: Rework** — replace `IMAGES` references with `FRAMES`; reconstruction input = the shared omega loader cell (verbatim from the preamble); primary backbone `vggt_omega`; BA outputs under `TUTORIAL_CACHE`; markdown notes other backbones work via `make_creator`.
- [ ] **Step 2: Execute headless (tmux)** — exits 0.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
git commit -m "docs(tutorials): bundle_adjustment on VGGT-Omega + new layout"
```

### Task 3: `slam_loop_closure.ipynb` — omega backbone + LC calibration

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`

- [ ] **Step 1: Rework** — same layout changes; primary backbone omega with its calibrated LC settings (`target_layer=13`, `similarity_threshold=1.55` — match the exact LoopClosureConfig field names used in the current notebook cells); markdown table of per-backbone calibrations (spark 0.95 native, vggtx L10/1.17, omega L13/1.55, mapanything L4/1.46).
- [ ] **Step 2: Execute headless (tmux)** — exits 0.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
git commit -m "docs(tutorials): slam_loop_closure on VGGT-Omega (L13/1.55) + new layout"
```

### Task 4: `04_semantics` triple

**Files:**
- Modify: `docs/source/tutorials/04_semantics/feature_extraction.ipynb`
- Modify: `docs/source/tutorials/04_semantics/segmentation.ipynb`
- Modify: `docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb`

- [ ] **Step 1: Rework all three** — `IMAGES` → `FRAMES` for input images; any reconstruction load uses the shared omega loader cell; any written artifact (features zarr, masks, figures) goes under `TUTORIAL_CACHE`. No method changes.
- [ ] **Step 2: Execute each headless (tmux, sequentially)** — each exits 0.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/04_semantics/feature_extraction.ipynb docs/source/tutorials/04_semantics/segmentation.ipynb docs/source/tutorials/04_semantics/maskclip_vs_talk2dino.ipynb
git commit -m "docs(tutorials): 04_semantics aligned to FRAMES/TUTORIAL_CACHE + omega cache"
```

### Task 5: `05_lifting/semantic_lifting.ipynb`

**Files:**
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`

- [ ] **Step 1: Rework** — shared omega loader cell (with `load_images=True` — `lift_features` needs pixel_indices/depth/confidence); lifted-feature output under `TUTORIAL_CACHE`; layout alignment.
- [ ] **Step 2: Execute headless (tmux)** — exits 0.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/05_lifting/semantic_lifting.ipynb
git commit -m "docs(tutorials): semantic_lifting on omega cache + new layout"
```

### Task 6: `07_localization/localization.ipynb` — LoMa primary

**Files:**
- Modify: `docs/source/tutorials/07_localization/localization.ipynb`

- [ ] **Step 1: Rework**

1. **Title markdown** — pipeline description updated: local feature matching with **LoMa** (dashboard default); XFeat/DISK available as drop-in alternatives.
2. **§1 load** — shared omega loader cell replaces the current zarr path.
3. **§3 localize** — the current notebook already has a working LoMa cell (its "second localizer with the LoMa-B extractor" section). Promote that construction to be THE localizer:
   ```python
   from collab_splats.localization import CameraLocalizer, LomaExtractor, plot_correspondences

   localizer = CameraLocalizer(result_trimmed, extractor=LomaExtractor())
   ```
   (Reuse the exact constructor call from the existing LoMa section — including any cache-dir argument, which must point under `TUTORIAL_CACHE`.)
4. **Delete** the XFeat run and the second-localizer comparison section.
5. **"Other matchers" markdown**:
   ```markdown
   ## Other matchers

   Swap the extractor to change the matching frontend — everything else is unchanged:

   - `XFeatExtractor()` — lighter/faster, fewer correspondences
   - `DiskExtractor()` — DISK + LightGlue
   - `LomaGExtractor()` — LoMa with global refinement
   ```
6. Query/trim/3D-view cells keep their current logic (paths already come from the loaded result).

- [ ] **Step 2: Execute headless (tmux)** — exits 0; localization finds a pose (`loc.pose is not None` cell passes).
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/07_localization/localization.ipynb
git commit -m "docs(tutorials): localization — LoMa primary matcher, omega cache, new layout"
```

### Task 7: layout-only pair — `colmap_sfm.ipynb`, `feedforward_mesh.ipynb`

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb`
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb`

- [ ] **Step 1: Rework paths only** — `FRAMES`/`TUTORIAL_CACHE` alignment. `feedforward_mesh.ipynb` is owned by the in-flight feedforward-mesh effort: diff working tree vs HEAD first, change path cells only, do not restructure.
- [ ] **Step 2: Execute both headless (tmux, sequentially)** — exit 0. If `colmap_sfm` exceeds the 1800 s cell timeout on 30 frames, cap its frame list (e.g. `image_paths[:15]`) in the notebook with a markdown note.
- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb
git commit -m "docs(tutorials): colmap_sfm + feedforward_mesh path alignment"
```

### Task 8: `03_splats` / `06_mesh` fieldwork-data migration

**Files:**
- Modify: `docs/source/tutorials/03_splats/derive_splats.ipynb`
- Modify: `docs/source/tutorials/03_splats/visualization.ipynb`
- Modify: `docs/source/tutorials/06_mesh/create_mesh.ipynb`

- [ ] **Step 1: Gate — verify C0043 splat training data exists**

Run: `ls /workspace/outputs/2024_02_06/C0043/` and check for the artifacts these notebooks consume (trained splat checkpoint / nerfstudio outputs — read each notebook's load cells to get the exact expected paths).
If ABSENT: keep the `BASE_DIR` override, add a dated markdown TODO cell at the top of each notebook ("2026-07-18: awaiting C0043 splat training outputs; still on fieldwork-data"), record the blocker in this plan's completion notes, commit that, and skip Steps 2-3.

- [ ] **Step 2: Migrate** — remove `BASE_DIR = /workspace/fieldwork-data/` overrides; standard `%run ../tutorial_config.py`; outputs under `TUTORIAL_CACHE`.
- [ ] **Step 3: Execute headless (tmux; derive_splats is the heaviest — no parallel work)** — exit 0.
- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/03_splats/derive_splats.ipynb docs/source/tutorials/03_splats/visualization.ipynb docs/source/tutorials/06_mesh/create_mesh.ipynb
git commit -m "docs(tutorials): 03_splats/06_mesh migrated to canonical C0043 layout"
```

### Task 9: `evals/ground_truth_evals.ipynb` results path

**Files:**
- Modify: `docs/source/tutorials/evals/ground_truth_evals.ipynb`

- [ ] **Step 1: Rework** — replace the hardcoded results path with a top-of-notebook constant:
   ```python
   # Results written by evals/eval_gt.py (CLI/tmux only — never run compute here)
   RESULTS_DIR = Path("../../../../evals/results").resolve()
   ```
   All downstream cells read via `RESULTS_DIR`. Visualization-only notebook — do NOT execute compute; run headless only if results exist locally, otherwise leave outputs as-is and note it.
- [ ] **Step 2: Commit**

```bash
git add docs/source/tutorials/evals/ground_truth_evals.ipynb
git commit -m "docs(tutorials): ground_truth_evals configurable results path"
```

### Task 10: `tutorial_config.py` cleanup + final gates

**Files:**
- Modify: `docs/source/tutorials/tutorial_config.py`

- [ ] **Step 1: Verify no notebook reads `CACHE_DIR`**

Run: `grep -l "CACHE_DIR" docs/source/tutorials/*/*.ipynb`
Expected: no matches (earlier tasks folded them into `OUTPUT_DIR`/`TUTORIAL_CACHE`). If matches remain, fix those notebooks first.

- [ ] **Step 2: Edit `tutorial_config.py`**

Remove the `CACHE_DIR = OUTPUT_DIR` line. In `_infer_video_path`, drop the dead first candidate (rclone-relative `video_ref` never resolves as an absolute path):

```python
            if raw:
                candidate = output_dir / Path(raw).name
                if candidate.exists():
                    return candidate
```

- [ ] **Step 3: Final gates**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q          # suite matches known-failures baseline (no library changes expected to alter it)
```

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/tutorial_config.py
git commit -m "docs(tutorials): retire CACHE_DIR alias; drop dead video_ref candidate"
```

---

## Completion notes (fill during execution)

- Blockers: _record here_
