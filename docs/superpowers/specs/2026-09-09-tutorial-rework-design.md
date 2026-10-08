# Tutorial rework — shared scene, ten pages, landed on clean/final

**Status:** revised 2026-10-08 (supersedes the 2026-10-02 fourteen-page revision), plan pending
**Branch:** `clean/tutorials` · **Worktree:** `.worktrees/tutorial-rework`
**Date:** 2026-09-09 · revised 2026-10-02 · revised 2026-10-08

## Revision 2026-10-08 — what changed and why

The 2026-10-02 revision was carried out: fourteen pages, all executed on `clean/final` at
`4781e72c` (rgbd-ba, LC-window BA and vismatch localization included). Since then 23 commits
landed on `clean/final` (semantics-storage, mesh-query-heat, scene-viewer). A trial merge
is textually clean, but the contract gate fails two pages: `lifting_and_query` and `ocr_lens`
import `extract_feature_cache`, which is gone, and both point at the deleted
`docs/examples/ocr_lens_viewer.py`.

A read of all fourteen pages also found the prose model-flavored (97 em-dashes, 49 bold
phrases, slogans such as "the count is the contract", plan language such as "Package gap"),
51 code lines over 100 characters, and pages that duplicate each other's compute.

This revision:

- replaces per-page isolation with **one shared scene** that pages build on
- cuts fourteen pages to **ten**; the quickstart notebook becomes the README's Getting Started
- moves the mesh to VGGT-Omega depth only (`mesh.source: feedforward`, the base default);
  splats are trained on `train_splats` alone
- adapts the two semantics pages to the codes-only store and the vertex arrays the
  semantics stage now writes
- adds a readability pass with gate checks
- lands the branch on `clean/final`

## Goals

1. Every page runs top to bottom on its own: opened cold, it builds what it needs.
2. Pages opened in order reuse each other's stage outputs; no stage runs twice.
3. One clear subject per page; no page duplicates another page's compute.
4. Plain, short prose and readable code, enforced by the contract gate.
5. Every call matches `clean/final`; package functions are used wherever they exist.

## Non-goals

- Changing `collab_splats` APIs or behavior.
- Teaching every backend. VGGT-Omega is the backend shown; the others get one line.
- `remote`, `dashboard`, evals: named in prose only.

## Shared scene

`docs/source/tutorials/tutorial.py` owns one scene for every page:

```python
SCENE_DIR = REPO_ROOT / "data/tutorial_scene"   # gitignored; persists across sessions
SCENE_CONFIG = {
    "preproc": {"max_frames": 96},              # SfM needs ~1 s spacing; 96 serves every page
    "mesh": {"texture": True},                  # source stays base.yaml's feedforward (VGGT-Omega)
    "splats": {"representation": "scaffold", "primitive": "2dgs", "max_steps": 1000},
    "semantics": {"extractor": "maskclip", "max_epochs": 20},
}

def tutorial_scene(*stages: str, extractor: str | None = None) -> Reconstructor
def work_dir(page: str) -> Path
```

- `tutorial_scene(*stages)` builds a `Reconstructor` over `SCENE_DIR` with `input_path =
  VIDEO_PATH` and `SCENE_CONFIG` merged over `base.yaml`, runs the named stages that are not
  yet `done()`, and returns the scene. Dropping done stages before `run` avoids the named-leaf
  refusal. Dependencies already on disk are reused by `Reconstructor.run` itself.
- Splats are a terminal stage: `STAGES["splats"] = ("pointcloud",)` and nothing depends on it.
  `enabled` filters only a bare `run()`; a named stage always runs. Only `train_splats` names
  it, so splats train on that page alone and `splats.enabled` stays at the base default (false).
- `extractor=` is the only per-page override. Each extractor writes its own
  `<extractor>_codes.zarr` / `<extractor>_lifted.zarr`, so pages never collide.
- `work_dir(page)` returns `SCENE_DIR/work/<page>/`, emptied and recreated on each call. A
  page's own experiments (longhand creators, fusions, BA, LC) write there and never touch
  stage outputs.
- **Staleness is manual.** `done()` checks existence only. `index.rst` says: delete
  `data/tutorial_scene/` after pulling code changes. The docs sweep always starts by deleting it.
- Pages never call `mkdtemp`, never pass `overwrite=True`, never set config keys on the scene.

The 96-frame budget removes the old special cases: no 16-vs-96 split, and `refinement` uses
the production `submap_size` 20 (about five submaps), so the `SUBMAP_SIZE` knob goes.

## Page set — ten

```
docs/source/tutorials/
  tutorial.py
  index.rst
  01_preprocessing/preprocessing.ipynb          # video_quality + keyframe_extraction
  02_pointcloud/reconstruction.ipynb            # + reconstruction_quality_report
  02_pointcloud/refinement.ipynb
  03_splats/train_splats.ipynb
  04_mesh/mesh.ipynb                            # tsdf_mesh + texturing
  05_semantics/feature_extraction.ipynb
  05_semantics/segmentation.ipynb
  05_semantics/lifting_and_query.ipynb
  05_semantics/ocr_lens.ipynb
  06_localization/localization.ipynb
```

Every page has the same shape:

1. `# Title` and one short paragraph: what goes in, what comes out.
2. One setup cell: imports, then `scene = tutorial_scene(...)` (or the committed image).
3. Numbered sections, `## 1. Title`.
4. `## In a pipeline run`: the config keys that run the same thing in a stage.

## Page by page

`stages` = what the page passes to `tutorial_scene`.

### Quickstart → README

No notebook. `README.md` "Getting Started" is rewritten (it still lists the old seven-section
layout and `06_mesh/splats_mesh.ipynb`): a five-line `Reconstructor` example, the
`python -m collab_splats local` equivalent, `python -m collab_splats.viewer <backend_dir>`, and
the stage → page table. `index.rst` carries the same table. Only that section of the README is
touched; other uncommitted README edits in the main checkout are left alone.

### 01 · `preprocessing`

Stages: `preproc`.
Reads the stage's `video_quality_report.json` (no second decode). Report plots
(`plot_photometric`, `plot_motion`, `plot_correlation`, `plot_frame_extremes`), then
`filter_frame_quality`, the three samplers and `plot_selection`, then the `images/` store the
stage wrote. Undistortion is a short prose note with its config key (`preproc.undistort`, off
by default); the 40-frame calibration demo is dropped.

### 02 · `reconstruction`

Stages: `preproc`, `pointcloud`, `reconstruction_quality_report`.
Feedforward: `scene.result` from the stage (VGGT-Omega), the `PointcloudResult` fields and the
two intrinsics, `clean_pointcloud`, cloud and frustums. SfM: `InstantSfMCreator` longhand into
`work_dir`. Judging both: the stage's report tables for feedforward,
`compute_reconstruction_quality` called directly on the in-memory SfM result, one comparison
figure, and how to read it. One SfM run per page, not two.

### 02 · `refinement`

Stages: `preproc`, `pointcloud`.
`BundleAdjustment.refine` on the scene's VGGT-Omega result (what the `refine` stage does), then
`LoopClosure(..., LoopClosureConfig(), ba=BundleAdjustmentConfig())` longhand into `work_dir`
with production `submap_size`, since loop closure re-runs the backend per submap. Loss terms, focal change, camera shift, trajectories before and after.

### 03 · `train_splats`

Stages: `preproc`, `pointcloud`, `splats`.
The scaffold-2DGS config and its two rules (no `sh_degree*`, `opacity_reg` 0), the stage's
quality report, `load_checkpoint` + `render_views` gallery. `train` is no longer called
longhand: the frame / depth-target glue goes, and the scene trains once.

### 04 · `mesh`

Stages: `preproc`, `pointcloud`, `mesh`.
TSDF prose (`sdf_trunc`, not `voxel_size`, sets the thinnest surviving structure). Longhand
into `work_dir`: `frame_depths` from the VGGT-Omega zarr, `sky_masks`, `create_tsdf_mesh`,
`clean_repair_mesh`, `prepare_mesh`. Then the stage's own `mesh.ply` and `texture/` (atlas,
textured render) with the occluder / color-gain / view-chart explanation. No splat source.

### 05 · `feature_extraction`, `segmentation`

No scene; `QUERY_IMAGE` only. Content unchanged apart from the readability pass.

### 05 · `lifting_and_query`

Stages: `preproc`, `pointcloud`, `mesh`, `semantics` (maskclip).
Longhand on points, since lifting is the subject: MaskCLIP `forward` over the keyframes in
memory, `FeatureAutoencoder` fit, `lift_features` onto the points, `write_point_features` /
`read_point_features` into `work_dir`. Vertices from the stage: read `vertex_features` from
`maskclip_lifted.zarr`, `score_queries`, `transfer_features` smoothing, plot. Ends with
`python -m collab_splats.viewer <backend_dir>`.

### 05 · `ocr_lens`

Stages: `preproc`, `pointcloud`, `mesh`, `semantics` with `extractor="ocr_lens"`.
Pipeline-first: read `vertex_word_ids`, `vertex_word_probs` and the `words` attr from
`ocr_lens_lifted.zarr`; top words, one word probe on the mesh. Prose keeps why the stage
decodes per frame before lifting (the decoder's RMSNorm is non-linear). Ends with the viewer.
Needs the LLaVA-1.6 weights cached (`HF_HUB_OFFLINE=1`).

### 06 · `localization`

Stages: `preproc`, `pointcloud`.
`CameraLocalizer.from_pointcloud(..., extractor=LocalMatcher("loma"))`, then two queries:

- **A. In-video frame** — `data/tutorial/tutorial_ref-frame.jpg`, frame 0 of the tutorial video
  (committed from the stray `07_localization/ref_image.jpg`; mean pixel difference 0.12 vs
  frame 0). `data/` ignores the whole tree and `!data/tutorial/` cannot re-include a child of
  an excluded directory, so the file is added with `git add -f` like the existing assets. Compared against the reconstruction's fitted pose for the same source frame:
  rotation error in degrees, translation error as a fraction of the trajectory extent. The plan
  verifies frame 0 is in the 96-frame selection before the page is written.
- **B. Cross-video frame** — `tutorial_example-frame.jpg` (GoPro `GX010119`): correspondences,
  inlier distribution, pose in the scene, reprojection overlay. Visual check only.

## Readability rules

- Plain statements. No slogans, no "X is the contract", no aphorisms.
- At most two bold phrases per page; at most three em-dashes per page.
- No plan language ("Package gap"), no development history ("registered 7 of 16").
- `## 1. Title` headings; no `§`.
- No `%load_ext autoreload` cell.
- Code lines ≤ 100 characters; long imports parenthesized; one call per line, no nested calls;
  no comprehension tricks.
- One sentence after each figure on what to look for.
- A cell stays only if it teaches the page's subject: no decorative prints, no duplicate plots.

## Contract gate

`tests/docs/test_tutorial_contract.py`:

- kept: banned shared-cache tokens (`data/outputs`, `TUTORIAL_CACHE`, `OUTPUT_DIR`,
  `IMAGES_DIR`), no `sys.path.insert`, no "run X first", imports resolve, no notebook-defined
  functions, matplotlib cap, executed outputs present, index lists every page
- changed: the page list is the ten above
- new: no `mkdtemp`, no `overwrite=True`; no `§`, `Package gap` or `autoreload`; em-dash and
  bold caps; code lines ≤ 100 characters

`tests/docs/test_tutorial_helpers.py` covers `tutorial_scene` (skips done stages, reuses
`SCENE_DIR`, `extractor=` selects the store) and `work_dir` (emptied on each call).

## Integration and landing

1. Back up the tip to `refs/backup/tutorial-release/pre-rebase`; `git rebase clean/final`
   (trial merge clean). Venv preflight: `gsplat.__version__`, GPU, cached LLaVA and LoMa.
2. Rework pages and gate per the sections above.
3. **Full sweep:** delete `data/tutorial_scene/`, then execute all ten pages in page order,
   one at a time, with outputs committed. Then re-execute one late page (`localization`) on its
   own after deleting the scene, to prove a cold start works.
4. Gates: `tests/docs`, docstring contract, import style, Sphinx build of `docs/`, committed
   notebook outputs ≤ 32 MB (21.3 MB today).
5. Land on `clean/final`: cherry-pick `7a03db99` (`fix(mesh)`) and `0fd93dbb`
   (`fix(semantics)`) as their own commits, then one squashed `docs(tutorials)` commit; append
   the CHANGELOG entry, drop `tutorial-rework` from CLAUDE.md In-Flight, replace the stale spec
   on `clean/final` with this file. Backup refs before the squash. Branch, worktree and the
   stale `clean/localization` branch deleted only after the user confirms.

If `clean/final` moves during the work, rebase again and re-run the import gate before the sweep.

## Risks

- **Wall-clock.** One full sweep is a scene build (96-frame VGGT-Omega, mesh with texture
  bake, one splat train, maskclip and OCR semantics) plus each page's own longhand work (InstantSfM, LC, longhand TSDF).
- **Stale scene after code changes.** Manual delete; stated on the index page and done by the sweep.
- **Under-converged splats** at 1000 steps; the page states the production 30000.
- **Notebook size** ≤ 32 MB: downscaled galleries, `n=2` montages, atlas shown at 1024 px.
