# Handoff: Outdoor evaluation of geometric verification

**Date:** 2026-08-17
**Branch:** `refactor/cu121-uv-migration` (single shared branch; concurrent sessions — commit with explicit pathspecs only)
**Predecessor effort:** geometric-verification v1, complete at `f06c8c1`
([spec](../specs/2026-08-14-geometric-verification-design.md) ·
[plan](../plans/2026-08-14-geometric-verification.md))

## Mission

Evaluate geometric verification on scenes larger than 7-Scenes chess, specifically outdoor
unbounded scenes. Two stages, in order:

- **Stage A — reference-free diagnosis on our own GoPro outdoor scenes** (no new code
  expected; hours). Run the `verify` stage + `evals/scripts/eval_verification.py` without
  `--gt_dir` on one or more processed scenes from `environments-processed`.
- **Stage B — ground-truth evaluation on Oxford Spires** (new code: downloader/undistorter/GT
  loader; days). Validates Tier 1 pose errors against a cm-accurate trajectory outdoors.

Stage A first. Its results shape how much of Stage B is worth doing — if the reference-free
diagnosis already shows large epipolar disagreement on outdoor scenes, Stage B quantifies it
against GT; if everything is clean, Stage B is confirmation and can be descoped to one sequence.

## What geometric verification is (context for a fresh agent)

`collab_splats/geometry/verification.py::verify_reconstruction(recon, features, matcher,
output_dir)` — self-diagnosis for feedforward reconstructions: does the model's geometry
survive classical checks?

- Exports keypoints + matches to a COLMAP `database.db`.
- **Tier 1:** `pycolmap.verify_matches` with `compute_relative_pose=True` → per-pair
  `PairStats` (rotation / translation-direction error of the epipolar-estimated relative pose
  vs the model's relative pose, inlier counts).
- **Tier 2:** `pycolmap.triangulate_points` from the model's (fixed) poses → per-frame track
  survival + reprojection stats.
- Writes `<backend>/colmap/verification.json` (all distributions median/p90/p99, nan→null)
  and `<backend>/colmap/verified/` model.
- **Report-only contract:** nothing feeds back into the reconstruction.
  `triangulate_points` MUTATES its reconstruction argument in place — the entry point copies
  via `pycolmap.Reconstruction(recon)` first. Do not remove that copy.
- **Sequential pairs only** (overlap window). Loop pairs are a known follow-on, not in scope
  here — but on long outdoor walks note in the report that revisits are unmeasured.
- Extractor is shared with localization via the zarr feature cache
  (`build_localization_db` + `load_reconstruction_features`); registry key comes from
  `localization.extractor` in the config (`loma` in base.yaml; `xfeat` and `disk` also valid).
  **XFeatStar is rejected by design** (no keypoint table indices) — `xfeat-star` will raise.
- Pipeline surface: leaf stage `verify` (`Reconstructor.verify(overwrite=...)`), config
  boolean `pointcloud.geometric_verification: false`, `--stages verify` re-runs it against a
  processed scene (see `configs/README.md` § "Re-running one stage against a processed scene").
- Negative control is mandatory in any measurement: `--perturb_deg 2.0` rotates every 5th
  pose; those frames must be flagged (~2° pair error, depressed survival) and clean pairs must
  stay <0.2°. A threshold nobody mutated is presumed inert.

The eval script `evals/scripts/eval_verification.py` (CLI/tmux only, never a notebook):

```
/opt/venv/reconstruction/bin/python evals/scripts/eval_verification.py \
    --backend_dir <scene>/<backend> [--gt_dir <7scenes seq dir>] \
    --extractor xfeat --overlap 10 [--perturb_deg 2.0] --out <json>
```

Without `--gt_dir` it emits the reference-free columns only: Tier 1 estimated-vs-model
rotation/translation-direction errors, Tier 2 triangulated-vs-model relative depth agreement
at track pixels (scale-free), yield/track-length/survival distributions. The GT path
(`_load_gt_poses`, `_gt_depth`) is 7-Scenes-specific (`frame-XXXXXX.pose.txt` c2w,
`frame-XXXXXX.depth.png` uint16 mm, 65535 invalid) — Stage B must generalize or parallel it.

## Also still owed (predecessor's open experiment)

The chess/seq-01 indoor experiment (plan Task 8 Step 4: `--extractor xfeat`, `--extractor
loma`, and the `--perturb_deg 2.0` control) was never run — compute, human-gated. Local data
exists at `data/7scenes/`. Running it first is cheap and gives the indoor baseline column the
outdoor numbers will be compared against. Recommended order: chess → Stage A → Stage B.

## Stage A: reference-free diagnosis on own GoPro scenes

Candidate scenes (outdoor GoPro walking captures, already in `environments-processed`; listed
via `rclone lsd collab-data:environments-processed`):

- `2026_07_08-GoproSplat-GH010223` … `GH010227` (pick 1–2; also `2026_06_29-GoproSplat-GH010221`)

Steps:

1. Pull the scene with the stage-rerun path — `--stages verify` is a leaf stage, so
   `docs/examples/run_pipeline_remote.py --stages verify` pulls from `environments-processed`
   instead of rebuilding (contract: `configs/README.md`). Pull with NO excludes — the
   viewer's `PULL_EXCLUDES` drops arrays the eval needs (depth, world_points).
2. Run the `verify` stage (tmux). It builds the localization feature cache if absent.
   If the processed scene already has a `local_features/<other extractor>` cache and you want
   a different extractor, expect the stem-order guard `ValueError` — extract fresh rather
   than mixing caches.
3. Run `eval_verification.py` without `--gt_dir`, once per extractor (`xfeat`, `loma`), plus
   one `--perturb_deg 2.0` negative-control run. Outputs to
   `evals/results/verification/<scene>_<extractor>.json` (`evals/results/` is gitignored).
4. **Append** (never replace — concurrent sessions edit these docs) a Stage A section to
   `docs/superpowers/specs/2026-08-14-geometric-verification-measured-report.md`: the
   distributions, the negative-control outcome, and an explicit note on frame count, scene
   length, and that loop/revisit pairs are unmeasured.

What "interesting" looks like: chess is small-baseline indoor; outdoor unbounded stresses
far-field triangulation angles, sky/low-texture regions (track survival), and exposure
changes. Report survival and pair errors per frame index, not only globally — a walking
capture degrades locally.

## Stage B: Oxford Spires ground-truth evaluation

Dataset (chosen 2026-08-16 over Newer College — grayscale/IR — and Tanks & Temples — orbit
motion, no GT poses): https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/

Known facts: backpack walking capture matching our production regime; color global-shutter
1440×1080 @ 126° FoV; 24 sequences; Leica RTC360 TLS map GT + cm trajectory GT on 12 core
sequences; published NeRF/3DGS baselines. Caveats: **CC BY-NC-SA (NonCommercial — eval use is
fine, flag before any productized use)**; fisheye → PINHOLE undistortion required; GT range
caps ~60 m.

Work items (agent must verify specifics against the dataset docs — download mechanics,
calibration format, and trajectory format were NOT confirmed by the predecessor):

1. **Acquire one core sequence** (one with trajectory GT). Confirm hosting (site/HF), size,
   and layout before pulling more.
2. **Undistort to PINHOLE** using the published calibration (126° FoV fisheye — expect a
   real crop; record the undistorted K). The pipeline and verification both assume PINHOLE.
3. **Frames → pipeline:** feed undistorted frames through the standard pipeline (frames →
   pointcloud backend → verify). Respect the 300-frame budget or run submapped; note the
   cgroup memory cap below.
4. **GT loader:** extend `eval_verification.py` with an Oxford Spires pose loader (likely
   TUM-style timestamped trajectory — must associate timestamps to selected frames; the
   frame-index association is the error-prone step, test it). Keep the 7-Scenes loader
   working; select by CLI flag, don't autodetect. There is no per-pixel GT depth analogous to
   7-Scenes — Tier 2 GT columns may be TLS-based or dropped; scope that consciously and say
   which in the report.
5. **Run the same battery:** xfeat + loma + `--perturb_deg 2.0` control, GT columns on.
   Remember the LoGeR lesson: a GT column can be a noise floor — always report the
   reference-free columns alongside, and if a residual looks like "our bug", attribute it by
   giving the code more freedom than it ships with and showing the error does not drop.
6. **Append** a Stage B section to the measured report; update CLAUDE.md's geometric-verification
   entry (append to the "Owed" clause, do not rewrite the entry — concurrent sessions).

## Environment and repo constraints (non-negotiable)

- Python: `/opt/venv/reconstruction/bin/python` (py3.11). Base-shell `python` may be 3.13.
- Compute in **tmux**, never notebooks; container cgroup cap **46.6 GB** — no parallel heavy
  processes during eval runs.
- Tests: `-p no:randomly`. Known failures list: `docs/known-test-failures.md`.
- Never run repo-wide `black .` (venv black is newer than repo formatting).
- Commits: conventional with scope, explicit pathspecs, trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Never `--amend` a subagent's
  commit. Never stage: `data/tutorial/README.md`, `uv.lock`,
  `docs/source/tutorials/07_localization/ref_image.jpg`.
- Nothing deletes remote GCS objects; pushes are `rclone copy` with anchored excludes
  (`/*/colmap/database.db` already excluded).
- `evals/results/` is gitignored — findings live in the measured report doc, not in JSON.

## Success criteria

- Stage A: reference-free distributions + negative control for ≥1 outdoor GoPro scene,
  appended to the measured report, with an explicit indoor-chess comparison column.
- Stage B: Tier 1 estimated-vs-model AND model-vs-GT columns on ≥1 Oxford Spires core
  sequence, negative control included, GT-association step tested, appended to the report.
- Both: no change to `verify_reconstruction`'s report-only contract; any eval-script change
  keeps the 7-Scenes path green (`tests/geometry/test_verification.py` +
  `tests/wrapper/test_verify_stage.py`, currently 8 passed).
