# Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate split specs/plans into `docs/superpowers/`, write public README + enriched CLAUDE.md agent context, and prune stale notebook structure.

**Architecture:** File moves (git mv for history), sed-based link fixes in STATE.md/WORKLOG.md, direct edits to CLAUDE.md, new README.md at repo root, and deletion of superseded notebook stage/ dirs.

**Tech Stack:** git, bash/sed, markdown

---

## Files Modified

| File | Change |
|------|--------|
| `worklog/specs/*.md` (19 files) | Moved → `docs/superpowers/specs/` |
| `worklog/plans/*.md` (19 files) | Moved → `docs/superpowers/plans/` |
| `worklog/STATE.md` | Fix spec/plan links to repo-root paths |
| `worklog/WORKLOG.md` | sed replace `worklog/specs/` → `docs/superpowers/specs/` etc. |
| `CLAUDE.md` | Update spec/plan location lines, add in-flight + known-issues section |
| `README.md` | Create at repo root |
| `docs/source/tutorials/02_pointcloud/stage/` | Delete entire dir |
| `docs/source/tutorials/06_mesh/stage/` | Delete entire dir |
| `docs/source/tutorials/06_mesh/feedforward_mesh.ipynb` | Delete (duplicate of `02_pointcloud/feedforward_mesh.ipynb`) |

---

## Task 1: Move specs and plans to docs/superpowers/

**Files:**
- Move: `worklog/specs/*.md` → `docs/superpowers/specs/`
- Move: `worklog/plans/*.md` → `docs/superpowers/plans/`

No filename conflicts exist between the two locations.

- [ ] **Step 1: Move all worklog spec files**

```bash
cd /workspace/collab-splats && git mv worklog/specs/2026-05-07-gt-eval-harness-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-08-feedforward-import-cleanup-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-14-feedforward-mesh-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-20-bae-vggt-parity-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-20-docs-site-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-20-feature-lifting-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-20-keyframe-extraction-tutorial-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-20-positional-debiasing-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-bae-parity-handoff.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-camera-localization-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-feedforward-notebook-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-inline-documentation-cleanup-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-localization-cleanup-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-semantic-lifting-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-semantic-lifting.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-video-decode-speedup-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-21-xfeat-matching-fix-design.md docs/superpowers/specs/
git mv worklog/specs/2026-05-23-semantic-lifting-7scenes.md docs/superpowers/specs/
git mv worklog/specs/localization-cleanup.md docs/superpowers/specs/
```

- [ ] **Step 2: Verify worklog/specs/ is now empty**

```bash
ls /workspace/collab-splats/worklog/specs/
```

Expected: empty output (directory exists but no .md files).

- [ ] **Step 3: Move all worklog plan files**

```bash
cd /workspace/collab-splats
git mv worklog/plans/2026-05-07-gt-eval-harness.md docs/superpowers/plans/
git mv worklog/plans/2026-05-08-feedforward-import-cleanup.md docs/superpowers/plans/
git mv worklog/plans/2026-05-14-feedforward-mesh.md docs/superpowers/plans/
git mv worklog/plans/2026-05-20-bae-vggt-parity.md docs/superpowers/plans/
git mv worklog/plans/2026-05-20-docs-site.md docs/superpowers/plans/
git mv worklog/plans/2026-05-20-feature-lifting.md docs/superpowers/plans/
git mv worklog/plans/2026-05-20-keyframe-extraction-tutorial.md docs/superpowers/plans/
git mv worklog/plans/2026-05-20-positional-debiasing.md docs/superpowers/plans/
git mv worklog/plans/2026-05-20-semantics-refactor.md docs/superpowers/plans/
git mv worklog/plans/2026-05-21-camera-localization.md docs/superpowers/plans/
git mv worklog/plans/2026-05-21-feedforward-notebook.md docs/superpowers/plans/
git mv worklog/plans/2026-05-21-inline-documentation-cleanup.md docs/superpowers/plans/
git mv worklog/plans/2026-05-21-localization-cleanup.md docs/superpowers/plans/
git mv worklog/plans/2026-05-21-xfeat-matching-fix.md docs/superpowers/plans/
git mv worklog/plans/2026-05-23-semantic-lifting-7scenes.md docs/superpowers/plans/
git mv worklog/plans/2026-05-26-docker-env-setup-cleanup.md docs/superpowers/plans/
git mv worklog/plans/2026-05-26-reconstructor-wrapper.md docs/superpowers/plans/
git mv worklog/plans/2026-05-26-test-validation.md docs/superpowers/plans/
git mv worklog/plans/2026-05-28-multiview-confidence-generalise.md docs/superpowers/plans/
```

- [ ] **Step 4: Verify worklog/plans/ is now empty**

```bash
ls /workspace/collab-splats/worklog/plans/
```

Expected: empty output.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git commit -m "refactor(worklog): move specs/plans to docs/superpowers/ — single canonical location

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Fix STATE.md cross-references

**Files:**
- Modify: `worklog/STATE.md`

STATE.md currently uses three link formats that need normalising to repo-root-relative paths:
1. `(specs/FILE)` / `(plans/FILE)` — relative to worklog/, now broken
2. `(../docs/superpowers/specs/FILE)` — correct dir but uses `../` relative prefix
3. `(specs/2026-05-24-ba-module-cleanup-design.md)` — already in superpowers, link was already stale

All become `(docs/superpowers/specs/FILE)` and `(docs/superpowers/plans/FILE)`.

- [ ] **Step 1: Fix plain `specs/` and `plans/` references**

```bash
sed -i 's|](specs/|](docs/superpowers/specs/|g' /workspace/collab-splats/worklog/STATE.md
sed -i 's|](plans/|](docs/superpowers/plans/|g' /workspace/collab-splats/worklog/STATE.md
```

- [ ] **Step 2: Fix `../docs/superpowers/` relative prefix**

```bash
sed -i 's|](../docs/superpowers/|](docs/superpowers/|g' /workspace/collab-splats/worklog/STATE.md
```

- [ ] **Step 3: Verify all spec/plan links now use repo-root format**

```bash
grep -n 'specs/\|plans/' /workspace/collab-splats/worklog/STATE.md | grep -v 'docs/superpowers' | grep -v 'history/'
```

Expected: no output (all non-history links now point to `docs/superpowers/`).

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats && git add worklog/STATE.md && git commit -m "docs(worklog): fix STATE.md spec/plan links to docs/superpowers/ paths

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Fix WORKLOG.md cross-references

**Files:**
- Modify: `worklog/WORKLOG.md`

WORKLOG.md references historical entries with paths like `` `worklog/specs/2026-05-21-localization-cleanup-design.md` ``. These won't resolve after the move. Replace with `docs/superpowers/specs/` prefix. Leave `worklog/history/`, `worklog/decisions/`, `worklog/notes/` paths untouched.

- [ ] **Step 1: Replace worklog/specs/ and worklog/plans/ references**

```bash
sed -i 's|worklog/specs/|docs/superpowers/specs/|g' /workspace/collab-splats/worklog/WORKLOG.md
sed -i 's|worklog/plans/|docs/superpowers/plans/|g' /workspace/collab-splats/worklog/WORKLOG.md
```

- [ ] **Step 2: Verify history/ paths are untouched**

```bash
grep 'worklog/history/' /workspace/collab-splats/worklog/WORKLOG.md | head -5
```

Expected: lines still show `worklog/history/` (not replaced).

- [ ] **Step 3: Verify no stale worklog/specs or worklog/plans references remain**

```bash
grep 'worklog/specs/\|worklog/plans/' /workspace/collab-splats/worklog/WORKLOG.md
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats && git add worklog/WORKLOG.md && git commit -m "docs(worklog): update WORKLOG.md spec/plan paths to docs/superpowers/

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Two existing lines need updating, and a new "In-flight work" section needs adding.

- [ ] **Step 1: Update spec/plan location line (line 9)**

Change:
```
- Active spec/plan for in-flight work lives in worklog/{specs,plans}/. Completed work is archived to worklog/history/.
```
To:
```
- Active spec/plan for in-flight work lives in docs/superpowers/specs/ and docs/superpowers/plans/. Completed work is archived to worklog/history/.
```

- [ ] **Step 2: Update superpowers location line (line 12)**

Change:
```
- Superpowers information belongs in worklog/ directory
```
To:
```
- Superpowers specs/plans belong in docs/superpowers/specs/ and docs/superpowers/plans/.
```

- [ ] **Step 3: Add In-Flight Work section after the "Always follow" block**

After the closing line of the "Always follow" bullet list (line 12), add a new section:

```markdown

## In-Flight Work

These tasks are started but not complete — do not assume their targets are done:

- **gt-eval-harness** — ground-truth ATE evaluation harness ([spec](docs/superpowers/specs/2026-05-07-gt-eval-harness-design.md) · [plan](docs/superpowers/plans/2026-05-07-gt-eval-harness.md))
- **feedforward-import-cleanup** — import hygiene pass ([spec](docs/superpowers/specs/2026-05-08-feedforward-import-cleanup-design.md) · [plan](docs/superpowers/plans/2026-05-08-feedforward-import-cleanup.md))
- **feedforward-mesh** — feedforward → TSDF mesh pipeline ([spec](docs/superpowers/specs/2026-05-14-feedforward-mesh-design.md) · [plan](docs/superpowers/plans/2026-05-14-feedforward-mesh.md))
- **docs-site** — Sphinx site setup ([spec](docs/superpowers/specs/2026-05-20-docs-site-design.md) · [plan](docs/superpowers/plans/2026-05-20-docs-site.md))
- **bae-vggt-parity** — verify BA matches upstream `zitongzhan/vggt --implementation bae` ([spec](docs/superpowers/specs/2026-05-20-bae-vggt-parity-design.md))

Known test failures: `worklog/known-test-failures.md`
```

- [ ] **Step 4: Verify CLAUDE.md no longer references worklog/{specs,plans}**

```bash
grep 'worklog/{specs\|worklog/specs\|worklog/plans' /workspace/collab-splats/CLAUDE.md
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add CLAUDE.md && git commit -m "docs(claude): update spec/plan location to docs/superpowers/, add in-flight work section

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Write public README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Check no README.md exists at repo root**

```bash
ls /workspace/collab-splats/README.md 2>/dev/null || echo "does not exist"
```

Expected: `does not exist` (or note its content if it exists and needs merging).

- [ ] **Step 2: Write README.md**

Create `/workspace/collab-splats/README.md` with this content:

```markdown
# collab-splats

Video/image → 3D pointcloud → mesh + semantic features. Feedforward reconstruction using VGGT-X or MapAnything, with optional bundle adjustment, SL(4) loop closure, TSDF/Poisson meshing, semantic feature lifting, and camera localization.

## Capabilities

- **Feedforward reconstruction** — VGGT-X and MapAnything pointcloud creators
- **Bundle adjustment** — Levenberg-Marquardt refinement (bae backend)
- **Loop closure** — SL(4) pose graph from MIT-SPARK/VGGT-SLAM
- **Semantic lifting** — DINOv2/SAM features lifted into 3D pointcloud
- **Meshing** — TSDF and Poisson surface reconstruction
- **Camera localization** — SALAD retrieval + hloc matching

## Install

```bash
bash setup.sh               # core install (nerfstudio conda env)
bash setup/feedforward.sh   # VGGT-X + MapAnything models
```

Python env: always use `/opt/conda/envs/reconstruction/bin/python` (py3.11).

## Getting Started

Tutorials in `docs/source/tutorials/`, numbered by pipeline stage:

| Stage | Topic |
|-------|-------|
| 01 · Preprocessing | Keyframe extraction |
| 02 · Pointcloud | Feedforward methods, bundle adjustment, loop closure, COLMAP |
| 03 · Splats | Derive splats, visualization |
| 04 · Semantics | Feature extraction, segmentation |
| 05 · Lifting | Semantic feature lifting |
| 06 · Mesh | Surface reconstruction |
| 07 · Localization | Camera localization |

## Evaluation

```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py --help
```

Results land in `evals/results/` (gitignored). See `docs/source/tutorials/evals/ground_truth_evals.ipynb` for visualization.
```

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats && git add README.md && git commit -m "docs: add public README.md — capabilities, install, tutorial index

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Notebook cleanup

**Files:**
- Delete: `docs/source/tutorials/02_pointcloud/stage/colmap_sfm.ipynb`
- Delete: `docs/source/tutorials/02_pointcloud/stage/slam_loop_closure.ipynb`
- Delete: `docs/source/tutorials/06_mesh/stage/feedforward_mesh.ipynb`
- Delete: `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb`
- Delete: `docs/source/tutorials/06_mesh/feedforward_mesh.ipynb`

These are draft copies in `stage/` subdirectories, all superseded by the live notebooks registered in `index.rst`. The `06_mesh/feedforward_mesh.ipynb` is a duplicate of `02_pointcloud/feedforward_mesh.ipynb` and is not in the tutorial index.

- [ ] **Step 1: Confirm none of the targets appear in index.rst**

```bash
grep -E 'stage/|06_mesh/feedforward_mesh' /workspace/collab-splats/docs/source/tutorials/index.rst
```

Expected: no output (none are in the index).

- [ ] **Step 2: Delete stage/ dirs and duplicate**

```bash
git -C /workspace/collab-splats rm docs/source/tutorials/02_pointcloud/stage/colmap_sfm.ipynb
git -C /workspace/collab-splats rm docs/source/tutorials/02_pointcloud/stage/slam_loop_closure.ipynb
git -C /workspace/collab-splats rm docs/source/tutorials/06_mesh/stage/feedforward_mesh.ipynb
git -C /workspace/collab-splats rm docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb
git -C /workspace/collab-splats rm docs/source/tutorials/06_mesh/feedforward_mesh.ipynb
```

- [ ] **Step 3: Verify index.rst notebooks all still exist on disk**

```bash
python3 -c "
import re, os
idx = open('/workspace/collab-splats/docs/source/tutorials/index.rst').read()
base = '/workspace/collab-splats/docs/source/tutorials/'
entries = re.findall(r'^\s{3}(\S+)$', idx, re.MULTILINE)
missing = [e for e in entries if not os.path.exists(base + e + '.ipynb')]
print('Missing:', missing if missing else 'none')
"
```

Expected: `Missing: none`

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats && git commit -m "chore(docs): delete stage/ draft notebooks and duplicate 06_mesh/feedforward_mesh

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Part 1 (consolidate specs/plans) → Tasks 1–3 ✓
- Part 2 (CLAUDE.md + README) → Tasks 4–5 ✓
- Part 3 (notebook cleanup) → Task 6 ✓

**Placeholder scan:** None found.

**Type consistency:** N/A — no code.

**Edge case:** Task 1 lists 19 plan files but `ls` showed 19 entries. If the file list diverges (e.g. a new file was added since this plan was written), the individual `git mv` commands will error on the unknown file. In that case: check `ls worklog/specs/` and `ls worklog/plans/` first, add any unlisted files, skip any listed files that don't exist.
