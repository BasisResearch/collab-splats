# Final Docs + Configs + CI Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop dead docs/configs from `clean/final`, fix the stale parts that stay, and make every CI workflow pass without changing what it does.

**Architecture:** Five independent commits (A untrack internal docs, B configs, C user docs, D CLAUDE.md, E CI). Each is gated by a link grep plus the contract tests. No runtime code changes except E's lint fixes, which must be behavior-neutral.

**Tech Stack:** git, Sphinx, ruff, mypy, GitHub Actions, `/opt/venv/reconstruction/bin/python` (py3.11).

**Spec:** `docs/superpowers/specs/2026-10-08-final-docs-configs-cleanup-design.md`

---

## Ground rules (read before any task)

- Work on `clean/final` in `/workspace/collab-splats`. Other sessions share this checkout and its git index.
- Commit with `git commit --only <paths>` (or `git rm --cached` then `git commit` with only those paths staged — check `git diff --cached --stat` first). Never `git add -A`, never `--amend`.
- Do NOT touch the pre-existing uncommitted edits in `README.md`, `uv.lock`, `docs/superpowers/specs/2026-09-09-clean-final-dead-code-design.md`.
- `docs/superpowers/` is gitignored: new or kept files there need `git add -f`.
- `PY=/opt/venv/reconstruction/bin/python`.
- Contract gate (run after every task):

```bash
cd /workspace/collab-splats && $PY -m pytest tests/test_docstring_contract.py tests/test_import_style.py -q -p no:randomly 2>&1 | tail -3; echo "exit=${PIPESTATUS[0]}"
```
Expected: `passed`, `exit=0`.

---

### Task A: Untrack completed `docs/superpowers/` files

**Files:** index only (`git rm --cached`); nothing deleted from disk.

- [ ] **Step 1: Write the keep-list**

```bash
cd /workspace/collab-splats && cat > /tmp/claude-0/-workspace-collab-splats/cc8a22c2-008e-4c3f-abe5-15817fbee14f/scratchpad/keep.txt <<'EOF'
docs/superpowers/CHANGELOG.md
docs/superpowers/decisions/014-defer-romav2.md
docs/superpowers/decisions/015-streaming-submap-centric-vs-scattered.md
docs/superpowers/decisions/017-release-cleanup-rules.md
docs/superpowers/decisions/018-sfm-backends.md
docs/superpowers/decisions/019-remove-pgsr.md
docs/superpowers/decisions/020-rgbd-ba.md
docs/superpowers/decisions/021-lift-features-onto-displayed-geometry.md
docs/superpowers/decisions/022-lc-window-ba.md
docs/superpowers/decisions/023-store-vertex-semantics.md
docs/superpowers/decisions/024-pixel-center-intrinsics.md
docs/superpowers/specs/2026-09-06-clean-final-integration-design.md
docs/superpowers/plans/2026-09-06-clean-final-integration.md
docs/superpowers/specs/2026-09-09-clean-final-dead-code-design.md
docs/superpowers/specs/2026-09-07-sky-segmentation-design.md
docs/superpowers/plans/2026-09-07-sky-segmentation.md
docs/superpowers/specs/2026-09-07-sky-mask-measured-report.md
docs/superpowers/specs/2026-09-25-vismatch-fork-design.md
docs/superpowers/specs/2026-09-25-vismatch-baseline.md
docs/superpowers/specs/2026-09-09-tutorial-rework-design.md
docs/superpowers/specs/2026-09-26-consistency-design.md
docs/superpowers/plans/2026-09-26-consistency-phase1.md
docs/superpowers/plans/2026-09-26-consistency-phase2.md
docs/superpowers/specs/2026-10-08-dashboard-merge-design.md
docs/superpowers/plans/2026-10-08-dashboard-merge.md
docs/superpowers/specs/2026-10-08-geometry-cleanup-design.md
docs/superpowers/specs/2026-10-08-final-docs-configs-cleanup-design.md
docs/superpowers/plans/2026-10-08-final-docs-configs-cleanup.md
docs/superpowers/specs/2026-09-24-preproc-release-cleanup-design.md
docs/superpowers/specs/2026-08-20-video-quality-report-measured.md
docs/superpowers/specs/2026-09-26-pycolmap-cuda-docker-design.md
docs/superpowers/specs/2026-08-17-vismatch-local-matcher-design.md
docs/superpowers/specs/2026-08-23-splats-measured-report.md
docs/superpowers/specs/2026-07-09-lc-parity-probe-results.md
EOF
wc -l < /tmp/claude-0/-workspace-collab-splats/cc8a22c2-008e-4c3f-abe5-15817fbee14f/scratchpad/keep.txt
```
Expected: `34` (33 from the spec plus this plan).

- [ ] **Step 2: Compute the untrack list and sanity-check it**

```bash
cd /workspace/collab-splats && S=/tmp/claude-0/-workspace-collab-splats/cc8a22c2-008e-4c3f-abe5-15817fbee14f/scratchpad
git ls-files docs/superpowers | sort > $S/tracked.txt
sort $S/keep.txt > $S/keep.sorted
comm -23 $S/tracked.txt $S/keep.sorted > $S/untrack.txt
comm -13 $S/tracked.txt $S/keep.sorted   # keep-list entries not tracked: expect empty after Step 3 of this task adds the plan
wc -l < $S/untrack.txt
```
Expected: untrack count `210` (242 originally tracked + spec = 243, minus 33 kept). The `comm -13` line may print only this plan's path if it is not yet committed; anything else means a typo in keep.txt — fix it.

- [ ] **Step 3: Untrack and commit**

```bash
cd /workspace/collab-splats && S=/tmp/claude-0/-workspace-collab-splats/cc8a22c2-008e-4c3f-abe5-15817fbee14f/scratchpad
xargs -a $S/untrack.txt git rm --cached --quiet
git diff --cached --stat | tail -1   # expect "210 files changed, ... deletions(-)" and nothing outside docs/superpowers
git commit -m "chore(docs): untrack completed superpowers specs, plans and handoffs

Files stay on disk; docs/superpowers/ is gitignored by design. Kept tracked:
CHANGELOG, decisions, in-flight specs/plans, and specs cited from code.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Verify files still on disk and nothing outside references an untracked path**

```bash
cd /workspace/collab-splats && S=/tmp/claude-0/-workspace-collab-splats/cc8a22c2-008e-4c3f-abe5-15817fbee14f/scratchpad
xargs -a $S/untrack.txt ls >/dev/null && echo "all on disk"
while read f; do b=$(basename "$f"); git grep -l -F "$b" -- ':!docs/superpowers' ; done < $S/untrack.txt | sort | uniq -c
```
Expected: `all on disk`. Grep hits are allowed ONLY in files Task C/D delete or edit (`docs/known-test-failures.md`, `docs/benchmarks/**`, `CLAUDE.md`). Any other hit: stop and report.

- [ ] **Step 5: Contract gate** (command in Ground rules).

---

### Task B: Configs

**Files:**
- Delete: `configs/loop_closure.yaml`, `evals/configs/rgbd_ba.yaml`
- Modify: `configs/base.yaml:2`, `evals/configs/7scenes.yaml:3`, `evals/README.md:55-56`, `docs/parity.md:60`, `configs/README.md`, `docs/pointcloud.md`

- [ ] **Step 1: Delete the two configs**

```bash
cd /workspace/collab-splats && git rm -q configs/loop_closure.yaml evals/configs/rgbd_ba.yaml
```

- [ ] **Step 2: Fix stale comments**

`configs/base.yaml` line 2 — replace
```yaml
# All dataset configs inherit from this and override specific values.
```
with
```yaml
# Run configs (--config, eval grid conditions) deep-merge over it and override specific values.
```

`evals/configs/7scenes.yaml` line 3 — replace
```yaml
# - no BA cells: the pipeline refuses BA with loop_closure, and 500 frames need windows
```
with
```yaml
# - no BA cells: 500 frames need windows; BA here is a separate grid condition
```

`evals/README.md` lines 55-56 — replace
```markdown
The pipeline refuses `bundle_adjustment` together with `loop_closure`, so a BA condition runs
single-pass and needs a `max_frames` the backend fits in one go.
```
with
```markdown
With `loop_closure` on, `bundle_adjustment` refines each window (decision 022); without it, a BA
condition runs single-pass and needs a `max_frames` the backend fits in one go.
```
Before writing this, confirm the claim: `grep -n "window_ba\|bundle_adjustment" collab_splats/reconstructor.py | sed -n 1,20p` must show BA running per LC window (spec cites `reconstructor.py` L722-727). If it does not, report instead of editing.

`docs/parity.md` line 60 — replace `64 (`configs/loop_closure.yaml`) and 50 (the chess eval below)` with `and 50 (the chess eval below)`, so the clause reads: ``ours run 20 (`LoopClosureConfig`) and 50 (the chess eval below)``.

- [ ] **Step 3: Trim `configs/README.md`**

Open the file and re-locate each section by heading (line numbers drift). Make these edits:
1. **Steps run per video** (~L35-41): add mesh as a step after splats/semantics, matching the stage order in `Reconstructor` (`grep -n "def preproc\|def pointcloud\|def semantics\|def mesh\|def splats\|STAGES" collab_splats/reconstructor.py`).
2. **Where outputs land** appears twice (~L53 and ~L747): keep the first, fold any tree entries unique to the second into it, delete the second.
3. **Migrations** (~L463-499), the "Historical" matcher note (~L511-514), the transforms.json note (~L804-808) and the layout-rename/gotcha notes (~L852-889): delete. Their history is already in `docs/superpowers/CHANGELOG.md` — confirm with `grep -n "transforms.json\|layout" docs/superpowers/CHANGELOG.md | head`; if a note has no CHANGELOG counterpart, append a one-line dated entry to CHANGELOG under the matching release instead of losing it.
4. **Key table** (~L361-461): add rows (key | default | meaning, same format as neighbors), defaults read from `configs/base.yaml`:
   - `semantics.extractor_kwargs`, `semantics.target_cosine`, `semantics.max_epochs`; add `ocr_lens` to the extractor values
   - `pointcloud.max_points`, `pointcloud.viz.enabled`, `pointcloud.viz.port`, `pointcloud.loop_closure.submap_size` (and the other `LoopClosureConfig` fields: `grep -n "^\s*[a-z_]*:" collab_splats/geometry/loop_closure/wrapper.py` or wherever `LoopClosureConfig` is defined — `git grep -n "class LoopClosureConfig"`)
   - `splats.representation`, `splats.scaffold`, `splats.appearance_opt`, `splats.appearance_lr`, `splats.losses.appearance_reg`
   - `reconstruction_quality_report.min_pair_overlap`
5. **L378** sentence that says `max_points`, `min_views`, `mv_rel_thresh`, `clean` are "rejected" in a backend block: change to say they collide with creator kwargs and fail with a `TypeError` when the pointcloud stage builds the creator.
6. **Backend sections** loger / instantsfm / colmap / hloc (~L520-746): cut them and paste under a new `## Backends` heading at the end of `docs/pointcloud.md`, unchanged except relative links (`../docs/x` → `x`, `docs/superpowers/decisions/018-sfm-backends.md` → `superpowers/decisions/018-sfm-backends.md`). In `configs/README.md` leave one line: `Backend config blocks: [docs/pointcloud.md](../docs/pointcloud.md#backends).`

Check: `wc -l configs/README.md` should drop from 897 to roughly 450-500.

- [ ] **Step 4: Gates**

```bash
cd /workspace/collab-splats && git grep -n "loop_closure.yaml\|rgbd_ba.yaml" -- ':!docs/superpowers' ; echo "grep exit=$? (1 = clean)"
for g in 7scenes cross_model_chess; do $PY -m evals.eval --config evals/configs/$g.yaml --dry_run >/dev/null; echo "$g exit=$?"; done
$PY -c "import yaml; yaml.safe_load(open('configs/base.yaml')); print('base ok')"
```
Expected: `grep exit=1`, both `exit=0`, `base ok`. Then the contract gate.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git commit --only configs/loop_closure.yaml evals/configs/rgbd_ba.yaml configs/base.yaml evals/configs/7scenes.yaml evals/README.md docs/parity.md configs/README.md docs/pointcloud.md -m "chore(config): drop unused overlay and finished grid, trim configs README

- loop_closure.yaml had no caller; rgbd_ba.yaml was the finished rgbd-ba grid
- stale comments: dataset-config inheritance, BA refused with loop closure
- README: drop migration history and duplicate outputs section, fill key table,
  move backend sections to docs/pointcloud.md

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
(add `docs/superpowers/CHANGELOG.md` to `--only` if Step 3.3 touched it)

---

### Task C: User docs

**Files:**
- Delete: `docs/benchmarks/` (all 12 tracked files), `docs/known-test-failures.md`, `docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb`
- Modify: `docs/source/tutorials/index.rst:18`, `docs/README.md`, `docs/source/api/semantics.rst`, `docs/source/api/pointcloud.rst`, `docs/source/api/index.rst`, `docs/source/conf.py:72`
- Create: `docs/source/api/localization.rst`, `docs/source/api/splats.rst`

- [ ] **Step 1: Delete tracked files**

```bash
cd /workspace/collab-splats && git rm -rq docs/benchmarks docs/known-test-failures.md docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb
```

- [ ] **Step 2: Remove `02_pointcloud/colmap_sfm` line from `docs/source/tutorials/index.rst`** (line 18).

- [ ] **Step 3: Rewrite the "Module Notebooks" and "Internal" sections of `docs/README.md`**

Replace from `## Module Notebooks` to end of file with:
```markdown
## Module docs

- [pointcloud.md](pointcloud.md) — depth alignment, bundle adjustment, backend config blocks
- [mesh.md](mesh.md) — TSDF fusion, cleaning, texturing, vertex features
- [splats.md](splats.md) — Gaussian-splat training on upstream gsplat
- [parity.md](parity.md) — loop-closure provenance and parity vs VGGT-SLAM

Tutorials and the API reference build with `make docs` from `docs/source/`.
```

- [ ] **Step 4: Fix `docs/source/api/semantics.rst`**

Replace the whole file with:
```rst
Semantics
=========

Feature extraction, lifting, compression and segmentation.

.. automodule:: collab_splats.semantics.features
   :members:
   :show-inheritance:

.. automodule:: collab_splats.semantics.lifting
   :members:
   :show-inheritance:

.. automodule:: collab_splats.semantics.compression
   :members:
   :show-inheritance:

.. automodule:: collab_splats.semantics.store
   :members:
   :show-inheritance:

.. automodule:: collab_splats.semantics.segmentation
   :members:
   :show-inheritance:
```

- [ ] **Step 5: Add VGGT-Omega to `docs/source/api/pointcloud.rst`**

After the `collab_splats.pointcloud.feedforward.vggtx` block insert:
```rst
.. automodule:: collab_splats.pointcloud.feedforward.vggt_omega
   :members:
   :show-inheritance:
```

- [ ] **Step 6: Create `docs/source/api/localization.rst`**

```rst
Localization
============

Query image to camera pose in a known reconstruction: retrieval, local matching, PnP.

.. automodule:: collab_splats.localization.retrieval
   :members:
   :show-inheritance:

.. automodule:: collab_splats.localization.extractors
   :members:
   :show-inheritance:

.. automodule:: collab_splats.localization.localizer
   :members:
   :show-inheritance:

.. automodule:: collab_splats.localization.viz
   :members:
```

- [ ] **Step 7: Create `docs/source/api/splats.rst`**

```rst
Splats
======

Gaussian-splat training on upstream gsplat.

.. automodule:: collab_splats.splats.trainer
   :members:
   :show-inheritance:

.. automodule:: collab_splats.splats.checkpoint
   :members:

.. automodule:: collab_splats.splats.gaussian
   :members:
   :show-inheritance:

.. automodule:: collab_splats.splats.scaffold
   :members:
   :show-inheritance:

.. automodule:: collab_splats.splats.losses
   :members:

.. automodule:: collab_splats.splats.rendering
   :members:

.. automodule:: collab_splats.splats.cameras
   :members:
```

- [ ] **Step 8: Register pages in `docs/source/api/index.rst`** — add `localization` after `pointcloud` and `splats` after `mesh` in the toctree.

- [ ] **Step 9: Drop the duplicate `"pypose",` at `docs/source/conf.py:72`** (keep line 58). Then check `autodoc_mock_imports` covers any heavy import the new pages pull in: `grep -h "^import\|^from" collab_splats/localization/*.py collab_splats/splats/*.py | awk '{print $2}' | cut -d. -f1 | sort -u` — any top-level package not installed in a plain docs env (gsplat, vismatch, fused_ssim, nvdiffrast, …) must be in the mock list; add missing ones.

- [ ] **Step 10: Build docs**

```bash
cd /workspace/collab-splats && source /opt/venv/reconstruction/bin/activate && LC_ALL=C.UTF-8 sphinx-build -b html docs/source /tmp/claude-0/-workspace-collab-splats/cc8a22c2-008e-4c3f-abe5-15817fbee14f/scratchpad/html -q 2>&1 | grep -iE "error|failed to import|no module" | head -20; echo "exit=${PIPESTATUS[0]}"
```
Expected: no `failed to import` lines for any `automodule` target; exit 0. Notebook warnings are allowed (tutorial-rework).

- [ ] **Step 11: Remove untracked local residue**

```bash
cd /workspace/collab-splats && git status --short --ignored docs/examples docs/_build | head   # confirm all '!!' (ignored) before deleting
rm -rf docs/examples docs/_build && find docs -name __pycache__ -type d -prune -exec rm -rf {} +
```

- [ ] **Step 12: Gates** — `git grep -n "known-test-failures\|docs/benchmarks\|colmap_sfm" -- ':!docs/superpowers' ':!CLAUDE.md'` returns nothing (CLAUDE.md fixed in Task D). `scripts/test_notebooks.sh` hits are fixed in Task E. Contract gate.

- [ ] **Step 13: Commit**

```bash
cd /workspace/collab-splats && git commit --only docs/benchmarks docs/known-test-failures.md docs/source docs/README.md -m "docs: drop archived benchmarks, dev test log and colmap stub; fix API pages

- api: dead semantics.retrieval removed; localization, splats, vggt_omega added
- docs/README: links match files that exist

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task D: CLAUDE.md in-flight list

**Files:** Modify `CLAUDE.md` (the `.claude/hooks/claude-md-guard.py` hook enforces size and in-flight-only rules; let it run).

- [ ] **Step 1: Delete the six in-flight bullets**: gt-eval-harness, feedforward-import-cleanup, feedforward-mesh, docs-site, bae-vggt-parity, loma-matcher.

- [ ] **Step 2: Replace the sky-mask bullet with**
```markdown
- **sky-mask** — smp SegFormer sky masks shipped (`semantics/segmentation/sky.py`) and `mesh.mask_sky: true` is the default; open: the GH010229 mesh A/B with the torch masks ([spec](docs/superpowers/specs/2026-09-07-sky-segmentation-design.md) · [plan](docs/superpowers/plans/2026-09-07-sky-segmentation.md))
```

- [ ] **Step 3: Replace the vismatch-fork bullet with**
```markdown
- **vismatch-fork** — collab-splats pins the `BasisResearch/vismatch` `basis` branch (batched extract, `match_batch`); open: the upstream batch + COLMAP-export PRs ([spec](docs/superpowers/specs/2026-09-25-vismatch-fork-design.md))
```
Before writing, confirm the pin: `grep -n "vismatch @" pyproject.toml` shows a `basis` SHA.

- [ ] **Step 4: Add the cleanup itself as in-flight**
```markdown
- **final-cleanup** — drop dead docs/configs, fix CI workflows ([spec](docs/superpowers/specs/2026-10-08-final-docs-configs-cleanup-design.md) · [plan](docs/superpowers/plans/2026-10-08-final-docs-configs-cleanup.md))
```

- [ ] **Step 5: Delete line `Known test failures: \`docs/known-test-failures.md\``** (~L44).

- [ ] **Step 6: Verify every link in CLAUDE.md is tracked**

```bash
cd /workspace/collab-splats && grep -o "docs/superpowers/[^)]*\.md" CLAUDE.md | sort -u | while read f; do git ls-files --error-unmatch "$f" >/dev/null 2>&1 || echo "UNTRACKED $f"; done
```
Expected: no output (the plan file must be committed first — commit it with this task if Task A did not include it: `git add -f docs/superpowers/plans/2026-10-08-final-docs-configs-cleanup.md`).

- [ ] **Step 7: Commit** — `git commit --only CLAUDE.md docs/superpowers/plans/2026-10-08-final-docs-configs-cleanup.md -m "docs(claude): prune finished in-flight items ..."` with the Co-Authored-By trailer.

---

### Task E1: Make `make lint` pass (ruff)

**Files:** `collab_splats/**`, `tests/**` (mechanical), `scripts/lint.sh` unchanged.

- [ ] **Step 1: Record the reference test count before any code change**

```bash
cd /workspace/collab-splats && $PY -m pytest tests/ --collect-only -q 2>&1 | tail -1 | tee /tmp/claude-0/-workspace-collab-splats/cc8a22c2-008e-4c3f-abe5-15817fbee14f/scratchpad/collect_before.txt
```

- [ ] **Step 2: Auto-fix and list the rest**

```bash
cd /workspace/collab-splats && source /opt/venv/reconstruction/bin/activate && ruff check --fix tests/ collab_splats/ ; ruff check tests/ collab_splats/ --output-format concise
```

- [ ] **Step 3: Hand-fix remaining errors**, one rule per pass:
  - `E402` (import not at top): move the import to the top-level import block in its isort group. If it sits below a `sys.path`/env-var setup that must run first (common in `tests/conftest.py`), keep it and add `# noqa: E402` on that line instead — moving it would change behavior.
  - `F401` unused import: delete, unless the module is a package `__init__.py` re-export — then add the name to `__all__`.
  - `F841` unused variable: delete the binding (keep the call if it has side effects).
  - `E741` ambiguous name (`l`, `I`, `O`): rename to a descriptive name within that scope.

- [ ] **Step 4: Format**

```bash
cd /workspace/collab-splats && source /opt/venv/reconstruction/bin/activate && ruff format tests/ collab_splats/ && isort tests/ collab_splats/ && ruff format --diff tests/ collab_splats/ ; echo "diff exit=$?"
```
Expected `diff exit=0`. If isort and ruff format fight (re-running either changes files), report — do not loop.

- [ ] **Step 5: Prove the reformat is AST-neutral** (on the formatting-only diff, before Step 3 edits are mixed in — so commit Step 3 first as its own commit, then format):

```bash
cd /workspace/collab-splats && for f in $(git diff --name-only -- '*.py'); do $PY - "$f" <<'EOF'
import ast, subprocess, sys
f = sys.argv[1]
old = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True).stdout
new = open(f).read()
same = ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))
print(("OK  " if same else "DIFF") + " " + f)
EOF
done | grep -c "^DIFF"
```
Expected: `0`.

- [ ] **Step 6: Gates** — `ruff check tests/ collab_splats/` exits 0; contract gate; collect count equals `collect_before.txt`.

- [ ] **Step 7: Commits** — two: `fix(lint): resolve ruff errors` (Step 3 files) then `style: ruff format tests and collab_splats` (Step 4 files), each via `git commit --only`.

---

### Task E2: Make `make lint` pass (mypy)

**Files:** `pyproject.toml` (new `[tool.mypy]` section), affected `collab_splats/**` files.

- [ ] **Step 1: Tabulate errors by code**

```bash
cd /workspace/collab-splats && source /opt/venv/reconstruction/bin/activate && mypy -p collab_splats --follow-imports=skip --cache-dir /tmp/claude-0/-workspace-collab-splats/cc8a22c2-008e-4c3f-abe5-15817fbee14f/scratchpad/mypy 2>&1 | grep -o "\[[a-z-]*\]$" | sort | uniq -c | sort -rn
```

- [ ] **Step 2: Add config to `pyproject.toml`** after `[tool.isort]`:

```toml
[tool.mypy]
# Third-party model/CUDA packages ship no stubs; check our code only
ignore_missing_imports = true
disable_error_code = [<codes chosen in Step 3>]
```

- [ ] **Step 3: Choose `disable_error_code`** by this rule, and report the table + choice to the user before committing:
  - disable codes that flag annotation style, not wrong behavior: `import-untyped`, `var-annotated`, `no-redef`, `annotation-unchecked`, and `assignment` only where every hit is a `None`-initialised attribute later set to a real type (e.g. `dashboard/serve.py:53`)
  - keep codes that can mean a bug: `attr-defined` (on our own classes), `call-arg`, `arg-type`, `return-value`, `index`, `operator`, `union-attr`, `misc`

- [ ] **Step 4: Fix remaining errors in code** — annotate or correct; each fix must be type-only or a real bug fix. Real bug fixes (behavior change) are listed in the commit message and reported to the user.

- [ ] **Step 5: Gates** — `make lint` exits 0; contract gate; full suite:

```bash
cd /workspace/collab-splats && timeout 3000 $PY -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -3
```
Pass/fail/skip counts must match a run of the same command at the Task E1 Step 1 commit (record it then — memory note: `| tail` hides the exit code; read the summary line).

- [ ] **Step 6: Commit** — `git commit --only pyproject.toml <fixed files> -m "fix(lint): mypy config and type fixes ..."`.

---

### Task E3: Workflows

**Files:** `.github/workflows/{docs,lint,lint_notebooks,test,test_notebooks}.yml`, `scripts/test_notebooks.sh`

- [ ] **Step 1: Python 3.11 everywhere**

```bash
cd /workspace/collab-splats && sed -i "s/python-version: \['3.10'\]/python-version: ['3.11']/; s/python-version: \"3.10\"/python-version: \"3.11\"/; s/Set up Python 3.10/Set up Python 3.11/" .github/workflows/*.yml && grep -n "3\.1[01]" .github/workflows/*.yml
```
Expected: only `3.11` hits.

- [ ] **Step 2: `docs.yml`** — the install step must cover the Sphinx extensions `conf.py` uses: `grep -n "extensions" -A12 docs/source/conf.py`; add any missing package (e.g. `sphinxcontrib-bibtex`) to the `pip install` line. The build imports `collab_splats` via autodoc: add `pip install numpy` plus whatever Task C Step 10 showed is imported unmocked, OR extend `autodoc_mock_imports` — prefer mocking.

- [ ] **Step 3: `test.yml` and `test_notebooks.yml`** — replace each job body from `runs-on` through the "Install collab-splats" step with:

```yaml
    runs-on: [self-hosted, gpu]

    steps:
      - uses: actions/checkout@v4

      # Runner image already holds the full env (bash setup.sh); use it as-is
      - name: Use project venv
        run: echo "/opt/venv/reconstruction/bin" >> "$GITHUB_PATH"
```
Delete the `strategy.matrix`, pip cache, setup-python, numpy<2, torch 2.1.2, and CPU-build-env steps. Keep the final `make test` / `make test-notebooks` step unchanged.

- [ ] **Step 4: `lint.yml` / `lint_notebooks.yml`** — the pip install list must include `isort` with the repo's config dependencies; keep as is apart from 3.11. Run their make targets locally:

```bash
cd /workspace/collab-splats && source /opt/venv/reconstruction/bin/activate && make lint; echo "lint=$?"; make lint-notebooks; echo "nblint=$?"
```
`make lint-notebooks` runs `nbqa isort`, which rewrites notebooks in place. After it runs, inspect `git diff --stat docs/source/tutorials` and commit those import-order fixes as their own commit. Remaining `nbqa black --check` / `flake8` failures: fix in the notebook cells (code-only, no output changes).

- [ ] **Step 5: `scripts/test_notebooks.sh`** — replace the stale `EXCLUDED_NOTEBOOKS` entries with
```bash
EXCLUDED_NOTEBOOKS=(
    "docs/source/tutorials/evals/ground_truth_evals.ipynb"  # reads gitignored evals/results/
)
```
and change `INCLUDED_NOTEBOOKS="docs/"` to `INCLUDED_NOTEBOOKS="docs/source/tutorials/"` so pytest no longer collects non-notebook files under `docs/`.

- [ ] **Step 6: Validate YAML syntax**

```bash
cd /workspace/collab-splats && for f in .github/workflows/*.yml; do $PY -c "import yaml,sys; yaml.safe_load(open('$f')); print('ok $f')"; done
```

- [ ] **Step 7: Commit** — `git commit --only .github/workflows scripts/test_notebooks.sh <any notebooks> -m "ci: python 3.11, GPU jobs on self-hosted runner, current notebook paths"`.

---

### Task F: Close out

- [ ] **Step 1:** Append a `final-cleanup` entry to `docs/superpowers/CHANGELOG.md` (newest first, same format as neighbors): what landed, commit SHAs, the two open items (self-hosted runner registration; `test_notebooks` red until tutorial-rework).
- [ ] **Step 2:** Move `final-cleanup` from CLAUDE.md "In-Flight Work" to "Recently Completed" (keep five newest there).
- [ ] **Step 3:** `graphify update .` (AST-only).
- [ ] **Step 4:** Commit `docs/superpowers/CHANGELOG.md CLAUDE.md` with `--only`. Do not push; report to the user.
