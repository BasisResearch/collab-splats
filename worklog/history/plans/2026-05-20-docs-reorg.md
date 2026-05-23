# Docs Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `docs/` (especially `docs/superpowers/`) into a clear separation of history / current state / future plans / architecture decisions per spec `worklog/history/specs/2026-05-20-docs-reorg-design.md`.

**Architecture:** Single big-bang commit. Create scaffolding (`STATE.md`, `ROADMAP.md`, `decisions/006-012`, `history/` dirs), `git mv` archived files (preserves blame), trim WORKLOG, update CLAUDE.md, run cross-ref scan, commit.

**Tech Stack:** git, shell (mv, grep, sed, ls), markdown.

---

## Spec Reference

All tasks implement `worklog/history/specs/2026-05-20-docs-reorg-design.md`. Section names below match spec section headings.

## File Structure

```
docs/
├── ROADMAP.md                  (new)
├── decisions/                  (existing, gain 7 new + 1 rename)
│   ├── 001-declip-integration.md      (renamed)
│   ├── 002-005-*.md                   (unchanged)
│   └── 006-012-*.md                   (new — 7 files)
├── history/                    (new)
│   ├── specs/   (58 git-mv'd)
│   ├── plans/   (41 git-mv'd)
│   └── prs/     (2 git-mv'd)
├── superpowers/
│   ├── WORKLOG.md              (trimmed in place)
│   ├── STATE.md                (new)
│   ├── known-test-failures.md  (unchanged)
│   ├── notes/                  (new, 3 git-mv'd files)
│   ├── specs/   (3 active stay; rest mv'd)
│   └── plans/   (3 active stay; rest mv'd)
```

CLAUDE.md edited at project root.

---

## Task 0: Verify Pre-Conditions

**Files:** none modified.

- [ ] **Step 1: Confirm on `refactor/core-modules` and clean tree**

```bash
git rev-parse --abbrev-ref HEAD
git status --short
```

Expected: branch is `refactor/core-modules`. Working tree may have unrelated changes (eval_results/, *.ipynb edits per gitStatus); none in `docs/superpowers/{specs,plans,prs}/` or `worklog/decisions/`.

- [ ] **Step 2: Confirm spec exists**

```bash
ls worklog/history/specs/2026-05-20-docs-reorg-design.md
```

Expected: file listed.

- [ ] **Step 3: Confirm starting file counts match spec inventory**

```bash
ls worklog/history/specs/ | wc -l   # expect 61
ls worklog/history/plans/ | wc -l   # expect 44
ls worklog/history/prs/ | wc -l     # expect 2
ls worklog/decisions/ | wc -l           # expect 5
```

If any count is off, stop and reconcile with spec inventory before continuing.

---

## Task 1: Create Scaffolding Directories

**Files:**
- Create: `worklog/history/specs/.gitkeep`
- Create: `worklog/history/plans/.gitkeep`
- Create: `worklog/history/prs/.gitkeep`
- Create: `worklog/notes/.gitkeep`

- [ ] **Step 1: Make directories**

```bash
mkdir -p worklog/history/specs worklog/history/plans worklog/history/prs docs/superpowers/notes
```

- [ ] **Step 2: Add `.gitkeep` placeholders**

Use Write tool to create empty `.gitkeep` in each. These get auto-removed once files are moved into the dirs.

- [ ] **Step 3: Verify**

```bash
ls -d worklog/history/specs worklog/history/plans worklog/history/prs docs/superpowers/notes
```

Expected: all four dirs listed.

---

## Task 2: Rename ADR 001

**Files:**
- Rename: `worklog/decisions/001-feat_declip-integration.md` → `worklog/decisions/001-declip-integration.md`

- [ ] **Step 1: `git mv`**

```bash
git mv worklog/decisions/001-feat_declip-integration.md worklog/decisions/001-declip-integration.md
```

- [ ] **Step 2: Update cross-refs to old name**

```bash
grep -rln '001-feat_declip-integration' . --exclude-dir=.git --exclude-dir=docs/history
```

For each hit, use Edit tool to replace `001-feat_declip-integration` → `001-declip-integration`.

- [ ] **Step 3: Verify**

```bash
grep -rn '001-feat_declip-integration' . --exclude-dir=.git --exclude-dir=docs/history
```

Expected: empty output.

---

## Task 3: Write ADR 006 — Single Working Branch

**Files:**
- Create: `worklog/decisions/006-single-working-branch.md`
- Source: `worklog/history/specs/2026-04-21-single-branch-strategy-design.md` + `WORKLOG.md` § Architecture Decisions → "Single working branch"

- [ ] **Step 1: Read source spec**

```bash
cat worklog/history/specs/2026-04-21-single-branch-strategy-design.md
```

- [ ] **Step 2: Write ADR**

Use Write tool. Content must follow the template:

```markdown
# ADR 006: Single Working Branch Strategy

**Status:** Accepted
**Date:** 2026-04-21
**Tags:** branch-strategy, refactor, workflow

## Context
Refactor work originally planned as stacked PRs (core-modules → semantics → dashboard → loop-closure). Rebase friction between stacked branches grew costly: every fix to a base branch required cascading rebases across descendants. Multiple working branches also fragmented the task list and made it unclear which branch held the canonical state of any given module.

## Decision
All refactor work lands on a single working branch `refactor/core-modules`. One PR to `main` with sectioned review (semantics / pointcloud / dashboard / utils / loop-closure). Commit prefixes tag scope: `refactor(core):`, `feat(dashboard):`, `fix(utils):`, etc. Other branches (`refactor/semantics`, `dashboard`, `refactor/dashboard-optical-flow`, `feat/loop-closure`) get absorbed via cherry-pick or squash-merge.

## Consequences
**Positive:**
- No rebase cascades.
- Single source of truth for in-flight state.
- Reviewer sees one coherent diff.

**Negative:**
- The single PR is large. Reviewer cost is concentrated in one sitting.
- Cannot ship a sub-section to `main` independently before the rest is ready.

**Revisit if:** any subsystem becomes large enough that the PR exceeds reviewable size (~10k LoC), or if multiple unrelated workstreams need to ship at different cadences.

## Alternatives Considered
- **Stacked PRs.** Rejected: rebase friction outweighed reviewability benefit.
- **One PR per module to `main` directly.** Rejected: dependencies between modules made ordering brittle.
```

- [ ] **Step 3: Verify**

```bash
test -f worklog/decisions/006-single-working-branch.md && head -3 worklog/decisions/006-single-working-branch.md
```

Expected: file exists, title line shown.

---

## Task 4: Write ADR 007 — Dashboard as Integration Harness

**Files:**
- Create: `worklog/decisions/007-dashboard-as-integration-harness.md`
- Source: `WORKLOG.md` § Architecture Decisions → "Dashboard = integration test harness"

- [ ] **Step 1: Write ADR**

Use Write tool. Content:

```markdown
# ADR 007: Dashboard as Integration Test Harness

**Status:** Accepted
**Date:** 2026-04-20
**Tags:** dashboard, testing, integration

## Context
Unit tests verify API contracts (one function, one path, one assertion). They do not catch problems where the contract is satisfied but the full pipeline produces visually wrong output: misaligned features, broken queries, wrong segmentation, drifted poses. Full end-to-end pipeline tests via CI are slow and brittle (dataset assumptions, hardware variability).

## Decision
The dashboard is treated as the integration test harness for the whole pipeline. Any subsystem change that ships also lands in the dashboard, where it is exercised against representative data and inspected visually before merge. Unit tests cover API contracts; the dashboard covers "does the pipeline still feel right."

## Consequences
**Positive:**
- Catches integration-level regressions that unit tests miss.
- Forces every refactor to remain runnable end-to-end.
- Doubles as developer documentation — running the dashboard demonstrates the full feature set.

**Negative:**
- Dashboard breakage becomes a release blocker.
- Requires GPU + dataset availability to validate.

**Revisit if:** a faster automated end-to-end harness (e.g. golden-output regression suite) supersedes the manual visual check.

## Alternatives Considered
- **CI-only integration tests.** Rejected: dataset + GPU requirements make them brittle in CI.
- **Skip integration testing.** Rejected: too many silent regressions slipped through unit tests alone.
```

- [ ] **Step 2: Verify**

```bash
test -f worklog/decisions/007-dashboard-as-integration-harness.md && head -3 worklog/decisions/007-dashboard-as-integration-harness.md
```

---

## Task 5: Write ADR 008 — MapAnything Stub Pattern

**Files:**
- Create: `worklog/decisions/008-mapanything-stub-pattern.md`

- [ ] **Step 1: Write ADR**

Use Write tool. Content:

```markdown
# ADR 008: MapAnything Stub Pattern

**Status:** Accepted
**Date:** 2026-04-20
**Tags:** mapanything, optional-deps, dashboard

## Context
MapAnything is an optional feedforward backend with heavy CUDA dependencies. Requiring it for the dashboard and main pipeline would block every developer who has not installed the optional environment. Dynamic feature flags in code paths multiply complexity. We want the dashboard to launch even when MapAnything is unavailable, while keeping the UI plumbing for it in place.

## Decision
A try/except import guard at module level produces a boolean `_MAPANYTHING_AVAILABLE`. The dashboard MapAnything tab and pointcloud option are visible but disabled when False. Tests that depend on MapAnything use `pytest.importorskip("mapanything")`.

```python
try:
    from collab_splats.pointcloud import MapAnythingCreator
    _MAPANYTHING_AVAILABLE = True
except ImportError:
    _MAPANYTHING_AVAILABLE = False
```

## Consequences
**Positive:**
- Dashboard launches without MapAnything installed.
- Pattern is reusable for any optional backend (vggtx, hloc-extras, etc.).
- Stub is the development seam — feature unlocks automatically when env has the dep.

**Negative:**
- Requires care to keep stubbed code paths in sync with active ones.
- "Disabled" UI elements can confuse users unaware of the optional dep.

**Revisit if:** MapAnything becomes a mandatory dependency, or if multiple optional backends create a tangle of `_FOO_AVAILABLE` flags requiring a registry approach.

## Alternatives Considered
- **Hard dependency.** Rejected: blocks developers without the optional env.
- **Plugin registry with lazy loading.** Deferred — adopt only if N>2 optional backends appear.
- **Separate dashboard build flavors.** Rejected: maintenance cost too high.
```

- [ ] **Step 2: Verify**

```bash
test -f worklog/decisions/008-mapanything-stub-pattern.md && head -3 worklog/decisions/008-mapanything-stub-pattern.md
```

---

## Task 6: Write ADR 009 — `frame_sampling` in `utils/`

**Files:**
- Create: `worklog/decisions/009-frame-sampling-in-utils.md`

- [ ] **Step 1: Write ADR**

Use Write tool. Content:

```markdown
# ADR 009: `frame_sampling` Lives in `utils/`, Not `semantics/`

**Status:** Accepted
**Date:** 2026-04-20
**Tags:** module-boundaries, utils, semantics

## Context
`frame_sampling` was originally placed in `collab_splats/semantics/` because that was where it was first consumed (optical-flow-based frame selection for semantic feature extraction). Subsequent integrations (dashboard preview, pointcloud preprocessing, mesh generation) all needed the same utility. Importing from `semantics/` for non-semantic consumers created a misleading dependency arrow: pointcloud should not depend on semantics.

## Decision
Canonical home is `collab_splats/utils/frame_sampling.py`. A re-export shim at `collab_splats/semantics/frame_sampling.py` preserves backwards-compat for the original import path. Test moved to `tests/utils/test_frame_sampling.py`.

## Consequences
**Positive:**
- Module dependency graph cleaner: pointcloud / dashboard / mesh import from `utils/`, not `semantics/`.
- `utils/` is the correct conceptual home for general preprocessing.

**Negative:**
- Two import paths for the same symbol during the transition.
- Future readers may be confused by the shim; comment it as transitional.

**Revisit if:** the shim is no longer used by any caller (drop it then).

## Alternatives Considered
- **Leave it in `semantics/`.** Rejected: misleads readers about layering.
- **Move and break compat.** Rejected: cherry-pick risk during the single-branch absorption.
```

- [ ] **Step 2: Verify**

```bash
test -f worklog/decisions/009-frame-sampling-in-utils.md && head -3 worklog/decisions/009-frame-sampling-in-utils.md
```

---

## Task 7: Write ADR 010 — Pointcloud Creator Registry

**Files:**
- Create: `worklog/decisions/010-pointcloud-creator-registry.md`
- Source: `worklog/history/specs/2026-04-16-pointcloud-submodule-design.md` (registry section)

- [ ] **Step 1: Read source spec for registry contract details**

```bash
grep -n -i 'registry\|creator\|colmap\|hloc\|vggtx\|mapanything' worklog/history/specs/2026-04-16-pointcloud-submodule-design.md | head -30
```

- [ ] **Step 2: Write ADR**

Use Write tool. Content:

```markdown
# ADR 010: Pointcloud Creator Registry

**Status:** Accepted
**Date:** 2026-04-16
**Tags:** pointcloud, registry, sfm, feedforward

## Context
Pointcloud reconstruction supports multiple backends with different requirements: COLMAP (binary, classical SfM), hloc (Python wrapper around classical SfM with learned features), VGGT-X (feedforward transformer, GPU), MapAnything (feedforward, optional dep). Each backend has different output paths, transform conventions, and progress reporting. Without a unifying interface, consumers (dashboard, pipeline scripts, eval harness) end up with backend-specific branches.

## Decision
A registry `{"colmap": ColmapCreator, "hloc": HlocCreator, "vggtx": VGGTXCreator, "mapanything": MapAnythingCreator}` keyed by short string. All creators subclass `BasePointcloudCreator` with method `reconstruct(images, output_dir) -> PointcloudResult`. Backends decide their own world coordinate convention internally; the `CoordinateFrame` enum and `world_transform` field on the result expose it (see ADR 011).

## Consequences
**Positive:**
- Switching backends is a one-line config change.
- Eval harness can iterate over the full backend set without per-backend code paths.
- Adding a new backend is a contained change: implement subclass + register.

**Negative:**
- Forces all backends through the same interface — leaky abstractions when a backend produces something the contract doesn't model (e.g. dense depth maps).
- Registry hard-codes the string keys; rename is a multi-call-site change.

**Revisit if:** the contract surface grows to include backend-specific features that pollute the base class.

## Alternatives Considered
- **Per-backend top-level entry points.** Rejected: forces every consumer to know about every backend.
- **Factory function with kwargs.** Rejected: harder to enumerate backends.
- **Plugin system with entry points.** Deferred: registry is enough at current backend count.
```

- [ ] **Step 3: Verify**

```bash
test -f worklog/decisions/010-pointcloud-creator-registry.md && head -3 worklog/decisions/010-pointcloud-creator-registry.md
```

---

## Task 8: Write ADR 011 — `CoordinateFrame` Enum

**Files:**
- Create: `worklog/decisions/011-coordinate-frame-enum.md`
- Source: `worklog/history/specs/2026-04-20-pointcloud-feedforward-design.md`

- [ ] **Step 1: Read source spec section**

```bash
grep -n -A 5 'CoordinateFrame\|world_transform' worklog/history/specs/2026-04-20-pointcloud-feedforward-design.md | head -50
```

- [ ] **Step 2: Write ADR**

Use Write tool. Content:

```markdown
# ADR 011: `CoordinateFrame` Enum + `world_transform`

**Status:** Accepted
**Date:** 2026-04-20
**Tags:** pointcloud, coordinate-frames, sfm

## Context
Each pointcloud backend produces points and poses in its own native coordinate convention: COLMAP world frame is z-up but origin-arbitrary; VGGT-X uses a normalized scene-centered frame; MapAnything assumes a metric reconstruction with a derived world frame. Downstream stages (mesh, dashboard render, eval against ground truth) require points in a known frame. Hard-coding conversions per backend at every consumer is brittle and easily inconsistent.

## Decision
A `CoordinateFrame` enum names the source convention (`COLMAP`, `VGGTX`, `MAPANYTHING`, `METRIC_WORLD`, …). Every `PointcloudResult` carries a `frame: CoordinateFrame` field plus a `world_transform: SE3` that maps from `frame` to the canonical metric world frame. Consumers that need world-frame points apply `world_transform`; consumers that need native frame use the points as-is.

## Consequences
**Positive:**
- Self-describing results — a `PointcloudResult` is interpretable without out-of-band knowledge.
- One conversion logic per backend instead of N consumer-side branches.
- Eval against ground truth becomes uniform.

**Negative:**
- Every backend must populate `world_transform` correctly. Mistakes here propagate silently.
- Adding a new frame requires a new enum variant and corresponding conversion.

**Revisit if:** backends start producing multi-frame outputs (e.g. dense scene + sparse landmarks in different frames) — the single-frame field will no longer be enough.

## Alternatives Considered
- **Force all backends to a canonical frame at creation time.** Rejected: backend-internal optimization stages benefit from native frame.
- **String frame identifiers instead of enum.** Rejected: typo risk + no IDE completion.
```

- [ ] **Step 3: Verify**

```bash
test -f worklog/decisions/011-coordinate-frame-enum.md && head -3 worklog/decisions/011-coordinate-frame-enum.md
```

---

## Task 9: Write ADR 012 — hloc Direct Call

**Files:**
- Create: `worklog/decisions/012-hloc-direct-call.md`
- Source: `worklog/history/specs/2026-04-20-pointcloud-feedforward-design.md` (hloc section)

- [ ] **Step 1: Write ADR**

Use Write tool. Content:

```markdown
# ADR 012: hloc Direct Call (No Nerfstudio Dependency)

**Status:** Accepted
**Date:** 2026-04-20
**Tags:** hloc, sfm, nerfstudio, dependencies

## Context
The original `HlocCreator` invoked hloc indirectly through `nerfstudio`'s `process_data` wrappers. This pulled the entire nerfstudio dependency tree into a code path that only needed hloc itself. It also made it harder to control hloc's output layout (nerfstudio reorganized files), which complicated downstream consumers expecting `colmap/sparse/0/` format.

## Decision
`HlocCreator` calls hloc directly via `hloc.extract_features` / `hloc.match_features` / `hloc.reconstruction` and writes COLMAP-compatible binaries to `<output_dir>/colmap/sparse/0/`. No nerfstudio import. The creator owns the full pipeline orchestration.

## Consequences
**Positive:**
- Pointcloud submodule has no nerfstudio dependency; importable from any environment with just hloc.
- Output layout matches `ColmapCreator` exactly — consumers cannot tell which backend produced the result.
- One less indirection layer to debug when hloc misbehaves.

**Negative:**
- Code duplicates orchestration logic that nerfstudio already had (feature extraction → matching → reconstruction sequence).
- Future hloc API changes hit us directly instead of being absorbed by nerfstudio's wrapper.

**Revisit if:** nerfstudio's `process_data` adds capabilities (e.g. learned matchers, automatic config tuning) that we want to reuse.

## Alternatives Considered
- **Keep nerfstudio indirection.** Rejected: dependency cost outweighed orchestration savings.
- **Fork nerfstudio's wrapper into our tree.** Rejected: hidden coupling to nerfstudio internals.
```

- [ ] **Step 2: Verify**

```bash
test -f worklog/decisions/012-hloc-direct-call.md && head -3 worklog/decisions/012-hloc-direct-call.md
```

---

## Task 10: Write `worklog/ROADMAP.md`

**Files:**
- Create: `worklog/ROADMAP.md`
- Source: `WORKLOG.md` § Phase 2 + Phase 3 + Deferred/Parked; `worklog/history/specs/2026-04-16-refactor-roadmap.md` (folded then archived in later task).

- [ ] **Step 1: Read source content**

```bash
sed -n '154,310p' worklog/WORKLOG.md
cat worklog/history/specs/2026-04-16-refactor-roadmap.md
```

- [ ] **Step 2: Write ROADMAP**

Use Write tool. Content follows the spec's "ROADMAP.md Shape" section exactly:

```markdown
# Roadmap

## What's Built

- **Semantics**: ANN feature splatting, MaskCLIP + Talk2DINO, batched extraction, dashboard query API.
  Key ADRs: [009 — frame_sampling in utils](decisions/009-frame-sampling-in-utils.md).
- **Pointcloud**: creator registry (colmap / hloc / vggtx / mapanything), `CoordinateFrame` world transform, hloc direct call.
  Key ADRs: [010 — creator registry](decisions/010-pointcloud-creator-registry.md), [011 — CoordinateFrame](decisions/011-coordinate-frame-enum.md), [012 — hloc direct](decisions/012-hloc-direct-call.md).
- **Loop Closure**: Sim(3) pose graph, geometric overlap gate, 3-way noise split, Huber robustification.
  Key ADRs: [002](decisions/002-defer-sl4.md), [003](decisions/003-defer-graphmap.md), [004](decisions/004-defer-frametracker.md), [005](decisions/005-defer-per-backend-noise-tuning.md).
- **Mesh**: TSDF integration from `PointcloudResult`; feedforward → direct mesh path.
- **Dashboard**: integration test harness — full pipeline visual validation.
  Key ADRs: [007 — dashboard as harness](decisions/007-dashboard-as-integration-harness.md), [008 — MapAnything stub](decisions/008-mapanything-stub-pattern.md).
- **BA**: PyPose BAE bundle adjustment, CO3Dv2 eval baseline.
- **Branch strategy**: single working branch.
  Key ADR: [006](decisions/006-single-working-branch.md).

## Where We Are

- Sole working branch: `refactor/core-modules` (~82 commits ahead of `main`).
- Final PR before merge to `main`.
- Active in-flight: `gt-eval-harness`, `feedforward-import-cleanup`, `feedforward-mesh`.
- Phase 1 (core modules + loop closure + BA) near complete.

## What's Next

### Phase 2 — Dashboard Iteration
Goal: harden dashboard as the canonical integration surface. Add golden-image regression captures; auto-launch on PR with representative dataset; surface per-subsystem health indicators.

### Phase 3 — Feature Extension
Goal: extend semantic feature space beyond MaskCLIP + Talk2DINO. Candidate: DINOv3, SAM2 panoptic masks for grouping, learned per-Gaussian features distilled from open-vocabulary backbones.

### Candidate Features (unprioritized)
- DeCLIP integration (see [ADR 001](decisions/001-declip-integration.md)).
- Multi-view consistent feature distillation across submaps.
- Live capture → reconstruction loop (no offline preprocessing).

### Parked Branches
- `tlb-grouping-segmentation` — Gaussian grouping + per-instance segmentation; separate plan TBD; waiting on stable feature splatting API.
- `tlb-improve-splatter` — Splatter wrapper rework; separate plan TBD; deferred until core refactor lands on `main`.
```

- [ ] **Step 3: Verify**

```bash
test -f worklog/ROADMAP.md && wc -l worklog/ROADMAP.md && grep -c '^## ' worklog/ROADMAP.md
```

Expected: file exists, ≥30 lines, three top-level `##` sections (What's Built / Where We Are / What's Next).

---

## Task 11: Write `worklog/STATE.md`

**Files:**
- Create: `worklog/STATE.md`

- [ ] **Step 1: Determine in-flight task statuses**

For each active spec/plan, identify the current status. Read each spec's header for hints; if status unclear, leave as `wip`.

```bash
head -5 worklog/specs/2026-05-07-gt-eval-harness-design.md
head -5 worklog/specs/2026-05-08-feedforward-import-cleanup-design.md
head -5 worklog/specs/2026-05-14-feedforward-mesh-design.md
```

- [ ] **Step 2: Write STATE.md**

Use Write tool. Content:

```markdown
# State — last-updated 2026-05-20

## Active Branch
| Branch | Status | Ahead of main | Notes |
|---|---|---|---|
| `refactor/core-modules` | active | ~82 commits | sole working branch — see [ADR 006](../decisions/006-single-working-branch.md) |

## In-Flight Work
- **gt-eval-harness** — [spec](specs/2026-05-07-gt-eval-harness-design.md) · [plan](plans/2026-05-07-gt-eval-harness.md) — status: wip
- **feedforward-import-cleanup** — [spec](specs/2026-05-08-feedforward-import-cleanup-design.md) · [plan](plans/2026-05-08-feedforward-import-cleanup.md) — status: wip
- **feedforward-mesh** — [spec](specs/2026-05-14-feedforward-mesh-design.md) · [plan](plans/2026-05-14-feedforward-mesh.md) — status: wip

## Open Blockers
- nerfstudio env: see [notes/2026-05-03-feedforward-env-debug](../history/specs/2026-05-03-feedforward-env-debug.md) if relevant; check whether the symptom is current.
- Known test failures: see [known-test-failures.md](known-test-failures.md).

## Parked
| Item | Why parked | Pointer |
|---|---|---|
| `tlb-grouping-segmentation` | separate plan TBD | [ROADMAP](../ROADMAP.md#parked-branches) |
| `tlb-improve-splatter` | separate plan TBD | [ROADMAP](../ROADMAP.md#parked-branches) |

## Recent Decisions (last 30 days)
- [ADR 012 — hloc direct call](../decisions/012-hloc-direct-call.md)
- [ADR 011 — CoordinateFrame enum](../decisions/011-coordinate-frame-enum.md)
- [ADR 010 — pointcloud creator registry](../decisions/010-pointcloud-creator-registry.md)
- [ADR 009 — frame_sampling in utils](../decisions/009-frame-sampling-in-utils.md)
- [ADR 008 — MapAnything stub pattern](../decisions/008-mapanything-stub-pattern.md)
- [ADR 007 — dashboard as integration harness](../decisions/007-dashboard-as-integration-harness.md)
- [ADR 006 — single working branch](../decisions/006-single-working-branch.md)
```

- [ ] **Step 3: Verify**

```bash
test -f worklog/STATE.md && wc -l worklog/STATE.md
```

Expected: file exists, <60 lines (one-page constraint).

---

## Task 12: `git mv` Loose Investigation Notes → `notes/`

**Files:**
- Move: `docs/superpowers/2026-05-08-ba-eval-script-split-summary.md` → `worklog/notes/`
- Move: `docs/superpowers/2026-05-08-bundle-adjustment-notebook-debug.md` → `worklog/notes/`
- Move: `docs/superpowers/2026-05-08-co3dv2-eval-gap-investigation.md` → `worklog/notes/`

- [ ] **Step 1: Move files**

```bash
git mv docs/superpowers/2026-05-08-ba-eval-script-split-summary.md worklog/notes/
git mv docs/superpowers/2026-05-08-bundle-adjustment-notebook-debug.md worklog/notes/
git mv docs/superpowers/2026-05-08-co3dv2-eval-gap-investigation.md worklog/notes/
```

- [ ] **Step 2: Remove `.gitkeep` (no longer needed once dir has content)**

```bash
git rm worklog/notes/.gitkeep
```

- [ ] **Step 3: Verify**

```bash
ls worklog/notes/
ls docs/superpowers/2026-05-08-*.md 2>/dev/null && echo "ERROR: loose notes remain" || echo "ok"
```

Expected: three files in `notes/`, no loose `2026-05-08-*.md` at superpowers root.

---

## Task 13: `git mv` Archive Specs

**Files:**
- Move 58 specs from `worklog/history/specs/` to `worklog/history/specs/`.
- Keep in place: 2026-05-07-gt-eval-harness-design.md, 2026-05-08-feedforward-import-cleanup-design.md, 2026-05-14-feedforward-mesh-design.md, **and** 2026-05-20-docs-reorg-design.md (this design itself).

- [ ] **Step 1: Build move list**

Active set: `gt-eval-harness`, `feedforward-import-cleanup`, `feedforward-mesh`, `docs-reorg`. Everything else moves.

```bash
# Preview what will move
for f in worklog/history/specs/*.md; do
  bn=$(basename "$f")
  case "$bn" in
    2026-05-07-gt-eval-harness-design.md|\
    2026-05-08-feedforward-import-cleanup-design.md|\
    2026-05-14-feedforward-mesh-design.md|\
    2026-05-20-docs-reorg-design.md)
      ;;
    *)
      echo "MOVE: $f -> worklog/history/specs/$bn"
      ;;
  esac
done | wc -l
```

Expected: 58. (Note: 2026-04-16-refactor-roadmap.md is in this set — folded into ROADMAP earlier, content now duplicated; archival preserves the original.)

- [ ] **Step 2: Execute moves**

```bash
for f in worklog/history/specs/*.md; do
  bn=$(basename "$f")
  case "$bn" in
    2026-05-07-gt-eval-harness-design.md|\
    2026-05-08-feedforward-import-cleanup-design.md|\
    2026-05-14-feedforward-mesh-design.md|\
    2026-05-20-docs-reorg-design.md)
      ;;
    *)
      git mv "$f" "worklog/history/specs/$bn"
      ;;
  esac
done
```

- [ ] **Step 3: Remove `.gitkeep`**

```bash
git rm worklog/history/specs/.gitkeep
```

- [ ] **Step 4: Verify**

```bash
ls worklog/history/specs/ | wc -l   # expect 4
ls worklog/history/specs/ | wc -l       # expect 58
ls worklog/history/specs/            # confirm the 4 active files
```

---

## Task 14: `git mv` Archive Plans

**Files:**
- Move 41 plans from `worklog/history/plans/` to `worklog/history/plans/`.
- Keep in place: 2026-05-07-gt-eval-harness.md, 2026-05-08-feedforward-import-cleanup.md, 2026-05-14-feedforward-mesh.md, **and** 2026-05-20-docs-reorg.md (this implementation plan).

- [ ] **Step 1: Execute moves**

```bash
for f in worklog/history/plans/*.md; do
  bn=$(basename "$f")
  case "$bn" in
    2026-05-07-gt-eval-harness.md|\
    2026-05-08-feedforward-import-cleanup.md|\
    2026-05-14-feedforward-mesh.md|\
    2026-05-20-docs-reorg.md)
      ;;
    *)
      git mv "$f" "worklog/history/plans/$bn"
      ;;
  esac
done
```

- [ ] **Step 2: Remove `.gitkeep`**

```bash
git rm worklog/history/plans/.gitkeep
```

- [ ] **Step 3: Verify**

```bash
ls worklog/history/plans/ | wc -l   # expect 4
ls worklog/history/plans/ | wc -l       # expect 41
ls worklog/history/plans/            # confirm the 4 active files
```

---

## Task 15: `git mv` PR Bodies

**Files:**
- Move: `worklog/history/prs/pr1-core-modules.md` → `worklog/history/prs/`
- Move: `worklog/history/prs/pr2-dashboard-complete.md` → `worklog/history/prs/`

- [ ] **Step 1: Move files**

```bash
git mv worklog/history/prs/pr1-core-modules.md worklog/history/prs/
git mv worklog/history/prs/pr2-dashboard-complete.md worklog/history/prs/
```

- [ ] **Step 2: Remove now-empty `prs/` dir at superpowers root + `.gitkeep`**

```bash
rmdir docs/superpowers/prs 2>/dev/null || true
git rm worklog/history/prs/.gitkeep
```

- [ ] **Step 3: Verify**

```bash
ls worklog/history/prs/        # expect 2 files
test -d docs/superpowers/prs && echo "ERROR: dir still exists" || echo "ok"
```

---

## Task 16: Trim `WORKLOG.md`

**Files:**
- Modify: `worklog/WORKLOG.md`

- [ ] **Step 1: Read current WORKLOG to map line ranges**

```bash
grep -n '^##\|^# ' worklog/WORKLOG.md
wc -l worklog/WORKLOG.md
```

Identify:
- Line 1: `# collab-splats Refactor Worklog`
- Line ~186: `## Session Log`
- Lines after: session entries (newest first or reverse-chronological — keep existing order).
- Everything between line 1 and `## Session Log` is non-session content to remove (Current State, Architecture Decisions, Core Modules, Dashboard, Semantics module, Pointcloud, Dashboard, Utils & wrapper, Config & packaging, Cleanup, Post-merge Cleanup, Phase 2, Phase 3).
- Lines after Session Log entries: `## Deferred / Parked` — also remove (content lives in ROADMAP).

- [ ] **Step 2: Replace WORKLOG with trimmed version**

Use Write tool. New content = original header (line 1, plus any intro paragraph if present) + `## Session Log` section + all existing dated `### YYYY-MM-DD` entries verbatim (preserve content; only the surrounding non-session sections go).

Concrete structure:

```markdown
# collab-splats Refactor Worklog

> **Append-only.** Newest entries at top of Session Log. See:
> - [STATE.md](STATE.md) for current state (branches, in-flight, blockers, parked)
> - [../ROADMAP.md](../ROADMAP.md) for future phases + architecture overview
> - [../decisions/](../decisions/) for ADRs
>
> Entry template:
> ```
> ### YYYY-MM-DD <optional tag>
> - **Focus:** one line
> - **Did:** bullets
> - **Decided:** bullets (link ADRs: `→ ADR NNN`)
> - **Hit:** blockers / surprises / reversals
> - **Files:** key paths
> - **Next:** one line
> ```

## Session Log

<existing dated entries pasted verbatim, in current order>
```

- [ ] **Step 3: Verify**

```bash
wc -l worklog/WORKLOG.md
grep -c '^### 2026' worklog/WORKLOG.md
grep -E '^## (Current State|Architecture Decisions|Core Modules|Phase 2|Phase 3|Deferred)' worklog/WORKLOG.md
```

Expected:
- Line count substantially reduced (was 307).
- All original `### 2026-…` dated entries still present.
- Last grep returns 0 hits (non-session sections gone).

---

## Task 17: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (project root)

- [ ] **Step 1: Locate lines to edit**

```bash
grep -n 'WORKLOG\|superpowers\|docs/' CLAUDE.md | head
```

Current line 7–8 reference WORKLOG.md.

- [ ] **Step 2: Edit**

Use Edit tool. Replace:

```
- At the start of every conversation always call /brainstorming -- use superpowers to accomplish tasks.
- Before anything, look through superpowers/WORKLOG.md
```

With:

```
- At the start of every conversation always call /brainstorming -- use superpowers to accomplish tasks.
- Before anything, read worklog/STATE.md (current state), then latest entries in worklog/WORKLOG.md, then worklog/decisions/NNN-*.md as referenced.
- Active spec/plan for in-flight work lives in docs/superpowers/{specs,plans}/. Completed work is archived to worklog/history/.
- Architecture decisions: worklog/decisions/NNN-slug.md (sequential numbering).
```

- [ ] **Step 3: Verify**

```bash
grep -n 'STATE.md\|docs/decisions\|docs/history' CLAUDE.md
```

Expected: at least three matches (one per added bullet).

---

## Task 18: Cross-Reference Scan + Fix

**Files:** any file referencing moved paths.

- [ ] **Step 1: Scan**

```bash
grep -rn 'docs/superpowers/specs\|docs/superpowers/plans\|docs/superpowers/prs' . \
    --include='*.md' --include='*.py' --include='*.ipynb' \
    --exclude-dir='.git' --exclude-dir='docs/history' \
    > /tmp/crossref.txt
wc -l /tmp/crossref.txt
cat /tmp/crossref.txt
```

- [ ] **Step 2: For each hit, decide**

Active path (one of the 4 stayed specs / 4 stayed plans / no PR active) → no change needed.

Archived path → rewrite:
- `worklog/history/specs/<archived>.md` → `worklog/history/specs/<archived>.md`
- `worklog/history/plans/<archived>.md` → `worklog/history/plans/<archived>.md`
- `worklog/history/prs/<file>.md` → `worklog/history/prs/<file>.md`

Use Edit tool per file. Filenames unchanged → mechanical replace of the directory prefix.

- [ ] **Step 3: Verify zero dangling**

```bash
# Re-scan and confirm any remaining refs point only to active files
grep -rn 'docs/superpowers/specs\|docs/superpowers/plans\|docs/superpowers/prs' . \
    --include='*.md' --include='*.py' --include='*.ipynb' \
    --exclude-dir='.git' --exclude-dir='docs/history' | \
    awk -F: '{print $3}' | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}-[a-z0-9-]+\.md' | sort -u
```

Expected: only the 4 active spec filenames and 4 active plan filenames appear (or empty).

- [ ] **Step 4: ADR rename refs**

Already handled in Task 2. Re-check:

```bash
grep -rn '001-feat_declip-integration' . --exclude-dir=.git --exclude-dir=docs/history
```

Expected: empty.

---

## Task 19: Final Verification Gate

**Files:** none modified.

- [ ] **Step 1: ADR count**

```bash
ls worklog/decisions/ | sort
```

Expected (12 files):
```
001-declip-integration.md
002-defer-sl4.md
003-defer-graphmap.md
004-defer-frametracker.md
005-defer-per-backend-noise-tuning.md
006-single-working-branch.md
007-dashboard-as-integration-harness.md
008-mapanything-stub-pattern.md
009-frame-sampling-in-utils.md
010-pointcloud-creator-registry.md
011-coordinate-frame-enum.md
012-hloc-direct-call.md
```

- [ ] **Step 2: Tree shape**

```bash
test -f worklog/ROADMAP.md
test -f worklog/STATE.md
test -d worklog/history/specs
test -d worklog/history/plans
test -d worklog/history/prs
test -d docs/superpowers/notes
```

All commands return 0.

- [ ] **Step 3: Active set intact**

```bash
ls worklog/history/specs/   # expect 4 (3 active + this design)
ls worklog/history/plans/   # expect 4 (3 active + this plan)
```

- [ ] **Step 4: STATE.md size**

```bash
wc -l worklog/STATE.md
```

Expected: <60 lines.

- [ ] **Step 5: WORKLOG trim sanity**

```bash
grep -c '^### 2026' worklog/WORKLOG.md   # original entry count preserved
grep -E '^## (Current State|Architecture Decisions|Core Modules|Phase 2|Phase 3|Deferred)' worklog/WORKLOG.md   # expect empty
```

- [ ] **Step 6: Blame preservation spot-check**

```bash
git log --follow --oneline worklog/history/specs/2026-04-22-loop-closure-design.md | head
git log --follow --oneline worklog/history/plans/2026-04-22-loop-closure.md | head
git log --follow --oneline worklog/decisions/001-declip-integration.md | head
```

Each returns the pre-move commit history (proves `git mv` preserved blame).

- [ ] **Step 7: No dangling cross-refs**

```bash
grep -rn 'docs/superpowers/prs' . --exclude-dir=.git --exclude-dir=docs/history
grep -rn '001-feat_declip-integration' . --exclude-dir=.git --exclude-dir=docs/history
```

Both empty.

- [ ] **Step 8: `git status` review**

```bash
git status
```

Expected `Changes to be committed` includes:
- Added: ROADMAP.md, STATE.md, decisions/006–012, notes/*.md (renames), history/specs/*.md (renames), history/plans/*.md (renames), history/prs/*.md (renames).
- Modified: WORKLOG.md, CLAUDE.md, decisions/001 (rename), this design + plan (untracked → staged).
- Deleted: nothing meaningful (only `.gitkeep` placeholders).

---

## Task 20: Commit

**Files:** all staged via prior tasks.

- [ ] **Step 1: Confirm only intended changes are staged**

```bash
git diff --stat --staged
git status --short | grep -v '^[MARDC]' | head    # any unstaged surprises?
```

Review the stat output for sanity (line counts moved should be near zero — pure renames). Any modified-not-renamed files outside the spec scope: investigate.

- [ ] **Step 2: Stage anything missed**

```bash
git add docs/ CLAUDE.md
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs: restructure superpowers/ into STATE + ROADMAP + decisions + history

Separates concerns previously merged in WORKLOG.md and accumulated
specs/plans/ directories. New layout:

- worklog/STATE.md — current-state dashboard
- worklog/ROADMAP.md — strategic architecture overview
- worklog/decisions/ — ADRs (extends existing 001-005 with new 006-012)
- worklog/history/{specs,plans,prs} — archived completed work
- worklog/notes/ — ad-hoc investigations

WORKLOG.md trimmed to session log only. CLAUDE.md updated to point at
new structure. All historical files preserved via `git mv` (blame intact).

Spec: worklog/history/specs/2026-05-20-docs-reorg-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify commit landed**

```bash
git log --oneline -1
git show --stat HEAD | head -20
```

Expected: top commit is the restructure; stat shows large number of renames + a small handful of edits (WORKLOG, CLAUDE.md, new ADRs, ROADMAP, STATE).

---

## Out of Scope (reminder from spec)

- Editing past WORKLOG entries.
- Reorganizing `docs/pointcloud/`, `docs/semantics/`, `docs/splats/`, `docs/notes/`.
- Reorg of `evals/`, `eval_results/`, `tests/`.
- Code changes.
- ADRs beyond the 7 extracted in this plan.
