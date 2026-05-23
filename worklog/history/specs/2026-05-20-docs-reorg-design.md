# Docs Reorganization — Design Spec

**Date:** 2026-05-20
**Branch:** `refactor/core-modules`
**Status:** Superseded by [2026-05-20-worklog-split-design.md](2026-05-20-worklog-split-design.md) (same day; split `docs/` vs `worklog/` instead of keeping all under `docs/`). Partially executed before supersession: ADRs 006–012 written, `history/` archive created.

---

## Goal

Separate concerns currently mashed into `worklog/WORKLOG.md` and the accumulated `specs/` + `plans/` directories. Target a layout where:

- **History** is append-only and untouched after the fact.
- **Current state** is editable, terse, readable in 30 seconds.
- **Future plans** are strategic, not tactical.
- **Architecture decisions** are extractable, numbered, and individually addressable.
- **Active work** is in `specs/` + `plans/`. Completed work lives in `history/`.

One PR, big-bang single commit.

---

## Target Tree

```
docs/
├── ROADMAP.md                          # arch overview: what's built / where we are / what's next
├── decisions/                          # ADRs (extant, grows to 012)
│   ├── 001-declip-integration.md       # renamed from 001-feat_declip-integration.md
│   ├── 002-defer-sl4.md                # existing
│   ├── 003-defer-graphmap.md           # existing
│   ├── 004-defer-frametracker.md       # existing
│   ├── 005-defer-per-backend-noise-tuning.md   # existing
│   ├── 006-single-working-branch.md            # new
│   ├── 007-dashboard-as-integration-harness.md # new
│   ├── 008-mapanything-stub-pattern.md         # new
│   ├── 009-frame-sampling-in-utils.md          # new
│   ├── 010-pointcloud-creator-registry.md      # new
│   ├── 011-coordinate-frame-enum.md            # new
│   └── 012-hloc-direct-call.md                 # new
├── history/                            # archive of completed work (read-only intent)
│   ├── specs/   (58 files)
│   ├── plans/   (41 files)
│   └── prs/     (2 files)
├── superpowers/
│   ├── WORKLOG.md                      # trimmed: session log only, newest first, append-only
│   ├── STATE.md                        # right-now dashboard, editable, ≤1 page
│   ├── known-test-failures.md          # stays; STATE links to it
│   ├── notes/                          # ad-hoc investigations
│   │   ├── 2026-05-08-ba-eval-script-split-summary.md
│   │   ├── 2026-05-08-bundle-adjustment-notebook-debug.md
│   │   └── 2026-05-08-co3dv2-eval-gap-investigation.md
│   ├── specs/  (3 active untracked: gt-eval-harness, feedforward-import-cleanup, feedforward-mesh)
│   └── plans/  (3 active untracked: same trio)
└── (existing per-module docs unchanged: docs/pointcloud/, docs/semantics/, docs/splats/)
```

---

## Inventory + Dispositions

### Existing ADRs (keep, normalize slug)

| Current path | Action |
|---|---|
| `worklog/decisions/001-feat_declip-integration.md` | rename → `001-declip-integration.md` |
| `worklog/decisions/002-defer-sl4.md` | keep |
| `worklog/decisions/003-defer-graphmap.md` | keep |
| `worklog/decisions/004-defer-frametracker.md` | keep |
| `worklog/decisions/005-defer-per-backend-noise-tuning.md` | keep |

### New ADRs (006–012)

| ADR | Title | Source |
|---|---|---|
| 006 | Single working branch strategy | `specs/2026-04-21-single-branch-strategy-design.md` (entire file, 41 lines) |
| 007 | Dashboard as integration test harness | `WORKLOG.md` § Architecture Decisions |
| 008 | MapAnything stub pattern | `WORKLOG.md` § Architecture Decisions |
| 009 | `frame_sampling` lives in `utils/`, not `semantics/` | `WORKLOG.md` session entry 2026-04-20 |
| 010 | Pointcloud creator registry | `specs/2026-04-16-pointcloud-submodule-design.md` (extract registry section) |
| 011 | `CoordinateFrame` enum + `world_transform` | `specs/2026-04-20-pointcloud-feedforward-design.md` (extract) |
| 012 | hloc direct call (no nerfstudio dependency) | `specs/2026-04-20-pointcloud-feedforward-design.md` (extract) |

Source specs remain referenced by ADRs (pointer to archived history copy).

### Active — stay in `superpowers/specs/` and `superpowers/plans/`

- `specs/2026-05-07-gt-eval-harness-design.md` + `plans/2026-05-07-gt-eval-harness.md`
- `specs/2026-05-08-feedforward-import-cleanup-design.md` + `plans/2026-05-08-feedforward-import-cleanup.md`
- `specs/2026-05-14-feedforward-mesh-design.md` + `plans/2026-05-14-feedforward-mesh.md`

### Archive to `worklog/history/specs/`

All 58 other spec files. `git mv` to preserve blame.

### Archive to `worklog/history/plans/`

All 41 other plan files. `git mv`.

### Archive to `worklog/history/prs/`

- `superpowers/prs/pr1-core-modules.md`
- `superpowers/prs/pr2-dashboard-complete.md`

### Loose `superpowers/` root → `superpowers/notes/`

- `2026-05-08-ba-eval-script-split-summary.md`
- `2026-05-08-bundle-adjustment-notebook-debug.md`
- `2026-05-08-co3dv2-eval-gap-investigation.md`

### Stays in place

- `superpowers/known-test-failures.md` — living doc; STATE.md links to it.

### Folded then archived

- `superpowers/specs/2026-04-16-refactor-roadmap.md` → content folded into new `ROADMAP.md`, then `git mv` to `worklog/history/specs/`.

---

## ADR Template

Matches existing 002-005. All new ADRs follow this.

```markdown
# ADR NNN: <Title>

**Status:** Accepted | Superseded by [ADR NNN](NNN-slug.md) | Deprecated
**Date:** YYYY-MM-DD
**Tags:** <comma,separated>

## Context
Why this decision was needed. Constraints. Prior state.

## Decision
What was decided. Concrete, falsifiable.

## Consequences
Positive + negative. Triggers that would revisit.

## Alternatives Considered
What else was on the table + why rejected.
```

Reversal: new ADR with `Status: Supersedes [ADR NNN]`; old ADR updated to `Status: Superseded by [ADR NNN]`. Never delete an ADR.

---

## STATE.md Shape

Editable, terse, ≤1 page. If it grows past one page, content leaked from WORKLOG or ROADMAP — rebalance.

```markdown
# State — last-updated YYYY-MM-DD

## Active Branch
| Branch | Status | Ahead of main | Notes |
|---|---|---|---|
| `refactor/core-modules` | active | 82 commits | sole working branch |

## In-Flight Work
- **gt-eval-harness** — [spec](specs/2026-05-07-gt-eval-harness-design.md) · [plan](plans/2026-05-07-gt-eval-harness.md) — status: <wip/blocked/review>
- **feedforward-import-cleanup** — [spec] · [plan] — status: …
- **feedforward-mesh** — [spec] · [plan] — status: …

## Open Blockers
- nerfstudio env: <symptom + link to notes/>
- known test failures: see [known-test-failures.md](known-test-failures.md)

## Parked
| Item | Why parked | Pointer |
|---|---|---|
| `tlb-grouping-segmentation` | separate plan TBD | [ROADMAP](../ROADMAP.md#parked) |
| `tlb-improve-splatter` | separate plan TBD | [ROADMAP](../ROADMAP.md#parked) |

## Recent Decisions (last 30 days)
- [ADR 012 — hloc direct call](../decisions/012-hloc-direct-call.md)
- [ADR 011 — CoordinateFrame enum](../decisions/011-coordinate-frame-enum.md)
```

---

## ROADMAP.md Shape

Strategic. Three sections: what's built, where we are, what's next.

```markdown
# Roadmap

## What's Built
One paragraph per major subsystem. Each lists key ADRs.

- **Semantics**: ANN feature splatting, MaskCLIP + Talk2DINO, batched extraction, dashboard query API.
  Key ADRs: [009](decisions/009-frame-sampling-in-utils.md).
- **Pointcloud**: creator registry (colmap / hloc / vggtx / mapanything), CoordinateFrame world transform, hloc direct call.
  Key ADRs: [010](decisions/010-pointcloud-creator-registry.md), [011](decisions/011-coordinate-frame-enum.md), [012](decisions/012-hloc-direct-call.md).
- **Loop Closure**: Sim(3) pose graph, geometric overlap gate, 3-way noise split, Huber robustification.
  Key ADRs: [002](decisions/002-defer-sl4.md)–[005](decisions/005-defer-per-backend-noise-tuning.md).
- **Mesh**: TSDF integration from pointcloud results; feedforward → direct mesh path.
- **Dashboard**: integration test harness — full pipeline visual validation.
  Key ADRs: [007](decisions/007-dashboard-as-integration-harness.md), [008](decisions/008-mapanything-stub-pattern.md).
- **BA**: PyPose BAE bundle adjustment, CO3Dv2 eval baseline.

## Where We Are
- Sole working branch: `refactor/core-modules` (~82 commits ahead).
- Final PR before merge to `main`.
- Active in-flight: gt-eval-harness, feedforward-import-cleanup, feedforward-mesh.

## What's Next

### Phase 2 — Dashboard Iteration
Goal, motivation, current sketch.

### Phase 3 — Feature Extension
Goal, motivation, scope.

### Candidate Features (unprioritized)
- <idea> — why it would help

### Parked Branches
- `tlb-grouping-segmentation` — separate plan TBD; intent: <one line>
- `tlb-improve-splatter` — separate plan TBD; intent: <one line>
```

---

## WORKLOG.md Template + Trim Policy

WORKLOG remains the append-only session history. After trim, only `# collab-splats Refactor Worklog` header + `## Session Log` section + existing dated entries.

**Entry template (newest first):**
```markdown
### YYYY-MM-DD <optional session tag>
- **Focus:** <one line>
- **Did:** <bullets — concrete actions>
- **Decided:** <bullets — link to ADRs: `→ ADR 011`>
- **Hit:** <bullets — blockers / surprises / reversals>
- **Files:** <key paths touched>
- **Next:** <one line>
```

**Trim policy:**
- Existing session log entries (current lines 186–307) preserved verbatim.
- Non-session sections (Current State, Architecture Decisions, Core Modules task lists, Dashboard scope, Phase 2/3, Cleanup, Deferred/Parked) → moved out per inventory; not deleted, just relocated.
- Past entries never edited. Reversals = new entry referencing prior decision.

---

## Migration Steps (single commit)

1. Create dirs: `worklog/history/{specs,plans,prs}/`, `worklog/notes/`.
2. Create files: `worklog/ROADMAP.md`, `worklog/STATE.md`.
3. Rename `worklog/decisions/001-feat_declip-integration.md` → `001-declip-integration.md`.
4. Write ADRs 006–012 using template, extracting from WORKLOG + named source specs.
5. `git mv` archived specs/plans/prs/notes per inventory.
6. Trim `WORKLOG.md`: keep header + `## Session Log` + dated entries; delete other sections.
7. Update `CLAUDE.md` doc pointers.
8. Run cross-ref scan, fix dangling links.

---

## Cross-Reference Preservation

Filenames preserved when archiving — only directory changes. Mechanical rewrites:

- `worklog/history/specs/<archived>.md` → `worklog/history/specs/<archived>.md`
- `worklog/history/plans/<archived>.md` → `worklog/history/plans/<archived>.md`
- `worklog/history/prs/<file>.md` → `worklog/history/prs/<file>.md`

**Scan command** (run before commit):
```bash
grep -rn 'docs/superpowers/specs\|docs/superpowers/plans\|docs/superpowers/prs' . \
    --include='*.md' --include='*.py' --include='*.ipynb' \
    --exclude-dir='.git' --exclude-dir='docs/history'
```

For each hit:
- Active file (in `superpowers/specs|plans/`): unchanged.
- Archived file: rewrite path to `worklog/history/...`.

WORKLOG self-refs in session entries: same rewrite logic.
Spec → spec refs: filenames unchanged; only update the prefix.
Spec → ADR refs (e.g. lc-04 → `worklog/decisions/001-feat_declip-integration.md`): update for renamed `001` only.

---

## CLAUDE.md Update

Replace current lines 7–8 ("look through superpowers/WORKLOG.md") with:

```
- At start of every conversation: read worklog/STATE.md first, then latest entries in worklog/WORKLOG.md, then worklog/decisions/NNN-*.md as referenced.
- Active spec/plan for in-flight work: docs/superpowers/{specs,plans}/. Completed work: worklog/history/.
- Architecture decisions: worklog/decisions/NNN-slug.md (sequential numbering).
```

---

## Verification Gate

Before commit:

1. **No dangling refs.** Cross-ref scan returns 0 hits pointing to non-existent paths.
2. **All 7 new ADRs written.** `ls worklog/decisions/` shows 001–012.
3. **WORKLOG trimmed.** Only header + Session Log section remain.
4. **STATE.md exists, ≤1 page.**
5. **ROADMAP.md exists, three sections populated.**
6. **`git status` clean of stray untracked files.** Active untracked specs/plans stay where they are (now tracked by commit).
7. **`git log --follow` works** on at least 3 archived files (blame preservation sanity check).

---

## Out of Scope

- Editing past WORKLOG entries.
- Deleting specs/plans (all moved, not deleted).
- Reorg of `docs/pointcloud/`, `docs/semantics/`, `docs/splats/`, `docs/notes/` (per-module docs).
- Reorg of `evals/`, `eval_results/`, `tests/`.
- New ADRs beyond the 7 extracted from existing material.
- Code changes — this is a docs-only PR.

---

## Files Touched (summary)

- **New**: `worklog/ROADMAP.md`, `worklog/STATE.md`, ADRs 006–012.
- **Renamed**: `worklog/decisions/001-feat_declip-integration.md`.
- **Moved (git mv)**: 58 specs + 41 plans + 2 PR bodies + 3 loose notes → `worklog/history/*` and `worklog/notes/`.
- **Edited**: `worklog/WORKLOG.md` (trimmed), `CLAUDE.md` (doc pointers).
