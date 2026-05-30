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
