# Single Working Branch Strategy

**Date:** 2026-04-21
**Status:** Approved

## Context

The stacked PR strategy (`refactor/core-modules` → `refactor/core-modules`) created friction: every core-scope change required rebasing dashboard-complete. Analysis showed dashboard-complete is a strict superset of core-modules (59 shared commits + 20 dashboard-only; 0 unique to core-modules). The branches aren't diverged — the pain was pure coordination overhead.

## Decision

1. **One working branch:** `refactor/core-modules` is the sole active branch. All new commits — core fixes, dashboard features, utils — go here.
2. **Freeze core-modules:** `refactor/core-modules` is a frozen snapshot. No further commits.
3. **One PR, sectioned review:** Submit one PR against `main` with the full branch. PR description organizes changes by module for focused review.
4. **Commit prefixes:** Tag scope via conventional commits (`refactor(core):`, `feat(dashboard):`, `fix(utils):`) for traceability.

## PR Review Sections

```
## Semantics module
Files: collab_splats/semantics/*

## Pointcloud / feedforward
Files: collab_splats/pointcloud/*, stage/*

## Dashboard
Files: collab_splats/dashboard/*, tests/dashboard/*

## Utils & wrapper
Files: collab_splats/utils/*, collab_splats/wrapper/*

## Config & packaging
Files: pyproject.toml, collab_splats/configs/*
```

## Tradeoffs

- **Pro:** Zero rebases, zero branch switching, one linear history
- **Pro:** Sectioned review gives same focused review benefit as split PRs
- **Con:** Single larger PR (79+ commits) — mitigated by section labels
- **Con:** Can't merge core independently before dashboard — acceptable since both are ready together
