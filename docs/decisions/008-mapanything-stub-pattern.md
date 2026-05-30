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
