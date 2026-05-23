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
