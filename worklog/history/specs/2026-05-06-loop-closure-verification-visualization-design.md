# Loop Closure Verification & Visualization Design

**Date:** 2026-05-06  
**Status:** Approved  
**Branch:** refactor/core-modules

## Problem

The loop closure module detects loops via DINO-SALAD retrieval but has two gaps:

1. No developer-facing visibility into which loops fired, why they were accepted/rejected, or whether drift was actually corrected.
2. `LoopMatch` is a `NamedTuple`, which is immutable. Adding outcome fields requires conversion to a mutable dataclass.

## Why No 3D Geometric Gate

VGGT-Long uses RANSAC-Umeyama on matched correspondences for geometric verification. This is not portable to our pipeline: `world_points` are dense depth grids in **submap-local coordinate frames** — each submap's frame 0 is the identity, and distant loop closure candidates have completely different local worlds. KD-tree overlap between them is meaningless.

Replicating VGGT-Long requires a second VGGT forward pass per loop candidate at gate time (expensive, major architectural change). The existing DINO cosine gate (`verify_match_ratio`) and `translation_jump_check` remain as the only gates. A 3D geometric gate can be added later when correspondence infrastructure is available.

## Scope

Two layers of change.

---

## Layer 1 — Richer `LoopMatch`

**File:** `collab_splats/pointcloud/loop_closure/retrieval.py`

`LoopMatch` is currently a `NamedTuple`. Convert to `@dataclass` and add one field: `accepted: bool = False` (set to `True` by the feedforward gate loop when all gates pass).

`LoopMatchQueue` pushes `(-score, match)` tuples onto a heap. When two scores are equal Python compares `match` directly — NamedTuple supports this implicitly, `@dataclass` does not. Fix: add a monotonic integer tiebreaker as the second heap element so `match` is never compared directly.

The feedforward gate loop collects all post-NMS candidates — accepted and rejected — in `creator._lc_all_matches`. Callers filter on `.accepted`. Pre-NMS candidates remain invisible.

No new files or dataclasses.

---

## Layer 2 — Notebook Restructure

**File:** `docs/pointcloud/loop_closure_eval.ipynb`

Existing §1–§8 cells are preserved. Two part headers divide the notebook. Two new sections are added:

```
## Part I — Detection

§1  Setup
§2  Config
§3  Run feedforward + LC
§4  Detection audit               ← NEW

## Part II — Correction

§5  Build pose graph
§6  Loss curve
§7  Per-edge residual breakdown
§8  PCD + frustums (before vs after)
§9  Endpoint gap                  ← NEW
§10 GT stubs (was §9)
```

**§4 Detection audit:** matplotlib grid — one row per post-NMS candidate (accepted and rejected). Each row shows the query and detected frame images side by side, with similarity score and accepted/rejected verdict. Gives a developer immediate visual confirmation of which loops fired.

**§9 Endpoint gap:** for each accepted match, bar chart comparing the world-space distance between matched frame positions before and after pose graph correction. Derived inline from `initial_chained` and `optimized` notebook variables. Summary table: candidate count, acceptance rate, mean gap reduction.

---

## Data Flow

```
feedforward run
  └── Submap (frames, poses, world_points, retrieval_vectors)

ImageRetrieval.find_loop_closures()
  └── list[LoopMatch]  — NMS-filtered candidates

feedforward.py gate loop
  ├── _verify_loop_candidate  →  DINO cosine gate (existing, verify_match_ratio)
  ├── translation_jump_check  →  reject if jump too large (accepted stays False)
  └── pass                    →  match.accepted = True, add LC submap to pose graph

creator._lc_all_matches = all post-NMS candidates (list[LoopMatch])

run_pose_graph_optimization()
  └── optimized poses (list of (4,4), notebook variable: `optimized`)
  └── initial poses (list of (4,4), notebook variable: `initial_chained`)

Notebook §4: creator._lc_all_matches → image pair grid
Notebook §9: accepted matches + initial_chained + optimized → endpoint gap chart
```

---

## Testing

- `retrieval.py`: dataclass field test, heap tiebreaker test (equal scores must not raise `TypeError`)
- Feedforward gate loop: `_lc_all_matches` accumulates all post-NMS candidates with `.accepted` flags set correctly
- Existing notebook smoke test passes with restructured sections

---

## Out of Scope

- 3D geometric gate — not feasible without correspondence infrastructure
- ATE/RPE ground-truth metrics — stubs unchanged
- Splitting into two notebooks
- New model dependencies (no LightGlue/LoFTR)
- Changes to PyVista rendering
