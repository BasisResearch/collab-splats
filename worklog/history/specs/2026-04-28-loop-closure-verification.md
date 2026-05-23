# Loop Closure Refactor — Verification Report

**Date:** 2026-04-28
**Branch:** refactor/core-modules
**Author:** Brainstorming session (Claude Opus 4.7)
**Purpose:** Independent verification of findings and design choices before writing the implementation plan.

---

## Mission for the verifying agent

You are reviewing a brainstorming session output. The session analyzed:
1. The current loop-closure implementation in `collab-splats`
2. Two upstream references: MIT-SPARK/VGGT-SLAM and DengKaiCQ/VGGT-Long
3. Identified silent bugs in current code, missing upstream features, and proposed a refactor

Your job is to **independently verify** each finding and design choice. You should:
- Read the cited files at the cited line numbers
- Confirm or refute each claim with direct evidence
- Flag claims that are wrong, overstated, or missing context
- Propose corrections or additions

Treat this report as a hypothesis, not a conclusion. Ground every check in actual code, not the report's assertions.

---

## Repositories

| Repo | URL | Use |
|---|---|---|
| collab-splats (this repo) | local: `/workspace/collab-splats` | Current implementation under review |
| MIT-SPARK/VGGT-SLAM | https://github.com/MIT-SPARK/VGGT-SLAM | Architectural ancestor of our LC code |
| DengKaiCQ/VGGT-Long | https://github.com/DengKaiCQ/VGGT-Long | Alternative paradigm (per-chunk Sim(3)/SE(3)) |
| VGGT-SLAM PR #39 | https://github.com/MIT-SPARK/VGGT-SLAM/pull/39 | FastVGGT integration WIP, surfaces SL(4)+metric-scale corruption |

When verifying, prefer fetching raw files from `main` branch; record commit SHAs you read so a re-run is reproducible.

---

## Current state — files involved

**LC implementation (port from VGGT-SLAM):**
- `collab_splats/pointcloud/loop_closure.py` (107 lines) — `LoopClosureConfig`, `LoopMatch`, `LoopMatchQueue`, `ImageRetrieval`
- `collab_splats/pointcloud/pose_graph.py` (125 lines) — `PoseGraph` (GTSAM SE(3) wrapper)
- `collab_splats/pointcloud/submap.py` (20 lines) — `Submap` dataclass

**LC orchestration (lives in feedforward.py):**
- `collab_splats/pointcloud/feedforward.py` lines 130–280 (`_run_loop_closure_inference`, `_loop_close`, `_merge_submap_outputs`)
- `collab_splats/pointcloud/feedforward.py` lines 137–139 (base `_verify_loop_candidate` — returns True)
- `collab_splats/pointcloud/feedforward.py` lines 786–794 (`VGGTXCreator._verify_loop_candidate` — VGGT image_match_ratio check)

**Existing spec/plan/tests:**
- `worklog/history/specs/2026-04-22-loop-closure-design.md`
- `worklog/history/plans/2026-04-22-loop-closure.md`
- `tests/pointcloud/test_pose_graph.py`
- `tests/pointcloud/test_loop_closure.py`
- `tests/pointcloud/test_loop_closure_integration.py`

**Related ADR:**
- `worklog/decisions/001-defer-declip-integration.md` (semantic search deferred)

---

## Section 1 — Findings about the current code (silent bugs)

For each finding: claim, evidence (file:line), why it matters, verification action.

### Finding 1 — `pose_graph.py::add_submaps` only adds intra-submap edges

**Claim:** Despite the docstring saying "intra-submap edges + inter-submap overlap edges", only intra-submap `BetweenFactorPose3` edges exist. Submaps are disconnected components in the graph.

**Evidence:** `collab_splats/pointcloud/pose_graph.py:54–76`. The inner loop (lines 62–74) inserts intra-submap edges only when `local_i > 0`. No code adds an edge between `(last frame of submap N)` and `(first frame of submap N+1)`.

**Why it matters:** Levenberg–Marquardt cannot propagate corrections across submap boundaries. Loop edges only constrain the specific (q_frame, d_frame) pair. The optimizer effectively has nothing to do for non-LC frames except enforce intra-submap continuity that VGGT already produced.

**Verify:** Read `pose_graph.py::add_submaps` end-to-end. Count `BetweenFactorPose3` calls per submap; confirm zero appear between submaps. Also check whether `add_loop_edges` (lines 78–95) compensates — it does not (only adds edges for verified LC pairs).

### Finding 2 — Submap initial poses live in incompatible local frames

**Claim:** Each VGGT/MapAnything chunk forward returns poses where frame 0 of the chunk is at identity (or near it). When `add_submaps` inserts submap N+1's poses, they are not transformed into submap N's world frame. The initial value passed to LM is geometrically incoherent across submap boundaries.

**Evidence:** `collab_splats/pointcloud/pose_graph.py:66` inserts `_pose3(pose)` (raw submap-local pose) directly into `self._initial`. Confirm by inspecting any submap's `submap.poses[0]` — it is an identity-like matrix, not the world-frame continuation of the previous submap's last pose.

**Why it matters:** Even if Finding 1 is fixed by adding inter-submap edges, LM starts from broken initial values. Convergence is compromised; LM may diverge or settle at a poor local minimum.

**Verify:**
1. Add a print to `_run_loop_closure_inference` to dump `submap.poses[0]` for each submap. Confirm they are near-identity (proves the convention).
2. Trace whether VGGT's per-chunk forward outputs cam-to-world or world-to-cam (relevant to Finding 7).

### Finding 3 — Overlap region duplicates frames as separate graph nodes

**Claim:** With `submap_size=20, submap_overlap=4`, consecutive submaps share 4 frames. Each shared frame becomes 2 separate global nodes (one per submap), free to drift apart. The output `np.concatenate` then produces an array of size `N + (n_submaps - 1) * O` instead of `N`.

**Evidence:**
- Submap loop: `feedforward.py:173` (`for wi, start in enumerate(range(0, N, step))`) where `step = K - O` (line 148).
- `pose_graph.py:54–76`: each call assigns new global indices via `self._total_frames += k`. No deduplication.
- `feedforward.py:261`: `np.concatenate([corrected[s.submap_id] for s in submaps], axis=0)` — naive concat, no dedup.

**Why it matters:** Output array shape mismatches input frame count. Downstream `_postprocess` and `build_colmap` likely consume the wrong indices. Silent corruption of which pose maps to which input image.

**Verify:**
1. Compute expected output shape: with `N=100, K=20, O=4`: step=16 → submaps cover frames `[0,20)`, `[16,36)`, ..., last submap ends at min(start+K, N)=100. n_submaps = ceil((100-4)/16) = 6. Total frames concatenated = 6*20 = 120 (last submap may be shorter). Expected: 100. Actual: ≈120. Confirm with a unit-style trace.
2. Read `_postprocess` / `build_colmap` to check what they expect.

### Finding 4 — Loop edge math uses incompatible frames

**Claim:** When building an LC submap (lines 219–236), the two poses come from two different submaps that live in different local frames. The relative pose `_relative_pose3(submap.poses[q], d_submap.poses[d])` (used later in `pose_graph.py:94`) is therefore meaningless.

**Evidence:** `feedforward.py:222–225` constructs the LC submap using `submap.poses[match.query_frame_idx]` and `d_submap.poses[match.detected_frame_idx]`. These poses come from two separate VGGT forwards on two different submap windows. They are NOT in a common coordinate frame.

`pose_graph.py:94` then computes `_relative_pose3(lc.poses[0], lc.poses[1])` and uses it as the BetweenFactor constraint.

**Why it matters:** Upstream VGGT-SLAM and VGGT-Long both **re-run the foundation model on the (q, d) pair** to get same-frame relative poses. Our `_verify_loop_candidate` does re-run VGGT (line 791–792) but only extracts `image_match_ratio`; the resulting poses are discarded.

**Verify:**
1. Confirm by reading lines 219–236: the LC submap's `poses` field is set from the two original submaps, not from a fresh VGGT pair re-run.
2. Check `VGGTXCreator._verify_loop_candidate` (786–794) and confirm `predictions` contains pose data that is discarded.
3. Compare against VGGT-SLAM upstream: read `vggt_slam/solver.py` to see how it constructs the LC pose constraint.

### Finding 5 — Default `_verify_loop_candidate` returns True

**Claim:** Base class `_verify_loop_candidate` returns `True` unconditionally (line 137–139). Only `VGGTXCreator` overrides. MapAnything path accepts every retrieval candidate as a verified loop.

**Evidence:** `feedforward.py:137–139`:
```python
def _verify_loop_candidate(self, frame1, frame2) -> bool:
    """Override in subclasses to add model-based verification. Default accepts all."""
    return True
```

`MapAnythingCreator` does not override this method (verify by grepping the file).

**Why it matters:** False-positive loop closures corrupt the graph. With `lc_threshold=0.95` (very permissive on L2 distance — see Finding 8), MapAnything path will admit many false loops.

**Verify:**
1. Grep `_verify_loop_candidate` in `feedforward.py` — confirm only one override (`VGGTXCreator`).
2. Check `MapAnythingCreator` class for any equivalent method.
3. Confirm `cfg.lc_threshold = 0.95` default in `loop_closure.py:24`.

### Finding 6 — Same noise model for intra and loop edges

**Claim:** Loop edges use `self._intra_noise` (the same model used for intra-submap continuity), not a separate noise model. Loop edges should be noisier because they come from a separate model re-run.

**Evidence:** `pose_graph.py:95` uses `self._intra_noise` for the loop BetweenFactor.

**Why it matters:** Loop edges become over-confident, can dominate the optimizer, and small loop errors propagate everywhere.

**Compare:** VGGT-SLAM upstream `vggt_slam/graph.py:18–21` explicitly defines both `inner_submap_noise` and `intra_submap_noise`. VGGT-Long `loop_utils/sim3loop.py` uses an IRLS robust scheme.

**Verify:**
1. Confirm `pose_graph.py:95` uses `_intra_noise`.
2. Confirm VGGT-SLAM upstream has separate noise models.

### Finding 7 — Pose convention unverified

**Claim:** `Submap.poses` docstring says "world-to-cam homogeneous" (`submap.py:14`). VGGT-typical output is cam-to-world. There is no assertion or test verifying the actual convention.

**Why it matters:** If mismatched, all relative pose math (`_pose3`, `_relative_pose3`, `Pose3.between`) is inverted. LM steps in the wrong direction.

**Verify:**
1. Read VGGT model output documentation or `vggt/utils/pose_enc.py` to determine actual convention of `extrinsic` field.
2. Check what `feedforward.py:183` (`raw["extrinsic"]`) actually contains.
3. Add a unit test asserting the convention.

---

## Section 2 — Findings about upstream features we silently dropped

### Finding 8 — Default `lc_threshold=0.95` is far more permissive than upstream

**Claim:** Our default `lc_threshold=0.95` (L2 distance on normalized embeddings) corresponds to cosine similarity ≈ 0.55 (angle ≈ 56°). VGGT-Long uses cosine similarity ≥ 0.85 (L2 ≈ 0.548), explicitly raised from 0.7 to "significantly reduce false matches" (per `configs/base_config.yaml`).

**Math:** On L2-normalized vectors: `L2² = 2 − 2·cos`. So `L2 ≤ 0.95` ⟺ `cos ≥ 1 − 0.95²/2 ≈ 0.549`.

**Evidence:**
- Local: `collab_splats/pointcloud/loop_closure.py:24`
- VGGT-Long: `configs/base_config.yaml`, key `Loop.SALAD.similarity_threshold: 0.85`

**Verify:**
1. Confirm both threshold values in source.
2. Re-derive the math.

### Finding 9 — No NMS on loop candidates

**Claim:** Our `LoopMatchQueue` keeps top-k by score but does not suppress candidates that are temporally adjacent. VGGT-Long suppresses candidates within a frame-distance window (`Loop.SALAD.nms_threshold: 25`).

**Evidence:**
- Local: `collab_splats/pointcloud/loop_closure.py:29–42` (LoopMatchQueue, no NMS).
- VGGT-Long: `configs/base_config.yaml::Loop.SALAD.nms_threshold` and `loop_utils/loop_refinement.py::reduce_edges` (numba-jit NMS over candidate edges).

**Verify:** Read both.

### Finding 10 — Default `max_loops_per_submap=1` is overly conservative

**Claim:** Our default keeps only the single best loop per submap. VGGT-Long default `top_k=5`.

**Evidence:**
- Local: `collab_splats/pointcloud/loop_closure.py:25`
- VGGT-Long: `configs/base_config.yaml::Loop.SALAD.top_k: 5`

### Finding 11 — No inter-submap overlap-region alignment (Umeyama)

**Claim:** VGGT-SLAM (`scale_solver.py::estimate_scale_pairwise`) and VGGT-Long (`loop_utils/sim3utils.py::estimate_sim3` and `loop_utils/loop_refinement.py::umeyama_alignment`) both compute a Sim(3) or SE(3) transform between consecutive submaps using overlap-region 3D point correspondences. We have nothing equivalent.

**Why it matters:** This single missing piece is what would solve Findings 1, 2, and 4 simultaneously. The Umeyama transform provides:
- Initial-value alignment for submap N+1's poses (Finding 2)
- An inter-submap BetweenFactor constraint (Finding 1)
- A path to compute the loop-edge relative pose in a common frame (Finding 4)

**Verify:** Read both upstream files. Confirm they consume per-frame point clouds + confidence to produce the transform.

### Finding 12 — No `FrameTracker` adaptive keyframing

**Claim:** Our submap windows are fixed-size (`submap_size=20, submap_overlap=4`). VGGT-SLAM upstream has an optical-flow-based `FrameTracker` that triggers a new keyframe based on inter-frame disparity, producing fewer redundant submaps in static scenes.

**Evidence:** VGGT-SLAM `vggt_slam/frame_overlap.py::FrameTracker`. We have no equivalent.

**Verify:** Read upstream file. Decide if we want this in scope (current proposal: defer to follow-up PR).

### Finding 13 — No `GraphMap` central submap registry

**Claim:** Upstream uses `vggt_slam/map.py::GraphMap` as a central registry with `retrieve_best_score_frame` and `retrieve_best_semantic_frame`. We use a plain `list[Submap]` and inline the search loop.

**Verify:** Read upstream file. Decide scope (current proposal: defer; intertwined with semantic search ADR 001 reversal).

### Finding 14 — Submap class is anemic compared to upstream

**Claim:** Upstream `Submap` (~10 KB, `vggt_slam/submap.py`) has methods like `get_points_in_world_frame(graph)`, `get_first_homography_world(graph)`, `add_all_points`, `get_all_semantic_vectors`, `set_lc_status`. Ours is a 20-line dataclass.

**Implication:** Not necessarily a bug, but a sign that growth into the upstream shape will require keeping `submap.py` as its own well-scoped file (so methods can be added without bloating other files).

**Verify:** Compare upstream and local Submap files.

---

## Section 3 — Three-way comparison

### Pose representation

| Repo | Group | DOF | Optimizer |
|---|---|---|---|
| VGGT-SLAM upstream | SL(4) | 15 | GTSAM LM (`BetweenFactorSL4`) |
| VGGT-Long | Sim(3) (default) / SE(3) (metric mode) | 7 / 6 | Custom pypose LM, fastloop C++/Py |
| **Ours** | SE(3) | 6 | GTSAM LM (`BetweenFactorPose3`) |

### Why we chose SE(3)

- `gtsam-develop` SL(4) wheels require Python 3.11; our env is 3.10
- SL(4) corrupts metric-scale outputs from MapAnything (the bug VGGT-SLAM PR #39 reports)

**Verify:**
- Confirm gtsam-develop Python version requirements.
- Read VGGT-SLAM PR #39 description and confirm the depth/scale bug stems from SL(4) on metric backends.

### Architectural overlap

| Dim | Ours vs VGGT-SLAM | Ours vs VGGT-Long |
|---|---|---|
| Same retrieval (DINO-SALAD) | yes | yes (SALAD is their preferred over DBoW2) |
| Same factor-graph paradigm | yes (per-frame nodes) | no (they use per-chunk) |
| Same optimizer family (GTSAM LM) | yes | no |
| Same noise-model split (intra/inter/loop) | no — we conflated | n/a (different math) |
| Multi-backend abstraction | yes (we have one, they have PR #39 WIP) | yes (they have `Base3DModel` ABC) |

**Estimated coverage:** we ported ~15% of VGGT-SLAM (~300 LOC of ~2000), and our paradigm overlaps ~30% with VGGT-Long.

**Verify:** LOC counts via `wc -l` on the relevant directories. Architectural overlap is qualitative — agree/disagree based on your reading.

### Will our SE(3) match VGGT-Long results?

**Claim:** No, not numerically. Different granularity (per-frame vs per-chunk), different math (factor graph vs Umeyama+custom LM), different verification (image_match_ratio vs Umeyama 3D fit), different retrieval backbones in default configs (DINO-SALAD vs DBoW2). Per-frame factor graph is **theoretically more expressive** (can correct mid-chunk poses); per-chunk Umeyama is **more robust** (single bad edge can't destabilize). Both should beat a no-LC baseline.

**Verify:** Mostly conceptual; check that the comparison aligns with the actual implementations in both repos. Flag overstatement.

---

## Section 4 — Proposed design

### File structure

```
collab_splats/pointcloud/loop_closure/
├── __init__.py       # public API exports
├── submap.py         # Submap dataclass — adds world_points, world_points_conf for Umeyama
├── retrieval.py      # ImageRetrieval, LoopMatchQueue, NMS              [DINO-SALAD coupled]
├── alignment.py      # Umeyama, overlap-region transform, dedup helpers [pure numpy/numba]
├── pose_graph.py     # PoseGraph SE(3): intra+inter+loop edges, 3 noise models, Huber  [GTSAM coupled]
└── closure.py        # run_loop_closure() orchestration                 [pure logic]
```

**Rationale per file:**
- `submap.py` separate (Finding 14): dataclass will grow upstream-style methods over time.
- `retrieval.py` separate: only file with DINO-SALAD coupling. Quarantines that dep.
- `alignment.py` separate: Umeyama + dedup is pure-math, reusable, easy to test in isolation.
- `pose_graph.py` separate: only file with GTSAM coupling. Quarantines that dep.
- `closure.py` separate: orchestration glue. Concrete name (mirrors package). `pipeline.py` rejected for being too generic.

**Verify:**
- Each file's stated coupling is correct (no GTSAM imports in `alignment.py`, etc.).
- File count is justified — flag if you think any file should fold into another.

### In-scope PR contents (Option X1: refactor + correctness combined)

**Refactor:**
1. Sub-package consolidation (6 files above)
2. Orchestration extraction from `feedforward.py` → `closure.py::run_loop_closure(...)`
3. ADR 002 records: defer SL(4); defer GraphMap; defer FrameTracker; defer per-backend noise tuning

**Correctness fixes (the bugs from Section 1):**
4. Inter-submap edges via overlap-region Umeyama (`alignment.py` + `pose_graph.py`)
5. Submap initial values aligned via accumulated overlap transforms (chained world frames)
6. Overlap-frame dedup at output (final shape `(N, 4, 4)` matching input)
7. LC submap stores re-run pair's same-frame poses (loop edge math correct)
8. Default `_verify_loop_candidate` either raises NotImplementedError or implements a model-agnostic gate
9. Three noise models: `intra`, `inter`, `loop`
10. Huber robust kernel on loop edges
11. Pose convention assertion + unit test

**Robustness:**
12. NMS on loop candidates (frame-distance threshold ~25)
13. Raise `max_loops_per_submap` default 1 → 5
14. Threshold stored as cosine (≥ 0.85) for human readability; converted to L2 internally
15. Translation-jump pre-add gate (SE(3) port of MR.ScaleMaster anchor-scale alarm). After `_verify_loop_candidate` and before `add_loop_edges`: compute implied correction `ΔT = T_loop ∘ T_odom_path^{-1}`; reject if `‖ΔT.translation‖ > 0.2 · odom_path_length`. Hard reject (no robustification). ~10 LOC in `closure.py`.
16. Loop-edge down-weighting via translation-magnitude kernel (SE(3) analog of MR.ScaleMaster `scale_weight_from_vertex`). For accepted loops, scale noise sigma by `1/(1 + α·‖t_loop‖²)` to soften long-range loops. α tuned per backend.
17. Three-way noise split (odom-intra / loop-intra / loop-inter) scaffold. Single-robot mode collapses inter-bucket to no-op; preserves shape for future multi-session work.

**Tests:**
15. Existing `test_pose_graph.py`, `test_loop_closure.py`, `test_loop_closure_integration.py` updated
16. New: synthetic 2-submap overlap test verifying alignment converges
17. New: synthetic 3-submap loop test verifying the loop closes and ATE drops

**Logging/debug (cheap additions):**
18. Log graph size, edge counts (intra/inter/loop), LM iterations, final cost
19. Gate "success" on final cost threshold; warn and return uncorrected on divergence

### Deferred to future PRs (with benchmark data)

- SL(4) factor graph (requires py3.11 + per-backend gating; ADR 002 records the trigger)
- `GraphMap` central registry (couples with ADR 001 reversal — when DeCLIP / semantic search returns)
- `FrameTracker` adaptive keyframing
- Per-backend noise tuning (different `_intra_noise` per foundation model)
- Standalone `submap_size` parameter without LC (scalability-only mode)

---

## Section 5 — Specific verification tasks

For the verifying agent: complete each of these independently. Report agree/disagree with one line of evidence per task.

### Task A — Verify each Finding 1–14 against the cited file:line

For each finding, open the cited file, read the cited lines, and confirm or refute the claim. If the line numbers have shifted, note the new location. If the claim is wrong, state what is actually true.

### Task B — Confirm pose convention (Finding 7)

Determine whether `raw["extrinsic"]` from VGGT and from MapAnything is cam-to-world or world-to-cam. Sources:
- VGGT: `vggt/utils/pose_enc.py::pose_encoding_to_extri_intri` — read what it outputs.
- MapAnything: equivalent inference function in the MapAnything model code.

Report findings. If they differ, that's a bug we did not catch.

### Task C — Confirm Bug 3 output shape

Compute the actual shape of `corrected_extrinsics` for representative inputs:
- N=100, K=20, O=4 (typical)
- N=50, K=20, O=4 (small)
- N=21, K=20, O=4 (just over threshold)

Trace through `_run_loop_closure_inference` window iteration and `_loop_close::np.concatenate`. Report expected output shape vs input frame count `N`.

### Task D — Confirm Bug 4 by reading upstream

Read VGGT-SLAM `vggt_slam/solver.py` and find the section that handles loop-closure verification. Confirm whether the upstream code:
1. Re-runs the foundation model on the (q, d) pair, AND
2. Uses the resulting fresh poses (in their own common frame) to construct the BetweenFactor, AND
3. Discards the original submap-local poses for the loop edge.

Compare against our implementation (`feedforward.py:222–225` + `pose_graph.py:94`). Confirm or refute Finding 4.

### Task E — Verify the L2 ↔ cosine math (Finding 8)

Re-derive: for unit-norm vectors `u`, `v`:
```
‖u − v‖² = ‖u‖² − 2⟨u,v⟩ + ‖v‖² = 1 − 2cos + 1 = 2(1 − cos)
```
Therefore `L2 = √(2(1 − cos))`. Plug in:
- `cos = 0.85` → L2 ≈ 0.5477
- `cos = 0.55` → L2 ≈ 0.9487
- our default `L2 = 0.95` → `cos ≈ 0.549`

Confirm. Then verify the actual default value in `loop_closure.py:24` and VGGT-Long's value in `configs/base_config.yaml`.

### Task F — Sanity-check the file structure proposal

Read the proposed 6-file structure. For each file:
1. Confirm the named external coupling is correct (e.g., `alignment.py` truly has no GTSAM/torch dep beyond numpy).
2. Estimate LOC each file would have (rough OK).
3. Flag any file you'd merge or split differently.

### Task G — Independent recommendation on `_verify_loop_candidate` default

Decide between:
- **Option A:** raise `NotImplementedError` — forces every backend to declare a verifier
- **Option B:** implement a model-agnostic gate (e.g., re-embed both frames with DINO and check cosine similarity above a stricter threshold)
- **Option C:** other

Recommend one with reasoning. The current `return True` is unacceptable per Finding 5.

### Task H — Independent recommendation on Option X1 vs X2

X1 = combined refactor + correctness PR. X2 = staged (refactor first, correctness follow-up).

Argument for X1: bugs touch the same files refactor touches, so X2 means modifying every file twice.
Argument for X2: smaller reviewable PRs, easier rollback.

Make your own recommendation with reasoning.

### Task I — Spot-check VGGT-Long claim of multi-backend support

Read `base_models/base_model.py`. Confirm that VGGT, Pi3, MapAnything are all wired through the `Base3DModel` ABC. Confirm the Oct 2025 SE(3) mode (`using_sim3` config flag) is present in their code, not just promised in the README.

### Task J — Spot-check VGGT-SLAM PR #39 relevance

Read PR #39's description and diff. Confirm:
1. The reported bug is depth/scale corruption when SparkFastVGGT is the backend.
2. The author suspects an SL(4) transformation in `set_point_cloud` / `get_points_in_world_frame`.
3. This is consistent with the broader "SL(4) corrupts metric scale" claim that justifies our SE(3) choice.

If the PR has been merged or closed since this report was written, note current status.

---

## Section 6 — Disagreement protocol

When you disagree with a finding or design choice:

1. **State the disagreement plainly** — "Finding N is wrong because…"
2. **Cite evidence** — file:line, command output, math derivation
3. **Propose a replacement** — what should the finding/decision be instead?
4. **Estimate confidence** — high (clear evidence), medium (judgment call), low (gut feeling)

When you agree but think the finding is incomplete or understated, say so explicitly. Don't perform agreement to be polite — the goal is to catch errors before we commit to a plan.

When you find new bugs not listed here, add them as Finding 15+ with the same evidence/why-it-matters/verification structure.

---

## Section 7 — Output format expected from the verifying agent

A markdown report with:
- One section per task (A–J), agree/disagree + evidence
- One section per Finding (1–14), confirmed/refuted/needs-correction
- A "new findings" section if applicable
- A final "recommended changes to this report" section

Saved to: `worklog/history/specs/2026-04-28-loop-closure-verification-response.md`.

---

## Appendix — Reproducibility checklist

Before starting, the verifying agent should:
- [ ] Confirm working directory is `/workspace/collab-splats`
- [ ] Confirm current branch is `refactor/core-modules`
- [ ] Record the SHA of HEAD: `git rev-parse HEAD`
- [ ] Record the SHA of `main` for VGGT-SLAM and VGGT-Long they are reading
- [ ] Have access to the network for fetching upstream files (or local clones of both)
- [ ] Have access to GTSAM and DINO-SALAD docs if pose-graph or retrieval claims need digging
