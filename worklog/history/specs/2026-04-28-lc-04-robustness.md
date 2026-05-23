# Spec 4 — Robustness Layers + ADR Batch

**Branch:** `lc-04-robustness` (worktree from `refactor/core-modules` post-spec-3 merge)
**Predecessor:** `lc-03-posegraph`
**Successor:** none (full sequence done — open PR to main)
**Risk:** medium (tuning + hard rejects, behavior changes within accepted/rejected loop boundary)
**Findings addressed:** F6, Section 4 items 10, 15, 16, 17 + ADRs 002–005

---

## Goal

Layer robustness defenses on the now-correct pose graph:
- Three-way noise split with concrete starting sigmas (F6, item 17)
- Huber robust kernel on loop edges (item 10)
- Translation-jump pre-add gate — SE(3) port of MR.ScaleMaster anchor-scale alarm (item 15)
- Loop-edge down-weighting via translation-magnitude kernel (item 16)
- Orchestration extraction: move LC orchestration from `feedforward.py` to `closure.py`
- ADR 002–005 records (deferred decisions)

After this spec, sequence is done → open PR `refactor/core-modules` → `main`.

---

## Tasks

### F6 + item 17 — Three-way noise split

**File:** `collab_splats/pointcloud/loop_closure/pose_graph.py`.

**Change:** replace `_intra_noise` with three named noise models:
- `_odom_intra_noise` — used for intra-submap consecutive-frame BetweenFactors
- `_loop_intra_noise` — used for loop edges within the same submap (rare, but possible if LC pair both belong to one submap)
- `_loop_inter_noise` — used for loop edges between different submaps (the common case)
- `_inter_noise` — used for inter-submap edges from spec 3 (was placeholder = `_intra_noise`); now properly tuned

Sigmas (starting values, ADR 005 trigger field tuning later):
```python
# translation σ (m), rotation σ (rad)
odom_intra:  σ_t=0.05, σ_r=0.02   # tight, VGGT/MapAnything intra-window is well-conditioned
inter:       σ_t=0.20, σ_r=0.05   # looser at submap boundary (Umeyama RMSE captures this)
loop_intra:  σ_t=0.30, σ_r=0.10   # rarely used
loop_inter:  σ_t=0.50, σ_r=0.15   # loosest, model re-run can fail; let robust kernel handle outliers
```

These sigmas are 6-dim diagonal; convert via GTSAM `Diagonal.Sigmas([σ_r]*3 + [σ_t]*3)`.

**Single-robot mode:** all loops are intra-session. `loop_inter` is the active loop-noise bucket. Future multi-session work would distinguish further.

**Acceptance:** unit test inspects `PoseGraph` instance and confirms 4 distinct noise models exist.

### Item 10 — Huber kernel on loop edges

**File:** `collab_splats/pointcloud/loop_closure/pose_graph.py:add_loop_edges`.

**Change:** wrap loop noise model with GTSAM `noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(k=1.0), base_noise)`. Apply only to loop edges, NOT intra/inter.

**Acceptance:** unit test injects one outlier loop edge; without Huber, LM result is corrupted; with Huber, LM result tracks ground truth.

### Item 15 — Translation-jump pre-add gate

**File:** `collab_splats/pointcloud/loop_closure/closure.py` (populate this module — orchestration moves here in this spec).

**Add:**
```python
def translation_jump_check(
    submaps: list[Submap],
    query_idx: int, query_frame: int,
    detected_idx: int, detected_frame: int,
    lc_relative_pose: np.ndarray,
    max_jump_ratio: float = 0.2,
) -> tuple[bool, float]:
    """Reject loop if implied correction translation exceeds fraction of odom-path length.

    Returns (accept, ratio). Accept iff ratio < max_jump_ratio.
    """
```

Implementation:
1. Compute `T_odom_path` = composition of intra+inter edges from query node → detected node
2. Compute `path_length` = sum of `‖t‖` along that path
3. Compute `ΔT = lc_relative_pose ∘ T_odom_path^{-1}`
4. Compute `ratio = ‖ΔT.translation‖ / max(path_length, ε)`
5. Reject if `ratio > max_jump_ratio`

**Position in pipeline:** after `_verify_loop_candidate` returns accept, before `add_loop_edges` is called. Logged regardless (accepted ratio + rejected ratio for telemetry).

**Acceptance:**
- Synthetic test: loop with `‖ΔT.t‖ = 0.5 · path_length` → rejected.
- Synthetic test: loop with `‖ΔT.t‖ = 0.05 · path_length` → accepted.

### Item 16 — Loop-edge down-weighting

**File:** `collab_splats/pointcloud/loop_closure/pose_graph.py:add_loop_edges`.

**Change:** for each loop edge, scale the noise sigma by `1 / (1 + α · ‖t_loop‖²)` where `α = 0.1` (configurable per backend, ADR 005 trigger). Long-translation loops get softer (larger sigma) — counters bad long-range matches without rejecting them.

Apply BEFORE Huber wrap (so Huber sees the down-weighted base).

**Acceptance:** unit test with two loop edges (short + long); inspect resulting noise sigmas; long-edge sigma > short-edge sigma by expected ratio.

### Orchestration extraction → `closure.py`

**Files:**
- Move `_run_loop_closure_inference`, `_loop_close`, `_merge_submap_outputs` from `feedforward.py` (lines 130-280) into `closure.py::run_loop_closure(...)`.
- `BaseFeedforwardCreator.reconstruct` calls `run_loop_closure(...)` instead of inline orchestration.
- Keep `_verify_loop_candidate` as a Creator method (per-backend), passed into `run_loop_closure` as a callable.

`run_loop_closure` signature:
```python
def run_loop_closure(
    submaps: list[Submap],
    config: LoopClosureConfig,
    retrieval: ImageRetrieval,
    verify_fn: Callable[[Frame, Frame], tuple[bool, np.ndarray | None]],
    *,
    log: Logger | None = None,
) -> np.ndarray:
    """Returns corrected (N, 4, 4) extrinsics."""
```

**Acceptance:** `feedforward.py` LC orchestration code is deleted; `closure.py` replaces it; integration tests still pass.

### ADR batch (ADR 002–005)

**Files:**
- `worklog/decisions/002-defer-sl4.md`
- `worklog/decisions/003-defer-graphmap.md`
- `worklog/decisions/004-defer-frametracker.md`
- `worklog/decisions/005-defer-per-backend-noise-tuning.md`

Each ADR follows existing template (see `worklog/decisions/001-defer-declip-integration.md`):
- Status: Accepted
- Context: from verification response Section 5
- Decision: defer
- Trigger condition: explicit
- Consequences: explicit

---

## Files touched

- `collab_splats/pointcloud/loop_closure/pose_graph.py` (3-way noise, Huber, down-weighting)
- `collab_splats/pointcloud/loop_closure/closure.py` (orchestration + translation-jump gate)
- `collab_splats/pointcloud/feedforward.py` (orchestration removed, calls `run_loop_closure`)
- `worklog/decisions/002-defer-sl4.md` (new)
- `worklog/decisions/003-defer-graphmap.md` (new)
- `worklog/decisions/004-defer-frametracker.md` (new)
- `worklog/decisions/005-defer-per-backend-noise-tuning.md` (new)
- `tests/pointcloud/test_pose_graph.py` (extend — noise split, Huber)
- `tests/pointcloud/test_translation_jump.py` (new)
- `tests/pointcloud/test_loop_closure_integration.py` (extend — outlier loop with Huber)

---

## Verification gate

1. **Unit tests pass:**
   ```bash
   pytest tests/pointcloud/test_pose_graph.py -v
   pytest tests/pointcloud/test_translation_jump.py -v
   ```

2. **Integration test pass:**
   ```bash
   pytest tests/pointcloud/test_loop_closure_integration.py -v
   ```

3. **Behavior assertions (concrete):**
   - **F6 split:** PoseGraph has 4 distinct noise models.
   - **Huber:** outlier loop test — LM result tracks ground truth (max ATE < 0.1m on synthetic 3-submap).
   - **Item 15:** translation-jump rejects bad loop, accepts good loop per spec.
   - **Item 16:** long-loop sigma > short-loop sigma per expected ratio.
   - **Orchestration:** `feedforward.py` line count drops by ~150 lines; `closure.py` populated; integration tests still green.
   - **End-to-end ATE robustness:** inject 1 outlier loop into synthetic 3-submap test. Without spec 4 layers: LM diverges or ATE >0.5m. With spec 4: ATE < 0.1m.

4. **Regression sweep:**
   ```bash
   pytest tests/ -v
   ```
   No previously-passing test fails. Pre-existing nerfstudio env failures + GPU smoke test exempt (per WORKLOG).

5. **Reviewer signoff** via `superpowers:requesting-code-review`. **Reviewer must validate ADR triggers are concrete and falsifiable.**

6. **Coupling re-check:**
   ```bash
   grep -E "^(import|from) (gtsam|torch|DINO)" collab_splats/pointcloud/loop_closure/alignment.py  # empty
   grep -E "^(import|from) (torch|DINO)" collab_splats/pointcloud/loop_closure/pose_graph.py  # empty
   ```

7. **Squash-merge** into `refactor/core-modules` with commit:
   ```
   feat(lc): robustness layers + ADRs 002-005

   - 3-way noise split: odom-intra / inter / loop-intra / loop-inter (F6, item 17)
   - Huber robust kernel on loop edges (item 10)
   - translation-jump pre-add gate, SE(3) port of MR.ScaleMaster anchor alarm (item 15)
   - loop-edge down-weighting via translation-magnitude kernel (item 16)
   - LC orchestration extracted into closure.py
   - ADR 002 (SL(4) defer), 003 (GraphMap defer), 004 (FrameTracker defer), 005 (per-backend noise tuning defer)
   ```

8. **Update `worklog/WORKLOG.md`** — record merge SHA, mark sequence complete, signal PR-to-main readiness.

9. **Open PR** `refactor/core-modules` → `main` via `gh pr create`. Body summarizes the 4-spec sequence + sectioned review per WORKLOG pattern.

---

## Out of scope

- SL(4) implementation (ADR 002 deferred)
- GraphMap (ADR 003 deferred)
- FrameTracker (ADR 004 deferred)
- Per-backend noise field-tuning (ADR 005 deferred)
- Standalone `submap_size` parameter without LC
- Semantic search reversal (ADR 001)
