# Loop Closure Refactor — Verification Response

**Date:** 2026-04-28
**Verifier:** Claude Opus 4.7 (3 parallel subagents)
**Local HEAD:** `0cfb069` on `refactor/core-modules`
**VGGT-SLAM SHA read:** `66630f7125622c26f0d8e69fea5135ae2d2e8a58` (main)
**VGGT-Long SHA read:** `c160869d` (main)
**VGGT SHA read:** `44b3afbd` (main)
**Source spec:** `worklog/history/specs/2026-04-28-loop-closure-verification.md`

---

## TL;DR

12 of 14 findings confirmed at high confidence. 1 finding (F7) has a factual error in its premise but the actionable recommendation still stands. 1 finding (F11) is correct in substance but loose in wording. All Tasks A–E and I–J validated. Tasks F–H carry recommendations below.

**No new bugs found.** **One spec correction needed (F7 premise).** **One language tightening (F11).** **One safety upgrade flagged (Task G).**

---

## Section A — Findings 1–14 verification

### Finding 1 — pose_graph::add_submaps no inter-submap edges
**CONFIRMED. High confidence.**
`pose_graph.py:62–74` inner loop guards with `if local_i > 0:` — only intra-submap `BetweenFactorPose3` calls. Outer loop ends with `self._total_frames += k`, zero code linking last frame of submap N to first frame of submap N+1. `add_loop_edges` (lines 78–95) only handles verified LC pairs, does not compensate.

### Finding 2 — Submap initial poses in incompatible local frames
**CONFIRMED. High confidence.**
`pose_graph.py:66`: `self._initial.insert(key, _pose3(pose))` inserts raw submap-local poses. No accumulated transform applied. LM starts from broken initial values.

### Finding 3 — Overlap region duplicates frames as separate nodes
**CONFIRMED. High confidence.** Output shape mismatch demonstrated:

| N | K | O | step | n_submaps | output shape | expected | overflow |
|---|---|---|------|-----------|--------------|----------|----------|
| 100 | 20 | 4 | 16 | 6 | (120, 4, 4) | (100, 4, 4) | +20 |
| 50 | 20 | 4 | 16 | 3 | (58, 4, 4) | (50, 4, 4) | +8 |
| 21 | 20 | 4 | 16 | 2 | (25, 4, 4) | (21, 4, 4) | +4 |

`np.concatenate` at `feedforward.py:261` naively concatenates per-submap arrays. `_total_frames += k` per submap with zero dedup. Downstream `_postprocess` and `build_colmap` consume wrong indices.

### Finding 4 — Loop edge math uses incompatible frames
**CONFIRMED. High confidence.**
`feedforward.py:222–225` stacks `submap.poses[match.query_frame_idx]` and `d_submap.poses[match.detected_frame_idx]` from two different submap-local frames. `pose_graph.py:94` then computes `_relative_pose3` between them — mathematically meaningless. `VGGTXCreator._verify_loop_candidate` re-runs VGGT but discards the fresh same-frame poses.

**Cross-check with VGGT-SLAM (Task D below):** upstream re-runs model on (q,d) pair, builds a 2-frame "lc_submap" carrying the fresh same-frame poses, then both BetweenFactors are derived from those LC-submap poses. Our code does not. **F4 is the bug; the fix is the upstream recipe.**

### Finding 5 — Default `_verify_loop_candidate` returns True
**CONFIRMED. High confidence.**
`feedforward.py:137–139` returns `True`. Only `VGGTXCreator` overrides (lines 786–794). `MapAnythingCreator` inherits the trivial accept. Combined with permissive default `lc_threshold=0.95` (F8), MapAnything path admits many false positives.

### Finding 6 — Same noise model intra + loop
**CONFIRMED. High confidence.**
`pose_graph.py:95` uses `self._intra_noise` for the loop BetweenFactor. VGGT-SLAM upstream `vggt_slam/graph.py:18–21` defines `inner_submap_noise` and `intra_submap_noise` separately (both 15-dim 0.05 sigma — same magnitude but separately addressable). VGGT-Long uses IRLS robust scheme with explicit `IRLS: {delta: 0.1, max_iters: 5}`.

### Finding 7 — Pose convention unverified
**PARTIALLY CONFIRMED. Spec premise wrong; recommendation still valid.**

VGGT `pose_encoding_to_extri_intri` docstring (vggt/utils/pose_enc.py):
> "extrinsics ... representing camera from world transformation"

"camera from world" = **world-to-cam** (W2C). Our `submap.py:14` docstring says "world-to-cam homogeneous" — **matches**. So the spec's claim that "VGGT-typical output is cam-to-world" is **factually wrong**.

However, the actionable recommendation (add an assertion + unit test) is still valid: there's nothing that catches a future regression or a backend mismatch (e.g., MapAnything may differ — VGGT-Long's `base_model.py` even has an internal comment incorrectly labeling extrinsic as C2W, suggesting confusion in the wider ecosystem).

**Action:** rewrite F7 premise: "Convention is world-to-cam per VGGT docstring AND our docstring; no assertion or test guards against backend mismatch or future regression. Add assertion in `_run_inference` per backend; add convention unit test."

### Finding 8 — `lc_threshold=0.95` too permissive
**CONFIRMED. High confidence.**
- Local: `loop_closure.py:24` `lc_threshold: float = 0.95` (L2)
- VGGT-Long: `configs/base_config.yaml::SALAD.similarity_threshold: 0.85` (cosine), with inline comment confirming raised from 0.7 to "significantly reduce false matches"
- Math: cos=0.85 → L2≈0.548; our L2=0.95 → cos≈0.549 — equivalent to VGGT-Long's *original* (rejected) 0.7 cosine, even looser

### Finding 9 — No NMS on loop candidates
**CONFIRMED. High confidence.**
- Local: `LoopMatchQueue` keeps top-k by score; no temporal suppression.
- VGGT-Long `loop_utils/loop_refinement.py::reduce_edges` is `@nb.njit(cache=True)`, sorts by flow magnitude, suppresses ±nms neighbors. `nms_threshold: 25`, `use_nms: True` in config.

### Finding 10 — `max_loops_per_submap=1` overly conservative
**CONFIRMED. High confidence.**
- Local default: `loop_closure.py:25` is 1
- VGGT-Long default: `top_k: 5`

### Finding 11 — No inter-submap Umeyama alignment
**CONFIRMED in substance; spec wording loose.**

- VGGT-SLAM `scale_solver.py::estimate_scale_pairwise` returns a **scalar scale only** (median of per-point distance ratios). The full Sim(3)-like alignment is composed inline in `solver.add_edge` from camera projection matrices: `H_scale = diag(s,s,s,1)` composed with `np.linalg.inv(prior.proj_mats[-1]) @ current.proj_mats[0]`. Inputs are overlap-region 3D points gated by `good_mask` from confidence thresholds. **So: yes, overlap-region 3D points + conf gates yield an inter-submap transform — but the function alone is scalar-only.**
- VGGT-Long `loop_utils/sim3utils.py::estimate_sim3` is closed-form Umeyama (centered SVD) returning `(s, R, t)`. Confidence is consumed at the call-site wrapper (`align_point_maps`, `robust_weighted_estimate_sim3*` pass `all_weights` derived from `pred['conf']`), not in `estimate_sim3` itself.

**Action:** tighten F11 wording to: "VGGT-SLAM's overlap-region pipeline composes inter-submap Sim(3) from a median-scale scalar + camera projection matrices. VGGT-Long's pipeline closed-form Umeyama-on-conf-weighted points. Both consume overlap-region 3D points + per-frame confidence. We have neither." Substantive claim unchanged.

### Finding 12 — No FrameTracker adaptive keyframing
**CONFIRMED. High confidence.**
VGGT-SLAM `vggt_slam/frame_overlap.py::FrameTracker` uses `cv2.goodFeaturesToTrack` for keypoint init + `cv2.calcOpticalFlowPyrLK` for tracking; `compute_disparity` triggers new keyframe when mean disparity exceeds threshold. We have nothing equivalent. Defer to follow-up PR per spec — agree.

### Finding 13 — No GraphMap central submap registry
**CONFIRMED. High confidence.**
VGGT-SLAM `map.py::GraphMap` (8.2 KB) has both `retrieve_best_score_frame` and `retrieve_best_semantic_frame`. We use a plain `list[Submap]`. Defer per spec — agree (bound to ADR 001 reversal).

### Finding 14 — Submap class anemic
**CONFIRMED. High confidence.**
VGGT-SLAM `submap.py` is 10087 bytes (~10 KB) with: `set_lc_status`, `add_all_poses`, `add_all_points(points, colors, conf, conf_threshold_percentile, intrinsics_inv)`, `get_first_homography_world(graph)`, `get_all_frames`, `get_all_retrieval_vectors`, `get_img_names_at_index`, `get_points_in_world_frame(graph)`, `get_all_semantic_vectors`. Ours is a 20-line dataclass. Refactor proposal correctly isolates `submap.py` for growth.

---

## Section B — Tasks B, C, D, E, I, J

### Task B — Pose convention
VGGT `pose_enc.py::pose_encoding_to_extri_intri` returns extrinsic as "camera from world" = world-to-cam. Matches our `Submap.poses` docstring. MapAnything not directly inspected here (deferred). VGGT-Long internally mislabels this as C2W in `base_model.py` — points to a real ecosystem-wide confusion risk. **Add an assertion test per backend before relying on the convention.**

### Task C — Output shape
Confirmed in F3 above. Bug is real, repeatable, deterministic.

### Task D — VGGT-SLAM LC re-run pattern
**CONFIRMED. High confidence.** Upstream `solver.py`:
```python
lc_frames = torch.stack((new_submap.get_frame_at_index(detected_loops[0].query_submap_frame),
                         retrieved_frames[0]), axis=0)
predictions_lc = model(lc_frames, compute_similarity=True)
if image_match_ratio < 0.85:
    predictions_lc = None
else:
    extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(predictions_lc["pose_enc"], ...)
    predictions["extrinsic_lc"] = extrinsic_lc
# 2-frame "LC submap" inserted carrying fresh poses:
self.add_edge(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame, is_loop_closure=False)
self.add_edge(loop.detected_submap_id, loop.detected_submap_frame, lc_submap_num, 1, is_loop_closure=True)
```

Three points all confirmed: (1) re-run on (q,d), (2) BetweenFactors derived from fresh same-frame poses via the lc_submap, (3) original submap-local poses NOT used for the loop edge. Our `feedforward.py:222–225` violates all three. **F4 fix recipe = adopt this pattern.**

Also note: upstream gates LC at `image_match_ratio >= 0.85`. Our `VGGTXCreator._verify_loop_candidate` uses `verify_match_ratio` — confirm threshold parity in plan.

### Task E — Math
Verified:
- cos=0.85 → L2 = 0.5477225575
- cos=0.55 → L2 = 0.9486832981
- L2=0.95 → cos = 0.5487500000

Local default `loop_closure.py:24` `lc_threshold: 0.95` confirmed.

### Task I — VGGT-Long Base3DModel + using_sim3
**CONFIRMED.** `base_models/base_model.py` has `class Base3DModel(ABC)` with concrete `VGGTAdapter`, `Pi3Adapter`, MapAnything adapter. `using_sim3` is a real config flag (`Model.using_sim3` in YAML), consumed at `sim3utils.py` call sites — not just README.

### Task J — VGGT-SLAM PR #39
**State:** open, not merged. Created 2026-04-12 by `stepeos`. Title: `[WIP] Incorrect scale / alignment of depth or world points when using SparkFastVGGT in VGGT-SLAM2.0`.

**Bug claim:** "depth/scale of the reconstructed point cloud appears incorrect" with SparkFastVGGT backend. **CONFIRMED.**

**Author suspicion (verbatim):** "I suspect the issue may originate in one of the following stages: `set_point_cloud`, `get_points_in_world_frame`, or an intermediate transformation step ... possibly SL(4) on top of SE(3)". **CONFIRMED.**

Consistent with the broader "SL(4) corrupts metric scale" justification for our SE(3) choice. Our SE(3) decision is empirically defensible.

---

## Section C — Tasks F, G, H (recommendations)

### Task F — File structure sanity check
6-file proposal sound. Estimated LOC after correctness fixes:

| File | LOC est | Coupling | Verdict |
|------|---------|----------|---------|
| `__init__.py` | 10–20 | none | OK |
| `submap.py` | 80–150 (grow) | numpy | OK — start at 30, grow into upstream-style methods |
| `retrieval.py` | 120–180 | DINO-SALAD, torch | OK — quarantines retrieval dep |
| `alignment.py` | 100–200 | numpy only | OK if Umeyama added — ensure no GTSAM/torch leaks in |
| `pose_graph.py` | 200–280 | gtsam | OK — quarantines gtsam dep, grows w/ 3 noise models + Huber |
| `closure.py` | 250–350 | imports above | OK — orchestration glue |

**No merges or splits recommended.** One callout: keep `closure.py` thin (orchestration only) — if it grows past ~400 LOC, signal that more logic should move into `submap.py` or `alignment.py` as methods.

### Task G — `_verify_loop_candidate` default
**Recommendation: Option A (raise NotImplementedError).**

Reasoning:
- F5 + F6 show false-positive loops corrupt the graph globally (one bad inter-submap edge propagates everywhere via LM).
- Option B (DINO re-embed) duplicates retrieval work for marginal benefit, and adds a hidden compute cost without per-backend tuning.
- NotImplementedError is the cheapest forcing function: it makes adoption of a new backend an explicit decision, not a silent default.
- For MapAnything specifically, the implementation can be a thin DINO-cosine gate (Option B logic) but **explicitly declared** in `MapAnythingCreator._verify_loop_candidate`, not inherited.

This also aligns with VGGT-SLAM upstream's pattern: their image_match_ratio gate is model-specific (uses VGGT's `compute_similarity=True` path), not a base-class default.

### Task H — Option X1 (combined) vs X2 (staged)
**Recommendation: X1 (combined) with disciplined commit splits.**

Reasoning:
- Bugs in F1–F6 all live in files the refactor moves/restructures. X2 means modifying every file twice — once to relocate, once to fix.
- `refactor/core-modules` is already a large landing zone (82 commits ahead of main). One more sectioned PR fits the repo's established review pattern (per WORKLOG: "one PR with sectioned review").
- Risk of X1 = bigger diff to review. Mitigate via commit prefixes: `refactor(lc): split into sub-package`, `fix(lc): add inter-submap Umeyama edges`, `fix(lc): dedup overlap frames`, etc. PR description references each commit per finding.

**Caveat:** if the implementation plan reveals Umeyama integration is itself >500 LOC of new code, split THAT into a follow-up. Refactor + F1+F3+F5+F7 (mechanical fixes) in PR1; F4+F11 (Umeyama, pose-graph rewiring) in PR2. Reassess after writing the plan.

---

## Section D — New findings

**None.** All identified bugs are captured by F1–F14.

Two ecosystem observations worth recording (not bugs per se):

1. **VGGT-Long's `base_model.py` mislabels VGGT extrinsic as C2W in a comment.** Means future contributors copying that pattern could introduce a real C2W/W2C bug. Reinforces Task B recommendation: assertion test per backend.

2. **VGGT-SLAM's `inner_submap_noise` and `intra_submap_noise` are currently the same value (0.05).** They are *separately addressable* but not yet *separately tuned*. We should still split them (per F6 + Section 4 design), but expect that initial sigma values will need tuning data — flag in plan as a "first-tune-after-baseline" task.

---

## Section E — Recommended changes to the source spec

1. **Finding 7 premise rewrite.** Drop "VGGT-typical output is cam-to-world" — it is wrong. Keep the recommendation to add assertion + unit test (justified by ecosystem confusion).

2. **Finding 11 wording.** Change "compute a Sim(3) or SE(3) transform" to: "VGGT-SLAM composes inter-submap alignment from `estimate_scale_pairwise` (median scalar) + camera projection matrices; VGGT-Long uses closed-form Umeyama on conf-weighted overlap points." Substantive claim ("we have nothing equivalent") unchanged.

3. **Section 3 pose-representation table for VGGT-SLAM.** Confirm 15-DOF entry — `BetweenFactorSL4`, 15-dim diagonal noise sigmas in `graph.py:18–21`. No change needed; entry is correct.

4. **Section 4 in-scope item 8.** Change "Default `_verify_loop_candidate` either raises NotImplementedError or implements a model-agnostic gate" to: "Default `_verify_loop_candidate` raises NotImplementedError; each Creator subclass declares a verifier (MapAnythingCreator gets a DINO-cosine gate; VGGTXCreator keeps image_match_ratio)." Removes ambiguity.

5. **Add to Section 4 in-scope.** Threshold parity with upstream: confirm our `verify_match_ratio` default matches VGGT-SLAM's 0.85.

6. **PR-split caveat in Task H.** Note that if Umeyama integration grows past ~500 LOC, refactor + mechanical fixes ship in PR1; pose-graph rewiring (F4 + F11) ships in PR2.

---

## Section F — Reproducibility checklist

- [x] Working directory `/workspace/collab-splats` — confirmed
- [x] Branch `refactor/core-modules` — confirmed
- [x] HEAD SHA `0cfb069` — recorded
- [x] VGGT-SLAM SHA `66630f7` — recorded
- [x] VGGT-Long SHA `c160869d` — recorded
- [x] VGGT SHA `44b3afbd` — recorded
- [x] Network access for upstream raw fetches — used
- [x] Local files read at cited line numbers — done
