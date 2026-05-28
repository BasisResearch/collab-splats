# LC Diagnostics & Model Parity — Design Spec

**Date:** 2026-05-28  
**Branch:** `refactor/cu121`  
**Status:** Empirically investigated — findings below, implementation plan updated

---

## Empirical Findings (2026-05-28 investigation)

### Convention — resolved, NOT the issue

Both VGGT-X and Omega use **world-to-camera (w2c)** extrinsics, confirmed from source:
- Omega `extri_intri_to_pose_encoding` docstring: *"camera-from-world matrices in OpenCV coordinates"*
- VGGT-X docstring: *"representing camera from world transformation"*
- `encoding_to_camera` and `pose_encoding_to_extri_intri` are symmetric inverses of the same encoding

MapAnything outputs **cam-to-world (c2w)** `camera_poses`, already correctly inverted in `_lc_collate_outputs` and `_postprocess`. No convention bug in any model.

### Scale drift — new critical finding

`estimate_scale_pairwise` formula is **identical** between our code and VGGT-SLAM (`median(||Y||/||X||)`). VGGT-SLAM returns `(scale, None)` and takes `[0]`; ours returns `float` directly. Same math.

Source of bias: **VGGT depth anchor-frame overestimation**.

When the same physical frame is the first frame of a new VGGT window (anchor), VGGT systematically overestimates its depth vs when it was the last frame of the previous window. Implied depth ratio: 1.015–1.198× per boundary, compounding to **2.76× total** across 13 boundaries (cumulative scale product = 0.363).

| Metric | Ours (VGGT-X) | VGGT-SLAM |
|--------|--------------|-----------|
| Scale factors | 0.835–0.985 (all <1) | 0.976–1.011 (near 1) |
| Cumulative scale | **0.363** | **0.919** |

VGGT-SLAM avoids this by comparing camera-space points (curr frame 0) directly against prev world-space points (prev frame 15) using `P_temp = inv(K_prev) @ K_curr ≈ I`. This accidentally works for chess_seq01 (tiny inter-submap motion vs scene depth) but is not geometrically general. Our approach (full SE3 T) is geometrically correct but exposes the depth anchor bias.

**Fix required**: switch from norm-ratio scale estimation to pairwise-distance-ratio estimation: `median(||Y_i - Y_j|| / ||X_i - X_j||)`. This is invariant to coordinate system translation and removes the anchor-bias sensitivity.

### Omega LC regression root cause — resolved

Primary cause: **verify gate is effectively disabled for Omega**.

- Omega `cross_frame_attention_ratio` at layer=16: mtq=1.328 on retrieved pairs
- Threshold: 0.85
- Result: **virtually all DINO-SALAD retrieved pairs pass the verify gate**

The attention similarity measure for Omega's `inter_frame_blocks` does not distinguish well between geometrically-correct and geometrically-incorrect frame pairs at threshold 0.85. Loop candidates accepted are based on DINO-SALAD retrieval alone (no geometric filtering).

Additionally:
- `max_jump_ratio=inf` (default): no geometric sanity check — wrong loop edges enter PGO unchecked
- 2-frame Omega relative pose quality for non-adjacent frames: unknown, but likely poor (chess repetitive texture → false retrievals accepted)
- Same depth anchor drift as VGGT-X: scale drift compounded by wrong LC corrections

**Primary fix**: raise `default_verify_match_ratio` for Omega (e.g., 0.99 or a model-specific threshold). Also: enable `max_jump_ratio` (e.g., 0.2) as a safety net.

**Secondary fix**: fix the scale estimation bias (pairwise distance ratio) to address the 0.363 cumulative scale issue.

### VGGT-X parity gap vs VGGT-SLAM — identified

In the parity run, 0/13 DINO-SALAD pairs (from VGGT-SLAM's retrieval) scored > 0.85. VGGT-SPARK's `image_match_ratio` gives 1.001–1.043 for the same pairs. Same algorithm, different model weights → different attention distribution. The successful 2026-05-28 VGGT-X LC run used OUR DINO-SALAD retrieval which finds different pairs — some of those DO exceed 0.85.

VGGT-SLAM LC baseline (30 sparse keyframes) is NOT comparable to our 200-frame run. A proper comparable baseline still requires running VGGT-SLAM on all 200 frames.

---

## Context

VGGT-X + LC works: ATE 0.3184m → 0.2664m (−16.3%). Two open failures:

1. **Omega LC regression** — LC worsens Omega ATE +54% (0.3225m → 0.496m). Root cause unknown.
2. **MapAnything windowed LC broken** — `_forward` ignores the window slice arg; architectural bug.

Plus: VGGT-SLAM comparison run (run 3) never completed — crashed on `compute_similarity=True`
TypeError. The only VGGT-SLAM baseline we have used 30 keyframes vs our 200-frame setup —
not comparable.

---

## Goals

1. Get a comparable VGGT-SLAM ATE baseline (same 200 frames, same scene).
2. Definitively identify Omega LC regression root cause via diagnostics.
3. Fix MapAnything windowed `_forward` + `_lc_collate_outputs`.
4. Confirm or rule out extrinsic convention mismatch for Omega and MapAnything.

---

## Scope and non-goals

**In scope:**
- Diagnostic scripts (read-only analysis)
- VGGT-SLAM run fix + re-execution
- MapAnything `_forward` windowing fix
- MapAnything `_lc_collate_outputs` camera_poses fix
- Omega convention check + conditional inversion if needed
- ATE comparison table across all conditions

**Not in scope:**
- Changing submap size / LC thresholds (that's calibration, done)
- PR #39 world_points convention refactor (not causing active bugs)
- Dashboard or visualization changes

---

## Architecture

### VGGT-SLAM comparison fix

`eval_vggt_slam_comparison.py` already injects `vggt_spark` PYTHONPATH (fixes the
`compute_similarity=True` TypeError). Run 3 failed because the wrong runner script
(`evals/runners/run_vggt_slam_lc.py`) had a dead `_VGGTCompatWrapper` that bypassed
the real similarity check. Fix: remove wrapper, use vggt_spark PYTHONPATH, run the
solver on the exact same 200 frames used in our pipeline.

Key parameter: VGGT-SLAM processes frames sequentially (online), while we batch.
For apples-to-apples comparison:
- Pass all 200 frames to the VGGT-SLAM solver in its natural sliding-window mode.
- Use the same `chess_seq01` seq dir as our eval.
- Save TUM trajectory to `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum`.
- Compute ATE via `evo_ape` against GT.

---

### Convention — resolved (no implementation needed)

Empirically confirmed: Omega = w2c, MapAnything = c2w (already inverted). No code changes
required for convention. The original diagnostic script idea is no longer needed.

---

### Scale estimation fix

Replace norm-ratio with pairwise-distance-ratio in `estimate_scale_pairwise` call site
inside `run_pose_graph_optimization`. The fix should NOT change the shared function signature
(VGGT-SLAM uses it too). Instead, pre-process points before calling:

```python
# Before: scale = estimate_scale_pairwise(curr_in_prev[mask], prev_pts[mask])
# After: compute pairwise distances, then median ratio

def _estimate_scale_pairwise_dist(X: np.ndarray, Y: np.ndarray) -> float:
    """Pairwise distance ratio — invariant to coordinate origin, removes anchor-bias."""
    if X.shape[0] < 2:
        return 1.0
    # Sample random pairs to avoid O(N^2)
    rng = np.random.default_rng(42)
    n = min(X.shape[0], 500)
    idx = rng.choice(X.shape[0], (n, 2), replace=True)
    i, j = idx[:, 0], idx[:, 1]
    x_dists = np.linalg.norm(X[i] - X[j], axis=1)
    y_dists = np.linalg.norm(Y[i] - Y[j], axis=1)
    valid = x_dists > 1e-6
    return float(np.median(y_dists[valid] / x_dists[valid])) if valid.any() else 1.0
```

Add this as a private helper in `closure.py`. Replace the `estimate_scale_pairwise` call in
`run_pose_graph_optimization` with `_estimate_scale_pairwise_dist`.

---

### Omega LC verify gate fix

The verify gate is effectively disabled for Omega (mtq=1.328 >> 0.85 threshold).

Fix: add `default_verify_match_ratio: ClassVar[float] = 0.99` to `VGGTOmegaCreator`.
The `LoopClosure.__init__` already reads this via `getattr(base, "default_verify_match_ratio", None)`.

Additionally: run Omega LC with `max_jump_ratio=0.3` (passed via `LoopClosureConfig`) to
enable the geometric sanity check. Add `default_max_jump_ratio: ClassVar[float] = 0.3` to
`VGGTOmegaCreator` and read it in `LoopClosure.__init__` the same way as `verify_match_ratio`.

These two changes together should significantly reduce false-positive loop edges for Omega.
After fixing, re-run eval to measure ATE. If still regressing, the issue is 2-frame pose
quality — in that case, the deeper fix is to log per-loop `lc_rel` vs GT and assess whether
a higher threshold (0.999) or a pose-consistency check is needed.

---

### Omega LC statistics logging

Add debug logging to `_run_lc_loop` (already has `console.log` for accepted loops, but needs
counts for all stages):

```python
# After find_loop_closures:
log.debug("submap %d: retrieved=%d", wi, len(loop_matches))
# After verify loop, inside the for match loop:
log.debug("  verify ratio=%.3f %s", ratio, "PASS" if verify_ok else "FAIL")
# After jump check:
log.debug("  jump ratio=%.3f %s", jump_ratio, "PASS" if jump_ok else "FAIL")
```

This is already partially present (console.log for rejections). Just ensure rejected loops
log the actual `cross_frame_attention_ratio` value, not just "rejected".

---

### MapAnything `_forward` windowing fix

**Bug:** `_forward(model, views)` ignores `views`, always runs `self._processed_views`
(all N frames). This causes shape mismatches and wrong poses in every submap.

**Fix:** Override `_forward` in `MapAnythingCreator` to use the passed `views` slice:

```python
def _forward(self, model: Any, views: Any, **kwargs: Any) -> list[dict]:
    # views: (K, C, H, W) tensor — window slice from _run_lc_loop
    # Build MapAnything view dicts for this window's frames only.
    device = next(model.parameters()).device
    raw_views = [
        {"img": f.unsqueeze(0), "data_norm_type": ["dinov2"]}
        for f in views.cpu()
    ]
    window_views = preprocess_input_views_for_inference(
        validate_input_views_for_inference(raw_views)
    )
    for view in window_views:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device)

    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            return model.forward(
                window_views,
                memory_efficient_inference=True,
                minibatch_size=self.minibatch_size,
            )
```

The full-sequence path (non-LC) must still work: `run_inference` calls `_forward(model, views)`
where `views` is the full preprocessed tensor. But MapAnything's full-sequence `_forward`
currently expects `self._processed_views` (already validated+preprocessed dicts), not a raw
tensor. Fix: detect tensor vs list input — if `isinstance(views, list)`, use directly (full
sequence path from `_preprocess`); if `isinstance(views, torch.Tensor)`, build from scratch
(LC window path).

---

### MapAnything `_lc_collate_outputs` fix

**Bug:** The stub tries to read `p["camera_poses"]` from the raw `model.forward()` output.
But `camera_poses` only exists after `postprocess_model_outputs_for_inference`, not in
the raw dict.

**Fix:** Run minimal postprocess inside `_lc_collate_outputs`:

```python
def _lc_collate_outputs(self, raw_list: list[dict]) -> dict:
    # Cast bf16 before postprocess (same as _postprocess)
    for pred in raw_list:
        pred["pts3d_cam"] = pred["pts3d_cam"].float()
        pred["pts3d"] = pred["pts3d"].float()
    processed = postprocess_model_outputs_for_inference(
        raw_list,
        self._processed_views[:len(raw_list)],  # match window size
        apply_mask=False,  # skip masking — LC only needs poses
    )
    exts = np.stack([
        invert_poses(p["camera_poses"][0].cpu().float().numpy())[:3, :4]
        for p in processed
    ])
    intrs = np.stack([p["intrinsics"][0].cpu().float().numpy() for p in processed])
    return {"extrinsic": exts, "intrinsics": intrs}
```

The `apply_mask=False` skips confidence/edge masking — we only need poses for LC.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `collab_splats/pointcloud/loop_closure/closure.py` | Modify | Add `_estimate_scale_pairwise_dist` helper; replace call site |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | Modify | `default_verify_match_ratio=0.99`; `default_max_jump_ratio=0.3` |
| `collab_splats/pointcloud/wrappers.py` | Modify | Read `default_max_jump_ratio` from base creator; add per-loop ratio debug logging |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Modify | `_forward` windowing fix + `_lc_collate_outputs` fix |
| `evals/runners/run_vggt_slam_lc.py` | Modify | Remove `_VGGTCompatWrapper`; use vggt_spark PYTHONPATH for 200-frame comparable run |
| `tests/pointcloud/test_scale_estimation.py` | Create | Unit tests for pairwise-dist scale estimator (anchor-bias regression) |

---

## Success Criteria

- VGGT-SLAM ATE on 200-frame chess_seq01 measured and added to table.
- Omega convention confirmed (w2c or c2w) from diagnostic output.
- Omega LC regression root cause identified (one of the 4 hypotheses confirmed/eliminated).
- MapAnything windowed LC runs end-to-end without crashes or shape mismatches.
- MapAnything + LC ATE measured on chess_seq01.

### Target ATE table (post-implementation)

| Condition | ATE RMSE | Status |
|-----------|----------|--------|
| VGGT-X baseline (s=16, 200fr) | 0.3184m | ✓ done |
| VGGT-X + LC (s=16, 200fr) | 0.2664m | ✓ done |
| VGGT-SLAM LC (200fr, comparable) | TBD | pending run 3 |
| Omega baseline (s=16, 200fr) | 0.3225m | ✓ done |
| Omega + LC (fixed) | TBD | pending root cause fix |
| MapAnything baseline | 0.1103m | ✓ done |
| MapAnything + LC (windowing fixed) | TBD | pending _forward fix |

---

## Implementation order

1. **Scale fix** (1 hour) — add `_estimate_scale_pairwise_dist` to `closure.py`, unit test.
   Expected: scale factors near 1.0, cumulative product > 0.9. Re-run VGGT-X eval.
2. **Omega verify gate + jump check** (30 min) — add class vars to `VGGTOmegaCreator`,
   wire `default_max_jump_ratio` in `LoopClosure.__init__`. Add LC ratio debug logging.
   Re-run Omega eval with `--conditions baseline lc` to measure improvement.
3. **MapAnything `_forward` + `_lc_collate_outputs`** (2 hours, self-contained).
   Re-run MapAnything eval.
4. **VGGT-SLAM comparable baseline** — remove `_VGGTCompatWrapper`, run on 200 frames.
   Add result to ATE table.
5. **Omega deep debug** (if step 2 doesn't fully fix regression) — log per-loop ratio
   and lc_rel vs GT, assess 2-frame pose quality.
