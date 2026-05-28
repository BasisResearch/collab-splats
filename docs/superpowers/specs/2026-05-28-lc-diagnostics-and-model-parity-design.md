# LC Diagnostics & Model Parity — Design Spec

**Date:** 2026-05-28  
**Branch:** `refactor/cu121`  
**Status:** Ready for implementation

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

### Convention diagnostic

Three models, three decoders:

| Model | Pose decoder | Expected convention |
|-------|-------------|---------------------|
| VGGT-X | `pose_encoding_to_extri_intri` (vggt pkg) | w2c — documented |
| Omega | `encoding_to_camera` (third_party/vggt-omega) | w2c — structurally identical to VGGT-X |
| MapAnything | `camera_poses` in pred dict | **c2w** — explicitly labeled; already inverted in our code |

Diagnostic script: run all 3 models on a shared 4-frame test set (chess_seq01 frames 0-3).
For each model, print:

```
poses[0]:       should be I (both w2c and c2w; VGGT anchors first frame to I)
poses[1][:3,3]: translation vector of 2nd frame
det(R[0]):      should be +1.0 (proper rotation)
R[0]^T @ R[0]: should be near I
sign of t[2] frame 1 vs frame 0: diagnostic — w2c and c2w differ here
```

Also run the 2-frame LC forward (`extract_intermediate_features`) on frames 0+1 and print
`lc_poses[0]`, `lc_poses[1]` — compare against the full-sequence poses for the same frames.
Mismatch in relative pose (vs known GT relative pose from 7-Scenes) = pose quality issue.

---

### Omega LC diagnostics

Beyond convention check, instrument `_run_lc_loop` (or a dedicated diagnostic run) to log
per-submap LC statistics for Omega:

```
submap N: found=K candidates  verified=J  accepted=I  (rejected by: verify/jump)
```

Also log: the `lc_rel` translation magnitude and direction for each accepted loop, and
compare vs the DINO-SALAD distance of that pair. This tells us whether accepted loops
have geometrically consistent poses.

**Root cause hypotheses (ranked by likelihood):**

1. **LC relative pose quality** — Omega's 2-frame joint `extract_intermediate_features`
   forward produces inaccurate relative poses for non-adjacent frames. The loop edge
   goes into PGO with wrong translation, pulling submap poses in wrong direction.
   Diagnosis: compare `lc_rel` translation vs GT relative pose for same pair.

2. **Scale estimation noise** — Omega's depth maps have different scale distribution
   than VGGT-X. `estimate_scale_pairwise` on sparse Omega world_points (world_points=None
   when MapAnything-style; check if Omega provides world_points) may fall back to scale=1.0
   consistently, making inter-submap edges wrong.
   Diagnosis: log `scale` value per submap transition.

3. **Verify gate too permissive** — even with `_lc_layer_index=16`, Omega might pass
   `verify_match_ratio=0.85` for poor pairs. The 1.328 mtq mean (retrieved mode) is a
   calibration-set average, not a guarantee per-pair.
   Diagnosis: log per-loop `cross_frame_attention_ratio` values, not just pass/fail.

4. **Convention mismatch** — Omega extrinsics are c2w (not w2c). Ruled out by
   `encoding_to_camera` code analysis (structurally identical to VGGT-X w2c decoder),
   but diagnostic confirms.

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
| `evals/runners/run_vggt_slam_lc.py` | Modify | Remove `_VGGTCompatWrapper`; use vggt_spark PYTHONPATH |
| `evals/eval_vggt_slam_comparison.py` | Modify | Add `vggt_slam_lc` condition for 200-frame run |
| `evals/runners/diagnose_lc_conventions.py` | Create | 4-frame convention diagnostic for all 3 models |
| `evals/runners/diagnose_omega_lc.py` | Create | Per-submap LC stats logging for Omega |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Modify | `_forward` windowing fix + `_lc_collate_outputs` fix |

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

1. Convention diagnostic (no code changes, 30 min) — eliminates/confirms hypothesis 4 for Omega.
2. Omega LC stats logging (30 min) — narrows root cause to hypothesis 1, 2, or 3.
3. VGGT-SLAM run 3 fix + GPU run (1 hour wall clock, unblocks comparison table).
4. MapAnything `_forward` + `_lc_collate_outputs` (2 hours, self-contained).
5. Omega fix based on diagnosed root cause (1-4 hours depending on cause).
6. Final ATE eval run for all conditions.
