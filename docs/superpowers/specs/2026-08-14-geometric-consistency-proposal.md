# Proposal — geometric consistency for feedforward reconstructions

**Date:** 2026-08-14
**Inputs:** `2026-08-14-accuracy-followups-handoff.md`, `2026-08-13-multiview-confidence-measured-report.md`, code audit of `pointcloud/feedforward/base.py`, `localization/`, `geometry/bundle_adjustment.py`, installed pycolmap 4.0.4.
**Status:** proposal for evaluation. No item is approved; each is written so its worth can be judged independently.

---

## Part 1 — What the multiview confidence filter does and does not impose

Mechanism (`compute_multiview_depth_confidence`, `pointcloud/feedforward/base.py:488-660`):
unproject view *i*'s depth to world, project into every overlapping view *j* (frustum-AABB
pair gate), nearest-sample *j*'s depth, count *j* as an inlier when the depths agree within
`abs + rel·d`. Occluded partners leave the denominator; free-space violations stay as
outliers. `multiview_mask` thresholds the **inlier count** (`min_views`, clamped per pixel)
and is ANDed with the learned-confidence percentile.

What it imposes:

- **Pairwise, per-pixel depth agreement** — cross-view consistency of the depth maps with
  each other, aggregated as a count of agreeing partners.
- **An asymmetric free-space constraint** — a partner seeing *through* a point is evidence
  against it; a partner blocked by nearer geometry is merely silent.

What it does **not** impose — the gaps this proposal targets:

- **It only deletes, never corrects.** Output is a boolean mask; no depth, pose, or point is
  ever adjusted.
- **It is self-referential.** Both sides of every comparison come from the same network's
  depth head. A coherently biased reconstruction scores 1.0. (The measured report's
  bad-frame finding — one frame at median ratio 0.136 with 9.9 partners — shows it catches
  *internal* inconsistency only.)
- **No correspondence/track constraint.** Only depth values are compared; there is no notion
  of "the same physical point observed in k views", no feature matching, no transitivity.
- **No baseline gate.** Two co-located views count as independent agreeing evidence
  (`test_compute_mv_conf_identical_cameras` asserts the degenerate case scores 1.0).
- **Measured standing:** ships `false`; on chess/seq-01 it never beat the learned percentile
  at matched retention, though it is ~2.6× more pixel-efficient at its best point and
  two-thirds disjoint from the learned filter (Jaccard ≈ 0.2) — complementary, not redundant.

## Part 2 — Can the localization module supply tracks?

**Yes — it is ~80% of a model-agnostic track source. One small blocking gap.**

Already in place:

- Keypoints are extracted **once per image** and cached (`CameraLocalizer._frame_features`,
  persisted CSR-style to `local_features/{extractor}/…` in `feedforward.zarr`). So
  `(frame_idx, keypoint_idx)` is already a stable global node identity — the hard
  prerequisite for chaining pairwise matches into tracks.
- Four model-agnostic matchers behind one interface (`extractors.py`: DISK, XFeat, LoMa,
  LomaG, all LightGlue-family), independent of any reconstruction backbone — exactly what
  breaks the self-referential circularity above.
- Retrieval extractors (`retrieval.py`: DinoSalad, PECLIP) already produce global
  descriptors for pair shortlisting — today used only by loop closure.
- 2D→3D machinery (`sample_world_points`, `pycolmap.estimate_and_refine_absolute_pose`)
  proves the pycolmap interop path works.

Missing:

1. **`MatchResult` discards keypoint indices.** Every matcher computes `idx_q/idx_db` and
   throws them away, returning only pixels (`extractors.py:167-173` and siblings). Without
   indices, matches cannot be chained. ~5-line change per extractor.
2. **No track container / union-find chaining**, and no DB×DB all-pairs driver (the
   localizer only matches query-vs-DB).
3. `XFeatStarExtractor` returns subpixel-refined semi-dense coords with no indices — not
   trackable as-is; exclude it from the track path.

Current BA track path, for contrast: `geometry/bundle_adjustment.py` uses
`vggt.dependency.track_predict` — **VGGT-locked** (needs the model's conf + world_points),
and its "3D points" are *sampled from the model's own world_points, never triangulated*.
`grep -i triangulat` across `collab_splats/` + `evals/`: zero implementations. And
`build_pycolmap_reconstruction` writes every point with an **empty `pycolmap.Track()`**,
documented as "cannot be used as input to COLMAP BA" — the exact hole a track source fills.

## Part 3 — The industry standard (pycolmap), verified

The user's framing is correct. The standard recipe for seeding a reconstruction from
feedforward poses is COLMAP's **known-pose point triangulator**: VGGT's own
`demo_colmap.py`, VGGSfM, and hloc's `triangulation.py` all export predicted poses/
intrinsics to COLMAP format, then triangulate matched features with **poses fixed** to get a
geometrically verified sparse cloud (this is also the standard 3DGS seed). Installed
pycolmap 4.0.4 exposes everything needed:

- `pycolmap.triangulate_points(reconstruction, database_path, image_path, out, …)` — the
  `colmap point_triangulator` equivalent; poses stay fixed.
- `IncrementalTriangulator` with `retriangulate()`, `complete_all_tracks()`,
  `merge_all_tracks()` — COLMAP's consistency loop: after each global BA it re-triangulates
  under-reconstructed pairs, extends tracks that previously failed, merges duplicate points,
  then filters. This complete→merge→BA→filter cycle is what "geometric consistency" means in
  the standard pipeline.
- Standard gates (defaults): **4.0 px** max reprojection error, **1.5°** minimum
  triangulation angle, 2.0° create/continue angle error, negative-depth rejection,
  `ignore_two_view_tracks=True`. Filtering entry point:
  `ObservationManager.filter_all_points3D(max_reproj_error, min_tri_angle)`.
- Ingredients: a COLMAP `database.db` (keypoints, matches, two-view geometries — importable
  from our matchers via `pycolmap.verify_matches` for geometric verification), plus a
  `Reconstruction` holding cameras and fixed poses with zero points.

Note the contrast with our mv filter: COLMAP's gates are **reprojection error and
triangulation angle against an independently triangulated point** — externally anchored —
where mv compares the model's depth against the model's depth. The two are complementary,
not competing.

---

## Proposal items

Ordered so each item is independently evaluable. P0 items are prerequisites from the
handoff (measurement hygiene, not new geometry); the P1→P4 chain is the geometric-
consistency core; P5/P6 are follow-ons unlocked by it. Costs assume reuse as stated.

### P0a. Re-baseline `depth_trunc` (handoff Workstream 1)
- **What:** re-mesh at the true shipping `depth_trunc: 1.0` (docs claim 2.0); evaluate a
  percentile-of-depth-distribution form, the only scale-invariant option for non-metric
  backbones.
- **Why:** every downstream mesh claim is measured through this cut; currently mislabeled.
- **Cost:** hours; zero new code for the measurement. **Risk:** none.

### P0b. Sweep the learned-confidence percentile (handoff Workstream 2)
- **What:** grid-sweep `conf_threshold` (10…80) per backend on the existing GT-depth
  harness (`eval_multiview_conf.py`); re-run the mv sweep on chess/seq-02.
- **Why:** our strongest filter; three of four defaults are round numbers with no recorded
  provenance. Must be settled before layering anything on top.
- **Cost:** ~1 day, mostly compute; harness exists. **Risk:** none.

### P1. Plumb keypoint indices through `MatchResult`
- **What:** add `idx_q`, `idx_db` (and match scores) to `MatchResult`; populate in Disk,
  XFeat, LoMa, LomaG (each already computes them). XFeatStar explicitly excluded.
- **Why:** the single blocking gap between "pairwise matcher" and "track source". Also
  benefits localization diagnostics for free.
- **Cost:** small (dataclass + 4 call sites + tests). **Risk:** none — additive fields,
  existing consumers untouched.

### P2. Track builder over existing matchers
- **What:** new small module (`geometry/tracks.py`): select pairs (frame adjacency for
  video + `DinoSalad` retrieval for loop pairs — both exist), match with a chosen
  `BaseLocalExtractor`, chain matches into tracks by union-find over
  `(frame_idx, keypoint_idx)`. Reuse the localizer's zarr feature cache so extraction stays
  decode-once.
- **Why:** produces the model-agnostic multi-view correspondences that neither mv confidence
  nor the VGGT-locked `track_predict` provides.
- **Cost:** moderate (one module + tests; no new deps). **Risk:** low; pure addition.

### P3. Known-pose triangulation via `pycolmap.triangulate_points`
- **What:** export P2 matches into a COLMAP database (`pycolmap.verify_matches` for
  two-view geometric verification), build the `Reconstruction` from our existing
  `build_pycolmap_reconstruction` cameras/poses (points cleared), run
  `triangulate_points` + `filter_all_points3D(4.0, 1.5)`. Optionally a point-only BA
  (poses/intrinsics constant).
- **Why:** this *is* the industry standard (Part 3). Yields a sparse cloud whose every point
  passed reprojection + triangulation-angle gates against fixed feedforward poses —
  externally verified geometry, fills the empty-`Track()` hole, and is the correct 3DGS/
  splat seed. Breaks the self-referential loop: verification no longer depends on the
  model's own depth.
- **Cost:** moderate; pycolmap already installed and used. **Risk:** low-moderate — needs
  the DB-import plumbing done carefully (camera models, pixel conventions at model vs
  original resolution — the 2026-08-11 regression class; the principal-point guard pattern
  applies).
- **First experiment (falsifiable):** on chess/seq-01, triangulate with fixed
  feedforward poses and score triangulated points against GT depth vs the model's own
  filtered world_points. If triangulated points are not measurably more accurate where the
  two disagree, the ceiling is low and P4/P5 lose their premise.

### P4. Triangulation-anchored depth audit (headroom for handoff Workstream 3)
- **What:** compare P3's triangulated depths against model depth per frame — an *external*
  version of the consensus-depth headroom measurement the handoff recommends before any
  estimator work. Optionally derive a per-scene depth scale/offset correction for
  non-metric backbones.
- **Why:** answers "is the model's depth biased where views disagree?" with evidence that
  does not share the model's bias. Directly gates whether Workstream 3 (consensus depth
  estimator) is worth building.
- **Cost:** small once P3 exists (analysis script beside `eval_multiview_conf.py`).
  **Risk:** none — measurement only.

### P5. Backbone-agnostic BA tracks
- **What:** let `BundleAdjustment` consume P2 tracks (+ P3 triangulated seeds) as an
  alternative to `vggt.dependency.track_predict`.
- **Why:** BA currently works for VGGT-family only and is recorded as a no-op on
  small-baseline scenes; model-agnostic tracks with triangulated (not depth-sampled) 3D
  seeds are the textbook fix, and align with the recorded BA-generalization intent
  (`project_ba_future_generalization`).
- **Cost:** moderate. **Risk:** moderate — BA convergence tuning; sequence after P3's first
  experiment confirms headroom.

### P6. Minimum-baseline pair gate for mv confidence
- **What:** add a triangulation-angle/baseline gate to the mv pair loop so co-located views
  stop counting as independent evidence (the degenerate case is currently a passing test).
- **Why:** cheap, standard (COLMAP's `min_angle=1.5°` is the same idea), and makes
  `min_views` count real evidence. Independent of P1-P5.
- **Cost:** small. **Risk:** low — but mv ships off, so measure with the P0b harness before
  deciding it matters; a gate on a disabled filter earns nothing by itself.

## Sequencing and evaluation

- **P0a → P0b first** (handoff sequencing stands): they fix the baselines every other item
  is measured against, and cost hours-to-a-day.
- **P1 → P2 → P3** is the geometric-consistency core; P3's first experiment is the go/no-go
  gate for P4 and P5.
- **P6** is independent and cheap but only meaningful if P0b re-opens the case for enabling mv.
- Measurement discipline from the handoff applies throughout: report p90/p95/out10 (never
  median alone), compare at matched retention, and mutate every new threshold once to prove
  it is not inert.

## Implementation principles (repo standard)

- Reuse: matchers, retrieval, zarr feature cache, `build_pycolmap_reconstruction`,
  `filter_all_points3D` — no new matching or triangulation math written in-repo; pycolmap
  is the triangulation engine.
- Delete/close as we go: P3 removes the "empty Track()" caveat; P5 can retire the
  VGGT-locked track path if it wins.
- No estimator code (Workstream 3) until P4's headroom number justifies it.
