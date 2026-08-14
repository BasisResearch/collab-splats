# Geometric verification of feedforward reconstructions — design

**Date:** 2026-08-14
**Decision record:** `2026-08-14-geometric-consistency-proposal.md` (4 review rounds; this
spec is the distillation and supersedes it for implementation).
**Status:** design for review.

## Goal

Produce **more accurate points and poses** from feedforward inference by adding the
industry-standard external check: known-pose triangulation and epipolar verification
(COLMAP's mechanism, via pycolmap). Verification and refinement are the same machinery at
different trust levels:

- **v1 (this spec):** a geometrically *verified sparse cloud* — every point triangulated
  from model-agnostic feature tracks and surviving COLMAP's reprojection/angle gates
  against the model's poses — plus per-pair and per-frame **pose verification statistics**.
  The verified cloud is itself the more-accurate pointcloud; the pose stats are the
  measured evidence that decides pose *correction*.
- **Follow-ons (gated on v1's numbers, out of scope here):** pose correction via
  backbone-agnostic BA over the same tracks; dense-depth correction from the
  triangulated-vs-model audit.

Everything is a **layer on top of any backbone** — same pattern as multiview confidence:
operates on the result contract (frames, poses, intrinsics), zero per-creator code, one
config boolean.

## Non-goals (v1)

- No refinement mode (point-only BA seeded from model `world_points`) — one flag away
  later, unjustified before the independent mode is measured.
- No per-frame pose re-estimation (Tier 3) — Tier 2 residuals likely suffice for the BA
  go/no-go; add only if that signal is too coarse.
- No pipeline gating on verification stats — v1 reports, never blocks; gating needs
  calibrated thresholds v1 exists to produce.
- No change to the mesh path, the dense zarr outputs, or `sparse_pc.ply`.
- No mv-confidence changes (baseline gate parked; conf-threshold sweep deferred).

## Design

### Component 1 — keypoint indices on `MatchResult`

`localization/extractors.py`: add `idx_q`, `idx_db` (int64 arrays, parallel to the pixel
arrays) to `MatchResult`; populate in Disk, XFeat, LoMa, LomaG — each already computes the
indices and discards them. Pixel fields stay (the localizer consumes them). Index pairs are
exactly COLMAP's match format, so this converges with the standard rather than adding a
concept. **XFeatStar excluded:** its per-pair subpixel refinement gives the same keypoint
different coordinates in different pairs, which COLMAP's one-keypoint-table-per-image model
cannot represent; it returns `idx_q=idx_db=None` and the verifier rejects it with a clear
error.

### Component 2 — `geometry/verification.py`

One public entry point (name indicative): `verify_reconstruction(result, frames, config) ->
VerificationResult`. Internally, four steps, all reusing existing machinery:

1. **Pair selection.** Sequential adjacency window (default `window=10`, config-exposed)
   plus loop pairs from the existing retrieval extractor (`DinoSalad`, as loop closure
   uses it). Pairs are O(N·window), never O(N²).
2. **Features and matches.** Extract keypoints once per frame with the configured
   `BaseLocalExtractor` (reusing the localizer's zarr feature-cache read/write), match
   selected pairs, keep index pairs.
3. **COLMAP database.** Write cameras/images/keypoints/matches into `pycolmap.Database`;
   run `pycolmap.verify_matches` to fill `two_view_geometries`. **Tier 1 output:** per-pair
   epipolar inlier counts and the estimated relative pose vs the model's relative pose
   (rotation error, translation-direction error) — pose verification with no triangulation.
   Follow `pointcloud/sfm.py`'s existing pycolmap DB/path conventions.
4. **Triangulation.** Build the `Reconstruction` from the existing
   `build_pycolmap_reconstruction` cameras/poses with points cleared; run
   `pycolmap.triangulate_points` (chains tracks internally — no track code of ours) and
   `filter_all_points3D(4.0, 1.5)` at COLMAP defaults (not config-exposed). **Tier 2
   output:** the verified points (with real tracks — fills the documented empty-`Track()`
   hole), per-frame track survival, and per-frame mean reprojection error.

Coordinate contract: localization keypoints are original-resolution pixels by contract and
the COLMAP export is original-resolution K on purpose — consistent by construction (unlike
the model-resolution zarr; the 2026-08-11 regression class). Camera model PINHOLE. One
guard: keypoint bounds vs the DB camera width/height.

### Component 3 — pipeline wiring and storage

- **Placement:** runs after `build_colmap`, i.e. after LC and BA have finalized poses
  (triangulating earlier is invalidated when LC rewrites poses). Registered as a leaf
  stage (`LEAF_STAGES`) so a processed scene can be re-verified from
  `environments-processed` without rebuilding.
- **Config:** one boolean, `pointcloud.geometric_verification` (default `false` until the
  first experiment reports); extractor and window under it. COLMAP thresholds stay at
  library defaults.
- **Storage principle — raw model outputs are immutable; verified geometry is a derived
  layer.** `feedforward.zarr` is never rewritten. Outputs land in the existing
  `<backend>/colmap/` artifact:
  - triangulated points as the reconstruction's `points3D` **with tracks** (poses/cameras
    unchanged — a COLMAP model that downstream COLMAP tooling and BA can actually consume);
  - `verification.json` — Tier 1 per-pair stats, Tier 2 per-frame stats, point count,
    track-length distribution (median/p90/p99 style, never median alone);
  - `database.db` — local build artifact, rebuildable from the zarr cache + poses, **not
    pushed to GCS**.
- `sparse_pc.ply` (dense subsample) is unchanged in v1. Whether the verified cloud
  replaces it for splat seeding is decided by the first experiment's **yield** numbers —
  small-baseline video may starve triangulation (`min_angle=1.5°`,
  `ignore_two_view_tracks=True`), and an accurate-but-sparse cloud may seed splats worse.
  The mesh consumes dense depth and is untouched either way.

## Validation

- **Negative control is mandatory.** A verifier validated only on good poses is
  unvalidated ("a threshold nobody mutated is presumed inert"). Test: perturb a subset of
  poses (e.g. +2° rotation) and assert Tier 1 relative-pose errors and Tier 2 track
  survival flag exactly the perturbed frames.
- **Unit tests** (flat functions, `tests/geometry/test_verification.py`): `MatchResult`
  index round-trip per extractor; DB write/read round-trip on synthetic matches;
  triangulation on a small synthetic two/three-view scene with known geometry; the
  XFeatStar rejection; the keypoint-bounds guard.
- **First experiment (falsifiable), chess/seq-01:**
  - *Accuracy:* triangulated depths vs model depths vs GT at the same pixels —
    median/p90/p99 and out10, plus the reference-free triangulated-vs-model agreement
    (the GT column can be a noise floor).
  - *Yield:* point count, track-length distribution, per-frame survival — the
    replace-vs-ship-both decision input.
  - *Negative control:* as above, on the real scene.
  - Run for at least XFeat (fast) and LoMa (heavy) to pick the default extractor.
  - **Falsification:** if triangulated points are not measurably more accurate than model
    points where the two disagree, the verified cloud is an audit artifact only, and the
    BA follow-on loses its premise.

## Sequencing

1. Component 1 (small, independently landable).
2. Component 2 + 3 with tests and negative control.
3. First experiment; record results in a measured report; then decide: `sparse_pc`
   replacement, BA follow-on, dense-depth audit.
