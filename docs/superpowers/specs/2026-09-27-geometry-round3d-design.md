# Geometry round 3d — third user review

Date: 2026-09-27 · branch `clean/geometry-round3d` (forked from `clean/geometry-round3` `e0d48ab8`)

Review points on round 3c. Decisions below are the user's.

## Decisions

| # | Point | Decision |
|---|---|---|
| BA 1 | `_load_or_extract_tracks`: should cache loading be implicit? | yes: public `BundleAdjustment.extract_tracks`, cached when `tracks_cache_dir` + `image_paths` are set |
| BA 2 | `_extract_tracks` name | public `extract_tracks_vggsfm` |
| graph 1 | `_cam_local_points` needed? | the transform was inline in several modules: new `transforms.transform_points(points, T)` (`R @ p + t`); helper deleted |
| graph 2 | helpers top or bottom? | bottom: public functions, class, then one Helpers section |
| graph 3 | `_conf_fallback_mask` duplicates `pointcloud.utils.confidence_mask`? | no: tiered fallback over paired arrays vs one percentile cutoff; kept, docstring cut to 2 bullets |
| graph 4 | `_lc_anchor_scale` name + docstring | `calculate_pairwise_frame_scale`, public, short docstring; also used by the sequential edge |
| submap 1 | `assert_world_to_cam` useful? | it only tests `poses[0] ≈ I`, and `inv(I) = I`, so it can't detect cam-to-world; inlined into the LC wrapper (sole caller) as a frame-0 normalization check |
| wrapper | useful in its current form? | deferred until pointcloud-release lands |

## Behavior changes

- sequential edge: both overlap sides move into their own camera (was prior side only); per-pair `inv(K_prior) @ K_curr`
  - identical at `submap_overlap=1` with frame 0 at identity (the golden incremental test holds at 1e-6)
  - closes round 3b's open item: overlap > 1 left current-side frames after the first un-moved
- sequential edge: a non-finite or <= 0 scale falls back to 1.0 with a warning (loop anchors already did)
- frame-0 check message: "Window poses not normalized to frame 0"

## Follow-ups (not done here)

- `transform_points` sites left for pointcloud-release: `pointcloud/feedforward/base.py` (4), `pointcloud/depth_align.py:104`
- LC wrapper form review, after pointcloud-release lands
