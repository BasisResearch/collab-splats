# 025 — Bundle adjustment strategy under loop closure

Date: 2026-10-08 · Status: accepted · Amends: 022

## Context

- 022 ran BA only inside each LC window and left whole-scene BA after LC out of scope
- a single-pass scene gets one solve over every frame; LC scenes could not, so cross-window drift stayed unrefined
- the LC store lacked the model-grid images (photometric term) and per-point source pixels (`reproject`) that `refine` reads

## Decision

- `pointcloud.bundle_adjustment.strategy`: `window` | `global` | `window+global`, read only with LC on
- default `global`: plain LC windows, then the `refine` stage over every frame
- `window` keeps 022's behavior; `window+global` runs both
- the LC result carries `images` (from each frame's owning submap colors) and `pixel_indices`

## Consequences

- LC + BA configs that relied on 022's per-window solve now get the global solve unless they set `strategy: window`
- exhaustive pairing grows as N²: 600 frames = 179,700 pairs, about 4x the 300-frame solve
