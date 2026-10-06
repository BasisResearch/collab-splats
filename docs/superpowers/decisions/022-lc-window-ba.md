# 022 — Bundle adjustment inside each loop-closure window

Date: 2026-10-05 · Status: accepted · Branch: `feat/rgbd-ba-cf`

## Context

- since the 2026-08-19 BA wiring, config validation refused BA with LC: LC submaps do not carry the per-frame model tensors the whole-scene `refine` stage needs; no decision recorded the refusal
- perf-1k needs ~1000-frame scenes, which only run windowed (LC), and BA is what makes the 294-frame GH010229 baseline good
- the baseline came from a scratch hook (`lc_inner_light.py`) that solved BA inside each window, on the worker thread, before the submap entered the pose graph

## Decision

- BA on + LC on = BA inside each LC window; no `refine` stage; no new config keys
- the first refined window sets the focal; later windows hold it (`refine_focal: false` for their solve)
- the solve runs in `LoopClosure._forward_window`, after the forward, on the existing worker thread; loop-verify pairs are never refined
- a window whose solve raises `ValueError` keeps its feedforward poses; per-window records land in the zarr attrs as `window_ba`

## Consequences

- an explicit `refine` stage under LC raises; whole-scene BA after LC is out of scope
- loop carriers come from 2-frame forwards with the feedforward focal (known limit, measured at the chess gate)
- spec: `docs/superpowers/specs/2026-10-05-lc-window-ba-design.md`
