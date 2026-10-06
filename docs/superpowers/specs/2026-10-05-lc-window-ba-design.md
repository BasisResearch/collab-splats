# lc-window-ba — bundle adjustment inside each loop-closure window

Date: 2026-10-05 · Status: draft for review · Branch `feat/rgbd-ba-cf` · Parent effort: perf-1k (`2026-10-03-perf-1k-design.md`)

## Goal

Make "windowed LC + per-window BA" a package feature, so the perf-1k gates run on package code only.

- today the package refuses BA with LC (`reconstructor.py`, "Refuse BA with LC")
- the 294-frame GH010229 baseline came from a scratch monkeypatch: `lc_inner_light.py` / `gh_lcba.py` (session d91a918d scratchpad)
- the package path must reproduce that hook's output (see Parity)

## Behavior contract

| Config | Result |
|---|---|
| BA on, LC off | unchanged: whole-scene `refine` stage |
| BA off, LC on | unchanged: LC, no BA, bit-identical to today |
| BA on, LC on | BA inside each LC window during the pointcloud stage; no `refine` stage |
| BA on, method sfm | unchanged: refused |

- one config block: `pointcloud.bundle_adjustment` keys tune the per-window solve exactly as they tune the whole-scene one
- no new config keys
- `configs/README.md` and `configs/base.yaml` say: with loop closure on, `bundle_adjustment.enabled` means per-window BA

Example (the 294 baseline settings):

```yaml
pointcloud:
  loop_closure: {submap_size: 100}
  bundle_adjustment:
    enabled: true
    query_frame_num: 4
    max_query_pts: 2048
    fine_tracking: false
    dtype: float64
```

## Design

### Reconstructor (`collab_splats/reconstructor.py`)

- delete the BA-with-LC refusal in config validation; both sfm refusals stay
- default stage set: `refine` enabled only when BA is on and LC is off
- `refine()` raises `ValueError` when LC is on (an explicit `--stages refine` under LC would re-solve an LC store BA already ran in)
- `pointcloud()` with both on: build `BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir / "window_ba", **terms)`, the same way `refine()` builds its config, and pass it as `LoopClosure(base=creator, config=lc_config, ba=ba_cfg)`
- `pointcloud()` adds `to_json_safe(creator.window_ba)` to the zarr provenance attrs under key `window_ba` when it is set
  - `save_zarr` writes `extra_attrs` raw, and zarr attrs must be strict JSON (no numpy scalars, no NaN); `utils.io.to_json_safe` already does that conversion

### LoopClosure (`collab_splats/geometry/loop_closure/wrapper.py`)

- `__init__(base, config=None, ba: BundleAdjustmentConfig | None = None)`; `ba=None` keeps today's code path untouched
- `_forward_window(window, start)`: the forward pass, then, when `ba` is set, the window solve below inline in the same method; no new function or method
  - runs on the existing single worker thread, inside the same `hold_matmul_precision()` + `no_grad` block
  - BA enters `enable_grad` itself (`bundle_adjustment.py`, refine)
  - worker order is strict (forward wi, BA wi, forward wi+1), so window 0's focal is known before window 1's solve
  - both `pool.submit` sites pass the window's `start` from `bounds`: `pool.submit(self._forward_window, windows[wi + 1], bounds[wi + 1][0])`
- loop-verify pair forwards go through `_verify_loop_candidate`, never `_forward_window`, so they are never refined (the hook's `k < 10` test is not needed)
- both no-LC fallbacks forward through `_forward_window(views, 0)`, so their one whole-scene forward is refined as window 0
  - fewer than `submap_size` frames: `run_inference` sets `self.base.raw_outputs = self._forward_window(self.base.views, 0)` then `pytorch_gc()`, replacing `self.base.run_inference()` (same forward + gc; only its timing log line is lost)
  - DINO-SALAD failing to load: `_run_lc_loop` sets `self.base.raw_outputs = self._forward_window(views, 0)`, replacing the direct `self.base._forward(...)`
  - BA off stays bit-identical: `hold_matmul_precision` is only a lock, and every `_forward` already runs under `no_grad`
  - without this BA would be silently skipped: the `refine` stage is not planned under LC
  - these run on the main thread before any worker exists, so the precision lock is uncontended
  - the hook refined these too: it patched `_forward`, so any forward of ≥ 10 frames was solved

### Window solve (inside `_forward_window`)

Steps after the forward, each mirroring the scratch hook; `raw` is the forward's output, returned at the end:

1. Frames on the model grid in [0, 1]: `window.float().cpu().numpy()` for a tensor window; `raw["images"]` for a MapAnything view list
2. Focal hold: when `self._ba_focal` is set, copy `raw["intrinsics"]` and write the held focal into fx and fy; the solve config is `dataclasses.replace(self.ba, refine_focal=False)`; otherwise `self.ba`
3. Inputs: `depth = raw["depth"]`, trailing channel squeezed when 4-D as `_postprocess` does (VGGT gives (k, H, W, 1), MapAnything (k, H, W)); world points from `unproject_frames(depth, raw["extrinsic"], K_in)`; poses from `extrinsics_to_homogeneous(raw["extrinsic"])`; confidence `torch.from_numpy(raw["depth_conf"])`; paths `self.base.image_paths[start : start + k]`
4. Solve: `BundleAdjustment(cfg_w).refine(frames, conf, world_points, poses, K_in, paths, depth=depth)`
5. Track cache: `cfg_w = dataclasses.replace(cfg_w, tracks_cache_dir=self.ba.tracks_cache_dir / f"w{start:06d}")` (one `tracks.zarr` per directory); `None` stays `None`; the solve in step 4 uses this `cfg_w`
6. Back to frame 0's camera: `E_local = E @ invert_poses(E[:1])[0]`; write `E_local[:, :3, :]` and the refined K into `raw["extrinsic"]` / `raw["intrinsics"]` as float32
7. Focal record: the first successful window sets `self._ba_focal = float(K[0, 0, 0])` only when `self.ba.refine_focal` is true; with `refine_focal: false` every window keeps the feedforward focal
   - window 0 failing: the next successful window sets it
   - `shared_camera: false`: frame 0's focal is held, as the hook did
8. Failure: `refine` raising `ValueError` logs a warning, leaves `raw` untouched (feedforward poses) and records `ok: false`
   - as the hook: a failed later window keeps its feedforward focal, not the held one; kept for parity, visible in `window_ba`
   - other exceptions (CUDA OOM, RuntimeError) propagate, as the hook
9. No extra `empty_cache`: `refine` already frees the cache per solve and per scale (475decff); the hook's extra call after each solve is dropped
10. Record: append `{start, n_frames, ok, seconds, focal, alignment_scale, loss_final, gpu_max_mib}` to `self.window_ba`; `run_inference` resets `self.window_ba = []` and `self._ba_focal = None` (not `_run_lc_loop`: the too-few-frames fallback never enters it)
   - `alignment_scale`: `ba.alignment_scale`; `loss_final`: `ba.loss_history[-1][-1]` (a list of per-solve lists), `None` when empty
   - `gpu_max_mib`: `torch.cuda.max_memory_allocated() >> 20` after `reset_peak_memory_stats()` at window start (the hook never reset, so its figure is a running max); `None` without CUDA

- scale is untouched: depth fixes it (`fix_scale` in `refine`), so window depth and the refined poses agree
- `run_predictions` then builds the submap and unprojects the dense points from the refined `raw`, as it does today

### Reused, not rewritten

| Need | Existing function |
|---|---|
| window solve | `BundleAdjustment.refine` |
| BA config + validation | `BundleAdjustmentConfig` (built as in `Reconstructor.refine`) |
| depth → world points | `geometry.projection.unproject_frames` (replaces the hook's vggt `unproject_depth_map_to_point_map`) |
| 3x4 → 4x4 | `geometry.transforms.extrinsics_to_homogeneous` |
| re-anchor to frame 0 | `geometry.transforms.invert_poses` (replaces the hook's `np.linalg.inv`) |
| track caching | `BundleAdjustment`'s own `tracks_cache_dir` store |
| provenance | `PointcloudResult.save_zarr(extra_attrs=...)` |

No new module, no new function, no new method: the solve lives in `_forward_window`, the reconstructor change lives in `pointcloud()` / `refine()` / validation. New state is two attributes only: `window_ba` and `_ba_focal`.

## GPU memory and solver interaction

- GPU: forward and BA never overlap (one worker, strict order); the main thread runs only CPU work (`finish`, gtsam solve) while the worker is busy; fp32 GPU work elsewhere waits on the precision lock
- no self-deadlock: `_matmul_precision_lock` is an `RLock`, so `unproject_frames`' `full_fp32_matmul` inside the worker's `hold_matmul_precision` re-enters it
- loop verification runs on the main thread inside `run_predictions`, before the next window is submitted, so it never overlaps a window solve
- measured: the hook peaked at 17.8 GB per 101-frame window (model included), flat across windows (`gh_lcba.log`); `gpu_max_mib` is recorded per window to confirm
- BA → pose graph is one-way: BA finishes before the submap enters the graph; the graph never feeds back
- known limit: loop carriers come from fresh 2-frame forwards with the feedforward focal, while their K is stamped from the refined window; the hook has the same behavior, so parity holds; measured at the chess gate (loop count, GT co-visibility precision); a fix is out of scope

## Parity — replicate the scratch hook

Both sides run on this branch, same code otherwise, outputs under the scratchpad (never `/workspace/outputs/ocr_viewer/*`):

1. chess seq-01, 200 frames, `submap_size: 50`: `lc_inner_light.py` cell `…ibfz` (float64) vs the package cell with the BA block above
2. GH010229, the 294-frame config of `gh_lcba.py`, pointcloud stage only: `gh_lcba.py` with `stages=["pointcloud"]` (a scratch copy; the original also meshes) vs the package
   - the old `/workspace/outputs/ocr_viewer/GH010229` store predates seeding, early stop and item 8, so it is not a parity reference

Pass:

- per-window focal equal within 5e-3 px; same windows ok / failed
- final zarr extrinsics max abs diff ≤ 1e-4; chess ATE vs GT equal within 0.1 mm
- the only deliberate difference is GPU fp32 `unproject_frames` vs the hook's vggt numpy unprojection; a miss is first re-run with the hook switched to `unproject_frames` to isolate it

Pass, as revised 2026-10-05 after the first runs (user chose to loosen, not seed):

- run-to-run noise sets the bar: VGGSfM shuffles query keypoints with an unseeded `torch.randperm` (`vggt/dependency/track_predict.py:173`) before the `max_query_pts` cut, so every run tracks different points, hook or package
- focal within 0.25 px; same windows ok / failed
- chess ATE within 0.1 mm; extrinsics diff package vs hook no larger than hook vs hook

Measured (scratchpad `parity/`):

| Run | Focal diff vs hook | Extrinsics diff | ATE |
|---|---|---|---|
| chess 200, pkg | 0.17 px (0.12 vs `unproject_frames` hook) | max 9.8e-5 | 13.285 vs 13.268 mm |
| GH010229 294, pkg | 0.011 px | cam-center max 0.69, median 0.017 | — |
| GH010229 294, hook rerun | 0.135 px | cam-center max 0.48, median 0.018 | — |

- every window ok on both sides; 0 loops on both GH runs
- GH worst frames 163-171 (window 1) move by the same amount between two hook runs

After parity, the perf-1k gates run on package config only: item 11 (float32 vs float64 BA, chess 1000), chess GT 1000, GH010229 1000 vs the 294 baseline.

## Tests

Flat functions, stub creator, `BundleAdjustment.refine` monkeypatched, no model weights.

- `tests/geometry/loop_closure/test_window_ba.py`
  - BA called once per window, never for loop-verify pairs
  - both fallbacks (too few frames, DINO-SALAD load failure) refine the whole-scene forward once
  - a MapAnything-shaped raw ((k, H, W) depth, view-dict window) reaches `refine` with `raw["images"]`
  - window 1 receives window 0's focal and `refine_focal=False`
  - `refine_focal: false`: every window gets the feedforward focal
  - `ValueError` keeps the feedforward poses and records `ok: False`
  - refined poses satisfy `poses[0] ≈ I`
  - per-window cache dirs `w000000`, `w000050`, …
  - `ba=None` output equals today's on the stub
- `tests/reconstructor/`
  - validation accepts BA + LC
  - default stage set drops `refine` under LC
  - `refine()` raises under LC
  - zarr attrs carry `window_ba`

## Docs

- decision `docs/superpowers/decisions/022-lc-window-ba.md`: why the refusal goes and BA moves inside the window (no existing decision records the refusal; it dates from the 2026-08-19 BA wiring)
- `configs/README.md`, `configs/base.yaml` comment, `docs/pointcloud.md`: the BA-with-LC meaning

## Out of scope

- whole-scene refine after LC
- xfeat / LoMa tracks (`track_source`, vismatch agent)
- a focal-policy key
- refining loop-verify pairs
