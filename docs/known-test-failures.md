# Known Test Failures

## 2026-07-22 — LC resident RAM scales with frame count (KNOWN LIMIT, follow-up owed)

P7.3 lean-RAM check (chess seq-01, **1000 frames**, vggt_omega, submap_size 16, lc): completed
EXIT=0, ATE 0.0200m, 78 loops, `points3D.bin` 25 MB (the `_assemble_result` `max_points` cap holds
at scale). **But peak cgroup = 50.0 GB / 50 — dead on the cap** (already 49.9 GB mid-loop at 94%,
so it's resident accumulation, not the final-assembly spike).

Root: `GraphMap` holds every submap's dense per-pixel `points`/`colors`/`conf` resident in a RAM
dict for correction-at-read (VGGT-SLAM fat-submap design). ~63 submaps × ~16 frames of dense cloud
≈ 50 GB at 1000 frames. P5.1b freed per-submap `raw_outputs` only; the dense cloud + `frames` stay.
This is distinct from the FIXED SIGKILL (5.1 GB colmap Point3D blow-up, see below) — that was the
assembly output; this is the in-loop working set. 1000 frames survives by luck; a larger scene or a
side-shell OOMs.

**Follow-up (not blocking):** subsample each submap's dense cloud at store time (`set_dense_points`,
reuse `subsample_points`) → bounds resident to `n_submaps × capped_points`. Touches viewer/
correction fidelity → needs its own validation (re-run P7.3, target < ~35 GB at 1k frames, confirm
ATE + viewer unaffected). Until then: keep long scenes ≤ ~1000 keyframes on the 50 GB box.

## 2026-07-21 — LC `ba` condition over windowed submaps: known gap (SCOPED OUT)

The VGGT-SLAM loop-closure refactor assembles its output in `LoopClosure._assemble_result`
(GraphMap dense cloud, correction-at-read) instead of `base._postprocess`. `_assemble_result`
populates `points/colors/extrinsics/intrinsics` but **not** the optional `FeedforwardResult.images`
tensor. The old `_postprocess` set `images=images`; BA's track extractor (`_extract_tracks_vggsfm`)
does `images.to(device)`, so running the eval `ba` condition on the windowed path now raises
`AttributeError: 'NoneType' object has no attribute 'to'`.

This was masked until 2026-07-21: baseline used to OOM (5.1 GB `points3D.bin` → pycolmap
Point3D blow-up → 50 GB cgroup SIGKILL) before the `ba` condition ran. The `max_points` cap fix
(`_assemble_result` → `subsample_points`, restoring the 500k budget the skipped `postprocess`
used to apply) fixed the OOM (baseline 0.0267m ATE, `points3D.bin` 25 MB, peak 22.9 GB), which
surfaced the pre-existing `images` gap underneath.

**Scoped out, not fixed:** BA-over-windowed is a parked feature (BA is currently
VGGT-X/MapAnything-specific; see `project_ba_future_generalization`). P7 parity validation runs
`--conditions baseline lc` (the actual LC-parity target). To revive `ba` over windowed submaps,
`_assemble_result` must populate `images` by concatenating per-submap `frames` with the same
first-occurrence overlap dedup it already uses for `intrinsics`.

## 2026-07-20 — frame-store refactor: notebook follow-ups (RESOLVED)

`docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` was broken by the
frame-store refactor (`frames.zarr` replacing the `output_path/images/` JPG dir): it imported
`extract_frames`/`load_frames` from `collab_splats.preproc` (both removed from the public API)
and called `plot_quality_examples(VIDEO_PATH, frame_scores)` against a stale signature. Fixed:
rewrote the broken cells against the current API (`FrameStore, sample_frames, score_frames,
get_video_info, extract_frame, compute_blur_score, check_frame_quality`); the per-frame quality
example cells (before any `frames.zarr` exists in the notebook's narrative) use a lightweight
duck-typed frame lookup wrapping `extract_frame` instead of decoding the full video into a store.
`docs/source/tutorials/07_localization/localization.ipynb` was fixed in place earlier (no
pipeline run needed): `result.image_paths` reference a per-run temp export dir that no longer
exists by notebook time, so the query-image load and `plot_correspondences(...)` now go through
`FrameStore.open(OUTPUT_DIR / "frames.zarr")` / `frames_zarr=` instead of `cv2.imread`.

The two follow-ups previously tracked in `docs/source/api/preproc.rst` ("Known limitations")
are also resolved: `transforms.json` no longer emits a dangling `../images/` `file_path` (drops
the key entirely, adds `frame_idx`), and the dashboard's localization reference thumbnails
(`_build_result_figures` in `collab_splats/dashboard/localize.py`) now read reconstruction-
sourced frames from `frames.zarr` via `plot_correspondences(..., frames_zarr=...)`. The
`out_dir/frames/` JPG dir itself remains — `creator.setup_inference` and semantic feature
extraction are path-locked consumers that still need a real on-disk image directory — see
`docs/source/api/preproc.rst` for the current state.

## 2026-07-20 — evals + loop-closure aggressive cleanup (GREEN)

Full suite (`/opt/venv/reconstruction/bin/python -m pytest tests/ -q`): **1121 passed, 2 skipped, 3 xfailed**. Migration gate `pytest tests/test_cu121_migration.py` → **24/24 PASS**.

Pass count dropped from ~1200 (pre-cleanup) to 1121 **entirely by retiring scripts + their covering tests** (allowed: deleting a script deletes its test) — no behavior regressions. Retired test files: `test_cross_model_runner`, `test_run_lc_parity`, `test_build_benchmark_table`, `test_build_parity_table`, `test_lc_loop_pr`, `test_visualize_lc_correction`, `test_loop_ablation`, `test_loop_ablation_json`, `test_reconstruction_quality`, `test_lc_parity_common`, `test_lc_decisions`, `test_ate_utils_tum`, `test_run_vggt_slam_lc` (folded into `test_run_vggt_slam`). New tests added: `test_eval_config`, extended `test_datasets`.

Known flaky (pre-existing, unrelated): `tests/dashboard/test_viz_utils.py::test_view_transform_scales_to_target_radius` — nondeterministic, fails ~1/3 of runs on identical code; surfaces intermittently, not a regression.

`evals/scripts/run_vggt_slam.py` no longer computes its own ATE — it emits the SLAM TUM (+ `selected_frames.txt` when `--max_loops >0`); `eval.py`/`eval_compare` scores it against GT downstream.

---

# Known Test Failures — 2026-06-01 (GREEN)

Run (conda env `reconstruction`, py3.11, **numpy 2.1.3**):
```
/opt/conda/envs/reconstruction/bin/python -u -m pytest tests/ -m 'not slow' \
    --ignore=tests/test_cu121_migration.py --continue-on-collection-errors -q -rfE \
    -o faulthandler_timeout=120 -p no:cacheprovider
```
Result: **964 passed, 0 failed, 2 skipped, 4 deselected, 3 xfailed**, 0 collection errors, 8 min.
Migration hard gate: `pytest tests/test_cu121_migration.py` → **24/24 PASS**.
Baseline saved: `evals/results/baseline-conda-tests-0601.txt` (gitignored).

The suite is green. This supersedes the 59-failure baseline earlier on 2026-06-01 and the
broken-env handoff snapshot (`73f/346p/16err`). Everything below is the record of how it got here.

## What was fixed (59 failed → 0)

Three structural unblocks (deps-by-default, `tests/nerfstudio_methods` rename, the hang fix) plus
~20 commits of test-API realignment. Highlights:

- **The "11h at 7%" hang** — `tests/dashboard/test_visualize.py` mesh-worker tests mocked
  `multiprocessing.Process` but never stubbed `is_alive()` → `while proc.is_alive()` spun forever.
  Unmasked once `panel` was installed (before, those tests failed collection). Fixed with
  `proc_mock.is_alive.return_value = False`.
- **Group C (webapp async)** — app is FastAPI/ASGI; added `pytest-asyncio` + `httpx` to
  `pyproject [dev]`. (Not tornado — the pytest "pytest-tornasync" hint was a red herring.)
- **Two real product bugs found + handled** (the "no regressions" assumption was wrong):
  - `docs/examples/reconstruct.py` `_REPO_ROOT` was `.parent.parent` after the move from
    `scripts/`, pointing `--config_dir` at the nonexistent `docs/configs`. **Fixed** to
    `.parent.parent.parent` (commit `1d331b6`).
  - BA `test_optimize_*` (CUDA+bae only) hit a real bae/pypose bug: bae's `LM.step` calls pypose
    `RobustModel.forward(input, target)` with only `input` → `missing 'target'`. **Deferred**:
    xfail(strict=False) with a follow-up reason (`a36d1d7`). Tests are correct; production needs a
    dedicated BA fix.
- **Retired (deprecated):** `tests/examples/test_run_c0043_pipeline.py` (script gone),
  `test_default_verifier_raises` + `test_base_verify_raises_with_tuple_signature` (base verifier is
  now concrete). The BA-dedup test was **rewritten** (not retired) against the relocated
  `LoopClosure.run() _dedup_rows` path.
- **Stale markers cleared:** 2 xfeat `@skip` + 1 lazy-torch `@xfail` (now pass).

## Residue (intentional, not failures)

- **3 xfailed** — BA `test_optimize_reduces_reproj_error`, `_captures_loss_history_when_flag_set`,
  `_no_loss_history_by_default`: real bae/pypose `RobustModel.forward(target)` bug, deferred. Only
  run when CUDA+bae present.
- **2 skipped** — environment-gated (e.g. ffmpeg / evo_ape CLI not installed).
- **1 flagged docstring** (no test impact) — `loop_closure/eval.py:18` `_classify_edges` still claims
  "Loop edges use Robust(Huber)"; post-`011c56f` loop and sequential edges use identical Gaussian
  noise so the function can no longer separate them. Instrumentation degraded by design; fixing
  needs a production change to track edge provenance. Owner's call.
- **1 nondeterministic flake** — `tests/dashboard/test_viz_utils.py::test_view_transform_scales_to_target_radius`
  fails ~1/3 of runs even in isolation on identical code (verified 2026-07-15: FAIL/PASS/PASS across
  three isolated runs at the same commit). Unseeded randomness or float-tolerance issue in the
  view-transform scaling math — pre-dates the localization-dashboard work; needs a seed or looser
  tolerance. Retry on failure until fixed.

## numpy version note (env, not a code bug)

The migration gate enforces `numpy>=2` (also required by rerun-sdk). The env had drifted to
`numpy 1.26.4` — a stray with-deps install of `vggt-omega` (which pins `numpy<2`). The documented
`setup/feedforward.sh:57` already installs vggt-omega with `--no-deps` to avoid exactly this, so a
clean build per the setup script is fine; the drift came from outside it. Restored to `numpy 2.1.3`
(vggt-omega imports and runs fine on 2.x — its `<2` pin is conservative). The uv migration inherits
the `--no-deps` install, so this won't recur.

## For the uv migration

Only Group C is env-sensitive and it lives in `pyproject [dev]` (uv installs it). Everything else is
env-independent test/code state. **Target:** the uv env passes iff it reproduces this baseline —
964 passed, the 3 BA xfails, the 2 env skips — and adds no new failures. Ensure the uv build keeps
`numpy>=2` (the `--no-deps` vggt-omega install already does).
