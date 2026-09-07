# Known Test Failures

## 2026-09-06 — RESOLVED: `tests/wrapper/test_splats_stage.py`: 3 `KeyError: 'splat_max_depth_frac'` failures

Reproduced on `clean/splats` at `6611ef7c`; **pre-existing, not caused by the splats cleanup.** At
that point the cleanup had not touched `reconstructor.py` at all, and its only edits to the other two
files involved were inside comments: `97c2fae0` renamed `CameraOptModule` to `CameraOpt` in a
`configs/base.yaml` comment, and `b4e67d22` renamed `prepare_training_target` to `prepare_target` in a
comment in this test file. Neither can change behaviour, so neither can raise a `KeyError`.

```
FAILED tests/wrapper/test_splats_stage.py::test_mesh_source_splats_fuses_from_splats_zarr
FAILED tests/wrapper/test_splats_stage.py::test_mesh_sfm_aligned_zarr_fuses
FAILED tests/wrapper/test_splats_stage.py::test_mesh_stage_forwards_splat_depth_from_the_config
3 failed, 13 passed
```

That output is kept verbatim; the first test is now called
`test_mesh_source_splats_fuses_from_the_checkpoint`, renamed by the same task that closed this entry
when `splats.zarr` was retired in favour of `splats/ckpt.pt`.

All three raised `KeyError: 'splat_max_depth_frac'` in `Reconstructor.mesh()`, on the `mesh_cfg`
subscripts that build the `_run_tsdf_mesh` call.

**Cause.** `c1b50e02` ("fix(mesh): ship the splat depth cuts off after measuring them") added
`splat_max_depth_frac` and `splat_max_depth_grad` to `configs/base.yaml` and read both by direct
subscript in `mesh()`. `configs/base.yaml` has carried both keys ever since and was never the
problem. The stale thing was the hand-rolled `recon.config` dict in this test file's own
`_stub_reconstructor`, whose `"mesh"` block stopped at `splat_depth`, so every test in the file that
reached the `_run_tsdf_mesh` call site died on the first missing key.

**Resolved 2026-09-06** by the splats cleanup's Task 18, which owns this test file: the two keys were
added to `_stub_reconstructor`'s `"mesh"` block with their shipping defaults.

```python
            "splat_max_depth_frac": None,
            "splat_max_depth_grad": None,
```

The wider point stands: a hand-rolled config fixture silently rots every time `base.yaml` gains a key
that `reconstructor.py` reads by subscript. Building the fixture from `base.yaml` with explicit
overrides would make this class of failure impossible, and is still not done.

## 2026-09-06 — `pgsr_multiview` + `pose_opt: true` produces NaN pose deltas (OPEN, pre-existing bug)

Not a test failure — no test covers this combination — but a real bug, recorded so the next person to
enable it does not rediscover it from scratch.

**Measured against the PRE-refactor tree (`edd1fafb`), so this is not caused by the splats cleanup.**
With `losses.pgsr_multiview.weight > 0` and `splats.pose_opt: true`, the pose deltas go NaN within
50 training steps and the final render dies inside `torch.linalg.inv`.

What was ruled out:

- The loss **value** stays finite the whole way through — it is the **backward** pass that produces
  the NaN, not the forward.
- The `geo` and `ncc` sub-weights make no difference at any setting tried.
- `pgsr_normal` alone (without `pgsr_multiview`) is fine with `pose_opt: true`.

Consequence: the splats parity harness pins `pose_opt: False` on its `3dgs_pgsr` config. That is a
workaround for this bug, not a preference, and the comment at that config says so. If this is ever
fixed, the parity baseline for that one config must be re-measured.

## 2026-09-05 — `av` is declared but missing from `uv.lock`, and `uv lock` cannot regenerate

`pyproject.toml:35` declares `"av>=17.0"` — PyAV is a hard runtime dependency of
`collab_splats/preproc/video.py` since the decode rewrite. **`uv.lock` has no `av` entry at
all** (`grep '^name = "av"' uv.lock` → nothing), so a fresh `uv sync` produces an environment
where importing `collab_splats.preproc` raises `ModuleNotFoundError: No module named 'av'`.
The current venv is fine (`av 17.0.1`, installed out of band); only a from-lock install breaks.

**The obvious fix does not work.** `uv lock` fails before it can add anything:

```
× Failed to build `bae @ git+https://github.com/pypose/bae.git@0.2.4`
├─▶ The build backend returned an error
hint: ... `bae` depends on `torch`, but doesn't declare it as a build dependency
```

`bae` needs `torch` at build time and does not declare it, so uv's build isolation cannot
resolve the graph at all. `uv lock --offline` fails earlier still — it cannot fetch the pinned
git commits. Neither failure has anything to do with `av`; the lock is simply not regenerable
in this container as `pyproject.toml` currently stands.

Unblocking it means adding

```toml
[tool.uv.extra-build-dependencies]
bae = ["torch"]
```

to `pyproject.toml` and re-locking, which rewrites the whole lockfile. That is a dependency-
owner change touching a file three concurrent sessions build against — deliberately **not**
done from the preproc session. Until then, do not hand-edit `uv.lock`: its entries carry
hashes and a hand-added block will not match what a re-lock produces.

## 2026-09-05 — env drift: `tests/wrapper/test_splats_stage.py` cannot collect, `gsplat.losses` missing (SUPERSEDED 2026-09-06 — venv rebuilt to the pinned `1.5.3 @ d2f5c0f`)

`tests/wrapper/test_splats_stage.py` fails at COLLECTION, so any full `tests/wrapper` run
either aborts or needs `--continue-on-collection-errors` to report the rest:

```
E   ImportError: cannot import name 'losses' from 'gsplat'
      (/opt/venv/reconstruction/lib/python3.11/site-packages/gsplat/__init__.py)
```

The venv holds released **gsplat 1.4.0 from PyPI**, which has no `gsplat.losses`.
`pyproject.toml:259` pins the source build instead — `rev = "d2f5c0f"`, upstream main
@ 2026-07-09, version string 1.5.3, which does have it. A plain `uv sync` or a stray
`pip install gsplat` replaces the source build with the PyPI wheel; the pin itself is
committed and correct.

Not a code regression — no repo change causes it, and it predates the pointcloud-cleanup
work. Workaround for a green run: `--continue-on-collection-errors`, or
`--ignore=tests/wrapper/test_splats_stage.py`. Real fix: rebuild gsplat from the pinned rev
in the venv.

`tests/wrapper/test_sfm_stage.py` was `test_vda_context.py` until 2026-09-05, when the VDA
context stream was deleted and the surviving `_run_sfm` tests were renamed with the file. It
used to be caught by both env gaps — it imported `_stub_reconstructor` from
`test_splats_stage.py`, so it failed the same collection, and its `_run_sfm` tests then died
on `importlib.metadata.version("instantsfm")` (`PackageNotFoundError`; `instantsfm` is not
installed in this venv at all). Neither applies any more: the shared stub moved to
`tests/wrapper/_stubs.py` (`9c09585b`), the misfiled splats test moved back to
`test_splats_stage.py`, dropping the cross-module import, and `_patched_sfm` stubs the version
lookup alongside every other leg of that same dependency. The module collects and passes
standalone in this venv — 10 tests, no ignore needed.

One test lost coverage in that move rather than gaining it:
`test_splats_conf_percentile_log_reports_the_masked_fraction` now lives in
`test_splats_stage.py`, which is the module that cannot collect. It passed while it was
misfiled in `test_sfm_stage.py` and does not run at all now — it is not a failure, it is
simply not executed until gsplat is rebuilt from the pinned rev.

## 2026-08-23 — pre-existing on `refactor/cu121-uv-migration`: 2 `test_qa` OpenCV-cannot-fit tests + 4 env/working-tree failures

**The two `test_qa` entries are RESOLVED as of 2026-09-05** — both pass on this branch
(measured 52/52 in `tests/preproc/test_qa.py` before the Task 26 collapse), and
`compute_parallax` no longer exists: the pair-motion helpers collapsed into
`qa.compute_pair_motion`, so the first test is now
`::test_parallax_is_nan_when_opencv_cannot_fit`. The original entry follows.

`tests/preproc/test_qa.py::test_compute_parallax_is_nan_when_opencv_cannot_fit` and
`::test_compute_video_quality_survives_a_pair_opencv_cannot_fit` fail identically on the base
branch (`compute_parallax` returns a finite value, 0.0123, where the test expects NaN) — not a
splats-module regression; owed to the preproc-cleanup owner. Seen in the same run and also
NOT regressions: `tests/evals/test_run_vggt_slam.py` ×2 (gitignored `third_party/VGGT-SLAM`
absent in a git worktree — env, pass in the main checkout) and
`tests/examples/test_run_pipeline_remote.py::test_main_rejects_a_scene_id_that_is_not_a_curated_dir_name`
×2 (the fix lives in a concurrent session's **uncommitted** edit to that test + `remote/rerun.py`
on the base checkout; committed HEAD fails).

## 2026-08-21 — RESOLVED: xfeat GPU parity test failed under TF32 import pollution

`tests/localization/test_local_matcher.py::test_real_xfeat_general_path_matches_pairwise`
(CUDA-gated; licenses `xfeat`'s membership in `FEATURE_MATCH_MODELS`) failed in any pytest
process that ALSO collected a module importing `collab_splats.pointcloud.feedforward.base`
(e.g. `tests/geometry/test_verification.py`): **mapanything sets
`torch.backends.cuda.matmul.allow_tf32 = True` and float32 matmul precision `high` at
module import**, and under TF32 the batched vs pairwise match paths flip near-threshold
matches (measured 969 vs 967), breaking the byte-equal parity assertion. Deterministic:
`pytest tests/localization/test_local_matcher.py tests/geometry/test_verification.py`
reproduced it 3/3 on an idle GPU; the file alone passed 20/20 and the single test passed
5/5 (serial and under a synthetic matmul load) — an earlier same-day diagnosis blaming
concurrent CUDA processes was wrong; both original failures had a polluting module in the
same process.

**Fixed** (same day): `strict_fp32` fixture in `test_local_matcher.py` pins
`allow_tf32=False` + precision `"highest"` (save/restore) on the three parity gates. The
combo repro passes 2/2 post-fix. Production is unaffected by design — it always runs with
feedforward imported (TF32 on), where a ±2 near-threshold match difference is not a
correctness issue; parity is asserted at full precision, where the paths are byte-equal.

## 2026-08-21 — NOT transient after all: 3 reconstructor/base.yaml failures, now committed state

**2026-09-05 correction.** The entry below called these a concurrent session's *uncommitted*
`configs/base.yaml` edit and predicted they would resolve when that session committed. The
session committed the yaml and never updated the tests, so three of the six are now a
standing red on `refactor/cu121-uv-migration`:

```
tests/wrapper/test_reconstructor.py::test_init_fills_defaults_from_base_yaml
    assert rec.config["preproc"]["fps"] == 1.0      ->  assert 2.0 == 1.0
tests/wrapper/test_reconstructor.py::test_mesh_clean_repair_defaults_off
    assert ...kwargs["clean_repair"] is False       ->  assert True is False
tests/wrapper/test_reconstructor.py::test_base_yaml_mesh_has_fidelity_keys
    assert cfg["mesh"]["conf_percentile"] is None   ->  assert 20 is None
```

`configs/base.yaml` carries `fps: 2.0`, `clean_repair: true`, `conf_percentile: 20` on
`refactor/cu121-uv-migration` (`391d44fb`), on `preproc/integration`, and on their merge base
— all three identical. The three asserts are byte-identical across the same three trees. So
this is a stale-test failure inherited from the mesh/splats work, not a merge artefact and
not preproc's to resolve: `test_mesh_clean_repair_defaults_off` asserts a default the shipped
config no longer has, and its *name* encodes the dead premise, so correcting it is a rename,
not a value edit. Owed to the mesh owner.

The other three names in the original entry (`test_build_localization_db_runs_when_missing`
and the two in `test_reconstructor_loger_kwargs.py`) do pass now.

The original entry follows.

## 2026-08-21 — transient: 6 reconstructor/base.yaml failures from a concurrent session's working tree (SUPERSEDED 2026-09-06)

`tests/wrapper/test_reconstructor.py` (`test_init_fills_defaults_from_base_yaml`,
`test_mesh_clean_repair_defaults_off`, `test_build_localization_db_runs_when_missing`,
`test_base_yaml_mesh_has_fidelity_keys`) and `tests/wrapper/test_reconstructor_loger_kwargs.py`
(both tests) failed in the 2026-08-21 full-suite run because a **concurrent session's
uncommitted `configs/base.yaml` edit** deleted the `pointcloud.loger` block and changed the
mesh defaults (`voxel_size`, `sdf_trunc`, `clean_repair: true`, `conf_percentile: 20`,
`native_resolution: true`, `color_map_iterations: 300`) that these tests assert. Working-tree
state, not repo state: at HEAD the tests' targets exist. Resolves when that session commits
(with test updates) or reverts. Do not "fix" the tests or the yaml from another session.

**Superseded 2026-09-06:** the yaml did commit; these are stale asserts, not working-tree
state, and they are durable debt owned by the mesh owner (see the 2026-09-06 entry above).

## 2026-08-18 — RESOLVED: 3 BA `test_optimize_*` xfails (bae/pypose target bug)

The deferred bae/pypose integration bug (bae `LM.step` calls `self.model(input)` without
`target`, pypose>=0.7 `RobustModel.forward` requires it — entries of 2026-06 below) is fixed
at `6e8ab15`: `_optimize` binds `target=None` on the wrapped model instance via
`functools.partial` right after constructing the LM optimizer. The three xfail markers are
removed; `tests/geometry/test_bundle_adjustment.py` is fully green (25 passed) on CUDA+bae.
Suite-count references below that say "3 xfailed" predate this.

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

## 2026-08-22 — preproc cleanup: `keyframe_extraction.ipynb` broken again (OPEN)

`docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` is broken by the
preproc cleanup. The 2026-07-20 fix below rewrote it against `sample_frames`,
`score_frames`, `compute_blur_score` and `check_frame_quality` — **all four are
deleted**. Sampling is now `sample_fps` / `sample_uniform` / `sample_optical_flow`
over a `video_quality_report.json`, and the per-frame quality gate is
`filter_frame_quality`, which reads report rows rather than scoring frames itself.

Not patched here. The notebook's whole narrative is "score frames, then gate them",
which is no longer how the module works; it is rebuilt in a separate pass where it
can demonstrate the measure-then-select flow instead of being patched to compile
(design doc §7, `docs/superpowers/specs/2026-08-22-preproc-cleanup-design.md`).

**2026-09-05 update — `plot_quality_examples` resolved by deletion.** It and
`plot_disparity_sensitivity` are gone from `collab_splats/preproc/viz.py`, along
with the notebook cells that called them and the `_VideoFrameLookup` shim that
faked a `FrameStore` for them — the committed `plot_quality_examples` traceback
goes with them. Neither had a pipeline caller. The rest of this entry still
stands: the notebook remains broken on `sample_frames` / `score_frames`, and the
rebuild pass must not restore the two deleted plots.

## 2026-07-20 — frame-store refactor: notebook follow-ups (RESOLVED, superseded above)

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
- **Group C (webapp async)** — HISTORICAL. Was a FastAPI/ASGI app needing `pytest-asyncio` +
  `httpx`. `collab_splats/webapp/` was deleted 2026-07-29 (dead prototype, superseded by
  `collab_splats/dashboard/`); both dev deps and `asyncio_mode = "auto"` went with it.
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

## vismatch not installed (env, not a code bug) — 2026-09-05

`vismatch==1.3.1` is a declared dependency (`pyproject.toml:70`) that is **absent from
`/opt/venv/reconstruction`** — not in `site-packages`, and no source tree on disk to put
on `PYTHONPATH` the way `collab_data` can be. It fails **16 tests** in
`tests/localization/test_local_matcher.py`, every one with
`ModuleNotFoundError: No module named 'vismatch'`:

```
test_extract_returns_local_features            test_match_images_no_indices_when_unstable
test_extract_handles_tensor_outputs            test_match_images_downgrades_on_recovery_failure
test_extract_asserts_pixel_frame               test_probe_sets_stability_flag
test_extract_passes_chw_unit_range_tensor      test_match_mutual_nn_for_allowlisted_model
test_descriptor_level_match_unsupported        test_match_empty_descriptors_returns_empty
test_match_images_pre_ransac_with_indices      test_match_still_raises_for_non_listed_model
test_split_flag_off_for_unknown_wrappers       test_loma_match_without_payload_names_the_rebuild
test_real_loma_split_extract_parity            test_real_loma_split_match_parity_after_zarr_roundtrip
test_real_xfeat_general_path_matches_pairwise
```

(17 names — `test_real_xfeat_general_path_matches_pairwise` is one of the 16 plus the
separately-listed TF32 entry above, which is a different, already-fixed failure mode.)

**Do not `pip install` it into this venv to make them pass.** It is shared with several
concurrent sessions, and `vismatch` hard-pins `uniception==0.1.1` and `lightning==2.3.3`,
which `pyproject.toml:236-240` deliberately overrides during resolution. A plain install
fights those overrides and can break the other sessions' environments. Restoring it is a
`uv sync` in a container of its own.

Confirm with:

```bash
/opt/venv/reconstruction/bin/python -c "import vismatch" ; ls -d \
  /opt/venv/reconstruction/lib/python3.11/site-packages/vismatch*
```

## gsplat 1.4.0 installed against a v1.5.3 pin (env, not a code bug) — 2026-09-05

`collab_splats/splats/losses.py` does `from gsplat import losses`. That module exists only in
the commit `setup.sh` pins, `d2f5c0f` (v1.5.3). The env has **gsplat 1.4.0**, so the import
raises and these suites **cannot be collected at all** — they are collection errors, not
failures, and a `pytest tests/ -q` run will report them as errors before any test body runs:

- `tests/wrapper/test_splats_stage.py`
- `tests/evals/test_eval_splats.py`
- `tests/splats/` (the whole directory)

`tests/wrapper/test_vda_context.py` **used to be on this list and no longer is.** It never
imported gsplat itself — it imported the `_stub_reconstructor` helper from
`test_splats_stage.py`, which imports `SplatsConfig` at module level, and inherited the
dependency through that. Commit `9c09585b` moved the helper to `tests/wrapper/_stubs.py`,
which imports nothing from `collab_splats.splats`, and the file collects 33 tests again.
Do not re-add it to any `--ignore` list.

One test in it *does* still fail on gsplat:
`test_splats_conf_percentile_log_reports_the_masked_fraction` patches
`collab_splats.splats.trainer.train`, and importing that package raises
`ImportError: cannot import name 'losses' from 'gsplat'`. It surfaces as
`AttributeError: module 'collab_splats' has no attribute 'splats'`. One test, not a
collection error — the other 32 still run.

Plus four real failures from the same cause:

- `tests/mesh/test_absent_confidence.py::test_splats_depth_targets_skip_masking_when_confidence_absent`
- `tests/test_cu121_migration.py::test_import_all_modules` — asserts on a list of import
  failures; the five it reports are `collab_splats.splats` and its `losses`, `rendering`,
  `trainer`, `outputs` submodules, all the same `ImportError`
- `tests/test_cu121_migration.py::test_gsplat_upstream_pinned` — `ModuleNotFoundError: No
  module named 'gsplat.losses'`
- `tests/test_cu121_migration.py::test_gsplat_commit_constant_matches_pyproject`

Confirm with:

```bash
/opt/venv/reconstruction/bin/python -c "import gsplat; print(gsplat.__version__); \
  print(hasattr(gsplat, 'losses'))"
```

Expected on a correct env: `1.5.3` / `True`. Observed 2026-09-05: `1.4.0` / `False`.

Fix by reinstalling the pin (`setup.sh`, or the `gsplat` git pin in `pyproject.toml`) — do not
work around it in code. Note that gsplat is the one break here that a pip install cannot fix:
building it needs `nvcc`, and `/usr/local/cuda-12.1` in this container is runtime-only (no
`bin/`). Restoring the pin needs a container with the CUDA build toolkit.

`_run_sfm` integration coverage is **not** blocked by this. `tests/wrapper/test_sfm_stage.py`
(ex `test_vda_context.py`) is the only suite exercising `Reconstructor._run_sfm` end to end,
and since `9c09585b` it collects; its `_run_sfm` tests fail on the separate `instantsfm`
break below.

## `instantsfm` not installed (env, not a code bug) — 2026-09-05

`collab_splats/wrapper/reconstructor.py:1197` calls `importlib.metadata.version("instantsfm")`
to record backend provenance, which raises:

```
importlib.metadata.PackageNotFoundError: No package metadata was found for instantsfm
```

This is what actually blocks end-to-end coverage of the SfM stage. It fails the four
`_run_sfm` tests in `tests/wrapper/test_sfm_stage.py` (the file was `test_vda_context.py`
until the VDA context stream was deleted on 2026-09-05). Measured on the
`preproc/integration` x `391d44fb` merge: that module reports `5 failed`, split **4
instantsfm + 1 gsplat**, and nothing else — the 14 earlier instantsfm failures went with
the VDA context tests, which are deleted.

Same root cause as the other env breaks — a plain `uv sync` pruned the venv and `setup.sh`'s
post-sync blocks were never re-run; `pyproject.toml:107-109` warns about exactly this. Unlike
gsplat it needs no CUDA toolkit, so re-running the InstantSfM block of `setup.sh` fixes it.

## `collab_data` not installed (env, not a code bug) — 2026-09-05

`collab_splats/remote/sources.py:14` imports `collab_data.data_dashboard.rclone_client`. The
package is not in the venv, so these fail at collection:

- `tests/remote/test_sources.py`
- `tests/remote/test_rerun.py`

Because a collection error aborts the whole pytest run, this and the gsplat break above must
both be `--ignore`d to get any result at all from `pytest tests/`. Neither is caused by repo
code; do not work around either in source.
