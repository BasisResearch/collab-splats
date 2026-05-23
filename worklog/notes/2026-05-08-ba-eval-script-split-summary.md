# BA Eval Script Split — Session Summary

**Date:** 2026-05-08 (session 2)
**Branch:** `refactor/core-modules`
**Spec:** `worklog/history/specs/2026-05-08-ba-eval-script-split-design.md`
**Plan:** `/root/.claude/plans/please-look-at-this-partitioned-gadget.md`
**Predecessor:** `worklog/notes/2026-05-08-bundle-adjustment-notebook-debug.md` (debug handoff that this session resolves)

---

## Goal

Run evaluations on downloaded datasets per existing eval-harness design (script = compute, notebook = viz). Prior session left BA condition untested — `bundle_adjustment.ipynb` repeatedly killed by SIGKILL mid-track-extraction. Realign with `evals/eval_gt.py` as the runner; repurpose the notebook as small API how-to.

---

## What was done

### 1. SIGKILL diagnosis corrected

Earlier debug log blamed Claude Code background-bash duration cap. Fresh probe shows real cgroup OOM:

| Probe | Value |
|-------|-------|
| `/sys/fs/cgroup/memory.max` | 46.6 GB (container cap) |
| `free -h` | 503 GB host total (irrelevant from inside container) |
| `dmesg` | denied (no kernel-log access) |
| `nvidia-smi` | A40, 46 GB, only 6.9 GB used |

Container cap, not host RAM, is what BA must fit under. Documented in spec + WORKLOG.

### 2. Dataset acquired

- `bash evals/download_7scenes.sh chess /workspace/collab-splats/data/7scenes`
- Outer zip extracts to `chess/chess/`; inner `seq-01.zip` had to be unzipped manually (download script does not recurse — script comment is also slightly wrong about layout, but the actual flat layout matches what `evals/datasets.py:_load_7scenes` expects).

### 3. Ran `evals/eval_gt.py` via tmux

Launched in detached tmux (no Claude harness mediating long compute). Three runs in total:

| Run | Frames | Outcome |
|-----|--------|---------|
| #1 | 50 | baseline OK; BA optimizer ran clean (Loss 7.1e10 → 1.8e5); crashed at `eval_gt.py:66` with `AttributeError: 'BundleAdjustment' object has no attribute 'outputs'` |
| #2 | 50 | killed silently mid-track-extraction (real cgroup OOM — concurrent RSS sampler in another shell ate budget) |
| #3 | 50 | clean — all 3 conditions completed end-to-end |

### 4. Bug fix (TDD)

**Surface:** `BundleAdjustment` lacked the `outputs` / `raw_outputs` proxy properties that `LoopClosure` has (`wrappers.py:112-122`). `eval_gt.py:66` reads `creator.outputs.extrinsics` after every condition; works for baseline (`VGGTXCreator` exposes `.outputs` directly) and for LC (LoopClosure proxies), fails for BA only.

**Fix:** added `@property outputs` + setter and `@property raw_outputs` + setter to `BundleAdjustment` in `collab_splats/pointcloud/wrappers.py`.

**Tests (TDD):** two regression tests added in `tests/pointcloud/test_wrappers.py`:
- `test_bundle_adjustment_outputs_proxies_to_base`
- `test_bundle_adjustment_raw_outputs_proxies_to_base`

Both fail before fix, pass after.

### 5. Stale mock cleanup

`tests/pointcloud/test_bundle_adjustment.py` mocked the pre-2026-05-08 `predict_tracks` API (returned per-query-frame lists). Session-1 fix to `extract_tracks_vggsfm` switched to single concatenated `np.ndarray` outputs; mocks rewritten to match.

### 6. Viz notebook refreshed

`docs/pointcloud/eval_7scenes_gt.ipynb` — `RESULTS_DIR` was pointed at stale path `../../eval_results/chess_seq01`; corrected to `../../evals/results/chess_seq01`. Re-executed in-place; all 3 conditions render in metrics table + 3D trajectory + per-frame ATE plot.

### 7. BA notebook rewrite

`docs/pointcloud/bundle_adjustment.ipynb` — full rewrite. Old version was a 194-frame BA debugging artifact mixing compute + viz, repeatedly OOM-killed. New version is a small API how-to:

- ≤ 10 frames (5 frames symlinked at runtime from chess seq-01 — no new committed binaries)
- Two cells: wrapper API (`BundleAdjustment(VGGTXCreator())`) + manual API (`extract_tracks_vggsfm` + `run_bundle_adjustment`)
- Header cell points readers to `eval_7scenes_gt.ipynb` for benchmarks and `evals/eval_gt.py` for compute
- Bug caught in rewrite: `PointcloudResult` exposes `points` / `camera_poses` (not `pts3d` / `extrinsics`); old notebook had this wrong but never finished running so the error was hidden. Corrected.

Runtime: <90 s end-to-end. `jupyter nbconvert --execute` exits 0.

---

## What was completed

### Acceptance criteria (all met)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `eval_gt.py … --conditions baseline ba lc` exits 0 on chess seq-01 | ✅ |
| 2 | `metrics.json` has 3 conditions, finite ATE/RPE | ✅ |
| 3 | `trajectories.npz` has `pred_baseline`, `pred_ba`, `pred_lc`, `gt` | ✅ |
| 4 | `nbconvert --execute eval_7scenes_gt.ipynb` exits 0; 3 conditions render | ✅ |
| 5 | `nbconvert --execute bundle_adjustment.ipynb` exits 0; ≤ 10 frames; runtime ≤ 90 s | ✅ |
| 6 | `pytest tests/pointcloud/test_bundle_adjustment.py` green | ✅ (30/30) |
| 7 | WORKLOG session 2 entry + spec committed; commits follow `<scope>:` prefix | ✅ |

### Metrics (chess seq-01, 50 frames)

| condition | ATE RMSE | ATE mean | ATE max  | RPE trans RMSE | RPE rot RMSE |
|-----------|----------|----------|----------|----------------|--------------|
| baseline  | 0.0553 m | 0.0485 m | 0.1191 m | 0.0069 m       | 0.146°       |
| ba        | 0.0577 m | 0.0506 m | 0.1256 m | 0.0068 m       | 0.150°       |
| lc        | 0.0548 m | 0.0480 m | 0.1193 m | 0.0073 m       | 0.186°       |

BA marginally worse than baseline on this short sequence (no loop closures to anchor LM solver). LC marginally better. Behaviour expected; both conditions runnable.

### Files committed (4 commits on `refactor/core-modules`)

| Commit | Scope | Files |
|--------|-------|-------|
| `150a36e` | docs(superpowers) | `worklog/history/specs/2026-05-08-ba-eval-script-split-design.md` |
| `fec2fe4` | fix(pointcloud) | `collab_splats/pointcloud/{bundle_adjustment.py, wrappers.py}` + `tests/pointcloud/test_{bundle_adjustment, wrappers}.py` |
| `405b26f` | feat(eval) | `evals/results/chess_seq01/**` + `docs/pointcloud/{eval_7scenes_gt, bundle_adjustment}.ipynb` |
| `cc29e89` | chore(worklog) | `worklog/WORKLOG.md` |

### Test count

- `tests/pointcloud/test_bundle_adjustment.py` — 5 pass
- `tests/pointcloud/test_wrappers.py` — 25 pass (23 prior + 2 new regression)
- Combined: 30 / 30 green

---

## What remains

Nothing in scope for this session. Open follow-ups recorded in spec under "Open questions":

- **vggsfm track extraction shape**: fully exercised end-to-end on chess seq-01; no further surprises here.
- **Long-sequence batching**: chess seq-01 has 1000 frames but only 50 fit safely under the 46.6 GB cgroup cap on the BA path. `LoopClosure` already partitions into submaps; `BundleAdjustment` does not. Future spec.
- **PR**: `refactor/core-modules` → `main` still queued per WORKLOG (predates this session).

---

## Lessons recorded

1. **Container memory cap ≠ host RAM.** `cat /sys/fs/cgroup/memory.max` is the number that matters when tracking OOM kills; `free -h` lies from inside a sandboxed container.
2. **tmux alone does not save you from cgroup OOM.** Earlier debug log assumed harness was killing tasks; switching to tmux + `nohup`/`disown` only fixes harness-mediated kills, not container memory caps.
3. **Don't run side-shell loops while heavy eval runs.** Concurrent RSS sampler ate cgroup budget mid-track-extraction in run #2 and pushed BA over the cap; same compute completed clean in run #3 with no concurrent samplers.
4. **Mocks rot quietly when the real API changes.** Two BA tests had been passing against pre-2026-05-08 `predict_tracks` mocks even after the real module was updated — caught only when running the full pytest suite under the new code path.
5. **`BundleAdjustment` and `LoopClosure` are sibling wrappers but had divergent surface area.** When extending one, mirror the other unless a deliberate reason not to.
