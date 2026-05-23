# BA Eval Script Split — Design Spec

**Date:** 2026-05-08
**Branch:** `refactor/core-modules`
**Plan:** `/root/.claude/plans/please-look-at-this-partitioned-gadget.md`
**Supersedes interactive debug:** `docs/superpowers/2026-05-08-bundle-adjustment-notebook-debug.md`

---

## Context

### Problem

Bundle adjustment (BA) condition in the GT eval harness is untested. The previous attempt to validate BA ran inside `docs/pointcloud/bundle_adjustment.ipynb` and failed repeatedly with SIGKILL (exit 137) mid-track-extraction. Earlier debug log misdiagnosed this as a Claude Code background-bash duration cap; **a fresh diagnosis confirms the kill is real cgroup OOM** (see "SIGKILL diagnosis" below).

This spec realigns BA validation with the design intent of the existing eval harness (spec `worklog/specs/2026-05-07-gt-eval-harness-design.md`): **scripts run compute, notebooks load and plot.**

### Outcome

1. `evals/eval_gt.py --conditions baseline ba lc` produces `metrics.json` + `trajectories.npz` + plots for all three conditions on 7-Scenes chess seq-01.
2. `docs/pointcloud/eval_7scenes_gt.ipynb` shows results — pure viz, zero inference.
3. `docs/pointcloud/bundle_adjustment.ipynb` repurposed as a small (≤10 frame, <60s) API-docs notebook for BA usage — example, not eval, not debug.
4. Future agents: long evals run via tmux + frame-budgeted to fit cgroup memory cap.

---

## SIGKILL diagnosis (2026-05-08)

| Probe | Value | Interpretation |
|-------|-------|----------------|
| `dmesg` | empty (denied in container) | Cannot read kernel log directly |
| `/sys/fs/cgroup/memory.max` | `49999998976` (46.6 GB) | **Container memory cap** — host RAM is irrelevant |
| `free -h` total | 503 GiB | Host total — misleading from inside container |
| `nvidia-smi` | A40, 46 GB total, 6.9 GB used | GPU not the bottleneck |
| `tmux` | 3.2a available | Use as canonical long-eval runner |

**Conclusion:** previous SIGKILLs were almost certainly real cgroup OOM, not Claude harness duration kills. VGGT-X + dinov2 + 30–194 frames + tracks + reproj structures pushed RSS over the 46.6 GB container cap. **Plan implication:** running via tmux alone is insufficient — must keep frame count modest and watch RSS during BA. `--max_frames 50` is reasonable for chess; larger sequences need staged batching (out of scope here).

---

## Architecture

| Layer | Role | Files |
|-------|------|-------|
| Compute | Run all 3 conditions, dump metrics + trajectories | `evals/eval_gt.py` (no changes expected unless BA bugs surface) |
| BA core | Track extraction + bundle adjustment | `collab_splats/pointcloud/bundle_adjustment.py`, `collab_splats/pointcloud/wrappers.py` |
| Eval viz | Load `metrics.json` + `trajectories.npz`, plot 3 conditions | `docs/pointcloud/eval_7scenes_gt.ipynb` (already correct shape) |
| API docs | Tiny how-to for BA wrapper + manual API | `docs/pointcloud/bundle_adjustment.ipynb` (full rewrite, small canned data) |
| Tracking | Cross-session continuity | `worklog/WORKLOG.md`, `worklog/known-test-failures.md` |

### Reused existing functions

- `evals.eval_gt._make_creator(condition)` — wraps VGGTXCreator with `BundleAdjustment` / `LoopClosure` (eval_gt.py:53–60)
- `collab_splats.pointcloud.wrappers.BundleAdjustment` — wrapper API (`_apply_ba` is BA driver)
- `collab_splats.pointcloud.bundle_adjustment.extract_tracks_vggsfm` + `run_bundle_adjustment` — manual API
- `collab_splats.pointcloud.__init__.make_creator(name, use_ba=True)` — factory shorthand

### Per-condition output contract

`eval_gt.py:159` calls `_run_condition(cond, tmp_image_dir, args.output_dir / cond)`. Each condition gets its own subdirectory: `{output_dir}/{baseline,ba,lc}/colmap/sparse/0/` plus `transforms.json` + `sparse_pc.ply`. Aggregated metrics + trajectories sit at the top level: `{output_dir}/metrics.json`, `{output_dir}/trajectories.npz`, `{output_dir}/plots/{trajectory.png,ate_per_frame.png}`.

### Notebook scope statement

**`docs/pointcloud/eval_7scenes_gt.ipynb`** — eval results. Loads `metrics.json` + `trajectories.npz`. **No inference.** Cells: imports + path config, metrics table (3 conditions), 3D camera trajectory, per-frame ATE.

**`docs/pointcloud/bundle_adjustment.ipynb`** — BA API how-to. ≤10 frames, <60s runtime. **No eval.** Cells: setup (5 frames sampled at runtime), wrapper-API demo (`BundleAdjustment(VGGTXCreator())`), manual-API demo (`extract_tracks_vggsfm` + `run_bundle_adjustment`). Markdown header cell points readers to `eval_7scenes_gt.ipynb` for benchmark results and `evals/eval_gt.py` for compute.

---

## Verification

Acceptance criteria:

1. `tmux new -d -s eval bash -lc 'evals/eval_gt.py … --conditions baseline ba lc'` exits 0 on chess seq-01 (max_frames=50)
2. `evals/results/chess_seq01/metrics.json` contains 3 conditions with finite ATE/RPE numbers
3. `evals/results/chess_seq01/trajectories.npz` contains `pred_baseline`, `pred_ba`, `pred_lc`, `gt`
4. `jupyter nbconvert --execute docs/pointcloud/eval_7scenes_gt.ipynb` exits 0; plots show 3 conditions
5. `jupyter nbconvert --execute docs/pointcloud/bundle_adjustment.ipynb` exits 0; runtime ≤ 90s; uses ≤ 10 frames
6. `pytest tests/pointcloud/test_bundle_adjustment.py -v` green
7. WORKLOG 2026-05-08 session 2 entry exists; this spec committed; commits follow `<scope>:` prefix convention

---

## Risks

| Risk | Mitigation |
|------|------------|
| BA crashes in eval_gt.py with new bug not caught by mocked unit tests (mock vggt + bae) | TDD repro in `test_bundle_adjustment.py` before fixing; commit fix separately under `fix(bundle_adjustment): ...` |
| 50 frames still triggers cgroup OOM on BA path | Step down to 30 → 20 → 10 frames; record minimum viable count in WORKLOG; no proactive batching changes (out of scope) |
| `_make_creator("ba")` output dir layout `eval_gt.py` postprocess can't read | Verified: per-condition `{output_dir}/{cond}/` per Explore audit; postprocess at `eval_gt.py:167` |
| Canned data for docs notebook bloats repo | Sample 5 frames at notebook runtime from already-downloaded chess seq-01; no new committed binaries |
| Spec drift if inline docs skipped mid-execution | User directive 2026-05-08: every step ends with WORKLOG/spec/comment update. Plan enforces. |

---

## Open questions

- **vggsfm track extraction shape** — unit tests mock `predict_tracks` so a shape mismatch with installed `vggt 0.0.1` could surface only at runtime (the bug fixed in `bundle_adjustment.py:98–106` per debug log). Step 3 will exercise this end-to-end for the first time inside the eval harness.
- **Long-sequence batching** — chess seq-01 has hundreds of frames; only 50 fit safely. Multi-submap path via `LoopClosure` already partitions; `BundleAdjustment` does not. Future work, not this spec.

---

## Resolved during execution

### `BundleAdjustment.outputs` proxy missing (2026-05-08)

First end-to-end run on chess seq-01 surfaced `AttributeError: 'BundleAdjustment' object has no attribute 'outputs'` at `eval_gt.py:66` after BA optimization succeeded. `LoopClosure` proxies `outputs`/`raw_outputs` to `self.base` (`wrappers.py:112-122`); `BundleAdjustment` did not. Mocked unit tests didn't catch this because `eval_gt._run_condition` was the only consumer reading `creator.outputs` after a BA run, and that path was never exercised.

**Fix:** added `@property outputs` (+ setter) and `@property raw_outputs` (+ setter) to `BundleAdjustment` in `collab_splats/pointcloud/wrappers.py`. Two regression tests added in `tests/pointcloud/test_wrappers.py`:
- `test_bundle_adjustment_outputs_proxies_to_base`
- `test_bundle_adjustment_raw_outputs_proxies_to_base`

Both fail before fix, pass after. All 25 wrapper tests green.
