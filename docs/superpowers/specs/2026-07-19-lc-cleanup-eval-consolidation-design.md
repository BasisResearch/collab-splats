# Loop Closure Cleanup & Eval Consolidation — Design

**Date:** 2026-07-19
**Status:** Draft (pending user review)
**Related:** [[project_geometry_package]] (2026-07-11 consolidation this builds on), `docs/superpowers/specs/2026-07-08-lc-parity-validation-design.md`, `evals/README.md`

## Context

Loop closure (`collab_splats/geometry/loop_closure/`) is functionally complete: it matches
upstream VGGT-SLAM results and has been validated against `vggt_omega`. This is a
finishing/hardening pass, not new functionality:

1. Is the `LoopClosure` wrapper easy to script against?
2. Clean up `evals/` into a clear, minimal set of scripts for "the complete evaluation."
3. Those scripts should be minimal enough that a reader can see how to use them as an example.
4. Audit `collab_splats/geometry/loop_closure/` for unused/unnecessary code; remove it;
   add inline block comments to everything that remains, per the repo's existing
   comment convention.

**Guardrail (explicit user decision):** every change in this pass is **behavior-preserving**.
The full test suite (19 files under `tests/geometry/loop_closure/` + LC-adjacent tests
under `tests/evals/`) must pass unchanged after every step. No test rewrites to accommodate
new behavior — if a change would require one, it's out of scope for this pass.

## Non-goals

- No restructuring of `closure.py` back into separate files (e.g. `alignment.py`,
  `pose_graph.py`) to mirror upstream's file layout. Rejected: undoes the deliberate
  2026-07-11 consolidation (`−1917 LOC`, see [[project_geometry_package]]).
- No new features, config options, or scale methods.
- No re-litigating the 2026-05-31 "keep-all" eval decision wholesale — this pass narrows
  it deliberately (see Section 3), it doesn't reverse it as a blanket policy.

---

## Section 1 — Wrapper scriptability

**Finding:** the `LoopClosure` class (`collab_splats/geometry/loop_closure/wrapper.py`)
already wraps any feedforward creator via a proxy/`__getattr__` pattern:
`LoopClosure(base_creator, config).reconstruct(image_dir, output_dir)`. A full worked
example already exists in `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`
(cell 7), but it's buried inside a 14-cell analysis/visualization notebook — there's no
short, standalone reference for "how do I call this."

**Change:**
- Add a minimal usage example (4-6 lines) to `wrapper.py`'s module docstring: construct
  a base creator (`VGGTXCreator` or `MapAnythingCreator`), wrap with `LoopClosure`, call
  `.reconstruct(image_dir, output_dir)`. One-line pointer from there to the tutorial
  notebook for the full walkthrough (candidate matching, plots) — don't duplicate the
  notebook's content.
- Add the same snippet to `docs/source/api/geometry.rst` near the LC section.
- No class/API restructuring — confirmed via upstream comparison (below) that the
  proxy pattern itself isn't the friction; missing documentation is.

**Upstream size comparison** (informs Section 2, not a rewrite target):
our module = 1932 lines (`closure.py` 897, `wrapper.py` 422, `eval.py` 295, `graph.py`
193, `submap.py` 77, `__init__.py` 48) vs. upstream MIT-SPARK/VGGT-SLAM's comparable
core (`graph.py` + `solver.py` + `slam_utils.py` + `scale_solver.py`) = 788 lines. Some
gap is structural — `eval.py` (GT metrics) and `wrapper.py` (integration layer) have no
upstream equivalent; `closure.py`+`graph.py` alone (1090 lines) is the closer comparison
to upstream's 788, and most of that delta is dedup/anchor-scale/verify-gate logic
upstream doesn't have — not obvious bloat. Section 2 removes the two confirmed dead
spots found during this comparison; it does not chase the full gap by feature removal
(that would violate the behavior-preserving guardrail).

---

## Section 2 — Dead-code removal + inline comments

Scope: all 6 files in `collab_splats/geometry/loop_closure/`
(`__init__.py`, `closure.py`, `eval.py`, `graph.py`, `submap.py`, `wrapper.py`).

**Confirmed dead code to remove:**
- `_ODOM_INTRA_SIGMA_R`, `_ODOM_INTRA_SIGMA_T`, `_INTER_SIGMA_R`, `_INTER_SIGMA_T`
  (`closure.py:145-146`) — defined, zero usages anywhere in `collab_splats/` or `tests/`.
  `graph.py`'s `PoseGraph.__init__` hardcodes its own sigma arrays (0.05 uniform for
  SL4; `[0.05,0.05,0.05,0.20,0.20,0.20]` for SE3) — these constants were superseded and
  never removed.
- Duplicate `######## Absorbed from alignment.py #####` section header in `closure.py`
  (appears twice, ~line 47 and ~line 260) — consolidate to one, at the correct section.

**Confirmed NOT dead (fix docs instead of removing):**
- `eval.py`'s module docstring claims these are "stubs... implemented when a GT pose
  dataset is sourced." False — `ate_translation`, `auc_at_threshold`, and
  `umeyama_align` are live, imported by `evals/eval_gt.py` (the main runner) and
  exercised by 6 test files. Fix the docstring; don't touch the functions' logic.

**Duplication to collapse (behavior-preserving):**
- `eval.py`'s private `_umeyama_sim3` (operates on transposed `(3,N)` arrays) duplicates
  `closure.py`'s canonical `umeyama_sim3` (operates on `(M,3)` arrays). `_umeyama_sim3`
  has exactly one call site (`capture_pose_graph_loss`, `eval.py:249`). Replace it with
  a call to `closure.py`'s `umeyama_sim3`, transposing inputs/outputs at that one call
  site. Net: one fewer near-duplicate implementation of the same math.

**Comments:** every remaining logical block gets a short comment per the repo
convention (e.g. `# Sort images by filename; reject non-image extensions`), matching
the standard already followed in `graph.py` (upstream line-reference comments,
documented historical bug fix) and parts of `closure.py` (PGO builder). `submap.py`'s
existing `TODO(spec-2)` comment about the `torch` coupling violation is left as-is —
it's a forward-looking note tied to a different spec, not this pass's scope.

**Process note:** the two confirmed items above are locked in; the rest of the pass is
an audit-as-you-go read through all 6 files during implementation — anything else
found unused gets removed, flagged in the plan/PR description rather than pre-listed
here exhaustively.

---

## Section 3 — Eval cleanup

**Principle:** distinguish "one-off investigation of a bug that's now fixed / a
parameter now absorbed into `eval_gt.py`'s conditions" (no regen value — delete) from
"a comparison/sweep tool with ongoing regen value" (generalize and keep, backed by the
existing `get_creator()`/`make_creator()` registry at
`collab_splats/pointcloud/__init__.py:19-65`, which already makes per-backbone
instantiation flexible).

**Delete outright (12 files — `git rm`):**

| file | why |
|---|---|
| `evals/runners/parity_trace.py` | LC↔SLAM pose-extraction bug hunt — fixed, commit `1372ac2`, outcome in `docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-findings.md` |
| `evals/runners/our_solver_dump.py` | consumed only by `parity_trace.py` |
| `evals/runners/vggt_slam_solver_dump.py` | consumed only by `parity_trace.py` |
| `evals/runners/compare_solver_internals.py` | manual precursor to `parity_trace.py`, same bug |
| `evals/runners/compare_vggt_outputs.py` | same investigation, confirmed Δ=0, done |
| `evals/diag_pose_graph.py` | same investigation |
| `evals/check_ate_methods.py` | same investigation (sanity check, passed) |
| `evals/_ba_finding_eval.py` | BA tuning — outcome folded into `eval_gt.py`'s `ba`/`ba_track-density-N` conditions |
| `evals/sweep_incremental_ba.py` | same — capability now native to `eval_gt.py` |
| `evals/plot_incremental_ba_sweep.py` | same |
| `evals/incremental_ba_sweep.png` | orphaned output of the above; not cited by any spec doc |
| `evals/incremental_ba_sweep_200.png` | same |

**Keep + generalize (3 files — strip hardcoding, expose via CLI flags):**

| file | change |
|---|---|
| `evals/runners/run_disparity_sweep.py` | backbone is hardcoded to `vggt_spark` (line 108) — make it a `--backbone` flag. Remove historical "Track A/B fix" debug print-language (`_check_similarity_parity`, lines ~144-159) — bug is fixed, that language is stale. Keep the sweep-and-gate mechanism (disparity levels, ATE parity gate) as a general ours-vs-SLAM comparison tool. |
| `evals/runners/run_cross_model_benchmark.py` | `core_matrix()` (lines 58-74) hardcodes 4 backbones + specific chess keyframe dirs — replace with `--backbones`, `--framesets`/`--keyframe_lists`, `--submap_sizes` CLI flags. `RunSpec`/`build_commands`/`main` (already generic) stay as-is. |
| `evals/runners/build_benchmark_table.py` | already generic (glob-based aggregation) — expose `reference_backbone` (currently hardcoded `"vggt_spark"` at the `main()` call site) as a `--reference_backbone` flag. |

**Keep unarchived, unchanged (re-runnable tools):** `evals/eval_similarity_calibration.py`,
`evals/eval_multiview_conf.py`.

**Untouched (frozen reference data, cited by 5 spec docs):**
`evals/baselines/disparity_sweep/`, `evals/baselines/cross_model/`.

**Active path (comment/trim pass only, no structural change):** `evals/eval_gt.py`
(canonical "complete evaluation" entrypoint — stays canonical), `evals/eval_compare.py`,
`evals/eval_suite.sh`, `evals/runners/run_lc_parity.py`, `evals/runners/lc_parity_common.py`,
`evals/runners/run_vggt_slam.py`, `evals/runners/run_vggt_slam_lc.py`,
`evals/runners/compare_loop_edges.py` (kept as-is — `compose_slam_chain` is imported by
`tests/geometry/loop_closure/test_loop_edge_chain.py`).

**README rewrite:** `evals/README.md` drops rows for the 12 deleted files, updates the 3
generalized scripts' usage examples to show the new flags, collapses "Investigations A
and B" to a short historical pointer ("bug fixed / parameter absorbed — see spec docs
for the trail"; keep the doc links). Investigations C and D descriptions stay, updated
to reference the new flag-driven usage instead of the old hardcoded matrix.

**Pre-deletion check (implementation-time, not a design decision):** grep each of the
12 delete-candidates for imports from `tests/` or other `evals/` files before running
`git rm` — confirm no live import was missed. (`compare_loop_edges.py` already confirmed
imported by a live test and is not on the delete list.)

Net: −12 files from `evals/`; 3 files generalized (kept, not deleted) so any
backbone/param combination stays reproducible on demand, not just the frozen 2026-05-31
numbers.

---

## Testing / validation plan

1. Baseline: `/opt/venv/reconstruction/bin/python -m pytest tests/` green before any
   change (establish the starting point).
2. After Section 2 (module dead-code + comments): full suite must still pass — this
   is pure removal-of-unused-code plus one call-site swap (`_umeyama_sim3` →
   `umeyama_sim3`), so no test should need updating. If any test imports
   `_umeyama_sim3`, `_ODOM_INTRA_SIGMA_R/T`, or `_INTER_SIGMA_R/T` directly, that's a
   stop-and-reassess signal, not something to patch around.
3. After Section 3 (eval deletions/generalization): re-run the full suite (catches any
   missed import of a deleted file). Manually smoke-test the 3 generalized scripts with
   their new flags against a small frame count, confirm output shape matches what
   `build_benchmark_table.py` expects.
4. No dashboard touched — smoke gate (`python -m collab_splats.dashboard --smoke`) not
   applicable to this pass.

## Risks

- **Missed import in a deleted eval script.** Mitigated by the pre-deletion grep check
  plus a full suite re-run after Section 3.
- **CLI flag generalization changes default behavior.** Mitigated by keeping defaults
  identical to the current hardcoded values (e.g. `run_cross_model_benchmark.py`'s
  default `--backbones` = the current 4-backbone list) so an unparameterized invocation
  reproduces today's exact behavior.
- **Comment pass introduces incorrect claims about behavior.** Mitigated by writing
  comments only for logic already read and verified during this design's exploration
  (Sections 1-2 above), not speculative.
