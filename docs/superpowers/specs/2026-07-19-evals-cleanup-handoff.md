# Handoff: Evals + Loop-Closure Aggressive Cleanup — Discussion In Progress

- **Status:** DISCUSSION OPEN — do not implement yet. Decisions below are unresolved.
- **Date:** 2026-07-19
- **Purpose:** Give a fresh agent the full research + goal + the live design debate so the user can continue deciding without re-deriving context.

---

## 0. Read these first (source of truth)

- **Approved (but now being superseded) spec:** `docs/superpowers/specs/2026-07-19-evals-loop-closure-cleanup-design.md` — the conservative 7-section plan. The user is now steering MORE aggressive than this spec, so treat it as background, not gospel.
- **Draft plan (WRONG BRANCH — see §5):** `docs/superpowers/plans/2026-07-19-evals-loop-closure-cleanup.md`, committed `99fc801` on `refactor/cu121-uv-migration`. Written before the user's aggressive-consolidation direction; parts are now stale (it kept all 3 sweep runners). Salvage the Section 5/6/7 loop-closure tasks; the evals-runner sections need a rewrite.
- **Older, mostly-committed plan (different scope):** `docs/superpowers/plans/2026-07-19-lc-cleanup-eval-consolidation.md` — docstrings + runner CLI-generalization + 12-file deletion. Already landed (git log).
- **Memory:** `project_lc_parity_harness` (LC parity COMPLETE 2026-07-10), `project_lc_layer_calibration`, `project_geometry_package`, `project_branch_strategy`.

---

## 1. Original goal (user's words, brainstorming)

> "Aggressively clean and minimize the evals suite... reduce the overall number of lines of code and consolidating functions. Always reproduce the test suite while reducing LOC. Look through loop-closure evals + loop-closure codebase for overengineering (e.g. `_assemble_precorrection_extrinsics` used once). Consider how vggt-slam makes itself modular."

Two fronts:
1. **`evals/`** — collapse the script zoo, minimal + logical scripts.
2. **`collab_splats/geometry/loop_closure/`** — remove dead code, split the 894-line `closure.py` along VGGT-SLAM's `loop_closure.py`/`graph.py`/`map.py` seams, rename the per-window loop to mirror `Solver`.

**Hard constraint (repeated):** `pytest tests/` stays green throughout. "Always reproduce the test suite" = suite still passes (deleting a script deletes its test — allowed; moving a test requires proving the new test covers the old assertions FIRST).

---

## 2. Research findings (verified this session — trust these)

### 2a. Loop-closure side (spec §5–§7 — LESS controversial, largely settled)

- **`_assemble_precorrection_extrinsics`** (`wrapper.py:54-65`): single caller (`wrapper.py:385`), pure passthrough to `dedup_overlap`. Inline + delete.
- **`normalize_to_sl4`** (`graph.py:71-80`): zero prod callers, dead upstream too. `PoseGraph.add_node` already SL4-normalizes on insert (`gtsam.SL4(H)`). Delete; test callers (`test_graph.py`, `tests/pointcloud/test_pose_extraction.py:154`) inline the trivial det-normalize or pass raw.
- **`manifold="se3"`** (`PoseGraph.__init__` + `run_pose_graph_optimization` param + `LoopClosureConfig.manifold` field): zero prod callers (`wrapper.py` never passes `manifold=`; config never plumbs it). Only `PoseGraph(manifold="se3")` in tests reaches it. Delete branch + `_pose3` helper + its tests.
- **`scale_method="none"` is NOT dead — KEEP.** `eval_gt.py --lc_scale_method` exposes it; `run_disparity_sweep.py` passes `--lc_scale_method none` unconditionally. (Spec's own correction. Earlier drafts wrongly bundled it with `manifold="se3"`.)
- **`closure.py` split** (894 lines → 3 files, mirrors VGGT-SLAM):
  - `matching.py` (new): `LoopMatch`, `LoopMatchQueue`, `find_loop_closures`, `translation_jump_check`.
  - `graph.py` (extend): `run_pose_graph_optimization` + scale/align helpers (`_estimate_scale_pairwise_dist`, `umeyama_se3`, `umeyama_sim3`, `_cam_local_points`, `_lc_anchor_scale`, `_loop_chain_relatives`, `_MIN_CONF_POINTS`, `_RNG`).
  - `merge.py` (new): `dedup_overlap`, `_resolve_frame_node`, `merge_submap_outputs`.
  - `LoopClosureConfig` → `wrapper.py`.
- **Import fallout of the split (grepped, exhaustive):**
  - `.matching`: `test_loop_closure.py:156`, `test_translation_jump.py:8`.
  - `.merge`: `test_alignment_dedup.py:3`, `test_loop_closure_eval.py`.
  - `.graph`: `test_closure_split.py`, `test_hw_formula.py`, `test_loop_ablation.py`, `test_graph.py:155` (collapse 2nd import), `test_loop_edge_chain.py:22` + `:188` (`_SL4PoseGraph` alias — re-alias `PoseGraph` locally) + logger-name assertion `~:355` (`...closure` → `...graph`), `test_pgo_parity.py:16`+`:20`.
  - **`collab_splats/geometry/loop_closure/eval.py:14`** `from .closure import umeyama_se3, umeyama_sim3` → `.graph`. **← spec §6 enumeration MISSED this package-internal importer. Critical.**
  - **Mock patches** in `test_feedforward_lc_state.py:54-55` patch `"...closure.find_loop_closures"` etc. by string path — break at patch time once `closure.py` is gone. Move to patching `wrapper.<name>` (the pattern the file's other patches already use), NOT a mechanical `.closure`→`.matching` rename.
  - **Lazy `__getattr__` (BOTH files):** `LoopClosureConfig` currently eager-imported in `loop_closure/__init__.py` AND `geometry/__init__.py`. Once it lives in `wrapper.py` (which pulls in `pointcloud` → cycles through `geometry.transforms`), those eager imports deadlock. Fix: add `LoopClosureConfig` branch to BOTH `__getattr__` hooks (mirror the existing `LoopClosure` case); drop from both eager import lines; keep in both `__all__`.
  - **Deferred, leave alone:** `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb` imports from `.closure` — out of scope per CLAUDE.md in-flight list.
- **`_run_lc_loop` rename** (§7, VGGT-SLAM `Solver` parity):
  - `run_predictions(window, submaps, ...)` → forward pass + build Submap + retrieval + verify + jump-check. Returns `(submap, lc_submaps: list[Submap], loop_matches: list[LoopMatch])`. **NOT a singular `lc_submap_or_None`** — `max_loops_per_submap` defaults 5, multiple accepted matches per window is the common case. `loop_matches` (all candidates, accepted+rejected) is needed so caller drives `all_loop_candidates` + pbar counters.
  - `add_points(...)` → `submaps.append` + `lc_submaps.extend` (bookkeeping only; PGO deferred to the batch call, unlike `Solver.add_points`).
  - Outer sweep stays a private method (pbar, `end >= N` break) — NOT exposed. Exactly one calling convention (BaseFeedforwardCreator template method).

### 2b. Evals side (the CONTESTED part)

- **Genuine library code buried in `evals/`:**
  - `trajectory_io._invert_se3` duplicates `geometry.transforms.invert_poses` (strict superset — arbitrary batch dims). Delete `_invert_se3`, swap 4 call sites. Keep `_check_poses`. **`trajectory_io.py` itself STAYS in `evals/`** (spec §2 correction: zero consumers outside `evals/`, so moving it is abstraction-without-a-use-case).
  - `reconstruction_quality.py` → `collab_splats/geometry/loop_closure/diagnostics.py` (reads private `_lc_*` attrs off `LoopClosure.base` — cohesion with the class it inspects). Update `eval_gt.py:405` inline import.
  - `ate_utils.py` dissolves: `_load_gt_as_tum_trajectory` → `datasets.py`; `compute_ate_rmse` duplicates `metrics.compute_ate` (two evo entry points) — its one caller (`run_vggt_slam_lc.py:179`) writes GT via `write_tum` + calls `metrics.compute_ate`. Kills the `importlib.spec_from_file_location` name-collision workaround at `run_vggt_slam_lc.py:51`.
- **The 3 sweep drivers TRIPLICATE one shape** (verified): build subprocess argv matrix → serial `subprocess.run` → `--dry_run` → skip-if-`metrics.json`-exists (resume).
  - `run_cross_model_benchmark.py` (`RunSpec`, `build_commands`, `build_matrix`) — backbones × framesets → `eval_gt`. Frozen output: `baselines/cross_model/` (2026-05-31). Test: `test_cross_model_runner.py`.
  - `run_disparity_sweep.py` — runs **SLAM baseline AND ours**, compares ATE within tolerance (inline pass/fail). Frozen: `baselines/disparity_sweep/`. **No test.**
  - `run_lc_parity.py` — Level-0/1 gate checks + scaling prefixes, drives `run_vggt_slam_lc.py` + `eval_gt`. Test: `test_run_lc_parity.py`. Uses `lc_parity_common.py` (SceneSpec registry, gates, `_serialize_lc_decisions`).
- **`eval_gt.py` ALREADY is the "one model + params → metrics" runner.** Args: `--backbone --conditions(list) --submap_size --lc_scale_method --keyframe_list --lc_layer --output_ate`. Runs each condition in its own subprocess (clean GPU). The 3 sweeps just loop it over axes.
- **`eval_gt.py` DEPENDS on `lc_parity_common`:** `from runners.lc_parity_common import _serialize_lc_decisions`. So `lc_parity_common.py` cannot fully die — keep `_serialize_lc_decisions`.
- **Analysis vs driving split:** the sweep loop is shared; the *analysis* differs (cross-model table, disparity SLAM-vs-ours compare, parity gates). Analysis lives in `build_benchmark_table.py`, `build_parity_table.py`, `lc_loop_pr.py`, `lc_parity_common.py`, `compare_loop_edges.py`, `visualize_lc_correction.py`.
- **Downloads:** 6 scripts, `download_7scenes.py`+`.sh` are dup; `data/download_parity_scenes.sh` re-hardcodes the same URLs a 3rd time. Consolidate → one `data/download_datasets.py` (function per dataset, subcommand CLI). `.gitignore:29` `!evals/data/download_parity_scenes.sh` must become `!evals/data/download_datasets.py` (+ `extract_waymo.py`) or new files are swallowed by `evals/data/*` ignore.
- **Dir reorg (spec §4):** `evals/runners/` → `evals/scripts/`, fold top-level `eval_*.py` in. `eval_suite.sh` `REPO_ROOT` goes up one dir today → up two after move. `CLAUDE.md` has 2 literal `evals/eval_gt.py` refs. ~10 test files do `sys.path.insert(.../"evals")` + `from runners.X` — need repointing.

---

## 3. Goal the user CONVERGED on (later in discussion — supersedes spec §3/§4)

Verbatim intent:
- "**One `eval.py` script** that allows specification of datasets // parameters. Avoid overengineering (is SweepConfig necessary?)."
- "Our goal is to be able to **run loop closure on a given model with a set of parameters and compare across parameter sets.**"
- "Minimize the number of **driver scripts**. One should be able to drive loop-closure evaluations from a single script with specified parameters" (bash wrappers may call it for scaled/preset runs).
- "Allow repointing of tests **as long as tests first exhibit equivalence** (when building a new test, confirm it properly fills the place of the previous test)."
- "I don't understand what these three different sweeps achieve and why they're necessary **given that we have validated the code base.**"

**Interpretation:** `eval_gt.py` IS essentially the target — extend it (or rename → `eval.py`) so swept params accept lists and it loops the grid internally (serial, resume, dry_run) + aggregates a comparison. The 3 bespoke sweep drivers were validation-era scaffolding whose results are frozen in `baselines/` + findings specs; the user is questioning whether they should survive at all.

---

## 4. OPEN DECISIONS (what the fresh discussion must resolve)

**D1 — Retire the validation sweep drivers?** Candidates: `run_cross_model_benchmark.py`, `run_disparity_sweep.py`, `run_lc_parity.py`, `build_benchmark_table.py`, `build_parity_table.py`, `lc_loop_pr.py`, `compare_loop_edges.py`, `visualize_lc_correction.py` (+ their tests).
  - Option A: retire all (validation done, baselines + specs are the evidence). Maximal cleanup.
  - Option B: retire the 3 drivers, keep table/viz as re-runnable analysis on existing baselines.
  - Option C: decide per-file.
  - **Risk to weigh:** `run_disparity_sweep.py` is the only ours-vs-VGGT-SLAM comparison harness. Retiring it drops that capability (re-derivable, but not for free). `run_vggt_slam.py`/`run_vggt_slam_lc.py` (the SLAM wrappers themselves) should be KEPT regardless — referenced by `eval_suite.sh` + `setup/vggt_slam.sh`, external baseline.

**D2 — Merge `eval_compare.py` into `eval.py`, or keep separate?** eval_compare ingests a results dir → multi-method `metrics.json`. Folding it makes `eval.py` the single run+compare entry; keeping it is less churn. Either way `test_eval_compare.py` needs equivalence-checked repointing if merged.

**D3 — `eval.py` internal sweep shape (avoid overengineering).** User asked "is SweepConfig necessary?" — recommendation: NO. Use argparse `nargs="+"` on the sweepable params + `itertools.product` + the existing per-condition subprocess loop + resume-on-`metrics.json`. No config dataclass. Confirm this is the wanted shape.

**D4 — Standalone tools' fate:** `eval_multiview_conf.py`, `eval_similarity_calibration.py` are re-runnable calibration/analysis tools (produced the per-model LC-layer calibration), not sweep drivers. Keep? (Recommend keep — distinct capability, not duplication.)

**D5 — `lc_parity_common.py` reduction:** keep only `_serialize_lc_decisions` (used by `eval_gt`/`eval.py`); drop `SceneSpec`/gate machinery IF only the retired parity sweep used them. Verify no other consumer before trimming.

---

## 5. Branch / worktree state — ACTION NEEDED

- Current shell: `/workspace/collab-splats`, **main checkout**, branch `refactor/cu121-uv-migration`. The draft plan `99fc801` landed HERE — **wrong place.**
- **Dedicated worktree exists:** `.worktrees/lc-cleanup-eval-consolidation` on branch `lc-cleanup-eval-consolidation` (clean). It already holds the spec `2026-07-19-evals-loop-closure-cleanup-design.md` + the older consolidation plan. It does NOT have the new plan.
- **Do:** execute this cleanup in the worktree/branch `lc-cleanup-eval-consolidation`. Move the revised plan there; drop/revert the plan commit from `refactor/cu121-uv-migration` (or leave it, but don't double-maintain).
- Other live worktrees (untouched): `docs-site`, a detached `wt-19fa95c`.

---

## 6. Environment / house rules (for the executing agent)

- Python: `/opt/venv/reconstruction/bin/python` (py3.11). NOT base `python`.
- Test: `/opt/venv/reconstruction/bin/python -m pytest tests/`. Baseline pass count vs `docs/known-test-failures.md`.
- Format before every commit: `black . && isort .`.
- Commit style: `refactor(evals):`, `refactor(geometry):`, `docs(specs):`. Co-author trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- `docs/superpowers/**` may need `git add -f` (gitignored path).
- Land as separable commits in spec order (dead files → lib extraction → eval.py consolidation → dir reorg → LC dead code → closure split → wrapper rename) so regressions bisect.
- **Tooling:** use graphify + context-mode tools for code exploration — they improve performance. The user wants them used. Honor caveman-terse prose in chat.

---

## 7. Suggested first move for the fresh agent

1. Re-read §3 (converged goal) + §4 (open decisions) with the user; resolve D1–D5.
2. Fix branch (§5): switch to the worktree, relocate the plan.
3. Rewrite the plan: keep the loop-closure tasks (§2a — near-final), replace the evals-runner sections with the agreed `eval.py` consolidation.
4. Then execute per plan, suite-green per commit.
