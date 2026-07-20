# Evals + Loop-Closure Aggressive Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the `evals/` script zoo into one config-driven `eval.py`, retire validation-era scaffolding, and slim `collab_splats/geometry/loop_closure/` to a `Solver`-shaped core — while keeping `pytest tests/` green at every commit.

**Architecture:** Spec `docs/superpowers/specs/2026-07-19-evals-loop-closure-cleanup-design.md` is authoritative. Work happens in worktree `.worktrees/evals-lc-cleanup` (branch `evals-lc-cleanup`); merge into `refactor/cu121-uv-migration` at the end. Tasks land in spec §11 commit order (LC-internal first, eval.py capability next, retirements after, reorg last) so nothing references a deleted file mid-sequence and regressions bisect cleanly.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest, evo (trajectory metrics), PyYAML 6.0.3, gtsam (SL4/PGO), numpy, torch.

---

## Conventions for every task

- **Interpreter:** `/opt/venv/reconstruction/bin/python` (aliased `PY` below). Never base `python`.
- **Test command:** `PY=/opt/venv/reconstruction/bin/python; $PY -m pytest <targets> -q`.
- **Format before every commit:** `black . && isort .` (from worktree root).
- **Baseline (green reference):** `known-test-failures.md` records 964 passed on the `-m 'not slow'` subset + 3 xfail; migration gate `pytest tests/test_cu121_migration.py` = 24/24. Full collection = 1218 tests. **Do not let a commit reduce pass count except by removing a retired script's own test.**
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **`docs/superpowers/**` is gitignored** — `git add -f` those paths.
- **Equivalence-first rule (spec §10):** to move/repoint a test, first make a new test asserting the same behavior, run both green in one commit, *then* delete the old. Deleting a script deletes its covering test outright (allowed).
- **All file paths below are relative to the worktree root** `/workspace/collab-splats/.worktrees/evals-lc-cleanup`.

Before starting, confirm the working tree is clean and on the right branch:

```bash
cd /workspace/collab-splats/.worktrees/evals-lc-cleanup
git branch --show-current   # -> evals-lc-cleanup
git status --porcelain      # -> empty
/opt/venv/reconstruction/bin/python -m pytest tests/ --co -q 2>/dev/null | tail -1   # -> 1218 tests collected
```

---

## Task 1: LC dead code removal (spec §9a)

Remove three zero-prod-caller code paths. `scale_method="none"` STAYS (exposed via `eval_gt.py --lc_scale_method`).

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py` (delete `_assemble_precorrection_extrinsics` def + its single call site)
- Modify: `collab_splats/geometry/loop_closure/graph.py` (delete `normalize_to_sl4`; delete `manifold="se3"` branch in `PoseGraph.__init__` + `run_pose_graph_optimization` + the `_pose3` helper)
- Modify: `collab_splats/geometry/loop_closure/closure.py` (delete `LoopClosureConfig.manifold` field)
- Test: `tests/geometry/loop_closure/test_graph.py`, `tests/pointcloud/test_pose_extraction.py` (inline the trivial det-normalize where they called `normalize_to_sl4`); delete `manifold="se3"`-specific test cases.

- [ ] **Step 1: Locate the exact call sites**

```bash
grep -n '_assemble_precorrection_extrinsics\|_lc_precorrection_extrinsics' collab_splats/geometry/loop_closure/wrapper.py
grep -n 'normalize_to_sl4\|manifold\|_pose3' collab_splats/geometry/loop_closure/graph.py collab_splats/geometry/loop_closure/closure.py
grep -rn 'normalize_to_sl4\|manifold=' tests/
```

- [ ] **Step 2: Delete `_assemble_precorrection_extrinsics` + its call**

In `wrapper.py`: delete the `def _assemble_precorrection_extrinsics(...)` function (currently ~lines 54-65) AND the line `self.base._lc_precorrection_extrinsics = _assemble_precorrection_extrinsics(submaps, N)` in `_run_lc_loop`. (The `_lc_precorrection_extrinsics` attribute has no surviving consumer — confirmed: only `reconstruction_quality.py`, which retires in Task 2.)

- [ ] **Step 3: Delete `normalize_to_sl4`, `manifold="se3"`, `_pose3`**

In `graph.py`: delete `def normalize_to_sl4(...)`. In `PoseGraph.__init__`, drop the `manifold` parameter and the `se3`/`_pose3` branch (keep the default SL4 path). In `run_pose_graph_optimization`, drop the `manifold` param. Delete the `_pose3` helper. In `closure.py`, delete the `manifold` field from `LoopClosureConfig`.

- [ ] **Step 4: Fix test callers (inline the trivial normalize)**

`normalize_to_sl4(H)` divided `H` by `det(H)**(1/4)`. Where tests called it, replace with the inline `H / np.linalg.det(H) ** 0.25` (or pass the raw matrix — `PoseGraph.add_node` already `gtsam.SL4(H)`-normalizes on insert, so raw is fine). Delete any test parametrization that passed `manifold="se3"`.

- [ ] **Step 5: Run the LC test subset**

Run: `$PY -m pytest tests/geometry/loop_closure/ tests/pointcloud/test_pose_extraction.py -q`
Expected: PASS (no `normalize_to_sl4`/`manifold` references remain).

- [ ] **Step 6: Full suite sanity + commit**

Run: `$PY -m pytest tests/ -q -x` (must not drop below baseline)

```bash
black . && isort .
git add -A
git commit -m "refactor(geometry): drop dead LC code — normalize_to_sl4, manifold=se3, precorrection stitch

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: LoopClosure de-boilerplate + diagnostic-side-channel removal (spec §9d)

Make `LoopClosure` `Solver`-shaped: expose only corrected `raw_outputs` + a scalar `n_loops_applied`. Delete all 7 `_lc_*` writes, `_ablate_loops`, `reconstruction_quality.py`, and the `eval_gt` diagnostic writers. Replace forwarding stubs with `__getattr__`. **Do this before the closure split (Task 4) so the split works on lean code.**

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py`
- Modify: `collab_splats/pointcloud/feedforward/base.py` (drop the `_lc_*` attribute docstring block; KEEP `_lc_layer_index`/`_lc_token_offset` ClassVars + `_lc_collate_outputs` — those feed the live loop, not the diagnostic channel)
- Modify: `evals/eval_gt.py` (drop `_write_loop_ablation`, `compute_alignment_metrics` block, `lc_decisions.json` writing; add `n_loops_applied` passthrough)
- Delete: `evals/reconstruction_quality.py`
- Delete/trim tests: `tests/geometry/loop_closure/test_loop_ablation.py`, `tests/evals/test_loop_ablation_json.py`, `tests/geometry/loop_closure/test_feedforward_lc_state.py`, `tests/pointcloud/feedforward/test_verify_lc_data.py`

- [ ] **Step 1: Map every reader before cutting**

```bash
for a in _lc_submaps _lc_loop_submaps _lc_overlap_frames _lc_all_matches _lc_precorrection_extrinsics _lc_corrected_extrinsics _lc_ablation_extrinsics; do echo "== $a =="; grep -rln "$a" --include=*.py collab_splats/ evals/ tests/; done
grep -rn '_ablate_loops\|reconstruction_quality\|compute_alignment_metrics\|_write_loop_ablation\|lc_decisions' --include=*.py collab_splats/ evals/ tests/
```

Confirm BA/mesh touch only `raw_outputs`: `grep -n '_lc_\|raw_outputs' collab_splats/geometry/bundle_adjustment.py collab_splats/mesh/*.py`. If any `_lc_*` reader outside eval/tests appears, STOP and reconcile against the spec before deleting.

- [ ] **Step 2: Add `n_loops_applied`, delete the `_lc_*` writes + `_ablate_loops`**

In `wrapper.py` `_run_lc_loop`, replace the block that wrote the seven `self.base._lc_*` attributes with a single scalar:

```python
# Number of accepted loop-closure submaps applied (user-visible summary).
self.base.n_loops_applied = len(lc_submaps)
```

Delete the `_ablate_loops` method entirely and its call `if lc_submaps: self._ablate_loops(...)`.

- [ ] **Step 3: Replace forwarding stubs with `__getattr__`**

In `wrapper.py`, delete the pure pass-through methods `load_model`, `setup_inference`, `postprocess`, `build_colmap`, `_reproject`, `reproject`. Add one delegator (place right after `__init__`):

```python
def __getattr__(self, name: str) -> Any:
    # Delegate any attribute not defined on the wrapper to the wrapped creator.
    # __getattr__ only fires for missing names, so the explicit overrides below
    # (run_inference, run, reconstruct, outputs/raw_outputs properties) still win.
    return getattr(self.base, name)
```

Keep the explicit overrides: `run_inference`, `run`, `reconstruct`, and the `outputs`/`raw_outputs` properties (they intercept get/set). `self.base`/`self.config` are set in `__init__` and are real attributes, so `__getattr__` never shadows them.

- [ ] **Step 4: Strip the diagnostic writers from `eval_gt.py`**

Delete `_write_loop_ablation` (def + its call ~line 565), the `compute_alignment_metrics` import + block (~lines 403-411), the `_serialize_lc_decisions` import + `_lc_all_matches` block (~lines 413-425), the `lc_decisions_{cond}.json` write (~lines 221-223), and `metrics[cond]["lc_decisions"]` (~line 562). Replace with the scalar read:

```python
# Loop-closure summary (only field LC now exposes).
if hasattr(creator, "base") and hasattr(creator.base, "n_loops_applied"):
    metrics[cond]["n_loops_applied"] = int(creator.base.n_loops_applied)
```

- [ ] **Step 5: Delete `reconstruction_quality.py` + retire its dependent tests**

```bash
git rm evals/reconstruction_quality.py
git rm tests/geometry/loop_closure/test_loop_ablation.py tests/evals/test_loop_ablation_json.py
```

For `test_feedforward_lc_state.py` and `test_verify_lc_data.py`: they asserted the removed `_lc_*` attributes. If a file's *only* purpose was the side-channel, `git rm` it; if it also covers live loop behavior (e.g. loop-detection counts), rewrite those assertions against `n_loops_applied` (equivalence-first — the new assertion must exercise the same loop-detection path). Inspect each:

```bash
$PY -m pytest tests/geometry/loop_closure/test_feedforward_lc_state.py tests/pointcloud/feedforward/test_verify_lc_data.py -q
```

- [ ] **Step 6: Run LC + eval test subsets**

Run: `$PY -m pytest tests/geometry/loop_closure/ tests/pointcloud/feedforward/ tests/evals/ -q`
Expected: PASS (retired tests gone; survivors green).

- [ ] **Step 7: Full suite + commit**

Run: `$PY -m pytest tests/ -q`

```bash
black . && isort .
git add -A
git commit -m "refactor(geometry): slim LoopClosure to Solver shape — drop _lc_* side-channel, __getattr__ delegation

Removes _ablate_loops, all 7 _lc_* attrs, reconstruction_quality, and the
eval_gt diagnostic writers (lc_decisions/loop_ablation JSON). LC now exposes
only corrected raw_outputs + scalar n_loops_applied. Forwarding stubs replaced
by one __getattr__ delegator.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Library extraction — `_invert_se3` + `compare_loop_edges` helpers (spec §6)

**Files:**
- Modify: `evals/trajectory_io.py` (delete `_invert_se3`, swap 4 call sites to `invert_poses`)
- Create: `collab_splats/geometry/loop_closure/edge_trace.py` (move `compose_slam_chain`, `edge_divergence`)
- Delete: `evals/runners/compare_loop_edges.py`
- Modify: `tests/geometry/loop_closure/test_loop_edge_chain.py` (repoint import), `tests/evals/test_compare_loop_edges.py` (repoint or retire)

- [ ] **Step 1: Swap `_invert_se3` → `invert_poses`**

```bash
grep -n '_invert_se3' evals/trajectory_io.py
```

`geometry.transforms.invert_poses` is a strict superset (arbitrary batch dims). Add `from collab_splats.geometry.transforms import invert_poses` at the top of `trajectory_io.py`, replace each `_invert_se3(x)` with `invert_poses(x)`, delete the `_invert_se3` def. Keep `_check_poses`.

- [ ] **Step 2: Run trajectory_io tests**

Run: `$PY -m pytest tests/evals/ -k 'trajectory or tum' -q`
Expected: PASS.

- [ ] **Step 3: Move edge-trace helpers into the LC package**

Create `collab_splats/geometry/loop_closure/edge_trace.py` with the two functions verbatim from `compare_loop_edges.py` (`compose_slam_chain`, `edge_divergence`) plus their module docstring. They encode VGGT-SLAM's 3-edge chain composition — cohesive with the LC package.

- [ ] **Step 4: Repoint the geometry test (equivalence-first)**

In `tests/geometry/loop_closure/test_loop_edge_chain.py:28`, change `from evals.runners.compare_loop_edges import compose_slam_chain` → `from collab_splats.geometry.loop_closure.edge_trace import compose_slam_chain`.

Run: `$PY -m pytest tests/geometry/loop_closure/test_loop_edge_chain.py -q` → PASS.

- [ ] **Step 5: Retire the evals copy + its test**

`tests/evals/test_compare_loop_edges.py` tested the same two functions. Either repoint it to `edge_trace` (equivalence — same assertions, run green) or `git rm` it since `test_loop_edge_chain.py` now covers the functions. Then:

```bash
git rm evals/runners/compare_loop_edges.py
```

- [ ] **Step 6: Full suite + commit**

Run: `$PY -m pytest tests/ -q`

```bash
black . && isort .
git add -A
git commit -m "refactor(geometry): move loop-edge trace helpers into LC package; use invert_poses in trajectory_io

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Split `closure.py` → matching / graph / merge (spec §9b)

Mechanical split of the 894-line `closure.py` into three files mirroring VGGT-SLAM, plus `LoopClosureConfig` → `wrapper.py`, plus the exhaustive import fallout. **No behavior change** — pure move + re-import.

**Files:**
- Create: `collab_splats/geometry/loop_closure/matching.py`, `collab_splats/geometry/loop_closure/merge.py`
- Modify: `collab_splats/geometry/loop_closure/graph.py` (extend), `wrapper.py` (host `LoopClosureConfig`), `collab_splats/geometry/loop_closure/__init__.py`, `collab_splats/geometry/__init__.py`, `collab_splats/geometry/loop_closure/eval.py`, and the test files listed below.
- Delete: `collab_splats/geometry/loop_closure/closure.py`

- [ ] **Step 1: Carve the three symbol groups**

- `matching.py`: `LoopMatch`, `LoopMatchQueue`, `find_loop_closures`, `translation_jump_check` (+ their imports).
- `graph.py` (append): `run_pose_graph_optimization`, `_estimate_scale_pairwise_dist`, `umeyama_se3`, `umeyama_sim3`, `_cam_local_points`, `_lc_anchor_scale`, `_loop_chain_relatives`, `_MIN_CONF_POINTS`, `_RNG`.
- `merge.py`: `dedup_overlap`, `_resolve_frame_node`, `merge_submap_outputs`.
- `LoopClosureConfig` → `wrapper.py` (it configures the wrapper).

Move each symbol with its imports; run `black`/`isort` after to catch unused imports (prune manually).

- [ ] **Step 2: Fix the package-internal importer**

`collab_splats/geometry/loop_closure/eval.py:14`: `from .closure import umeyama_se3, umeyama_sim3` → `from .graph import umeyama_se3, umeyama_sim3`.

- [ ] **Step 3: Fix both lazy `__getattr__` hooks**

`LoopClosureConfig` now lives in `wrapper.py` (which imports `pointcloud` → cycles through `geometry.transforms`). Eager-importing it deadlocks. In BOTH `collab_splats/geometry/loop_closure/__init__.py` AND `collab_splats/geometry/__init__.py`: drop `LoopClosureConfig` from the eager import line, add a branch to the existing `__getattr__` (mirror the `LoopClosure` case), keep it in `__all__`:

```python
def __getattr__(name):
    if name == "LoopClosure":
        from .loop_closure.wrapper import LoopClosure
        return LoopClosure
    if name == "LoopClosureConfig":
        from .loop_closure.wrapper import LoopClosureConfig
        return LoopClosureConfig
    raise AttributeError(name)
```

(Adjust the relative import path to each `__init__`'s location.)

- [ ] **Step 4: Repoint test imports (exhaustive list from spec §9b)**

- `.matching`: `test_loop_closure.py:156`, `test_translation_jump.py:8`.
- `.merge`: `test_alignment_dedup.py:3`, `test_loop_closure_eval.py`.
- `.graph`: `test_closure_split.py`, `test_hw_formula.py`, `test_graph.py:155` (collapse duplicate import), `test_loop_edge_chain.py:22`+`:188` (re-alias `PoseGraph` locally where `_SL4PoseGraph` was used) + logger-name assertion (`...closure` → `...graph`), `test_pgo_parity.py:16`+`:20`.
- Mock patches in `test_feedforward_lc_state.py`: patch `wrapper.<name>` (string path) instead of `...closure.<name>`. (This file may already be trimmed/removed in Task 2 — if it survives, fix here.)
- `test_loop_ablation.py` — already removed in Task 2; skip.

```bash
grep -rn 'from .*\.closure import\|loop_closure\.closure' --include=*.py collab_splats/ tests/ evals/
```
Every hit outside the retiring tutorial must be repointed. The tutorial `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb` is OUT OF SCOPE — leave it.

- [ ] **Step 5: Delete `closure.py`, verify no dangling imports**

```bash
git rm collab_splats/geometry/loop_closure/closure.py
grep -rn '\.closure import\|loop_closure\.closure' --include=*.py collab_splats/ tests/ evals/ | grep -v '\.ipynb'
```
Expected: no output (all repointed).

- [ ] **Step 6: Import smoke + LC suite + commit**

```bash
$PY -c "from collab_splats.geometry import LoopClosure, LoopClosureConfig; from collab_splats.geometry.loop_closure import matching, graph, merge; print('ok')"
$PY -m pytest tests/geometry/loop_closure/ tests/pointcloud/ -q
```
Expected: `ok` then PASS.

```bash
black . && isort .
git add -A
git commit -m "refactor(geometry): split closure.py into matching/graph/merge; LoopClosureConfig -> wrapper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Split `_run_lc_loop` → `run_predictions`/`add_points` + thin driver (spec §9c)

Restructure the ~180-line monolith into two named per-window steps + a thin loop driver, mirroring VGGT-SLAM `main.py`. **Behavior-preserving** — the merged `raw_outputs` and `n_loops_applied` must be identical.

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py`
- Test: existing `tests/geometry/loop_closure/` loop tests are the equivalence guard.

- [ ] **Step 1: Capture the current behavior as the guard**

Run the existing LC loop tests and note pass count:
`$PY -m pytest tests/geometry/loop_closure/ -q` → record N passed.

- [ ] **Step 2: Extract `run_predictions`**

Pull the per-window body (forward pass + build Submap + retrieval + verify + jump-check) into:

```python
def run_predictions(self, window, **kwargs):
    """Forward-pass one window, build its Submap, detect + verify loop candidates.

    Returns (submap, lc_submaps, loop_matches): the window's Submap, the list of
    verified loop-closure submaps (each 2 frames), and all post-NMS candidates
    (accepted + rejected) for the caller's counters. Not a singular match —
    max_loops_per_submap defaults to 5.
    """
    ...
    return submap, lc_submaps, loop_matches
```

- [ ] **Step 3: Extract `add_points`**

```python
def add_points(self, submap, lc_submaps, submaps, lc_submaps_acc):
    """Append the window's submap + verified loop submaps to the driver's lists.

    Bookkeeping only — PGO is deferred to the batch call after the sweep (unlike
    VGGT-SLAM's Solver.add_points which optimizes incrementally).
    """
    submaps.append(submap)
    lc_submaps_acc.extend(lc_submaps)
```

- [ ] **Step 4: Thin driver keeps state local**

`_run_lc_loop` becomes the driver: owns local `submaps = []`, `lc_submaps = []`, runs the `step`/`end >= N` sweep calling `run_predictions` → `add_points`, then batch `run_pose_graph_optimization` + `merge_submap_outputs`, then sets `self.base.raw_outputs` + `self.base.n_loops_applied`. No `self.base._lc_*` writes.

- [ ] **Step 5: Verify equivalence**

Run: `$PY -m pytest tests/geometry/loop_closure/ -q`
Expected: same N passed as Step 1. If a test asserted intermediate state that no longer exists, it was a side-channel test that should have died in Task 2 — reconcile.

- [ ] **Step 6: Full suite + commit**

Run: `$PY -m pytest tests/ -q`

```bash
black . && isort .
git add -A
git commit -m "refactor(geometry): split _run_lc_loop into run_predictions/add_points + thin driver (Solver parity)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `eval.py` config-driven runner + `eval_compare` module (spec §1, §2)

Rename `eval_gt.py` → `eval.py`; add YAML-config grid + resume + aggregation. Fold `eval_compare.py` into an imported module. **Keep the single-cell CLI path byte-compatible** — existing `test_eval_gt*` tests exercise the per-condition subprocess leaf; only *add* the `--config` grid wrapper around it.

**Files:**
- Rename: `evals/eval_gt.py` → `evals/eval.py` (git mv; the `runners/`→`scripts/` move is Task 11)
- Modify: `evals/eval_compare.py` (expose `scan_results_dir`, `format_markdown`, add `collect_grid_metrics`/`format_markdown_rows`)
- Create: `evals/configs/7scenes.yaml`, `evals/configs/cross_model_chess.yaml`
- Test: `tests/evals/test_eval_config.py` (new), repoint `tests/evals/test_eval_compare.py`, `tests/evals/test_eval_gt*.py`

- [ ] **Step 1: git mv + update self-references**

```bash
git mv evals/eval_gt.py evals/eval.py
grep -rn 'eval_gt' evals/ tests/ CLAUDE.md
```
Update in-file docstrings/usage strings `eval_gt.py` → `eval.py`. (Test import repoints happen below + Task 11.)

- [ ] **Step 2: Write the failing config-grid test**

Create `tests/evals/test_eval_config.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
from eval import load_eval_config, build_grid   # noqa: E402


def test_load_config_flat(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(
        "name: t\n"
        "datasets:\n  - {name: 7scenes, seq_dir: /d/seq, keyframe_list: null}\n"
        "backbones: [vggt_omega, vggt_spark]\n"
        "conditions: [baseline, lc]\n"
        "submap_size: 50\n"
        "max_frames: 200\n"
        "output_dir: /out\n"
    )
    cfg = load_eval_config(cfg_file)
    assert cfg.name == "t"
    assert cfg.backbones == ["vggt_omega", "vggt_spark"]
    assert cfg.submap_size == 50


def test_build_grid_product(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(
        "name: t\n"
        "datasets:\n  - {name: 7scenes, seq_dir: /d/seq, keyframe_list: null}\n"
        "backbones: [vggt_omega, vggt_spark]\n"
        "conditions: [baseline, lc]\n"
        "output_dir: /out\n"
    )
    cfg = load_eval_config(cfg_file)
    grid = build_grid(cfg)
    assert len(grid) == 4                       # 1 dataset x 2 backbones x 2 conditions
    assert {c.backbone for c in grid} == {"vggt_omega", "vggt_spark"}
```

Run: `$PY -m pytest tests/evals/test_eval_config.py -q` → FAIL (`load_eval_config` undefined).

- [ ] **Step 3: Implement the flat config + grid**

In `evals/eval.py` add (no `SweepConfig` hierarchy — one flat dataclass):

```python
import itertools
from dataclasses import dataclass
import yaml

@dataclass
class EvalCell:
    """One grid cell: a backbone × condition over one dataset at fixed params."""
    dataset_name: str
    seq_dir: Path
    keyframe_list: Path | None
    backbone: str
    condition: str
    submap_size: int | None
    max_frames: int | None
    lc_layer: int | None
    output_dir: Path

@dataclass
class EvalConfig:
    """Flat, declarative eval experiment loaded from YAML."""
    name: str
    datasets: list[dict]
    backbones: list[str]
    conditions: list[str]
    output_dir: Path
    submap_size: int | None = None
    max_frames: int | None = None
    lc_layer: int | None = None

def load_eval_config(path: Path) -> EvalConfig:
    """Parse a flat experiment YAML into an EvalConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    return EvalConfig(
        name=raw["name"], datasets=raw["datasets"],
        backbones=raw["backbones"], conditions=raw["conditions"],
        output_dir=Path(raw["output_dir"]),
        submap_size=raw.get("submap_size"), max_frames=raw.get("max_frames"),
        lc_layer=raw.get("lc_layer"),
    )

def build_grid(cfg: EvalConfig) -> list[EvalCell]:
    """Expand the config into one EvalCell per (dataset × backbone × condition)."""
    cells = []
    for ds, backbone in itertools.product(cfg.datasets, cfg.backbones):
        for cond in cfg.conditions:
            cells.append(EvalCell(
                dataset_name=ds["name"], seq_dir=Path(ds["seq_dir"]),
                keyframe_list=(Path(ds["keyframe_list"]) if ds.get("keyframe_list") else None),
                backbone=backbone, condition=cond,
                submap_size=cfg.submap_size, max_frames=cfg.max_frames,
                lc_layer=cfg.lc_layer, output_dir=cfg.output_dir,
            ))
    return cells
```

Run: `$PY -m pytest tests/evals/test_eval_config.py -q` → PASS.

- [ ] **Step 4: Wire `--config` into `main()` with resume**

In `eval.py` `main()`, add `--config` (mutually exclusive with the single-cell args). When set: `cfg = load_eval_config(args.config); grid = build_grid(cfg)`. Loop cells serially; per cell compute the output dir `output_dir/<dataset>__<backbone>__<cond>`; **skip if its `metrics.json` exists** (resume); honor `--dry_run` (print argv, run nothing). Each cell reuses the existing per-condition subprocess leaf (already how `eval_gt` runs a condition — do not modify the leaf). After the grid, call the aggregation (Step 6).

- [ ] **Step 5: Make `eval_compare` importable**

In `evals/eval_compare.py`: rename internal `_scan_results_dir`/`_format_markdown` to public `scan_results_dir`/`format_markdown` (keep thin `_`-aliases if a test needs them, equivalence-first), and add:

```python
def collect_grid_metrics(output_root: Path) -> list[dict]:
    """Read every <cell>/metrics.json under a grid output root into rows."""
    rows = []
    for mj in sorted(output_root.glob("*/metrics.json")):
        rows.append(json.loads(mj.read_text()) | {"_cell": mj.parent.name})
    return rows

def format_markdown_rows(rows: list[dict]) -> str:
    """Render grid rows (dataset/backbone/condition + ATE/RPE/AUC) as a markdown table."""
    header = ("| cell | ATE RMSE | RPE trans | RPE rot° | AUC@30 |\n"
              "|---|---|---|---|---|")
    lines = [header]
    for r in sorted(rows, key=lambda x: x.get("_cell", "")):
        ate = r.get("ate", {}).get("rmse", float("nan"))
        rpe_t = r.get("rpe", {}).get("trans_rmse", float("nan"))
        rpe_r = r.get("rpe", {}).get("rot_rmse_deg", float("nan"))
        auc = r.get("auc", {}).get("auc_30", float("nan"))
        lines.append(f"| {r.get('_cell','?')} | {ate:.4f} | {rpe_t:.4f} | {rpe_r:.4f} | {auc:.4f} |")
    return "\n".join(lines)
```

- [ ] **Step 6: Aggregate after the grid**

In `eval.py`, after the cell loop:

```python
from eval_compare import collect_grid_metrics, format_markdown_rows
rows = collect_grid_metrics(cfg.output_dir)
(cfg.output_dir / "comparison.md").write_text(format_markdown_rows(rows))
(cfg.output_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
```

- [ ] **Step 7: Repoint + equivalence-check `test_eval_compare.py`**

Point it at the public `scan_results_dir`/`format_markdown`. Add an assertion that `collect_grid_metrics` on a fixture of two `metrics.json` yields two rows and `format_markdown_rows` renders two data lines. Run: `$PY -m pytest tests/evals/test_eval_compare.py tests/evals/test_eval_config.py -q` → PASS.

- [ ] **Step 8: Author the two preset configs**

`evals/configs/7scenes.yaml` (chess/fire/office, 6 conditions — the retired `eval_suite.sh` set) and `evals/configs/cross_model_chess.yaml` (backbones × chess). Use the example in spec §1. Validate they load:

```bash
$PY -c "import sys; sys.path.insert(0,'evals'); from eval import load_eval_config, build_grid; [print(c.name, len(build_grid(c))) for c in [load_eval_config('evals/configs/7scenes.yaml'), load_eval_config('evals/configs/cross_model_chess.yaml')]]"
```

- [ ] **Step 9: Repoint remaining `eval_gt`→`eval` test imports + full suite + commit**

```bash
grep -rln 'from eval_gt\|import eval_gt\|eval_gt.py' tests/ evals/
```
Repoint each (equivalence — same tests, new module name). Run: `$PY -m pytest tests/ -q`.

```bash
black . && isort .
git add -A
git commit -m "feat(evals): config-driven eval.py (YAML grid, resume, aggregation); fold eval_compare into a module

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Merge the two VGGT-SLAM wrappers → one `run_vggt_slam.py` (spec §4)

`run_vggt_slam_lc.py` with `--max_loops 0` IS the no-LC baseline. Merge its TUM+ATE+`selected_frames.txt` capability into `run_vggt_slam.py`; LC via `--max_loops`. Equivalence-first.

**Files:**
- Modify: `evals/runners/run_vggt_slam.py` (absorb the LC path)
- Delete: `evals/runners/run_vggt_slam_lc.py`
- Modify: `setup/vggt_slam.sh` (example command), `evals/README.md`
- Test: merge `tests/evals/test_run_vggt_slam.py` + `test_run_vggt_slam_lc.py`

- [ ] **Step 1: Diff the two wrappers' arg surfaces + bodies**

```bash
diff <(grep -n 'add_argument\|def ' evals/runners/run_vggt_slam.py) <(grep -n 'add_argument\|def ' evals/runners/run_vggt_slam_lc.py)
```
Identify what `run_vggt_slam_lc.py` adds: `--max_loops`, `--min_disparity`, `--out_tum`, ATE computation, `selected_frames.txt` write.

- [ ] **Step 2: Absorb the LC path into `run_vggt_slam.py`**

Add the missing args (`--max_loops` default 0, `--min_disparity`, `--out_tum`) and the post-run steps (write TUM, compute ATE via `metrics.compute_ate`, write `selected_frames.txt`). Guard LC on `max_loops > 0`. Preserve `run_vggt_slam.py`'s existing image-prep (`_prepare_image_dir`) and `.pending` sentinel handling.

- [ ] **Step 3: Equivalence check both output modes**

Both modes shell to third-party VGGT-SLAM (heavy/GPU); the CLI-surface tests assert flags/paths, not a real run. Merge `test_run_vggt_slam_lc.py`'s assertions into `test_run_vggt_slam.py` (cover both `--max_loops 0` and `>0` surfaces), run green, then `git rm tests/evals/test_run_vggt_slam_lc.py`:

```bash
$PY -m pytest tests/evals/test_run_vggt_slam.py -q
git rm tests/evals/test_run_vggt_slam_lc.py
```

- [ ] **Step 4: Delete the LC wrapper + repoint docs**

```bash
git rm evals/runners/run_vggt_slam_lc.py
```
`evals/README.md`: document the parity workflow (run `run_vggt_slam.py --max_loops 0` → drop the TUM into a results dir → `eval.py` aggregation shows it as a row). `setup/vggt_slam.sh`'s example already names `run_vggt_slam.py` (its `scripts/` path is fixed in Task 11).

- [ ] **Step 5: Full suite + commit**

Run: `$PY -m pytest tests/ -q`

```bash
black . && isort .
git add -A
git commit -m "refactor(evals): merge run_vggt_slam_lc into run_vggt_slam (LC via --max_loops); document parity in README

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Dissolve `lc_parity_common.py` (spec §5)

Only `write_tum_allowed_frames` + frame helpers survive (used by merged `run_vggt_slam.py`). Everything else — including `_serialize_lc_decisions` (consumer removed in Task 2) — is dropped.

**Files:**
- Modify: `evals/datasets.py` (receive `write_tum_allowed_frames`, `collect_frames`, `list_scene_images`, `filter_images_to_list`)
- Modify: `evals/runners/run_vggt_slam.py` (import from `datasets`)
- Delete: `evals/runners/lc_parity_common.py`, `tests/evals/test_lc_parity_common.py`, `tests/evals/test_lc_decisions.py`
- Create/extend: `tests/evals/test_datasets.py` (equivalence coverage for the moved frame helpers)

- [ ] **Step 1: Write equivalence tests for the 4 moved helpers in their new home**

Add to `tests/evals/test_datasets.py` tests mirroring the current `test_lc_parity_common.py` assertions for `write_tum_allowed_frames`/`collect_frames`/`list_scene_images`/`filter_images_to_list`, importing from `datasets`. Run → FAIL (functions not yet in `datasets`).

- [ ] **Step 2: Move the 4 helpers into `datasets.py`**

Cut `write_tum_allowed_frames`, `collect_frames`, `list_scene_images`, `filter_images_to_list` (+ `_IMG_EXTS`) from `lc_parity_common.py` into `evals/datasets.py`. Run Step 1 tests → PASS.

- [ ] **Step 3: Repoint `run_vggt_slam.py`**

Change `from lc_parity_common import write_tum_allowed_frames` → `from datasets import write_tum_allowed_frames` (path adjusted for its dir). Run `$PY -m pytest tests/evals/test_run_vggt_slam.py -q` → PASS.

- [ ] **Step 4: Delete the file + its orphaned tests**

```bash
git rm evals/runners/lc_parity_common.py tests/evals/test_lc_parity_common.py tests/evals/test_lc_decisions.py
grep -rn 'lc_parity_common' --include=*.py evals/ tests/
```
Expected: no output.

- [ ] **Step 5: Full suite + commit**

Run: `$PY -m pytest tests/ -q`

```bash
black . && isort .
git add -A
git commit -m "refactor(evals): dissolve lc_parity_common — frame helpers to datasets.py, drop parity-gate machinery

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Retire sweep drivers / analysis / ate_utils / eval_suite.sh (spec §3)

Delete the validation-era scaffolding. Each script's covering test dies with it (allowed).

**Files (all deletions):**
- `evals/runners/run_cross_model_benchmark.py` + `tests/evals/test_cross_model_runner.py`
- `evals/runners/run_disparity_sweep.py`
- `evals/runners/run_lc_parity.py` + `tests/evals/test_run_lc_parity.py`
- `evals/runners/build_benchmark_table.py` + `tests/evals/test_build_benchmark_table.py`
- `evals/runners/build_parity_table.py` + `tests/evals/test_build_parity_table.py`
- `evals/runners/lc_loop_pr.py` + `tests/evals/test_lc_loop_pr.py`
- `evals/runners/visualize_lc_correction.py` + `tests/evals/test_visualize_lc_correction.py`
- `evals/ate_utils.py` + `tests/evals/test_ate_utils_tum.py`
- `evals/eval_suite.sh`

- [ ] **Step 1: Confirm no surviving importer**

```bash
for m in run_cross_model_benchmark run_disparity_sweep run_lc_parity build_benchmark_table build_parity_table lc_loop_pr visualize_lc_correction ate_utils; do echo "== $m =="; grep -rln "$m" --include=*.py evals/ tests/ collab_splats/ | grep -v "runners/$m.py\|test_.*$m\|/$m.py"; done
grep -rn 'eval_suite.sh' . --include=*.sh --include=*.md
```
Any hit outside the file itself + its test means a surviving consumer — STOP and reconcile (should be none after Tasks 2/6/8).

- [ ] **Step 2: Delete**

```bash
git rm evals/runners/run_cross_model_benchmark.py tests/evals/test_cross_model_runner.py \
       evals/runners/run_disparity_sweep.py \
       evals/runners/run_lc_parity.py tests/evals/test_run_lc_parity.py \
       evals/runners/build_benchmark_table.py tests/evals/test_build_benchmark_table.py \
       evals/runners/build_parity_table.py tests/evals/test_build_parity_table.py \
       evals/runners/lc_loop_pr.py tests/evals/test_lc_loop_pr.py \
       evals/runners/visualize_lc_correction.py tests/evals/test_visualize_lc_correction.py \
       evals/ate_utils.py tests/evals/test_ate_utils_tum.py \
       evals/eval_suite.sh
```

- [ ] **Step 3: Update README**

Edit `evals/README.md`: remove the `eval_suite.sh` row; add a line pointing to `eval.py --config configs/7scenes.yaml`.

- [ ] **Step 4: Full suite + commit**

Run: `$PY -m pytest tests/ -q` (pass count drops only by the retired tests)

```bash
black . && isort .
git add -A
git commit -m "refactor(evals): retire validation sweep drivers, table builders, ate_utils, eval_suite.sh

Superseded by config-driven eval.py; baselines frozen in evals/baselines/ + specs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Consolidate downloads → `evals/data/download_datasets.py` (spec §7)

**Files:**
- Create: `evals/data/download_datasets.py` (subcommand CLI, function per dataset)
- Move: `evals/runners/extract_waymo.py` → `evals/data/extract_waymo.py`
- Delete: `evals/download_7scenes.py`, `evals/download_7scenes.sh`, `evals/download_co3dv2.sh`, `evals/download_kitti.sh`, `evals/download_tum.sh`, `evals/download_waymo.sh`, `evals/data/download_parity_scenes.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Inventory the URLs/logic in each downloader**

```bash
for f in evals/download_7scenes.py evals/download_7scenes.sh evals/download_co3dv2.sh evals/download_kitti.sh evals/download_tum.sh evals/download_waymo.sh evals/data/download_parity_scenes.sh; do echo "== $f =="; cat "$f"; done
```

- [ ] **Step 2: Write `download_datasets.py`**

One function per dataset (`download_7scenes`, `download_co3dv2`, `download_kitti`, `download_tum`, `download_waymo`) preserving each script's URLs + output paths verbatim, dispatched by an argparse subcommand CLI (`python download_datasets.py 7scenes`, etc.). Preserve the existing output-dir conventions (do NOT unify paths — spec non-goal).

- [ ] **Step 3: Move extract_waymo**

```bash
git mv evals/runners/extract_waymo.py evals/data/extract_waymo.py
grep -rn 'extract_waymo' evals/ tests/   # repoint any reference
```

- [ ] **Step 4: Fix `.gitignore` allowlist**

`.gitignore` ignores `evals/data/*`; the old allowlist entry was `!evals/data/download_parity_scenes.sh`. Replace with:
```
!evals/data/download_datasets.py
!evals/data/extract_waymo.py
```
Verify: `git check-ignore evals/data/download_datasets.py` → no output (tracked).

- [ ] **Step 5: Delete the 7 old downloaders + commit**

```bash
git rm evals/download_7scenes.py evals/download_7scenes.sh evals/download_co3dv2.sh evals/download_kitti.sh evals/download_tum.sh evals/download_waymo.sh evals/data/download_parity_scenes.sh
$PY -m pytest tests/ -q
black . && isort .
git add -A
git commit -m "refactor(evals): consolidate 7 downloaders into data/download_datasets.py; move extract_waymo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: `runners/` → `scripts/` reorg (spec §8)

Pure path churn — do last. After Tasks 7-10, `runners/` holds only `run_vggt_slam.py` + surviving entrypoints.

**Files:**
- Move: `evals/runners/` → `evals/scripts/`; move `evals/eval.py`, `evals/eval_compare.py`, `evals/eval_multiview_conf.py`, `evals/eval_similarity_calibration.py` into `evals/scripts/`
- Keep at `evals/` top: `metrics.py`, `trajectory_io.py`, `datasets.py`, `configs/`, `data/`, `baselines/`, `results/`
- Modify: all `tests/evals/*.py` `sys.path`/`from runners.X` refs, `CLAUDE.md` (2 refs), `evals/README.md`, `setup/vggt_slam.sh`

- [ ] **Step 1: Move the directory + entrypoints**

```bash
git mv evals/runners evals/scripts
for f in eval.py eval_compare.py eval_multiview_conf.py eval_similarity_calibration.py; do git mv evals/$f evals/scripts/$f; done
```

- [ ] **Step 2: Repoint test path arithmetic**

```bash
grep -rln 'runners\|sys.path.insert.*evals\|from eval import\|from eval_compare' tests/
```
For each test: `from runners.X` → `from scripts.X`; a `sys.path.insert(..., "evals")` that imported top-level `eval`/`eval_compare` must now point at `evals/scripts`. Fix the path depth in each `sys.path.insert`. Run `$PY -m pytest tests/evals/ tests/geometry/loop_closure/ -q` iteratively until green.

- [ ] **Step 3: Repoint docs/config refs + sibling paths**

- `CLAUDE.md`: 2 literal `evals/eval_gt.py` → `evals/scripts/eval.py` (both the architecture tree line and the Development Commands line).
- `evals/README.md`: script paths → `scripts/`.
- `setup/vggt_slam.sh`: `python evals/runners/run_vggt_slam.py` → `python evals/scripts/run_vggt_slam.py`.
- In `eval.py`/`run_vggt_slam.py`, any `Path(__file__).parent`-relative reference to `metrics.py`/`trajectory_io.py`/`datasets.py`/`configs/`/`data/` now needs `..` (they moved a dir deeper). Update those relative paths / `sys.path.insert`.

- [ ] **Step 4: Import smoke + full suite**

```bash
$PY -c "import sys; sys.path.insert(0,'evals/scripts'); import eval, eval_compare; print('ok')"
$PY -m pytest tests/ -q
```
Expected: `ok`, then full green.

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add -A
git commit -m "refactor(evals): runners/ -> scripts/; fold eval entrypoints in; repoint tests + docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification + merge back

- [ ] **Step 1: Green full suite from a clean tree**

```bash
cd /workspace/collab-splats/.worktrees/evals-lc-cleanup
git status --porcelain            # empty
$PY -m pytest tests/ -q -rfE      # compare pass count vs baseline; only retired-script tests should be gone
$PY -m pytest tests/test_cu121_migration.py -q   # 24/24 gate
```

- [ ] **Step 2: Update `docs/known-test-failures.md`** with the new pass count + a one-line note that the drop = retired scripts (not regressions). `git add -f` it, commit.

- [ ] **Step 3: Merge into `refactor/cu121-uv-migration`**

```bash
cd /workspace/collab-splats
git checkout refactor/cu121-uv-migration
git merge --no-ff evals-lc-cleanup -m "Merge evals + loop-closure aggressive cleanup"
$PY -m pytest tests/ -q            # re-verify post-merge
```

- [ ] **Step 4: Clean up the worktree**

```bash
git worktree remove .worktrees/evals-lc-cleanup
git branch -d evals-lc-cleanup
```

---

## Self-review notes (coverage map)

- Spec §1/§2 → Task 6. §3 → Task 9. §4 → Task 7. §5 → Task 8. §6 → Tasks 2 (reconstruction_quality retire) + 3 (`_invert_se3`, edge helpers). §7 → Task 10. §8 → Task 11. §9a → Task 1. §9b → Task 4. §9c → Task 5. §9d → Task 2. §10 (equivalence) → embedded per task. §11 (order) → task ordering.
- **Ordering dependency:** Task 2 removes the `_lc_*`/diagnostic consumers BEFORE Task 4 (closure split) and Task 8 (`_serialize_lc_decisions` drop) — those tasks operate on already-lean code. Task 6 renames `eval_gt`→`eval` before Task 9 retires the drivers that shelled it (drivers die, not repoint). Task 11 (path reorg) strictly last.
- **Risk note:** Task 6's single-cell CLI path must stay byte-compatible with existing `test_eval_gt*` tests (they exercise the leaf subprocess). Do not alter the per-condition leaf — only add the `--config` grid wrapper around it.
- **Naming consistency:** `load_eval_config`/`build_grid`/`EvalConfig`/`EvalCell` (Task 6) used identically in test + impl; `scan_results_dir`/`format_markdown`/`collect_grid_metrics`/`format_markdown_rows` (Task 6) used identically in `eval.py` + `eval_compare.py` + tests; `n_loops_applied` (Task 2) used identically in wrapper + eval_gt + surviving tests.
