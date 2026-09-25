# Geometry Release Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `collab_splats/geometry/` releasable: brief prose, no dead or deprecated code, tunables as kwargs, evaluation code in `evals/`. Loop closure output must not change.

**Architecture:** Round 1 edits only docstrings and comments. Each Round 1 commit is proven behavior-free by an AST comparison. Round 2 makes one logical code change per commit, test first, and runs the gate after each one. Two loop-closure parity gates guard every commit that touches `loop_closure/` or `transforms.py`. The end state adds `"geometry"` to `PACKAGES` and `RELEASED` in the docstring contract.

**Tech Stack:** Python 3.11, numpy, torch, GTSAM, pycolmap, zarr 3.1.5, pytest, `ast`.

**Spec:** [2026-09-25-geometry-release-cleanup-design.md](../specs/2026-09-25-geometry-release-cleanup-design.md) · Rules: [decision 017](../decisions/017-release-cleanup-rules.md)

---

## Conventions for every task

- `WT=/workspace/collab-splats/.worktrees/geometry-release` (branch `clean/geometry-release`).
  Every command starts with `cd $WT &&`, because the working directory resets between Bash calls.
- `SP=/tmp/claude-0/-workspace-collab-splats/351261bd-d06a-403c-8d8b-cd81503fd03c/scratchpad`
  holds scratch files: the proof tools, the parity captures and the baseline. Nothing in `$SP`
  is committed.
- `PY=/opt/venv/reconstruction/bin/python`.
- **Gate** (`G`), run after every commit-bound change:

  ```bash
  cd $WT && PYTHONPATH=$WT $PY -c "import collab_splats; print(collab_splats.__file__)" \
    && cd $WT && PYTHONPATH=$WT $PY -m pytest \
       tests/geometry tests/evals tests/pointcloud/feedforward tests/pointcloud/test_pose_extraction.py \
       tests/wrapper tests/test_docstring_contract.py -q -p no:cacheprovider
  ```

  - The printed path must start with `$WT/`.
  - Never pipe through `tail`, and never use `--tb=no`.
  - Compare the counts against `$SP/baseline.txt`. Every change in a count must be
    explained by the task's own test adds and deletes. Compare the SKIP count too.
  - Never run the full suite. Other sessions gate at the same time, and two full suites
    exceed the 46.6 GB cgroup.
- **LC parity gate 1** (`P1`), run after every Round 2 commit that touches
  `collab_splats/geometry/loop_closure/` or `collab_splats/geometry/transforms.py`:

  ```bash
  cd $WT && PYTHONPATH=$WT $PY $SP/lc_parity.py compare $SP/lc_parity_base.npz
  ```

  It must print `PARITY OK`. Any other output means the commit is reverted with
  `git revert --no-edit HEAD`. Then stop and report. The gate is never loosened.
- Do not `pip install` or `uv sync`. The venv is shared with other sessions.
- Do not edit notebooks, anything under `.worktrees/tutorial-rework`, or `configs/`.
- Commit only your own paths: `git add <paths> && git commit --only <paths> -m ...`. Other
  sessions share the git index. Never `--amend`, rebase, reset or bare `git stash`. Fix a
  mistake with a new commit. Do not merge or push.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Spelling is US (`color`, `normalize`, `center`, `optimize`).
- Use `rtk proxy grep ...` when grep output looks truncated or rewritten.
- Prose shape (CLAUDE.md, decision 017):
  - docstring summary on the line after `"""`, ≤100 chars, not restating the name
  - then `- ` bullets (≤6), then `Args:` / `Returns:` / `Raises:`
  - comment runs ≤4 lines; a run of 3+ lines is a header line followed by `- ` bullets
  - no `measured`, `hypothesis`, `Nx faster`, or scene ids

---

## Task 0: Verify worktree, proof tools, baselines

**Files:**
- Create (scratch, not committed): `$SP/prose_proof.py`, `$SP/contract_hits.py`,
  `$SP/lc_parity.py`, `$SP/baseline.txt`, `$SP/lc_parity_base.npz`

The worktree and branch already exist, forked from `clean/final` at `311ca5c8`. The spec
commit `bfcf4819` and this plan are on it.

- [ ] **Step 1: Verify the worktree, the symlinks and a clean tree**

```bash
cd $WT && git branch --show-current && git status --short && ls third_party
```

Expected output:
- `clean/geometry-release`
- empty status
- `third_party` lists `LoGeR VGGT-SLAM VGGT-X Video-Depth-Anything bae hloc vggt-omega vggt_spark xfeat`

- [ ] **Step 2: Write the prose-proof tool**

```bash
cat > $SP/prose_proof.py <<'EOF'
"""
Prove a commit range changed only docstrings and comments in the given .py files.

Usage: prose_proof.py <base-ref> <file> [<file> ...]   (run from the worktree root)
"""
import ast
import subprocess
import sys


def stripped(src: str) -> str:
    """
    AST dump with every docstring statement deleted; comments never reach the AST.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [
                s for s in body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
            ] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


base, files = sys.argv[1], sys.argv[2:]
bad = []
for f in files:
    old = subprocess.run(["git", "show", f"{base}:{f}"], capture_output=True, text=True, check=True).stdout
    new = open(f).read()
    if stripped(old) != stripped(new):
        bad.append(f)
print("CODE CHANGED:" if bad else "PROSE ONLY", *bad)
sys.exit(1 if bad else 0)
EOF
```

- [ ] **Step 3: Sanity-check the proof tool on a known-bad mutation**

```bash
cd $WT && cp collab_splats/geometry/bundle_adjustment.py $SP/ba.bak \
  && sed -i 's/    lm_steps: int = 40$/    lm_steps: int = 41/' collab_splats/geometry/bundle_adjustment.py \
  && $PY $SP/prose_proof.py HEAD collab_splats/geometry/bundle_adjustment.py; \
  cp $SP/ba.bak collab_splats/geometry/bundle_adjustment.py && git -C $WT status --short
```

Expected: `CODE CHANGED: collab_splats/geometry/bundle_adjustment.py`, then an empty `git status`.
If the tool prints `PROSE ONLY`, it is broken. Stop.

- [ ] **Step 4: Write the contract checker**

Round 1 is driven by this list. It runs the release checks and the three shape checks of
`tests/test_docstring_contract.py` over each file, before geometry is in `PACKAGES`.

```bash
cat > $SP/contract_hits.py <<'EOF'
"""
Every docstring-contract hit in the given files (default: all of collab_splats/geometry).

Usage: contract_hits.py [<file> ...]   (run from the worktree root)
"""
import pathlib
import sys

sys.path.insert(0, "tests")
import test_docstring_contract as t

files = [pathlib.Path(a) for a in sys.argv[1:]] or sorted(pathlib.Path("collab_splats/geometry").rglob("*.py"))
n = 0
for p in files:
    src = p.read_text()
    for name, fn in t.RELEASE_CHECKS.items():
        for h in fn(src):
            print(f"{p}\t{name}\t{h}")
            n += 1
    for check in (
        t.test_module_docstring_is_a_bulleted_summary,
        t.test_public_defs_are_documented_and_annotated,
        t.test_comment_runs_state_the_problem_then_bullet_it,
    ):
        try:
            check(p)
        except AssertionError as e:
            print(f"{p}\t{check.__name__}\t{e}")
            n += 1
print(f"TOTAL {n}")
EOF
cd $WT && $PY $SP/contract_hits.py > $SP/hits_base.txt; tail -1 $SP/hits_base.txt
```

Expected: a `TOTAL` line above 0. If the import fails because a name in
`test_docstring_contract` differs, read that file and fix the three names, not the test file.

- [ ] **Step 5: Write the LC parity script (gate 1)**

It is a self-contained copy of the `test_wrapper.py` harness (`_rot_z`, `_make_raw_nontrivial`,
`_run_lc_harness_with_timing`), run over every `scale_method` and both `loop_edge_timing`
values. It also runs the real `find_loop_closures` on seeded retrieval vectors, because the
harness patches it out. It must not import `test_wrapper`, which Round 2 edits.

```bash
cat > $SP/lc_parity.py <<'EOF'
"""
LC parity gate 1: bit-exact LoopClosure output on the stubbed test_wrapper harness.

Usage: lc_parity.py capture|compare <npz>   (run from the worktree root, PYTHONPATH=$WT)
"""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import collab_splats
from collab_splats.geometry.loop_closure.matching import LoopMatch, find_loop_closures
from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig

H = W = 32
N_FRAMES = 9
SCALE_METHODS = ("rotation_only", "se3", "pairwise_dist", "none")
TIMINGS = ("deferred", "live")


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _make_raw_nontrivial(k, H, W):
    extrinsic = np.zeros((k, 3, 4), dtype=np.float32)
    extrinsic[0] = np.eye(3, 4, dtype=np.float32)
    for i in range(1, k):
        extrinsic[i, :, :3] = _rot_z(0.15 * i)
        extrinsic[i, :, 3] = np.array([0.1 * i, -0.05 * i, 0.2 * i], dtype=np.float32)
    K = np.array([[200.0, 0.0, W * 0.5 + 3.0], [0.0, 200.0, H * 0.5 - 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    intr = np.tile(K, (k, 1, 1))
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    base_depth = 1.5 + 0.01 * (rows + cols)
    depth = np.stack([base_depth + 0.1 * i for i in range(k)], axis=0)[..., None].astype(np.float32)
    conf_grid = 50.0 + 0.5 * (rows + cols)
    depth_conf = np.tile(conf_grid, (k, 1, 1)).astype(np.float32)
    return {
        "extrinsic": extrinsic,
        "intrinsics": intr,
        "intrinsics_downsampled": intr,
        "depth": depth,
        "depth_conf": depth_conf,
    }


def run_case(scale_method, timing):
    """One stubbed LC run; returns a flat dict of arrays."""
    find_args, verify_ratios = [], []
    cfg = LoopClosureConfig(
        submap_size=3,
        submap_overlap=1,
        min_submap_gap=0,
        max_loops_per_submap=5,
        loop_edge_timing=timing,
        scale_method=scale_method,
    )

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        verify_ratios.append(float(verify_match_ratio))
        pose0 = np.eye(4, dtype=np.float32)
        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, :3] = _rot_z(0.1)
        pose1[:3, 3] = np.array([0.05, 0.02, 0.1], dtype=np.float32)
        wp = np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32)
        conf = np.full((2, H, W), 100.0, dtype=np.float32)
        return True, {"poses": np.stack([pose0, pose1]), "world_points": wp, "conf": conf}

    def fake_find(submap, past, *args, **kwargs):
        find_args.append((float(args[0]), float(args[1]), float(kwargs["nms_frame_distance"])))
        if len(past) >= 1:
            return [
                LoopMatch(
                    similarity_score=0.1,
                    query_submap_id=submap.submap_id,
                    detected_submap_id=0,
                    query_frame_idx=1,
                    detected_frame_idx=1,
                )
            ]
        return []

    base = MagicMock()
    base.max_points = 500_000
    base.default_verify_match_ratio = 0.85
    base.views = torch.full((N_FRAMES, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(N_FRAMES)]
    base.original_coords = np.zeros((N_FRAMES, 6), dtype=np.float32)
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)
    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    out = base.outputs
    return {
        "points": out.points,
        "colors": out.colors,
        "extrinsics": out.extrinsics,
        "intrinsics": out.intrinsics,
        "model_hw": np.array([out.model_height, out.model_width]),
        "n_lc": np.array(len(lc._last_lc_submaps)),
        "find_args": np.array(find_args, dtype=np.float64),
        "verify_ratios": np.array(verify_ratios, dtype=np.float64),
    }


def run_find(nms):
    """Real find_loop_closures on seeded retrieval vectors; (M, 5) rows of match fields."""
    g = torch.Generator().manual_seed(0)
    past = [
        Submap(
            submap_id=i,
            poses=np.tile(np.eye(4, dtype=np.float32), (6, 1, 1)),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (6, 1, 1)),
            retrieval_vectors=torch.nn.functional.normalize(torch.randn(6, 16, generator=g), dim=1),
            image_paths=[f"p{i}_{j}.png" for j in range(6)],
        )
        for i in range(4)
    ]
    query = Submap(
        submap_id=4,
        poses=np.tile(np.eye(4, dtype=np.float32), (6, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (6, 1, 1)),
        retrieval_vectors=torch.nn.functional.normalize(torch.randn(6, 16, generator=g), dim=1),
        image_paths=[f"q_{j}.png" for j in range(6)],
    )
    matches = find_loop_closures(query, past, 1.3, 5, nms_frame_distance=nms)
    return np.array(
        [[m.similarity_score, m.query_submap_id, m.detected_submap_id, m.query_frame_idx, m.detected_frame_idx]
         for m in matches],
        dtype=np.float64,
    ).reshape(-1, 5)


def run_all():
    """Every case, keyed '<scale_method>/<timing>/<field>' and 'find/nms<k>'."""
    res = {}
    for sm in SCALE_METHODS:
        for tm in TIMINGS:
            for k, v in run_case(sm, tm).items():
                res[f"{sm}/{tm}/{k}"] = np.asarray(v)
    for nms in (0, 2):
        res[f"find/nms{nms}"] = run_find(nms)
    return res


def diff(a, b):
    """Keys that differ between two result dicts, bit-exact."""
    keys = sorted(set(a) | set(b))
    return [k for k in keys if k not in a or k not in b or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])]


print("collab_splats from", collab_splats.__file__)
mode, path = sys.argv[1], sys.argv[2]
cur = run_all()
if mode == "capture":
    again = run_all()
    bad = diff(cur, again)
    if bad:
        print("NONDETERMINISTIC:", bad)
        sys.exit(1)
    n_lc = {k: int(v) for k, v in cur.items() if k.endswith("/n_lc")}
    if not all(n_lc.values()):
        print("HARNESS CLOSES NO LOOP:", n_lc)
        sys.exit(1)
    np.savez(path, **cur)
    print(f"CAPTURED {len(cur)} arrays -> {path}")
else:
    ref = dict(np.load(path))
    bad = diff(ref, cur)
    print("PARITY OK" if not bad else f"PARITY FAILED: {bad}")
    sys.exit(1 if bad else 0)
EOF
```

- [ ] **Step 6: Capture parity on the untouched tree, then self-compare**

```bash
cd $WT && PYTHONPATH=$WT $PY $SP/lc_parity.py capture $SP/lc_parity_base.npz \
  && cd $WT && PYTHONPATH=$WT $PY $SP/lc_parity.py compare $SP/lc_parity_base.npz
```

Expected:
- both runs print `collab_splats from $WT/collab_splats/__init__.py`
- `CAPTURED 66 arrays -> ...`: 4 scale methods × 2 timings × 8 fields, plus 2 find rows
- then `PARITY OK`

If capture prints `NONDETERMINISTIC` or `HARNESS CLOSES NO LOOP`, stop. A gate that cannot
fail is worthless.

- [ ] **Step 7: Sanity-check the parity gate on a known-bad mutation**

```bash
cd $WT && cp collab_splats/geometry/loop_closure/graph.py $SP/graph.bak \
  && rtk proxy grep -n '^                    scale = 1.0$' collab_splats/geometry/loop_closure/graph.py \
  && sed -i 's/^                    scale = 1.0$/                    scale = 1.01/' collab_splats/geometry/loop_closure/graph.py \
  && PYTHONPATH=$WT $PY $SP/lc_parity.py compare $SP/lc_parity_base.npz; \
  cp $SP/graph.bak collab_splats/geometry/loop_closure/graph.py && git -C $WT status --short
```

The grep must print exactly one line: the `scale_method == "none"` branch of
`PoseGraph.add_submap`, which is 20 spaces deep. The line at 12 spaces is the default
initializer, and the sed does not touch it.

Expected: `PARITY FAILED: [...]` naming `none/*` `points` or `extrinsics` keys, then an
empty `git status`. If it prints `PARITY OK`, stop: the harness does not reach the
sequential-edge scale path.

Two mutations were rejected, and neither may be used here:
- `_MIN_CONF_POINTS = 10**9` is a no-op. The harness conf is ≥50, above `conf_threshold`
  25, so every fallback mask is all-true.
- Changing the `scale_method` default is a no-op. The harness passes every method explicitly.

- [ ] **Step 8: Run the baseline gate and record the counts**

Run gate `G` on the untouched tree. Write two things into `$SP/baseline.txt`:
- the final summary line, e.g. `N passed, M skipped, K xfailed, F failed`
- every failing node id

Every later gate is compared against this file.

- [ ] **Step 9: Real-scene parity (gate 2) — data check. STOP if the data is missing**

```bash
ls /workspace/collab-splats/data/7scenes/chess/seq-01 2>&1 | head -3
```

The spec's gate 2 runs `evals/scripts/eval.py` on 7-Scenes chess. On 2026-09-25 `data/` holds
only `outputs/` and `tutorial/`, so this prints `No such file or directory`. **STOP and ask
the user** which of these to run:

- **A. Download chess.** The user fetches `chess.zip` from the Microsoft 7-Scenes page into
  `/workspace/collab-splats/data/7scenes/` and unzips `seq-01`. Then run step 10A.
- **B. Tutorial-video substitute.** No download. Step 10B runs the pointcloud stage with LC on
  `data/tutorial/tutorial_example-video.mp4`. Its extrinsics and loop count are compared
  bit-exact, provided two baseline runs agree bit-exact with each other.

- [ ] **Step 10A: Gate 2 baseline on chess (if A)**

In tmux, with nothing else running on the GPU:

```bash
tmux new -d -s geo-gate2-base "cd $WT && PYTHONPATH=$WT $PY evals/scripts/eval.py --dataset 7scenes \
  --seq_dir /workspace/collab-splats/data/7scenes/chess/seq-01 --backbone vggt_omega --conditions lc \
  --submap_size 20 --lc_scale_method rotation_only --output_dir $SP/gate2/base \
  > $SP/gate2_base.log 2>&1"
```

Check the flags with `$PY evals/scripts/eval.py --help` first, and use the names it prints.
When it finishes, record ATE, RPE and the accepted-loop count (`grep -c "↩ Loop:" $SP/gate2_base.log`)
in `$SP/baseline.txt`.

- [ ] **Step 10B: Gate 2 baseline on the tutorial video (if B)**

```bash
cat > $SP/lc_real.yaml <<'EOF'
preproc:
  frame_selection: fps
  fps: 4.0
  max_frames: 120
  min_frames: 60
pointcloud:
  method: feedforward
  backend: vggt_omega
  loop_closure:
    enabled: true
    submap_size: 20
    submap_overlap: 1
    scale_method: rotation_only
  viz:
    enabled: false
semantics:
  enabled: false
EOF
```

Run the baseline twice in tmux, one after the other. The first run includes preproc, and the
second reruns the pointcloud stage on a copy of the first run's output:

```bash
tmux new -d -s geo-gate2-base "cd $WT && PYTHONPATH=$WT $PY docs/examples/run_pipeline.py --config $SP/lc_real.yaml \
  --stages preproc,pointcloud --output-root $SP/lc_real/base data/tutorial/tutorial_example-video.mp4 \
  > $SP/lc_real_base.log 2>&1 && cp -r $SP/lc_real/base $SP/lc_real/base2 \
  && PYTHONPATH=$WT $PY docs/examples/run_pipeline.py --config $SP/lc_real.yaml --stages pointcloud --overwrite \
  --output-root $SP/lc_real/base2 data/tutorial/tutorial_example-video.mp4 > $SP/lc_real_base2.log 2>&1"
```

Then compare the two runs:

```bash
cd $WT && $PY - <<'EOF'
import glob, numpy as np, zarr
SP = "/tmp/claude-0/-workspace-collab-splats/351261bd-d06a-403c-8d8b-cd81503fd03c/scratchpad"
a, b = (glob.glob(f"{SP}/lc_real/{r}/**/pointcloud.zarr", recursive=True)[0] for r in ("base", "base2"))
za, zb = zarr.open(a, mode="r"), zarr.open(b, mode="r")
print("keys", sorted(za.array_keys()))
for k in ("extrinsics", "intrinsics", "points"):
    print(k, np.array_equal(za[k][:], zb[k][:]))
EOF
grep -c "↩ Loop:" $SP/lc_real_base.log $SP/lc_real_base2.log
```

- If all three arrays print `True` and the loop counts match, gate 2 is bit-exact. Record the
  loop count in `$SP/baseline.txt`.
- If the runs differ, record the max abs diff per array as the noise floor. Report it to the
  user before Round 2. Do not pick a tolerance alone.
- If `semantics` is not a config key, drop that block. Read `configs/base.yaml` for the real
  stage keys.

---

## Round 1 — prose only (Tasks 1-9)

Every Round 1 task follows the same five steps. Only the file list and the named edits differ.

1. Run `cd $WT && $PY $SP/contract_hits.py <files>` and save the output to `$SP/hits_T<n>.txt`.
2. Rewrite every hit, plus the spec's named edits for these files. Use the prose shape in
   Conventions. Change only docstrings and `#` comments. Do not touch code, string literals
   that are not docstrings, log messages, or blank lines inside code.
3. **Proof:** `cd $WT && $PY $SP/prose_proof.py HEAD <files>` must print `PROSE ONLY`.
   `CODE CHANGED` means a code edit slipped in. Find it with `git diff` and undo it.
4. Run `cd $WT && $PY $SP/contract_hits.py <files>`. It must report no `comment-cap`,
   `banned-word`, docstring-shape or annotation-*docstring* hits. Three hit types are allowed
   to remain, because Round 2 removes the code behind them:
   - `silent-fallback`
   - `numeric-constant`
   - "missing annotation" (Task 30 fixes it)
5. Run gate `G`. The counts must equal the baseline. Then commit:
   ```bash
   cd $WT && git add <files> && git commit --only <files> -m "docs(geometry): <file> prose to contract

   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
   ```

Contract wording applies in every file:
- Where the code is method-agnostic, prose that says "FeedforwardResult" or
  "feedforward reconstruction" becomes "the `pointcloud.zarr` result".
- "feedforward" stays only where it describes loop closure, which wraps a creator's forward pass.
- A `measured ...` sentence is deleted, never reworded to hide the word.

### Task 1: Package `__init__` files and the API page

**Files:** `collab_splats/geometry/__init__.py`, `collab_splats/geometry/loop_closure/__init__.py`,
`docs/source/api/geometry.rst`

- [ ] **Step 1: Check the contract hits** (procedure step 1).
- [ ] **Step 2: Rewrite the two module docstrings.** Each becomes a summary line plus 2-4
  bullets, one per module it exports. Package:

  ```python
  """
  Pose and geometry backend: bundle adjustment, loop closure, verification, scene metrics.

  - transforms: pose conversions, Umeyama alignment, intrinsics from points
  - bundle_adjustment: LM refinement of a `pointcloud.zarr` result (feedforward only at refine)
  - verification / metrics: report-only geometric checks over a `pointcloud.zarr` result
  - loop_closure: submap pose graph around a feedforward creator's forward pass
  """
  ```

  The `global_alignment` line stays in `__init__` until Task 11.
- [ ] **Step 3: Remove the `loop_closure.closure` automodule block from `geometry.rst`.**
  The module does not exist. The `.rst` file is not Python, so step 3 of the procedure
  (prose proof) skips it. Check it with
  `rtk proxy grep -n "closure.closure" docs/source/api/geometry.rst`, which must print nothing.
- [ ] **Step 4: Run the proof, the contract hits and the gate, then commit** (procedure
  steps 3-5). Commit message: `docs(geometry): package docstrings and API page`.

### Task 2: `transforms.py`

**Files:** `collab_splats/geometry/transforms.py`

- [ ] **Step 1: Check the contract hits.** Expect 3 `banned-word` hits and the comment-cap hits.
- [ ] **Step 2: Cut the LoGeR lore in `estimate_intrinsics_from_points`.** Cut the docstring
  bullets and the 13-line comment at :185 down to one line of *why*. For example:
  `# Bounds reject pixels whose ray is near the principal point (fx is unstable there)`.
  Keep the `Args:`/`Returns:`/`Raises:` sections, each with every parameter.
- [ ] **Step 3: Put the `umeyama_se3`/`umeyama_sim3` docstrings in contract shape.** The
  summary line goes after `"""`. Do not describe the degenerate-input behavior yet; Task 10
  changes it.
- [ ] **Step 4: Run the proof, the contract hits and the gate, then commit.**

### Task 3: `bundle_adjustment.py`

**Files:** `collab_splats/geometry/bundle_adjustment.py`

- [ ] **Step 1: Check the contract hits.**
- [ ] **Step 2: Make the spec's named edits.**
  - `BundleAdjustmentConfig` gets a docstring: a summary line plus bullets for the three
    modes (`increment_size`, `shared_camera`, `max_reproj_error=None`).
  - Each field keeps exactly one comment line. Delete the `increment_size` sweep results (:76-77).
  - The `BundleAdjustment` class docstring (:89) says:
    - it refines a `pointcloud.zarr` result with K at model resolution
    - the refine stage refuses sfm
    - Delete "any FeedforwardResult regardless of source creator".
  - `_scale_intrinsics_to_model` docstring (:480-484): delete the claim that stored K is at
    original resolution. Say it maps K to the model grid when K is not already there.
    Task 18 then cuts the function.
  - The `_extract_tracks_vggsfm` comment (:680) becomes
    `# numpy images: sfm's result_from_reconstruction stores float32 arrays, not tensors`.
  - Split the 6-line and 5-line comment runs into a header line plus bullets.
- [ ] **Step 3: Run the proof, the contract hits and the gate, then commit.**

### Task 4: `verification.py`

**Files:** `collab_splats/geometry/verification.py`

- [ ] **Step 1: Check the contract hits.**
- [ ] **Step 2: Make the named edits.**
  - Module docstring: "verifies any `pointcloud.zarr` reconstruction (feedforward or sfm)".
    Delete "feedforward reconstructions".
  - Delete the stale "Task 5 fills in" (:311).
  - `PairStats`: delete the "Measured" sentence.
  - Split the comment runs into a header line plus bullets.
  - The reconstruction copy gets `# Copy first: pycolmap triangulation mutates its argument`.
- [ ] **Step 3: Run the proof, the contract hits and the gate, then commit.**

### Task 5: `metrics.py`

**Files:** `collab_splats/geometry/metrics.py`

- [ ] **Step 1: Check the contract hits.** Expect 11 `banned-word` hits.
- [ ] **Step 2: Make the named edits.**
  - Module docstring:
    ```python
    """
    Reference-free scene error metrics over a `pointcloud.zarr` result.

    - depth cross-view residuals, photometric NCC, and verify's epipolar rows
    - report-only: distributions, no verdicts, nothing fed back into a reconstruction
    - reads the result of either pointcloud method (feedforward or sfm)
    """
    ```
  - `build_reconstruction_quality_report`: summary line plus ≤6 bullets.
    - Delete the "A function, not a class" paragraph.
    - Delete the "absolute thresholds" paragraph and keep one bullet: "no grades, causes or
      flagged frames".
    - Add an `Args:` entry for each of the 5 parameters and a `Returns:` entry.
  - Image-range comment (:337):
    ```python
    # Guide must be uint8 [0, 255]; the contract's `images` may be [0, 1] or [0, 255]
    # - scale decided once over the whole array, never per frame
    # - a per-frame decision would amplify a dark [0, 255] frame 255x
    ```
  - Every `measured` goes: 11 hits, in docstrings and comments.
  - The 8-line runs (`_FRAME_STEM_RE`, confidence, running-error, source-index, crop-coverage)
    each become a header line plus ≤3 bullets.
  - `_running_error` docstring: summary line, 3 bullets, `Args:`, `Returns:`.
- [ ] **Step 3: Run the proof, the contract hits and the gate, then commit.**

### Task 6: `loop_closure/graph.py`

**Files:** `collab_splats/geometry/loop_closure/graph.py`

- [ ] **Step 1: Check the contract hits.**
- [ ] **Step 2: Make the named edits.**
  - `decompose_camera` (:76-96): summary line plus bullets. Delete "see closure.py" and
    the ATE-gap history. Keep `Args:`/`Returns:`.
  - Comment at :307: delete the claim that `se3` is the default. The default is
    `rotation_only`.
  - `add_loop_edge` docstring: delete "(monolith lines ~581-624)".
  - `PoseGraph` class docstring and every method docstring follow the contract shape.
  - Every multi-line comment run becomes a header line plus bullets. Keep the VGGT-SLAM
    file:line citations.
- [ ] **Step 3: Run the proof, the contract hits and the gate, then commit.** Round 1 needs
  no P1: the AST proof covers it.

### Task 7: `loop_closure/submap.py` and `loop_closure/map.py`

**Files:** `collab_splats/geometry/loop_closure/submap.py`, `collab_splats/geometry/loop_closure/map.py`

- [ ] **Step 1: Check the contract hits.**
- [ ] **Step 2: Make the named edits.**
  - Delete the `TODO(spec-2)` trailing comment on `import torch`. The import stays.
  - `Submap` gets a class docstring:
    ```python
    """
    One window of frames from a single forward pass, in that window's local frame.

    - poses are world-to-cam (frame 0 ≈ identity); `assert_world_to_cam` checks it
    - dense fields (points/colors/conf) are per pixel; conf_threshold is their 25th percentile
    - is_lc_submap marks a 2-frame loop carrier, which never contributes points
    """
    ```
  - Every method docstring in both files follows the contract shape. Methods deleted in
    Round 2 (`get_world_points`, `get_poses_world`, `get_submap`, `get_largest_key`,
    `get_latest_submap`, `get_corrected_extrinsics`, `_keys`) get a one-line summary only.
  - Add a module docstring to `submap.py`, which has none.
- [ ] **Step 3: Run the proof, the contract hits and the gate, then commit.**

### Task 8: `loop_closure/matching.py`

**Files:** `collab_splats/geometry/loop_closure/matching.py`

- [ ] **Step 1: Check the contract hits.**
- [ ] **Step 2: Make the named edits.**
  - Module docstring: drop the translation-jump sentence. Task 26 deletes that function,
    and a docstring that names it now would be false.
  - `LoopMatch`: document `similarity_score` as the L2 distance between unit DINO-SALAD
    descriptors, where lower means more similar.
  - `LoopMatchQueue.push` and `get_matches` get docstrings.
  - `translation_jump_check` gets no edit, because Task 26 deletes it.
- [ ] **Step 3: Run the proof, the contract hits and the gate, then commit.**

### Task 9: `loop_closure/wrapper.py`

**Files:** `collab_splats/geometry/loop_closure/wrapper.py`

- [ ] **Step 1: Check the contract hits.**
- [ ] **Step 2: Make the named edits.**
  - `LoopClosureConfig` gets a docstring: a summary line plus one bullet per field group.
  - Each field keeps one comment line:
    - `submap_overlap`: drop "4 = old default". Keep "1 = VGGT-SLAM parity".
    - `verify_match_ratio`: drop "(fallback 0.85)".
    - `loop_edge_timing`: cut the 150-character comment to
      `# deferred: all loop edges after the window loop; live: each edge as found (VGGT-SLAM)`.
  - `__init__` comment: drop "fallback 0.85 (VGGT-SPARK calibration)" and "(P6.3)".
  - `LoopClosure` docstring, including the Quickstart:
    - contract shape
    - the Quickstart must name the real entry point, `run_inference()`
  - Split the 7-line run at :605 into a header line plus bullets.
  - The raw-outputs 5-line run at :290 is left alone, because Task 25 deletes it with its code.
- [ ] **Step 3: Run the proof, the contract hits and the gate, then commit.**

---

## Round 2 — code (Tasks 10-30)

Every Round 2 task is one commit. The order inside a task is fixed:
1. Write the test.
2. Run it and watch it fail, when there is a new behavior.
3. Make the edit.
4. Run the targeted test.
5. Run gate `G`, plus `P1` if the task touches `loop_closure/` or `transforms.py`.
6. Commit with `git commit --only`.

Callers, tests and docs change in the same commit as the code.
Commit messages use `refactor(geometry):`, `fix(geometry):` or `feat(geometry):`, and each
names the task number in its body.

### Task 10: `umeyama_se3` / `umeyama_sim3` raise on degenerate input

**Files:** `collab_splats/geometry/transforms.py:270-340`, `tests/geometry/test_transforms.py:353-360`

- [ ] **Step 1: Replace the identity test with raise tests**

Delete `test_umeyama_sim3_too_few_points_returns_identity` and add:

```python
@pytest.mark.parametrize("fn", [umeyama_se3, umeyama_sim3])
def test_umeyama_raises_on_fewer_than_three_points(fn):
    with pytest.raises(ValueError, match="at least 3"):
        fn(np.zeros((2, 3)), np.ones((2, 3)))


@pytest.mark.parametrize("fn", [umeyama_se3, umeyama_sim3])
def test_umeyama_raises_on_zero_total_weight(fn):
    rng = np.random.default_rng(0)
    src = rng.normal(size=(5, 3))
    with pytest.raises(ValueError, match="zero total weight"):
        fn(src, src, weights=np.zeros(5))
```

Put `import pytest` and `from collab_splats.geometry.transforms import umeyama_se3, umeyama_sim3`
at the top of the file if they are not there already.

- [ ] **Step 2: Run the tests.** Expected: all 4 FAIL, with DID NOT RAISE.

```bash
cd $WT && PYTHONPATH=$WT $PY -m pytest tests/geometry/test_transforms.py -q -p no:cacheprovider -k "umeyama_raises"
```

- [ ] **Step 3: Implement.** Both functions start with the same guard. It replaces the
  `if M < 3` block and both `w_sum < 1e-9` returns:

```python
    M = source.shape[0]
    if M < 3:
        raise ValueError(f"Umeyama alignment needs at least 3 correspondences, got {M}")
    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        raise ValueError("Umeyama alignment got zero total weight")
    w = w / w_sum
```

Add `Raises:` to both docstrings.

- [ ] **Step 4: Check that no LC-path caller exists.**

```bash
cd $WT && rtk proxy git grep -n "umeyama_s" -- collab_splats evals
```

The allowed callers are:
- `transforms.py`
- `bundle_adjustment.py:648` (`_carry_dropped_frames`, which guards <3 points before calling)
- `loop_closure/graph.py:19`, an import only (Task 23 removes it)
- `loop_closure/eval.py`, which moves in Task 12
- `evals/scripts/ba_start_at_gt.py`

Stop if anything else appears.

- [ ] **Step 5: Run gate `G`, then `P1`, then commit** with
  `fix(geometry): umeyama raises on degenerate input`.

### Task 11: Delete `global_alignment.py`

**Files:** delete `collab_splats/geometry/global_alignment.py`. Modify:
- `collab_splats/geometry/__init__.py:28-32,50`
- `collab_splats/pointcloud/feedforward/vggtx.py:28,347`
- `CLAUDE.md:74`
- the package docstring from Task 1

- [ ] **Step 1: Confirm there are no callers.**

```bash
cd $WT && rtk proxy git grep -n "global_alignment\|run_global_alignment" -- collab_splats tests evals configs scripts docs/source CLAUDE.md
```

Expected hits:
- the module itself
- `geometry/__init__.py`
- the two commented-out lines in `vggtx.py`
- `CLAUDE.md:74`

- [ ] **Step 2: Delete and edit.**
  - `git rm collab_splats/geometry/global_alignment.py`.
  - Delete the `run_global_alignment` branch of the lazy `__getattr__` and its `__all__` entry.
  - Delete the two commented-out `vggtx.py` lines.
  - Delete the CLAUDE.md architecture line.
  - Delete the global_alignment line from the package docstring, if Task 1 wrote one.
  - `rtk proxy git grep -n global_alignment -- collab_splats tests CLAUDE.md` must print nothing.
- [ ] **Step 3: Run gate `G` and commit.** Commit these paths with `--only`:
  - the deleted path
  - `collab_splats/geometry/__init__.py`
  - `collab_splats/pointcloud/feedforward/vggtx.py`
  - `CLAUDE.md`

  Message: `refactor(geometry): delete parked global_alignment`.

### Task 12: Move `loop_closure/eval.py` and `edge_trace.py` to `evals/`

**Files:**
- Create: `evals/trajectory_metrics.py`, `evals/pose_graph_diagnostics.py`
- Delete: `collab_splats/geometry/loop_closure/eval.py`, `collab_splats/geometry/loop_closure/edge_trace.py`
- Modify:
  - `collab_splats/geometry/loop_closure/__init__.py`
  - `evals/metrics.py:119`
  - `evals/scripts/eval.py:62-66`
  - `evals/scripts/ba_start_at_gt.py:37`
  - `docs/source/api/geometry.rst:26`
- Tests:
  - `git mv tests/geometry/loop_closure/test_eval_metrics.py tests/evals/test_trajectory_metrics.py`
  - `git mv tests/geometry/loop_closure/test_auc_metric.py tests/evals/test_auc_metric.py`
  - `git mv tests/geometry/loop_closure/test_loop_closure_eval.py tests/evals/test_pose_graph_diagnostics.py`
  - modify `tests/evals/test_compare_loop_edges.py:7`, `tests/evals/test_eval_gt_helpers.py:192`
    and `tests/geometry/loop_closure/test_loop_edge_chain.py:23`

This is a pure move. Function bodies are byte-identical, and only imports change.

- [ ] **Step 1: Split the source.**
  - `evals/trajectory_metrics.py` gets `umeyama_align`, `ate_translation`, `rpe` and
    `auc_at_threshold`, copied verbatim from `eval.py:114-283`.
  - `evals/pose_graph_diagnostics.py` gets these, copied verbatim:
    - `_classify_edges`, `_per_edge_error` and `capture_pose_graph_loss`, from `eval.py:17-112`
    - `compose_slam_chain` and `edge_divergence`, from `edge_trace.py`

  Each new module gets a module docstring in contract shape, and only the imports it uses:

```python
# evals/trajectory_metrics.py
import numpy as np

from collab_splats.geometry.transforms import umeyama_se3, umeyama_sim3

# evals/pose_graph_diagnostics.py
import gtsam
import numpy as np

from collab_splats.geometry.loop_closure.graph import PoseGraph
```

  Read `edge_trace.py`'s imports and carry the ones its two functions use.

- [ ] **Step 2: Delete the old modules and update the package.**
  - `git rm` both modules.
  - In `loop_closure/__init__.py`, delete `from .eval import capture_pose_graph_loss` and the
    `"capture_pose_graph_loss"` entry in `__all__`.
- [ ] **Step 3: Update the callers.** The convention in `evals/` is that `evals/` is on
  `sys.path`, so bare module names are used.
  - `evals/metrics.py:119`: the inline import becomes `from trajectory_metrics import auc_at_threshold`.
    It stays inline, next to its existing comment.
  - `evals/scripts/eval.py:62-66`: `from trajectory_metrics import ate_translation, auc_at_threshold, rpe`.
  - `evals/scripts/ba_start_at_gt.py:37`: `from trajectory_metrics import ate_translation`.
  - Before editing either script, check that it inserts the `evals/` dir into `sys.path`:
    `rtk proxy grep -n "sys.path" evals/scripts/eval.py evals/scripts/ba_start_at_gt.py`.
    If one does not, add
    `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` beside its existing insert.
  - `docs/source/api/geometry.rst`: drop the `loop_closure.eval` automodule block.
- [ ] **Step 4: Update the moved tests and the importers.**
  - Each of the three moved test files and `tests/evals/test_compare_loop_edges.py` get this
    after their imports, then bare imports:
    ```python
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from trajectory_metrics import ate_translation, auc_at_threshold, rpe, umeyama_align  # noqa: E402
    ```
    Import only the names each file uses. Check first whether `tests/evals/conftest.py`
    already inserts the path (`rtk proxy grep -rn "sys.path" tests/evals`). If it does, skip
    the insert.
  - `tests/evals/test_pose_graph_diagnostics.py`:
    - delete `test_capture_pose_graph_loss_importable_from_package` (:189-193) and the NOTE
      comment (:196-199)
    - keep `from collab_splats.geometry.loop_closure import PoseGraph, Submap`
  - `tests/evals/test_eval_gt_helpers.py:192`: the inline import becomes
    `from trajectory_metrics import ...`.
  - `tests/geometry/loop_closure/test_loop_edge_chain.py:23`: add
    `sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evals"))` and import
    `compose_slam_chain, edge_divergence` from `pose_graph_diagnostics`.
- [ ] **Step 5: Verify that nothing still imports the old path.**

```bash
cd $WT && rtk proxy git grep -n "loop_closure.eval\|loop_closure import.*capture_pose_graph_loss\|edge_trace" -- collab_splats tests evals docs/source
```

Expected: no output.

- [ ] **Step 6: Gate.** Run gate `G` and `P1`.
  - Test count in `G` = baseline − 1 (the importable-from-package test). The moved tests
    still run, because `tests/evals` is in `G`.
  - Also check the eval CLI still starts:
    `cd $WT && PYTHONPATH=$WT $PY evals/scripts/eval.py --help`, which must exit 0.
  - Commit every created, deleted and modified path above with
    `refactor(geometry): move trajectory metrics and pose-graph diagnostics to evals`.

### Task 13: `_last_loss_history` → `loss_history`

**Files:**
- `collab_splats/geometry/bundle_adjustment.py:96,212,263-264,420`
- `collab_splats/wrapper/reconstructor.py:1225`
- `evals/scripts/ba_start_at_gt.py:161`
- `tests/geometry/test_bundle_adjustment.py:670,684,688,918-946`

- [ ] **Step 1: Rename.** This is an identifier-only commit, with no other edit.

```bash
cd $WT && rtk proxy git grep -ln "_last_loss_history" -- collab_splats tests evals \
  | xargs sed -i 's/\b_last_loss_history\b/loss_history/g' \
  && rtk proxy git grep -n "_last_loss_history" -- collab_splats tests evals docs/source
```

Expected: the second grep prints nothing. The `refine.json` key `"loss_history"` is
unchanged. Check that `reconstructor.py` still writes `"loss_history": ba.loss_history`.

- [ ] **Step 2: Run gate `G` and commit.** Commit the four files with
  `refactor(geometry): BundleAdjustment.loss_history is public`. The notebook reader is listed
  in the spec's Tutorial impact section and is not edited.

### Task 14: Delete `_UNSET` and the `_optimize` override kwargs

**Files:** `collab_splats/geometry/bundle_adjustment.py:40-43,268-290`,
`tests/geometry/test_bundle_adjustment.py:289-296,380-382,426,640,686,1161`

- [ ] **Step 1: Move each test from a kwarg override to the config.** Every
  `ba._optimize(..., max_reproj_error=X[, lm_steps=Y])` becomes a call without those kwargs,
  on a `BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=X[, lm_steps=Y]))` built for
  that test. For example, :380-382:

```python
ba = BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=None, lm_steps=20))
refined_pts3d, refined_ext, refined_intr = ba._optimize(pts3d, ext, intr, tracks, vis)
```

  Keep every other config field at the value the test used before. If the test built its `ba`
  from a config, add the two fields to that config.

- [ ] **Step 2: Implement.**
  - Delete `_UNSET` and its 2-line comment.
  - Delete the `*, max_reproj_error=..., lm_steps=...` parameters.
  - Replace :288-290 with:

```python
        cfg = self.config
        max_reproj = cfg.max_reproj_error
        n_steps = cfg.lm_steps
```

- [ ] **Step 3: Verify.** `rtk proxy git grep -n "_UNSET\|max_reproj_error=\|lm_steps=" -- collab_splats/geometry tests/geometry evals`
  must show only config constructions. Run gate `G`: the counts equal the previous gate.
  Commit with `refactor(geometry): BA overrides come from the config only`.

### Task 15: `_optimize` raises when <2 frames or points stay active

**Files:** `collab_splats/geometry/bundle_adjustment.py:181,307-313`, `tests/geometry/test_bundle_adjustment.py`

- [ ] **Step 1: Write the failing tests.**

```python
def test_optimize_raises_when_too_few_observations_survive():
    """All-zero visibility leaves no active frame: refine must fail, not return input poses."""
    N, P = 3, 8
    ba = BundleAdjustment(BundleAdjustmentConfig(max_reproj_error=None))
    ext = np.tile(np.eye(3, 4, dtype=np.float32), (N, 1, 1))
    intr = np.tile(np.diag([100.0, 100.0, 1.0]).astype(np.float32), (N, 1, 1))
    with pytest.raises(ValueError, match=r"0 frames, 0 points"):
        ba._optimize(np.zeros((P, 3)), ext, intr, np.zeros((N, P, 2), np.float32), np.zeros((N, P), np.float32))


def test_incremental_steps_skip_single_frame_windows(monkeypatch):
    """increment_size=1 must not hand _optimize a 1-frame window (it would now raise)."""
    seen = []

    def fake_optimize(self, pts3d, ext, intr, tracks, vis):
        seen.append(len(ext))
        return pts3d, ext, intr

    monkeypatch.setattr(BundleAdjustment, "_optimize", fake_optimize)
    N = 4
    ba = BundleAdjustment(BundleAdjustmentConfig(increment_size=1))
    result = SimpleNamespace(
        images=np.zeros((N, 3, 8, 8), np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)),
    )
    intr = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    ba._refine_incremental(result, np.zeros((N, 5, 2)), np.zeros((N, 5)), np.zeros((5, 3)), intr, 1)
    assert seen == [2, 3, 4]
```

  Add `from types import SimpleNamespace` at the top of the test file if it is missing.
  Before running, read `_refine_incremental`'s signature and match the positional order.
  It is `(result, tracks, vis_scores, pts3d_tracks, intrinsics_model, increment_size)`.

- [ ] **Step 2: Run the tests.** Expected: the first FAILS (DID NOT RAISE). The second FAILS
  with `seen == [1, 2, 3, 4]`.
- [ ] **Step 3: Implement.** Replace the early return:

```python
        if len(active_frames) < 2 or len(active_pts) < 2:
            raise ValueError(
                f"BA: too few active frames/points after filtering ({len(active_frames)} frames, "
                f"{len(active_pts)} points); need >= 2 of each"
            )
```

  and the steps line:

```python
        # Step sequence ends at exactly N; a 1-frame window cannot be adjusted, so it is skipped
        steps = sorted({min(k, N) for k in range(increment_size, N + increment_size, increment_size)} - {1})
```

  A 1-frame window always hit the old early return (a single frame is <2 active frames), so
  removing it changes no output.
- [ ] **Step 4: Check the tests that relied on the warning.**
  `rtk proxy grep -n "BA skipped\|too few active" tests/`. Any test asserting the old warning
  or the unrefined return flips to `pytest.raises(ValueError)`.
- [ ] **Step 5: Run gate `G` and commit** with `fix(geometry): BA raises when too few observations survive filtering`.

### Task 16: Narrow the track-cache `except`

**Files:** `collab_splats/geometry/bundle_adjustment.py:128-139`, `tests/geometry/test_bundle_adjustment.py`

- [ ] **Step 1: Write the tests.** Reuse `_make_ff_result_for_ba` and the `fake_extract`
  pattern of `test_tracks_cache_save_load` (:709).

```python
def _cache_ba(tmp_path):
    """A cache-enabled BA whose extractor is counted, plus a result to feed it."""
    result = _make_ff_result_for_ba(3, 8, 8)
    ba = BundleAdjustment(config=BundleAdjustmentConfig(tracks_cache_dir=tmp_path))
    calls = []

    def fake_extract(images, confidence, world_points, max_query_pts, query_frame_num, fine_tracking, device):
        calls.append(1)
        return np.ones((3, 5, 2), np.float32), np.ones((3, 5), np.float32), np.ones((5, 3), np.float32)

    return ba, result, calls, fake_extract


def test_unreadable_track_cache_is_re_extracted(tmp_path):
    """A cache dir that is not a zarr store is rebuilt, not fatal."""
    ba, result, calls, fake_extract = _cache_ba(tmp_path)
    (tmp_path / "tracks.zarr").mkdir()
    (tmp_path / "tracks.zarr" / "zarr.json").write_text("not json")
    with patch("collab_splats.geometry.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        ba._load_or_extract_tracks(result)
    assert calls == [1]


def test_unexpected_track_cache_error_propagates(tmp_path, monkeypatch):
    """A bug inside the cache read is not an unreadable cache: it must surface."""
    ba, result, calls, fake_extract = _cache_ba(tmp_path)
    (tmp_path / "tracks.zarr").mkdir()

    def broken_open(*args, **kwargs):
        raise TypeError("bug")

    monkeypatch.setattr("collab_splats.geometry.bundle_adjustment.zarr.open", broken_open)
    with patch("collab_splats.geometry.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        with pytest.raises(TypeError, match="bug"):
            ba._load_or_extract_tracks(result)
```

  `fake_extract` takes no defaults, which matches Task 17. Tasks 16 and 17 may land in either
  order, because the tests pass every kwarg by name either way. Read the import block at the
  top of the test file before adding any name: `BundleAdjustment`, `BundleAdjustmentConfig`
  and `patch` may be imported inside other tests only. Put missing imports at the top.

- [ ] **Step 2: Run the tests.** Expected: the second FAILS, because the old
  `except Exception` swallows `TypeError`.
- [ ] **Step 3: Implement.** Run the garbage-bytes case in a python one-liner to see the
  exception type zarr 3.1.5 raises. Then catch exactly `(KeyError, ValueError, OSError)`, plus
  that type if it is not a subclass of those:

```python
            except (KeyError, ValueError, OSError) as exc:
                logger.warning("Track cache unreadable (%s), re-extracting: %s", exc, cache_path)
```

- [ ] **Step 4: Run gate `G` and commit** with `fix(geometry): track cache catches only unreadable-store errors`.

### Task 17: Drop the `_extract_tracks_vggsfm` defaults

**Files:** `collab_splats/geometry/bundle_adjustment.py:661-670`

- [ ] **Step 1: Check the callers.**
  `rtk proxy git grep -n "_extract_tracks_vggsfm(" -- collab_splats tests evals`. Every call
  must pass `max_query_pts`, `query_frame_num`, `fine_tracking` and `device`. Add the missing
  kwargs to any test call, using the values the old defaults had: `2048`, `5`, `True`, `None`.
- [ ] **Step 2: Implement.** In the signature, `max_query_pts: int`, `query_frame_num: int`,
  `fine_tracking: bool` and `device: str | None` lose their defaults.
- [ ] **Step 3: Run gate `G` and commit** with `refactor(geometry): _extract_tracks_vggsfm takes config values only`.

### Task 18: BA raises on original-res K; `_scale_intrinsics_to_original` → metrics

**Files:**
- `collab_splats/geometry/bundle_adjustment.py:205-267,475-560`
- `collab_splats/geometry/metrics.py:328-360`
- `evals/scripts/ba_start_at_gt.py:33-36,139-143`
- `tests/geometry/test_bundle_adjustment.py:550,950-1020`

- [ ] **Step 1: Rewrite the guard tests at :950-1020.**

```python
def test_check_model_resolution_accepts_model_res_K():
    K = np.tile(np.array([[10.0, 0, 6.0], [0, 10.0, 4.0], [0, 0, 1]]), (2, 1, 1))
    coords = np.tile(np.array([11, 7, 59, 47, 64, 48], np.float32), (2, 1))
    _check_model_resolution(K, np.zeros((2, 3, 8, 12)), coords)  # 2*cx == W_model: passes


def test_check_model_resolution_rejects_original_res_K():
    K = np.tile(np.array([[40.0, 0, 35.0], [0, 50.0, 27.0], [0, 0, 1]]), (2, 1, 1))
    coords = np.tile(np.array([11, 7, 59, 47, 64, 48], np.float32), (2, 1))
    with pytest.raises(ValueError, match="re-run the pointcloud stage"):
        _check_model_resolution(K, np.zeros((2, 3, 8, 12)), coords)


def test_scale_intrinsics_to_original_known_answer():
    """Crop (11,7)-(59,47) resized to 12x8: sx=0.25, sy=0.2."""
    K = np.array([[10.0, 0, 6.0], [0, 10.0, 4.0], [0, 0, 1]])
    out = _scale_intrinsics_to_original(K, 0.25, 0.2, 11.0, 7.0)
    assert np.allclose(out, [[40.0, 0, 35.0], [0, 50.0, 27.0], [0, 0, 1]])
```

  Imports:
  - `from collab_splats.geometry.bundle_adjustment import _check_model_resolution`
  - `from collab_splats.geometry.metrics import _scale_intrinsics_to_original`

  Delete the old `_scale_intrinsics_to_model` tests.
  Fixture at :550: `original_coords=None` → `np.tile(np.array([0, 0, W, H, W, H], np.float32), (N, 1))`,
  using that fixture's own `W`, `H` and `N`.
- [ ] **Step 2: Run the tests.** Expected: they fail with an ImportError on `_check_model_resolution`.
- [ ] **Step 3: Implement in `bundle_adjustment.py`.** Replace `_scale_intrinsics_to_model`
  (:475-526) with:

```python
def _check_model_resolution(intrinsics: np.ndarray, images: Any, original_coords: np.ndarray) -> None:
    """
    Raise unless K is at model (depth) resolution, the `pointcloud.zarr` contract.

    - model-res K has 2·cx ≈ W_model; original-res K has 2·cx ≈ the crop center in original pixels
    - only a `pointcloud.zarr` written before the model-res K contract fails

    Args:
        intrinsics: (N, 3, 3) stored K.
        images: (N, 3, H, W) model-resolution images.
        original_coords: (N, 6) crop boxes `[tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]`.

    Raises:
        ValueError: K is at original resolution.
    """
    W_model = float(images.shape[-1])
    tl_x, cr_x = float(original_coords[0, 0]), float(original_coords[0, 2])
    cx2 = 2.0 * float(intrinsics[0, 0, 2])
    if abs(cx2 - W_model) > abs(cx2 - (tl_x + cr_x)):
        raise ValueError(
            "BA: intrinsics are at original resolution, not model resolution — this "
            "pointcloud.zarr predates the model-res K contract; re-run the pointcloud stage"
        )
```

  In `refine`:
  - Replace the `_scale_intrinsics_to_model` call and its 2-line comment with:

```python
        # VGGSfM tracks live on the model grid, so K must too (the pointcloud.zarr contract)
        _check_model_resolution(result.intrinsics, result.images, result.original_coords)
        intrinsics_model = result.intrinsics
```

  - Delete the rescale-back block (:256-261) and use
    `intrinsics=refined_intrinsics_model` in the `replace(...)` call.
  - This is output-identical on model-res K: the old guard returned `sx = sy = 1.0` and
    `tl = 0.0`, and `x / 1.0 + 0.0` is exact in IEEE arithmetic.

  Move `_scale_intrinsics_to_original` (:529-560) into `metrics.py`, above
  `compute_photometric_ncc`. Keep its body. Cut its docstring to a summary line, 2 bullets,
  `Args:` and `Returns:`. Delete its inline import in `metrics.py` (:331).
- [ ] **Step 4: Update `ba_start_at_gt.py`.**
  - The import becomes `_check_model_resolution`.
  - :139-143 become:

```python
    _check_model_resolution(result.intrinsics, result.images, result.original_coords)
    intr_model = result.intrinsics
```

  Read the lines around :139 first. If the script later used `sx`, `sy` or `tl`, those uses go too.
- [ ] **Step 5: Verify.** `rtk proxy git grep -n "_scale_intrinsics_to_model\|_scale_intrinsics_to_original" -- collab_splats tests evals`
  must show only `metrics.py` and its test. Run gate `G` and commit with
  `fix(geometry): BA raises on original-resolution K instead of rescaling`.


### Task 19: `verification.py` — inline `DEFAULT_OVERLAP`, publish `pair_pose_errors`

**Files:**
- `collab_splats/geometry/verification.py:30-35,94,161,295`
- `evals/scripts/eval_verification.py:29,180`

Line numbers from here on are pre-Round-1. Locate each anchor by grep, not by number.

- [ ] **Step 1: Implement.**
  - Delete the `DEFAULT_OVERLAP` constant and its comment. If the `# Constants` section is
    then empty, delete the divider too.
  - The signature becomes `overlap: int = 10`.
  - Rename `_pair_pose_errors` → `pair_pose_errors` at the definition and at :295.
  - In `eval_verification.py`, rename the import (:29) and the call (:180).
- [ ] **Step 2: Verify.** `rtk proxy git grep -n "_pair_pose_errors\|DEFAULT_OVERLAP" -- collab_splats tests evals`
  must print nothing. `pair_pose_errors` is public now, so it needs a docstring that
  passes the contract (summary line, `Args:`, `Returns:`).
- [ ] **Step 3: Run gate `G` and commit** with
  `refactor(geometry): verification overlap default inline; pair_pose_errors public`.

### Task 20: `extract_photometric` raises

**Files:**
- `collab_splats/geometry/metrics.py` (`extract_photometric`, ~:476-497)
- `tests/geometry/test_metrics.py` (or whichever file holds the `extract_photometric`
  tests: `rtk proxy git grep -ln "extract_photometric" -- tests`)

- [ ] **Step 1: Write the failing test.**

```python
def test_extract_photometric_propagates_measurement_errors(tmp_path, monkeypatch):
    """A failure inside the correlation raises; only missing images return unavailable."""
    from collab_splats.geometry import metrics

    monkeypatch.setattr(metrics.frames, "frame_paths", lambda d: [d / "frame_000000.png"])
    monkeypatch.setattr(metrics.frames, "read_frames", lambda d: np.zeros((1, 4, 4, 3), np.uint8))

    def boom(*args, **kwargs):
        raise RuntimeError("correlation failed")

    monkeypatch.setattr(metrics, "compute_photometric_ncc", boom)
    result = SimpleNamespace(
        depth=np.ones((1, 4, 4)), intrinsics=np.tile(np.eye(3), (1, 1, 1)),
        extrinsics=np.tile(np.eye(4), (1, 1, 1)), original_coords=np.zeros((1, 6)),
    )
    with pytest.raises(RuntimeError, match="correlation failed"):
        metrics.extract_photometric(result, tmp_path, 1)
```

  Add `from types import SimpleNamespace` at the top of the test file if missing.
- [ ] **Step 2: Run it.** Expected: FAIL. The call returns `{"available": False, ...}`
  instead of raising.
- [ ] **Step 3: Implement.**
  - Delete the `try:` / `except Exception` wrapper. The body dedents one level.
  - Docstring `Returns:` drops "or the measurement raises". A `Raises:` section is not
    needed: nothing is raised on purpose.
- [ ] **Step 4: Run gate `G`** (the new test passes, and the existing no-images test still
  returns `available: False`) **and commit** with
  `fix(geometry): photometric measurement failures raise instead of disabling the channel`.

### Task 21: Metrics report — running-error raise, direct epipolar keys, `rel_thresh` kwarg

**Files:**
- `collab_splats/geometry/metrics.py` (`build_reconstruction_quality_report`: ~:570, ~:599, ~:663-672)
- the test file holding `test_running_error_says_unavailable_rather_than_shipping_empty_arrays`
  (`rtk proxy git grep -ln "test_running_error_says_unavailable" -- tests`)

- [ ] **Step 1: Read the existing helpers.** In that test file, `_write_tiny_scene(tmp_path, names)`
  writes a minimal scene and `_build` calls `build_reconstruction_quality_report`. `_build`
  hardcodes absent paths for verification.json and images, so the new tests call
  `build_reconstruction_quality_report` directly.
- [ ] **Step 2: Write the failing tests.** Adapt the zarr path to what `_write_tiny_scene`
  returns (read it first).

```python
def test_epipolar_row_missing_num_inliers_raises(tmp_path):
    """verify always writes num_matches and num_inliers; a row without one is corrupt."""
    zarr_path = _write_tiny_scene(tmp_path, ["frame_000000", "frame_000001"])
    vj = tmp_path / "verification.json"
    vj.write_text(json.dumps({"pair_stats": [
        {"idx1": 0, "idx2": 1, "num_matches": 10, "rot_error_deg": 1.0},
    ]}))
    with pytest.raises(KeyError, match="num_inliers"):
        build_reconstruction_quality_report(
            zarr_path, vj, tmp_path / "no_images", tmp_path / "report.json", "vggtx"
        )


def test_available_channel_without_rows_raises(tmp_path, monkeypatch):
    """A channel that claims available must ship its rows; a missing key is a bug, not zero."""
    from collab_splats.geometry import metrics

    zarr_path = _write_tiny_scene(tmp_path, ["frame_000000", "frame_000001"])
    monkeypatch.setattr(metrics, "compute_depth_error", lambda *a, **k: {"available": True})
    with pytest.raises(KeyError, match="pair_directions"):
        build_reconstruction_quality_report(
            zarr_path, tmp_path / "none.json", tmp_path / "no_images",
            tmp_path / "report.json", "vggtx",
        )


def test_rel_thresh_reaches_depth_confidence(tmp_path, monkeypatch):
    """rel_thresh is a kwarg on the report, threaded to the dense confidence pass."""
    from collab_splats.pointcloud.feedforward import base as ff_base

    zarr_path = _write_tiny_scene(tmp_path, ["frame_000000", "frame_000001"])
    seen = {}
    real = ff_base.compute_multiview_depth_confidence

    def spy(*args, **kwargs):
        seen["rel_thresh"] = kwargs["rel_thresh"]
        return real(*args, **kwargs)

    monkeypatch.setattr(ff_base, "compute_multiview_depth_confidence", spy)
    build_reconstruction_quality_report(
        zarr_path, tmp_path / "none.json", tmp_path / "no_images",
        tmp_path / "report.json", "vggtx", rel_thresh=0.2,
    )
    assert seen["rel_thresh"] == 0.2
```

  The spy works because the import of `compute_multiview_depth_confidence` stays inline
  in the function (an import cycle), so it resolves the patched module attribute at call time.
- [ ] **Step 3: Run them.** Expected: all three FAIL. The first two return a report; the third
  fails with a TypeError on `rel_thresh`.
- [ ] **Step 4: Implement.**
  - Signature: `..., backend: str, rel_thresh: float = 0.05) -> dict:`. Add
    `rel_thresh: relative depth tolerance for the dense multiview pass.` under `Args:`.
  - The confidence call passes `rel_thresh=rel_thresh`.
  - Epipolar row: `n_m, n_i = s["num_matches"], s["num_inliers"]`.
  - Running error:

```python
    running = {
        name: (
            _running_error(m[rows_key], key)
            if m["available"]
            else {"available": False, "reason": m["reason"]}
        )
        for name, rows_key, key, m in (
            ("depth", "pair_directions", "median_rel_depth_error", depth_m),
            ("epipolar", "pairs", "rot_error_deg", epipolar_m),
        )
    }
```

  - Every unavailable block in this module already carries `"reason"`. Confirm with
    `rtk proxy grep -n '"available": False' collab_splats/geometry/metrics.py`: each hit
    must have a `"reason"` key on the same dict.
- [ ] **Step 5: Run gate `G`.** The three new tests and
  `test_running_error_says_unavailable_rather_than_shipping_empty_arrays` pass. Commit with
  `fix(geometry): quality report indexes its own rows; rel_thresh is a kwarg`.

### Task 22: `PoseGraph.optimize` catches `RuntimeError` only; `decompose_camera` raises `ValueError`

**Files:**
- `collab_splats/geometry/loop_closure/graph.py:76-96,196-206`
- `tests/geometry/loop_closure/test_graph.py`

- [ ] **Step 1: Write the failing tests** (append to `test_graph.py`; move its mid-file
  imports to the top while you are there, since they are in the file you touch).

```python
def test_decompose_camera_rejects_bad_shape():
    with pytest.raises(ValueError, match="expected"):
        decompose_camera(np.ones((2, 4)))


def _optimizer_raising(exc):
    def build(*args, **kwargs):
        raise exc
    return build


def test_optimize_keeps_initial_values_on_gtsam_runtime_error(monkeypatch):
    from types import SimpleNamespace

    from collab_splats.geometry.loop_closure import graph as graph_mod

    pg = PoseGraph()
    H0 = _translate_H(0.3, 0, 0)
    pg.add_homography(0, H0)
    before = pg.get_homography(0)
    monkeypatch.setattr(graph_mod, "gtsam", SimpleNamespace(
        LevenbergMarquardtParams=lambda: None,
        LevenbergMarquardtOptimizer=_optimizer_raising(RuntimeError("indeterminant system")),
    ))
    pg.optimize()
    monkeypatch.undo()
    assert np.array_equal(pg.get_homography(0), before)


def test_optimize_propagates_non_gtsam_errors(monkeypatch):
    from types import SimpleNamespace

    from collab_splats.geometry.loop_closure import graph as graph_mod

    pg = PoseGraph()
    pg.add_homography(0, _identity_H())
    monkeypatch.setattr(graph_mod, "gtsam", SimpleNamespace(
        LevenbergMarquardtParams=lambda: None,
        LevenbergMarquardtOptimizer=_optimizer_raising(TypeError("bad argument")),
    ))
    with pytest.raises(TypeError, match="bad argument"):
        pg.optimize()
```

  `get_homography` also reads `gtsam`, which is why the first test calls `monkeypatch.undo()`
  before reading back. Add `import pytest` at the top.
- [ ] **Step 2: Run them.** Expected: the decompose test fails with AssertionError, and the
  TypeError test fails because the error is swallowed.
- [ ] **Step 3: Implement.**
  - `decompose_camera`: `if P.shape != (3, 4): raise ValueError(f"expected (3,4) after strip, got {P.shape}")`.
    Add a `Raises:` entry.
  - `optimize`: `except RuntimeError as exc:`. The warning text and the kept initial values stay.
- [ ] **Step 4: Run gate `G`, then `P1`.** Commit with
  `fix(geometry): pose-graph optimize catches GTSAM RuntimeError only; decompose_camera raises ValueError`.

### Task 23: `graph.py` — drop the umeyama re-export, the `K_*` defaults; `min_conf_points` kwarg

**Files:**
- `collab_splats/geometry/loop_closure/graph.py:19,153,289,294,367-368,440,473-545,571-594`
- `tests/geometry/loop_closure/test_loop_edge_chain.py:233,247,261,262,271,282,288,298`
- `tests/geometry/loop_closure/test_graph.py`

- [ ] **Step 1: Confirm the umeyama import has no reader.** After Task 12 moved `eval.py`,
  `rtk proxy git grep -n "graph import.*umeyama\|graph\.umeyama" -- collab_splats tests evals`
  prints nothing. Delete the import at :19.
- [ ] **Step 2: Write the failing test** (in `test_graph.py`).

```python
def _submap_pair_with_conf_split(n_high: int):
    """Two submaps whose overlap scale differs between high- and zero-confidence points."""
    rng = np.random.default_rng(7)
    k, P = 2, 200
    base_pts = rng.standard_normal((P, 3)) + np.array([0, 0, 5.0])
    conf = np.zeros((k, P), np.float32)
    conf[:, :n_high] = 50.0

    s0 = _make_real_submap(0, k=k, frame_start=0)
    s1 = _make_real_submap(1, k=k, frame_start=k - 1)
    s0.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s0.world_points = np.tile(base_pts, (k, 1, 1)).astype(np.float32)
    curr = base_pts.copy()
    curr[:n_high] *= 0.5   # high-conf points say scale 2
    curr[n_high:] *= 0.25  # the rest say scale 4
    s1.world_points = np.tile(curr, (k, 1, 1)).astype(np.float32)
    s0.world_points_conf = conf
    s1.world_points_conf = conf.copy()
    return s0, s1


def _drive(pg, submaps):
    for s in submaps:
        pg.add_submap(s, 1, 25.0, "rotation_only")
        pg.optimize()
    return pg.extract_extrinsics(3)


def test_min_conf_points_sets_the_confidence_mask_floor():
    """50 high-conf points: below the default floor (100) all points count; above 10 only they do."""
    s0, s1 = _submap_pair_with_conf_split(n_high=50)
    default = _drive(PoseGraph(), [s0, s1])
    explicit = _drive(PoseGraph(min_conf_points=100), [s0, s1])
    lowered = _drive(PoseGraph(min_conf_points=10), [s0, s1])
    assert np.array_equal(default, explicit)
    assert not np.allclose(default, lowered)
```

  Read `PoseGraph.add_submap` before running. If the first submap needs its own positional
  args or `extract_extrinsics` takes a different count, fix the helper; the assertion stays.
  If `default` and `lowered` come out equal, the fixture does not reach the mask: print
  `joint_mask.sum()` once from a scratch copy to see why, and fix the fixture, never the
  assertion.
- [ ] **Step 3: Run it.** Expected: FAIL with TypeError on `min_conf_points`.
- [ ] **Step 4: Implement.**
  - `PoseGraph.__init__(self, min_conf_points: int = 100) -> None:` stores
    `self.min_conf_points = min_conf_points`. Docstring `Args:` gains
    `min_conf_points: fewest confident points before the scale fit uses the confidence mask.`
  - `add_submap` :289/:294 read `self.min_conf_points`.
  - `_lc_anchor_scale` gains a trailing `min_conf_points: int` parameter (no default),
    used at :539/:541. `add_loop_edge` (:367-368) passes `self.min_conf_points`.
  - Delete `_MIN_CONF_POINTS` (:440).
  - `_loop_chain_relatives`: `K_q`, `K_lc0`, `K_lc1`, `K_d` become required `np.ndarray`
    parameters, and the four `eye if ... is None` lines go.
  - Tests: `test_loop_edge_chain.py:233,247` pass `np.eye(4)` four times; the six
    `_lc_anchor_scale` calls add a trailing `100`.
- [ ] **Step 5: Run gate `G`, then `P1`.** P1 must print `PARITY OK`. It cannot see
  `min_conf_points` (always 100), which is why Step 2 exists. Commit with
  `refactor(geometry): PoseGraph min_conf_points kwarg; loop-chain K args required`.

### Task 24: `map.py` — delete the test-only readers, inline `get_corrected_extrinsics`

**Files:**
- `collab_splats/geometry/loop_closure/map.py:21,32,37,42,70`
- `collab_splats/geometry/loop_closure/wrapper.py:603`
- `tests/geometry/loop_closure/test_map.py:30,38,46-63,154-164`
- `tests/geometry/loop_closure/test_wrapper.py:635,718`

- [ ] **Step 1: Check callers.**
  `rtk proxy git grep -n "get_submap\b\|_keys()\|get_largest_key\|get_latest_submap\|get_corrected_extrinsics" -- collab_splats tests evals`.
  Only `map.py`, `wrapper.py:603`, `test_map.py` and `test_wrapper.py` may appear.
  Anything else: stop and report.
- [ ] **Step 2: Implement.**
  - Delete `get_submap`, `_keys`, `get_largest_key`, `get_latest_submap` and `get_corrected_extrinsics`.
  - `wrapper.py:603`: `extrinsics = self.graph.extract_extrinsics(n_frames)`.
  - `test_wrapper.py:635,718`: `lc.map.get_corrected_extrinsics(lc.graph, n)` →
    `lc.graph.extract_extrinsics(n)`, keeping each site's own frame-count argument.
  - Delete the tests at `test_map.py:30,38,46-63,154-164` that only exercise the deleted readers.
- [ ] **Step 3: Run gate `G`, then `P1`.** Commit with
  `refactor(geometry): GraphMap drops test-only readers`.

### Task 25: `submap.py` — delete `raw_outputs`, the test-only readers; inline the confidence filter

**Files:**
- `collab_splats/geometry/loop_closure/submap.py:21,49-84,88-94,136`
- `collab_splats/geometry/loop_closure/wrapper.py:269,290-298`
- `tests/geometry/loop_closure/test_submap_reprojection.py`
- `tests/geometry/loop_closure/test_hw_formula.py:37`, `test_pgo_parity.py:44`,
  `tests/pointcloud/test_pose_extraction.py:35`, `tests/geometry/loop_closure/test_wrapper.py:844-923`

- [ ] **Step 1: Check callers.**
  `rtk proxy git grep -n "raw_outputs=\|\.raw_outputs\b" -- collab_splats tests evals` and
  `rtk proxy git grep -n "get_world_points\b\|get_poses_world\b\|filter_data_by_confidence" -- collab_splats tests evals`.
  - `raw_outputs` on a `Submap` appears only in the files listed above.
  - `LoopClosure.raw_outputs` (the property at wrapper :167) and `base.raw_outputs` are a
    different name. They stay.
- [ ] **Step 2: Implement.**
  - Delete the `raw_outputs` field.
  - `wrapper.py`: delete `raw_outputs=raw,` from the `Submap(...)` call, and delete the
    5-line comment plus `submap.raw_outputs = None`. `raw` and `raw_lc` go out of scope
    at the end of the window iteration as before. The dense fields and `frames` are unchanged.
  - Delete `get_world_points` and `get_poses_world`, and delete `test_submap_reprojection.py`
    (every test in it exercises one of the two).
  - `get_points_colors` returns
    `self.colors[skip_first:][self.conf[skip_first:] > self.conf_threshold].reshape(-1, 3)`,
    and `filter_data_by_confidence` goes.
  - Drop `raw_outputs={}` from the four test constructors.
  - `test_lc_frees_raw_outputs_after_unproject_keeps_frames`: delete the
    `s.raw_outputs is None` assert, keep the frames assert, and rename the test
    `test_lc_submaps_keep_frames_after_unproject`.
- [ ] **Step 3: Run gate `G`, then `P1`.** Commit with
  `refactor(geometry): Submap drops raw_outputs and test-only readers`.

### Task 26: `matching.py` — delete `translation_jump_check`; finite guard in its place

**Files:**
- `collab_splats/geometry/loop_closure/matching.py:39,67-77,99-175`
- `collab_splats/geometry/loop_closure/__init__.py`
- `collab_splats/geometry/loop_closure/wrapper.py:42,70,336-365`
- `tests/geometry/loop_closure/test_translation_jump.py`
- `tests/pointcloud/feedforward/test_verify_lc_data.py:277`
- `tests/geometry/loop_closure/test_loop_closure.py:54,76,118,148`,
  `test_loop_closure_integration.py:68`

`max_jump_ratio` defaults to `inf` and nothing sets it. At `inf`, `ratio < inf` is False
only when the ratio is NaN or inf, which happens only for a non-finite `lc_rel`. A finite
guard on `lc_rel` keeps exactly that rejection. It does not re-check the odometry poses;
the old path could raise inside `inv(T_odom)` on a singular pose, which the guard no longer
reaches. Flagged to the user at the plan STOP.

- [ ] **Step 1: Write the failing test** (in `test_wrapper.py`, next to
  `_run_lc_harness_with_timing`).

```python
def test_lc_rejects_non_finite_loop_pose():
    """A NaN loop relative is rejected before it reaches the graph."""
    from collab_splats.geometry.loop_closure.matching import LoopMatch
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig

    H = W = 32
    n_frames = 9
    cfg = LoopClosureConfig(submap_size=3, submap_overlap=1, min_submap_gap=0)

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        return _make_raw_nontrivial(sz, H, W)

    def fake_verify(q_frame, d_frame, verify_match_ratio=None):
        poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        poses[1, 0, 3] = np.nan
        wp = np.zeros((2, H, W, 3), np.float32)
        return True, {"poses": poses, "world_points": wp, "conf": np.full((2, H, W), 100.0, np.float32)}

    def fake_find(submap, past, *args, **kwargs):
        if past:
            return [LoopMatch(similarity_score=0.1, query_submap_id=submap.submap_id,
                              detected_submap_id=0, query_frame_idx=1, detected_frame_idx=1)]
        return []

    base = MagicMock()
    base.max_points = 500_000
    base.views = torch.full((n_frames, 3, H, W), 0.5)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.zeros((n_frames, 6), dtype=np.float32)
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._verify_loop_candidate = fake_verify

    lc = LoopClosure(base, config=cfg)
    with (
        patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as mock_retrieval,
        patch("collab_splats.geometry.loop_closure.wrapper.find_loop_closures", side_effect=fake_find),
    ):
        mock_retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        lc.run_inference()

    assert lc._last_lc_submaps == []
    assert np.isfinite(base.outputs.extrinsics).all()
```

- [ ] **Step 2: Run it on the unchanged code.** Expected: PASS. The old check rejects
  NaN at `inf` too. This pins the behavior the refactor must keep.
- [ ] **Step 3: Implement.**
  - `matching.py`: delete `translation_jump_check` and its divider; `LoopMatchQueue` and
    `find_loop_closures` take `nms_frame_distance: int` with no default.
  - `loop_closure/__init__.py`: drop `translation_jump_check` from the imports and `__all__`.
  - `wrapper.py`: the import becomes `from .matching import find_loop_closures`; delete
    `max_jump_ratio` from `LoopClosureConfig`, and `import math` if nothing else uses it.
    The jump block becomes:

```python
                # Reject a loop whose relative pose is not finite
                if not np.isfinite(lc_rel).all():
                    console.log(
                        f"  ✗ Loop rejected (non-finite pose): "
                        f"submap {match.query_submap_id} → {match.detected_submap_id}"
                    )
                    match.reject_reason = "non_finite_pose"
                else:
                    match.accepted = True
                    match.reject_reason = None
                    console.log(
                        f"  ↩ Loop: submap {match.query_submap_id} → {match.detected_submap_id}"
                        f"  dist={match.similarity_score:.3f}"
                    )
```

    `rtk proxy git grep -n "jump_ratio" -- collab_splats evals tests` must then print only
    `vggt_omega.py` (Task 27 removes it).
  - Tests: delete `test_translation_jump.py` and the `translation_jump_check` patch at
    `test_verify_lc_data.py:277` (keep the rest of that test). Pass `nms_frame_distance=0`
    at `test_loop_closure.py:54,76,118,148` and `test_loop_closure_integration.py:68`.
  - `"intrinsic"` stubs at `test_feedforward_lc_state.py:37` and `test_verify_lc_data.py:226`
    become `"intrinsics"` here only if Task 28 lands in the same session; otherwise leave them
    for Task 28.
- [ ] **Step 4: Run gate `G`, then `P1`.** Check `reject_reason` values in the P1 capture:
  if `lc_parity.py` records them, the saved baseline says `jump_ratio` for none of them (no
  jump rejection at `inf` on finite fixtures). Commit with
  `refactor(geometry): replace the disabled translation-jump check with a finite-pose guard`.

### Task 27: `wrapper.py` config — `lc_threshold_l2`, `getattr` fallback, `_trim_forward_outputs`, `viz_max_points`

**Files:**
- `collab_splats/geometry/loop_closure/wrapper.py:48-78,92-118,121-139,305,458`
- `collab_splats/pointcloud/feedforward/vggt_omega.py:141-145`
- `tests/geometry/loop_closure/test_loop_closure.py:50`
- `tests/geometry/loop_closure/test_verify_threshold_resolution.py:30,46`
- `tests/geometry/loop_closure/test_wrapper.py:11,314-351`

- [ ] **Step 1: Write the failing test** (in `test_wrapper.py`, after `test_lc_loop_pushes_to_viz_when_set`).

```python
def test_viz_max_points_caps_the_viewer_push():
    """LoopClosureConfig.viz_max_points is the per-submap viewer point cap."""
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig

    H = W = 32
    sm = Submap(
        submap_id=0,
        frames=torch.zeros(2, 3, H, W),
        poses=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        retrieval_vectors=np.zeros((2, 8), dtype=np.float32),
        image_paths=["a.jpg", "b.jpg"],
        frame_start=0,
    )
    sm.set_dense_points(
        np.random.RandomState(0).rand(2, H, W, 3).astype(np.float32),
        np.zeros((2, H, W, 3), np.uint8),
        np.full((2, H, W), 100.0, np.float32),
    )

    class _SizedViz(_RecordingViz):
        def add_points(self, name, points, colors, **kwargs):
            self.n_points = points.shape[0]

    lc = LoopClosure(MagicMock(), config=LoopClosureConfig(viz_max_points=100))
    lc.graph.add_submap(sm, 1)
    lc.graph.optimize()
    lc.viz = _SizedViz()
    lc._viz_push_submap(sm)
    assert lc.viz.n_points == 100
```

  Read `PoseGraph.add_submap` and `Submap.set_dense_points` first and match their argument
  order. `sm.points` must not be None after `set_dense_points`, or the push returns early.
- [ ] **Step 2: Run it.** Expected: FAIL with TypeError on `viz_max_points`.
- [ ] **Step 3: Implement.**
  - `LoopClosureConfig`: add `viz_max_points: int = 50000  # viewer per-submap point cap`;
    delete the `lc_threshold_l2` property.
  - :305 passes `cfg.lc_retrieval_threshold`; :458 passes `max_points=self.config.viz_max_points`.
  - `__init__`: `verify_match_ratio=base.default_verify_match_ratio`. The 4-line comment
    above it shrinks to one line: `# None resolves to the creator's per-model calibration`.
    The config comment at :55-56 drops "(fallback 0.85)".
  - Delete `_trim_forward_outputs` (:92-118).
  - `vggt_omega.py`: delete `default_max_jump_ratio` and its comment line (:141, :145).
  - Tests: delete `test_loop_closure.py:50` (the `lc_threshold_l2` assert, or its whole test if
    that is all it checks), the two 0.85-fallback tests in
    `test_verify_threshold_resolution.py` (:30, :46), and the `_trim_forward_outputs` import (:11)
    and tests (:314-351) in `test_wrapper.py`.
- [ ] **Step 4: Verify.**
  `rtk proxy git grep -n "lc_threshold_l2\|_trim_forward_outputs\|default_max_jump_ratio\|max_jump_ratio" -- collab_splats tests evals configs`
  prints nothing.
- [ ] **Step 5: Run gate `G`, then `P1`.** Commit with
  `refactor(geometry): LoopClosure config drops dead knobs; viz_max_points kwarg`.

### Task 28: `wrapper.py` — raise instead of the unreachable fallbacks

**Files:**
- `collab_splats/geometry/loop_closure/wrapper.py:244-255,625-645`
- `tests/pointcloud/feedforward/test_feedforward_lc_state.py:37`
- `tests/pointcloud/feedforward/test_verify_lc_data.py:226`
- `tests/geometry/loop_closure/test_wrapper.py:924-960`

- [ ] **Step 1: Prove the two fallbacks unreachable.**
  - `rtk proxy git grep -n "def _lc_collate_outputs\|\"intrinsics\"\]\s*=\|\[\"intrinsics\"\]" -- collab_splats/pointcloud/feedforward`:
    every LC-capable creator's `_forward` / `_lc_collate_outputs` writes `"intrinsics"`.
  - Every creator's LC window is a tensor or a list of `{"img": ...}` dicts
    (`rtk proxy git grep -n "def _forward" -- collab_splats/pointcloud/feedforward`, then read each).
  - If either proof fails, a backend really emits that shape: stop and report. Never keep a
    fallback that invents intrinsics or frames (user decision 2026-09-25); the fix belongs in
    that backend's output.
- [ ] **Step 2: Implement.**

```python
        intrinsics = raw_lc["intrinsics"]

        # Stack the window's frames to (K, C, H, W)
        if hasattr(window, "cpu"):
            frames_cpu = window.cpu()
        elif isinstance(window, list) and window and isinstance(window[0], dict) and "img" in window[0]:
            frames_cpu = torch.cat([v["img"].cpu() for v in window], dim=0)
        else:
            raise TypeError(f"LC window must be a tensor or a list of {{'img': ...}} dicts, got {type(window).__name__}")
```

  - `_assemble_result`: delete the `getattr(self.base, "model_height"/"model_width", 0)`
    block and its comment. The fail-fast condition becomes
    `if model_height is None or points.shape[0] == 0 or model_width == 0 or model_height == 0:`.
    The message is unchanged.
  - Stubs: `"intrinsic"` → `"intrinsics"` at `test_feedforward_lc_state.py:37` and
    `test_verify_lc_data.py:226` (skip if Task 26 already did it).
  - `test_assemble_result_raises_on_empty_cloud`: delete `base.model_width = 0` and
    `base.model_height = 0`. With them gone, `MagicMock` would have answered `int(...) == 1`
    through the old `getattr`, so this test now pins that LC with no dense submap still
    raises the same `ValueError` from the `model_height is None` arm.
- [ ] **Step 3: Run gate `G`, then `P1`.** Commit with
  `fix(geometry): LoopClosure raises on malformed backend output instead of placeholder data`.

### Task 29: `wrapper.py` — narrow the excepts, drop the outer re-upload guard, one loop counter

**Files:**
- `collab_splats/geometry/loop_closure/wrapper.py:438-500,518-525,528-533,564-567`
- `tests/geometry/loop_closure/test_wrapper.py`

- [ ] **Step 1: Pin the viewer exception types.** Read `collab_splats/wrapper/viewer.py`
  (or wherever `add_points` / `add_frustum` / `add_lines` live:
  `rtk proxy git grep -n "def add_frustum" -- collab_splats`). The narrowed tuple is
  `(ValueError, ZeroDivisionError, RuntimeError, OSError)`, which covers viser's
  connection errors (`OSError` subclasses) and numpy shape errors. If the viewer raises
  another named type on purpose, add it and say so in the commit body.
- [ ] **Step 2: Write the failing tests.**

```python
class _FailingViz(_RecordingViz):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def add_lines(self, name, segments, **kwargs):
        raise self.exc


def _accepted_match():
    from collab_splats.geometry.loop_closure.matching import LoopMatch

    m = LoopMatch(similarity_score=0.1, query_submap_id=0, detected_submap_id=0,
                  query_frame_idx=0, detected_frame_idx=0)
    m.accepted = True
    return m


def _lc_with_one_submap():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    sm = _make_real_submap_for_viz()
    lc = LoopClosure(MagicMock())
    lc.graph.add_submap(sm, 1)
    lc.graph.optimize()
    return lc, sm


def test_viz_draw_loops_swallows_viewer_errors():
    lc, sm = _lc_with_one_submap()
    lc.viz = _FailingViz(OSError("socket closed"))
    lc._viz_draw_loops([_accepted_match()], [sm])  # logged, not raised


def test_viz_draw_loops_propagates_programming_errors():
    lc, sm = _lc_with_one_submap()
    lc.viz = _FailingViz(AttributeError("no such method"))
    with pytest.raises(AttributeError):
        lc._viz_draw_loops([_accepted_match()], [sm])


def test_dino_salad_load_failure_falls_back_to_full_inference():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig

    base = MagicMock()
    base.views = torch.zeros(6, 3, 8, 8)
    lc = LoopClosure(base, config=LoopClosureConfig(submap_size=3))
    with patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as r:
        r.get.side_effect = OSError("weights missing")
        lc.run_inference()
    base._forward.assert_called_once()


def test_dino_salad_unexpected_error_propagates():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig

    base = MagicMock()
    base.views = torch.zeros(6, 3, 8, 8)
    lc = LoopClosure(base, config=LoopClosureConfig(submap_size=3))
    with patch("collab_splats.geometry.loop_closure.wrapper.BaseRetrievalExtractor") as r:
        r.get.side_effect = TypeError("bad kwarg")
        with pytest.raises(TypeError, match="bad kwarg"):
            lc.run_inference()
```

  `_make_real_submap_for_viz` is the dense submap built in Task 27's
  `test_viz_max_points_caps_the_viewer_push`: lift that construction into a module-level
  helper `_make_real_submap_for_viz()` and have both tests use it.
  Check `run_inference` routes to `_run_lc_loop` when `views` holds ≥ `submap_size` frames
  (`_enough_frames`); if the fallback test calls `base.run_inference` instead, raise
  `views` until it does not.
- [ ] **Step 3: Run them.** Expected: the two `propagates` tests FAIL (swallowed by
  `except Exception`); the two fallback tests PASS already.
- [ ] **Step 4: Implement.**
  - `_viz_push_submap`, `_viz_draw_loops`:
    `except (ValueError, ZeroDivisionError, RuntimeError, OSError) as e:`. Warning text unchanged.
  - `_viz_reupload_all`: delete the outer `try`/`except` and its 2-line comment; the loop stays.
  - DINO-SALAD load: `except (ImportError, OSError, RuntimeError) as e:`. Warning and fallback unchanged.
  - Counter: replace `loops_found` and `verified` with one `loops = 0`, then
    `loops += len(window_lc_submaps)` and `pbar.set_postfix(loops=loops)`.
- [ ] **Step 5: Run gate `G`, then `P1`.** Commit with
  `fix(geometry): LoopClosure fallbacks catch named errors only`.

### Task 29b: Drop `scale_method` `se3` and `pairwise_dist`

Added after the plan was written (user decision 2026-09-25; spec amended in `62c00e6d`).

- `graph.py`: collapse `add_submap` and `_lc_anchor_scale` to the `rotation_only` path;
  delete `_estimate_scale_pairwise_dist`; unknown `scale_method` raises `ValueError` at
  `add_submap`, `add_loop_edge` and `LoopClosureConfig.__post_init__`.
- `wrapper.py` `Literal`, `configs/loop_closure.yaml`, `evals/scripts/eval.py` choices,
  `evals/README.md`, `tests/geometry/loop_closure/_helpers.py` default, and the
  `test_pose_graph_incremental.py` parametrize follow.
- P1 runs `$SP/lc_parity_t29b.py`: `SCALE_METHODS = ("rotation_only", "none")`, the
  baseline filtered to kept methods, plus a check that baseline `se3` equals `rotation_only`.
- Commit `refactor(geometry): LoopClosure scale_method keeps rotation_only and none only`.

### Task 30: Annotations

**Files:** every file in `collab_splats/geometry/`.

- [ ] **Step 1: List the gaps.** `cd $WT && $PY $SP/contract_hits.py $(git ls-files 'collab_splats/geometry/*.py')`
  and keep only the "missing annotation" hits.
- [ ] **Step 2: Add annotations** to every flagged parameter and return. Use the types the
  body already implies; `Any` only where the value is a backend-specific object
  (`base`; `graph` on `Submap` methods becomes `"PoseGraph"` under `TYPE_CHECKING`).
  An import added under `TYPE_CHECKING` is the one allowed non-annotation change; the proof
  below drops `if TYPE_CHECKING:` blocks for that reason.
- [ ] **Step 3: Prove annotation-only.** Strip annotations and `TYPE_CHECKING` blocks on
  both sides and compare ASTs:

```bash
cat > $SP/annot_proof.py <<'EOF'
"""Usage: annot_proof.py REV FILE... -> ANNOTATIONS ONLY, or the first differing file."""
import ast, subprocess, sys

class Strip(ast.NodeTransformer):
    def visit_arg(self, n):
        n.annotation = None
        return n
    def visit_FunctionDef(self, n):
        self.generic_visit(n)
        n.returns = None
        return n
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_If(self, n):
        if isinstance(n.test, ast.Name) and n.test.id == "TYPE_CHECKING":
            return None
        return self.generic_visit(n)
    def visit_ImportFrom(self, n):
        n.names = [a for a in n.names if a.name != "TYPE_CHECKING"]
        return n if n.names else None

def norm(src):
    return ast.dump(Strip().visit(ast.parse(src)))

rev, files = sys.argv[1], sys.argv[2:]
for f in files:
    old = subprocess.run(["git", "show", f"{rev}:{f}"], capture_output=True, text=True, check=True).stdout
    if norm(old) != norm(open(f).read()):
        sys.exit(f"DIFFERS: {f}")
print("ANNOTATIONS ONLY")
EOF
cd $WT && $PY $SP/annot_proof.py HEAD $(git diff --name-only -- collab_splats/geometry)
```

  It must print `ANNOTATIONS ONLY`. Sanity mutation: change one default value in a touched
  file, rerun, and confirm `DIFFERS`; then undo the mutation.
- [ ] **Step 4: Run gate `G`, then `P1`.** Commit with
  `style(geometry): annotate every parameter and return`.

## Finish (Tasks 31-33)

### Task 31: LC parity gate 2 at the tip

- [ ] **Step 1: Data.** The user chose **B**, the tutorial video already on disk (2026-09-25),
  over downloading 7-Scenes chess.
- [ ] **Step 2: Run** the chosen scene through `evals/scripts/eval.py` (`lc` condition) on
  the baseline commit `311ca5c8` and on the tip, each in its own tmux window, one at a
  time, nothing else running. Use a detached worktree at `311ca5c8` for the baseline
  (`git worktree add --detach $SP/lc-base 311ca5c8`, symlink `third_party/*` in) and
  remove it after.
- [ ] **Step 3: Compare.** ATE, RPE and the accepted-loop count must match exactly. A
  mismatch: bisect over the Round 2 commits with P1 green, report, stop.

### Task 32: Contract

**Files:** `tests/test_docstring_contract.py`, `CLAUDE.md`

- [ ] **Step 1:** add `"geometry"` to `PACKAGES` and to `RELEASED`.
- [ ] **Step 2:** the `CLAUDE.md` Code Style line
  "enforces all of the above ... for `preproc`, `semantics` and `pointcloud`" becomes
  "... for `preproc`, `semantics`, `pointcloud` and `geometry`".
- [ ] **Step 3:** also drop the `global_alignment.py` line from the `CLAUDE.md`
  architecture tree if Task 11 did not.
- [ ] **Step 4: Run gate `G`.** `tests/test_docstring_contract.py` must pass with geometry
  enrolled. Commit both paths with `test(geometry): enroll geometry in the docstring contract`.

### Task 33: Sweep, graph update, report

- [ ] **Step 1: Sweep.**
  `rtk proxy git grep -n "loop_closure\.eval\|loop_closure\.edge_trace\|global_alignment\|_last_loss_history\|_UNSET\|translation_jump\|max_jump_ratio\|lc_threshold_l2\|_trim_forward_outputs\|_scale_intrinsics_to_model\|_pair_pose_errors\|DEFAULT_OVERLAP\|get_corrected_extrinsics" -- collab_splats tests evals configs scripts docs/source ':!*.ipynb'`
  prints nothing. Notebooks are listed in the spec's Tutorial impact section and are not edited.
- [ ] **Step 2:** `cd $WT && graphify update .`
- [ ] **Step 3: Report** to the user: pass/fail/skip vs `$SP/baseline.txt`, commit list,
  P1 result per commit, gate 2 numbers, lines removed (`git diff --stat 311ca5c8..HEAD -- collab_splats/geometry`),
  and the tutorial notebooks that now need the renamed imports. Do not merge or push.

---

## Self-review: spec → tasks

| spec row | task |
|---|---|
| Round 1 prose, per file | 1-9 |
| `umeyama_*` raise | 10 |
| `global_alignment.py` delete | 11 |
| `eval.py` / `edge_trace.py` → `evals/` | 12 |
| `_last_loss_history` → `loss_history` | 13 |
| `_UNSET` + override kwargs | 14 |
| BA <2 active frames/points raise | 15 (user approved: raises in global and incremental) |
| track-cache `except` narrowed | 16 |
| `_extract_tracks_vggsfm` defaults | 17 |
| BA original-res K raise; `_scale_intrinsics_to_original` → metrics | 18 |
| `DEFAULT_OVERLAP`, `pair_pose_errors` | 19 |
| photometric `except` delete | 20 |
| running-error raise, `or 0`, `rel_thresh` | 21 |
| `optimize` `RuntimeError`, `decompose_camera` `ValueError` | 22 |
| umeyama import, `K_*` defaults, `min_conf_points` | 23 |
| `GraphMap` deletes, `get_corrected_extrinsics` inline | 24 |
| `Submap.raw_outputs`, test-only readers, filter inline | 25 |
| `translation_jump_check`, `nms_frame_distance` defaults | 26 (user approved: finite guard replaces it) |
| `max_jump_ratio`, `lc_threshold_l2`, `getattr` 0.85, `viz_max_points`, `_trim_forward_outputs`, omega ClassVar | 26, 27 |
| intrinsics `.get`, zero frames, `model_height` getattr | 28 |
| viz / DINO-SALAD excepts, re-upload guard, one counter | 29 |
| `scale_method` `se3` + `pairwise_dist` delete; unknown value raises | 29b (user decision 2026-09-25, spec `62c00e6d`) |
| annotations | 30 |
| LC parity gate 1 | every task touching `loop_closure/` or `transforms.py` |
| LC parity gate 2 | 31 |
| contract + CLAUDE.md | 32 |
| caller sweep, graphify, report | 33 |
| kept by decision: `scale_method` `rotation_only` + `none`, `add_submap` + `debug_out`, incremental BA, `_enough_frames`, scale estimators' 1.0 | no task; nothing touches them |

## Amendment — Task 34: remove VGGT-SPARK and VGGT-SLAM

Spec: "Amendment 2026-09-25" section. Three lanes, each in its own worktree off the tip,
cherry-picked back in order C → A → B.

- [ ] **Lane C (docs first — reads baselines before they go):** write `docs/parity.md`
  (what SPARK/VGGT-SLAM are, pinned commits, what was compared, headline numbers from
  `evals/baselines/{lc_parity*,cross_model,vggt_slam,disparity_sweep,results}`, which of
  our modules port from VGGT-SLAM); cut `evals/README.md` parity narrative to a link;
  fix `third_party/README.md` + `pyproject.toml` comments; then delete SPARK/SLAM-produced
  baseline files. Commits: `docs(parity): ...`, `chore(evals): drop SPARK/SLAM baselines`.
- [ ] **Lane A (collab_splats + its tests):** delete SPARK creator + registry + tests;
  reword SPARK comments (astcmp-proved, separate commit). Gate G′ + P1.
- [ ] **Lane B (evals code + setup):** delete SLAM overlay, `vggt_slam` compare method,
  SPARK reference row, `compose_slam_chain`/`edge_divergence`, `run_vggt_slam.py`,
  `setup/vggt_slam.sh`, Dockerfile line; update tests + known-test-failures. Gate G′.
- [ ] **Merge + final gate:** cherry-pick, G′, P1, `grep -rniI "spark\|vggt.slam"`
  outside `docs/parity.md`, `docs/superpowers`, attributions → must be empty.
