# Splats Release Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `collab_splats/splats/` releasable: PGSR deleted, package helpers reused, checkpoint I/O in its own layer with no inline import, prose and the `base.yaml` `splats:` block brought to decision 017.

**Architecture:** Round 1 edits only docstrings, comments and yaml comments; an AST proof (`.py`) and a `yaml.safe_load` proof (`base.yaml`) show it is behavior-free. Round 2 is one logical change per commit, test first where behavior is new, gate after each. Two scratch parity captures (3dgs render, Scaffold export) guard the refactors that must not change numbers. The end state adds `"splats"` to `PACKAGES` and `RELEASED` in the docstring contract.

**Tech Stack:** Python 3.11, torch, gsplat 1.5.3 (`d2f5c0f`), numpy, scikit-learn `NearestNeighbors`, OpenCV, pytest, `ast`, PyYAML.

**Spec:** [2026-09-28-splats-release-cleanup-design.md](../specs/2026-09-28-splats-release-cleanup-design.md) · Rules: [decision 017](../decisions/017-release-cleanup-rules.md)

**Deviations from the spec (naming only, same behavior):**
- the kNN helper is public `utils.knn_spacing`, not `_knn_spacing`: two modules import it, and a
  cross-module import of a `_` name is what 017's make-private verdict exists to prevent
- the Scaffold decode helper is a module-level private function
  `_offsets_to_gaussians(anchors, scaling, offsets, neural_opacity, cov, color, n_offsets)` that
  also owns the all-closed `keep` guard, because both call sites carry that guard verbatim

---

## Conventions for every task

- `WT=/workspace/collab-splats/.worktrees/splats-release` (branch `clean/splats-release`).
  Every command starts with `cd $WT &&`; the working directory resets between Bash calls.
- `SP=/tmp/claude-0/-workspace-collab-splats/fcebd1aa-d7fc-4fb5-8aec-a875ab81ecb1/scratchpad`
  holds scratch files: proof tools, parity captures, baseline. Nothing in `$SP` is committed.
- `PY=/opt/venv/reconstruction/bin/python`.
- **Gate** (`G`), run after every commit-bound change:

  ```bash
  cd $WT && PYTHONPATH=$WT $PY -c "import collab_splats, gsplat; print(collab_splats.__file__, gsplat.__version__)" \
    && cd $WT && PYTHONPATH=$WT $PY -m pytest \
       tests/splats tests/wrapper tests/mesh tests/semantics tests/evals \
       tests/test_cu121_migration.py tests/test_docstring_contract.py -q -p no:cacheprovider
  ```

  - The printed path must start with `$WT/`; the gsplat version must be `1.5.3` (it has flipped
    between sessions — if not, stop and report, do not reinstall).
  - Never pipe through `tail`, never `--tb=no`.
  - Compare counts against `$SP/baseline.txt`. Every change in pass/fail/skip must be explained by
    the task's own test adds and deletes.
  - Never run the full suite (46.6 GB cgroup, other sessions gate concurrently).
- **Render parity** (`P1`), after every Round 2 commit touching `rendering.py`, `gaussian.py` or
  `scaffold.py`: `cd $WT && PYTHONPATH=$WT $PY $SP/render_parity.py compare $SP/render_base.npz`
  must print `PARITY OK`.
- **Export parity** (`P2`), after every Round 2 commit touching `scaffold.py`:
  `cd $WT && PYTHONPATH=$WT $PY $SP/export_parity.py compare $SP/export_base.npz` must print
  `PARITY OK`.
- A failed `P1`/`P2`: `git revert --no-edit HEAD`, stop, report. The tolerance is never loosened.
- Do not `pip install` or `uv sync`; the venv is shared.
- Commit only your own paths: `git add -f <paths> && git commit --only <paths> -m ...`. Never
  `--amend`, rebase, reset or bare `git stash`. Fix a mistake with a new commit. No merge, no push.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- US spelling (`color`, `normalize`, `center`).
- No inline imports (CLAUDE.md); `gsplat` is a hard import in `splats/`.
- Use `rtk proxy grep ...` when grep output looks truncated or rewritten.
- Prose shape (CLAUDE.md, decision 017):
  - docstring summary on the line after `"""`, ≤100 chars, not restating the name
  - then `- ` bullets (≤6), then `Args:` / `Returns:` / `Raises:`
  - comment runs ≤4 lines; a run of 3+ lines is a header line followed by `- ` bullets
  - no measurements, scene ids (GH010229, C0043) or history
  - `base.yaml`: one comment line per key

---

## Task 0: Worktree, proof tools, parity captures, baseline

**Files:**
- Create (scratch): `$SP/prose_proof.py`, `$SP/yaml_proof.py`, `$SP/contract_hits.py`,
  `$SP/render_parity.py`, `$SP/export_parity.py`, `$SP/baseline.txt`, `$SP/render_base.npz`,
  `$SP/export_base.npz`

- [ ] **Step 1: Back up PGSR, create the worktree, symlink `third_party`**

```bash
cd /workspace/collab-splats && git update-ref refs/backup/splats-release/pgsr clean/final \
  && git worktree add -b clean/splats-release .worktrees/splats-release clean/final \
  && for d in third_party/*/; do ln -s "$PWD/${d%/}" ".worktrees/splats-release/${d%/}"; done
cd /workspace/collab-splats/.worktrees/splats-release && git branch --show-current && git status --short && ls third_party
```

Expected: `clean/splats-release`, empty status, `third_party` lists the same directories as the
main checkout. If `ln` reports "File exists" for a tracked directory, leave the tracked one.

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
cat > $SP/yaml_proof.py <<'EOF'
"""
Prove a yaml file's parsed content is unchanged since <base-ref>.

Usage: yaml_proof.py <base-ref> <file>
"""
import subprocess
import sys

import yaml

base, path = sys.argv[1], sys.argv[2]
old = yaml.safe_load(subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, text=True, check=True).stdout)
new = yaml.safe_load(open(path).read())
print("COMMENTS ONLY" if old == new else "CONTENT CHANGED")
sys.exit(0 if old == new else 1)
EOF
```

- [ ] **Step 3: Sanity-check both proofs on a known-bad mutation**

```bash
cd $WT && cp collab_splats/splats/trainer.py $SP/trainer.bak \
  && sed -i 's/    grow_grad2d: float = 2e-4$/    grow_grad2d: float = 3e-4/' collab_splats/splats/trainer.py \
  && $PY $SP/prose_proof.py HEAD collab_splats/splats/trainer.py; \
  cp $SP/trainer.bak collab_splats/splats/trainer.py
cd $WT && cp configs/base.yaml $SP/base.bak \
  && sed -i 's/^  max_steps: 30000$/  max_steps: 30001/' configs/base.yaml \
  && $PY $SP/yaml_proof.py HEAD configs/base.yaml; \
  cp $SP/base.bak configs/base.yaml && git -C $WT status --short
```

Expected: `CODE CHANGED: collab_splats/splats/trainer.py`, then `CONTENT CHANGED`, then an empty
status. Either proof printing its OK line means it is broken: stop.

- [ ] **Step 4: Write the contract checker**

```bash
cat > $SP/contract_hits.py <<'EOF'
"""
Every docstring-contract hit in the given files (default: all of collab_splats/splats).

Usage: contract_hits.py [<file> ...]   (run from the worktree root)
"""
import pathlib
import sys

sys.path.insert(0, "tests")
import test_docstring_contract as t

files = [pathlib.Path(a) for a in sys.argv[1:]] or sorted(pathlib.Path("collab_splats/splats").rglob("*.py"))
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

Expected: a `TOTAL` line above 0. If `RELEASE_CHECKS` or a test name moved, read
`tests/test_docstring_contract.py` and fix the checker, not the test file. If the round-3 checks
(`test_round3_rules`) use a separate dict, add it to the loop the same way.

- [ ] **Step 5: Write the render parity script (P1)**

Seeded 3dgs and 2dgs scenes through `render_gaussians`, with and without normals. The call adapts
to the signature so the same script runs before and after `absgrad`/`render_plane` are deleted.

```bash
cat > $SP/render_parity.py <<'EOF'
"""
P1: render_gaussians output bit-stable across the splats cleanup.

Usage: render_parity.py capture|compare <npz>   (run from the worktree root, PYTHONPATH=$WT)
"""
import inspect
import sys

import numpy as np
import torch
import torch.nn.functional as F

import collab_splats
from collab_splats.splats.rendering import render_gaussians

print(collab_splats.__file__)
torch.manual_seed(0)
n = 400
decoded = {
    "means": torch.rand(n, 3, device="cuda") * torch.tensor([2.0, 2.0, 2.0], device="cuda") + torch.tensor([-1.0, -1.0, 2.0], device="cuda"),
    "quats": F.normalize(torch.randn(n, 4, device="cuda"), dim=-1),
    "scales": torch.rand(n, 3, device="cuda") * 0.05 + 0.01,
    "opacities": torch.rand(n, device="cuda") * 0.8 + 0.1,
    "colors": torch.rand(n, 3, device="cuda"),
}
cam_to_world = torch.eye(4, device="cuda")[None]
intrinsics = torch.tensor([[[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]]], device="cuda")
params = inspect.signature(render_gaussians).parameters

out = {}
for primitive in ("3dgs", "2dgs"):
    for normals in (True, False):
        kwargs = {"render_normals": normals}
        if "absgrad" in params:
            kwargs["absgrad"] = False
        render, _ = render_gaussians(primitive, decoded, cam_to_world, intrinsics, 64, 48, None, **kwargs)
        for key, value in render.items():
            out[f"{primitive}/{normals}/{key}"] = value.detach().cpu().numpy()

mode, path = sys.argv[1], sys.argv[2]
if mode == "capture":
    np.savez(path, **out)
    print("CAPTURED", len(out))
    sys.exit(0)
base = dict(np.load(path))
bad = sorted(set(base) ^ set(out))
bad += [k for k in sorted(set(base) & set(out)) if not np.array_equal(base[k], out[k])]
print("PARITY OK" if not bad else f"PARITY FAIL {bad}")
sys.exit(1 if bad else 0)
EOF
cd $WT && PYTHONPATH=$WT $PY $SP/render_parity.py capture $SP/render_base.npz \
  && PYTHONPATH=$WT $PY $SP/render_parity.py compare $SP/render_base.npz
```

Expected: path inside `$WT`, `CAPTURED <k>`, then `PARITY OK`. If the compare of an unchanged tree
fails, the kernel is non-deterministic here: switch `np.array_equal` to
`np.allclose(base[k], out[k], rtol=0, atol=1e-6)` in both places, re-capture, re-compare, and
record the tolerance in `$SP/baseline.txt`.

- [ ] **Step 6: Write the Scaffold export parity script (P2)**

```bash
cat > $SP/export_parity.py <<'EOF'
"""
P2: Scaffold export/decode and Gaussians initial scales bit-stable across the splats cleanup.

Usage: export_parity.py capture|compare <npz>   (run from the worktree root, PYTHONPATH=$WT)
"""
import sys

import numpy as np
import torch

import collab_splats
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.scaffold import Scaffold
from collab_splats.splats.trainer import SplatsConfig
from tests.splats.synthetic import make_scene

print(collab_splats.__file__)
torch.manual_seed(0)
images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=4)
cfg = SplatsConfig.from_dict({"representation": "scaffold", "scaffold": {"n_offsets": 4, "feat_dim": 8}})
model = Scaffold(cfg, points, colors, scene_scale=1.0, n_views=4, device="cuda")
cam_to_world = torch.from_numpy(np.linalg.inv(world_to_cam)).float().cuda()
K = torch.from_numpy(intrinsics).float().cuda()

out = {f"export/{k}": v.cpu().numpy() for k, v in model.export_gaussians(cam_to_world, K, 64, 64).items()}
with torch.no_grad():
    decoded, index = model.decode("3dgs", cam_to_world[:1], K[:1], 64, 64, torch.tensor([0], device="cuda"))
out.update({f"decode/{k}": v.cpu().numpy() for k, v in decoded.items()})
out["decode/index"] = index.cpu().numpy()
out["voxel_size"] = np.array(model.voxel_size)

# Vanilla initial scales: the other kNN-spacing caller
vanilla = Gaussians(SplatsConfig(), points, colors, 1.0, 4, "cpu")
out["gaussians/scales"] = vanilla.params["scales"].detach().numpy()

mode, path = sys.argv[1], sys.argv[2]
if mode == "capture":
    np.savez(path, **out)
    print("CAPTURED", len(out))
    sys.exit(0)
base = dict(np.load(path))
bad = sorted(set(base) ^ set(out))
bad += [k for k in sorted(set(base) & set(out)) if not np.array_equal(base[k], out[k])]
print("PARITY OK" if not bad else f"PARITY FAIL {bad}")
sys.exit(1 if bad else 0)
EOF
cd $WT && PYTHONPATH=$WT $PY $SP/export_parity.py capture $SP/export_base.npz \
  && PYTHONPATH=$WT $PY $SP/export_parity.py compare $SP/export_base.npz
```

Expected: `CAPTURED <k>` then `PARITY OK`. The config and constructor mirror
`tests/splats/test_scaffold.py::_scaffold_config` / `_field`; if either rejects the call, copy that
fixture's call, do not edit the source. If `Gaussians(...)` positionals differ, copy
`trainer.train`'s `MODEL_CLASSES[...]` call.

- [ ] **Step 7: Baseline gate**

Run `G` and save it:

```bash
cd $WT && PYTHONPATH=$WT $PY -c "import collab_splats, gsplat; print(collab_splats.__file__, gsplat.__version__)" \
  && cd $WT && PYTHONPATH=$WT $PY -m pytest \
     tests/splats tests/wrapper tests/mesh tests/semantics tests/evals \
     tests/test_cu121_migration.py tests/test_docstring_contract.py -q -p no:cacheprovider > $SP/baseline.txt 2>&1; \
  echo "exit $?" >> $SP/baseline.txt; grep -E "passed|failed|error|^exit" $SP/baseline.txt
```

Record the pass/fail/skip/xfail counts. Every pre-existing failure must match an entry in
`docs/known-test-failures.md`; one that does not is reported before any edit.

---

## Round 1 — prose only

Each task: edit, run `prose_proof.py clean/final <files>` → `PROSE ONLY`, run `contract_hits.py
<files>` → no hits from the edited lines, run `G`, commit. PGSR prose (`pgsr.py`, pgsr docstrings
in `losses.py`/`rendering.py`, the `base.yaml` PGSR block, `rendering.py` module-docstring lines
about `render_plane` and GS-SR citations) is left alone — it leaves with the code in Task 2.1.

### Task 1.1: `trainer.py` prose

**Files:** Modify `collab_splats/splats/trainer.py:1-17`, `:98-101`

- [ ] **Step 1: Replace the module docstring**

```python
"""
Gaussian-splat training on upstream gsplat.

- ``primitive`` picks the rasterizer (3dgs / 2dgs), ``representation`` the model (vanilla / scaffold)
- both models answer one interface, so ``train`` never branches
- in: frames, COLMAP poses, intrinsics, seed points; out: splats.ply, ckpt.pt, quality report
- ``scene_scale``: 1.1 x the largest camera distance from their centroid; scales the means lr,
  the densification thresholds and depth-unit losses
- ``normalize_scene``: splatfacto's unit cube at ``scene_scale`` 1.0, outputs mapped back to world
- pose lrs: rotation x world extent, translation x ``scene_scale``; 2dgs distortion rescaled too
"""
```

- [ ] **Step 2: Replace the `grow_grad2d` comment (lines 98-100)**

```python
    # 2dgs (Default): 2D-gradient split/duplicate threshold
    # - gsplat's non-absgrad default; the absgrad-calibrated 8e-4 under-densifies
```

- [ ] **Step 3: Contract hits for the rest of the file**

```bash
cd $WT && $PY $SP/contract_hits.py collab_splats/splats/trainer.py
```

Fix every non-PGSR hit by prose only (bullet caps, comment runs, over-long summaries). Hits on
`near_ids`/pgsr lines are skipped here.

- [ ] **Step 4: Prove, gate, commit**

```bash
cd $WT && $PY $SP/prose_proof.py clean/final collab_splats/splats/trainer.py
```

Expected `PROSE ONLY`. Run `G`; counts equal the baseline.

```bash
cd $WT && git add collab_splats/splats/trainer.py && git commit --only collab_splats/splats/trainer.py \
  -m "docs(splats): trainer prose to decision 017" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 1.2: `gaussian.py` prose

**Files:** Modify `collab_splats/splats/gaussian.py:28-30`, `:63-66`

- [ ] **Step 1: `SH_C0` comment (lines 28-30)**

```python
# Degree-0 spherical harmonic: rgb = SH_C0 * sh0 + 0.5
# - do not rewrite the literal: it changes the bytes of every existing splats.ply
```

- [ ] **Step 2: `make_strategy` comment, last bullet (line 66)**

```python
    # - grow_grad2d 2e-4: gsplat's non-absgrad threshold
```

- [ ] **Step 3: Remaining contract hits** — `contract_hits.py collab_splats/splats/gaussian.py`, fix
  non-PGSR hits by prose only (`render_plane` Args line is left for Task 2.1).

- [ ] **Step 4: Prove, gate, commit** — `prose_proof.py clean/final collab_splats/splats/gaussian.py`
  → `PROSE ONLY`; `G`; commit `docs(splats): gaussian prose to decision 017` with `--only`.

### Task 1.3: `rendering.py` prose

**Files:** Modify `collab_splats/splats/rendering.py:14`, `:270`

- [ ] **Step 1: Module docstring** — delete line 14 (`- GS-SR refactored ``pgsr_scene.py`` Oct 2025: ...`).
  Lines 12-13 (`render_plane`, GS-SR citation) go in Task 2.1.

- [ ] **Step 2: `render_views` bullet (line 270)**

```python
    - a generator, not a list: a whole scene's renders do not fit in memory at once
```

- [ ] **Step 3: Remaining non-PGSR contract hits**, prose only. The module docstring bullet cap is
  fixed in Task 2.4, which rewrites this docstring.

- [ ] **Step 4: Prove, gate, commit** — `docs(splats): rendering prose to decision 017`.

### Task 1.4: `scaffold.py` prose

**Files:** Modify `collab_splats/splats/scaffold.py:469-474`

- [ ] **Step 1: `_frustum_anchors` docstring**

```python
        """
        CPU stand-in for the projection prefilter: anchor centers in front of the camera, in frame.

        - whole-frame margin: keeps anchors whose Gaussians spill into frame, as GPU radii do
        """
```

(Private def: summary + bullets only, no `Returns:` block — CLAUDE.md docstring rule.)

- [ ] **Step 2: Remaining non-PGSR contract hits** in `scaffold.py`, prose only (the
  `render_plane` Args line and the "PGSR renders a neighbor view" test docstring are Task 2.1).

- [ ] **Step 3: Prove, gate, commit** — `docs(splats): scaffold prose to decision 017`.

### Task 1.5: `cameras.py`, `utils.py`, `losses.py`, `__init__.py` pass

**Files:** Modify only where `contract_hits.py` reports a hit.

- [ ] **Step 1:** `cd $WT && $PY $SP/contract_hits.py collab_splats/splats/cameras.py collab_splats/splats/utils.py collab_splats/splats/losses.py collab_splats/splats/__init__.py`
- [ ] **Step 2:** Fix every non-PGSR hit by prose only. No content change is expected; if a file
  has zero hits, it is not touched.
- [ ] **Step 3:** Prove the touched files, gate, commit `docs(splats): cameras/utils/losses prose to decision 017`.
  Skip the commit if nothing changed and say so in the task report.

### Task 1.6: `configs/base.yaml` `splats:` block comments

**Files:** Modify `configs/base.yaml` (the `# Gaussian splats …` preamble through `losses:`, not the
PGSR block)

- [ ] **Step 1: Verify the depth-target claim before keeping it**

```bash
cd $WT && rtk proxy grep -n "conf_percentile" collab_splats/wrapper/reconstructor.py
```

If the splats stage does not mask depth targets by `mesh.conf_percentile`, drop that bullet below.

- [ ] **Step 2: Replace the preamble and the multi-line key comments**

```yaml
# Gaussian splats from the pointcloud stage (leaf stage `splats`, upstream gsplat)
# - photometric 0.8 L1 + 0.2 (1 - SSIM) always on; a loss is active iff weight > 0 and step >= start
# - depth targets: pointcloud.zarr depth masked by mesh.conf_percentile
# - 3dgs densifies with MCMC (cap_max), 2dgs with DefaultStrategy (grow_grad2d) + distortion loss
splats:
  enabled: false
  primitive: 3dgs            # 3dgs (fast kernel, antialiased) | 2dgs (surface-aligned)
  representation: vanilla    # vanilla (per-gaussian) | scaffold (anchors + MLP; set losses.opacity_reg 0)
```

(the commented `# scaffold:` sub-block is unchanged)

```yaml
  max_steps: 30000
  pose_opt: true             # refine camera poses jointly (CameraOpt)
  appearance_opt: false      # per-image affine color, train views only; regularized by losses.appearance_reg
  sh_degree: 3               # vanilla only; a scaffold override raises
  sh_degree_interval: 1000   # vanilla only
```

(the lr keys and `cap_max` unchanged)

```yaml
  grow_grad2d: 2.0e-4        # 2dgs only: gsplat's non-absgrad default
  num_downscales: 2          # coarse-to-fine (splatfacto): start at 1/2^n res; 0 disables
  resolution_schedule: 3000  # steps per resolution doubling
  normalize_scene: false     # splatfacto Sim3 to a unit cube; outputs mapped back to world units
  log_every: 500
  # Each loss: {weight[, start, end, end_weight]}; with end, weight decays log-linearly to end_weight
  # e.g. depth: {weight: 0.01, end: 12000, end_weight: 0.001}
  losses:
    depth: {weight: 0.01}
    normal_consistency: {weight: 0.05, start: 7000}   # 2dgs: depth_ratio blends in the median-depth normal
```

Check `sh_degree_interval` really is vanilla-only by reading the scaffold validation in
`SplatsConfig.__post_init__`; if only `sh_degree` raises, write `# vanilla only` on both anyway
(true: scaffold ignores both) and nothing about raising on the interval.

- [ ] **Step 3: Prove, gate, commit**

```bash
cd $WT && $PY $SP/yaml_proof.py clean/final configs/base.yaml
```

Expected `COMMENTS ONLY`. `G`. Commit `docs(config): splats block comments to decision 017` with
`--only configs/base.yaml`.

---

## Round 2 — code, one commit per logical change

### Task 2.1: Delete PGSR

**Files:**
- Delete: `collab_splats/splats/pgsr.py`, `tests/splats/test_pgsr.py`
- Modify: `collab_splats/splats/losses.py`, `trainer.py`, `rendering.py`, `gaussian.py`,
  `scaffold.py`, `configs/base.yaml`, `tests/test_cu121_migration.py:129`,
  `tests/splats/test_rendering.py`, `test_gaussian.py`, `test_scaffold.py`, `test_losses.py`,
  `test_trainer.py`

- [ ] **Step 1: Delete the module and its test**

```bash
cd $WT && git rm -q collab_splats/splats/pgsr.py tests/splats/test_pgsr.py
```

- [ ] **Step 2: `losses.py`**
  - delete the `from collab_splats.splats.pgsr import (...)` block
  - delete `neighbor_selection`, `pgsr_normal_loss`, `pgsr_multiview_loss`
  - delete their `OPTIONAL_LOSSES` entries and the `pgsr_*` entries in `LOSS_SPEC_KEYS`; the
    `LOSS_SPEC_KEYS` comment loses its pgsr sentence
  - `rtk proxy grep -n "pgsr\|plane\|neighbor" collab_splats/splats/losses.py` → no hits except
    unrelated words; read any hit before deleting it

- [ ] **Step 3: `trainer.py`**
  - imports: drop `random`, `neighbor_selection`, and the `pgsr` import line
  - delete the block from `# PGSR losses are only defined against the 3dgs kernel` through the
    `near_ids = select_near_views(...)` call
  - in the loop: delete `render_plane = ...`, the `render_plane=render_plane,` kwarg and the whole
    `# PGSR multi-view: render one co-visible neighbor` block
  - the loss comment `# Loss + backward; `info` is the MAIN view's, never the neighbor's` becomes
    `# Loss + backward`

  The render call becomes:

```python
        # Render with the SH bands unlocked so far
        # - 3DGS normals cost an extra pass: rendered only once the consistency loss is on
        render_normals = loss_active(step, normal_spec)
        render, info = model.render(
            view_cam_to_world,
            view_intrinsics,
            step_width,
            step_height,
            camera_id,
            step=step,
            render_normals=render_normals,
        )
```

- [ ] **Step 4: `rendering.py`**
  - drop `from collab_splats.splats.pgsr import plane_depth as compute_plane_depth`
  - module docstring: delete the `render_plane` and `GS-SR` bullets
  - `gaussian_normals_in_camera_frame`: return only the normals; annotation `-> Tensor`; `Returns:`
    becomes `(N, 3) camera-frame normals.`; summary line `Per-Gaussian camera-frame normal: shortest
    scale axis, flipped to face the camera.` The `means_cam` local stays (the flip needs it)
  - `render_gaussians`: delete the `render_plane` param, its Args line, the 2dgs rejection, the
    `render_normals = render_normals or render_plane` line, the `render_plane only` Returns bullets
    (three lines), the 4th-channel `if render_plane` branch and the whole plane tail. The normals
    block becomes:

```python
    # 3DGS normals ride as an extra per-Gaussian signal
    # - padded to 4 channels: the compiled kernel takes 8 total (rgb + depth + 4), not 7
    normals_cam = gaussian_normals_in_camera_frame(quats, scales, means, world_to_cam[0])
    extra_signals = torch.cat([normals_cam, torch.zeros_like(normals_cam[:, :1])], dim=-1)

    rgb_depth, alpha, info = rasterization(**shared_kwargs, rasterize_mode="antialiased", extra_signals=extra_signals)
    depth = rgb_depth[..., 3:4]
    render = {
        "rgb": rgb_depth[..., :3],
        "alpha": alpha,
        "depth": depth,
        "normal": F.normalize(info["render_extra_signals"][..., :3], dim=-1),
        "depth_normal": depth_to_normal(depth, identity_pose, intrinsics),
    }
    return render, info
```

- [ ] **Step 5: `gaussian.py`, `scaffold.py`** — in each `render`: delete the `render_plane` param,
  its Args line and the `render_plane=render_plane,` kwarg.

- [ ] **Step 6: `configs/base.yaml`** — delete the `# PGSR (3dgs only …` comment block (from that
  line through `# neighbor selection.`).

- [ ] **Step 7: Tests**
  - `tests/test_cu121_migration.py`: delete the `"collab_splats.splats.pgsr",` line
  - `test_rendering.py`: delete `test_render_plane_rejects_2dgs`,
    `test_render_plane_adds_exactly_the_four_plane_keys`,
    `test_render_plane_normal_is_the_raw_accumulated_map`,
    `test_plane_depth_matches_rasterized_depth_on_a_fronto_parallel_plane`,
    `test_render_plane_forces_the_extra_signal_pass_on`,
    `test_plane_path_does_not_perturb_the_ordinary_render`, and `_flat_disc` if no test still uses
    it. The `gaussian_normals_in_camera_frame` test (line ~139) becomes:

```python
    normals = gaussian_normals_in_camera_frame(gaussians["quats"], scales, gaussians["means"], world_to_cam)
    rotation_w2c = world_to_cam[:3, :3]
    translation_w2c = world_to_cam[:3, 3]
    means_cam = gaussians["means"] @ rotation_w2c.T + translation_w2c
    facing = (normals * means_cam).sum(-1)
```

    and loses its two `means_cam` shape/allclose asserts (keep every normal assert). Read the test
    first and keep its local names; if `rotation_w2c`/`translation_w2c` already exist, reuse them.
  - `test_gaussian.py::test_render_forwards_the_normal_and_plane_flags` → rename to
    `test_render_forwards_the_normal_flag`:

```python
def test_render_forwards_the_normal_flag(monkeypatch):
    model = _model()
    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return {}, {}

    monkeypatch.setattr("collab_splats.splats.gaussian.render_gaussians", _capture)
    view = (torch.eye(4)[None], torch.eye(3)[None], 64, 64, torch.zeros(1, dtype=torch.long))

    # Default on (the mesh path needs normals)
    model.render(*view, step=0)
    assert captured == {"render_normals": True}

    # A pass-through: a hardcoded default here would silently ignore the caller
    model.render(*view, step=0, render_normals=False)
    assert captured == {"render_normals": False}
```

  - `test_scaffold.py::test_scaffold_render_passes_the_extra_signal_flags_through` → rename to
    `test_scaffold_render_passes_the_normal_flag_through`; docstring `The regularizers ask for
    normals; that cannot be decided in here.`; delete the `planar` render and its assert. The next
    test's docstring (`PGSR renders a neighbor view and discards its info …`) becomes `A render's
    per-view state travels in its info, so a second render cannot clobber the first.`
  - `test_losses.py`: drop `neighbor_selection` from the import; delete `"pgsr_normal",` and
    `"pgsr_multiview",` from `test_registry_names`; delete the three `test_neighbor_selection_*`
  - `test_trainer.py`: delete `test_neighbor_render_uses_the_neighbors_own_camera_id`; delete
    `import itertools` only if nothing else uses it

- [ ] **Step 8: No PGSR residue**

```bash
cd $WT && rtk proxy grep -rn "pgsr\|render_plane\|plane_depth\|select_near_views\|render_neighbor\|neighbor_selection" collab_splats configs tests evals docs/source
```

Expected: no hits. A hit in `docs/source` (tutorial) is reported, not edited.

- [ ] **Step 9: Gate, parity, commit**

`G`: counts drop by exactly the deleted tests (`test_pgsr.py`'s count plus the 6 + 1 + 3 + 1
named above); nothing else moves. `P1` → `PARITY OK` (the capture never set `render_plane`).

```bash
cd $WT && git add -A collab_splats/splats configs/base.yaml tests/splats tests/test_cu121_migration.py \
  && git commit --only collab_splats/splats configs/base.yaml tests/splats tests/test_cu121_migration.py \
  -m "refactor(splats): delete PGSR" \
  -m "gsplat has no native plane rasterizer; render_plane rebuilt plane depth after the fact, was never validated and no config enabled it. Backup: refs/backup/splats-release/pgsr." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2.2: One 3dgs rasterization call

**Files:** Modify `collab_splats/splats/rendering.py` (`render_gaussians` 3dgs tail); Test
`tests/splats/test_rendering.py`

- [ ] **Step 1: Write the test**

Append to `tests/splats/test_rendering.py` (CUDA-marked like its neighbors; use the file's own
`cuda` marker name and camera helper):

```python
@cuda
def test_3dgs_normals_do_not_change_rgb_or_depth():
    # One rasterization call serves both settings; the extra signal must not leak into rgb/depth
    cam_to_world, intrinsics = _camera()
    with_normals, _ = _render("3dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, render_normals=True)
    without, _ = _render("3dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, render_normals=False)

    assert torch.equal(with_normals["rgb"], without["rgb"])
    assert torch.equal(with_normals["depth"], without["depth"])
    assert {"normal", "depth_normal"} <= set(with_normals)
    assert not {"normal", "depth_normal"} & set(without)
```

Replace `_camera()` with whatever the file's existing 3dgs tests call to build `(cam_to_world,
intrinsics)` (read `test_render_plane_*`'s former neighbors). If `torch.equal` fails on the
current code, the two branches already differ: use `torch.allclose(..., atol=1e-6)`, and note it.

- [ ] **Step 2: Run it on the current code**

```bash
cd $WT && PYTHONPATH=$WT $PY -m pytest tests/splats/test_rendering.py -k normals_do_not_change -q -p no:cacheprovider
```

Expected: PASS (it pins today's behavior before the merge).

- [ ] **Step 3: Merge the branches**

Replace the `# 3DGS without normals` early return and the normals block with:

```python
    # 3DGS normals ride as an extra per-Gaussian signal, only when asked for
    # - padded to 4 channels: the compiled kernel takes 8 total (rgb + depth + 4), not 7
    extra_signals = None
    if render_normals:
        normals_cam = gaussian_normals_in_camera_frame(quats, scales, means, world_to_cam[0])
        extra_signals = torch.cat([normals_cam, torch.zeros_like(normals_cam[:, :1])], dim=-1)

    rgb_depth, alpha, info = rasterization(**shared_kwargs, rasterize_mode="antialiased", extra_signals=extra_signals)
    depth = rgb_depth[..., 3:4]
    render = {"rgb": rgb_depth[..., :3], "alpha": alpha, "depth": depth}
    if render_normals:
        render["normal"] = F.normalize(info["render_extra_signals"][..., :3], dim=-1)
        render["depth_normal"] = depth_to_normal(depth, identity_pose, intrinsics)
    return render, info
```

- [ ] **Step 4: Test, gate, parity, commit**

The new test PASSES; `G` = previous + 1 pass; `P1` → `PARITY OK`. If `P1` fails on
`3dgs/False/*`, gsplat takes a different code path for `extra_signals=None` than for omitting the
kwarg: revert per convention and report.

Commit `refactor(splats): one 3dgs rasterization call for both normal settings` (`--only
collab_splats/splats/rendering.py tests/splats/test_rendering.py`).

### Task 2.3: Delete `absgrad`

**Files:** Modify `rendering.py` (`render_gaussians`), `gaussian.py` (`render`), `scaffold.py`
(`render`); tests `test_rendering.py:_render`, `test_gaussian.py`, `test_scaffold.py:422-442`

- [ ] **Step 1: Source**
  - `render_gaussians`: delete the `absgrad: bool,` param, its Args line and `absgrad=absgrad,` in
    `shared_kwargs`. gsplat's default is `absgrad=False`; confirm with
    `$PY -c "import inspect, gsplat; print(inspect.signature(gsplat.rasterization).parameters['absgrad'].default, inspect.signature(gsplat.rasterization_2dgs).parameters['absgrad'].default)"` → `False False`. If either is not False, keep `absgrad=False` in `shared_kwargs` as a literal with a one-line comment.
  - `Gaussians.render`: delete the `absgrad = False` literal, its comment run, and the positional
    `absgrad,` argument
  - `Scaffold.render`: delete `absgrad=False,`
  - `make_strategy`'s `absgrad=False` stays (gsplat strategy arg)

- [ ] **Step 2: Tests**
  - `test_rendering.py::_render`: call becomes `render_gaussians(primitive, model.activate(),
    cam_to_world, intrinsics, width, height, 0, **kwargs)`; docstring summary `Rasterize a
    ParameterDict fixture at SH degree 0.`
  - `test_gaussian.py`: every `_capture(primitive, decoded, cam_to_world, intrinsics, width,
    height, sh_degree, absgrad, **kwargs)` loses `absgrad, `; delete
    `test_render_never_asks_for_absolute_gradients` (the literal it guarded is gone; the strategy
    half stays pinned at line ~179)
  - `test_scaffold.py:427,442`: drop `, absgrad=False`

- [ ] **Step 3: Gate, parity, commit** — `G` = previous − 1 (the deleted test); `P1` →
  `PARITY OK` (the parity script already adapts). `rtk proxy grep -rn "absgrad" collab_splats/splats tests/splats`
  → only `make_strategy`, `scaffold.py`'s AnchorStrategy comment, and strategy asserts.
  Commit `refactor(splats): drop the always-False absgrad render arg`.

### Task 2.4: `checkpoint.py` layer

**Files:**
- Create: `collab_splats/splats/checkpoint.py`
- Modify: `rendering.py`, `trainer.py`, `__init__.py`, `collab_splats/mesh/io.py:179`,
  `evals/scripts/eval_splats.py:39`, `evals/scripts/analyze_splats.py:32`,
  `tests/mesh/test_io.py`, `tests/test_cu121_migration.py`, `tests/splats/test_trainer.py`,
  `test_rendering.py`, `test_model_interface.py`

- [ ] **Step 1: Create `checkpoint.py`**

Move `write_outputs` and `load_checkpoint` verbatim from `rendering.py` (bodies unchanged except
the inline import and its comment, which are deleted), plus `MODEL_CLASSES`/`REPRESENTATIONS`
with their comment from `trainer.py`. Header:

```python
"""
The splat stage's artifacts: writing them, and reading a checkpoint back.

- ``write_outputs``: splats.ply / ckpt.pt / splats_quality_report.json
- ``load_checkpoint``: ckpt.pt back into a render-only model and its cameras
- ``MODEL_CLASSES``: representation -> model class, shared by the trainer and the reader
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from gsplat.exporter import export_splats
from gsplat.losses import ssim_loss
from torch import Tensor

from collab_splats.splats.cameras import CameraOpt
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.rendering import render_views
from collab_splats.splats.scaffold import Scaffold
from collab_splats.utils.io import write_json
from collab_splats.utils.progress import progress

if TYPE_CHECKING:
    from collab_splats.splats.trainer import SplatsConfig

logger = logging.getLogger(__name__)

# Representation -> model class; the config validator takes its allow-list from the same mapping
MODEL_CLASSES = {"vanilla": Gaussians, "scaffold": Scaffold}
REPRESENTATIONS = tuple(MODEL_CLASSES)
```

Then the two functions. Keep only the imports the moved bodies use: run
`cd $WT && $PY -m flake8 --select=F401,F811,F821 collab_splats/splats/checkpoint.py collab_splats/splats/rendering.py collab_splats/splats/trainer.py`
and resolve every hit (pass `--max-line-length=200` if E501 noise appears; the config is inert).

- [ ] **Step 2: `rendering.py`** — delete `write_outputs`, `load_checkpoint`, the imports only they
  used and the `TYPE_CHECKING` block if nothing left references it. Module docstring:

```python
"""
Rasterization: the one call into gsplat, and view-by-view re-rendering.

- ``render_gaussians``: 3DGS via ``gsplat.rasterization`` (antialiased), 2DGS via ``rasterization_2dgs``
- 2DGS adds a distortion map and RaDe-GS median depth; its normals arrive WORLD-frame, rotated here
- depth normals: finite-differenced at an identity pose, both primitives and both depths
- ``render_views``: re-render a trained model one view at a time
"""
```

- [ ] **Step 3: `trainer.py`** — delete `MODEL_CLASSES`/`REPRESENTATIONS` and their comment; import
  `from collab_splats.splats.checkpoint import MODEL_CLASSES, REPRESENTATIONS, write_outputs`; drop
  the `Gaussians`, `Scaffold` and `rendering` imports if now unused (keep `ScaffoldConfig`). Check
  how `SplatsConfig` validates `representation` still resolves `REPRESENTATIONS`.

- [ ] **Step 4: `__init__.py`**

```python
from .checkpoint import load_checkpoint  # noqa: E402
from .gaussian import Gaussians  # noqa: E402
from .rendering import render_views  # noqa: E402
from .scaffold import Scaffold  # noqa: E402
from .trainer import SplatsConfig, train  # noqa: E402

__all__ = [
    "GSPLAT_COMMIT",
    "Gaussians",
    "Scaffold",
    "SplatsConfig",
    "load_checkpoint",
    "render_views",
    "train",
]
```

- [ ] **Step 5: External callers**
  - `collab_splats/mesh/io.py:179`: `from collab_splats.splats import load_checkpoint, render_views`
    (keep it inside the function with its existing "gsplat needs CUDA at import" comment — that
    is the CLAUDE.md optional-heavy-dep exception, and moving it is mesh-release's call)
  - `tests/mesh/test_io.py`: every `monkeypatch.setitem(sys.modules, "collab_splats.splats.rendering", ...)`
    → `"collab_splats.splats"`. Read `_fake_rendering` first: the stub module must expose both
    `load_checkpoint` and `render_views`; if it exposes only those, it is a valid stand-in for the
    package import
  - `evals/scripts/eval_splats.py:39`, `evals/scripts/analyze_splats.py:32`: the same package import
  - `tests/test_cu121_migration.py`: add `"collab_splats.splats.checkpoint",` next to the other
    splats modules

- [ ] **Step 6: Tests**
  - `test_trainer.py`: import `MODEL_CLASSES, REPRESENTATIONS` from
    `collab_splats.splats.checkpoint`; `_recorder` still patches `trainer_module.write_outputs`
    (the trainer's own binding) — unchanged. `test_public_api_surface`: add `"render_views"`, comment
    `# Exactly seven: two model classes, the config, the entry point, the ckpt reader, the view
    re-renderer, the commit`
  - `test_rendering.py`: import `load_checkpoint, write_outputs` from `collab_splats.splats.checkpoint`,
    `REPRESENTATIONS` from the same; the comment `# The class comes off the trainer's MODEL_CLASSES`
    → `# The class comes off checkpoint.MODEL_CLASSES`; module docstring line 5 unchanged
  - `test_model_interface.py`: `from collab_splats.splats import checkpoint, rendering, trainer`;
    `consumed` becomes the union over all three files:

```python
    consumed = set()
    for module in (trainer, rendering, checkpoint):
        consumed |= _members_read_off_the_model(module.__file__)
```

    and the line-194 comment names `checkpoint.write_outputs`. The line-24 comment names the three
    files.

- [ ] **Step 7: Prove the cycle is gone**

```bash
cd $WT && rtk proxy grep -n "^    from \|^        from " collab_splats/splats/*.py; \
  PYTHONPATH=$WT $PY -c "import collab_splats.splats.checkpoint as c, collab_splats.splats as s; print(c.__file__, s.render_views, s.load_checkpoint)"
```

Expected: no indented imports in `splats/` (the `TYPE_CHECKING` block is the only indented import
allowed), then the path inside `$WT`.

- [ ] **Step 8: Gate, parity, commit** — `G` equal to previous (moves only); `P1` OK.
  Commit `refactor(splats): checkpoint.py owns write_outputs, load_checkpoint and MODEL_CLASSES`
  with `--only` over every path above. Body: `Callers import from collab_splats.splats. Sibling
  note: mesh-release's render_tsdf_inputs belongs in checkpoint.py.`

### Task 2.5: Reuse `invert_poses`

**Files:** Modify `collab_splats/splats/trainer.py` (`cam_to_world_np = np.linalg.inv(world_to_cam)`)

- [ ] **Step 1:** `from collab_splats.geometry.transforms import invert_poses` (isort order), then

```python
    cam_to_world_np = invert_poses(np.asarray(world_to_cam, dtype=np.float64))
```

Check the dtype the old line produced: `np.linalg.inv` of a float32 input returns float32. If
`make_scene` / the reconstructor pass float32, keep the dtype with
`invert_poses(world_to_cam)` instead (no cast) and confirm `cam_to_world_np[:, :3, 3] = ...` still
writes in place (`invert_poses` returns a fresh array, so it does).

- [ ] **Step 2:** `G`: equal to previous. A test comparing floats bit-exactly against a
  trained output may shift at 1e-7; if one fails, read it — a failure there is an honest
  precision change, fix the test's tolerance only if the test is asserting an incidental value,
  otherwise revert and report.
- [ ] **Step 3:** Commit `refactor(splats): invert poses with geometry.transforms.invert_poses`.

### Task 2.6: Reuse `rescale_intrinsics`, K computed once per downscale factor

**Files:** Modify `collab_splats/splats/utils.py` (`downscale_view` → `downscale_image`),
`trainer.py`; Test `tests/splats/test_utils.py:103-137`, `tests/splats/test_trainer.py`

- [ ] **Step 1: Write the failing trainer test**

Append to `tests/splats/test_trainer.py`:

```python
@cuda
def test_downscaled_intrinsics_match_the_resized_image(monkeypatch, tmp_path):
    # Odd size: W // 2 is not W / 2, so K must follow the resized grid, not a plain divide
    images, world_to_cam, intrinsics, points, colors, _ = make_scene(n_views=2, height=45, width=63)
    _recorder(monkeypatch)
    seen = []
    real_render = gaussian_module.Gaussians.render

    def record(self, cam_to_world, view_intrinsics, width, height, *args, **kwargs):
        seen.append((view_intrinsics.detach().cpu().numpy()[0], width, height))
        return real_render(self, cam_to_world, view_intrinsics, width, height, *args, **kwargs)

    monkeypatch.setattr(gaussian_module.Gaussians, "render", record)
    cfg = SplatsConfig.from_dict({"primitive": "3dgs", "max_steps": 1, "num_downscales": 1, "resolution_schedule": 10})
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, image_ids=_ids(images))

    K, width, height = seen[0]
    assert (width, height) == (31, 22)
    assert K[0, 2] == pytest.approx(intrinsics[0, 0, 2] * 31 / 63)
    assert K[1, 2] == pytest.approx(intrinsics[0, 1, 2] * 22 / 45)
    assert K[0, 0] == pytest.approx(intrinsics[0, 0, 0] * 31 / 63)
```

- [ ] **Step 2: Run it** — `pytest tests/splats/test_trainer.py -k downscaled_intrinsics -q`.
  Expected FAIL: `K[0, 2]` is `31.5 / 2 = 15.75`, not `31.5 * 31 / 63 = 15.5`.

- [ ] **Step 3: `utils.py`** — replace `downscale_view` with:

```python
def downscale_image(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Bilinear resize to (H // factor, W // factor); the input itself at factor 1.

    - K for the resized grid comes from `geometry.transforms.rescale_intrinsics`, not from here

    Args:
        image: (H, W, 3) uint8 image.
        factor: integer divisor from `downscale_factor`.

    Returns:
        (H // factor, W // factor, 3) image; never a mutated input.
    """
    if factor == 1:
        return image
    height, width = image.shape[:2]
    return cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR)
```

- [ ] **Step 4: `trainer.py`** — import `rescale_intrinsics` next to `invert_poses` and
  `downscale_image` instead of `downscale_view`. Replace `intrinsics_gpu = ...` with:

```python
    intrinsics_gpu = torch.from_numpy(intrinsics).float().to(device)

    # K per coarse-to-fine factor, on the exact resized grid (H // f, W // f)
    native_hw = np.array([height, width])
    intrinsics_by_factor = {}
    for level in range(max(cfg.num_downscales, 0) + 1):
        factor = 2**level
        scaled = rescale_intrinsics(intrinsics, native_hw, native_hw // factor)
        intrinsics_by_factor[factor] = torch.from_numpy(scaled).float().to(device)
```

and in the loop:

```python
        # Coarse-to-fine (splatfacto): 1/4 -> 1/2 -> native; prepare_target resizes depth to match
        factor = downscale_factor(step, cfg.num_downscales, cfg.resolution_schedule)
        view_image = downscale_image(view_image, factor)
        view_intrinsics = intrinsics_by_factor[factor][view : view + 1]
        step_height, step_width = view_image.shape[:2]
```

`intrinsics_gpu` stays: `write_outputs` takes native K. At factor 1, `rescale_intrinsics` returns
float64 equal to the input; the `.float()` round-trip gives the same float32 values as
`intrinsics_gpu`.

- [ ] **Step 5: `test_utils.py`** — import `downscale_image`; replace the three `downscale_view`
  tests with:

```python
def test_downscale_image_halves_the_image():
    image = np.zeros((64, 32, 3), np.uint8)
    assert downscale_image(image, 2).shape == (32, 16, 3)


def test_downscale_image_floors_an_odd_size():
    image = np.zeros((45, 63, 3), np.uint8)
    assert downscale_image(image, 4).shape == (11, 15, 3)


def test_downscale_image_passes_through_at_factor_one():
    image = np.zeros((8, 8, 3), np.uint8)
    assert downscale_image(image, 1) is image
```

- [ ] **Step 6:** New trainer test PASSES; `rtk proxy grep -rn "downscale_view" collab_splats tests evals docs/source` → none; `G` = previous + 1.
- [ ] **Step 7:** Commit `fix(splats): coarse-to-fine K from rescale_intrinsics on the exact resized grid`
  with body `Odd sizes: K now matches the (H // f, W // f) image instead of dividing by f.`

### Task 2.7: One kNN-spacing helper

**Files:** Modify `collab_splats/splats/utils.py`, `gaussian.py:151-156`, `scaffold.py:255-270,343`;
Test `tests/splats/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
def test_knn_spacing_is_the_rms_distance_to_the_k_nearest_neighbors():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (50, 3))
    spacing = knn_spacing(points, 3)

    # Brute force: sorted pairwise distances, self (0) excluded
    dists = np.sort(np.linalg.norm(points[:, None] - points[None], axis=-1), axis=1)[:, 1:4]
    assert spacing.shape == (50,)
    assert np.allclose(spacing, np.sqrt((dists**2).mean(-1)))
```

(add `knn_spacing` to the `test_utils.py` import). Run → FAIL with `ImportError`.

- [ ] **Step 2: `utils.py`** — add `from sklearn.neighbors import NearestNeighbors` and:

```python
def knn_spacing(points: np.ndarray, k: int) -> np.ndarray:
    """
    Per-point RMS distance to its k nearest neighbors, itself excluded.

    Args:
        points: (N, 3) positions.
        k: neighbors per point.

    Returns:
        (N,) spacing, in the units of `points`.
    """
    neighbor_dists, _ = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
    return np.sqrt((neighbor_dists[:, 1:] ** 2).mean(-1))
```

- [ ] **Step 3: `gaussian.py`** — replace the three kNN lines:

```python
        # Initial scale: RMS distance to the (knn - 1) nearest neighbors, stored as log-scale
        spacing = torch.from_numpy(knn_spacing(points, knn - 1)).float()
        log_scales = torch.log(spacing).unsqueeze(-1).repeat(1, 3)
```

`knn` keeps its meaning (neighbors including self); `test_gaussian.py:255` pins `knn=4` equal to
the default. Drop the `NearestNeighbors` import if unused.

- [ ] **Step 4: `scaffold.py`** — delete `median_knn_spacing`; the voxel line becomes

```python
            self.voxel_size = float(np.median(knn_spacing(points_t.numpy(), 3)) * cfg_scaffold.voxel_multiplier)
```

`points_t` is the float32 CPU tensor built two lines up, so the input dtype matches the old
`points.detach().cpu().numpy()`. Drop `NearestNeighbors` if unused.

- [ ] **Step 5:** Test PASSES; `P2` → `PARITY OK` (`voxel_size` and every decode key bit-equal);
  `gaussians/scales` is in `P2` too, so the vanilla initial scales are pinned bit-exact; `P1` OK;
  `G` = previous + 1. A `P2` failure on `gaussians/scales` alone means sklearn returns neighbors in
  a different order for `k + 1` vs `k`: revert and report, do not loosen.

- [ ] **Step 6:** Commit `refactor(splats): one knn_spacing helper for Gaussians and Scaffold`.

### Task 2.8: One Scaffold decode helper

**Files:** Modify `collab_splats/splats/scaffold.py` (`decode`, `export_gaussians`, new
`_offsets_to_gaussians` above `class Scaffold`); Test `tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the test**

```python
def test_offsets_to_gaussians_keeps_the_most_opaque_offset_when_all_are_closed():
    anchors = torch.zeros(2, 3)
    scaling = torch.ones(2, 6)
    offsets = torch.zeros(2, 3, 3)
    neural_opacity = torch.tensor([[-0.5], [-0.1], [-0.9], [-0.3], [-0.2], [-0.8]])
    cov = torch.zeros(6, 7)
    cov[:, 3] = 1.0
    color = torch.rand(6, 3)

    keep, gaussians = scaffold_module._offsets_to_gaussians(anchors, scaling, offsets, neural_opacity, cov, color, 3)

    assert keep.tolist() == [False, True, False, False, False, False]
    assert gaussians["opacities"].tolist() == pytest.approx([-0.1])
    assert set(gaussians) == {"means", "scales", "quats", "opacities", "colors"}
```

Run → FAIL with `AttributeError`.

- [ ] **Step 2: Add the helper** (module level, after `voxelize`):

```python
def _offsets_to_gaussians(
    anchors: Tensor,
    scaling: Tensor,
    offsets: Tensor,
    neural_opacity: Tensor,
    cov: Tensor,
    color: Tensor,
    n_offsets: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Anchor slots plus MLP heads -> the open neural Gaussians, and the keep mask over the slots.

    - an offset with non-positive opacity is dropped
    - none open: the most opaque one is kept, since gsplat's kernel and ply writer need >= 1 splat
    - means = anchor + offset x offset extent (scaling[:, :3]); scales = extent (scaling[:, 3:6])
      x sigmoid of the cov head
    """
    keep = (neural_opacity > 0).reshape(-1)
    if not bool(keep.any()):
        keep = torch.zeros_like(keep)
        keep[neural_opacity.reshape(-1).argmax()] = True

    cov = cov.reshape(-1, 7)[keep]
    gaussians = {
        "means": (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep],
        "scales": scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[keep] * torch.sigmoid(cov[:, :3]),
        "quats": F.normalize(cov[:, 3:7], dim=-1),
        "opacities": neural_opacity.reshape(-1)[keep],
        "colors": color.reshape(-1, 3)[keep],
    }
    return keep, gaussians
```

- [ ] **Step 3: `decode`** — from `# Offsets with non-positive opacity contribute nothing` through
  `colors = color.reshape(-1, 3)[keep]` becomes:

```python
        # Open offsets only; the helper keeps one when none is open (empty decode is a SIGFPE)
        keep, gaussians = _offsets_to_gaussians(anchors, scaling, offsets, neural_opacity, cov, color, n_offsets)
        slot_index = (anchor_ids[:, None] * n_offsets + torch.arange(n_offsets, device=anchors.device)).reshape(-1)
        decode_index = slot_index[keep]
        scales = gaussians["scales"]
```

and the `decoded` dict reads `means`, `quats`, `opacities`, `colors` from `gaussians`; the 2dgs
block still rebinds `scales`.

- [ ] **Step 4: `export_gaussians`** — from `keep = (neural_opacity > 0)...` through
  `colors = color.reshape(-1, 3)[keep]` becomes:

```python
        # Same decode as a render, one view direction per anchor; never an empty ply
        _, gaussians = _offsets_to_gaussians(anchors, scaling, offsets, neural_opacity, cov, color, n_offsets)
        means, scales, colors = gaussians["means"], gaussians["scales"], gaussians["colors"]
```

and the return dict reads `quats`/`opacities` from `gaussians`.

- [ ] **Step 5:** Test PASSES; `P2` → `PARITY OK`; `P1` OK; `G` = previous + 1.
- [ ] **Step 6:** Commit `refactor(splats): one Scaffold offsets-to-gaussians decode`.

### Task 2.9: `Scaffold` `adam_eps` kwarg

**Files:** Modify `collab_splats/splats/scaffold.py` (`__init__`, lines 376 and 394); Test
`tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the failing test**

`_field`'s extra kwargs go into the config block, not the constructor, so build directly (as
line ~200 does). `self.optimizers` already holds every param optimizer plus `mlp_optimizer`.

```python
def test_scaffold_adam_epsilon_is_a_kwarg():
    cfg = _scaffold_config(n_offsets=2, feat_dim=8)
    points, colors = _seed_points(n=200)
    default = Scaffold(cfg, points, colors, 1.0, n_views=1, device="cpu")
    custom = Scaffold(cfg, points, colors, 1.0, n_views=1, device="cpu", adam_eps=1e-8)

    # Every optimizer, the MLP one included, takes the kwarg; the default stays upstream's 1e-15
    assert {group["eps"] for opt in default.optimizers for group in opt.param_groups} == {1e-15}
    assert {group["eps"] for opt in custom.optimizers for group in opt.param_groups} == {1e-8}
```

Run → FAIL (`unexpected keyword argument 'adam_eps'`).

- [ ] **Step 2: Source** — `__init__` keyword-only block gains `adam_eps: float = 1e-15,` after
  `lr_decay`; Args line `adam_eps: Adam epsilon for every optimizer, 1e-15 as upstream.`; both
  `eps=1e-15` → `eps=adam_eps`.
- [ ] **Step 3:** PASS; `P2` OK; `G` = previous + 1.
- [ ] **Step 4:** Commit `refactor(splats): Scaffold adam_eps kwarg, matching Gaussians`.

---

## Task 3: Docstring contract

**Files:** Modify `tests/test_docstring_contract.py:22,198`, any `collab_splats/splats/*.py` hit

- [ ] **Step 1:** `cd $WT && $PY $SP/contract_hits.py` → fix every remaining hit, prose only
  (new code from Round 2 included). Prove prose-only against `HEAD` for each touched file.
- [ ] **Step 2:** `PACKAGES = ("preproc", "semantics", "pointcloud", "geometry", "splats")` and add
  `"splats"` to `RELEASED`.
- [ ] **Step 3:** `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/test_docstring_contract.py -q -p no:cacheprovider`
  → all pass. Then `G`.
- [ ] **Step 4:** Commit `test(contract): enforce the docstring contract on splats` (and a separate
  `docs(splats): …` commit first if Step 1 touched source).

## Task 4: Tutorial, graph, report

- [ ] **Step 1: Tutorial import** — `docs/source/tutorials/03_splats/train_splats.ipynb`: if it
  imports from `collab_splats.splats.rendering`, switch to `from collab_splats.splats import …` via
  `NotebookEdit` on the one cell (sibling `clean/tutorials` owns the rewrite; this is the minimal
  edit). Commit `docs(tutorials): splats notebook imports from the package`.
- [ ] **Step 2: Re-execute once, in tmux, not in the gate shell**

```bash
cd $WT && tmux new -d -s splats-nb "PYTHONPATH=$WT $PY -m jupyter nbconvert --to notebook --execute --output $SP/train_splats_run.ipynb docs/source/tutorials/03_splats/train_splats.ipynb > $SP/nb.log 2>&1; echo done >> $SP/nb.log"
```

Wait for `done` in `$SP/nb.log` (Monitor with an until-loop, not `pgrep -f`). Report success or
the first traceback. The executed copy stays in `$SP`; the committed notebook is not overwritten.

- [ ] **Step 3:** `cd $WT && graphify update .`
- [ ] **Step 4: Final gate and report** — `G` once more. Report against `$SP/baseline.txt`: pass /
  fail / skip deltas, each attributed to a task (deleted pgsr/plane/neighbor/absgrad tests, the 5
  added tests), the parity results, and the sibling notes (evals-release deletes the two
  eval scripts; mesh-release's `render_tsdf_inputs` goes in `checkpoint.py`). Do not merge.
