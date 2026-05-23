# Eval Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize eval infrastructure: gitignore results, replace `ba_hightrack` with parameterized `ba_track-density-N`, make `--output_dir` auto-named, and create a unified `ground-truth-evals.ipynb` notebook with a `docs/pointcloud/README.md`.

**Architecture:** `eval_gt.py` stays as the headless CLI workhorse. The notebook wraps it with helper functions (`run_eval`, `load_results`, `compare_runs`, `plot_trajectories`, `plot_ate_per_frame`) — no globals. Results live in `evals/results/{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}/`.

**Tech Stack:** Python 3.10, argparse, numpy, matplotlib, pandas, Jupyter nbformat 4.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `evals/eval_gt.py` | Dynamic condition parsing; optional `--output_dir`; auto-naming |
| Modify | `tests/evals/test_eval_gt_helpers.py` | Tests for `_make_creator` (density) and `_default_output_dir` |
| Modify | `.gitignore` | Ignore `evals/results/` |
| Create | `docs/pointcloud/ground-truth-evals.ipynb` | Unified notebook: concepts + run + load + plot |
| Create | `docs/pointcloud/README.md` | 1-page written overview |
| Delete | `docs/pointcloud/eval_7scenes_gt.ipynb` | Retired |
| Delete | `docs/pointcloud/eval_co3dv2_gt.ipynb` | Retired |

---

## Task 1: Gitignore `evals/results/`

`evals/results/` already exists on disk. It just needs to be gitignored.

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add entry**

Open `.gitignore` and append:
```
evals/results/
```

- [ ] **Step 2: Verify gitignore takes effect**

Run: `git status --short evals/results/`
Expected: no output (directory is ignored)

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore evals/results/"
```

---

## Task 2: Dynamic `ba_track-density-N` condition in `eval_gt.py`

Replace the hardcoded `ba_hightrack` with a pattern-parsed `ba_track-density-{N}` condition. `N` is extracted from the condition string and passed as `max_query_pts` to `BundleAdjustmentConfig`. `query_frame_num = max(5, N // 512)`.

**Files:**
- Modify: `evals/eval_gt.py` (lines 43–97, 179–198)
- Modify: `tests/evals/test_eval_gt_helpers.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/evals/test_eval_gt_helpers.py`:
```python
def test_make_creator_ba_track_density_4096():
    """ba_track-density-4096 → BundleAdjustment with max_query_pts=4096, query_frame_num=8."""
    from eval_gt import _make_creator
    from collab_splats.pointcloud.wrappers import BundleAdjustment
    creator = _make_creator("ba_track-density-4096")
    assert isinstance(creator, BundleAdjustment)
    assert creator.config.max_query_pts == 4096
    assert creator.config.query_frame_num == 8  # max(5, 4096 // 512)


def test_make_creator_ba_track_density_2048():
    """ba_track-density-2048 → query_frame_num=5 (max(5, 2048//512) = max(5,4) = 5)."""
    from eval_gt import _make_creator
    from collab_splats.pointcloud.wrappers import BundleAdjustment
    creator = _make_creator("ba_track-density-2048")
    assert isinstance(creator, BundleAdjustment)
    assert creator.config.max_query_pts == 2048
    assert creator.config.query_frame_num == 5


def test_validate_condition_accepts_known():
    from eval_gt import _validate_condition
    for cond in ["baseline", "ba", "lc", "ba_track-density-4096", "ba_track-density-1024"]:
        _validate_condition(cond)  # must not raise


def test_validate_condition_rejects_ba_hightrack():
    """ba_hightrack is no longer valid — was replaced by ba_track-density-N."""
    import pytest
    from eval_gt import _validate_condition
    with pytest.raises(ValueError, match="ba_hightrack"):
        _validate_condition("ba_hightrack")


def test_validate_condition_rejects_unknown():
    import pytest
    from eval_gt import _validate_condition
    with pytest.raises(ValueError):
        _validate_condition("mystery_condition")
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_eval_gt_helpers.py::test_make_creator_ba_track_density_4096 tests/evals/test_eval_gt_helpers.py::test_validate_condition_rejects_ba_hightrack -v`
Expected: FAIL — `_validate_condition` not defined; `_make_creator("ba_track-density-4096")` falls through to `return base`

- [ ] **Step 3: Replace lines 43–45 in `evals/eval_gt.py`**

Old:
```python
_CONDITION_CHOICES = ["baseline", "ba", "lc", "ba_hightrack"]

_COLORS = {"gt": "black", "baseline": "tab:red", "ba": "tab:blue", "lc": "tab:green", "ba_hightrack": "tab:purple"}
```

New:
```python
_FIXED_CONDITIONS = {"baseline", "ba", "lc"}
_COLORS = {"gt": "black", "baseline": "tab:red", "ba": "tab:blue", "lc": "tab:green"}


def _validate_condition(cond: str) -> None:
    """Raise ValueError if cond is not a recognised condition string."""
    import re
    if cond in _FIXED_CONDITIONS:
        return
    if re.fullmatch(r"ba_track-density-\d+", cond):
        return
    raise ValueError(
        f"Unknown condition {cond!r}. "
        f"Valid: {sorted(_FIXED_CONDITIONS)} or ba_track-density-{{N}} (e.g. ba_track-density-4096)"
    )
```

- [ ] **Step 4: Replace `_make_creator` (lines 63–97)**

```python
def _make_creator(condition: str, submap_size: int | None = None):
    """Build a creator for the given condition.

    Conditions:
        baseline             VGGT-X feedforward, no refinement
        ba                   + bundle adjustment (max_query_pts=2048, query_frame_num=5)
        ba_track-density-N   + bundle adjustment with N query pts; query_frame_num=max(5, N//512)
        lc                   full loop closure pipeline

    When submap_size is set, baseline and ba use windowed inference (LoopClosure with
    detection disabled) so sequences exceeding GPU memory can be processed in windows.
    lc always uses the full loop-closure pipeline regardless of submap_size.
    """
    import re
    base = get_creator("vggtx")()
    if condition == "lc":
        return LoopClosure(base)
    m = re.fullmatch(r"ba_track-density-(\d+)", condition)
    if m:
        n = int(m.group(1))
        cfg = BundleAdjustmentConfig(
            max_query_pts=n,
            query_frame_num=max(5, n // 512),
        )
        if submap_size is not None:
            _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_cosine_threshold=1.0)
            windowed = LoopClosure(base, config=_no_lc_cfg)
            return BundleAdjustment(windowed, config=cfg)
        return BundleAdjustment(base, config=cfg)
    if submap_size is not None:
        _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_cosine_threshold=1.0)
        windowed = LoopClosure(base, config=_no_lc_cfg)
        if condition == "ba":
            return BundleAdjustment(windowed)
        return windowed  # baseline
    if condition == "ba":
        return BundleAdjustment(base)
    return base  # baseline
```

- [ ] **Step 5: Remove `choices=` from `--conditions` arg in `_build_parser`**

Old:
```python
parser.add_argument("--conditions", nargs="+", default=["baseline", "ba", "lc"],
                    choices=_CONDITION_CHOICES)
```

New:
```python
parser.add_argument("--conditions", nargs="+", default=["baseline", "ba", "lc"],
                    help="Conditions: baseline | ba | lc | ba_track-density-{N}")
```

- [ ] **Step 6: Add validation in `main()` after `args = _build_parser().parse_args()`**

Add immediately after that line (before the subprocess leaf check):
```python
    for cond in (args.conditions or []):
        _validate_condition(cond)
    if args._condition is not None:
        _validate_condition(args._condition)
```

- [ ] **Step 7: Run all eval helper tests**

Run: `cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add evals/eval_gt.py tests/evals/test_eval_gt_helpers.py
git commit -m "feat(eval): replace ba_hightrack with parameterised ba_track-density-N condition"
```

---

## Task 3: Auto-naming `--output_dir` in `eval_gt.py`

`--output_dir` becomes optional. When omitted, path is auto-generated as `evals/results/{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}/`. Subprocess invocations inside `main()` already pass explicit `--output_dir`, so no change needed there.

**Files:**
- Modify: `evals/eval_gt.py`
- Modify: `tests/evals/test_eval_gt_helpers.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/evals/test_eval_gt_helpers.py`:
```python
def test_default_output_dir_structure():
    """Auto path: evals/results/{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}/"""
    import re
    from eval_gt import _default_output_dir
    result = _default_output_dir("co3dv2", Path("/data/co3dv2/apple/seq1"))
    parts = list(result.parts)
    idx = parts.index("results")
    assert parts[idx - 1] == "evals"
    assert parts[idx + 1] == "co3dv2"
    assert parts[idx + 2] == "seq1"
    assert re.fullmatch(r"run-\d{8}-\d{6}", parts[idx + 3])


def test_default_output_dir_uses_basename():
    """seq_name = last component of seq_dir."""
    from eval_gt import _default_output_dir
    r1 = _default_output_dir("7scenes", Path("/long/path/chess"))
    r2 = _default_output_dir("7scenes", Path("/other/chess"))
    assert r1.parent.name == "chess"
    assert r2.parent.name == "chess"
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_eval_gt_helpers.py::test_default_output_dir_structure -v`
Expected: FAIL — `_default_output_dir` not defined

- [ ] **Step 3: Add `datetime` import and `_default_output_dir` to `evals/eval_gt.py`**

Add `from datetime import datetime` to the existing imports block (after `import time`).

Add `_default_output_dir` after `_validate_condition`:
```python
def _default_output_dir(dataset: str, seq_dir: Path) -> Path:
    """Auto-generate a timestamped output directory under evals/results/."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("evals/results") / dataset / seq_dir.name / f"run-{ts}"
```

- [ ] **Step 4: Make `--output_dir` optional in `_build_parser`**

Old:
```python
parser.add_argument("--output_dir", type=Path, required=True,
                    help="Where to write results")
```

New:
```python
parser.add_argument("--output_dir", type=Path, default=None,
                    help="Where to write results "
                         "(default: evals/results/{dataset}/{seq_name}/run-{timestamp})")
```

- [ ] **Step 5: Wire auto-naming into `main()`**

In `main()`, after the subprocess leaf check block and before the orchestrator section, add:
```python
    # ── auto-name output dir if not provided ──────────────────────────────────
    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.dataset, args.seq_dir)
        print(f"Output directory: {args.output_dir}")
```

- [ ] **Step 6: Run all eval helper tests**

Run: `cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add evals/eval_gt.py tests/evals/test_eval_gt_helpers.py
git commit -m "feat(eval): make --output_dir optional; auto-name under evals/results/{dataset}/{seq}/{ts}"
```

---

## Task 4: `docs/pointcloud/README.md`

**Files:**
- Create: `docs/pointcloud/README.md`

- [ ] **Step 1: Write README**

```markdown
# Ground-Truth Evaluation

This directory contains notebooks and documentation for evaluating camera pose estimation accuracy against ground-truth trajectories.

## What is being evaluated?

Each pipeline takes a sequence of images and predicts a **camera pose** (position + orientation) for every frame. We compare these predictions against **ground-truth poses** recorded by the dataset (e.g. a motion capture system or an RGB-D sensor).

### Metrics

| Metric | Plain English | Units | Better = |
|--------|--------------|-------|----------|
| **ATE RMSE** | "On average, how far off was the predicted camera position?" After optimal rigid alignment between predicted and GT trajectories. | metres | Lower |
| **RPE trans RMSE** | "Between consecutive frames, how much does the predicted motion drift from the true motion?" Captures local drift. | metres | Lower |
| **AUC@30** | "What fraction of frames had position error under 30 cm?" | % (0–100) | Higher |

### Conditions

| Condition | What it runs | When to use |
|-----------|-------------|-------------|
| `baseline` | VGGT-X feedforward only — no refinement | Fastest; use as starting point |
| `ba` | `baseline` + bundle adjustment (2048 tracks, 5 query frames) | Default improvement step |
| `ba_track-density-{N}` | `baseline` + BA with N feature tracks per frame | When `ba` leaves residual drift; slower |
| `lc` | Full loop closure — detects revisited places, corrects drift globally | Long sequences where camera loops back |

Use `--submap_size 50` for sequences longer than ~200 frames (GPU memory limit).

## Supported datasets

| Dataset | Type | Download |
|---------|------|----------|
| 7-Scenes | Indoor RGB-D | `evals/download_7scenes.sh` |
| TUM RGB-D | Indoor RGB-D | `evals/download_tum.sh` |
| CO3Dv2 | Object-centric | `evals/download_co3dv2.sh` |
| KITTI | Outdoor driving | `evals/download_kitti.sh` |
| Waymo | Outdoor driving | `evals/download_waymo.sh` |

## Running an evaluation (headless)

Run from the repo root:

```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
    --dataset   co3dv2 \
    --seq_dir   /data/co3dv2/apple/110_13051_23361 \
    --conditions baseline ba ba_track-density-4096 \
    --submap_size 50
```

Results are written to `evals/results/{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}/`.

## Interactive use

Open `docs/pointcloud/ground-truth-evals.ipynb` for the full interactive experience: concepts explained, run from the notebook, load and compare results across runs.
```

- [ ] **Step 2: Commit**

```bash
git add docs/pointcloud/README.md
git commit -m "docs(pointcloud): add ground-truth eval README"
```

---

## Task 5: Notebook — Concepts + `run_eval`

Create `docs/pointcloud/ground-truth-evals.ipynb` as a valid nbformat 4 notebook. Write it with the `Write` tool. The notebook must be **run from the repo root** (`/workspace/collab-splats`).

**Files:**
- Create: `docs/pointcloud/ground-truth-evals.ipynb`

- [ ] **Step 1: Create notebook**

Write `docs/pointcloud/ground-truth-evals.ipynb` with the following cells (valid nbformat 4 JSON):

**Cell 1** — markdown:
```markdown
# Ground-Truth Evaluation

Single entry point for running ground-truth camera pose evaluations and comparing results across conditions and datasets.

**Run this notebook from the repo root** (`/workspace/collab-splats`) so relative paths resolve correctly.

---

## 1. What are we measuring?

Each pipeline takes a sequence of images and predicts a **camera pose** (position + orientation) for every frame. We compare against **ground-truth poses** recorded by the dataset (e.g. a motion capture system or RGB-D sensor).

### Metrics

| Metric | Plain English | Units | Better = |
|--------|--------------|-------|----------|
| **ATE RMSE** | "On average, how far off was the predicted camera position?" After optimal rigid alignment between predicted and GT trajectories. | metres | Lower |
| **RPE trans RMSE** | "Between consecutive frames, how much does predicted motion drift from true motion?" Captures local drift. | metres | Lower |
| **AUC@30** | "What fraction of frames had position error under 30 cm?" | % (0–100) | Higher |

### Conditions

| Condition | What it runs | When to use |
|-----------|-------------|-------------|
| `baseline` | VGGT-X feedforward only — no refinement | Fastest; use as starting point |
| `ba` | `baseline` + bundle adjustment (2048 tracks, 5 query frames) | Default improvement step |
| `ba_track-density-{N}` | `baseline` + BA with N feature tracks per frame. E.g. `ba_track-density-4096` doubles the track count vs `ba`. | When `ba` leaves residual drift; slower |
| `lc` | Full loop closure — detects when camera revisits a place, corrects accumulated drift globally via pose graph optimisation | Long sequences where the camera loops back |

### When to use `submap_size`

VGGT-X loads all frames into GPU at once. Sequences longer than ~200 frames exceed GPU memory. Pass `submap_size=50` to process in overlapping windows of 50 frames. `baseline` and `ba` run per-window; `lc` stitches windows with loop closure.

### What does "error" mean?

All metrics compare predicted camera positions to ground-truth positions. Lower error = the pipeline estimated camera location more accurately.
```

**Cell 2** — markdown:
```markdown
---

## 2. Run an evaluation

Call `run_eval()` with your dataset, sequence directory, and desired conditions. Results are saved to `evals/results/{dataset}/{seq_name}/run-{timestamp}/` and the path is returned.
```

**Cell 3** — code:
```python
import subprocess
import sys
from pathlib import Path


def run_eval(
    dataset: str,
    seq_dir: str | Path,
    conditions: list[str] | None = None,
    submap_size: int | None = None,
    max_frames: int = 500,
) -> Path:
    """Run eval_gt.py and return the output run directory path.

    Results are written to evals/results/{dataset}/{seq_name}/run-{timestamp}/.
    Run this notebook from /workspace/collab-splats so relative paths resolve.

    Args:
        dataset:      Dataset name: 7scenes | tum | co3dv2 | kitti | waymo
        seq_dir:      Path to the sequence directory.
        conditions:   Conditions to evaluate. Default: [baseline, ba, lc].
                      Options: baseline | ba | lc | ba_track-density-{N}
        submap_size:  Window size for sequences >~200 frames. None = single-pass.
        max_frames:   Maximum frames to load from the sequence.

    Returns:
        Path to the run directory containing metrics.json and trajectories.npz.
    """
    if conditions is None:
        conditions = ["baseline", "ba", "lc"]

    cmd = [
        sys.executable, "evals/eval_gt.py",
        "--dataset", dataset,
        "--seq_dir", str(seq_dir),
        "--max_frames", str(max_frames),
        "--conditions", *conditions,
    ]
    if submap_size is not None:
        cmd += ["--submap_size", str(submap_size)]

    subprocess.run(cmd, check=True)

    # Discover the most recently created run dir for this dataset/seq
    seq_name = Path(seq_dir).name
    run_dirs = sorted(
        (Path("evals/results") / dataset / seq_name).glob("run-*"),
        key=lambda p: p.name,
    )
    if not run_dirs:
        raise RuntimeError(
            f"No run directory found under evals/results/{dataset}/{seq_name}"
        )
    return run_dirs[-1]
```

**Cell 4** — code:
```python
# Example — edit these values for your sequence
out = run_eval(
    dataset="co3dv2",
    seq_dir="/data/co3dv2/apple/110_13051_23361",
    conditions=["baseline", "ba", "ba_track-density-4096"],
    submap_size=50,
)
print(f"Results written to: {out}")
```

- [ ] **Step 2: Verify valid JSON**

Run: `python -c "import json; json.load(open('docs/pointcloud/ground-truth-evals.ipynb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/pointcloud/ground-truth-evals.ipynb
git commit -m "feat(docs): add ground-truth-evals.ipynb — concepts + run_eval"
```

---

## Task 6: Notebook — `load_results` + `compare_runs`

**Files:**
- Modify: `docs/pointcloud/ground-truth-evals.ipynb`

Append two cells. Use `NotebookEdit` or re-write the full notebook with `Write`.

**Cell 5** — markdown:
```markdown
---

## 3. Load and compare results

`load_results()` discovers all completed runs under `evals/results/`. `compare_runs()` renders a summary DataFrame — call with no arguments to compare all runs, or pass specific run directories for a subset. Columns: run, condition, ATE RMSE, RPE trans, AUC@30, runtime.
```

**Cell 6** — code:
```python
import json
import pandas as pd
from pathlib import Path


def load_results(
    results_root: str | Path = "evals/results",
    dataset: str | None = None,
    seq_name: str | None = None,
) -> dict[str, dict]:
    """Discover all run-{YYYYMMDD-HHMMSS} directories containing metrics.json.

    Returns {"{dataset}/{seq_name}/run-{timestamp}": metrics_dict}.
    Optionally filter by dataset and/or seq_name.
    results_root is relative to repo root.
    """
    root = Path(results_root)
    results = {}
    for metrics_file in sorted(root.glob("*/*/run-*/metrics.json")):
        run_dir = metrics_file.parent
        ds = run_dir.parts[-3]
        seq = run_dir.parts[-2]
        if dataset is not None and ds != dataset:
            continue
        if seq_name is not None and seq != seq_name:
            continue
        key = f"{ds}/{seq}/{run_dir.name}"
        results[key] = json.loads(metrics_file.read_text())
    return results


def compare_runs(
    run_dirs: list[str | Path] | None = None,
    results_root: str | Path = "evals/results",
) -> pd.DataFrame:
    """Return a DataFrame summarising ATE / RPE / AUC / runtime per run and condition.

    If run_dirs is None, discovers all runs under results_root.
    Pass a single-element list to inspect one run: compare_runs([run_dir]).
    """
    if run_dirs is not None:
        entries = {}
        for rd in run_dirs:
            rd = Path(rd)
            mf = rd / "metrics.json"
            if not mf.exists():
                raise FileNotFoundError(f"metrics.json not found in {rd}")
            key = "/".join(rd.parts[-3:])
            entries[key] = json.loads(mf.read_text())
    else:
        entries = load_results(results_root)

    rows = []
    for run_key, metrics in entries.items():
        for cond, m in metrics.items():
            rows.append({
                "run": run_key,
                "condition": cond,
                "ATE RMSE (m)": m.get("ate", {}).get("rmse"),
                "RPE trans (m)": m.get("rpe", {}).get("trans_rmse"),
                "AUC@30 (%)": m.get("auc_30"),
                "time (s)": m.get("time_s"),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["run", "condition"]).reset_index(drop=True)
```

**Cell 7** — code:
```python
# Load all results and display a summary table
df = compare_runs()
df
```

- [ ] **Step 2: Verify valid JSON**

Run: `python -c "import json; json.load(open('docs/pointcloud/ground-truth-evals.ipynb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/pointcloud/ground-truth-evals.ipynb
git commit -m "feat(docs): add load_results + compare_runs to ground-truth-evals.ipynb"
```

---

## Task 7: Notebook — Trajectory + per-frame ATE plots

**Files:**
- Modify: `docs/pointcloud/ground-truth-evals.ipynb`

**Cell 8** — markdown:
```markdown
---

## 4. Trajectory visualisation

`plot_trajectories()` renders a 3D plot comparing the GT trajectory (black) against each condition's predicted trajectory. Pass a run directory path returned by `run_eval()` or discovered via `load_results()`.
```

**Cell 9** — code:
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_COND_COLORS = {
    "gt": "black",
    "baseline": "tab:red",
    "ba": "tab:blue",
    "lc": "tab:green",
}


def _color_for(cond: str) -> str:
    return _COND_COLORS.get(cond, "tab:purple")


def _cam_positions(poses: np.ndarray) -> np.ndarray:
    """(N,4,4) world-to-cam → (N,3) camera positions in world."""
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)


def plot_trajectories(
    run_dir: str | Path,
    conditions: list[str] | None = None,
) -> None:
    """3D trajectory comparison: GT (black) vs each predicted condition.

    Args:
        run_dir:    Path to a run directory containing trajectories.npz.
        conditions: Subset of conditions to plot. None = all available.
    """
    run_dir = Path(run_dir)
    npz = np.load(run_dir / "trajectories.npz", allow_pickle=False)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    gt_pos = _cam_positions(npz["gt"])
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
            label="gt", color="black", linewidth=2)

    available = [k[5:] for k in npz.files if k.startswith("pred_")]
    to_plot = conditions if conditions is not None else available
    for cond in to_plot:
        key = f"pred_{cond}"
        if key not in npz.files:
            print(f"Warning: {key} not in trajectories.npz — skipping")
            continue
        pos = _cam_positions(npz[key])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                label=cond, color=_color_for(cond), linewidth=1)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Camera Trajectory — {run_dir.name}")
    ax.legend()
    plt.tight_layout()
    plt.show()
```

**Cell 10** — markdown:
```markdown
---

## 5. Per-frame ATE

`plot_ate_per_frame()` shows how position error accumulates frame-by-frame. Spikes reveal where a condition drifts — useful for diagnosing where loop closure or denser BA tracks make the most difference.
```

**Cell 11** — code:
```python
def plot_ate_per_frame(
    run_dir: str | Path,
    conditions: list[str] | None = None,
) -> None:
    """Line chart of per-frame ATE for each condition.

    Args:
        run_dir:    Path to a run directory containing trajectories.npz.
        conditions: Subset of conditions to plot. None = all available.
    """
    run_dir = Path(run_dir)
    npz = np.load(run_dir / "trajectories.npz", allow_pickle=False)

    available = [k[len("ate_per_frame_"):] for k in npz.files if k.startswith("ate_per_frame_")]
    to_plot = conditions if conditions is not None else available

    fig, ax = plt.subplots(figsize=(12, 4))
    for cond in to_plot:
        key = f"ate_per_frame_{cond}"
        if key not in npz.files:
            print(f"Warning: {key} not in trajectories.npz — skipping")
            continue
        per_frame = npz[key]
        rmse = float(np.sqrt(np.mean(per_frame ** 2)))
        ax.plot(per_frame, label=f"{cond} (RMSE={rmse:.3f}m)", color=_color_for(cond))

    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.set_title(f"Per-frame Absolute Trajectory Error — {run_dir.name}")
    ax.legend()
    plt.tight_layout()
    plt.show()
```

**Cell 12** — code:
```python
# Replace with an actual run directory path
# run_dir = "evals/results/co3dv2/apple/run-20260520-143201"
# plot_trajectories(run_dir)
# plot_ate_per_frame(run_dir)
```

- [ ] **Step 2: Verify valid JSON**

Run: `python -c "import json; json.load(open('docs/pointcloud/ground-truth-evals.ipynb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/pointcloud/ground-truth-evals.ipynb
git commit -m "feat(docs): add trajectory + per-frame ATE plots to ground-truth-evals.ipynb"
```

---

## Task 8: Retire old notebooks

**Files:**
- Delete: `docs/pointcloud/eval_7scenes_gt.ipynb`
- Delete: `docs/pointcloud/eval_co3dv2_gt.ipynb`

- [ ] **Step 1: Delete and commit**

```bash
git rm docs/pointcloud/eval_7scenes_gt.ipynb docs/pointcloud/eval_co3dv2_gt.ipynb
git commit -m "chore(docs): retire eval_7scenes_gt + eval_co3dv2_gt — superseded by ground-truth-evals.ipynb"
```

---

## Self-Review

**Spec coverage:**
| Spec requirement | Task |
|-----------------|------|
| `evals/results/` gitignored | Task 1 |
| `ba_track-density-N` dynamic parsing | Task 2 |
| `_validate_condition` function | Task 2 |
| `--output_dir` optional, auto-named | Task 3 |
| `docs/pointcloud/README.md` | Task 4 |
| Notebook Section 1: concepts | Task 5 |
| Notebook Section 2: `run_eval` | Task 5 |
| Notebook Section 3: `load_results` + `compare_runs` | Task 6 |
| Notebook Section 4: `plot_trajectories` | Task 7 |
| Notebook Section 5: `plot_ate_per_frame` | Task 7 |
| Retire old notebooks | Task 8 |

**Type consistency:**
- `run_eval` → `Path` ✓
- `load_results` → `dict[str, dict]` ✓
- `compare_runs` → `pd.DataFrame` ✓
- `plot_trajectories`, `plot_ate_per_frame` → `None` ✓
- `_cam_positions(poses: np.ndarray) → np.ndarray` used identically in notebook (Task 7) ✓
- `_color_for` defined in Task 7 Cell 9, used by both `plot_trajectories` and `plot_ate_per_frame` in same cell block ✓
- `compare_runs([run_dir])` replaces removed `compare_conditions` — no orphan references ✓
