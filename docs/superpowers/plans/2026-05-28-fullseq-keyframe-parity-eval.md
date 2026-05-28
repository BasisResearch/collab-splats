# Full-Sequence Keyframe-Parity Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce VGGT-SLAM ~0.038m chess/seq-01 ATE with full sequence + min_disparity=50, then run VGGT-X / Omega / MapAnything on the exact same keyframes for a clean baseline comparison table.

**Architecture:** Shared ATE utility reads GT pose.txt files → TUM trajectory → evo alignment. VGGT-SLAM runner gets max_frames=None + min_disparity=50 defaults and saves selected keyframe paths. eval_gt.py filters dataset to those same paths before image prep.

**Tech Stack:** Python 3.11, `/opt/conda/envs/nerfstudio/bin/python`, evo (already installed), VGGT-SLAM solver (third_party), tmux for long runs.

---

## File map

| File | Action |
|------|--------|
| `evals/ate_utils.py` | **Create** — `_load_gt_as_tum_trajectory`, `compute_ate_rmse` |
| `evals/runners/run_vggt_slam_lc.py` | **Modify** — max_frames→None, min_disparity→50, save_keyframes, call compute_ate_rmse |
| `evals/eval_gt.py` | **Modify** — add `--keyframe_list` arg, filter dataset before _prepare_image_dir |

---

### Task 1: ATE utility module

**Files:**
- Create: `evals/ate_utils.py`

- [ ] **Step 1: Write `evals/ate_utils.py`**

```python
"""ATE computation against 7-Scenes GT poses using evo."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _load_gt_as_tum_trajectory(seq_dir: Path):
    """Load 7-Scenes GT cam-to-world poses → evo PoseTrajectory3D (cam-to-world)."""
    from evo.core.trajectory import PoseTrajectory3D

    pose_files = sorted(
        seq_dir.glob("frame-*.pose.txt"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    timestamps, positions, quats = [], [], []
    for pf in pose_files:
        idx = int(re.search(r"(\d+)", pf.stem).group(1))
        c2w = np.loadtxt(pf)  # (4,4) cam-to-world
        t = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # xyzw
        timestamps.append(float(idx))
        positions.append(t)
        quats.append(q)

    return PoseTrajectory3D(
        positions_xyz=np.array(positions),
        orientations_quat_wxyz=np.array(quats)[:, [3, 0, 1, 2]],  # xyzw→wxyz
        timestamps=np.array(timestamps),
    )


def compute_ate_rmse(tum_path: Path, seq_dir: Path) -> float:
    """Umeyama-aligned ATE RMSE (metres) between a TUM trajectory and 7-Scenes GT.

    tum_path: path to TUM file written by VGGT-SLAM (cam-to-world, frame index timestamps).
    seq_dir:  7-Scenes sequence directory containing frame-*.pose.txt files.
    """
    from evo.tools import file_interface
    from evo.core import metrics, sync
    import evo.main_ape as main_ape

    traj_est = file_interface.read_tum_trajectory_file(str(tum_path))
    traj_ref = _load_gt_as_tum_trajectory(seq_dir)

    # Associate by nearest timestamp (frame index).
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=1.0)

    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="est",
        pose_relation=metrics.PoseRelation.translation_part,
        align=True,
        correct_scale=True,
        verbose=False,
    )
    return float(result.stats["rmse"])
```

- [ ] **Step 2: Verify evo import works**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -c "
from evals.ate_utils import compute_ate_rmse
print('ate_utils imports OK')
"
```

Expected: `ate_utils imports OK`

- [ ] **Step 3: Commit**

```bash
git add evals/ate_utils.py
git commit -m "feat(evals): ate_utils — _load_gt_as_tum_trajectory + compute_ate_rmse"
```

---

### Task 2: Update run_vggt_slam_lc.py

**Files:**
- Modify: `evals/runners/run_vggt_slam_lc.py`

Key changes:
- `max_frames: int | None = None` (None = all frames)
- `min_disparity: float = 50.0` (VGGT-SLAM paper default)
- Track `selected_image_paths` list during frame-selection loop
- After writing TUM: save `selected_frames.txt`, call `compute_ate_rmse`, write `metrics.json`
- CLI: `--max_frames` default → `None` (pass `int | None` via argparse type)

- [ ] **Step 1: Update function signature**

In `evals/runners/run_vggt_slam_lc.py`, change the `run_vggt_slam_lc` signature:

```python
def run_vggt_slam_lc(
    seq_dir: Path,
    out_tum: Path,
    max_frames: int | None = None,      # None = all frames
    submap_size: int = 16,
    overlapping_window_size: int = 1,
    conf_threshold: float = 25.0,
    max_loops: int = 0,                 # 0 = disable LC (baseline run)
    min_disparity: float = 50.0,        # VGGT-SLAM paper default
    lc_thres: float = 0.95,
) -> None:
```

- [ ] **Step 2: Update image collection to respect max_frames=None and track selected**

Replace the existing image collection block (around `all_images = sorted(all_images)[:max_frames]`) with:

```python
    all_images = [
        f for f in glob.glob(str(seq_dir / "*"))
        if "depth" not in Path(f).name.lower()
        and "txt" not in Path(f).name.lower()
        and "db" not in Path(f).name.lower()
    ]
    all_images = sorted(all_images)
    if max_frames is not None:
        all_images = all_images[:max_frames]
    logger.info("Found %d images (max_frames=%s)", len(all_images), max_frames)
```

- [ ] **Step 3: Track selected_image_paths in the frame loop**

Add `selected_image_paths: list[str] = []` before the `for image_name in tqdm(all_images)` loop.

Inside the loop, in the `if enough_disparity:` block, add:

```python
        if enough_disparity:
            image_names_subset.append(image_name)
            image_count += 1
            selected_image_paths.append(image_name)   # track for keyframe export
```

- [ ] **Step 4: After write_poses_to_file — save keyframes + compute ATE + write metrics**

Replace the section after `solver.map.write_poses_to_file(...)` (keep the existing similarity-log block) and add immediately after the TUM write:

```python
    # Save selected keyframe paths for use by eval_gt.py --keyframe_list
    kf_path = out_tum.parent / "selected_frames.txt"
    kf_path.write_text("\n".join(selected_image_paths))
    logger.info("Keyframes (%d) saved → %s", len(selected_image_paths), kf_path)

    # Compute ATE against 7-Scenes GT
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evals.ate_utils import compute_ate_rmse
    try:
        ate_rmse = compute_ate_rmse(out_tum, seq_dir)
        logger.info("ATE RMSE: %.6f m", ate_rmse)
    except Exception as exc:
        logger.warning("ATE computation failed: %s", exc)
        ate_rmse = None

    import json as _json
    metrics_out = out_tum.parent / "metrics.json"
    metrics_out.write_text(_json.dumps({
        "ate_rmse": ate_rmse,
        "keyframes": len(selected_image_paths),
        "submaps": solver.map.get_num_submaps(),
        "loop_closures": solver.graph.get_num_loops(),
        "min_disparity": min_disparity,
        "max_frames": max_frames,
    }, indent=2))
    logger.info("Metrics → %s", metrics_out)
```

- [ ] **Step 5: Update CLI defaults**

In `main()`, update the parser:

```python
    parser.add_argument(
        "--out_tum",
        default="evals/baselines/vggt_slam/chess_seq01/vggt_slam_fullseq.tum",
        help="Output TUM path.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Max frames to process. None (default) = all frames in sequence.",
    )
    parser.add_argument("--submap_size", type=int, default=16)
    parser.add_argument("--conf_threshold", type=float, default=25.0)
    parser.add_argument(
        "--max_loops", type=int, default=0,
        help="Max loop closures per submap. 0 = disable LC (default for baseline runs).",
    )
    parser.add_argument(
        "--min_disparity",
        type=float,
        default=50.0,
        help="Min optical-flow disparity for keyframe selection. "
             "VGGT-SLAM paper default is 50. Use 0 to accept all frames.",
    )
```

Update the `run_vggt_slam_lc(...)` call at the bottom of `main()` to pass `max_loops=args.max_loops`.

- [ ] **Step 6: Verify the file is syntactically valid**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import ast, pathlib; ast.parse(pathlib.Path('evals/runners/run_vggt_slam_lc.py').read_text()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 7: Commit**

```bash
git add evals/runners/run_vggt_slam_lc.py
git commit -m "feat(evals): run_vggt_slam_lc — full-seq support, min_disparity=50 default, save keyframes + ATE"
```

---

### Task 3: Update eval_gt.py — keyframe_list support

**Files:**
- Modify: `evals/eval_gt.py`

- [ ] **Step 1: Add --keyframe_list argument**

In `_build_parser()`, after the existing `--max_frames` argument, add:

```python
    parser.add_argument(
        "--keyframe_list",
        type=Path,
        default=None,
        help="Path to selected_frames.txt from run_vggt_slam_lc.py. "
             "When set, filters dataset to only these frames (by filename) "
             "so all models run on identical keyframes as VGGT-SLAM.",
    )
```

- [ ] **Step 2: Apply filter in main() after dataset loading**

In `main()`, immediately after:
```python
    dataset = get_dataset(args.dataset)(args.seq_dir, max_frames=args.max_frames)
```

Add:

```python
    if args.keyframe_list is not None:
        allowed_names = set(Path(args.keyframe_list).read_text().splitlines())
        # Match by filename (basename) — robust to absolute vs relative path differences.
        allowed_basenames = {Path(p).name for p in allowed_names}
        indices = [
            i for i, img_path in enumerate(dataset.images)
            if Path(img_path).name in allowed_basenames
        ]
        from evals.datasets import EvalDataset
        dataset = EvalDataset(
            images=[dataset.images[i] for i in indices],
            gt_poses=dataset.gt_poses[indices],
        )
        logger.info(
            "keyframe_list: filtered to %d / %d frames",
            len(indices), len(indices) + (len(dataset.images) - len(indices)),
        )
```

Note: `EvalDataset` is a dataclass — check `evals/datasets.py` for the exact field names
(`images`, `gt_poses`, and optionally `intrinsics`). If it has `intrinsics`, pass
`intrinsics=dataset.intrinsics[indices]` as well.

- [ ] **Step 3: Verify syntax**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import ast, pathlib; ast.parse(pathlib.Path('evals/eval_gt.py').read_text()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add evals/eval_gt.py
git commit -m "feat(evals): eval_gt — add --keyframe_list to filter frames to VGGT-SLAM keyframe set"
```

---

### Task 4: Run VGGT-SLAM full-sequence eval

> **Run in tmux** — inference takes ~10-30 min for full chess seq-01.

- [ ] **Step 1: Launch in tmux**

```bash
tmux new-session -d -s vggt_slam_fullseq \
  "/opt/conda/envs/nerfstudio/bin/python evals/runners/run_vggt_slam_lc.py \
    --seq_dir data/7scenes/chess/seq-01 \
    --out_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_fullseq.tum \
    2>&1 | tee evals/baselines/vggt_slam/chess_seq01/fullseq_run.log"
```

This uses defaults: `min_disparity=50`, `max_frames=None`, `max_loops=0`.

- [ ] **Step 2: Monitor until complete**

```bash
tail -f evals/baselines/vggt_slam/chess_seq01/fullseq_run.log
```

Expected log lines:
- `Keyframes (N) saved → ...selected_frames.txt` (N should be ~30-60)
- `ATE RMSE: 0.0XXXXX m` (target: 0.037–0.045m)
- `Metrics → ...metrics.json`

- [ ] **Step 3: Verify result**

```bash
cat evals/baselines/vggt_slam/chess_seq01/metrics.json
wc -l evals/baselines/vggt_slam/chess_seq01/selected_frames.txt
wc -l evals/baselines/vggt_slam/chess_seq01/vggt_slam_fullseq.tum
```

Expected:
- `ate_rmse` ≈ 0.037–0.045 (matching issue #43)
- `selected_frames.txt` line count: 30–80 keyframes
- TUM line count: should match selected keyframes

**If ATE is much higher (>0.10m):** check that the TUM timestamps match GT frame indices.
The VGGT-SLAM `write_poses_to_file` may use a different timestamp scheme. If so, update
`_load_gt_as_tum_trajectory` to match the actual timestamps in the TUM file (inspect first
few lines of TUM and first few pose.txt filenames).

---

### Task 5: Run feedforward models with same keyframes

> Run each model sequentially (OOM risk with concurrent runs — CLAUDE.md cgroup cap 46.6 GB).
> Use tmux for each. Wait for completion before starting next.

- [ ] **Step 1: Run VGGT-X baseline**

```bash
tmux new-session -d -s eval_vggtx \
  "/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir data/7scenes/chess/seq-01 \
    --backbone vggtx \
    --conditions baseline \
    --keyframe_list evals/baselines/vggt_slam/chess_seq01/selected_frames.txt \
    --submap_size 16 \
    2>&1 | tee /tmp/eval_vggtx.log"
```

Monitor: `tail -f /tmp/eval_vggtx.log`

- [ ] **Step 2: Run Omega baseline**

After VGGT-X completes:

```bash
tmux new-session -d -s eval_omega \
  "/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir data/7scenes/chess/seq-01 \
    --backbone vggt_omega \
    --conditions baseline \
    --keyframe_list evals/baselines/vggt_slam/chess_seq01/selected_frames.txt \
    --submap_size 16 \
    2>&1 | tee /tmp/eval_omega.log"
```

- [ ] **Step 3: Run MapAnything baseline**

After Omega completes:

```bash
tmux new-session -d -s eval_mapanything \
  "/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir data/7scenes/chess/seq-01 \
    --backbone mapanything \
    --conditions baseline \
    --keyframe_list evals/baselines/vggt_slam/chess_seq01/selected_frames.txt \
    --submap_size 16 \
    2>&1 | tee /tmp/eval_mapanything.log"
```

- [ ] **Step 4: Collect ATE results**

Each eval_gt.py run writes `metrics.json` in its output dir under `evals/results/`.
Collect:

```bash
for f in evals/results/7scenes/seq-01/run-*/metrics.json; do
  echo "$f:"; cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ate_rmse','?'), d.get('backbone','?'), d.get('condition','?'))"; echo
done
```

- [ ] **Step 5: Report results table in worklog**

Append to `worklog/WORKLOG.md`:

```markdown
### 2026-05-28 — full-seq keyframe-parity eval

**Setup**: chess/seq-01 full sequence, min_disparity=50, same keyframes across all models.
Reproduces VGGT-SLAM paper setup (issue #43 fix applied).

| Model | Condition | ATE RMSE | Keyframes | Submaps |
|-------|-----------|----------|-----------|---------|
| VGGT-SLAM (SPARK) | baseline | X.XXXm | N | M |
| VGGT-X | baseline | X.XXXm | N | M |
| Omega | baseline | X.XXXm | N | M |
| MapAnything | baseline | X.XXXm | N | M |

**Finding**: [fill in after runs complete]
```

- [ ] **Step 6: Report results here in the conversation**

Post the filled-in table as a reply so the user has the comparison at a glance.

---

## Self-review

**Spec coverage:**
- ✓ max_frames=None + min_disparity=50 → Task 2
- ✓ save selected_frames.txt → Task 2 Step 3-4
- ✓ inline ATE → Task 2 Step 4, ate_utils Task 1
- ✓ eval_gt.py --keyframe_list → Task 3
- ✓ run all three models → Task 5
- ✓ comparison table → Task 5 Step 5-6

**Placeholder check:** All steps have exact commands and code. No TBDs.

**Type consistency:**
- `compute_ate_rmse(tum_path: Path, seq_dir: Path) → float` used consistently in Tasks 1, 2
- `EvalDataset` fields: `images`, `gt_poses` — verify `intrinsics` field in Task 3 Step 2 note
- `selected_image_paths: list[str]` populated in Task 2 Step 3, consumed in Task 2 Step 4
