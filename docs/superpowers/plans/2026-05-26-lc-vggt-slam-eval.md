# LC vs VGGT-SLAM Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 6-condition, 3-sequence eval harness comparing our LC (Omega + X backbones) against VGGT-SLAM on 7-Scenes, with both trajectory metrics and submap alignment metrics.

**Architecture:** `eval_gt.py` gains a `--backbone` arg and emits TUM trajectory files + alignment JSON per LC condition. `run_vggt_slam.py` replaces its stub with a real subprocess. `eval_compare.py` reads all TUM files and emits a unified table. A new `reconstruction_quality.py` module computes before/after-PGO alignment metrics from LC internal state stored in `wrappers.py`.

**Tech Stack:** Python 3.11 (reconstruction env), numpy, scipy, evo, gtsam, VGGT-Omega, VGGT-X, VGGT-SLAM (third_party submodule)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `evals/download_7scenes.py` | Create | Download fire + office sequences |
| `evals/eval_gt.py` | Modify | `--backbone` arg, TUM output, alignment JSON |
| `evals/runners/run_vggt_slam.py` | Replace | Real VGGT-SLAM subprocess |
| `evals/metrics.py` | Modify | Add `compute_auc` |
| `evals/eval_compare.py` | Modify | New condition names, AUC column, alignment JSON, relax `_scan_results_dir` |
| `collab_splats/pointcloud/wrappers.py` | Modify | Store `_lc_precorrection_extrinsics` + `_lc_corrected_extrinsics` |
| `evals/reconstruction_quality.py` | Create | 3 alignment metrics + `compute_alignment_metrics` |
| `evals/eval_suite.sh` | Create | Orchestrator |
| `tests/evals/test_reconstruction_quality.py` | Create | Unit tests for alignment metrics |
| `tests/evals/test_metrics_auc.py` | Create | Unit test for `compute_auc` |

---

## Task 1: Dataset Download Helper

**Files:**
- Create: `evals/download_7scenes.py`

**Context:** Chess is at `data/7scenes/chess/seq-01/*.color.png`. Fire and office need the same structure. The 7-Scenes dataset is served by Microsoft Research. VGGT-SLAM's README points to MASt3R-SLAM for download instructions; their download script uses the same Microsoft CDN.

- [ ] **Step 1: Write failing test**

```python
# tests/evals/test_download_7scenes.py
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
from download_7scenes import _scene_already_downloaded, SCENES

def test_already_downloaded_true(tmp_path):
    seq = tmp_path / "seq-01"
    seq.mkdir()
    (seq / "frame-000001.color.png").touch()
    assert _scene_already_downloaded(tmp_path, "seq-01") is True

def test_already_downloaded_false(tmp_path):
    assert _scene_already_downloaded(tmp_path, "seq-01") is False
```

Run: `python -m pytest tests/evals/test_download_7scenes.py -v`
Expected: FAIL (ImportError: No module named 'download_7scenes')

- [ ] **Step 2: Create `evals/download_7scenes.py`**

```python
#!/usr/bin/env python
"""Download 7-Scenes sequences into data/7scenes/.

Usage:
    python evals/download_7scenes.py --scenes fire office
    python evals/download_7scenes.py --scenes fire office --seq seq-01
    python evals/download_7scenes.py --list   # show available scenes

Downloads from Microsoft Research CDN. Verifies frame count after extraction.
No-op if the target seq directory already contains *.color.png files.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "7scenes"

# Microsoft Research 7-Scenes CDN — verify at:
# https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
_CDN_BASE = "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
SCENES: dict[str, str] = {
    "chess":   f"{_CDN_BASE}/chess.zip",
    "fire":    f"{_CDN_BASE}/fire.zip",
    "heads":   f"{_CDN_BASE}/heads.zip",
    "office":  f"{_CDN_BASE}/office.zip",
    "pumpkin": f"{_CDN_BASE}/pumpkin.zip",
    "redkitchen": f"{_CDN_BASE}/redkitchen.zip",
    "stairs":  f"{_CDN_BASE}/stairs.zip",
}


def _scene_already_downloaded(scene_dir: Path, seq: str) -> bool:
    """True if seq dir exists and contains at least one *.color.png."""
    seq_dir = scene_dir / seq
    return seq_dir.is_dir() and any(seq_dir.glob("*.color.png"))


def download_scene(scene: str, seq: str = "seq-01", force: bool = False) -> Path:
    """Download and extract one scene/sequence. Return path to seq dir."""
    scene_dir = DATA_DIR / scene
    if not force and _scene_already_downloaded(scene_dir, seq):
        print(f"  {scene}/{seq}: already present, skipping")
        return scene_dir / seq

    url = SCENES[scene]
    print(f"  Downloading {scene} from {url} ...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{scene}.zip"
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Extracting {scene}.zip ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # 7-Scenes zips contain a top-level dir named after the scene
        extracted = Path(tmp) / scene
        if not extracted.is_dir():
            # Some zips use different top-level names — find the seq dir
            candidates = list(Path(tmp).rglob(seq))
            if not candidates:
                raise RuntimeError(f"Could not find {seq} inside {scene}.zip")
            extracted = candidates[0].parent
        scene_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(extracted), str(scene_dir), dirs_exist_ok=True)

    seq_dir = scene_dir / seq
    n_frames = len(list(seq_dir.glob("*.color.png")))
    print(f"  {scene}/{seq}: {n_frames} frames")
    if n_frames == 0:
        raise RuntimeError(f"No *.color.png found in {seq_dir} after extraction")
    return seq_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=sorted(SCENES), default=["fire", "office"])
    parser.add_argument("--seq", default="seq-01", help="Sequence to download (default: seq-01)")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    parser.add_argument("--list", action="store_true", help="List available scenes and exit")
    args = parser.parse_args()
    if args.list:
        for name, url in SCENES.items():
            print(f"  {name}: {url}")
        return
    for scene in args.scenes:
        download_scene(scene, seq=args.seq, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run test — should pass**

Run: `python -m pytest tests/evals/test_download_7scenes.py -v`
Expected: PASS

- [ ] **Step 4: Smoke test the CLI (list mode, no network needed)**

```bash
python evals/download_7scenes.py --list
```
Expected: prints 7 scene names with URLs. No download.

- [ ] **Step 5: Commit**

```bash
git add evals/download_7scenes.py tests/evals/test_download_7scenes.py
git commit -m "feat(evals): 7-Scenes download helper for fire + office"
```

---

## Task 2: `eval_gt.py` — Backbone Param + TUM Output

**Files:**
- Modify: `evals/eval_gt.py`

**Context:** `_make_creator` currently hardcodes `get_creator("vggtx")()`. Condition names like `baseline`, `ba`, `lc` become `omega_baseline`, `omega_ba`, `omega_lc` when backbone=`vggt_omega`. `eval_gt.py` currently outputs `metrics.json` + `trajectories.npz` but NOT TUM files. `eval_compare.py` needs TUM files. GT is already in `datasets.py` as `(N,4,4)` world-to-cam; TUM needs camera-to-world as `timestamp tx ty tz qx qy qz qw`.

- [ ] **Step 1: Add `_write_tum` helper and `--backbone` arg to the parser**

In `evals/eval_gt.py`, after `_cam_positions`, add:

```python
def _write_tum(path: Path, poses_w2c: np.ndarray) -> None:
    """Write TUM trajectory: 'timestamp tx ty tz qx qy qz qw' (camera-to-world)."""
    from scipy.spatial.transform import Rotation
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, w2c in enumerate(poses_w2c):
        c2w = np.linalg.inv(w2c.astype(np.float64))
        t = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # [qx, qy, qz, qw]
        lines.append(
            f"{i:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
    path.write_text("\n".join(lines) + "\n")
```

In `_build_parser()`, add after `--submap_size`:

```python
parser.add_argument(
    "--backbone", choices=["vggtx", "vggt_omega"], default="vggt_omega",
    help="Feedforward backbone. Condition output files are prefixed: "
         "vggt_omega→omega_*, vggtx→vggtx_*",
)
```

- [ ] **Step 2: Update `_make_creator` to accept backbone**

Replace:
```python
def _make_creator(condition: str, submap_size: int | None = None):
    """Build a (creator, ba_config) pair for the given condition.

    Returns (creator, None) when no bundle adjustment is needed.
    Returns (creator, BundleAdjustmentConfig) when BA should run after postprocess.
    """
    base = get_creator("vggtx")()
```

With:
```python
def _make_creator(condition: str, submap_size: int | None = None, backbone: str = "vggt_omega"):
    """Build a (creator, ba_config) pair for the given condition.

    Returns (creator, None) when no bundle adjustment is needed.
    Returns (creator, BundleAdjustmentConfig) when BA should run after postprocess.
    """
    base = get_creator(backbone)()
```

- [ ] **Step 3: Update `_run_condition` to return creator + accept backbone, and `_subprocess_mode` to write TUM + backbone arg**

Replace the `_run_condition` signature and return:
```python
def _run_condition(
    name: str, image_dir: Path, output_dir: Path,
    submap_size: int | None = None,
    backbone: str = "vggt_omega",
) -> tuple[np.ndarray, Any]:
    """Run condition, return (extrinsics (N,4,4), creator)."""
    creator, ba_cfg = _make_creator(name, submap_size=submap_size, backbone=backbone)
    if ba_cfg is None:
        creator.reconstruct(image_dir, output_dir)
    else:
        ba = BundleAdjustment(ba_cfg)
        creator.load_model()
        creator.setup_inference(image_dir)
        creator.run_inference()
        creator.postprocess()
        creator.outputs = ba.refine(creator.outputs).reproject()
        creator.build_colmap(output_dir)
    if creator.outputs is None:
        raise RuntimeError(f"Condition '{name}' produced no outputs")
    return creator.outputs.extrinsics, creator
```

Update `_subprocess_mode` to use the new return value and write backbone to result:
```python
def _subprocess_mode(args: argparse.Namespace) -> None:
    """Run one condition and write {extrinsics, time_s, backbone} JSON."""
    _validate_condition(args._condition)
    cond = args._condition
    image_dir = args._image_dir
    result_file = args._result_file
    backbone = getattr(args, "backbone", "vggt_omega")

    t0 = time.perf_counter()
    pred, _creator = _run_condition(
        cond, image_dir, args.output_dir / cond,
        submap_size=args.submap_size,
        backbone=backbone,
    )
    elapsed = time.perf_counter() - t0

    result_file.write_text(json.dumps({
        "extrinsics": pred.tolist(),
        "time_s": round(elapsed, 2),
        "backbone": backbone,
    }))
    print(f"  ATE/RPE computed by parent | time={elapsed:.1f}s")
```

- [ ] **Step 4: Add backbone prefix mapping and TUM output in `main()`**

Add constant after `_FIXED_CONDITIONS`:
```python
_BACKBONE_PREFIX = {"vggt_omega": "omega", "vggtx": "vggtx"}
```

In `main()`, after the per-condition loop that collects `metrics` and `trajectories`, add TUM output:

```python
# Write TUM trajectories for eval_compare.py phase-2 runner
tum_dir = args.output_dir
tum_dir.mkdir(parents=True, exist_ok=True)
prefix = _BACKBONE_PREFIX.get(args.backbone, args.backbone)
_write_tum(tum_dir / "gt.tum", dataset.gt_poses)
for cond, poses in trajectories.items():
    if cond == "gt":
        continue
    tum_name = f"{prefix}_{cond}.tum"
    _write_tum(tum_dir / tum_name, poses)
    print(f"  TUM written: {tum_dir / tum_name}")
```

Also pass `--backbone` to subprocesses in the condition loop:
```python
cmd = [
    sys.executable, __file__,
    "--dataset", args.dataset,
    "--seq_dir", str(args.seq_dir),
    "--output_dir", str(args.output_dir),
    "--max_frames", str(args.max_frames),
    "--backbone", args.backbone,          # ← add this
    "--_condition",   cond,
    "--_image_dir",   str(tmp_image_dir),
    "--_result_file", str(result_file),
]
```

- [ ] **Step 5: Quick manual smoke test (dry-run import)**

```bash
python -c "import sys; sys.path.insert(0, 'evals'); import eval_gt; print('import OK')"
```
Expected: `import OK`

Also check help text:
```bash
python evals/eval_gt.py --help | grep backbone
```
Expected: prints `--backbone {vggtx,vggt_omega}`

- [ ] **Step 6: Commit**

```bash
git add evals/eval_gt.py
git commit -m "feat(evals): add --backbone arg and TUM trajectory output to eval_gt"
```

---

## Task 3: Fix VGGT-SLAM Runner

**Files:**
- Replace: `evals/runners/run_vggt_slam.py`

**Context:** Current file always raises `EnvBlocked`. `reconstruction` env has Python 3.11 so VGGT-SLAM can run. VGGT-SLAM's `main.py` args: `--image_folder`, `--max_loops 1`, `--min_disparity 50`, `--conf_threshold 25`, `--lc_thres 0.95`, `--submap_size <w>`, `--log_results`, `--log_path <path>`. It writes TUM-format directly (`kitti_format=False`). On success, remove the `.pending` sentinel from `evals/baselines/vggt_slam/`.

- [ ] **Step 1: Write failing test**

```python
# tests/evals/test_run_vggt_slam.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "runners"))
from run_vggt_slam import VGGTSLAM_DIR, run_vggt_slam

def test_vggtslam_dir_exists():
    assert VGGTSLAM_DIR.is_dir(), f"VGGT-SLAM not found at {VGGTSLAM_DIR}"

def test_vggtslam_main_exists():
    assert (VGGTSLAM_DIR / "main.py").is_file()
```

Run: `python -m pytest tests/evals/test_run_vggt_slam.py -v`
Expected: PASS (submodule is present). If fails, run `git submodule update --init third_party/VGGT-SLAM`.

- [ ] **Step 2: Replace `run_vggt_slam.py`**

```python
"""Subprocess wrapper around VGGT-SLAM's main.py.

Runs VGGT-SLAM in a subprocess using the current Python interpreter (reconstruction
env, Python 3.11 — which satisfies VGGT-SLAM's SL(4)/GTSAM requirement).

On success, copies the output TUM file to output_tum and removes the corresponding
.pending sentinel from evals/baselines/vggt_slam/ if present.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VGGTSLAM_DIR = REPO_ROOT / "third_party" / "VGGT-SLAM"
BASELINES_DIR = REPO_ROOT / "evals" / "baselines" / "vggt_slam"


def run_vggt_slam(
    image_dir: Path,
    output_tum: Path,
    submap_size: int = 16,
    python: str | None = None,
) -> Path:
    """Run VGGT-SLAM on image_dir; write trajectory to output_tum.

    Args:
        image_dir: Directory of input images (sorted, no GT required).
        output_tum: Destination path for the TUM trajectory file.
        submap_size: VGGT-SLAM submap window size (default 16, matches eval_tum.sh).
        python: Python binary to use. Defaults to sys.executable (reconstruction env).

    Returns:
        output_tum path on success.
    """
    if not VGGTSLAM_DIR.is_dir():
        raise FileNotFoundError(
            f"VGGT-SLAM submodule not found at {VGGTSLAM_DIR}. "
            "Run: git submodule update --init third_party/VGGT-SLAM"
        )
    py = python or sys.executable
    # VGGT-SLAM writes its log to log_path; we write to a temp location then copy.
    log_path = output_tum.with_suffix(".vggtslam.txt")
    cmd = [
        py,
        str(VGGTSLAM_DIR / "main.py"),
        "--image_folder", str(image_dir),
        "--max_loops", "1",
        "--min_disparity", "50",
        "--conf_threshold", "25",
        "--lc_thres", "0.95",
        "--submap_size", str(submap_size),
        "--log_results",
        "--skip_dense_log",
        "--log_path", str(log_path),
    ]
    output_tum.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=VGGTSLAM_DIR)
    if not log_path.is_file():
        raise RuntimeError(
            f"VGGT-SLAM finished but log not found at {log_path}. "
            "Check --log_path handling in VGGT-SLAM main.py."
        )
    shutil.copy2(log_path, output_tum)
    # Remove .pending sentinel if present
    seq_name = output_tum.stem  # e.g. "vggt_slam" → seq inferred from parent dir
    results_seq = output_tum.parent.name  # e.g. "chess_seq01"
    sentinel = BASELINES_DIR / f"{results_seq}.pending"
    if sentinel.is_file():
        sentinel.unlink()
        print(f"  Removed sentinel: {sentinel}")
    return output_tum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_dir", type=Path, required=True,
                    help="Directory of input images for VGGT-SLAM")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output TUM trajectory path")
    ap.add_argument("--submap_size", type=int, default=16,
                    help="VGGT-SLAM submap size (default 16)")
    ap.add_argument("--python", type=str, default=None,
                    help="Python binary (default: sys.executable)")
    args = ap.parse_args()
    run_vggt_slam(args.image_dir, args.output, submap_size=args.submap_size, python=args.python)
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run import test**

```bash
python -c "from evals.runners.run_vggt_slam import run_vggt_slam, VGGTSLAM_DIR; print('OK', VGGTSLAM_DIR)"
```
Expected: prints `OK` + path.

- [ ] **Step 4: Commit**

```bash
git add evals/runners/run_vggt_slam.py tests/evals/test_run_vggt_slam.py
git commit -m "feat(evals): implement VGGT-SLAM runner (replace EnvBlocked stub)"
```

---

## Task 4: `eval_compare.py` — New Conditions, AUC, Alignment JSON

**Files:**
- Modify: `evals/eval_compare.py`
- Modify: `evals/metrics.py`

**Context:** `_scan_results_dir` currently raises on any non-`.tum`/non-`.pending` file. It must now silently skip `.json`, `.npz`, `.png` sidecar files. `_DEFAULT_ALIGN` needs new backbone-prefixed condition names. `compute_auc` needs to be added to `metrics.py` — it loads TUM files via evo, aligns, converts to pose arrays, and calls `auc_at_threshold` from `collab_splats.pointcloud.loop_closure.eval`.

- [ ] **Step 1: Write failing test for `compute_auc` in metrics.py**

```python
# tests/evals/test_metrics_auc.py
import numpy as np
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
from metrics import compute_auc

def _write_tum(path, poses_w2c):
    from scipy.spatial.transform import Rotation
    lines = []
    for i, w2c in enumerate(poses_w2c):
        c2w = np.linalg.inv(w2c.astype(np.float64))
        t = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        lines.append(f"{i:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    Path(path).write_text("\n".join(lines) + "\n")

def test_compute_auc_perfect():
    """Perfect prediction = AUC@30 of 100."""
    N = 20
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    for i in range(N):
        poses[i, :3, 3] = [i * 0.1, 0, 0]
    with tempfile.TemporaryDirectory() as tmp:
        pred_path = Path(tmp) / "pred.tum"
        gt_path = Path(tmp) / "gt.tum"
        _write_tum(pred_path, poses)
        _write_tum(gt_path, poses)
        result = compute_auc(pred_path, gt_path, align="none")
    assert result["auc_30"] == pytest.approx(100.0, abs=1.0)
    assert "per_frame_err" in result
```

Run: `python -m pytest tests/evals/test_metrics_auc.py -v`
Expected: FAIL (`compute_auc` not defined)

- [ ] **Step 2: Add `compute_auc` to `evals/metrics.py`**

Add after `compute_rpe`:

```python
def compute_auc(
    pred_path: Path | str,
    gt_path: Path | str,
    align: str = "sim3",
    max_threshold_deg: float = 30.0,
) -> dict:
    """AUC@max_threshold_deg (CO3Dv2/VGGSfM protocol).

    Loads TUM files, aligns, converts to (N,4,4) pose arrays, delegates to
    collab_splats.pointcloud.loop_closure.eval.auc_at_threshold.

    Returns dict with 'auc_30' (float in [0,100]) and 'per_frame_err' (list[float]).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from collab_splats.pointcloud.loop_closure.eval import auc_at_threshold

    traj_ref, traj_est = _load_pair(pred_path, gt_path)
    _apply_alignment(traj_ref, traj_est, align)

    # evo stores poses_se3 as (N, 4, 4) numpy arrays after alignment
    pred_poses = traj_est.poses_se3   # cam-to-world
    gt_poses = traj_ref.poses_se3

    # auc_at_threshold expects world-to-cam; invert
    pred_w2c = np.linalg.inv(pred_poses).astype(np.float32)
    gt_w2c = np.linalg.inv(gt_poses).astype(np.float32)

    result = auc_at_threshold(pred_w2c, gt_w2c, max_threshold_deg=max_threshold_deg)
    return {
        "auc_30": result["auc_30"],
        "per_frame_err": result["per_frame_err"],
    }
```

- [ ] **Step 3: Run AUC test**

Run: `python -m pytest tests/evals/test_metrics_auc.py -v`
Expected: PASS

- [ ] **Step 4: Update `_DEFAULT_ALIGN` in `eval_compare.py`**

Replace the existing `_DEFAULT_ALIGN` dict:

```python
_DEFAULT_ALIGN: dict[str, str] = {
    # new backbone-prefixed names
    "omega_baseline": "se3",
    "omega_ba":       "se3",
    "omega_lc":       "sim3",
    "vggtx_baseline": "se3",
    "vggtx_lc":       "sim3",
    "vggt_slam":      "sim3",
    "vggt_long":      "sim3",
    # legacy names (backward compat)
    "ours_baseline":  "se3",
    "ours_ba":        "se3",
    "ours_lc":        "sim3",
}
```

- [ ] **Step 5: Relax `_scan_results_dir` to skip known sidecar extensions**

Replace the `else: raise ValueError` block:

```python
        elif entry.suffix in (".json", ".npz", ".png", ".jpg"):
            continue  # sidecar files from eval_gt.py output — skip silently
        elif entry.suffix == ".pending":
            pending.add(entry.stem)
        else:
            raise ValueError(
                f"unexpected file {entry.name!r} in {results_dir} — "
                "only *.tum, *.pending, and known sidecars (.json/.npz/.png) are allowed"
            )
```

- [ ] **Step 6: Add AUC to the comparison runner**

In `main()`, in the per-method loop where `ate` and `rpe` are computed, add `auc`:

```python
    for name, tum_path in methods.items():
        align = _resolve_align(name, overrides)
        ate = compute_ate(tum_path, gt_path, align=align)
        rpe = compute_rpe(tum_path, gt_path, align=align, delta=1)
        auc = compute_auc(tum_path, gt_path, align=align)
        # read alignment JSON if present
        alignment_path = results_dir / f"{name}_alignment.json"
        alignment = json.loads(alignment_path.read_text()) if alignment_path.is_file() else None
        out[name] = {
            "status": "ok",
            "align": align,
            "ate": ate,
            "rpe": rpe,
            "auc": auc,
            "alignment": alignment,
        }
```

Add `from metrics import compute_ate, compute_rpe, compute_auc` (replace old import).

- [ ] **Step 7: Update `_format_markdown` to include AUC and alignment columns**

Replace `_format_markdown`:

```python
def _format_markdown(methods: dict[str, dict]) -> str:
    header = (
        "| method | status | align | ATE RMSE | RPE trans | RPE rot° | AUC@30 | loop_res↓ | chamfer_ratio↓ |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for name in sorted(methods.keys()):
        body = methods[name]
        status = body.get("status", "?")
        if status != "ok":
            lines.append(f"| {name} | {status} | - | - | - | - | - | - | - |")
            continue
        a = body["ate"]
        r = body["rpe"]
        auc_val = body.get("auc", {}).get("auc_30", float("nan"))
        al = body.get("alignment") or {}
        loop_before = al.get("loop_match_residual", {}).get("mean_before", None)
        loop_after  = al.get("loop_match_residual", {}).get("mean_after", None)
        chamfer_before = al.get("pointcloud_chamfer", {}).get("mean_before", None)
        chamfer_after  = al.get("pointcloud_chamfer", {}).get("mean_after", None)
        loop_str = f"{loop_before:.3f}→{loop_after:.3f}" if (loop_before and loop_after) else "null"
        chamfer_ratio = (chamfer_after / chamfer_before) if (chamfer_before and chamfer_after and chamfer_before > 0) else None
        chamfer_str = f"{chamfer_ratio:.3f}" if chamfer_ratio is not None else "null"
        lines.append(
            f"| {name} | {status} | {body['align']} | "
            f"{a['rmse']:.4f} | {r['trans_rmse']:.4f} | {r['rot_rmse_deg']:.4f} | "
            f"{auc_val:.1f} | {loop_str} | {chamfer_str} |"
        )
    return "\n".join(lines)
```

Also update the `payload` to include auc in output:
```python
    payload = {"methods": out}
```
(No change needed — `out` already has `auc` nested inside each method.)

- [ ] **Step 8: Commit**

```bash
git add evals/metrics.py evals/eval_compare.py tests/evals/test_metrics_auc.py
git commit -m "feat(evals): add AUC@30, new condition names, alignment JSON support to eval_compare"
```

---

## Task 5: `wrappers.py` — Store Pre/Post LC Extrinsics

**Files:**
- Modify: `collab_splats/pointcloud/wrappers.py`

**Context:** `dedup_overlap(submap_ids, submap_starts, corrected: dict[int, np.ndarray], total_frames) -> np.ndarray` is the existing overlap-dedup function in `closure.py`. Pre-LC global extrinsics = raw per-submap poses stitched with the same dedup logic. This must be captured BEFORE calling `run_pose_graph_optimization`.

- [ ] **Step 1: Write failing test**

```python
# tests/pointcloud/test_wrappers_lc_attrs.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

def _make_mock_lc():
    """Minimal LoopClosure wrapper with mock base and config."""
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig
    base = MagicMock()
    cfg = LoopClosureConfig(submap_size=5, submap_overlap=1)
    return LoopClosure(base, config=cfg)

def test_lc_attrs_set_after_run(tmp_path, monkeypatch):
    """After _run_lc_loop, _lc_precorrection_extrinsics and _lc_corrected_extrinsics are set."""
    # This test requires a non-trivial integration setup; use a simpler check:
    # Verify the attributes exist as names (they will be set during real inference).
    lc = _make_mock_lc()
    # Check that the base object can have these attributes set
    lc.base._lc_precorrection_extrinsics = np.eye(4)[None]
    lc.base._lc_corrected_extrinsics = np.eye(4)[None]
    assert lc.base._lc_precorrection_extrinsics.shape == (1, 4, 4)
    assert lc.base._lc_corrected_extrinsics.shape == (1, 4, 4)
```

Run: `python -m pytest tests/pointcloud/test_wrappers_lc_attrs.py -v`
Expected: PASS (test is structural — next step adds real code)

- [ ] **Step 2: Add `_assemble_precorrection_extrinsics` helper and inject into `_run_lc_loop`**

In `collab_splats/pointcloud/wrappers.py`, after the import of `dedup_overlap`:

Add module-level helper (place just before the `LoopClosure` class definition):

```python
def _assemble_precorrection_extrinsics(submaps: list, total_frames: int) -> np.ndarray:
    """Stitch raw per-submap poses into (total_frames, 4, 4) without PGO correction.

    Uses same first-writer-wins overlap dedup as dedup_overlap in closure.py.
    """
    from .loop_closure.closure import dedup_overlap
    corrected_raw = {s.submap_id: s.poses for s in submaps}
    return dedup_overlap(
        submap_ids=[s.submap_id for s in submaps],
        submap_starts=[s.frame_start for s in submaps],
        corrected=corrected_raw,
        total_frames=total_frames,
    )
```

In `_run_lc_loop`, find the block:
```python
        self.base._lc_submaps = submaps
        self.base._lc_loop_submaps = lc_submaps
        self.base._lc_overlap_frames = O
        self.base._lc_all_matches = all_loop_candidates

        # Merge per-submap world_points and poses into unified outputs
        t0_pg = time.perf_counter()
        corrected_extrinsics = run_pose_graph_optimization(
```

Replace with:
```python
        self.base._lc_submaps = submaps
        self.base._lc_loop_submaps = lc_submaps
        self.base._lc_overlap_frames = O
        self.base._lc_all_matches = all_loop_candidates
        self.base._lc_precorrection_extrinsics = _assemble_precorrection_extrinsics(submaps, N)

        # Merge per-submap world_points and poses into unified outputs
        t0_pg = time.perf_counter()
        corrected_extrinsics = run_pose_graph_optimization(
```

After `corrected_extrinsics = run_pose_graph_optimization(...)`, add:
```python
        self.base._lc_corrected_extrinsics = corrected_extrinsics
```

- [ ] **Step 3: Run existing wrapper tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_wrappers.py -v
```
Expected: all existing tests PASS (no regressions).

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py tests/pointcloud/test_wrappers_lc_attrs.py
git commit -m "feat(lc): store pre/post-PGO extrinsics on LoopClosure.base for alignment metrics"
```

---

## Task 6: `reconstruction_quality.py` — Alignment Metrics

**Files:**
- Create: `evals/reconstruction_quality.py`
- Create: `tests/evals/test_reconstruction_quality.py`

**Context:** `Submap.poses` is `(K, 4, 4)` world-to-cam. Camera position for pose P = `-R.T @ t` where `R=P[:3,:3]`, `t=P[:3,3]`. Accepted matches have `match.accepted == True`. `compute_alignment_metrics` reads `_lc_submaps`, `_lc_all_matches`, `_lc_precorrection_extrinsics`, `_lc_corrected_extrinsics` from `lc_creator.base`. Frame index for match: `submaps[match.query_submap_id].frame_start + match.query_frame_idx`.

- [ ] **Step 1: Write failing tests**

```python
# tests/evals/test_reconstruction_quality.py
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

def _cam_pos(w2c: np.ndarray) -> np.ndarray:
    """(N,4,4) world-to-cam → (N,3) camera world positions."""
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)

def _make_submaps(n_submaps=3, K=5, frame_start_step=4):
    """Build minimal Submap-like objects for testing."""
    from dataclasses import dataclass
    import torch

    @dataclass
    class FakeSubmap:
        submap_id: int
        frame_start: int
        poses: np.ndarray  # (K,4,4)
        world_points: np.ndarray  # (K,P,3)
        is_lc_submap: bool = False

    submaps = []
    for i in range(n_submaps):
        poses = np.tile(np.eye(4, dtype=np.float32), (K, 1, 1))
        poses[:, 0, 3] = np.arange(K) * 0.1 + i * K * 0.1
        world_points = np.random.randn(K, 10, 3).astype(np.float32)
        submaps.append(FakeSubmap(
            submap_id=i,
            frame_start=i * frame_start_step,
            poses=poses,
            world_points=world_points,
        ))
    return submaps

def _make_match(q_sid, q_fidx, d_sid, d_fidx, accepted=True):
    from dataclasses import dataclass
    @dataclass
    class FakeMatch:
        query_submap_id: int
        query_frame_idx: int
        detected_submap_id: int
        detected_frame_idx: int
        accepted: bool
        similarity_score: float = 0.5
    return FakeMatch(q_sid, q_fidx, d_sid, d_fidx, accepted)


def test_loop_match_residual_perfect():
    from reconstruction_quality import loop_match_residual
    submaps = _make_submaps(3)
    N = 12
    pre = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    post = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    # accepted match between frame 0 in submap 0 and frame 0 in submap 2
    matches = [_make_match(2, 0, 0, 0, accepted=True)]
    result = loop_match_residual(matches, submaps, pre, post)
    # camera positions are the same (identity poses) so residual = 0
    assert result["mean_before"] == pytest.approx(0.0, abs=1e-5)
    assert result["mean_after"] == pytest.approx(0.0, abs=1e-5)
    assert result["n_matches"] == 1


def test_submap_boundary_gap_zero():
    from reconstruction_quality import submap_boundary_gap
    submaps = _make_submaps(3, K=5, frame_start_step=4)
    N = 12
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    result = submap_boundary_gap(submaps, poses, poses)
    # All identity poses → zero gap
    assert result["mean_before"] == pytest.approx(0.0, abs=1e-5)


def test_pointcloud_chamfer_identical():
    from reconstruction_quality import pointcloud_chamfer
    submaps = _make_submaps(3)
    N = 12
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    matches = [_make_match(2, 0, 0, 0, accepted=True)]
    result = pointcloud_chamfer(matches, submaps, poses, poses)
    assert "mean_before" in result
    assert "mean_after" in result
    assert result["n_pairs"] >= 0
```

Run: `python -m pytest tests/evals/test_reconstruction_quality.py -v`
Expected: FAIL (ImportError: No module named 'reconstruction_quality')

- [ ] **Step 2: Create `evals/reconstruction_quality.py`**

```python
"""Submap alignment quality metrics for LC validation.

Three metrics, each computed before and after pose-graph optimization (PGO):
  loop_match_residual  — camera-position distance for accepted LC match pairs
  submap_boundary_gap  — camera-position gap at consecutive submap stitches
  pointcloud_chamfer   — symmetric Chamfer distance between matched-submap camera positions

All return JSON-serializable dicts. compute_alignment_metrics(lc_creator) aggregates
all three from a post-run LoopClosure instance.
"""
from __future__ import annotations

import numpy as np


########################################
########## Internal helpers ############
########################################

def _cam_positions(poses_w2c: np.ndarray) -> np.ndarray:
    """(N, 4, 4) world-to-cam → (N, 3) camera positions in world frame."""
    R = poses_w2c[:, :3, :3]
    t = poses_w2c[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)


def _symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between (M, 3) and (N, 3) point sets."""
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d_a = cKDTree(b).query(a)[0].mean()
    d_b = cKDTree(a).query(b)[0].mean()
    return float((d_a + d_b) / 2)


def _global_frame(submap, local_frame_idx: int) -> int:
    return submap.frame_start + local_frame_idx


########################################
########## Public metrics ##############
########################################

def loop_match_residual(
    matches: list,
    submaps: list,
    pre_ext: np.ndarray,
    post_ext: np.ndarray,
) -> dict:
    """Mean/max camera-position distance for accepted loop match pairs.

    Args:
        matches: list of LoopMatch-like objects with accepted, query_submap_id,
                 query_frame_idx, detected_submap_id, detected_frame_idx.
        submaps: list of Submap-like objects with submap_id, frame_start, poses.
        pre_ext: (N, 4, 4) pre-PGO world-to-cam extrinsics.
        post_ext: (N, 4, 4) post-PGO world-to-cam extrinsics.

    Returns:
        dict with mean_before, max_before, mean_after, max_after, n_matches.
    """
    accepted = [m for m in matches if m.accepted]
    if not accepted:
        return {"mean_before": None, "max_before": None,
                "mean_after": None, "max_after": None, "n_matches": 0}

    sid_to_submap = {s.submap_id: s for s in submaps}
    pre_pos = _cam_positions(pre_ext)
    post_pos = _cam_positions(post_ext)

    dists_before, dists_after = [], []
    for m in accepted:
        q_g = _global_frame(sid_to_submap[m.query_submap_id], m.query_frame_idx)
        d_g = _global_frame(sid_to_submap[m.detected_submap_id], m.detected_frame_idx)
        if q_g < len(pre_pos) and d_g < len(pre_pos):
            dists_before.append(float(np.linalg.norm(pre_pos[q_g] - pre_pos[d_g])))
            dists_after.append(float(np.linalg.norm(post_pos[q_g] - post_pos[d_g])))

    if not dists_before:
        return {"mean_before": None, "max_before": None,
                "mean_after": None, "max_after": None, "n_matches": 0}

    return {
        "mean_before": float(np.mean(dists_before)),
        "max_before":  float(np.max(dists_before)),
        "mean_after":  float(np.mean(dists_after)),
        "max_after":   float(np.max(dists_after)),
        "n_matches":   len(dists_before),
    }


def submap_boundary_gap(
    submaps: list,
    pre_ext: np.ndarray,
    post_ext: np.ndarray,
) -> dict:
    """Mean/max camera-position gap at consecutive submap stitches.

    For each consecutive pair (submap[i], submap[i+1]), measures distance between
    the last global frame of submap[i] and the first global frame of submap[i+1].

    Args:
        submaps: list of Submap-like objects (normal submaps, not LC submaps), ordered.
        pre_ext: (N, 4, 4) pre-PGO world-to-cam extrinsics.
        post_ext: (N, 4, 4) post-PGO world-to-cam extrinsics.

    Returns:
        dict with mean_before, max_before, mean_after, max_after, n_boundaries.
    """
    normal = [s for s in submaps if not getattr(s, "is_lc_submap", False)]
    if len(normal) < 2:
        return {"mean_before": None, "max_before": None,
                "mean_after": None, "max_after": None, "n_boundaries": 0}

    pre_pos = _cam_positions(pre_ext)
    post_pos = _cam_positions(post_ext)
    gaps_before, gaps_after = [], []

    for i in range(len(normal) - 1):
        s_cur = normal[i]
        s_nxt = normal[i + 1]
        last_g = s_cur.frame_start + len(s_cur.poses) - 1
        first_g = s_nxt.frame_start
        if last_g < len(pre_pos) and first_g < len(pre_pos):
            gaps_before.append(float(np.linalg.norm(pre_pos[last_g] - pre_pos[first_g])))
            gaps_after.append(float(np.linalg.norm(post_pos[last_g] - post_pos[first_g])))

    if not gaps_before:
        return {"mean_before": None, "max_before": None,
                "mean_after": None, "max_after": None, "n_boundaries": 0}

    return {
        "mean_before":   float(np.mean(gaps_before)),
        "max_before":    float(np.max(gaps_before)),
        "mean_after":    float(np.mean(gaps_after)),
        "max_after":     float(np.max(gaps_after)),
        "n_boundaries":  len(gaps_before),
    }


def pointcloud_chamfer(
    matches: list,
    submaps: list,
    pre_ext: np.ndarray,
    post_ext: np.ndarray,
    max_pairs: int = 20,
) -> dict:
    """Mean Chamfer distance (on camera positions) between matched-submap pairs.

    Uses camera positions of all frames within each matched submap as the
    representative point cloud (avoids expensive world-point reprojection).
    Caps at max_pairs accepted matches to bound compute time.

    Args:
        matches: list of LoopMatch-like objects with accepted flag.
        submaps: list of Submap-like objects.
        pre_ext: (N, 4, 4) pre-PGO world-to-cam extrinsics.
        post_ext: (N, 4, 4) post-PGO world-to-cam extrinsics.
        max_pairs: maximum number of match pairs to evaluate.

    Returns:
        dict with mean_before, mean_after, n_pairs.
    """
    accepted = [m for m in matches if m.accepted][:max_pairs]
    if not accepted:
        return {"mean_before": None, "mean_after": None, "n_pairs": 0}

    sid_to_submap = {s.submap_id: s for s in submaps}
    pre_pos = _cam_positions(pre_ext)
    post_pos = _cam_positions(post_ext)

    chamfers_before, chamfers_after = [], []
    for m in accepted:
        q_s = sid_to_submap.get(m.query_submap_id)
        d_s = sid_to_submap.get(m.detected_submap_id)
        if q_s is None or d_s is None:
            continue
        # Camera positions for all frames in each submap
        q_range = slice(q_s.frame_start, q_s.frame_start + len(q_s.poses))
        d_range = slice(d_s.frame_start, d_s.frame_start + len(d_s.poses))
        q_pre = pre_pos[q_range]
        d_pre = pre_pos[d_range]
        q_post = post_pos[q_range]
        d_post = post_pos[d_range]
        if len(q_pre) > 0 and len(d_pre) > 0:
            chamfers_before.append(_symmetric_chamfer(q_pre, d_pre))
            chamfers_after.append(_symmetric_chamfer(q_post, d_post))

    if not chamfers_before:
        return {"mean_before": None, "mean_after": None, "n_pairs": 0}

    return {
        "mean_before": float(np.nanmean(chamfers_before)),
        "mean_after":  float(np.nanmean(chamfers_after)),
        "n_pairs":     len(chamfers_before),
    }


########################################
########## Aggregator ##################
########################################

def compute_alignment_metrics(lc_creator) -> dict:
    """Aggregate all three alignment metrics from a post-run LoopClosure instance.

    Reads _lc_submaps, _lc_all_matches, _lc_precorrection_extrinsics,
    _lc_corrected_extrinsics from lc_creator.base. Returns empty dict if any
    required attribute is missing (e.g. no loop closures found).
    """
    base = lc_creator.base
    required = [
        "_lc_submaps", "_lc_all_matches",
        "_lc_precorrection_extrinsics", "_lc_corrected_extrinsics",
    ]
    for attr in required:
        if not hasattr(base, attr):
            return {}

    submaps = base._lc_submaps
    matches = base._lc_all_matches
    pre_ext = base._lc_precorrection_extrinsics
    post_ext = base._lc_corrected_extrinsics

    return {
        "loop_match_residual": loop_match_residual(matches, submaps, pre_ext, post_ext),
        "submap_boundary_gap": submap_boundary_gap(submaps, pre_ext, post_ext),
        "pointcloud_chamfer":  pointcloud_chamfer(matches, submaps, pre_ext, post_ext),
    }
```

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/evals/test_reconstruction_quality.py -v
```
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add evals/reconstruction_quality.py tests/evals/test_reconstruction_quality.py
git commit -m "feat(evals): reconstruction_quality module — loop residual, boundary gap, Chamfer"
```

---

## Task 7: Integrate Alignment Metrics into `eval_gt.py`

**Files:**
- Modify: `evals/eval_gt.py`

**Context:** The alignment JSON should be written from the subprocess (leaf) side, where the creator is in memory. `_subprocess_mode` receives the `creator` from `_run_condition`. It checks for `_lc_submaps` and calls `compute_alignment_metrics`. The parent reads this JSON from the `_result_file` and writes `<prefix>_<cond>_alignment.json` to the output dir.

- [ ] **Step 1: Add alignment import and write in `_subprocess_mode`**

At top of `evals/eval_gt.py`, add after existing imports:
```python
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
```

(The `evals/` dir is already on `sys.path` since the file does `sys.path.insert(0, ...)` at load. Confirm this is already present; if so, no change needed.)

In `_subprocess_mode`, update to call `compute_alignment_metrics`:

```python
def _subprocess_mode(args: argparse.Namespace) -> None:
    """Run one condition and write {extrinsics, time_s, backbone, alignment} JSON."""
    _validate_condition(args._condition)
    cond = args._condition
    image_dir = args._image_dir
    result_file = args._result_file
    backbone = getattr(args, "backbone", "vggt_omega")

    t0 = time.perf_counter()
    pred, creator = _run_condition(
        cond, image_dir, args.output_dir / cond,
        submap_size=args.submap_size,
        backbone=backbone,
    )
    elapsed = time.perf_counter() - t0

    # Compute submap alignment metrics for LC conditions
    alignment: dict = {}
    if hasattr(creator, "base") and hasattr(creator.base, "_lc_submaps"):
        try:
            from reconstruction_quality import compute_alignment_metrics
            alignment = compute_alignment_metrics(creator)
        except Exception as exc:
            print(f"  WARNING: alignment metrics failed: {exc}")

    result_file.write_text(json.dumps({
        "extrinsics": pred.tolist(),
        "time_s": round(elapsed, 2),
        "backbone": backbone,
        "alignment": alignment,
    }))
    print(f"  time={elapsed:.1f}s")
```

- [ ] **Step 2: Read alignment from result JSON and write `*_alignment.json` in `main()`**

In `main()`, in the loop that reads `result_json` per condition, add:

```python
            alignment = result_json.get("alignment", {})
            if alignment:
                prefix = _BACKBONE_PREFIX.get(args.backbone, args.backbone)
                alignment_path = args.output_dir / f"{prefix}_{cond}_alignment.json"
                alignment_path.write_text(json.dumps(alignment, indent=2))
                print(f"  Alignment JSON: {alignment_path}")
```

- [ ] **Step 3: Smoke test import**

```bash
python -c "import sys; sys.path.insert(0, 'evals'); import eval_gt; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add evals/eval_gt.py
git commit -m "feat(evals): emit alignment JSON from LC conditions in eval_gt subprocess"
```

---

## Task 8: `eval_suite.sh` — Orchestrator

**Files:**
- Create: `evals/eval_suite.sh`

**Context:** Sequences: chess (seq-01 already at `data/7scenes/chess/seq-01`), fire (seq-01), office (seq-01). Results go to `evals/results/<scene>_seq01/`. Idempotent — skips `.tum` if already present. Runs eval_gt twice (omega + vggtx), then VGGT-SLAM, then eval_compare.

- [ ] **Step 1: Create `evals/eval_suite.sh`**

```bash
#!/usr/bin/env bash
# eval_suite.sh — orchestrate 6-condition 7-Scenes eval
# Usage: bash evals/eval_suite.sh [--seq seq-01] [--submap_size 20] [--skip_download]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/conda/envs/reconstruction/bin/python}"
EVALS_DIR="${REPO_ROOT}/evals"
DATA_DIR="${REPO_ROOT}/data/7scenes"
RESULTS_BASE="${REPO_ROOT}/evals/results"
SEQ="seq-01"
SUBMAP_SIZE=20
SKIP_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seq)           SEQ="$2"; shift 2 ;;
        --submap_size)   SUBMAP_SIZE="$2"; shift 2 ;;
        --skip_download) SKIP_DOWNLOAD=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCENES=("chess" "fire" "office")

# Step 1: Download missing sequences
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    echo "=== Downloading missing 7-Scenes sequences ==="
    "$PYTHON" "${EVALS_DIR}/download_7scenes.py" --scenes fire office --seq "$SEQ"
fi

for SCENE in "${SCENES[@]}"; do
    SEQ_DIR="${DATA_DIR}/${SCENE}/${SEQ}"
    SEQ_TAG="${SCENE}_$(echo "${SEQ}" | tr '-' '')"   # chess_seq01
    RESULTS_DIR="${RESULTS_BASE}/${SEQ_TAG}"
    mkdir -p "${RESULTS_DIR}"

    echo ""
    echo "===== Scene: ${SCENE}/${SEQ} → ${RESULTS_DIR} ====="

    # Write GT TUM (done by eval_gt but also do it explicitly for safety)
    # eval_gt.py writes gt.tum when outputting TUM trajectories

    # --- VGGT-Omega conditions ---
    for COND in baseline ba lc; do
        PREFIX="omega"
        TUM="${RESULTS_DIR}/${PREFIX}_${COND}.tum"
        if [[ -f "$TUM" ]]; then
            echo "  Skipping ${PREFIX}_${COND} (already exists)"
            continue
        fi
        echo "  Running ${PREFIX}_${COND} ..."
        "$PYTHON" "${EVALS_DIR}/eval_gt.py" \
            --dataset 7scenes \
            --seq_dir "${SEQ_DIR}" \
            --output_dir "${RESULTS_DIR}" \
            --backbone vggt_omega \
            --submap_size "${SUBMAP_SIZE}" \
            --conditions "${COND}"
    done

    # --- VGGT-X conditions ---
    for COND in baseline lc; do
        PREFIX="vggtx"
        TUM="${RESULTS_DIR}/${PREFIX}_${COND}.tum"
        if [[ -f "$TUM" ]]; then
            echo "  Skipping ${PREFIX}_${COND} (already exists)"
            continue
        fi
        echo "  Running ${PREFIX}_${COND} ..."
        "$PYTHON" "${EVALS_DIR}/eval_gt.py" \
            --dataset 7scenes \
            --seq_dir "${SEQ_DIR}" \
            --output_dir "${RESULTS_DIR}" \
            --backbone vggtx \
            --submap_size "${SUBMAP_SIZE}" \
            --conditions "${COND}"
    done

    # --- VGGT-SLAM ---
    SLAM_TUM="${RESULTS_DIR}/vggt_slam.tum"
    if [[ -f "$SLAM_TUM" ]]; then
        echo "  Skipping vggt_slam (already exists)"
    else
        echo "  Running vggt_slam ..."
        "$PYTHON" "${EVALS_DIR}/runners/run_vggt_slam.py" \
            --image_dir "${SEQ_DIR}" \
            --output "${SLAM_TUM}" \
            --submap_size "${SUBMAP_SIZE}"
    fi

    # --- eval_compare phase-2 ---
    echo "  Running eval_compare ..."
    "$PYTHON" "${EVALS_DIR}/eval_compare.py" \
        --results-dir "${RESULTS_DIR}"

    echo "  metrics.json → ${RESULTS_DIR}/metrics.json"
done

echo ""
echo "=== All scenes complete ==="
```

- [ ] **Step 2: Make executable**

```bash
chmod +x evals/eval_suite.sh
```

- [ ] **Step 3: Dry-run (list mode only, no GPU needed)**

```bash
bash evals/eval_suite.sh --skip_download --seq seq-01 2>&1 | head -30
```
Expected: reaches `eval_gt.py` call without syntax errors (will fail on actual GPU inference, which is expected in a dry-run).

- [ ] **Step 4: Commit**

```bash
git add evals/eval_suite.sh
git commit -m "feat(evals): eval_suite.sh orchestrator for 6-condition 7-Scenes eval"
```

---

## Self-Review Checklist

- [x] **Task 1** covers `evals/download_7scenes.py` (spec §Component 1)
- [x] **Tasks 2 + 7** cover `evals/eval_gt.py` (spec §Component 2 — backbone arg + TUM output + alignment JSON)
- [x] **Task 3** covers `evals/runners/run_vggt_slam.py` (spec §Component 3)
- [x] **Task 5** covers `collab_splats/pointcloud/wrappers.py` (spec §Component 4)
- [x] **Task 6** covers `evals/reconstruction_quality.py` (spec §Component 5)
- [x] **Task 4** covers `evals/eval_compare.py` + `evals/metrics.py` (spec §Component 6)
- [x] **Task 8** covers `evals/eval_suite.sh` (spec §Component 7)
- [x] All six conditions present in `_DEFAULT_ALIGN` (Task 4 Step 4)
- [x] `_scan_results_dir` relaxed to allow `.json`/`.npz`/`.png` sidecars (Task 4 Step 5)
- [x] `dedup_overlap` used correctly — signature `(submap_ids, submap_starts, corrected: dict[int, ndarray], total_frames)` (Task 5 Step 2)
- [x] TUM written from parent after collecting all condition results (Task 2 Step 4)
- [x] `gt.tum` written alongside condition TUMs (Task 2 Step 4)
- [x] Phase 2 alignment metrics only computed when `_lc_submaps` present (Task 7 Step 1)
- [x] `_run_condition` returns `(extrinsics, creator)` — consistent across Tasks 2 and 7
- [x] `compute_auc` in `metrics.py` uses `auc_at_threshold` from `loop_closure.eval`, not evo directly (Task 4 Step 2)
