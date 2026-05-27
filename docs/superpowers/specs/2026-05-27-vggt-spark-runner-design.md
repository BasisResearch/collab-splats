# VGGT-SPARK Integration for run_vggt_slam_lc.py

**Amends:** `2026-05-27-vggt-slam-parity-verification-design.md` (Phase 1 runner only)  
**Status:** Approved — supersedes the `_VGGTCompatWrapper` approach

---

## Goal

Produce a valid `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` by running the
MIT-SPARK VGGT-SLAM pipeline with a model that natively supports `compute_similarity=True`.

---

## Problem with Current State

`run_vggt_slam_lc.py` contains `_VGGTCompatWrapper` which injects `image_match_ratio=1.0`
when `compute_similarity=True` is called. This is wrong: solver checks
`image_match_ratio > lc_thres` (default 0.95), so 1.0 forces **every** loop candidate to
pass regardless of actual frame similarity. Resulting TUM trajectory has false loop
closures → parity analysis is invalid.

---

## Decision: VGGT-SPARK via sys.path shadow

### Why VGGT-SPARK (not Option A QKV wrapper)

Option A wraps our VGGT-X model to compute similarity via attention QKV hooks. This
produces real similarity values but from VGGT-X, not VGGT-1B. VGGT-SLAM was designed and
tuned with VGGT-SPARK (MIT-SPARK's fork of VGGT-1B). Using VGGT-SPARK gives the oracle
baseline that matches published VGGT-SLAM results exactly.

VGGT-SPARK properties (confirmed):
- Same weights as VGGT-1B (`facebook/VGGT-1B` checkpoint loads directly)
- Same API — additive only: adds `compute_similarity=True` kwarg to `VGGT.forward()`
- Returns `predictions["image_match_ratio"]` (real attention-based similarity)
- Public repo: `https://github.com/MIT-SPARK/VGGT_SPARK`

### Why sys.path shadow (not separate conda env)

A separate conda env adds ~8-10 GB disk, requires activation discipline across multiple
runner scripts, and is unnecessary since VGGT-SLAM uses no custom CUDA extensions
(pure PyTorch — cu121/torch 2.5.1 confirmed compatible via runs 1 & 2).

The sys.path shadow is per-process and isolated. `run_vggt_slam_lc.py` is a standalone
script; inserting `third_party/vggt_spark/` at `sys.path[0]` before any `vggt` import
causes Python to load VGGT-SPARK for that process only. `sys.modules` cache propagates
to all VGGT-SLAM internal imports in the same process. `reconstruction` env's installed
`vggt` package (VGGT-X based) is completely unaffected.

---

## Setup

Clone VGGT-SPARK as an uninstalled source tree (no pip install):

```bash
cd third_party
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git vggt_spark
cd ..
```

Verify package structure exists:

```bash
ls third_party/vggt_spark/vggt/__init__.py  # must exist
```

No changes to `setup_feedforward.sh` needed for the eval runner. If this becomes
part of the standard setup, add a `third_party/vggt_spark` clone step there.

---

## Changes to `run_vggt_slam_lc.py`

### 1. sys.path shadow — top of file, before all imports

```python
# Shadow installed vggt with VGGT-SPARK (MIT-SPARK fork that supports compute_similarity).
# This must come before any vggt import. Affects this process only — reconstruction env
# vggt package (VGGT-X) is unaffected.
import sys
from pathlib import Path
_vggt_spark = str(Path(__file__).resolve().parents[2] / "third_party" / "vggt_spark")
sys.path.insert(0, _vggt_spark)
```

### 2. Remove `_VGGTCompatWrapper`

Delete the entire class definition and the `model = _VGGTCompatWrapper(_vggt)` line.
Replace with direct assignment:

```python
model = _vggt  # VGGT-SPARK natively handles compute_similarity=True
```

### 3. Keep VGGT-1B weights

```python
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
_vggt = VGGT()
_vggt.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
```

Same checkpoint — VGGT-SPARK's `VGGT` class loads VGGT-1B weights unchanged.

---

## min_disparity=0 — intentional deviation from paper defaults

VGGT-SLAM paper uses `min_disparity=50` (optical flow keyframe filter). On chess_seq01
with `--max_frames 200`, `min_disparity=50` rejects enough frames that **no loop closure
candidates are detected** — making the parity comparison vacuous.

Decision: both `run_vggt_slam_lc.py` and `our_solver_dump.py` use `min_disparity=0`
(accept all frames). This deviates from the paper's deployment defaults but is correct
for this evaluation: both systems see identical frame sets, loop closures are triggered,
and the comparison isolates algorithm parity rather than keyframe selection differences.

This deviation is intentional and must not be "fixed" back to 50.

---

## Correctness Check: runner vs main.py

| Aspect | VGGT-SLAM main.py | run_vggt_slam_lc.py | Status |
|--------|-------------------|---------------------|--------|
| optical flow | `use_optical_flow_downsample = True` hardcoded | `min_disparity <= 0.0` bypass | ✓ intentional (see above) |
| dtype | `torch.bfloat16` hardcoded | adaptive bfloat16/float16 | ✓ equivalent on A100 |
| image sort | `glob.glob` (no sort) | `glob.glob` (no sort) | ✓ consistent |
| solver init | `Solver(conf, lc_thres, vis_voxel_size)` | same | ✓ |
| `run_predictions` call | `(names, model, max_loops, clip_model, clip_preprocess)` | same, clip=None | ✓ |
| `add_points` + `optimize` order | sequential | same | ✓ |
| overlap reset | `image_names_subset[-overlapping_window_size:]` | same | ✓ |
| TUM output | `write_poses_to_file(..., kitti_format=False)` | same | ✓ |

---

## Execution

After setup:

```bash
PY=/opt/conda/envs/reconstruction/bin/python

$PY evals/runners/run_vggt_slam_lc.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --out_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
  --max_frames 200
```

Produces `vggt_slam_lc.tum`. Then proceed with CPU analysis per handoff doc:

```bash
$PY evals/runners/compare_solver_internals.py

$PY evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
  --ours_tum evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16
```

---

## Success Criteria

- `run_vggt_slam_lc.py` completes without `TypeError: compute_similarity`
- `vggt_slam_lc.tum` exists with ~200 timestamp entries
- Loop closure count > 0 in solver log (confirms LC was triggered and passed real similarity gate)
- `image_match_ratio` values in solver output are < 1.0 (confirms real computation, not stub)
