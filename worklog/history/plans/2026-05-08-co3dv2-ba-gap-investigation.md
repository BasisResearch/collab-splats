# CO3Dv2 BA Gap Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose why BA hurts at 50fr (70.1 vs 74.4 baseline) and close the gap to paper's 90.0 AUC@30 by testing square preprocessing and full-sequence evaluation.

**Architecture:** Add a configurable `image_preproc` parameter to `VGGTXCreator` and thread it through `eval_gt.py`. Then run a series of eval experiments (seq1 50fr square vs ratio → seq1 full sequence → seq2) recording AUC@30 at each decision gate.

**Tech Stack:** Python, `vggt.utils.load_fn`, `evals/eval_gt.py`, `collab_splats/pointcloud/feedforward/vggtx.py`

---

## Baseline reference (already recorded)

| Seq | Frames | Condition | AUC@30 |
|---|---|---|---|
| seq1 (110_13051_23361) | 50 | baseline | 74.36 |
| seq1 | 50 | ba | 70.12 |
| seq2 (189_20393_38136) | 50 | baseline | 54.22 |
| seq2 | 50 | ba | 50.20 |
| seq3 (540_79043_153212) | 50 | baseline | 0.0 (AR crash) |

Paper targets: VGGT init = 88.2, VGGT+BAE = 90.0

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/feedforward/vggtx.py` | Add `image_preproc` field to `VGGTXCreator`; branch in `_preprocess` |
| `evals/eval_gt.py` | Add `--image_preproc` CLI flag; thread to `_make_creator` |
| `tests/pointcloud/test_vggtx_preproc.py` | New: unit tests for preprocessing mode switch |

---

## Task 1: Add `image_preproc` field to VGGTXCreator

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py:21` (import)
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py:30-32` (comment)
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py` (VGGTXCreator dataclass + `_preprocess`)
- Create: `tests/pointcloud/test_vggtx_preproc.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pointcloud/test_vggtx_preproc.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import torch


def _make_fake_images(n=2, size=518):
    return torch.zeros(n, 3, size, size), torch.zeros(n, 6)


def _make_image_dir(tmp_path, n=2):
    for i in range(n):
        (tmp_path / f"frame_{i:04d}.jpg").touch()
    return tmp_path


def test_vggtx_creator_default_preproc_is_ratio():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    c = VGGTXCreator()
    assert c.image_preproc == "ratio"


def test_vggtx_creator_accepts_square_preproc():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    c = VGGTXCreator(image_preproc="square")
    assert c.image_preproc == "square"


def test_vggtx_creator_rejects_invalid_preproc():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    with pytest.raises(ValueError, match="image_preproc"):
        VGGTXCreator(image_preproc="invalid")


def test_preprocess_ratio_calls_ratio_fn(tmp_path):
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    img_dir = _make_image_dir(tmp_path)
    c = VGGTXCreator(image_preproc="ratio")
    fake = _make_fake_images()
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_ratio", return_value=fake) as m, \
         patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_square") as ms:
        c._preprocess(img_dir)
        assert m.called
        assert not ms.called


def test_preprocess_square_calls_square_fn(tmp_path):
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    img_dir = _make_image_dir(tmp_path)
    c = VGGTXCreator(image_preproc="square")
    fake = _make_fake_images()
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_square", return_value=fake) as m, \
         patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_ratio") as mr:
        c._preprocess(img_dir)
        assert m.called
        assert not mr.called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_preproc.py -v 2>&1 | tail -20
```

Expected: 4–5 failures (`VGGTXCreator` has no `image_preproc` field, `load_and_preprocess_images_square` not imported).

- [ ] **Step 3: Update vggtx.py import and VGGTXCreator**

In `collab_splats/pointcloud/feedforward/vggtx.py`, make these changes:

**Line 21** — replace the single import with both:
```python
from vggt.utils.load_fn import load_and_preprocess_images_ratio, load_and_preprocess_images_square
```

**Lines 30–32** — update comment:
```python
# VGGT-X processes images at a fixed square resolution.
# image_preproc="ratio" preserves aspect ratio (pads); "square" center-pads to square (paper default).
# original_coords stores original pixel positions for downstream intrinsic rescaling.
```

**VGGTXCreator dataclass** — find the `@dataclass` class definition (search for `class VGGTXCreator`) and add the field. The class uses `@dataclass` decorator. Add `image_preproc: str = "ratio"` as a field, and add `__post_init__` validation:

```python
image_preproc: str = "ratio"  # "ratio" | "square"

def __post_init__(self) -> None:
    if self.image_preproc not in ("ratio", "square"):
        raise ValueError(f"image_preproc must be 'ratio' or 'square', got {self.image_preproc!r}")
```

If `VGGTXCreator` already has a `__post_init__`, append the validation to it rather than adding a new one.

**`_preprocess` method** — replace lines 253–255:
```python
        _load_fn = (
            load_and_preprocess_images_square
            if self.image_preproc == "square"
            else load_and_preprocess_images_ratio
        )
        images, original_coords = _load_fn(
            image_names, VGGTX_IMG_LOAD_RESOLUTION
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_preproc.py -v 2>&1 | tail -20
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_vggtx_preproc.py
git commit -m "feat(vggtx): add image_preproc field to VGGTXCreator (ratio|square)"
```

---

## Task 2: Thread `--image_preproc` through eval_gt.py

**Files:**
- Modify: `evals/eval_gt.py` — add CLI flag, thread to `_make_creator` and `get_creator`

- [ ] **Step 1: Add `--image_preproc` argument to `_build_parser`**

In `evals/eval_gt.py`, find `_build_parser()` and add after the `--conditions` argument:

```python
parser.add_argument("--image_preproc", default="ratio", choices=["ratio", "square"],
                    help="Image preprocessing mode passed to VGGTXCreator.")
```

- [ ] **Step 2: Thread `image_preproc` through `_make_creator`**

Change `_make_creator` signature:

```python
def _make_creator(condition: str, submap_size: int | None = None, image_preproc: str = "ratio"):
```

Replace `base = get_creator("vggtx")()` with:

```python
    base = get_creator("vggtx")(image_preproc=image_preproc)
```

- [ ] **Step 3: Pass `image_preproc` at all `_make_creator` call sites**

Find every call to `_make_creator(...)` in eval_gt.py. There are two: one in `_run_condition` and one in the subprocess mode. Update both:

In `_run_condition`:
```python
def _run_condition(name: str, image_dir: Path, output_dir: Path, submap_size: int | None = None, image_preproc: str = "ratio") -> np.ndarray:
    creator = _make_creator(name, submap_size=submap_size, image_preproc=image_preproc)
```

In the orchestrator loop (where conditions are iterated), pass `image_preproc=args.image_preproc` to `_run_condition`.

In subprocess mode, pass `--image_preproc` through to the subprocess command so the child process inherits the flag. Find where subprocess args are built and add:
```python
"--image_preproc", args.image_preproc,
```

- [ ] **Step 4: Verify CLI help shows the new flag**

```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py --help 2>&1 | grep image_preproc
```

Expected output: `--image_preproc {ratio,square}`

- [ ] **Step 5: Commit**

```bash
git add evals/eval_gt.py
git commit -m "feat(eval): add --image_preproc flag to eval_gt.py"
```

---

## Task 3: Run preprocessing comparison — seq1, 50 frames

**Decision gate:** Does `--image_preproc square` raise baseline AUC@30 above 74.36? Does BA help (ba > baseline)?

- [ ] **Step 1: Verify seq1 data exists**

```bash
ls /workspace/collab-splats/data/co3dv2/apple/110_13051_23361/images/ | wc -l
```

Expected: some number ≥ 50. If directory missing, re-run download:
```bash
bash evals/download_co3dv2.sh apple 110_13051_23361
```

- [ ] **Step 2: Run square preprocessing eval, seq1 50fr**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/110_13051_23361 \
  --output_dir ./eval_results/co3dv2_apple_seq1_square_50fr \
  --max_frames 50 \
  --image_preproc square \
  --conditions baseline ba ba_hightrack
```

Expected runtime: ~15–25 min (3 conditions × ~50fr).

- [ ] **Step 3: Record results**

```bash
cat eval_results/co3dv2_apple_seq1_square_50fr/metrics.json
```

Fill in this table (add to this plan file or a results log):

| Preproc | Frames | Condition | AUC@30 |
|---|---|---|---|
| ratio | 50 | baseline | 74.36 (existing) |
| ratio | 50 | ba | 70.12 (existing) |
| square | 50 | baseline | **___ (fill in)** |
| square | 50 | ba | **___ (fill in)** |
| square | 50 | ba_hightrack | **___ (fill in)** |

- [ ] **Step 4: Evaluate decision gate**

- If `square baseline > ratio baseline` → preprocessing is a factor. Proceed to Task 4 with square.
- If `square ba > square baseline` → BA now helps with square preproc at 50fr.
- If `square baseline ≤ ratio baseline` → preprocessing not the issue. Check: did vggtx.py changes take effect? Re-run with a print statement in `_preprocess` to confirm the branch is hit.

---

## Task 4: Full sequence eval — seq1, winning preprocessing

**Decision gate:** Does BA help over baseline at full-sequence length?

- [ ] **Step 1: Check actual frame count**

```bash
ls /workspace/collab-splats/data/co3dv2/apple/110_13051_23361/images/ | wc -l
```

If frame count > 200, add `--submap_size 50` to avoid GPU OOM (see Step 2 variant).

- [ ] **Step 2: Run full-sequence eval**

Use the winning `--image_preproc` from Task 3 (expected: `square`). No `--max_frames` (default is 500, effectively uncapped for CO3Dv2 sequences):

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/110_13051_23361 \
  --output_dir ./eval_results/co3dv2_apple_seq1_square_full \
  --image_preproc square \
  --conditions baseline ba ba_hightrack
```

If OOM (`CUDA out of memory`), rerun with windowed inference:
```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/110_13051_23361 \
  --output_dir ./eval_results/co3dv2_apple_seq1_square_full \
  --image_preproc square \
  --submap_size 50 \
  --conditions baseline ba ba_hightrack
```

Expected runtime: ~45–90 min.

- [ ] **Step 3: Record results**

```bash
cat eval_results/co3dv2_apple_seq1_square_full/metrics.json
```

| Preproc | Frames | Condition | AUC@30 |
|---|---|---|---|
| square | full | baseline | **___ (fill in)** |
| square | full | ba | **___ (fill in)** |
| square | full | ba_hightrack | **___ (fill in)** |

- [ ] **Step 4: Evaluate decision gate**

- If `ba > baseline` → sequence length was the root cause of BA regression. 
- If `ba ≤ baseline` on full seq → BA code issue. Investigate: `run_bundle_adjustment` in `collab_splats/pointcloud/bundle_adjustment.py`, check Sim3 graph construction and track confidence filtering.
- Target: `ba_hightrack ≥ ba ≥ baseline`, baseline approaching 85+.

---

## Task 5: Scale to seq2 — full sequence, winning config

- [ ] **Step 1: Verify seq2 data exists**

```bash
ls /workspace/collab-splats/data/co3dv2/apple/189_20393_38136/images/ | wc -l
```

If missing: `bash evals/download_co3dv2.sh apple 189_20393_38136`

- [ ] **Step 2: Run full-sequence eval on seq2**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/189_20393_38136 \
  --output_dir ./eval_results/co3dv2_apple_seq2_square_full \
  --image_preproc square \
  --conditions baseline ba ba_hightrack
```

Add `--submap_size 50` if OOM.

- [ ] **Step 3: Record results**

```bash
cat eval_results/co3dv2_apple_seq2_square_full/metrics.json
```

| Seq | Preproc | Frames | Condition | AUC@30 |
|---|---|---|---|---|
| seq2 | square | full | baseline | **___** |
| seq2 | square | full | ba | **___** |
| seq2 | square | full | ba_hightrack | **___** |

Note: seq2 had AUC@30=54.22 at 50fr with ratio. If square full-seq ≥ 70, that's strong validation. If seq2 stays low (hard sequence), that's expected — note as outlier.

---

## Task 6: Compile final results table and commit

- [ ] **Step 1: Write results summary**

Create `eval_results/co3dv2_investigation_summary.md`:

```markdown
# CO3Dv2 Investigation Results — 2026-05-08

| Seq | Preproc | Frames | Condition | AUC@30 | Notes |
|---|---|---|---|---|---|
| seq1 | ratio | 50 | baseline | 74.36 | Original |
| seq1 | ratio | 50 | ba | 70.12 | BA hurts |
| seq1 | square | 50 | baseline | ___ | Task 3 |
| seq1 | square | 50 | ba | ___ | Task 3 |
| seq1 | square | 50 | ba_hightrack | ___ | Task 3 |
| seq1 | square | full | baseline | ___ | Task 4 |
| seq1 | square | full | ba | ___ | Task 4 |
| seq1 | square | full | ba_hightrack | ___ | Task 4 |
| seq2 | square | full | baseline | ___ | Task 5 |
| seq2 | square | full | ba | ___ | Task 5 |
| seq2 | square | full | ba_hightrack | ___ | Task 5 |
| paper | — | full | VGGT init | 88.2 | Reference |
| paper | — | full | VGGT+BAE | 90.0 | Target |
```

- [ ] **Step 2: Update investigation doc with findings**

Edit `docs/superpowers/2026-05-08-co3dv2-eval-gap-investigation.md` — mark completed steps, add actual AUC values next to hypotheses.

- [ ] **Step 3: Commit all results and code**

```bash
git add eval_results/co3dv2_investigation_summary.md \
        docs/superpowers/2026-05-08-co3dv2-eval-gap-investigation.md
git commit -m "chore(eval): CO3Dv2 preprocessing investigation results"
```

---

## Fallback Reference

| Symptom | Action |
|---|---|
| `square baseline ≤ ratio baseline` at 50fr | Add debug print in `_preprocess` to confirm branch; check `original_coords` shape matches expected (N, 6) |
| `ba ≤ baseline` at full seq | Inspect BA: add `--conditions baseline ba` only, check `run_bundle_adjustment` logs for convergence |
| GPU OOM on full seq | Add `--submap_size 50` |
| seq2 AUC < 40 on full seq | Mark as hard sequence; proceed to different category |
| `load_and_preprocess_images_square` import error | Check vggt version: `/opt/conda/envs/nerfstudio/bin/python -c "from vggt.utils.load_fn import load_and_preprocess_images_square; print('ok')"` |
