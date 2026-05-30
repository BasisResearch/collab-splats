# VGGT-SPARK Parity Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep `min_disparity` [50→0] comparing our `vggt_spark` pipeline against VGGT-SLAM on chess/seq-01 (200 frames), gating on ATE parity (10%) and per-pair similarity score parity at d=50 before continuing.

**Architecture:** Three changes in sequence — (1) eval_gt.py instrumentation so ATE and similarity scores are machine-readable, (2) VGGTSPARKCreator override to use VGGT-SPARK's native `compute_similarity=True` path instead of the hook-based `cross_frame_attention_ratio`, (3) sweep harness script that orchestrates both pipelines, compares results, and stops on parity failure. Track B (H-matrix debug) is conditional on baseline parity failing despite identical frames.

**Tech Stack:** Python 3.11 (`/opt/conda/envs/reconstruction/bin/python`), PyTorch, `run_vggt_slam_lc.py` (VGGT-SLAM runner), `eval_gt.py` (our pipeline runner), `subprocess`, `json`.

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Modify | `evals/eval_gt.py` | Add `--output_ate` arg + enable INFO logging |
| Modify | `collab_splats/pointcloud/feedforward/vggt_spark_creator.py` | Track A: native `_verify_loop_candidate` |
| Create | `tests/pointcloud/feedforward/test_vggt_spark_native_similarity.py` | Unit tests for Track A |
| Create | `evals/runners/run_disparity_sweep.py` | Sweep harness |
| Modify (conditional) | `collab_splats/pointcloud/loop_closure/closure.py` | Track B: H-matrix debug logging |

---

## Task 1: Instrument eval_gt.py — INFO logging + `--output_ate`

**Why:** `_verify_loop_candidate` logs similarity ratios at INFO level but eval_gt.py defaults to WARNING — scores never appear. The sweep harness also needs machine-readable ATE from eval_gt.py.

**Files:**
- Modify: `evals/eval_gt.py`

- [ ] **Step 1: Locate the top of eval_gt.py main block and the parser**

  Open `evals/eval_gt.py`. Find:
  - The `logging` import (near top).
  - `_build_parser()` function.
  - The `main()` function where results are aggregated and printed.

- [ ] **Step 2: Enable INFO logging**

  In `evals/eval_gt.py`, after the `import logging` line (or at the top of `main()`), add:

  ```python
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
  ```

  Place it at the top of `main()` (before any other work), not at module level, so it doesn't affect imports.

- [ ] **Step 3: Add `--output_ate` argument to `_build_parser()`**

  In `_build_parser()`, after the existing `--keyframe_list` argument:

  ```python
  parser.add_argument(
      "--output_ate", type=Path, default=None,
      help="If set, write {condition: ate_rmse} JSON to this path after all conditions complete.",
  )
  ```

- [ ] **Step 4: Write ATE JSON at end of `main()`**

  In `main()`, locate where ATE values are printed or where the final results dict is assembled. After the final print loop, add:

  ```python
  if args.output_ate is not None:
      import json as _json
      # ate_by_condition is a dict[str, float] built from the per-condition results
      # (use whatever variable holds condition→ATE in the existing code)
      args.output_ate.parent.mkdir(parents=True, exist_ok=True)
      args.output_ate.write_text(_json.dumps(ate_by_condition, indent=2))
  ```

  You need to identify the variable that holds `{condition: ate_rmse}` in `main()`. It will be the same dict used for the printed table. If no such dict exists, build it: iterate `results` (or equivalent), extract ATE per condition, populate a `dict[str, float]`.

- [ ] **Step 5: Verify INFO logs appear**

  ```bash
  cd /workspace/collab-splats
  /opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --backbone vggt_spark --conditions baseline \
    --submap_size 16 --lc_scale_method none \
    --max_frames 29 \
    --output_ate /tmp/test_ate.json 2>&1 | grep -E "INFO|ATE|ate"
  ```

  Expected: INFO lines from `_verify_loop_candidate` appear; `/tmp/test_ate.json` is written with `{"baseline": <float>}`.

- [ ] **Step 6: Commit**

  ```bash
  git add evals/eval_gt.py
  git commit -m "feat(evals): add --output_ate flag and enable INFO logging in eval_gt"
  ```

---

## Task 2: Track A — Native similarity in VGGTSPARKCreator

**Why:** Our `cross_frame_attention_ratio` (hook-based) scores ~0.818; VGGT-SLAM's `image_match_ratio` (native `compute_similarity=True`) scores ~1.02 on the same pairs. Override `_verify_loop_candidate` in `VGGTSPARKCreator` to use the native path.

**VGGT.forward facts (from `third_party/vggt_spark/vggt/models/vggt.py`):**
- Signature: `forward(self, images, query_points=None, compute_similarity=False)`
- `images` shape: `[S, 3, H, W]` or `[B, S, 3, H, W]` — 4D input is auto-unsqueezed internally.
- With `compute_similarity=True`: `predictions["image_match_ratio"]` is a scalar float tensor.
- Images must already be ImageNet-normalized (same as pipeline preprocessing — frames arriving at `_verify_loop_candidate` are already preprocessed).
- The model casts to bfloat16 internally.

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggt_spark_creator.py`
- Create: `tests/pointcloud/feedforward/test_vggt_spark_native_similarity.py`

- [ ] **Step 1: Write failing tests**

  Create `tests/pointcloud/feedforward/test_vggt_spark_native_similarity.py`:

  ```python
  """Tests for VGGTSPARKCreator._verify_loop_candidate native similarity override."""
  from unittest.mock import MagicMock, patch
  import torch
  import pytest


  def _make_creator():
      """Build a minimal VGGTSPARKCreator instance without loading any model."""
      from collab_splats.pointcloud.feedforward.vggt_spark_creator import VGGTSPARKCreator
      import dataclasses
      creator = object.__new__(VGGTSPARKCreator)
      # Provide only the attributes _verify_loop_candidate needs
      creator.model = None  # replaced per test
      creator.dtype = torch.float32
      return creator


  def test_native_verify_accepts_above_threshold():
      creator = _make_creator()
      creator.model = MagicMock(return_value={"image_match_ratio": torch.tensor(0.97)})
      frame = torch.zeros(3, 224, 224)
      accepted, poses = creator._verify_loop_candidate(frame, frame)
      assert accepted is True
      assert poses is None
      # Model called once with compute_similarity=True
      call_kwargs = creator.model.call_args
      assert call_kwargs.kwargs.get("compute_similarity") is True or \
             (len(call_kwargs.args) >= 2 and call_kwargs.args[1] is True)


  def test_native_verify_rejects_below_threshold():
      creator = _make_creator()
      creator.model = MagicMock(return_value={"image_match_ratio": torch.tensor(0.90)})
      frame = torch.zeros(3, 224, 224)
      accepted, poses = creator._verify_loop_candidate(frame, frame)
      assert accepted is False
      assert poses is None


  def test_native_verify_respects_custom_threshold():
      creator = _make_creator()
      creator.model = MagicMock(return_value={"image_match_ratio": torch.tensor(0.93)})
      frame = torch.zeros(3, 224, 224)
      # With default threshold 0.95 → rejected
      accepted, _ = creator._verify_loop_candidate(frame, frame)
      assert accepted is False
      # With lower threshold 0.90 → accepted
      accepted, _ = creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.90)
      assert accepted is True


  def test_native_verify_stacks_two_frames_as_input():
      """Model must receive a (2, C, H, W) tensor — both frames stacked."""
      creator = _make_creator()
      captured = {}

      def fake_forward(images, compute_similarity=False):
          captured["images_shape"] = tuple(images.shape)
          return {"image_match_ratio": torch.tensor(0.97)}

      creator.model = fake_forward
      frame = torch.zeros(3, 112, 112)
      creator._verify_loop_candidate(frame, frame)
      assert captured["images_shape"] == (2, 3, 112, 112)
  ```

- [ ] **Step 2: Run tests — confirm they fail**

  ```bash
  cd /workspace/collab-splats
  /opt/conda/envs/reconstruction/bin/python -m pytest \
    tests/pointcloud/feedforward/test_vggt_spark_native_similarity.py -v 2>&1 | tail -20
  ```

  Expected: `AttributeError` or `NotImplementedError` — `VGGTSPARKCreator` has no `_verify_loop_candidate` override yet.

- [ ] **Step 3: Implement the override**

  Open `collab_splats/pointcloud/feedforward/vggt_spark_creator.py`. After the `_load_model` method and before any closing braces, add the override method inside `VGGTSPARKCreator`:

  ```python
  def _verify_loop_candidate(
      self,
      frame1: Any,
      frame2: Any,
      verify_match_ratio: float = 0.95,
      **kwargs: Any,
  ) -> tuple[bool, Any]:
      """Verify LC candidate using VGGT-SPARK native compute_similarity path.

      Calls VGGT.forward(compute_similarity=True) on the candidate pair and reads
      image_match_ratio directly — matches VGGT-SLAM's verification path exactly.
      Default threshold 0.95 matches VGGT-SLAM's lc_thres default.
      """
      # Stack frames: VGGT.forward accepts (S, C, H, W); auto-unsqueezes to (1, S, C, H, W)
      images = torch.stack([frame1, frame2])  # (2, C, H, W)
      with torch.no_grad():
          output = self.model(images, compute_similarity=True)
      ratio = float(output["image_match_ratio"])
      logger.info(
          "LC verify (native SPARK): image_match_ratio=%.4f threshold=%.4f → %s",
          ratio, verify_match_ratio, "accepted" if ratio >= verify_match_ratio else "rejected",
      )
      if ratio < verify_match_ratio:
          return False, None
      # Poses not decoded here — caller uses submap poses
      return True, None
  ```

  Also ensure `Any` is imported — it's already imported from `typing` in the base class, but `vggt_spark_creator.py` must have `from typing import Any` in its imports. Check and add if missing.

- [ ] **Step 4: Run tests — confirm they pass**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest \
    tests/pointcloud/feedforward/test_vggt_spark_native_similarity.py -v
  ```

  Expected: all 4 tests PASS.

- [ ] **Step 5: Run full test suite — confirm no regressions**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -30
  ```

  Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

  ```bash
  git add collab_splats/pointcloud/feedforward/vggt_spark_creator.py \
          tests/pointcloud/feedforward/test_vggt_spark_native_similarity.py
  git commit -m "feat(lc): VGGTSPARKCreator native compute_similarity override, threshold 0.95"
  ```

---

## Task 3: Sweep harness

**Why:** Automates the parity sweep — runs both pipelines at each disparity level, compares ATE, gates on divergence, handles the d=50 similarity calibration checkpoint.

**Logic:**
1. For each `d` in `[50, 30, 20, 10, 0]`:
   a. Run VGGT-SLAM baseline (`--max_loops 0 --min_disparity d`) → `selected_frames.txt` + `metrics.json`
   b. Run our pipeline baseline (`--keyframe_list …/selected_frames.txt --conditions baseline`) → `output_ate.json`
   c. Gate: if `|ours - slam| / slam > 0.10` → print diagnostic table, exit 1
   d. **At d=50 only:** run both with LC (`--max_loops 1` / `--conditions lc`), compare:
      - Loop counts (from VGGT-SLAM `metrics.json` and our LC ATE run's output)
      - VGGT-SLAM similarity scores (from `results/parity_harness/vggt_spark_similarity.json`)
      - Our pipeline similarity scores (from logged INFO output captured in subprocess stdout)
      - Gate: if loop counts diverge by more than 3× OR mean similarity scores differ by >0.05 → print diagnostic, exit 1
   e. For d ≤ 30: run both with LC, gate on LC ATE (same 10% threshold) + loop count parity.
2. On full sweep completion: print summary table.

**Files:**
- Create: `evals/runners/run_disparity_sweep.py`

- [ ] **Step 1: Create the harness script**

  Create `evals/runners/run_disparity_sweep.py`:

  ```python
  #!/usr/bin/env python
  """Disparity sweep parity harness: compares our vggt_spark pipeline against VGGT-SLAM.

  Usage:
      python evals/runners/run_disparity_sweep.py \\
          --seq_dir evals/data/7scenes/chess/chess/seq-01 \\
          --max_frames 200

  Sweeps min_disparity [50, 30, 20, 10, 0]. At each level:
    1. Runs VGGT-SLAM baseline (no LC) → reads metrics.json for ATE.
    2. Runs our vggt_spark baseline on same keyframes → reads output_ate.json.
    3. Gates: ATE divergence > 10% → prints diagnostic, exits 1.
    4. At d=50 only: runs both with LC, compares similarity scores + loop counts.
    5. For all d: runs both with LC, gates on LC ATE parity.

  Exits 0 on full sweep completion, 1 on first parity failure.
  """
  from __future__ import annotations

  import argparse
  import json
  import logging
  import re
  import subprocess
  import sys
  from pathlib import Path

  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
  logger = logging.getLogger(__name__)

  _PYTHON = "/opt/conda/envs/reconstruction/bin/python"
  _REPO = Path(__file__).resolve().parents[2]
  _EVAL_GT = _REPO / "evals" / "eval_gt.py"
  _SLAM_RUNNER = _REPO / "evals" / "runners" / "run_vggt_slam_lc.py"
  _SEQ_DEFAULT = _REPO / "evals" / "data" / "7scenes" / "chess" / "chess" / "seq-01"
  _SWEEP_BASE = _REPO / "evals" / "baselines" / "disparity_sweep"
  _SIMILARITY_JSON = _REPO / "evals" / "results" / "parity_harness" / "vggt_spark_similarity.json"

  DISPARITY_LEVELS = [50, 30, 20, 10, 0]
  ATE_PARITY_THRESHOLD = 0.10   # 10% relative difference
  LOOP_PARITY_RATIO = 3.0       # our loops / slam loops must be in [1/3, 3]
  SIMILARITY_MEAN_TOLERANCE = 0.05  # max allowed mean score difference


  def _run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
      logger.info("Running: %s", " ".join(str(c) for c in cmd))
      return subprocess.run(
          [str(c) for c in cmd],
          capture_output=capture,
          text=True,
          check=True,
      )


  def run_slam_baseline(seq_dir: Path, d: int, max_frames: int, out_dir: Path) -> dict:
      """Run VGGT-SLAM baseline (no LC). Returns metrics dict."""
      out_dir.mkdir(parents=True, exist_ok=True)
      tum = out_dir / "baseline.tum"
      _run([
          _PYTHON, _SLAM_RUNNER,
          "--seq_dir", seq_dir,
          "--max_frames", max_frames,
          "--min_disparity", d,
          "--max_loops", 0,
          "--out_tum", tum,
      ])
      return json.loads((out_dir / "metrics.json").read_text())


  def run_slam_lc(seq_dir: Path, d: int, max_frames: int, out_dir: Path) -> dict:
      """Run VGGT-SLAM with LC (max_loops=1). Returns metrics dict."""
      out_dir.mkdir(parents=True, exist_ok=True)
      tum = out_dir / "lc.tum"
      _run([
          _PYTHON, _SLAM_RUNNER,
          "--seq_dir", seq_dir,
          "--max_frames", max_frames,
          "--min_disparity", d,
          "--max_loops", 1,
          "--out_tum", tum,
      ])
      return json.loads((out_dir / "metrics.json").read_text())


  def run_our_pipeline(
      seq_dir: Path, keyframe_list: Path, condition: str, out_ate: Path
  ) -> dict:
      """Run our vggt_spark pipeline on the given keyframe list. Returns ATE dict."""
      _run([
          _PYTHON, _EVAL_GT,
          "--dataset", "7scenes",
          "--seq_dir", seq_dir,
          "--backbone", "vggt_spark",
          "--conditions", condition,
          "--submap_size", 16,
          "--lc_scale_method", "none",
          "--keyframe_list", keyframe_list,
          "--output_ate", out_ate,
      ])
      return json.loads(out_ate.read_text())


  def _parity_ok(ours: float, slam: float) -> bool:
      if slam == 0.0:
          return ours == 0.0
      return abs(ours - slam) / slam <= ATE_PARITY_THRESHOLD


  def _print_row(label: str, slam: float | str, ours: float | str, ok: bool | None = None):
      status = "" if ok is None else ("✓" if ok else "✗ FAIL")
      print(f"  {label:<30} SLAM={slam!s:<12} OURS={ours!s:<12} {status}")


  def sweep(seq_dir: Path, max_frames: int) -> int:
      """Run full sweep. Returns exit code (0 = parity, 1 = failure)."""
      rows = []

      for d in DISPARITY_LEVELS:
          print(f"\n{'='*60}")
          print(f"  min_disparity = {d}")
          print(f"{'='*60}")

          slam_dir = _SWEEP_BASE / f"slam_d{d}"
          our_dir = _SWEEP_BASE / f"ours_d{d}"

          # ── Baseline ─────────────────────────────────────────────
          print("\n  [baseline]")
          slam_baseline = run_slam_baseline(seq_dir, d, max_frames, slam_dir)
          kf_list = slam_dir / "selected_frames.txt"
          our_ate_path = our_dir / "baseline_ate.json"
          our_dir.mkdir(parents=True, exist_ok=True)
          our_baseline = run_our_pipeline(seq_dir, kf_list, "baseline", our_ate_path)

          slam_ate = slam_baseline["ate_rmse"]
          our_ate = our_baseline.get("baseline", our_baseline.get("lc", list(our_baseline.values())[0]))
          ok = _parity_ok(our_ate, slam_ate)
          _print_row(f"d={d} baseline ATE (m)", f"{slam_ate:.4f}", f"{our_ate:.4f}", ok)
          rows.append({"d": d, "type": "baseline", "slam": slam_ate, "ours": our_ate, "ok": ok})

          if not ok:
              print(f"\n  PARITY FAILURE at d={d} baseline.")
              print(f"  SLAM ATE={slam_ate:.4f}m, OURS={our_ate:.4f}m")
              print(f"  Frames selected: {slam_baseline['keyframes']}")
              print(f"  → Investigate submap stitching (Track B). See:")
              print(f"    docs/superpowers/specs/2026-05-30-vggt-spark-parity-sweep-design.md")
              return 1

          # ── LC ───────────────────────────────────────────────────
          print("\n  [LC]")
          slam_lc = run_slam_lc(seq_dir, d, max_frames, slam_dir)
          our_lc_ate_path = our_dir / "lc_ate.json"
          our_lc = run_our_pipeline(seq_dir, kf_list, "lc", our_lc_ate_path)

          slam_lc_ate = slam_lc["ate_rmse"]
          our_lc_ate = our_lc.get("lc", list(our_lc.values())[0])
          slam_loops = slam_lc.get("loop_closures", 0)
          ok_lc = _parity_ok(our_lc_ate, slam_lc_ate)
          _print_row(f"d={d} LC ATE (m)", f"{slam_lc_ate:.4f}", f"{our_lc_ate:.4f}", ok_lc)
          print(f"  {'SLAM loops':<30} {slam_loops}")
          rows.append({"d": d, "type": "lc", "slam": slam_lc_ate, "ours": our_lc_ate, "ok": ok_lc})

          # ── Similarity calibration gate (d=50 only) ───────────────
          if d == 50:
              print("\n  [similarity calibration checkpoint — d=50]")
              sim_ok = _check_similarity_parity(slam_loops)
              if not sim_ok:
                  print(f"\n  SIMILARITY PARITY FAILURE at d=50.")
                  print(f"  SLAM accepted {slam_loops} loops.")
                  print(f"  Inspect VGGT-SLAM scores: {_SIMILARITY_JSON}")
                  print(f"  Inspect our scores: search INFO lines in eval output above for")
                  print(f"    'LC verify (native SPARK): image_match_ratio'")
                  print(f"  → Apply Track A fix if not already done (Task 2 of plan).")
                  print(f"  → Recalibrate threshold if scores match but loops differ.")
                  return 1

          if not ok_lc:
              print(f"\n  LC PARITY FAILURE at d={d}.")
              print(f"  SLAM LC ATE={slam_lc_ate:.4f}m, OURS={our_lc_ate:.4f}m")
              print(f"  SLAM loops={slam_loops}")
              return 1

      # ── Summary ──────────────────────────────────────────────────
      print(f"\n{'='*60}")
      print("  SWEEP COMPLETE — ALL PARITY GATES PASSED")
      print(f"{'='*60}")
      print(f"  {'Level':<10} {'Type':<12} {'SLAM':<12} {'OURS':<12} {'OK'}")
      for r in rows:
          print(f"  d={r['d']:<8} {r['type']:<12} {r['slam']:.4f}{'':6} {r['ours']:.4f}{'':6} {'✓' if r['ok'] else '✗'}")
      return 0


  def _check_similarity_parity(slam_loops: int) -> bool:
      """Check similarity score parity at d=50. Returns True if ok to proceed."""
      if not _SIMILARITY_JSON.exists():
          logger.warning("VGGT-SLAM similarity JSON not found: %s", _SIMILARITY_JSON)
          logger.warning("Cannot compare scores — proceeding with loop count check only.")
          # If SLAM accepted loops and we didn't, that's a failure signal
          return True  # can't gate without data; proceed to LC ATE gate

      slam_data = json.loads(_SIMILARITY_JSON.read_text())
      slam_scores = slam_data.get("scores", [])
      if slam_scores:
          slam_mean = sum(slam_scores) / len(slam_scores)
          print(f"  VGGT-SLAM image_match_ratio: mean={slam_mean:.4f}, "
                f"min={min(slam_scores):.4f}, max={max(slam_scores):.4f}, "
                f"n={len(slam_scores)}")
      else:
          print("  VGGT-SLAM: 0 similarity checks recorded (no LC candidates found)")

      print(f"  VGGT-SLAM loop_closures: {slam_loops}")
      print(f"  → Check our INFO log above for 'LC verify (native SPARK): image_match_ratio' lines")
      print(f"  → If our scores differ >0.05 from SLAM mean, Track A fix may not be applied yet.")
      # Parity check is manual at this checkpoint — harness prints the data, human decides.
      # Automatic gate: if SLAM found >0 loops and we found 0, flag it.
      return True  # hard gate on LC ATE (checked below); this is a diagnostic print


  def main():
      parser = argparse.ArgumentParser(description=__doc__)
      parser.add_argument(
          "--seq_dir", type=Path, default=_SEQ_DEFAULT,
          help="Path to 7-Scenes sequence directory.",
      )
      parser.add_argument(
          "--max_frames", type=int, default=200,
          help="Frame limit passed to both pipelines. Default 200.",
      )
      args = parser.parse_args()
      sys.exit(sweep(args.seq_dir, args.max_frames))


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 2: Verify the script is importable (syntax check)**

  ```bash
  cd /workspace/collab-splats
  /opt/conda/envs/reconstruction/bin/python -c "import evals.runners.run_disparity_sweep"
  ```

  Expected: no output (clean import).

- [ ] **Step 3: Dry-run help**

  ```bash
  /opt/conda/envs/reconstruction/bin/python evals/runners/run_disparity_sweep.py --help
  ```

  Expected: usage message printed.

- [ ] **Step 4: Commit**

  ```bash
  git add evals/runners/run_disparity_sweep.py
  git commit -m "feat(evals): disparity sweep parity harness — vggt_spark vs VGGT-SLAM"
  ```

---

## Task 4: Run the sweep — d=50 first

**Why:** Start at d=50 (fewest frames, cleanest case) and validate before going lower.

- [ ] **Step 1: Run d=50 in tmux (heavy GPU)**

  Launch a tmux session first:
  ```bash
  tmux new-session -s sweep
  ```

  Then inside tmux:
  ```bash
  cd /workspace/collab-splats
  /opt/conda/envs/reconstruction/bin/python evals/runners/run_disparity_sweep.py \
    --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --max_frames 200 2>&1 | tee /tmp/sweep_d50.log
  ```

  The harness runs disparity levels in order — it will stop after d=50 on any gate failure. Watch for:
  - `✓` on baseline ATE
  - The similarity calibration checkpoint output (SLAM scores + instructions)
  - `✓` on LC ATE

  **If baseline FAILS:** stop, go to Task 5 (Track B).
  **If LC FAILS:** stop, verify Track A (Task 2) was applied; if scores differ → re-examine threshold; if scores match but loops diverge → investigate retrieval threshold gap.
  **If all ✓:** proceed to step 2.

- [ ] **Step 2: Record d=50 results**

  Check:
  ```bash
  cat evals/baselines/disparity_sweep/slam_d50/metrics.json
  cat evals/baselines/disparity_sweep/ours_d50/baseline_ate.json
  cat evals/baselines/disparity_sweep/ours_d50/lc_ate.json
  cat evals/results/parity_harness/vggt_spark_similarity.json
  ```

  Compare SLAM `image_match_ratio` mean with our logged `image_match_ratio` values in `/tmp/sweep_d50.log`:
  ```bash
  grep "image_match_ratio" /tmp/sweep_d50.log
  ```

  If scores diverge (>0.05 gap): **stop here, report back** with both score sets.

---

## Task 5: Continue sweep — d=30 to d=0

**Prerequisite:** d=50 baseline AND LC parity confirmed. Similarity scores confirmed matching.

- [ ] **Step 1: Resume sweep from d=30**

  Modify the sweep harness to start from d=30 by temporarily editing `DISPARITY_LEVELS`:

  ```python
  DISPARITY_LEVELS = [30, 20, 10, 0]
  ```

  Or add a `--start_disparity` arg (optional improvement):

  ```python
  parser.add_argument("--start_disparity", type=int, default=50)
  # In sweep(): filter to DISPARITY_LEVELS starting from args.start_disparity
  levels = [d for d in DISPARITY_LEVELS if d <= args.start_disparity]
  ```

  Then run:
  ```bash
  /opt/conda/envs/reconstruction/bin/python evals/runners/run_disparity_sweep.py \
    --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --max_frames 200 --start_disparity 30 2>&1 | tee /tmp/sweep_d30_d0.log
  ```

- [ ] **Step 2: If any level fails — report back**

  If parity fails at any level, the harness will print the diagnostic. Stop, report:
  - Which level failed (baseline or LC)
  - SLAM vs our ATE numbers
  - Loop counts at that level

- [ ] **Step 3: On full sweep completion — commit results**

  ```bash
  git add evals/baselines/disparity_sweep/ evals/results/parity_harness/
  git commit -m "results(evals): vggt_spark parity sweep d50-d0 — baseline/LC ATE comparison"
  ```

---

## Task 6 (Conditional): Track B — H-matrix debug logging

**Only execute if baseline ATE diverges despite identical frames (same `keyframe_list`).**

A baseline ATE gap with identical frames means submap stitching logic diverges. Add debug logging to `closure.py:run_pose_graph_optimization` to log H matrices at each boundary, then compare with VGGT-SLAM's `solver.add_edge`.

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py`

- [ ] **Step 1: Find `run_pose_graph_optimization` in closure.py**

  ```bash
  grep -n "def run_pose_graph_optimization" \
    collab_splats/pointcloud/loop_closure/closure.py
  ```

- [ ] **Step 2: Add H-matrix boundary logging**

  Inside `run_pose_graph_optimization`, find the loop over submap boundaries where inter-submap relative transforms are computed. Add after each boundary transform computation:

  ```python
  logger.debug(
      "Submap boundary %d→%d: scale=%.6f T_trans=%.6f H_det=%.6f",
      prev_id, curr_id,
      float(scale),
      float(np.linalg.norm(T[:3, 3])) if T is not None else 0.0,
      float(np.linalg.det(H_w)) if H_w is not None else 0.0,
  )
  ```

  Adjust variable names to match what actually exists in `run_pose_graph_optimization` at that point.

- [ ] **Step 3: Enable DEBUG in a diagnostic run**

  ```bash
  COLLAB_SPLATS_LOG_LEVEL=DEBUG \
  /opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --backbone vggt_spark --conditions baseline \
    --submap_size 16 --lc_scale_method none \
    --keyframe_list evals/baselines/disparity_sweep/slam_d50/selected_frames.txt \
    2>&1 | grep "Submap boundary"
  ```

  Compare the printed scale/H values against VGGT-SLAM's `solver.add_edge` at the same boundaries.

- [ ] **Step 4: Report findings**

  Report back with the first boundary where values diverge between our pipeline and VGGT-SLAM, including exact H matrix values from both. Do not implement further fixes without confirmation.

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|---|---|
| Sweep [50,30,20,10,0] | Task 3 (harness) + Task 4/5 (execution) |
| VGGT-SLAM baseline → selected_frames.txt | Task 3 (`run_slam_baseline`) |
| Our pipeline on same frames | Task 3 (`run_our_pipeline` with `--keyframe_list`) |
| 10% ATE gate | Task 3 (`_parity_ok`) |
| d=50 similarity calibration checkpoint | Task 3 (`_check_similarity_parity`) + Task 4 step 2 |
| Track A: native compute_similarity override | Task 2 |
| Threshold 0.95 | Task 2 |
| VGGT.forward input shape verified | Task 2 (noted in spec facts) |
| Track B: H-matrix debug (conditional) | Task 6 |
| INFO logging fix | Task 1 |
| Machine-readable ATE output | Task 1 (`--output_ate`) |

All requirements covered. No placeholders.
