# Feedforward Progress Logging — rich + tqdm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all `print()` calls in `feedforward.py` with `rich.Console.log()` (adds auto-timestamps) and add a `tqdm` progress bar with milestone prints to the loop closure submap loop.

**Architecture:** Single file change (`collab_splats/pointcloud/feedforward.py`). One module-level `Console` instance. `tqdm` wraps the existing `range(0, N, step)` loop in `_run_loop_closure_inference()`. No new files, no behavioral changes to existing logic.

**Tech Stack:** `rich` (already installed via nerfstudio), `tqdm` (already installed via nerfstudio), Python `math` stdlib

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/pointcloud/feedforward.py` | Add `Console`/`tqdm`/`math` imports; replace all `print()` with `console.log()`; add tqdm + milestone prints in `_run_loop_closure_inference()` |

---

## Task 1: Add imports + replace print() with console.log()

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py`

Work in the git worktree at `/workspace/collab-splats/.worktrees/loop-closure/`.
All commands: `cd /workspace/collab-splats/.worktrees/loop-closure && ...`
Python: `/opt/conda/envs/nerfstudio/bin/python`

- [ ] **Step 1: Verify rich and tqdm are installed**

```bash
cd /workspace/collab-splats/.worktrees/loop-closure && \
  /opt/conda/envs/nerfstudio/bin/python -c "from rich.console import Console; from tqdm import tqdm; import math; print('OK')"
```
Expected: `OK`

- [ ] **Step 2: Run existing tests to establish baseline**

```bash
cd /workspace/collab-splats/.worktrees/loop-closure && \
  /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_shared.py tests/pointcloud/test_loop_closure.py tests/pointcloud/test_pose_graph.py -v 2>&1 | tail -15
```
Expected: all pass.

- [ ] **Step 3: Add module-level imports to feedforward.py**

At the top of `collab_splats/pointcloud/feedforward.py`, after the existing `import numpy as np` line and before the `from .base import` line, add:

```python
import math

from rich.console import Console
from tqdm import tqdm

console = Console()
```

- [ ] **Step 4: Replace print() calls in load_model()**

Find:
```python
        print(f"Loading model ({device})...", end=" ", flush=True)
        t0 = time.perf_counter()
        self.model = self._load_model(device)
        print(f"done in {time.perf_counter() - t0:.1f}s")
```

Replace with:
```python
        t0 = time.perf_counter()
        console.log(f"Loading model ({device})...")
        self.model = self._load_model(device)
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")
```

- [ ] **Step 5: Replace print() calls in setup_inference()**

Find:
```python
        print("Preprocessing images...", end=" ", flush=True)
        t0 = time.perf_counter()
        self.views, self.image_paths, self.original_coords = self._preprocess(Path(image_dir))
        print(f"→ {len(self.image_paths)} images  done in {time.perf_counter() - t0:.1f}s")
```

Replace with:
```python
        t0 = time.perf_counter()
        console.log("Preprocessing images...")
        self.views, self.image_paths, self.original_coords = self._preprocess(Path(image_dir))
        console.log(f"  → {len(self.image_paths)} images  done in {time.perf_counter() - t0:.1f}s")
```

- [ ] **Step 6: Replace print() calls in run_inference()**

Find:
```python
        print("Running inference...")
        t0 = time.perf_counter()
```

Replace with:
```python
        t0 = time.perf_counter()
        console.log("Running inference...")
```

Find (the closing print of run_inference — immediately after the if/else block):
```python
        print(f"done in {time.perf_counter() - t0:.1f}s")
```

Replace with:
```python
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")
```

- [ ] **Step 7: Replace print() calls in postprocess()**

Find:
```python
        print("Postprocessing...", end=" ", flush=True)
        t0 = time.perf_counter()
        self.outputs = self._postprocess(self.raw_outputs, **kwargs)
        n_pts = len(self.outputs.pts3d)
        print(f"→ {n_pts:,} pts  done in {time.perf_counter() - t0:.1f}s")
```

Replace with:
```python
        t0 = time.perf_counter()
        console.log("Postprocessing...")
        self.outputs = self._postprocess(self.raw_outputs, **kwargs)
        n_pts = len(self.outputs.pts3d)
        console.log(f"  → {n_pts:,} pts  done in {time.perf_counter() - t0:.1f}s")
```

- [ ] **Step 8: Replace print() calls in build_colmap()**

Find:
```python
        print("Building COLMAP reconstruction...", end=" ", flush=True)
        t0 = time.perf_counter()
```

Replace with:
```python
        t0 = time.perf_counter()
        console.log("Building COLMAP reconstruction...")
```

Find (closing print of build_colmap):
```python
        print(f"done in {time.perf_counter() - t0:.1f}s")
        return colmap_reconstruction_to_result(recon)
```

Replace with:
```python
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")
        return colmap_reconstruction_to_result(recon)
```

- [ ] **Step 9: Replace print() calls in _rescale_reconstruction_to_original_dimensions()**

Find:
```python
        print(
            f"Rescaling reconstruction from {image_size[0]}x{image_size[1]} "
            f"to original dimensions"
        )
        print(f"  Original image sizes (WxH): {int(original_width)}x{int(original_height)}")
```

Replace with:
```python
        console.log(
            f"Rescaling reconstruction from {image_size[0]}x{image_size[1]} "
            f"to original dimensions"
        )
        console.log(f"  Original image sizes (WxH): {int(original_width)}x{int(original_height)}")
```

Find:
```python
        print("Rescaled reconstruction to original dimensions")
```

Replace with:
```python
        console.log("Rescaled reconstruction to original dimensions")
```

- [ ] **Step 10: Replace print() in MapAnythingCreator._forward()**

Find:
```python
        print(f"  → {len(views)} images, minibatch_size={self.minibatch_size}")
```

Replace with:
```python
        console.log(f"  → {len(views)} images, minibatch_size={self.minibatch_size}")
```

- [ ] **Step 11: Confirm no print() calls remain**

```bash
cd /workspace/collab-splats/.worktrees/loop-closure && \
  grep -n 'print(' collab_splats/pointcloud/feedforward.py
```
Expected: no output (zero matches).

- [ ] **Step 12: Run tests**

```bash
cd /workspace/collab-splats/.worktrees/loop-closure && \
  /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_shared.py tests/pointcloud/test_loop_closure.py tests/pointcloud/test_pose_graph.py -v 2>&1 | tail -15
```
Expected: all pass.

- [ ] **Step 13: Commit**

```bash
cd /workspace/collab-splats/.worktrees/loop-closure && \
  git add collab_splats/pointcloud/feedforward.py && \
  git commit -m "refactor(pointcloud): replace print() with rich Console.log() in feedforward pipeline"
```

---

## Task 2: Add tqdm progress bar + milestone prints to _run_loop_closure_inference()

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py`

Work in `/workspace/collab-splats/.worktrees/loop-closure/`.

- [ ] **Step 1: Run baseline**

```bash
cd /workspace/collab-splats/.worktrees/loop-closure && \
  /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_loop_closure_integration.py tests/pointcloud/test_feedforward_shared.py -v 2>&1 | tail -10
```
Expected: all pass.

- [ ] **Step 2: Add n_submaps + counters + milestone print before the loop**

In `_run_loop_closure_inference()`, find:
```python
    submaps: list[Submap] = []
    lc_submaps: list[Submap] = []

    for wi, start in enumerate(range(0, N, step)):
```

Replace with:
```python
    submaps: list[Submap] = []
    lc_submaps: list[Submap] = []
    n_submaps = math.ceil(max(1, N - O) / step)
    loops_found = 0
    verified = 0

    console.log(f"Loop closure: {N} frames → {n_submaps} submaps (size={K}, overlap={O})")

    with tqdm(total=n_submaps, desc="Loop closure", unit="submap") as pbar:
        for wi, start in enumerate(range(0, N, step)):
```

**CRITICAL:** Every line of the existing `for` loop body gains one level of indentation (4 spaces) since it now lives inside the `with tqdm(...) as pbar:` block. The `if end >= N: break` at the end of the loop body stays inside the `for`, which is inside the `with`.

- [ ] **Step 3: Add console.log for verified loops inside the loop body**

Inside the loop body, find the `if self._verify_loop_candidate(...)` block:
```python
                if self._verify_loop_candidate(q_frame, d_frame):
                    lc_submaps.append(Submap(
```

Replace with:
```python
                if self._verify_loop_candidate(q_frame, d_frame):
                    verified += 1
                    loops_found += 1
                    console.log(
                        f"  ↩ Loop: submap {match.query_submap_id} → {match.detected_submap_id}"
                        f"  dist={match.similarity_score:.3f}"
                    )
                    lc_submaps.append(Submap(
```

- [ ] **Step 4: Add pbar.update() + pbar.set_postfix() at end of loop body**

Find (inside the loop body, after `submaps.append(submap)`):
```python
        submaps.append(submap)
        if end >= N:
            break
```

Replace with:
```python
        submaps.append(submap)
        pbar.update(1)
        pbar.set_postfix(loops=loops_found, verified=verified)
        if end >= N:
            break
```

- [ ] **Step 5: Add Pose graph timing milestone after the tqdm block**

Find (after the `with tqdm` block ends):
```python
    corrected_extrinsics = self._loop_close(submaps, lc_submaps)  # (N, 4, 4)
    self.raw_outputs = self._merge_submap_outputs(submaps, corrected_extrinsics)
```

Replace with:
```python
    t0_pg = time.perf_counter()
    corrected_extrinsics = self._loop_close(submaps, lc_submaps)  # (N, 4, 4)
    console.log(f"  Pose graph: {N} frames, {len(lc_submaps)} loop edges → {time.perf_counter() - t0_pg:.1f}s")
    self.raw_outputs = self._merge_submap_outputs(submaps, corrected_extrinsics)
```

- [ ] **Step 6: Run full pointcloud + semantics test suite**

```bash
cd /workspace/collab-splats/.worktrees/loop-closure && \
  /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ tests/semantics/ -q 2>&1 | tail -10
```
Expected: same results as before this task (4 pre-existing failures in test_mapanything and test_query_api; all new loop closure + feedforward tests pass).

- [ ] **Step 7: Commit**

```bash
cd /workspace/collab-splats/.worktrees/loop-closure && \
  git add collab_splats/pointcloud/feedforward.py && \
  git commit -m "feat(pointcloud): add tqdm progress bar and milestone logs to loop closure inference"
```

---

## Verification Checklist

- [ ] `grep -n 'print(' collab_splats/pointcloud/feedforward.py` → zero matches
- [ ] All existing tests pass (no behavioral change)
- [ ] `console` is module-level — instantiated once at import time
- [ ] tqdm bar appears when `enable_loop_closure=True` and `N >= submap_size`
- [ ] No tqdm bar when `enable_loop_closure=False` (default path unchanged)
