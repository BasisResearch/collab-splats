# Feedforward Progress Logging — rich + tqdm Upgrade

**Date:** 2026-04-22
**Status:** Draft
**Branch:** feat/loop-closure

---

## Context

`feedforward.py` already has `print()` for pipeline stage progress (implemented). This design upgrades those prints to `rich.Console.log()` (adds auto-timestamps, color) and adds a `tqdm` progress bar for the loop closure submap loop, which is currently completely silent.

Both `rich` and `tqdm` are already in the nerfstudio dependency tree — no new packages needed.

---

## Architecture

**Single file changed:** `collab_splats/pointcloud/feedforward.py`

**New module-level additions:**
```python
import math
from rich.console import Console
from tqdm import tqdm

console = Console()
```

One `Console` instance at module level — simple, sufficient for a batch pipeline.

---

## Changes

### 1. Replace all `print()` with `console.log()`

All `print()` calls in `BaseFeedforwardCreator`, `VGGTXCreator`, `MapAnythingCreator`, and `_rescale_reconstruction_to_original_dimensions` become `console.log()`. This adds auto-timestamps to every line. The `end=" ", flush=True` pattern is dropped — `console.log()` always writes a complete line with timestamp.

Manual `time.perf_counter()` is kept where "done in Xs" timing is shown — the timestamp shows when a call started, the duration message shows how long it took.

Before/after example:
```python
# before
print(f"Loading model ({device})...", end=" ", flush=True)
self.model = self._load_model(device)
print(f"done in {time.perf_counter() - t0:.1f}s")

# after
console.log(f"Loading model ({device})...")
self.model = self._load_model(device)
console.log(f"  done in {time.perf_counter() - t0:.1f}s")
```

### 2. tqdm progress bar for loop closure submap loop

In `_run_loop_closure_inference()`, wrap the window loop with `tqdm`. Postfix shows live loop counts.

```python
n_submaps = math.ceil(max(1, N - O) / step)
loops_found = 0
verified = 0

with tqdm(total=n_submaps, desc="Loop closure", unit="submap") as pbar:
    for wi, start in enumerate(range(0, N, step)):
        ...
        # after verification check:
        if loop_verified:
            verified += 1
            loops_found += 1
        pbar.update(1)
        pbar.set_postfix(loops=loops_found, verified=verified)
```

### 3. Milestone prints for loop closure

Added to `_run_loop_closure_inference()` around the tqdm loop:

| Location | Output |
|----------|--------|
| Before loop | `Loop closure: {N} frames → {n_submaps} submaps (size={K}, overlap={O})` |
| Each verified loop | `  ↩ Loop: submap {q_id} → {d_id}  dist={dist:.3f}` |
| After GTSAM optimize | `  Pose graph: {total_frames} frames, {verified} loop edges → {t:.1f}s` |

Rejected candidates are not printed — only verified loops matter to the user.

### 4. `logging.warning()` calls unchanged

All existing `log.warning()` calls remain:
- DINO-SALAD load failure → fallback to single-pass
- GTSAM optimization failure → fallback to uncorrected poses
- LC submap key not found → skip edge

---

## Output Example

```
[10:23:01] Loading model (cuda)...
[10:23:05]   done in 4.1s
[10:23:05] Preprocessing images...
[10:23:06]   → 80 images  done in 0.8s
[10:23:06] Running inference...
[10:23:06] Loop closure: 80 frames → 4 submaps (size=20, overlap=4)
Loop closure  50%|████████          | 2/4 [01:02<01:02, loops=1, verified=1]
[10:24:10]   ↩ Loop: submap 3 → 0  dist=0.082
Loop closure 100%|████████████████  | 4/4 [02:08<00:00, loops=1, verified=1]
[10:25:14]   Pose graph: 80 frames, 1 loop edges → 0.3s
[10:25:14]   done in 128.4s
[10:25:14] Postprocessing...
[10:25:16]   → 142,331 pts  done in 1.8s
[10:25:16] Building COLMAP reconstruction...
[10:25:17]   done in 0.6s
```

---

## Verification

- Existing tests pass (no behavioral change — output format only)
- Manual: run `VGGTXCreator(enable_loop_closure=True)` on small image set, confirm tqdm renders in notebook and CLI, timestamps appear on all console.log lines
- Manual: run without `enable_loop_closure` — confirm no tqdm bar, all print lines still appear as console.log

---

## Out of Scope

- `rich.Progress` for major pipeline stages (load/preprocess/postprocess) — current pattern sufficient
- Semantics `BaseFeatureExtractor.forward()` / `score_queries()` — covered by separate existing spec
- Structured/JSON log output
