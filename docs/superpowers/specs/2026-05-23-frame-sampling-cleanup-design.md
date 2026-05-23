# frame_sampling cleanup — design spec

**Date:** 2026-05-23  
**Branch:** refactor/cu121  
**Scope:** `collab_splats/utils/frame_sampling.py`, `keyframe_extraction.ipynb`, all callers

---

## Problem

`frame_sampling.py` and its tutorial notebook violate several project code-style principles:

| Location | Violation |
|---|---|
| `save_frame_scores` / `load_frame_scores` | `import json` + `from pathlib import Path` inline — both already at module top |
| `sample_frames_fps` | Returns `list[np.ndarray]`; callers must manually recompute stride indices, duplicating internal logic |
| `OpticalFlowFrameSelector._compute_normalized_entropy` | Defined, never called, no tests — dead code |
| `frame_sampling.py` | No `########`-style section dividers; `OpticalFlowFrameSelector` defined 500 lines after the functions that use it |
| `frame_sampling.py` | Several blocks missing inline comments |
| `keyframe_extraction.ipynb` Cell 1 | Imports `cv2`, `json`, `Path`, `matplotlib.pyplot`, `%matplotlib inline` — low-level internals the notebook never uses directly |
| `keyframe_extraction.ipynb` Cell 1 | Imports `extract_video_frames`, `load_video_frames` — never used in notebook |
| `keyframe_extraction.ipynb` Cell 6 | 3-line manual `fps_indices` computation duplicates stride logic inside `sample_frames_fps` |

---

## Design

### 1. `frame_sampling.py` — section restructuring

Add `########`-style dividers and reorder into logical sections. `OpticalFlowFrameSelector` moves before the functions that use it:

```
########################################################################
# Helpers
########################################################################
_get_rotation_degrees, _apply_rotation, get_video_info

########################################################################
# OpticalFlowFrameSelector
########################################################################
class OpticalFlowFrameSelector

########################################################################
# Frame Selection
########################################################################
score_all_frames, sample_frames_fps, sample_frames_optical_flow

########################################################################
# Frame I/O
########################################################################
load_video_frames, extract_video_frames

########################################################################
# Score I/O
########################################################################
save_frame_scores, load_frame_scores

########################################################################
# Visualization
########################################################################
plot_frame_grid, plot_selection, plot_frame_scores, plot_disparity_sensitivity
```

### 2. `frame_sampling.py` — targeted fixes

**Fix: inline imports in Score I/O** — remove redundant `import json` and `from pathlib import Path` inside `save_frame_scores` and `load_frame_scores`; both are already at module top.

**Fix: dead method** — delete `OpticalFlowFrameSelector._compute_normalized_entropy`. Defined at line 380, never called, no tests.

**Fix: `sample_frames_fps` return type** — change from `list[np.ndarray]` to `tuple[list[np.ndarray], list[int]]`. The indices are the stride-computed frame positions already computed inside the function; surfacing them eliminates the leaky abstraction in callers.

```python
# before
def sample_frames_fps(...) -> list[np.ndarray]:
    ...
    return frames

# after
def sample_frames_fps(...) -> tuple[list[np.ndarray], list[int]]:
    ...
    indices = list(range(0, total, interval))[:len(frames)]
    return frames, indices
```

**Fix: inline block comments** — add block-level comments to any logical block in the file that lacks one, following the `# Sort images by filename; reject non-image extensions` style. No line-by-line comments.

### 3. `keyframe_extraction.ipynb` — import cleanup

Remove from Cell 1:
- `import cv2` — internal to `frame_sampling.py`
- `import json` — internal to `frame_sampling.py`
- `from pathlib import Path` — `CACHE_DIR` already constructed via `Path(...)` but `Path` is re-exported from `frame_sampling` via the cache helpers; notebook can keep it only if used directly
- `import matplotlib.pyplot as plt` + `%matplotlib inline` — `plot_*` functions call `plt.show()` internally; notebook never calls `plt` directly
- `extract_video_frames` — imported, never used in notebook
- `load_video_frames` — imported, never used in notebook

Keep: `get_video_info`, `sample_frames_fps`, `sample_frames_optical_flow`, `score_all_frames`, `save_frame_scores`, `load_frame_scores`, `plot_frame_grid`, `plot_selection`, `plot_frame_scores`, `plot_disparity_sensitivity`.

Note: `Path` is used directly in the notebook for `CACHE_DIR = Path("../.cache/c0043")`. Keep `from pathlib import Path`.

### 4. `keyframe_extraction.ipynb` — fps_indices fix

Replace the manual stride computation with tuple unpack:

```python
# before (Cell 6 + Cell 7):
fps_frames = sample_frames_fps(video_path, fps=target_fps)
interval = max(1, round(info['fps'] / target_fps))
fps_indices = list(range(0, info['total_frames'], interval))[:len(fps_frames)]

# after (single cell):
fps_frames, fps_indices = sample_frames_fps(video_path, fps=target_fps)
```

Same pattern for the preview call (Cell 4, `max_frames=12`):
```python
preview_frames, _ = sample_frames_fps(video_path, fps=1.0, max_frames=12)
```

### 5. Caller updates

All callers updated to unpack the new return type:

| File | Change |
|---|---|
| `collab_splats/dashboard/semantics.py:523` | `frames, _ = sample_frames_fps(...)` |
| `tests/utils/test_frame_sampling.py` | Unpack tuple in all test bodies; update assertions |
| `tests/dashboard/test_semantics_smoke.py` | Unpack empty-path result |
| `evals/datasets.py:254` | Fix pre-existing broken call: `sample_frames_fps(str(seq_dir), frames_dir, fps=fps)` passes a directory as video path and an unknown `frames_dir` as the second positional. During implementation: read `datasets.py` context to determine correct video path and whether `frames_dir` was intended as `output_dir` for `extract_video_frames`. Fix call and unpack return. |

---

## Files changed

| File | Change type |
|---|---|
| `collab_splats/utils/frame_sampling.py` | Reorder, section dividers, inline comments, dead method removed, return type changed, inline imports removed |
| `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` | Import cleanup, fps_indices fix, re-execute |
| `collab_splats/dashboard/semantics.py` | Caller update |
| `evals/datasets.py` | Caller fix |
| `tests/utils/test_frame_sampling.py` | Test update for new return type |
| `tests/dashboard/test_semantics_smoke.py` | Test update for new return type |

---

## Out of scope

- Moving `plot_*` functions to `visualization.py` — kept co-located with selection logic (deliberate)
- Moving `save_frame_scores`/`load_frame_scores` to a separate I/O module — not warranted at current usage level
- Any changes to `sample_frames_optical_flow` return type
