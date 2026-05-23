# Tutorial Config Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a single `tutorial_config.py` at the tutorials root and update all 15 notebooks to source it, so `DATASET`, `MAX_FRAMES`, `CACHE_DIR`, and `IMAGES` are defined in one place.

**Architecture:** A Python script at `docs/source/tutorials/tutorial_config.py` is executed via `%run ../tutorial_config.py` in each notebook's config cell, injecting shared constants into the notebook namespace. A programmatic patch script handles all notebooks in one pass. Two notebooks get additional changes: `keyframe_extraction` gains `max_frames=MAX_FRAMES`, and `slam_loop_closure` switches from a hardcoded `SCENE_DIR` to the shared `IMAGES`.

**Tech Stack:** Python, Jupyter notebook JSON patching, `/opt/conda/envs/nerfstudio/bin/python`

---

## Files Changed

| Action | Path |
|---|---|
| **Create** | `docs/source/tutorials/tutorial_config.py` |
| **Create** | `tools/patch_tutorial_configs.py` (patch script, run once, can be deleted after) |
| **Modify** | All 15 notebooks under `docs/source/tutorials/` |

---

## Task 1: Create tutorial_config.py

**Files:**
- Create: `docs/source/tutorials/tutorial_config.py`

- [ ] **Step 1: Write the config script**

```python
# docs/source/tutorials/tutorial_config.py
from pathlib import Path

from collab_splats.utils.paths import get_cache_dir

DATASET = "birds_c0043"
MAX_FRAMES = 30  # cap for light tutorial runs

CACHE_DIR = get_cache_dir(DATASET)
IMAGES = CACHE_DIR / "images"
```

- [ ] **Step 2: Smoke-test it executes cleanly**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/python docs/source/tutorials/tutorial_config.py
```
Expected: no output, no errors.

- [ ] **Step 3: Verify values are correct**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/python -c "
exec(open('docs/source/tutorials/tutorial_config.py').read())
assert DATASET == 'birds_c0043', f'got {DATASET}'
assert MAX_FRAMES == 30, f'got {MAX_FRAMES}'
assert CACHE_DIR.name == 'birds_c0043', f'got {CACHE_DIR}'
assert IMAGES == CACHE_DIR / 'images', f'got {IMAGES}'
print('OK — DATASET={}, MAX_FRAMES={}, CACHE_DIR={}'.format(DATASET, MAX_FRAMES, CACHE_DIR))
"
```
Expected: `OK — DATASET=birds_c0043, MAX_FRAMES=30, CACHE_DIR=<path>`

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/tutorial_config.py
git commit -m "feat(tutorials): add shared tutorial_config.py with DATASET, MAX_FRAMES, CACHE_DIR, IMAGES"
```

---

## Task 2: Write the notebook patch script

**Files:**
- Create: `tools/patch_tutorial_configs.py`

This script patches all 15 notebooks in one pass. It finds the config cell in each notebook (the cell containing `DATASET =` and `get_cache_dir`), strips `DATASET`/`CACHE_DIR`/`IMAGES` lines, and prepends `%run ../tutorial_config.py`. It also removes the now-unused `get_cache_dir` import from imports cells.

- [ ] **Step 1: Create the patch script**

```python
# tools/patch_tutorial_configs.py
"""
Patch all tutorial notebooks to source tutorial_config.py.

Run from the repo root:
    /opt/conda/envs/nerfstudio/bin/python tools/patch_tutorial_configs.py
"""
import json
import re
from pathlib import Path

TUTORIALS_ROOT = Path("docs/source/tutorials")

NOTEBOOKS = [
    "01_preprocessing/keyframe_extraction.ipynb",
    "02_pointcloud/bundle_adjustment.ipynb",
    "02_pointcloud/colmap_sfm.ipynb",
    "02_pointcloud/feedforward_mesh.ipynb",
    "02_pointcloud/feedforward_methods.ipynb",
    "02_pointcloud/slam_loop_closure.ipynb",
    "03_splats/derive_splats.ipynb",
    "03_splats/visualization.ipynb",
    "04_semantics/feature_extraction.ipynb",
    "04_semantics/maskclip_vs_talk2dino.ipynb",
    "04_semantics/segmentation.ipynb",
    "05_lifting/semantic_lifting.ipynb",
    "06_mesh/create_mesh.ipynb",
    "07_localization/localization.ipynb",
    "evals/ground_truth_evals.ipynb",
]

# Lines to strip from config cells (matched as stripped prefixes)
STRIP_PREFIXES = (
    "DATASET",
    "CACHE_DIR",
    "IMAGES",
)

RUN_LINE = "%run ../tutorial_config.py"


def cell_source(cell: dict) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def set_cell_source(cell: dict, src: str) -> None:
    if isinstance(cell["source"], list):
        cell["source"] = src.splitlines(keepends=True)
    else:
        cell["source"] = src


def is_config_cell(src: str) -> bool:
    # Match any cell with a DATASET = assignment (covers notebooks that don't use get_cache_dir)
    return bool(re.search(r"^DATASET\s*=", src, re.MULTILINE))


def is_imports_cell(src: str) -> bool:
    return "get_cache_dir" in src and "import" in src


def patch_config_cell(src: str) -> str:
    lines = src.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        # Drop DATASET/CACHE_DIR/IMAGES assignments and blank lines they leave behind
        if any(stripped.startswith(p) for p in STRIP_PREFIXES):
            continue
        kept.append(line)
    # Remove leading/trailing blank lines
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    # Prepend %run line
    result_lines = [RUN_LINE] + ([""] if kept else []) + kept
    return "\n".join(result_lines)


def patch_imports_cell(src: str) -> str:
    """Remove 'from collab_splats.utils.paths import get_cache_dir' if present."""
    lines = src.splitlines()
    kept = [l for l in lines if "get_cache_dir" not in l]
    # Remove blank lines left at end
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def patch_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text())
    changed = False
    config_patched = False
    imports_patched = False

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)

        if not config_patched and is_config_cell(src):
            new_src = patch_config_cell(src)
            if new_src != src:
                set_cell_source(cell, new_src)
                changed = True
            config_patched = True
            continue

        if not imports_patched and is_imports_cell(src):
            new_src = patch_imports_cell(src)
            if new_src != src:
                set_cell_source(cell, new_src)
                changed = True
            imports_patched = True

    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"  patched: {path.relative_to(Path('.'))}")
    else:
        print(f"  skip (no change): {path.relative_to(Path('.'))}")
    return changed


def main():
    n_changed = 0
    for rel in NOTEBOOKS:
        path = TUTORIALS_ROOT / rel
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        if patch_notebook(path):
            n_changed += 1
    print(f"\nDone. {n_changed}/{len(NOTEBOOKS)} notebooks patched.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit the patch script**

```bash
git add tools/patch_tutorial_configs.py
git commit -m "chore(tools): add notebook config patch script"
```

---

## Task 3: Run the patch script and verify

- [ ] **Step 1: Run the patch**

```bash
/opt/conda/envs/nerfstudio/bin/python tools/patch_tutorial_configs.py
```
Expected output: 15 lines, each showing `patched: ...` or `skip (no change): ...`. All 15 should show `patched`.

- [ ] **Step 2: Spot-check feedforward_methods.ipynb**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
from pathlib import Path
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
for c in nb['cells']:
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    if '%run' in src:
        print('CONFIG CELL:')
        print(src[:300])
        break
"
```
Expected: config cell starts with `%run ../tutorial_config.py`, no `DATASET =` or `CACHE_DIR =` lines.

- [ ] **Step 3: Spot-check that get_cache_dir import was removed**

```bash
grep -l 'get_cache_dir' docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb && echo "STILL PRESENT" || echo "REMOVED OK"
```
Expected: `REMOVED OK`

- [ ] **Step 4: Commit patched notebooks**

```bash
git add docs/source/tutorials/
git commit -m "refactor(tutorials): source tutorial_config.py in all notebooks — remove per-notebook DATASET/CACHE_DIR/IMAGES"
```

---

## Task 4: Handle keyframe_extraction special case

`keyframe_extraction.ipynb` needs `max_frames=MAX_FRAMES` passed to the `sample_frames_optical_flow` call so extraction is capped at 30 frames.

**Files:**
- Modify: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`

- [ ] **Step 1: Find the OF sampling cell**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb'))
for i, c in enumerate(nb['cells']):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    if 'sample_frames_optical_flow' in src and 'VIDEO_PATH' in src:
        print(f'Cell {i}:')
        print(src)
"
```

- [ ] **Step 2: Edit the cell**

Use NotebookEdit to replace the `sample_frames_optical_flow` call. The existing call looks like:
```python
frame_scores = score_all_frames(VIDEO_PATH)
```
or:
```python
of_frames = sample_frames_optical_flow(VIDEO_PATH)
```

Replace the `sample_frames_optical_flow(VIDEO_PATH)` call (in whichever cell it appears) to add `max_frames=MAX_FRAMES`:

**Before:**
```python
frame_scores = score_all_frames(VIDEO_PATH)
```
*(or whatever the exact call is — read the cell first)*

**After** (add `max_frames=MAX_FRAMES` to `score_all_frames` or `sample_frames_optical_flow` — whichever caps frame extraction):

The key function to cap is `score_all_frames` (which drives OF selection). Find the call and add the kwarg:
```python
frame_scores = score_all_frames(VIDEO_PATH, max_frames=MAX_FRAMES)
```

> **Note:** Read the actual cell content from Step 1 before editing. The exact function name and call site may differ from this template. `MAX_FRAMES` is injected by `%run ../tutorial_config.py` so it will be in scope.

- [ ] **Step 3: Verify max_frames appears in the notebook**

```bash
grep -c 'max_frames' docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
```
Expected: at least `1`

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
git commit -m "feat(tutorials): cap keyframe extraction at MAX_FRAMES=30 for light tutorial runs"
```

---

## Task 5: Handle slam_loop_closure special case

`slam_loop_closure.ipynb` uses `SCENE_DIR = Path("/workspace/bicycle/images_4")` as its image source. Replace with `IMAGES` from the shared config.

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`

- [ ] **Step 1: Find the SCENE_DIR config cell**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb'))
for i, c in enumerate(nb['cells']):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    if 'SCENE_DIR' in src:
        print(f'Cell {i}:')
        print(src)
        print()
"
```

- [ ] **Step 2: Remove SCENE_DIR and OUTPUT_DIR (bicycle-specific) lines**

Use NotebookEdit on the cell containing `SCENE_DIR`. The cell currently looks like:
```python
# ── Configuration ──────────────────────────────────────────────────────────────
DATASET    = "birds_c0043"
CACHE_DIR  = get_cache_dir(DATASET)

BACKEND    = "vggtx"          # "vggtx" | "mapanything"
SCENE_DIR  = Path("/workspace/bicycle/images_4")
OUTPUT_DIR = Path("/workspace/bicycle/lc_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

After Task 3's generic patch, `DATASET`/`CACHE_DIR` are already gone and `%run ../tutorial_config.py` is prepended. So the remaining cell will look like:
```python
%run ../tutorial_config.py

BACKEND    = "vggtx"          # "vggtx" | "mapanything"
SCENE_DIR  = Path("/workspace/bicycle/images_4")
OUTPUT_DIR = Path("/workspace/bicycle/lc_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

Replace `SCENE_DIR` line with `# SCENE_DIR = IMAGES  (set by tutorial_config.py)` comment or simply remove it — `IMAGES` is already in scope from `%run`. Also replace `OUTPUT_DIR` with a CACHE_DIR-relative path:

Target cell content:
```python
%run ../tutorial_config.py

BACKEND    = "vggtx"          # "vggtx" | "mapanything"
OUTPUT_DIR = CACHE_DIR / "lc_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

All downstream references to `SCENE_DIR` in the notebook must be replaced with `IMAGES`. Find them:

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb'))
for i, c in enumerate(nb['cells']):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    if 'SCENE_DIR' in src:
        print(f'Cell {i}: {src[:200]}')
"
```

For each cell that uses `SCENE_DIR`, replace `SCENE_DIR` → `IMAGES` using NotebookEdit.

- [ ] **Step 3: Verify no SCENE_DIR references remain**

```bash
grep -c 'SCENE_DIR\|bicycle' docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
```
Expected: `0`

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
git commit -m "refactor(tutorials): slam_loop_closure use shared IMAGES dir instead of hardcoded bicycle path"
```

---

## Task 6: Final verification across all notebooks

- [ ] **Step 1: Verify every notebook has %run line**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
from pathlib import Path

notebooks = list(Path('docs/source/tutorials').rglob('*.ipynb'))
missing = []
for nb_path in sorted(notebooks):
    nb = json.load(open(nb_path))
    has_run = any(
        '%run ../tutorial_config.py' in (
            c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        )
        for c in nb['cells'] if c['cell_type'] == 'code'
    )
    if not has_run:
        missing.append(nb_path.relative_to('docs/source/tutorials'))
        
if missing:
    print('MISSING %run in:')
    for m in missing:
        print(f'  {m}')
else:
    print(f'OK — all {len(notebooks)} notebooks have %run ../tutorial_config.py')
"
```
Expected: `OK — all 15 notebooks have %run ../tutorial_config.py`

- [ ] **Step 2: Verify no notebook has a bare DATASET= assignment**

```bash
grep -rl 'DATASET\s*=\s*"birds_c0043"' docs/source/tutorials/ && echo "FOUND — needs cleanup" || echo "OK — no bare DATASET assignments"
```
Expected: `OK — no bare DATASET assignments`

- [ ] **Step 3: Verify MAX_FRAMES appears in keyframe_extraction**

```bash
grep -c 'max_frames\|MAX_FRAMES' docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
```
Expected: at least `2` (once in config from `%run`, once in the OF call)

- [ ] **Step 4: Final commit**

```bash
git add -p  # review any remaining unstaged changes
git commit -m "chore(tutorials): final cleanup — unified tutorial config across all notebooks"
```
