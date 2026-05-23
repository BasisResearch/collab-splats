# Inline Documentation Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply CLAUDE.md style rules (`########` section dividers, inline block comments, `logging` not `print()`) to four under-documented core modules.

**Architecture:** Pure documentation and logging changes — no new functions, no logic changes. Existing tests must pass at each step. File-by-file commits in priority order.

**Tech Stack:** Python standard `logging` module. No new dependencies.

---

### Task 0: Set up worktree

**Files:** None (setup only)

- [ ] **Step 1: Create worktree on refactor/core-modules**

```bash
git worktree add /workspace/collab-splats-docs-wt refactor/core-modules
```

- [ ] **Step 2: Verify target files exist**

```bash
ls /workspace/collab-splats-docs-wt/collab_splats/pointcloud/utils.py \
   /workspace/collab-splats-docs-wt/collab_splats/pointcloud/wrappers.py \
   /workspace/collab-splats-docs-wt/collab_splats/pointcloud/bundle_adjustment.py \
   /workspace/collab-splats-docs-wt/collab_splats/mesh/utils.py
```

Expected: all 4 listed without error.

All remaining tasks work inside `/workspace/collab-splats-docs-wt/`.

---

### Task 1: `pointcloud/utils.py` — section dividers

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`

The file already has `import logging` and `logger = logging.getLogger(__name__)` — no import changes needed.

- [ ] **Step 1: Confirm function layout**

```bash
grep -n '^def \|^class ' collab_splats/pointcloud/utils.py
```

Note line numbers for: `colmap_reconstruction_to_result`, `_radial_mask`, `_bbox_mask`, `filter_distance`, `filter_density`, `clean_pointcloud`, `voxel_downsample`, `clean_pcd`, `remove_far_points`, `density_filter`, plus any `lift_features` / `reproject_pixels`:

```bash
grep -n 'def lift_features\|def reproject_pixels' collab_splats/pointcloud/utils.py
```

- [ ] **Step 2: Add section dividers**

After the constants block (the `_UNSET = object()` line), add:

```python
########################################################
########## COLMAP → result conversion ##################
########################################################
```

Before `def _radial_mask`, add:

```python
########################################################
########## Confidence masking helpers ##################
########################################################
```

Before `def filter_distance`, add:

```python
########################################################
########## Distance and density filters ################
########################################################
```

Before `def clean_pointcloud`, add:

```python
########################################################
########## Primary cleaning pipeline ##################
########################################################
```

If `lift_features` / `reproject_pixels` exist (found in Step 1), add before the first one:

```python
########################################################
########## Feature projection ##########################
########################################################
```

Before `def clean_pcd`, add:

```python
########################################################
########## Legacy cleaning utilities ##################
########################################################
```

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -x -q 2>&1 | tail -8
```

Expected: pass (4 pre-existing failures are OK — check they haven't increased).

---

### Task 2: `pointcloud/utils.py` — print() → logging

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`

- [ ] **Step 1: List all remaining print() calls**

```bash
grep -n 'print(' collab_splats/pointcloud/utils.py
```

- [ ] **Step 2: Replace each print() call**

Apply the rule: size/count diagnostics → `logger.debug`, pipeline progress → `logger.info`.

Specific replacements for the known calls:

```python
# voxel_downsample — adaptive size chosen
# OLD: print(f"Using adaptive voxel size: {adaptive_voxel_size}")
logger.debug("voxel_downsample: adaptive size %.4f", adaptive_voxel_size)

# clean_pcd — statistical outlier removal result
# OLD: print(f"Removed {len(indices) - len(ind)} statistical outliers")
logger.debug("clean_pcd: removed %d statistical outliers", len(indices) - len(ind))

# clean_pcd — final point count
# OLD: print(f"Point cloud has {len(indices)} points after enhanced cleaning")
logger.debug("clean_pcd: %d points after cleaning", len(indices))

# density_filter — phase header
# OLD: print("Finding sparse regions...")
logger.debug("density_filter: estimating densities")

# density_filter — density stats
# OLD: print(f"Density stats - Min: {np.min(densities)}, Max: {np.max(densities)}, Mean: {np.mean(densities):.1f}")
logger.debug("density_filter: min=%d max=%d mean=%.1f", np.min(densities), np.max(densities), np.mean(densities))

# density_filter — removed count
# OLD: print(f"Removed {len(pcd.points) - len(pcd_dense.points)} sparse points (threshold: {density_threshold})")
logger.debug("density_filter: removed %d sparse points (threshold %.3f)", len(pcd.points) - len(pcd_dense.points), density_threshold)
```

For any additional print() calls found in Step 1 (e.g. in `filter_points_by_spatial_extent`), apply the same rule: reporting sizes/stats → `logger.debug`, structural steps → `logger.info`.

- [ ] **Step 3: Verify no print() remain**

```bash
grep -c 'print(' collab_splats/pointcloud/utils.py
```

Expected: `0`.

- [ ] **Step 4: Check inline block comments in multi-step functions**

`clean_pcd` has numbered step comments (`# 1.`, `# 2.` etc.) — verify steps 1–5 are all present:

```bash
grep -n '# [0-9]\.' collab_splats/pointcloud/utils.py
```

`voxel_downsample` — check it has block comments before its adaptive-sizing logic and voxel-trace call. If not, add:

```python
# Compute adaptive voxel size scaled to point density
...
# Voxel downsample with trace to preserve original point indices
```

`density_filter` — verify `# Find points...` and `# Remove points...` comments are present.

- [ ] **Step 5: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -x -q 2>&1 | tail -8
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git -C /workspace/collab-splats-docs-wt add collab_splats/pointcloud/utils.py
git -C /workspace/collab-splats-docs-wt commit -m "docs(pointcloud): section dividers + print→logging in utils.py"
```

---

### Task 3: `pointcloud/wrappers.py` — divider style + block comments

**Files:**
- Modify: `collab_splats/pointcloud/wrappers.py`

- [ ] **Step 1: List current dash dividers**

```bash
grep -n '# --' collab_splats/pointcloud/wrappers.py
```

Expected output shows 4 lines inside `LoopClosure` using `# ------------------------------------------------------------------` style.

- [ ] **Step 2: Replace dash dividers with `########` style**

Replace the delegation block header (two `# --` lines surrounding "Delegation — proxy..."):

```python
########################################################
########## Delegation — proxy to self.base #############
########################################################
```

Replace the inference block header (two `# --` lines surrounding "Inference — override with LC loop"):

```python
########################################################
########## Inference — LC loop override ################
########################################################
```

- [ ] **Step 3: Add class-level divider between BundleAdjustment and LoopClosure**

Find the blank lines between the last method of `BundleAdjustment` and `class LoopClosure:`. Insert before `class LoopClosure`:

```python
########################################################
########## LoopClosure wrapper ########################
########################################################
```

- [ ] **Step 4: Add block comments to `_apply_ba`**

`_apply_ba` runs three phases. Locate the `extract_tracks_vggsfm` call and `run_bundle_adjustment` call. Add a block comment before each:

```python
# Extract 2D tracks across frames — VGGSfM tracker requires square input (padded in extract_tracks_vggsfm)
tracks, vis_scores, pts3d_kp = extract_tracks_vggsfm(...)

# Run Levenberg-Marquardt BA to refine poses and 3D points
_, refined_ext_3x4, refined_intr = run_bundle_adjustment(...)

# Write refined poses and intrinsics back into the feedforward result
```

- [ ] **Step 5: Add block comments to `_run_lc_loop`**

`_run_lc_loop` has these phases — add a block comment before each:

```python
# Load DINO-SALAD retrieval model; fall back to full-sequence inference if unavailable
retrieval = ImageRetrieval(device=device)

# Slide submap window across frames (stride = submap_size - overlap)
for start in range(0, N - K + 1, step):

# Run feedforward inference on this submap's frames
self.base.run_inference(frame_indices=...)

# Query retrieval index for loop candidates against prior submaps
loop_candidates = retrieval.search(...)

# Merge per-submap world_points and poses into unified outputs
```

(Insert the comment at the actual location; the exact call names may differ — read the function before editing.)

- [ ] **Step 6: Verify no # -- style dividers remain**

```bash
grep -c '# --' collab_splats/pointcloud/wrappers.py
```

Expected: `0`.

- [ ] **Step 7: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -x -q 2>&1 | tail -8
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git -C /workspace/collab-splats-docs-wt add collab_splats/pointcloud/wrappers.py
git -C /workspace/collab-splats-docs-wt commit -m "docs(pointcloud): ########-style dividers + block comments in wrappers.py"
```

---

### Task 4: `pointcloud/bundle_adjustment.py` — section dividers

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`

- [ ] **Step 1: List current dash dividers**

```bash
grep -n '# --' collab_splats/pointcloud/bundle_adjustment.py
```

Expected: multiple lines inside `run_bundle_adjustment` using `# ------------------------------------------------------------------` style.

- [ ] **Step 2: Add top-level section dividers**

After the `__all__` line and before `@dataclass`, add:

```python
########################################################
########## Configuration ##############################
########################################################
```

Before `def extract_tracks_vggsfm`, add:

```python
########################################################
########## Track extraction ############################
########################################################
```

Before `def run_bundle_adjustment`, add:

```python
########################################################
########## Bundle adjustment (LM) #####################
########################################################
```

- [ ] **Step 3: Replace internal dash dividers with `########` style**

Inside `run_bundle_adjustment`, each `# ------------------------------------------------------------------` with a label line becomes:

```python
########################################################
########## Step 1: Filter observations ################
########################################################
```

Use the existing label text (e.g. "Filter observations", "Initialise pose parameters", "Run LM optimisation", "Refine intrinsics") to fill in the header.

- [ ] **Step 4: Verify no # -- style dividers remain**

```bash
grep -c '# --' collab_splats/pointcloud/bundle_adjustment.py
```

Expected: `0`.

- [ ] **Step 5: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -x -q 2>&1 | tail -8
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git -C /workspace/collab-splats-docs-wt add collab_splats/pointcloud/bundle_adjustment.py
git -C /workspace/collab-splats-docs-wt commit -m "docs(pointcloud): ########-style dividers in bundle_adjustment.py"
```

---

### Task 5: `mesh/utils.py` — print() → logging

**Files:**
- Modify: `collab_splats/mesh/utils.py`

- [ ] **Step 1: Confirm no logger exists yet**

```bash
grep -n 'import logging\|logger' collab_splats/mesh/utils.py | head -5
```

Expected: no output (file currently has no logging setup).

- [ ] **Step 2: Add logger after the existing imports block**

Find the last `import` / `from` line in the file header. Add immediately after:

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 3: Fix line 214**

Replace:
```python
print(f"Removed {n_removed} components")
```
With:
```python
logger.info("Removed %d components", n_removed)
```

- [ ] **Step 4: Fix line 252**

Replace:
```python
print(f"Skipping hole {he} of perimeter {mesh.holePerimiter(he)}")
```
With:
```python
logger.debug("Skipping hole %s of perimeter %s", he, mesh.holePerimiter(he))
```

- [ ] **Step 5: Verify no print() remain**

```bash
grep -c 'print(' collab_splats/mesh/utils.py
```

Expected: `0`.

- [ ] **Step 6: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/ -x -q 2>&1 | tail -8
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git -C /workspace/collab-splats-docs-wt add collab_splats/mesh/utils.py
git -C /workspace/collab-splats-docs-wt commit -m "docs(mesh): print→logging in utils.py"
```

---

### Task 6: Full verification

**Files:**
- Modify: `worklog/STATE.md`

- [ ] **Step 1: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: same pass count as before this work (4 pre-existing failures remain OK — verify they haven't increased).

- [ ] **Step 2: Verify no print() remain in core modules**

```bash
grep -rn 'print(' collab_splats/pointcloud/ collab_splats/mesh/utils.py
```

Expected: no output.

- [ ] **Step 3: Verify ########-style dividers are present in all 3 pointcloud files**

```bash
grep -l '########' collab_splats/pointcloud/utils.py \
                   collab_splats/pointcloud/wrappers.py \
                   collab_splats/pointcloud/bundle_adjustment.py
```

Expected: all 3 listed.

- [ ] **Step 4: Update worklog/STATE.md**

Move `inline-documentation-cleanup` to **Recently Completed** with a one-line summary:
> `pointcloud/utils.py`, `wrappers.py`, `bundle_adjustment.py`, `mesh/utils.py` — `########` dividers, inline block comments, `print()→logging`.

Commit:
```bash
git -C /workspace/collab-splats-docs-wt add worklog/STATE.md
git -C /workspace/collab-splats-docs-wt commit -m "docs(worklog): mark inline-documentation-cleanup complete"
```
