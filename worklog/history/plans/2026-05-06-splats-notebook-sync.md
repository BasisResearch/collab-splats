# Splats Notebook Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two broken imports and one stale method name in docs/splats/ notebooks after core-modules refactor.

**Architecture:** Direct JSON edits to three notebook cells. `clean_pcd` moved from `collab_splats.utils.pointcloud` → `collab_splats.pointcloud.utils`. `mesh_clustering` moved from `collab_splats.utils.mesh` → `collab_splats.mesh.utils`. Method name `visualize` → `viewer` in docs.

**Tech Stack:** Jupyter notebooks (JSON), Python imports

---

## Files

- Modify: `docs/splats/visualization.ipynb` — cell 1 import
- Modify: `docs/splats/create_mesh.ipynb` — cell 13 import
- Modify: `docs/splats/derive_splats.ipynb` — cell 0 markdown

---

### Task 1: Fix `clean_pcd` import in visualization.ipynb

**Files:**
- Modify: `docs/splats/visualization.ipynb`

- [ ] **Step 1: Edit cell 1 import**

Use the Edit tool to replace in `docs/splats/visualization.ipynb`:

```
"from collab_splats.utils.pointcloud import clean_pcd\n"
```
→
```
"from collab_splats.pointcloud.utils import clean_pcd\n"
```

- [ ] **Step 2: Verify**

```bash
python3 -c "
import json
nb = json.load(open('docs/splats/visualization.ipynb'))
cell1_src = ''.join(nb['cells'][1]['source'])
assert 'from collab_splats.pointcloud.utils import clean_pcd' in cell1_src, 'import not updated'
assert 'utils.pointcloud' not in cell1_src, 'old import still present'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/splats/visualization.ipynb
git commit -m "fix(notebook): update clean_pcd import path after utils restructure"
```

---

### Task 2: Fix `mesh_clustering` import in create_mesh.ipynb

**Files:**
- Modify: `docs/splats/create_mesh.ipynb`

- [ ] **Step 1: Edit cell 13 import**

Use the Edit tool to replace in `docs/splats/create_mesh.ipynb`:

```
"from collab_splats.utils.mesh import mesh_clustering\n"
```
→
```
"from collab_splats.mesh.utils import mesh_clustering\n"
```

- [ ] **Step 2: Verify**

```bash
python3 -c "
import json
nb = json.load(open('docs/splats/create_mesh.ipynb'))
cell13_src = ''.join(nb['cells'][13]['source'])
assert 'from collab_splats.mesh.utils import mesh_clustering' in cell13_src, 'import not updated'
assert 'utils.mesh' not in cell13_src, 'old import still present'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/splats/create_mesh.ipynb
git commit -m "fix(notebook): update mesh_clustering import path after utils restructure"
```

---

### Task 3: Fix stale method name in derive_splats.ipynb

**Files:**
- Modify: `docs/splats/derive_splats.ipynb`

- [ ] **Step 1: Edit cell 0 markdown**

Use the Edit tool to replace in `docs/splats/derive_splats.ipynb`:

```
"3. **visualize:** visualize splats via [ns-viewer]"
```
→
```
"3. **viewer:** visualize splats via [ns-viewer]"
```

- [ ] **Step 2: Verify**

```bash
python3 -c "
import json
nb = json.load(open('docs/splats/derive_splats.ipynb'))
cell0_src = ''.join(nb['cells'][0]['source'])
assert '**viewer:**' in cell0_src, 'method name not updated'
assert '**visualize:**' not in cell0_src, 'old name still present'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/splats/derive_splats.ipynb
git commit -m "docs(notebook): rename visualize → viewer in derive_splats docs"
```
