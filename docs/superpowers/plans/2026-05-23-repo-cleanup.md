# Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove dead files (`patches/`, `VGGT-Long` submodule, `run_vggt_long.py`) and add `vendor/` + `third_party/` to `.gitignore` in one squash commit.

**Architecture:** Pure deletion + minor edits. No new code. No TDD ceremony. Verification = git status clean + test suite still passes.

**Tech Stack:** git, bash

**Spec:** `docs/superpowers/specs/2026-05-23-repo-cleanup-design.md`

---

### Task 1: Delete patches/ and update setup_nerfstudio.sh

**Files:**
- Delete: `patches/nerfstudio-pyproject.orig.toml`
- Modify: `setup_nerfstudio.sh`

- [ ] **Step 1: Remove the patches/ directory from git tracking**

```bash
git -C /workspace/collab-splats rm -r patches/
```

Expected output: `rm 'patches/nerfstudio-pyproject.orig.toml'`

- [ ] **Step 2: Remove stale comment from setup_nerfstudio.sh**

Open `setup_nerfstudio.sh`. Find and delete this line (currently line 15):

```bash
# Original preserved at patches/nerfstudio-pyproject.orig.toml.
```

The surrounding context looks like:
```bash
# Apply 5 compatibility patches inline via Python.
# Original preserved at patches/nerfstudio-pyproject.orig.toml.   ← DELETE THIS LINE
$PYTHON - <<'PYEOF'
```

After edit:
```bash
# Apply 5 compatibility patches inline via Python.
$PYTHON - <<'PYEOF'
```

- [ ] **Step 3: Verify**

```bash
git -C /workspace/collab-splats status patches/
```

Expected: `patches/nerfstudio-pyproject.orig.toml` shows as deleted.

---

### Task 2: Remove VGGT-Long submodule and associated runner

**Files:**
- Delete: `third_party/VGGT-Long/` (submodule)
- Delete: `evals/runners/run_vggt_long.py`
- Modify: `third_party/README.md`
- Modify: `.gitmodules` (handled automatically by git rm)

- [ ] **Step 1: Deinit the submodule**

```bash
git -C /workspace/collab-splats submodule deinit -f third_party/VGGT-Long
```

Expected: `Cleared directory 'third_party/VGGT-Long'` (or similar)

- [ ] **Step 2: Remove from git index and disk**

```bash
git -C /workspace/collab-splats rm third_party/VGGT-Long
```

Expected: `rm 'third_party/VGGT-Long'`

This also removes the VGGT-Long stanza from `.gitmodules` automatically.

- [ ] **Step 3: Clean git internal modules state**

```bash
rm -rf /workspace/collab-splats/.git/modules/third_party/VGGT-Long
```

No output expected.

- [ ] **Step 4: Remove run_vggt_long.py**

```bash
git -C /workspace/collab-splats rm evals/runners/run_vggt_long.py
```

Expected: `rm 'evals/runners/run_vggt_long.py'`

- [ ] **Step 5: Update third_party/README.md**

Open `third_party/README.md`. Remove the VGGT-Long row from the "Current entries" table:

```markdown
| `VGGT-Long/` | `DengKaiCQ/VGGT-Long` | `evals/runners/run_vggt_long.py` |
```

Table after edit:
```markdown
| Path | Upstream | Used by |
|---|---|---|
| `VGGT-SLAM/` | `MIT-SPARK/VGGT-SLAM` | `evals/runners/run_vggt_slam.py` |
| `VGGT-X/` | `Linketic/VGGT-X` | `setup_feedforward.sh`, `collab_splats/pointcloud/feedforward/vggtx.py` |
```

Note: add the `VGGT-X/` row if it's missing (the README predated VGGT-X being added).

- [ ] **Step 6: Verify .gitmodules no longer contains VGGT-Long**

```bash
grep -c 'VGGT-Long' /workspace/collab-splats/.gitmodules
```

Expected: `0`

---

### Task 3: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add vendor/ and third_party/ entries**

Open `.gitignore`. Append at the end of the file:

```gitignore

# Populated by setup scripts / git submodule update — not tracked directly
vendor/
third_party/
```

> **Note:** These entries do NOT affect already-tracked submodule paths
> (`third_party/VGGT-SLAM`, `third_party/VGGT-X`). Git ignores gitignore for
> paths it already tracks. They only prevent new untracked content from
> appearing in `git status`.

- [ ] **Step 2: Verify existing submodules still show up correctly**

```bash
git -C /workspace/collab-splats submodule status
```

Expected: VGGT-SLAM and VGGT-X still listed (with their SHAs). VGGT-Long absent.

---

### Task 4: Verify tests pass and create squash commit

**Files:** none new

- [ ] **Step 1: Run the test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all tests pass (same count as before this change — no test references VGGT-Long or patches/).

If any test fails mentioning `run_vggt_long` or `patches`, investigate — those should not exist.

- [ ] **Step 2: Check overall git status**

```bash
git -C /workspace/collab-splats status
```

Expect to see staged deletions for:
- `patches/nerfstudio-pyproject.orig.toml`
- `third_party/VGGT-Long` (submodule)
- `evals/runners/run_vggt_long.py`

And modifications to:
- `setup_nerfstudio.sh`
- `third_party/README.md`
- `.gitignore`
- `.gitmodules`

- [ ] **Step 3: Stage any unstaged modifications**

```bash
git -C /workspace/collab-splats add setup_nerfstudio.sh third_party/README.md .gitignore .gitmodules
```

- [ ] **Step 4: Create the squash commit**

```bash
git -C /workspace/collab-splats commit -m "$(cat <<'EOF'
chore: drop patches/ backup, VGGT-Long submodule, and gitignore vendor/+third_party/

patches/nerfstudio-pyproject.orig.toml was a reference backup for
setup_nerfstudio.sh inline patching. Not read by any script.

third_party/VGGT-Long was added speculatively; no production code
imports it. run_vggt_long.py removed alongside.

vendor/ and third_party/ added to .gitignore to prevent accidental
tracking of setup-script-populated dirs. Does not affect existing
tracked submodules (VGGT-SLAM, VGGT-X).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Verify commit looks right**

```bash
git -C /workspace/collab-splats show --stat HEAD
```

Expected: 6-7 files changed, all deletions or small modifications, no new source files.
