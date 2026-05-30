# hloc Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `setup_hloc.sh` to clone and install hloc into the nerfstudio conda env for standalone SfM map building.

**Architecture:** Single shell script clones the hloc repo with submodules into `third_party/hloc/` and pip-installs it editable into the nerfstudio env. Guard clause makes it re-runnable. `.gitignore` already covers `vendor/*` — no change needed.

**Tech Stack:** bash, git submodules, `/opt/conda/envs/nerfstudio/bin/pip`

---

### Task 1: Write `setup_hloc.sh`

**Files:**
- Create: `setup_hloc.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
set -euo pipefail

VENDOR_DIR="$(dirname "$0")/third_party/hloc"
PIP="/opt/conda/envs/nerfstudio/bin/pip"
PYTHON="/opt/conda/envs/nerfstudio/bin/python"

echo "==> hloc setup"

if [ -d "$VENDOR_DIR" ]; then
    echo "    third_party/hloc/ already exists — skipping clone"
else
    echo "    Cloning Hierarchical-Localization with submodules..."
    git clone --recursive https://github.com/cvg/Hierarchical-Localization "$VENDOR_DIR"
fi

echo "    Installing hloc into nerfstudio env..."
"$PIP" install -e "$VENDOR_DIR"

echo "    Verifying install..."
"$PYTHON" -c "import hloc; print('hloc', hloc.__version__, 'installed OK')"

echo "==> Done."
```

- [ ] **Step 2: Make executable**

```bash
chmod +x setup_hloc.sh
```

- [ ] **Step 3: Commit**

```bash
git add setup_hloc.sh
git commit -m "feat(setup): add setup_hloc.sh — clone and install hloc into nerfstudio env"
```

---

### Task 2: Run and verify

**Files:** none (verification only)

- [ ] **Step 1: Run the script**

```bash
bash setup_hloc.sh
```

Expected output (last lines):
```
    hloc 1.x installed OK
==> Done.
```

If clone already present (re-run):
```
    third_party/hloc/ already exists — skipping clone
    Installing hloc into nerfstudio env...
    hloc 1.x installed OK
==> Done.
```

- [ ] **Step 2: Confirm importable from nerfstudio env**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import hloc
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval
print('hloc modules OK:', hloc.__version__)
"
```

Expected: `hloc modules OK: 1.x`

- [ ] **Step 3: Confirm submodule extractors present**

```bash
ls third_party/hloc/hloc/extractors/
```

Expected: files including `superpoint.py`, `disk.py`, `sift.py`, etc.

- [ ] **Step 4: No commit needed** — this task is verification only.
