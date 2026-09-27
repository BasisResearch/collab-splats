# setup.sh Prerequisite Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `setup.sh` runs anywhere: it no longer exits on a missing nvcc, and prints the toolkit recipe only when a CUDA build actually fails.

**Architecture:** nvcc lookup honors `CUDA_HOME` and sets `HAVE_NVCC`; the CCCL overlay runs only with nvcc; both `uv sync` calls get a `||` handler that prints the recipe when `HAVE_NVCC=0`. README Install gets five in-place line edits.

**Tech Stack:** bash, uv, git plumbing (`hash-object` / `update-index`) for a partial-file commit.

**Spec:** `docs/superpowers/specs/2026-09-27-setup-prereq-gating-design.md`

---

## Ground rules for the executor

- Branch `clean/final` is live; another session may commit. Never `git add -A`, never amend,
  never rebase. Commit with `git commit --only <path>` except in Task 3 (explained there).
- `README.md` holds someone else's uncommitted intro hunk (lines 3-8, module list). Never
  commit or revert it.
- `SCRATCH=/tmp/claude-0/-workspace-collab-splats/bedeb7f9-e580-4035-ab6d-f6e063841965/scratchpad`
  — all throwaway files go there.
- This pod has no nvcc (`/usr/local/cuda/bin` absent). That is the test condition, not a bug.

## File map

- Modify: `setup.sh` — header comment, nvcc/CCCL block (lines 10-54), two `uv sync` calls (lines 89, 94)
- Modify: `README.md` — Install §2 callout, §3 collab-data + rclone, §4 lead line, §5 one line
- Modify: `CLAUDE.md` In-Flight (add, then remove) and `docs/superpowers/CHANGELOG.md` (append)
- Tests: shell checks in `$SCRATCH` only; no pytest (setup.sh has no pytest coverage today)

---

### Task 1: setup.sh — warn on missing nvcc, recipe on failed sync

**Files:**
- Modify: `setup.sh:1-54`, `setup.sh:89`, `setup.sh:94`

- [ ] **Step 1: Write the failing check (stub harness)**

Create `$SCRATCH/check_setup.sh`:

```bash
#!/bin/bash
# Stub harness for setup.sh's nvcc handling
# - copies setup.sh, swaps both `uv sync` calls for `false` (no network, no venv mutation)
# - case A: CUDA_HOME without nvcc -> WARN line, recipe after failed sync, exit 1
# - case B: CUDA_HOME with a fake nvcc -> no WARN, no recipe, exit 1
SCRATCH=/tmp/claude-0/-workspace-collab-splats/bedeb7f9-e580-4035-ab6d-f6e063841965/scratchpad
REPO=/workspace/collab-splats
H="$SCRATCH/harness"
rm -rf "$H" && mkdir -p "$H/fakecuda/bin" "$H/nocuda"
# one pattern covers both calls (deps-only is indented); `false` ignores the leftover flags
sed 's#^\( *\)/root/.local/bin/uv sync --locked --all-extras#\1false#' \
    "$REPO/setup.sh" > "$H/setup.sh"
grep -c '^ *false' "$H/setup.sh" | grep -qx 2 || { echo "HARNESS: expected 2 stubbed syncs"; exit 2; }
printf '#!/bin/sh\necho fake\n' > "$H/fakecuda/bin/nvcc" && chmod +x "$H/fakecuda/bin/nvcc"

fail=0
CUDA_HOME="$H/nocuda" bash "$H/setup.sh" > "$H/a.log" 2>&1; rc=$?
grep -q 'WARN no nvcc' "$H/a.log"      || { echo "A: missing WARN line"; fail=1; }
grep -q 'micromamba create' "$H/a.log" || { echo "A: missing recipe"; fail=1; }
[ $rc = 1 ] || { echo "A: rc=$rc want 1"; fail=1; }

# case B: CCCL overlay goes to a scratch prefix so /opt is untouched
CUDA_HOME="$H/fakecuda" CCCL_PREFIX="$H/cccl" bash "$H/setup.sh" > "$H/b.log" 2>&1; rc=$?
grep -q 'WARN no nvcc' "$H/b.log"      && { echo "B: WARN printed with nvcc present"; fail=1; }
grep -q 'micromamba create' "$H/b.log" && { echo "B: recipe printed with nvcc present"; fail=1; }
[ $rc = 1 ] || { echo "B: rc=$rc want 1"; fail=1; }

[ $fail = 0 ] && echo "ALL PASS" || { echo "--- a.log"; tail -5 "$H/a.log"; echo "--- b.log"; tail -5 "$H/b.log"; exit 1; }
```

- [ ] **Step 2: Run it — expect FAIL on case A**

Run: `bash $SCRATCH/check_setup.sh`
Expected: `A: missing WARN line` and `B: recipe printed with nvcc present` — the current script
checks the hardcoded `/usr/local/cuda/bin/nvcc` (absent in this pod), so it prints the recipe and
exits before sync in both cases. Case B needs network (CCCL wheel to `$H/cccl`) once fixed.

- [ ] **Step 3: Header comment — document SETUP_DEPS_ONLY**

In `setup.sh`, replace lines 2-3:

```bash
# Single source of truth for env setup. Runs in the Docker build AND standalone.
# Floor required from the host/image: gcc/g++ (build-essential) + (at runtime) NVIDIA driver.
```

with:

```bash
# Single source of truth for env setup. Runs in the Docker build AND standalone.
# Floor required from the host/image: gcc/g++ (build-essential) + (at runtime) NVIDIA driver.
# SETUP_DEPS_ONLY=1: Docker pass 1 — lock only, no project install; keeps the CUDA compile layer
# cached across source edits.
```

- [ ] **Step 4: Replace the nvcc block + gate the CCCL block**

Replace everything from `# CUDA build environment for the source-compiled extensions` through the
closing `fi` of the CCCL block (the one after `gsplat d2f5c0f will fail to compile.` / `exit 1`)
with:

```bash
# CUDA build environment for the source-compiled extensions (bae, gsplat, fused-ssim, nvdiffrast)
# - toolkit is the user's to provide (README §4); setup.sh never installs system packages
# - honors a pre-set CUDA_HOME (conda, /usr/local/cuda-12.1); default /usr/local/cuda
# - no nvcc: warn and continue — an already-built env (runtime Docker image) compiles nothing;
#   a failed `uv sync` then prints the recipe below
cuda_toolkit_help() {
    cat >&2 <<'MSG'
setup.sh: uv sync failed with no nvcc under $CUDA_HOME — bae, gsplat, fused-ssim and nvdiffrast
build from source and need it. The CUDA toolkit is yours to provide (README §4). Options:
Use the Docker image (every extension prebuilt), or the nvidia/cuda:12.1.1-devel image (it supplies
nvcc + dev headers, but NOT a new enough CCCL — setup.sh overlays that), or install just the two
apt packages the image ships (verified 2026-09-06 on a bare host, exact match to torch cu121):
  apt-get install -y --no-install-recommends cuda-nvcc-12-1 cuda-libraries-dev-12-1
Or build the toolkit with micromamba (verified 2026-08-22):
  micromamba create -p /opt/cuda-nvcc-12.1 -c nvidia -c conda-forge \
      cuda-version=12.1 cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1
  ln -sfn libcudart.so.12 /opt/cuda-nvcc-12.1/lib/libcudart.so   # solver leaves it dangling
  ln -sfn lib /opt/cuda-nvcc-12.1/lib64 && ln -sfn /opt/cuda-nvcc-12.1 /usr/local/cuda
MSG
}

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
HAVE_NVCC=0
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    HAVE_NVCC=1
    export PATH="$CUDA_HOME/bin:$PATH"
else
    echo "setup.sh: WARN no nvcc at $CUDA_HOME/bin/nvcc — fine if the CUDA extensions are already built." >&2
fi

# libcu++ (CCCL) floor — REQUIRED even inside nvidia/cuda:12.1.1-devel. gsplat d2f5c0f includes
# <cuda/std/optional>, which arrived in CCCL 2.2. CUDA 12.1 ships cuda/std/detail/libcxx/include/
# optional but NOT cuda/std/optional, so with a newer overlay on the path the OUTER header resolves
# to the overlay while the NESTED one falls back to 12.1's pre-2.2 libcxx and dies with
# "fatal error: __config: No such file or directory". Measured 2026-09-06: cccl 12.3.101 and
# 12.4.127 still lack cuda/std/optional; 12.6.77 is the first wheel that carries it. Staged to its
# own prefix so the venv and the system toolkit are both left alone.
# The overlay reaches nvcc ONLY through NVCC_PREPEND_FLAGS. Do NOT also put it on CPLUS_INCLUDE_PATH
# (or CPATH): those rank BELOW the toolkit's own -I on the host preprocessor's search order, which
# resurrects the exact __config failure this block exists to prevent.
# Skipped without nvcc: the overlay only feeds nvcc.
if [ "$HAVE_NVCC" = 1 ]; then
    CCCL_PREFIX="${CCCL_PREFIX:-/opt/cccl-12.6.77}"
    CCCL_INC="$CCCL_PREFIX/nvidia/cuda_cccl/include"
    if [ ! -f "$CCCL_INC/cuda/std/optional" ]; then
        /root/.local/bin/uv pip install --target "$CCCL_PREFIX" nvidia-cuda-cccl-cu12==12.6.77
    fi
    if [ -f "$CCCL_INC/cuda/std/optional" ]; then
        export NVCC_PREPEND_FLAGS="-I$CCCL_INC ${NVCC_PREPEND_FLAGS:-}"
    else
        echo "setup.sh: no <cuda/std/optional> at $CCCL_INC — gsplat d2f5c0f will fail to compile." >&2
        exit 1
    fi
fi
```

Note the heredoc stays quoted (`'MSG'`): the recipe's `\` line continuations must not be
interpreted, so `$CUDA_HOME` prints literally — intended.

- [ ] **Step 5: Wrap both `uv sync` calls**

Replace:

```bash
    /root/.local/bin/uv sync --locked --all-extras --no-install-project
```

with:

```bash
    /root/.local/bin/uv sync --locked --all-extras --no-install-project \
        || { [ "$HAVE_NVCC" = 0 ] && cuda_toolkit_help; exit 1; }
```

and replace:

```bash
/root/.local/bin/uv sync --locked --all-extras
```

(the standalone line after the `SETUP_DEPS_ONLY` block) with:

```bash
/root/.local/bin/uv sync --locked --all-extras \
    || { [ "$HAVE_NVCC" = 0 ] && cuda_toolkit_help; exit 1; }
```

- [ ] **Step 6: Syntax check + harness passes**

Run: `bash -n /workspace/collab-splats/setup.sh && bash $SCRATCH/check_setup.sh`
Expected: `ALL PASS`

- [ ] **Step 7: Commit**

```bash
cd /workspace/collab-splats && git commit --only setup.sh -m "fix(setup): warn on missing nvcc; print toolkit recipe only when uv sync fails

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Real run in the runtime pod (no nvcc)

**Files:** none modified. Mutates the shared venv (editable reinstall of collab-data /
collab-splats, InstantSfM extras pruned then re-added) — expected by design.

- [ ] **Step 1: Confirm nothing heavy is running**

Run: `pgrep -af "python|pytest" | grep -v pgrep`
Expected: no eval/training jobs. If any, stop and ask the user.

- [ ] **Step 2: Run setup.sh in the background, log to scratch**

Run (Bash `run_in_background: true`):
`cd /workspace/collab-splats && bash setup.sh > $SCRATCH/setup_run.log 2>&1; echo "rc=$?" >> $SCRATCH/setup_run.log`

- [ ] **Step 3: Verify**

Run: `grep -E 'WARN no nvcc|=== setup complete ===|\[OK\]|rc=' $SCRATCH/setup_run.log`
Expected: the WARN line, both `[OK]` smoke lines, `=== setup complete ===`, `rc=0`.

Run: `/opt/venv/reconstruction/bin/python -c "import instantsfm, pycolmap, gsplat; print('ok')"`
Expected: `ok`

If `rc` ≠ 0: report the tail of the log to the user; do not retry blindly.

---

### Task 3: README Install — five in-place edits

**Files:**
- Modify: `README.md` (Install §2-§5 only)

- [ ] **Step 1: Write the edit script**

Create `$SCRATCH/readme_edits.py`:

```python
"""
Apply the five Install-section edits to a README path in place.

- asserts each old string exists exactly once, so a drifted README fails loudly
"""
import sys

EDITS = [
    # §2 callout: warn, not fail-fast
    (
        "and fails fast with a micromamba recipe when nvcc is absent (no pip wheel ships nvcc).",
        "warns when nvcc is absent and prints the install recipe only if a build needs it (no pip wheel ships nvcc).",
    ),
    # §3 collab-data: locked path dependency, cloned by setup.sh
    (
        "`collab-data` is a private BasisResearch repo, kept out of the locked graph and installed post-sync (needs git credentials). `setup.sh` installs it best-effort; on a credential-less build it is skipped — re-run `setup.sh` at deploy, or install it directly:\n\n```sh\nuv pip install \"git+https://github.com/BasisResearch/collab-data.git\"\n```\n",
        "`collab-data` is a private BasisResearch repo, locked as a path dependency at `/workspace/collab-data` — `uv sync` cannot resolve without it. `setup.sh` clones it there if missing (needs GitHub credentials).\n",
    ),
    # §3 rclone: apt ships 1.53 (2020)
    (
        "sudo apt install rclone jq             # Debian / Ubuntu\n",
        "curl https://rclone.org/install.sh | sudo bash && sudo apt install jq   # Linux (apt's rclone is 1.53, from 2020)\n",
    ),
    # §4 lead sentence: the user provides these
    (
        "### 4. System requirements\n\n",
        "### 4. System requirements\n\nYou provide these; `setup.sh` checks for them but never installs system packages.\n\n",
    ),
    # §5 one line: runtime image has no nvcc
    (
        "The image runs the same `setup.sh`, with every CUDA extension compiled ahead of time.",
        "The image runs the same `setup.sh`, with every CUDA extension compiled ahead of time. The runtime\nimage has no nvcc by design; re-running `setup.sh` there is safe.",
    ),
]

path = sys.argv[1]
text = open(path).read()
for old, new in EDITS:
    assert text.count(old) == 1, f"expected exactly one match for: {old[:60]!r}"
    text = text.replace(old, new)
open(path, "w").write(text)
print(f"edited {path}")
```

- [ ] **Step 2: Apply to the working tree AND a HEAD copy**

```bash
cd /workspace/collab-splats
git show HEAD:README.md > $SCRATCH/README.head.md
/opt/venv/reconstruction/bin/python $SCRATCH/readme_edits.py README.md
/opt/venv/reconstruction/bin/python $SCRATCH/readme_edits.py $SCRATCH/README.head.md
```

Expected: two `edited ...` lines. The HEAD copy = committed README + Install edits, without the
uncommitted intro.

- [ ] **Step 3: Stage the HEAD copy only (not the working-tree file)**

`git add -p` is interactive (unavailable). Stage via plumbing:

```bash
cd /workspace/collab-splats
git diff --cached --name-only          # must print nothing; if not, STOP and ask the user
blob=$(git hash-object -w $SCRATCH/README.head.md)
git update-index --cacheinfo 100644,$blob,README.md
git diff --cached --stat               # README.md only
git diff --cached | grep '^[+-]' | grep -v '^[+-][+-]'   # only the five edits, no intro lines
git diff README.md | grep '^[+-]' | grep -v '^[+-][+-]'  # remaining unstaged diff = intro hunk only
```

Expected: staged diff = Install edits only; unstaged diff = the intro module-list lines only.

- [ ] **Step 4: Commit the index (plain commit, NOT --only)**

```bash
git commit -m "docs(readme): install — setup.sh warns on missing nvcc; collab-data + rclone corrected

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git diff README.md | head -20          # intro hunk still uncommitted
```

---

### Task 4: Bookkeeping

**Files:**
- Modify: `docs/superpowers/CHANGELOG.md` (insert after the header paragraph)
- Modify: `CLAUDE.md` (Recently Completed list)

- [ ] **Step 1: Append CHANGELOG entry** (newest first — insert above the pycolmap-cuda-docker entry)

```markdown
Recently completed (2026-09-27): **setup-prereq-gating** — `setup.sh` runs in any environment, including the nvcc-less runtime Docker image ([spec](specs/2026-09-27-setup-prereq-gating-design.md) · [plan](plans/2026-09-27-setup-prereq-gating.md)). Missing nvcc is a warning, not an exit; the CCCL overlay runs only with nvcc; both `uv sync` calls print the toolkit recipe when they fail without nvcc. nvcc lookup honors a pre-set `CUDA_HOME`. The user provides driver + toolkit; `setup.sh` never installs system packages. README Install: collab-data described as the locked path dep it is, rclone via rclone.org (apt ships 1.53). Measured in the runtime pod: `setup.sh` completes with zero compiles. **Owed:** editing `setup.sh` invalidates the Docker pass-1 CUDA layer — next Mac rebuild recompiles (~2.5 h).
```

- [ ] **Step 2: CLAUDE.md Recently Completed**

Add `- **setup-prereq-gating** (2026-09-27)` at the top of the "Five newest" list and drop the
last line (`- **mesh-cleanup** (2026-09-07)`).

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats
git add -f docs/superpowers/CHANGELOG.md
git commit --only CLAUDE.md docs/superpowers/CHANGELOG.md -m "docs(changelog): setup-prereq-gating complete

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
