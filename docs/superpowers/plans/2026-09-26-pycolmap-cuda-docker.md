# pycolmap-cuda12 Docker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the colmap 3.10 binary with the `pycolmap-cuda12==4.1.1` wheel, delete the colmap stage from the Docker image, and take `README.md` out of the cached CUDA layer.

**Architecture:** `_generate_sift_database` (the one colmap CLI caller) moves to `pycolmap.extract_features` + `pycolmap.match_exhaustive` with an explicit `Device`. The GPU wheel replaces the plain `pycolmap` dependency, so every pycolmap caller gets 4.1.1. The Dockerfile loses the `colmap/colmap` stage and its system libs, and gains a runtime import smoke test.

**Tech Stack:** pycolmap-cuda12 4.1.1, uv, Docker (two-pass builder), pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-pycolmap-cuda-docker-design.md`

---

## Ground rules (read before any task)

- Branch `clean/final`, main checkout `/workspace/collab-splats`. Another session commits to this branch: commit with `git commit --only <paths>`, never `git add -A`, never amend.
- `README.md` has another session's uncommitted edit (lines 3-8, intro module list). Commit only this plan's hunks (Task 6 uses `git add -p`).
- Python: always `/opt/venv/reconstruction/bin/python`. The venv is SHARED with other sessions; Task 2 swaps pycolmap in it.
- Never run pytest piped through `| tail` (it eats the exit code). Write output to a file, then read it.
- Scratch dir: `S=/tmp/claude-0/-workspace-collab-splats/dbf6ce7f-5534-4974-b714-f60d28f56839/scratchpad`. `$S/pcprobe/imgs` holds 100 GH010229 frames (`frame_*.png` symlinks).
- Code style: CLAUDE.md "Code Style" (block comments, header+bullet comment runs, docstrings with `"""` on their own lines).

## File map

| File | Change |
|---|---|
| `pyproject.toml:71` | `pycolmap>=3.1` -> `pycolmap-cuda12==4.1.1` + pin note |
| `uv.lock` | relock (one package changes) |
| `collab_splats/pointcloud/sfm/instantsfm.py` | module docstring bullet, imports, section header, `_generate_sift_database` body + docstring |
| `tests/pointcloud/sfm/test_instantsfm.py` | 3 new flat tests |
| `Dockerfile` | header, pass-1 COPY, colmap stage, runtime apt, smoke test |
| `README.md` | section 4 optional tools; commit section 5 |
| `CLAUDE.md` | in-flight entry (Task 0), removed at Task 8 |
| `docs/superpowers/CHANGELOG.md` | completion entry (Task 8) |

---

### Task 0: In-flight entry

**Files:** Modify `CLAUDE.md` (In-Flight Work list, after the `tutorial-rework` bullet)

- [ ] **Step 1: Add the bullet**

```markdown
- **pycolmap-cuda-docker** — `pycolmap-cuda12==4.1.1` replaces the colmap 3.10 binary (InstantSfM SIFT DB); Docker drops the colmap stage and README from the cached CUDA layer ([spec](docs/superpowers/specs/2026-09-26-pycolmap-cuda-docker-design.md) · [plan](docs/superpowers/plans/2026-09-26-pycolmap-cuda-docker.md))
```

- [ ] **Step 2: Commit (plan + CLAUDE.md)**

```bash
cd /workspace/collab-splats
git add -f docs/superpowers/plans/2026-09-26-pycolmap-cuda-docker.md
git commit --only CLAUDE.md docs/superpowers/plans/2026-09-26-pycolmap-cuda-docker.md \
  -m "docs(plans): pycolmap-cuda-docker plan + in-flight entry"
```

---

### Task 1: Baselines on the current venv (pycolmap 4.0.4 + colmap binary)

Must run BEFORE Task 2 touches the venv. Nothing is committed.

- [ ] **Step 1: Record the pycolmap version being baselined**

Run: `/opt/venv/reconstruction/bin/python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"`
Expected: `4.0.4 False`

- [ ] **Step 2: Test-suite baseline**

```bash
cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python -m pytest \
  tests/pointcloud tests/geometry tests/wrapper tests/localization -q -rfE -p no:cacheprovider \
  > $S/gate_before.txt 2>&1; echo "exit=$?" >> $S/gate_before.txt
grep -E "^(FAILED|ERROR)|passed|failed|exit=" $S/gate_before.txt > $S/gate_before_summary.txt
```

(`$S` as defined in Ground rules; expand it inline.) Expected: a summary line `N failed, M passed, ...`. Run this with no other heavy job on the host (OOM risk + false failures).

- [ ] **Step 3: Real InstantSfM run with the binary-built DB**

Write `$S/sfm_run.py`:

```python
import sqlite3
import sys
import time
from pathlib import Path

import pycolmap

from collab_splats.pointcloud.sfm.instantsfm import InstantSfMCreator

# Fresh data dir per run; images read in place from the probe set
data_dir = Path(sys.argv[1])
images = Path(sys.argv[2])
data_dir.mkdir(parents=True, exist_ok=False)

t = time.time()
recon = InstantSfMCreator(use_depths=False).reconstruct(data_dir, images_dir=images)
elapsed = time.time() - t

# DB counts + reconstruction stats
db = sqlite3.connect(data_dir / "colmap" / "instantsfm.db")
pairs = db.execute("select count(*) from two_view_geometries where rows > 0").fetchone()[0]
db.close()
print(
    f"pycolmap={pycolmap.__version__} elapsed={elapsed:.0f}s pairs={pairs} "
    f"registered={recon.num_reg_images()} points3D={recon.num_points3D()} "
    f"mean_track={recon.compute_mean_track_length():.2f} "
    f"mean_reproj={recon.compute_mean_reprojection_error():.3f}"
)
```

Run: `cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python $S/sfm_run.py $S/sfm_before $S/pcprobe/imgs 2>&1 | tail -n 3 > $S/sfm_before.txt; cat $S/sfm_before.txt`
Expected: `pycolmap=4.0.4 ... registered=<~100> points3D=<N> ...`. If `instantsfm` is not importable, stop and report; the real-run comparison needs it.

---

### Task 2: Pin the wheel, relock, swap the venv

**Files:** Modify `pyproject.toml:71`, `uv.lock`

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the line `    "pycolmap>=3.1",` with:

```toml
    # pycolmap-cuda12, pinned at 4.1.1
    # - GPU SIFT wheel: replaces the colmap CLI (docs/superpowers/specs/2026-09-26-pycolmap-cuda-docker-design.md)
    # - not 4.2: shared-focal two-view estimator, ~14x verification CPU for single-camera scenes
    "pycolmap-cuda12==4.1.1",
```

- [ ] **Step 2: Relock**

Run: `cd /workspace/collab-splats && /root/.local/bin/uv lock 2>&1 | tail -n 5`
Expected: `Resolved 403 packages`, with `Removed pycolmap v4.0.4` and `Added pycolmap-cuda12 v4.1.1` and no other package added/removed/updated. Verify:

```bash
git diff uv.lock | grep -E '^[+-]name = ' 
```
Expected exactly: `-name = "pycolmap"` and `+name = "pycolmap-cuda12"`. Any other name = stop and report.

- [ ] **Step 3: Swap the shared venv**

Both wheels own the same `pycolmap/` directory; uninstall the old one first so its RECORD cannot delete the new files. `--inexact` keeps the non-lock extras (InstantSfM, pyceres, scikit-sparse, easydict) that a plain sync prunes.

```bash
cd /workspace/collab-splats
/root/.local/bin/uv pip uninstall --python /opt/venv/reconstruction/bin/python pycolmap
UV_PROJECT_ENVIRONMENT=/opt/venv/reconstruction /root/.local/bin/uv sync --locked --all-extras --inexact
/opt/venv/reconstruction/bin/python -c "import pycolmap, instantsfm, torch; print(pycolmap.__version__, pycolmap.has_cuda, torch.__version__)"
```
Expected: `4.1.1 True 2.5.1+cu121`.

- [ ] **Step 4: Commit**

```bash
git commit --only pyproject.toml uv.lock -m "build(deps): pin pycolmap-cuda12==4.1.1 in place of pycolmap"
```

---

### Task 3: Failing tests for `_generate_sift_database`

**Files:** Modify `tests/pointcloud/sfm/test_instantsfm.py` (append at end)

- [ ] **Step 1: Append the tests**

```python
def _record_sift_calls(monkeypatch, *, cuda: bool) -> list[tuple[str, dict]]:
    """
    Stub pycolmap SIFT entry points and torch's CUDA probe; return the call log.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        cuda: value torch.cuda.is_available() reports.

    Returns:
        (function name, kwargs) per pycolmap call, in call order.
    """
    calls = []
    monkeypatch.setattr(instantsfm.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(pycolmap, "extract_features", lambda *a, **kw: calls.append(("extract", kw)))
    monkeypatch.setattr(pycolmap, "match_exhaustive", lambda *a, **kw: calls.append(("match", kw)))
    return calls


def test_generate_sift_database_gpu_uses_cuda_device(monkeypatch, tmp_path):
    calls = _record_sift_calls(monkeypatch, cuda=True)

    instantsfm._generate_sift_database(tmp_path / "images", tmp_path / "db.db")

    # Extraction then matching, both pinned to CUDA; one shared SIMPLE_RADIAL camera
    assert [name for name, _ in calls] == ["extract", "match"]
    assert all(kw["device"] == pycolmap.Device.cuda for _, kw in calls)
    extract = calls[0][1]
    assert extract["camera_mode"] == pycolmap.CameraMode.SINGLE
    assert extract["reader_options"].camera_model == "SIMPLE_RADIAL"


def test_generate_sift_database_cpu_caps_threads(monkeypatch, tmp_path):
    calls = _record_sift_calls(monkeypatch, cuda=False)

    instantsfm._generate_sift_database(tmp_path / "images", tmp_path / "db.db", num_threads=5)

    # CPU device on both steps; the thread cap reaches both option structs
    extract, match = calls[0][1], calls[1][1]
    assert extract["device"] == match["device"] == pycolmap.Device.cpu
    assert extract["extraction_options"].num_threads == 5
    assert match["matching_options"].num_threads == 5


def test_generate_sift_database_failure_unlinks_partial_db(monkeypatch, tmp_path):
    db = tmp_path / "db.db"
    db.write_bytes(b"partial")

    def crash(*_args, **_kwargs):
        raise RuntimeError("sift died")

    monkeypatch.setattr(instantsfm.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pycolmap, "extract_features", lambda *a, **kw: None)
    monkeypatch.setattr(pycolmap, "match_exhaustive", crash)

    # Matching crash surfaces as RuntimeError and leaves no DB behind
    with pytest.raises(RuntimeError, match="SIFT database build failed"):
        instantsfm._generate_sift_database(tmp_path / "images", db)
    assert not db.exists()
```

Notes for the implementer:
- `num_threads=5` (not the default 8) so the CPU test proves the argument is forwarded, not that a constant happens to match.
- The GPU test asserts `extraction_options`/`matching_options` are passed but does not check `num_threads` on them: GPU path leaves colmap's default.

- [ ] **Step 2: Run, verify they fail**

Run: `cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm/test_instantsfm.py -k generate_sift_database -v -p no:cacheprovider > $S/t3.txt 2>&1; echo exit=$?; grep -E "PASS|FAIL|Error" $S/t3.txt`
Expected: the GPU and CPU tests FAIL (the current body runs the real `colmap` binary on a missing dir, never the stubs, so it raises `RuntimeError` before any call is logged). The failure test may already PASS on the old code (the binary also fails and the old path also unlinks + raises the same message) — it is a regression guard for the new except clause, not the TDD driver.

---

### Task 4: Move `_generate_sift_database` onto pycolmap

**Files:** Modify `collab_splats/pointcloud/sfm/instantsfm.py` (lines 1-30, 108-167)

- [ ] **Step 1: Module docstring** — replace the bullet

```
- SIFT database is built here with system colmap; upstream's own DB step is bypassed
```
with
```
- SIFT database is built here with pycolmap (GPU wheel); upstream's own DB step is bypassed
```

- [ ] **Step 2: Imports** — delete `import os` and `import subprocess` (line 13, 15). Confirm no other use: `grep -nE "\bos\.|subprocess" collab_splats/pointcloud/sfm/instantsfm.py` must print only the docstring mention at line ~78 ("A crashed colmap subprocess"); reword that one to "A crashed SIFT run".

- [ ] **Step 3: Section header** — `# SIFT feature database (system colmap)` -> `# SIFT feature database (pycolmap)`.

- [ ] **Step 4: Replace the whole function (lines 108-167)**

```python
def _generate_sift_database(image_path: Path, database_path: Path, *, num_threads: int = 8) -> None:
    """
    Build the COLMAP SIFT feature database: extraction + exhaustive matching.

    - num_threads: CPU SIFT thread cap. colmap's default (-1) spawns one thread per HOST core
      — 96 here — and per-thread RAM blows past the 46.6 GB container cgroup cap (measured:
      OOM-kill at default, clean 1.5 min run at 8 threads on 100 frames of 1920x1080).
    - GPU via the pycolmap-cuda12 4.1.1 wheel (measured 100x1907x1072 on an A40: extraction
      8 s, matching 25 s vs 10 s / 42 s for the colmap 3.10 binary it replaces); why 4.1.1 and
      not 4.2: docs/superpowers/specs/2026-09-26-pycolmap-cuda-docker-design.md.
    - Reimplements upstream GenerateDatabase (cre185/InstantSfM
      instantsfm/controllers/feature_handler.py:18-57 @ d3e599e), which forces CPU with no
      thread cap and swallows CalledProcessError — a colmap crash there surfaces only as an
      empty-tracks IndexError much later.
    - On failure the partial DB is unlinked so a re-run rebuilds from scratch.

    Args:
        image_path: directory of images to extract from.
        database_path: SQLite database to create.
        num_threads: CPU-path thread cap for extraction and matching; ignored on GPU.
    """
    use_gpu = torch.cuda.is_available()
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    # One shared SIMPLE_RADIAL camera over the whole set
    # - the sfm path stages frames from a single video/scene, so single_camera holds by
    #   construction (it was a creator field once and was never set False)
    # - per-image cameras would leave every intrinsic solved from one view
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_RADIAL"

    # CPU path only: cap threads on both steps (GPU path keeps colmap's defaults)
    extraction_options = pycolmap.FeatureExtractionOptions()
    matching_options = pycolmap.FeatureMatchingOptions()
    if not use_gpu:
        extraction_options.num_threads = num_threads
        matching_options.num_threads = num_threads

    try:
        logger.info("InstantSfM: SIFT extraction + exhaustive matching (%s)", "gpu" if use_gpu else "cpu")
        pycolmap.extract_features(
            database_path,
            image_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=device,
        )
        pycolmap.match_exhaustive(database_path, matching_options=matching_options, device=device)
    except (RuntimeError, ValueError) as err:
        database_path.unlink(missing_ok=True)
        raise RuntimeError(f"COLMAP SIFT database build failed ({err}) — is there enough memory?") from err
```

- [ ] **Step 5: Run the Task 3 tests**

Run: `cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm/test_instantsfm.py -v -p no:cacheprovider > $S/t4.txt 2>&1; echo exit=$?; grep -cE "PASSED" $S/t4.txt; grep -E "FAILED|ERROR" $S/t4.txt`
Expected: `exit=0`, the 3 new tests PASSED, no FAILED/ERROR in the file.

- [ ] **Step 6: Sanity mutation** — temporarily change `extraction_options.num_threads = num_threads` to `= 8`; rerun `-k cpu_caps_threads`; expected FAIL (`8 == 5`). Revert, rerun, expected PASS. This proves the CPU test sees the forwarding.

- [ ] **Step 7: Docstring contract + format**

```bash
cd /workspace/collab-splats
/opt/venv/reconstruction/bin/python -m pytest tests/test_docstring_contract.py -q -p no:cacheprovider > $S/t4doc.txt 2>&1; echo exit=$?
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/sfm/instantsfm.py tests/pointcloud/sfm/test_instantsfm.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/sfm/instantsfm.py tests/pointcloud/sfm/test_instantsfm.py
git diff --stat
```
Expected: `exit=0`; diff touches only the two files (black must not reformat unrelated lines — if it does, revert those hunks; never run repo-wide black).

- [ ] **Step 8: Commit**

```bash
git commit --only collab_splats/pointcloud/sfm/instantsfm.py tests/pointcloud/sfm/test_instantsfm.py \
  -m "feat(pointcloud): build the InstantSfM SIFT database with pycolmap-cuda12, not the colmap CLI"
```

---

### Task 5: Gates on the new wheel

Nothing committed unless a regression needs a fix (then: new failing test, fix, commit separately).

- [ ] **Step 1: Suite gate**

```bash
cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python -m pytest \
  tests/pointcloud tests/geometry tests/wrapper tests/localization -q -rfE -p no:cacheprovider \
  > $S/gate_after.txt 2>&1; echo "exit=$?" >> $S/gate_after.txt
grep -E "^(FAILED|ERROR)|passed|failed|exit=" $S/gate_after.txt > $S/gate_after_summary.txt
diff <(grep -E "^(FAILED|ERROR)" $S/gate_before_summary.txt | sort) <(grep -E "^(FAILED|ERROR)" $S/gate_after_summary.txt | sort)
```
Expected: passed count = before + 3; the diff prints nothing. Any NEW failure: read its traceback in `gate_after.txt`, decide whether it is a 4.0.4 -> 4.1.1 API change, report before fixing. A test that DISAPPEARS from the failure list is also reported (a shrinking failure list is not self-evidently better).

- [ ] **Step 2: Real InstantSfM run**

Run: `cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python $S/sfm_run.py $S/sfm_after $S/pcprobe/imgs 2>&1 | tail -n 3 > $S/sfm_after.txt; cat $S/sfm_before.txt $S/sfm_after.txt`
Expected: `pycolmap=4.1.1`, `pairs` ~838 (before ~809-826), `registered` equal to before (or higher), `points3D` and `mean_track` within ~5% of before, `mean_reproj` not worse by >10%. Record both lines in the Task 8 changelog entry. Worse on registered images = stop and report.

---

### Task 6: Dockerfile

**Files:** Modify `Dockerfile`

- [ ] **Step 1: Header** — line 6 tag `collab-env:cu121` -> `collab-splats:release` (matches README); line 13 becomes:

```
# - first build still takes hours; the CUDA layer is then cached until pyproject.toml, uv.lock,
#   setup.sh or collab-data change
```

- [ ] **Step 2: Pass-1 COPY** — line 72:

```dockerfile
COPY pyproject.toml uv.lock LICENSE setup.sh /workspace/collab-splats/
```

Check nothing in pass 1 reads README: `grep -n "readme\|README" pyproject.toml` — if `readme = "README.md"` is set, `uv sync --no-install-project` still does not build the project, so the file is unread (spec: verified). Keep the check output in the task report.

- [ ] **Step 3: Delete the colmap stage** — remove lines 77-82 (the `# Pre-built sources for runtime stage` divider block and `FROM colmap/colmap:20240213.23 AS colmap-source`) and lines 115-117 (`# Colmap binary` + two `COPY --from=colmap-source`).

- [ ] **Step 4: Runtime apt list** — replace lines 91-100 with:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
        libc6 libgcc-s1 libgl1 libgl1-mesa-glx libx11-6 \
        libhdf5-dev xvfb \
        build-essential ffmpeg libsuitesparse-dev \
        wget curl unzip xz-utils git vim htop tmux less \
        openssh-server gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
```

Removed (COLMAP-only): `libboost-filesystem1.74.0 libboost-program-options1.74.0 libceres2 libfreeimage3 libglew2.2 libgoogle-glog0v5 libqt5core5a libqt5gui5 libqt5widgets5`.

- [ ] **Step 5: Runtime smoke test** — replace lines 127-138 with:

```dockerfile
# Smoke test: torch, the AOT extensions and the system-lib-heavy wheels survived the stage copy
# - no GPU during build, so the creator chain is checked at run time (--gpus), not here
# - a missing .so would JIT-compile on first use, and this stage has no nvcc
# - open3d/cv2/pycolmap dlopen X11/GL/etc. from apt: a trimmed apt list fails here, not at first run
RUN python - <<'EOF'
import glob, sysconfig
import torch
import cv2, open3d, pycolmap

site = sysconfig.get_paths()["purelib"]
for pattern in ("gsplat/csrc*.so", "_nvdiffrast_c*.so", "bae/sparse/*.so"):
    assert glob.glob(f"{site}/{pattern}"), f"missing AOT build: {pattern}"
assert pycolmap.has_cuda, "pycolmap is the CPU wheel: install pycolmap-cuda12"
print(f"[Runtime] torch={torch.__version__} cuda={torch.version.cuda} pycolmap={pycolmap.__version__} "
      f"open3d={open3d.__version__} cv2={cv2.__version__}; AOT extensions present")
EOF
```

- [ ] **Step 6: Static check of system libs on this host (no Docker here)** — the runtime image is not buildable in this container; verify the claim the apt list rests on:

```bash
SITE=/opt/venv/reconstruction/lib/python3.11/site-packages
for so in $SITE/open3d/cpu/pybind*.so $SITE/pycolmap/_core*.so $SITE/cv2/cv2*.so; do
  echo "== $so"; ldd "$so" | grep -v -E "$SITE|linux-vdso|ld-linux" | awk '{print $1, $3}'
done
```
Expected: every non-venv lib is one of `libc`, `libm`, `libdl`, `libpthread`, `librt`, `libstdc++`, `libgcc_s`, `libX11`, `libGL`, `libudev`, `libtbb`, `libgomp`, `libc++`/`libc++abi`, `libxcb*`/`libXau`/`libXdmcp` (pulled by libx11-6), GLVND libs (pulled by libgl1). If anything else appears (e.g. `libtbb.so.12` / `libc++.so.1` / `libudev.so.1`), check whether the nvidia/cuda runtime base provides it (`docker run` is unavailable here — look it up in ubuntu 22.04 package lists: `libtbb12`, `libc++1-14`, `libudev1`) and add the package to Step 4 if it was only arriving through a removed Qt/GLEW/boost package. Record the list in the task report.

- [ ] **Step 7: Commit**

```bash
git commit --only Dockerfile -m "build(docker): drop the colmap 3.10 stage and README from the cached CUDA layer"
```

---

### Task 7: README

**Files:** Modify `README.md` (line 92; section 5 already in working tree)

- [ ] **Step 1: Section 4** — line 92:

```markdown
- Optional: `ffmpeg`, `rclone` for the video and data pipelines.
```

- [ ] **Step 2: Section 5 check** — the rebuild bullet (lines 106-107) already lists `pyproject.toml`, `uv.lock`, `setup.sh` or `../collab-data`: matches the new pass-1 COPY, no edit.

- [ ] **Step 3: Stage only this plan's hunks**

`git add -p README.md`: stage the section-4 hunk and the `### 5. Docker image` hunk; answer `n` to the intro hunk (lines 3-8: module list, "Gaussian splatting from", belongs to another session). Verify:

```bash
git diff --cached README.md | grep -E '^[+-]' | grep -v '^[+-]{3}' | head -60
git diff README.md   # must still show only the intro hunk
```

- [ ] **Step 4: Commit (staged hunks only — NOT `--only`, which would take the whole file)**

```bash
git diff --cached --name-only   # must print exactly README.md; anything else = unstage it first
git commit -m "docs(readme): Docker image section; colmap no longer a system requirement"
```

---

### Task 8: Close out

- [ ] **Step 1: Graph** — `cd /workspace/collab-splats && graphify update .`
- [ ] **Step 2: Changelog** — append to `docs/superpowers/CHANGELOG.md` (match the existing entry format there; read the top two entries first):
  - pycolmap-cuda-docker (2026-09-26): commits from Tasks 2/4/6/7; gate before/after summary lines; `sfm_before.txt` / `sfm_after.txt` lines; owed: user Mac rebuild + GPU-host `docker run --gpus all collab-splats:release python -c "import pycolmap, torch; print(pycolmap.has_cuda, torch.cuda.get_device_name())"` before any push.
- [ ] **Step 3: CLAUDE.md** — remove the Task 0 in-flight bullet; add `- **pycolmap-cuda-docker** (2026-09-26)` at the top of "Recently Completed" and drop the oldest of the five.
- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/CHANGELOG.md
git commit --only CLAUDE.md docs/superpowers/CHANGELOG.md -m "docs(changelog): record pycolmap-cuda-docker"
```

- [ ] **Step 5: Hand the user the Docker steps** (no Docker in this container):

```sh
docker build --platform=linux/amd64 --progress=plain --build-context collab-data=../collab-data --build-arg MAX_JOBS=4 -t collab-splats:release .
```
Expect one pass-1 cache miss (pyproject/uv.lock changed, ~2.5 h), then the runtime smoke line printing `pycolmap=4.1.1`. Nothing is pushed.
