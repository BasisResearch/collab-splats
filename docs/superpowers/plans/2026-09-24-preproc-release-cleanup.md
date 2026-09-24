# Preproc Release Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `collab_splats/preproc/` release-ready per
[decision 017](../decisions/017-release-cleanup-rules.md): brief docs, tunables as kwargs,
silent fallbacks raise, over-engineered helpers folded.

**Architecture:** Three phases on one worktree branch. (1) Contract checks that encode
017's mechanical rules are added first, while `preproc` stays exempt, so the noise is
measured before any edit. (2) Round 1 changes prose only. An AST-equality proof backs
every commit. (3) Round 2 makes code changes, one commit per logical change, each gated.
Last, `preproc` joins the release set and must pass every check.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest, PyAV, OpenCV, pycolmap, matplotlib.

**Spec:** [2026-09-24-preproc-release-cleanup-design.md](../specs/2026-09-24-preproc-release-cleanup-design.md)

---

## Conventions for every task

- `WT=/workspace/collab-splats/.worktrees/preproc-release`. Every command starts with `cd $WT &&`,
  because the working directory resets between Bash calls.
- **Gate** (`G`), run after every commit-bound change:

  ```bash
  cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" \
    && cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest \
       tests/preproc tests/wrapper tests/dashboard tests/test_docstring_contract.py -q -p no:cacheprovider
  ```

  - The printed path must start with `$WT/`.
  - Never pipe through `tail`, and never use `--tb=no`.
  - Compare the pass/fail/skip counts against the Task 0 baseline. Every change to a
    count must be explained by the task's own test adds and deletes.
- Commit only your own paths: `git add <paths> && git commit --only <paths> -m ...`. Other
  sessions share the git index. Never `--amend`, rebase or reset; fix mistakes with a new commit.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Spelling is US (`color`, `honor`).

---

## Task 0: Worktree, symlinks, proof tool, baseline

**Files:**
- Create: `.worktrees/preproc-release/` (branch `clean/preproc-release`)
- Create (scratch, not committed): `/tmp/preproc_release/prose_proof.py`

- [ ] **Step 1: Create the worktree off `clean/final`**

```bash
cd /workspace/collab-splats && git worktree add .worktrees/preproc-release -b clean/preproc-release clean/final
```

- [ ] **Step 2: Symlink `third_party/*`.** It is gitignored; without it, guarded tests skip instead of failing.

```bash
cd /workspace/collab-splats/.worktrees/preproc-release && for d in /workspace/collab-splats/third_party/*/; do ln -sfn "$d" "third_party/$(basename "$d")"; done && ls third_party
```

Expected: `LoGeR VGGT-SLAM VGGT-X Video-Depth-Anything bae hloc vggt-omega vggt_spark xfeat` (plus README).

- [ ] **Step 3: Write the prose-proof tool**

```bash
mkdir -p /tmp/preproc_release && cat > /tmp/preproc_release/prose_proof.py <<'EOF'
"""
Prove a commit range changed only docstrings and comments in the given .py files.

Usage: prose_proof.py <base-ref> <file> [<file> ...]   (run from the worktree root)
"""
import ast
import subprocess
import sys


def stripped(src: str) -> str:
    """
    AST dump with every docstring statement deleted; comments never reach the AST.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [
                s for s in body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
            ] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


base, files = sys.argv[1], sys.argv[2:]
bad = []
for f in files:
    old = subprocess.run(["git", "show", f"{base}:{f}"], capture_output=True, text=True, check=True).stdout
    new = open(f).read()
    if stripped(old) != stripped(new):
        bad.append(f)
print("CODE CHANGED:" if bad else "PROSE ONLY", *bad)
sys.exit(1 if bad else 0)
EOF
```

- [ ] **Step 4: Sanity-check the proof tool on a known-bad mutation**

```bash
cd $WT && cp collab_splats/preproc/frames.py /tmp/preproc_release/frames.bak \
  && sed -i 's/^SCHEMA_VERSION = 2$/SCHEMA_VERSION = 3/' collab_splats/preproc/frames.py \
  && /opt/venv/reconstruction/bin/python /tmp/preproc_release/prose_proof.py HEAD collab_splats/preproc/frames.py; \
  cp /tmp/preproc_release/frames.bak collab_splats/preproc/frames.py && git -C $WT status --short
```

Expected: `CODE CHANGED: collab_splats/preproc/frames.py`, then an empty `git status`.

- [ ] **Step 5: Run the baseline gate and record the counts**

Run the gate `G`. Write the final summary line (e.g. `N passed, M skipped, K failed`) and
every failing node id into `/tmp/preproc_release/baseline.txt`. Every later gate is compared
against this file.

---

## Task 1: Release contract checks (preproc still exempt)

**Files:**
- Modify: `tests/test_docstring_contract.py`

The five checks are pure functions from source text to messages. Each one is proven on a
known-bad fixture first. The per-file tests run over every package in `PACKAGES`. Files
outside `RELEASED` are `xfail(strict=False)`: they are listed as known failures, not
exempted. `RELEASED` starts empty, and Task 20 adds `"preproc"`.

- [ ] **Step 1: Write the fixture tests first (they fail: the checkers do not exist)**

Append to `tests/test_docstring_contract.py`:

```python
########################################################################
# Release checks (decision 017) — fixtures prove each check can fail
########################################################################


def test_numeric_constant_check_flags_a_tunable_and_spares_a_fact():
    src = "SCHEMA_VERSION = 2\n_PNG_COMPRESSION = 1\nNEG = -0.5\nIMAGE_EXTS = ('.png',)\nFLAG = True\n"
    assert _numeric_constants(src) == ["2: _PNG_COMPRESSION = 1", "3: NEG = -0.5"]


def test_banned_word_check_flags_lore_and_ignores_paths():
    src = (
        '"""\nSummary.\n\n- measured 3x faster on GH010229\n"""\n'
        "# see docs/superpowers/specs/2026-08-20-video-quality-report-measured.md\n"
        "# HYPOTHESIS: the ffmpeg pipe\n"
    )
    hits = _banned_words(src)
    assert len(hits) == 2 and hits[0].startswith("docstring") and hits[1].startswith("7:")


def test_bullet_cap_flags_a_seventh_bullet():
    body = "\n".join(f"    - b{i}" for i in range(7))
    src = f'def f() -> None:\n    """\n    Summary.\n\n{body}\n    """\n'
    assert _bullet_overflow(src) == ["f: 7 bullets (max 6)"]
    assert _bullet_overflow(src.replace("    - b6\n", "")) == []


def test_comment_run_cap_ignores_dividers_and_flags_a_fifth_line():
    ok = "#####\n# a\n# - b\n# - c\n# - d\n#####\nx = 1\n"
    bad = "# a\n# - b\n# - c\n# - d\n# - e\nx = 1\n"
    assert _long_comment_runs(ok) == []
    assert _long_comment_runs(bad) == ["1: 5-line comment run (max 4)"]


def test_silent_fallback_check_flags_or_number_and_swallowed_exception():
    src = (
        "fps = info['fps'] or 30.0\n"
        "name = x or 'default'\n"
        "try:\n    pass\nexcept Exception:\n    pass\n"
        "try:\n    pass\nexcept Exception:\n    raise\n"
        "try:\n    pass\nexcept ValueError:\n    pass\n"
    )
    assert _silent_fallbacks(src) == ["1: `or 30.0` fallback", "5: except Exception without re-raise"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/test_docstring_contract.py -q -k "check"`
Expected: 5 errors/failures with `NameError: name '_numeric_constants' is not defined` (and similar).

- [ ] **Step 3: Implement the checkers.** Add them above the fixture tests, after `test_comment_runs_state_the_problem_then_bullet_it`. Extend the module docstring by one bullet: `- release checks (017) run on every file and must pass for packages in RELEASED`.

Add `import io` and `import tokenize` to the imports. Then add:

```python
# Packages that finished their release cleanup; the rest xfail the release checks
RELEASED: frozenset[str] = frozenset()

# Module-level numbers that are facts, not tunables
FIXED_FACTS = frozenset({"SCHEMA_VERSION"})

UPPER_RE = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
BANNED_RE = re.compile(r"\bmeasured\b|\bhypothesis\b|\d+(\.\d+)?x faster|ffmpeg pipe|replaces the old", re.I)
SCENE_ID_RE = re.compile(r"\b(GH\d{6}|C\d{4})\b")
MAX_BULLETS = 6
MAX_COMMENT_LINES = 4


def _numeric_constants(src: str) -> list[str]:
    """
    Module-level UPPER_CASE names bound to a bare number, minus FIXED_FACTS.

    Args:
        src: module source.

    Returns:
        "<line>: <NAME> = <value>" per offender.
    """
    out = []
    for node in ast.parse(src).body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
        value = getattr(node, "value", None)
        if not isinstance(target, ast.Name) or not UPPER_RE.match(target.id) or target.id in FIXED_FACTS:
            continue
        num = value.operand if isinstance(value, ast.UnaryOp) else value
        if isinstance(num, ast.Constant) and isinstance(num.value, (int, float)) and not isinstance(num.value, bool):
            out.append(f"{node.lineno}: {target.id} = {ast.unparse(value)}")
    return out


def _comments(src: str) -> list[tuple[int, str]]:
    """
    (line, text) of every `#` comment, via tokenize so `#` inside strings is ignored.

    Args:
        src: module source.

    Returns:
        One entry per comment token, text without the leading `#`.
    """
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    return [(t.start[0], t.string.lstrip("#").strip()) for t in toks if t.type == tokenize.COMMENT]


def _docstrings(src: str) -> list[tuple[str, str]]:
    """
    (label, docstring) for the module and every def and class, private ones included.

    Args:
        src: module source.

    Returns:
        One entry per documented node.
    """
    tree = ast.parse(src)
    nodes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return [(getattr(n, "name", "module"), ast.get_docstring(n)) for n in nodes if ast.get_docstring(n)]


def _banned_words(src: str) -> list[str]:
    """
    Lore words and scene ids in docstrings and comments; paths are exempt.

    Args:
        src: module source.

    Returns:
        "docstring <label>: <word>" or "<line>: <word>" per hit.
    """
    texts = [(f"docstring {label}", doc) for label, doc in _docstrings(src)]
    texts += [(str(line), text) for line, text in _comments(src)]
    out = []
    for where, text in texts:
        text = re.sub(r"\S*/\S*", "", text)
        for rx in (BANNED_RE, SCENE_ID_RE):
            out += [f"{where}: {m.group(0)}" for m in rx.finditer(text)]
    return out


def _bullet_overflow(src: str) -> list[str]:
    """
    Docstrings with more than MAX_BULLETS top-level bullets before the first section.

    Args:
        src: module source.

    Returns:
        "<label>: <n> bullets (max 6)" per offender.
    """
    out = []
    for label, doc in _docstrings(src):
        body = []
        for line in doc.splitlines()[1:]:
            if SECTION_RE.match(line):
                break
            body.append(line)
        n = sum(1 for line in body if line.startswith("- "))
        if n > MAX_BULLETS:
            out.append(f"{label}: {n} bullets (max {MAX_BULLETS})")
    return out


def _long_comment_runs(src: str) -> list[str]:
    """
    Runs of consecutive comment lines longer than MAX_COMMENT_LINES; divider lines do not count.

    Args:
        src: module source.

    Returns:
        "<first line>: <n>-line comment run (max 4)" per offender.
    """
    lines = src.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("#"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip().startswith("#"):
            j += 1
        text = [k for k in range(i, j) if lines[k].strip().strip("#").strip()]
        if len(text) > MAX_COMMENT_LINES:
            out.append(f"{text[0] + 1}: {len(text)}-line comment run (max {MAX_COMMENT_LINES})")
        i = j
    return out


def _silent_fallbacks(src: str) -> list[str]:
    """
    `x or <number>` expressions and `except Exception:` handlers that never re-raise.

    Args:
        src: module source.

    Returns:
        "<line>: <idiom>" per offender.
    """
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            last = node.values[-1]
            if isinstance(last, ast.Constant) and isinstance(last.value, (int, float)) and not isinstance(last.value, bool):
                out.append(f"{node.lineno}: `or {last.value!r}` fallback")
        if isinstance(node, ast.ExceptHandler):
            broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"))
            if broad and not any(isinstance(n, ast.Raise) for n in ast.walk(node)):
                out.append(f"{node.lineno}: except Exception without re-raise")
    return out


RELEASE_CHECKS = {
    "numeric-constant": _numeric_constants,
    "banned-word": _banned_words,
    "bullet-cap": _bullet_overflow,
    "comment-cap": _long_comment_runs,
    "silent-fallback": _silent_fallbacks,
}


def _release_params() -> list:
    """
    One param per (file, check); files outside RELEASED are known failures.

    Returns:
        pytest params with ids "<file>::<check>".
    """
    params = []
    for path, pid in zip(SOURCES, SOURCE_IDS):
        pkg = path.relative_to(ROOT / "collab_splats").parts[0]
        marks = [] if pkg in RELEASED else [pytest.mark.xfail(strict=False, reason=f"{pkg}: release cleanup pending")]
        for name in RELEASE_CHECKS:
            params.append(pytest.param(path, name, marks=marks, id=f"{pid}::{name}"))
    return params


@pytest.mark.parametrize(("path", "check"), _release_params())
def test_release_rules(path, check):
    bad = RELEASE_CHECKS[check](path.read_text())
    assert not bad, "\n".join(bad)
```

- [ ] **Step 4: Run the fixture tests**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/test_docstring_contract.py -q -k "check"`
Expected: 5 passed. If one fails, fix the checker, not the fixture. The fixture encodes 017.

- [ ] **Step 5: Measure preproc noise before round 1**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python - <<'EOF'
import pathlib, sys
sys.path.insert(0, "tests")
import test_docstring_contract as c
for p in sorted(pathlib.Path("collab_splats/preproc").glob("*.py")):
    for name, fn in c.RELEASE_CHECKS.items():
        for hit in fn(p.read_text()):
            print(f"{p.name}\t{name}\t{hit}")
EOF
```

Save the output to `/tmp/preproc_release/noise.txt`. Expected hits, at minimum:
- numeric constants: `_PNG_COMPRESSION`, `_SIFT_NUM_THREADS`, `_MIN_REGISTERED_FRACTION`, `_MIN_CALIBRATION_IMAGES`, `_FIG_WIDTH_IN`, `_PANEL_HEIGHT_IN`, `_PNG_DPI`, `_THUMB_DPI`, `_HIST_BINS`;
- banned words: `measured` / `GH010229` in `qa.py`, `sampling.py`, `frames.py`, `undistort.py`;
- silent fallbacks: `or 30.0`, `or 1`, `or 0.0`, and the `except Exception` blocks in `video.py` and `sampling.py`.

A hit not covered by a later task is a gap. Add it to the matching round 1 or round 2 task before continuing.

- [ ] **Step 6: Gate and commit**

Run the gate `G`. Expected: baseline counts, plus 5 passed fixtures and N xfailed/xpassed release params. No new failures.

```bash
cd $WT && git add tests/test_docstring_contract.py && git commit --only tests/test_docstring_contract.py -m "test(contract): release checks from decision 017, packages xfail until released

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Round 1 — prose only

Each round 1 task edits only docstrings and `#` comments. Before each commit:

```bash
cd $WT && /opt/venv/reconstruction/bin/python /tmp/preproc_release/prose_proof.py clean/final <files touched>
```

Expected: `PROSE ONLY`. Then run the gate `G` and commit with `docs(preproc): ...`.
Summaries must not restate the function name, and `Args:` must still name every parameter
(the existing contract test enforces both).

### Task 2: `__init__.py` and `video.py` prose

**Files:** Modify `collab_splats/preproc/__init__.py:1-10`, `collab_splats/preproc/video.py`

- [ ] **Step 1: `__init__.py` module docstring** → replace lines 1-10 with:

```python
"""
Video preprocessing: probe and decode, measure quality, select and store keyframes.

- measure then select: qa.compute_video_quality writes a report, the samplers pick from it
- frames.write_frames stores picks as images/frame_NNNNNN.png plus frames.json
- plots live in preproc.viz, not re-exported, so pipeline imports skip matplotlib
"""
```

- [ ] **Step 2: `video.py` module docstring** (this also drops the false "grid lives here" bullet) → replace lines 1-12 with:

```python
"""
Video probe and decode, in-process via PyAV.

- iter_frames yields BGR (for cv2); extract_frame returns RGB; both uint8 HWC
- frames come out in display orientation; get_video_info reports display dims
- no selection logic here, so qa and sampling both import it without a cycle
"""
```

- [ ] **Step 3: `_rotation_degrees` docstring** →

```python
    """
    Clockwise display rotation, read off the first decoded frame.

    - PyAV exposes the display matrix only on decoded frames
    - advances the container: call it after reading stream metadata
    """
```

- [ ] **Step 4: `get_video_info` docstring** →

```python
    """
    Frame count, rate, duration and display size from a container parse.

    Args:
        video_path: source video.

    Returns:
        {"total_frames", "fps", "duration_s", "width", "height"}; width/height are display
        dims (rotation applied). Every value is zero if the container cannot be probed.
    """
```

- [ ] **Step 5: `_frame_index` and `_upright` docstrings** →

```python
    """
    Source frame index of a decoded frame, from its presentation timestamp.

    - needed after a seek, where the decode count is no longer the absolute index
    - assumes a constant frame rate
    """
```

```python
    """
    Decoded frame as BGR uint8 HWC in display orientation.

    - PyAV returns the stored orientation; the display rotation is applied here
    """
```

- [ ] **Step 6: `iter_frames`.** Replace the `Args:` entries for `video_path`, `indices` and `start` with:

```
        video_path: source video. An unopenable path yields nothing.
        indices: only these source frames, deduped and ascending; one linear scan, no seek.
        start: first frame of a contiguous window, reached by a container seek.
```

Delete the two-line comment above `try: container = av.open(...)` (the "ffmpeg pipe" note
duplicates `Args:`). Replace the three-line "Contiguous windows keep the input seek the
ffmpeg pipe had" comment with:

```python
        # Contiguous window: seek to the keyframe at or before `start`
        # - the loop below drops frames before `start`
```

- [ ] **Step 7: Prove, gate, commit**

```bash
cd $WT && /opt/venv/reconstruction/bin/python /tmp/preproc_release/prose_proof.py clean/final collab_splats/preproc/__init__.py collab_splats/preproc/video.py
```

Expected: `PROSE ONLY`. Run the gate `G`, then:

```bash
cd $WT && git add collab_splats/preproc/__init__.py collab_splats/preproc/video.py && git commit --only collab_splats/preproc/__init__.py collab_splats/preproc/video.py -m "docs(preproc): brief video and package docstrings; drop ffmpeg-era history

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: `qa.py` prose

**Files:** Modify `collab_splats/preproc/qa.py`

- [ ] **Step 1: Module docstring** (lines 1-17) →

```python
"""
Capture quality report: per-frame photometry and per-pair motion.

- frames: blur, laplacian, exposure_{mean,median,std}, clipped_{low,high}_frac
- pairs: n_matches, translation_px, parallax
- report-only: thresholds live in preproc.sampling.filter_frame_quality
- column evidence: docs/superpowers/specs/2026-08-20-video-quality-report-measured.md
"""
```

- [ ] **Step 2: `compute_exposure`.** Delete the `~98x faster` bullet. Replace the four-line
"Histogram median must match np.median" comment with:

```python
    # Histogram median must match np.median on an even pixel count
    # - np.median averages the two central values; searchsorted alone gives the lower
```

- [ ] **Step 3: `compute_frame_quality` docstring.** The old one claimed laplacian is native-resolution, which is false: `compute_blur` runs on the analysis gray. Replace it with:

```python
    """
    Photometry for one BGR frame: blur and exposure together.

    - blur and laplacian read a gray downscaled to analysis_width; exposure reads native
    - so blur and laplacian are not comparable across videos whose width straddles it

    Args:
        bgr: (H, W, 3) uint8 BGR frame.
        analysis_width: width blur and laplacian are computed at; exposure ignores it.
        blur_h_size: forwarded to compute_blur as `h_size`.

    Returns:
        compute_blur's keys merged with compute_exposure's, one flat dict.
    """
```

Replace the three-line "Blur reads the analysis-width gray" comment with
`    # Blur and laplacian read the analysis-width gray: cheaper, tied to analysis_width`.

- [ ] **Step 4: `detect_orb` docstring bullets** → one bullet: `- split from matching so a video run detects each frame once`.

- [ ] **Step 5: `compute_pair_motion` docstring** →

```python
    """
    Match two frames' ORB features and measure the motion between them.

    - pixels are those of the grid detect_orb ran on (analysis_gray in the report)
    - unmeasurable is nan, never 0.0, which would read as a still camera
    - parallax in [0, 1]: one minus the homography/fundamental inlier ratio; ~0 = no depth
    - crossCheck cannot detect unrelated frames: a scene cut reads as large motion

    Args:
        feat_a: (keypoints, descriptors) from detect_orb for the earlier frame.
        feat_b: (keypoints, descriptors) for the later frame.
        ransac_thresh_px: inlier threshold for both fits; looser lowers parallax.

    Returns:
        {'n_matches': int, 'translation_px': float, 'parallax': float}. translation_px is the
        median match displacement, nan with no matches; parallax is nan below 8 matches or
        when either fit fails.
    """
```

- [ ] **Step 6: `_measure_photometry_and_motion` docstring and thread-pin comment** →

```python
    """
    Measure one contiguous frame range; returns (frame rows, pair rows).

    - decodes `stride` lead-in frames before `emit_from` so boundary pairs have a partner
    - emits rows from `emit_from` on only, so ranges tile the video exactly once
    - module-level so ProcessPoolExecutor can pickle it
    """
```

```python
    # Pin cv2 and BLAS to one thread per worker
    # - unpinned, process fan-out oversubscribes the cores and runs slower than serial
```

- [ ] **Step 7: `compute_video_quality` docstring** →

```python
    """
    Per-frame photometry and per-pair motion across a whole video.

    - report-only: filter_frame_quality turns these columns into a keep mask

    Args:
        video_path: source video; every frame is decoded and measured.
        output_path: where to write the report as JSON; None skips the write.
        motion_stride: frames between the two members of each pair, >= 1; None = round(fps).
        workers: contiguous frame ranges measured in parallel; 1 is serial.

    Returns:
        {"available": True, "video": {path, mtime, **get_video_info}, "params":
        {"motion_stride"}, "frames": {column: list per frame}, "pairs": {column: list per
        pair}}, or {"available": False, "reason": str} for an unreadable video.
    """
```

Replace the five-line "Cheap integrity check on the seeks" comment with:

```python
    # Contiguity check: a bad seek silently shifts every index
    # - int comparison, so rounding cannot false-alarm
```

- [ ] **Step 8: Prove, gate, commit** (`prose_proof.py clean/final collab_splats/preproc/qa.py` → `PROSE ONLY`; gate `G`)

```bash
cd $WT && git add collab_splats/preproc/qa.py && git commit --only collab_splats/preproc/qa.py -m "docs(qa): contract-only docstrings; fix the false native-laplacian claim

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: `sampling.py` prose

**Files:** Modify `collab_splats/preproc/sampling.py`

- [ ] **Step 1: Module docstring** (lines 1-11) →

```python
"""
Keyframe selection from a quality report.

- filter_frame_quality turns report columns into a keep mask; qa never decides
- sample_fps fixes spacing, sample_uniform fixes count, sample_optical_flow follows motion
- every sampler picks only from frames the mask keeps
"""
```

- [ ] **Step 2: `OpticalFlowFrameSelector`.** Delete the two docstring bullets starting
"no cv2/kornia/open3d equivalent" and "the pieces it is built from". Replace the
four-line "Built here, not as defaults" comment with `        # Lucas-Kanade sparse flow params`.

- [ ] **Step 3: Grid section header** (lines 249-252) →

```python
########################################################################
# Selection grid — constant-rate source indices for sample_fps
########################################################################
```

- [ ] **Step 4: `_decode_selection` comments.** Change
`# One ffmpeg select pass over exactly the frames we keep` to
`# One decode pass over exactly the frames we keep`. Change
`continue  # ffmpeg dropped the frame (should not happen)` to
`continue  # decoder skipped the frame (should not happen)`.

- [ ] **Step 5: `sample_fps` comments.** Replace the five-line "Take the SHARPEST" comment with:

```python
    # Sharpest eligible frame per slot, ties to the nearest target
    # - the gate is video-wide, so nearest-in-time alone picks an arbitrary survivor
```

Replace the six-line "Slot holds no eligible frame" comment with:

```python
        # Slot holds no eligible frame: the policy decides
        # - rescue: keep the slot's sharpest frame anyway
        # - drop: emit nothing and accept the gap
```

- [ ] **Step 6: Prove, gate, commit** (`PROSE ONLY` on `sampling.py`; gate `G`)

```bash
cd $WT && git add collab_splats/preproc/sampling.py && git commit --only collab_splats/preproc/sampling.py -m "docs(sampling): drop scene figures and the stale VDA grid claim

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: `frames.py`, `undistort.py`, `viz.py` prose

**Files:** Modify `collab_splats/preproc/frames.py`, `collab_splats/preproc/undistort.py`, `collab_splats/preproc/viz.py`

- [ ] **Step 1: `frames.py` module docstring** →

```python
"""
Keyframe store: a COLMAP-style images/ directory plus frames.json beside it.

- images/frame_NNNNNN.png, lossless PNG, RGB at both boundaries
- frames.json holds selection records and provenance COLMAP has no slot for
"""
```

PNG comment → `# OpenCV's default: higher levels cost far more time for little size`.

- [ ] **Step 2: `undistort.py` module docstring** →

```python
"""
Camera self-calibration and undistortion for the images/ directory.

- pycolmap self-calibrates one shared OPENCV camera; cv2 remaps the pixels
- framing is COLMAP's (pycolmap.undistort_camera): the canvas grows, nothing is cropped
"""
```

Replace the six-line SIFT comment with `# Cap SIFT threads: the default (one per host core) OOMs the container`.

- [ ] **Step 3: `viz.py`.** Module docstring →

```python
"""
Matplotlib plots for keyframe sampling and the quality report.

- not re-exported from collab_splats.preproc, so pipeline imports skip matplotlib
"""
```

Replace the section header at lines 112-122 with:

```python
########################################################################
# Video quality report plots: raw columns, no thresholds; saved, never shown
########################################################################
```

On `_PNG_DPI`, drop the `# matches collab-data/...` comment. In `plot_frame_extremes`, change
the bullet `one ffmpeg seek per thumbnail (2n decodes)` to `one seek per thumbnail (2n decodes)`.

- [ ] **Step 4: Prove, gate, commit** (`PROSE ONLY` on all three; gate `G`)

```bash
cd $WT && git add collab_splats/preproc/frames.py collab_splats/preproc/undistort.py collab_splats/preproc/viz.py && git commit --only collab_splats/preproc/frames.py collab_splats/preproc/undistort.py collab_splats/preproc/viz.py -m "docs(preproc): brief frames, undistort and viz docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: `configs/base.yaml` preproc comments

**Files:** Modify `configs/base.yaml:16-59`

- [ ] **Step 1: Replace the preproc block** (keys and values unchanged; comments one line per key):

```yaml
preproc:
  # Output: images/frame_NNNNNN.png + frames.json beside it
  frame_selection: fps        # fps | uniform | optical_flow
  fps: 2.0                    # fps only: samples per second; fixes spacing, count floats
  min_frames: null            # fps only: floor on the count; null honors fps literally
  max_frames: 300             # frame budget: count for uniform, ceiling otherwise; vggt_omega OOMs above ~300
  n_workers: 4                # quality report: parallel decode ranges; 1 = serial
  undistort: false            # self-calibrate + undistort before images/ is written (canvas grows)
  quality:                    # eligibility gate every sampler picks from
    sharpness_k: 2.0          # MAD z-score cut on log(laplacian); larger keeps more
    on_empty_slot: rescue     # fps only: rescue keeps an all-ineligible slot's sharpest frame, drop skips it
    max_clipped_frac: 0.25    # ceiling on clipped_low_frac + clipped_high_frac; larger keeps more
```

- [ ] **Step 2: Prove the YAML is unchanged as data**

```bash
cd $WT && /opt/venv/reconstruction/bin/python -c "
import subprocess, yaml
old = yaml.safe_load(subprocess.run(['git','show','clean/final:configs/base.yaml'],capture_output=True,text=True).stdout)
new = yaml.safe_load(open('configs/base.yaml'))
print('DATA EQUAL' if old == new else 'DATA CHANGED')"
```

Expected: `DATA EQUAL`. Sanity: temporarily change `fps: 2.0` to `fps: 3.0`, re-run, see `DATA CHANGED`, then revert.

- [ ] **Step 3: Gate, commit**

```bash
cd $WT && git add configs/base.yaml && git commit --only configs/base.yaml -m "docs(config): one line per preproc key; fix the stale alpha=0 crop note

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Round 2 — code

One commit per task. TDD: write or adjust the tests first, see them fail, implement, then gate.

### Task 7: `get_video_info` and `iter_frames` raise; sweep callers

**Files:**
- Modify: `collab_splats/preproc/video.py` (`get_video_info`, `iter_frames`, `extract_frame`)
- Modify: `collab_splats/dashboard/localize.py:479`, `collab_splats/dashboard/app.py:384`, `collab_splats/wrapper/reconstructor.py:288-297`
- Modify: `collab_splats/preproc/qa.py` (`compute_video_quality` probe fallbacks)
- Modify: `collab_splats/preproc/sampling.py` (`total == 0` guard, `or 30.0` in `sample_fps`/`context_indices`)
- Test: `tests/preproc/test_video.py`, `tests/preproc/test_sampling.py`, `tests/preproc/test_qa.py`

- [ ] **Step 1: Rewrite the tests that encoded the silent behavior**

In `tests/preproc/test_video.py`, replace `test_get_video_info_missing_file` and `test_iter_frames_missing_file_yields_nothing` with:

```python
def test_get_video_info_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        get_video_info("/nonexistent/video.mp4")


def test_get_video_info_unprobeable_file_raises(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="cannot probe"):
        get_video_info(broken)


def test_iter_frames_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        list(iter_frames("/nonexistent/video.mp4"))
```

In `tests/preproc/test_sampling.py`, replace `test_sample_uniform_missing_file_returns_empty` with:

```python
def test_sample_uniform_missing_file_raises(clean_report):
    with pytest.raises(FileNotFoundError):
        sample_uniform("/nonexistent/video.mp4", max_frames=6, report=clean_report)
```

Delete `test_context_indices_empty_video_returns_no_indices`: its zero-frame probe can no
longer happen.

In `tests/preproc/test_qa.py`, replace `test_compute_video_quality_reports_unavailable_for_an_undecodable_file`
and `test_compute_video_quality_names_a_missing_file_as_missing` with:

```python
def test_compute_video_quality_raises_on_an_undecodable_file(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"")
    with pytest.raises(ValueError, match="broken.mp4"):
        compute_video_quality(broken)


def test_compute_video_quality_names_a_missing_file_as_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        compute_video_quality(tmp_path / "nope.mp4")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py tests/preproc/test_sampling.py tests/preproc/test_qa.py -q -k "missing or unprobeable or undecodable"`
Expected: FAIL (`DID NOT RAISE`).

- [ ] **Step 3: Implement `get_video_info`** (replace the body; update the docstring's Returns and add Raises):

```python
    """
    Frame count, rate, duration and display size from a container parse.

    Args:
        video_path: source video.

    Returns:
        {"total_frames", "fps", "duration_s", "width", "height"}; width/height are display
        dims (rotation applied).

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the container cannot be probed or reports no frames or no rate.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"get_video_info: {video_path} does not exist")

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]

            # codec_context carries the STORED dimensions, before display rotation
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            width, height = stream.codec_context.width, stream.codec_context.height

            # Frame count from the container; mkv / mpeg-ts omit it, so fall back to duration x rate
            total = int(stream.frames)
            if total == 0 and container.duration:
                total = int(round(container.duration / av.time_base * fps))

            # Decode output is auto-rotated, so 90/270 swaps the display W/H
            if _rotation_degrees(container) in (90, 270):
                width, height = height, width
    except (av.error.FFmpegError, IndexError) as exc:
        raise ValueError(f"get_video_info: cannot probe {video_path}: {exc}") from exc

    if total == 0 or fps == 0:
        raise ValueError(f"get_video_info: cannot probe {video_path}: {total} frames at {fps} fps")

    return {"total_frames": total, "fps": fps, "duration_s": total / fps, "width": width, "height": height}
```

(`IndexError` is what `container.streams.video[0]` raises on an audio-only file.)

- [ ] **Step 4: Implement `iter_frames`.** Change `Args:` `video_path` to `source video.`, add
`Raises:\n        FileNotFoundError: the path does not exist.` and replace the `try/except` open with:

```python
    if not Path(video_path).exists():
        raise FileNotFoundError(f"iter_frames: {video_path} does not exist")

    container = av.open(str(video_path))
```

In `extract_frame`, change `(total and frame_idx >= total)` to `frame_idx >= total`. The
probe now guarantees `total > 0`.

- [ ] **Step 5: Drop the now-dead fallbacks in `qa.py` and `sampling.py`**
  - `qa.py` `compute_video_quality`:
    - `max(1, round(info["fps"] or 1))` → `max(1, round(info["fps"]))`;
    - the log argument `info["fps"] or 0.0` → `info["fps"]`;
    - drop the `%s on the ints so a None from the probe cannot crash` bullet from the comment above it.
  - `qa.py`, the `if not frames["frame_idx"]` branch: the missing-file case is now raised by the probe. Replace the whole branch with a raise for a probeable video that decodes nothing (Task 9 then deletes the `available` key):

    ```python
        if not frames["frame_idx"]:
            raise ValueError(f"video quality: no frames decoded from {video_path}")
        report = {
            "available": True,
            ...  # unchanged
        }
    ```

    Dedent the former `else:` body accordingly.
  - `sampling.py` `sample_fps`:
    - delete `if total == 0: return [], []`;
    - `(info["fps"] or 30.0)` → `info["fps"]`.
  - `sampling.py` `context_indices`:
    - delete `if total == 0: return []`;
    - `info["fps"] or 30.0` → `info["fps"]`.

- [ ] **Step 6: Sweep the external callers**
  - `collab_splats/dashboard/localize.py:479`: `total = int(get_video_info(str(video)).get("total_frames") or 1)` → `total = get_video_info(str(video))["total_frames"]`. It is already inside `try/except`, which logs the failure.
  - `collab_splats/dashboard/app.py:384`: `return int(get_video_info(str(video)).get("total_frames") or 0)` → `return get_video_info(str(video))["total_frames"]`. `run_off_loop`'s `on_error` already reports a raise.
  - `collab_splats/wrapper/reconstructor.py:288-297`: the probe now raises by itself. Replace the block with:

    ```python
            # Probe first: a bad path or undecodable file raises here, before any measuring
            total_frames = get_video_info(str(input_path))["total_frames"]
    ```

    `total_frames` is still used by the zero-selected error message below.

- [ ] **Step 7: Run the targeted tests, then the gate**

Run the Step 2 command. Expected: PASS. Then run the gate `G`. The only allowed count
change is -1 (the deleted `context_indices` empty test) and +1 net in `test_video.py`.
Check that `tests/dashboard/test_app.py:525` still passes: it stubs `{"total_frames": 777}`.

- [ ] **Step 8: Commit**

```bash
cd $WT && P="collab_splats/preproc/video.py collab_splats/preproc/qa.py collab_splats/preproc/sampling.py collab_splats/dashboard/localize.py collab_splats/dashboard/app.py collab_splats/wrapper/reconstructor.py tests/preproc/test_video.py tests/preproc/test_sampling.py tests/preproc/test_qa.py" && git add $P && git commit --only $P -m "fix(preproc)!: probe and decode raise on a missing or unprobeable video

get_video_info no longer returns zeros and iter_frames no longer yields nothing;
callers drop their or-0/or-1/or-30 fallbacks.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: `on_empty_slot` becomes a `sample_fps` kwarg

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (delete `SLOT_POLICIES`, `_split_quality`; `_eligible`; `sample_fps`)
- Modify: `collab_splats/wrapper/reconstructor.py` (`_frames_from_video`, `extract_frames`, `preprocess`)
- Modify: `configs/base.yaml` (move the key)
- Test: `tests/preproc/test_sampling.py:750-791`, `tests/wrapper/test_reconstructor.py:180-240`

- [ ] **Step 1: Tests first.** In `test_sampling.py`:
  - `test_sample_fps_drops_a_condemned_slot_under_the_drop_policy`: `quality={"on_empty_slot": "drop"}` → `on_empty_slot="drop"`.
  - `test_sample_fps_rejects_an_unknown_empty_slot_policy`: `quality={"on_empty_slot": "keep"}` → `on_empty_slot="keep"`.
  - Replace `test_split_quality_keeps_the_policy_out_of_the_threshold_kwargs` with:

    ```python
    def test_quality_block_no_longer_carries_the_slot_policy(monkeypatch, tmp_path):
        """
        on_empty_slot is a sample_fps kwarg; inside quality it is an unknown filter kwarg.
        """
        report = _fps_fixture(monkeypatch, [400.0] * 60)
        assert not hasattr(sampling, "_split_quality") and not hasattr(sampling, "SLOT_POLICIES")
        with pytest.raises(TypeError, match="on_empty_slot"):
            sampling.sample_fps(str(tmp_path / "v.mp4"), fps=3.0, report=report, quality={"on_empty_slot": "drop"})
    ```

  In `test_reconstructor.py`:
  - in the dispatch test's fps `calls` dict, add `"on_empty_slot": "rescue",`;
  - append to `test_extract_frames_forwards_quality_overrides_to_every_sampler`:

    ```python
    def test_extract_frames_forwards_on_empty_slot_to_fps_only(tmp_path, monkeypatch):
        from collab_splats.wrapper import reconstructor as R

        seen = {}

        def fake(name):
            def _f(path, **kwargs):
                seen[name] = kwargs
                return [np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

            return _f

        monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"frames": {}})
        monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
        for name in ("sample_fps", "sample_uniform", "sample_optical_flow"):
            monkeypatch.setattr(R, name, fake(name))
        monkeypatch.setattr(
            R, "preproc_viz", types.SimpleNamespace(plot_photometric=lambda *a, **k: None, plot_motion=lambda *a, **k: None)
        )

        video = tmp_path / "v.mp4"
        video.touch()
        for i, selection in enumerate(("fps", "uniform", "optical_flow")):
            R.extract_frames(video, tmp_path / str(i) / "images", selection, 2.0, 5, 50, on_empty_slot="drop")

        assert seen["sample_fps"]["on_empty_slot"] == "drop"
        assert "on_empty_slot" not in seen["sample_uniform"] and "on_empty_slot" not in seen["sample_optical_flow"]
    ```

- [ ] **Step 2: Run to verify they fail**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py tests/wrapper/test_reconstructor.py -q -k "slot or extract_frames"`
Expected: FAIL (`unexpected keyword argument 'on_empty_slot'`).

- [ ] **Step 3: Implement in `sampling.py`**
  - Delete `SLOT_POLICIES` and `_split_quality`.
  - `_eligible` body becomes:

    ```python
        pool = np.flatnonzero(filter_frame_quality(report, **(quality or {})))
        if pool.size == 0:
            raise ValueError("no eligible frames: the quality filter rejected every frame")
        return pool
    ```

  - `sample_fps`:
    - add the kwarg `on_empty_slot: str = "rescue",` after `quality`;
    - add the Args line `on_empty_slot: "rescue" keeps an all-ineligible slot's sharpest frame; "drop" skips it.`;
    - change the `quality` Args line to `overrides for filter_frame_quality's thresholds.`;
    - change the third docstring bullet to `- a slot with no eligible frame is handled by on_empty_slot; picks never leave the slot`.
  - Replace `_, policy = _split_quality(quality)` with this validation, placed right after the fps check:

    ```python
        if on_empty_slot not in ("rescue", "drop"):
            raise ValueError(f"on_empty_slot must be 'rescue' or 'drop', got {on_empty_slot!r}")
    ```

    Then rename `policy == "drop"` → `on_empty_slot == "drop"`.

- [ ] **Step 4: Implement in `reconstructor.py`**
  - `_frames_from_video`:
    - add `on_empty_slot: str = "rescue",` after `quality`;
    - add the Args line `on_empty_slot: empty-slot policy for frame_selection="fps".`;
    - pass `on_empty_slot=on_empty_slot` in the `sample_fps` call only;
    - add `"on_empty_slot": on_empty_slot,` to `prov`.
  - `extract_frames`:
    - add the parameter `on_empty_slot: str = "rescue",` after `quality`;
    - pass it to `_frames_from_video`.
  - `preprocess`: add `on_empty_slot=pre_cfg["on_empty_slot"],` after `quality=pre_cfg["quality"],`.

- [ ] **Step 5: Move the config key.** In `configs/base.yaml`, delete `on_empty_slot` from
`quality:` and add it after `undistort:` at the `preproc:` level:

```yaml
  on_empty_slot: rescue       # fps only: rescue keeps an all-ineligible slot's sharpest frame, drop skips it
```

Check that no other config sets it:
`cd $WT && grep -rn on_empty_slot configs docs/source collab_splats tests`. Every hit
should be the new form.

- [ ] **Step 6: Run targeted, gate, commit**

Run the Step 2 command (expected PASS), then the gate `G`.

```bash
cd $WT && P="collab_splats/preproc/sampling.py collab_splats/wrapper/reconstructor.py configs/base.yaml tests/preproc/test_sampling.py tests/wrapper/test_reconstructor.py" && git add $P && git commit --only $P -m "refactor(sampling)!: on_empty_slot is a sample_fps kwarg, not a quality key

Deletes _split_quality and SLOT_POLICIES; config key moves to preproc.on_empty_slot.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 9: `sample_fps` inlines its grid; `_spread` and `_sharpest` helpers

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (delete `context_indices` and its section; add `_spread`, `_sharpest`; rewrite `sample_fps`, `sample_uniform`)
- Test: `tests/preproc/test_sampling.py`, `tests/preproc/test_video.py:359-363`

- [ ] **Step 1: Tests first.** In `test_sampling.py`:
  - drop `context_indices` from the import block;
  - delete `test_context_indices_lives_in_sampling`, `test_context_indices_matches_sample_fps_stride`, `test_context_indices_floors_stride_at_one` and `test_context_indices_rejects_a_nonpositive_fps`;
  - replace `test_sample_fps_targets_lie_on_the_context_grid` with the tests below.

  In `test_video.py`, delete `test_video_no_longer_exports_context_indices`.

```python
def test_sample_fps_targets_lie_on_the_stride_grid(tiny_video, clean_report):
    # 30 fps source at fps=2 -> stride 15; every frame is eligible, so picks sit on the grid
    _frames, records = sample_fps(tiny_video, fps=2.0, report=clean_report)
    assert {r["frame_idx"] for r in records} <= set(range(0, 60, 15))


def test_sample_fps_stride_floors_at_one(tiny_video, clean_report):
    _frames, records = sample_fps(tiny_video, fps=1000.0, report=clean_report)
    assert [r["frame_idx"] for r in records] == list(range(60))


def test_context_indices_is_gone():
    assert not hasattr(sampling, "context_indices")


def test_spread_picks_evenly_and_keeps_a_short_pool():
    pool = np.arange(0, 100, 2)
    assert sampling._spread(pool, 3) == [0, 48, 98]  # linspace 24.5 rounds half-to-even
    assert sampling._spread(pool[:4], 10) == [0, 2, 4, 6]


def test_sharpest_breaks_ties_to_the_target():
    laplacian = np.array([1.0, 5.0, 5.0, 2.0])
    assert sampling._sharpest(np.arange(4), 2, laplacian) == 2
    assert sampling._sharpest(np.arange(4), 0, laplacian) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -q -k "stride or spread or sharpest or context"`
Expected: FAIL on `_spread`, `_sharpest` and `test_context_indices_is_gone`.

- [ ] **Step 3: Implement.** Delete the "Selection grid" section and `context_indices`. Add these after `_eligible`:

```python
def _spread(pool: np.ndarray, n: int) -> list[int]:
    """
    n picks evenly spaced by position in pool; all of pool when n >= its size.

    Args:
        pool: ascending eligible source indices.
        n: how many to pick.

    Returns:
        Ascending source indices.
    """
    if n >= pool.size:
        return pool.tolist()
    return pool[np.linspace(0, pool.size - 1, n).round().astype(int)].tolist()


def _sharpest(candidates: np.ndarray, target: int, laplacian: np.ndarray) -> int:
    """
    Candidate with the highest laplacian; ties go to the one nearest target.

    Args:
        candidates: source indices to choose from, non-empty.
        target: the grid index the slot is centered on.
        laplacian: the report's per-frame laplacian column.

    Returns:
        One source index.
    """
    rank = np.lexsort((np.abs(candidates - target), -laplacian[candidates]))
    return int(candidates[rank[0]])
```

In `sample_uniform`, replace the if/else that picks `chosen` with:

```python
    # Spacing is even in pool index, not time: no budget spent in condemned footage
    if pool.size <= max_frames:
        logger.warning("max_frames=%d but only %d eligible frames; keeping the whole pool", max_frames, pool.size)
    chosen = _spread(pool, max_frames)
```

Replace the `sample_fps` body after the two validations with:

```python
    info = get_video_info(str(video_path))
    total, native_fps = info["total_frames"], info["fps"]
    pool = _eligible(report, quality=quality)
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    # Constant-rate grid at fps; stride floors at 1
    targets = list(range(0, total, max(1, int(round(native_fps / fps)))))

    # Clamp the count into [min_frames, max_frames] by re-spreading, never truncating
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))

    if bounded != requested:
        targets = _spread(pool, bounded)
        logger.warning(
            "fps=%.3f wanted %d frames, outside [min_frames=%s, max_frames=%s]; re-spread to "
            "%d frames over the whole video (effective %.3f fps)",
            fps,
            requested,
            min_frames,
            max_frames,
            len(targets),
            native_fps * len(targets) / total,
        )

    # Sharpest eligible frame per slot, ties to the nearest target
    # - the gate is video-wide, so nearest-in-time alone picks an arbitrary survivor
    targets = np.asarray(targets)
    half = max(int(np.median(np.diff(targets)) // 2), 1) if targets.size > 1 else 1
    lo = np.searchsorted(pool, targets - half, side="left")
    hi = np.searchsorted(pool, targets + half, side="right")

    chosen = []
    for target, start, stop in zip(targets.tolist(), lo.tolist(), hi.tolist()):
        if start < stop:
            chosen.append(_sharpest(pool[start:stop], target, laplacian))
        elif on_empty_slot == "rescue":
            # Slot holds no eligible frame: keep its sharpest frame anyway
            window = np.arange(max(target - half, 0), min(target + half + 1, laplacian.size))
            chosen.append(_sharpest(window, target, laplacian))

    # Two targets either side of an excised stretch can land on the same survivor
    chosen = sorted(set(chosen))

    return _decode_selection(video_path, chosen, report=report, on_progress=on_progress, desc="fps sampling")
```

This deletes the dead `pool.size == 0` branch (`_eligible` raises first) and the silent
`window.size == 0` skip. Remove the now-unused `from pathlib import Path` import if nothing else uses it.

- [ ] **Step 4: Run targeted, gate, commit**

Run the Step 2 command (PASS), then the gate `G`. Expected count change: -5 context tests
(-4 in sampling, -1 in video), +5 new.

```bash
cd $WT && P="collab_splats/preproc/sampling.py tests/preproc/test_sampling.py tests/preproc/test_video.py" && git add $P && git commit --only $P -m "refactor(sampling)!: inline the fps grid; share _spread and _sharpest

Deletes context_indices (one caller) and the dead empty-pool branch.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 10: Private `_OpticalFlowSelector`; drop `selected`; `_decode_selection` raises

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (class, `sample_optical_flow`, `_decode_selection`)
- Modify: `collab_splats/preproc/viz.py` (`plot_frame_scores`)
- Test: `tests/preproc/test_sampling.py:12, 70-109, 388-400`, `tests/preproc/test_viz.py:27-63`

- [ ] **Step 1: Tests first.** In `test_sampling.py`, replace the import `OpticalFlowFrameSelector` with nothing (use `sampling._OpticalFlowSelector`). Replace the selector section (lines 73-109) with:

```python
def _selector(**kw):
    return sampling._OpticalFlowSelector(min_disparity=kw.pop("min_disparity", 50.0), rotation_threshold_deg=5.0, **kw)


def test_selector_first_frame_scores_one_without_seeding(noise_gray):
    selector = _selector()
    score, components = selector.score(noise_gray)
    assert score == 1.0
    assert components == {"disparity": 0.0, "rotation": 0.0, "histogram_similarity": 1.0}
    assert selector.keyframe is None


def test_selector_scores_are_normalized():
    selector = _selector()
    rng = np.random.default_rng(2)
    selector.accept((rng.random((240, 320)) * 255).astype(np.uint8))
    for _ in range(5):
        score, _ = selector.score((rng.random((240, 320)) * 255).astype(np.uint8))
        assert 0.0 <= score <= 1.0


def test_selector_identical_frame_scores_low(noise_gray):
    selector = _selector()
    selector.accept(noise_gray)
    score, components = selector.score(noise_gray)
    assert score < 0.3
    assert components["disparity"] < 1.0


def test_selector_score_is_monotonic_in_disparity():
    selector = _selector(motion_weight=1.0)
    assert selector._combine(10.0, 0.0, 0.5) < selector._combine(60.0, 0.0, 0.5) <= 1.0


def test_optical_flow_selector_is_private():
    assert not hasattr(sampling, "OpticalFlowFrameSelector")
    assert "lk_params" not in inspect.signature(sample_optical_flow).parameters


def test_decode_selection_raises_on_a_skipped_frame(monkeypatch, tmp_path):
    report = _fps_fixture(monkeypatch, [400.0] * 60)
    monkeypatch.setattr(sampling, "iter_frames", lambda p, indices=None: [(0, np.zeros((4, 4, 3), np.uint8))])
    with pytest.raises(ValueError, match="skipped"):
        sampling._decode_selection(str(tmp_path / "v.mp4"), [0, 5], report=report, on_progress=None, desc="x")
```

In `test_sample_optical_flow_records_have_source_indices`, remove `"selected",` from the
expected key set.

In `test_viz.py` `_fake_records`, remove the `"selected": i % 5 == 0,` line.

- [ ] **Step 2: Run to verify they fail**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py tests/preproc/test_viz.py -q -k "selector or optical or decode_selection or frame_scores"`
Expected: FAIL (`no attribute '_OpticalFlowSelector'`, `KeyError: 'selected'`).

- [ ] **Step 3: Replace the class** (the whole `OpticalFlowFrameSelector` block):

```python
class _OpticalFlowSelector:
    """
    Streaming keyframe scorer: LK motion and histogram coverage against the last keyframe.

    - score() a candidate; accept() it to make it the new reference
    """

    def __init__(
        self,
        *,
        min_disparity: float,
        rotation_threshold_deg: float,
        lk_window: int = 21,
        lk_levels: int = 3,
        max_corners: int = 1000,
        min_inliers: int = 10,
        hist_bins: int = 64,
        motion_weight: float = 0.6,
    ):
        self.min_disparity = min_disparity
        self.rotation_threshold_deg = rotation_threshold_deg
        self.lk_window = lk_window
        self.lk_levels = lk_levels
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.hist_bins = hist_bins
        self.motion_weight = motion_weight

        # Reference keyframe, set by accept()
        self.keyframe: np.ndarray | None = None
        self.keyframe_pts: np.ndarray | None = None

    def accept(self, gray: np.ndarray) -> None:
        """
        Make gray the reference keyframe and seed its Shi-Tomasi corners.
        """
        self.keyframe = gray.copy()
        self.keyframe_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=8, blockSize=7
        )

    def score(self, gray: np.ndarray) -> tuple[float, dict]:
        """
        Score in [0, 1] against the keyframe, plus its components; 1.0 before any accept.
        """
        if self.keyframe is None:
            return 1.0, {"disparity": 0.0, "rotation": 0.0, "histogram_similarity": 1.0}

        # Motion: mean LK displacement and in-plane rotation of the keyframe corners
        disparity, rotation = 0.0, 0.0
        prev_pts, curr_pts = self._flow(gray)
        if prev_pts is not None:
            disparity = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))
            rotation = self._rotation(prev_pts, curr_pts)

        similarity = self._hist_similarity(gray)
        components = {"disparity": disparity, "rotation": rotation, "histogram_similarity": similarity}
        return self._combine(disparity, rotation, similarity), components

    def _combine(self, disparity: float, rotation: float, similarity: float) -> float:
        """
        Weighted max(translation, rotation) motion plus (1 - similarity) coverage.
        """
        translation_score = min(disparity / max(self.min_disparity, 1e-6), 1.0)
        rotation_score = min(rotation / self.rotation_threshold_deg, 1.0)
        motion = max(translation_score, rotation_score)
        return self.motion_weight * motion + (1.0 - self.motion_weight) * (1.0 - similarity)

    def _flow(self, gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        LK-tracked (keyframe, current) corners; (None, None) below min_inliers.
        """
        if self.keyframe_pts is None or len(self.keyframe_pts) == 0:
            return None, None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.keyframe,
            gray,
            self.keyframe_pts,
            None,
            winSize=(self.lk_window, self.lk_window),
            maxLevel=self.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if curr_pts is None:
            return None, None

        # (N, 1, 2) indexed by an (N, 1) mask -> (M, 2), as the old selector did
        good = status == 1
        if good.sum() < self.min_inliers:
            return None, None

        return self.keyframe_pts[good], curr_pts[good]

    def _rotation(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> float:
        """
        In-plane rotation in degrees from a RANSAC partial-affine fit; 0.0 when unfittable.
        """
        if len(prev_pts) < 4:
            return 0.0

        try:
            M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
        except cv2.error:
            return 0.0

        return 0.0 if M is None else float(np.abs(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))

    def _hist_similarity(self, gray: np.ndarray) -> float:
        """
        Intensity-histogram correlation with the keyframe, clamped to [0, 1].
        """
        h1 = cv2.calcHist([self.keyframe], [0], None, [self.hist_bins], [0, 256])
        h2 = cv2.calcHist([gray], [0], None, [self.hist_bins], [0, 256])
        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.normalize(h2, h2).flatten()
        return float(max(0.0, min(1.0, cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))))
```

- [ ] **Step 4: Parity check against `clean/final`.** The records must be identical minus `selected`:

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python - <<'EOF'
import json, subprocess, sys, tempfile, pathlib
import cv2, numpy as np
from collab_splats.preproc import load_video_quality, sample_optical_flow
d = pathlib.Path(tempfile.mkdtemp())
v = d / "v.mp4"
w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
rng = np.random.default_rng(0)
base = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
for i in range(90):
    w.write(base[i:i + 240, 2 * i:2 * i + 320])
w.release()
rep = load_video_quality(v, d / "r.json", motion_stride=5)
_, new = sample_optical_flow(str(v), report=rep, min_disparity=5.0)
old = subprocess.run(
    ["/opt/venv/reconstruction/bin/python", "-c",
     f"import json;from collab_splats.preproc import sample_optical_flow;"
     f"print(json.dumps(sample_optical_flow('{v}', report=json.load(open('{d}/r.json')), min_disparity=5.0)[1]))"],
    capture_output=True, text=True, cwd="/workspace/collab-splats", env={"PATH": "/usr/bin"}).stdout
old = [{k: v for k, v in r.items() if k != "selected"} for r in json.loads(old)]
print("PARITY" if old == new else f"DIFF {len(old)} vs {len(new)}")
EOF
```

Expected: `PARITY`, with more than 2 records. `cwd` is the main tree, whose editable
install is `clean/final`-equivalent code, so it is the reference. If the main tree's
branch has moved, check it out at `clean/final` in a scratch worktree instead.

- [ ] **Step 5: Rewrite the loop in `sample_optical_flow`**
  - Signature: delete `lk_params` and `feature_params` (and their Args lines).
  - Construct the selector as `_OpticalFlowSelector(min_disparity=min_disparity, rotation_threshold_deg=rotation_threshold_deg)`.
  - Loop body after the pool gate:

    ```python
            # One gray per frame; a frame is accepted once, when it is selected
            gray = analysis_gray(bgr)
            score, components = selector.score(gray)
            if score < select_threshold:
                continue

            selector.accept(gray)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            records.append({"frame_idx": int(idx), "blur_score": float(laplacian[idx]), "score": score, **components})
    ```

- [ ] **Step 6: `_decode_selection` raises.** Replace the `decoded.get(idx)` / `continue` pair with a check before the loop:

```python
    decoded = dict(iter_frames(video_path, indices=list(chosen)))
    missing = sorted(set(chosen) - decoded.keys())
    if missing:
        raise ValueError(f"decode of {video_path} skipped frames {missing[:5]}")
```

The loop then reads `bgr = decoded[idx]`.

- [ ] **Step 7: `plot_frame_scores` stops reading `selected`.** Every record is a kept frame now. Replace the body after the empty check:

```python
    idxs = [d["frame_idx"] for d in frame_scores]
    panels = [
        ([d["disparity"] for d in frame_scores], "Disparity (px)", "steelblue"),
        ([d["rotation"] for d in frame_scores], "Rotation (deg)", "seagreen"),
        ([d["histogram_similarity"] for d in frame_scores], "Histogram similarity", "tomato"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for ax, (values, ylabel, color) in zip(axes, panels):
        ax.plot(idxs, values, color=color, linewidth=0.8, marker="o", markersize=3)
        ax.set_ylabel(ylabel, fontsize=9)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Optical-flow scores of the kept frames (vs the previous keyframe)", fontsize=11)
    fig.tight_layout()
    plt.show()
```

The docstring loses its "selected frames are marked" bullet. `frame_scores` Args becomes `records from sample_optical_flow, one per kept frame.`

- [ ] **Step 8: Run targeted, gate, commit**

Run the Step 2 command (PASS), then the gate `G`.

```bash
cd $WT && P="collab_splats/preproc/sampling.py collab_splats/preproc/viz.py tests/preproc/test_sampling.py tests/preproc/test_viz.py" && git add $P && git commit --only $P -m "refactor(sampling)!: private optical-flow selector with explicit kwargs

combine folds into score, lk/feature dicts become kwargs, each frame is grayed and
accepted once, rotation catches cv2.error only, records drop the constant 'selected',
and _decode_selection raises on a skipped frame.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 11: qa raises instead of `available`; `load_video_quality` owns the write

**Files:**
- Modify: `collab_splats/preproc/qa.py` (`compute_video_quality`, `load_video_quality`)
- Test: `tests/preproc/test_qa.py:384-388, 437-460, 646-660`

- [ ] **Step 1: Tests first**
  - Top-level keys test → `assert set(report) == {"video", "params", "frames", "pairs"}`. Drop the `available` assert.
  - `test_compute_video_quality_writes_json` → rename to `test_load_video_quality_writes_json` with body:

    ```python
        out = tmp_path / "nested" / "video_quality_report.json"
        report = load_video_quality(tiny_video, out, motion_stride=5)
        assert json.loads(out.read_text()) == report
    ```

  - `test_compute_video_quality_serializes_unmatched_pairs_as_null`: `compute_video_quality(path, motion_stride=5, output_path=out)` → `load_video_quality(path, out, motion_stride=5)`.
  - `test_load_video_quality_writes_then_reuses`: `report_path.exists() and first["available"]` → `report_path.exists() and first["frames"]["frame_idx"]`.
  - Add:

    ```python
    def test_compute_video_quality_never_touches_disk():
        assert "output_path" not in inspect.signature(compute_video_quality).parameters
    ```

    (Add `import inspect` to the imports.)

- [ ] **Step 2: Run to verify they fail**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -q -k "keys or json or null or disk or reuses"`
Expected: FAIL (the `available` key is still present; `output_path` is still a parameter).

- [ ] **Step 3: Implement**
  - `compute_video_quality`:
    - delete the `output_path` parameter, its Args line and the trailing `if output_path is not None:` write block;
    - delete `"available": True,` from the report;
    - Returns becomes `{"video": {path, mtime, **get_video_info}, "params": {"motion_stride"}, "frames": {column: list per frame}, "pairs": {column: list per pair}}.`;
    - add `Raises:\n        FileNotFoundError: the video does not exist.\n        ValueError: the video cannot be probed or decodes no frames.`
  - `load_video_quality`, after the reuse branch:

    ```python
        report = compute_video_quality(video_path, motion_stride=motion_stride, workers=workers)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        logger.info("video quality: wrote %s (%.1f kB)", report_path, report_path.stat().st_size / 1000)
        return report
    ```

- [ ] **Step 4: Check that no reader of `available` remains**

Run: `cd $WT && grep -rn '"available"\]\|\["available"\]\|get("available")' collab_splats | grep -v geometry/`
Expected: no output. `tests/dashboard/test_pipeline.py:70,118` and `tests/wrapper/test_reconstructor.py` stub reports that carry an extra `available` key. The key is harmless and nothing reads it, so leave them.

- [ ] **Step 5: Run targeted, gate, commit**

```bash
cd $WT && P="collab_splats/preproc/qa.py tests/preproc/test_qa.py" && git add $P && git commit --only $P -m "refactor(qa)!: report drops the available sentinel; load_video_quality owns the write

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 12: qa tunables thread from `compute_video_quality`; `_ranges`; pair-column loop

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Test: `tests/preproc/test_qa.py:~600-643`

- [ ] **Step 1: Tests first**
  - At line ~605, the direct call becomes:

    ```python
        frames, pairs = _measure_photometry_and_motion(
            (tiny_video, 18, 42, 20, 2), analysis_width=480, blur_h_size=11, n_features=1000, ransac_thresh_px=3.0
        )
    ```

  - In the contiguity test, the stub becomes `def one_row_per_range(args, **tuning):`.
  - Add:

    ```python
    def test_compute_video_quality_threads_its_tuning_and_records_it(tiny_video):
        default = compute_video_quality(tiny_video, motion_stride=5)
        tuned = compute_video_quality(tiny_video, motion_stride=5, analysis_width=240, blur_h_size=3)
        assert tuned["frames"]["blur"] != default["frames"]["blur"]
        assert tuned["params"] == {
            "motion_stride": 5, "analysis_width": 240, "blur_h_size": 3, "n_features": 1000, "ransac_thresh_px": 3.0
        }


    def test_ranges_tile_the_video_once_with_a_lead_in():
        assert qa._ranges(100, workers=1, stride=5) == [(0, None, 0)]
        assert qa._ranges(100, workers=4, stride=5) == [(0, 25, 0), (20, 30, 25), (45, 30, 50), (70, 30, 75)]
        assert qa._ranges(8, workers=4, stride=5) == [(0, None, 0)]
    ```

- [ ] **Step 2: Run to verify they fail** (`-k "tuning or ranges or measure or contiguous"`). Expected: FAIL.

- [ ] **Step 3: Implement**
  - `_measure_photometry_and_motion(args: tuple, *, analysis_width: int, blur_h_size: int, n_features: int, ransac_thresh_px: float)`:
    - `compute_frame_quality(bgr)` → `compute_frame_quality(bgr, analysis_width=analysis_width, blur_h_size=blur_h_size)`;
    - `analysis_gray(bgr)` → `analysis_gray(bgr, width=analysis_width)`;
    - `detect_orb(gray_small)` → `detect_orb(gray_small, n_features=n_features)`;
    - `compute_pair_motion(pending[partner], pending[idx])` → `compute_pair_motion(pending[partner], pending[idx], ransac_thresh_px=ransac_thresh_px)`.
  - Add, above `compute_video_quality`:

    ```python
    def _ranges(total: int, *, workers: int, stride: int) -> list[tuple[int, int | None, int]]:
        """
        (start, count, emit_from) per worker; each range decodes a stride-frame lead-in.

        Args:
            total: frames in the video.
            workers: ranges wanted; a short video gets one.
            stride: pair spacing, the lead-in length.

        Returns:
            Ascending ranges that tile [0, total) exactly once by emit_from.
        """
        if workers == 1 or total <= stride * 2:
            return [(0, None, 0)]

        per = total // workers
        out = []
        for k in range(workers):
            emit_from = k * per
            start = max(emit_from - stride, 0)
            end = total if k == workers - 1 else (k + 1) * per
            out.append((start, end - start, emit_from))
        return out
    ```

  - `compute_video_quality`:
    - signature adds `analysis_width: int = 480, blur_h_size: int = 11, n_features: int = 1000, ransac_thresh_px: float = 3.0`;
    - Args lines: `analysis_width: width blur, laplacian and ORB run at.`, `blur_h_size: Crete-Roffet re-blur kernel width.`, `n_features: ORB feature cap per frame.`, `ransac_thresh_px: inlier threshold for the parallax fits.`
  - Replace the range construction and dispatch with:

    ```python
        tuning = {
            "analysis_width": analysis_width,
            "blur_h_size": blur_h_size,
            "n_features": n_features,
            "ransac_thresh_px": ransac_thresh_px,
        }
        measure = partial(_measure_photometry_and_motion, **tuning)
        ranges = [(str(video_path), start, count, emit_from, stride) for start, count, emit_from in _ranges(total, workers=workers, stride=stride)]

        started = time.perf_counter()
        if len(ranges) == 1:
            results = [measure(ranges[0])]
        else:
            with ProcessPoolExecutor(len(ranges)) as pool:
                results = list(pool.map(measure, ranges))
    ```

    Add `from functools import partial`.
  - Replace the `"pairs": {...}` literal with a block before the report:

    ```python
        # Pair columns; nan -> null, since bare NaN is invalid JSON and 0.0 would read as "no motion"
        pairs = {k: [r[k] for r in pair_rows] for k in ("frame_idx_a", "frame_idx_b", "n_matches")}
        for k in ("translation_px", "parallax"):
            pairs[k] = [None if np.isnan(r[k]) else r[k] for r in pair_rows]
    ```

    Then use `"pairs": pairs` and `"params": {"motion_stride": stride, **tuning}`.

- [ ] **Step 4: Parity.** Default-tuning reports must be byte-identical to `clean/final` except `params`:

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -q
```

Every pre-existing column test must pass unchanged. This is the parity evidence: those
tests pin exact values on `tiny_video`.

- [ ] **Step 5: Gate, commit**

```bash
cd $WT && P="collab_splats/preproc/qa.py tests/preproc/test_qa.py" && git add $P && git commit --only $P -m "refactor(qa): thread analysis_width/blur/ORB/RANSAC tuning from compute_video_quality

Tuning is recorded in report params; ranges split into _ranges.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 13: `frames.py` compression kwarg; delete the zarr migration

**Files:**
- Modify: `collab_splats/preproc/frames.py`
- Delete: `scripts/migrate_frames_zarr.py`
- Test: `tests/preproc/test_frames.py`

- [ ] **Step 1: Tests first**
  - Delete lines 5-7 (`importlib.util`, `Path`), `import zarr`, the `_MIGRATE_PATH`/`_spec`/`migrate_scene` block (lines 15-24) and `test_migrate_converts_a_zarr_store_without_decoding`. Keep `import json`.
  - Replace `test_read_manifest_names_the_migration_script_when_absent` with:

    ```python
    def test_read_manifest_raises_when_absent(tmp_path):
        (tmp_path / "images").mkdir()
        with pytest.raises(FileNotFoundError, match="frames.json"):
            fr.read_manifest(tmp_path / "images")


    def test_write_frames_png_compression_is_a_kwarg(tmp_path):
        fast = fr.write_frames(tmp_path / "a" / "images", _frames(1, 64, 64), _records([0]), {})[0]
        small = fr.write_frames(tmp_path / "b" / "images", _frames(1, 64, 64), _records([0]), {}, png_compression=9)[0]
        assert small.stat().st_size <= fast.stat().st_size
        assert not hasattr(fr, "_PNG_COMPRESSION")
    ```

- [ ] **Step 2: Run to verify they fail** (`pytest tests/preproc/test_frames.py -q`). Expected: FAIL (`unexpected keyword argument 'png_compression'`).

- [ ] **Step 3: Implement**
  - Delete `_PNG_COMPRESSION` and its comment.
  - `write_frames` gains `*, png_compression: int = 1` after `provenance`, with the Args line `png_compression: cv2 PNG level 0-9; higher is smaller and slower.` The imwrite uses `png_compression`.
  - `read_manifest`'s raise message → `f"read_manifest: {path} not found"`. Add `Raises:\n        FileNotFoundError: frames.json is missing.`
  - `git rm scripts/migrate_frames_zarr.py`. Then check nothing else references it:
    `grep -rn migrate_frames_zarr --include=*.py --include=*.md --include=*.yaml . | grep -v docs/superpowers`. Expected: no output.

- [ ] **Step 4: Gate, commit**

```bash
cd $WT && P="collab_splats/preproc/frames.py scripts/migrate_frames_zarr.py tests/preproc/test_frames.py" && git add -A $P && git commit --only $P -m "refactor(frames)!: png_compression kwarg; drop the pre-release frames.zarr migration

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 14: `calibrate_camera` tunables as kwargs

**Files:**
- Modify: `collab_splats/preproc/undistort.py:29-107`
- Test: `tests/preproc/test_undistort.py`

- [ ] **Step 1: Tests first.** Append:

```python
def test_calibrate_camera_floors_are_kwargs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    _write_textured_sequence(images, n=4, width=64, height=48)
    with pytest.raises(ValueError, match="at least 5 images"):
        calibrate_camera(images, min_images=5)
    params = inspect.signature(calibrate_camera).parameters
    assert params["num_threads"].default == 8 and params["min_registered_frac"].default == 0.6
```

Add `import inspect` if it is absent. If `test_calibrate_camera_caps_sift_threads`
monkeypatches `_SIFT_NUM_THREADS`, change it to assert the captured value equals the
`num_threads` default.

- [ ] **Step 2: Run to verify it fails.** Expected: `unexpected keyword argument 'min_images'`.

- [ ] **Step 3: Implement.** Delete the three constants and their comments. New signature:

```python
def calibrate_camera(
    images_dir: Path,
    *,
    max_frames: int = 60,
    min_images: int = 8,
    min_registered_frac: float = 0.6,
    num_threads: int = 8,
) -> pycolmap.Camera:
```

Args lines to add:

```
        min_images: fewer images cannot constrain k1 k2 p1 p2; raise below it.
        min_registered_frac: share of the subset that must register; raise below it.
        num_threads: SIFT threads; the pycolmap default (one per host core) can OOM.
```

Add `Raises:\n        ValueError: fewer than min_images images.\n        RuntimeError: no model, or too few images registered.`
Substitute `min_images`, `min_registered_frac` and `num_threads` at the five use sites.
`wrapper/reconstructor.py:156` calls `calibrate_camera(images_dir)` and needs no change.

- [ ] **Step 4: Gate, commit**

```bash
cd $WT && P="collab_splats/preproc/undistort.py tests/preproc/test_undistort.py" && git add $P && git commit --only $P -m "refactor(undistort): calibration floors and SIFT threads are kwargs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 15: `viz.py` — `_save`, `_mark_selected`, `plot_selection(selections)`, `dpi=` kwargs

**Files:**
- Modify: `collab_splats/preproc/viz.py`
- Test: `tests/preproc/test_viz.py`

- [ ] **Step 1: Tests first.** Replace `test_plot_selection_both_sets` with:

```python
def test_plot_selection_one_panel_per_method():
    plot_selection(100, {"fps": [0, 10, 20], "uniform": [0, 50], "optical_flow": [0, 5, 30]})
    axes = plt.gcf().axes
    assert [ax.get_title().split()[0] for ax in axes] == ["fps", "uniform", "optical_flow"]


def test_plotters_take_dpi(tmp_path):
    report = _fake_video_quality_report()
    lo = plot_photometric(report, tmp_path / "lo", dpi=40)
    hi = plot_photometric(report, tmp_path / "hi", dpi=120)
    assert lo.stat().st_size < hi.stat().st_size
    assert not any(hasattr(viz_module, n) for n in ("_PNG_DPI", "_THUMB_DPI", "_HIST_BINS", "_FIG_WIDTH_IN"))
```

- [ ] **Step 2: Run to verify they fail.** Expected: `plot_selection() takes 1 positional argument` / `unexpected keyword argument 'dpi'`.

- [ ] **Step 3: Implement**
  - Delete the five module constants.
  - `plot_selection`:

    ```python
    def plot_selection(total_frames: int, selections: dict[str, Sequence[int]]) -> None:
        """
        Vertical-line timeline of selected frame indices, one panel per method.

        Args:
            total_frames: source frame count, setting the x extent.
            selections: method name -> selected source frame indices.
        """
        fig, axes = plt.subplots(len(selections), 1, figsize=(12, 2 * len(selections)), squeeze=False)
        for ax, (label, indices) in zip(axes[:, 0], selections.items()):
            ax.vlines(indices, 0, 1, linewidth=1.5, alpha=0.8)
            ax.set_xlim(0, total_frames)
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])
            ax.set_xlabel("Frame index")
            ax.set_title(f"{label}  (n={len(indices)})", fontsize=10)
        fig.tight_layout()
        plt.show()
    ```

  - Helpers, placed after the section header:

    ```python
    def _mark_selected(axes: list[Axes], selected: Sequence[int] | None, fps: float) -> None:
        """
        1 px green line per selected frame on every axis; axvspan would vanish on long videos.
        """
        if selected is None:
            return
        t_sel = np.asarray(list(selected), dtype=float) / fps
        for ax in axes:
            ax.vlines(t_sel, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)


    def _save(fig: Figure, out_dir: str | Path, name: str, *, title: str, dpi: int, tight_grid: bool = False) -> Path:
        """
        Title, save as out_dir/name (creating out_dir) and close the figure.
        """
        fig.suptitle(title)
        fig.tight_layout()
        if tight_grid:
            fig.subplots_adjust(wspace=0)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path
    ```

  - `_marginal_hist(ax_hist, values, *, bins: int = 40)`: use `bins=bins`.
  - `_timeseries_grid(n_panels)`: `figsize=(13.5, 2.8 * n_panels)`.
  - `plot_photometric(..., selected=None, dpi: int = 90)` and `plot_motion(..., selected=None, dpi: int = 90)`: replace each overlay block with `_mark_selected(axes, selected, fps)`. Replace each title/save/close block with:

    ```python
        title = f"{Path(video['path']).name} — {len(report['frames']['frame_idx'])} frames @ {fps:.2f} fps, stride {report['params']['motion_stride']}"
        return _save(fig, out_dir, "photometric.png", title=title, dpi=dpi, tight_grid=True)
    ```

    (Use `"motion.png"` in `plot_motion`.)
  - `plot_frame_extremes(..., n=6, dpi: int = 150)` → `return _save(fig, out_dir, f"extremes-{column}.png", title=..., dpi=dpi)`.
  - `plot_correlation(report, x, y, out_dir, *, dpi: int = 90)` → `return _save(fig, out_dir, f"correlation-{x}-{y}.png", title=..., dpi=dpi)`.
  - Add `dpi: PNG resolution.` to each Args.

- [ ] **Step 4: Check that the PNGs still render.** Run `pytest tests/preproc/test_viz.py tests/wrapper -q -k "png or plot"`. Expected: PASS.

- [ ] **Step 5: Gate, commit**

```bash
cd $WT && P="collab_splats/preproc/viz.py tests/preproc/test_viz.py" && git add $P && git commit --only $P -m "refactor(viz)!: plot_selection takes {method: indices}; shared _save/_mark_selected; dpi kwargs

Fixes the fps set being labeled Uniform.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 16: Tutorial 01 notebook — imports only

**Files:**
- Modify: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`

The notebook was already stale on `clean/final`: it imports `sample_frames` and
`score_frames`, which do not exist. The **tutorial-rework** effort owns rebuilding it. This
task fixes only the calls this plan breaks. It does not rewrite the notebook.

- [ ] **Step 1: Find the broken call sites**

Run: `cd $WT && grep -n "plot_selection\|sample_frames\|score_frames\|OpticalFlowFrameSelector\|context_indices\|output_path\|available" docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`

- [ ] **Step 2: Update the three `plot_selection` calls** (notebook lines ~258, ~486, ~561) with NotebookEdit:
  - `plot_selection(info["total_frames"], fps_indices=fps_indices)` → `plot_selection(info["total_frames"], {"fps": fps_indices})`
  - `plot_selection(info["total_frames"], of_indices=of_indices)` → `plot_selection(info["total_frames"], {"optical_flow": of_indices})`
  - the two-set call → `plot_selection(info["total_frames"], {"fps": fps_indices, "optical_flow": of_indices})`

- [ ] **Step 3: Leave `sample_frames` / `score_frames` alone.** Record them in the final
report as pre-existing breakage owned by tutorial-rework. Do not re-execute the notebook:
it fails on its first import cell with or without this plan.

- [ ] **Step 4: Commit**

```bash
cd $WT && P=docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb && git add $P && git commit --only $P -m "docs(tutorial): plot_selection takes a {method: indices} dict

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 17: `configs/README.md` and `docs/` sweep

**Files:**
- Modify: `configs/README.md` (the `quality` / `on_empty_slot` section near line 303)
- Modify: any `docs/*.md` hit (not `docs/superpowers/`)

- [ ] **Step 1: Find the stale mentions**

Run: `cd $WT && grep -rn "on_empty_slot\|context_indices\|OpticalFlowFrameSelector\|migrate_frames_zarr\|available.: False\|output_path=" configs/README.md docs/*.md docs/source --include=*.md --include=*.rst`

- [ ] **Step 2: Fix each hit**
  - `on_empty_slot` is described as `preproc.on_empty_slot` (not under `quality`).
  - Deleted names are removed.
  - Scene measurements quoted beside a key stay in README. README is user docs, not code, and 017 bans lore only from code.

- [ ] **Step 3: Commit** the touched paths with `docs(config): on_empty_slot moved to preproc level`.

---

## Task 18: Contract — release preproc

**Files:**
- Modify: `tests/test_docstring_contract.py` (`RELEASED`)

- [ ] **Step 1: Flip `RELEASED`.** Set `RELEASED: frozenset[str] = frozenset({"preproc"})`.

- [ ] **Step 2: Run the contract**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/test_docstring_contract.py -q -k preproc`
Expected: all preproc params PASS. Any failure means a hit from `noise.txt` was missed. Fix
it in the source with a `docs(preproc):` or `refactor(preproc):` commit. Never add to
`FIXED_FACTS` for a tunable.

- [ ] **Step 3: Check no semantics/pointcloud param xpasses by accident.** If one does, list it in the report (it is ready for release), but do not add the package.

- [ ] **Step 4: Gate, commit**

```bash
cd $WT && git add tests/test_docstring_contract.py && git commit --only tests/test_docstring_contract.py -m "test(contract): preproc passes the release checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

## Task 19: Graph, changelog, report

- [ ] **Step 1:** `cd $WT && graphify update .`
- [ ] **Step 2: Run the final gate `G`** and diff it against `/tmp/preproc_release/baseline.txt`. Account for every count change by task number.
- [ ] **Step 3: Verify no stale name survives**

```bash
cd $WT && grep -rn "OpticalFlowFrameSelector\|context_indices\|_split_quality\|SLOT_POLICIES\|_PNG_COMPRESSION\|_SIFT_NUM_THREADS\|_PNG_DPI\|migrate_frames_zarr" collab_splats tests configs scripts evals docs/source
```

Expected: no output, except test assertions of absence (`hasattr` checks).

- [ ] **Step 4: Report to the user.** Include:
  - the commit list;
  - baseline vs final counts;
  - the tutorial 01 pre-existing breakage (`sample_frames` / `score_frames`, owned by tutorial-rework);
  - that the branch is not merged. Merging into `clean/final` is the user's call.

---

## Self-review (against the spec)

| Spec item | Task |
|---|---|
| Round 1: `__init__`, video | 2 |
| Round 1: qa | 3 |
| Round 1: sampling (GH010229, VDA claim, mutable-default lecture) | 4 |
| Round 1: frames, undistort, viz | 5 |
| Round 1: base.yaml one line per key, alpha=0 fix | 6 |
| sampling: `_split_quality`/`SLOT_POLICIES` → kwarg, config move, reconstructor | 8 |
| sampling: `_eligible` 3 lines | 8 |
| sampling: inline `context_indices`, dead branch, `_spread`, `_sharpest` | 9 |
| sampling: `fps or 30.0` raise | 7 (probe raises on fps 0), 9 |
| sampling: private selector, kwargs, combine folded, double accept/gray, `cv2.error` | 10 |
| sampling: drop `selected`; `_decode_selection` raises | 10 |
| qa: tunables threaded | 12 |
| qa: `output_path` out; `available` out, raises | 7, 11 |
| qa: `_ranges`, pair-column loop | 12 |
| video: raises; caller sweep (localize, app, reconstructor) | 7 |
| frames: `png_compression`; delete zarr hint + script | 13 |
| undistort: three kwargs | 14 |
| viz: `_save`, `_mark_selected`, `plot_selection(dict)`, `plot_frame_scores`, dpi, bins/sizes | 10, 15 |
| Contract additions, validated on fixtures, known failures for other packages | 1, 18 |
| Tutorial 01 | 16 (call sites only; re-execution blocked by pre-existing breakage) |
| Gate per commit with `__file__` + skip count | Conventions, 0 |

Deviation from the spec: "Tutorial 01 notebook re-executed once at the end" is not
possible. The notebook fails on its first import on `clean/final` already. Task 16 records
this and leaves the rebuild to tutorial-rework.
