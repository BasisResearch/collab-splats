# Scene-id Path-Safe Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen `SCENE_ID_RE` from date-shaped to path-safe so the nine `audiomoth_only_deployments-*` curated dirs are discoverable/processable, and make skipped dirs visible in the log.

**Architecture:** One shared regex (`collab_splats/remote/sources.py`) already gates both discovery (`list_scenes`, `list_processed_scenes`) and explicit ids (`run_pipeline_remote.py`). Change the pattern in one place, add a skip-warning in `list_scenes`, update user-facing wording. Spec: `docs/superpowers/specs/2026-08-17-scene-id-path-safe-design.md`.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest (`-p no:randomly`), existing fake-rclone fixtures in `tests/remote/test_sources.py` (`_fake_run`, `_dirs`, `_files`, `_FakeClient`).

**Trap:** existing fixtures use `"tmp"` as the "dir that fails SCENE_ID_RE" negative. Under the new regex `tmp` is a VALID scene id (correct — it is path-safe). Both fixtures must switch their negative to `.cache` (leading dot → still rejected).

---

### Task 1: Widen SCENE_ID_RE (TDD)

**Files:**
- Modify: `collab_splats/remote/sources.py:29-33` (comment + pattern)
- Test: `tests/remote/test_sources.py` (new test + 2 fixture updates)

- [ ] **Step 1: Add `SCENE_ID_RE` to the test file's import and write the failing test**

In `tests/remote/test_sources.py`, extend the existing `from collab_splats.remote.sources import ...` line with `SCENE_ID_RE`. Then add after `test_list_scenes_returns_dirs_at_bucket_root`:

```python
def test_scene_id_re_is_path_safety_not_date_shape():
    """The regex guards the output-path join; the YYYY_MM_DD shape is convention, not contract."""
    accepted = [
        # legacy convention
        "2026_07_20-birds-C0043",
        # all nine live audiomoth curated dirs (verified against the bucket 2026-08-17)
        "audiomoth_only_deployments-20260810_20260831-boston_frontageroad_tracks-splat_videos-GH010250",
        "audiomoth_only_deployments-20260810_20260831-boston_frontageroad_tracks-splat_videos-GH010251",
        "audiomoth_only_deployments-20260810_20260831-boston_ringerpark_east-splat_videos-GH010248",
        "audiomoth_only_deployments-20260810_20260831-boston_ringerpark_west-splat_videos-GH010247",
        "audiomoth_only_deployments-20260810_20260831-boston_ringerpark_west-splat_videos-GH010249",
        "audiomoth_only_deployments-20260817_20260824-boston_alley443_trees-splat_videos-GH010258",
        "audiomoth_only_deployments-20260817_20260824-boston_charlesgateeast_riverbank-splat_videos-GH010259",
        "audiomoth_only_deployments-20260817_20260824-boston_publicgarden_maintenance-splat_videos-GH010255",
        "audiomoth_only_deployments-20260817_20260824-boston_publicgarden_maintenance-splat_videos-GH010257",
    ]
    for name in accepted:
        assert SCENE_ID_RE.match(name), name
    # One path segment, no leading dot (kills "..", ".", hidden dirs), no leading "-" (argv-safe)
    rejected = ["../x", "..", ".", ".hidden", "a/b", "", "-flag"]
    for name in rejected:
        assert not SCENE_ID_RE.match(name), name
```

- [ ] **Step 2: Run it, verify it fails on the audiomoth names**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/remote/test_sources.py::test_scene_id_re_is_path_safety_not_date_shape -v -p no:randomly
```

Expected: FAIL — `AssertionError: audiomoth_only_deployments-...GH010250`.

- [ ] **Step 3: Replace the pattern and its comment in `sources.py`**

Replace lines 29-33:

```python
# The scene id IS the curated dir name and is joined onto a local output path by the remote
# driver, so this regex enforces PATH SAFETY, not a naming shape: one path segment (no "/"),
# no leading "." (kills "..", ".", hidden dirs), no leading "-" (argv-safe). The historical
# YYYY_MM_DD-PARENTFOLDER-VIDEONAME shape is a convention some dirs follow, not the contract —
# audiomoth deployments ship names like audiomoth_only_deployments-<range>-<site>-...-<video>.
# Public because discovery and the driver's explicit-id check must accept exactly the same ids.
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
```

- [ ] **Step 4: Fix the two fixtures whose negative dir (`"tmp"`) is now a valid id**

In `test_list_scenes_returns_dirs_at_bucket_root`: change `"tmp"` → `".cache"` in the `_dirs(...)` call and update its comment:

```python
    # Negative fixtures isolate each filter: ".cache" is a dir that fails SCENE_ID_RE (leading
    # dot), and the scene-named .mp4 matches SCENE_ID_RE but is not a dir. Unsorted input pins
    # the sort.
```

Same edit in `test_list_processed_scenes_reads_processed_bucket` (`"tmp"` → `".cache"`; its comment already says "a non-scene dir", leave it).

- [ ] **Step 5: Run the remote suite, verify green**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/remote/ -v -p no:randomly
```

Expected: all pass (new test + existing ~sources/rerun tests).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/remote/sources.py tests/remote/test_sources.py
git commit -m "fix(remote): SCENE_ID_RE enforces path safety, not date shape — audiomoth curated dirs now discoverable"
```

---

### Task 2: Log skipped non-scene dirs in list_scenes (TDD)

**Files:**
- Modify: `collab_splats/remote/sources.py` (`list_scenes._produce`, ~line 268)
- Test: `tests/remote/test_sources.py`

- [ ] **Step 1: Write the failing test** (place next to `test_list_scenes_logs_when_the_curated_bucket_is_empty`)

```python
def test_list_scenes_warns_about_skipped_non_scene_dirs(monkeypatch, caplog):
    """A dir that fails SCENE_ID_RE must be named in the log — the audiomoth folders sat
    undiscoverable for weeks because the filter dropped them silently."""
    _fake_run(
        monkeypatch,
        listings={f"collab-data:{CURATED_BUCKET}": _dirs("2026_07_20-birds-C0043", ".cache")},
    )
    with caplog.at_level(logging.WARNING, logger="collab_splats.remote.sources"):
        assert SceneSource(_FakeClient()).list_scenes() == ["2026_07_20-birds-C0043"]
    assert any(".cache" in r.getMessage() for r in caplog.records), "skipped dir must be logged"
```

- [ ] **Step 2: Run it, verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/remote/test_sources.py::test_list_scenes_warns_about_skipped_non_scene_dirs -v -p no:randomly
```

Expected: FAIL on the `caplog.records` assertion (list_scenes emits nothing for skipped dirs).

- [ ] **Step 3: Implement in `list_scenes._produce`**

Replace the current `_produce` body's return line:

```python
        def _produce():
            entries = self._lsjson(CURATED_BUCKET, "")
            if not entries:
                logger.info("%s listed empty — no curated scenes to process", CURATED_BUCKET)
                return []
            dirs = [e["Name"] for e in entries if e.get("IsDir")]
            # A present-but-unmatched dir is exactly how a naming-convention drift looks; name it,
            # or the next batch of curated scenes silently vanishes from --all like audiomoth did.
            skipped = sorted(d for d in dirs if not SCENE_ID_RE.match(d))
            if skipped:
                logger.warning("skipping non-scene dirs in %s: %s", CURATED_BUCKET, ", ".join(skipped))
            return sorted(d for d in dirs if SCENE_ID_RE.match(d))
```

(`list_processed_scenes` stays as-is per spec: its non-matching dirs are stray uploads, and it inherits the widened constant.)

- [ ] **Step 4: Run remote suite, verify green**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/remote/ -v -p no:randomly
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/remote/sources.py tests/remote/test_sources.py
git commit -m "feat(remote): list_scenes names the curated dirs it skips"
```

---

### Task 3: User-facing wording — driver error, docstring, README

**Files:**
- Modify: `docs/examples/run_pipeline_remote.py` (docstring line + comment + `parser.error` text, ~lines 249-254)
- Modify: `configs/README.md` (~line 84)

- [ ] **Step 1: Driver — comment + error message**

Replace:

```python
    # --all ids come from list_scenes, which filters on SCENE_ID_RE; explicit ids skipped that filter
    # entirely and are joined straight onto --output-root, so "../x" would write outside it. One
    # shared regex, or discovery could yield an id an explicit re-run of the same scene then refuses.
    bad = [s for s in args.scenes if not SCENE_ID_RE.match(s)]
    if bad:
        parser.error(f"not curated scene ids (expected YYYY_MM_DD-PARENT-VIDEO): {', '.join(bad)}")
```

with:

```python
    # --all ids come from list_scenes, which filters on SCENE_ID_RE; explicit ids skipped that filter
    # entirely and are joined straight onto --output-root, so "../x" would write outside it. One
    # shared regex, or discovery could yield an id an explicit re-run of the same scene then refuses.
    bad = [s for s in args.scenes if not SCENE_ID_RE.match(s)]
    if bad:
        parser.error(f"not safe scene ids (one path segment, no leading '.' or '-'): {', '.join(bad)}")
```

- [ ] **Step 2: Driver docstring**

Replace the docstring line:

```
Scene ids are the curated directory names: YYYY_MM_DD-PARENTFOLDER-VIDEONAME.
```

with:

```
Scene ids are the curated directory names (e.g. YYYY_MM_DD-PARENTFOLDER-VIDEONAME by
convention; any flat path-safe name works).
```

- [ ] **Step 3: README wording** — in `configs/README.md` "Remote scenes" section, replace:

```markdown
Scene ids are the curated directory names: `YYYY_MM_DD-PARENTFOLDER-VIDEONAME`.
```

with:

```markdown
Scene ids are the curated directory names. `YYYY_MM_DD-PARENTFOLDER-VIDEONAME` is the common
convention, but any flat path-safe name is a valid scene (e.g. the
`audiomoth_only_deployments-...` deployments); dirs that fail the safety filter are named in
the driver's log.
```

- [ ] **Step 4: Sanity — driver still imports and rejects an unsafe id**

```bash
/opt/venv/reconstruction/bin/python docs/examples/run_pipeline_remote.py --output-root /tmp/x '../evil' 2>&1 | tail -1
```

Expected: `... error: not safe scene ids (one path segment, no leading '.' or '-'): ../evil`

- [ ] **Step 5: Full remote suite once more, then commit**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/remote/ -q -p no:randomly
git add docs/examples/run_pipeline_remote.py configs/README.md
git commit -m "docs(remote): scene-id wording — date shape is convention, path safety is the contract"
```

---

### Task 4: Live discovery check (read-only)

**Files:** none (verification only)

- [ ] **Step 1: Confirm all nine audiomoth scenes are now discoverable**

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
from collab_splats.remote.sources import SceneSource
scenes = SceneSource().list_scenes()
am = [s for s in scenes if s.startswith("audiomoth")]
print(len(am), "audiomoth scenes discoverable")
for s in am: print(" ", s)
EOF
```

Expected: `9 audiomoth scenes discoverable` + the nine names. (Actual reconstruction runs stay a separate human-watched step — compute + billed.)
