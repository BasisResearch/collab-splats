# MapAnything torch ≤2.3 Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Patch `mapanything==1.1`'s torch≥2.4-only `tensor.any(dim=(1, 3))` reduction at runtime so `MapAnythingCreator.run_inference()` works on the env's pinned `torch==2.1.2+cu118`.

**Architecture:** Add a small idempotent helper, `_patch_mapanything_torch_compat`, at module scope in `collab_splats/pointcloud/feedforward.py`. The helper calls `inspect.getsource` on `mapanything.utils.wai.intersection_check.frustum_intersection_check`, replaces `batch_intersect.any(dim=(1, 3))` with `batch_intersect.any(dim=3).any(dim=1)`, `exec`s the patched source against the original module's namespace, and reassigns the function on the module. `MapAnythingCreator._load_model` calls the helper before constructing the model. Patch is intentionally temporary; sunset trigger is the env moving to `torch>=2.4`.

**Tech Stack:** Python 3.10, PyTorch 2.1.2+cu118, mapanything 1.1, pytest, nerfstudio conda env at `/opt/conda/envs/nerfstudio/bin/python`.

**Spec:** `worklog/history/specs/2026-05-03-mapanything-torch-compat.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `collab_splats/pointcloud/feedforward.py` | feedforward creators | add `_patch_mapanything_torch_compat`; call from `MapAnythingCreator._load_model`; replace existing punt comment with one-line pointer |
| `tests/pointcloud/test_mapanything_creator.py` | MapAnything creator unit tests | add patch helper tests (idempotency, source mutation assertion, signature drift detection) |
| `setup_feedforward.sh` | env install script | remove the "MapAnything will fail on this torch" note in the header block |
| `worklog/history/specs/2026-05-03-feedforward-env-debug.md` | predecessor brief | mark Bug 2 RESOLVED with pointer to this plan and the compat spec |
| `docs/pointcloud/loop_closure_eval.ipynb` | LC eval notebook (optional) | parametrize backend cell to also run `mapanything` once smoke passes |

Tests for the patch helper live alongside existing `MapAnythingCreator` tests in `test_mapanything_creator.py` rather than in a new `test_mapanything.py` (spec mentioned the latter as a working name; consolidating in the existing file keeps mapanything coverage in one place).

---

## Task 1: Add `_patch_mapanything_torch_compat` helper (TDD)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py` (imports near top + new helper at module scope after existing module-level helpers like `_raw_to_world_points`)
- Test: `tests/pointcloud/test_mapanything_creator.py`

- [ ] **Step 1: Write the failing test for source mutation**

Add at the bottom of `tests/pointcloud/test_mapanything_creator.py`:

```python
import inspect


def test_patch_mapanything_torch_compat_replaces_multi_dim_any():
    pytest.importorskip("mapanything")
    from collab_splats.pointcloud.feedforward import _patch_mapanything_torch_compat
    import mapanything.utils.wai.intersection_check as ic

    _patch_mapanything_torch_compat()
    src = inspect.getsource(ic.frustum_intersection_check)
    assert "any(dim=(1, 3))" not in src
    assert "any(dim=3).any(dim=1)" in src


def test_patch_mapanything_torch_compat_is_idempotent():
    pytest.importorskip("mapanything")
    from collab_splats.pointcloud.feedforward import _patch_mapanything_torch_compat
    import mapanything.utils.wai.intersection_check as ic

    _patch_mapanything_torch_compat()
    first = ic.frustum_intersection_check
    _patch_mapanything_torch_compat()
    _patch_mapanything_torch_compat()
    second = ic.frustum_intersection_check
    assert first is second
    assert getattr(second, "_torch_compat_patched", False) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud/test_mapanything_creator.py::test_patch_mapanything_torch_compat_replaces_multi_dim_any tests/pointcloud/test_mapanything_creator.py::test_patch_mapanything_torch_compat_is_idempotent -v
```

Expected: both fail with `ImportError: cannot import name '_patch_mapanything_torch_compat'`.

- [ ] **Step 3: Implement the helper**

In `collab_splats/pointcloud/feedforward.py`, locate the imports block at the top (lines 1–22). Confirm `inspect` and `textwrap` are not already imported; if not, add to the top-of-file imports:

```python
import inspect
import textwrap
```

Then add this helper at module scope **after** `_raw_to_world_points` and **before** `# Concrete creators` section header (~line 580):

```python
def _patch_mapanything_torch_compat() -> None:
    """Replace mapanything's torch>=2.4 multi-dim any() with sequential calls.

    mapanything==1.1's intersection_check uses ``tensor.any(dim=(1, 3))`` which
    requires PyTorch 2.4+. The nerfstudio env is pinned to 2.1.2+cu118 (gsplat-rade
    CUDA kernels). We rewrite the function in place at import time. Idempotent.

    TEMPORARY BRIDGE — see "Sunset Path" in
    worklog/history/specs/2026-05-03-mapanything-torch-compat.md.
    Delete this helper once the env moves to torch>=2.4.
    """
    import mapanything.utils.wai.intersection_check as ic

    target = ic.frustum_intersection_check
    if getattr(target, "_torch_compat_patched", False):
        return

    src = textwrap.dedent(inspect.getsource(target))
    patched_src = src.replace(
        "batch_intersect.any(dim=(1, 3))",
        "batch_intersect.any(dim=3).any(dim=1)",
    )
    if patched_src == src:
        raise RuntimeError(
            "mapanything intersection_check.py no longer matches expected pattern "
            "for torch<2.4 compat patch. Re-inspect "
            "mapanything.utils.wai.intersection_check.frustum_intersection_check "
            "and update _patch_mapanything_torch_compat."
        )

    ns: dict = {}
    exec(patched_src, ic.__dict__, ns)
    patched = ns["frustum_intersection_check"]
    patched._torch_compat_patched = True
    ic.frustum_intersection_check = patched
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud/test_mapanything_creator.py::test_patch_mapanything_torch_compat_replaces_multi_dim_any tests/pointcloud/test_mapanything_creator.py::test_patch_mapanything_torch_compat_is_idempotent -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full pointcloud suite to check no regression**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud -q
```

Expected: all 48 existing tests still pass + 2 new tests pass = 50 PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_mapanything_creator.py
git commit -m "$(cat <<'EOF'
feat(pointcloud): add _patch_mapanything_torch_compat for torch<2.4

Patch mapanything==1.1's frustum_intersection_check at import time to
rewrite tensor.any(dim=(1,3)) — a torch>=2.4 idiom — into chained
.any(dim=3).any(dim=1). Idempotent; raises if upstream source drifts so
the patch fails loud rather than silently no-op.

Bridge until env moves to torch>=2.4; see
worklog/history/specs/2026-05-03-mapanything-torch-compat.md sunset section.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Drift-detection test (defensive)

**Files:**
- Test: `tests/pointcloud/test_mapanything_creator.py`

The drift case (upstream source no longer contains the target string) is critical: if it ever silently reverts, MapAnything inference will start raising `TypeError` on torch 2.1 again. Lock it down.

- [ ] **Step 1: Write the failing test**

Add to `tests/pointcloud/test_mapanything_creator.py`:

```python
def test_patch_mapanything_torch_compat_raises_on_source_drift(monkeypatch):
    pytest.importorskip("mapanything")
    from collab_splats.pointcloud import feedforward
    import mapanything.utils.wai.intersection_check as ic

    # Force a fresh patch attempt by clearing the marker, then swap the
    # function with a stub that does NOT contain the target string.
    def stub_func():
        return None

    monkeypatch.setattr(ic, "frustum_intersection_check", stub_func)
    with pytest.raises(RuntimeError, match="no longer matches expected pattern"):
        feedforward._patch_mapanything_torch_compat()
```

- [ ] **Step 2: Run to verify it passes**

Already implemented in Task 1; this test exercises the drift branch.

Run:
```bash
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud/test_mapanything_creator.py::test_patch_mapanything_torch_compat_raises_on_source_drift -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/pointcloud/test_mapanything_creator.py
git commit -m "$(cat <<'EOF'
test(pointcloud): cover source-drift branch of mapanything torch compat patch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire helper into `MapAnythingCreator._load_model`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py:612-628` (current `_load_model` body)
- Test: `tests/pointcloud/test_mapanything_creator.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/pointcloud/test_mapanything_creator.py`:

```python
def test_load_model_invokes_torch_compat_patch():
    pytest.importorskip("mapanything")
    from collab_splats.pointcloud import feedforward

    creator = MapAnythingCreator()
    fake_model = MagicMock()

    with patch.object(feedforward, "_patch_mapanything_torch_compat") as mock_patch, \
         patch("mapanything.models.MapAnything") as mock_ma:
        mock_ma.from_pretrained.return_value = fake_model
        fake_model.to.return_value = fake_model
        creator._load_model(device="cpu")

    mock_patch.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud/test_mapanything_creator.py::test_load_model_invokes_torch_compat_patch -v
```

Expected: FAIL — `_patch_mapanything_torch_compat` not called from `_load_model`.

- [ ] **Step 3: Replace `_load_model` body**

Edit `collab_splats/pointcloud/feedforward.py:612-628`. Replace the existing method (including the multi-line punt comment) with:

```python
    def _load_model(self, device: str) -> Any:
        try:
            from mapanything.models import MapAnything
        except ImportError as e:
            raise ImportError(
                "MapAnything required. "
                "pip install git+https://github.com/facebookresearch/map-anything.git"
            ) from e
        # In-tree workaround: see _patch_mapanything_torch_compat (torch<2.4).
        _patch_mapanything_torch_compat()
        model = MapAnything.from_pretrained(self.model_name)
        model = model.to(device)
        model.eval()
        return model
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud/test_mapanything_creator.py::test_load_model_invokes_torch_compat_patch -v
```

Expected: PASS.

- [ ] **Step 5: Run full pointcloud suite**

```bash
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud -q
```

Expected: all tests PASS (51 now: 48 original + 3 new).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_mapanything_creator.py
git commit -m "$(cat <<'EOF'
feat(pointcloud): call torch compat patch from MapAnythingCreator._load_model

Replace multi-line punt comment with one-line pointer + active patch call.
MapAnythingCreator.run_inference() now works on torch 2.1.2+cu118.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update install-script note

**Files:**
- Modify: `setup_feedforward.sh:16-18`

- [ ] **Step 1: Read current note**

Run:
```bash
sed -n '14,20p' /workspace/collab-splats/setup_feedforward.sh
```

Expected output (lines 14–20):
```
#
# Known runtime limitation: MapAnythingCreator.run_inference() will fail on this
# torch version because mapanything==1.1 uses multi-dim `tensor.any(dim=(1,3))`
# (PyTorch >=2.4). Use VGGTXCreator. See feedforward.py MapAnythingCreator._load_model.
#
```

- [ ] **Step 2: Replace the limitation note**

Replace lines 16–18 in `setup_feedforward.sh` with:

```bash
# Compat note: mapanything==1.1 uses `tensor.any(dim=(1,3))` (PyTorch >=2.4).
# This env pins torch 2.1.2+cu118 (gsplat-rade kernels). The torch<2.4 fallback
# is patched at runtime by collab_splats.pointcloud.feedforward
# ._patch_mapanything_torch_compat (called from MapAnythingCreator._load_model).
# Sunset on torch>=2.4 — see worklog/history/specs/2026-05-03-mapanything-torch-compat.md.
```

- [ ] **Step 3: Verify the script is still syntactically valid**

Run:
```bash
bash -n /workspace/collab-splats/setup_feedforward.sh
```

Expected: exit 0, no output.

- [ ] **Step 4: Commit**

```bash
git add setup_feedforward.sh
git commit -m "$(cat <<'EOF'
docs(setup_feedforward): replace MapAnything torch-fail note with compat pointer

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Mark Bug 2 resolved in env-debug spec

**Files:**
- Modify: `worklog/history/specs/2026-05-03-feedforward-env-debug.md`

- [ ] **Step 1: Locate Bug 2 section**

Run:
```bash
grep -n "Bug 2\|bug 2" /workspace/collab-splats/worklog/history/specs/2026-05-03-feedforward-env-debug.md
```

Note the line numbers of the Bug 2 heading and the next `##` heading. The block between them is the Bug 2 section.

- [ ] **Step 2: Prepend a RESOLVED banner**

Insert immediately after the Bug 2 heading line:

```markdown
**Status:** RESOLVED 2026-05-03 — fixed in `collab_splats/pointcloud/feedforward.py:_patch_mapanything_torch_compat`. See follow-up spec [`2026-05-03-mapanything-torch-compat.md`](2026-05-03-mapanything-torch-compat.md) for design and sunset plan.
```

- [ ] **Step 3: Commit**

```bash
git add worklog/history/specs/2026-05-03-feedforward-env-debug.md
git commit -m "$(cat <<'EOF'
docs(spec): mark feedforward env-debug Bug 2 RESOLVED

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: GPU smoke test for full LC inference

**Files:**
- Test: `tests/pointcloud/test_mapanything_creator.py`

This is a heavyweight, gpu-marked, opt-in test mirroring the spec's "smoke test" reproducer. It loads MapAnything weights and runs end-to-end against the bicycle scene.

- [ ] **Step 1: Add the smoke test**

Append to `tests/pointcloud/test_mapanything_creator.py`:

```python
@pytest.mark.gpu
def test_mapanything_run_inference_loop_closure_smoke(tmp_path):
    pytest.importorskip("mapanything")
    pytest.importorskip("torch")
    bicycle = Path("/workspace/bicycle/images_4")
    if not bicycle.exists():
        pytest.skip(f"{bicycle} not available on this host")

    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig

    creator = MapAnythingCreator(
        camera_model="PINHOLE",
        enable_loop_closure=True,
        loop_closure_config=LoopClosureConfig(),
    )
    creator.load_model()
    creator.setup_inference(bicycle)
    result = creator.run_inference()
    assert result is not None
```

- [ ] **Step 2: Run the smoke test on a GPU host**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_run_inference_loop_closure_smoke -v -s -m gpu
```

Expected: PASS. The previous failure mode (`TypeError: any() received an invalid combination of arguments — got (dim=tuple, )`) must not appear.

If the test environment has no GPU, this test is skipped via `@pytest.mark.gpu`. The unit-level patch tests (Task 1, 2) cover correctness without GPU.

- [ ] **Step 3: Commit**

```bash
git add tests/pointcloud/test_mapanything_creator.py
git commit -m "$(cat <<'EOF'
test(pointcloud): gpu smoke test for MapAnything inference with loop closure

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Notebook backend parametrization (optional)

**Files:**
- Modify: `docs/pointcloud/loop_closure_eval.ipynb`

Skip this task if the notebook is not part of the deliverable for this branch. Only proceed if Tasks 1–6 are green and a follow-up smoke against `BACKEND='mapanything'` is desired.

- [ ] **Step 1: Inspect current backend dispatch cell**

Run:
```bash
/opt/conda/envs/nerfstudio/bin/jupyter nbconvert --to script docs/pointcloud/loop_closure_eval.ipynb --stdout | grep -n "BACKEND\|MapAnythingCreator\|VGGTXCreator" | head -20
```

Locate the cell that branches on `BACKEND`.

- [ ] **Step 2: Run the notebook headlessly with mapanything backend**

Run:
```bash
BACKEND=mapanything /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
  --to notebook --execute docs/pointcloud/loop_closure_eval.ipynb \
  --output /tmp/loop_closure_eval.mapanything.ipynb
```

Expected: exit 0. If the notebook hard-codes `BACKEND='vggtx'`, edit the BACKEND cell to read `import os; BACKEND = os.environ.get("BACKEND", "vggtx")` and re-run.

- [ ] **Step 3: Commit**

```bash
git add docs/pointcloud/loop_closure_eval.ipynb
git commit -m "$(cat <<'EOF'
docs(loop_closure_eval): allow BACKEND env override for headless mapanything runs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Verification

After all tasks land, run end-to-end verification:

```bash
# Unit suite
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud -q

# (GPU host only) Smoke test
/opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud/test_mapanything_creator.py -q -m gpu

# Spec acceptance reproducer
/opt/conda/envs/nerfstudio/bin/python -c "
from pathlib import Path
from collab_splats.pointcloud.feedforward import MapAnythingCreator
from collab_splats.pointcloud.loop_closure import LoopClosureConfig
c = MapAnythingCreator(camera_model='PINHOLE', enable_loop_closure=True,
                      loop_closure_config=LoopClosureConfig())
c.load_model()
c.setup_inference(Path('/workspace/bicycle/images_4'))
c.run_inference()
print('MA_LC_INFERENCE_OK')
"
```

Expected:
- All pointcloud unit tests pass (48 prior + 4 new = 52).
- All retrieval tests still pass (existing 3).
- Smoke test prints `MA_LC_INFERENCE_OK`.
- VGGTX path regression check (existing tests) still green.
