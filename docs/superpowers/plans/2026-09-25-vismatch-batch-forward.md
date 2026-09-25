# vismatch fork setup + batched `forward()` (1a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clone the BasisResearch vismatch fork, record upstream's test baseline, then land a base-only
`feat/batch-forward` branch where `BaseMatcher.forward()` accepts a batch of pairs and returns a list of
result dicts, with zero model-file changes.

**Architecture:** Input normalization moves into a new `utils.to_tensor_images()` that returns a list of
`(3, H, W)` tensors plus a `batched` flag. `BaseMatcher.forward()` splits into normalize → run `_forward`
(once per pair by default, or once on stacked `(B, 3, H, W)` tensors when a model sets
`supports_batches = True`) → per-pair `_postprocess()` (today's checks, numpy conversion, out-of-bounds
filter, RANSAC, dict assembly, moved verbatim). Single-pair input keeps returning one dict, so every
existing caller and test is untouched.

**Tech Stack:** Python 3.10 (upstream CI version), PyTorch, numpy, pytest + pytest-timeout, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-25-vismatch-fork-design.md` (steps 0 and 1a).

---

## Ground rules for this plan

- **Repo:** all code work happens in `/workspace/vismatch` (the fork), NOT in collab-splats. The only
  collab-splats edits are this plan's checkboxes and the final CHANGELOG/CLAUDE.md step.
- **Style inside the fork is upstream's, not collab-splats'.** Ruff (line length 120), upstream docstring
  shape (`Args:` with types in parentheses, as in `base_matcher.py`), plain imperative commit subjects
  like upstream's history (`Add upal matcher`, `Fix ransac_conf ...`). Do NOT apply collab-splats'
  docstring contract, `########` dividers, or conventional-commit prefixes inside the fork.
- **Python:** `/opt/venv/vismatch/bin/python` (created in Task 2). Never the base-shell `python` and never
  `/opt/venv/reconstruction` (vismatch is not installed there and its deps differ).
- **Commits in the fork end with:**
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- **Pushing is outward-facing:** Task 9 pushes to `origin` only after the user confirms. Never push to
  `upstream`. Never open the upstream PR yourself (`gh` is not installed; the user opens it in the browser).
- **pytest exit codes:** never pipe pytest into `tail`/`head` (eats the exit code). Use `tee` to a log and
  check `${PIPESTATUS[0]}`, or read the junit XML.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `vismatch/utils.py` | Modify (add after `to_tensor_image`, ~L491) | `to_tensor_images()`: single image or batch → list of `(3,H,W)` tensors + `batched` flag |
| `vismatch/base_matcher.py` | Modify | `supports_batches` attr; `forward()` normalize/dispatch; new `_postprocess()` (moved body) and `_split_batch_outputs()`; `extract()` batch-aware |
| `tests/test_batch_forward.py` | Create | Mock-matcher tests for every batch path; no weight downloads |
| `tests/test_utils.py` | Modify | `to_tensor_images()` unit tests |
| `README.md` | Modify (Python API block, ~L127-136) | Batch usage snippet |
| `docs/source/quickstart.md` | Modify (add section after "Keypoint Extraction") | Batch usage snippet |

---

### Task 1: Clone the fork and wire remotes

**Files:** none (repo setup)

- [ ] **Step 1: Clone recursively**

```bash
cd /workspace && git clone --recursive https://github.com/BasisResearch/vismatch.git vismatch
```

Expected: clone completes; `vismatch/third_party/*` submodules populated.

- [ ] **Step 2: Add upstream remote and fetch**

```bash
cd /workspace/vismatch && git remote add upstream https://github.com/gmberton/vismatch.git && git fetch upstream && git remote -v
```

Expected: `origin` → `BasisResearch/vismatch`, `upstream` → `gmberton/vismatch`.

- [ ] **Step 3: Confirm fork is even with upstream**

```bash
cd /workspace/vismatch && git rev-parse origin/main upstream/main && git submodule status | grep -c '^-' || true
```

Expected: both SHAs equal (`9d49b892...` as of 2026-09-25, or a newer equal pair). Submodule count of
uninitialized (`-` prefix) lines is `0`. If SHAs differ, stop and report — the fork has drifted.

- [ ] **Step 4: Set commit identity for this repo**

```bash
cd /workspace/vismatch && git config user.name "Tommy Botch" && git config user.email "tommy@basis.ai"
```

No commit in this task.

---

### Task 2: Create the vismatch dev environment

**Files:** none (env setup)

- [ ] **Step 1: Create a py3.10 venv (matches upstream CI)**

```bash
uv venv /opt/venv/vismatch --python 3.10
```

Expected: `Creating virtual environment at: /opt/venv/vismatch`.

- [ ] **Step 2: Editable install, same fallback as upstream CI**

```bash
cd /workspace/vismatch && (uv pip install --python /opt/venv/vismatch/bin/python -e ".[all]" || uv pip install --python /opt/venv/vismatch/bin/python -e .)
```

Expected: install succeeds on one of the two commands. Record which one in the Task 3 baseline note.

- [ ] **Step 3: Test tooling**

```bash
uv pip install --python /opt/venv/vismatch/bin/python pytest pytest-timeout ruff
```

- [ ] **Step 4: Proof the venv imports the fork, not a PyPI copy**

```bash
/opt/venv/vismatch/bin/python -c "import vismatch, torch; print(vismatch.__file__, torch.__version__, torch.cuda.is_available())"
```

Expected: path starts with `/workspace/vismatch/vismatch/`, `cuda` True on the A40.

No commit in this task.

---

### Task 3: Record the upstream baseline gate

**Files:**
- Create: `/workspace/collab-splats/docs/superpowers/specs/2026-09-25-vismatch-baseline.md` (collab-splats, gitignored dir → `git add -f`)

The full suite instantiates every model and downloads weights; it takes a long time. Run it in tmux.

- [ ] **Step 1: Lint baseline**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/ruff check . ; echo "check exit $?" ; /opt/venv/vismatch/bin/ruff format --check . ; echo "format exit $?"
```

Expected: both exit 0 on clean upstream. Record exit codes.

- [ ] **Step 2: Full test baseline in tmux**

```bash
tmux new-session -d -s vm-baseline "cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests -vv -rs --timeout=300 --junitxml=/workspace/logs/vismatch-baseline.xml 2>&1 | tee /workspace/logs/vismatch-baseline.log; echo \"EXIT \${PIPESTATUS[0]}\" >> /workspace/logs/vismatch-baseline.log"
```

- [ ] **Step 3: Wait for completion, then read the result**

Wait until the log's last line is `EXIT <n>` (use Monitor on the file; do not poll with `sleep`). Then:

```bash
grep -E "^(FAILED|ERROR)|passed|EXIT" /workspace/logs/vismatch-baseline.log | tail -80
```

- [ ] **Step 4: Write the baseline note**

Write `/workspace/collab-splats/docs/superpowers/specs/2026-09-25-vismatch-baseline.md` with:
upstream SHA, install command used (Task 2 Step 2), ruff exit codes, the pytest summary line
(passed/failed/skipped/errors), the full list of FAILED/ERROR node ids with one-line reasons, and the
skip reasons grouped by count. This list is the reference every later branch diffs against.

- [ ] **Step 5: Commit the note (collab-splats)**

```bash
cd /workspace/collab-splats && git add -f docs/superpowers/specs/2026-09-25-vismatch-baseline.md && git commit --only docs/superpowers/specs/2026-09-25-vismatch-baseline.md -m "docs(specs): vismatch upstream baseline gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Create the feature branch

**Files:** none

- [ ] **Step 1: Branch from upstream/main**

```bash
cd /workspace/vismatch && git switch -c feat/batch-forward upstream/main && git log --oneline -1
```

Expected: HEAD equals the baseline SHA from Task 3.

---

### Task 5: `to_tensor_images()` — normalize single image or batch

**Files:**
- Modify: `/workspace/vismatch/vismatch/utils.py` (append directly after `to_tensor_image`, ~L491)
- Test: `/workspace/vismatch/tests/test_utils.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_utils.py` (add `import numpy as np`, `import pytest`, `import torch`, and
`from vismatch.utils import to_tensor_images` to its imports if missing):

```python
def test_to_tensor_images_single_tensor_is_not_batched():
    """A (3, H, W) tensor is one image: a one-element list, flagged unbatched."""
    img = torch.rand(3, 32, 48)
    imgs, batched = to_tensor_images(img)
    assert batched is False
    assert len(imgs) == 1
    assert torch.equal(imgs[0], img)


def test_to_tensor_images_4d_tensor_is_batched():
    """A (B, 3, H, W) tensor splits into B (3, H, W) tensors, flagged batched."""
    batch = torch.rand(4, 3, 32, 48)
    imgs, batched = to_tensor_images(batch)
    assert batched is True
    assert len(imgs) == 4
    for img, ref in zip(imgs, batch):
        assert torch.equal(img, ref)


def test_to_tensor_images_4d_numpy_is_batched():
    """A (B, 3, H, W) numpy array is a batch too, converted to tensors."""
    batch = np.random.default_rng(0).random((2, 3, 16, 16), dtype=np.float32)
    imgs, batched = to_tensor_images(batch)
    assert batched is True
    assert all(isinstance(img, torch.Tensor) for img in imgs)
    np.testing.assert_array_equal(imgs[1].numpy(), batch[1])


def test_to_tensor_images_list_is_batched_and_keeps_sizes():
    """A list of single images is a batch; images may differ in size."""
    imgs, batched = to_tensor_images([torch.rand(3, 16, 16), torch.rand(3, 20, 30)])
    assert batched is True
    assert [tuple(img.shape) for img in imgs] == [(3, 16, 16), (3, 20, 30)]


def test_to_tensor_images_empty_list_is_empty_batch():
    """An empty list is an empty batch; forward() is responsible for rejecting it."""
    assert to_tensor_images([]) == ([], True)


def test_to_tensor_images_rejects_bad_channel_count():
    """Every image still goes through to_tensor_image's (3, H, W) check."""
    with pytest.raises(AssertionError):
        to_tensor_images(torch.rand(2, 4, 16, 16))
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests/test_utils.py -v -k to_tensor_images
```

Expected: collection ERROR — `ImportError: cannot import name 'to_tensor_images' from 'vismatch.utils'`.

- [ ] **Step 3: Implement**

Add to `vismatch/utils.py` directly after `to_tensor_image`:

```python
def to_tensor_images(
    imgs: torch.Tensor | np.ndarray | str | Path | Image.Image | list | tuple,
) -> tuple[list[torch.Tensor], bool]:
    """Normalize a single image or a batch of images to a list of (3, H, W) tensors.

    Args:
        imgs (torch.Tensor | np.ndarray | str | Path | Image.Image | list | tuple): a single image
            (path, PIL Image, or (3, H, W) array), or a batch: a (B, 3, H, W) array or a list/tuple
            of single images. Images in a list may differ in size.

    Returns:
        tuple: (list of (3, H, W) tensors, True if the input was a batch)
    """
    if isinstance(imgs, (list, tuple)):
        return [to_tensor_image(img) for img in imgs], True
    if isinstance(imgs, (torch.Tensor, np.ndarray)) and imgs.ndim == 4:
        return [to_tensor_image(img) for img in imgs], True
    return [to_tensor_image(imgs)], False
```

- [ ] **Step 4: Run to verify they pass**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests/test_utils.py -v
```

Expected: all tests in the file PASS (new and pre-existing).

- [ ] **Step 5: Lint and commit**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/ruff format vismatch/utils.py tests/test_utils.py && /opt/venv/vismatch/bin/ruff check vismatch/utils.py tests/test_utils.py && git add vismatch/utils.py tests/test_utils.py && git commit -m "Add to_tensor_images to normalize single images and batches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Extract `_postprocess()` from `forward()` (pure refactor)

**Files:**
- Modify: `/workspace/vismatch/vismatch/base_matcher.py` (`forward`, ~L114-199)

No behavior change, so no new test: the existing mock tests in `tests/test_matchers.py`
(`test_forward_matcher_confidences`, `test_forward_requires_confidence_slot`,
`test_forward_bad_confidence_shape_fails`, `test_forward_removes_out_of_bounds_matches`) are the gate.

- [ ] **Step 1: Run the existing mock tests (green before the refactor)**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests/test_matchers.py -v -k "confidence or out_of_bounds or import"
```

Expected: all PASS.

- [ ] **Step 2: Move the body**

In `forward()`, replace everything from `# self._forward() is implemented by the children modules`
through the final `return {...}` with:

```python
        # self._forward() is implemented by the children modules
        return self._postprocess(self._forward(img0, img1), img0, img1)
```

and add this method directly below `forward()` (the body is the moved code, unchanged except that it
reads `outputs` from its argument):

```python
    def _postprocess(self, outputs: tuple, img0: torch.Tensor, img1: torch.Tensor) -> dict:
        """Turn one pair's raw _forward() outputs into the forward() result dict.

        Args:
            outputs (tuple): the 7 objects returned by _forward() for one pair
            img0 (torch.Tensor): (3, H, W) image the outputs refer to, used for the bounds check
            img1 (torch.Tensor): (3, H, W) image the outputs refer to, used for the bounds check

        Returns:
            dict: result dict, see forward()
        """
        assert len(outputs) == 7, (
            f"{self.name}._forward() must return 7 values "
            f"(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences), "
            f"got {len(outputs)}. Return None for matched_confidences if the matcher has no per-match confidence."
        )
        matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences = outputs

        # Check that returned objects are of accepted types (nd.array, torch.tensor or None)
        self.check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences)

        # Convert torch tensors to numpy. None objects stay None
        matched_kpts0, matched_kpts1 = to_numpy(matched_kpts0), to_numpy(matched_kpts1)
        all_kpts0, all_kpts1 = to_numpy(all_kpts0), to_numpy(all_kpts1)
        all_desc0, all_desc1 = to_numpy(all_desc0), to_numpy(all_desc1)
        matched_confidences = to_numpy(matched_confidences)

        # Some models might return kpts=None if no kpts are found. In this case, set an empty array with dim (0, 2)
        matched_kpts0 = self.get_empty_array_if_none(matched_kpts0)
        matched_kpts1 = self.get_empty_array_if_none(matched_kpts1)
        all_kpts0 = self.get_empty_array_if_none(all_kpts0)
        all_kpts1 = self.get_empty_array_if_none(all_kpts1)
        # Same for descriptors: if it is empty set as descriptor an array with dim (0, 2)
        all_desc0 = self.get_empty_array_if_none(all_desc0)
        all_desc1 = self.get_empty_array_if_none(all_desc1)

        # Check that shapes are correct and consistent
        self.check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences)

        # Drop matches with a kpt outside its image, e.g. on regions added by padding (see issue #69)
        (h0, w0), (h1, w1) = img0.shape[-2:], img1.shape[-2:]
        valid = (
            (matched_kpts0 >= 0) & (matched_kpts0 < [w0, h0]) & (matched_kpts1 >= 0) & (matched_kpts1 < [w1, h1])
        ).all(1)
        matched_kpts0, matched_kpts1 = matched_kpts0[valid], matched_kpts1[valid]
        if matched_confidences is not None:
            matched_confidences = matched_confidences[valid]

        # Compute RANSAC to obtain the inliers and homography matrix
        H, inlier_kpts0, inlier_kpts1 = self.compute_ransac(matched_kpts0, matched_kpts1)

        return {
            "num_inliers": len(inlier_kpts0),
            "H": H,
            "all_kpts0": all_kpts0,
            "all_kpts1": all_kpts1,
            "all_desc0": all_desc0,
            "all_desc1": all_desc1,
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
            "matched_confidences": matched_confidences,
        }
```

- [ ] **Step 3: Prove it is a pure move**

```bash
cd /workspace/vismatch && git diff -U0 vismatch/base_matcher.py | grep '^[-+]' | grep -v '^[-+][-+]' | sed 's/^[-+]//' | sort | uniq -u
```

Moved lines keep their 8-space indentation, so each appears once as `-` and once as `+` and cancels.
Expected output is exactly: `outputs = self._forward(img0, img1)` (removed), `return self._postprocess(...)`
(added), and the `def _postprocess(...)` line plus its docstring lines (added). Any other line means code
changed during the move — fix it.

- [ ] **Step 4: Re-run the mock tests**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests/test_matchers.py -v -k "confidence or out_of_bounds or import"
```

Expected: all PASS, same as Step 1.

- [ ] **Step 5: Lint and commit**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/ruff format vismatch/base_matcher.py && /opt/venv/vismatch/bin/ruff check vismatch/base_matcher.py && git add vismatch/base_matcher.py && git commit -m "Move forward() post-processing into _postprocess

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Batched `forward()` — loop fallback and native path

**Files:**
- Modify: `/workspace/vismatch/vismatch/base_matcher.py` (imports; class attrs; `forward`; new `_split_batch_outputs`)
- Create: `/workspace/vismatch/tests/test_batch_forward.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_batch_forward.py`:

```python
"""Tests for batched BaseMatcher.forward().

Uses mock matchers only (no weight downloads):

- Unbatched input keeps returning one dict.
- Batched input (a (B, 3, H, W) tensor/array or a list) returns a list of B dicts, each equal to the
  single-pair result for that pair.
- Matchers with supports_batches=False are looped per pair; supports_batches=True receives stacked
  (B, 3, H, W) tensors in one _forward call, unless image sizes differ.
"""

import numpy as np
import pytest
import torch

from vismatch import BaseMatcher


def _pair_outputs(img0, img1):
    """Two in-bounds matches whose first x coordinate depends on each image's content."""
    x0, x1 = float(img0.mean()) * 100, float(img1.mean()) * 100
    kpts0 = np.array([[x0, 1.0], [2.0, 3.0]], dtype=np.float32)
    kpts1 = np.array([[x1, 1.0], [4.0, 5.0]], dtype=np.float32)
    desc = np.ones((2, 4), dtype=np.float32)
    conf = np.array([0.5, 0.9], dtype=np.float32)
    return kpts0, kpts1, kpts0, kpts1, desc, desc, conf


class _LoopMatcher(BaseMatcher):
    """Pair-only mock matcher that records the shape of every _forward input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def _forward(self, img0, img1):
        self.calls.append(tuple(img0.shape))
        return _pair_outputs(img0, img1)


class _NativeMatcher(_LoopMatcher):
    """Batch-native mock: a (B, 3, H, W) input returns 7 length-B sequences."""

    supports_batches = True

    def _forward(self, img0, img1):
        self.calls.append(tuple(img0.shape))
        if img0.ndim == 3:
            return _pair_outputs(img0, img1)
        per_pair = [_pair_outputs(a, b) for a, b in zip(img0, img1)]
        return tuple(list(field) for field in zip(*per_pair))


def _assert_results_equal(a, b):
    """Two forward() result dicts hold the same keys and values."""
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key])
        else:
            assert a[key] == b[key] or (a[key] is None and b[key] is None), key


@pytest.fixture
def batch():
    """Seeded (3, 3, 64, 64) batches for img0 and img1."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(3, 3, 64, 64, generator=generator), torch.rand(3, 3, 64, 64, generator=generator)


def test_single_pair_returns_dict(test_images):
    """Unbatched input keeps the pre-batching return type."""
    assert isinstance(_LoopMatcher().forward(*test_images), dict)


def test_tensor_batch_returns_list_matching_per_pair(batch):
    """A (B, 3, H, W) batch returns B dicts, each equal to that pair's single result."""
    b0, b1 = batch
    matcher = _LoopMatcher()
    results = matcher.forward(b0, b1)
    assert isinstance(results, list) and len(results) == 3
    for result, img0, img1 in zip(results, b0, b1):
        _assert_results_equal(result, matcher.forward(img0, img1))


def test_numpy_batch_matches_tensor_batch(batch):
    """A (B, 3, H, W) numpy batch gives the same results as the tensor batch."""
    b0, b1 = batch
    matcher = _LoopMatcher()
    for from_numpy, from_tensor in zip(matcher.forward(b0.numpy(), b1.numpy()), matcher.forward(b0, b1)):
        _assert_results_equal(from_numpy, from_tensor)


def test_list_batch_accepts_mixed_sizes():
    """A list batch may mix image sizes; each pair is matched at its own size."""
    imgs0 = [torch.rand(3, 64, 64), torch.rand(3, 48, 80)]
    imgs1 = [torch.rand(3, 64, 64), torch.rand(3, 48, 80)]
    matcher = _LoopMatcher()
    results = matcher.forward(imgs0, imgs1)
    assert len(results) == 2
    for result, img0, img1 in zip(results, imgs0, imgs1):
        _assert_results_equal(result, matcher.forward(img0, img1))


def test_path_list_batch(test_image_paths):
    """A list of paths is a batch, loaded like single-path input."""
    path0, path1 = test_image_paths
    matcher = _LoopMatcher()
    results = matcher.forward([path0, path1], [path1, path0])
    assert len(results) == 2
    _assert_results_equal(results[0], matcher.forward(path0, path1))
    _assert_results_equal(results[1], matcher.forward(path1, path0))


def test_batch_size_mismatch_raises(batch):
    """img0 and img1 batches must have the same length (pairwise matching)."""
    b0, b1 = batch
    with pytest.raises(ValueError, match="equal size"):
        _LoopMatcher().forward(b0, b1[:2])


def test_single_vs_batch_mismatch_raises(batch):
    """A single image cannot be matched against a batch."""
    b0, b1 = batch
    with pytest.raises(ValueError, match="equal size"):
        _LoopMatcher().forward(b0, b1[0])


def test_empty_batch_raises():
    """An empty batch is rejected instead of returning an empty list silently."""
    with pytest.raises(ValueError, match="empty batch"):
        _LoopMatcher().forward([], [])


def test_loop_calls_forward_once_per_pair(batch):
    """supports_batches=False: _forward sees one (3, H, W) pair at a time."""
    matcher = _LoopMatcher()
    matcher.forward(*batch)
    assert matcher.calls == [(3, 64, 64)] * 3


def test_native_batch_single_call_equals_loop(batch):
    """supports_batches=True: one _forward call on stacked inputs, same results as the loop."""
    native, loop = _NativeMatcher(), _LoopMatcher()
    native_results = native.forward(*batch)
    assert native.calls == [(3, 3, 64, 64)]
    for from_native, from_loop in zip(native_results, loop.forward(*batch)):
        _assert_results_equal(from_native, from_loop)


def test_native_batch_mixed_sizes_falls_back_to_loop():
    """Images that cannot be stacked are looped even for batch-native matchers."""
    matcher = _NativeMatcher()
    matcher.forward([torch.rand(3, 64, 64), torch.rand(3, 48, 80)], [torch.rand(3, 64, 64), torch.rand(3, 48, 80)])
    assert matcher.calls == [(3, 64, 64), (3, 48, 80)]


def test_native_batch_broadcasts_none_fields(batch):
    """A None output field from a batched _forward applies to every pair."""

    class Matcher(_NativeMatcher):
        def _forward(self, img0, img1):
            fields = list(super()._forward(img0, img1))
            fields[6] = None  # no confidences for the whole batch
            return tuple(fields)

    results = Matcher().forward(*batch)
    assert all(result["matched_confidences"] is None for result in results)


def test_native_batch_wrong_length_raises(batch):
    """A batched _forward must return one entry per pair in every non-None field."""

    class Matcher(_NativeMatcher):
        def _forward(self, img0, img1):
            return tuple(field[:-1] for field in super()._forward(img0, img1))

    with pytest.raises(AssertionError, match="one entry per pair"):
        Matcher().forward(*batch)


def test_extract_batch_returns_list(batch):
    """extract() on a batch returns one keypoint/descriptor dict per image."""
    b0, _ = batch
    matcher = _LoopMatcher()
    results = matcher.extract(b0)
    assert isinstance(results, list) and len(results) == 3
    for result, img in zip(results, b0):
        single = matcher.extract(img)
        np.testing.assert_array_equal(result["all_kpts0"], single["all_kpts0"])
        np.testing.assert_array_equal(result["all_desc0"], single["all_desc0"])
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests/test_batch_forward.py -v
```

Expected: `test_single_pair_returns_dict` PASSES (already true); the others FAIL — batched inputs hit
`to_tensor_image`'s `img should have shape (3, H, W)` AssertionError, and the list inputs hit
`img should be a torch.Tensor, a path, or a PIL Image`.

- [ ] **Step 3: Implement**

In `vismatch/base_matcher.py`:

(a) Import: change the utils import line to

```python
from vismatch.utils import to_normalized_coords, to_px_coords, to_numpy, _load_image, to_tensor_images
```

(`to_tensor_image` is no longer used in this file after this change; `ruff check` will flag it if it is.)

(b) Class attribute, directly under the class docstring:

```python
    # Set to True in a matcher whose _forward() accepts (B, 3, H, W) batches and returns 7 length-B
    # sequences (or None for a field absent on every pair). Otherwise forward() loops over pairs.
    supports_batches: bool = False
```

(c) Replace `forward()`'s signature, docstring, and body (keep `@torch.inference_mode()`):

```python
    @torch.inference_mode()
    def forward(
        self,
        img0: torch.Tensor | np.ndarray | str | Path | Image.Image | list,
        img1: torch.Tensor | np.ndarray | str | Path | Image.Image | list,
    ) -> dict | list[dict]:
        """Run matching pipeline on two images, or on a batch of image pairs. All sub-classes implement this interface.

        Args:
            img0 (torch.Tensor | np.ndarray | str | Path | Image.Image | list): image as (3, H, W) array in [0, 1]
                range, path, or PIL Image; or a batch as a (B, 3, H, W) array or a list of such images
            img1 (torch.Tensor | np.ndarray | str | Path | Image.Image | list): same as img0; a batch must have the
                same length as img0's, and pair i is (img0[i], img1[i])

        Returns:
            dict | list[dict]: for a single pair, a result dict with keys:
                - num_inliers (int): number of inliers after RANSAC, i.e. len(inlier_kpts0)
                - H (np.ndarray): (3 x 3) homography matrix to map matched_kpts0 to matched_kpts1
                - all_kpts0 (np.ndarray): (N0 x 2) all detected keypoints from img0
                - all_kpts1 (np.ndarray): (N1 x 2) all detected keypoints from img1
                - all_desc0 (np.ndarray): (N0 x D) all descriptors from img0
                - all_desc1 (np.ndarray): (N1 x D) all descriptors from img1
                - matched_kpts0 (np.ndarray): (N2 x 2) keypoints from img0 that match matched_kpts1 (pre-RANSAC)
                - matched_kpts1 (np.ndarray): (N2 x 2) keypoints from img1 that match matched_kpts0 (pre-RANSAC)
                - inlier_kpts0 (np.ndarray): (N3 x 2) filtered matched_kpts0 that fit the H model (post-RANSAC)
                - inlier_kpts1 (np.ndarray): (N3 x 2) filtered matched_kpts1 that fit the H model (post-RANSAC)
                - matched_confidences (np.ndarray | None): (N2,) per-match confidence scores, None if the matcher does not provide confidence (pre-RANSAC).
                For a batch, a list with one such dict per pair.
        """
        imgs0, batched0 = to_tensor_images(img0)
        imgs1, batched1 = to_tensor_images(img1)
        if batched0 != batched1 or len(imgs0) != len(imgs1):
            size0 = len(imgs0) if batched0 else "a single image"
            size1 = len(imgs1) if batched1 else "a single image"
            raise ValueError(f"img0 and img1 must be single images or batches of equal size, got {size0} and {size1}")
        if len(imgs0) == 0:
            raise ValueError("Cannot match an empty batch")
        imgs0 = [img.to(self.device) for img in imgs0]
        imgs1 = [img.to(self.device) for img in imgs1]

        # Batch-native matchers get one stacked call when every image on a side has the same size;
        # everything else runs _forward() once per pair. self._forward() is implemented by the children modules
        stackable = len({img.shape for img in imgs0}) == 1 and len({img.shape for img in imgs1}) == 1
        if batched0 and self.supports_batches and stackable:
            outputs = self._forward(torch.stack(imgs0), torch.stack(imgs1))
            pair_outputs = self._split_batch_outputs(outputs, len(imgs0))
        else:
            pair_outputs = [self._forward(i0, i1) for i0, i1 in zip(imgs0, imgs1)]

        results = [self._postprocess(out, i0, i1) for out, i0, i1 in zip(pair_outputs, imgs0, imgs1)]
        return results if batched0 else results[0]

    def _split_batch_outputs(self, outputs: tuple, batch_size: int) -> list[tuple]:
        """Split a batched _forward() output into one 7-tuple per pair.

        Args:
            outputs (tuple): the 7 objects returned by a batched _forward(), each a length-B sequence or None
            batch_size (int): number of pairs B

        Returns:
            list[tuple]: B tuples of 7 objects, in the single-pair _forward() format
        """
        assert len(outputs) == 7, f"{self.name}._forward() must return 7 values, got {len(outputs)}"
        fields = [[None] * batch_size if field is None else list(field) for field in outputs]
        for field in fields:
            assert len(field) == batch_size, (
                f"{self.name}._forward() must return one entry per pair in each batched output, "
                f"got {len(field)} for a batch of {batch_size}"
            )
        return list(zip(*fields))
```

(d) Replace `extract()` so a batch returns a list:

```python
    def extract(
        self, img: torch.Tensor | np.ndarray | str | Path | Image.Image | list
    ) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
        """Extract keypoints and descriptors from a single image, or from a batch of images.

        Args:
            img (torch.Tensor | np.ndarray | str | Path | Image.Image | list): image as (3, H, W) array in [0, 1]
                range, path, or PIL Image; or a batch as a (B, 3, H, W) array or a list of such images

        Returns:
            dict | list[dict]: for a single image, a result dict with keys:
                - all_kpts0 (np.ndarray): (N, 2) detected keypoints
                - all_desc0 (np.ndarray): (N, D) descriptors
                For a batch, a list with one such dict per image.
        """
        result = self.forward(img, img)
        results = result if isinstance(result, list) else [result]
        kpts_key = "matched_kpts0" if isinstance(self, EnsembleMatcher) else "all_kpts0"
        extracted = [{"all_kpts0": r[kpts_key], "all_desc0": r["all_desc0"]} for r in results]
        return extracted if isinstance(result, list) else extracted[0]
```

- [ ] **Step 4: Run the new tests**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests/test_batch_forward.py tests/test_utils.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run every fast (no-download) test**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests -v -k "not test_create_matcher and not test_forward_synthetic_images and not test_extract_keypoints" --timeout=300
```

Expected: all PASS (includes `test_import_sandbox.py`, `test_viz.py`, the confidence/out-of-bounds
mock tests).

- [ ] **Step 6: Lint and commit**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/ruff format . && /opt/venv/vismatch/bin/ruff check . && git add vismatch/base_matcher.py tests/test_batch_forward.py && git commit -m "Support batched image pairs in BaseMatcher.forward

A (B, 3, H, W) array or a list of images returns a list of B result dicts;
single-pair input still returns one dict. Matchers loop over pairs unless they
set supports_batches = True, in which case _forward receives stacked batches.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Document batched usage

**Files:**
- Modify: `/workspace/vismatch/README.md` (Python API code block, after the `matcher.extract(img0)` / `plot_keypoints` lines)
- Modify: `/workspace/vismatch/docs/source/quickstart.md` (new section between "Keypoint Extraction" and "Ensemble Matching")

- [ ] **Step 1: README**

Inside the `### Python API` code block, after `plot_keypoints(img0, result, save_path="plot_keypoints.png")`, add:

```python

# Batches of pairs: pass (B, 3, H, W) tensors or lists of images, get a list of B result dicts
results = matcher([img0, img1], [img1, img0])
# results[0] matches img0 -> img1, results[1] matches img1 -> img0
```

- [ ] **Step 2: Quickstart**

Insert before `## Ensemble Matching`:

````markdown
## Batch Matching

Pass a batch of pairs as `(B, 3, H, W)` tensors or as lists of images (paths, PIL Images, or
`(3, H, W)` tensors; sizes may differ within a list). The result is a list with one result dict per
pair; a single pair still returns a single dict.

```python
results = matcher([img0, img1], [img1, img0])
# results[0] matches img0 -> img1, results[1] matches img1 -> img0

kpts = matcher.extract([img0, img1])
# kpts[0]["all_kpts0"], kpts[1]["all_kpts0"]
```

Matchers run one pair at a time unless they natively support batches (`matcher.supports_batches`).
````

- [ ] **Step 3: Verify the README snippet runs**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/python -c "
from vismatch import get_matcher
m = get_matcher('sift-lightglue', device='cuda')
a = m.load_image('vismatch/assets/example_pairs/outdoor/montmartre_close.jpg', resize=512)
b = m.load_image('vismatch/assets/example_pairs/outdoor/montmartre_far.jpg', resize=512)
r = m([a, b], [b, a]); print(type(r).__name__, len(r), [x['num_inliers'] for x in r])
k = m.extract([a, b]); print(len(k), [x['all_kpts0'].shape for x in k])
"
```

Expected: `list 2 [<int>, <int>]` then `2 [(N0, 2), (N1, 2)]`. If `sift-lightglue` is not in
`vismatch.available_models`, use any detector-based name from that list and note it.

- [ ] **Step 4: Commit**

```bash
cd /workspace/vismatch && git add README.md docs/source/quickstart.md && git commit -m "Document batched matching

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Full gate vs baseline, then push

**Files:**
- Modify: `/workspace/collab-splats/docs/superpowers/specs/2026-09-25-vismatch-baseline.md` (append a `feat/batch-forward` section)

- [ ] **Step 1: Lint gate**

```bash
cd /workspace/vismatch && /opt/venv/vismatch/bin/ruff check . ; echo "check exit $?" ; /opt/venv/vismatch/bin/ruff format --check . ; echo "format exit $?"
```

Expected: both `exit 0`.

- [ ] **Step 2: Full suite in tmux (every model goes through the new forward())**

```bash
tmux new-session -d -s vm-batch "cd /workspace/vismatch && /opt/venv/vismatch/bin/python -m pytest tests -vv -rs --timeout=300 --junitxml=/workspace/logs/vismatch-batch-forward.xml 2>&1 | tee /workspace/logs/vismatch-batch-forward.log; echo \"EXIT \${PIPESTATUS[0]}\" >> /workspace/logs/vismatch-batch-forward.log"
```

Wait for the `EXIT` line (Monitor, not sleep).

- [ ] **Step 3: Diff against the baseline**

```bash
diff <(grep -E "^(FAILED|ERROR)" /workspace/logs/vismatch-baseline.log | sort) <(grep -E "^(FAILED|ERROR)" /workspace/logs/vismatch-batch-forward.log | sort); echo "diff exit $?"
grep -E "passed|failed" /workspace/logs/vismatch-baseline.log /workspace/logs/vismatch-batch-forward.log | grep "==="
```

Expected: `diff exit 0` (identical FAILED/ERROR sets). Passed count = baseline passed + the new tests
(20 new: 6 in test_utils.py, 14 in test_batch_forward.py). Skips unchanged. Any new failure: stop,
investigate with superpowers:systematic-debugging — do not push.

- [ ] **Step 4: Record the result**

Append a `## feat/batch-forward @ <sha>` section to the baseline note with the ruff exit codes, the
pytest summary line, and the diff verdict. Commit it:

```bash
cd /workspace/collab-splats && git add -f docs/superpowers/specs/2026-09-25-vismatch-baseline.md && git commit --only docs/superpowers/specs/2026-09-25-vismatch-baseline.md -m "docs(specs): vismatch batch-forward gate result

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Ask the user before pushing**

Show `git -C /workspace/vismatch log --oneline upstream/main..feat/batch-forward` and ask for
confirmation. Only after a yes:

```bash
cd /workspace/vismatch && git push -u origin feat/batch-forward
```

- [ ] **Step 6: Hand the user the PR text**

The user opens the PR in the browser (base `gmberton/vismatch:main`, head
`BasisResearch/vismatch:feat/batch-forward`). Give them this body:

```markdown
Implements the batching part of #74, following the design discussed there.

- `forward()` accepts a batch of pairs: a `(B, 3, H, W)` tensor/array or a list of images
  (paths, PIL Images, `(3, H, W)` tensors; sizes may differ within a list). A batch returns a
  list of B result dicts; a single pair still returns one dict, so existing code is unaffected.
- New `supports_batches` class attribute (default `False`). BaseMatcher loops `_forward()` over
  pairs for every model today; a model that sets it to `True` receives stacked `(B, 3, H, W)`
  inputs in one call and returns 7 length-B sequences.
- `extract()` accepts a batch the same way.
- No model files changed. Native batching for individual models (LoMa, XFeat, LightGlue) will
  follow as separate small PRs.

Tests: `tests/test_batch_forward.py` (mock matchers, no downloads) and `to_tensor_images` tests
in `tests/test_utils.py`. Full suite: same failures/skips as `main` locally.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 10: Bookkeeping in collab-splats

**Files:**
- Modify: `/workspace/collab-splats/CLAUDE.md` (vismatch-fork in-flight entry: add plan link)

- [ ] **Step 1: Link the plan in the in-flight entry**

Change the `vismatch-fork` line's link group from `([spec](...))` to
`([spec](docs/superpowers/specs/2026-09-25-vismatch-fork-design.md) · [plan](docs/superpowers/plans/2026-09-25-vismatch-batch-forward.md))`.

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats && git add -f docs/superpowers/plans/2026-09-25-vismatch-batch-forward.md && git commit --only CLAUDE.md docs/superpowers/plans/2026-09-25-vismatch-batch-forward.md -m "docs(plans): vismatch batch-forward plan status

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

The in-flight entry stays until steps 1-5 of the spec are done; this plan covers only steps 0 and 1a.
