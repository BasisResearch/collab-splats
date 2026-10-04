# vismatch feature matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land batched `extract`, a lean `match(feats0, feats1)` and the uint8 input branch in the BasisResearch vismatch fork, in upstream style, then switch collab-splats `LocalMatcher` onto them. Every measured speedup must be kept.

**Architecture:**
- Fork branches:
  - `feat/fast-input`: uint8 branch in `to_tensor_image`.
  - `feat/batch-forward` (1a): add `self.supports_batches = False`.
  - `feat/feature-matching`, stacked on 1a: `_extract_features` / `_match_features` hooks, `extract(list)`, `match()`, the native batched `forward`, and the xfeat and loma hooks.
  - `basis` merges all of them and is what collab-splats pins.
- collab-splats deletes its local workarounds (`_split_loma_forward`, `FEATURE_MATCH_MODELS`, kornia `match_mnn`) and calls vismatch directly.

**Tech Stack:**
- vismatch (py3.10, ruff, flat pytest with the `device` / `test_images` / `test_image_paths` fixtures)
- torch, XFeat (`accelerated_features`), LoMa
- collab-splats: py3.11, `/opt/venv/reconstruction/bin/python`

**Spec:** [2026-10-03-vismatch-feature-matching-design.md](../specs/2026-10-03-vismatch-feature-matching-design.md)

---

## Spec deltas (found while planning; Task 13 amends the spec)

1. **No base hook defs.** `BaseMatcher` defines no abstract `_forward` either.
   - `supports_batches` is the single gate: `extract` uses the hooks when it is True, and `match()` raises `NotImplementedError` when it is False.
   - The hook names still go into the sandbox tuple.
   - Saves the spec's "+8 base hooks" row.
2. **`image_size` at load comes from the existing zarr `hw` attr.** No new zarr attribute.
3. **The `sift-nn` exact test is dropped.** 1a's `_CornerMatcher` mock tests already prove loop == per-pair. A new `_GridMatcher` mock covers the native path on CPU without model downloads.
4. **No bf16 casts on LoMa kpts/desc.**
   - `forward` already does `to_numpy(desc0[0])`, and the current split does `kpts[0].cpu().numpy()`. Both would raise on bf16, and both work today, so the tensors are fp32.
   - Only the confidences get `.float()`, exactly as `_forward` does.
5. **xfeat batching stacks only when every image has the same size, otherwise it runs per image.** This matches the spec. The uncommitted group-by-size `_extract_batch` in the 1a worktree is backed up (Task 0) and not reused.
6. **The LoMa "rebuild" guard is dropped.** The guard is `test_loma_match_without_payload_names_the_rebuild` plus its `ValueError`. A cache without `keypoints_normalized` now fails with vismatch's `KeyError: 'kpts_normalized'`. `save_index` writes the payload all-or-none.

## Ground rules (every task)

- **Do not touch** `/workspace/collab-splats/.worktrees/rgbd-ba`. It belongs to the user and holds an uncommitted `match_batch`.
- **collab-splats commits:** use `git commit --only <paths>`; use `git add -f` for gitignored `docs/superpowers` files. Never sweep in other sessions' changes: `CLAUDE.md`, `README.md`, `collab_splats/mesh/*`, `docs/mesh.md` and others are dirty and are not ours.
- **Commit message trailer:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- **vismatch commits:** upstream style, so a plain imperative subject and no conventional-commit prefix. Example: `Add uint8 input branch to to_tensor_image`. Add the same trailer.
- **vismatch tests:** run from the worktree root so `import vismatch` resolves to that tree. Check with `python -c "import vismatch; print(vismatch.__file__)"`.
- **vismatch upstream CI gate:** run this in the worktree before every vismatch commit that touches code:
  ```bash
  /opt/venv/reconstruction/bin/python -m ruff check . && /opt/venv/reconstruction/bin/python -m ruff format --check .
  ```
  - On code changes, also run the touched test file with `--timeout=300`.
  - CI runs on py3.10 and CPU only. Don't use py3.11-only syntax (`ExceptionGroup`, `typing.Self`).
- `SP=/tmp/claude-0/-workspace-collab-splats/24ab2370-1ed5-4d10-9685-ecbfed2a1b04/scratchpad` (this session's scratchpad).
- **Pushing to `origin` (BasisResearch) and installing into the shared venv are outward-facing.** Ask the user before each one; the plan marks these steps 🔒.

## File map

| Repo / file | Change |
|---|---|
| vismatch `vismatch/utils.py` | uint8 branch in `to_tensor_image` (fast-input) |
| vismatch `tests/test_utils.py` | `test_to_tensor_image_uint8` (fast-input) |
| vismatch `vismatch/base_matcher.py` | `supports_batches` (1a); hook names in sandbox tuple, native `extract`, `match()`, native batched `forward` (feature-matching) |
| vismatch `vismatch/im_models/xfeat.py` | `supports_batches = mode == "sparse"`, `_extract_features`, `_match_features` |
| vismatch `vismatch/im_models/loma.py` | `supports_batches = True`, `_extract_features`, `_match_features` |
| vismatch `tests/test_matchers.py` | `_GridMatcher` mock tests + xfeat/loma real-model tests |
| vismatch `README.md` | batched extract + `match()` paragraph |
| collab-splats `pyproject.toml`, `uv.lock` | git pin on basis SHA, drop `lomatch` |
| collab-splats `collab_splats/localization/extractors.py` | `LocalFeatures.image_size`, `skip_ransac`, uint8 `_to_tensor`, list `extract`, vismatch `match` |
| collab-splats `collab_splats/localization/localizer.py` | `read_localization_db` fills `image_size` from `hw` |
| collab-splats `tests/localization/test_local_matcher.py` | mock + parity tests moved to the vismatch API |
| collab-splats `docs/known-test-failures.md` | rename stale test names |
| `$SP/gate_fm.py` (scratch, not committed) | perf gate script |

---

### Task 0: Back up uncommitted fork work and record baselines

**Files:**
- Create: `$SP/backup_1a_uncommitted.patch`, `$SP/backup_batch_loma.patch`, `$SP/gate_fm.py`, `$SP/gate_baseline.txt`

- [ ] **Step 1: Back up the two dirty fork worktrees**

  RTK mangles diffs, so use `rtk proxy`.

```bash
SP=/tmp/claude-0/-workspace-collab-splats/24ab2370-1ed5-4d10-9685-ecbfed2a1b04/scratchpad
cd /workspace/vismatch-batch-forward && rtk proxy git diff -- tests vismatch/base_matcher.py vismatch/im_models > $SP/backup_1a_uncommitted.patch
cd /workspace/vismatch-batch-loma && rtk proxy git diff -- tests vismatch > $SP/backup_batch_loma.patch
wc -l $SP/backup_*.patch
```
Expected: both files non-empty. The 1a patch contains `_extract_batch`.

- [ ] **Step 2: Reset the 1a worktree's tracked files to `ceb0022`**

  Leave the submodule `m` markers alone; they are pre-existing.

```bash
cd /workspace/vismatch-batch-forward && git checkout -- tests/test_matchers.py vismatch/base_matcher.py vismatch/im_models/xfeat.py && git status --short
```
Expected: only `m vismatch/third_party/...` lines remain.

- [ ] **Step 3: Write the perf gate script**

  It runs in `baseline` mode (current site-packages vismatch 1.3.1, current `LocalMatcher`) and in `gate` mode (the new API).

Create `$SP/gate_fm.py`:

```python
# Perf gate for vismatch feature matching (spec 2026-10-03, "Gates / Performance")
# - usage: gate_fm.py baseline|gate [vismatch_root]
# - frames: GH010229 full-res PNGs (1920x1080); 640x480 via resize
# - match ratios are same-run: replicas of rgbd-ba match_batch and the old loma split
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

MODE = sys.argv[1]
if len(sys.argv) > 2:
    sys.path.insert(0, sys.argv[2])

import vismatch  # noqa: E402
from vismatch import get_matcher  # noqa: E402

print("vismatch", vismatch.__file__)
FR = sorted(Path("/workspace/outputs/ocr_viewer/GH010229_f1000_ref_loger/images").glob("frame_*.png"))
B, N_PAIRS = 32, 32
full = [np.ascontiguousarray(cv2.imread(str(p))[..., ::-1]) for p in FR[: 2 * B : 2]]
small = [cv2.resize(f, (640, 480), interpolation=cv2.INTER_AREA) for f in full]
nrm = torch.nn.functional.normalize


def timed(fn, reps=5):
    """Mean synced seconds of fn() after one warmup."""
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def upload(rgb):
    """HxWx3 uint8 -> (3,H,W) uint8 cuda (the new LocalMatcher._to_tensor)."""
    return torch.from_numpy(rgb).cuda().permute(2, 0, 1)


def rgbd_batch(pairs):
    """rgbd-ba match_batch core: padded cdist mutual-NN, GPU features."""
    n0 = [len(q) for q, _ in pairs]
    n1 = [len(d) for _, d in pairs]
    d0 = torch.full((len(pairs), max(n0), 64), 1e3, device="cuda")
    d1 = torch.full((len(pairs), max(n1), 64), 1e3, device="cuda")
    for b, (q, d) in enumerate(pairs):
        d0[b, : n0[b]] = nrm(q, dim=1)
        d1[b, : n1[b]] = nrm(d, dim=1)
    dist = torch.cdist(d0, d1)
    nn01, nn10 = dist.argmin(2).cpu().numpy(), dist.argmin(1).cpu().numpy()
    out = []
    for b in range(len(pairs)):
        iq = np.arange(n0[b])
        idb = nn01[b, : n0[b]]
        k = nn10[b, idb] == iq
        out.append((iq[k], idb[k]))
    return out


with torch.inference_mode():
    xm = get_matcher("xfeat", device="cuda")

    if MODE == "baseline":
        # Old input path: on-device float + max sync; old extract: per-image self-pair forward
        def old_input():
            for f in small:
                t = torch.from_numpy(f).cuda().permute(2, 0, 1).float()
                t = t / 255.0 if t.max() > 1.5 else t

        print(f"input old            {timed(old_input) / B * 1e3:7.2f} ms/frame")
        for name, imgs in (("640x480", small), ("full-res", full)):
            ts = [upload(f).float() / 255 for f in imgs]
            print(f"xfeat extract {name:9s} {timed(lambda: [xm.extract(t) for t in ts], 2) / B * 1e3:7.2f} ms/frame (per image)")
        sys.exit()

    # Input conversion: uint8 upload + to_tensor_image
    from vismatch.utils import to_tensor_image

    print(f"input uint8          {timed(lambda: [to_tensor_image(upload(f)) for f in small]) / B * 1e3:7.2f} ms/frame  gate <= 0.5")

    # xfeat extract(list), B=32, both resolutions
    for name, imgs, gate in (("640x480", small, 16.2), ("full-res", full, 5.3)):
        ts = [upload(f) for f in imgs]
        print(f"xfeat extract {name:9s} {timed(lambda: xm.extract(ts)) / B * 1e3:7.2f} ms/frame  gate <= {gate}")

    # xfeat match(): GPU features vs rgbd-ba match_batch, same run
    xm.skip_ransac = True
    feats = [{k: torch.as_tensor(v, device="cuda") if k != "image_size" else v for k, v in f.items()}
             for f in xm.extract([upload(f) for f in small])]
    pairs = [(feats[i], feats[(i + 1 + i % 7) % B]) for i in range(N_PAIRS)]
    t_new = timed(lambda: [xm.match(a, b) for a, b in pairs], 10) / N_PAIRS
    t_ref = timed(lambda: rgbd_batch([(a["all_desc0"], b["all_desc0"]) for a, b in pairs]), 10) / N_PAIRS
    print(f"xfeat match()        {t_new * 1e3:7.3f} ms/pair  match_batch {t_ref * 1e3:7.3f}  ratio {t_new / t_ref:.2f}  gate <= 1.10")

    # loma match() vs the old LocalMatcher loma split (CPU-held features both sides), same run
    from vismatch.im_models import loma as loma_mod

    lm = get_matcher("loma", device="cuda")
    lm.skip_ransac = True
    lfeats = lm.extract([upload(f) for f in small[:8]])
    lpairs = [(lfeats[i], lfeats[(i + 1) % 8]) for i in range(8)]

    def old_split():
        for a, b in lpairs:
            k0 = torch.as_tensor(a["kpts_normalized"]).cuda()[None]
            k1 = torch.as_tensor(b["kpts_normalized"]).cuda()[None]
            d0 = torch.as_tensor(a["all_desc0"]).cuda()[None]
            d1 = torch.as_tensor(b["all_desc0"]).cuda()[None]
            m0 = loma_mod.filter_matches(lm.matcher(k0, k1, d0, d1)["scores"], lm.matcher.cfg.filter_threshold)[0]
            valid = m0[0] > -1
            torch.where(valid)[0].cpu().numpy(), m0[0][valid].cpu().numpy()

    t_new = timed(lambda: [lm.match(a, b) for a, b in lpairs], 3) / 8
    t_ref = timed(old_split, 3) / 8
    print(f"loma match()         {t_new * 1e3:7.2f} ms/pair  old split {t_ref * 1e3:7.2f}  ratio {t_new / t_ref:.2f}  gate <= 1.10")
```

- [ ] **Step 4: Record the baseline (current vismatch 1.3.1 from site-packages)**

```bash
cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python $SP/gate_fm.py baseline 2>&1 | tee $SP/gate_baseline.txt
```
Expected:
- The first line prints `.../site-packages/vismatch/__init__.py`.
- Then input and per-image extract numbers. They are informational (handoff: ~8 ms/frame input; 132.5 / 40.3 ms/frame extract at B=32 batched-equivalent).

No commit (scratch only).

---

### Task 1: `feat/fast-input` — uint8 branch in `to_tensor_image`

Worktree: reuse `/workspace/vismatch-batch-loma`. Its batching work is superseded and backed up in Task 0, and its submodules are already initialized.

**Files:**
- Modify: `vismatch/utils.py` (`to_tensor_image`, after the shape assert)
- Test: `tests/test_utils.py`

- [ ] **Step 1: Cut the branch from `upstream/main`**

```bash
cd /workspace/vismatch-batch-loma && git checkout -- tests vismatch && git fetch upstream && git switch -c feat/fast-input upstream/main && git log --oneline -1
```
Expected: `9d49b89 Add upal matcher (#75)`. If upstream has moved, use the new tip and note it in the PR.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_utils.py`:

```python
import numpy as np
import torch

from vismatch.utils import to_tensor_image


@pytest.mark.parametrize("to_input", [lambda x: x, lambda x: x.numpy()], ids=["tensor", "numpy"])
def test_to_tensor_image_uint8(to_input):
    """A uint8 (3, H, W) image is scaled to [0, 1] floats, exactly as converting it by hand."""
    img = torch.randint(0, 256, (3, 20, 30), dtype=torch.uint8)
    out = to_tensor_image(to_input(img))
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, img.float() / 255, rtol=0, atol=0)
```

Move the three new imports up to join the file's existing import block (`import pytest`, `from pathlib import Path`, `from vismatch.utils import get_image_pairs_paths`). Merge the two `vismatch.utils` imports into one line: `from vismatch.utils import get_image_pairs_paths, to_tensor_image`.

- [ ] **Step 3: Run the test and check that it fails**

```bash
cd /workspace/vismatch-batch-loma && /opt/venv/reconstruction/bin/python -m pytest tests/test_utils.py -k uint8 -v --timeout=300
```
Expected: 2 FAILED with `AssertionError: img should be in [0, 1], got [0, 255]` or close to it.

- [ ] **Step 4: Write the minimal implementation**

In `vismatch/utils.py` `to_tensor_image`, directly after `assert img.ndim == 3 and img.shape[0] == 3, ...`:

```python
    # uint8 is always in range; convert on the image's own device (e.g. after a cheap uint8 upload)
    if img.dtype == torch.uint8:
        return img.float() / 255
```

- [ ] **Step 5: Run the tests and the CI gate**

```bash
cd /workspace/vismatch-batch-loma && /opt/venv/reconstruction/bin/python -m pytest tests/test_utils.py -v --timeout=300 && /opt/venv/reconstruction/bin/python -m ruff check . && /opt/venv/reconstruction/bin/python -m ruff format --check .
```
Expected: all pass, `All checks passed!`, and `N files already formatted`.

- [ ] **Step 6: Commit**

```bash
cd /workspace/vismatch-batch-loma && git add vismatch/utils.py tests/test_utils.py && git commit -m "$(cat <<'EOF'
Accept uint8 images in to_tensor_image

A uint8 image is scaled to [0, 1] on its own device, so callers can
upload uint8 (4x fewer bytes) and convert on the GPU. Skips the min/max
range check, which forces a device sync and rejects uint8.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `feat/batch-forward` (1a) — `supports_batches` attribute

**Files:**
- Modify: `/workspace/vismatch-batch-forward/vismatch/base_matcher.py` (`BaseMatcher.__init__`)
- Test: `/workspace/vismatch-batch-forward/tests/test_matchers.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_matchers.py`, directly after `test_extract_batch`:

```python
def test_supports_batches_default():
    """Matchers default to looping over a batch one pair at a time."""
    assert _CornerMatcher().supports_batches is False
```

- [ ] **Step 2: Run the test and check that it fails**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k supports_batches -v --timeout=300
```
Expected: FAIL with `AttributeError: '_CornerMatcher' object has no attribute 'supports_batches'`.

- [ ] **Step 3: Implement**

In `BaseMatcher.__init__`, after `self.device: str = device`:

```python

        # Matchers that set this to True batch natively; others loop over a batch one pair at a time
        self.supports_batches: bool = False
```

- [ ] **Step 4: Run the tests and the CI gate**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "batch or supports" -v --timeout=300 && /opt/venv/reconstruction/bin/python -m ruff check . && /opt/venv/reconstruction/bin/python -m ruff format --check .
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/vismatch-batch-forward && git add vismatch/base_matcher.py tests/test_matchers.py && git commit -m "$(cat <<'EOF'
Add supports_batches flag to BaseMatcher

Defaults to False, so batches loop one pair at a time as agreed in #74.
Matchers with native batching set it to True.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 🔒 Push 1a (ask the user first)**

```bash
cd /workspace/vismatch-batch-forward && git push origin feat/batch-forward
```

---

### Task 3: `feat/feature-matching` — hook names in the sandbox, native `extract`

Branch off 1a in the same worktree.

**Files:**
- Modify: `vismatch/base_matcher.py` (`__init_subclass__` tuple and comment; `extract`)
- Test: `tests/test_matchers.py`

- [ ] **Step 1: Cut the branch**

```bash
cd /workspace/vismatch-batch-forward && git switch -c feat/feature-matching && git log --oneline -1
```
Expected: the Task 2 commit at the tip.

- [ ] **Step 2: Write the failing tests**

In `tests/test_matchers.py`, after `test_supports_batches_default`:

```python
class _GridMatcher(BaseMatcher):
    """Detects a 3-point grid scaled to each image and matches keypoint i to keypoint i, natively batched."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_batches = True

    def _extract_features(self, imgs):
        feats = []
        for img in imgs:
            h, w = img.shape[-2:]
            kpts = torch.tensor([[0.0, 0.0], [w / 2, h / 2], [w - 1, h - 1]])
            feats.append({"all_kpts0": kpts, "all_desc0": torch.eye(3), "extra": kpts * 2})
        return feats

    def _match_features(self, feats0, feats1):
        idxs = torch.arange(3)
        return idxs, idxs, torch.ones(3)

    def _forward(self, img0, img1):
        f0, f1 = self._extract_features([img0, img1])
        return f0["all_kpts0"], f1["all_kpts0"], f0["all_kpts0"], f1["all_kpts0"], f0["all_desc0"], f1["all_desc0"], None


def test_extract_native_matches_forward():
    """With supports_batches, extract() returns forward()'s keypoints and descriptors, the image size and model extras."""
    matcher, img = _GridMatcher(), torch.rand(3, 40, 60)
    feats, expected = matcher.extract(img), matcher.forward(img, img)
    np.testing.assert_array_equal(feats["all_kpts0"], expected["all_kpts0"])
    np.testing.assert_array_equal(feats["all_desc0"], expected["all_desc0"])
    assert feats["image_size"] == (60, 40)
    assert isinstance(feats["extra"], np.ndarray)


def test_extract_native_batch():
    """With supports_batches, extract() on a batch returns one result per image, each with its own image size."""
    results = _GridMatcher().extract([torch.rand(3, 40, 60), torch.rand(3, 20, 30)])
    assert [r["image_size"] for r in results] == [(60, 40), (30, 20)]
    assert [r["all_kpts0"][-1].tolist() for r in results] == [[59, 39], [29, 19]]
```

- [ ] **Step 3: Run the tests and check that they fail**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k extract_native -v --timeout=300
```
Expected: 2 FAILED with `KeyError: 'image_size'`, because 1a's extract runs `forward(img, img)`.

- [ ] **Step 4: Implement**

(a) In `__init_subclass__`, replace the tuple and extend the comment's first line:

```python
        # Run each wrapper-defined matcher's __init__, _forward and feature hooks inside that wrapper's
```
```python
        for method_name in ("__init__", "_forward", "_extract_features", "_match_features"):
```

(b) In `extract`, insert this block as the first statement of the body, before 1a's `if is_batch(img):` loop:

```python
        # Matchers with native batching detect all images at once, skipping forward()'s self-pair match and RANSAC
        if self.supports_batches:
            imgs = [to_tensor_image(i).to(self.device) for i in (img if is_batch(img) else [img])]
            with torch.inference_mode():
                feats = self._extract_features(imgs)
            for f, i in zip(feats, imgs):
                to_numpy(f)
                f["image_size"] = (i.shape[-1], i.shape[-2])
            return feats if is_batch(img) else feats[0]
```

(c) Extend the `extract` docstring `Returns:` list. Keep the existing two keys and add:

```
                - image_size (tuple): (W, H) of the image, only for matchers with supports_batches
                - any model-specific extras that match() needs, only for matchers with supports_batches
```

- [ ] **Step 5: Run the tests**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "extract or batch" -v --timeout=300
```
Expected: the new tests and 1a's batch tests pass. `test_extract_keypoints[*]` is parametrized over models and slow: run it here only with `-k "extract_native or extract_batch"`.

- [ ] **Step 6: CI gate + commit**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m ruff check . && /opt/venv/reconstruction/bin/python -m ruff format --check . && git add vismatch/base_matcher.py tests/test_matchers.py && git commit -m "$(cat <<'EOF'
Extract features without a self-pair forward for batching matchers

Matchers with supports_batches implement _extract_features(imgs) and
extract() calls it on the whole batch, skipping forward()'s self-match
and RANSAC. Results add image_size (W, H) and any model extras.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `match(feats0, feats1)`

**Files:**
- Modify: `vismatch/base_matcher.py` (new method after `extract`)
- Test: `tests/test_matchers.py`

- [ ] **Step 1: Write the failing tests**

Append after `test_extract_native_batch`:

```python
def test_match_matches_forward():
    """match() on extract() outputs gives forward()'s matches, plus the keypoint rows behind them."""
    matcher = _GridMatcher()
    matcher.skip_ransac = True
    img0, img1 = torch.rand(3, 40, 60), torch.rand(3, 30, 50)
    feats0, feats1 = matcher.extract(img0), matcher.extract(img1)
    result, expected = matcher.match(feats0, feats1), matcher.forward(img0, img1)

    np.testing.assert_array_equal(result["matched_kpts0"], expected["matched_kpts0"])
    np.testing.assert_array_equal(result["matched_kpts1"], expected["matched_kpts1"])
    np.testing.assert_array_equal(result["matched_kpts0"], feats0["all_kpts0"][result["matched_idxs0"]])
    np.testing.assert_array_equal(result["matched_kpts1"], feats1["all_kpts0"][result["matched_idxs1"]])
    assert result["num_inliers"] == expected["num_inliers"] == 0


def test_match_drops_out_of_bounds_matches():
    """match() drops matches with a keypoint outside its image, keeping indices aligned with keypoints."""
    matcher = _GridMatcher()
    feats0, feats1 = matcher.extract(torch.rand(3, 40, 60)), matcher.extract(torch.rand(3, 40, 60))
    feats1["image_size"] = (31, 21)  # only the (0, 0) and (30, 20) keypoints are inside
    result = matcher.match(feats0, feats1)
    assert result["matched_idxs1"].tolist() == [0, 1]
    assert result["matched_confidences"].tolist() == [1.0, 1.0]


def test_match_not_implemented():
    """match() needs a matcher with supports_batches; others must use forward()."""
    with pytest.raises(NotImplementedError, match="use forward"):
        _CornerMatcher().match({}, {})
```

- [ ] **Step 2: Run the tests and check that they fail**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "test_match_" -v --timeout=300
```
Expected: 3 FAILED. torch.nn.Module raises `AttributeError: '_GridMatcher' object has no attribute 'match'`.

- [ ] **Step 3: Implement**

Insert into `BaseMatcher` directly after `extract`:

```python
    @torch.inference_mode()
    def match(self, feats0: dict, feats1: dict) -> dict:
        """Match two images from their extract() outputs, without detecting again. Needs supports_batches.

        Args:
            feats0 (dict): extract() output for img0; values may be np.ndarray or torch.Tensor on any device
            feats1 (dict): extract() output for img1

        Returns:
            dict: result dict with the keys of forward() except all_kpts0/1 and all_desc0/1, plus:
                - matched_idxs0 (np.ndarray): (N2,) rows of feats0["all_kpts0"] behind matched_kpts0
                - matched_idxs1 (np.ndarray): (N2,) rows of feats1["all_kpts0"] behind matched_kpts1
        """
        if not self.supports_batches:
            raise NotImplementedError(f"{self.name} cannot match precomputed features, use forward()")

        # Move features to the matcher's device, a no-op for features already there
        (w0, h0), (w1, h1) = feats0["image_size"], feats1["image_size"]
        feats0, feats1 = (
            {k: torch.as_tensor(v, device=self.device) for k, v in f.items() if k != "image_size"}
            for f in (feats0, feats1)
        )

        # self._match_features() returns indices into each keypoint table, and confidences or None
        idxs0, idxs1, matched_confidences = self._match_features(feats0, feats1)
        matched_kpts0, matched_kpts1 = to_numpy(feats0["all_kpts0"][idxs0]), to_numpy(feats1["all_kpts0"][idxs1])
        idxs0, idxs1, matched_confidences = to_numpy(idxs0), to_numpy(idxs1), to_numpy(matched_confidences)

        # Drop matches with a kpt outside its image, as forward() does
        valid = (
            (matched_kpts0 >= 0) & (matched_kpts0 < [w0, h0]) & (matched_kpts1 >= 0) & (matched_kpts1 < [w1, h1])
        ).all(1)
        matched_kpts0, matched_kpts1, idxs0, idxs1 = matched_kpts0[valid], matched_kpts1[valid], idxs0[valid], idxs1[valid]
        if matched_confidences is not None:
            matched_confidences = matched_confidences[valid]

        H, inlier_kpts0, inlier_kpts1 = self.compute_ransac(matched_kpts0, matched_kpts1)

        return {
            "num_inliers": len(inlier_kpts0),
            "H": H,
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
            "matched_confidences": matched_confidences,
            "matched_idxs0": idxs0,
            "matched_idxs1": idxs1,
        }
```

If `ruff format` reflows the long `matched_kpts0, ... = ...[valid]` line, accept its output.

- [ ] **Step 4: Run the tests**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "test_match_ or extract_native or batch" -v --timeout=300
```
Expected: all pass.

- [ ] **Step 5: CI gate + commit**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m ruff format . && /opt/venv/reconstruction/bin/python -m ruff check . && git add vismatch/base_matcher.py tests/test_matchers.py && git commit -m "$(cat <<'EOF'
Add match() over precomputed features for batching matchers

match(feats0, feats1) runs only the model's matching stage
(_match_features) on two extract() outputs, then applies forward()'s
out-of-image filter and RANSAC. Returns forward()'s match keys plus
matched_idxs0/1, the keypoint rows behind each match.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Native batched `forward`

**Files:**
- Modify: `vismatch/base_matcher.py` (the batch block at the top of `forward`)
- Test: `tests/test_matchers.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_forward_batch_native():
    """With supports_batches, a batch of pairs matches through extract() and match(), equal to each pair on its own."""
    matcher = _GridMatcher()
    imgs0, imgs1 = [torch.rand(3, 40, 60), torch.rand(3, 20, 30)], [torch.rand(3, 30, 50), torch.rand(3, 50, 70)]
    with patch.object(matcher, "_forward", wraps=matcher._forward) as forward_spy:
        results = matcher.forward(imgs0, imgs1)
    assert forward_spy.call_count == 0

    for result, img0, img1 in zip(results, imgs0, imgs1):
        expected = matcher.forward(img0, img1)
        for key in ("matched_kpts0", "matched_kpts1", "all_kpts0", "all_kpts1", "all_desc0", "all_desc1"):
            np.testing.assert_array_equal(result[key], expected[key])
```

Add `from unittest.mock import patch` to the file's imports.

- [ ] **Step 2: Run the test and check that it fails**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k forward_batch_native -v --timeout=300
```
Expected: FAIL on `assert forward_spy.call_count == 0`, because the 1a loop calls `_forward` twice.

- [ ] **Step 3: Implement**

Replace the 1a batch block in `forward` with:

```python
        # A batch of pairs is matched one pair at a time, unless the matcher batches natively
        if is_batch(img0) or is_batch(img1):
            assert is_batch(img0) and is_batch(img1) and len(img0) == len(img1), (
                "img0 and img1 must both be single images or batches of the same length"
            )
            if self.supports_batches:
                feats0, feats1 = self.extract(img0), self.extract(img1)
                return [
                    self.match(f0, f1)
                    | {"all_kpts0": f0["all_kpts0"], "all_kpts1": f1["all_kpts0"]}
                    | {"all_desc0": f0["all_desc0"], "all_desc1": f1["all_desc0"]}
                    for f0, f1 in zip(feats0, feats1)
                ]
            return [self.forward(i0, i1) for i0, i1 in zip(img0, img1)]
```

- [ ] **Step 4: Run all mock tests**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "not available_models and (batch or match or extract_native or supports or forward_matcher or bounds or confidence)" -v --timeout=300
```
Expected: all pass. That includes 1a's `test_forward_batch_matches_each_pair`, since `_CornerMatcher` still loops.

- [ ] **Step 5: CI gate + commit**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m ruff format --check . && /opt/venv/reconstruction/bin/python -m ruff check . && git add vismatch/base_matcher.py tests/test_matchers.py && git commit -m "$(cat <<'EOF'
Batch forward() natively for matchers with supports_batches

A batch of pairs is extracted in one extract() call per side, then each
pair is matched with match(); all_kpts/all_desc are merged back in, so
results carry forward()'s keys. Other matchers keep the per-pair loop.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: xfeat hooks

**Files:**
- Modify: `vismatch/im_models/xfeat.py`
- Test: `tests/test_matchers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_matchers.py`:

```python
def _overlap(kpts_a, kpts_b):
    """Fraction of keypoints in kpts_a that also appear in kpts_b (within 1e-3 px)."""
    if len(kpts_a) == 0:
        return 1.0 if len(kpts_b) == 0 else 0.0
    dists = np.linalg.norm(kpts_a[:, None] - kpts_b[None], axis=-1)
    return float((dists.min(1) < 1e-3).mean())


def _load_pair(matcher, test_image_paths):
    """The indoor test pair at 256 px, as (3, H, W) tensors."""
    return [matcher.load_image(p, resize=256) for p in test_image_paths]


@pytest.mark.parametrize("model_name", ["xfeat", "loma"])
def test_extract_matches_forward(model_name, device, test_image_paths):
    """For batching matchers, extract() returns exactly forward()'s keypoints and descriptors."""
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
    assert matcher.supports_batches
    img, _ = _load_pair(matcher, test_image_paths)
    feats, expected = matcher.extract(img), matcher.forward(img, img)
    np.testing.assert_array_equal(feats["all_kpts0"], expected["all_kpts0"])
    np.testing.assert_array_equal(feats["all_desc0"], expected["all_desc0"])


@pytest.mark.parametrize("model_name", ["xfeat", "loma"])
def test_match_matches_forward_model(model_name, device, test_image_paths):
    """For batching matchers, match() on extract() outputs gives exactly forward()'s matches."""
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
    matcher.skip_ransac = True
    img0, img1 = _load_pair(matcher, test_image_paths)
    feats0, feats1 = matcher.extract(img0), matcher.extract(img1)
    result, expected = matcher.match(feats0, feats1), matcher.forward(img0, img1)

    assert len(result["matched_kpts0"]) > 0
    np.testing.assert_array_equal(result["matched_kpts0"], expected["matched_kpts0"])
    np.testing.assert_array_equal(result["matched_kpts1"], expected["matched_kpts1"])
    if expected["matched_confidences"] is not None:
        np.testing.assert_array_equal(result["matched_confidences"], expected["matched_confidences"])
    np.testing.assert_array_equal(result["matched_kpts0"], feats0["all_kpts0"][result["matched_idxs0"]])
    np.testing.assert_array_equal(result["matched_kpts1"], feats1["all_kpts0"][result["matched_idxs1"]])


def test_extract_batch_xfeat(device, test_images):
    """xfeat extract() on a batch matches single-image extract(): exactly for one image, closely for several."""
    try:
        matcher = get_matcher("xfeat", device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate xfeat on {device}: {e}")
    singles = [matcher.extract(img) for img in test_images]

    (one,) = matcher.extract([test_images[0]])
    np.testing.assert_array_equal(one["all_kpts0"], singles[0]["all_kpts0"])
    np.testing.assert_array_equal(one["all_desc0"], singles[0]["all_desc0"])

    for batched, single in zip(matcher.extract(list(test_images)), singles):
        assert _overlap(batched["all_kpts0"], single["all_kpts0"]) >= 0.99


@pytest.mark.parametrize("model_name", ["xfeat", "loma"])
def test_forward_batch_native_model(model_name, device, test_image_paths):
    """A natively batched forward() matches nearly the same keypoints as forward() on each pair."""
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
    img0, img1 = _load_pair(matcher, test_image_paths)
    results = matcher.forward([img0, img1], [img1, img0])
    for result, (i0, i1) in zip(results, [(img0, img1), (img1, img0)]):
        expected = matcher.forward(i0, i1)
        assert _overlap(result["matched_kpts0"], expected["matched_kpts0"]) >= 0.99


def test_xfeat_supports_batches_sparse_only():
    """Only xfeat's sparse mode matches by descriptors alone, so only it batches natively."""
    try:
        flags = {name: get_matcher(name, device="cpu").supports_batches for name in ("xfeat", "xfeat-star")}
    except Exception as e:
        pytest.skip(f"Cannot instantiate xfeat: {e}")
    assert flags == {"xfeat": True, "xfeat-star": False}
```

`get_matcher` hard-codes `mode=` per name (`vismatch/__init__.py:325-333`: `xfeat` = sparse, `xfeat-star` = semi-dense). Passing `mode=` as well would raise `TypeError: got multiple values`.

- [ ] **Step 2: Run the tests and check that they fail**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "xfeat" -v --timeout=300
```
Expected: the xfeat cases FAIL on `assert matcher.supports_batches` or with `AttributeError: ... '_extract_features'`.

- [ ] **Step 3: Implement**

In `xFeatMatcher.__init__`, after `self.mode = mode`:

```python
        # Sparse mode matches descriptors by mutual nearest neighbour, so it can match precomputed features
        self.supports_batches = mode == "sparse"
```

Add after `preprocess`:

```python
    def _extract_features(self, imgs: list[Tensor]) -> list[dict]:
        # Same-size images are detected in one batched forward, others one at a time
        if all(img.shape == imgs[0].shape for img in imgs):
            outputs = self.model.detectAndCompute(self.preprocess(torch.stack(imgs)), top_k=self.max_num_keypoints)
        else:
            outputs = [self.model.detectAndCompute(self.preprocess(img), top_k=self.max_num_keypoints)[0] for img in imgs]
        return [{"all_kpts0": out["keypoints"], "all_desc0": out["descriptors"]} for out in outputs]

    def _match_features(self, feats0: dict, feats1: dict) -> tuple:
        idxs0, idxs1 = self.model.match(feats0["all_desc0"], feats1["all_desc0"], min_cossim=-1)
        return idxs0, idxs1, None
```

- [ ] **Step 4: Run the tests on CPU (the CI device) and on GPU**

```bash
cd /workspace/vismatch-batch-forward && CUDA_VISIBLE_DEVICES= /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "xfeat" -v --timeout=300
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "xfeat" -v --timeout=300
```
Expected: the xfeat cases pass on both. The loma cases still fail; Task 7 fixes them. Under TF32 a GPU bit-exact test can flip; if `test_match_matches_forward_model[xfeat]` fails only on GPU, rerun with `NVIDIA_TF32_OVERRIDE=0` and record the result in the PR notes.

- [ ] **Step 5: CI gate + commit**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m ruff format --check . && /opt/venv/reconstruction/bin/python -m ruff check . && git add vismatch/im_models/xfeat.py tests/test_matchers.py && git commit -m "$(cat <<'EOF'
Batch xfeat sparse extraction and match precomputed features

Same-size images go through one detectAndCompute call (7x faster per
image at B=32); match() runs XFeat.match on stored descriptors. Batched
keypoints are not bit-exact at B>1 (overlap >= 0.99); B=1 is exact.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: loma hooks

**Files:**
- Modify: `vismatch/im_models/loma.py`
- Test: the Task 6 tests already cover loma (`test_extract_matches_forward[loma]`, `test_match_matches_forward_model[loma]`, `test_forward_batch_native_model[loma]`)

- [ ] **Step 1: Run the loma tests and check that they fail**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "loma and (extract_matches or match_matches or batch_native)" -v --timeout=300
```
Expected: 3 FAILED, on `assert matcher.supports_batches` and similar.

- [ ] **Step 2: Implement**

In `LoMaMatcher.__init__`, after `self.matcher = LoMa(cfg).to(self.device)`:

```python

        # LoMa detects and describes each image on its own, so it can match precomputed features
        self.supports_batches = True
```

Add after `preprocess`:

```python
    def _extract_features(self, imgs):
        feats = []
        for img in imgs:
            img, (h, w) = self.preprocess(img)
            H, W = img.shape[-2:]
            kpts, desc, _, _ = self.matcher.detect_and_describe(img, self.max_num_keypoints)
            # Same pixel coords as _forward's all_kpts; the matcher consumes the normalized kpts, kept as an extra
            all_kpts = self.rescale_coords(to_pixel_coords(kpts[0], H, W), h, w, H, W) - 0.5
            feats.append({"all_kpts0": all_kpts, "all_desc0": desc[0], "kpts_normalized": kpts[0]})
        return feats

    def _match_features(self, feats0, feats1):
        k0, k1 = feats0["kpts_normalized"][None], feats1["kpts_normalized"][None]
        d0, d1 = feats0["all_desc0"][None], feats1["all_desc0"][None]
        scores = self.matcher(k0, k1, d0, d1)["scores"]
        m0, _, mscores0, _ = filter_matches(scores, self.matcher.cfg.filter_threshold)

        # LoMa returns bfloat16 confidences; cast to float so numpy can convert them.
        valid = m0[0] > -1
        return torch.where(valid)[0], m0[0][valid], mscores0[0][valid].float()
```

- [ ] **Step 3: Run the tests on CPU and GPU**

```bash
cd /workspace/vismatch-batch-forward && CUDA_VISIBLE_DEVICES= /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "loma and (extract_matches or match_matches or batch_native)" -v --timeout=300
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m pytest tests/test_matchers.py -k "loma and (extract_matches or match_matches or batch_native)" -v --timeout=300
```
Expected: 3 passed on each device.

- [ ] **Step 4: Full upstream CI gate for the branch**

```bash
cd /workspace/vismatch-batch-forward && /opt/venv/reconstruction/bin/python -m ruff check . && /opt/venv/reconstruction/bin/python -m ruff format --check . && CUDA_VISIBLE_DEVICES= CI=1 /opt/venv/reconstruction/bin/python -m pytest tests -vv -rs --timeout=300 2>&1 | tee $SP/fm_ci.log; echo "exit ${PIPESTATUS[0]}"
```
Expected:
- `exit 0`.
- Skips only for models that fail to instantiate. Compare the `SKIPPED` lines against the same command run on `upstream/main` in `/workspace/vismatch` (clean checkout of `main` @ `9d49b89`): if a skip is new, investigate.
- The run takes long (every model, CPU). Run it in tmux if it exceeds 10 min: `tmux new -d -s fmci '...'`.

- [ ] **Step 5: Commit**

```bash
cd /workspace/vismatch-batch-forward && git add vismatch/im_models/loma.py && git commit -m "$(cat <<'EOF'
Match precomputed LoMa features

_extract_features keeps LoMa's normalized keypoints as an extra so
match() can run only the learned matcher: 1033 -> 44 ms per pair when
features are reused. Pixel coords and matches are bit-exact with
forward().

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: README for batched extract + `match()`

**Files:**
- Modify: `/workspace/vismatch-batch-forward/README.md` (the section 1a added for batched matching)

- [ ] **Step 1: Find 1a's batch paragraph**

```bash
cd /workspace/vismatch-batch-forward && git show ceb0022 --stat && grep -n -i "batch" README.md
```

- [ ] **Step 2: Append directly after that paragraph**

```markdown
Matchers that detect each image independently (currently `xfeat` and `loma`) set `matcher.supports_batches = True`.
For them, `extract()` runs one batched detection, and `match()` reuses its outputs without detecting again:

```python
feats = matcher.extract([img0, img1, img2])  # one detection pass
result = matcher.match(feats[0], feats[2])  # forward()'s match keys + matched_idxs0/1
```
```

- [ ] **Step 3: Commit**

```bash
cd /workspace/vismatch-batch-forward && git add README.md && git commit -m "$(cat <<'EOF'
Document batched extract() and match()

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: 🔒 Push both branches (ask the user first)**

```bash
cd /workspace/vismatch-batch-loma && git push -u origin feat/fast-input
cd /workspace/vismatch-batch-forward && git push -u origin feat/feature-matching
```

  Opening the upstream PRs is a separate user decision:
  - fast-input and 1a as PRs
  - feature-matching as a draft
  - post the measured numbers on #74

  Do not open them from this plan.

---

### Task 9: `basis` branch and perf gate

**Files:**
- None committed in collab-splats; `$SP/gate_after.txt`

- [ ] **Step 1: Build `basis` in the main fork checkout**

  feature-matching contains 1a.

```bash
cd /workspace/vismatch && git status --short | grep -v '^ m ' ; git switch -c basis upstream/main && git merge --no-ff feat/fast-input -m "Merge feat/fast-input into basis" && git merge --no-ff feat/feature-matching -m "Merge feat/feature-matching into basis" && git log --oneline -5
```
Expected:
- The first command prints nothing (only the pre-existing submodule `m` markers are dirty).
- Both merges are clean, because fast-input touches only `utils.py` / `test_utils.py`.

- [ ] **Step 2: Run the perf gate against the basis tree**

  The second argument puts `/workspace/vismatch` first on `sys.path`.

```bash
cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python $SP/gate_fm.py gate /workspace/vismatch 2>&1 | tee $SP/gate_after.txt
```
Expected:
- The first line prints `/workspace/vismatch/vismatch/__init__.py`.
- Every line meets its gate:
  - input ≤ 0.5 ms/frame
  - xfeat extract ≤ 16.2 at 640×480 and ≤ 5.3 at full-res
  - xfeat match ratio ≤ 1.10
  - loma match ratio ≤ 1.10
- If a gate fails, stop and report the numbers; do not tune.

- [ ] **Step 3: 🔒 Push `basis` (ask the user first)**

```bash
cd /workspace/vismatch && git push -u origin basis && git rev-parse basis
```
Record the SHA as `<BASIS_SHA>` for Task 10.

---

### Task 10: collab-splats pin + drop `lomatch`

**Files:**
- Modify: `pyproject.toml:81-84` (the dependency lines), `pyproject.toml:282` (the `dataclasses` override comment, which names lomatch)
- Modify: `uv.lock`

- [ ] **Step 1: Edit the dependencies**

In `pyproject.toml`:
- Delete the two comment lines above `"lomatch>=1.0.0",` and the line itself.
- Replace `"vismatch==1.3.1",` with:

```toml
    # BasisResearch fork, `basis` branch: batched extract + match() (spec 2026-10-03-vismatch-feature-matching)
    "vismatch @ git+https://github.com/BasisResearch/vismatch@<BASIS_SHA>",
```

Check whether the `dataclasses` override is still needed:

```bash
cd /workspace/collab-splats && grep -n '"dataclasses' uv.lock | head; grep -n -B2 -A2 'name = "dataclasses"' uv.lock | head
```
- If only lomatch pulled `dataclasses`, delete the override line and its comment (`pyproject.toml:282-283`).
- Otherwise keep it and reword the comment to name the remaining dependent.

- [ ] **Step 2: Re-lock without touching the venv**

```bash
cd /workspace/collab-splats && uv lock 2>&1 | tail -5 && git diff --stat uv.lock && rtk proxy git diff uv.lock | grep -E '^[-+]name = |^[-+]source' | head -20
```
Expected:
- `lomatch` is removed.
- vismatch's source becomes `git = "https://github.com/BasisResearch/vismatch?rev=<BASIS_SHA>#<BASIS_SHA>"`.
- No other package versions move. If others move, stop and report.

- [ ] **Step 3: 🔒 Install the pin into the shared venv (ask the user first)**

  The venv is shared with concurrent sessions. Use `--no-deps`, because vismatch hard-pins `uniception==0.1.1` and `lightning==2.3.3` (see `docs/known-test-failures.md`, vismatch entry).

```bash
uv pip install --python /opt/venv/reconstruction/bin/python --no-deps --reinstall "vismatch @ git+https://github.com/BasisResearch/vismatch@<BASIS_SHA>"
uv pip uninstall --python /opt/venv/reconstruction/bin/python lomatch
cd /tmp && /opt/venv/reconstruction/bin/python - <<'EOF'
import json, pathlib, importlib.metadata as md
import vismatch
d = md.distribution("vismatch")
print("vismatch", vismatch.__file__)
print("direct_url", json.loads(d.read_text("direct_url.json"))["vcs_info"]["commit_id"])
print("loma vendored", (pathlib.Path(vismatch.__file__).parent / "third_party/LoMa/src/loma").is_dir())
print("xfeat vendored", (pathlib.Path(vismatch.__file__).parent / "third_party/accelerated_features/modules").is_dir())
EOF
```
Expected:
- `site-packages/vismatch/__init__.py`
- `direct_url <BASIS_SHA>`
- both vendored dirs `True`

If the vendored dirs are `False` (the git install skipped submodules), stop. The fallback in fork spec §3 is to build and pin a wheel; ask the user.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats && git commit --only pyproject.toml uv.lock -m "$(cat <<'EOF'
build(deps): pin vismatch to BasisResearch basis, drop lomatch

basis = upstream + uint8 input + batched extract/match() (spec
2026-10-03-vismatch-feature-matching). lomatch was never imported; a
bare `import loma` could shadow vismatch's vendored LoMa.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: `LocalMatcher` onto the vismatch API

**Files:**
- Modify: `collab_splats/localization/extractors.py` (imports :5-13, `LocalFeatures` :18-31, `FEATURE_MATCH_MODELS` :64-68, `__init__` :115-132, `_to_tensor` :138-144, `extract` :157-190, `match` :192-254)
- Modify: `collab_splats/localization/localizer.py:166-173` (`read_localization_db`)
- Test: `tests/localization/test_local_matcher.py`

- [ ] **Step 1: Rewrite the mock and the unit tests (failing first)**

In `tests/localization/test_local_matcher.py`:

(a) Imports: drop `FEATURE_MATCH_MODELS` from the `extractors` import.

(b) `_fake_vismatch_matcher`: replace the last three lines (`m = MagicMock()` through `return m`) with:

```python
    m = MagicMock()
    m.side_effect = lambda i0, i1: dict(result)  # __call__(img0, img1)
    m.supports_batches = False  # a MagicMock attribute would be truthy
    feats = {"all_kpts0": all_kpts0, "all_desc0": result["all_desc0"]}
    # vismatch extract: a list in -> one dict per image; a single image in -> one dict
    m.extract.side_effect = lambda imgs: [dict(feats) for _ in imgs] if isinstance(imgs, list) else dict(feats)
    return m
```

(c) `test_extract_handles_tensor_outputs` and `test_extract_asserts_pixel_frame` set `m.extract.return_value`. Change each so it builds its dict, then sets `m.extract.side_effect = lambda imgs: [dict(bad)]` (or `[dict(tensor_feats)]`). The tensor test becomes:

```python
@patch("vismatch.get_matcher")
def test_extract_handles_tensor_outputs(mock_get):
    # Real matchers may return torch tensors (check_types allows tensor or ndarray).
    m = _fake_vismatch_matcher()
    prev = m.extract([None])[0]
    tensor_feats = {"all_kpts0": torch.from_numpy(prev["all_kpts0"]), "all_desc0": torch.from_numpy(prev["all_desc0"])}
    m.extract.side_effect = lambda imgs: [dict(tensor_feats)]
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    feats = lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert feats.keypoints.shape == (8, 2) and feats.keypoints.dtype == torch.float32
    assert feats.descriptors.shape == (8, 64)
```

and the pixel-frame test:

```python
@patch("vismatch.get_matcher")
def test_extract_asserts_pixel_frame(mock_get):
    # Keypoints outside the input image bounds = coordinate-frame violation (92f2e4a class)
    m = _fake_vismatch_matcher()
    bad = {"all_kpts0": np.array([[500.0, 500.0]], dtype=np.float32), "all_desc0": np.zeros((1, 64), dtype=np.float32)}
    m.extract.side_effect = lambda imgs: [dict(bad)]
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    with pytest.raises(ValueError, match="pixel frame"):
        lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
```

(d) Replace `test_extract_passes_chw_unit_range_tensor` with:

```python
@patch("vismatch.get_matcher")
def test_extract_passes_chw_uint8_list(mock_get):
    # vismatch input: a list of (3, H, W) uint8 tensors; vismatch scales uint8 on device itself
    m = _fake_vismatch_matcher()
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    lm.extract(np.full((100, 120, 3), 255, dtype=np.uint8))
    (imgs,), _ = m.extract.call_args
    assert isinstance(imgs, list) and len(imgs) == 1
    assert imgs[0].shape == (3, 100, 120) and imgs[0].dtype == torch.uint8


@patch("vismatch.get_matcher")
def test_extract_list_returns_one_feature_set_per_image(mock_get):
    m = _fake_vismatch_matcher()
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    feats = lm.extract([np.zeros((100, 120, 3), np.uint8), np.zeros((100, 110, 3), np.uint8)])
    assert m.extract.call_count == 1  # one batched vismatch call
    assert [f.image_size for f in feats] == [(120, 100), (110, 100)]
```

(e) Replace `test_to_tensor_matches_cpu_conversion` with:

```python
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])
@patch("vismatch.get_matcher")
def test_to_tensor_keeps_dtype_on_device(mock_get, device, dtype):
    # Upload as-is (uint8 is 4x fewer bytes); vismatch's to_tensor_image scales uint8 to [0, 1]
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device=device, probe=False)
    rgb = np.random.default_rng(0).integers(0, 256, (37, 53, 3)).astype(np.uint8)
    image = rgb if dtype == np.uint8 else rgb.astype(np.float32) / 255.0

    out = lm._to_tensor(image)

    assert out.device.type == device and out.shape == (3, 37, 53)
    torch.testing.assert_close(out.cpu(), torch.from_numpy(image).permute(2, 0, 1), atol=0, rtol=0)
```

(f) Replace `test_descriptor_level_match_unsupported`, `test_match_mutual_nn_for_allowlisted_model`, `test_match_still_raises_for_non_listed_model`, `test_split_flag_off_for_unknown_wrappers` and `test_loma_match_without_payload_names_the_rebuild` with:

```python
@patch("vismatch.get_matcher")
def test_match_unsupported_without_vismatch_batching(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()  # supports_batches False
    lm = LocalMatcher("roma", device="cpu", probe=False)
    q = _one_hot_features([0, 1])
    with pytest.raises(NotImplementedError, match="match_images"):
        lm.match(q, q)


@patch("vismatch.get_matcher")
def test_match_passes_features_and_maps_indices(mock_get):
    m = _fake_vismatch_matcher()
    m.supports_batches = True
    m.match.return_value = {
        "matched_kpts0": np.array([[0.0, 0.0], [20.0, 20.0]], np.float32),
        "matched_kpts1": np.array([[30.0, 30.0], [10.0, 10.0]], np.float32),
        "matched_idxs0": np.array([0, 2]),
        "matched_idxs1": np.array([3, 1]),
    }
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu", probe=False)
    q = _one_hot_features([0, 1, 2, 3])
    db = _one_hot_features([3, 2, 1, 0], keypoints_normalized=True)
    res = lm.match(q, db)

    (f0, f1), _ = m.match.call_args
    assert f0["image_size"] == (100, 80) and "kpts_normalized" not in f0
    assert torch.equal(f1["kpts_normalized"], db.keypoints_normalized)
    np.testing.assert_array_equal(res.idx_q, [0, 2])
    np.testing.assert_array_equal(res.idx_db, [3, 1])
    np.testing.assert_array_equal(res.ref_px, [[30.0, 30.0], [10.0, 10.0]])
    assert res.idx_q.dtype == np.int64 and res.query_px.dtype == np.float32


@patch("vismatch.get_matcher")
def test_init_skips_vismatch_ransac(mock_get):
    # Callers verify geometry with pycolmap; vismatch's homography RANSAC is wasted work
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    assert lm._matcher.skip_ransac is True
```

and update `_one_hot_features` so the new test can request the extra and an image size:

```python
def _one_hot_features(rows, d=8, keypoints_normalized=False):
    """LocalFeatures whose descriptors are one-hot rows — mutual-NN is exactly identity."""
    kpts = np.stack([np.arange(len(rows)), np.arange(len(rows))], axis=1).astype(np.float32) * 10
    desc = np.eye(d, dtype=np.float32)[rows]
    norm = torch.from_numpy(kpts / 100) if keypoints_normalized else None
    return LocalFeatures(
        keypoints=torch.from_numpy(kpts), descriptors=torch.from_numpy(desc), keypoints_normalized=norm, image_size=(100, 80)
    )
```

`test_match_empty_descriptors_returns_empty` stays, with one change: add `mock_get.return_value.supports_batches = True` after the `mock_get.return_value = ...` line.

(g) GPU parity tests: replace the three `test_real_*` tests with:

```python
@requires_cuda
def test_real_loma_extract_keeps_normalized_payload(strict_fp32):
    """loma extract() fills keypoints_normalized (the payload match() needs) and image_size."""
    lm = LocalMatcher("loma")
    a, _ = _real_pair()
    feats = lm.extract(a)
    assert feats.keypoints_normalized is not None and len(feats.keypoints_normalized) == len(feats.keypoints)
    assert feats.image_size == (320, 240)


@requires_cuda
def test_real_loma_match_parity_after_zarr_roundtrip(tmp_path, strict_fp32):
    """match() on zarr-roundtripped features == the plain pair forward, byte-identical."""
    lm = LocalMatcher("loma")
    a, b = _real_pair(seed=4)
    ref = lm.match_images(a, b)
    assert len(ref) > 0

    # extract -> save via CameraLocalizer -> load -> match()
    feats = [lm.extract(a), lm.extract(b)]
    extractor = MagicMock(spec=LocalMatcher)
    extractor.extract.side_effect = list(feats)
    loc = CameraLocalizer(
        world_points=np.zeros((2, 8, 8, 3), dtype=np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        images=[a, b],
        ids=["a", "b"],
        extractor=extractor,
    )
    loc.save_index(tmp_path / "ff.zarr", "loma")
    loaded, _, _ = read_localization_db(tmp_path / "ff.zarr", "loma")
    assert all(f.keypoints_normalized is not None and f.image_size == (320, 240) for f in loaded)
    m = lm.match(loaded[0], loaded[1])
    np.testing.assert_array_equal(m.query_px, ref.query_px)
    np.testing.assert_array_equal(m.ref_px, ref.ref_px)
    np.testing.assert_array_equal(m.idx_q, ref.idx_q)  # native == recovered (probe-exact)
    np.testing.assert_array_equal(m.idx_db, ref.idx_db)


@requires_cuda
def test_real_xfeat_match_matches_pairwise(strict_fp32):
    """xfeat match() over extract() features == match_images() on the pair."""
    lm = LocalMatcher("xfeat")
    a, b = _real_pair(seed=5)
    m = lm.match(lm.extract(a), lm.extract(b))
    ref = lm.match_images(a, b)
    assert len(m) == len(ref) and len(m) > 0
    order_m, order_ref = np.argsort(m.idx_q), np.argsort(ref.idx_q)
    np.testing.assert_array_equal(m.idx_q[order_m], ref.idx_q[order_ref])
    np.testing.assert_array_equal(m.idx_db[order_m], ref.idx_db[order_ref])
```

Update the section divider comment from `# GPU parity gates (real models) — these license FEATURE_MATCH_MODELS membership` to `# GPU parity gates (real models) — vismatch match() vs the pair forward`.

- [ ] **Step 2: Run the tests and check that they fail**

```bash
cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -q 2>&1 | tail -20
```
Expected: failures in the rewritten tests: `TypeError ... image_size`, `skip_ransac` is a MagicMock, `extract` returns a dict, and so on.

- [ ] **Step 3: Implement `extractors.py`**

(a) Imports: delete `import sys` and `from kornia.feature import match_mnn`.

(b) `LocalFeatures`: change the `keypoints_normalized` comment and add a field:

```python
    keypoints_normalized: torch.Tensor | None = None  # (N, 2) loma model-grid coords its matcher consumes
    image_size: tuple[int, int] | None = None  # (W, H) of the image the keypoints were detected in
```

(c) Delete the `FEATURE_MATCH_MODELS` comment block and constant (`:64-68`).

(d) In `__init__`, replace the two `_split_loma_forward` lines (`:126-128`) with:

```python
        # Callers verify geometry with pycolmap; vismatch's homography RANSAC is wasted work
        self._matcher.skip_ransac = True
```

(e) `_to_tensor`:

```python
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """HxWx3 RGB (uint8, or float in [0, 1]) -> (3,H,W) on device, dtype kept; vismatch scales uint8."""
        # Upload as-is and convert on device: uint8 is 4x fewer bytes, and no max() sync
        return torch.from_numpy(np.ascontiguousarray(image)).to(self._device).permute(2, 0, 1)
```

(f) `extract`:

```python
    def extract(self, images: np.ndarray | list[np.ndarray]) -> LocalFeatures | list[LocalFeatures]:
        """Keypoints+descriptors for one HxWx3 RGB image, or for each image of a list in one vismatch call."""
        batch = isinstance(images, list)
        images = images if batch else [images]
        with torch.inference_mode():
            outs = self._matcher.extract([self._to_tensor(image) for image in images])

        # vismatch may hand back numpy or on-device tensors depending on the model
        feats = []
        for image, out in zip(images, outs):
            kpts = _to_numpy(out["all_kpts0"])
            self._check_pixel_frame(kpts, image.shape[:2], self._model_name)
            norm = out.get("kpts_normalized")
            feats.append(
                LocalFeatures(
                    keypoints=torch.from_numpy(kpts),
                    descriptors=torch.from_numpy(_to_numpy(out["all_desc0"])),
                    keypoints_normalized=None if norm is None else torch.from_numpy(_to_numpy(norm)),
                    image_size=(image.shape[1], image.shape[0]),
                )
            )
        return feats if batch else feats[0]
```

(g) `match`: replace the whole body:

```python
    def match(self, query: LocalFeatures, db: LocalFeatures) -> MatchResult:
        """Match precomputed features via vismatch match(); rows are native keypoint-table indices.

        Only for models with vismatch supports_batches (xfeat, loma) — no _recover_indices.
        """
        if not self._matcher.supports_batches:
            raise NotImplementedError(
                f"LocalMatcher('{self._model_name}') cannot match precomputed features "
                "(vismatch supports_batches is False). Use match_images()."
            )
        if len(query.descriptors) == 0 or len(db.descriptors) == 0:
            return _empty_match()
        out = self._matcher.match(self._vismatch_features(query), self._vismatch_features(db))
        if len(out["matched_idxs0"]) == 0:
            return _empty_match()
        return MatchResult(
            query_px=_to_numpy(out["matched_kpts0"]),
            ref_px=_to_numpy(out["matched_kpts1"]),
            idx_q=out["matched_idxs0"].astype(np.int64),
            idx_db=out["matched_idxs1"].astype(np.int64),
        )

    @staticmethod
    def _vismatch_features(feats: LocalFeatures) -> dict:
        """LocalFeatures -> vismatch extract() dict; keypoints_normalized feeds loma's matcher."""
        out = {"all_kpts0": feats.keypoints, "all_desc0": feats.descriptors, "image_size": feats.image_size}
        if feats.keypoints_normalized is not None:
            out["kpts_normalized"] = feats.keypoints_normalized
        return out
```

(h) Class docstring line 3: `extract() fills the zarr feature cache; match() matches cached features (xfeat, loma); match_images() is the pairwise path.`

- [ ] **Step 4: Implement `localizer.py`**

In `read_localization_db`'s `LocalFeatures(...)` construction (`:167-173`), add after `keypoints_normalized=f_norm,`:

```python
                image_size=(hw[1], hw[0]),
```

- [ ] **Step 5: Run the localization + geometry suites**

```bash
cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python -c "import collab_splats, vismatch; print(collab_splats.__file__); print(vismatch.__file__)" && /opt/venv/reconstruction/bin/python -m pytest tests/localization tests/geometry -q 2>&1 | tail -15; echo "exit ${PIPESTATUS[0]}"
```
Expected:
- `/workspace/collab-splats/collab_splats/__init__.py`, then site-packages vismatch.
- `exit 0`, or only failures already listed in `docs/known-test-failures.md`.
- The three `test_real_*` tests pass on GPU; check they ran with `-rA | grep test_real`.

- [ ] **Step 6: Check nothing else references the deleted names**

```bash
cd /workspace/collab-splats && git grep -n "FEATURE_MATCH_MODELS\|_split_loma_forward\|match_mnn\|lomatch" -- ':!docs/superpowers' ':!graphify-out' ':!CLAUDE.md'
```
Expected: hits only in `docs/known-test-failures.md`, which Task 12 fixes.

- [ ] **Step 7: Commit**

```bash
cd /workspace/collab-splats && black collab_splats/localization/extractors.py collab_splats/localization/localizer.py tests/localization/test_local_matcher.py && isort collab_splats/localization/extractors.py tests/localization/test_local_matcher.py && git commit --only collab_splats/localization/extractors.py collab_splats/localization/localizer.py tests/localization/test_local_matcher.py -m "$(cat <<'EOF'
refactor(localization): LocalMatcher on vismatch batched extract + match()

- extract takes one image or a list -> one vismatch extract() call
- match() delegates to vismatch match(); FEATURE_MATCH_MODELS, kornia
  match_mnn and the hand-rolled loma split are gone
- uint8 upload (vismatch scales on device), skip_ransac on
- LocalFeatures.image_size, filled from zarr hw at load

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

  If black/isort touch lines outside our change, revert those hunks before committing. The memory file notes that the venv black is newer than the repo's.

---

### Task 12: Docs — known-test-failures names

**Files:**
- Modify: `docs/known-test-failures.md` (~:102-105, :340-352)

- [ ] **Step 1: Edit**
  - In the 2026-08-21 TF32 entry, append to the heading line: ` (test renamed 2026-10-03: test_real_xfeat_match_matches_pairwise)`.
  - In line 105's parenthetical, replace `licenses \`xfeat\`'s membership in \`FEATURE_MATCH_MODELS\`` with `gates xfeat's vismatch match() parity`.
  - In the 16-test vismatch-missing entry (:340-352), add one line under the code block: `Names as of that entry; the 2026-10-03 vismatch feature-matching refactor renamed or removed the split/allowlist tests.`
  - Do not rewrite the historic names: that entry records what failed at the time.

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats && git commit --only docs/known-test-failures.md -m "$(cat <<'EOF'
docs(tests): note local-matcher test renames after vismatch match()

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Amend the spec with the deltas

**Files:**
- Modify: `docs/superpowers/specs/2026-10-03-vismatch-feature-matching-design.md`

- [ ] **Step 1: Edit**
  - §2 table: delete the "Base `_extract_features` / `_match_features` raising NotImplementedError" row. Change the `extract(list)` row to "uses the hooks when `supports_batches`".
  - §2 notes: replace the bf16 bullet with `LoMa kpts/desc are fp32 (forward's to_numpy would raise otherwise); only confidences get .float(), as in _forward`.
  - §3 table: in the `extract` row, append `; image_size at load from the zarr hw attr`.
  - Gates table: remove the `test_forward_batch_matches_loop | sift-nn` row. Add `mock _GridMatcher tests: extract/match/native forward/out-of-bounds | — | exact`. Change `test_match_not_implemented | sift-nn` to `| _CornerMatcher mock |`.
  - Status line: `approved design; plan 2026-10-03-vismatch-feature-matching.md`.

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats && git add -f docs/superpowers/specs/2026-10-03-vismatch-feature-matching-design.md && git commit --only docs/superpowers/specs/2026-10-03-vismatch-feature-matching-design.md -m "$(cat <<'EOF'
docs(specs): vismatch feature matching — planning deltas

No base hook defs (supports_batches is the gate), image_size from zarr
hw, mock tests replace sift-nn, no LoMa bf16 casts.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Track-build phase gate

**Files:**
- Create: `$SP/gate_tracks.py` (scratch)

- [ ] **Step 1: Write the script**

  It measures the two gated phases on the 984-frame GH run with batched extract and GPU-resident features. Filter, db and verify are report-only in the spec, and this script leaves them out.

```python
# Track-build phase gate: extract + match.match on GH010229 f1000 (spec 2026-10-03, "Performance")
# - usage: gate_tracks.py <n frames> <overlap>; full-scale = 984 frames / 24195 pairs
# - extract: LocalMatcher.extract(list) in chunks of 32, features moved to GPU once
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from collab_splats.localization.extractors import LocalMatcher

N, OVERLAP = int(sys.argv[1]), int(sys.argv[2])
paths = sorted(Path("/workspace/outputs/ocr_viewer/GH010229_f1000_ref_loger/images").glob("frame_*.png"))[:N]
with ThreadPoolExecutor(8) as pool:
    src = list(pool.map(lambda q: np.ascontiguousarray(cv2.imread(str(q))[..., ::-1]), paths))

m = LocalMatcher("xfeat", probe=False)
m._matcher.max_num_keypoints = 2048
m.extract(src[:2])

# Batched extract, features resident on GPU
torch.cuda.synchronize()
t = time.perf_counter()
feats = [f for i in range(0, N, 32) for f in m.extract(src[i : i + 32])]
feats = [replace(f, keypoints=f.keypoints.cuda(), descriptors=f.descriptors.cuda()) for f in feats]
torch.cuda.synchronize()
t_extract = time.perf_counter() - t

# Sequential pairs, vismatch match() per pair
pairs = [(a, b) for a in range(N) for b in range(a + 1, min(N, a + OVERLAP + 1))]
t = time.perf_counter()
n = sum(len(m.match(feats[a], feats[b])) for a, b in pairs)
torch.cuda.synchronize()
t_match = time.perf_counter() - t

print(f"collab_splats {sys.modules['collab_splats'].__file__}")
print(f"N {N} pairs {len(pairs)} matches {n}")
print(f"extract     {t_extract:7.2f}s  {1e3 * t_extract / N:6.2f} ms/frame  ~{t_extract / N * 984:6.1f}s full  gate <= 15")
print(f"match.match {t_match:7.2f}s  {1e3 * t_match / len(pairs):6.2f} ms/pair  ~{t_match / len(pairs) * 24195:6.1f}s full  gate <= 28")
```

- [ ] **Step 2: Run it (heavy: tmux, no parallel GPU jobs)**

```bash
tmux new -d -s gtracks "cd /workspace/collab-splats && /opt/venv/reconstruction/bin/python $SP/gate_tracks.py 984 25 > $SP/gate_tracks.txt 2>&1"
```
Then wait for the `tmux` session to exit, using `Monitor` with an until-loop on `tmux has-session -t gtracks`. Never use `pgrep -f`; it matches itself. Then:

```bash
cat $SP/gate_tracks.txt
```
Expected:
- `collab_splats` path under `/workspace/collab-splats`.
- extract full ≤ 15 s; match.match full ≤ 28 s (handoff before: ~142 s extract, ~26–28 s match at 1.1 ms/pair).
- If a gate fails, report the numbers; do not tune.

---

### Task 15: Final self-review and report

- [ ] **Step 1: Owners' agreement check (#74).** Confirm each of the following:
  - batching only via `forward()` with a batch input
  - `supports_batches` gates native batching
  - the non-native loop is unchanged
  - no caching code
  - the split is limited to descriptor models (xfeat sparse, loma)
  - every other model's `_forward` is untouched: `git -C /workspace/vismatch diff upstream/main basis --stat` lists only `base_matcher.py`, `utils.py`, `xfeat.py`, `loma.py`, tests and the README
- [ ] **Step 2: Acceleration check.** Put `$SP/gate_after.txt` and `$SP/gate_tracks.txt` side by side with the spec's "Measured levers" table.
- [ ] **Step 3: Caller check.**
  - `git grep -n "\.extract(\|\.match(\|match_images(" -- collab_splats` must list only these call sites:
    - `localizer.py:264,506,896` (single image, now carrying `image_size`)
    - `localizer.py:909` (`match_images`)
    - `extractors.py` internals
  - `match_images` and `_probe_index_stability` call vismatch `forward` / `extract` on single images: unchanged contract.
- [ ] **Step 4: Report to the user.** Include:
  - the gate numbers and branch SHAs
  - which 🔒 steps ran and which are still owed
  - the PRs still to open: fast-input, 1a, the feature-matching draft and #74 numbers
  - that the CLAUDE.md `vismatch-fork` in-flight entry needs an update once the pending CLAUDE.md change by another session lands (don't fold it in)
  - that the rgbd-ba worktree's `match_batch` will conflict when it rebases
