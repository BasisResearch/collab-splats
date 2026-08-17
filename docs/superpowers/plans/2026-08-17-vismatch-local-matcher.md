# VisMatch Local Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** vismatch becomes the sole Stage-2 local-matching provider via one `LocalMatcher` class; Stage 1 retrieval and Stage 3 PnP stay untouched.

**Architecture:** `LocalMatcher` wraps `vismatch.get_matcher` behind the existing `LocalFeatures`/`MatchResult` contracts. vismatch has no descriptor-level matching (its `extract()` is literally `forward(img, img)`), so `CameraLocalizer` gains a pairwise image path gated by DinoSalad retrieval top-K, and `geometry/verification.py` gains an image-pair branch with keypoint-index recovery. Legacy extractors retire only after a human-gated parity benchmark.

**Tech Stack:** vismatch (PyPI, exact pin) + `[tool.uv] override-dependencies`, torch, pycolmap, zarr.

**Spec:** `docs/superpowers/specs/2026-08-17-vismatch-local-matcher-design.md`

**Environment:** `/opt/venv/reconstruction/bin/python` (py3.11). Tests: `/opt/venv/reconstruction/bin/python -m pytest`. GPU/model tests marked; unit tests all run with mocked vismatch.

---

## File map

| File | Change |
|---|---|
| `pyproject.toml` | Add `vismatch` dep + `[tool.uv] override-dependencies` |
| `collab_splats/localization/extractors.py` | Add `LocalMatcher`, `resolve_matcher()`; legacy classes untouched until Task 10 |
| `collab_splats/localization/localizer.py` | Pairwise path in `localize()`, ref-image retention + retrieval gate in `from_feedforward` |
| `collab_splats/geometry/verification.py` | Image-pair branch with index recovery |
| `collab_splats/wrapper/reconstructor.py` | `resolve_matcher` in `_build_localization_db`; verify-capability hard error |
| `configs/base.yaml` | `localization.extractor` accepts vismatch model names; `localization.top_k` |
| `setup.sh` | Weights pre-fetch step |
| `evals/scripts/eval_localization_parity.py` | Parity benchmark (Task 9) |
| `tests/localization/test_local_matcher.py` | New unit tests |

---

### Task 1: Dependency — vismatch via uv override

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency and override**

In `pyproject.toml` `[project] dependencies`, add (near the other localization deps, after `"lomatch>=1.0.0"`):

```toml
    "vismatch==1.3.1",
```

Add a new section (after the existing `[tool.uv.sources]` block if present, else at the first `[tool.uv]` position):

```toml
[tool.uv]
# vismatch hard-pins uniception==0.1.1 (its ufm model) and lightning==2.3.3 (its edm
# model). We need uniception 0.1.7 (MapAnything imports 15 symbols absent in 0.1.1)
# and lightning>=2.6. Overrides replace vismatch's constraints during resolution;
# ufm/edm are blocklisted in LocalMatcher. Measured 2026-08-17 (see spec).
override-dependencies = ["uniception==0.1.7", "lightning>=2.6"]
```

If a `[tool.uv]` table already exists, merge the key into it instead of duplicating the table.

- [ ] **Step 2: Lock and sync**

Run: `cd /workspace/collab-splats && uv lock && uv sync`
Expected: lock resolves (vismatch 1.3.1, uniception stays 0.1.7, lightning stays >=2.6, kornia downgrades 0.8.3→0.8.2). If `uv lock` errors, STOP and report — do not force.

- [ ] **Step 3: Smoke-verify imports still work**

Run: `/opt/venv/reconstruction/bin/python -c "import vismatch; import kornia; from mapanything.models.mapanything import model as _m; print('vismatch', vismatch.__version__ if hasattr(vismatch,'__version__') else 'ok', '| kornia', kornia.__version__)"`
Expected: prints versions, no ImportError. MapAnything import proves the uniception override held.

- [ ] **Step 4: Run existing localization + verification tests (kornia downgrade canary)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/ tests/geometry/test_verification.py -q -p no:randomly`
Expected: same pass/fail set as before this task (check `docs/known-test-failures.md`). Any NEW failure = kornia 0.8.2 fallout — STOP and report.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(localization): add vismatch dep with uv override-dependencies"
```

---

### Task 2: `LocalMatcher.extract()` (TDD, mocked vismatch)

**Files:**
- Modify: `collab_splats/localization/extractors.py`
- Create: `tests/localization/test_local_matcher.py`

vismatch import stays inside `LocalMatcher.__init__` (optional-heavy-dep exception in CLAUDE.md code style) so `extractors.py` imports without vismatch's model zoo.

- [ ] **Step 1: Write failing tests for extract()**

Create `tests/localization/test_local_matcher.py`:

```python
"""Unit tests for LocalMatcher — vismatch mocked throughout; no model downloads."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from collab_splats.localization.extractors import LocalFeatures, LocalMatcher, MatchResult


def _fake_vismatch_matcher(n_kpts=8, d=64, stable_indices=True):
    """Mock of a vismatch BaseMatcher: forward(img0, img1) -> result dict."""
    rng = np.random.default_rng(0)
    all_kpts0 = rng.uniform(0, 100, (n_kpts, 2)).astype(np.float32)
    all_kpts1 = rng.uniform(0, 100, (n_kpts, 2)).astype(np.float32)
    if stable_indices:
        matched0, matched1 = all_kpts0[:4], all_kpts1[:4]  # exact rows
    else:
        matched0, matched1 = all_kpts0[:4] + 0.3, all_kpts1[:4] + 0.3  # refined coords
    result = {
        "num_inliers": 4, "H": np.eye(3),
        "all_kpts0": all_kpts0, "all_kpts1": all_kpts1,
        "all_desc0": rng.standard_normal((n_kpts, d)).astype(np.float32),
        "all_desc1": rng.standard_normal((n_kpts, d)).astype(np.float32),
        "matched_kpts0": matched0, "matched_kpts1": matched1,
        "inlier_kpts0": matched0[:3], "inlier_kpts1": matched1[:3],
        "matched_confidences": np.ones(4, dtype=np.float32),
    }
    m = MagicMock()
    m.side_effect = lambda i0, i1: dict(result)  # __call__(img0, img1)
    m.extract.return_value = {"all_kpts0": all_kpts0, "all_desc0": result["all_desc0"]}
    return m


@patch("vismatch.get_matcher")
def test_extract_returns_local_features(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    feats = lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert isinstance(feats, LocalFeatures)
    assert feats.keypoints.shape == (8, 2) and feats.keypoints.dtype == torch.float32
    assert feats.descriptors.shape == (8, 64)
    mock_get.assert_called_once_with("disk-lightglue", device="cpu")


@patch("vismatch.get_matcher")
def test_extract_asserts_pixel_frame(mock_get):
    # Keypoints outside the input image bounds = coordinate-frame violation (92f2e4a class)
    m = _fake_vismatch_matcher()
    bad = m.extract.return_value.copy()
    bad["all_kpts0"] = np.array([[500.0, 500.0]], dtype=np.float32)
    bad_desc = np.zeros((1, 64), dtype=np.float32)
    bad["all_desc0"] = bad_desc
    m.extract.return_value = bad
    mock_get.return_value = m
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    with pytest.raises(ValueError, match="pixel frame"):
        lm.extract(np.zeros((100, 100, 3), dtype=np.uint8))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -q -p no:randomly`
Expected: FAIL — `ImportError: cannot import name 'LocalMatcher'`.

- [ ] **Step 3: Implement `LocalMatcher.__init__` + `extract()`**

Append to `collab_splats/localization/extractors.py` (after `LomaGExtractor`, before end of file):

```python
########################################
# VisMatch-backed matcher
########################################

# Models whose base deps we override away — vismatch would crash at model load.
_VISMATCH_DEP_BLOCKLIST = {
    "ufm": "requires uniception==0.1.1; this project pins 0.1.7 for MapAnything",
    "edm": "requires lightning==2.3.3; this project resolves lightning>=2.6",
}

# Upstream model licenses that forbid commercial use. Seeded from upstream LICENSE
# files (verified in Task 5); vismatch's own wrapper is BSD-3 but does not relicense
# the models it wraps.
_VISMATCH_LICENSE_BLOCKLIST = {
    "superglue": "Magic Leap research-only license",
    "superpoint-lightglue": "SuperPoint weights: Magic Leap research-only license",
    "superpoint-sphereglue": "SuperPoint weights: Magic Leap research-only license",
    "minima-superpoint-lightglue": "SuperPoint weights: Magic Leap research-only license",
    "duster": "DUSt3R: CC BY-NC-SA 4.0 (non-commercial)",
    "master": "MASt3R: CC BY-NC-SA 4.0 (non-commercial)",
    "gim-lightglue": "GIM: academic-use-only license",
    "gim-dkm": "GIM: academic-use-only license",
}


class LocalMatcher(BaseLocalExtractor):
    """Stage-2 local matcher backed by the vismatch model zoo.

    One class for every vismatch model; the model name is data, not a subclass.
    extract() fills the zarr feature cache; match_images() is the pairwise path
    (vismatch exposes no descriptor-level matching). The features-based match()
    inherited from BaseLocalExtractor is unsupported and raises.
    """

    def __init__(self, model_name: str, device: str | None = None, probe: bool = True):
        for blocklist, kind in ((_VISMATCH_DEP_BLOCKLIST, "dependency"), (_VISMATCH_LICENSE_BLOCKLIST, "license")):
            if model_name in blocklist:
                raise ValueError(f"vismatch model '{model_name}' blocked ({kind}): {blocklist[model_name]}")
        # Heavy optional dep: vismatch pulls the full model zoo machinery.
        import vismatch

        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._matcher = vismatch.get_matcher(model_name, device=self._device)
        # Set by _probe_index_stability(); None until probed.
        self.has_stable_indices: bool | None = None
        if probe:
            self._probe_index_stability()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 RGB -> (3,H,W) float [0,1] on device (vismatch input contract)."""
        t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        if t.max() > 1.5:  # uint8-scale input
            t = t / 255.0
        return t.to(self._device)

    @staticmethod
    def _check_pixel_frame(kpts: np.ndarray, hw: tuple[int, int], what: str) -> None:
        """Guard the 92f2e4a bug class: keypoints must be in the INPUT image's pixel frame."""
        if len(kpts) and (kpts.min() < -0.5 or kpts[:, 0].max() > hw[1] - 0.5 or kpts[:, 1].max() > hw[0] - 0.5):
            raise ValueError(
                f"vismatch '{what}' keypoints outside input pixel frame {hw}: "
                f"x∈[{kpts[:, 0].min():.1f},{kpts[:, 0].max():.1f}] y∈[{kpts[:, 1].min():.1f},{kpts[:, 1].max():.1f}] — "
                f"model '{kpts.shape}' likely returns coords at its internal resolution"
            )

    def extract(self, image: np.ndarray) -> LocalFeatures:
        """Extract keypoints+descriptors (vismatch runs a self-pair forward internally)."""
        hw = image.shape[:2]
        with torch.inference_mode():
            out = self._matcher.extract(self._to_tensor(image))
        kpts = np.asarray(out["all_kpts0"], dtype=np.float32)
        descs = np.asarray(out["all_desc0"], dtype=np.float32)
        self._check_pixel_frame(kpts, hw, self._model_name)
        return LocalFeatures(keypoints=torch.from_numpy(kpts), descriptors=torch.from_numpy(descs))

    def match(self, query: LocalFeatures, db: LocalFeatures, image_hw: tuple[int, int]) -> MatchResult:
        raise NotImplementedError(
            f"LocalMatcher('{self._model_name}') has no descriptor-level matching — "
            "vismatch matches image pairs only. Use match_images()."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -q -p no:randomly`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git commit -m "feat(localization): LocalMatcher.extract via vismatch (mock-tested)"
```

---

### Task 3: `LocalMatcher.match_images()` — pre-RANSAC matches + index recovery

**Files:**
- Modify: `collab_splats/localization/extractors.py`
- Test: `tests/localization/test_local_matcher.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/localization/test_local_matcher.py`:

```python
@patch("vismatch.get_matcher")
def test_match_images_pre_ransac_with_indices(mock_get):
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=True)
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    lm.has_stable_indices = True
    q = np.zeros((100, 100, 3), dtype=np.uint8)
    m = lm.match_images(q, q)
    assert isinstance(m, MatchResult)
    assert len(m) == 4  # pre-RANSAC matched_kpts, NOT the 3 homography inliers
    assert m.idx_q is not None and m.idx_db is not None
    # idx must point at the exact rows of the extract()-visible keypoint table
    fake = mock_get.return_value(q, q)
    np.testing.assert_array_equal(fake["all_kpts0"][m.idx_q], m.query_px)
    np.testing.assert_array_equal(fake["all_kpts1"][m.idx_db], m.ref_px)


@patch("vismatch.get_matcher")
def test_match_images_no_indices_when_unstable(mock_get):
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=False)
    lm = LocalMatcher("roma", device="cpu", probe=False)
    lm.has_stable_indices = False
    m = lm.match_images(np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8))
    assert len(m) == 4 and m.idx_q is None and m.idx_db is None


@patch("vismatch.get_matcher")
def test_features_match_raises(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    f = lm.extract(np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(NotImplementedError, match="match_images"):
        lm.match(f, f, (100, 100))


@patch("vismatch.get_matcher")
def test_probe_sets_stability_flag(mock_get):
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=True)
    lm = LocalMatcher("disk-lightglue", device="cpu")  # probe=True default
    assert lm.has_stable_indices is True
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=False)
    lm2 = LocalMatcher("roma", device="cpu")
    assert lm2.has_stable_indices is False


def test_blocklists_raise():
    with pytest.raises(ValueError, match="uniception"):
        LocalMatcher("ufm", device="cpu")
    with pytest.raises(ValueError, match="research-only"):
        LocalMatcher("superglue", device="cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -q -p no:randomly`
Expected: new tests FAIL (`match_images` / `_probe_index_stability` missing); Task 2 tests still pass.

- [ ] **Step 3: Implement**

Append to the `LocalMatcher` class:

```python
    @staticmethod
    def _recover_indices(matched: np.ndarray, table: np.ndarray) -> np.ndarray | None:
        """Map matched coordinates to exact rows of the keypoint table; None if any miss.

        Exact float equality on purpose: a coordinate a matcher refined off its table
        row (XFeatStar behaviour) must fail here, not silently map to the nearest row.
        """
        if len(table) == 0:
            return None
        # (K, N) exact row-equality; argmax over N gives the row per match
        eq = (matched[:, None, :] == table[None, :, :]).all(axis=2)
        if not eq.any(axis=1).all():
            return None
        return eq.argmax(axis=1).astype(np.int64)

    def match_images(self, query_image: np.ndarray, ref_image: np.ndarray) -> MatchResult:
        """Pairwise match two HxWx3 uint8 RGB images. Pre-RANSAC matches.

        vismatch's own RANSAC fits a homography — a planar-scene model that is the
        wrong geometric filter for 3D localization. We take matched_kpts (pre-RANSAC)
        and let PnP LO-RANSAC / epipolar verification do the filtering.
        """
        q_hw, r_hw = query_image.shape[:2], ref_image.shape[:2]
        with torch.inference_mode():
            out = self._matcher(self._to_tensor(query_image), self._to_tensor(ref_image))
        q_px = np.asarray(out["matched_kpts0"], dtype=np.float32)
        r_px = np.asarray(out["matched_kpts1"], dtype=np.float32)
        if len(q_px) == 0:
            return _empty_match()
        self._check_pixel_frame(q_px, q_hw, self._model_name)
        self._check_pixel_frame(r_px, r_hw, self._model_name)

        idx_q = idx_db = None
        if self.has_stable_indices:
            idx_q = self._recover_indices(q_px, np.asarray(out["all_kpts0"], dtype=np.float32))
            idx_db = self._recover_indices(r_px, np.asarray(out["all_kpts1"], dtype=np.float32))
            if idx_q is None or idx_db is None:
                logger.warning(
                    "LocalMatcher(%s): index recovery failed on a pair despite passing the "
                    "probe — treating this pair as index-less", self._model_name,
                )
                idx_q = idx_db = None
        return MatchResult(query_px=q_px, ref_px=r_px, idx_q=idx_q, idx_db=idx_db)

    def _probe_index_stability(self) -> None:
        """One synthetic pair through the model: are matched kpts exact keypoint-table rows?

        Two conditions must BOTH hold for verify-compatibility:
          (a) within-call: matched_kpts are exact rows of all_kpts (no per-pair refinement);
          (b) cross-call: extract() keypoints reproduce (deterministic detection),
              so cache-time and match-time tables agree.
        """
        rng = np.random.default_rng(7)
        img = (rng.uniform(0, 255, (256, 320, 3))).astype(np.uint8)
        img2 = np.roll(img, 8, axis=1)  # shifted copy — guarantees some matches for most models
        with torch.inference_mode():
            out = self._matcher(self._to_tensor(img), self._to_tensor(img2))
            ext = self._matcher.extract(self._to_tensor(img))
        within = (
            len(out["matched_kpts0"]) > 0
            and self._recover_indices(np.asarray(out["matched_kpts0"], np.float32),
                                      np.asarray(out["all_kpts0"], np.float32)) is not None
            and self._recover_indices(np.asarray(out["matched_kpts1"], np.float32),
                                      np.asarray(out["all_kpts1"], np.float32)) is not None
        )
        pair_table = np.asarray(out["all_kpts0"], np.float32)
        ext_table = np.asarray(ext["all_kpts0"], np.float32)
        cross = pair_table.shape == ext_table.shape and np.array_equal(pair_table, ext_table)
        self.has_stable_indices = bool(within and cross)
        logger.info(
            "LocalMatcher(%s): index probe — within-call %s, cross-call %s → stable_indices=%s",
            self._model_name, within, cross, self.has_stable_indices,
        )
```

Note for the mocked probe test: the fake matcher's `extract` returns the same `all_kpts0` as its forward, so `cross` is True; `stable_indices=True/False` follows the `matched` rows being exact/offset.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -q -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git commit -m "feat(localization): LocalMatcher.match_images with index recovery + probe"
```

---

### Task 4: `resolve_matcher()` + reconstructor wiring

**Files:**
- Modify: `collab_splats/localization/extractors.py`
- Modify: `collab_splats/wrapper/reconstructor.py:465` (`_build_localization_db`)
- Test: `tests/localization/test_local_matcher.py`

- [ ] **Step 1: Write failing test**

```python
@patch("vismatch.get_matcher")
def test_resolve_matcher(mock_get):
    from collab_splats.localization.extractors import DiskExtractor, resolve_matcher

    # Legacy registry key -> legacy class untouched
    assert resolve_matcher.__module__ == "collab_splats.localization.extractors"
    with patch.object(DiskExtractor, "__init__", return_value=None) as init:
        m = resolve_matcher("disk")
        assert isinstance(m, DiskExtractor) and init.called
    # Unknown key -> vismatch model
    mock_get.return_value = _fake_vismatch_matcher()
    m = resolve_matcher("disk-lightglue", probe=False)
    assert isinstance(m, LocalMatcher) and m.model_name == "disk-lightglue"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py::test_resolve_matcher -q -p no:randomly`
Expected: FAIL — `resolve_matcher` not defined.

- [ ] **Step 3: Implement resolver**

Append to `extractors.py` (module level, after `LocalMatcher`):

```python
def resolve_matcher(name: str, **kwargs) -> BaseLocalExtractor:
    """Registry key -> legacy extractor; anything else -> LocalMatcher(vismatch name).

    Transition-period seam: goes away with the registry once the parity gate
    retires the legacy extractors (see spec, Migration step 3).
    """
    if name in BaseLocalExtractor._registry:
        return BaseLocalExtractor.get(name)()
    return LocalMatcher(name, **kwargs)
```

- [ ] **Step 4: Wire into `_build_localization_db`**

In `collab_splats/wrapper/reconstructor.py`, replace (currently line 459 + 465):

```python
    from collab_splats.localization.extractors import BaseLocalExtractor
```
```python
    extractor = BaseLocalExtractor.get(extractor_name)()
```

with:

```python
    from collab_splats.localization.extractors import resolve_matcher
```
```python
    extractor = resolve_matcher(extractor_name)
```

- [ ] **Step 5: Run tests + wrapper suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py tests/wrapper/ -q -p no:randomly`
Expected: PASS (same known-failure set as `docs/known-test-failures.md`).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/localization/extractors.py collab_splats/wrapper/reconstructor.py tests/localization/test_local_matcher.py
git commit -m "feat(localization): resolve_matcher seam — registry keys or vismatch names"
```

---

### Task 5: Verify the license blocklist against upstream LICENSE files

**Files:**
- Modify: `collab_splats/localization/extractors.py` (`_VISMATCH_LICENSE_BLOCKLIST` only, if findings differ)

- [ ] **Step 1: Check each seeded entry + scan for missed restricted models**

For each model in `_VISMATCH_LICENSE_BLOCKLIST` plus these candidates — `roma`, `tiny-roma`, `minima-roma`, `loftr`, `aspanformer`, `omniglue`, `patch2pix`, `r2d2`, `d2net`, `xfeat`, `disk-lightglue`, `aliked-lightglue`, `dedode`, `loma`, `loma-r`, `rdd-star`, `topicfm` — open the upstream repo's LICENSE (repo links: https://github.com/gmberton/vismatch/blob/main/docs/source/model_details.md). Classification rule: CC-*-NC, "research/academic use only", Magic Leap license → blocklist; MIT/BSD/Apache-2.0 → allowed. Record the finding as a comment line per non-obvious entry.

- [ ] **Step 2: Update the dict to match findings; add/remove entries as measured**

- [ ] **Step 3: Run blocklist test**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py::test_blocklists_raise -q -p no:randomly`
Expected: PASS (update the test's model names if Step 2 changed entries).

- [ ] **Step 4: Commit**

```bash
git add collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git commit -m "chore(localization): verify vismatch license blocklist against upstream LICENSEs"
```

---

### Task 6: Localizer pairwise path (ref images + DinoSalad top-K gate)

**Files:**
- Modify: `collab_splats/localization/localizer.py`
- Test: `tests/localization/test_local_matcher.py`

Mechanism: when the extractor is a `LocalMatcher`, `from_feedforward` retains the model-res `ff.images` (aligned index-for-index with `world_points`) and DinoSalad global descriptors for the reconstruction frames. `localize()` ranks refs by cosine similarity, pairwise-matches the query image against the top-K ref images, and samples `world_points` directly with the matched ref pixels — no grid rescale needed because ref images and `world_points` share the model-res grid.

- [ ] **Step 1: Write failing test (mocked matcher + tiny synthetic scene)**

```python
def _make_pairwise_localizer(monkeypatch, n_frames=3, hw=(64, 64)):
    """CameraLocalizer with a mocked LocalMatcher and synthetic world_points."""
    from collab_splats.localization.localizer import CameraLocalizer

    lm = MagicMock(spec=LocalMatcher)
    lm.has_stable_indices = False
    q_px = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], np.float32)
    lm.match_images.return_value = MatchResult(query_px=q_px, ref_px=q_px.copy())
    lm.extract.return_value = LocalFeatures(
        keypoints=torch.zeros((4, 2)), descriptors=torch.zeros((4, 8))
    )
    # world_points: planar grid so sampled 3D points are valid and non-degenerate
    yy, xx = np.mgrid[0 : hw[0], 0 : hw[1]].astype(np.float32)
    wp = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    loc = CameraLocalizer.__new__(CameraLocalizer)
    loc._extractor = lm
    loc._frame_features = [lm.extract.return_value] * n_frames
    loc._frame_sources = ["reconstruction"] * n_frames
    loc._world_points = [wp] * n_frames
    loc._image_hw = hw
    loc._image_paths = [f"frame_{i:06d}.jpg" for i in range(n_frames)]
    loc._localized_extrinsics = []
    loc.config = {}
    loc._ref_images = [np.zeros((*hw, 3), np.uint8)] * n_frames
    loc._ref_global_desc = np.eye(n_frames, 8, dtype=np.float32)  # frame i ~ basis vec i
    loc._retrieval = MagicMock()
    loc._retrieval.forward.return_value = torch.from_numpy(np.eye(1, 8, dtype=np.float32))  # ~frame 0
    loc._top_k = 2
    return loc, lm


def test_pairwise_localize_matches_topk_only(monkeypatch):
    loc, lm = _make_pairwise_localizer(monkeypatch, n_frames=3)
    query = np.zeros((64, 64, 3), np.uint8)
    result = loc.localize(query)
    # top_k=2 -> exactly 2 pairwise calls, not 3
    assert lm.match_images.call_count == 2
    assert result.n_correspondences == 8  # 4 matches x 2 frames
    assert result.ref_hw == (64, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py::test_pairwise_localize_matches_topk_only -q -p no:randomly`
Expected: FAIL — `AttributeError` (`_ref_images` unused / `NotImplementedError` from features-match path).

- [ ] **Step 3: Implement — `from_feedforward` retention**

In `localizer.py` `from_feedforward` (the classmethod building the index, around line 690-770): after the extractor/feature-cache logic, add retention when pairwise. Insert after `self._extractor` and world_points assignment (adapt to the exact local variable names in the method — `ff` is the FeedforwardResult):

```python
        # Pairwise matchers (LocalMatcher) match query IMAGE vs ref IMAGE — retain the
        # model-res ff.images (index-aligned with world_points; same grid, so matched
        # ref pixels sample world_points with no rescale) plus a DinoSalad retrieval
        # gate so localize() matches top-K refs, not all N.
        localizer._ref_images = None
        localizer._ref_global_desc = None
        localizer._retrieval = None
        localizer._top_k = top_k
        if isinstance(extractor, LocalMatcher):
            from collab_splats.localization.retrieval import BaseRetrievalExtractor

            imgs = np.asarray(ff.images)
            if imgs.max() <= 1.5:  # MapAnything stores [0,1]; VGGT stores [0,255]
                imgs = (imgs * 255.0).round()
            localizer._ref_images = [im.astype(np.uint8) for im in imgs]
            localizer._retrieval = BaseRetrievalExtractor.get("dino-salad")()
            with torch.inference_mode():
                desc = localizer._retrieval.forward(localizer._ref_images)
            localizer._ref_global_desc = torch.nn.functional.normalize(desc, dim=-1).cpu().numpy()
```

Add `top_k: int = 8` to the `from_feedforward` signature (documented: only used by pairwise matchers). Also add the same four attribute defaults (`_ref_images = None` etc.) to `__init__` so descriptor-path instances are unaffected, and raise in `load_index` when the loaded extractor is a `LocalMatcher` and `ff.images` is unavailable:

```python
        if isinstance(self._extractor, LocalMatcher) and self._ref_images is None:
            raise RuntimeError(
                "Pairwise matcher needs reference images: build via from_feedforward "
                "(feedforward.zarr images array), not load_index alone."
            )
```

- [ ] **Step 4: Implement — `localize()` branch**

In `localize()` (line 794), replace the single match loop with a branch. Keep the existing loop verbatim as the descriptor path; add before it:

```python
        if isinstance(self._extractor, LocalMatcher):
            return self._localize_pairwise(query_image, query_feats, query_intrinsics)
```

and add the method (mirrors the existing loop's accumulation/PnP code paths — the PnP block from line 862 onward is shared; factor it into `self._solve_pnp(pts2d, pts3d_matched, pts2d_ref, ref_frame_indices, query_image, query_feats, query_intrinsics, ref_hw)` used by both paths rather than duplicating):

```python
    def _localize_pairwise(self, query_image, query_feats, query_intrinsics):
        """Pairwise path: match query image vs top-K retrieved model-res ref images."""
        # Rank reconstruction frames by DinoSalad cosine similarity
        with torch.inference_mode():
            q_desc = self._retrieval.forward([query_image])
        q_desc = torch.nn.functional.normalize(q_desc, dim=-1).cpu().numpy()[0]
        recon_idx = [i for i, s in enumerate(self._frame_sources) if s == "reconstruction"]
        sims = self._ref_global_desc[recon_idx] @ q_desc
        top = [recon_idx[j] for j in np.argsort(-sims)[: self._top_k]]

        model_hw = self._ref_images[0].shape[:2]
        all_q, all_3d, all_ref, all_frame = [], [], [], []
        for i in top:
            m = self._extractor.match_images(query_image, self._ref_images[i])
            if len(m) == 0:
                continue
            # ref px already in the world_points grid (both model-res) — no rescale
            pts3d, valid = sample_world_points(self._world_points[i], m.ref_px)
            if not valid.any():
                continue
            all_q.append(m.query_px[valid])
            all_3d.append(pts3d[valid])
            all_ref.append(m.ref_px[valid])
            all_frame.append(np.full(int(valid.sum()), i, dtype=np.int32))
        return self._solve_pnp(all_q, all_3d, all_ref, all_frame, query_image, query_feats,
                               query_intrinsics, ref_hw=model_hw)
```

`_solve_pnp` is the extracted lines 841-945 of the current `localize()` (n_corr guard, concatenation, pycolmap camera/options/solve, both failure and success `LocalizationResult` construction) with `ref_hw` a parameter instead of `tuple(self._image_hw)`. The descriptor path calls it with `ref_hw=tuple(self._image_hw)`.

- [ ] **Step 5: Run new test + full localization suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/ -q -p no:randomly`
Expected: new test PASS; all pre-existing tests PASS (descriptor path byte-identical through `_solve_pnp` refactor).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/localization/localizer.py tests/localization/test_local_matcher.py
git commit -m "feat(localization): pairwise localize path — ref-image retention + DinoSalad top-K gate"
```

---

### Task 7: Verification image-pair branch + pipeline hard error

**Files:**
- Modify: `collab_splats/geometry/verification.py` (match export, line ~172)
- Modify: `collab_splats/wrapper/reconstructor.py` (verify stage entry)
- Test: `tests/geometry/test_verification.py`, `tests/wrapper/test_verify_stage.py`

- [ ] **Step 1: Write failing tests**

In `tests/geometry/test_verification.py` add (reuse the file's existing fixture helpers for recon/features):

```python
def test_verify_pairwise_matcher_uses_images_and_recovered_indices():
    """A LocalMatcher-style matcher exports matches via match_images + idx recovery."""
    from collab_splats.localization.extractors import LocalMatcher, MatchResult

    matcher = MagicMock(spec=LocalMatcher)
    matcher.has_stable_indices = True
    matcher.match_images.return_value = MatchResult(
        query_px=np.array([[1.0, 2.0]], np.float32),
        ref_px=np.array([[3.0, 4.0]], np.float32),
        idx_q=np.array([0], np.int64),
        idx_db=np.array([0], np.int64),
    )
    # ... build the minimal recon/features/images the file's other tests use,
    # call verify_reconstruction(..., images=images), assert matcher.match_images called
    # per sequential pair and the DB match table is non-empty.


def test_verify_refuses_index_incapable_pairwise_matcher():
    from collab_splats.localization.extractors import LocalMatcher

    matcher = MagicMock(spec=LocalMatcher)
    matcher.has_stable_indices = False
    with pytest.raises(ValueError, match="stable keypoint indices"):
        # same minimal fixtures; verify_reconstruction must refuse up front
        ...
```

Fill the `...` from the existing fixture pattern in that file (it already builds a 2-frame recon + LocalFeatures for the XFeatStar-skips test — copy that arrangement).

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -q -p no:randomly -k pairwise`
Expected: FAIL — `verify_reconstruction` has no `images` param / no refusal.

- [ ] **Step 3: Implement in `verification.py`**

`verify_reconstruction(...)` gains `images: "list[np.ndarray] | None" = None` (model-res RGB, aligned with `sorted(recon.images)` — same alignment contract as `features`). At the top, refuse early:

```python
    # Pairwise matchers must prove index stability up front — a silent skip here would
    # surface later as a missing verification.json with no explanation.
    if isinstance(matcher, LocalMatcher):
        if not matcher.has_stable_indices:
            raise ValueError(
                f"matcher '{matcher.model_name}' cannot provide stable keypoint indices "
                "(failed the index-stability probe) — geometric verification requires them. "
                "Choose a sparse index-stable model or disable pointcloud.geometric_verification."
            )
        if images is None:
            raise ValueError("pairwise matcher requires `images` aligned with recon frames")
```

At the match-export site (line ~172), branch:

```python
        if isinstance(matcher, LocalMatcher):
            m = matcher.match_images(images[id_to_pos[id1]], images[id_to_pos[id2]])
        else:
            m = matcher.match(features[id_to_pos[id1]], features[id_to_pos[id2]], hw)
```

Import `LocalMatcher` alongside the existing `BaseLocalExtractor, LocalFeatures` import (line 22).

- [ ] **Step 4: Wire the caller**

In `reconstructor.py`'s verify stage (the block near line 969 that calls `load_reconstruction_features` and `verify_reconstruction`): resolve the matcher via `resolve_matcher(extractor_name)`, and when it is a `LocalMatcher`, load model-res images from feedforward.zarr (`FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)`, apply the same `>1.5`-scale uint8 conversion as Task 6) and pass `images=` through. The refusal inside `verify_reconstruction` is the hard error — reconstructor adds no duplicate check.

- [ ] **Step 5: Run verification + wrapper suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py tests/wrapper/test_verify_stage.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/verification.py collab_splats/wrapper/reconstructor.py tests/geometry/test_verification.py
git commit -m "feat(geometry): verification accepts pairwise matchers with index recovery; hard-refuses index-incapable models"
```

---

### Task 8: Config + docs + weights pre-fetch

**Files:**
- Modify: `configs/base.yaml` (localization block, lines ~102-104)
- Modify: `configs/README.md`
- Modify: `setup.sh`

- [ ] **Step 1: base.yaml**

Update the localization block (keep `extractor` as the key during transition — `resolve_matcher` accepts both name spaces):

```yaml
localization:
  enabled: true
  # Legacy registry keys (disk|xfeat|xfeat-star|loma|loma-g) or any vismatch model
  # name (e.g. disk-lightglue, aliked-lightglue, xfeat-steerers). vismatch names are
  # matched pairwise against the retrieval top-K; some are blocked (license/deps) —
  # see _VISMATCH_LICENSE_BLOCKLIST / _VISMATCH_DEP_BLOCKLIST in localization/extractors.py.
  extractor: loma
  top_k: 8  # pairwise (vismatch) path only: refs matched per query
```

Thread `top_k` through the reconstructor's `CameraLocalizer.from_feedforward` call sites (`_build_localization_db` passes it; default 8 preserved when absent).

- [ ] **Step 2: configs/README.md**

In the localization section, document: vismatch names accepted, pairwise semantics, the geometric-verification constraint (index-stable sparse models only; hard error otherwise), and the blocklists.

- [ ] **Step 3: setup.sh weights pre-fetch**

After the existing model-download steps, add:

```bash
# Pre-fetch vismatch default-model weights so remote/tmux runs never download mid-run.
/opt/venv/reconstruction/bin/python - <<'EOF'
import vismatch
for name in ("disk-lightglue",):  # extend when configs reference more models
    vismatch.get_matcher(name, device="cpu")
    print(f"vismatch weights cached: {name}")
EOF
```

- [ ] **Step 4: Smoke the config path**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ tests/localization/ -q -p no:randomly`
Expected: PASS (known-failures excepted).

- [ ] **Step 5: Commit**

```bash
git add configs/base.yaml configs/README.md setup.sh collab_splats/wrapper/reconstructor.py
git commit -m "feat(configs): vismatch model names in localization.extractor + top_k; weights pre-fetch"
```

---

### Task 9: Parity gate (compute — human-gated, tmux only)

**Files:**
- Create: `evals/scripts/eval_localization_parity.py`

**GATE: do not run the benchmark from this plan's executor — it is GPU compute. Write the script, verify `--help`, then hand off to the user per repo convention (tmux).**

- [ ] **Step 1: Write the script**

`evals/scripts/eval_localization_parity.py` — CLI comparing legacy vs vismatch matchers on one scene:

- Inputs: `--scene` (dir with `feedforward.zarr` + `frames.zarr`), `--matchers` (comma list, default `disk,disk-lightglue,xfeat,xfeat,loma,loma`), `--n_queries` (default 10, held-out frames), `--top_k 8`, `--output` (json under `evals/results/`).
- For each matcher name via `resolve_matcher`: build the index (`from_feedforward`), localize each query, record per-query `n_correspondences`, `n_inliers`, inlier ratio, wall-time; plus pose-vs-reconstruction-pose rotation/translation deltas for frames whose GT pose is in the recon (leave-one-out).
- Report median/p90 per matcher (median-only hides tails — repo lesson).
- Baseline anchors for sanity, from the 2026-07-22 measurements: disk 3314/2099, xfeat 2603/1211 (correspondences/inliers on the reference scene).

- [ ] **Step 2: Verify CLI parses**

Run: `/opt/venv/reconstruction/bin/python evals/scripts/eval_localization_parity.py --help`
Expected: usage text, exit 0.

- [ ] **Step 3: Commit script; STOP and hand off**

```bash
git add evals/scripts/eval_localization_parity.py
git commit -m "feat(evals): localization parity benchmark — legacy vs vismatch matchers"
```

Report to the user: parity run + verify-stage e2e with one index-stable model are theirs to run (tmux). **Task 10 is blocked until they confirm parity.**

---

### Task 10: Retirement (BLOCKED on Task 9 parity confirmation)

**Files:**
- Modify: `collab_splats/localization/extractors.py`, `localizer.py`, `verification.py`, `configs/base.yaml`, `docs/superpowers/specs/2026-07-08-loma-matcher-integration-design.md` (mark superseded)
- Delete: `third_party/xfeat` path hook, legacy tests

Only after the user confirms parity:

- [ ] Delete `DiskExtractor`, `XFeatExtractor`, `XFeatStarExtractor`, `LomaExtractor`, `LomaGExtractor`, the `BaseLocalExtractor` ABC + `RegistryMixin` usage, `resolve_matcher` (constructor call `LocalMatcher(name)` inlined at the two call sites), and the `third_party/xfeat` sys.path block + kornia/loma imports at the top of `extractors.py`.
- [ ] Rename config key `localization.extractor` → `localization.matcher` everywhere (`base.yaml`, reconstructor, dashboard `run_localization`, notebooks nb07, `configs/README.md`).
- [ ] Type hints `BaseLocalExtractor` → `LocalMatcher` in `localizer.py`, `verification.py`, `reconstructor.py`, eval scripts.
- [ ] Delete legacy-extractor tests; keep contract tests. Run full suite: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:randomly`.
- [ ] Mark the loma-matcher spec superseded (header note pointing at this spec) — only if loma parity held; otherwise keep `LomaExtractor` and record the exception in the spec.
- [ ] Remove `lomatch`, `kornia` DISK/LightGlue extras from pyproject **only if** nothing else imports them (grep first: `grep -rn "from loma\|from kornia.feature" collab_splats/`).
- [ ] Commit: `refactor(localization): retire legacy extractors — vismatch is the sole Stage-2 provider`

---

## Self-review notes (completed)

- Spec coverage: install route (T1), LocalMatcher+probe+blocklists+frame guard (T2-T3, T5), resolver/config seam (T4, T8), pairwise localizer + retrieval gate (T6 — spec said "existing retrieval top-K"; the localizer had no retrieval wiring, so T6 adds the gate explicitly), verify branch + hard error (T7), weights pre-fetch (T8), parity gate (T9), retirement + loma-spec supersession (T10). Upstream extras PR: out of plan scope, tracked in spec as non-blocking.
- The `match()`-raises design keeps `LocalMatcher` a legal `BaseLocalExtractor` member during transition without dead descriptor code.
- `_solve_pnp` extraction (T6) is the only refactor of existing code; descriptor path must stay behavior-identical — the existing localization tests are the guard.
