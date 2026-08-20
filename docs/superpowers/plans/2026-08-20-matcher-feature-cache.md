# Matcher Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the 12.4× per-image extraction redundancy in `verify()`'s pair-matching loop: descriptor-NN models match precomputed zarr features directly (extraction inside verify → 0), loma-class learned matchers get an identity-keyed per-image encode cache with a byte-identical pair replay (~1853 s → ~250 s), plus phase-timing instrumentation to attribute the ~658 s residual.

**Architecture:** Everything lands in `collab_splats/localization/extractors.py` (LocalMatcher: `match()` over kornia mutual-NN gated by a static model allowlist, plain-dict encode cache, statically-activated loma pair adapter) plus a capability dispatch and phase timings in `collab_splats/geometry/verification.py`. No runtime probes for the new paths — equivalence is enforced by GPU parity tests in the suite (env is pinned); the pre-existing `_probe_index_stability` is untouched. Both consumer seams already exist: `LocalMatcher.match()` is a reserved NotImplementedError seam, and `verify_reconstruction`'s else-branch already calls `matcher.match(features…)`. Fallback for everything else is the current pairwise path, untouched.

**Tech Stack:** vismatch model zoo (ImportSandbox reach-in for loma), `kornia.feature.match_mnn` (kornia 0.8.2 in env), pycolmap DB export. Spec: `docs/superpowers/specs/2026-08-20-matcher-feature-cache-design.md`.

**Repo rules that bind every task:** stage named files only (never `git add -A`/`.`); commit with `git commit --only <files>`; `docs/superpowers/` needs `git add -f`; python is `/opt/venv/reconstruction/bin/python`; never repo-wide `black .`; lint gate is `ruff check <touched files>` only; GPU runs serial in tmux (single A40).

---

### Task 1: Implement the reserved `match()` seam — kornia mutual-NN behind a static allowlist

The general path: mutual-NN over precomputed descriptors via `kornia.feature.match_mnn`
(L2 on unit vectors orders identically to cosine, and match_mnn admits every mutual pair —
the min_cossim=-1 semantics). Gated by `supports_descriptor_matching`, a property over a
static allowlist `_DESCRIPTOR_NN_MODELS = {"xfeat"}`; membership is licensed by the Task 4
GPU parity test, not a runtime probe.

**Files:**
- Modify: `collab_splats/localization/extractors.py` (`match()` at ~line 153; `__init__` at ~line 104; imports)
- Test: `tests/localization/test_local_matcher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/localization/test_local_matcher.py`:

```python
def _one_hot_features(rows, d=8):
    """LocalFeatures whose descriptors are one-hot rows — mutual-NN is exactly identity."""
    kpts = np.stack([np.arange(len(rows)), np.arange(len(rows))], axis=1).astype(np.float32) * 10
    desc = np.eye(d, dtype=np.float32)[rows]
    return LocalFeatures(keypoints=torch.from_numpy(kpts), descriptors=torch.from_numpy(desc))


@patch("vismatch.get_matcher")
def test_match_mutual_nn_for_allowlisted_model(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("xfeat", device="cpu", probe=False)  # "xfeat" is in _DESCRIPTOR_NN_MODELS
    assert lm.supports_descriptor_matching is True
    q = _one_hot_features([0, 1, 2, 3])
    db = _one_hot_features([3, 2, 1, 0])  # same one-hot basis, permuted rows
    m = lm.match(q, db, image_hw=(100, 100))
    assert isinstance(m, MatchResult)
    assert len(m) == 4
    # mutual NN of a permuted one-hot basis is that permutation, with native table indices
    order = np.argsort(m.idx_q)
    np.testing.assert_array_equal(m.idx_q[order], [0, 1, 2, 3])
    np.testing.assert_array_equal(m.idx_db[order], [3, 2, 1, 0])
    # pixel coords are the table rows the indices point at
    np.testing.assert_array_equal(m.query_px, q.keypoints.numpy()[m.idx_q])
    np.testing.assert_array_equal(m.ref_px, db.keypoints.numpy()[m.idx_db])


@patch("vismatch.get_matcher")
def test_match_empty_descriptors_returns_empty(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("xfeat", device="cpu", probe=False)
    empty = LocalFeatures(keypoints=torch.zeros((0, 2)), descriptors=torch.zeros((0, 8)))
    m = lm.match(empty, _one_hot_features([0, 1]), image_hw=(100, 100))
    assert len(m) == 0 and m.idx_q is not None  # empty but indexable


@patch("vismatch.get_matcher")
def test_match_still_raises_for_non_allowlisted_model(mock_get):
    # Not in _DESCRIPTOR_NN_MODELS — the NotImplementedError contract survives.
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("roma", device="cpu", probe=False)
    q = _one_hot_features([0, 1])
    with pytest.raises(NotImplementedError, match="match_images"):
        lm.match(q, q, image_hw=(100, 100))
```

Note: `test_descriptor_level_match_unsupported` (existing, ~line 103) uses
"disk-lightglue" — not in the allowlist, keeps passing as-is.

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly`
Expected: 3 new FAIL (`AttributeError: supports_descriptor_matching` / NotImplementedError), existing PASS.

- [ ] **Step 3: Implement**

In `extractors.py`, add to the top-of-file imports (imports-at-top rule):

```python
from kornia.feature import match_mnn
```

Add a module-level constant under the existing `# VisMatch-backed matcher` divider (near the
blocklists):

```python
# Models whose own match stage IS descriptor mutual-NN — match() may serve them from
# precomputed features. Membership is licensed by a GPU parity test in the suite
# (test_real_xfeat_general_path_matches_pairwise); extending this set requires a new
# passing parity test. Never silently substitute NN for a learned matcher.
_DESCRIPTOR_NN_MODELS = {"xfeat"}
```

Add the property (near `has_stable_indices` usage, after `__init__`):

```python
    @property
    def supports_descriptor_matching(self) -> bool:
        """Whether match() can serve this model — static allowlist, parity-tested in the suite."""
        return self._model_name in _DESCRIPTOR_NN_MODELS
```

Replace the `match()` body (the NotImplementedError message must keep the substring
`match_images` for the existing test):

```python
    def match(self, query: LocalFeatures, db: LocalFeatures, image_hw: tuple[int, int]) -> MatchResult:
        """Descriptor-level mutual-NN match over precomputed features (allowlisted models only).

        Match rows ARE keypoint-table indices by construction — no _recover_indices.
        """
        if not self.supports_descriptor_matching:
            raise NotImplementedError(
                f"LocalMatcher('{self._model_name}') has no descriptor-level matching — "
                "its match stage is not descriptor-NN. Use match_images()."
            )
        if len(query.descriptors) == 0 or len(db.descriptors) == 0:
            return _empty_match()
        # L2 mutual-NN on unit vectors == cosine mutual-NN; match_mnn admits every mutual pair
        d0 = torch.nn.functional.normalize(query.descriptors.to(self._device), dim=1)
        d1 = torch.nn.functional.normalize(db.descriptors.to(self._device), dim=1)
        _, idxs = match_mnn(d0, d1)
        if len(idxs) == 0:
            return _empty_match()
        idx_q = idxs[:, 0].cpu().numpy().astype(np.int64)
        idx_db = idxs[:, 1].cpu().numpy().astype(np.int64)
        return MatchResult(
            query_px=query.keypoints.numpy()[idx_q],
            ref_px=db.keypoints.numpy()[idx_db],
            idx_q=idx_q,
            idx_db=idx_db,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Lint + commit**

```bash
ruff check collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git add collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git commit --only collab_splats/localization/extractors.py --only tests/localization/test_local_matcher.py -m "feat(localization): descriptor mutual-NN match() via kornia behind static allowlist"
```

---

### Task 2: verification.py dispatch — descriptor path for allowlisted LocalMatchers

`verify_reconstruction` currently forces every `LocalMatcher` down the pairwise path
(`verification.py:184`/`:221`). Route allowlisted models to the existing descriptor
else-branch instead: no images touched, no extraction, match rows are table indices.

**Files:**
- Modify: `collab_splats/geometry/verification.py:182-237`
- Test: `tests/geometry/test_verification.py`

- [ ] **Step 1: Update the fixtures + write the failing test**

In `tests/geometry/test_verification.py`, EVERY `MagicMock(spec=LocalMatcher)` construction
(three sites: `_pairwise_matcher` ~line 95, and the inline mocks ~lines 228 and 244) gets one
line added right after its `has_stable_indices` assignment:

```python
    matcher.supports_descriptor_matching = False
```

(MagicMock(spec=…) returns a truthy child mock for unset attributes — without this every old
pairwise test would silently take the new descriptor path.)

Append the new test (uses the existing `_synthetic_scene`/`_make_recon`/
`_features_from_keypoints` helpers):

```python
def test_descriptor_capable_localmatcher_skips_images(tmp_path):
    """An allowlisted LocalMatcher takes the descriptor branch: match() on features,
    match_images and `images` untouched, no stable-indices requirement."""
    _, extrinsics, kps = _synthetic_scene()
    matcher = MagicMock(spec=LocalMatcher)
    matcher.model_name = "stub-descriptor"
    matcher.supports_descriptor_matching = True
    matcher.has_stable_indices = False  # irrelevant on the descriptor path

    def _match(query, db, image_hw):
        n = min(len(query.keypoints), len(db.keypoints))
        idx = np.arange(n, dtype=np.int64)
        return MatchResult(
            query_px=query.keypoints.numpy()[:n], ref_px=db.keypoints.numpy()[:n],
            idx_q=idx, idx_db=idx.copy(),
        )

    matcher.match.side_effect = _match
    result = verify_reconstruction(
        recon=_make_recon(extrinsics),
        features=_features_from_keypoints(kps),
        matcher=matcher,
        output_dir=tmp_path,
        # no `images` passed — the descriptor path must not require them
    )
    matcher.match_images.assert_not_called()
    assert matcher.match.call_count == 3  # 3 sequential pairs for 3 frames
    assert result.summary["n_points"] > 0
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -v -p no:randomly`
Expected: new test FAIL (`ValueError: pairwise matcher requires \`images\`` — the isinstance
branch fires), existing PASS.

- [ ] **Step 3: Implement the dispatch**

In `verification.py`, replace the guard block (starting `if isinstance(matcher, LocalMatcher):`
~line 184) with:

```python
    # Allowlisted LocalMatchers match precomputed descriptors directly (no images, no
    # extraction — the features came from the zarr cache the pipeline already built).
    # Everything else pairwise: those must prove index stability up front — a silent skip
    # here would surface later as a missing verification.json with no explanation.
    pairwise = isinstance(matcher, LocalMatcher) and not matcher.supports_descriptor_matching
    if pairwise:
        if not matcher.has_stable_indices:
            raise ValueError(
                f"matcher '{matcher.model_name}' cannot provide stable keypoint indices "
                "(failed or never ran the index-stability probe) — geometric verification requires them. "
                "Choose a sparse index-stable model or disable pointcloud.geometric_verification."
            )
        if images is None or len(images) != len(image_ids):
            raise ValueError(
                f"pairwise matcher requires `images` aligned with recon frames "
                f"(got {'none' if images is None else len(images)} for {len(image_ids)} frames)"
            )
```

In the pair loop (~line 221), change the branch condition from
`if isinstance(matcher, LocalMatcher):` to `if pairwise:` (both branch bodies stay as they are).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py tests/localization/test_local_matcher.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Lint + commit**

```bash
ruff check collab_splats/geometry/verification.py tests/geometry/test_verification.py
git add collab_splats/geometry/verification.py tests/geometry/test_verification.py
git commit --only collab_splats/geometry/verification.py --only tests/geometry/test_verification.py -m "feat(geometry): descriptor-path dispatch for allowlisted matchers in verification"
```

---

### Task 3: Per-image encode cache + loma pair adapter (byte-identical replay)

For learned-matcher models the win is caching the per-image encode inside `match_images`.
Plain identity-keyed dict (cleared wholesale at capacity) + one statically-registered adapter
(loma) that replays the wrapper's own `_forward` in two halves. No probe, no rename: the
adapter branch is an early return at the top of `match_images`; the parity tests in Task 4
compare against the plain path by setting `lm._pair_adapter = None`.

**Files:**
- Modify: `collab_splats/localization/extractors.py`
- Test: `tests/localization/test_local_matcher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/localization/test_local_matcher.py`:

```python
@patch("vismatch.get_matcher")
def test_encode_cache_identity_keyed_and_cleared_at_cap(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    calls = []
    encode = lambda img: calls.append(img) or f"payload_{len(calls)}"
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.zeros((4, 4, 3), np.uint8)  # equal content, different object
    assert lm._cached_encode(a, encode) == "payload_1"
    assert lm._cached_encode(a, encode) == "payload_1"  # hit: same object, no re-encode
    assert lm._cached_encode(b, encode) == "payload_2"  # miss: identity, not content
    assert len(calls) == 2
    # fill to capacity: next insert clears wholesale, then re-encodes
    for i in range(510):
        lm._cached_encode(np.full((1, 1, 3), i % 255, np.uint8), encode)
    c = np.ones((4, 4, 3), np.uint8)
    lm._cached_encode(c, encode)  # 513th distinct image -> clear happened before insert
    assert len(lm._encode_cache) < 512
    assert lm._cached_encode(c, encode) == lm._cached_encode(c, encode)  # still a hit after clear


@patch("vismatch.get_matcher")
def test_pair_adapter_off_for_unknown_wrappers(mock_get):
    # _fake_vismatch_matcher is a MagicMock — its class name is not in _PAIR_ADAPTERS,
    # so match_images must run the plain path untouched.
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=True)
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    assert lm._pair_adapter is None
    lm.has_stable_indices = True
    q = np.zeros((100, 100, 3), dtype=np.uint8)
    m = lm.match_images(q, q)
    assert len(m) == 4  # plain-path behavior byte-identical to before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly`
Expected: 2 FAIL (`AttributeError: _cached_encode` / `_pair_adapter`).

- [ ] **Step 3: Implement cache, adapter, dispatch**

In `extractors.py`:

Top of file — extend the stdlib imports:

```python
import sys
```

Module-level, next to `_DESCRIPTOR_NN_MODELS`:

```python
# Wrapper-class-name -> (encode method, match method) on LocalMatcher. An adapter replays
# its wrapper's _forward in two halves (per-image encode, per-pair match) so the encode
# half can be cached across the pair loop. Byte-parity with the plain forward is enforced
# by a GPU parity test in the suite (test_real_loma_adapter_matches_plain_path); extending
# this dict requires a new passing parity test.
_PAIR_ADAPTERS: dict[str, tuple[str, str]] = {"LoMaMatcher": ("_loma_encode", "_loma_match")}

# Encode-cache capacity: covers a 300-frame verify() loop with headroom; cleared wholesale
# at the cap (identity keys are worthless once the caller drops its arrays anyway).
_ENCODE_CACHE_CAP = 512
```

In `__init__` (right after `self.has_stable_indices: bool | None = None`):

```python
        # Statically-activated pair adapter (wrapper-class keyed); None = plain pairwise path.
        self._pair_adapter = _PAIR_ADAPTERS.get(type(self._matcher).__name__)
        self._encode_cache: dict[int, tuple[np.ndarray, object]] = {}
```

At the TOP of the existing `match_images` body (before the current code, which stays
byte-identical as the fallback):

```python
        # Adapter path: cached per-image encode + per-pair match replay of the wrapper's
        # own forward (byte-parity enforced by the suite's GPU parity test).
        if self._pair_adapter is not None:
            encode, pair_match = (getattr(self, n) for n in self._pair_adapter)
            p0 = self._cached_encode(query_image, encode)
            p1 = self._cached_encode(ref_image, encode)
            m = pair_match(p0, p1)
            if len(m):
                self._check_pixel_frame(m.query_px, query_image.shape[:2], self._model_name)
                self._check_pixel_frame(m.ref_px, ref_image.shape[:2], self._model_name)
            return m
```

Add the cache helper and the loma adapter methods (new `######## Learned-matcher pair
adapters` section at the end of the class). The match half mirrors
`vismatch/im_models/loma.py:68-102` (vismatch in `/opt/venv/reconstruction`,
`LoMaMatcher._forward`) line-for-line; the sandbox must be entered explicitly because
vismatch only wraps `__init__`/`_forward` (`vismatch/base_matcher.py:19-30`):

```python
    def _cached_encode(self, image: np.ndarray, encode):
        """Identity-keyed per-image encode cache; cleared wholesale at capacity.

        The `is` check makes id() reuse after GC a miss, never a wrong hit. Both consumers
        (verification's frame list, localizer's query/refs) hold their arrays alive for
        the duration of the loop, so identity keying is exact and free — no hashing.
        """
        entry = self._encode_cache.get(id(image))
        if entry is not None and entry[0] is image:
            return entry[1]
        if len(self._encode_cache) >= _ENCODE_CACHE_CAP:
            self._encode_cache.clear()
        payload = encode(image)
        self._encode_cache[id(image)] = (image, payload)
        return payload

    def _loma_encode(self, image: np.ndarray) -> dict:
        """Per-image half of LoMaMatcher._forward: preprocess + detect_and_describe.

        Payload keeps PRE-pixel-coords tensors (normalized kpts, desc on device) plus the
        shapes the match half needs to replay the wrapper's coordinate chain exactly —
        pixel-frame zarr features cannot serve here (inverting the affine chain is
        float-inexact and would break byte-parity).
        """
        from vismatch.import_sandbox import ImportSandbox

        m = self._matcher
        with ImportSandbox.get(type(m).__module__), torch.inference_mode():
            img, orig_shape = m.preprocess(self._to_tensor(image))
            kpts, desc, _, _ = m.matcher.detect_and_describe(img, m.max_num_keypoints)
        return {"kpts": kpts, "desc": desc, "orig_shape": orig_shape, "hw": tuple(img.shape[-2:])}

    def _loma_match(self, p0: dict, p1: dict) -> MatchResult:
        """Per-pair half of LoMaMatcher._forward replayed on two cached encodes.

        Learned matcher -> filter_matches -> to_pixel_coords -> rescale_coords -> the
        -0.5 COLMAP offset, exactly as the wrapper. Native indices: match row k IS
        keypoint-table row idx_q[k] — no _recover_indices.
        """
        from vismatch.import_sandbox import ImportSandbox

        m = self._matcher
        mod = sys.modules[type(m).__module__]  # wrapper module: filter_matches, to_pixel_coords
        (H0, W0), (H1, W1) = p0["hw"], p1["hw"]
        with ImportSandbox.get(type(m).__module__), torch.inference_mode():
            scores = m.matcher(p0["kpts"], p1["kpts"], p0["desc"], p1["desc"])["scores"]
            m0, _, _, _ = mod.filter_matches(scores, m.matcher.cfg.filter_threshold)
            valid = m0[0] > -1
            if not bool(valid.any()):
                return _empty_match()
            idx_q = torch.where(valid)[0]
            idx_db = m0[0][valid]
            kq = mod.to_pixel_coords(p0["kpts"][0][idx_q], H0, W0)
            kr = mod.to_pixel_coords(p1["kpts"][0][idx_db], H1, W1)
            kq = m.rescale_coords(kq, *p0["orig_shape"], H0, W0) - 0.5
            kr = m.rescale_coords(kr, *p1["orig_shape"], H1, W1) - 0.5
        return MatchResult(
            query_px=_to_numpy(kq),
            ref_px=_to_numpy(kr),
            idx_q=idx_q.cpu().numpy().astype(np.int64),
            idx_db=idx_db.cpu().numpy().astype(np.int64),
        )
```

The `from vismatch.import_sandbox import ImportSandbox` imports stay inside the methods —
this is the documented optional-heavy-dep exception (vismatch is already lazy in `__init__`
today).

- [ ] **Step 4: Run the full localization + geometry suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/ tests/geometry/ -v -p no:randomly`
Expected: all PASS (plain path untouched; adapter never activates under mocks — MagicMock's
class name is not in `_PAIR_ADAPTERS`).

- [ ] **Step 5: Lint + commit**

```bash
ruff check collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git add collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git commit --only collab_splats/localization/extractors.py --only tests/localization/test_local_matcher.py -m "feat(localization): per-image encode cache + loma pair adapter (static activation)"
```

---

### Task 4: GPU parity tests (real models — the load-bearing gates that license the allowlists)

With no runtime probes, THESE tests are the equivalence mechanism: the loma adapter must be
byte-identical to the plain forward, and xfeat's `match()` must equal its own
`match_images`. Extending `_DESCRIPTOR_NN_MODELS` or `_PAIR_ADAPTERS` requires adding a new
passing test here.

**Files:**
- Test: `tests/localization/test_local_matcher.py` (append; CUDA-gated)

- [ ] **Step 1: Write the tests**

```python
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-model parity needs CUDA")


@requires_cuda
def test_real_loma_adapter_matches_plain_path():
    """Byte-parity gate for _PAIR_ADAPTERS['LoMaMatcher'] — adapter path == plain forward."""
    lm = LocalMatcher("loma")
    assert lm._pair_adapter is not None, "LoMaMatcher wrapper class no longer registered"
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 255, (240, 320, 3)).astype(np.uint8)
    b = np.roll(a, 12, axis=1)
    fast = lm.match_images(a, b)  # adapter + cache path
    adapter, lm._pair_adapter = lm._pair_adapter, None
    try:
        ref = lm.match_images(a, b)  # plain wrapper forward
    finally:
        lm._pair_adapter = adapter
    assert len(fast) == len(ref) and len(fast) > 0
    np.testing.assert_array_equal(fast.query_px, ref.query_px)
    np.testing.assert_array_equal(fast.ref_px, ref.ref_px)
    if ref.idx_q is not None:  # plain-path indices are recovered rows; adapter's are native
        np.testing.assert_array_equal(fast.idx_q, ref.idx_q)
        np.testing.assert_array_equal(fast.idx_db, ref.idx_db)


@requires_cuda
def test_real_xfeat_general_path_matches_pairwise():
    """Parity gate for _DESCRIPTOR_NN_MODELS entry 'xfeat' — match() == match_images()."""
    lm = LocalMatcher("xfeat")
    assert lm.supports_descriptor_matching is True
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 255, (240, 320, 3)).astype(np.uint8)
    b = np.roll(a, 12, axis=1)
    m = lm.match(lm.extract(a), lm.extract(b), image_hw=a.shape[:2])
    ref = lm.match_images(a, b)
    assert len(m) == len(ref) and len(m) > 0
    order_m, order_ref = np.argsort(m.idx_q), np.argsort(ref.idx_q)
    np.testing.assert_array_equal(m.idx_q[order_m], ref.idx_q[order_ref])
    np.testing.assert_array_equal(m.idx_db[order_m], ref.idx_db[order_ref])
```

- [ ] **Step 2: Run on the A40 (serial — no concurrent GPU jobs)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly -k "real_"`
Expected: 2 PASS. **If either fails, STOP — do not weaken the assertions.** For loma: diff
the two paths tensor-by-tensor (encode halves first — non-determinism in
`detect_and_describe` under autocast is the known suspect). For xfeat: read the vismatch
xfeat wrapper's match stage — if it thresholds cossim or is not plain mutual-NN, remove
"xfeat" from `_DESCRIPTOR_NN_MODELS` (Task 1's non-allowlisted tests then cover it) and
record why in the spec.

- [ ] **Step 3: Commit**

```bash
git add tests/localization/test_local_matcher.py
git commit --only tests/localization/test_local_matcher.py -m "test(localization): GPU parity gates licensing the loma adapter + xfeat allowlist"
```

---

### Task 5: Phase-timing instrumentation (the residual-attribution work)

The ~658 s residual has never been measured directly. Instrument `verify_reconstruction`'s
phases into `summary["phase_seconds"]` (lands in verification.json automatically) and log
the reconstructor-side cold-start phases.

**Files:**
- Modify: `collab_splats/geometry/verification.py`
- Modify: `collab_splats/wrapper/reconstructor.py` (`verify()`, ~lines 1107-1157)
- Test: `tests/geometry/test_verification.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_verification.py`:

```python
def test_summary_carries_phase_seconds(tmp_path):
    _, extrinsics, kps = _synthetic_scene()
    result = verify_reconstruction(
        recon=_make_recon(extrinsics),
        features=_features_from_keypoints(kps),
        matcher=_IdentityMatcher(),
        output_dir=tmp_path,
    )
    phases = result.summary["phase_seconds"]
    assert set(phases) == {"db_export", "pair_matching", "db_match_writes", "verify_matches", "triangulate", "report"}
    assert all(isinstance(v, float) and v >= 0.0 for v in phases.values())
    # and it round-trips through the JSON report
    on_disk = json.loads((tmp_path / "verification.json").read_text())
    assert "phase_seconds" in on_disk["summary"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -v -p no:randomly`
Expected: new test FAIL (`KeyError: 'phase_seconds'`).

- [ ] **Step 3: Implement**

`verification.py`: add `import time` to the stdlib imports. In `verify_reconstruction`:

```python
    timings: dict[str, float] = {}
```
right after `output_dir.mkdir(...)`. Then:

- Around `_write_frames(db, recon, features)` and the pair generation:
  `t = time.perf_counter()` before, `timings["db_export"] = time.perf_counter() - t` after
  `pairs = ...all_pairs()`.
- In the pair loop, accumulate two counters initialized to 0.0 before the loop:
  `t_match`, `t_write`. Wrap the matcher call (either branch) with perf_counter deltas into
  `t_match`, and `db.write_matches(...)` into `t_write`. After the loop:
  `timings["pair_matching"] = t_match`, `timings["db_match_writes"] = t_write`.
- Around `pycolmap.verify_matches(...)`: `timings["verify_matches"] = ...`.
- Around `_triangulate_and_summarize(...)`: `timings["triangulate"] = ...`.
- Around `_write_report(...)`: measure into `timings["report"]` — set
  `summary["phase_seconds"]` BEFORE calling `_write_report` with the report value included:

```python
    t = time.perf_counter()
    summary["phase_seconds"] = {k: round(v, 2) for k, v in timings.items()} | {"report": 0.0}
    result = VerificationResult(
        reconstruction=verified, pair_stats=pair_stats, frame_stats=frame_stats, summary=summary
    )
    _write_report(result, output_dir / "verification.json")
    summary["phase_seconds"]["report"] = round(time.perf_counter() - t, 2)
    logger.info("Verification phase seconds: %s", summary["phase_seconds"])
    return result
```

(The on-disk report's `report` entry reads 0.0 — it cannot time its own write; the returned
object and the log line carry the real value. Note this in a comment.)

`reconstructor.py` `verify()`: add `import time` to the stdlib import block if it is not
already there. Wrap the three cold-start phases with
`logger.info("verify(): %s took %.1f s", name, dt)` lines: `build_localization_db()`,
`load_reconstruction_features(...)`, `LocalMatcher(extractor_name)` construction (model load
+ index-stability probe — the cold start the residual analysis needs).

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Lint + commit**

```bash
ruff check collab_splats/geometry/verification.py collab_splats/wrapper/reconstructor.py tests/geometry/test_verification.py
git add collab_splats/geometry/verification.py collab_splats/wrapper/reconstructor.py tests/geometry/test_verification.py
git commit --only collab_splats/geometry/verification.py --only collab_splats/wrapper/reconstructor.py --only tests/geometry/test_verification.py -m "feat(geometry): per-phase timing in verification for residual attribution"
```

---

### Task 6: CLAUDE.md stale architecture line

**Files:**
- Modify: `CLAUDE.md` (architecture tree, `localization/` block)

- [ ] **Step 1: Fix the line**

Replace:
```
    extractors.py          # Stage 2: BaseLocalExtractor, Disk/XFeat/Loma/LomaG local matchers
```
with:
```
    extractors.py          # Stage 2: LocalMatcher over the vismatch model zoo (allowlist-gated fast paths)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit --only CLAUDE.md -m "docs: fix stale extractors.py line — legacy matcher classes are gone"
```

---

### Task 7: Evidence runs (tmux, serial, human-gated compute)

Measured, not assumed. Full verify() against the spec's baselines: per-pair 993.8 ms and
full verify() 2511.6 s on `/workspace/outputs/2026_07_15-Goprosplat-GH010229`
(vggt_omega + loma).

**Files:**
- Create: `/tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/verify_bench.py`
- Modify: `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md` (append results)

- [ ] **Step 1: Write the driver**

```python
"""Full-scene verify() re-run with matcher fast paths + phase timings vs the 2511.6 s baseline."""

import logging
import time
from pathlib import Path

import yaml

from collab_splats.wrapper.reconstructor import Reconstructor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

SCENE = Path("/workspace/outputs/2026_07_15-Goprosplat-GH010229")
cfg = yaml.safe_load((SCENE / "run_config.yaml").read_text())

t0 = time.perf_counter()
r = Reconstructor(cfg)
r.verify(overwrite=True)
print(f"[verify_bench] total {time.perf_counter() - t0:.1f} s (baseline 2511.6 s)")
```

- [ ] **Step 2: Run in tmux (nothing else on the GPU)**

```bash
tmux new-session -d -s verify_bench
tmux send-keys -t verify_bench "/opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/verify_bench.py 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/verify_bench.log" Enter
```

Wait for completion (poll `tmux has-session`), then read the log. Expected: total well under
1000 s; `phase_seconds` in the log attributes the former ~658 s residual.

- [ ] **Step 3: Attack the biggest attributed residual term — only if it is ours**

Per the spec: if `db_export`/`db_match_writes`/cold start dominates the residual, fix that
one term (vectorize the write loop / amortize the load) as a follow-up commit with a second
timed run. If pycolmap's C++ (`verify_matches`, `triangulate`) dominates, record the number
and stop — that floor belongs to the pycolmap-native migration.

- [ ] **Step 4: Append the measured numbers to the report + commit**

Append to `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`: total wall
time vs 2511.6 s, the full `phase_seconds` dict, cold-start log lines, and the per-pair
warm number implied by `pair_matching / n_pairs` vs 993.8 ms.

```bash
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit --only docs/superpowers/specs/2026-08-20-scene-error-report-measured.md -m "docs(specs): measured verify() timings with matcher fast paths"
```

---

### Final gate

- [ ] Full suite: `/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly` — no new failures vs `docs/known-test-failures.md`.
- [ ] `graphify update .` (repo rule: keep the graph current after code changes).
