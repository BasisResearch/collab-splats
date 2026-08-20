# Matcher Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the 12.4× per-image extraction redundancy in `verify()`'s pair-matching loop: descriptor-NN models match precomputed zarr features directly, and loma — the shipping default — gets its pair forward split into per-image and per-pair halves with the per-image half persisted in the existing zarr feature cache, so loma too matches from stored features (extraction inside verify → 0; matching ~1853 s → ~100 s on payload-bearing caches, ~250 s fallback), plus phase-timing instrumentation to attribute the ~658 s residual.

**Architecture:** `collab_splats/localization/extractors.py` (LocalMatcher: `match()` over kornia mutual-NN gated by static allowlist `_DESCRIPTOR_NN_MODELS`; loma split activated by one boolean `_split_loma_forward` — `_loma_detect_and_describe` / `_loma_match_features` / `_loma_features_cached`; `can_match_features()` as the single dispatch predicate; `LocalFeatures` gains ONE optional field `keypoints_normalized`). `collab_splats/localization/localizer.py` persists/restores that field in the existing zarr cache (one added CSR array — no new cache layer). `collab_splats/geometry/verification.py` dispatches on `can_match_features` and gains phase timings. No runtime probes for the new paths — byte-parity is enforced by GPU parity tests in the suite (env pinned); the pre-existing `_probe_index_stability` is untouched. Fallback for everything else is the current pairwise path, byte-identical.

**Tech Stack:** vismatch model zoo (explicit `ImportSandbox` reach-in for loma — vismatch only wraps `__init__`/`_forward`, `vismatch/base_matcher.py:19-30`), `kornia.feature.match_mnn` (kornia 0.8.2 in env), zarr v3 (`compressors=[BloscCodec]`). Spec: `docs/superpowers/specs/2026-08-20-matcher-feature-cache-design.md`.

**Repo rules that bind every task:** stage named files only (never `git add -A`/`.`); commit with `git commit --only <files>`; `docs/superpowers/` needs `git add -f`; python is `/opt/venv/reconstruction/bin/python`; never repo-wide `black .`; lint gate is `ruff check <touched files>` only; GPU runs serial in tmux (single A40).

**Key parity fact used throughout:** loma's coordinate chain (`to_pixel_coords` → `rescale_coords` → −0.5) is elementwise, so indexing the pre-chained full `keypoints` table by match indices is byte-identical to chaining the indexed subset (what `LoMaMatcher._forward` does, `vismatch/im_models/loma.py:68-102`). Hence NO shape metadata is stored — only `keypoints_normalized` (pre-transform coords the learned matcher consumes) plus the existing `keypoints`/`descriptors`.

---

### Task 1: Implement the reserved `match()` seam — kornia mutual-NN behind a static allowlist

The general path: mutual-NN over precomputed descriptors via `kornia.feature.match_mnn`
(L2 on unit vectors orders identically to cosine, and match_mnn admits every mutual pair —
the min_cossim=-1 semantics). Gated by `supports_descriptor_matching`, a property over a
static allowlist `_DESCRIPTOR_NN_MODELS = {"xfeat"}`; membership is licensed by the Task 5
GPU parity test, not a runtime probe. (Task 2 extends the guard to `can_match_features` and
adds the loma branch — this task lands the NN core.)

**Files:**
- Modify: `collab_splats/localization/extractors.py` (`match()` at ~line 153; imports)
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

Add the property (after `__init__`):

```python
    @property
    def supports_descriptor_matching(self) -> bool:
        """Whether the NN match() path can serve this model — static allowlist, parity-tested."""
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

### Task 2: Loma split — per-image / per-pair halves, `keypoints_normalized`, `can_match_features`

Loma's learned match stage cannot take the NN path. Split its pair forward
(`vismatch/im_models/loma.py:68-102`, vismatch in `/opt/venv/reconstruction`) into a
per-image half (`_loma_detect_and_describe`) and a per-pair half (`_loma_match_features`),
activated by one boolean keyed on the wrapper class name — no registry. `LocalFeatures`
gains ONE optional field. `can_match_features(f)` becomes the single dispatch predicate.
Byte-parity with the plain forward is enforced by Task 5's GPU gates; this task's tests are
mock-level (routing, cache semantics, predicate logic).

**Files:**
- Modify: `collab_splats/localization/extractors.py`
- Test: `tests/localization/test_local_matcher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/localization/test_local_matcher.py`:

```python
@patch("vismatch.get_matcher")
def test_split_flag_off_for_unknown_wrappers(mock_get):
    # _fake_vismatch_matcher is a MagicMock — class name "MagicMock" != "LoMaMatcher",
    # so the split never activates and match_images runs the plain path untouched.
    mock_get.return_value = _fake_vismatch_matcher(stable_indices=True)
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    assert lm._split_loma_forward is False
    lm.has_stable_indices = True
    q = np.zeros((100, 100, 3), dtype=np.uint8)
    m = lm.match_images(q, q)
    assert len(m) == 4  # plain-path behavior byte-identical to before


@patch("vismatch.get_matcher")
def test_can_match_features_predicate(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    with_norm = LocalFeatures(
        keypoints=torch.zeros((2, 2)), descriptors=torch.zeros((2, 8)),
        keypoints_normalized=torch.zeros((2, 2)),
    )
    without_norm = LocalFeatures(keypoints=torch.zeros((2, 2)), descriptors=torch.zeros((2, 8)))
    # allowlisted NN model: always True, payload irrelevant
    lm = LocalMatcher("xfeat", device="cpu", probe=False)
    assert lm.can_match_features(without_norm) is True
    # non-allowlisted, no split: always False
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    assert lm.can_match_features(with_norm) is False
    # split active (forced — real activation needs the real wrapper): payload decides
    lm._split_loma_forward = True
    assert lm.can_match_features(with_norm) is True
    assert lm.can_match_features(without_norm) is False


@patch("vismatch.get_matcher")
def test_match_raises_for_loma_features_without_payload(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("loma", device="cpu", probe=False)
    lm._split_loma_forward = True  # loma not allowlisted; split forced for the unit test
    bare = LocalFeatures(keypoints=torch.zeros((2, 2)), descriptors=torch.zeros((2, 8)))
    with pytest.raises(NotImplementedError, match="match_images"):
        lm.match(bare, bare, image_hw=(100, 100))


@patch("vismatch.get_matcher")
def test_loma_cache_identity_keyed_and_cleared_at_cap(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("disk-lightglue", device="cpu", probe=False)
    calls = []

    def _fake_dnd(img):
        calls.append(img)
        return LocalFeatures(
            keypoints=torch.zeros((1, 2)), descriptors=torch.zeros((1, 8)),
            keypoints_normalized=torch.zeros((1, 2)),
        )

    with patch.object(lm, "_loma_detect_and_describe", side_effect=_fake_dnd):
        a = np.zeros((4, 4, 3), np.uint8)
        b = np.zeros((4, 4, 3), np.uint8)  # equal content, different object
        fa = lm._loma_features_cached(a)
        assert lm._loma_features_cached(a) is fa  # hit: same object, no re-extract
        lm._loma_features_cached(b)  # miss: identity, not content
        assert len(calls) == 2
        # fill to capacity: next insert clears wholesale, then re-extracts
        for i in range(510):
            lm._loma_features_cached(np.full((1, 1, 3), i % 255, np.uint8))
        c = np.ones((4, 4, 3), np.uint8)
        fc = lm._loma_features_cached(c)  # 513th distinct image -> clear happened before insert
        assert len(lm._loma_cache) < 512
        assert lm._loma_features_cached(c) is fc  # still a hit after clear
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly`
Expected: 3 new FAIL (`AttributeError: _split_loma_forward` / `can_match_features` /
`keypoints_normalized` unexpected keyword / `_loma_detect_and_describe` missing for
patch.object). `test_match_raises_for_loma_features_without_payload` may already pass —
Task 1's NotImplementedError also carries "match_images"; it pins the contract.

- [ ] **Step 3: Implement**

In `extractors.py`:

Top of file — ensure `import sys` is present in the stdlib imports (it is used to reach the
wrapper module's helpers).

`LocalFeatures` (~line 14) — add one field after `scales`:

```python
    keypoints_normalized: torch.Tensor | None = None  # (N, 2) pre-transform coords for the loma split
```

Module-level, next to `_DESCRIPTOR_NN_MODELS`:

```python
# Loma feature-cache capacity (match_images path): covers a 300-frame loop with headroom;
# cleared wholesale at the cap — identity keys are worthless once the caller drops its arrays.
_LOMA_CACHE_CAP = 512
```

In `__init__` (right after `self.has_stable_indices: bool | None = None`):

```python
        # Loma split: extract-once/match-from-features fast path (spec §2). One wrapper
        # class, one boolean — byte-parity with the plain forward is enforced by GPU suite tests.
        self._split_loma_forward = type(self._matcher).__name__ == "LoMaMatcher"
        self._loma_cache: dict[int, tuple[np.ndarray, LocalFeatures]] = {}
```

`can_match_features` (next to `supports_descriptor_matching`):

```python
    def can_match_features(self, features: LocalFeatures) -> bool:
        """Single dispatch predicate: can match() serve these features for this model?"""
        if self.supports_descriptor_matching:
            return True
        return self._split_loma_forward and features.keypoints_normalized is not None
```

`extract()` (~line 142) — route the split at the top (existing body stays as the fallback):

```python
        # Loma split: per-image half directly — skips the wrapper's self-pair match stage
        if self._split_loma_forward:
            return self._loma_detect_and_describe(image)
```

`match()` — change the guard from `supports_descriptor_matching` to `can_match_features` on
both inputs and add the loma branch before the NN code (final body):

```python
    def match(self, query: LocalFeatures, db: LocalFeatures, image_hw: tuple[int, int]) -> MatchResult:
        """Feature-level match over precomputed features — allowlisted NN models and the loma split.

        Match rows ARE keypoint-table indices by construction — no _recover_indices.
        """
        if not (self.can_match_features(query) and self.can_match_features(db)):
            raise NotImplementedError(
                f"LocalMatcher('{self._model_name}') cannot match these features at the "
                "feature level (no NN allowlist entry / no keypoints_normalized payload). "
                "Use match_images()."
            )
        if len(query.descriptors) == 0 or len(db.descriptors) == 0:
            return _empty_match()
        if self._split_loma_forward:
            return self._loma_match_features(query, db)
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

`match_images()` (~line 179) — split branch at the TOP of the body (current code stays
byte-identical as the fallback):

```python
        # Loma split: cached per-image halves + feature-level pair match (byte-parity
        # enforced by the suite's GPU parity tests).
        if self._split_loma_forward:
            f0 = self._loma_features_cached(query_image)
            f1 = self._loma_features_cached(ref_image)
            return self._loma_match_features(f0, f1)
```

New section at the end of the class (`######## Loma split — per-image / per-pair halves`).
The halves mirror `vismatch/im_models/loma.py:68-102` (`LoMaMatcher._forward`); the sandbox
must be entered explicitly because vismatch only wraps `__init__`/`_forward`
(`vismatch/base_matcher.py:19-30`); wrapper-module helpers (`to_pixel_coords`,
`filter_matches`) are reached via `sys.modules`:

```python
    def _loma_detect_and_describe(self, image: np.ndarray) -> LocalFeatures:
        """Per-image half of LoMaMatcher._forward: preprocess + detect_and_describe + coord chain.

        keypoints_normalized keeps the pre-transform coords the learned matcher consumes;
        keypoints replays the wrapper's own chain (to_pixel_coords -> rescale_coords -> the
        -0.5 COLMAP offset) over the FULL table — indexing a chained table equals chaining
        an indexed table (elementwise ops), so match-time pixel coords stay byte-identical.
        """
        from vismatch.import_sandbox import ImportSandbox

        m = self._matcher
        mod = sys.modules[type(m).__module__]  # wrapper module: to_pixel_coords
        with ImportSandbox.get(type(m).__module__), torch.inference_mode():
            img, orig_shape = m.preprocess(self._to_tensor(image))
            H, W = img.shape[-2:]
            kpts, desc, _, _ = m.matcher.detect_and_describe(img, m.max_num_keypoints)
            px = m.rescale_coords(mod.to_pixel_coords(kpts[0], H, W), *orig_shape, H, W) - 0.5
        feats = LocalFeatures(
            keypoints=torch.from_numpy(_to_numpy(px)),
            descriptors=torch.from_numpy(_to_numpy(desc[0])),
            keypoints_normalized=torch.from_numpy(_to_numpy(kpts[0])),
        )
        self._check_pixel_frame(feats.keypoints.numpy(), image.shape[:2], self._model_name)
        return feats

    def _loma_match_features(self, f0: LocalFeatures, f1: LocalFeatures) -> MatchResult:
        """Per-pair half of LoMaMatcher._forward on two extracted feature sets.

        Learned matcher on keypoints_normalized + descriptors, filter_matches, native
        indices; pixel coords by indexing the pre-chained keypoints tables. float32 in is
        fine — the matcher's autocast recasts at op boundaries either way (parity-gated).
        """
        from vismatch.import_sandbox import ImportSandbox

        m = self._matcher
        mod = sys.modules[type(m).__module__]  # wrapper module: filter_matches
        k0 = f0.keypoints_normalized.to(self._device).unsqueeze(0)
        k1 = f1.keypoints_normalized.to(self._device).unsqueeze(0)
        d0 = f0.descriptors.to(self._device).unsqueeze(0)
        d1 = f1.descriptors.to(self._device).unsqueeze(0)
        with ImportSandbox.get(type(m).__module__), torch.inference_mode():
            scores = m.matcher(k0, k1, d0, d1)["scores"]
            m0, _, _, _ = mod.filter_matches(scores, m.matcher.cfg.filter_threshold)
            valid = m0[0] > -1
            if not bool(valid.any()):
                return _empty_match()
            idx_q = torch.where(valid)[0].cpu().numpy().astype(np.int64)
            idx_db = m0[0][valid].cpu().numpy().astype(np.int64)
        return MatchResult(
            query_px=f0.keypoints.numpy()[idx_q],
            ref_px=f1.keypoints.numpy()[idx_db],
            idx_q=idx_q,
            idx_db=idx_db,
        )

    def _loma_features_cached(self, image: np.ndarray) -> LocalFeatures:
        """Identity-keyed per-image feature cache for the match_images path; cleared at cap.

        The `is` check makes id() reuse after GC a miss, never a wrong hit. Callers hold
        their image lists alive for the loop duration, so identity keying is exact and free.
        """
        entry = self._loma_cache.get(id(image))
        if entry is not None and entry[0] is image:
            return entry[1]
        if len(self._loma_cache) >= _LOMA_CACHE_CAP:
            self._loma_cache.clear()
        feats = self._loma_detect_and_describe(image)
        self._loma_cache[id(image)] = (image, feats)
        return feats
```

The `from vismatch.import_sandbox import ImportSandbox` imports stay inside the methods —
this is the documented optional-heavy-dep exception (vismatch is already lazy in `__init__`).

- [ ] **Step 4: Run the localization suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/ -v -p no:randomly`
Expected: all PASS (plain path untouched; split never activates under mocks — MagicMock's
class name is not "LoMaMatcher").

- [ ] **Step 5: Lint + commit**

```bash
ruff check collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git add collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git commit --only collab_splats/localization/extractors.py --only tests/localization/test_local_matcher.py -m "feat(localization): loma pair-forward split — feature-level matching via keypoints_normalized"
```

---

### Task 3: Persist `keypoints_normalized` in the zarr feature cache

`build_localization_db` already runs `detect_and_describe` on every image; this persists
what is currently computed and thrown away. One CSR-aligned float32 array next to
`keypoints`. Old caches without the array are NOT backfilled (mv_* precedent: absent, never
zeros) — they load as `keypoints_normalized=None` and loma falls back to pairwise.

**Files:**
- Modify: `collab_splats/localization/localizer.py` (`save_index` ~line 284; `load_reconstruction_features` ~line 112)
- Test: `tests/localization/test_localizer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/localization/test_localizer.py` (imports `LocalFeatures`, `LocalMatcher`,
`CameraLocalizer`, `load_reconstruction_features`, `MagicMock`, `numpy`, `torch` — add any
missing to the file's import block):

```python
def _norm_feats(n=5, d=8, with_norm=True):
    return LocalFeatures(
        keypoints=torch.rand(n, 2) * 50,
        descriptors=torch.rand(n, d),
        keypoints_normalized=torch.rand(n, 2) * 2 - 1 if with_norm else None,
    )


def _localizer_replaying(feats):
    """CameraLocalizer whose mock extractor replays the given per-frame features."""
    n = len(feats)
    extractor = MagicMock(spec=LocalMatcher)
    extractor.extract.side_effect = list(feats)
    return CameraLocalizer(
        world_points=np.zeros((n, 8, 8, 3), dtype=np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        images=[np.zeros((8, 8, 3), dtype=np.uint8)] * n,
        ids=[f"f{i}" for i in range(n)],
        extractor=extractor,
    )


def test_save_load_roundtrips_keypoints_normalized(tmp_path):
    feats = [_norm_feats() for _ in range(3)]
    _localizer_replaying(feats).save_index(tmp_path / "ff.zarr", "loma")
    loaded, _, _ = load_reconstruction_features(tmp_path / "ff.zarr", "loma")
    for orig, got in zip(feats, loaded):
        assert got.keypoints_normalized is not None
        np.testing.assert_array_equal(got.keypoints_normalized.numpy(), orig.keypoints_normalized.numpy())


def test_save_omits_keypoints_normalized_when_any_frame_lacks_it(tmp_path):
    # Unlike scores/scales, a zero-filled normalized table would be WRONG DATA — the array
    # is written only when every frame carries it; otherwise absent, never zeros.
    feats = [_norm_feats(), _norm_feats(with_norm=False), _norm_feats()]
    _localizer_replaying(feats).save_index(tmp_path / "ff.zarr", "loma")
    loaded, _, _ = load_reconstruction_features(tmp_path / "ff.zarr", "loma")
    assert all(f.keypoints_normalized is None for f in loaded)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localizer.py -v -p no:randomly`
Expected: first test FAIL (`AttributeError`/`AssertionError`: loaded `keypoints_normalized`
missing); second may already pass — that's fine, it pins the omission contract.

- [ ] **Step 3: Implement**

`save_index` — after the `scales` block (~line 358), add:

```python
        # keypoints_normalized: loma-split pre-transform coords the learned matcher consumes.
        # Written only when EVERY frame carries them — a zero-filled normalized table would be
        # wrong data, unlike scores/scales (absent, never zeros — mv_* precedent).
        has_norm = bool(self._frame_features) and all(
            f.keypoints_normalized is not None for f in self._frame_features
        )
        if has_norm and offsets[-1] > 0:
            all_norm = np.concatenate(
                [f.keypoints_normalized.numpy() for f in self._frame_features], axis=0
            ).astype(np.float32)
            rec_group.create_array(
                "keypoints_normalized",
                data=all_norm,
                chunks=(max(all_norm.shape[0], 1), 2),
                compressors=lz4,
            )
```

`load_reconstruction_features` — next to the `scores`/`scales` reads (~line 146):

```python
    all_norm = rec_group["keypoints_normalized"][:] if "keypoints_normalized" in rec_group else None
```

and in the per-frame loop (~line 156), build the slice and pass it through:

```python
        f_norm = torch.from_numpy(all_norm[s:e]) if all_norm is not None else None
```

adding `keypoints_normalized=f_norm` to the `LocalFeatures(...)` constructor call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localizer.py tests/localization/test_localization_cache.py -v -p no:randomly`
Expected: all PASS (existing cache tests unchanged — the new array is optional).

- [ ] **Step 5: Lint + commit**

```bash
ruff check collab_splats/localization/localizer.py tests/localization/test_localizer.py
git add collab_splats/localization/localizer.py tests/localization/test_localizer.py
git commit --only collab_splats/localization/localizer.py --only tests/localization/test_localizer.py -m "feat(localization): persist keypoints_normalized in the zarr feature cache"
```

---

### Task 4: verification.py dispatch — feature-level path via `can_match_features`

`verify_reconstruction` currently forces every `LocalMatcher` down the pairwise path
(`verification.py:184`/`:221`). Route feature-capable matchers (allowlisted NN models AND
loma on payload-bearing caches) to the existing feature-level else-branch: no images
touched, no extraction, match rows are table indices.

**Files:**
- Modify: `collab_splats/geometry/verification.py:182-237`
- Test: `tests/geometry/test_verification.py`

- [ ] **Step 1: Update the fixtures + write the failing test**

In `tests/geometry/test_verification.py`, EVERY `MagicMock(spec=LocalMatcher)` construction
(three sites: `_pairwise_matcher` ~line 95, and the inline mocks ~lines 228 and 244) gets one
line added right after its `has_stable_indices` assignment:

```python
    matcher.can_match_features.return_value = False
```

(MagicMock(spec=…) returns a truthy child mock for unset attributes — without this every old
pairwise test would silently take the new feature-level path.)

Append the new test (uses the existing `_synthetic_scene`/`_make_recon`/
`_features_from_keypoints` helpers):

```python
def test_feature_capable_localmatcher_skips_images(tmp_path):
    """A feature-capable LocalMatcher takes the feature-level branch: match() on features,
    match_images and `images` untouched, no stable-indices requirement."""
    _, extrinsics, kps = _synthetic_scene()
    matcher = MagicMock(spec=LocalMatcher)
    matcher.model_name = "stub-feature-capable"
    matcher.can_match_features.return_value = True
    matcher.has_stable_indices = False  # irrelevant on the feature-level path

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
        # no `images` passed — the feature-level path must not require them
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
    # Feature-capable LocalMatchers (NN allowlist, or loma with a payload-bearing cache)
    # match precomputed features directly — no images, no extraction. Everything else
    # pairwise: those must prove index stability up front — a silent skip here would
    # surface later as a missing verification.json with no explanation.
    pairwise = isinstance(matcher, LocalMatcher) and not all(
        matcher.can_match_features(f) for f in features
    )
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
git commit --only collab_splats/geometry/verification.py --only tests/geometry/test_verification.py -m "feat(geometry): feature-level dispatch via can_match_features in verification"
```

---

### Task 5: GPU parity tests (real models — the load-bearing gates that license the fast paths)

With no runtime probes, THESE tests are the equivalence mechanism: the loma split must be
byte-identical to the plain forward (extract, match_images, and match-after-zarr-roundtrip),
and xfeat's `match()` must equal its own `match_images`. Extending `_DESCRIPTOR_NN_MODELS`
or adding another wrapper split requires adding a new passing test here.

**Files:**
- Test: `tests/localization/test_local_matcher.py` (append; CUDA-gated)

- [ ] **Step 1: Write the tests**

```python
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-model parity needs CUDA")


def _real_pair(seed=3):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 255, (240, 320, 3)).astype(np.uint8)
    return a, np.roll(a, 12, axis=1)


@requires_cuda
def test_real_loma_split_extract_parity():
    """Split extract() byte-identical to the plain self-pair extract it replaces."""
    lm = LocalMatcher("loma")
    assert lm._split_loma_forward, "LoMaMatcher wrapper class no longer detected"
    a, _ = _real_pair()
    fast = lm.extract(a)
    lm._split_loma_forward = False
    ref = lm.extract(a)
    lm._split_loma_forward = True
    np.testing.assert_array_equal(fast.keypoints.numpy(), ref.keypoints.numpy())
    np.testing.assert_array_equal(fast.descriptors.numpy(), ref.descriptors.numpy())
    assert fast.keypoints_normalized is not None and ref.keypoints_normalized is None


@requires_cuda
def test_real_loma_split_match_parity_and_zarr_roundtrip(tmp_path):
    """match_images (split) and match() after a zarr save/load both == plain pair forward."""
    lm = LocalMatcher("loma")
    a, b = _real_pair(seed=4)
    # reference: plain wrapper forward
    lm._split_loma_forward = False
    ref = lm.match_images(a, b)
    lm._split_loma_forward = True
    assert len(ref) > 0

    # split path through match_images (in-memory cache)
    fast = lm.match_images(a, b)
    np.testing.assert_array_equal(fast.query_px, ref.query_px)
    np.testing.assert_array_equal(fast.ref_px, ref.ref_px)
    np.testing.assert_array_equal(fast.idx_q, ref.idx_q)  # native == recovered (probe-exact)
    np.testing.assert_array_equal(fast.idx_db, ref.idx_db)

    # zarr roundtrip: extract -> save via CameraLocalizer -> load -> match()
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
    from collab_splats.localization.localizer import load_reconstruction_features

    loaded, _, _ = load_reconstruction_features(tmp_path / "ff.zarr", "loma")
    assert lm.can_match_features(loaded[0]) and lm.can_match_features(loaded[1])
    m = lm.match(loaded[0], loaded[1], image_hw=a.shape[:2])
    np.testing.assert_array_equal(m.query_px, ref.query_px)
    np.testing.assert_array_equal(m.ref_px, ref.ref_px)
    np.testing.assert_array_equal(m.idx_q, ref.idx_q)
    np.testing.assert_array_equal(m.idx_db, ref.idx_db)


@requires_cuda
def test_real_xfeat_general_path_matches_pairwise():
    """Parity gate for _DESCRIPTOR_NN_MODELS entry 'xfeat' — match() == match_images()."""
    lm = LocalMatcher("xfeat")
    assert lm.supports_descriptor_matching is True
    a, b = _real_pair(seed=5)
    m = lm.match(lm.extract(a), lm.extract(b), image_hw=a.shape[:2])
    ref = lm.match_images(a, b)
    assert len(m) == len(ref) and len(m) > 0
    order_m, order_ref = np.argsort(m.idx_q), np.argsort(ref.idx_q)
    np.testing.assert_array_equal(m.idx_q[order_m], ref.idx_q[order_ref])
    np.testing.assert_array_equal(m.idx_db[order_m], ref.idx_db[order_ref])
```

(The `load_reconstruction_features` import inside the test keeps the module import block
untouched — move it to the top-of-file imports instead if Task 3 already added it there.)

- [ ] **Step 2: Run on the A40 (serial — no concurrent GPU jobs)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly -k "real_"`
Expected: 3 PASS. **If any fails, STOP — do not weaken the assertions.** For loma: diff the
halves tensor-by-tensor (extract parity first; then dtype — if `detect_and_describe`
returns bf16 tensors, the float32 store must be shown equivalent under the matcher's
autocast, and if it is not, keep original-dtype tensors in the in-memory path and record
the dtype in zarr attrs for cast-on-load). For xfeat: read the vismatch xfeat wrapper's
match stage — if it thresholds cossim or is not plain mutual-NN, remove "xfeat" from
`_DESCRIPTOR_NN_MODELS` (Task 1's non-allowlisted tests then cover it) and record why in
the spec.

- [ ] **Step 3: Commit**

```bash
git add tests/localization/test_local_matcher.py
git commit --only tests/localization/test_local_matcher.py -m "test(localization): GPU parity gates licensing the loma split + xfeat allowlist"
```

---

### Task 6: Phase-timing instrumentation (the residual-attribution work)

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

### Task 7: CLAUDE.md stale architecture line

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

### Task 8: Evidence runs (tmux, serial, human-gated compute)

Measured, not assumed. Two runs against the spec's baselines: per-pair 993.8 ms and full
verify() 2511.6 s on `/workspace/outputs/2026_07_15-Goprosplat-GH010229` (vggt_omega + loma).

**Files:**
- Create: `/tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/pair_bench.py`
- Create: `/tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/verify_bench.py`
- Modify: `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md` (append results)

- [ ] **Step 1: Write the 30-pair warm harness**

`pair_bench.py`:

```python
"""30-pair warm loma benchmark: split feature-level match vs plain pair forward."""

import time
from pathlib import Path

import numpy as np
import torch
import zarr

from collab_splats.localization.extractors import LocalMatcher

SCENE = Path("/workspace/outputs/2026_07_15-Goprosplat-GH010229")
images = zarr.open_group(SCENE / "frames.zarr", mode="r")["images"]
rng = np.random.default_rng(0)
n = images.shape[0]
# 30 pairs stratified over the verify() gap distribution (1, 2, 4, 8, 16)
pairs = [(int(i), min(int(i) + g, n - 1)) for i, g in zip(rng.integers(0, n - 17, 30), [1, 2, 4, 8, 16] * 6)]

lm = LocalMatcher("loma")
frames = {i: np.asarray(images[i]) for i in sorted({i for p in pairs for i in p})}
feats = {i: lm.extract(img) for i, img in frames.items()}  # warm extraction, off the clock

# split path: feature-level match on precomputed features
torch.cuda.synchronize()
t0 = time.perf_counter()
for i, j in pairs:
    lm.match(feats[i], feats[j], image_hw=frames[i].shape[:2])
torch.cuda.synchronize()
split_ms = 1000 * (time.perf_counter() - t0) / len(pairs)

# plain pair forward: the pre-split baseline path
lm._split_loma_forward = False
lm.match_images(frames[pairs[0][0]], frames[pairs[0][1]])  # warmup
torch.cuda.synchronize()
t0 = time.perf_counter()
for i, j in pairs:
    lm.match_images(frames[i], frames[j])
torch.cuda.synchronize()
plain_ms = 1000 * (time.perf_counter() - t0) / len(pairs)

print(f"[pair_bench] split {split_ms:.1f} ms/pair vs plain {plain_ms:.1f} ms/pair (baseline 993.8 ms)")
```

- [ ] **Step 2: Write the full-scene driver**

`verify_bench.py`:

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

Note: `verify(overwrite=True)` reruns `build_localization_db`, so the rebuilt cache carries
`keypoints_normalized` and the run exercises the zero-extraction feature-level path.

- [ ] **Step 3: Run both in tmux (nothing else on the GPU, sequential)**

```bash
tmux new-session -d -s verify_bench
tmux send-keys -t verify_bench "/opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/pair_bench.py 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/pair_bench.log && /opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/verify_bench.py 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/86f2bc5a-dfc8-49fc-931d-c9ace1e3ce72/scratchpad/verify_bench.log" Enter
```

Wait for completion (poll `tmux has-session` / tail the logs), then read both logs.
Expected: split well under 100 ms/pair; verify() total well under 1000 s; `phase_seconds`
in the log attributes the former ~658 s residual.

- [ ] **Step 4: Attack the biggest attributed residual term — only if it is ours**

Per the spec: if `db_export`/`db_match_writes`/cold start dominates the residual, fix that
one term (vectorize the write loop / amortize the load) as a follow-up commit with a second
timed run. If pycolmap's C++ (`verify_matches`, `triangulate`) dominates, record the number
and stop — that floor belongs to the pycolmap-native migration.

- [ ] **Step 5: Append the measured numbers to the report + commit**

Append to `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`: pair_bench
split/plain ms vs 993.8 ms, verify() total wall time vs 2511.6 s, the full `phase_seconds`
dict, cold-start log lines, and the per-pair warm number implied by
`pair_matching / n_pairs`.

```bash
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit --only docs/superpowers/specs/2026-08-20-scene-error-report-measured.md -m "docs(specs): measured verify() timings with matcher fast paths"
```

---

### Final gate

- [ ] Full suite: `/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly` — no new failures vs `docs/known-test-failures.md`.
- [ ] `graphify update .` (repo rule: keep the graph current after code changes).
