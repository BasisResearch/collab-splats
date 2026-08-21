# Matcher Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the 12.4× per-image extraction redundancy in `verify()`'s pair-matching loop: feature-capable models match precomputed zarr features directly, and loma — the shipping default — gets its pair forward split into per-image and per-pair inline branches with the per-image payload persisted in the localization DB, so loma too matches from stored features (extraction inside verify → 0; matching ~1853 s → ~100 s; a payload-less old cache is rebuilt once, ~66 s, not served by a fallback path), plus phase-timing instrumentation to attribute the ~658 s residual.

**Architecture:** `collab_splats/localization/extractors.py` (LocalMatcher: `match()` — kornia mutual-NN for allowlisted NN models, loma per-pair branch for the split; gating is ONE exported list `FEATURE_MATCH_MODELS`, no predicate methods; the loma per-image half is an inline branch in `extract()` activated by one boolean `_split_loma_forward`; `LocalFeatures` gains ONE optional field `keypoints_normalized`). `collab_splats/localization/localizer.py`: loader renamed `load_reconstruction_features` → `load_localization_db` (the zarr `local_features/<extractor>/reconstruction` group IS the localization DB — pycolmap Database shape) and the save/load pair persists/restores the payload (one added CSR array — no new cache layer, no in-memory cache). `collab_splats/geometry/verification.py` dispatches on `matcher.model_name not in FEATURE_MATCH_MODELS` and gains phase timings. `collab_splats/wrapper/reconstructor.py` `verify()` rebuilds the DB once when a split-capable matcher meets a payload-less cache. `match_images()` fully untouched. No runtime probes for the new paths — byte-parity is enforced by GPU parity tests in the suite (env pinned); the pre-existing `_probe_index_stability` is untouched.

**Tech Stack:** vismatch model zoo (explicit `ImportSandbox` reach-in for loma — vismatch only wraps `__init__`/`_forward`, `vismatch/base_matcher.py:19-30`), `kornia.feature.match_mnn` (kornia 0.8.2 in env), zarr v3 (`compressors=[BloscCodec]`). Spec: `docs/superpowers/specs/2026-08-20-matcher-feature-cache-design.md`.

**Repo rules that bind every task:** stage named files only (never `git add -A`/`.`); commit with `git commit --only <files>`; `docs/superpowers/` needs `git add -f`; python is `/opt/venv/reconstruction/bin/python`; never repo-wide `black .`; lint gate is `ruff check <touched files>` only; GPU runs serial in tmux (single A40); repo text search with `git grep` (plain `grep -r` hangs under the rtk hook).

**Key parity fact used throughout:** loma's coordinate chain (`to_pixel_coords` → `rescale_coords` → −0.5) is elementwise, so indexing the pre-chained full `keypoints` table by match indices is byte-identical to chaining the indexed subset (what `LoMaMatcher._forward` does, `vismatch/im_models/loma.py:68-102`). Hence NO shape metadata is stored — only `keypoints_normalized` (pre-transform coords the learned matcher consumes) plus the existing `keypoints`/`descriptors`.

---

### Task 1: Implement the reserved `match()` seam — kornia mutual-NN behind `FEATURE_MATCH_MODELS`

The general path: mutual-NN over precomputed descriptors via `kornia.feature.match_mnn`
(L2 on unit vectors orders identically to cosine, and match_mnn admits every mutual pair —
the min_cossim=-1 semantics). Gated by membership in ONE exported list,
`FEATURE_MATCH_MODELS = {"xfeat"}` for now; Task 2 adds `"loma"` together with its branch.
Membership is licensed by the Task 6 GPU parity test, not a runtime probe.

The reserved signature's `image_hw` parameter is DROPPED (`match(query, db)`): neither
branch uses it — pixel coords come from the stored keypoint tables. The only existing
caller is the `NotImplementedError` test; `verification.py`'s else-branch call sheds the
argument in Task 5.

**Files:**
- Modify: `collab_splats/localization/extractors.py` (`match()` at ~line 153; imports)
- Test: `tests/localization/test_local_matcher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/localization/test_local_matcher.py` (add `FEATURE_MATCH_MODELS` to the
existing `from collab_splats.localization.extractors import ...` line):

```python
def _one_hot_features(rows, d=8):
    """LocalFeatures whose descriptors are one-hot rows — mutual-NN is exactly identity."""
    kpts = np.stack([np.arange(len(rows)), np.arange(len(rows))], axis=1).astype(np.float32) * 10
    desc = np.eye(d, dtype=np.float32)[rows]
    return LocalFeatures(keypoints=torch.from_numpy(kpts), descriptors=torch.from_numpy(desc))


@patch("vismatch.get_matcher")
def test_match_mutual_nn_for_allowlisted_model(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()
    assert "xfeat" in FEATURE_MATCH_MODELS
    lm = LocalMatcher("xfeat", device="cpu", probe=False)
    q = _one_hot_features([0, 1, 2, 3])
    db = _one_hot_features([3, 2, 1, 0])  # same one-hot basis, permuted rows
    m = lm.match(q, db)
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
    m = lm.match(empty, _one_hot_features([0, 1]))
    assert len(m) == 0 and m.idx_q is not None  # empty but indexable


@patch("vismatch.get_matcher")
def test_match_still_raises_for_non_listed_model(mock_get):
    # Not in FEATURE_MATCH_MODELS — the NotImplementedError contract survives.
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("roma", device="cpu", probe=False)
    q = _one_hot_features([0, 1])
    with pytest.raises(NotImplementedError, match="match_images"):
        lm.match(q, q)
```

Also edit `test_descriptor_level_match_unsupported` (existing, ~line 103): its call drops
the `image_hw=(100, 100)` argument (otherwise the removed parameter raises TypeError before
the NotImplementedError). It uses "disk-lightglue" — not in the list, contract survives.

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly`
Expected: collection error — `ImportError: cannot import name 'FEATURE_MATCH_MODELS'` (the
whole file fails to collect until the constant exists; after a stub constant it would be
3 new FAIL with NotImplementedError on the xfeat tests).

- [ ] **Step 3: Implement**

In `extractors.py`, add to the top-of-file imports (imports-at-top rule):

```python
from kornia.feature import match_mnn
```

Add a module-level constant under the existing `# VisMatch-backed matcher` divider (near the
blocklists):

```python
# Models match() can serve from precomputed features. "xfeat": its own match stage IS
# descriptor mutual-NN. "loma" (added with its split): pair forward split into per-image /
# per-pair halves. Membership is licensed by GPU parity tests in the suite — extending this
# set requires a new passing parity test. Never silently substitute NN for a learned matcher.
FEATURE_MATCH_MODELS = {"xfeat"}
```

Replace the `match()` body (the NotImplementedError message must keep the substring
`match_images` for the existing test):

```python
    def match(self, query: LocalFeatures, db: LocalFeatures) -> MatchResult:
        """Feature-level match over precomputed features (FEATURE_MATCH_MODELS only).

        Match rows ARE keypoint-table indices by construction — no _recover_indices.
        """
        if self._model_name not in FEATURE_MATCH_MODELS:
            raise NotImplementedError(
                f"LocalMatcher('{self._model_name}') has no feature-level matching "
                "(not in FEATURE_MATCH_MODELS — its match stage is not parity-proven "
                "on precomputed features). Use match_images()."
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
git commit --only collab_splats/localization/extractors.py --only tests/localization/test_local_matcher.py -m "feat(localization): feature-level match() — kornia mutual-NN behind FEATURE_MATCH_MODELS"
```

---

### Task 2: Loma split — inline per-image / per-pair branches + `keypoints_normalized`

Loma's learned match stage cannot take the NN path. Split its pair forward
(`vismatch/im_models/loma.py:68-102`, vismatch in `/opt/venv/reconstruction`) into a
per-image branch in `extract()` and a per-pair branch in `match()` — NO new methods,
activated by one boolean keyed on the wrapper class name. `LocalFeatures` gains ONE
optional field. `"loma"` joins `FEATURE_MATCH_MODELS`. `match_images()` untouched.
Byte-parity with the plain forward is enforced by Task 6's GPU gates; this task's tests
are mock-level (flag activation, payload guard).

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
def test_loma_match_without_payload_names_the_rebuild(mock_get):
    # "loma" is in FEATURE_MATCH_MODELS, so the list guard passes; the split branch then
    # refuses payload-less features with the rebuild hint (defensive — verify() pre-rebuilds).
    mock_get.return_value = _fake_vismatch_matcher()
    lm = LocalMatcher("loma", device="cpu", probe=False)
    lm._split_loma_forward = True  # real activation needs the real wrapper; forced here
    bare = LocalFeatures(keypoints=torch.ones((2, 2)), descriptors=torch.ones((2, 8)))
    with pytest.raises(ValueError, match="rebuild"):
        lm.match(bare, bare)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly`
Expected: 2 new FAIL — `AttributeError: _split_loma_forward`, and on the second test
NotImplementedError (loma not yet in the list) instead of the ValueError.

- [ ] **Step 3: Implement**

In `extractors.py`:

Top of file — add `import sys` to the stdlib imports (used to reach the wrapper module's
helpers).

`LocalFeatures` (~line 14) — add one field after `scales`:

```python
    keypoints_normalized: torch.Tensor | None = None  # (N, 2) pre-transform coords for the loma split
```

`FEATURE_MATCH_MODELS` — add `"loma"`:

```python
FEATURE_MATCH_MODELS = {"xfeat", "loma"}
```

In `__init__` (right after `self._matcher = vismatch.get_matcher(...)`):

```python
        # Loma split: extract-once/match-from-features fast path (spec §2). One wrapper
        # class, one boolean — byte-parity with the plain forward is enforced by GPU suite tests.
        self._split_loma_forward = type(self._matcher).__name__ == "LoMaMatcher"
```

`extract()` (~line 142) — insert the per-image branch after the `hw = image.shape[:2]` line
(existing body stays as the general path):

```python
        # Loma split: per-image half of LoMaMatcher._forward — detect_and_describe once,
        # replay the wrapper's coord chain (to_pixel_coords -> rescale_coords -> the -0.5
        # COLMAP offset) over the FULL table, and keep the pre-transform coords the learned
        # matcher consumes. Indexing a chained table equals chaining an indexed table
        # (elementwise ops), so match-time pixel coords stay byte-identical. Also skips the
        # wrapper's self-pair match stage. Sandbox entered explicitly — vismatch only wraps
        # __init__/_forward.
        if self._split_loma_forward:
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
            self._check_pixel_frame(feats.keypoints.numpy(), hw, self._model_name)
            return feats
```

`match()` — insert the per-pair branch between the empty guard and the mutual-NN code:

```python
        if self._split_loma_forward:
            # Per-pair half of LoMaMatcher._forward: learned matcher on the stored
            # pre-transform coords + descriptors, wrapper's filter, native indices.
            # float32 in is fine — the matcher's autocast recasts at op boundaries
            # either way (parity-gated).
            if query.keypoints_normalized is None or db.keypoints_normalized is None:
                raise ValueError(
                    "loma feature-level match needs keypoints_normalized — this feature "
                    "cache predates the payload; rebuild the localization DB "
                    "(build_localization_db(overwrite=True))."
                )
            from vismatch.import_sandbox import ImportSandbox

            m = self._matcher
            mod = sys.modules[type(m).__module__]  # wrapper module: filter_matches
            k0 = query.keypoints_normalized.to(self._device).unsqueeze(0)
            k1 = db.keypoints_normalized.to(self._device).unsqueeze(0)
            d0 = query.descriptors.to(self._device).unsqueeze(0)
            d1 = db.descriptors.to(self._device).unsqueeze(0)
            with ImportSandbox.get(type(m).__module__), torch.inference_mode():
                scores = m.matcher(k0, k1, d0, d1)["scores"]
                m0, _, _, _ = mod.filter_matches(scores, m.matcher.cfg.filter_threshold)
                valid = m0[0] > -1
                if not bool(valid.any()):
                    return _empty_match()
                idx_q = torch.where(valid)[0].cpu().numpy().astype(np.int64)
                idx_db = m0[0][valid].cpu().numpy().astype(np.int64)
            return MatchResult(
                query_px=query.keypoints.numpy()[idx_q],
                ref_px=db.keypoints.numpy()[idx_db],
                idx_q=idx_q,
                idx_db=idx_db,
            )
```

The `from vismatch.import_sandbox import ImportSandbox` imports stay inside the branches —
the documented optional-heavy-dep exception (vismatch is already lazy in `__init__`).
`match_images()` is NOT touched.

- [ ] **Step 4: Run the localization suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/ -v -p no:randomly`
Expected: all PASS (plain paths untouched; split never activates under mocks — MagicMock's
class name is not "LoMaMatcher").

- [ ] **Step 5: Lint + commit**

```bash
ruff check collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git add collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git commit --only collab_splats/localization/extractors.py --only tests/localization/test_local_matcher.py -m "feat(localization): loma pair-forward split — inline extract()/match() branches via keypoints_normalized"
```

---

### Task 3: Rename `load_reconstruction_features` → `load_localization_db`

The zarr `local_features/<extractor>/reconstruction` group IS the localization DB
(pycolmap Database shape); the old name reads as if it loaded reconstruction geometry.
Mechanical rename across the 6 live code files — historical docs/superpowers files are NOT
edited (repo rule).

**Files:**
- Modify: `collab_splats/localization/localizer.py`, `collab_splats/localization/__init__.py`,
  `collab_splats/wrapper/reconstructor.py`, `evals/scripts/eval_verification.py`,
  `tests/localization/test_localization_cache.py`, `tests/wrapper/test_verify_stage.py`

- [ ] **Step 1: Rename**

`git grep -n load_reconstruction_features -- '*.py'` — expect exactly the 6 files above.
Rename every occurrence (definition, imports, calls, and the monkeypatch string
`"collab_splats.localization.localizer.load_reconstruction_features"` in
`test_verify_stage.py`). Update the function's docstring to say "localization DB" and any
log-message strings that carry the old name.

- [ ] **Step 2: Verify + run the touched suites**

`git grep -n load_reconstruction_features -- '*.py'` — expect zero hits.
Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/ tests/wrapper/test_verify_stage.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 3: Lint + commit**

```bash
ruff check collab_splats/localization/localizer.py collab_splats/localization/__init__.py collab_splats/wrapper/reconstructor.py evals/scripts/eval_verification.py tests/localization/test_localization_cache.py tests/wrapper/test_verify_stage.py
git add collab_splats/localization/localizer.py collab_splats/localization/__init__.py collab_splats/wrapper/reconstructor.py evals/scripts/eval_verification.py tests/localization/test_localization_cache.py tests/wrapper/test_verify_stage.py
git commit --only collab_splats/localization/localizer.py --only collab_splats/localization/__init__.py --only collab_splats/wrapper/reconstructor.py --only evals/scripts/eval_verification.py --only tests/localization/test_localization_cache.py --only tests/wrapper/test_verify_stage.py -m "refactor(localization): rename load_reconstruction_features to load_localization_db"
```

---

### Task 4: Persist `keypoints_normalized` in the localization DB

`build_localization_db` already runs `detect_and_describe` on every image; this persists
what is currently computed and thrown away. One CSR-aligned float32 array next to
`keypoints`, written only when EVERY frame carries it — a zero-filled normalized table
would be wrong data, unlike scores/scales (absent, never zeros — mv_* precedent). Old
caches load as `keypoints_normalized=None` (Task 5 makes `verify()` rebuild them once).

**Files:**
- Modify: `collab_splats/localization/localizer.py` (`save_index` ~line 284; `load_localization_db` ~line 112)
- Test: `tests/localization/test_localizer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/localization/test_localizer.py` (imports `LocalFeatures`, `LocalMatcher`,
`CameraLocalizer`, `load_localization_db`, `MagicMock`, `numpy`, `torch` — add any missing
to the file's import block):

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
    loaded, _, _ = load_localization_db(tmp_path / "ff.zarr", "loma")
    for orig, got in zip(feats, loaded):
        assert got.keypoints_normalized is not None
        np.testing.assert_array_equal(got.keypoints_normalized.numpy(), orig.keypoints_normalized.numpy())


def test_save_omits_keypoints_normalized_when_any_frame_lacks_it(tmp_path):
    # Unlike scores/scales, a zero-filled normalized table would be WRONG DATA — the array
    # is written only when every frame carries it; otherwise absent, never zeros.
    feats = [_norm_feats(), _norm_feats(with_norm=False), _norm_feats()]
    _localizer_replaying(feats).save_index(tmp_path / "ff.zarr", "loma")
    loaded, _, _ = load_localization_db(tmp_path / "ff.zarr", "loma")
    assert all(f.keypoints_normalized is None for f in loaded)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localizer.py -v -p no:randomly`
Expected: first test FAIL (loaded `keypoints_normalized` is None); second may already pass —
that's fine, it pins the omission contract.

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

`load_localization_db` — next to the `scores`/`scales` reads (~line 146):

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
git commit --only collab_splats/localization/localizer.py --only tests/localization/test_localizer.py -m "feat(localization): persist keypoints_normalized in the localization DB"
```

---

### Task 5: verification dispatch via `FEATURE_MATCH_MODELS` + rebuild-once in `verify()`

Two consumers of the new paths. (a) `verify_reconstruction` currently forces every
`LocalMatcher` down the pairwise path (`verification.py:184`/`:221`) — route
`FEATURE_MATCH_MODELS` members to the existing feature-level else-branch: no images
touched, no extraction, match rows are table indices. Existing mock fixtures need ZERO
edits: a `MagicMock` `model_name` is never in the set, so they stay pairwise. (b)
`Reconstructor.verify()` rebuilds the localization DB once (~66 s) when a split-capable
matcher meets a payload-less old cache — no degraded fallback path.

**Files:**
- Modify: `collab_splats/geometry/verification.py:182-237`
- Modify: `collab_splats/wrapper/reconstructor.py` (`verify()`, ~lines 1117-1140)
- Test: `tests/geometry/test_verification.py`, `tests/wrapper/test_verify_stage.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_verification.py` (uses the existing `_synthetic_scene`/
`_make_recon`/`_features_from_keypoints` helpers):

```python
def test_feature_capable_localmatcher_skips_images(tmp_path):
    """A FEATURE_MATCH_MODELS matcher takes the feature-level branch: match() on features,
    match_images and `images` untouched, no stable-indices requirement."""
    _, extrinsics, kps = _synthetic_scene()
    matcher = MagicMock(spec=LocalMatcher)
    matcher.model_name = "xfeat"  # in FEATURE_MATCH_MODELS -> feature-level dispatch
    matcher.has_stable_indices = False  # irrelevant on the feature-level path

    def _match(query, db):
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

In `tests/wrapper/test_verify_stage.py`, extend `_stub_verify_call` and add the rebuild test:

- `_stub_verify_call` gains an optional `load_fn=None` parameter; when given, it replaces
  the default load lambda in the monkeypatch (target string is now
  `"collab_splats.localization.localizer.load_localization_db"` after Task 3).
- Replace `r.build_localization_db = lambda: None` with kwargs capture, and return it:

```python
    rebuilds = []
    r.build_localization_db = lambda **kw: rebuilds.append(kw)
    ...
    return captured, accesses, rebuilds
```

  (Update the two existing callers' unpacking. Note verify()'s unconditional up-front
  `build_localization_db()` call records `{}` as the first entry.)

```python
def test_verify_rebuilds_db_when_loma_payload_missing(tmp_path, monkeypatch):
    """Split-capable matcher + payload-less cache: one rebuild + reload before verification."""
    matcher = MagicMock(spec=LocalMatcher)
    matcher.has_stable_indices = True
    matcher._split_loma_forward = True  # assignment is allowed on spec mocks; get-after-set works
    stale = SimpleNamespace(keypoints_normalized=None)
    fresh = SimpleNamespace(keypoints_normalized=np.zeros((1, 2), np.float32))
    loads = []

    def _load(path, name):
        loads.append(name)
        feats = [stale, stale] if len(loads) == 1 else [fresh, fresh]
        return feats, ["frame_000003.jpg", "frame_000007.jpg"], (4, 4)

    captured, _, rebuilds = _stub_verify_call(tmp_path, monkeypatch, matcher, load_fn=_load)
    assert rebuilds == [{}, {"overwrite": True}]  # up-front build, then the payload rebuild
    assert len(loads) == 2  # reloaded after the rebuild
    assert captured["features"] == [fresh, fresh]
```

The two existing `_stub_verify_call` tests must keep passing WITHOUT edits to their
matchers: the reconstructor reads the flag via `getattr(matcher, "_split_loma_forward",
False)` — unset on the spec mock and absent on the SimpleNamespace stub, so the rebuild
block never touches their string-typed features.

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py tests/wrapper/test_verify_stage.py -v -p no:randomly`
Expected: `test_feature_capable_localmatcher_skips_images` FAIL (`ValueError: pairwise
matcher requires \`images\`` — the isinstance branch fires);
`test_verify_rebuilds_db_when_loma_payload_missing` FAIL (`rebuilds == [{}]`, one load);
existing PASS.

- [ ] **Step 3: Implement the dispatch**

In `verification.py`, add `FEATURE_MATCH_MODELS` to the existing
`from collab_splats.localization.extractors import ...` line, and replace the guard block
(starting `if isinstance(matcher, LocalMatcher):` ~line 184) with:

```python
    # FEATURE_MATCH_MODELS members (NN-parity models + the loma split) match precomputed
    # features directly — no images, no extraction. Everything else pairwise: those must
    # prove index stability up front — a silent skip here would surface later as a missing
    # verification.json with no explanation.
    pairwise = isinstance(matcher, LocalMatcher) and matcher.model_name not in FEATURE_MATCH_MODELS
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
`if isinstance(matcher, LocalMatcher):` to `if pairwise:`, and the else-branch call sheds
its third argument: `matcher.match(features[i], features[j])` (the `image_hw` parameter is
gone — Task 1). Branch bodies otherwise stay as they are.

- [ ] **Step 4: Implement rebuild-once in `reconstructor.py` `verify()`**

Move the `matcher = LocalMatcher(extractor_name)` construction (currently ~line 1137)
ABOVE the `load_localization_db` call (~line 1125), then insert after the load, BEFORE the
stem check:

```python
        # Loma matches from stored features (keypoints_normalized). A cache from before
        # that array existed gets rebuilt once (~1 min) — no degraded fallback path.
        # getattr: duck-typed test stubs and non-split matchers lack the attribute.
        if getattr(matcher, "_split_loma_forward", False) and any(
            f.keypoints_normalized is None for f in features
        ):
            logger.info("verify(): feature cache lacks keypoints_normalized — rebuilding localization DB")
            self.build_localization_db(overwrite=True)
            features, ids, _ = load_localization_db(
                self.backend_dir / "feedforward.zarr", extractor_name
            )
```

(Private-attr read from inside the package — accepted, commented.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py tests/wrapper/test_verify_stage.py tests/localization/test_local_matcher.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 6: Lint + commit**

```bash
ruff check collab_splats/geometry/verification.py collab_splats/wrapper/reconstructor.py tests/geometry/test_verification.py tests/wrapper/test_verify_stage.py
git add collab_splats/geometry/verification.py collab_splats/wrapper/reconstructor.py tests/geometry/test_verification.py tests/wrapper/test_verify_stage.py
git commit --only collab_splats/geometry/verification.py --only collab_splats/wrapper/reconstructor.py --only tests/geometry/test_verification.py --only tests/wrapper/test_verify_stage.py -m "feat(geometry): feature-level verification dispatch + one-time localization DB rebuild"
```

---

### Task 6: GPU parity tests (real models — the load-bearing gates that license the fast paths)

With no runtime probes, THESE tests are the equivalence mechanism: the loma split must be
byte-identical to the plain forward (extract, and match-after-zarr-roundtrip), and xfeat's
`match()` must equal its own `match_images`. Extending `FEATURE_MATCH_MODELS` requires
adding a new passing test here.

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
def test_real_loma_split_match_parity_after_zarr_roundtrip(tmp_path):
    """match() on zarr-roundtripped split features == the plain pair forward, byte-identical."""
    lm = LocalMatcher("loma")
    a, b = _real_pair(seed=4)
    # reference: plain wrapper forward
    lm._split_loma_forward = False
    ref = lm.match_images(a, b)
    lm._split_loma_forward = True
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
    loaded, _, _ = load_localization_db(tmp_path / "ff.zarr", "loma")
    assert all(f.keypoints_normalized is not None for f in loaded)
    m = lm.match(loaded[0], loaded[1])
    np.testing.assert_array_equal(m.query_px, ref.query_px)
    np.testing.assert_array_equal(m.ref_px, ref.ref_px)
    np.testing.assert_array_equal(m.idx_q, ref.idx_q)  # native == recovered (probe-exact)
    np.testing.assert_array_equal(m.idx_db, ref.idx_db)


@requires_cuda
def test_real_xfeat_general_path_matches_pairwise():
    """Parity gate for the FEATURE_MATCH_MODELS entry 'xfeat' — match() == match_images()."""
    lm = LocalMatcher("xfeat")
    a, b = _real_pair(seed=5)
    m = lm.match(lm.extract(a), lm.extract(b))
    ref = lm.match_images(a, b)
    assert len(m) == len(ref) and len(m) > 0
    order_m, order_ref = np.argsort(m.idx_q), np.argsort(ref.idx_q)
    np.testing.assert_array_equal(m.idx_q[order_m], ref.idx_q[order_ref])
    np.testing.assert_array_equal(m.idx_db[order_m], ref.idx_db[order_ref])
```

Add `load_localization_db` to the file's top import block (from
`collab_splats.localization.localizer`) if Task 4's tests did not already.

- [ ] **Step 2: Run on the A40 (serial — no concurrent GPU jobs)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -v -p no:randomly -k "real_"`
Expected: 3 PASS. **If any fails, STOP — do not weaken the assertions.** For loma: diff the
halves tensor-by-tensor (extract parity first; then dtype — if `detect_and_describe`
returns bf16 tensors, the float32 store must be shown equivalent under the matcher's
autocast, and if it is not, record the dtype in zarr attrs and cast on load). For xfeat:
read the vismatch xfeat wrapper's match stage — if it thresholds cossim or is not plain
mutual-NN, remove "xfeat" from `FEATURE_MATCH_MODELS` (Task 1's non-listed tests then
cover it) and record why in the spec.

- [ ] **Step 3: Commit**

```bash
git add tests/localization/test_local_matcher.py
git commit --only tests/localization/test_local_matcher.py -m "test(localization): GPU parity gates licensing FEATURE_MATCH_MODELS"
```

---

### Task 7: Phase-timing instrumentation (the residual-attribution work)

The ~658 s residual has never been measured directly. Instrument `verify_reconstruction`'s
phases into `summary["phase_seconds"]` (lands in verification.json automatically) and log
the reconstructor-side cold-start phases.

**Files:**
- Modify: `collab_splats/geometry/verification.py`
- Modify: `collab_splats/wrapper/reconstructor.py` (`verify()`)
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
    assert set(phases) == {"db_export", "pair_matching", "db_match_writes", "verify_matches", "triangulate"}
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
- Set `summary["phase_seconds"]` BEFORE calling `_write_report` so it lands in the JSON:

```python
    summary["phase_seconds"] = {k: round(v, 2) for k, v in timings.items()}
    result = VerificationResult(
        reconstruction=verified, pair_stats=pair_stats, frame_stats=frame_stats, summary=summary
    )
    _write_report(result, output_dir / "verification.json")
    logger.info("Verification phase seconds: %s", summary["phase_seconds"])
    return result
```

(No `report` key: report writing is sub-second JSON serialization — not a residual
candidate, and timing your own report write needs a self-referential trick that buys
nothing. Five phases cover the measured residual.)

`reconstructor.py` `verify()`: add `import time` to the stdlib import block if it is not
already there. Wrap the cold-start phases with
`logger.info("verify(): %s took %.1f s", name, dt)` lines: `build_localization_db()`
(including the rebuild-once branch), `load_localization_db(...)`, and the
`LocalMatcher(extractor_name)` construction (model load + index-stability probe — the cold
start the residual analysis needs).

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

### Task 8: CLAUDE.md stale architecture line

**Files:**
- Modify: `CLAUDE.md` (architecture tree, `localization/` block)

- [ ] **Step 1: Fix the line**

Replace:
```
    extractors.py          # Stage 2: BaseLocalExtractor, Disk/XFeat/Loma/LomaG local matchers
```
with:
```
    extractors.py          # Stage 2: LocalMatcher over the vismatch model zoo (FEATURE_MATCH_MODELS fast paths)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit --only CLAUDE.md -m "docs: fix stale extractors.py line — legacy matcher classes are gone"
```

---

### Task 9: Evidence runs (tmux, serial, human-gated compute)

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
    lm.match(feats[i], feats[j])
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
`keypoints_normalized` and the run exercises the zero-extraction feature-level path. (The
rebuild-once branch is exercised anyway if the existing cache predates the payload.)

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
