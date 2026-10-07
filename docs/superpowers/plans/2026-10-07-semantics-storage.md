# Semantics Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut semantics storage from ~20 GB to under 1 GB per scene and store OCR-lens word probabilities so the viewer never decodes them.

**Architecture:** The semantics stage extracts full-width states to a temporary store, trains the AE on every frame by streaming, encodes each frame to fp16 codes in `<extractor>_codes.zarr`, and deletes the states. Codes lift onto points as before. For ocr_lens, each frame's codes decode to word probabilities, which lift onto the points and the mesh vertices; only the top-64 per target are stored. `lift_features` samples visible points only.

**Tech Stack:** torch, zarr v3, open3d (mesh read), viser viewer, pytest.

**Spec:** [2026-10-07-semantics-storage-design.md](../specs/2026-10-07-semantics-storage-design.md)

---

## Ground rules for every task

- Work in the worktree `.worktrees/semantics-storage` on branch `feat/semantics-storage` (Task 0).
- Run Python only as `cd <wt> && PYTHONPATH=<wt> /opt/venv/reconstruction/bin/python ...`, or you are testing the main checkout.
  - Any test step that prints a pass count first prints `collab_splats.__file__` once per session, to prove the path.
- Do not pipe pytest through `| tail`: it hides the exit code. Write to a log with `> log 2>&1; echo exit=$?`.
- Commit with `git commit --only <paths>`. Use `git add -f` for `docs/superpowers/**`. End each message with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Code style is CLAUDE.md's:
  - every logical block gets a one-line `#` comment, with a blank line above it
  - docstrings: `"""` on their own lines, a one-line summary, `- ` bullets, then `Args:`/`Returns:`/`Raises:` for public functions
  - blank line around every `for`/`if`/`with`/`try`
  - absolute `collab_splats.` imports
- Never run repo-wide `black`. Run `black` and `isort` on the touched files only.

## File map

| file | change |
|---|---|
| `collab_splats/semantics/lifting.py` | `lift_features`: sample and `index_add_` visible points only; in-place weight and mean |
| `collab_splats/semantics/store.py` | `valid_feature_cache(store_path, name, images_dir, extractor_kwargs, latent_dim)`; `extract_feature_cache` → `write_feature_cache(store_path, maps, n_frames, attrs, ae=None)`; `write_point_features` fp16, `codes` optional, `arrays`/`attrs` |
| `collab_splats/semantics/__init__.py` | export rename |
| `collab_splats/semantics/compression.py` | `FeatureAutoencoder.fit` streams any (N, D, ...) array |
| `collab_splats/utils/torch_utils.py` | delete `load_features` |
| `collab_splats/reconstructor.py` | `STAGES` order; `semantics()` flow; new `_word_frame` |
| `docs/examples/ocr_lens_viewer.py` | read the vertex store; no decoder |
| `docs/semantics.md`, `configs/base.yaml`, `CLAUDE.md` | docs |
| `tests/semantics/test_lifting.py`, `test_store.py`, `test_compression_target.py`, `features/test_extract_from_zarr.py` | tests |
| `tests/reconstructor/test_reconstructor.py`, `test_sfm_stage.py`, `tests/utils/test_torch_utils.py` | tests |

One deviation from the spec's signature table: `write_feature_cache` takes `ae=None` and saves `autoencoder.pt` before the validity attrs. Without it, a crash between writing the attrs and saving the AE leaves a "valid" codes store with no decoder. This mirrors `write_point_features(store_path, codes, ae)`.

`PUSH_EXCLUDES` already holds `/semantics/**`, which keeps the scene-level codes store local. No change is needed.

---

### Task 0: Worktree

**Files:** none

- [ ] **Step 1: Create the worktree off `clean/final`**

```bash
cd /workspace/collab-splats
git worktree add .worktrees/semantics-storage -b feat/semantics-storage clean/final
cd .worktrees/semantics-storage
for d in /workspace/collab-splats/third_party/*/; do ln -sfn "$d" "third_party/$(basename "$d")"; done
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
```

Expected: a path under `.worktrees/semantics-storage/collab_splats/`.

- [ ] **Step 2: Baseline the gate**

```bash
cd /workspace/collab-splats/.worktrees/semantics-storage
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/reconstructor tests/utils tests/test_docstring_contract.py tests/test_import_style.py -q -p no:cacheprovider > /tmp/claude-0/ss_baseline.log 2>&1; echo exit=$?
grep -E "passed|failed" /tmp/claude-0/ss_baseline.log | tail -1
```

Record the pass/fail counts. Any failure that is listed in `docs/known-test-failures.md` is the baseline, not a regression.

---

### Task 1: `lift_features` samples visible points only

**Files:**
- Modify: `collab_splats/semantics/lifting.py` (the per-frame accumulate and the mean, in `lift_features`)
- Test: `tests/semantics/test_lifting.py`

- [ ] **Step 1: Write the equality test against a reference copy of today's loop**

Append to `tests/semantics/test_lifting.py`. Add `from collab_splats.geometry.projection import depth_residual` and `from collab_splats.semantics.lifting import _grid_sample_at_pixels` to the imports (keep isort order).

```python
def _lift_reference(maps, result, depth_tol=0.05):
    """
    Today's lift_features loop: every point sampled and weighted, no fallback.
    """
    H, W = result.model_height, result.model_width
    pts = torch.as_tensor(result.points, dtype=torch.float32)
    ext = torch.as_tensor(result.extrinsics, dtype=torch.float32)
    intr = torch.as_tensor(result.model_intrinsics, dtype=torch.float32)
    depth = torch.as_tensor(result.depth, dtype=torch.float32)
    conf = result.confidence.float()
    total = torch.zeros(len(pts), maps[0].shape[0])
    weights = torch.zeros(len(pts))

    for i, fmap in enumerate(maps):
        residual, expected, _, valid, pixels = depth_residual(pts, ext[i], intr[i], depth[i])
        ok = valid & (residual.abs() / (expected.abs() + 1e-8) < depth_tol)
        u = torch.nan_to_num(pixels[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        v = torch.nan_to_num(pixels[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
        ui = torch.round(u).clamp(0, W - 1).long()
        vi = torch.round(v).clamp(0, H - 1).long()
        w = conf[i, vi, ui] * ok.float()
        total += _grid_sample_at_pixels(fmap.float(), v, u, (H, W)) * w.unsqueeze(-1)
        weights += w

    return total / (weights.unsqueeze(-1) + 1e-8)


def test_lift_features_matches_the_all_points_reference(monkeypatch):
    """
    Sampling only visible points changes nothing: invisible points carried weight 0.

    - frame 2's depth map is far behind every point, so no point is visible there
    """
    monkeypatch.setattr("collab_splats.semantics.lifting.get_device", lambda: torch.device("cpu"))
    rng = np.random.default_rng(0)
    n, h, w, p = 3, 16, 20, 200
    pts = np.stack(
        [rng.uniform(-0.1, 0.1, p), rng.uniform(-0.08, 0.08, p), rng.uniform(1.0, 2.0, p)], axis=1
    ).astype(np.float32)
    depth = rng.uniform(1.0, 2.0, (n, h, w)).astype(np.float32)
    depth[2] = 100.0
    conf = rng.uniform(0.1, 1.0, (n, h, w)).astype(np.float32)
    result = _make_lift_result(pts, None, depth, conf, n=n, h=h, w=w)
    maps = [torch.from_numpy(rng.standard_normal((5, 4, 5)).astype(np.float32)) for _ in range(n)]

    expected = _lift_reference(maps, result)
    out = lift_features(maps.__getitem__, result)

    # Partial visibility: some points observed, some not
    observed = expected.abs().sum(1) > 0
    assert 0 < int(observed.sum()) < p
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
```

- [ ] **Step 2: Run it; it passes on today's code (it pins today's output)**

```bash
cd /workspace/collab-splats/.worktrees/semantics-storage
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_lifting.py -q -p no:cacheprovider > /tmp/claude-0/ss_t1.log 2>&1; echo exit=$?
```

Expected: `exit=0`. If the "partial visibility" assert fails, widen `depth`'s uniform range so that some points fail the 5% depth test, then rerun. This test is the refactor's safety net, so it must pass before any change.

- [ ] **Step 3: Change the accumulate and the mean**

In `lift_features`, replace this block:

```python
        visible = valid & depth_ok
        w = conf[i, v_idx, u_idx] * visible.float()  # (P,)

        # Bilinear sample feature map at projected coords via the shared grid_sample helper
        sampled = _grid_sample_at_pixels(fmap, v_safe, u_safe, image_size)  # (P, D)

        features_sum += sampled * w.unsqueeze(-1)
        weights_sum += w

    # Weighted mean with eps for numerical safety
    features = features_sum / (weights_sum.unsqueeze(-1) + 1e-8)
```

with:

```python
        visible = valid & depth_ok
        w = conf[i, v_idx, u_idx] * visible.float()  # (P,)

        # Sample and accumulate visible points only; the rest carry weight 0
        idx = torch.nonzero(w > 0).squeeze(1)
        sampled = _grid_sample_at_pixels(fmap, v_safe[idx], u_safe[idx], image_size)  # (P_vis, D)
        sampled *= w[idx].unsqueeze(-1)
        features_sum.index_add_(0, idx, sampled)
        weights_sum += w

    # Weighted mean in place, eps for numerical safety
    features = features_sum
    features /= weights_sum.unsqueeze(-1) + 1e-8
```

Add a bullet to the docstring: `- only visible points are sampled; each frame costs its visible count, not P`.

- [ ] **Step 4: Run the lifting tests**

Run the same command as Step 2. Expected: `exit=0`, with every `test_lifting.py` test passing.

- [ ] **Step 5: Commit**

```bash
git add tests/semantics/test_lifting.py collab_splats/semantics/lifting.py
git commit --only tests/semantics/test_lifting.py collab_splats/semantics/lifting.py -m "perf(semantics): lift_features samples visible points only

index_add_ over visible points; in-place weight and mean. Same output:
invisible points carried weight 0. GH010229 word lift 264 s / 21.5 GiB
-> 74 s / 12.2 GiB.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `write_feature_cache` and `valid_feature_cache(latent_dim)`

**Files:**
- Modify: `collab_splats/semantics/store.py` (`valid_feature_cache`; `extract_feature_cache` replaced)
- Modify: `collab_splats/semantics/__init__.py:33-55` (export rename)
- Test: `tests/semantics/test_store.py` (the 2D cache section)
- Test: `tests/semantics/features/test_extract_from_zarr.py` (its `extract_feature_cache` section)

- [ ] **Step 1: Replace the 2D-cache tests**

In `tests/semantics/test_store.py`:
- Delete every `test_extract_feature_cache_*` and `test_valid_feature_cache_*` test, and the `_one_frame_scene` and `_fake_extractor` helpers. These tests drove the extractor through the store; the store no longer sees an extractor.
- In their place, under the same section header, add:

```python
def _frames_dir(tmp_path: Path, n: int) -> Path:
    images = tmp_path / "images"
    images.mkdir()

    for i in range(n):
        (images / f"frame_{i:06d}.png").touch()

    return images


def _maps(n: int, dim: int = 3) -> list[torch.Tensor]:
    return [torch.full((dim, 2, 2), float(i)) for i in range(n)]


ATTRS = {"extractor": "fake", "patch_size": 2, "n_frames": 2, "extractor_kwargs": {"layer": 17}, "latent_dim": 3}


def test_write_feature_cache_writes_fp16_one_chunk_per_frame(tmp_path):
    path = tmp_path / "fake_codes.zarr"
    store.write_feature_cache(path, iter(_maps(2)), 2, ATTRS)

    arr = zarr.open(str(path), mode="r")["features"]
    assert arr.dtype == np.float16
    assert arr.shape == (2, 3, 2, 2) and arr.chunks == (1, 3, 2, 2)
    assert float(arr[1, 0, 0, 0]) == 1.0
    assert dict(zarr.open(str(path), mode="r").attrs) == ATTRS


def test_write_feature_cache_saves_the_autoencoder_before_the_attrs(tmp_path, monkeypatch):
    """A store whose AE never landed must read invalid."""
    path = tmp_path / "fake_codes.zarr"

    def boom(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(FeatureAutoencoder, "save", boom)

    with pytest.raises(OSError):
        store.write_feature_cache(path, iter(_maps(2)), 2, ATTRS, ae=FeatureAutoencoder(8, 3))

    assert "extractor" not in zarr.open(str(path), mode="r").attrs


def test_write_feature_cache_crash_mid_frames_reads_invalid(tmp_path):
    images = _frames_dir(tmp_path, 2)
    path = tmp_path / "fake_codes.zarr"

    def crashing():
        yield _maps(1)[0]
        raise RuntimeError("crash mid-extraction")

    with pytest.raises(RuntimeError):
        store.write_feature_cache(path, crashing(), 2, ATTRS)

    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, 3) is None


def test_write_feature_cache_rejects_a_short_frame_stream(tmp_path):
    with pytest.raises(ValueError, match="1 of 2"):
        store.write_feature_cache(tmp_path / "fake_codes.zarr", iter(_maps(1)), 2, ATTRS)


def test_valid_feature_cache_checks_name_kwargs_frames_and_latent_dim(tmp_path):
    images = _frames_dir(tmp_path, 2)
    path = tmp_path / "fake_codes.zarr"
    store.write_feature_cache(path, iter(_maps(2)), 2, ATTRS)

    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, 3) == path
    assert store.valid_feature_cache(path, "other", images, {"layer": 17}, 3) is None
    assert store.valid_feature_cache(path, "fake", images, {"layer": 18}, 3) is None
    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, 8) is None
    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, None) is None


def test_valid_feature_cache_rejects_a_frame_count_mismatch(tmp_path):
    images = _frames_dir(tmp_path, 3)
    path = tmp_path / "fake_codes.zarr"
    store.write_feature_cache(path, iter(_maps(2)), 2, ATTRS)

    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, 3) is None


def test_valid_feature_cache_propagates_unexpected_errors(tmp_path, monkeypatch):
    """A bug inside the validity check must surface, not read as a miss."""
    images = _frames_dir(tmp_path, 1)
    (tmp_path / "fake_codes.zarr").mkdir()

    def boom(*args, **kwargs):
        raise RuntimeError("not a store error")

    monkeypatch.setattr(store.zarr, "open", boom)

    with pytest.raises(RuntimeError, match="not a store error"):
        store.valid_feature_cache(tmp_path / "fake_codes.zarr", "fake", images, {}, None)
```

Before deleting `test_extract_feature_cache_propagates_unexpected_errors`, check that `open_valid` really calls `zarr.open` through `store.zarr`. If it imports zarr in `utils/io.py`, patch `collab_splats.utils.io.zarr.open` instead. The test's intent stays the same.

In `tests/semantics/features/test_extract_from_zarr.py`, delete the `# extract_feature_cache` section and its four tests, and remove the `extract_feature_cache` import. The `features_to_rgb` tests stay. Update the module docstring's first line to `Tests for BaseFeatureExtractor.features_to_rgb.`

- [ ] **Step 2: Run them; they fail**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_store.py -q -p no:cacheprovider > /tmp/claude-0/ss_t2.log 2>&1; echo exit=$?
grep -m3 "Error" /tmp/claude-0/ss_t2.log
```

Expected: `exit=1`, with `AttributeError: module 'collab_splats.semantics.store' has no attribute 'write_feature_cache'`.

- [ ] **Step 3: Implement**

In `collab_splats/semantics/store.py`:
- Imports: drop `IMAGE_EXTS`, `BaseFeatureExtractor`, `read_image` and `batch_iterator`. Add `from collections.abc import Iterable`. Keep `frame_paths` (validity counts frames).
- `__all__`: replace `"extract_feature_cache"` with `"write_feature_cache"`.
- Section header: `########## 2D feature cache (<extractor>_codes.zarr) ###`.
- Replace `valid_feature_cache` and `extract_feature_cache` with:

```python
def valid_feature_cache(
    store_path: Path,
    name: str,
    images_dir: Path,
    extractor_kwargs: Optional[dict[str, Any]],
    latent_dim: Optional[int],
) -> Optional[Path]:
    """
    The 2D feature cache at store_path when reusable, else None.

    - valid: extractor name, frame count, kwargs and code width match the store attrs
    - attrs are written last, so a crashed write reads invalid

    Args:
        store_path: the `<extractor>_codes.zarr` path.
        name: extractor registry name.
        images_dir: scene images/ directory.
        extractor_kwargs: constructor kwargs the cache must match.
        latent_dim: AE code width the cache must hold; None for uncompressed features.

    Returns:
        store_path, or None.
    """
    n_frames = len(frame_paths(images_dir))
    expected = {
        "extractor": name,
        "n_frames": n_frames,
        "extractor_kwargs": extractor_kwargs or {},
        "latent_dim": latent_dim,
    }

    if open_valid(store_path, expected) is None:
        return None

    return Path(store_path)


def write_feature_cache(
    store_path: Path,
    maps: Iterable[torch.Tensor],
    n_frames: int,
    attrs: dict[str, Any],
    ae: Optional[FeatureAutoencoder] = None,
) -> None:
    """
    Write per-frame (D, H_p, W_p) maps to a zarr store as fp16, one chunk per frame.

    - serves extraction (full-width states) and encoding (AE codes)
    - ae saved as `autoencoder.pt` before the attrs; attrs last, so a crash reads invalid
    - overwrites whatever is at store_path

    Args:
        store_path: the store to write.
        maps: frame maps in store-row order, any float dtype and device.
        n_frames: number of maps expected.
        attrs: store attrs, written last.
        ae: autoencoder that decodes the maps, saved inside the store; None for none.

    Raises:
        ValueError: when maps yields other than n_frames maps.
    """
    store = zarr.open(str(store_path), mode="w")
    arr = None
    n_written = 0

    for fmap in maps:
        # First map fixes (D, H_p, W_p); one chunk per frame, so reading frame i loads 1 chunk
        if arr is None:
            arr = store.create_array(
                "features",
                shape=(n_frames, *fmap.shape),
                chunks=(1, *fmap.shape),
                dtype="float16",
                fill_value=0,
            )

        fmap = fmap.detach().cpu()
        arr[n_written] = fmap.half().numpy()
        n_written += 1

    if n_written != n_frames:
        raise ValueError(f"write_feature_cache: got {n_written} of {n_frames} frame maps for {store_path}")

    if ae is not None:
        ae.save(Path(store_path) / "autoencoder.pt")

    # Validity attrs last
    store.attrs.update(to_json_safe(attrs))
    logger.info("feature cache written: %s  shape=%s", store_path, tuple(arr.shape))
```

`n_written` stays at 0 when `maps` is empty, so the `ValueError` fires before `arr.shape` is read.

In `collab_splats/semantics/__init__.py`, replace `extract_feature_cache` with `write_feature_cache` in both the import and `__all__`, keeping alphabetical order.

- [ ] **Step 4: Run the store tests**

Run the same command as Step 2. Expected: the 2D-cache tests pass. The lifted-store tests are unchanged and still pass. `collab_splats.reconstructor` now fails to import (it still names `extract_feature_cache`), but Task 5 fixes that and no store test imports it.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/store.py collab_splats/semantics/__init__.py tests/semantics/test_store.py tests/semantics/features/test_extract_from_zarr.py
git commit --only collab_splats/semantics/store.py collab_splats/semantics/__init__.py tests/semantics/test_store.py tests/semantics/features/test_extract_from_zarr.py -m "refactor(semantics): write_feature_cache writes any per-frame maps; validity keys latent_dim

extract_feature_cache -> write_feature_cache(store_path, maps, n_frames,
attrs, ae): serves extraction and AE encoding; the caller checks validity.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `write_point_features` fp16, optional codes, word arrays

**Files:**
- Modify: `collab_splats/semantics/store.py` (`write_point_features`)
- Test: `tests/semantics/test_store.py` (the lifted-store section)

- [ ] **Step 1: Write the failing tests**

Append to the lifted-store section of `tests/semantics/test_store.py`:

```python
def test_write_point_features_stores_fp16_and_reads_float32_unit_rows(tmp_path):
    store_path = tmp_path / "talk2dino_lifted.zarr"
    feats = np.random.default_rng(0).random((6, 5), dtype=np.float32)
    write_point_features(store_path, feats, None)

    assert zarr.open(str(store_path), mode="r")["features"].dtype == np.float16
    out = read_point_features(store_path)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_write_point_features_lands_arrays_and_attrs_in_the_same_store(tmp_path):
    store_path = tmp_path / "ocr_lens_lifted.zarr"
    feats = np.ones((4, 5), dtype=np.float32)
    ids = np.arange(8, dtype=np.uint16).reshape(4, 2)
    write_point_features(store_path, feats, None, arrays={"word_ids": ids}, attrs={"words": ["a", "b"]})

    store = zarr.open(str(store_path), mode="r")
    np.testing.assert_array_equal(store["word_ids"][:], ids)
    assert store["word_ids"].dtype == np.uint16
    assert store.attrs["words"] == ["a", "b"]
    assert store.attrs["latent_dim"] == 5


def test_write_point_features_without_codes_writes_arrays_only(tmp_path):
    store_path = tmp_path / "ocr_lens_vertices.zarr"
    write_point_features(store_path, None, None, arrays={"word_probs": np.zeros((3, 2), np.float16)})

    store = zarr.open(str(store_path), mode="r")
    assert "features" not in store
    assert "latent_dim" not in store.attrs
    assert store["word_probs"].shape == (3, 2)
```

- [ ] **Step 2: Run them; they fail**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_store.py -q -p no:cacheprovider > /tmp/claude-0/ss_t3.log 2>&1; echo exit=$?
```

Expected: `exit=1`. The fp16 test fails on dtype (`float32`), and the others fail with `TypeError: ... unexpected keyword argument 'arrays'` or on `None.shape`.

- [ ] **Step 3: Implement**

Replace `write_point_features` with:

```python
def write_point_features(
    store_path: Path,
    codes: Optional[np.ndarray],
    ae: Optional[FeatureAutoencoder],
    *,
    arrays: Optional[dict[str, np.ndarray]] = None,
    attrs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Write a lifted store atomically via a `.tmp` dir renamed into place.

    - codes stored fp16; read_point_features returns them float32
    - arrays (e.g. ocr_lens word_ids / word_probs / dropped_mass) land as given, same atomic write

    Args:
        store_path: the lifted store path.
        codes: (P, latent) codes, (P, D) when `ae` is None, or None for an arrays-only store.
        ae: autoencoder that decodes `codes`, or None.
        arrays: extra named per-row arrays, stored in their own dtype.
        attrs: extra store attrs.
    """
    store_path = Path(store_path)
    tmp = store_path.with_name(f"{store_path.name}.tmp")

    store_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # Codes, extra arrays, weights, then attrs into the tmp dir; a failure removes it
    try:
        store = zarr.open(str(tmp), mode="w")
        store_attrs = dict(attrs or {})

        if codes is not None:
            codes = np.asarray(codes)
            store["features"] = codes.astype(np.float16)
            width = int(codes.shape[1])
            store_attrs["input_dim"] = int(ae.input_dim) if ae is not None else width
            store_attrs["latent_dim"] = width

        for name, array in (arrays or {}).items():
            store[name] = array

        if ae is not None:
            ae.save(tmp / "autoencoder.pt")

        store.attrs.update(to_json_safe(store_attrs))

    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Swap the finished store in under its real name
    shutil.rmtree(store_path, ignore_errors=True)
    tmp.rename(store_path)
```

Check `read_point_features`: it must cast to float32 before normalizing (the spec says it already does). If it normalizes the raw fp16 array, insert `.astype(np.float32)` at the read.

- [ ] **Step 4: Run the store tests**

Run the same command as Step 2. Expected: `exit=0`. `test_write_point_features_puts_the_autoencoder_inside_the_store` still sees attrs `{"input_dim": 32, "latent_dim": 8}`. The `assert_allclose` round-trip tests compare fp16-stored values at `rtol=1e-6`; loosen those to `rtol=1e-3` (fp16 has an 11-bit mantissa), because the precision change is intended.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/store.py tests/semantics/test_store.py
git commit --only collab_splats/semantics/store.py tests/semantics/test_store.py -m "feat(semantics): lifted stores hold fp16 codes plus optional named arrays

write_point_features: codes cast to fp16 and optional (vertex store);
arrays/attrs ride the same atomic write.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Streaming `FeatureAutoencoder.fit`; delete `load_features`

**Files:**
- Modify: `collab_splats/semantics/compression.py` (`fit`, imports)
- Modify: `collab_splats/utils/torch_utils.py` (delete `load_features`; drop `math`/`zarr` imports if nothing else uses them)
- Test: `tests/semantics/test_compression_target.py`
- Test: `tests/utils/test_torch_utils.py` (delete the `load_features` section and its import; delete `_as_zarr` if only those tests used it)

- [ ] **Step 1: Write the failing tests**

Append to `tests/semantics/test_compression_target.py`. Add `import numpy as np` and `import zarr` to its imports.

```python
def _patch_maps(n=6, d=16, h=2, w=3):
    """
    (n, d, h, w) maps whose frame k holds only feature k: a fit that skips a frame never sees it.
    """
    maps = np.zeros((n, d, h, w), dtype=np.float32)

    for k in range(n):
        maps[k, k] = 1.0 + k

    return maps


def test_fit_streams_every_frame_of_a_zarr_in_blocks(tmp_path, monkeypatch):
    """read_gb small enough for 1-frame blocks: every frame still reaches the loss."""
    maps = _patch_maps()
    arr = zarr.open(str(tmp_path / "m.zarr"), mode="w", shape=maps.shape, chunks=(1, *maps.shape[1:]), dtype="float16")
    arr[:] = maps
    seen = []
    ae = FeatureAutoencoder(input_dim=16, latent_dim=4)
    hook = ae.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().cpu()))
    ae.fit(arr, epochs=1, batch_size=4, read_gb=1e-9)
    hook.remove()

    rows = torch.cat(seen)
    assert rows.shape == (6 * 2 * 3, 16)
    assert set(rows.argmax(1).tolist()) == set(range(6))


def test_fit_tensor_and_zarr_inputs_both_train(tmp_path):
    maps = _patch_maps()
    arr = zarr.open(str(tmp_path / "m.zarr"), mode="w", shape=maps.shape, dtype="float32")
    arr[:] = maps

    for source in (torch.from_numpy(maps), arr):
        ae = FeatureAutoencoder(input_dim=16, latent_dim=4)
        ae.fit(source, epochs=2, read_gb=1e-9)
        assert ae.epochs_run == 2
        assert -1.0 <= ae.recon_cosine <= 1.0


def test_fit_rejects_empty_zarr(tmp_path):
    arr = zarr.open(str(tmp_path / "m.zarr"), mode="w", shape=(0, 16, 2, 2), dtype="float16")

    with pytest.raises(ValueError, match="at least one sample"):
        FeatureAutoencoder(input_dim=16, latent_dim=4).fit(arr, epochs=1)
```

- [ ] **Step 2: Run them; they fail**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_compression_target.py tests/semantics/test_compression.py -q -p no:cacheprovider > /tmp/claude-0/ss_t4.log 2>&1; echo exit=$?
```

Expected: `exit=1`, with `TypeError: fit() got an unexpected keyword argument 'read_gb'`, and for the zarr case `AttributeError: ... 'device'`.

- [ ] **Step 3: Implement**

In `collab_splats/semantics/compression.py`:
- Add `import math` (stdlib group) and `import numpy as np` and `import zarr` (third-party group).
- Change the `torch_utils` import to `from collab_splats.utils.torch_utils import batch_iterator, get_device`.

Replace `fit` with:

```python
    def fit(
        self,
        features: Tensor | np.ndarray | zarr.Array,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        on_epoch: Optional[Callable[[int, int, float], None]] = None,
        target_cosine: Optional[float] = None,
        read_gb: float = 1.0,
    ) -> None:
        """
        Train in place on MSE(recon, x) + (1 - cosine(recon, x)), streaming axis-0 blocks.

        - axis 1 = features; axis 0 + trailing axes = samples ((N, D) rows or (N, D, H, W) patch maps)
        - each epoch reads every item: blocks of about read_gb in random order, rows shuffled per block
        - a zarr never sits in memory whole; a tensor trains on its own device, others on get_device()
        - fit quality lands on `self` as recon_cosine, recon_mse, epochs_run (training set, optimistic)

        Args:
            features: tensor, ndarray or zarr array, feature width on axis 1.
            epochs: epoch ceiling.
            batch_size: mini-batch size.
            lr: Adam learning rate.
            on_epoch: callback(epoch, epochs, avg_loss), once per epoch; a progress hook for UIs.
            target_cosine: stop once mean reconstruction cosine reaches this; None runs all epochs.
            read_gb: float32 size of one axis-0 block, in GiB.

        Raises:
            ValueError: if `features` has no samples.
        """
        n_items, dim = features.shape[:2]
        per_item = math.prod(features.shape[2:])

        # Empty input raises a clear ValueError, not a ZeroDivisionError at the epoch mean
        if n_items * per_item == 0:
            raise ValueError(f"fit() requires at least one sample; got features with shape {tuple(features.shape)}")

        device = features.device if isinstance(features, Tensor) else get_device()
        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Axis-0 blocks of about read_gb float32
        item_gb = per_item * dim * 4 / 2**30
        items_per_block = max(1, math.floor(read_gb / item_gb))
        starts = list(range(0, n_items, items_per_block))

        pbar = tqdm(range(epochs), desc="fit autoencoder", unit="epoch")

        for epoch in pbar:
            epoch_loss = 0.0
            epoch_cos = 0.0
            epoch_mse = 0.0
            n_batches = 0

            for b in torch.randperm(len(starts)).tolist():
                # Read one block as (rows, dim) float32 on device
                start = starts[b]
                block = features[start : start + items_per_block]

                # zarr and ndarray slices arrive as numpy; tensors stay on their device
                if not isinstance(block, Tensor):
                    block = torch.from_numpy(np.asarray(block))

                block = block.to(device=device, dtype=torch.float32)
                block = block.reshape(len(block), dim, per_item)
                block = block.transpose(1, 2)
                block = block.reshape(-1, dim)

                # Shuffle rows within the block for unbiased mini-batches
                perm = torch.randperm(len(block), device=device)

                for first in range(0, len(block), batch_size):
                    x = block[perm[first : first + batch_size]]

                    recon = self.decoder_out(self.decoder_hidden(self.encoder(x)))
                    mse = F.mse_loss(recon, x)
                    cos = F.cosine_similarity(recon, x).mean()
                    loss = mse + (1 - cos)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_cos += cos.item()
                    epoch_mse += mse.item()
                    n_batches += 1

            # At least one sample (checked above), so n_batches >= 1
            avg_loss = epoch_loss / n_batches
            self.recon_cosine = epoch_cos / n_batches
            self.recon_mse = epoch_mse / n_batches
            self.epochs_run = epoch + 1

            pbar.set_postfix(loss=f"{avg_loss:.6f}", cos=f"{self.recon_cosine:.4f}")
            logger.debug("epoch %d/%d  loss=%.6f  cos=%.4f", epoch + 1, epochs, avg_loss, self.recon_cosine)

            if on_epoch is not None:
                on_epoch(epoch + 1, epochs, avg_loss)

            # Early stop once reconstruction is good enough; epochs is the ceiling
            if target_cosine is not None and self.recon_cosine >= target_cosine:
                logger.info(
                    "target cosine %.4f reached at epoch %d/%d (cos=%.4f) — stopping",
                    target_cosine,
                    epoch + 1,
                    epochs,
                    self.recon_cosine,
                )
                break

        self.eval()
```

Update the module docstring's first bullet: "trained once on every frame's patch features, streamed in blocks".

In `collab_splats/utils/torch_utils.py`, delete `load_features` whole. Remove `math` and `zarr` from its imports only when `grep -n "math\.\|zarr\." collab_splats/utils/torch_utils.py` shows no other use. In `tests/utils/test_torch_utils.py`, delete the `########## load_features` section, `load_features` from the import, and `_as_zarr` if nothing else calls it.

- [ ] **Step 4: Run the compression and utils tests**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_compression_target.py tests/semantics/test_compression.py tests/utils/test_torch_utils.py -q -p no:cacheprovider > /tmp/claude-0/ss_t4.log 2>&1; echo exit=$?
grep -rn "load_features" collab_splats tests evals docs --include=*.py | grep -v "eager_load_features"
```

Expected: `exit=0`. The grep shows only `collab_splats/reconstructor.py` (fixed in Task 5). `test_target_cosine_exactly_met_stops` compares two identically seeded fits, so the extra `randperm` over blocks does not break it.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/compression.py collab_splats/utils/torch_utils.py tests/semantics/test_compression_target.py tests/utils/test_torch_utils.py
git commit --only collab_splats/semantics/compression.py collab_splats/utils/torch_utils.py tests/semantics/test_compression_target.py tests/utils/test_torch_utils.py -m "feat(semantics): FeatureAutoencoder.fit streams every frame; drop load_features

fit reads axis-0 blocks of read_gb from a tensor, ndarray or zarr,
shuffling blocks and rows within a block each epoch, so the AE trains on
all frames instead of an 8 GiB strided sample.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Semantics stage: extract → fit → encode → delete → lift; mesh before semantics

**Files:**
- Modify: `collab_splats/reconstructor.py`:
  - imports (lines ~16-75)
  - `STAGES` (~line 86)
  - `semantics_cache_dir` docstring (~line 419)
  - `semantics()` (~lines 841-901)
- Test: `tests/reconstructor/test_reconstructor.py` (semantics tests, lines ~400-575 and ~1320-1345)
- Test: `tests/reconstructor/test_sfm_stage.py` (`test_semantics_lifts_only_the_rows_the_pointcloud_holds`)

- [ ] **Step 1: Rewrite the stage tests**

In `tests/reconstructor/test_reconstructor.py`:

(a) Add a stage-order test near the other `STAGES` tests (~line 943):

```python
def test_mesh_runs_before_semantics():
    """Semantics lifts word probabilities onto mesh.ply's vertices, so mesh comes first."""
    order = list(STAGES)
    assert order.index("mesh") < order.index("semantics")
```

(b) Add a codes-store seeding helper beside `_touch_frames`:

```python
def _seed_codes(rec, name, maps, latent_dim=None, extractor_kwargs=None):
    """A valid <name>_codes.zarr over rec's images/, as a finished extract + encode leaves it."""
    path = rec.semantics_cache_dir / f"{name}_codes.zarr"
    attrs = {
        "extractor": name,
        "patch_size": 14,
        "n_frames": len(maps),
        "extractor_kwargs": extractor_kwargs or {},
        "latent_dim": latent_dim,
    }
    write_feature_cache(path, iter(torch.from_numpy(m) for m in maps), len(maps), attrs)
    return path
```

Add `from collab_splats.semantics.store import write_feature_cache` to the test imports.

(c) Replace `test_semantics_valid_cache_skips_the_extractor`:

```python
def test_semantics_valid_cache_skips_the_extractor(tmp_path):
    """A codes store valid for this extractor, kwargs and width is lifted without building the extractor."""
    config = _make_config(tmp_path, {"semantics": {"enabled": True, "extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    _touch_frames(rec, [0, 1])
    _seed_disk_reconstruction(rec, [0, 1])
    codes_path = _seed_codes(rec, "dinov2", [np.zeros((4, 2, 2), np.float16)] * 2)

    with (
        patch.object(R, "BaseFeatureExtractor") as extractor_base,
        patch.object(R, "write_feature_cache") as write,
        patch.object(R, "lift_features", return_value=torch.zeros(1, 4)),
        patch.object(R, "write_point_features"),
    ):
        rec.semantics()

    extractor_base.get.assert_not_called()
    write.assert_not_called()
    assert codes_path.exists()
```

(d) Replace `test_semantics_invalid_cache_extracts_with_extractor_kwargs`:

```python
def test_semantics_invalid_cache_extracts_with_extractor_kwargs(tmp_path):
    """semantics.extractor_kwargs reaches the extractor build and the codes store's attrs."""
    semantics = {"enabled": True, "extractor": "dinov2", "extractor_kwargs": {"layer": 20}, "n_components": None}
    rec = _semantics_rec(tmp_path, semantics)

    with _stub_extraction() as extractor_base:
        rec.semantics()

    extractor_base.get.assert_called_once_with("dinov2")
    extractor_base.get.return_value.assert_called_once_with(layer=20)
    attrs = dict(zarr.open(str(rec.semantics_cache_dir / "dinov2_codes.zarr"), mode="r").attrs)
    assert attrs["extractor_kwargs"] == {"layer": 20}
    assert attrs["latent_dim"] is None
```

(e) Replace `_run_semantics`, `test_semantics_writes_weights_inside_the_lifted_store`, `test_semantics_reuses_the_ae_stored_with_the_cache` and `test_semantics_refits_when_n_components_changes` with:

```python
def _semantics_rec(tmp_path, semantics):
    """A Reconstructor over two images/ frames and a one-point reconstruction."""
    rec = Reconstructor(_make_config(tmp_path, {"semantics": semantics}))
    _touch_frames(rec, [0, 1])
    _seed_disk_reconstruction(rec, [0, 1])
    return rec


@contextlib.contextmanager
def _stub_extraction(dim=32):
    """
    Extractor, image decode and lift stubbed; frame k's map is constant k + 1 over (dim, 2, 2).

    - lift returns frame 0's first cell repeated per point (6 points), so its width is the loader's
    """
    calls = []

    def forward(frames):
        first = len(calls)
        calls.extend(frames)
        return [torch.full((dim, 2, 2), float(first + j + 1)) for j in range(len(frames))]

    def lift(frame_features, result):
        return frame_features(0)[:, 0, 0].float().cpu().repeat(6, 1)

    with (
        patch.object(R, "BaseFeatureExtractor") as extractor_base,
        patch.object(R, "read_image", return_value=np.zeros((4, 4, 3), np.uint8)),
        patch.object(R, "lift_features", side_effect=lift),
    ):
        extractor = extractor_base.get.return_value.return_value
        extractor.forward.side_effect = forward
        extractor.patch_size = 14
        yield extractor_base


COMPRESSED = {"extractor": "dinov2", "n_components": 8, "max_epochs": 1, "target_cosine": None}


def test_semantics_encodes_every_frame_and_deletes_the_states(tmp_path):
    rec = _semantics_rec(tmp_path, COMPRESSED)

    with _stub_extraction():
        rec.semantics()

    codes_path = rec.semantics_cache_dir / "dinov2_codes.zarr"
    codes = zarr.open(str(codes_path), mode="r")["features"]
    assert codes.shape == (2, 8, 2, 2) and codes.dtype == np.float16
    assert (codes_path / "autoencoder.pt").is_file()
    assert not (rec.semantics_cache_dir / "dinov2.zarr").exists()


def test_semantics_writes_weights_inside_the_lifted_store(tmp_path):
    rec = _semantics_rec(tmp_path, COMPRESSED)

    with _stub_extraction():
        rec.semantics()

    out_dir = rec.backend_dir / "semantics"
    assert rec.done("semantics")
    assert (out_dir / "dinov2_lifted.zarr" / "autoencoder.pt").is_file()
    store = zarr.open(str(out_dir / "dinov2_lifted.zarr"), mode="r")
    assert store["features"].shape == (6, 8) and store["features"].dtype == np.float16
    assert dict(store.attrs) == {"input_dim": 32, "latent_dim": 8}
    assert "word_ids" not in store


def test_semantics_second_run_neither_extracts_nor_fits(tmp_path):
    rec = _semantics_rec(tmp_path, COMPRESSED)

    with _stub_extraction():
        rec.semantics()

    with _stub_extraction() as extractor_base, patch.object(FeatureAutoencoder, "fit") as fit:
        rec.semantics()

    extractor_base.get.assert_not_called()
    fit.assert_not_called()


def test_semantics_n_components_change_re_extracts(tmp_path):
    rec = _semantics_rec(tmp_path, COMPRESSED)

    with _stub_extraction():
        rec.semantics()

    rec.config["semantics"]["n_components"] = 4

    with _stub_extraction() as extractor_base:
        rec.semantics()

    extractor_base.get.assert_called_once()
    stored = FeatureAutoencoder.load(rec.semantics_cache_dir / "dinov2_codes.zarr" / "autoencoder.pt")
    assert stored.latent_dim == 4


def test_semantics_uncompressed_writes_full_width_codes_without_states(tmp_path, monkeypatch):
    rec = _semantics_rec(tmp_path, {"extractor": "dinov2", "n_components": None})
    written = []
    real_write = R.write_feature_cache
    monkeypatch.setattr(R, "write_feature_cache", lambda path, *a, **k: (written.append(path), real_write(path, *a, **k)))

    with _stub_extraction():
        rec.semantics()

    codes_path = rec.semantics_cache_dir / "dinov2_codes.zarr"
    assert written == [codes_path]
    assert zarr.open(str(codes_path), mode="r")["features"].shape == (2, 32, 2, 2)
    assert not (codes_path / "autoencoder.pt").exists()
```

Add `import contextlib` to the test imports.

(f) In `test_run_refuses_semantics_if_lifted_exists`, rename the patched name `extract_feature_cache` to `write_feature_cache`.

(g) Replace the cache setup and patches in `test_semantics_reads_pointcloud_zarr_from_disk`:

```python
    # Scene codes row r is constant r; the writer stubbed so only the pick runs
    _seed_codes(rec, "dinov2", [np.full((4, 2, 2), r, np.float16) for r in range(3)])

    with (
        patch.object(R, "BaseFeatureExtractor"),
        patch.object(R, "lift_features", return_value=torch.zeros(1, 4)) as lift,
        patch.object(R, "write_point_features"),
    ):
        rec.semantics()
```

(h) In `tests/reconstructor/test_sfm_stage.py`, `test_semantics_lifts_only_the_rows_the_pointcloud_holds`:
- seed the cache at `recon.semantics_cache_dir / "dinov2_codes.zarr"` via `write_feature_cache`, with attrs `{"extractor": "dinov2", "patch_size": 14, "n_frames": len(FRAME_IDX), "extractor_kwargs": {}, "latent_dim": None}`
- drop the `extract_feature_cache` patch

Check that `_seed_subset_scene` touches all `FRAME_IDX` frames in `images/` (validity counts them). If it does not, patch `valid_feature_cache` to return the seeded path instead.

- [ ] **Step 2: Run them; they fail**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor/test_reconstructor.py tests/reconstructor/test_sfm_stage.py -q -p no:cacheprovider -k "semantics or stage or STAGES or load_frame" > /tmp/claude-0/ss_t5.log 2>&1; echo exit=$?
```

Expected: `exit=1` (or a collection error): `ImportError: cannot import name 'extract_feature_cache'` from the reconstructor.

- [ ] **Step 3: Implement**

Imports in `collab_splats/reconstructor.py`:
- add `import itertools` (stdlib)
- store import becomes `valid_feature_cache, write_feature_cache, write_point_features`
- `from collab_splats.utils.io import read_image, to_json_safe, write_json`
- torch_utils import: drop `load_features`, add `batch_iterator`

`STAGES`:

```python
STAGES: dict[str, tuple[str, ...]] = {
    "preproc": (),
    "pointcloud": ("preproc",),
    "refine": ("pointcloud",),
    "splats": ("pointcloud",),
    "mesh": ("pointcloud",),
    "semantics": ("pointcloud",),
    "localize": ("pointcloud",),
    "reconstruction_quality_report": ("pointcloud",),
}
```

`semantics_cache_dir` docstring: "Scene-level 2D feature cache: `<extractor>_codes.zarr`, plus a temporary `<extractor>.zarr` of full-width states while the AE trains."

Replace `semantics()` (Task 6 adds the ocr_lens word arrays at the marked spot):

```python
    def semantics(self) -> None:
        """
        Extract 2D features, compress every frame to fp16 codes, lift the codes onto the points.

        - codes store `<extractor>_codes.zarr` reused when valid (name, frames, kwargs, latent_dim)
        - a miss extracts full-width states to a temporary `<extractor>.zarr`, trains the AE on all of
          them, encodes every frame, then deletes the states; n_components null keeps full width, no AE
        - lift reads one frame of codes at a time; rows follow the zarr's frames (maybe a subset)
        - lifted store written atomically with its own autoencoder.pt (pushed; the 2D cache is not)
        """
        cfg = self.config["semantics"]
        name = cfg["extractor"]
        extractor_kwargs = cfg["extractor_kwargs"]
        latent_dim = cfg["n_components"]
        codes_path = self.semantics_cache_dir / f"{name}_codes.zarr"
        states_path = self.semantics_cache_dir / f"{name}.zarr"
        self.semantics_cache_dir.mkdir(parents=True, exist_ok=True)

        # 2D codes for every images/ frame; the model loads only on a miss
        if valid_feature_cache(codes_path, name, self.images_dir, extractor_kwargs, latent_dim) is None:
            extractor = BaseFeatureExtractor.get(name)(**extractor_kwargs)
            paths = frames.frame_paths(self.images_dir)
            n_frames = len(paths)
            attrs = {
                "extractor": name,
                "patch_size": extractor.patch_size,
                "n_frames": n_frames,
                "extractor_kwargs": extractor_kwargs,
                "latent_dim": latent_dim,
            }

            # Extractor maps over every frame, decoded and run 4 at a time
            batches = (extractor.forward([read_image(p) for p in batch]) for (batch,) in batch_iterator(4, paths))
            maps = itertools.chain.from_iterable(batches)

            # Uncompressed: full-width maps are the codes; else they are temporary states
            target = codes_path if latent_dim is None else states_path

            with torch.no_grad():
                write_feature_cache(target, maps, n_frames, attrs if latent_dim is None else {})

            # Free the extractor's GPU memory now; a forward hook can hold it in a reference cycle
            del extractor
            pytorch_gc()

            # Train the AE on every frame's states, encode each frame into the codes store, drop the states
            if latent_dim is not None:
                states = zarr.open(str(states_path), mode="r")["features"]
                ae = FeatureAutoencoder(input_dim=states.shape[1], latent_dim=latent_dim)
                ae.fit(states, epochs=cfg["max_epochs"], target_cosine=cfg["target_cosine"])
                ae.to(get_device())
                encoded = map(partial(_load_frame, states, range(n_frames), ae), range(n_frames))
                write_feature_cache(codes_path, encoded, n_frames, attrs, ae=ae)
                shutil.rmtree(states_path)

        # Codes read lazily; the AE only rides along into the lifted store
        codes = zarr.open(str(codes_path), mode="r")["features"]
        ae = None

        if latent_dim is not None:
            ae = FeatureAutoencoder.load(codes_path / "autoencoder.pt")
            ae.to(get_device())

        # Pick the zarr's frames, in the zarr's order, and lift the stored codes onto the points
        pointcloud = PointcloudResult.load_zarr(self.pointcloud_zarr, load_world_points=False)
        rows = store_rows(self.images_dir, pointcloud.image_paths)
        lifted = lift_features(partial(_load_frame, codes, rows, None), pointcloud)

        # Word arrays for the points and mesh vertices (ocr_lens only); Task 6
        point_words = {}
        word_attrs = {}

        # Write the per-point codes beside the lifted-store marker
        write_point_features(self.outputs["semantics"], to_numpy(lifted), ae, arrays=point_words, attrs=word_attrs)
```

`partial(_load_frame, states, range(n_frames), ae)` indexes `range` like a list (`rows[i]` = `i`), so the encode reads store row `i` for frame `i`. `_load_frame` already runs under `no_grad`.

- [ ] **Step 4: Run the reconstructor tests**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor -q -p no:cacheprovider > /tmp/claude-0/ss_t5.log 2>&1; echo exit=$?
grep -E "passed|failed" /tmp/claude-0/ss_t5.log | tail -1
```

Expected: `exit=0`, with no failures beyond the Task 0 baseline. If a test elsewhere relied on `semantics` preceding `mesh` in `STAGES` (for example a full-run call order), update its expected order. That reorder is the spec's decision.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py tests/reconstructor/test_sfm_stage.py
git commit --only collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py tests/reconstructor/test_sfm_stage.py -m "feat(reconstructor): semantics keeps fp16 codes only; AE on all frames; mesh runs first

Extract to a temporary full-width store, fit the AE streaming every frame,
encode into <extractor>_codes.zarr, delete the states. Codes lift onto the
points. STAGES puts mesh before semantics for the vertex word store.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: OCR-lens word arrays on points and mesh vertices

**Files:**
- Modify: `collab_splats/reconstructor.py`:
  - imports
  - new `_word_frame` next to `_load_frame` (~line 141)
  - `semantics()`, at the Task 5 marker
- Test: `tests/reconstructor/test_reconstructor.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/reconstructor/test_reconstructor.py`. Add `from collab_splats.semantics.features.ocr_lens import WordVocab, word_probabilities` to the imports.

```python
def _toy_lens(n_words, dim=4):
    """A fp32 linear 'lm_head' over one token per word."""
    torch.manual_seed(0)
    decoder = torch.nn.Linear(dim, n_words)
    vocab = WordVocab(
        words=[f"w{j}" for j in range(n_words)],
        token_ids=torch.arange(n_words),
        word_index=torch.arange(n_words),
    )
    return decoder, vocab


def test_word_frame_decodes_one_store_row_to_word_maps(tmp_path):
    decoder, vocab = _toy_lens(5)
    data = np.random.default_rng(0).standard_normal((2, 4, 2, 3)).astype(np.float16)
    codes = zarr.open(str(tmp_path / "c.zarr"), mode="w")
    codes["features"] = data

    out = R._word_frame(codes["features"], [1, 0], None, decoder, vocab, 0)

    states = torch.from_numpy(data[1]).reshape(4, -1).T
    expected = torch.cat([p for p, _ in word_probabilities(states, decoder, vocab)]).T.reshape(5, 2, 3)
    assert out.shape == (5, 2, 3)
    torch.testing.assert_close(out.cpu(), expected.cpu())


@pytest.mark.parametrize("n_words", [3, 70])
def test_semantics_ocr_lens_writes_top_k_words_on_points_and_vertices(tmp_path, n_words):
    """
    Top-min(64, n_words) word ids/probs plus dropped_mass, on the points and on mesh.ply's vertices.

    - lift stubbed: frame 0's first cell per target row; row 0 zeroed (unobserved)
    """
    decoder, vocab = _toy_lens(n_words)
    rec = _semantics_rec(tmp_path, {"extractor": "ocr_lens", "n_components": None})
    _seed_codes(rec, "ocr_lens", [np.random.default_rng(r).standard_normal((4, 2, 2)).astype(np.float16) for r in range(2)])
    mesh = o3d.geometry.TriangleMesh.create_tetrahedron()
    rec.backend_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(rec.outputs["mesh"]), mesh)
    targets = []

    def lift(frame_features, target):
        targets.append(target)
        out = frame_features(0)[:, 0, 0].float().cpu().repeat(len(target.points), 1)
        out[0] = 0
        return out

    with (
        patch.object(R, "lift_features", side_effect=lift),
        patch.object(R, "load_decoder", return_value=decoder),
        patch.object(R, "load_processor"),
        patch.object(R, "word_vocabulary", return_value=vocab),
    ):
        rec.semantics()

    k = min(64, n_words)

    # Expected row: frame 0's first cell decoded, as the stub lifts it
    codes = zarr.open(str(rec.semantics_cache_dir / "ocr_lens_codes.zarr"), mode="r")["features"]
    row = R._word_frame(codes, [0, 1], None, decoder, vocab, 0)[:, 0, 0].float().cpu()
    top = row.topk(k)

    for store_name, n_rows in (("ocr_lens_lifted.zarr", 1), ("ocr_lens_vertices.zarr", len(mesh.vertices))):
        store = zarr.open(str(rec.backend_dir / "semantics" / store_name), mode="r")
        assert store.attrs["words"] == vocab.words
        assert store["word_ids"].dtype == np.uint16 and store["word_ids"].shape == (n_rows, k)
        assert store["word_probs"].dtype == np.float16
        assert store["dropped_mass"].shape == (n_rows,)

        # Row 0 unobserved: zero probabilities and nothing dropped
        assert not store["word_probs"][0].any() and store["dropped_mass"][0] == 0

    # Observed vertex rows: the top-k of the dense row, and the mass top-k left out
    vertices = zarr.open(str(rec.backend_dir / "semantics" / "ocr_lens_vertices.zarr"), mode="r")
    np.testing.assert_array_equal(vertices["word_ids"][1], top.indices.numpy())
    np.testing.assert_allclose(vertices["word_probs"][1], top.values.numpy(), atol=1e-3)
    np.testing.assert_allclose(vertices["dropped_mass"][1], float(row.sum() - top.values.sum()), atol=1e-3)
    assert "features" not in vertices

    # The vertex lift has no source pixels: unseen vertices stay zero
    assert targets[-1].pixel_indices is None
    assert len(targets[-1].points) == len(mesh.vertices)


def test_semantics_ocr_lens_without_mesh_writes_no_vertex_store(tmp_path):
    decoder, vocab = _toy_lens(3)
    rec = _semantics_rec(tmp_path, {"extractor": "ocr_lens", "n_components": None})
    _seed_codes(rec, "ocr_lens", [np.ones((4, 2, 2), np.float16)] * 2)

    with (
        patch.object(R, "lift_features", side_effect=lambda ff, target: ff(0)[:, 0, 0].float().cpu().repeat(1, 1)),
        patch.object(R, "load_decoder", return_value=decoder),
        patch.object(R, "load_processor"),
        patch.object(R, "word_vocabulary", return_value=vocab),
    ):
        rec.semantics()

    assert "word_ids" in zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert not (rec.backend_dir / "semantics" / "ocr_lens_vertices.zarr").exists()
```

The dinov2 run in `test_semantics_writes_weights_inside_the_lifted_store` (Task 5) already asserts `"word_ids" not in store`, which covers "word arrays only for ocr_lens".

- [ ] **Step 2: Run them; they fail**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor/test_reconstructor.py -q -p no:cacheprovider -k "word or ocr_lens" > /tmp/claude-0/ss_t6.log 2>&1; echo exit=$?
```

Expected: `exit=1`: `AttributeError: module 'collab_splats.reconstructor' has no attribute '_word_frame'`.

- [ ] **Step 3: Implement**

Imports in `collab_splats/reconstructor.py`:
- `from dataclasses import replace`: the module already has `import dataclasses`, so use `dataclasses.replace` instead and add nothing
- `import inspect` (stdlib)
- `from collab_splats.semantics.features.ocr_lens import OCRLensExtractor, WordVocab, load_decoder, load_processor, word_probabilities, word_vocabulary`

Add after `_load_frame`:

```python
def _word_frame(
    codes: zarr.Array,
    rows: list[int],
    ae: FeatureAutoencoder | None,
    decoder: torch.nn.Module,
    vocab: WordVocab,
    i: int,
) -> torch.Tensor:
    """
    Pointcloud frame i's codes decoded to OCR-lens word probabilities, (n_words, H_p, W_p).

    - store row rows[i]; ae decodes codes to lens states, None when the store holds states
    - lives on the decoder's device; lift_features moves it on
    """
    # Patch codes as rows, (H_p * W_p, latent)
    fmap = torch.from_numpy(codes[rows[i]])
    dim, height, width = fmap.shape
    states = fmap.reshape(dim, -1)
    states = states.T

    # Word probabilities per patch, back onto the patch grid
    blocks = [probs for probs, _ in word_probabilities(states, decoder, vocab, ae=ae)]
    probs = torch.cat(blocks)
    probs = probs.T
    return probs.reshape(-1, height, width)
```

In `semantics()`, replace the Task 5 marker block:

```python
        # Word arrays for the points and mesh vertices (ocr_lens only); Task 6
        point_words = {}
        word_attrs = {}
```

with:

```python
        # OCR lens: top-64 word probabilities per point, and per mesh vertex when mesh.ply exists
        point_words = {}
        word_attrs = {}

        if name == "ocr_lens":
            model_id = extractor_kwargs.get("model_id", inspect.signature(OCRLensExtractor).parameters["model_id"].default)
            decoder = load_decoder(model_id)
            vocab = word_vocabulary(load_processor(model_id).tokenizer)
            word_frame = partial(_word_frame, codes, rows, ae, decoder, vocab)
            word_attrs = {"words": vocab.words}
            targets = {"points": pointcloud}

            # Vertices have no source pixel: no fallback, unseen stays zero
            if self.outputs["mesh"].exists():
                mesh = o3d.io.read_triangle_mesh(str(self.outputs["mesh"]))
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                targets["vertices"] = dataclasses.replace(pointcloud, points=vertices, pixel_indices=None)

            words = {}

            for key, target in targets.items():
                # Decode each frame, lift the full vocabulary, keep each row's top-k
                probs = lift_features(word_frame, target)
                top = probs.topk(min(64, probs.shape[1]), dim=1)
                dropped = probs.sum(1) - top.values.sum(1)
                words[key] = {
                    "word_ids": top.indices.numpy().astype(np.uint16),
                    "word_probs": top.values.half().numpy(),
                    "dropped_mass": dropped.clamp_min(0).half().numpy(),
                }
                del probs

            point_words = words["points"]

            if "vertices" in words:
                vertex_path = self.backend_dir / "semantics" / f"{name}_vertices.zarr"
                write_point_features(vertex_path, None, None, arrays=words["vertices"], attrs=word_attrs)

            del decoder
            pytorch_gc()
```

`dataclasses.replace(pointcloud, points=vertices, ...)` keeps `colors` at the point count. Check that `lift_features` never reads `colors` (it does not today). If `PointcloudResult.__post_init__` validates `colors` against `points`, also pass `colors=np.zeros((len(vertices), 3), np.uint8)`.

`dropped.clamp_min(0)` stops float rounding from storing tiny negative masses. On a zero row, `sum - sum` is `0`.

Wrap the `model_id` line to stay within 120 characters (two statements: `params = inspect.signature(OCRLensExtractor).parameters`, then the `get`).

- [ ] **Step 4: Run the reconstructor tests**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor -q -p no:cacheprovider > /tmp/claude-0/ss_t6.log 2>&1; echo exit=$?
grep -E "passed|failed" /tmp/claude-0/ss_t6.log | tail -1
```

Expected: `exit=0`, with no failures beyond the baseline.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py
git commit --only collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py -m "feat(semantics): store OCR-lens top-64 word probabilities on points and mesh vertices

Each frame's codes decode to word probabilities, lift onto the points
(into the lifted store) and mesh.ply's vertices (<extractor>_vertices.zarr):
word_ids uint16, word_probs fp16, dropped_mass fp16, attrs words.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Viewer reads the vertex store

**Files:**
- Modify: `docs/examples/ocr_lens_viewer.py` (whole file)

- [ ] **Step 1: Rewrite the viewer**

Replace the module docstring, the imports, `_probability_maps`, `_lift_onto` and `main()`. Keep `_chart` unchanged.

```python
#!/usr/bin/env python3
"""
Probe a scene's mesh for the OCR lens's words as a continuous heat overlay.

- needs mesh.ply and semantics/ocr_lens_vertices.zarr under the backend dir (semantics stage, ocr_lens)
- the store holds each vertex's top-64 words; no decoder loads
- unseen vertices (depth test) are grey and score zero; scene terms are every observed top-10 word
- label list: words ranked by probability mass (expected vertex count); a click queries that word
- query: comma-separated words; summed probability draws as heat, faces under min p hidden
- click a vertex to chart its top-10 scene terms; --textured shows texture/mesh.obj, picks stay on mesh.ply

Usage:
    python docs/examples/ocr_lens_viewer.py /workspace/outputs/<scene>/<backend> --port 8080
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import zarr
from PIL import Image

from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)
```

`main()`:

```python
def main() -> None:
    """
    Serve the mesh with a mass-ranked word list, a word query drawn as heat and a click probe.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backend_dir", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--textured", action="store_true", help="show texture/mesh.obj over the same surface")
    parser.add_argument("--texture_size", type=int, default=4096, help="displayed texture edge, pixels")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Mesh with its vertex colors; light grey when the mesh has none
    mesh = trimesh.load(args.backend_dir / "mesh.ply", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)

    if mesh.visual.kind == "vertex":
        colors = np.array(mesh.visual.vertex_colors[:, :3])
    else:
        colors = np.full((len(vertices), 3), 200, dtype=np.uint8)

    # Per-vertex top-64 words from the semantics stage
    store = zarr.open(str(args.backend_dir / "semantics" / "ocr_lens_vertices.zarr"), mode="r")
    word_ids = store["word_ids"][:].astype(np.int64)
    word_probs = store["word_probs"][:].astype(np.float32)
    words = list(store.attrs["words"])
    row_of = {word: row for row, word in enumerate(words)}

    # A mesh re-run alone leaves the store describing other vertices
    if len(word_ids) != len(vertices):
        raise SystemExit(
            f"ocr_lens_vertices.zarr has {len(word_ids)} rows for {len(vertices)} mesh vertices: "
            "mesh.ply changed since semantics ran; re-run the semantics stage"
        )

    # Unobserved vertices have zero probabilities and are grey
    observed = word_probs[:, 0] > 0
    colors[~observed] = 128
    logger.info("unobserved vertices: %d/%d (%.1f%%)", (~observed).sum(), len(vertices), 100 * (~observed).mean())

    # Scene terms: every word in some observed vertex's top-10
    is_term = np.zeros(len(words), dtype=bool)
    is_term[np.unique(word_ids[observed, :10])] = True

    # Photo texture, when asked: the corner-split OBJ of this same mesh, downscaled for display
    textured = None

    if args.textured:
        textured = trimesh.load(args.backend_dir / "texture" / "mesh.obj", process=False)
        material = textured.visual.material
        size = (args.texture_size, args.texture_size)
        material.image = material.image.resize(size, Image.LANCZOS)

    # Scene, probe chart and query box
    viewer = Viewer(port=args.port)
    viewer.server.gui.configure_theme(control_layout="fixed")
    viewer.add_mesh("mesh", vertices, faces, colors, textured=textured)
    panel = viewer.server.gui.add_html("")
    query = viewer.server.gui.add_text("Query", initial_value="")
    min_prob = viewer.server.gui.add_slider("Query min p", min=0.0, max=1.0, step=0.01, initial_value=0.3)
    search_button = viewer.server.gui.add_button("Search")
    query_note = viewer.server.gui.add_markdown("")
    marker_radius = 0.005 * float(np.linalg.norm(np.ptp(vertices, axis=0)))

    def search(_=None) -> None:
        """
        Draw the summed stored probability of the query words as heat; faces under min p hidden.
        """
        typed = [word.strip().lower() for word in query.value.split(",") if word.strip()]
        known = [word for word in typed if word in row_of]
        unknown = [word for word in typed if word not in row_of]
        query_note.content = f"not in vocabulary: {', '.join(unknown)}" if unknown else ""

        # Empty query clears; a word outside a vertex's top-64 reads as 0 there
        score = None

        if known:
            hit = np.isin(word_ids, [row_of[word] for word in known])
            score = (word_probs * hit).sum(axis=1)

        viewer.show_heat("mesh", score, min_prob.value)

    def select(word: Optional[str]) -> None:
        """
        Put a label-list word in the query box (Clear empties it) and search.
        """
        query.value = word or ""
        search()

    def show(vertex: int) -> None:
        """
        Mark the clicked vertex and chart its top-10 scene terms, renormalized, in the top left.
        """
        viewer.server.scene.add_icosphere(
            "/probe", radius=marker_radius, color=(255, 0, 255), position=vertices[vertex]
        )

        if not observed[vertex]:
            panel.content = _chart(f"vertex {vertex}: unobserved", [], np.zeros(0))
            return

        # Stored entries that are scene terms, already sorted by probability
        ids = word_ids[vertex]
        probs = word_probs[vertex]
        keep = is_term[ids] & (probs > 0)
        ids = ids[keep]
        probs = probs[keep] / probs[keep].sum()
        panel.content = _chart(f"vertex {vertex}", [words[j] for j in ids[:10]], probs[:10])

    # Words ranked by probability mass: expected vertex count, the total of a click's heat
    mass = np.bincount(word_ids.ravel(), weights=word_probs.ravel(), minlength=len(words))
    viewer.add_label_list("mesh", words, mass, select)

    # Wire the query and clicks; block
    search_button.on_click(search)
    viewer.on_click("mesh", show)
    viewer.serve_forever()
```

- [ ] **Step 2: Lint and import-check**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m black --check docs/examples/ocr_lens_viewer.py; PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m isort --check docs/examples/ocr_lens_viewer.py
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python docs/examples/ocr_lens_viewer.py --help
```

Expected: both checks clean (run `black`/`isort` on this one file if not), and `--help` prints the usage without `--model_id` or `--chunk`. The viewer is exercised live in Task 9.

- [ ] **Step 3: Commit**

```bash
git add docs/examples/ocr_lens_viewer.py
git commit --only docs/examples/ocr_lens_viewer.py -m "refactor(examples): ocr_lens_viewer reads the vertex word store

No decoder, no per-frame decode, no .npy cache: word_ids/word_probs from
semantics/ocr_lens_vertices.zarr drive heat, mass ranking and the probe.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Docs

**Files:**
- Modify: `docs/semantics.md` (the "On-disk layout" section)
- Modify: `configs/base.yaml:114` (`n_components` comment)
- Modify: `CLAUDE.md` (the semantics-storage in-flight entry: add the plan link)

- [ ] **Step 1: Rewrite the layout section of `docs/semantics.md`**

Replace everything from `## On-disk layout` up to the next `---` with:

```markdown
## On-disk layout

| store | contents | written by |
|---|---|---|
| `<scene>/semantics/<extractor>_codes.zarr` | `features` (N, latent, H_p, W_p) fp16, one chunk per frame; `autoencoder.pt`; attrs `extractor`, `patch_size`, `n_frames`, `extractor_kwargs`, `latent_dim` | `write_feature_cache` |
| `<scene>/semantics/<extractor>.zarr` | full-width states, (N, D, H_p, W_p) fp16; temporary, deleted once encoded | `write_feature_cache` |
| `<scene>/<backend>/semantics/<extractor>_lifted.zarr` | `features` (P, latent) fp16, `autoencoder.pt`; ocr_lens adds the word arrays | `write_point_features` |
| `<scene>/<backend>/semantics/<extractor>_vertices.zarr` | ocr_lens only: word arrays per `mesh.ply` vertex | `write_point_features` |

- `n_components: null`: `_codes.zarr` holds full-width features, `latent_dim` null, no AE, no temporary store
- the AE trains once on every frame (streamed); changing `n_components` re-runs the extractor
- word arrays: `word_ids` (T, 64) uint16, `word_probs` (T, 64) fp16 sorted descending, `dropped_mass` (T,) fp16; attrs `words`
- unobserved rows have all-zero `word_probs`
- `<scene>/semantics/` stays local (`PUSH_EXCLUDES`); the backend stores are pushed
- semantics runs after mesh; a mesh re-run alone leaves `_vertices.zarr` stale, and the viewer refuses it
```

Also grep the file for `extract_feature_cache`, `load_features` and `_ae.pt`, and update every hit to the new names.

- [ ] **Step 2: `configs/base.yaml` comment**

```yaml
  n_components: 64            # AE code width stored in <extractor>_codes.zarr; null = full width; 128 for ocr_lens; changing it re-extracts
```

- [ ] **Step 3: CLAUDE.md in-flight entry gets the plan link**

In `/workspace/collab-splats/.worktrees/semantics-storage/CLAUDE.md`, change the entry's link group from `([spec](docs/superpowers/specs/2026-10-07-semantics-storage-design.md))` to `([spec](docs/superpowers/specs/2026-10-07-semantics-storage-design.md) · [plan](docs/superpowers/plans/2026-10-07-semantics-storage.md))`. Skip this when the plan commit on `clean/final` already made the change.

- [ ] **Step 4: Commit**

```bash
git add docs/semantics.md configs/base.yaml CLAUDE.md
git commit --only docs/semantics.md configs/base.yaml CLAUDE.md -m "docs(semantics): codes-only 2D cache, word arrays, vertex store

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Gate and real run on GH010229

**Files:** none (numbers are reported in chat and recorded in the CHANGELOG entry)

- [ ] **Step 1: Gate**

```bash
cd /workspace/collab-splats/.worktrees/semantics-storage
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/reconstructor tests/utils tests/test_docstring_contract.py tests/test_import_style.py -q -p no:cacheprovider > /tmp/claude-0/ss_gate.log 2>&1; echo exit=$?
grep -E "passed|failed" /tmp/claude-0/ss_gate.log | tail -1
grep -rn "extract_feature_cache\|load_features\b" collab_splats tests evals docs --include=*.py --include=*.md | grep -v "docs/superpowers\|dashboard"
```

Expected:
- the printed path is under the worktree
- the pass count is at or above the baseline, with failures only from the baseline list
- the grep is empty; the dashboard imports were already broken (spec: out of scope)

Run `/graphify update .` afterwards (per `/workspace/CLAUDE.md`).

- [ ] **Step 2: Ask before the real run**

The run overwrites `/workspace/outputs/ocr_viewer/GH010229/semantics/ocr_lens.zarr` (today's full-state cache, 294 frames) and the backend's `ocr_lens_lifted.zarr`. Re-creating either means re-extracting with LLaVA. Get the user's go-ahead in chat before Step 3. Keep `vggt_omega/ocr_lens_vertices.npy`: it is the top-1 reference.

- [ ] **Step 3: Run the semantics stage in tmux**

```bash
tmux new -d -s ss_run "cd /workspace/collab-splats/.worktrees/semantics-storage && \
  PYTHONPATH=\$PWD HF_HOME=/workspace/models HF_HUB_OFFLINE=1 /usr/bin/time -v \
  /opt/venv/reconstruction/bin/python -m collab_splats local /workspace/outputs/ocr_viewer/GH010229 \
    --output-root /workspace/outputs/ocr_viewer --stages semantics --overwrite \
    --set semantics.extractor=ocr_lens --set semantics.n_components=128 \
  > /tmp/claude-0/ss_run.log 2>&1; echo exit=\$? >> /tmp/claude-0/ss_run.log"
```

- In a second tmux window, sample GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv -l 5 > /tmp/claude-0/ss_gpu.log`.
- Wait with Monitor on `exit=` in the log. Do not run other heavy processes meanwhile (OOM risk).
- Stage times come from the log's timestamps (extract, `fit autoencoder`, `feature cache written` ×2, then the three lifts). Peak RSS comes from `time -v` "Maximum resident set size".

If `local` rejects a processed scene dir as input, read `collab_splats/__main__.py` `_scene_name` and the leaf-stage re-run path (memory: "leaf-only `--stages` pulls from processed"), then use the documented form.

- [ ] **Step 4: Measure**

```bash
du -sh /workspace/outputs/ocr_viewer/GH010229/semantics/* /workspace/outputs/ocr_viewer/GH010229/vggt_omega/semantics/*
```

Top-1 agreement against the old `.npy`, from the scratchpad:

```python
import numpy as np, zarr
old = np.load("/workspace/outputs/ocr_viewer/GH010229/vggt_omega/ocr_lens_vertices.npy", mmap_mode="r")
new = zarr.open("/workspace/outputs/ocr_viewer/GH010229/vggt_omega/semantics/ocr_lens_vertices.zarr", mode="r")
ids = new["word_ids"][:, 0]
seen_new = new["word_probs"][:, 0] > 0
top_old = np.concatenate([np.asarray(old[s:s + 65536]).argmax(1) for s in range(0, len(old), 65536)])
seen_old = np.concatenate([np.asarray(old[s:s + 65536]).any(1) for s in range(0, len(old), 65536)])
both = seen_new & seen_old
print("observed old/new/both", seen_old.sum(), seen_new.sum(), both.sum())
print("top-1 agreement", (ids[both] == top_old[both]).mean())
```

Viewer start-up: launch the viewer in tmux and time from launch to the `serve` log line:

```bash
time (PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python docs/examples/ocr_lens_viewer.py /workspace/outputs/ocr_viewer/GH010229/vggt_omega --port 8080)
```

Then open the viewer:
- click a vertex
- query `tree`
- check that the label list ranks

- [ ] **Step 5: Report**

Post a table in chat, with no thresholds:
- disk per store, before vs after
- stage times
- peak GPU and RSS
- observed counts
- top-1 agreement
- viewer start-up

Remove `ocr_lens_vertices.npy` only after the user says so.

- [ ] **Step 6: Changelog and in-flight entry**

Append a `semantics-storage` entry to `docs/superpowers/CHANGELOG.md` with the Step 5 numbers and the branch tip. Remove the semantics-storage line from CLAUDE.md "In-Flight Work" and add it to "Recently Completed" (five newest only). Commit with `git add -f docs/superpowers/CHANGELOG.md` and `git commit --only docs/superpowers/CHANGELOG.md CLAUDE.md`. Merging to `clean/final` is the user's call; ask.
