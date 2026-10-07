# Semantics Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut semantics storage from ~20 GB to under 1 GB per scene by keeping fp16 AE codes only (option B: no stored words, no vertex store).

**Architecture:** The semantics stage extracts full-width states to a temporary `<extractor>_states.zarr`, trains the AE on every frame by streaming, encodes each frame to fp16 codes in `<extractor>_codes.zarr`, and deletes the states. Codes lift onto the points into `<backend>/semantics/<extractor>_lifted.zarr`. Codes stores are pushed; states never. The OCR-lens viewer decodes each frame's codes to top-64 words per patch at start-up and lifts them onto the mesh vertices with `lift_features(..., num_classes=len(vocab))` over indexed `(ids, probs)` maps. `lift_features` samples visible points only.

**Tech Stack:** torch, zarr v3, trimesh (viewer mesh read), viser viewer, pytest.

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
| `collab_splats/semantics/lifting.py` | `lift_features`: sample and `index_add_` visible points only; in-place weight and mean; indexed `(ids, values)` maps with `num_classes=` (misuse raises; no fallback for them); private `_add_indexed_bilinear`, the indexed sibling of `_grid_sample_at_pixels` |
| `collab_splats/semantics/store.py` | `valid_feature_cache(store_path, name, images_dir, extractor_kwargs, latent_dim)`; `extract_feature_cache` → `write_feature_cache(store_path, maps, n_frames, attrs, ae=None)`; `write_point_features` casts to fp16 |
| `collab_splats/semantics/__init__.py` | export rename |
| `collab_splats/semantics/compression.py` | `FeatureAutoencoder.fit` streams any (N, D, ...) array |
| `collab_splats/utils/torch_utils.py` | delete `load_features` |
| `collab_splats/reconstructor.py` | `semantics()` flow; `STAGES` unchanged |
| `collab_splats/remote.py` | `PUSH_EXCLUDES`: `/semantics/**` → `/semantics/*_states.zarr/**` |
| `docs/examples/ocr_lens_viewer.py` | start-up: decode codes to top-64 per patch, indexed lift onto vertices in chunks; no `.npy` cache |
| `docs/semantics.md`, `configs/README.md`, `configs/base.yaml`, `CLAUDE.md` | docs |
| `tests/semantics/test_lifting.py`, `test_store.py`, `test_compression_target.py`, `features/test_extract_from_zarr.py` | tests |
| `tests/reconstructor/test_reconstructor.py`, `test_sfm_stage.py`, `tests/utils/test_torch_utils.py`, `tests/remote/test_remote.py` | tests |

One deviation from the spec's signature table: `write_feature_cache` takes `ae=None` and saves `autoencoder.pt` before the validity attrs. Without it, a crash between writing the attrs and saving the AE leaves a "valid" codes store with no decoder. This mirrors `write_point_features(store_path, codes, ae)`.

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
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/reconstructor tests/utils tests/remote tests/test_docstring_contract.py tests/test_import_style.py -q -p no:cacheprovider > /tmp/claude-0/ss_baseline.log 2>&1; echo exit=$?
grep -E "passed|failed" /tmp/claude-0/ss_baseline.log | tail -1
```

Record the pass/fail counts. Any failure that is listed in `docs/known-test-failures.md` is the baseline, not a regression.

---

### Task 1: `lift_features` samples visible points only; indexed-map input

**Files:**
- Modify: `collab_splats/semantics/lifting.py` (`lift_features`; new private `_add_indexed_bilinear` after `_grid_sample_at_pixels`)
- Test: `tests/semantics/test_lifting.py`

- [ ] **Step 1: Write the equality test against a reference copy of today's loop**

Append to `tests/semantics/test_lifting.py`. Add `from collab_splats.geometry.projection import depth_residual` and `_grid_sample_at_pixels` to the `collab_splats.semantics.lifting` import (keep isort order).

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


def _partial_scene(rng, pixel_indices=False, n=3, h=16, w=20, p=200):
    """
    Toy scene where some points are visible in some frames only.

    - frame 2's depth map is far behind every point, so no point is visible there
    - pixel_indices=True gives every point a random source pixel, so the fallback runs
    """
    pts = np.stack(
        [rng.uniform(-0.1, 0.1, p), rng.uniform(-0.08, 0.08, p), rng.uniform(1.0, 2.0, p)], axis=1
    ).astype(np.float32)
    depth = rng.uniform(1.0, 2.0, (n, h, w)).astype(np.float32)
    depth[2] = 100.0
    conf = rng.uniform(0.1, 1.0, (n, h, w)).astype(np.float32)
    pixels = None

    if pixel_indices:
        pixels = np.stack([rng.integers(0, n, p), rng.integers(0, h, p), rng.integers(0, w, p)], axis=1)
        pixels = pixels.astype(np.int32)

    return _make_lift_result(pts, pixels, depth, conf, n=n, h=h, w=w)


def test_lift_features_matches_the_all_points_reference(monkeypatch):
    """
    Sampling only visible points changes nothing: invisible points carried weight 0.
    """
    monkeypatch.setattr("collab_splats.semantics.lifting.get_device", lambda: torch.device("cpu"))
    rng = np.random.default_rng(0)
    result = _partial_scene(rng)
    maps = [torch.from_numpy(rng.standard_normal((5, 4, 5)).astype(np.float32)) for _ in range(3)]

    expected = _lift_reference(maps, result)
    out = lift_features(maps.__getitem__, result)

    # Partial visibility: some points observed, some not
    observed = expected.abs().sum(1) > 0
    assert 0 < int(observed.sum()) < len(observed)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
```

- [ ] **Step 2: Run it; it passes on today's code (it pins today's output)**

```bash
cd /workspace/collab-splats/.worktrees/semantics-storage
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_lifting.py -q -p no:cacheprovider > /tmp/claude-0/ss_t1.log 2>&1; echo exit=$?
```

Expected: `exit=0`. If the "partial visibility" assert fails, widen `depth`'s uniform range in `_partial_scene` so that some points fail the 5% depth test, then rerun. This test is the refactor's safety net, so it must pass before any change.

- [ ] **Step 3: Write the indexed-equals-dense and misuse tests**

Append to `tests/semantics/test_lifting.py`:

```python
def test_indexed_lift_equals_the_dense_lift_of_the_scattered_maps(monkeypatch):
    """
    Indexed (ids, probs) maps lift exactly as the same entries scattered to num_classes channels.

    - points visible nowhere stay zero, as in the dense lift without pixel_indices
    """
    monkeypatch.setattr("collab_splats.semantics.lifting.get_device", lambda: torch.device("cpu"))
    rng = np.random.default_rng(1)
    result = _partial_scene(rng)
    num_classes, k = 12, 4

    # Per patch: k distinct class ids with random probabilities, and the same entries dense
    torch.manual_seed(0)
    indexed, dense = [], []

    for _ in range(3):
        ids = torch.rand(num_classes, 4, 5).topk(k, dim=0).indices
        probs = torch.rand(k, 4, 5)
        indexed.append((ids, probs))
        dense.append(torch.zeros(num_classes, 4, 5).scatter_(0, ids, probs))

    out = lift_features(indexed.__getitem__, result, num_classes=num_classes)
    expected = lift_features(dense.__getitem__, result)

    assert out.shape == (len(result.points), num_classes) and out.dtype == torch.float32
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)

    # Some rows unobserved, and zero
    assert int((expected.abs().sum(1) == 0).sum()) > 0


@pytest.mark.parametrize(
    "indexed, num_classes, first_id",
    [(True, None, 0), (True, 1, 0), (True, 4, -1), (False, 5, 0)],
    ids=["indexed-without-num-classes", "id-past-num-classes", "negative-id", "dense-with-num-classes"],
)
def test_num_classes_misuse_raises_instead_of_truncating(monkeypatch, indexed, num_classes, first_id):
    """
    num_classes is the id space of indexed maps; any other use raises.
    """
    monkeypatch.setattr("collab_splats.semantics.lifting.get_device", lambda: torch.device("cpu"))
    result = _partial_scene(np.random.default_rng(2))

    # Indexed maps hold ids first_id and first_id + 1 at every patch; dense maps have 5 channels
    ids = torch.arange(first_id, first_id + 2).view(2, 1, 1).expand(2, 4, 5)
    fmap = (ids, torch.ones(2, 4, 5)) if indexed else torch.ones(5, 4, 5)

    with pytest.raises(ValueError, match="num_classes"):
        lift_features(lambda i: fmap, result, num_classes=num_classes)


def test_indexed_maps_refuse_the_pixel_indices_fallback(monkeypatch):
    """
    The fallback samples dense maps only, so indexed maps with pixel_indices raise.
    """
    monkeypatch.setattr("collab_splats.semantics.lifting.get_device", lambda: torch.device("cpu"))
    result = _partial_scene(np.random.default_rng(3), pixel_indices=True)
    fmap = (torch.zeros(2, 4, 5, dtype=torch.long), torch.ones(2, 4, 5))

    with pytest.raises(ValueError, match="pixel_indices"):
        lift_features(lambda i: fmap, result, num_classes=4)
```

- [ ] **Step 4: Run it; it fails**

Run the Step 2 command. Expected: `exit=1`; every new test fails with `TypeError: lift_features() got an unexpected keyword argument 'num_classes'`.

- [ ] **Step 5: Implement visible-only accumulation and the indexed input**

Add `num_classes: Optional[int] = None` after `depth_tol` in the `lift_features` signature (add `from typing import Optional`; the module already imports `TYPE_CHECKING` from `typing`). Docstring changes:
- `frame_features` arg: `frame i's features, called once per frame, then once per fallback frame (dense maps only); a list passes maps.__getitem__. Either a dense (D, H_p, W_p) map, any float dtype, or an indexed map: an (ids, values) pair, each (K, H_p, W_p), listing K entries per patch of a num_classes-long vector whose other entries are 0.`
- new arg `num_classes: indexed maps only; number of distinct ids, i.e. the output width. Values at the K listed ids per patch are summed per point; K is never a dimension. None for dense maps.`
- bullets: `- only visible points are sampled; each frame costs its visible count, not P` and `- indexed maps add each point's bilinear blend of its 4 nearest patches' lists at the id columns: the dense lift's result without the zero columns`
- `Raises:` `ValueError: indexed maps without num_classes, with pixel_indices, or with an id outside [0, num_classes); num_classes with dense maps.`
- `Returns:` `(P, D) or (P, num_classes) float32 tensor ...`

In the main loop, replace:

```python
    for i in range(N):
        fmap = frame_features(i)
        fmap = fmap.to(device=device, dtype=torch.float32)  # (D, H_p, W_p)

        # Accumulator width comes from the first frame
        if features_sum is None:
            features_sum = torch.zeros((P, fmap.shape[0]), dtype=torch.float32, device=device)
```

with:

```python
    for i in range(N):
        fmap = frame_features(i)

        # Accumulator width: num_classes for indexed maps, else the first frame's channels
        if features_sum is None:
            indexed = not isinstance(fmap, torch.Tensor)

            if indexed and num_classes is None:
                raise ValueError("indexed (ids, values) maps need num_classes, the number of distinct ids")

            if indexed and result.pixel_indices is not None:
                raise ValueError("indexed maps take no pixel_indices fallback; pass a result without pixel_indices")

            if not indexed and num_classes is not None:
                raise ValueError("num_classes applies to indexed maps only; dense maps keep their channels")

            width = num_classes if indexed else fmap.shape[0]
            features_sum = torch.zeros((P, width), dtype=torch.float32, device=device)
```

Replace:

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

        if indexed:
            ids, values = fmap
            _add_indexed_bilinear(features_sum, ids, values, idx, v_safe[idx], u_safe[idx], w[idx], image_size)
        else:
            fmap = fmap.to(device=device, dtype=torch.float32)  # (D, H_p, W_p)
            sampled = _grid_sample_at_pixels(fmap, v_safe[idx], u_safe[idx], image_size)  # (P_vis, D)
            sampled *= w[idx].unsqueeze(-1)
            features_sum.index_add_(0, idx, sampled)

        weights_sum += w

    # Weighted mean in place, eps for numerical safety
    features = features_sum
    features /= weights_sum.unsqueeze(-1) + 1e-8
```

The fallback loop stays as it is: it runs on dense maps only, since indexed maps with `pixel_indices` raised above.

Add after `_grid_sample_at_pixels`:

```python
def _add_indexed_bilinear(
    acc: torch.Tensor,
    ids: torch.Tensor,
    values: torch.Tensor,
    idx: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    weights: torch.Tensor,
    image_size: tuple[int, int],
) -> None:
    """
    Add each point's bilinear blend of its 4 nearest patches' (id, value) lists into acc, in place.

    - ids, values (K, H_p, W_p); point j's blend goes to row idx[j], each value at column id
    - same corners and weights as grid_sample (border, align_corners=False), so it equals the
      dense lift of the same lists scattered to acc's width
    - raises ValueError on an id < 0 or >= acc width, which would land in another point's row
    """
    # An id outside [0, num_classes) would write into another point's row
    lo, hi = int(ids.min()), int(ids.max())

    if lo < 0 or hi >= acc.shape[1]:
        raise ValueError(f"ids span [{lo}, {hi}], outside [0, num_classes={acc.shape[1]})")

    # Lists as patch rows: (H_p * W_p, K) ids and values
    k, height, width = ids.shape
    ids = ids.to(device=acc.device, dtype=torch.long).reshape(k, -1).T
    values = values.to(device=acc.device, dtype=torch.float32).reshape(k, -1).T
    H, W = image_size

    # Patch-grid coords, clamped as border padding clamps them
    x = (cols + 0.5) * width / W - 0.5
    y = (rows + 0.5) * height / H - 0.5
    x = x.clamp(0, width - 1)
    y = y.clamp(0, height - 1)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x - x0
    wy = y - y0

    # Four nearest patches per point, each weighted by its bilinear share times the point weight
    corners = torch.cat([y0 * width + x0, y0 * width + x1, y1 * width + x0, y1 * width + x1])
    corner_w = torch.cat([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
    corner_w *= weights.repeat(4)

    # K entries per patch into the flat accumulator: row * num_classes + id
    flat = idx.repeat(4).unsqueeze(1) * acc.shape[1] + ids[corners]
    weighted = values[corners] * corner_w.unsqueeze(1)
    acc.view(-1).index_add_(0, flat.reshape(-1), weighted.reshape(-1))
```

`grid_sample` with `align_corners=False` unnormalizes `x_n = (2c + 1) / W - 1` to `((x_n + 1) * w - 1) / 2 = (c + 0.5) * w / W - 0.5`, and border padding clamps that to `[0, w - 1]`. `_add_indexed_bilinear` uses the same coordinate, so a dense map scattered from `(ids, values)` samples identically.

Bilinear is required, not nearest: on GH010229 (top-64 per patch), reading only the nearest patch kept the bilinear top-1 word on 82.4% of vertices and 89.9% of points (`nearest_lift.py` in the session scratchpad).

- [ ] **Step 6: Run the lifting tests**

Run the Step 2 command. Expected: `exit=0`, with every `test_lifting.py` test passing (the reference test, the indexed-equals-dense test, all four misuse cases, the fallback refusal).

- [ ] **Step 7: Commit**

```bash
git add tests/semantics/test_lifting.py collab_splats/semantics/lifting.py
git commit --only tests/semantics/test_lifting.py collab_splats/semantics/lifting.py -m "perf(semantics): lift_features samples visible points only; indexed-map input

index_add_ over visible points; in-place weight and mean. Same output:
invisible points carried weight 0. frame_features may return indexed
(ids, values) maps with num_classes=, the id space and output width;
_add_indexed_bilinear adds each point's bilinear blend of its 4 nearest
patches' lists at the id columns, equal to the dense lift. Misusing
num_classes raises instead of writing into other rows; indexed maps
take no pixel_indices fallback. GH010229 word lift 264 s / 21.5 GiB
-> 74 s / 12.2 GiB dense; vertex top-64 lift 2.4 s / 9.1 GiB indexed.

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

`store.zarr` is the same module object `open_valid` (in `utils/io.py`) calls, so patching `store.zarr.open` reaches the validity check.

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

### Task 3: `write_point_features` stores fp16

**Files:**
- Modify: `collab_splats/semantics/store.py` (`write_point_features`, one line of `read_point_features`)
- Test: `tests/semantics/test_store.py` (the lifted-store section)

- [ ] **Step 1: Write the failing test**

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
```

- [ ] **Step 2: Run it; it fails**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_store.py -q -p no:cacheprovider > /tmp/claude-0/ss_t3.log 2>&1; echo exit=$?
```

Expected: `exit=1`; the dtype assert sees `float32`.

- [ ] **Step 3: Implement**

In `write_point_features`, replace `store["features"] = codes` with:

```python
        store["features"] = codes.astype(np.float16)
```

Add the docstring bullet `- codes stored fp16; read_point_features returns them float32`.

In `read_point_features`, read the codes as float32. Today it does `codes = np.asarray(store["features"])`; on an fp16 store the no-AE branch would return `F.normalize` of an fp16 tensor, i.e. float16, against its float32 contract (the AE branch is safe: `iter_decode` casts). Replace that line with:

```python
    codes = np.asarray(store["features"], dtype=np.float32)
```

- [ ] **Step 4: Run the store tests**

Run the same command as Step 2. Expected: `exit=0`. `test_write_point_features_puts_the_autoencoder_inside_the_store` still sees attrs `{"input_dim": 32, "latent_dim": 8}`. The `assert_allclose` round-trip tests compare fp16-stored values at `rtol=1e-6`; loosen those to `rtol=1e-3` (fp16 has an 11-bit mantissa), because the precision change is intended. A failed write leaving no store is already covered by the existing atomic-write test; check it still passes.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/store.py tests/semantics/test_store.py
git commit --only collab_splats/semantics/store.py tests/semantics/test_store.py -m "feat(semantics): lifted stores hold fp16 codes

write_point_features casts codes to fp16; read_point_features reads them
back as float32.

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

### Task 5: Semantics stage: extract → fit → encode → delete → lift

**Files:**
- Modify: `collab_splats/reconstructor.py`:
  - imports (lines ~16-75)
  - `semantics_cache_dir` docstring (~line 419)
  - `semantics()` (~lines 841-901)
- Test: `tests/reconstructor/test_reconstructor.py` (semantics tests, lines ~400-575 and ~1320-1345)
- Test: `tests/reconstructor/test_sfm_stage.py` (`test_semantics_lifts_only_the_rows_the_pointcloud_holds`)

- [ ] **Step 1: Rewrite the stage tests**

In `tests/reconstructor/test_reconstructor.py`:

(a) `STAGES` stays as it is: semantics needs only `pointcloud` and never reads the mesh.

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

    - lift returns frame 0's first cell repeated per target row, so its width is the loader's
    """
    calls = []

    def forward(frames):
        first = len(calls)
        calls.extend(frames)
        return [torch.full((dim, 2, 2), float(first + j + 1)) for j in range(len(frames))]

    def lift(frame_features, result):
        return frame_features(0)[:, 0, 0].float().cpu().repeat(len(result.points), 1)

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
    assert not (rec.semantics_cache_dir / "dinov2_states.zarr").exists()


def test_semantics_writes_weights_inside_the_lifted_store(tmp_path):
    rec = _semantics_rec(tmp_path, COMPRESSED)

    with _stub_extraction():
        rec.semantics()

    out_dir = rec.backend_dir / "semantics"
    assert rec.done("semantics")
    assert (out_dir / "dinov2_lifted.zarr" / "autoencoder.pt").is_file()
    store = zarr.open(str(out_dir / "dinov2_lifted.zarr"), mode="r")
    assert store["features"].shape == (1, 8) and store["features"].dtype == np.float16
    assert dict(store.attrs) == {"input_dim": 32, "latent_dim": 8}


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
- `_sfm_reconstructor` builds a real `images/` holding every `FRAME_IDX` frame, so the seeded store reads valid

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

`semantics_cache_dir` docstring: "Scene-level 2D feature cache: `<extractor>_codes.zarr`, plus a temporary `<extractor>_states.zarr` of full-width states while the AE trains."

Replace `semantics()`:

```python
    def semantics(self) -> None:
        """
        Extract 2D features, compress every frame to fp16 codes, lift the codes onto the points.

        - codes store `<extractor>_codes.zarr` reused when valid (name, frames, kwargs, latent_dim)
        - a miss extracts full-width states to a temporary `<extractor>_states.zarr`, trains the AE on all of
          them, encodes every frame, then deletes the states; n_components null keeps full width, no AE
        - lift reads one frame of codes at a time; rows follow the zarr's frames (maybe a subset)
        - lifted store written atomically with its own autoencoder.pt; codes and lifted stores are pushed
        """
        cfg = self.config["semantics"]
        name = cfg["extractor"]
        extractor_kwargs = cfg["extractor_kwargs"]
        latent_dim = cfg["n_components"]
        codes_path = self.semantics_cache_dir / f"{name}_codes.zarr"
        states_path = self.semantics_cache_dir / f"{name}_states.zarr"
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

        # Write the per-point codes; the lifted store is the stage's done marker
        write_point_features(self.outputs["semantics"], to_numpy(lifted), ae)
```

`partial(_load_frame, states, range(n_frames), ae)` indexes `range` like a list (`rows[i]` = `i`), so the encode reads store row `i` for frame `i`. `_load_frame` already runs under `no_grad`.

- [ ] **Step 4: Run the reconstructor tests**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor -q -p no:cacheprovider > /tmp/claude-0/ss_t5.log 2>&1; echo exit=$?
grep -E "passed|failed" /tmp/claude-0/ss_t5.log | tail -1
```

Expected: `exit=0`, with no failures beyond the Task 0 baseline.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py tests/reconstructor/test_sfm_stage.py
git commit --only collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py tests/reconstructor/test_sfm_stage.py -m "feat(reconstructor): semantics keeps fp16 codes only; AE on all frames

Extract to a temporary full-width store, fit the AE streaming every frame,
encode into <extractor>_codes.zarr, delete the states. Codes lift onto the
points.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Push the codes store; exclude only the temporary states

**Files:**
- Modify: `collab_splats/remote.py:29-31` (`PUSH_EXCLUDES` and its comment)
- Test: `tests/remote/test_remote.py` (`test_push_excludes_raw_feature_maps`)

The viewer and any re-lift (local, or remote after a pull) read `<scene>/semantics/<extractor>_codes.zarr`, so it is pushed. Only `<extractor>_states.zarr`, which exists while the AE trains, stays local.

- [ ] **Step 1: Rewrite the test**

Replace `test_push_excludes_raw_feature_maps` in `tests/remote/test_remote.py` with:

```python
def test_push_excludes_only_the_temporary_feature_states():
    """
    The temporary full-width states stay local; the 2D codes and the lifted stores are pushed.

    - rclone reads a leading slash as "relative to the transfer root"; strip it to model that
    """
    patterns = [p.lstrip("/") for p in PUSH_EXCLUDES]
    assert "/semantics/*_states.zarr/**" in PUSH_EXCLUDES
    assert any(fnmatch.fnmatchcase("semantics/dinov2_states.zarr/features/c/0/0/0/0", p) for p in patterns)

    # Codes store, its AE and the backend's lifted store all travel
    kept = (
        "semantics/dinov2_codes.zarr/features/c/0/0/0/0",
        "semantics/dinov2_codes.zarr/autoencoder.pt",
        "vggt_omega/semantics/dinov2_lifted.zarr/features/c/0/0",
    )

    for name in kept:
        assert not any(fnmatch.fnmatchcase(name, p) for p in patterns), name
```

- [ ] **Step 2: Run it; it fails**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/remote/test_remote.py -q -p no:cacheprovider > /tmp/claude-0/ss_t6.log 2>&1; echo exit=$?
```

Expected: `exit=1`; the `"/semantics/*_states.zarr/**" in PUSH_EXCLUDES` assert fails.

- [ ] **Step 3: Implement**

In `collab_splats/remote.py`, replace:

```python
    # Scene-root 2D patch cache; the leading slash keeps <backend>/semantics pushed
    "/semantics/**",
```

with:

```python
    # Scene-root temporary full-width states; the 2D codes and <backend>/semantics are pushed
    "/semantics/*_states.zarr/**",
```

- [ ] **Step 4: Run the remote and CLI tests**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/remote tests/reconstructor/test_cli.py -q -p no:cacheprovider > /tmp/claude-0/ss_t6.log 2>&1; echo exit=$?
grep -E "passed|failed" /tmp/claude-0/ss_t6.log | tail -1
```

Expected: `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/remote.py tests/remote/test_remote.py
git commit --only collab_splats/remote.py tests/remote/test_remote.py -m "feat(remote): push the 2D semantics codes; exclude only the temporary states

PUSH_EXCLUDES /semantics/** -> /semantics/*_states.zarr/**. The fp16
codes store (~360 MB on GH010229) is what the viewer and any re-lift read.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Viewer decodes the codes and lifts top-64 words per patch at start-up

**Files:**
- Modify: `docs/examples/ocr_lens_viewer.py` (whole file except `_chart`)

Stopgap: the scene-viewer effort deletes this script and moves its word mode into `collab_splats/viewer.py`. This rewrite keeps the viewer working on the codes store until then.

- [ ] **Step 1: Rewrite the viewer**

Replace the module docstring, the imports, `_probability_maps`, `_lift_onto` and `main()`. Keep `_chart` unchanged.

```python
#!/usr/bin/env python3
"""
Probe a scene's mesh for the OCR lens's words as a continuous heat overlay.

- needs mesh.ply and pointcloud.zarr under the backend dir, <scene>/semantics/ocr_lens_codes.zarr
- start-up: each frame's codes decode to its top-64 words per patch, lifted as indexed maps onto the vertices
- unseen vertices (depth test) are grey and score zero; scene terms are every observed top-10 word
- label list: words ranked by probability mass (expected vertex count); a click queries that word
- query: comma-separated words; summed probability draws as heat, faces under min p hidden
- click a vertex to chart its top-10 scene terms; --textured shows texture/mesh.obj, picks stay on mesh.ply

Usage:
    HF_HOME=/workspace/models HF_HUB_OFFLINE=1 python docs/examples/ocr_lens_viewer.py \\
        /workspace/outputs/<scene>/<backend> --port 8080
"""

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
import zarr
from PIL import Image

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.reconstructor import store_rows
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features.ocr_lens import (
    WordVocab,
    load_decoder,
    load_processor,
    word_probabilities,
    word_vocabulary,
)
from collab_splats.semantics.lifting import lift_features
from collab_splats.utils.torch_utils import get_device
from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)


@torch.no_grad()
def _top_words(
    codes: zarr.Array,
    rows: list[int],
    ae: Optional[FeatureAutoencoder],
    decoder: torch.nn.Module,
    vocab: WordVocab,
    k: int = 64,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Each pointcloud frame's codes decoded to its top-k words per patch, (ids, probs) each (k, H_p, W_p).

    - store row rows[i] for frame i; ae None when the store holds full-width states
    - held on the decoder's device, sorted by probability within each patch
    """
    maps = []

    for row in rows:
        # Patch codes as rows, (H_p * W_p, latent)
        fmap = torch.from_numpy(codes[row])
        channels, height, width = fmap.shape
        states = fmap.reshape(channels, -1)
        states = states.T

        # Word probabilities per patch; top-k kept as an indexed map over the vocabulary
        probs = torch.cat([p for p, _ in word_probabilities(states, decoder, vocab, ae=ae)])
        top = probs.topk(k, dim=1)
        ids = top.indices.T.reshape(k, height, width)
        values = top.values.T.reshape(k, height, width)
        maps.append((ids, values))

    return maps


def _lift_top_words(
    vertices: np.ndarray,
    colors: np.ndarray,
    cloud: PointcloudResult,
    maps: list[tuple[torch.Tensor, torch.Tensor]],
    n_words: int,
    chunk: int,
    k: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Top-k words per vertex from the indexed per-frame maps, chunk vertices per lift_features call.

    - vertices have no source pixel, so pixel_indices=None: no fallback, unseen rows stay zero
    - (V, k) int64 word ids and (V, k) float32 probabilities, sorted descending per row
    """
    device = get_device()
    ids, probs = [], []

    for start in range(0, len(vertices), chunk):
        # One (chunk, n_words) accumulator at a time; its top-k ranked on the GPU
        stop = start + chunk
        part = replace(cloud, points=vertices[start:stop], colors=colors[start:stop], pixel_indices=None)
        lifted = lift_features(maps.__getitem__, part, num_classes=n_words)
        top = lifted.to(device).topk(k, dim=1)
        ids.append(top.indices.cpu().numpy())
        probs.append(top.values.cpu().numpy())
        logger.info("lifted vertices %d/%d", min(stop, len(vertices)), len(vertices))

    return np.concatenate(ids), np.concatenate(probs)
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
    parser.add_argument("--chunk", type=int, default=131_072, help="vertices lifted per lift_features call")
    parser.add_argument("--model_id", default="llava-hf/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--textured", action="store_true", help="show texture/mesh.obj over the same surface")
    parser.add_argument("--texture_size", type=int, default=4096, help="displayed texture edge, pixels")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Mesh with its vertex colors; light grey when the mesh has none
    mesh = trimesh.load(args.backend_dir / "mesh.ply", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)

    if mesh.visual.kind == "vertex":
        colors = np.array(mesh.visual.vertex_colors[:, :3])
    else:
        colors = np.full((len(vertices), 3), 200, dtype=np.uint8)

    # Cameras and depth the lift tests visibility against
    cloud = PointcloudResult.load_zarr(
        args.backend_dir / "pointcloud.zarr",
        load_world_points=False,
        load_pixel_indices=False,
    )

    # Scene-level codes store; its rows follow the images/ store
    scene_dir = args.backend_dir.parent
    codes_path = scene_dir / "semantics" / "ocr_lens_codes.zarr"
    store = zarr.open(str(codes_path), mode="r")
    rows = store_rows(scene_dir / "images", cloud.image_paths)

    # The codes' AE; none when n_components was null and the store holds full-width states
    ae = None

    if store.attrs["latent_dim"] is not None:
        ae = FeatureAutoencoder.load(codes_path / "autoencoder.pt")
        ae.to(get_device())

    # Word vocabulary and lens decoder
    processor = load_processor(args.model_id)
    vocab = word_vocabulary(processor.tokenizer)
    words = vocab.words
    row_of = {word: row for row, word in enumerate(words)}
    decoder = load_decoder(args.model_id)

    # Top-64 words per patch, then per vertex; the decoder goes once the frames are decoded
    logger.info("decoding %d frames", len(rows))
    maps = _top_words(store["features"], rows, ae, decoder, vocab)
    del decoder, ae
    word_ids, word_probs = _lift_top_words(vertices, colors, cloud, maps, len(words), args.chunk)
    del maps

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
        Draw the summed probability of the query words as heat; faces under min p hidden.
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

        # Kept entries that are scene terms, already sorted by probability
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

Memory at GH010229 scale (294 frames, 545k vertices): the per-frame top-64 maps are ~1 MB each on the GPU; one `--chunk` accumulator is 131k × 3,512 fp32 = 1.8 GB, freed per chunk.

- [ ] **Step 2: Lint and import-check**

```bash
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m black --check docs/examples/ocr_lens_viewer.py; PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m isort --check docs/examples/ocr_lens_viewer.py
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python docs/examples/ocr_lens_viewer.py --help
```

Expected: both checks clean (run `black`/`isort` on this one file if not), and `--help` prints the usage with `--chunk` and `--model_id`. The viewer is exercised live in Task 9.

- [ ] **Step 3: Commit**

```bash
git add docs/examples/ocr_lens_viewer.py
git commit --only docs/examples/ocr_lens_viewer.py -m "refactor(examples): ocr_lens_viewer decodes codes to top-64 words at start-up

Reads ocr_lens_codes.zarr and its AE; each frame decodes to its top-64
words per patch, lifted as indexed maps onto the vertices in chunks. No .npy
cache, no full-vocabulary vertex array.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Docs

**Files:**
- Modify: `docs/semantics.md` (the "On-disk layout" section)
- Modify: `configs/README.md` (output trees ~lines 60 and 741, the scene-level cache note ~line 671, the not-pushed paragraph ~line 820, the layout note ~line 867)
- Modify: `configs/base.yaml:114` (`n_components` comment)
- Modify: `CLAUDE.md` (the semantics-storage in-flight entry)

- [ ] **Step 1: Rewrite the layout section of `docs/semantics.md`**

Replace everything from `## On-disk layout` up to the next `---` with:

```markdown
## On-disk layout

| store | contents | written by |
|---|---|---|
| `<scene>/semantics/<extractor>_codes.zarr` | `features` (N, latent, H_p, W_p) fp16, one chunk per frame; `autoencoder.pt`; attrs `extractor`, `patch_size`, `n_frames`, `extractor_kwargs`, `latent_dim` | `write_feature_cache` |
| `<scene>/semantics/<extractor>_states.zarr` | full-width states, (N, D, H_p, W_p) fp16; temporary, deleted once encoded | `write_feature_cache` |
| `<scene>/<backend>/semantics/<extractor>_lifted.zarr` | `features` (P, latent) fp16, `autoencoder.pt` | `write_point_features` |

- `n_components: null`: `_codes.zarr` holds full-width features, `latent_dim` null, no AE, no temporary store
- the AE trains once on every frame (streamed); changing `n_components` re-runs the extractor
- codes and lifted stores are pushed; `_states.zarr` never is (`PUSH_EXCLUDES`)
- no word probabilities on disk: `docs/examples/ocr_lens_viewer.py` decodes the codes to top-64 words per patch and lifts them onto the mesh at start-up
```

Also grep the file for `extract_feature_cache`, `load_features`, `_ae.pt` and `<extractor>.zarr`, and update every hit to the new names.

- [ ] **Step 2: `configs/README.md`**

- both output trees: `<extractor>.zarr ← 2D patch cache ...` becomes `<extractor>_codes.zarr ← fp16 2D codes + autoencoder.pt, one per extractor (backend-agnostic)`
- the cache note (~line 671) and the layout note (~line 867): `semantics/<extractor>.zarr` → `semantics/<extractor>_codes.zarr`
- the not-pushed paragraph (~line 820): replace "`/semantics/**` at the scene root (raw 2D patch maps, regenerable from frames + extractor — note the leading slash, which is what keeps `<backend>/semantics/**` in the push)" with "`/semantics/*_states.zarr/**` at the scene root (temporary full-width states, deleted once the codes are written; the codes store and `<backend>/semantics/**` are pushed)"

Grep `configs/README.md` for `<extractor>.zarr` afterwards; only the `features/<extractor>/<extractor>.zarr` history note (~line 868) may remain.

- [ ] **Step 3: `configs/base.yaml` comment**

```yaml
  n_components: 64            # AE code width stored in <extractor>_codes.zarr; null = full width; 128 for ocr_lens; changing it re-extracts
```

- [ ] **Step 4: CLAUDE.md in-flight entry**

In `/workspace/collab-splats/.worktrees/semantics-storage/CLAUDE.md`, replace the semantics-storage entry's description (between the bold name and the link group) with: `2D cache keeps fp16 AE codes only (\`<extractor>_codes.zarr\`, pushed; full states deleted after an all-frames fit); no stored words or vertex store — the ocr_lens viewer decodes codes and lifts top-64 words per patch at start-up`. Keep the link group `([spec](...) · [plan](...))`. Skip whatever the plan commit on `clean/final` already changed.

- [ ] **Step 5: Commit**

```bash
git add docs/semantics.md configs/README.md configs/base.yaml CLAUDE.md
git commit --only docs/semantics.md configs/README.md configs/base.yaml CLAUDE.md -m "docs(semantics): codes-only 2D cache, pushed; states temporary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Gate and real run on GH010229

**Files:** none (numbers are reported in chat and recorded in the CHANGELOG entry)

- [ ] **Step 1: Gate**

```bash
cd /workspace/collab-splats/.worktrees/semantics-storage
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/reconstructor tests/utils tests/remote tests/test_docstring_contract.py tests/test_import_style.py -q -p no:cacheprovider > /tmp/claude-0/ss_gate.log 2>&1; echo exit=$?
grep -E "passed|failed" /tmp/claude-0/ss_gate.log | tail -1
grep -rn "extract_feature_cache\|load_features\b\|ocr_lens_vertices" collab_splats tests evals docs --include=*.py --include=*.md | grep -v "docs/superpowers\|dashboard"
```

Expected:
- the printed path is under the worktree
- the pass count is at or above the baseline, with failures only from the baseline list
- the grep is empty; the dashboard imports were already broken (spec: out of scope)

Run `graphify update .` afterwards (per `/workspace/CLAUDE.md`).

- [ ] **Step 2: Ask before the real run**

The run writes `ocr_lens_codes.zarr` beside today's full-state `semantics/ocr_lens.zarr` (294 frames) and overwrites the backend's `ocr_lens_lifted.zarr`. Get the user's go-ahead in chat before Step 3. Keep `vggt_omega/ocr_lens_vertices.npy` until Step 5: it is the top-1 reference.

- [ ] **Step 3: Run the semantics stage in tmux**

```bash
tmux new -d -s ss_run "cd /workspace/collab-splats/.worktrees/semantics-storage && \
  PYTHONPATH=\$PWD HF_HOME=/workspace/models HF_HUB_OFFLINE=1 /usr/bin/time -v \
  /opt/venv/reconstruction/bin/python -m collab_splats local /workspace/outputs/ocr_viewer/GH010229 \
    --output-root /workspace/outputs/ocr_viewer --stages semantics --overwrite \
    --set semantics.extractor=ocr_lens --set semantics.n_components=128 \
  > /tmp/claude-0/ss_run.log 2>&1; echo exit=\$? >> /tmp/claude-0/ss_run.log"
```

- In a second tmux window, sample GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv -l 5 > /tmp/claude-0/ss_gpu.log`. Kill it by PID afterwards, never `pkill -f`.
- Wait with Monitor on `exit=` in the log. Do not run other heavy processes meanwhile (OOM risk).
- Stage times come from the log's timestamps (extract, `fit autoencoder`, encode, point lift). Peak RSS comes from `time -v` "Maximum resident set size".

If `local` rejects a processed scene dir as input, read `collab_splats/__main__.py` `_scene_name` and the leaf-stage re-run path (memory: "leaf-only `--stages` pulls from processed"), then use the documented form.

- [ ] **Step 4: Measure the stores and the viewer**

```bash
du -sh /workspace/outputs/ocr_viewer/GH010229/semantics/* /workspace/outputs/ocr_viewer/GH010229/vggt_omega/semantics/*
ls /workspace/outputs/ocr_viewer/GH010229/semantics/   # no ocr_lens_states.zarr
```

Viewer start-up: launch in tmux, with `/usr/bin/time -v` for peak RSS, and sample `nvidia-smi` as in Step 3. The viewer logs with timestamps: decode = `decoding N frames` to the first `lifted vertices` line; vertex lift = first to last `lifted vertices` line; start-up = the `date` printed at launch to the last `lifted vertices` line (the `time -v` wall clock includes serving, so it is not start-up).

```bash
date +%T.%N; tmux new -d -s ss_viewer "cd /workspace/collab-splats/.worktrees/semantics-storage && \
  PYTHONPATH=\$PWD HF_HOME=/workspace/models HF_HUB_OFFLINE=1 /usr/bin/time -v \
  /opt/venv/reconstruction/bin/python docs/examples/ocr_lens_viewer.py /workspace/outputs/ocr_viewer/GH010229/vggt_omega --port 8080 \
  > /tmp/claude-0/ss_viewer.log 2>&1"
```

Top-1 agreement against the old `.npy`: in the scratchpad, import `_top_words` and `_lift_top_words` from the viewer (add `docs/examples` to `sys.path`), run them as `main()` does, and compare:

```python
seen_new = word_probs[:, 0] > 0
old = np.load("/workspace/outputs/ocr_viewer/GH010229/vggt_omega/ocr_lens_vertices.npy", mmap_mode="r")
top_old = np.concatenate([np.asarray(old[s:s + 65536]).argmax(1) for s in range(0, len(old), 65536)])
seen_old = np.concatenate([np.asarray(old[s:s + 65536]).any(1) for s in range(0, len(old), 65536)])
both = seen_new & seen_old
print("observed old/new/both", seen_old.sum(), seen_new.sum(), both.sum())
print("top-1 agreement", (word_ids[both, 0] == top_old[both]).mean())
```

Then open the viewer:
- click a vertex
- query `tree`
- check that the label list ranks

- [ ] **Step 5: Report**

Post a table in chat, with no thresholds:
- disk per store, before vs after
- stage times: extract, AE fit, encode, point lift
- peak GPU and RSS, stage and viewer
- viewer start-up: decode, vertex lift
- observed counts
- top-1 agreement

- [ ] **Step 6: Delete the old stores, asked first**

These are the pre-B files, now unread by any code. Ask the user in chat, listing each path with its `du -sh`, and delete only after a yes:
- `/workspace/outputs/ocr_viewer/GH010229/semantics/ocr_lens.zarr`
- `/workspace/outputs/ocr_viewer/GH010229/vggt_omega/ocr_lens_vertices.npy`
- `/workspace/outputs/ocr_viewer/GH010229/vggt_omega/semantics/ocr_lens_ae.pt`
- the same three paths under `/workspace/outputs/2026_07_15-Goprosplat-GH010229/` (that scene has no mesh, so it may lack the `.npy`; its semantics re-run when needed)

`ls` each path first; report any that is missing rather than guessing another name.

- [ ] **Step 7: Changelog and in-flight entry**

Append a `semantics-storage` entry to `docs/superpowers/CHANGELOG.md` with the Step 5 numbers and the branch tip. Remove the semantics-storage line from CLAUDE.md "In-Flight Work" and add it to "Recently Completed" (five newest only). Commit with `git add -f docs/superpowers/CHANGELOG.md` and `git commit --only docs/superpowers/CHANGELOG.md CLAUDE.md`. Merging to `clean/final` is the user's call; ask.

Tell the scene-viewer session (`ListAgents`, then `SendMessage`) that no `*_vertices.zarr` exists: codes stores are pushed and vertices lift from them at start-up.
