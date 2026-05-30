# XFeat Matching Fix Design

## Goal

Fix `XFeatExtractor` to use canonical XFeat+LighterGlue matching instead of `kornia.match_mnn`. Unify the `extract()` / `match()` interface across `DiskExtractor` and `XFeatExtractor`.

## Context

The original implementation of `XFeatExtractor` has three bugs:

1. **`extract()` drops `scores`** — `detectAndCompute` returns `{'keypoints', 'scores', 'descriptors'}` but only `kpts, descs` were kept. `match_lighterglue` requires `scores` in its input dict.
2. **`match()` uses `kornia.match_mnn`** — brute-force MNN, not XFeat's learned LighterGlue matcher. The docstring says "XFeat+MNN" but the canonical notebook and design doc both specify XFeat+LighterGlue via `xfeat.match_lighterglue(d0, d1)`.
3. **`image_hw` ignored in `match()`** — `match_lighterglue` requires `image_size: (W, H)` in the input dict; the parameter was accepted but unused.

`DiskExtractor` is correct but drops `detection_scores` from its `extract()` output, creating an asymmetric interface. It should return 3-tuples for parity.

## Non-Goals

- Changing `CameraLocalizer`'s PnP logic or the `_build_frame_assignments` helper
- Adding per-match confidence deduplication (score=1.0 uniform is acceptable; `match_lighterglue` doesn't expose per-match scores in its public API)
- Supporting `match_xfeat_star` (requires raw images; incompatible with cached-features design)

## Architecture

### Unified extractor interface

Both extractors adopt a 3-tuple return from `extract()`:

```python
extract(image: np.ndarray) -> tuple[Tensor (N,2), Tensor (N,), Tensor (N,D)]
#                                   keypoints      scores        descriptors
```

Both `match()` methods gain `scores_q, scores_db` parameters. `DiskExtractor.match()` ignores them (LightGlue doesn't need pre-extracted scores). `XFeatExtractor.match()` uses them to build the dict for `match_lighterglue`.

```python
match(kpts_q, scores_q, descs_q, kpts_db, scores_db, descs_db, image_hw) -> Tensor (K,2) int64
```

`CameraLocalizer` stores 3-tuples in `self._frame_features` and unpacks all three when calling `match()`. The `_build_frame_assignments` call is unchanged — it only uses `f[0]` (keypoints).

### `DiskExtractor` changes

`extract()` adds `scores = features[0].detection_scores.cpu()` to the return tuple. `match()` gains `scores_q, scores_db` keyword args (unused, accepted for interface parity).

### `XFeatExtractor` changes

`extract()` returns `(kpts, scores, descs)` all from `detectAndCompute` output.

`match()` builds XFeat input dicts and calls `xfeat.match_lighterglue`:

```python
d0 = {
    'keypoints': kpts_q.unsqueeze(0).to(self._device),
    'scores': scores_q.to(self._device),
    'descriptors': descs_q.unsqueeze(0).to(self._device),
    'image_size': (image_hw[1], image_hw[0]),  # (W, H)
}
d1 = { ... same for db ... }
_, _, idx = self._xfeat.match_lighterglue(d0, d1)
return torch.from_numpy(idx).long()  # (K, 2) int64
```

`image_hw` is now **required** (not optional) for `XFeatExtractor.match()`, same as `DiskExtractor`.

### `CameraLocalizer` changes

`self._frame_features` stores `list[tuple[Tensor, Tensor, Tensor]]` (kpts, scores, descs).

`localize()` unpacks:
```python
kpts_q, scores_q, descs_q = self._extractor.extract(query_image)
...
for i, (kpts_db, scores_db, descs_db) in enumerate(self._frame_features):
    matches = self._extractor.match(
        kpts_q, scores_q, descs_q,
        kpts_db, scores_db, descs_db,
        self._image_hw,
    )
```

The `assignment` lookup and PnP logic are unchanged.

## File structure

Single file change: `collab_splats/pointcloud/localization.py`

Test file: `tests/pointcloud/test_localization.py` — update `test_xfeat_extractor_returns_keypoints_and_descriptors` and `test_xfeat_extractor_match_returns_index_pairs` for 3-tuple; update `test_disk_extractor_*` similarly.

## Testing

- `test_disk_extractor_returns_keypoints_and_descriptors` — assert 3-tuple, scores shape `(N,)`
- `test_xfeat_extractor_returns_keypoints_and_descriptors` — assert 3-tuple, scores shape `(N,)`
- `test_xfeat_extractor_match_returns_index_pairs` — assert `(K,2)` int64, `image_hw` now required arg
- `test_camera_localizer_recovers_known_pose` — no change expected (synthetic geometry test)
