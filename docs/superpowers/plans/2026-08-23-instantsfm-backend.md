# InstantSfM Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate InstantSfM (global SfM, IROS 2026) as `pointcloud.method: sfm` / `backend: instantsfm`, fed by Video Depth Anything metric depth, emitting the unified `pointcloud.zarr` artifact that every downstream stage (splats, mesh, semantics, localize, verify) consumes.

**Architecture:** A new `InstantSfMCreator` plus a `generate_vda_depth` helper — both in `collab_splats/pointcloud/sfm.py` — drive InstantSfM through its Python API (`ReadData → GenerateDatabase → ReadColmapDatabase → Config → ReadDepthsIntoFeatures → SolveGlobalMapper → WriteGlomapReconstruction`). VDA itself lives in a `third_party/Video-Depth-Anything` clone (not pip-installable; sys.path shim at call time). `Reconstructor._run_sfm` orchestrates: stage images → VDA depth → creator → build a `FeedforwardResult` (dataclass container reused; artifact renamed) → save `pointcloud.zarr` with provenance attrs. **The artifact rename is a hard cutover: `feedforward.zarr` is replaced by `pointcloud.zarr` everywhere — no resolver, no fallback.** Legacy scenes migrate with a one-line rename (documented in Task 8), or reprocess.

**Tech Stack:** InstantSfM v0.3.0 (`cre185/InstantSfM` @ `d3e599e1a42b4c5a806a84d9f383e1005d25f61b`, CC-BY-NC-4.0 — user cleared; pin lives ONLY in setup.sh), Video-Depth-Anything (metric, vitl), pycolmap, zarr v3, torch 2.5.1+cu121, py3.11 (`/opt/venv/reconstruction/bin/python`).

**Spec:** `docs/superpowers/specs/2026-08-23-instantsfm-backend-design.md`

**Conventions for this plan:**
- Python = `/opt/venv/reconstruction/bin/python` always.
- Commits: `git add -f` may be needed for `docs/superpowers/`; use `git commit --only <paths>` (concurrent sessions share the index — never bare `git commit -a`).
- Line numbers cited are as of branch tip `6204e0b9` — verify with the shown `old_string` context before editing; a concurrent session may have shifted them.
- Tests: flat functions, `tests/` mirrors `collab_splats/`.

**Spec deviations locked in during planning (verified against installed v0.3.0):**
1. `features` allowlist is `{"colmap"}` only. Installed `GenerateDatabase` ignores `feature_handler_name` (always subprocesses the system `colmap` binary: SIMPLE_RADIAL + exhaustive matcher, CPU SIFT) and `Config` maps only `'colmap'`. The config key stays so future upstream values slot in without config migration.
2. `pixel_indices` is synthesized from COLMAP track observations (first observation per point3D, keypoint xy scaled to depth res) — `lift_features` hard-requires it.
3. `Config` aliasing trap: `Config.__init__` sets `self.OPTIONS = GENERAL_OPTIONS` / `self.RUNTIME_OPTIONS = RUNTIME_OPTIONS` — **module-level dict aliases**. The creator must copy both dicts before mutating, or `use_depths=True` leaks into every later `Config` in-process (breaks the eval `instantsfm_nodepth` ablation).
4. (Review feedback) No resolver/fallback layer: `pointcloud.zarr` fully replaces `feedforward.zarr`. No new `paths.py` / `vda.py` modules. No hardcoded upstream-commit constant — provenance version read from installed package metadata.

---

## File Structure

```
collab_splats/pointcloud/
  sfm.py                   # MODIFY — add generate_vda_depth + InstantSfMCreator + _pixel_indices_from_reconstruction
  feedforward/base.py      # MODIFY — save_zarr(extra_attrs=...)
  utils.py                 # MODIFY — lift_features tolerates confidence=None
collab_splats/mesh/utils.py        # MODIFY — absent-confidence: log-and-skip, not raise
collab_splats/wrapper/reconstructor.py  # MODIFY — rename artifact, _run_sfm, validate_config,
                                        #          refine_poses refusal, splats log
collab_splats/dashboard/pipeline.py     # MODIFY — rename artifact
collab_splats/dashboard/app.py          # MODIFY — rename artifact
collab_splats/remote/sources.py         # MODIFY — rename artifact in excludes + helpers
configs/base.yaml                       # MODIFY — pointcloud.instantsfm block
configs/README.md                       # MODIFY — pointcloud.zarr contract + migration
pyproject.toml                          # MODIFY — comment: instantsfm deps via setup.sh
setup.sh                                # MODIFY — instantsfm deps + VDA clone + checkpoint
evals/scripts/eval.py                   # MODIFY — instantsfm / instantsfm_nodepth conditions
tests/pointcloud/test_zarr_attrs.py     # NEW
tests/pointcloud/test_instantsfm.py     # NEW — VDA helper + pixel indices + config copy
tests/wrapper/test_sfm_config.py        # NEW
tests/mesh/test_absent_confidence.py    # NEW
```

---

### Task 1: `save_zarr(extra_attrs=...)`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py` (`FeedforwardResult.save_zarr`)
- Test: `tests/pointcloud/test_zarr_attrs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pointcloud/test_zarr_attrs.py
import numpy as np
import zarr

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _tiny_result():
    n, h, w = 2, 4, 6
    return FeedforwardResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        image_paths=[f"frame_{i:06d}.jpg" for i in range(n)],
        original_coords=np.zeros((n, 6), dtype=np.float32),
        model_width=w,
        model_height=h,
    )


def test_save_zarr_writes_extra_attrs(tmp_path):
    out = tmp_path / "pointcloud.zarr"
    _tiny_result().save_zarr(
        out, extra_attrs={"method": "sfm", "backend": "instantsfm", "instantsfm_version": "0.3.0"}
    )
    store = zarr.open(str(out), mode="r")
    assert store.attrs["method"] == "sfm"
    assert store.attrs["backend"] == "instantsfm"
    assert store.attrs["instantsfm_version"] == "0.3.0"


def test_save_zarr_no_extra_attrs_unchanged(tmp_path):
    out = tmp_path / "pointcloud.zarr"
    _tiny_result().save_zarr(out)
    store = zarr.open(str(out), mode="r")
    assert "method" not in store.attrs
    assert store.attrs["model_width"] == 6
```

Note: if `FeedforwardResult` requires more constructor args than shown, extend `_tiny_result()` to match the actual dataclass signature (all optional fields default to None) — do NOT change the dataclass.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_zarr_attrs.py -v`
Expected: FAIL — `TypeError: save_zarr() got an unexpected keyword argument 'extra_attrs'`

- [ ] **Step 3: Add the parameter**

In `collab_splats/pointcloud/feedforward/base.py`, change the `save_zarr` signature from:

```python
    def save_zarr(self, path: Path) -> None:
```

to:

```python
    def save_zarr(self, path: Path, extra_attrs: dict | None = None) -> None:
```

Add to the docstring bullet list: `- extra_attrs: optional provenance attrs (method, backend, upstream version) merged into store.attrs.`

Directly after the existing block that writes `store.attrs["image_paths"]` / `"model_width"` / `"model_height"`, add:

```python
        # Provenance attrs — which method/backend produced this artifact
        if extra_attrs:
            for key, value in extra_attrs.items():
                store.attrs[key] = value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_zarr_attrs.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/pointcloud/test_zarr_attrs.py collab_splats/pointcloud/feedforward/base.py
git commit --only tests/pointcloud/test_zarr_attrs.py collab_splats/pointcloud/feedforward/base.py -m "feat(pointcloud): save_zarr provenance attrs (method, backend, upstream version)"
```

---

### Task 2: Hard rename `feedforward.zarr` → `pointcloud.zarr` (whole repo)

No resolver, no fallback — every literal changes. Legacy scenes migrate by rename (Task 8 documents the commands) or reprocess.

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (~12 sites)
- Modify: `collab_splats/dashboard/pipeline.py`, `collab_splats/dashboard/app.py`
- Modify: `collab_splats/remote/sources.py`
- Modify: existing tests under `tests/` that use the old literal

- [ ] **Step 1: Add the property (single source of the path)**

In `reconstructor.py`, next to the existing properties (`backend_dir` / `frames_zarr`, ~line 704-721):

```python
    @property
    def pointcloud_zarr(self) -> Path:
        """
        Unified reconstruction zarr for this backend (all pointcloud methods).
        """
        return self.backend_dir / "pointcloud.zarr"
```

- [ ] **Step 2: Convert every reconstructor site**

Replace each `self.backend_dir / "feedforward.zarr"` (and local `feedforward_zarr = ...` constructions from it) with `self.pointcloud_zarr`. Sites as of `6204e0b9` (grep to confirm none are missed):

| ~Line | Method |
|---|---|
| 347-354 | `_run_feedforward` write site (see Step 3) |
| 957 | `refine_poses` |
| 1049 | `extract_semantics` |
| 1092 | `mesh` |
| 1130 | `build_localization_db` |
| 1186, 1197 | `verify` |
| 1267 | `splats` depth-targets |
| 1348 | `reconstruction_quality_report` |
| 1362-1364 | `_stage_output_exists` "pointcloud" |
| 1378-1381 | `_stage_output_exists` "localize" |

Rename local variables `feedforward_zarr` → `pointcloud_zarr` in the methods you touch, and the corresponding params of the private helpers they feed (`_lift_and_save`, `_run_tsdf_mesh`, `_localization_db_exists`, `_build_localization_db`). Private helpers only — no public API rename. Update docstrings/log strings mentioning the old name.

- [ ] **Step 3: Feedforward write site gets provenance attrs**

In `_run_feedforward` (~line 347-354), find:

```python
            zarr_path = output_dir / "feedforward.zarr"
            ff_outputs.save_zarr(zarr_path)
```

replace with:

```python
            zarr_path = output_dir / "pointcloud.zarr"
            ff_outputs.save_zarr(
                zarr_path,
                extra_attrs={"method": "feedforward", "backend": self.config["pointcloud"]["backend"]},
            )
```

(Adapt the config access to whatever variable already holds the backend name in that scope — `_run_feedforward` may have a `backend` local; use it if present.)

- [ ] **Step 4: dashboard/pipeline.py**

Every `out_dir / "feedforward.zarr"` → `out_dir / "pointcloud.zarr"`:
- write site (~477): `result.save_zarr(out_dir / "pointcloud.zarr", extra_attrs={"method": "feedforward", "backend": backend})` (`backend` = the name already in scope; grep for the variable used to build `out_dir`)
- `_load_feedforward_result` (~552), browse existence check (~692), `read_localized_group` (~696), localize pull check (~726), `_build_localizer` (~741-747), `_stamp_db_provenance`, `zarr_path=` (~779)

- [ ] **Step 5: dashboard/app.py**

Same literal rename at ~417, ~476, ~596-608 (including the step label `f"{scene}: reading feedforward.zarr"` → `"... reading pointcloud.zarr"`), ~751, ~790.

- [ ] **Step 6: remote/sources.py**

- `PULL_EXCLUDES` (~28-40): rename each `feedforward.zarr/<member>/**` entry to `pointcloud.zarr/<member>/**` (same member list, no additions).
- `list_localization_dbs` (~341): `path = f"{scene}/pointcloud.zarr/local_features"`.
- `pull_zarr_members` (~405-420): `dest = Path(dest_dir) / "pointcloud.zarr"`, `remote = f"{self._remote}:{PROCESSED_BUCKET}/{scene}/pointcloud.zarr"`; update the docstring.
- `PUSH_EXCLUDES`: unchanged — `/*/colmap/database.db` anchor already covers the InstantSfM database (Task 5 moves it under `colmap/`).

- [ ] **Step 7: Grep gate + fix tests**

```bash
grep -rn 'feedforward\.zarr' collab_splats/ tests/ docs/examples/
```

Expected in `collab_splats/`: **zero matches**. In `tests/` and `docs/examples/`: update every fixture/assertion to the new name (the write path IS the contract — updating them is the point, not a regression). Do not touch `docs/superpowers/` history or notebooks in this task; notebooks that break are recorded in Task 8's migration note.

- [ ] **Step 8: Run suites + smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ tests/pointcloud/ tests/dashboard/ tests/remote/ tests/localization/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```
Expected: PASS; `SMOKE PASS`.

- [ ] **Step 9: Commit**

```bash
git add collab_splats/ tests/ docs/examples/
git commit --only collab_splats tests docs/examples -m "refactor!: pointcloud.zarr replaces feedforward.zarr (hard rename, no fallback)

Legacy scenes: mv <backend>/feedforward.zarr <backend>/pointcloud.zarr (local),
rclone moveto for environments-processed, or reprocess."
```

---

### Task 3: Absent-confidence seams (mesh, lift_features, splats log)

`pointcloud.zarr` from instantsfm has **no confidence array** (absent, never zeros). Three seams must tolerate that.

**Files:**
- Modify: `collab_splats/mesh/utils.py:555-560`
- Modify: `collab_splats/pointcloud/utils.py:774-778` + confidence tensor block ~801-805
- Modify: `collab_splats/wrapper/reconstructor.py` splats depth-targets block (~1288)
- Test: `tests/mesh/test_absent_confidence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/mesh/test_absent_confidence.py
import numpy as np
import pytest

from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _result_no_confidence():
    n, h, w = 2, 8, 8
    return FeedforwardResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32) * 8, (n, 1, 1)),
        image_paths=[f"frame_{i:06d}.jpg" for i in range(n)],
        original_coords=np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32),
        model_width=w,
        model_height=h,
        images=np.zeros((n, 3, h, w), dtype=np.float32),
        depth=np.ones((n, h, w), dtype=np.float32),
    )


def test_tsdf_inputs_skip_masking_when_confidence_absent(caplog):
    # conf_percentile set + no confidence -> proceed unmasked with a log, not ValueError
    result = _result_no_confidence()
    with caplog.at_level("INFO"):
        out = _feedforward_to_tsdf_inputs(result, conf_percentile=20)
    assert out is not None
    assert any("no confidence" in r.message for r in caplog.records)


def test_lift_features_uniform_weights_when_confidence_absent():
    import torch

    from collab_splats.pointcloud.utils import lift_features

    result = _result_no_confidence()
    result.pixel_indices = np.zeros((5, 3), dtype=np.int32)
    fmaps = [torch.zeros(4, 8, 8) for _ in range(2)]
    feats = lift_features(fmaps, result)  # must not assert on confidence
    assert feats.shape == (5, 4)
```

(Adjust `_feedforward_to_tsdf_inputs`'s actual signature if it takes `frame_store`/`native_intrinsics` positionally — pass the defaults `None`. If `FeedforwardResult` needs more required args, mirror Task 1's note.)

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_absent_confidence.py -v`
Expected: FAIL — first test raises `ValueError("conf_percentile is set but this result has no confidence — ...")`, second trips the `lift_features` assert.

- [ ] **Step 3: Fix mesh/utils.py**

Replace lines 555-560's raise with a log-and-skip (`mesh/utils.py`):

```python
    if conf_percentile is not None:
        if result.confidence is None:
            # SfM-derived results carry no per-pixel confidence — fuse unmasked rather
            # than fail; the percentile gate only applies when the model produced one.
            logger.info(
                "conf_percentile=%s set but result has no confidence — fusing unmasked",
                conf_percentile,
            )
        else:
            conf = result.confidence
            if hasattr(conf, "numpy"):
                conf = conf.detach().cpu().numpy()
            dropped = ~confidence_mask(conf, conf_percentile)
            depths = depths.copy()  # copy only when mutating — the default path fuses read-only
            depths[dropped] = 0.0
            logger.info(
                "Confidence mask (p%.0f): %.1f%% of depth pixels dropped",
                conf_percentile,
                100.0 * float(dropped.mean()),
            )
```

- [ ] **Step 4: Fix lift_features**

In `collab_splats/pointcloud/utils.py`:

Required-fields loop (774-778): drop `"confidence"` from the tuple:

```python
    for name in ("points", "pixel_indices", "depth", "extrinsics", "intrinsics"):
```

Confidence tensor block (~801-805) becomes:

```python
    # Conf → (N, H, W) float32 on device; absent confidence (SfM results) = uniform weights
    if result.confidence is None:
        conf = torch.ones((N, H, W), dtype=torch.float32, device=device)
    else:
        conf_np = (
            result.confidence.detach().cpu().numpy()
            if isinstance(result.confidence, torch.Tensor)
            else result.confidence
        )
        conf = torch.as_tensor(np.ascontiguousarray(conf_np), dtype=torch.float32, device=device)
```

**Ordering note:** `H, W` (line 782) and `device` (789) and `N` (779) all precede this block — no reordering needed. Update the docstring: `confidence` moves from required to "optional; absent → uniform visibility weights". Weight line ~833 needs no change (uniform conf = pure visibility mask).

- [ ] **Step 5: splats log line**

In `reconstructor.py` splats depth-targets block (~1288), the masking branch already guards `if conf_percentile is not None and feedforward.confidence is not None:` — add an `else`-side log so the skip is visible:

```python
            if conf_percentile is not None and feedforward.confidence is not None:
                ...  # existing masking body unchanged
            elif conf_percentile is not None:
                logger.info("splats depth targets: no confidence in zarr — using unmasked depth")
```

- [ ] **Step 6: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/ tests/pointcloud/ -q`
Expected: PASS (including the two new tests).

- [ ] **Step 7: Commit**

```bash
git add tests/mesh/test_absent_confidence.py collab_splats/mesh/utils.py collab_splats/pointcloud/utils.py collab_splats/wrapper/reconstructor.py
git commit --only tests/mesh/test_absent_confidence.py collab_splats/mesh/utils.py collab_splats/pointcloud/utils.py collab_splats/wrapper/reconstructor.py -m "feat(mesh,semantics,splats): tolerate absent confidence — SfM results carry none"
```

---

### Task 4: Config surface + validation

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:52-54` (`_SFM_BACKENDS`), `validate_config` (~677-696), `refine_poses` (~928)
- Modify: `configs/base.yaml` pointcloud block
- Test: `tests/wrapper/test_sfm_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/wrapper/test_sfm_config.py
import pytest
import yaml

from collab_splats.wrapper.reconstructor import _SFM_BACKENDS, Reconstructor


def _base_config():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


def test_instantsfm_is_valid_sfm_backend():
    assert "instantsfm" in _SFM_BACKENDS


def test_sfm_instantsfm_config_validates():
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "instantsfm"
    Reconstructor.validate_config(cfg)  # must not raise


def test_sfm_rejects_bundle_adjustment():
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "instantsfm"
    cfg["pointcloud"]["bundle_adjustment"] = True
    with pytest.raises(ValueError, match="bundle_adjustment"):
        Reconstructor.validate_config(cfg)


def test_instantsfm_features_allowlist():
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "instantsfm"
    cfg["pointcloud"]["instantsfm"]["features"] = "superglue"
    with pytest.raises(ValueError, match="features"):
        Reconstructor.validate_config(cfg)


def test_base_yaml_has_instantsfm_block():
    cfg = _base_config()
    assert cfg["pointcloud"]["instantsfm"] == {"features": "colmap"}
```

(If `validate_config` is an instance method rather than static/class, adapt calls to how existing `tests/wrapper/` config tests invoke it — copy their pattern.)

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_config.py -v`
Expected: FAIL — `"instantsfm" not in _SFM_BACKENDS`, missing yaml block, etc.

- [ ] **Step 3: Implement**

`reconstructor.py:53`:

```python
_SFM_BACKENDS = {"colmap", "hloc", "instantsfm"}
```

New module constant next to it:

```python
_INSTANTSFM_FEATURES = {"colmap"}  # installed v0.3.0 GenerateDatabase ignores the handler name; key kept for future upstream values
```

In `validate_config`, after the existing backend-membership checks (~686) and near the BA×LC exclusion (~690-696), add:

```python
        # SfM path: BA re-refinement is InstantSfM's own job — refuse the flag
        if method == "sfm" and pointcloud_cfg.get("bundle_adjustment", False):
            raise ValueError(
                "pointcloud.bundle_adjustment is not supported with method: sfm — "
                "InstantSfM runs its own global bundle adjustment"
            )

        # InstantSfM feature-handler allowlist (v0.3.0 supports only colmap)
        if method == "sfm" and backend == "instantsfm":
            features = pointcloud_cfg.get("instantsfm", {}).get("features", "colmap")
            if features not in _INSTANTSFM_FEATURES:
                raise ValueError(
                    f"pointcloud.instantsfm.features={features!r} not in {sorted(_INSTANTSFM_FEATURES)}"
                )
```

(Use the same local variable names the surrounding code uses for the pointcloud config dict / method / backend.)

`configs/base.yaml` pointcloud block — update the method comment and add the sub-block (keep every existing key untouched):

```yaml
pointcloud:
  method: feedforward  # feedforward | sfm
  backend: vggt_omega  # feedforward: vggtx|mapanything|vggt_omega|loger; sfm: instantsfm (colmap/hloc not implemented)
  # ... existing keys unchanged ...
  instantsfm:
    features: colmap  # feature/matching handler; v0.3.0 upstream supports only colmap (CPU SIFT + exhaustive)
```

`refine_poses` (~928) — at the top, before any work:

```python
        # SfM results are already globally bundle-adjusted; LM re-refinement is undefined here
        if self.config["pointcloud"]["method"] == "sfm":
            raise ValueError("refine_poses is not supported for pointcloud.method: sfm")
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_config.py tests/wrapper/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/wrapper/test_sfm_config.py collab_splats/wrapper/reconstructor.py configs/base.yaml
git commit --only tests/wrapper/test_sfm_config.py collab_splats/wrapper/reconstructor.py configs/base.yaml -m "feat(config): instantsfm sfm backend — allowlist, BA rejection, refine refusal"
```

---

### Task 5: `generate_vda_depth` + `InstantSfMCreator` in `pointcloud/sfm.py`

One file gains both: the VDA helper exists only to feed InstantSfM depth, so it lives beside the creator — not as its own module. VDA code itself stays in the `third_party/` clone.

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py` (ColmapCreator/HlocCreator untouched)
- Test: `tests/pointcloud/test_instantsfm.py`

- [ ] **Step 1: Write the failing tests (no GPU, no installs needed except the last, which skips)**

```python
# tests/pointcloud/test_instantsfm.py
import numpy as np
import pytest

from collab_splats.pointcloud import sfm


def test_vda_missing_clone_raises_actionable_import_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match="setup.sh"):
        sfm.generate_vda_depth(frames, fps=30.0, out_dir=tmp_path)


def test_vda_skips_when_depths_exist(tmp_path):
    out = tmp_path / "depth_vda"
    out.mkdir()
    np.savez_compressed(out / "depths.npz", depths=np.ones((2, 4, 4), dtype=np.float32))
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    # Existing depths.npz -> returned untouched, no VDA import attempted
    path = sfm.generate_vda_depth(frames, fps=30.0, out_dir=tmp_path)
    assert path == out / "depths.npz"


def test_pixel_indices_from_reconstruction_scales_to_depth_res():
    # Minimal stand-in for pycolmap objects: 1 point3D observed in frame 0 at
    # original-res keypoint (200, 100); original 800x600 -> depth 80x60 = scale 0.1
    class _Elem:
        image_id, point2D_idx = 1, 0

    class _Track:
        elements = [_Elem()]

    class _P3D:
        track = _Track()

    class _Pt2D:
        xy = np.array([200.0, 100.0])

    class _Img:
        name = "frame_000000.jpg"
        points2D = [_Pt2D()]

    class _Recon:
        points3D = {7: _P3D()}
        images = {1: _Img()}

    idx = sfm._pixel_indices_from_reconstruction(
        _Recon(), point3d_ids=[7], name_to_row={"frame_000000.jpg": 0},
        scale_x=0.1, scale_y=0.1, depth_hw=(60, 80),
    )
    assert idx.shape == (1, 3)
    assert idx.dtype == np.int32
    assert idx.tolist() == [[0, 10, 20]]  # [frame_row, row=y*0.1, col=x*0.1]


def test_pixel_indices_clamped_to_grid():
    class _Elem:
        image_id, point2D_idx = 1, 0

    class _Track:
        elements = [_Elem()]

    class _P3D:
        track = _Track()

    class _Pt2D:
        xy = np.array([799.9, 599.9])  # edge keypoint -> scaled index must stay in-grid

    class _Img:
        name = "frame_000000.jpg"
        points2D = [_Pt2D()]

    class _Recon:
        points3D = {7: _P3D()}
        images = {1: _Img()}

    idx = sfm._pixel_indices_from_reconstruction(
        _Recon(), point3d_ids=[7], name_to_row={"frame_000000.jpg": 0},
        scale_x=0.1, scale_y=0.1, depth_hw=(60, 80),
    )
    assert 0 <= idx[0, 1] < 60 and 0 <= idx[0, 2] < 80


def test_creator_config_copy_prevents_module_dict_leak():
    pytest.importorskip("instantsfm")
    from instantsfm.controllers.config import RUNTIME_OPTIONS

    before = dict(RUNTIME_OPTIONS)
    cfg = sfm.InstantSfMCreator(features="colmap")._build_config()
    cfg.RUNTIME_OPTIONS["use_depths"] = True
    # Module-level dict must be untouched — Config aliases it; creator must copy
    assert RUNTIME_OPTIONS == before
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_instantsfm.py -v`
Expected: FAIL — `AttributeError: module 'collab_splats.pointcloud.sfm' has no attribute 'generate_vda_depth'`

- [ ] **Step 3: Implement in `collab_splats/pointcloud/sfm.py`**

Append after the existing creators. File-top imports to ensure present (top of file, not inline): `logging`, `shutil`, `sys`, `numpy as np`, `pycolmap`, `dataclass`, `Path`; `logger = logging.getLogger(__name__)` if missing. `torch` and `instantsfm` imports stay lazy inside functions (optional heavy deps, per code style exception).

```python
########################################################################
# Video Depth Anything — metric depth for the SfM path
########################################################################

# Repo root -> third_party clone (setup.sh owns creation); module-level so tests can monkeypatch
VDA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Video-Depth-Anything"
VDA_CHECKPOINT = "metric_video_depth_anything_vitl.pth"


def generate_vda_depth(
    frames: np.ndarray,
    fps: float,
    out_dir: Path,
    *,
    encoder: str = "vitl",
    input_size: int = 518,
    device: str = "cuda",
) -> Path:
    """
    Run Video Depth Anything metric depth over keyframes; write InstantSfM's depth layout.

    - frames: (N, H, W, 3) uint8 RGB (frames.zarr order).
    - fps: effective keyframe rate — VDA is temporal.
    - out_dir: parent dir; depths land at out_dir/depth_vda/depths.npz key 'depths'
      (the exact layout instantsfm.ReadDepths consumes).
    - Returns the depths.npz path. Skips inference when it already exists.

    Attribution: inference pattern follows
    https://github.com/DepthAnything/Video-Depth-Anything metric_depth/run.py.
    """
    depth_dir = Path(out_dir) / "depth_vda"
    npz_path = depth_dir / "depths.npz"

    # Idempotent: an existing depth archive is authoritative (overwrite = delete upstream)
    if npz_path.exists():
        logger.info("VDA depth exists at %s — skipping inference", npz_path)
        return npz_path

    # Lazy heavy import — VDA lives in a third_party clone, not site-packages
    metric_dir = VDA_ROOT / "metric_depth"
    if not metric_dir.exists():
        raise ImportError(
            f"Video-Depth-Anything clone not found at {VDA_ROOT} — run setup.sh "
            "(clones the repo and downloads the metric vitl checkpoint)"
        )
    if str(metric_dir) not in sys.path:
        sys.path.insert(0, str(metric_dir))
    import torch
    from video_depth_anything.video_depth import VideoDepthAnything

    ckpt = VDA_ROOT / "checkpoints" / VDA_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"VDA metric checkpoint missing: {ckpt} — run setup.sh")

    # Upstream model_configs table (metric_depth/run.py)
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = VideoDepthAnything(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    # Metric inference over the whole keyframe sequence
    logger.info("VDA metric inference: %d frames @ %.2f fps (encoder=%s)", len(frames), fps, encoder)
    depths, _fps = model.infer_video_depth(frames, fps, input_size=input_size, device=device, fp32=False)

    depth_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, depths=np.asarray(depths, dtype=np.float32))
    logger.info("VDA depths written: %s shape=%s", npz_path, np.asarray(depths).shape)
    return npz_path


########################################################################
# InstantSfM
########################################################################


def _pixel_indices_from_reconstruction(
    recon,
    point3d_ids: "list[int]",
    name_to_row: "dict[str, int]",
    scale_x: float,
    scale_y: float,
    depth_hw: "tuple[int, int]",
) -> np.ndarray:
    """
    Synthesize (P, 3) int32 [frame_row, row, col] pixel indices from COLMAP tracks.

    - First track observation per point3D; keypoint xy is original-res, scaled to
      the depth grid and clamped in-bounds.
    - lift_features requires pixel_indices; SfM results have no dense source pixel,
      so the observing keypoint is the honest substitute.
    """
    h, w = depth_hw
    out = np.zeros((len(point3d_ids), 3), dtype=np.int32)

    for i, pid in enumerate(point3d_ids):
        elem = recon.points3D[pid].track.elements[0]
        image = recon.images[elem.image_id]
        xy = image.points2D[elem.point2D_idx].xy
        col = min(max(int(xy[0] * scale_x), 0), w - 1)
        row = min(max(int(xy[1] * scale_y), 0), h - 1)
        out[i] = (name_to_row[image.name], row, col)

    return out


@dataclass
class InstantSfMCreator:
    """
    Global SfM via InstantSfM (https://github.com/cre185/InstantSfM, IROS 2026).

    License: CC-BY-NC-4.0 (non-commercial) — cleared for this repo's research use;
    revisit before any commercial deployment. Install pinned in setup.sh.

    Drives the upstream Python API directly (never their CLI): ReadData ->
    GenerateDatabase (system colmap binary, CPU SIFT, exhaustive) ->
    ReadColmapDatabase -> Config -> ReadDepthsIntoFeatures (VDA metric depth) ->
    SolveGlobalMapper -> WriteGlomapReconstruction. Call pattern follows
    instantsfm/scripts/sfm.py::run_sfm at the installed version.
    """

    features: str = "colmap"
    single_camera: bool = True
    use_depths: bool = True

    def _build_config(self):
        """
        Upstream Config with OPTIONS/RUNTIME_OPTIONS copied — Config.__init__
        aliases module-level dicts, so in-place mutation leaks across instances.
        """
        from instantsfm.controllers.config import Config

        config = Config(self.features)
        config.OPTIONS = dict(config.OPTIONS)
        config.RUNTIME_OPTIONS = dict(config.RUNTIME_OPTIONS)
        return config

    def reconstruct(self, data_dir: Path) -> "pycolmap.Reconstruction":
        """
        Run InstantSfM over data_dir (must hold images/; depth_vda/depths.npz when use_depths).

        - Writes COLMAP binary to data_dir/colmap/sparse/0 and moves the SIFT DB
          to data_dir/colmap/database.db (contract layout; GCS push excludes the DB).
        - Returns the pycolmap.Reconstruction read back from the written model.
        """
        from instantsfm.controllers.data_reader import (
            ReadColmapDatabase,
            ReadData,
            ReadDepthsIntoFeatures,
        )
        from instantsfm.controllers.feature_handler import GenerateDatabase
        from instantsfm.controllers.global_mapper import SolveGlobalMapper
        from instantsfm.controllers.reconstruction_writer import WriteGlomapReconstruction

        data_dir = Path(data_dir)
        path_info = ReadData(str(data_dir))
        if not path_info:
            raise RuntimeError(f"InstantSfM ReadData rejected {data_dir} — is images/ staged?")

        # Depth is the shipped mode; the nodepth path exists only for the eval ablation
        if self.use_depths and not path_info.depth_path:
            raise RuntimeError(
                f"no depth_vda/ under {data_dir} — generate_vda_depth must run first"
            )

        # SIFT database: reuse an existing one (idempotent re-runs), else build via
        # the system colmap binary (upstream subprocesses it; CPU SIFT, exhaustive)
        if not path_info.database_exists:
            logger.info("InstantSfM: building COLMAP feature database (CPU SIFT, exhaustive)")
            GenerateDatabase(
                str(path_info.image_path),
                str(path_info.database_path),
                self.features,
                None,
                single_camera=self.single_camera,
            )
        if not Path(path_info.database_path).exists():
            raise RuntimeError(
                "COLMAP database missing after GenerateDatabase — is the `colmap` binary installed?"
            )

        view_graph, cameras, images, feature_name, _rig = ReadColmapDatabase(path_info.database_path)
        if view_graph is None or cameras is None or images is None:
            raise RuntimeError(f"InstantSfM could not read {path_info.database_path}")

        # Config with copied dicts; depth-aware mode per creator field
        config = self._build_config()
        config.RUNTIME_OPTIONS["use_depths"] = self.use_depths
        if self.use_depths:
            logger.info("InstantSfM: loading depths from %s", path_info.depth_path)
            ReadDepthsIntoFeatures(path_info.depth_path, cameras, images)

        # Global mapping. Upstream crashes with IndexError (numpy-2 empty float64 mask,
        # scene/defs.py filter_by_mask) when track filtering leaves zero tracks —
        # translate to an actionable error instead of the raw traceback.
        try:
            cameras, images, tracks = SolveGlobalMapper(view_graph, cameras, images, config, visualizer=None)
        except IndexError as err:
            raise RuntimeError(
                "InstantSfM global mapping failed — all tracks were filtered out. "
                "Sparse/low-overlap frame sets do this (upstream numpy-2 bug on the "
                "empty-track path); use more frames or higher overlap."
            ) from err
        if not tracks:
            raise RuntimeError("InstantSfM produced zero tracks — reconstruction is empty")

        WriteGlomapReconstruction(
            str(path_info.output_path), cameras, images, tracks, str(path_info.image_path)
        )

        # Move to contract layout: sparse/0 -> colmap/sparse/0, database.db -> colmap/database.db
        colmap_dir = data_dir / "colmap"
        sparse_dst = colmap_dir / "sparse" / "0"
        if sparse_dst.exists():
            shutil.rmtree(sparse_dst)
        sparse_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(path_info.output_path) / "0"), str(sparse_dst))
        shutil.rmtree(path_info.output_path, ignore_errors=True)
        db_dst = colmap_dir / "database.db"
        if db_dst.exists():
            db_dst.unlink()
        shutil.move(str(path_info.database_path), str(db_dst))

        recon = pycolmap.Reconstruction(str(sparse_dst))
        logger.info(
            "InstantSfM: %d registered images, %d points3D", recon.num_reg_images(), recon.num_points3D()
        )
        return recon
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_instantsfm.py -v`
Expected: 5 PASS (last skips if instantsfm not installed in the test env).

- [ ] **Step 5: Commit**

```bash
git add tests/pointcloud/test_instantsfm.py collab_splats/pointcloud/sfm.py
git commit --only tests/pointcloud/test_instantsfm.py collab_splats/pointcloud/sfm.py -m "feat(pointcloud): InstantSfMCreator + VDA metric depth helper"
```

---

### Task 6: `Reconstructor._run_sfm` — orchestration + pointcloud.zarr build

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:924-926` (`_run_sfm` stub)
- Test: extend `tests/wrapper/test_sfm_config.py`

- [ ] **Step 1: Write the failing test (colmap/hloc stay stubs)**

Append to `tests/wrapper/test_sfm_config.py`:

```python
def test_run_sfm_colmap_hloc_still_not_implemented(tmp_path):
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "colmap"
    r = _make_reconstructor(cfg, tmp_path)  # copy the construction pattern used by existing tests/wrapper reconstructor fixtures
    with pytest.raises(NotImplementedError, match="instantsfm"):
        r._run_sfm()
```

(`_make_reconstructor`: reuse the existing wrapper-test fixture/helper for building a `Reconstructor` against a tmp output dir — `tests/wrapper/test_reconstructor.py` has one; import or replicate it. If constructing is heavyweight, `Reconstructor.__new__` + setting `self.config` is acceptable since `_run_sfm` reads only config before dispatch.)

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_config.py -v -k not_implemented`
Expected: FAIL — current stub raises with message "SfM path not yet implemented — use method: feedforward" (no "instantsfm" in it).

- [ ] **Step 3: Implement `_run_sfm`**

Replace the stub at `reconstructor.py:924-926`:

```python
    def _run_sfm(self) -> FeedforwardResult:
        """
        SfM pointcloud path (backend: instantsfm) — VDA metric depth + InstantSfM global mapping.

        - Stages frames.zarr keyframes to backend_dir/images/ (InstantSfM reads a dir).
        - Generates depth_vda/depths.npz (skipped when present), runs InstantSfMCreator,
          builds a FeedforwardResult at VDA depth resolution, saves pointcloud.zarr with
          provenance attrs, returns the result for the shared tail.
        """
        pc_cfg = self.config["pointcloud"]
        backend = pc_cfg["backend"]
        if backend != "instantsfm":
            raise NotImplementedError(
                f"sfm backend {backend!r} is not implemented — only 'instantsfm' is"
            )

        backend_dir = self.backend_dir
        backend_dir.mkdir(parents=True, exist_ok=True)
        store = FrameStore.open(self.frames_zarr)

        # Stage keyframes as jpgs (idempotent: skip when count matches)
        image_dir = backend_dir / "images"
        expected = len(store)
        if not image_dir.exists() or len(list(image_dir.glob("frame_*.jpg"))) != expected:
            logger.info("staging %d keyframes to %s", expected, image_dir)
            image_dir.mkdir(parents=True, exist_ok=True)
            store.export(image_dir, ext="jpg")

        # VDA metric depth — the only shipped mode (use_depths=True)
        frames = np.ascontiguousarray(store.images())  # (N, H, W, 3) uint8
        fps = float(self.config.get("preproc", {}).get("fps", 1.0) or 1.0)
        generate_vda_depth(frames, fps=fps, out_dir=backend_dir)

        # Global SfM via the upstream python API
        insfm_cfg = pc_cfg.get("instantsfm", {})
        creator = InstantSfMCreator(features=insfm_cfg.get("features", "colmap"))
        recon = creator.reconstruct(backend_dir)

        # Assemble the unified result at VDA depth resolution; provenance version from
        # the installed package, never a hardcoded pin
        result = self._sfm_result_from_reconstruction(recon, backend_dir, store)
        result.save_zarr(
            backend_dir / "pointcloud.zarr",
            extra_attrs={
                "method": "sfm",
                "backend": "instantsfm",
                "instantsfm_version": importlib.metadata.version("instantsfm"),
            },
        )
        return result
```

Then add the builder directly below:

```python
    def _sfm_result_from_reconstruction(
        self, recon, backend_dir: Path, store: FrameStore
    ) -> FeedforwardResult:
        """
        FeedforwardResult from a COLMAP reconstruction + VDA depths + frames.zarr images.

        - Everything is at VDA depth resolution (model_width/height); K rescaled from
          the original-res COLMAP camera — original-res K on model-res depth is the
          2026-08-11 mesh-regression class.
        - confidence and mv_* stay absent (never zeros).
        """
        # Depths define the working resolution
        depths = np.load(backend_dir / "depth_vda" / "depths.npz")["depths"].astype(np.float32)
        n_depth, h, w = depths.shape

        # Registered images in staged-name order; require full registration for v1 —
        # partial registration would silently misalign depth rows with poses
        images_sorted = sorted(recon.images.values(), key=lambda im: im.name)
        if len(images_sorted) != n_depth:
            raise RuntimeError(
                f"InstantSfM registered {len(images_sorted)}/{n_depth} frames — "
                "partial registration is not supported; re-run with more overlap"
            )
        name_to_row = {im.name: i for i, im in enumerate(images_sorted)}

        # Poses: w2c homogeneous, one row per staged frame
        extrinsics = np.stack(
            [
                np.vstack([im.cam_from_world().matrix(), [0.0, 0.0, 0.0, 1.0]])
                for im in images_sorted
            ]
        ).astype(np.float32)

        # Intrinsics: COLMAP K is at original (staged jpg) res — rescale to depth res
        orig_h, orig_w = store.image(0).shape[:2]
        sx, sy = w / orig_w, h / orig_h
        intrinsics = np.zeros((n_depth, 3, 3), dtype=np.float32)
        for i, im in enumerate(images_sorted):
            K = recon.cameras[im.camera_id].calibration_matrix()
            intrinsics[i] = np.array(
                [[K[0, 0] * sx, 0.0, K[0, 2] * sx], [0.0, K[1, 1] * sy, K[1, 2] * sy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

        # Sparse cloud + colors from points3D; pixel_indices from first observations
        point3d_ids = sorted(recon.points3D.keys())
        points = np.array([recon.points3D[pid].xyz for pid in point3d_ids], dtype=np.float32)
        colors = np.array([recon.points3D[pid].color for pid in point3d_ids], dtype=np.uint8)
        pixel_indices = _pixel_indices_from_reconstruction(
            recon, point3d_ids, name_to_row, scale_x=sx, scale_y=sy, depth_hw=(h, w)
        )

        # Images at depth res, [0,1] float32 (N, 3, H, W) — resized from frames.zarr
        rgb = np.stack(
            [cv2.resize(store.image(i), (w, h), interpolation=cv2.INTER_AREA) for i in range(n_depth)]
        )
        images_arr = (rgb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

        # Dense world points from metric depth + refined poses (shared unproject helper)
        world_points = unproject_depth_map_to_point_map(
            depths[..., None], extrinsics[:, :3, :], intrinsics
        ).astype(np.float32)

        # No crop was applied: full depth grid maps to the full original frame
        original_coords = np.array(
            [[0, 0, w, h, orig_w, orig_h]] * n_depth, dtype=np.float32
        )

        return FeedforwardResult(
            points=points,
            colors=colors,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_paths=[Path(im.name) for im in images_sorted],
            original_coords=original_coords,
            model_width=w,
            model_height=h,
            images=images_arr,
            world_points=world_points,
            depth=depths,
            pixel_indices=pixel_indices,
        )
```

Imports to add at reconstructor.py top (with existing import groups): `import importlib.metadata`, `cv2` (already imported — check), and

```python
from collab_splats.pointcloud.sfm import InstantSfMCreator, _pixel_indices_from_reconstruction, generate_vda_depth
from vggt.utils.geometry import unproject_depth_map_to_point_map
```

**Import-weight check:** `reconstructor.py` already imports feedforward modules (vggt warm-up) — these adds change nothing for dashboard fast-bind (`remote/__init__.py` never imports reconstructor). If `pycolmap` API differs (`im.cam_from_world()` vs `.cam_from_world` property by pycolmap version), match the accessor style already used in `_load_pointcloud_from_disk` (~line 805-836) — copy it exactly.

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -q`
Expected: PASS incl. the new NotImplementedError test.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_sfm_config.py
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_sfm_config.py -m "feat(wrapper): _run_sfm — stage frames, VDA depth, InstantSfM, unified pointcloud.zarr"
```

---

### Task 7: Install plumbing — setup.sh + pyproject note

**Files:**
- Modify: `setup.sh`, `pyproject.toml`, `.gitignore`

- [ ] **Step 1: setup.sh additions**

Append after the vismatch pre-fetch block. The upstream commit pin lives HERE and only here:

```bash
# --- InstantSfM backend (optional, CC-BY-NC-4.0 — research use) ------------------
# Their pyproject pins numpy==1.26.4; --no-deps is load-bearing (we run numpy 2.1.3).
# NOTE: plain `uv sync` prunes these (outside the lock) — rerun this block after any sync.
uv pip install --no-deps 'git+https://github.com/cre185/InstantSfM@d3e599e1a42b4c5a806a84d9f383e1005d25f61b'
uv pip install pyceres==2.3 scikit-sparse==0.4.15   # needs: apt-get install -y libsuitesparse-dev

# --- Video Depth Anything (metric) — clone + checkpoint, not pip-installable -----
VDA_DIR="third_party/Video-Depth-Anything"
if [ ! -d "$VDA_DIR" ]; then
    git clone https://github.com/DepthAnything/Video-Depth-Anything "$VDA_DIR"
fi
mkdir -p "$VDA_DIR/checkpoints"
if [ ! -f "$VDA_DIR/checkpoints/metric_video_depth_anything_vitl.pth" ]; then
    wget -q -O "$VDA_DIR/checkpoints/metric_video_depth_anything_vitl.pth" \
        "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth"
fi
```

Ensure `third_party/` is gitignored (check `.gitignore`; add `third_party/` if absent).

- [ ] **Step 2: pyproject.toml note**

InstantSfM/pyceres/scikit-sparse are installed via setup.sh `--no-deps` (their numpy pin conflicts with the lock), NOT via pyproject. Add a comment near the existing dependency notes:

```toml
# InstantSfM backend deps (instantsfm, pyceres, scikit-sparse, VDA clone) are installed
# by setup.sh with --no-deps — upstream pins numpy==1.26.4 which conflicts with the lock.
```

- [ ] **Step 3: Commit**

```bash
git add setup.sh pyproject.toml .gitignore
git commit --only setup.sh pyproject.toml .gitignore -m "build: instantsfm + VDA install plumbing (setup.sh, --no-deps guard)"
```

---

### Task 8: Docs — configs/README.md contract + migration

**Files:**
- Modify: `configs/README.md`

- [ ] **Step 1: Update the processed-scene output contract**

In the output-contract section, rename the artifact and document the hard cutover:

```markdown
- `<backend>/pointcloud.zarr` — the unified reconstruction artifact for every
  `pointcloud.method` (feedforward and sfm). Store attrs carry provenance:
  `method`, `backend`, and for instantsfm the installed upstream version.
  `confidence` and `mv_*` arrays are present only when the method produces them
  (absent, never zeros).

  **Migration (breaking, 2026-08-23):** `feedforward.zarr` was renamed with no
  fallback. Scenes written before the rename need a one-time rename or a re-run:
  - local: `mv <scene>/<backend>/feedforward.zarr <scene>/<backend>/pointcloud.zarr`
  - remote: `rclone moveto <remote>:environments-processed/<scene>/feedforward.zarr \
      <remote>:environments-processed/<scene>/pointcloud.zarr`
  Notebooks/tools reading the old name break until the scene is migrated.
```

- [ ] **Step 2: Add sfm/instantsfm subsection**

Config keys (`pointcloud.method: sfm`, `backend: instantsfm`, `instantsfm.features: colmap`), the VDA depth requirement (setup.sh clone + checkpoint, system `colmap` binary required), CC-BY-NC-4.0 license note, `bundle_adjustment`/`refine` unsupported, `database.db` excluded from GCS push (existing anchor), outputs land in the standard layout (`colmap/sparse/0`, `pointcloud.zarr`, `sparse_pc.ply`, `transforms.json`).

- [ ] **Step 3: Commit**

```bash
git add configs/README.md
git commit --only configs/README.md -m "docs(configs): pointcloud.zarr contract, migration, instantsfm backend notes"
```

---

### Task 9: Live probes + GPU smoke (human-gated, tmux)

No code — evidence gates. Run in tmux (memory: heavy inference never in notebooks; 46.6 GB cgroup).

- [ ] **Step 1: Install per setup.sh block (if not already)**

```bash
/opt/venv/reconstruction/bin/python -c "import instantsfm, pyceres; import importlib.metadata as m; print('instantsfm', m.version('instantsfm'))"
/opt/venv/reconstruction/bin/python -c "import numpy, torch, gsplat; print(numpy.__version__, torch.__version__, gsplat.__version__)"
```
Expected: `instantsfm 0.3.0`; pins intact: `2.1.3 2.5.1+cu121 1.5.3`. **If numpy moved, stop — the `--no-deps` guard failed.**

- [ ] **Step 2: VDA import + inference probe vs torch 2.5.1 (open risk from spec)**

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "third_party/Video-Depth-Anything/metric_depth")
from video_depth_anything.video_depth import VideoDepthAnything  # exact path may differ — fix sfm.py if so
import torch
m = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
sd = torch.load("third_party/Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth", map_location="cpu")
m.load_state_dict(sd, strict=True); m = m.cuda().eval()
frames = np.random.randint(0, 255, (8, 518, 518, 3), dtype=np.uint8)
depths, fps = m.infer_video_depth(frames, 1.0, input_size=518, device="cuda", fp32=False)
print("OK", np.asarray(depths).shape, np.asarray(depths).dtype, float(np.median(depths)))
EOF
```
Expected: `OK (8, ...)` with positive metric-scale medians. Any import/signature mismatch → fix `generate_vda_depth` in `sfm.py` (single place) and note in the measured report.

- [ ] **Step 3: Realistic-frame-count InstantSfM smoke (open risk: the 12-frame 81→0 crash)**

Run the full pipeline on the tutorial video at real density:

```bash
tmux new -s insfm_smoke
which colmap   # system binary required by GenerateDatabase; apt-get install colmap if missing
/opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py \
    --video data/tutorial/*.MP4 --output /tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad/insfm_e2e \
    --set pointcloud.method=sfm --set pointcloud.backend=instantsfm
```

(Adapt to the actual run_pipeline.py CLI — check `--help`; if there is no `--set`, write a one-off yaml overlay next to the output dir.)

Expected: ≥30 keyframes staged; SIFT DB built; tracks survive filtering (log line `Before filtering: N , after filtering: M` with M > 0); `pointcloud.zarr` + `colmap/sparse/0` + `sparse_pc.ply` + `transforms.json` written. **If M = 0 again at realistic density, STOP and report — that's a design-level blocker (spec open-risk #1), not a bug to patch silently.**

- [ ] **Step 4: Downstream stage matrix on the smoke output**

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
import logging, yaml
from pathlib import Path
from collab_splats.wrapper.reconstructor import Reconstructor

logging.basicConfig(level=logging.INFO)
out = Path("/tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad/insfm_e2e")

cfg = yaml.safe_load(open("configs/base.yaml"))
cfg["pointcloud"]["method"] = "sfm"
cfg["pointcloud"]["backend"] = "instantsfm"
cfg["mesh"]["enabled"] = True
cfg["splats"]["enabled"] = True

# Construct against the smoke output dir the same way run_pipeline.py does (copy its
# Reconstructor(...) call — video/output args); then drive each stage explicitly:
r = Reconstructor(config=cfg, output_path=out)   # adapt kwargs to the actual __init__
r.build_pointcloud()          # loads the Step-3 result from disk (skip/load path)
r.splats()                    # expect Task 3 log: "no confidence in zarr — using unmasked depth"
r.mesh()                      # expect Task 3 log: "fusing unmasked"
r.extract_semantics()         # lift_features with uniform weights — no assert
r.build_localization_db()
r.verify()
try:
    r.refine_poses()
    raise SystemExit("FAIL: refine_poses did not refuse method=sfm")
except ValueError as e:
    print("refine refusal OK:", e)
EOF
```

Checks:
- `splats`: trains; depth targets loaded from pointcloud.zarr (VDA metric depth); Task 3 log line present.
- `mesh`: fuses; visually sane ply.
- `semantics`, `build_localization_db`, `verify`: run to completion.
- `refine_poses`: raises `ValueError` (Task 4).

- [ ] **Step 5: Record results**

Append a "Measured (smoke)" section to the spec `docs/superpowers/specs/2026-08-23-instantsfm-backend-design.md`: frame count, track counts, runtime per phase, registered/total, stage matrix outcomes. Commit:

```bash
git add -f docs/superpowers/specs/2026-08-23-instantsfm-backend-design.md
git commit --only docs/superpowers/specs/2026-08-23-instantsfm-backend-design.md -m "docs(specs): instantsfm smoke — measured results"
```

---

### Task 10: Eval conditions (human-gated, gates the fast-follow)

**Files:**
- Modify: `evals/scripts/eval.py`

- [ ] **Step 1: Add conditions**

Follow the existing condition-registration pattern in `eval.py` (e.g. how `ba_percam` was added) — two new conditions:
- `instantsfm`: `pointcloud.method=sfm`, `backend=instantsfm` (depth-aware, the shipped mode)
- `instantsfm_nodepth`: same but `InstantSfMCreator(use_depths=False)` — the field already exists (Task 5); the condition threads it through however eval conditions override creator kwargs. The Task 5 config-copy test is what makes this ablation valid in-process.

- [ ] **Step 2: Run (tmux, human-watched) on 7-Scenes chess/seq-01**

```bash
/opt/venv/reconstruction/bin/python evals/scripts/eval.py --help   # confirm flags
# then the same invocation pattern as the ba-track-quality-parity runs, conditions instantsfm,instantsfm_nodepth
```
Compare ATE/RPE against the four feedforward backends' recorded numbers. Results append to the spec's measured section.

- [ ] **Step 3: Commit**

```bash
git add evals/scripts/eval.py
git commit --only evals/scripts/eval.py -m "feat(evals): instantsfm / instantsfm_nodepth conditions"
```

---

### Task 11: Full-suite gate + graph update

- [ ] **Step 1: Full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:randomly`
Expected: green modulo `docs/known-test-failures.md`. Fix regressions before declaring done.

- [ ] **Step 2: Format + graph**

```bash
black collab_splats/pointcloud/sfm.py tests/pointcloud/test_instantsfm.py tests/pointcloud/test_zarr_attrs.py tests/mesh/test_absent_confidence.py tests/wrapper/test_sfm_config.py
isort collab_splats/pointcloud/sfm.py
graphify update .
```

(**Never repo-wide `black .`** — venv black 26.5.1 is newer than repo formatting; format only the files this plan touched.)

- [ ] **Step 3: Final commit if formatting changed anything**

```bash
git add <changed files only>
git commit --only <changed files> -m "style: format instantsfm backend files"
```

---

## Deliberately out of scope (spec non-goals)

- colmap/hloc creator implementations (stubs stay).
- Partial-registration handling (v1 requires full registration; RuntimeError names the fix).
- Automated backfill of legacy scenes — migration is the documented one-line rename (Task 8) or a reprocess.
- Loop closure with sfm method (LC is a feedforward-creator concern; config combination untested, not blocked).
- BFMatcher/feature tuning inside InstantSfM; upstream numpy-2 `filter_by_mask` fix (report upstream instead).
- Docker rebuild (no docker binary here — owed alongside the splats-module Docker debt).
