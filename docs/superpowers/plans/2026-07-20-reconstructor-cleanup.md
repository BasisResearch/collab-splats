# Reconstructor Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `Reconstructor` config path read as one coherent thing — `base.yaml` becomes the single source of defaults, inline code defaults and dead/aspirational knobs are removed, and each stage method reads as labeled logical blocks.

**Architecture:** Config stays a plain dict (full YAML override flexibility). `Reconstructor.__init__` deep-merges the passed config over `configs/base.yaml`, so every key is guaranteed present and every `cfg.get(k, default)` collapses to a flat `cfg[k]` read. Unimplemented paths raise `NotImplementedError` instead of silently no-op'ing; poisson (not needed) is removed outright; dead `from_config_file` is deleted.

**Tech Stack:** Python 3.11, `mergedeep`, `pyyaml`, `pytest`. Run tests with `/opt/venv/reconstruction/bin/python -m pytest`.

---

## Reference: current state (verified 2026-07-20)

- File under change: `collab_splats/wrapper/reconstructor.py`
- Tests: `tests/wrapper/test_reconstructor.py`
- Entrypoint: `docs/examples/run_pipeline.py`
- Config: `configs/base.yaml` (repo root)
- Current drifts to fix:
  - `min_frames`: base.yaml `150` vs code `pre_cfg.get("min_frames", 300)` (line ~409)
  - `backend`: base.yaml `vggt_omega`, `validate_config` default `vggtx` (line ~349), `backend_dir` default `nerfstudio` (line ~398)
  - `frame_selection: fps` — dishonest label; `_extract_frames` only branches on `optical_flow`
- Dead: `Reconstructor.from_config_file` (no callers; calls `ConfigLoader.load(dataset=...)` against the deleted `configs/datasets/`)
- Poisson: `_VALID_MESHERS = {"tsdf", "poisson"}` + check in `validate_config` (lines ~27, ~363-365) — remove

`DEFAULT_CONFIG_DIR` derivation: `Path(__file__).parents[2] / "configs"` → `<repo>/configs` (parents: `[0]=wrapper`, `[1]=collab_splats`, `[2]=repo root`).

---

## Task 1: base.yaml — single source of defaults

**Files:**
- Modify: `configs/base.yaml`

- [ ] **Step 1: Read the current file**

Run: read `configs/base.yaml` to see the exact current content before editing.

- [ ] **Step 2: Rename the frame_selection label to be honest**

In the `preprocessing:` block, change:
```yaml
  frame_selection: fps        # fps | optical_flow
```
to:
```yaml
  frame_selection: uniform    # uniform | optical_flow
```

- [ ] **Step 3: Remove the poisson mesher knob**

In the `mesh:` block, delete the `mesher:` line entirely:
```yaml
  mesher: tsdf                # tsdf | poisson   <-- DELETE THIS LINE
```
Leave `enabled`, `voxel_size`, `sdf_trunc`.

- [ ] **Step 4: Add a NOT-YET-IMPLEMENTED banner over the aspirational knobs**

`bundle_adjustment` (in `pointcloud:`) and `confidence_threshold` (in `pointcloud.clean:`) are not wired at the Reconstructor level. Add a banner comment so status is visible at config time. In the `pointcloud:` block, above `bundle_adjustment`:
```yaml
  # ── NOT YET IMPLEMENTED (raises NotImplementedError if enabled) ──
  bundle_adjustment: false    # BA at Reconstructor level not wired; pass to creator config directly
```
And annotate the clean field:
```yaml
    confidence_threshold: null  # NOT YET READ by the clean step
```

- [ ] **Step 5: Add a header note naming base.yaml as the single source of defaults**

Under the existing top comment block, add:
```yaml
# This file is the SINGLE SOURCE OF DEFAULTS. Reconstructor.__init__ deep-merges
# any passed config over these values — do not duplicate defaults in Python code.
```

- [ ] **Step 6: Commit**

```bash
git add -f configs/base.yaml
git commit -m "config(base): honest frame_selection label, drop poisson, mark unimplemented knobs"
```

---

## Task 2: `__init__` merges config over base.yaml defaults

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/wrapper/test_reconstructor.py`:
```python
def test_init_fills_defaults_from_base_yaml(tmp_path):
    """A partial config gets missing keys filled from configs/base.yaml."""
    partial = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
    }
    rec = Reconstructor(partial)
    # min_frames comes from base.yaml (150), NOT a stale code default (300)
    assert rec.config["preprocessing"]["min_frames"] == 150
    # backend comes from base.yaml (vggt_omega)
    assert rec.config["pointcloud"]["backend"] == "vggt_omega"


def test_init_user_override_wins_over_base(tmp_path):
    """User-supplied value overrides the base.yaml default."""
    partial = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"backend": "mapanything"},
    }
    rec = Reconstructor(partial)
    assert rec.config["pointcloud"]["backend"] == "mapanything"
    # sibling keys still filled from base
    assert rec.config["pointcloud"]["method"] == "feedforward"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_init_fills_defaults_from_base_yaml -v`
Expected: FAIL — `min_frames` KeyError or wrong value (no base merge yet).

- [ ] **Step 3: Add the module constant and imports**

At the top of `collab_splats/wrapper/reconstructor.py`, add to the imports:
```python
import yaml
from mergedeep import merge
```
In the Constants section, add:
```python
# base.yaml is the single source of defaults; __init__ merges any passed config over it.
DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"
```

- [ ] **Step 4: Rewrite `__init__` to merge base then validate**

Replace the current `__init__`:
```python
    def __init__(self, config: dict[str, Any], config_dir: str | Path = DEFAULT_CONFIG_DIR) -> None:
        """Merge config over base.yaml defaults, validate, and store."""
        # Load base defaults; deep-merge the caller's config over them so every key is present
        base_path = Path(config_dir) / "base.yaml"
        with open(base_path) as f:
            defaults = yaml.safe_load(f) or {}
        merged = merge({}, defaults, config)

        # Validate shape, then store the fully-populated config
        self.config = self.validate_config(merged)
        self.pointcloud: PointcloudResult | None = None
```

- [ ] **Step 5: Run to verify both tests pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_init_fills_defaults_from_base_yaml tests/wrapper/test_reconstructor.py::test_init_user_override_wins_over_base -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): Reconstructor.__init__ merges config over base.yaml defaults"
```

---

## Task 3: Remove inline defaults — base.yaml is the only default source

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test (drift guard)**

Add:
```python
def test_no_inline_defaults_in_source():
    """No cfg.get(key, default) with an inline default remains — base.yaml is the only source."""
    import re
    from pathlib import Path

    src = Path("collab_splats/wrapper/reconstructor.py").read_text()
    # Match .get("key", <default>) with a VALUE default. Structural {}/[] defaults
    # (e.g. validate_config's config.get("pointcloud", {}) on a raw partial config) are allowed.
    offenders = re.findall(r"\.get\(\s*['\"][^'\"]+['\"]\s*,\s*(?!\{\}|\[\])[^)]+\)", src)
    assert offenders == [], f"inline value defaults still present: {offenders}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_no_inline_defaults_in_source -v`
Expected: FAIL — lists the current `.get(k, default)` offenders.

- [ ] **Step 3: Replace every two-arg `.get` in Reconstructor methods with indexed reads**

Every config key is guaranteed present after the base merge. Convert each. Specific sites:

`preprocess`:
```python
        pre_cfg = self.config["preprocessing"]
        extracted = _extract_frames(
            input_path=Path(self.config["input_path"]),
            output_dir=self.images_dir,
            frame_selection=pre_cfg["frame_selection"],
            frame_proportion=pre_cfg["frame_proportion"],
            min_frames=pre_cfg["min_frames"],
            max_frames=pre_cfg["max_frames"],
        )
```

`build_pointcloud`:
```python
        pc_cfg = self.config["pointcloud"]
        method = pc_cfg["method"]
        ...
            result = _run_feedforward(
                backend=pc_cfg["backend"],
                images_dir=self.images_dir,
                output_dir=self.backend_dir,
                bundle_adjustment=pc_cfg["bundle_adjustment"],
                loop_closure=pc_cfg["loop_closure"],
            )
        # Apply cleaning step if enabled
        clean_cfg = pc_cfg["clean"]
        if clean_cfg["enabled"]:
            result = self._clean_pointcloud(result, clean_cfg)
```

`_clean_pointcloud` — `cfg.get("outlier_removal", True)` → `cfg["outlier_removal"]`; `cfg.get("voxel_size")` (single-arg, returns None) stays as `cfg["voxel_size"]` (base.yaml supplies `null`).

`extract_semantics`:
```python
        sem_cfg = self.config["semantics"]
        extractor_name = sem_cfg["extractor"]
        n_components = sem_cfg["n_components"]
```

`mesh`:
```python
        mesh_cfg = self.config["mesh"]
        out = _run_tsdf_mesh(
            result=result,
            feedforward_zarr=feedforward_zarr,
            output_dir=self.backend_dir / "mesh",
            voxel_size=mesh_cfg["voxel_size"],
            sdf_trunc=mesh_cfg["sdf_trunc"],
        )
```

`build_localization_db`:
```python
        loc_cfg = self.config["localization"]
        extractor_name = loc_cfg["extractor"]
        radius = loc_cfg["radius"]
```

`backend_dir` property:
```python
    @property
    def backend_dir(self) -> Path:
        """output_path / backend — backend subdir for all stage 2+ artifacts."""
        return Path(self.config["output_path"]) / self.config["pointcloud"]["backend"]
```

`_run_nerfstudio` — `ns_cfg.get("sfm_tool", "hloc")` / `ns_cfg.get("train_method", ...)`: base.yaml has no `nerfstudio` section. Add one to `configs/base.yaml` so these become `ns_cfg["sfm_tool"]` / `ns_cfg["train_method"]`:
```yaml
nerfstudio:
  sfm_tool: hloc
  train_method: rade-features
```
Then in code:
```python
        ns_cfg = self.config["nerfstudio"]
        sfm_tool = ns_cfg["sfm_tool"]
        train_method = ns_cfg["train_method"]
```

`run_pipeline` — the `self.config.get("semantics", {}).get("enabled", False)` chain becomes direct reads (sections guaranteed present):
```python
            stages = ["preprocess", "pointcloud"]
            if self.config["semantics"]["enabled"]:
                stages.append("semantics")
            if self.config["mesh"]["enabled"]:
                stages.append("mesh")
            if self.config["localization"]["enabled"]:
                stages.append("localize")
```

- [ ] **Step 4: Update base.yaml enabled/localization defaults if missing**

Ensure `configs/base.yaml` has `localization.radius` and every `enabled` flag present (they are, per current file). Add the `nerfstudio:` block from Step 3.

- [ ] **Step 5: Run the drift-guard test + full reconstructor suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -v`
Expected: PASS. If any test used `_make_config` with keys now sourced from base, they still pass (explicit values win over base).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py configs/base.yaml tests/wrapper/test_reconstructor.py
git commit -m "refactor(wrapper): drop all inline config defaults; base.yaml is sole source"
```

---

## Task 4: Honest frame_selection + validation

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_extract_frames_uniform_is_default_branch(tmp_path, monkeypatch):
    """frame_selection='uniform' takes the uniform sampler branch (not optical_flow)."""
    from collab_splats.wrapper import reconstructor as R

    calls = {}

    def fake_sample_frames(path, method, max_frames):
        calls["method"] = method
        import numpy as np
        return [np.zeros((4, 4, 3), dtype=np.uint8)], None

    def fake_video_info(path):
        return {"total_frames": 100}

    monkeypatch.setattr(R, "sample_frames", fake_sample_frames, raising=False)
    monkeypatch.setattr(R, "get_video_info", fake_video_info, raising=False)

    out = tmp_path / "out"
    video = tmp_path / "v.mp4"
    video.touch()
    R._extract_frames(video, out, "uniform", 0.1, 5, 50)
    assert calls["method"] == "uniform"
```

Note: `_extract_frames` currently imports `sample_frames`/`get_video_info` *inside* the function. For this test's monkeypatch to bind, move those two imports to module top (see Step 3). If a heavy-import constraint blocks that, instead assert via a real short video fixture — but module-top import of `collab_splats.preproc` is already safe (no GPU).

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_extract_frames_uniform_is_default_branch -v`
Expected: FAIL (imports are function-local, monkeypatch on module doesn't bind).

- [ ] **Step 3: Hoist the preproc imports and honour the uniform label**

At module top of `reconstructor.py`:
```python
from collab_splats.preproc import get_video_info, sample_frames
```
Remove the inline `from collab_splats.preproc import ...` inside `_extract_frames`. Keep the `import cv2` inline (cv2 is heavy-ish but fine either way; hoist if preferred). The branch logic already treats non-`optical_flow` as uniform; make the comment honest:
```python
    # Video — 'uniform' spreads target_count evenly; 'optical_flow' picks high-motion frames
    if frame_selection == "optical_flow":
        ...
    else:  # uniform
        ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_extract_frames_uniform_is_default_branch -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "refactor(wrapper): honest 'uniform' frame_selection label; hoist preproc imports"
```

---

## Task 5: Drop poisson from validate_config

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_validate_config_ignores_mesher(tmp_path):
    """mesh.mesher is no longer a validated knob — its absence/presence never raises."""
    config = {
        "input_path": str(tmp_path / "v.mp4"),
        "output_path": str(tmp_path / "out"),
    }
    # Reaches validate via __init__ (base-merged); must not raise on missing mesher
    rec = Reconstructor(config)
    assert "mesher" not in rec.config["mesh"]


def test_valid_meshers_symbol_removed():
    """_VALID_MESHERS constant is gone."""
    from collab_splats.wrapper import reconstructor as R
    assert not hasattr(R, "_VALID_MESHERS")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_valid_meshers_symbol_removed -v`
Expected: FAIL — `_VALID_MESHERS` still defined.

- [ ] **Step 3: Remove the constant and the check**

Delete the `_VALID_MESHERS = {"tsdf", "poisson"}` line. In `validate_config`, delete:
```python
        mesh_cfg = config.get("mesh", {})
        mesher = mesh_cfg.get("mesher", "tsdf")
        if mesher not in _VALID_MESHERS:
            raise ValueError(f"mesh.mesher must be one of {_VALID_MESHERS}, got '{mesher}'")
```

- [ ] **Step 4: Run to verify both pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_validate_config_ignores_mesher tests/wrapper/test_reconstructor.py::test_valid_meshers_symbol_removed -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "refactor(wrapper): drop poisson mesher knob (tsdf-only), remove _VALID_MESHERS"
```

---

## Task 6: Loud NotImplementedError for reconstructor-level bundle_adjustment

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_reconstructor_bundle_adjustment_raises(tmp_path):
    """bundle_adjustment=True at the Reconstructor level raises NotImplementedError (was a silent no-op)."""
    config = {
        "input_path": str(tmp_path / "v.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"backend": "vggtx", "bundle_adjustment": True},
    }
    rec = Reconstructor(config)
    with patch("collab_splats.wrapper.reconstructor._run_feedforward") as mock_ff:
        mock_ff.return_value = _make_mock_pointcloud_result(tmp_path)
        with pytest.raises(NotImplementedError, match="bundle_adjustment"):
            rec.build_pointcloud(overwrite=True)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_reconstructor_bundle_adjustment_raises -v`
Expected: FAIL — currently logs a warning, returns normally.

- [ ] **Step 3: Raise in build_pointcloud before feedforward dispatch**

In `build_pointcloud`, after `pc_cfg = self.config["pointcloud"]`, add a guard:
```python
        # BA at the Reconstructor level is not wired — fail loud instead of silently no-op'ing
        if pc_cfg["bundle_adjustment"]:
            raise NotImplementedError(
                "pointcloud.bundle_adjustment is not wired at the Reconstructor level. "
                "Pass bundle_adjustment to the creator config directly for now."
            )
```
Then remove the old `bundle_adjustment` warning block inside `_run_feedforward` (lines that `logger.warning(... "bundle_adjustment=True is not yet wired ...")`) and drop the now-unused `bundle_adjustment` parameter from `_run_feedforward`'s signature and its call site.

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_reconstructor_bundle_adjustment_raises -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "refactor(wrapper): reconstructor-level bundle_adjustment raises NotImplementedError"
```

---

## Task 7: Delete dead `from_config_file`

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test**

Add:
```python
def test_from_config_file_removed():
    """The dead, dataset-based from_config_file constructor is gone."""
    assert not hasattr(Reconstructor, "from_config_file")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_from_config_file_removed -v`
Expected: FAIL — method still present.

- [ ] **Step 3: Delete the method**

Remove the entire `from_config_file` classmethod (and its now-unused `ConfigLoader` import inside it). Confirm no other file imports it:

Run: `grep -rn "from_config_file" collab_splats/ docs/ evals/ tests/ | grep -v splatter`
Expected: no Reconstructor hits (splatter.py has its own, untouched).

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_from_config_file_removed -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "refactor(wrapper): delete dead Reconstructor.from_config_file"
```

---

## Task 8: Simplify run_pipeline.build_scene_config (drop manual base merge)

**Files:**
- Modify: `docs/examples/run_pipeline.py`
- Test: `tests/scripts/test_run_pipeline.py` (create if absent; check `tests/scripts/` first)

- [ ] **Step 1: Check for an existing test file**

Run: `ls tests/scripts/`
If a run_pipeline test exists, add to it; else create `tests/scripts/test_run_pipeline.py`.

- [ ] **Step 2: Write the failing test**

Load `build_scene_config` by file path (mirrors the ConfigLoader import trick already used in `test_reconstructor.py`):
```python
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_pipeline",
    Path(__file__).parent.parent.parent / "docs" / "examples" / "run_pipeline.py",
)
run_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pipeline)


def test_build_scene_config_sets_paths_only(tmp_path):
    """build_scene_config sets input/output paths and defers all defaults to Reconstructor."""
    video = tmp_path / "scene.MP4"
    video.touch()
    config_dir = Path(__file__).parent.parent.parent / "configs"
    cfg = run_pipeline.build_scene_config(video, tmp_path / "out", config_dir)
    assert cfg["input_path"] == str(video)
    assert cfg["output_path"].endswith("scene")
    # No manual base merge here anymore: only the keys this function sets are present
    assert set(cfg) <= {"input_path", "output_path"} | set(  # override_config keys, if any
        {}
    )
```

- [ ] **Step 3: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/scripts/test_run_pipeline.py::test_build_scene_config_sets_paths_only -v`
Expected: FAIL — current `build_scene_config` merges `loader.base_config`, so `cfg` has many keys.

- [ ] **Step 4: Simplify build_scene_config**

Replace:
```python
def build_scene_config(video, output_root, config_dir, override_config=None):
    """Build a per-video config from base.yaml with input/output paths set."""
    loader = ConfigLoader(config_dir)
    config = merge({}, loader.base_config)  # copy of base.yaml
    if override_config:
        config = merge({}, config, override_config)  # shared --config overrides
    config["input_path"] = str(video)
    config["output_path"] = str(scene_output_dir(video, output_root))
    return config
```
with:
```python
def build_scene_config(video, output_root, config_dir, override_config=None):
    """Build a per-video override dict. Reconstructor merges base.yaml defaults itself."""
    # Only carry the shared --config overrides plus per-video paths; defaults come from base.yaml
    config = merge({}, override_config) if override_config else {}
    config["input_path"] = str(video)
    config["output_path"] = str(scene_output_dir(video, output_root))
    return config
```
Then, so `Reconstructor` reads base.yaml from the intended dir, pass `config_dir` through in `run_scene`:
```python
    r = Reconstructor(config, config_dir=config_dir)
```
Remove the now-unused `ConfigLoader` import if nothing else in the file uses it (keep `merge`).

- [ ] **Step 5: Run to verify it passes + the script still imports**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/scripts/test_run_pipeline.py -v`
Run: `/opt/venv/reconstruction/bin/python -c "import importlib.util,pathlib; s=importlib.util.spec_from_file_location('rp', 'docs/examples/run_pipeline.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('import OK')"`
Expected: PASS + `import OK`.

- [ ] **Step 6: Commit**

```bash
git add docs/examples/run_pipeline.py tests/scripts/test_run_pipeline.py
git commit -m "refactor(examples): run_pipeline defers config defaults to Reconstructor"
```

---

## Task 9: Readability pass + update stale test fixtures

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Update the `_make_config` test helper**

In `tests/wrapper/test_reconstructor.py`, change the helper so it matches the cleaned surface (`fps` → `uniform`, drop `mesher`):
```python
        "preprocessing": {"frame_selection": "uniform", "frame_proportion": 0.1, "min_frames": 10},
        ...
        "mesh": {"enabled": False, "voxel_size": 0.01, "sdf_trunc": 0.04},
```
Leave the other keys as-is (explicit values still win over base after merge).

- [ ] **Step 2: Readability pass on each stage method**

Apply the house style (CLAUDE.md) to `reconstructor.py` — do NOT change behavior:
- Ensure `########` dividers separate Constants / Helpers / Reconstructor.
- Each logical block in `preprocess`, `build_pointcloud`, `extract_semantics`, `mesh`, `build_localization_db`, `run_pipeline` has one short block comment (`# Skip if outputs exist`, `# Merge over base defaults`, `# Dispatch by method`, `# Validate stage dependencies`).
- One-line docstrings, no padding.
- No leftover `.get(k, default)` (Task 3 covered these — verify none reintroduced).

- [ ] **Step 3: Format**

Run: `black collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py && isort collab_splats/wrapper/reconstructor.py`

- [ ] **Step 4: Run the FULL suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ tests/scripts/ -v`
Expected: PASS (green). Investigate any failure before committing.

- [ ] **Step 5: Run the drift-guard once more**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_no_inline_defaults_in_source -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "refactor(wrapper): readability pass + update test fixtures to cleaned config surface"
```

---

## Task 10: Full regression + graph update

**Files:** none (verification)

- [ ] **Step 1: Run the whole test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/`
Expected: no NEW failures vs the pre-change baseline (compare against `docs/known-test-failures.md`).

- [ ] **Step 2: Update the code graph**

Run: `graphify update .`

- [ ] **Step 3: Final commit if the graph changed**

```bash
git add -A && git commit -m "chore: refresh code graph after reconstructor cleanup" || echo "no graph changes"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- One populate path (`__init__` merges base) → Task 2 ✓
- Kill inline defaults + fix drift → Task 3 ✓
- frame_selection honest label → Task 1 (config) + Task 4 (code/label) ✓
- Poisson cut → Task 1 (base.yaml) + Task 5 (validate_config) ✓
- bundle_adjustment loud stub → Task 6 ✓
- sfm stays NotImplementedError → unchanged (already raises; no task needed) ✓
- confidence_threshold marked → Task 1 ✓
- Delete from_config_file → Task 7 ✓
- Entrypoint simplification → Task 8 ✓
- Readability pass → Task 9 ✓
- No TypedDict / no new type → nothing added anywhere ✓
- Out-of-scope (module helpers structure, Splatter, ConfigLoader internals) → untouched ✓

**Placeholder scan:** no TBD/TODO; every code step shows code.

**Type/name consistency:** `DEFAULT_CONFIG_DIR`, `build_scene_config`, `_run_feedforward` (bundle_adjustment param removed in Task 6 — call site updated same task), `_VALID_MESHERS` removed in Task 5, helper `_make_config` updated in Task 9. `sample_frames`/`get_video_info` hoisted to module top in Task 4 and referenced there in Task 3's `_extract_frames` reads.
