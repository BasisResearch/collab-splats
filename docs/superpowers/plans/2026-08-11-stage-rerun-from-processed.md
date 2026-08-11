# Stage Re-run From Processed — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `run_pipeline_remote.py --stages mesh` (or `semantics` / `localize`) pull an already-processed scene out of `environments-processed`, re-run only those stages, and push the result back.

**Architecture:** A stage set that contains only *leaf* stages (nothing depends on them) is re-runnable from processed data; anything else starts from the curated video, so no run can leave a stale downstream artifact. Routing, transport and config assembly live in one new function, `collab_splats/remote/rerun.py:prepare_scene`. Filesystem-layout knowledge stays in `Reconstructor`, which gains leaf branches on its existing `_stage_output_exists` and a `_resolve_result()` that loads a `PointcloudResult` off disk when a leaf stage runs alone.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), rclone/GCS via `collab_splats.remote.SceneSource`, `mergedeep.merge`, zarr v3, pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-stage-rerun-from-processed-design.md` (committed, `1f00d39`).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `collab_splats/wrapper/reconstructor.py` | Modify | `LEAF_STAGES`; leaf branches on `_stage_output_exists`; `_resolve_result()`; no-op refusal in `run_pipeline`; retire `build_localization_db`'s dead `result` param |
| `collab_splats/remote/rerun.py` | Create | `prepare_scene()` (route + pull + config merge) and `discover_scenes()` (`--all` bucket choice) |
| `collab_splats/remote/__init__.py` | Modify | Export both |
| `collab_splats/wrapper/batch.py` | Modify | `build_scene_config` tolerates `video=None`; `run_scene` always writes `run_config.yaml` |
| `docs/examples/run_pipeline_remote.py` | Modify | Call `prepare_scene` / `discover_scenes`; document the re-run usage |
| `tests/wrapper/test_reconstructor.py` | Modify | Leaf markers, disk resolution, refusal |
| `tests/wrapper/test_batch.py` | Modify | `video=None`, unconditional config write |
| `tests/remote/test_rerun.py` | Create | Routing, failure modes, config merge, `--all` bucket choice |
| `configs/README.md` | Modify | Document the re-run contract |
| `CLAUDE.md` | Modify | Move the item to "Recently completed" |

**Status note:** Task 1's *source* edits are already applied in the working tree (uncommitted). Task 1 below is written so a fresh worker verifies them against the shown code and then writes the missing tests. Every other task is unstarted.

**Environment for every command:** run from `/workspace/collab-splats`. `pytest-randomly` is installed and reorders tests, so **always** pass `-p no:randomly`.

---

### Task 1: Reconstructor — leaf markers, disk resolution, no-op refusal

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Verify the source edits are present**

Run: `grep -n "LEAF_STAGES\|_resolve_result\|output already exists" collab_splats/wrapper/reconstructor.py`

Expected: matches at the constant, both `_resolve_result` call sites, its definition, and the refusal. If any are missing, apply them from Steps 2–6 below; if all are present, tick Steps 2–6 and go to Step 7.

- [ ] **Step 2: `LEAF_STAGES`, immediately after `_STAGE_DEPS`**

```python
# A stage is re-runnable on its own iff nothing depends on it → {semantics, mesh, localize}.
# Derived from the graph above rather than hardcoded: a future stage that depends on mesh drops
# mesh from this set automatically, so callers gating on it can never disagree with _STAGE_DEPS.
LEAF_STAGES = frozenset(s for s in _STAGE_ORDER if not any(s in deps for deps in _STAGE_DEPS.values()))
```

- [ ] **Step 3: Leaf branches on `_stage_output_exists`, before its final `return False`**

```python
        # Leaf-stage markers. Only preproc/pointcloud are ever depended on, but run_pipeline also
        # needs these to refuse a named stage whose output already exists — and each leaf stage's
        # own skip-check reads them, so they live here once instead of three times.
        if stage == "mesh":
            return (self.backend_dir / "mesh.ply").exists()
        if stage == "semantics":
            lifted = lifted_store_path(self.backend_dir / "semantics", self.config["semantics"]["extractor"])
            return lifted.exists()
        if stage == "localize":
            feedforward_zarr = self.backend_dir / "feedforward.zarr"
            return feedforward_zarr.exists() and _localization_db_exists(
                feedforward_zarr, self.config["localization"]["extractor"]
            )
```

- [ ] **Step 4: `_resolve_result()`, directly after `_stage_output_exists`**

```python
    def _resolve_result(self) -> "PointcloudResult | None":
        """PointcloudResult for a stage-2+ run, loading from COLMAP on disk if not in memory."""
        # A stage run on its own never calls build_pointcloud(), so self.pointcloud is None even
        # when a complete reconstruction is already sitting in backend_dir.
        if self.pointcloud is None and self._stage_output_exists("pointcloud"):
            self.pointcloud = self._load_pointcloud_from_disk()
        return self.pointcloud
```

- [ ] **Step 5: Route the three leaf stages through both helpers**

In `extract_semantics`, replace the inline lifted-store check and the bare `result` use:

```python
        if not overwrite and self._stage_output_exists("semantics"):
            logger.info("Lifted features exist at %s, skipping", lifted_store_path(lifted_dir, extractor_name))
            return lifted_dir

        result = result or self._resolve_result()
```

In `mesh`, the same shape:

```python
        if not overwrite and self._stage_output_exists("mesh"):
            logger.info("Mesh exists at %s, skipping", mesh_path)
            return mesh_path

        result = result or self._resolve_result()
```

In `build_localization_db`, drop the unused `result` parameter and use the shared check:

```python
    def build_localization_db(self, overwrite: bool = False) -> Path:
```
```python
        if not overwrite and self._stage_output_exists("localize"):
```

- [ ] **Step 6: The no-op refusal in `run_pipeline`**

Capture the distinction before `stages` is reassigned, at the top of the method:

```python
        # Naming a stage means asking for it; inheriting it from config does not. Capture the
        # distinction before `stages` is reassigned below.
        named = stages is not None
```

Then extend the existing dependency-validation loop, inside `for stage in stages:` and after the `for dep ...` inner loop:

```python
            # Refuse a stage the caller NAMED whose output already exists, instead of silently
            # no-op'ing. A remote re-run would otherwise pull the whole scene, skip every stage,
            # push nothing and report success.
            if named and not overwrite and self._stage_output_exists(stage):
                raise ValueError(f"Stage '{stage}' output already exists; pass overwrite=True to replace it.")
```

And drop the dead argument at the dispatch site:

```python
            elif stage == "localize":
                self.build_localization_db(overwrite=overwrite)
```

- [ ] **Step 7: Write the failing tests**

Append to `tests/wrapper/test_reconstructor.py`:

```python
########################################
# Leaf-stage re-run (stage-rerun-from-processed)
########################################


def test_leaf_stages_derived_from_dep_graph():
    """LEAF_STAGES is whatever nothing depends on — not a hardcoded list."""
    from collab_splats.wrapper import reconstructor as R

    assert R.LEAF_STAGES == frozenset({"semantics", "mesh", "localize"})


def test_stage_output_exists_mesh(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec._stage_output_exists("mesh") is False
    rec.backend_dir.mkdir(parents=True, exist_ok=True)
    (rec.backend_dir / "mesh.ply").touch()
    assert rec._stage_output_exists("mesh") is True


def test_stage_output_exists_semantics_is_per_extractor(tmp_path):
    """The marker is this run's extractor — another extractor's lifted store must not satisfy it."""
    from collab_splats.semantics.compression import lifted_store_path

    config = _make_config(tmp_path, {"semantics": {"extractor": "dinov2"}})
    rec = Reconstructor(config)
    sem_dir = rec.backend_dir / "semantics"
    sem_dir.mkdir(parents=True)
    lifted_store_path(sem_dir, "talk2dino").mkdir()
    assert rec._stage_output_exists("semantics") is False
    lifted_store_path(sem_dir, "dinov2").mkdir()
    assert rec._stage_output_exists("semantics") is True


def test_stage_output_exists_localize(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma"}})
    rec = Reconstructor(config)
    # No zarr at all: absent, and _localization_db_exists must not even be consulted.
    assert rec._stage_output_exists("localize") is False
    (rec.backend_dir / "feedforward.zarr").mkdir(parents=True)
    with patch.object(R, "_localization_db_exists", return_value=True):
        assert rec._stage_output_exists("localize") is True
    with patch.object(R, "_localization_db_exists", return_value=False):
        assert rec._stage_output_exists("localize") is False


def _seed_pointcloud_markers(rec):
    """Make _stage_output_exists('pointcloud') true without running the stage."""
    colmap_dir = rec.backend_dir / "colmap" / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    (colmap_dir / "cameras.bin").touch()
    (rec.backend_dir / "feedforward.zarr").mkdir(parents=True, exist_ok=True)


def test_mesh_resolves_result_from_disk_when_not_in_memory(tmp_path):
    """`--stages mesh` on a pulled scene: self.pointcloud is None but COLMAP is on disk."""
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    loaded = _make_mock_pointcloud_result(tmp_path)
    rec._load_pointcloud_from_disk = lambda: loaded

    with patch.object(R, "_run_tsdf_mesh", return_value=rec.backend_dir / "mesh.ply") as run_mesh:
        rec.mesh()

    assert run_mesh.call_args.kwargs["result"] is loaded
    assert rec.pointcloud is loaded  # cached, so a second leaf stage does not re-read COLMAP


def test_mesh_without_pointcloud_on_disk_still_raises(tmp_path):
    """Nothing in memory and nothing on disk is still a hard error, not a silent skip."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    with pytest.raises(ValueError, match="No PointcloudResult"):
        rec.mesh()


def test_extract_semantics_resolves_result_from_disk(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"semantics": {"extractor": "dinov2"}})
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    loaded = _make_mock_pointcloud_result(tmp_path)
    rec._load_pointcloud_from_disk = lambda: loaded
    # 2D cache hit so the extractor never loads; only the lift is exercised.
    rec.semantics_cache_dir.mkdir(parents=True, exist_ok=True)
    (rec.semantics_cache_dir / "dinov2.zarr").mkdir()

    with patch.object(R, "_lift_and_save", return_value=rec.backend_dir / "semantics") as lift:
        rec.extract_semantics()

    lift.assert_called_once()
    assert rec.pointcloud is loaded


def test_run_pipeline_refuses_named_stage_whose_output_exists(tmp_path):
    """A named no-op must fail loudly: a remote re-run would otherwise pull GBs and push nothing."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    (rec.backend_dir / "mesh.ply").touch()

    with pytest.raises(ValueError, match="already exists"):
        rec.run_pipeline(stages=["mesh"])


def test_run_pipeline_named_stage_with_overwrite_runs(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    (rec.backend_dir / "mesh.ply").touch()
    calls = []
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh")

    rec.run_pipeline(stages=["mesh"], overwrite=True)
    assert calls == ["mesh"]


def test_run_pipeline_config_derived_stages_still_skip_silently(tmp_path):
    """stages=None comes from config enabled flags — resume behaviour must not become an error."""
    config = _make_config(tmp_path, {"mesh": {"enabled": True}})
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    (rec.backend_dir / "mesh.ply").touch()
    calls = []
    rec.preprocess = lambda overwrite=False: calls.append("preproc") or rec.frames_zarr
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud")
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh")

    rec.run_pipeline()  # must not raise
    assert calls == ["preproc", "pointcloud", "mesh"]
```

- [ ] **Step 8: Run the new tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -p no:randomly -q -k "leaf or stage_output_exists or resolves_result or refuses_named or named_stage_with_overwrite or config_derived_stages"`

Expected: all PASS. (If Task 1's source edits were absent and you applied them in Steps 2–6, run this before and after to see them go red→green.)

- [ ] **Step 9: Run the whole reconstructor + wrapper suite for regressions**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -p no:randomly -q`

Expected: PASS. Pay attention to `test_build_localization_db_skips_when_exists` and `test_build_localization_db_runs_when_missing` — they patch the module-level `_localization_db_exists`, which `_stage_output_exists("localize")` still calls, so they must stay green.

- [ ] **Step 10: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): leaf-stage markers, disk result resolution, no-op refusal"
```

---

### Task 2: batch.py — tolerate a scene with no local video

**Files:**
- Modify: `collab_splats/wrapper/batch.py:81-118`
- Test: `tests/wrapper/test_batch.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/wrapper/test_batch.py`:

```python
########
# Processed re-run: no local video
########


def test_build_scene_config_without_video_keeps_pulled_input_path(tmp_path):
    """A processed re-run has no local video; the pulled config's input_path must survive."""
    override = {"input_path": "/on/another/machine/C0043.MP4", "mesh": {"voxel_size": 0.01}}
    config = batch.build_scene_config(None, tmp_path / "out", override, name="2026_07_20-birds-C0043")
    assert config["input_path"] == "/on/another/machine/C0043.MP4"
    assert config["output_path"] == str(tmp_path / "out" / "2026_07_20-birds-C0043")


def test_build_scene_config_with_video_sets_input_path(tmp_path):
    video = tmp_path / "C0043.MP4"
    config = batch.build_scene_config(video, tmp_path / "out", {"input_path": "/stale/path.mp4"})
    assert config["input_path"] == str(video)


def test_run_scene_rewrites_existing_run_config(tmp_path, monkeypatch):
    """A pulled scene always ships a run_config.yaml; it must be replaced by what actually ran."""
    out_dir = tmp_path / "out" / "scene"
    out_dir.mkdir(parents=True)
    (out_dir / "run_config.yaml").write_text(yaml.dump({"mesh": {"voxel_size": 0.99}}))

    class _FakeReconstructor:
        def __init__(self, config, config_dir=None):
            self.config = dict(config)
            self.config["mesh"] = {"voxel_size": 0.005}

        def run_pipeline(self, stages=None, overwrite=False):
            pass

    monkeypatch.setattr(batch, "Reconstructor", _FakeReconstructor)
    batch.run_scene(None, tmp_path / "out", None, {"output_path": str(out_dir)}, ["mesh"], False, name="scene")

    written = yaml.safe_load((out_dir / "run_config.yaml").read_text())
    assert written["mesh"]["voxel_size"] == 0.005
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_batch.py -p no:randomly -q -k "without_video or with_video_sets or rewrites_existing"`

Expected: FAIL — `build_scene_config` raises/overwrites `input_path` with `"None"`, and `run_scene` leaves the stale `voxel_size: 0.99`.

- [ ] **Step 3: Implement**

In `build_scene_config`:

```python
    config = merge({}, override_config) if override_config else {}
    # A processed re-run has no local video: keep the pulled config's input_path, which records the
    # original run's input and is read by nothing when preproc does not run.
    if video is not None:
        config["input_path"] = str(video)
    config["output_path"] = str(scene_output_dir(video, output_root, name=name))
```

In `run_scene`, drop the condition around the write:

```python
    # Always rewrite: r.config is by definition what ran. A pulled scene arrives with the original
    # run's config, so a conditional write would push back a file describing a different run.
    run_cfg = output_path / "run_config.yaml"
    with open(run_cfg, "w") as f:
        yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_batch.py -p no:randomly -q`

Expected: PASS, whole file.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/batch.py tests/wrapper/test_batch.py
git commit -m "feat(wrapper): batch tolerates a scene with no local video"
```

---

### Task 3: `remote/rerun.py` — routing, pull, config merge

**Files:**
- Create: `collab_splats/remote/rerun.py`
- Modify: `collab_splats/remote/__init__.py`
- Test: `tests/remote/test_rerun.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/remote/test_rerun.py`:

```python
"""prepare_scene/discover_scenes: leaf-only routing, failure modes, config merge."""

from pathlib import Path

import pytest
import yaml

from collab_splats.remote.rerun import discover_scenes, prepare_scene

SCENE = "2026_07_20-birds-C0043"

PULLED_CONFIG = {
    "input_path": "/on/another/machine/C0043.MP4",
    "output_path": "/on/another/machine/out",
    "preprocessing": {"frame_selection": "uniform", "max_frames": 250},
    "pointcloud": {"method": "feedforward", "backend": "vggtx"},
    "mesh": {"enabled": True, "voxel_size": 0.02},
}


class _FakeSource:
    """SceneSource stand-in: records calls, writes the files a real pull would land."""

    def __init__(self, processed=True, run_config=PULLED_CONFIG, processed_scenes=(SCENE,)):
        self._processed = processed
        self._run_config = run_config
        self._processed_scenes = list(processed_scenes)
        self.calls = []

    def has_processed(self, scene):
        self.calls.append(("has_processed", scene))
        return self._processed

    def pull_processed(self, scene, dest_dir, on_line=None):
        self.calls.append(("pull_processed", scene, Path(dest_dir)))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if self._run_config is not None:
            (dest / "run_config.yaml").write_text(yaml.dump(self._run_config))
        return dest

    def fetch_video(self, scene, dest_dir, on_line=None):
        self.calls.append(("fetch_video", scene, Path(dest_dir)))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        video = dest / "C0043.MP4"
        video.touch()
        return video

    def list_scenes(self):
        self.calls.append(("list_scenes",))
        return ["curated-only-scene", SCENE]

    def list_processed_scenes(self):
        self.calls.append(("list_processed_scenes",))
        return self._processed_scenes


########
# Routing
########


def test_leaf_stages_pull_from_processed(tmp_path):
    source = _FakeSource()
    video, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)
    assert video is None
    assert ("pull_processed", SCENE, tmp_path / SCENE) in source.calls
    assert not any(c[0] == "fetch_video" for c in source.calls)
    # Provenance for stages NOT being re-run survives verbatim.
    assert config["pointcloud"]["backend"] == "vggtx"
    assert config["preprocessing"]["max_frames"] == 250


def test_any_upstream_stage_fetches_the_curated_video(tmp_path):
    """pointcloud is not a leaf, so the whole scene is rebuilt from the video."""
    source = _FakeSource()
    video, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["pointcloud", "mesh"], None)
    assert video == tmp_path / SCENE / "C0043.MP4"
    assert config is None
    assert not any(c[0] == "pull_processed" for c in source.calls)


def test_stages_none_fetches_the_curated_video(tmp_path):
    source = _FakeSource()
    video, _ = prepare_scene(source, SCENE, tmp_path / SCENE, None, None)
    assert video == tmp_path / SCENE / "C0043.MP4"


def test_override_config_passes_through_on_the_curated_path(tmp_path):
    source = _FakeSource()
    override = {"mesh": {"voxel_size": 0.001}}
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, None, override)
    assert config is override


########
# Failure modes — each raises before any compute, landing as a FAIL row in the driver
########


def test_unprocessed_scene_raises(tmp_path):
    source = _FakeSource(processed=False)
    with pytest.raises(FileNotFoundError, match="no processed outputs"):
        prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)


def test_pull_without_run_config_raises(tmp_path):
    source = _FakeSource(run_config=None)
    with pytest.raises(FileNotFoundError, match="run_config.yaml"):
        prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)


def test_backend_mismatch_raises(tmp_path):
    """Config is authoritative and never silently retargeted: disagreeing with the data is fatal."""
    source = _FakeSource()
    override = {"pointcloud": {"backend": "vggt_omega"}}
    with pytest.raises(ValueError, match="vggtx"):
        prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], override)


def test_matching_backend_is_accepted(tmp_path):
    source = _FakeSource()
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], {"pointcloud": {"backend": "vggtx"}})
    assert config["pointcloud"]["backend"] == "vggtx"


########
# Config merge
########


def test_rerun_stage_section_is_dropped_so_base_yaml_supplies_it(tmp_path):
    source = _FakeSource()
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)
    assert "mesh" not in config  # Reconstructor merges base.yaml's mesh section over the gap


def test_override_config_wins_over_pulled(tmp_path):
    source = _FakeSource()
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], {"mesh": {"voxel_size": 0.001}})
    assert config["mesh"]["voxel_size"] == 0.001


def test_localize_drops_the_localization_section(tmp_path):
    """Stage name and config section differ only for localize."""
    source = _FakeSource(run_config={**PULLED_CONFIG, "localization": {"enabled": True, "extractor": "disk"}})
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["localize"], None)
    assert "localization" not in config
    assert config["mesh"]["voxel_size"] == 0.02  # untouched stage keeps its provenance


def test_plan_is_logged(tmp_path, caplog):
    source = _FakeSource()
    with caplog.at_level("INFO"):
        prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)
    assert "mesh" in caplog.text and "vggtx" in caplog.text


########
# --all bucket choice
########


def test_discover_scenes_lists_processed_for_a_leaf_rerun():
    source = _FakeSource()
    assert discover_scenes(source, ["mesh"]) == [SCENE]
    assert ("list_processed_scenes",) in source.calls


def test_discover_scenes_lists_curated_otherwise():
    source = _FakeSource()
    assert discover_scenes(source, None) == ["curated-only-scene", SCENE]
    assert discover_scenes(source, ["preproc", "pointcloud"]) == ["curated-only-scene", SCENE]
    assert not any(c[0] == "list_processed_scenes" for c in source.calls)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/remote/test_rerun.py -p no:randomly -q`

Expected: collection error — `ModuleNotFoundError: No module named 'collab_splats.remote.rerun'`.

- [ ] **Step 3: Create the module**

Write `collab_splats/remote/rerun.py`:

```python
"""Route a scene's inputs for a re-run: curated video, or an already-processed scene pulled back.

A stage set containing only LEAF_STAGES (nothing depends on them) can be re-run against processed
data. Anything upstream rebuilds from the curated video, so a run can never leave a stale
downstream artifact beside a freshly-regenerated one.
"""

import logging
from pathlib import Path

import yaml
from mergedeep import merge

from collab_splats.wrapper.reconstructor import LEAF_STAGES

logger = logging.getLogger(__name__)


def _is_rerun(stages) -> bool:
    """True when every requested stage is a leaf, so processed outputs are enough to run it."""
    return bool(stages) and set(stages) <= LEAF_STAGES


def discover_scenes(source, stages) -> list[str]:
    """Scene ids for an --all run: the processed bucket for a leaf re-run, curated otherwise."""
    # Listing curated for a leaf re-run would hand the batch every scene that has no processed
    # outputs and fail each one on a FileNotFoundError it could have avoided asking for.
    if _is_rerun(stages):
        return source.list_processed_scenes()
    return source.list_scenes()


def prepare_scene(source, scene: str, scene_dir: Path, stages, override_config, on_line=None):
    """Fetch a scene's inputs; return (video, override_config) for batch.run_scene.

    Leaf-only stage sets re-run from environments-processed and return video=None; anything else
    starts from the curated video and passes override_config through untouched.
    """
    # Route on the dependency graph, not a hardcoded list.
    if not _is_rerun(stages):
        return source.fetch_video(scene, scene_dir, on_line=on_line), override_config

    if not source.has_processed(scene):
        raise FileNotFoundError(f"{scene} has no processed outputs — run the full pipeline first")

    # No excludes: PULL_EXCLUDES is the viewer's default and drops depth/images, which is exactly
    # what meshing reads. Correct-by-construction beats a per-stage member table that silently
    # starves a stage when what it reads changes.
    source.pull_processed(scene, scene_dir, on_line=on_line)

    # The pulled run_config is the only record of which backend produced this scene, and the
    # backend names the subdir every artifact path is built from — no config, no run.
    run_cfg = Path(scene_dir) / "run_config.yaml"
    if not run_cfg.exists():
        raise FileNotFoundError(f"{scene}: pulled scene has no run_config.yaml; backend is unknowable")
    with open(run_cfg) as f:
        pulled = yaml.safe_load(f)
    backend = pulled["pointcloud"]["backend"]

    # Retargeting must be typed, and a typed one that disagrees with the data is a mistake.
    asked = (override_config or {}).get("pointcloud", {}).get("backend")
    if asked and asked != backend:
        raise ValueError(f"{scene} was built with backend '{backend}', --config asks for '{asked}'")

    # The pulled config carries provenance for every stage NOT being re-run; dropping the re-run
    # stages' sections lets base.yaml + --config supply fresh params for exactly those.
    cfg = merge({}, pulled)
    for stage in stages:
        # Stage name == config section, except localize → localization.
        cfg.pop("localization" if stage == "localize" else stage, None)
    logger.info("%s: re-run %s from processed (backend=%s)", scene, ",".join(stages), backend)
    return None, merge(cfg, override_config or {})
```

- [ ] **Step 4: Export it**

In `collab_splats/remote/__init__.py`, add the import and both `__all__` entries:

```python
from collab_splats.remote.rerun import discover_scenes, prepare_scene
```
```python
    "discover_scenes",
    "prepare_scene",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/remote/ -p no:randomly -q`

Expected: PASS, including the existing `test_sources.py`.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/remote/rerun.py collab_splats/remote/__init__.py tests/remote/test_rerun.py
git commit -m "feat(remote): prepare_scene routes leaf-stage re-runs to processed outputs"
```

---

### Task 4: Wire the driver

**Files:**
- Modify: `docs/examples/run_pipeline_remote.py:50` (import), `:89` (`--all` listing), `:107-113` (per-scene fetch), `:11-40` (docstring)

- [ ] **Step 1: Import both helpers**

```python
from collab_splats.remote import PUSH_EXCLUDES, SCENE_ID_RE, SceneSource, discover_scenes, prepare_scene
```

- [ ] **Step 2: Pick the right bucket for `--all`**

Replace `scene_ids = source.list_scenes()` inside the existing `try:` with:

```python
            # A leaf-only --stages set re-runs from environments-processed, so that is the bucket
            # the work list must come from; anything else starts from curated video.
            scene_ids = discover_scenes(source, stages)
```

- [ ] **Step 3: Replace the per-scene fetch**

```python
            # 1. Fetch inputs: the curated video, or the processed scene pulled back for a
            # leaf-stage re-run (which returns video=None and a config carrying its provenance)
            video, scene_config = prepare_scene(
                source, scene, scene_dir, stages, override_config, on_line=logger.info
            )

            # 2. Same pipeline as the local driver; name= pins the output dir to the scene id
            out, _ = batch.run_scene(
                video, output_root, config_dir, scene_config, stages, overwrite, name=scene
            )
```

- [ ] **Step 4: Document it in the module docstring**

Insert after the `--keep-local` usage block:

```
    # Re-run only the leaf stages (mesh, semantics, localize) against already-processed
    # scenes: the scene is pulled back out of environments-processed instead of being
    # rebuilt from video. --all then lists environments-processed, not curated.
    python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs \\
        --stages mesh --overwrite --config remesh.yaml --all

A --stages set that includes preproc or pointcloud always rebuilds from the curated video,
so a re-run can never leave a stale downstream artifact behind. A leaf re-run of a scene with
no processed outputs fails that scene; the config is authoritative, so a --config backend that
disagrees with the pulled run_config.yaml is an error rather than a silent retarget. Naming a
stage whose output already exists is refused — pass --overwrite to replace it.
```

- [ ] **Step 5: Verify the driver still parses and its help renders**

Run: `/opt/venv/reconstruction/bin/python docs/examples/run_pipeline_remote.py --help`

Expected: exit 0, help text includes the new re-run paragraph.

- [ ] **Step 6: Dry-check the routing without touching GCS**

Run:

```bash
/opt/venv/reconstruction/bin/python - <<'PY'
from collab_splats.remote import discover_scenes, prepare_scene
from collab_splats.wrapper.reconstructor import LEAF_STAGES
print("leaf:", sorted(LEAF_STAGES))
class S:
    def list_scenes(self): return ["curated"]
    def list_processed_scenes(self): return ["processed"]
print("--stages mesh ->", discover_scenes(S(), ["mesh"]))
print("--stages pointcloud,mesh ->", discover_scenes(S(), ["pointcloud", "mesh"]))
print("--stages None ->", discover_scenes(S(), None))
PY
```

Expected:
```
leaf: ['localize', 'mesh', 'semantics']
--stages mesh -> ['processed']
--stages pointcloud,mesh -> ['curated']
--stages None -> ['curated']
```

- [ ] **Step 7: Commit**

```bash
git add docs/examples/run_pipeline_remote.py
git commit -m "feat(remote): wire the remote driver to leaf-stage re-runs"
```

---

### Task 5: Docs and full-suite gate

**Files:**
- Modify: `configs/README.md`, `CLAUDE.md`

- [ ] **Step 1: Document the contract in `configs/README.md`**

Append to the processed-scene output-contract section:

```markdown
### Re-running a single stage from processed outputs

`--stages` containing only leaf stages (`mesh`, `semantics`, `localize` — nothing depends on
them) pulls the scene back out of `environments-processed` instead of rebuilding it from the
curated video. Any set that includes `preproc` or `pointcloud` rebuilds from video, so a re-run
can never leave a stale downstream artifact beside a fresh one.

Rules:
- The scene must already be in `environments-processed`; if it is not, that scene fails.
- The pulled `run_config.yaml` names the backend. A `--config` that asks for a different backend
  is an error — nothing is silently retargeted.
- The re-run stage's own config section is dropped from the pulled config, so `base.yaml` plus
  `--config` supply fresh params for exactly that stage; every other stage keeps its provenance.
- Naming a stage whose output already exists is refused. Pass `--overwrite` to replace it.
- The whole scene is pulled (no `PULL_EXCLUDES`) — `depth` and `images` are what meshing reads.
- The push is still `rclone copy`: nothing in this path deletes a remote object.
```

- [ ] **Step 2: Move the item to "Recently completed" in `CLAUDE.md`**

Add above the `gcloud-remote-pipeline` entry:

```markdown
Recently completed (2026-08-11): **stage-rerun-from-processed** — leaf-only stage re-runs
(`mesh`/`semantics`/`localize`) pull from `environments-processed` instead of rebuilding from
curated video; `LEAF_STAGES` derived from `_STAGE_DEPS`, named no-op stages refused,
`collab_splats/remote/rerun.py:prepare_scene` ([spec](docs/superpowers/specs/2026-08-11-stage-rerun-from-processed-design.md) · [plan](docs/superpowers/plans/2026-08-11-stage-rerun-from-processed.md)).
```

- [ ] **Step 3: Format**

Run: `/opt/venv/reconstruction/bin/python -m black collab_splats/remote/rerun.py collab_splats/wrapper/reconstructor.py collab_splats/wrapper/batch.py docs/examples/run_pipeline_remote.py tests/remote/test_rerun.py tests/wrapper/test_reconstructor.py tests/wrapper/test_batch.py && /opt/venv/reconstruction/bin/python -m isort collab_splats/remote/rerun.py collab_splats/remote/__init__.py docs/examples/run_pipeline_remote.py tests/remote/test_rerun.py`

Expected: reformats or reports unchanged. **Never** run repo-wide `black .` — the venv's black is newer than the repo's formatting.

- [ ] **Step 4: Full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly -q`

Expected: no new failures against `docs/known-test-failures.md`.

- [ ] **Step 5: Dashboard smoke gate**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`

Expected: `SMOKE PASS`. (`build_localization_db` lost a parameter; this proves no dashboard path passed it.)

- [ ] **Step 6: Refresh the knowledge graph**

Run: `graphify update .`

- [ ] **Step 7: Commit**

```bash
git add configs/README.md CLAUDE.md docs/superpowers/plans/2026-08-11-stage-rerun-from-processed.md
git commit -m "docs(remote): document leaf-stage re-runs from processed outputs"
```

---

## Deviation from the plan as written

Task 1's refusal was scoped to leaf stages after review: `if named and stage in LEAF_STAGES and
not overwrite and self._stage_output_exists(stage)`. As originally written it fired for *any*
named stage, which breaks the retry documented at `docs/examples/run_pipeline.py:31`
(`--stages preproc,pointcloud,localize scene.MP4`) — `scene_output_dir` is deterministic, so
retrying after a `localize` failure re-enters the same dir with the two upstream stages already
done and would now raise instead of resuming. Every re-run set is leaf-only by construction, so
the scoping costs the feature nothing. Spec §3 updated to match; covered by
`test_run_pipeline_named_upstream_stages_resume_instead_of_refusing`.

Task 3's re-export was also dropped. `collab_splats/remote/__init__.py` must NOT import
`rerun.py`: `dashboard/operation_log.py` imports `collab_splats.remote` on the fast-bind path, and
`rerun` imports `LEAF_STAGES` from `wrapper.reconstructor`, which pulls torch and pyvista — so the
export as planned broke `tests/dashboard/test_serve.py::test_light_import_path_stays_light`. There
is no light home for the stage graph under `wrapper/` (`wrapper/__init__.py` imports
`Reconstructor`), so the boundary is structural instead: the driver and tests import
`from collab_splats.remote.rerun import ...`. Pinned by
`test_importing_the_remote_package_stays_light`.

## Owed after this plan

One live run watched by a human — no test touches GCS:

```bash
/opt/venv/reconstruction/bin/python docs/examples/run_pipeline_remote.py \
    --output-root /workspace/outputs --stages mesh --overwrite <one-scene-id>
```

Confirm: pull streams progress → `mesh.ply` rewritten under the right backend dir → push verifies
→ local dir removed → exit 0.
