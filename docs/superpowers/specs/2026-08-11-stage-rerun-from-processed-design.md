# Stage re-run from processed scenes — design

Date: 2026-08-11
Status: approved, not implemented
Related: [gcloud-remote-pipeline](2026-07-29-gcloud-remote-pipeline-design.md)

## Problem

`docs/examples/run_pipeline_remote.py` reconstructs a curated scene end to end and pushes the
result to `environments-processed/<scene>/`. There is no way to re-run one stage. Re-meshing a
scene with a new `voxel_size` today means re-running VGGT inference for the whole scene.

The data needed for a partial re-run is already in the bucket. `PUSH_EXCLUDES` strips only
`/semantics/**` (the regenerable 2D patch cache) and video files, so a processed scene carries
`frames.zarr`, `<backend>/colmap/sparse/0/`, and a complete `<backend>/feedforward.zarr`
including the `depth` and `images` arrays `_run_tsdf_mesh` reads. Nothing new has to be
uploaded to make re-runs possible.

Four things block it:

1. `run_pipeline_remote.py` only ever calls `SceneSource.fetch_video()`. `pull_processed()`
   exists on `SceneSource` and is unwired.
2. `--stages mesh` passes `run_pipeline`'s dependency check (`reconstructor.py:896`) when
   COLMAP + zarr are on disk, then dies inside `mesh()`: `result or self.pointcloud`
   (`:807`) and `self.pointcloud` is `None` until `build_pointcloud()` sets it. Same hole in
   `extract_semantics()` (`:768`). No disk fallback exists.
3. `--overwrite` is a single per-run flag, so beating a stage's exists-skip also forces
   pointcloud re-inference when pointcloud is in the stage list.
4. `PULL_EXCLUDES` — the dashboard's viewer default — excludes `depth/**` and `images/**`,
   the two arrays meshing needs. Any re-run pull must not use it.

## Goals

Re-run `mesh`, `semantics`, or `localize` for a scene in `environments-processed` without
re-running preprocessing or pointcloud inference, and push the result back to the same place.

## Non-goals

- Re-running `pointcloud` while keeping downstream artifacts. Structurally forbidden (below).
- Any destructive operation against `environments-processed`. `push_outputs` stays
  `rclone copy`; nothing in this design deletes a remote object.
- Fixing the `RcloneClient._cmd` private-API coupling (see Known limitations).

## Core rule

```
--stages ⊆ LEAF_STAGES        → pull environments-processed/<scene>/ → re-run → push
otherwise, including no --stages → fetch curated video → full pipeline → push
```

A stage is a leaf when nothing depends on it. That is a derived property of the dependency
graph already in `reconstructor.py:47`, not a new declaration:

```python
# Re-runnable in isolation iff no other stage depends on it → {semantics, mesh, localize}.
# Derived, not hardcoded: a future stage that depends on mesh drops mesh from this set and
# the re-run path refuses it without anyone remembering to update a list.
LEAF_STAGES = frozenset(s for s in _STAGE_ORDER if not any(s in d for d in _STAGE_DEPS.values()))
```

### Why leaf-only

Every stage worth re-running in isolation is a leaf. `preproc` and `pointcloud` have nothing
but dependents, so re-running either one invalidates everything downstream — and a run that
regenerates everything downstream is the full pipeline. Splitting on leaf-ness therefore
removes, rather than solves, the hard problems:

- **No invalidation logic.** Leaf re-runs cannot stale anything. Upstream re-runs regenerate
  everything by definition.
- **No remote deletes.** Nothing goes stale in the bucket, so nothing has to be pruned from
  it. `push_outputs` stays non-destructive.
- **No config staleness classes.** A leaf re-run only reconfigures leaves.
- **No silent retargeting.** The backend comes from the pulled scene's own `run_config.yaml`,
  so pointing a run at a different backend requires typing it (and is then refused if it
  disagrees with the data — see below).

The only capability this forbids is re-running `pointcloud` while keeping the old `mesh.ply`,
which would leave a mesh describing a pointcloud that no longer exists.

## Components

### 1. `Reconstructor` — resolve the pointcloud from disk

`collab_splats/wrapper/reconstructor.py`. Required for `--stages mesh` to work at all,
independent of GCS; the local driver gains the same ability.

```python
def _resolve_result(self) -> "PointcloudResult | None":
    """PointcloudResult for a stage-2+ run, loading from COLMAP on disk if not in memory."""
    # A stage run on its own never calls build_pointcloud(), so self.pointcloud is None even
    # when a complete reconstruction is sitting in backend_dir.
    if self.pointcloud is None and self._stage_output_exists("pointcloud"):
        self.pointcloud = self._load_pointcloud_from_disk()
    return self.pointcloud
```

`mesh()` (`:807`) and `extract_semantics()` (`:768`) change `result or self.pointcloud` to
`result or self._resolve_result()`, keeping their existing `ValueError` for the case where
nothing is on disk either.

`build_localization_db()` (`:831`) needs no such change and instead loses a parameter: it
declares `result: PointcloudResult | None = None` and never reads it, while `run_pipeline:916`
dutifully passes `result=result`. The stage dispatch is an if/elif chain with distinct calls,
not a uniform loop, so the parameter buys no interface symmetry. Drop it from both sites.

### 2. `Reconstructor._stage_output_exists` — cover the leaf stages

Already exists (`:854`) and already returns the right answer for `preproc` and `pointcloud`.
It returns `False` for everything else ("others have no reusable marker"). Filling in the
three leaves lets one function serve three callers instead of one, and lets the three stage
methods drop their hand-rolled duplicates of the same probes:

```python
    # Leaf markers: previously unmodelled because only preproc/pointcloud are ever depended
    # on. Needed now so a re-run can be refused before pulling GBs, and so each stage's own
    # skip-check has a single implementation.
    if stage == "mesh":
        return (self.backend_dir / "mesh.ply").exists()
    if stage == "semantics":
        return lifted_store_path(self.backend_dir / "semantics", self.config["semantics"]["extractor"]).exists()
    if stage == "localize":
        ff = self.backend_dir / "feedforward.zarr"
        return ff.exists() and _localization_db_exists(ff, self.config["localization"]["extractor"])
```

`lifted_store_path` and `_localization_db_exists` are both already imported/defined in the
module. `mesh()`, `extract_semantics()`, and `build_localization_db()` then replace their
inline exists-checks with `if not overwrite and self._stage_output_exists(<stage>)`.

### 3. `Reconstructor.run_pipeline` — refuse a no-op named stage

Extends the loop that already validates dependencies (`:894`):

```python
        # Refuse a stage the caller NAMED whose output already exists, instead of silently
        # no-op'ing. A remote re-run would otherwise pull the whole scene, skip every stage,
        # push nothing and report success.
        if not overwrite and self._stage_output_exists(stage):
            raise ValueError(
                f"Stage '{stage}' output already exists; pass overwrite=True to replace it."
            )
```

**This applies only when `stages` was passed explicitly.** When `stages is None` the set comes
from the config's `enabled` flags, and today's skip-completed-stages behaviour is what makes a
re-run of `run_pipeline.py` resume rather than fail. Naming a stage means asking for it;
inheriting it from config does not. `run_pipeline` reassigns `stages` when it defaults, so
capture the distinction first: `named = stages is not None`, and guard the new check with it.

Audited against every caller: `dashboard/pipeline.py:380` defines its own unrelated
`run_pipeline` and constructs no `Reconstructor`, so the dashboard is untouched. The six
explicit-stage test call sites in `tests/wrapper/test_reconstructor.py` all pass, because the
check fires only when a *named* stage's own output exists —
`test_run_pipeline_dep_satisfied_by_existing_output` (`:680`) creates `frames.zarr` but names
only `pointcloud`, whose marker is absent.

### 4. `collab_splats/remote/rerun.py` — new, one function

Routing, transport and config assembly. No filesystem-layout knowledge: that lives in
`Reconstructor`.

```python
def prepare_scene(source, scene, scene_dir, stages, override_config, on_line=None):
    """Fetch a scene's inputs; return (video, override_config) for batch.run_scene.

    Leaf-only stage sets re-run from environments-processed; anything else starts from the
    curated video, so no run can leave a stale downstream artifact behind.
    """
    # Route on the dependency graph, not a hardcoded list.
    if not stages or not set(stages) <= LEAF_STAGES:
        return source.fetch_video(scene, scene_dir, on_line=on_line), override_config

    if not source.has_processed(scene):
        raise FileNotFoundError(f"{scene} has no processed outputs — run the full pipeline first")

    # No excludes: PULL_EXCLUDES is the viewer's default and drops depth/images, which is
    # exactly what meshing reads. Correct-by-construction beats a per-stage member table that
    # silently starves a stage when what it reads changes.
    source.pull_processed(scene, scene_dir, on_line=on_line)

    # The pulled run_config is the only record of which backend produced this scene, and the
    # backend names the subdir every artifact path is built from — no config, no run.
    run_cfg = scene_dir / "run_config.yaml"
    if not run_cfg.exists():
        raise FileNotFoundError(f"{scene}: pulled scene has no run_config.yaml; backend is unknowable")
    with open(run_cfg) as f:
        pulled = yaml.safe_load(f)
    backend = pulled["pointcloud"]["backend"]

    # Retargeting must be typed, and a typed one that disagrees with the data is a mistake.
    asked = (override_config or {}).get("pointcloud", {}).get("backend")
    if asked and asked != backend:
        raise ValueError(f"{scene} was built with backend '{backend}', --config asks for '{asked}'")

    # Pulled config carries provenance for every stage NOT being re-run; dropping the re-run
    # stages' sections lets base.yaml + --config supply fresh params for exactly those.
    cfg = merge({}, pulled)
    for stage in stages:
        cfg.pop(_STAGE_CONFIG_SECTION[stage], None)
    logger.info("%s: re-run %s from processed (backend=%s)", scene, ",".join(stages), backend)
    return None, merge(cfg, override_config or {})
```

`merge` is `mergedeep.merge`, already the merge tool in `batch.py` and `reconstructor.py`.
Stage names and config sections are identical except for `localize`, so the section lookup is a
conditional rather than a table:

```python
    # Stage name == config section, except localize → localization.
    cfg.pop("localization" if stage == "localize" else stage, None)
```

### 5. `wrapper/batch.py` — two small changes

- `build_scene_config`: set `input_path` only when `video is not None`. A processed re-run has
  no local video and keeps the pulled value, which records the original run's input and is
  read by nothing when `preproc` does not run.
- `run_scene`: `run_config.yaml` is currently written `if not run_cfg.exists() or overwrite`. A
  pulled scene always has one, so adding `semantics` to a scene that never had it (no
  `--overwrite`, nothing to replace) would push back a config describing the original run's
  params. Drop the condition and always write it: `r.config` is by definition what ran, so
  rewriting is never wrong, and it costs no new code. The `--overwrite` path already rewrote.

### 6. `docs/examples/run_pipeline_remote.py` — one call site

Step 1 of the per-scene loop becomes `video, scene_config = prepare_scene(...)`, and
`scene_config` is passed to `batch.run_scene` in place of `override_config`. Everything else —
the failure isolation, the `check_available()` re-probe, the SKIPPED rows, the summary, the
exit codes — is unchanged.

## Usage

```bash
# Re-mesh every processed scene with a new voxel size
/opt/venv/reconstruction/bin/python docs/examples/run_pipeline_remote.py \
    --output-root /workspace/outputs --stages mesh --overwrite --config remesh.yaml --all

# Add semantics to one scene that was reconstructed without it (no --overwrite: nothing to replace)
/opt/venv/reconstruction/bin/python docs/examples/run_pipeline_remote.py \
    --output-root /workspace/outputs --stages semantics 2026_07_20-birds-C0043
```

## Data flow

`--stages mesh 2026_05_02-birds-C0041`:

```
has_processed ✓ → pull full scene → read run_config (backend=vggtx)
  → cfg = pulled − {mesh section} + base.yaml mesh + --config
  → log resolved plan → run_scene(video=None, stages=["mesh"], overwrite=True)
      → run_pipeline: dep check (pointcloud on disk ✓), no-op check (overwrite ✓)
      → mesh(): _resolve_result() reads COLMAP → TSDF → vggtx/mesh.ply
  → push (copy) → rclone check --one-way ✓ → rmtree local
```

## Error handling

Every new failure raises before any compute and lands in the existing per-scene `except
Exception` handler as a `FAIL` row; the batch continues, the `check_available()` re-probe still
promotes a dead remote to exit 3, and exit codes are unchanged. No new exception classes: the
driver records `str(exc)`, so message quality is what matters.

| Condition | Raised | Message |
|---|---|---|
| Leaf stages, scene not in `environments-processed` | `FileNotFoundError` | `no processed outputs — run the full pipeline first` |
| Pulled scene has no `run_config.yaml` | `FileNotFoundError` | `backend is unknowable` |
| `--config` backend ≠ pulled backend | `ValueError` | names both values |
| Named stage's output exists, no `--overwrite` | `ValueError` | `pass overwrite=True to replace it` |
| Pull landed without COLMAP/zarr | `FileNotFoundError` | from the existing checks in `mesh()` / `build_localization_db()` |

## Testing

Extend existing files where they exist; one new file for the new module.

- `tests/remote/test_rerun.py` (new; fake `SceneSource`, `tmp_path`): leaf set routes to
  processed · any upstream stage routes to curated · `stages=None` routes to curated ·
  missing processed raises · missing `run_config.yaml` raises · backend mismatch raises ·
  config merge keeps non-re-run sections verbatim and drops re-run sections · plan logged.
- `tests/wrapper/test_reconstructor.py`: `mesh`/`extract_semantics` resolve from disk when
  `self.pointcloud is None`; still `ValueError` with nothing on disk; `_stage_output_exists`
  for each leaf; named stage with existing output raises; `stages=None` still skips.
- `tests/wrapper/test_batch.py`: `video=None` preserves `input_path`; `run_config.yaml`
  rewritten on config drift.

No live GCS in tests. Run with `-p no:randomly`.

## Implementation principles

- Reuse before adding: `_stage_output_exists`, `_load_pointcloud_from_disk`,
  `lifted_store_path`, `_localization_db_exists`, `mergedeep.merge`, `pull_processed`,
  `has_processed`, and the whole batch loop already exist and are used as-is.
- Retire what the change obsoletes: the three inline exists-checks in `mesh()`,
  `extract_semantics()`, and `build_localization_db()` collapse into `_stage_output_exists`,
  and `build_localization_db`'s unused `result` parameter goes with them.
- One new module, one new function, no new classes, no new exception types.
- Every added block gets an inline comment saying why, per the repo's comment standard.

## Necessity audit

Cut on review, each replaced by something already present:

| Cut | Why it wasn't needed |
|---|---|
| `ScenePlan` dataclass | Two fields. `prepare_scene` returns a tuple, as `run_scene` already does. |
| `SceneNotProcessed` / `BackendMismatch` / `OutputExists` classes | The driver catches `Exception` and records `str(exc)`. Message quality is what the operator sees; the type is never matched on. |
| `_STAGE_CONFIG_SECTION` dict | Only `localize` differs from its section name — a conditional, not a table. |
| Config-diff helper for `run_config.yaml` | Always writing it is correct by definition and is strictly less code than diffing and logging. |
| Per-stage zarr member allowlists | A full pull cannot starve a stage; a member table can, silently, whenever what a stage reads changes. |
| Invalidation graph + remote pruner | The leaf-only rule makes staleness unreachable rather than handled. |

Surviving additions, and what breaks without each:

- `_resolve_result()` — without it every leaf stage raises `ValueError` on a scene it could
  have read from disk. Two call sites, so a method rather than a duplicated probe.
- `_stage_output_exists` leaf branches — without them the no-op refusal cannot be made, and a
  `--stages mesh` run pulls GBs, skips, pushes nothing, and reports OK. Net LOC is negative:
  three inline checks collapse into it.
- The refusal itself — the only thing standing between an operator and a silent no-op.
- `LEAF_STAGES` — one derived line; the alternative hardcoded set rots when a stage is added.
- `rerun.py` — the one genuinely new file. Could live in the driver, but `docs/examples/` has
  no test file, and the routing and merge rules are the parts most likely to be wrong.
- `build_scene_config`'s `video is not None` guard — the alternative is passing the pulled
  video path as a `video` that does not exist locally, which is one line shorter and a lie.

## Known limitations

- `localize` is a leaf that mutates an upstream artifact: it writes the
  `local_features/<extractor>/` group *inside* `feedforward.zarr` rather than a file of its
  own. Benign (additive; `rclone copy` re-sends only changed chunks) but it means
  `feedforward.zarr` is not immutable after the pointcloud stage.
- A from-curated re-run leaves orphan lifted stores when the configured extractor changed
  (old `talk2dino_lifted.zarr` beside a new `dinov2_lifted.zarr`). Pre-existing; not
  introduced here.
- `verify_push` re-hashes the whole scene tree even when one file changed.
- A processed re-run's `run_config.yaml` keeps the original machine's `input_path`.
- `SceneSource` depends on `RcloneClient._cmd`, a private method of `collab-data`
  (`BasisResearch/collab-data`, installed non-editable). Six call sites would break at runtime
  if it were renamed. Out of scope here; worth its own decision doc.
