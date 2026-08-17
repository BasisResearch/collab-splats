# Scene-id filter: path-safe, not date-shaped

**Date:** 2026-08-17
**Status:** Approved
**Problem:** `run_pipeline_remote.py --all` never processed the nine
`audiomoth_only_deployments-*` folders in `environments-curated`.

## Root cause (verified)

`SCENE_ID_RE = re.compile(r"^\d{4}_\d{2}_\d{2}-.+$")` (`collab_splats/remote/sources.py:33`)
requires a leading `YYYY_MM_DD-`. The audiomoth curated dirs are named
`audiomoth_only_deployments-<YYYYMMDD_YYYYMMDD>-<site>-splat_videos-<VIDEO>` and do not match.

Consequences, each confirmed against the live bucket:

1. `list_scenes()` filters them out of the `--all` work list — silently; nothing logs a
   skipped dir, only an entirely-empty bucket logs.
2. Naming a scene explicitly also fails: `run_pipeline_remote.py` `main()` rejects ids that
   fail the same regex with `parser.error(... expected YYYY_MM_DD-PARENT-VIDEO ...)`.
3. `list_processed_scenes()` shares the regex, so even a pushed audiomoth scene would be
   invisible to leaf-stage re-runs.
4. `environments-processed` holds zero audiomoth entries.

The data itself is processable: all nine dirs hold exactly one video (seven `.mp4`, two
`.mov` — both in `_VIDEO_EXTS`) plus `<video>_metadata.json` and `<video>_telemetry.parquet`,
which discovery already ignores.

Per its own comment, the regex exists so a scene id joined onto a local output path cannot
escape it (`../x`). The date shape is incidental strictness, not the safety property.

## Design

### `collab_splats/remote/sources.py`

- Replace the pattern with the safety property it actually needs:

  ```python
  SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
  ```

  One path segment: `/` cannot appear; a leading `.` is rejected, which kills `..`, `.`,
  and hidden dirs; empty string cannot match. Interior dots (`a..b`) are safe — they are
  ordinary filename characters, not traversal. Legacy `YYYY_MM_DD-...` ids and the
  audiomoth names both match. Rewrite the comment: the regex enforces path safety for the
  output-path join; the `YYYY_MM_DD-PARENT-VIDEO` shape is a naming convention, not the
  contract.
- `list_scenes()`: when a curated dir is present but fails the regex, log it
  (`logger.warning("skipping non-scene dir in %s: %s", ...)`) so the next naming-convention
  drift is visible instead of silent. Same in `list_processed_scenes()` is unnecessary —
  its comment ("stray upload") stays accurate, but it inherits the widened constant
  automatically, keeping discovery and re-run views consistent.

### `docs/examples/run_pipeline_remote.py`

- The explicit-id validation keeps using the shared `SCENE_ID_RE` (unchanged code), but the
  `parser.error` text drops `expected YYYY_MM_DD-PARENT-VIDEO` for
  `not a safe scene id (one path segment, no leading dot)`.

### `configs/README.md`

- Scene-id wording: the curated dir name IS the scene id; the `YYYY_MM_DD-PARENT-VIDEO`
  shape is the common convention, any flat path-safe name works.

## Testing

`tests/remote/` (flat functions):

- `SCENE_ID_RE` accepts: a legacy id, a representative audiomoth id (and ideally all nine
  real names as a parametrized list).
- Rejects: `../x`, `..`, `.hidden`, `a/b`, `""`, a name starting with `-` (keeps ids safe
  as argv tokens too — current pattern's first-char class already excludes it).
- `list_scenes()` skip-logging: a mocked listing containing one matching dir and one
  non-matching dir yields only the match and emits the warning.

## Non-goals

- No processing-pipeline changes — the audiomoth videos are ordinary `.mp4`/`.mov`.
- No bucket mutation, no renames, no `index.csv` handling.
- `*_metadata.json` / `*_telemetry.parquet` stay ignored, as today.

## Verification after implementation

`run_pipeline_remote.py --all` discovery lists all nine audiomoth scenes (dry check:
`SceneSource().list_scenes()` contains them). Actual reconstruction runs stay a separate,
human-watched step (GCS runs are compute + billed).
