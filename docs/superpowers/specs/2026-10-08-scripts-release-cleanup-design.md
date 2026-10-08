# scripts/ release cleanup — design

Rules: [decision 017](../decisions/017-release-cleanup-rules.md). Reference run:
[preproc](2026-09-24-preproc-release-cleanup-design.md).

Scope: the gdrive curation files ported onto `clean/final` in `3433b62e`
(`scripts/preprocess_gdrive_videos.py`, `scripts/push_curated.sh`,
`tests/scripts/test_preprocess_gdrive_videos.py`), plus `scripts/update_kernelspecs.py`.

## Round 1 — prose only

`preprocess_gdrive_videos.py`
- module docstring: summary line, bullets, usage block; no scene ids
- every def: summary on the line after `"""`, ≤ 6 bullets, `Args:`/`Returns:` for public defs
- block comments: one plain line; no measurements, scene ids or history
- drop the `_GPMF_STREAMS` "VERIFY AGAINST REAL FOOTAGE" note and the `GPSTrack` history
- "colour" → "color" in comments; the sidecar `intrinsics.reason` string is stored data, unchanged

`push_curated.sh`
- `push_outputs` lives at `collab_splats/remote.py` (`SceneSource.push_outputs`), not `dashboard/sources.py`
- multi-line comment runs → one line; the header block is the `--help` text and stays

`test_preprocess_gdrive_videos.py`
- comments and docstrings trimmed to one line; history ("before this…", "the branch this replaces") dropped

Proof: AST equal after deleting every docstring statement on both sides, plus one sanity mutation.

## Round 2 — code, one commit each

| change | detail |
|---|---|
| delete `update_kernelspecs.py` | no callers; hardcoded stale worktree roots and the retired nerfstudio env |
| annotations | every def in `preprocess_gdrive_videos.py` fully annotated |
| constants → kwargs | `AUDIO_RATE` → `rate=8000`; `DEFAULT_ALIGN_MIN_R` → `min_r=0.95`; `ALIGN_TOLERANCE_S` → `tolerance_s=1.0` on `solve_offset` |
| `logging.basicConfig` | moved into `main`; import no longer configures logging |
| inline `has_gps_track` | one caller, `gps_payload` |
| `needs_copy` | drop `or {}`; a missing sidecar is an explicit `None` check |
| GPMF timing | `SampleTime` / `SampleDuration` are required; a chunk missing either raises `KeyError` instead of defaulting to 0.0 / 1.0 |
| `inject` | an ffmpeg or exiftool non-zero exit raises `RuntimeError`; `main` already records the clip as failed and a re-run retries it (no sidecar was written) |
| contract | `"scripts"` added to `TOP_LEVEL` and `RELEASED` in `tests/test_docstring_contract.py` |

Unchanged on purpose: `sanitize` / `flat_dir_name` behavior (flat names are pinned in the bucket);
`main`'s broad `except` (logs and records the failure); `solve_offset`'s `ok=False` result for an
impossible alignment.

## Testing

Gate: `tests/scripts tests/test_import_style.py tests/test_docstring_contract.py`, baseline
1387 passed / 2 skipped / 80 xpassed. Tests updated where a constant or a fallback is removed.

## Out of scope

Real-footage rerun of the curation pipeline; pushing to the bucket.
