# preprocess_gdrive_videos.py — design

Date: 2026-08-11
Status: approved, ready for an implementation plan

## Problem

DaVinci Resolve destroys all capture metadata on export. Verified on `GH010234`: the
camera original carries `tmcd + gpmd + fdsc` tracks, container GPS, and the full GoPro
UserData block; the Resolve export carries `tmcd` alone plus `encoder=Blackmagic Design
DaVinci Resolve`. Resolve decodes to frames, applies the timeline, and re-encodes through
its own muxer — GPMF is not a format it parses or emits, so data tracks have no path
through the NLE. Everything downstream of `environments-curated/` therefore sees video
with no GPS, no capture time, no camera identity, and no IMU.

The IMU is the valuable part for reconstruction. The GoPro `gpmd` track carries ACCL,
GYRO, GRAV, CORI and IORI at ~200 Hz plus GPS9 at ~18 Hz, and it is discarded at the exact
point the footage enters the splat pipeline.

A second problem surfaced while reading the existing code. `scripts/flatten_dataset.py`
ran once (2026-07-29/30) and was never re-run, so `environments-curated/` is stale at 18
folders, and the script has two structural bugs (below). Fixing curation and adding
metadata preservation are the same edit, so this replaces `flatten_dataset.py`.

## Measured baseline

67 videos in `gdrive-src`: 48 camera originals, 19 Resolve exports.

| Camera | Originals | Timed track | IMU | GPS |
|---|---|---|---|---|
| GoPro Max | 33 | `gpmd` (GPMF) | ACCL, GYRO, GRAV, CORI, IORI | GPS9, 29/33 locked |
| Sony ILCE-6700 | 3 | `rtmd` (KLV) | Accelerometer + PitchRollYaw | none |
| Google Pixel | 9 | `mett` x2 | none | container `location` |
| iPhone 16 Pro / iPad Pro M5 | 3 | `mebx` x4-6 | none | container `location` + altitude |

In scope (pairs only): **19 clips, 16 with IMU (84%)**. The three without are `IMG_4085`
(iPhone) and the two Pixel clips `PXL_20260629_225753909.TS` and
`PXL_20260630_002106958.TS`. No Sony clip has been edited, so the only other IMU-bearing
camera contributes nothing today.

The phone gap is not recoverable at capture time. Pixel writes `application/meta`
(per-frame capture parameters — 943 packets for 943 frames) and
`application/microvideo-image-meta`, never the Android-standard `camm` IMU track. iOS keeps
CoreMotion out of the file: `CoreMotionVersion` is a version stamp, and `mebx` holds face
detection, video orientation and LivePhoto data.

Every export is trimmed and conformed to 59.94 fps — `GH010232` 59.7 s to 43.4 s,
`GH010238` 93.8 s to 84.1 s. Telemetry therefore cannot be copied across verbatim; it must
be placed against the edit or it becomes confidently wrong.

## Goal

One script, `collab-splats/scripts/preprocess_gdrive_videos.py`, that flattens the nested
Google Drive capture tree into the curated layout and carries capture metadata across on
the way. It supersedes and deletes `scripts/flatten_dataset.py`.

Single entry point because the current split is what caused the staleness, the stages share
the non-trivial edit-to-source pairing derivation, and injection must complete before rclone
or bytes get uploaded and then immediately re-uploaded. In one process that ordering is
structural rather than remembered.

`scripts/push_curated.sh` stays as-is and is invoked via `subprocess`. Its rclone flags
deliberately mirror `SessionSource.push_outputs` in `collab_splats/dashboard/sources.py` —
its own header comment says so — and reimplementing them in Python creates two places to
drift.

Layout follows the repo's existing script convention: a single file with `########` section
dividers, imports at top, `logging` rather than `print`, one-line docstrings. A module split
under `collab_splats/preproc/` was considered and rejected as over-complication for a
script whose job is to flatten a directory tree.

## Pairing rule

A video is an **edit** if and only if `<parent>/src/<stem>.*` exists. The match is on stem
only, case-insensitive, across any video extension. The real tree needs all three forms:
`GH010234.mp4` matches `src/GH010234.MP4` (case differs) and `IMG_4085.mp4` matches
`src/IMG_4085.MOV` (extension differs).

Only pairs are processed. An original with no edit is skipped and logged with its reason,
and enters scope automatically once it is exported. Today that skips 29 originals: five
orphans inside `src/` folders (`GH010224`-`GH010227`, `GH010233`) and 24 in date folders
with no `src/` sibling.

This replaces two bugs in `flatten_dataset.py`:

1. `plan_copies` only walked `<date>/<parent>/`, so videos sitting directly in a date
   folder — `2026-06-03/GH010218-220.MP4` — were silently skipped. Three fully-telemetered
   GoPro clips were invisible to the pipeline.
2. It treated "video directly inside parent" as *processed*. That holds only when a `src/`
   sibling exists. For the seven folders without one, those videos **are** the camera
   originals — all three Sony and all nine Pixel files — and would have been flattened as
   if they were Resolve exports.

Run against today's tree, the old rule emits 40 folders where 18 exist, and
`2026_07_08-GoproSplat-GH010223` is missing entirely. The pairing rule yields 19: the
existing 18 plus that one.

## Output layout

`sanitize` and `flat_dir_name` port over verbatim from `flatten_dataset.py`, so existing
folder names are unchanged. Nothing on disk is renamed — renaming a curated mp4 would break
`collab_splats/dashboard/app.py`, which matches curated video stems against processed-output
stems to decide what to show.

```
environments-curated/2026_07_22-splats-GH010234/
    GH010234.mp4                  # edit + injected static tags + retimed gpmd track
    GH010234_metadata.json        # static tags, provenance, alignment scores, exiftool dump
    GH010234_telemetry.parquet    # ~200 Hz IMU and GPS columns; only when IMU is present
```

The date is underscored, the parent folder sanitized, the video stem carried over verbatim
so dots survive (`PXL_20260630_002106958.TS`). The stem is last, so any hyphens it contains
stay unambiguous under `split("-", 2)`. Two folders sanitizing to the same flat name raise
rather than silently overwriting.

The Parquet table carries `source_time`, `edit_time`, `in_edit`, `accl_*`, `gyro_*`,
`grav_*`, `cori_*`, `iori_*` and `gps_*`. The streams do not share a time axis: each GPMF
chunk carries its own sample count, so the per-sample step `SampleDuration / count` differs
per stream and the axes are nearly disjoint. `source_time` is therefore the *union* of every
stream's sample times and each column is sparse — a typical row holds one stream's values and
nulls in every other column, and the row count runs well above any single stream's rate. The
nulls are interleaved rather than in runs, so they do not RLE away; the file is still one
table because splitting it would multiply sidecars without removing the sparsity. Nothing is
resampled, so a null means "no sample at this instant", never missing data. Each column's fill
ratio is logged at INFO on every run. `edit_time` is null when alignment failed or the sample
falls outside the cut.

## Stages

Default run: `pair` then `copy`, `align`, `extract`, `inject`. Push requires `--push`,
because it is a ~7 GB write to a shared bucket while everything else is local and
reversible.

**copy** — `copy_video` verbatim from the old script: write a `.partial` sidecar, then
`os.replace`, so an interrupted run cannot leave a truncated file that looks complete.

**align** — decode both audio tracks (`ffmpeg -vn -ac 1 -ar 8000 -f f32le -`) and locate the
edit within the original by FFT cross-correlation using `scipy.signal`, already a declared
dependency. Audio is untouched by the fps conform and by color correction, so this is
sample-accurate. Below threshold, the clip gets source-time telemetry only, no gpmd
injection, and a flag in the run summary. An offset the correlation cannot support is never
emitted.

### Acceptance criterion

A single gate: **normalized cross-correlation `r >= 0.95` at the peak.** Pearson over the
overlap window, so the value is scale-free and reads directly as variance explained.
Default overridable with `--align-min-r`.

The threshold is high because the edit's audio is a literal subsegment of the source —
Resolve cuts and colour-corrects but never alters audio content — so at the true offset `r`
is expected to sit near 1.0, degraded only by the AAC re-encode. At a wrong offset over `N`
samples `r` is approximately `1/sqrt(N)`, about 0.0006 for a minute at 8 kHz. The gap
between the two cases spans three orders of magnitude, so the exact placement inside it is
not delicate.

The known risk of 0.95 is at the top of the band: if the re-encode degrades a genuine match
to 0.90-0.94, that clip is rejected. The consequence is a fallback to source-time telemetry
with no injection, reported in the run summary rather than swallowed, and `--align-min-r`
lowers the bar if the real footage lands there.

Structural feasibility needs no separate gate, but the search range does need slack at the
end. **Corrected after implementation:** this section originally claimed the lag search was
restricted to `0 <= lag <= len(source) - len(edit)`, "so an offset that would run the edit
past the end of the source cannot be returned at all" — describing a bug as though it were a
safety property.

Resolve re-encodes audio to AAC, and the encoder's priming padding leaves the decoded edit
slightly longer than the region it was cut from. The true lag can therefore sit just *above*
`len(source) - len(edit)` and be unreachable. Measured on the real pair `2026-06-03/GH010218`:
the true lag was 272 samples (34 ms) past that ceiling, so the clip scored `r = 0.09` and was
rejected as unalignable — while windowed correlation showed a clean contiguous head trim
matching at `r = 0.993-0.998` across its whole length with zero drift.

The fix appends `ALIGN_TOLERANCE_S` (1.0 s, roughly 25x the observed overrun) of silence to
the source before searching, so such a lag stays reachable. Lags landing inside the padding
have no energy, so the existing `denominator > 0` guard scores them 0 and they cannot win the
argmax; the widened range admits no false matches. Verified: the same pair now returns
`offset 6.9902 s, r 0.9967`, identical at 0.25 s, 1 s and 5 s of padding, while two genuinely
mismatched real pairs score 0.027 and 0.014. An edit longer than source plus tolerance is
still rejected outright.

This does not cover an edit whose audio begins *before* the source's first sample, which
would need negative lags. Such a clip scores low and lands in the rejected list — it fails
visibly rather than silently.

**extract** — `exiftool -ee -api LargeFileSupport=1 -json -n -G3` on the original. This uses
the tool already installed and adds no gpmf-parser or node dependency. GPMF comes back as
one value-set per 1 Hz chunk with all samples concatenated — a single `Accelerometer` chunk
is a flat run of triplets — alongside `SampleTime`, `SampleDuration` and the GPMF
`TimeStamp`, which is enough to place every sample in source time.

**inject** — the order is load-bearing. ffmpeg runs first because it rewrites the container;
exiftool runs second to write tags into the final container. Reversed, ffmpeg drops the tags.

- gpmd track: resolve the stream index via ffprobe, then
  `ffmpeg -i <curated> -ss <offset> -t <dur> -i <original> -map 0 -map 1:<gpmd> -c copy -copy_unknown`
- static tags: `exiftool -overwrite_original -api QuickTimeUTC` for capture date, GPS,
  make, model, serial and firmware, plus a provenance blob so the mp4 alone is
  self-describing.

**push** — `subprocess.run(["./scripts/push_curated.sh"])`, forwarding `--dry-run`.

### Why both an injected track and a sidecar

This is not redundancy. The injected track makes the mp4 readable by `gpmf-parser`,
`gopro-telemetry` and exiftool directly, and it travels to GCS with the file. But GPMF
payloads are 1 Hz chunks, so a trimmed gpmd track can only be cut on roughly one-second
boundaries. The sidecar holds the exact sub-frame offset and full-rate samples — precision
the injected track structurally cannot carry. The JSON records the actual first-chunk time
so downstream can see the residual.

The three phone clips get static tags only. No IMU is synthesized.

## CSV index

Derived state, never authored. `build_index(curated_root)` scans `environments-curated/`,
reads each `*_metadata.json`, and writes `environments-curated/index.csv`. There is no merge
logic and therefore no way for the CSV to drift from the sidecars: a partial run such as
`--only GH010234` still yields a complete index, because every untouched clip still has its
JSON on disk. This also supports `--index-only`, which regenerates the CSV without touching
a byte of video.

Two columns, deliberately. The file exists to be pasted into Notion.

```csv
unique_id,gps
2026_07_22-splats-GH010234,"42 deg 21' 11.52"" N, 71 deg 3' 57.24"" W"
2026_06_29-Phone_pics_and_splat_videos-PXL_20260629_225753909.TS,
```

`unique_id` is the flat directory name. `gps` is formatted as degrees, minutes and decimal
seconds with a hemisphere letter. A clip with no fix gets a blank cell: never `0,0`, never a
sentinel string. Python's `csv` module quotes the DMS field because it contains commas, and
doubles the inch mark.

The GPS value resolves in this order, so that every camera in the tree lands somewhere
defined:

1. **First locked fix inside the trimmed edit**, when the clip has a GPS track and alignment
   succeeded. This is where the clip starts.
2. **First locked fix in the source**, when the clip has a GPS track but the previous step
   found nothing — either because alignment failed, so the edit window is unknown, or
   because the camera's GPS lock ended before the cut began. The JSON sets
   `gps_source_anchored: true` to record that the value is approximate rather than taken
   from inside the cut.
3. **Container `location`**, for the Pixel and iPhone clips, which carry a single point and
   no track.
4. **Blank**, when there is no fix at all — the two Pixel `.TS` clips, and any GoPro whose
   GPS9 never locked (4 of 33 originals).

Case 2 covering the lock-ended-early situation, not only the alignment-failed one, was a
decision taken during implementation: the alternative left the cell blank for a clip whose
location is in fact known, and the flag keeps the approximate value honest.

Everything else — decimal latitude and longitude, capture time, camera, IMU presence,
duration, alignment confidence — stays in `_metadata.json`, where a script can read it.

DMS formatting is a pure function: `-71.0659` becomes `71 deg 3' 57.24" W`. The sign becomes
the hemisphere letter and never a leading minus. Seconds carry two decimals, matching the
exiftool output convention already in use.

## Idempotency

Injection makes the curated mp4 **larger than the source it was copied from**, so
`copy_video`'s existing check — destination size equals source size — fails on every
subsequent run. Left alone it would re-copy (clobbering the injection), re-inject, and
re-upload ~7 GB every time.

The fix: compare against `source.size_bytes` and `source.mtime` recorded in
`_metadata.json`, not against the destination's own size.

That single gate is sufficient, and no separate "already injected" check is needed: a run
either returns early without injecting, or re-copies the pristine edit before injecting. So
injection always operates on freshly copied bytes and can never be applied twice to the same
file. A `-Software` provenance tag is still written, as a marker for anyone inspecting a
curated mp4 by hand.

## Intrinsics validity

The edits are cuts plus color correction, with no geometric transform, so field of view,
lens projection (`GPRO`), diagonal FOV and EIS state remain true of the exported pixels. The
JSON marks these `valid_for_edit: true` together with that reason, making it a recorded
assumption rather than a silent one. If a clip is ever reframed or stabilized, that flag is
what has to flip.

## Error handling

- Missing `exiftool`, `ffmpeg` or `ffprobe`: fail at startup with the missing binary named,
  before any copying begins.
- Alignment below the confidence threshold: not fatal. Write source-time telemetry, skip
  injection, flag the clip in the summary, continue to the next pair.
- Injection failure on one clip: log and continue; the curated mp4 keeps its pre-injection
  bytes because ffmpeg writes to a temporary file that is only moved into place on success.
- Flat-name collision: fatal, raised before any work is done, since it would silently
  overwrite one capture with another.

## Testing

Test-driven: tests are written first and watched failing before implementation.
`tests/scripts/test_flatten_dataset.py` is renamed to `test_preprocess_gdrive_videos.py`.
The module is still loaded by file path, since `scripts/` is not an importable package.

Ported unchanged: all eight `sanitize` cases, all four `flat_dir_name` cases, all three
`copy_video` cases.

**One existing test must change.** `test_plan_copies_skips_src_only_folders` asserts
`GH010229` is included, but `GH010229` has no `src/` counterpart, so the pairing rule now
skips it. That is the rule change made visible, not a regression.

New tests, all pure-function or `tmp_path`-based, with no real video decoded:

- pairing across case (`.mp4` to `.MP4`), across extension (`.mp4` to `.MOV`), and for
  videos sitting directly in a date folder — the case the old script missed
- DMS formatting: both hemispheres, zero, sign to hemisphere letter, seconds rounding
- CSV: two columns, blank-GPS row, inch-mark quoting round-tripping through `csv.reader`
- alignment: synthesize a signal, offset it by a known amount, assert the recovered offset
  is within one audio frame and that `r` is near 1.0; assert uncorrelated noise returns no
  offset because `r` falls below 0.95; assert an edit longer than its source yields an empty
  lag range rather than a bogus offset
- idempotency: a second run over a destination that is *larger* than its source copies zero
  bytes

## Files

| File | Change |
|---|---|
| `scripts/preprocess_gdrive_videos.py` | new — absorbs `flatten_dataset.py` |
| `scripts/flatten_dataset.py` | delete (superseded) |
| `tests/scripts/test_flatten_dataset.py` | rename to `test_preprocess_gdrive_videos.py` |
| `scripts/push_curated.sh` | unchanged, invoked via subprocess |
| `pyproject.toml` | declare `pyarrow` — currently in `uv.lock` only transitively via gradio |
| `README.md` | rewrite the curation block, lines ~94-103 |

Sony `rtmd` extraction is deferred. The parse would ride the same exiftool call, but no Sony
clip has an edit, so there is nothing to test it against. The extractor dispatches on track
type, so adding it later is a function rather than a refactor.

## Verification

```bash
# 1. Plan only — expect 19 pairs, 16 with IMU, 29 originals skipped with reasons
python scripts/preprocess_gdrive_videos.py --dry-run

# 2. One clip end to end
python scripts/preprocess_gdrive_videos.py --only GH010234

# 3. gpmd survived injection (expect tmcd AND gpmd; before, tmcd alone)
ffprobe -v error -show_entries stream=codec_tag_string -of csv=p=0 \
  ../environments-curated/2026_07_22-splats-GH010234/GH010234.mp4

# 4. Static tags landed — GPS and a 2026-07-22 capture date, not the Resolve export date
exiftool -G1 -GPSCoordinates -CreateDate -Model -SerialNumber \
  ../environments-curated/2026_07_22-splats-GH010234/GH010234.mp4

# 5. Telemetry sane. The streams do NOT share a time axis: each GPMF chunk carries its own
#    sample count, so SampleDuration/count differs per stream and the per-stream axes are
#    nearly disjoint. The table is their union, so expect MORE rows than any single stream's
#    rate implies and expect every column to be mostly null. That is the design, not a bug —
#    a null means "this stream has no sample at this instant". What to check instead:
#      - edit_time starts near 0 and in_edit is True across the middle of the clip
#      - each accl_/gyro_/gps_ column has a non-zero count (an all-null column means the
#        exiftool tag name is wrong; the run also logs the fill ratio of every column)
#      - source_time is sorted and spans the source duration
python -c "
import pandas as pd
d = pd.read_parquet('../environments-curated/2026_07_22-splats-GH010234/GH010234_telemetry.parquet')
print(d.shape); print(d[['source_time','edit_time','in_edit']].describe())
print('non-null per column:'); print(d.count())
assert d['source_time'].is_monotonic_increasing
assert (d.filter(like='accl_').count() > 0).all(), 'accl tag name is wrong'
assert (d.filter(like='gps_').count() > 0).all(), 'gps tag names are wrong'"

# 6. Idempotency — the trap above. A second run must copy 0 bytes and re-inject nothing.
python scripts/preprocess_gdrive_videos.py --only GH010234

# 7. Index regenerates standalone and stays complete
python scripts/preprocess_gdrive_videos.py --index-only
head -3 ../environments-curated/index.csv

# 8. Full run, then push
python scripts/preprocess_gdrive_videos.py
./scripts/push_curated.sh --dry-run
python scripts/preprocess_gdrive_videos.py --push

# 9. Tests
python -m pytest tests/scripts/test_preprocess_gdrive_videos.py
```

## Consequences to expect

- `2026_07_08-GoproSplat-GH010223` appears for the first time; the curated tree goes from 18
  folders to 19.
- Every curated mp4 is rewritten, so the next `push_curated.sh` re-uploads roughly 7 GB
  once. rclone is stable again afterwards.

## Known follow-ups after implementation

Recorded at the end of the build so they are not lost with the scratch workspace. None
block the branch; the first two need an environment this laptop does not have.

**`uv.lock` is stale — must be closed before anyone runs `uv sync --frozen` or `--locked`.**
`pyarrow` was added to `pyproject.toml` on this branch but the lock was never regenerated,
so the `collab-splats` block still lists `scipy` and not `pyarrow`. `uv lock` cannot run on
a machine without `/workspace/collab-data` mounted, because `pyproject.toml` declares
`collab_data @ file:///workspace/collab-data` as a direct-URL requirement. One `uv lock`
inside the container closes it. Until then a freshly locked environment has no `pyarrow`
and the script will not import.

**The two integration tests that guard the injection path skip silently in CI.**
`test_inject_really_grafts_a_gpmd_track_onto_the_curated_video` and its exiftool
counterpart are the only tests that catch the two Critical defects the whole-branch review
found — an unusable ffmpeg temp filename, and exiftool exiting 0 while writing nothing.
Both carry `skipif(not _HAS_MEDIA_TOOLS)`, and `.github/workflows/test.yml` never installs
ffmpeg or exiftool, so both skip rather than fail. Adding those two binaries to the CI
image is what makes the guard real.

**~~The GPMF tag names are still unverified against real footage.~~ RESOLVED, and the run
found three further defects** — every one invisible to the test suite because the fixtures
were invented from an assumed exiftool output shape rather than sampled from a real dump.
That is the root cause worth carrying forward: a fixture that encodes a guess validates the
guess.

1. **`exiftool -json -G3` returns one object for the whole file**, not one per document. The
   document is encoded in the key prefix (`Main:`, `Doc1:`, `Doc1-1:`). The original
   `_ungrouped` stripped that prefix, collapsing all 380 accelerometer chunks and 6910 GPS
   samples onto one key, last-write-wins. Telemetry would have carried 1 chunk of 381, and
   `first_fix` returned the clip's *last* fix. Fixed by parsing the prefixes into ordered
   documents: `Main` for container tags, `DocN` per 1 Hz chunk, `DocN-M` for the ~18 Hz GPS
   sub-samples inside chunk N.
2. **The IMU tags come back as `(Binary data N bytes, use -b option to extract)`** and
   crashed the parser. `exiftool` needs `-b`. Cost measured: 4.9 s and a 15 MB dump for a
   2.15 GB original, with no junk blobs.
3. **The remux could never write.** `-map 0` pulled in the Resolve export's `tmcd` stream,
   whose codec the mp4 muxer cannot tag, so ffmpeg aborted before the gpmd track landed —
   silently costing the pipeline its entire purpose. Fixed with `-map -0:d`. Timecode is not
   lost: the muxer regenerates a `tmcd` track from metadata.

Verified end to end on `2026-06-03/GH010218`: 104,416 Parquet rows (76,783 accelerometer and
gyroscope at 202 Hz, 6,909 GPS), curated streams `avc1 mp4a gpmd tmcd`, injected track
carrying 374 chunks against the source's 381, `gpmd_injected: true`, and the mp4 reading back
`GoPro Max` with `42 deg 21' 10.08" N, 71 deg 3' 55.80" W`.

**Three cosmetic residuals, left deliberately.** `gpmd_first_chunk_s` and `gpmd_residual_s`
are emitted as `0.0` even when alignment failed and no track was injected — recoverable
from the adjacent `gpmd_injected: false`, but misleading read alone. `has_imu` is derived
from "any telemetry stream produced rows", so a GPMF dump carrying GPS and no accelerometer
would set it true; unreachable on the footage in hand. And the full repository test suite
has never run on this branch — it needs CUDA and a container-only interpreter, so only
`tests/scripts/test_preprocess_gdrive_videos.py` (110 tests) has been exercised.

## Verification against the real tree (2026-08-12)

Run after the three real-footage defects above were fixed. **Several figures earlier in this
spec are stale and are superseded here** — the capture tree grew between design and
verification, gaining `2026-03-27`, `2026-03-30` and `2026-04-03` entirely, and gaining
exports for `GH010224`-`GH010227`, which the pairing section lists as unedited orphans.

| Quantity | Spec said | Actually measured |
|---|---|---|
| Pairs in scope | 19 | **34** |
| Transfer size | ~7 GB | **12.7 GB** |
| Originals skipped | 29 | **0** |
| Curated folders | 18 to 19 | 34 |

### Alignment calibration

Every one of the 34 pairs aligns and is accepted. **Minimum r among true pairs: 0.9882**
(`PXL_20260630_002106958.TS`); 22 of 34 score 0.999 or better. Against this, two
deliberately mismatched real pairs score 0.027 and 0.014. The gate at `r >= 0.95` therefore
sits in an empty band with roughly 15x margin on both sides, and needs no adjustment. This
supersedes the earlier note that the threshold was an unvalidated prediction.

Six pairs have an edit *longer* than its source — `GH010219`, `GH010224`, `GH010227`,
`GH010230`, `GH010235`, `GH010236` — which the pre-`ALIGN_TOLERANCE_S` code rejected
outright as "alignment impossible". The tolerance fix was not a one-clip patch.

### Ten pairs are copies, not edits

Ten parent-level files are byte-identical to their `src/` counterpart — matching size and
matching head hash — so the parent file *is* the camera original, copied up a level rather
than exported from Resolve. All ten are phone clips (Pixel `.TS`, `IMG_*`).

They satisfy the pairing rule and process harmlessly: offset 0.0, `r` 1.0, static tags
rewritten, no telemetry to recover. But the premise of this pipeline does not apply to them
— Resolve never touched them, so nothing was destroyed. They are recorded here rather than
special-cased, because the stated rule is "process files that have both" and they do.

### Confirmed working end to end

- GoPro clip: 3,266 telemetry rows, `gpmd` injected, streams `avc1 mp4a gpmd tmcd`
- Phone clip with no IMU: processes cleanly, no telemetry sidecar, no injection, no crash
- Idempotency: a second run skips and leaves the curated mp4 byte-identical
- `index.csv`: two columns, correct DMS, and a phone clip picks up its container GPS
- 125 tests pass; ruff reports only the pre-existing `EXE001`

### Exhaustive pairing check

Every edit was correlated against every source — all 34 x 34 = 1,156 combinations — using a
20 s probe from each edit's midpoint. A probe is the wrong tool for deriving a precise offset
but the right one for screening content identity, and it makes the sweep tractable.

**All 34 edits match their assigned source best.** Correct pairs score 0.9882 to 1.0000; the
best incorrect match anywhere in the matrix is 0.1344; the smallest winning margin is 0.8656.
The `r >= 0.95` gate therefore sits in a band that is empty from 0.1344 to 0.9882, and no
pairing in the tree is wrong or even close to ambiguous.

This supersedes the three hand-picked negative controls quoted above as the basis for the
threshold: those sampled 3 of 1,122 possible wrong pairings, and this covers all of them.
