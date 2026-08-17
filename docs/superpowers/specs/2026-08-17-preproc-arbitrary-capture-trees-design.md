# preprocess_gdrive_videos.py — arbitrary capture trees

Date: 2026-08-17
Status: approved, ready for an implementation plan

Extends [2026-08-11-preprocess-gdrive-videos-design.md](2026-08-11-preprocess-gdrive-videos-design.md).
That spec's pairing rule, alignment, extraction, injection, telemetry and index sections all
stand unchanged; only the tree walk and the naming function are replaced.

## Problem

`gdrive-src/audiomoth-only-deployments/` is invisible to the pipeline. Its nine editable
clips have never been curated, and no log line reports them as skipped for a reason anyone
would act on.

The new drop nests two levels deeper than anything the script has seen, and its top level is
a deployment window rather than a capture date:

```
gdrive-src/audiomoth-only-deployments/<YYYYMMDD-YYYYMMDD>/<site-slug>/splat_videos/GH0102xx.mp4
                                                                      splat_videos/src/GH0102XX.MP4
```

against the shape the script was written for:

```
gdrive-src/<YYYY-MM-DD>/[<parent>/]GH0102xx.mp4
                         [<parent>/]src/GH0102XX.MP4
```

Two functions encode the old shape and nothing else does.

`plan_pairs` (`scripts/preprocess_gdrive_videos.py:149`) walks exactly two levels and gates
the first on `_DATE_RE.fullmatch`. `audiomoth-only-deployments` is not `YYYY-MM-DD`, so the
entire subtree is discarded at the top with a log line about date formatting — technically
accurate and useless.

`flat_dir_name` (`:109`) takes `(date, parent, video)`. There is no slot for a deployment
window, a site and a `splat_videos` container.

Everything else is already path-agnostic. `find_source` matches on stem alone,
case-insensitive, across any video extension, so the audiomoth tree's `.mov` edits against
`.MP4` sources need no work. Alignment, GPMF extraction, injection, the Parquet sidecar and
the CSV index all take a `Pair` and never inspect its path.

The deeper problem is that a fixed-depth walk makes every new capture layout a code change
before its footage is visible. That is the same failure mode the 2026-08-11 spec was written
to close when it replaced `flatten_dataset.py`: work that has to be remembered eventually
is not.

## Measured tree (2026-08-17)

Seven sites across two deployment windows, 38 files.

| Site | Edits | Sources |
|---|---|---|
| `20260810-20260831/boston-frontageroad-tracks` | `GH010250.mp4`, `GH010251.mp4` | both |
| `20260810-20260831/boston-ringerpark-east` | `GH010248.mp4` | yes |
| `20260810-20260831/boston-ringerpark-west` | `GH010247.mov`, `GH010249.mov` | both |
| `20260817-20260824/boston-alley443-trees` | `GH010258.mp4` | yes |
| `20260817-20260824/boston-charlesgateeast-riverbank` | `GH010259.mp4` | yes |
| `20260817-20260824/boston-copps-hill-shrubbery` | none | `GH010252.MP4` only |
| `20260817-20260824/boston-publicgarden-maintenance` | `GH010255.mp4`, `GH010257.mp4` | both |

Nine pairs, 7.3 GB of edits. `GH010252` is an unedited original and stays out of scope until
someone exports it, exactly as `GH010233` does today.

The two `boston-ringerpark-west` clips are the first non-`.mp4` edits in the tree. They are
`.mov` carrying `avc1` video and `lpcm` audio rather than `mp4a`.

## Approach

Replace the fixed-depth walk with a recursive descent, and derive the flat name mechanically
from the path the video was found at. Depth and date-shape stop being concepts the script
holds.

Two alternatives were considered and rejected.

A **layout profile table** — one entry per top-level drop, declaring its depth and naming
rule — was rejected because it is configuration for something the tree already states, and
because every future drop would need a code edit before its footage became visible. That is
the failure this change exists to remove.

**Anchoring on `src/`** — `rglob("src")`, treating each hit's parent as a capture folder —
produces identical output in one pass instead of a full descent, and leans on the fact that
the pairing rule already requires a `src/` sibling. It was rejected on readability: the tree
is 38 files against a few hundred in the older drops, so the saving is unmeasurable, and an
inverted walk is harder to follow than a plain one.

## Walk

`plan_pairs` descends recursively from `source_root`. At every directory reached it runs the
existing `find_videos` (non-recursive, extension-filtered) and `find_source`, both unchanged.

Two kinds of directory are pruned and never descended into:

- any directory named exactly `src`, so a camera original can never be mistaken for an edit at
  any depth
- any directory whose name begins with `.`

The `src` match is case-sensitive deliberately, because `find_source` resolves
`video.parent / "src"` case-sensitively and the two must agree. Both capture trees use
lowercase throughout. A directory named `SRC` would be descended into rather than pruned, but
its videos would then look for a `SRC/src/` sibling, find none, and be reported as unedited
originals — the same outcome `find_source` produces today.

`_DATE_RE` and its `skip <name>: not a YYYY-MM-DD directory` log line are deleted. Nothing
replaces them. A directory with no `src/` anywhere beneath it yields no pairs and costs a
stat, so junk at the root is self-limiting. This is what makes the next drop work without an
edit: an audiomoth `.WAV` folder appearing beside `splat_videos` later contains no videos and
is passed over in silence.

The existing behaviour for a video with no `src/` counterpart is unchanged — logged as an
unedited original, skipped, and picked up automatically once it is exported.

### AppleDouble filter

`find_videos` filters on suffix alone, so `2024-07-09/SplatsSD/._C0104.MP4` currently passes
as a video. These are macOS AppleDouble files: on filesystems with no resource fork — the
FAT32 and exFAT of camera SD cards — macOS splits a file's Finder metadata and extended
attributes into a companion `._<name>` beside it. A few KB of flags, named `.MP4` only
because it mirrors the file it describes.

Six exist today and all are harmless, because `SplatsSD/` has no `src/` folder so nothing
pairs. The failure needs both halves: a folder that does have `src/`, on a volume that
produced AppleDouble files. Then `._GH010247.MP4` and `src/._GH010247.MP4` both exist, their
stems match, and two metadata blobs are handed to ffmpeg and exiftool as footage.

`find_videos` gains a leading-dot filename filter, which closes this and `.DS_Store` in the
same condition.

## Naming

`flat_dir_name(date, parent, video)` becomes `flat_dir_name(video, source_root)`:

```python
parts = [sanitize(p) for p in video.parent.relative_to(source_root).parts]
return "-".join(parts + [video.stem])
```

`sanitize` is unchanged and applies per path component. The stem is carried over verbatim and
placed last, so `PXL_20260630_002106958.TS` keeps its dot and a hyphen inside a stem stays
unambiguous. A video sitting directly in `source_root` yields the bare stem.

Nothing in the repository parses a flat name back apart — it is an opaque identifier used as
the curated directory name and as the `unique_id` column of `index.csv`. The constraint on
this function is therefore not structural but historical: 40 curated folders already exist on
disk and in the `environments-curated` bucket, and `push_curated.sh` runs `rclone copy` rather
than `sync`, so a renamed folder is uploaded again beside the old one and the old one never
goes away.

### Verified against the real tree

The rule above was run over `gdrive-src` before being written down.

| Quantity | Result |
|---|---|
| Pairs derived | 49, up from 40 |
| Existing names reproduced byte-identically | 38 of 40 |
| Existing names changed | 2 |
| New names | 9, all audiomoth |

Representative derivations:

| Video | Flat name |
|---|---|
| `2026-07-22/splats/GH010234.mp4` | `2026_07_22-splats-GH010234` |
| `2024-07-13/GPM_SPLAT/GH010198.mp4` | `2024_07_13-GPM_SPLAT-GH010198` |
| `2026-06-29/Phone pics and splat videos/PXL_20260629_225753909.TS.mp4` | `2026_06_29-Phone_pics_and_splat_videos-PXL_20260629_225753909.TS` |
| `2026-06-03/GH010218.mp4` | `2026_06_03-GH010218` |
| `audiomoth-only-deployments/20260817-20260824/boston-charlesgateeast-riverbank/splat_videos/GH010259.mp4` | `audiomoth_only_deployments-20260817_20260824-boston_charlesgateeast_riverbank-splat_videos-GH010259` |

The longest new name is 101 characters, well inside the 255-byte filename limit.

### The two renames

`2026_06_03-2026_06_03-GH010218` and `-GH010219` become `2026_06_03-GH010218` and
`2026_06_03-GH010219`.

The duplicated component was never a decision. Those two videos sit directly in a date folder,
and the current `plan_pairs` passes `folder.name` for the parent in a branch where `folder`
*is* the date directory, so the date is emitted twice. The generalized rule joins one path
component and produces the name that was always intended.

Preserving the old names was considered and rejected. It would cost one line —
`if len(parts) == 1: parts = parts * 2` — but that line turns an artifact into deliberate,
tested behaviour that outlives everyone who remembers why it is there. Two folders is a
cheaper price.

### Collisions

The existing flat-name collision check in `plan_pairs` is unchanged and now carries more
weight, because sibling directories differing only in punctuation — `boston-ringerpark-west`
against `boston_ringerpark_west` — sanitize to the same component. It raises before any bytes
move, which is the correct outcome: silently overwriting one capture with another is worse
than a failed run.

## Injection is unaffected

The `.mov` edits are the first non-`.mp4` files to reach `inject`, so the path was checked
rather than assumed.

`temp = curated.with_suffix(".inject" + curated.suffix)` yields `GH010247.inject.mov`, so
ffmpeg infers the mov muxer from the final suffix as intended. `-map -0:d` still applies:
the mov muxer regenerates a `tmcd` track from container metadata the same way the mp4 muxer
does, so excluding the input's data streams loses nothing.

`lpcm` audio, if anything, aligns more cleanly than `mp4a`. AAC priming padding is the entire
reason `ALIGN_TOLERANCE_S` exists; uncompressed audio has none. The tolerance stays as it is
and costs nothing on these clips.

No change to `gpmd_command`, `tag_command` or `inject`.

## What the first run does

- **38 clips skip.** The idempotency gate compares `source.size_bytes` and `mtime` against the
  sidecar, so untouched captures cost a stat each.
- **9 clips process fully** — copy, align, extract, inject, telemetry. 7.3 GB of edits.
- **2 clips reprocess** under their new names, because no sidecar exists at the new path.
  1.8 GB.
- **`index.csv` regenerates** to 49 rows from the sidecars, as it does on every run.

A subsequent `--push` transfers roughly 9 GB: the 9 new clips and the 2 renamed ones.

### Manual cleanup

Four stale folders must be removed by hand. The script does not delete them: removing curated
output is not something a preprocessing run should do unprompted, and `rclone copy` will not
do it either.

```bash
rm -rf ../environments-curated/2026_06_03-2026_06_03-GH010218
rm -rf ../environments-curated/2026_06_03-2026_06_03-GH010219
rclone purge collab-data:environments-curated/2026_06_03-2026_06_03-GH010218
rclone purge collab-data:environments-curated/2026_06_03-2026_06_03-GH010219
```

Run the local removals before `--push`, and the bucket removals after it succeeds.

## Testing

Test-driven, as with the original: tests are written first and watched failing.

**Four `flat_dir_name` tests change signature** from `(date, parent, video)` to
`(video, source_root)`. Their assertions are unchanged in substance — underscore date, hyphen
delimiter, parent case preserved, stem verbatim, hyphens in stem safe.

**One test is deleted and one inverted.** `test_plan_pairs_ignores_undated_top_level_dirs`
asserts the behaviour being removed and becomes `test_plan_pairs_walks_undated_top_level_dirs`.
`test_plan_pairs_walks_videos_directly_in_a_date_folder` keeps its purpose; its expected name
loses the duplicated component.

New tests, all `tmp_path`-based with no real video decoded:

- a video nested four directories deep produces the joined name — the audiomoth shape
- `._C0104.MP4` and `.DS_Store` are never returned by `find_videos`
- `src/` is pruned at any depth, not only at depth two
- a video directly in `source_root` yields the bare stem
- **a regression table pinning real relative paths to their known curated names.** This is the
  test that matters. It hardcodes the five derivations tabulated above, so a future change to
  `sanitize` or to the join cannot silently rename folders already sitting in the bucket.

No test changes below `plan_pairs`. Alignment, extraction, injection, telemetry and the index
never see a path shape.

## Files

| File | Change |
|---|---|
| `scripts/preprocess_gdrive_videos.py` | `plan_pairs` recursive, `flat_dir_name` path-derived, `find_videos` dot filter, `_DATE_RE` deleted |
| `tests/scripts/test_preprocess_gdrive_videos.py` | 4 signature changes, 1 deletion, 1 inversion, 5 new tests |
| `README.md` | note that the capture tree may nest arbitrarily |

## Verification

```bash
# 1. Plan only — expect 49 pairs and the 9 audiomoth names
python scripts/preprocess_gdrive_videos.py --dry-run

# 2. The 38 unchanged names must appear exactly as they do on disk today
python scripts/preprocess_gdrive_videos.py --dry-run 2>&1 | grep -c audiomoth_only_deployments  # 9

# 3. One audiomoth clip end to end, including the .mov path
python scripts/preprocess_gdrive_videos.py --only GH010247

# 4. gpmd survived injection into a .mov container
ffprobe -v error -show_entries stream=codec_tag_string -of csv=p=0 \
  "../environments-curated/audiomoth_only_deployments-20260810_20260831-boston_ringerpark_west-splat_videos-GH010247/GH010247.mov"

# 5. Idempotency — a second run copies 0 bytes and re-injects nothing
python scripts/preprocess_gdrive_videos.py --only GH010247

# 6. Tests
python -m pytest tests/scripts/test_preprocess_gdrive_videos.py

# 7. Full run, manual cleanup, then push
python scripts/preprocess_gdrive_videos.py
rm -rf ../environments-curated/2026_06_03-2026_06_03-GH01021{8,9}
./scripts/push_curated.sh --dry-run
python scripts/preprocess_gdrive_videos.py --push
```

## Consequences to expect

- The curated tree goes from 40 folders to 49.
- Two folders change name; four stale folders need manual removal, two of them in GCS.
- Roughly 9 GB is uploaded once on the next push.
- A future capture drop at any nesting depth is curated with no code change, provided it keeps
  the `src/` convention. That convention is now the only structural assumption the script makes
  about the tree.
