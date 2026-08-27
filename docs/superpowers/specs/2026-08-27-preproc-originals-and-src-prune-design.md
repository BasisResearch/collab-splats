# preprocess_gdrive_videos.py — originals in scope, redundant src/ pruned

Date: 2026-08-27
Status: approved, ready for an implementation plan

Extends [2026-08-11-preprocess-gdrive-videos-design.md](2026-08-11-preprocess-gdrive-videos-design.md)
and [2026-08-17-preproc-arbitrary-capture-trees-design.md](2026-08-17-preproc-arbitrary-capture-trees-design.md).
The depth-agnostic walk, flat naming, alignment solver, GPMF extraction, injection, telemetry
and index all stand. What changes is the rule deciding *which* videos are in scope, the shape
of the code that carries a video through the pipeline, and a new mode that deletes redundant
copies out of the capture tree.

## Problem

The script curates a video only when it can pair it with a camera original under `src/`:

> A video is an *edit* if and only if `<parent>/src/<stem>.*` exists. [...] Only pairs are
> processed; an original with no export is skipped and logged.

That rule was written when `src/` was assumed to hold the original behind every capture. It
does not. `src/` exists only where somebody actually cut a video in DaVinci Resolve; where no
edit was needed the camera original sits alone at the top level. Under the current rule those
captures are invisible.

Measured against the real tree on 2026-08-27:

| | count | bytes |
|---|---|---|
| Videos paired with a `src/` counterpart | 57 | 25.2 GB |
| Videos with no `src/` counterpart — **skipped today** | 46 | 19.2 GB |
| Files inside a `src/` directory | 63 | |
| Files in `src/` with no sibling above them (orphans) | 6 | |
| `src/` directories | 25 | |
| Curated output folders | 52 | 23.8 GB |

Forty-six captures — nearly half the corpus — are logged as `skip <name>: no src/ counterpart
(unedited original)` and never curated, when in fact they are the cleanest inputs in the
tree: the camera original itself, carrying its own `gpmd` track, its own GPS, and its own
UserData block, with no Resolve export to align against or reconstruct.

A second defect compounds it. Sixteen of the 57 "pairs" are not pairs at all. The top-level
file and its `src/` counterpart are byte-for-byte identical — somebody copied the original
into `src/` without editing it. Verified by size plus a head/tail 1 MB MD5 over all 16:

```
2024-07-01/phone/PXL_20240702_013733661.TS.mp4       2026-03-30/videos_for_splats/PXL_20260330_141332968.TS.mp4
2024-07-11/SplatsSD/C0119.MP4                        2026-03-30/videos_for_splats/PXL_20260330_142756473.TS.mp4
2024-07-11/SplatsSD/C0120.MP4                        2026-03-30/videos_for_splats/PXL_20260330_151944642.TS.mp4
2024-07-11/SplatsSD/C0121.MP4                        2026-04-03/videos_for_splats/IMG_0005.MOV
2024-07-12/GPM_SPLAT/GH010191.MP4                    2026-04-03/videos_for_splats/IMG_0006.MOV
2025-07-17/GOPROC_Splat_plus/GH010209.MP4            2026-06-29/Phone pics and splat videos/IMG_4085.MOV
2026-03-27/videos_for_splats/PXL_20260327_172112473.TS.mp4   2026-06-29/Phone pics and splat videos/PXL_20260629_225753909.TS.mp4
2026-03-27/videos_for_splats/PXL_20260327_173522801.TS.mp4
2026-03-27/videos_for_splats/PXL_20260327_182052402.TS.mp4
```

Each duplicate costs a full audio decode of both files, a cross-correlation, and a whole-file
`gpmd` remux to reproduce metadata the file already carried — and doubles the footage stored
on Drive.

## Rules

One rule, applied per directory in the existing walk, decides everything:

```
<dir>/<stem>.mp4  +  <dir>/src/<stem>.*   → EDIT      align, inject
<dir>/<stem>.mp4  and no src/ match       → ORIGINAL  copy, read its own metadata
<dir>/src/<stem>.* with no sibling above  → SKIP, logged as an orphan
```

`src/` stays pruned from the walk, so a file inside it is never itself treated as an edit.
Every non-`src` video in the tree is now in scope: 57 edits plus 46 originals = 103, becoming
41 edits plus 62 originals once the prune (below) reclassifies the 16 duplicates.

Stem matching is unchanged — case-insensitive, extension-insensitive, dotfiles and
AppleDouble sidecars excluded — because the real tree differs in both (`GH010234.mp4` against
`src/GH010234.MP4`, `IMG_4085.mp4` against `src/IMG_4085.MOV`).

### Orphans

Six files sit inside a `src/` directory with no counterpart above them, and none is curated
today:

```
2024-07-11/SplatsSD/src/C0122.MP4
2024-07-11/SplatsSD/src/C0123.MP4
2025-07-17/GPM_SPLAT/src/GH010206.MP4
2026-06-03/src/GH010220.MP4
2026-07-22/splats/src/GH010233.MP4
audiomoth-only-deployments/20260817-20260824/boston-copps-hill-shrubbery/splat_videos/src/GH010252.MP4
```

They are skipped and logged at `WARNING`, one line each plus a count. Curating them in place
would bake `src` into their flat name; moving them up would mutate Drive beyond what the
prune does. Neither is worth doing automatically for six files whose correct home is a
judgement call. The log line is what makes them actionable, which is precisely what the
current silent behaviour fails to provide.

## Curating an original

An original produces the same curated output as an edit — `<stem>.mp4`, the JSON sidecar,
a Parquet telemetry sidecar when IMU data exists, one row in `index.csv`. Downstream cannot
tell them apart except by reading a field.

Two steps are skipped, both because there is nothing for them to do:

- **No alignment.** Audio cross-correlation exists to locate an edit inside its source. An
  original is its own source, so the alignment is the identity `{"offset_s": 0.0, "r": 1.0,
  "ok": True}` by construction, not by measurement. This skips two full ffmpeg audio decodes.
- **No injection.** The `gpmd` remux exists because Resolve strips timed data tracks on
  export. An original was never exported; its `gpmd` track is present and already on the
  correct time axis. This skips a whole-file remux.

Together these are the expensive half of the pipeline, so 46 extra videos cost roughly a copy
each rather than a full reprocess each.

Metadata is read from the file itself: `exif_dump` is called on `source or video`, and for an
original that is the video. Every downstream consumer — `static_tags`, `telemetry_table`,
the GPS derivation — is unchanged, because it was always reading the file that holds the
GPMF payload.

### Sidecar fields

The sidecar gains two fields — `edited` and `gpmd_native` — so no consumer has to infer
provenance from path shapes. `source_path` and `gpmd_injected` already exist and are shown
for contrast:

| field | edit | original |
|---|---|---|
| `edited` | `true` | `false` |
| `source_path` | the `src/` original | the video itself |
| `gpmd_injected` | `true` when the remux succeeded | `false` |
| `gpmd_native` | absent | `true` |

`gpmd_native` exists to stop `gpmd_injected: false` from being read as "this mp4 carries no
telemetry". On an edit that reading is correct — injection failed, the track is missing. On
an original it is exactly wrong: the track is there and was never touched.

## Code shape

The change is carried by functions taking explicit arguments. `Pair`, `Alignment` and `Chunk`
— the three `NamedTuple`s — are removed. `Pair` in particular is threaded through six
functions and is what the new rule would otherwise have grown a variant of.

```python
def find_source(video):                       # unchanged: stem match in <parent>/src, or None
def walk_folders(root):                       # unchanged: prunes src/ and dotted dirs
def plan_videos(root):                        # replaces plan_pairs
    """[(video, source_or_None, flat_name)] sorted by name; logs orphans; raises on collision."""

def solve_offset(edit_audio, source_audio, rate, min_r)   -> dict
def align(video, source, name, rate, min_r)               -> dict
def gps_payload(dump, alignment)                          -> dict
def telemetry_table(dump, offset_s, duration_s)
def inject(curated, source, offset_s, duration_s, tags)   -> bool
def process_video(video, source, name, output_root, min_r, force) -> dict
```

`iter_documents` returns `(main, [(number, tags, subs), ...])` and `gps_fixes` unpacks the
tuple. The flat-name collision check in `plan_videos` is unchanged and still raises
`ValueError`; it now covers originals too, which widens the set of names it protects.

### alignment as a dict

`align` returns `{"offset_s": float, "r": float, "ok": bool}`. That is already the exact shape
the sidecar stores under `"alignment"`, so the record is written through verbatim with no
repacking step. `telemetry_table` and `inject` take the scalar `offset_s`, because they need
one number rather than the record — so the dict never spreads past `process_video` and the
sidecar assembly. The cost is real and accepted: a mistyped key fails at runtime rather than
at read time. Three keys, four call sites, one file.

### build_payload is removed

The old `build_payload` mixed two jobs: assembling a flat dict, and deriving GPS anchoring.
Only the second has logic — the first-locked-fix-inside-the-window search, its whole-source
fallback, `gps_source_anchored`, and the `gpmd` chunk-boundary residual. That part becomes
`gps_payload(dump, alignment)`, two arguments. The assembly becomes a literal in
`process_video`, where the sidecar's shape is readable in one place:

```python
payload = {
    "unique_id": name,
    "edit": str(video),
    "source_path": str(meta_src),
    "source": source_fingerprint(video),
    "edited": edited,
    "duration_s": duration_s,
    "has_imu": table is not None,
    "alignment": alignment,
    "static_tags": static_tags(dump),
    "intrinsics": {"valid_for_edit": True, "reason": "cuts and colour correction only, no reframe"},
    **gps_payload(dump, alignment),
}
```

`source_fingerprint` is computed here rather than passed in, and `has_imu` is derived from the
table at the point the table exists. Both were arguments only because the builder could not
see what it needed.

### process_video

```
dest_dir = output_root / name
edited   = source is not None
meta_src = source or video                    # an original is its own metadata source

copy, refresh, or skip per the checks below
alignment = align(video, source, name, ...) if edited else {"offset_s": 0.0, "r": 1.0, "ok": True}
dump      = exif_dump(meta_src)
duration  = probe_duration(dest_dir / video.name)
table     = telemetry_table(dump, alignment["offset_s"], duration); write when not None
payload   = { ... as above ... }
if edited:
    payload["gpmd_injected"] = inject(dest_dir / video.name, source, alignment["offset_s"], duration, payload["static_tags"])
else:
    payload["gpmd_injected"], payload["gpmd_native"] = False, True
write_metadata(...)
```

## Pruning redundant src/ copies

```
preprocess_gdrive_videos.py [DIR] --prune-src                    # report only, the default
preprocess_gdrive_videos.py [DIR] --prune-src --apply            # delete
preprocess_gdrive_videos.py [DIR] --prune-src --only 2026_07     # one drop at a time
```

`DIR` is the capture tree, taken positionally and defaulting to `../gdrive-src`. `--only`
reuses the existing flat-name substring filter, so the same flag means the same thing in both
modes and the tree can be worked through a drop at a time.

### The redundancy test

Three conditions, ordered cheapest first, all required:

1. the video pairs — `<dir>/<stem>.mp4` and `<dir>/src/<stem>.*` both exist;
2. the two byte sizes are equal;
3. the full SHA-256 of both files matches, streamed in chunks.

No partial-hash shortcut: the head/tail probe used to survey the tree is a screen, not a
proof. Condition 2 gates condition 3, so only genuinely equal-sized candidates are ever read
in full — 16 pairs today, not all 57.

Anything failing condition 3 is a real export and stays a pair. A same-duration but
re-encoded export differs in bytes, therefore remains an edit, therefore is still aligned and
injected. The test can only ever delete a file that is a bit-for-bit duplicate of one that
remains on disk.

### What is deleted

The copy **inside `src/`**. The top-level file is never touched. A `src/` directory left
holding no videos is removed; one still holding orphans or real sources is kept.

The report prints, per candidate, both paths, the size, both hashes, and a running total.
`--apply` re-verifies the hash immediately before each `unlink` — the tree is Drive-synced and
may have changed since the scan — then logs one line per deletion and a closing count plus
bytes freed.

### Ordering

Prune first, curate second. After the prune those 16 videos have no `src/` counterpart, so
`plan_videos` reclassifies them as originals with no special-case code anywhere. The prune is
the migration; the curation rule needs no knowledge that it happened.

## Idempotency

`needs_copy` is unchanged in principle: it compares the sidecar's recorded `size_bytes` and
`mtime` for the top-level file against the file on disk, never against the destination's own
size — injection makes the curated mp4 larger than what was copied, so a destination-size
check would fail every run and re-upload gigabytes.

A metadata-refresh check is added beside it. When the copy is current but the sidecar is
stale — no `edited` key, an `edited` value disagreeing with the plan, or a `source_path` that
no longer exists on disk — the run re-reads metadata and rewrites the sidecar and telemetry,
skipping both the copy and the injection.

This is what handles the 16 flipped folders. They are already curated through the paired path:
their mp4 carries an injected `gpmd` track and their sidecar records a `source_path` inside
`src/` that the prune deletes. Their top-level file never changes, so the fingerprint check
alone would report them up to date and leave a pointer to a file that no longer exists. The
refresh corrects provenance without re-copying or re-pushing 3.59 GB of byte-identical
footage. It is written as a general staleness rule, not a one-off migration, so any future
reclassification is handled by the same path.

Per-video status becomes `processed | refreshed | skipped | failed`, each counted in the
closing summary.

## Errors

The per-video `try/except` stays: one unreadable clip — no audio track, a truncated Drive
download — must not discard a multi-hour run that has already copied gigabytes. Failures are
recorded and the run continues, so `write_index` and the summary still execute.

Loud at end of run, unchanged: every alignment rejection (telemetry left on the source time
axis) and every outright failure (nothing curated). Added: the orphan list, which is the one
category that needs a person to file something.

`--index-only` still runs without ffmpeg or exiftool, since it reads sidecars alone.

## Testing

`tests/scripts/test_preprocess_gdrive_videos.py` is 1585 lines and ~146 tests. Its
`plan_pairs` block encodes the rule being inverted —
`test_plan_pairs_returns_only_videos_with_a_src_counterpart` asserts exactly the behaviour
that changes. That block is rewritten against `plan_videos` rather than deleted: case and
extension matching, AppleDouble rejection, the depth-agnostic walk, `src/` pruning at any
depth, and collision detection are all still live requirements and all still need coverage.

New tests:

- a video with no `src/` counterpart plans with `source is None`
- a video with a counterpart plans with that counterpart, unchanged across case and extension
- an orphan inside `src/` is skipped and logged, and never planned
- an original's sidecar carries `edited: false`, identity alignment, `gpmd_native: true`, and
  `source_path` equal to its own path
- an original is never handed to `inject` or `align`
- byte-identical files are reported as prune candidates; same-size-different-bytes are not
- `--prune-src` without `--apply` deletes nothing
- `--apply` removes the `src/` copy, keeps the top-level file, and removes a `src/` left empty
- a `src/` still holding an orphan survives the prune
- a sidecar with a missing or disagreeing `edited`, or a vanished `source_path`, triggers a
  refresh with no re-copy
- a current sidecar triggers neither

All on `tmp_path` fixtures with tiny synthetic files. No Drive mount, no real media, no
network.

## Rollout

Every step gated on explicit go-ahead; nothing runs against real footage before then.

1. `--prune-src` report over the whole tree; read it.
2. `--prune-src --apply`, optionally per drop with `--only`.
3. Curate in `--only` batches, watching the summary counts and the alignment-rejection list.
4. `--push` to upload the delta.

Expected scale after the change: 41 edits plus 62 originals curated, output growing from
23.8 GB to roughly 43 GB, and 3.59 GB of duplicated footage removed from the capture tree.

## Out of scope

- Moving orphans out of `src/`. They are logged; filing them is a human decision.
- Re-encoding, reframing or any pixel change to curated footage.
- Changing `index.csv`'s schema, the alignment threshold, or the GPMF extraction path.
- Anything about the capture tree's folder naming, which the depth-agnostic walk already
  absorbs.
