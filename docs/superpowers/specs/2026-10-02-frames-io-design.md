# Keyframe I/O thread pools — design

Status: approved by user 2026-10-02 ("yes please write the spec and implement"). Implementation on
worktree `.worktrees/frames-io`, branch `perf/frames-io`, forked from `clean/final` @ `596041a3`.

## Problem

GH010229 at 984 keyframes (VGGT-Omega + LC) took 1252 s; model compute was ~4 min of it. The keyframe
PNG roundtrip through `/workspace` (MooseFS fuse) is serial at both ends:

| Step | Where | Per 1080p frame | At 984 frames |
|---|---|---|---|
| PNG write, level 1 | `preproc/frames.py` `write_frames`, plain loop | 452 ms (193 ms encode + ~260 ms fuse) | ~445 s |
| Omega preprocess | `vggt_omega.py` `_preprocess` -> upstream `load_and_preprocess_images`, plain loop over paths | 430 ms (PIL decode + crop + bicubic resize) | ~423 s |
| Video decode | `preproc/video.py` | 4.2 ms | 54 s, full video — not a target |

`read_frames` already decodes on a thread pool (`fbb47af0`, report-speed); neither path above uses it.

## Measurements behind the decisions

Scratch: `prof_write_pool.py`, `prof_omega_pool.py` (session scratchpad), GH010229_f1000 frames, load avg ~30.

| Path | Serial | 8 threads | Output |
|---|---|---|---|
| `cv2.imwrite` PNG, 48 frames to `/workspace` | 30.3 s | 3.8 s | bytes identical |
| Omega `load_and_preprocess_images`, 96 frames | 12.7–15.9 s | 1.7–1.8 s | `torch.equal` |
| Omega, 16 threads | — | 2.2 s | no gain over 8 |
| `Image.open(p).size` crop-box scan | 3.4 ms/frame | — | not worth touching |

cv2 and PIL release the GIL in decode/encode/resize, so threads suffice; no processes.

## Design

**A. `write_frames` thread pool.**
- New keyword `workers: int = 8`, same as `read_frames`.
- Each worker converts RGB->BGR and `cv2.imwrite`s one frame; returned paths keep idxs order.
- A failed write (`cv2.imwrite` returns False) raises `OSError` naming the path. The serial loop
  ignored the return value; a worker would hide it the same way.
- Stale-frame removal stays serial, before the pool.

**B. Omega `_preprocess` on path chunks.**
- Private `_load_chunked(paths, resolution, mode, *, workers=8)` in `vggt_omega.py`: split paths into
  `workers` contiguous chunks, run upstream `load_and_preprocess_images` per chunk on a pool, `torch.cat`.
- Upstream is per-image except one cross-image step: frames of different shapes are padded to a common
  size. Chunks that come back with different shapes therefore fall back to one serial upstream call over
  all paths, so output is always exactly what upstream returns.
- Upstream code untouched; crop-box scan untouched.

## Out of scope

- VGGT-X / MapAnything / LoGeR preprocess: same pattern applies; follow-up if wanted.
- JPEG store, in-memory preproc->pointcloud handoff, local-disk staging, single video decode.

## Tests

- `write_frames`: threaded write equals `workers=1` byte for byte, paths in idxs order; failed write raises.
- Omega: real upstream on small PNGs, chunked == single upstream call (`torch.equal`), uniform sizes and
  mixed sizes (fallback); existing `_preprocess` mock test updated for per-chunk calls.
- Gate: `tests/preproc`, `tests/pointcloud`, `tests/test_docstring_contract.py`, `tests/test_import_style.py`.
