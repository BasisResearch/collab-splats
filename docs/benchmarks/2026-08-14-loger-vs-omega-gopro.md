# LoGeR vs VGGT-Omega — GoPro walking scene (2026-08-14)

**Scene:** `2026_07_15-Goprosplat-GH010230` from `environments-curated`.
1920×1080 @ 59.94 fps, 15009 frames, 250.4 s. Handheld/worn GoPro, walking pace ~1.16 m/s,
GPS path 288.9 m over a 24.9 × 70.5 m extent. **Not a loop** — start and end are 62.4 m
apart, so no loop closure was available to either backend.

**Question:** LoGeR does 32-frame sliding-window inference; VGGT-Omega attends over every
frame jointly. On the same footage, how do they compare?

---

## 1. What was controlled

The first version of this comparison would have given Omega 300 uniformly-spaced frames and
LoGeR 1 fps — different counts *and* different spacing. That confounds the mechanism under
test, because frame spacing interacts asymmetrically with a sliding window. So instead:

**Both backends consumed one identical frame set.** Omega ran first and produced
`frames.zarr` (251 frames at 1 fps); that store was copied into LoGeR's output directory and
verified byte-for-byte with `diff -r`, and LoGeR's preproc stage logged
`Frames already extracted ... skipping`. Same pixels, same order. Both backends were capped
at the same `max_points: 500000`, so point counts are not a distinguishing metric here —
both saturate the cap.

LoGeR was deliberately *not* handed a larger frame budget. The point was to see the window
mechanism working against global attention on equal input, not to let one backend see more.

---

## 2. Metric definitions

### Reference streams (what "truth" means here)

The GoPro's GPMF telemetry carries three relevant streams:

| stream | rate | what it is |
| --- | --- | --- |
| `CORI` | 59.94 Hz | **C**amera **ORI**entation quaternion — the physical attitude of the camera body |
| `IORI` | 59.94 Hz | **I**mage **ORI**entation quaternion — the HyperSmooth EIS warp applied to each frame |
| `GPS` | 18.18 Hz | latitude/longitude/altitude, projected to a local ENU plane in metres |

CORI samples at exactly one per video frame (15010 samples against 15009 frames), so
telemetry aligns to frames by index — no interpolation, no guessing. COLMAP image names
encode the source frame index (`frame_000063`), which makes that alignment exact.

**Electronic stabilization matters and is not negligible.** IORI reaches 25.7°, median 4.6°,
and changes by a median 5.2° between frames 1 s apart. The model sees the *stabilized* image,
so scoring a reconstruction against raw CORI would compare it to a camera orientation that no
frame actually shows. Which composition of CORI and IORI describes the stabilized image is
undocumented, so all five candidates were tested (see §3).

### The metrics

**Invariant turn-angle difference** *(fit-free — the honest discriminator)*
For each consecutive frame pair, take the *angle* of the relative rotation in the
reconstruction and in the reference, and report `|difference|`. A rotation angle survives
conjugation, so this quantity depends on **neither** the unknown world alignment **nor** the
unknown camera-convention offset. It needs no fitting at all, which is why it — and not any
fitted quantity — was used to choose the orientation convention. It is also a hard lower
bound on the relative-rotation error below.

**Relative rotation error** *(full rotation, after solving the camera offset)*
The same frame pairs, but comparing the complete relative rotation rather than only its
magnitude. This requires first solving the constant rotation between the reconstruction's
camera convention and the GoPro's, because that offset does not cancel: it survives into
relative poses as a conjugation. The identical fit runs per backend, so it cannot favour
either. Reported as median / p95 / max in degrees.

**RPE — Relative Pose Error at stride 1** *(`evals/metrics.py:compute_rpe`)*
Standard evo metric. Compares each consecutive relative pose against the reference's,
reporting translation RMSE and rotation RMSE separately. Measures **local, per-step**
consistency — it is insensitive to slow global drift.

**ATE — Absolute Trajectory Error, Sim(3)-aligned** *(`evals/metrics.py:compute_ate`)*
Standard evo metric. Fits a single global rotation, translation, and **scale** carrying the
reconstruction onto the reference, then reports per-frame position error. Scale must be
fitted because feedforward reconstructions are not metric. Measures **global** trajectory
shape and accumulated drift. Reported in metres.

**Cross-reconstruction agreement** *(the control on the reference itself)*
The same Sim(3) ATE, but between the two reconstructions instead of against telemetry. This
involves no GPS at all, so it separates "the backends disagree" from "the reference is
imprecise". Without this control the ATE column below would be over-read.

**Inference wall-clock** — the model forward pass only, excluding model load, preprocessing,
and COLMAP assembly, all of which were logged separately.

---

## 3. Resolving the stabilization convention

Five CORI/IORI compositions, scored on the fit-free invariant (median, degrees — lower is
better):

| mode | omega | loger |
| --- | --- | --- |
| `cori` (raw) | 2.741 | 2.869 |
| `cori · iori` | 1.624 | 1.547 |
| `cori · conj(iori)` | 5.006 | 5.004 |
| **`iori · cori`** | **0.288** | **0.172** |
| `conj(iori) · cori` | 6.207 | 6.328 |

`iori · cori` wins by more than 5×, and **both backends select it independently** — the
stated validity criterion, since a convention chosen by fitting one backend's error would not
be expected to also win for the other.

Independent corroboration: the reconstructions turn a median **12.02°** per frame;
`iori · cori` says **12.29°**, raw CORI says **15.00°**. Stabilization removing rotation is
exactly the expected physical signature.

---

## 4. Results

| metric | VGGT-Omega (all 251 joint) | LoGeR (32-frame window) | winner |
| --- | --- | --- | --- |
| invariant turn-angle diff, median | 0.288° | **0.172°** | LoGeR, 1.7× |
| relative rotation error, median | 0.592° | **0.359°** | LoGeR, 1.6× |
| relative rotation error, p95 | 1.957° | **1.305°** | LoGeR |
| RPE d=1, rotation RMSE | 1.010° | **0.698°** | LoGeR, 1.4× |
| RPE d=1, translation RMSE | 1.442 m | **1.429 m** | tie |
| ATE Sim(3), RMSE vs GPS | **6.43 m** | 6.53 m | tie (see below) |
| ATE Sim(3), median vs GPS | **5.90 m** | 6.15 m | tie (see below) |
| inference wall-clock | **68.9 s** | 384.9 s | Omega, 5.6× |
| model load | 43.7 s | 47.0 s | — |
| points (both hit the cap) | 500 000 | 500 000 | n/a |

**Control:** LoGeR vs Omega directly, no GPS involved — RMSE **1.21 m**, median 0.62 m,
max 3.66 m, i.e. **0.49 % of the 244 m sampled path**.

### In brief

- **LoGeR is more accurate in rotation, by roughly 1.6×.** Median relative rotation error
  0.359° against Omega's 0.592°. This is not an artifact of the pose-fitting step: LoGeR also
  wins on the fit-free invariant (0.172° vs 0.288°), which involves no fitting whatsoever.
- **Translation is a tie, and GPS cannot resolve it.** The 6.43 vs 6.53 m ATE gap looks like a
  result but is not one — the two reconstructions agree *with each other* to 1.21 m RMSE,
  **5× tighter than either agrees with GPS**. The ~6.4 m is a consumer-GPS noise floor, not
  backend drift. Any conclusion drawn from that column would be reading GPS error.
- **The sliding window did not cost accuracy.** LoGeR's 32-frame window spans only ~37 m of
  walking, against Omega attending over all 251 frames and the full 244 m. It matched Omega on
  trajectory and beat it on rotation. On this scene, global attention bought nothing.
- **LoGeR costs 5.6× in inference time** — 384.9 s vs 68.9 s for the same 251 frames. Model
  load is comparable (47.0 vs 43.7 s).
- **Both backends saturated `max_points: 500000`,** so point density says nothing here. A
  future run should raise or remove the cap to compare reconstruction density.
- **Electronic stabilization is active and material to any telemetry comparison on this
  camera.** Scoring against raw CORI instead of `iori · cori` would have inflated the rotation
  error roughly 10× (2.74° vs 0.288° on the invariant) and would have been silently wrong.

---

## 5. Limits of this result

- **One clip, one scene, one trajectory type** (outdoor walking, no loop). Nothing here
  generalises to indoor, to loops, or to other motion profiles without re-measurement.
- **Translation accuracy is unresolved,** bounded below by GPS quality. Establishing which
  backend drifts less needs a reference better than consumer GPS.
- **Frame count is not near LoGeR's ceiling.** 251 frames was chosen to match Omega's limit,
  not LoGeR's. The sliding-window architecture is built for far longer sequences and its
  actual ceiling is still unmeasured.
- **GPU peak was captured for LoGeR only** (17.3 GB during inference); Omega's was not
  recorded.
- Both backends ran with `loop_closure: false`, `use_multiview_confidence: false`,
  `bundle_adjustment: false` — base.yaml defaults, unchanged.

## 6. Reproducing

Both runs used `docs/examples/run_pipeline.py --stages preproc,pointcloud`, LoGeR with a
single-key override (`pointcloud.backend: loger`) merged over `configs/base.yaml`. The
telemetry→TUM conversion, the orientation-convention search, and the camera-offset fit are
not part of the shipped package; metrics themselves come from `evals/metrics.py`
(`compute_ate`, `compute_rpe`) unchanged.

Outputs: `data/outputs/gopro-compare/{omega,loger}/GH010230/` (gitignored).
