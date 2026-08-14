# Archived diagnostics

One-off scripts written to answer a specific question during a benchmark, kept because a
report or a docstring elsewhere asserts a number they produced. **They are evidence, not
tools.** Nothing in the package imports them and nothing runs them in CI.

Anything from this work that was worth keeping as a tool was promoted instead, and lives in
`evals/`:

| promoted to | what it does |
| --- | --- |
| `evals/gopro_telemetry.py` | GPMF telemetry → pose reference sampled at reconstructed frames |
| `evals/rotation_alignment.py` | fit-free turn-angle invariant + camera-frame offset fit |
| `evals/scripts/eval_gopro_reference.py` | writes `gt.tum` + `<method>.tum` for `eval_compare.py` |

## Why keep the rest

A diagnostic earns a place here when it is the only record of *how* a claim was established.
Three of the four below are cited by a docstring in the promoted code: the docstring states a
number, and the script is what produced it. Deleting them would leave assertions with no
provenance — and each of these scripts exists because a first answer was wrong.

| script | question it answered | where its conclusion lives now |
| --- | --- | --- |
| `diag_rotation.py` | Does the rotation disagreement live in the poses or in my fit? | `evals/gopro_telemetry.py` module docstring (stabilisation is material); re-measured every run by `eval_gopro_reference.py --orientation auto` |
| `diag_handeye.py` | Why does the closed-form hand-eye fit miss the invariant floor — axis degeneracy, or the wrong transpose? | `rotation_alignment.fit_camera_offset` docstring (axis-Kabsch numbers) and `gopro_telemetry.reference_rotations_c2w` docstring (the transpose) |
| `decompose_residual.py` | Is the 0.272 % pinhole residual LoGeR's non-pinhole ray field, our K fit, or sharing one K across frames? | `docs/superpowers/specs/2026-08-13-loger-feedforward-backend-design.md` |
| `compare_upstream_focal.py` | Does upstream LoGeR's own focal estimator beat ours on the same fixture? | same spec |

## Running them

The two GoPro diagnostics take the telemetry parquet and a COLMAP model, from the repo root:

```bash
python docs/benchmarks/scripts/diag_rotation.py <telemetry.parquet> <colmap/sparse/0>
python docs/benchmarks/scripts/diag_handeye.py  <telemetry.parquet> <colmap/sparse/0>
```

The two LoGeR residual scripts run against the committed tutorial fixture and need a GPU:

```bash
PYTHONPATH=. python docs/benchmarks/scripts/decompose_residual.py
PYTHONPATH=. python docs/benchmarks/scripts/compare_upstream_focal.py
```

Both GoPro scripts were re-run against the promoted library on 2026-08-14 and reproduce the
figures their docstrings cite (axis singular values 11.6 / 8.12 / 7.04; transpose residuals
0.592 / 0.762 / 7.395 / 7.093 deg; turn angles 12.02 recon vs 12.29 `iori_cori` vs 15.00 raw
CORI). The scene they were measured on is gitignored, so reproducing them needs the
`2026_07_15-Goprosplat-GH010230` scene pulled from `environments-curated`.

One deliberate difference: `diag_handeye.py` keeps its original seven-seed local search, so
its absolute residuals are slightly pessimistic against the promoted
`rotation_alignment.fit_camera_offset`, which sweeps SO(3) first (0.592 → 0.584 deg on this
scene). Its *ranking* of the four transpose conventions — the question it was written to
answer — is unaffected, and keeping the original search is what makes it a faithful record.
