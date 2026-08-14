# Backend benchmarks

Measured comparisons between reconstruction backends on real scenes. One file per
comparison, named `YYYY-MM-DD-<what>-<scene>.md`.

These are **measurements, not decisions.** Architecture decisions live in
`docs/superpowers/decisions/`; in-flight design work lives in `docs/superpowers/specs/`.
A report here records what was measured, on which scene, under which controls, and what the
result does *not* support.

## Reports

| date | comparison | scene | headline |
| --- | --- | --- | --- |
| 2026-08-14 | [LoGeR vs VGGT-Omega](2026-08-14-loger-vs-omega-gopro.md) | GoPro walking, 251 frames @ 1 fps | LoGeR's 32-frame sliding window matches Omega's global attention on trajectory and beats it ~1.6× on rotation, at 5.6× the inference time. Translation is GPS-limited and unresolved. |

## Tooling

A report must be reproducible from tracked code, but measurement scaffolding is not package
code. Everything a comparison needed and the package did not already have is archived in
[`scripts/`](scripts/README.md) — libraries, driver, their tests, and the one-off diagnostics
that back a specific claim — tracked and runnable, but imported by nothing and collected by no
CI run. Metrics themselves are the exception: they come from `evals/metrics.py` via
`evals/scripts/eval_compare.py`, unmodified, because that runner already existed.

## What makes a report trustworthy

Reports in this directory are expected to state these explicitly, because each one has
changed a conclusion at least once:

- **What was held constant.** Backends must consume an identical frame set, verified rather
  than assumed — differing frame counts or spacing confound the mechanism under test.
- **A fit-free metric wherever a fitted one is reported.** A metric that requires solving an
  alignment can be wrong because the fit failed rather than because the reconstruction is
  bad. Report a quantity that needs no fitting alongside it, and treat it as a lower bound.
- **A control that excludes the reference.** Comparing the reconstructions to *each other*
  separates "the backends disagree" from "the reference is imprecise". In the 2026-08-14
  report this control demoted an apparent trajectory result to a GPS noise floor.
- **Saturated quantities called out.** A metric pinned at a configured cap (`max_points`,
  frame ceilings) distinguishes nothing and must not be read as a tie.
- **What the result does not support.** One scene is one scene.
