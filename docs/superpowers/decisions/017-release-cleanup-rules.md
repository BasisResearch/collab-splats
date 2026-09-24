# 017 — Release cleanup rules

Date: 2026-09-24 · Status: accepted · First applied: `preproc`
([spec](../specs/2026-09-24-preproc-release-cleanup-design.md))

Every package cleaned for release follows these rules. Module specs link here and never
restate them. The mechanical subset is enforced by `tests/test_docstring_contract.py`
for every package in its `PACKAGES` tuple.

## Goal

A naive user reads a docstring and knows how to call the function. The package does not
hardcode tunables, does not over-explain, and fails loudly.

## API

- The public API may break before release. Configs, callers, tests and tutorials are
  updated in the same commit as the break.
- A helper earns its existence with two or more callers or a real readability win. A
  single-caller helper is inlined; a single-caller class is made private and shrunk.
- A parameter no caller sets is either deleted or kept as a documented kwarg — the verdict
  table decides which, not habit.

## Tunables

- A number someone could tune is a **kwarg with a default** on the function that uses
  it, threaded up to the public entry point where a caller needs it.
- Fixed facts stay module-level: lookup tables and format identifiers (file extensions,
  enum-to-op maps, schema versions, file names).
- No module-level UPPER_CASE name bound to a bare number, except fixed facts on the
  contract test's allowlist.

## Failures

- Silent fallbacks become errors: no `x or <default>` on a probed value, no
  zeros-on-failure return, no `{"available": False}` sentinel in place of an exception.
- `except` names the exception it expects; a bare `except Exception:` must re-raise.
- "Should not happen" branches raise instead of `continue`.

## Docstrings

- Summary line (≤ 100 chars, does not restate the name), then `- ` bullets only for
  caller-facing contract: units, color order, shapes, what raises.
- Then `Args:` / `Returns:` (or `Yields:`), per CLAUDE.md Code Style.
- At most 6 bullets.
- Never: measurements, speedups, dataset or scene names, "replaces the old X",
  hypotheses, history of how the code got here.

## Comments

- Say what the block does, in 5-10 words.
- Keep a *why* only where the code would otherwise be "fixed" wrongly; 1-3 lines, plus a
  spec path when evidence exists.
- At most 4 lines per comment run.
- Measurement evidence lives in `docs/superpowers/specs/` and git history, not the code.

## Config

- One comment line per key: what it means and which direction does what.

## Process

- Findings, then a verdict table: one row per function and module-level name, verdict in
  {keep, merge, inline, delete, make-kwarg, make-private, raise}, one-line why. The user
  approves the table before any code change.
- Worktree off the integration branch. Every gate runs as
  `cd <wt> && PYTHONPATH=<wt> /opt/venv/reconstruction/bin/python ...` and prints
  `collab_splats.__file__`.
- Round 1 is prose-only: AST equal after deleting every docstring statement on both sides,
  plus one sanity mutation showing the check can fail.
- Round 2 is code: one commit per logical change, each with its gate.
- The package is added to `PACKAGES` in `tests/test_docstring_contract.py` and passes.

## Banned words (docstrings and comments)

`measured`, `HYPOTHESIS`, `x faster`, `ffmpeg pipe`, `replaces the old`, and scene ids
(`GH010229`, `C0043`, …). The contract test owns the authoritative list.
