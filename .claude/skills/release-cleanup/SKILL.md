---
name: release-cleanup
description: Use when cleaning a collab_splats package for release — brief docstrings/comments, removing over-engineering, turning hardcoded tunables into kwargs, making silent fallbacks raise. Takes the package path as argument.
---

# Release cleanup

Rules: `docs/superpowers/decisions/017-release-cleanup-rules.md`. Read it first. Do not
restate or reinterpret it; if a rule does not fit, stop and ask.

Reference run: `docs/superpowers/specs/2026-09-24-preproc-release-cleanup-design.md`.
Match its structure and verdict-table shape.

## Steps

Stop for user approval at every **STOP**.

1. **Read every file** in the package in full, `viz`/plotting modules included. Do not
   sample.
2. **Find callers** of every public name and every kwarg (`grep -rn` over
   `collab_splats configs scripts tests evals docs/source`). A claim in your prompt about
   who calls what is unverified until you grep it.
3. **Findings** — three lists: stale/false docs (verify each against the code),
   over-explained docs/comments (file:line), silent fallbacks and hardcoded tunables.
4. **Verdict table** — one row per function and module-level name:
   `| name | verdict | why |`, verdict in {keep, merge, inline, delete, make-kwarg,
   make-private, raise}. **STOP.**
5. **Spec** at `docs/superpowers/specs/<date>-<package>-release-cleanup-design.md`:
   link 017, then Round 1 per file, Round 2 per file, caller sweep, testing, out of scope.
   Commit with `git commit --only <spec>` (shared index; other sessions stage too).
   **STOP.**
6. **Plan** via superpowers:writing-plans. **STOP.**
7. **Worktree** `.worktrees/<package>-release` off `clean/final`. Symlink `third_party/*`
   in, or guarded tests SKIP instead of failing.
8. **Baseline gate** before any edit; record pass/fail/skip counts.
9. **Round 1 — prose only.** Proof: AST equal after deleting every docstring statement
   on both sides, plus one sanity mutation that makes the proof fail.
10. **Round 2 — code**, one commit per logical change, gate after each.
11. **Contract**: add the package to `PACKAGES` in `tests/test_docstring_contract.py`;
    it must pass.
12. `graphify update .`; report counts against the baseline. Do not merge — the user
    decides.

## Gate

```bash
cd <wt> && PYTHONPATH=<wt> /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" \
  && cd <wt> && PYTHONPATH=<wt> /opt/venv/reconstruction/bin/python -m pytest tests/<package> <caller test dirs> -q
```

The printed path must be inside `<wt>`. Never pipe pytest through `tail` (it hides the
exit code). Never `--tb=no`.

## Traps

- `cwd` resets between Bash calls: prefix every command with `cd <wt> &&`.
- Never `git commit --amend`, rebase or reset while other sessions are active; fix with
  a new commit.
- Split identifier renames from prose edits — a mixed commit voids the AST proof.
- A renamed state-dict key or config key breaks saved artifacts: grep checkpoints and
  configs for the old name.
- US spelling (`color`, not `colour`).
