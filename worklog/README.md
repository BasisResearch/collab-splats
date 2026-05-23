# Worklog

Internal progress tracking. Not user documentation — see [`/docs`](../docs/) for that.

## Layout

- **STATE.md** — current dashboard. Active branch, in-flight work, blockers, parked items. Editable, ≤1 page.
- **WORKLOG.md** — append-only session log. Newest entries on top. Never edit past entries; add new ones.
- **ROADMAP.md** — strategic plan. What's built, where we are, what's next.
- **known-test-failures.md** — living list of expected-broken tests with reasons.
- **specs/** — design specs for in-flight work. Move to `history/specs/` on completion.
- **plans/** — implementation plans for in-flight work. Move to `history/plans/` on completion.
- **notes/** — ad-hoc investigation notes, eval findings, debug write-ups. Dated filenames.
- **decisions/** — ADRs (architectural decisions). Sequential numbering. Never delete; supersede with a new ADR.
- **history/** — archive of completed work: `specs/`, `plans/`, `prs/`. Read-only intent.

## Workflow

1. Start of session: read `STATE.md` → latest entries in `WORKLOG.md` → referenced ADRs in `decisions/`.
2. Active work: read/write `specs/<topic>-design.md` + `plans/<topic>.md`.
3. On completion: `git mv` spec + plan to `history/`; append WORKLOG entry; update STATE.
4. Architectural call: write new ADR in `decisions/NNN-slug.md` (next number). Existing ADRs are immutable — supersede, never edit.
