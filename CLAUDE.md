# FarmHub

Full specification: @SPEC.md — read it at the start of every session.

## Rules of engagement
- Build milestones in §11 order. Current milestone: M0.
- A milestone is done only when its tests pass. Do not start the next one until I say so.
- §3 is non-negotiable. Refuse conflicting requests and quote the rule number.
- If something is ambiguous, stop and ask. One good question beats 400 wrong lines.
- No dependencies outside the approved list in SPEC.md §12 without asking.
- Log every architectural choice in docs/DECISIONS.md (dated, one short entry).

## Commands
- Install: uv sync
- Test: uv run pytest            (tests/safety/ must never be skipped)
- Lint/format: uv run ruff check . && uv run ruff format .
- Types: uv run mypy --strict src/farmhub/core

## Workflow
- One branch per milestone, conventional commits.
- Dry-run defaults ON in code permanently; production config turns it off explicitly (from M7).