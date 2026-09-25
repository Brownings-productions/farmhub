# FarmHub

Full specification: @SPEC.md — read it at the start of every session.

## Rules of engagement
- Build milestones in §11 order. Current milestone: M1.
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
- **Never run `git push`, in any form.** I push. Not with `GIT_SSH_COMMAND`, not with any
  other environment prefix, not via an alias, a config change or a script, and not
  `--force`. Any command whose text contains `git push` is covered by this, prefixed or
  not. When a push is needed, say so and stop.

## Boundaries
- **This machine only.** Never contact Home Assistant or any other host — no HTTP, no
  WebSocket, no SSH, no ping. Two exceptions, both local containers on this machine: the
  vLLM container, and the throwaway Home Assistant dev container used for the M1
  end-to-end test. **Any real Home Assistant instance is off limits.** The `ha` box does
  not exist yet; when it does it will run heating and pumps (SPEC §3.4, §10), so the
  habit has to be right before it is there.
- **Never print a token or a secret**, in a command, a log line, a test fixture or a
  commit. Report whether one is set, never its value.
- **If a rule or a deny rule blocks something you think you need, stop and ask.** Never
  rephrase a command to get past a block, and never reach for a different tool to do what
  the blocked command would have done. A block is an answer, not an obstacle.

## Workflow
- One branch per milestone, conventional commits.
- Dry-run defaults ON in code permanently; production config turns it off explicitly (from M7).