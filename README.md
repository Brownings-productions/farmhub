# farmhub

Smart functions and AI control: a local-first AI hub for a Norwegian homestead.

The specification is [SPEC.md](SPEC.md); working rules are in [CLAUDE.md](CLAUDE.md); decisions are
logged in [docs/DECISIONS.md](docs/DECISIONS.md) and safety-test progress in
[docs/SAFETY_CHECKLIST.md](docs/SAFETY_CHECKLIST.md).

## Development

```sh
uv sync                                   # uses the uv-managed Python pinned in .python-version
uv run farmhub --version
uv run farmhub config check               # validate config, show effective safety settings
uv run pytest                             # tests/safety must never be skipped
uv run ruff check . && uv run ruff format .
uv run mypy --strict src/farmhub/core
uv run lint-imports
```

Dry-run is ON by default. Current milestone: M0 (scaffold).
