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

Dry-run is ON by default. Current milestone: **M1** (llm + FastAPI + `/v1/chat/completions`).

## Running it

```sh
export FARMHUB_API__TOKEN="$(openssl rand -hex 32)"   # required; no safe default
uv run farmhub config check                           # effective safety settings
uv run farmhub serve                                  # the HA conversation endpoint
```

The LLM backend does not have to be running: the `llm` module starts degraded and is
retried, so the server comes up either way and `/health` says so.

| | |
|---|---|
| vLLM, start and stop | `deploy/vllm/` — see [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Home Assistant integration | [custom_components/farmhub/](custom_components/farmhub/) |
| Throwaway HA for development | `deploy/dev/ha-compose.yaml` |
| Model evaluation (needs the GPU) | `uv run python -m evals.run --list`, [docs/MODEL_EVAL.md](docs/MODEL_EVAL.md) |
