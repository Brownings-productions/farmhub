# Safety test checklist

Each SPEC §9 test lands in the milestone that creates its subject (docs/DECISIONS.md, "§9
tests land with their subject"). `tests/safety/` is never skipped: the guard in
`tests/safety/conftest.py` turns any skip or xfail into a failure, so an unwritten test
is listed here rather than stubbed there.

Status: **done** = passing in `tests/safety/`. **partial** = the M0 part exists, the rest
lands at the milestone shown. **todo** = not yet possible.

## SPEC §9

| # | Test | Milestone | Status |
|---|------|-----------|--------|
| 1 | Model-visible tool list contains no T3 tool or entity, under every scope | M6 | partial: `available_tools()` never lists T3 (`test_baseline.py`); per-scope lists at M6 |
| 2 | A turn containing retrieved content is issued with no tool above T0 | M6 | todo |
| 3 | The `action` turn tool list contains no `reads_untrusted_content` tool | M6 | todo |
| 4 | After an untrusted tool result, a T1+ call in that turn is denied and audited, no HA call | M6 | todo |
| 5 | A classifier failure (bad JSON, timeout) routes to `question` | M6 | todo |
| 6 | No satellite identity, or an unknown one, receives only T0 tools | M6 (identity: M1) | todo |
| 7 | A T2 call without confirmation produces no HA service call | M8 | todo |
| 8 | A PendingAction past its TTL is rejected | M8 | todo |
| 9 | A confirmation from a different session is rejected | M8 | todo |
| 10 | A push confirmation from a device not in the area's `confirmers` is rejected | M8 | todo |
| 11 | A PendingAction cannot be confirmed twice | M8 | todo |
| 12 | Scope filtering: the workshop scope cannot reach greenhouse tools | M6 | todo |
| 13 | Every executed tool call produced exactly one audit row | M0, extended at M2 (JSONL, Postgres) | partial: gateway writes intent + one outcome row (`test_gateway.py`) |
| 14 | Dry-run mode issues zero outbound HA service calls | M6 (HA client: M5) | todo |
| 15 | A tool config referencing a missing HA script fails startup | M5/M6 | todo |
| 16 | A tool config missing `required`, or with unbounded parameters, fails startup | M5/M6 | todo |
| 17 | cad generated code cannot open a network socket | M12 | todo |

## Added by decisions (2026-09-19)

| Test | Milestone | Status |
|------|-----------|--------|
| No handler can be invoked except through the gateway (runtime seal + static scan) | M0 | done: `test_gateway_only.py` |
| Unknown-tool, T3 and T1+ denials each write an intent row and an outcome row with `decision=denied`; unknown tool records the name (truncated) and tier null | M0 | done: `test_gateway.py` |
| Audit intent-write failure denies the call and nothing runs; no audit sink means no tool runs | M0 | done: `test_gateway.py` |
| Outcome-write failure after execution is logged critical and the real result is returned | M0 | done: `test_gateway.py` |
| Dry-run is ON with no configuration; turning it off is explicit | M0 | done: `test_baseline.py` |
| Unknown config keys and misspelled `FARMHUB_*` variables fail startup | M0 | done: `test_baseline.py` |
| `ToolSpec.parameters` must be a strict object schema | M0 | done: `test_baseline.py` |
| Skips and xfails under `tests/safety` fail | M0 | done: `tests/unit/test_safety_conftest.py` |
| Tools of a degraded module are withheld from every tool list | M0 | done: `tests/unit/test_registry.py` (not in `tests/safety`) |
| JSONL audit sink: the intent row is fsynced to the file before the handler runs, and every line is valid JSON | M2 | todo |
| JSONL write failure (unwritable path, simulated full disk) denies the call and nothing runs | M2 | todo |
| JSONL sink appends across restarts without truncating, and concurrent writes never interleave | M2 | todo |
| Startup fails fast when the JSONL path cannot be opened for append | M2 | todo |
| Postgres down: tools still run and both rows land in the JSONL file | M2 | todo |
| Catch-up from JSONL to Postgres is idempotent on `(call_id, phase)`, skips a torn last line, and resumes from the checkpoint | M2 | todo |
| The Postgres audit role cannot UPDATE or DELETE, and the trigger rejects both | M2 | todo |
| A tool file with an error fails startup before any HA connection is attempted | M5 | todo |
| A call to a tool of a not-yet-connected or degraded module, including a T3 declaration while HA is down, is denied and audited with the correct tier, not as an unknown tool | M5 | todo |
| Client-supplied `tools` and system prompts are ignored | M1 | todo |
| Sessions are minted server-side and bound to the authenticated identity; the bearer token is required on the chat and confirm endpoints | M1 | todo |
| Action turns receive only the current utterance: replayed history is dropped, including the follow-up turn of a `mixed` flow | M6 | todo |
| Free-text T0 results are stripped or marked untrusted; `query_service_history` is untrusted | M5 | todo |
| A T3 declaration with an actuating handler or script is rejected by the loader | M5/M6 | todo |
| `"*"` in `allowed_scopes` is rejected on T2 tools | M5/M6 | todo |
| `max_runtime_s` is validated against parameter maximums | M5 | todo |
| A satellite tier ceiling above T2 is rejected | M6/M9 | todo |
| Optional parameters are generated as required-but-nullable | M6 | todo |
| In dry-run a T2 call still creates a PendingAction and needs confirmation; push is simulated | M8 | todo |
| PendingAction rows are expired and audited at startup; duplicate HA events are idempotent | M8 | todo |
| The printer uploader never sends a start flag (Q7) | M12 | todo |
