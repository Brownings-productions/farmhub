# Decisions

One short entry per architectural choice, dated. Status tags:

- **In force (M0)**: implemented in M0.
- **Recorded**: decided now, implemented at the named milestone.
- **Needs v1.2**: differs from the current text of SPEC.md. It is not in force until the SPEC v1.2 patch is reviewed and accepted.

SPEC references (§, item numbers) are to SPEC.md v1.1 and the M0 review of 2026-09-19.

---

## Shaping M0

### 2026-09-19: Explicit module list lives in the composition root
`registry.py` (core) takes the module list as a parameter. The explicit list is in `farmhub/app.py`, called from `__main__`. Keeps `core` free of imports from `modules`. Refines §6. **In force (M0).**

### 2026-09-19: Two failure kinds
Invalid config, including bad tool files, fails fast for every module. A dependency unavailable at runtime (HA unreachable, DB down) starts the module degraded: marked unhealthy, its tools removed from every tool list, retried with backoff. A core failure fails fast. **In force (M0)** for the registry.

### 2026-09-19: Safety-relevant defaults
Fail-safe defaults are allowed in code (dry-run on, TTL 120 s, T2 rate limit). The effective safety settings are logged at startup and printed by `farmhub config check`. Values with no safe default (confirmers, satellite registry) are required. Interprets §7. **In force (M0).**

### 2026-09-19: Build and CI
GitHub Actions for CI. `uv_build` as the build backend (ships with uv, so it adds no third-party dependency). uv-managed Python 3.12 (`uv python pin 3.12`), not the system interpreter. **In force (M0).**

### 2026-09-19: `farmhub config check`; audit types only in M0
Extra CLI command that validates config and prints the effective safety settings. `AuditEntry` and `AuditSink` exist as types only in M0. Sinks come later (JSONL at M5/M6, Postgres at M2). **In force (M0).**

### 2026-09-19: `core/gateway.py` is the single enforcement point
Every tool call (tier, scope, rate limit, taint, dry-run, audit, PendingAction lifecycle) passes through it. Modules supply handlers and clients only. mypy-strict. Safety test: no handler can be invoked except through the gateway. Settles item 2. Refines §3.2 and §8.1, which put enforcement in the `ha` module, and §5, which has no home for it. **In force (M0)** for the sealing mechanism and fail-closed skeleton. Policy checks land at M5/M6.

### 2026-09-19: Sessions are minted server-side
A session is minted by FarmHub and bound to the authenticated satellite identity. HA's `conversation_id` is a lookup key only, never trusted alone. Transport details settle with Q1 at M1. Settles item 3. **Recorded (M1).**

### 2026-09-19: Action turns see only the current utterance
An action turn receives the current utterance plus a fixed server-side system prompt. No replayed history, including the follow-up turn in a `mixed` flow. Closes taint laundering through history. Safety test that history is dropped lands with the pipeline. Extends §3.5 and §4. **Recorded (M6).**

### 2026-09-19: Audit semantics
- Write-ahead intent row, then an outcome row, linked by `call_id`. "Exactly one" (§9) means one outcome row per call.
- A resolved T2 gets its own outcome row. Working reading: the confirmation or expiry is its own call with its own `call_id` and a `parent_call_id` pointing at the proposal, so each call still has exactly one outcome row.
- Audit write failure means deny (fail closed).
- Immutability: a DB role without UPDATE/DELETE plus a trigger. JSONL is append-only by convention. Hash chain deferred.
- `AuditEntry` is shaped accordingly: `call_id`, `phase` (intent/outcome), `parent_call_id`, `pending_action_id`, plus the §3.7 fields.
- Denials are audited too (§3.7): an unknown tool, a T3 tool and a not-yet-permitted T1+ tool each produce an intent row and an outcome row with `decision=denied`. For an unknown tool the name is recorded as given (truncated to a sane length) and the tier is null.
- `parent_call_id` reading confirmed: a T2 confirmation or expiry is its own call, linked to the proposal.

Refines §3.7 and §9. **In force (M0)** for the types and the gateway's fail-closed behaviour.

### 2026-09-19: Audit sink composition when Postgres is down
Leaning yes, decide at M2: JSONL is the mandatory write-ahead record (a failed JSONL write means deny). Postgres is written as well, with idempotent catch-up from JSONL after an outage, so a DB outage does not deny every tool. **Recorded (M2).**

### 2026-09-19: Provisional protocols; mypy scope; phased import-linter
- Protocols named but not specified in §6 (`LLMBackend`, `Retriever`, `Embedder`, `Reranker`, `DocumentParser`, `Chunker`, `ToolCall`, `HealthReport`) get minimal shapes in M0. Any change is logged at the milestone that first uses them.
- mypy: `--strict` on `src/farmhub/core` now, widened as modules land.
- import-linter: core isolation and layering at M0. The module-independence contract is added when two modules exist. The vLLM-import ban is added when `modules/llm` exists.

**In force (M0).**

### 2026-09-19: Strict tool schemas
Optional parameters become required-but-nullable in generated schemas, so strict JSON-schema modes work. Refines §7. **Recorded (M6).**

---

## Recorded now, implemented later

### 2026-09-19: HA tools are TOML script wrappers; non-HA tools are Python
Every HA-actuating tool is a TOML declaration wrapping one HA script. Non-HA tools (records, cad) are Python `ToolSpec`s through the same gateway. Parameterized tools use enums or bounded numbers only, never free-text entity or service arguments. "No Python to add a tool" applies to HA tools. Refines §3.1 and §7. **Recorded (M5/M6).**

### 2026-09-19: Voice confirmation via a fixed HA intent (proposed Q6)
A fixed HA intent ("confirm" / "bekreft"), handled deterministically by HA, calls a FarmHub confirm endpoint with the satellite's `device_id`. It confirms the single pending action for that satellite's session. With zero or several pending, it does nothing and says so. Not model-interpreted. Add to §14 as Q6, finalize at M8.
**Needs v1.2:** §3.3 as written requires a structured callback *carrying the UUID*. This design does not carry one. The v1.2 patch must amend §3.3 before M8. To settle at M8: read the exact arguments aloud in the confirmation prompt, and what a stray "confirm" heard near a satellite can do.

### 2026-09-19: Service-to-service bearer token
A shared bearer token for `/v1/chat/completions` and the confirm endpoint, a configurable bind address, and a firewall to the `ha` host. §13 means no user accounts; service tokens are fine. Settle with Q1 at M1. **Recorded (M1).**

### 2026-09-19: Client-supplied tools and system prompts are ignored
FarmHub builds its own tool list and system prompt. Safety test at M1. **Recorded (M1).**

### 2026-09-19: Free-text T0 results
Free-text results are typed-and-stripped or marked untrusted. `query_service_history` is marked `reads_untrusted_content=True` (notes are STT text). `get_entity_state` and `list_area_status` return state plus allowlisted typed attributes only. Amends the §8.1 table. Settle at M5. **Recorded (M5).**

### 2026-09-19: Dry-run semantics
- (a) T2 still creates a PendingAction and needs confirmation.
- (b) Push notifications are simulated; confirmations come from the test harness.
- (c) External side effects (HA calls, printer uploads) are blocked. Local effects (service-event writes, sandbox runs) proceed, with `dry_run=true` in the audit row.
- (d) The code default stays ON permanently. Production config sets it off explicitly, and startup logs a warning when it is off.
- (e) Settable from every layer. The effective value is logged at startup.

**Needs v1.2:** (c) contradicts §3.8 as written ("every tool at T1 and above ... returns a simulated success"), and (e) differs from "flag and env var". CLAUDE.md ("ON by default until M7") also needs updating. Not implemented until v1.2 is accepted. Until then the M0 default (ON, all layers) is the only part in force.

### 2026-09-19: T3 declarations are status-only
The loader rejects any T3 entry with an actuating handler or script. Reconciles §3.2, §3.9 and §7. **Recorded (M5/M6).**

### 2026-09-19: Scopes, ceilings, confirmers
Each satellite has a home area and a scope (a set of areas it may act on). Effective tools = tier ≤ satellite ceiling AND `allowed_scopes` intersects scope. Confirmers are per area. `"*"` is rejected on T2 tools. Working addition: a satellite ceiling above T2 is rejected. **Recorded (M5/M9).**

### 2026-09-19: PendingAction storage
PendingActions live in Postgres. Resolution is an atomic conditional UPDATE. All pending rows are expired and audited at startup. Duplicate HA events are idempotent. **Recorded (M8).**

### 2026-09-19: HA-side watchdogs (§3.4)
Tool TOML carries `max_runtime_s`, validated against parameter maximums, plus a RUNBOOK checklist for the HA side. Assume the non-admin HA user does NOT restrict service calls; verify at M5 and record. Containment rests on the gateway, the HA-side watchdogs and the §10 hardware cutoffs. **Recorded (M5).**

### 2026-09-19: New open questions
- **Q6 (M8):** voice confirmation mechanism (see above).
- **Q7 (M12):** printer stack (Moonraker / OctoPrint / PrusaLink). The uploader never sends a start flag or enables auto-start, with a test.
- **Q3 (M6):** T0/T1 rate-limit values, still open. "Per-session" is now well-defined by the session decision.

To be added to §14 in the v1.2 patch. **Recorded.**

### 2026-09-19: §9 tests land with their subject
Each §9 test lands in the milestone that creates its subject. Checklist kept in `docs/SAFETY_CHECKLIST.md`. Refines §11 M6 ("all of §9 passes"). **In force (M0)** for the checklist.

### 2026-09-19: Per-model LLM profiles
The config schema carries per-model profiles (model id, quantization, `gpu_memory_utilization`, `max_model_len`), including a single-GPU profile with a smaller model. The §2 numbers were wrong (about 30 GB of FP8 weights exceed 0.90 x 32 GB). The model settles at M1. First candidate is the same model in NVFP4, evaluated on tool-call fidelity and part numbers. **Recorded (M1).**

---

## Made while implementing M0

### 2026-09-19: Gateway ordering and edge cases
Order: resolve the tool, write the INTENT row (failure means deny, nothing runs), decide, run or deny, write the OUTCOME row. Denial reasons are checked in this order: unknown tool, T3, module not running, T1+ (denied until the M6 policy exists). T0 scope filtering is deliberately not in the M0 gateway: it arrives with the M6 policy. A handler exception or timeout is an `executed` outcome with `ok=false` and a generic error for the model (exception text is logged, not shown). If the OUTCOME row cannot be written after the call ran, the real result is still returned and a critical log line carries the row, because hiding an executed action would misreport what happened. Audit payloads (arguments, results) are capped at 4096 serialized characters and replaced by a truncation marker beyond that. Refines §3.7. **In force (M0).**

### 2026-09-19: `UnconfiguredAuditSink`
Until a real sink exists, `build_app` uses a sink that refuses every write, so the gateway denies every call. This is the fail-closed reading of "audit write failure means deny" and means M0 cannot run a tool by accident. **In force (M0).**

### 2026-09-19: Handler sealing
`ToolSpec` wraps its handler so it only runs when the `ToolContext` carries the gateway's seal. Python cannot make this private, so `tests/safety/test_gateway_only.py` also fails if any source file except `core/protocols.py` and `core/gateway.py` reads `.handler` or references `GATEWAY_SEAL`. **In force (M0).**

### 2026-09-19: `ToolSpec.parameters` is validated at construction
It must be an object schema with `additionalProperties: false` (§6, §7), otherwise `ConfigError`. Full per-parameter rules (bounds, `required`) arrive with the tool loader at M5/M6. **In force (M0).**

### 2026-09-19: Registry behaviour beyond the SPEC
- `find_tool` (gateway only) returns any registered tool including T3 and tools of degraded modules, so a denial is audited with the correct tier. `available_tools` never lists T3 and only lists RUNNING modules.
- Known limitation: a module that is degraded from its very first start has never returned its tools, so a call to one of them is audited as an unknown tool. It is still denied.
- The registry calls `Module.shutdown()` after a failed or degraded startup and before each retry, so modules must make `shutdown` idempotent. An unexpected (non-dependency) error during a retry marks the module FAILED and stops retrying.
- Periodic health polling of running modules is deferred to M1, when the first real module needs it. Modules can already call `mark_unhealthy`.

**In force (M0).**

### 2026-09-19: Misspelled `FARMHUB_*` variables fail startup
pydantic-settings silently ignores unknown top-level environment variables, so `FARMHUB_DRY_RUM=false` would leave the default in force unnoticed. `load_settings` rejects any `FARMHUB_*` name that matches no setting, per §7 "fail loudly". **In force (M0).**

### 2026-09-19: The safety-suite guard converts xfail as well as skip
`tests/safety/conftest.py` marks skipped and xfailed reports as failed and clears `wasxfail`. Without clearing it, pytest still filed an xfail under "xfailed" and exited 0 while printing "1 failed"; `tests/unit/test_safety_conftest.py` found this. **In force (M0).**

### 2026-09-19: CI workflow action versions are unverified
`.github/workflows/ci.yml` uses `actions/checkout@v4` and `astral-sh/setup-uv@v6`, written without network access to confirm current versions. Confirm on the first run. **In force (M0).**
