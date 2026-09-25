# Decisions

One short entry per architectural choice, dated. Status tags:

- **In force (M0)**: implemented in M0.
- **Recorded**: decided now, implemented at the named milestone.
- **Accepted (SPEC v1.2, 2026-09-20)**: differed from SPEC v1.1 and is now part of the accepted SPEC v1.2. The entry says which parts are implemented.
- **Accepted (SPEC v1.3)**: part of the accepted SPEC v1.3 (2026-09-20), which amends §2 and §11 only.

SPEC references (§, item numbers) are to SPEC.md v1.1 and the M0 review of 2026-09-19. SPEC v1.2 was accepted on 2026-09-20 and folds in every decision dated 2026-09-19 below. SPEC v1.3, accepted the same day, carries the hardware change and touches no §3 rule.

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
Extra CLI command that validates config and prints the effective safety settings. `AuditEntry` and `AuditSink` exist as types only in M0. Sinks come at M2 (JSONL and Postgres; see the Q8 plan below). **In force (M0).**

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

### 2026-09-20: Audit sinks when Postgres is down: the JSONL sink is planned for M2 (Q8)
Direction accepted 2026-09-20 (it was "leaning yes, decide at M2"): JSONL is the mandatory write-ahead record, and a failed JSONL write denies the call. Postgres is written as well, with idempotent catch-up from JSONL, so a database outage does not deny every tool. This moves the JSONL sink from M5/M6 to M2, alongside the Postgres sink. SPEC §11's M2 row said the composition is merely "decided"; it now says the JSONL sink is implemented (SPEC v1.3, row 8).

Plan for M2:
- **`JsonlAuditSink`** in `core/audit.py` (mypy-strict). One JSON object per line (UUIDs and datetimes as strings, enums by value, tier as an int). File opened append-only and created 0600. Each entry is a single `write`, then flush and fsync before `write()` returns, so the intent row is on disk before any handler runs. Blocking IO runs off the event loop and writes are serialized by a lock. Any `OSError` (disk full, permission, missing directory) becomes `AuditWriteError`, so the gateway denies.
- **Path** comes from config (`audit.jsonl_path`). Startup opens the file for append and fails fast if it cannot (a core failure). The effective path is logged and printed by `config check`. Whether it needs an explicit value or a development default is settled at M2 under the safety-defaults policy.
- **`PostgresAuditSink`** in `storage/` (`core` cannot import `storage`). Its DB role has INSERT only, plus a trigger that rejects UPDATE and DELETE. `(call_id, phase)` is unique, because each call has exactly one intent and one outcome row, so it is the idempotency key. Inserts use `ON CONFLICT DO NOTHING`, which needs no UPDATE privilege.
- **Composite sink**, wired in `app.py`: write to JSONL first (failure raises `AuditWriteError`), then to Postgres. A Postgres failure logs a warning, marks the sink behind, and never denies the call. With no database configured the JSONL sink runs alone.
- **Catch-up** runs at startup and after the database recovers (on the `module.recovered` event): it replays JSONL rows not yet in Postgres, from a byte-offset checkpoint kept in a small table. Replay is idempotent, so restarting from any offset is safe. A torn last line (crash mid-write) is skipped and logged, not fatal. Rows can reach Postgres out of order, so consumers order by the entry's timestamp, not by insertion order.
- **Rotation** is not in the first cut. Copy-truncate rotation would break both append semantics and the offset checkpoint, so it must not be used; settle a scheme before the file grows large and note it in the RUNBOOK. `chattr +a` on the file is optional hardening. The hash chain stays deferred.
- `UnconfiguredAuditSink` (M0) remains as a test double; `build_app` will wire the composite sink.
- Tests are listed in `docs/SAFETY_CHECKLIST.md`.

Extends the 2026-09-19 audit semantics. **Recorded (M2).**

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

### 2026-09-19: Voice confirmation via a fixed HA intent (Q6)
A fixed HA intent ("confirm" / "bekreft"), handled deterministically by HA, calls a FarmHub confirm endpoint with the satellite's `device_id`. It confirms the single pending action for that satellite's session. With zero or several pending, it does nothing and says so. Not model-interpreted. Q6 is in §14, to finalize at M8.
**Accepted (SPEC v1.2, 2026-09-20)** as §3.3 form (b), amending the earlier requirement that every satellite confirmation carry the UUID. It stays provisional until Q6 is finalized at M8. To settle at M8: read the exact arguments aloud in the confirmation prompt, what a stray "confirm" heard near a satellite can do, and behaviour with several pending actions. **Recorded (M8).**

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

**Accepted (SPEC v1.2, 2026-09-20)** as §3.8, including (c) (external effects only, which amended "every tool at T1 and above ... returns a simulated success") and (e) (all layers, which amended "flag and env var"). CLAUDE.md's dry-run line was updated to match and stays. In force (M0): (d) the code default ON, and (e) settable from every layer with the effective value available in `config check`. Wiring the startup warning into server startup arrives with M1. (a) to (c) are implemented with the dry-run policy at M6, and production config turns dry-run off at M7.

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

Added to §14 in SPEC v1.2 (accepted 2026-09-20), together with Q8. **Recorded.**

### 2026-09-19: §9 tests land with their subject
Each §9 test lands in the milestone that creates its subject. Checklist kept in `docs/SAFETY_CHECKLIST.md`. Refines §11 M6 ("all of §9 passes"). **In force (M0)** for the checklist.

### 2026-09-19: Per-model LLM profiles
The config schema carries per-model profiles (model id, quantization, `gpu_memory_utilization`, `max_model_len`), including a single-GPU profile with a smaller model. The §2 numbers were wrong (about 30 GB of FP8 weights exceed 0.90 x 32 GB). The model settles at M1. First candidate is the same model in NVFP4, evaluated on tool-call fidelity and part numbers. **Recorded (M1).**

Superseded in part by the 2026-09-20 hardware entry below: the single-GPU profile is now the default rather than a degradation and keeps the primary model, the profile gains `revision`, `kv_cache_dtype`, `weights_gb`, `kv_cache_gb`, `aux_reserve_gb` and `measured_by`, and "the same model in NVFP4" turned out not to exist as a vendor build — see that entry for the candidate list that replaces it.

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
- Known limitation: a module that is degraded from its very first start has never returned its tools, so a call to one of them is audited as an unknown tool. It is still denied. Resolved at M5 by the load/startup split (2026-09-20 entry below).
- The registry calls `Module.shutdown()` after a failed or degraded startup and before each retry, so modules must make `shutdown` idempotent. An unexpected (non-dependency) error during a retry marks the module FAILED and stops retrying.
- Periodic health polling of running modules is deferred to M1, when the first real module needs it. Modules can already call `mark_unhealthy`.

**In force (M0).**

### 2026-09-19: Misspelled `FARMHUB_*` variables fail startup
pydantic-settings silently ignores unknown top-level environment variables, so `FARMHUB_DRY_RUM=false` would leave the default in force unnoticed. `load_settings` rejects any `FARMHUB_*` name that matches no setting, per §7 "fail loudly". **In force (M0).**

### 2026-09-19: The safety-suite guard converts xfail as well as skip
`tests/safety/conftest.py` marks skipped and xfailed reports as failed and clears `wasxfail`. Without clearing it, pytest still filed an xfail under "xfailed" and exited 0 while printing "1 failed"; `tests/unit/test_safety_conftest.py` found this. **In force (M0).**

### 2026-09-19: CI workflow action versions
`.github/workflows/ci.yml` uses `actions/checkout@v4` and `astral-sh/setup-uv@v6`, written without network access to confirm current versions. The first CI run was green on 2026-09-20, which confirms them. **In force (M0).**

---

## Made while implementing M1

### 2026-09-20: Q1 settled — a custom HA integration passes identity in headers
A standard OpenAI chat-completions request has no field for the Home Assistant `device_id`, and §3.6 forbids taking satellite identity from message content. The built-in "OpenAI Conversation" integration therefore cannot carry it, and it also builds its own prompt and tool list, which collides with FarmHub building its own. Overloading the `user` field was rejected: FarmHub owns both ends, so a purpose-named header is clearer than a field that means something else.

`custom_components/farmhub/` is a conversation agent kept in this repo. It posts to `/v1/chat/completions` with `Authorization: Bearer …`, `X-FarmHub-Device-Id` and `X-FarmHub-Conversation-Id`, and no tools or system prompt.

**The trust boundary, stated plainly:** the bearer token authenticates Home Assistant and is the only thing FarmHub verifies. `device_id` is an authorization *input* that HA vouches for, not an authentication — anyone holding the token can assert any `device_id`. So the token is the real security boundary, the satellite registry bounds the blast radius, and §3.6's firewall to the `ha` host is load-bearing rather than belt-and-braces.

Consequences implemented at M1:
- The token has no safe default, so `farmhub serve` refuses to start without one. `config check` and the startup log report only whether one is set, never its value.
- `api.bind_host` defaults to loopback; a non-loopback value is allowed but logged as a warning on every startup, so the dev arrangement cannot travel to `hub` unnoticed.
- Sessions are minted server-side and keyed on `(satellite, conversation_id)`, so a conversation id alone can never reach another satellite's session.
- An unregistered `device_id` is served T0-only with a warning rather than refused: §3.6 asks for limited, not mute, and refusing would make a mis-registered satellite silent.
- A minimal satellite registry lands now rather than at M9, so the fail-closed path has something to fail closed *against*. M9 extends it with input mode; `confirmers` are per area (§3.3).

Removed from SPEC §14. **In force (M1).**

### 2026-09-20: the eval harness emits the profile, so nobody types the numbers
`evals/` lives outside `tests/` because it needs a real GPU and SPEC §9 forbids a test that does. It is run by hand; CI never touches it.

- **Pins are checked before anything downloads.** A candidate whose `revision` is not a 40-character commit sha is refused and nothing is fetched. `--check` also reports when a pin has fallen behind upstream, so following a move is a decision rather than a surprise. This is the operational half of the §2 revision rule.
- **The harness writes the `[profiles.*]` block.** `weights_gib` and `kv_cache_gib` come from vLLM's startup log, and `chat_template_kwargs` and `temperature` from what the request actually carried; all of it goes straight into a pasteable TOML block carrying `measured_by`. A human retyping them is exactly how "measured, not estimated" quietly becomes false. A candidate whose numbers could not be parsed gets no block at all, only a comment pointing at its log — a plausible guess there would defeat the rule the block exists to satisfy.
- A test asserts the emitted block is accepted by `ModelProfile`, so the two halves cannot drift apart without the suite noticing.
- **The kernel actually selected is recorded**, not the one requested, because an NVFP4 checkpoint on sm_120 can fall back to Marlin W4A16 and the run would otherwise look like NVFP4.
- **Unparsed numbers stay `None`, never 0.0.** A zero looks like a measurement.
- The matrix expands each candidate over both KV cache dtypes (§2), and Qwen3.6 is run text-only since FarmHub has no multimodal path and the vision tower's memory is memory the KV cache needs.
- 5 GB is held back for Whisper, BGE-M3 and the reranker, and usable context is reported as KV tokens ÷ 3 concurrent sessions (§1), not as `max_model_len`.
- Synthetic data only, committed under `evals/data/`. Scoring is deliberately split: schema violations, invented tools and false positives mean different things, and a single accuracy number would hide which.

**In force (M1).**

### 2026-09-20: `LLMBackend` gains streaming, usage and finish_reason
`core/protocols.py` marked `LLMBackend` PROVISIONAL, to be revised at the milestone that first uses it. M1 is that milestone. It was too thin for §8.2, which requires streaming and token accounting:

- `LLMResponse` gains `finish_reason`, `usage` (a new `TokenUsage`) and `model`. `finish_reason` is worth surfacing because a silent `length` truncation otherwise looks like a short answer.
- `stream_chat` is added, yielding content deltas. It takes no `tools`: a streamed turn is for latency-sensitive prose, and tool calls are resolved by `chat`, where the whole call is seen at once.
- `structured` gains `schema_name` and is documented to raise `LLMError` rather than return something unusable, so the §4 classifier applies its safe default instead of guessing.
- `ChatMessage.role` is narrowed to `system | user | assistant`. The tool-result fields (`tool_call_id` and the rest) arrive at M6, when tool results first have to be fed back.
- New `LLMError`, distinct from `DependencyUnavailable`: the latter means the backend is down and the module should start degraded, the former that one call failed.

**In force (M1).**

### 2026-09-20: `AppContext` carries the LLM backend, not the llm module
`build_app` constructs `OpenAICompatBackend` and puts it in the context; `LlmModule` gets the same object and owns its lifecycle. Constructing the client opens no connection, so building an app stays free of I/O and works with every backend down. The orchestrator can then issue completions without importing `modules.llm`, which keeps SPEC §5's "modules never import each other" intact as the orchestrator lands at M6.

`ENABLED_MODULES` became `default_modules(ctx, backend)` for the same reason: the composition root already holds the concrete object it built, so passing it in beats narrowing the protocol back down. `build_app(settings, [])` still gives an empty app for tests. **In force (M1).**

### 2026-09-20: `respx` cannot test the LLM client; a mock transport does
SPEC §12 lists `respx` for mocking HTTP in tests, and the M1 plan assumed it. It does not work here: `openai` 3.x uses `httpx2`, while `respx` patches `httpx`, so it never sees the SDK's requests.

Instead `OpenAICompatBackend` takes an `http_client` seam, and `tests/unit/test_llm.py` passes a mock transport from the SDK's own HTTP library. This runs in process with no network and no GPU (§9), and exercises the real client, real serialization and real error mapping rather than a stub.

That parameter is typed loosely (`Any`) on purpose, so the transitive HTTP library is named only in tests and never in `src/`. `openai.Timeout` and `openai.omit` are used in the client for the same reason — the SDK re-exports what is needed, so nothing unapproved appears in the runtime imports. `respx` stays in the dev dependencies for the `web` module at M11, which uses `httpx` directly.

**Worth raising:** if the loose typing or the test-only import is not acceptable, the alternative is writing the client on plain `httpx` instead of the SDK. That is more code but removes the mismatch entirely. **In force (M1).**

### 2026-09-22: `httpx2` is a declared dev dependency, tests only
Reviewed and kept: the mock-transport approach above stays, and the plain-`httpx` rewrite is not done. `httpx2` is now declared in the dev dependency group rather than arriving only because `openai` pulls it in, since the tests import it by name and a transitive dependency can change under them. It is used only in `tests/`: a `lint-imports` contract forbids `farmhub` from importing it (indirect imports through the SDK are allowed, direct ones are not). **In force (M1).**

### 2026-09-22: Qwen3.6 runs text-only via `--language-model-only`
The first eval run never started: `--limit-mm-per-prompt {"image":0,"video":0}` lost its JSON quotes on the way from `.env` through compose's whitespace-split command, and vLLM rejected the argument. `--language-model-only` replaces it. It is a plain boolean, so nothing needs quoting, and in v0.29.0 it sets every modality limit to zero. Read from the source: `MultiModalConfig.language_model_only` zeroes the limits, and `_mark_tower_model` skips loading any tower whose modalities are all zero. So the vision tower's weights are not loaded at all, rather than loaded and then left unused, which answers the open question in the M1 plan. **In force (M1).**

### 2026-09-22: vLLM on WSL2 needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`
The second run failed at device init with "UVA is not available". vLLM v0.29.0 turns pinned host memory off under WSL unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is set (upstream opt-in, gated on WSL2 kernel ≥ 4.19.121; the dev PC runs 6.18), and the V2 model runner's staging buffers cannot be allocated without it. `compose.yaml` passes the variable through, defaulting to `0`. `env.example` sets it to `1` for the dev PC, and hub leaves it at `0`, where vLLM ignores it on native Linux anyway. Since this is the dev PC only, it has no bearing on the numbers the eval carries over to hub. **In force (M1, dev PC only).**

### 2026-09-23: the hardware plan has two phases, and Phase 1 is the dev PC
The plan is no longer "develop anywhere, then build `hub`". It is two named phases, and the first one lasts about a year.

**Phase 1, now to roughly September 2027.** FarmHub is developed, measured **and run** on the dev PC's RTX 5090. An RTX 3070 cannot be added to that machine — the PSU will not take it — so the LLM and the helper models (faster-whisper, BGE-M3, the reranker, ~5 GB) share the one card, with `aux_reserve_gib ≈ 5`. This is the production profile for the next year, not a fallback and not a degradation, which is the part that matters for planning: it constrains **M3** (embedding while an LLM is loaded), **M4** (reranking on the same card) and **M9** (Whisper and Piper beside both) exactly as much as it constrains M1. Each of those milestones has to measure what it costs rather than assume there is room.

**Phase 2, `hub`.** RTX 5090 32 GB **plus** RTX 3070 8 GB, decided rather than optional. The 5090 then serves the LLM alone, the helpers move to the 3070, and `aux_reserve_gib` becomes 0. The dev PC takes an RTX 5080 16 GB or a used RTX 4090 24 GB — undecided, settled at the swap — and from then on talks to vLLM on `hub` over the LAN, keeping a small-model local profile as the offline fallback.

**A profile belongs to one machine.** `weights_gib` carries from the dev PC to `hub`, because it is the same card and the same checkpoint. `kv_cache_gib` does not: it is whatever is left once the helpers are elsewhere and no desktop is using the GPU. So **no Phase 2 profile is adopted until an eval run on `hub` produces it**, and the 2026-09-22/23 numbers are explicitly Phase 1 numbers.

**Model inference runs on one machine at a time.** `nas`, `tv` and `ha` never run it, whatever GPU they hold. The only exception is wake-word detection on the satellites, which is a keyword spotter rather than a model FarmHub serves; §8.8 does not currently say where it runs and M9 settles it.

**There is no overnight escape hatch.** An earlier draft of this said the LLM could be stopped overnight to make room for §8.4's contextual retrieval. It cannot: that batch job uses the local LLM to write its blurbs, so it is one more tenant on the shared card. Whether serving goes offline overnight at all is not decided here.

**What Phase 1 does not settle** is uptime, recorded as SPEC §14 **Q9 (M1)**: the dev PC is the running system for a year, yet vLLM is started by hand (`restart: "no"` on purpose, `deploy/vllm/compose.yaml`) and Windows reboots for updates. How both come back, and what a satellite hears meanwhile, is settled before the M1 end-to-end test through Home Assistant.

Supersedes the 2026-09-20 entry below, which wrote the 3070 out of the plan and made the second GPU conditional on M1's measurements. Part of SPEC v1.4, amending §2 and §11 only — no §3 rule is touched. **Proposed (SPEC v1.4).**

### 2026-09-23: storage and media split into `nas` and `tv`
The `nas` row in SPEC §2 described a machine that no longer matches the plan. `nas` is now a new TrueNAS build — corpus, media and backups — starting on integrated graphics, with a transcoding GPU possible later. The existing TrueNAS box becomes `tv`, a TV and emulation machine that keeps its GTX 1060 until that card is upgraded, and is not part of the FarmHub system at all.

Both keep the old rule: **never an AI target.** Model inference runs on one machine at a time — the dev PC in Phase 1, `hub` in Phase 2 (see the entry below). A spare GPU in a media box is a standing temptation, and spreading inference across machines would make the runtime host's measured memory budget (§2) meaningless while putting model serving behind a TV's uptime. The corpus is read over the network from `nas`; nothing about RAG needs a GPU there.

Part of SPEC v1.4 (2026-09-23), which amends §2 and §11 only — no §3 rule is touched. **Proposed (SPEC v1.4)**, accepted with the revision as a whole.

### 2026-09-25: the profile carries its request parameters, and every memory field says GiB
Two changes to `ModelProfile`, both from the M1 runs.

**`chat_template_kwargs` and `temperature` are required profile fields**, sent by `modules/llm` on every request and used whenever a caller names no temperature of its own. Thinking-off is the reason for the first (see the entry below). Sampling is the reason for the second: the runs of 2026-09-22 and 2026-09-23 both took the server's default, so each tool score was one unreproducible draw and the two could not be compared with each other — one case flipped between them and nothing in either run could say why. A profile now states what it was measured at, the harness emits both fields from what the request actually carried, and an empty `chat_template_kwargs` is allowed but warned about. Neither is defaulted in code: a default nobody declared is exactly what produced the two incomparable runs.

**Every memory field is renamed to the unit it was always in:** `weights_gib`, `kv_cache_gib`, `aux_reserve_gib`, `card_total_gib`. vLLM reports GiB, the eval records GiB, the validator compares GiB — but the fields said `_gb`, and the first write-up of the measurements duly mixed the two bases and reported a figure that was neither (`docs/MODEL_EVAL.md`). No profile has been adopted yet, so nothing in use breaks. Part of SPEC v1.4's §2 field list. **Recorded (M1), implemented.**

### 2026-09-23: thinking mode is switched off for serving, not just for the eval
With Qwen3.6's thinking mode on, the model spent its whole token budget reasoning and never reached an answer: the 2026-09-22 run scored 0.33 on part-number grounding purely because every answer was cut off mid-thought. With `chat_template_kwargs {"enable_thinking": false}`, the same weights score 6/6 and return a complete answer in 0.225 s (0.41 s at three concurrent sessions).

So this is a serving requirement, not an eval detail. FarmHub's own client has to send it, or the first real voice question produces a truncated ramble; §1 answers are read aloud. **To implement at M1:** `modules/llm` sends it, and which switch a model needs belongs in the profile beside `quantization` — a future model may spell it differently.

The eval harness also records whether a reply reasoned anyway, and warns, because the parameter is silently ignorable by a backend or a chat template, and a run that ignored it would otherwise look like a clean result. **Recorded (M1); the client change is still to do.**

### 2026-09-25: a spoken confirmation reads back the gateway's resolved arguments
A T2 confirmation prompt (§3.3) must state the arguments **the gateway resolved and stored**, not a paraphrase of what the person said. "Water the greenhouse benches for ten minutes — confirm?" is right; "shall I start the watering?" is not.

The reason is measured rather than hypothetical. In the 2026-09-23 eval, "Turn on the light in the barn" produced `set_indoor_light(area="workshop")`: `barn` is not in the enum, so instead of declining the model substituted a *different real area*. Schema-valid, and the wrong room. A prompt built from the utterance, or from the model's own sentence, would have read back "the light" and been confirmed happily while the stored argument pointed somewhere else. The whole value of §3.3's "the arguments executed are the arguments stored in the PendingAction" is lost if the human confirms against a different set of words than the ones that will run.

So the confirmation text is rendered from the PendingAction row — tool, area, duration, asset — by the gateway, never by the model, and it names each argument rather than summarising them. Where a value has no natural spoken form, the prompt says the value as stored rather than smoothing it.

**Recorded (M5/M8).** M5 builds the gateway's policy checks and is where the rendering belongs; M8 builds the confirmation flow and is where it is tested, including the case where the resolved area differs from the area the person named. Ties into Q6, which settles the voice form.

### 2026-09-22: the NVFP4 Marlin fallback on sm_120 is confirmed, not theoretical
SPEC §2 warned that NVFP4 on the RTX 5090 could fall back to Marlin W4A16. The first loading run of `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a` on `vllm/vllm-openai:v0.29.0` confirms it for both paths:

- linear layers: `Using MarlinNvFp4LinearKernel for NVFP4 GEMM`;
- MoE experts: `Using 'MARLIN' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', …, 'VLLM_CUTLASS', 'MARLIN', …]`.

The checkpoint also mixes precisions: vLLM detects both `NVFP4` and `W4A16_NVFP4` quantization algorithms, and some linear layers run FP8 (`FlashInferFP8ScaledMMLinearKernel`). So on this card and this vLLM version, NVFP4 is a *memory format*: the 4-bit memory saving holds, but compute is W4A16 on Marlin, not native FP4. The numbers in `docs/MODEL_EVAL.md` are for the Marlin path, and a later vLLM that dispatches a native FP4 backend on sm_120 needs a fresh run rather than inheriting them. **Recorded (M1).**

### 2026-09-20: the vLLM and SDK import boundaries are enforced twice
SPEC §2 says not to import vLLM outside `modules/llm/`. In practice nothing imports it at all, because it is a container rather than a library, and that is what keeps "switching to Ollama is a config change" true.

- `lint-imports` gains a `no module imports vllm` contract (which needed `include_external_packages = true`).
- `tests/safety/test_llm_boundary.py` asserts the same by parsing every source file, plus that only `modules/llm` imports the `openai` SDK. Anywhere else would mean a second way to talk to the model, outside the protocol the gateway and orchestrator are built around.

Both were verified to fail when a violation is introduced. The module-independence contract still waits for a second module. **In force (M1).**

### 2026-09-20: probing the model list, not just the port
`probe()` calls `/v1/models` and fails with `DependencyUnavailable` if the configured model is not among those served. A backend that answers but serves something else would otherwise silently invalidate the M1 evaluation, and the whole point of pinning a revision is knowing which weights answered. The connect timeout is separate from and much shorter than the request timeout, so a stopped vLLM degrades the module in seconds rather than after a minute. **In force (M1).**

### 2026-09-20: unparseable tool arguments become an empty mapping
A tool call whose `arguments` are not a JSON object yields `ToolCall(name, {})` rather than raising. Model output is untrusted text, and if malformed arguments raised inside the client, the turn would die before any audit row named the tool that was attempted. Instead the gateway sees a named call, denies it and audits it (§3.7). **In force (M1).**

### 2026-09-20: vLLM is a pinned container started by hand on dev, by systemd on hub
The inference backend is the official `vllm/vllm-openai` image, defined once in `deploy/vllm/compose.yaml` and used unchanged on both machines. It is a deployment artifact, not a Python dependency: nothing imports vLLM, so SPEC §2's "do not import vLLM outside `modules/llm/`" is satisfied by never importing it at all.

- **Image pinned to `v0.29.0`**, the first release at or above the 0.28.0 that the RTX 5090 NVFP4 recipe requires. Not `:latest`: the NVFP4 kernel path on sm_120 depends on the vLLM version, so an unpinned image would change which kernel is selected without anything in the repo changing.
- **Start and stop are manual on the dev PC.** Docker Desktop on WSL2 has no systemd socket activation and vLLM has no idle shutdown, so an automatic scheme would mean a bespoke sidecar. `deploy/vllm/vllm.sh` wraps `up | down | restart | wait | status | logs | config | resolve`, and `restart: "no"` in the compose file keeps the container from returning by itself after a Docker Desktop restart. On hub `deploy/systemd/farmhub-vllm.service` runs the same file always-on. An idle-stop sidecar polling `/metrics` is noted in the RUNBOOK as deferred, not built.
- **The model profile lives entirely in `deploy/vllm/.env`**, interpolated into the compose command, so changing profile never edits YAML and `./vllm.sh config` prints the exact command before anything runs. Required variables use compose's `:?` guard, so a missing revision or model fails loudly rather than starting something unintended. `env.example` is tracked under that name because `.gitignore` excludes `.env.*`.
- **`./vllm.sh resolve <repo>`** turns a repo name into the commit sha that SPEC §2 requires, so pinning is a command rather than a manual hunt. The five M1 candidates were resolved on 2026-09-20 and their shas recorded in `env.example`; all five repository IDs in SPEC §2 are confirmed to exist.
- **Neither backend is authenticated**, so both bind to loopback on every host. FarmHub is the only client and carries the §3.6 bearer token.
- **`deploy/ollama/compose.yaml`** exists so the §2 promise that switching to Ollama is a config change is actually testable. It is a fallback, not a peer.

**In force (M1).**

---

## Decided after M0 (2026-09-20)

### 2026-09-20: hub is a new single-GPU build; the 5090 is borrowed from the dev PC until it exists
**Superseded by SPEC v1.4 (2026-09-23)** — see "the hardware plan has two phases" above. The topology below is still right; what changed is that the second GPU is a committed RTX 3070 rather than an optional 8–12 GB card decided after M1, that the dev PC is the runtime host for about a year rather than a development machine, and that its post-swap card is undecided. Kept because it is why the 3070 was written out of the plan and then back into it, and because the `aux_reserve_gb` rules below are unchanged and still in force.

The hardware plan changed. hub is no longer the repurposed desktop: it is a new build that does not exist yet. There is one RTX 5090, currently in the dev PC (ClevatessPrime, Windows + WSL2); it moves to hub when hub is built, and the dev PC then takes an RTX 5080 16 GB. hub is specified with a free PCIe x16 slot and PSU headroom for an optional 8–12 GB auxiliary card, bought or not after M1 measures what the single-GPU profile leaves. The RTX 3070 is gone from the plan.

Consequences, all in §2 and §11 only — **no §3 rule is touched**:

- **The single-GPU profile is the default.** vLLM, faster-whisper, BGE-M3 and the reranker share the 5090. "The 5090 serves the LLM and nothing else" is kept, but scoped to the optional dual-GPU profile. The old rule was the only thing preventing an auxiliary model from starving the LLM, so it is replaced rather than deleted: every profile declares `aux_reserve_gb`, config validation rejects a profile whose `weights_gb + kv_cache_gb + aux_reserve_gb` exceeds the card or whose `gpu_memory_utilization` does not leave the reservation free, and vLLM starts before the auxiliary models so its fraction is computed against a known-free card.
- **Budget numbers are measured, not estimated.** `weights_gb` and `kv_cache_gb` come from an M1 evaluation run on the real card and carry the run id that produced them. A profile with hand-written numbers is rejected. Note the limit honestly: this is arithmetic over declared numbers, not a live VRAM probe (NVML is not on the §12 list and would be hub-only). The real OOM guard stays vLLM's preflight plus the measured numbers.
- **Every profile pins a `revision` (commit sha).** A tag is not a pin: one NVFP4 upload of Qwen3.6-35B-A3B was silently replaced on 2026-07-10 with looping weights. The eval harness refuses an unpinned profile and verifies the resolved sha.
- **`kv_cache_dtype` is a declared, evaluated field.** FP8 KV cache has reported quality collapse on this hybrid architecture, so M1 measures each candidate at the default dtype and at FP8, on quality as well as memory.
- **Model candidates.** Repository IDs verified 2026-09-20. There is no vendor NVFP4 build of `Qwen3-30B-A3B-Instruct-2507` — `nvidia/Qwen3-30B-A3B-NVFP4` and `RedHatAI/Qwen3-30B-A3B-NVFP4` quantize the older *Base* model and the NVIDIA one targets TensorRT-LLM. On sm_120 (consumer Blackwell) vLLM's NVFP4 MoE dispatch has open bugs and ModelOpt checkpoints fall back to Marlin W4A16 with a "no native FP4 support" warning, so a run can look like NVFP4 without being it and the eval records the kernel actually selected. `Qwen/Qwen3.6-35B-A3B` (`nvidia/Qwen3.6-35B-A3B-NVFP4`) is therefore added as candidate A: same 35B/3B-active MoE shape, and the only candidate with an official vLLM recipe validated on the RTX 5090. It is multimodal, so FarmHub runs it text-only — §1 has no multimodal path and the vision tower's memory is memory the KV cache needs.
- After the swap the dev PC uses vLLM on hub over the LAN, with a small-model 5080 profile as the offline fallback. Config, not code.

Accepted as SPEC v1.3 (2026-09-20). The model itself is chosen at M1 from `docs/MODEL_EVAL.md`. **Accepted (SPEC v1.3)**; the profile fields and their validation are **Recorded (M1)**.

### 2026-09-20: Tool declarations load before dependencies connect (registry contract change at M5)
The `ha` module loads and validates its tool TOML files (config: fail fast on any error) before it connects to HA (dependency: degradable). Its tools, including T3 declarations, are therefore registered even when HA is down, and the gateway audits a call to any of them with the correct tier and the reason "module not running", never as an unknown tool. This resolves the M0 known limitation.

The registry contract changes when M5 lands. Proposed shape: split the `Module` lifecycle into a config phase and a dependency phase, for example `configure(ctx)` (validates config and produces the tool list; no dependency I/O; raises `ConfigError`; run for every module before any `startup`) and `startup(ctx)` (connects; `DependencyUnavailable` starts the module degraded). The registry registers a module's tools after `configure` and marks them unavailable until `startup` succeeds. The exact signature settles at M5 and SPEC §6 is updated then; the M0 contract is unchanged until then. Modules with static tools benefit the same way. **Recorded (M5).**
