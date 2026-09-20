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
