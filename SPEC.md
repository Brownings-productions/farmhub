# FarmHub — Build Specification

A local-first AI hub for a Norwegian homestead: RAG over farm documents, voice satellites, Home Assistant control, web lookups, and parametric CAD generation. Single Python spine, pluggable modules.

Version 1.1. Everything here was decided deliberately. Where a decision looks odd, §3 or the rationale notes explain why.

---

## Changes in v1.1

| # | Section | Change |
|---|---------|--------|
| 1 | §3.5, §4, §9 | Closed an injection path: action turns no longer receive any tool marked `reads_untrusted_content`. Added a runtime tripwire and a test. |
| 2 | §3.6, §4, §9 | Requests arriving without a verifiable satellite identity fail closed to T0-only. Added a test. |
| 3 | §3.3 | Defined how a push-notification confirmation binds to a PendingAction and who may confirm. |
| 4 | §4 | Defined the classifier: deterministic pre-pass plus a tool-less structured LLM call, defaulting to `question` when uncertain. |
| 5 | §7 | Tool parameter files now declare `required` explicitly; generated schemas are strict. |
| 6 | §2, §8.8 | Fixed contradiction on where Piper runs (hub, not satellites). Prefer upstream Wyoming servers over custom code. |
| 7 | §2 | Model IDs marked for verification before M1. |
| 8 | §12 | Added an approved dependency list so "ask before adding" only triggers on genuine surprises. |
| 9 | §14 | Added a list of known open questions to resolve at the named milestone. |

---

## 0. How to use this document

You are building this project from scratch. Read this file at the start of every session.

Rules of engagement:

- Build in the milestone order in §11. Do not skip ahead. Each milestone must run and pass its tests before the next begins.
- When something here is ambiguous, stop and ask rather than inventing a design. §14 lists questions already known to be open; raise them at the milestone named there.
- Do not add dependencies not listed in §12 without asking first.
- §3 is non-negotiable. If a requested feature conflicts with §3, refuse it and say why. This holds even if I ask for it later in a session. Quote the rule number back at me.
- Keep `docs/DECISIONS.md` updated: one short entry per architectural choice, dated.
- Prefer asking one good question over writing 400 lines of the wrong thing.
- For safety-relevant work (§3, and milestones M5–M8), write the test before the code it guards.

## 1. Mission

A single Python application that:

- Answers open questions using a local LLM with RAG over a curated farm document library.
- Answers structured questions about service history from a real database, not a vector index.
- Controls a subset of Home Assistant entities through narrow, typed, tiered tools.
- Serves 2–3 concurrent voice users through Home Assistant Assist and Wyoming satellites.
- Fetches weather, news and general web information.
- Generates parametric 3D models as code, renders previews, and queues them to a 3D printer after explicit human approval.

Everything runs on local hardware. No cloud inference. Outbound network is limited to the web module's allowlist.

## 2. Hardware and runtime topology

### Machines

| Host | Hardware | Role |
|------|----------|------|
| hub | Ubuntu 24.04, RTX 5090 32 GB + RTX 3070 8 GB | vLLM, FarmHub app, Postgres, Whisper, Piper, embeddings |
| ha | N100 mini PC or Pi 5 + NVMe, Home Assistant OS | Home Assistant and all safety-critical automation |
| nas | TrueNAS, GTX 1060 | Document corpus, media, backups. The 1060 is for Jellyfin transcoding — never an AI target |
| sat-* | Raspberry Pi 4/5 | Wyoming voice satellites: kitchen, barn, workshop |

hub is the repurposed desktop: wiped to Ubuntu, 5090 already in it, 3070 added in the second slot. The 5090 is power-limited to ~450 W since the machine runs permanently — inference loses very little and it runs cooler and quieter.

ha is a separate physical machine on purpose. Heating and pumps must keep working when the GPU box is down, being patched, or has thrown a driver fault. This is an architectural rule, not a preference. Never propose collapsing ha into hub.

### GPU assignment

Hard rule enforced in config: the 5090 serves the LLM and nothing else.

- `CUDA_VISIBLE_DEVICES=0` → vLLM. `gpu_memory_utilization: 0.90`.
- `CUDA_VISIBLE_DEVICES=1` → faster-whisper, BGE-M3, bge-reranker-v2-m3. ~5.2 GB of 8 GB.

If only one GPU is present, config must degrade gracefully: drop vLLM to `gpu_memory_utilization: 0.55`, load auxiliary models onto the same device, log a warning at startup. Never silently OOM.

### Models

| Role | Model | Notes |
|------|-------|-------|
| LLM | Qwen3-30B-A3B-Instruct, FP8 | MoE, ~3B active. Chosen for tool calling. ~30 GB fits the 5090 |
| LLM fallback | Mistral-Small-3.2-24B-Instruct, AWQ | Config switch only |
| STT | faster-whisper large-v3, `int8_float16` | CTranslate2. This compute type requires CUDA |
| TTS | Piper | CPU, on hub, served over Wyoming. Satellites only play audio |
| Embeddings | BAAI/bge-m3 | Dense + sparse from one model |
| Reranker | BAAI/bge-reranker-v2-m3 | Top-30 → top-5 |

**Before M1:** verify the exact Hugging Face repository IDs for the LLM and fallback (current releases carry date suffixes, and FP8/AWQ variants may be separate repos), and record them in `config/farmhub.example.toml` and `docs/DECISIONS.md`. Model IDs live only in config, never in code.

FP8 rather than 4-bit because the 5090 has the VRAM for it and quantization damage shows up first on exactly the things this system does: part numbers, torque figures, tool-call argument fidelity. If VRAM becomes tight, try NVFP4 before falling back to AWQ.

### Inference backend

Primary: vLLM, OpenAI-compatible server, automatic prefix caching enabled. The system prompt and tool schemas are near-constant, so prefix caching is the main latency win.

The application talks to the LLM only through an `LLMBackend` protocol implemented by an OpenAI-compatible client. Switching to Ollama must be a config change, never a code change. Do not import vLLM anywhere outside `modules/llm/`.

## 3. Non-negotiable safety rules

These are acceptance criteria. Write tests in `tests/safety/` that fail if any is violated.

### 3.1 No generic actuation primitive

There must be no tool of the form `call_service(domain, service, entity_id, data)` or any equivalent that lets the model reach arbitrary Home Assistant entities. Every actuation tool is a named, typed, individually-declared tool backed by one specific HA script.

### 3.2 Capability tiers

Every tool declares a tier. The bridge enforces tiers server-side. Model output is never trusted to respect them.

- **T0 READ** — see §8.1 for the full list. Auto-execute, always available (subject to §3.5 and §3.6).
- **T1 COMFORT** — indoor lights, scenes, media, notifications, logging a service record. Auto-execute. Logged.
- **T2 CONFIRMED** — greenhouse watering, garden and field irrigation, ventilation, CAD generation and slicing. The model may propose; execution requires explicit human confirmation per §3.3.
- **T3 FORBIDDEN** — heating, well and pressure pumps, anything affecting livestock, mains electrical, starting a 3D print. Never exposed to the LLM in any form. Read-only status is permitted (asking whether the barn heating is on is fine). Control lives exclusively in deterministic Home Assistant automations and fixed HA intents, so it still works by voice — just not through the model, and not when hub is down.

Rationale to preserve: heating protects pipes from freezing, and a pump can flood a building or drain the well that also supplies the house. Neither may depend on a language model's judgement.

### 3.3 Confirmation flow for T2

A T2 tool call returns a `PendingAction` with a UUID and a TTL of 120 seconds. The PendingAction records the originating satellite, session, scope, tool, and exact arguments. It executes only after a matching confirmation arrives through one of two channels. Expired actions are discarded.

**Satellite confirmation.** A structured confirm callback carrying the UUID, arriving from the same satellite session that created the action.

**Push-notification confirmation.** FarmHub sends a Home Assistant actionable notification whose action identifier embeds the UUID. The confirmation arrives as the HA `mobile_app_notification_action` event over the WebSocket connection. It is accepted only if:

- the UUID matches a live, unexpired PendingAction;
- the originating device is in the `confirmers` allowlist configured for the action's scope (a list of HA mobile-app device IDs); and
- the action has not already been confirmed, denied, or expired (each PendingAction resolves exactly once).

The audit row for the execution records which channel confirmed it and, for push, which device.

Confirmations are never inferred from conversational text such as "yeah go ahead" — require a structured confirm callback carrying the UUID. The arguments executed are the arguments stored in the PendingAction; a confirmation cannot modify them.

### 3.4 Runtime watchdogs live in Home Assistant

Every irrigation, watering and ventilation script has a maximum runtime enforced inside the Home Assistant script itself, not in Python. A hung hub, a crashed process or a bug in this repo must not be able to leave a valve open. Assume the Python side will fail eventually.

### 3.5 Untrusted content and actuation tools are never in the same context

Document text from the corpus and content fetched from the web are untrusted input. This includes the results of any tool marked `reads_untrusted_content=True` (`search_documents`, `get_news`, `fetch_url`), not only content retrieved by the pipeline before the completion.

Three rules enforce this:

1. **Pipeline retrieval.** A completion whose context includes retrieved content must be issued with `tools=[]` or only T0 tools.
2. **Action turns exclude untrusted tools.** The tool list for an `action` turn contains only T0 tools with `reads_untrusted_content=False`, plus scope-permitted T1 and T2 tools. The model therefore cannot fetch untrusted content mid-turn and then act on it.
3. **Runtime tripwire.** The orchestrator tracks a per-turn `context_tainted` flag, set whenever untrusted content enters the context by any route. Any completion issued while the flag is set is restricted to T0 tools, and any attempt to execute a T1+ tool while it is set is denied and audited. This is defence in depth: rule 2 should make it unreachable.

This blocks prompt injection from a scanned PDF or a web page into the actuation path. Enforce all three with explicit assertions in the orchestrator, and test them.

### 3.6 Satellite scope is server-side

Satellite identity comes from the Wyoming connection or the HA `device_id`, and the bridge derives the allowed tool set from it. The model never supplies its own location, area or scope as a tool argument. The workshop satellite cannot actuate greenhouse valves.

**Fail closed.** A request that arrives without a satellite identity, or with one not present in the satellite registry, is served with T0 tools only (trusted and untrusted T0, per §3.5), and the condition is logged as a warning. Identity is never taken from message content.

### 3.7 Audit log

Every tool invocation writes an immutable row: timestamp, satellite, session, tool name, tier, arguments, decision (executed / denied / pending / expired), result, duration, and for T2 the confirmation channel and confirming device. Append-only table plus a JSONL file. No tool call may execute without producing an audit row.

### 3.8 Dry run

A global `--dry-run` flag and `FARMHUB_DRY_RUN` env var. In dry-run mode, every tool at T1 and above logs its intended call and returns a simulated success. No HTTP or WebSocket service call reaches Home Assistant. Default this on until milestone M7.

### 3.9 3D printing

Slicing and upload to the printer queue are permitted. Starting a print is not. Unattended ignition risk, and the bed may not be clear. The pipeline ends at "file uploaded, notification sent." There is no tool, flag or config option that starts a print.

### 3.10 Sandboxing generated code

The cad module executes LLM-written CadQuery/OpenSCAD source. It runs in a subprocess with no network, a tmpfs working directory, a 30-second timeout and a memory cap. A container is better if available. Never `exec()` generated code in the main process.

### 3.11 Rate limits

Per-tier, per-session limits enforced in the bridge. T2 defaults to 3 proposals per 10 minutes. Denials produce audit rows.

## 4. Request flow

```
Satellite → HA Assist
  │
  ├── HA intent matches a known command → HA handles it. Deterministic. Never reaches FarmHub.
  │
  └── No match → HA conversation agent → FarmHub /v1/chat/completions
        │         (must carry satellite identity; absent → T0-only, §3.6)
        │
        ├── classify: question | action | mixed   (see "Classifier" below)
        │
        ├── question → retrieve → completion with T0 tools only (§3.5) → answer
        │
        ├── action   → completion with trusted T0 + scope-filtered T1/T2 tools,
        │               NO retrieved content, NO untrusted tools (§3.5 rule 2)
        │                ├── T1 → execute → audit → respond
        │                └── T2 → PendingAction → ask → await confirm → execute → audit
        │
        └── mixed    → answer first, then offer the action as a separate turn. Never fuse them.
```

Home Assistant keeps handling known commands itself. FarmHub is the fallback for open questions, not a replacement for HA's intent system.

### Classifier

`orchestrator/router.py` implements two tiers:

1. **Deterministic pre-pass.** Cheap rules over the normalized utterance: if no configured action verb or tool-name alias appears, classify as `question` without calling the LLM.
2. **LLM classification.** Otherwise, a separate completion issued with `tools=[]` and no retrieved content, constrained to return JSON `{"kind": "question" | "action" | "mixed"}` via structured output.

If the JSON fails to parse, fails validation, or the call times out, the result is `question`. The safe default is always the path with no actuation tools. The classifier's output only selects a pipeline; it never grants tools directly — tool lists are always derived from tier, scope, and §3.5.

## 5. Repository layout

```
farmhub/
  pyproject.toml
  SPEC.md
  CLAUDE.md
  docs/
    DECISIONS.md
    RUNBOOK.md
  config/
    farmhub.example.toml
    satellites.example.toml     # satellite registry: name, area, scope, tier ceiling, input mode
    tools/                      # one TOML file per exposed tool
  src/farmhub/
    __init__.py
    __main__.py
    core/
      config.py                 # pydantic-settings, layered TOML + env
      context.py                # AppContext: shared clients, no globals
      registry.py               # module discovery and lifecycle
      protocols.py              # all Protocol definitions
      events.py                 # async pub/sub bus
      audit.py                  # append-only audit sink
      errors.py
      logging.py                # structlog, JSON to file, pretty to tty
    api/
      server.py                 # FastAPI
      openai_compat.py          # /v1/chat/completions for the HA conversation agent
      routes_admin.py           # health, reindex, pending actions, audit tail
    orchestrator/
      router.py                 # two-tier classification, §4
      pipeline.py               # turn construction; enforces §3.5
      prompts/
    modules/
      llm/  rag/  ingest/  records/  ha/  web/  cad/  voice/
    storage/
      db.py                     # SQLAlchemy async engine
      models.py
      migrations/               # alembic
  tests/
    unit/  integration/
    safety/                     # §3 enforcement — never skipped
  deploy/
    docker-compose.yml
    systemd/
```

Each module directory contains `__init__.py`, `module.py` (the Module implementation), `tools.py`, and module-local logic. Modules never import each other. Cross-module needs go through `AppContext` or the event bus. Enforce with an import-linter rule in CI.

## 6. Core contracts

Define these in `core/protocols.py` before writing any module.

```python
class Module(Protocol):
    name: str
    version: str

    async def startup(self, ctx: AppContext) -> None: ...
    async def shutdown(self) -> None: ...
    def tools(self) -> list[ToolSpec]: ...
    async def health(self) -> HealthReport: ...
```

```python
class Tier(IntEnum):
    READ = 0
    COMFORT = 1
    CONFIRMED = 2
    FORBIDDEN = 3        # never returned to the model; exists so config can express it


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict                  # JSON Schema, strict, no additionalProperties
    tier: Tier
    handler: Callable[[ToolCall, ToolContext], Awaitable[ToolResult]]
    allowed_scopes: frozenset[str]    # satellite areas, or {"*"}
    timeout_s: float = 10.0
    reads_untrusted_content: bool = False
```

Also define: `LLMBackend`, `Retriever`, `Embedder`, `Reranker`, `DocumentParser`, `Chunker`, `ToolContext` (carries satellite, session, scope, dry_run, context_tainted), `ToolResult` (carries an `untrusted: bool` flag the pipeline uses to set `context_tainted`), `PendingAction`, `HealthReport`.

Modules are registered in an explicit list in `registry.py` — no entry-point scanning. Explicit is easier to reason about when something fails to load at 06:00.

## 7. Configuration

Layered: defaults in code → `config/farmhub.toml` → environment (`FARMHUB_*`) → CLI flags. Validated by pydantic-settings at startup. Fail loudly and exit on invalid config. No silent defaults for anything safety-relevant.

Tool exposure is config-driven, one TOML file per tool:

```toml
name = "start_greenhouse_watering"
description = "Start watering in the greenhouse for a set number of minutes."
tier = "CONFIRMED"
ha_script = "script.greenhouse_water_timed"
allowed_scopes = ["kitchen", "greenhouse"]
required = ["duration_min"]

[parameters.duration_min]
type = "integer"
minimum = 1
maximum = 20
description = "Minutes to run. Hard-capped by the Home Assistant script."
```

Schema generation rules:

- The generated JSON Schema always sets `additionalProperties: false`.
- `required` must be present and list parameter names explicitly; an empty list is allowed but must be written. A missing `required` key fails validation.
- Every parameter must declare `type` and `description`. Numeric parameters must declare `minimum` and `maximum`. String parameters must declare either `enum` or `maxLength`.

Adding a tool must never require touching Python. A tool file referencing a nonexistent HA script fails validation at startup, not at first use. A tool file declaring `tier = "FORBIDDEN"` is loaded (so config can express it) but is never placed in any model-visible list.

## 8. Module specifications

### 8.1 ha — Home Assistant bridge

Carries most of §3. Connects over the HA WebSocket API using a long-lived token scoped to a dedicated non-admin HA user.

Responsibilities: load tool definitions from `config/tools/`, validate each against the live HA entity and script registry at startup, enforce tier/scope/rate-limit/timeout/dry-run/audit before any service call, manage PendingAction lifecycle including push-notification confirmations (§3.3), filter T3 out of every tool list returned to the model, surface real HA results rather than letting the model narrate success.

T0 tool list (read-only, always available subject to §3.5):

| Tool | Returns | Untrusted |
|------|---------|-----------|
| get_entity_state | Current value of one allowlisted entity | no |
| get_sensor_history | Time series for a sensor, bounded window | no |
| list_area_status | Summary of one area's sensors in a single call | no |
| search_documents | Chunks with source path, page, heading path | yes |
| query_service_history | Filtered service events for an asset | no |
| next_service_due | Computed due date or hours | no |
| list_assets | Known machines and buildings | no |
| get_weather | Yr / met.no forecast | no |
| get_news | Headlines from configured RSS feeds | yes |
| fetch_url | Readable text from an allowlisted domain | yes |

`get_weather` is marked trusted because it returns structured numeric data from a single fixed API, parsed into typed fields; no free text from the response reaches the model.

`get_entity_state` runs against an allowlist, not the whole entity registry. Cameras, presence and device trackers stay out.

`list_area_status` exists so "how's the greenhouse doing" is one call, not five round trips with a decent chance the model forgets one.

### 8.2 llm

Thin OpenAI-compatible client. Streaming, retry with backoff, tool-call parsing, structured output (JSON schema) for the classifier, token accounting into structured logs. Exposes no tools. Configurable base URL and model so vLLM, Ollama and a test double are interchangeable.

### 8.3 rag

Store: Postgres + pgvector (HNSW) + tsvector for lexical. One database for vectors, metadata, full text and service records.

Retrieval: hybrid — dense (BGE-M3) and sparse/BM25 in parallel → Reciprocal Rank Fusion → rerank top 30 → return top 5. Metadata filters (asset, doctype, domain) applied before retrieval, not after.

Dense-only retrieval fails on part numbers, error codes and model designations — a query for SKF 6205-2RS needs lexical matching. Hybrid is not optional here.

### 8.4 ingest

**Library taxonomy.** The corpus path carries metadata:

```
/corpus/{domain}/{asset}/{doctype}/{filename}.pdf

domain   = machinery | buildings | land | livestock | household | admin
asset    = slug, e.g. kubota-l4240, barn, greenhouse-1, well-pump
doctype  = manual | spec | schematic | invoice | certificate | correspondence | photo
```

Non-conforming paths go to `/corpus/_inbox/` and are reported, never guessed at. Provide a `farmhub library check` CLI command that lints the tree.

**Pipeline**, idempotent and content-hash addressed:

```
watch/scan → sha256 → manifest lookup → skip if (hash, parser_ver, chunker_ver) unchanged
  → text-layer coverage check
  → digital: PyMuPDF  |  scanned: OCRmyPDF (eng) or docling
  → normalize to markdown + structure tree
  → chunk → contextualize → embed → upsert → manifest row
```

**Chunking rules** — the quality-critical part:

- Split on document structure, then token-bound to 400–700 tokens with 10–15% overlap. Count tokens with the BGE-M3 tokenizer.
- Never separate a table from its header. Emit whole tables as markdown, one chunk each.
- Prepend the heading path to every chunk: `Kubota L4240 › Maintenance › Fluid capacities`.
- Carry metadata: asset, doctype, domain, manufacturer, model, page, source path, hash.
- Contextual retrieval: generate a one-sentence situating blurb per chunk with the local LLM and prepend it before embedding. Batch overnight via a systemd timer; the GPU is idle.

Store `parser_version` and `chunker_version` in the manifest so a strategy change reindexes only what's affected. Resumable — the first full run over scanned manuals takes hours.

No tools. Ingestion is CLI and scheduler only, never triggered by a conversation.

### 8.5 records

Service history is structured data, not RAG. "When did I last change the hydraulic oil" must be a SQL answer, not a similarity search.

Tables: `asset`, `service_event` (date, hours, type, parts, cost, notes, doc_ref), `consumable`, `reminder`.

T0 tools as listed in §8.1. One T1 tool: `log_service_event(...)` — recording maintenance by voice while your hands are dirty is the single highest-value write in the system.

### 8.6 web

All T0. `get_news` and `fetch_url` are marked `reads_untrusted_content=True`; see §8.1 for why `get_weather` is not.

- `get_weather` — met.no / Yr API. Free, accurate for Norway, requires a proper User-Agent and respects caching headers. Honour both.
- `get_news` — configured RSS feeds only. No open-ended crawling.
- `fetch_url` — domain allowlist in config. Strip scripts, extract readable text, cap at 20 000 characters, never follow redirects off the allowlist.

Cache aggressively to disk. Rate-limit outbound requests.

### 8.7 cad

Parametric CAD generation. Code, not meshes. Text-to-mesh models produce organic blobs with bad topology that don't print as functional parts.

```
request → LLM generates CadQuery (preferred) or OpenSCAD source
        → execute sandboxed per §3.10
        → export STL + render 3 preview PNGs (iso, front, top)
        → present previews and source to the human
        → on approval: slice via PrusaSlicer/OrcaSlicer CLI with a named profile
        → upload G-code to the printer queue
        → notify. STOP. (§3.9)
```

Keep generated source in `/cad/{slug}/{version}/` with the parameters used, so the bracket can be regenerated 3 mm wider next year. That reusability is the whole reason for choosing code over mesh generation.

Tools: `generate_model(description, constraints)` at T2 (it runs generated code), and `slice_and_queue(model_id, profile)` at T2. No start-print tool exists.

### 8.8 voice

Not a satellite implementation — satellites run `wyoming-satellite` and are configured, not coded. Satellites capture audio and play TTS audio; they do not run Piper or Whisper themselves.

This module is responsible for Whisper STT and Piper TTS as Wyoming services on hub, plus the satellite registry (`config/satellites.toml`): name → area → scope → tier ceiling → input mode → confirmers.

Prefer running the upstream `wyoming-faster-whisper` and `wyoming-piper` servers, configured and supervised by this module's deployment files, over reimplementing them. Write custom code only where they fall short, and record why in `docs/DECISIONS.md`.

Workshop satellite: `wake_word: none`, `ptt: gpio`, close-talk headset mic. Far-field arrays do not work against machine noise regardless of push-to-talk.

## 9. Testing

`tests/safety/` is mandatory and part of the default test run:

- Model-visible tool list contains no T3 tool or entity, under every scope.
- A turn containing retrieved content is issued with no tool above T0.
- The tool list for an `action` turn contains no tool with `reads_untrusted_content=True`.
- After any untrusted tool result enters a turn, a subsequent T1+ tool call in that turn is denied and audited, and no HA service call is made.
- A classifier failure (bad JSON, timeout) routes to `question`.
- A request with no satellite identity, or an unknown one, receives only T0 tools.
- A T2 tool call without confirmation produces no service call to HA.
- A PendingAction past its TTL is rejected.
- A confirmation from a different session is rejected.
- A push confirmation from a device not in the scope's `confirmers` list is rejected.
- A PendingAction cannot be confirmed twice.
- Scope filtering: the workshop scope cannot reach greenhouse tools.
- Every executed tool call produced exactly one audit row.
- Dry-run mode issues zero outbound HA service calls.
- A tool config referencing a missing HA script fails startup.
- A tool config missing `required`, or with unbounded numeric or string parameters, fails startup.
- cad generated code cannot open a network socket.

Integration tests run against a mocked HA WebSocket server and a test-double LLM. Tests needing Postgres use a pgvector-enabled container via testcontainers. No test may require a real GPU — the whole suite must pass on a laptop with Docker.

## 10. Electrical — NEK 400

Not software, but it belongs in the plan. In Norway, fixed installation work requires a registered electrical contractor (registrert elvirksomhet).

- The barn is currently fed from the house board; a grid upgrade is planned. The barn sub-board and its feed are contractor work.
- NEK 400-705 (agricultural premises) applies to the barn: 30 mA RCD, fire and dust requirements, equipotential bonding if livestock are present.
- Outdoor irrigation and pump circuits: 30 mA RCD mandatory, IP-rated enclosures, buried-cable requirements for field runs.
- Well/pressure pump wiring and motor protection: contractor work regardless of upgrade timing.
- Dedicated 16 A course for the rack (hub ~800 W under load, plus NAS, switches, UPS).
- Confirm 230 V IT vs 400 V TN before buying a UPS — some line-interactive units assume a bonded neutral and misbehave on IT.
- Specify to the contractor: every irrigation valve and pump circuit needs an independent hardware means of cutting it that does not route through Home Assistant.
- PoE satellites and the fiber run are not contractor work.

## 11. Build order

Each milestone ends with something runnable and tested. Do not begin the next until the current is green.

| # | Milestone | Done when |
|---|-----------|-----------|
| M0 | Scaffold: pyproject, config, logging, AppContext, registry, protocols, CI | `farmhub --version` runs, an empty module loads, CI green |
| M1 | llm + FastAPI + /v1/chat/completions | HA conversation agent gets an answer from vLLM. Model IDs verified (§2). Satellite identity reaches FarmHub (§14 Q1) |
| M2 | Storage, migrations, records | Service events insert and query by CLI |
| M3 | ingest: parse, chunk, embed, manifest | Corpus indexes; rerun is a no-op; library check lints |
| M4 | rag: hybrid retrieval + rerank + search_documents | Cited answers from manuals, with pages |
| M5 | ha bridge, T0 read-only, audit log | Reports sensor states. Zero write paths exist yet |
| M6 | Tiers, scopes, rate limits, dry-run, classifier, taint tracking, safety suite | All of §9 passes. Still dry-run by default |
| M7 | T1 comfort actions live | Lights work by voice. Audit rows correct |
| M8 | T2 + PendingAction confirmation flow (satellite and push) | Greenhouse watering works with confirm; TTL expiry and confirmer allowlist tested |
| M9 | voice: Whisper + Piper services, satellite registry | Kitchen satellite end-to-end |
| M10 | Workshop PTT satellite | GPIO trigger, no wake word, headset mic |
| M11 | web: weather, news, allowlisted fetch | Yr forecast by voice |
| M12 | cad: generate, sandbox, preview, approve, slice, queue | Bracket reaches printer queue. Does not start |

M5 through M8 are where haste causes real damage. Slow down there. Write each §9 test before the code it guards.

Note: nothing before M9 requires dedicated hardware. M0–M8 can be developed against WSL2, Ollama, or a mocked LLM on any machine.

## 12. Standards

- Python 3.12. uv for dependency management. ruff format + lint. mypy --strict on `core/` and all protocol implementations.
- Async throughout: asyncio, httpx, SQLAlchemy 2.x async, asyncpg.
- FastAPI + Pydantic v2. structlog. pytest + pytest-asyncio. alembic.
- No globals. Dependencies flow through AppContext.
- No bare except. No silent failure. Every error path logs with context.
- Type hints everywhere. Protocols over ABCs.
- Docstrings on every public function, explaining why where it isn't obvious.
- Conventional commits, one milestone per branch.

### Approved dependencies

Anything on this list may be added when its milestone needs it. Ask before adding anything else.

| Area | Packages |
|------|----------|
| Core / API | pydantic, pydantic-settings, fastapi, uvicorn, httpx, structlog, typer (CLI) |
| Storage | sqlalchemy[asyncio], asyncpg, alembic, pgvector |
| LLM | openai (as the OpenAI-compatible client only) |
| HA | websockets |
| Ingest | pymupdf, ocrmypdf, docling, watchfiles |
| RAG | FlagEmbedding (BGE-M3 dense+sparse and reranker), tokenizers |
| Voice | faster-whisper, wyoming; upstream wyoming-faster-whisper and wyoming-piper as services |
| Web | feedparser, trafilatura, hishel (HTTP caching for httpx) |
| CAD | cadquery; OpenSCAD and PrusaSlicer/OrcaSlicer as external binaries |
| Dev / test | pytest, pytest-asyncio, respx, testcontainers[postgres], ruff, mypy, import-linter |

Scheduling uses systemd timers (in `deploy/systemd/`), not a Python scheduler library.

## 13. Out of scope

Do not build: a custom web UI (Home Assistant is the UI), user accounts or auth beyond the HA token, cloud fallback inference, text-to-mesh 3D generation, custom satellite firmware, or anything that starts a 3D print.

## 14. Open questions

Known unresolved decisions. Raise each one at the milestone named, propose options, and wait for an answer. Record the outcome in `docs/DECISIONS.md` and remove it from this list.

- **Q1 (M1): How satellite identity reaches FarmHub.** The standard OpenAI chat-completions request has no field for the HA `device_id`. Decide which HA conversation integration calls FarmHub and how it passes identity (a header, the `user` field, or a small custom integration). Until resolved, §3.6 fail-closed applies and every request is T0-only.
- **Q2 (M6): Action-verb and alias list for the classifier pre-pass.** Proposed source: derived automatically from tool names and descriptions in `config/tools/`, plus a hand-maintained Norwegian and English verb list.
- **Q3 (M6): Rate limits for T0 and T1.** Only T2 has a default. Propose values.
- **Q4 (M12): Preview rendering.** How the three PNG previews are rendered inside the sandbox (OpenSCAD's renderer, a headless mesh renderer, or CadQuery SVG export converted to PNG). Must not add a network-capable dependency to the sandbox.
- **Q5 (M12): Sandbox mechanism.** Plain subprocess with resource limits versus a container, given what is available on hub.
