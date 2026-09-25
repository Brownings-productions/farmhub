# FarmHub — Build Specification

A local-first AI hub for a Norwegian homestead: RAG over farm documents, voice satellites, Home Assistant control, web lookups, and parametric CAD generation. Single Python spine, pluggable modules.

Version 1.4. Everything here was decided deliberately. Where a decision looks odd, §3 or the rationale notes explain why.

---

## Changes in v1.4

Folds in the hardware plan and the storage split of 2026-09-23 (see `docs/DECISIONS.md`). **Amends §2 and §11, and adds one question to §14's open list**; no §3 rule is touched, so like v1.3 it needs no safety acceptance. Rows 2–5 supersede rows 1, 2 and 7 of v1.3, which are left in place as history.

**Status: accepted, 2026-09-25.** Accepted as a whole, with one correction folded in: the memory resident on the card beside vLLM is **not** a desktop compositor. The monitor runs on the motherboard's integrated graphics and the 5090 reads 0 MiB whenever vLLM is down, so that memory is vLLM's own overhead outside its profiled figures (`docs/MODEL_EVAL.md`). Gaming still competes for the card — §14 Q9.

| # | Section | Change |
|---|---------|--------|
| 1 | §2 Machines | The `nas` row was out of date. `nas` is now a **new TrueNAS build** for the corpus, media and backups, on a Xeon E5-1650 v4 — which has **no integrated graphics**, so it starts with no GPU at all and an Intel Arc A310 may be added later for transcoding. The old TrueNAS box becomes `tv`, a TV and emulation machine keeping the GTX 1060 until that card is upgraded; it is **not part of the FarmHub system**. Both rows keep "never an AI target". |
| 2 | §2 Machines, §2 GPU assignment | **The hardware plan has two phases.** Phase 1 (now to ~Sept 2027) runs FarmHub on the **dev PC's RTX 5090**, sharing that card between the LLM and the helper models. Phase 2 is `hub` with an **RTX 5090 + RTX 3070 8 GB**. The 3070 is no longer "gone" (v1.3 row 1) and the second GPU is no longer "optional, decided after M1" (v1.3 row 2). |
| 3 | §2 Machines | The `dev` row is no longer "not part of the running system": in Phase 1 it **is** the running system. An RTX 3070 cannot be added to it — the PSU will not take it — which is why Phase 1 shares one card. |
| 4 | §2 GPU assignment | **A profile belongs to one machine.** `weights_gib` carries from the dev PC to `hub`; `kv_cache_gib` does not, and no dual-GPU profile is adopted until an eval run on `hub` produces it. Replaces v1.3's "everything it measures is about the card, not the chassis". Every memory field is renamed to the unit it was always in — `weights_gib`, `kv_cache_gib`, `aux_reserve_gib`, `card_total_gib` — and the list gains `chat_template_kwargs` and `temperature`, both required. |
| 5 | §2 Machines, §2 Inference backend | The dev PC's card after the swap is an **RTX 5080 16 GB or a used RTX 4090 24 GB, undecided** (v1.3 row 7 named the 5080). Its role is the offline fallback; normally it uses vLLM on `hub` over the LAN. |
| 6 | §2 Machines | States the rule plainly: **model inference runs on one machine at a time**, and `nas`, `tv` and `ha` never run it. |
| 7 | §11 | Phase 1 means M1–M12 are all built and measured on the dev PC's 5090, so the shared card constrains M3, M4 and M9 as much as M1. The "nothing before M9 requires dedicated hardware" note is rewritten; §9's "no test may require a GPU" is unchanged. |
| 8 | §2 GPU assignment | Drops the claim that overnight batch work is where the LLM can be stopped to make room. §8.4's contextual retrieval **needs** the local LLM to write its blurbs, so it is another tenant on the shared card, and taking serving offline overnight is not a decision this SPEC makes. |
| 9 | §14 | **New Q9 (M1): Phase 1 uptime.** The dev PC is the running system for a year, but vLLM is started by hand, Windows reboots for updates, and the machine is also used for gaming — which wants the card vLLM has reserved. How gaming and serving coexist, how both come back after a reboot, and what a satellite hears meanwhile, is unsettled. |
| 10 | §2 Machines | The `ha` row lists the real options — **Home Assistant Green, an N100 mini PC, or a Pi 5 + NVMe** — and states that there is **one `ha` box and no redundancy pair**: resilience comes from the §10 hardware interlocks at the equipment, backups to `nas`, and a UPS. |

---

## Changes in v1.3

Folds in the hardware change of 2026-09-20 (see `docs/DECISIONS.md`). **No §3 rule is touched**: this revision amends §2 and §11 only, so unlike v1.2 it needs no safety acceptance.

| # | Section | Change |
|---|---------|--------|
| 1 | §2 Machines | `hub` is a **new build that does not exist yet**, not the repurposed desktop. The single RTX 5090 sits in the dev PC until `hub` is built; the dev PC then takes an RTX 5080 16 GB. The RTX 3070 is gone. A `dev` row is added. |
| 2 | §2 GPU assignment | The **single-GPU profile is the default**: LLM and auxiliary models share the 5090. "The 5090 serves the LLM and nothing else" is kept but scoped to the optional dual-GPU profile. Every profile declares `aux_reserve_gb`, validated in config, and vLLM starts before the auxiliary models. |
| 3 | §2 GPU assignment | A profile's memory budget is **measured, not estimated**: `weights_gb` and `kv_cache_gb` come from an M1 evaluation run on the real card and carry the run that produced them. |
| 4 | §2 Models | Verified Hugging Face repository IDs recorded, with the sm_120 NVFP4 caveat. `Qwen/Qwen3.6-35B-A3B` added as a co-candidate, run text-only. |
| 5 | §2 Models | **New rule:** every profile pins a `revision` (commit sha). A tag is not a pin. |
| 6 | §2 Models | **New rule:** `kv_cache_dtype` is a declared profile field, evaluated at M1 rather than assumed. |
| 7 | §2 Inference backend | After the card swap the dev PC uses vLLM on `hub` over the LAN, with a small-model profile on the 5080 as the offline fallback. |
| 8 | §11 | M1 notes the evaluation runs on the dev PC's 5090 before the card moves. M2's row says the JSONL sink is implemented, not merely decided (Q8). The "no dedicated hardware before M9" note is qualified. |
| 9 | §14 | **Q1 resolved** (M1): a custom Home Assistant integration passes `device_id` and `conversation_id` as headers alongside the bearer token. Removed from the open list. |

---

## Changes in v1.2

Folds in the decisions of the M0 review (2026-09-19; see `docs/DECISIONS.md`). Two changes **amend the letter of §3** and need explicit acceptance: row 3 (the satellite confirmation form no longer always carries a UUID) and row 8 (dry-run no longer blocks local effects).

| # | Section | Change |
|---|---------|--------|
| 1 | §3 intro, §3.2, §3.11, §5, §8.1 | The gateway (`core/gateway.py`) is the single enforcement point for every tool call. Modules supply handlers and clients only. Handlers are sealed so they cannot run except through it. |
| 2 | §3.1, §7 | HA-actuating tools are TOML script wrappers; non-HA tools (records, cad) are Python `ToolSpec`s through the same gateway. Parameters are enums or bounded numbers, never free-text entity or service arguments. T3 declarations are status-only. |
| 3 | §3.3, §14 Q6 | **Amends §3.3.** Voice confirmation may use a fixed HA intent that carries no UUID. `confirmers` are per area. PendingActions live in Postgres with atomic resolution, are expired and audited at startup, and duplicate HA events are idempotent. |
| 4 | §3.4 | `max_runtime_s` in the tool TOML, RUNBOOK checklist for the HA side, and the assumption that the non-admin HA user does not restrict service calls. |
| 5 | §3.5, §4, §8.1, §8.5 | New rule 4: action turns receive only the current utterance, never replayed history. `query_service_history` is untrusted; free-text T0 results are typed-and-stripped or marked untrusted. |
| 6 | §3.6, §13 | Sessions are minted server-side and bound to the authenticated identity; service-to-service bearer token; client-supplied tools and system prompts are ignored; effective tools = tier ≤ ceiling AND scope intersection; `"*"` rejected on T2 tools. |
| 7 | §3.7 | Audit semantics: intent row then one outcome row per call, denials audited, unknown tools recorded with a null tier, audit failure denies, immutability mechanism. |
| 8 | §3.8 | **Amends §3.8.** Dry-run blocks external side effects only; local effects proceed with `dry_run=true` in the audit row. T2 still creates a PendingAction. The code default is ON permanently; production turns it off explicitly. Settable from every layer. |
| 9 | §3.9, §14 Q7 | The printer upload client never sends a start flag or enables auto-start. |
| 10 | §2 | GPU numbers corrected (the FP8 weights exceed 0.90 × 32 GB). Config carries per-model profiles, including a single-GPU profile with a smaller model. |
| 11 | §5, §6, §7, §12 | Composition root (`app.py`) holds the module list. Failure policy: invalid config fails fast, an unavailable dependency starts the module degraded. Fail-safe defaults are logged; unknown keys and misspelled `FARMHUB_*` variables fail startup; optional parameters become required-but-nullable. |
| 12 | §9, §11 | Each §9 test lands in the milestone that creates its subject (`docs/SAFETY_CHECKLIST.md`); new tests added; M6 no longer claims all of §9. |
| 13 | §14 | Added Q6 (voice confirmation, M8), Q7 (printer stack, M12), Q8 (audit sink composition, M2); Q1 extended. |

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
| hub | Ubuntu 24.04, RTX 5090 32 GB + RTX 3070 8 GB (Phase 2; does not exist yet) | vLLM, FarmHub app, Postgres, Whisper, Piper, embeddings |
| ha | Home Assistant Green, an N100 mini PC, or a Pi 5 + NVMe — Home Assistant OS | Home Assistant and all safety-critical automation. **One box, not a redundant pair** |
| nas (media) | New TrueNAS build, Xeon E5-1650 v4 — **no GPU at first** (that CPU has no integrated graphics); an Intel Arc A310 may be added later for transcoding | Document corpus, media, backups. **Never an AI target**, the Arc included |
| tv (console) | The current TrueNAS box repurposed, keeping the GTX 1060 until it is upgraded | TV and emulation. **Not part of the FarmHub system, and never an AI target** |
| sat-* | Raspberry Pi 4/5 | Wyoming voice satellites: kitchen, barn, workshop |
| dev | ClevatessPrime: Windows + WSL2 + Docker Desktop, RTX 5090 32 GB | **Phase 1: the runtime host.** Development, every milestone's measurements, and the running system until `hub` exists |

**Model inference runs on one machine at a time** — the dev PC in Phase 1, `hub` in Phase 2. `nas`, `tv` and `ha` never run it, whatever spare GPU they happen to hold: the corpus is read over the network, a GPU in `nas` would be for transcoding and a GPU in `tv` is for games. After the swap the dev PC keeps a small-model profile for offline work, so "one machine" means one *serving* host at a time rather than one machine forever. The single exception is wake-word detection on the satellites, which is a keyword spotter rather than a model FarmHub serves; §8.8 does not say today where it runs, and M9 settles it.

Where §2 and §8 say `hub` — Piper, Whisper, Postgres, the app itself — read "the Phase 1 runtime host" until `hub` is built.

**Phase 1, now to roughly September 2027.** FarmHub is developed, measured and *run* on the dev PC's RTX 5090. An RTX 3070 cannot be added to that machine: the PSU will not take it. So the LLM and the helper models (faster-whisper, BGE-M3, bge-reranker-v2-m3, ~5 GB together) share the one card. This is the profile every milestone is built and measured on, not a fallback and not a degradation, and it constrains M3, M4 and M9 as much as M1. It is not a dedicated card, either: the same machine is also used for gaming, so it is sometimes wanted whole by something else — see §14 Q9. The desktop itself is not a tenant; the monitor runs on the motherboard's integrated graphics, and this card reads 0 MiB whenever vLLM is down.

**Phase 2, `hub`.** A new build with an RTX 5090 32 GB **and** an RTX 3070 8 GB — decided, not optional. The 5090 then serves the LLM alone and the helpers move to the 3070. The 5090 is power-limited to ~450 W since the machine runs permanently: inference loses very little and it runs cooler and quieter.

The 5090 moves from the dev PC to `hub` when `hub` is built. The dev PC then takes an **RTX 5080 16 GB or a used RTX 4090 24 GB — undecided**, settled at the swap.

ha is a separate physical machine on purpose. Heating and pumps must keep working when the GPU box is down, being patched, or has thrown a driver fault. This is an architectural rule, not a preference. Never propose collapsing ha into hub.

**There is one `ha` box, and no plan for a second.** A redundancy pair is not the answer here, because the failure that matters — a valve left open, a pump running dry — is contained at the equipment rather than in software. Resilience comes from three things instead: the §10 hardware interlocks and independent cutoffs at every irrigation valve and pump circuit, which work with `ha` powered off; configuration backed up to `nas`; and a UPS, so a brief outage is not a restart. Do not propose clustering Home Assistant, and do not make FarmHub depend on `ha` being reachable.

### GPU assignment

**Phase 1 — one card, shared.** vLLM, faster-whisper, BGE-M3 and bge-reranker-v2-m3 all live on the dev PC's 5090, and each profile says in numbers how that card is divided. This is the production arrangement until `hub` exists, so M3's embedding during ingest, M4's reranking during retrieval and M9's Whisper all have to fit beside a loaded LLM. There is no "stop the LLM overnight" escape hatch: §8.4's contextual retrieval needs the local LLM to write the blurbs, so the batch job is one more tenant on the shared card rather than a reason to unload it, and taking serving offline overnight is not a decision this SPEC makes.

**Phase 2 — two cards, and the 5090 serves the LLM and nothing else:**

- `CUDA_VISIBLE_DEVICES=0` → vLLM, configured by a model profile (below).
- `CUDA_VISIBLE_DEVICES=1` → faster-whisper, BGE-M3, bge-reranker-v2-m3. ~5.2 GiB on the 3070's 8 GB — an estimate, see `aux_reserve_gib` below.

Config carries per-model profiles: `repo_id`, `revision`, `quantization`, `kv_cache_dtype`, `gpu_memory_utilization`, `max_model_len`, `weights_gib`, `kv_cache_gib`, `aux_reserve_gib`, `card_total_gib`, `measured_by`, `chat_template_kwargs` and `temperature`. None of these is hard-coded.

Every memory figure is **GiB** — the unit vLLM reports and the evaluation records — and the field names say so, so no reader has to guess whether a conversion happened.

The last two are request parameters rather than memory, and both are **required**:

- `chat_template_kwargs` is what every request must carry for this model to answer at all. For the M1 candidates that is `{ enable_thinking = false }`: with thinking on, Qwen3.6 spends the whole token budget reasoning and never reaches an answer, which scored 2 of 6 on part numbers by never finishing (`docs/MODEL_EVAL.md`). It lives in the profile because the next model may spell the switch differently. An empty table is allowed for a model that needs nothing, and startup warns.
- `temperature` is the sampling setting the profile was measured at, sent whenever a caller names none. The server's default is not an answer: the first two evaluation runs took it, and each tool score was a single unreproducible draw.

Rules that make "never silently OOM" keep teeth while one card is shared:

- Every profile declares `aux_reserve_gib`, the memory the auxiliary models need: ~5 in Phase 1, **zero** in Phase 2, where they live on the other card. Until a helper model has actually been loaded and measured (M3, M4, M9), the Phase 1 figure is an **estimate**, and `docs/MODEL_EVAL.md` says so where it is used.
- Config validation rejects a profile whose `weights_gib + kv_cache_gib + aux_reserve_gib` exceeds the card, and one whose `gpu_memory_utilization` does not leave at least `aux_reserve_gib` free.
- **vLLM starts before the auxiliary models**, so its utilization fraction is computed against a known-free card rather than against whatever happened to load first.
- **The budget numbers are measured, not estimated.** `weights_gib` and `kv_cache_gib` come from an evaluation run on the real card (§11 M1, `docs/MODEL_EVAL.md`), and a profile records the run that produced them. A profile carrying hand-written numbers is not a valid profile.
- **A profile belongs to one machine.** `weights_gib` carries from the dev PC to `hub` — same card, same checkpoint — but `kv_cache_gib` does not: it is whatever is left once the helpers are on another card. **No Phase 2 profile is adopted until an eval run on `hub` produces it.**

This validation is arithmetic over declared numbers, not a live VRAM probe. The real guard against OOM is still vLLM's own preflight plus the measured numbers; the validator catches a profile edited into something impossible. The evaluation harness does read the card from outside — `nvidia-smi` while the model is serving — so how much is really left beside vLLM is a measured figure even though `aux_reserve_gib` is not; FarmHub itself does no such probe (NVML is not on the §12 list). It is also deliberately incomplete: it counts weights and KV cache but not peak activation or CUDA graph memory, which live inside vLLM's share too (about 1.2 GiB on the measured profile, `docs/MODEL_EVAL.md`). Treat a profile that only just passes as one that has not been tried.

FP8 weights for a 30B-class model run to ~30 GB, which exceeds 0.90 × 32 GB before any auxiliary reservation is taken out. Sharing one card therefore needs a smaller quantization — see Models below.

After the swap the dev PC's profile selects a smaller model and loads the auxiliary models onto the same device, logging a warning at startup. That profile is the offline fallback only; normally the dev PC uses vLLM on `hub`. Lowering the utilization of the same model is not a valid degradation. Never silently OOM.

### Models

| Role | Model | Notes |
|------|-------|-------|
| LLM candidate A | `Qwen/Qwen3.6-35B-A3B`, NVFP4 at `nvidia/Qwen3.6-35B-A3B-NVFP4` | MoE, 35B total / 3B active, 262K context, function calling and structured output. The only candidate with an official vLLM recipe validated on the RTX 5090: `--quantization modelopt_fp4`, `--block-size 128`, `VLLM_HAS_FLASHINFER_CUBIN=1`, vLLM ≥ 0.28.0, 32 GB caps context at 64K before any auxiliary reservation. Multimodal; **FarmHub runs it text-only** (§1 has no multimodal path, and the vision tower's memory is memory the KV cache needs) |
| LLM candidate B | `Qwen/Qwen3-30B-A3B-Instruct-2507`, 4-bit AWQ or GPTQ-Int4 | MoE, ~3B active. Chosen originally for tool calling. Official FP8 is `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` at ~32 GB, which does not fit with a KV cache. No vendor NVFP4 build of this variant exists (see the caveat below), so 4-bit community builds are the realistic path: `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ`, `JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4`, `Intel/Qwen3-30B-A3B-Instruct-2507-int4-asym-AutoRound` |
| LLM fallback | `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, AWQ | Profile switch only. No official AWQ; community builds only |
| STT | faster-whisper large-v3, `int8_float16` | CTranslate2. This compute type requires CUDA |
| TTS | Piper | CPU, on hub, served over Wyoming. Satellites only play audio |
| Embeddings | BAAI/bge-m3 | Dense + sparse from one model |
| Reranker | BAAI/bge-reranker-v2-m3 | Top-30 → top-5 |

The repository IDs above were verified on 2026-09-20. Model IDs live only in config, never in code.

**NVFP4 on sm_120 — the caveat that shaped the table.** The 5090 is consumer Blackwell (sm_120), not datacenter Blackwell. vLLM's NVFP4 MoE backend selection has open bugs there: the device-capability check omits SM12.0, so the kernel dispatch either fails outright or a ModelOpt checkpoint falls back to **Marlin W4A16** with a "no native FP4 support" warning. The memory saving survives that fallback; the FP4 compute does not, and a run can look like NVFP4 without being it. So an evaluation run records which kernel vLLM actually selected, not which one was requested. `nvidia/Qwen3-30B-A3B-NVFP4` and `RedHatAI/Qwen3-30B-A3B-NVFP4` are not a way around this: both quantize the older *Base* model rather than Instruct-2507, and the NVIDIA one targets TensorRT-LLM.

**Every profile pins a `revision` (commit sha).** A tag or a branch is not a pin: one NVFP4 upload of Qwen3.6-35B-A3B was silently replaced on 2026-07-10 with weights that produce looping output, and an unpinned profile would have picked that up on the next pull. The evaluation harness refuses to run an unpinned profile and verifies the resolved sha against the one requested.

**`kv_cache_dtype` is declared and evaluated, never assumed.** FP8 KV cache buys memory, and on this hybrid architecture it has reported quality collapse. M1 measures each candidate at the backend default and at FP8, scoring quality as well as memory; a memory win that costs part-number fidelity is not a win here.

Higher precision is preferred over 4-bit because quantization damage shows up first on exactly the things this system does: part numbers, torque figures, tool-call argument fidelity. That is why the M1 evaluation scores those directly rather than throughput alone, and it is the same reason FP8 KV cache is not taken on trust.

**At M1:** evaluate candidate A first — it is the proof that the NVFP4 stack works on this card at all — then candidate B in 4-bit, then the fallback, each at both KV cache dtypes. Record the whole matrix in `docs/MODEL_EVAL.md`, the chosen profile in `config/farmhub.example.toml` and `docs/DECISIONS.md`.

### Inference backend

Primary: vLLM, OpenAI-compatible server, automatic prefix caching enabled. The system prompt and tool schemas are near-constant, so prefix caching is the main latency win.

The application talks to the LLM only through an `LLMBackend` protocol implemented by an OpenAI-compatible client. Switching to Ollama must be a config change, never a code change. Do not import vLLM anywhere outside `modules/llm/`.

In Phase 1 the backend is local to the dev PC. After the card swap the dev PC talks to vLLM on `hub` over the LAN through that same protocol, with a small-model profile on whichever card it ends up with (5080 16 GB or used 4090 24 GB) as the offline fallback. Both are config, not code.

## 3. Non-negotiable safety rules

These are acceptance criteria. Write tests in `tests/safety/` that fail if any is violated.

The gateway (`core/gateway.py`) is the single enforcement point for every tool call: tier, scope, rate limit, taint, dry-run, audit and PendingAction lifecycle. Modules supply handlers and clients only, and no handler can run except through the gateway.

### 3.1 No generic actuation primitive

There must be no tool of the form `call_service(domain, service, entity_id, data)` or any equivalent that lets the model reach arbitrary Home Assistant entities. Every HA-actuating tool is a named, typed, individually-declared TOML wrapper around one specific HA script (§7). Parameters are enums or bounded numbers only, never free-text entity or service arguments: a tool with a free-text `scene` or `entity` argument is `call_service` by another name. Non-HA tools (records, cad) are Python `ToolSpec`s and pass through the same gateway.

### 3.2 Capability tiers

Every tool declares a tier. The gateway enforces tiers server-side. Model output is never trusted to respect them.

- **T0 READ** — see §8.1 for the full list. Auto-execute, always available (subject to §3.5 and §3.6).
- **T1 COMFORT** — indoor lights, scenes, media, notifications, logging a service record. Auto-execute. Logged.
- **T2 CONFIRMED** — greenhouse watering, garden and field irrigation, ventilation, CAD generation and slicing. The model may propose; execution requires explicit human confirmation per §3.3.
- **T3 FORBIDDEN** — heating, well and pressure pumps, anything affecting livestock, mains electrical, starting a 3D print. Never exposed to the LLM in any form. Read-only status is permitted (asking whether the barn heating is on is fine). Control lives exclusively in deterministic Home Assistant automations and fixed HA intents, so it still works by voice — just not through the model, and not when hub is down. A T3 declaration is status-only: the loader rejects any T3 entry that carries an actuating handler or script.

Rationale to preserve: heating protects pipes from freezing, and a pump can flood a building or drain the well that also supplies the house. Neither may depend on a language model's judgement.

### 3.3 Confirmation flow for T2

A T2 tool call returns a `PendingAction` with a UUID and a TTL of 120 seconds. The PendingAction records the originating satellite, session, scope, tool, and exact arguments, and is stored in Postgres. Resolution (confirmed, denied or expired) is an atomic conditional UPDATE, so each PendingAction resolves exactly once and a duplicate HA event is idempotent. All pending rows are expired and audited at startup. A resolution is its own audited call (§3.7). It executes only after a matching confirmation arrives through one of two channels. Expired actions are discarded.

**Satellite confirmation.** Either (a) a structured confirm callback carrying the UUID, arriving from the same session that created the action, or (b) for voice, a fixed Home Assistant intent ("confirm" / "bekreft"), handled deterministically by HA and never interpreted by the model, which calls the FarmHub confirm endpoint with the satellite's `device_id`. Form (b) confirms the single pending action for that satellite's session; with zero or several pending it does nothing and says so. Form (b) is provisional until Q6 is finalized at M8.

**Push-notification confirmation.** FarmHub sends a Home Assistant actionable notification whose action identifier embeds the UUID. The confirmation arrives as the HA `mobile_app_notification_action` event over the WebSocket connection. It is accepted only if:

- the UUID matches a live, unexpired PendingAction;
- the originating device is in the `confirmers` allowlist configured for the action's area (a list of HA mobile-app device IDs); and
- the action has not already been confirmed, denied, or expired (each PendingAction resolves exactly once).

The audit row for the execution records which channel confirmed it and, for push, which device.

Confirmations are never inferred from conversational text such as "yeah go ahead", or from model output. They require either a structured callback carrying the UUID or the fixed-intent form above. The arguments executed are the arguments stored in the PendingAction; a confirmation cannot modify them.

### 3.4 Runtime watchdogs live in Home Assistant

Every irrigation, watering and ventilation script has a maximum runtime enforced inside the Home Assistant script itself, not in Python. A hung hub, a crashed process or a bug in this repo must not be able to leave a valve open. Assume the Python side will fail eventually.

Each such tool's TOML declares `max_runtime_s`, the watchdog configured in the HA script. It is validated against the tool's parameter maximums (a parameter may not permit more than the watchdog allows). FarmHub cannot read the HA script body, so `docs/RUNBOOK.md` carries a checklist for the HA side. Assume the non-admin HA user does not restrict which services a token may call (verify at M5): containment rests on the gateway, the HA-side watchdogs and the §10 hardware cutoffs.

### 3.5 Untrusted content and actuation tools are never in the same context

Document text from the corpus and content fetched from the web are untrusted input. This includes the results of any tool marked `reads_untrusted_content=True` (`search_documents`, `get_news`, `fetch_url`, `query_service_history`), not only content retrieved by the pipeline before the completion. Free-text results from any other T0 tool are either typed and stripped to allowlisted fields or the tool is marked untrusted.

Four rules enforce this:

1. **Pipeline retrieval.** A completion whose context includes retrieved content must be issued with `tools=[]` or only T0 tools.
2. **Action turns exclude untrusted tools.** The tool list for an `action` turn contains only T0 tools with `reads_untrusted_content=False`, plus scope-permitted T1 and T2 tools. The model therefore cannot fetch untrusted content mid-turn and then act on it.
3. **Runtime tripwire.** The orchestrator tracks a per-turn `context_tainted` flag, set whenever untrusted content enters the context by any route. Any completion issued while the flag is set is restricted to T0 tools, and any attempt to execute a T1+ tool while it is set is denied and audited. This is defence in depth: rule 2 should make it unreachable.
4. **No replayed history in action turns.** An `action` turn receives only the current utterance plus a fixed server-side system prompt. It never receives replayed conversation history, including the follow-up turn of a `mixed` flow, because earlier assistant answers may summarize untrusted content and would launder it into the actuation context.

This blocks prompt injection from a scanned PDF or a web page into the actuation path. Enforce all four with explicit assertions in the orchestrator, and test them.

### 3.6 Satellite scope is server-side

Satellite identity comes from the Wyoming connection or the HA `device_id`, authenticated at the endpoint, and the gateway derives the allowed tool set from it. The model never supplies its own location, area or scope as a tool argument. The workshop satellite cannot actuate greenhouse valves.

Sessions are minted server-side and bound to the authenticated satellite identity; HA's `conversation_id` is a lookup key only, never trusted alone. `/v1/chat/completions` and the confirm endpoint require a shared service-to-service bearer token, listen on a configurable bind address and are firewalled to the `ha` host. FarmHub ignores client-supplied `tools` and system prompts and builds its own. Transport: a custom Home Assistant integration (`custom_components/farmhub/`) sends the bearer token, `device_id` and `conversation_id` as headers (Q1, settled at M1).

Each satellite has a home area, a scope (the set of areas it may act on) and a tier ceiling that may not exceed T2. The effective tools are those with tier ≤ the satellite's ceiling AND `allowed_scopes` intersecting its scope. `"*"` is rejected in `allowed_scopes` for T2 tools. `confirmers` are configured per area.

**Fail closed.** A request that arrives without a satellite identity, or with one not present in the satellite registry, is served with T0 tools only (trusted and untrusted T0, per §3.5), and the condition is logged as a warning. Identity is never taken from message content.

### 3.7 Audit log

Every tool invocation is audited, denials included. A call writes an immutable INTENT row before anything runs, then exactly one OUTCOME row, linked by `call_id`. Fields: timestamp, satellite, session, tool name, tier, arguments, decision (executed / denied / pending / expired), result, duration, `dry_run`, and for T2 the confirmation channel and confirming device. A T2 confirmation or expiry is its own call whose `parent_call_id` is the proposing call. An unknown tool is recorded with the name as given (truncated) and a null tier.

If the intent row cannot be written, the call is denied and nothing runs. If the outcome row cannot be written after a call already ran, the real result is still returned and a critical log line carries the row. Append-only table plus a JSONL file: the table's database role has no UPDATE or DELETE and a trigger rejects them; the JSONL file is append-only by convention (a hash chain is deferred). No tool call may execute without producing an audit row. How the two sinks combine when Postgres is down is Q8.

### 3.8 Dry run

A global `--dry-run` flag and `FARMHUB_DRY_RUN` env var, also settable in `farmhub.toml`; the effective value is logged at startup. The default in code is ON and never changes. Production config turns it off explicitly (from M7), and startup logs a warning whenever it is off.

In dry-run mode, every tool at T1 and above with an external side effect (Home Assistant service calls, notifications, printer uploads) logs its intended call and returns a simulated success. No HTTP or WebSocket service call reaches Home Assistant and nothing reaches the printer. Local effects (service-event writes, sandbox runs) proceed, with `dry_run=true` in the audit row. T2 tools still create a PendingAction and require confirmation; push notifications are simulated and confirmations come from the test harness.

### 3.9 3D printing

Slicing and upload to the printer queue are permitted. Starting a print is not. Unattended ignition risk, and the bed may not be clear. The pipeline ends at "file uploaded, notification sent." There is no tool, flag or config option that starts a print. The printer upload client is a fixed call that never sends a start or print flag and never enables an auto-start queue, and a test asserts this (Q7 picks the printer stack).

### 3.10 Sandboxing generated code

The cad module executes LLM-written CadQuery/OpenSCAD source. It runs in a subprocess with no network, a tmpfs working directory, a 30-second timeout and a memory cap. A container is better if available. Never `exec()` generated code in the main process.

### 3.11 Rate limits

Per-tier, per-session limits enforced in the gateway. T2 defaults to 3 proposals per 10 minutes. Denials produce audit rows.

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
        │               NO retrieved content, NO untrusted tools (§3.5 rule 2),
        │               current utterance only, NO history (§3.5 rule 4)
        │                ├── T1 → execute → audit → respond
        │                └── T2 → PendingAction → ask → await confirm → execute → audit
        │
        └── mixed    → answer first, then offer the action as a separate turn. Never fuse them.
                       The follow-up turn carries no history from the answer (§3.5 rule 4).
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
    SAFETY_CHECKLIST.md
  config/
    farmhub.example.toml
    satellites.example.toml     # satellite registry: name, home area, scope, tier ceiling, input mode
    tools/                      # one TOML file per exposed tool
  src/farmhub/
    __init__.py
    __main__.py
    app.py                      # composition root: explicit module list and wiring
    core/
      config.py                 # pydantic-settings, layered TOML + env
      context.py                # AppContext: shared clients, no globals
      registry.py               # module lifecycle and tool registration (list supplied by app.py)
      gateway.py                # single enforcement point for every tool call
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
    FORBIDDEN = 3  # never returned to the model; exists so config can express it


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema, strict, no additionalProperties
    tier: Tier
    handler: Callable[[ToolCall, ToolContext], Awaitable[ToolResult]]
    allowed_scopes: frozenset[str]  # satellite areas, or {"*"}
    timeout_s: float = 10.0
    reads_untrusted_content: bool = False
```

Also define: `LLMBackend`, `Retriever`, `Embedder`, `Reranker`, `DocumentParser`, `Chunker`, `ToolContext` (carries satellite, session, scope, dry_run, context_tainted), `ToolResult` (carries an `untrusted: bool` flag the pipeline uses to set `context_tainted`), `CallOrigin`, `PendingAction`, `HealthReport`, `AuditEntry` and `AuditSink`. A `ToolSpec` handler is sealed at construction: it refuses to run unless the gateway invoked it (a runtime seal plus a static test).

Modules are registered in an explicit list in the composition root (`app.py`) and passed to `registry.py` — no entry-point scanning, and `core` never imports a module. Explicit is easier to reason about when something fails to load at 06:00.

**Failure policy.** Invalid config, including bad tool files, fails fast for every module: `startup` raises `ConfigError`, already-started modules are stopped and the app exits. A runtime dependency that is unavailable (HA unreachable, database down) raises `DependencyUnavailable`: the module starts degraded, is marked unhealthy, its tools are removed from every tool list, and startup is retried with backoff. A core failure fails fast.

## 7. Configuration

Layered: defaults in code → `config/farmhub.toml` → environment (`FARMHUB_*`) → CLI flags. Validated by pydantic-settings at startup. Fail loudly and exit on invalid config. No silent defaults for anything safety-relevant: fail-safe defaults in code are allowed (dry-run on, the T2 rate limit), but their effective values are logged at startup and printed by `farmhub config check`, and values with no safe default (`confirmers`, the satellite registry) are required. Unknown keys in `farmhub.toml` and misspelled `FARMHUB_*` variables fail startup.

HA-actuating tools are config-driven, one TOML file per tool. Non-HA tools (records, cad) are Python `ToolSpec`s and go through the same gateway.

```toml
name = "start_greenhouse_watering"
description = "Start watering in the greenhouse for a set number of minutes."
tier = "CONFIRMED"
ha_script = "script.greenhouse_water_timed"
allowed_scopes = ["kitchen", "greenhouse"]
required = ["duration_min"]
max_runtime_s = 1200

[parameters.duration_min]
type = "integer"
minimum = 1
maximum = 20
description = "Minutes to run. Hard-capped by the Home Assistant script."
```

Schema generation rules:

- The generated JSON Schema always sets `additionalProperties: false`.
- `required` must be present and list parameter names explicitly; an empty list is allowed but must be written. A missing `required` key fails validation.
- Every parameter must declare `type` and `description`. Numeric parameters must declare `minimum` and `maximum`. String parameters must declare either `enum` or `maxLength`. Parameters are never free-text entity or service names (§3.1).
- Optional parameters (those not in `required`) are generated as required-but-nullable, so strict JSON-schema modes work.
- `max_runtime_s` is required on every T2 tool that runs for a duration and is validated against the parameter maximums (§3.4).
- `"*"` in `allowed_scopes` is rejected on T2 tools (§3.6).

Adding an HA-actuating tool must never require touching Python. A tool file referencing a nonexistent HA script fails validation at startup, not at first use. A tool file declaring `tier = "FORBIDDEN"` is loaded (so config can express it) but is never placed in any model-visible list. A FORBIDDEN declaration is status-only: the loader rejects one that carries an actuating handler or `ha_script`.

## 8. Module specifications

### 8.1 ha — Home Assistant bridge

Supplies the Home Assistant half of §3; the gateway enforces it. Connects over the HA WebSocket API using a long-lived token scoped to a dedicated non-admin HA user.

Responsibilities: load tool definitions from `config/tools/`, validate each against the live HA entity and script registry at startup, supply the HA client and PendingAction storage that the gateway uses to enforce tier/scope/rate-limit/timeout/dry-run/audit before any service call, handle push-notification confirmation events (§3.3), and surface real HA results rather than letting the model narrate success. The gateway, not this module, filters T3 out of every tool list returned to the model.

T0 tool list (read-only, always available subject to §3.5):

| Tool | Returns | Untrusted |
|------|---------|-----------|
| get_entity_state | State plus allowlisted typed attributes of one allowlisted entity | no |
| get_sensor_history | Time series for a sensor, bounded window | no |
| list_area_status | Summary of one area's sensors in a single call (state plus allowlisted typed attributes) | no |
| search_documents | Chunks with source path, page, heading path | yes |
| query_service_history | Filtered service events for an asset | yes |
| next_service_due | Computed due date or hours | no |
| list_assets | Known machines and buildings | no |
| get_weather | Yr / met.no forecast | no |
| get_news | Headlines from configured RSS feeds | yes |
| fetch_url | Readable text from an allowlisted domain | yes |

`get_weather` is marked trusted because it returns structured numeric data from a single fixed API, parsed into typed fields; no free text from the response reaches the model.

`query_service_history` is marked untrusted because the notes field is speech-to-text. `get_entity_state` and `list_area_status` return the state plus allowlisted typed attributes only, never free-form attribute text.

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

T0 tools as listed in §8.1 (`query_service_history` is untrusted). One T1 tool: `log_service_event(...)` — recording maintenance by voice while your hands are dirty is the single highest-value write in the system.

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

This module is responsible for Whisper STT and Piper TTS as Wyoming services on hub, plus the satellite registry (`config/satellites.toml`): name → home area → scope (set of areas) → tier ceiling (at most T2) → input mode. `confirmers` are configured per area (§3.3).

Prefer running the upstream `wyoming-faster-whisper` and `wyoming-piper` servers, configured and supervised by this module's deployment files, over reimplementing them. Write custom code only where they fall short, and record why in `docs/DECISIONS.md`.

Workshop satellite: `wake_word: none`, `ptt: gpio`, close-talk headset mic. Far-field arrays do not work against machine noise regardless of push-to-talk.

## 9. Testing

`tests/safety/` is mandatory, part of the default test run, and never skipped (a skip or xfail there fails the run). Each test lands in the milestone that creates its subject; `docs/SAFETY_CHECKLIST.md` tracks which exist.

- Model-visible tool list contains no T3 tool or entity, under every scope.
- A turn containing retrieved content is issued with no tool above T0.
- The tool list for an `action` turn contains no tool with `reads_untrusted_content=True`.
- After any untrusted tool result enters a turn, a subsequent T1+ tool call in that turn is denied and audited, and no HA service call is made.
- A classifier failure (bad JSON, timeout) routes to `question`.
- A request with no satellite identity, or an unknown one, receives only T0 tools.
- A T2 tool call without confirmation produces no service call to HA.
- A PendingAction past its TTL is rejected.
- A confirmation from a different session is rejected.
- A push confirmation from a device not in the area's `confirmers` list is rejected.
- A PendingAction cannot be confirmed twice.
- Scope filtering: the workshop scope cannot reach greenhouse tools.
- Every tool call, denials included, produced an intent row and exactly one outcome row.
- An unknown tool, a T3 tool and a not-yet-permitted tool are denied and audited (unknown: name as given and truncated, tier null).
- An audit write failure denies the call and nothing runs.
- No handler can be invoked except through the gateway.
- Client-supplied tools and system prompts are ignored.
- An `action` turn receives no replayed history, including the follow-up turn of a `mixed` flow.
- Dry-run mode issues zero outbound HA service calls.
- A tool config referencing a missing HA script fails startup.
- A tool config missing `required`, or with unbounded numeric or string parameters, fails startup.
- A T3 declaration with an actuating handler or script fails startup; `"*"` on a T2 tool fails startup; a parameter maximum above `max_runtime_s` fails startup.
- Unknown config keys and misspelled `FARMHUB_*` variables fail startup; dry-run is on with no configuration.
- In dry-run a T2 call still creates a PendingAction and needs confirmation.
- PendingAction rows are expired and audited at startup; a duplicate HA confirmation event is idempotent.
- The printer upload client never sends a start flag.
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
| M1 | llm + FastAPI + /v1/chat/completions | HA conversation agent gets an answer from vLLM. Model IDs pinned by revision and a fitting model profile verified against measured numbers (§2), evaluated on the dev PC's 5090 — the Phase 1 runtime host, where it stays. Satellite identity reaches FarmHub (§14 Q1), with the bearer token and server-minted sessions of §3.6 |
| M2 | Storage, migrations, records | Service events insert and query by CLI. JSONL audit sink implemented alongside the Postgres sink and its catch-up (Q8) |
| M3 | ingest: parse, chunk, embed, manifest | Corpus indexes; rerun is a no-op; library check lints |
| M4 | rag: hybrid retrieval + rerank + search_documents | Cited answers from manuals, with pages |
| M5 | ha bridge, T0 read-only, audit log | Reports sensor states. Zero write paths exist yet |
| M6 | Tiers, scopes, rate limits, dry-run, classifier, taint tracking, safety suite | Every §9 test whose subject exists passes (T2 confirmation tests arrive at M8, the cad sandbox test at M12; see `docs/SAFETY_CHECKLIST.md`). Dry-run is still on |
| M7 | T1 comfort actions live | Lights work by voice. Audit rows correct |
| M8 | T2 + PendingAction confirmation flow (satellite and push) | Greenhouse watering works with confirm; TTL expiry and confirmer allowlist tested. Voice confirmation form settled (Q6) |
| M9 | voice: Whisper + Piper services, satellite registry | Kitchen satellite end-to-end |
| M10 | Workshop PTT satellite | GPIO trigger, no wake word, headset mic |
| M11 | web: weather, news, allowlisted fetch | Yr forecast by voice |
| M12 | cad: generate, sandbox, preview, approve, slice, queue | Bracket reaches printer queue. Does not start. Printer stack settled (Q7) |

M5 through M8 are where haste causes real damage. Slow down there. Write each §9 test before the code it guards.

Note on hardware: in Phase 1 every milestone runs on the dev PC's RTX 5090 (§2), so "dedicated hardware" is not a later concern — it is the one card, shared, from M1 onwards. That is a real constraint on M3 (embedding while the LLM is loaded), M4 (reranking on the same card) and M9 (Whisper and Piper beside both), and each of those has to measure what it costs rather than assume there is room. `hub` and its second GPU change the arithmetic when they arrive, not the design.

Development itself still needs none of this: M0–M12 can be *developed* against Ollama or a mocked LLM, and **no test in any milestone may require a GPU** (§9).

## 12. Standards

- Python 3.12 (uv-managed, pinned in `.python-version`). uv for dependency management, `uv_build` as the build backend. ruff format + lint. mypy --strict on `core/`, and on each module and protocol implementation as it lands.
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

Do not build: a custom web UI (Home Assistant is the UI), user accounts, or auth beyond the HA token and service-to-service bearer tokens, cloud fallback inference, text-to-mesh 3D generation, custom satellite firmware, or anything that starts a 3D print.

## 14. Open questions

Known unresolved decisions. Raise each one at the milestone named, propose options, and wait for an answer. Record the outcome in `docs/DECISIONS.md` and remove it from this list.

Resolved: **Q1 (M1)** — satellite identity reaches FarmHub through a custom Home Assistant integration (`custom_components/farmhub/`) that forwards the utterance with the bearer token, `device_id` and `conversation_id` as headers. See `docs/DECISIONS.md`.

- **Q2 (M6): Action-verb and alias list for the classifier pre-pass.** Proposed source: derived automatically from tool names and descriptions in `config/tools/`, plus a hand-maintained Norwegian and English verb list.
- **Q3 (M6): Rate limits for T0 and T1.** Only T2 has a default. Propose values.
- **Q4 (M12): Preview rendering.** How the three PNG previews are rendered inside the sandbox (OpenSCAD's renderer, a headless mesh renderer, or CadQuery SVG export converted to PNG). Must not add a network-capable dependency to the sandbox.
- **Q5 (M12): Sandbox mechanism.** Plain subprocess with resource limits versus a container, given what is available on the runtime host (the dev PC's WSL2 in Phase 1, `hub` in Phase 2 — §2).
- **Q6 (M8): Voice confirmation.** The fixed-intent form in §3.3 ("confirm" / "bekreft" handled deterministically by HA, calling the confirm endpoint with the satellite's `device_id`) is provisional. Settle: reading the exact arguments aloud in the confirmation prompt, what a stray "confirm" heard near a satellite can do, and behaviour with several pending actions.
- **Q7 (M12): Printer stack.** Moonraker, OctoPrint or PrusaLink. Their upload APIs can start a print (a `print` flag or an auto-start queue); the upload client must never use either, with a test.
- **Q8 (M2): Audit sink composition.** Leaning yes: JSONL is the mandatory write-ahead record (a failed JSONL write denies the call), and Postgres is written as well with idempotent catch-up from JSONL after an outage, so a database outage does not deny every tool.
- **Q9 (M1): Phase 1 uptime.** The dev PC is the running system for about a year (§2), but it is not run like a server: vLLM is started by hand (`deploy/vllm/vllm.sh`, `restart: "no"` on purpose so a Docker Desktop restart cannot silently reclaim the card), and Windows reboots for updates whenever it likes. The same machine is also used for **gaming**, which wants the whole 5090 that vLLM has reserved — so vLLM cannot simply be left running, and "started by hand" is partly deliberate. Settle how gaming and serving coexist (stop vLLM and lose FarmHub for the evening, cap `gpu_memory_utilization` low enough for both, or accept that the two do not overlap), how vLLM and the FarmHub app come back after a reboot, and what a satellite hears while they are down — HA's own intents keep working (§3.2), so the answer may be "nothing, and FarmHub says it is unavailable", but that has to be chosen rather than discovered. To settle before the M1 end-to-end test through Home Assistant.
