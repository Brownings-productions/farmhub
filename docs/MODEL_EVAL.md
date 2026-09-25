# Model evaluation (SPEC §2, §11 M1)

Which model FarmHub serves, and the measurements behind that choice.

**Status: candidate A measured; candidate B and the fallback still to run.** Candidate A
loads and serves on the dev PC's RTX 5090, reproduces every part number and torque figure
with thinking disabled, and answers a spoken question in well under a second — including on
a 10,000-token RAG prompt. Its memory numbers have reproduced across three runs. The matrix
is **three runs, not six**: the fp8 KV cache runs were dropped (§"The candidates"). No
profile is adopted yet. Read the newest run;
the two below it predate the harness changes (pinned sampling, repeated tool cases,
bucketed out-of-enum outcomes, a RAG-sized prompt, `nvidia-smi` while serving) and their
tool scores cannot be compared with anything.

## Why this exists

SPEC §2 requires a serving profile's memory budget to be **measured on the real card,
not estimated**, and the profile to carry the run that produced the numbers. Config
validation rejects a profile without `measured_by`, so a profile cannot exist without
a run like this one.

It also requires the *quality* questions to be settled by measurement rather than
assumption:

- **Which kernel actually runs.** The RTX 5090 is consumer Blackwell (sm_120). vLLM's
  NVFP4 MoE dispatch has open bugs there, and a ModelOpt checkpoint can fall back to
  Marlin W4A16 with a "no native FP4 support" warning. A run can look like NVFP4 and
  not be, so every run records the kernel lines from the startup log.
- **Whether FP8 KV cache is usable.** It buys memory and has reported quality collapse
  on this architecture, so it is declared and evaluated rather than assumed. Measured
  answer as of 2026-09-25: **not needed**, because at `auto` the cache already holds
  twice `max_model_len` per session — so the fp8 runs were dropped rather than run
  (`docs/DECISIONS.md`). The rule stands; the matrix got smaller.
- **Whether 4-bit damages what this system does.** §2 prefers higher precision because
  quantization damage shows up first on part numbers, torque figures and tool-call
  argument fidelity — which is exactly what is scored here, rather than throughput.

## Running it

Needs the RTX 5090 and will download tens of GB. It is outside `tests/` because SPEC
§9 says no test may require a GPU, and CI never runs it.

```sh
uv run python -m evals.run --list     # the matrix, and nothing else
uv run python -m evals.run --check    # validate pins and report upstream drift
uv run python -m evals.run            # the whole matrix
uv run python -m evals.run --only qwen36-35b-a3b-nvfp4 --kv-dtype auto
```

Each candidate is a full cycle: write `deploy/vllm/.env`, bring vLLM up, wait, read
the startup log, score, bring it down. A candidate that will not load is recorded as
such and the run continues — on this card "does it load at all" is a real result.

Before anything is downloaded, every candidate's pin is checked. **An unpinned
candidate is refused and nothing is fetched**: a tag is not a pin, and one NVFP4
upload of Qwen3.6-35B-A3B was silently replaced on 2026-07-10 with weights that loop.
`--check` also reports when a pin has fallen behind upstream, so a move is a decision
rather than a surprise.

Results land in `evals/results/` (gitignored, with the raw vLLM logs). The run prints
a markdown block to paste below, and a `[profiles.*]` block to paste into config —
**nobody retypes the numbers**, which is what keeps `measured_by` honest. A candidate
whose memory numbers could not be parsed gets no profile block at all, only a comment
pointing at its log.

## The candidates

Defined in `evals/data/candidates.json`. Revisions resolved 2026-09-20.

| name | repo | why |
|---|---|---|
| `qwen36-35b-a3b-nvfp4` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Candidate A. The only one with an official vLLM recipe validated on the RTX 5090. Multimodal, run **text-only**: FarmHub has no multimodal path (§1) and the vision tower's memory is memory the KV cache needs. |
| `qwen3-30b-a3b-2507-awq` | `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ` | Candidate B. No vendor NVFP4 build of Instruct-2507 exists, so 4-bit is the realistic path for the model SPEC originally chose. Official FP8 is ~32 GB and does not fit with a KV cache. |
| `mistral-small-32-24b-awq` | `gghfez/Mistral-Small-3.2-24B-Instruct-hf-AWQ` | The §2 fallback. Community AWQ; no official build exists. |

Each is run at `kv_cache_dtype` **`auto` only, so the matrix is three runs.** The fp8 KV
cache runs were dropped on 2026-09-25: candidate A at `auto` gives 207,842 KV tokens —
about 69,000 per session at three concurrent — against a `max_model_len` of 32,768, so the
cache is not the binding constraint and fp8 would trade quality for room this system does
not need. `kv_cache_dtype` remains a declared, evaluated profile field, and fp8 becomes
worth measuring if a candidate's weights leave much less room, `max_model_len` grows for
RAG turns, or more than three sessions have to be served (`docs/DECISIONS.md`).

## What is measured

| Metric | How | Why it decides anything |
|---|---|---|
| Memory fit | vLLM's own startup log: weights GiB, KV cache GiB and tokens | These are the numbers a profile must declare, in GiB. 5 GiB is held back for Whisper, BGE-M3 and the reranker, which share the card in Phase 1 (SPEC §2) — an **estimate**, unlike everything else here |
| Card in use while serving | `nvidia-smi --query-gpu=memory.used`, read twice: model loaded and idle, then again after the concurrent long-context case | The outside view. vLLM's log is a self-report, and it under-reports its own process by the 0.65 GiB the readings found; `card_total − this` is what the helpers really have left |
| Usable context | KV cache tokens ÷ 3 concurrent sessions | §1 serves 2–3 concurrent voice users; `max_model_len` alone flatters the result |
| Time to first token | Streaming, cold prefix and warm prefix, then 3 concurrent | The cold/warm gap is the evidence prefix caching works (§2) |
| Time to a complete answer | Total generation seconds and tokens/s, at 1 and at 3 concurrent | What a person waiting in the barn actually experiences. Marlin's cost lands in generation, not TTFT, so TTFT alone hid it entirely |
| Latency on a RAG-sized prompt | ~6,000 tokens of synthetic manual context, alone and at 3 concurrent | M4 will send prompts this size; a figure from a one-line question does not describe them |
| Tool-call fidelity | Strict schemas mirroring §7: enums, bounded integers, explicit `required`, `additionalProperties: false`. **Every case run three times**, scored as a range | Scored separately for schema violations, invented tools and false positives. A model that ignores the bounds makes the tool TOML decorative; one that acts on a question opens a valve. Three passes because one draw decided the first two runs' scores |
| Out-of-enum behaviour | Six cases naming an area or zone outside the enum, in English and Norwegian, bucketed **declined / substituted / invented** | Substituting a *different real area* passes validation and acts in the wrong room — a worse failure than inventing a value the schema rejects, and one a single "false positive" count hid. Paired with in-enum Norwegian controls using the same verbs, so a decline cannot be confused with failing at the language |
| Part numbers and torque | Synthetic manual extracts, exact string match | `SKF 6205-2RS` and `SKF 6250-2RS` are equally fluent and one orders the wrong bearing. Includes a case whose answer is absent: inventing a figure scores as failure, because a wrong torque is worse than none |
| Classifier JSON | The §4 structured call over Norwegian and English utterances | Validity is scored before accuracy: an invalid response routes to `question` by design, so it is safe but useless |

All data is synthetic and committed under `evals/data/`. The machines, part numbers
and figures are invented; nothing there should be believed outside this harness.

## Results

### Run `evals/2026-09-25T15-07-51Z` — candidate A, KV cache `auto`, current harness

Candidate A on the harness as it now stands: `temperature: 0` on every request, the tool
cases run three times, out-of-enum outcomes bucketed, a RAG-sized prompt timed, and
`nvidia-smi` read while the model is serving. **These are the numbers the other five
matrix runs will be compared against**; the two runs below predate the harness and
cannot be.

| candidate | loaded | kernel | weights GiB | KV GiB | KV tokens | ctx @3 | card used GiB | left for aux GiB | TTFT cold/warm | answer s 1/3 | gen tok/s 1/3 | RAG answer s 1/3 | tools min-max | out-of-enum d/s/i | grounding | classifier JSON | reasoning | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen36-35b-a3b-nvfp4` | yes | marlin-fallback | 19.55 | 4.9 | 207,842 | 69,280 | 25.76 | 6.08 | 0.405 / 0.076 | 0.272 / 0.356 | 207.3 / 174.8 | 0.715 / 0.371 | **0.8 to 0.867** | **15/3/0** | **1.0** | 1.0 | **0** | **0** |

Budgets: latency 256 tokens, tools 512, grounding 512, classifier 128. Reasoning emitted:
0 replies. Memory reproduces a third time to the decimal — 19.55 and 4.9 GiB on the same
pin and image — which is what makes them profile numbers.

#### The card from outside, and what the helpers really get

New in this run: `nvidia-smi --query-gpu=memory.used`, read with the model loaded and
idle (25.67 GiB) and again after the concurrent long-context case (25.76 GiB). The larger
reading is the one below. It reconciles with vLLM's self-report, which is the point of
taking it:

| Line | GiB | Where it comes from |
|---|---|---|
| Card total | **31.84** | 32,607 MiB |
| weights | 19.55 | vLLM's log |
| non-torch overhead | 0.58 | vLLM's log (20.13 consumed − 19.55 weights) |
| CUDA graphs | 0.08 | vLLM's log |
| KV cache | 4.90 | vLLM's log |
| Sum of vLLM's profiled figures | 25.11 | the four above |
| **In use while serving, measured** | **25.76** | `nvidia-smi` |
| Gap: vLLM process overhead outside the profiled figures (unattributed) | 0.65 | 25.76 − 25.11 |
| **Card total minus the measured reading: what the helpers actually get** | **6.08** | 31.84 − 25.76 |
| `aux_reserve_gib`, the estimate | 5.00 | declared |
| **Margin over the estimate** | **1.08** | 6.08 − 5.00 |

**Nothing but vLLM holds memory on this card.** The monitor runs on the motherboard's
integrated graphics, and `nvidia-smi` reads 0 MiB whenever vLLM is down — before this run
and after it. So the 0.65 GiB gap is not a desktop: it is vLLM's own footprint outside
the figures its profiler reports — CUDA context, library workspaces, allocator reserve —
and these logs do not say how it splits. It is recorded as unattributed rather than
guessed at. The same correction applies to the **1.64 GiB "free memory on device
(30.2/31.84)" on startup** in the run below, which an earlier version of this document
called the Windows desktop: that is what vLLM's worker had already taken when it read the
card, before any weights were loaded.

The two figures are not the same quantity and must not be added: 1.64 GiB is what was
resident at worker init, and 0.65 GiB is what the steady-state reading exceeds the
profiled sum by. **How to settle it:** read `nvidia-smi` at three points instead of two —
before `docker compose up`, after engine init but before weights load, and while serving.
That separates CUDA context from allocator growth without guessing. Not built; the two
readings answer the question that matters, which is how much is left.

So the estimate holds, with about a gigabyte spare. Two caveats on that margin:

1. **It is steady-state.** Peak activation (1.08 GiB by vLLM's profiler) is transient and
   does not appear in either reading, so a burst can eat the whole margin. Two samples
   are not a peak.
2. **It is one profile's overhead, not a constant.** The 0.65 GiB belongs to this
   checkpoint, this vLLM version and these flags. A different candidate can differ, which
   is why the reading is taken per run rather than assumed once.

#### Latency, and one figure that flatters

Warm TTFT 0.076 s, a complete 40-token answer in 0.272 s alone and 0.356 s with three
sessions at once. Generation runs 207 tokens/s alone and 175 per stream at three
concurrent — a gentler drop than the 1.6× of 2026-09-23, on the same weights and the
same kernel, which is itself a reminder of how much run-to-run spread there is here.

**The RAG-sized numbers need reading carefully.** The prompt measured **10,107 tokens**
(the estimate that sized the text said ~6,000 — the run records the server's own
`prompt_tokens` for exactly this reason). Alone: TTFT 0.579 s, complete answer 0.715 s.
At three concurrent: TTFT 0.169 s, answer 0.371 s — *faster*, which is not a load
benefit. The single stream paid for the cold prefill of 10k tokens and the three
concurrent streams then hit the prefix cache it left behind. **The honest figure for a
cited answer on a cold prompt this size is 0.579 s to first token and 0.715 s to a whole
answer**, and it is comfortably inside what a spoken reply can absorb.

#### Tools: 0.8 to 0.867, and temperature 0 is not determinism

Three passes over 15 cases: 12, 12 and 13 correct. No schema violations, no invented
tools, no unparseable arguments across all 45 calls — the §7 bounds hold.

**Pinning the temperature did not make the model repeatable.** Identical utterances went
different ways between passes: `out-of-enum-fjoset` declined, substituted, declined;
`out-of-enum-loft` substituted, substituted, declined. vLLM at `temperature: 0` is not
bit-deterministic — batching and MoE expert routing vary — so *repetition*, not pinning,
is what makes a tool score mean anything. Three passes was the right call and one pass
would still mislead. Pinning is still worth keeping: it removes one source of spread
rather than all of them.

**Out-of-enum, 18 observations over six cases: 15 declined, 3 substituted, 0 invented.**
Nothing ever emitted a value the schema would reject. Every failure was the dangerous
kind instead — a schema-valid call in the wrong place:

- `Slå på lyset på loftet` (the loft, not in the enum) → `set_indoor_light(area="hall")`,
  twice in three passes.
- `Skru på lyset i fjøset` (the barn) → `set_indoor_light(area="workshop")`, once.
- The English `barn` case that substituted `workshop` on 2026-09-23 declined 3/3 here.
  One draw, last time.

This is the §3.1/§3.6 argument restated in measurements: the enum stops the model
inventing an entity, and then the model picks a *real* one it was not asked about. Only
the gateway's server-side scope and T2 confirmation stand between that and a valve.
It is also why a spoken confirmation must read back the resolved arguments rather than
the request (`docs/DECISIONS.md`, 2026-09-25).

**The new in-enum controls immediately earned their place.** `Kan du skru på lyset i
verkstedet?` — the workshop, *in* the enum, a request that should produce a call —
produced **no call in 2 of 3 passes**. The polite Norwegian question form is being read
as a question rather than a command. Which means `out-of-enum-stabburet` ("Kan du skru på
lyset i stabburet?") passing 3/3 proves nothing about enum discipline: the model declines
that phrasing whether or not the area exists. Without the control that would have been
counted as the model correctly refusing an unknown area. Two consequences:

- **For the classifier (§4, Q2 at M6):** the action-verb pre-pass needs the polite
  interrogative forms Norwegian actually uses — `kan du`, `kunne du`, `vil du` — or real
  requests will be classified as questions. Worth a case in the classifier set too.
- **For scoring:** every out-of-enum case needs an in-enum twin in the same language and
  the same phrasing. Four of six have one now; `garage` and the lawn zone do not.

`Water the propagator for two hours` still declines 3/3 rather than clamping to the
20-minute maximum or asking. Consistent, safe, and a worse experience than asking.

**Grounding 6/6 and classifier 10/10 valid JSON**, both unchanged across three runs now.
Same single classifier disagreement: one ambiguous utterance read as `action` where
`question` was expected — which §4's safe default handles the other way round, so it is
the harmless direction.

#### Emitted profile (validated, not yet adopted)

Pasted through `ModelProfile` before being written here: **valid**. weights + kv = 24.45
GiB against vLLM's 26.11 GiB share, budget + aux = 29.45 against a 31.84 GiB card, and
5.73 GiB left free for a 5.0 GiB reservation — with 6.08 GiB actually measured.

```toml
[profiles.qwen36_35b_a3b_nvfp4]
repo_id = "nvidia/Qwen3.6-35B-A3B-NVFP4"
revision = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
quantization = "modelopt_fp4"
kv_cache_dtype = "auto"
gpu_memory_utilization = 0.82
max_model_len = 32768
weights_gib = 19.55
kv_cache_gib = 4.9
aux_reserve_gib = 5.0
card_total_gib = 31.84
measured_by = "evals/2026-09-25T15-07-51Z"
chat_template_kwargs = { enable_thinking = false }
temperature = 0.0
```

Still not adopted: candidate B and the fallback have not run, and `aux_reserve_gib` is still an estimate.

### Run `evals/2026-09-23T16-13-25Z` — candidate A, KV cache `auto`, thinking disabled

The repeat of the run below, with `chat_template_kwargs {"enable_thinking": false}` on
every request and larger token budgets. **This is the run with usable quality scores.**

| candidate | loaded | kernel | weights GiB | KV GiB | KV tokens | ctx @3 | TTFT cold/warm | answer s 1/3 | gen tok/s 1/3 | tools | grounding | classifier JSON | reasoning | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen36-35b-a3b-nvfp4` | yes | marlin-fallback | 19.55 | 4.9 | 207,842 | 69,280 | 0.472 / 0.074 | 0.225 / 0.41 | 253.6 / 155.9 | 0.75 | **1.0** | 1.0 | **0** | **0** |

Budgets: latency 256 tokens, tools 512, grounding 512, classifier 128. Reasoning
emitted: 0 of 24 replies, so the parameter was honoured rather than ignored — the
tripwire that makes the rest of the row meaningful.

**Memory reproduces exactly**, and the whole card is accounted for below. Weights 19.55
GiB, KV cache 4.9 GiB, 207,842 tokens, on the same pin and the same image. Two runs a day
apart agreeing to the decimal is the evidence that these numbers can be written into a
profile.

#### Where the card goes — one table, GiB throughout

Every figure is GiB, taken from the 2026-09-23 startup log. Earlier notes mixed GB with
GiB and reported "roughly 6.8 GiB outside vLLM", which was wrong: that was 26.11 − 20.13,
which is neither the free space nor anything else meaningful. The free space is 5.73.

| Line | GiB | Where it comes from |
|---|---|---|
| Card total | **31.84** | 32,607 MiB, as the driver reports it |
| Already resident when vLLM read the card | 1.64 | "Free memory on device (30.2/31.84 GiB)" — **vLLM's own, before weights**; see the 2026-09-25 run. Not a desktop: the monitor is on the iGPU and `nvidia-smi` reads 0 MiB with vLLM down |
| vLLM's share at `gpu_memory_utilization = 0.82` | **26.11** | 0.82 × 31.84 |
| ├ weights | 19.55 | "Model loading took 19.55 GiB memory" |
| ├ non-torch overhead | 0.58 | 20.13 consumed − 19.55 weights |
| ├ peak activation | 1.08 | memory profiler |
| ├ CUDA graphs | 0.08 | "CUDA graph pool memory: 0.08 GiB (actual)" |
| └ KV cache | 4.90 | "Available KV cache memory: 4.9 GiB" → 207,842 tokens |
| **Sum inside vLLM's share** | **26.19** | 19.55 + 0.58 + 1.08 + 0.08 + 4.90 |
| **Outside vLLM, for the helper models** | **5.73** | 31.84 − 26.11 |
| `aux_reserve_gib` the profile declares | 5.00 | Whisper + BGE-M3 + reranker (SPEC §2) — **an estimate, see below** |
| **Margin over the reservation** | **0.73** | 5.73 − 5.00 |

Two things worth taking from that arithmetic:

1. **The sum inside vLLM's share (26.19) slightly exceeds the share itself (26.11).**
   vLLM sizes the KV cache from what is actually free, so it is self-consistent; what it
   means is that the 0.82 fraction is not a hard ceiling on vLLM's footprint.
2. **The config validator is deliberately optimistic.** It checks
   `weights + kv ≤ utilization × card` — 24.45 ≤ 26.11 here — and ignores activation and
   CUDA graphs, the 1.16 GiB between those two figures. So a profile that only just
   passes validation has not been tried; vLLM's own preflight is the real guard. SPEC §2
   now says this in as many words.
3. **The 1.64 GiB is vLLM's, not the machine's.** An earlier version of this document
   called it the Windows desktop; it is not (2026-09-25 run). It does **not** come back on
   `hub`, because vLLM will be running there too. What does change on `hub` is
   `aux_reserve_gib` dropping to 0 with the helpers on the 3070, which is why SPEC §2 says
   `kv_cache_gib` does not carry over between machines.

#### `aux_reserve_gib = 5.0` is an estimate, and the only one here

Every other number above was measured. This one was not. **No helper model has ever been
loaded on this card and measured** — not faster-whisper, not BGE-M3, not
bge-reranker-v2-m3. The 5 GiB is arithmetic over published model sizes, and it is doing
real work: it is subtracted from vLLM's share, it is what `gpu_memory_utilization = 0.82`
was chosen to leave free, and config validation rejects profiles against it. A profile
can therefore pass every check and still not fit, if the estimate is low.

One thing makes the estimate optimistic rather than conservative: the figure counts
weights and nothing else. CTranslate2's and PyTorch's own allocator overhead, activation
during a batch of chunks, and whatever fragmentation results from three processes taking
and returning memory beside a long-lived vLLM are all outside it — and vLLM's own
unattributed overhead (0.65 GiB on 2026-09-25) shows that per-process overhead beyond the
weights is real and not small.

What does *not* threaten it is a desktop: the monitor runs on the motherboard's integrated
graphics and this card holds 0 MiB whenever vLLM is down. Gaming does compete for the card
(SPEC §14 Q9), but that is a scheduling question rather than a few hundred megabytes.

**How to measure it, when the milestone arrives** — proposed, not built:

1. **One model at a time, on an idle card.** Load faster-whisper `large-v3` at
   `int8_float16`, read `nvidia-smi --query-gpu=memory.used` before and after, then
   transcribe a 30-second clip and read it again under load. Repeat for BGE-M3 (embedding
   a realistic batch of 32 chunks) and for the reranker (scoring 30 candidates). Three
   numbers, each with an idle and a loaded reading.
2. **Then all three together, beside a loaded vLLM**, because the sum of three separate
   measurements is not the cost of running three at once. This is the number
   `aux_reserve_gib` should carry: the peak of the whole helper set while the LLM is
   serving.
3. **Where it lands:** M3 owns BGE-M3 (embedding during ingest), M4 the reranker
   (retrieval), M9 Whisper. Each of those milestones already has to measure what it costs
   on the shared card (SPEC §2, Phase 1), so the natural shape is a small harness beside
   `evals/` that each milestone adds one row to, and `aux_reserve_gib` becomes measured
   the moment the third row exists.
4. **Until then**, the run records `nvidia-smi` readings while serving, so "how much is
   actually left beside vLLM" is a measured figure even while what the helpers *need*
   is not. If the two ever cross — less left than the estimate assumes — the run says so.

**Grounding 6/6.** Every part number, torque figure, capacity, filter designation and
error code reproduced exactly, no wrong figure quoted from the surrounding context, and
the case whose answer is absent was refused rather than invented. This is the measure
SPEC §2 cares most about, and on 4-bit-with-FP8-layers weights it is perfect here.

**Latency, including the part TTFT hides.** Warm TTFT 0.074 s. A whole answer takes
0.225 s alone and 0.41 s with three sessions at once (worst 0.428 s), generating 253.6
tokens/s alone and 155.9 tokens/s per stream at three concurrent — a 1.6× slowdown
under the load §1 actually asks for. Answers here are 35–43 tokens; a 256-token answer
at the concurrent rate would take about 1.6 s. Marlin's cost is visible in this column
and invisible in TTFT, which is why it is now measured.

**Tools 6/8, and the two failures are not the same as last time.** No schema
violations, no invented tools, no unparseable arguments.

- The Norwegian watering request now works (`duration_min: 10`, `zone: benches`). That
  miss *was* the token budget.
- "Water the propagator for two hours" still produces no call. 120 minutes exceeds the
  parameter maximum of 20, and the case treats clamping or asking as acceptable — it
  did neither, it simply declined. Safe, and a worse experience than asking.
- **New, and the one worth attention:** "Turn on the light in the barn" called
  `set_indoor_light` with `area: "workshop"`. `barn` is not in the enum, so rather than
  inventing a value that would fail validation, it substituted a *different real area*.
  Schema-valid and wrong. This is a model picking a plausible-looking argument rather
  than declining, which is precisely why §3.1 keeps parameters to enums, why the
  gateway enforces scope server-side (§3.6), and why T2 needs confirmation: nothing
  about tier or scope may rest on the model getting this right. With the workshop
  satellite's own scope this call would have been permitted — in the correct area, for
  the wrong request.

**Classifier unchanged:** 10/10 valid JSON, 9/10 correct, the same ambiguous utterance
classified `action` where `question` was expected.

### Neither run above pinned sampling — so 0.75 is one sample, not a figure

Neither the 2026-09-22 nor the 2026-09-23 run set `temperature`, so both took the server
default and neither tool score is reproducible. That is the most likely explanation for
the `out-of-enum-area` case passing on 2026-09-22 and failing on 2026-09-23: two samples
of the same weights, not a change in the model. Grounding and the classifier came out
identical or nearly so both times, so the damage is concentrated in the tool score.
**Read 0.75 as "6 of 8 on one draw", and do not compare it with any later run.**

The harness no longer works this way. It now sends `temperature: 0` and records it in the
run header, repeats the tool cases three times and reports the score as a range with the
per-case pass counts, and separates an out-of-enum refusal from a *substitution* — the
`barn → workshop` failure above was scored as a plain miss, which hid that the model
picked a different real area rather than declining. It also times a RAG-sized prompt,
since a 6,000-token context is what M4 will actually send.

None of that is retrofittable onto the two runs above, so candidate A is re-run on the
new harness before candidate B or the fallback is scored. Until that run exists, the
numbers to trust here are the memory figures and grounding, both of which reproduced
across two runs; the tool score is a placeholder.

### Emitted profile (not yet adopted)

```toml
[profiles.qwen36_35b_a3b_nvfp4]
repo_id = "nvidia/Qwen3.6-35B-A3B-NVFP4"
revision = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
quantization = "modelopt_fp4"
kv_cache_dtype = "auto"
gpu_memory_utilization = 0.82
max_model_len = 32768
weights_gib = 19.55
kv_cache_gib = 4.9
aux_reserve_gib = 5.0
card_total_gib = 31.84
measured_by = "evals/2026-09-23T16-13-25Z"
chat_template_kwargs = { enable_thinking = false }
```

### Run `evals/2026-09-22T17-39-18Z` — candidate A, KV cache `auto`, thinking on (superseded)

- vLLM image `vllm/vllm-openai:v0.29.0`; GPU RTX 5090, driver 596.36, 32607 MiB
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a052410cfd62085d94b58866fd0f2c3c5` (pin verified current before download)
- Text-only via `--language-model-only`; `--block-size 128`; `--gpu-memory-utilization 0.82`; `--max-model-len 32768`
- 3 concurrent sessions assumed; 5.0 GB held back for Whisper, BGE-M3 and the reranker

| candidate | loaded | kernel | weights GiB | KV GiB | KV tokens | ctx @3 | TTFT cold/warm | tools | grounding | classifier JSON |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen36-35b-a3b-nvfp4` | yes | marlin-fallback | 19.55 | 4.9 | 207,842 | 69,280 | 0.396 / 0.082 | 0.75 | 0.33 † | 1.0 |

† Not a usable quality score. See *Thinking mode* below.

**Memory.** Weights 19.55 GiB, KV cache 4.9 GiB, peak activation 1.08 GiB, CUDA graphs
0.08 GiB. vLLM's own summary: 20.13 GiB consumed (weights + non-torch) of the 26.11 GiB
that `0.82` utilization allows. The free-space and budget figures first written here were
wrong; the corrected arithmetic is the table under the 2026-09-23 run above.

**Context.** 207,842 KV tokens is 69,280 per session at 3 concurrent, well beyond the
32,768 `max_model_len` this run configured. vLLM reports 6.34× concurrency at 32K, so
context is not the binding constraint here — weights are.

**Latency.** Cold TTFT 0.396 s; warm TTFT median 0.082 s (max 0.087 s); 3 concurrent
0.100 s median. The cold-to-warm gap is prefix caching doing its job (§2). These are
time to *first* token, which is not the same as time to a usable answer — see below.

**Kernel: the Marlin fallback is real, not theoretical.** vLLM selected
`MarlinNvFp4LinearKernel` for NVFP4 GEMM and the `MARLIN` NvFp4 MoE backend, passing
over `FLASHINFER_TRTLLM`, `FLASHINFER_CUTLASS` and `VLLM_CUTLASS`, and warned: "Your
GPU does not have native support for FP4 computation … Weight-only FP4 compression
will be used leveraging the Marlin kernel." The checkpoint also mixes schemes — vLLM
detected both `NVFP4` and `W4A16_NVFP4`, and some linear layers run FP8 through
FlashInfer. So on sm_120 with v0.29.0, NVFP4 buys the memory and not the compute. The
numbers above are Marlin numbers, and a later vLLM that dispatches a native FP4
backend needs its own run rather than inheriting them.

**Classifier.** 10/10 valid JSON, 9/10 correct. The one miss is the deliberately
ambiguous utterance, classified `action` where the case expects `question`. §4 routes
an invalid or failed classification to `question`, but a *confidently wrong* `action`
is not caught by that default — which is an argument for the deterministic pre-pass at
M6 (Q2), not against this model.

**Tools.** 6/8, with zero schema violations, zero invented tools, zero unparseable
arguments and zero false positives. Both failures are the same shape: no tool call at
all where one was expected (the Norwegian watering request, and the request whose
duration exceeds the parameter maximum). Nothing dangerous happened — it under-acted
rather than over-acted, and it correctly declined to call anything for the barn
heating (T3, not exposed), the out-of-enum area and the plain question. Both misses
are plausibly the same token-budget problem as the grounding cases.

**Thinking mode, and why the grounding score is not usable.** Every grounding answer
begins "Here's a thinking process:" and the stored answers are ~300 characters, cut
off at `max_completion_tokens = 256` while still reasoning. The model never reached
its answer, so the scorer saw the context figures quoted inside the reasoning and
counted them as wrong figures. 0.33 therefore measures the harness's budget, not the
model's part-number fidelity.

This is the open question from the M1 plan — *does Qwen3.6's thinking mode hurt
latency?* — and the answer is worse than "it hurts TTFT". TTFT is excellent (0.082 s
warm) because the first token arrives promptly; it is just a token of reasoning. What
matters for voice is time to a *usable* answer, which this run did not reach at all.

The checkpoint's chat template accepts `enable_thinking`, so the fix is a request
parameter (`chat_template_kwargs: {"enable_thinking": false}`), not a different model.
Until the harness sends it and the run is repeated, candidate A has **no valid quality
score**, and the comparison against candidates B and the fallback would be meaningless.

**Vision tower.** Recorded as unknown, honestly: no log line states whether it was
skipped. Reading the v0.29.0 source, `--language-model-only` zeroes every modality
limit and `_mark_tower_model` then skips loading a tower whose modalities are all zero,
so the weights should not be loaded. The circumstantial numbers agree — the
checkpoint's vision tensors are 0.83 GiB of 21.80 GiB, and vLLM loaded 19.55 GiB — but
Marlin repacking changes sizes too, so this is consistent rather than conclusive. To
settle it, load once with and once without the flag and compare the reported weights.

### Emitted profile (superseded by the run above)

Same memory numbers, `measured_by = "evals/2026-09-22T17-39-18Z"`. Superseded because
the quality half of this run was void, not because the numbers were wrong.

## What the first runs settled (2026-09-22 and 2026-09-23)

Beyond the numbers, three things that were assumptions before:

1. **The NVFP4 stack loads and serves on this card** — the point of running candidate A
   first — but through Marlin W4A16, not native FP4.
2. **vLLM on WSL2 needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`**, or v0.29.0 fails at device
   init with "UVA is not available". Dev PC only; hub is native Linux (`docs/DECISIONS.md`).
3. **Two harness bugs that a dry run could not have caught**: the text-only flag was
   passed as JSON that lost its quotes through compose, and v0.29.0 reworded both
   memory log lines, so the profile could not be emitted. Both are fixed and covered
   by tests built from the real log.

### Operational notes

- The checkpoint took 27 minutes to download (21.8 GiB) and the rate swung between
  ~90 MB and ~3 GB per minute. Weight load is 42 s, engine init 116 s, so a repeat run
  on a warm cache is ~3 minutes to serving.
- The vLLM image is 30.5 GB unpacked, which is the larger disk cost on `C:`.
- vLLM's non-weight footprint is not small: 1.64 GiB was resident when the worker first
  read the card, before any weights, plus 0.58 GiB of profiled non-torch overhead and
  0.65 GiB the steady-state reading exceeds the profiled sum by. (An earlier note here
  said "~20 GiB from engine start, before any weights" — that was a misreading of the
  20.13 GiB weights-plus-non-torch figure, which is measured after the load.)

## Decision

_Pending, but candidate A has a measured result to beat:_ it leaves **6.08 GiB** of the
card free beside vLLM against a 5.0 GiB reservation, reproduces every part number and
torque figure, answers in 0.356 s with three satellites talking at once and in 0.715 s on
a cold 10k-token RAG prompt, breaks no schema bound in 45 tool calls, and needs thinking
disabled to do any of it.

Still required before a profile is adopted: candidate B and the fallback, both at `auto`.
The chosen profile then goes here, in `docs/DECISIONS.md` and in
`config/farmhub.example.toml`.

**What the runs have already settled for the application**, whichever model wins:

- FarmHub's client sends the thinking switch and the profile's temperature on every
  request, both required profile fields. Implemented (`docs/DECISIONS.md`, 2026-09-25).
- A tool score is a range over repeated passes, not a number: `temperature: 0` does not
  make this stack repeatable.
- The failure mode to design against is **substitution**, not invention. Every
  out-of-enum failure so far was a schema-valid call in a real area nobody asked about,
  which is a gateway and confirmation problem rather than a schema one.
- The §4 classifier pre-pass must handle Norwegian polite interrogatives (`kan du …`), or
  it will route real requests to the question path (Q2, M6).
