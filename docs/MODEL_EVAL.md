# Model evaluation (SPEC §2, §11 M1)

Which model FarmHub serves, and the measurements behind that choice.

**Status: 1 of 6 runs recorded (2026-09-22).** Candidate A loads and serves on the
dev PC's RTX 5090. No profile is chosen yet: the quality scores from that run are not
usable as they stand, because the model answered in thinking mode and never reached an
answer inside the token budget. See *Results* and *What the first run settled*.

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
  on this architecture. Each candidate is run at both dtypes and scored on quality,
  not only on what fits.
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

Each is run at `kv_cache_dtype` `auto` and `fp8`, so the matrix is six runs.

## What is measured

| Metric | How | Why it decides anything |
|---|---|---|
| Memory fit | vLLM's own startup log: weights GiB, KV cache GiB and tokens | These are the numbers a profile must declare. 5 GB is held back for Whisper, BGE-M3 and the reranker, which share the card on the default single-GPU profile |
| Usable context | KV cache tokens ÷ 3 concurrent sessions | §1 serves 2–3 concurrent voice users; `max_model_len` alone flatters the result |
| Time to first token | Streaming, cold prefix and warm prefix, then 3 concurrent | Answers are read aloud, so TTFT is what a person in a barn experiences. The cold/warm gap is the evidence prefix caching works (§2) |
| Tool-call fidelity | Strict schemas mirroring §7: enums, bounded integers, explicit `required`, `additionalProperties: false` | Scored separately for schema violations, invented tools and false positives. A model that ignores the bounds makes the tool TOML decorative; one that acts on a question opens a valve |
| Part numbers and torque | Synthetic manual extracts, exact string match | `SKF 6205-2RS` and `SKF 6250-2RS` are equally fluent and one orders the wrong bearing. Includes a case whose answer is absent: inventing a figure scores as failure, because a wrong torque is worse than none |
| Classifier JSON | The §4 structured call over Norwegian and English utterances | Validity is scored before accuracy: an invalid response routes to `question` by design, so it is safe but useless |

All data is synthetic and committed under `evals/data/`. The machines, part numbers
and figures are invented; nothing there should be believed outside this harness.

## Results

### Run `evals/2026-09-22T17-39-18Z` — candidate A, KV cache `auto`

- vLLM image `vllm/vllm-openai:v0.29.0`; GPU RTX 5090, driver 596.36, 32607 MiB
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a052410cfd62085d94b58866fd0f2c3c5` (pin verified current before download)
- Text-only via `--language-model-only`; `--block-size 128`; `--gpu-memory-utilization 0.82`; `--max-model-len 32768`
- 3 concurrent sessions assumed; 5.0 GB held back for Whisper, BGE-M3 and the reranker

| candidate | loaded | kernel | weights GB | KV GB | KV tokens | ctx @3 | TTFT cold/warm | tools | grounding | classifier JSON |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen36-35b-a3b-nvfp4` | yes | marlin-fallback | 19.55 | 4.9 | 207,842 | 69,280 | 0.396 / 0.082 | 0.75 | 0.33 † | 1.0 |

† Not a usable quality score. See *Thinking mode* below.

**Memory.** Weights 19.55 GiB, KV cache 4.9 GiB, peak activation 1.08 GiB, CUDA graphs
0.08 GiB. vLLM's own summary: 20.13 GiB consumed (weights + non-torch) of the 26.11 GiB
that `0.82` utilization allows, leaving roughly 6.8 GiB of the card outside vLLM —
comfortably above the 5.0 GB auxiliary reservation the single-GPU profile needs.
Budget: 19.55 + 4.9 + 5.0 = 29.45 GB against a 31.8 GB card.

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

### Emitted profile (not yet adopted)

```toml
[profiles.qwen36_35b_a3b_nvfp4]
repo_id = "nvidia/Qwen3.6-35B-A3B-NVFP4"
revision = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
quantization = "modelopt_fp4"
kv_cache_dtype = "auto"
gpu_memory_utilization = 0.82
max_model_len = 32768
weights_gb = 19.55
kv_cache_gb = 4.9
aux_reserve_gb = 5.0
card_total_gb = 31.8
measured_by = "evals/2026-09-22T17-39-18Z"
```

The memory numbers are sound and this block is what the harness emitted. It is not
adopted yet, because adopting a profile means accepting its quality, and the quality
half of this run has to be repeated.

## What the first run settled

Beyond the numbers, three things that were assumptions before:

1. **The NVFP4 stack loads and serves on this card** — the point of running candidate A
   first — but through Marlin W4A16, not native FP4.
2. **vLLM on WSL2 needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`**, or v0.29.0 fails at device
   init with "UVA is not available". Dev PC only; hub is native Linux (`docs/DECISIONS.md`).
3. **Two harness bugs that a dry run could not have caught**: the text-only flag was
   passed as JSON that lost its quotes through compose, and v0.29.0 reworded both
   memory log lines, so the profile could not be emitted. Both are fixed and covered
   by tests built from the real log.

### Operational notes for the next run

- The checkpoint took 27 minutes to download (21.8 GiB) and the rate swung between
  ~90 MB and ~3 GB per minute. Weight load is 42 s, engine init 116 s, so a repeat run
  on a warm cache is ~3 minutes to serving.
- The vLLM image is 30.5 GB unpacked, which is the larger disk cost on `C:`.
- vLLM holds ~20 GB of VRAM from engine start, before any weights are loaded.

## Decision

_Pending._ Candidate A is a plausible profile on memory and latency, but no model is
chosen until the quality scores are re-measured with thinking disabled, and until the
remaining five runs (candidate A at `fp8`, candidates B and the fallback at both
dtypes) say what they cost. The chosen profile then goes here, in `docs/DECISIONS.md`
and in `config/farmhub.example.toml`.
