# Model evaluation (SPEC §2, §11 M1)

Which model FarmHub serves, and the measurements behind that choice.

**Status: the harness is built; no run has been recorded yet.** Results go below as
they are produced. The chosen profile is recorded here, in `docs/DECISIONS.md` and in
`config/farmhub.example.toml`.

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

_No run recorded yet._

<!--
Paste the markdown block printed by `uv run python -m evals.run` here, newest first.
Keep every candidate, including the ones that would not load and the dtype that lost:
a later re-evaluation needs to know what was already ruled out and why.
-->

## Decision

_Pending the first run._ Record the chosen profile here and in `docs/DECISIONS.md`,
and paste its emitted block into `config/farmhub.example.toml`.
