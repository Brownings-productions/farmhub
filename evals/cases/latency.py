"""What a person waiting actually experiences (SPEC §2, §8.2).

Two different numbers, and the second is the one that was missing.

**Time to first token** is measured twice against the same prefix: the system prompt
and tool schemas are near-constant, so prefix caching is the main latency win (§2), and
the cold-to-warm gap is the evidence it works.

**Time to a complete answer** is measured because TTFT can be excellent while the
answer is still far away. On this card NVFP4 falls back to Marlin W4A16
(`docs/MODEL_EVAL.md`), so the quantization's cost lands in generation throughput, not
in the first token. An answer is read aloud, so nobody hears anything until it is
finished.

Both are measured with several sessions at once, because §1 serves 2-3 concurrent voice
users and a single-stream number flatters the result.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from evals.cases import common

DATA = Path(__file__).resolve().parent.parent / "data" / "latency.json"

# Long enough for a spoken answer to finish rather than be cut off, since the point is
# to time a whole answer. Recorded in the run, with the truncation count beside it.
MAX_TOKENS = 256


@dataclass
class Stream:
    """One streamed completion, timed end to end."""

    ttft_s: float | None
    total_s: float
    completion_tokens: int | None
    truncated: bool
    reasoning: str | None
    prompt_tokens: int | None = None

    @property
    def tokens_per_s(self) -> float | None:
        """Generation rate after the first token, which is what Marlin affects."""
        if self.completion_tokens is None or self.ttft_s is None:
            return None
        generating = self.total_s - self.ttft_s
        if generating <= 0 or self.completion_tokens <= 1:
            return None
        return (self.completion_tokens - 1) / generating


async def _stream(
    client: httpx.AsyncClient, model: str, prefix: str, prompt: str, *, context: str | None = None
) -> Stream:
    """Stream one answer to completion, timing the first token and the last.

    ``context`` is retrieved manual text, prepended to the user turn the way the §4
    pipeline will at M4.
    """
    user = prompt if context is None else f"Context:\n{context}\n\nQuestion: {prompt}"
    body = common.payload(
        model,
        [
            {"role": "system", "content": prefix},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=MAX_TOKENS,
        stream=True,
        # vLLM then sends a final chunk carrying usage, so the token count is the
        # server's own rather than a guess from counting deltas.
        stream_options={"include_usage": True},
    )
    started = time.perf_counter()
    ttft: float | None = None
    tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    text_parts: list[str] = []

    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload_text = line[6:].strip()
            if payload_text == "[DONE]":
                break
            try:
                chunk = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if usage := chunk.get("usage"):
                tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    text_parts.append(content)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    total = time.perf_counter() - started
    answer = "".join(text_parts)
    return Stream(
        ttft_s=ttft,
        total_s=total,
        completion_tokens=tokens,
        truncated=finish_reason == "length",
        reasoning=common.reasoning_in({"content": answer}),
        prompt_tokens=prompt_tokens,
    )


def _summary(streams: list[Stream]) -> dict[str, Any]:
    """Medians and worst cases over a set of streams."""
    ttfts = [s.ttft_s for s in streams if s.ttft_s is not None]
    totals = [s.total_s for s in streams]
    rates = [r for r in (s.tokens_per_s for s in streams) if r is not None]
    counts = [s.completion_tokens for s in streams if s.completion_tokens is not None]
    prompts = [s.prompt_tokens for s in streams if s.prompt_tokens is not None]
    return {
        "streams": len(streams),
        "prompt_tokens_median": round(statistics.median(prompts)) if prompts else None,
        "ttft_s_median": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_s_max": round(max(ttfts), 3) if ttfts else None,
        "answer_s_median": round(statistics.median(totals), 3) if totals else None,
        "answer_s_max": round(max(totals), 3) if totals else None,
        "gen_tokens_per_s_median": round(statistics.median(rates), 1) if rates else None,
        "gen_tokens_per_s_min": round(min(rates), 1) if rates else None,
        "completion_tokens_median": round(statistics.median(counts)) if counts else None,
        "truncated": sum(1 for s in streams if s.truncated),
        "reasoning_emitted": sum(1 for s in streams if s.reasoning),
    }


async def run(base_url: str, model: str, timeout_s: float = 300.0) -> dict[str, Any]:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    prefix: str = data["prefix"]
    prompts: list[str] = data["prompts"]
    concurrent: int = data["concurrent"]

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        # Cold: the first request against this prefix, before anything is cached.
        cold = await _stream(client, model, prefix, prompts[0])

        # Warm: the same prefix again, so the difference is the prefix cache.
        warm = [await _stream(client, model, prefix, prompt) for prompt in prompts]

        # Concurrent: what it looks like with several satellites talking at once.
        gathered = await asyncio.gather(
            *(_stream(client, model, prefix, prompts[i % len(prompts)]) for i in range(concurrent)),
            return_exceptions=True,
        )
        together = [s for s in gathered if isinstance(s, Stream)]

        # A RAG-sized turn: thousands of tokens of retrieved manual, alone and under the
        # same concurrency. Prefill dominates, which is where Marlin's cost lands.
        long_context = data.get("long_context")
        long_single: list[Stream] = []
        long_together: list[Stream] = []
        if long_context is not None:
            question, context = long_context["question"], long_context["context"]
            long_single = [await _stream(client, model, prefix, question, context=context)]
            long_gathered = await asyncio.gather(
                *(
                    _stream(client, model, prefix, question, context=context)
                    for _ in range(concurrent)
                ),
                return_exceptions=True,
            )
            long_together = [s for s in long_gathered if isinstance(s, Stream)]

    every = [cold, *warm, *together, *long_single, *long_together]
    return {
        "max_completion_tokens": MAX_TOKENS,
        # Flat, beside the other cases' counters, because the run-level total reads this
        # level. Nested under `single`/`at_concurrency` they were silently skipped.
        "reasoning_emitted": sum(1 for s in every if s.reasoning),
        "truncated": sum(1 for s in every if s.truncated),
        "cold_ttft_s": round(cold.ttft_s, 3) if cold.ttft_s is not None else None,
        "cold_answer_s": round(cold.total_s, 3),
        # Kept under their original names so runs stay comparable.
        "warm_ttft_s_median": _summary(warm)["ttft_s_median"],
        "warm_ttft_s_max": _summary(warm)["ttft_s_max"],
        "concurrent": concurrent,
        "concurrent_ttft_s_median": _summary(together)["ttft_s_median"],
        "concurrent_ttft_s_max": _summary(together)["ttft_s_max"],
        "samples": len([s for s in warm if s.ttft_s is not None]),
        # The new half: a whole answer, alone and under load.
        "single": _summary(warm),
        "at_concurrency": _summary(together),
        # The same, for a retrieval-sized prompt (M4's shape, measured at M1).
        "long_context_single": _summary(long_single) if long_single else None,
        "long_context_at_concurrency": _summary(long_together) if long_together else None,
    }
