"""Time to first token, cold and warm (SPEC §2, §8.2).

Answers are read aloud (§1), so TTFT is what a person standing in a barn actually
experiences. It is measured twice against the same prefix: the system prompt and tool
schemas are near-constant, so prefix caching is the main latency win, and the gap
between cold and warm is the evidence that it is working.

Also measured with several sessions at once, because §1 serves 2-3 concurrent voice
users and a single-stream number would flatter the result.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

DATA = Path(__file__).resolve().parent.parent / "data" / "latency.json"


async def _ttft(client: httpx.AsyncClient, model: str, prefix: str, prompt: str) -> float | None:
    """Seconds until the first content token arrives, or None if none did."""
    started = time.perf_counter()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_completion_tokens": 64,
    }
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                return None
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                return time.perf_counter() - started
    return None


async def run(base_url: str, model: str, timeout_s: float = 120.0) -> dict[str, Any]:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    prefix: str = data["prefix"]
    prompts: list[str] = data["prompts"]
    concurrent: int = data["concurrent"]

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        # Cold: the first request against this prefix, before anything is cached.
        cold = await _ttft(client, model, prefix, prompts[0])

        # Warm: the same prefix again, so the difference is the prefix cache.
        warm_samples: list[float] = []
        for prompt in prompts:
            value = await _ttft(client, model, prefix, prompt)
            if value is not None:
                warm_samples.append(value)

        # Concurrent: what it looks like with several satellites talking at once.
        results = await asyncio.gather(
            *(_ttft(client, model, prefix, prompts[i % len(prompts)]) for i in range(concurrent)),
            return_exceptions=True,
        )
        concurrent_samples = [r for r in results if isinstance(r, float)]

    return {
        "cold_ttft_s": round(cold, 3) if cold is not None else None,
        "warm_ttft_s_median": round(statistics.median(warm_samples), 3) if warm_samples else None,
        "warm_ttft_s_max": round(max(warm_samples), 3) if warm_samples else None,
        "concurrent": concurrent,
        "concurrent_ttft_s_median": (
            round(statistics.median(concurrent_samples), 3) if concurrent_samples else None
        ),
        "concurrent_ttft_s_max": (
            round(max(concurrent_samples), 3) if concurrent_samples else None
        ),
        "samples": len(warm_samples),
    }
