"""Exact reproduction of part numbers and torque figures from supplied context.

SPEC §2: "quantization damage shows up first on exactly the things this system does:
part numbers, torque figures". This is the other half of the argument for higher
precision, and the reason FP8 KV cache is measured on quality rather than assumed.

Scored on exact string presence, not similarity. "SKF 6205-2RS" and "SKF 6250-2RS"
are equally fluent and one of them orders the wrong bearing.

The refusal case matters as much as the rest: a wrong torque figure is worse than no
answer, so a model that invents one when the context does not contain it is scored as
failing, not as merely unhelpful.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

DATA = Path(__file__).resolve().parent.parent / "data" / "grounding.json"

SYSTEM = (
    "Answer the question using only the supplied context. Quote part numbers, codes "
    "and figures exactly as they appear. If the context does not contain the answer, "
    "say so plainly and do not guess."
)

# What a refusal looks like in either language. Deliberately broad: the point is
# whether the model declined, not how gracefully.
REFUSALS = (
    "not in",
    "does not",
    "doesn't",
    "no information",
    "cannot",
    "can't",
    "unable",
    "ikke oppgitt",
    "ikke nevnt",
    "står ikke",
    "finner ikke",
    "vet ikke",
    "ingen informasjon",
)


async def run(base_url: str, model: str, timeout_s: float = 120.0) -> dict[str, Any]:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        for case in data["cases"]:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Context:\n{case['context']}\n\nQuestion: {case['question']}",
                    },
                ],
                "max_completion_tokens": 256,
            }
            record: dict[str, Any] = {"id": case["id"]}
            try:
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                answer = response.json()["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # noqa: BLE001 - recorded as a failed case
                record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                results.append(record)
                continue

            record["answer"] = answer.strip()[:300]
            lowered = answer.lower()

            if case.get("expect_refusal"):
                refused = any(phrase in lowered for phrase in REFUSALS)
                record.update(ok=refused, refused=refused)
                if not refused:
                    record["note"] = "invented an answer the context does not support"
                results.append(record)
                continue

            wanted = case.get("must_contain", [])
            if case.get("must_contain_any"):
                hit = any(w in answer for w in wanted)
            elif case.get("must_contain_all_lowercase"):
                hit = all(w.lower() in lowered for w in wanted)
            else:
                hit = all(w in answer for w in wanted)

            leaked = [w for w in case.get("must_not_contain", []) if w in answer]
            record.update(ok=hit and not leaked, matched=hit, wrong_figures=leaked)
            results.append(record)

    total = len(results)
    correct = sum(1 for r in results if r.get("ok"))
    return {
        "cases": total,
        "correct": correct,
        "wrong_figure_quoted": sum(1 for r in results if r.get("wrong_figures")),
        "invented_when_absent": sum(1 for r in results if r.get("refused") is False),
        "score": round(correct / total, 3) if total else None,
        "detail": results,
    }
