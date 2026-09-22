"""Structured-output validity for the §4 classifier.

The classifier is a tool-less completion constrained to
``{"kind": "question" | "action" | "mixed"}``. SPEC §4 says a failure to parse, a
failed validation or a timeout all route to `question`, so an invalid response is
never dangerous — it is merely useless, and a model that produces them often makes the
whole two-tier classifier pointless.

Validity is therefore scored before correctness, and reported separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

DATA = Path(__file__).resolve().parent.parent / "data" / "classifier.json"
KINDS = {"question", "action", "mixed"}


async def run(base_url: str, model: str, timeout_s: float = 120.0) -> dict[str, Any]:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    schema = data["schema"]
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        for case in data["cases"]:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": data["system"]},
                    {"role": "user", "content": case["utterance"]},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "classification", "schema": schema, "strict": True},
                },
                "max_completion_tokens": 64,
            }
            record: dict[str, Any] = {"id": case["id"], "expect": case["expect"]}
            try:
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # noqa: BLE001 - recorded as an invalid case
                record.update(valid=False, ok=False, error=f"{type(exc).__name__}: {exc}")
                results.append(record)
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                record.update(valid=False, ok=False, raw=content[:200], note="not JSON")
                results.append(record)
                continue

            valid = (
                isinstance(parsed, dict) and set(parsed) == {"kind"} and parsed.get("kind") in KINDS
            )
            record.update(
                valid=valid,
                got=parsed.get("kind") if isinstance(parsed, dict) else None,
                ok=valid and parsed.get("kind") == case["expect"],
            )
            if not valid:
                record["raw"] = content[:200]
            results.append(record)

    total = len(results)
    valid = sum(1 for r in results if r.get("valid"))
    return {
        "cases": total,
        "valid_json": valid,
        "json_validity": round(valid / total, 3) if total else None,
        "correct": sum(1 for r in results if r.get("ok")),
        "accuracy": round(sum(1 for r in results if r.get("ok")) / total, 3) if total else None,
        "detail": results,
    }
