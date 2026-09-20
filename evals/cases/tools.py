"""Tool-call argument fidelity against strict schemas (SPEC §2, §3.1, §7).

This is the measurement the model choice hangs on. SPEC §2 prefers higher precision
over 4-bit precisely because quantization damage shows up first on tool-call argument
fidelity, so a candidate that is fast and wrong here is not a candidate.

Scored separately, because the failures mean different things:

* **schema violations** — an argument outside its enum or bounds, or an invented
  parameter. §7's whole point is that these declarations constrain the model; a model
  that ignores them makes the tool TOML decorative.
* **invented tools** — calling something that was never offered. The gateway denies
  and audits it (§3.7), so it is contained, but a model that does it often will also
  reach for T3 tools it cannot see.
* **false positives** — acting on a question. In FarmHub that means a valve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

DATA = Path(__file__).resolve().parent.parent / "data" / "tools.json"


def _violations(name: str, args: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> list[str]:
    """Where the arguments break the declared schema."""
    schema = schemas.get(name)
    if schema is None:
        return [f"unknown tool {name!r}"]
    properties: dict[str, Any] = schema.get("properties", {})
    problems: list[str] = []

    for key, value in args.items():
        spec = properties.get(key)
        if spec is None:
            problems.append(f"invented parameter {key!r}")
            continue
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{key}={value!r} outside enum {spec['enum']}")
        if spec.get("type") == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"{key}={value!r} is not an integer")
                continue
            if "minimum" in spec and value < spec["minimum"]:
                problems.append(f"{key}={value} below minimum {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                problems.append(f"{key}={value} above maximum {spec['maximum']}")

    for required in schema.get("required", []):
        if required not in args:
            problems.append(f"missing required {required!r}")
    return problems


async def run(base_url: str, model: str, timeout_s: float = 120.0) -> dict[str, Any]:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    tools = data["tools"]
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    offered = set(schemas)

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        for case in data["cases"]:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": case["utterance"]}],
                "tools": tools,
                "max_completion_tokens": 256,
            }
            record: dict[str, Any] = {"id": case["id"], "utterance": case["utterance"]}
            try:
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
            except Exception as exc:  # noqa: BLE001 - recorded as a failed case
                record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                results.append(record)
                continue

            calls = message.get("tool_calls") or []
            expected = case.get("expect_tool")

            if not calls:
                record.update(
                    called=None,
                    ok=expected is None,
                    note="no tool call" + ("" if expected is None else f"; expected {expected}"),
                )
                results.append(record)
                continue

            name = calls[0]["function"]["name"]
            raw = calls[0]["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
                parse_failed = not isinstance(args, dict)
                if parse_failed:
                    args = {}
            except json.JSONDecodeError:
                args, parse_failed = {}, True

            problems = _violations(name, args, schemas)
            record.update(
                called=name,
                args=args,
                arguments_parsed=not parse_failed,
                invented_tool=name not in offered,
                schema_violations=problems,
            )

            if expected is None:
                # Acting on a question, or inventing a tool for something not offered.
                record["ok"] = False
                record["note"] = "false positive: called a tool where none was expected"
            elif name != expected:
                record["ok"] = False
                record["note"] = f"called {name}, expected {expected}"
            elif case.get("expect_within_bounds"):
                record["ok"] = not problems
                record["note"] = "bounds honoured" if not problems else "bound broken"
            else:
                wanted = case.get("expect_args", {})
                mismatched = {k: (args.get(k), v) for k, v in wanted.items() if args.get(k) != v}
                record["ok"] = not problems and not mismatched
                if mismatched:
                    record["mismatched"] = mismatched
            results.append(record)

    total = len(results)
    return {
        "cases": total,
        "correct": sum(1 for r in results if r.get("ok")),
        "schema_violations": sum(1 for r in results if r.get("schema_violations")),
        "invented_tools": sum(1 for r in results if r.get("invented_tool")),
        "unparseable_arguments": sum(1 for r in results if r.get("arguments_parsed") is False),
        "false_positives": sum(1 for r in results if "false positive" in str(r.get("note", ""))),
        "score": round(sum(1 for r in results if r.get("ok")) / total, 3) if total else None,
        "detail": results,
    }
