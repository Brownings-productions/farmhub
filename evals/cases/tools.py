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
import statistics
from pathlib import Path
from typing import Any

import httpx

from evals.cases import common

DATA = Path(__file__).resolve().parent.parent / "data" / "tools.json"

# A tool call is short, but the budget has to cover whatever precedes it. Recorded in
# the run alongside how many replies were cut off.
MAX_TOKENS = 512

# Every case is run this many times. With temperature pinned at 0 the replies should be
# identical, but "should be" is what the first two runs assumed: one case passed in one
# and failed in the next. Three passes make instability visible instead of deciding the
# score by luck, and the reported score is the worst of them.
REPEATS = 3


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


def _out_of_enum_bucket(
    case: dict[str, Any], record: dict[str, Any], schemas: dict[str, dict[str, Any]]
) -> str | None:
    """Which of three things the model did with a value that is not in the enum.

    The distinction is the point. "Turn on the light in the barn" produced
    `set_indoor_light(area="workshop")` — schema-valid, and the wrong room. Scored as a
    plain false positive that looked identical to inventing `area="barn"`, which the
    gateway would have rejected outright. They are not the same failure:

    * ``declined`` — no call. Correct.
    * ``substituted`` — called with a *different valid* enum value. The dangerous one:
      it passes validation and acts somewhere the person did not ask about.
    * ``invented`` — called with a value outside the enum. Wrong, but stopped by §7's
      schema and audited by the gateway (§3.7).
    """
    parameter = case.get("out_of_enum_param")
    if parameter is None:
        return None
    if record.get("called") is None:
        return "declined"
    schema = schemas.get(str(record["called"]))
    allowed = ((schema or {}).get("properties", {}).get(parameter) or {}).get("enum")
    if allowed is None:
        return "invented"
    value = (record.get("args") or {}).get(parameter)
    return "substituted" if value in allowed else "invented"


async def _one_case(
    client: httpx.AsyncClient,
    model: str,
    case: dict[str, Any],
    tools: list[dict[str, Any]],
    schemas: dict[str, dict[str, Any]],
    offered: set[str],
) -> dict[str, Any]:
    """One utterance, one reply, scored."""
    body = common.payload(
        model,
        [{"role": "user", "content": case["utterance"]}],
        max_completion_tokens=MAX_TOKENS,
        tools=tools,
    )
    record: dict[str, Any] = {"id": case["id"], "utterance": case["utterance"]}
    try:
        response = await client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        message = choice["message"]
    except Exception as exc:  # noqa: BLE001 - recorded as a failed case
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        return record

    record["truncated"] = common.truncated(choice)
    record["reasoning"] = common.reasoning_in(message)
    calls = message.get("tool_calls") or []
    expected = case.get("expect_tool")

    if not calls:
        record.update(
            called=None,
            ok=expected is None,
            note="no tool call" + ("" if expected is None else f"; expected {expected}"),
        )
        record["out_of_enum"] = _out_of_enum_bucket(case, record, schemas)
        return record

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
    record["out_of_enum"] = _out_of_enum_bucket(case, record, schemas)
    return record


def _pass_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The counts for one pass over every case."""
    total = len(results)
    return {
        "cases": total,
        "correct": sum(1 for r in results if r.get("ok")),
        "schema_violations": sum(1 for r in results if r.get("schema_violations")),
        "invented_tools": sum(1 for r in results if r.get("invented_tool")),
        "unparseable_arguments": sum(1 for r in results if r.get("arguments_parsed") is False),
        "false_positives": sum(1 for r in results if "false positive" in str(r.get("note", ""))),
        "truncated": sum(1 for r in results if r.get("truncated")),
        "reasoning_emitted": sum(1 for r in results if r.get("reasoning")),
        "out_of_enum": {
            bucket: sum(1 for r in results if r.get("out_of_enum") == bucket)
            for bucket in ("declined", "substituted", "invented")
        },
        "score": round(sum(1 for r in results if r.get("ok")) / total, 3) if total else None,
    }


async def run(
    base_url: str, model: str, timeout_s: float = 120.0, repeats: int = REPEATS
) -> dict[str, Any]:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    tools = data["tools"]
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    offered = set(schemas)
    cases: list[dict[str, Any]] = data["cases"]

    passes: list[list[dict[str, Any]]] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        for _ in range(repeats):
            passes.append(
                [await _one_case(client, model, case, tools, schemas, offered) for case in cases]
            )

    summaries = [_pass_summary(results) for results in passes]
    scores = [s["score"] for s in summaries if s["score"] is not None]
    # How often each case passed, so a case that is merely unstable is distinguishable
    # from one that always fails.
    per_case = {
        case["id"]: sum(1 for results in passes if results[index].get("ok"))
        for index, case in enumerate(cases)
    }
    return {
        "max_completion_tokens": MAX_TOKENS,
        "repeats": repeats,
        "score": min(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_median": round(statistics.median(scores), 3) if scores else None,
        "passes_per_case": per_case,
        "unstable_cases": sorted(cid for cid, hits in per_case.items() if 0 < hits < repeats),
        # Summed across passes, so these stay comparable with the counts each pass gives.
        "cases": sum(s["cases"] for s in summaries),
        "correct": sum(s["correct"] for s in summaries),
        "schema_violations": sum(s["schema_violations"] for s in summaries),
        "invented_tools": sum(s["invented_tools"] for s in summaries),
        "unparseable_arguments": sum(s["unparseable_arguments"] for s in summaries),
        "false_positives": sum(s["false_positives"] for s in summaries),
        "truncated": sum(s["truncated"] for s in summaries),
        "reasoning_emitted": sum(s["reasoning_emitted"] for s in summaries),
        "out_of_enum": {
            bucket: sum(s["out_of_enum"][bucket] for s in summaries)
            for bucket in ("declined", "substituted", "invented")
        },
        "per_pass": summaries,
        "detail": passes[0] if passes else [],
        "detail_all_passes": passes,
    }
