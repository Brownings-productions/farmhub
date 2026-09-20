"""Turning a run into something a person and a config file can both use.

Two outputs, on purpose:

* a JSON result per run, kept whole so a later question can be answered without
  re-running anything;
* a ready-to-paste ``[profiles.*]`` block, because SPEC §2 requires a profile's
  numbers to be measured rather than estimated, and the surest way to keep that true
  is for no human ever to type them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"


def run_id(now: datetime | None = None) -> str:
    moment = now if now is not None else datetime.now(UTC)
    return "evals/" + moment.strftime("%Y-%m-%dT%H-%M-%SZ")


def save(result: dict[str, Any], identifier: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / (identifier.replace("evals/", "") + ".json")
    path.write_text(json.dumps(result, indent=2, sort_keys=False), encoding="utf-8")
    return path


def profile_toml(entry: dict[str, Any], identifier: str, card_total_gb: float) -> str:
    """The `[profiles.*]` block for one candidate, or a comment explaining why not.

    A candidate whose memory numbers could not be read does not get a block: an
    unmeasured profile is exactly what the §2 rule exists to prevent, and emitting one
    with a plausible guess would defeat it.
    """
    candidate = entry["candidate"]
    startup = entry.get("startup") or {}
    weights = startup.get("weights_gb")
    kv = startup.get("kv_cache_gb")
    name = candidate["name"].replace("-", "_")

    if weights is None or kv is None:
        return (
            f"# [profiles.{name}] not emitted: vLLM's startup log did not report "
            f"{'weights' if weights is None else 'kv cache'} size.\n"
            f"# See {startup.get('log_path', 'the captured log')} and fix the parser "
            "rather than filling these in by hand.\n"
        )

    return (
        f"[profiles.{name}]\n"
        f'repo_id = "{candidate["repo_id"]}"\n'
        f'revision = "{candidate["revision"]}"\n'
        f'quantization = "{candidate["quantization"]}"\n'
        f'kv_cache_dtype = "{candidate["kv_cache_dtype"]}"\n'
        f"gpu_memory_utilization = {candidate['gpu_memory_utilization']}\n"
        f"max_model_len = {candidate['max_model_len']}\n"
        f"weights_gb = {weights}\n"
        f"kv_cache_gb = {kv}\n"
        f"aux_reserve_gb = {entry.get('aux_reserve_gb', 5.0)}\n"
        f"card_total_gb = {card_total_gb}\n"
        f'measured_by = "{identifier}"\n'
    )


def markdown(result: dict[str, Any]) -> str:
    """A summary table plus the emitted profiles, for docs/MODEL_EVAL.md."""
    env = result["environment"]
    lines = [
        f"## Run {result['run_id']}",
        "",
        f"- vLLM image: `{env['vllm_image']}`",
        f"- GPU: {env['gpu']}, driver {env['driver']}, {env['memory_total']}",
        f"- Concurrent sessions assumed: {result['concurrent_sessions']}",
        f"- Auxiliary reservation: {result['aux_reserve_gb']} GB "
        "(Whisper, BGE-M3, reranker share the card on the single-GPU profile)",
        "",
        "| candidate | loaded | kernel | weights GB | KV GB | KV tokens | ctx @3 | "
        "TTFT cold/warm | tools | grounding | classifier JSON |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for entry in result["candidates"]:
        candidate = entry["candidate"]
        if not entry.get("loaded"):
            lines.append(f"| `{candidate['name']}` | **no** | — | — | — | — | — | — | — | — | — |")
            continue
        startup = entry.get("startup") or {}
        latency = entry.get("latency") or {}
        tools = entry.get("tools") or {}
        grounding = entry.get("grounding") or {}
        classifier = entry.get("classifier") or {}
        cold = latency.get("cold_ttft_s")
        warm = latency.get("warm_ttft_s_median")
        lines.append(
            f"| `{candidate['name']}` | yes | {entry.get('kernel', '?')} | "
            f"{startup.get('weights_gb', '?')} | {startup.get('kv_cache_gb', '?')} | "
            f"{startup.get('kv_cache_tokens', '?')} | "
            f"{entry.get('usable_context_at_concurrency', '?')} | "
            f"{cold if cold is not None else '?'} / {warm if warm is not None else '?'} | "
            f"{tools.get('score', '?')} | {grounding.get('score', '?')} | "
            f"{classifier.get('json_validity', '?')} |"
        )

    lines += ["", "### Emitted profiles", "", "```toml"]
    for entry in result["candidates"]:
        lines.append(entry.get("profile_toml", "").rstrip())
    lines += ["```", ""]

    failures = [e for e in result["candidates"] if not e.get("loaded")]
    if failures:
        lines += ["### Candidates that would not load", ""]
        for entry in failures:
            lines.append(
                f"- `{entry['candidate']['name']}`: {entry.get('error', 'see the captured log')}"
            )
        lines.append("")
    return "\n".join(lines)
