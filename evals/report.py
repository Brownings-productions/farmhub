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


def profile_toml(entry: dict[str, Any], identifier: str, card_total_gib: float) -> str:
    """The `[profiles.*]` block for one candidate, or a comment explaining why not.

    A candidate whose memory numbers could not be read does not get a block: an
    unmeasured profile is exactly what the §2 rule exists to prevent, and emitting one
    with a plausible guess would defeat it.
    """
    candidate = entry["candidate"]
    startup = entry.get("startup") or {}
    weights = startup.get("weights_gib")
    kv = startup.get("kv_cache_gib")
    temperature = entry.get("temperature")
    name = candidate["name"].replace("-", "_")

    if weights is None or kv is None:
        return (
            f"# [profiles.{name}] not emitted: vLLM's startup log did not report "
            f"{'weights' if weights is None else 'kv cache'} size.\n"
            f"# See {startup.get('log_path', 'the captured log')} and fix the parser "
            "rather than filling these in by hand.\n"
        )
    if temperature is None:
        # A profile's temperature is what the run sampled at, so a run that did not
        # record it cannot produce one. Defaulting here would put a number in config
        # that no score behind it was measured at.
        return (
            f"# [profiles.{name}] not emitted: the run recorded no temperature, so the "
            "scores above were taken at the backend's default.\n"
        )

    return (
        f"[profiles.{name}]\n"
        f'repo_id = "{candidate["repo_id"]}"\n'
        f'revision = "{candidate["revision"]}"\n'
        f'quantization = "{candidate["quantization"]}"\n'
        f'kv_cache_dtype = "{candidate["kv_cache_dtype"]}"\n'
        f"gpu_memory_utilization = {candidate['gpu_memory_utilization']}\n"
        f"max_model_len = {candidate['max_model_len']}\n"
        f"weights_gib = {weights}\n"
        f"kv_cache_gib = {kv}\n"
        f"aux_reserve_gib = {entry.get('aux_reserve_gib', 5.0)}\n"
        f"card_total_gib = {card_total_gib}\n"
        f'measured_by = "{identifier}"\n'
        # Emitted from what the run actually sent, so the profile serves the model the
        # way it was measured. A profile that dropped this would score differently from
        # the run that produced its numbers (docs/MODEL_EVAL.md).
        f"chat_template_kwargs = {_toml_inline(entry.get('chat_template_kwargs') or {})}\n"
        # Likewise emitted from the request, not from a constant: the profile has to
        # sample the way the run that scored it sampled.
        f"temperature = {temperature}\n"
    )


def _toml_inline(table: dict[str, Any]) -> str:
    """A TOML inline table, for the small flat maps a profile carries."""
    if not table:
        return "{}"
    pairs = ", ".join(f"{k} = {json.dumps(v)}" for k, v in table.items())
    return f"{{ {pairs} }}"


def _baseline(entry: dict[str, Any]) -> str:
    """What the card held before vLLM started, flagged when it is not zero."""
    baseline = entry.get("baseline_gib")
    if baseline is None:
        return "?"
    return "0" if not baseline else f"**{baseline}**"


def _pair(single: object, loaded: object) -> str:
    """One cell for "alone / under load", so the gap is read at a glance."""
    left = single if single is not None else "?"
    right = loaded if loaded is not None else "?"
    return f"{left} / {right}"


def _range(tools: dict[str, Any]) -> str:
    """The tool score across repeated passes, as a range.

    One number would hide that the score moves between passes, which is what made the
    first two runs look like they disagreed about the model.
    """
    low, high = tools.get("score_min"), tools.get("score_max")
    if low is None or high is None:
        return str(tools.get("score", "?"))
    return str(low) if low == high else f"{low} to {high}"


def _buckets(tools: dict[str, Any]) -> str:
    """Out-of-enum outcomes as declined / substituted / invented.

    The middle figure is the one to watch: a schema-valid call in the wrong area.
    """
    buckets = tools.get("out_of_enum") or {}
    if not buckets:
        return "—"
    return "/".join(str(buckets.get(name, 0)) for name in ("declined", "substituted", "invented"))


def total(entry: dict[str, Any], key: str) -> int:
    """Sum a per-case counter across every scoring case in one run."""
    return sum(
        int((entry.get(case) or {}).get(key) or 0)
        for case in ("latency", "tools", "grounding", "classifier")
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
        f"- Auxiliary reservation: {result['aux_reserve_gib']} GiB "
        "(Whisper, BGE-M3, reranker share the card in Phase 1, SPEC §2)"
        + (
            " — **an estimate; no helper model has been loaded and measured yet**"
            if result.get("aux_reserve_is_estimate")
            else ""
        ),
        f"- Sampling: temperature {result.get('temperature', '?')}; tool cases repeated "
        f"{result.get('tool_repeats', '?')}x",
        "",
        "| candidate | loaded | kernel | baseline GiB | weights GiB | KV GiB | "
        "KV tokens | ctx @3 | card used GiB | left for aux GiB | TTFT cold/warm | "
        "answer s 1/3 | gen tok/s 1/3 | RAG answer s 1/3 | tools min-max | "
        "out-of-enum d/s/i | grounding | classifier JSON | reasoning | truncated |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for entry in result["candidates"]:
        candidate = entry["candidate"]
        if not entry.get("loaded"):
            lines.append(f"| `{candidate['name']}` | **no** |" + " — |" * 18)
            continue
        startup = entry.get("startup") or {}
        latency = entry.get("latency") or {}
        tools = entry.get("tools") or {}
        grounding = entry.get("grounding") or {}
        classifier = entry.get("classifier") or {}
        cold = latency.get("cold_ttft_s")
        warm = latency.get("warm_ttft_s_median")
        single = latency.get("single") or {}
        loaded = latency.get("at_concurrency") or {}
        long_single = latency.get("long_context_single") or {}
        long_loaded = latency.get("long_context_at_concurrency") or {}
        # Reasoning and truncation are reported per run, not per case: a run where the
        # model reasoned anyway has no comparable quality score at all, so it has to be
        # visible in the same table as the scores it invalidates.
        rate_loaded = loaded.get("gen_tokens_per_s_median")
        reasoning = total(entry, "reasoning_emitted")
        truncations = total(entry, "truncated")
        lines.append(
            f"| `{candidate['name']}` | yes | {entry.get('kernel', '?')} | "
            # A non-zero baseline means every memory figure in this row includes
            # something that is not vLLM — usually a desktop on the same card.
            f"{_baseline(entry)} | "
            f"{startup.get('weights_gib', '?')} | {startup.get('kv_cache_gib', '?')} | "
            f"{startup.get('kv_cache_tokens', '?')} | "
            f"{entry.get('usable_context_at_concurrency', '?')} | "
            # The outside view: nvidia-smi while serving, and what that leaves for the
            # helper models beside vLLM. vLLM's own log cannot see either.
            f"{entry.get('memory_used_gib_while_serving', '?')} | "
            f"{entry.get('card_left_for_aux_gib', '?')} | "
            f"{cold if cold is not None else '?'} / {warm if warm is not None else '?'} | "
            f"{_pair(single.get('answer_s_median'), loaded.get('answer_s_median'))} | "
            f"{_pair(single.get('gen_tokens_per_s_median'), rate_loaded)} | "
            f"{_pair(long_single.get('answer_s_median'), long_loaded.get('answer_s_median'))} | "
            f"{_range(tools)} | {_buckets(tools)} | {grounding.get('score', '?')} | "
            f"{classifier.get('json_validity', '?')} | "
            f"{reasoning} | {truncations} |"
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
