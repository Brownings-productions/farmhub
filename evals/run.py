"""Run the M1 model evaluation (SPEC §2, §11 M1).

    uv run python -m evals.run --list           # what would run, and nothing else
    uv run python -m evals.run --check          # validate pins and drift; no download
    uv run python -m evals.run                  # the whole matrix
    uv run python -m evals.run --only qwen36-35b-a3b-nvfp4 --kv-dtype auto

Needs the RTX 5090 and a real checkpoint, which is why it lives outside ``tests/``:
SPEC §9 says no test may require a GPU, and CI never runs this.

Each candidate is a full cycle: write the env, bring vLLM up, wait, read the startup
log for the memory numbers and the kernel actually selected, score it, bring it down.
A candidate that will not load is recorded as such and the run continues — on this
card "does it load at all" is a real result, not an error.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from evals import backend, report
from evals.cases import classifier, grounding, latency, tools
from evals.cases import common as cases_common
from evals.profile import Candidate, Matrix, ProfileError

DATA = Path(__file__).resolve().parent / "data" / "candidates.json"
LOGS = Path(__file__).resolve().parent / "results" / "logs"

SERVED_NAME = "farmhub-eval"
# Whisper, BGE-M3 and the reranker share the 5090 in Phase 1 (SPEC §2), the
# arrangement FarmHub runs on until hub exists. Held back from vLLM's share so the
# measured numbers describe the real arrangement, not a card with nothing else on it.
#
# This one figure is an ESTIMATE, unlike everything else the run records: no helper
# model has been loaded and measured yet (M3, M4 and M9 do that). It is carried in the
# results as such, and `nvidia-smi` readings taken while serving say how much the card
# really has left beside vLLM — which is the number that will confirm or refute it.
AUX_RESERVE_GIB = 5.0
AUX_RESERVE_IS_ESTIMATE = True
# SPEC §1 serves 2-3 concurrent voice users. The usable context is the KV cache
# divided by this, not the configured max_model_len.
CONCURRENT_SESSIONS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evals.run", description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the matrix and exit")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate pins and report upstream drift, without downloading anything",
    )
    parser.add_argument("--only", action="append", default=[], help="candidate name (repeatable)")
    parser.add_argument(
        "--kv-dtype",
        action="append",
        default=[],
        help=(
            "KV cache dtype to evaluate; repeatable. Default: auto only. fp8 was dropped "
            "from the M1 matrix (docs/DECISIONS.md 2026-09-25): at auto the KV cache "
            "already holds twice max_model_len per session, so fp8 would trade quality "
            "for room this system does not need. Pass --kv-dtype fp8 to measure it anyway."
        ),
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--load-timeout", type=float, default=1800.0)
    parser.add_argument("--card-total-gib", type=float, default=31.84)
    return parser.parse_args(argv)


def select(matrix: Matrix, only: list[str], kv_dtypes: list[str]) -> list[Candidate]:
    candidates = matrix.expand(kv_dtypes)
    if not only:
        return candidates
    chosen = [c for c in candidates if any(name in c.name for name in only)]
    if not chosen:
        known = ", ".join(sorted({c.name for c in candidates}))
        raise SystemExit(f"no candidate matches {only}. Known: {known}")
    return chosen


def check_pins(candidates: list[Candidate]) -> int:
    """Validate every pin, and say where a pin has fallen behind upstream.

    Run before anything downloads. An unpinned candidate produces numbers that
    describe weights nobody can identify afterwards, which is the whole reason SPEC §2
    requires a commit sha.
    """
    failures = 0
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.repo_id in seen:
            continue
        seen.add(candidate.repo_id)
        try:
            candidate.validate()
        except ProfileError as exc:
            print(f"REFUSED  {exc}")
            failures += 1
            continue
        # Advisory, but it saves a download: vLLM rejects a declared quantization that
        # the checkpoint disagrees with, and only after the weights are on disk.
        matches, reason = candidate.declared_quantization_matches()
        if not matches:
            print(f"MISMATCH {candidate.repo_id}: {reason}")
            failures += 1
            continue
        try:
            upstream = candidate.resolve_upstream_sha()
        except Exception as exc:  # noqa: BLE001 - reported, not fatal: may be offline
            print(f"?        {candidate.repo_id}: could not check upstream ({exc})")
            continue
        if upstream == candidate.revision:
            print(f"pinned   {candidate.repo_id} @ {candidate.revision[:12]} (current)")
        else:
            print(
                f"BEHIND   {candidate.repo_id} pinned at {candidate.revision[:12]}, "
                f"upstream is now {upstream[:12]}. Find out what changed before moving it."
            )
    return failures


async def score(base_url: str) -> dict[str, Any]:
    return {
        "latency": await latency.run(base_url, SERVED_NAME),
        "tools": await tools.run(base_url, SERVED_NAME),
        "grounding": await grounding.run(base_url, SERVED_NAME),
        "classifier": await classifier.run(base_url, SERVED_NAME),
    }


# Lines that name the kernel serving the *weights*. Everything else in the captured
# kernel lines — FlashInfer sampling, autotune, the attention backend, the non-default
# args echo — mentions kernel names too, and matching those reported the Mistral AWQ
# fallback as "flashinfer" when its weights ran through Marlin (2026-09-25).
QUANT_KERNEL = re.compile(
    r"(LinearKernel|LinearMethod|MoE backend|no native FP4|quant_algo)",
    re.I,
)
NOT_A_QUANT_KERNEL = re.compile(
    r"(top-p|top-k|sampling|autotune|Autotuner|kernel_warmup|attention backend|non-default args)",
    re.I,
)


def kernel_summary(startup: backend.Startup) -> str:
    """One word for what actually served the weights.

    A run can look like NVFP4 and not be (SPEC §2), so this reads only the lines that
    name a weight kernel, and says so in the fallback's own terms rather than the
    requested format's.
    """
    lines = [
        line
        for line in startup.kernel_lines
        if QUANT_KERNEL.search(line) and not NOT_A_QUANT_KERNEL.search(line)
    ]
    joined = " ".join(lines).lower()
    if not joined:
        return "unknown"
    # FP4 asked for, Marlin W4A16 delivered: the memory saving survives, the compute
    # does not. Reported first because it is the case that matters most here.
    if "no native fp4" in joined or ("marlin" in joined and "fp4" in joined):
        return "marlin-fallback"
    if "marlin" in joined and "awq" in joined:
        return "awq-marlin"
    if "marlin" in joined and ("gptq" in joined or "compressed" in joined):
        return "gptq-marlin"
    for token in ("modelopt", "cutlass", "marlin", "flashinfer", "awq", "gptq"):
        if token in joined:
            return token
    return "unknown"


def run_one(candidate: Candidate, args: argparse.Namespace) -> dict[str, Any]:
    """One full cycle for one candidate."""
    entry: dict[str, Any] = {"candidate": candidate.__dict__.copy(), "loaded": False}
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{candidate.name}.log"

    print(f"\n=== {candidate.name} ===")
    backend.write_env(candidate.env(SERVED_NAME))
    backend.down()

    # What the card already holds before vLLM starts. Not always zero: the dev PC's
    # monitor can be plugged into the 5090 rather than the motherboard, and a Windows
    # desktop on this card costs ~1.7 GiB (docs/MODEL_EVAL.md, 2026-09-25). Every
    # "in use while serving" reading includes it, so a run that does not record it
    # cannot be compared with one that ran against a different baseline.
    entry["baseline_gib"] = backend.memory_used_gib()
    if entry["baseline_gib"]:
        print(
            f"  baseline: {entry['baseline_gib']} GiB already on the card before vLLM "
            "started. WARNING: the memory readings below include it, and this run is not "
            "comparable with one taken on an idle card."
        )
    backend.up()

    serving = backend.wait_until_serving(args.port, args.load_timeout)
    log_text = backend.capture_log(log_path)
    startup = backend.parse_startup(log_text, log_path)
    entry["startup"] = startup.__dict__.copy()
    entry["kernel"] = kernel_summary(startup)

    if not serving:
        entry["error"] = "did not start; see the captured log"
        print(f"  did not load. log: {log_path}")
        backend.down()
        return entry

    entry["loaded"] = True
    entry["served_models"] = backend.served_models(args.port)
    # The outside view of the card, taken twice: once with the model loaded and idle,
    # once after the concurrent long-context case, which is the heaviest thing the run
    # does. The larger of the two is what "in use while serving" means.
    entry["memory_used_gib_idle"] = backend.memory_used_gib()
    entry["usable_context_at_concurrency"] = startup.usable_context_at(CONCURRENT_SESSIONS)
    print(
        f"  loaded: weights={startup.weights_gib} GiB kv={startup.kv_cache_gib} GiB "
        f"tokens={startup.kv_cache_tokens} kernel={entry['kernel']}"
    )

    try:
        entry.update(asyncio.run(score(f"http://127.0.0.1:{args.port}")))
    except Exception as exc:  # noqa: BLE001 - a scoring failure must not lose the run
        entry["scoring_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  scoring failed: {exc}")
    finally:
        # Read before the container goes down, or it measures an empty card.
        entry["memory_used_gib_after_scoring"] = backend.memory_used_gib()
        backend.down()

    readings = [
        value
        for value in (entry.get("memory_used_gib_idle"), entry.get("memory_used_gib_after_scoring"))
        if value is not None
    ]
    entry["memory_used_gib_while_serving"] = max(readings) if readings else None
    if entry["memory_used_gib_while_serving"] is not None:
        left = round(args.card_total_gib - entry["memory_used_gib_while_serving"], 2)
        entry["card_left_for_aux_gib"] = left
        # What would be left on an idle card, so a run taken with a desktop on the GPU
        # can still be read against one taken without. Reported beside the real figure,
        # never instead of it: the desktop was really there.
        baseline = entry.get("baseline_gib") or 0.0
        entry["card_left_for_aux_gib_less_baseline"] = round(left + baseline, 2)
        print(
            f"  card in use while serving: {entry['memory_used_gib_while_serving']} GiB, "
            f"leaving {left} GiB beside vLLM (aux_reserve estimate {AUX_RESERVE_GIB})"
        )
        if left < AUX_RESERVE_GIB:
            print(
                "  WARNING: less is left than aux_reserve_gib assumes, so the helper "
                "models do not fit beside this profile as configured."
            )
    entry["aux_reserve_gib"] = AUX_RESERVE_GIB
    entry["aux_reserve_is_estimate"] = AUX_RESERVE_IS_ESTIMATE
    # Recorded so a run states its own request settings: thinking was asked to be off,
    # and how many replies reasoned anyway despite that. A backend that ignores the
    # parameter cannot then pass as a clean result (docs/MODEL_EVAL.md).
    entry["chat_template_kwargs"] = dict(cases_common.CHAT_TEMPLATE_KWARGS)
    entry["temperature"] = cases_common.TEMPERATURE
    entry["server_args"] = candidate.server_args()
    entry["max_completion_tokens"] = {
        name: (entry.get(name) or {}).get("max_completion_tokens")
        for name in ("latency", "tools", "grounding", "classifier")
    }
    entry["reasoning_emitted"] = report.total(entry, "reasoning_emitted")
    entry["unparsed_tool_calls_suspected"] = report.total(entry, "unparsed_tool_calls_suspected")
    if entry["unparsed_tool_calls_suspected"]:
        print(
            f"  WARNING: {entry['unparsed_tool_calls_suspected']} repl(y|ies) look like a "
            "tool call the server's --tool-call-parser did not recognise. The tool score "
            "describes the parser, not the model."
        )
    entry["truncated"] = report.total(entry, "truncated")
    if entry["reasoning_emitted"]:
        print(
            f"  WARNING: {entry['reasoning_emitted']} repl(y|ies) still reasoned although "
            "enable_thinking=false was sent. Quality scores from this run are not comparable."
        )
    return entry


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # auto only: see docs/DECISIONS.md, 2026-09-25. fp8 stays available on request.
    kv_dtypes = args.kv_dtype or ["auto"]
    matrix = Matrix.load(DATA)
    candidates = select(matrix, args.only, kv_dtypes)

    if args.list:
        for candidate in candidates:
            print(f"{candidate.name:<34} {candidate.repo_id}@{candidate.revision[:12]}")
        return 0

    if args.check:
        return 1 if check_pins(candidates) else 0

    # Never download anything for a candidate that is not properly pinned (SPEC §2).
    refused = check_pins(candidates)
    if refused:
        print(f"\n{refused} candidate(s) refused. Nothing was downloaded.", file=sys.stderr)
        return 1

    identifier = report.run_id()
    result: dict[str, Any] = {
        "run_id": identifier,
        "environment": {"vllm_image": backend.image_tag(), **backend.driver_info()},
        "aux_reserve_gib": AUX_RESERVE_GIB,
        "aux_reserve_is_estimate": AUX_RESERVE_IS_ESTIMATE,
        "concurrent_sessions": CONCURRENT_SESSIONS,
        "card_total_gib": args.card_total_gib,
        # Sampling belongs in the run header: without it a score cannot be compared with
        # another run's, which is what the first two runs learned the hard way.
        "temperature": cases_common.TEMPERATURE,
        "tool_repeats": tools.REPEATS,
        "candidates": [],
    }

    for candidate in candidates:
        entry = run_one(candidate, args)
        entry["profile_toml"] = report.profile_toml(entry, identifier, args.card_total_gib)
        result["candidates"].append(entry)

    path = report.save(result, identifier)
    print(f"\nresults: {path}")
    print("\n--- paste into docs/MODEL_EVAL.md ---\n")
    print(report.markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
