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
# Whisper, BGE-M3 and the reranker share the 5090 on the default single-GPU profile
# (SPEC §2). Held back from vLLM's share so the measured numbers describe the real
# arrangement, not a card with nothing else on it.
AUX_RESERVE_GB = 5.0
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
        help="KV cache dtype to evaluate; repeatable. Default: auto and fp8 (SPEC §2)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--load-timeout", type=float, default=1800.0)
    parser.add_argument("--card-total-gb", type=float, default=31.8)
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


def kernel_summary(startup: backend.Startup) -> str:
    """One word for what actually ran, because a run can look like NVFP4 and not be."""
    joined = " ".join(startup.kernel_lines).lower()
    if "no native fp4" in joined or ("marlin" in joined and "fp4" in joined):
        return "marlin-fallback"
    for token in ("modelopt", "flashinfer", "cutlass", "marlin", "awq", "gptq"):
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
    entry["usable_context_at_concurrency"] = startup.usable_context_at(CONCURRENT_SESSIONS)
    print(
        f"  loaded: weights={startup.weights_gb} GB kv={startup.kv_cache_gb} GB "
        f"tokens={startup.kv_cache_tokens} kernel={entry['kernel']}"
    )

    try:
        entry.update(asyncio.run(score(f"http://127.0.0.1:{args.port}")))
    except Exception as exc:  # noqa: BLE001 - a scoring failure must not lose the run
        entry["scoring_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  scoring failed: {exc}")
    finally:
        backend.down()

    entry["aux_reserve_gb"] = AUX_RESERVE_GB
    # Recorded so a run states its own request settings: thinking was asked to be off,
    # and how many replies reasoned anyway despite that. A backend that ignores the
    # parameter cannot then pass as a clean result (docs/MODEL_EVAL.md).
    entry["chat_template_kwargs"] = dict(cases_common.CHAT_TEMPLATE_KWARGS)
    entry["max_completion_tokens"] = {
        name: (entry.get(name) or {}).get("max_completion_tokens")
        for name in ("latency", "tools", "grounding", "classifier")
    }
    entry["reasoning_emitted"] = report.total(entry, "reasoning_emitted")
    entry["truncated"] = report.total(entry, "truncated")
    if entry["reasoning_emitted"]:
        print(
            f"  WARNING: {entry['reasoning_emitted']} repl(y|ies) still reasoned although "
            "enable_thinking=false was sent. Quality scores from this run are not comparable."
        )
    return entry


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kv_dtypes = args.kv_dtype or ["auto", "fp8"]
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
        "aux_reserve_gb": AUX_RESERVE_GB,
        "concurrent_sessions": CONCURRENT_SESSIONS,
        "card_total_gb": args.card_total_gb,
        "candidates": [],
    }

    for candidate in candidates:
        entry = run_one(candidate, args)
        entry["profile_toml"] = report.profile_toml(entry, identifier, args.card_total_gb)
        result["candidates"].append(entry)

    path = report.save(result, identifier)
    print(f"\nresults: {path}")
    print("\n--- paste into docs/MODEL_EVAL.md ---\n")
    print(report.markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
