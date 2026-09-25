"""Driving vLLM for one candidate, and reading what it says about itself.

The startup log is the only honest source for the memory numbers. vLLM prints what it
actually allocated after loading the weights and sizing the KV cache, which is exactly
the pair SPEC §2 requires a profile to declare.

It is also the only place that says which kernel was selected. That matters here more
than usual: on this card an NVFP4 checkpoint can silently fall back to Marlin W4A16,
so a run can look like NVFP4 without being it.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VLLM_DIR = REPO / "deploy" / "vllm"
COMPOSE = VLLM_DIR / "compose.yaml"
ENV_FILE = VLLM_DIR / ".env"

# Tolerant on purpose: vLLM rewords these between releases. A miss leaves the value
# None and the raw log is kept, so a missing number is visible rather than zero.
# Two wordings each, because v0.29.0 renamed both lines. Older releases say "model
# weights take X GiB"; v0.29.0 says "Model loading took X GiB memory". Likewise the KV
# cache: "Available KV cache memory: X GiB" is the v0.29.0 spelling.
WEIGHTS = re.compile(
    r"model weights take\s+([\d.]+)\s*GiB|model loading took\s+([\d.]+)\s*GiB", re.I
)
KV_CACHE_GIB = re.compile(
    r"KV cache (?:size|takes)\D+?([\d.]+)\s*GiB|available KV cache memory:\s*([\d.]+)\s*GiB",
    re.I,
)
KV_CACHE_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.I)
MAX_CONCURRENCY = re.compile(r"Maximum concurrency for\s+([\d,]+)\s+tokens.*?([\d.]+)x", re.I)

# Which kernel actually ran. The Marlin line is the one to watch for.
# A line that names loading the vision tower's weights, not merely mentioning vision.
VISION_WEIGHTS = re.compile(r"(loading|loaded).{0,40}(visual\.|vision tower)", re.I)

KERNEL_HINTS = (
    "marlin",
    "modelopt",
    "flashinfer",
    "cutlass",
    "no native fp4",
    "moe backend",
    "quantization",
    "compressed-tensors",
    "awq",
    "gptq",
)


@dataclass
class Startup:
    """What vLLM reported about itself while coming up."""

    weights_gib: float | None = None
    kv_cache_gib: float | None = None
    kv_cache_tokens: int | None = None
    max_concurrency: float | None = None
    kernel_lines: list[str] = field(default_factory=list)
    vision_tower_loaded: bool | None = None
    log_path: str = ""

    def usable_context_at(self, concurrent: int) -> int | None:
        """How long a context each of ``concurrent`` sessions can have.

        This is the number SPEC §2 actually cares about: not the configured
        ``max_model_len``, but what the KV cache supports with several satellites
        talking at once (§1: 2-3 concurrent voice users).
        """
        if self.kv_cache_tokens is None or concurrent <= 0:
            return None
        return self.kv_cache_tokens // concurrent


def write_env(values: dict[str, str], base: Path | None = None) -> None:
    """Write deploy/vllm/.env for one candidate, keeping the operator's own settings.

    The image tag, cache directory and port come from the existing .env; only the
    model-specific keys are replaced. That way the harness never silently changes
    which vLLM image is being measured.
    """
    source = (
        base if base is not None else (ENV_FILE if ENV_FILE.exists() else VLLM_DIR / "env.example")
    )
    kept: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        key = line.split("=", 1)[0].strip()
        if key in values:
            continue
        kept.append(line)
    body = "\n".join(kept).rstrip() + "\n\n# --- written by evals ---\n"
    body += "".join(f"{k}={v}\n" for k, v in values.items())
    ENV_FILE.write_text(body, encoding="utf-8")


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "compose", "--env-file", str(ENV_FILE), "-f", str(COMPOSE), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def image_tag() -> str:
    """The pinned vLLM image actually in use, for the record."""
    result = _compose("config", "--images")
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"


def driver_info() -> dict[str, str]:
    """Driver and GPU, so a later run can be compared with this one."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"gpu": "unknown", "driver": "unknown", "memory_total": "unknown"}
    name, driver, memory = (part.strip() for part in result.stdout.strip().split(",")[:3])
    return {"gpu": name, "driver": driver, "memory_total": memory}


MIB_PER_GIB = 1024.0


def memory_used_gib() -> float | None:
    """What the card actually holds right now, in GiB, or None if unreadable.

    vLLM's startup log is a self-report: it accounts for its own share and knows
    nothing about the desktop, the helper models, or anything else on the card. This is
    the outside view, and the difference between it and `card_total_gib` is what the
    auxiliary models really have left (SPEC §2, Phase 1).
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return round(float(result.stdout.strip().splitlines()[0]) / MIB_PER_GIB, 2)
    except ValueError:
        return None


def down() -> None:
    _compose("down", "--remove-orphans")


def up() -> None:
    _compose("up", "-d")


def wait_until_serving(port: int, timeout_s: float) -> bool:
    """Block until /health answers, or the container exits."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if not _compose("ps", "-q", "vllm").stdout.strip():
            return False
        time.sleep(5)
    return False


def capture_log(destination: Path) -> str:
    result = _compose("logs", "--no-color", "vllm")
    text = result.stdout + result.stderr
    destination.write_text(text, encoding="utf-8")
    return text


def parse_startup(log: str, log_path: Path) -> Startup:
    """Mine the startup log for the numbers a profile has to declare."""
    startup = Startup(log_path=str(log_path))

    if match := WEIGHTS.search(log):
        startup.weights_gib = float(match.group(1) or match.group(2))
    if match := KV_CACHE_GIB.search(log):
        startup.kv_cache_gib = float(match.group(1) or match.group(2))
    if match := KV_CACHE_TOKENS.search(log):
        startup.kv_cache_tokens = int(match.group(1).replace(",", ""))
    if match := MAX_CONCURRENCY.search(log):
        startup.max_concurrency = float(match.group(2))

    startup.kernel_lines = sorted(
        {
            line.strip()
            for line in log.splitlines()
            if any(hint in line.lower() for hint in KERNEL_HINTS)
        }
    )[:40]
    # Deliberately narrow. The first version of this looked for "vision" anywhere in
    # the log and so reported the tower loaded for every run: vLLM names the encoder's
    # attention backend while it is picking backends, whether or not the tower is then
    # built. Only a line that names loading visual *weights* counts, and anything else
    # leaves this None — unknown rather than a confident wrong answer. To settle it
    # properly, load once with and once without --language-model-only and compare the
    # reported weights size.
    if VISION_WEIGHTS.search(log):
        startup.vision_tower_loaded = True
    return startup


def served_models(port: int, timeout_s: float = 10.0) -> list[str]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/models", timeout=timeout_s
    ) as response:
        body = json.loads(response.read())
    return [entry["id"] for entry in body.get("data", [])]
