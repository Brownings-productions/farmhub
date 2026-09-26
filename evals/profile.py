"""What is being evaluated, and the rules a candidate must satisfy to be run.

The rules are enforced before anything is downloaded. A run that discovers a problem
after pulling 20 GB has wasted an hour; more importantly, an unpinned run produces
numbers that describe weights nobody can identify afterwards.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
HF_API = "https://huggingface.co/api/models/"


class ProfileError(Exception):
    """A candidate that must not be run as written."""


@dataclass(frozen=True)
class Candidate:
    """One model configuration to evaluate.

    ``revision`` is required and must be a commit sha. SPEC §2: a tag is not a pin.
    One NVFP4 upload of Qwen3.6-35B-A3B was silently replaced on 2026-07-10 with
    weights that produce looping output, and a tag would have followed it.
    """

    name: str
    repo_id: str
    revision: str
    quantization: str
    # Evaluated at both, never assumed: FP8 KV cache buys memory and has reported
    # quality collapse on this architecture (SPEC §2).
    kv_cache_dtype: str = "auto"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.82
    block_size: int = 16
    tool_call_parser: str = "hermes"
    # Model-specific flags, such as those that exclude a vision tower from a
    # multimodal checkpoint (SPEC §2: FarmHub runs text-only).
    extra_args: str = ""
    # True when extra_args is expected to keep the vision tower out, so the report can
    # say what was actually measured rather than what was intended.
    text_only: bool = False
    notes: str = ""

    def validate(self) -> None:
        """Refuse anything that would produce unattributable numbers."""
        if not COMMIT_SHA.match(self.revision):
            raise ProfileError(
                f"{self.name}: revision {self.revision!r} is not a 40-character commit "
                "sha. A tag or branch is not a pin (SPEC §2). Resolve it with "
                f"./deploy/vllm/vllm.sh resolve {self.repo_id}"
            )
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ProfileError(f"{self.name}: gpu_memory_utilization must be in (0, 1]")
        if self.max_model_len <= 0:
            raise ProfileError(f"{self.name}: max_model_len must be positive")

    def declared_quantization_matches(self, timeout_s: float = 20.0) -> tuple[bool, str]:
        """Compare the declared ``quantization`` with the checkpoint's own config.

        vLLM refuses a mismatch outright, and it refuses it *after* the weights are on
        disk: `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ` is a `compressed-tensors`
        checkpoint despite the AWQ in its name, and a declared `awq_marlin` cost a 17 GB
        download before the engine said so (2026-09-25). Reads the repository's
        `config.json` over the API, so `--check` can say it beforehand.

        Returns whether it matches and a human-readable reason. A config that cannot be
        read is reported as a match: this is a convenience check, not a gate, and the
        engine remains the authority.
        """
        url = f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/config.json"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                config: dict[str, Any] = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 - advisory only; the engine decides
            return True, f"could not read config.json ({exc})"
        method = (config.get("quantization_config") or {}).get("quant_method")
        if not isinstance(method, str):
            return True, "checkpoint declares no quantization_config"
        # vLLM's argument names are not the checkpoint's method names, so compare on the
        # family rather than the string: awq_marlin serves an "awq" checkpoint.
        family = self.quantization.split("_")[0].lower()
        if method.lower().startswith(family) or family.startswith(method.lower()):
            return True, f"checkpoint says {method}, declared {self.quantization}"
        return False, (
            f"checkpoint says quant_method={method!r} but the candidate declares "
            f"quantization={self.quantization!r}. vLLM rejects the mismatch after "
            "downloading the weights"
        )

    def resolve_upstream_sha(self, timeout_s: float = 20.0) -> str:
        """What the repository's default branch points at right now.

        Used to report drift: if this differs from ``revision``, the pin is holding
        something back, and it is worth knowing what before following it.
        """
        with urllib.request.urlopen(  # noqa: S310 - fixed https host
            HF_API + self.repo_id, timeout=timeout_s
        ) as response:
            body: dict[str, Any] = json.loads(response.read())
        sha = body.get("sha")
        if not isinstance(sha, str):
            raise ProfileError(f"{self.name}: could not read a sha for {self.repo_id}")
        return sha

    def server_args(self) -> list[str]:
        """The server flags this candidate runs with, beyond a profile's typed fields.

        What goes into a profile's ``server_args`` (SPEC §2), so the chosen profile can
        regenerate the exact serving environment it was measured with rather than
        inheriting whatever the last candidate left in ``deploy/vllm/.env``.

        ``--max-num-seqs`` is included although ``env()`` does not set it: compose
        defaults it to 8, that default was in force for every measurement, and a profile
        that does not say so would let it change silently.
        """
        args = [
            "--block-size",
            str(self.block_size),
            "--max-num-seqs",
            "8",
            "--tool-call-parser",
            self.tool_call_parser,
        ]
        return args + self.extra_args.split()

    def env(self, served_name: str = "farmhub-eval") -> dict[str, str]:
        """The deploy/vllm/.env this candidate corresponds to."""
        return {
            "VLLM_MODEL": self.repo_id,
            "VLLM_REVISION": self.revision,
            "VLLM_SERVED_MODEL_NAME": served_name,
            "VLLM_QUANTIZATION": self.quantization,
            "VLLM_KV_CACHE_DTYPE": self.kv_cache_dtype,
            "VLLM_MAX_MODEL_LEN": str(self.max_model_len),
            "VLLM_GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            "VLLM_BLOCK_SIZE": str(self.block_size),
            "VLLM_TOOL_CALL_PARSER": self.tool_call_parser,
            "VLLM_EXTRA_ARGS": self.extra_args,
        }


@dataclass
class Matrix:
    """The candidates to run, expanded over the KV cache dtypes."""

    candidates: list[Candidate] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Matrix:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls([Candidate(**entry) for entry in raw["candidates"]])

    def expand(self, kv_dtypes: list[str]) -> list[Candidate]:
        """One candidate per (candidate, kv dtype) pair.

        A candidate that already names a dtype other than the default is left alone:
        it was pinned deliberately.
        """
        out: list[Candidate] = []
        for candidate in self.candidates:
            for dtype in kv_dtypes:
                suffix = "" if dtype == "auto" else f"-kv{dtype}"
                out.append(
                    Candidate(
                        **{
                            **candidate.__dict__,
                            "name": f"{candidate.name}{suffix}",
                            "kv_cache_dtype": dtype,
                        }
                    )
                )
        return out
