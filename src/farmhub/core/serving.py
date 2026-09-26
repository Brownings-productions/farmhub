"""The bridge between a measured profile and the vLLM server's environment (SPEC §2).

A profile records what was measured. `deploy/vllm/.env` decides what is actually served.
Nothing connected the two, and they can disagree in a way no validation catches: drop
candidate A's ``--language-model-only`` and the vision tower loads, weights grow, the KV
cache shrinks, and the profile's ``weights_gib``/``kv_cache_gib`` become fiction while
config validation still passes — it is arithmetic over declared numbers, not a probe.
Worse, the eval harness rewrites that file for every candidate it runs, so after a matrix
run it holds the *last* candidate measured rather than the chosen one.

So the profile is the single source of truth and this module is the only place that knows
how it maps onto the server's environment:

* :func:`env_from_profile` renders the keys a profile determines;
* :func:`compare_env` says which of those keys the file on disk disagrees about.

The mapping mirrors ``deploy/vllm/compose.yaml``'s command. When a flag is added there,
it is added here, and :func:`compare_env` starts guarding it.

**Values are never reported.** A mismatch names the keys that differ and nothing else:
this file sits next to secrets, and the report reaches logs and a terminal.
"""

from collections.abc import Mapping
from pathlib import Path

from farmhub.core.config import ModelProfile, Settings
from farmhub.core.errors import ConfigError

# Where the serving environment lives, relative to the repository root. Configurable
# because hub and the dev PC may lay out deploy/ differently.
DEFAULT_ENV_PATH = Path("deploy/vllm/.env")

# Flags that compose reads from their own env key rather than from VLLM_EXTRA_ARGS.
# A profile carries them inside `server_args`, because that is how the eval harness
# recorded them; rendering splits them back out so compose does not pass them twice.
_FLAG_TO_KEY = {
    "--block-size": "VLLM_BLOCK_SIZE",
    "--max-num-seqs": "VLLM_MAX_NUM_SEQS",
    "--tool-call-parser": "VLLM_TOOL_CALL_PARSER",
}

# Keys this module owns. Everything else in the file — image tag, cache directory, port,
# bind address, the WSL2 pin-memory flag — belongs to the operator and is never touched
# or compared.
OWNED_KEYS = (
    "VLLM_MODEL",
    "VLLM_REVISION",
    "VLLM_SERVED_MODEL_NAME",
    "VLLM_QUANTIZATION",
    "VLLM_KV_CACHE_DTYPE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    *_FLAG_TO_KEY.values(),
    "VLLM_EXTRA_ARGS",
)


def env_from_profile(profile: ModelProfile, served_model_name: str) -> dict[str, str]:
    """The serving environment a profile determines.

    ``served_model_name`` is ``llm.model``: what vLLM advertises and what the client
    asks for. It is included because a mismatch there is the one failure the existing
    probe already catches, and it costs nothing to keep them in step.
    """
    env = {
        "VLLM_MODEL": profile.repo_id,
        "VLLM_REVISION": profile.revision,
        "VLLM_SERVED_MODEL_NAME": served_model_name,
        "VLLM_QUANTIZATION": profile.quantization,
        "VLLM_KV_CACHE_DTYPE": profile.kv_cache_dtype,
        "VLLM_MAX_MODEL_LEN": str(profile.max_model_len),
        "VLLM_GPU_MEMORY_UTILIZATION": str(profile.gpu_memory_utilization),
    }

    # Walk server_args once: flags compose reads separately go to their own key, the
    # rest keep their order in VLLM_EXTRA_ARGS. Order is preserved so the rendered file
    # can be compared with a previously rendered one character for character.
    extra: list[str] = []
    args = list(profile.server_args)
    index = 0
    while index < len(args):
        token = args[index]
        key = _FLAG_TO_KEY.get(token)
        if key is not None:
            if index + 1 >= len(args):
                raise ConfigError(
                    f"profile server_args ends with {token}, which takes a value "
                    "(SPEC §2: a profile records the configuration it was measured with)"
                )
            env[key] = args[index + 1]
            index += 2
            continue
        extra.append(token)
        index += 1
    env["VLLM_EXTRA_ARGS"] = " ".join(extra)

    # Defaults matching compose.yaml, so a comparison against a rendered file does not
    # trip over a key the profile happened not to mention.
    env.setdefault("VLLM_BLOCK_SIZE", "16")
    env.setdefault("VLLM_MAX_NUM_SEQS", "8")
    env.setdefault("VLLM_TOOL_CALL_PARSER", "hermes")
    return env


def parse_env_file(path: Path) -> dict[str, str]:
    """Read a dotenv file into a mapping, keeping only ``KEY=value`` lines.

    Deliberately small: compose's own interpolation rules are richer than this, but the
    keys this module owns are plain scalars, and a parser that guessed at quoting could
    report a mismatch that is not one.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read the serving environment {path}: {exc}") from exc

    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        # Strip one layer of matching quotes, which compose also does.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def render_env_file(
    profile: ModelProfile, served_model_name: str, existing: Mapping[str, str] | None = None
) -> str:
    """The text of a ``.env`` that serves ``profile``, keeping the operator's own keys.

    ``existing`` is the file being replaced. Its keys are preserved except the ones this
    module owns, so the image tag, cache directory, port, bind address and the WSL2
    pin-memory flag survive a profile switch — those describe the machine, not the model.
    """
    owned = env_from_profile(profile, served_model_name)
    kept = {k: v for k, v in (existing or {}).items() if k not in OWNED_KEYS}

    lines = [
        "# deploy/vllm/.env — generated by `farmhub vllm env --write`. Do not hand-edit:",
        f"# the model half is rendered from profile `{profile.measured_by}`, and",
        "# `farmhub config check` fails when this file and the active profile disagree.",
        "#",
        "# Keys above the marker are yours (image, cache, port, bind address, WSL flag).",
        "",
    ]
    lines += [f"{key}={value}" for key, value in kept.items()]
    lines += ["", "# --- rendered from the active profile; edit the profile instead ---"]
    lines += [f"{key}={owned[key]}" for key in OWNED_KEYS]
    return "\n".join(lines) + "\n"


def compare_env(profile: ModelProfile, served_model_name: str, path: Path) -> tuple[str, ...]:
    """The owned keys on which ``path`` disagrees with ``profile``.

    Empty means the server would be started with the configuration the profile was
    measured with. Only key names are returned: the caller puts them in a log line or a
    terminal, and this file sits beside secrets.
    """
    wanted = env_from_profile(profile, served_model_name)
    actual = parse_env_file(path)
    return tuple(key for key in OWNED_KEYS if actual.get(key, "") != wanted[key])


def verify_settings(settings: Settings) -> str | None:
    """Fail when the serving environment disagrees with the active profile.

    Returns a note when there was nothing to compare — no profile, the check switched
    off, or the file absent — so the caller can say so rather than implying a pass.
    Raises :class:`ConfigError` on a real mismatch, naming keys and no values.

    Called from ``farmhub config check`` and from server startup. It cannot stop someone
    editing the file and running ``vllm.sh up`` by hand; it stops FarmHub *serving*
    against a configuration its profile did not measure.
    """
    profile = settings.active_profile()
    if profile is None:
        return "no active profile, so nothing was compared"
    path = settings.llm.serving_env_file
    if path is None:
        return "llm.serving_env_file is unset, so nothing was compared"
    if not path.is_file():
        return f"{path} does not exist, so nothing was compared"

    differing = compare_env(profile, settings.llm.model, path)
    if differing:
        raise ConfigError(
            f"{path} does not match profile {settings.llm.profile!r}: "
            f"{', '.join(differing)} differ. vLLM would serve a configuration the "
            f"profile never measured, so its measured numbers would not describe it. "
            f"Regenerate with `farmhub vllm env --profile {settings.llm.profile} --write`"
        )
    return None
