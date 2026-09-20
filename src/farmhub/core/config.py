"""Layered configuration (SPEC §7).

Precedence, highest first: CLI overrides > ``FARMHUB_*`` environment > TOML file > code
defaults. Invalid config raises ``ConfigError``; callers exit rather than continue.

Safety-relevant settings may have fail-safe defaults in code (docs/DECISIONS.md,
"Safety-relevant defaults"), but their effective values are always logged at startup
and printed by ``farmhub config check``.
"""

import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from farmhub.core.errors import ConfigError

DEFAULT_CONFIG_PATH = Path("config/farmhub.toml")

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path | None = Path("logs/farmhub.jsonl")


class ModelProfile(BaseModel):
    """One evaluated LLM serving profile (SPEC §2).

    Every field here is a fact measured on the real card, not a guess: the M1
    evaluation emits this block and ``measured_by`` names the run that produced it.
    A profile assembled by hand is rejected, because the whole point of the budget
    check below is that the numbers it adds up are real.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_id: str
    # SPEC §2: a tag or branch is not a pin. One NVFP4 upload of Qwen3.6-35B-A3B was
    # silently replaced with weights that loop, and a tag would have followed it.
    revision: str
    quantization: str
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    max_model_len: int = Field(gt=0)
    weights_gb: float = Field(gt=0.0)
    kv_cache_gb: float = Field(gt=0.0)
    aux_reserve_gb: float = Field(ge=0.0)
    card_total_gb: float = Field(gt=0.0)
    # The evals run id that measured weights_gb and kv_cache_gb.
    measured_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "ModelProfile":
        if not _COMMIT_SHA.match(self.revision):
            raise ValueError(
                f"revision must be a 40-character commit sha, got {self.revision!r}. "
                "A tag or branch is not a pin (SPEC §2)"
            )
        total = self.weights_gb + self.kv_cache_gb + self.aux_reserve_gb
        if total > self.card_total_gb:
            raise ValueError(
                f"budget exceeds the card: weights {self.weights_gb} + kv "
                f"{self.kv_cache_gb} + aux {self.aux_reserve_gb} = {total:.2f} GB > "
                f"{self.card_total_gb} GB"
            )
        free_for_aux = (1.0 - self.gpu_memory_utilization) * self.card_total_gb
        if free_for_aux < self.aux_reserve_gb:
            raise ValueError(
                f"gpu_memory_utilization {self.gpu_memory_utilization} leaves "
                f"{free_for_aux:.2f} GB free, less than aux_reserve_gb "
                f"{self.aux_reserve_gb}. vLLM would take memory the auxiliary models "
                "need (SPEC §2)"
            )
        vllm_share = self.gpu_memory_utilization * self.card_total_gb
        if self.weights_gb + self.kv_cache_gb > vllm_share:
            raise ValueError(
                f"weights + kv cache ({self.weights_gb + self.kv_cache_gb:.2f} GB) "
                f"exceed vLLM's own share ({vllm_share:.2f} GB) at "
                f"gpu_memory_utilization {self.gpu_memory_utilization}"
            )
        return self


class LlmSettings(BaseModel):
    """How to reach the OpenAI-compatible backend (SPEC §8.2).

    Only a base URL and a model name, so vLLM, Ollama and a test double are
    interchangeable without a code change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "http://127.0.0.1:8000/v1"
    # Must match vLLM's --served-model-name, or Ollama's model tag.
    model: str = "farmhub-primary"
    # vLLM ignores it, but the OpenAI client refuses to construct without one.
    api_key: str = "not-used"
    # Names a [profiles.<name>] block. Optional: a test double or Ollama has no
    # measured profile, and requiring one would make the mock path impossible.
    profile: str | None = None
    request_timeout_s: float = Field(default=60.0, gt=0.0)
    connect_timeout_s: float = Field(default=5.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)
    # How often the registry re-probes the backend. vLLM is not always running
    # (docs/RUNBOOK.md), so this is what notices it came back.
    health_interval_s: float = Field(default=30.0, gt=0.0)


class ApiSettings(BaseModel):
    """The HTTP listener for the HA conversation agent (SPEC §3.6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Fail-safe default. Production and the dev HA container both set this
    # explicitly, and a non-loopback value is logged as a warning at startup.
    bind_host: str = "127.0.0.1"
    port: int = Field(default=8099, gt=0, lt=65536)
    # The service-to-service bearer token (§3.6). It has no safe default, so it is
    # required before the server starts; `config check` reports only whether it is
    # set. Prefer token_file or FARMHUB_API__TOKEN over writing it into the TOML.
    token: str | None = None
    token_file: Path | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "ApiSettings":
        if self.token is not None and self.token_file is not None:
            raise ValueError("set api.token or api.token_file, not both")
        return self


class SatellitesSettings(BaseModel):
    """Where the satellite registry lives (SPEC §8.8; the full shape lands at M9)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Absent means no satellite is known, so every request is served T0-only and the
    # condition is logged (§3.6 fail closed). That is a safe default, so it is not
    # required here, but it is never silent.
    registry: Path | None = None


class Settings(BaseSettings):
    """Application settings. Unknown keys in the TOML file or overrides are rejected."""

    model_config = SettingsConfigDict(
        env_prefix="FARMHUB_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    # Dry-run is ON unless something explicitly turns it off (SPEC §3.8). The code
    # default never changes; production config sets it off, and startup warns when off.
    dry_run: bool = True
    logging: LoggingSettings = LoggingSettings()
    llm: LlmSettings = LlmSettings()
    api: ApiSettings = ApiSettings()
    satellites: SatellitesSettings = SatellitesSettings()
    profiles: dict[str, ModelProfile] = {}

    @model_validator(mode="after")
    def _profile_exists(self) -> "Settings":
        if self.llm.profile is not None and self.llm.profile not in self.profiles:
            known = ", ".join(sorted(self.profiles)) or "none"
            raise ValueError(
                f"llm.profile is {self.llm.profile!r} but no [profiles.{self.llm.profile}] "
                f"block is defined (known profiles: {known})"
            )
        return self

    def active_profile(self) -> ModelProfile | None:
        """The measured serving profile in force, if one is selected."""
        return self.profiles.get(self.llm.profile) if self.llm.profile else None

    def resolve_api_token(self) -> str | None:
        """The bearer token, read from ``token_file`` when that is what is configured.

        Returns None when no token is configured at all. Callers that need one
        (the server) fail; ``config check`` only reports whether it is set, and never
        prints the value.
        """
        if self.api.token is not None:
            return self.api.token
        if self.api.token_file is None:
            return None
        try:
            token = self.api.token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"cannot read api.token_file {self.api.token_file}: {exc}") from exc
        if not token:
            raise ConfigError(f"api.token_file {self.api.token_file} is empty")
        return token

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order is precedence: init (CLI) > env > TOML. No dotenv or secrets files."""
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))

    def safety_summary(self) -> dict[str, object]:
        """The effective safety-relevant settings, for startup logs and ``config check``.

        Never include the bearer token itself, only whether one is configured: this
        dict is written to the log file and printed to a terminal.
        """
        return {
            "dry_run": self.dry_run,
            "api_bind_host": self.api.bind_host,
            "api_port": self.api.port,
            "api_token_configured": self.api.token is not None or self.api.token_file is not None,
            "satellite_registry": str(self.satellites.registry or "<none: every request is T0>"),
            "llm_base_url": self.llm.base_url,
            "llm_profile": self.llm.profile or "<none>",
        }

    def safety_warnings(self) -> list[tuple[str, str]]:
        """Conditions that are allowed but must never pass unnoticed (SPEC §7).

        Each is an event name and a detail, logged as a warning at startup and
        printed by ``config check``.
        """
        warnings: list[tuple[str, str]] = []
        if not self.dry_run:
            warnings.append(("dry_run_disabled", "tools at T1+ may reach Home Assistant"))
        if self.api.bind_host not in ("127.0.0.1", "::1", "localhost"):
            warnings.append(
                (
                    "api_bind_not_loopback",
                    f"the API listens on {self.api.bind_host}; it must be firewalled "
                    "to the ha host (SPEC §3.6)",
                )
            )
        if self.satellites.registry is None:
            warnings.append(
                (
                    "no_satellite_registry",
                    "no satellite registry is configured, so every request is served "
                    "T0-only (SPEC §3.6 fail closed)",
                )
            )
        return warnings


def _valid_env_names(model: type[BaseModel], prefix: str, delimiter: str) -> set[str]:
    """Every environment variable name ``model`` accepts, upper-cased."""
    names: set[str] = set()
    for name, field in model.model_fields.items():
        full = f"{prefix}{name}".upper()
        names.add(full)
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            names |= _valid_env_names(field.annotation, f"{full}{delimiter}", delimiter)
    return names


def _check_env_typos(env: Mapping[str, str]) -> None:
    """Reject FARMHUB_* variables that match no setting.

    pydantic-settings ignores unknown top-level variables, so a typo such as
    ``FARMHUB_DRY_RUM`` would silently leave the default in force. SPEC §7: fail loudly.
    """
    prefix = str(Settings.model_config.get("env_prefix", ""))
    delimiter = str(Settings.model_config.get("env_nested_delimiter", "__"))
    valid = _valid_env_names(Settings, prefix, delimiter)
    unknown = sorted(
        k for k in env if k.upper().startswith(prefix.upper()) and k.upper() not in valid
    )
    if unknown:
        raise ConfigError(f"unknown environment variable(s): {', '.join(unknown)}")


def load_settings(
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Load and validate settings.

    ``config_path`` names an explicit TOML file, which must exist. When omitted, the
    default ``config/farmhub.toml`` is used if present. ``overrides`` are the CLI layer.
    """
    _check_env_typos(os.environ)
    if config_path is not None and not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    toml_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH

    class _Loaded(Settings):
        model_config = SettingsConfigDict(toml_file=toml_path)

    try:
        return _Loaded(**dict(overrides or {}))
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"invalid configuration: {details}") from exc
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ConfigError(f"cannot read config file {toml_path}: {exc}") from exc


def log_effective_safety_settings(
    settings: Settings, log: structlog.typing.FilteringBoundLogger
) -> None:
    """Log the effective safety settings, then every fail-safe default in force.

    Called from server startup as well as the CLI, so that whatever starts FarmHub,
    the log says on its first lines what the safety posture actually is.
    """
    log.info("effective_safety_settings", **settings.safety_summary())
    for event, detail in settings.safety_warnings():
        log.warning(event, detail=detail)
