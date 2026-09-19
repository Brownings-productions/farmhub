"""Layered configuration (SPEC §7).

Precedence, highest first: CLI overrides > ``FARMHUB_*`` environment > TOML file > code
defaults. Invalid config raises ``ConfigError``; callers exit rather than continue.

Safety-relevant settings may have fail-safe defaults in code (docs/DECISIONS.md,
"Safety-relevant defaults"), but their effective values are always logged at startup
and printed by ``farmhub config check``.
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from farmhub.core.errors import ConfigError

DEFAULT_CONFIG_PATH = Path("config/farmhub.toml")


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path | None = Path("logs/farmhub.jsonl")


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
        """The effective safety-relevant settings, for startup logs and ``config check``."""
        return {"dry_run": self.dry_run}


def load_settings(
    config_path: Path | None = None, overrides: Mapping[str, Any] | None = None
) -> Settings:
    """Load and validate settings.

    ``config_path`` names an explicit TOML file, which must exist. When omitted, the
    default ``config/farmhub.toml`` is used if present. ``overrides`` are the CLI layer.
    """
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
    """Log the effective safety settings; warn loudly when dry-run has been turned off."""
    log.info("effective_safety_settings", **settings.safety_summary())
    if not settings.dry_run:
        log.warning("dry_run_disabled", detail="tools at T1+ may reach Home Assistant")
