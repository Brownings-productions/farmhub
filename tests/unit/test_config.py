from pathlib import Path

import pytest
import structlog
from structlog.testing import capture_logs

from farmhub.core.config import load_settings, log_effective_safety_settings
from farmhub.core.errors import ConfigError


def write_toml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "farmhub.toml"
    path.write_text(text)
    return path


def test_defaults_when_nothing_is_configured() -> None:
    settings = load_settings()
    assert settings.dry_run is True
    assert settings.logging.level == "INFO"


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    path = write_toml(tmp_path, 'dry_run = false\n[logging]\nlevel = "DEBUG"\n')
    settings = load_settings(path)
    assert settings.dry_run is False
    assert settings.logging.level == "DEBUG"


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_toml(tmp_path, 'dry_run = false\n[logging]\nlevel = "DEBUG"\n')
    monkeypatch.setenv("FARMHUB_DRY_RUN", "true")
    monkeypatch.setenv("FARMHUB_LOGGING__LEVEL", "WARNING")
    settings = load_settings(path)
    assert settings.dry_run is True
    assert settings.logging.level == "WARNING"


def test_cli_overrides_env_and_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_toml(tmp_path, "dry_run = true\n")
    monkeypatch.setenv("FARMHUB_DRY_RUN", "true")
    settings = load_settings(path, {"dry_run": False, "logging": {"level": "ERROR"}})
    assert settings.dry_run is False
    assert settings.logging.level == "ERROR"


def test_default_config_path_is_read_when_present(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "farmhub.toml").write_text("dry_run = false\n")
    assert load_settings().dry_run is False


def test_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.toml")


def test_invalid_toml_syntax_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_settings(write_toml(tmp_path, "dry_run = = ="))


def test_invalid_value_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"logging\.level"):
        load_settings(write_toml(tmp_path, '[logging]\nlevel = "LOUD"\n'))


def test_unknown_toml_key_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dry_rn"):
        load_settings(write_toml(tmp_path, "dry_rn = false\n"))


def test_effective_safety_settings_are_logged() -> None:
    settings = load_settings()
    with capture_logs() as logs:
        log_effective_safety_settings(settings, structlog.get_logger())
    assert logs == [{"event": "effective_safety_settings", "log_level": "info", "dry_run": True}]


def test_warning_when_dry_run_is_off() -> None:
    settings = load_settings(overrides={"dry_run": False})
    with capture_logs() as logs:
        log_effective_safety_settings(settings, structlog.get_logger())
    assert any(e["event"] == "dry_run_disabled" and e["log_level"] == "warning" for e in logs)
