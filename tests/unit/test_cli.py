import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from farmhub import __version__
from farmhub.__main__ import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"farmhub {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_config_check_prints_effective_safety_settings_first() -> None:
    result = runner.invoke(app, ["config", "check"])
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0] == "configuration OK"
    assert lines[1] == "safety.dry_run = true"


def test_config_check_warns_when_dry_run_is_off() -> None:
    result = runner.invoke(app, ["--no-dry-run", "config", "check"])
    assert result.exit_code == 0
    assert "safety.dry_run = false" in result.stdout
    assert "WARNING: dry_run_disabled" in result.stderr


def test_cli_flag_beats_env_and_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "farmhub.toml"
    toml.write_text("dry_run = false\n")
    monkeypatch.setenv("FARMHUB_DRY_RUN", "false")
    result = runner.invoke(app, ["--config", str(toml), "--dry-run", "config", "check"])
    assert "safety.dry_run = true" in result.stdout


def test_env_beats_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "farmhub.toml"
    toml.write_text("dry_run = false\n")
    monkeypatch.setenv("FARMHUB_DRY_RUN", "true")
    result = runner.invoke(app, ["--config", str(toml), "config", "check"])
    assert "safety.dry_run = true" in result.stdout


def test_invalid_config_exits_nonzero_and_says_why(tmp_path: Path) -> None:
    toml = tmp_path / "farmhub.toml"
    toml.write_text("dry_rn = false\n")
    result = runner.invoke(app, ["--config", str(toml), "config", "check"])
    assert result.exit_code == 2
    assert "dry_rn" in result.stderr


def test_missing_config_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(tmp_path / "nope.toml"), "config", "check"])
    assert result.exit_code == 2


def test_invalid_log_level_exits_nonzero() -> None:
    result = runner.invoke(app, ["--log-level", "LOUD", "config", "check"])
    assert result.exit_code == 2


def test_console_script_reports_the_version() -> None:
    script = Path(sys.executable).parent / "farmhub"
    done = subprocess.run([str(script), "--version"], capture_output=True, text=True, check=False)
    assert done.returncode == 0
    assert done.stdout.strip() == f"farmhub {__version__}"


def test_python_dash_m_reports_the_version() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "farmhub", "--version"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0
    assert done.stdout.strip() == f"farmhub {__version__}"
