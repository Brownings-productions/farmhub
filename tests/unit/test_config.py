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
    summary = next(e for e in logs if e["event"] == "effective_safety_settings")
    assert summary["log_level"] == "info"
    # Every safety-relevant setting reaches the log, not only dry_run (SPEC §7).
    assert summary["dry_run"] is True
    assert summary["api_bind_host"] == "127.0.0.1"
    assert summary["api_token_configured"] is False
    assert summary["llm_base_url"] == "http://127.0.0.1:8000/v1"


def test_the_bearer_token_never_reaches_the_safety_summary() -> None:
    """The summary is written to the log file and printed to a terminal."""
    settings = load_settings(overrides={"api": {"token": "s3cret-do-not-log"}})
    summary = settings.safety_summary()
    assert summary["api_token_configured"] is True
    assert "s3cret-do-not-log" not in repr(summary)


def test_warning_when_dry_run_is_off() -> None:
    settings = load_settings(overrides={"dry_run": False})
    with capture_logs() as logs:
        log_effective_safety_settings(settings, structlog.get_logger())
    assert any(e["event"] == "dry_run_disabled" and e["log_level"] == "warning" for e in logs)


def test_unknown_farmhub_env_var_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FARMHUB_DRY_RUM", "false")
    with pytest.raises(ConfigError, match="FARMHUB_DRY_RUM"):
        load_settings()


def test_unknown_nested_env_var_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FARMHUB_LOGGING__LEVL", "DEBUG")
    with pytest.raises(ConfigError):
        load_settings()


def test_known_env_vars_are_accepted_in_any_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("farmhub_dry_run", "false")
    monkeypatch.setenv("FARMHUB_LOGGING__LEVEL", "DEBUG")
    settings = load_settings()
    assert settings.dry_run is False
    assert settings.logging.level == "DEBUG"


# --- model profiles (SPEC §2) ---------------------------------------------------------

# A profile that satisfies every rule, so each test below can break exactly one thing.
GOOD_PROFILE = """
[llm]
profile = "primary"

[profiles.primary]
repo_id = "nvidia/Qwen3.6-35B-A3B-NVFP4"
revision = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
quantization = "modelopt_fp4"
kv_cache_dtype = "auto"
gpu_memory_utilization = 0.82
max_model_len = 32768
weights_gib = 17.5
kv_cache_gib = 8.0
aux_reserve_gib = 5.0
card_total_gib = 31.8
measured_by = "evals/2026-09-20T12-00-00Z"
chat_template_kwargs = { enable_thinking = false }
temperature = 0.0
server_args = ["--block-size", "128", "--tool-call-parser", "hermes", "--language-model-only"]
"""


def test_a_measured_profile_loads(tmp_path: Path) -> None:
    settings = load_settings(write_toml(tmp_path, GOOD_PROFILE))
    profile = settings.active_profile()
    assert profile is not None
    assert profile.repo_id == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert settings.safety_summary()["llm_profile"] == "primary"
    assert profile.chat_template_kwargs == {"enable_thinking": False}


def test_a_profile_without_chat_template_kwargs_fails_startup(tmp_path: Path) -> None:
    """The switch that keeps the model answering is not optional (docs/DECISIONS.md).

    Left to the model's default, Qwen3.6 reasons out loud and never reaches an answer
    inside a sane token budget, so a profile that forgets this serves truncated
    ramblings to a satellite.
    """
    toml = "\n".join(
        line for line in GOOD_PROFILE.splitlines() if not line.startswith("chat_template_kwargs")
    )
    with pytest.raises(ConfigError, match="chat_template_kwargs"):
        load_settings(write_toml(tmp_path, toml))


def test_an_empty_chat_template_kwargs_loads_but_warns(tmp_path: Path) -> None:
    """Some model may need no switch. That is a declaration, not an omission."""
    toml = GOOD_PROFILE.replace(
        "chat_template_kwargs = { enable_thinking = false }", "chat_template_kwargs = {}"
    )
    settings = load_settings(write_toml(tmp_path, toml))
    assert settings.active_profile() is not None
    events = [event for event, _ in settings.safety_warnings()]
    assert "profile_declares_no_chat_template_kwargs" in events


def test_the_startup_summary_shows_the_chat_template_switches(tmp_path: Path) -> None:
    settings = load_settings(write_toml(tmp_path, GOOD_PROFILE))
    assert settings.safety_summary()["llm_chat_template_kwargs"] == {"enable_thinking": False}


def test_a_profile_without_temperature_fails_startup(tmp_path: Path) -> None:
    """The backend's own default is not an answer (docs/DECISIONS.md 2026-09-25).

    Both M1 eval runs took it, and neither tool score could be compared with the other:
    one case flipped between them and nothing in either run could say why.
    """
    toml = "\n".join(
        line for line in GOOD_PROFILE.splitlines() if not line.startswith("temperature")
    )
    with pytest.raises(ConfigError, match="temperature"):
        load_settings(write_toml(tmp_path, toml))


def test_the_startup_summary_shows_the_sampling_temperature(tmp_path: Path) -> None:
    settings = load_settings(write_toml(tmp_path, GOOD_PROFILE))
    assert settings.safety_summary()["llm_temperature"] == 0.0


def test_a_profile_without_server_args_fails_startup(tmp_path: Path) -> None:
    """The flags decide what is actually served, and nothing else records them.

    Drop candidate A's --language-model-only and the vision tower loads: weights grow,
    the KV cache shrinks, and weights_gib/kv_cache_gib become fiction while every check
    in this file still passes (docs/MODEL_EVAL.md, core/serving.py).
    """
    toml = "\n".join(
        line for line in GOOD_PROFILE.splitlines() if not line.startswith("server_args")
    )
    with pytest.raises(ConfigError, match="server_args"):
        load_settings(write_toml(tmp_path, toml))


def test_an_empty_server_args_loads(tmp_path: Path) -> None:
    """A backend that needs no flags is a declaration, not an omission."""
    toml = GOOD_PROFILE.replace(
        'server_args = ["--block-size", "128", "--tool-call-parser", "hermes", '
        '"--language-model-only"]',
        "server_args = []",
    )
    profile = load_settings(write_toml(tmp_path, toml)).active_profile()
    assert profile is not None
    assert profile.server_args == []


def test_the_startup_summary_shows_the_server_flags(tmp_path: Path) -> None:
    settings = load_settings(write_toml(tmp_path, GOOD_PROFILE))
    assert "--language-model-only" in settings.safety_summary()["llm_server_args"]  # type: ignore[operator]


@pytest.mark.parametrize("revision", ["main", "v1.0", "1355db6", "z" * 40])
def test_a_revision_that_is_not_a_commit_sha_fails_startup(tmp_path: Path, revision: str) -> None:
    """SPEC §2: a tag is not a pin. An upstream tag was moved under a released model."""
    toml = GOOD_PROFILE.replace("1355db6a052410cfd62085d94b58866fd0f2c3c5", revision)
    with pytest.raises(ConfigError, match="commit sha"):
        load_settings(write_toml(tmp_path, toml))


def test_a_profile_without_measured_by_fails_startup(tmp_path: Path) -> None:
    """Budget numbers come from an evals run, never from hand (SPEC §2)."""
    toml = "\n".join(
        line for line in GOOD_PROFILE.splitlines() if not line.startswith("measured_by")
    )
    with pytest.raises(ConfigError, match="measured_by"):
        load_settings(write_toml(tmp_path, toml))


def test_a_budget_larger_than_the_card_fails_startup(tmp_path: Path) -> None:
    toml = GOOD_PROFILE.replace("weights_gib = 17.5", "weights_gib = 26.0")
    with pytest.raises(ConfigError, match="exceed"):
        load_settings(write_toml(tmp_path, toml))


def test_utilization_that_starves_the_auxiliary_models_fails_startup(tmp_path: Path) -> None:
    """vLLM must leave aux_reserve_gib free for Whisper, BGE-M3 and the reranker."""
    toml = GOOD_PROFILE.replace("gpu_memory_utilization = 0.82", "gpu_memory_utilization = 0.95")
    with pytest.raises(ConfigError, match="aux_reserve_gib"):
        load_settings(write_toml(tmp_path, toml))


def test_selecting_an_undefined_profile_fails_startup(tmp_path: Path) -> None:
    toml = GOOD_PROFILE.replace('profile = "primary"', 'profile = "does-not-exist"')
    with pytest.raises(ConfigError, match="does-not-exist"):
        load_settings(write_toml(tmp_path, toml))


def test_an_unknown_profile_key_fails_startup(tmp_path: Path) -> None:
    toml = GOOD_PROFILE + "looks_fine = true\n"
    with pytest.raises(ConfigError):
        load_settings(write_toml(tmp_path, toml))


# --- api token (SPEC §3.6) ------------------------------------------------------------


def test_token_is_read_from_a_file_so_it_need_not_be_in_the_toml(tmp_path: Path) -> None:
    secret = tmp_path / "token"
    secret.write_text("  hunter2\n")
    settings = load_settings(overrides={"api": {"token_file": str(secret)}})
    assert settings.resolve_api_token() == "hunter2"


def test_an_empty_token_file_is_an_error(tmp_path: Path) -> None:
    secret = tmp_path / "token"
    secret.write_text("\n")
    settings = load_settings(overrides={"api": {"token_file": str(secret)}})
    with pytest.raises(ConfigError, match="empty"):
        settings.resolve_api_token()


def test_setting_both_token_and_token_file_fails_startup(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not both"):
        load_settings(overrides={"api": {"token": "x", "token_file": str(tmp_path / "t")}})


def test_a_non_loopback_bind_address_is_warned_about() -> None:
    """Allowed, but never silent: §3.6 requires it to be firewalled to the ha host."""
    settings = load_settings(overrides={"api": {"bind_host": "0.0.0.0"}})  # noqa: S104
    events = [event for event, _ in settings.safety_warnings()]
    assert "api_bind_not_loopback" in events
