from pathlib import Path

from farmhub.core.config import load_settings

EXAMPLE = Path(__file__).parents[2] / "config" / "farmhub.example.toml"


def test_example_config_is_valid_and_keeps_dry_run_on() -> None:
    settings = load_settings(EXAMPLE)
    assert settings.dry_run is True
