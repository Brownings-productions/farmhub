"""The satellite registry (SPEC §3.6, §8.8)."""

from pathlib import Path

import pytest

from farmhub.core.errors import ConfigError
from farmhub.core.protocols import Tier
from farmhub.core.satellites import load_satellites

EXAMPLE = Path(__file__).parents[2] / "config" / "satellites.example.toml"

ONE = """
[satellites.kitchen]
device_id = "dev-kitchen"
home_area = "kitchen"
scope = ["kitchen", "greenhouse"]
tier_ceiling = "CONFIRMED"
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "satellites.toml"
    path.write_text(text)
    return path


def test_the_example_registry_loads() -> None:
    registry = load_satellites(EXAMPLE)
    assert len(registry) == 3


def test_lookup_by_device_id(tmp_path: Path) -> None:
    registry = load_satellites(write(tmp_path, ONE))
    satellite = registry.get("dev-kitchen")
    assert satellite is not None
    assert satellite.scope == frozenset({"kitchen", "greenhouse"})
    assert satellite.tier_ceiling is Tier.CONFIRMED


def test_an_unknown_device_id_is_unknown_not_a_guess() -> None:
    """SPEC §3.6 fail closed. The caller then gets T0 only."""
    registry = load_satellites(EXAMPLE)
    assert registry.get("not-registered") is None
    assert registry.get(None) is None


def test_no_registry_configured_means_nothing_is_known() -> None:
    """Not an error: it means every request is T0-only, which startup warns about."""
    assert len(load_satellites(None)) == 0


def test_a_missing_registry_file_fails_startup(tmp_path: Path) -> None:
    """Configuring a path that is not there is a mistake, not a fail-closed default."""
    with pytest.raises(ConfigError, match="not found"):
        load_satellites(tmp_path / "absent.toml")


def test_a_tier_ceiling_above_t2_fails_startup(tmp_path: Path) -> None:
    """SPEC §3.6: a ceiling may not exceed T2, or config alone would defeat §3.2."""
    toml = ONE.replace('tier_ceiling = "CONFIRMED"', 'tier_ceiling = "FORBIDDEN"')
    with pytest.raises(ConfigError, match="may not exceed"):
        load_satellites(write(tmp_path, toml))


def test_a_numeric_tier_ceiling_above_t2_also_fails(tmp_path: Path) -> None:
    toml = ONE.replace('tier_ceiling = "CONFIRMED"', "tier_ceiling = 3")
    with pytest.raises(ConfigError, match="may not exceed"):
        load_satellites(write(tmp_path, toml))


def test_an_unknown_tier_name_fails_startup(tmp_path: Path) -> None:
    toml = ONE.replace('tier_ceiling = "CONFIRMED"', 'tier_ceiling = "ADMIN"')
    with pytest.raises(ConfigError, match="unknown tier"):
        load_satellites(write(tmp_path, toml))


def test_two_satellites_sharing_a_device_id_fail_startup(tmp_path: Path) -> None:
    """Otherwise which scope a request gets would depend on dict ordering."""
    toml = ONE + ONE.replace("kitchen]", "barn]").replace(
        'home_area = "kitchen"', 'home_area = "kitchen"'
    )
    with pytest.raises(ConfigError, match="share device_id"):
        load_satellites(write(tmp_path, toml))


def test_a_home_area_outside_its_own_scope_fails_startup(tmp_path: Path) -> None:
    toml = ONE.replace('scope = ["kitchen", "greenhouse"]', 'scope = ["greenhouse"]')
    with pytest.raises(ConfigError, match="home_area"):
        load_satellites(write(tmp_path, toml))


def test_an_unknown_key_fails_startup(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_satellites(write(tmp_path, ONE + 'confirmers = ["phone"]\n'))


def test_broken_toml_fails_startup(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_satellites(write(tmp_path, "[satellites.kitchen\n"))
