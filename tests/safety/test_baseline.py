"""M0 baseline safety guarantees: defaults, config strictness, tier visibility, schemas."""

import dataclasses
from pathlib import Path

import pytest
from support import EMPTY_SCHEMA, StubModule, make_app_context, make_tool

from farmhub.core.config import load_settings
from farmhub.core.errors import ConfigError
from farmhub.core.protocols import Tier, ToolSpec
from farmhub.core.registry import ModuleRegistry


def test_dry_run_is_on_with_no_configuration_at_all() -> None:
    assert load_settings().dry_run is True


def test_dry_run_can_be_turned_off_only_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert load_settings(overrides={"dry_run": False}).dry_run is False
    monkeypatch.setenv("FARMHUB_DRY_RUN", "false")
    assert load_settings().dry_run is False
    monkeypatch.delenv("FARMHUB_DRY_RUN")
    toml = tmp_path / "farmhub.toml"
    toml.write_text("dry_run = false\n")
    assert load_settings(toml).dry_run is False
    assert load_settings().dry_run is True


def test_unknown_config_keys_fail_startup(tmp_path: Path) -> None:
    toml = tmp_path / "farmhub.toml"
    toml.write_text("dry_run = true\nallow_everything = true\n")
    with pytest.raises(ConfigError):
        load_settings(toml)


def test_forbidden_is_the_highest_tier() -> None:
    assert Tier.FORBIDDEN == 3
    assert max(Tier) is Tier.FORBIDDEN


def test_tool_specs_are_immutable() -> None:
    spec = make_tool("read").spec
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.tier = Tier.READ  # type: ignore[misc]


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {}},
        {"type": "object", "properties": {}, "additionalProperties": True},
        {"type": "string"},
    ],
)
def test_tool_parameters_must_be_strict_object_schemas(schema: dict[str, object]) -> None:
    good = make_tool("read").spec
    with pytest.raises(ConfigError):
        ToolSpec(
            name="bad",
            description="d",
            parameters=schema,
            tier=Tier.READ,
            handler=good.handler,
            allowed_scopes=frozenset({"*"}),
        )
    assert EMPTY_SCHEMA["additionalProperties"] is False


async def test_t3_tools_are_never_model_visible() -> None:
    tools = [make_tool("read").spec, make_tool("heating_off", Tier.FORBIDDEN).spec]
    registry = ModuleRegistry([StubModule("m", tools=tools)])
    await registry.startup(make_app_context())
    assert Tier.FORBIDDEN not in {t.tier for t in registry.available_tools()}
    await registry.shutdown()
