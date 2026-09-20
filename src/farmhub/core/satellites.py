"""The satellite registry (SPEC §3.6, §8.8).

Maps an authenticated Home Assistant ``device_id`` to a home area, a scope and a tier
ceiling. The gateway derives the allowed tool set from this and never from anything the
model or the request body says.

M1 ships the identity half: name, device_id, home area, scope, ceiling. M9 adds input
mode, and `confirmers` are per area (§3.3), not per satellite.
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from farmhub.core.errors import ConfigError
from farmhub.core.protocols import Tier


class Satellite(BaseModel):
    """One registered satellite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    # The HA device_id the conversation integration sends. Authenticated by the bearer
    # token that HA holds: HA vouches for this value, FarmHub does not verify it
    # independently (docs/DECISIONS.md).
    device_id: str
    home_area: str
    # The areas this satellite may act on. The workshop satellite cannot reach
    # greenhouse valves because greenhouse is not in its scope, not because any tool
    # says so.
    scope: frozenset[str]
    tier_ceiling: Tier = Tier.COMFORT

    @field_validator("tier_ceiling", mode="before")
    @classmethod
    def _accept_tier_names(cls, value: object) -> object:
        """Config names tiers the way SPEC §7 does: "COMFORT", not 1.

        ``Tier`` is an IntEnum, so pydantic would otherwise demand the number. Writing
        a tier ceiling as a digit in a config file is exactly the kind of thing that
        gets mistyped without anyone noticing.
        """
        if isinstance(value, str):
            try:
                return Tier[value.strip().upper()]
            except KeyError:
                names = ", ".join(t.name for t in Tier)
                raise ValueError(f"unknown tier {value!r}; expected one of {names}") from None
        return value

    @field_validator("tier_ceiling", mode="after")
    @classmethod
    def _ceiling_at_most_t2(cls, value: Tier) -> Tier:
        # SPEC §3.6: "a tier ceiling that may not exceed T2". A satellite that could
        # reach T3 would defeat the entire tier system from config alone.
        if value > Tier.CONFIRMED:
            raise ValueError(f"tier_ceiling may not exceed CONFIRMED (T2), got {value.name}")
        return value

    @model_validator(mode="after")
    def _home_area_is_in_scope(self) -> "Satellite":
        if self.home_area not in self.scope:
            raise ValueError(
                f"home_area {self.home_area!r} is not in scope {sorted(self.scope)}; a "
                "satellite that cannot act in its own room is almost certainly a typo"
            )
        return self


class SatelliteRegistry:
    """Lookup by device id. Unknown means unknown — never a guess (§3.6 fail closed)."""

    def __init__(self, satellites: Mapping[str, Satellite]) -> None:
        self._by_device: dict[str, Satellite] = {}
        for sat in satellites.values():
            if sat.device_id in self._by_device:
                raise ConfigError(
                    f"two satellites share device_id {sat.device_id!r}: "
                    f"{self._by_device[sat.device_id].name} and {sat.name}"
                )
            self._by_device[sat.device_id] = sat

    def __len__(self) -> int:
        return len(self._by_device)

    def get(self, device_id: str | None) -> Satellite | None:
        """The satellite for ``device_id``, or None when there is no such thing."""
        if device_id is None:
            return None
        return self._by_device.get(device_id)


EMPTY_REGISTRY = SatelliteRegistry({})


def load_satellites(path: Path | None) -> SatelliteRegistry:
    """Load the registry, or return the empty one when none is configured.

    An empty registry is not an error: §3.6 says a request whose identity is not in
    the registry is served T0-only, so "no satellites configured" simply means every
    request is T0-only. That is logged at startup rather than left silent.

    A registry file that exists but cannot be parsed is a ``ConfigError`` and fails
    startup, because in that case the operator clearly intended something.
    """
    if path is None:
        return EMPTY_REGISTRY
    if not path.is_file():
        raise ConfigError(f"satellite registry not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ConfigError(f"cannot read satellite registry {path}: {exc}") from exc

    section = raw.get("satellites", raw)
    if not isinstance(section, dict):
        raise ConfigError(f"{path}: expected a table of satellites")

    satellites: dict[str, Satellite] = {}
    for name, body in section.items():
        if not isinstance(body, dict):
            raise ConfigError(f"{path}: satellite {name!r} is not a table")
        try:
            satellites[name] = Satellite(name=name, **body)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ConfigError(f"{path}: satellite {name!r}: {details}") from exc
    return SatelliteRegistry(satellites)
