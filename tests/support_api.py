"""Building a test server without a network, a backend or a GPU (SPEC §9)."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from support import StubLLM

from farmhub.api.server import create_app
from farmhub.app import build_app
from farmhub.core.config import Settings

TOKEN = "test-token-not-a-real-one"  # noqa: S105 - a fixture, not a credential

REGISTRY = """
[satellites.kitchen]
device_id = "dev-kitchen"
home_area = "kitchen"
scope = ["kitchen", "greenhouse"]
tier_ceiling = "CONFIRMED"

[satellites.workshop]
device_id = "dev-workshop"
home_area = "workshop"
scope = ["workshop"]
tier_ceiling = "COMFORT"
"""


def write_registry(tmp_path: Path) -> Path:
    path = tmp_path / "satellites.toml"
    path.write_text(REGISTRY)
    return path


def make_settings(tmp_path: Path | None = None, **overrides: Any) -> Settings:
    """Settings with a token set and, optionally, a satellite registry."""
    data: dict[str, Any] = {"api": {"token": TOKEN}}
    if tmp_path is not None:
        data["satellites"] = {"registry": str(write_registry(tmp_path))}
    data.update(overrides)
    return Settings(**data)


def make_client(
    settings: Settings,
    llm: StubLLM | None = None,
) -> tuple[TestClient, StubLLM]:
    """A TestClient over the real HTTP layer, with a stub backend and no modules.

    Both seams are the composition root's own: ``build_app`` takes the backend and
    ``create_app`` takes the wired app. Nothing is monkeypatched, so what is exercised
    here is the same wiring production uses.
    """
    backend = llm if llm is not None else StubLLM()
    app = build_app(settings, [], backend=backend)
    api = create_app(settings, app=app, configure_logs=False)
    return TestClient(api), backend


def auth(**extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", **extra}
