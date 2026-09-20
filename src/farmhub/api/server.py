"""The FastAPI application and its lifespan (SPEC §5, §11 M1).

The lifespan is where the safety posture is established and announced: settings are
already validated by the time we get here, the effective values are logged before
anything serves, and the modules start under the §6 failure policy.

The bearer token is resolved here rather than at first request. A server that starts
without one and only discovers it at 06:00 on the first utterance is worse than one
that refuses to start at all (§7: values with no safe default are required).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import structlog
from fastapi import Depends, FastAPI

from farmhub.api import openai_compat, routes_admin
from farmhub.api.identity import require_bearer_token
from farmhub.app import App, build_app
from farmhub.core.config import Settings, log_effective_safety_settings
from farmhub.core.context import AppContext
from farmhub.core.errors import ConfigError
from farmhub.core.logging import configure_logging, get_logger
from farmhub.core.satellites import SatelliteRegistry, load_satellites
from farmhub.core.sessions import SessionStore


@dataclass
class ServerState:
    """What the routes need. Reached through ``request.app.state.farmhub``."""

    app: App
    ctx: AppContext
    satellites: SatelliteRegistry
    sessions: SessionStore


def create_app(
    settings: Settings, *, app: App | None = None, configure_logs: bool = True
) -> FastAPI:
    """Build the ASGI app.

    Nothing here does I/O or reaches a backend: that happens in the lifespan, so the
    app can be constructed and inspected without a running vLLM.
    """
    if configure_logs:
        configure_logging(settings.logging.level, settings.logging.file)
    log = get_logger("farmhub.api")

    token = settings.resolve_api_token()
    if token is None:
        raise ConfigError(
            "no API token is configured. Set api.token_file, or FARMHUB_API__TOKEN. "
            "The chat endpoint authenticates Home Assistant with it (SPEC §3.6)"
        )

    satellites = load_satellites(settings.satellites.registry)
    # ``app`` lets a caller supply an already-wired application — the seam tests use to
    # run the HTTP layer without a backend to dial.
    farmhub_app = app if app is not None else build_app(settings)
    state = ServerState(
        app=farmhub_app,
        ctx=farmhub_app.ctx,
        satellites=satellites,
        sessions=SessionStore(),
    )

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        await _startup(state, log, settings)
        try:
            yield
        finally:
            await state.app.registry.shutdown()
            # Modules never close the shared client; this layer built it, so it closes
            # it, once, after nothing can still be using it.
            await state.app.aclose()
            log.info("farmhub_stopped")

    api = FastAPI(
        title="FarmHub",
        summary="Local-first AI hub for a Norwegian homestead.",
        lifespan=lifespan,
        # No interactive docs: §13 rules out a web UI, and the only client is a
        # Home Assistant integration that already knows the shape.
        docs_url=None,
        redoc_url=None,
    )
    api.state.farmhub = state

    guard = Depends(require_bearer_token(token))
    api.include_router(openai_compat.build_router(), dependencies=[guard])
    api.include_router(routes_admin.build_router(), dependencies=[guard])
    # Liveness only, and deliberately last so the guarded routes above are the default.
    api.include_router(routes_admin.build_public_router())
    return api


async def _startup(
    state: ServerState,
    log: structlog.typing.FilteringBoundLogger,
    settings: Settings,
) -> None:
    """Announce the safety posture, then start the modules."""
    # First lines in the log, whatever started FarmHub: what dry-run is, where it
    # listens, whether a token is set, and every fail-safe default in force (§7).
    log_effective_safety_settings(settings, log)
    log.info("satellite_registry_loaded", satellites=len(state.satellites))

    await state.app.registry.startup(state.ctx)
    # From here a module whose dependency goes away is noticed and degraded, and one
    # that comes back is retried. This is what makes stopping vLLM by hand safe.
    state.app.registry.start_health_polling(settings.llm.health_interval_s)
    log.info(
        "farmhub_started",
        modules=dict(sorted((k, v.value) for k, v in state.app.registry.status().items())),
    )


__all__ = ["ServerState", "create_app"]
