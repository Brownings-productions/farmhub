"""Composition root: wires settings, shared services, the module list and the gateway.

The explicit list of modules lives here, not in ``core/registry.py``, so ``core`` never
imports a module (docs/DECISIONS.md, "Explicit module list lives in the composition
root"). No entry-point scanning: adding a module means adding it below, in review.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from farmhub.core.audit import UnconfiguredAuditSink
from farmhub.core.config import Settings
from farmhub.core.context import AppContext
from farmhub.core.events import EventBus
from farmhub.core.gateway import Gateway
from farmhub.core.logging import get_logger
from farmhub.core.protocols import Module
from farmhub.core.registry import ModuleRegistry
from farmhub.modules.llm import LlmModule, LLMService, OpenAICompatBackend

ModuleFactory = Callable[[], Module]


@dataclass(frozen=True)
class App:
    """A fully wired application. Start it with ``await app.registry.startup(app.ctx)``."""

    ctx: AppContext
    registry: ModuleRegistry
    gateway: Gateway
    # The shared LLM connection. Held here because this is what constructed it, and
    # closing it is this layer's job, not any module's.
    backend: LLMService

    async def aclose(self) -> None:
        """Release what the composition root owns. Call after the registry has stopped."""
        await self.backend.aclose()


def default_modules(ctx: AppContext, backend: LLMService) -> tuple[ModuleFactory, ...]:
    """The modules FarmHub runs, in start order.

    This is the explicit list of SPEC §6 — no entry-point scanning, and ``core`` never
    imports a module. New milestones add a line here, in review (records at M2, ingest
    at M3, ...).

    ``backend`` is passed in rather than taken from ``ctx.llm`` because the composition
    root already holds the concrete object it built; reaching back through the protocol
    would only mean narrowing it again.
    """
    return (lambda: LlmModule(backend),)


def build_app(
    settings: Settings,
    module_factories: Sequence[ModuleFactory] | None = None,
    *,
    backend: LLMService | None = None,
) -> App:
    """Construct the app from validated settings.

    Nothing here does I/O: the LLM client opens no connection until the llm module
    probes it, so building an app is safe even with every backend down.

    Until a real audit sink exists the context carries ``UnconfiguredAuditSink``, which
    refuses every write, so the gateway denies every call rather than running unaudited.
    """
    log = get_logger("farmhub")
    # ``backend`` is the composition root's one injection point: production builds the
    # real client, tests hand in a double, and nothing below has to know which.
    if backend is None:
        backend = OpenAICompatBackend(settings.llm, log)
    ctx = AppContext(
        settings=settings,
        log=log,
        events=EventBus(log),
        audit=UnconfiguredAuditSink(),
        llm=backend,
    )
    factories = (
        default_modules(ctx, backend) if module_factories is None else tuple(module_factories)
    )
    registry = ModuleRegistry([factory() for factory in factories])
    return App(ctx=ctx, registry=registry, gateway=Gateway(ctx, registry), backend=backend)
