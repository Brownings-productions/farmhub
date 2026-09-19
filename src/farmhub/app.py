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

# Modules are added here as their milestones land (llm at M1, records at M2, ...).
ENABLED_MODULES: tuple[Callable[[], Module], ...] = ()


@dataclass(frozen=True)
class App:
    """A fully wired application. Start it with ``await app.registry.startup(app.ctx)``."""

    ctx: AppContext
    registry: ModuleRegistry
    gateway: Gateway


def build_app(
    settings: Settings,
    module_factories: Sequence[Callable[[], Module]] = ENABLED_MODULES,
) -> App:
    """Construct the app from validated settings.

    Until a real audit sink exists the context carries ``UnconfiguredAuditSink``, which
    refuses every write, so the gateway denies every call rather than running unaudited.
    """
    log = get_logger("farmhub")
    ctx = AppContext(
        settings=settings,
        log=log,
        events=EventBus(log),
        audit=UnconfiguredAuditSink(),
    )
    registry = ModuleRegistry([factory() for factory in module_factories])
    return App(ctx=ctx, registry=registry, gateway=Gateway(ctx, registry))
