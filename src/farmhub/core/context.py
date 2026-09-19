"""AppContext: the shared dependencies handed to modules. No globals."""

from dataclasses import dataclass

import structlog

from farmhub.core.config import Settings
from farmhub.core.events import EventBus
from farmhub.core.protocols import AuditSink


@dataclass(frozen=True)
class AppContext:
    """Everything a module may share with the rest of the app.

    Modules never import each other; cross-module needs go through this object or the
    event bus. Shared clients (database, HA, LLM) are added here as their milestones land.
    """

    settings: Settings
    log: structlog.typing.FilteringBoundLogger
    events: EventBus
    audit: AuditSink
