"""AppContext: the shared dependencies handed to modules. No globals."""

from dataclasses import dataclass

import structlog

from farmhub.core.config import Settings
from farmhub.core.events import EventBus
from farmhub.core.protocols import AuditSink, LLMBackend


@dataclass(frozen=True)
class AppContext:
    """Everything a module may share with the rest of the app.

    Modules never import each other; cross-module needs go through this object or the
    event bus. Shared clients (database, HA, LLM) are added here as their milestones land.

    ``llm`` is the backend itself, not the module that owns it: the orchestrator needs
    to issue completions without importing ``modules.llm``. Constructing the client
    opens no connection, so it is safe to place here before anything has started;
    whether it is *usable* is the llm module's health, not this field's presence.
    """

    settings: Settings
    log: structlog.typing.FilteringBoundLogger
    events: EventBus
    audit: AuditSink
    llm: LLMBackend
