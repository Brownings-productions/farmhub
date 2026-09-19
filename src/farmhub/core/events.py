"""Async pub/sub bus for cross-module needs (modules never import each other)."""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import structlog


@dataclass(frozen=True)
class Event:
    topic: str
    payload: Mapping[str, Any] = field(default_factory=dict)


type EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Delivers events to subscribers concurrently.

    A failing subscriber is logged with context and never breaks the publisher or the
    other subscribers: an observer must not be able to take down the publishing module.
    """

    def __init__(self, log: structlog.typing.FilteringBoundLogger | None = None) -> None:
        self._log = log if log is not None else structlog.get_logger("farmhub.events")
        self._handlers: defaultdict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: EventHandler) -> Callable[[], None]:
        """Register ``handler`` for ``topic``. Returns a function that unsubscribes it."""
        self._handlers[topic].append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers[topic]:
                self._handlers[topic].remove(handler)

        return unsubscribe

    async def publish(self, topic: str, payload: Mapping[str, Any] | None = None) -> None:
        """Deliver an event to every current subscriber of ``topic`` and wait for them."""
        event = Event(topic, dict(payload or {}))
        handlers = list(self._handlers.get(topic, ()))
        await asyncio.gather(*(self._deliver(h, event) for h in handlers))

    async def _deliver(self, handler: EventHandler, event: Event) -> None:
        try:
            await handler(event)
        except Exception:  # noqa: BLE001 - logged with context; observers must not break publishers
            self._log.exception("event_handler_failed", topic=event.topic, handler=repr(handler))
