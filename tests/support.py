"""Shared test doubles: audit sinks, tools and a stub module."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from farmhub.core.config import Settings
from farmhub.core.context import AppContext
from farmhub.core.errors import AuditWriteError
from farmhub.core.events import EventBus
from farmhub.core.protocols import (
    AuditEntry,
    AuditSink,
    HealthReport,
    HealthState,
    Tier,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
)

EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


class MemoryAuditSink:
    """Keeps rows in memory. The test double for the real sinks that land later."""

    def __init__(self) -> None:
        self.rows: list[AuditEntry] = []

    async def write(self, entry: AuditEntry) -> None:
        self.rows.append(entry)


class FailingAuditSink:
    """Fails every write, or every write after the first ``ok_writes`` succeed."""

    def __init__(self, ok_writes: int = 0) -> None:
        self.rows: list[AuditEntry] = []
        self._ok_writes = ok_writes

    async def write(self, entry: AuditEntry) -> None:
        if len(self.rows) >= self._ok_writes:
            raise AuditWriteError("simulated audit failure")
        self.rows.append(entry)


@dataclass
class ToolProbe:
    """A tool whose handler counts how often it really ran."""

    spec: ToolSpec
    calls: list[ToolCall] = field(default_factory=list)


def make_tool(
    name: str = "probe",
    tier: Tier = Tier.READ,
    *,
    untrusted: bool = False,
    result: ToolResult | None = None,
    raises: Exception | None = None,
    scopes: frozenset[str] = frozenset({"*"}),
    timeout_s: float = 10.0,
    delay_s: float = 0.0,
    on_call: Callable[[ToolCall, ToolContext], None] | None = None,
) -> ToolProbe:
    probe: ToolProbe

    async def handler(call: ToolCall, ctx: ToolContext) -> ToolResult:
        probe.calls.append(call)
        if on_call is not None:
            on_call(call, ctx)
        if delay_s:
            await asyncio.sleep(delay_s)
        if raises is not None:
            raise raises
        return result if result is not None else ToolResult(ok=True, data={"value": 1})

    spec = ToolSpec(
        name=name,
        description=f"test tool {name}",
        parameters=dict(EMPTY_SCHEMA),
        tier=tier,
        handler=handler,
        allowed_scopes=scopes,
        timeout_s=timeout_s,
        reads_untrusted_content=untrusted,
    )
    probe = ToolProbe(spec)
    return probe


class StubModule:
    """A configurable Module. ``start_errors`` are raised one per startup attempt, in order."""

    version = "0.0.1"

    def __init__(
        self,
        name: str,
        tools: list[ToolSpec] | None = None,
        start_errors: list[Exception] | None = None,
    ) -> None:
        self.name = name
        self._tools = tools or []
        self._start_errors = list(start_errors or [])
        self.events: list[str] = []

    async def startup(self, ctx: AppContext) -> None:
        self.events.append("startup")
        if self._start_errors:
            raise self._start_errors.pop(0)

    async def shutdown(self) -> None:
        self.events.append("shutdown")

    def tools(self) -> list[ToolSpec]:
        return list(self._tools)

    async def health(self) -> HealthReport:
        return HealthReport(HealthState.HEALTHY)


def make_app_context(
    audit: AuditSink | None = None, settings: Settings | None = None
) -> AppContext:
    log = structlog.get_logger("test")
    return AppContext(
        settings=settings if settings is not None else Settings(),
        log=log,
        events=EventBus(log),
        audit=audit if audit is not None else MemoryAuditSink(),
    )
