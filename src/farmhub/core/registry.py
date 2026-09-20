"""Module lifecycle and tool registration.

The registry takes an explicit module list from its caller (the composition root,
``farmhub/app.py``); it never discovers modules and never imports them.

Failure policy (docs/DECISIONS.md, "Two failure kinds"):

* Invalid config, or any unexpected error, during the first startup attempt fails fast:
  already-started modules are stopped and ``ModuleLoadError`` is raised.
* ``DependencyUnavailable`` starts the module DEGRADED: its tools are withheld from every
  tool list and startup is retried with capped exponential backoff.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from farmhub.core.context import AppContext
from farmhub.core.errors import ConfigError, DependencyUnavailable, ModuleLoadError
from farmhub.core.protocols import HealthReport, HealthState, Module, Tier, ToolSpec


class ModuleStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DEGRADED = "degraded"  # dependency unavailable; retrying
    FAILED = "failed"  # unexpected error on a retry; not retried
    STOPPED = "stopped"


@dataclass(frozen=True)
class BackoffPolicy:
    base_s: float = 1.0
    factor: float = 2.0
    max_s: float = 60.0

    def delay(self, attempt: int) -> float:
        """Delay before retry number ``attempt`` (0-based), capped at ``max_s``."""
        return min(self.max_s, self.base_s * self.factor**attempt)


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    module: str
    available: bool  # False while the owning module is not RUNNING


@dataclass
class _Entry:
    module: Module
    status: ModuleStatus = ModuleStatus.PENDING
    attempted: bool = False  # startup was called at least once, so shutdown is owed
    reason: str = ""
    tools: tuple[ToolSpec, ...] = ()
    retry: asyncio.Task[None] | None = field(default=None, repr=False)


class ModuleRegistry:
    """Owns module start/stop, health, and the set of registered tools."""

    def __init__(
        self,
        modules: Sequence[Module],
        *,
        backoff: BackoffPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        names = [m.name for m in modules]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ModuleLoadError(f"duplicate module name(s): {', '.join(duplicates)}")
        self._entries = [_Entry(m) for m in modules]
        self._backoff = backoff if backoff is not None else BackoffPolicy()
        self._sleep = sleep
        self._ctx: AppContext | None = None
        self._log: structlog.typing.FilteringBoundLogger = structlog.get_logger("farmhub.registry")
        self._health_task: asyncio.Task[None] | None = None

    async def startup(self, ctx: AppContext) -> None:
        """Start every module in order. See the module docstring for the failure policy."""
        self._ctx = ctx
        self._log = ctx.log.bind(component="registry")
        for entry in self._entries:
            try:
                await self._start(entry)
            except DependencyUnavailable as exc:
                await self._degrade(entry, str(exc))
            except ModuleLoadError:
                await self._rollback()
                raise
            except Exception as exc:
                self._log.error("module_startup_failed", module=entry.module.name, error=str(exc))
                await self._rollback()
                kind = "invalid config" if isinstance(exc, ConfigError) else "unexpected error"
                raise ModuleLoadError(
                    f"module {entry.module.name!r} failed to start ({kind}): {exc}"
                ) from exc

    async def shutdown(self) -> None:
        """Cancel retries, then stop modules in reverse start order."""
        await self._rollback()

    def status(self) -> dict[str, ModuleStatus]:
        return {e.module.name: e.status for e in self._entries}

    def available_tools(self) -> list[ToolSpec]:
        """Tools of RUNNING modules. Never includes T3: those are never model-visible."""
        return [
            spec
            for entry in self._entries
            if entry.status is ModuleStatus.RUNNING
            for spec in entry.tools
            if spec.tier is not Tier.FORBIDDEN
        ]

    def find_tool(self, name: str) -> RegisteredTool | None:
        """Look up a registered tool by name, for the gateway only.

        Unlike ``available_tools`` this includes T3 tools and tools of degraded modules,
        so the gateway can audit a denial with the correct tier.
        """
        for entry in self._entries:
            for spec in entry.tools:
                if spec.name == name:
                    return RegisteredTool(
                        spec, entry.module.name, entry.status is ModuleStatus.RUNNING
                    )
        return None

    async def health(self) -> dict[str, HealthReport]:
        report: dict[str, HealthReport] = {}
        for entry in self._entries:
            name = entry.module.name
            if entry.status is ModuleStatus.RUNNING:
                report[name] = await entry.module.health()
            elif entry.status in (ModuleStatus.DEGRADED, ModuleStatus.FAILED):
                report[name] = HealthReport(HealthState.DEGRADED, entry.reason)
            else:
                report[name] = HealthReport(HealthState.UNHEALTHY, entry.status.value)
        return report

    async def mark_unhealthy(self, name: str, reason: str) -> None:
        """Called by a module that noticed its dependency went away at runtime."""
        entry = next((e for e in self._entries if e.module.name == name), None)
        if entry is None:
            raise ModuleLoadError(f"unknown module {name!r}")
        if entry.status is ModuleStatus.RUNNING:
            await self._degrade(entry, reason)

    def start_health_polling(self, interval_s: float) -> None:
        """Poll running modules so a dependency that goes away is noticed.

        Without this, a module stays RUNNING and keeps its tools in every tool list
        long after its backend disappeared: nothing would find out until the next call
        failed. An unhealthy report routes into the same degrade-and-retry path as a
        failed startup, so recovery is already handled.

        Only RUNNING modules are polled. A degraded one already has a retry loop.
        """
        if self._health_task is not None and not self._health_task.done():
            return
        self._health_task = asyncio.create_task(self._health_loop(interval_s))

    async def poll_health(self) -> None:
        """One polling pass. Separate from the loop so tests need no wall-clock wait."""
        for entry in list(self._entries):
            if entry.status is not ModuleStatus.RUNNING:
                continue
            name = entry.module.name
            try:
                report = await entry.module.health()
            except Exception as exc:  # noqa: BLE001 - a health check that raises is unhealthy
                self._log.warning("health_check_raised", module=name, error=str(exc))
                await self.mark_unhealthy(name, f"health check raised: {exc}")
                continue
            if report.status is HealthState.UNHEALTHY:
                self._log.warning("module_unhealthy", module=name, detail=report.detail)
                await self.mark_unhealthy(name, report.detail or "reported unhealthy")

    async def _health_loop(self, interval_s: float) -> None:
        while True:
            await self._sleep(interval_s)
            await self.poll_health()

    async def _start(self, entry: _Entry) -> None:
        ctx = self._require_ctx()
        entry.attempted = True
        await entry.module.startup(ctx)
        tools = tuple(entry.module.tools())
        self._check_tool_names(entry, tools)
        entry.tools = tools
        entry.status = ModuleStatus.RUNNING
        entry.reason = ""

    def _require_ctx(self) -> AppContext:
        if self._ctx is None:
            raise ModuleLoadError("registry used before startup(ctx)")
        return self._ctx

    def _check_tool_names(self, entry: _Entry, tools: tuple[ToolSpec, ...]) -> None:
        taken = {s.name for e in self._entries if e is not entry for s in e.tools}
        seen: set[str] = set()
        for spec in tools:
            if spec.name in taken or spec.name in seen:
                raise ModuleLoadError(
                    f"duplicate tool name {spec.name!r} (module {entry.module.name!r})"
                )
            seen.add(spec.name)

    async def _degrade(self, entry: _Entry, reason: str) -> None:
        ctx = self._require_ctx()
        entry.status = ModuleStatus.DEGRADED
        entry.reason = reason
        self._log.warning("module_degraded", module=entry.module.name, reason=reason)
        await self._safe_shutdown(entry)
        await ctx.events.publish("module.degraded", {"module": entry.module.name, "reason": reason})
        if entry.retry is None or entry.retry.done():
            entry.retry = asyncio.create_task(self._retry_loop(entry))

    async def _retry_loop(self, entry: _Entry) -> None:
        ctx = self._require_ctx()
        name = entry.module.name
        attempt = 0
        while True:
            await self._sleep(self._backoff.delay(attempt))
            attempt += 1
            try:
                await self._start(entry)
            except DependencyUnavailable as exc:
                entry.reason = str(exc)
                self._log.warning(
                    "module_retry_failed", module=name, attempt=attempt, reason=str(exc)
                )
                await self._safe_shutdown(entry)
                continue
            except Exception as exc:  # noqa: BLE001 - logged, module marked FAILED, event published
                entry.status = ModuleStatus.FAILED
                entry.reason = str(exc)
                self._log.error("module_retry_crashed", module=name, error=str(exc), exc_info=True)
                await self._safe_shutdown(entry)
                await ctx.events.publish("module.failed", {"module": name, "reason": str(exc)})
                return
            self._log.info("module_recovered", module=name, attempts=attempt)
            await ctx.events.publish("module.recovered", {"module": name})
            return

    async def _rollback(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            await asyncio.gather(self._health_task, return_exceptions=True)
            self._health_task = None
        retries = [e.retry for e in self._entries if e.retry is not None]
        for task in retries:
            task.cancel()
        await asyncio.gather(*retries, return_exceptions=True)
        for entry in reversed(self._entries):
            entry.retry = None
            if entry.attempted:
                await self._safe_shutdown(entry)
                entry.status = ModuleStatus.STOPPED

    async def _safe_shutdown(self, entry: _Entry) -> None:
        try:
            await entry.module.shutdown()
        except Exception:  # noqa: BLE001 - logged; one module's failed cleanup must not block others
            self._log.exception("module_shutdown_failed", module=entry.module.name)
