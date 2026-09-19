"""The single enforcement point for every tool call (docs/DECISIONS.md; SPEC §3.2, §3.7).

Modules supply handlers and clients only. Nothing else may run a handler: handlers are
sealed (see ``protocols.ToolSpec``) and only this module holds the seal.

M0 SKELETON. It fails closed. T0 tools run; unknown tools, T3 tools, tools of degraded
modules and every T1+ tool are denied. Tier/scope policy, rate limits, taint tracking,
dry-run and PendingAction handling arrive at M5-M8, each behind its safety tests.

Every call writes an INTENT row, then exactly one OUTCOME row, linked by ``call_id``.
Denials are audited like executions. If the intent row cannot be written the call is
denied (§3.7): nothing runs without an audit trail.
"""

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from farmhub.core.context import AppContext
from farmhub.core.errors import AuditWriteError
from farmhub.core.protocols import (
    GATEWAY_SEAL,
    AuditEntry,
    AuditPhase,
    CallOrigin,
    Decision,
    Tier,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from farmhub.core.registry import ModuleRegistry, RegisteredTool

MAX_AUDIT_TOOL_NAME_CHARS = 128
MAX_AUDIT_PAYLOAD_CHARS = 4096

DENY_UNKNOWN = "unknown tool"
DENY_FORBIDDEN = "tier T3 is never model-callable"
DENY_UNAVAILABLE = "tool unavailable (module not running)"
DENY_NO_POLICY = "T1+ tools are denied until the tier policy exists (M6)"
DENY_AUDIT = "audit unavailable"


@dataclass(frozen=True)
class ToolOutcome:
    """What the gateway returns. ``decision`` distinguishes a denial from an executed call."""

    call_id: UUID
    decision: Decision
    result: ToolResult


class Gateway:
    """Runs tool calls, enforcing policy and auditing every one."""

    def __init__(self, ctx: AppContext, registry: ModuleRegistry) -> None:
        self._ctx = ctx
        self._registry = registry
        self._log = ctx.log.bind(component="gateway")

    async def invoke(
        self, call: ToolCall, origin: CallOrigin, *, context_tainted: bool = False
    ) -> ToolOutcome:
        """Run ``call`` on behalf of ``origin``, or deny it. Always audited."""
        started = time.perf_counter()
        found = self._registry.find_tool(call.name)
        tier = found.spec.tier if found is not None else None
        tool_name = call.name[:MAX_AUDIT_TOOL_NAME_CHARS]
        dry_run = self._ctx.settings.dry_run

        def row(phase: AuditPhase, **extra: Any) -> AuditEntry:
            return AuditEntry(
                timestamp=datetime.now(UTC),
                call_id=call.call_id,
                phase=phase,
                tool=tool_name,
                tier=tier,
                dry_run=dry_run,
                satellite=origin.satellite,
                session=origin.session,
                **extra,
            )

        try:
            await self._ctx.audit.write(row(AuditPhase.INTENT, arguments=_bounded(call.arguments)))
        except AuditWriteError as exc:
            self._log.error("audit_intent_write_failed", tool=tool_name, error=str(exc))
            return ToolOutcome(call.call_id, Decision.DENIED, _denied(DENY_AUDIT))

        denial = self._denial_reason(found)
        if denial is None and found is not None:
            result = await self._run(call, found.spec, origin, context_tainted)
            decision = Decision.EXECUTED
        else:
            reason = denial or DENY_UNKNOWN
            self._log.warning("tool_call_denied", tool=tool_name, reason=reason)
            result = _denied(reason)
            decision = Decision.DENIED

        outcome = row(
            AuditPhase.OUTCOME,
            decision=decision,
            result=_bounded(
                {
                    "ok": result.ok,
                    "error": result.error,
                    "untrusted": result.untrusted,
                    "data": result.data,
                }
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        try:
            await self._ctx.audit.write(outcome)
        except AuditWriteError as exc:
            # The call already happened (or was already denied), so the real result is
            # still returned rather than misreporting it. Log at critical so the row can
            # be reconstructed and the sink fixed.
            self._log.critical(
                "audit_outcome_write_failed",
                call_id=str(call.call_id),
                tool=tool_name,
                decision=decision.value,
                error=str(exc),
            )
        return ToolOutcome(call.call_id, decision, result)

    @staticmethod
    def _denial_reason(found: RegisteredTool | None) -> str | None:
        """First applicable denial, or None to run. Order: unknown, T3, unavailable, T1+."""
        if found is None:
            return DENY_UNKNOWN
        if found.spec.tier is Tier.FORBIDDEN:
            return DENY_FORBIDDEN
        if not found.available:
            return DENY_UNAVAILABLE
        if found.spec.tier > Tier.READ:
            return DENY_NO_POLICY
        return None

    async def _run(
        self, call: ToolCall, spec: ToolSpec, origin: CallOrigin, context_tainted: bool
    ) -> ToolResult:
        tool_ctx = ToolContext(
            satellite=origin.satellite,
            session=origin.session,
            scope=origin.scope,
            dry_run=self._ctx.settings.dry_run,
            context_tainted=context_tainted,
            call_id=call.call_id,
            _seal=GATEWAY_SEAL,
        )
        try:
            async with asyncio.timeout(spec.timeout_s):
                return await spec.handler(call, tool_ctx)
        except TimeoutError:
            self._log.warning("tool_timeout", tool=spec.name, timeout_s=spec.timeout_s)
            return ToolResult(ok=False, error="timeout")
        except Exception as exc:  # noqa: BLE001 - logged; the model gets a generic error
            self._log.exception("tool_handler_failed", tool=spec.name)
            return ToolResult(ok=False, error=f"handler error ({type(exc).__name__})")


def _denied(reason: str) -> ToolResult:
    return ToolResult(ok=False, error=f"denied: {reason}")


def _bounded(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Cap what an audit row stores so arbitrary model or tool output cannot bloat it."""
    try:
        size = len(json.dumps(payload, default=str, sort_keys=True))
    except (TypeError, ValueError):
        return {"_unserializable": True}
    if size <= MAX_AUDIT_PAYLOAD_CHARS:
        return dict(payload)
    return {"_truncated": True, "_chars": size}
