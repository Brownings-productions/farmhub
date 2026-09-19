"""The gateway is the single enforcement point (docs/DECISIONS.md; SPEC §3.2, §3.7).

M0 skeleton: T0 tools run; unknown tools, T3 tools and every T1+ tool are denied.
Every call, denials included, writes an intent row and exactly one outcome row.
"""

import pytest
from structlog.testing import capture_logs
from support import (
    FailingAuditSink,
    MemoryAuditSink,
    StubModule,
    ToolProbe,
    make_app_context,
    make_tool,
)

from farmhub.core.audit import UnconfiguredAuditSink
from farmhub.core.config import Settings
from farmhub.core.gateway import MAX_AUDIT_TOOL_NAME_CHARS, Gateway
from farmhub.core.protocols import (
    AuditPhase,
    AuditSink,
    CallOrigin,
    Decision,
    Tier,
    ToolCall,
    ToolContext,
    ToolResult,
)
from farmhub.core.registry import ModuleRegistry

ORIGIN = CallOrigin(satellite="kitchen", session="s-1", scope=frozenset({"kitchen"}))


async def build(
    *tools: ToolProbe, sink: AuditSink, settings: Settings | None = None
) -> tuple[Gateway, ModuleRegistry]:
    ctx = make_app_context(audit=sink, settings=settings)
    registry = ModuleRegistry([StubModule("m", tools=[t.spec for t in tools])])
    await registry.startup(ctx)
    return Gateway(ctx, registry), registry


def assert_intent_then_one_outcome(
    sink: MemoryAuditSink, call: ToolCall, decision: Decision
) -> None:
    rows = [r for r in sink.rows if r.call_id == call.call_id]
    assert [r.phase for r in rows] == [AuditPhase.INTENT, AuditPhase.OUTCOME]
    assert rows[0].decision is None
    assert rows[1].decision is decision
    assert rows[1].duration_ms is not None


# --- denials are audited ---------------------------------------------------------------


async def test_unknown_tool_is_denied_and_audited_with_null_tier() -> None:
    sink = MemoryAuditSink()
    gateway, _ = await build(make_tool("known"), sink=sink)
    call = ToolCall("does_not_exist", {"x": 1})
    outcome = await gateway.invoke(call, ORIGIN)
    assert outcome.decision is Decision.DENIED
    assert outcome.result.ok is False
    assert_intent_then_one_outcome(sink, call, Decision.DENIED)
    intent = sink.rows[0]
    assert intent.tool == "does_not_exist"
    assert intent.tier is None
    assert intent.satellite == "kitchen"
    assert intent.session == "s-1"


async def test_unknown_tool_name_is_recorded_as_given_but_truncated() -> None:
    sink = MemoryAuditSink()
    gateway, _ = await build(sink=sink)
    await gateway.invoke(ToolCall("x" * 100_000, {}), ORIGIN)
    assert {r.tool for r in sink.rows} == {"x" * MAX_AUDIT_TOOL_NAME_CHARS}


async def test_t3_tool_is_denied_and_audited_and_never_runs() -> None:
    sink = MemoryAuditSink()
    forbidden = make_tool("heating_off", Tier.FORBIDDEN)
    gateway, _ = await build(forbidden, sink=sink)
    call = ToolCall("heating_off", {})
    outcome = await gateway.invoke(call, ORIGIN)
    assert outcome.decision is Decision.DENIED
    assert forbidden.calls == []
    assert_intent_then_one_outcome(sink, call, Decision.DENIED)
    assert sink.rows[0].tier is Tier.FORBIDDEN


@pytest.mark.parametrize("tier", [Tier.COMFORT, Tier.CONFIRMED])
async def test_t1_and_above_are_denied_and_audited_until_policy_exists(tier: Tier) -> None:
    sink = MemoryAuditSink()
    tool = make_tool("lights_on", tier)
    gateway, _ = await build(tool, sink=sink)
    call = ToolCall("lights_on", {})
    outcome = await gateway.invoke(call, ORIGIN)
    assert outcome.decision is Decision.DENIED
    assert tool.calls == []
    assert_intent_then_one_outcome(sink, call, Decision.DENIED)
    assert sink.rows[0].tier is tier


async def test_tool_of_a_degraded_module_is_denied_and_audited() -> None:
    sink = MemoryAuditSink()
    tool = make_tool("ha_read")
    gateway, registry = await build(tool, sink=sink)
    await registry.mark_unhealthy("m", "dependency down")
    call = ToolCall("ha_read", {})
    outcome = await gateway.invoke(call, ORIGIN)
    assert outcome.decision is Decision.DENIED
    assert tool.calls == []
    assert_intent_then_one_outcome(sink, call, Decision.DENIED)
    await registry.shutdown()


# --- T0 executes, with exactly one outcome row ---------------------------------------


async def test_t0_tool_runs_with_a_sealed_context_and_is_audited() -> None:
    sink = MemoryAuditSink()
    seen: list[ToolContext] = []
    tool = make_tool(
        "read",
        result=ToolResult(ok=True, data={"v": 7}, untrusted=True),
        on_call=lambda _call, ctx: seen.append(ctx),
    )
    gateway, _ = await build(tool, sink=sink, settings=Settings(dry_run=False))
    call = ToolCall("read", {"a": 1})
    outcome = await gateway.invoke(call, ORIGIN, context_tainted=True)
    assert outcome.decision is Decision.EXECUTED
    assert outcome.result.data == {"v": 7}
    assert outcome.result.untrusted is True
    assert len(tool.calls) == 1
    ctx = seen[0]
    assert (ctx.satellite, ctx.session, ctx.scope) == ("kitchen", "s-1", frozenset({"kitchen"}))
    assert ctx.context_tainted is True
    assert ctx.dry_run is False
    assert_intent_then_one_outcome(sink, call, Decision.EXECUTED)
    intent, out = sink.rows
    assert intent.arguments == {"a": 1}
    assert intent.dry_run is False
    assert out.result is not None
    assert out.result["ok"] is True


async def test_dry_run_flag_is_carried_into_audit_rows() -> None:
    sink = MemoryAuditSink()
    gateway, _ = await build(make_tool("read"), sink=sink)
    await gateway.invoke(ToolCall("read", {}), ORIGIN)
    assert [r.dry_run for r in sink.rows] == [True, True]


async def test_handler_exception_is_an_executed_outcome_with_error() -> None:
    sink = MemoryAuditSink()
    tool = make_tool("read", raises=RuntimeError("secret internals"))
    gateway, _ = await build(tool, sink=sink)
    call = ToolCall("read", {})
    outcome = await gateway.invoke(call, ORIGIN)
    assert outcome.decision is Decision.EXECUTED
    assert outcome.result.ok is False
    assert "secret internals" not in (outcome.result.error or "")
    assert_intent_then_one_outcome(sink, call, Decision.EXECUTED)


async def test_handler_timeout_is_an_executed_outcome_with_error() -> None:
    sink = MemoryAuditSink()
    tool = make_tool("slow", timeout_s=0.01, delay_s=1.0)
    gateway, _ = await build(tool, sink=sink)
    call = ToolCall("slow", {})
    outcome = await gateway.invoke(call, ORIGIN)
    assert outcome.decision is Decision.EXECUTED
    assert outcome.result.ok is False
    assert outcome.result.error == "timeout"
    assert_intent_then_one_outcome(sink, call, Decision.EXECUTED)


async def test_oversized_arguments_are_bounded_in_the_audit_row() -> None:
    sink = MemoryAuditSink()
    gateway, _ = await build(make_tool("read"), sink=sink)
    await gateway.invoke(ToolCall("read", {"blob": "y" * 50_000}), ORIGIN)
    args = sink.rows[0].arguments
    assert args is not None
    assert args.get("_truncated") is True


# --- audit failure means deny -------------------------------------------------------


@pytest.mark.parametrize("name", ["read", "heating_off", "lights_on", "unknown"])
async def test_intent_write_failure_denies_everything_and_nothing_runs(name: str) -> None:
    sink = FailingAuditSink()
    tools = [
        make_tool("read"),
        make_tool("heating_off", Tier.FORBIDDEN),
        make_tool("lights_on", Tier.COMFORT),
    ]
    gateway, _ = await build(*tools, sink=sink)
    outcome = await gateway.invoke(ToolCall(name, {}), ORIGIN)
    assert outcome.decision is Decision.DENIED
    assert all(t.calls == [] for t in tools)
    assert sink.rows == []


async def test_no_audit_sink_configured_means_no_tool_runs() -> None:
    tool = make_tool("read")
    gateway, _ = await build(tool, sink=UnconfiguredAuditSink())
    outcome = await gateway.invoke(ToolCall("read", {}), ORIGIN)
    assert outcome.decision is Decision.DENIED
    assert tool.calls == []


async def test_outcome_write_failure_is_logged_critical_and_the_real_result_is_returned() -> None:
    # The handler already ran, so hiding its result would misreport what happened.
    tool = make_tool("read", result=ToolResult(ok=True, data={"v": 1}))
    gateway, _ = await build(tool, sink=FailingAuditSink(ok_writes=1))
    with capture_logs() as logs:
        outcome = await gateway.invoke(ToolCall("read", {}), ORIGIN)
    assert outcome.decision is Decision.EXECUTED
    assert outcome.result.data == {"v": 1}
    assert any(
        e["event"] == "audit_outcome_write_failed" and e["log_level"] == "critical" for e in logs
    )
