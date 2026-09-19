from support import StubModule, make_tool

from farmhub.app import build_app
from farmhub.core.config import Settings
from farmhub.core.protocols import CallOrigin, Decision, ToolCall
from farmhub.core.registry import ModuleStatus

ORIGIN = CallOrigin("kitchen", "s", frozenset({"kitchen"}))


async def test_empty_app_starts_and_stops() -> None:
    app = build_app(Settings())
    await app.registry.startup(app.ctx)
    assert app.registry.status() == {}
    await app.registry.shutdown()


async def test_explicit_module_list_is_used() -> None:
    app = build_app(Settings(), [lambda: StubModule("only")])
    await app.registry.startup(app.ctx)
    assert app.registry.status() == {"only": ModuleStatus.RUNNING}
    await app.registry.shutdown()


async def test_default_app_denies_every_call_because_no_audit_sink_exists() -> None:
    tool = make_tool("read")
    app = build_app(Settings(), [lambda: StubModule("m", tools=[tool.spec])])
    await app.registry.startup(app.ctx)
    outcome = await app.gateway.invoke(ToolCall("read", {}), ORIGIN)
    assert outcome.decision is Decision.DENIED
    assert tool.calls == []
    await app.registry.shutdown()
