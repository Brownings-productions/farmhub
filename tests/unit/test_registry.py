import asyncio

import pytest
from support import StubModule, make_app_context, make_tool

from farmhub.core.context import AppContext
from farmhub.core.errors import ConfigError, DependencyUnavailable, ModuleLoadError
from farmhub.core.protocols import HealthState, Module, Tier
from farmhub.core.registry import BackoffPolicy, ModuleRegistry, ModuleStatus


class Recorder:
    """Records the sleeps the registry asks for, then yields control instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        await asyncio.sleep(0)


async def wait_for_event(ctx: AppContext, topic: str) -> None:
    done = asyncio.Event()

    async def handler(event: object) -> None:
        done.set()

    ctx.events.subscribe(topic, handler)
    await asyncio.wait_for(done.wait(), timeout=2)


def test_stub_module_satisfies_the_module_protocol() -> None:
    assert isinstance(StubModule("a"), Module)


async def test_empty_module_loads_and_stops() -> None:
    module = StubModule("empty")
    registry = ModuleRegistry([module])
    await registry.startup(make_app_context())
    assert registry.status() == {"empty": ModuleStatus.RUNNING}
    assert registry.available_tools() == []
    await registry.shutdown()
    assert module.events == ["startup", "shutdown"]
    assert registry.status() == {"empty": ModuleStatus.STOPPED}


async def test_start_in_order_and_stop_in_reverse() -> None:
    order: list[str] = []

    class Ordered(StubModule):
        async def startup(self, ctx: AppContext) -> None:
            order.append(f"start {self.name}")

        async def shutdown(self) -> None:
            order.append(f"stop {self.name}")

    registry = ModuleRegistry([Ordered("a"), Ordered("b"), Ordered("c")])
    await registry.startup(make_app_context())
    await registry.shutdown()
    assert order == ["start a", "start b", "start c", "stop c", "stop b", "stop a"]


def test_duplicate_module_names_are_rejected() -> None:
    with pytest.raises(ModuleLoadError, match="duplicate module"):
        ModuleRegistry([StubModule("a"), StubModule("a")])


async def test_config_error_fails_fast_and_rolls_back() -> None:
    first = StubModule("first")
    bad = StubModule("bad", start_errors=[ConfigError("bad tool file")])
    never = StubModule("never")
    registry = ModuleRegistry([first, bad, never])
    with pytest.raises(ModuleLoadError, match="bad"):
        await registry.startup(make_app_context())
    assert first.events == ["startup", "shutdown"]
    assert bad.events == ["startup", "shutdown"]
    assert never.events == []


async def test_duplicate_tool_names_fail_fast() -> None:
    a = StubModule("a", tools=[make_tool("same").spec])
    b = StubModule("b", tools=[make_tool("same").spec])
    registry = ModuleRegistry([a, b])
    with pytest.raises(ModuleLoadError, match="duplicate tool"):
        await registry.startup(make_app_context())
    assert a.events == ["startup", "shutdown"]


async def test_available_tools_never_include_forbidden_but_find_tool_does() -> None:
    read = make_tool("read", Tier.READ).spec
    forbidden = make_tool("heating_off", Tier.FORBIDDEN).spec
    registry = ModuleRegistry([StubModule("m", tools=[read, forbidden])])
    await registry.startup(make_app_context())
    assert [t.name for t in registry.available_tools()] == ["read"]
    found = registry.find_tool("heating_off")
    assert found is not None
    assert found.spec is forbidden
    assert registry.find_tool("nonexistent") is None
    await registry.shutdown()


async def test_dependency_unavailable_starts_degraded_and_withholds_tools() -> None:
    tool = make_tool("ha_read").spec
    flaky = StubModule("ha", tools=[tool], start_errors=[DependencyUnavailable("HA down")] * 50)
    healthy = StubModule("web", tools=[make_tool("weather").spec])
    registry = ModuleRegistry([flaky, healthy], sleep=Recorder())
    await registry.startup(make_app_context())
    assert registry.status()["ha"] is ModuleStatus.DEGRADED
    assert registry.status()["web"] is ModuleStatus.RUNNING
    assert [t.name for t in registry.available_tools()] == ["weather"]
    await registry.shutdown()


async def test_degraded_module_retries_with_backoff_and_recovers() -> None:
    tool = make_tool("ha_read").spec
    module = StubModule("ha", tools=[tool], start_errors=[DependencyUnavailable("down")] * 3)
    sleeper = Recorder()
    registry = ModuleRegistry([module], sleep=sleeper)
    ctx = make_app_context()
    recovered = asyncio.create_task(wait_for_event(ctx, "module.recovered"))
    await asyncio.sleep(0)
    await registry.startup(ctx)
    await recovered
    assert sleeper.delays == [1.0, 2.0, 4.0]
    assert registry.status()["ha"] is ModuleStatus.RUNNING
    assert [t.name for t in registry.available_tools()] == ["ha_read"]
    await registry.shutdown()


def test_backoff_is_exponential_and_capped() -> None:
    policy = BackoffPolicy(base_s=1.0, factor=2.0, max_s=10.0)
    assert [policy.delay(n) for n in range(6)] == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]


async def test_unexpected_error_during_retry_marks_module_failed_and_stops_retrying() -> None:
    module = StubModule("ha", start_errors=[DependencyUnavailable("down"), RuntimeError("bug")])
    sleeper = Recorder()
    registry = ModuleRegistry([module], sleep=sleeper)
    ctx = make_app_context()
    failed = asyncio.create_task(wait_for_event(ctx, "module.failed"))
    await asyncio.sleep(0)
    await registry.startup(ctx)
    await failed
    assert registry.status()["ha"] is ModuleStatus.FAILED
    assert sleeper.delays == [1.0]
    assert registry.available_tools() == []
    await registry.shutdown()


async def test_mark_unhealthy_withholds_tools_then_recovers() -> None:
    module = StubModule("ha", tools=[make_tool("ha_read").spec])
    sleeper = Recorder()
    registry = ModuleRegistry([module], sleep=sleeper)
    ctx = make_app_context()
    await registry.startup(ctx)
    assert len(registry.available_tools()) == 1
    recovered = asyncio.create_task(wait_for_event(ctx, "module.recovered"))
    await asyncio.sleep(0)
    await registry.mark_unhealthy("ha", "websocket dropped")
    assert registry.available_tools() == []
    found = registry.find_tool("ha_read")
    assert found is not None
    assert found.available is False
    await recovered
    assert [t.name for t in registry.available_tools()] == ["ha_read"]
    await registry.shutdown()


async def test_shutdown_cancels_pending_retries() -> None:
    module = StubModule("ha", start_errors=[DependencyUnavailable("down")] * 1000)
    registry = ModuleRegistry([module], sleep=Recorder())
    await registry.startup(make_app_context())
    await registry.shutdown()
    events_after = list(module.events)
    await asyncio.sleep(0.01)
    assert module.events == events_after
    assert registry.status()["ha"] is ModuleStatus.STOPPED


async def test_health_aggregates_module_states() -> None:
    down = StubModule("ha", start_errors=[DependencyUnavailable("HA unreachable")] * 100)
    up = StubModule("web")
    registry = ModuleRegistry([down, up], sleep=Recorder())
    await registry.startup(make_app_context())
    report = await registry.health()
    assert report["web"].status is HealthState.HEALTHY
    assert report["ha"].status is HealthState.DEGRADED
    assert "HA unreachable" in report["ha"].detail
    await registry.shutdown()
