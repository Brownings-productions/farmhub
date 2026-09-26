from support import StubModule, make_tool

from farmhub.app import build_app
from farmhub.core.config import Settings
from farmhub.core.protocols import CallOrigin, Decision, ToolCall
from farmhub.core.registry import ModuleStatus
from farmhub.modules.llm import OpenAICompatBackend

ORIGIN = CallOrigin("kitchen", "s", frozenset({"kitchen"}))


async def test_empty_app_starts_and_stops() -> None:
    app = build_app(Settings(), [])
    await app.registry.startup(app.ctx)
    assert app.registry.status() == {}
    await app.registry.shutdown()


def test_the_default_module_list_is_the_one_in_the_composition_root() -> None:
    """SPEC §6: an explicit list, no entry-point scanning.

    Built but not started: starting it would reach for a real LLM backend, and no test
    may depend on one being there.
    """
    app = build_app(Settings())
    assert list(app.registry.status()) == ["llm"]


def test_the_shared_llm_backend_is_on_the_context() -> None:
    """The orchestrator issues completions without importing modules.llm."""
    app = build_app(Settings())
    assert isinstance(app.ctx.llm, OpenAICompatBackend)


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


def test_the_composition_root_hands_the_profiles_switches_to_the_backend() -> None:
    """The profile decides what every request carries (docs/DECISIONS.md 2026-09-23).

    Checked here rather than only in the client, because this wiring is the part that
    would silently go missing: the client would keep working and answer with reasoning.
    """
    settings = Settings(
        llm={"profile": "primary"},
        profiles={
            "primary": {
                "repo_id": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "revision": "1355db6a052410cfd62085d94b58866fd0f2c3c5",
                "quantization": "modelopt_fp4",
                "kv_cache_dtype": "auto",
                "gpu_memory_utilization": 0.82,
                "max_model_len": 32768,
                "weights_gib": 19.55,
                "kv_cache_gib": 4.9,
                "aux_reserve_gib": 5.0,
                "card_total_gib": 31.84,
                "measured_by": "evals/2026-09-23T16-13-25Z",
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0.0,
                "server_args": ["--block-size", "128", "--language-model-only"],
            }
        },
    )
    app = build_app(settings)
    assert app.backend.chat_template_kwargs == {"enable_thinking": False}
    assert app.backend.temperature == 0.0


def test_a_backend_built_without_a_profile_carries_no_switches() -> None:
    backend = build_app(Settings()).backend
    assert backend.chat_template_kwargs == {}
    # No profile means no claim about sampling, so the backend's default stands.
    assert backend.temperature is None
